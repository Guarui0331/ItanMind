import os
import sys

__package__ = "trainer"
sys.path.append(os.path.abspath(os.path.join(os.path.dirname(__file__), '..')))

import argparse
import time
import warnings
import torch
import torch.distributed as dist
from contextlib import nullcontext
from torch import optim, nn
from torch.nn.parallel import DistributedDataParallel
from torch.utils.data import DataLoader, DistributedSampler
from model.model import ItanMindConfig
from dataset.llm_dataset import PretrainDataset
from trainer.trainer_utils import get_lr, Logger, is_main_process, lm_checkpoint, init_distributed_mode, setup_seed, init_model, SkipBatchSampler

warnings.filterwarnings('ignore')



def train_epoch(epoch, loader, iters, start_step=0, wandb=None):
    start_time = time.time()
    last_step = start_step
    for step, (input_ids, labels) in enumerate(loader, start=start_step + 1):
        input_ids = input_ids.to(args.device)
        labels = labels.to(args.device)
        last_step = step
        lr = get_lr(epoch * iters + step, args.epochs * iters, args.learning_rate)
        for param_group in optimizer.param_groups:
            param_group['lr'] = lr

        with autocast_ctx:
            res = model(input_ids, labels=labels)
            aux_loss = getattr(res, 'aux_loss', None)
            loss = res.loss + aux_loss if aux_loss is not None else res.loss
            loss = loss / args.accumulation_steps

        scaler.scale(loss).backward()

        if step % args.accumulation_steps == 0:
            scaler.unscale_(optimizer)
            torch.nn.utils.clip_grad_norm_(model.parameters(), args.grad_clip)

            scaler.step(optimizer)
            scaler.update()

            optimizer.zero_grad(set_to_none=True)

        if step % args.log_interval == 0 or step == iters:
            spend_time = time.time() - start_time
            current_loss = loss.item() * args.accumulation_steps
            current_aux_loss = aux_loss.item() if aux_loss is not None else 0.0
            current_logits_loss = current_loss - current_aux_loss
            current_lr = optimizer.param_groups[-1]['lr']
            eta_min = spend_time / max(step - start_step, 1) * (iters - step) // 60
            Logger(f'Epoch:[{epoch + 1}/{args.epochs}]({step}/{iters}), loss: {current_loss:.4f}, logits_loss: {current_logits_loss:.4f}, aux_loss: {current_aux_loss:.4f}, lr: {current_lr:.8f}, epoch_time: {eta_min:.1f}min')
            if wandb: wandb.log({"loss": current_loss, "logits_loss": current_logits_loss, "aux_loss": current_aux_loss, "learning_rate": current_lr, "epoch_time": eta_min})

        if (step % args.save_interval == 0 or step == iters) and is_main_process():
            model.eval()
            lm_checkpoint(lm_config, weight=args.save_weight, model=model, optimizer=optimizer, scaler=scaler, epoch=epoch, step=step, wandb=wandb, save_dir=args.save_dir)
            model.train()

        del input_ids, labels, res, loss

    if last_step > start_step and last_step % args.accumulation_steps != 0:
        scaler.unscale_(optimizer)
        torch.nn.utils.clip_grad_norm_(model.parameters(), args.grad_clip)
        scaler.step(optimizer)
        scaler.update()
        optimizer.zero_grad(set_to_none=True)


if __name__ == '__main__':
    parser = argparse.ArgumentParser()
    parser.add_argument('--hidden_size', type=int, default=768)
    parser.add_argument('--num_hidden_layers', type=int, default=8)
    parser.add_argument('--use_moe', action='store_true')
    parser.add_argument('--epochs', type=int, default=1)
    parser.add_argument('--batch_size', type=int, default=8)
    parser.add_argument('--learning_rate', type=float, default=3e-4)
    parser.add_argument('--accumulation_steps', type=int, default=8)
    parser.add_argument('--grad_clip', type=float, default=1.0)
    parser.add_argument('--log_interval', type=int, default=100)
    parser.add_argument('--save_interval', type=int, default=1000)
    parser.add_argument('--save_dir', type=str, default='../out')
    parser.add_argument('--save_weight', type=str, default='pretrain')
    parser.add_argument('--data_path', type=str, default='../dataset/pretrain_data.jsonl')
    parser.add_argument('--tokenizer_path', type=str, default='../model')
    parser.add_argument('--max_seq_len', type=int, default=512)
    parser.add_argument('--num_workers', type=int, default=2)
    parser.add_argument('--use_wandb', action='store_true')
    parser.add_argument('--seed', type=int, default=42)
    parser.add_argument('--resume', action='store_true')
    args = parser.parse_args()

    # 分布式初始化
    local_rank = init_distributed_mode()
    args.device = f'cuda:{local_rank}' if torch.cuda.is_available() else 'cpu'

    setup_seed(args.seed)
    os.makedirs(args.save_dir, exist_ok=True)

    # 模型配置
    lm_config = ItanMindConfig(
        hidden_size=args.hidden_size,
        num_hidden_layers=args.num_hidden_layers,
        use_moe=args.use_moe,
    )

    # 模型 & tokenizer
    model, tokenizer = init_model(lm_config, from_weight='none' if not args.resume else args.save_weight,
                                  tokenizer_path=args.tokenizer_path, save_dir=args.save_dir, device=args.device)

    # 混合精度
    dtype = 'bfloat16' if torch.cuda.is_bf16_supported() else 'float16'
    autocast_ctx = torch.cuda.amp.autocast(dtype=torch.bfloat16 if dtype == 'bfloat16' else torch.float16)
    scaler = torch.cuda.amp.GradScaler(enabled=(dtype == 'float16'))

    # 优化器
    optimizer = optim.AdamW(model.parameters(), lr=args.learning_rate, weight_decay=0.1, betas=(0.9, 0.95))

    # 断点续训
    start_epoch, start_step = 0, 0
    if args.resume:
        ckp_data = lm_checkpoint(lm_config, weight=args.save_weight, save_dir=args.save_dir)
        if ckp_data:
            model.load_state_dict(ckp_data['model'], strict=False)
            optimizer.load_state_dict(ckp_data['optimizer'])
            if 'scaler' in ckp_data:
                scaler.load_state_dict(ckp_data['scaler'])
            start_epoch = ckp_data.get('epoch', 0)
            start_step = ckp_data.get('step', 0)

    # DDP 包装
    if dist.is_initialized():
        model = DistributedDataParallel(model, device_ids=[local_rank])

    # wandb
    wandb = None
    if args.use_wandb and is_main_process():
        import wandb as wb
        wandb = wb.init(project='ItanMind-pretrain', config=vars(args))

    # 数据集
    train_ds = PretrainDataset(args.data_path, tokenizer, max_length=args.max_seq_len)
    sampler = DistributedSampler(train_ds) if dist.is_initialized() else None
    train_loader = DataLoader(train_ds, batch_size=args.batch_size, sampler=sampler,
                              shuffle=(sampler is None), num_workers=args.num_workers, pin_memory=True)

    model.train()
    for epoch in range(start_epoch, args.epochs):
        if sampler:
            sampler.set_epoch(epoch)
        iters = len(train_loader)
        step_offset = start_step if epoch == start_epoch else 0
        train_epoch(epoch, train_loader, iters, start_step=step_offset, wandb=wandb)

    if wandb:
        wandb.finish()
