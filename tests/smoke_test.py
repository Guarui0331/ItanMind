"""
CPU smoke test: 在不依赖 GPU 的前提下，把训练链路从头到尾打通一遍，
尽可能覆盖 model.py / dataset / trainer 可能出 bug 的地方。
跑法（项目根目录）:
    python tests/smoke_test.py
任何一项 FAIL 都说明上 GPU 前需要先修。
"""
import os
import sys
import json
import tempfile
import traceback

sys.path.append(os.path.abspath(os.path.join(os.path.dirname(__file__), '..')))

import torch
import torch.nn.functional as F
from torch.utils.data import DataLoader

from model.model import ItanMindConfig, ItanMind4CausalLM
from dataset.llm_dataset import PretrainDataset
from transformers import AutoTokenizer


PASS, FAIL = "\033[92m[PASS]\033[0m", "\033[91m[FAIL]\033[0m"
results = []


def run(name, fn):
    print(f"\n=== {name} ===")
    try:
        fn()
        print(f"{PASS} {name}")
        results.append((name, True, None))
    except Exception as e:
        traceback.print_exc()
        print(f"{FAIL} {name}: {e}")
        results.append((name, False, str(e)))


def make_small_config(flash_attn: bool):
    # 用极小规模，CPU 上秒级完成
    return ItanMindConfig(
        hidden_size=64,
        num_hidden_layers=2,
        num_attention_heads=4,
        num_key_value_heads=2,
        vocab_size=256,
        max_position_embeddings=128,
        flash_attn=flash_attn,
    )


# ---------- 1. 配置 / 模型实例化 ----------
def test_config_and_build():
    cfg = make_small_config(flash_attn=False)
    model = ItanMind4CausalLM(cfg)
    n_params = sum(p.numel() for p in model.parameters())
    assert n_params > 0, "模型参数为 0"
    # 权重绑定检查
    assert model.model.embed_tokens.weight.data_ptr() == model.lm_head.weight.data_ptr(), \
        "embed_tokens 和 lm_head 没有共享权重"
    print(f"  params={n_params/1e3:.1f}K  (tied embed/lm_head OK)")


# ---------- 2. 前向 + 反向（两种 attention 路径都测） ----------
def _fwd_bwd(flash_attn: bool):
    cfg = make_small_config(flash_attn=flash_attn)
    model = ItanMind4CausalLM(cfg)
    model.train()
    bs, seq = 2, 16
    x = torch.randint(0, cfg.vocab_size, (bs, seq))
    out = model(input_ids=x, labels=x)
    assert out.logits.shape == (bs, seq, cfg.vocab_size), f"logits shape = {out.logits.shape}"
    assert torch.isfinite(out.loss), f"loss is not finite: {out.loss}"
    out.loss.backward()
    # 至少一个参数拿到梯度
    got_grad = any(p.grad is not None and p.grad.abs().sum() > 0 for p in model.parameters())
    assert got_grad, "所有参数都没有梯度"
    print(f"  flash_attn={flash_attn}  loss={out.loss.item():.4f}  grad OK")


def test_forward_backward_noflash():
    _fwd_bwd(flash_attn=False)


def test_forward_backward_flash():
    _fwd_bwd(flash_attn=True)


# ---------- 3. 两条 attention 路径数值应一致 ----------
def test_flash_vs_manual_consistency():
    torch.manual_seed(0)
    cfg_a = make_small_config(flash_attn=False)
    cfg_b = make_small_config(flash_attn=True)
    m_a = ItanMind4CausalLM(cfg_a)
    m_b = ItanMind4CausalLM(cfg_b)
    m_b.load_state_dict(m_a.state_dict())
    m_a.eval(); m_b.eval()
    x = torch.randint(0, cfg_a.vocab_size, (2, 16))
    with torch.no_grad():
        la = m_a(input_ids=x).logits
        lb = m_b(input_ids=x).logits
    diff = (la - lb).abs().max().item()
    assert diff < 1e-4, f"flash 与 manual attention logits 差异过大: {diff}"
    print(f"  max |Δlogits| = {diff:.2e}")


# ---------- 4. KV cache 单步推理 ----------
def test_kv_cache_inference():
    cfg = make_small_config(flash_attn=False)
    model = ItanMind4CausalLM(cfg)
    model.eval()
    x = torch.randint(0, cfg.vocab_size, (1, 8))
    with torch.no_grad():
        # 一次性前向
        full = model(input_ids=x, use_cache=True)
        assert full.past_key_values is not None, "use_cache=True 时 past_key_values 为 None"
        # 增量前向 1 个 token
        new_tok = torch.randint(0, cfg.vocab_size, (1, 1))
        step = model(input_ids=new_tok, past_key_values=full.past_key_values, use_cache=True)
        assert step.logits.shape == (1, 1, cfg.vocab_size), f"增量 logits shape = {step.logits.shape}"
    print("  past_key_values 前向 & 增量解码 OK")


# ---------- 5. Dataset + DataLoader + 训练步 ----------
def _make_tiny_jsonl(path):
    samples = [{"text": f"hello world sample number {i} " * 3} for i in range(16)]
    with open(path, "w") as f:
        for s in samples:
            f.write(json.dumps(s, ensure_ascii=False) + "\n")


def test_dataset_and_train_step():
    tokenizer_path = os.path.abspath(os.path.join(os.path.dirname(__file__), '..', 'model'))
    tokenizer = AutoTokenizer.from_pretrained(tokenizer_path)

    with tempfile.TemporaryDirectory() as td:
        data_path = os.path.join(td, "tiny.jsonl")
        _make_tiny_jsonl(data_path)
        ds = PretrainDataset(data_path, tokenizer, max_length=32)
        sample = ds[0]
        assert isinstance(sample, tuple), f"__getitem__ 应返回 tuple，实际 {type(sample)}"
        print(f"  dataset 返回 {len(sample)} 个元素")
        loader = DataLoader(ds, batch_size=4, shuffle=False, num_workers=0)
        batch = next(iter(loader))
        # 关键检查：labels 里不应全是 0/1（那说明被 attention_mask 顶包了）
        if len(batch) >= 2:
            labels = batch[1]
            uniq = torch.unique(labels[labels != -100])
            assert uniq.numel() > 2, \
                f"labels 看起来像 attention_mask（唯一值={uniq.tolist()}），接口错位！"
            print(f"  labels 唯一值数={uniq.numel()}（健康）")

    # 真跑一步训练
    cfg = ItanMindConfig(
        hidden_size=64, num_hidden_layers=2,
        num_attention_heads=4, num_key_value_heads=2,
        vocab_size=tokenizer.vocab_size, max_position_embeddings=64,
        flash_attn=False,
    )
    model = ItanMind4CausalLM(cfg)
    model.train()
    opt = torch.optim.AdamW(model.parameters(), lr=1e-3)

    # 模拟 trainer 的 batch 解包
    if len(batch) == 3:
        input_ids, labels, attention_mask = batch
    else:
        input_ids, labels = batch
        attention_mask = None

    before = None
    losses = []
    for step in range(5):
        res = model(input_ids=input_ids, labels=labels, attention_mask=attention_mask)
        assert torch.isfinite(res.loss), f"step {step} loss 非有限: {res.loss}"
        losses.append(res.loss.item())
        opt.zero_grad()
        res.loss.backward()
        opt.step()
        if before is None:
            before = next(model.parameters()).detach().clone()
    after = next(model.parameters()).detach()
    assert not torch.equal(before, after), "训练 5 步后参数没变化，优化器没生效"
    print(f"  5-step losses: {[f'{l:.4f}' for l in losses]}")
    assert losses[-1] < losses[0] + 0.5, "loss 明显没有下降趋势（允许噪声，但不该飙升）"


# ---------- 6. DDP API 在单进程下不应误触发 ----------
def test_no_ddp_in_single_process():
    import torch.distributed as dist
    assert not dist.is_initialized(), "单进程下不应初始化 dist"
    print("  dist 未初始化（符合单卡/CPU 预期）")


if __name__ == "__main__":
    run("1. config & model build", test_config_and_build)
    run("2. forward+backward (manual attn)", test_forward_backward_noflash)
    run("3. forward+backward (flash attn)", test_forward_backward_flash)
    run("4. flash vs manual consistency", test_flash_vs_manual_consistency)
    run("5. KV cache inference", test_kv_cache_inference)
    run("6. dataset + 5-step training", test_dataset_and_train_step)
    run("7. no accidental DDP init", test_no_ddp_in_single_process)

    print("\n" + "=" * 50)
    passed = sum(1 for _, ok, _ in results if ok)
    print(f"Summary: {passed}/{len(results)} passed")
    for name, ok, err in results:
        tag = PASS if ok else FAIL
        print(f"  {tag} {name}" + (f"  -> {err}" if err else ""))
    sys.exit(0 if passed == len(results) else 1)
