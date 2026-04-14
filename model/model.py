from transformers import PretrainedConfig
import math

from transformers.activations import ACT2FN
from transformers import PretrainedModel, GenerationMixin
from transformers.modeling_outputs import CausalLMOutputWithPast


class ItanMindConfig(PretrainedConfig):
    model_type = "Itanmind"
    def __init__(self, hidden_size=768, num_hidden_layers=8, use_moe=False, **kwargs):
        super().__init__(**kwargs)
        self.hidden_size = hidden_size
        self.num_hidden_layers = num_hidden_layers
        self.use_moe = use_moe
        self.dropout = kwargs.get("dropout", 0.0)
        self.vocab_size = kwargs.get("vocab_size", 6400)
        self.bos_token_id = kwargs.get("bos_token_id", 1)
        self.eos_token_id = kwargs.get("eos_token_id", 2)
        self.flash_attn = kwargs.get("flash_attn", True)
        self.num_attention_heads = kwargs.get("num_attention_heads", 8)
        self.num_key_value_heads = kwargs.get("num_key_value_heads", 4)
        self.head_dim = kwargs.get("head_dim", self.hidden_size // self.num_attention_heads)
        self.hidden_act = kwargs.get("hidden_act", 'silu')
        self.intermediate_size = kwargs.get("intermediate_size", math.ceil(hidden_size * math.pi / 64) * 64)
        self.max_position_embeddings = kwargs.get("max_position_embeddings", 32768)
        self.rms_norm_eps = kwargs.get("rms_norm_eps", 1e-6)
        self.rope_theta = kwargs.get("rope_theta", 1e6)
        self.inference_rope_scaling = kwargs.get("inference_rope_scaling", False)
        self.rope_scaling = {
            "beta_fast": 32,
            "beta_slow": 1,
            "factor": 16,
            "original_max_position_embeddings": 2048,
            "attention_factor": 1.0,
            "type": "yarn"
        } if self.inference_rope_scaling else None
        ### MoE specific configs (ignored if use_moe = False)
        self.num_experts = kwargs.get("num_experts", 4)
        self.num_experts_per_tok = kwargs.get("num_experts_per_tok", 1)
        self.moe_intermediate_size = kwargs.get("moe_intermediate_size", self.intermediate_size)
        self.norm_topk_prob = kwargs.get("norm_topk_prob", True)
        self.router_aux_loss_coef = kwargs.get("router_aux_loss_coef", 5e-4)

import torch
import torch.nn as nn
import torch.nn.functional as F
from typing import Optional, Tuple

class RMSNorm(nn.Module):
    def __init__(self, dim: int, eps: float = 1e-5):
        super().__init__()
        self.dim = dim
        self.eps = eps
        self.weight = nn.Parameter(torch.ones(dim))

    def _norm(self, x: torch.Tensor) -> torch.Tensor:
        return torch.rsqrt(x.pow(2).mean(-1, keepdim=True) + self.eps)
    
    def forward(self, x: torch.Tensor) -> torch.Tensor:
        return self.weight * self._norm(x) * x

# init RoPE freqs
def precompute_freqs_cis(dim: int, end: int = 32*1024, rope_base: float = 10000.0, rope_scaling: Optional[dict] = None):
    freqs, attn_factors = (1.0 / (rope_base ** (torch.arange(0, dim, 2)[: (dim // 2)].float() / dim))), 1.0
    if rope_scaling is not None:
        orig_max, factor, beta_fast, beta_slow = (
            rope_scaling["original_max_position_embeddings"],
            rope_scaling["factor"],
            rope_scaling["beta_fast"],
            rope_scaling["beta_slow"]
        )

        if end > orig_max:
            # b < low 为高频（转圈数 > beta_fast）
            # b > high 为低频（转圈数 < beta_slow）
            low  = max(math.floor((dim * math.log(orig_max / (beta_fast * 2 * math.pi))) / (2 * math.log(rope_base))), 0)
            high = min(math.ceil( (dim * math.log(orig_max / (beta_slow * 2 * math.pi))) / (2 * math.log(rope_base))), dim // 2 - 1)

            ramp = torch.arange(dim // 2, dtype=torch.float32)  # [0, 1, 2, ..., dim//2 - 1]
            ramp = (ramp - low) / (high - low + 1e-3)
            ramp = torch.clamp(ramp, 0.0, 1.0)

            # 当 ramp=0 时（高频）：系数为 1，保持原频率不变。
            # 当 ramp=1 时（低频）：系数为 1/factor，即对频率进行线性插值缩放。
            # ramp在0-1之间时：平滑过渡。
            freqs = freqs * (1 - ramp + ramp / factor)
            attn_factors = 0.1 * math.log(factor) + 1.0    # mscale

    t = torch.arange(end, device=freqs.device)
    freqs = torch.outer(t, freqs).float()
    emb = torch.cat([freqs, freqs], dim=-1)
    cos = emb.cos() * attn_factors
    sin = emb.sin() * attn_factors
    return cos, sin

# RoPE (Rotary Position Embedding)
def apply_rotary_pos_emb(q: torch.Tensor, k: torch.Tensor, cos:torch.Tensor, sin:torch.Tensor, unsqueeze_dim: int = 1):

    def rotate_half(x):
        return torch.cat([-x[..., x.shape[-1] // 2 : ], x[..., : x.shape[-1] // 2]], dim=-1)

    cos = cos.unsqueeze(unsqueeze_dim)
    sin = sin.unsqueeze(unsqueeze_dim)
    q_embed = (q * cos) + (rotate_half(q) * sin)
    k_embed = (k * cos) + (rotate_half(k) * sin)

    return q_embed, k_embed

def repeat_kv(x: torch.Tensor, n_rep: int) -> torch.Tensor:
    bs, slen, num_kv_heads, head_dim = x.shape
    if n_rep == 1:
        return x
    x = x[..., None, :].expand(bs, slen, num_kv_heads, n_rep, head_dim)
    return x.reshape(bs, slen, num_kv_heads * n_rep, head_dim)

class Attention(nn.Module):
    # GQA (Grouped Query Attention) 把 Q 头分成若干组，每组共享一个 K 头和 V 头。
    def __init__(self, args: ItanMindConfig):
        super().__init__()
        self.n_local_heads = args.num_attention_heads
        self.num_kv_heads = args.num_key_value_heads if args.num_key_value_heads is not None else args.num_attention_heads
        assert self.n_local_heads % self.num_kv_heads == 0, "num_attention_heads 必须能被 num_key_value_heads 整除"

        self.n_rep = self.num_kv_heads // self.n_local_heads
        self.head_dim = args.hidden_size // args.num_attention_heads

        self.q_proj = nn.Linear(args.hidden_size, args.num_attention_heads * self.head_dim, bias=False)
        self.k_proj = nn.Linear(args.hidden_size, self.num_kv_heads * self.head_dim, bias=False)
        self.v_proj = nn.Linear(args.hidden_size, self.num_kv_heads * self.head_dim, bias=False)
        self.o_proj = nn.Linear(args.num_attention_heads * self.head_dim, args.hidden_size, bias=False)

        self.attn_dropout = nn.Dropout(args.dropout)
        self.resid_dropout = nn.Dropout(args.dropout)
        self.dropout = args.dropout

        self.flash = hasattr(torch.nn.functional, 'scaled_dot_product_attention') and args.flash_attn

    def forward(self, x: torch.Tensor, position_embeddings:Tuple[torch.Tensor, torch.Tensor], 
                past_key_value: Optional[Tuple[torch.Tensor, torch.Tensor]] = None, 
                use_cache: bool = False,
                attention_mask: Optional[torch.Tensor] = None,) -> torch.Tensor:
        
        bs, seq_len, _ = x.shape
        xq = self.q_proj(x)
        xk = self.k_proj(x)
        xv = self.v_proj(x)

        xq = xq.view(bs, seq_len, self.n_local_heads, self.head_dim)
        xk = xk.view(bs, seq_len, self.num_kv_heads, self.head_dim)
        xv = xv.view(bs, seq_len, self.num_kv_heads, self.head_dim)

        #use rope on qk
        cos, sin = position_embeddings
        xq, xk = apply_rotary_pos_emb(xq, xk, cos, sin)

        if past_key_value is not None:
            xk = torch.cat([past_key_value[0], xk], dim=1)
            xv = torch.cat([past_key_value[1], xv], dim=1)
        past_kv = (xk, xv)

        xq, xk, xv = (
            xq.transpose(1, 2),
            repeat_kv(xk, self.n_rep).transpose(1, 2),
            repeat_kv(xv, self.n_rep).transpose(1, 2)
        )

        if self.flash and seq_len > 1:
            dropout_p = self.dropout if self.training else 0.0
            if attention_mask is None:
                output = F.scaled_dot_product_attention(
                    xq, xk, xv,
                    attn_mask=None,
                    dropout_p=dropout_p,
                    is_causal=True
                )

        else:
            kv_seq_len = xk.shape[-2]
            scores = (xq @ xk.transpose(-2, -1)) / math.sqrt(self.head_dim)
            if seq_len > 1:
                # 上三角（未来位置）填 -inf，对角线及以下保留
                causal_mask = torch.full((seq_len, kv_seq_len), float('-inf'), device=xq.device)
                causal_mask = causal_mask.triu(kv_seq_len - seq_len + 1)
                scores = scores + causal_mask
            if attention_mask is not None:
                scores = scores + attention_mask
            scores = F.softmax(scores.float(), dim=-1).type_as(xq)
            scores = self.attn_dropout(scores)
            output = (scores @ xv).transpose(1, 2).contiguous().view(bs, seq_len, -1)
            output = self.resid_dropout(self.o_proj(output))

        return output, past_kv if use_cache else None

class FeedForward(nn.Module):
    # 初始化
    # 升维
    # 降维
    # 门控
    # droout
    # 激活函数
    def __init__(self, args: ItanMindConfig):
        super().__init__()
        if args.intermediate_size is None:
            intermediate_size = int(args.hidden_size*8/3)
            args.intermediate_size = 64 * ((intermediate_size + 64 -1) // 64)

        self.up_proj = nn.Linear(args.hidden_size, args.intermediate_size, bias=False)
        self.down_proj = nn.Linear(args.intermediate_size, args.hidden_size, bias=False)
        self.gate_proj = nn.Linear(args.hidden_size, args.intermediate_size, bias=False)
        self.dropout = nn.Dropout(args.dropout)
        self.act_fn = ACT2FN[args.hidden_act]

    def forward(self, x):
        return self.dropout(self.down_proj(self.act_fn(self.gate_proj(x)) * self.up_proj(x)))

class ItanMindBlock(nn.Module):
    def __init__(self, layer_id: int, config: ItanMindConfig):
        self.num_attention_heads = config.num_attention_heads
        self.hidden_size = config.hidden_size
        self.head_dim = self.hidden_size // self.num_attention_heads
        self.self_attn = Attention(config)

        self.layer_id = layer_id
        self.input_layernorm = RMSNorm(config.hidden_size, eps=config.rms_norm_eps)
        self.post_attention_layernorm = RMSNorm(config.hidden_size, eps=config.rms_norm_eps)
        self.mlp = FeedForward(config)

    def forward(
        self,
        hidden_states,
        position_embeddings,
        past_key_value = None,
        use_cache = False,
        attention_mask = None
    ):
        residual = hidden_states
        hidden_states, present_key_value = self.self_attn(
            self.input_layernorm(hidden_states),
            position_embeddings,
            past_key_value,
            use_cache,
            attention_mask
        )

        hidden_states = hidden_states + residual
        hidden_states = hidden_states + self.mlp(self.post_attention_layernorm(hidden_states))

        return hidden_states, present_key_value

class ItanMindModel(nn.Module):
    def __init__(self, config: ItanMindConfig):
        super().__init__()
        self.vocab_size = config.vocab_size 
        self.num_hidden_layers = config.num_hidden_layers

        self.embed_tokens = nn.Embedding(config.vocab_size, config.hidden_size)
        self.dropout = nn.Dropout(config.dropout)

        self.layer = nn.ModuleList(
            [ItanMindBlock(i, config) for i in range(self.num_hidden_layers)]
        )

        self.norm = RMSNorm(config.hidden_size, eps=config.rms_norm_eps)

        # RoPE pre compute
        cos, sin = precompute_freqs_cis(
            config.head_dim,
            config.max_position_embeddings,
            config.rope_theta,
            config.rope_scaling,
        )
        self.register_buffer("cos", cos, persistent=False)
        self.register_buffer("sin", sin, persistent=False)

    def forward(
        self,
        input_ids: torch.Tensor,
        attention_mask: Optional[torch.Tensor] = None,
        past_key_values: Optional[list] = None,
        use_cache: bool = False,
    ):
        bs, seq_len = input_ids.shape
        hidden_states = self.dropout(self.embed_tokens(input_ids))

        start_pos = past_key_values[0][0].shape[1] if past_key_values[0] is not None else 0
        position_embeddings = (self.cos[start_pos:start_pos + seq_len], self.sin[start_pos:start_pos + seq_len])

        if past_key_values is None:
            past_key_values = [None] * self.num_hidden_layers

        present_key_values = []
        for i, layer in enumerate(self.layer):
            hidden_states, present_kv = layer(
                hidden_states,
                position_embeddings,
                past_key_values[i],
                use_cache,
                attention_mask,
            )
            present_key_values.append(present_kv)

        hidden_states = self.norm(hidden_states)
        return hidden_states, present_key_values

class ItanMind4CausalLM(PretrainedModel, GenerationMixin):
    config_class = ItanMindConfig

    def __init__(self, config: ItanMindConfig):
        self.config = config

        super().__init__(config)

        self.model = ItanMindModel(config)

        self.lm_head = nn.Linear(
            self.config.hidden_size, self.config.vocab_size, bias=False
        )

        self.model.embed_tokens.weight = self.lm_head.weight

    def forward(
        self,
        input_ids: Optional[torch.Tensor] = None,
        attention_mask: Optional[torch.Tensor] = None,
        past_key_values: Optional[list] = None,
        use_cache: bool = False,
        labels: Optional[torch.Tensor] = None,
    ):
        hidden_states, present_key_values = self.model(
            input_ids, attention_mask, past_key_values, use_cache
        )

        logits = self.lm_head(hidden_states)

        loss = None
        if labels is not None:
            # 向左移一位：用前 n-1 个 token 预测后 n-1 个 token
            shift_logits = logits[..., :-1, :].contiguous()
            shift_labels = labels[..., 1:].contiguous()
            loss = F.cross_entropy(
                shift_logits.view(-1, self.config.vocab_size),
                shift_labels.view(-1),
                ignore_index=-100,
            )

        return CausalLMOutputWithPast(
            loss=loss,
            logits=logits,
            past_key_values=present_key_values if use_cache else None,
        )
