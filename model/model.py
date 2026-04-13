from transformers import PretrainedConfig
import math

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

