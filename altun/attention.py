"""Causal self-attention (GQA + RoPE + PyTorch SDPA). T4 uyumlu: özel CUDA kernel zorlanmaz."""
from __future__ import annotations

from typing import Optional, Tuple

import torch
import torch.nn as nn
import torch.nn.functional as F

from .config import ModelConfig
from .layers import apply_rope

KV = Tuple[torch.Tensor, torch.Tensor]


def repeat_kv(x: torch.Tensor, n_rep: int) -> torch.Tensor:
    if n_rep == 1:
        return x
    b, h, t, d = x.shape
    return x[:, :, None, :, :].expand(b, h, n_rep, t, d).reshape(b, h * n_rep, t, d)


class CausalSelfAttention(nn.Module):
    def __init__(self, cfg: ModelConfig):
        super().__init__()
        self.n_heads, self.n_kv, self.hd = cfg.n_heads, cfg.n_kv_heads, cfg.head_dim
        self.n_rep = cfg.n_heads // cfg.n_kv_heads
        self.q_proj = nn.Linear(cfg.d_model, cfg.n_heads * self.hd, bias=False)
        self.k_proj = nn.Linear(cfg.d_model, cfg.n_kv_heads * self.hd, bias=False)
        self.v_proj = nn.Linear(cfg.d_model, cfg.n_kv_heads * self.hd, bias=False)
        self.o_proj = nn.Linear(cfg.n_heads * self.hd, cfg.d_model, bias=False)

    def forward(self, x: torch.Tensor, cos: torch.Tensor, sin: torch.Tensor,
                past_kv: Optional[KV] = None, use_cache: bool = False):
        b, t, _ = x.shape
        q = self.q_proj(x).view(b, t, self.n_heads, self.hd).transpose(1, 2)
        k = self.k_proj(x).view(b, t, self.n_kv, self.hd).transpose(1, 2)
        v = self.v_proj(x).view(b, t, self.n_kv, self.hd).transpose(1, 2)
        q, k = apply_rope(q, cos, sin), apply_rope(k, cos, sin)

        if past_kv is not None:
            k = torch.cat([past_kv[0], k], dim=2)
            v = torch.cat([past_kv[1], v], dim=2)
        present = (k, v) if use_cache else None

        kf, vf = repeat_kv(k, self.n_rep), repeat_kv(v, self.n_rep)
        s = kf.shape[2]
        if past_kv is None:
            out = F.scaled_dot_product_attention(q, kf, vf, is_causal=True)
        elif t == 1:
            out = F.scaled_dot_product_attention(q, kf, vf)  # tek sorgu tüm geçmişi görebilir
        else:
            i = torch.arange(t, device=x.device)[:, None] + (s - t)
            j = torch.arange(s, device=x.device)[None, :]
            out = F.scaled_dot_product_attention(q, kf, vf, attn_mask=(j <= i))
        out = out.transpose(1, 2).contiguous().view(b, t, self.n_heads * self.hd)
        return self.o_proj(out), present
