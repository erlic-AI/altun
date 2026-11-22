"""Temel katmanlar: RMSNorm, RoPE, SwiGLU."""
from __future__ import annotations

import torch
import torch.nn as nn
import torch.nn.functional as F


class RMSNorm(nn.Module):
    def __init__(self, dim: int, eps: float = 1e-5):
        super().__init__()
        self.eps = eps
        self.weight = nn.Parameter(torch.ones(dim))

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        # Hesap float32'de yapılır (FP16 taşmasını önler), çıktı giriş dtype'ına döner.
        xf = x.float()
        xf = xf * torch.rsqrt(xf.pow(2).mean(-1, keepdim=True) + self.eps)
        return (xf * self.weight.float()).to(x.dtype)


def precompute_rope(head_dim: int, max_len: int, theta: float = 10000.0):
    """cos/sin tabloları: [max_len, head_dim/2] (float32)."""
    inv_freq = 1.0 / (theta ** (torch.arange(0, head_dim, 2, dtype=torch.float32) / head_dim))
    t = torch.arange(max_len, dtype=torch.float32)
    freqs = torch.outer(t, inv_freq)
    return torch.cos(freqs), torch.sin(freqs)


def apply_rope(x: torch.Tensor, cos: torch.Tensor, sin: torch.Tensor) -> torch.Tensor:
    """x: [B, H, T, D]; cos/sin: [T, D/2] (ilgili pozisyonlara önceden dilimlenmiş)."""
    d2 = x.shape[-1] // 2
    x1, x2 = x[..., :d2], x[..., d2:]
    cos = cos.to(x.dtype)[None, None, :, :]
    sin = sin.to(x.dtype)[None, None, :, :]
    return torch.cat((x1 * cos - x2 * sin, x1 * sin + x2 * cos), dim=-1)


class SwiGLU(nn.Module):
    def __init__(self, d_model: int, intermediate: int):
        super().__init__()
        self.gate_proj = nn.Linear(d_model, intermediate, bias=False)
        self.up_proj = nn.Linear(d_model, intermediate, bias=False)
        self.down_proj = nn.Linear(intermediate, d_model, bias=False)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        return self.down_proj(F.silu(self.gate_proj(x)) * self.up_proj(x))
