"""ALTUN 1: sıfırdan decoder-only Transformer (RMSNorm, RoPE, SwiGLU, GQA, SDPA)."""
from __future__ import annotations

import math
from typing import List, Optional, Tuple

import torch
import torch.nn as nn
import torch.nn.functional as F
from torch.utils.checkpoint import checkpoint

from .attention import CausalSelfAttention
from .config import ModelConfig
from .layers import RMSNorm, SwiGLU, precompute_rope


class Block(nn.Module):
    def __init__(self, cfg: ModelConfig):
        super().__init__()
        self.attn_norm = RMSNorm(cfg.d_model, cfg.norm_eps)
        self.attn = CausalSelfAttention(cfg)
        self.mlp_norm = RMSNorm(cfg.d_model, cfg.norm_eps)
        self.mlp = SwiGLU(cfg.d_model, cfg.intermediate_size)

    def forward(self, x, cos, sin, past_kv=None, use_cache=False):
        h, present = self.attn(self.attn_norm(x), cos, sin, past_kv, use_cache)
        x = x + h
        x = x + self.mlp(self.mlp_norm(x))
        return x, present


class ALTUNModel(nn.Module):
    def __init__(self, cfg: ModelConfig):
        super().__init__()
        cfg.validate()
        self.cfg = cfg
        self.grad_checkpointing = False
        self.embed = nn.Embedding(cfg.vocab_size, cfg.d_model)
        self.blocks = nn.ModuleList([Block(cfg) for _ in range(cfg.n_layers)])
        self.final_norm = RMSNorm(cfg.d_model, cfg.norm_eps)
        self.lm_head = nn.Linear(cfg.d_model, cfg.vocab_size, bias=False)
        if cfg.tie_embeddings:
            self.lm_head.weight = self.embed.weight
        cos, sin = precompute_rope(cfg.head_dim, cfg.context_length, cfg.rope_theta)
        self.register_buffer("rope_cos", cos, persistent=False)
        self.register_buffer("rope_sin", sin, persistent=False)
        self.apply(self._init)
        for name, p in self.named_parameters():  # GPT-2 tarzı rezidüel ölçekleme
            if name.endswith("o_proj.weight") or name.endswith("down_proj.weight"):
                nn.init.normal_(p, mean=0.0, std=0.02 / math.sqrt(2 * cfg.n_layers))

    @staticmethod
    def _init(m: nn.Module) -> None:
        if isinstance(m, nn.Linear):
            nn.init.normal_(m.weight, mean=0.0, std=0.02)
        elif isinstance(m, nn.Embedding):
            nn.init.normal_(m.weight, mean=0.0, std=0.02)

    def num_parameters(self) -> int:
        return sum(p.numel() for p in self.parameters())  # paylaşılan ağırlık bir kez sayılır

    def forward(self, input_ids: torch.Tensor, targets: Optional[torch.Tensor] = None,
                past_kvs: Optional[List] = None, use_cache: bool = False):
        b, t = input_ids.shape
        start = 0 if past_kvs is None else past_kvs[0][0].shape[2]
        if start + t > self.cfg.context_length:
            raise ValueError(f"Dizi uzunluğu {start + t} > context_length {self.cfg.context_length}")
        x = self.embed(input_ids)
        cos, sin = self.rope_cos[start:start + t], self.rope_sin[start:start + t]
        presents: List[Tuple[torch.Tensor, torch.Tensor]] = []
        for i, blk in enumerate(self.blocks):
            if self.grad_checkpointing and self.training and past_kvs is None and not use_cache:
                x, _ = checkpoint(blk, x, cos, sin, None, False, use_reentrant=False)
            else:
                x, pr = blk(x, cos, sin, None if past_kvs is None else past_kvs[i], use_cache)
                if use_cache:
                    presents.append(pr)
        x = self.final_norm(x)
        logits = self.lm_head(x)
        loss = None
        if targets is not None:
            loss = F.cross_entropy(logits.float().view(-1, logits.size(-1)),
                                   targets.reshape(-1), ignore_index=-100)
        return {"logits": logits, "loss": loss, "past_kvs": presents if use_cache else None}
