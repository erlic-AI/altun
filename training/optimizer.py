"""AdamW: weight decay yalnızca 2B+ ağırlıklara (matris/embedding); norm ağırlıkları decay'siz."""
from __future__ import annotations

import inspect

import torch


def build_optimizer(model, lr: float, weight_decay: float, beta1: float, beta2: float, device_type: str):
    params = {n: p for n, p in model.named_parameters() if p.requires_grad}
    decay = [p for p in params.values() if p.dim() >= 2]
    no_decay = [p for p in params.values() if p.dim() < 2]
    groups = [{"params": decay, "weight_decay": weight_decay}, {"params": no_decay, "weight_decay": 0.0}]
    kwargs = {"lr": lr, "betas": (beta1, beta2), "eps": 1e-8}
    if device_type == "cuda" and "fused" in inspect.signature(torch.optim.AdamW).parameters:
        kwargs["fused"] = True
    return torch.optim.AdamW(groups, **kwargs)
