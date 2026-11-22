"""DDP yardımcıları. Her GPU'da ayrı model kopyası vardır (32 GB birleşik bellek YOKTUR)."""
from __future__ import annotations

import os
import sys
from datetime import timedelta

import torch
import torch.distributed as dist


def setup():
    """torchrun ortam değişkenlerinden (RANK/LOCAL_RANK/WORLD_SIZE) başlatır. Döner: rank, local_rank, world, device."""
    rank = int(os.environ.get("RANK", 0))
    local = int(os.environ.get("LOCAL_RANK", 0))
    world = int(os.environ.get("WORLD_SIZE", 1))
    if torch.cuda.is_available():
        torch.cuda.set_device(local)
        device = torch.device("cuda", local)
    else:
        device = torch.device("cpu")
    if world > 1 and not dist.is_initialized():
        dist.init_process_group("nccl" if device.type == "cuda" else "gloo", timeout=timedelta(minutes=30))
    return rank, local, world, device


def cleanup() -> None:
    if dist.is_available() and dist.is_initialized():
        dist.destroy_process_group()


def barrier() -> None:
    if dist.is_available() and dist.is_initialized():
        dist.barrier()


def all_reduce_mean(t: torch.Tensor) -> torch.Tensor:
    if dist.is_available() and dist.is_initialized():
        dist.all_reduce(t, op=dist.ReduceOp.SUM)
        t = t / dist.get_world_size()
    return t


def all_reduce_max(t: torch.Tensor) -> torch.Tensor:
    if dist.is_available() and dist.is_initialized():
        dist.all_reduce(t, op=dist.ReduceOp.MAX)
    return t


def broadcast(t: torch.Tensor, src: int = 0) -> torch.Tensor:
    if dist.is_available() and dist.is_initialized():
        dist.broadcast(t, src)
    return t


def ddp_selfcheck() -> None:
    """torchrun ile çalıştırılır: iki süreç all_reduce yapabiliyor mu? (DDP deadlock/NCCL sorunu tespiti)"""
    rank, _, world, device = setup()
    t = torch.ones(1, device=device) * (rank + 1)
    if world > 1:
        dist.all_reduce(t)
    expected = sum(range(1, world + 1))
    ok = abs(t.item() - expected) < 1e-6
    print(f"[ddp-check] rank={rank} world={world} all_reduce={t.item()} beklenen={expected} {'OK' if ok else 'HATA'}", flush=True)
    barrier()
    cleanup()
    sys.exit(0 if ok else 3)


if __name__ == "__main__":
    ddp_selfcheck()
