"""Kısa performans denemesi: micro batch / grad checkpointing kombinasyonlarını ölçer, OOM'u yakalar,
en yüksek tokens/s'li güvenli konfigürasyonu seçer ve 12 saatlik token kapasitesini TAHMİN eder."""
from __future__ import annotations

import gc
import json
import os
import sys
import time
from typing import Dict, List

import torch

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

from altun.config import ModelConfig, estimate_vram_gb  # noqa: E402
from altun.model import ALTUNModel  # noqa: E402
from training.optimizer import build_optimizer  # noqa: E402


def _trial(cfg: ModelConfig, mbs: int, ckpt: bool, device, steps: int = 4) -> Dict:
    model, opt = None, None
    try:
        torch.cuda.empty_cache()
        torch.cuda.reset_peak_memory_stats(device)
        model = ALTUNModel(cfg).to(device)
        model.grad_checkpointing = ckpt
        model.train()
        opt = build_optimizer(model, 1e-4, 0.1, 0.9, 0.95, "cuda")
        scaler = torch.amp.GradScaler("cuda") if hasattr(torch, "amp") and hasattr(torch.amp, "GradScaler") else torch.cuda.amp.GradScaler()
        x = torch.randint(0, cfg.vocab_size, (mbs, cfg.context_length), device=device)
        times = []
        for i in range(steps):
            torch.cuda.synchronize()
            t0 = time.time()
            with torch.autocast("cuda", dtype=torch.float16):
                loss = model(x, targets=x)["loss"]
            scaler.scale(loss).backward()
            scaler.step(opt)
            scaler.update()
            opt.zero_grad(set_to_none=True)
            torch.cuda.synchronize()
            if i >= 1:  # ilk adım ısınma
                times.append(time.time() - t0)
        tps = mbs * cfg.context_length / (sum(times) / len(times))
        return {"micro_batch_size": mbs, "grad_checkpointing": ckpt, "tokens_per_sec_per_gpu": round(tps, 1),
                "peak_mem_gb": round(torch.cuda.max_memory_allocated(device) / 1e9, 2), "status": "ok"}
    except torch.cuda.OutOfMemoryError:
        return {"micro_batch_size": mbs, "grad_checkpointing": ckpt, "status": "oom"}
    finally:
        del model, opt
        gc.collect()
        torch.cuda.empty_cache()


def autotune(cfg: ModelConfig, preset: str, out_path: str, duration_hours: float = 12.0, world: int = 1,
             candidates=(16, 8, 4, 2), vram_margin: float = 0.92) -> Dict:
    if not torch.cuda.is_available():
        raise RuntimeError("Autotune CUDA gerektirir")
    device = torch.device("cuda", 0)
    total_gb = torch.cuda.get_device_properties(0).total_memory / 1e9
    trials: List[Dict] = []
    for ckpt in (False, True):
        for mbs in candidates:
            if estimate_vram_gb(cfg, mbs, ckpt) > total_gb * 1.25:  # tahmin çok yüksekse denemeden atla
                trials.append({"micro_batch_size": mbs, "grad_checkpointing": ckpt, "status": "skipped_by_estimate"})
                continue
            r = _trial(cfg, mbs, ckpt, device)
            trials.append(r)
            print("[autotune]", r, flush=True)
    ok = [t for t in trials if t["status"] == "ok" and t["peak_mem_gb"] <= total_gb * vram_margin]
    if not ok:
        raise RuntimeError("Hiçbir konfigürasyon belleğe sığmadı; daha küçük bir preset (altun1-small) deneyin.")
    best = max(ok, key=lambda t: t["tokens_per_sec_per_gpu"])
    # DDP verimliliği bilinmiyor: 2 GPU için %90 varsayımı (TAHMİN) — gerçek değer eğitim logunda ölçülür.
    est_tps = best["tokens_per_sec_per_gpu"] * world * (0.9 if world > 1 else 1.0)
    res = {"preset": preset, "gpu": torch.cuda.get_device_name(0), "gpu_total_gb": round(total_gb, 1), "world": world,
           "trials": trials, "best": best, "estimated_global_tokens_per_sec": round(est_tps, 1),
           "estimated_tokens_in_duration": int(est_tps * duration_hours * 3600), "duration_hours": duration_hours,
           "note": "tokens/s tek GPU üzerinde ÖLÇÜLDÜ (4 adım); çok GPU ve süre projeksiyonu TAHMİNDİR."}
    os.makedirs(os.path.dirname(out_path), exist_ok=True)
    with open(out_path, "w", encoding="utf-8") as f:
        json.dump(res, f, indent=2)
    return res
