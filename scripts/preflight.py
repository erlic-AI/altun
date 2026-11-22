"""Preflight: donanım, bağımlılık, config tutarlılığı, disk, veri, DDP ve kısa performans denemesi.
Başarısızsa eğitim BAŞLATILMAZ. Sonuç: <results>/reports/preflight.json"""
from __future__ import annotations

import argparse
import importlib
import json
import os
import shutil
import subprocess
import sys
from typing import Dict

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

from altun.config import (PROJECT_ROOT, cfg_path, estimate_vram_gb, load_data_config, load_model_config,  # noqa: E402
                          load_train_config, load_yaml)
from scripts.environment_check import collect, render_md, validate  # noqa: E402

REQUIRED = ["torch", "numpy", "yaml", "psutil", "matplotlib"]
OPTIONAL = {"safetensors": "model.safetensors için (yoksa model.pt'ye düşülür)"}


def check_deps() -> Dict:
    missing, optional_missing = [], []
    for m in REQUIRED:
        try:
            importlib.import_module(m)
        except ImportError:
            missing.append(m)
    for m, why in OPTIONAL.items():
        try:
            importlib.import_module(m)
        except ImportError:
            optional_missing.append(f"{m} ({why})")
    return {"missing": missing, "optional_missing": optional_missing}


def ddp_check(timeout: int = 180) -> Dict:
    """2 süreçli NCCL all_reduce denemesi (deadlock/NCCL sorunlarını eğitimden ÖNCE yakalar)."""
    cmd = [sys.executable, "-m", "torch.distributed.run", "--standalone", "--nproc_per_node=2", "-m", "training.distributed"]
    try:
        p = subprocess.run(cmd, cwd=PROJECT_ROOT, capture_output=True, text=True, timeout=timeout,
                           env={**os.environ, "PYTHONPATH": PROJECT_ROOT, "NCCL_ASYNC_ERROR_HANDLING": "1"})
        return {"ok": p.returncode == 0, "output": (p.stdout + p.stderr)[-1500:]}
    except subprocess.TimeoutExpired:
        return {"ok": False, "output": f"DDP kontrolü {timeout} sn içinde bitmedi (olası NCCL deadlock)"}


def run_preflight(results_dir: str, preset: str = None, duration_hours: float = None, autotune: bool = True,
                  skip_ddp: bool = False, allow_cpu: bool = False, require_data: bool = True) -> Dict:
    rep_dir = os.path.join(results_dir, "reports")
    os.makedirs(rep_dir, exist_ok=True)
    errors, warnings = [], []
    info = collect(results_dir)
    errors += validate(info, require_gpu=not allow_cpu)
    deps = check_deps()
    if deps["missing"]:
        errors.append(f"Eksik zorunlu paketler: {deps['missing']} (pip install ile kurun)")
    warnings += [f"Opsiyonel paket yok: {m}" for m in deps["optional_missing"]]

    summary: Dict = {"hardware": info, "deps": deps}
    try:
        mcfg = load_model_config(preset=preset)
        tcfg = load_train_config()
        dcfg = load_data_config()
        if mcfg.vocab_size != dcfg["vocab_size"]:
            errors.append(f"Config uyuşmazlığı: model vocab {mcfg.vocab_size} != data vocab {dcfg['vocab_size']}")
        summary["model"] = {"preset": preset, "params_estimate": mcfg.param_count(), "context": mcfg.context_length}
        summary["vram_estimate_gb"] = round(estimate_vram_gb(mcfg, tcfg.micro_batch_size, tcfg.grad_checkpointing), 2)
    except Exception as e:
        errors.append(f"Config doğrulama hatası: {e}")
        mcfg = None

    free = shutil.disk_usage(results_dir).free / 1e9
    if mcfg and free < 8 + 2 * mcfg.param_count() * 16 / 1e9:
        errors.append(f"Disk yetersiz: {free:.1f} GB boş; checkpoint + veri için daha fazla alan gerekir")

    if require_data:
        meta = os.path.join(PROJECT_ROOT, "outputs", "data", "meta.json")
        from data.prepare_data import discover_sources
        n_src = len(discover_sources(load_data_config()["data_roots"])) if os.path.isfile(os.path.join(PROJECT_ROOT, "config", "data_config.yaml")) else 0
        if not os.path.isfile(meta) and n_src == 0:
            errors.append("Eğitim verisi yok: raw_data/ veya /kaggle/input altında source.json içeren kaynak bulunamadı (README → Veri sağlama).")
        summary["data_sources_found"] = n_src

    world = 1
    if info["cuda_available"]:
        world = info["gpu_count"] if info["gpu_count"] in (1, 2) else min(info["gpu_count"], 2)
        if world == 2 and not skip_ddp and not errors:
            d = ddp_check()
            summary["ddp_check"] = d
            if not d["ok"]:
                warnings.append("DDP kontrolü BAŞARISIZ → tek GPU stratejisine düşüldü. Ayrıntı preflight.json içinde.")
                world = 1
    summary["world_size"] = world
    strategy = (f"DDP, {world} GPU (her GPU'da tam model kopyası; VRAM birleşmez), FP16 AMP" if world > 1 else
                "Tek GPU, FP16 AMP" if info["cuda_available"] else "CPU (yalnızca test)")

    if autotune and not errors and info["cuda_available"] and mcfg:
        from training.autotune import autotune as run_autotune
        try:
            dh = duration_hours or tcfg.duration_hours
            summary["autotune"] = run_autotune(mcfg, preset or load_yaml(cfg_path("model_config.yaml"))["active_preset"], os.path.join(rep_dir, "autotune.json"), dh, world)
        except Exception as e:
            errors.append(f"Autotune başarısız: {e}")

    summary.update({"ok": not errors, "errors": errors, "warnings": warnings, "strategy": strategy})
    with open(os.path.join(rep_dir, "hardware_report.md"), "w", encoding="utf-8") as f:
        f.write(render_md(info, errors, strategy))
    with open(os.path.join(rep_dir, "preflight.json"), "w", encoding="utf-8") as f:
        json.dump(summary, f, indent=2, default=str)
    return summary


def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("--results-dir", required=True)
    ap.add_argument("--preset", default=None)
    ap.add_argument("--duration-hours", type=float, default=None)
    ap.add_argument("--no-autotune", action="store_true")
    ap.add_argument("--skip-ddp", action="store_true")
    ap.add_argument("--allow-cpu", action="store_true")
    ap.add_argument("--no-data-check", action="store_true")
    a = ap.parse_args()
    s = run_preflight(a.results_dir, a.preset, a.duration_hours, not a.no_autotune, a.skip_ddp, a.allow_cpu, not a.no_data_check)
    for w in s["warnings"]:
        print("[UYARI]", w)
    if not s["ok"]:
        print("\n[PREFLIGHT BAŞARISIZ]\n- " + "\n- ".join(s["errors"]), file=sys.stderr)
        sys.exit(2)
    print(f"[PREFLIGHT OK] strateji: {s['strategy']}")


if __name__ == "__main__":
    main()
