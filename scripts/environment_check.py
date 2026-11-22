"""Donanım/ortam denetimi. Başarısızsa eğitim başlatılmaz (çıkış kodu 2)."""
from __future__ import annotations

import argparse
import json
import os
import platform
import shutil
import sys

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

import psutil  # noqa: E402


def collect(path_for_disk: str = ".") -> dict:
    info = {"python": platform.python_version(), "platform": platform.platform(), "cpu_cores_logical": os.cpu_count(),
            "cpu_cores_physical": psutil.cpu_count(logical=False), "ram_total_gb": round(psutil.virtual_memory().total / 1e9, 1),
            "ram_available_gb": round(psutil.virtual_memory().available / 1e9, 1),
            "disk_free_gb": round(shutil.disk_usage(path_for_disk).free / 1e9, 1),
            "torch": None, "cuda_version": None, "cuda_available": False, "gpu_count": 0, "gpus": [], "problems": []}
    try:
        import torch
        info["torch"] = torch.__version__
        info["cuda_version"] = torch.version.cuda
        info["cuda_available"] = torch.cuda.is_available()
        if torch.cuda.is_available():
            info["gpu_count"] = torch.cuda.device_count()
            for i in range(info["gpu_count"]):
                p = torch.cuda.get_device_properties(i)
                info["gpus"].append({"index": i, "name": p.name, "vram_gb": round(p.total_memory / 1e9, 1),
                                     "compute_capability": f"{p.major}.{p.minor}",
                                     "bf16_supported": bool(torch.cuda.is_bf16_supported()) if i == 0 else None})
    except ImportError:
        info["problems"].append("PyTorch kurulu değil")
    return info


def validate(info: dict, require_gpu: bool = True, min_disk_gb: float = 10.0) -> list:
    probs = list(info["problems"])
    if info["torch"] is None:
        return probs
    if require_gpu and not info["cuda_available"]:
        probs.append("CUDA kullanılamıyor: Kaggle notebook ayarlarında Accelerator → 'GPU T4 x2' seçin ve oturumu yeniden başlatın.")
    if require_gpu and info["cuda_available"]:
        for g in info["gpus"]:
            if g["vram_gb"] < 8:
                probs.append(f"GPU{g['index']} VRAM yetersiz: {g['vram_gb']} GB")
    if info["disk_free_gb"] < min_disk_gb:
        probs.append(f"Boş disk yetersiz: {info['disk_free_gb']} GB (< {min_disk_gb} GB)")
    return probs


def render_md(info: dict, probs: list, strategy: str = "") -> str:
    L = ["# ALTUN 1 Donanım Raporu", "", f"- Python: {info['python']}", f"- PyTorch: {info['torch']} | CUDA: {info['cuda_version']} | CUDA kullanılabilir: {info['cuda_available']}",
         f"- CPU çekirdek (mantıksal/fiziksel): {info['cpu_cores_logical']}/{info['cpu_cores_physical']}",
         f"- RAM: {info['ram_total_gb']} GB (kullanılabilir {info['ram_available_gb']} GB)", f"- Boş disk: {info['disk_free_gb']} GB",
         f"- GPU sayısı: {info['gpu_count']}", ""]
    for g in info["gpus"]:
        L.append(f"  - GPU{g['index']}: {g['name']} | {g['vram_gb']} GB | compute capability {g['compute_capability']} | BF16: {g['bf16_supported']}")
    if strategy:
        L += ["", f"**Seçilen strateji:** {strategy}"]
    L += ["", "**Sorunlar:** " + ("; ".join(probs) if probs else "yok")]
    return "\n".join(L) + "\n"


def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("--allow-cpu", action="store_true")
    ap.add_argument("--out", default=None, help="hardware_report.md yolu")
    a = ap.parse_args()
    info = collect()
    probs = validate(info, require_gpu=not a.allow_cpu)
    print(json.dumps(info, indent=2))
    if a.out:
        os.makedirs(os.path.dirname(a.out), exist_ok=True)
        with open(a.out, "w", encoding="utf-8") as f:
            f.write(render_md(info, probs))
    if probs:
        print("\n[DONANIM HATASI]\n- " + "\n- ".join(probs), file=sys.stderr)
        sys.exit(2)


if __name__ == "__main__":
    main()
