"""Loglama (CSV + JSONL), sistem metrikleri (GPU/CPU/RAM) ve grafikler."""
from __future__ import annotations

import csv
import json
import os
import subprocess
import time
from typing import Dict, List, Optional

import psutil

TRAIN_FIELDS = ["time", "step", "tokens", "train_loss", "val_loss", "lr", "grad_norm", "tokens_per_sec",
                "steps_per_sec", "data_time", "fwd_bwd_time", "elapsed_s", "eta_s", "skipped_steps"]
SYS_FIELDS = ["time", "step", "cpu_percent", "ram_used_gb", "ram_total_gb", "gpu_index", "gpu_util_percent",
              "gpu_mem_used_mb", "gpu_mem_total_mb", "torch_max_alloc_mb"]


class MetricsLogger:
    def __init__(self, log_dir: str, resume: bool):
        os.makedirs(log_dir, exist_ok=True)
        self.csv_path = os.path.join(log_dir, "training_log.csv")
        self.jsonl_path = os.path.join(log_dir, "training_log.jsonl")
        self.sys_path = os.path.join(log_dir, "system_metrics.csv")
        if not resume:  # yeni eğitim: eski logların üstüne yaz (log biriktirme yok)
            for p in (self.csv_path, self.jsonl_path, self.sys_path):
                if os.path.exists(p):
                    os.remove(p)
        for p, fields in ((self.csv_path, TRAIN_FIELDS), (self.sys_path, SYS_FIELDS)):
            if not os.path.exists(p):
                with open(p, "w", newline="", encoding="utf-8") as f:
                    csv.writer(f).writerow(fields)

    def log_train(self, row: Dict) -> None:
        row = {"time": time.strftime("%Y-%m-%d %H:%M:%S"), **row}
        with open(self.csv_path, "a", newline="", encoding="utf-8") as f:
            csv.DictWriter(f, fieldnames=TRAIN_FIELDS, extrasaction="ignore").writerow({k: row.get(k, "") for k in TRAIN_FIELDS})
        with open(self.jsonl_path, "a", encoding="utf-8") as f:
            f.write(json.dumps(row) + "\n")

    def log_system(self, step: int, rows: List[Dict]) -> None:
        with open(self.sys_path, "a", newline="", encoding="utf-8") as f:
            w = csv.DictWriter(f, fieldnames=SYS_FIELDS, extrasaction="ignore")
            for r in rows:
                w.writerow({"time": time.strftime("%Y-%m-%d %H:%M:%S"), "step": step, **r})


def sample_system(torch_module=None) -> List[Dict]:
    vm = psutil.virtual_memory()
    base = {"cpu_percent": psutil.cpu_percent(interval=None), "ram_used_gb": round(vm.used / 1e9, 2),
            "ram_total_gb": round(vm.total / 1e9, 2)}
    rows = []
    try:
        out = subprocess.run(["nvidia-smi", "--query-gpu=index,utilization.gpu,memory.used,memory.total",
                              "--format=csv,noheader,nounits"], capture_output=True, text=True, timeout=5)
        if out.returncode == 0:
            for line in out.stdout.strip().splitlines():
                i, u, mu, mt = [x.strip() for x in line.split(",")]
                row = {**base, "gpu_index": int(i), "gpu_util_percent": float(u), "gpu_mem_used_mb": float(mu), "gpu_mem_total_mb": float(mt)}
                if torch_module is not None and torch_module.cuda.is_available() and int(i) < torch_module.cuda.device_count():
                    row["torch_max_alloc_mb"] = round(torch_module.cuda.max_memory_allocated(int(i)) / 1e6, 1)
                rows.append(row)
    except Exception:
        pass
    return rows or [base]


def _read_csv(path: str) -> List[Dict]:
    with open(path, encoding="utf-8") as f:
        return list(csv.DictReader(f))


def _f(x) -> Optional[float]:
    try:
        return float(x)
    except (TypeError, ValueError):
        return None


def make_plots(log_dir: str, plot_dir: str) -> List[str]:
    """loss_curve.png, learning_rate.png, gpu_usage.png, training_speed.png (veri varsa)."""
    import matplotlib
    matplotlib.use("Agg")
    import matplotlib.pyplot as plt
    os.makedirs(plot_dir, exist_ok=True)
    made = []
    tr = _read_csv(os.path.join(log_dir, "training_log.csv")) if os.path.exists(os.path.join(log_dir, "training_log.csv")) else []

    def series(rows, key):
        pts = [(_f(r["step"]), _f(r[key])) for r in rows if _f(r.get(key)) is not None and _f(r.get("step")) is not None]
        return [p[0] for p in pts], [p[1] for p in pts]

    if tr:
        fig, ax = plt.subplots(figsize=(8, 4.5))
        for key, lab in (("train_loss", "train"), ("val_loss", "validation")):
            x, y = series(tr, key)
            if x:
                ax.plot(x, y, label=lab, marker="o" if key == "val_loss" else None, ms=3)
        ax.set_xlabel("step"); ax.set_ylabel("loss"); ax.set_title("ALTUN 1 — loss"); ax.legend(); ax.grid(alpha=.3)
        p = os.path.join(plot_dir, "loss_curve.png"); fig.savefig(p, dpi=130, bbox_inches="tight"); plt.close(fig); made.append(p)
        for key, fname, title in (("lr", "learning_rate.png", "Learning rate"), ):
            x, y = series(tr, key)
            if x:
                fig, ax = plt.subplots(figsize=(8, 4)); ax.plot(x, y); ax.set_xlabel("step"); ax.set_title(title); ax.grid(alpha=.3)
                p = os.path.join(plot_dir, fname); fig.savefig(p, dpi=130, bbox_inches="tight"); plt.close(fig); made.append(p)
        x, y = series(tr, "tokens_per_sec")
        if x:
            fig, ax = plt.subplots(figsize=(8, 4)); ax.plot(x, y); ax.set_xlabel("step"); ax.set_ylabel("tokens/s (global)")
            ax.set_title("Training speed"); ax.grid(alpha=.3)
            p = os.path.join(plot_dir, "training_speed.png"); fig.savefig(p, dpi=130, bbox_inches="tight"); plt.close(fig); made.append(p)
    sp = os.path.join(log_dir, "system_metrics.csv")
    if os.path.exists(sp):
        rows = [r for r in _read_csv(sp) if _f(r.get("gpu_util_percent")) is not None]
        if rows:
            fig, axes = plt.subplots(2, 1, figsize=(8, 6), sharex=True)
            for gi in sorted({r["gpu_index"] for r in rows}):
                rr = [r for r in rows if r["gpu_index"] == gi]
                xs = list(range(len(rr)))
                axes[0].plot(xs, [_f(r["gpu_util_percent"]) for r in rr], label=f"GPU {gi}")
                axes[1].plot(xs, [_f(r["gpu_mem_used_mb"]) / 1024 for r in rr], label=f"GPU {gi}")
            axes[0].set_ylabel("utilization %"); axes[1].set_ylabel("memory GB"); axes[1].set_xlabel("sample")
            axes[0].legend(); axes[0].set_title("GPU usage"); [a.grid(alpha=.3) for a in axes]
            p = os.path.join(plot_dir, "gpu_usage.png"); fig.savefig(p, dpi=130, bbox_inches="tight"); plt.close(fig); made.append(p)
    return made
