"""Eğitim girişi. Tek GPU: `python -m training.train ...`; iki GPU: `torchrun --nproc_per_node=2 -m training.train ...`
Genellikle scripts/run_training.py tarafından başlatılır."""
from __future__ import annotations

import argparse
import json
import os
import sys

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

from altun.config import PROJECT_ROOT, load_model_config, load_train_config  # noqa: E402


def parse_resume(v: str):
    v = str(v).lower()
    if v in ("auto", "true", "1", "yes"):
        return "auto" if v == "auto" else True
    if v in ("false", "0", "no"):
        return False
    raise argparse.ArgumentTypeError("resume: auto|true|false")


def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("--results-dir", required=True)
    ap.add_argument("--data-dir", default="outputs/data")
    ap.add_argument("--preset", default=None)
    ap.add_argument("--duration-hours", type=float, default=None)
    ap.add_argument("--resume", type=parse_resume, default="auto")
    ap.add_argument("--max-steps", type=int, default=None)
    ap.add_argument("--micro-batch-size", type=int, default=None)
    ap.add_argument("--tokens-per-step", type=int, default=None)
    ap.add_argument("--init-from", default=None, help="Önceki final_model klasörü: ağırlıklarla başla (yeni LR programı)")
    ap.add_argument("--peak-lr", type=float, default=None)
    ap.add_argument("--compile", default=None, choices=["off", "auto", "on"])
    ap.add_argument("--vocab-size", type=int, default=None, help="Yalnızca smoke testi: model vocab'ını veriye göre ayarla")
    ap.add_argument("--allow-smoke", action="store_true", help="Sentetik duman verisiyle çalışmaya izin ver (yalnızca test)")
    a = ap.parse_args()

    from training.trainer import Trainer
    mcfg = load_model_config(preset=a.preset)
    if a.vocab_size:
        mcfg.vocab_size = a.vocab_size
    over = {"micro_batch_size": a.micro_batch_size, "tokens_per_step": a.tokens_per_step, "peak_lr": a.peak_lr, "compile": a.compile}
    ap_path = os.path.join(a.results_dir, "reports", "autotune.json")
    if os.path.isfile(ap_path) and a.micro_batch_size is None:
        with open(ap_path, encoding="utf-8") as f:
            at = json.load(f)
        if at.get("preset") == (a.preset or mcfg_name()) and at.get("best"):
            over["micro_batch_size"] = at["best"]["micro_batch_size"]
            over["grad_checkpointing"] = at["best"]["grad_checkpointing"]
            print(f"[train] autotune sonucu uygulandı: {at['best']}")
    tcfg = load_train_config(overrides=over)
    data_dir = a.data_dir if os.path.isabs(a.data_dir) else os.path.join(PROJECT_ROOT, a.data_dir)
    tr = Trainer(mcfg, tcfg, data_dir, a.results_dir, resume=a.resume, max_steps=a.max_steps, allow_smoke=a.allow_smoke,
                 init_from=a.init_from)
    try:
        res = tr.run(a.duration_hours)
        if tr.main:
            print("[train] SONUÇ:", json.dumps(res, default=str))
    finally:
        from training.distributed import cleanup
        cleanup()


def mcfg_name() -> str:
    from altun.config import load_yaml, cfg_path
    return load_yaml(cfg_path("model_config.yaml")).get("active_preset")


if __name__ == "__main__":
    main()
