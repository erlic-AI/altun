"""Atomik checkpoint, resume ve final model export."""
from __future__ import annotations

import glob
import hashlib
import json
import os
import random
import shutil
from typing import Any, Dict, List, Optional

import numpy as np
import torch

CKPT_PREFIX = "ckpt_"


def _sha256(path: str, chunk: int = 1 << 24) -> str:
    h = hashlib.sha256()
    with open(path, "rb") as f:
        while True:
            b = f.read(chunk)
            if not b:
                break
            h.update(b)
    return h.hexdigest()


def _atomic_write_json(path: str, obj: Dict[str, Any]) -> None:
    tmp = path + ".tmp"
    with open(tmp, "w", encoding="utf-8") as f:
        json.dump(obj, f, indent=2)
        f.flush()
        os.fsync(f.fileno())
    os.replace(tmp, path)


def capture_rng_state() -> Dict[str, Any]:
    st = {"python": random.getstate(), "numpy": np.random.get_state(),
          "torch": torch.get_rng_state()}
    if torch.cuda.is_available():
        st["cuda"] = torch.cuda.get_rng_state_all()
    return st


def restore_rng_state(st: Dict[str, Any]) -> None:
    try:
        random.setstate(st["python"])
        np.random.set_state(st["numpy"])
        torch.set_rng_state(st["torch"].cpu())
        if "cuda" in st and torch.cuda.is_available():
            states = st["cuda"]
            if len(states) == torch.cuda.device_count():
                torch.cuda.set_rng_state_all([s.cpu() for s in states])
    except Exception as e:  # RNG geri yükleme kritik değil; veri sırası step'ten deterministik türetilir
        print(f"[checkpoint] UYARI: RNG durumu geri yüklenemedi: {e}")


def list_checkpoints(ckpt_dir: str) -> List[str]:
    return sorted(glob.glob(os.path.join(ckpt_dir, CKPT_PREFIX + "*.pt")), reverse=True)


def _is_valid(path: str) -> bool:
    meta_path = path[:-3] + ".json"
    if not (os.path.isfile(path) and os.path.isfile(meta_path)):
        return False
    try:
        with open(meta_path, encoding="utf-8") as f:
            meta = json.load(f)
        return os.path.getsize(path) == meta["size"]
    except Exception:
        return False


def find_latest_checkpoint(ckpt_dir: str) -> Optional[str]:
    """En yeni GEÇERLİ checkpoint (boyut + manifest kontrolü). Bozuksa bir öncekine düşer."""
    for p in list_checkpoints(ckpt_dir):
        if _is_valid(p):
            return p
    return None


def save_checkpoint(ckpt_dir: str, state: Dict[str, Any], step: int, keep: int = 2,
                    min_free_gb: float = 3.0) -> str:
    os.makedirs(ckpt_dir, exist_ok=True)
    free_gb = shutil.disk_usage(ckpt_dir).free / 1e9
    if free_gb < min_free_gb:
        raise RuntimeError(f"Yetersiz disk: {free_gb:.1f} GB boş (< {min_free_gb} GB). Checkpoint yazılmadı; önceki checkpoint korunuyor.")
    final = os.path.join(ckpt_dir, f"{CKPT_PREFIX}{step:09d}.pt")
    tmp = final + ".tmp"
    try:
        torch.save(state, tmp)
        with open(tmp, "rb") as f:
            os.fsync(f.fileno())
        size = os.path.getsize(tmp)
        os.replace(tmp, final)
        _atomic_write_json(final[:-3] + ".json", {"step": step, "size": size, "sha256": _sha256(final)})
    except BaseException:
        if os.path.exists(tmp):
            os.remove(tmp)
        raise
    _cleanup(ckpt_dir, keep)
    return final


def _cleanup(ckpt_dir: str, keep: int) -> None:
    """Yalnızca en yeni `keep` GEÇERLİ checkpoint kalır. Geçerli sayısı keep'ten azsa hiçbir şey silinmez."""
    valid = [p for p in list_checkpoints(ckpt_dir) if _is_valid(p)]
    if len(valid) <= keep or not valid:
        return
    for p in valid[keep:]:
        for f in (p, p[:-3] + ".json"):
            if os.path.exists(f):
                os.remove(f)
    for t in glob.glob(os.path.join(ckpt_dir, "*.tmp")):  # yarım kalmış geçici dosyalar
        os.remove(t)


def clear_checkpoints(ckpt_dir: str) -> int:
    """Sıfırdan eğitim başlarken eski eğitimin yanlışlıkla yüklenmesini önler."""
    n = 0
    for f in glob.glob(os.path.join(ckpt_dir, CKPT_PREFIX + "*")):
        os.remove(f)
        n += 1
    return n


def load_checkpoint(path: str, map_location="cpu") -> Dict[str, Any]:
    if not _is_valid(path):
        raise RuntimeError(f"Checkpoint doğrulanamadı (eksik/bozuk): {path}")
    return torch.load(path, map_location=map_location, weights_only=False)


# ------------------------------------------------------------------ export
def export_final_model(model, model_cfg, out_dir: str, tokenizer=None, extra: Optional[Dict] = None) -> str:
    """final_model/: model.safetensors + config.json + tokenizer/."""
    os.makedirs(out_dir, exist_ok=True)
    raw = model.module if hasattr(model, "module") else model
    sd = {k: v.detach().cpu().contiguous() for k, v in raw.state_dict().items()}
    if model_cfg.tie_embeddings:
        sd.pop("lm_head.weight", None)  # paylaşılan tensör; yüklemede yeniden bağlanır
    weights_path = os.path.join(out_dir, "model.safetensors")
    try:
        from safetensors.torch import save_file
        save_file(sd, weights_path + ".tmp", metadata={"format": "pt", "model": "ALTUN-1"})
        os.replace(weights_path + ".tmp", weights_path)
    except ImportError:
        weights_path = os.path.join(out_dir, "model.pt")
        print("[export] UYARI: safetensors yok, model.pt olarak kaydedildi.")
        torch.save(sd, weights_path)
    cfg = {"model_type": "altun", "architecture": "ALTUNModel", **model_cfg.to_dict(), **(extra or {})}
    _atomic_write_json(os.path.join(out_dir, "config.json"), cfg)
    if tokenizer is not None:
        tokenizer.save(os.path.join(out_dir, "tokenizer"))
    return weights_path


def load_final_model(model_dir: str, device: str = "cpu", dtype: Optional[torch.dtype] = None):
    from .config import ModelConfig
    from .model import ALTUNModel
    with open(os.path.join(model_dir, "config.json"), encoding="utf-8") as f:
        raw = json.load(f)
    fields = ModelConfig.__dataclass_fields__.keys()
    cfg = ModelConfig.from_dict({k: v for k, v in raw.items() if k in fields})
    model = ALTUNModel(cfg)
    st_path = os.path.join(model_dir, "model.safetensors")
    if os.path.isfile(st_path):
        from safetensors.torch import load_file
        sd = load_file(st_path)
    else:
        sd = torch.load(os.path.join(model_dir, "model.pt"), map_location="cpu")
    missing, unexpected = model.load_state_dict(sd, strict=False)
    missing = [m for m in missing if not (cfg.tie_embeddings and m == "lm_head.weight")]
    if missing or unexpected:
        raise RuntimeError(f"Ağırlık uyuşmazlığı. Eksik={missing} Fazla={unexpected}")
    if dtype is not None:
        model = model.to(dtype)
    return model.to(device).eval(), cfg
