"""Config yükleme ve doğrulama (torch gerektirmez)."""
from __future__ import annotations

import os
from dataclasses import asdict, dataclass, fields
from typing import Any, Dict, Optional

import yaml

PROJECT_ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))


def load_yaml(path: str) -> Dict[str, Any]:
    if not os.path.isfile(path):
        raise FileNotFoundError(f"Config bulunamadı: {path}")
    with open(path, encoding="utf-8") as f:
        data = yaml.safe_load(f)
    if not isinstance(data, dict):
        raise ValueError(f"Config geçersiz (dict bekleniyordu): {path}")
    return data


def cfg_path(name: str) -> str:
    return os.path.join(PROJECT_ROOT, "config", name)


@dataclass
class ModelConfig:
    vocab_size: int = 16384
    n_layers: int = 16
    d_model: int = 1024
    n_heads: int = 16
    n_kv_heads: int = 4
    intermediate_size: int = 2816
    context_length: int = 1024
    rope_theta: float = 10000.0
    norm_eps: float = 1e-5
    tie_embeddings: bool = True

    @property
    def head_dim(self) -> int:
        return self.d_model // self.n_heads

    def validate(self) -> None:
        if self.d_model % self.n_heads != 0:
            raise ValueError("d_model, n_heads'e tam bölünmeli")
        if self.n_heads % self.n_kv_heads != 0:
            raise ValueError("n_heads, n_kv_heads'e tam bölünmeli")
        if self.head_dim % 2 != 0:
            raise ValueError("head_dim çift olmalı (RoPE)")
        if self.vocab_size >= 65536:
            raise ValueError("vocab_size < 65536 olmalı (uint16 veri formatı)")
        for k in ("n_layers", "d_model", "intermediate_size", "context_length"):
            if getattr(self, k) <= 0:
                raise ValueError(f"{k} pozitif olmalı")

    def param_count(self) -> int:
        """Analitik parametre sayısı (modelin gerçek sayısıyla testte karşılaştırılır)."""
        d, hd = self.d_model, self.head_dim
        attn = d * self.n_heads * hd + 2 * d * self.n_kv_heads * hd + self.n_heads * hd * d
        mlp = 3 * d * self.intermediate_size
        layer = attn + mlp + 2 * d
        total = self.n_layers * layer + d + self.vocab_size * d
        if not self.tie_embeddings:
            total += self.vocab_size * d
        return total

    def to_dict(self) -> Dict[str, Any]:
        return asdict(self)

    @classmethod
    def from_dict(cls, d: Dict[str, Any]) -> "ModelConfig":
        names = {f.name for f in fields(cls)}
        unknown = set(d) - names
        if unknown:
            raise ValueError(f"Bilinmeyen model config alanları: {sorted(unknown)}")
        cfg = cls(**d)
        cfg.validate()
        return cfg


def load_model_config(path: Optional[str] = None, preset: Optional[str] = None) -> ModelConfig:
    raw = load_yaml(path or cfg_path("model_config.yaml"))
    presets = raw.pop("presets", {})
    active = preset or raw.pop("active_preset", None)
    raw.pop("active_preset", None)
    if active not in presets:
        raise ValueError(f"Preset '{active}' yok. Mevcut: {sorted(presets)}")
    merged = {**raw, **presets[active]}
    return ModelConfig.from_dict(merged)


@dataclass
class TrainConfig:
    duration_hours: float = 12.0
    checkpoint_interval_minutes: float = 30
    eval_interval_steps: int = 250
    eval_batches: int = 20
    log_interval_steps: int = 10
    system_metrics_interval_seconds: float = 30
    seed: int = 1337
    tokens_per_step: int = 131072
    micro_batch_size: int = 8
    peak_lr: float = 6e-4
    min_lr_ratio: float = 0.1
    warmup_steps: int = 200
    weight_decay: float = 0.1
    beta1: float = 0.9
    beta2: float = 0.95
    grad_clip: float = 1.0
    fp16: bool = True
    grad_checkpointing: bool = True
    prefetch_batches: int = 4
    max_checkpoints: int = 2
    min_free_disk_gb: float = 3.0
    compile: str = "auto"          # off | auto (ölç, kazanç varsa kullan) | on

    def validate(self) -> None:
        if self.compile not in ("off", "auto", "on"):
            raise ValueError("compile: off|auto|on olmalı")
        if self.duration_hours <= 0:
            raise ValueError("duration_hours pozitif olmalı")
        if self.micro_batch_size <= 0 or self.tokens_per_step <= 0:
            raise ValueError("batch parametreleri pozitif olmalı")
        if not (0 < self.min_lr_ratio <= 1):
            raise ValueError("min_lr_ratio (0,1] aralığında olmalı")
        if self.max_checkpoints < 1:
            raise ValueError("max_checkpoints >= 1 olmalı")

    def to_dict(self) -> Dict[str, Any]:
        return asdict(self)

    @classmethod
    def from_dict(cls, d: Dict[str, Any]) -> "TrainConfig":
        names = {f.name for f in fields(cls)}
        unknown = set(d) - names
        if unknown:
            raise ValueError(f"Bilinmeyen eğitim config alanları: {sorted(unknown)}")
        cfg = cls(**d)
        cfg.validate()
        return cfg


def load_train_config(path: Optional[str] = None, overrides: Optional[Dict[str, Any]] = None) -> TrainConfig:
    raw = load_yaml(path or cfg_path("training_config.yaml"))
    raw.update({k: v for k, v in (overrides or {}).items() if v is not None})
    return TrainConfig.from_dict(raw)


def load_data_config(path: Optional[str] = None) -> Dict[str, Any]:
    raw = load_yaml(path or cfg_path("data_config.yaml"))
    mix = raw.get("mix_targets", {})
    if abs(sum(mix.values()) - 1.0) > 1e-6:
        raise ValueError(f"mix_targets toplamı 1 olmalı, şu an {sum(mix.values())}")
    if not raw.get("allowed_licenses"):
        raise ValueError("allowed_licenses boş olamaz")
    return raw


def estimate_vram_gb(cfg: ModelConfig, micro_batch: int, grad_checkpointing: bool = True) -> float:
    """TAHMİN (ölçüm değil): GPU başına VRAM. Gerçek tepe değer autotune/eğitim sırasında ölçülür."""
    n = cfg.param_count()
    static = n * (4 + 4 + 8)  # fp32 ağırlık + grad + AdamW(m,v)
    t, d, v = cfg.context_length, cfg.d_model, cfg.vocab_size
    per_layer_full = micro_batch * t * (16 * d + 6 * cfg.intermediate_size) * 2
    acts = micro_batch * t * cfg.n_layers * 2 * d + (per_layer_full if grad_checkpointing else per_layer_full * cfg.n_layers)
    logits = micro_batch * t * v * (2 + 4 + 4)
    return (static + acts + logits) / 1e9 + 0.6  # + CUDA context/fragmentasyon payı
