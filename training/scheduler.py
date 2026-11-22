"""Wall-clock tabanlı LR programı: step bazlı warmup + geçen süreye (0..1) bağlı cosine."""
from __future__ import annotations

import math


def lr_at(step: int, progress: float, peak_lr: float, min_lr_ratio: float, warmup_steps: int) -> float:
    """progress = geçen_süre / hedef_süre. Warmup bitince cosine, progress'e göre peak→min_lr'a iner."""
    min_lr = peak_lr * min_lr_ratio
    if warmup_steps > 0 and step < warmup_steps:
        return peak_lr * (step + 1) / warmup_steps
    p = min(max(progress, 0.0), 1.0)
    return min_lr + 0.5 * (peak_lr - min_lr) * (1.0 + math.cos(math.pi * p))
