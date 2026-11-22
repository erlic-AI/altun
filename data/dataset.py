"""Memmap token veri seti. Batch'ler (seed, step, micro, rank)'dan deterministik türetilir;
bu sayede resume sonrası veri sırası bozulmaz ve ayrı bir 'veri konumu' saklamaya gerek kalmaz."""
from __future__ import annotations

import queue
import threading
import time
from typing import Callable, List, Optional, Tuple

import numpy as np


class TokenDataset:
    def __init__(self, path: str, seq_len: int):
        self.path, self.seq_len = path, seq_len
        self.data = np.memmap(path, dtype=np.uint16, mode="r")
        if len(self.data) < seq_len + 2:
            raise ValueError(f"{path}: yalnızca {len(self.data)} token var, seq_len={seq_len} için yetersiz")

    def __len__(self) -> int:
        return len(self.data)

    def get_batch(self, batch_size: int, *seed_parts: int) -> Tuple[np.ndarray, np.ndarray]:
        rng = np.random.default_rng(list(seed_parts))
        hi = len(self.data) - self.seq_len - 1
        offs = rng.integers(0, hi, size=batch_size)
        x = np.stack([self.data[o:o + self.seq_len] for o in offs]).astype(np.int64)
        y = np.stack([self.data[o + 1:o + 1 + self.seq_len] for o in offs]).astype(np.int64)
        return x, y


class Prefetcher:
    """Arka plan thread'iyle sonraki step'lerin micro-batch'lerini hazırlar (GPU'yu bekletmez)."""

    def __init__(self, make_step: Callable[[int], List], start_step: int, depth: int = 4):
        self.make_step, self.step = make_step, start_step
        self.q: "queue.Queue" = queue.Queue(maxsize=max(1, depth))
        self._stop = threading.Event()
        self.wait_time = 0.0
        self.thread = threading.Thread(target=self._run, daemon=True)
        self.thread.start()

    def _run(self) -> None:
        s = self.step
        while not self._stop.is_set():
            try:
                item = (s, self.make_step(s))
            except BaseException as e:  # hatayı tüketiciye ilet
                item = (s, e)
            while not self._stop.is_set():
                try:
                    self.q.put(item, timeout=0.5)
                    break
                except queue.Full:
                    continue
            if isinstance(item[1], BaseException):
                return
            s += 1

    def next(self) -> Tuple[int, List]:
        t0 = time.time()
        while True:
            try:
                s, item = self.q.get(timeout=120)
                break
            except queue.Empty:
                if not self.thread.is_alive():
                    raise RuntimeError("Veri yükleme thread'i öldü")
                raise RuntimeError("DataLoader 120 sn yanıt vermedi (hang)")
        self.wait_time = time.time() - t0
        if isinstance(item, BaseException):
            raise item
        return s, item

    def close(self) -> None:
        self._stop.set()
