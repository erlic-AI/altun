"""Paralel tokenization ve uint16 .bin yazımı."""
from __future__ import annotations

import multiprocessing as mp
import os
from typing import List, Sequence

import numpy as np

from altun.tokenizer import ALTUNTokenizer

_TOK = None


def _init(tok_dir: str) -> None:
    global _TOK
    _TOK = ALTUNTokenizer.load(tok_dir)


def _encode(text: str) -> np.ndarray:
    ids = _TOK.encode(text, add_bos=True, add_eos=True, allow_special=False)
    return np.asarray(ids, dtype=np.uint16)


def tokenize_texts(tok_dir: str, texts: Sequence[str], workers: int = 0) -> List[np.ndarray]:
    workers = workers or max(1, (os.cpu_count() or 2) - 0)
    if workers <= 1 or len(texts) < 64:
        _init(tok_dir)
        return [_encode(t) for t in texts]
    ctx = mp.get_context("fork") if "fork" in mp.get_all_start_methods() else mp.get_context()
    with ctx.Pool(workers, initializer=_init, initargs=(tok_dir,)) as pool:
        return pool.map(_encode, texts, chunksize=max(1, len(texts) // (workers * 8)))


def write_bin(path: str, arrays: Sequence[np.ndarray]) -> int:
    """Atomik yazım. Döner: toplam token sayısı."""
    tmp = path + ".tmp"
    total = 0
    with open(tmp, "wb") as f:
        for a in arrays:
            a.astype(np.uint16, copy=False).tofile(f)
            total += int(a.shape[0])
    os.replace(tmp, path)
    return total
