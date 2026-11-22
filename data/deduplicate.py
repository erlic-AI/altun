"""Tekilleştirme (tam + MinHash-LSH yakın kopya) ve benchmark dekontaminasyonu."""
from __future__ import annotations

import hashlib
import re
import zlib
from collections import defaultdict
from typing import Dict, Iterable, List, Sequence, Set, Tuple

import numpy as np

_WS = re.compile(r"\s+")
_WORD = re.compile(r"\w+", re.UNICODE)
_PRIME = (1 << 61) - 1


def normalize_for_hash(text: str) -> str:
    return _WS.sub(" ", text.lower()).strip()


def exact_key(text: str) -> str:
    return hashlib.sha1(normalize_for_hash(text).encode("utf-8")).hexdigest()


def _signature(text: str, a: np.ndarray, b: np.ndarray, shingle: int = 5) -> np.ndarray:
    words = _WORD.findall(text.lower())
    if len(words) < shingle:
        grams = [" ".join(words)] if words else [""]
    else:
        grams = [" ".join(words[i:i + shingle]) for i in range(len(words) - shingle + 1)]
    h = np.fromiter((zlib.crc32(g.encode("utf-8")) for g in grams), dtype=np.uint64, count=len(grams))
    # (a*h + b) mod 2^61-1 yerine taşmasız yaklaşık: uint64 wrap-around evrensel hash olarak yeterli
    sig = ((h[None, :] * a[:, None]) + b[:, None]) >> np.uint64(32)
    return sig.min(axis=1)


def deduplicate(docs: Sequence[Dict], near: bool = True, threshold: float = 0.85,
                num_perm: int = 64, bands: int = 16, seed: int = 0,
                skip_near_categories: Sequence[str] = ()) -> Tuple[List[Dict], Dict[str, int]]:
    """docs: {'text': str, ...}. Giriş sırasını koruyarak ilk görülen kopyayı tutar."""
    stats = {"input": len(docs), "exact_dups": 0, "near_dups": 0}
    seen: Set[str] = set()
    uniq: List[Dict] = []
    for d in docs:
        k = exact_key(d["text"])
        if k in seen:
            stats["exact_dups"] += 1
            continue
        seen.add(k)
        uniq.append(d)
    if not near or len(uniq) < 2:
        stats["output"] = len(uniq)
        return uniq, stats

    rng = np.random.default_rng(seed)
    a = rng.integers(1, 2**63 - 1, size=num_perm, dtype=np.uint64) | np.uint64(1)
    b = rng.integers(0, 2**63 - 1, size=num_perm, dtype=np.uint64)
    rows = num_perm // bands
    buckets: Dict[Tuple[int, bytes], List[int]] = defaultdict(list)
    sigs: Dict[int, np.ndarray] = {}
    keep: List[Dict] = []
    skip = set(skip_near_categories)
    stats["near_skipped_docs"] = 0
    for d in uniq:
        if d.get("category") in skip:  # milyonlarca satırlık sentetik/QA verisinde MinHash çok yavaş; tam kopya yeterli
            keep.append(d)
            stats["near_skipped_docs"] += 1
            continue
        sig = _signature(d["text"], a, b)
        is_dup = False
        cand: Set[int] = set()
        keys = [(bi, sig[bi * rows:(bi + 1) * rows].tobytes()) for bi in range(bands)]
        for key in keys:
            cand.update(buckets.get(key, ()))
        for ci in cand:
            if float(np.mean(sigs[ci] == sig)) >= threshold:
                is_dup = True
                break
        if is_dup:
            stats["near_dups"] += 1
            continue
        idx = len(keep)
        keep.append(d)
        sigs[idx] = sig
        for key in keys:
            buckets[key].append(idx)
    stats["output"] = len(keep)
    return keep, stats


# ------------------------------------------------------- dekontaminasyon
def _ngram_hashes(text: str, n: int) -> Set[int]:
    w = _WORD.findall(text.lower())
    if len(w) < n:
        return set()
    return {zlib.crc32(" ".join(w[i:i + n]).encode("utf-8")) for i in range(len(w) - n + 1)}


def build_eval_ngrams(eval_texts: Iterable[str], n: int = 13) -> Set[int]:
    out: Set[int] = set()
    for t in eval_texts:
        out |= _ngram_hashes(t, n)
    return out


def decontaminate(docs: Sequence[Dict], eval_ngrams: Set[int], n: int = 13) -> Tuple[List[Dict], int]:
    """Benchmark/test metinleriyle n-gram kesişimi olan eğitim belgelerini çıkarır."""
    if not eval_ngrams:
        return list(docs), 0
    kept, removed = [], 0
    for d in docs:
        if _ngram_hashes(d["text"], n) & eval_ngrams:
            removed += 1
        else:
            kept.append(d)
    return kept, removed
