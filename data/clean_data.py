"""Veri temizleme ve kalite/bozuk/boş örnek tespiti."""
from __future__ import annotations

import re
import unicodedata
from typing import Dict, Tuple

_CTRL = re.compile(r"[\x00-\x08\x0b\x0c\x0e-\x1f\x7f]")
_MANY_BLANK = re.compile(r"\n{4,}")


def clean_text(text: str) -> str:
    text = text.replace("\r\n", "\n").replace("\r", "\n")
    text = _CTRL.sub("", text)
    text = unicodedata.normalize("NFC", text)
    text = "\n".join(line.rstrip() for line in text.split("\n"))
    text = _MANY_BLANK.sub("\n\n\n", text)
    return text.strip("\n") + "\n" if text.strip() else ""


def quality_check(text: str, min_chars: int = 64, max_chars: int = 2_000_000) -> Tuple[bool, str]:
    """(kabul, red_nedeni). Boş, bozuk, aşırı uzun/kısa, minify edilmiş ve tekrarlı örnekleri eler."""
    n = len(text)
    if n == 0 or not text.strip():
        return False, "empty"
    if n < min_chars:
        return False, "too_short"
    if n > max_chars:
        return False, "too_long"
    if text.count("�") / n > 0.01:
        return False, "corrupt_encoding"
    lines = text.split("\n")
    longest = max(len(l) for l in lines)
    if longest > 20000 and n / max(len(lines), 1) > 2000:
        return False, "minified_or_blob"
    alnum = sum(c.isalnum() for c in text[:200000])
    if alnum / min(n, 200000) < 0.25:
        return False, "low_alnum_ratio"
    nonempty = [l for l in lines if l.strip()]
    if len(nonempty) >= 20 and len(set(nonempty)) / len(nonempty) < 0.2:
        return False, "highly_repetitive"
    return True, "ok"


def outlier_stats(lengths) -> Dict[str, float]:
    """Uzunluk dağılımına göre aykırı değer eşiği (Q3 + 3*IQR)."""
    if not lengths:
        return {"q1": 0, "q3": 0, "upper_fence": 0, "n_outliers": 0}
    s = sorted(lengths)
    q1, q3 = s[len(s) // 4], s[(3 * len(s)) // 4]
    fence = q3 + 3 * (q3 - q1)
    return {"q1": q1, "q3": q3, "upper_fence": fence, "n_outliers": sum(1 for x in s if x > fence)}


def split_long(text: str, max_chars: int = 200_000):
    """Çok uzun belgeleri (örn. tek dosyalık Metamath set.mm) satır sınırlarında parçalara böler."""
    if len(text) <= max_chars:
        return [text]
    parts, cur, size = [], [], 0
    for line in text.split("\n"):
        if size + len(line) + 1 > max_chars and cur:
            parts.append("\n".join(cur) + "\n")
            cur, size = [], 0
        cur.append(line)
        size += len(line) + 1
    if cur:
        parts.append("\n".join(cur) + "\n")
    return parts
