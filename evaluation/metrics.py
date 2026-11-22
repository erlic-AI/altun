"""Değerlendirme metrikleri ve cevap kontrolü (torch gerektirmez)."""
from __future__ import annotations

import math
import re
from collections import defaultdict
from typing import Dict, List

_NUM = re.compile(r"-?\d+(?:\.\d+)?(?:/\d+)?")
_BOXED = re.compile(r"\\boxed\{([^{}]*(?:\{[^{}]*\}[^{}]*)*)\}")


def _to_float(s: str):
    try:
        if "/" in s:
            a, b = s.split("/")
            return float(a) / float(b)
        return float(s)
    except Exception:
        return None


def first_number(text: str):
    m = _NUM.search(text.replace(",", ""))
    return _to_float(m.group(0)) if m else None


def norm_text(s: str) -> str:
    return re.sub(r"[\s`$\\]", "", s.lower())


def last_number(text: str):
    ms = _NUM.findall(text.replace(",", ""))
    return _to_float(ms[-1]) if ms else None


def final_answer(generated: str) -> tuple:
    """Çözüm metninden nihai cevabı ayır. (metin, boxed_var_mı). Sonraki 'Problem:' bloğu kesilir."""
    g = generated.split("\nProblem:")[0].strip()
    b = extract_boxed(g)
    return (b, True) if b is not None else (g, False)


def check_answer(answer_type: str, expected: str, generated: str) -> bool:
    g, boxed = final_answer(generated)
    if answer_type == "number":
        v, e = (first_number(g) if boxed else last_number(g)), _to_float(str(expected))
        return v is not None and e is not None and abs(v - e) < 1e-6
    if answer_type == "yesno":
        words = re.findall(r"[a-zçğıöşü]+", g.lower())
        mp = {"yes": "yes", "true": "yes", "evet": "yes", "no": "no", "false": "no", "hayır": "no"}
        first = next((mp[w] for w in words if w in mp), None)
        return first == expected
    return norm_text(expected) in norm_text(g)[: len(norm_text(expected)) + 24]


def extract_boxed(text: str):
    m = _BOXED.findall(text)
    return m[-1].strip() if m else None


def accuracy_report(records: List[Dict]) -> Dict:
    """records: {'category','correct'(bool), ...} → toplam ve kategori doğrulukları."""
    total = len(records)
    correct = sum(1 for r in records if r["correct"])
    cats: Dict[str, List[bool]] = defaultdict(list)
    for r in records:
        cats[r["category"]].append(bool(r["correct"]))
    return {"total_questions": total, "correct": correct, "incorrect": total - correct,
            "accuracy": correct / total if total else 0.0,
            "category_accuracy": {c: {"n": len(v), "correct": sum(v), "accuracy": sum(v) / len(v)} for c, v in sorted(cats.items())}}


def is_degenerate(text: str) -> bool:
    """Rastgele/anlamsız çıktı sezgisi: boş, tek karakter döngüsü, çok düşük çeşitlilik, bozuk karakter."""
    t = text.strip()
    if len(t) == 0:
        return True
    if t.count("\ufffd") / len(t) > 0.05:
        return True
    toks = t.split()
    if len(toks) >= 8 and len(set(toks)) / len(toks) < 0.25:
        return True
    if len(t) >= 16 and len(set(t)) <= 3:
        return True
    return False


def degenerate_ratio(texts: List[str]) -> float:
    return sum(is_degenerate(t) for t in texts) / max(len(texts), 1)


def perplexity(loss: float) -> float:
    return math.exp(min(loss, 50.0))
