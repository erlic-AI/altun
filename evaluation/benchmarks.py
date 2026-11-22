"""Standart benchmark'lar (GSM8K, MATH, AIME, MiniF2F, Lean).

Dürüstlük kuralları:
- Veri dosyası yoksa benchmark "not_run" olarak raporlanır; sonuç UYDURULMAZ.
- Benchmark dosyaları kullanıcı tarafından lisansına uygun biçimde sağlanır:
  <veri_kökü>/_benchmarks/gsm8k*.jsonl, math*.jsonl, aime*.jsonl, minif2f*.jsonl
  Bu dosyalar eğitim hattında dekontaminasyon için de kullanılır (n-gram kesişimi olan belgeler atılır).
- MiniF2F / Lean formal ispat doğrulaması gerçek bir Lean 4 + Mathlib kurulumu gerektirir; bu sistemde
  Lean doğrulayıcı YOKTUR, bu yüzden 'not_run' raporlanır.
"""
from __future__ import annotations

import glob
import json
import os
import re
import shutil
from typing import Callable, Dict, List, Optional

from altun.config import PROJECT_ROOT
from evaluation.metrics import check_answer, extract_boxed, first_number

SPECS = {
    "GSM8K": {"glob": "gsm8k*.jsonl", "license": "MIT"},
    "MATH": {"glob": "math*.jsonl", "license": "MIT"},
    "AIME": {"glob": "aime*.jsonl", "license": "Kullanıcı sorumluluğunda (MAA telifli olabilir)"},
    "MiniF2F": {"glob": "minif2f*.jsonl", "license": "MIT/Apache-2.0 (Lean doğrulayıcı gerekir)"},
}


def find_benchmark_files(roots: List[str]) -> Dict[str, Optional[str]]:
    found: Dict[str, Optional[str]] = {k: None for k in SPECS}
    for root in roots:
        root = root if os.path.isabs(root) else os.path.join(PROJECT_ROOT, root)
        for name, spec in SPECS.items():
            if found[name]:
                continue
            hits = glob.glob(os.path.join(root, "**", "_benchmarks", spec["glob"]), recursive=True) + \
                   glob.glob(os.path.join(root, "_benchmarks", spec["glob"]))
            if hits:
                found[name] = sorted(hits)[0]
    return found


def _load(path: str, limit: Optional[int]) -> List[Dict]:
    rows = []
    with open(path, encoding="utf-8") as f:
        for line in f:
            if line.strip():
                try:
                    rows.append(json.loads(line))
                except Exception:
                    continue
            if limit and len(rows) >= limit:
                break
    return rows


def _gsm_gold(ans: str) -> Optional[float]:
    return first_number(ans.split("####")[-1].strip()) if "####" in ans else first_number(ans)


def run_benchmark(name: str, path: Optional[str], generate: Callable[[str], str], limit: Optional[int] = 200) -> Dict:
    if name == "MiniF2F":
        return {"status": "not_run", "reason": "Lean 4 + Mathlib doğrulayıcı bu ortamda yok; formal ispat doğrulaması yapılamaz.",
                "data_file": path}
    if not path:
        return {"status": "not_run", "reason": f"Veri dosyası bulunamadı (beklenen: _benchmarks/{SPECS[name]['glob']})."}
    rows = _load(path, limit)
    if not rows:
        return {"status": "not_run", "reason": f"Dosya boş/okunamadı: {path}"}
    correct, examples = 0, []
    for row in rows:
        if name == "GSM8K":
            prompt = f"Problem: {row['question']}\nSolution:"
            gold = _gsm_gold(str(row["answer"]))
            out = generate(prompt)
            nums = re.findall(r"-?\d+(?:\.\d+)?", out.replace(",", ""))
            pred = float(nums[-1]) if nums else None
            ok = pred is not None and gold is not None and abs(pred - gold) < 1e-6
        elif name == "MATH":
            prompt = f"Problem: {row['problem']}\nSolution:"
            gold = extract_boxed(str(row.get("solution", row.get("answer", ""))))
            out = generate(prompt)
            pred = extract_boxed(out)
            ok = gold is not None and pred is not None and re.sub(r"\s", "", pred) == re.sub(r"\s", "", gold)
        else:  # AIME
            prompt = f"Problem: {row.get('problem', row.get('question'))}\nFinal answer (integer):"
            gold = str(row.get("answer"))
            out = generate(prompt)
            ok = check_answer("number", gold, out)
        correct += int(ok)
        if not ok and len(examples) < 5:
            examples.append({"prompt": prompt[:300], "gold": str(gold), "output": out[:300]})
    return {"status": "run", "data_file": path, "n_evaluated": len(rows), "correct": correct,
            "accuracy": correct / len(rows), "license": SPECS[name]["license"], "failed_examples": examples,
            "note": "0-shot, greedy; ilk n örnek"}


def lean_available() -> bool:
    return shutil.which("lean") is not None
