"""Veri hattı: ingestion → inceleme → temizleme → dedup → lisans/provenance → split →
tokenizer eğitimi → tokenization → paketleme → istatistik → kalite raporu.

Kullanım:
  python -m data.prepare_data                       # gerçek veri (raw_data/ veya /kaggle/input)
  python -m data.prepare_data --smoke               # yalnızca sentetik duman verisi (gerçek eğitim için KULLANILMAZ)
  python -m data.prepare_data --budget-tokens 6e8   # token bazlı karışım oranını bu bütçeye göre uygula
"""
from __future__ import annotations

import argparse
import hashlib
import json
import os
import random
import sys
from collections import Counter, defaultdict
from typing import Dict, List, Optional, Tuple

import numpy as np

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
from altun.config import PROJECT_ROOT, load_data_config  # noqa: E402
from altun.tokenizer import ALTUNTokenizer  # noqa: E402
from data.clean_data import clean_text, outlier_stats, quality_check, split_long  # noqa: E402
from data.deduplicate import build_eval_ngrams, decontaminate, deduplicate, exact_key  # noqa: E402
from data.tokenize_data import tokenize_texts, write_bin  # noqa: E402

TEXT_EXT = {".txt", ".md", ".lean", ".tex", ".py", ".rst", ".v", ".thy", ".mm"}


class DataError(RuntimeError):
    pass


# ----------------------------------------------------------------- ingestion
def discover_sources(roots: List[str], max_depth: int = 4) -> List[Dict]:
    found = []
    for root in roots:
        root = root if os.path.isabs(root) else os.path.join(PROJECT_ROOT, root)
        if not os.path.isdir(root):
            continue
        base_depth = root.rstrip(os.sep).count(os.sep)
        for dp, dn, fn in os.walk(root):
            dn.sort()  # deterministik sıra (oturumlar arası aynı tokenizer/split için)
            if dp.count(os.sep) - base_depth >= max_depth:
                dn[:] = []
            if "source.json" in fn:
                try:
                    with open(os.path.join(dp, "source.json"), encoding="utf-8") as f:
                        meta = json.load(f)
                except Exception as e:
                    found.append({"path": dp, "name": os.path.basename(dp), "error": f"source.json okunamadı: {e}"})
                    continue
                meta.setdefault("name", os.path.basename(dp))
                meta["path"] = dp
                found.append(meta)
    found.sort(key=lambda m: m.get("path", ""))
    return found


def check_license(src: Dict, allowed: List[str], categories: List[str]) -> Optional[str]:
    if "error" in src:
        return src["error"]
    lic = src.get("license")
    if not lic:
        return "lisans belirtilmemiş"
    if lic not in allowed:
        return f"lisans izinli listede değil: {lic}"
    if src.get("category") not in categories:
        return f"geçersiz kategori: {src.get('category')} (geçerli: {categories})"
    return None


def read_source(src: Dict, counters: Counter) -> List[Dict]:
    docs = []
    for dp, dn, fns in os.walk(src["path"]):
        dn.sort()
        for fn in sorted(fns):
            if fn == "source.json":
                continue
            path = os.path.join(dp, fn)
            ext = os.path.splitext(fn)[1].lower()
            rel = os.path.relpath(path, src["path"])
            try:
                if ext in TEXT_EXT:
                    with open(path, encoding="utf-8", errors="strict") as f:
                        docs.append({"text": f.read(), "source": src["name"], "category": src["category"], "file": rel})
                elif ext == ".jsonl":
                    with open(path, encoding="utf-8") as f:
                        for i, line in enumerate(f):
                            line = line.strip()
                            if not line:
                                continue
                            try:
                                rec = json.loads(line)
                                txt = rec.get("text") or rec.get("content") if isinstance(rec, dict) else None
                                if not isinstance(txt, str):
                                    raise ValueError("text alanı yok")
                                docs.append({"text": txt, "source": src["name"], "category": src["category"], "file": f"{rel}:{i}"})
                            except Exception:
                                counters["corrupt_jsonl_lines"] += 1
            except UnicodeDecodeError:
                counters["undecodable_files"] += 1
            except OSError:
                counters["unreadable_files"] += 1
    return docs


def smoke_docs() -> List[Dict]:
    """YALNIZCA duman testi için sentetik veri (own-work). Gerçek eğitim verisi değildir."""
    rnd = random.Random(0)
    docs = []
    for i in range(300):
        a, b = rnd.randint(2, 99), rnd.randint(2, 99)
        docs.append({"category": "english_math", "source": "smoke", "file": f"e{i}",
                     "text": f"Problem {i}: Compute {a} + {b} and {a} * {b}. Solution: {a}+{b}={a+b}, and {a}*{b}={a*b}. "
                             f"Hence by direct computation the sum is {a+b} while the product equals {a*b}. QED {i}\n" * 2})
        docs.append({"category": "formal", "source": "smoke", "file": f"f{i}",
                     "text": f"theorem add_zero_{i} (n : Nat) : n + 0 = n := by\n  induction n with\n  | zero => rfl\n  | succ k ih => simp\n"
                             f"lemma sq_nonneg_{i} (x : Int) : 0 <= x * x := by\n  nlinarith [sq_nonneg x]  -- case {i}\n"})
        docs.append({"category": "cs_theory", "source": "smoke", "file": f"c{i}",
                     "text": f"Algorithm {i}: binary search runs in O(log n) time because each step halves the interval of size n. "
                             f"Therefore T(n) = T(n/2) + O(1) = O(log n). Variant {i} uses {a} probes.\n" * 2})
        docs.append({"category": "turkish_math", "source": "smoke", "file": f"t{i}",
                     "text": f"Soru {i}: {a} ile {b} sayılarının toplamını bulunuz. Çözüm: {a} + {b} = {a+b} olur. "
                             f"Dolayısıyla toplam {a+b} değerine eşittir. Örnek {i}.\n" * 2})
    for d in docs:
        d["license"] = "own-work"
    return docs


# ------------------------------------------------------------------ helpers
def split_of(text: str, val_frac: float, test_frac: float) -> str:
    h = int(exact_key(text)[:8], 16) / 0xFFFFFFFF
    return "test" if h < test_frac else "val" if h < test_frac + val_frac else "train"


def load_benchmark_texts(roots: List[str]) -> List[str]:
    texts: List[str] = []
    for root in roots:
        root = root if os.path.isabs(root) else os.path.join(PROJECT_ROOT, root)
        bdir = os.path.join(root, "_benchmarks")
        if not os.path.isdir(bdir):
            continue
        for dp, _, fns in os.walk(bdir):
            for fn in fns:
                if fn.endswith(".jsonl"):
                    with open(os.path.join(dp, fn), encoding="utf-8") as f:
                        for line in f:
                            try:
                                rec = json.loads(line)
                                texts.extend(v for v in rec.values() if isinstance(v, str)) if isinstance(rec, dict) else None
                            except Exception:
                                pass
    try:
        from evaluation.custom_math_eval import build_custom_eval
        for q in build_custom_eval():
            texts.append(q["question"] + " " + str(q["answer"]))
    except Exception:
        pass
    return texts


def select_by_mix(docs: List[Dict], targets: Dict[str, float], budget_tokens: float, bytes_per_token: float,
                  seed: int) -> Tuple[List[Dict], Dict]:
    """Token bazlı karışım: hedef oran * bütçe; kıt kategoriler tamamen alınır, kalan bütçe fazlası olanlara dağıtılır."""
    by_cat: Dict[str, List[Dict]] = defaultdict(list)
    for d in docs:
        by_cat[d["category"]].append(d)
    rng = random.Random(seed)
    avail = {c: sum(len(d["text"].encode("utf-8")) for d in by_cat[c]) / bytes_per_token for c in targets}
    take = {c: min(avail[c], targets[c] * budget_tokens) for c in targets}
    left = budget_tokens - sum(take.values())
    while left > 1 and any(avail[c] - take[c] > 1 for c in targets):
        spare = {c: avail[c] - take[c] for c in targets if avail[c] - take[c] > 1}
        tot = sum(spare.values())
        for c, s in spare.items():
            take[c] += min(s, left * s / tot)
        new_left = budget_tokens - sum(take.values())
        if abs(new_left - left) < 1:
            break
        left = new_left
    chosen: List[Dict] = []
    for c in targets:
        lst = by_cat[c][:]
        rng.shuffle(lst)
        acc = 0.0
        for d in lst:
            if acc >= take[c]:
                break
            chosen.append(d)
            acc += len(d["text"].encode("utf-8")) / bytes_per_token
    return chosen, {"available_est_tokens": avail, "target_est_tokens": take}


# --------------------------------------------------------------------- main
def prepare(smoke: bool = False, budget_tokens: Optional[float] = None, workers: int = 0,
            config_path: Optional[str] = None, force: bool = False, work_dir: Optional[str] = None,
            vocab_size: Optional[int] = None, tokenizer_dir: Optional[str] = None) -> Dict:
    cfg = load_data_config(config_path)
    work = work_dir or cfg["work_dir"]
    work = work if os.path.isabs(work) else os.path.join(PROJECT_ROOT, work)
    os.makedirs(work, exist_ok=True)
    vocab = vocab_size or cfg["vocab_size"]
    categories = list(cfg["mix_targets"])
    report: Dict = {"smoke": smoke, "sources_used": [], "sources_excluded": [], "counters": {}}

    # 1) ingestion + lisans
    counters: Counter = Counter()
    if smoke:
        docs = smoke_docs()
        report["sources_used"].append({"name": "smoke", "license": "own-work", "category": "mixed", "n_docs": len(docs)})
        fingerprint_src = "smoke"
    else:
        sources = discover_sources(cfg["data_roots"])
        if not sources:
            raise DataError(
                "Hiç veri kaynağı bulunamadı (source.json içeren klasör yok).\n"
                f"Aranan kökler: {cfg['data_roots']}\n"
                "Her kaynak için bir klasör oluşturun: <kök>/<kaynak_adı>/ içinde metin dosyaları + source.json "
                '({"license": "Apache-2.0", "category": "formal|english_math|cs_theory|turkish_math"}). '
                "Ayrıntı: README.md → 'Veri sağlama'. Eğitim BOŞ VERİYLE başlatılmaz.")
        docs = []
        fp = hashlib.sha1()
        for src in sources:
            why = check_license(src, cfg["allowed_licenses"], categories)
            if why:
                report["sources_excluded"].append({"name": src.get("name"), "path": src.get("path"), "reason": why})
                continue
            sd = read_source(src, counters)
            for dp, dn, fns in os.walk(src["path"]):
                dn.sort()
                for fn in sorted(fns):
                    fp.update(f"{dp}/{fn}:{os.path.getsize(os.path.join(dp, fn))}".encode())
            report["sources_used"].append({"name": src["name"], "license": src["license"], "category": src["category"],
                                           "origin": src.get("origin", ""), "n_docs": len(sd)})
            for d in sd:
                d["license"] = src["license"]
            docs.extend(sd)
        fingerprint_src = fp.hexdigest()
    reuse_tok = ALTUNTokenizer.load(tokenizer_dir) if tokenizer_dir else None
    if reuse_tok is not None:
        vocab = reuse_tok.vocab_size
    fingerprint = hashlib.sha1((fingerprint_src + json.dumps(cfg, sort_keys=True, default=str) + str(budget_tokens)
                                + str(vocab) + str(smoke) + (hashlib.sha1(json.dumps(reuse_tok.merges).encode()).hexdigest() if reuse_tok else "")).encode()).hexdigest()
    meta_path = os.path.join(work, "meta.json")
    if not force and os.path.isfile(meta_path):
        with open(meta_path, encoding="utf-8") as f:
            old = json.load(f)
        if old.get("fingerprint") == fingerprint and all(os.path.isfile(os.path.join(work, b)) for b in ("train.bin", "val.bin", "test.bin")):
            print("[data] Aynı girdi ve ayarlarla işlenmiş veri zaten var, yeniden işlenmiyor.")
            return old
    if not docs:
        raise DataError("Kullanılabilir (lisansı izinli) hiç belge yok. Dışlanan kaynaklar: "
                        + json.dumps(report["sources_excluded"], ensure_ascii=False))
    report["counters"]["raw_docs"] = len(docs)

    # 2) temizleme + kalite
    rejected: Counter = Counter()
    clean: List[Dict] = []
    for d in docs:
        t = clean_text(d["text"])
        for pi, piece in enumerate(split_long(t) if len(t) > 400_000 else [t]):
            ok, why = quality_check(piece, cfg["min_doc_chars"], cfg["max_doc_chars"])
            if ok:
                clean.append({**d, "text": piece, "file": f"{d.get('file', '')}#{pi}" if pi else d.get("file", "")})
            else:
                rejected[why] += 1
    report["counters"]["rejected_by_reason"] = dict(rejected)
    report["counters"].update(dict(counters))
    report["length_outliers"] = outlier_stats([len(d["text"]) for d in clean])

    # 3) dedup + dekontaminasyon
    clean, dstats = deduplicate(clean, near=cfg["near_dup_enabled"], threshold=cfg["near_dup_threshold"], seed=cfg["seed"],
                                skip_near_categories=cfg.get("near_dup_skip_categories", []))
    report["dedup"] = dstats
    ev = load_benchmark_texts(cfg["data_roots"])
    clean, n_contam = decontaminate(clean, build_eval_ngrams(ev, cfg["decontamination_ngram"]), cfg["decontamination_ngram"])
    report["decontamination"] = {"eval_texts": len(ev), "removed_docs": n_contam}
    if not clean:
        raise DataError("Temizleme/dedup sonrası belge kalmadı.")

    # 4) split (belge hash'i ile, sızıntısız)
    splits: Dict[str, List[Dict]] = {"train": [], "val": [], "test": []}
    for d in clean:
        splits[split_of(d["text"], cfg["val_fraction"], cfg["test_fraction"])].append(d)
    if smoke or len(clean) < 200:  # küçük korpusta boş val/test kalmasın
        order = sorted(clean, key=lambda d: exact_key(d["text"]))
        n = max(2, len(order) // 20)
        splits = {"test": order[:n], "val": order[n:2 * n], "train": order[2 * n:]}
    if not splits["train"] or not splits["val"]:
        raise DataError("Train/val ayrımı boş kaldı; daha fazla veri gerekli.")

    # 5) tokenizer: verilmişse AYNEN kullan (önceki modelle uyum), yoksa yalnızca train belgeleriyle eğit
    tok_dir = os.path.join(work, "tokenizer")
    rng = random.Random(cfg["seed"])
    tr = splits["train"][:]
    rng.shuffle(tr)
    sample, chars = [], 0
    for d in tr:
        if chars >= cfg["tokenizer_train_max_chars"]:
            break
        sample.append(d["text"])
        chars += len(d["text"])
    if reuse_tok is not None:
        tok = reuse_tok
        print(f"[data] Mevcut tokenizer kullanılıyor (vocab={tok.vocab_size}); yeniden eğitilmiyor.")
    else:
        print(f"[data] Tokenizer eğitiliyor ({len(sample)} belge, {chars/1e6:.1f}M karakter, vocab={vocab})")
        tok = ALTUNTokenizer.train(sample, vocab_size=vocab, min_frequency=2, verbose=True)
    tok.save(tok_dir)
    bpt = max(1.0, tok.stats(sample[:200])["bytes_per_token"])
    report["tokenizer"] = {"vocab_size": tok.vocab_size, "bytes_per_token_est": bpt, "reused": reuse_tok is not None,
                           "stats_on_train_sample": tok.stats(sample[:500])}

    # 6) karışım seçimi (token bazlı)
    if budget_tokens and not smoke:
        splits["train"], mix_info = select_by_mix(splits["train"], cfg["mix_targets"], budget_tokens, bpt, cfg["seed"])
        report["mix_selection"] = mix_info

    # 7) tokenization + paketleme (belgeler bos/eos ile art arda; pencereler eğitimde rastgele kesilir)
    meta_tokens: Dict[str, int] = {}
    cat_tokens: Counter = Counter()
    for name in ("train", "val", "test"):
        docs_s = splits[name]
        arrays = tokenize_texts(tok_dir, [d["text"] for d in docs_s], workers)
        meta_tokens[name] = write_bin(os.path.join(work, f"{name}.bin"), arrays)
        if name == "train":
            for d, a in zip(docs_s, arrays):
                cat_tokens[d["category"]] += int(a.shape[0])
    total_train = meta_tokens["train"]
    mix_actual = {c: cat_tokens[c] / total_train for c in categories}
    report.update({"tokens": meta_tokens, "docs": {k: len(v) for k, v in splits.items()},
                   "train_mix_actual": mix_actual, "train_mix_target": cfg["mix_targets"]})

    # 8) lisans/provenance kaydı + rapor
    prov = [{"source": s["name"], "license": s["license"], "category": s["category"], "n_docs": s["n_docs"]}
            for s in report["sources_used"]]
    with open(os.path.join(work, "provenance.json"), "w", encoding="utf-8") as f:
        json.dump(prov, f, indent=2, ensure_ascii=False)
    meta = {"fingerprint": fingerprint, "smoke": smoke, "vocab_size": tok.vocab_size, "tokens": meta_tokens,
            "train_mix_actual": mix_actual, "tokenizer_hash": hashlib.sha1(json.dumps(tok.merges).encode()).hexdigest(),
            "min_total_tokens_ok": smoke or total_train >= cfg["min_total_tokens"]}
    if not smoke and total_train < cfg["min_total_tokens"]:
        meta["warning"] = f"Train token sayısı ({total_train}) min_total_tokens ({cfg['min_total_tokens']}) altında"
    with open(os.path.join(work, "data_quality_report.json"), "w", encoding="utf-8") as f:
        json.dump(report, f, indent=2, ensure_ascii=False, default=str)
    with open(os.path.join(work, "data_quality_report.md"), "w", encoding="utf-8") as f:
        f.write(render_report(report, meta))
    with open(meta_path, "w", encoding="utf-8") as f:
        json.dump(meta, f, indent=2)
    print(f"[data] Tamam. train={meta_tokens['train']:,} val={meta_tokens['val']:,} test={meta_tokens['test']:,} token")
    return meta


def render_report(r: Dict, meta: Dict) -> str:
    L = ["# ALTUN 1 Veri Kalite Raporu", ""]
    if r["smoke"]:
        L += ["> **SMOKE VERİSİ**: sentetik, yalnızca duman testi içindir. Gerçek eğitim bu veriyle yapılmamalıdır.", ""]
    L += ["## Kaynaklar (kullanılan)", "", "| Kaynak | Lisans | Kategori | Belge |", "|---|---|---|---|"]
    L += [f"| {s['name']} | {s['license']} | {s['category']} | {s['n_docs']} |" for s in r["sources_used"]]
    L += ["", "## Dışlanan kaynaklar", ""]
    L += [f"- {s['name']}: {s['reason']}" for s in r["sources_excluded"]] or ["- (yok)"]
    L += ["", "## Temizleme", "", f"- Ham belge: {r['counters'].get('raw_docs')}",
          f"- Red nedenleri: {r['counters'].get('rejected_by_reason')}",
          f"- Bozuk jsonl satırı / okunamayan dosya: {r['counters'].get('corrupt_jsonl_lines', 0)} / "
          f"{r['counters'].get('undecodable_files', 0) + r['counters'].get('unreadable_files', 0)}",
          f"- Uzunluk aykırı değerleri: {r['length_outliers']}",
          f"- Dedup: {r['dedup']}", f"- Dekontaminasyon: {r['decontamination']}",
          "", "## Token istatistikleri", "", f"- Belgeler: {r['docs']}", f"- Tokenlar: {r['tokens']}",
          f"- Tokenizer: {r['tokenizer']['stats_on_train_sample']}", "", "## Karışım (train, token bazlı)", "",
          "| Kategori | Hedef | Gerçekleşen |", "|---|---|---|"]
    L += [f"| {c} | {r['train_mix_target'][c]:.2%} | {r['train_mix_actual'][c]:.2%} |" for c in r["train_mix_target"]]
    if meta.get("warning"):
        L += ["", f"**UYARI:** {meta['warning']}"]
    return "\n".join(L) + "\n"


def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("--smoke", action="store_true")
    ap.add_argument("--budget-tokens", type=float, default=None)
    ap.add_argument("--workers", type=int, default=0)
    ap.add_argument("--force", action="store_true")
    ap.add_argument("--work-dir", default=None)
    ap.add_argument("--vocab-size", type=int, default=None)
    ap.add_argument("--tokenizer-dir", default=None, help="Mevcut tokenizer klasörü (yeniden eğitme)")
    a = ap.parse_args()
    try:
        prepare(a.smoke, a.budget_tokens, a.workers, force=a.force, work_dir=a.work_dir, vocab_size=a.vocab_size, tokenizer_dir=a.tokenizer_dir)
    except DataError as e:
        print(f"\n[VERİ HATASI] {e}", file=sys.stderr)
        sys.exit(2)


if __name__ == "__main__":
    main()
