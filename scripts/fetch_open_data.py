"""Otomatik açık-lisanslı veri toplama (Kaggle'da İnternet açıkken). Elle veri sağlamaya gerek bırakmaz.

Kaynaklar (hepsi lisansıyla kaydedilir; izinli listede değilse veri hattı dışlar):
  formal       : Mathlib4 (Apache-2.0), Lean4 core (Apache-2.0), Metamath set.mm (CC0-1.0)
  cs_theory    : TheAlgorithms/Python (MIT), İngilizce Wikipedia TCS kategorileri (CC-BY-SA-4.0)
  math_qa       : NuminaMath-CoT (Apache-2.0), MetaMathQA (MIT), GSM8K train (MIT), MATH train (MIT)
  synthetic_math: kodla üretilen adım adım problemler (own-work)
  english_math : Open Logic Project (CC-BY-4.0), HoTT book (CC-BY-SA-4.0),
                 İngilizce Wikipedia matematik kategorileri (CC-BY-SA-4.0)
  turkish_math : Türkçe Wikipedia matematik/TCS kategorileri (CC-BY-SA-4.0)
Her kaynak bağımsızdır: biri başarısız olursa diğerleri devam eder; sonuç reports/fetch_report.json'a yazılır.
Kullanım: python -m scripts.fetch_open_data [--only a,b] [--force]
"""
from __future__ import annotations

import argparse
import io
import json
import os
import re
import shutil
import sys
import time
import urllib.parse
import urllib.request
import zipfile
from typing import Callable, Dict, List, Optional, Set

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
from altun.config import PROJECT_ROOT  # noqa: E402

UA = "ALTUN-1-research-fetch/1.0 (erlic AI; contact: erlic.research@gmail.com)"

GITHUB = {
    "mathlib4": ("https://github.com/leanprover-community/mathlib4/archive/refs/heads/master.zip", {".lean"}, "Mathlib/",
                 "Apache-2.0", "formal", "https://github.com/leanprover-community/mathlib4"),
    "lean4_core": ("https://github.com/leanprover/lean4/archive/refs/heads/master.zip", {".lean"}, "src/",
                   "Apache-2.0", "formal", "https://github.com/leanprover/lean4"),
    "metamath_setmm": ("https://github.com/metamath/set.mm/archive/refs/heads/develop.zip", {".mm"}, "set.mm",
                       "CC0-1.0", "formal", "https://github.com/metamath/set.mm"),
    "thealgorithms_python": ("https://github.com/TheAlgorithms/Python/archive/refs/heads/master.zip", {".py"}, "",
                             "MIT", "cs_theory", "https://github.com/TheAlgorithms/Python"),
    "open_logic_project": ("https://github.com/OpenLogicProject/OLP/archive/refs/heads/master.zip", {".tex"}, "",
                           "CC-BY-4.0", "english_math", "https://github.com/OpenLogicProject/OLP"),
    "hott_book": ("https://github.com/HoTT/book/archive/refs/heads/master.zip", {".tex"}, "",
                  "CC-BY-SA-4.0", "english_math", "https://github.com/HoTT/book"),
}

WIKI = {  # ad: (dil, kategori seed'leri, veri kategorisi, sayfa üst sınırı)
    "wikipedia_en_math": ("en", ["Category:Number theory", "Category:Algebra", "Category:Combinatorics", "Category:Mathematical logic",
                                 "Category:Graph theory", "Category:Calculus", "Category:Linear algebra", "Category:Mathematical proofs",
                                 "Category:Theorems in number theory", "Category:Theorems in algebra", "Category:Probability theory"],
                          "english_math", 6000),
    "wikipedia_en_cs_theory": ("en", ["Category:Theoretical computer science", "Category:Computational complexity theory",
                                      "Category:Analysis of algorithms", "Category:Graph algorithms", "Category:Sorting algorithms",
                                      "Category:Complexity classes", "Category:Data structures", "Category:Automata theory"],
                               "cs_theory", 4000),
    "wikipedia_tr_math": ("tr", ["Kategori:Matematik", "Kategori:Sayı teorisi", "Kategori:Cebir", "Kategori:Kombinatorik",
                                 "Kategori:Matematiksel mantık", "Kategori:Çizge teorisi", "Kategori:Analiz", "Kategori:Lineer cebir",
                                 "Kategori:Teorik bilgisayar bilimi", "Kategori:Algoritmalar", "Kategori:Olasılık"],
                         "turkish_math", 6000),
}
# HF veri setleri: ad: (repo, [config'ler], split, soru kolonu, cevap kolonu, lisans, kategori, max satır)
HF_SETS = {
    "numinamath_cot": ("AI-MO/NuminaMath-CoT", ["default"], "train", "problem", "solution", "Apache-2.0", "math_qa", 1_000_000),
    "metamathqa": ("meta-math/MetaMathQA", ["default"], "train", "query", "response", "MIT", "math_qa", 300_000),
    "gsm8k_train": ("openai/gsm8k", ["main"], "train", "question", "answer", "MIT", "math_qa", 8_000),
    "math_train": ("EleutherAI/hendrycks_math", ["algebra", "counting_and_probability", "geometry", "intermediate_algebra",
                                                  "number_theory", "prealgebra", "precalculus"], "train", "problem", "solution",
                   "MIT", "math_qa", 8_000),
}
SYNTHETIC_EXAMPLES = 3_000_000
SYNTHETIC_SEED = 20261122   # run 2'den (777) farklı: aynı örnekler tekrar görülmez
PACK = 16  # satırları belgelere paketle (dedup/temizleme hızı için)
ALL_NAMES = ["benchmarks"] + list(GITHUB) + list(HF_SETS) + ["synthetic_math"] + list(WIKI)


# ------------------------------------------------------------------- HTTP
def http(url: str, data: Optional[bytes] = None, timeout: int = 120, retries: int = 4) -> bytes:
    last: Optional[Exception] = None
    for i in range(retries):
        try:
            req = urllib.request.Request(url, data=data, headers={"User-Agent": UA})
            with urllib.request.urlopen(req, timeout=timeout) as r:
                return r.read()
        except Exception as e:  # noqa: BLE001
            last = e
            time.sleep(2 ** i)
    raise RuntimeError(f"{url[:90]}: {type(last).__name__}: {last}")


def write_source_json(dest: str, name: str, lic: str, cat: str, origin: str, **extra) -> None:
    with open(os.path.join(dest, "source.json"), "w", encoding="utf-8") as f:
        json.dump({"name": name, "license": lic, "category": cat, "origin": origin,
                   "retrieved": time.strftime("%Y-%m-%d"), **extra}, f, indent=2)


# ----------------------------------------------------------------- GitHub
def fetch_github(name: str, dest_root: str, getter: Callable[[str], bytes] = http) -> Dict:
    url, exts, prefix, lic, cat, origin = GITHUB[name]
    dest = os.path.join(dest_root, name)
    data = getter(url)
    shutil.rmtree(dest, ignore_errors=True)
    os.makedirs(dest)
    n = size = 0
    with zipfile.ZipFile(io.BytesIO(data)) as z:
        for info in z.infolist():
            rel = info.filename.split("/", 1)[-1]
            if info.is_dir() or os.path.splitext(rel)[1].lower() not in exts or not rel.startswith(prefix):
                continue
            if "/.lake/" in info.filename or info.file_size > 60_000_000:
                continue
            with z.open(info) as src, open(os.path.join(dest, rel.replace("/", "__")), "wb") as f:
                shutil.copyfileobj(src, f)
            n += 1
            size += info.file_size
    if n == 0:
        shutil.rmtree(dest, ignore_errors=True)
        raise RuntimeError("hiç dosya çıkarılamadı (depo yapısı değişmiş olabilir)")
    write_source_json(dest, name, lic, cat, origin, files=n)
    return {"files": n, "mb": round(size / 1e6, 1)}


# --------------------------------------------------------------- HF + sentetik
def _pack_write(out, texts) -> int:
    """Soru-çözüm metinlerini PACK'lı belgeler halinde jsonl'a yazar. Döner: yazılan örnek sayısı."""
    buf, n = [], 0
    for t in texts:
        buf.append(t)
        n += 1
        if len(buf) == PACK:
            out.write(json.dumps({"text": "\n\n".join(buf)}, ensure_ascii=False) + "\n")
            buf = []
    if buf:
        out.write(json.dumps({"text": "\n\n".join(buf)}, ensure_ascii=False) + "\n")
    return n


def fetch_hf(name: str, dest_root: str, getter: Callable[[str], bytes] = http) -> Dict:
    repo, configs, split, qcol, acol, lic, cat, max_rows = HF_SETS[name]
    try:
        import pyarrow.parquet as pq
    except ImportError:
        raise RuntimeError("pyarrow yok (Kaggle imajında normalde vardır)")
    dest = os.path.join(dest_root, name)
    shutil.rmtree(dest, ignore_errors=True)
    os.makedirs(dest)
    seen: Set[int] = set()
    per_cfg = max(1, max_rows // len(configs))

    def rows_iter():
        for cfg in configs:
            urls = json.loads(getter(f"https://huggingface.co/api/datasets/{repo}/parquet/{cfg}/{split}"))
            if not isinstance(urls, list) or not urls:
                raise RuntimeError(f"{repo}/{cfg}: parquet listesi alınamadı")
            got = 0
            for u in urls:
                table = pq.read_table(io.BytesIO(getter(u)), columns=[qcol, acol])
                for q, a in zip(table.column(qcol).to_pylist(), table.column(acol).to_pylist()):
                    if not q or not a:
                        continue
                    h = hash((q, a))
                    if h in seen:
                        continue
                    seen.add(h)
                    yield f"Problem: {q}\nSolution: {a}"
                    got += 1
                    if got >= per_cfg:
                        break
                if got >= per_cfg:
                    break

    with open(os.path.join(dest, "qa.jsonl"), "w", encoding="utf-8") as out:
        rows = _pack_write(out, rows_iter())
    if rows == 0:
        shutil.rmtree(dest, ignore_errors=True)
        raise RuntimeError("hiç satır alınamadı")
    write_source_json(dest, name, lic, cat, f"https://huggingface.co/datasets/{repo}", rows=rows)
    return {"rows": rows}


def fetch_synthetic(dest_root: str, n_examples: int = SYNTHETIC_EXAMPLES) -> Dict:
    """Kodla üretilen doğrulanabilir matematik (ALTUN değerlendirme soruları hariç tutulur)."""
    from data.synthetic_math import generate_documents
    from evaluation.custom_math_eval import build_custom_eval
    dest = os.path.join(dest_root, "synthetic_math")
    shutil.rmtree(dest, ignore_errors=True)
    os.makedirs(dest)
    from evaluation.turkish_eval import build_turkish_eval
    ex = {q["question"] for q in build_custom_eval()} | {q["question"] for q in build_turkish_eval()}
    n_docs = 0
    with open(os.path.join(dest, "synthetic.jsonl"), "w", encoding="utf-8") as out:
        for doc in generate_documents(n_examples, per_doc=24, seed=SYNTHETIC_SEED, exclude_questions=ex):
            out.write(json.dumps({"text": doc}, ensure_ascii=False) + "\n")
            n_docs += 1
    write_source_json(dest, "synthetic_math", "own-work", "synthetic_math", f"kodla üretildi (data/synthetic_math.py, seed={SYNTHETIC_SEED})",
                      examples=n_examples, docs=n_docs)
    return {"examples": n_examples, "docs": n_docs}


# ------------------------------------------- bağımsız benchmark test setleri
def fetch_benchmarks(dest_root: str, getter: Callable[[str], bytes] = http, n_math: int = 500) -> Dict:
    """GSM8K ve MATH TEST bölümleri (eğitime GİRMEZ; dekontaminasyon ve bağımsız değerlendirme için)."""
    import pyarrow.parquet as pq
    bdir = os.path.join(dest_root, "_benchmarks")
    os.makedirs(bdir, exist_ok=True)

    def table(repo, cfg, split, cols):
        urls = json.loads(getter(f"https://huggingface.co/api/datasets/{repo}/parquet/{cfg}/{split}"))
        out = []
        for u in urls:
            t = pq.read_table(io.BytesIO(getter(u)), columns=cols)
            out += [dict(zip(cols, row)) for row in zip(*[t.column(c).to_pylist() for c in cols])]
        return out

    gsm = table("openai/gsm8k", "main", "test", ["question", "answer"])
    cfgs = ["algebra", "counting_and_probability", "geometry", "intermediate_algebra", "number_theory", "prealgebra", "precalculus"]
    per = max(1, n_math // len(cfgs))
    math_rows = []
    for c in cfgs:
        math_rows += table("EleutherAI/hendrycks_math", c, "test", ["problem", "solution"])[:per]
    if not gsm or not math_rows:
        raise RuntimeError("benchmark test setleri boş geldi")
    for fn, rows in (("gsm8k_test.jsonl", gsm), ("math_test.jsonl", math_rows)):
        with open(os.path.join(bdir, fn), "w", encoding="utf-8") as f:
            for row in rows:
                f.write(json.dumps(row, ensure_ascii=False) + "\n")
    return {"gsm8k_test": len(gsm), "math_test": len(math_rows)}


# -------------------------------------------------------------- Wikipedia
_SECTION_CUT = re.compile(r"^==\s*(References|See also|External links|Notes|Further reading|Kaynakça|Dış bağlantılar|Notlar|Ayrıca bakınız|Konuyla ilgili yayınlar)\s*==\s*$",
                          re.I | re.M)
_MATH = re.compile(r"<math(?P<attr>[^>]*)>(?P<body>.*?)</math>", re.S)


def clean_wikitext(text: str) -> str:
    """Wikitext → düz metin; <math> LaTeX'i $...$ olarak korunur."""
    maths: List[str] = []

    def stash(m):
        body = m.group("body").strip()
        maths.append(f"$${body}$$" if "block" in m.group("attr") else f"${body}$")
        return f"\x00M{len(maths) - 1}\x00"

    text = _MATH.sub(stash, text)
    cut = _SECTION_CUT.search(text)
    if cut:
        text = text[:cut.start()]
    text = re.sub(r"<!--.*?-->", "", text, flags=re.S)
    text = re.sub(r"<ref[^>/]*/>|<ref[^>]*>.*?</ref>", "", text, flags=re.S)
    text = re.sub(r"<gallery.*?</gallery>", "", text, flags=re.S)
    for _ in range(6):  # iç içe şablonlar
        new = re.sub(r"\{\{[^{}]*\}\}", "", text)
        if new == text:
            break
        text = new
    text = re.sub(r"\{\|.*?\|\}", "", text, flags=re.S)
    text = re.sub(r"\[\[(?:File|Image|Category|Dosya|Resim|Kategori):[^\]]*\]\]", "", text, flags=re.I)
    text = re.sub(r"\[\[[^\]|]*\|([^\]]*)\]\]", r"\1", text)
    text = re.sub(r"\[\[([^\]]*)\]\]", r"\1", text)
    text = re.sub(r"\[https?://\S+ ([^\]]*)\]|\[https?://\S+\]", r"\1", text)
    text = re.sub(r"'{2,5}", "", text)
    text = re.sub(r"^=+\s*(.*?)\s*=+\s*$", r"\1", text, flags=re.M)
    text = re.sub(r"<[^>]+>", "", text)
    text = re.sub(r"\n{3,}", "\n\n", text)
    text = re.sub(r"\x00M(\d+)\x00", lambda m: maths[int(m.group(1))], text)
    return text.strip()


def wiki_api(lang: str, params: Dict, post: bool = False, getter: Callable = http) -> Dict:
    base = f"https://{lang}.wikipedia.org/w/api.php"
    q = urllib.parse.urlencode({**params, "format": "json", "formatversion": "2"})
    raw = getter(base, q.encode()) if post else getter(base + "?" + q)
    return json.loads(raw)


def crawl_categories(lang: str, seeds: List[str], max_pages: int, depth: int = 2, getter: Callable = http,
                     sleep: float = 0.2) -> List[str]:
    pages: Set[str] = set()
    seen_cats: Set[str] = set()
    frontier = [(c, 0) for c in seeds]
    while frontier and len(pages) < max_pages:
        cat, d = frontier.pop(0)
        if cat in seen_cats:
            continue
        seen_cats.add(cat)
        cont: Dict = {}
        while True:
            try:
                r = wiki_api(lang, {"action": "query", "list": "categorymembers", "cmtitle": cat, "cmlimit": "500",
                                    "cmtype": "page|subcat", **cont}, getter=getter)
            except RuntimeError:
                break  # kategori yok / erişilemiyor → atla
            for m in r.get("query", {}).get("categorymembers", []):
                title = m["title"]
                if m.get("ns") == 14 and d < depth:
                    frontier.append((title, d + 1))
                elif m.get("ns") == 0 and not title.lower().startswith(("list of", "liste")):
                    pages.add(title)
            if "continue" in r and len(pages) < max_pages:
                cont = r["continue"]
                time.sleep(sleep)
            else:
                break
        time.sleep(sleep)
    return sorted(pages)[:max_pages]


def fetch_wikipedia(name: str, dest_root: str, getter: Callable = http, sleep: float = 0.3) -> Dict:
    lang, seeds, cat, max_pages = WIKI[name]
    titles = crawl_categories(lang, seeds, max_pages, getter=getter, sleep=sleep)
    if not titles:
        raise RuntimeError("kategorilerden hiç sayfa alınamadı (kategori adları/erişim)")
    dest = os.path.join(dest_root, name)
    shutil.rmtree(dest, ignore_errors=True)
    os.makedirs(dest)
    n = chars = 0
    with open(os.path.join(dest, "articles.jsonl"), "w", encoding="utf-8") as out:
        for i in range(0, len(titles), 40):
            batch = titles[i:i + 40]
            try:
                r = wiki_api(lang, {"action": "query", "prop": "revisions", "rvprop": "content", "rvslots": "main",
                                    "titles": "|".join(batch)}, post=True, getter=getter)
            except RuntimeError:
                continue
            for p in r.get("query", {}).get("pages", []):
                revs = p.get("revisions")
                if not revs:
                    continue
                txt = clean_wikitext(revs[0]["slots"]["main"]["content"])
                if len(txt) >= 400 and not txt.lower().startswith(("#redirect", "#yönlendirme")):
                    out.write(json.dumps({"text": f"{p['title']}\n\n{txt}"}, ensure_ascii=False) + "\n")
                    n += 1
                    chars += len(txt)
            time.sleep(sleep)
    if n == 0:
        shutil.rmtree(dest, ignore_errors=True)
        raise RuntimeError("hiç makale metni alınamadı")
    write_source_json(dest, name, "CC-BY-SA-4.0", cat, f"https://{lang}.wikipedia.org (CC BY-SA 4.0; yazarlar sayfa geçmişinde)", articles=n)
    return {"articles": n, "mb": round(chars / 1e6, 1)}


# ------------------------------------------------------------------- ana
def ensure_data(dest_root: str, only: Optional[List[str]] = None, force: bool = False,
                report_path: Optional[str] = None) -> Dict:
    """Eksik kaynakları indirir (var olanı atlar). En az bir kaynak varsa başarılı sayılır."""
    os.makedirs(dest_root, exist_ok=True)
    report: Dict = {"fetched": {}, "skipped_existing": [], "failed": {}}
    for name in only or ALL_NAMES:
        if name not in ALL_NAMES:
            report["failed"][name] = "bilinmeyen kaynak"
            continue
        if name == "benchmarks" and not force and os.path.isfile(os.path.join(dest_root, "_benchmarks", "gsm8k_test.jsonl")):
            report["skipped_existing"].append(name)
            continue
        if not force and name != "benchmarks" and os.path.isfile(os.path.join(dest_root, name, "source.json")):
            report["skipped_existing"].append(name)
            continue
        print(f"[fetch] {name} ...", flush=True)
        try:
            if name in GITHUB:
                res = fetch_github(name, dest_root)
            elif name in HF_SETS:
                res = fetch_hf(name, dest_root)
            elif name == "synthetic_math":
                res = fetch_synthetic(dest_root)
            elif name == "benchmarks":
                res = fetch_benchmarks(dest_root)
            else:
                res = fetch_wikipedia(name, dest_root)
            report["fetched"][name] = res
            print(f"[fetch] {name} OK {res}", flush=True)
        except Exception as e:  # noqa: BLE001
            report["failed"][name] = f"{type(e).__name__}: {e}"
            print(f"[fetch] {name} BAŞARISIZ: {e}", flush=True)
    report["ok"] = bool(report["fetched"] or report["skipped_existing"])
    if report_path:
        os.makedirs(os.path.dirname(report_path), exist_ok=True)
        with open(report_path, "w", encoding="utf-8") as f:
            json.dump(report, f, indent=2, ensure_ascii=False)
    return report


def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("--only", default="")
    ap.add_argument("--dest", default=os.path.join(PROJECT_ROOT, "raw_data"))
    ap.add_argument("--force", action="store_true")
    a = ap.parse_args()
    rep = ensure_data(a.dest, [x for x in a.only.split(",") if x] or None, a.force)
    print(json.dumps(rep, indent=2, ensure_ascii=False))
    sys.exit(0 if rep["ok"] else 2)


if __name__ == "__main__":
    main()
