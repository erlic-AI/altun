"""Torch gerektirmeyen testler: config, scheduler, veri hattı, lisans kapısı, dekontaminasyon, ALTUN eval seti, benchmark yükleyici."""
import json
import math
import os
import tempfile

import pytest
import yaml

from altun.config import ModelConfig, TrainConfig, load_data_config, load_model_config, load_train_config
from data.dataset import TokenDataset
from data.prepare_data import DataError, prepare
from evaluation.benchmarks import find_benchmark_files, run_benchmark
from evaluation.custom_math_eval import CATEGORIES, build_custom_eval
from evaluation.metrics import accuracy_report, check_answer, extract_boxed, is_degenerate
from training.scheduler import lr_at


def test_scheduler_warmup_and_cosine():
    assert lr_at(0, 0.0, 1e-3, 0.1, 10) == pytest.approx(1e-4)
    assert lr_at(9, 0.0, 1e-3, 0.1, 10) == pytest.approx(1e-3)
    assert lr_at(50, 0.0, 1e-3, 0.1, 10) == pytest.approx(1e-3)
    assert lr_at(50, 1.0, 1e-3, 0.1, 10) == pytest.approx(1e-4)
    assert 1e-4 < lr_at(50, 0.5, 1e-3, 0.1, 10) < 1e-3
    assert lr_at(50, 5.0, 1e-3, 0.1, 10) == pytest.approx(1e-4)


def test_all_configs_load_and_are_consistent():
    d = load_data_config()
    for p in ("altun1-smoke", "altun1-small", "altun1-main"):
        c = load_model_config(preset=p)
        assert c.vocab_size == d["vocab_size"]
    assert load_train_config().duration_hours == 12.0
    with pytest.raises(ValueError):
        ModelConfig.from_dict({"bogus": 1})
    with pytest.raises(ValueError):
        ModelConfig(d_model=100, n_heads=16).validate()
    with pytest.raises(ValueError):
        TrainConfig(duration_hours=0).validate()


def _write_source(root, name, license_, category, docs):
    d = os.path.join(root, name)
    os.makedirs(d)
    if license_ is not None:
        json.dump({"license": license_, "category": category}, open(os.path.join(d, "source.json"), "w"))
    else:
        json.dump({"category": category}, open(os.path.join(d, "source.json"), "w"))
    for i, t in enumerate(docs):
        open(os.path.join(d, f"d{i}.txt"), "w", encoding="utf-8").write(t)


def _docs(prefix, n):
    return [f"{prefix} document {i}: Let n be a natural number. Then n + {i} > n holds because {i} is positive. "
            f"Proof by direct inequality, number {i * 7}, lemma {prefix}{i}. " * 3 for i in range(1, n + 1)]


def test_license_gate_dedup_decontamination_split(tmp_path):
    root = str(tmp_path / "raw")
    os.makedirs(root)
    ok_docs = _docs("alpha", 60)
    _write_source(root, "good_formal", "Apache-2.0", "formal", ok_docs)
    _write_source(root, "gpl_stuff", "GPL-3.0", "formal", _docs("gplzzz", 20))
    _write_source(root, "no_license", None, "english_math", _docs("nolic", 20))
    _write_source(root, "english_ok", "CC-BY-4.0", "english_math", _docs("beta", 40) + [_docs("beta", 40)[0]])  # + kopya
    os.makedirs(os.path.join(root, "_benchmarks"))
    leak = ok_docs[5]  # benchmark ile birebir örtüşen eğitim belgesi
    open(os.path.join(root, "_benchmarks", "gsm8k_test.jsonl"), "w").write(json.dumps({"question": leak, "answer": "#### 1"}) + "\n")
    cfgd = load_data_config()
    cfgd.update({"data_roots": [root], "work_dir": str(tmp_path / "work"), "vocab_size": 400, "min_total_tokens": 100})
    cfg_path = str(tmp_path / "data.yaml")
    yaml.safe_dump(cfgd, open(cfg_path, "w"))
    meta = prepare(config_path=cfg_path, workers=1)
    rep = json.load(open(tmp_path / "work" / "data_quality_report.json"))
    used = {s["name"] for s in rep["sources_used"]}
    excluded = {s["name"]: s["reason"] for s in rep["sources_excluded"]}
    assert used == {"good_formal", "english_ok"}
    assert "GPL-3.0" in excluded["gpl_stuff"] and "lisans" in excluded["no_license"]
    assert rep["dedup"]["exact_dups"] >= 1
    assert rep["decontamination"]["removed_docs"] >= 1
    assert meta["tokens"]["train"] > 0 and meta["tokens"]["val"] > 0 and meta["tokens"]["test"] > 0
    prov = json.load(open(tmp_path / "work" / "provenance.json"))
    assert all(p["license"] in ("Apache-2.0", "CC-BY-4.0") for p in prov)
    # cache: aynı girdiyle yeniden işleme yapılmaz
    assert prepare(config_path=cfg_path, workers=1)["fingerprint"] == meta["fingerprint"]


def test_no_data_stops_with_clear_error(tmp_path):
    cfgd = load_data_config()
    cfgd.update({"data_roots": [str(tmp_path / "empty")], "work_dir": str(tmp_path / "w")})
    p = str(tmp_path / "c.yaml")
    yaml.safe_dump(cfgd, open(p, "w"))
    with pytest.raises(DataError, match="BOŞ VERİYLE"):
        prepare(config_path=p)


def test_only_unlicensed_data_stops(tmp_path):
    root = str(tmp_path / "r")
    os.makedirs(root)
    _write_source(root, "bad", "Proprietary", "formal", _docs("x", 5))
    cfgd = load_data_config()
    cfgd.update({"data_roots": [root], "work_dir": str(tmp_path / "w")})
    p = str(tmp_path / "c.yaml")
    yaml.safe_dump(cfgd, open(p, "w"))
    with pytest.raises(DataError, match="lisansı izinli"):
        prepare(config_path=p)


def test_smoke_pipeline_and_dataset(tmp_path):
    meta = prepare(smoke=True, vocab_size=512, work_dir=str(tmp_path / "d"), workers=1)
    assert meta["smoke"] and meta["tokens"]["train"] > 10000
    ds = TokenDataset(str(tmp_path / "d" / "train.bin"), 32)
    x, y = ds.get_batch(4, 1, 2, 3)
    assert x.shape == (4, 32) and (y[:, :-1] == x[:, 1:]).all() and x.max() < 512


def test_custom_eval_set_is_complete_and_deterministic():
    qs = build_custom_eval()
    assert {q["category"] for q in qs} == set(CATEGORIES)
    assert qs == build_custom_eval()
    assert {q["difficulty"] for q in qs} == {1, 2, 3}
    assert len({q["id"] for q in qs}) == len(qs)
    assert any(q["lang"] == "tr" for q in qs)


def test_answer_checkers_and_metrics():
    assert check_answer("number", "42", "The answer is 42.") and not check_answer("number", "42", "no idea")
    assert check_answer("yesno", "yes", "True") and check_answer("yesno", "no", "Hayır") and not check_answer("yesno", "yes", "maybe")
    assert check_answer("text", "O(log n)", "It is O(log n).")
    assert extract_boxed(r"so \boxed{\frac{1}{2}} done") == r"\frac{1}{2}"
    assert is_degenerate("") and is_degenerate("aaaaaaaaaaaaaaaaaaaa") and not is_degenerate("The sum is 5.")
    rep = accuracy_report([{"category": "A", "correct": True}, {"category": "A", "correct": False}, {"category": "B", "correct": False}])
    assert rep["total_questions"] == 3 and rep["correct"] == 1 and rep["incorrect"] == 2
    assert rep["category_accuracy"]["A"]["accuracy"] == 0.5


def test_benchmarks_never_fabricate(tmp_path):
    os.makedirs(tmp_path / "_benchmarks")
    open(tmp_path / "_benchmarks" / "gsm8k_test.jsonl", "w").write(json.dumps({"question": "2+3?", "answer": "#### 5"}) + "\nbozuk\n")
    f = find_benchmark_files([str(tmp_path)])
    assert run_benchmark("GSM8K", f["GSM8K"], lambda p: "5")["accuracy"] == 1.0
    assert run_benchmark("MATH", f["MATH"], lambda p: "")["status"] == "not_run"
    assert run_benchmark("MiniF2F", None, lambda p: "")["status"] == "not_run"


def test_split_long_documents():
    from data.clean_data import split_long
    text = "\n".join(f"line {i} " + "x" * 50 for i in range(5000))
    parts = split_long(text, 20000)
    assert len(parts) > 5 and all(len(p) <= 20100 for p in parts)
    assert "".join(parts).replace("\n", "") == text.replace("\n", "")
    assert split_long("kısa", 100) == ["kısa"]


# ---------------- otomatik veri toplama (ağ yerine sahte HTTP ile) ----------------
def test_clean_wikitext_keeps_math_and_drops_markup():
    from scripts.fetch_open_data import clean_wikitext
    w = ("{{Infobox x|a=b}}'''Prime''' numbers are [[natural number|natural numbers]] <ref>cite</ref> greater than 1 with "
         "<math>n \\mid p \\Rightarrow n \\in \\{1,p\\}</math>.\n== Proof ==\nSee [[File:a.png|thumb]] {{math|x}} <math display=\"block\">a^2+b^2=c^2</math>\n"
         "== References ==\n* junk")
    c = clean_wikitext(w)
    assert "Prime numbers are natural numbers" in c and "$n \\mid p" in c and "$$a^2+b^2=c^2$$" in c
    assert "cite" not in c and "Infobox" not in c and "junk" not in c and "File:" not in c


def test_wikipedia_fetch_with_mock_http(tmp_path):
    from scripts import fetch_open_data as F

    def fake(url, data=None, **kw):
        q = urllib.parse.parse_qs(urllib.parse.urlparse(url).query) if data is None else urllib.parse.parse_qs(data.decode())
        if q.get("list") == ["categorymembers"]:
            if q["cmtitle"][0] == "Kategori:Matematik":
                return json.dumps({"query": {"categorymembers": [{"title": "Asal sayı", "ns": 0}, {"title": "Kategori:Alt", "ns": 14},
                                                                 {"title": "Liste of x", "ns": 0}]}}).encode()
            return json.dumps({"query": {"categorymembers": [{"title": "Cebir", "ns": 0}]}}).encode()
        pages = [{"title": t, "revisions": [{"slots": {"main": {"content": f"'''{t}''' bir matematik terimidir. " * 30 + "<math>x^2</math>"}}}]}
                 for t in q["titles"][0].split("|")]
        return json.dumps({"query": {"pages": pages}}).encode()

    import urllib.parse
    res = F.fetch_wikipedia("wikipedia_tr_math", str(tmp_path), getter=fake, sleep=0)
    assert res["articles"] == 2
    meta = json.load(open(tmp_path / "wikipedia_tr_math" / "source.json"))
    assert meta["license"] == "CC-BY-SA-4.0" and meta["category"] == "turkish_math"
    rows = [json.loads(l) for l in open(tmp_path / "wikipedia_tr_math" / "articles.jsonl")]
    assert any("Asal sayı" in r["text"] for r in rows) and not any(r["text"].startswith("Liste") for r in rows)


def test_github_fetch_with_mock_zip(tmp_path):
    import io, zipfile
    from scripts import fetch_open_data as F
    buf = io.BytesIO()
    with zipfile.ZipFile(buf, "w") as z:
        z.writestr("mathlib4-master/Mathlib/Algebra/Basic.lean", "theorem foo : 1 = 1 := rfl\n" * 10)
        z.writestr("mathlib4-master/docs/readme.md", "ignored")
    res = F.fetch_github("mathlib4", str(tmp_path), getter=lambda u: buf.getvalue())
    assert res["files"] == 1
    assert json.load(open(tmp_path / "mathlib4" / "source.json"))["license"] == "Apache-2.0"


def test_ensure_data_tolerates_partial_failure(tmp_path):
    from scripts import fetch_open_data as F
    rep = F.ensure_data(str(tmp_path), only=["no_such_source"], report_path=str(tmp_path / "r.json"))
    assert not rep["ok"] and "no_such_source" in rep["failed"]


# ---------------- sentetik veri ----------------
def test_synthetic_math_answers_are_correct():
    import re
    from data.synthetic_math import generate
    seen = set()
    for ex in generate(6000, seed=1):
        seen.add(ex["kind"])
        m = re.findall(r"\\boxed\{([^{}]*)\}", ex["solution"])
        assert m, ex["text"]
        q = ex["question"]
        if ex["kind"] in ("add", "sub", "mul") and ex["lang"] == "en":
            a, b = [int(x) for x in re.findall(r"\d+", q)][:2]
            exp = {"add": a + b, "sub": (a - b) if "minus" in q or "-" in q else None, "mul": a * b}[ex["kind"]]
            if ex["kind"] == "sub":  # 'Subtract b from a' biçimi ters yazılır
                a, b = max(a, b), min(a, b)
                exp = a - b
            assert int(m[-1]) == exp, ex["text"]
        if ex["kind"] == "gcd":
            a, b = [int(x) for x in re.findall(r"\d+", q)][-2:]
            assert int(m[-1]) == math.gcd(a, b)
        if ex["kind"] == "linear" and ex["lang"] == "en":
            assert str(ex["answer"]) == m[-1]
    assert len(seen) >= 25


def test_synthetic_excludes_eval_questions_and_is_deterministic():
    from data.synthetic_math import generate
    ev = {q["question"] for q in build_custom_eval()}
    a = [e["question"] for e in generate(8000, seed=5, exclude_questions=ev)]
    assert not (set(a) & ev)
    assert a == [e["question"] for e in generate(8000, seed=5, exclude_questions=ev)]
    # eval ile aynı kalıpta üretilebilen sorular dışlanıyor mu? (zorla: tüm ürettiklerini exclude et)
    first = {e["question"] for e in generate(200, seed=5)}
    again = [e["question"] for e in generate(200, seed=5, exclude_questions=first)]
    assert not (set(again) & first)


def test_synthetic_documents_pack_examples():
    from data.synthetic_math import generate_documents
    docs = list(generate_documents(100, per_doc=24))
    assert len(docs) == 5 and docs[0].count("Problem:") == 24


def test_fetch_hf_with_fake_pyarrow(tmp_path, monkeypatch):
    import sys, types
    from scripts import fetch_open_data as F

    class Col:
        def __init__(self, v): self.v = v
        def to_pylist(self): return self.v

    class Table:
        def __init__(self, cols): self.cols = cols
        def column(self, n): return Col(self.cols[n])

    pq = types.ModuleType("pyarrow.parquet")
    pq.read_table = lambda buf, columns=None: Table({"problem": [f"p{i}" for i in range(40)] + ["p1", None],
                                                     "solution": [f"s{i}" for i in range(40)] + ["s1", "x"]})
    pa = types.ModuleType("pyarrow")
    pa.parquet = pq
    monkeypatch.setitem(sys.modules, "pyarrow", pa)
    monkeypatch.setitem(sys.modules, "pyarrow.parquet", pq)

    def getter(url, data=None):
        return json.dumps(["https://x/a.parquet"]).encode() if "/parquet/" in url else b"PQ"

    res = F.fetch_hf("numinamath_cot", str(tmp_path), getter=getter)
    assert res["rows"] == 40  # tekrar eden (p1) ve boş satırlar atlandı
    lines = [json.loads(l)["text"] for l in open(tmp_path / "numinamath_cot" / "qa.jsonl")]
    assert len(lines) == 3 and lines[0].count("Problem:") == 16  # 16'lı paketleme
    meta = json.load(open(tmp_path / "numinamath_cot" / "source.json"))
    assert meta["license"] == "Apache-2.0" and meta["category"] == "math_qa"
    res2 = F.fetch_hf("math_train", str(tmp_path), getter=getter)  # çok config'li
    assert res2["rows"] > 0


def test_v2_mix_with_synthetic_and_tokenizer_reuse(tmp_path):
    from altun.tokenizer import ALTUNTokenizer
    from scripts import fetch_open_data as F
    root = str(tmp_path / "raw")
    os.makedirs(root)
    F.fetch_synthetic(root, 6000)
    _write_source(root, "formal_small", "Apache-2.0", "formal", _docs("lean", 40))
    cfgd = load_data_config()
    cfgd.update({"data_roots": [root], "work_dir": str(tmp_path / "w1"), "vocab_size": 400, "min_total_tokens": 100})
    p1 = str(tmp_path / "d1.yaml")
    yaml.safe_dump(cfgd, open(p1, "w"))
    m1 = prepare(config_path=p1, workers=1, budget_tokens=200_000)
    rep = json.load(open(tmp_path / "w1" / "data_quality_report.json"))
    assert rep["dedup"]["near_skipped_docs"] > 0
    assert set(rep["train_mix_actual"]) == set(load_data_config()["mix_targets"])
    assert rep["train_mix_actual"]["synthetic_math"] > 0 and rep["train_mix_actual"]["formal"] > 0
    # aynı tokenizer ile ikinci hazırlık: merges birebir aynı, yeniden eğitilmedi
    cfgd["work_dir"] = str(tmp_path / "w2")
    p2 = str(tmp_path / "d2.yaml")
    yaml.safe_dump(cfgd, open(p2, "w"))
    m2 = prepare(config_path=p2, workers=1, budget_tokens=200_000, tokenizer_dir=str(tmp_path / "w1" / "tokenizer"))
    assert m2["tokenizer_hash"] == m1["tokenizer_hash"]
    assert ALTUNTokenizer.load(str(tmp_path / "w2" / "tokenizer")).merges == ALTUNTokenizer.load(str(tmp_path / "w1" / "tokenizer")).merges
    assert json.load(open(tmp_path / "w2" / "data_quality_report.json"))["tokenizer"]["reused"] is True


def test_final_answer_extraction_and_previous_model_finder(tmp_path, monkeypatch):
    from evaluation.metrics import check_answer
    out = "Step: 12+15 = 27. Answer $\\boxed{27}$\nProblem: 3+4"
    assert check_answer("number", "27", out)
    assert check_answer("number", "27", "12 + 15 = 27")  # boxed yok: son sayı
    assert not check_answer("number", "12", "12 + 15 = 27")
    import scripts.run_training as RT
    monkeypatch.setattr(RT.glob, "glob", lambda pat: [])
    assert RT.find_previous_model() is None  # girdi yoksa bulunamaz


def test_v3_generators_consistent_and_independently_verified():
    import itertools, random, re
    from data import synthetic_math as sm
    new = ["loop_cx", "master", "ds_cost", "kth", "cmp_count", "stack_queue", "dfs_order", "linalg", "calc2", "lean", "proof_blank"]
    assert all(k in sm.GENERATORS and k in sm.WEIGHTS for k in new)
    for k in new:
        for lang in ("en", "tr"):
            r = random.Random(11)
            for _ in range(400):
                q, s, ans = sm.GENERATORS[k](r, lang)
                b = re.findall(r"\\boxed\{(.*)\}", s)
                assert b, (k, s)
                if k != "linalg" or ans not in ("yes", "no"):
                    assert b[-1].replace(" ", "") == str(ans).replace(" ", ""), (k, q, s, ans)
    # bağımsız doğrulama: 3x3 determinant (Leibniz), DFS, kth
    r = random.Random(3)
    checked = 0
    while checked < 30:
        q, s, ans = sm.gen_linalg(r, "en")
        m = re.search(r"3×3 matrix with rows \((.*?)\), \((.*?)\), \((.*?)\)", q)
        if not m:
            continue
        a = [[int(x) for x in g.split(",")] for g in m.groups()]
        det = sum((1 if sum(p[i] > p[j] for i in range(3) for j in range(i + 1, 3)) % 2 == 0 else -1) * a[0][p[0]] * a[1][p[1]] * a[2][p[2]]
                  for p in itertools.permutations(range(3)))
        assert det == ans
        checked += 1
    for _ in range(50):
        q, s, ans = sm.gen_dfs_order(r, "en")
        edges = eval(re.search(r"edges (\[.*?\])\.", q).group(1))
        seen, order, stack = set(), [], [0]
        adj = {i: sorted({b for a, b in edges if a == i} | {a for a, b in edges if b == i}) for i in range(8)}
        def go(u):
            seen.add(u); order.append(u)
            for w in adj[u]:
                if w not in seen: go(w)
        go(0)
        assert ans == " ".join(map(str, order))


def test_v3_turkish_eval_and_exclusion(tmp_path):
    from evaluation.turkish_eval import build_turkish_eval
    from evaluation.metrics import check_answer
    from data.synthetic_math import generate
    qs = build_turkish_eval()
    assert len(qs) >= 30 and len({q["question"] for q in qs}) == len(qs)
    assert check_answer("number", "72", "3·24 = 72. $\\boxed{72}$")
    ex = {q["question"] for q in qs}
    assert not any(e["question"] in ex for e in generate(3000, seed=1, exclude_questions=ex))
    tr = sum(e["lang"] == "tr" for e in generate(3000, seed=2))
    assert 0.2 < tr / 3000 < 0.4


def test_fetch_benchmarks_with_fake_pyarrow(tmp_path, monkeypatch):
    import sys, types
    from scripts import fetch_open_data as F

    class Col:
        def __init__(self, v): self.v = v
        def to_pylist(self): return self.v

    class Table:
        def __init__(self, c): self.c = c
        def column(self, n): return Col(self.c[n])

    def read_table(buf, columns=None):
        return Table({"question": ["q1", "q2"], "answer": ["a #### 3", "b #### 4"], "problem": ["p1", "p2", "p3"], "solution": ["\\boxed{1}"] * 3})
    pq = types.ModuleType("pyarrow.parquet"); pq.read_table = read_table
    pa = types.ModuleType("pyarrow"); pa.parquet = pq
    monkeypatch.setitem(sys.modules, "pyarrow", pa); monkeypatch.setitem(sys.modules, "pyarrow.parquet", pq)
    getter = lambda url, data=None: json.dumps(["https://x/a.parquet"]).encode() if "/parquet/" in url else b"PQ"
    res = F.fetch_benchmarks(str(tmp_path), getter=getter, n_math=14)
    assert res["gsm8k_test"] == 2 and res["math_test"] == 14
    row = json.loads(open(tmp_path / "_benchmarks" / "gsm8k_test.jsonl").readline())
    assert "question" in row and "answer" in row
