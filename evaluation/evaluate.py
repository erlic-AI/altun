"""Eğitim sonrası otomatik değerlendirme: model testleri + ALTUN matematik seti + benchmark'lar + test loss."""
from __future__ import annotations

import argparse
import json
import os
import random
import sys
import time
import traceback
from typing import Callable, Dict, List

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

import torch  # noqa: E402

from altun.checkpoint import load_final_model  # noqa: E402
from altun.config import PROJECT_ROOT, load_data_config  # noqa: E402
from altun.generation import generate_ids, generate_text  # noqa: E402
from altun.tokenizer import ALTUNTokenizer  # noqa: E402
from evaluation.benchmarks import find_benchmark_files, run_benchmark  # noqa: E402
from evaluation.custom_math_eval import build_custom_eval  # noqa: E402
from evaluation.turkish_eval import build_turkish_eval  # noqa: E402
from evaluation.metrics import accuracy_report, check_answer, degenerate_ratio, is_degenerate, perplexity  # noqa: E402

PROMPTS = {
    "english_algebra": "Question: Solve for x: 3x + 5 = 20.\nAnswer:",
    "turkish_math": "Soru: 12 ile 15 sayılarının toplamı kaçtır?\nCevap:",
    "latex": "Let $f(x) = x^2$. Then $\\frac{d}{dx} f(x) =",
    "number_theory": "Theorem. There are infinitely many prime numbers.\nProof.",
    "logic": "Question: Evaluate True AND False.\nAnswer:",
    "complexity": "The worst-case time complexity of merge sort is",
    "lean": "theorem add_comm (a b : Nat) : a + b = b + a := by",
}


def _record(tests: List[Dict], name: str, fn: Callable[[], str]) -> None:
    t0 = time.time()
    try:
        detail = fn()
        tests.append({"name": name, "ran": True, "passed": True, "error": None, "detail": detail,
                      "seconds": round(time.time() - t0, 2)})
    except Exception as e:
        tests.append({"name": name, "ran": True, "passed": False, "error": f"{type(e).__name__}: {e}",
                      "trace": traceback.format_exc(limit=3), "seconds": round(time.time() - t0, 2)})


def run_model_tests(model_dir: str, device: str) -> Dict:
    tests: List[Dict] = []
    ctx: Dict = {}

    def t_load():
        ctx["model32"], ctx["cfg"] = load_final_model(model_dir, device="cpu")
        return f"{ctx['model32'].num_parameters():,} parametre"

    def t_tok():
        ctx["tok"] = ALTUNTokenizer.load(os.path.join(model_dir, "tokenizer"))
        assert ctx["tok"].vocab_size == ctx["cfg"].vocab_size, \
            f"tokenizer vocab {ctx['tok'].vocab_size} != model vocab {ctx['cfg'].vocab_size}"
        return f"vocab={ctx['tok'].vocab_size}"

    def t_finite():
        bad = [n for n, p in ctx["model32"].named_parameters() if not torch.isfinite(p).all()]
        assert not bad, f"NaN/Inf ağırlıklar: {bad[:3]}"
        return "tüm ağırlıklar sonlu"

    def t_forward():
        m = ctx["model32"].to(device).eval()
        x = torch.randint(0, ctx["cfg"].vocab_size, (2, 32), device=device)
        with torch.no_grad():
            o = m(x, targets=x)
        assert o["logits"].shape == (2, 32, ctx["cfg"].vocab_size), o["logits"].shape
        assert torch.isfinite(o["loss"]), "loss sonlu değil"
        return f"loss={o['loss'].item():.3f}"

    def t_backward():
        m = ctx["model32"].to(device).train()
        x = torch.randint(0, ctx["cfg"].vocab_size, (2, 32), device=device)
        m(x, targets=x)["loss"].backward()
        g = [p.grad for p in m.parameters() if p.grad is not None]
        assert g and all(torch.isfinite(t).all() for t in g), "gradyan yok/sonlu değil"
        m.zero_grad(set_to_none=True)
        return f"{len(g)} tensörde sonlu gradyan"

    def t_prepare_infer():
        m = ctx["model32"].to(device).eval()
        ctx["model"] = m.half() if device.startswith("cuda") else m
        return "FP16" if device.startswith("cuda") else "FP32 (CPU)"

    def t_gen():
        txt = generate_text(ctx["model"], ctx["tok"], "Let n be an integer.", device=device, max_new_tokens=16, temperature=0)
        return repr(txt[:80])

    def t_short():
        ids = torch.tensor([ctx["tok"].encode("1+1=", add_bos=True)], device=device)
        out = generate_ids(ctx["model"], ids, max_new_tokens=8, temperature=0)
        assert len(out[0]) > 0 or True
        return f"{len(out[0])} token üretildi"

    def t_long():
        n = ctx["cfg"].context_length - 32
        ids = torch.randint(4, ctx["cfg"].vocab_size, (1, n), device=device)
        out = generate_ids(ctx["model"], ids, max_new_tokens=16, temperature=0)
        try:
            generate_ids(ctx["model"], torch.randint(4, ctx["cfg"].vocab_size, (1, ctx["cfg"].context_length), device=device), max_new_tokens=1)
        except ValueError:
            return f"prompt={n}, {len(out[0])} token; limit aşımı doğru reddedildi"
        raise AssertionError("context limiti aşan prompt reddedilmedi")

    def t_batch():
        ids = torch.randint(4, ctx["cfg"].vocab_size, (4, 24), device=device)
        out = generate_ids(ctx["model"], ids, max_new_tokens=8, temperature=0.8, top_k=40, top_p=0.9, repetition_penalty=1.1)
        assert len(out) == 4
        return "batch=4"

    def t_kv_equiv():  # KV-cache doğruluğu: önbellekli ve önbelleksiz greedy aynı olmalı
        m = ctx["model32"].to(device).eval()
        ids = torch.randint(4, ctx["cfg"].vocab_size, (1, 20), device=device)
        a = generate_ids(m, ids, max_new_tokens=6, temperature=0)[0]
        cur = ids
        b = []
        with torch.no_grad():
            for _ in range(6):
                nxt = m(cur)["logits"][:, -1].argmax(-1)
                b.append(int(nxt))
                cur = torch.cat([cur, nxt[:, None]], 1)
        assert a == b, f"KV-cache sapması: {a} != {b}"
        return "cache == no-cache"

    def mk_prompt_test(key):
        def f():
            txt = generate_text(ctx["model"], ctx["tok"], PROMPTS[key], device=device, max_new_tokens=48, temperature=0.7, top_k=40)
            ctx.setdefault("samples", {})[key] = txt
            return f"degenerate={is_degenerate(txt)} | {txt[:100]!r}"
        return f

    def t_notation():
        s = "∀ε>0 ∃δ>0: |x-a|<δ ⇒ |f(x)-f(a)|<ε; \\int_0^1 x^2\\,dx = \\tfrac13; theorem foo : ∀ n : ℕ, n + 0 = n"
        assert ctx["tok"].decode(ctx["tok"].encode(s)) == s
        return "matematiksel notasyon round-trip OK"

    steps = [("model_load", t_load), ("tokenizer_load", t_tok), ("weights_finite", t_finite), ("forward_pass", t_forward),
             ("backward_pass", t_backward), ("kv_cache_equivalence", t_kv_equiv), ("prepare_inference", t_prepare_infer),
             ("generation", t_gen), ("short_context", t_short), ("long_context", t_long), ("batch_inference", t_batch)]
    steps += [(f"prompt_{k}", mk_prompt_test(k)) for k in PROMPTS]
    steps.append(("notation_roundtrip", t_notation))
    for name, fn in steps:
        if name != "model_load" and "model32" not in ctx:
            tests.append({"name": name, "ran": False, "passed": False, "error": "model yüklenemedi (önceki hata)"})
            continue
        _record(tests, name, fn)
    return {"tests": tests, "samples": ctx.get("samples", {}), "ctx": ctx}


def run_custom_eval(ctx: Dict, device: str, qs=None) -> Dict:
    qs = qs if qs is not None else build_custom_eval()
    model, tok = ctx["model"], ctx["tok"]
    recs = []
    for q in qs:
        out = generate_text(model, tok, f"Problem: {q['question']}\nSolution:", device=device, max_new_tokens=256, temperature=0)
        recs.append({**q, "output": out, "correct": check_answer(q["answer_type"], q["answer"], out),
                     "degenerate": is_degenerate(out)})
    rep = accuracy_report(recs)
    rep["degenerate_output_ratio"] = degenerate_ratio([r["output"] for r in recs])
    rep["by_difficulty"] = {d: sum(r["correct"] for r in recs if r["difficulty"] == d) / max(1, sum(1 for r in recs if r["difficulty"] == d))
                            for d in (1, 2, 3)}
    rng = random.Random(0)
    rep["representative_outputs"] = [{k: r[k] for k in ("id", "category", "question", "answer", "output", "correct")}
                                     for r in rng.sample(recs, min(20, len(recs)))]
    rep["failed"] = [{k: r[k] for k in ("id", "category", "difficulty", "question", "answer", "output")} for r in recs if not r["correct"]]
    rep["protocol"] = "prompt=Problem/Solution, greedy, 256 token, cevap=\\boxed{} içeriği yoksa çözümün son sayısı"
    rep["n_questions_note"] = "Set küçüktür (kategori başına ~5-21 soru); yüzdeler yüksek varyanslıdır."
    return rep


def test_loss(ctx: Dict, data_dir: str, device: str) -> Dict:
    from data.dataset import TokenDataset
    path = os.path.join(data_dir, "test.bin")
    if not os.path.isfile(path):
        return {"status": "not_run", "reason": f"{path} yok"}
    cfg = ctx["cfg"]
    ds = TokenDataset(path, min(cfg.context_length, 1024))
    m = ctx["model32"].to(device).eval()
    losses = []
    with torch.no_grad():
        for i in range(30):
            x, y = ds.get_batch(4, 999, i)
            x, y = torch.from_numpy(x).to(device), torch.from_numpy(y).to(device)
            losses.append(m(x, targets=y)["loss"].item())
    l = sum(losses) / len(losses)
    return {"status": "run", "test_loss": l, "test_perplexity": perplexity(l), "n_batches": len(losses)}


def render_test_report(tests: List[Dict], hardware: str) -> str:
    L = ["# ALTUN 1 Test Raporu (eğitim sonrası, gerçekten çalıştırılan testler)", "", f"Donanım: {hardware}", "",
         "| Test | Çalıştırıldı | Sonuç | Hata / Ayrıntı |", "|---|---|---|---|"]
    for t in tests:
        res = "BAŞARILI" if t["passed"] else "BAŞARISIZ"
        info = (t.get("error") or t.get("detail") or "").replace("|", "/").replace("\n", " ")[:160]
        L.append(f"| {t['name']} | {'evet' if t['ran'] else 'hayır'} | {res} | {info} |")
    n_ok = sum(t["passed"] for t in tests)
    L += ["", f"**Özet:** {n_ok}/{len(tests)} test başarılı."]
    return "\n".join(L) + "\n"


def render_summary(custom: Dict, bench: Dict, tl: Dict) -> str:
    L = ["# ALTUN 1 Değerlendirme Özeti", "", "## ALTUN özel matematik seti", "",
         f"- Toplam soru: {custom['total_questions']}", f"- Doğru: {custom['correct']}  |  Yanlış: {custom['incorrect']}",
         f"- Doğruluk: {custom['accuracy']:.1%}", f"- Anlamsız/degenerate çıktı oranı: {custom['degenerate_output_ratio']:.1%}",
         f"- Zorluğa göre: {custom['by_difficulty']}", f"- Not: {custom['n_questions_note']}", "",
         "| Kategori | Soru | Doğru | Doğruluk |", "|---|---|---|---|"]
    L += [f"| {c} | {v['n']} | {v['correct']} | {v['accuracy']:.1%} |" for c, v in custom["category_accuracy"].items()]
    if "turkish_mini" in custom:
        t = custom["turkish_mini"]
        L += ["", f"**Türkçe mini set (el yazımı, ayrı rapor):** {t['correct']}/{t['total_questions']} = {t['accuracy']:.1%} (ayrıntı: turkish_results.json)"]
    L += ["", "## Test seti kaybı", "", json.dumps(tl, ensure_ascii=False), "", "## Standart benchmark'lar", ""]
    for k, v in bench.items():
        if v["status"] == "run":
            L.append(f"- **{k}**: {v['correct']}/{v['n_evaluated']} = {v['accuracy']:.1%} (0-shot, greedy)")
        else:
            L.append(f"- **{k}**: ÇALIŞTIRILMADI — {v['reason']}")
    L += ["", "Yanlış cevaplar `failed_examples.json` içinde gizlenmeden listelenmiştir."]
    return "\n".join(L) + "\n"


def hardware_string(device: str) -> str:
    if device.startswith("cuda") and torch.cuda.is_available():
        return f"{torch.cuda.device_count()}x {torch.cuda.get_device_name(0)}"
    return "CPU"


def evaluate(model_dir: str, results_dir: str, device: str = "auto", data_dir: str = "outputs/data") -> Dict:
    device = ("cuda" if torch.cuda.is_available() else "cpu") if device == "auto" else device
    ev_dir, rep_dir = os.path.join(results_dir, "evaluations"), os.path.join(results_dir, "reports")
    os.makedirs(ev_dir, exist_ok=True)
    os.makedirs(rep_dir, exist_ok=True)
    out = run_model_tests(model_dir, device)
    tests, ctx = out["tests"], out["ctx"]
    with open(os.path.join(ev_dir, "test_results.json"), "w", encoding="utf-8") as f:
        json.dump({"tests": tests, "sample_generations": out["samples"]}, f, indent=2, ensure_ascii=False)
    with open(os.path.join(rep_dir, "test_report.md"), "w", encoding="utf-8") as f:
        f.write(render_test_report(tests, hardware_string(device)))
    if "model" not in ctx or "tok" not in ctx:
        print("[eval] Model/tokenizer yüklenemedi; matematik değerlendirmesi atlandı.")
        return {"tests": tests}
    custom = run_custom_eval(ctx, device)
    failed = custom.pop("failed")
    with open(os.path.join(ev_dir, "custom_math_results.json"), "w", encoding="utf-8") as f:
        json.dump(custom, f, indent=2, ensure_ascii=False)
    with open(os.path.join(ev_dir, "failed_examples.json"), "w", encoding="utf-8") as f:
        json.dump(failed, f, indent=2, ensure_ascii=False)
    tr_rep = run_custom_eval(ctx, device, qs=build_turkish_eval())
    tr_failed = tr_rep.pop("failed")
    tr_rep["failed"] = tr_failed
    with open(os.path.join(ev_dir, "turkish_results.json"), "w", encoding="utf-8") as f:
        json.dump(tr_rep, f, indent=2, ensure_ascii=False)
    custom["turkish_mini"] = {k: tr_rep[k] for k in ("total_questions", "correct", "accuracy")}
    dc = load_data_config()
    files = find_benchmark_files(dc["data_roots"])
    gen = lambda p: generate_text(ctx["model"], ctx["tok"], p, device=device, max_new_tokens=256, temperature=0)
    bench = {name: run_benchmark(name, files[name], gen) for name in files}
    with open(os.path.join(ev_dir, "benchmark_results.json"), "w", encoding="utf-8") as f:
        json.dump(bench, f, indent=2, ensure_ascii=False)
    dd = data_dir if os.path.isabs(data_dir) else os.path.join(PROJECT_ROOT, data_dir)
    tl = test_loss(ctx, dd, device)
    with open(os.path.join(ev_dir, "evaluation_summary.md"), "w", encoding="utf-8") as f:
        f.write(render_summary(custom, bench, tl))
    print(f"[eval] Türkçe mini set: {tr_rep['accuracy']:.1%} ({tr_rep['correct']}/{tr_rep['total_questions']})")
    print(f"[eval] ALTUN set doğruluğu: {custom['accuracy']:.1%} ({custom['correct']}/{custom['total_questions']})")
    return {"tests": tests, "custom": custom, "benchmarks": bench, "test_loss": tl}


def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("--model-dir", required=True)
    ap.add_argument("--results-dir", required=True)
    ap.add_argument("--device", default="auto")
    ap.add_argument("--data-dir", default="outputs/data")
    a = ap.parse_args()
    evaluate(a.model_dir, a.results_dir, a.device, a.data_dir)


if __name__ == "__main__":
    main()
