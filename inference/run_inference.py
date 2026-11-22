"""ALTUN 1 inference: model/tokenizer yükleme, örnekleme parametreleri, bağlam doğrulama, FP16.
CLI:  python -m inference.run_inference --model-dir <final_model> --prompt "..."
Notebook:  from inference.run_inference import load_altun, ask, demo"""
from __future__ import annotations

import argparse
import os
import sys

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

import torch  # noqa: E402

from altun.checkpoint import load_final_model  # noqa: E402
from altun.generation import generate_ids  # noqa: E402
from altun.tokenizer import ALTUNTokenizer  # noqa: E402

DEMO_QUESTIONS = {
    "Algebra": "Solve for x: 2x + 6 = 14.",
    "Number Theory": "What is gcd(48, 18)?",
    "Mathematical Logic": "Evaluate: True IMPLIES False. Answer true or false.",
    "Combinatorics": "What is C(6, 2) (n choose k)?",
    "Algorithm Complexity": "What is the worst-case time complexity of binary search?",
    "Theoretical CS": "Is SAT in NP? Answer yes or no.",
    "Türkçe": "12 ile 15 sayılarının toplamı kaçtır?",
}


def load_altun(model_dir: str, device: str = "auto", fp16: bool = True):
    device = ("cuda" if torch.cuda.is_available() else "cpu") if device == "auto" else device
    if device.startswith("cuda") and not torch.cuda.is_available():
        raise RuntimeError("CUDA istendi ama kullanılamıyor")
    model, cfg = load_final_model(model_dir, device="cpu")
    tok = ALTUNTokenizer.load(os.path.join(model_dir, "tokenizer"))
    if tok.vocab_size != cfg.vocab_size:
        raise RuntimeError(f"Model/tokenizer uyuşmazlığı: {tok.vocab_size} != {cfg.vocab_size}")
    if device.startswith("cuda") and fp16:
        model = model.half()
    return model.to(device).eval(), tok, device


def ask(model, tok, device: str, prompt: str, max_new_tokens: int = 256, temperature: float = 0.0, top_k: int = 40,
        top_p: float = 0.95, repetition_penalty: float = 1.0) -> str:
    ids = tok.encode(prompt, add_bos=True)
    limit = model.cfg.context_length
    if len(ids) >= limit:
        raise ValueError(f"Prompt {len(ids)} token; context_length {limit} sınırını aşıyor")
    x = torch.tensor([ids], dtype=torch.long, device=device)
    new = generate_ids(model, x, max_new_tokens=max_new_tokens, temperature=temperature, top_k=top_k, top_p=top_p,
                       repetition_penalty=repetition_penalty, eos_id=tok.eos_id)[0]
    return tok.decode(new, skip_special=True)


def demo(model, tok, device: str, **kw) -> None:
    for name, q in DEMO_QUESTIONS.items():
        print(f"\n### {name}\nSoru : {q}\nALTUN: {ask(model, tok, device, f'Problem: {q}' + chr(10) + 'Solution:', **kw).strip()}")


def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("--model-dir", default="/kaggle/working/ALTUN-1-RESULTS/final_model")
    ap.add_argument("--prompt", default=None)
    ap.add_argument("--device", default="auto")
    ap.add_argument("--max-new-tokens", type=int, default=256)
    ap.add_argument("--temperature", type=float, default=0.0, help="0 = greedy (değerlendirmede kullanılan)")
    ap.add_argument("--top-k", type=int, default=40)
    ap.add_argument("--top-p", type=float, default=0.95)
    ap.add_argument("--repetition-penalty", type=float, default=1.0)
    ap.add_argument("--raw", action="store_true", help="Prompt'u olduğu gibi ver (varsayılan: Problem/Solution biçimine sarılır)")
    a = ap.parse_args()
    model, tok, dev = load_altun(a.model_dir, a.device)
    kw = dict(max_new_tokens=a.max_new_tokens, temperature=a.temperature, top_k=a.top_k, top_p=a.top_p,
              repetition_penalty=a.repetition_penalty)
    if a.prompt:
        prompt = a.prompt if a.raw else f"Problem: {a.prompt}\nSolution:"
        print(ask(model, tok, dev, prompt, **kw))
    else:
        demo(model, tok, dev, **kw)


if __name__ == "__main__":
    main()
