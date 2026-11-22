"""Metin üretimi: KV-cache, temperature, top-k, top-p, repetition penalty."""
from __future__ import annotations

from typing import List, Optional

import torch


def _rep_penalty(logits: torch.Tensor, seen: torch.Tensor, penalty: float) -> torch.Tensor:
    if penalty == 1.0:
        return logits
    score = torch.gather(logits, 1, seen)
    score = torch.where(score < 0, score * penalty, score / penalty)
    return logits.scatter(1, seen, score)


def _filter(logits: torch.Tensor, top_k: int, top_p: float) -> torch.Tensor:
    if top_k and top_k > 0:
        k = min(top_k, logits.size(-1))
        kth = torch.topk(logits, k)[0][..., -1, None]
        logits = logits.masked_fill(logits < kth, float("-inf"))
    if top_p and 0.0 < top_p < 1.0:
        sorted_logits, idx = torch.sort(logits, descending=True)
        probs = torch.softmax(sorted_logits, dim=-1)
        cum = torch.cumsum(probs, dim=-1)
        remove = (cum - probs) > top_p  # ilk token her zaman kalır
        sorted_logits = sorted_logits.masked_fill(remove, float("-inf"))
        logits = torch.full_like(logits, float("-inf")).scatter(1, idx, sorted_logits)
    return logits


@torch.no_grad()
def generate_ids(model, input_ids: torch.Tensor, max_new_tokens: int = 128,
                 temperature: float = 0.8, top_k: int = 50, top_p: float = 0.95,
                 repetition_penalty: float = 1.0, eos_id: Optional[int] = None) -> List[List[int]]:
    """input_ids: [B, T] (aynı uzunlukta). Her satır için yeni token listesi döner (eos dahil değil)."""
    model.eval()
    cfg = model.cfg
    b, t = input_ids.shape
    if t == 0:
        raise ValueError("Boş prompt")
    if t >= cfg.context_length:
        raise ValueError(f"Prompt ({t}) context_length ({cfg.context_length}) sınırını aşıyor")
    max_new_tokens = min(max_new_tokens, cfg.context_length - t)
    if temperature < 0 or repetition_penalty <= 0:
        raise ValueError("Geçersiz örnekleme parametresi")

    out = model(input_ids, use_cache=True)
    past, logits = out["past_kvs"], out["logits"][:, -1, :].float()
    seen = input_ids
    new_tokens = torch.empty((b, 0), dtype=torch.long, device=input_ids.device)
    finished = torch.zeros(b, dtype=torch.bool, device=input_ids.device)
    for _ in range(max_new_tokens):
        logits = _rep_penalty(logits, seen, repetition_penalty)
        if temperature == 0:
            nxt = torch.argmax(logits, dim=-1)
        else:
            logits = _filter(logits / temperature, top_k, top_p)
            nxt = torch.multinomial(torch.softmax(logits, dim=-1), 1).squeeze(1)
        if eos_id is not None:
            nxt = torch.where(finished, torch.full_like(nxt, eos_id), nxt)
            finished = finished | (nxt == eos_id)
        new_tokens = torch.cat([new_tokens, nxt[:, None]], dim=1)
        seen = torch.cat([seen, nxt[:, None]], dim=1)
        if eos_id is not None and bool(finished.all()):
            break
        out = model(nxt[:, None], past_kvs=past, use_cache=True)
        past, logits = out["past_kvs"], out["logits"][:, -1, :].float()
    results = []
    for row in new_tokens.tolist():
        if eos_id is not None and eos_id in row:
            row = row[:row.index(eos_id)]
        results.append(row)
    return results


def generate_text(model, tokenizer, prompt: str, device: str = "cpu", **kw) -> str:
    ids = tokenizer.encode(prompt, add_bos=True)
    x = torch.tensor([ids], dtype=torch.long, device=device)
    new = generate_ids(model, x, eos_id=tokenizer.eos_id, **kw)[0]
    return tokenizer.decode(new, skip_special=True)
