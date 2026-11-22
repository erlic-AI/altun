"""ALTUN byte-level BPE tokenizer (saf Python, harici bağımlılık yok).

Özellikler:
- Byte-level: her girdi kayıpsız kodlanır/çözülür (unknown token gerekmez).
- Matematik duyarlı ön-bölme: LaTeX komutları (\\frac), tekil rakamlar,
  tekil Unicode matematik sembolleri (∀ ∃ ≤ ∈ ...), ASCII operatör grupları (:=, ->, <=).
- Özel tokenlar en başta (id 0..N-1), ardından 256 byte, ardından birleştirmeler.
"""
from __future__ import annotations

import json
import os
import re
from collections import Counter, defaultdict
from typing import Dict, Iterable, List, Optional, Sequence, Tuple

SPECIAL_TOKENS = [
    "<|pad|>", "<|bos|>", "<|eos|>", "<|unk|>",
    "<|user|>", "<|assistant|>", "<|proof|>", "<|lean|>",
]

# Ön-bölme deseni (yalnızca stdlib `re`).
_PRETOKEN_RE = re.compile(
    r"""\\[A-Za-z]+"""            # LaTeX komutu: \frac, \forall
    r"""|\\[^A-Za-z\s]"""          # \{ \\ \,
    r"""| ?[^\W\d_]+"""            # (boşluk +) harf dizisi (Unicode: Türkçe dahil)
    r"""|\d"""                     # tek rakam
    r"""|[^\x00-\x7f\w\s]"""       # tek Unicode sembol (∀, ≤, ∈, →, ...)
    r"""| ?(?:[^\w\s\x80-\U0010ffff]|_)+"""  # ASCII noktalama/operatör grubu (+ alt çizgi)
    r"""|\s*\n+\s*"""              # satır sonları
    r"""|\s+(?!\S)|\s+""",
    re.UNICODE,
)


def pretokenize(text: str) -> List[bytes]:
    return [m.group(0).encode("utf-8") for m in _PRETOKEN_RE.finditer(text)]


class ALTUNTokenizer:
    def __init__(self, merges: Optional[List[Tuple[int, int]]] = None,
                 special_tokens: Optional[Sequence[str]] = None):
        self.special_tokens: List[str] = list(special_tokens or SPECIAL_TOKENS)
        self.n_special = len(self.special_tokens)
        self.merges: List[Tuple[int, int]] = [tuple(m) for m in (merges or [])]
        self._build()

    # ------------------------------------------------------------------ build
    def _build(self) -> None:
        self.special_to_id = {t: i for i, t in enumerate(self.special_tokens)}
        self.byte_offset = self.n_special
        self.vocab: Dict[int, bytes] = {}
        for i in range(256):
            self.vocab[self.byte_offset + i] = bytes([i])
        self.merge_rank: Dict[Tuple[int, int], int] = {}
        self.merge_to_id: Dict[Tuple[int, int], int] = {}
        for r, (a, b) in enumerate(self.merges):
            new_id = self.byte_offset + 256 + r
            self.vocab[new_id] = self.vocab[a] + self.vocab[b]
            self.merge_rank[(a, b)] = r
            self.merge_to_id[(a, b)] = new_id
        self._cache: Dict[bytes, List[int]] = {}
        if self.special_tokens:
            pat = "|".join(re.escape(t) for t in sorted(self.special_tokens, key=len, reverse=True))
            self._special_re = re.compile("(" + pat + ")")
        else:
            self._special_re = None

    @property
    def vocab_size(self) -> int:
        return self.byte_offset + 256 + len(self.merges)

    @property
    def pad_id(self) -> int: return self.special_to_id["<|pad|>"]
    @property
    def bos_id(self) -> int: return self.special_to_id["<|bos|>"]
    @property
    def eos_id(self) -> int: return self.special_to_id["<|eos|>"]
    @property
    def unk_id(self) -> int: return self.special_to_id["<|unk|>"]

    # ----------------------------------------------------------------- train
    @classmethod
    def train(cls, texts: Iterable[str], vocab_size: int = 16384,
              min_frequency: int = 2, verbose: bool = False) -> "ALTUNTokenizer":
        n_special = len(SPECIAL_TOKENS)
        n_merges = vocab_size - n_special - 256
        if n_merges < 0:
            raise ValueError(f"vocab_size en az {n_special + 256} olmalı")
        offset = n_special
        word_counts: Counter = Counter()
        for t in texts:
            if not t:
                continue
            for piece in _split_specials(t):
                word_counts.update(pretokenize(piece))
        words = [list(offset + b for b in w) for w in word_counts]
        freqs = [word_counts[w] for w in word_counts]

        pair_counts: Dict[Tuple[int, int], int] = defaultdict(int)
        pair_where: Dict[Tuple[int, int], set] = defaultdict(set)
        for wi, w in enumerate(words):
            for p in zip(w, w[1:]):
                pair_counts[p] += freqs[wi]
                pair_where[p].add(wi)

        merges: List[Tuple[int, int]] = []
        for step in range(n_merges):
            if not pair_counts:
                break
            best = max(pair_counts.items(), key=lambda kv: (kv[1], -kv[0][0], -kv[0][1]))
            pair, cnt = best
            if cnt < min_frequency:
                break
            new_id = offset + 256 + len(merges)
            merges.append(pair)
            for wi in list(pair_where[pair]):
                w = words[wi]
                f = freqs[wi]
                # eski çiftleri çıkar
                for p in zip(w, w[1:]):
                    pair_counts[p] -= f
                    if pair_counts[p] <= 0:
                        del pair_counts[p]
                # birleştir
                out, i = [], 0
                while i < len(w):
                    if i < len(w) - 1 and w[i] == pair[0] and w[i + 1] == pair[1]:
                        out.append(new_id); i += 2
                    else:
                        out.append(w[i]); i += 1
                words[wi] = out
                for p in zip(out, out[1:]):
                    pair_counts[p] += f
                    pair_where[p].add(wi)
            pair_counts.pop(pair, None)
            pair_where.pop(pair, None)
            if verbose and (step + 1) % 500 == 0:
                print(f"[tokenizer] merge {step + 1}/{n_merges}")
        return cls(merges)

    # ---------------------------------------------------------------- encode
    def _bpe(self, word: bytes) -> List[int]:
        hit = self._cache.get(word)
        if hit is not None:
            return hit
        ids = [self.byte_offset + b for b in word]
        while len(ids) > 1:
            best_rank, best_i = None, -1
            for i in range(len(ids) - 1):
                r = self.merge_rank.get((ids[i], ids[i + 1]))
                if r is not None and (best_rank is None or r < best_rank):
                    best_rank, best_i = r, i
            if best_rank is None:
                break
            a, b = ids[best_i], ids[best_i + 1]
            new_id = self.merge_to_id[(a, b)]
            out, i = [], 0
            while i < len(ids):
                if i < len(ids) - 1 and ids[i] == a and ids[i + 1] == b:
                    out.append(new_id); i += 2
                else:
                    out.append(ids[i]); i += 1
            ids = out
        if len(self._cache) < 500_000:
            self._cache[word] = ids
        return ids

    def encode(self, text: str, add_bos: bool = False, add_eos: bool = False,
               allow_special: bool = True) -> List[int]:
        ids: List[int] = [self.bos_id] if add_bos else []
        for piece in _split_specials(text, self._special_re if allow_special else None):
            if allow_special and piece in self.special_to_id:
                ids.append(self.special_to_id[piece])
                continue
            for w in pretokenize(piece):
                ids.extend(self._bpe(w))
        if add_eos:
            ids.append(self.eos_id)
        return ids

    def decode(self, ids: Sequence[int], skip_special: bool = False) -> str:
        out = bytearray()
        for i in ids:
            i = int(i)
            if i < self.n_special:
                if not skip_special:
                    out += self.special_tokens[i].encode("utf-8")
            elif i in self.vocab:
                out += self.vocab[i]
            else:
                raise ValueError(f"Geçersiz token id: {i} (vocab_size={self.vocab_size})")
        return out.decode("utf-8", errors="replace")

    # ------------------------------------------------------------ save/load
    def save(self, directory: str) -> None:
        os.makedirs(directory, exist_ok=True)
        data = {"type": "altun-byte-bpe", "version": 1,
                "special_tokens": self.special_tokens,
                "merges": [list(m) for m in self.merges]}
        tmp = os.path.join(directory, "tokenizer.json.tmp")
        with open(tmp, "w", encoding="utf-8") as f:
            json.dump(data, f, ensure_ascii=False)
        os.replace(tmp, os.path.join(directory, "tokenizer.json"))

    @classmethod
    def load(cls, directory: str) -> "ALTUNTokenizer":
        path = os.path.join(directory, "tokenizer.json")
        if not os.path.isfile(path):
            raise FileNotFoundError(f"Tokenizer dosyası yok: {path}")
        with open(path, encoding="utf-8") as f:
            data = json.load(f)
        if data.get("type") != "altun-byte-bpe":
            raise ValueError("Tanınmayan tokenizer formatı")
        return cls(merges=[tuple(m) for m in data["merges"]],
                   special_tokens=data["special_tokens"])

    # ----------------------------------------------------------------- stats
    def stats(self, texts: Iterable[str]) -> Dict[str, float]:
        n_chars = n_bytes = n_tokens = 0
        for t in texts:
            n_chars += len(t)
            n_bytes += len(t.encode("utf-8"))
            n_tokens += len(self.encode(t))
        return {"vocab_size": self.vocab_size, "chars": n_chars, "bytes": n_bytes,
                "tokens": n_tokens,
                "chars_per_token": n_chars / max(n_tokens, 1),
                "bytes_per_token": n_bytes / max(n_tokens, 1)}


def _split_specials(text: str, special_re=None) -> List[str]:
    """Özel token metinlerini ayır. special_re=None ise tek parça döner."""
    if special_re is None:
        return [text]
    return [p for p in special_re.split(text) if p != ""]
