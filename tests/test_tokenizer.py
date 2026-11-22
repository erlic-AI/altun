import os
import sys
import tempfile

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
from altun.tokenizer import ALTUNTokenizer, pretokenize  # noqa: E402

CORPUS = [
    "Let $f(x)=x^2+1$. Then \\frac{d}{dx} f(x) = 2x. \\forall x \\in \\mathbb{R}, x^2 \\geq 0.",
    "theorem add_comm (a b : Nat) : a + b = b + a := by\n  induction a with\n  | zero => simp\n  | succ n ih => omega\n",
    "∀ x ∈ S, ∃ y, x ≤ y ∧ y ≠ 0 → x ∨ y",
    "Bir asal sayı yalnızca 1 ve kendisine bölünebilen doğal sayıdır. Öklid ispatı çelişkiyle yapılır.",
    "def quicksort(a): return a if len(a)<2 else quicksort([x for x in a[1:] if x<a[0]])+[a[0]]",
    "Proof. By induction on n. The base case n = 0 is trivial. QED",
] * 20


def _tok(vocab=600):
    return ALTUNTokenizer.train(CORPUS, vocab_size=vocab, min_frequency=2)


def test_roundtrip_math_latex_lean_turkish():
    tok = _tok()
    samples = CORPUS[:6] + ["Şğüöçİı ∑ ∫ ∂ ℝ ℕ ⊢ λ ⟨a, b⟩", "", "  \n\n\t x", "🙂 emoji"]
    for s in samples:
        assert tok.decode(tok.encode(s)) == s


def test_special_tokens():
    tok = _tok()
    ids = tok.encode("<|bos|>x<|eos|>", allow_special=True)
    assert ids[0] == tok.bos_id and ids[-1] == tok.eos_id
    ids2 = tok.encode("<|eos|>", allow_special=False)
    assert tok.eos_id not in ids2
    assert tok.decode(tok.encode("a", add_bos=True, add_eos=True), skip_special=True) == "a"


def test_save_load_identical():
    tok = _tok()
    with tempfile.TemporaryDirectory() as d:
        tok.save(d)
        tok2 = ALTUNTokenizer.load(d)
    s = CORPUS[1]
    assert tok.encode(s) == tok2.encode(s)
    assert tok.vocab_size == tok2.vocab_size


def test_math_symbols_are_single_pretokens():
    pt = pretokenize("∀x∈S")
    assert "∀".encode() in pt and "∈".encode() in pt
    assert b"\\forall" in pretokenize("\\forall x")
    assert [p for p in pretokenize("123")] == [b"1", b"2", b"3"]


def test_invalid_id_raises():
    tok = _tok()
    try:
        tok.decode([tok.vocab_size + 5])
    except ValueError:
        return
    raise AssertionError("ValueError bekleniyordu")


def test_compression_and_vocab_bounds():
    tok = _tok()
    st = tok.stats(CORPUS[:6])
    assert st["bytes_per_token"] > 1.5
    assert tok.vocab_size <= 600
    assert max(tok.encode(CORPUS[0])) < tok.vocab_size
