import pytest

torch = pytest.importorskip("torch")
from altun.config import ModelConfig  # noqa: E402
from altun.generation import _filter, _rep_penalty, generate_ids  # noqa: E402
from altun.model import ALTUNModel  # noqa: E402


def model():
    torch.manual_seed(0)
    return ALTUNModel(ModelConfig(vocab_size=128, n_layers=2, d_model=64, n_heads=4, n_kv_heads=2,
                                  intermediate_size=128, context_length=64)).eval()


def test_greedy_is_deterministic_and_bounded():
    m = model()
    x = torch.randint(4, 128, (1, 8))
    a = generate_ids(m, x, max_new_tokens=10, temperature=0)
    b = generate_ids(m, x, max_new_tokens=10, temperature=0)
    assert a == b and len(a[0]) == 10


def test_batch_generation():
    out = generate_ids(model(), torch.randint(4, 128, (3, 8)), max_new_tokens=5, temperature=0.9, top_k=20, top_p=0.9,
                       repetition_penalty=1.2)
    assert len(out) == 3 and all(len(r) == 5 for r in out)


def test_prompt_too_long_rejected():
    with pytest.raises(ValueError):
        generate_ids(model(), torch.randint(4, 128, (1, 64)), max_new_tokens=1)


def test_max_new_tokens_clipped_to_context():
    out = generate_ids(model(), torch.randint(4, 128, (1, 60)), max_new_tokens=50, temperature=0)
    assert len(out[0]) == 4


def test_top_k_filter():
    l = torch.tensor([[1.0, 5.0, 3.0, 4.0]])
    f = _filter(l, 2, 1.0)
    assert torch.isfinite(f).sum() == 2 and torch.isfinite(f[0, 1]) and torch.isfinite(f[0, 3])


def test_top_p_keeps_top_token():
    l = torch.tensor([[10.0, 0.0, 0.0, 0.0]])
    f = _filter(l, 0, 0.5)
    assert torch.isfinite(f[0, 0]) and torch.isfinite(f).sum() == 1


def test_repetition_penalty():
    l = torch.tensor([[2.0, -2.0, 1.0]])
    r = _rep_penalty(l, torch.tensor([[0, 1]]), 2.0)
    assert r[0, 0] == 1.0 and r[0, 1] == -4.0 and r[0, 2] == 1.0


def test_eos_truncation():
    m = model()
    x = torch.randint(4, 128, (1, 8))
    first = generate_ids(m, x, max_new_tokens=6, temperature=0)[0]
    out = generate_ids(m, x, max_new_tokens=6, temperature=0, eos_id=first[2])[0]
    assert out == first[:2] or first[:2].count(first[2]) > 0
