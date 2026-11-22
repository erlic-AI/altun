import pytest

torch = pytest.importorskip("torch")
from altun.config import ModelConfig  # noqa: E402
from altun.layers import apply_rope, precompute_rope  # noqa: E402
from altun.model import ALTUNModel  # noqa: E402


def tiny(**kw):
    base = dict(vocab_size=128, n_layers=2, d_model=64, n_heads=4, n_kv_heads=2, intermediate_size=128, context_length=64)
    base.update(kw)
    return ModelConfig(**base)


def test_forward_shape_and_loss():
    m = ALTUNModel(tiny())
    x = torch.randint(0, 128, (3, 16))
    o = m(x, targets=x)
    assert o["logits"].shape == (3, 16, 128)
    assert torch.isfinite(o["loss"])
    assert 3.0 < o["loss"].item() < 7.0  # ~ln(128)=4.85 başlangıçta


def test_backward_grads_finite():
    m = ALTUNModel(tiny())
    x = torch.randint(0, 128, (2, 16))
    m(x, targets=x)["loss"].backward()
    grads = [p.grad for p in m.parameters()]
    assert all(g is not None and torch.isfinite(g).all() for g in grads)


def test_param_count_matches_analytic():
    for kw in ({}, {"tie_embeddings": False}, {"n_kv_heads": 4}):
        cfg = tiny(**kw)
        assert ALTUNModel(cfg).num_parameters() == cfg.param_count()


def test_tied_weights():
    m = ALTUNModel(tiny())
    assert m.lm_head.weight is m.embed.weight


def test_causality():
    m = ALTUNModel(tiny()).eval()
    x = torch.randint(0, 128, (1, 12))
    y = x.clone()
    y[0, 8:] = (y[0, 8:] + 1) % 128
    with torch.no_grad():
        a, b = m(x)["logits"], m(y)["logits"]
    assert torch.allclose(a[:, :8], b[:, :8], atol=1e-5)
    assert not torch.allclose(a[:, 8:], b[:, 8:], atol=1e-5)


def test_kv_cache_equals_full_forward():
    m = ALTUNModel(tiny()).eval()
    x = torch.randint(0, 128, (2, 10))
    with torch.no_grad():
        full = m(x)["logits"]
        o = m(x[:, :6], use_cache=True)
        o2 = m(x[:, 6:9], past_kvs=o["past_kvs"], use_cache=True)  # çok tokenli devam (maskeli yol)
        o3 = m(x[:, 9:], past_kvs=o2["past_kvs"], use_cache=True)  # tek token yolu
    assert torch.allclose(full[:, 6:9], o2["logits"], atol=1e-4)
    assert torch.allclose(full[:, 9:], o3["logits"], atol=1e-4)


def test_context_overflow_raises():
    m = ALTUNModel(tiny())
    with pytest.raises(ValueError):
        m(torch.randint(0, 128, (1, 65)))


def test_grad_checkpointing_equivalent():
    torch.manual_seed(0)
    m = ALTUNModel(tiny()).train()
    x = torch.randint(0, 128, (2, 16))
    m(x, targets=x)["loss"].backward()
    g1 = [p.grad.clone() for p in m.parameters()]
    m.zero_grad()
    m.grad_checkpointing = True
    m(x, targets=x)["loss"].backward()
    for a, b in zip(g1, [p.grad for p in m.parameters()]):
        assert torch.allclose(a, b, atol=1e-5)


def test_rope_preserves_norm():
    cos, sin = precompute_rope(16, 8)
    x = torch.randn(1, 2, 8, 16)
    y = apply_rope(x, cos, sin)
    assert torch.allclose(x.norm(dim=-1), y.norm(dim=-1), atol=1e-5)


def test_ignore_index_targets():
    m = ALTUNModel(tiny())
    x = torch.randint(0, 128, (1, 8))
    t = x.clone()
    t[:, :4] = -100
    assert torch.isfinite(m(x, targets=t)["loss"])
