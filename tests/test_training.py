import os
import tempfile

import pytest


torch = pytest.importorskip("torch")
from altun.checkpoint import (clear_checkpoints, export_final_model, find_latest_checkpoint, list_checkpoints,  # noqa: E402
                              load_checkpoint, load_final_model, save_checkpoint)
from altun.config import ModelConfig, TrainConfig  # noqa: E402
from altun.model import ALTUNModel  # noqa: E402

CFG = dict(vocab_size=512, n_layers=2, d_model=64, n_heads=4, n_kv_heads=2, intermediate_size=128, context_length=64)


def test_checkpoint_atomic_keep_and_fallback():
    with tempfile.TemporaryDirectory() as d:
        for s in (10, 20, 30):
            save_checkpoint(d, {"step": s, "x": torch.ones(3)}, s, keep=2, min_free_gb=0)
        cks = list_checkpoints(d)
        assert len(cks) == 2 and cks[0].endswith("000000030.pt")
        assert not [f for f in os.listdir(d) if f.endswith(".tmp")]
        assert load_checkpoint(find_latest_checkpoint(d))["step"] == 30
        with open(cks[0], "ab") as f:  # en yeni bozulur → bir öncekine düşmeli
            f.write(b"garbage")
        assert find_latest_checkpoint(d).endswith("000000020.pt")
        assert clear_checkpoints(d) > 0 and find_latest_checkpoint(d) is None


def test_never_delete_only_valid_checkpoint():
    with tempfile.TemporaryDirectory() as d:
        save_checkpoint(d, {"step": 1}, 1, keep=1, min_free_gb=0)
        save_checkpoint(d, {"step": 2}, 2, keep=1, min_free_gb=0)
        assert find_latest_checkpoint(d) is not None and len(list_checkpoints(d)) == 1


def test_export_and_load_final_model_roundtrip():
    cfg = ModelConfig(**CFG)
    m = ALTUNModel(cfg).eval()
    with tempfile.TemporaryDirectory() as d:
        export_final_model(m, cfg, d)
        m2, cfg2 = load_final_model(d)
    x = torch.randint(0, 512, (1, 10))
    with torch.no_grad():
        assert torch.allclose(m(x)["logits"], m2(x)["logits"], atol=1e-5)
    assert m2.lm_head.weight is m2.embed.weight


def test_trainer_dry_run_and_resume():
    """Küçük model + sentetik duman verisiyle uçtan uca: eğitim, checkpoint, final export, resume."""
    from data.prepare_data import prepare
    from training.trainer import Trainer
    with tempfile.TemporaryDirectory() as tmp:
        data_dir, res = os.path.join(tmp, "data"), os.path.join(tmp, "res")
        prepare(smoke=True, vocab_size=512, work_dir=data_dir, workers=1)
        cfg = ModelConfig(**CFG)
        t = TrainConfig(duration_hours=1.0, checkpoint_interval_minutes=0.0001, eval_interval_steps=3, eval_batches=2,
                        log_interval_steps=1, tokens_per_step=128, micro_batch_size=2, warmup_steps=2, fp16=False,
                        grad_checkpointing=True, prefetch_batches=2, max_checkpoints=2, min_free_disk_gb=0)
        r1 = Trainer(cfg, t, data_dir, res, resume=False, max_steps=6, allow_smoke=True).run()
        assert r1["step"] == 6 and r1["complete"] and r1["final_val_loss"] == r1["final_val_loss"]
        assert os.path.isfile(os.path.join(res, "final_model", "config.json"))
        assert os.path.isdir(os.path.join(res, "final_model", "tokenizer"))
        assert 1 <= len(list_checkpoints(os.path.join(res, "checkpoints"))) <= 2
        assert os.path.getsize(os.path.join(res, "logs", "training_log.csv")) > 100
        r2 = Trainer(cfg, t, data_dir, res, resume="auto", max_steps=9, allow_smoke=True).run()
        assert r2["step"] == 9 and r2["tokens"] == 9 * 128  # kaldığı yerden devam etti
        r3 = Trainer(cfg, t, data_dir, res, resume=False, max_steps=2, allow_smoke=True).run()
        assert r3["step"] == 2  # sıfırdan başladı, eski checkpoint yüklenmedi


def test_trainer_refuses_smoke_data_without_flag():
    from data.prepare_data import prepare
    from training.trainer import Trainer
    with tempfile.TemporaryDirectory() as tmp:
        prepare(smoke=True, vocab_size=512, work_dir=os.path.join(tmp, "d"), workers=1)
        t = TrainConfig(tokens_per_step=128, micro_batch_size=2, fp16=False, min_free_disk_gb=0)
        with pytest.raises(RuntimeError, match="SMOKE"):
            Trainer(ModelConfig(**CFG), t, os.path.join(tmp, "d"), os.path.join(tmp, "r"), resume=False, max_steps=1).run()


def test_trainer_refuses_missing_data():
    from training.trainer import Trainer
    with tempfile.TemporaryDirectory() as tmp:
        t = TrainConfig(tokens_per_step=128, micro_batch_size=2, fp16=False, min_free_disk_gb=0)
        with pytest.raises(RuntimeError, match="Veri hazır değil"):
            Trainer(ModelConfig(**CFG), t, tmp, os.path.join(tmp, "r"), resume=False, max_steps=1).run()
