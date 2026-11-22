"""ALTUN 1 Trainer: FP16 AMP, DDP, gradient accumulation/checkpointing, wall-clock tabanlı süre,
atomik checkpoint + otomatik resume, CSV/JSONL log, sistem metrikleri."""
from __future__ import annotations

import hashlib
import json
import os
import signal
import sys
import time
from typing import Dict, Optional

import numpy as np
import torch
from torch.nn.parallel import DistributedDataParallel as DDP

from altun.checkpoint import (capture_rng_state, clear_checkpoints, export_final_model, find_latest_checkpoint,
                              load_checkpoint, restore_rng_state, save_checkpoint)
from altun.config import ModelConfig, TrainConfig, estimate_vram_gb
from altun.model import ALTUNModel
from altun.tokenizer import ALTUNTokenizer
from data.dataset import Prefetcher, TokenDataset
from training import distributed as D
from training.monitoring import MetricsLogger, make_plots, sample_system
from training.optimizer import build_optimizer
from training.scheduler import lr_at


def fmt_time(s: float) -> str:
    s = int(max(s, 0))
    return f"{s // 3600:02d}:{(s % 3600) // 60:02d}:{s % 60:02d}"


def make_scaler(enabled: bool):
    try:
        return torch.amp.GradScaler("cuda", enabled=enabled)
    except (AttributeError, TypeError):
        return torch.cuda.amp.GradScaler(enabled=enabled)


class Trainer:
    def __init__(self, model_cfg: ModelConfig, tcfg: TrainConfig, data_dir: str, results_dir: str,
                 resume="auto", max_steps: Optional[int] = None, allow_smoke: bool = False,
                 init_from: Optional[str] = None):
        self.mcfg, self.tcfg = model_cfg, tcfg
        self.data_dir, self.results = data_dir, results_dir
        self.resume, self.max_steps, self.allow_smoke = resume, max_steps, allow_smoke
        self.init_from = init_from
        self.rank, self.local, self.world, self.device = D.setup()
        self.main = self.rank == 0
        self.use_cuda = self.device.type == "cuda"
        self.ckpt_dir = os.path.join(results_dir, "checkpoints")
        self.log_dir = os.path.join(results_dir, "logs")
        self._sig = False
        self.stop_reason = "bilinmiyor"

    # -------------------------------------------------------------- setup
    def _check_data(self) -> None:
        mp = os.path.join(self.data_dir, "meta.json")
        if not os.path.isfile(mp):
            raise RuntimeError(f"Veri hazır değil: {mp} yok. Önce `python -m data.prepare_data` çalıştırın. Boş veriyle eğitim başlatılmaz.")
        with open(mp, encoding="utf-8") as f:
            meta = json.load(f)
        if meta.get("smoke") and not self.allow_smoke:
            raise RuntimeError("Veri SMOKE (sentetik) veri; gerçek eğitim reddedildi. Gerçek veri sağlayın (README → Veri sağlama).")
        if not meta.get("min_total_tokens_ok", False) and not self.allow_smoke:
            raise RuntimeError(f"Veri yetersiz: {meta.get('warning', 'min_total_tokens altında')}")
        if meta["vocab_size"] != self.mcfg.vocab_size:
            raise RuntimeError(f"Model/tokenizer uyuşmazlığı: veri vocab={meta['vocab_size']} model vocab={self.mcfg.vocab_size}")
        tok = ALTUNTokenizer.load(os.path.join(self.data_dir, "tokenizer"))
        if tok.vocab_size != self.mcfg.vocab_size:
            raise RuntimeError("Tokenizer vocab boyutu model ile uyuşmuyor")
        self.tokenizer, self.meta = tok, meta

    def _setup(self) -> None:
        t, m = self.tcfg, self.mcfg
        self._check_data()
        self.seq_len = m.context_length
        self.train_ds = TokenDataset(os.path.join(self.data_dir, "train.bin"), self.seq_len)
        self.val_ds = TokenDataset(os.path.join(self.data_dir, "val.bin"), self.seq_len)
        per_micro = t.micro_batch_size * self.seq_len * self.world
        self.accum = max(1, round(t.tokens_per_step / per_micro))
        self.tokens_per_step = self.accum * per_micro

        torch.manual_seed(t.seed)
        np.random.seed(t.seed)
        torch.backends.cuda.matmul.allow_tf32 = False  # T4'te TF32 yok; açıkça kapalı
        model = ALTUNModel(m)
        model.grad_checkpointing = t.grad_checkpointing
        self.n_params = model.num_parameters()
        model.to(self.device)
        self.raw = model
        fwd = self._maybe_compile(model)
        self.model = DDP(fwd, device_ids=[self.local]) if self.world > 1 and self.use_cuda else (DDP(fwd) if self.world > 1 else fwd)
        self.opt = build_optimizer(self.raw, t.peak_lr, t.weight_decay, t.beta1, t.beta2, self.device.type)
        self.scaler = make_scaler(t.fp16 and self.use_cuda)
        self.step, self.tokens, self.elapsed_base, self.best_val, self.skipped = 0, 0, 0.0, float("inf"), 0

        ck = None
        if self.resume is False:
            if self.main:
                n = clear_checkpoints(self.ckpt_dir)
                print(f"[resume] SIFIRDAN eğitim: {n} eski checkpoint dosyası temizlendi.")
        else:
            path = find_latest_checkpoint(self.ckpt_dir)
            if path:
                ck = load_checkpoint(path, map_location="cpu")
                if ck["model_config"] != m.to_dict():
                    raise RuntimeError("Checkpoint model konfigürasyonu mevcut config ile uyuşmuyor. "
                                       "Yeni eğitim için RESUME_FROM_CHECKPOINT=False kullanın.")
                if ck.get("tokenizer_hash") and ck["tokenizer_hash"] != self.meta.get("tokenizer_hash"):
                    raise RuntimeError("Checkpoint farklı bir tokenizer ile eğitilmiş (hash uyuşmuyor). Aynı veri kaynakları/ayarlarla "
                                       "yeniden hazırlayın ya da RESUME_FROM_CHECKPOINT=False ile sıfırdan başlayın.")
                self.raw.load_state_dict(ck["model"])
                self.opt.load_state_dict(ck["optimizer"])
                self.scaler.load_state_dict(ck["scaler"])
                self.step, self.tokens = ck["step"], ck["tokens"]
                self.elapsed_base, self.best_val = ck["elapsed_s"], ck.get("best_val", float("inf"))
                self.skipped = ck.get("skipped", 0)
                restore_rng_state(ck["rng"])
                if self.main:
                    print(f"[resume] {path} yüklendi → step={self.step}, geçen süre={fmt_time(self.elapsed_base)}")
            elif self.resume is True and self.main:
                print("[resume] Checkpoint bulunamadı, sıfırdan başlanıyor.")
        if ck is None and self.init_from:
            self._init_from_previous()
        self.resumed = ck is not None
        D.barrier()
        self.logger = MetricsLogger(self.log_dir, resume=self.resumed) if self.main else None

    # ------------------------------------------------- önceki modelden devam
    def _init_from_previous(self) -> None:
        """Önceki final_model ağırlıklarıyla başla (yeni optimizer/zamanlayıcı; tokenizer AYNI olmalı)."""
        from altun.checkpoint import load_final_model
        prev, prev_cfg = load_final_model(self.init_from, device="cpu")
        if prev_cfg.to_dict() != self.mcfg.to_dict():
            raise RuntimeError(f"init-from model config'i mevcut config ile uyuşmuyor.\n önceki={prev_cfg.to_dict()}\n şimdiki={self.mcfg.to_dict()}")
        tok_prev = ALTUNTokenizer.load(os.path.join(self.init_from, "tokenizer"))
        h_prev = hashlib.sha1(json.dumps(tok_prev.merges).encode()).hexdigest()
        if h_prev != self.meta.get("tokenizer_hash"):
            raise RuntimeError("init-from tokenizer'ı ile bu veri hazırlığının tokenizer'ı farklı: veriyi `--tokenizer-dir` ile aynı tokenizer'dan üretin.")
        self.raw.load_state_dict(prev.state_dict())
        del prev
        if self.main:
            print(f"[init] Önceki model ağırlıkları yüklendi: {self.init_from} (optimizer/LR programı yeniden başlıyor)", flush=True)

    # ------------------------------------------------------- torch.compile
    def _maybe_compile(self, model):
        """auto: eager ve compile hızını bu donanımda ÖLÇ; hata varsa veya kazanç <%8 ise eager'da kal."""
        mode = self.tcfg.compile
        self.compile_info = {"mode": mode, "used": False}
        if mode == "off" or not self.use_cuda or not hasattr(torch, "compile"):
            return model
        t = self.tcfg
        fail = torch.zeros(1, device=self.device)
        speed = torch.zeros(1, device=self.device)
        comp, err = None, None
        try:
            model.train()
            import torch._dynamo as _dy
            _dy.config.suppress_errors = True  # eğitim sırasında derleme hatası olursa o parça eager çalışır
            comp = torch.compile(model)
            x = torch.randint(0, self.mcfg.vocab_size, (t.micro_batch_size, self.mcfg.context_length), device=self.device)

            def bench(m, n):
                ts = []
                for _ in range(n):
                    torch.cuda.synchronize()
                    t0 = time.time()
                    with torch.autocast("cuda", dtype=torch.float16, enabled=t.fp16):
                        loss = m(x, targets=x)["loss"]
                    loss.backward()
                    torch.cuda.synchronize()
                    ts.append(time.time() - t0)
                model.zero_grad(set_to_none=True)
                return sum(ts[2:]) / len(ts[2:])

            t_eager = bench(model, 4)
            t_comp = bench(comp, 5)  # ilk çağrılar derleme süresini içerir, ölçümden hariç
            speed[0] = t_eager / t_comp
            self.compile_info.update({"t_eager_s": round(t_eager, 3), "t_compiled_s": round(t_comp, 3)})
        except Exception as e:  # noqa: BLE001
            fail[0] = 1.0
            err = f"{type(e).__name__}: {str(e)[:200]}"
        model.zero_grad(set_to_none=True)
        torch.cuda.empty_cache()
        fail = D.all_reduce_max(fail)
        speed = D.all_reduce_mean(speed)
        use = fail.item() == 0 and (mode == "on" or speed.item() >= 1.08)
        self.compile_info.update({"used": bool(use), "speedup": round(speed.item(), 3), "error": err})
        if self.main:
            print(f"[compile] mod={mode} | hata={'var: ' + str(err) if fail.item() else 'yok'} | hız oranı (eager/compile)={speed.item():.2f} "
                  f"→ {'compile KULLANILIYOR' if use else 'eager (compile kullanılmıyor)'}", flush=True)
        return comp if use else model

    # --------------------------------------------------------------- data
    def _make_step(self, step: int):
        out = []
        for mi in range(self.accum):
            x, y = self.train_ds.get_batch(self.tcfg.micro_batch_size, self.tcfg.seed, step, mi, self.rank)
            x, y = torch.from_numpy(x), torch.from_numpy(y)
            if self.use_cuda:
                x, y = x.pin_memory(), y.pin_memory()
            out.append((x, y))
        return out

    @torch.no_grad()
    def evaluate(self) -> float:
        self.raw.eval()
        n = max(1, self.tcfg.eval_batches // self.world)
        tot = torch.zeros(1, device=self.device)
        for i in range(n):
            x, y = self.val_ds.get_batch(self.tcfg.micro_batch_size, 777, i, self.rank)
            x, y = torch.from_numpy(x).to(self.device), torch.from_numpy(y).to(self.device)
            with torch.autocast(self.device.type, dtype=torch.float16, enabled=self.tcfg.fp16 and self.use_cuda):
                tot += self.raw(x, targets=y)["loss"].float()
        tot = D.all_reduce_mean(tot / n)
        self.raw.train()
        return float(tot.item())

    def _save(self, complete: bool = False) -> None:
        if self.main:
            state = {"model": self.raw.state_dict(), "optimizer": self.opt.state_dict(), "scaler": self.scaler.state_dict(),
                     "step": self.step, "tokens": self.tokens, "elapsed_s": self.elapsed(), "best_val": self.best_val,
                     "skipped": self.skipped, "rng": capture_rng_state(), "model_config": self.mcfg.to_dict(),
                     "train_config": self.tcfg.to_dict(), "world_size": self.world, "complete": complete,
                     "data_position": {"step": self.step, "seed": self.tcfg.seed},
                     "tokenizer_hash": self.meta.get("tokenizer_hash")}
            t0 = time.time()
            p = save_checkpoint(self.ckpt_dir, state, self.step, keep=self.tcfg.max_checkpoints, min_free_gb=self.tcfg.min_free_disk_gb)
            print(f"[checkpoint] kaydedildi: {os.path.basename(p)} ({time.time() - t0:.1f}s)", flush=True)
        D.barrier()

    def elapsed(self) -> float:
        return self.elapsed_base + (time.time() - self.t0)

    # ---------------------------------------------------------------- run
    def run(self, duration_hours: Optional[float] = None) -> Dict:
        t = self.tcfg
        self._setup()
        target_s = (duration_hours if duration_hours is not None else t.duration_hours) * 3600
        signal.signal(signal.SIGTERM, lambda *_: setattr(self, "_sig", True))
        if self.main:
            print(f"[train] params={self.n_params:,} | world={self.world} | micro_bs={t.micro_batch_size} | accum={self.accum} | "
                  f"tokens/step={self.tokens_per_step:,} | hedef süre={fmt_time(target_s)} | tahmini VRAM/GPU≈"
                  f"{estimate_vram_gb(self.mcfg, t.micro_batch_size, t.grad_checkpointing):.1f} GB (tahmin)", flush=True)
        already_done = self.resumed and self.elapsed_base >= target_s
        self.t0 = time.time()
        last_ckpt_t, last_sys_t = time.time(), 0.0
        pf = Prefetcher(self._make_step, self.step, t.prefetch_batches)
        self.model.train()
        win_t, win_loss, win_n, win_data, win_tok = time.time(), torch.zeros(1, device=self.device), 0, 0.0, 0
        speeds, last_gn, last_val = [], float("nan"), float("nan")
        consecutive_skips = 0
        complete = False
        if already_done:
            self.stop_reason = "hedef süre önceki oturumlarda zaten tamamlanmış"
            complete = True
        while not already_done:
            ctl = torch.zeros(4, device=self.device)
            if self.main:
                el = self.elapsed()
                stop = el >= target_s or (self.max_steps is not None and self.step >= self.max_steps) or self._sig
                ctl[0] = min(el / target_s, 1.0)
                ctl[1] = 1.0 if stop else 0.0
                ctl[2] = 1.0 if (time.time() - last_ckpt_t) >= t.checkpoint_interval_minutes * 60 else 0.0
                ctl[3] = 1.0 if (self.step > 0 and self.step % t.eval_interval_steps == 0) else 0.0
            D.broadcast(ctl, 0)
            progress, stop, do_ckpt, do_eval = ctl.tolist()
            if stop:
                if self.main:
                    self.stop_reason = ("SIGTERM alındı" if self._sig else
                                        "max_steps" if (self.max_steps is not None and self.step >= self.max_steps) else "hedef süre doldu")
                complete = not self._sig
                break

            lr = lr_at(self.step, progress, t.peak_lr, t.min_lr_ratio, t.warmup_steps)
            for g in self.opt.param_groups:
                g["lr"] = lr
            ts = time.time()
            s_idx, micro = pf.next()
            win_data += pf.wait_time
            step_loss = torch.zeros(1, device=self.device)
            for mi, (x, y) in enumerate(micro):
                x, y = x.to(self.device, non_blocking=True), y.to(self.device, non_blocking=True)
                sync_ctx = self.model.no_sync() if (self.world > 1 and mi < self.accum - 1) else _null()
                with sync_ctx:
                    with torch.autocast(self.device.type, dtype=torch.float16, enabled=t.fp16 and self.use_cuda):
                        loss = self.model(x, targets=y)["loss"] / self.accum
                    self.scaler.scale(loss).backward()
                step_loss += loss.detach().float()
            self.scaler.unscale_(self.opt)
            gn = torch.nn.utils.clip_grad_norm_(self.raw.parameters(), t.grad_clip)
            scale_before = self.scaler.get_scale() if self.scaler.is_enabled() else 1.0
            self.scaler.step(self.opt)
            self.scaler.update()
            self.opt.zero_grad(set_to_none=True)
            skipped = self.scaler.is_enabled() and self.scaler.get_scale() < scale_before
            if skipped:
                self.skipped += 1
                consecutive_skips += 1
            else:
                consecutive_skips = 0
            if consecutive_skips > 100 or not torch.isfinite(step_loss).all() and not self.scaler.is_enabled():
                raise RuntimeError("Eğitim kararsız (ardışık NaN/Inf). Son geçerli checkpoint korunuyor; lr düşürüp resume edin.")
            self.step += 1
            self.tokens += self.tokens_per_step
            win_loss += D.all_reduce_mean(step_loss.clone()) if self.world > 1 else step_loss
            win_n += 1
            win_tok += self.tokens_per_step
            last_gn = float(gn)  # senkron nokta (step süresi doğru ölçülsün)
            step_time = time.time() - ts

            if self.step % t.log_interval_steps == 0 and self.main:
                now = time.time()
                dt = max(now - win_t, 1e-9)
                tps = win_tok / dt
                speeds.append(tps)
                row = {"step": self.step, "tokens": self.tokens, "train_loss": round(float(win_loss) / win_n, 5), "lr": lr,
                       "grad_norm": round(last_gn, 4), "tokens_per_sec": round(tps, 1), "steps_per_sec": round(win_n / dt, 4),
                       "data_time": round(win_data / win_n, 4), "fwd_bwd_time": round(max(dt / win_n - win_data / win_n, 0), 4),
                       "elapsed_s": round(self.elapsed(), 1), "eta_s": round(max(target_s - self.elapsed(), 0), 1),
                       "skipped_steps": self.skipped}
                self.logger.log_train(row)
                print(f"step {self.step:>7} | loss {row['train_loss']:.4f} | lr {lr:.2e} | gn {last_gn:.2f} | "
                      f"{tps / 1000:.1f}k tok/s | data {row['data_time']:.3f}s | elapsed {fmt_time(row['elapsed_s'])} | "
                      f"kalan {fmt_time(row['eta_s'])} | tokens {self.tokens / 1e6:.1f}M", flush=True)
                win_t, win_loss, win_n, win_data, win_tok = now, torch.zeros(1, device=self.device), 0, 0.0, 0
            elif self.step % t.log_interval_steps == 0:
                win_t, win_loss, win_n, win_data, win_tok = time.time(), torch.zeros(1, device=self.device), 0, 0.0, 0

            if do_eval:
                v = self.evaluate()
                last_val = v
                self.best_val = min(self.best_val, v)
                if self.main:
                    self.logger.log_train({"step": self.step, "tokens": self.tokens, "val_loss": round(v, 5)})
                    print(f"[eval] step {self.step} | val_loss {v:.4f} | ppl {np.exp(min(v, 50)):.1f}", flush=True)
            if self.main and time.time() - last_sys_t >= t.system_metrics_interval_seconds:
                self.logger.log_system(self.step, sample_system(torch))
                last_sys_t = time.time()
            if do_ckpt:
                self._save()
                last_ckpt_t = time.time()
        pf.close()

        # ------------------------------------------------------------ bitiş
        if not already_done:
            if self.step > 0:
                v = self.evaluate()
                last_val = v
                self.best_val = min(self.best_val, v)
                if self.main:
                    self.logger.log_train({"step": self.step, "tokens": self.tokens, "val_loss": round(v, 5)})
            self._save(complete=complete)
        result = {"step": self.step, "tokens": self.tokens, "complete": complete, "stop_reason": self.stop_reason,
                  "final_val_loss": last_val, "best_val_loss": self.best_val, "elapsed_s": self.elapsed(),
                  "mean_tokens_per_sec_measured": float(np.mean(speeds)) if speeds else None,
                  "max_mem_gb": torch.cuda.max_memory_allocated(self.device) / 1e9 if self.use_cuda else None,
                  "world_size": self.world, "accum": self.accum, "tokens_per_step": self.tokens_per_step,
                  "n_params": self.n_params, "skipped_steps": self.skipped}
        if self.main and complete:
            final_dir = os.path.join(self.results, "final_model")
            export_final_model(self.raw, self.mcfg, final_dir, self.tokenizer, extra={"train_tokens": self.tokens, "train_steps": self.step})
            print(f"[export] Final model kaydedildi: {final_dir}", flush=True)
        if self.main:
            try:
                make_plots(self.log_dir, os.path.join(self.results, "plots"))
            except Exception as e:
                print(f"[plots] UYARI: grafik üretilemedi: {e}")
            self._write_report(result, target_s)
        D.barrier()
        return result

    def _write_report(self, r: Dict, target_s: float) -> None:
        t, m = self.tcfg, self.mcfg
        est = None
        ap = os.path.join(self.results, "reports", "autotune.json")
        if os.path.isfile(ap):
            with open(ap, encoding="utf-8") as f:
                est = json.load(f)
        L = ["# ALTUN 1 Eğitim Raporu", "",
             f"- Durum: {'TAMAMLANDI' if r['complete'] else 'YARIM (resume edilebilir)'} — neden: {r['stop_reason']}",
             f"- Adım: {r['step']:,} | İşlenen token: {r['tokens']:,} | Geçen süre: {fmt_time(r['elapsed_s'])} / hedef {fmt_time(target_s)}",
             f"- Final val loss: {r['final_val_loss']:.4f} | En iyi val loss: {r['best_val_loss']:.4f}",
             f"- Atlanan (inf/NaN) adım: {r['skipped_steps']}", "", "## Model (gerçek sayılar)", "",
             f"- Toplam parametre: {r['n_params']:,}", f"- Katman: {m.n_layers} | Hidden: {m.d_model} | Head: {m.n_heads} | KV head: {m.n_kv_heads}",
             f"- Intermediate: {m.intermediate_size} | Context: {m.context_length} | Vocab: {m.vocab_size}", "",
             "## Eğitim ayarları", "",
             f"- Optimizer: AdamW (β1={t.beta1}, β2={t.beta2}, wd={t.weight_decay}), peak LR {t.peak_lr}, min LR oranı {t.min_lr_ratio}",
             f"- Warmup: {t.warmup_steps} adım, cosine (süreye bağlı) | Grad clip: {t.grad_clip} | FP16 AMP: {t.fp16}",
             f"- GPU sayısı (DDP world): {r['world_size']} | micro batch/GPU: {t.micro_batch_size} | accumulation: {r['accum']} | "
             f"global token/adım: {r['tokens_per_step']:,} | grad checkpointing: {t.grad_checkpointing}", "",
             "## Performans", "",
             f"- **ÖLÇÜLEN** ortalama tokens/s (global): {r['mean_tokens_per_sec_measured']}",
             f"- **ÖLÇÜLEN** tepe VRAM (GPU0, torch allocated): {r['max_mem_gb']}",
             f"- **TAHMİN** VRAM/GPU: {estimate_vram_gb(m, t.micro_batch_size, t.grad_checkpointing):.1f} GB",
             f"- Ön-ölçüm (autotune, kısa deneme): {json.dumps(est) if est else 'yapılmadı'}",
             f"- torch.compile: {json.dumps(getattr(self, 'compile_info', {}))}",
             f"- Önceki modelden devam: {self.init_from or 'hayır (sıfırdan)'}"]
        os.makedirs(os.path.join(self.results, "reports"), exist_ok=True)
        with open(os.path.join(self.results, "reports", "training_report.md"), "w", encoding="utf-8") as f:
            f.write("\n".join(L) + "\n")
        with open(os.path.join(self.results, "reports", "training_result.json"), "w", encoding="utf-8") as f:
            json.dump(r, f, indent=2, default=str)


class _null:
    def __enter__(self):
        return self

    def __exit__(self, *a):
        return False
