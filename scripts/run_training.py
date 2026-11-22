"""ALTUN 1 uçtan uca orkestrasyon: preflight → veri → autotune → eğitim (DDP) → değerlendirme → paketleme.
start.py bunu çağırır (Kaggle'da veya herhangi bir GPU makinesinde). Çıktılar HER ZAMAN tek sabit klasöre yazılır."""
from __future__ import annotations

import glob
import json
import os
import shutil
import subprocess
import sys
import time
import zipfile
from typing import Dict, Optional

PROJECT_ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
sys.path.insert(0, PROJECT_ROOT)

FIXED_RESULTS = ("/kaggle/working/ALTUN-1-RESULTS" if os.path.isdir("/kaggle/working")
                 else os.path.join(PROJECT_ROOT, "ALTUN-1-RESULTS"))
SUBDIRS = ["final_model", "checkpoints", "evaluations", "logs", "reports", "plots"]
ZIP_EXCLUDE_DIRS = {"checkpoints"}  # büyük ara dosyalar ZIP'e girmez (final_model girer)


def _run(cmd, cwd=PROJECT_ROOT, env_extra=None) -> int:
    env = {**os.environ, "PYTHONPATH": PROJECT_ROOT, "PYTHONUNBUFFERED": "1", **(env_extra or {})}
    p = subprocess.Popen(cmd, cwd=cwd, env=env, stdout=subprocess.PIPE, stderr=subprocess.STDOUT, text=True, bufsize=1)
    for line in p.stdout:
        print(line, end="", flush=True)
    return p.wait()


def import_previous_run(results_dir: str) -> Optional[str]:
    """Yeni Kaggle oturumunda önceki oturumun çıktısı Input olarak eklenmişse en son geçerli checkpoint'i içeri al."""
    from altun.checkpoint import find_latest_checkpoint
    ck_dir = os.path.join(results_dir, "checkpoints")
    if find_latest_checkpoint(ck_dir):
        return None
    cands = []
    for pat in ("/kaggle/input/*/ALTUN-1-RESULTS/checkpoints", "/kaggle/input/*/*/ALTUN-1-RESULTS/checkpoints",
                "/kaggle/input/*/checkpoints", "/kaggle/input/*/*/checkpoints"):
        cands += glob.glob(pat)
    best = None
    for d in cands:
        p = find_latest_checkpoint(d)
        if p and (best is None or p > best):
            best = p
    if not best:
        return None
    os.makedirs(ck_dir, exist_ok=True)
    for src in (best, best[:-3] + ".json"):
        shutil.copy2(src, os.path.join(ck_dir, os.path.basename(src)))
    # önceki loglar (devam eden grafikler için)
    prev_logs = os.path.join(os.path.dirname(os.path.dirname(best)), "logs")
    if os.path.isdir(prev_logs):
        os.makedirs(os.path.join(results_dir, "logs"), exist_ok=True)
        for f in os.listdir(prev_logs):
            dst = os.path.join(results_dir, "logs", f)
            if not os.path.exists(dst):
                shutil.copy2(os.path.join(prev_logs, f), dst)
    print(f"[resume] Önceki oturumdan checkpoint alındı: {best}")
    return best


def find_previous_model() -> Optional[str]:
    """Önceki notebook çıktısındaki (Input) en yeni geçerli final_model klasörü (model.safetensors + config + tokenizer)."""
    cands = []
    for pat in ("/kaggle/input/*/ALTUN-1-RESULTS/final_model", "/kaggle/input/*/*/ALTUN-1-RESULTS/final_model",
                "/kaggle/input/*/*/*/ALTUN-1-RESULTS/final_model", "/kaggle/input/*/final_model", "/kaggle/input/*/*/final_model"):
        cands += glob.glob(pat)
    ok = [d for d in cands if all(os.path.exists(os.path.join(d, f)) for f in ("model.safetensors", "config.json", "tokenizer"))]
    return max(ok, key=lambda d: os.path.getmtime(os.path.join(d, "model.safetensors"))) if ok else None


def build_zip(results_dir: str) -> str:
    zip_path = os.path.join(results_dir, "ALTUN-1-RESULTS.zip")
    tmp = zip_path + ".tmp"
    with zipfile.ZipFile(tmp, "w", zipfile.ZIP_DEFLATED, compresslevel=6) as z:
        for dp, dn, fns in os.walk(results_dir):
            rel_dir = os.path.relpath(dp, results_dir)
            top = rel_dir.split(os.sep)[0]
            if top in ZIP_EXCLUDE_DIRS:
                dn[:] = []
                continue
            for fn in fns:
                if fn.endswith(".tmp") or os.path.join(dp, fn) in (zip_path, tmp):
                    continue
                z.write(os.path.join(dp, fn), os.path.normpath(os.path.join("ALTUN-1-RESULTS", rel_dir, fn)))
        for name in ("training_config.yaml", "model_config.yaml", "data_config.yaml"):
            z.write(os.path.join(PROJECT_ROOT, "config", name), f"ALTUN-1-RESULTS/config/{name}")
    with zipfile.ZipFile(tmp) as z:
        bad = z.testzip()
        if bad:
            raise RuntimeError(f"ZIP bozuk: {bad}")
    os.replace(tmp, zip_path)
    parent = os.path.dirname(results_dir.rstrip("/"))
    mirror = os.path.join(parent, "ALTUN-1-RESULTS.zip")
    if os.path.abspath(mirror) != os.path.abspath(zip_path):
        try:
            if os.path.exists(mirror):
                os.remove(mirror)
            os.link(zip_path, mirror)
        except OSError:
            shutil.copy2(zip_path, mirror)
    return zip_path


def copy_data_reports(results_dir: str) -> None:
    src = os.path.join(PROJECT_ROOT, "outputs", "data")
    for name in ("data_quality_report.md", "data_quality_report.json", "provenance.json"):
        if os.path.isfile(os.path.join(src, name)):
            shutil.copy2(os.path.join(src, name), os.path.join(results_dir, "reports", name))


def run(settings: Dict) -> int:
    """settings: DRIVE_FILE_ID (kullanılmaz burada), TRAINING_DURATION_HOURS, RESUME_FROM_CHECKPOINT, OUTPUT_DIR (+ opsiyonel)."""
    t_all = time.time()
    results = settings.get("OUTPUT_DIR") or FIXED_RESULTS
    hours = float(settings.get("TRAINING_DURATION_HOURS", 12))
    resume = settings.get("RESUME_FROM_CHECKPOINT", "auto")
    smoke = bool(settings.get("SMOKE_TEST", False))
    preset = settings.get("PRESET") or ("altun1-smoke" if smoke else None)
    for d in SUBDIRS:
        os.makedirs(os.path.join(results, d), exist_ok=True)
    print(f"[ALTUN-1] Sonuç klasörü: {results}")

    from altun.checkpoint import find_latest_checkpoint
    init_from = settings.get("INIT_FROM")  # elle yol verilebilir
    own_ckpt = bool(find_latest_checkpoint(os.path.join(results, "checkpoints")))
    if not init_from and not smoke and settings.get("INIT_FROM_PREVIOUS", False) and not own_ckpt:
        init_from = find_previous_model()
        if not init_from:
            print("\n[ALTUN-1] INIT_FROM_PREVIOUS=True ama önceki final_model bulunamadı (/kaggle/input altında). "
                  "Sağ panelden önceki notebook çıktısını 'Add Input' ile ekleyin. Eğitim BAŞLATILMADI.")
            return 6
    if init_from:
        print(f"[ALTUN-1] Önceki modelden devam edilecek: {init_from}")
    elif resume is not False:
        import_previous_run(results)

    # 0) Veri: yoksa otomatik topla (Internet açık olmalı). Elle veri sağlamaya gerek yok.
    if not smoke and settings.get("AUTO_FETCH_DATA", True):
        from data.prepare_data import discover_sources
        from altun.config import load_data_config
        have = discover_sources(load_data_config()["data_roots"])
        if not have or settings.get("FORCE_FETCH", False):
            print("[ALTUN-1] Eğitim verisi toplanıyor (açık lisanslı kaynaklar; birkaç on dakika sürebilir) ...")
            from scripts.fetch_open_data import ensure_data
            rep = ensure_data(os.path.join(PROJECT_ROOT, "raw_data"), report_path=os.path.join(results, "reports", "fetch_report.json"))
            if not rep["ok"]:
                print("\n[ALTUN-1] Veri indirilemedi → eğitim BAŞLATILMADI.\n"
                      f"Hatalar: {json.dumps(rep['failed'], ensure_ascii=False, indent=1)}\n"
                      "Notebook ayarlarında Internet = ON olmalı. Alternatif: README bölüm 1 (kendi verinizi Dataset olarak ekleyin).")
                return 3
        else:
            print(f"[ALTUN-1] {len(have)} veri kaynağı zaten mevcut, indirme atlandı.")

    # 1) Preflight (donanım, bağımlılık, config, DDP, autotune)
    pf = ["-m", "scripts.preflight", "--results-dir", results, "--duration-hours", str(hours)]
    if preset:
        pf += ["--preset", preset]
    if smoke:
        pf += ["--no-autotune", "--skip-ddp", "--no-data-check"] + ([] if settings.get("REQUIRE_GPU", False) else ["--allow-cpu"])
    if _run([sys.executable] + pf) != 0:
        print("\n[ALTUN-1] Preflight başarısız → eğitim BAŞLATILMADI. Yukarıdaki hata mesajlarını düzeltin.")
        return 2
    with open(os.path.join(results, "reports", "preflight.json"), encoding="utf-8") as f:
        world = json.load(f).get("world_size", 1)

    # 1b) Birim/duman testleri (gerçek donanımda). Başarısızlık UYARIDIR: kodda hata varsa eğitim zaten ilk adımda net hata verir.
    print("[ALTUN-1] Birim testleri çalışıyor ...")
    t = subprocess.run([sys.executable, "-m", "pytest", "tests", "-q", "--tb=short", "-p", "no:cacheprovider"], cwd=PROJECT_ROOT,
                       capture_output=True, text=True, env={**os.environ, "PYTHONPATH": PROJECT_ROOT})
    with open(os.path.join(results, "reports", "unit_tests_pre_training.txt"), "w", encoding="utf-8") as f:
        f.write(t.stdout + t.stderr)
    print("   " + (t.stdout.strip().splitlines() or ["(çıktı yok)"])[-1])
    if t.returncode != 0:
        print("   !!! UYARI: bazı birim testleri BAŞARISIZ. Ayrıntı: reports/unit_tests_pre_training.txt\n"
              "   Eğitim yine de denenecek; hata kodda ise ilk adımlarda net bir hata görürsünüz.")
        print(t.stdout[-2500:])
    with open(os.path.join(results, "reports", "preflight.json"), encoding="utf-8") as f:
        world = json.load(f).get("world_size", 1)

    # 2) Veri
    budget = settings.get("DATA_BUDGET_TOKENS")
    if not budget and not smoke:  # ölçülen hıza göre veri bütçesi (tahmini token kapasitesi x 1.25)
        ap_json = os.path.join(results, "reports", "autotune.json")
        if os.path.isfile(ap_json):
            with open(ap_json, encoding="utf-8") as f:
                budget = int(json.load(f)["estimated_tokens_in_duration"] * 1.25 * hours / max(json.load(open(ap_json))["duration_hours"], 1e-9))
            settings["DATA_BUDGET_TOKENS"] = budget
            print(f"[ALTUN-1] Veri token bütçesi (tahmin tabanlı): {budget:,}")
    dp = [sys.executable, "-m", "data.prepare_data"] + (["--smoke", "--vocab-size", "512"] if smoke else [])
    if settings.get("DATA_BUDGET_TOKENS"):
        dp += ["--budget-tokens", str(settings["DATA_BUDGET_TOKENS"])]
    if init_from:
        dp += ["--tokenizer-dir", os.path.join(init_from, "tokenizer")]
    if smoke:
        dp += ["--work-dir", os.path.join(PROJECT_ROOT, "outputs", "data_smoke")]
    if _run(dp) != 0:
        print("\n[ALTUN-1] Veri hazırlama başarısız → eğitim BAŞLATILMADI (boş veriyle eğitim yapılmaz).")
        return 3
    copy_data_reports(results)
    data_dir = os.path.join(PROJECT_ROOT, "outputs", "data_smoke" if smoke else "data")

    # 3) Eğitim
    targs = ["-m", "training.train", "--results-dir", results, "--data-dir", data_dir, "--duration-hours", str(hours),
             "--resume", str(resume).lower()]
    if preset:
        targs += ["--preset", preset]
    if init_from and not own_ckpt:
        targs += ["--init-from", init_from]
        if settings.get("PEAK_LR"):
            targs += ["--peak-lr", str(settings["PEAK_LR"])]
    if settings.get("COMPILE"):
        targs += ["--compile", str(settings["COMPILE"])]
    if smoke:
        targs += ["--allow-smoke", "--max-steps", str(settings.get("MAX_STEPS", 30)), "--tokens-per-step", "4096", "--micro-batch-size", "2", "--vocab-size", "512"]
    if world > 1:
        cmd = [sys.executable, "-m", "torch.distributed.run", "--standalone", f"--nproc_per_node={world}"] + targs
    else:
        cmd = [sys.executable] + targs
    print(f"[ALTUN-1] Eğitim başlıyor (world={world}) ...")
    rc = _run(cmd)
    res_path = os.path.join(results, "reports", "training_result.json")
    if rc != 0:
        print(f"\n[ALTUN-1] Eğitim süreci hata ile bitti (kod {rc}). Son geçerli checkpoint korunuyor; "
              "hücreyi RESUME_FROM_CHECKPOINT='auto' ile yeniden çalıştırarak devam edebilirsiniz.")
        build_zip(results)
        return 4
    complete = False
    if os.path.isfile(res_path):
        with open(res_path, encoding="utf-8") as f:
            complete = json.load(f).get("complete", False)
    if not complete:
        print("\n[ALTUN-1] Eğitim tamamlanmadı (kesildi). Checkpoint korundu; aynı hücreyi 'auto' resume ile çalıştırın.")
        build_zip(results)
        return 5

    # 4) Değerlendirme (tek süreç, tek GPU)
    print("[ALTUN-1] Otomatik değerlendirme başlıyor ...")
    rc = _run([sys.executable, "-m", "evaluation.evaluate", "--model-dir", os.path.join(results, "final_model"),
               "--results-dir", results, "--data-dir", data_dir])
    if rc != 0:
        print(f"[ALTUN-1] UYARI: değerlendirme hata verdi (kod {rc}); model ve eğitim çıktıları yine de kaydedildi.")

    # 5) Paketleme
    zp = build_zip(results)
    print("\n" + "=" * 70)
    print("ALTUN 1 TAMAMLANDI")
    print(f"Sonuç klasörü : {results}")
    print(f"ZIP           : {zp}  ({os.path.getsize(zp) / 1e6:.1f} MB)")
    print(f"Toplam süre   : {(time.time() - t_all) / 3600:.2f} saat")
    print("Sonuçlar yukarıdaki klasörde; ZIP aynı klasörün içinde. Kaggle'da Output sekmesinden indirin.")
    print("=" * 70)
    return 0 if rc == 0 else 6


if __name__ == "__main__":
    import argparse
    ap = argparse.ArgumentParser()
    ap.add_argument("--hours", type=float, default=12)
    ap.add_argument("--resume", default="auto")
    ap.add_argument("--output-dir", default=None)
    ap.add_argument("--preset", default=None)
    ap.add_argument("--smoke", action="store_true")
    ap.add_argument("--max-steps", type=int, default=30)
    a = ap.parse_args()
    r = {"TRAINING_DURATION_HOURS": a.hours, "RESUME_FROM_CHECKPOINT": {"false": False, "true": True}.get(a.resume.lower(), "auto"),
         "OUTPUT_DIR": a.output_dir, "PRESET": a.preset, "SMOKE_TEST": a.smoke, "MAX_STEPS": a.max_steps}
    sys.exit(run(r))
