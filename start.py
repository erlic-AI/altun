"""ALTUN 1 — tek komutla eğitim başlatıcı (Kaggle, Colab, kendi makineniz; en az 1 NVIDIA GPU gerekir).

Örnekler:
  python start.py                          # yayınlanmış ağırlıklardan (Hugging Face) devam, 7 saat
  python start.py --hours 10               # süreyi değiştir
  python start.py --continue-from none     # SIFIRDAN eğit (--peak-lr verilmezse config'teki 6e-4)
  python start.py --continue-from /yol/final_model   # yerel bir final_model klasöründen devam
  python start.py --smoke                  # çok kısa duman testi (gerçek eğitim DEĞİL)
Kaggle notebook hücresi:  !git clone https://github.com/erlic-AI/altun ALTUN-1 && cd ALTUN-1 && python start.py
Önkoşul: pip install -r requirements.txt (Kaggle'da hazır), Internet açık (veri + ağırlık indirme)."""
import argparse
import os
import subprocess
import sys

ROOT = os.path.dirname(os.path.abspath(__file__))


def fail(msg):
    print("\n" + "=" * 70 + "\n[ALTUN-1 HATA] " + msg + "\n" + "=" * 70)
    raise SystemExit(msg)


def download_from_hf(repo_id: str, dest: str) -> str:
    try:
        from huggingface_hub import snapshot_download
    except ImportError:
        subprocess.run([sys.executable, "-m", "pip", "install", "-q", "huggingface_hub"], check=False)
        from huggingface_hub import snapshot_download
    snapshot_download(repo_id=repo_id, local_dir=dest, allow_patterns=["model.safetensors", "config.json", "tokenizer/*"])
    for f in ("model.safetensors", "config.json", "tokenizer"):
        if not os.path.exists(os.path.join(dest, f)):
            fail(f"İndirilen modelde eksik: {f}")
    return dest


def main():
    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--hours", type=float, default=7.0, help="eğitim süresi (saat); veri hazırlığı ve değerlendirme buna dahil değildir")
    ap.add_argument("--continue-from", default="hf", help="hf | input (Kaggle Input'taki önceki çıktı) | none (sıfırdan) | <final_model klasörü>")
    ap.add_argument("--hf-repo", default="erlic/ALTUN-1")
    ap.add_argument("--peak-lr", type=float, default=None, help="varsayılan: devam eğitiminde 2.5e-4, sıfırdan eğitimde config değeri")
    ap.add_argument("--resume", default="auto", help="auto | false (temiz başla)")
    ap.add_argument("--output-dir", default=None, help="varsayılan: Kaggle'da /kaggle/working/ALTUN-1-RESULTS, aksi halde ./ALTUN-1-RESULTS")
    ap.add_argument("--smoke", action="store_true")
    a = ap.parse_args()

    sys.path.insert(0, ROOT)
    os.chdir(ROOT)
    from scripts import run_training as RT

    init_from, from_input = None, False
    if not a.smoke:
        if a.continue_from == "hf":
            print(f"[ALTUN-1] Ağırlıklar Hugging Face'ten indiriliyor: {a.hf_repo}")
            try:
                init_from = download_from_hf(a.hf_repo, os.path.join(ROOT, "ALTUN-1-PREV", "final_model"))
            except SystemExit:
                raise
            except Exception as e:  # noqa: BLE001
                fail(f"HF indirme başarısız: {e}\nİnternet açık mı? Alternatif: --continue-from none")
        elif a.continue_from == "input":
            from_input = True
        elif a.continue_from != "none":
            init_from = a.continue_from
            if not os.path.isdir(init_from):
                fail(f"final_model klasörü yok: {init_from}")
    continuing = bool(init_from) or from_input
    rc = RT.run({"TRAINING_DURATION_HOURS": a.hours, "RESUME_FROM_CHECKPOINT": False if str(a.resume).lower() == "false" else "auto",
                 "OUTPUT_DIR": a.output_dir or RT.FIXED_RESULTS, "SMOKE_TEST": a.smoke, "INIT_FROM": init_from,
                 "INIT_FROM_PREVIOUS": from_input,
                 "PEAK_LR": a.peak_lr if a.peak_lr is not None else (2.5e-4 if continuing else None)})
    print("\nÇıkış kodu:", rc, "(0 = başarılı)")
    sys.exit(rc)


if __name__ == "__main__":
    main()
