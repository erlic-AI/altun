# =====================================================================================
# ALTUN 1 — KAGGLE TEK HÜCRE BAŞLATICI   (Google Drive GEREKMEZ; kod GitHub'dan alınır)
# Kaggle notebook ayarları: Accelerator = GPU T4 x2, Internet = ON.  Bu hücrenin tamamını yapıştırıp çalıştırın.
# =====================================================================================
REPO_URL = "https://github.com/erlic-AI/ALTUN-1"
BRANCH = "main"
TRAINING_DURATION_HOURS = 7.0     # Kaggle oturumu ≤ ~12 sa: veri hazırlığı (~1-1.5 sa) + değerlendirme için pay bırakın
CONTINUE_FROM = "hf"              # "hf"   : Hugging Face'teki ALTUN-1 ağırlıklarından devam (Input eklemeden)
                                  # "input": notebook'a Input olarak eklenmiş önceki çıktıdan devam (final_model aranır)
                                  # "none" : SIFIRDAN eğit
HF_MODEL_REPO = "erlic/ALTUN-1"
PEAK_LR = 2.5e-4                  # devam eğitimi için; sıfırdan eğitimde config'teki 6e-4 kullanılır (None yapın)
RESUME_FROM_CHECKPOINT = "auto"   # "auto": aynı oturumda kaldığı yerden | False: temiz başla
OUTPUT_DIR = "/kaggle/working/ALTUN-1-RESULTS"   # değiştirmeyin: tek sabit sonuç klasörü
SMOKE_TEST = False                # True: çok kısa duman testi (gerçek eğitim DEĞİL)
# ------------------------------------------------------------------------------------
import os, subprocess, sys

WORK = "/kaggle/working"
PROJECT = f"{WORK}/ALTUN-1"


def fail(msg):
    print("\n" + "=" * 70 + "\n[ALTUN-1 HATA] " + msg + "\n" + "=" * 70)
    raise SystemExit(msg)


def sh(cmd, **kw):
    return subprocess.run(cmd, text=True, capture_output=True, **kw)


print("[1] Ortam / GPU")
if not os.path.isdir(WORK):
    fail("/kaggle/working yok: bu hücre Kaggle notebook içinde çalıştırılmalı.")
g = sh(["nvidia-smi", "--query-gpu=index,name,memory.total", "--format=csv,noheader"])
print(g.stdout.strip() or "GPU yok")
if g.returncode != 0 or not g.stdout.strip():
    fail("GPU bulunamadı: Accelerator → 'GPU T4 x2' seçip oturumu yeniden başlatın.")

print("[2] Kod GitHub'dan alınıyor:", REPO_URL)
if os.path.isdir(f"{PROJECT}/.git"):
    r = sh(["git", "-C", PROJECT, "pull", "--ff-only"])
else:
    r = sh(["git", "clone", "--depth", "1", "--branch", BRANCH, REPO_URL, PROJECT])
if r.returncode != 0:
    fail("git clone başarısız (Internet = ON mu? repo adı doğru mu?):\n" + r.stderr[-800:])
print(sh(["git", "-C", PROJECT, "log", "-1", "--format=%h %s"]).stdout.strip())

init_from = None
if CONTINUE_FROM == "hf" and not SMOKE_TEST:
    print(f"[3] Ağırlıklar Hugging Face'ten indiriliyor: {HF_MODEL_REPO}")
    try:
        from huggingface_hub import snapshot_download
    except ImportError:
        sh([sys.executable, "-m", "pip", "install", "-q", "huggingface_hub"])
        from huggingface_hub import snapshot_download
    prev = f"{WORK}/ALTUN-1-PREV/final_model"
    try:
        snapshot_download(repo_id=HF_MODEL_REPO, local_dir=prev, allow_patterns=["model.safetensors", "config.json", "tokenizer/*"])
    except Exception as e:  # noqa: BLE001
        fail(f"HF indirme başarısız: {e}\nCONTINUE_FROM='input' veya 'none' deneyin.")
    for f in ("model.safetensors", "config.json", "tokenizer"):
        if not os.path.exists(f"{prev}/{f}"):
            fail(f"İndirilen modelde eksik: {f}")
    init_from = prev
    print("   tamam:", prev)

sys.path.insert(0, PROJECT)
os.chdir(PROJECT)
from scripts.run_training import run  # noqa: E402

print("[4] Preflight, veri, eğitim, değerlendirme ve paketleme başlıyor")
rc = run({"TRAINING_DURATION_HOURS": TRAINING_DURATION_HOURS, "RESUME_FROM_CHECKPOINT": RESUME_FROM_CHECKPOINT,
          "OUTPUT_DIR": OUTPUT_DIR, "SMOKE_TEST": SMOKE_TEST, "INIT_FROM": init_from,
          "INIT_FROM_PREVIOUS": CONTINUE_FROM == "input", "PEAK_LR": PEAK_LR if CONTINUE_FROM != "none" else None})
print("\nÇıkış kodu:", rc, "(0 = başarılı)")
print("Sonuçlar:", OUTPUT_DIR, "| ZIP:", f"{WORK}/ALTUN-1-RESULTS.zip")
if rc != 0:
    raise SystemExit(f"ALTUN-1 işlemi kod {rc} ile bitti; yukarıdaki mesajlara bakın.")
