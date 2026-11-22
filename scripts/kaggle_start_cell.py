# =====================================================================================
# ALTUN 1 — KAGGLE TEK HÜCRE BAŞLATICI  (erlic AI)
# Bu hücrenin tamamını Kaggle notebook'una yapıştırıp çalıştırın.
# Gereksinimler: Accelerator = GPU T4 x2, Internet = ON (Drive ZIP + eğitim verisi otomatik indirilir).
# =====================================================================================
DRIVE_FILE_ID = "BURAYA_DRIVE_DOSYA_ID"   # drive.google.com/file/d/<BU_KISIM>/view  (klasör değil, ZIP dosyası!)
TRAINING_DURATION_HOURS = 10.0            # Kaggle oturum sınırı 12 saat; veri hazırlama+değerlendirme için pay bırakın
RESUME_FROM_CHECKPOINT = "auto"           # "auto": varsa devam et | False: sıfırdan (eski checkpoint'leri siler) | True: zorunlu devam
OUTPUT_DIR = "/kaggle/working/ALTUN-1-RESULTS"   # değiştirmeyin: tek sabit sonuç klasörü
# ------------------------------------------------------------------------------------
FORCE_REDOWNLOAD = False   # True: ZIP'i yeniden indir/aç
SMOKE_TEST = False         # True: sentetik veriyle çok kısa duman testi (gerçek eğitim DEĞİL)

import hashlib, importlib, json, os, re, shutil, subprocess, sys, time, urllib.request, urllib.error, zipfile

WORK = "/kaggle/working"
PROJECT = f"{WORK}/ALTUN-1"
ZIP_PATH = f"{WORK}/ALTUN-1.zip"
MARKER = f"{WORK}/.altun1_zip_marker.json"


def fail(msg):
    print("\n" + "=" * 70 + "\n[ALTUN-1 HATA] " + msg + "\n" + "=" * 70)
    raise SystemExit(msg)


def step(n, msg):
    print(f"\n[{n}] {msg}", flush=True)


# 1) Kaggle ortamı -------------------------------------------------------------------
step(1, "Kaggle ortamı kontrol ediliyor")
if not os.path.isdir("/kaggle/working"):
    fail("/kaggle/working yok: bu hücre Kaggle notebook içinde çalıştırılmalı.")
print("Python", sys.version.split()[0])

# 2) Donanım -------------------------------------------------------------------------
step(2, "GPU / CPU donanımı tespit ediliyor")
try:
    print(subprocess.run(["nvidia-smi", "--query-gpu=index,name,memory.total,compute_cap", "--format=csv"],
                         capture_output=True, text=True, timeout=20).stdout or "nvidia-smi çıktısı yok")
except Exception as e:
    print("nvidia-smi çalışmadı:", e)
    fail("GPU bulunamadı. Notebook ayarlarında Accelerator → 'GPU T4 x2' seçin ve oturumu yeniden başlatın.")
print("CPU çekirdek:", os.cpu_count())


# 3-6) ZIP indirme + doğrulama ---------------------------------------------------------
def _looks_like_html(path):
    with open(path, "rb") as f:
        head = f.read(512).lstrip().lower()
    return head.startswith(b"<!doctype html") or head.startswith(b"<html")


def _download_direct(file_id, dest):
    """Önce drive.usercontent.google.com (büyük dosya onay sayfasını confirm=t ile atlar)."""
    urls = [f"https://drive.usercontent.google.com/download?id={file_id}&export=download&confirm=t",
            f"https://drive.google.com/uc?export=download&id={file_id}&confirm=t"]
    last = None
    for url in urls:
        try:
            req = urllib.request.Request(url, headers={"User-Agent": "Mozilla/5.0"})
            with urllib.request.urlopen(req, timeout=60) as r, open(dest + ".part", "wb") as out:
                shutil.copyfileobj(r, out, 1 << 20)
            if _looks_like_html(dest + ".part"):
                html = open(dest + ".part", "rb").read(20000).decode("utf-8", "ignore")
                last = "Drive HTML sayfası döndürdü (kota aşıldı / paylaşım kapalı / onay sayfası)."
                if "Quota exceeded" in html or "quota" in html.lower():
                    last = "Drive indirme KOTASI aşılmış (24 saat sonra tekrar deneyin ya da dosyanın kopyasını kendi Drive'ınıza alın)."
                elif "Sign in" in html or "ServiceLogin" in html:
                    last = "Dosya herkese açık değil: Paylaş → 'Bağlantıya sahip herkes' (Görüntüleyen) yapın."
                os.remove(dest + ".part")
                continue
            os.replace(dest + ".part", dest)
            return None
        except (urllib.error.URLError, OSError) as e:
            last = f"{type(e).__name__}: {e}"
    return last


def _download_gdown(file_id, dest):
    try:
        importlib.import_module("gdown")
    except ImportError:
        subprocess.run([sys.executable, "-m", "pip", "install", "-q", "gdown"], capture_output=True)
    try:
        import gdown
        gdown.download(id=file_id, output=dest, quiet=False)
        return None if os.path.isfile(dest) else "gdown dosyayı indiremedi"
    except Exception as e:
        return f"gdown: {e}"


def _validate_zip(path):
    if not os.path.isfile(path) or os.path.getsize(path) < 50_000:
        return "ZIP çok küçük/yok"
    if not zipfile.is_zipfile(path):
        return "Dosya geçerli bir ZIP değil"
    with zipfile.ZipFile(path) as z:
        bad = z.testzip()
        if bad:
            return f"ZIP bütünlük hatası: {bad}"
        names = z.namelist()
        if not any(n.endswith("scripts/run_training.py") for n in names):
            return "ZIP içinde ALTUN-1/scripts/run_training.py yok (yanlış dosya?)"
    return None


def _sha(path):
    h = hashlib.sha256()
    with open(path, "rb") as f:
        for b in iter(lambda: f.read(1 << 24), b""):
            h.update(b)
    return h.hexdigest()


step(3, "Proje ZIP'i hazırlanıyor (Google Drive)")
need_download = FORCE_REDOWNLOAD or not os.path.isfile(f"{PROJECT}/scripts/run_training.py")
if need_download:
    if "BURAYA" in DRIVE_FILE_ID or len(DRIVE_FILE_ID) < 20:
        fail("DRIVE_FILE_ID girilmemiş. Drive paylaşım bağlantısındaki /d/<ID>/ kısmını yazın.")
    err = _download_direct(DRIVE_FILE_ID, ZIP_PATH)
    if err or _validate_zip(ZIP_PATH):
        print("Doğrudan indirme başarısız:", err or _validate_zip(ZIP_PATH), "→ gdown deneniyor")
        err2 = _download_gdown(DRIVE_FILE_ID, ZIP_PATH)
        v = _validate_zip(ZIP_PATH)
        if err2 or v:
            fail("ZIP indirilemedi/doğrulanamadı.\n  - Neden: " + str(err2 or v) +
                 "\n  - Notebook sağ panelinde Internet = ON olmalı.\n  - Dosya 'Bağlantıya sahip herkes' ile paylaşılmış olmalı.\n"
                 "  - Alternatif: ZIP'i Kaggle Dataset olarak yükleyin ve /kaggle/input altından kopyalayın.")
    print(f"İndirildi: {os.path.getsize(ZIP_PATH) / 1e6:.1f} MB | sha256={_sha(ZIP_PATH)[:16]}…")
    step(7, "ZIP açılıyor")
    tmp_extract = f"{WORK}/.altun1_extract"
    shutil.rmtree(tmp_extract, ignore_errors=True)
    with zipfile.ZipFile(ZIP_PATH) as z:
        z.extractall(tmp_extract)
    root = tmp_extract if os.path.isfile(f"{tmp_extract}/scripts/run_training.py") else f"{tmp_extract}/ALTUN-1"
    if not os.path.isfile(f"{root}/scripts/run_training.py"):
        fail("Çıkarılan ZIP'te scripts/run_training.py bulunamadı (eksik çıkarma).")
    # Önceki proje klasörünü güncelle; kullanıcı verisi (raw_data) ve işlenmiş veri korunur
    if os.path.isdir(PROJECT):
        for keep in ("raw_data", "outputs"):
            if os.path.isdir(f"{PROJECT}/{keep}"):
                shutil.copytree(f"{PROJECT}/{keep}", f"{root}/{keep}", dirs_exist_ok=True)
        shutil.rmtree(PROJECT)
    shutil.move(root, PROJECT)
    shutil.rmtree(tmp_extract, ignore_errors=True)
    json.dump({"sha256": _sha(ZIP_PATH), "time": time.time()}, open(MARKER, "w"))
else:
    print("Proje zaten mevcut, yeniden indirilmedi (FORCE_REDOWNLOAD=True ile zorlayabilirsiniz).")
print("Proje dizini:", PROJECT)
os.chdir(PROJECT)
sys.path.insert(0, PROJECT)

# 11-12) Bağımlılıklar -------------------------------------------------------------------
step(11, "Python paketleri kontrol ediliyor")
missing = []
for mod, pip in [("torch", "torch"), ("numpy", "numpy"), ("yaml", "pyyaml"), ("psutil", "psutil"),
                 ("matplotlib", "matplotlib"), ("safetensors", "safetensors")]:
    try:
        importlib.import_module(mod)
    except ImportError:
        missing.append(pip)
if missing:
    print("Eksik paketler:", missing, "→ kuruluyor")
    r = subprocess.run([sys.executable, "-m", "pip", "install", "-q"] + [m for m in missing if m != "torch"],
                       capture_output=True, text=True)
    if r.returncode != 0:
        fail("Eksik paketler kurulamadı (İnternet kapalı olabilir): " + ", ".join(missing) + "\n" + r.stderr[-500:])
    if "torch" in missing:
        fail("PyTorch eksik: Kaggle'ın standart GPU imajını kullanın.")

# 13-28) Preflight → veri → eğitim → test → rapor → ZIP -----------------------------------
step(14, "Preflight, veri, eğitim, değerlendirme ve paketleme başlıyor (ayrıntılı çıktı aşağıda)")
from scripts.run_training import run  # noqa: E402

rc = run({"TRAINING_DURATION_HOURS": TRAINING_DURATION_HOURS, "RESUME_FROM_CHECKPOINT": RESUME_FROM_CHECKPOINT,
          "OUTPUT_DIR": OUTPUT_DIR, "SMOKE_TEST": SMOKE_TEST})
print("\nÇıkış kodu:", rc, "(0 = başarılı)")
print("Sonuçlar:", OUTPUT_DIR, "| ZIP:", f"{WORK}/ALTUN-1-RESULTS.zip")
if rc != 0:
    raise SystemExit(f"ALTUN-1 işlemi kod {rc} ile bitti; yukarıdaki mesajlara bakın.")
