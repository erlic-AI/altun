# ALTUN 1 — erlic AI (özel geliştirme sistemi)

Matematik ve teorik bilgisayar bilimi odaklı, **sıfırdan** eğitilen decoder-only Transformer (RMSNorm, RoPE, SwiGLU, GQA, PyTorch SDPA, FP16 AMP) için uçtan uca eğitim–değerlendirme sistemi. Kaggle 2×T4 için tasarlandı. Hazır ağırlık kullanılmaz; ALTUN 1 P vs NP'yi çözdüğünü iddia etmez — ilk araştırma prototipidir.

> **Dürüst durum notu (v0.1):** Bu sistem Kaggle 2×T4 üzerinde gerçekten çalıştırıldı (3 tur, toplam ≈1.94B token). Sonuçlar `results/evaluation_summary.md` ve ana `README.md` içinde olduğu gibi verilmiştir: ALTUN 119'luk set %53.8 (dağılım içi), bağımsız GSM8K %1.5, MATH %5.0, Türkçe mini set %25. Model güvenilir bir hesap makinesi/çözücü DEĞİLDİR. Test kapsamı: `docs/TEST_REPORT.md`.

---

## 1. Veri (OTOMATİK — elle veri hazırlamanız gerekmez)

Başlangıç hücresi, eğitimden önce veri yoksa **açık lisanslı kaynakları kendisi indirir** (Kaggle'da **Internet = On** olmalı). Sonuç `reports/fetch_report.json`'da görünür; bir kaynak başarısız olursa diğerleriyle devam eder, hiç veri alınamazsa eğitim başlamaz.

| Kategori (hedef oran) | Kaynak | Lisans |
|---|---|---|
| formal (%60) | Mathlib4, Lean 4 core, Metamath set.mm | Apache-2.0 / Apache-2.0 / CC0-1.0 |
| english_math (%25) | NuminaMath-CoT (150 bin satır), Open Logic Project, HoTT book, İngilizce Wikipedia matematik kategorileri | Apache-2.0 / CC-BY-4.0 / CC-BY-SA-4.0 / CC-BY-SA-4.0 |
| cs_theory (%10) | TheAlgorithms/Python, İngilizce Wikipedia TCS kategorileri | MIT / CC-BY-SA-4.0 |
| turkish_math (%5) | Türkçe Wikipedia matematik/TCS kategorileri | CC-BY-SA-4.0 |

Sistem lisansı izinli listede olmayan/belirtilmeyen veriyi otomatik dışlar, lisans/provenance kaydı tutar (`reports/provenance.json`), eğitim verisini benchmark'lardan ve test setinden ayırır. Wikipedia metinleri CC BY-SA 4.0 gereği atıf gerektirir (kaynak `source.json`'da kayıtlı); modeli yayınlamadan önce lisans koşullarını (özellikle SA) gözden geçirin.

**Gerçekçi beklenti:** otomatik toplanan veri yaklaşık 150–300 milyon token ölçeğindedir (tahmin, ölçülmedi); Türkçe kısım küçük kalır (birkaç milyon token), bu yüzden hedef %5 oranı tutmayabilir (gerçekleşen oran `data_quality_report.md`'de). Veri 12 saatlik kapasiteden az olursa model veriyi birkaç tur görür. Daha fazla/özel veri isterseniz yine ekleyebilirsiniz: `raw_data/<ad>/source.json` + metin dosyaları (veya Kaggle Dataset); `/kaggle/input` otomatik taranır. İndirme gecikmesi eğitim süresinden önce 10–30 dakika sürebilir.

**Benchmark (isteğe bağlı):** GSM8K/MATH/AIME değerlendirmesi için `raw_data/_benchmarks/gsm8k*.jsonl` vb. dosyaları sağlarsanız çalışır (eğitime girmez, dekontaminasyonda kullanılır); yoksa "çalıştırılmadı" diye raporlanır.

---

## 2. Adım adım kullanım

1. **ZIP'i Drive'a yükleyin:** `ALTUN-1.zip` → drive.google.com → Yeni → Dosya yükleme.
2. **Paylaşım bağlantısı:** dosyaya sağ tık → Paylaş → Genel erişim → **"Bağlantıya sahip herkes"** (Görüntüleyen) → Bağlantıyı kopyala.
3. **Dosya ID'si:** bağlantı `https://drive.google.com/file/d/`**`1AbCdEfG...`**`/view?usp=sharing` biçimindedir; `/d/` ile `/view` arasındaki kısım ID'dir. (Klasör bağlantısı değil, ZIP dosyası bağlantısı kullanın.)
4. **Kaggle notebook:** kaggle.com → Create → New Notebook.
5. **GPU/İnternet ayarları:** sağ panel → Session options → **Accelerator: GPU T4 x2**, **Internet: On** (Drive'dan indirme için; telefon doğrulaması gerekebilir). Verinizi Dataset olarak eklediyseniz "Add Input" ile ekleyin.
6. **Başlangıç hücresi:** `KAGGLE_START_CELL.py` içeriğinin tamamını tek hücreye yapıştırın ve en üstteki ayarları düzenleyin: `DRIVE_FILE_ID`, `TRAINING_DURATION_HOURS`, `RESUME_FROM_CHECKPOINT`, `OUTPUT_DIR`.
7. **Çalıştırın** (Run). Hücre sırayla: ortam → GPU → ZIP indir/doğrula/aç → bağımlılık → preflight (DDP + autotune) → birim testleri → veri hazırlama → eğitim → değerlendirme → grafik/rapor → ZIP.
8. **Takip:** hücre çıktısında her 10 adımda loss, lr, grad norm, tokens/s, geçen/kalan süre görünür; her 250 adımda val loss; 30 dakikada bir checkpoint. Sağ panel → Logs; ayrıca `ALTUN-1-RESULTS/logs/*.csv`.
9. **Sonuçları indirme:** Kaggle'da **Save Version → Save & Run All (Commit)** ile çalıştırırsanız (kapalı sekmeyle de koşar), bittikten sonra notebook sayfasında **Output** sekmesinden `ALTUN-1-RESULTS.zip` veya klasörü indirin. Etkileşimli oturumda sağ panel → Output → `/kaggle/working`.

### Oturum sınırı ve devam etme
Kaggle GPU oturumları en fazla ~12 saattir ve kesintisiz çalışma garantisi yoktur. `TRAINING_DURATION_HOURS` **toplam** eğitim süresidir (oturumlar arası birikir). 12 saatlik oturumda veri hazırlama + değerlendirme için pay bırakmak üzere **9.5–10 saat** önerilir.
- Oturum kesilirse: Commit çalıştırdıysanız çıktıdaki `checkpoints/` korunur. Yeni notebook'ta önceki versiyonun **Output'unu Input olarak ekleyin** ve hücreyi `RESUME_FROM_CHECKPOINT="auto"` ile tekrar çalıştırın; en son geçerli checkpoint otomatik alınır.
- Aynı oturumda tekrar çalıştırmak da kaldığı yerden devam eder (`"auto"`).
- `RESUME_FROM_CHECKPOINT=False` **sıfırdan** başlatır ve eski checkpoint'leri siler (yanlışlıkla eski eğitim yüklenmez).
- Resume'da tokenizer hash'i ve model config'i kontrol edilir; uyuşmazlık net hata verir.

---

## 3. Eğitim sonrası beklenen dosyalar (`/kaggle/working/ALTUN-1-RESULTS/`)

```
final_model/model.safetensors, config.json, tokenizer/tokenizer.json
checkpoints/ckpt_*.pt (+ .json)            # en fazla 2 geçerli checkpoint (ZIP'e girmez)
evaluations/benchmark_results.json, custom_math_results.json, evaluation_summary.md,
            failed_examples.json, test_results.json
logs/training_log.csv, training_log.jsonl, system_metrics.csv
reports/training_report.md, hardware_report.md, test_report.md, data_quality_report.md/.json,
        provenance.json, autotune.json, preflight.json, unit_tests_pre_training.txt, training_result.json
plots/loss_curve.png, learning_rate.png, gpu_usage.png, training_speed.png
ALTUN-1-RESULTS.zip      (ayrıca /kaggle/working/ALTUN-1-RESULTS.zip)
```
Her çalıştırmada **aynı** klasör ve dosya adları kullanılır (üzerine yazılır); `-1/-2/new` gibi kopya klasör üretilmez. Ham/işlenmiş veri `ALTUN-1/outputs/data/` içindedir (sonuç klasörünün dışında).

## 4. Çıkarım

```python
from inference.run_inference import load_altun, ask, demo
model, tok, dev = load_altun("/kaggle/working/ALTUN-1-RESULTS/final_model")
demo(model, tok, dev)                                   # örnek cebir/sayılar teorisi/mantık/kombinatorik/karmaşıklık/TCS/Türkçe soruları
print(ask(model, tok, dev, "Question: What is gcd(48, 18)?\nAnswer:", temperature=0.2))
```

## 5. Sorun giderme
| Belirti | Neden / çözüm |
|---|---|
| `ZIP indirilemedi` | Internet Off; paylaşım "herkes" değil; Drive kotası (24 sa. bekleyin / kopyasını kendi Drive'ınıza alın). Alternatif: ZIP'i Kaggle Dataset yapıp `/kaggle/input`'tan kopyalayın |
| `Hiç veri kaynağı bulunamadı` | Bölüm 1'e göre veri ekleyin |
| `Preflight başarısız: CUDA` | Accelerator'ı GPU T4 x2 yapıp oturumu yeniden başlatın |
| DDP uyarısı → tek GPU | `preflight.json` içinde ayrıntı; eğitim tek GPU ile (aynı token/adım, daha uzun accumulation) devam eder |
| OOM | Autotune otomatik küçültür; elle: `config/training_config.yaml` → `micro_batch_size` |
| Loss NaN | GradScaler adımı atlar; 100 ardışık atlamada eğitim durur, checkpoint korunur; `peak_lr`'i düşürüp resume edin |

## 6. Yapı
`altun/` model+tokenizer+checkpoint+generation · `data/` veri hattı · `training/` trainer/DDP/izleme · `evaluation/` testler+benchmark+ALTUN seti · `inference/` · `scripts/` orkestrasyon + Kaggle hücresi · `tests/` · `config/` · `docs/` teknik & test raporu.

Gizlilik: Bu depo özel geliştirme içindir; GitHub/Hugging Face yayını hazırlanmamıştır.


## 7. İkinci tur (v2): önceki modelden devam
1. Kaggle'da yeni notebook → Accelerator **GPU T4 x2**, **Internet ON**.
2. Sağ panel **Add Input** → önceki ALTUN notebook'unun **Output**'unu ekleyin (içinde `ALTUN-1-RESULTS/final_model` olmalı).
3. `KAGGLE_START_CELL.py`'yi yapıştırın, `DRIVE_FILE_ID`'yi girin (`INIT_FROM_PREVIOUS=True`, süre 9 saat). Çalıştırın, ardından **Save & Run All (Commit)** ile çıktıyı kalıcılaştırın.
- Önceki model bulunamazsa eğitim **başlamaz** (sessizce sıfırdan başlamaz).
- Tokenizer önceki modelle aynı kalır; veri v2 karışımıyla (matematik QA + sentetik adım adım işlem çözümleri ağırlıklı) yeniden hazırlanır.
- `torch.compile` ilk adımlarda gerçek donanımda ölçülür; kazanç <%8 veya hata varsa otomatik kapanır (`training_result`/rapordaki "torch.compile" satırı).
- Not: v2 kodunun GPU kısımları (init-from, compile ölçümü) bu geliştirme ortamında çalıştırılamadı (torch yok); Kaggle'daki ilk dakikalarda log'da `[init]` ve `[compile]` satırlarını kontrol edin.

### v3 notu (üçüncü tur)
- Önceki tur modelinin üstüne devam; sentetik veri yeni seed'le yeniden üretilir (aynı örnekler tekrar görülmez), zayıf kategoriler (algoritma analizi, lineer cebir, kalkülüs, Lean, ispat boşlukları) ve Türkçe için üreteçler eklendi.
- Eğitim süresi varsayılanı 7 saat (9 saatlik GPU kotası için), tepe LR 2.5e-4.
- Değerlendirme artık ayrıca: **Türkçe mini set** (32 el yazımı soru) ve **bağımsız GSM8K/MATH test örnekleri** (indirilip eğitimden dışlanır; sonuçlar olduğu gibi raporlanır).
- Dürüstlük notu: ALTUN 119'luk set ile eğitim üreteçleri aynı beceri alanlarını hedefler, yani o set "dağılım içi"dir; bağımsız ölçüm için GSM8K/MATH ve Türkçe mini sete bakın.
