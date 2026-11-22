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

1. **Kaggle notebook:** kaggle.com → Create → New Notebook.
2. **Ayarlar:** sağ panel → Session options → **Accelerator: GPU T4 x2**, **Internet: On** (GitHub, Hugging Face ve veri indirmesi için; telefon doğrulaması gerekebilir).
3. **Başlangıç hücresi:** `KAGGLE_START_CELL.py` içeriğinin tamamını tek hücreye yapıştırın. Google Drive gerekmez; kod GitHub'dan (`erlic-AI/ALTUN-1`) çekilir. İsterseniz en üstteki ayarları düzenleyin: `TRAINING_DURATION_HOURS`, `CONTINUE_FROM` (`"hf"`: Hugging Face'teki yayınlanmış ağırlıklardan devam, `"input"`: önceki notebook çıktısından devam, `"none"`: sıfırdan), `PEAK_LR`.
4. **Çalıştırın** (Run). Hücre sırayla: ortam → GPU → kodu GitHub'dan al → (ağırlıkları HF'den indir) → preflight (DDP + autotune) → birim testleri → veri indirme/hazırlama → eğitim → değerlendirme → grafik/rapor → ZIP.
5. **Takip:** hücre çıktısında her 10 adımda loss, lr, grad norm, tokens/s, geçen/kalan süre görünür; her 250 adımda val loss; 30 dakikada bir checkpoint. Sağ panel → Logs; ayrıca `ALTUN-1-RESULTS/logs/*.csv`.
6. **Sonuçları indirme:** Kaggle'da **Save Version → Save & Run All (Commit)** ile çalıştırırsanız (kapalı sekmeyle de koşar), bittikten sonra notebook sayfasında **Output** sekmesinden `ALTUN-1-RESULTS.zip` veya klasörü indirin.

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
| `git clone başarısız` / `HF indirme başarısız` | Internet Off; repo/model adı yanlış; geçici ağ hatası (tekrar deneyin). `CONTINUE_FROM="none"` ile HF'siz de başlatabilirsiniz |
| `Hiç veri kaynağı bulunamadı` | Bölüm 1'e göre veri ekleyin |
| `Preflight başarısız: CUDA` | Accelerator'ı GPU T4 x2 yapıp oturumu yeniden başlatın |
| DDP uyarısı → tek GPU | `preflight.json` içinde ayrıntı; eğitim tek GPU ile (aynı token/adım, daha uzun accumulation) devam eder |
| OOM | Autotune otomatik küçültür; elle: `config/training_config.yaml` → `micro_batch_size` |
| Loss NaN | GradScaler adımı atlar; 100 ardışık atlamada eğitim durur, checkpoint korunur; `peak_lr`'i düşürüp resume edin |

## 6. Yapı
`altun/` model+tokenizer+checkpoint+generation · `data/` veri hattı · `training/` trainer/DDP/izleme · `evaluation/` testler+benchmark+ALTUN seti · `inference/` · `scripts/` orkestrasyon + Kaggle hücresi · `tests/` · `config/` · `docs/` teknik & test raporu.

Yayın: kod GitHub'da, model ağırlıkları Hugging Face'te (`erlic/ALTUN-1`) yayınlanmıştır; Apache-2.0, veri atıfları için `NOTICE`.


## 7. Devam eğitimi (v2/v3 notları)
- `CONTINUE_FROM="hf"` (varsayılan): yayınlanmış ağırlıklardan devam, yeni optimizer/LR programı; tokenizer aynı kalır. Önceki model bulunamazsa eğitim **başlamaz** (sessizce sıfırdan başlamaz).
- Veri karışımı: matematik QA + sentetik adım adım çözümler ağırlıklı; sentetik veri yeni seed'le üretilir (aynı örnekler tekrar görülmez); zayıf kategoriler ve Türkçe için üreteçler vardır.
- `torch.compile` ilk adımlarda gerçek donanımda ölçülür; kazanç <%8 veya hata varsa otomatik kapanır.
- Tepe LR varsayılanı 2.5e-4, süre 7 saat. Değerlendirme: ALTUN 119'luk set (dağılım içi), Türkçe mini set, bağımsız GSM8K/MATH örnekleri.
