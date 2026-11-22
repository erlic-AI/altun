# ALTUN 1 — Teknik Rapor

Etiketler: **[ÖLÇÜLDÜ]** gerçekten ölçüldü · **[ANALİTİK]** koddan hesaplandı (burada doğrulandı) · **[TAHMİN]** ölçülmedi; Kaggle'da ilk çalıştırmada autotune ve eğitim logları gerçek değerleri üretir.

## 1. Mimari
Decoder-only Transformer, sıfırdan rastgele başlatma (önceden eğitilmiş ağırlık yok): Pre-norm **RMSNorm** (float32 hesap), **RoPE** (θ=10000), **GQA** (4 KV head), **SwiGLU**, bias yok, giriş/çıkış embedding bağlı (tied), **PyTorch SDPA** (T4'te zorunlu özel kernel yok; flash yoksa mem-efficient/math otomatik), rezidüel çıkış projeksiyonlarında 1/√(2L) ölçekli init.

| Preset | Katman | Hidden | Head | KV head | Intermediate | Context | Vocab | Parametre [ANALİTİK] |
|---|---|---|---|---|---|---|---|---|
| altun1-smoke | 2 | 128 | 4 | 2 | 352 | 1024 | 16384* | 2.47 M (yalnızca duman testi) |
| altun1-small | 12 | 768 | 12 | 4 | 2048 | 1024 | 16384 | 88.1 M |
| **altun1-main (varsayılan)** | **16** | **1024** | **16** | **4** | **2816** | **1024** | **16384** | **197.2 M** |

\*duman testinde vocab 512. Parametre formülü `ModelConfig.param_count()`; `tests/test_model.py::test_param_count_matches_analytic` bunu gerçek modelle karşılaştırır (Kaggle'da çalışacak).

**Neden 350M değil?** 12 saatte ~0.3–0.5 B token işlenebilir **[TAHMİN]**. 350M model için bu çok düşük token/parametre oranıdır (≈1–1.5 token/param; Chinchilla ≈20). 197M ≈ 2–2.5 token/param; yine düşük ama daha dengeli. Seçim ölçüm değil bu hesaba dayanır; `altun1-small` daha az veriyle daha doygun olur — `active_preset` ile değiştirilebilir.

## 2. Tokenizer
Kendi **byte-level BPE** (saf Python, harici bağımlılık yok), 16 384 vocab (8 özel token + 256 byte + 16 120 birleşme). Matematik duyarlı ön-bölme: LaTeX komutları (`\frac`) tek birim, rakamlar tek tek (aritmetik için), her Unicode sembol (∀ ∃ ≤ ∈ ⊢ …) tek ön-token, ASCII operatör grupları (`:=`, `->`), `_` dahil. Kayıpsız round-trip (3000 rastgele Unicode dizisiyle fuzz **[ÖLÇÜLDÜ]**), UNK gerekmez. Türkçe harfler Unicode harf sınıfıyla bütün kalır. Eğitim yalnızca train bölümü belgelerinden yapılır.

## 3. Veri
Lisans kapısı (izinli listede olmayan/boş lisans → dışlanır, rapora yazılır), temizleme (NFC, kontrol karakteri, bozuk kodlama, minify/tekrar/düşük alfanümerik oran), tam + MinHash-LSH yakın kopya tekilleştirme, benchmark 13-gram dekontaminasyonu, **belge hash'ine göre** train/val/test (%98/1/1; sızıntı yok), token bazlı karışım seçimi (kıt kategoriler tamamen alınır, kalan bütçe fazlalara dağıtılır), uint16 `.bin` paketleme (belgeler BOS/EOS ile art arda; rastgele pencere kesimi). Veri yoksa/lisanssızsa/yetersizse **durur** (`DataError`, çıkış kodu 2). Tüm bu adımlar test edildi **[ÖLÇÜLDÜ, `tests/test_pipeline.py`]**. Not: paketlemede pencereler belge sınırlarını aşabilir (cross-document attention maskesi yok) — bilinen basitleştirme.

## 4. Eğitim stratejisi
- **FP16 AMP + GradScaler** (T4'te BF16 yok), AdamW (β=0.9/0.95, wd 0.1 yalnızca matris/embedding), grad clip 1.0, fused AdamW (varsa).
- **Wall-clock süre**: LR = step bazlı warmup (200) + geçen süreye (0..1) bağlı cosine (peak 6e-4 → %10). Durma kriteri süredir; epoch sabit değildir. Süre checkpoint'te birikir (resume'da toplam hedefe saygı).
- **İki GPU**: `torch.distributed.run` ile DDP (NCCL); **her GPU'da tam model kopyası** vardır, VRAM birleşmez. DDP self-check (2 süreç all_reduce) başarısızsa tek GPU'ya düşer (aynı global token/adım korunur).
- Global batch: 131 072 token/adım (GPU başına micro-batch × 1024 × GPU × accumulation). Gradient checkpointing ve micro-batch, `training/autotune.py` kısa ölçümüyle seçilir (OOM yakalanır, en hızlı güvenli konfig).
- **Veri konumu**: batch'ler `(seed, step, micro, rank)`'ten deterministik türetilir; resume'da aynı akış devam eder, ayrı iterator durumu gerekmez. Arka plan thread'iyle prefetch + pin_memory.
- Rank 0 her adım `[progress, stop, checkpoint, eval]` kontrol tensörünü yayınlar → tüm rank'ler aynı adımda durur/checkpoint alır (DDP deadlock önleme). Val loss iki GPU'da bölünüp all_reduce edilir.
- SIGTERM'de graceful durma (checkpoint) ve `complete=False` işareti.

## 5. Checkpoint
Model, optimizer, GradScaler, step, token, geçen süre, RNG durumları, config'ler, tokenizer hash'i. `.tmp` → fsync → `os.replace` (atomik) + `.json` manifest (boyut, sha256). Yalnızca en yeni **2 geçerli** checkpoint tutulur; geçerli sayısı ≤ keep ise hiçbir şey silinmez; bozuk en yeni dosyada bir öncekine düşülür; disk eşiğin altındaysa yazılmaz (eski korunur). `RESUME=False` eski checkpoint'leri temizler; `"auto"` en son geçerli olanı yükler. Final model `model.safetensors` (+ config.json + tokenizer/).

## 6. Performans ve VRAM — tahmin vs ölçüm
| Kalem | Değer | Durum |
|---|---|---|
| VRAM/GPU (197M, micro-batch 8, grad-ckpt) | ≈ 5.9 GB | **[TAHMİN]** `estimate_vram_gb` |
| VRAM/GPU (aynı, ckpt yok) | ≈ 14.1 GB | **[TAHMİN]** (T4 15 GB'a çok yakın; autotune ölçer) |
| FLOP/token (6N + ckpt tekrar hesabı ≈ +33%) | ≈ 1.6×10⁹ | **[ANALİTİK]** |
| Tokens/s (2×T4, DDP) | ≈ 10–16 bin (geniş belirsizlik) | **[TAHMİN]** — T4'te gerçekçi ~12–18 TFLOPs/GPU varsayımı |
| 10 saatte işlenecek token | ≈ 0.35–0.55 B | **[TAHMİN]** |
| Gerçek tokens/s, VRAM, token kapasitesi | — | Kaggle'da ölçülür: `reports/autotune.json`, `training_report.md` ("ÖLÇÜLEN" etiketli) |

Hiçbir performans değeri burada ölçülmedi (GPU yok).

## 7. Değerlendirme
Eğitim sonrası otomatik: model/tokenizer yükleme, ağırlık sonluluğu, forward, backward, **KV-cache ≡ cache'siz** eşdeğerliği, kısa/uzun bağlam (limit aşımı reddi), batch çıkarım, Türkçe/İngilizce/LaTeX/Lean istemleri (degenerate-çıktı kontrolü), notasyon round-trip, test-seti loss/perplexity. **ALTUN özel seti**: 13 kategori, 3 zorluk seviyesi, 119 soru; cevaplar kodla hesaplanır, greedy üretim, kategori/zorluk doğrulukları, tüm yanlışlar `failed_examples.json`'da. Küçük set → yüksek varyans (raporda belirtilir). **Benchmark**: GSM8K/MATH/AIME yalnızca kullanıcı veri dosyası sağlarsa çalışır (0-shot, ilk 200 örnek); dosya yoksa `not_run`. **MiniF2F/Lean formal ispat doğrulaması çalıştırılmaz** (Lean 4 + Mathlib doğrulayıcı yok). 200M, ≤0.5B token ile sıfırdan eğitilmiş bir modelden GSM8K/MATH/AIME'de anlamlı başarı **beklenmemelidir**; ALTUN seti ve val loss ilk ilerleme göstergeleridir.

## 8. Bilinen sınırlar
Torch kodu burada çalıştırılmadı (bkz. TEST_REPORT); gerçek 2×T4 denenmedi; veri indirme betiği ağ engeli nedeniyle denenmedi; cross-document maskeleme yok; context 1024; Drive indirmesi kota/izin sorunlarına açıktır (hücre net hata verir).
