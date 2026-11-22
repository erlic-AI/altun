# ALTUN 1 — Test Raporu (geliştirme ortamında GERÇEKTEN çalıştırılanlar)

Ortam: Python 3.13, **PyTorch YOK, GPU YOK** (2×T4 hiç denenmedi). Tarih: 2026-11-20.

## A. Çalıştırıldı ve geçti (pytest)

| Test | Çalıştırıldı | Sonuç |
|---|---|---|
| test_pipeline::test_scheduler_warmup_and_cosine | evet | BAŞARILI |
| test_pipeline::test_all_configs_load_and_are_consistent | evet | BAŞARILI |
| test_pipeline::test_license_gate_dedup_decontamination_split | evet | BAŞARILI |
| test_pipeline::test_no_data_stops_with_clear_error | evet | BAŞARILI |
| test_pipeline::test_only_unlicensed_data_stops | evet | BAŞARILI |
| test_pipeline::test_smoke_pipeline_and_dataset | evet | BAŞARILI |
| test_pipeline::test_custom_eval_set_is_complete_and_deterministic | evet | BAŞARILI |
| test_pipeline::test_answer_checkers_and_metrics | evet | BAŞARILI |
| test_pipeline::test_benchmarks_never_fabricate | evet | BAŞARILI |
| test_pipeline::test_split_long_documents | evet | BAŞARILI |
| test_pipeline::test_clean_wikitext_keeps_math_and_drops_markup | evet | BAŞARILI |
| test_pipeline::test_wikipedia_fetch_with_mock_http | evet | BAŞARILI |
| test_pipeline::test_github_fetch_with_mock_zip | evet | BAŞARILI |
| test_pipeline::test_ensure_data_tolerates_partial_failure | evet | BAŞARILI |
| test_tokenizer::test_roundtrip_math_latex_lean_turkish | evet | BAŞARILI |
| test_tokenizer::test_special_tokens | evet | BAŞARILI |
| test_tokenizer::test_save_load_identical | evet | BAŞARILI |
| test_tokenizer::test_math_symbols_are_single_pretokens | evet | BAŞARILI |
| test_tokenizer::test_invalid_id_raises | evet | BAŞARILI |
| test_tokenizer::test_compression_and_vocab_bounds | evet | BAŞARILI |

**Toplam: 20 test çalıştı, 20 başarılı. Atlanan modül: 3 (torch yok → test_model, test_generation, test_training; bkz. bölüm C).**

## B. Diğer gerçek kontroller

| Kontrol | Çalıştırıldı | Sonuç |
|---|---|---|
| Python syntax (compileall: altun, data, training, evaluation, inference, scripts, tests) | evet | BAŞARILI |
| Import kontrolü (torch gerektirmeyen 11 modül) | evet | BAŞARILI |
| Tokenizer fuzz round-trip (3000 rastgele Unicode dizisi) | evet | BAŞARILI (süreçte `_` karakteri hatası bulundu ve düzeltildi) |
| Veri hattı smoke (sentetik veri, 1080 belge → train/val/test .bin) | evet | BAŞARILI |
| Veri yokken eğitimin durması | evet | BAŞARILI (net hata, çıkış kodu 2) |
| Lisanssız/GPL kaynakların dışlanması, dekontaminasyon, dedup | evet | BAŞARILI |
| Donanım denetimi / preflight (torch yok) | evet | Beklenen şekilde BAŞARISIZ (net hata mesajı) |
| Sonuç ZIP'i: checkpoint hariç, kendini içermez | evet | BAŞARILI |
| Grafik üretimi (loss/lr/gpu/speed PNG) sahte log ile | evet | BAŞARILI |

## C. ÇALIŞTIRILMADI (torch/GPU/ağ yok) — Kaggle'da ilk çalıştırmada koşacak

| Test/Bileşen | Çalıştırıldı | Durum |
|---|---|---|
| tests/test_model.py (10 test: forward, backward, param sayısı, causality, KV-cache, grad-ckpt, RoPE) | hayır | YAZILDI, çalıştırılmadı (torch yok) |
| tests/test_generation.py (8 test) | hayır | YAZILDI, çalıştırılmadı |
| tests/test_training.py (checkpoint atomik/fallback/temizlik, export-load, trainer dry-run + resume + sıfırdan başlatma) | hayır | YAZILDI, çalıştırılmadı |
| CPU smoke / CUDA smoke / küçük eğitim dry-run | hayır | test_training.py içinde; Kaggle preflight sonrası `unit_tests_pre_training.txt` |
| DDP (2 GPU) çalışması, NCCL, deadlock | hayır | `training.distributed` self-check preflight'ta koşar |
| Autotune, gerçek tokens/s ve VRAM ölçümü | hayır | Kaggle'da ölçülür |
| evaluation/evaluate.py model testleri (forward/backward/generation/long-context/batch) | hayır | Eğitim sonrası koşar → `reports/test_report.md` |
| GitHub/HF tabanlı başlatıcı (`start.py`) | hayır | Ağ yok; yalnızca syntax kontrolü (Drive tabanlı eski başlatıcı Kaggle'da çalıştı; start.py henüz Kaggle/başka makinede denenmedi) |
| `fetch_open_data.py` gerçek ağ indirmesi (GitHub/HF/Wikipedia) | hayır | Sandbox tüm bu adreslere 403 verdi; yalnızca metin temizleme, kategori tarama, ZIP ayıklama **sahte HTTP ile test edildi (4 test)** |
| MiniF2F / Lean formal doğrulama | hayır | Lean doğrulayıcı yok; kapsam dışı (benchmark `not_run` raporlar) |

Not: Bu bölümdeki kod, torch'suz ortamda yalnızca söz dizimi düzeyinde doğrulandı ve elle gözden geçirildi; **çalışıyor** diye iddia edilmez.


## v2 eki
- torch'suz testler: **26 geçti, 3 atlandı** (atlananlar torch gerektirir; Kaggle'da çalışır).
- Test edilenler: sentetik veri doğruluğu, v2 karışım + tokenizer yeniden kullanımı, HF fetch (sahte pyarrow), dedup kategori muafiyeti, boxed/son-sayı cevap çıkarımı.
- **Test EDİLMEDİ (torch yok):** init-from yükleme, torch.compile ölçümü/yedekleme, gerçek HF/GitHub/Wikipedia uçları (yalnızca mock). Bunlar Kaggle'da ilk kez çalışacak.

## v3 eki
- torch'suz testler: **29 geçti, 3 atlandı**. Eklenenler: 11 yeni üretecin EN/TR tutarlılık taraması (cevap == \boxed), 3x3 determinant (Leibniz) ve DFS sırası için bağımsız doğrulama, Türkçe mini set + eval dışlama, benchmark indirme (sahte pyarrow).
- Run 2 (Kaggle, gerçek donanım): 49 test geçti + 1 (benim testimin /kaggle/input varsayımı hatalıydı, düzeltildi); 19/19 eğitim sonrası model testi geçti.
- **Test EDİLMEDİ:** v3'ün gerçek HF benchmark indirmesi ve yeni üreteçlerle eğitim (Kaggle'da ilk kez çalışır); Open Logic Project kaynağı 404 veriyor (atlanıyor).
