# ALTUN 1 Değerlendirme Özeti (v0.1, run 3 çıktısı — Kaggle 2×T4, olduğu gibi)

## ALTUN özel matematik seti

- Toplam soru: 119
- Doğru: 64  |  Yanlış: 55
- Doğruluk: 53.8%
- Anlamsız/degenerate çıktı oranı: 1.7%
- Zorluğa göre: {1: 0.575, 2: 0.5609756097560976, 3: 0.47368421052631576}
- Not: Set küçüktür (kategori başına ~5-21 soru); yüzdeler yüksek varyanslıdır. Eğitim üreteçleri aynı beceri alanlarını hedeflediği için bu set dağılım içidir.

| Kategori | Soru | Doğru | Doğruluk |
|---|---|---|---|
| Algebra | 9 | 4 | 44.4% |
| Algorithm Analysis | 8 | 5 | 62.5% |
| Algorithms | 9 | 0 | 0.0% |
| Arithmetic | 21 | 17 | 81.0% |
| Calculus | 9 | 3 | 33.3% |
| Combinatorics | 9 | 5 | 55.6% |
| Computational Complexity | 7 | 4 | 57.1% |
| Formal Mathematics | 6 | 5 | 83.3% |
| Graph Theory | 9 | 4 | 44.4% |
| Linear Algebra | 9 | 5 | 55.6% |
| Mathematical Logic | 9 | 2 | 22.2% |
| Number Theory | 9 | 6 | 66.7% |
| Proof Completion | 5 | 4 | 80.0% |

**Türkçe mini set (el yazımı, ayrı rapor):** 8/32 = 25.0%

## Test seti kaybı

{"status": "run", "test_loss": 0.6731061697006225, "test_perplexity": 1.9603169506822762, "n_batches": 30}

## Standart benchmark'lar

- **GSM8K**: 3/200 = 1.5% (0-shot, greedy)
- **MATH**: 10/200 = 5.0% (0-shot, greedy)
- **AIME**: ÇALIŞTIRILMADI — Veri dosyası bulunamadı.
- **MiniF2F**: ÇALIŞTIRILMADI — Lean 4 + Mathlib doğrulayıcı bu ortamda yok.
