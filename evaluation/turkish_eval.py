"""Türkçe mini değerlendirme (el yazımı 32 soru). Eğitim üreteçlerinden bağımsız yazıldı; ALTUN 119'luk setten AYRI raporlanır.
Küçük bir sette yüzdeler çok oynaktır; yalnızca kaba bir Türkçe göstergesidir."""
from __future__ import annotations

from typing import Dict, List

_Q = [
    ("Aritmetik", "Bir kitapçıda 3 rafın her birinde 24 kitap varsa toplam kaç kitap vardır?", "72", "number"),
    ("Aritmetik", "250 liralık bir ürüne %20 indirim yapılırsa yeni fiyat kaç liradır?", "200", "number"),
    ("Aritmetik", "Ali'nin 45 bilyesi var. 17 tanesini arkadaşına verirse kaç bilyesi kalır?", "28", "number"),
    ("Aritmetik", "144 sayısının karekökü kaçtır?", "12", "number"),
    ("Aritmetik", "Bir kenarı 9 cm olan karenin çevresi kaç cm'dir?", "36", "number"),
    ("Aritmetik", "7 ile 8'in çarpımına 5 eklenirse sonuç kaç olur?", "61", "number"),
    ("Aritmetik", "Bir araç saatte 80 km hızla 3 saat giderse kaç km yol alır?", "240", "number"),
    ("Aritmetik", "1000 sayısının yüzde 15'i kaçtır?", "150", "number"),
    ("Cebir", "2x + 7 = 19 denkleminde x kaçtır?", "6", "number"),
    ("Cebir", "5x - 3 = 22 ise x değeri nedir?", "5", "number"),
    ("Cebir", "x + y = 10 ve x - y = 4 ise x kaçtır?", "7", "number"),
    ("Cebir", "x^2 = 49 denkleminin pozitif kökü kaçtır?", "7", "number"),
    ("Sayı Kuramı", "12 ile 18'in en büyük ortak böleni (EBOB) kaçtır?", "6", "number"),
    ("Sayı Kuramı", "4 ile 6'nın en küçük ortak katı (EKOK) kaçtır?", "12", "number"),
    ("Sayı Kuramı", "20'den küçük kaç tane asal sayı vardır?", "8", "number"),
    ("Sayı Kuramı", "100 sayısı 7'ye bölündüğünde kalan kaçtır?", "2", "number"),
    ("Kombinatorik", "5 kişi arasından 2 kişilik bir ekip kaç farklı şekilde seçilir?", "10", "number"),
    ("Kombinatorik", "4 farklı kitap bir rafa kaç farklı sırayla dizilebilir?", "24", "number"),
    ("Olasılık", "Hilesiz bir zarın atılışında 6 gelme olasılığı kesir olarak nedir? (payda 6)", "1/6", "text"),
    ("Kalkülüs", "f(x) = x^3 ise f'(2) kaçtır?", "12", "number"),
    ("Kalkülüs", "f(x) = 3x^2 + 2x fonksiyonunun x = 1 noktasındaki türevi kaçtır?", "8", "number"),
    ("Kalkülüs", "0'dan 2'ye 3x^2 fonksiyonunun belirli integrali kaçtır?", "8", "number"),
    ("Lineer Cebir", "[[2, 1], [3, 4]] matrisinin determinantı kaçtır?", "5", "number"),
    ("Lineer Cebir", "(1, 2, 3) ve (4, 5, 6) vektörlerinin iç çarpımı kaçtır?", "32", "number"),
    ("Mantık", "P doğru, Q yanlış ise 'P ve Q' ifadesi doğru mudur? evet/hayır", "no", "yesno"),
    ("Mantık", "Hiçbir kuş memeli değildir. Serçe bir kuştur. Serçe memeli midir? evet/hayır", "no", "yesno"),
    ("Mantık", "'Eğer yağmur yağarsa yer ıslanır' önermesinde yağmur yağmış ve yer ıslanmışsa önerme doğru mudur? evet/hayır", "yes", "yesno"),
    ("Algoritma", "İkili aramanın en kötü durum zaman karmaşıklığı nedir? Big-O ile yazın.", "O(log n)", "text"),
    ("Algoritma", "Birleştirme sıralamasının (merge sort) zaman karmaşıklığı nedir? Big-O ile yazın.", "O(n log n)", "text"),
    ("Algoritma", "[5, 3, 9, 1] listesi artan sıralanırsa ilk eleman kaç olur?", "1", "number"),
    ("Graf", "10 düğümlü bir ağacın kaç kenarı vardır?", "9", "number"),
    ("İspat", "Teorem: iki tek sayının toplamı çifttir. İspat: a = 2m+1, b = 2k+1 ise a + b = 2(m + k + __). Boşluğu doldurun.", "1", "text"),
]


def build_turkish_eval() -> List[Dict]:
    return [{"id": f"tr{i:02d}", "category": c, "question": q, "answer": a, "answer_type": t, "difficulty": 1}
            for i, (c, q, a, t) in enumerate(_Q)]
