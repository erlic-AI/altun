"""Kodla üretilen, DOĞRULANABİLİR matematik problemleri + adım adım çözümler (İngilizce ve Türkçe).

Amaç: modele hesaplamayı 'scratchpad' ile öğretmek (basamak/parça parça). Cevaplar Python ile hesaplanır.
Biçim, eğitimdeki NuminaMath ile aynıdır:  "Problem: ...\nSolution: ... $\\boxed{cevap}$"
ALTUN değerlendirme sorularıyla birebir aynı soru metinleri ÜRETİLMEZ (exclude kümesi); ifadeler farklıdır.
"""
from __future__ import annotations

import math
import random
from typing import Callable, Dict, Iterator, List, Optional, Set

TR_LANG = "tr"


def _norm(s: str) -> str:
    return " ".join(s.lower().split())


def places(n: int) -> List[int]:
    """123 -> [100, 20, 3] (sıfırlar atlanır)."""
    s, out = str(abs(n)), []
    for i, ch in enumerate(s):
        if ch != "0":
            out.append(int(ch) * 10 ** (len(s) - 1 - i))
    return out


def rnum(r: random.Random, digits: int) -> int:
    return r.randint(10 ** (digits - 1), 10 ** digits - 1) if digits > 1 else r.randint(1, 9)


def box(x) -> str:
    return f"$\\boxed{{{x}}}$"


# ---------------------------------------------------------------- aritmetik
def gen_add(r, lang):
    d = r.choice([1, 2, 2, 3, 3, 4])
    a, b = rnum(r, d), rnum(r, r.choice([1, d, d]))
    cur, steps = a, []
    for p in places(b):
        steps.append(f"{cur} + {p} = {cur + p}")
        cur += p
    if lang == "tr":
        q = r.choice([f"{a} + {b} işleminin sonucu kaçtır?", f"{a} ile {b} sayılarını toplayınız."])
        s = f"{b} sayısını basamaklarına ayırıp sırayla ekleyelim:\n" + "\n".join(steps) + f"\nSonuç {box(a + b)} olur."
    else:
        q = r.choice([f"Find the sum of {a} and {b}.", f"Calculate {a} + {b}.", f"What do you get when you add {b} to {a}?"])
        s = f"Split {b} into its place values and add them one by one:\n" + "\n".join(steps) + f"\nThe sum is {box(a + b)}."
    return q, s, a + b


def gen_sub(r, lang):
    d = r.choice([1, 2, 2, 3, 3, 4])
    a, b = rnum(r, d), rnum(r, d)
    a, b = max(a, b), min(a, b)
    cur, steps = a, []
    for p in places(b):
        steps.append(f"{cur} - {p} = {cur - p}")
        cur -= p
    if lang == "tr":
        q = r.choice([f"{a} - {b} işleminin sonucu nedir?", f"{a} sayısından {b} çıkarılırsa kaç kalır?"])
        s = f"{b} sayısını basamaklarına ayırıp sırayla çıkaralım:\n" + "\n".join(steps) + f"\nSonuç {box(a - b)} olur."
    else:
        q = r.choice([f"Compute {a} minus {b}.", f"Subtract {b} from {a}.", f"What is the difference {a} - {b}?"])
        s = f"Split {b} into place values and subtract them one by one:\n" + "\n".join(steps) + f"\nThe difference is {box(a - b)}."
    return q, s, a - b


def gen_mul(r, lang):
    a, b = rnum(r, r.choice([1, 2, 2, 3])), rnum(r, r.choice([1, 2, 2, 3]))
    parts = [(a * p, f"{a} × {p} = {a * p}") for p in places(b)]
    total = " + ".join(str(v) for v, _ in parts)
    if lang == "tr":
        q = r.choice([f"{a} × {b} işlemini yapınız.", f"{a} ile {b} sayısının çarpımı kaçtır?"])
        s = f"{b} sayısını basamaklarına ayırıp kısmi çarpımları bulalım:\n" + "\n".join(t for _, t in parts)
        s += f"\nToplayalım: {total} = {a * b}\nÇarpım {box(a * b)} olur."
    else:
        q = r.choice([f"Multiply {a} by {b}.", f"What is {a} times {b}?", f"Evaluate the product {a} * {b}."])
        s = f"Split {b} into place values and compute partial products:\n" + "\n".join(t for _, t in parts)
        s += f"\nAdd them: {total} = {a * b}\nThe product is {box(a * b)}."
    return q, s, a * b


def gen_div(r, lang):
    b = rnum(r, r.choice([1, 2])) + 1
    qt = rnum(r, r.choice([1, 2, 3]))
    rem = r.randint(0, b - 1) if r.random() < 0.4 else 0
    a = b * qt + rem
    if lang == "tr":
        q = f"{a} sayısını {b} sayısına bölünüz; bölüm ve kalan kaçtır?" if rem else f"{a} ÷ {b} işleminin sonucu kaçtır?"
        s = f"{b} × {qt} = {b * qt} ve {a} - {b * qt} = {rem} olduğundan bölüm {qt}, kalan {rem} olur."
        s += f"\nCevap {box(qt if not rem else f'{qt} kalan {rem}')}."
    else:
        q = f"Divide {a} by {b}. What are the quotient and remainder?" if rem else f"What is {a} divided by {b}?"
        s = f"We look for the largest multiple of {b} not exceeding {a}: {b} × {qt} = {b * qt}, and {a} - {b * qt} = {rem}."
        s += f"\nSo the quotient is {qt} and the remainder is {rem}. Answer: {box(qt if not rem else f'{qt} R {rem}')}."
    return q, s, qt


# -------------------------------------------------------------------- cebir
def gen_linear(r, lang):
    x = r.randint(-12, 25)
    a = r.choice([2, 3, 4, 5, 6, 7, 8, 9, -2, -3])
    b = r.randint(-20, 30)
    c = a * x + b
    sign = f"+ {b}" if b >= 0 else f"- {-b}"
    if lang == "tr":
        q = f"{a}x {sign} = {c} denklemini çözünüz."
        s = f"Her iki taraftan {b} çıkarırsak {a}x = {c - b} elde ederiz.\nHer iki tarafı {a} ile bölersek x = {x} bulunur.\nCevap {box(x)}."
    else:
        q = r.choice([f"Solve {a}x {sign} = {c} for x.", f"Find x if {a}x {sign} = {c}."])
        s = f"Subtract {b} from both sides: {a}x = {c - b}.\nDivide both sides by {a}: x = {x}.\nThe solution is {box(x)}."
    return q, s, x


def gen_linear2(r, lang):
    x = r.randint(-9, 15)
    a, c = r.randint(3, 9), r.randint(1, 2)
    b, d = r.randint(-10, 20), r.randint(-10, 20)
    # a x + b = c x + d  ->  (a-c) x = d - b ;  d'yi çözüm tam sayı olacak şekilde ayarla
    d = (a - c) * x + b
    q = f"Solve for x: {a}x + {b} = {c}x + {d}." if lang != "tr" else f"{a}x + {b} = {c}x + {d} denkleminde x kaçtır?"
    if lang == "tr":
        s = f"x'li terimleri sola, sayıları sağa toplayalım: {a - c}x = {d - b}.\nBuradan x = {x} bulunur. Cevap {box(x)}."
    else:
        s = f"Move the x terms to the left and constants to the right: {a - c}x = {d - b}.\nSo x = {x}. The answer is {box(x)}."
    return q, s, x


def gen_quadratic(r, lang):
    p, q_ = r.randint(-9, 9), r.randint(-9, 9)
    b, c = -(p + q_), p * q_
    bs = f"+ {b}" if b >= 0 else f"- {-b}"
    cs = f"+ {c}" if c >= 0 else f"- {-c}"
    roots = sorted({p, q_})
    ans = ", ".join(str(v) for v in roots)
    if lang == "tr":
        qq = f"x² {bs}x {cs} = 0 denkleminin köklerini bulunuz."
        s = f"Toplamları {-b}, çarpımları {c} olan iki sayı arıyoruz: {p} ve {q_}.\nDenklem (x - {p})(x - {q_}) = 0 biçiminde çarpanlanır.\nKökler {box(ans)}."
    else:
        qq = f"Find all real roots of x^2 {bs}x {cs} = 0."
        s = f"We need two numbers with sum {-b} and product {c}: they are {p} and {q_}.\nSo the equation factors as (x - {p})(x - {q_}) = 0.\nThe roots are {box(ans)}."
    return qq, s, ans


# ------------------------------------------------------------ sayılar teorisi
def gen_gcd(r, lang):
    g = r.randint(2, 15)
    a, b = g * r.randint(2, 20), g * r.randint(2, 20)
    steps, x, y = [], max(a, b), min(a, b)
    while y:
        steps.append(f"{x} = {x // y} × {y} + {x % y}")
        x, y = y, x % y
    ans = x
    if lang == "tr":
        q = f"{a} ve {b} sayılarının en büyük ortak bölenini (EBOB) bulunuz."
        s = "Öklid algoritmasını uygulayalım:\n" + "\n".join(steps) + f"\nSon sıfırdan farklı kalan {ans} olduğundan EBOB {box(ans)}."
    else:
        q = r.choice([f"Use the Euclidean algorithm to find gcd({a}, {b}).", f"Determine the greatest common divisor of {a} and {b}."])
        s = "Apply the Euclidean algorithm:\n" + "\n".join(steps) + f"\nThe last nonzero remainder is {ans}, so the gcd is {box(ans)}."
    return q, s, ans


def gen_lcm(r, lang):
    a, b = r.randint(4, 40), r.randint(4, 40)
    g = math.gcd(a, b)
    ans = a * b // g
    q = f"Find lcm({a}, {b})." if lang != "tr" else f"{a} ve {b} sayılarının EKOK'unu bulunuz."
    if lang == "tr":
        s = f"EBOB({a}, {b}) = {g}. EKOK = {a} × {b} / {g} = {ans}. Cevap {box(ans)}."
    else:
        s = f"gcd({a}, {b}) = {g}. Using lcm = ab / gcd: {a} × {b} / {g} = {ans}. The answer is {box(ans)}."
    return q, s, ans


def gen_prime(r, lang):
    n = r.randint(20, 400)
    divs = [p for p in range(2, int(n ** 0.5) + 1) if n % p == 0]
    prime = not divs
    tested = [p for p in range(2, int(n ** 0.5) + 1)]
    if lang == "tr":
        q = f"{n} asal sayı mıdır? (evet/hayır)"
        s = (f"√{n} yaklaşık {int(n ** 0.5)} olduğundan 2'den {int(n ** 0.5)}'e kadar bölenleri deneriz. " +
             (f"Hiçbiri {n} sayısını bölmez, yani asaldır. Cevap {box('evet')}." if prime else f"{divs[0]} sayısı {n} sayısını böler ({n} = {divs[0]} × {n // divs[0]}), asal değildir. Cevap {box('hayır')}."))
    else:
        q = r.choice([f"Is {n} a prime number? Answer yes or no.", f"Determine whether {n} is prime."])
        s = (f"It suffices to test divisors up to √{n} ≈ {int(n ** 0.5)}. " +
             (f"None of {tested[0]}..{tested[-1]} divides {n}, so it is prime. Answer: {box('yes')}." if prime
              else f"{divs[0]} divides {n} since {n} = {divs[0]} × {n // divs[0]}, so it is composite. Answer: {box('no')}."))
    return q, s, "yes" if prime else "no"


def gen_modpow(r, lang):
    a, e, m = r.randint(2, 12), r.randint(2, 9), r.randint(5, 17)
    steps, cur = [], 1
    for i in range(1, e + 1):
        cur = cur * a % m
        steps.append(f"{a}^{i} ≡ {cur} (mod {m})")
    q = f"Compute {a}^{e} mod {m}." if lang != "tr" else f"{a}^{e} sayısının {m} ile bölümünden kalan kaçtır?"
    if lang == "tr":
        s = "Üsleri sırayla hesaplayıp her adımda mod alalım:\n" + "\n".join(steps) + f"\nKalan {box(cur)}."
    else:
        s = "Compute powers step by step, reducing modulo each time:\n" + "\n".join(steps) + f"\nThe remainder is {box(cur)}."
    return q, s, cur


def gen_factor(r, lang):
    n = r.choice([r.randint(12, 300) for _ in range(3)])
    f, m, p = [], n, 2
    while m > 1:
        while m % p == 0:
            f.append(p)
            m //= p
        p += 1
    ans = " × ".join(str(x) for x in f)
    q = f"Write {n} as a product of primes." if lang != "tr" else f"{n} sayısını asal çarpanlarına ayırınız."
    chain, cur = [], n
    for x in f:
        chain.append(f"{cur} ÷ {x} = {cur // x}")
        cur //= x
    s = ("Divide by the smallest prime factor repeatedly:\n" if lang != "tr" else "En küçük asal çarpana tekrar tekrar bölelim:\n") + "\n".join(chain) + \
        (f"\nSo {n} = {ans}. " if lang != "tr" else f"\nYani {n} = {ans}. ") + box(ans)
    return q, s, ans


# ------------------------------------------------------------- kombinatorik
def gen_comb(r, lang):
    n = r.randint(5, 12)
    k = r.randint(2, min(5, n - 1))
    ans = math.comb(n, k)
    num = " × ".join(str(n - i) for i in range(k))
    den = " × ".join(str(i + 1) for i in range(k))
    q = (f"In how many ways can a committee of {k} be chosen from {n} people?" if lang != "tr"
         else f"{n} kişi arasından {k} kişilik bir komite kaç farklı biçimde seçilebilir?")
    if lang == "tr":
        s = f"Sıra önemli olmadığından C({n},{k}) = ({num}) / ({den}) = {ans}. Cevap {box(ans)}."
    else:
        s = f"Order does not matter, so the count is C({n},{k}) = ({num}) / ({den}) = {ans}. The answer is {box(ans)}."
    return q, s, ans


def gen_perm(r, lang):
    n = r.randint(3, 8)
    ans = math.factorial(n)
    chain = " × ".join(str(i) for i in range(n, 0, -1))
    q = f"How many different orderings of {n} distinct objects are there?" if lang != "tr" else f"{n} farklı nesne kaç farklı biçimde sıralanabilir?"
    s = (f"There are {n}! = {chain} = {ans} orderings. Answer: {box(ans)}." if lang != "tr" else f"{n}! = {chain} = {ans} farklı sıralama vardır. Cevap {box(ans)}.")
    return q, s, ans


def gen_binom_sum(r, lang):
    n = r.randint(3, 10)
    ans = 2 ** n
    q = f"How many subsets does a set with {n} elements have?" if lang != "tr" else f"{n} elemanlı bir kümenin kaç alt kümesi vardır?"
    s = (f"Each element is either in or out, giving 2^{n} = {ans} subsets. Answer: {box(ans)}." if lang != "tr"
         else f"Her eleman ya vardır ya yoktur: 2^{n} = {ans}. Cevap {box(ans)}.")
    return q, s, ans


# ---------------------------------------------------------------- kalkülüs
def _poly_str(coefs):
    terms = []
    for p, c in sorted(coefs.items(), reverse=True):
        if c == 0:
            continue
        t = f"{abs(c) if abs(c) != 1 or p == 0 else ''}{'x' if p else ''}{f'^{p}' if p > 1 else ''}"
        terms.append(("-" if c < 0 else "+", t))
    out = ""
    for i, (sg, t) in enumerate(terms):
        out += (("-" if sg == "-" else "") if i == 0 else f" {sg} ") + t
    return out or "0"


def gen_deriv(r, lang):
    coefs = {p: r.randint(-5, 6) for p in range(0, r.randint(2, 4))}
    coefs[max(coefs)] = coefs[max(coefs)] or 2
    d = {p - 1: c * p for p, c in coefs.items() if p > 0 and c * p != 0}
    a = r.randint(-3, 4)
    val = sum(c * a ** p for p, c in d.items())
    q = f"If f(x) = {_poly_str(coefs)}, compute f'({a})." if lang != "tr" else f"f(x) = {_poly_str(coefs)} ise f'({a}) değeri kaçtır?"
    steps = [f"d/dx of {c}x^{p} is {c * p}x^{p - 1}" for p, c in sorted(coefs.items(), reverse=True) if p > 0 and c]
    s = (("Differentiate term by term:\n" if lang != "tr" else "Terim terim türev alalım:\n") + "\n".join(steps) +
         f"\nf'(x) = {_poly_str(d)}\n" + (f"Evaluating at x = {a}: " if lang != "tr" else f"x = {a} için: ") + f"{val}. " + box(val))
    return q, s, val


def gen_integral(r, lang):
    n, c = r.randint(1, 4), r.randint(1, 6)
    hi = r.randint(1, 4)
    # ∫_0^hi (c(n+1)) x^n dx = c hi^(n+1)
    coef = c * (n + 1)
    ans = c * hi ** (n + 1)
    q = (f"Evaluate the definite integral of {coef}x^{n} from 0 to {hi}." if lang != "tr" else f"{coef}x^{n} fonksiyonunun 0'dan {hi}'e belirli integralini hesaplayınız.")
    s = (f"An antiderivative is {c}x^{n + 1}. Evaluate: {c}·{hi}^{n + 1} - 0 = {ans}. " if lang != "tr" else f"Ters türev {c}x^{n + 1}'dir. Uç değerler: {c}·{hi}^{n + 1} - 0 = {ans}. ") + box(ans)
    return q, s, ans


# ------------------------------------------------------------ lineer cebir
def gen_det(r, lang):
    m = [r.randint(-6, 9) for _ in range(4)]
    ans = m[0] * m[3] - m[1] * m[2]
    q = (f"Find the determinant of the 2×2 matrix with rows ({m[0]}, {m[1]}) and ({m[2]}, {m[3]})." if lang != "tr"
         else f"Satırları ({m[0]}, {m[1]}) ve ({m[2]}, {m[3]}) olan 2×2 matrisin determinantı kaçtır?")
    s = f"det = ad - bc = {m[0]}·{m[3]} - {m[1]}·{m[2]} = {m[0] * m[3]} - {m[1] * m[2]} = {ans}. " + box(ans)
    return q, s, ans


def gen_dot(r, lang):
    u, v = [r.randint(-5, 9) for _ in range(3)], [r.randint(-5, 9) for _ in range(3)]
    prods = [a * b for a, b in zip(u, v)]
    ans = sum(prods)
    q = f"Compute u·v for u = {tuple(u)} and v = {tuple(v)}." if lang != "tr" else f"u = {tuple(u)} ve v = {tuple(v)} vektörlerinin iç çarpımı kaçtır?"
    s = " + ".join(f"({a})({b})" for a, b in zip(u, v)) + " = " + " + ".join(str(p) for p in prods) + f" = {ans}. " + box(ans)
    return q, s, ans


# ------------------------------------------------------------------- mantık
def gen_logic(r, lang):
    P, Q = r.random() < .5, r.random() < .5
    form = r.choice(["and", "or", "imp", "xor", "nand"])
    name = {"and": "P AND Q", "or": "P OR Q", "imp": "P → Q", "xor": "P XOR Q", "nand": "NOT (P AND Q)"}[form]
    val = {"and": P and Q, "or": P or Q, "imp": (not P) or Q, "xor": P != Q, "nand": not (P and Q)}[form]
    tf = lambda b: "true" if b else "false"
    q = (f"Let P be {tf(P)} and Q be {tf(Q)}. Is {name} true or false?" if lang != "tr" else
         f"P {('doğru' if P else 'yanlış')}, Q {('doğru' if Q else 'yanlış')} ise {name} ifadesi doğru mu yanlış mı?")
    s = f"Substitute P = {tf(P)}, Q = {tf(Q)} into {name}. " + ("Recall P → Q is false only when P is true and Q is false. " if form == "imp" else "") + f"The result is {tf(val)}. " + box(tf(val))
    return q, s, tf(val)


# -------------------------------------------------------------------- graf
def gen_graph(r, lang):
    kind = r.choice(["complete", "tree", "degree", "cycle"])
    n = r.randint(4, 12)
    if kind == "complete":
        ans, q, s = n * (n - 1) // 2, f"How many edges are in the complete graph on {n} vertices?", f"Every pair of vertices is joined: C({n},2) = {n}·{n - 1}/2 = {n * (n - 1) // 2}."
    elif kind == "tree":
        ans, q, s = n - 1, f"A connected acyclic graph has {n} vertices. How many edges does it have?", f"A tree on n vertices has n - 1 edges, so {n} - 1 = {n - 1}."
    elif kind == "cycle":
        ans, q, s = n, f"How many edges does the cycle graph C_{n} have?", f"A cycle on {n} vertices has one edge per vertex: {n}."
    else:
        e = r.randint(n, 2 * n)
        ans, q, s = 2 * e, f"A graph has {e} edges. What is the sum of all vertex degrees?", f"Each edge contributes 2 to the degree sum (handshake lemma): 2 × {e} = {2 * e}."
    if lang == "tr":
        q = {"complete": f"{n} köşeli tam çizgenin kaç kenarı vardır?", "tree": f"{n} köşeli bir ağacın kaç kenarı vardır?",
             "cycle": f"{n} köşeli çevrim çizgesinin kaç kenarı vardır?", "degree": f"{ans // 2} kenarlı bir çizgede derecelerin toplamı kaçtır?"}[kind]
    return q, s + f" Answer: {box(ans)}." if lang != "tr" else s + f" Cevap {box(ans)}.", ans


def gen_bfs(r, lang):
    n = r.randint(5, 7)
    edges = {(i, i + 1) for i in range(n - 1)}
    for _ in range(r.randint(1, 3)):
        a, b = r.randint(0, n - 1), r.randint(0, n - 1)
        if a != b:
            edges.add((min(a, b), max(a, b)))
    adj = {i: [] for i in range(n)}
    for a, b in sorted(edges):
        adj[a].append(b)
        adj[b].append(a)
    dist, order, layer = {0: 0}, [0], [0]
    lines = []
    while layer:
        nxt = []
        for u in layer:
            for v in sorted(adj[u]):
                if v not in dist:
                    dist[v] = dist[u] + 1
                    nxt.append(v)
        if nxt:
            lines.append(f"distance {dist[nxt[0]]}: {nxt}")
        layer = nxt
    t = n - 1
    q = f"An undirected graph has edges {sorted(edges)}. Using BFS from vertex 0, what is the shortest distance to vertex {t}?" if lang != "tr" else \
        f"Kenarları {sorted(edges)} olan yönsüz çizgede 0 köşesinden {t} köşesine en kısa yol kaç kenardır?"
    s = ("BFS layers from vertex 0:\n" if lang != "tr" else "0 köşesinden BFS katmanları:\n") + "\n".join(lines) + \
        (f"\nVertex {t} is in layer {dist[t]}. " if lang != "tr" else f"\n{t} köşesi {dist[t]}. katmandadır. ") + box(dist[t])
    return q, s, dist[t]


# ------------------------------------------------------------- algoritmalar
def gen_sort_trace(r, lang):
    arr = [r.randint(1, 60) for _ in range(r.randint(4, 7))]
    s_arr = sorted(arr)
    k = r.randint(1, len(arr))
    q = f"Sort {arr} in ascending order. What is the {k}-th smallest element?" if lang != "tr" else f"{arr} dizisini küçükten büyüğe sıralayınız. {k}. en küçük eleman kaçtır?"
    s = (f"Sorted: {s_arr}. " if lang != "tr" else f"Sıralı dizi: {s_arr}. ") + (f"The {k}-th element is {s_arr[k - 1]}. " if lang != "tr" else f"{k}. eleman {s_arr[k - 1]}. ") + box(s_arr[k - 1])
    return q, s, s_arr[k - 1]


def gen_binsearch(r, lang):
    n = r.randint(8, 1000)
    steps = math.ceil(math.log2(n + 1))
    q = f"What is the maximum number of comparisons binary search needs on a sorted array of {n} elements?" if lang != "tr" else \
        f"{n} elemanlı sıralı bir dizide ikili arama en çok kaç karşılaştırma yapar?"
    s = (f"Each step halves the range. We need the smallest k with 2^k ≥ {n + 1}: 2^{steps} = {2 ** steps}. So k = {steps}. " if lang != "tr" else
         f"Her adım aralığı yarıya böler. 2^k ≥ {n + 1} koşulunu sağlayan en küçük k: 2^{steps} = {2 ** steps}. ") + box(steps)
    return q, s, steps


def gen_recurrence(r, lang):
    a = r.choice([1, 2, 3, 4])
    b = 2
    f = r.choice([("1", 0), ("n", 1), ("n^2", 2)])
    crit = math.log(a, b)
    if abs(f[1] - crit) < 1e-9:
        ans = f"Θ(n^{f[1]} log n)" if f[1] else "Θ(log n)"
        case = "f(n) matches n^{log_b a}"
    elif f[1] < crit:
        ans = f"Θ(n^{int(crit)})" if crit == int(crit) else f"Θ(n^{crit:.2f})"
        case = "n^{log_b a} dominates f(n)"
    else:
        ans = f"Θ(n^{f[1]})" if f[1] > 1 else "Θ(n)"
        case = "f(n) dominates"
    q = f"Solve T(n) = {a}T(n/{b}) + {f[0]} using the Master theorem." if lang != "tr" else f"T(n) = {a}T(n/{b}) + {f[0]} özyinelemesini Master teoremi ile çözünüz."
    s = f"Here a = {a}, b = {b}, so log_b a = {crit:g}. Compare with f(n) = {f[0]} (exponent {f[1]}): {case}. Therefore T(n) = {ans}. " + box(ans)
    return q, s, ans


def gen_fib(r, lang):
    n = r.randint(6, 25)
    a, b, seq = 0, 1, [0, 1]
    for _ in range(n - 1):
        a, b = b, a + b
        seq.append(b)
    q = f"Using F(0)=0, F(1)=1, F(k)=F(k-1)+F(k-2), compute F({n})." if lang != "tr" else f"F(0)=0, F(1)=1, F(k)=F(k-1)+F(k-2) ise F({n}) kaçtır?"
    s = ("Compute iteratively: " if lang != "tr" else "Sırayla hesaplayalım: ") + ", ".join(str(x) for x in seq[:n + 1]) + f". F({n}) = {seq[n]}. " + box(seq[n])
    return q, s, seq[n]


def gen_coin_dp(r, lang):
    coins = sorted(r.sample([1, 2, 3, 4, 5, 6, 10], 3))
    amt = r.randint(6, 18)
    INF = 10 ** 9
    dp = [0] + [INF] * amt
    for v in range(1, amt + 1):
        for c in coins:
            if v >= c and dp[v - c] + 1 < dp[v]:
                dp[v] = dp[v - c] + 1
    ans = dp[amt] if dp[amt] < INF else -1
    q = f"With coin denominations {coins}, what is the minimum number of coins to make {amt}? (-1 if impossible)" if lang != "tr" else \
        f"{coins} değerli madeni paralarla {amt} elde etmek için gereken en az para sayısı kaçtır? (olanaksızsa -1)"
    s = ("Dynamic programming table dp[v] = min coins for v: " if lang != "tr" else "dp[v] = v için en az para sayısı: ") + ", ".join(
        f"dp[{v}]={dp[v] if dp[v] < INF else 'inf'}" for v in range(amt + 1)) + f". {box(ans)}"
    return q, s, ans


def gen_complexity_fact(r, lang):
    facts = [("Is the Hamiltonian path problem in NP?", "yes", "A Hamiltonian path can be verified in polynomial time given the path."),
             ("Is 2-colorability of a graph decidable in polynomial time?", "yes", "Run BFS and check that no edge joins two vertices of the same layer parity."),
             ("Is vertex cover NP-complete?", "yes", "It is one of Karp's 21 NP-complete problems."),
             ("Can comparison sorting be done in o(n log n) time in the worst case?", "no", "The decision-tree argument gives an Ω(n log n) lower bound."),
             ("Is the shortest path problem with nonnegative weights solvable in polynomial time?", "yes", "Dijkstra's algorithm solves it."),
             ("Is the traveling salesman decision problem NP-complete?", "yes", "It is NP-complete by reduction from Hamiltonian cycle."),
             ("Is every regular language decidable?", "yes", "A DFA decides it in linear time."),
             ("Is the language {a^n b^n} regular?", "no", "The pumping lemma for regular languages rules it out."),
             ("Is integer multiplication of n-digit numbers possible in o(n^2) time?", "yes", "Karatsuba's algorithm runs in O(n^1.585).")]
    qq, ans, why = r.choice(facts)
    if lang == "tr":
        return qq + " (Cevap: evet/hayır)", f"{why} Cevap {box('evet' if ans == 'yes' else 'hayır')}.", ans
    return qq + " Answer yes or no.", f"{why} Answer: {box(ans)}.", ans


# ======================================================= v3: ek üreteçler (zayıf kategoriler + Türkçe)
# Not: ALTUN değerlendirme setiyle AYNI BECERİ ALANLARINI hedefler (bu yüzden o set artık "dağılım içi"dir);
# soru metinleri farklı biçimlendirilir ve birebir eval soruları exclude ile atlanır.
def _T(lang, en, tr):
    return tr if lang == "tr" else en


def gen_loop_cx(r, lang):
    patterns = [
        ("for i in 1..n:\n    work()", "O(n)", "The body runs once per iteration and there are n iterations.", "Gövde n kez çalışır."),
        ("for i in 1..n:\n    for j in 1..n:\n        work()", "O(n^2)", "The inner loop runs n times for each of the n outer iterations: n·n.", "İç döngü her dış adımda n kez çalışır: n·n."),
        ("for i in 1..n:\n    for j in i..n:\n        work()", "O(n^2)", "The inner loop runs n, n-1, ..., 1 times, a sum of n(n+1)/2, which is Θ(n^2).", "İç döngü n, n-1, ..., 1 kez çalışır; toplam n(n+1)/2 = Θ(n^2)."),
        ("i = 1\nwhile i < n:\n    work()\n    i = 2 * i", "O(log n)", "i doubles each time, so the loop runs about log2(n) times.", "i her adımda ikiye katlanır; döngü yaklaşık log2(n) kez çalışır."),
        ("i = n\nwhile i > 1:\n    work()\n    i = i // 2", "O(log n)", "i is halved each time, so there are about log2(n) iterations.", "i her adımda yarılanır; yaklaşık log2(n) tekrar olur."),
        ("for i in 1..n:\n    j = 1\n    while j < n:\n        work()\n        j = 2 * j", "O(n log n)", "The outer loop runs n times and the inner loop about log2(n) times: n·log n.", "Dış döngü n kez, iç döngü yaklaşık log2(n) kez: n·log n."),
        ("for i in 1..n:\n    for j in 1..n:\n        for k in 1..n:\n            work()", "O(n^3)", "Three nested loops of n iterations each give n·n·n.", "İç içe üç döngü n·n·n adım yapar."),
        ("i = 1\nwhile i * i <= n:\n    work()\n    i = i + 1", "O(sqrt(n))", "The loop stops when i exceeds sqrt(n), so it runs about sqrt(n) times.", "Döngü i, sqrt(n)'i aşınca durur; yaklaşık sqrt(n) kez çalışır."),
        ("work()\nwork()\nwork()", "O(1)", "The number of steps does not depend on n.", "Adım sayısı n'ye bağlı değildir."),
        ("for i in 1..n:\n    work()\nfor j in 1..n:\n    work()", "O(n)", "Two sequential loops give n + n = 2n steps, which is O(n).", "Ardışık iki döngü n + n = 2n adım yapar: O(n)."),
    ]
    code, ans, why_en, why_tr = r.choice(patterns)
    head = r.choice(["What is the time complexity of this pseudocode in terms of n?", "Analyze the running time of the following pseudocode as a function of n.",
                     "Give the tightest big-O bound on the number of calls to work() made by this code."])
    q = _T(lang, f"{head} Answer in big-O.\n{code}", f"Aşağıdaki sözde kodun n cinsinden zaman karmaşıklığı nedir? Big-O ile yazınız.\n{code}")
    s = _T(lang, why_en, why_tr) + f" {_T(lang, 'Therefore the complexity is', 'Dolayısıyla karmaşıklık')} {box(ans)}."
    return q, s, ans


def gen_master(r, lang):
    a, b = r.choice([(1, 2), (2, 2), (3, 2), (4, 2), (8, 2), (9, 3), (2, 4), (3, 3), (1, 3), (5, 2), (4, 4)])
    d = r.choice([0, 1, 2])
    crit = math.log(a, b)
    f = {0: "1", 1: "n", 2: "n^2"}[d]
    if abs(crit - d) < 1e-9:
        ans = f"O(n^{d} log n)" if d > 1 else ("O(n log n)" if d == 1 else "O(log n)")
        case = f"log_{b}({a}) = {d} equals the exponent of f(n), so the work is balanced (case 2)"
        case_tr = f"log_{b}({a}) = {d}, f(n)'in üssüne eşit; iş dengelidir (durum 2)"
    elif crit > d:
        k = int(round(crit)) if abs(crit - round(crit)) < 1e-9 else None
        ans = f"O(n^{k})" if k not in (None, 1) else ("O(n)" if k == 1 else f"O(n^(log_{b} {a}))")
        case = f"log_{b}({a}) {'= ' + str(k) if k else '≈ ' + format(crit, '.2f')} is larger than {d}, so the leaves dominate (case 1)"
        case_tr = f"log_{b}({a}) {'= ' + str(k) if k else '≈ ' + format(crit, '.2f')}, {d}'den büyük; yapraklar baskındır (durum 1)"
    else:
        ans = {0: "O(1)", 1: "O(n)", 2: "O(n^2)"}[d]
        case = f"log_{b}({a}) ≈ {crit:.2f} is smaller than {d}, so the root work f(n) dominates (case 3)"
        case_tr = f"log_{b}({a}) ≈ {crit:.2f}, {d}'den küçük; kök işi f(n) baskındır (durum 3)"
    rec = f"T(n) = {a if a != 1 else ''}T(n/{b}) + {f}"
    q = _T(lang, f"Use the Master theorem to solve {rec}. Answer in big-O.", f"Ana teoremi (Master theorem) kullanarak {rec} özyinelemesini çözün. Big-O ile yazınız.")
    s = _T(lang, f"Here a = {a}, b = {b}, f(n) = {f}. {case}. Hence T(n) is ", f"Burada a = {a}, b = {b}, f(n) = {f}. {case_tr}. Dolayısıyla T(n) = ") + box(ans)
    return q, s, ans


_DS_FACTS = [
    ("inserting one element into a binary heap of n elements", "bir ikili yığına (heap) eleman eklemenin", "O(log n)", "The new element sifts up at most the height log n.", "Yeni eleman en fazla yükseklik kadar (log n) yukarı çıkar."),
    ("extracting the minimum from a binary min-heap with n elements", "n elemanlı ikili min-yığından en küçüğü çıkarmanın", "O(log n)", "The last element is moved to the root and sifts down log n levels.", "Son eleman köke taşınır ve log n seviye aşağı iner."),
    ("building a binary heap from an unsorted array of n elements (bottom-up)", "sırasız n elemanlı diziden aşağıdan yukarı ikili yığın kurmanın", "O(n)", "Most nodes are near the leaves, so the sum of sift-down costs is linear.", "Düğümlerin çoğu yapraklara yakındır; toplam maliyet doğrusaldır."),
    ("looking up a key in a hash table with a good hash function (average case)", "iyi bir özet fonksiyonlu karma tablosunda anahtar aramanın (ortalama durum)", "O(1)", "Expected chain length is constant when the load factor is bounded.", "Yük faktörü sınırlıysa beklenen zincir uzunluğu sabittir."),
    ("searching a balanced binary search tree with n nodes", "n düğümlü dengeli ikili arama ağacında aramanın", "O(log n)", "The height of a balanced tree is O(log n).", "Dengeli ağacın yüksekliği O(log n)'dir."),
    ("searching an unbalanced binary search tree with n nodes (worst case)", "dengesiz ikili arama ağacında aramanın (en kötü durum)", "O(n)", "The tree can degenerate into a chain of height n.", "Ağaç n yüksekliğinde bir zincire dönüşebilir."),
    ("accessing the i-th element of an array", "bir dizinin i-inci elemanına erişmenin", "O(1)", "The address is computed directly from the index.", "Adres doğrudan indeksten hesaplanır."),
    ("searching for a value in an unsorted singly linked list of n nodes", "sırasız n düğümlü tek bağlı listede değer aramanın", "O(n)", "In the worst case every node is visited.", "En kötü durumda her düğüm gezilir."),
    ("insertion sort on n elements (worst case)", "n elemanlı ekleme sıralamasının (en kötü durum)", "O(n^2)", "Each element may move past all earlier ones: 1 + 2 + ... + (n-1).", "Her eleman öncekilerin hepsini geçebilir: 1 + 2 + ... + (n-1)."),
    ("insertion sort on an already sorted array of n elements", "zaten sıralı n elemanlı dizide ekleme sıralamasının", "O(n)", "Each element needs only one comparison.", "Her eleman için tek karşılaştırma yeterlidir."),
    ("heapsort on n elements", "n elemanlı yığın sıralamasının (heapsort)", "O(n log n)", "Build the heap in O(n), then n extractions cost O(log n) each.", "Yığın O(n)'de kurulur, sonra n çıkarma işlemi O(log n)'dir."),
    ("depth-first search on a graph with V vertices and E edges (adjacency list)", "V düğümlü E kenarlı grafta (komşuluk listesi) derinlik öncelikli aramanın", "O(V+E)", "Every vertex and every edge is processed a constant number of times.", "Her düğüm ve kenar sabit sayıda işlenir."),
    ("Dijkstra's algorithm with a binary heap on V vertices and E edges", "ikili yığınlı Dijkstra algoritmasının (V düğüm, E kenar)", "O((V+E) log V)", "Each edge relaxation may cost one heap operation of O(log V).", "Her kenar gevşetmesi O(log V) maliyetli bir yığın işlemi gerektirebilir."),
    ("the Floyd–Warshall algorithm on V vertices", "V düğümlü grafta Floyd–Warshall algoritmasının", "O(V^3)", "Three nested loops over V vertices.", "V düğüm üzerinde üç iç içe döngü."),
    ("naive multiplication of two n×n matrices", "iki n×n matrisin klasik çarpımının", "O(n^3)", "Each of the n^2 output entries needs n multiplications.", "n^2 çıktı elemanının her biri n çarpma ister."),
    ("pushing onto a stack implemented with a dynamic array (amortized)", "dinamik dizi ile gerçeklenen yığına eleman eklemenin (amortize)", "O(1)", "Occasional resizes are spread over many pushes.", "Ara sıra yapılan büyütmeler çok sayıda ekleme üzerine yayılır."),
    ("merging two sorted lists of sizes m and n", "m ve n uzunluğundaki iki sıralı listeyi birleştirmenin", "O(m+n)", "Each step outputs one element after one comparison.", "Her adım bir karşılaştırmayla bir eleman çıkarır."),
    ("binary search on a sorted array of n elements", "sıralı n elemanlı dizide ikili aramanın", "O(log n)", "Each probe halves the remaining range.", "Her deneme kalan aralığı yarıya indirir."),
    ("quicksort on n elements when pivots are always the smallest element", "pivotun hep en küçük eleman seçildiği hızlı sıralamanın", "O(n^2)", "The partitions have sizes 0 and n-1, giving n + (n-1) + ... + 1 work.", "Bölümler 0 ve n-1 boyutundadır: n + (n-1) + ... + 1 iş."),
    ("merge sort on n elements", "n elemanlı birleştirme sıralamasının", "O(n log n)", "There are log n levels and each level does O(n) merging work.", "log n seviye vardır ve her seviye O(n) birleştirme işi yapar."),
]


def gen_ds_cost(r, lang):
    en, tr, ans, why_en, why_tr = r.choice(_DS_FACTS)
    if lang == "tr":
        q = f"{tr[0].upper() + tr[1:]} zaman karmaşıklığı nedir? Big-O ile yazınız."
        return q, f"{why_tr} Cevap {box(ans)}.", ans
    q = r.choice([f"What is the time complexity of {en}? Answer in big-O.", f"State the big-O running time for {en}."])
    return q, f"{why_en} The answer is {box(ans)}.", ans


def gen_kth(r, lang):
    n = r.randint(5, 9)
    a = [r.randint(1, 60) for _ in range(n)]
    s = sorted(a)
    kind = r.choice(["kth_small", "kth_large", "median", "two_small_sum", "distinct", "range"])
    if kind == "kth_small":
        k = r.randint(2, 4)
        ans, text = s[k - 1], _T(lang, f"Which value is the {k}th smallest in {a}?", f"{a} listesinde {k}. en küçük değer hangisidir?")
    elif kind == "kth_large":
        k = r.randint(1, 3)
        ans, text = s[-k], _T(lang, f"Find the {k}th largest element of {a}.", f"{a} listesinin {k}. en büyük elemanını bulun.")
    elif kind == "median":
        a = a if len(a) % 2 else a[:-1]
        s = sorted(a)
        ans, text = s[len(s) // 2], _T(lang, f"What is the median of {a}?", f"{a} listesinin medyanı kaçtır?")
    elif kind == "two_small_sum":
        ans, text = s[0] + s[1], _T(lang, f"What is the sum of the two smallest elements of {a}?", f"{a} listesindeki en küçük iki elemanın toplamı kaçtır?")
    elif kind == "distinct":
        ans, text = len(set(a)), _T(lang, f"How many distinct values does {a} contain?", f"{a} listesinde kaç farklı değer vardır?")
    else:
        ans, text = max(a) - min(a), _T(lang, f"What is max - min for the list {a}?", f"{a} listesi için en büyük - en küçük farkı kaçtır?")
    sol = _T(lang, f"Sorted ascending: {s}. ", f"Artan sıralı hali: {s}. ") + _T(lang, "So the answer is ", "Dolayısıyla cevap ") + box(ans) + "."
    if kind == "median":
        sol = _T(lang, f"Sorted: {s}. The middle element (position {len(s) // 2 + 1}) is ", f"Sıralı: {s}. Ortadaki eleman ({len(s) // 2 + 1}. sıra) ") + box(ans) + "."
    return text, sol, ans


def gen_cmp_count(r, lang):
    kind = r.choice(["scan_max", "sel_sort", "bin_probe", "merge", "bubble_swaps", "minmax"])
    if kind == "scan_max":
        n = r.randint(3, 40)
        ans = n - 1
        q = _T(lang, f"Finding the maximum of {n} unsorted numbers by scanning once takes how many comparisons in the worst case?", f"Sırasız {n} sayının en büyüğünü tek taramayla bulmak en kötü durumda kaç karşılaştırma ister?")
        s = _T(lang, f"Each element except the first is compared once: {n} - 1 = {ans}. ", f"İlk eleman dışındaki her eleman bir kez karşılaştırılır: {n} - 1 = {ans}. ")
    elif kind == "sel_sort":
        n = r.randint(3, 20)
        ans = n * (n - 1) // 2
        q = _T(lang, f"How many comparisons does selection sort make on {n} elements?", f"Seçme sıralaması {n} elemanda kaç karşılaştırma yapar?")
        s = f"(n-1) + (n-2) + ... + 1 = n(n-1)/2 = {n}·{n - 1}/2 = {ans}. "
    elif kind == "bin_probe":
        n = r.randint(3, 1000)
        ans = int(math.floor(math.log2(n))) + 1
        q = _T(lang, f"What is the maximum number of probes binary search needs on a sorted array of {n} elements?", f"Sıralı {n} elemanlı dizide ikili arama en fazla kaç deneme yapar?")
        s = f"floor(log2({n})) + 1 = {ans - 1} + 1 = {ans}. "
    elif kind == "merge":
        m, n = r.randint(2, 15), r.randint(2, 15)
        ans = m + n - 1
        q = _T(lang, f"Merging sorted lists of lengths {m} and {n} takes at most how many comparisons?", f"Uzunlukları {m} ve {n} olan sıralı listeleri birleştirmek en fazla kaç karşılaştırma ister?")
        s = f"{m} + {n} - 1 = {ans}. "
    elif kind == "bubble_swaps":
        a = r.sample(range(1, 30), r.randint(4, 7))
        pairs = [(a[i], a[j]) for i in range(len(a)) for j in range(i + 1, len(a)) if a[i] > a[j]]
        ans = len(pairs)
        q = _T(lang, f"How many swaps does bubble sort perform on {a}?", f"Kabarcık sıralaması {a} listesinde kaç yer değiştirme yapar?")
        s = _T(lang, f"Swaps equal the number of inversions (pairs i<j with a[i]>a[j]). Inversions: {pairs}. Count = {ans}. ",
               f"Yer değiştirme sayısı ters çevrilmiş çiftlerin (i<j, a[i]>a[j]) sayısına eşittir. Çiftler: {pairs}. Sayı = {ans}. ")
    else:
        n = r.randint(2, 30)
        ans = 2 * (n - 1)
        q = _T(lang, f"Finding both the minimum and the maximum of {n} numbers with two separate scans uses how many comparisons?", f"{n} sayının en küçüğünü ve en büyüğünü iki ayrı taramayla bulmak kaç karşılaştırma ister?")
        s = f"2·({n} - 1) = {ans}. "
    return q, s + box(ans), ans


def gen_stack_queue(r, lang):
    is_stack = r.random() < .5
    ops, cont, out, nxt = [], [], [], r.randint(1, 5)
    for _ in range(r.randint(5, 8)):
        if cont and r.random() < .4:
            v = cont.pop() if is_stack else cont.pop(0)
            ops.append("pop" if is_stack else "dequeue")
            out.append(v)
        else:
            nxt += r.randint(1, 4)
            cont.append(nxt)
            ops.append(f"{'push' if is_stack else 'enqueue'}({nxt})")
    if not out:
        return gen_stack_queue(r, lang)
    name = _T(lang, "stack" if is_stack else "queue", "yığın" if is_stack else "kuyruk")
    q = _T(lang, f"Starting from an empty {name}, apply: {', '.join(ops)}. What is the last value removed?", f"Boş bir {name} ile başlayıp şu işlemleri uygulayın: {', '.join(ops)}. En son çıkarılan değer nedir?")
    rule = _T(lang, "last-in first-out" if is_stack else "first-in first-out", "son giren ilk çıkar" if is_stack else "ilk giren ilk çıkar")
    s = _T(lang, f"The {name} is {rule}. Removed values in order: {out}. ", f"{name.capitalize()} {rule} çalışır. Çıkan değerler sırayla: {out}. ") + box(out[-1])
    return q, s, out[-1]


def gen_dfs_order(r, lang):
    n = r.randint(5, 7)
    edges = set()
    for v in range(1, n):
        edges.add((r.randint(0, v - 1), v))
    for _ in range(r.randint(0, 2)):
        a, b = r.sample(range(n), 2)
        edges.add((min(a, b), max(a, b)))
    adj = {i: sorted({b for a, b in edges if a == i} | {a for a, b in edges if b == i}) for i in range(n)}
    seen, order = set(), []

    def dfs(u):
        seen.add(u)
        order.append(u)
        for w in adj[u]:
            if w not in seen:
                dfs(w)
    dfs(0)
    el = sorted(edges)
    ans = " ".join(map(str, order))
    q = _T(lang, f"An undirected graph has edges {el}. List the vertices in the order depth-first search visits them from vertex 0, always taking the smallest-numbered unvisited neighbor first.",
           f"Yönsüz bir grafın kenarları {el}. 0 düğümünden başlayıp her zaman en küçük numaralı ziyaret edilmemiş komşuya giderek DFS ziyaret sırasını yazın.")
    s = _T(lang, f"Adjacency lists: {adj}. Following DFS: {ans}. ", f"Komşuluk listeleri: {adj}. DFS sırası: {ans}. ") + box(ans)
    return q, s, ans


def _m2(r):
    return [[r.randint(-5, 9) for _ in range(2)] for _ in range(2)]


def gen_linalg(r, lang):
    kind = r.choice(["det3", "trace", "matvec", "matmul", "norm2", "eig", "det2", "transpose_entry"])
    if kind == "det3":
        m = [[r.randint(-3, 5) for _ in range(3)] for _ in range(3)]
        (a, b, c), (d, e, f), (g, h, i) = m
        t1, t2, t3 = a * (e * i - f * h), b * (d * i - f * g), c * (d * h - e * g)
        ans = t1 - t2 + t3
        q = _T(lang, f"Compute the determinant of the 3×3 matrix with rows {tuple(m[0])}, {tuple(m[1])}, {tuple(m[2])}.", f"Satırları {tuple(m[0])}, {tuple(m[1])}, {tuple(m[2])} olan 3×3 matrisin determinantını hesaplayın.")
        s = _T(lang, f"Expand along the first row: {a}·({e}·{i} - {f}·{h}) - {b}·({d}·{i} - {f}·{g}) + {c}·({d}·{h} - {e}·{g}) = {t1} - ({t2}) + ({t3}) = {ans}. ",
               f"İlk satıra göre açın: {a}·({e}·{i} - {f}·{h}) - {b}·({d}·{i} - {f}·{g}) + {c}·({d}·{h} - {e}·{g}) = {t1} - ({t2}) + ({t3}) = {ans}. ")
    elif kind == "trace":
        n = r.randint(2, 4)
        m = [[r.randint(-9, 9) for _ in range(n)] for _ in range(n)]
        diag = [m[i][i] for i in range(n)]
        ans = sum(diag)
        q = _T(lang, f"What is the trace of the {n}×{n} matrix {m}?", f"{n}×{n} boyutlu {m} matrisinin izi (trace) kaçtır?")
        s = _T(lang, f"The trace is the sum of the diagonal entries: {' + '.join(map(str, diag))} = {ans}. ", f"İz, köşegen elemanların toplamıdır: {' + '.join(map(str, diag))} = {ans}. ")
    elif kind == "matvec":
        m, v = _m2(r), [r.randint(-4, 6) for _ in range(2)]
        res = [m[0][0] * v[0] + m[0][1] * v[1], m[1][0] * v[0] + m[1][1] * v[1]]
        idx = r.randint(0, 1)
        ans = res[idx]
        q = _T(lang, f"Let A = {m} and x = {v}. What is entry {idx + 1} of Ax?", f"A = {m} ve x = {v} olsun. Ax çarpımının {idx + 1}. elemanı kaçtır?")
        s = _T(lang, f"Entry {idx + 1} is row {idx + 1} of A dotted with x: {m[idx][0]}·{v[0]} + {m[idx][1]}·{v[1]} = {ans}. ", f"{idx + 1}. eleman, A'nın {idx + 1}. satırı ile x'in iç çarpımıdır: {m[idx][0]}·{v[0]} + {m[idx][1]}·{v[1]} = {ans}. ")
    elif kind == "matmul":
        A, B = _m2(r), _m2(r)
        i, j = r.randint(0, 1), r.randint(0, 1)
        ans = A[i][0] * B[0][j] + A[i][1] * B[1][j]
        q = _T(lang, f"For A = {A} and B = {B}, what is the entry in row {i + 1}, column {j + 1} of AB?", f"A = {A} ve B = {B} için AB çarpımının {i + 1}. satır, {j + 1}. sütun elemanı kaçtır?")
        s = _T(lang, f"Row {i + 1} of A times column {j + 1} of B: {A[i][0]}·{B[0][j]} + {A[i][1]}·{B[1][j]} = {ans}. ", f"A'nın {i + 1}. satırı ile B'nin {j + 1}. sütunu: {A[i][0]}·{B[0][j]} + {A[i][1]}·{B[1][j]} = {ans}. ")
    elif kind == "norm2":
        v = [r.randint(-6, 7) for _ in range(r.randint(2, 4))]
        ans = sum(x * x for x in v)
        q = _T(lang, f"What is the squared Euclidean norm of the vector {tuple(v)}?", f"{tuple(v)} vektörünün Öklid normunun karesi kaçtır?")
        s = " + ".join(f"({x})^2" for x in v) + " = " + " + ".join(str(x * x) for x in v) + f" = {ans}. "
    elif kind == "eig":
        a, b, d = r.randint(-4, 8), r.randint(-3, 5), r.randint(-4, 8)
        ans = max(a, d)
        q = _T(lang, f"What is the largest eigenvalue of the upper triangular matrix [[{a}, {b}], [0, {d}]]?", f"[[{a}, {b}], [0, {d}]] üst üçgen matrisinin en büyük özdeğeri kaçtır?")
        s = _T(lang, f"The eigenvalues of a triangular matrix are its diagonal entries {a} and {d}. The larger is ", f"Üçgen matrisin özdeğerleri köşegen elemanlarıdır: {a} ve {d}. Büyük olan ") + box(ans) + ". "
        return q, s.rstrip(), ans
    elif kind == "det2":
        m = _m2(r)
        ans = m[0][0] * m[1][1] - m[0][1] * m[1][0]
        q = _T(lang, f"Is the matrix {m} invertible? Answer yes or no.", f"{m} matrisi tersinir midir? evet/hayır olarak cevaplayın.")
        yn = "yes" if ans != 0 else "no"
        yn_tr = "evet" if ans != 0 else "hayır"
        s = _T(lang, f"det = {m[0][0]}·{m[1][1]} - {m[0][1]}·{m[1][0]} = {ans}. A matrix is invertible exactly when det ≠ 0, so the answer is ", f"det = {m[0][0]}·{m[1][1]} - {m[0][1]}·{m[1][0]} = {ans}. Matris yalnızca det ≠ 0 iken tersinirdir; cevap ") + box(_T(lang, yn, yn_tr)) + "."
        return q, s, yn
    else:
        m = [[r.randint(-9, 9) for _ in range(3)] for _ in range(2)]
        i, j = r.randint(0, 2), r.randint(0, 1)
        ans = m[j][i]
        q = _T(lang, f"Let A = {m}. What is the entry in row {i + 1}, column {j + 1} of the transpose of A?", f"A = {m} olsun. A'nın devriğinin (transpose) {i + 1}. satır, {j + 1}. sütun elemanı kaçtır?")
        s = _T(lang, f"(A^T)[{i + 1}][{j + 1}] = A[{j + 1}][{i + 1}] = {ans}. ", f"(A^T)[{i + 1}][{j + 1}] = A[{j + 1}][{i + 1}] = {ans}. ")
    return q, s + box(ans), ans


def gen_calc2(r, lang):
    kind = r.choice(["poly_deriv", "product", "chain", "limit_poly", "limit_factor", "def_int", "critical", "tangent"])
    if kind == "poly_deriv":
        c1, n1, c2, n2, c3 = r.randint(1, 6), r.randint(2, 5), r.randint(-5, 5), r.randint(1, 3), r.randint(-9, 9)
        k = r.randint(-2, 3)
        d1, d2 = c1 * n1, c2 * n2
        ans = d1 * k ** (n1 - 1) + d2 * k ** (n2 - 1)
        f = f"{c1}x^{n1} + {c2}x^{n2} + {c3}".replace("+ -", "- ")
        q = _T(lang, f"If f(x) = {f}, find f'({k}).", f"f(x) = {f} ise f'({k}) kaçtır?")
        s = _T(lang, f"f'(x) = {d1}x^{n1 - 1} + {d2}x^{n2 - 1}. At x = {k}: {d1}·({k})^{n1 - 1} + {d2}·({k})^{n2 - 1} = {ans}. ", f"f'(x) = {d1}x^{n1 - 1} + {d2}x^{n2 - 1}. x = {k} için: {d1}·({k})^{n1 - 1} + {d2}·({k})^{n2 - 1} = {ans}. ")
    elif kind == "product":
        a, b, k = r.randint(-4, 5), r.randint(-4, 5), r.randint(-3, 4)
        ans = (k + a) + (k + b)
        q = _T(lang, f"Let f(x) = (x + {a})(x + {b}). Use the product rule to find f'({k}).".replace("+ -", "- "), f"f(x) = (x + {a})(x + {b}) olsun. Çarpım kuralıyla f'({k}) değerini bulun.".replace("+ -", "- "))
        s = _T(lang, f"f'(x) = (x + {a}) + (x + {b}). At x = {k}: ({k} + {a}) + ({k} + {b}) = {ans}. ", f"f'(x) = (x + {a}) + (x + {b}). x = {k} için: ({k} + {a}) + ({k} + {b}) = {ans}. ").replace("+ -", "- ")
    elif kind == "chain":
        a, b, n, k = r.randint(1, 4), r.randint(-4, 5), r.randint(2, 4), r.randint(-2, 3)
        inner = a * k + b
        ans = n * a * inner ** (n - 1)
        q = _T(lang, f"Find d/dx of ({a}x + {b})^{n} at x = {k}.".replace("+ -", "- "), f"({a}x + {b})^{n} ifadesinin x = {k} noktasındaki türevini bulun.".replace("+ -", "- "))
        s = _T(lang, f"By the chain rule the derivative is {n}·{a}·({a}x + {b})^{n - 1}. At x = {k} the inner value is {inner}, so {n}·{a}·{inner}^{n - 1} = {ans}. ", f"Zincir kuralıyla türev {n}·{a}·({a}x + {b})^{n - 1}. x = {k} için iç değer {inner}; {n}·{a}·{inner}^{n - 1} = {ans}. ").replace("+ -", "- ")
    elif kind == "limit_poly":
        a, b, c, k = r.randint(1, 4), r.randint(-3, 4), r.randint(-5, 6), r.randint(-3, 4)
        ans = a * k * k + b * k + c
        q = _T(lang, f"Evaluate the limit of {a}x^2 + {b}x + {c} as x → {k}.".replace("+ -", "- "), f"x → {k} iken {a}x^2 + {b}x + {c} ifadesinin limitini hesaplayın.".replace("+ -", "- "))
        s = _T(lang, f"A polynomial is continuous, so substitute: {a}·({k})^2 + {b}·({k}) + {c} = {ans}. ", f"Polinom süreklidir, yerine koyun: {a}·({k})^2 + {b}·({k}) + {c} = {ans}. ")
    elif kind == "limit_factor":
        a = r.randint(1, 9) * r.choice([1, -1])
        ans = 2 * a
        q = _T(lang, f"Find the limit of (x^2 - {a * a})/(x - {a}) as x → {a}.".replace("- -", "+ "), f"x → {a} iken (x^2 - {a * a})/(x - {a}) ifadesinin limiti nedir?".replace("- -", "+ "))
        s = _T(lang, f"Factor: x^2 - {a * a} = (x - {a})(x + {a}). After cancelling, the limit is {a} + {a} = {ans}. ", f"Çarpanlara ayırın: x^2 - {a * a} = (x - {a})(x + {a}). Sadeleştirince limit {a} + {a} = {ans}. ").replace("- -", "+ ")
    elif kind == "def_int":
        n, hi = r.randint(1, 4), r.randint(1, 4)
        c = (n + 1) * r.randint(1, 3)
        ans = c * hi ** (n + 1) // (n + 1)
        q = _T(lang, f"Compute the definite integral of {c}x^{n} from x = 0 to x = {hi}.", f"{c}x^{n} fonksiyonunun x = 0'dan x = {hi}'e belirli integralini hesaplayın.")
        s = _T(lang, f"An antiderivative is {c // (n + 1)}x^{n + 1}. At {hi}: {c // (n + 1)}·{hi}^{n + 1} = {ans}; at 0 it is 0. ", f"Ters türev {c // (n + 1)}x^{n + 1}. {hi}'de: {c // (n + 1)}·{hi}^{n + 1} = {ans}; 0'da 0. ")
    elif kind == "critical":
        a, x0, c = r.randint(1, 4), r.randint(-5, 6), r.randint(-9, 9)
        b = -2 * a * x0
        q = _T(lang, f"At what x does f(x) = {a}x^2 + {b}x + {c} have its minimum?".replace("+ -", "- "), f"f(x) = {a}x^2 + {b}x + {c} fonksiyonu hangi x değerinde minimum olur?".replace("+ -", "- "))
        ans = x0
        s = _T(lang, f"f'(x) = {2 * a}x + {b} = 0 gives x = {-b}/{2 * a} = {x0}. Since the leading coefficient is positive this is a minimum. ", f"f'(x) = {2 * a}x + {b} = 0 denkleminden x = {-b}/{2 * a} = {x0}. Başkatsayı pozitif olduğundan bu bir minimumdur. ").replace("+ -", "- ")
    else:
        n, k = r.randint(2, 4), r.randint(1, 3)
        ans = n * k ** (n - 1)
        q = _T(lang, f"What is the slope of the tangent line to y = x^{n} at x = {k}?", f"y = x^{n} eğrisinin x = {k} noktasındaki teğetinin eğimi kaçtır?")
        s = _T(lang, f"The slope is the derivative {n}x^{n - 1} at x = {k}: {n}·{k}^{n - 1} = {ans}. ", f"Eğim, türev {n}x^{n - 1} değerinin x = {k}'daki karşılığıdır: {n}·{k}^{n - 1} = {ans}. ")
    return q, s + box(ans), ans


_LEAN_LEMMAS = [("a + b = b + a", "Nat.add_", "comm", "Nat.add_comm"), ("a * b = b * a", "Nat.mul_", "comm", "Nat.mul_comm"),
                ("(a + b) + c = a + (b + c)", "Nat.add_", "assoc", "Nat.add_assoc"), ("0 + a = a", "Nat.zero_", "add", "Nat.zero_add"),
                ("a + 0 = a", "Nat.add_", "zero", "Nat.add_zero"), ("a * 1 = a", "Nat.mul_", "one", "Nat.mul_one"),
                ("1 * a = a", "Nat.one_", "mul", "Nat.one_mul"), ("a * 0 = 0", "Nat.mul_", "zero", "Nat.mul_zero"),
                ("(a * b) * c = a * (b * c)", "Nat.mul_", "assoc", "Nat.mul_assoc"), ("a ≤ a", "Nat.le_", "refl", "Nat.le_refl")]
_LEAN_TACS = [("`2 + 2 = 4` over Nat, closed by evaluating both sides", "rfl"), ("`x + 1 > x` for a natural number x (linear arithmetic)", "omega"),
              ("`p ∧ q → q ∧ p` for propositions p and q", "tauto"), ("`10 < 20` over Nat, decided by computation", "decide"),
              ("`n + 0 = n` for all n, by rewriting with simp lemmas", "simp"), ("`a + b + c = c + b + a` over Nat", "omega"),
              ("an inductive proof over the natural number n", "induction")]


def gen_lean(r, lang):
    kind = r.choice(["eval", "eval", "lemma", "tactic"])
    if kind == "eval":
        a, b = r.randint(2, 40), r.randint(2, 9)
        lst = [r.randint(1, 9) for _ in range(r.randint(2, 5))]
        opts = [(f"{a} / {b}", str(a // b), "natural-number division rounds down"), (f"{a} % {b}", str(a % b), "% is the remainder"),
                (f"{b} - {a}", str(max(0, b - a)), "natural-number subtraction truncates at zero"), (f"{a} * {b}", str(a * b), "ordinary multiplication"),
                (f"Nat.succ {a}", str(a + 1), "Nat.succ adds one"), (f"{b} ^ 3", str(b ** 3), "^ is exponentiation"),
                (f"Nat.gcd {a * b} {a}", str(math.gcd(a * b, a)), "gcd of a multiple and its factor"),
                (f"{lst}.length", str(len(lst)), "length counts the elements"), (f"{lst}.reverse", str(lst[::-1]).replace("[", "[").replace("]", "]"), "reverse flips the order of the list")]
        expr, ans, why = r.choice(opts)
        q = _T(lang, f"In Lean 4, what does `#eval {expr}` print?", f"Lean 4'te `#eval {expr}` ne yazdırır?")
        s = (f"{why.capitalize()}. The output is " if lang != "tr" else "Çıktı ") + box(ans) + "."
        return q, s, ans
    if kind == "lemma":
        stmt, pre, ans, full = r.choice(_LEAN_LEMMAS)
        q = _T(lang, f"In Lean 4 core, the lemma stating `{stmt}` for natural numbers is named {pre}_____. Fill in the missing word.", f"Lean 4'te doğal sayılar için `{stmt}` önermesini veren teoremin adı {pre}_____ biçimindedir. Boşluğu doldurun.")
        s = _T(lang, f"The standard name is {full}, so the missing word is ", f"Standart ad {full}; eksik sözcük ") + box(ans) + "."
        return q, s, ans
    desc, ans = r.choice(_LEAN_TACS)
    q = _T(lang, f"In Lean 4, which single tactic is the natural choice to prove {desc}? Answer with the tactic name.", f"Lean 4'te {desc} ifadesini kanıtlamak için en uygun tek taktik hangisidir? Taktik adını yazın.")
    s = _T(lang, f"The natural tactic here is ", "Uygun taktik ") + box(ans) + "."
    return q, s, ans


def gen_proof_blank(r, lang):
    m = r.choice(["m", "k", "t"])
    items = [
        ("Claim: the sum of two even numbers is even. Proof: write a = 2m and b = 2k. Then a + b = 2(__). Fill in the blank.", "Teorem: iki çift sayının toplamı çifttir. İspat: a = 2m, b = 2k yazalım. O zaman a + b = 2(__). Boşluğu doldurun.", "m+k", "a + b = 2m + 2k = 2(m + k)."),
        ("Claim: the sum of two odd numbers is even. Proof: a = 2m+1, b = 2k+1, so a + b = 2(m + k + __). Fill in the number.", "Teorem: iki tek sayının toplamı çifttir. İspat: a = 2m+1, b = 2k+1 için a + b = 2(m + k + __). Sayıyı yazın.", "1", "a + b = 2m + 2k + 2 = 2(m + k + 1)."),
        ("Claim: the product of an even number and any integer is even. Proof: a = 2m, so a·n = 2(__). Fill in the blank.", "Teorem: çift sayı ile herhangi bir tam sayının çarpımı çifttir. İspat: a = 2m için a·n = 2(__). Boşluğu doldurun.", "mn", "a·n = 2m·n = 2(mn)."),
        ("Claim: if a and b are multiples of 3, so is a + b. Proof: a = 3m, b = 3k, so a + b = 3(__). Fill in the blank.", "Teorem: a ve b, 3'ün katıysa a + b de 3'ün katıdır. İspat: a = 3m, b = 3k için a + b = 3(__). Boşluğu doldurun.", "m+k", "a + b = 3m + 3k = 3(m + k)."),
        ("Induction for 1 + 2 + ... + n = n(n+1)/2: in the inductive step we add (__) to the sum up to k to reach the sum up to k+1. Fill in the term.", "1 + 2 + ... + n = n(n+1)/2 için tümevarım adımında k'ya kadar olan toplama, k+1'e ulaşmak için (__) eklenir. Terimi yazın.", "k+1", "The next term after k is k + 1."),
        ("Induction: to prove P(n) for all n ≥ 1, the base case checks n = __. Fill in the number.", "Tümevarımla n ≥ 1 için P(n) kanıtlanırken taban durumunda n = __ kontrol edilir. Sayıyı yazın.", "1", "The smallest allowed value is n = 1."),
        ("Claim: sqrt(3) is irrational. Proof by contradiction: if sqrt(3) = p/q in lowest terms, then p^2 = 3q^2, so p^2 is divisible by __. Fill in the number.", "Teorem: sqrt(3) irrasyoneldir. Çelişkiyle ispat: sqrt(3) = p/q sadeleşmişse p^2 = 3q^2 olur, yani p^2, __ ile bölünür. Sayıyı yazın.", "3", "p^2 = 3q^2 is a multiple of 3."),
        ("Claim: sqrt(2) is irrational. Proof: if sqrt(2) = p/q in lowest terms then p^2 = 2q^2, so p^2 is __ (even/odd). Fill in one word.", "Teorem: sqrt(2) irrasyoneldir. İspat: sqrt(2) = p/q sadeleşmişse p^2 = 2q^2, yani p^2 __ (çift/tek). Tek sözcük yazın.", "even", "2q^2 is divisible by 2."),
        ("Claim: there are infinitely many primes. Proof: assume only p1, ..., pk. Let N = p1·p2·...·pk + __. Then N has a prime factor not in the list. Fill in the number.", "Teorem: sonsuz sayıda asal vardır. İspat: yalnızca p1, ..., pk olsun. N = p1·p2·...·pk + __ alalım; N'nin listede olmayan bir asal çarpanı vardır. Sayıyı yazın.", "1", "N leaves remainder 1 when divided by any listed prime."),
        ("Pigeonhole principle: if n + 1 objects are placed into n boxes, some box contains at least __ objects. Fill in the number.", "Güvercin yuvası ilkesi: n + 1 nesne n kutuya konursa bir kutuda en az __ nesne vardır. Sayıyı yazın.", "2", "Otherwise every box holds at most 1, for a total of at most n."),
        ("Claim: the square of an odd number is odd. Proof: (2m+1)^2 = 4m^2 + 4m + 1 = 2(__) + 1. Fill in the expression.", "Teorem: tek sayının karesi tektir. İspat: (2m+1)^2 = 4m^2 + 4m + 1 = 2(__) + 1. İfadeyi yazın.", "2m^2+2m", "4m^2 + 4m = 2(2m^2 + 2m)."),
        ("Proof technique: to prove 'if P then Q' by contraposition, we assume not __ and show not __. Which letter fills both blanks?", "İspat tekniği: 'P ise Q' önermesini karşıt-evrik ile kanıtlamak için değil __ varsayıp değil __ gösteririz. İki boşluğa da hangi harfler gelir? (önce, sonra)", "Q,P", "The contrapositive of P → Q is ¬Q → ¬P."),
    ]
    en, tr, ans, why = r.choice(items)
    q = _T(lang, en, tr)
    s = why + " " + _T(lang, "The blank is ", "Boşluk: ") + box(ans) + "."
    if ans == "Q,P":
        s = why + " " + _T(lang, "First blank Q, second blank P: ", "Birinci boşluk Q, ikinci boşluk P: ") + box(ans) + "."
    return q, s, ans


GENERATORS: Dict[str, Callable] = {
    "add": gen_add, "sub": gen_sub, "mul": gen_mul, "div": gen_div, "linear": gen_linear, "linear2": gen_linear2,
    "quadratic": gen_quadratic, "gcd": gen_gcd, "lcm": gen_lcm, "prime": gen_prime, "modpow": gen_modpow, "factor": gen_factor,
    "comb": gen_comb, "perm": gen_perm, "subsets": gen_binom_sum, "deriv": gen_deriv, "integral": gen_integral,
    "det": gen_det, "dot": gen_dot, "logic": gen_logic, "graph": gen_graph, "bfs": gen_bfs, "sort": gen_sort_trace,
    "binsearch": gen_binsearch, "recurrence": gen_recurrence, "fib": gen_fib, "coin": gen_coin_dp, "cfact": gen_complexity_fact,
    # v3
    "loop_cx": gen_loop_cx, "master": gen_master, "ds_cost": gen_ds_cost, "kth": gen_kth, "cmp_count": gen_cmp_count,
    "stack_queue": gen_stack_queue, "dfs_order": gen_dfs_order, "linalg": gen_linalg, "calc2": gen_calc2, "lean": gen_lean,
    "proof_blank": gen_proof_blank,
}
# Aritmetik daha sık: hesaplama temeli
WEIGHTS = {"add": 6, "sub": 6, "mul": 6, "div": 3, "linear": 4, "linear2": 2, "quadratic": 2, "gcd": 3, "lcm": 1, "prime": 2,
           "modpow": 2, "factor": 2, "comb": 3, "perm": 1, "subsets": 1, "deriv": 3, "integral": 2, "det": 2, "dot": 2,
           "logic": 3, "graph": 3, "bfs": 2, "sort": 1, "binsearch": 1, "recurrence": 2, "fib": 1, "coin": 1, "cfact": 1,
           "loop_cx": 3, "master": 3, "ds_cost": 4, "kth": 3, "cmp_count": 3, "stack_queue": 2, "dfs_order": 2, "linalg": 5,
           "calc2": 5, "lean": 4, "proof_blank": 3}


def generate(n: int, seed: int = 777, tr_fraction: float = 0.30, exclude_questions: Optional[Set[str]] = None) -> Iterator[Dict]:
    """n örnek üretir. exclude_questions: normalize edilmiş soru metinleri (ALTUN eval) — bunlar atlanır."""
    r = random.Random(seed)
    names, weights = list(WEIGHTS), list(WEIGHTS.values())
    ex = {_norm(x) for x in (exclude_questions or set())}
    made = 0
    while made < n:
        name = r.choices(names, weights)[0]
        lang = "tr" if r.random() < tr_fraction else "en"
        q, s, ans = GENERATORS[name](r, lang)
        if _norm(q) in ex:
            continue
        made += 1
        yield {"kind": name, "lang": lang, "question": q, "solution": s, "answer": ans,
               "text": f"Problem: {q}\nSolution: {s}"}


def generate_documents(n_examples: int, per_doc: int = 24, seed: int = 777, exclude_questions: Optional[Set[str]] = None) -> Iterator[str]:
    buf: List[str] = []
    for ex in generate(n_examples, seed, exclude_questions=exclude_questions):
        buf.append(ex["text"])
        if len(buf) == per_doc:
            yield "\n\n".join(buf)
            buf = []
    if buf:
        yield "\n\n".join(buf)
