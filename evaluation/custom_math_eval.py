"""ALTUN özel matematik değerlendirme seti (13 kategori, 3 zorluk seviyesi).
Cevaplar kodla hesaplanır (uydurma yok). Seed sabit → her çalıştırmada aynı set.
Çalıştırma (model gerekir): evaluation/evaluate.py içinden.
"""
from __future__ import annotations

import math
import random
from collections import deque
from typing import Dict, List

CATEGORIES = ["Arithmetic", "Algebra", "Number Theory", "Combinatorics", "Calculus", "Linear Algebra",
              "Mathematical Logic", "Proof Completion", "Graph Theory", "Algorithms", "Algorithm Analysis",
              "Computational Complexity", "Formal Mathematics"]


def _q(cat, diff, q, a, typ="number", lang="en"):
    return {"category": cat, "difficulty": diff, "question": q, "answer": str(a), "answer_type": typ, "lang": lang}


def _is_prime(n):
    return n > 1 and all(n % p for p in range(2, int(n ** 0.5) + 1))


def _bfs(adj, s, t):
    dist, dq = {s: 0}, deque([s])
    while dq:
        u = dq.popleft()
        for v in adj[u]:
            if v not in dist:
                dist[v] = dist[u] + 1
                dq.append(v)
    return dist.get(t, -1)


def build_custom_eval(seed: int = 12345) -> List[Dict]:
    r = random.Random(seed)
    out: List[Dict] = []
    for d, (lo, hi) in enumerate([(2, 20), (20, 200), (200, 5000)], 1):
        for _ in range(3):
            a, b = r.randint(lo, hi), r.randint(lo, hi)
            out.append(_q("Arithmetic", d, f"Compute {a} + {b}.", a + b))
            out.append(_q("Arithmetic", d, f"Compute {a} * {b}.", a * b) if d < 3 else _q("Arithmetic", d, f"Compute {a} - {b}.", a - b))
        a, b = r.randint(lo, hi), r.randint(2, 9)
        out.append(_q("Arithmetic", d, f"{a} sayısının {b} katı kaçtır?", a * b, lang="tr"))
        # Algebra
        x, p, q_ = r.randint(1, 5 * d + 5), r.randint(2, 9), r.randint(1, 30)
        out.append(_q("Algebra", d, f"Solve for x: {p}x + {q_} = {p * x + q_}.", x))
        r1, r2 = r.randint(1, 4 + d), r.randint(1, 4 + d)
        out.append(_q("Algebra", d, f"What is the sum of the roots of x^2 - {r1 + r2}x + {r1 * r2} = 0?", r1 + r2))
        out.append(_q("Algebra", d, f"x + {q_} = {x + q_} denkleminde x kaçtır?", x, lang="tr"))
        # Number theory
        a, b = r.randint(2, 12 * d) * r.randint(2, 6), r.randint(2, 12 * d) * r.randint(2, 6)
        out.append(_q("Number Theory", d, f"What is gcd({a}, {b})?", math.gcd(a, b)))
        n = r.randint(10 ** d, 10 ** (d + 1))
        out.append(_q("Number Theory", d, f"Is {n} prime? Answer yes or no.", "yes" if _is_prime(n) else "no", "yesno"))
        m, e, mod = r.randint(2, 9), r.randint(2, 6 + d), r.randint(5, 13)
        out.append(_q("Number Theory", d, f"What is {m}^{e} mod {mod}?", pow(m, e, mod)))
        # Combinatorics
        n, k = r.randint(4 + d, 6 + 2 * d), r.randint(1, 3)
        out.append(_q("Combinatorics", d, f"What is C({n}, {k}) (n choose k)?", math.comb(n, k)))
        out.append(_q("Combinatorics", d, f"What is {n}! / {n - 1}! ?", n))
        out.append(_q("Combinatorics", d, f"In how many ways can {k + 2} distinct books be arranged on a shelf?", math.factorial(k + 2)))
        # Calculus
        n, a = r.randint(2, 3 + d), r.randint(1, 4)
        out.append(_q("Calculus", d, f"If f(x) = x^{n}, what is f'({a})?", n * a ** (n - 1)))
        c = r.randint(1, 5)
        out.append(_q("Calculus", d, f"Evaluate the integral of {2 * c}x dx from 0 to {a + d}.", c * (a + d) ** 2))
        out.append(_q("Calculus", d, f"What is the limit of ({a}x + {c}) as x approaches {d}?", a * d + c))
        # Linear algebra
        M = [r.randint(-3 * d, 3 * d) for _ in range(4)]
        out.append(_q("Linear Algebra", d, f"What is the determinant of the matrix [[{M[0]}, {M[1]}], [{M[2]}, {M[3]}]]?", M[0] * M[3] - M[1] * M[2]))
        u, v = [r.randint(-4, 9) for _ in range(3)], [r.randint(-4, 9) for _ in range(3)]
        out.append(_q("Linear Algebra", d, f"Compute the dot product of {tuple(u)} and {tuple(v)}.", sum(i * j for i, j in zip(u, v))))
        out.append(_q("Linear Algebra", d, f"What is the trace of [[{M[0]}, {M[1]}], [{M[2]}, {M[3]}]]?", M[0] + M[3]))
        # Logic
        P, Q = r.random() < .5, r.random() < .5
        s = lambda b_: "True" if b_ else "False"
        out.append(_q("Mathematical Logic", d, f"Evaluate: {s(P)} AND {s(Q)}. Answer true or false.", "yes" if P and Q else "no", "yesno"))
        out.append(_q("Mathematical Logic", d, f"Evaluate: ({s(P)} IMPLIES {s(Q)}). Answer true or false.", "yes" if (not P) or Q else "no", "yesno"))
        out.append(_q("Mathematical Logic", d, f"Evaluate: NOT({s(P)} OR {s(Q)}). Answer true or false.", "yes" if not (P or Q) else "no", "yesno"))
        # Graph theory
        n = r.randint(4 + d, 7 + d)
        out.append(_q("Graph Theory", d, f"How many edges does the complete graph K_{n} have?", n * (n - 1) // 2))
        out.append(_q("Graph Theory", d, f"A tree has {n} vertices. How many edges does it have?", n - 1))
        adj = {i: [] for i in range(6)}
        edges = [(0, 1), (1, 2), (2, 3), (3, 4), (4, 5)] + ([(0, 3)] if d >= 2 else []) + ([(1, 4)] if d >= 3 else [])
        for u_, v_ in edges:
            adj[u_].append(v_); adj[v_].append(u_)
        out.append(_q("Graph Theory", d, f"Undirected graph with edges {edges}. What is the shortest path length (in edges) from 0 to 5?", _bfs(adj, 0, 5)))
        # Algorithms
        arr = [r.randint(1, 50) for _ in range(4 + d)]
        out.append(_q("Algorithms", d, f"What is the second smallest element after sorting {arr} in ascending order?", sorted(arr)[1]))
        n = 5 + 2 * d
        fib = [0, 1]
        for _ in range(n):
            fib.append(fib[-1] + fib[-2])
        out.append(_q("Algorithms", d, f"With F(0)=0 and F(1)=1, what is F({n})?", fib[n]))
        out.append(_q("Algorithms", d, f"How many comparisons does a linear scan need, in the worst case, to find the maximum of {len(arr)} elements?", len(arr) - 1))
    # --- el yapımı sabit sorular (algoritma analizi / karmaşıklık / ispat / formal) ---
    fixed = [
        ("Algorithm Analysis", 1, "What is the worst-case time complexity of binary search on n sorted elements? Answer in big-O.", "O(log n)", "text"),
        ("Algorithm Analysis", 1, "What is the worst-case time complexity of merge sort? Answer in big-O.", "O(n log n)", "text"),
        ("Algorithm Analysis", 2, "What is the worst-case time complexity of quicksort? Answer in big-O.", "O(n^2)", "text"),
        ("Algorithm Analysis", 2, "What is the time complexity of building a binary heap from n elements? Answer in big-O.", "O(n)", "text"),
        ("Algorithm Analysis", 2, "Solve T(n) = 2T(n/2) + n. Answer in big-O.", "O(n log n)", "text"),
        ("Algorithm Analysis", 3, "Solve T(n) = T(n/2) + 1. Answer in big-O.", "O(log n)", "text"),
        ("Algorithm Analysis", 3, "What is the time complexity of BFS on a graph with V vertices and E edges using adjacency lists? Answer in big-O.", "O(V+E)", "text"),
        ("Algorithm Analysis", 1, "İkili aramanın en kötü durum zaman karmaşıklığı nedir? Big-O ile yazınız.", "O(log n)", "text"),
        ("Computational Complexity", 1, "Is SAT in NP? Answer yes or no.", "yes", "yesno"),
        ("Computational Complexity", 1, "Is every problem in P also in NP? Answer yes or no.", "yes", "yesno"),
        ("Computational Complexity", 2, "Is the Halting Problem decidable? Answer yes or no.", "no", "yesno"),
        ("Computational Complexity", 2, "Is 2-SAT solvable in polynomial time? Answer yes or no.", "yes", "yesno"),
        ("Computational Complexity", 3, "Has P vs NP been proven (as of today)? Answer yes or no.", "no", "yesno"),
        ("Computational Complexity", 3, "Is 3-SAT NP-complete? Answer yes or no.", "yes", "yesno"),
        ("Computational Complexity", 2, "P ≠ NP sorusu çözülmüş müdür? Evet ya da hayır.", "no", "yesno"),
        ("Proof Completion", 1, "Claim: the sum of two even numbers is even. Proof: let a = 2m and b = 2k. Then a + b = 2(__). Fill in the blank.", "m+k", "text"),
        ("Proof Completion", 2, "Claim: sqrt(2) is irrational. Proof by contradiction: assume sqrt(2) = p/q in lowest terms. Then p^2 = 2q^2, so p^2 is __. Fill in one word (even/odd).", "even", "text"),
        ("Proof Completion", 2, "Induction proof of 1+2+...+n = n(n+1)/2: the base case is n = __. Fill in the number.", "1", "text"),
        ("Proof Completion", 3, "There are infinitely many primes. Assume finitely many p1..pk and consider N = p1*p2*...*pk + __. Fill in the number.", "1", "text"),
        ("Proof Completion", 1, "Teorem: iki çift sayının toplamı çifttir. İspat: a = 2m, b = 2k ise a + b = 2(__). Boşluğu doldurunuz.", "m+k", "text", "tr"),
        ("Formal Mathematics", 1, "In Lean 4, what does `#eval 2 + 3` output?", "5", "number"),
        ("Formal Mathematics", 1, "In Lean 4, what does `#eval Nat.succ 4` output?", "5", "number"),
        ("Formal Mathematics", 2, "In Lean 4, what does `#eval 10 / 3` output for natural numbers?", "3", "number"),
        ("Formal Mathematics", 2, "In Lean 4, which tactic proves goals like `a + b = b + a` for Nat by calling the omega decision procedure? Answer with the tactic name.", "omega", "text"),
        ("Formal Mathematics", 3, "In Lean 4 Mathlib, the name of the lemma stating a + b = b + a for natural numbers is Nat.add_____. Fill in the missing word.", "comm", "text"),
        ("Formal Mathematics", 3, "In Lean 4, `theorem foo : 2 + 2 = 4 := by` followed by which tactic closes the goal by computation? Answer with one tactic name (rfl/simp/decide).", "rfl", "text"),
    ]
    for item in fixed:
        out.append(_q(item[0], item[1], item[2], item[3], item[4], item[5] if len(item) > 5 else "en"))
    for i, q in enumerate(out):
        q["id"] = f"altun-{i:04d}"
    return out
