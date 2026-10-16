"""Эталонные решения слоёв «Матрёшки v2».

Три слоя построены на задачах IMO 2026 (Шанхай, июль 2026) — они моложе обучающих данных моделей 2026 года:
  №6 — последовательность «наименьшее следующее число, не взаимно простое со всеми предыдущими»;
  №3 — Лю Бан и Сян Юй делят палку;
  №4 — Шань Юй и Мулан режут треугольник.
Остальные — модифицированная классика: бильярд Гальперина с неупругими ударами (при малом коэффициенте
восстановления ударов бесконечно много — «неупругий коллапс») и тасовка Монжа.

Все функции работают для любого входа: это нужно для оценки «по слоям», где вход слоя — ответ модели.
"""
from __future__ import annotations

import bisect
import math
from fractions import Fraction
from functools import lru_cache


# ---------------------------------------------------------------------------
# простые множители
# ---------------------------------------------------------------------------

def prime_set(n):
    s, d = set(), 2
    while d * d <= n:
        if n % d == 0:
            s.add(d)
            while n % d == 0:
                n //= d
        d += 1 if d == 2 else 2
    if n > 1:
        s.add(n)
    return frozenset(s)


def factorize(n):
    f, d = {}, 2
    while d * d <= n:
        while n % d == 0:
            f[d] = f.get(d, 0) + 1
            n //= d
        d += 1 if d == 2 else 2
    if n > 1:
        f[n] = f.get(n, 0) + 1
    return f


def num_divisors(n):
    r = 1
    for e in factorize(n).values():
        r *= e + 1
    return r


# ---------------------------------------------------------------------------
# слой 5 — IMO 2026/6: a_{n+1} = наименьшее число > a_n, имеющее общий делитель > 1 с каждым из a_1..a_n
# ---------------------------------------------------------------------------

def p6_prefix(a1, count):
    """Первые count членов и семейство минимальных (по включению) множеств простых делителей членов."""
    mins = [prime_set(a1)]
    out = [a1]
    x = a1
    while len(out) < count:
        x += 1
        ps = prime_set(x)
        if all(ps & S for S in mins):
            out.append(x)
            if not any(S <= ps for S in mins):
                mins = [S for S in mins if not ps <= S] + [ps]
    return out, mins


@lru_cache(maxsize=16)
def p6_structure(a1, warm=6000):
    """Хвост последовательности периодичен: члены — все числа, чьи простые делители из семейства F
    пересекают каждое множество F. Возвращает (префикс, F, L, отсортированные вычеты mod L)."""
    pre, mins = p6_prefix(a1, warm)
    primes = sorted(set().union(*mins))
    L = 1
    for p in primes:
        L *= p
    if L > 10 ** 7:
        raise ValueError("слишком большой период")

    def good(x):
        ps = {p for p in primes if x % p == 0}
        return all(ps & S for S in mins)
    R = [r for r in range(L) if good(r)]
    # проверяем: начиная с некоторого места префикс — подряд идущие «хорошие» числа
    pos = {x: i for i, x in enumerate(pre)}
    i0 = None
    for i in range(len(pre) - 1, 0, -1):
        x, y = pre[i - 1], pre[i]
        if not good(x) or any(good(z) for z in range(x + 1, y)):
            i0 = i
            break
    i0 = 0 if i0 is None else i0
    if i0 > warm // 2:
        raise ValueError("последовательность не стабилизировалась")
    del pos
    return tuple(pre), tuple(map(frozenset, mins)), L, tuple(R), i0


def p6_term(a1, N):
    """a_N (нумерация с 1)."""
    pre, mins, L, R, i0 = p6_structure(a1)
    if N <= len(pre):
        return pre[N - 1]
    T = len(R)
    x0 = pre[i0]
    k0 = (x0 // L) * T + bisect.bisect_left(R, x0 % L)  # номер x0 среди «хороших» чисел (с нуля)
    k = k0 + (N - 1 - i0)
    q, r = divmod(k, T)
    return q * L + R[r]


def p6_count(a1, X):
    """Сколько членов последовательности не превосходят X."""
    pre, mins, L, R, i0 = p6_structure(a1)
    T = len(R)

    def upto(x):  # «хорошие» числа в [0, x]
        q, r = divmod(x, L)
        return q * T + bisect.bisect_right(R, r)
    # последовательность периодична с самого начала (a_{n+T} = a_n + L для всех n ≥ 1, IMO 2026/6),
    # поэтому её члены — ровно «хорошие» числа, начиная с a_1
    return upto(X) - upto(a1 - 1)


L5_A1 = 1925
L5_X = 10 ** 7


def solve5():
    return p6_count(L5_A1, L5_X)


# ---------------------------------------------------------------------------
# слой 4 — бильярд Гальперина: стена упругая, бруски — с коэффициентом восстановления e
# ---------------------------------------------------------------------------

E4 = Fraction(4, 5)


def param4(a5):
    return 100 + int(a5) % 400


def collapse4(A, e=E4):
    """Бесконечно много ударов, когда собственные числа отображения «удар брусков + удар о стену»
    вещественны: след² ≥ 4·det, след = (1+e)(A−1)/(A+1), det = e."""
    A = Fraction(A)
    return ((1 + e) * (A - 1) / (A + 1)) ** 2 >= 4 * e


def solve4(A, e=E4, cap=200000):
    """Число ударов (0 — если их бесконечно много)."""
    if collapse4(A, e):
        return 0
    m, M = Fraction(1), Fraction(A)
    v1, v2 = Fraction(0), Fraction(-1)
    n = 0
    while n < cap:
        if v2 < v1:
            u = (m * v1 + M * v2) / (m + M)
            v1, v2 = u - e * (v1 - u), u - e * (v2 - u)
            n += 1
        elif v1 < 0:
            v1 = -v1
            n += 1
        else:
            return n
    raise RuntimeError("слишком много ударов")


def sim4_float(A, e=0.8, cap=10 ** 6):
    """Проверка без точных дробей (нормируем скорость — процесс линейный)."""
    v1, v2 = 0.0, -1.0
    n = 0
    while n < cap:
        if v2 < v1:
            u = (v1 + A * v2) / (1 + A)
            v1, v2 = u - e * (v1 - u), u - e * (v2 - u)
            n += 1
        elif v1 < 0:
            v1 = -v1
            n += 1
        else:
            return n
        s = math.hypot(v1, v2)
        v1, v2 = v1 / s, v2 / s
    return None


# ---------------------------------------------------------------------------
# слой 3 — IMO 2026/3: палка, не более n отметок у каждого; c_n = 2^n / (2^(n+1) − 1)
# ---------------------------------------------------------------------------

def param3(a4):
    return 5 + int(a4) % 30


def solve3(n):
    c = Fraction(2 ** n, 2 ** (n + 1) - 1)
    return c.numerator + c.denominator


# ---------------------------------------------------------------------------
# слой 2 — тасовка Монжа: порядок = наименьшее k, при котором 2^k ≡ ±1 (mod 2D − 1)
# ---------------------------------------------------------------------------

def param2(a3):
    return 2 * (500000 + int(a3) % 500000)


def monge_order(D):
    m = 2 * D - 1
    if m == 1:
        return 1
    k, x = 1, 2 % m
    while x != 1 and x != m - 1:
        x = x * 2 % m
        k += 1
    return k


def monge_order_brute(D):
    """Порядок перестановки из программы (разложение на циклы)."""
    from collections import deque
    d = deque()
    for i in range(D):
        if i % 2 == 0:
            d.appendleft(i)
        else:
            d.append(i)
    new = list(d)
    pos = [0] * D
    for p, c in enumerate(new):
        pos[c] = p
    seen = [False] * D
    L = 1
    for s in range(D):
        if not seen[s]:
            n, x = 0, s
            while not seen[x]:
                seen[x] = True
                x = pos[x]
                n += 1
            L = L * n // math.gcd(L, n)
    return L


def solve2(a3):
    return monge_order(param2(a3))


# ---------------------------------------------------------------------------
# слой 1 — IMO 2026/4: Мулан гарантирует победу ровно при θ = 180°/n, n ≥ 2
# ---------------------------------------------------------------------------

def param1(a2):
    return 2 + int(a2) % 999999


def solve1(a2):
    m = param1(a2)
    return num_divisors(m) - 1  # θ_k = k·180°/m = 180°/n  ⇔  k | m, k < m
