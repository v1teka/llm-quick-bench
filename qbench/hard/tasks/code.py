"""Набор «Программирование»: четыре задачи олимпиадного уровня, решение — программа на Python 3,
которая запускается на скрытых тестах с ограничением времени (как на соревнованиях).

Задачи авторские: условие нельзя найти и вспомнить. Для каждой есть эталонное решение и
переборное решение; эталон сверяется с перебором на малых тестах (tests/test_pro.py).
Скрытые тесты генерируются детерминированно из seed: малые (сверка с перебором), средние и
максимальные (проверка асимптотики). Балл задачи — доля пройденных тестов.
"""
from __future__ import annotations

import itertools
import os
import random
import re
import subprocess
import sys
import tempfile
from fractions import Fraction
from math import gcd

from .common import Task

MOD = 998244353

# ---------------------------------------------------------------------------
# Условия
# ---------------------------------------------------------------------------

STATEMENTS = [
    {
        "name": "Кольцо ламп",
        "tl": 4.0,
        "text": """По кругу стоят n ламп, пронумерованных от 0 до n−1; лампа i горит, если s_i = 1. За одну операцию
можно выбрать любое i и переключить k подряд идущих ламп i, i+1, …, i+k−1 (номера по модулю n): горящие гаснут,
погашенные загораются. Найдите наименьшее число операций, после которых все лампы погаснут, или выведите −1,
если это невозможно.

**Ввод.** В первой строке T — число наборов (1 ≤ T ≤ 10⁴). Каждый набор — строка «n k» (1 ≤ k ≤ n ≤ 2·10⁵)
и строка s из n символов 0/1. Сумма n по всем наборам не превосходит 3·10⁵.

**Вывод.** Для каждого набора — одно число в отдельной строке.

**Пример.**
```
4
3 2
110
4 2
1000
5 5
11111
6 4
101101
```
Ответ:
```
1
-1
1
2
```""",
    },
    {
        "name": "Повторы без наложения",
        "tl": 6.0,
        "text": """Дана строка s из строчных латинских букв. Посчитайте, сколько существует **различных** непустых строк t,
которые входят в s как подстрока хотя бы два раза **без наложения**: найдутся вхождения, начинающиеся с позиций
i и j, где j ≥ i + |t|.

**Ввод.** Одна строка s, 1 ≤ |s| ≤ 10⁵.

**Вывод.** Одно число.

**Пример.** Для `aaaa` ответ 2 (`a` и `aa`), для `abab` — 3 (`a`, `b`, `ab`), для `abcabcab` — 9.""",
    },
    {
        "name": "Квадраты НОД",
        "tl": 8.0,
        "text": f"""Вычислите S(n) = Σ_{{i=1..n}} Σ_{{j=1..n}} НОД(i, j)² по модулю {MOD}.

**Ввод.** Одно число n, 1 ≤ n ≤ 2·10⁹.

**Вывод.** S(n) mod {MOD}.

**Пример.** S(1) = 1, S(2) = 7, S(10) = 655.""",
    },
    {
        "name": "Кактус под дождём",
        "tl": 6.0,
        "text": f"""Дан неориентированный граф без петель и кратных рёбер: n вершин, m рёбер, каждое ребро лежит не более
чем на одном простом цикле (рёберный кактус; граф может быть несвязным). Начинается дождь, и вершина i
независимо от остальных смывается с вероятностью p_i/100 вместе со всеми своими рёбрами. Найдите математическое
ожидание числа компонент связности оставшегося графа (граф без вершин имеет 0 компонент).

Ответ — рациональное число P/Q; выведите P·Q⁻¹ по модулю {MOD}.

**Ввод.** Первая строка — n и m (1 ≤ n ≤ 2·10⁵, 0 ≤ m ≤ 3·10⁵). Вторая — p_1 … p_n (0 ≤ p_i ≤ 100).
Далее m строк по два номера вершин u v (1 ≤ u, v ≤ n).

**Вывод.** Одно число.

**Пример.**
```
3 3
50 50 50
1 2
2 3
3 1
```
Ответ: `124780545` (это 7/8: E = 3·½ − 3·¼ + ⅛).""",
    },
]

# ---------------------------------------------------------------------------
# Эталонные решения (запускаются так же, как решения модели)
# ---------------------------------------------------------------------------

REF = [
    r'''
import sys
from math import gcd
def solve(n, k, s):
    g = gcd(n, k); L = n // g
    b = [s[j] ^ s[j - 1] for j in range(n)]          # x_j ^ x_{j-k} = b_j
    x = [0] * n
    costs = []
    for r in range(g):
        j = r; x[j] = 0; c = 0
        for _ in range(L - 1):
            nj = (j + k) % n
            x[nj] = x[j] ^ b[nj]; c += x[nj]; j = nj
        if x[j] ^ x[(j + k) % n] != b[(j + k) % n]:
            return -1
        costs.append(c)
    w = 0
    for i in range(n - k + 1, n + 1):                 # окно, покрывающее лампу 0
        w ^= x[i % n]
    need = w ^ s[0]                                   # нужно ли изменить чётность окна
    base = sum(min(c, L - c) for c in costs)
    if (k // g) % 2 == 0:
        return -1 if need else base
    par = sum(1 for c in costs if L - c < c) % 2      # сколько циклов перевёрнуто при жадном выборе
    if par == need:
        return base
    return base + min(abs(L - 2 * c) for c in costs)
def main():
    data = sys.stdin.read().split()
    t = int(data[0]); p = 1; out = []
    for _ in range(t):
        n, k = int(data[p]), int(data[p + 1]); s = [int(ch) for ch in data[p + 2]]; p += 3
        out.append(str(solve(n, k, s)))
    print("\n".join(out))
main()
''',
    r'''
import sys
def main():
    s = sys.stdin.readline().strip()
    n = len(s)
    maxn = 2 * n + 2
    nxt = [None] * maxn
    link = [-1] * maxn
    ln = [0] * maxn
    fp = [0] * maxn
    lp = [-1] * maxn
    nxt[0] = {}
    sz = 1; last = 0
    for i, ch in enumerate(s):
        cur = sz; sz += 1
        ln[cur] = ln[last] + 1; fp[cur] = i; lp[cur] = i; nxt[cur] = {}
        p = last
        while p != -1 and ch not in nxt[p]:
            nxt[p][ch] = cur; p = link[p]
        if p == -1:
            link[cur] = 0
        else:
            q = nxt[p][ch]
            if ln[p] + 1 == ln[q]:
                link[cur] = q
            else:
                cl = sz; sz += 1
                ln[cl] = ln[p] + 1; nxt[cl] = dict(nxt[q]); link[cl] = link[q]; fp[cl] = fp[q]
                while p != -1 and nxt[p].get(ch) == q:
                    nxt[p][ch] = cl; p = link[p]
                link[q] = cl; link[cur] = cl
        last = cur
    order = sorted(range(1, sz), key=lambda v: -ln[v])
    for v in order:
        u = link[v]
        if u > 0 and lp[v] > lp[u]:
            lp[u] = lp[v]
    ans = 0
    for v in range(1, sz):
        m = min(ln[v], lp[v] - fp[v])
        if m > ln[link[v]]:
            ans += m - ln[link[v]]
    print(ans)
main()
''',
    r'''
import sys
def main():
    n = int(sys.stdin.readline())
    M = 998244353
    N = min(n, max(10, int(n ** (2 / 3)) + 1))
    phi = list(range(N + 1))
    for i in range(2, N + 1):
        if phi[i] == i:
            for j in range(i, N + 1, i):
                phi[j] -= phi[j] // i
    pre = [0] * (N + 1)
    acc = 0
    for i in range(1, N + 1):
        acc += phi[i]; pre[i] = acc
    memo = {}
    def Phi(m):
        if m <= N: return pre[m]
        if m in memo: return memo[m]
        r = m * (m + 1) // 2; d = 2
        while d <= m:
            q = m // d; e = m // q
            r -= (e - d + 1) * Phi(q); d = e + 1
        memo[m] = r
        return r
    def sq(x): return x * (x + 1) * (2 * x + 1) // 6
    ans = 0; d = 1
    while d <= n:
        q = n // d; e = n // q
        ans += (sq(e) - sq(d - 1)) % M * ((2 * Phi(q) - 1) % M)
        d = e + 1
    print(ans % M)
main()
''',
    r'''
import sys
def main():
    data = sys.stdin.buffer.read().split()
    M = 998244353
    n, m = int(data[0]), int(data[1])
    inv100 = pow(100, M - 2, M)
    q = [0] + [(100 - int(data[2 + i])) * inv100 % M for i in range(n)]
    adj = [[] for _ in range(n + 1)]
    p = 2 + n
    eu = []
    for e in range(m):
        u, v = int(data[p]), int(data[p + 1]); p += 2
        adj[u].append((v, e)); adj[v].append((u, e)); eu.append((u, v))
    ans = sum(q) % M
    for u, v in eu:
        ans -= q[u] * q[v]
    par = [0] * (n + 1); pe = [-1] * (n + 1); dep = [-1] * (n + 1)
    for r in range(1, n + 1):
        if dep[r] != -1: continue
        dep[r] = 0; st = [(r, 0)]
        while st:
            u, i = st[-1]
            if i < len(adj[u]):
                st[-1] = (u, i + 1)
                v, e = adj[u][i]
                if e == pe[u]: continue
                if dep[v] == -1:
                    dep[v] = dep[u] + 1; par[v] = u; pe[v] = e; st.append((v, 0))
                elif dep[v] < dep[u]:                # обратное ребро: цикл v … u
                    pr = q[u]; w = u
                    while w != v:
                        w = par[w]; pr = pr * q[w] % M
                    ans += pr
            else:
                st.pop()
    print(ans % M)
main()
''',
]

# ---------------------------------------------------------------------------
# Переборные решения (для малых тестов) — функции от текста ввода
# ---------------------------------------------------------------------------


def brute_lamps(inp):
    data = inp.split()
    t = int(data[0]); p = 1; out = []
    for _ in range(t):
        n, k = int(data[p]), int(data[p + 1]); s = [int(c) for c in data[p + 2]]; p += 3
        best = -1
        for mask in range(1 << n):
            st = s[:]
            for i in range(n):
                if mask >> i & 1:
                    for d in range(k):
                        st[(i + d) % n] ^= 1
            if not any(st):
                c = bin(mask).count("1")
                if best == -1 or c < best:
                    best = c
        out.append(str(best))
    return "\n".join(out)


def brute_repeats(inp):
    s = inp.strip()
    n = len(s)
    res = set()
    for L in range(1, n // 2 + 1):
        first = {}
        for i in range(n - L + 1):
            t = s[i:i + L]
            if t in first:
                if i >= first[t] + L:
                    res.add(t)
            else:
                first[t] = i
    return str(len(res))


def brute_gcd2(inp):
    n = int(inp)
    return str(sum(gcd(i, j) ** 2 for i in range(1, n + 1) for j in range(1, n + 1)) % MOD)


def brute_cactus(inp):
    data = inp.split()
    n, m = int(data[0]), int(data[1])
    ps = [Fraction(int(data[2 + i]), 100) for i in range(n)]
    edges = [(int(data[2 + n + 2 * e]) - 1, int(data[3 + n + 2 * e]) - 1) for e in range(m)]
    E = Fraction(0)
    for alive in itertools.product((0, 1), repeat=n):
        pr = Fraction(1)
        for i in range(n):
            pr *= (1 - ps[i]) if alive[i] else ps[i]
        if pr == 0:
            continue
        par = list(range(n))

        def f(x):
            while par[x] != x:
                par[x] = par[par[x]]; x = par[x]
            return x
        comp = sum(alive)
        for u, v in edges:
            if alive[u] and alive[v] and f(u) != f(v):
                par[f(u)] = f(v); comp -= 1
        E += pr * comp
    return str(E.numerator % MOD * pow(E.denominator % MOD, MOD - 2, MOD) % MOD)


BRUTE = [brute_lamps, brute_repeats, brute_gcd2, brute_cactus]

# ---------------------------------------------------------------------------
# Генераторы тестов
# ---------------------------------------------------------------------------


def _solvable_lamps(rng, n, k):
    x = [rng.randint(0, 1) for _ in range(n)]
    diff = [0] * (n + 1)  # разностный массив: окно [i, i+k) по модулю n
    for i in range(n):
        if x[i]:
            j = i + k
            diff[i] ^= 1
            if j <= n:
                diff[j] ^= 1
            else:
                diff[n] ^= 1; diff[0] ^= 1; diff[j - n] ^= 1
    s, cur = [0] * n, 0
    for i in range(n):
        cur ^= diff[i]; s[i] = cur
    return s


def _lamps_case(rng, n, k, mode):
    if mode == "solvable":
        s = _solvable_lamps(rng, n, k)
    elif mode == "near":  # разрешимый с одной испорченной лампой — чаще всего −1
        s = _solvable_lamps(rng, n, k)
        s[rng.randrange(n)] ^= 1
    else:
        s = [rng.randint(0, 1) for _ in range(n)]
    return f"{n} {k}\n{''.join(map(str, s))}"


def _big_k_case(rng, n):
    """k с большим НОД и нечётным k/g — там, где важна поправка на чётность."""
    divs = [d for d in range(1, n + 1) if n % d == 0]
    g = rng.choice(divs)
    mult = [t for t in range(1, n // g + 1) if gcd(t, n // g) == 1]
    k = g * rng.choice(mult)
    return n, k


def gen_lamps(rng, size):
    cases = []
    if size == "small":
        for _ in range(60):
            n = rng.randint(1, 10)
            k = rng.randint(1, n)
            cases.append(_lamps_case(rng, n, k, rng.choice(["solvable", "solvable", "near", "rand"])))
    elif size == "mid":
        tot = 0
        while tot < 2000:
            n = rng.randint(1, 60)
            n, k = _big_k_case(rng, n) if rng.random() < 0.6 else (n, rng.randint(1, n))
            cases.append(_lamps_case(rng, n, k, rng.choice(["solvable", "solvable", "near"])))
            tot += n
    else:
        n1 = 200000 - rng.randint(0, 50)
        n2 = 100000
        for n in (n1, n2):
            divs = [d for d in range(1, n // 3) if n % d == 0]
            g = max(d for d in divs if (n // d) % 2 == 1 or d < 50)
            k = g * next(t for t in range(n // g // 3, n // g) if gcd(t, n // g) == 1 and t % 2 == 1)
            cases.append(_lamps_case(rng, n, k, "solvable"))
    return f"{len(cases)}\n" + "\n".join(cases) + "\n"


def gen_repeats(rng, size):
    if size == "small":
        n = rng.randint(1, 14)
        return "".join(rng.choice("ab" if rng.random() < 0.6 else "abc") for _ in range(n)) + "\n"
    if size == "mid":
        n = rng.randint(200, 2000)
    else:
        n = 100000
    kind = rng.choice(["rand2", "rand26", "period", "fib"])
    if kind == "rand2":
        s = "".join(rng.choice("ab") for _ in range(n))
    elif kind == "rand26":
        s = "".join(rng.choice("abcdefghijklmnopqrstuvwxyz") for _ in range(n))
    elif kind == "period":
        per = "".join(rng.choice("abc") for _ in range(rng.randint(1, 7)))
        s = (per * (n // len(per) + 1))[:n]
        s = list(s)
        for _ in range(rng.randint(0, 3)):
            s[rng.randrange(n)] = "z"
        s = "".join(s)
    else:
        a, b = "a", "ab"
        while len(b) < n:
            a, b = b, b + a
        s = b[:n]
    return s + "\n"


def gen_gcd2(rng, size):
    if size == "small":
        return f"{rng.randint(1, 60)}\n"
    if size == "mid":
        return f"{rng.randint(10 ** 5, 10 ** 7)}\n"
    return f"{rng.choice([2 * 10 ** 9, 2 * 10 ** 9 - rng.randint(1, 10 ** 6), 1999999973])}\n"


def _cactus(rng, n, cyc_frac, max_cyc):
    """Случайный рёберный кактус: дерево, в котором часть путей замкнута в циклы."""
    edges = set()
    order = list(range(1, n + 1))
    rng.shuffle(order)
    i = 1
    comps_roots = [order[0]]
    while i < n:
        if rng.random() < 0.02:  # иногда начинаем новую компоненту
            comps_roots.append(order[i]); i += 1
            continue
        u = order[rng.randrange(i)]
        if rng.random() < cyc_frac and n - i >= 2:
            L = rng.randint(2, min(max_cyc, n - i))
            chain = order[i:i + L]
            prev = u
            for w in chain:
                edges.add((prev, w)); prev = w
            edges.add((prev, u))
            i += L
        else:
            edges.add((u, order[i])); i += 1
    edges = [(a, b) if rng.random() < 0.5 else (b, a) for a, b in edges]
    rng.shuffle(edges)
    return edges


def gen_cactus(rng, size):
    if size == "small":
        n = rng.randint(1, 10)
        edges = _cactus(rng, n, 0.4, 5)
    elif size == "mid":
        n = rng.randint(300, 3000)
        edges = _cactus(rng, n, 0.5, rng.choice([3, 20, 300]))
    else:
        n = 200000
        edges = _cactus(rng, n, 0.6, rng.choice([3, 50, 100000]))
    ps = [rng.choice([0, 100, rng.randint(0, 100)]) if rng.random() < 0.2 else rng.randint(0, 100) for _ in range(n)]
    return f"{n} {len(edges)}\n{' '.join(map(str, ps))}\n" + "\n".join(f"{u} {v}" for u, v in edges) + "\n"


GENS = [gen_lamps, gen_repeats, gen_gcd2, gen_cactus]
# Состав скрытых тестов: (размер, сколько)
PLAN = [("small", 4), ("mid", 4), ("max", 2)]
SAMPLES = [
    ("4\n3 2\n110\n4 2\n1000\n5 5\n11111\n6 4\n101101\n", "1\n-1\n1\n2"),
    ("abcabcab\n", "9"),
    ("10\n", "655"),
    ("3 3\n50 50 50\n1 2\n2 3\n3 1\n", "124780545"),
]


def make_tests(seed, k):
    """Скрытые тесты задачи k (с 0): список входов. Ответы считает эталон."""
    rng = random.Random(seed * 7919 + 104729 * (k + 1))
    tests = [SAMPLES[k][0]]
    for size, cnt in PLAN:
        for _ in range(cnt):
            tests.append(GENS[k](rng, size))
    return tests


# ---------------------------------------------------------------------------
# Запуск решений
# ---------------------------------------------------------------------------

def _limits():
    try:
        import resource
        resource.setrlimit(resource.RLIMIT_AS, (1 << 30, 1 << 30))
    except Exception:  # noqa: BLE001
        pass


def run_code(code, inp, tl):
    """-> (статус, вывод). Статус: OK / TL / RE."""
    with tempfile.TemporaryDirectory() as td:
        path = os.path.join(td, "sol.py")
        with open(path, "w", encoding="utf-8") as f:
            f.write(code)
        try:
            p = subprocess.run([sys.executable, "-I", path], input=inp.encode(), capture_output=True,
                               timeout=tl, cwd=td, preexec_fn=_limits if os.name == "posix" else None)
        except subprocess.TimeoutExpired:
            return "TL", ""
        if p.returncode != 0:
            err = p.stderr.decode(errors="replace").strip().splitlines()
            return "RE", (err[-1] if err else f"код возврата {p.returncode}")[:160]
        return "OK", p.stdout.decode(errors="replace")


def same(a, b):
    return a.split() == b.split()


def expected_outputs(seed, k):
    """Ответы эталона на скрытые тесты (кэшируются в памяти)."""
    key = (seed, k)
    if key not in _EXP:
        outs = []
        for inp in make_tests(seed, k):
            st, out = run_code(REF[k], inp, 120)
            assert st == "OK", f"эталон задачи {k + 1}: {st} {out}"
            outs.append(out)
        _EXP[key] = outs
    return _EXP[key]


_EXP = {}


def extract_programs(text, n):
    """Блоки ```python … ``` из ответа; задача определяется по комментарию «# задача N» в первых строках,
    иначе — по порядку блоков."""
    blocks = re.findall(r"```(?:python|py|Python|python3)?[ \t]*\n(.*?)```", text or "", flags=re.S)
    progs = {}
    rest = []
    for b in blocks:
        head = "\n".join(b.strip().splitlines()[:3])
        m = re.search(r"#\s*(?:задача|task|problem)\s*(\d+)", head, flags=re.I)
        if m and 1 <= int(m.group(1)) <= n and int(m.group(1)) not in progs:
            progs[int(m.group(1))] = b
        else:
            rest.append(b)
    for k in range(1, n + 1):
        if k not in progs and rest:
            progs[k] = rest.pop(0)
    return progs


class CodeTask(Task):
    key = "code"
    title = "Программирование"
    channel = 5
    ability = "coding"

    def generate(self, seed):
        return {"seed": seed, "problems": [{"name": s["name"], "tl": s["tl"], "n_tests": 1 + sum(c for _, c in PLAN)}
                                           for s in STATEMENTS]}

    def n_items(self, inst):
        return len(inst["problems"])

    def prompt(self, inst, only=None):
        idx = range(len(STATEMENTS)) if only is None else [only]
        L = ["# Олимпиадное программирование", "",
             "Реши задачи: для каждой напиши программу на Python 3 (только стандартная библиотека). Программа читает "
             "ввод из stdin и пишет ответ в stdout. Её запустят на скрытых тестах, включая тесты максимального размера; "
             "ограничение времени указано у задачи (интерпретатор CPython на обычном сервере), память — 1 ГБ. "
             "Балл задачи — доля пройденных тестов.", ""]
        for j, i in enumerate(idx, 1):
            s = STATEMENTS[i]
            L += [f"## Задача {j}. {s['name']} (ограничение времени {s['tl']:.0f} с)", "", s["text"], ""]
        L += ["## Формат ответа",
              "Для каждой задачи — один блок кода ```python, первая строка которого — комментарий `# задача N`. "
              "Пояснения вне блоков не нужны."]
        return "\n".join(L)

    def prompt_item(self, inst, k):
        return self.prompt(inst, only=k - 1).replace("# задача N", "# задача 1")

    @staticmethod
    def merge_items(texts):
        parts = []
        for k, t in enumerate(texts, 1):
            progs = extract_programs(t, 1)
            if 1 in progs:
                body = re.sub(r"^\s*#\s*(?:задача|task|problem)\s*\d+[^\n]*\n", "", progs[1], flags=re.I)
                parts.append(f"```python\n# задача {k}\n{body}```")
        return "\n\n".join(parts)

    def score(self, inst, text):
        seed = inst["seed"]
        progs = extract_programs(text, len(STATEMENTS))
        items = []
        for k, s in enumerate(STATEMENTS):
            code = progs.get(k + 1)
            n_t = 1 + sum(c for _, c in PLAN)
            if not code:
                items.append({"ok": 0.0, "label": s["name"], "detail": "нет программы", "tests": []})
                continue
            exp = expected_outputs(seed, k)
            res = []
            for inp, want in zip(make_tests(seed, k), exp):
                st, out = run_code(code, inp, s["tl"])
                res.append("OK" if st == "OK" and same(out, want) else ("WA" if st == "OK" else st))
            passed = res.count("OK")
            bad = [f"{v}×{res.count(v)}" for v in ("WA", "TL", "RE") if res.count(v)]
            first_bad = next((i for i, v in enumerate(res) if v != "OK"), None)
            det = f"пройдено {passed} из {n_t}" + (f" ({', '.join(bad)}; первый сбой — тест {first_bad + 1})" if bad else "")
            items.append({"ok": passed / n_t, "label": s["name"], "detail": det, "tests": res})
        return sum(it["ok"] for it in items) / len(items), items
