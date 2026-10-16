"""Канал «Матрёшка»: задача о кубике, в которую пять раз вложена олимпиадная задача.

Упражнение «усложни вдвое и повтори пять раз»: берём задачу из скоупа «Кубик» (кратчайший маршрут
к нужной ориентации) и прячем её параметр в ответ новой задачи, затем параметр этой задачи — в ответ
следующей и т. д. Слои (изнутри наружу):

  5. теория чисел (IMO 1988/6, прыжки Виета)      — без входа
  4. физика (бильярд Гальперина)                   — вход: ответ слоя 5
  3. программирование (что выведет DP-программа)   — вход: ответ слоя 4
  2. теория игр (игра Уитхоффа на больших кучках)  — вход: ответ слоя 3
  1. программирование (рекурсия «Ханой на 4 стержнях») — вход: ответ слоя 2
  0. кубик: кратчайший маршрут, целевая ориентация задаётся ответом слоя 1

Оценка двойная. «По слоям»: каждый слой проверяется по входу, который дала сама модель, поэтому ошибка
внутри не обнуляет внешние слои. «Цепочка»: сколько слоёв подряд изнутри совпало с верной цепочкой.
Итог задачи — среднее этих двух чисел.
"""
from __future__ import annotations

import json
import math
import random
import re
from fractions import Fraction
from functools import lru_cache

from .common import Task
from .cube import INIT, STEP, land, roll, run2, plan_bfs, path_to

COLORS = ["R", "O", "G", "B", "Y", "W"]
OPP = {"R": "O", "O": "R", "G": "B", "B": "G", "Y": "W", "W": "Y"}
N5 = 10 ** 6
N5_TXT = "1 000 000"
LAYERS = [5, 4, 3, 2, 1, 0]


# ---------------------------------------------------------------------------
# эталонные решения слоёв (работают для любого входа — нужно для оценки «по слоям»)
# ---------------------------------------------------------------------------

def solve5(N=N5):
    """Сумма k = (a²+b²)/(ab+1) по упорядоченным парам 1 ≤ a, b ≤ N, где k — целое."""
    total = 1  # (1, 1), k = 1
    t = 2
    while t ** 3 <= N:
        x0, x1 = 0, t
        while True:
            x2 = t * t * x1 - x0
            if x2 > N:
                break
            total += 2 * t * t  # пары (x1, x2) и (x2, x1)
            x0, x1 = x1, x2
        t += 1
    return total


def brute5(N):
    s = 0
    for a in range(1, N + 1):
        for b in range(1, N + 1):
            q, r = divmod(a * a + b * b, a * b + 1)
            if r == 0:
                s += q
    return s


def param4(a5):
    return 100 + int(a5) % 9901


def solve4(A):
    """Число упругих ударов: лёгкий брусок m у стены, тяжёлый M = A·m налетает со скоростью 1."""
    m, M = Fraction(1), Fraction(A)
    v1, v2 = Fraction(0), Fraction(-1)
    n = 0
    while True:
        if v2 < v1:  # бруски сближаются (лёгкий между стеной и тяжёлым)
            v1, v2 = ((m - M) * v1 + 2 * M * v2) / (m + M), ((M - m) * v2 + 2 * m * v1) / (m + M)
            n += 1
        elif v1 < 0:
            v1 = -v1
            n += 1
        else:
            return n


def param3(a4):
    return 10 + int(a4) % 11


@lru_cache(maxsize=None)
def solve3(n):
    """Число замощений прямоугольника 4 × n доминошками (перебор профилей по столбцам)."""
    H = 4

    def fills(mask):
        out = []

        def rec(row, cur, nxt):
            if row == H:
                out.append(nxt)
                return
            if cur >> row & 1:
                rec(row + 1, cur, nxt)
                return
            rec(row + 1, cur | 1 << row, nxt | 1 << row)  # горизонтальная доминошка торчит в следующий столбец
            if row + 1 < H and not cur >> (row + 1) & 1:
                rec(row + 2, cur | 3 << row, nxt)  # вертикальная доминошка внутри столбца
        rec(0, mask, 0)
        return out
    dp = {0: 1}
    for _ in range(n):
        nd = {}
        for mask, w in dp.items():
            for nm in fills(mask):
                nd[nm] = nd.get(nm, 0) + w
        dp = nd
    return dp.get(0, 0)


def param2(a3):
    a3 = int(a3)
    x = 100000 + a3 % 900000
    y = x + 1 + (a3 // 7) % 100000
    return x, y


def wythoff_a(k):
    return (k + math.isqrt(5 * k * k)) // 2  # ⌊kφ⌋ точно


def wythoff_moves(x, y):
    """Все выигрышные ходы из (x, y): список позиций (x', y'), куда можно пойти, чтобы соперник проиграл."""
    if x > y:
        x, y = y, x
    res = set()

    def is_p(a, b):
        a, b = min(a, b), max(a, b)
        k = b - a
        return wythoff_a(k) == a
    # уменьшаем одну кучку
    # (1) кучка y -> y' при фиксированном x: P-позиция с меньшей координатой x или с большей координатой x
    for k in _k_with_value(x):
        a, b = wythoff_a(k), wythoff_a(k) + k
        if a == x and b < y:
            res.add((x, b))
        if b == x and a < y:
            res.add((a, x))
    # (2) кучка x -> x' при фиксированном y
    for k in _k_with_value(y):
        a, b = wythoff_a(k), wythoff_a(k) + k
        if b == y and a < x:
            res.add((a, y))
        if a == y and b < x:
            res.add((b, y))
    # (3) обе кучки поровну: разность сохраняется
    k = y - x
    a = wythoff_a(k)
    if a < x:
        res.add((a, a + k))
    return sorted((min(p), max(p)) for p in res if is_p(*p))


def _k_with_value(v):
    """k, для которых ⌊kφ⌋ = v или ⌊kφ⌋ + k = v (кандидаты около v/φ и v/φ²)."""
    phi = (1 + 5 ** 0.5) / 2
    out = set()
    for est in (v / phi, v / (phi * phi)):
        k0 = int(est)
        for k in range(max(0, k0 - 3), k0 + 4):
            out.add(k)
    return out


def solve2(a3):
    x, y = param2(a3)
    mv = wythoff_moves(x, y)
    return sum(a + b for a, b in mv), mv


def brute_wythoff_moves(x, y):
    """Проверка на малых кучках: P-позиции перебором."""
    lim = max(x, y) + 1
    P = [[False] * lim for _ in range(lim)]
    for a in range(lim):
        for b in range(lim):
            win = False
            for t in range(1, a + 1):
                if P[a - t][b]:
                    win = True
                    break
            if not win:
                for t in range(1, b + 1):
                    if P[a][b - t]:
                        win = True
                        break
            if not win:
                for t in range(1, min(a, b) + 1):
                    if P[a - t][b - t]:
                        win = True
                        break
            P[a][b] = not win
    res = set()
    for t in range(1, x + 1):
        if P[x - t][y]:
            res.add((min(x - t, y), max(x - t, y)))
    for t in range(1, y + 1):
        if P[x][y - t]:
            res.add((min(x, y - t), max(x, y - t)))
    for t in range(1, min(x, y) + 1):
        if P[x - t][y - t]:
            res.add((x - t, y - t))
    return sorted(res)


def param1(a2):
    return 40 + int(a2) % 60


@lru_cache(maxsize=None)
def solve1(n):
    """h(n) = min_k 2·h(k) + 2^(n−k) − 1 (числа Фрейма — Стюарта для 4 стержней)."""
    h = [0] * (n + 1)
    for i in range(1, n + 1):
        h[i] = min(2 * h[k] + 2 ** (i - k) - 1 for k in range(i))
    return h[n]


def target_from(a1):
    """Ответ слоя 1 -> целевая ориентация (верх, север) на G: i = a1 mod 24."""
    i = int(a1) % 24
    top = COLORS[i // 4]
    side = [c for c in COLORS if c not in (top, OPP[top])]
    return {"top": top, "north": side[i % 4]}


# ---------------------------------------------------------------------------
# слой 0: поле кубика
# ---------------------------------------------------------------------------

def _plan_table(rows):
    """Кратчайшие маршруты до G для всех 24 ориентаций (верх, север)."""
    prev = plan_bfs(rows)
    g = next((r, row.index("G")) for r, row in enumerate(rows) if "G" in row)
    best = {}
    for st in prev:
        if (st[0], st[1]) != g:
            continue
        key = (st[2], st[4])  # top, north
        L = len(path_to(prev, st))
        if key not in best or L < best[key][0]:
            best[key] = (L, path_to(prev, st))
    return best


def gen_board(rng, H=8, W=8, holes=0.2, turns=3, flips=3, min_opt=12):
    for _ in range(4000):
        b = [["." for _ in range(W)] for _ in range(H)]
        cells = [(r, c) for r in range(H) for c in range(W)]
        rng.shuffle(cells)
        s, g = cells[0], cells[1]
        if abs(s[0] - g[0]) + abs(s[1] - g[1]) < (H + W) // 2:
            continue
        b[s[0]][s[1]] = "S"
        b[g[0]][g[1]] = "G"
        k = 2
        for _ in range(int(H * W * holes)):
            r, c = cells[k]
            b[r][c] = "#"
            k += 1
        for ch, n in (("@", turns), ("%", flips)):
            for _ in range(n):
                r, c = cells[k]
                b[r][c] = ch
                k += 1
        rows = ["".join(x) for x in b]
        best = _plan_table(rows)
        if len(best) < 24:
            continue
        if min(L for L, _ in best.values()) < min_opt:
            continue
        return rows, best
    raise RuntimeError("не удалось сгенерировать поле матрёшки")


def solve0(rows, a1):
    tg = target_from(a1)
    L, sol = _plan_table(tuple(rows))[(tg["top"], tg["north"])] if False else _plan_cached(tuple(rows))[(tg["top"], tg["north"])]
    return tg, L, sol


@lru_cache(maxsize=8)
def _plan_cached(rows):
    return _plan_table(list(rows))


# ---------------------------------------------------------------------------
# задача
# ---------------------------------------------------------------------------

CUBE_RULES = [
    "Кубик стоит на клетке `S` поля 8 × 8. Верхняя строка поля — север, правый столбец — восток.",
    "Грани кубика в начале: верх R, низ O, север G, юг B, восток Y, запад W "
    "(R — красный, O — оранжевый, G — зелёный, B — синий, Y — жёлтый, W — белый; противоположные пары R–O, G–B, Y–W).",
    "Ход N/S/E/W — кубик перекатывается через нижнее ребро на соседнюю клетку в этом направлении "
    "(при ходе E грань, смотревшая на восток, становится нижней, а верхняя — восточной).",
    "Когда кубик встаёт на клетку: `.`, `S`, `G` — ничего; `@` — поворотный круг: поворот на 90° по часовой стрелке, "
    "если смотреть сверху (северная грань становится восточной, восточная — южной…; верх и низ не меняются); "
    "`%` — переворот на 180° вокруг оси север–юг (верх и низ меняются местами, восток и запад — тоже); "
    "`#` — яма, вставать нельзя; уходить за край поля тоже нельзя.",
]


class MatryoshkaTask(Task):
    key = "matryoshka"
    title = "Матрёшка"
    channel = 4
    ability = "olympiad"
    multi_turn = True
    turn_names = ["L5", "L4", "L3", "L2", "L1", "L0"]

    def __init__(self, difficulty=5):
        self.difficulty = difficulty

    def generate(self, seed):
        rng = random.Random(seed * 1000003 + 77)
        rows, best = gen_board(rng)
        a5 = solve5()
        A = param4(a5)
        a4 = solve4(A)
        n3 = param3(a4)
        a3 = solve3(n3)
        a2, moves2 = solve2(a3)
        n1 = param1(a2)
        a1 = solve1(n1)
        tg, L, sol = solve0(rows, a1)
        chain = {"L5": a5, "L4": a4, "L3": a3, "L2": a2, "L1": a1, "L0": sol}
        return {"seed": seed, "difficulty": self.difficulty, "board": rows, "chain": chain,
                "params": {"A": A, "n3": n3, "xy": list(param2(a3)), "moves2": moves2, "n1": n1, "target": tg},
                "optimal": L}

    # ---- промпт
    def statement(self, inst):
        L = [
            "# Матрёшка: шесть вложенных задач",
            "",
            "Внешняя задача (слой 0) — маршрут кубика. Её условие зависит от ответа слоя 1, условие слоя 1 — от ответа "
            "слоя 2 и так далее; слой 5 ни от чего не зависит. Решать нужно изнутри наружу: 5 → 4 → 3 → 2 → 1 → 0. "
            "Все ответы слоёв 1–5 — целые неотрицательные числа, «mod» — остаток от деления.",
            "",
            "## Слой 5. Теория чисел",
            "Найдите сумму значений (a² + b²) / (ab + 1) по всем упорядоченным парам натуральных чисел (a, b) "
            f"с a ≤ {N5_TXT} и b ≤ {N5_TXT}, для которых это значение — целое число. (Пары (a, b) и (b, a) при a ≠ b считаются разными.)",
            "",
            "## Слой 4. Физика",
            "На гладком горизонтальном полу у вертикальной стены покоится брусок массы m. К нему издалека, перпендикулярно "
            "стене, движется брусок массы M = A·m, где A = 100 + (ответ слоя 5 mod 9901). Бруски движутся поступательно по одной "
            "прямой, лёгкий — между стеной и тяжёлым; все удары абсолютно упругие, трения нет, размеры брусков пренебрежимо малы "
            "по сравнению с расстояниями. Сколько всего ударов произойдёт — брусок о брусок и брусок о стену вместе?",
            "",
            "## Слой 3. Программирование",
            "Что выведет программа при n = 10 + (ответ слоя 4 mod 11)?",
            "```python",
            "def fills(mask, H=4):",
            "    out = []",
            "    def rec(row, cur, nxt):",
            "        if row == H:",
            "            out.append(nxt); return",
            "        if cur >> row & 1:",
            "            rec(row + 1, cur, nxt); return",
            "        rec(row + 1, cur | 1 << row, nxt | 1 << row)",
            "        if row + 1 < H and not cur >> (row + 1) & 1:",
            "            rec(row + 2, cur | 3 << row, nxt)",
            "    rec(0, mask, 0)",
            "    return out",
            "",
            "dp = {0: 1}",
            "for _ in range(n):",
            "    nd = {}",
            "    for mask, w in dp.items():",
            "        for nm in fills(mask):",
            "            nd[nm] = nd.get(nm, 0) + w",
            "    dp = nd",
            "print(dp.get(0, 0))",
            "```",
            "",
            "## Слой 2. Теория игр",
            "Игра Уитхоффа: на столе две кучки камней. Игроки ходят по очереди; за ход можно взять любое положительное число "
            "камней из одной кучки или одинаковое положительное число из обеих. Кто не может сделать ход, проиграл. "
            "Пусть t — ответ слоя 3; в кучках x = 100000 + (t mod 900000) и y = x + 1 + (⌊t / 7⌋ mod 100000) камней, ходит первый игрок. "
            "Найдите все его выигрышные ходы (ходы в позицию, проигрышную для соперника). Ответ слоя — сумма (x′ + y′) по всем "
            "позициям (x′, y′), в которые ведут выигрышные ходы; если выигрышных ходов нет, ответ — 0.",
            "",
            "## Слой 1. Программирование",
            "Что выведет программа при n = 40 + (ответ слоя 2 mod 60)?",
            "```python",
            "from functools import lru_cache",
            "@lru_cache(maxsize=None)",
            "def h(n):",
            "    if n == 0:",
            "        return 0",
            "    return min(2 * h(k) + 2 ** (n - k) - 1 for k in range(n))",
            "print(h(n))",
            "```",
            "",
            "## Слой 0. Кубик",
            *CUBE_RULES,
            "Пусть i = (ответ слоя 1) mod 24. Цвета пронумерованы в порядке R, O, G, B, Y, W. Целевая верхняя грань — цвет "
            "номер ⌊i / 4⌋ (с нуля); целевая северная — цвет номер (i mod 4) среди четырёх цветов, соседних с верхним "
            "(в том же порядке R, O, G, B, Y, W, без верхнего и противоположного ему).",
            "Поле:",
            "```",
            *inst["board"],
            "```",
            "Найдите кратчайший маршрут из S в G, после которого у кубика на G целевые верхняя и северная грани. "
            "Полный балл — за самый короткий маршрут; верный, но длиннее оптимума на 1–2 хода — 0.85, ещё длиннее — 0.7.",
            "",
            "## Как сдавать",
            "Ответы сдаются по слоям, по одному сообщению на слой, от 5 к 0; каждый ответ — ОДИН блок JSON без пояснений. "
            "Проверяется каждый слой по вашему же входу из предыдущего слоя, а также вся цепочка целиком.",
        ]
        return "\n".join(L)

    def turns(self, inst):
        first = self.statement(inst) + "\n\nСейчас пришлите ответ слоя 5: {\"L5\": <целое>}"
        rest = [f"Принято. Теперь слой {k}: {{\"L{k}\": <целое>}}" for k in (4, 3, 2, 1)]
        rest.append("Принято. Теперь слой 0: {\"L0\": {\"moves\": \"<ходы N/S/E/W>\"}}")
        return [first] + rest

    def prompt(self, inst):  # для совместимости: весь текст условия
        return self.statement(inst)

    # ---- оценка
    @staticmethod
    def parse(text):
        """Ответы из всех сообщений: {'L5': int, ..., 'L0': 'moves'}."""
        got = {}
        for m in re.finditer(r"\{[^{}]*(\{[^{}]*\}[^{}]*)*\}", text or ""):
            try:
                d = json.loads(m.group(0))
            except Exception:  # noqa: BLE001
                continue
            if not isinstance(d, dict):
                continue
            for k, v in d.items():
                ks = str(k).upper()
                if ks in ("L5", "L4", "L3", "L2", "L1"):
                    try:
                        got[ks] = int(str(v).replace(" ", "").replace("_", ""))
                    except ValueError:
                        pass
                elif ks == "L0":
                    mv = v.get("moves") if isinstance(v, dict) else v
                    got["L0"] = "".join(ch for ch in str(mv or "").upper() if ch in "NSEW")
        return got

    def score(self, inst, text):
        got = self.parse(text)
        ch = inst["chain"]
        items = []
        # вход каждого слоя — ответ модели на предыдущий слой (если его нет — верный ответ)
        inp = {}
        exp = {}
        exp["L5"] = ch["L5"]
        a5 = got.get("L5", ch["L5"])
        exp["L4"] = solve4(param4(a5))
        a4 = got.get("L4", exp["L4"])
        exp["L3"] = solve3(param3(a4))
        a3 = got.get("L3", exp["L3"])
        exp["L2"] = solve2(a3)[0]
        a2 = got.get("L2", exp["L2"])
        exp["L1"] = solve1(param1(a2))
        a1 = got.get("L1", exp["L1"])
        inp.update(L4=a5, L3=a4, L2=a3, L1=a2, L0=a1)
        names = {"L5": "Слой 5 · теория чисел", "L4": "Слой 4 · физика", "L3": "Слой 3 · программа (DP)",
                 "L2": "Слой 2 · игра Уитхоффа", "L1": "Слой 1 · программа (Ханой)", "L0": "Слой 0 · кубик"}
        cond = {}
        for k in ("L5", "L4", "L3", "L2", "L1"):
            ok = got.get(k) == exp[k]
            cond[k] = 1.0 if ok else 0.0
            det = (f"ответ {got.get(k, '—')}; верно {exp[k]}" + ("" if k == "L5" else f" (по вашему входу {inp[k]})"))
            items.append({"ok": cond[k], "label": names[k], "detail": det, "got": got.get(k)})
        # слой 0
        tg = target_from(a1)
        L_opt = _plan_cached(tuple(inst["board"]))[(tg["top"], tg["north"])][0]
        mv = got.get("L0", "")
        stamps, f, trace = run2(inst["board"], mv)
        g = next((r, row.index("G")) for r, row in enumerate(inst["board"]) if "G" in row)
        if not mv:
            s0, det = 0.0, "маршрута нет"
        elif f is None:
            s0, det = 0.0, f"ход {len(trace)} невозможен (яма или край)"
        elif (trace[-1][0], trace[-1][1]) != g:
            s0, det = 0.0, "кубик не на G"
        elif f["top"] != tg["top"] or f["north"] != tg["north"]:
            s0, det = 0.2, f"на G, но верх/север {f['top']}/{f['north']} вместо {tg['top']}/{tg['north']}"
        else:
            s0 = 1.0 if len(mv) <= L_opt else (0.85 if len(mv) <= L_opt + 2 else 0.7)
            det = f"верно за {len(mv)} ходов (оптимум {L_opt}); цель {tg['top']}/{tg['north']}"
        cond["L0"] = s0
        items.append({"ok": s0, "label": names["L0"], "detail": det, "got": mv})
        # цепочка: сколько слоёв подряд изнутри совпали с верной цепочкой
        strict = 0
        for k in ("L5", "L4", "L3", "L2", "L1"):
            if got.get(k) == ch[k]:
                strict += 1
            else:
                break
        else:
            if cond["L0"] >= 0.999 and a1 == ch["L1"]:
                strict += 1
        s_cond = sum(cond.values()) / 6
        s_strict = strict / 6
        for it in items:
            it["chain"] = strict
        total = (s_cond + s_strict) / 2
        items[0]["summary"] = {"by_layer": s_cond, "chain": strict, "total": total}
        return total, items
