"""Набор «Программирование» (v2): оптимизационные задачи на скрытых тестах.

Модель пишет программу на Python 3; программу запускают на скрытых экземплярах с ограничением
времени, проверяют корректность вывода и меряют качество решения. Балл экземпляра растёт плавно:
0 — уровень простой жадной эвристики, 1 — уровень эталона: долгий прогон сильного алгоритма на C
(исходники в opt_ref/), а для упаковки — нижняя граница по площади. Потолка «решил — не решил» нет: чем лучше алгоритм и чем аккуратнее он
распоряжается временем, тем выше балл.

Первая версия набора — четыре классические олимпиадные задачи (code.py) — оказалась для
Opus 5.5 слишком лёгкой: 4 из 4 за 2.5 минуты; она осталась доступна как набор `code1`.
"""
from __future__ import annotations

import json
import math
import os
import random
import re

from .code import CodeTask, run_code, extract_programs
from .common import Task

DATA = os.path.join(os.path.dirname(os.path.dirname(__file__)), "data")

# ---------------------------------------------------------------------------
# Экземпляры
# ---------------------------------------------------------------------------


def gen_tsp(rng, n, kind):
    pts = []
    if kind == "uniform":
        pts = [(rng.randint(0, 10 ** 6), rng.randint(0, 10 ** 6)) for _ in range(n)]
    else:  # кластеры разного размера
        centers = [(rng.randint(10 ** 5, 9 * 10 ** 5), rng.randint(10 ** 5, 9 * 10 ** 5), rng.choice([8000, 20000, 60000]))
                   for _ in range(25)]
        while len(pts) < n:
            cx, cy, sd = rng.choice(centers)
            x, y = int(rng.gauss(cx, sd)), int(rng.gauss(cy, sd))
            if 0 <= x <= 10 ** 6 and 0 <= y <= 10 ** 6:
                pts.append((x, y))
    return f"{n}\n" + "\n".join(f"{x} {y}" for x, y in pts) + "\n"


def gen_color(rng, n, p):
    edges = [(u, v) for u in range(n) for v in range(u + 1, n) if rng.random() < p]
    return f"{n} {len(edges)}\n" + "\n".join(f"{u} {v}" for u, v in edges) + "\n"


def gen_strip(rng, n, W):
    rects = []
    for _ in range(n):
        k = rng.random()
        if k < 0.25:
            w, h = rng.randint(30, 60), rng.randint(2, 12)    # широкие плоские
        elif k < 0.5:
            w, h = rng.randint(2, 12), rng.randint(30, 60)    # узкие высокие
        else:
            w, h = rng.randint(5, 30), rng.randint(5, 30)
        rects.append((w, h))
    return f"{n} {W}\n" + "\n".join(f"{w} {h}" for w, h in rects) + "\n"


PROBLEMS = [
    {
        "key": "tsp", "name": "Коммивояжёр", "tl": 10.0,
        "text": """На плоскости даны n городов с целыми координатами. Найдите замкнутый маршрут, проходящий через
каждый город ровно один раз, как можно меньшей длины (евклидово расстояние).

**Ввод.** n (n = 2000), затем n строк «x y» (0 ≤ x, y ≤ 10⁶). Города нумеруются с 0.

**Вывод.** Перестановка чисел 0…n−1 — порядок обхода (через пробел или по строкам).

**Оценка.** Длина L сравнивается с туром «ближайший сосед» от города 0 (L₀, 0 баллов) и с эталоном
(сильный алгоритм, долгий прогон; полный балл): балл = (L₀ − L)/(L₀ − L*), от 0 до 1.
Экземпляры: равномерно случайные точки и точки, собранные в кластеры.""",
        "instances": [("uniform", 2000), ("clustered", 2000)],
    },
    {
        "key": "color", "name": "Раскраска графа", "tl": 10.0,
        "text": """Дан неориентированный граф. Раскрасьте вершины в как можно меньшее число цветов так, чтобы концы
каждого ребра были разного цвета.

**Ввод.** n m, затем m строк «u v» (0 ≤ u, v < n). Экземпляры — случайные графы: n = 250, вероятность
ребра ½ (около 15 500 рёбер), и n = 500, вероятность ребра ⅒.

**Вывод.** n целых чисел — цвета вершин 0…n−1 (любые неотрицательные целые).

**Оценка.** Число цветов C сравнивается с жадной раскраской DSATUR (C₀, 0 баллов) и с эталоном
(долгий прогон табу-поиска; полный балл): балл = (C₀ − C)/(C₀ − C*), от 0 до 1. Неправильная раскраска — 0.""",
        "instances": [(250, 0.5), (500, 0.1)],
    },
    {
        "key": "strip", "name": "Упаковка в полосу", "tl": 10.0,
        "text": """Есть полоса ширины W и n прямоугольников w_i × h_i. Разместите все прямоугольники в полосе без
поворотов и без наложений (касаться сторонами можно) так, чтобы высота занятой части была как можно меньше.

**Ввод.** n W, затем n строк «w h» (целые; n = 150 или 250, W = 100).

**Вывод.** n строк «x y» — целые координаты левого нижнего угла каждого прямоугольника: 0 ≤ x, x + w ≤ W, y ≥ 0.

**Оценка.** Высота H = max(y + h) сравнивается с алгоритмом полок NFDH (H₀, 0 баллов) и с нижней границей
H* = ⌈Σ w_i·h_i / W⌉ (полный балл; обычно недостижима): балл = (H₀ − H)/(H₀ − H*), от 0 до 1.
Наложение или выход за полосу — 0.""",
        "instances": [(150, 100), (250, 100)],
    },
]


def make_instances(seed):
    out = []
    for k, p in enumerate(PROBLEMS):
        rng = random.Random(seed * 1000003 + 31 * k)
        if p["key"] == "tsp":
            out.append([gen_tsp(rng, n, kind) for kind, n in p["instances"]])
        elif p["key"] == "color":
            out.append([gen_color(rng, n, pr) for n, pr in p["instances"]])
        else:
            out.append([gen_strip(rng, n, W) for n, W in p["instances"]])
    return out


# ---------------------------------------------------------------------------
# Проверка и целевые функции
# ---------------------------------------------------------------------------

def _ints(text):
    return [int(t) for t in re.findall(r"-?\d+", text or "")]


def eval_tsp(inp, out):
    d = _ints(inp)
    n = d[0]
    pts = [(d[1 + 2 * i], d[2 + 2 * i]) for i in range(n)]
    perm = _ints(out)
    if sorted(perm) != list(range(n)):
        return None, "вывод — не перестановка городов"
    return sum(math.dist(pts[perm[i]], pts[perm[(i + 1) % n]]) for i in range(n)), ""


def eval_color(inp, out):
    d = _ints(inp)
    n, m = d[0], d[1]
    col = _ints(out)
    if len(col) != n or min(col) < 0:
        return None, "нужно n неотрицательных цветов"
    for e in range(m):
        u, v = d[2 + 2 * e], d[3 + 2 * e]
        if col[u] == col[v]:
            return None, f"ребро {u}–{v} одного цвета"
    return len(set(col)), ""


def eval_strip(inp, out):
    d = _ints(inp)
    n, W = d[0], d[1]
    R = [(d[2 + 2 * i], d[3 + 2 * i]) for i in range(n)]
    xy = _ints(out)
    if len(xy) != 2 * n:
        return None, "нужно n пар координат"
    boxes = []
    for i, (w, h) in enumerate(R):
        x, y = xy[2 * i], xy[2 * i + 1]
        if x < 0 or y < 0 or x + w > W:
            return None, f"прямоугольник {i} вне полосы"
        boxes.append((x, y, x + w, y + h))
    order = sorted(range(n), key=lambda i: boxes[i][0])
    for a in range(n):  # sweep по x
        i = order[a]
        for b in range(a + 1, n):
            j = order[b]
            if boxes[j][0] >= boxes[i][2]:
                break
            if boxes[i][1] < boxes[j][3] and boxes[j][1] < boxes[i][3]:
                return None, f"прямоугольники {i} и {j} перекрываются"
    return max(b[3] for b in boxes), ""


EVAL = {"tsp": eval_tsp, "color": eval_color, "strip": eval_strip}

# ---------------------------------------------------------------------------
# Базовые эвристики (0 баллов)
# ---------------------------------------------------------------------------


def base_tsp(inp):
    d = _ints(inp)
    n = d[0]
    pts = [(d[1 + 2 * i], d[2 + 2 * i]) for i in range(n)]
    left = set(range(1, n))
    cur, L = 0, 0.0
    while left:
        nx = min(left, key=lambda j: (pts[j][0] - pts[cur][0]) ** 2 + (pts[j][1] - pts[cur][1]) ** 2)
        L += math.dist(pts[cur], pts[nx])
        left.discard(nx)
        cur = nx
    return L + math.dist(pts[cur], pts[0])


def base_color(inp):
    d = _ints(inp)
    n, m = d[0], d[1]
    adj = [set() for _ in range(n)]
    for e in range(m):
        u, v = d[2 + 2 * e], d[3 + 2 * e]
        adj[u].add(v)
        adj[v].add(u)
    col = [-1] * n
    sat = [set() for _ in range(n)]
    for _ in range(n):
        v = max((u for u in range(n) if col[u] < 0), key=lambda u: (len(sat[u]), len(adj[u])))
        c = 0
        while c in sat[v]:
            c += 1
        col[v] = c
        for u in adj[v]:
            sat[u].add(c)
    return max(col) + 1


def base_strip(inp):
    """NFDH: прямоугольники по убыванию высоты, полки слева направо."""
    d = _ints(inp)
    n, W = d[0], d[1]
    R = sorted(((d[2 + 2 * i], d[3 + 2 * i]) for i in range(n)), key=lambda r: -r[1])
    y, x, shelf = 0, 0, 0
    for w, h in R:
        if x + w > W:
            y += shelf
            x, shelf = 0, 0
        if shelf == 0:
            shelf = h
        x += w
    return y + shelf


BASE = {"tsp": base_tsp, "color": base_color, "strip": base_strip}


def load_ref(seed):
    """Эталонные значения L*, C* (программы из opt_ref/) и нижние границы H* для упаковки."""
    path = os.path.join(DATA, f"opt_ref_{seed}.json")
    if not os.path.exists(path):
        return None
    with open(path, encoding="utf-8") as f:
        return json.load(f)


class OptTask(Task):
    key = "opt"
    cache_key = "opt"
    title = "Программирование"
    channel = 6
    ability = "coding"

    def generate(self, seed):
        insts = make_instances(seed)
        ref = load_ref(seed) or {}
        base = {p["key"]: [BASE[p["key"]](x) for x in insts[k]] for k, p in enumerate(PROBLEMS)}
        return {"seed": seed, "problems": [{"name": p["name"], "key": p["key"], "tl": p["tl"], "n_tests": len(p["instances"]),
                                            "base": base[p["key"]], "ref": ref.get(p["key"])} for p in PROBLEMS]}

    def n_items(self, inst):
        return len(PROBLEMS)

    def prompt(self, inst, only=None):
        idx = range(len(PROBLEMS)) if only is None else [only]
        L = ["# Оптимизационное программирование", "",
             "Для каждой задачи напиши программу на Python 3 (только стандартная библиотека). Программа читает экземпляр "
             "из stdin и пишет решение в stdout. Её запустят на скрытых экземплярах; ограничение времени указано у задачи "
             "(CPython на обычном сервере, одно ядро), память — 1 ГБ. Если программа не уложится во время, упадёт или выдаст "
             "некорректное решение, экземпляр даёт 0 баллов — следи за временем внутри программы. Точного ответа нет: "
             "балл растёт с качеством решения.", ""]
        for j, i in enumerate(idx, 1):
            p = PROBLEMS[i]
            L += [f"## Задача {j}. {p['name']} (ограничение времени {p['tl']:.0f} с)", "", p["text"], ""]
        L += ["## Формат ответа",
              "Для каждой задачи — один блок кода ```python, первая строка которого — комментарий `# задача N`. "
              "Пояснения вне блоков не нужны."]
        return "\n".join(L)

    def prompt_item(self, inst, k):
        return self.prompt(inst, only=k - 1).replace("# задача N", "# задача 1")

    merge_items = staticmethod(CodeTask.merge_items)

    def score(self, inst, text):
        insts = make_instances(inst["seed"])
        progs = extract_programs(text, len(PROBLEMS))
        items = []
        for k, p in enumerate(PROBLEMS):
            meta = inst["problems"][k]
            code = progs.get(k + 1)
            if not code:
                items.append({"ok": 0.0, "label": p["name"], "detail": "нет программы", "tests": [], "vals": []})
                continue
            scores, tests, vals, notes = [], [], [], []
            for t, inp in enumerate(insts[k]):
                st, out = run_code(code, inp, p["tl"])
                if st != "OK":
                    scores.append(0.0); tests.append(st); vals.append(None)
                    notes.append(st if st == "TL" else f"RE: {out[:60]}")
                    continue
                v, err = EVAL[p["key"]](inp, out)
                if v is None:
                    scores.append(0.0); tests.append("WA"); vals.append(None); notes.append(err)
                    continue
                b, r = meta["base"][t], (meta["ref"] or [None] * len(insts[k]))[t]
                s = max(0.0, min(1.0, (b - v) / (b - r))) if r is not None and b > r else (1.0 if v <= b else 0.0)
                scores.append(s); tests.append("OK"); vals.append(v)
                fmt = (lambda z: f"{z / 1e6:.3f}·10⁶") if p["key"] == "tsp" else (lambda z: f"{z:g}")
                notes.append(f"{fmt(v)} (база {fmt(b)}, эталон {fmt(r) if r is not None else '—'})")
            items.append({"ok": sum(scores) / len(scores), "label": p["name"], "detail": "; ".join(notes),
                          "tests": tests, "vals": vals, "parts": scores})
        return sum(it["ok"] for it in items) / len(items), items
