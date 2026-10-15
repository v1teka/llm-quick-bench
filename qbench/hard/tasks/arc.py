"""Канал «Логика»: головоломки в стиле ARC со скрытыми правилами.

По трём примерам «вход → выход» нужно понять правило и применить его к тестовому
входу. Ответ засчитывается только при точном совпадении всей сетки.
"""
from __future__ import annotations

import random

import numpy as np

from .common import Task, extract_json, get_item

COLORS = [1, 2, 3, 4, 6, 7, 8, 9]  # 0 — фон, 5 — служебный


def to_rows(g):
    return ["".join(str(int(v)) for v in row) for row in g]


def from_rows(rows):
    if not isinstance(rows, list) or not rows:
        return None
    try:
        if isinstance(rows[0], list):
            arr = [[int(v) for v in r] for r in rows]
        else:
            arr = [[int(ch) for ch in str(r).strip().replace(" ", "")] for r in rows]
    except (TypeError, ValueError):
        return None
    if len({len(r) for r in arr}) != 1:
        return None
    return np.array(arr, dtype=int)


def polyomino(rng, n, max_side=4):
    cells = {(0, 0)}
    while len(cells) < n:
        r, c = rng.choice(sorted(cells))
        dr, dc = rng.choice([(0, 1), (1, 0), (0, -1), (-1, 0)])
        cells.add((r + dr, c + dc))
        rs = [p[0] for p in cells]
        cs = [p[1] for p in cells]
        if max(rs) - min(rs) >= max_side or max(cs) - min(cs) >= max_side:
            cells.discard((r + dr, c + dc))
    mr, mc = min(p[0] for p in cells), min(p[1] for p in cells)
    return frozenset((r - mr, c - mc) for r, c in cells)


def canon(shape):
    """Каноническая форма с учётом поворотов и отражений."""
    best = None
    pts = list(shape)
    for flip in (False, True):
        for k in range(4):
            q = []
            for r, c in pts:
                if flip:
                    c = -c
                for _ in range(k):
                    r, c = c, -r
                q.append((r, c))
            mr, mc = min(p[0] for p in q), min(p[1] for p in q)
            t = tuple(sorted((r - mr, c - mc) for r, c in q))
            if best is None or t < best:
                best = t
    return best


def place_shapes(rng, H, W, shapes, margin=1, gap=1, occupied=None, tries=400):
    occ = set(occupied or ())
    out = []
    for sh in shapes:
        h = max(r for r, _ in sh) + 1
        w = max(c for _, c in sh) + 1
        for _ in range(tries):
            r0 = rng.randint(margin, H - h - margin)
            c0 = rng.randint(margin, W - w - margin)
            cells = {(r0 + r, c0 + c) for r, c in sh}
            halo = {(r + dr, c + dc) for r, c in cells for dr in range(-gap, gap + 1) for dc in range(-gap, gap + 1)}
            if not (halo & occ):
                occ |= cells
                out.append(cells)
                break
        else:
            return None
    return out


# ---------------------------------------------------------------------------
# правила
# ---------------------------------------------------------------------------

def r_gravity(rng, P):
    H, W = rng.randint(P["min"], P["max"]), rng.randint(P["min"], P["max"])
    side = rng.choice("NESW")
    g = np.zeros((H, W), int)
    floor = set()
    if side == "S":
        floor = {(H - 1, c) for c in range(W)}
    elif side == "N":
        floor = {(0, c) for c in range(W)}
    elif side == "W":
        floor = {(r, 0) for r in range(H)}
    else:
        floor = {(r, W - 1) for r in range(H)}
    for r, c in floor:
        g[r, c] = 5
    n = rng.randint(P["nobj"][0], P["nobj"][1])
    shapes = [polyomino(rng, rng.randint(2, 5), 3) for _ in range(n)]
    halo_floor = {(r + dr, c + dc) for r, c in floor for dr in (-1, 0, 1) for dc in (-1, 0, 1)}
    objs = place_shapes(rng, H, W, shapes, margin=1, gap=1, occupied=halo_floor)
    if objs is None:
        return None
    cols = rng.sample(COLORS, n)
    d = {"S": (1, 0), "N": (-1, 0), "E": (0, 1), "W": (0, -1)}[side]

    def dist(cells):
        if side == "S":
            return min(H - 1 - r for r, _ in cells)
        if side == "N":
            return min(r for r, _ in cells)
        if side == "E":
            return min(W - 1 - c for _, c in cells)
        return min(c for _, c in cells)
    order = sorted(range(n), key=lambda i: dist(objs[i]))
    occ = set(floor)
    out = g.copy()
    moved = 0
    for i in order:
        cells = objs[i]
        while True:
            nxt = {(r + d[0], c + d[1]) for r, c in cells}
            if any(not (0 <= r < H and 0 <= c < W) or (r, c) in occ for r, c in nxt):
                break
            cells = nxt
            moved += 1
        occ |= cells
        for r, c in cells:
            out[r, c] = cols[i]
    for i in range(n):
        for r, c in objs[i]:
            g[r, c] = cols[i]
    if moved < n:
        return None
    return g, out


def r_holes(rng, P):
    pal = P["pal"]
    for _ in range(50):
        H, W = rng.randint(P["min"], P["max"]), rng.randint(P["min"], P["max"])
        n = rng.randint(3, 4)
        shapes, ks = [], []
        for _ in range(n):
            h, w = rng.randint(3, 5), rng.randint(4, 6)
            k = rng.choice([0, 1, 2, 3])
            inner = [(r, c) for r in range(1, h - 1) for c in range(1, w - 1)]
            rng.shuffle(inner)
            holes = []
            for p in inner:
                if len(holes) == k:
                    break
                if all(max(abs(p[0] - q[0]), abs(p[1] - q[1])) > 1 for q in holes):
                    holes.append(p)
            if len(holes) < k:
                k = len(holes)
            cells = frozenset((r, c) for r in range(h) for c in range(w) if (r, c) not in holes)
            shapes.append((h, w, cells))
            ks.append(k)
        rects = []
        occ = set()
        ok = True
        for h, w, cells in shapes:
            for _ in range(300):
                r0, c0 = rng.randint(0, H - h), rng.randint(0, W - w)
                box = {(r0 + r, c0 + c) for r in range(-1, h + 1) for c in range(-1, w + 1)}
                if not (box & occ):
                    occ |= {(r0 + r, c0 + c) for r in range(h) for c in range(w)}
                    rects.append((r0, c0, cells))
                    break
            else:
                ok = False
                break
        if not ok:
            continue
        g = np.zeros((H, W), int)
        out = g.copy()
        for (r0, c0, cells), k in zip(rects, ks):
            for r, c in cells:
                g[r0 + r, c0 + c] = 1
                out[r0 + r, c0 + c] = pal[k]
        return g, out, set(ks)
    return None


def r_outline(rng, P):
    for _ in range(80):
        H, W = rng.randint(P["min"], P["max"]), rng.randint(P["min"], P["max"])
        n = rng.randint(P["nobj"][0], P["nobj"][1])
        shapes = [polyomino(rng, rng.randint(3, 6), 4) for _ in range(n)]
        objs = place_shapes(rng, H, W, shapes, margin=2, gap=1)
        if objs is None:
            continue
        cols = [rng.choice([c for c in COLORS if c not in (8, 9)]) for _ in range(n)]
        g = np.zeros((H, W), int)
        for cells, col in zip(objs, cols):
            for r, c in cells:
                g[r, c] = col
        cnt = np.zeros((H, W), int)
        for cells in objs:
            r0 = min(r for r, _ in cells) - 1
            r1 = max(r for r, _ in cells) + 1
            c0 = min(c for _, c in cells) - 1
            c1 = max(c for _, c in cells) + 1
            ring = {(r, c) for r in range(r0, r1 + 1) for c in range(c0, c1 + 1) if r in (r0, r1) or c in (c0, c1)}
            for r, c in ring:
                if 0 <= r < H and 0 <= c < W:
                    cnt[r, c] += 1
        out = g.copy()
        free = g == 0
        out[free & (cnt == 1)] = 8
        out[free & (cnt >= 2)] = 9
        if not (free & (cnt >= 2)).any():
            continue
        return g, out
    return None


def r_majority(rng, P):
    S = rng.randint(P["min"], P["max"])
    cols = rng.sample(COLORS, 3)
    base = np.array([[rng.choice([0, 0] + cols) for _ in range(S)] for _ in range(S)])
    cells = [(r, c) for r in range(S) for c in range(S)]
    rng.shuffle(cells)
    k = P["noise"]
    copies = []
    for i in range(3):
        cp = base.copy()
        for r, c in cells[i * k:(i + 1) * k]:
            cp[r, c] = rng.choice([v for v in [0] + COLORS if v != base[r, c]])
        copies.append(cp)
    sep = np.full((S, 1), 5)
    g = np.concatenate([copies[0], sep, copies[1], sep, copies[2]], 1)
    return g, base


def r_lrays(rng, P):
    for _ in range(100):
        H, W = rng.randint(P["min"], P["max"]), rng.randint(P["min"], P["max"])
        n = rng.randint(2, 4)
        rows = sorted(rng.sample(range(0, H - 3), n))
        if any(b - a < 2 for a, b in zip(rows, rows[1:])):
            continue
        wcols = sorted(rng.sample(range(3, W - 1), n), reverse=True)
        cols = rng.sample(COLORS, n)
        g = np.zeros((H, W), int)
        for r, wc, col in zip(rows, wcols, cols):
            g[r, 0] = col
            g[r, wc] = 5
        for _ in range(rng.randint(2, 5)):
            r, c = rng.randint(0, H - 1), rng.randint(2, W - 1)
            if g[r, c] == 0 and (r not in rows):
                g[r, c] = 5
        out = g.copy()
        good = True
        for r, wc, col in zip(rows, wcols, cols):
            c = 1
            while c < W and g[r, c] == 0:
                c += 1
            if c >= W or g[r, c] != 5:
                good = False
                break
            for cc in range(1, c):
                out[r, cc] = col
            cc = c - 1
            rr = r + 1
            while rr < H and g[rr, cc] == 0 and out[rr, cc] == 0:
                out[rr, cc] = col
                rr += 1
        if good:
            return g, out
    return None


def r_bars(rng, P):
    H, W = rng.randint(P["min"], P["max"]), rng.randint(P["min"], P["max"])
    cols = rng.sample(COLORS, 3)
    counts = rng.sample(range(2, P["maxc"] + 1), 3)
    g = np.zeros((H, W), int)
    cells = [(r, c) for r in range(H) for c in range(W)]
    rng.shuffle(cells)
    k = 0
    for col, n in zip(cols, counts):
        for _ in range(n):
            r, c = cells[k]
            g[r, c] = col
            k += 1
    order = sorted(range(3), key=lambda i: -counts[i])
    hmax = max(counts)
    out = np.zeros((hmax, 3), int)
    for j, i in enumerate(order):
        for t in range(counts[i]):
            out[hmax - 1 - t, j] = cols[i]
    return g, out


def r_symmetry(rng, P):
    N = rng.choice(P["sizes"])
    h2 = N // 2
    pal = rng.sample([1, 2, 3, 4, 6, 7, 8], 4)
    q = np.array([[rng.choice([0] + pal) for _ in range(h2)] for _ in range(h2)])
    top = np.concatenate([q, q[:, ::-1]], 1)
    full = np.concatenate([top, top[::-1, :]], 0)
    mh, mw = rng.randint(2, 3), rng.randint(3, 4)
    r0 = rng.choice([rng.randint(0, h2 - mh), rng.randint(h2, N - mh)])
    c0 = rng.choice([rng.randint(0, h2 - mw), rng.randint(h2, N - mw)])
    g = full.copy()
    g[r0:r0 + mh, c0:c0 + mw] = 9
    return g, full[r0:r0 + mh, c0:c0 + mw].copy()


def r_odd(rng, P):
    for _ in range(100):
        H, W = rng.randint(P["min"], P["max"]), rng.randint(P["min"], P["max"])
        size = rng.randint(4, 6)
        base = polyomino(rng, size, 3)
        odd = polyomino(rng, size, 3)
        if canon(base) == canon(odd):
            continue
        n = rng.randint(4, 6)
        shapes = [base] * (n - 1) + [odd]
        idx = list(range(n))
        rng.shuffle(idx)
        shapes = [shapes[i] for i in idx]
        objs = place_shapes(rng, H, W, shapes, margin=0, gap=1)
        if objs is None:
            continue
        g = np.zeros((H, W), int)
        cols = [rng.choice(COLORS) for _ in range(n)]
        out = None
        for cells, col, i in zip(objs, cols, idx):
            for r, c in cells:
                g[r, c] = col
            if i == n - 1:
                r0, c0 = min(r for r, _ in cells), min(c for _, c in cells)
                r1, c1 = max(r for r, _ in cells), max(c for _, c in cells)
                out = np.zeros((r1 - r0 + 1, c1 - c0 + 1), int)
                for r, c in cells:
                    out[r - r0, c - c0] = col
        return g, out
    return None


RULES = [
    ("Гравитация к стене", r_gravity, lambda rng, d: {"min": 10 + d, "max": 12 + d, "nobj": (3, 4 + d)}),
    ("Цвет по числу дыр", r_holes, lambda rng, d: {"min": 12 + d, "max": 14 + d, "pal": dict(zip(range(4), rng.sample([2, 3, 4, 6, 7, 8], 4)))}),
    ("Рамки и пересечения", r_outline, lambda rng, d: {"min": 11 + d, "max": 13 + d, "nobj": (3, 4 + d)}),
    ("Голосование копий", r_majority, lambda rng, d: {"min": 5 + d // 2, "max": 6 + d // 2, "noise": 2 + d // 2}),
    ("Лучи с поворотом", r_lrays, lambda rng, d: {"min": 10 + d, "max": 12 + d}),
    ("Гистограмма цветов", r_bars, lambda rng, d: {"min": 10 + d, "max": 12 + d, "maxc": 7 + d}),
    ("Симметрия", r_symmetry, lambda rng, d: {"sizes": [12, 14] if d < 2 else [14, 16]}),
    ("Лишняя фигура", r_odd, lambda rng, d: {"min": 11 + d, "max": 13 + d}),
]


def make_puzzle(rng, rule, params_fn, difficulty):
    for _ in range(200):
        P = params_fn(rng, difficulty)
        pairs = []
        for _ in range(4):
            res = None
            for _ in range(200):
                res = rule(rng, P)
                if res is not None:
                    break
            if res is None:
                break
            pairs.append(res)
        if len(pairs) < 4:
            continue
        if rule is r_holes:
            train_k = set().union(*[p[2] for p in pairs[:3]])
            if not pairs[3][2] <= train_k:
                continue
            pairs = [(a, b) for a, b, _ in pairs]
        if any(np.array_equal(pairs[i][0], pairs[3][0]) for i in range(3)):
            continue
        return pairs
    raise RuntimeError("не удалось сгенерировать головоломку")


class ArcTask(Task):
    key = "arc"
    title = "Логика"
    channel = 1
    ability = "logic"

    def __init__(self, difficulty=1):
        self.difficulty = difficulty

    def generate(self, seed):
        rng = random.Random(seed * 7919 + 11)
        puzzles = []
        rules = HARDER_RULES if self.difficulty >= 3 else (HARD_RULES if self.difficulty >= 2 else RULES)
        for name, rule, pfn in rules:
            pairs = make_puzzle_hard(rng, rule, pfn) if self.difficulty >= 2 else make_puzzle(rng, rule, pfn, self.difficulty)
            puzzles.append({
                "name": name,
                "train": [{"input": to_rows(a), "output": to_rows(b)} for a, b in pairs[:3]],
                "test": {"input": to_rows(pairs[3][0])},
                "answer": to_rows(pairs[3][1]),
            })
        return {"seed": seed, "difficulty": self.difficulty, "puzzles": puzzles}

    def prompt(self, inst):
        L = [
            "# Головоломки на абстрактное мышление",
            "",
            "В каждой задаче есть скрытое правило, которое превращает входную сетку в выходную. "
            "Даны три примера «вход → выход» и тестовый вход. Найди правило и примени его к тестовому входу.",
            "Сетки записаны построчно, каждая цифра — цвет клетки (0 — фон). Размер выходной сетки тоже нужно определить самому.",
            "",
        ]
        for i, p in enumerate(inst["puzzles"], 1):
            L.append(f"## Задача {i}")
            for j, ex in enumerate(p["train"], 1):
                L.append(f"Пример {j}, вход:")
                L += ex["input"]
                L.append(f"Пример {j}, выход:")
                L += ex["output"]
                L.append("")
            L.append("Тест, вход:")
            L += p["test"]["input"]
            L.append("")
        L += [
            "## Формат ответа",
            "Ответь ОДНИМ блоком JSON без пояснений вне него: ключ — номер задачи, значение — список строк выходной сетки.",
            '{"1": ["0120", "0000"], "2": [...], ...}',
        ]
        return "\n".join(L)

    def score(self, inst, text):
        ans = extract_json(text)
        items = []
        for i, p in enumerate(inst["puzzles"], 1):
            got = from_rows(get_item(ans, i)) if ans else None
            want = from_rows(p["answer"])
            ok = got is not None and got.shape == want.shape and np.array_equal(got, want)
            if got is None:
                det = "нет ответа"
            elif got.shape != want.shape:
                det = f"размер {got.shape[0]}×{got.shape[1]}, нужно {want.shape[0]}×{want.shape[1]}"
            else:
                det = "верно" if ok else f"ошибок в клетках: {int((got != want).sum())}"
            items.append({"ok": 1.0 if ok else 0.0, "label": p["name"], "detail": det,
                          "got": to_rows(got) if got is not None else None})
        return sum(it["ok"] for it in items) / len(items), items


# ===========================================================================
# Жёсткий набор (уровень ARC-AGI-2): контекстные правила, композиции, счёт
# ===========================================================================

def _rot_shape(shape, k, flip):
    pts = []
    for r, c in shape:
        if flip:
            c = -c
        for _ in range(k):
            r, c = c, -r
        pts.append((r, c))
    mr, mc = min(p[0] for p in pts), min(p[1] for p in pts)
    return frozenset((r - mr, c - mc) for r, c in pts)


def h_legend(rng, P):
    """Легенда в рамке задаёт цвета фигур; серые (8) фигуры перекрашиваются по совпадению формы
    с учётом поворотов/отражений; фигуры без пары в легенде исчезают."""
    for _ in range(200):
        H, W = rng.randint(P["min"], P["max"]), rng.randint(P["min"], P["max"])
        shapes = []
        while len(shapes) < 3:
            s = polyomino(rng, rng.randint(4, 5), 3)
            if all(canon(s) != canon(t) for t in shapes):
                shapes.append(s)
        cols = rng.sample([1, 2, 3, 4, 6, 7, 9], 3)
        g = np.zeros((H, W), int)
        # легенда: фигуры в столбик в левом верхнем углу, рамка 5
        r = 1
        occ = set()
        for s, col in zip(shapes, cols):
            for (a, b) in s:
                g[r + a, 1 + b] = col
            r += max(a for a, _ in s) + 2
        lh = r
        lw = 1 + max(max(b for _, b in s) for s in shapes) + 2
        if lh >= H - 4 or lw >= W - 5:
            continue
        g[lh, :lw + 1] = 5
        g[:lh + 1, lw] = 5
        occ = {(rr, cc) for rr in range(lh + 2) for cc in range(lw + 2)}
        n = rng.randint(P["nobj"][0], P["nobj"][1])
        objs, kinds = [], []
        for _ in range(n):
            if rng.random() < 0.25:
                while True:
                    d = polyomino(rng, rng.randint(4, 5), 3)
                    if all(canon(d) != canon(t) for t in shapes):
                        break
                objs.append(d)
                kinds.append(-1)
            else:
                k = rng.randrange(3)
                objs.append(_rot_shape(shapes[k], rng.randrange(4), rng.random() < 0.5))
                kinds.append(k)
        placed = place_shapes(rng, H, W, objs, margin=0, gap=1, occupied=occ)
        if placed is None or -1 not in kinds:
            continue
        out = g.copy()
        for cells, k in zip(placed, kinds):
            for rr, cc in cells:
                g[rr, cc] = 8
                out[rr, cc] = cols[k] if k >= 0 else 0
        return g, out
    return None


def h_marker_slide(rng, P):
    """Серые фигуры с одной цветной клеткой-маркером скользят в направлении, которое задаёт цвет маркера,
    до стены (5), края или другой фигуры."""
    dirs = P["dirs"]  # цвет -> (dr, dc)
    for _ in range(300):
        H, W = rng.randint(P["min"], P["max"]), rng.randint(P["min"], P["max"])
        g = np.zeros((H, W), int)
        for _ in range(rng.randint(2, 5)):
            g[rng.randrange(H), rng.randrange(W)] = 5
        n = rng.randint(*P.get("nobj", (3, 5)))
        shapes = [polyomino(rng, rng.randint(3, 5), 3) for _ in range(n)]
        occ0 = {(r, c) for r in range(H) for c in range(W) if g[r, c] == 5}
        halo = {(r + dr, c + dc) for r, c in occ0 for dr in (-1, 0, 1) for dc in (-1, 0, 1)}
        placed = place_shapes(rng, H, W, shapes, margin=1, gap=1, occupied=halo)
        if placed is None:
            continue
        markers = [rng.choice(sorted(dirs)) for _ in range(n)]
        mcell = [rng.choice(sorted(cells)) for cells in placed]
        # каждая фигура скользит независимо (стены и край); пути фигур не должны пересекаться —
        # тогда результат не зависит от порядка движения
        finals, sweeps = [], []
        blocked = set(occ0)
        ok = True
        for i, cells in enumerate(placed):
            dr, dc = dirs[markers[i]]
            cur = set(cells)
            sweep = set(cur)
            while True:
                nxt = {(r + dr, c + dc) for r, c in cur}
                if any(not (0 <= r < H and 0 <= c < W) or (r, c) in blocked for r, c in nxt):
                    break
                cur = nxt
                sweep |= cur
            finals.append(cur)
            sweeps.append(sweep)
        for i in range(n):
            for j in range(n):
                if i != j:
                    halo = {(r + a, c + b) for r, c in sweeps[j] for a in (-1, 0, 1) for b in (-1, 0, 1)}
                    if sweeps[i] & halo:
                        ok = False
        if not ok or sum(1 for i in range(n) if finals[i] != placed[i]) < n - 1:
            continue
        out = g.copy()
        for i, cells in enumerate(placed):
            off = None
            for r, c in cells:
                g[r, c] = 8
            g[mcell[i]] = markers[i]
            # смещение
            r0 = min(r for r, _ in cells)
            c0 = min(c for _, c in cells)
            r1 = min(r for r, _ in finals[i])
            c1 = min(c for _, c in finals[i])
            off = (r1 - r0, c1 - c0)
            for r, c in cells:
                out[r + off[0], c + off[1]] = 8
            out[mcell[i][0] + off[0], mcell[i][1] + off[1]] = markers[i]
        return g, out, set(markers)
    return None


def h_snakes(rng, P):
    """Голова (2) и шея (3) задают направление; змейка ползёт, оставляя след 2;
    перед клеткой 6 поворачивает направо, перед 7 — налево, перед 5 или краем останавливается."""
    R = {(0, 1): (1, 0), (1, 0): (0, -1), (0, -1): (-1, 0), (-1, 0): (0, 1)}
    Lt = {v: k for k, v in R.items()}
    for _ in range(500):
        H, W = rng.randint(P["min"], P["max"]), rng.randint(P["min"], P["max"])
        g = np.zeros((H, W), int)
        out = g.copy()
        used = set()
        snakes = 0
        for _ in range(rng.randint(*P.get("nsnakes", (2, 3)))):
            for _try in range(60):
                r, c = rng.randrange(1, H - 1), rng.randrange(1, W - 1)
                d = rng.choice(list(R))
                nr, nc = r - d[0], c - d[1]
                if not (0 <= nr < H and 0 <= nc < W):
                    continue
                cells = [(nr, nc), (r, c)]
                if any(p in used for p in cells):
                    continue
                # прокладываем путь с 1–3 поворотами
                path, marks = [(r, c)], []
                turns = rng.randint(1, P["turns"])
                good = True
                cur, cd = (r, c), d
                for t in range(turns + 1):
                    L = rng.randint(2, 5)
                    for _ in range(L):
                        q = (cur[0] + cd[0], cur[1] + cd[1])
                        if not (0 <= q[0] < H and 0 <= q[1] < W) or q in used or q in path:
                            good = False
                            break
                        path.append(q)
                        cur = q
                    if not good:
                        break
                    q = (cur[0] + cd[0], cur[1] + cd[1])
                    if t < turns:
                        right = rng.random() < 0.5
                        if not (0 <= q[0] < H and 0 <= q[1] < W) or q in used or q in path:
                            good = False
                            break
                        marks.append((q, 6 if right else 7))
                        cd = R[cd] if right else Lt[cd]
                    else:
                        if 0 <= q[0] < H and 0 <= q[1] < W:
                            if q in used or q in path:
                                good = False
                                break
                            marks.append((q, 5))
                if not good:
                    continue
                halo = {(a + x, b + y) for a, b in path + [m for m, _ in marks] + cells for x in (-1, 0, 1) for y in (-1, 0, 1)}
                if halo & used:
                    continue
                used |= halo
                g[nr, nc] = 3
                g[r, c] = 2
                out[nr, nc] = 3
                for (a, b) in path:
                    out[a, b] = 2
                for (q, col) in marks:
                    g[q] = col
                    out[q] = col
                snakes += 1
                break
        if snakes >= P.get("nsnakes", (2, 3))[0]:
            return g, out
    return None


def h_count_objects(rng, P):
    """Выход: по строке на цвет, длина строки = число отдельных фигур этого цвета (а не клеток);
    строки упорядочены по убыванию."""
    for _ in range(200):
        H, W = rng.randint(P["min"], P["max"]), rng.randint(P["min"], P["max"])
        ncol = rng.randint(*P.get("ncol", (3, 4)))
        cols = rng.sample(COLORS, ncol)
        counts = rng.sample(range(1, P["maxk"] + 1), ncol)
        shapes, owner = [], []
        for col, k in zip(cols, counts):
            for _ in range(k):
                shapes.append(polyomino(rng, rng.randint(1, 4), 2))
                owner.append(col)
        placed = place_shapes(rng, H, W, shapes, margin=0, gap=1)
        if placed is None:
            continue
        g = np.zeros((H, W), int)
        for cells, col in zip(placed, owner):
            for r, c in cells:
                g[r, c] = col
        order = sorted(range(ncol), key=lambda i: -counts[i])
        out = np.zeros((ncol, max(counts)), int)
        for row, i in enumerate(order):
            out[row, :counts[i]] = cols[i]
        return g, out
    return None


def h_cups(rng, P):
    """Чаши из 5 с разной высотой стенок; источник 1 над чашей. Вода падает столбом и заполняет
    чашу до уровня НИЖНЕЙ из двух стенок."""
    for _ in range(300):
        H, W = rng.randint(P["min"], P["max"]), rng.randint(P["min"], P["max"])
        g = np.zeros((H, W), int)
        out = g.copy()
        x = 0
        n = 0
        while True:
            iw = rng.randint(2, 4)
            hl, hr = rng.randint(2, 5), rng.randint(2, 5)
            x0 = x + rng.randint(0, 2)
            if x0 + iw + 2 > W:
                break
            base = H - 1 - rng.randint(0, 2)
            if base - max(hl, hr) < 4:
                break
            # дно
            for cc in range(x0, x0 + iw + 2):
                g[base, cc] = 5
            for t in range(1, hl + 1):
                g[base - t, x0] = 5
            for t in range(1, hr + 1):
                g[base - t, x0 + iw + 1] = 5
            lvl = min(hl, hr)
            src_c = x0 + 1 + rng.randrange(iw)
            src_r = rng.randint(0, base - max(hl, hr) - 2)
            g[src_r, src_c] = 1
            n += 1
            out[:] = np.where(g == 0, out, g)
            for t in range(1, lvl + 1):
                for cc in range(x0 + 1, x0 + 1 + iw):
                    out[base - t, cc] = 1
            for rr in range(src_r, base - lvl):
                out[rr, src_c] = 1
            x = x0 + iw + 2 + 1
        if n >= 2:
            out = np.where(g > 0, g, out)
            return g, out
    return None


def h_checker_tiling(rng, P):
    """Узор-плитка повторяется в шахматном порядке: на «чёрных» местах — зеркальная копия.
    Часть картины закрыта цветом 0-квадрата… нет: закрыта цветом 9; нужно восстановить всю картину."""
    t = P["tile"]
    pal = rng.sample([1, 2, 3, 4, 6, 7, 8], 3)
    while True:
        M = np.array([[rng.choice([0] + pal) for _ in range(t)] for _ in range(t)])
        if not np.array_equal(M, M[:, ::-1]):
            break
    n = P["n"]
    full = np.zeros((n * t, n * t), int)
    for i in range(n):
        for j in range(n):
            full[i * t:(i + 1) * t, j * t:(j + 1) * t] = M if (i + j) % 2 == 0 else M[:, ::-1]
    g = full.copy()
    hh, ww = rng.randint(t, t + P.get("mask", 2)), rng.randint(t, t + P.get("mask", 2) + 1)
    r0, c0 = rng.randint(0, n * t - hh), rng.randint(0, n * t - ww)
    g[r0:r0 + hh, c0:c0 + ww] = 9
    return g, full


def h_xor_mirror(rng, P):
    """Две половины (1 и 2) через столбец 5. Выход: 6 там, где закрашена ровно одна из клеток:
    левой половины и ЗЕРКАЛЬНО отражённой правой."""
    S = rng.randint(P["min"], P["max"])
    A = (np.array([[rng.random() < 0.45 for _ in range(S)] for _ in range(S)]))
    B = (np.array([[rng.random() < 0.45 for _ in range(S)] for _ in range(S)]))
    g = np.concatenate([A * 1, np.full((S, 1), 5), B * 2], 1)
    out = np.where(A ^ B[:, ::-1], 6, 0)
    return g, out


def h_voronoi(rng, P):
    """Каждая клетка 8 получает цвет ближайшего «семени» (манхэттенское расстояние);
    при равенстве расстояний до двух разных цветов остаётся 8."""
    H, W = rng.randint(P["min"], P["max"]), rng.randint(P["min"], P["max"])
    g = np.full((H, W), 8)
    k = rng.randint(3, P["seeds"])
    cols = rng.sample([1, 2, 3, 4, 6, 7], k)
    seeds = []
    cells = [(r, c) for r in range(H) for c in range(W)]
    rng.shuffle(cells)
    for col in cols:
        r, c = cells.pop()
        seeds.append((r, c, col))
        g[r, c] = col
    out = g.copy()
    for r in range(H):
        for c in range(W):
            if g[r, c] != 8:
                continue
            d = sorted((abs(r - a) + abs(c - b), col) for a, b, col in seeds)
            if len(d) > 1 and d[0][0] == d[1][0] and d[0][1] != d[1][1]:
                continue
            out[r, c] = d[0][1]
    return g, out


HARD_RULES = [
    ("Легенда форм", h_legend, lambda rng, d: {"min": 16, "max": 19, "nobj": (6, 8)}),
    ("Маркеры направления", h_marker_slide,
     lambda rng, d: {"min": 14, "max": 17, "dirs": dict(zip([1, 2, 3, 4], rng.sample([(-1, 0), (0, 1), (1, 0), (0, -1)], 4)))}),
    ("Змейки и повороты", h_snakes, lambda rng, d: {"min": 14, "max": 17, "turns": 3}),
    ("Счёт фигур", h_count_objects, lambda rng, d: {"min": 13, "max": 16, "maxk": 6}),
    ("Вода в чашах", h_cups, lambda rng, d: {"min": 12, "max": 16}),
    ("Шахматная плитка", h_checker_tiling, lambda rng, d: {"tile": 3, "n": 5}),
    ("Зеркальный XOR", h_xor_mirror, lambda rng, d: {"min": 7, "max": 8}),
    ("Ближайшее семя", h_voronoi, lambda rng, d: {"min": 11, "max": 13, "seeds": 4}),
]


HARDER_RULES = [
    ("Легенда форм", h_legend, lambda rng, d: {"min": 22, "max": 25, "nobj": (9, 12)}),
    ("Маркеры направления", h_marker_slide,
     lambda rng, d: {"min": 18, "max": 22, "nobj": (5, 7), "dirs": dict(zip([1, 2, 3, 4], rng.sample([(-1, 0), (0, 1), (1, 0), (0, -1)], 4)))}),
    ("Змейки и повороты", h_snakes, lambda rng, d: {"min": 18, "max": 21, "turns": 4, "nsnakes": (3, 4)}),
    ("Счёт фигур", h_count_objects, lambda rng, d: {"min": 17, "max": 20, "maxk": 8, "ncol": (4, 5)}),
    ("Вода в чашах", h_cups, lambda rng, d: {"min": 16, "max": 21}),
    ("Шахматная плитка", h_checker_tiling, lambda rng, d: {"tile": 4, "n": 6, "mask": 3}),
    ("Зеркальный XOR", h_xor_mirror, lambda rng, d: {"min": 10, "max": 12}),
    ("Ближайшее семя", h_voronoi, lambda rng, d: {"min": 17, "max": 20, "seeds": 6}),
]


def make_puzzle_hard(rng, rule, params_fn):
    for _ in range(300):
        P = params_fn(rng, 2)
        pairs = []
        for _ in range(4):
            res = None
            for _ in range(100):
                res = rule(rng, P)
                if res is not None:
                    break
            if res is None:
                break
            pairs.append(res)
        if len(pairs) < 4:
            continue
        if rule is h_marker_slide:
            seen = set().union(*[p[2] for p in pairs[:3]])
            if not pairs[3][2] <= seen:
                continue
            pairs = [(a, b) for a, b, _ in pairs]
        if any(np.array_equal(pairs[i][0], pairs[3][0]) for i in range(3)):
            continue
        return pairs
    raise RuntimeError(f"не удалось сгенерировать {rule.__name__}")
