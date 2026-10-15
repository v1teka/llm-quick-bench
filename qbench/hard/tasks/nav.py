"""Канал «Навигация»: планирование в клеточном мире (лёд, ключи и двери, порталы, сокобан)."""
from __future__ import annotations

import random
from collections import deque

from .common import Task, extract_json, get_item

MOVES = {"U": (-1, 0), "D": (1, 0), "L": (0, -1), "R": (0, 1)}
ARROW = {">": "R", "<": "L", "^": "U", "v": "D"}


class Level:
    def __init__(self, rows):
        self.rows = rows
        self.H, self.W = len(rows), len(rows[0])
        self.portals = {}
        self.boxes0 = []
        self.targets = set()
        pos = {}
        for r, row in enumerate(rows):
            for c, ch in enumerate(row):
                if ch == "S":
                    self.start = (r, c)
                elif ch == "G":
                    self.goal = (r, c)
                elif ch.isdigit():
                    pos.setdefault(ch, []).append((r, c))
                elif ch == "$":
                    self.boxes0.append((r, c))
                elif ch == "x":
                    self.targets.add((r, c))
                elif ch == "%":  # ящик сразу на цели
                    self.boxes0.append((r, c))
                    self.targets.add((r, c))
        for d, ps in pos.items():
            if len(ps) == 2:
                self.portals[ps[0]] = ps[1]
                self.portals[ps[1]] = ps[0]
        self.sokoban = bool(self.boxes0)
        if not hasattr(self, "goal"):
            self.goal = None

    def ch(self, r, c):
        if not (0 <= r < self.H and 0 <= c < self.W):
            return "#"
        ch = self.rows[r][c]
        if ch in "S$x%G":
            return "." if ch != "G" else "G"
        return ch

    def can_enter(self, r, c, mv, keys):
        ch = self.ch(r, c)
        if ch == "#":
            return False
        if ch in "ABC":
            return ch.lower() in keys
        if ch in ARROW:
            return ARROW[ch] == mv
        return True

    def step(self, state, mv):
        """-> новое состояние или None (ход невозможен)."""
        (r, c), keys, boxes = state
        dr, dc = MOVES[mv]
        nr, nc = r + dr, c + dc
        boxes = set(boxes)
        if (nr, nc) in boxes:
            br, bc = nr + dr, nc + dc
            if not self.can_enter(br, bc, mv, keys) or (br, bc) in boxes or self.ch(br, bc) == "~":
                return None
            boxes.discard((nr, nc))
            boxes.add((br, bc))
        elif not self.can_enter(nr, nc, mv, keys):
            return None
        r, c = nr, nc
        guard = 0
        while self.ch(r, c) == "~" and guard < 200:
            guard += 1
            tr, tc = r + dr, c + dc
            if not self.can_enter(tr, tc, mv, keys) or (tr, tc) in boxes:
                break
            r, c = tr, tc
        ch = self.ch(r, c)
        if ch in "abc":
            keys = frozenset(keys | {ch})
        if (r, c) in self.portals:
            r, c = self.portals[(r, c)]
        return ((r, c), keys, frozenset(boxes))

    def init_state(self):
        return (self.start, frozenset(), frozenset(self.boxes0))

    def solved(self, state):
        if self.sokoban:
            return set(state[2]) == self.targets
        return state[0] == self.goal

    def bfs(self, limit=400000):
        s0 = self.init_state()
        prev = {s0: None}
        q = deque([s0])
        while q:
            s = q.popleft()
            if self.solved(s):
                path = []
                while prev[s] is not None:
                    s, mv = prev[s]
                    path.append(mv)
                return "".join(reversed(path))
            for mv in "UDLR":
                t = self.step(s, mv)
                if t is not None and t not in prev:
                    prev[t] = (s, mv)
                    q.append(t)
                    if len(prev) > limit:
                        return None
        return None

    def simulate(self, moves):
        """-> (solved, n_valid_moves, trace list[(r,c)], error str)"""
        s = self.init_state()
        trace = [s[0]]
        for i, mv in enumerate(moves):
            t = self.step(s, mv)
            if t is None:
                return False, i, trace, f"ход {i + 1} ({mv}) невозможен"
            s = t
            trace.append(s[0])
        return self.solved(s), len(moves), trace, "" if self.solved(s) else "цель не достигнута"


# ---------------------------------------------------------------------------
# генерация уровней
# ---------------------------------------------------------------------------

def _blank(H, W):
    g = [["." for _ in range(W)] for _ in range(H)]
    for r in range(H):
        g[r][0] = g[r][W - 1] = "#"
    for c in range(W):
        g[0][c] = g[H - 1][c] = "#"
    return g


def _free(g):
    return [(r, c) for r in range(1, len(g) - 1) for c in range(1, len(g[0]) - 1) if g[r][c] == "."]


def gen_ice(rng, H, W, ice_frac, wall_frac, min_len, extras="", samples=2500):
    """Лучший из N случайных уровней: с самым длинным оптимальным решением."""
    best = None
    for _ in range(samples):
        g = _blank(H, W)
        cells = _free(g)
        rng.shuffle(cells)
        n_ice = int(len(cells) * ice_frac)
        n_wall = int(len(cells) * wall_frac)
        for r, c in cells[:n_wall]:
            g[r][c] = "#"
        for r, c in cells[n_wall:n_wall + n_ice]:
            g[r][c] = "~"
        rest = cells[n_wall + n_ice:]
        if len(rest) < 4:
            continue
        k = 0
        for ch in "SG" + extras:
            r, c = rest[k]
            g[r][c] = ch
            k += 1
        if "1" in extras:
            r, c = rest[k]
            g[r][c] = "1"
            k += 1
        rows = ["".join(x) for x in g]
        lv = Level(rows)
        sol = lv.bfs()
        if not sol:
            continue
        if "1" in extras:  # решение должно проходить через портал
            st = lv.init_state()
            used = False
            for mv in sol:
                nxt = lv.step(st, mv)
                if abs(nxt[0][0] - st[0][0]) + abs(nxt[0][1] - st[0][1]) > 0 and nxt[0] in lv.portals:
                    used = True
                st = nxt
            if not used:
                continue
        if best is None or len(sol) > len(best[1]):
            best = (rows, sol)
    if best is None:
        raise RuntimeError("не удалось сгенерировать уровень")
    return best


def gen_keys(rng, H, W, min_len):
    """Комнаты, разделённые стенами с дверями; ключи разбросаны."""
    for _ in range(3000):
        g = _blank(H, W)
        c1 = rng.randint(3, W // 2 - 1)
        c2 = rng.randint(W // 2 + 1, W - 4)
        for r in range(1, H - 1):
            g[r][c1] = "#"
            g[r][c2] = "#"
        d1, d2 = rng.randint(1, H - 2), rng.randint(1, H - 2)
        g[d1][c1], g[d2][c2] = "A", "B"
        cells = _free(g)
        rng.shuffle(cells)
        for r, c in cells[: int(len(cells) * 0.22)]:
            g[r][c] = "#"
        for r, c in cells[int(len(cells) * 0.22): int(len(cells) * 0.32)]:
            g[r][c] = "~"
        left = [(r, c) for (r, c) in _free(g) if c < c1]
        mid = [(r, c) for (r, c) in _free(g) if c1 < c < c2]
        right = [(r, c) for (r, c) in _free(g) if c > c2]
        if len(left) < 3 or len(mid) < 2 or len(right) < 2:
            continue
        s = rng.choice(left)
        g[s[0]][s[1]] = "S"
        ka = rng.choice([p for p in left + mid if p != s and g[p[0]][p[1]] == "."])
        g[ka[0]][ka[1]] = "a"
        kb = rng.choice([p for p in mid + left if g[p[0]][p[1]] == "."])
        g[kb[0]][kb[1]] = "b"
        gl = rng.choice(right)
        g[gl[0]][gl[1]] = "G"
        rows = ["".join(x) for x in g]
        lv = Level(rows)
        sol = lv.bfs()
        if not sol or len(sol) < min_len:
            continue
        return rows, sol
    raise RuntimeError("не удалось сгенерировать уровень с ключами")


def gen_sokoban(rng, H, W, n_boxes, min_len, min_push):
    for _ in range(4000):
        g = _blank(H, W)
        cells = _free(g)
        rng.shuffle(cells)
        for r, c in cells[: int(len(cells) * 0.18)]:
            g[r][c] = "#"
        free = _free(g)
        if len(free) < n_boxes * 2 + 2:
            continue
        rng.shuffle(free)
        tg = free[:n_boxes]
        bx = free[n_boxes:2 * n_boxes]
        pl = free[2 * n_boxes]
        for p in tg:
            g[p[0]][p[1]] = "x"
        for p in bx:
            g[p[0]][p[1]] = "$"
        g[pl[0]][pl[1]] = "S"
        rows = ["".join(x) for x in g]
        lv = Level(rows)
        sol = lv.bfs(limit=300000)
        if not sol or len(sol) < min_len:
            continue
        # число толчков
        s = lv.init_state()
        pushes = 0
        for mv in sol:
            t = lv.step(s, mv)
            if t[2] != s[2]:
                pushes += 1
            s = t
        if pushes < min_push:
            continue
        return rows, sol
    raise RuntimeError("не удалось сгенерировать сокобан")


class NavTask(Task):
    key = "nav"
    title = "Навигация"
    channel = 3
    ability = "planning"

    def __init__(self, difficulty=1):
        self.difficulty = difficulty

    def generate(self, seed):
        rng = random.Random(seed * 15485863 + 7)
        d = self.difficulty
        lv = []
        rows, sol = gen_ice(rng, 9, 11, 0.62, 0.11, 9)
        lv.append(("Ледяное поле", rows, sol))
        rows, sol = gen_ice(rng, 11, 13, 0.62, 0.10, 9, extras="1")
        lv.append(("Лёд и порталы", rows, sol))
        rows, sol = gen_keys(rng, 11, 15, 26 + 4 * d)
        lv.append(("Ключи и двери", rows, sol))
        rows, sol = gen_sokoban(rng, 7, 8, 2, 14 + 2 * d, 4 + d)
        lv.append(("Сокобан: 2 ящика", rows, sol))
        if d >= 1:
            rows, sol = gen_sokoban(rng, 8, 9, 3, 22 + 2 * d, 7 + d)
            lv.append(("Сокобан: 3 ящика", rows, sol))
        return {"seed": seed, "difficulty": d,
                "levels": [{"name": n, "rows": r, "optimal": len(s), "solution": s} for n, r, s in lv]}

    def prompt(self, inst):
        L = [
            "# Головоломки на планирование",
            "",
            "Каждый уровень — клеточное поле. Игрок ходит на одну клетку: U — вверх, D — вниз, L — влево, R — вправо.",
            "Обозначения: `#` — стена; `.` — пол; `S` — старт; `G` — выход;",
            "`~` — лёд: встав на лёд, игрок скользит в том же направлении, пока не сойдёт со льда на обычную клетку (там он останавливается) "
            "или пока следующая клетка не окажется непроходимой (тогда он останавливается на последней клетке льда);",
            "`a` `b` — ключи (подбираются, когда игрок встаёт на клетку, и остаются у него навсегда); `A` `B` — двери, проходимы только при наличии ключа той же буквы;",
            "`1` — портал: встав на клетку с цифрой, игрок мгновенно переносится на другую клетку с той же цифрой и ход заканчивается;",
            "`$` — ящик, `x` — место для ящика. Ящик можно только толкать, входя в его клетку; ящик сдвигается на одну клетку, "
            "если за ним пол или место для ящика (не стена, не другой ящик, не лёд, не дверь без ключа). Тянуть ящики нельзя.",
            "Ход в стену, в закрытую дверь или невозможный толчок — ошибка: такое решение не засчитывается.",
            "Цель: на обычных уровнях — оказаться на `G` после последнего хода; на уровнях с ящиками — расставить все ящики на места `x`.",
            "Чем короче решение, тем лучше: полный балл — за решение не длиннее оптимального.",
            "",
        ]
        for i, l in enumerate(inst["levels"], 1):
            L.append(f"## Уровень {i}: {l['name']}")
            L.append("```")
            L += l["rows"]
            L.append("```")
            L.append("")
        L += ["## Формат ответа",
              "ОДИН блок JSON без пояснений: ключ — номер уровня, значение — строка ходов.",
              '{"1": "RRDLU", "2": "...", ...}']
        return "\n".join(L)

    def score(self, inst, text):
        ans = extract_json(text)
        items = []
        for i, l in enumerate(inst["levels"], 1):
            mv = get_item(ans, i) if ans else None
            mv = "".join(ch for ch in str(mv or "").upper() if ch in "UDLR")
            lv = Level(l["rows"])
            ok, n, trace, err = lv.simulate(mv)
            opt = l["optimal"]
            if ok:
                s = 1.0 if len(mv) <= opt else (0.8 if len(mv) <= opt + 4 else 0.6)
                det = f"решено за {len(mv)} ходов (оптимум {opt})"
            else:
                s = 0.0
                det = f"{err or 'не решено'}; ходов {len(mv)} (оптимум {opt})"
            items.append({"ok": s, "label": l["name"], "detail": det, "got": mv})
        return sum(it["ok"] for it in items) / len(items), items


# ===========================================================================
# Жёсткий набор: большие уровни, сокобан на 4 ящика и «почтальон» (обход с возвратом)
# ===========================================================================

def bfs_dist(rows, src):
    H, W = len(rows), len(rows[0])
    dist = {src: 0}
    q = deque([src])
    while q:
        r, c = q.popleft()
        for dr, dc in MOVES.values():
            nr, nc = r + dr, c + dc
            if 0 <= nr < H and 0 <= nc < W and rows[nr][nc] != "#" and (nr, nc) not in dist:
                dist[(nr, nc)] = dist[(r, c)] + 1
                q.append((nr, nc))
    return dist


def gen_postman(rng, H, W, n_mail, wall_frac=0.3):
    for _ in range(500):
        g = _blank(H, W)
        cells = _free(g)
        rng.shuffle(cells)
        for r, c in cells[: int(len(cells) * wall_frac)]:
            g[r][c] = "#"
        free = _free(g)
        if len(free) < n_mail + 5:
            continue
        rows = ["".join(x) for x in g]
        comp = bfs_dist(rows, free[0])
        if len(comp) < 0.85 * len(free):
            continue
        pts = rng.sample(sorted(comp), n_mail + 1)
        s, mails = pts[0], pts[1:]
        for (r, c) in mails:
            g[r][c] = "*"
        g[s[0]][s[1]] = "S"
        rows = ["".join(x) for x in g]
        nodes = [s] + mails
        D = [[bfs_dist(rows, a).get(b, 10 ** 6) for b in nodes] for a in nodes]
        n = len(mails)
        INF = 10 ** 9
        dp = [[INF] * (n + 1) for _ in range(1 << n)]
        for i in range(n):
            dp[1 << i][i + 1] = D[0][i + 1]
        for m in range(1 << n):
            for j in range(1, n + 1):
                if dp[m][j] >= INF or not (m >> (j - 1)) & 1:
                    continue
                for k in range(1, n + 1):
                    if (m >> (k - 1)) & 1:
                        continue
                    v = dp[m][j] + D[j][k]
                    if v < dp[m | (1 << (k - 1))][k]:
                        dp[m | (1 << (k - 1))][k] = v
        full = (1 << n) - 1
        opt = min(dp[full][j] + D[j][0] for j in range(1, n + 1))
        if opt >= INF or opt < (H + W) * 1.5:
            continue
        return rows, opt
    raise RuntimeError("не удалось сгенерировать почтальона")


def simulate_postman(rows, moves):
    H, W = len(rows), len(rows[0])
    r, c = next((r, row.index("S")) for r, row in enumerate(rows) if "S" in row)
    start = (r, c)
    mails = {(i, j) for i, row in enumerate(rows) for j, ch in enumerate(row) if ch == "*"}
    seen = set()
    trace = [(r, c)]
    for k, mv in enumerate(moves):
        dr, dc = MOVES[mv]
        nr, nc = r + dr, c + dc
        if not (0 <= nr < H and 0 <= nc < W) or rows[nr][nc] == "#":
            return False, trace, f"ход {k + 1} ({mv}) в стену", len(seen)
        r, c = nr, nc
        trace.append((r, c))
        if (r, c) in mails:
            seen.add((r, c))
    ok = seen == mails and (r, c) == start
    err = "" if ok else (f"обойдено ящиков {len(seen)} из {len(mails)}" if seen != mails else "не вернулся на S")
    return ok, trace, err, len(seen)


class NavTaskHard(NavTask):
    def generate(self, seed):
        rng = random.Random(seed * 15485863 + 19)
        lv = []
        rows, sol = gen_ice(rng, 13, 15, 0.62, 0.10, 12, samples=3000)
        lv.append({"name": "Большое ледяное поле", "rows": rows, "optimal": len(sol), "solution": sol})
        rows, sol = gen_ice(rng, 13, 15, 0.6, 0.10, 10, extras="1", samples=3000)
        lv.append({"name": "Лёд и порталы", "rows": rows, "optimal": len(sol), "solution": sol})
        rows, sol = gen_keys(rng, 13, 19, 45)
        lv.append({"name": "Ключи и двери", "rows": rows, "optimal": len(sol), "solution": sol})
        rows, sol = gen_sokoban(rng, 8, 10, 4, 30, 10)
        lv.append({"name": "Сокобан: 4 ящика", "rows": rows, "optimal": len(sol), "solution": sol})
        rows, opt = gen_postman(rng, 15, 17, 9)
        lv.append({"name": "Почтальон: 9 ящиков", "rows": rows, "optimal": opt, "solution": None, "kind": "postman"})
        return {"seed": seed, "difficulty": 3, "levels": lv}

    def prompt(self, inst):
        base = super().prompt(inst)
        extra = ("Уровень «Почтальон»: `*` — почтовые ящики (обычный пол, лёд и прочие механики там не действуют). "
                 "Нужно наступить на каждый ящик хотя бы раз и вернуться на `S`; полный балл — за кратчайший возможный маршрут, "
                 "за более длинный — пропорционально.")
        return base.replace("Чем короче решение, тем лучше: полный балл — за решение не длиннее оптимального.",
                            "Чем короче решение, тем лучше: полный балл — за решение не длиннее оптимального.\n" + extra)

    def score(self, inst, text):
        ans = extract_json(text)
        items = []
        for i, l in enumerate(inst["levels"], 1):
            mv = get_item(ans, i) if ans else None
            mv = "".join(ch for ch in str(mv or "").upper() if ch in "UDLR")
            opt = l["optimal"]
            if l.get("kind") == "postman":
                ok, trace, err, n = simulate_postman(l["rows"], mv)
                if ok:
                    s = min(1.0, opt / max(1, len(mv)))
                    det = f"обход за {len(mv)} ходов (оптимум {opt})"
                else:
                    s, det = 0.0, f"{err}; ходов {len(mv)} (оптимум {opt})"
                items.append({"ok": s, "label": l["name"], "detail": det, "got": mv})
                continue
            lvo = Level(l["rows"])
            ok, n, trace, err = lvo.simulate(mv)
            if ok:
                s = 1.0 if len(mv) <= opt else (0.8 if len(mv) <= opt + 4 else 0.6)
                det = f"решено за {len(mv)} ходов (оптимум {opt})"
            else:
                s, det = 0.0, f"{err or 'не решено'}; ходов {len(mv)} (оптимум {opt})"
            items.append({"ok": s, "label": l["name"], "detail": det, "got": mv})
        return sum(it["ok"] for it in items) / len(items), items
