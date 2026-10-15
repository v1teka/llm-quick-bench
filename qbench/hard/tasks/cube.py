"""Канал «Кубик»: мысленное вращение — кубик с цветными гранями катится по полю."""
from __future__ import annotations

import random

from .common import Task, extract_json, get_item

COLOR_NAMES = {"R": "красный", "O": "оранжевый", "G": "зелёный", "B": "синий", "Y": "жёлтый", "W": "белый",
               "K": "чёрный", "M": "пурпурный", "C": "голубой"}
INIT = {"top": "R", "bottom": "O", "north": "G", "south": "B", "east": "Y", "west": "W"}
STEP = {"N": (-1, 0), "S": (1, 0), "E": (0, 1), "W": (0, -1)}


def roll(f, mv):
    f = dict(f)
    t, b, n, s, e, w = f["top"], f["bottom"], f["north"], f["south"], f["east"], f["west"]
    if mv == "N":
        f.update(bottom=n, north=t, top=s, south=b)
    elif mv == "S":
        f.update(bottom=s, south=t, top=n, north=b)
    elif mv == "E":
        f.update(bottom=e, east=t, top=w, west=b)
    elif mv == "W":
        f.update(bottom=w, west=t, top=e, east=b)
    return f


def turn_cw(f):
    f = dict(f)
    n, e, s, w = f["north"], f["east"], f["south"], f["west"]
    f.update(east=n, south=e, west=s, north=w)
    return f


def run(board, moves):
    """-> (stamps list, final faces, trace list[(r, c, faces, event)])"""
    H, W = len(board), len(board[0])
    r, c = next((r, row.index("S")) for r, row in enumerate(board) if "S" in row)
    f = dict(INIT)
    stamps, trace = [], [(r, c, dict(f), "")]
    for mv in moves:
        dr, dc = STEP[mv]
        r, c = r + dr, c + dc
        f = roll(f, mv)
        ch = board[r][c]
        ev = ""
        if ch == "@":
            f = turn_cw(f)
            ev = "@"
        elif ch in "kmc":
            f["bottom"] = ch.upper()
            ev = "paint"
        elif ch == "?":
            stamps.append(f["bottom"])
            ev = "stamp"
        trace.append((r, c, dict(f), ev))
    return stamps, f, trace


def gen(rng, H, W, n_moves, n_stamp, n_paint, n_turn):
    for _ in range(2000):
        board = [["." for _ in range(W)] for _ in range(H)]
        r, c = rng.randrange(H), rng.randrange(W)
        board[r][c] = "S"
        path = [(r, c)]
        moves = ""
        last = None
        while len(moves) < n_moves:
            opts = []
            for mv, (dr, dc) in STEP.items():
                nr, nc = r + dr, c + dc
                if 0 <= nr < H and 0 <= nc < W:
                    back = {"N": "S", "S": "N", "E": "W", "W": "E"}[mv] == last
                    opts += [mv] * (1 if back else 3)
            mv = rng.choice(opts)
            dr, dc = STEP[mv]
            r, c = r + dr, c + dc
            moves += mv
            last = mv
            path.append((r, c))
        visited = [p for p in dict.fromkeys(path[1:]) if board[p[0]][p[1]] == "."]
        if len(visited) < n_stamp + n_paint + n_turn + 2:
            continue
        rng.shuffle(visited)
        k = 0
        for ch, n in (("?", n_stamp), ("@", n_turn)):
            for _ in range(n):
                rr, cc = visited[k]
                board[rr][cc] = ch
                k += 1
        for _ in range(n_paint):
            rr, cc = visited[k]
            board[rr][cc] = rng.choice("kmc")
            k += 1
        rows = ["".join(x) for x in board]
        stamps, f, _ = run(rows, moves)
        if len(stamps) < n_stamp + 1:  # клетки-штампы посещаются и повторно
            continue
        # краска должна влиять на ответ
        if not any(s in "KMC" for s in stamps + [f["top"], f["north"], f["east"]]):
            continue
        return rows, moves, stamps, f
    raise RuntimeError("не удалось сгенерировать маршрут кубика")


LEVELS = [  # H, W, moves, stamps, paints, turns
    (5, 6, 18, 3, 2, 1),
    (6, 7, 30, 4, 3, 2),
    (7, 8, 44, 5, 4, 3),
]


class CubeTask(Task):
    key = "cube"
    title = "Кубик"
    channel = 4
    ability = "rotation"

    def __init__(self, difficulty=1):
        self.difficulty = difficulty

    def generate(self, seed):
        rng = random.Random(seed * 2750159 + 5)
        out = []
        for (H, W, n, s, p, t) in LEVELS:
            n2 = n + 8 * self.difficulty
            rows, moves, stamps, f = gen(rng, H, W, n2, s + self.difficulty, p, t + self.difficulty)
            out.append({"board": rows, "moves": moves, "stamps": stamps,
                        "final": {"top": f["top"], "north": f["north"], "east": f["east"]}})
        return {"seed": seed, "difficulty": self.difficulty, "puzzles": out}

    def prompt(self, inst):
        L = [
            "# Катящийся кубик",
            "",
            "Кубик стоит на клеточном поле на клетке `S`. Верхняя строка поля — север, правый столбец — восток.",
            "Грани кубика в начале: верх R, низ O, север G, юг B, восток Y, запад W "
            "(R — красный, O — оранжевый, G — зелёный, B — синий, Y — жёлтый, W — белый).",
            "Ход N/S/E/W — кубик перекатывается через нижнее ребро на соседнюю клетку в этом направлении "
            "(например, при ходе E грань, смотревшая на восток, становится нижней, а верхняя — восточной).",
            "Когда кубик встаёт на клетку:",
            "- `.` и `S` — ничего не происходит;",
            "- `k` `m` `c` — краска: нижняя грань перекрашивается в K (чёрный), M (пурпурный) или C (голубой) соответственно, навсегда;",
            "- `@` — поворотный круг: кубик поворачивается на месте на 90° по часовой стрелке, если смотреть сверху "
            "(северная грань становится восточной, восточная — южной и т. д.; верх и низ не меняются);",
            "- `?` — печать: записывается текущий цвет нижней грани (каждый раз, когда кубик встаёт на эту клетку).",
            "",
            "Для каждой задачи ответь: список записанных печатью цветов по порядку и цвета верхней, северной и восточной граней после всех ходов.",
            "",
        ]
        for i, p in enumerate(inst["puzzles"], 1):
            L.append(f"## Задача {i}")
            L.append("Поле:")
            L.append("```")
            L += p["board"]
            L.append("```")
            L.append(f"Ходы ({len(p['moves'])}): {p['moves']}")
            L.append("")
        L += ["## Формат ответа",
              "ОДИН блок JSON без пояснений, цвета — одной буквой:",
              '{"1": {"stamps": ["G", "K"], "top": "R", "north": "W", "east": "Y"}, "2": {...}, ...}']
        return "\n".join(L)

    def score(self, inst, text):
        ans = extract_json(text)
        items = []
        for i, p in enumerate(inst["puzzles"], 1):
            a = get_item(ans, i) if ans else None
            want = p["stamps"]
            if isinstance(a, dict):
                got = [str(x).strip().upper()[:1] for x in (a.get("stamps") or [])]
                faces = {k: str(a.get(k, "")).strip().upper()[:1] for k in ("top", "north", "east")}
            else:
                got, faces = [], {}
            st = sum(1 for k, w in enumerate(want) if k < len(got) and got[k] == w) / len(want)
            if len(got) != len(want):
                st *= 0.8
            fc = sum(faces.get(k) == v for k, v in p["final"].items()) / 3
            s = 0.6 * st + 0.4 * fc
            items.append({"ok": s, "label": f"Маршрут {i} ({len(p['moves'])} ходов)",
                          "detail": f"печати {''.join(got) or '—'} (верно {''.join(want)}); "
                                    f"грани {faces.get('top', '?')}{faces.get('north', '?')}{faces.get('east', '?')} "
                                    f"(верно {p['final']['top']}{p['final']['north']}{p['final']['east']})",
                          "got": {"stamps": got, **faces}})
        return sum(it["ok"] for it in items) / len(items), items


# ===========================================================================
# Жёсткий набор: переворот «%» и задачи на планирование ориентации
# ===========================================================================

def flip(f):
    f = dict(f)
    f["top"], f["bottom"] = f["bottom"], f["top"]
    f["east"], f["west"] = f["west"], f["east"]
    return f


def land(f, ch):
    """Эффект клетки после прихода; -> (faces, event)."""
    if ch == "@":
        return turn_cw(f), "@"
    if ch == "%":
        return flip(f), "%"
    if ch in "kmc":
        f = dict(f)
        f["bottom"] = ch.upper()
        return f, "paint"
    return f, ""


def run2(board, moves):
    H, W = len(board), len(board[0])
    r, c = next((r, row.index("S")) for r, row in enumerate(board) if "S" in row)
    f = dict(INIT)
    stamps, trace = [], [(r, c, dict(f), "")]
    for mv in moves:
        dr, dc = STEP[mv]
        r, c = r + dr, c + dc
        if not (0 <= r < H and 0 <= c < W) or board[r][c] == "#":
            return None, None, trace
        f = roll(f, mv)
        ch = board[r][c]
        f, ev = land(f, ch)
        if ch == "?":
            stamps.append(f["bottom"])
            ev = "stamp"
        trace.append((r, c, dict(f), ev))
    return stamps, f, trace


def _key(r, c, f):
    return (r, c, f["top"], f["bottom"], f["north"], f["south"], f["east"], f["west"])


def plan_bfs(board, limit=500000):
    """BFS по состояниям (клетка, раскраска граней). -> dict state -> (prev_state, move)."""
    from collections import deque
    H, W = len(board), len(board[0])
    r, c = next((r, row.index("S")) for r, row in enumerate(board) if "S" in row)
    f0 = dict(INIT)
    s0 = _key(r, c, f0)
    prev = {s0: None}
    q = deque([(r, c, f0)])
    while q and len(prev) < limit:
        r, c, f = q.popleft()
        for mv, (dr, dc) in STEP.items():
            nr, nc = r + dr, c + dc
            if not (0 <= nr < H and 0 <= nc < W) or board[nr][nc] == "#":
                continue
            g, _ = land(roll(f, mv), board[nr][nc])
            k = _key(nr, nc, g)
            if k not in prev:
                prev[k] = (_key(r, c, f), mv)
                q.append((nr, nc, g))
    return prev


def path_to(prev, k):
    out = []
    while prev[k] is not None:
        k, mv = prev[k]
        out.append(mv)
    return "".join(reversed(out))


def gen_plan(rng, H, W, holes, paints, need_paint, need_faces=("top", "north"), min_len=None):
    for _ in range(3000):
        b = [["." for _ in range(W)] for _ in range(H)]
        cells = [(r, c) for r in range(H) for c in range(W)]
        rng.shuffle(cells)
        s, g = cells[0], cells[1]
        if abs(s[0] - g[0]) + abs(s[1] - g[1]) < (H + W) // 3:
            continue
        b[s[0]][s[1]] = "S"
        b[g[0]][g[1]] = "G"
        k = 2
        for _ in range(int(H * W * holes)):
            r, c = cells[k]
            b[r][c] = "#"
            k += 1
        for _ in range(paints):
            r, c = cells[k]
            b[r][c] = rng.choice("kmc")
            k += 1
        rows = ["".join(x) for x in b]
        prev = plan_bfs(rows)
        at_g = [st for st in prev if (st[0], st[1]) == g]
        if not at_g:
            continue
        if need_paint:
            at_g = [st for st in at_g if st[2] in "KMC"]
            if not at_g:
                continue
        # цель — набор граней; оптимум — кратчайший путь к любому состоянию с этими гранями
        pos = {"top": 2, "bottom": 3, "north": 4, "south": 5, "east": 6, "west": 7}
        best = {}
        for st in at_g:
            key = tuple(st[pos[f]] for f in need_faces)
            if need_paint and sum(ch in "KMC" for ch in key) < (2 if len(need_faces) > 2 else 1):
                continue
            L = len(path_to(prev, st))
            if key not in best or L < best[key][0]:
                best[key] = (L, st)
        if not best:
            continue
        L, st = max(best.values())
        if L < (min_len or (H + W)):
            continue
        return rows, {f: st[pos[f]] for f in need_faces}, path_to(prev, st)
    raise RuntimeError("не удалось сгенерировать план кубика")


class CubeTaskHard(CubeTask):
    """Симуляция длинного маршрута + две задачи на поиск маршрута с нужной ориентацией."""

    def generate(self, seed):
        rng = random.Random(seed * 2750159 + 17)
        sims = []
        for (H, W, n, s, p, t) in [(7, 8, 64, 6, 4, 3)]:
            for _ in range(500):
                rows, moves, _, _ = gen(rng, H, W, n, s, p, t)
                # добавляем клетки переворота на посещаемые пустые клетки
                b = [list(r) for r in rows]
                vis = []
                r0, c0 = next((r, row.index("S")) for r, row in enumerate(rows) if "S" in row)
                rr, cc = r0, c0
                for mv in moves:
                    dr, dc = STEP[mv]
                    rr, cc = rr + dr, cc + dc
                    if b[rr][cc] == "." and (rr, cc) not in vis:
                        vis.append((rr, cc))
                if len(vis) < 3:
                    continue
                for (a, d) in rng.sample(vis, 2):
                    b[a][d] = "%"
                rows = ["".join(x) for x in b]
                stamps, f, _ = run2(rows, moves)
                if stamps is None or len(stamps) < s:
                    continue
                sims.append({"kind": "sim", "board": rows, "moves": moves, "stamps": stamps,
                             "final": {"top": f["top"], "north": f["north"], "east": f["east"]}})
                break
        plans = []
        for (H, W, holes, paints, need) in [(6, 6, 0.22, 0, False), (7, 7, 0.2, 3, True)]:
            rows, target, sol = gen_plan(rng, H, W, holes, paints, need)
            plans.append({"kind": "plan", "board": rows, "target": target, "optimal": len(sol), "solution": sol})
        return {"seed": seed, "difficulty": 2, "puzzles": sims + plans}

    def prompt(self, inst):
        L = [
            "# Катящийся кубик",
            "",
            "Кубик стоит на клеточном поле на клетке `S`. Верхняя строка поля — север, правый столбец — восток.",
            "Грани кубика в начале: верх R, низ O, север G, юг B, восток Y, запад W "
            "(R — красный, O — оранжевый, G — зелёный, B — синий, Y — жёлтый, W — белый).",
            "Ход N/S/E/W — кубик перекатывается через нижнее ребро на соседнюю клетку в этом направлении "
            "(например, при ходе E грань, смотревшая на восток, становится нижней, а верхняя — восточной).",
            "Когда кубик встаёт на клетку:",
            "- `.`, `S`, `G` — ничего не происходит;",
            "- `k` `m` `c` — краска: нижняя грань перекрашивается в K (чёрный), M (пурпурный) или C (голубой) соответственно, навсегда;",
            "- `@` — поворотный круг: кубик поворачивается на месте на 90° по часовой стрелке, если смотреть сверху "
            "(северная грань становится восточной, восточная — южной и т. д.; верх и низ не меняются);",
            "- `%` — переворот: кубик переворачивается на 180° вокруг оси север–юг (верх и низ меняются местами, восток и запад — тоже);",
            "- `?` — печать: записывается текущий цвет нижней грани (каждый раз, когда кубик встаёт на эту клетку);",
            "- `#` — яма: на эту клетку вставать нельзя; уходить за край поля тоже нельзя.",
            "",
        ]
        for i, p in enumerate(inst["puzzles"], 1):
            L.append(f"## Задача {i}")
            L.append("Поле:")
            L.append("```")
            L += p["board"]
            L.append("```")
            if p["kind"] == "sim":
                L.append(f"Ходы ({len(p['moves'])}): {p['moves']}")
                L.append("Ответь: список записанных печатью цветов по порядку и цвета верхней, северной и восточной граней после всех ходов.")
            else:
                names = {"top": "верхняя", "north": "северная", "east": "восточная"}
                want = ", ".join(f"{names[k]} — {v}" for k, v in p["target"].items())
                L.append(f"Найди маршрут из S в G, после которого на G у кубика грани: {want}. "
                         f"Полный балл — за самый короткий возможный маршрут; верный маршрут длиннее оптимума на 1–2 хода — 0.85, "
                         f"ещё длиннее — 0.7; неверный — 0.")
            L.append("")
        L += ["## Формат ответа",
              "ОДИН блок JSON без пояснений, цвета — одной буквой:",
              '{"1": {"stamps": ["G", "K"], "top": "R", "north": "W", "east": "Y"}, "2": {"moves": "NNEES"}, "3": {"moves": "..."}}']
        return "\n".join(L)

    def score(self, inst, text):
        ans = extract_json(text)
        items = []
        for i, p in enumerate(inst["puzzles"], 1):
            a = get_item(ans, i) if ans else None
            if p["kind"] == "sim":
                want = p["stamps"]
                if isinstance(a, dict):
                    got = [str(x).strip().upper()[:1] for x in (a.get("stamps") or [])]
                    faces = {k: str(a.get(k, "")).strip().upper()[:1] for k in ("top", "north", "east")}
                else:
                    got, faces = [], {}
                st = sum(1 for k, w in enumerate(want) if k < len(got) and got[k] == w) / len(want)
                if len(got) != len(want):
                    st *= 0.8
                fc = sum(faces.get(k) == v for k, v in p["final"].items()) / 3
                s = 0.6 * st + 0.4 * fc
                items.append({"ok": s, "label": f"Маршрут {len(p['moves'])} ходов",
                              "detail": f"печати {''.join(got) or '—'} (верно {''.join(want)}); грани "
                                        f"{faces.get('top', '?')}{faces.get('north', '?')}{faces.get('east', '?')} "
                                        f"(верно {p['final']['top']}{p['final']['north']}{p['final']['east']})",
                              "got": {"stamps": got, **faces}})
            else:
                mv = a.get("moves") if isinstance(a, dict) else a
                mv = "".join(ch for ch in str(mv or "").upper() if ch in "NSEW")
                stamps, f, trace = run2(p["board"], mv)
                g = next((r, row.index("G")) for r, row in enumerate(p["board"]) if "G" in row)
                if f is None:
                    s, det = 0.0, f"ход {len(trace)} невозможен (яма или край)"
                elif (trace[-1][0], trace[-1][1]) != g:
                    s, det = 0.0, "кубик не на G"
                elif any(f[k] != v for k, v in p["target"].items()):
                    s, det = 0.2, "на G, но грани " + "/".join(f[k] for k in p["target"]) + " вместо " + "/".join(p["target"].values())
                else:
                    L = len(mv)
                    s = 1.0 if L <= p["optimal"] else (0.85 if L <= p["optimal"] + 2 else 0.7)
                    det = f"верно за {L} ходов (оптимум {p['optimal']})"
                items.append({"ok": s, "label": "План: " + "/".join(p["target"].values()) + " на G",
                              "detail": det, "got": mv})
        return sum(it["ok"] for it in items) / len(items), items



class CubeTaskHarder(CubeTaskHard):
    def generate(self, seed):
        rng = random.Random(seed * 2750159 + 23)
        sims = []
        for (H, W, n, st, p, t) in [(8, 8, 100, 9, 5, 4)]:
            for _ in range(500):
                rows, moves, _, _ = gen(rng, H, W, n, st, p, t)
                b = [list(r) for r in rows]
                vis = []
                r0, c0 = next((r, row.index("S")) for r, row in enumerate(rows) if "S" in row)
                rr, cc = r0, c0
                for mv in moves:
                    dr, dc = STEP[mv]
                    rr, cc = rr + dr, cc + dc
                    if b[rr][cc] == "." and (rr, cc) not in vis:
                        vis.append((rr, cc))
                if len(vis) < 4:
                    continue
                for (a, d) in rng.sample(vis, 3):
                    b[a][d] = "%"
                rows = ["".join(x) for x in b]
                stamps, f, _ = run2(rows, moves)
                if stamps is None or len(stamps) < st:
                    continue
                sims.append({"kind": "sim", "board": rows, "moves": moves, "stamps": stamps,
                             "final": {"top": f["top"], "north": f["north"], "east": f["east"]}})
                break
        plans = []
        for (H, W, holes, paints, need, faces, ml) in [(7, 8, 0.24, 0, False, ("top", "north"), 15),
                                                       (8, 8, 0.2, 3, True, ("top", "north"), 20),
                                                       (8, 8, 0.18, 4, True, ("top", "north", "east"), 22)]:
            rows, target, sol = gen_plan(rng, H, W, holes, paints, need, faces, ml)
            plans.append({"kind": "plan", "board": rows, "target": target, "optimal": len(sol), "solution": sol})
        return {"seed": seed, "difficulty": 3, "puzzles": sims + plans}
