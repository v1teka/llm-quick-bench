"""Канал «Оптика»: мысленная трассировка лазерных лучей через зеркала и делители."""
from __future__ import annotations

import random

from .common import Task, extract_json, get_item

DIRS = {"R": (0, 1), "L": (0, -1), "U": (-1, 0), "D": (1, 0)}
EMIT = {">": "R", "<": "L", "^": "U", "v": "D"}
SLASH = {"R": "U", "U": "R", "L": "D", "D": "L"}       # /
BACK = {"R": "D", "D": "R", "L": "U", "U": "L"}        # \
TARGETS = "ABCDEFGHJK"


def trace(board):
    """-> (lit targets set, cells set, segments list[(r0,c0,r1,c1)] в порядке распространения)."""
    H, W = len(board), len(board[0])
    beams = []
    for r in range(H):
        for c in range(W):
            if board[r][c] in EMIT:
                beams.append((r, c, EMIT[board[r][c]]))
    seen, lit, cells, path = set(), set(), set(), []
    queue = list(beams)
    steps = 0
    while queue and steps < 20000:
        r, c, d = queue.pop(0)
        while True:
            steps += 1
            dr, dc = DIRS[d]
            nr, nc = r + dr, c + dc
            if not (0 <= nr < H and 0 <= nc < W):
                path.append((r, c, nr, nc, "out"))
                break
            ch = board[nr][nc]
            path.append((r, c, nr, nc, ch))
            if ch == "#" or ch in EMIT:
                break
            if ch in TARGETS or ch in "XYZ":
                lit.add(ch)
                break
            if (nr, nc, d) in seen:
                break
            seen.add((nr, nc, d))
            cells.add((nr, nc))
            r, c = nr, nc
            if ch == "/":
                d = SLASH[d]
            elif ch == "\\":
                d = BACK[d]
            elif ch == "*":
                queue.append((r, c, SLASH[d]))
    return lit, cells, path


def gen_board(rng, W, H, n_emit, n_turns, n_split, n_wall, n_lit, n_unlit, n_distract, all_lit=False):
    """Конструктивно прокладывает лучи (повороты и делители), затем добавляет отвлекающие объекты."""
    for _ in range(3000):
        b = [["." for _ in range(W)] for _ in range(H)]
        beams = []
        for _ in range(n_emit):
            side = rng.choice("NSWE")
            if side == "W":
                r, c, ch = rng.randint(1, H - 2), 0, ">"
            elif side == "E":
                r, c, ch = rng.randint(1, H - 2), W - 1, "<"
            elif side == "N":
                r, c, ch = 0, rng.randint(1, W - 2), "v"
            else:
                r, c, ch = H - 1, rng.randint(1, W - 2), "^"
            if b[r][c] != ".":
                continue
            b[r][c] = ch
            beams.append((r, c, EMIT[ch], n_turns))
        splits, lit_left = n_split, n_lit
        guard = 0
        while beams and guard < 200:
            guard += 1
            r, c, d, turns = beams.pop(0)
            dr, dc = DIRS[d]
            steps = rng.randint(2, 5)
            moved = 0
            while moved < steps:
                nr, nc = r + dr, c + dc
                if not (1 <= nr < H - 1 and 1 <= nc < W - 1) or b[nr][nc] != ".":
                    break
                r, c = nr, nc
                moved += 1
            if moved == 0:
                continue
            if turns > 0:
                if splits > 0 and rng.random() < 0.35:
                    b[r][c] = "*"
                    splits -= 1
                    beams.append((r, c, d, turns - 1))
                    beams.append((r, c, SLASH[d], turns - 1))
                else:
                    ch = rng.choice("/\\")
                    b[r][c] = ch
                    beams.append((r, c, SLASH[d] if ch == "/" else BACK[d], turns - 1))
            elif lit_left > 0:
                b[r][c] = TARGETS[n_lit - lit_left]
                lit_left -= 1
        board = ["".join(row) for row in b]
        lit, cells, _ = trace(board)
        if not lit:
            continue
        # отвлекающие объекты — только вне путей лучей
        free = [(r, c) for r in range(1, H - 1) for c in range(1, W - 1) if b[r][c] == "." and (r, c) not in cells]
        rng.shuffle(free)
        used = set(ch for row in board for ch in row if ch in TARGETS)
        spare = [t for t in TARGETS if t not in used]
        k = 0
        for ch, n in (("#", n_wall), ("/", n_distract // 2 + n_distract % 2), ("\\", n_distract // 2)):
            for _ in range(n):
                if k < len(free):
                    rr, cc = free[k]
                    b[rr][cc] = ch
                    k += 1
        for t in spare[:n_unlit]:
            if k < len(free):
                rr, cc = free[k]
                b[rr][cc] = t
                k += 1
        # переименовываем мишени по алфавиту в порядке чтения, чтобы буквы не выдавали горящие
        order = [ch for row in b for ch in row if ch in TARGETS]
        ren = {old: TARGETS[i] for i, old in enumerate(order)}
        b = [[ren.get(ch, ch) for ch in row] for row in b]
        board = ["".join(row) for row in b]
        lit, cells, _ = trace(board)
        n_t = len(order)
        if all_lit:
            if len(lit) != n_t or n_t < n_lit:
                continue
        elif not (1 <= len(lit) < n_t):
            continue
        if n_split and sum(1 for (r, c) in cells if board[r][c] == "*") < 1:
            continue
        if len(cells) < (W + H) * 0.9:
            continue
        return board, lit, cells
    raise RuntimeError("не удалось сгенерировать поле")


LEVELS = [  # W, H, emit, turns, split, wall, lit, unlit, distract
    (11, 8, 1, 4, 0, 4, 1, 2, 3),
    (14, 10, 1, 6, 1, 5, 2, 2, 5),
    (16, 11, 2, 6, 2, 6, 3, 2, 6),
    (19, 13, 2, 8, 3, 8, 3, 3, 8),
    (22, 15, 3, 8, 4, 10, 4, 3, 10),
]


class OpticsTask(Task):
    key = "optics"
    title = "Оптика"
    channel = 2
    ability = "simulation"

    def __init__(self, difficulty=1):
        self.difficulty = difficulty

    def generate(self, seed):
        rng = random.Random(seed * 104729 + 3)
        boards = []
        lv = LEVELS[: 4 + (self.difficulty >= 1)]
        for (W, H, e, tr, s, w, nl, nu, nd) in lv:
            board, lit, cells = gen_board(rng, W + self.difficulty, H, e, tr + self.difficulty, s, w, nl, nu, nd)
            boards.append({"board": board, "targets": "".join(sorted(lit)), "cells": len(cells)})
        return {"seed": seed, "difficulty": self.difficulty, "boards": boards}

    def prompt(self, inst):
        L = [
            "# Лазеры и зеркала",
            "",
            "Поле — сетка клеток. Обозначения:",
            "- `.` — пусто; `#` — стена (поглощает луч);",
            "- `>` `<` `^` `v` — лазер, испускает луч в указанную сторону из своей клетки;",
            "- `/` и `\\` — зеркала, поворачивают луч на 90°: для `/` луч, идущий вправо, уходит вверх, идущий вниз — уходит влево; для `\\` луч, идущий вправо, уходит вниз, идущий вверх — уходит влево;",
            "- `*` — полупрозрачное зеркало: луч проходит дальше прямо И одновременно отражается, как от `/`;",
            "- заглавные латинские буквы — мишени: луч, попавший в мишень, зажигает её и поглощается;",
            "- лазер, попавший лучом в другой лазер, поглощается им; луч, вышедший за край поля, пропадает;",
            "- если луч входит в клетку в том же направлении, в каком уже проходил через неё, он дальше не прослеживается (так исключены бесконечные петли).",
            "",
            "Для каждого поля ответь: (1) какие мишени горят; (2) сколько различных клеток поля прошёл хотя бы один луч. "
            "Считаются пустые клетки и клетки с зеркалами/делителями, через которые луч прошёл или в которых отразился; "
            "клетки лазеров, стен и мишеней не считаются, каждая клетка считается один раз.",
            "",
        ]
        for i, b in enumerate(inst["boards"], 1):
            L.append(f"## Поле {i} ({len(b['board'][0])}×{len(b['board'])})")
            L.append("```")
            L += b["board"]
            L.append("```")
            L.append("")
        L += [
            "## Формат ответа",
            "ОДИН блок JSON без пояснений: мишени — строка букв в алфавитном порядке.",
            '{"1": {"targets": "AC", "cells": 23}, "2": {...}, ...}',
        ]
        return "\n".join(L)

    def score(self, inst, text):
        ans = extract_json(text)
        items = []
        for i, b in enumerate(inst["boards"], 1):
            a = get_item(ans, i) if ans else None
            got_t, got_c = None, None
            if isinstance(a, dict):
                t = a.get("targets", "")
                if isinstance(t, list):
                    t = "".join(map(str, t))
                got_t = "".join(sorted(set(str(t).upper().replace(",", "").replace(" ", ""))))
                try:
                    got_c = int(a.get("cells"))
                except (TypeError, ValueError):
                    got_c = None
            st = 0.5 if got_t == b["targets"] else 0.0
            if got_c == b["cells"]:
                sc = 0.5
            elif got_c is not None and abs(got_c - b["cells"]) <= 2:
                sc = 0.25
            else:
                sc = 0.0
            items.append({"ok": st + sc, "label": f"Поле {i}",
                          "detail": f"мишени {got_t or '—'} (верно {b['targets']}), клеток {got_c} (верно {b['cells']})",
                          "got": {"targets": got_t, "cells": got_c}})
        return sum(it["ok"] for it in items) / len(items), items


# ===========================================================================
# Жёсткий набор: большие поля + задачи на проектирование (расставить зеркала)
# ===========================================================================

def gen_design(rng, W, H, n_emit, n_turns, n_split, n_wall, n_targets, n_remove):
    """Поле, где горят все мишени; убираем n_remove зеркал на путях лучей -> часть мишеней гаснет."""
    for _ in range(400):
        board, lit, cells = gen_board(rng, W, H, n_emit, n_turns, n_split, n_wall, n_targets, 0, n_turns, all_lit=True)
        n_t = sum(ch in TARGETS for row in board for ch in row)
        if len(lit) != n_t:
            continue
        mir = [(r, c) for (r, c) in cells if board[r][c] in "/\\"]
        if len(mir) < n_remove + 2:
            continue
        rng.shuffle(mir)
        b = [list(row) for row in board]
        removed = []
        for (r, c) in mir:
            if len(removed) == n_remove:
                break
            old = b[r][c]
            b[r][c] = "."
            lit2, _, _ = trace(["".join(x) for x in b])
            if len(lit2) < len(lit) and (not removed or len(lit2) < prev):
                removed.append((r, c, old))
                prev = len(lit2)
            else:
                b[r][c] = old
        if len(removed) != n_remove:
            continue
        rows = ["".join(x) for x in b]
        lit3, _, _ = trace(rows)
        if len(lit3) >= n_t:
            continue
        return rows, removed, n_t
    raise RuntimeError("не удалось сгенерировать задачу на проектирование")


class OpticsTaskHard(OpticsTask):
    def generate(self, seed):
        rng = random.Random(seed * 104729 + 31)
        boards = []
        for (W, H, e, tr, s, w, nl, nu, nd) in [(24, 16, 3, 9, 5, 12, 4, 3, 12), (28, 18, 4, 9, 6, 14, 5, 3, 14)]:
            board, lit, cells = gen_board(rng, W, H, e, tr, s, w, nl, nu, nd)
            boards.append({"kind": "trace", "board": board, "targets": "".join(sorted(lit)), "cells": len(cells)})
        for (W, H, e, tr, s, w, nt, k) in [(14, 10, 1, 6, 1, 6, 2, 1), (18, 12, 2, 7, 2, 8, 3, 2), (22, 14, 2, 8, 3, 10, 4, 3)]:
            rows, removed, n_t = gen_design(rng, W, H, e, tr, s, w, nt, k)
            boards.append({"kind": "design", "board": rows, "k": k, "n_targets": n_t,
                           "solution": [[r + 1, c + 1, ch] for r, c, ch in removed]})
        return {"seed": seed, "difficulty": 2, "boards": boards}

    def prompt(self, inst):
        base = super().prompt({"boards": []}).split("## Формат ответа")[0].rstrip()
        L = [base.replace("Для каждого поля ответь", "Для полей «трассировка» ответь"), "",
             "Для полей «проектирование»: добавь на пустые клетки `.` не более K зеркал (`/` или `\\`) так, чтобы загорелись ВСЕ мишени поля. "
             "Координаты — (строка, столбец), счёт с 1 от левого верхнего угла.", ""]
        for i, b in enumerate(inst["boards"], 1):
            kind = "трассировка" if b["kind"] == "trace" else f"проектирование, K = {b['k']}"
            L.append(f"## Поле {i} ({len(b['board'][0])}×{len(b['board'])}) — {kind}")
            L.append("```")
            L += b["board"]
            L.append("```")
            L.append("")
        L += ["## Формат ответа",
              "ОДИН блок JSON без пояснений:",
              '{"1": {"targets": "AC", "cells": 23}, "3": {"mirrors": [[5, 7, "/"], [2, 3, "\\\\"]]}, ...}']
        return "\n".join(L)

    def score(self, inst, text):
        ans = extract_json(text)
        items = []
        for i, b in enumerate(inst["boards"], 1):
            a = get_item(ans, i) if ans else None
            if b["kind"] == "trace":
                got_t, got_c = None, None
                if isinstance(a, dict):
                    t = a.get("targets", "")
                    if isinstance(t, list):
                        t = "".join(map(str, t))
                    got_t = "".join(sorted(set(str(t).upper().replace(",", "").replace(" ", ""))))
                    try:
                        got_c = int(a.get("cells"))
                    except (TypeError, ValueError):
                        got_c = None
                st = 0.5 if got_t == b["targets"] else 0.0
                sc = 0.5 if got_c == b["cells"] else (0.25 if got_c is not None and abs(got_c - b["cells"]) <= 2 else 0.0)
                items.append({"ok": st + sc, "label": f"Трассировка {i}",
                              "detail": f"мишени {got_t or '—'} (верно {b['targets']}), клеток {got_c} (верно {b['cells']})",
                              "got": {"targets": got_t, "cells": got_c}})
                continue
            mirrors = a.get("mirrors") if isinstance(a, dict) else None
            rows = [list(r) for r in b["board"]]
            err = ""
            placed = []
            if not isinstance(mirrors, list):
                err = "нет списка зеркал"
            else:
                for m in mirrors:
                    try:
                        r, c, ch = int(m[0]) - 1, int(m[1]) - 1, str(m[2])
                    except (TypeError, ValueError, IndexError):
                        err = f"неверная запись {m!r}"
                        break
                    if ch not in ("/", "\\"):
                        err = f"неверный символ {ch!r}"
                        break
                    if not (0 <= r < len(rows) and 0 <= c < len(rows[0])) or rows[r][c] != ".":
                        err = f"клетка ({r + 1},{c + 1}) не пустая или вне поля"
                        break
                    rows[r][c] = ch
                    placed.append((r, c, ch))
            if not err and len(placed) > b["k"]:
                err = f"зеркал {len(placed)} > K = {b['k']}"
            lit, _, _ = trace(["".join(x) for x in rows])
            if err:
                s, det = 0.0, err
            elif len(lit) == b["n_targets"]:
                s, det = 1.0, f"горят все {b['n_targets']} мишеней, зеркал {len(placed)}"
            else:
                s, det = 0.4 * len(lit) / b["n_targets"], f"горит {len(lit)} из {b['n_targets']} мишеней"
            items.append({"ok": s, "label": f"Проект {i} (K={b['k']})", "detail": det, "got": placed})
        return sum(it["ok"] for it in items) / len(items), items



class OpticsTaskHarder(OpticsTaskHard):
    """Поля крупнее + «запретные» мишени X/Y, которые должны остаться тёмными."""

    def generate(self, seed):
        rng = random.Random(seed * 104729 + 47)
        boards = []
        for (W, H, e, tr, s, w, nl, nu, nd) in [(34, 22, 5, 10, 8, 14, 5, 3, 16), (40, 26, 6, 10, 10, 18, 6, 4, 20)]:
            board, lit, cells = gen_board(rng, W, H, e, tr, s, w, nl, nu, nd)
            boards.append({"kind": "trace", "board": board, "targets": "".join(sorted(lit)), "cells": len(cells)})
        for (W, H, e, tr, s, w, nt, k, nf) in [(18, 12, 2, 7, 2, 8, 3, 2, 2), (24, 15, 2, 8, 3, 10, 4, 3, 2), (28, 18, 3, 8, 4, 12, 5, 4, 3)]:
            for _ in range(200):
                rows, removed, n_t = gen_design(rng, W, H, e, tr, s, w, nt, k)
                # запретные мишени — на клетках вне путей решённого поля
                solved = [list(r) for r in rows]
                for (r, c, ch) in removed:
                    solved[r][c] = ch
                _, cells_s, _ = trace(["".join(x) for x in solved])
                _, cells_u, _ = trace(rows)
                free = [(r, c) for r in range(1, H - 1) for c in range(1, W - 1)
                        if rows[r][c] == "." and (r, c) not in cells_s and (r, c) not in cells_u]
                if len(free) < nf:
                    continue
                rng.shuffle(free)
                b = [list(r) for r in rows]
                for (r, c), ch in zip(free[:nf], "XYZ"):
                    b[r][c] = ch
                rows2 = ["".join(x) for x in b]
                boards.append({"kind": "design", "board": rows2, "k": k, "n_targets": n_t, "forbidden": "XYZ"[:nf],
                               "solution": [[r + 1, c + 1, ch] for r, c, ch in removed]})
                break
        return {"seed": seed, "difficulty": 3, "boards": boards}

    def prompt(self, inst):
        txt = super().prompt(inst)
        return txt.replace("так, чтобы загорелись ВСЕ мишени поля.",
                           "так, чтобы загорелись ВСЕ мишени A, B, C… поля, а «запретные» мишени X, Y, Z (если они есть) остались тёмными.")

    def score(self, inst, text):
        total, items = super().score(inst, text)
        # пересчёт для полей с запретными мишенями
        ans = extract_json(text)
        for i, b in enumerate(inst["boards"], 1):
            if b["kind"] != "design" or not b.get("forbidden"):
                continue
            it = items[i - 1]
            rows = [list(r) for r in b["board"]]
            for (r, c, ch) in it.get("got") or []:
                rows[r][c] = ch
            lit, _, _ = trace(["".join(x) for x in rows])
            bad = sorted(ch for ch in lit if ch in b["forbidden"])
            good = len([ch for ch in lit if ch not in "XYZ"])
            if it["ok"] >= 0.999 and bad:
                it["ok"] = 0.3
                it["detail"] += f"; но горят запретные {''.join(bad)}"
            elif it["ok"] < 0.999 and it.get("got") and "зеркал" not in it["detail"][:7]:
                it["ok"] = (0.4 * good / b["n_targets"]) * (0.5 if bad else 1.0)
                it["detail"] = f"горит {good} из {b['n_targets']} мишеней" + (f", и запретные {''.join(bad)}" if bad else "")
        return sum(it["ok"] for it in items) / len(items), items
