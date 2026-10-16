"""Картинки для телевизоров (каналы задач) и карточки ролика в формате 16:9."""
from __future__ import annotations

import math

import numpy as np
from PIL import Image, ImageDraw

from ..render.cards import font, score_color, pct, _shorten, BG, PANEL, TEXT, DIM, ACCENT, GOOD, BAD, MID
from . import config as C

ARC_COLORS = {0: (17, 17, 17), 1: (0, 116, 217), 2: (255, 65, 54), 3: (46, 204, 64), 4: (255, 220, 0),
              5: (170, 170, 170), 6: (240, 18, 190), 7: (255, 133, 27), 8: (127, 219, 255), 9: (135, 12, 37)}
CUBE_COLORS = {"R": (220, 50, 47), "O": (245, 140, 30), "G": (60, 170, 80), "B": (40, 90, 200), "Y": (245, 210, 40),
               "W": (240, 240, 240), "K": (25, 25, 25), "M": (200, 60, 200), "C": (60, 210, 230)}


def _base(W, H, ch, title, score):
    im = Image.new("RGB", (W, H), (12, 14, 20))
    d = ImageDraw.Draw(im)
    s = W / 480
    d.rectangle([0, 0, W, int(30 * s)], fill=(22, 25, 33))
    d.rounded_rectangle([int(8 * s), int(6 * s), int(8 * s) + int(70 * s), int(24 * s)], radius=int(4 * s), fill=ACCENT)
    d.text((int(14 * s), int(8 * s)), f"КАНАЛ {ch}", font=font(int(11 * s), True), fill=BG)
    d.text((int(86 * s), int(6 * s)), title, font=font(int(15 * s), True), fill=TEXT)
    if score is not None:
        t = pct(score)
        f = font(int(16 * s), True)
        d.text((W - int(10 * s) - d.textlength(t, font=f), int(5 * s)), t, font=f, fill=score_color(score))
    return im, d, s


def note_strip(im, text, bad=False):
    """Полоса-примечание внизу канала (например, «набор одним ответом: нет ответа · по одной: 100%»)."""
    im = im.copy()
    W, H = im.size
    s = W / 480
    h = int(22 * s)
    ov = Image.new("RGB", (W, h), (30, 12, 14) if bad else (18, 22, 30))
    im.paste(ov, (0, H - h))
    d = ImageDraw.Draw(im)
    f = font(int(11 * s), True)
    t = _shorten(d, text, f, W - int(16 * s))
    d.text((int(8 * s), H - h + int(4 * s)), t, font=f, fill=BAD if bad else DIM)
    return im


def _grid_img(rows, cell, maxw=None, maxh=None):
    g = [[int(ch) for ch in r] for r in rows] if rows else [[0]]
    h, w = len(g), len(g[0])
    if maxw:
        cell = max(2, min(cell, maxw // max(w, 1), (maxh or 9999) // max(h, 1)))
    im = Image.new("RGB", (w * cell, h * cell), (40, 40, 40))
    d = ImageDraw.Draw(im)
    for r in range(h):
        for c in range(len(g[r])):
            d.rectangle([c * cell, r * cell, (c + 1) * cell - 2, (r + 1) * cell - 2], fill=ARC_COLORS.get(g[r][c], (90, 90, 90)))
    return im


# ---------------------------------------------------------------------------
# каналы
# ---------------------------------------------------------------------------

def channel_arc(inst, res, t, W=480, H=270):
    im, d, s = _base(W, H, 1, "Логика · головоломки ARC", res["score"])
    items = res["items"]
    n = len(items)
    cols = 4
    tw, th = (W - int(20 * s)) / cols, (H - int(40 * s)) / 2
    shown = max(1, int(math.ceil(n * min(1.0, t * 1.3 + 0.05))))
    for k, (p, it) in enumerate(zip(inst["puzzles"], items)):
        if k >= shown:
            break
        cx, cy = int(10 * s + (k % cols) * tw), int(36 * s + (k // cols) * th)
        d.rounded_rectangle([cx + 2, cy + 2, cx + tw - 4, cy + th - 4], radius=int(5 * s), fill=PANEL)
        a = _grid_img(p["test"]["input"], 6, int(tw * 0.42), int(th * 0.6))
        got = it.get("got")
        b = _grid_img(got, 6, int(tw * 0.42), int(th * 0.6)) if got else None
        im.paste(a, (cx + int(5 * s), cy + int(16 * s)))
        if b is not None:
            im.paste(b, (cx + int(tw * 0.52), cy + int(16 * s)))
        d.text((cx + int(5 * s), cy + int(3 * s)), _shorten(d, p["name"], font(int(9 * s)), tw - 30 * s), font=font(int(9 * s)), fill=DIM)
        ok = it["ok"] >= 0.999
        d.text((cx + tw - int(18 * s), cy + int(2 * s)), "✓" if ok else "✗", font=font(int(12 * s), True), fill=GOOD if ok else BAD)
        d.text((cx + int(tw * 0.44), cy + int(th * 0.35)), "→", font=font(int(12 * s), True), fill=DIM)
    return im


def channel_optics(inst, res, t, W=480, H=270):
    from .tasks.optics import trace
    im, d, s = _base(W, H, 2, "Оптика · лазеры и зеркала", res["score"])
    items = res["items"]
    k = min(len(inst["boards"]) - 1, int(t * len(inst["boards"])))
    b = inst["boards"][k]
    rows = [list(r) for r in b["board"]]
    if b.get("kind") == "design":
        for (r, c, ch) in (items[k].get("got") or []):
            if 0 <= r < len(rows) and 0 <= c < len(rows[0]) and rows[r][c] == ".":
                rows[r][c] = ch
    rows = ["".join(r) for r in rows]
    lit, cells, path = trace(rows)
    Hh, Ww = len(rows), len(rows[0])
    cell = int(min((W * 0.62) / Ww, (H - 46 * s) / Hh))
    ox, oy = int(10 * s), int(38 * s)
    for r in range(Hh):
        for c in range(Ww):
            ch = rows[r][c]
            x0, y0 = ox + c * cell, oy + r * cell
            col = (28, 31, 40) if ch != "#" else (95, 95, 105)
            d.rectangle([x0, y0, x0 + cell - 1, y0 + cell - 1], fill=col)
            if ch in "/\\*":
                if ch == "/":
                    d.line([x0 + 1, y0 + cell - 2, x0 + cell - 2, y0 + 1], fill=(200, 210, 230), width=max(1, cell // 6))
                elif ch == "\\":
                    d.line([x0 + 1, y0 + 1, x0 + cell - 2, y0 + cell - 2], fill=(200, 210, 230), width=max(1, cell // 6))
                else:
                    d.line([x0 + 1, y0 + cell - 2, x0 + cell - 2, y0 + 1], fill=(120, 200, 255), width=max(1, cell // 6))
            elif ch in "><^v":
                d.rectangle([x0 + 1, y0 + 1, x0 + cell - 2, y0 + cell - 2], fill=(200, 40, 40))
            elif ch.isalpha():
                on = ch in lit
                d.ellipse([x0 + 1, y0 + 1, x0 + cell - 2, y0 + cell - 2], fill=(255, 220, 60) if on else (70, 70, 80))
    frac = min(1.0, (t * len(inst["boards"]) - k) * 1.4)
    nseg = int(len(path) * frac)
    for (r0, c0, r1, c1, _) in path[:nseg]:
        p0 = (ox + c0 * cell + cell // 2, oy + r0 * cell + cell // 2)
        p1 = (ox + c1 * cell + cell // 2, oy + r1 * cell + cell // 2)
        d.line([p0, p1], fill=(255, 60, 60), width=max(1, cell // 4))
    # список результатов
    x = int(W * 0.66)
    y = int(40 * s)
    for i, it in enumerate(items):
        ok = it["ok"] >= 0.999
        col = GOOD if ok else (MID if it["ok"] > 0 else BAD)
        d.text((x, y), ("▶ " if i == k else "  ") + it["label"], font=font(int(10 * s), i == k), fill=TEXT if i == k else DIM)
        d.text((W - int(40 * s), y), pct(it["ok"]), font=font(int(10 * s), True), fill=col)
        y += int(20 * s)
    return im


def channel_nav(inst, res, t, W=480, H=270):
    from .tasks.nav import Level
    im, d, s = _base(W, H, 3, "Навигация · поиск плана", res["score"])
    items = res["items"]
    k = min(len(inst["levels"]) - 1, int(t * len(inst["levels"])))
    lvl = inst["levels"][k]
    L = Level(lvl["rows"])
    mv = items[k].get("got") or ""
    if lvl.get("kind") == "postman":
        from .tasks.nav import simulate_postman
        ok, trace, err, _ = simulate_postman(lvl["rows"], mv)
    else:
        ok, nvalid, trace, err = L.simulate(mv)
    Hh, Ww = L.H, L.W
    cell = int(min((W * 0.6) / Ww, (H - 46 * s) / Hh))
    ox, oy = int(10 * s), int(38 * s)
    palette = {"#": (90, 92, 100), "~": (150, 210, 240), ".": (36, 40, 50), "G": (60, 200, 90), "S": (36, 40, 50),
               "a": (240, 200, 40), "b": (240, 140, 40), "A": (180, 150, 30), "B": (180, 100, 30), "1": (170, 90, 220),
               "x": (60, 90, 60), "$": (36, 40, 50), "%": (36, 40, 50), "*": (230, 120, 40)}
    for r in range(Hh):
        for c in range(Ww):
            ch = lvl["rows"][r][c]
            d.rectangle([ox + c * cell, oy + r * cell, ox + (c + 1) * cell - 1, oy + (r + 1) * cell - 1], fill=palette.get(ch, (36, 40, 50)))
    frac = min(1.0, (t * len(inst["levels"]) - k) * 1.3)
    n = max(1, int(len(trace) * frac))
    # ящики — пересчитываем состояние по ходам
    st = L.init_state()
    if lvl.get("kind") != "postman":
        for m in mv[:n - 1]:
            nx_ = L.step(st, m)
            if nx_ is None:
                break
            st = nx_
    for (br, bc) in st[2]:
        d.rectangle([ox + bc * cell + 2, oy + br * cell + 2, ox + (bc + 1) * cell - 3, oy + (br + 1) * cell - 3], fill=(170, 110, 60))
    pts = [(ox + c * cell + cell // 2, oy + r * cell + cell // 2) for r, c in trace[:n]]
    if len(pts) > 1:
        d.line(pts, fill=(255, 90, 90), width=max(1, cell // 5))
    if pts:
        x, y = pts[-1]
        d.ellipse([x - cell // 3, y - cell // 3, x + cell // 3, y + cell // 3], fill=(255, 255, 255))
    x = int(W * 0.64)
    y = int(40 * s)
    for i, it in enumerate(items):
        col = GOOD if it["ok"] >= 0.999 else (MID if it["ok"] > 0 else BAD)
        d.text((x, y), ("▶ " if i == k else "  ") + _shorten(d, it["label"], font(int(10 * s)), W * 0.26), font=font(int(10 * s), i == k), fill=TEXT if i == k else DIM)
        d.text((W - int(40 * s), y), pct(it["ok"]), font=font(int(10 * s), True), fill=col)
        y += int(20 * s)
    d.text((int(W * 0.64), H - int(40 * s)), _shorten(d, items[k]["detail"], font(int(9 * s)), W * 0.34), font=font(int(9 * s)), fill=DIM)
    return im


def channel_cube(inst, res, t, W=480, H=270):
    from .tasks.cube import run2, INIT
    im, d, s = _base(W, H, 4, "Кубик · 3D-вращение", res["score"])
    items = res["items"]
    k = min(len(inst["puzzles"]) - 1, int(t * len(inst["puzzles"])))
    p = inst["puzzles"][k]
    moves = p["moves"] if p["kind"] == "sim" else (items[k].get("got") or "")
    _, _, trace = run2(p["board"], moves)
    Hh, Ww = len(p["board"]), len(p["board"][0])
    cell = int(min((W * 0.5) / Ww, (H - 46 * s) / Hh))
    ox, oy = int(10 * s), int(38 * s)
    for r in range(Hh):
        for c in range(Ww):
            ch = p["board"][r][c]
            col = (40, 44, 55)
            if ch == "#":
                col = (10, 10, 12)
            elif ch in "kmc":
                col = CUBE_COLORS[ch.upper()]
            elif ch == "G":
                col = (60, 130, 70)
            d.rectangle([ox + c * cell, oy + r * cell, ox + (c + 1) * cell - 2, oy + (r + 1) * cell - 2], fill=col)
            if ch in "?@%":
                d.text((ox + c * cell + cell * 0.3, oy + r * cell + cell * 0.15), ch, font=font(int(cell * 0.55), True), fill=TEXT)
    frac = min(1.0, (t * len(inst["puzzles"]) - k) * 1.3)
    n = max(1, int(len(trace) * frac))
    r, c, f, _ = trace[n - 1]
    x0, y0 = ox + c * cell, oy + r * cell
    d.rectangle([x0 + 3, y0 + 3, x0 + cell - 5, y0 + cell - 5], fill=CUBE_COLORS.get(f["top"], (200, 200, 200)), outline=(255, 255, 255))
    # развёртка граней
    nx0, ny0 = int(W * 0.56), int(52 * s)
    q = int(22 * s)
    layout = {"north": (1, 0), "west": (0, 1), "top": (1, 1), "east": (2, 1), "south": (1, 2), "bottom": (3, 1)}
    for face, (gx, gy) in layout.items():
        d.rectangle([nx0 + gx * q, ny0 + gy * q, nx0 + gx * q + q - 2, ny0 + gy * q + q - 2], fill=CUBE_COLORS.get(f[face], (128, 128, 128)))
    d.text((nx0, ny0 + 3 * q + 4), f"ход {n - 1} из {len(moves)}", font=font(int(10 * s)), fill=DIM)
    y = int(H * 0.62)
    for i, it in enumerate(items):
        col = GOOD if it["ok"] >= 0.999 else (MID if it["ok"] > 0 else BAD)
        d.text((nx0, y), ("▶ " if i == k else "  ") + _shorten(d, it["label"], font(int(9 * s)), W * 0.3), font=font(int(9 * s), i == k), fill=TEXT if i == k else DIM)
        d.text((W - int(36 * s), y), pct(it["ok"]), font=font(int(9 * s), True), fill=col)
        y += int(16 * s)
    return im


def channel_matryoshka(inst, res, t, W=480, H=270):
    """Шесть вложенных слоёв (5 → 0) раскрываются по очереди, справа — маршрут кубика слоя 0."""
    from .tasks.cube import run2
    im, d, s = _base(W, H, 4, "Матрёшка · 5 слоёв + кубик", res["score"])
    items = res["items"]
    n_show = min(6, int(t * 7.5) + 1)
    x0, y0 = int(10 * s), int(38 * s)
    bw, bh = int(W * 0.52), int(29 * s)
    for i, it in enumerate(items[:6]):
        ind = int(i * 7 * s)
        x, y = x0 + ind, y0 + i * (bh + int(4 * s))
        shown = i < n_show
        ok = it["ok"]
        col = (GOOD if ok >= 0.999 else (MID if ok > 0 else BAD)) if shown else (60, 64, 76)
        d.rounded_rectangle([x, y, x0 + bw - ind, y + bh], radius=int(9 * s), outline=col, width=max(1, int(2 * s)),
                            fill=(20, 23, 31))
        d.text((x + int(8 * s), y + int(3 * s)), it["label"], font=font(int(10 * s), True), fill=TEXT if shown else DIM)
        if shown:
            got = it.get("got")
            g = (got if isinstance(got, str) else ("—" if got is None else f"{got:,}".replace(",", " ")))
            d.text((x + int(8 * s), y + int(15 * s)), _shorten(d, f"ответ: {g}", font(int(9 * s)), bw - ind - 60 * s),
                   font=font(int(9 * s)), fill=DIM)
            mark = "✓" if ok >= 0.999 else ("½" if ok > 0 else "✗")
            d.text((x0 + bw - ind - int(20 * s), y + int(6 * s)), mark, font=font(int(14 * s), True), fill=col)
    # кубик слоя 0
    board = inst["board"]
    mv = items[5].get("got") or "" if len(items) > 5 else ""
    _, _, trace = run2(board, mv)
    Hh, Ww = len(board), len(board[0])
    cell = int(min((W * 0.4) / Ww, (H - 70 * s) / Hh))
    ox, oy = int(W * 0.58), int(40 * s)
    for r in range(Hh):
        for c in range(Ww):
            ch = board[r][c]
            colc = (10, 10, 12) if ch == "#" else ((60, 130, 70) if ch == "G" else (40, 44, 55))
            d.rectangle([ox + c * cell, oy + r * cell, ox + (c + 1) * cell - 2, oy + (r + 1) * cell - 2], fill=colc)
            if ch in "@%S":
                d.text((ox + c * cell + cell * 0.28, oy + r * cell + cell * 0.12), ch, font=font(int(cell * 0.55), True), fill=TEXT)
    frac = max(0.0, min(1.0, (t - 0.55) / 0.4))
    k = max(1, int(len(trace) * frac)) if n_show >= 6 else 1
    pts = [(ox + c * cell + cell / 2, oy + r * cell + cell / 2) for r, c, _, _ in trace[:k]]
    if len(pts) > 1:
        d.line(pts, fill=ACCENT, width=max(2, int(2 * s)))
    r, c, f, _ = trace[k - 1]
    d.rectangle([ox + c * cell + 3, oy + r * cell + 3, ox + (c + 1) * cell - 5, oy + (r + 1) * cell - 5],
                fill=CUBE_COLORS.get(f["top"], (200, 200, 200)), outline=(255, 255, 255))
    summ = items[0].get("summary", {}) if items else {}
    d.text((ox, oy + Hh * cell + int(6 * s)),
           f"по слоям {pct(summ.get('by_layer', 0))} · цепочка {summ.get('chain', 0)}/6",
           font=font(int(10 * s), True), fill=TEXT)
    return im


def radar16(im, cx, cy, R, items, s):
    n = len(items)
    d = ImageDraw.Draw(im)
    f = font(max(8, int(8.5 * s)))
    for k in (0.25, 0.5, 0.75, 1.0):
        pts = [(cx + R * k * math.sin(2 * math.pi * i / n), cy - R * k * math.cos(2 * math.pi * i / n)) for i in range(n)]
        d.polygon(pts, outline=(55, 60, 72))
    for i, (label, v) in enumerate(items):
        a = 2 * math.pi * i / n
        d.line([cx, cy, cx + R * math.sin(a), cy - R * math.cos(a)], fill=(55, 60, 72))
        lx, ly = cx + (R + 6 * s) * math.sin(a), cy - (R + 6 * s) * math.cos(a)
        tw = d.textlength(label, font=f)
        sa = math.sin(a)
        ax = lx - tw / 2 + (tw / 2 + 2 * s) * (1 if sa > 0.2 else -1 if sa < -0.2 else 0)
        ay = ly - 5 * s + (-5 * s if math.cos(a) > 0.9 else 5 * s if math.cos(a) < -0.9 else 0)
        d.text((ax, ay), label, font=f, fill=TEXT if v is not None else DIM)
    pts = [(cx + R * (v or 0) * math.sin(2 * math.pi * i / n), cy - R * (v or 0) * math.cos(2 * math.pi * i / n))
           for i, (_, v) in enumerate(items)]
    ov = Image.new("RGBA", im.size, (0, 0, 0, 0))
    ImageDraw.Draw(ov).polygon(pts, fill=(232, 163, 61, 80))
    base = im.convert("RGBA")
    base.alpha_composite(ov)
    d2 = ImageDraw.Draw(base)
    d2.line(pts + [pts[0]], fill=ACCENT, width=max(1, int(2 * s)))
    for (x, y), (_, v) in zip(pts, items):
        if v is not None:
            r = 2.5 * s
            d2.ellipse([x - r, y - r, x + r, y + r], fill=score_color(v))
    return base.convert("RGB")


def channel_summary(summary, t, W=480, H=270):
    im, d, s = _base(W, H, 5, "Итоги бенчмарка", summary["overall"])
    items = [(name, summary["abilities"].get(key)) for key, name in C.ABILITIES]
    grow = min(1.0, 0.15 + t * 1.5)
    shown = [(n, (v * grow if v is not None else None)) for n, v in items]
    im = radar16(im, int(W * 0.27), int(H * 0.58), int(H * 0.27), shown, s)
    d = ImageDraw.Draw(im)
    x0, y = int(W * 0.56), int(38 * s)
    bw = int(W * 0.40)
    for name, v in shown:
        d.text((x0, y), name, font=font(int(9 * s)), fill=TEXT if v is not None else DIM)
        tv = pct(v)
        d.text((x0 + bw - d.textlength(tv, font=font(int(9 * s), True)), y), tv, font=font(int(9 * s), True), fill=score_color(v))
        yb = y + int(12 * s)
        d.rectangle([x0, yb, x0 + bw, yb + int(3 * s)], fill=PANEL)
        if v:
            d.rectangle([x0, yb, x0 + int(bw * v), yb + int(3 * s)], fill=score_color(v))
        y += int(18.5 * s)
    return im


# ---------------------------------------------------------------------------
# карточки ролика 16:9
# ---------------------------------------------------------------------------

def overlay16(im, model, caption, score=None, step=""):
    W, H = im.size
    s = W / 640
    im = im.convert("RGBA")
    ov = Image.new("RGBA", im.size, (0, 0, 0, 0))
    d = ImageDraw.Draw(ov)
    th = int(24 * s)
    d.rectangle([0, 0, W, th], fill=(10, 12, 16, 160))
    d.text((int(10 * s), int(5 * s)), "llm-quick-bench · Дом-3D Pro", font=font(int(12 * s), True), fill=ACCENT)
    f2 = font(int(12 * s))
    d.text((W - d.textlength(model, font=f2) - int(10 * s), int(5 * s)), model, font=f2, fill=TEXT)
    if caption:
        bh = int(34 * s)
        d.rectangle([0, H - bh, W, H], fill=(10, 12, 16, 180))
        x = int(10 * s)
        if step:
            fs = font(int(11 * s), True)
            tw = d.textlength(step, font=fs) + int(12 * s)
            d.rounded_rectangle([x, H - bh + int(8 * s), x + tw, H - int(8 * s)], radius=int(5 * s), fill=ACCENT)
            d.text((x + int(6 * s), H - bh + int(10 * s)), step, font=fs, fill=BG)
            x += tw + int(10 * s)
        fc = font(int(15 * s), True)
        d.text((x, H - bh + int(7 * s)), _shorten(d, caption, fc, W - x - 80 * s), font=fc, fill=TEXT)
        if score is not None:
            fsc = font(int(18 * s), True)
            t = pct(score)
            d.text((W - d.textlength(t, font=fsc) - int(12 * s), H - bh + int(5 * s)), t, font=fsc, fill=score_color(score))
    return Image.alpha_composite(im, ov).convert("RGB")


def title16(W, H, model, date, lines):
    s = W / 640
    im = Image.new("RGB", (W, H), BG)
    d = ImageDraw.Draw(im)
    d.text((int(40 * s), int(70 * s)), "llm-quick-bench", font=font(int(18 * s), True), fill=ACCENT)
    d.text((int(40 * s), int(96 * s)), "Дом-3D Pro", font=font(int(46 * s), True), fill=TEXT)
    d.rectangle([int(40 * s), int(152 * s), int(120 * s), int(155 * s)], fill=ACCENT)
    d.text((int(40 * s), int(166 * s)), _shorten(d, model, font(int(24 * s), True), W - 80 * s), font=font(int(24 * s), True), fill=TEXT)
    d.text((int(40 * s), int(200 * s)), date, font=font(int(13 * s)), fill=DIM)
    y = int(236 * s)
    for line in lines:
        d.text((int(40 * s), y), line, font=font(int(12 * s)), fill=DIM)
        y += int(19 * s)
    return im


def checks16(W, H, model, groups):
    s = W / 640
    im = Image.new("RGB", (W, H), BG)
    d = ImageDraw.Draw(im)
    rows = []
    for title, cs in groups:
        rows.append(("h", title))
        rows += [("c", c) for c in cs]
    ncol = 3
    per = math.ceil(len(rows) / ncol)
    lh = (H - int(70 * s)) / max(per, 1)
    lh = min(lh, 15 * s)
    cw = (W - int(24 * s)) / ncol
    f, fb = font(max(8, int(9 * s))), font(max(8, int(9 * s)), True)
    for i, (kind, item) in enumerate(rows):
        col, row = divmod(i, per)
        x, y = int(12 * s + col * cw), int(30 * s + row * lh)
        if kind == "h":
            d.text((x, y), item, font=fb, fill=ACCENT)
            continue
        ok = item.passed if hasattr(item, "passed") else item.get("ok", 0) >= 0.999
        sc = item.score if hasattr(item, "score") else item.get("ok", 0)
        label = f"{item.id} {item.name}" if hasattr(item, "id") else item.get("label", "")
        mark, colr = ("✓", GOOD) if ok else ("✗", BAD)
        d.text((x, y), mark, font=fb, fill=colr)
        d.text((x + int(12 * s), y), _shorten(d, label, f, cw - int(46 * s)), font=f, fill=TEXT)
        t = pct(sc)
        d.text((x + cw - int(12 * s) - d.textlength(t, font=f), y), t, font=f, fill=colr if not ok else DIM)
    return overlay16(im, model, "Все проверки", None, "ДЕТАЛИ")
