"""Эталонный детальный дом для «Дом-3D Pro» (≈11–12 тыс. примитивов в DSL v2).

Строится программно и проходит все проверки валидатора. Используется mock-моделью,
тестами и как доказательство решаемости задания.
"""
from __future__ import annotations

import copy
import json
import math
import random

from . import config as C

T = 0.30   # толщина наружных стен
P = 0.12   # толщина перегородок
SLAB_B = 3.20  # низ перекрытия 2 этажа

WALL = "#efe7da"
TRIM = "#f7f4ee"
SIDING = "#e4dccd"
SIDING2 = "#d9cfbd"
GLASS = "#8fc3e6"
WOOD = "#7a4a2a"
WOOD_L = "#b08356"
FRAME = "#f2f0eb"
ROOF = "#8f3a26"
ROOF_D = "#6e2c1d"
CONCRETE = "#a19d96"
SLAB = "#c8c0b4"
PLANK = "#b98a5c"
PLANK2 = "#a97b50"
TILE_W = "#dfe3e6"
TILE_K = "#cfc7bb"
STAIR = "#9a6c42"
RAIL = "#3e434b"
METAL = "#8a9099"
BRICK = "#8a4031"
GRASS_D = "#4f8a3c"
GRASS_L = "#6aa84f"
TRUNK = "#6b4a2e"
FENCE = "#e9e4da"
PAVER = "#bdb6a8"
FABRIC = "#5d6f86"
FABRIC2 = "#8a5a6a"
LINEN = "#e8e2d6"
BLACK = "#121417"


def box(tag, x0, x1, y0, y1, z0, z1, color, oid=None, **kw):
    d = {"type": "box", "tag": tag,
         "center": [round((x0 + x1) / 2, 4), round((y0 + y1) / 2, 4), round((z0 + z1) / 2, 4)],
         "size": [round(x1 - x0, 4), round(y1 - y0, 4), round(z1 - z0, 4)], "color": color}
    if oid:
        d["id"] = oid
    d.update(kw)
    return d


def prim(typ, tag, center, size, color, oid=None, **kw):
    d = {"type": typ, "tag": tag, "center": [round(v, 4) for v in center], "size": [round(v, 4) for v in size],
         "color": color}
    if oid:
        d["id"] = oid
    d.update(kw)
    return d


class Builder:
    def __init__(self):
        self.components = {}
        self.objects = []
        self.instances = []
        self.rooms = []

    def comp(self, name, objects, instances=None):
        self.components[name] = {"objects": objects}
        if instances:
            self.components[name]["instances"] = instances
        return name

    def inst(self, use, at, rot=0, array=None, grid=None, iid=None):
        d = {"use": use, "at": [round(v, 4) for v in at]}
        if rot:
            d["rot"] = rot
        if array:
            d["array"] = array
        if grid:
            d["grid"] = grid
        if iid:
            d["id"] = iid
        self.instances.append(d)

    def obj(self, d):
        self.objects.append(d)

    def room(self, rid, typ, level, rects):
        self.rooms.append({"id": rid, "type": typ, "level": level, "rects": rects})


# ---------------------------------------------------------------------------
# стены с проёмами
# ---------------------------------------------------------------------------

def wall_pieces(axis, a0, a1, b0, b1, z0, z1, openings, color, tag="wall"):
    """Стена вдоль оси axis ('x' или 'y'): по оси — a0..a1, толщина — b0..b1.
    openings: [(o0, o1, oz0, oz1)] — реальные проёмы. -> список боксов."""
    cuts = sorted({a0, a1} | {v for o in openings for v in (o[0], o[1])})
    out = []
    for u, v in zip(cuts, cuts[1:]):
        if v - u < 1e-6:
            continue
        ops = [o for o in openings if o[0] <= u + 1e-6 and o[1] >= v - 1e-6]
        bands = [(z0, z1)] if not ops else [(z0, ops[0][2]), (ops[0][3], z1)]
        for za, zb in bands:
            if zb - za < 1e-6:
                continue
            if axis == "x":
                out.append(box(tag, u, v, b0, b1, za, zb, color))
            else:
                out.append(box(tag, b0, b1, u, v, za, zb, color))
    return out


def siding_pieces(axis, a0, a1, face, outward, z0, z1, openings, rows_h=0.16):
    """Доски обшивки снаружи стены (толщина 0.02), с разрывами у проёмов."""
    out = []
    z = z0
    k = 0
    while z < z1 - 1e-6:
        zb = min(z + rows_h, z1)
        # интервалы, где ряд не пересекает проём
        blocks = sorted((o[0], o[1]) for o in openings if o[2] < zb - 1e-6 and o[3] > z + 1e-6)
        segs, cur = [], a0
        for b0, b1 in blocks:
            if b0 > cur + 1e-6:
                segs.append((cur, b0))
            cur = max(cur, b1)
        if a1 > cur + 1e-6:
            segs.append((cur, a1))
        col = SIDING if k % 2 == 0 else SIDING2
        for s0, s1 in segs:
            if s1 - s0 < 0.05:
                continue
            if axis == "x":
                y0, y1 = (face, face + 0.02) if outward > 0 else (face - 0.02, face)
                out.append(box("decor", s0, s1, y0, y1, z + 0.004, zb - 0.004, col))
            else:
                x0, x1 = (face, face + 0.02) if outward > 0 else (face - 0.02, face)
                out.append(box("decor", x0, x1, s0, s1, z + 0.004, zb - 0.004, col))
        z = zb
        k += 1
    return out


def brick_fill(B, axis, face, outward, regions):
    """Кирпичная облицовка: массивы кирпичей по прямоугольникам стены без проёмов."""
    for (u, v, z0, z1) in regions:
        if v - u < 0.3 or z1 - z0 < 0.1:
            continue
        rows = int((z1 - z0) // 0.08)
        if rows <= 0:
            continue
        off = face + outward * 0.015
        for par in (0, 1):
            shift = 0.125 * par
            n = int((v - u - shift) // 0.25)
            m = (rows - par + 1) // 2
            if n <= 0 or m <= 0:
                continue
            zst = z0 + 0.04 + 0.08 * par
            if axis == "x":
                B.inst("brick_x" if par == 0 else "brick_x2", [u + shift + 0.125, off, zst],
                       grid={"count": [n, m], "step": [[0.25, 0, 0], [0, 0, 0.16]]})
            else:
                B.inst("brick_y" if par == 0 else "brick_y2", [off, u + shift + 0.125, zst],
                       grid={"count": [n, m], "step": [[0, 0.25, 0], [0, 0, 0.16]]})


def wall_regions(a0, a1, z0, z1, openings):
    cuts = sorted({a0, a1} | {v for o in openings for v in (o[0], o[1])})
    out = []
    for u, v in zip(cuts, cuts[1:]):
        if v - u < 1e-6:
            continue
        ops = [o for o in openings if o[0] <= u + 1e-6 and o[1] >= v - 1e-6]
        bands = [(z0, z1)] if not ops else [(z0, ops[0][2]), (ops[0][3], z1)]
        for za, zb in bands:
            if zb - za > 1e-6:
                out.append((u, v, za, zb))
    return out


# ---------------------------------------------------------------------------
# компоненты
# ---------------------------------------------------------------------------

def window_component(B, w, h):
    """Оконный блок: локально ширина по X, глубина по Y (наружу +Y), начало — низ проёма в центре стены."""
    name = f"win_{int(round(w * 100))}x{int(round(h * 100))}"
    if name in B.components:
        return name
    f, dpt = 0.06, 0.12
    objs = [
        prim("box", "window_frame", [-w / 2 + f / 2, 0, h / 2], [f, dpt, h], FRAME),
        prim("box", "window_frame", [w / 2 - f / 2, 0, h / 2], [f, dpt, h], FRAME),
        prim("box", "window_frame", [0, 0, f / 2], [w - 2 * f, dpt, f], FRAME),
        prim("box", "window_frame", [0, 0, h - f / 2], [w - 2 * f, dpt, f], FRAME),
        prim("box", "glass", [0, 0, h / 2], [w - 2 * f, 0.02, h - 2 * f], GLASS),
        prim("box", "sill", [0, T / 2 + 0.03, -0.02], [w + 0.1, 0.12, 0.04], TRIM),
    ]
    if w >= 1.5:
        objs.append(prim("box", "window_frame", [0, 0.012, h / 2], [0.05, 0.03, h - 2 * f], FRAME))
    if h >= 1.3:
        objs.append(prim("box", "window_frame", [0, 0.012, h * 0.62], [w - 2 * f, 0.03, 0.04], FRAME))
    return B.comp(name, objs)


def place_window(B, fac, along, w, z0, h, level_ffl):
    name = window_component(B, w, h)
    if fac == "S":
        B.inst(name, [along, T / 2, z0], rot=180)
    elif fac == "N":
        B.inst(name, [along, C.Y1 - T / 2, z0], rot=0)
    elif fac == "W":
        B.inst(name, [T / 2, along, z0], rot=90)
    else:
        B.inst(name, [C.X1 - T / 2, along, z0], rot=-90)


def door_frame_component(B, w, h, depth, name):
    if name in B.components:
        return name
    f = 0.05
    objs = [
        prim("box", "door_frame", [-w / 2 + f / 2, 0, h / 2], [f, depth, h], FRAME),
        prim("box", "door_frame", [w / 2 - f / 2, 0, h / 2], [f, depth, h], FRAME),
        prim("box", "door_frame", [0, 0, h - f / 2], [w - 2 * f, depth, f], FRAME),
    ]
    return B.comp(name, objs)


# ---------------------------------------------------------------------------
# мебель (локально: начало — центр основания)
# ---------------------------------------------------------------------------

def furniture_components(B):
    leg = lambda x, y, hgt, d=0.05, col=WOOD: prim("cylinder", "furniture", [x, y, hgt / 2], [d, d, hgt], col)  # noqa: E731
    # кровать двуспальная 1.8 × 2.1 (длина по X), изголовье на −X
    def bed(name, wdt, lng, blanket):
        objs = [
            prim("box", "bed", [0, 0, 0.27], [lng, wdt, 0.18], WOOD_L),
            prim("box", "bed", [0.03, 0, 0.44], [lng - 0.06, wdt - 0.06, 0.16], LINEN),
            prim("box", "bed", [-lng / 2 + 0.04, 0, 0.55], [0.08, wdt, 1.0], WOOD),
            prim("box", "bed", [0.25, 0, 0.53], [lng - 0.6, wdt - 0.02, 0.04], blanket),
        ]
        for yy in ([-wdt / 4, wdt / 4] if wdt > 1.2 else [0]):
            objs.append(prim("box", "bed", [-lng / 2 + 0.32, yy, 0.56], [0.4, min(0.6, wdt / 2 - 0.06), 0.08], "#f4f1ea"))
        for sx in (-1, 1):
            for sy in (-1, 1):
                objs.append(prim("cylinder", "bed", [sx * (lng / 2 - 0.08), sy * (wdt / 2 - 0.08), 0.09], [0.06, 0.06, 0.18], WOOD))
        return B.comp(name, objs)
    bed("bed_double", 1.8, 2.1, "#7d8fb3")
    bed("bed_single", 1.4, 2.0, "#9b7bb0")
    bed("bed_kid", 0.9, 2.0, "#e0a050")

    # диван 2.2 × 0.9, спинка на −Y, сидящий смотрит на +Y
    objs = [prim("box", "sofa", [0, 0, 0.25], [2.2, 0.9, 0.3], FABRIC),
            prim("box", "sofa", [0, -0.35, 0.6], [2.0, 0.2, 0.5], FABRIC),
            prim("box", "sofa", [-1.0, 0.0, 0.5], [0.2, 0.9, 0.3], FABRIC),
            prim("box", "sofa", [1.0, 0.0, 0.5], [0.2, 0.9, 0.3], FABRIC)]
    for i in range(3):
        objs.append(prim("box", "sofa", [-0.6 + 0.6 * i, 0.08, 0.45], [0.58, 0.66, 0.1], "#6f84a1"))
        objs.append(prim("box", "sofa", [-0.6 + 0.6 * i, -0.2, 0.7], [0.56, 0.12, 0.36], "#6f84a1"))
    for sx in (-1, 1):
        for sy in (-1, 1):
            objs.append(prim("cylinder", "sofa", [sx * 1.0, sy * 0.38, 0.05], [0.05, 0.05, 0.1], WOOD))
    B.comp("sofa", objs)

    # стол журнальный 1.0 × 0.6
    objs = [prim("box", "table", [0, 0, 0.42], [1.0, 0.6, 0.04], WOOD_L)]
    objs += [leg(sx * 0.44, sy * 0.24, 0.4) for sx in (-1, 1) for sy in (-1, 1)]
    for o in objs[1:]:
        o["tag"] = "table"
    B.comp("coffee_table", objs)

    # обеденный стол 1.8 × 0.9
    objs = [prim("box", "table", [0, 0, 0.74], [1.8, 0.9, 0.04], WOOD_L)]
    objs += [prim("box", "table", [sx * 0.82, sy * 0.38, 0.36], [0.06, 0.06, 0.72], WOOD) for sx in (-1, 1) for sy in (-1, 1)]
    B.comp("dining_table", objs)

    # стул: сиденье 0.45, спинка на −Y
    objs = [prim("box", "chair", [0, 0, 0.45], [0.44, 0.44, 0.04], WOOD_L),
            prim("box", "chair", [0, -0.2, 0.7], [0.44, 0.04, 0.46], WOOD_L)]
    objs += [prim("box", "chair", [sx * 0.19, sy * 0.19, 0.215], [0.04, 0.04, 0.43], WOOD) for sx in (-1, 1) for sy in (-1, 1)]
    B.comp("chair", objs)

    # кухонный модуль 0.6 × 0.6 × 0.9 (фасад на +Y)
    objs = [prim("box", "counter", [0, 0, 0.43], [0.6, 0.58, 0.86], "#e9e5dc"),
            prim("box", "counter", [0, 0.295, 0.45], [0.56, 0.02, 0.74], "#cfd6d9"),
            prim("box", "decor", [0, 0.31, 0.7], [0.18, 0.02, 0.02], METAL),
            prim("box", "counter", [0, 0.01, 0.88], [0.6, 0.62, 0.04], "#59524a")]
    B.comp("counter_unit", objs)
    objs = [prim("box", "shelf", [0, 0, 0.35], [0.6, 0.35, 0.7], "#e9e5dc"),
            prim("box", "shelf", [0, 0.18, 0.35], [0.56, 0.02, 0.62], "#cfd6d9"),
            prim("box", "decor", [0, 0.195, 0.08], [0.18, 0.02, 0.02], METAL)]
    B.comp("upper_cabinet", objs)

    # плита: на модуле
    objs = [prim("box", "stove", [0, 0, 0.005], [0.58, 0.56, 0.01], "#24272b")]
    objs += [prim("cylinder", "stove", [sx * 0.14, sy * 0.13, 0.015], [0.18, 0.18, 0.01], "#3a3e44") for sx in (-1, 1) for sy in (-1, 1)]
    B.comp("stove", objs)
    objs = [prim("box", "sink", [0, 0, 0.005], [0.5, 0.42, 0.01], "#c7ccd1"),
            prim("box", "sink", [0, 0, 0.012], [0.42, 0.34, 0.004], "#5e666e"),
            prim("cylinder", "sink", [0, -0.18, 0.16], [0.03, 0.03, 0.3], METAL)]
    B.comp("kitchen_sink", objs)
    objs = [prim("box", "fridge", [0, 0, 0.95], [0.62, 0.62, 1.9], "#e8ebee"),
            prim("box", "decor", [0.2, 0.32, 1.2], [0.03, 0.03, 0.5], METAL),
            prim("box", "decor", [0.2, 0.32, 0.45], [0.03, 0.03, 0.3], METAL)]
    B.comp("fridge", objs)

    # стеллаж 1.2 × 0.3 × 2.0 с книгами (спиной к −Y)
    objs = [prim("box", "shelf", [-0.59, 0, 1.0], [0.02, 0.3, 2.0], WOOD_L),
            prim("box", "shelf", [0.59, 0, 1.0], [0.02, 0.3, 2.0], WOOD_L),
            prim("box", "shelf", [0, -0.14, 1.0], [1.16, 0.02, 2.0], WOOD_L)]
    shelf_z = [0.02, 0.42, 0.82, 1.22, 1.62, 1.98]
    for z in shelf_z:
        objs.append(prim("box", "shelf", [0, 0, z], [1.16, 0.3, 0.02], WOOD_L))
    book_cols = ["#8c2f39", "#2f5d8c", "#3c7a4b", "#c9a227", "#6b4c8c", "#d06a3a", "#2d2d2d", "#7a8c99"]
    rng = random.Random(4)
    books = []
    for z in shelf_z[:-1]:
        x = -0.56
        while x < 0.52:
            bw = rng.choice([0.03, 0.035, 0.04, 0.045])
            bh = rng.choice([0.24, 0.27, 0.3, 0.33])
            books.append(prim("box", "decor", [x + bw / 2, 0.0, z + 0.01 + bh / 2], [bw, 0.22, bh], rng.choice(book_cols)))
            x += bw + 0.004
    B.comp("bookshelf", objs + books)

    objs = [prim("box", "wardrobe", [0, 0, 1.05], [1.2, 0.6, 2.1], "#d8cfc0"),
            prim("box", "decor", [-0.3, 0.305, 1.05], [0.58, 0.01, 2.0], "#cfc4b2"),
            prim("box", "decor", [0.3, 0.305, 1.05], [0.58, 0.01, 2.0], "#cfc4b2"),
            prim("box", "decor", [-0.04, 0.315, 1.1], [0.02, 0.02, 0.3], METAL),
            prim("box", "decor", [0.04, 0.315, 1.1], [0.02, 0.02, 0.3], METAL)]
    B.comp("wardrobe", objs)

    objs = [prim("box", "table", [0, 0, 0.74], [1.3, 0.65, 0.03], WOOD_L)]
    objs += [prim("box", "table", [sx * 0.62, 0, 0.365], [0.04, 0.6, 0.73], WOOD) for sx in (-1, 1)]
    objs.append(prim("box", "decor", [0.3, -0.1, 0.9], [0.5, 0.03, 0.3], BLACK))
    objs.append(prim("box", "decor", [0.3, -0.08, 0.77], [0.05, 0.05, 0.03], METAL))
    B.comp("desk", objs)

    # санфаянс
    objs = [prim("box", "toilet", [0, 0.05, 0.2], [0.38, 0.5, 0.4], "#f5f6f7"),
            prim("box", "toilet", [0, -0.24, 0.6], [0.4, 0.18, 0.4], "#f5f6f7"),
            prim("box", "toilet", [0, 0.05, 0.42], [0.38, 0.46, 0.04], "#e9ebee")]
    B.comp("toilet", objs)
    objs = [prim("cylinder", "sink", [0, 0, 0.4], [0.16, 0.16, 0.8], "#f5f6f7"),
            prim("box", "sink", [0, 0.05, 0.83], [0.6, 0.45, 0.06], "#f5f6f7"),
            prim("cylinder", "sink", [0, -0.12, 0.95], [0.03, 0.03, 0.18], METAL),
            prim("box", "decor", [0, -0.255, 1.4], [0.5, 0.02, 0.7], "#cfe3ee")]
    B.comp("bath_sink", objs)
    objs = [prim("box", "shower", [0, 0, 0.01], [1.2, 1.2, 0.02], "#e4e7ea"),
            prim("box", "glass", [0.6, 0, 1.0], [0.01, 1.2, 1.96], "#bfe0f0"),
            prim("cylinder", "shower", [-0.55, -0.55, 1.05], [0.03, 0.03, 2.1], METAL),
            prim("cylinder", "shower", [-0.48, -0.48, 2.08], [0.2, 0.2, 0.03], METAL)]
    B.comp("shower", objs)
    objs = [prim("box", "bathtub", [0, 0, 0.28], [1.7, 0.75, 0.56], "#f5f6f7"),
            prim("box", "bathtub", [0, 0, 0.5], [1.5, 0.55, 0.12], "#d8eef7"),
            prim("cylinder", "bathtub", [-0.7, -0.3, 0.68], [0.04, 0.04, 0.24], METAL)]
    B.comp("bathtub", objs)

    # торшер, растение в горшке, подвесной светильник
    objs = [prim("cylinder", "decor", [0, 0, 0.02], [0.3, 0.3, 0.04], METAL),
            prim("cylinder", "decor", [0, 0, 0.82], [0.03, 0.03, 1.56], METAL),
            prim("cylinder", "decor", [0, 0, 1.68], [0.4, 0.4, 0.3], "#f3e3c3")]
    B.comp("floor_lamp", objs)
    objs = [prim("cylinder", "decor", [0, 0, 0.18], [0.36, 0.36, 0.36], "#b0603a")]
    prng = random.Random(9)
    for k in range(7):
        a = 2 * math.pi * k / 7
        objs.append(prim("sphere", "plant", [0.1 * math.cos(a), 0.1 * math.sin(a), 0.5 + 0.08 * (k % 3)], [0.26, 0.26, 0.26],
                         prng.choice([GRASS_D, GRASS_L, "#3f7a35"])))
    objs.append(prim("sphere", "plant", [0, 0, 0.72], [0.3, 0.3, 0.3], GRASS_L))
    B.comp("plant", objs)
    objs = [prim("cylinder", "decor", [0, 0, -0.35], [0.015, 0.015, 0.7], BLACK),
            prim("cylinder", "lamp", [0, 0, -0.82], [0.42, 0.42, 0.24], "#f2e6c9")]
    B.comp("pendant", objs)

    # камин
    objs = [prim("box", "fireplace", [0, 0, 0.55], [1.0, 1.1, 1.1], "#8c8378"),
            prim("box", "decor", [0.505, 0, 0.4], [0.02, 0.6, 0.5], "#2a1d17"),
            prim("box", "decor", [0.08, 0, 1.13], [1.16, 1.14, 0.06], "#b8afa3")]
    B.comp("fireplace", objs)

    # тумба под ТВ
    objs = [prim("box", "furniture", [0, 0, 0.25], [1.6, 0.4, 0.5], "#3b3f45"),
            prim("box", "decor", [-0.4, 0.205, 0.25], [0.76, 0.01, 0.44], "#4a4f56"),
            prim("box", "decor", [0.4, 0.205, 0.25], [0.76, 0.01, 0.44], "#4a4f56")]
    B.comp("tv_console", objs)

    # ковёр, картина
    B.comp("rug_big", [prim("box", "rug", [0, 0, 0.005], [2.6, 1.8, 0.01], "#a86b5a")])
    B.comp("rug_small", [prim("box", "rug", [0, 0, 0.005], [1.6, 1.0, 0.01], "#7b8fa8")])

    # садовое: дерево, цветок, пикет, плитка
    def tree(name, h, seed):
        r = random.Random(seed)
        objs = [prim("cylinder", "tree", [0, 0, h * 0.3], [0.26, 0.26, h * 0.6], TRUNK)]
        for k in range(14):
            a, rr = r.uniform(0, 2 * math.pi), r.uniform(0.2, 0.9)
            zz = h * 0.6 + r.uniform(0.0, 1.4)
            objs.append(prim("sphere", "tree", [rr * math.cos(a), rr * math.sin(a), zz], [1.2 + r.uniform(0, 0.6)] * 3,
                             r.choice([GRASS_D, "#45803a", "#5b9a45"])))
        return B.comp(name, objs)
    tree("tree_a", 4.0, 1)
    tree("tree_b", 3.2, 2)
    B.comp("flower_r", [prim("cylinder", "plant", [0, 0, 0.15], [0.02, 0.02, 0.3], "#3f7a35"),
                        prim("sphere", "plant", [0, 0, 0.33], [0.1, 0.1, 0.1], "#d9435a")])
    B.comp("flower_y", [prim("cylinder", "plant", [0, 0, 0.13], [0.02, 0.02, 0.26], "#3f7a35"),
                        prim("sphere", "plant", [0, 0, 0.29], [0.1, 0.1, 0.1], "#f0c419")])
    B.comp("flower_v", [prim("cylinder", "plant", [0, 0, 0.17], [0.02, 0.02, 0.34], "#3f7a35"),
                        prim("sphere", "plant", [0, 0, 0.37], [0.1, 0.1, 0.1], "#8e5cc6")])
    B.comp("picket", [prim("box", "fence", [0, 0, 0.55], [0.08, 0.03, 1.1], FENCE)])
    B.comp("fence_post", [prim("box", "fence", [0, 0, 0.65], [0.12, 0.12, 1.3], "#d6d0c4")])
    B.comp("paver", [prim("box", "paving", [0, 0, 0.015], [0.42, 0.42, 0.03], PAVER)])
    B.comp("paver_dark", [prim("box", "paving", [0, 0, 0.015], [0.42, 0.42, 0.03], "#a8a091")])
    B.comp("plank_x", [prim("box", "rug", [0, 0, 0.006], [1.98, 0.118, 0.012], PLANK)])
    B.comp("plank_x2", [prim("box", "rug", [0, 0, 0.006], [1.98, 0.118, 0.012], PLANK2)])
    B.comp("floor_tile", [prim("box", "rug", [0, 0, 0.005], [0.296, 0.296, 0.01], TILE_W)])
    B.comp("floor_tile_k", [prim("box", "rug", [0, 0, 0.005], [0.396, 0.396, 0.01], TILE_K)])
    B.comp("veg", [prim("sphere", "plant", [0, 0, 0.12], [0.26, 0.26, 0.26], "#5aa043")])
    B.comp("brick_x", [prim("box", "decor", [0, 0, 0], [0.24, 0.03, 0.07], "#a4553f")])
    B.comp("brick_x2", [prim("box", "decor", [0, 0, 0], [0.24, 0.03, 0.07], "#94493a")])
    B.comp("brick_y", [prim("box", "decor", [0, 0, 0], [0.03, 0.24, 0.07], "#a4553f")])
    B.comp("brick_y2", [prim("box", "decor", [0, 0, 0], [0.03, 0.24, 0.07], "#94493a")])
    B.comp("bench", [prim("box", "furniture", [0, 0, 0.45], [1.5, 0.45, 0.05], WOOD_L),
                     prim("box", "furniture", [0, -0.2, 0.7], [1.5, 0.05, 0.45], WOOD_L),
                     prim("box", "furniture", [-0.65, 0, 0.215], [0.06, 0.4, 0.43], RAIL),
                     prim("box", "furniture", [0.65, 0, 0.215], [0.06, 0.4, 0.43], RAIL)])
    B.comp("garden_lamp", [prim("cylinder", "decor", [0, 0, 0.5], [0.06, 0.06, 1.0], RAIL),
                           prim("sphere", "lamp", [0, 0, 1.08], [0.2, 0.2, 0.2], "#fff3c9")])


# ---------------------------------------------------------------------------
# дом
# ---------------------------------------------------------------------------

OPEN = {  # (фасад, этаж) -> [(вдоль0, вдоль1, z0, z1, тип)]
    ("S", 0): [(0.5, 2.5, 1.25, 2.85, "w"), (3.5, 5.5, 1.25, 2.85, "w"), (9.0, 10.6, 1.35, 2.75, "w")],
    ("S", 1): [(0.7, 2.3, 4.35, 5.75, "w"), (3.5, 4.4, 3.45, 5.65, "d"), (5.15, 6.05, 4.35, 5.75, "w"),
               (8.1, 9.7, 4.35, 5.75, "w"), (10.6, 12.2, 4.35, 5.75, "w")],
    ("N", 0): [(1.6, 2.8, 1.35, 2.75, "w"), (7.85, 8.95, 0.45, 2.55, "d"), (10.55, 11.45, 1.95, 2.85, "w")],
    ("N", 1): [(1.6, 2.8, 4.35, 5.75, "w"), (7.8, 9.0, 4.35, 5.75, "w"), (10.55, 11.45, 4.95, 5.85, "w")],
    ("W", 0): [(0.9, 2.1, 1.35, 2.75, "w"), (5.4, 6.6, 1.35, 2.75, "w")],
    ("W", 1): [(3.3, 4.2, 4.35, 5.75, "w"), (6.8, 8.0, 4.35, 5.75, "w")],
    ("E", 0): [(3.4, 4.6, 1.35, 2.75, "w"), (7.55, 8.45, 1.95, 2.85, "w")],
    ("E", 1): [(1.9, 3.1, 4.35, 5.75, "w"), (7.55, 8.45, 4.95, 5.85, "w")],
}

RISERS = 17
RISE = (C.FFL1 - C.FFL0) / RISERS
TREAD = 0.28
SW = 1.1  # ширина марша


def stairs_geometry():
    """-> список (k, x0, x1, y0, y1, top) ступеней (k = 1..16), где 9 — площадка."""
    steps = []
    for k in range(1, 9):
        x1 = 7.66 - TREAD * (k - 1)
        steps.append((k, x1 - TREAD, x1, 8.6, 9.7, C.FFL0 + k * RISE))
    steps.append((9, 4.32, 5.42, 8.6, 9.7, C.FFL0 + 9 * RISE))
    for k in range(10, 17):
        y1 = 8.6 - TREAD * (k - 10)
        steps.append((k, 4.32, 5.42, y1 - TREAD, y1, C.FFL0 + k * RISE))
    return steps


def build_house():
    B = Builder()
    furniture_components(B)

    # --- фундамент, перекрытия
    B.obj(box("foundation", C.X0, C.X1, C.Y0, C.Y1, 0.0, 0.3, CONCRETE, "foundation"))
    B.obj(box("floor_slab", T, C.X1 - T, T, C.Y1 - T, 0.3, C.FFL0, SLAB, "slab0"))
    for k, (x0, y0, x1, y1) in enumerate([(T, T, C.X1 - T, 6.64), (5.42, 6.64, C.X1 - T, 8.6),
                                          (T, 6.64, 4.32, C.Y1 - T), (6.9, 8.6, C.X1 - T, C.Y1 - T)]):
        B.obj(box("floor_slab", x0, x1, y0, y1, SLAB_B, C.FFL1, SLAB, f"slab1_{k}"))

    # --- наружные стены с проёмами + обшивка
    levels = [(C.FFL0, C.FFL1), (C.FFL1, C.EAVE)]
    for lvl, (z0, z1) in enumerate(levels):
        ops = {f: [(a, b, c, d) for a, b, c, d, _ in OPEN[(f, lvl)]] for f in "SNWE"}
        for d in wall_pieces("x", C.X0, C.X1, 0.0, T, z0, z1, ops["S"], WALL):
            B.obj(d)
        for d in wall_pieces("x", C.X0, C.X1, C.Y1 - T, C.Y1, z0, z1, ops["N"], WALL):
            B.obj(d)
        for d in wall_pieces("y", T, C.Y1 - T, 0.0, T, z0, z1, ops["W"], WALL):
            B.obj(d)
        for d in wall_pieces("y", T, C.Y1 - T, C.X1 - T, C.X1, z0, z1, ops["E"], WALL):
            B.obj(d)
        if lvl == 1:
            for d in siding_pieces("x", C.X0, C.X1, 0.0, -1, z0, z1, ops["S"]):
                B.obj(d)
            for d in siding_pieces("x", C.X0, C.X1, C.Y1, 1, z0, z1, ops["N"]):
                B.obj(d)
            for d in siding_pieces("y", C.Y0 - 0.02, C.Y1 + 0.02, 0.0, -1, z0, z1, ops["W"]):
                B.obj(d)
            for d in siding_pieces("y", C.Y0 - 0.02, C.Y1 + 0.02, C.X1, 1, z0, z1, ops["E"]):
                B.obj(d)
        else:
            zb0 = 0.32
            brick_fill(B, "x", 0.0, -1, wall_regions(C.X0 + 0.02, C.X1 - 0.02, zb0, z1, ops["S"]))
            brick_fill(B, "x", C.Y1, 1, wall_regions(C.X0 + 0.02, C.X1 - 0.02, zb0, z1, ops["N"]))
            brick_fill(B, "y", 0.0, -1, wall_regions(C.Y0 + 0.02, C.Y1 - 0.02, zb0, z1, ops["W"]))
            brick_fill(B, "y", C.X1, 1, wall_regions(C.Y0 + 0.02, C.Y1 - 0.02, zb0, z1, ops["E"]))
        # окна
        for f in "SNWE":
            for a, b, c, d, kind in OPEN[(f, lvl)]:
                if kind == "w":
                    place_window(B, f, (a + b) / 2, b - a, c, d - c, z0)

    # угловые наличники
    for (x, y) in [(0, 0), (C.X1, 0), (0, C.Y1), (C.X1, C.Y1)]:
        B.obj(box("decor", x - 0.03 if x else -0.03, x + 0.03 if x else 0.03, y - 0.03 if y else -0.03,
                  y + 0.03 if y else 0.03, 0.3, C.EAVE, TRIM))

    # --- входная дверь (север) и балконная дверь (юг, 2 этаж)
    B.obj(box("door", 7.9, 8.9, C.Y1 - 0.25, C.Y1 - 0.2, C.FFL0 + 0.01, 2.5, WOOD, "door_main"))
    B.obj(box("decor", 8.75, 8.8, C.Y1 - 0.17, C.Y1 - 0.15, 1.35, 1.6, METAL, "door_main_handle"))
    dfc = door_frame_component(B, 1.1, 2.1, T, "frame_main")
    B.inst(dfc, [8.4, C.Y1 - T / 2, C.FFL0])
    B.obj(box("door", 3.55, 4.35, 0.12, 0.17, C.FFL1 + 0.01, 5.6, WOOD, "door_balcony"))
    dfb = door_frame_component(B, 0.9, 2.2, T, "frame_balcony")
    B.inst(dfb, [3.95, T / 2, C.FFL1])

    # --- перегородки
    g0, g1 = C.FFL0, SLAB_B
    parts = [
        ("x", T, C.X1 - T, 5.0, 5.12, g0, g1, [(5.9, 6.9, g0, g0 + 2.1), (8.15, 9.15, g0, g0 + 2.1)]),
        ("y", T, 5.0, 7.0, 7.12, g0, g1, [(1.0, 3.4, g0, g0 + 2.4)]),
        ("y", 5.12, C.Y1 - T, 4.2, 4.32, g0, g1, [(5.4, 6.4, g0, g0 + 2.1)]),
        ("y", 5.12, C.Y1 - T, 9.2, 9.32, g0, g1, [(6.0, 7.0, g0, g0 + 2.1)]),
        ("x", T, C.X1 - T, 5.0, 5.12, C.FFL1, C.EAVE, [(5.5, 6.4, C.FFL1, C.FFL1 + 2.1), (7.6, 8.6, C.FFL1, C.FFL1 + 2.1)]),
        ("y", T, 5.0, 6.5, 6.62, C.FFL1, C.EAVE, []),
        ("y", 5.12, C.Y1 - T, 4.2, 4.32, C.FFL1, C.EAVE, [(5.4, 6.3, C.FFL1, C.FFL1 + 2.1)]),
        ("y", 5.12, C.Y1 - T, 9.2, 9.32, C.FFL1, C.EAVE, [(6.0, 6.9, C.FFL1, C.FFL1 + 2.1)]),
    ]
    for axis, a0, a1, b0, b1, z0, z1, ops in parts:
        for d in wall_pieces(axis, a0, a1, b0, b1, z0, z1, ops, WALL):
            B.obj(d)
        for (o0, o1, oz0, oz1) in ops:
            w = o1 - o0
            name = door_frame_component(B, w, oz1 - oz0, b1 - b0, f"frame_{int(round(w * 100))}x{int(round((oz1 - oz0) * 100))}_{int(round((b1 - b0) * 100))}")
            if axis == "x":
                B.inst(name, [(o0 + o1) / 2, (b0 + b1) / 2, oz0])
            else:
                B.inst(name, [(b0 + b1) / 2, (o0 + o1) / 2, oz0], rot=90)

    # --- комнаты
    for rid, typ, lvl, rect in [
        ("living", "living", 0, [T, T, 7.0, 5.0]), ("kitchen", "kitchen", 0, [7.12, T, C.X1 - T, 5.0]),
        ("mom", "bedroom_mom", 0, [T, 5.12, 4.2, C.Y1 - T]), ("hall0", "hall", 0, [4.32, 5.12, 9.2, C.Y1 - T]),
        ("bath0", "bath_access", 0, [9.32, 5.12, C.X1 - T, C.Y1 - T]),
        ("master", "bedroom_master", 1, [T, T, 6.5, 5.0]), ("kids", "bedroom_kids", 1, [6.62, T, C.X1 - T, 5.0]),
        ("study", "study", 1, [T, 5.12, 4.2, C.Y1 - T]), ("hall1", "hall", 1, [4.32, 5.12, 9.2, C.Y1 - T]),
        ("bath1", "bath", 1, [9.32, 5.12, C.X1 - T, C.Y1 - T]),
    ]:
        B.room(rid, typ, lvl, [[round(v, 3) for v in rect]])

    # --- полы: паркет / плитка
    def planks(x0, x1, y0, y1, z, alt):
        nx = int((x1 - x0) // 2.0)
        ny = int((y1 - y0) // 0.12)
        if nx <= 0 or ny <= 0:
            return
        B.inst("plank_x" if not alt else "plank_x2", [x0 + 1.0, y0 + 0.06, z],
               grid={"count": [nx, ny], "step": [[2.0, 0, 0], [0, 0.12, 0]]})

    def tiles(name, x0, x1, y0, y1, z, s):
        nx, ny = int((x1 - x0) // s), int((y1 - y0) // s)
        B.inst(name, [x0 + s / 2, y0 + s / 2, z], grid={"count": [nx, ny], "step": [[s, 0, 0], [0, s, 0]]})

    planks(T + 0.05, 7.0, T + 0.02, 5.0, C.FFL0, False)            # гостиная
    planks(T + 0.05, 4.2, 5.12, C.Y1 - T, C.FFL0, True)            # комната мамы
    tiles("floor_tile_k", 7.12, C.X1 - T, T, 5.0, C.FFL0, 0.4)      # кухня
    tiles("floor_tile", 9.32, C.X1 - T, 5.12, C.Y1 - T, C.FFL0, 0.3)  # санузел 1
    planks(T + 0.05, 6.5, T + 0.02, 5.0, C.FFL1, True)             # спальня
    planks(6.62, C.X1 - T, T + 0.02, 5.0, C.FFL1, False)           # детская
    planks(T + 0.05, 4.2, 5.12, C.Y1 - T, C.FFL1, False)           # кабинет
    tiles("floor_tile", 9.32, C.X1 - T, 5.12, C.Y1 - T, C.FFL1, 0.3)  # санузел 2

    # --- лестница
    steps = stairs_geometry()
    for k, x0, x1, y0, y1, top in steps:
        B.obj(box("stringer", x0, x1, y0, y1, C.FFL0, top - 0.03, STAIR, f"step_{k:02d}"))
        B.obj(box("stair_step", x0 - (0.02 if k <= 8 else 0), x1, y0 - (0.02 if k >= 10 else 0), y1, top - 0.03, top,
                  WOOD_L, f"tread_{k:02d}"))
    # ограждения открытых сторон (балясины каждые 0.14, поручень +0.9)
    bal = 0.04
    for k, x0, x1, y0, y1, top in steps:
        if k <= 8:
            for xc in ((x0 + x1) / 2 - 0.07, (x0 + x1) / 2 + 0.07):
                B.obj(box("baluster", xc - bal / 2, xc + bal / 2, 8.62, 8.66, top, top + 0.86, RAIL))
            B.obj(box("handrail", x0, x1, 8.6, 8.68, top + 0.86, top + 0.9, WOOD))
        elif k >= 10:
            for yc in ((y0 + y1) / 2 - 0.07, (y0 + y1) / 2 + 0.07):
                B.obj(box("baluster", 5.36, 5.40, yc - bal / 2, yc + bal / 2, top, top + 0.86, RAIL))
            B.obj(box("handrail", 5.34, 5.42, y0, y1, top + 0.86, top + 0.9, WOOD))
    # перила вокруг проёма на 2 этаже
    for (x0, x1, y0, y1) in [(5.42, 6.9, 8.56, 8.6), (6.9, 6.94, 8.6, C.Y1 - T), (5.42, 5.46, 6.64, 8.56)]:
        long_x = (x1 - x0) > (y1 - y0)
        n = int(((x1 - x0) if long_x else (y1 - y0)) / 0.13)
        for i in range(n + 1):
            if long_x:
                xc = x0 + 0.03 + i * ((x1 - x0 - 0.06) / max(n, 1))
                B.obj(box("baluster", xc - 0.02, xc + 0.02, y0, y1, C.FFL1, C.FFL1 + 1.0, RAIL))
            else:
                yc = y0 + 0.03 + i * ((y1 - y0 - 0.06) / max(n, 1))
                B.obj(box("baluster", x0, x1, yc - 0.02, yc + 0.02, C.FFL1, C.FFL1 + 1.0, RAIL))
        B.obj(box("handrail", x0 - 0.01, x1 + 0.01, y0 - 0.01, y1 + 0.01, C.FFL1 + 1.0, C.FFL1 + 1.05, WOOD))

    # --- камин и труба (из гостиной сквозь крышу)
    B.inst("fireplace", [0.8, 4.45, C.FFL0], iid="fireplace")
    B.obj(box("chimney", 0.4, 1.0, 4.4, 5.0, C.FFL0 + 1.1, C.RIDGE + 0.6, BRICK, "chimney"))
    for i in range(7):  # кладка над крышей
        z = 9.05 + i * 0.16
        B.obj(box("decor", 0.38, 1.02, 4.38, 5.02, z, z + 0.14, "#9a4a3a" if i % 2 else "#7e3a2d"))
    B.obj(box("chimney", 0.34, 1.06, 4.34, 5.06, C.RIDGE + 0.6, C.RIDGE + 0.68, "#5a5a5a", "chimney_cap"))

    # --- крыша: подложка + черепица (клинья по скату) + конёк
    B.obj(prim("gable", "roof", [6.5, 5.0, (C.EAVE + C.RIDGE) / 2], [14.0, 11.0, C.RIDGE - C.EAVE], ROOF_D, "roof_base", ridge="x"))
    k = (C.RIDGE - C.EAVE) / 5.5
    tw, td = 0.25, 0.275
    th = td * k
    B.comp("tile_s", [prim("wedge", "roof_tile", [0, 0, 0], [tw - 0.006, td, th], ROOF, slope="+y")])
    B.comp("tile_n", [prim("wedge", "roof_tile", [0, 0, 0], [tw - 0.006, td, th], ROOF, slope="-y")])
    ncols, nrows = int(14.0 / tw), int(5.5 / td)
    zc = C.EAVE + 0.03 + th / 2
    B.inst("tile_s", [-0.5 + tw / 2, -0.5 + td / 2, zc], grid={"count": [ncols, nrows], "step": [[tw, 0, 0], [0, td, th]]})
    B.inst("tile_n", [-0.5 + tw / 2, 10.5 - td / 2, zc], grid={"count": [ncols, nrows], "step": [[tw, 0, 0], [0, -td, th]]})
    B.comp("ridge_cap", [prim("gable", "roof_tile", [0, 0, 0.06], [0.48, 0.36, 0.14], ROOF_D, ridge="y")])
    B.inst("ridge_cap", [-0.25, 5.0, C.RIDGE], array={"count": 28, "step": [0.5, 0, 0]})
    # водостоки
    B.obj(box("gutter", -0.5, 13.5, -0.62, -0.5, C.EAVE - 0.12, C.EAVE, METAL, "gutter_s"))
    B.obj(box("gutter", -0.5, 13.5, 10.5, 10.62, C.EAVE - 0.12, C.EAVE, METAL, "gutter_n"))
    for (x, y) in [(-0.4, -0.56), (13.4, -0.56), (-0.4, 10.56), (13.4, 10.56)]:
        B.obj(prim("cylinder", "gutter", [x, y, (C.EAVE - 0.12) / 2], [0.09, 0.09, C.EAVE - 0.12], METAL))

    # --- балкон (юг, 2 этаж)
    B.obj(box("balcony", 2.5, 6.0, -1.6, 0.0, SLAB_B, C.FFL1, SLAB, "balcony"))
    B.inst("plank_x", [3.0, -1.54, C.FFL1], grid={"count": [1, 12], "step": [[2.0, 0, 0], [0, 0.125, 0]]})
    B.inst("plank_x2", [5.0, -1.54, C.FFL1], grid={"count": [1, 12], "step": [[2.0, 0, 0], [0, 0.125, 0]]})
    for x in (2.62, 5.88):
        B.obj(prim("cylinder", "column", [x, -1.48, SLAB_B / 2], [0.2, 0.2, SLAB_B], TRIM))
    B.comp("bal_baluster", [prim("box", "baluster", [0, 0, 0.5], [0.04, 0.04, 1.0], RAIL)])
    B.inst("bal_baluster", [2.56, -1.58, C.FFL1], array={"count": 26, "step": [0.1352, 0, 0]})
    B.inst("bal_baluster", [2.52, -1.44, C.FFL1], array={"count": 11, "step": [0, 0.135, 0]})
    B.inst("bal_baluster", [5.98, -1.44, C.FFL1], array={"count": 11, "step": [0, 0.135, 0]})
    B.obj(box("handrail", 2.48, 6.02, -1.62, -1.54, C.FFL1 + 1.0, C.FFL1 + 1.08, WOOD, "bal_rail_s"))
    B.obj(box("handrail", 2.48, 2.56, -1.54, 0.0, C.FFL1 + 1.0, C.FFL1 + 1.08, WOOD, "bal_rail_w"))
    B.obj(box("handrail", 5.94, 6.02, -1.54, 0.0, C.FFL1 + 1.0, C.FFL1 + 1.08, WOOD, "bal_rail_e"))

    # --- вход: площадка, пандус 1:15, поручни
    B.obj(box("landing", 7.65, 9.15, C.Y1, 11.5, 0.0, C.FFL0, CONCRETE, "landing"))
    rx0, rx1 = 0.9, 7.65
    B.obj(prim("wedge", "ramp", [(rx0 + rx1) / 2, 10.75, C.FFL0 / 2], [rx1 - rx0, 1.2, C.FFL0], CONCRETE, "ramp", slope="+x"))
    slope = C.FFL0 / (rx1 - rx0)
    seg = 0.5
    n = int((rx1 - rx0) / seg)
    for side, yy in (("s", 10.18), ("n", 11.32)):
        for i in range(n):
            xa = rx0 + i * seg
            xm = xa + seg / 2
            zt = slope * (xm - rx0) + 0.9
            B.obj(box("handrail", xa, xa + seg, yy - 0.03, yy + 0.03, zt - 0.05, zt, WOOD))
        for i in range(0, n + 1, 3):
            xp = min(rx0 + i * seg + 0.03, rx1 - 0.03)
            zs = slope * (xp - rx0)
            zt = zs + 0.86
            B.obj(prim("cylinder", "railing", [xp, yy, (zs + zt) / 2 + 0.0], [0.05, 0.05, zt - zs + 0.02], RAIL))
    # перила площадки
    B.obj(box("handrail", 7.65, 9.15, 11.46, 11.5, C.FFL0 + 0.86, C.FFL0 + 0.9, WOOD))
    B.obj(box("handrail", 9.11, 9.15, C.Y1, 11.5, C.FFL0 + 0.86, C.FFL0 + 0.9, WOOD))
    for (x, y) in [(7.7, 11.48), (8.4, 11.48), (9.13, 11.48), (9.13, 10.75), (9.13, 10.05)]:
        B.obj(prim("cylinder", "railing", [x, y, C.FFL0 + 0.43], [0.04, 0.04, 0.86], RAIL))

    # --- мебель: гостиная
    B.inst("sofa", [2.5, 1.6, C.FFL0], iid="sofa")
    B.inst("coffee_table", [2.5, 3.0, C.FFL0], iid="coffee_table")
    B.inst("rug_big", [2.5, 2.6, C.FFL0], iid="rug_living")
    B.inst("tv_console", [2.5, 4.79, C.FFL0], rot=180, iid="tv_console")
    B.inst("bookshelf", [4.3, 4.84, C.FFL0], rot=180, iid="shelf_living")
    B.inst("floor_lamp", [4.1, 1.0, C.FFL0], iid="lamp_living")
    B.inst("plant", [6.6, 0.7, C.FFL0], iid="plant_living")
    # кухня
    for i in range(4):
        B.inst("counter_unit", [9.6 + 0.6 * i, 4.69, C.FFL0], rot=180, iid=f"cnt_n{i}")
    B.inst("stove", [10.2, 4.69, C.FFL0 + 0.9], iid="stove")
    B.inst("kitchen_sink", [11.4, 4.69, C.FFL0 + 0.9], iid="ksink")
    for i in range(2):
        B.inst("counter_unit", [12.39, 3.4 + 0.6 * i, C.FFL0], rot=90, iid=f"cnt_e{i}")
    B.inst("fridge", [12.36, 4.66, C.FFL0], rot=180, iid="fridge")
    for i in range(4):
        B.inst("upper_cabinet", [9.6 + 0.6 * i, 4.825, C.FFL0 + 1.6], rot=180, iid=f"ucab{i}")
    B.inst("dining_table", [9.6, 2.6, C.FFL0], iid="dining")
    for k, (x, y, r) in enumerate([(9.0, 1.85, 0), (9.6, 1.85, 0), (10.2, 1.85, 0), (9.0, 3.35, 180), (9.6, 3.35, 180), (10.2, 3.35, 180)]):
        B.inst("chair", [x, y, C.FFL0], rot=r, iid=f"chair{k}")
    B.inst("pendant", [9.6, 2.6, SLAB_B], iid="pendant_kitchen")
    B.inst("plant", [7.5, 0.7, C.FFL0], iid="plant_kitchen")
    # комната мамы
    B.inst("bed_single", [1.3, 7.4, C.FFL0], iid="bed_mom")
    B.inst("wardrobe", [3.55, 9.39, C.FFL0], rot=180, iid="wardrobe_mom")
    B.inst("rug_small", [3.1, 7.4, C.FFL0], iid="rug_mom")
    B.inst("pendant", [2.3, 7.4, SLAB_B], iid="pendant_mom")
    # санузел 1 (доступный)
    B.inst("toilet", [12.3, 9.38, C.FFL0], rot=180, iid="wc0")
    B.inst("bath_sink", [12.43, 6.6, C.FFL0], rot=90, iid="sink0")
    B.inst("shower", [10.0, 9.0, C.FFL0], iid="shower0")
    # холл
    B.inst("pendant", [7.2, 7.0, SLAB_B], iid="pendant_hall")
    # спальня (2 этаж)
    B.inst("bed_double", [5.45, 2.5, C.FFL1], rot=180, iid="bed_master")
    B.inst("wardrobe", [2.6, 4.69, C.FFL1], rot=180, iid="wardrobe_master")
    B.inst("rug_big", [3.6, 2.5, C.FFL1], iid="rug_master")
    B.inst("plant", [1.0, 0.75, C.FFL1], iid="plant_master")
    # детская
    B.inst("bed_kid", [9.4, 0.8, C.FFL1], iid="bed_kids")
    B.inst("desk", [12.27, 3.9, C.FFL1], rot=90, iid="desk_kids")
    B.inst("chair", [11.65, 3.9, C.FFL1], rot=-90, iid="chair_kids")
    B.inst("bookshelf", [11.95, 4.84, C.FFL1], rot=180, iid="shelf_kids")
    B.inst("rug_small", [9.4, 2.6, C.FFL1], iid="rug_kids")
    # кабинет
    B.inst("desk", [0.63, 7.4, C.FFL1], rot=-90, iid="desk_study")
    B.inst("chair", [1.25, 7.4, C.FFL1], rot=90, iid="chair_study")
    B.inst("bookshelf", [2.3, 9.54, C.FFL1], rot=180, iid="shelf_study1")
    B.inst("bookshelf", [3.5, 9.54, C.FFL1], rot=180, iid="shelf_study2")
    B.inst("floor_lamp", [1.0, 5.6, C.FFL1], iid="lamp_study")
    # санузел 2
    B.inst("bathtub", [10.3, 9.3, C.FFL1], iid="tub1")
    B.inst("toilet", [12.3, 7.0, C.FFL1], rot=90, iid="wc1")
    B.inst("bath_sink", [12.43, 6.0, C.FFL1], rot=90, iid="sink1")

    # --- телевизоры (экран + рамка)
    def tv(tid, center, w, h, facing):
        x, y, z = center
        if facing in ("-y", "+y"):
            sz = [w, 0.05, h]
            fr = [w + 0.05, 0.04, h + 0.05]
            fy = y + (0.01 if facing == "-y" else -0.01)
            fc = [x, fy, z]
        else:
            sz = [0.05, w, h]
            fr = [0.04, w + 0.05, h + 0.05]
            fx = x + (0.01 if facing == "-x" else -0.01)
            fc = [fx, y, z]
        B.obj(prim("box", "tv_screen", [x, y, z], sz, BLACK, tid, facing=facing))
        B.obj(prim("box", "tv_frame", fc, fr, "#2b2e33", f"{tid}_frame"))

    def wh(d):
        return d * 16 / math.hypot(16, 9), d * 9 / math.hypot(16, 9)
    w, h = wh(1.42)
    tv("tv1", [2.5, 4.97, C.FFL0 + 1.25], w, h, "-y")
    w, h = wh(1.2)
    tv("tv2", [12.67, 2.6, C.FFL0 + 1.3], w, h, "-x")
    w, h = wh(1.1)
    tv("tv3", [4.17, 7.4, C.FFL0 + 1.2], w, h, "-x")
    w, h = wh(1.3)
    tv("tv4", [10.6, 4.97, C.FFL1 + 1.3], w, h, "-y")
    w, h = wh(1.42)
    tv("tv5", [0.33, 2.5, C.FFL1 + 1.25], w, h, "+x")

    # --- участок: ограда с калиткой, дорожки, сад
    x0, x1, y0, y1 = C.PLOT
    sp = 0.15
    # юг
    B.inst("picket", [x0 + 0.1, y0 + 0.05, 0], array={"count": int((x1 - x0 - 0.2) / sp) + 1, "step": [sp, 0, 0]})
    # запад/восток
    for xx in (x0 + 0.05, x1 - 0.05):
        B.inst("picket", [xx, y0 + 0.2, 0], rot=90, array={"count": int((y1 - y0 - 0.4) / sp) + 1, "step": [0, sp, 0]})
    # север с калиткой x 0.0 … 1.4
    gx0, gx1 = 0.0, 1.4
    B.inst("picket", [x0 + 0.1, y1 - 0.05, 0], array={"count": int((gx0 - 0.1 - x0 - 0.1) / sp) + 1, "step": [sp, 0, 0]})
    B.inst("picket", [gx1 + 0.1, y1 - 0.05, 0], array={"count": int((x1 - 0.1 - gx1 - 0.1) / sp) + 1, "step": [sp, 0, 0]})
    for (a, b, yy) in [(x0, x1, y0 + 0.05), (x0, gx0 - 0.06, y1 - 0.05), (gx1 + 0.06, x1, y1 - 0.05)]:
        for z in (0.25, 0.85):
            B.obj(box("fence", a + 0.06, b - 0.06, yy - 0.045, yy - 0.015, z, z + 0.08, "#d6d0c4"))
    for xx in (x0 + 0.05, x1 - 0.05):
        for z in (0.25, 0.85):
            B.obj(box("fence", xx - 0.045 + (0.03 if xx > 0 else 0), xx - 0.015 + (0.03 if xx > 0 else 0), y0 + 0.06, y1 - 0.06, z, z + 0.08, "#d6d0c4"))
    for xp in list(range(int(x0), int(x1) + 1, 3)):
        if gx0 - 0.3 < xp < gx1 + 0.3:
            continue
        B.inst("fence_post", [xp + (0.1 if xp == x0 else (-0.1 if xp == x1 else 0)), y0 + 0.12, 0])
    for gp in (gx0 - 0.06, gx1 + 0.06):
        B.inst("fence_post", [gp, y1 - 0.06, 0])
    # калитка
    for i in range(9):
        xg = gx0 + 0.08 + i * 0.155
        B.obj(box("gate", xg - 0.04, xg + 0.04, y1 - 0.08, y1 - 0.05, 0.05, 1.15, "#c9b48a"))
    B.obj(box("gate", gx0, gx1, y1 - 0.11, y1 - 0.08, 0.3, 0.4, "#b39b6e", "gate_rail_low"))
    B.obj(box("gate", gx0, gx1, y1 - 0.11, y1 - 0.08, 0.9, 1.0, "#b39b6e", "gate_rail_high"))
    # дорожка от калитки к пандусу и терраса на юге
    B.inst("paver", [0.25, 11.6, 0], grid={"count": [3, 10], "step": [[0.45, 0, 0], [0, 0.45, 0]]})
    B.inst("paver", [0.25 - 0.45 * 1, 10.25, 0], grid={"count": [2, 3], "step": [[0.45, 0, 0], [0, 0.45, 0]]})
    B.inst("paver_dark", [0.6, -5.4, 0], grid={"count": [13, 7], "step": [[0.45, 0, 0], [0, 0.45, 0]]})
    B.inst("bench", [3.6, -5.6, 0.03], rot=0, iid="bench")
    # цветники
    B.inst("flower_r", [7.5, -1.2, 0], grid={"count": [24, 2], "step": [[0.2, 0, 0], [0, 0.25, 0]]})
    B.inst("flower_y", [7.6, -0.7, 0], grid={"count": [24, 2], "step": [[0.2, 0, 0], [0, 0.2, 0]]})
    B.inst("flower_v", [-1.2, 0.5, 0], grid={"count": [3, 40], "step": [[0.25, 0, 0], [0, 0.22, 0]]})
    B.inst("flower_r", [13.6, 0.5, 0], grid={"count": [3, 40], "step": [[0.25, 0, 0], [0, 0.22, 0]]})
    B.inst("veg", [14.5, -6.5, 0], grid={"count": [12, 6], "step": [[0.5, 0, 0], [0, 0.6, 0]]})
    # деревья, фонари
    for k, (x, y, name) in enumerate([(-5.5, -5.0, "tree_a"), (19.0, -5.5, "tree_a"), (19.5, 6.0, "tree_b"),
                                      (-5.0, 7.5, "tree_b"), (16.5, 12.5, "tree_a"), (-4.5, 13.0, "tree_b")]):
        B.inst(name, [x, y, 0], iid=f"tree{k}")
    for k, (x, y) in enumerate([(1.6, 12.2), (1.6, 14.4), (-0.4, 10.9), (7.0, -3.2)]):
        B.inst("garden_lamp", [x, y, 0], iid=f"glamp{k}")
    B.obj(box("decor", 2.6, 2.7, 15.6, 15.7, 0.0, 1.0, RAIL, "mailbox_post"))
    B.obj(box("decor", 2.45, 2.85, 15.5, 15.8, 1.0, 1.3, "#2f5d8c", "mailbox"))

    return B


def reference_house():
    """DSL-документ эталона с заполненным планом (числа посчитаны по развёртке)."""
    from .dsl import build_scene
    B = build_house()
    data = {"components": B.components, "objects": B.objects, "instances": B.instances, "rooms": B.rooms}
    sc = build_scene(data)
    n_tiles = int(sum(1 for o in sc.objects if o.tag == "roof_tile"))
    areas = {}
    for r in B.rooms:
        x0, y0, x1, y1 = r["rects"][0]
        areas[r["id"]] = round((x1 - x0) * (y1 - y0), 2)
    data = {
        "plan": {
            "primitives_total": len(sc.objects),
            "roof_tiles": n_tiles,
            "risers": RISERS,
            "riser_height": round(RISE, 4),
            "tread_depth": TREAD,
            "ramp_length": 6.75,
            "ridge_height": C.RIDGE,
            "room_areas": areas,
            "notes": "Вход с севера (улица) через площадку и пандус 1:15 с поручнями; гостиная и балкон на юг; "
                     "Г-образная лестница с площадкой в холле, проём в перекрытии; комната мамы и доступный санузел "
                     "на 1 этаже с разворотными зонами 1.5 м; черепица — клинья массивами по скатам.",
        },
        **data,
    }
    return data


def flawed_house():
    """Типичные ошибки сильной модели: окна без реальных проёмов, ТВ спиной в комнату,
    просчёт в числе примитивов, крутой пандус, частые балясины забыты."""
    d = copy.deepcopy(reference_house())
    objs = d["objects"]
    # 1) западная стена гостиной без проёма (окно «наложено» на сплошную стену)
    objs.append(box("wall", 0.0, T, 0.9, 2.1, 1.35, 2.75, WALL, "wall_w_patch"))
    # 2) ТВ на кухне повёрнут не туда
    for o in objs:
        if o.get("id") == "tv2":
            o["facing"] = "+x"
    # 3) пандус короче — уклон 1:9
    for o in objs:
        if o.get("id") == "ramp":
            o["center"][0] = 4.275 + 1.5
            o["size"][0] = 3.75
    # 4) план: неверное число примитивов
    d["plan"]["primitives_total"] = int(d["plan"]["primitives_total"] * 0.8)
    # 5) балясины балкона с шагом 0.2
    for inst in d["instances"]:
        if inst["use"] == "bal_baluster":
            st = inst["array"]["step"]
            inst["array"]["step"] = [v * 1.6 for v in st]
            inst["array"]["count"] = max(1, int(inst["array"]["count"] / 1.6))
    return d


def as_response(d):
    return "```json\n" + json.dumps(d, ensure_ascii=False, separators=(",", ":")) + "\n```"
