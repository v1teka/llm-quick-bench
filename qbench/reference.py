"""Эталонный дом, проходящий все проверки, и «испорченная» версия для демо.

Используется mock-моделью, тестами и визуальной проверкой (фаза vision).
"""
from __future__ import annotations

import copy
import json

from . import config as C

WALL = "#efe6d8"
GLASS = "#7fb2d9"
WOOD = "#7a4a2a"
ROOF = "#8b2e1f"
CONCRETE = "#9c9a96"
SLAB = "#b8b2a7"
STAIR = "#a0784e"
RAIL = "#4a4f57"
BRICK = "#7b3b2c"


def _o(oid, tag, center, size, color, typ="box", **kw):
    d = {"id": oid, "type": typ, "tag": tag, "center": [round(v, 4) for v in center],
         "size": [round(v, 4) for v in size], "color": color}
    d.update(kw)
    return d


def _box(oid, tag, x0, x1, y0, y1, z0, z1, color, **kw):
    return _o(oid, tag, [(x0 + x1) / 2, (y0 + y1) / 2, (z0 + z1) / 2], [x1 - x0, y1 - y0, z1 - z0], color, **kw)


def reference_house():
    t = 0.3  # толщина стен
    objs = []
    objs.append(_box("foundation", "foundation", 0, 10, 0, 8, 0, 0.30, CONCRETE))
    objs.append(_box("floor_g", "floor_slab", 0.25, 9.75, 0.25, 7.75, 0.30, C.FFL0, SLAB))
    for lvl, (z0, z1) in enumerate([(C.FFL0, C.FFL1), (C.FFL1, C.EAVE)]):
        L = lvl + 1
        objs.append(_box(f"wall_s{L}", "wall", 0, 10, 0, t, z0, z1, WALL))
        objs.append(_box(f"wall_n{L}", "wall", 0, 10, 8 - t, 8, z0, z1, WALL))
        objs.append(_box(f"wall_w{L}", "wall", 0, t, t, 8 - t, z0, z1, WALL))
        objs.append(_box(f"wall_e{L}", "wall", 10 - t, 10, t, 8 - t, z0, z1, WALL))

    # перекрытие 2 этажа с проёмом над лестницей
    sz0, sz1 = C.FFL1 - 0.2, C.FFL1
    objs.append(_box("floor_1_main", "floor_slab", 1.3, 9.75, 0.25, 7.75, sz0, sz1, SLAB))
    objs.append(_box("floor_1_s", "floor_slab", 0.25, 1.3, 0.25, 1.84, sz0, sz1, SLAB))
    objs.append(_box("floor_1_n", "floor_slab", 0.25, 1.3, 5.2, 7.75, sz0, sz1, SLAB))
    objs.append(_box("rail_opening", "railing", 1.3, 1.35, 1.84, 5.2, C.FFL1, C.FFL1 + 1.0, RAIL))

    # лестница: 16 подъёмов по 0.1875, проступь 0.28, ширина 1.0
    rise, tread = (C.FFL1 - C.FFL0) / 16, 0.28
    for i in range(1, 16):
        y0 = 1.0 + (i - 1) * tread
        objs.append(_box(f"step_{i:02d}", "stair_step", t, 1.3, y0, y0 + tread, C.FFL0, C.FFL0 + i * rise, STAIR))

    def win(oid, fac, along, lvl, w, h, sill):
        ffl = C.FFL0 if lvl == 0 else C.FFL1
        z0, z1 = ffl + sill, ffl + sill + h
        d = 0.36
        if fac == "S":
            return _box(oid, "window", along - w / 2, along + w / 2, t / 2 - d / 2, t / 2 + d / 2, z0, z1, GLASS)
        if fac == "N":
            return _box(oid, "window", along - w / 2, along + w / 2, 8 - t / 2 - d / 2, 8 - t / 2 + d / 2, z0, z1, GLASS)
        if fac == "W":
            return _box(oid, "window", t / 2 - d / 2, t / 2 + d / 2, along - w / 2, along + w / 2, z0, z1, GLASS)
        return _box(oid, "window", 10 - t / 2 - d / 2, 10 - t / 2 + d / 2, along - w / 2, along + w / 2, z0, z1, GLASS)

    for k, x in enumerate([2, 5, 8]):
        objs.append(win(f"win_s1_{k}", "S", x, 0, 2.0, 1.6, 0.8))
    objs.append(win("win_n1_0", "N", 2.5, 0, 1.2, 1.4, 0.9))
    objs.append(win("win_w1_0", "W", 6.5, 0, 1.2, 1.4, 0.9))
    objs.append(win("win_e1_0", "E", 1.5, 0, 1.2, 1.4, 0.9))
    objs.append(win("win_s2_0", "S", 2, 1, 1.6, 1.4, 0.9))
    objs.append(win("win_s2_1", "S", 8, 1, 1.6, 1.4, 0.9))
    objs.append(win("win_n2_0", "N", 2.5, 1, 1.2, 1.4, 0.9))
    objs.append(win("win_n2_1", "N", 7.5, 1, 1.2, 1.4, 0.9))
    objs.append(win("win_w2_0", "W", 4, 1, 1.2, 1.4, 0.9))
    objs.append(win("win_e2_0", "E", 4, 1, 1.2, 1.4, 0.9))

    # входная дверь (север, к улице), пандус 1:13.3 и площадка
    objs.append(_box("door_main", "door", 6.0, 7.0, 8 - t / 2 - 0.18, 8 - t / 2 + 0.18, C.FFL0, C.FFL0 + 2.1, WOOD))
    objs.append(_box("landing", "landing", 5.75, 7.25, 8.0, 9.5, 0, C.FFL0, CONCRETE))
    objs.append(_box("ramp", "ramp", -0.25, 5.75, 8.0, 9.5, 0, C.FFL0, CONCRETE, type="wedge", slope="+x"))
    objs[-1]["type"] = "wedge"
    objs.append(_box("ramp_rail", "railing", -0.25, 7.25, 9.45, 9.5, C.FFL0, C.FFL0 + 0.9, RAIL))

    # балкон
    objs.append(_box("door_balcony", "door", 4.5, 5.5, t / 2 - 0.18, t / 2 + 0.18, C.FFL1, C.FFL1 + 2.2, WOOD))
    objs.append(_box("balcony", "balcony", 3.0, 7.0, -1.4, 0.0, C.FFL1 - 0.2, C.FFL1, SLAB))
    objs.append(_box("bal_rail_front", "railing", 3.0, 7.0, -1.4, -1.35, C.FFL1, C.FFL1 + 1.05, RAIL))
    objs.append(_box("bal_rail_w", "railing", 3.0, 3.05, -1.4, 0.0, C.FFL1, C.FFL1 + 1.05, RAIL))
    objs.append(_box("bal_rail_e", "railing", 6.95, 7.0, -1.4, 0.0, C.FFL1, C.FFL1 + 1.05, RAIL))

    # крыша и труба
    objs.append(_o("roof", "roof", [5, 4, (C.EAVE + C.RIDGE) / 2], [10.8, 8.8, C.RIDGE - C.EAVE], ROOF, typ="gable", ridge="x"))
    objs.append(_box("chimney", "chimney", 7.0, 7.6, 5.0, 5.6, 5.0, 9.7, BRICK))

    # немного жизни вокруг
    objs.append(_o("tree_trunk", "decor", [-3.5, -4, 1.0], [0.35, 0.35, 2.0], "#6b4a2e", typ="cylinder"))
    objs.append(_o("tree_crown", "decor", [-3.5, -4, 3.0], [2.6, 2.6, 2.6], "#4f8a3c", typ="sphere"))
    objs.append(_box("path", "decor", 6.0, 7.0, 9.5, 14.0, 0, 0.03, "#c9c2b2"))

    plan = {
        "entrance_facade": "N", "risers": 16, "riser_height": 0.1875, "tread_depth": 0.28,
        "ramp_length": 6.0, "ridge_height": 9.0,
        "notes": "Вход с севера (улица), пандус 1:13 вдоль фасада; большие окна на юг; "
                 "прямая лестница у западной стены, проём в перекрытии над ней.",
    }
    return {"plan": plan, "objects": objs}


def reference_with_garage():
    d = reference_house()
    o = d["objects"]
    t = 0.3
    gx0, gx1, gy0, gy1, h = C.GX0, C.GX1, C.GY0, C.GY1, C.GARAGE_ROOF_TOP - 0.2
    o.append(_box("g_floor", "other", gx0, gx1, gy0, gy1, 0, 0.05, CONCRETE))
    o.append(_box("g_wall_e", "wall", gx1 - t, gx1, gy0, gy1, 0, h, WALL))
    o.append(_box("g_wall_s", "wall", gx0, gx1 - t, gy0, gy0 + t, 0, h, WALL))
    o.append(_box("g_wall_n_w", "wall", gx0, gx0 + 0.6, gy1 - t, gy1, 0, h, WALL))
    o.append(_box("g_wall_n_e", "wall", gx1 - 0.6, gx1, gy1 - t, gy1, 0, h, WALL))
    o.append(_box("g_wall_n_top", "wall", gx0 + 0.6, gx1 - 0.6, gy1 - t, gy1, 2.4, h, WALL))
    o.append(_box("g_door", "garage_door", gx0 + 0.65, gx1 - 0.65, gy1 - t / 2 - 0.18, gy1 - t / 2 + 0.18, 0, 2.35, "#d8d8d8"))
    o.append(_box("g_roof", "roof", gx0, gx1 + 0.2, gy0 - 0.2, gy1 + 0.2, h, C.GARAGE_ROOF_TOP, "#5b5f63"))
    o.append(_box("g_house_door", "door", 9.7 - 0.06, 10.06, 5.05, 5.95, C.FFL0, C.FFL0 + 2.1, WOOD))
    o.append(_box("g_step1", "other", 10.0, 10.9, 5.0, 6.0, 0, 0.225, CONCRETE))
    o.append(_box("g_step2", "other", 10.0, 10.6, 5.0, 6.0, 0.225, C.FFL0, CONCRETE))
    d["plan"]["notes"] += " Гараж 4×5 м с востока, ворота на север, дверь в дом через восточную стену."
    return d


def flawed_house():
    """Типичные ошибки: крутой пандус, мало ступеней, нет проёма, висящая труба, нет свесов."""
    d = copy.deepcopy(reference_house())
    objs = [o for o in d["objects"] if not o["id"].startswith(("step_", "floor_1_", "rail_opening"))]
    rise, tread = 0.25, 0.28
    for i in range(1, 12):
        y0 = 1.0 + (i - 1) * tread
        objs.append(_box(f"step_{i:02d}", "stair_step", 0.3, 1.3, y0, y0 + tread, C.FFL0, C.FFL0 + i * rise, STAIR))
    objs.append(_box("floor_1", "floor_slab", 0.25, 9.75, 0.25, 7.75, C.FFL1 - 0.2, C.FFL1, SLAB))
    for o in objs:
        if o["id"] == "ramp":
            o.update(_box("ramp", "ramp", 2.75, 5.75, 8.0, 9.5, 0, C.FFL0, CONCRETE, type="wedge", slope="+x"))
        if o["id"] == "ramp_rail":
            o.update(_box("ramp_rail", "railing", 2.75, 7.25, 9.45, 9.5, C.FFL0, C.FFL0 + 0.9, RAIL))
        if o["id"] == "roof":
            o["size"] = [10.0, 8.0, C.RIDGE - C.EAVE]
        if o["id"] == "chimney":
            o.update(_box("chimney", "chimney", 7.0, 7.6, 5.0, 5.6, 9.1, 9.9, BRICK))
    d["objects"] = objs
    d["plan"].update({"risers": 12, "riser_height": 0.25, "ramp_length": 6.0})
    return d


def as_response(d):
    return "```json\n" + json.dumps(d, ensure_ascii=False, indent=1) + "\n```"
