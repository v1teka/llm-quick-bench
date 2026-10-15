"""DSL v2: компоненты, экземпляры и массивы → десятки тысяч примитивов.

{
  "plan": {...},
  "components": {"имя": {"objects": [примитив...], "instances": [экземпляр...]}},
  "objects": [примитив...],          # верхний уровень; id обязательны для tv1..tv5
  "instances": [экземпляр...],
  "rooms": [{"id", "type", "level", "rects": [[x0, y0, x1, y1], ...]}]
}
экземпляр: {"use": имя, "at": [x, y, z], "rot": град (вокруг Z), "array": {"count": n, "step": [dx, dy, dz]},
            "grid": {"count": [n, m], "step": [[...], [...]]}, "id": строка}
"""
from __future__ import annotations

import json
import math
import re

import numpy as np

from ..scene import Obj, _strip_comments_and_commas, hex_to_rgb, DEFAULT_COLOR

TYPES = {"box", "cylinder", "sphere", "wedge", "gable"}
TAGS = {
    "foundation", "floor_slab", "wall", "window_frame", "glass", "sill", "door", "door_frame",
    "stair_step", "stringer", "handrail", "baluster", "railing", "ramp", "landing", "balcony",
    "column", "roof", "roof_tile", "gutter", "chimney", "fireplace",
    "bed", "sofa", "table", "chair", "counter", "stove", "sink", "fridge", "toilet", "shower", "bathtub",
    "wardrobe", "shelf", "lamp", "rug", "furniture", "tv_screen", "tv_frame",
    "fence", "gate", "paving", "tree", "plant", "decor", "other",
}
FURNITURE = {"bed", "sofa", "table", "chair", "counter", "stove", "sink", "fridge", "toilet", "shower",
             "bathtub", "wardrobe", "shelf", "furniture", "fireplace"}
MAX_PRIMS = 60000
MAX_DEPTH = 5


class DSLError(Exception):
    pass


class HouseScene:
    """Развёрнутая сцена + метаданные."""

    def __init__(self, objs, groups, plan, rooms, errors, n_components, n_arrays, raw_chars, tvs):
        self.objects: list[Obj] = objs
        self.groups = np.array(groups, int)
        self.plan = plan or {}
        self.rooms = rooms
        self.errors = errors
        self.n_components = n_components
        self.n_arrays = n_arrays
        self.raw_chars = raw_chars
        self.tvs = tvs  # id -> Obj
        n = len(objs)
        self.amin = np.array([o.aabb_min for o in objs]) if n else np.zeros((0, 3))
        self.amax = np.array([o.aabb_max for o in objs]) if n else np.zeros((0, 3))
        self.tags = np.array([o.tag for o in objs]) if n else np.zeros(0, str)
        self.index = {o.id: k for k, o in enumerate(objs)}

    def idx(self, *tags):
        return np.nonzero(np.isin(self.tags, list(tags)))[0]

    def by_tag(self, *tags):
        return [self.objects[i] for i in self.idx(*tags)]

    def bounds(self):
        if not len(self.objects):
            return np.array([0, 0, 0.0]), np.array([12, 9, 9.6])
        return self.amin.min(0), self.amax.max(0)


def _num_list(v, n):
    if not isinstance(v, (list, tuple)) or len(v) < n:
        return None
    try:
        out = [float(x) for x in v[:n]]
    except (TypeError, ValueError):
        return None
    return out if all(math.isfinite(x) for x in out) else None


def _rot(v, yaw):
    c, s = math.cos(yaw), math.sin(yaw)
    return [c * v[0] - s * v[1], s * v[0] + c * v[1], v[2]]


class Expander:
    def __init__(self, data):
        self.data = data
        self.components = data.get("components") or {}
        if not isinstance(self.components, dict):
            self.components = {}
        self.errors: list[str] = []
        self.objs: list[Obj] = []
        self.groups: list[int] = []
        self.n_arrays = 0
        self.used = set()
        self._group = 0

    def err(self, msg):
        if len(self.errors) < 200:
            self.errors.append(msg)

    def prim(self, r, origin, yaw, pid, group):
        if not isinstance(r, dict):
            self.err(f"{pid}: примитив не объект")
            return
        t = str(r.get("type", "")).lower()
        if t not in TYPES:
            self.err(f"{pid}: неизвестный type {t!r}")
            return
        c = _num_list(r.get("center"), 3)
        size = r.get("size")
        if t in ("sphere", "cylinder") and isinstance(size, (int, float)):
            size = [size, size, size]
        sz = _num_list(size, 3) if t != "sphere" else (_num_list(size, 1) and _num_list(size, 1) * 3)
        if c is None or sz is None or min(sz) <= 0:
            self.err(f"{pid}: нет корректных center/size")
            return
        tag = str(r.get("tag", "other")).lower()
        if tag not in TAGS:
            self.err(f"{pid}: неизвестный tag {tag!r}")
            tag = "other"
        col = hex_to_rgb(r.get("color"))
        if col is None:
            self.err(f"{pid}: неверный color")
            col = DEFAULT_COLOR
        try:
            y0 = math.radians(float(r.get("yaw", 0) or 0))
        except (TypeError, ValueError):
            y0 = 0.0
            self.err(f"{pid}: неверный yaw")
        slope = str(r.get("slope", "+x")).lower()
        if t == "wedge" and slope not in ("+x", "-x", "+y", "-y"):
            self.err(f"{pid}: wedge без slope")
            slope = "+x"
        ridge = str(r.get("ridge", "x")).lower()
        if t == "gable" and ridge not in ("x", "y"):
            self.err(f"{pid}: gable без ridge")
            ridge = "x"
        wc = [a + b for a, b in zip(origin, _rot(c, yaw))]
        o = Obj(pid, t, tag, wc, sz, col, y0 + yaw, slope, ridge, len(self.objs), {})
        o.facing = r.get("facing")
        o.group = group
        o.raw_local = r
        self.objs.append(o)
        self.groups.append(group)
        if len(self.objs) > MAX_PRIMS:
            raise DSLError(f"больше {MAX_PRIMS} примитивов после развёртки")

    def instance(self, inst, origin, yaw, prefix, depth, group, top):
        if not isinstance(inst, dict):
            self.err(f"{prefix}: экземпляр не объект")
            return
        name = inst.get("use")
        comp = self.components.get(name) if isinstance(name, str) else None
        if comp is None:
            self.err(f"{prefix}: нет компонента {name!r}")
            return
        if depth > MAX_DEPTH:
            self.err(f"{prefix}: слишком глубокая вложенность")
            return
        self.used.add(name)
        at = _num_list(inst.get("at", [0, 0, 0]), 3) or [0.0, 0.0, 0.0]
        try:
            rot = math.radians(float(inst.get("rot", 0) or 0))
        except (TypeError, ValueError):
            rot = 0.0
        offsets = [(0, (0.0, 0.0, 0.0))]
        if isinstance(inst.get("array"), dict):
            a = inst["array"]
            n = int(a.get("count", 1) or 1)
            st = _num_list(a.get("step"), 3) or [0, 0, 0]
            offsets = [(i, (st[0] * i, st[1] * i, st[2] * i)) for i in range(max(0, min(n, 5000)))]
            self.n_arrays += 1
        elif isinstance(inst.get("grid"), dict):
            g = inst["grid"]
            cnt = g.get("count") or [1, 1]
            steps = g.get("step") or [[0, 0, 0], [0, 0, 0]]
            try:
                n, m = int(cnt[0]), int(cnt[1])
                s1, s2 = _num_list(steps[0], 3), _num_list(steps[1], 3)
            except (TypeError, ValueError, IndexError):
                n = m = 0
                s1 = s2 = None
            if s1 is None or s2 is None:
                self.err(f"{prefix}: неверный grid")
                return
            offsets = [(i * m + j, (s1[0] * i + s2[0] * j, s1[1] * i + s2[1] * j, s1[2] * i + s2[2] * j))
                       for i in range(max(0, min(n, 2000))) for j in range(max(0, min(m, 2000)))]
            self.n_arrays += 1
        base_at = [a + b for a, b in zip(origin, _rot(at, yaw))]
        for k, off in offsets:
            if top:
                self._group += 1
                group = self._group
            o2 = [base_at[0] + off[0], base_at[1] + off[1], base_at[2] + off[2]] if depth == 0 else \
                [a + b for a, b in zip(base_at, _rot(off, yaw))]
            pre = f"{prefix}[{k}]" if len(offsets) > 1 else prefix
            y2 = yaw + rot
            for j, r in enumerate(comp.get("objects") or []):
                pid = f"{pre}/{r.get('id', j) if isinstance(r, dict) else j}"
                self.prim(r, o2, y2, pid, group)
            for j, sub in enumerate(comp.get("instances") or []):
                sid = sub.get("id", f"{sub.get('use', '?')}{j}") if isinstance(sub, dict) else j
                self.instance(sub, o2, y2, f"{pre}/{sid}", depth + 1, group, False)

    def run(self):
        for j, r in enumerate(self.data.get("objects") or []):
            self._group += 1
            pid = str(r.get("id", f"obj{j}")) if isinstance(r, dict) else f"obj{j}"
            self.prim(r, [0.0, 0.0, 0.0], 0.0, pid, self._group)
        for j, inst in enumerate(self.data.get("instances") or []):
            iid = inst.get("id", f"{inst.get('use', '?')}#{j}") if isinstance(inst, dict) else f"#{j}"
            self.instance(inst, [0.0, 0.0, 0.0], 0.0, str(iid), 0, 0, True)


def parse_rooms(raw, errors):
    rooms = []
    if not isinstance(raw, list):
        errors.append("нет списка rooms")
        return rooms
    for k, r in enumerate(raw):
        if not isinstance(r, dict):
            continue
        rects = r.get("rects") or ([r.get("rect")] if r.get("rect") else [])
        good = []
        for q in rects:
            v = _num_list(q, 4)
            if v is None:
                errors.append(f"room {r.get('id', k)}: неверный прямоугольник")
                continue
            x0, y0, x1, y1 = v
            good.append((min(x0, x1), min(y0, y1), max(x0, x1), max(y0, y1)))
        try:
            lvl = int(r.get("level", 0))
        except (TypeError, ValueError):
            lvl = 0
        rooms.append({"id": str(r.get("id", f"room{k}")), "type": str(r.get("type", "")).lower(),
                      "level": lvl, "rects": good,
                      "area": sum((x1 - x0) * (y1 - y0) for x0, y0, x1, y1 in good)})
    return rooms


class ParseResult2:
    def __init__(self, scene, strict, outside, error=""):
        self.scene, self.strict, self.outside_chars, self.error = scene, strict, outside, error


def extract_doc(text):
    """JSON-документ сцены из ответа -> (data | None, strict, символов вне JSON, ошибка)."""
    text = text or ""
    blocks = re.findall(r"```(?:json|JSON)?\s*\n(.*?)```", text, flags=re.S)
    cands = sorted(blocks, key=len, reverse=True)
    i, j = text.find("{"), text.rfind("}")
    if i != -1 and j > i:
        cands.append(text[i:j + 1])
    data, strict, used = None, False, ""
    for c in cands:
        try:
            data, strict, used = json.loads(c), True, c
            break
        except Exception:  # noqa: BLE001
            continue
    if data is None:
        for c in cands:
            try:
                data, used = json.loads(_strip_comments_and_commas(c)), c
                break
            except Exception:  # noqa: BLE001
                continue
    if not isinstance(data, dict):
        return None, False, len(text), ("пустой ответ" if not text.strip() else "JSON сцены не найден или не разбирается")
    outside = len(text.replace(used, "").replace("```json", "").replace("```", "").strip())
    return data, strict, outside, None


def parse_house(text):
    data, strict, outside, err = extract_doc(text)
    if data is None:
        return ParseResult2(None, False, outside, err)
    try:
        return ParseResult2(build_scene(data, len(text or "")), strict, outside)
    except DSLError as e:
        return ParseResult2(None, strict, outside, str(e))


def merge_doc(base, part):
    """Склейка частей сцены и патчей исправления.

    "reset": true — начать сцену заново; "remove": [id, ...] — удалить объекты, экземпляры, комнаты
    или компоненты прошлых частей; компоненты с тем же именем и объекты/экземпляры/комнаты с тем же id
    заменяются, остальное добавляется; поля "plan" дописываются поверх.
    """
    base = {} if part.get("reset") else (base or {})
    out = {"plan": dict(base.get("plan") or {}), "components": dict(base.get("components") or {}),
           "objects": list(base.get("objects") or []), "instances": list(base.get("instances") or []),
           "rooms": list(base.get("rooms") or [])}
    rm = {str(x) for x in (part.get("remove") or []) if isinstance(x, (str, int))}
    if rm:
        for key in ("objects", "instances", "rooms"):
            out[key] = [o for o in out[key] if not (isinstance(o, dict) and str(o.get("id")) in rm)]
        for name in list(out["components"]):
            if name in rm:
                del out["components"][name]
    if isinstance(part.get("plan"), dict):
        out["plan"].update(part["plan"])
    if isinstance(part.get("components"), dict):
        out["components"].update(part["components"])
    for key in ("objects", "instances", "rooms"):
        new = part.get(key)
        if not isinstance(new, list):
            continue
        pos = {str(o.get("id")): i for i, o in enumerate(out[key]) if isinstance(o, dict) and o.get("id") is not None}
        for o in new:
            oid = str(o.get("id")) if isinstance(o, dict) and o.get("id") is not None else None
            if oid is not None and oid in pos:
                out[key][pos[oid]] = o
            else:
                if oid is not None:
                    pos[oid] = len(out[key])
                out[key].append(o)
    return out


def parse_house_parts(texts):
    """Сцена из нескольких ответов (части постройки + патчи) -> (ParseResult2, склеенный документ)."""
    doc, strict, outside, errs, got = {}, True, 0, [], 0
    for k, t in enumerate(texts, 1):
        data, st, out, err = extract_doc(t)
        outside += out
        if data is None:
            strict = False
            errs.append(f"ответ {k}: {err}")
            continue
        got += 1
        strict = strict and st
        doc = merge_doc(doc, data)
    if not got:
        return ParseResult2(None, False, outside, "; ".join(errs) or "JSON сцены не найден"), doc
    try:
        sc = build_scene(doc, sum(len(t or "") for t in texts))
    except DSLError as e:
        return ParseResult2(None, strict, outside, str(e)), doc
    sc.errors[:0] = errs
    return ParseResult2(sc, strict, outside), doc


def build_scene(data, raw_chars=0):
    ex = Expander(data)
    ex.run()
    rooms = parse_rooms(data.get("rooms"), ex.errors)
    tvs = {o.id: o for o in ex.objs if o.tag == "tv_screen"}
    return HouseScene(ex.objs, ex.groups, data.get("plan"), rooms, ex.errors,
                      len(ex.used), ex.n_arrays, raw_chars, tvs)
