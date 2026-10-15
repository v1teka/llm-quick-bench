"""Сцена из примитивов: разбор ответа модели, геометрия, меши."""
from __future__ import annotations

import json
import math
import re
from dataclasses import dataclass, field

import numpy as np

from . import config as C

DEFAULT_COLOR = (0.7, 0.7, 0.7)


def hex_to_rgb(s):
    if not isinstance(s, str):
        return None
    s = s.strip().lstrip("#")
    if len(s) == 3:
        s = "".join(ch * 2 for ch in s)
    if len(s) != 6 or not re.fullmatch(r"[0-9a-fA-F]{6}", s):
        return None
    return tuple(int(s[i:i + 2], 16) / 255.0 for i in (0, 2, 4))


@dataclass
class Obj:
    id: str
    type: str
    tag: str
    center: np.ndarray
    size: np.ndarray
    color: tuple = DEFAULT_COLOR
    yaw: float = 0.0  # радианы
    slope: str = "+x"
    ridge: str = "x"
    index: int = 0
    raw: dict = field(default_factory=dict)

    def __post_init__(self):
        self.center = np.asarray(self.center, float)
        self.size = np.abs(np.asarray(self.size, float))
        if self.type in ("cylinder", "sphere"):
            d = self.size[0]
            self.size = np.array([d, d, d if self.type == "sphere" else self.size[2]])
            self.yaw = 0.0
        self.half = self.size / 2
        c, s = math.cos(self.yaw), math.sin(self.yaw)
        self._rot = np.array([[c, -s], [s, c]])
        corners = np.array([[sx, sy] for sx in (-1, 1) for sy in (-1, 1)]) * self.half[:2]
        w = corners @ self._rot.T
        self.aabb_min = np.array([*(w.min(0) + self.center[:2]), self.center[2] - self.half[2]])
        self.aabb_max = np.array([*(w.max(0) + self.center[:2]), self.center[2] + self.half[2]])

    # --- координаты ---------------------------------------------------------
    def to_local_xy(self, x, y):
        dx = np.asarray(x, float) - self.center[0]
        dy = np.asarray(y, float) - self.center[1]
        c, s = math.cos(self.yaw), math.sin(self.yaw)
        return c * dx + s * dy, -s * dx + c * dy

    def _height_local(self, lx, ly):
        """Высота верхней грани над низом примитива (в локальных координатах)."""
        hx, hy, _ = self.half
        sz = self.size[2]
        if self.type == "wedge":
            t = {"+x": (lx + hx) / (2 * hx), "-x": (hx - lx) / (2 * hx),
                 "+y": (ly + hy) / (2 * hy), "-y": (hy - ly) / (2 * hy)}[self.slope]
            return sz * np.clip(t, 0, 1)
        if self.type == "gable":
            if self.ridge == "x":
                return sz * np.clip(1 - np.abs(ly) / hy, 0, 1)
            return sz * np.clip(1 - np.abs(lx) / hx, 0, 1)
        if self.type == "sphere":
            r = self.half[0]
            rr = np.clip(r * r - lx * lx - ly * ly, 0, None)
            return r + np.sqrt(rr)
        return np.full(np.shape(lx), sz, float)

    def _inside_xy(self, lx, ly, eps=1e-9):
        hx, hy, _ = self.half
        if self.type in ("cylinder", "sphere"):
            return lx * lx + ly * ly <= hx * hx + eps
        return (np.abs(lx) <= hx + eps) & (np.abs(ly) <= hy + eps)

    def top_at(self, x, y):
        """Мировая z верхней поверхности над точкой (x, y) или nan."""
        lx, ly = self.to_local_xy(x, y)
        inside = self._inside_xy(lx, ly)
        z = self.aabb_min[2] + self._height_local(lx, ly)
        return np.where(inside, z, np.nan)

    def contains(self, p):
        p = np.atleast_2d(np.asarray(p, float))
        lx, ly = self.to_local_xy(p[:, 0], p[:, 1])
        zb = self.aabb_min[2]
        inside = self._inside_xy(lx, ly) & (p[:, 2] >= zb - 1e-9)
        top = zb + self._height_local(lx, ly)
        if self.type == "sphere":
            r = self.half[0]
            d2 = lx * lx + ly * ly + (p[:, 2] - self.center[2]) ** 2
            return d2 <= r * r
        return inside & (p[:, 2] <= top + 1e-9)

    # --- меш ----------------------------------------------------------------
    def triangles(self):
        cache = self.__dict__.get("_tris")
        if cache is None:
            cache = self._triangles()
            self.__dict__["_tris"] = cache
        return cache.copy()

    def _triangles(self):
        hx, hy, hz = self.half
        t = self.type
        if t == "box":
            v = np.array([[x, y, z] for z in (-hz, hz) for y in (-hy, hy) for x in (-hx, hx)])
            f = [(0, 1, 3), (0, 3, 2), (4, 6, 7), (4, 7, 5), (0, 4, 5), (0, 5, 1),
                 (2, 3, 7), (2, 7, 6), (0, 2, 6), (0, 6, 4), (1, 5, 7), (1, 7, 3)]
            tris = v[np.array(f)]
        elif t in ("wedge", "gable"):
            tris = self._prism_tris()
        elif t == "cylinder":
            n = 8 if hx * 2 < 0.15 else (12 if hx * 2 < 0.6 else 18)
            a = np.linspace(0, 2 * np.pi, n + 1)
            ring = np.stack([np.cos(a) * hx, np.sin(a) * hx], 1)
            tl = []
            for i in range(n):
                p0, p1 = ring[i], ring[i + 1]
                b0, b1 = [*p0, -hz], [*p1, -hz]
                u0, u1 = [*p0, hz], [*p1, hz]
                tl += [[b0, b1, u1], [b0, u1, u0], [[0, 0, hz], u0, u1], [[0, 0, -hz], b1, b0]]
            tris = np.array(tl, float)
        else:  # sphere
            nu, nv = (8, 5) if hx * 2 < 0.3 else ((10, 6) if hx * 2 < 1.0 else (14, 8))
            r = hx
            us = np.linspace(0, 2 * np.pi, nu + 1)
            vs = np.linspace(-np.pi / 2, np.pi / 2, nv + 1)

            def P(u, v):
                return [r * np.cos(v) * np.cos(u), r * np.cos(v) * np.sin(u), r * np.sin(v)]
            tl = []
            for i in range(nu):
                for j in range(nv):
                    a, b = P(us[i], vs[j]), P(us[i + 1], vs[j])
                    c, d = P(us[i + 1], vs[j + 1]), P(us[i], vs[j + 1])
                    tl += [[a, b, c], [a, c, d]]
            tris = np.array(tl, float)
        # поворот + перенос
        xy = tris[..., :2] @ self._rot.T
        out = np.concatenate([xy, tris[..., 2:]], -1) + self.center
        return out

    def _prism_tris(self):
        hx, hy, hz = self.half
        # сечение в плоскости (u, z), выдавливание вдоль w
        if self.type == "gable":
            sec = [(-1, -1), (1, -1), (0, 1)]
            axis = self.ridge  # выдавливание вдоль конька
        else:
            sl = self.slope
            axis = "y" if sl[1] == "x" else "x"
            sec = [(-1, -1), (1, -1), (1, 1)] if sl[0] == "+" else [(-1, -1), (1, -1), (-1, 1)]

        def V(u, z, w):
            if axis == "x":  # выдавливание вдоль X, сечение в (y, z)
                return [w * hx, u * hy, z * hz]
            return [u * hx, w * hy, z * hz]
        a = [V(u, z, -1) for u, z in sec]
        b = [V(u, z, 1) for u, z in sec]
        tl = [a, b[::-1]]
        for i in range(3):
            j = (i + 1) % 3
            tl += [[a[i], a[j], b[j]], [a[i], b[j], b[i]]]
        return np.array(tl, float)


# ---------------------------------------------------------------------------
# Разбор ответа модели
# ---------------------------------------------------------------------------

@dataclass
class ParseResult:
    scene: "Scene | None"
    strict: bool
    outside_chars: int
    error: str = ""


def _strip_comments_and_commas(s):
    s = re.sub(r"/\*.*?\*/", "", s, flags=re.S)
    s = re.sub(r"(?m)(?<![:\"\w])//.*$", "", s)
    s = re.sub(r",\s*([}\]])", r"\1", s)
    return s


def _candidates(text):
    blocks = re.findall(r"```(?:json|JSON)?\s*\n(.*?)```", text, flags=re.S)
    cands = sorted(blocks, key=len, reverse=True)
    i, j = text.find("{"), text.rfind("}")
    if i != -1 and j > i:
        cands.append(text[i:j + 1])
    i, j = text.find("["), text.rfind("]")
    if i != -1 and j > i:
        cands.append(text[i:j + 1])
    return cands


def parse_response(text: str) -> ParseResult:
    text = text or ""
    cands = _candidates(text)
    if not cands:
        return ParseResult(None, False, len(text), "JSON не найден")
    data, strict, used = None, False, ""
    for c in cands:
        try:
            data, strict, used = json.loads(c), True, c
            break
        except Exception:
            continue
    if data is None:
        for c in cands:
            try:
                data, used = json.loads(_strip_comments_and_commas(c)), c
                break
            except Exception:
                continue
    if data is None:
        return ParseResult(None, False, len(text), "JSON не разбирается даже после починки")
    outside = len(text.replace(used, "").replace("```json", "").replace("```", "").strip())
    try:
        scene = Scene.from_data(data)
    except Exception as e:  # noqa: BLE001
        return ParseResult(None, strict, outside, f"Неверная структура: {e}")
    return ParseResult(scene, strict, outside)


class Scene:
    def __init__(self, objects, plan=None, errors=None, total=0):
        self.objects: list[Obj] = objects
        self.plan: dict = plan or {}
        self.errors: list[str] = errors or []
        self.total = total or len(objects)

    @classmethod
    def from_data(cls, data):
        if isinstance(data, list):
            raw_objs, plan = data, {}
        elif isinstance(data, dict):
            raw_objs = data.get("objects")
            plan = data.get("plan") or {}
            if raw_objs is None:
                raise ValueError("нет поля objects")
        else:
            raise ValueError("ожидался объект или список")
        if not isinstance(plan, dict):
            plan = {}
        objs, errors, seen = [], [], set()
        for k, r in enumerate(raw_objs):
            o, err = _make_obj(r, k)
            if err:
                errors.append(f"#{k}: {err}")
            if o is None:
                continue
            if o.id in seen:
                errors.append(f"#{k}: повтор id {o.id}")
                o.id = f"{o.id}__{k}"
            seen.add(o.id)
            objs.append(o)
        return cls(objs, plan, errors, len(raw_objs))

    def to_data(self):
        return {"plan": self.plan, "objects": [o.raw for o in self.objects]}

    @classmethod
    def load(cls, path):
        with open(path, encoding="utf-8") as f:
            return cls.from_data(json.load(f))

    def by_tag(self, *tags):
        return [o for o in self.objects if o.tag in tags]

    def bounds(self):
        if not self.objects:
            return np.array([0, 0, 0.0]), np.array([10, 8, 9.0])
        mn = np.min([o.aabb_min for o in self.objects], 0)
        mx = np.max([o.aabb_max for o in self.objects], 0)
        return mn, mx


def _num_list(v, n):
    if not isinstance(v, (list, tuple)) or len(v) < n:
        return None
    try:
        out = [float(x) for x in v[:n]]
    except (TypeError, ValueError):
        return None
    if not all(math.isfinite(x) for x in out):
        return None
    return out


def _make_obj(r, k):
    if not isinstance(r, dict):
        return None, "не объект"
    errs = []
    t = str(r.get("type", "")).lower()
    if t not in C.TYPES:
        return None, f"неизвестный type {t!r}"
    center = _num_list(r.get("center"), 3)
    size = r.get("size")
    if t in ("sphere", "cylinder") and isinstance(size, (int, float)):
        size = [size, size, size]
    size = _num_list(size, 3) if t not in ("sphere",) else (_num_list(size, 1) and (_num_list(size, 1) * 3))
    if center is None or size is None:
        return None, "нет корректных center/size"
    if min(size) <= 0:
        return None, "нулевой/отрицательный размер"
    tag = str(r.get("tag", "other")).lower()
    if tag not in C.TAGS:
        errs.append(f"неизвестный tag {tag!r}")
        tag = "other"
    color = hex_to_rgb(r.get("color"))
    if color is None:
        errs.append("неверный color")
        color = DEFAULT_COLOR
    try:
        yaw = math.radians(float(r.get("yaw", 0) or 0))
    except (TypeError, ValueError):
        yaw, _ = 0.0, errs.append("неверный yaw")
    slope = str(r.get("slope", "+x")).lower()
    if t == "wedge" and slope not in ("+x", "-x", "+y", "-y"):
        errs.append("wedge без корректного slope")
        slope = "+x"
    ridge = str(r.get("ridge", "x")).lower()
    if t == "gable" and ridge not in ("x", "y"):
        errs.append("gable без корректного ridge")
        ridge = "x"
    oid = str(r.get("id", f"obj{k}"))
    o = Obj(oid, t, tag, center, size, color, yaw, slope, ridge, k, dict(r))
    return o, "; ".join(errs)
