"""Реалистичный рендер дома: Intel OSPRay (path tracer, CPU) + OpenImageDenoise, PBR-материалы по тегам,
физическое небо (Hošek–Wilkie), процедурные текстуры, модели деревьев, кустов и сантехники.

Геометрия дома — та, что построила тестируемая модель. Деревья, кусты, цветы, унитазы и раковины
заменяются процедурными моделями в тех же габаритах; мягкая мебель рисуется скруглёнными коробками.
На оценку это не влияет: проверки идут по исходной сцене.
"""
from __future__ import annotations

import math
import os

import numpy as np

from ..render import assets as A
from ..render import textures as TX
from ..render.ospray import (OSPRay, T, FB_RGBA32F, FB_COLOR, FB_NORMAL, FB_ALBEDO, DENOISE_HIGH)
from . import config as C

SUN = np.array([0.45, 0.55, -0.70])  # направление света солнца (из film.LIGHT со сменой знака)


def lin(c):
    return np.clip(np.asarray(c, float), 0, 1) ** 2.2


# ---------------------------------------------------------------------------
# материалы: ключ поверхности -> (тип OSPRay, параметры, текстура, масштаб uv (м на тайл))
# ---------------------------------------------------------------------------

MATERIALS = {
    "plaster": ("principled", dict(roughness=0.92, specular=0.25), "plaster", 1.5),
    "concrete": ("principled", dict(roughness=0.88, specular=0.2), "concrete", 2.0),
    "parquet": ("principled", dict(roughness=0.35, specular=0.6, coat=0.25, coatRoughness=0.2), "parquet", 2.4),
    "wood": ("principled", dict(roughness=0.45, specular=0.5, coat=0.2, coatRoughness=0.3), "wood", 0.8),
    "paint": ("principled", dict(roughness=0.38, specular=0.6), None, 1.0),
    "metal": ("principled", dict(metallic=1.0, roughness=0.32), None, 1.0),
    "chrome": ("principled", dict(metallic=1.0, roughness=0.06), None, 1.0),
    "glass": ("thinGlass", dict(eta=1.5, attenuationColor=(0.82, 0.92, 0.95), attenuationDistance=0.6, thickness=0.006), None, 1.0),
    "tile": ("principled", dict(roughness=0.5, specular=0.55), "tile", 0.5),
    "roofbase": ("principled", dict(roughness=0.8), None, 1.0),
    "brick": ("principled", dict(roughness=0.85, specular=0.25), "bricks", 0.9),
    "stone": ("principled", dict(roughness=0.75, specular=0.3), "stone", 1.6),
    "asphalt": ("principled", dict(roughness=0.92, specular=0.2), "asphalt", 3.0),
    "grass": ("principled", dict(roughness=0.95, specular=0.15), "grass", 2.2),
    "fabric": ("principled", dict(roughness=0.95, specular=0.2, sheen=0.7, sheenRoughness=0.4), "fabric", 0.35),
    "ceramic": ("principled", dict(roughness=0.05, specular=1.0, coat=0.6, coatRoughness=0.03, ior=1.5), None, 1.0),
    "plastic_white": ("principled", dict(roughness=0.22, specular=0.7), None, 1.0),
    "counter": ("principled", dict(roughness=0.18, specular=0.8), "concrete", 1.2),
    "blackglass": ("principled", dict(roughness=0.04, specular=1.0, baseColor=(0.02, 0.02, 0.02)), None, 1.0),
    "screen_frame": ("principled", dict(roughness=0.3, specular=0.6), None, 1.0),
    "lamp": ("principled", dict(roughness=0.7, emissiveColor=(3.2, 2.6, 1.8)), None, 1.0),
    "leaf": ("principled", dict(roughness=0.55, specular=0.35, thin=True, backlight=0.65, thickness=0.05), None, 1.0),
    "bark": ("principled", dict(roughness=0.92, specular=0.15), "bark", 0.8),
    "stem": ("principled", dict(roughness=0.7, baseColor=(0.08, 0.18, 0.05)), None, 1.0),
    "petal": ("principled", dict(roughness=0.55, thin=True, backlight=0.5), None, 1.0),
    "water": ("principled", dict(roughness=0.02, specular=1.0, transmission=0.6, ior=1.33, baseColor=(0.55, 0.75, 0.85)), None, 1.0),
    "default": ("principled", dict(roughness=0.6, specular=0.4), None, 1.0),
}

SOFT = {"sofa", "bed", "rug"}


def _is_dark(c):
    return max(c) < 0.35


def _is_woody(c):
    r, g, b = c
    return r > g > b and r - b > 0.12 and r < 0.8


def surface_key(o):
    """Материал примитива по тегу и цвету."""
    t, c = o.tag, tuple(o.color)
    if t in ("wall", "column", "roof"):
        return "roofbase" if t == "roof" else "plaster"
    if t == "foundation":
        return "concrete"
    if t in ("ramp", "landing", "balcony"):
        return "concrete"
    if t == "floor_slab":
        return "parquet"  # верхние грани; боковые см. ниже
    if t == "glass":
        return "glass"
    if t in ("window_frame", "door_frame", "sill"):
        return "wood" if _is_woody(c) else "paint"
    if t == "door":
        return "wood"
    if t in ("stair_step", "stringer"):
        return "wood" if not _is_dark(c) or _is_woody(c) else "metal"
    if t in ("handrail", "baluster", "railing"):
        return "wood" if _is_woody(c) else "metal"
    if t == "roof_tile":
        return "tile"
    if t == "gutter":
        return "metal"
    if t == "chimney":
        return "brick" if not _is_dark(c) else "metal"
    if t == "fireplace":
        return "stone" if not _is_dark(c) else "metal"
    if t in ("sofa", "bed", "rug"):
        return "wood" if (_is_woody(c) and t == "bed") else "fabric"
    if t in ("table", "chair", "wardrobe", "shelf"):
        return "wood" if _is_woody(c) or not _is_dark(c) else "paint"
    if t == "counter":
        return "counter"
    if t == "stove":
        return "blackglass" if _is_dark(c) else "metal"
    if t == "sink":
        return "chrome" if max(c) < 0.75 else "ceramic"
    if t in ("toilet", "shower", "bathtub"):
        return "ceramic" if min(c) > 0.7 else ("chrome" if max(c) < 0.75 else "water")
    if t == "fridge":
        return "paint"
    if t == "lamp":
        return "lamp" if min(c) > 0.6 else "metal"
    if t in ("tv_frame",):
        return "screen_frame"
    if t == "fence":
        return "wood" if _is_woody(c) else "paint"
    if t == "gate":
        return "wood"
    if t == "paving":
        return "stone"
    if t == "furniture":
        if _is_woody(c):
            return "wood"
        return "fabric" if min(c) > 0.35 else "paint"
    if t == "decor":
        if 0.75 < c[2] and c[1] > 0.8 and c[0] < 0.9:
            return "glass"  # зеркала/вода
        return "paint"
    if t == "other":
        return "paint"
    return "default"


# ---------------------------------------------------------------------------
# сцена
# ---------------------------------------------------------------------------

class Photoreal:
    def __init__(self, sc, screens=None, build_rank=None, stages=48, threads=None, log=print, seed=7):
        self.sc = sc
        self.log = log
        self.o = OSPRay(threads=threads or os.cpu_count(), log=log)
        self.rng = np.random.default_rng(seed)
        self.stages = stages
        n = len(sc.objects)
        rank = build_rank if build_rank is not None else np.arange(n)
        self.rank = rank
        self.screen_ids = {o.id for o in sc.objects if o.tag == "tv_screen"}
        self.screens = screens or {}
        self._tex = {}
        self._mat = {}
        log("  фотореализм: материалы и модели…")
        builders = [A.MeshBuilder() for _ in range(stages)]
        used = np.zeros(n, bool)
        self._substitute(builders, used)
        stage_of = np.minimum(stages - 1, (rank * stages) // max(1, n))
        for i, o in enumerate(sc.objects):
            if used[i] or o.id in self.screen_ids:
                continue
            self._primitive(builders[stage_of[i]], o)
        self.groups = []
        for s, mb in enumerate(builders):
            self.groups.append(self._group(mb))
        self.stage_start = [int(math.ceil(s * n / stages)) for s in range(stages)]
        self._env()
        self._screens_init()
        self._light()
        self.renderer = self.o.lib.ospNewRenderer(b"pathtracer")
        self._fb = {}
        tri = sum(sum(len(t) for t in mb.parts[k][0]) for mb in builders for k in mb.parts)
        log(f"  фотореализм: {tri} треугольников, {len(self._mat)} материалов")

    # ---- текстуры и материалы
    def tex(self, name):
        if name not in self._tex:
            self._tex[name] = self.o.texture(TX.ALL[name]())
        return self._tex[name]

    def mat(self, key):
        if key in self._mat:
            return self._mat[key]
        typ, params, tex, scale = MATERIALS[key]
        L = self.o.lib
        m = L.ospNewMaterial(typ.encode())
        params = dict(params)
        if typ == "principled" and "baseColor" not in params:
            params["baseColor"] = (1.0, 1.0, 1.0)
        self.o.set(m, **params)
        if tex:
            L.ospSetObject(m, b"map_baseColor", self.tex(tex))
            self.o.set(m, **{"map_baseColor__scale": (1.0 / scale, 1.0 / scale)})
        L.ospCommit(m)
        self._mat[key] = m
        return m

    # ---- примитивы модели
    def _primitive(self, mb, o):
        key = surface_key(o)
        col = lin(o.color)
        tris = o.triangles().astype(np.float32)
        fn = np.cross(tris[:, 1] - tris[:, 0], tris[:, 2] - tris[:, 0])
        fn /= np.linalg.norm(fn, axis=1, keepdims=True) + 1e-12
        # мягкая мебель — скруглённые коробки
        if o.type == "box" and o.tag in SOFT | {"furniture"} and key == "fabric":
            r = min(0.06, 0.35 * float(min(o.half)))
            T_, N_ = A.rounded_box(o.center, o.half * 2, r, o.yaw)
            mb.add(key, T_, N_, colors=np.broadcast_to(col, T_.shape).copy())
            return
        if o.type == "box" and key in ("ceramic",) and o.tag in ("bathtub", "shower"):
            r = min(0.03, 0.4 * float(min(o.half)))
            T_, N_ = A.rounded_box(o.center, o.half * 2, r, o.yaw)
            mb.add(key, T_, N_)
            return
        if o.type in ("cylinder", "sphere"):
            if o.type == "sphere":
                N_ = tris - o.center
            else:
                N_ = tris.copy()
                N_[..., :2] -= o.center[:2]
                N_[..., 2] = 0
                cap = np.abs(fn[:, 2]) > 0.5
                N_[cap] = fn[cap][:, None]
            N_ /= np.linalg.norm(N_, axis=-1, keepdims=True) + 1e-12
        else:
            N_ = np.repeat(fn[:, None], 3, 1)
        colors = np.broadcast_to(col, tris.shape).copy()
        if key == "tile":
            colors *= 0.85 + 0.3 * self.rng.random()
        if key == "parquet":
            top = fn[:, 2] > 0.9
            if (~top).any():
                mb.add("plaster", tris[~top], N_[~top], colors=np.ones_like(tris[~top]) * 0.9)
            tris, N_, colors = tris[top], N_[top], np.ones_like(tris[top])
            if not len(tris):
                return
        if MATERIALS[key][2] in ("grass", "parquet", "bricks", "stone", "asphalt", "bark"):
            colors = np.ones_like(tris)
        mb.add(key, tris, N_, colors=colors)

    def _substitute(self, builders, used):
        """Деревья, кусты, цветы, живые изгороди, унитазы, раковины — процедурными моделями."""
        sc = self.sc
        groups = {}
        for i, o in enumerate(sc.objects):
            groups.setdefault(int(sc.groups[i]), []).append(i)
        n = len(sc.objects)
        stage_of = lambda i: int(min(self.stages - 1, (self.rank[i] * self.stages) // max(1, n)))  # noqa: E731
        cnt = {"tree": 0, "bush": 0, "hedge": 0, "flower": 0, "toilet": 0, "basin": 0}
        for g, idx in groups.items():
            objs = [sc.objects[i] for i in idx]
            tags = {o.tag for o in objs}
            st = builders[stage_of(min(idx))]
            if "tree" in tags:
                trunks = [o for o in objs if o.tag == "tree" and o.type == "cylinder"]
                crowns = [o for o in objs if o.tag == "tree" and o.type == "sphere"]
                if trunks and crowns:
                    tr = trunks[0]
                    lo = np.min([o.aabb_min for o in crowns], 0)
                    hi = np.max([o.aabb_max for o in crowns], 0)
                    A.tree(st, [tr.center[0], tr.center[1], tr.aabb_min[2]], tr.half[2] * 2, tr.half[0],
                           (lo + hi) / 2, (hi - lo) / 2 * 1.15, self.rng)
                    for i in idx:
                        if sc.objects[i].tag == "tree":
                            used[i] = True
                    cnt["tree"] += 1
            for i in idx:
                o = sc.objects[i]
                if o.tag != "plant" or used[i]:
                    continue
                if o.type == "sphere" and o.half[0] * 2 < 0.2:
                    A.flower(st, o.center, o.color, self.rng, size=max(0.08, o.half[0] * 2))
                    used[i] = True
                    cnt["flower"] += 1
                elif o.type == "sphere":
                    A.bush(st, o.center, o.half * np.array([1.05, 1.05, 1.0]), self.rng)
                    used[i] = True
                    cnt["bush"] += 1
                elif o.type == "box" and o.half[2] * 2 > 0.3:
                    A.hedge(st, o.aabb_min, o.aabb_max, self.rng)
                    used[i] = True
                    cnt["hedge"] += 1
            if "toilet" in tags:
                parts = [i for i in idx if sc.objects[i].tag == "toilet"]
                lo = np.min([sc.objects[i].aabb_min for i in parts], 0)
                hi = np.max([sc.objects[i].aabb_max for i in parts], 0)
                tank = max(parts, key=lambda i: sc.objects[i].aabb_max[2])
                d = sc.objects[tank].center[:2] - (lo[:2] + hi[:2]) / 2
                back = np.array([np.sign(d[0]), 0.0]) if abs(d[0]) >= abs(d[1]) else np.array([0.0, np.sign(d[1])])
                seat = max((sc.objects[i] for i in parts if sc.objects[i] is not sc.objects[tank]),
                           key=lambda o: o.aabb_max[2], default=None)
                sh = (seat.aabb_max[2] - lo[2]) if seat is not None else 0.42
                A.toilet(st, lo, hi, back, seat_h=float(np.clip(sh, 0.36, 0.48)))
                for i in parts:
                    used[i] = True
                cnt["toilet"] += 1
            for i in idx:
                o = sc.objects[i]
                if o.tag == "sink" and o.type == "box" and not used[i] and o.half[2] * 2 >= 0.08 and min(o.color) > 0.7:
                    A.basin(st, o.aabb_min, o.aabb_max)
                    used[i] = True
                    cnt["basin"] += 1
        self.log("  фотореализм: заменено " + ", ".join(f"{k} {v}" for k, v in cnt.items() if v))

    def _group(self, mb):
        L = self.o.lib
        models = []
        for key in mb.parts:
            Tt, Nn, UV, Cc = mb.arrays(key)
            if not len(Tt):
                continue
            g = L.ospNewGeometry(b"mesh")
            V = Tt.reshape(-1, 3)
            for name, arr, typ in (("vertex.position", V, T["VEC3F"]), ("vertex.normal", Nn.reshape(-1, 3), T["VEC3F"]),
                                   ("vertex.texcoord", UV.reshape(-1, 2), T["VEC2F"]),
                                   ("vertex.color", Cc.reshape(-1, 3), T["VEC3F"])):
                d = self.o.data(np.ascontiguousarray(arr, np.float32), typ)
                L.ospSetObject(g, name.encode(), d)
                L.ospRelease(d)
            idx = np.arange(len(V), dtype=np.uint32).reshape(-1, 3)
            d = self.o.data(idx, T["VEC3UI"])
            L.ospSetObject(g, b"index", d)
            L.ospRelease(d)
            L.ospCommit(g)
            gm = L.ospNewGeometricModel(g)
            L.ospSetObject(gm, b"material", self.mat(key))
            L.ospCommit(gm)
            models.append(gm)
        if not models:
            return None
        grp = L.ospNewGroup()
        d = self.o.objects(models, T["GEOMETRIC_MODEL"])
        L.ospSetObject(grp, b"geometry", d)
        L.ospCommit(grp)
        inst = L.ospNewInstance(grp)
        L.ospCommit(inst)
        return inst

    def _env(self):
        mb = A.MeshBuilder()
        S = 600.0
        cx, cy = 6.5, 5.0
        g = np.array([[[cx - S, cy - S, 0], [cx + S, cy - S, 0], [cx + S, cy + S, 0]],
                      [[cx - S, cy - S, 0], [cx + S, cy + S, 0], [cx - S, cy + S, 0]]], np.float32)
        # трава с «дырой» под улицей: улица — отдельная полоса чуть выше
        mb.add("grass", g, np.tile([0, 0, 1.0], (2, 3, 1)))
        y0, y1 = C.STREET_Y
        s = np.array([[[-S, y0, 0.012], [S, y0, 0.012], [S, y1, 0.012]], [[-S, y0, 0.012], [S, y1, 0.012], [-S, y1, 0.012]]],
                     np.float32)
        mb.add("asphalt", s, np.tile([0, 0, 1.0], (2, 3, 1)))
        # бордюры
        for yy in (y0, y1):
            Tc, Nc = A.rounded_box([cx, yy, 0.06], [2 * S, 0.2, 0.12], 0.02)
            mb.add("concrete", Tc, Nc)
        self.env = self._group(mb)

    def _screens_init(self):
        """Экраны телевизоров: передняя грань — излучающий материал с картинкой канала."""
        L = self.o.lib
        self.screen_models = {}
        for k, S in self.screens.items():
            c, rgt, up, nrm = S["center"], S["right"], S["up"], S["normal"]
            w, h = S["w"], S["h"]
            p = c + nrm * 0.002
            quad = [p - rgt * w / 2 - up * h / 2, p + rgt * w / 2 - up * h / 2, p + rgt * w / 2 + up * h / 2, p - rgt * w / 2 + up * h / 2]
            V = np.array(quad, np.float32)
            I = np.array([[0, 1, 2], [0, 2, 3]], np.uint32)
            UV = np.array([[0, 0], [1, 0], [1, 1], [0, 1]], np.float32)
            g = L.ospNewGeometry(b"mesh")
            for name, arr, typ in (("vertex.position", V, T["VEC3F"]), ("index", I, T["VEC3UI"]),
                                   ("vertex.texcoord", UV, T["VEC2F"]),
                                   ("vertex.normal", np.tile(nrm, (4, 1)).astype(np.float32), T["VEC3F"])):
                d = self.o.data(arr, typ)
                L.ospSetObject(g, name.encode(), d)
                L.ospRelease(d)
            L.ospCommit(g)
            m = L.ospNewMaterial(b"principled")
            self.o.set(m, baseColor=(0.01, 0.01, 0.01), roughness=0.08, specular=0.8, emissiveColor=(1.0, 1.0, 1.0))
            L.ospCommit(m)
            gm = L.ospNewGeometricModel(g)
            L.ospSetObject(gm, b"material", m)
            L.ospCommit(gm)
            grp = L.ospNewGroup()
            d = self.o.objects([gm], T["GEOMETRIC_MODEL"])
            L.ospSetObject(grp, b"geometry", d)
            L.ospCommit(grp)
            inst = L.ospNewInstance(grp)
            L.ospCommit(inst)
            self.screen_models[k] = (m, gm, grp, inst)
        # корпус экрана (сам примитив tv_screen) — тёмный
        mb = A.MeshBuilder()
        for o in self.sc.objects:
            if o.id in self.screen_ids:
                mb.add("screen_frame", o.triangles(), colors=np.full((len(o.triangles()), 3, 3), 0.02))
        self.screen_body = self._group(mb)

    def set_screen(self, k, img, gain=1.6):
        if k not in self.screen_models:
            return
        L = self.o.lib
        m, gm, grp, inst = self.screen_models[k]
        tex = self.o.texture(img)
        L.ospSetObject(m, b"map_emissiveColor", tex)
        self.o.set(m, emissiveColor=(gain, gain, gain))
        L.ospCommit(m)
        L.ospRelease(tex)
        L.ospCommit(gm)
        L.ospCommit(grp)
        L.ospCommit(inst)

    def _light(self):
        L = self.o.lib
        lt = L.ospNewLight(b"sunSky")
        d = SUN / np.linalg.norm(SUN)
        self.o.set(lt, up=(0.0, 0.0, 1.0), direction=tuple(d), turbidity=2.6, albedo=0.25, horizonExtension=0.15,
                   intensity=0.025)
        L.ospCommit(lt)
        self.light = lt

    # ---- кадр
    def render(self, eye, tgt, fov, W, H, upto=None, spp=8, denoise=True):
        L = self.o.lib
        inst = [self.env, self.screen_body] + [m[3] for m in self.screen_models.values()]
        for s, g in enumerate(self.groups):
            if g is None:
                continue
            if upto is not None and self.stage_start[s] >= upto:
                continue
            inst.append(g)
        inst = [i for i in inst if i]
        world = L.ospNewWorld()
        d = self.o.objects(inst, T["INSTANCE"])
        L.ospSetObject(world, b"instance", d)
        L.ospRelease(d)
        d = self.o.objects([self.light], T["LIGHT"])
        L.ospSetObject(world, b"light", d)
        L.ospRelease(d)
        L.ospCommit(world)
        cam = L.ospNewCamera(b"perspective")
        eye, tgt = np.asarray(eye, float), np.asarray(tgt, float)
        dirv = tgt - eye
        self.o.set(cam, position=tuple(eye), direction=tuple(dirv / np.linalg.norm(dirv)), up=(0.0, 0.0, 1.0),
                   fovy=float(fov), aspect=W / H, nearClip=0.08)
        L.ospCommit(cam)
        self.o.set(self.renderer, pixelSamples=int(spp), maxPathLength=6, roulettePathLength=3,
                   backgroundColor=(0.0, 0.0, 0.0, 1.0))
        L.ospCommit(self.renderer)
        key = (W, H, denoise)
        if key not in self._fb:
            fb = L.ospNewFrameBuffer(W, H, FB_RGBA32F, FB_COLOR | FB_NORMAL | FB_ALBEDO)
            if denoise and self.o.has_denoiser:
                dn = L.ospNewImageOperation(b"denoiser")
                self.o.uint(dn, "quality", DENOISE_HIGH)
                L.ospCommit(dn)
                d = self.o.objects([dn], T["IMAGE_OPERATION"])
                L.ospSetObject(fb, b"imageOperation", d)
            L.ospCommit(fb)
            self._fb[key] = fb
        fb = self._fb[key]
        L.ospResetAccumulation(fb)
        L.ospRenderFrameBlocking(fb, self.renderer, cam, world)
        img = self.o.map(fb, FB_COLOR, np.float32, (H, W, 4))[::-1, :, :3]
        L.ospRelease(cam)
        L.ospRelease(world)
        return img


# ---------------------------------------------------------------------------
# тонмаппинг (ACES) + автоэкспозиция + лёгкое свечение и виньетка
# ---------------------------------------------------------------------------

def luminance(img):
    return 0.2126 * img[..., 0] + 0.7152 * img[..., 1] + 0.0722 * img[..., 2]


def auto_exposure(img, key=0.20):
    lum = luminance(img)
    lg = np.exp(np.mean(np.log(np.clip(lum, 1e-4, None))))
    return float(np.clip(key / lg, 0.05, 40))


def aces(x):
    a, b, c, d, e = 2.51, 0.03, 2.43, 0.59, 0.14
    return np.clip((x * (a * x + b)) / (x * (c * x + d) + e), 0, 1)


def tonemap(img, exposure, bloom=0.06, vignette=0.22):
    from scipy.ndimage import gaussian_filter
    x = img * exposure
    if bloom > 0:
        br = np.clip(x - 1.2, 0, None)
        if br.max() > 0:
            g = np.stack([gaussian_filter(br[..., k], sigma=x.shape[1] / 90) for k in range(3)], -1)
            x = x + bloom * g * 4
    y = aces(x)
    if vignette > 0:
        H, W = y.shape[:2]
        yy, xx = np.mgrid[0:H, 0:W]
        r = np.sqrt(((xx - W / 2) / (W / 2)) ** 2 + ((yy - H / 2) / (H / 2)) ** 2) / math.sqrt(2)
        y = y * (1 - vignette * r[..., None] ** 2.2)
    srgb = np.where(y <= 0.0031308, 12.92 * y, 1.055 * np.power(np.clip(y, 0, 1), 1 / 2.4) - 0.055)
    return (np.clip(srgb, 0, 1) * 255 + 0.5).astype(np.uint8)
