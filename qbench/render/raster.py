"""Небольшой программный рендерер на numpy: z-буфер, тени, контуры.

Без OpenGL и внешних зависимостей, работает в headless-окружении и CI.
"""
from __future__ import annotations

import math

import numpy as np
from PIL import Image

from .. import config as C

SKY_TOP = np.array([0.55, 0.70, 0.86])
SKY_HOR = np.array([0.88, 0.92, 0.96])
GRASS = np.array([0.56, 0.70, 0.45])
GRASS_FAR = np.array([0.64, 0.74, 0.56])
ASPHALT = np.array([0.36, 0.37, 0.39])
OPENING_TAGS = {"window", "door", "garage_door"}


def _norm(v):
    return v / (np.linalg.norm(v, axis=-1, keepdims=True) + 1e-12)


class Mesh:
    """Треугольники всей сцены + атрибуты для затенения."""

    def __init__(self, objects):
        tris, cols, oid, bias = [], [], [], []
        for k, o in enumerate(objects):
            t = o.triangles()
            n = _norm(np.cross(t[:, 1] - t[:, 0], t[:, 2] - t[:, 0]))
            cen = t.mean(1)
            flip = np.einsum("ij,ij->i", n, cen - o.center) < 0
            t[flip] = t[flip][:, ::-1]
            tris.append(t)
            cols.append(np.repeat([o.color], len(t), 0))
            oid.append(np.full(len(t), k))
            # стабильный приоритет при совпадающих гранях (иначе z-fighting)
            bias.append(np.full(len(t), (1e-3 if o.tag in OPENING_TAGS else 0.0) + 1e-7 * k))
        if tris:
            self.tris = np.concatenate(tris)
            self.colors = np.concatenate(cols)
            self.oid = np.concatenate(oid)
            self.bias = np.concatenate(bias)
        else:
            self.tris = np.zeros((0, 3, 3))
            self.colors = np.zeros((0, 3))
            self.oid = np.zeros(0, int)
            self.bias = np.zeros(0)
        self.normals = _norm(np.cross(self.tris[:, 1] - self.tris[:, 0], self.tris[:, 2] - self.tris[:, 0])) \
            if len(self.tris) else np.zeros((0, 3))


class Camera:
    def __init__(self, eye, target, W, H, fov_deg=38.0, up=(0, 0, 1)):
        self.eye = np.asarray(eye, float)
        self.target = np.asarray(target, float)
        self.W, self.H = W, H
        f = _norm(self.target - self.eye)
        r = _norm(np.cross(f, up))
        u = np.cross(r, f)
        self.f, self.r, self.u = f, r, u
        self.fy = (H / 2) / math.tan(math.radians(fov_deg) / 2)

    @classmethod
    def orbit(cls, target, dist, azim_deg, elev_deg, W, H, fov_deg=38.0):
        a, e = math.radians(azim_deg), math.radians(elev_deg)
        eye = np.asarray(target) + dist * np.array([math.cos(e) * math.cos(a), math.cos(e) * math.sin(a), math.sin(e)])
        return cls(eye, target, W, H, fov_deg)

    def project(self, p):
        rel = p - self.eye
        xc, yc, zc = rel @ self.r, rel @ self.u, rel @ self.f
        zs = np.maximum(zc, 1e-6)
        sx = self.W / 2 + self.fy * xc / zs
        sy = self.H / 2 - self.fy * yc / zs
        return sx, sy, zc

    def rays(self):
        xs = (np.arange(self.W) + 0.5 - self.W / 2) / self.fy
        ys = -(np.arange(self.H) + 0.5 - self.H / 2) / self.fy
        X, Y = np.meshgrid(xs, ys)
        d = self.f[None, None] + X[..., None] * self.r + Y[..., None] * self.u
        return _norm(d)


def _rasterize(sx, sy, key, W, H, keybuf, idbuf, idx):
    """sx, sy, key: (N, 3). Записывает в keybuf/idbuf (больший key — ближе)."""
    for t in range(len(sx)):
        x0, x1, x2 = sx[t]
        y0, y1, y2 = sy[t]
        xmin = max(int(math.floor(min(x0, x1, x2))), 0)
        xmax = min(int(math.ceil(max(x0, x1, x2))), W - 1)
        ymin = max(int(math.floor(min(y0, y1, y2))), 0)
        ymax = min(int(math.ceil(max(y0, y1, y2))), H - 1)
        if xmin > xmax or ymin > ymax:
            continue
        area = (x1 - x0) * (y2 - y0) - (x2 - x0) * (y1 - y0)
        if abs(area) < 1e-9:
            continue
        px = np.arange(xmin, xmax + 1) + 0.5
        py = (np.arange(ymin, ymax + 1) + 0.5)[:, None]
        w0 = ((x1 - px) * (y2 - py) - (x2 - px) * (y1 - py)) / area
        w1 = ((x2 - px) * (y0 - py) - (x0 - px) * (y2 - py)) / area
        w2 = 1.0 - w0 - w1
        inside = (w0 >= -1e-7) & (w1 >= -1e-7) & (w2 >= -1e-7)
        if not inside.any():
            continue
        k0, k1, k2 = key[t]
        kv = w0 * k0 + w1 * k1 + w2 * k2
        sub = keybuf[ymin:ymax + 1, xmin:xmax + 1]
        m = inside & (kv > sub)
        sub[m] = kv[m]
        idbuf[ymin:ymax + 1, xmin:xmax + 1][m] = idx[t]


class ShadowMap:
    def __init__(self, mesh: Mesh, light_dir, bounds, res=1400):
        self.L = _norm(np.asarray(light_dir, float))  # направление НА солнце
        f = -self.L
        up = np.array([0, 0, 1.0]) if abs(f[2]) < 0.95 else np.array([0, 1.0, 0])
        self.r = _norm(np.cross(f, up))
        self.u = np.cross(self.r, f)
        self.f = f
        mn, mx = bounds
        corners = np.array([[x, y, z] for x in (mn[0], mx[0]) for y in (mn[1], mx[1]) for z in (mn[2], mx[2])])
        px, py = corners @ self.r, corners @ self.u
        pad = 1.0
        self.x0, self.y0 = px.min() - pad, py.min() - pad
        span = max(px.max() - self.x0, py.max() - self.y0) + pad
        self.res = res
        self.scale = res / span
        self.buf = np.full((res, res), -np.inf)
        if len(mesh.tris):
            t = mesh.tris
            # для теней рисуем грани, отвёрнутые от света (меньше «акне»)
            keep = mesh.normals @ self.L <= 0.02
            t = t[keep]
            sx = (t @ self.r - self.x0) * self.scale
            sy = (t @ self.u - self.y0) * self.scale
            key = -(t @ self.f)
            idb = np.zeros((res, res), int)
            _rasterize(sx, sy, key, res, res, self.buf, idb, np.arange(len(t)))

    def lit(self, P):
        """P: (..., 3) -> доля освещённости 0..1 (PCF 2×2)."""
        sx = (P @ self.r - self.x0) * self.scale - 0.5
        sy = (P @ self.u - self.y0) * self.scale - 0.5
        key = -(P @ self.f)
        out = np.zeros(P.shape[:-1])
        for dx in (0, 1):
            for dy in (0, 1):
                ix = np.clip(np.floor(sx).astype(int) + dx, 0, self.res - 1)
                iy = np.clip(np.floor(sy).astype(int) + dy, 0, self.res - 1)
                occ = self.buf[iy, ix]
                out += (occ <= key + 0.06)
        return out / 4


def render(objects, cam: Camera, ss=2, light_dir=(-0.45, -0.55, 0.70), shadow: ShadowMap | None = None,
           shadow_bounds=None, ground=True, edges=True):
    """Рендерит список Obj. Возвращает PIL.Image размера cam.W/ss × cam.H/ss."""
    W, H = cam.W, cam.H
    mesh = Mesh(objects)
    L = _norm(np.asarray(light_dir, float))
    rays = cam.rays()
    keybuf = np.full((H, W), -np.inf)
    idbuf = np.full((H, W), -1, int)

    # земля аналитически
    ground_mask = np.zeros((H, W), bool)
    if ground:
        rz = rays[..., 2]
        hit = rz < -1e-4
        tg = np.where(hit, -cam.eye[2] / np.where(hit, rz, -1), np.inf)
        dz = tg * (rays @ cam.f)
        hit &= dz < 400
        keybuf[hit] = 1.0 / dz[hit]
        idbuf[hit] = -2
        ground_mask = hit

    # треугольники
    if len(mesh.tris):
        cen = mesh.tris.mean(1)
        front = np.einsum("ij,ij->i", mesh.normals, cen - cam.eye) < 0
        sel = np.nonzero(front)[0]
        t = mesh.tris[sel]
        sx, sy, zc = cam.project(t.reshape(-1, 3))
        sx, sy, zc = sx.reshape(-1, 3), sy.reshape(-1, 3), zc.reshape(-1, 3)
        ok = (zc > 0.2).all(1)
        sel, sx, sy, zc = sel[ok], sx[ok], sy[ok], zc[ok]
        key = (1.0 / zc) * (1 + mesh.bias[sel])[:, None]
        _rasterize(sx, sy, key, W, H, keybuf, idbuf, sel)

    obj_mask = idbuf >= 0
    ground_mask = idbuf == -2
    img = np.zeros((H, W, 3))
    # небо
    rz = np.clip(rays[..., 2], 0, 1)[..., None]
    sky = SKY_HOR + (SKY_TOP - SKY_HOR) * np.sqrt(rz)
    img[:] = sky

    depth = np.where(np.isfinite(keybuf) & (keybuf > 0), 1.0 / np.where(keybuf > 0, keybuf, 1), np.inf)
    P = cam.eye + rays * (depth / (rays @ cam.f))[..., None]

    if shadow is None and len(mesh.tris):
        b = shadow_bounds or (mesh.tris.reshape(-1, 3).min(0), mesh.tris.reshape(-1, 3).max(0))
        shadow = ShadowMap(mesh, L, (np.minimum(b[0], [-8, -10, 0]), np.maximum(b[1], [22, 16, 1])))

    # объекты
    if obj_mask.any():
        tid = idbuf[obj_mask]
        n = mesh.normals[tid]
        base = mesh.colors[tid]
        ndl = np.clip(n @ L, 0, 1)
        lit = shadow.lit(P[obj_mask]) if shadow is not None else 1.0
        lit = np.where(ndl > 0.02, lit, 0.0)
        amb = 0.40 + 0.14 * n[:, 2] + 0.05 * np.abs(n[:, 0])
        shade = amb + 0.62 * ndl * lit
        img[obj_mask] = base * shade[:, None]

    # земля: трава, сетка 1 м, участок, улица
    if ground_mask.any():
        G = P[ground_mask]
        x, y = G[:, 0], G[:, 1]
        dist = np.linalg.norm(G - cam.eye, axis=1)
        col = np.repeat(GRASS[None], len(G), 0)
        far = np.clip((dist - 30) / 80, 0, 1)[:, None]
        col = col * (1 - far) + GRASS_FAR * far
        x0, x1, y0, y1 = C.PLOT
        in_plot = (x >= x0) & (x <= x1) & (y >= y0) & (y <= y1)
        fx, fy = np.abs(x - np.round(x)), np.abs(y - np.round(y))
        lw = np.clip(dist * 0.0012, 0.012, 0.05)
        grid = in_plot & ((fx < lw) | (fy < lw))
        col[grid] *= 0.93
        border = in_plot & ((np.abs(x - x0) < 0.06) | (np.abs(x - x1) < 0.06) | (np.abs(y - y0) < 0.06) | (np.abs(y - y1) < 0.06))
        col[border] = [0.93, 0.93, 0.88]
        street = (y >= C.STREET_Y[0]) & (y <= C.STREET_Y[1])
        col[street] = ASPHALT
        dash = street & (np.abs(y - sum(C.STREET_Y) / 2) < 0.08) & ((np.floor(x / 1.5) % 2) == 0)
        col[dash] = [0.92, 0.92, 0.85]
        curb = (np.abs(y - C.STREET_Y[0]) < 0.12)
        col[curb] = [0.75, 0.75, 0.74]
        lit = shadow.lit(G) if shadow is not None else np.ones(len(G))
        shade = 0.62 + 0.38 * lit
        col = col * shade[:, None]
        fog = np.clip((dist - 60) / 140, 0, 1)[:, None]
        hz = sky[ground_mask]
        img[ground_mask] = col * (1 - fog) + hz * fog

    # контуры
    if edges and obj_mask.any():
        ob = np.where(obj_mask, mesh.oid[np.clip(idbuf, 0, None)], -1)
        nb = np.zeros((H, W, 3))
        nb[obj_mask] = mesh.normals[idbuf[obj_mask]]
        e = np.zeros((H, W), bool)
        for ax in (0, 1):
            a = np.s_[:, :-1] if ax == 1 else np.s_[:-1, :]
            b = np.s_[:, 1:] if ax == 1 else np.s_[1:, :]
            diff = (ob[a] != ob[b]) & ((ob[a] >= 0) | (ob[b] >= 0))
            nd = (np.sum(nb[a] * nb[b], -1) < 0.9) & (ob[a] >= 0) & (ob[b] >= 0)
            da, db = depth[a], depth[b]
            with np.errstate(invalid="ignore"):
                jump = (np.abs(da - db) > 0.03 * np.minimum(da, db)) & (ob[a] >= 0) & (ob[b] >= 0)
            m = diff | nd | jump
            e[a] |= m
        img[e] *= 0.45

    img = np.clip(img, 0, 1)
    im = Image.fromarray((img ** (1 / 1.05) * 255).astype(np.uint8))
    if ss > 1:
        im = im.resize((W // ss, H // ss), Image.LANCZOS)
    return im
