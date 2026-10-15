"""Рендерер v2: векторная растеризация десятков тысяч примитивов, клиппинг по ближней плоскости,
прозрачные стёкла, светящиеся экраны с текстурами, тени, контуры. Только numpy + Pillow."""
from __future__ import annotations

import math

import numpy as np
from PIL import Image

from .. import config as C1

SKY_TOP = np.array([0.50, 0.66, 0.85])
SKY_HOR = np.array([0.86, 0.90, 0.95])
GRASS = np.array([0.52, 0.67, 0.42])
GRASS_FAR = np.array([0.62, 0.72, 0.55])
ASPHALT = np.array([0.34, 0.35, 0.37])
OPENING_TAGS = {"window_frame", "door", "door_frame", "garage_door", "tv_frame", "sill"}


def _norm(v):
    return v / (np.linalg.norm(v, axis=-1, keepdims=True) + 1e-12)


class Mesh:
    """Треугольники сцены, собранные один раз. Подмножества — по маске объектов."""

    def __init__(self, objects, screens=None):
        screens = screens or {}
        tris, cols, oid, flags, scr = [], [], [], [], []
        for k, o in enumerate(objects):
            t = o.triangles()
            n = _norm(np.cross(t[:, 1] - t[:, 0], t[:, 2] - t[:, 0]))
            flip = np.einsum("ij,ij->i", n, t.mean(1) - o.center) < 0
            t[flip] = t[flip][:, ::-1]
            tris.append(t)
            cols.append(np.repeat([o.color], len(t), 0))
            oid.append(np.full(len(t), k))
            f = 0
            if o.tag == "glass":
                f = 1
            elif o.tag in OPENING_TAGS:
                f = 2
            flags.append(np.full(len(t), f))
            si = screens.get(o.id, -1)
            scr.append(np.full(len(t), si))
        if tris:
            self.tris = np.concatenate(tris)
            self.colors = np.concatenate(cols)
            self.oid = np.concatenate(oid)
            self.flags = np.concatenate(flags)
            self.screen = np.concatenate(scr)
        else:
            self.tris = np.zeros((0, 3, 3))
            self.colors = np.zeros((0, 3))
            self.oid = np.zeros(0, int)
            self.flags = np.zeros(0, int)
            self.screen = np.zeros(0, int)
        self.normals = _norm(np.cross(self.tris[:, 1] - self.tris[:, 0], self.tris[:, 2] - self.tris[:, 0])) \
            if len(self.tris) else np.zeros((0, 3))
        self.n_obj = len(objects)

    @staticmethod
    def concat(a, b):
        m = Mesh([])
        off = a.n_obj
        m.tris = np.concatenate([a.tris, b.tris])
        m.colors = np.concatenate([a.colors, b.colors])
        m.oid = np.concatenate([a.oid, b.oid + off])
        m.flags = np.concatenate([a.flags, b.flags])
        m.screen = np.concatenate([a.screen, b.screen])
        m.normals = np.concatenate([a.normals, b.normals])
        m.n_obj = a.n_obj + b.n_obj
        return m

    def subset(self, obj_mask):
        keep = obj_mask[self.oid]
        m = Mesh([])
        m.tris, m.colors, m.oid = self.tris[keep], self.colors[keep], self.oid[keep]
        m.flags, m.screen, m.normals = self.flags[keep], self.screen[keep], self.normals[keep]
        m.n_obj = self.n_obj
        return m


class Camera:
    def __init__(self, eye, target, W, H, fov_deg=50.0, up=(0, 0, 1)):
        self.eye = np.asarray(eye, float)
        self.target = np.asarray(target, float)
        self.W, self.H = W, H
        f = _norm(self.target - self.eye)
        upv = np.asarray(up, float)
        if abs(f @ upv) > 0.995:
            upv = np.array([0, 1.0, 0])
        r = _norm(np.cross(f, upv))
        u = np.cross(r, f)
        self.f, self.r, self.u = f, r, u
        self.fy = (H / 2) / math.tan(math.radians(fov_deg) / 2)
        self.tx = (W / 2) / self.fy
        self.ty = (H / 2) / self.fy

    @classmethod
    def orbit(cls, target, dist, azim_deg, elev_deg, W, H, fov_deg=40.0):
        a, e = math.radians(azim_deg), math.radians(elev_deg)
        eye = np.asarray(target) + dist * np.array([math.cos(e) * math.cos(a), math.cos(e) * math.sin(a), math.sin(e)])
        return cls(eye, target, W, H, fov_deg)

    def view(self, p):
        rel = p - self.eye
        return rel @ self.r, rel @ self.u, rel @ self.f

    def rays(self):
        xs = (np.arange(self.W) + 0.5 - self.W / 2) / self.fy
        ys = -(np.arange(self.H) + 0.5 - self.H / 2) / self.fy
        X, Y = np.meshgrid(xs, ys)
        return _norm(self.f[None, None] + X[..., None] * self.r + Y[..., None] * self.u)


# ---------------------------------------------------------------------------
# растеризация
# ---------------------------------------------------------------------------

def _raster(sx, sy, key, W, H, keybuf, idbuf, ids):
    """sx, sy, key: (N, 3). Больший key — ближе. Мелкие треугольники — пачками, крупные — по одному."""
    if not len(sx):
        return
    xmin = np.clip(np.floor(sx.min(1)), 0, W - 1).astype(int)
    xmax = np.clip(np.ceil(sx.max(1)), 0, W - 1).astype(int)
    ymin = np.clip(np.floor(sy.min(1)), 0, H - 1).astype(int)
    ymax = np.clip(np.ceil(sy.max(1)), 0, H - 1).astype(int)
    vis = (sx.max(1) >= 0) & (sx.min(1) <= W) & (sy.max(1) >= 0) & (sy.min(1) <= H)
    area = (sx[:, 1] - sx[:, 0]) * (sy[:, 2] - sy[:, 0]) - (sx[:, 2] - sx[:, 0]) * (sy[:, 1] - sy[:, 0])
    vis &= np.abs(area) > 1e-9
    w = xmax - xmin + 1
    h = ymax - ymin + 1
    size = np.maximum(w, h)
    for S in (4, 8, 16, 32):
        sel = np.nonzero(vis & (size <= S) & (size > S // 2 if S > 4 else True))[0]
        if len(sel):
            _batch(sel, S, sx, sy, key, area, xmin, ymin, W, H, keybuf, idbuf, ids)
    big = np.nonzero(vis & (size > 32))[0]
    for t in big:
        _one(t, sx, sy, key, area, xmin, xmax, ymin, ymax, W, keybuf, idbuf, ids)


def _batch(sel, S, sx, sy, key, area, xmin, ymin, W, H, keybuf, idbuf, ids):
    chunk = max(1, 400000 // (S * S))
    o = np.arange(S) + 0.5
    for c0 in range(0, len(sel), chunk):
        t = sel[c0:c0 + chunk]
        x0, x1, x2 = sx[t, 0, None, None], sx[t, 1, None, None], sx[t, 2, None, None]
        y0, y1, y2 = sy[t, 0, None, None], sy[t, 1, None, None], sy[t, 2, None, None]
        px = xmin[t, None, None] + o[None, None, :]
        py = ymin[t, None, None] + o[None, :, None]
        a = area[t, None, None]
        w0 = ((x1 - px) * (y2 - py) - (x2 - px) * (y1 - py)) / a
        w1 = ((x2 - px) * (y0 - py) - (x0 - px) * (y2 - py)) / a
        w2 = 1.0 - w0 - w1
        inside = (w0 >= -1e-7) & (w1 >= -1e-7) & (w2 >= -1e-7) & (px < W) & (py < H)
        if not inside.any():
            continue
        kv = w0 * key[t, 0, None, None] + w1 * key[t, 1, None, None] + w2 * key[t, 2, None, None]
        kk, yy, xx = np.nonzero(inside)
        pix = (ymin[t][kk] + yy) * W + (xmin[t][kk] + xx)
        kv = kv[kk, yy, xx]
        tid = ids[t][kk]
        order = np.lexsort((-kv, pix))
        pix, kv, tid = pix[order], kv[order], tid[order]
        first = np.ones(len(pix), bool)
        first[1:] = pix[1:] != pix[:-1]
        pix, kv, tid = pix[first], kv[first], tid[first]
        flat_k = keybuf.reshape(-1)
        better = kv > flat_k[pix]
        flat_k[pix[better]] = kv[better]
        idbuf.reshape(-1)[pix[better]] = tid[better]


def _one(t, sx, sy, key, area, xmin, xmax, ymin, ymax, W, keybuf, idbuf, ids):
    x0, x1, x2 = sx[t]
    y0, y1, y2 = sy[t]
    px = np.arange(xmin[t], xmax[t] + 1) + 0.5
    py = (np.arange(ymin[t], ymax[t] + 1) + 0.5)[:, None]
    a = area[t]
    w0 = ((x1 - px) * (y2 - py) - (x2 - px) * (y1 - py)) / a
    w1 = ((x2 - px) * (y0 - py) - (x0 - px) * (y2 - py)) / a
    w2 = 1.0 - w0 - w1
    inside = (w0 >= -1e-7) & (w1 >= -1e-7) & (w2 >= -1e-7)
    if not inside.any():
        return
    k0, k1, k2 = key[t]
    kv = w0 * k0 + w1 * k1 + w2 * k2
    sub = keybuf[ymin[t]:ymax[t] + 1, xmin[t]:xmax[t] + 1]
    m = inside & (kv > sub)
    sub[m] = kv[m]
    idbuf[ymin[t]:ymax[t] + 1, xmin[t]:xmax[t] + 1][m] = ids[t]


def _clip_near(V, near):
    """V: (N, 3, 3) в координатах камеры (x, y, z). -> треугольники с z ≥ near и индексы исходных."""
    z = V[..., 2]
    inside = z >= near
    cnt = inside.sum(1)
    keep = np.nonzero(cnt == 3)[0]
    out_v = [V[keep]]
    out_i = [keep]
    for t in np.nonzero((cnt == 1) | (cnt == 2))[0]:
        poly = []
        tri = V[t]
        for i in range(3):
            a, b = tri[i], tri[(i + 1) % 3]
            ia, ib = a[2] >= near, b[2] >= near
            if ia:
                poly.append(a)
            if ia != ib:
                s = (near - a[2]) / (b[2] - a[2])
                poly.append(a + s * (b - a))
        for k in range(1, len(poly) - 1):
            out_v.append(np.array([[poly[0], poly[k], poly[k + 1]]]))
            out_i.append(np.array([t]))
    return np.concatenate(out_v), np.concatenate(out_i)


class ShadowMap:
    def __init__(self, mesh: Mesh, light_dir, bounds, res=2048):
        self.L = _norm(np.asarray(light_dir, float))
        f = -self.L
        up = np.array([0, 0, 1.0]) if abs(f[2]) < 0.95 else np.array([0, 1.0, 0])
        self.r = _norm(np.cross(f, up))
        self.u = np.cross(self.r, f)
        self.f = f
        mn, mx = bounds
        cs = np.array([[x, y, z] for x in (mn[0], mx[0]) for y in (mn[1], mx[1]) for z in (mn[2], mx[2])])
        px, py = cs @ self.r, cs @ self.u
        self.x0, self.y0 = px.min() - 0.5, py.min() - 0.5
        span = max(px.max() - self.x0, py.max() - self.y0) + 0.5
        self.res = res
        self.scale = res / span
        self.buf = np.full((res, res), -np.inf)
        if len(mesh.tris):
            keep = (mesh.normals @ self.L <= 0.02) & (mesh.flags != 1) & (mesh.screen < 0)
            t = mesh.tris[keep]
            sx = (t @ self.r - self.x0) * self.scale
            sy = (t @ self.u - self.y0) * self.scale
            key = -(t @ self.f)
            idb = np.zeros((res, res), int)
            _raster(sx, sy, key, res, res, self.buf, idb, np.arange(len(t)))

    def lit(self, P):
        sx = (P @ self.r - self.x0) * self.scale - 0.5
        sy = (P @ self.u - self.y0) * self.scale - 0.5
        key = -(P @ self.f)
        out = np.zeros(P.shape[:-1])
        bx, by = np.floor(sx).astype(int), np.floor(sy).astype(int)
        for dx in (0, 1):
            for dy in (0, 1):
                ix = np.clip(bx + dx, 0, self.res - 1)
                iy = np.clip(by + dy, 0, self.res - 1)
                out += self.buf[iy, ix] <= key + 0.05
        return out / 4


def render(mesh: Mesh, cam: Camera, ss=2, light_dir=(-0.45, -0.55, 0.70), shadow: ShadowMap | None = None,
           screens=None, textures=None, exposure=1.0, ground=True, edges=True, plot=None, street=None,
           near=0.05, hide_mask=None):
    """screens: список dict(center, right, w, h, facing) по индексу; textures: индекс -> RGB float (h, w, 3)."""
    W, H = cam.W, cam.H
    if hide_mask is not None and hide_mask.any():
        mesh = mesh.subset(~hide_mask)
    L = _norm(np.asarray(light_dir, float))
    rays = cam.rays()
    keybuf = np.full((H, W), -np.inf)
    idbuf = np.full((H, W), -1, int)

    if ground:
        rz = rays[..., 2]
        hit = rz < -1e-4
        tg = np.where(hit, -cam.eye[2] / np.where(hit, rz, -1), np.inf)
        dz = tg * (rays @ cam.f)
        hit &= (dz < 600) & (cam.eye[2] > 0.01)
        keybuf[hit] = 1.0 / dz[hit]
        idbuf[hit] = -2

    opaque = mesh.flags != 1
    glass = ~opaque
    tri_glass = None
    if len(mesh.tris):
        cen = mesh.tris.mean(1)
        front = np.einsum("ij,ij->i", mesh.normals, cen - cam.eye) < 0
        rel = mesh.tris - cam.eye
        V = np.stack([rel @ cam.r, rel @ cam.u, rel @ cam.f], -1)
        # отсечение по пирамиде видимости
        z = V[..., 2]
        out = (z < near).all(1)
        out |= (V[..., 0] > z * cam.tx * 1.05 + 0.05).all(1) | (V[..., 0] < -z * cam.tx * 1.05 - 0.05).all(1)
        out |= (V[..., 1] > z * cam.ty * 1.05 + 0.05).all(1) | (V[..., 1] < -z * cam.ty * 1.05 - 0.05).all(1)
        for sel_mask, which in ((front & ~out & opaque, "o"), (front & ~out & glass, "g")):
            idx = np.nonzero(sel_mask)[0]
            if not len(idx):
                continue
            Vc, src = _clip_near(V[idx], near)
            ids = idx[src]
            zc = Vc[..., 2]
            sx = W / 2 + cam.fy * Vc[..., 0] / zc
            sy = H / 2 - cam.fy * Vc[..., 1] / zc
            key = 1.0 / zc
            if which == "o":
                key = key * (1 + np.where(mesh.flags[ids] == 2, 1e-3, 0.0)[:, None]) * (1 + 1e-7 * mesh.oid[ids][:, None] / max(1, mesh.n_obj))
                _raster(sx, sy, key, W, H, keybuf, idbuf, ids)
            else:
                gk = keybuf.copy()
                gid = np.full((H, W), -1, int)
                _raster(sx, sy, key, W, H, gk, gid, ids)
                tri_glass = (gk, gid)

    obj_mask = idbuf >= 0
    ground_mask = idbuf == -2
    img = np.zeros((H, W, 3))
    rzc = np.clip(rays[..., 2], 0, 1)[..., None]
    sky = SKY_HOR + (SKY_TOP - SKY_HOR) * np.sqrt(rzc)
    img[:] = sky
    with np.errstate(divide="ignore", invalid="ignore"):
        depth = np.where(keybuf > 0, 1.0 / keybuf, np.inf)
        P = cam.eye + rays * (depth / (rays @ cam.f))[..., None]

    if obj_mask.any():
        tid = idbuf[obj_mask]
        n = mesh.normals[tid]
        base = mesh.colors[tid]
        ndl = np.clip(n @ L, 0, 1)
        Pm = P[obj_mask]
        lit = shadow.lit(Pm) if shadow is not None else 1.0
        lit = np.where(ndl > 0.02, lit, 0.0)
        amb = 0.42 + 0.14 * n[:, 2] + 0.05 * np.abs(n[:, 0])
        col = base * (amb + 0.62 * ndl * lit)[:, None]
        # экраны: текстура, свечение
        sc_idx = mesh.screen[tid]
        if screens and (sc_idx >= 0).any():
            for si in np.unique(sc_idx[sc_idx >= 0]):
                S = screens[si]
                m = (sc_idx == si) & (n @ S["normal"] > 0.9)
                if not m.any():
                    continue
                rel = Pm[m] - S["center"]
                u = rel @ S["right"] / S["w"] + 0.5
                v = 0.5 - rel[:, 2] / S["h"]
                tex = textures.get(int(si)) if textures else None
                if tex is None:
                    c = np.tile([0.05, 0.06, 0.08], (m.sum(), 1))
                else:
                    th, tw = tex.shape[:2]
                    fu = np.clip(u, 0, 0.9999) * (tw - 1)
                    fv = np.clip(v, 0, 0.9999) * (th - 1)
                    iu, iv = fu.astype(int), fv.astype(int)
                    du, dv = (fu - iu)[:, None], (fv - iv)[:, None]
                    iu1, iv1 = np.minimum(iu + 1, tw - 1), np.minimum(iv + 1, th - 1)
                    c = (tex[iv, iu] * (1 - du) * (1 - dv) + tex[iv, iu1] * du * (1 - dv) +
                         tex[iv1, iu] * (1 - du) * dv + tex[iv1, iu1] * du * dv)
                col[m] = c * 1.02 / max(exposure, 1e-3)
        img[obj_mask] = col

    if ground_mask.any():
        G = P[ground_mask]
        x, y = G[:, 0], G[:, 1]
        dist = np.linalg.norm(G - cam.eye, axis=1)
        col = np.repeat(GRASS[None], len(G), 0)
        far = np.clip((dist - 35) / 90, 0, 1)[:, None]
        col = col * (1 - far) + GRASS_FAR * far
        if plot is not None:
            x0, x1, y0, y1 = plot
            inp = (x >= x0) & (x <= x1) & (y >= y0) & (y <= y1)
            fx, fy = np.abs(x - np.round(x)), np.abs(y - np.round(y))
            lw = np.clip(dist * 0.001, 0.01, 0.04)
            col[inp & ((fx < lw) | (fy < lw))] *= 0.95
        if street is not None:
            st = (y >= street[0]) & (y <= street[1])
            col[st] = ASPHALT
            dash = st & (np.abs(y - sum(street) / 2) < 0.08) & ((np.floor(x / 1.5) % 2) == 0)
            col[dash] = [0.92, 0.92, 0.85]
            col[np.abs(y - street[0]) < 0.12] = [0.74, 0.74, 0.73]
        lit = shadow.lit(G) if shadow is not None else np.ones(len(G))
        col = col * (0.60 + 0.40 * lit)[:, None]
        fog = np.clip((dist - 70) / 160, 0, 1)[:, None]
        img[ground_mask] = col * (1 - fog) + sky[ground_mask] * fog

    # стекло: один слой полупрозрачности
    if tri_glass is not None:
        gk, gid = tri_glass
        gm = gid >= 0
        if gm.any():
            tint = mesh.colors[gid[gm]]
            nrm = mesh.normals[gid[gm]]
            refl = np.clip(0.25 + 0.5 * (1 - np.abs(np.einsum("ij,ij->i", nrm, -rays[gm]))), 0, 1)[:, None]
            img[gm] = img[gm] * 0.62 + tint * 0.28 + SKY_HOR * 0.10 * refl

    if edges and obj_mask.any():
        ob = np.where(obj_mask, mesh.oid[np.clip(idbuf, 0, None)], -1)
        nb = np.zeros((H, W, 3))
        nb[obj_mask] = mesh.normals[idbuf[obj_mask]]
        e = np.zeros((H, W), bool)
        for ax in (0, 1):
            a = np.s_[:, :-1] if ax == 1 else np.s_[:-1, :]
            b = np.s_[:, 1:] if ax == 1 else np.s_[1:, :]
            both = (ob[a] >= 0) & (ob[b] >= 0)
            diff = (ob[a] != ob[b]) & ((ob[a] >= 0) | (ob[b] >= 0))
            nd = (np.sum(nb[a] * nb[b], -1) < 0.9) & both
            with np.errstate(invalid="ignore"):
                da, db = depth[a], depth[b]
                jump = (np.abs(da - db) > 0.025 * np.minimum(da, db)) & both
            e[a] |= diff | nd | jump
        dd = np.where(np.isfinite(depth), depth, 1e9)
        strength = np.clip(1.0 - (dd - 8) / 60, 0.35, 1.0)
        img[e] *= (1 - 0.5 * strength[e])[:, None]

    img = np.clip(img * exposure, 0, 1)
    im = Image.fromarray((img ** (1 / 1.05) * 255).astype(np.uint8))
    if ss > 1:
        im = im.resize((W // ss, H // ss), Image.LANCZOS)
    return im


def screen_info(o, facing):
    f = {"+x": (1, 0, 0), "-x": (-1, 0, 0), "+y": (0, 1, 0), "-y": (0, -1, 0)}.get(str(facing), (0, -1, 0))
    nrm = np.array(f, float)
    right = np.cross(-nrm, [0, 0, 1.0])
    e = o.aabb_max - o.aabb_min
    w = e[1] if abs(nrm[0]) > 0.5 else e[0]
    return {"center": o.center + nrm * (e @ np.abs(nrm)) / 2, "right": right, "w": float(w), "h": float(e[2]),
            "normal": nrm}
