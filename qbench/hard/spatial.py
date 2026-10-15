"""Пространственные помощники: хеш-сетка контактов, карты занятости, лучи, достижимость."""
from __future__ import annotations

import math

import numpy as np
from scipy import ndimage


def touching_pairs(amin, amax, eps=0.03, cell=1.5):
    """Пары (i, j), i < j, у которых AABB касаются/пересекаются (с допуском eps)."""
    n = len(amin)
    if n < 2:
        return np.zeros((0, 2), int)
    lo = np.floor((amin[:, :2] - eps) / cell).astype(int)
    hi = np.floor((amax[:, :2] + eps) / cell).astype(int)
    span = (hi - lo + 1)
    counts = span[:, 0] * span[:, 1]
    idx = np.repeat(np.arange(n), counts)
    # координаты ячеек для каждого вхождения
    offs = np.concatenate([np.arange(c) for c in counts]) if n else np.zeros(0, int)
    w = np.repeat(span[:, 1], counts)
    cx = np.repeat(lo[:, 0], counts) + offs // w
    cy = np.repeat(lo[:, 1], counts) + offs % w
    key = (cx.astype(np.int64) + 100000) * 200003 + (cy.astype(np.int64) + 100000)
    order = np.argsort(key, kind="stable")
    key, idx = key[order], idx[order]
    bounds = np.flatnonzero(np.diff(key)) + 1
    groups = np.split(idx, bounds)
    out = []
    for g in groups:
        k = len(g)
        if k < 2:
            continue
        if k > 1500:  # очень плотная ячейка — режем на куски по x
            g = g[np.argsort(amin[g, 0])]
        a, b = np.triu_indices(k, 1)
        i, j = g[a], g[b]
        ok = np.all((amin[i] <= amax[j] + eps) & (amin[j] <= amax[i] + eps), 1)
        if ok.any():
            out.append(np.stack([np.minimum(i[ok], j[ok]), np.maximum(i[ok], j[ok])], 1))
    if not out:
        return np.zeros((0, 2), int)
    pairs = np.concatenate(out)
    code = pairs[:, 0].astype(np.int64) * n + pairs[:, 1]
    _, u = np.unique(code, return_index=True)
    return pairs[u]


def surface_touch(a, b, eps):
    """Касание с учётом наклонных поверхностей wedge/gable (как в v1)."""
    if b.type not in ("wedge", "gable"):
        return True
    xs = np.linspace(a.aabb_min[0], a.aabb_max[0], 4)
    ys = np.linspace(a.aabb_min[1], a.aabb_max[1], 4)
    X, Y = np.meshgrid(xs, ys)
    t = b.top_at(X, Y)
    if np.isfinite(t).any() and np.nanmax(t) >= a.aabb_min[2] - eps:
        return True
    return b.aabb_min[2] >= a.aabb_max[2] - eps


def floating(objs, amin, amax, eps=0.03):
    n = len(objs)
    if n == 0:
        return []
    pairs = touching_pairs(amin, amax, eps)
    shaped = np.array([o.type in ("wedge", "gable") for o in objs])
    keep = np.ones(len(pairs), bool)
    for k in np.nonzero(shaped[pairs[:, 0]] | shaped[pairs[:, 1]])[0]:
        i, j = pairs[k]
        if not (surface_touch(objs[i], objs[j], eps) and surface_touch(objs[j], objs[i], eps)):
            keep[k] = False
    pairs = pairs[keep]
    # объединение компонент связности
    parent = np.arange(n)

    def find(x):
        while parent[x] != x:
            parent[x] = parent[parent[x]]
            x = parent[x]
        return x
    for i, j in pairs:
        ri, rj = find(i), find(j)
        if ri != rj:
            parent[ri] = rj
    roots = np.array([find(i) for i in range(n)])
    grounded = set(roots[amin[:, 2] <= 0.05].tolist())
    return [objs[i] for i in range(n) if roots[i] not in grounded]


# ---------------------------------------------------------------------------
# карты занятости на этаже
# ---------------------------------------------------------------------------

class Occupancy:
    def __init__(self, scene, z0, z1, extent, res=0.05, exclude_tags=(), exclude_idx=(), min_height=0.0):
        self.x0, self.x1, self.y0, self.y1 = extent
        self.res = res
        nx = int(math.ceil((self.x1 - self.x0) / res))
        ny = int(math.ceil((self.y1 - self.y0) / res))
        self.grid = np.zeros((nx, ny), bool)
        sel = (scene.amax[:, 2] > z0) & (scene.amin[:, 2] < z1)
        sel &= (scene.amax[:, 0] > self.x0) & (scene.amin[:, 0] < self.x1)
        sel &= (scene.amax[:, 1] > self.y0) & (scene.amin[:, 1] < self.y1)
        if exclude_tags:
            sel &= ~np.isin(scene.tags, list(exclude_tags))
        if min_height:
            sel &= (scene.amax[:, 2] - scene.amin[:, 2]) >= min_height
        ex = set(int(i) for i in exclude_idx)
        xs = self.x0 + (np.arange(nx) + 0.5) * res
        ys = self.y0 + (np.arange(ny) + 0.5) * res
        for i in np.nonzero(sel)[0]:
            if int(i) in ex:
                continue
            o = scene.objects[i]
            ia = max(0, int((o.aabb_min[0] - self.x0) / res))
            ib = min(nx, int(math.ceil((o.aabb_max[0] - self.x0) / res)))
            ja = max(0, int((o.aabb_min[1] - self.y0) / res))
            jb = min(ny, int(math.ceil((o.aabb_max[1] - self.y0) / res)))
            if ia >= ib or ja >= jb:
                continue
            if o.type == "box" and abs(math.sin(o.yaw * 2)) < 1e-6:
                self.grid[ia:ib, ja:jb] = True
                continue
            X, Y = np.meshgrid(xs[ia:ib], ys[ja:jb], indexing="ij")
            lx, ly = o.to_local_xy(X, Y)
            m = o._inside_xy(lx, ly)
            self.grid[ia:ib, ja:jb] |= m

    def cell(self, x, y):
        return int((x - self.x0) / self.res), int((y - self.y0) / self.res)

    def coords(self):
        nx, ny = self.grid.shape
        xs = self.x0 + (np.arange(nx) + 0.5) * self.res
        ys = self.y0 + (np.arange(ny) + 0.5) * self.res
        return np.meshgrid(xs, ys, indexing="ij")

    def free_with_clearance(self, radius):
        dist = ndimage.distance_transform_edt(~self.grid) * self.res
        return dist >= radius

    def reachable(self, free, start):
        i, j = self.cell(*start)
        nx, ny = free.shape
        if not (0 <= i < nx and 0 <= j < ny):
            return np.zeros_like(free)
        if not free[i, j]:
            # ближайшая свободная клетка в радиусе 0.4 м
            r = int(0.4 / self.res)
            sub = free[max(0, i - r):i + r + 1, max(0, j - r):j + r + 1]
            pts = np.argwhere(sub)
            if not len(pts):
                return np.zeros_like(free)
            d = ((pts - [min(r, i), min(r, j)]) ** 2).sum(1)
            pi, pj = pts[np.argmin(d)]
            i, j = max(0, i - r) + pi, max(0, j - r) + pj
        lab, _ = ndimage.label(free)
        if lab[i, j] == 0:
            return np.zeros_like(free)
        return lab == lab[i, j]

    def rect_mask(self, rect, inset=0.0):
        X, Y = self.coords()
        x0, y0, x1, y1 = rect
        return (X >= x0 + inset) & (X <= x1 - inset) & (Y >= y0 + inset) & (Y <= y1 - inset)


def segment_hits(scene, p0, p1, exclude=(), skip_tags=("rug", "glass")):
    """Индексы примитивов, пересекающих отрезок p0→p1 (по AABB, slab-метод)."""
    p0, p1 = np.asarray(p0, float), np.asarray(p1, float)
    d = p1 - p0
    with np.errstate(divide="ignore", invalid="ignore"):
        inv = np.where(np.abs(d) > 1e-12, 1.0 / d, np.inf)
        t1 = (scene.amin - p0) * inv
        t2 = (scene.amax - p0) * inv
    tmin = np.nanmax(np.minimum(t1, t2), 1)
    tmax = np.nanmin(np.maximum(t1, t2), 1)
    par = np.abs(d) <= 1e-12
    inside = np.all(~par | ((p0 >= scene.amin - 1e-9) & (p0 <= scene.amax + 1e-9)), 1)
    hit = inside & (tmax >= np.maximum(tmin, 0.02)) & (tmin <= 0.98)
    if skip_tags:
        hit &= ~np.isin(scene.tags, list(skip_tags))
    ex = np.zeros(len(hit), bool)
    ex[list(exclude)] = True
    return np.nonzero(hit & ~ex)[0]
