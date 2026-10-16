"""Процедурные модели для реалистичного рендера: скруглённые коробки, дерево, куст, живая изгородь,
цветок, унитаз, раковина. Все функции пишут треугольники в MeshBuilder (по ключу материала).

Модели строятся в габаритах и на месте групп примитивов, которые сделала тестируемая модель:
оценка дома не меняется, меняется только картинка.
"""
from __future__ import annotations

import math

import numpy as np


class MeshBuilder:
    """Треугольники по материалам: позиции, нормали (по вершинам), uv (метры), цвета вершин."""

    def __init__(self):
        self.parts = {}

    def add(self, key, tris, normals=None, uv=None, colors=None):
        tris = np.asarray(tris, np.float32).reshape(-1, 3, 3)
        if not len(tris):
            return
        if normals is None:
            n = np.cross(tris[:, 1] - tris[:, 0], tris[:, 2] - tris[:, 0])
            n /= np.linalg.norm(n, axis=1, keepdims=True) + 1e-12
            normals = np.repeat(n[:, None], 3, 1)
        if uv is None:
            uv = box_uv(tris, np.asarray(normals, np.float32).reshape(-1, 3, 3).mean(1))
        if colors is None:
            colors = np.ones_like(tris)
        p = self.parts.setdefault(key, [[], [], [], []])
        p[0].append(tris)
        p[1].append(np.asarray(normals, np.float32).reshape(-1, 3, 3))
        p[2].append(np.asarray(uv, np.float32).reshape(-1, 3, 2))
        p[3].append(np.asarray(colors, np.float32).reshape(-1, 3, 3))

    def arrays(self, key):
        p = self.parts[key]
        return [np.concatenate(x) for x in p]

    def merge(self, other):
        for k, p in other.parts.items():
            q = self.parts.setdefault(k, [[], [], [], []])
            for i in range(4):
                q[i] += p[i]


def box_uv(tris, n):
    """uv в метрах проекцией на плоскость, перпендикулярную главной оси нормали."""
    ax = np.argmax(np.abs(n), 1)
    uv = np.empty(tris.shape[:2] + (2,), np.float32)
    for a, (i, j) in enumerate(((1, 2), (0, 2), (0, 1))):
        m = ax == a
        uv[m, :, 0] = tris[m][:, :, i]
        uv[m, :, 1] = tris[m][:, :, j]
    return uv


def _rotz(yaw):
    c, s = math.cos(math.radians(yaw)), math.sin(math.radians(yaw))
    return np.array([[c, -s, 0], [s, c, 0], [0, 0, 1.0]])


# ---------------------------------------------------------------------------
# скруглённая коробка (сумма Минковского внутренней коробки и шара)
# ---------------------------------------------------------------------------

def rounded_box(center, size, radius, yaw=0.0, k=3):
    h = np.asarray(size, float) / 2
    r = float(min(radius, h.min() * 0.95))
    inner = h - r

    def coords(H, I):
        if I <= 1e-9:
            return np.array([-H, H])
        side = [I + r * math.sin(math.radians(a)) for a in np.linspace(0, 90, k + 1)]
        return np.array(sorted(set([-s for s in side] + side)))
    cs = [coords(h[i], inner[i]) for i in range(3)]
    tris, nors = [], []
    for ax in range(3):
        a1, a2 = [i for i in range(3) if i != ax]
        for sgn in (-1, 1):
            U, V = np.meshgrid(cs[a1], cs[a2], indexing="ij")
            Q = np.zeros(U.shape + (3,))
            Q[..., ax] = sgn * h[ax]
            Q[..., a1] = U
            Q[..., a2] = V
            Cc = np.clip(Q, -inner, inner)
            D = Q - Cc
            D[..., ax] = np.where(np.abs(D[..., ax]) < 1e-12, sgn * 1e-6, D[..., ax])
            N = D / np.linalg.norm(D, axis=-1, keepdims=True)
            P = Cc + r * N
            nu, nv = U.shape
            for i in range(nu - 1):
                for j in range(nv - 1):
                    q = [(i, j), (i + 1, j), (i + 1, j + 1), (i, j + 1)]
                    if sgn * (1 if (ax, a1, a2) in ((0, 1, 2), (1, 0, 2), (2, 0, 1)) else 1) < 0:
                        q = q[::-1]
                    a, b, c, d = [P[x] for x in q]
                    na, nb, nc, nd = [N[x] for x in q]
                    tris += [[a, b, c], [a, c, d]]
                    nors += [[na, nb, nc], [na, nc, nd]]
    T = np.array(tris)
    Nn = np.array(nors)
    # ориентация: нормаль треугольника наружу
    fn = np.cross(T[:, 1] - T[:, 0], T[:, 2] - T[:, 0])
    flip = np.einsum("ij,ij->i", fn, Nn.mean(1)) < 0
    T[flip] = T[flip][:, ::-1]
    Nn[flip] = Nn[flip][:, ::-1]
    R = _rotz(yaw)
    T = T @ R.T + np.asarray(center, float)
    Nn = Nn @ R.T
    return T, Nn


# ---------------------------------------------------------------------------
# тела вращения (с эллиптическим сечением)
# ---------------------------------------------------------------------------

def lathe(profile, seg=24, sx=1.0, sy=1.0, center=(0, 0, 0), R=None, close_top=False):
    """profile: [(r, z)] снизу вверх; sx, sy — растяжение по осям (эллипс). -> (tris, normals)."""
    prof = np.asarray(profile, float)
    a = np.linspace(0, 2 * np.pi, seg + 1)
    ring = np.stack([np.cos(a) * sx, np.sin(a) * sy], 1)
    tris, nors = [], []
    for i in range(len(prof) - 1):
        (r0, z0), (r1, z1) = prof[i], prof[i + 1]
        dr, dz = r1 - r0, z1 - z0
        for j in range(seg):
            p = [[ring[j][0] * r0, ring[j][1] * r0, z0], [ring[j + 1][0] * r0, ring[j + 1][1] * r0, z0],
                 [ring[j + 1][0] * r1, ring[j + 1][1] * r1, z1], [ring[j][0] * r1, ring[j][1] * r1, z1]]
            nn = []
            for jj in (j, j + 1, j + 1, j):
                c, s = math.cos(a[jj]), math.sin(a[jj])
                v = np.array([c / sx * dz, s / sy * dz, -dr])
                nn.append(v / (np.linalg.norm(v) + 1e-12))
            tris += [[p[0], p[1], p[2]], [p[0], p[2], p[3]]]
            nors += [[nn[0], nn[1], nn[2]], [nn[0], nn[2], nn[3]]]
    if close_top:
        r, z = prof[-1]
        for j in range(seg):
            tris.append([[0, 0, z], [ring[j][0] * r, ring[j][1] * r, z], [ring[j + 1][0] * r, ring[j + 1][1] * r, z]])
            nors.append([[0, 0, 1]] * 3)
    T, N = np.array(tris), np.array(nors)
    if R is not None:
        T, N = T @ R.T, N @ R.T
    return T + np.asarray(center, float), N


def tube(p0, p1, r0, r1, seg=7):
    """Сужающийся цилиндр между точками (ветки, стебли)."""
    p0, p1 = np.asarray(p0, float), np.asarray(p1, float)
    d = p1 - p0
    L = np.linalg.norm(d)
    if L < 1e-6:
        return np.zeros((0, 3, 3)), np.zeros((0, 3, 3))
    w = d / L
    u = np.cross(w, [0, 0, 1.0] if abs(w[2]) < 0.9 else [1.0, 0, 0])
    u /= np.linalg.norm(u)
    v = np.cross(w, u)
    a = np.linspace(0, 2 * np.pi, seg + 1)
    tris, nors = [], []
    for j in range(seg):
        c0, s0, c1, s1 = math.cos(a[j]), math.sin(a[j]), math.cos(a[j + 1]), math.sin(a[j + 1])
        n0, n1 = c0 * u + s0 * v, c1 * u + s1 * v
        A, B = p0 + r0 * n0, p0 + r0 * n1
        Cc, D = p1 + r1 * n1, p1 + r1 * n0
        tris += [[A, B, Cc], [A, Cc, D]]
        nors += [[n0, n1, n1], [n0, n1, n0]]
    return np.array(tris), np.array(nors)


# ---------------------------------------------------------------------------
# листва
# ---------------------------------------------------------------------------

LEAF_COLORS = np.array([[0.13, 0.30, 0.07], [0.20, 0.38, 0.09], [0.28, 0.44, 0.12], [0.16, 0.26, 0.08], [0.33, 0.42, 0.13]])


def leaves(points, size, rng, up_bias=0.5, colors=LEAF_COLORS):
    """Листья-четырёхугольники в точках (n, 3) -> (tris, normals, colors)."""
    n = len(points)
    if n == 0:
        return np.zeros((0, 3, 3)), np.zeros((0, 3, 3)), np.zeros((0, 3, 3))
    nrm = rng.normal(size=(n, 3))
    nrm[:, 2] += up_bias * 2.5
    nrm /= np.linalg.norm(nrm, axis=1, keepdims=True)
    t = np.cross(nrm, rng.normal(size=(n, 3)))
    t /= np.linalg.norm(t, axis=1, keepdims=True) + 1e-12
    b = np.cross(nrm, t)
    s = size * (0.7 + 0.6 * rng.random(n))[:, None]
    L, Wd = s * 1.0, s * 0.55
    P = np.asarray(points, float)
    v0 = P - t * L * 0.5
    v1 = P + b * Wd * 0.5
    v2 = P + t * L * 0.5
    v3 = P - b * Wd * 0.5
    T = np.stack([np.stack([v0, v1, v2], 1), np.stack([v0, v2, v3], 1)], 1).reshape(-1, 3, 3)
    N = np.repeat(np.repeat(nrm[:, None], 3, 1)[:, None], 2, 1).reshape(-1, 3, 3)
    col = colors[rng.integers(0, len(colors), n)] * (0.8 + 0.4 * rng.random((n, 1)))
    Cc = np.repeat(np.repeat(col[:, None], 3, 1)[:, None], 2, 1).reshape(-1, 3, 3)
    return T, N, Cc


def tree(mb, base, trunk_h, trunk_r, crown_c, crown_r, rng):
    """Дерево: изогнутый ствол, ветви трёх порядков, листва гроздьями в эллипсоиде кроны."""
    base = np.asarray(base, float)
    crown_c = np.asarray(crown_c, float)
    crown_r = np.asarray(crown_r, float)
    top = np.array([base[0] + rng.normal(0, 0.05), base[1] + rng.normal(0, 0.05), crown_c[2] + crown_r[2] * 0.35])
    pts = [base + (top - base) * t + np.array([math.sin(t * 3.1 + rng.random()) * 0.06, math.cos(t * 2.3) * 0.05, 0])
           for t in np.linspace(0, 1, 7)]
    rads = np.linspace(trunk_r * 1.15, trunk_r * 0.35, 7)
    for i in range(6):
        T, N = tube(pts[i], pts[i + 1], rads[i], rads[i + 1], 9)
        mb.add("bark", T, N)
    # корневой «расклёш»
    T, N = lathe([(trunk_r * 1.9, 0.0), (trunk_r * 1.25, 0.12), (trunk_r * 1.1, 0.3)], 12, center=base)
    mb.add("bark", T, N)
    tips = []

    def branch(p, d, length, r, depth):
        steps = 3
        cur = p.copy()
        for s in range(steps):
            d = d + rng.normal(0, 0.18, 3)
            d[2] += 0.08
            d /= np.linalg.norm(d)
            nxt = cur + d * length / steps
            # не выходить за крону сильно
            rel = (nxt - crown_c) / crown_r
            if np.linalg.norm(rel) > 1.05:
                nxt = crown_c + rel / np.linalg.norm(rel) * 1.0 * crown_r
            r1 = r * (1 - 0.25 * (s + 1) / steps)
            T, N = tube(cur, nxt, r * (1 - 0.25 * s / steps), r1, 6 if depth == 0 else 5)
            mb.add("bark", T, N)
            cur = nxt
        tips.append(cur)
        if depth < 2:
            for _ in range(rng.integers(2, 4)):
                nd = d + rng.normal(0, 0.6, 3)
                nd[2] = abs(nd[2]) * 0.6 + 0.2
                branch(cur - d * length * rng.uniform(0, 0.5), nd / np.linalg.norm(nd), length * 0.6, r * 0.55, depth + 1)
    nb = rng.integers(6, 9)
    for k in range(nb):
        h = rng.uniform(0.45, 0.95)
        p = base + (top - base) * h
        az = 2 * math.pi * (k + rng.random() * 0.6) / nb
        el = math.radians(rng.uniform(25, 55))
        d = np.array([math.cos(az) * math.cos(el), math.sin(az) * math.cos(el), math.sin(el)])
        branch(p, d, float(np.mean(crown_r[:2])) * rng.uniform(0.7, 1.0), trunk_r * 0.45, 0)
    # листва: гроздья у концов веток + наполнение кроны
    P = []
    for t in tips:
        P.append(t + rng.normal(0, 0.22, (40, 3)))
    fill = rng.normal(size=(2500, 3))
    fill /= np.linalg.norm(fill, axis=1, keepdims=True)
    fill *= rng.random((2500, 1)) ** 0.35
    P.append(crown_c + fill * crown_r)
    P = np.concatenate(P)
    rel = (P - crown_c) / crown_r
    P = P[np.linalg.norm(rel, axis=1) < 1.08]
    T, N, Cc = leaves(P, 0.14, rng)
    mb.add("leaf", T, N, colors=Cc)


def bush(mb, center, radii, rng, density=1.0):
    center, radii = np.asarray(center, float), np.asarray(radii, float)
    n = int(1400 * density * np.prod(radii) / 0.064 ** 1) if False else int(min(4000, 900 * density * (radii.mean() / 0.4) ** 2))
    d = rng.normal(size=(n, 3))
    d /= np.linalg.norm(d, axis=1, keepdims=True)
    d *= rng.random((n, 1)) ** 0.25
    P = center + d * radii
    P[:, 2] = np.maximum(P[:, 2], center[2] - radii[2] * 0.9)
    T, N, Cc = leaves(P, 0.09, rng, up_bias=0.3)
    mb.add("leaf", T, N, colors=Cc)


def hedge(mb, lo, hi, rng):
    lo, hi = np.asarray(lo, float), np.asarray(hi, float)
    vol = np.prod(hi - lo)
    n = int(min(6000, 5200 * vol))
    P = lo + rng.random((n, 3)) * (hi - lo)
    # уплотнение к поверхности
    c = (lo + hi) / 2
    h = (hi - lo) / 2
    rel = (P - c) / h
    k = np.max(np.abs(rel), axis=1, keepdims=True)
    P = c + rel / k * (k ** 0.3) * h
    T, N, Cc = leaves(P, 0.08, rng, up_bias=0.2,
                      colors=np.array([[0.10, 0.25, 0.06], [0.14, 0.31, 0.08], [0.18, 0.35, 0.09]]))
    mb.add("leaf", T, N, colors=Cc)


def flower(mb, p, color, rng, size=0.12):
    p = np.asarray(p, float)
    T, N = tube(p - [0, 0, p[2] - max(0.0, p[2] - 0.35)], p, 0.006, 0.004, 4)
    mb.add("stem", T, N)
    k = 5
    tris, nors = [], []
    for i in range(k):
        a = 2 * math.pi * i / k + rng.random()
        d = np.array([math.cos(a), math.sin(a), 0.25])
        tip = p + d * size * 0.45
        side = np.cross(d, [0, 0, 1.0])
        side /= np.linalg.norm(side) + 1e-9
        tris.append([p, tip + side * size * 0.15, tip - side * size * 0.15])
        nors.append([[0, 0, 1.0]] * 3)
    col = np.repeat(np.repeat(np.asarray(color, float)[None, None], 3, 1), k, 0)
    mb.add("petal", np.array(tris), np.array(nors), colors=col)


# ---------------------------------------------------------------------------
# сантехника
# ---------------------------------------------------------------------------

def toilet(mb, lo, hi, back, seat_h=0.42):
    """Унитаз в габаритах (lo, hi); back — единичный вектор (x/y) от центра к бачку."""
    lo, hi = np.asarray(lo, float), np.asarray(hi, float)
    c = (lo + hi) / 2
    back = np.asarray(back, float)
    ext = hi - lo
    Lx = float(abs(back[0]) * ext[0] + abs(back[1]) * ext[1])  # длина вдоль «вперёд»
    Wy = float(abs(back[1]) * ext[0] + abs(back[0]) * ext[1])
    fz = lo[2]
    yaw = math.degrees(math.atan2(-back[1], -back[0]))  # локальная +x — вперёд (от бачка)
    R = _rotz(yaw)
    tank_d = min(0.2, Lx * 0.3)
    bowl_len = Lx - tank_d
    # центр чаши
    bc_local = np.array([-Lx / 2 + tank_d + bowl_len * 0.55, 0, 0])
    sx, sy = bowl_len * 0.55 / 0.2, min(Wy, 0.40) / 0.36
    prof = [(0.11, 0.0), (0.12, 0.05), (0.13, 0.15), (0.155, 0.25), (0.18, 0.33), (0.195, seat_h - 0.05), (0.19, seat_h - 0.02)]
    T, N = lathe(prof, 32, sx * 0.95, sy * 0.95, close_top=True)
    mb.add("ceramic", (T + bc_local) @ R.T + [c[0], c[1], fz], N @ R.T)
    # сиденье и крышка
    ring = [(0.2, seat_h - 0.02), (0.205, seat_h - 0.005), (0.2, seat_h + 0.012), (0.17, seat_h + 0.02), (0.0, seat_h + 0.022)]
    T, N = lathe(ring, 32, sx, sy)
    mb.add("plastic_white", (T + bc_local) @ R.T + [c[0], c[1], fz], N @ R.T)
    # перемычка к бачку
    T, N = rounded_box([-Lx / 2 + tank_d + 0.06, 0, (seat_h - 0.05) / 2], [0.16, 0.2, seat_h - 0.05], 0.05)
    mb.add("ceramic", T @ R.T + [c[0], c[1], fz], N @ R.T)
    # бачок
    th = max(0.3, min(0.42, (hi[2] - fz) - seat_h + 0.02))
    T, N = rounded_box([-Lx / 2 + tank_d / 2, 0, seat_h + th / 2], [tank_d, min(Wy, 0.46), th], 0.035)
    mb.add("ceramic", T @ R.T + [c[0], c[1], fz], N @ R.T)
    T, N = rounded_box([-Lx / 2 + tank_d / 2, 0, seat_h + th + 0.012], [tank_d + 0.01, min(Wy, 0.46) + 0.01, 0.024], 0.01)
    mb.add("ceramic", T @ R.T + [c[0], c[1], fz], N @ R.T)
    T, N = lathe([(0.025, 0), (0.025, 0.008), (0.0, 0.01)], 16, center=(-Lx / 2 + tank_d / 2, 0, seat_h + th + 0.024))
    mb.add("chrome", T @ R.T + [c[0], c[1], fz], N @ R.T)


def basin(mb, lo, hi, with_pedestal=True):
    lo, hi = np.asarray(lo, float), np.asarray(hi, float)
    c = (lo + hi) / 2
    ext = hi - lo
    top = hi[2]
    prof = [(0.0, -0.13), (0.12, -0.12), (0.2, -0.07), (0.235, -0.02), (0.24, 0.0), (0.23, 0.012), (0.19, 0.012),
            (0.16, -0.02), (0.1, -0.06), (0.0, -0.075)]
    sx, sy = ext[0] / 0.48, ext[1] / 0.48
    T, N = lathe(prof, 32, sx, sy, center=(c[0], c[1], top - 0.012))
    mb.add("ceramic", T, N)
    if with_pedestal and top - lo[2] > 0.4:
        floor = lo[2] - (top - lo[2]) if False else None
    return


def pedestal(mb, x, y, z0, z1):
    T, N = lathe([(0.11, 0), (0.08, 0.06), (0.065, 0.3), (0.075, 0.6), (0.11, 1.0)], 24)
    T[..., 2] = z0 + T[..., 2] * (z1 - z0)
    T[..., 0] += x
    T[..., 1] += y
    mb.add("ceramic", T, N)
