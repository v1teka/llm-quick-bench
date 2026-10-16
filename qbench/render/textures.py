"""Процедурные бесшовные текстуры (numpy): трава, штукатурка, дерево, кирпич, камень, асфальт, кора, ткань.

Все текстуры тайлятся (шум с периодическими краями) и возвращаются как uint8 RGB (sRGB).
Текстуры «деталей» (штукатурка, дерево, ткань) — почти белые, их цвет умножается на цвет материала.
"""
from __future__ import annotations

import numpy as np


def _smooth(t):
    return t * t * (3 - 2 * t)


def value_noise(n, freq, rng, aspect=(1, 1)):
    """Периодический value noise n×n с частотой freq ячеек (по осям можно растянуть aspect)."""
    fx, fy = max(1, int(freq * aspect[0])), max(1, int(freq * aspect[1]))
    g = rng.random((fy, fx))
    y = np.linspace(0, fy, n, endpoint=False)
    x = np.linspace(0, fx, n, endpoint=False)
    y0, x0 = np.floor(y).astype(int), np.floor(x).astype(int)
    ty, tx = _smooth(y - y0), _smooth(x - x0)
    y1, x1 = (y0 + 1) % fy, (x0 + 1) % fx
    a = g[y0][:, x0]
    b = g[y0][:, x1]
    c = g[y1][:, x0]
    d = g[y1][:, x1]
    tx, ty = tx[None, :], ty[:, None]
    return (a * (1 - tx) + b * tx) * (1 - ty) + (c * (1 - tx) + d * tx) * ty


def fbm(n, rng, base=4, octaves=5, gain=0.5, aspect=(1, 1)):
    out = np.zeros((n, n))
    amp, tot = 1.0, 0.0
    for k in range(octaves):
        out += amp * value_noise(n, base * 2 ** k, rng, aspect)
        tot += amp
        amp *= gain
    return out / tot


def _u8(rgb):
    return np.clip(rgb * 255 + 0.5, 0, 255).astype(np.uint8)


def _mix(c1, c2, t):
    c1, c2 = np.asarray(c1, float), np.asarray(c2, float)
    return c1 * (1 - t[..., None]) + c2 * t[..., None]


def grass(n=512, seed=1):
    rng = np.random.default_rng(seed)
    low = fbm(n, rng, base=3, octaves=4)
    mid = fbm(n, rng, base=12, octaves=3)
    hi = rng.random((n, n))
    col = _mix((0.20, 0.33, 0.10), (0.38, 0.52, 0.18), np.clip(low * 1.4 - 0.2, 0, 1))
    col = _mix(col, (0.55, 0.55, 0.28), np.clip((mid - 0.62) * 3, 0, 1) * 0.5)  # сухие пятна
    col *= (0.75 + 0.5 * hi[..., None] ** 2)  # травинки
    return _u8(np.clip(col, 0, 1))


def plaster(n=256, seed=2):
    rng = np.random.default_rng(seed)
    v = 0.9 + 0.08 * fbm(n, rng, base=8, octaves=5) + 0.03 * rng.random((n, n))
    return _u8(np.repeat(v[..., None], 3, 2))


def concrete(n=256, seed=3):
    rng = np.random.default_rng(seed)
    v = 0.78 + 0.18 * fbm(n, rng, base=6, octaves=6) + 0.05 * rng.random((n, n))
    return _u8(np.repeat(v[..., None], 3, 2))


def wood_grain(n=256, seed=4):
    """Волокна вдоль оси u (белёсая деталь, цвет даёт материал)."""
    rng = np.random.default_rng(seed)
    w = fbm(n, rng, base=4, octaves=4, aspect=(0.25, 4))
    rings = 0.5 + 0.5 * np.sin((w * 30 + fbm(n, rng, base=2, octaves=2) * 6) * np.pi)
    v = 0.78 + 0.18 * rings + 0.04 * rng.random((n, n))
    return _u8(np.repeat(v[..., None], 3, 2))


def parquet(n=512, seed=5, planks=6, length=1):
    """Доски пола: planks досок поперёк текстуры, цвет дуба."""
    rng = np.random.default_rng(seed)
    g = wood_grain(n, seed).astype(float) / 255
    out = np.zeros((n, n, 3))
    pw = n // planks
    base = np.array([0.70, 0.52, 0.33])
    for i in range(planks):
        y0, y1 = i * pw, (i + 1) * pw
        shift = rng.integers(0, n)
        tint = base * (0.85 + 0.25 * rng.random())
        seg = np.roll(g[y0:y1], shift, axis=1) * tint
        cut = rng.integers(0, n)
        seg[:, max(0, cut - 1):cut + 1] *= 0.55  # стык досок
        seg[-2:] *= 0.5  # шов между досками
        out[y0:y1] = seg
    return _u8(out)


def bricks(n=512, seed=6, rows=8, cols=4, color=(0.55, 0.22, 0.14)):
    rng = np.random.default_rng(seed)
    out = np.zeros((n, n, 3))
    bh, bw = n // rows, n // cols
    mortar = np.array([0.72, 0.70, 0.66])
    noise = fbm(n, rng, base=16, octaves=3)
    for r in range(rows):
        off = (bw // 2) * (r % 2)
        for c in range(cols + 1):
            x0 = c * bw - off
            tint = np.asarray(color) * (0.8 + 0.4 * rng.random())
            ys, xs = slice(r * bh, (r + 1) * bh), np.arange(x0, x0 + bw) % n
            out[ys][:, xs] = tint
    out *= (0.85 + 0.3 * noise[..., None])
    y = np.arange(n) % bh
    out[(y < 3)[:, None].repeat(n, 1)] = mortar
    for r in range(rows):
        off = (bw // 2) * (r % 2)
        for c in range(cols + 1):
            x = (c * bw - off) % n
            out[r * bh:(r + 1) * bh, max(0, x - 1):x + 2] = mortar
    return _u8(np.clip(out, 0, 1))


def stone_paving(n=512, seed=7, cells=5):
    rng = np.random.default_rng(seed)
    # плитка с чуть неровными швами
    u = (np.arange(n) / n * cells)
    U, V = np.meshgrid(u, u)
    jit = fbm(n, rng, base=6, octaves=3) * 0.25
    fu, fv = (U + jit) % 1, (V + jit) % 1
    seam = (np.minimum(fu, 1 - fu) < 0.03) | (np.minimum(fv, 1 - fv) < 0.03)
    idx = (np.floor(U + jit) + 17 * np.floor(V + jit)).astype(int)
    tints = 0.85 + 0.3 * rng.random(int(idx.max()) + 2)
    base = np.array([0.62, 0.59, 0.55])
    col = base * tints[idx][..., None] * (0.85 + 0.25 * fbm(n, rng, base=10, octaves=4)[..., None])
    col[seam] = (0.35, 0.34, 0.32)
    return _u8(np.clip(col, 0, 1))


def asphalt(n=256, seed=8):
    rng = np.random.default_rng(seed)
    v = 0.16 + 0.06 * fbm(n, rng, base=8, octaves=5) + 0.08 * (rng.random((n, n)) > 0.985)
    return _u8(np.repeat(v[..., None], 3, 2) * np.array([1.0, 1.0, 1.03]))


def bark(n=256, seed=9):
    rng = np.random.default_rng(seed)
    w = fbm(n, rng, base=6, octaves=5, aspect=(6, 0.5))
    v = 0.55 + 0.45 * w
    col = np.array([0.30, 0.22, 0.15]) * v[..., None]
    return _u8(np.clip(col, 0, 1))


def fabric(n=128, seed=10):
    rng = np.random.default_rng(seed)
    x = np.arange(n)
    weave = 0.5 + 0.5 * np.sin(x[None, :] * np.pi / 2) * np.sin(x[:, None] * np.pi / 2)
    v = 0.86 + 0.08 * weave + 0.06 * fbm(n, rng, base=8, octaves=3)
    return _u8(np.repeat(v[..., None], 3, 2))


def roof_tile(n=256, seed=11):
    rng = np.random.default_rng(seed)
    v = 0.82 + 0.16 * fbm(n, rng, base=6, octaves=5) + 0.04 * rng.random((n, n))
    return _u8(np.repeat(v[..., None], 3, 2))


ALL = {"grass": grass, "plaster": plaster, "concrete": concrete, "wood": wood_grain, "parquet": parquet,
       "bricks": bricks, "stone": stone_paving, "asphalt": asphalt, "bark": bark, "fabric": fabric, "tile": roof_tile}
