"""Проверки «Дом-3D Pro». Каждая возвращает score ∈ [0, 1], passed и диагностику."""
from __future__ import annotations

import colorsys
import math

import numpy as np

from ..checks import CheckResult
from . import config as C
from .dsl import FURNITURE, HouseScene, ParseResult2
from .spatial import Occupancy, floating, segment_hits, touching_pairs

CHECKS = {
    # id: (название, способности, вес)
    "H1": ("JSON и DSL без ошибок", ("format",), 1.5),
    "H2": ("≥ 10 000 примитивов", ("detail",), 2.0),
    "H3": ("Компоненты и массивы", ("detail", "format"), 1.0),
    "H4": ("Черепица покрывает крышу", ("detail", "spatial"), 1.5),
    "H5": ("Габариты 13×10, стены", ("spec", "spatial"), 1.5),
    "H6": ("Перекрытия и отметки", ("spec", "spatial"), 1.0),
    "H7": ("Реальные проёмы", ("spatial", "inference"), 2.0),
    "H8": ("Крыша: конёк, свесы", ("spec", "spatial"), 1.0),
    "H9": ("Камин и труба", ("physics", "spec"), 1.0),
    "H10": ("Окна на всех фасадах", ("spec",), 1.0),
    "H11": ("Солнечный фасад", ("inference",), 1.0),
    "H12": ("Комнаты по ТЗ", ("spec", "spatial"), 2.0),
    "H13": ("Комнаты замкнуты", ("spatial",), 1.5),
    "H14": ("Дневной свет 1/8", ("spec", "spatial"), 1.0),
    "H15": ("Проход ко всем комнатам", ("spatial", "inference"), 2.0),
    "H16": ("Маршрут коляски", ("inference", "spatial"), 2.0),
    "H17": ("Разворот 1.5 м", ("inference", "spatial"), 1.5),
    "H18": ("Кровати и подход", ("spec", "inference"), 1.0),
    "H19": ("Кухня", ("spec",), 1.0),
    "H20": ("Санузлы", ("spec", "inference"), 1.0),
    "H21": ("Мебель без пересечений", ("spatial", "physics"), 1.5),
    "H22": ("5 телевизоров и обзор", ("spatial", "spec"), 2.5),
    "H23": ("Лестница: нормы", ("spec", "spatial"), 1.5),
    "H24": ("Лестница: холлы, просвет", ("spatial", "inference"), 1.5),
    "H25": ("Ограждения и балясины", ("physics", "spec"), 1.5),
    "H26": ("Вход с улицы, пандус", ("inference", "spec"), 1.5),
    "H27": ("Балкон", ("spec", "physics"), 1.0),
    "H28": ("Эвакуационные окна", ("spec", "inference"), 1.0),
    "H29": ("Ничего не висит", ("physics",), 2.0),
    "H30": ("Ограда и калитка", ("spec", "detail"), 1.0),
    "H31": ("Цвета", ("spec",), 0.5),
    "H32": ("План = постройка", ("planning",), 2.0),
    "H33": ("Расчёты в плане", ("planning",), 1.0),
}

FFL = {0: C.FFL0, 1: C.FFL1}
WALLISH = ("wall", "glass", "window_frame", "door", "door_frame", "sill")


def _hsv(o):
    return colorsys.rgb_to_hsv(*o.color)


def facade_of(o, tol=0.5):
    cx, cy = o.center[:2]
    d = {"S": abs(cy - C.Y0), "N": abs(cy - C.Y1), "W": abs(cx - C.X0), "E": abs(cx - C.X1)}
    f = min(d, key=d.get)
    if d[f] > tol:
        return None
    if f in "SN" and not (C.X0 - 0.2 <= cx <= C.X1 + 0.2):
        return None
    if f in "WE" and not (C.Y0 - 0.2 <= cy <= C.Y1 + 0.2):
        return None
    return f


def level_of(z):
    if C.FFL0 - 0.2 < z < C.FFL1 - 0.05:
        return 0
    if C.FFL1 - 0.05 <= z < C.EAVE + 0.3:
        return 1
    return None


def in_rect(x, y, r, pad=0.0):
    return r[0] - pad <= x <= r[2] + pad and r[1] - pad <= y <= r[3] + pad


def room_of(rooms, x, y, level):
    for r in rooms:
        if r["level"] == level and any(in_rect(x, y, q) for q in r["rects"]):
            return r
    return None


def rooms_of_type(rooms, typ, level=None):
    return [r for r in rooms if r["type"] == typ and (level is None or r["level"] == level)]


def opening_wh(o, f):
    e = o.aabb_max - o.aabb_min
    return (e[0] if f in "SN" else e[1]), e[2]


def groups_with_tag(sc, tag, rect=None, level=None):
    """{group: (amin, amax)} для групп, содержащих примитивы с тегом."""
    out = {}
    for i in sc.idx(tag):
        o = sc.objects[i]
        g = sc.groups[i]
        mn, mx = out.get(g, (o.aabb_min.copy(), o.aabb_max.copy()))
        out[g] = (np.minimum(mn, o.aabb_min), np.maximum(mx, o.aabb_max))
    res = {}
    for g, (mn, mx) in out.items():
        c = (mn + mx) / 2
        if rect is not None and not in_rect(c[0], c[1], rect):
            continue
        if level is not None and level_of(mn[2] + 0.05) != level:
            continue
        res[g] = (mn, mx)
    return res


def in_any_rect(x, y, rects, pad=0.0):
    return any(in_rect(x, y, q, pad) for q in rects)


# ---------------------------------------------------------------------------
# лестница
# ---------------------------------------------------------------------------

def stair_levels(sc):
    steps = [sc.objects[i] for i in sc.idx("stair_step")]
    steps = [o for o in steps if C.X0 - 0.5 < o.center[0] < C.X1 + 0.5 and C.Y0 - 0.5 < o.center[1] < C.Y1 + 0.5]
    if len(steps) < 4:
        return None
    steps.sort(key=lambda o: o.aabb_max[2])
    lv = []
    for o in steps:
        t = o.aabb_max[2]
        if lv and abs(lv[-1]["top"] - t) < 0.03:
            lv[-1]["objs"].append(o)
        else:
            lv.append({"top": t, "objs": [o]})
    for L in lv:
        mn = np.min([o.aabb_min for o in L["objs"]], 0)
        mx = np.max([o.aabb_max for o in L["objs"]], 0)
        L["min"], L["max"], L["c"] = mn, mx, (mn[:2] + mx[:2]) / 2
    tops = [L["top"] for L in lv]
    risers = [tops[0] - C.FFL0] + list(np.diff(tops))
    last = C.FFL1 - tops[-1]
    if abs(last) >= 0.03:
        risers.append(last)
    gdir = lv[-1]["c"] - lv[0]["c"]
    gdir = gdir / (np.linalg.norm(gdir) + 1e-9)

    def proj(L, d):
        cs = np.array([[x, y] for x in (L["min"][0], L["max"][0]) for y in (L["min"][1], L["max"][1])])
        p = cs @ d
        return p.min(), p.max()
    dirs, treads, widths = [], [], []
    for i in range(len(lv)):
        j = i + 1 if i + 1 < len(lv) else i - 1
        d = (lv[j]["c"] - lv[i]["c"]) * (1 if j > i else -1)
        n = np.linalg.norm(d)
        d = d / n if n > 0.05 else gdir
        # ось марша — ближайшая к d ось координат
        d = np.array([np.sign(d[0]), 0.0]) if abs(d[0]) >= abs(d[1]) else np.array([0.0, np.sign(d[1])])
        dirs.append(d)
        perp = np.array([-d[1], d[0]])
        a, b = proj(lv[i], perp)
        widths.append(b - a)
        if i + 1 < len(lv):
            f0, b0 = proj(lv[i], d)
            f1, b1 = proj(lv[i + 1], d)
            treads.append(max(f1 - f0, b1 - b0))
    return {"levels": lv, "risers": risers, "treads": treads, "widths": widths, "dirs": dirs}


def side_enclosed(sc, pts):
    """Есть ли стена в любой из точек."""
    idx = sc.idx("wall", "door_frame", "floor_slab")
    if not len(idx):
        return False
    P = np.asarray(pts, float)
    for i in idx:
        o = sc.objects[i]
        lo, hi = o.aabb_min - 1e-6, o.aabb_max + 1e-6
        m = np.all((P >= lo) & (P <= hi), 1)
        if m.any() and o.contains(P[m]).any():
            return True
    return False


def guard_ok(sc, span_lo, span_hi, axis_d, edge_coord, side_sign, surf_z, rail_h, gap_max, bal_idx, hr_idx, zmin_need=None):
    """Проверка ограждения вдоль края: поручень с нужной высотой и балясины с просветом ≤ gap_max.
    axis_d: 0/1 — ось вдоль края; edge_coord — координата края по другой оси;
    side_sign: +1/−1 — наружу; surf_z(t) — высота поверхности по координате t вдоль края."""
    other = 1 - axis_d
    t_mid = (span_lo + span_hi) / 2
    # поручень
    rail = False
    for i in hr_idx:
        o = sc.objects[i]
        if o.aabb_max[axis_d] < span_lo - 0.05 or o.aabb_min[axis_d] > span_hi + 0.05:
            continue
        c = (o.aabb_min[other] + o.aabb_max[other]) / 2
        if abs(c - edge_coord) > 0.25:
            continue
        h = o.aabb_max[2] - surf_z(min(max(t_mid, o.aabb_min[axis_d]), o.aabb_max[axis_d]))
        if rail_h[0] - 0.02 <= h <= rail_h[1] + 0.02:
            rail = True
            break
    # балясины (или сплошное ограждение)
    ivs = []
    for i in bal_idx:
        o = sc.objects[i]
        c = (o.aabb_min[other] + o.aabb_max[other]) / 2
        if abs(c - edge_coord) > 0.2:
            continue
        if o.aabb_max[axis_d] < span_lo - 0.15 or o.aabb_min[axis_d] > span_hi + 0.15:
            continue
        zt = surf_z((o.aabb_min[axis_d] + o.aabb_max[axis_d]) / 2)
        if o.aabb_min[2] > zt + 0.11 or o.aabb_max[2] < zt + 0.6:
            continue
        ivs.append((o.aabb_min[axis_d], o.aabb_max[axis_d]))
    ivs.sort()
    gaps = []
    cur = span_lo + 0.02
    for a, b in ivs:
        if b <= cur:
            continue
        gaps.append(max(0.0, a - cur))
        cur = max(cur, b)
    gaps.append(max(0.0, span_hi - 0.02 - cur))
    gmax = max(gaps) if gaps else 9.0
    return rail, gmax


# ---------------------------------------------------------------------------
# главный прогон
# ---------------------------------------------------------------------------

def run_house(pr: ParseResult2):
    """Безопасная обёртка: внутренняя ошибка валидатора не роняет прогон, а помечается в отчёте."""
    out = []
    try:
        _run_house(pr, out)
    except Exception as e:  # noqa: BLE001
        import traceback
        done = {c.id for c in out}
        msg = f"внутренняя ошибка валидатора: {type(e).__name__}: {e}"
        traceback.print_exc()
        for cid, (name, ab, w) in CHECKS.items():
            if cid not in done:
                out.append(CheckResult(cid, name, ab, w, 0.0, False, msg, "house"))
    order = {cid: k for k, cid in enumerate(CHECKS)}
    out.sort(key=lambda c: order.get(c.id, 99))
    return out


def _run_house(pr: ParseResult2, out):

    def add(cid, score, passed, detail):
        name, ab, w = CHECKS[cid]
        out.append(CheckResult(cid, name, ab, w, float(np.clip(score, 0, 1)), bool(passed), detail, "house"))

    sc: HouseScene = pr.scene
    if sc is None or not sc.objects:
        add("H1", 0, False, pr.error or "сцена не получена")
        for cid in CHECKS:
            if cid != "H1":
                add(cid, 0, False, "нет сцены")
        return out

    n = len(sc.objects)
    nerr = len(sc.errors)
    s1 = (1.0 if pr.strict else 0.4) * max(0.0, 1 - nerr / max(20, n * 0.01))
    add("H1", s1, pr.strict and nerr == 0,
        ("JSON разобран" if pr.strict else "JSON разобран только после починки") +
        (f"; ошибок DSL: {nerr}, напр. {sc.errors[0]}" if nerr else "; ошибок DSL нет"))
    add("H2", min(1.0, n / C.MIN_PRIMS), n >= C.MIN_PRIMS, f"примитивов после развёртки: {n}")
    s3 = (min(1, sc.n_components / C.MIN_COMPONENTS) + min(1, sc.n_arrays / 10)) / 2
    add("H3", s3, sc.n_components >= C.MIN_COMPONENTS and sc.n_arrays >= 10,
        f"использовано компонентов: {sc.n_components}, массивов: {sc.n_arrays}")

    rooms = sc.rooms
    roofs = sc.by_tag("roof")
    tiles_idx = sc.idx("roof_tile")
    walls_idx = sc.idx("wall")

    # ---- H4 черепица
    if roofs and len(tiles_idx):
        xs = np.arange(-0.3, 13.31, 0.25)
        ys = np.arange(-0.3, 10.31, 0.25)
        X, Y = np.meshgrid(xs, ys, indexing="ij")
        base = np.full(X.shape, np.nan)
        for o in roofs:
            base = np.fmax(base, o.top_at(X, Y))
        top = np.full(X.shape, np.nan)
        tl = [sc.objects[i] for i in tiles_idx]
        # ускорение: только клетки внутри AABB плитки
        for o in tl:
            ia = np.searchsorted(xs, o.aabb_min[0] - 1e-9)
            ib = np.searchsorted(xs, o.aabb_max[0] + 1e-9)
            ja = np.searchsorted(ys, o.aabb_min[1] - 1e-9)
            jb = np.searchsorted(ys, o.aabb_max[1] + 1e-9)
            if ia >= ib or ja >= jb:
                continue
            t = o.top_at(X[ia:ib, ja:jb], Y[ia:ib, ja:jb])
            top[ia:ib, ja:jb] = np.fmax(top[ia:ib, ja:jb], t)
        mask = np.isfinite(base)
        cov = (np.nan_to_num(top, nan=-9) >= base - 0.02) & (np.nan_to_num(top, nan=-9) <= base + 0.25)
        frac = float(cov[mask].mean()) if mask.any() else 0.0
        nt = len(tiles_idx)
        add("H4", 0.5 * min(1, nt / C.MIN_TILES) + 0.5 * min(1, frac / 0.95), nt >= C.MIN_TILES and frac >= 0.95,
            f"черепиц: {nt}; покрывают {frac:.0%} поверхности крыши (нужно ≥ 95%, плитки лежат на скате)")
    else:
        add("H4", 0, False, "нет крыши (roof) или черепицы (roof_tile)")

    # ---- H5 габариты и стены
    if len(walls_idx):
        W = [sc.objects[i] for i in walls_idx if C.X0 - 1 < sc.objects[i].center[0] < C.X1 + 1 and C.Y0 - 1 < sc.objects[i].center[1] < C.Y1 + 1]
        mn = np.min([o.aabb_min for o in W], 0)
        mx = np.max([o.aabb_max for o in W], 0)
        err = max(abs(mn[0] - C.X0), abs(mx[0] - C.X1), abs(mn[1] - C.Y0), abs(mx[1] - C.Y1))
        pts = []
        for z in [C.FFL0 + 0.4, C.FFL0 + 1.6, C.FFL0 + 2.6, C.FFL1 + 0.4, C.FFL1 + 1.6, C.FFL1 + 2.6]:
            for x in np.arange(0.5, 12.51, 0.5):
                pts += [(x, 0.12, z), (x, C.Y1 - 0.12, z)]
            for y in np.arange(0.5, 9.51, 0.5):
                pts += [(0.12, y, z), (C.X1 - 0.12, y, z)]
        cov = _covered_depth(sc, pts, WALLISH)
        s = (1 if err <= 0.15 else max(0, 1 - err)) * 0.5 + 0.5 * min(1, cov / 0.95)
        add("H5", s, err <= 0.15 and cov >= 0.95,
            f"стены: x {mn[0]:.2f}…{mx[0]:.2f}, y {mn[1]:.2f}…{mx[1]:.2f} (нужно 0…13 × 0…10); периметр закрыт на {cov:.0%}")
    else:
        add("H5", 0, False, "нет стен")

    # ---- H6 перекрытия
    xs, ys = np.arange(0.6, 12.41, 0.3), np.arange(0.6, 9.41, 0.3)
    X, Y = np.meshgrid(xs, ys, indexing="ij")

    def slab_cov(tags, target):
        ok = np.zeros(X.shape, bool)
        for i in sc.idx(*tags):
            o = sc.objects[i]
            if abs(o.aabb_max[2] - target) > 0.13:
                continue
            ok |= np.abs(o.top_at(X, Y) - target) <= 0.13
        return float(ok.mean())
    c0 = slab_cov(("floor_slab", "foundation"), C.FFL0)
    c1 = slab_cov(("floor_slab",), C.FFL1)
    slab1 = [sc.objects[i] for i in sc.idx("floor_slab") if abs(sc.objects[i].aabb_max[2] - C.FFL1) <= 0.13]
    bot = min((o.aabb_min[2] for o in slab1), default=9)
    add("H6", (min(1, c0 / 0.85) + min(1, c1 / 0.7)) / 2, c0 >= 0.85 and c1 >= 0.7,
        f"пол 1 эт. покрывает {c0:.0%}, перекрытие 2 эт. — {c1:.0%}; низ перекрытия {bot:.2f}")

    # ---- окна/двери по фасадам
    glass = [sc.objects[i] for i in sc.idx("glass")]
    gl = []
    for o in glass:
        f = facade_of(o)
        lv = level_of(o.center[2])
        if f is None or lv is None:
            continue
        w, h = opening_wh(o, f)
        if w < 0.25 or h < 0.25:
            continue
        gl.append((o, f, lv, w, h))
    doors = [sc.objects[i] for i in sc.idx("door")]
    ext_doors = [(o, facade_of(o), level_of(o.aabb_min[2] + 0.1)) for o in doors if facade_of(o)]

    # ---- H7 реальные проёмы
    openings = [(o, f) for o, f, lv, w, h in gl] + [(o, f) for o, f, lv in ext_doors]
    if openings:
        good = 0
        bad = []
        for o, f in openings:
            pts = []
            for a in (-0.3, 0.0, 0.3):
                for b in (-0.3, 0.0, 0.3):
                    for depth in (-0.15, 0.05, 0.15, 0.25, 0.38):
                        if f in "SN":
                            x = o.center[0] + a * (o.aabb_max[0] - o.aabb_min[0])
                            y = (C.Y0 + depth) if f == "S" else (C.Y1 - depth)
                        else:
                            y = o.center[1] + a * (o.aabb_max[1] - o.aabb_min[1])
                            x = (C.X0 + depth) if f == "W" else (C.X1 - depth)
                        z = o.center[2] + b * (o.aabb_max[2] - o.aabb_min[2])
                        pts.append((x, y, z))
            if _covered(sc, pts, ("wall",)) == 0:
                good += 1
            else:
                bad.append(o.id)
        s = good / len(openings)
        add("H7", s, s >= 0.98, f"сквозных проёмов {good} из {len(openings)}"
            + (f"; окно/дверь заложены стеной: {', '.join(bad[:5])}" if bad else ""))
    else:
        add("H7", 0, False, "нет окон (glass) и дверей")

    # ---- H8 крыша
    if roofs:
        def H(x, y):
            v = [o.top_at(x, y) for o in roofs]
            v = [float(t) for t in v if np.isfinite(t)]
            return max(v) if v else float("nan")
        hm, h1, h12 = H(6.5, 5.0), H(1.0, 5.0), H(12.0, 5.0)
        hs, hn = H(6.5, 0.3), H(6.5, 9.7)
        a = abs(max(hm, h1, h12) - C.RIDGE) <= 0.2
        b = abs(h1 - h12) <= 0.25 and min(h1, h12) >= C.RIDGE - 0.4
        c = np.isfinite(hs) and np.isfinite(hn) and hs < C.RIDGE - 2 and hn < C.RIDGE - 2
        o = C.MIN_OVERHANG - 0.02
        sp = [(C.X0 - o, y) for y in np.arange(0.5, 9.6, 1)] + [(C.X1 + o, y) for y in np.arange(0.5, 9.6, 1)]
        sp += [(x, C.Y0 - o) for x in np.arange(0.5, 12.6, 1)] + [(x, C.Y1 + o) for x in np.arange(0.5, 12.6, 1)]
        ov = np.mean([np.isfinite(H(x, y)) and H(x, y) >= C.EAVE - 1.0 for x, y in sp])
        parts = [a, b, c, ov >= 0.95]
        add("H8", sum(parts) / 4, all(parts),
            f"конёк {max(hm, h1, h12):.2f} (нужно {C.RIDGE}); вдоль X: {'да' if b else 'нет'}; скаты: {'да' if c else 'нет'}; "
            f"свес ≥ {C.MIN_OVERHANG} м: {ov:.0%}")
    else:
        add("H8", 0, False, "нет roof")

    # ---- H9 камин и труба
    chims = sc.by_tag("chimney")
    fps = sc.by_tag("fireplace")
    living = rooms_of_type(rooms, "living", 0)
    if chims and fps:
        main = max(chims, key=lambda o: o.aabb_max[2] - o.aabb_min[2])
        cx, cy = main.center[:2]
        top = max(o.aabb_max[2] for o in chims)
        fp_top = max(o.aabb_max[2] for o in fps)
        fp_in = bool(living) and any(in_any_rect(o.center[0], o.center[1], living[0]["rects"], 0.2) for o in fps)
        zs = np.arange(fp_top + 0.05, C.RIDGE + 0.4, 0.1)
        cont = _covered(sc, [(cx, cy, z) for z in zs], ("chimney",))
        through = any(np.isfinite(r.top_at(cx, cy)) and r.top_at(cx, cy) < top for r in roofs)
        parts = [top >= C.RIDGE + C.CHIMNEY_ABOVE_RIDGE - 0.05, fp_in, cont >= 0.97, through]
        add("H9", sum(parts) / 4, all(parts),
            f"верх трубы {top:.2f} (нужно ≥ {C.RIDGE + 0.5}); камин в гостиной: {'да' if fp_in else 'нет'}; "
            f"труба непрерывна от камина: {cont:.0%}; выходит сквозь крышу: {'да' if through else 'нет'}")
    else:
        add("H9", 0, False, "нет камина (fireplace) или трубы (chimney)")

    # ---- H10 окна на фасадах
    cnt = {(f, l): 0 for f in "NSEW" for l in (0, 1)}
    for o, f, lv, w, h in gl:
        cnt[(f, lv)] += 1
    miss = [f"{f}{l + 1}" for (f, l), k in cnt.items() if k == 0]
    hok = []
    for o, f, lv, w, h in gl:
        sill = o.aabb_min[2] - FFL[lv]
        topz = o.aabb_max[2] - FFL[lv]
        hok.append(sill >= 0.74 and topz <= 2.56)
    hfrac = float(np.mean(hok)) if hok else 0
    add("H10", (8 - len(miss)) / 8 * 0.6 + 0.4 * hfrac, not miss and hfrac >= 0.9,
        "окон: " + ", ".join(f"{f}{l + 1}={k}" for (f, l), k in cnt.items()) + f"; по высоте в норме {hfrac:.0%}")

    # ---- H11 солнце
    area = {f: sum(w * h for o, ff, lv, w, h in gl if ff == f) for f in "NSEW"}
    ratio = area["S"] / area["N"] if area["N"] > 0 else (9 if area["S"] > 0 else 0)
    liv_s = bool(living) and any(q[1] <= C.Y0 + 0.5 for q in living[0]["rects"])
    s11 = 0.7 * min(1, ratio / 2) + 0.3 * liv_s
    add("H11", s11, ratio >= 1.95 and liv_s,
        f"остекление юг {area['S']:.1f} м² / север {area['N']:.1f} м² = {ratio:.2f} (нужно ≥ 2); гостиная у южного фасада: {'да' if liv_s else 'нет'}")

    # ---- H12 комнаты
    req_ok, notes = [], []
    for typ, lst in C.ROOMS_REQUIRED.items():
        for (lvl, amin) in lst:
            cand = rooms_of_type(rooms, typ, lvl)
            ok = any(r["area"] >= amin - 0.05 for r in cand)
            req_ok.append(ok)
            if not ok:
                notes.append(f"{typ}@{lvl + 1}: " + (f"площадь {max(r['area'] for r in cand):.1f} < {amin}" if cand else "нет"))
    over = 0
    for i, a in enumerate(rooms):
        for b in rooms[i + 1:]:
            if a["level"] != b["level"]:
                continue
            for p in a["rects"]:
                for q in b["rects"]:
                    w = min(p[2], q[2]) - max(p[0], q[0])
                    h = min(p[3], q[3]) - max(p[1], q[1])
                    if w > 0.05 and h > 0.05 and w * h > 0.1:
                        over += 1
    outside = sum(1 for r in rooms for q in r["rects"] if q[0] < C.X0 - 0.05 or q[2] > C.X1 + 0.05 or q[1] < C.Y0 - 0.05 or q[3] > C.Y1 + 0.05)
    s12 = float(np.mean(req_ok)) * (1 if not over else 0.7) * (1 if not outside else 0.8)
    add("H12", s12, all(req_ok) and not over and not outside,
        f"обязательных комнат {sum(req_ok)} из {len(req_ok)}" + (f"; {'; '.join(notes[:4])}" if notes else "") +
        (f"; пересечений комнат: {over}" if over else "") + (f"; вне дома: {outside}" if outside else ""))

    # ---- H13 комнаты замкнуты стенами
    if rooms:
        covs = []
        for r in rooms:
            ffl = FFL.get(r["level"], C.FFL0)
            pts_in, total, hit = [], 0, 0
            for q in r["rects"]:
                x0, y0, x1, y1 = q
                edges = []
                for x in np.arange(x0 + 0.15, x1 - 0.1, 0.25):
                    edges += [(x, y0, 0, -1), (x, y1, 0, 1)]
                for y in np.arange(y0 + 0.15, y1 - 0.1, 0.25):
                    edges += [(x0, y, -1, 0), (x1, y, 1, 0)]
                for (x, y, dx, dy) in edges:
                    # внутренняя граница с другим прямоугольником той же комнаты — не считаем
                    if in_any_rect(x + dx * 0.2, y + dy * 0.2, [t for t in r["rects"] if t is not q]):
                        continue
                    for z in (ffl + 1.2, ffl + 2.35):
                        total += 1
                        p = [(x + dx * d, y + dy * d, z) for d in (0.03, 0.08, 0.15)]
                        if _covered(sc, p, WALLISH + ("stair_step", "chimney"), any_point=True) > 0:
                            hit += 1
            covs.append(hit / total if total else 0)
        m = float(np.mean(covs))
        add("H13", min(1, m / 0.8), min(covs) >= 0.7,
            f"границы комнат закрыты стенами в среднем на {m:.0%}, хуже всего {min(covs):.0%} (нужно ≥ 70% с учётом дверей)")
    else:
        add("H13", 0, False, "нет rooms")

    # ---- H14 дневной свет
    hab = [r for r in rooms if r["type"] in C.HABITABLE]
    if hab:
        ok = []
        det = []
        for r in hab:
            ga = 0.0
            for o, f, lv, w, h in gl:
                if lv != r["level"]:
                    continue
                if any(in_rect(o.center[0], o.center[1], q, 0.5) for q in r["rects"]):
                    ga += w * h
            need = r["area"] / 8
            ok.append(ga >= need - 0.02)
            if ga < need - 0.02:
                det.append(f"{r['id']}: {ga:.1f} < {need:.1f}")
        add("H14", float(np.mean(ok)), all(ok), "свет во всех жилых комнатах" if all(ok) else "мало окон: " + "; ".join(det[:4]))
    else:
        add("H14", 0, False, "нет жилых комнат")

    # ---- маршруты
    ext = (-1.5, 14.5, -2.0, 12.5)
    main_door = None
    for o, f, lv in ext_doors:
        if f == "N" and lv == 0:
            w = o.aabb_max[0] - o.aabb_min[0]
            if main_door is None or w > (main_door.aabb_max[0] - main_door.aabb_min[0]):
                main_door = o
    door_ids = set(int(i) for i in sc.idx("door"))
    stair = stair_levels(sc)

    # ---- H15 проход ко всем комнатам (человек r = 0.3)
    reach = {}
    if main_door is not None:
        occ0 = Occupancy(sc, C.FFL0 + 0.2, C.FFL0 + 1.8, ext, 0.05, exclude_tags=("rug", "door", "paving"), min_height=0.0)
        free0 = occ0.free_with_clearance(0.3)
        start0 = (main_door.center[0], C.Y1 + 0.8)
        R0 = occ0.reachable(free0, start0)
        for r in rooms:
            if r["level"] == 0:
                reach[r["id"]] = any((R0 & occ0.rect_mask(q, 0.05)).any() for q in r["rects"])
        if stair is not None:
            Lst = stair["levels"][-1]
            d = stair["dirs"][-1]
            arr = Lst["c"] + d * (abs((Lst["max"][:2] - Lst["min"][:2]) @ d) / 2 + 0.35)
            occ1 = Occupancy(sc, C.FFL1 + 0.2, C.FFL1 + 1.8, ext, 0.05, exclude_tags=("rug", "door"), min_height=0.0)
            free1 = occ1.free_with_clearance(0.3)
            R1 = occ1.reachable(free1, (float(arr[0]), float(arr[1])))
            for r in rooms:
                if r["level"] == 1:
                    reach[r["id"]] = any((R1 & occ1.rect_mask(q, 0.05)).any() for q in r["rects"])
        else:
            for r in rooms:
                if r["level"] == 1:
                    reach[r["id"]] = False
        okn = sum(reach.values())
        bad = [k for k, v in reach.items() if not v]
        add("H15", okn / max(1, len(reach)), not bad and len(reach) > 0,
            f"достижимо комнат: {okn} из {len(reach)}" + (f"; нет прохода в: {', '.join(bad[:6])}" if bad else ""))
    else:
        add("H15", 0, False, "входная дверь на северном фасаде 1 этажа не найдена")

    # ---- H16–H17 коляска
    if main_door is not None:
        occw = Occupancy(sc, C.FFL0 + 0.15, C.FFL0 + 0.95, ext, 0.05, exclude_tags=("rug", "door", "paving"))
        freew = occw.free_with_clearance(0.44)
        Rw = occw.reachable(freew, (main_door.center[0], C.Y1 + 0.9))
        targets = ["living", "kitchen", "bedroom_mom", "bath_access"]
        res = {}
        for t in targets:
            rr = rooms_of_type(rooms, t, 0)
            res[t] = bool(rr) and any((Rw & occw.rect_mask(q, 0.0)).any() for q in rr[0]["rects"])
        okw = sum(res.values())
        add("H16", okw / 4, okw == 4,
            "коляска (ширина прохода ≥ 0.9 м) проезжает от входа в: " + ", ".join(f"{k} {'да' if v else 'нет'}" for k, v in res.items()))
        freet = occw.free_with_clearance(0.75)
        tc = {}
        for t in ("bedroom_mom", "bath_access"):
            rr = rooms_of_type(rooms, t, 0)
            tc[t] = bool(rr) and any((freet & occw.rect_mask(q, 0.0) & _dilate(Rw, 0.4, occw.res)).any() for q in rr[0]["rects"])
        add("H17", sum(tc.values()) / 2, all(tc.values()),
            "свободный круг Ø1.5 м: " + ", ".join(f"{k} {'есть' if v else 'нет'}" for k, v in tc.items()))
    else:
        add("H16", 0, False, "нет входной двери на севере")
        add("H17", 0, False, "нет входной двери на севере")

    # ---- H18 кровати
    bed_need = {"bedroom_mom": (0.9, 0.9), "bedroom_kids": (0.9, 0.6), "bedroom_master": (1.6, 0.6)}
    br = []
    for t, (wmin, side) in bed_need.items():
        rr = rooms_of_type(rooms, t)
        if not rr:
            br.append((t, False, "нет комнаты"))
            continue
        r = rr[0]
        beds = groups_with_tag(sc, "bed", level=r["level"])
        beds = {g: v for g, v in beds.items() if in_any_rect((v[0][0] + v[1][0]) / 2, (v[0][1] + v[1][1]) / 2, r["rects"])}
        if not beds:
            br.append((t, False, "нет кровати"))
            continue
        g, (mn, mx) = max(beds.items(), key=lambda kv: (kv[1][1] - kv[1][0])[:2].prod())
        ext_xy = mx[:2] - mn[:2]
        width = float(min(ext_xy))
        long_ax = 0 if ext_xy[0] >= ext_xy[1] else 1
        ffl = FFL.get(r["level"], C.FFL0)
        occb = Occupancy(sc, ffl + 0.15, ffl + 0.9, (mn[0] - 1.5, mx[0] + 1.5, mn[1] - 1.5, mx[1] + 1.5), 0.05,
                         exclude_tags=("rug", "door"), exclude_idx=np.nonzero(sc.groups == g)[0])
        okside = False
        for sgn in (-1, 1):
            if long_ax == 0:
                y0 = (mn[1] - side) if sgn < 0 else mx[1]
                rect = (mn[0], y0, mx[0], y0 + side)
            else:
                x0 = (mn[0] - side) if sgn < 0 else mx[0]
                rect = (x0, mn[1], x0 + side, mx[1])
            inside_room = in_any_rect(rect[0] + 0.02, rect[1] + 0.02, r["rects"], 0.02) and in_any_rect(rect[2] - 0.02, rect[3] - 0.02, r["rects"], 0.02)
            m = occb.rect_mask((rect[0] + 0.03, rect[1] + 0.03, rect[2] - 0.03, rect[3] - 0.03))
            if inside_room and m.any() and not (occb.grid & m).any():
                okside = True
        good = width >= wmin - 0.02 and okside
        br.append((t, good, f"ширина {width:.2f} (≥ {wmin}), свободно у длинной стороны {side} м: {'да' if okside else 'нет'}"))
    add("H18", sum(b[1] for b in br) / 3, all(b[1] for b in br), "; ".join(f"{t}: {d}" for t, _, d in br))

    # ---- H19 кухня
    kit = rooms_of_type(rooms, "kitchen", 0)
    if kit:
        rect = kit[0]["rects"]
        cg = {g: v for g, v in groups_with_tag(sc, "counter", level=0).items() if in_any_rect(*(((v[0] + v[1]) / 2)[:2]), rect, 0.1)}
        length = sum(float(max((v[1] - v[0])[:2])) for v in cg.values())
        tops = [float(v[1][2] - C.FFL0) for v in cg.values()]
        hok = bool(tops) and np.mean([0.84 <= t <= 0.93 for t in tops]) >= 0.8
        has = {t: bool([g for g, v in groups_with_tag(sc, t, level=0).items() if in_any_rect(*(((v[0] + v[1]) / 2)[:2]), rect, 0.1)])
               for t in ("stove", "sink", "fridge", "table")}
        chairs = len([g for g, v in groups_with_tag(sc, "chair", level=0).items() if in_any_rect(*(((v[0] + v[1]) / 2)[:2]), rect, 0.1)])
        parts = [length >= 2.98, hok, all(has.values()), chairs >= 4]
        add("H19", sum(parts) / 4, all(parts),
            f"столешницы {length:.1f} м (≥ 3), высота в норме: {'да' if hok else 'нет'}; плита/мойка/холодильник/стол: "
            + "/".join('да' if has[t] else 'нет' for t in ('stove', 'sink', 'fridge', 'table')) + f"; стульев {chairs}")
    else:
        add("H19", 0, False, "нет кухни")

    # ---- H20 санузлы
    bparts, bdet = [], []
    for t, lvl, need in (("bath_access", 0, ("toilet", "sink", "shower")), ("bath", 1, ("toilet", "sink"))):
        rr = rooms_of_type(rooms, t, lvl)
        if not rr:
            bparts.append(False)
            bdet.append(f"{t}: нет")
            continue
        rect = rr[0]["rects"]
        have = {k: [v for g, v in groups_with_tag(sc, k, level=lvl).items() if in_any_rect(*(((v[0] + v[1]) / 2)[:2]), rect, 0.05)] for k in need + ("bathtub", "shower")}
        ok = all(have[k] for k in need)
        if t == "bath":
            ok = ok and bool(have["bathtub"] or have["shower"])
        if t == "bath_access" and have["shower"]:
            low = min(float(v[0][2]) for v in have["shower"])
            tray = [sc.objects[i] for i in sc.idx("shower") if in_any_rect(sc.objects[i].center[0], sc.objects[i].center[1], rect)]
            flat = min((o.aabb_max[2] for o in tray if o.aabb_max[2] - o.aabb_min[2] < 0.2), default=9) - C.FFL0
            ok = ok and flat <= 0.035 and low >= C.FFL0 - 0.05
            bdet.append(f"{t}: {'ок' if ok else 'нет'} (поддон душа +{flat:.2f} м)")
        else:
            bdet.append(f"{t}: {'ок' if ok else 'нет'}")
        bparts.append(ok)
    add("H20", sum(bparts) / 2, all(bparts), "; ".join(bdet))

    # ---- H21 пересечения мебели
    fidx = np.nonzero(np.isin(sc.tags, list(FURNITURE)))[0]
    widx = walls_idx
    if len(fidx):
        sel = np.concatenate([fidx, widx])
        pairs = touching_pairs(sc.amin[sel], sc.amax[sel], eps=-0.005)
        bad_pairs = 0
        bad_g = set()
        isw = np.isin(sc.tags[sel], ["wall"])
        for a, b in pairs:
            i, j = sel[a], sel[b]
            if isw[a] and isw[b]:
                continue
            if sc.groups[i] == sc.groups[j]:
                continue
            lo = np.maximum(sc.amin[i], sc.amin[j])
            hi = np.minimum(sc.amax[i], sc.amax[j])
            vol = float(np.prod(np.clip(hi - lo, 0, None)))
            if vol > 0.002:
                bad_pairs += 1
                bad_g.add(sc.objects[i].id.split("/")[0])
                bad_g.add(sc.objects[j].id.split("/")[0])
        ng = len(set(sc.groups[fidx]))
        s = max(0.0, 1 - len(bad_g) / max(1, ng) * 2)
        add("H21", s, bad_pairs == 0, "мебель не пересекается со стенами и друг с другом" if not bad_pairs else
            f"пересечений: {bad_pairs}; напр. {', '.join(sorted(bad_g)[:6])}")
    else:
        add("H21", 0, False, "нет мебели")

    # ---- H22 телевизоры
    tv_scores, tv_det = [], []
    for tid, (rtype, seat_tag, dmin) in C.TVS.items():
        o = sc.tvs.get(tid)
        if o is None:
            tv_scores.append(0)
            tv_det.append(f"{tid}: нет")
            continue
        fac = str(getattr(o, "facing", "") or "")
        rr = rooms_of_type(rooms, rtype)
        ok = []
        e = o.aabb_max - o.aabb_min
        if fac in ("+x", "-x"):
            w, th, nrm = e[1], e[0], np.array([1.0 if fac == "+x" else -1.0, 0])
        elif fac in ("+y", "-y"):
            w, th, nrm = e[0], e[1], np.array([0, 1.0 if fac == "+y" else -1.0])
        else:
            tv_scores.append(0)
            tv_det.append(f"{tid}: нет поля facing")
            continue
        h = e[2]
        diag = math.hypot(w, h)
        ok.append(abs(w / max(h, 1e-6) - 16 / 9) <= 0.05 and dmin - 0.01 <= diag <= C.TV_DIAG_MAX and th <= 0.12)
        room = rr[0] if rr else None
        lvl = room["level"] if room else 0
        ok.append(room is not None and in_any_rect(o.center[0], o.center[1], room["rects"], 0.15))
        hc = o.center[2] - FFL.get(lvl, 0)
        ok.append(1.0 <= hc <= 1.7)
        back = np.array([o.center[0], o.center[1]]) - nrm * (th / 2 + 0.08)
        ok.append(_covered(sc, [(back[0], back[1], o.center[2])], ("wall",)) > 0)
        front = np.array([o.center[0], o.center[1]]) + nrm * 0.5
        ok.append(room is not None and in_any_rect(front[0], front[1], room["rects"]))
        seat_ok = dist_ok = sight_ok = False
        if room is not None:
            seats = {g: v for g, v in groups_with_tag(sc, seat_tag, level=lvl).items()
                     if in_any_rect(*(((v[0] + v[1]) / 2)[:2]), room["rects"], 0.05)}
            if seats:
                g, (mn, mx) = min(seats.items(), key=lambda kv: np.linalg.norm(((kv[1][0] + kv[1][1]) / 2)[:2] - o.center[:2]))
                spot = np.array([(mn[0] + mx[0]) / 2, (mn[1] + mx[1]) / 2, FFL.get(lvl, 0) + 1.1])
                v = spot[:2] - o.center[:2]
                dist = float(np.linalg.norm(v))
                ang = math.degrees(math.acos(np.clip(v @ nrm / max(dist, 1e-6), -1, 1)))
                seat_ok = ang <= 35
                dist_ok = 1.5 * diag - 0.05 <= dist <= 4.0 * diag + 0.05
                scr = np.array([o.center[0], o.center[1], o.center[2]]) + np.array([*(nrm * (th / 2 + 0.01)), 0])
                excl = list(np.nonzero(sc.groups == g)[0]) + [sc.index[o.id]]
                hits = segment_hits(sc, spot, scr, exclude=excl)
                hits = [i for i in hits if sc.tags[i] not in ("tv_frame",)]
                sight_ok = not hits
        ok += [seat_ok, dist_ok, sight_ok]
        tv_scores.append(sum(ok) / len(ok))
        tags = ["16:9/размер", "комната", "высота", "на стене", "в комнату", "угол", "дистанция", "обзор"]
        bad = [t for t, v in zip(tags, ok) if not v]
        tv_det.append(f"{tid}: {'ок' if not bad else 'не так: ' + ', '.join(bad)}")
    add("H22", float(np.mean(tv_scores)), all(s >= 0.999 for s in tv_scores), "; ".join(tv_det))

    # ---- H23–H25 лестница
    if stair is None:
        for cid in ("H23", "H24", "H25"):
            add(cid, 0, False, "ступеней (stair_step) нет или слишком мало")
    else:
        rs = np.array(stair["risers"])
        tr = np.array(stair["treads"]) if stair["treads"] else np.zeros(1)
        wd = np.array(stair["widths"])
        r_ok = (rs >= 0.12) & (rs <= C.MAX_RISER + 0.005)
        t_ok = tr >= C.MIN_TREAD - 0.005
        w_ok = wd >= C.MIN_STAIR_WIDTH - 0.02
        s23 = (r_ok.mean() + t_ok.mean() + w_ok.mean()) / 3
        add("H23", s23, bool(r_ok.all() and t_ok.mean() >= 0.9 and w_ok.mean() >= 0.9),
            f"подъёмов {len(rs)}, {rs.min():.3f}…{rs.max():.3f} (≤ {C.MAX_RISER}); проступь мин. {tr.min():.2f} (≥ {C.MIN_TREAD}); "
            f"ширина мин. {wd.min():.2f} (≥ {C.MIN_STAIR_WIDTH})")
        lv = stair["levels"]
        halls0 = rooms_of_type(rooms, "hall", 0)
        halls1 = rooms_of_type(rooms, "hall", 1)
        b0 = lv[0]["c"]
        d_last = stair["dirs"][-1]
        arr = lv[-1]["c"] + d_last * (abs((lv[-1]["max"][:2] - lv[-1]["min"][:2]) @ d_last) / 2 + 0.3)
        bot_ok = bool(halls0) and in_any_rect(b0[0], b0[1], halls0[0]["rects"], 0.05)
        top_ok = bool(halls1) and in_any_rect(arr[0], arr[1], halls1[0]["rects"], 0.05)
        slabs = [sc.objects[i] for i in sc.idx("floor_slab")]
        blocked = 0
        for L in lv:
            for o in L["objs"]:
                cx, cy = o.center[:2]
                for s in slabs:
                    if s.aabb_min[2] < L["top"] + C.HEADROOM - 0.02 and s.aabb_max[2] > L["top"] + 0.05 and np.isfinite(s.top_at(cx, cy)):
                        blocked += 1
                        break
        nst = sum(len(L["objs"]) for L in lv)
        hr_ok = blocked == 0
        s24 = (bot_ok + top_ok + (1 - blocked / max(1, nst))) / 3
        add("H24", s24, bot_ok and top_ok and hr_ok,
            f"начало в холле 1 эт.: {'да' if bot_ok else 'нет'}; выход в холл 2 эт.: {'да' if top_ok else 'нет'}; "
            f"просвет ≥ {C.HEADROOM} м над всеми ступенями: {'да' if hr_ok else f'нет ({blocked} ступ.)'}")
        # ограждения открытых сторон
        bal_idx = sc.idx("baluster", "railing")
        hr_idx = sc.idx("handrail", "railing")
        sides_total, sides_ok, worst_gap = 0, 0, 0.0
        for i, L in enumerate(lv):
            d = stair["dirs"][i]
            perp = np.array([-d[1], d[0]])
            ax = 0 if abs(d[0]) > 0.5 else 1  # ось вдоль марша
            oth = 1 - ax
            lo, hi = L["min"][ax], L["max"][ax]
            for sgn in (-1, 1):
                edge = L["max"][oth] if (perp[oth] * sgn) > 0 else L["min"][oth]
                out_dir = 1 if (perp[oth] * sgn) > 0 else -1
                mid = (lo + hi) / 2
                pts = []
                for dd in (0.03, 0.06, 0.1):
                    for dz in (0.3, 0.8):
                        p = [0, 0, L["top"] + dz]
                        p[ax] = mid
                        p[oth] = edge + out_dir * dd
                        pts.append(tuple(p))
                # соседняя ступень/площадка вплотную — край не открыт
                p2 = [0.0, 0.0]
                p2[ax] = mid
                p2[oth] = edge + out_dir * 0.08
                neighbor = any(in_rect(p2[0], p2[1], (M["min"][0], M["min"][1], M["max"][0], M["max"][1])) and M["top"] >= L["top"] - 0.2
                               for M in lv if M is not L)
                if neighbor or side_enclosed(sc, pts):
                    continue
                sides_total += 1
                rail, gmax = guard_ok(sc, lo, hi, ax, edge, out_dir, lambda t, z=L["top"]: z, C.HANDRAIL, C.MAX_BALUSTER_GAP, bal_idx, hr_idx)
                worst_gap = max(worst_gap, gmax)
                if rail and gmax <= C.MAX_BALUSTER_GAP + 0.006:
                    sides_ok += 1
        s25 = sides_ok / sides_total if sides_total else 1.0
        add("H25", s25, sides_ok == sides_total,
            f"открытых сторон ступеней: {sides_total}, с поручнем {C.HANDRAIL[0]}–{C.HANDRAIL[1]} м и балясинами (просвет ≤ 0.10): {sides_ok}; "
            f"макс. просвет {worst_gap:.2f} м")

    # ---- H26 вход и пандус
    ramps = sc.by_tag("ramp")
    lands = sc.by_tag("landing")
    if main_door is None:
        add("H26", 0, False, "входной двери со стороны улицы (север) нет")
    else:
        parts, det = [], []
        dw = main_door.aabb_max[0] - main_door.aabb_min[0]
        parts.append(dw >= 0.88 and abs(main_door.aabb_min[2] - C.FFL0) <= 0.1)
        det.append(f"дверь {dw:.2f} м")
        wedges = [r for r in ramps if r.type == "wedge"]
        if wedges:
            slopes = [r.size[2] / (r.size[0] if r.slope[1] == "x" else r.size[1]) for r in wedges]
            widths = [(r.size[1] if r.slope[1] == "x" else r.size[0]) for r in wedges]
            parts.append(max(slopes) <= C.MAX_RAMP_SLOPE + 0.004)
            parts.append(min(widths) >= C.MIN_RAMP_WIDTH - 0.02)
            rise = max(o.aabb_max[2] for o in wedges + lands) >= C.FFL0 - 0.05 and min(o.aabb_min[2] for o in wedges) <= 0.05
            parts.append(rise)
            det.append(f"уклон 1:{1 / max(slopes):.0f}, ширина {min(widths):.2f}")
            # поручни с двух сторон
            hr_idx = sc.idx("handrail")
            okp, tot = 0, 0
            for r in wedges:
                ax = 0 if r.slope[1] == "x" else 1
                oth = 1 - ax
                for t in np.linspace(r.aabb_min[ax] + 0.3, r.aabb_max[ax] - 0.3, 6):
                    for edge in (r.aabb_min[oth], r.aabb_max[oth]):
                        tot += 1
                        p = [0.0, 0.0]
                        p[ax], p[oth] = t, (r.aabb_min[oth] + r.aabb_max[oth]) / 2
                        zs = float(r.top_at(p[0], p[1]))
                        for i in hr_idx:
                            o = sc.objects[i]
                            if o.aabb_min[ax] - 0.05 <= t <= o.aabb_max[ax] + 0.05 and abs((o.aabb_min[oth] + o.aabb_max[oth]) / 2 - edge) <= 0.25:
                                if C.HANDRAIL[0] - 0.03 <= o.aabb_max[2] - zs <= C.HANDRAIL[1] + 0.03:
                                    okp += 1
                                    break
            parts.append(okp >= tot * 0.9)
            det.append(f"поручни {okp}/{tot}")
        else:
            parts += [False, False, False, False]
            det.append("пандуса (wedge, tag ramp) нет")
        land = [o for o in lands if abs(o.aabb_max[2] - C.FFL0) <= 0.06 and o.aabb_min[1] >= C.Y1 - 0.4
                and o.aabb_min[0] <= main_door.center[0] <= o.aabb_max[0]]
        lok = any((o.aabb_max[0] - o.aabb_min[0]) >= 1.45 and (o.aabb_max[1] - max(o.aabb_min[1], C.Y1)) >= 1.45 for o in land)
        parts.append(lok)
        det.append(f"площадка 1.5×1.5 у двери: {'да' if lok else 'нет'}")
        add("H26", sum(parts) / len(parts), all(parts), "; ".join(det))

    # ---- H27 балкон
    balc = sc.by_tag("balcony")
    if balc:
        b = max(balc, key=lambda o: (o.aabb_max[0] - o.aabb_min[0]) * (o.aabb_max[1] - o.aabb_min[1]))
        depth = C.Y0 - b.aabb_min[1]
        width = b.aabb_max[0] - b.aabb_min[0]
        p = [b.aabb_max[1] <= C.Y0 + 0.4, depth >= C.BALCONY_DEPTH - 0.03, width >= C.BALCONY_WIDTH - 0.03,
             abs(b.aabb_max[2] - C.FFL1) <= 0.12]
        bal_idx = sc.idx("baluster", "railing")
        hr_idx = sc.idx("handrail", "railing")
        bt = b.aabb_max[2]
        sides = [(0, b.aabb_min[0], b.aabb_max[0], b.aabb_min[1], -1),
                 (1, b.aabb_min[1], min(b.aabb_max[1], C.Y0), b.aabb_min[0], -1),
                 (1, b.aabb_min[1], min(b.aabb_max[1], C.Y0), b.aabb_max[0], 1)]
        g_ok = 0
        gm = 0.0
        for ax, lo, hi, edge, od in sides:
            rail, gmax = guard_ok(sc, lo, hi, ax, edge, od, lambda t: bt, C.BALCONY_RAIL, C.MAX_BALUSTER_GAP, bal_idx, hr_idx)
            gm = max(gm, gmax)
            g_ok += rail and gmax <= C.MAX_BALUSTER_GAP + 0.006
        p.append(g_ok == 3)
        master = rooms_of_type(rooms, "bedroom_master", 1)
        bd = [o for o, f, lv in ext_doors if f == "S" and lv == 1 and o.aabb_max[0] > b.aabb_min[0] and o.aabb_min[0] < b.aabb_max[0]
              and master and in_any_rect(o.center[0], o.center[1] + 0.3, master[0]["rects"], 0.35)]
        p.append(bool(bd))
        add("H27", sum(p) / len(p), all(p),
            f"глубина {depth:.2f}, ширина {width:.2f}, верх {b.aabb_max[2]:.2f}; ограждение 1.0–1.15 и просвет ≤ 0.10: {g_ok}/3 "
            f"(макс. просвет {gm:.2f}); дверь из спальни: {'да' if bd else 'нет'}")
    else:
        add("H27", 0, False, "нет балкона")

    # ---- H28 эвакуационные окна
    eg = []
    for t in ("bedroom_kids", "bedroom_master"):
        rr = rooms_of_type(rooms, t, 1)
        ok = False
        if rr:
            for o, f, lv, w, h in gl:
                if lv == 1 and any(in_rect(o.center[0], o.center[1], q, 0.5) for q in rr[0]["rects"]):
                    if w >= 0.6 and h >= 1.0 and o.aabb_min[2] - C.FFL1 <= 1.17:
                        ok = True
        eg.append((t, ok))
    add("H28", sum(v for _, v in eg) / 2, all(v for _, v in eg),
        "окно ≥ 0.6×1.0 м с подоконником ≤ 1.1 м: " + ", ".join(f"{t} {'есть' if v else 'нет'}" for t, v in eg))

    # ---- H29 физика
    fl = floating(sc.objects, sc.amin, sc.amax)
    frac = len(fl) / n
    add("H29", max(0.0, 1 - 20 * frac), not fl,
        "всё опирается на землю или на другие объекты" if not fl else
        f"висят в воздухе {len(fl)} примитивов ({frac:.1%}): {', '.join(o.id for o in fl[:5])}")

    # ---- H30 ограда
    fx0, fx1, fy0, fy1 = C.PLOT
    fence = sc.idx("fence", "gate")
    if len(fence):
        pts = []
        for x in np.arange(fx0 + 0.5, fx1 - 0.4, 0.5):
            pts += [(x, fy0, 0), (x, fy1, 0)]
        for y in np.arange(fy0 + 0.5, fy1 - 0.4, 0.5):
            pts += [(fx0, y, 0), (fx1, y, 0)]
        P = np.array(pts, float)
        hit = np.zeros(len(P), bool)
        for i in fence:
            o = sc.objects[i]
            if o.aabb_max[2] < 0.5 or o.aabb_min[2] > 0.7:
                continue
            m = (P[:, 0] >= o.aabb_min[0] - 0.09) & (P[:, 0] <= o.aabb_max[0] + 0.09) & \
                (P[:, 1] >= o.aabb_min[1] - 0.25) & (P[:, 1] <= o.aabb_max[1] + 0.25)
            m2 = (P[:, 0] >= o.aabb_min[0] - 0.25) & (P[:, 0] <= o.aabb_max[0] + 0.25) & \
                 (P[:, 1] >= o.aabb_min[1] - 0.09) & (P[:, 1] <= o.aabb_max[1] + 0.09)
            hit |= m | m2
        cov = float(hit.mean())
        gates = [sc.objects[i] for i in sc.idx("gate") if sc.objects[i].center[1] > fy1 - 0.5]
        gw = (max(o.aabb_max[0] for o in gates) - min(o.aabb_min[0] for o in gates)) if gates else 0
        add("H30", 0.6 * min(1, cov / 0.85) + 0.4 * (gw >= 0.98), cov >= 0.85 and gw >= 0.98,
            f"ограда закрывает {cov:.0%} периметра участка; калитка на улицу {gw:.2f} м (≥ 1.0)")
    else:
        add("H30", 0, False, "нет ограды (fence)")

    # ---- H31 цвета
    def maj(objs, pred):
        return bool(objs) and np.mean([pred(*_hsv(o)) for o in objs]) >= 0.7
    tl = [sc.objects[i] for i in tiles_idx]
    g = {
        "черепица": maj(tl, lambda h, s, v: (h <= 45 / 360 or h >= 330 / 360) and s >= 0.25 and 0.2 <= v <= 0.85),
        "стены": maj([sc.objects[i] for i in walls_idx], lambda h, s, v: v >= 0.7 and s <= 0.35),
        "стекло": maj([o for o, *_ in gl], lambda h, s, v: 0.45 <= h <= 0.72 and s >= 0.08),
        "двери": maj(doors, lambda h, s, v: (h <= 50 / 360 or h >= 345 / 360) and s >= 0.2 and v <= 0.75),
        "экраны": maj(list(sc.tvs.values()), lambda h, s, v: v <= 0.25),
    }
    add("H31", sum(g.values()) / len(g), all(g.values()), "; ".join(f"{k}: {'ок' if v else 'не то'}" for k, v in g.items()))

    # ---- H32–H33 план
    plan = sc.plan or {}
    if not plan:
        add("H32", 0, False, "нет plan")
        add("H33", 0, False, "нет plan")
    else:
        def num(k):
            try:
                return float(plan.get(k))
            except (TypeError, ValueError):
                return None
        m, notes = [], []
        pt = num("primitives_total")
        m.append(pt is not None and abs(pt - n) <= max(5, 0.02 * n))
        notes.append(f"примитивов {pt}/{n}")
        rt = num("roof_tiles")
        m.append(rt is not None and abs(rt - len(tiles_idx)) <= max(3, 0.02 * len(tiles_idx)))
        notes.append(f"черепиц {rt}/{len(tiles_idx)}")
        if stair:
            nr = len(stair["risers"])
            pr_ = num("risers")
            m.append(pr_ is not None and round(pr_) in (nr, len(stair["levels"])))
            notes.append(f"подъёмов {pr_}/{nr}")
            rh = num("riser_height")
            m.append(rh is not None and abs(rh - float(np.mean(stair["risers"]))) <= 0.006)
            notes.append(f"подъём {rh}/{np.mean(stair['risers']):.3f}")
        else:
            m += [False, False]
        rl = num("ramp_length")
        run = sum((r.size[0] if r.slope[1] == "x" else r.size[1]) for r in ramps if r.type == "wedge")
        m.append(rl is not None and abs(rl - run) <= max(0.3, 0.1 * run))
        notes.append(f"пандус {rl}/{run:.2f}")
        ra = plan.get("room_areas") or {}
        ok_a = 0
        for r in rooms:
            v = ra.get(r["id"]) if isinstance(ra, dict) else None
            try:
                ok_a += abs(float(v) - r["area"]) <= max(0.1, 0.03 * r["area"])
            except (TypeError, ValueError):
                pass
        m.append(bool(rooms) and ok_a >= len(rooms) - 0)
        notes.append(f"площади комнат {ok_a}/{len(rooms)}")
        s = sum(m) / len(m)
        add("H32", s, s >= 0.99, "заявлено/построено: " + "; ".join(notes))
        rn, rh, td = num("risers"), num("riser_height"), num("tread_depth")
        q = [rn is not None and rh is not None and abs(rn * rh - (C.FFL1 - C.FFL0)) <= 0.02,
             rh is not None and rh <= C.MAX_RISER + 1e-6,
             td is not None and td >= C.MIN_TREAD - 1e-6,
             rl is not None and rl >= C.FFL0 / C.MAX_RAMP_SLOPE - 0.05]
        add("H33", sum(q) / 4, all(q),
            f"подъёмы×высота = {(rn or 0) * (rh or 0):.3f} (нужно 3.0); подъём ≤ {C.MAX_RISER}: {'да' if q[1] else 'нет'}; "
            f"проступь ≥ {C.MIN_TREAD}: {'да' if q[2] else 'нет'}; пандус ≥ 5.4 м: {'да' if q[3] else 'нет'}")
    return out


def _covered_depth(sc, pts, tags):
    """Точка у наружной стены считается закрытой, если на любой глубине 0…0.45 м внутрь стены есть стена/проём."""
    P = np.asarray(pts, float)
    variants = []
    for d in np.arange(-0.1, 0.46, 0.015):
        Q = P.copy()
        # сдвиг по нормали к ближайшему фасаду
        dx = np.where(Q[:, 0] < 1, d, np.where(Q[:, 0] > C.X1 - 1, -d, 0.0))
        dy = np.where(Q[:, 1] < 1, d, np.where(Q[:, 1] > C.Y1 - 1, -d, 0.0))
        Q[:, 0] += dx
        Q[:, 1] += dy
        variants.append(Q)
    V = np.concatenate(variants)
    idx = sc.idx(*tags)
    hit = np.zeros(len(V), bool)
    lo, hi = sc.amin[idx] - 1e-6, sc.amax[idx] + 1e-6
    for k, i in enumerate(idx):
        m = ~hit & np.all((V >= lo[k]) & (V <= hi[k]), 1)
        if m.any():
            w = np.nonzero(m)[0]
            hit[w] |= sc.objects[i].contains(V[w])
    H = hit.reshape(len(variants), len(P)).any(0)
    return float(H.mean())


def _dilate(mask, r, res):
    from scipy import ndimage
    k = max(1, int(r / res))
    return ndimage.binary_dilation(mask, iterations=k)


def _covered(sc, pts, tags, any_point=False):
    """Доля точек внутри примитивов с тегами (или 1/0 при any_point)."""
    idx = sc.idx(*tags)
    P = np.asarray(pts, float)
    if not len(P):
        return 0.0
    hit = np.zeros(len(P), bool)
    if len(idx):
        lo = sc.amin[idx] - 1e-6
        hi = sc.amax[idx] + 1e-6
        for k, i in enumerate(idx):
            m = ~hit & np.all((P >= lo[k]) & (P <= hi[k]), 1)
            if m.any():
                w = np.nonzero(m)[0]
                hit[w] |= sc.objects[i].contains(P[w])
    if any_point:
        return 1.0 if hit.any() else 0.0
    return float(hit.mean())


def feedback(checks):
    return "\n".join(f"- [{c.id} {c.name}] {c.detail}" for c in checks if not c.passed)
