"""Автоматические проверки сцены «Дом-3D».

Каждая проверка возвращает score ∈ [0, 1], флаг passed и человекочитаемую
диагностику (она же уходит модели в фазе самокоррекции).
"""
from __future__ import annotations

import colorsys
from dataclasses import dataclass, asdict

import numpy as np

from . import config as C
from .scene import Obj, ParseResult, Scene


@dataclass
class CheckResult:
    id: str
    name: str
    abilities: tuple
    weight: float
    score: float
    passed: bool
    detail: str
    phase: str = "base"

    def to_dict(self):
        d = asdict(self)
        d["abilities"] = list(self.abilities)
        return d


# Описание проверок: id -> (название, способности, вес)
BASE = {
    "F1": ("Строгий JSON", ("format",), 1.0),
    "F2": ("Схема объектов", ("format",), 1.0),
    "F3": ("Бюджет объектов", ("format",), 0.5),
    "S1": ("Габариты 10×8", ("spec", "spatial"), 2.0),
    "S2": ("Отметки этажей", ("spec", "spatial"), 2.0),
    "S3": ("Замкнутые стены", ("spatial", "spec"), 2.0),
    "S4": ("Двускатная крыша", ("spec", "spatial"), 2.0),
    "S5": ("Свесы крыши", ("spatial",), 1.0),
    "S6": ("Цвета", ("spec",), 1.0),
    "S7": ("Дымоход", ("spec", "physics"), 1.0),
    "S8": ("Окна на всех фасадах", ("spec", "spatial"), 2.0),
    "S9": ("Окна в стенах", ("spatial",), 1.0),
    "S10": ("Высоты окон", ("spec", "math"), 1.0),
    "S11": ("Вход со стороны улицы", ("inference", "spatial"), 2.0),
    "S12": ("Пандус ≤ 1:12", ("inference", "math"), 2.0),
    "S13": ("Пандус ведёт к двери", ("spatial", "inference"), 1.0),
    "S14": ("Солнечный фасад", ("inference",), 1.5),
    "S15": ("Геометрия лестницы", ("math", "spatial"), 2.0),
    "S16": ("Лестница соединяет этажи", ("spatial",), 1.5),
    "S17": ("Просвет над лестницей", ("spatial", "inference"), 2.0),
    "S18": ("Балкон", ("spec", "spatial"), 1.5),
    "S19": ("Ограждение балкона", ("spec", "physics"), 1.5),
    "S20": ("Выход на балкон", ("inference", "spatial"), 1.0),
    "S21": ("Ничего не висит в воздухе", ("physics",), 2.0),
    "S22": ("В границах участка", ("spatial",), 0.5),
    "P1": ("План = постройка", ("planning",), 2.0),
    "P2": ("Расчёты в плане", ("math", "planning"), 1.5),
}
CHANGE = {
    "G1": ("Стены гаража", ("change", "spatial"), 1.5),
    "G2": ("Плоская крыша гаража", ("change", "spec"), 1.0),
    "G3": ("Ворота к улице", ("change", "inference"), 1.5),
    "G4": ("Дверь гараж→дом", ("change", "spatial"), 1.0),
    "G5": ("Окна не в гараж", ("change", "inference"), 1.5),
    "G6": ("Пандус и подъезд к воротам", ("change", "spatial", "inference"), 1.5),
    "G7": ("Без регрессий", ("change",), 2.0),
    "G8": ("Сохранены объекты", ("change", "planning"), 1.0),
}


# ---------------------------------------------------------------------------
# геометрические помощники
# ---------------------------------------------------------------------------

def in_garage_zone(o: Obj) -> bool:
    cx, cy = o.center[:2]
    return cx > C.X1 + 0.05 and C.GY0 - 0.5 <= cy <= C.GY1 + 0.5


def facade(o: Obj, tol=0.45):
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


def level_of(o: Obj):
    cz = o.center[2]
    if C.FFL0 - 0.2 < cz < C.FFL1:
        return 0
    if C.FFL1 <= cz < C.EAVE + 0.3:
        return 1
    return None


def opening_wh(o: Obj, f):
    ext = o.aabb_max - o.aabb_min
    return (ext[0] if f in "SN" else ext[1]), ext[2]


def covered(points, objs):
    pts = np.asarray(points, float)
    res = np.zeros(len(pts), bool)
    for o in objs:
        m = ~res
        if not m.any():
            break
        lo, hi = o.aabb_min - 1e-6, o.aabb_max + 1e-6
        cand = m & np.all((pts >= lo) & (pts <= hi), 1)
        if cand.any():
            idx = np.where(cand)[0]
            res[idx] |= o.contains(pts[idx])
    return res


def height_field(objs, xs, ys):
    X, Y = np.meshgrid(xs, ys, indexing="ij")
    H = np.full(X.shape, np.nan)
    for o in objs:
        t = o.top_at(X, Y)
        H = np.fmax(H, t)
    return H


def H_at(objs, x, y):
    return float(height_field(objs, [x], [y])[0, 0])


def rect_dist(a: Obj, b: Obj):
    dx = max(0.0, max(a.aabb_min[0] - b.aabb_max[0], b.aabb_min[0] - a.aabb_max[0]))
    dy = max(0.0, max(a.aabb_min[1] - b.aabb_max[1], b.aabb_min[1] - a.aabb_max[1]))
    return float(np.hypot(dx, dy))


def hsv(o: Obj):
    return colorsys.rgb_to_hsv(*o.color)


def _frac(n, d):
    return n / d if d else 0.0


# ---------------------------------------------------------------------------
# анализ лестницы (общий для S15, S16, S17, P1)
# ---------------------------------------------------------------------------

def stair_analysis(steps):
    if not steps:
        return None
    tops = sorted(steps, key=lambda o: o.aabb_max[2])
    levels = []  # [(top, [objs])]
    for o in tops:
        t = o.aabb_max[2]
        if levels and abs(levels[-1][0] - t) < 0.03:
            levels[-1][1].append(o)
        else:
            levels.append([t, [o]])
    lv = []
    for t, objs in levels:
        mn = np.min([o.aabb_min for o in objs], 0)
        mx = np.max([o.aabb_max for o in objs], 0)
        lv.append({"top": t, "min": mn, "max": mx, "c": (mn[:2] + mx[:2]) / 2})
    top_vals = [l["top"] for l in lv]
    risers = [top_vals[0] - C.FFL0] + list(np.diff(top_vals))
    last = C.FFL1 - top_vals[-1]
    if abs(last) >= 0.03:
        risers.append(last)
    # проступи
    gdir = lv[-1]["c"] - lv[0]["c"]
    gdir = gdir / (np.linalg.norm(gdir) + 1e-9)
    treads, widths = [], []

    def proj(l, d):
        corners = np.array([[x, y] for x in (l["min"][0], l["max"][0]) for y in (l["min"][1], l["max"][1])])
        p = corners @ d
        return p.min(), p.max()
    for i in range(len(lv) - 1):
        d = lv[i + 1]["c"] - lv[i]["c"]
        n = np.linalg.norm(d)
        d = d / n if n > 0.05 else gdir
        f0, b0 = proj(lv[i], d)
        f1, b1 = proj(lv[i + 1], d)
        treads.append(max(f1 - f0, b1 - b0))
        perp = np.array([-d[1], d[0]])
        p0 = proj(lv[i], perp)
        widths.append(p0[1] - p0[0])
    if len(lv) >= 2:
        d = lv[-1]["c"] - lv[-2]["c"]
        n = np.linalg.norm(d)
        d = d / n if n > 0.05 else gdir
        perp = np.array([-d[1], d[0]])
        p = proj(lv[-1], perp)
        widths.append(p[1] - p[0])
    return {"levels": lv, "risers": risers, "treads": treads, "widths": widths}


def main_door(doors):
    lvl0 = [d for d in doors if level_of(d) == 0 and facade(d)]
    if not lvl0:
        return None
    north = [d for d in lvl0 if facade(d) == "N"]
    return (north or lvl0)[0]


FACADE_ALIASES = {
    "n": "N", "north": "N", "с": "N", "север": "N", "северный": "N",
    "s": "S", "south": "S", "ю": "S", "юг": "S", "южный": "S",
    "e": "E", "east": "E", "в": "E", "восток": "E", "восточный": "E",
    "w": "W", "west": "W", "з": "W", "запад": "W", "западный": "W",
}


def _num(v):
    try:
        return float(v)
    except (TypeError, ValueError):
        return None


# ---------------------------------------------------------------------------
# базовые проверки
# ---------------------------------------------------------------------------

def run_base(pr: ParseResult, phase="base"):
    out = []

    def add(cid, score, passed, detail):
        name, ab, w = BASE[cid]
        out.append(CheckResult(cid, name, ab, w, float(np.clip(score, 0, 1)), bool(passed), detail, phase))

    sc = pr.scene
    # ---- формат
    if sc is None:
        add("F1", 0, False, pr.error or "сцена не получена")
        for cid in BASE:
            if cid != "F1":
                add(cid, 0, False, "нет сцены")
        return out
    if pr.strict:
        s = 1.0 if pr.outside_chars <= 300 else 0.7
        add("F1", s, pr.outside_chars <= 300,
            "JSON разобран" + ("" if pr.outside_chars <= 300 else f", но вне JSON {pr.outside_chars} симв. текста"))
    else:
        add("F1", 0.3, False, "JSON разобран только после починки (комментарии/висячие запятые)")
    nerr = len(sc.errors)
    s = 1 - _frac(nerr, max(sc.total, 1))
    add("F2", s, s >= 0.95, f"ошибок в объектах: {nerr} из {sc.total}" + (f"; напр. {sc.errors[0]}" if nerr else ""))
    add("F3", 1.0 if sc.total <= C.MAX_OBJECTS else 0.0, sc.total <= C.MAX_OBJECTS, f"объектов: {sc.total}")

    allobj = sc.objects
    objs = [o for o in allobj if not in_garage_zone(o)] if phase == "change" else allobj
    T = lambda *tags: [o for o in objs if o.tag in tags]  # noqa: E731
    walls, windows, doors = T("wall"), T("window"), T("door")
    roofs, steps = T("roof"), T("stair_step")
    slabs = T("floor_slab")

    # ---- S1 габариты
    if walls:
        mn = np.min([o.aabb_min for o in walls], 0)
        mx = np.max([o.aabb_max for o in walls], 0)
        err = max(abs(mn[0] - C.X0), abs(mx[0] - C.X1), abs(mn[1] - C.Y0), abs(mx[1] - C.Y1))
        add("S1", 1 if err <= C.TOL else max(0, 1 - (err - C.TOL)), err <= C.TOL,
            f"наружные грани стен: x {mn[0]:.2f}…{mx[0]:.2f}, y {mn[1]:.2f}…{mx[1]:.2f} (нужно 0…10 × 0…8)")
    else:
        add("S1", 0, False, "нет объектов wall")

    # ---- S2 отметки этажей
    xs, ys = np.arange(0.5, 9.51, 0.25), np.arange(0.5, 7.51, 0.25)

    def cov(cands, target):
        if not cands:
            return 0.0
        X, Y = np.meshgrid(xs, ys, indexing="ij")
        ok = np.zeros(X.shape, bool)
        for o in cands:
            t = o.top_at(X, Y)
            ok |= np.abs(t - target) <= 0.12
        return float(ok.mean())
    c0 = cov(T("floor_slab", "foundation"), C.FFL0)
    c1 = cov(slabs, C.FFL1)
    add("S2", (min(c0 / 0.85, 1) + min(c1 / 0.7, 1)) / 2, c0 >= 0.85 and c1 >= 0.7,
        f"пол 1 эт. (верх {C.FFL0}) покрывает {c0:.0%} площади; пол 2 эт. (верх {C.FFL1}) — {c1:.0%}")

    # ---- S3 замкнутые стены
    pts = []
    for z in [C.FFL0 + 0.3, C.FFL0 + 1.5, C.FFL0 + 2.7, C.FFL1 + 0.3, C.FFL1 + 1.5, C.FFL1 + 2.7]:
        for x in np.arange(0.5, 9.51, 0.5):
            pts += [(x, C.Y0 + 0.12, z), (x, C.Y1 - 0.12, z)]
        for y in np.arange(0.5, 7.51, 0.5):
            pts += [(C.X0 + 0.12, y, z), (C.X1 - 0.12, y, z)]
    cv = covered(pts, T("wall", "window", "door", "garage_door")).mean()
    add("S3", cv, cv >= 0.92, f"периметр закрыт стенами/проёмами на {cv:.0%}")

    # ---- S4 крыша
    if roofs:
        gx, gy = np.arange(0.25, 9.76, 0.25), np.arange(0.25, 7.76, 0.25)
        H = height_field(roofs, gx, gy)
        hmax = float(np.nanmax(H)) if np.isfinite(H).any() else float("nan")
        h1, h9, hm = H_at(roofs, 1, 4), H_at(roofs, 9, 4), H_at(roofs, 5, 4)
        hs, hn = H_at(roofs, 5, 0.3), H_at(roofs, 5, 7.7)
        a = abs(hmax - C.RIDGE) <= 0.3
        b = all(np.isfinite([h1, h9, hm])) and abs(h1 - h9) <= 0.3 and min(h1, h9) >= C.RIDGE - 0.5
        c = np.isfinite(hs) and np.isfinite(hn) and hs <= C.RIDGE - 1.5 and hn <= C.RIDGE - 1.5
        d = bool(np.all(np.nan_to_num(H, nan=-1) >= C.EAVE - 0.3))
        parts = [a, b, c, d]
        add("S4", sum(parts) / 4, all(parts),
            f"макс. высота {hmax:.2f} (нужно {C.RIDGE}); конёк вдоль X: {'да' if b else 'нет'}; "
            f"скаты на юг/север: {'да' if c else 'нет'}; крыша над всем домом: {'да' if d else 'нет'}")
    else:
        add("S4", 0, False, "нет объектов roof")

    # ---- S5 свесы
    if roofs:
        o = C.MIN_OVERHANG - 0.02
        sp = [(C.X0 - o, y) for y in np.arange(0.5, 7.6, 0.5)] + [(C.X1 + o, y) for y in np.arange(0.5, 7.6, 0.5)]
        sp += [(x, C.Y0 - o) for x in np.arange(0.5, 9.6, 0.5)] + [(x, C.Y1 + o) for x in np.arange(0.5, 9.6, 0.5)]
        hv = np.array([H_at(roofs, x, y) for x, y in sp])
        ok = np.nan_to_num(hv, nan=-1) >= C.EAVE - 1.0
        add("S5", ok.mean(), ok.mean() >= 0.95, f"свес ≥ {C.MIN_OVERHANG} м есть в {ok.mean():.0%} контрольных точек")
    else:
        add("S5", 0, False, "нет крыши")

    # ---- S6 цвета
    def maj(group, pred):
        return bool(group) and np.mean([pred(*hsv(o)) for o in group]) >= 0.7
    red = lambda h, s, v: (h <= 45 / 360 or h >= 330 / 360) and s >= 0.25 and 0.2 <= v <= 0.85  # noqa: E731
    light = lambda h, s, v: v >= 0.7 and s <= 0.4  # noqa: E731
    blue = lambda h, s, v: 0.45 <= h <= 0.72 and s >= 0.08  # noqa: E731
    brown = lambda h, s, v: (h <= 50 / 360 or h >= 345 / 360) and s >= 0.2 and v <= 0.75  # noqa: E731
    g = {"крыша": maj(roofs, red), "стены": maj(walls, light), "стекло": maj(windows, blue), "двери": maj(doors, brown)}
    sc6 = sum(g.values()) / 4
    add("S6", sc6, all(g.values()), "; ".join(f"{k}: {'ок' if v else 'не то'}" for k, v in g.items()))

    # ---- S7 дымоход
    chims = T("chimney")
    best, bd = 0, "нет объектов chimney"
    for ch in chims:
        top_ok = ch.aabb_max[2] >= C.RIDGE + C.CHIMNEY_ABOVE_RIDGE - 0.05
        inside = C.X0 <= ch.center[0] <= C.X1 and C.Y0 <= ch.center[1] <= C.Y1
        through = any(np.nan_to_num(r.top_at(ch.center[0], ch.center[1]), nan=-1) >= ch.aabb_min[2] for r in roofs)
        s = sum(map(float, (top_ok, inside, through))) / 3
        if s >= best:
            best = s
            bd = (f"верх трубы {ch.aabb_max[2]:.2f} (нужно ≥ {C.RIDGE + C.CHIMNEY_ABOVE_RIDGE}); "
                  f"над домом: {'да' if inside else 'нет'}; проходит сквозь кровлю: {'да' if through else 'нет'}")
    add("S7", best, best == 1, bd)

    # ---- окна и двери по фасадам
    wf = [(o, facade(o), level_of(o)) for o in windows]
    counts = {(f, l): 0 for f in "NSEW" for l in (0, 1)}
    for o, f, l in wf:
        if f and l is not None:
            counts[(f, l)] += 1
    missing = [f"{f}{l + 1}" for (f, l), n in counts.items() if n == 0]
    add("S8", (8 - len(missing)) / 8, not missing,
        "окон по фасадам: " + ", ".join(f"{f}{l + 1}={n}" for (f, l), n in counts.items())
        + (f"; нет окон: {', '.join(missing)}" if missing else ""))

    if windows:
        okw = [f is not None and any(_aabb_overlap(o, w) for w in walls) for o, f, l in wf]
        add("S9", np.mean(okw), np.mean(okw) >= 0.9, f"в наружных стенах {sum(okw)} из {len(okw)} окон")
        okh, bad = [], []
        for o, f, l in wf:
            if f is None or l is None:
                continue
            ffl = C.FFL0 if l == 0 else C.FFL1
            sill, top = o.aabb_min[2] - ffl, o.aabb_max[2] - ffl
            good = sill >= 0.75 and top <= 2.55
            okh.append(good)
            if not good:
                bad.append(f"{o.id}: подоконник {sill:.2f}, верх {top:.2f}")
        m = np.mean(okh) if okh else 0
        add("S10", m, m >= 0.9, f"по высоте в норме {sum(okh)} из {len(okh)}" + (f"; напр. {bad[0]}" if bad else ""))
    else:
        add("S9", 0, False, "нет окон")
        add("S10", 0, False, "нет окон")

    # ---- S11 вход со стороны улицы
    dinfo = [(d, facade(d), level_of(d)) for d in doors]
    s11, d11 = 0.0, "входная дверь на северном (уличном) фасаде не найдена"
    found = [f"{d.id}:{f or '?'}{'' if l is None else l + 1}" for d, f, l in dinfo]
    for d, f, l in dinfo:
        if f == "N" and l == 0:
            w, h = opening_wh(d, f)
            good = w >= 0.88 and 1.95 <= h <= 2.35 and abs(d.aabb_min[2] - C.FFL0) <= 0.1
            if good:
                s11, d11 = 1.0, f"дверь {d.id} на северном фасаде, {w:.2f}×{h:.2f}"
                break
            if s11 < 0.5:
                s11, d11 = 0.5, f"дверь на севере есть, но размеры/порог не те: {w:.2f}×{h:.2f}, низ {d.aabb_min[2]:.2f}"
    if s11 == 0 and found:
        d11 += f"; найдены двери: {', '.join(found[:6])}"
    add("S11", s11, s11 == 1, d11)

    # ---- S12 пандус
    ramps = T("ramp")
    landings = T("landing")
    entry = main_door(doors)
    ramp_run, slopes = 0.0, []
    for r in ramps:
        if r.type == "wedge":
            run = r.size[0] if r.slope[1] == "x" else r.size[1]
            slopes.append(r.size[2] / run)
            ramp_run += run
        else:
            slopes.append(float("inf") if r.size[2] > 0.25 else 0.0)
    if not ramps:
        add("S12", 0, False, "пандуса (tag ramp) нет — вход недоступен для коляски")
    else:
        smax = max(slopes)
        tops = [o.aabb_max[2] for o in ramps + landings]
        rise_ok = min(r.aabb_min[2] for r in ramps) <= 0.05 and max(tops) >= C.FFL0 - 0.05
        slope_ok = smax <= C.MAX_RAMP_SLOPE + 0.004
        add("S12", 0.6 * slope_ok + 0.4 * rise_ok, slope_ok and rise_ok,
            f"макс. уклон {smax:.3f} (≈1:{(1 / smax) if smax > 0 else 0:.0f}), нужно ≤ 1:12; "
            f"от земли до пола 1 эт.: {'да' if rise_ok else 'нет'}")

    # ---- S13 пандус к двери
    if entry is None or not (ramps or landings):
        add("S13", 0, False, "нет входной двери или пандуса")
    else:
        near = [o for o in ramps + landings if abs(o.aabb_max[2] - C.FFL0) <= 0.08 and rect_dist(o, entry) <= 0.6]
        add("S13", 1.0 if near else 0.0, bool(near),
            "верх пандуса/площадка примыкает к входной двери" if near else
            "верхняя точка пандуса/площадки не примыкает к входной двери на уровне пола")

    # ---- S14 солнечный фасад
    area = {"N": 0.0, "S": 0.0, "E": 0.0, "W": 0.0}
    for o, f, l in wf:
        if f:
            w, h = opening_wh(o, f)
            area[f] += w * h
    ratio = area["S"] / area["N"] if area["N"] > 0 else (float("inf") if area["S"] > 0 else 0)
    add("S14", min(1.0, ratio / 2) if np.isfinite(ratio) else 1.0, ratio >= 1.95,
        f"остекление: юг {area['S']:.1f} м², север {area['N']:.1f} м² (юг/север = {ratio:.2f}, нужно ≥ 2)")

    # ---- лестница
    st = stair_analysis(steps)
    if st is None or len(st["levels"]) < 4:
        for cid in ("S15", "S16"):
            add(cid, 0, False, "ступеней (tag stair_step) нет или слишком мало")
    else:
        rs = np.array(st["risers"])
        r_ok = (rs >= 0.10) & (rs <= C.MAX_RISER + 0.01)
        tr = np.array(st["treads"]) if st["treads"] else np.array([0.0])
        wd = np.array(st["widths"])
        t_ok, w_ok = (tr >= C.MIN_TREAD - 0.01), (wd >= C.MIN_STAIR_WIDTH - 0.02)
        sc15 = (r_ok.mean() + t_ok.mean() + w_ok.mean()) / 3
        add("S15", sc15, r_ok.all() and t_ok.mean() >= 0.85 and w_ok.mean() >= 0.85,
            f"подъёмов {len(rs)}, высота {rs.min():.3f}…{rs.max():.3f} (≤ {C.MAX_RISER}); "
            f"проступь мин. {tr.min():.2f} (≥ {C.MIN_TREAD}); ширина мин. {wd.min():.2f} (≥ {C.MIN_STAIR_WIDTH})")
        lv = st["levels"]
        first = lv[0]["top"] - C.FFL0
        last = C.FFL1 - lv[-1]["top"]
        a = 0.10 <= first <= 0.20
        b = (0.10 <= last <= 0.20) or abs(last) < 0.03
        c = all(C.X0 + 0.1 <= o.center[0] <= C.X1 - 0.1 and C.Y0 + 0.1 <= o.center[1] <= C.Y1 - 0.1 for o in steps)
        topl = lv[-1]
        dummy = Obj("_top", "box", "other", [*topl["c"], topl["top"]], [*(topl["max"][:2] - topl["min"][:2]), 0.01])
        d = any(abs(s.aabb_max[2] - C.FFL1) <= 0.12 and rect_dist(s, dummy) <= 0.6 for s in slabs)
        add("S16", sum(map(float, (a, b, c, d))) / 4, bool(a and b and c and d),
            f"первая ступень +{first:.2f} от пола 1 эт.: {'ок' if a else 'нет'}; до пола 2 эт. {last:.2f}: "
            f"{'ок' if b else 'нет'}; внутри дома: {'да' if c else 'нет'}; выход на перекрытие 2 эт.: {'да' if d else 'нет'}")

    # ---- S17 просвет
    if not steps:
        add("S17", 0, False, "нет лестницы")
    else:
        blocked = []
        for s in steps:
            top = s.aabb_max[2]
            for sl in slabs:
                if sl.aabb_min[2] < top + C.HEADROOM - 0.02 and sl.aabb_max[2] > top + 0.05:
                    if np.isfinite(sl.top_at(s.center[0], s.center[1])):
                        blocked.append(s.id)
                        break
        ok = 1 - len(blocked) / len(steps)
        add("S17", ok, ok >= 0.95,
            "над всеми ступенями просвет ≥ 2 м" if not blocked else
            f"перекрытие нависает ниже 2 м над {len(blocked)} из {len(steps)} ступеней (нужен проём в перекрытии)")

    # ---- S18–S20 балкон
    balcs = T("balcony")
    best_b, s18, d18 = None, 0.0, "плиты балкона (tag balcony) нет"
    for b in balcs:
        depth = C.Y0 - b.aabb_min[1]
        width = b.aabb_max[0] - b.aabb_min[0]
        parts = [b.aabb_max[1] <= C.Y0 + 0.4, depth >= 1.2 - 0.05, width >= 3.0 - 0.05, abs(b.aabb_max[2] - C.FFL1) <= 0.12]
        s = sum(parts) / 4
        if s > s18 or best_b is None:
            best_b, s18 = b, s
            d18 = f"балкон {b.id}: с юга {'да' if parts[0] else 'нет'}, глубина {depth:.2f}, ширина {width:.2f}, верх {b.aabb_max[2]:.2f}"
    add("S18", s18, s18 == 1, d18)

    rails = T("railing")
    if best_b is None:
        add("S19", 0, False, "нет балкона")
        add("S20", 0, False, "нет балкона")
    else:
        b = best_b
        bt = b.aabb_max[2]
        good = [r for r in rails if 0.9 <= r.aabb_max[2] - bt <= 1.2]
        sides = {
            "фронт": ("x", b.aabb_min[0], b.aabb_max[0], lambda r: abs(r.aabb_min[1] - b.aabb_min[1]) <= 0.25),
            "запад": ("y", b.aabb_min[1], min(b.aabb_max[1], C.Y0), lambda r: abs(r.aabb_min[0] - b.aabb_min[0]) <= 0.25),
            "восток": ("y", b.aabb_min[1], min(b.aabb_max[1], C.Y0), lambda r: abs(r.aabb_max[0] - b.aabb_max[0]) <= 0.25),
        }
        res = {}
        for k, (ax, lo, hi, near) in sides.items():
            i = 0 if ax == "x" else 1
            segs = [(r.aabb_min[i], r.aabb_max[i]) for r in good if near(r)]
            res[k] = _coverage(segs, lo, hi) >= 0.8
        add("S19", sum(res.values()) / 3, all(res.values()),
            "ограждение 0.95–1.15 м: " + ", ".join(f"{k} {'ок' if v else 'нет'}" for k, v in res.items()))
        bd = [d for d, f, l in dinfo if f == "S" and l == 1 and d.aabb_max[0] > b.aabb_min[0] and d.aabb_min[0] < b.aabb_max[0]]
        add("S20", 1.0 if bd else 0.0, bool(bd),
            "есть дверь на балкон" if bd else "двери (tag door) со 2 этажа на балкон нет")

    # ---- S21 устойчивость
    floating = floating_objects(allobj)
    s21 = 1 - _frac(len(floating), len(allobj))
    add("S21", s21, not floating,
        "все объекты опираются на землю или друг на друга" if not floating else
        f"висят в воздухе {len(floating)} объект(ов): {', '.join(o.id for o in floating[:8])}")

    # ---- S22 участок
    x0, x1, y0, y1 = C.PLOT
    outside = [o for o in allobj if o.aabb_min[0] < x0 - 0.01 or o.aabb_max[0] > x1 + 0.01
               or o.aabb_min[1] < y0 - 0.01 or o.aabb_max[1] > y1 + 0.01 or o.aabb_min[2] < -3]
    add("S22", 1 - _frac(len(outside), len(allobj)), not outside,
        "всё на участке" if not outside else f"за границей участка: {', '.join(o.id for o in outside[:6])}")

    # ---- P1/P2 план
    plan = sc.plan or {}
    if not plan:
        add("P1", 0, False, "нет поля plan")
        add("P2", 0, False, "нет поля plan")
    else:
        m, notes = [], []
        ef = FACADE_ALIASES.get(str(plan.get("entrance_facade", "")).strip().lower()[:12], str(plan.get("entrance_facade", "")).strip().upper()[:1])
        real_f = facade(entry) if entry else None
        m.append(ef == real_f)
        notes.append(f"вход {ef}/{real_f}")
        pr_ = _num(plan.get("risers"))
        if st:
            nr = len(st["risers"])
            m.append(pr_ is not None and round(pr_) in (nr, len(st["levels"])))
            notes.append(f"подъёмов {pr_}/{nr}")
            rh = _num(plan.get("riser_height"))
            m.append(rh is not None and abs(rh - float(np.mean(st["risers"]))) <= 0.012)
            notes.append(f"подъём {rh}/{np.mean(st['risers']):.3f}")
            td = _num(plan.get("tread_depth"))
            med = float(np.median(st["treads"])) if st["treads"] else 0
            m.append(td is not None and abs(td - med) <= 0.03)
            notes.append(f"проступь {td}/{med:.2f}")
        else:
            m += [False, False, False]
        rl = _num(plan.get("ramp_length"))
        m.append(rl is not None and abs(rl - ramp_run) <= max(0.3, 0.1 * ramp_run))
        notes.append(f"пандус {rl}/{ramp_run:.2f}")
        rh2 = _num(plan.get("ridge_height"))
        hmax_r = max((r.aabb_max[2] for r in roofs), default=float("nan"))
        m.append(rh2 is not None and abs(rh2 - hmax_r) <= 0.15)
        notes.append(f"конёк {rh2}/{hmax_r:.2f}")
        s = sum(m) / len(m)
        add("P1", s, s >= 0.83, "заявлено/построено: " + "; ".join(notes))

        rn, rh, td = _num(plan.get("risers")), _num(plan.get("riser_height")), _num(plan.get("tread_depth"))
        q = [
            rn is not None and rh is not None and abs(rn * rh - (C.FFL1 - C.FFL0)) <= 0.03,
            rh is not None and rh <= C.MAX_RISER + 1e-6,
            td is not None and td >= C.MIN_TREAD - 1e-6,
            rl is not None and rl >= (C.FFL0 / C.MAX_RAMP_SLOPE) - 0.05,
        ]
        add("P2", sum(q) / 4, all(q),
            f"подъёмы×высота = {(rn or 0) * (rh or 0):.3f} (нужно 3.0); подъём ≤ 0.19: {'да' if q[1] else 'нет'}; "
            f"проступь ≥ 0.26: {'да' if q[2] else 'нет'}; длина пандуса ≥ 5.4: {'да' if q[3] else 'нет'}")
    return out


def _aabb_overlap(a: Obj, b: Obj, eps=0.0):
    return bool(np.all(a.aabb_min <= b.aabb_max + eps) and np.all(b.aabb_min <= a.aabb_max + eps))


def _coverage(segs, lo, hi):
    if hi <= lo:
        return 0.0
    segs = sorted((max(lo, a), min(hi, b)) for a, b in segs if b > lo and a < hi)
    tot, cur = 0.0, lo
    for a, b in segs:
        if b <= cur:
            continue
        tot += b - max(a, cur)
        cur = b
    return tot / (hi - lo)


def floating_objects(objs, eps=0.03):
    """Объекты, не связанные с землёй цепочкой касаний."""
    n = len(objs)
    if n == 0:
        return []
    mn = np.array([o.aabb_min for o in objs])
    mx = np.array([o.aabb_max for o in objs])
    touch = np.all((mn[:, None, :] <= mx[None, :, :] + eps) & (mn[None, :, :] <= mx[:, None, :] + eps), -1)
    np.fill_diagonal(touch, False)
    shaped = [o.type in ("wedge", "gable") for o in objs]
    for i, j in zip(*np.nonzero(touch)):
        if i >= j:
            continue
        if shaped[i] or shaped[j]:
            if not (_surface_touch(objs[i], objs[j], eps) or _surface_touch(objs[j], objs[i], eps)):
                touch[i, j] = touch[j, i] = False
    grounded = mn[:, 2] <= 0.05
    seen = grounded.copy()
    stack = list(np.nonzero(grounded)[0])
    while stack:
        i = stack.pop()
        nb = np.nonzero(touch[i] & ~seen)[0]
        seen[nb] = True
        stack.extend(nb.tolist())
    return [objs[i] for i in np.nonzero(~seen)[0]]


def _surface_touch(a: Obj, b: Obj, eps):
    """a опирается на / проходит сквозь фигурную поверхность b (или b не фигурный)."""
    if b.type not in ("wedge", "gable"):
        if a.type in ("wedge", "gable"):
            return True
        return True
    xs = np.linspace(a.aabb_min[0], a.aabb_max[0], 4)
    ys = np.linspace(a.aabb_min[1], a.aabb_max[1], 4)
    X, Y = np.meshgrid(xs, ys)
    t = b.top_at(X, Y)
    if np.isfinite(t).any() and np.nanmax(t) >= a.aabb_min[2] - eps:
        return True
    # a стоит под b (b лежит на a)
    if b.aabb_min[2] >= a.aabb_max[2] - eps:
        return True
    return False


# ---------------------------------------------------------------------------
# проверки фазы «изменение ТЗ»
# ---------------------------------------------------------------------------

def run_change(pr: ParseResult, prev_scene: Scene | None, prev_base: list[CheckResult]):
    base = run_base(pr, phase="change")
    out = []

    def add(cid, score, passed, detail):
        name, ab, w = CHANGE[cid]
        out.append(CheckResult(cid, name, ab, w, float(np.clip(score, 0, 1)), bool(passed), detail, "change"))

    sc = pr.scene
    if sc is None:
        for cid in CHANGE:
            add(cid, 0, False, "нет сцены")
        return base, out
    gz = [o for o in sc.objects if in_garage_zone(o)]
    gw = [o for o in gz if o.tag == "wall"]
    if gw:
        mn = np.min([o.aabb_min for o in gw], 0)
        mx = np.max([o.aabb_max for o in gw], 0)
        p = [abs(mx[0] - C.GX1) <= C.TOL, abs(mn[1] - C.GY0) <= C.TOL, abs(mx[1] - C.GY1) <= C.TOL,
             mn[0] <= C.GX0 + 0.35, mn[2] <= 0.05 and mx[2] <= C.GARAGE_ROOF_TOP + 0.05]
        add("G1", sum(p) / 5, all(p), f"стены гаража: x {mn[0]:.2f}…{mx[0]:.2f}, y {mn[1]:.2f}…{mx[1]:.2f}, z {mn[2]:.2f}…{mx[2]:.2f}")
    else:
        add("G1", 0, False, "стен гаража нет (ожидались x 10…14, y 3…8)")

    gr = [o for o in gz if o.tag == "roof"]
    if gr:
        H = height_field(gr, np.arange(10.3, 13.71, 0.25), np.arange(3.3, 7.71, 0.25))
        okc = np.abs(np.nan_to_num(H, nan=-9) - C.GARAGE_ROOF_TOP) <= 0.12
        flat = np.isfinite(H).any() and (np.nanmax(H) - np.nanmin(H)) <= 0.1
        add("G2", okc.mean() * (1 if flat else 0.5), okc.mean() >= 0.85 and flat,
            f"крыша гаража на {C.GARAGE_ROOF_TOP} м покрывает {okc.mean():.0%}; плоская: {'да' if flat else 'нет'}")
    else:
        add("G2", 0, False, "крыши гаража нет")

    gd = [o for o in sc.objects if o.tag == "garage_door"]
    best, bdet = 0.0, "ворот (tag garage_door) нет"
    for d in gd:
        w, h = d.aabb_max[0] - d.aabb_min[0], d.aabb_max[2] - d.aabb_min[2]
        p = [abs(d.center[1] - C.GY1) <= 0.4, C.GX0 - 0.05 <= d.center[0] <= C.GX1 + 0.05, w >= 2.48, h >= 2.08, d.aabb_min[2] <= 0.1]
        s = sum(p) / 5
        if s >= best:
            best = s
            side = "север (улица)" if p[0] else f"не север (y={d.center[1]:.2f})"
            bdet = f"ворота {d.id}: {side}, ширина {w:.2f}, высота {h:.2f}, низ {d.aabb_min[2]:.2f}"
    add("G3", best, best == 1, bdet)

    hd = [o for o in sc.objects if o.tag == "door" and abs(o.center[0] - C.X1) <= 0.4 and C.GY0 < o.center[1] < C.GY1
          and -0.05 <= o.aabb_min[2] <= C.FFL0 + 0.15 and o.aabb_max[2] - o.aabb_min[2] >= 1.9]
    add("G4", 1.0 if hd else 0.0, bool(hd), "дверь из гаража в дом есть" if hd else "двери в восточной стене дома в пределах гаража нет")

    house_w = [o for o in sc.objects if o.tag == "window" and not in_garage_zone(o)]
    bad = [o for o in house_w if abs(o.center[0] - C.X1) <= 0.45 and level_of(o) == 0 and C.GY0 < o.center[1] < C.GY1]
    east0 = [o for o in house_w if facade(o) == "E" and level_of(o) == 0 and not (C.GY0 < o.center[1] < C.GY1)]
    s5 = 1.0 if (not bad and east0) else (0.5 if not bad else 0.0)
    add("G5", s5, s5 == 1, (f"окна в гараж: {', '.join(o.id for o in bad)}" if bad else "окна в гараж не выходят")
        + ("" if east0 else "; на восточном фасаде 1 эт. не осталось окна"))

    gbox = Obj("_g", "box", "other", [(C.GX0 + C.GX1) / 2, (C.GY0 + C.GY1) / 2, C.GARAGE_ROOF_TOP / 2],
               [C.GX1 - C.GX0 - 0.02, C.GY1 - C.GY0 - 0.02, C.GARAGE_ROOF_TOP])
    # пандус/площадка, заходящие в гараж снаружи (целиком внутренняя площадка у двери — допустима)
    inside = lambda o: bool(np.all(o.aabb_min >= gbox.aabb_min - 0.01) and np.all(o.aabb_max <= gbox.aabb_max + 0.01))  # noqa: E731
    hit = [o for o in sc.objects if o.tag in ("ramp", "landing") and _aabb_overlap(o, gbox) and not inside(o)]
    # коридор от ворот до улицы должен быть свободен (машина заезжает прямо с улицы)
    door = max(gd, key=lambda d: d.aabb_max[0] - d.aabb_min[0]) if gd else None
    dx0, dx1 = (door.aabb_min[0], door.aabb_max[0]) if door is not None else (C.GX0 + 0.3, C.GX1 - 0.3)
    lane = Obj("_lane", "box", "other", [(dx0 + dx1) / 2, (C.GY1 + 0.05 + C.STREET_Y[0]) / 2, 1.0],
               [max(dx1 - dx0 - 0.1, 0.1), C.STREET_Y[0] - C.GY1 - 0.1, 1.9])
    lane.aabb_min[2] = 0.06
    block = [o for o in sc.objects if o is not door and o.tag != "garage_door" and _aabb_overlap(o, lane)]
    parts = []
    if hit:
        parts.append(f"пандус/площадка пересекает гараж: {', '.join(o.id for o in hit)}")
    if block:
        parts.append(f"проезд от ворот к улице перекрыт: {', '.join(o.id for o in block[:6])}")
    s6 = 1.0 - 0.5 * bool(hit) - 0.5 * bool(block)
    add("G6", s6, not parts, "; ".join(parts) or "пандус не задет, проезд от ворот до улицы свободен")

    prev_ok = {c.id for c in prev_base if c.passed and not c.id.startswith("F")}
    now_ok = {c.id for c in base if c.passed}
    lost = sorted(prev_ok - now_ok)
    s7 = 1 - _frac(len(lost), len(prev_ok)) if prev_ok else 1.0
    add("G7", s7, not lost, "все прежде пройденные проверки сохранились" if not lost else
        f"сломалось после изменения: {', '.join(f'{i} {BASE[i][0]}' for i in lost)}")

    if prev_scene and prev_scene.objects:
        # сам дом (пятно застройки) не должен перестраиваться; восточный фасад 1 этажа и всё,
        # что снаружи (пандус и т.п.), можно менять — это прямое следствие пристройки гаража
        cur = {o.id: o for o in sc.objects}
        core = [o for o in prev_scene.objects
                if C.X0 <= o.center[0] <= C.X1 and C.Y0 <= o.center[1] <= C.Y1
                and not (o.tag in ("window", "door") and facade(o) == "E" and level_of(o) == 0)]
        same = 0
        for o in core:
            c = cur.get(o.id)
            if c is not None and c.type == o.type and np.allclose(c.center, o.center, atol=0.011) and np.allclose(c.size, o.size, atol=0.011):
                same += 1
        f = same / len(core) if core else 0.0
        add("G8", f, f >= 0.9, f"без изменений осталось {f:.0%} объектов дома ({same} из {len(core)})")
    else:
        add("G8", 0, False, "нет предыдущей сцены")
    return base, out


# ---------------------------------------------------------------------------
# агрегация
# ---------------------------------------------------------------------------

def weighted(checks):
    w = sum(c.weight for c in checks)
    return sum(c.weight * c.score for c in checks) / w if w else 0.0


def ability_scores(first_base, change_checks, self_fix, vision):
    res = {}
    pool = list(first_base) + list(change_checks or [])
    for key, _ in C.ABILITIES:
        if key == "self_fix":
            res[key] = self_fix
        elif key == "vision":
            res[key] = vision
        else:
            cs = [c for c in pool if key in c.abilities]
            res[key] = weighted(cs) if cs else None
    return res


def self_fix_score(first, final):
    f0 = {c.id for c in first if not c.passed}
    if not f0:
        return None
    fin = {c.id: c for c in final}
    first_ok = {c.id for c in first if c.passed}
    regress = sum(1 for i in first_ok if fin.get(i) and not fin[i].passed)
    # исправлено полностью = 1, частично = половина относительного прогресса
    part = []
    for c in first:
        if c.id not in f0:
            continue
        f = fin.get(c.id)
        if f is None:
            part.append(0.0)
        elif f.passed:
            part.append(1.0)
        else:
            part.append(0.5 * max(0.0, f.score - c.score) / max(1e-9, 1 - c.score))
    s = float(np.mean(part)) - regress / max(len(first), 1)
    return float(np.clip(s, 0, 1))


def overall(abilities):
    vals = [v for v in abilities.values() if v is not None]
    return float(np.mean(vals)) if vals else 0.0


def feedback_text(checks):
    bad = [c for c in checks if not c.passed]
    lines = [f"- [{c.id} {c.name}] {c.detail}" for c in bad]
    return "\n".join(lines)
