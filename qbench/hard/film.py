"""Ролик «Дом-3D Pro»: стройка, облёт, дверь открывается, камера влетает в дом и по очереди
подлетает к пяти телевизорам: на четырёх — результаты задач, на пятом — итоги бенчмарка."""
from __future__ import annotations

import heapq
import math
import os
from multiprocessing import get_context

import numpy as np
from PIL import Image
from scipy import ndimage

from ..render.raster2 import Camera, Mesh, ShadowMap, render, screen_info
from ..scene import Obj
from . import config as C
from . import tv

QUALITY = {"fast": (480, 270, 2), "normal": (640, 360, 2), "high": (960, 540, 2)}
LIGHT = (-0.45, -0.55, 0.70)
G: dict = {}


# ---------------------------------------------------------------------------
# воксели и путь камеры
# ---------------------------------------------------------------------------

class Voxels:
    def __init__(self, sc, bounds=(-3.0, 16.0, -3.0, 14.0, 0.0, 8.5), res=0.2, skip=()):
        self.x0, self.x1, self.y0, self.y1, self.z0, self.z1 = bounds
        self.res = res
        self.shape = (int((self.x1 - self.x0) / res), int((self.y1 - self.y0) / res), int((self.z1 - self.z0) / res))
        occ = np.zeros(self.shape, bool)
        skip = set(skip)
        lo = np.floor((sc.amin - [self.x0, self.y0, self.z0]) / res).astype(int)
        hi = np.ceil((sc.amax - [self.x0, self.y0, self.z0]) / res).astype(int)
        lo = np.clip(lo, 0, np.array(self.shape))
        hi = np.clip(hi, 0, np.array(self.shape))
        for i in range(len(sc.objects)):
            if i in skip or sc.tags[i] in ("rug", "paving", "door"):
                continue
            a, b = lo[i], hi[i]
            if (b > a).all():
                occ[a[0]:b[0], a[1]:b[1], a[2]:b[2]] = True
        occ[:, :, 0] = True  # земля
        self.occ = occ
        self.clear = ndimage.distance_transform_edt(~occ) * res

    def idx(self, p):
        return tuple(int(np.clip((p[k] - (self.x0, self.y0, self.z0)[k]) / self.res, 0, self.shape[k] - 1)) for k in range(3))

    def pos(self, ijk):
        return np.array([self.x0 + (ijk[0] + 0.5) * self.res, self.y0 + (ijk[1] + 0.5) * self.res, self.z0 + (ijk[2] + 0.5) * self.res])

    def free(self, ijk, rad=0.3):
        return self.clear[ijk] >= rad

    def los(self, a, b, rad=0.25):
        n = int(np.linalg.norm(b - a) / (self.res * 0.5)) + 2
        for t in np.linspace(0, 1, n):
            if self.clear[self.idx(a + (b - a) * t)] < rad:
                return False
        return True

    def nearest_free(self, p, rad=0.3, max_r=1.2):
        i0 = np.array(self.idx(p))
        best, bd = None, 1e9
        r = int(max_r / self.res)
        for di in range(-r, r + 1):
            for dj in range(-r, r + 1):
                for dk in range(-r // 2, r // 2 + 1):
                    q = tuple(np.clip(i0 + (di, dj, dk), 0, np.array(self.shape) - 1))
                    if self.clear[q] >= rad:
                        d = di * di + dj * dj + 2 * dk * dk
                        if d < bd:
                            best, bd = q, d
        return best


def eye_height(p):
    inside = C.X0 - 0.2 <= p[0] <= C.X1 + 0.2 and C.Y0 - 0.2 <= p[1] <= C.Y1 + 0.2
    if not inside:
        return C.FFL0 + 1.55
    return (C.FFL1 if p[2] > C.FFL1 - 0.3 else C.FFL0) + 1.55


def astar(V, a, b, rad=0.28):
    sa, sb = V.nearest_free(a, rad), V.nearest_free(b, rad)
    if sa is None or sb is None:
        return None
    nbr = [(i, j, k) for i in (-1, 0, 1) for j in (-1, 0, 1) for k in (-1, 0, 1) if (i, j, k) != (0, 0, 0)]
    cost0 = {(i, j, k): math.sqrt(i * i + j * j + k * k) for i, j, k in nbr}
    goal = np.array(sb)
    openq = [(0.0, sa)]
    g = {sa: 0.0}
    prev = {sa: None}
    nexp = 0
    while openq and nexp < 250000:
        _, cur = heapq.heappop(openq)
        if cur == sb:
            break
        nexp += 1
        p = V.pos(cur)
        eh = eye_height(p)
        for d in nbr:
            q = (cur[0] + d[0], cur[1] + d[1], cur[2] + d[2])
            if not (0 <= q[0] < V.shape[0] and 0 <= q[1] < V.shape[1] and 0 <= q[2] < V.shape[2]):
                continue
            if V.clear[q] < rad:
                continue
            pz = V.z0 + (q[2] + 0.5) * V.res
            pen = 0.8 * abs(pz - eh) * V.res + (0.4 if d[2] else 0.0) * V.res
            ng = g[cur] + cost0[d] * V.res + pen
            if ng < g.get(q, 1e18):
                g[q] = ng
                prev[q] = cur
                h = np.linalg.norm((np.array(q) - goal)) * V.res
                heapq.heappush(openq, (ng + h, q))
    if sb not in prev:
        return None
    path = []
    k = sb
    while k is not None:
        path.append(V.pos(k))
        k = prev[k]
    path.reverse()
    return [np.asarray(a, float)] + path + [np.asarray(b, float)]


def smooth(V, pts, rad=0.22):
    if len(pts) < 3:
        return pts
    out = [pts[0]]
    i = 0
    while i < len(pts) - 1:
        j = len(pts) - 1
        while j > i + 1 and not V.los(pts[i], pts[j], rad):
            j -= 1
        out.append(pts[j])
        i = j
    return out


def resample(pts, step):
    pts = [np.asarray(p, float) for p in pts]
    if len(pts) == 1:
        return pts
    # Катмулл–Ром
    P = [pts[0]] + pts + [pts[-1]]
    dense = []
    for i in range(1, len(P) - 2):
        p0, p1, p2, p3 = P[i - 1], P[i], P[i + 1], P[i + 2]
        n = max(2, int(np.linalg.norm(p2 - p1) / 0.05))
        for t in np.linspace(0, 1, n, endpoint=False):
            t2, t3 = t * t, t * t * t
            dense.append(0.5 * ((2 * p1) + (-p0 + p2) * t + (2 * p0 - 5 * p1 + 4 * p2 - p3) * t2 + (-p0 + 3 * p1 - 3 * p2 + p3) * t3))
    dense.append(pts[-1])
    out = [dense[0]]
    acc = 0.0
    for a, b in zip(dense, dense[1:]):
        acc += np.linalg.norm(b - a)
        if acc >= step:
            out.append(b)
            acc = 0.0
    if np.linalg.norm(out[-1] - pts[-1]) > 1e-6:
        out.append(pts[-1])
    return out


# ---------------------------------------------------------------------------
# дверь
# ---------------------------------------------------------------------------

def find_main_door(sc):
    """-> (проём входной двери на северном фасаде или None, индекс закрытого полотна для анимации или None).
    Полотно, нарисованное открытым (поперёк фасада), не анимируется."""
    from .house_checks import door_openings
    best = None
    for op in door_openings(sc):
        if op.center[1] < C.Y1 - 0.6 or op.aabb_min[2] > C.FFL0 + 0.3 or op.aabb_max[2] > C.FFL1:
            continue
        w = op.aabb_max[0] - op.aabb_min[0]
        if best is None or w > best[1]:
            best = (op, w)
    if best is None:
        return None, None
    op = best[0]
    closed = [i for i in op.leaf_idx
              if sc.objects[i].aabb_max[0] - sc.objects[i].aabb_min[0] > sc.objects[i].aabb_max[1] - sc.objects[i].aabb_min[1]]
    leaf = max(closed, key=lambda i: sc.objects[i].aabb_max[0] - sc.objects[i].aabb_min[0]) if closed else None
    return op, leaf


def door_group(sc, i):
    o = sc.objects[i]
    lo, hi = o.aabb_min - [0.08, 0.18, 0.05], o.aabb_max + [0.08, 0.18, 0.05]
    out = [i]
    for j in sc.idx("decor", "door"):
        if j == i:
            continue
        q = sc.objects[j]
        if np.all(q.aabb_min >= lo) and np.all(q.aabb_max <= hi) and (q.aabb_max - q.aabb_min).max() < 0.6:
            out.append(j)
    return out


def rotated(o, hinge, ang):
    c, s = math.cos(ang), math.sin(ang)
    d = o.center[:2] - hinge
    nc = hinge + np.array([c * d[0] - s * d[1], s * d[0] + c * d[1]])
    return Obj(o.id, o.type, o.tag, [nc[0], nc[1], o.center[2]], o.size, o.color, o.yaw + ang, o.slope, o.ridge)


# ---------------------------------------------------------------------------
# рендер-воркер
# ---------------------------------------------------------------------------

def _job(job):
    kind = job["kind"]
    W, H, ss = G["size"]
    if kind == "card":
        return np.asarray(G["cards"][job["card"]])
    if kind == "channel":
        im = letterbox(channel_image(job["ch"], job["t"]), W, H)
        return np.asarray(tv.overlay16(im, G["model"], job["caption"], job["score"], job["step"]))
    eye, tgt, fov = job["eye"], job["tgt"], job["fov"]
    cam = Camera(eye, tgt, W * ss, H * ss, fov)
    mesh = G["mesh"]
    hide = None
    if job.get("upto") is not None:
        mesh = G["mesh"].subset(G["build_rank"] < job["upto"])
    if job.get("door") is not None and G.get("door_idx"):
        ang = job["door"]
        dyn = [rotated(G["scene"].objects[i], G["hinge"], ang) for i in G["door_idx"]]
        mesh = Mesh.concat(mesh.subset(~G["door_mask"]), Mesh(dyn))
    am, ax = G["amin"], G["amax"]
    inside = np.all((am - 0.14 <= eye) & (ax + 0.14 >= eye), 1)
    if inside.any() and job.get("upto") is None and job.get("door") is None:
        hide = inside
    textures = {k: np.asarray(channel_image(k + 1, job.get("tv_t", {}).get(k + 1, 0.5)), float) / 255 for k in range(5)} \
        if job.get("tv", True) else None
    sm = G["shadow"] if job.get("upto") is None else G["stage_shadows"][job["stage"]]
    im = render(mesh, cam, ss=ss, light_dir=LIGHT, shadow=sm, screens=G["screens"], textures=textures,
                exposure=job.get("exposure", 1.0), plot=C.PLOT, street=C.STREET_Y, hide_mask=hide)
    if job.get("blend") is not None:
        b, ch, t = job["blend"]
        full = letterbox(channel_image(ch, t), im.size[0], im.size[1])
        im = Image.blend(im, full, b)
    return np.asarray(tv.overlay16(im, G["model"], job["caption"], job.get("score"), job.get("step", "")))


_CH_CACHE: dict = {}


def letterbox(ch_img, W, H):
    s = W / 640
    top, bot = int(26 * s), int(36 * s)
    avail_h = H - top - bot
    w = int(avail_h * ch_img.width / ch_img.height)
    if w > W:
        w = W
        avail_h = int(W * ch_img.height / ch_img.width)
    bg = Image.new("RGB", (W, H), (8, 9, 12))
    bg.paste(ch_img.resize((w, avail_h), Image.LANCZOS), ((W - w) // 2, top + (H - top - bot - avail_h) // 2))
    return bg


def channel_image(ch, t):
    t = round(float(t), 3)
    key = (ch, t)
    if key in _CH_CACHE:
        return _CH_CACHE[key]
    tasks = G["tasks"]
    if ch == 5:
        im = tv.channel_summary(G["summary"], t)
    else:
        name = {1: "arc", 2: "optics", 3: "nav", 4: "cube"}[ch]
        tsk = tasks.get(name)
        if tsk is None:
            im, d, s = tv._base(480, 270, ch, {1: "Логика", 2: "Оптика", 3: "Навигация", 4: "Кубик"}[ch], None)
            d.text((40, 120), "задача не запускалась", font=tv.font(16), fill=tv.DIM)
        else:
            fn = {"arc": tv.channel_arc, "optics": tv.channel_optics, "nav": tv.channel_nav, "cube": tv.channel_cube}[name]
            im = fn(tsk["inst"], tsk["res"], t)
    if len(_CH_CACHE) > 400:
        _CH_CACHE.clear()
    _CH_CACHE[key] = im
    return im


# ---------------------------------------------------------------------------
# сценарий
# ---------------------------------------------------------------------------

def build_film(summary, scene, tasks, out_dir, quality="normal", fps=12, workers=None, log=print):
    W, H, ss = QUALITY[quality]
    model = summary["model"]
    sc = scene
    G.clear()
    G.update(size=(W, H, ss), model=model, summary=summary, tasks=tasks, scene=sc, amin=sc.amin, amax=sc.amax)
    tv_ids = ["tv1", "tv2", "tv3", "tv4", "tv5"]
    scr_map = {tid: k for k, tid in enumerate(tv_ids) if tid in sc.tvs}
    screens = {}
    for tid, k in scr_map.items():
        o = sc.tvs[tid]
        screens[k] = screen_info(o, getattr(o, "facing", "-y"))
    G["screens"] = screens
    log("  меш сцены…")
    mesh = Mesh(sc.objects, scr_map)
    G["mesh"] = mesh
    bnd = (np.array([-9, -9, 0.0]), np.array([23, 17, 11.5]))
    G["shadow"] = ShadowMap(mesh, LIGHT, bnd, 2048)
    # порядок стройки: снизу вверх
    order = np.lexsort((sc.amin[:, 0], sc.amin[:, 1], np.round(sc.amin[:, 2], 1)))
    rank = np.empty(len(order), int)
    rank[order] = np.arange(len(order))
    G["build_rank"] = rank
    n = len(order)
    stages = [int(n * f) for f in (0.15, 0.35, 0.55, 0.75, 0.9)]
    log("  тени этапов стройки…")
    G["stage_shadows"] = [ShadowMap(mesh.subset(rank < s), LIGHT, bnd, 1024) for s in stages]
    # дверь
    op, di = find_main_door(sc)
    if di is not None:
        grp = door_group(sc, di)
        G["door_idx"] = grp
        mask = np.zeros(len(sc.objects), bool)
        mask[grp] = True
        G["door_mask"] = mask
        d = sc.objects[di]
        G["hinge"] = np.array([d.aabb_min[0], d.center[1]])
        door_c = d.center.copy()
    elif op is not None:
        door_c = op.center.copy()
    else:
        door_c = np.array([6.5, C.Y1, C.FFL0 + 1.0])
    # карточки
    title = tv.title16(W, H, model, summary.get("date", ""), [
        "Модель строит детальный дом (≥ 10 000 примитивов) по ТЗ с неявными требованиями,",
        "исправляет ошибки по отчёту валидатора и решает 4 задачи — их результаты",
        "показывают телевизоры внутри дома. Пятый телевизор — итоги бенчмарка.",
        f"Проверок дома: {summary.get('n_house_checks', 0)} · способностей: {len([v for v in summary['abilities'].values() if v is not None])}"])
    G["cards"] = {"title": title, "checks": tv.checks16(W, H, model, summary.get("checks_groups", []))}
    jobs = []
    hs = summary.get("house_score")
    jobs += [{"kind": "card", "card": "title"}] * int(2.2 * fps)

    # --- стройка + облёт снаружи
    tgt = np.array([6.5, 4.5, 3.2])
    dist = 30.0
    nb = int(4.0 * fps)
    for i in range(nb):
        f = (i + 1) / nb
        upto = int(n * f)
        st = max(0, sum(1 for s in stages if s <= upto) - 1)
        a = math.radians(215 + 45 * f)
        e = math.radians(24)
        eye = tgt + dist * np.array([math.cos(e) * math.cos(a), math.cos(e) * math.sin(a), math.sin(e)])
        jobs.append({"kind": "scene", "eye": eye, "tgt": tgt, "fov": 42, "upto": upto, "stage": st, "tv": False,
                     "caption": f"Стройка: {upto} из {n} примитивов", "score": None, "step": "ДОМ"})
    no = int(3.0 * fps)
    for i in range(no):
        f = i / no
        a = math.radians(260 + 70 * f)
        e = math.radians(24 - 6 * f)
        eye = tgt + dist * np.array([math.cos(e) * math.cos(a), math.cos(e) * math.sin(a), math.sin(e)])
        jobs.append({"kind": "scene", "eye": eye, "tgt": tgt, "fov": 42, "tv": False,
                     "caption": f"Дом: {n} примитивов", "score": hs, "step": "ДОМ"})
    # --- подлёт к двери (с улицы), дверь открывается
    out_n = np.array([0, 1.0, 0])
    p_out = np.array([door_c[0], door_c[1] + 7.0, 3.2])
    p_door = np.array([door_c[0], door_c[1] + 1.3, C.FFL0 + 1.5])
    last_eye = jobs[-1]["eye"]
    na = int(2.2 * fps)
    for i in range(na):
        f = (i + 1) / na
        s_ = f * f * (3 - 2 * f)
        eye = last_eye * (1 - s_) + p_out * s_
        t_ = tgt * (1 - s_) + np.array([door_c[0], door_c[1], C.FFL0 + 1.3]) * s_
        jobs.append({"kind": "scene", "eye": eye, "tgt": t_, "fov": 42 + 8 * s_, "tv": False,
                     "caption": "Вход со стороны улицы", "score": hs, "step": "ВХОД"})
    nd = int(1.6 * fps)
    for i in range(nd):
        f = (i + 1) / nd
        s_ = f * f * (3 - 2 * f)
        eye = p_out * (1 - s_) + p_door * s_
        jobs.append({"kind": "scene", "eye": eye, "tgt": np.array([door_c[0], door_c[1] - 2, C.FFL0 + 1.4]), "fov": 55,
                     "door": -math.radians(98) * s_ if di is not None else None, "tv": False,
                     "caption": "Дверь открывается…" if di is not None else "Вход в дом", "score": hs, "step": "ВХОД"})
    # --- маршрут по дому
    log("  воксели и маршрут камеры…")
    V = Voxels(sc, skip=G.get("door_idx", []))
    vps = []
    for k, tid in enumerate(tv_ids):
        if tid not in sc.tvs:
            vps.append(None)
            continue
        S = screens[scr_map[tid]]
        diag = math.hypot(S["w"], S["h"])
        best = None
        for dd in (1.5 * diag, 1.3 * diag, 1.8 * diag, 1.1 * diag, 2.2 * diag):
            for lat in (0.0, 0.3, -0.3, 0.6, -0.6):
                p = S["center"] + S["normal"] * dd + S["right"] * lat
                p = np.array([p[0], p[1], S["center"][2] + 0.05])
                q = V.idx(p)
                if V.clear[q] >= 0.25 and V.los(p, S["center"] + S["normal"] * 0.15, 0.08):
                    best = p
                    break
            if best is not None:
                break
        if best is None:
            best = S["center"] + S["normal"] * 1.4 * diag
        vps.append((best, S))
    cur = p_door
    cur_dir = np.array([0, -1.0, 0])
    step = 2.1 / fps
    ch_names = {1: "Логика", 2: "Оптика", 3: "Навигация", 4: "Кубик", 5: "Итоги бенчмарка"}
    ch_scores = {1: _ts(tasks, "arc"), 2: _ts(tasks, "optics"), 3: _ts(tasks, "nav"), 4: _ts(tasks, "cube"), 5: summary["overall"]}
    for k, vp in enumerate(vps):
        ch = k + 1
        if vp is None:
            jobs += [{"kind": "channel", "ch": ch, "t": t, "caption": f"Канал {ch}: {ch_names[ch]} (телевизор не найден)",
                      "score": ch_scores[ch], "step": f"ТВ {ch}"} for t in np.linspace(0, 1, int(3 * fps))]
            continue
        goal, S = vp
        path = astar(V, cur, goal)
        ghost = path is None
        if ghost:
            path = [cur, goal]
        else:
            path = smooth(V, path)
        pts = resample(path, step)
        look = S["center"]
        nseg = len(pts)
        for i, p in enumerate(pts):
            ahead = pts[min(len(pts) - 1, i + 6)]
            d = ahead - p
            if np.linalg.norm(d) < 1e-3:
                d = cur_dir
            d = d / (np.linalg.norm(d) + 1e-9)
            cur_dir = d
            w = max(0.0, (i - (nseg - 14)) / 14)
            tgt_p = (p + d * 2.0) * (1 - w) + look * w
            tgt_p = tgt_p.copy()
            if w == 0:
                tgt_p[2] = p[2] - 0.15
            jobs.append({"kind": "scene", "eye": p, "tgt": tgt_p, "fov": 62, "exposure": _expo(p),
                         "caption": f"К телевизору {ch}…" if not ghost else f"К телевизору {ch} (сквозь стены: прохода нет)",
                         "score": hs, "step": "ДОМ"})
        cur = pts[-1]
        dwell = int(2.4 * fps)
        for i in range(dwell):
            t = i / dwell * 0.45
            jobs.append({"kind": "scene", "eye": cur, "tgt": look, "fov": 58, "exposure": _expo(cur),
                         "tv_t": {ch: t}, "caption": f"Канал {ch}: {ch_names[ch]}", "score": ch_scores[ch], "step": f"ТВ {ch}"})
        nb_ = int(0.5 * fps)
        for i in range(nb_):
            jobs.append({"kind": "scene", "eye": cur, "tgt": look, "fov": 58, "exposure": _expo(cur), "tv_t": {ch: 0.45},
                         "blend": ((i + 1) / nb_, ch, 0.45), "caption": f"Канал {ch}: {ch_names[ch]}", "score": ch_scores[ch],
                         "step": f"ТВ {ch}"})
        hold = int((3.4 if ch < 5 else 4.5) * fps)
        for i in range(hold):
            jobs.append({"kind": "channel", "ch": ch, "t": 0.45 + 0.55 * i / hold, "caption": f"Канал {ch}: {ch_names[ch]}",
                         "score": ch_scores[ch], "step": f"ТВ {ch}"})
        if ch < 5:
            for i in range(nb_):
                jobs.append({"kind": "scene", "eye": cur, "tgt": look, "fov": 58, "exposure": _expo(cur), "tv_t": {ch: 1.0},
                             "blend": (1 - (i + 1) / nb_, ch, 1.0), "caption": f"Канал {ch}: {ch_names[ch]}",
                             "score": ch_scores[ch], "step": f"ТВ {ch}"})
    jobs += [{"kind": "card", "card": "checks"}] * int(4.0 * fps)

    log(f"  кадров: {len(jobs)}; рендер…")
    workers = workers or min(4, os.cpu_count() or 1)
    if workers > 1:
        ctx = get_context("fork")
        with ctx.Pool(workers) as pool:
            frames = pool.map(_job, jobs, chunksize=4)
    else:
        frames = [_job(j) for j in jobs]
    frames = [Image.fromarray(f) for f in frames]
    return frames


def _ts(tasks, key):
    t = tasks.get(key)
    return t["res"]["score"] if t else None


def _expo(p):
    inside = C.X0 < p[0] < C.X1 and C.Y0 < p[1] < C.Y1 and p[2] < C.EAVE
    return 1.32 if inside else 1.0


def save_outputs(frames, out_dir, fps=12, gif_width=480, gif_every=2):
    import imageio.v2 as imageio
    paths = {}
    mp4 = os.path.join(out_dir, "video.mp4")
    w = imageio.get_writer(mp4, fps=fps, codec="libx264", quality=8, pixelformat="yuv420p", macro_block_size=8)
    for f in frames:
        w.append_data(np.asarray(f))
    w.close()
    paths["mp4"] = mp4
    # GIF — облегчённый (каждый 2-й кадр, меньше размер)
    sub = frames[::gif_every]
    h = round(sub[0].height * gif_width / sub[0].width)
    sub = [f.resize((gif_width, h), Image.LANCZOS) for f in sub]
    step = max(1, len(sub) // 24)
    strip = Image.new("RGB", (sub[0].width, sub[0].height * len(sub[::step][:24])))
    for i, f in enumerate(sub[::step][:24]):
        strip.paste(f, (0, i * f.height))
    pal = strip.quantize(colors=255, method=Image.Quantize.MEDIANCUT)
    q = [f.quantize(palette=pal, dither=Image.Dither.NONE) for f in sub]
    gif = os.path.join(out_dir, "video.gif")
    q[0].save(gif, save_all=True, append_images=q[1:], duration=round(1000 * gif_every / fps), loop=0, optimize=False)
    paths["gif"] = gif
    return paths
