"""Сборка итогового ролика: титры → стройка → исправления → гараж → разрез → итоги."""
from __future__ import annotations

import math
import os
from multiprocessing import get_context

import numpy as np
from PIL import Image

from .. import config as C
from . import cards
from .raster import Camera, Mesh, ShadowMap, render

QUALITY = {"fast": (480, 360, 2), "normal": (640, 480, 2), "high": (960, 720, 2)}
FOV = 38.0
LIGHT = (-0.45, -0.55, 0.70)  # солнце на юго-западе
_SHADOW_CACHE: dict = {}


def framing(scenes):
    objs = [o for sc in scenes if sc for o in sc.objects]
    x0, x1, y0, y1 = C.PLOT
    if objs:
        mn = np.min([o.aabb_min for o in objs], 0)
        mx = np.max([o.aabb_max for o in objs], 0)
        mn = np.maximum(mn, [x0, y0, 0])
        mx = np.minimum(mx, [x1, y1, 16])
        mn = np.minimum(mn, [C.X0, C.Y0, 0])
        mx = np.maximum(mx, [C.X1, C.Y1, C.RIDGE])
    else:
        mn, mx = np.array([C.X0, C.Y0, 0]), np.array([C.X1, C.Y1, C.RIDGE])
    target = (mn + mx) / 2
    target[2] = mx[2] * 0.36
    radius = 0.5 * np.linalg.norm(mx - mn)
    dist = radius / math.tan(math.radians(FOV) / 2) * 0.84
    return target, dist


def _render_job(job):
    W, H, ss = job["size"]
    key = job.get("shadow_key")
    sm = _SHADOW_CACHE.get(key) if key else None
    objs = job["objs"]
    if sm is None and objs:
        sm = ShadowMap(Mesh(objs), LIGHT, (np.array([-9, -11, 0.0]), np.array([23, 17, 14.0])))
        if key:
            if len(_SHADOW_CACHE) > 6:
                _SHADOW_CACHE.clear()
            _SHADOW_CACHE[key] = sm
    target, dist, az, el = job["cam"]
    cam = Camera.orbit(target, dist, az, el, W * ss, H * ss, FOV)
    im = render(objs, cam, ss=ss, light_dir=LIGHT, shadow=sm)
    m, cap, score, step = job["overlay"]
    return np.asarray(cards.overlay(im, m, cap, score, step))


class Timeline:
    def __init__(self, W, H, ss, model):
        self.W, self.H, self.ss, self.model = W, H, ss, model
        self.items = []
        self.az = 215.0

    def card(self, im, seconds, fps):
        self.items += [("img", im)] * max(1, int(seconds * fps))

    def job(self, objs, target, dist, az, el, caption, score, step, shadow_key=None):
        self.items.append(("job", {
            "size": (self.W, self.H, self.ss), "objs": objs, "cam": (target, dist, az, el),
            "overlay": (self.model, caption, score, step), "shadow_key": shadow_key,
        }))

    def render(self, workers=None):
        jobs = [it[1] for it in self.items if it[0] == "job"]
        workers = workers or min(4, os.cpu_count() or 1)
        if workers > 1 and len(jobs) > 4:
            ctx = get_context("fork") if hasattr(os, "fork") else get_context()
            with ctx.Pool(workers) as pool:
                done = pool.map(_render_job, jobs, chunksize=max(1, len(jobs) // (workers * 4)))
        else:
            done = [_render_job(j) for j in jobs]
        it = iter(done)
        out = []
        for kind, v in self.items:
            out.append(Image.fromarray(next(it)) if kind == "job" else v)
        return out


def cutaway_objects(objs):
    keep = {"floor_slab", "stair_step", "balcony", "railing", "landing", "ramp", "foundation"}
    res = []
    for o in objs:
        if o.tag in ("roof", "chimney"):
            continue
        if o.aabb_min[2] >= C.FFL1 - 0.3 and o.tag not in keep:
            continue
        res.append(o)
    return res


def build_timeline(run, quality="normal", fps=12):
    W, H, ss = QUALITY[quality]
    model = run["model"]
    tl = Timeline(W, H, ss, model)
    scenes = [p["scene"] for p in run["phases"]]
    target, dist = framing(scenes)
    el = 21.0
    sub = ("Модель строит двухэтажный дом по ТЗ с неявными требованиями,\n"
           "исправляет ошибки по отчёту валидатора и достраивает гараж.\n"
           f"Проверок: {run.get('n_checks', 0)} · способностей: {len([a for a in run['abilities'].values() if a is not None])}")
    tl.card(cards.title_card(W, H, model, run.get("date", ""), sub), 2.0, fps)

    prev = None
    for k, ph in enumerate(run["phases"]):
        sc = ph["scene"]
        label, step = ph["label"], ph["step"]
        if sc is None or not sc.objects:
            tl.card(cards.message_card(W, H, model, f"{label}: нет модели", ph.get("error") or "ответ не разобран"), 1.6, fps)
            continue
        key = f"ph{k}"
        objs = sc.objects
        if ph["kind"] == "build" or prev is None:
            n = 40
            for i in range(1, n + 1):
                cnt = max(1, round(len(objs) * i / n))
                tl.job(objs[:cnt], target, dist, tl.az, el, label, ph["score"], step, shadow_key=f"{key}_{cnt}")
                tl.az += 2.2
            for _ in range(int(0.6 * fps)):
                tl.job(objs, target, dist, tl.az, el, label, ph["score"], step, shadow_key=key)
                tl.az += 1.2
        elif ph["kind"] == "change":
            old = {o.id: o for o in prev.objects}
            same = [o for o in objs if o.id in old and np.allclose(o.center, old[o.id].center, atol=0.011)
                    and np.allclose(o.size, old[o.id].size, atol=0.011)]
            ids = {o.id for o in same}
            new = [o for o in objs if o.id not in ids]
            # переходим к виду с северо-востока, где гараж и ворота
            goal = 40.0 + 360 * math.ceil((tl.az - 40) / 360)
            n = 24
            start = tl.az
            for i in range(1, n + 1):
                cnt = round(len(new) * i / n)
                tl.job(same + new[:cnt], target, dist, start + (goal - start) * i / n, el, label, ph["score"], step,
                       shadow_key=f"{key}_{cnt}")
            tl.az = goal
            for _ in range(int(2.5 * fps)):
                tl.job(objs, target, dist, tl.az, el, label, ph["score"], step, shadow_key=key)
                tl.az += 3.0
        else:  # fix
            for _ in range(int(2.0 * fps)):
                tl.job(objs, target, dist, tl.az, el, label, ph["score"], step, shadow_key=key)
                tl.az += 3.5
        prev = sc

    final = next((p["scene"] for p in reversed(run["phases"]) if p["scene"] is not None and p["scene"].objects), None)
    if final is not None:
        cut = cutaway_objects(final.objects)
        ct = np.array([5.0, 4.0, C.FFL1 * 0.6])
        for _ in range(int(2.6 * fps)):
            tl.job(cut, ct, dist * 0.78, tl.az, 52.0, "Разрез: лестница и проём в перекрытии", None, "РАЗРЕЗ", "cut")
            tl.az += 3.0

    tl.card(cards.scorecard(W, H, model, run["abilities"], run["overall"], run.get("stats_lines", [])), 4.0, fps)
    tl.card(cards.checks_card(W, H, model, run.get("checks_groups", [])), 3.5, fps)
    return tl


def save_gif(frames, path, fps=12, width=None):
    if width and frames[0].width != width:
        h = round(frames[0].height * width / frames[0].width)
        frames = [f.resize((width, h), Image.LANCZOS) for f in frames]
    step = max(1, len(frames) // 24)
    sample = frames[::step][:24]
    strip = Image.new("RGB", (sample[0].width, sample[0].height * len(sample)))
    for i, f in enumerate(sample):
        strip.paste(f, (0, i * f.height))
    pal = strip.quantize(colors=255, method=Image.Quantize.MEDIANCUT)
    q = [f.quantize(palette=pal, dither=Image.Dither.NONE) for f in frames]
    # склеиваем одинаковые подряд кадры (карточки) — меньше файл
    out, dur, prev = [], [], None
    ms = round(1000 / fps)
    for f, src in zip(q, frames):
        if out and src is prev:
            dur[-1] += ms
        else:
            out.append(f)
            dur.append(ms)
        prev = src
    out[0].save(path, save_all=True, append_images=out[1:], duration=dur, loop=0, optimize=False, disposal=1)
    return path


def save_mp4(frames, path, fps=12):
    try:
        import imageio.v2 as imageio
    except ImportError:
        return None
    try:
        w = imageio.get_writer(path, fps=fps, codec="libx264", quality=8, pixelformat="yuv420p", macro_block_size=8)
        for f in frames:
            w.append_data(np.asarray(f))
        w.close()
        return path
    except Exception as e:  # noqa: BLE001
        print(f"[video] MP4 не записан: {e}")
        return None


def make_video(run, out_dir, quality="normal", fps=12, gif=True, mp4=True, workers=None, gif_width=None):
    tl = build_timeline(run, quality, fps)
    frames = tl.render(workers)
    paths = {}
    if gif:
        paths["gif"] = save_gif(frames, os.path.join(out_dir, "video.gif"), fps, gif_width)
    if mp4:
        p = save_mp4(frames, os.path.join(out_dir, "video.mp4"), fps)
        if p:
            paths["mp4"] = p
    # постер — финальный кадр облёта
    poster = next((f for f, (k, _) in zip(reversed(frames), reversed(tl.items)) if k == "job"), frames[-1])
    poster.save(os.path.join(out_dir, "poster.png"))
    W, H, _ = QUALITY[quality]
    cards.scorecard(W, H, run["model"], run["abilities"], run["overall"], run.get("stats_lines", [])).save(
        os.path.join(out_dir, "scorecard.png"))
    paths["poster"] = os.path.join(out_dir, "poster.png")
    return paths


def render_still(scene, path, azim=225, elev=22, quality="normal"):
    W, H, ss = QUALITY[quality]
    target, dist = framing([scene])
    cam = Camera.orbit(target, dist, azim, elev, W * ss, H * ss, FOV)
    im = render(scene.objects, cam, ss=ss, light_dir=LIGHT)
    im.save(path)
    return path
