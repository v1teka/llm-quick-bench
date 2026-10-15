"""Титры, подписи и итоговые карточки (PIL)."""
from __future__ import annotations

import math
import os
from functools import lru_cache

from PIL import Image, ImageDraw, ImageFont

from .. import config as C

BG = (17, 19, 24)
PANEL = (29, 32, 40)
TEXT = (234, 234, 236)
DIM = (139, 145, 156)
ACCENT = (232, 163, 61)
GOOD = (76, 195, 138)
BAD = (229, 83, 75)
MID = (230, 190, 70)

_FONT_DIRS = [
    "/usr/share/fonts/truetype/dejavu", "/usr/share/fonts/dejavu", "/usr/share/fonts/TTF",
    "/Library/Fonts", "/System/Library/Fonts/Supplemental", "C:/Windows/Fonts",
]
_FONT_NAMES = {
    False: ["DejaVuSans.ttf", "Arial.ttf", "arial.ttf", "LiberationSans-Regular.ttf"],
    True: ["DejaVuSans-Bold.ttf", "Arial Bold.ttf", "arialbd.ttf", "LiberationSans-Bold.ttf"],
}


@lru_cache(maxsize=64)
def font(size, bold=False):
    for d in _FONT_DIRS:
        for n in _FONT_NAMES[bold]:
            p = os.path.join(d, n)
            if os.path.exists(p):
                return ImageFont.truetype(p, size)
    try:
        import matplotlib
        p = os.path.join(os.path.dirname(matplotlib.__file__), "mpl-data", "fonts", "ttf",
                         "DejaVuSans-Bold.ttf" if bold else "DejaVuSans.ttf")
        if os.path.exists(p):
            return ImageFont.truetype(p, size)
    except Exception:  # noqa: BLE001
        pass
    return ImageFont.load_default()


def score_color(v):
    if v is None:
        return DIM
    return GOOD if v >= 0.75 else (MID if v >= 0.5 else BAD)


def pct(v):
    return "—" if v is None else f"{round(v * 100)}%"


def _shorten(d, text, f, maxw):
    if d.textlength(text, font=f) <= maxw:
        return text
    while text and d.textlength(text + "…", font=f) > maxw:
        text = text[:-1]
    return text + "…"


def overlay(im: Image.Image, model: str, caption: str, score=None, step: str = ""):
    W, H = im.size
    s = W / 640
    im = im.convert("RGBA")
    ov = Image.new("RGBA", im.size, (0, 0, 0, 0))
    d = ImageDraw.Draw(ov)
    th = int(28 * s)
    d.rectangle([0, 0, W, th], fill=(10, 12, 16, 170))
    f1 = font(int(13 * s), True)
    d.text((int(10 * s), int(6 * s)), "llm-quick-bench · Дом-3D", font=f1, fill=ACCENT)
    f2 = font(int(13 * s))
    mw = d.textlength(model, font=f2)
    d.text((W - mw - int(10 * s), int(6 * s)), model, font=f2, fill=TEXT)
    if caption:
        bh = int(40 * s)
        d.rectangle([0, H - bh, W, H], fill=(10, 12, 16, 185))
        fc = font(int(17 * s), True)
        x = int(12 * s)
        if step:
            fs = font(int(12 * s), True)
            tw = d.textlength(step, font=fs) + int(12 * s)
            d.rounded_rectangle([x, H - bh + int(10 * s), x + tw, H - int(10 * s)], radius=int(6 * s), fill=ACCENT)
            d.text((x + int(6 * s), H - bh + int(13 * s)), step, font=fs, fill=BG)
            x += tw + int(10 * s)
        maxw = W - x - int(90 * s)
        d.text((x, H - bh + int(9 * s)), _shorten(d, caption, fc, maxw), font=fc, fill=TEXT)
        if score is not None:
            fsc = font(int(20 * s), True)
            t = pct(score)
            tw = d.textlength(t, font=fsc)
            d.text((W - tw - int(14 * s), H - bh + int(7 * s)), t, font=fsc, fill=score_color(score))
    return Image.alpha_composite(im, ov).convert("RGB")


def title_card(W, H, model, date, subtitle):
    s = W / 640
    im = Image.new("RGB", (W, H), BG)
    d = ImageDraw.Draw(im)
    d.text((int(40 * s), int(110 * s)), "llm-quick-bench", font=font(int(20 * s), True), fill=ACCENT)
    d.text((int(40 * s), int(140 * s)), "Дом-3D", font=font(int(54 * s), True), fill=TEXT)
    d.text((int(40 * s), int(215 * s)), _shorten(d, model, font(int(26 * s), True), W - 80 * s),
           font=font(int(26 * s), True), fill=TEXT)
    d.text((int(40 * s), int(255 * s)), date, font=font(int(15 * s)), fill=DIM)
    y = int(310 * s)
    for line in subtitle.split("\n"):
        d.text((int(40 * s), y), line, font=font(int(14 * s)), fill=DIM)
        y += int(22 * s)
    d.rectangle([int(40 * s), int(200 * s), int(120 * s), int(203 * s)], fill=ACCENT)
    return im


def message_card(W, H, model, title, text):
    s = W / 640
    im = Image.new("RGB", (W, H), BG)
    d = ImageDraw.Draw(im)
    d.text((int(40 * s), int(150 * s)), title, font=font(int(30 * s), True), fill=BAD)
    y = int(205 * s)
    f = font(int(15 * s))
    for line in _wrap(d, text, f, W - 80 * s)[:8]:
        d.text((int(40 * s), y), line, font=f, fill=TEXT)
        y += int(22 * s)
    return overlay(im, model, "", None)


def _wrap(d, text, f, maxw):
    out = []
    for para in text.split("\n"):
        cur = ""
        for w in para.split(" "):
            t = (cur + " " + w).strip()
            if d.textlength(t, font=f) <= maxw:
                cur = t
            else:
                if cur:
                    out.append(cur)
                cur = w
        out.append(cur)
    return out


def radar(im, cx, cy, R, items, s):
    n = len(items)
    d = ImageDraw.Draw(im)
    f = font(max(9, int(10 * s)))
    for k in (0.25, 0.5, 0.75, 1.0):
        pts = [(cx + R * k * math.sin(2 * math.pi * i / n), cy - R * k * math.cos(2 * math.pi * i / n)) for i in range(n)]
        d.polygon(pts, outline=(55, 60, 72))
    for i, (label, v) in enumerate(items):
        a = 2 * math.pi * i / n
        x, y = cx + R * math.sin(a), cy - R * math.cos(a)
        d.line([cx, cy, x, y], fill=(55, 60, 72))
        lx, ly = cx + (R + 8 * s) * math.sin(a), cy - (R + 8 * s) * math.cos(a)
        tw = d.textlength(label, font=f)
        sa = math.sin(a)
        ax = lx - tw / 2 + (tw / 2 + 2 * s) * (1 if sa > 0.2 else -1 if sa < -0.2 else 0)
        ay = ly - 6 * s + (-6 * s if math.cos(a) > 0.9 else 6 * s if math.cos(a) < -0.9 else 0)
        d.text((ax, ay), label, font=f, fill=TEXT if v is not None else DIM)
    pts = [(cx + R * (v or 0) * math.sin(2 * math.pi * i / n), cy - R * (v or 0) * math.cos(2 * math.pi * i / n))
           for i, (_, v) in enumerate(items)]
    ov = Image.new("RGBA", im.size, (0, 0, 0, 0))
    od = ImageDraw.Draw(ov)
    od.polygon(pts, fill=(232, 163, 61, 80))
    im.alpha_composite(ov)
    d = ImageDraw.Draw(im)
    d.line(pts + [pts[0]], fill=ACCENT, width=max(1, int(2 * s)))
    for (x, y), (_, v) in zip(pts, items):
        if v is not None:
            r = 3 * s
            d.ellipse([x - r, y - r, x + r, y + r], fill=score_color(v))


def scorecard(W, H, model, abilities, total, stats_lines):
    s = W / 640
    im = Image.new("RGBA", (W, H), BG + (255,))
    d = ImageDraw.Draw(im)
    d.text((int(22 * s), int(38 * s)), "Итог", font=font(int(14 * s), True), fill=DIM)
    d.text((int(22 * s), int(54 * s)), pct(total), font=font(int(42 * s), True), fill=score_color(total))
    items = [(name, abilities.get(key)) for key, name in C.ABILITIES]
    radar(im, int(190 * s), int(228 * s), int(92 * s), items, s)
    d = ImageDraw.Draw(im)
    x0, y = int(395 * s), int(42 * s)
    fl, fv = font(int(12 * s)), font(int(12 * s), True)
    bw = int(222 * s)
    for name, v in items:
        d.text((x0, y), name, font=fl, fill=TEXT if v is not None else DIM)
        tv = pct(v)
        d.text((x0 + bw - d.textlength(tv, font=fv), y), tv, font=fv, fill=score_color(v))
        yb = y + int(17 * s)
        d.rounded_rectangle([x0, yb, x0 + bw, yb + int(4 * s)], radius=int(2 * s), fill=PANEL)
        if v:
            d.rounded_rectangle([x0, yb, x0 + max(int(4 * s), int(bw * v)), yb + int(4 * s)], radius=int(2 * s), fill=score_color(v))
        y += int(29 * s)
    fs = font(max(9, int(11 * s)))
    y = H - int(42 * s) - int(17 * s) * len(stats_lines) - int(8 * s)
    d.line([int(22 * s), y - int(8 * s), W - int(22 * s), y - int(8 * s)], fill=PANEL, width=1)
    for line in stats_lines:
        d.text((int(22 * s), y), _shorten(d, line, fs, W - 44 * s), font=fs, fill=DIM)
        y += int(17 * s)
    return overlay(im.convert("RGB"), model, "Профиль способностей", total, "ИТОГ")


def checks_card(W, H, model, groups):
    """groups: [(заголовок, [CheckResult...])]"""
    s = W / 640
    im = Image.new("RGB", (W, H), BG)
    d = ImageDraw.Draw(im)
    rows = []
    for title, cs in groups:
        rows.append(("h", title, None))
        rows += [("c", c, None) for c in cs]
    col_h = H - int(80 * s)
    lh = int(15 * s)
    per_col = max(1, col_h // lh)
    ncol = max(1, math.ceil(len(rows) / per_col))
    ncol = min(ncol, 3)
    per_col = math.ceil(len(rows) / ncol)
    cw = (W - int(30 * s)) / ncol
    f, fb = font(max(9, int(10 * s))), font(max(9, int(10 * s)), True)
    for i, (kind, item, _) in enumerate(rows):
        col, row = divmod(i, per_col)
        x, y = int(15 * s + col * cw), int(36 * s + row * lh)
        if kind == "h":
            d.text((x, y), item, font=fb, fill=ACCENT)
        else:
            mark, colr = ("✓", GOOD) if item.passed else ("✗", BAD)
            d.text((x, y), mark, font=fb, fill=colr)
            label = f"{item.id} {item.name}"
            d.text((x + int(14 * s), y), _shorten(d, label, f, cw - int(52 * s)), font=f, fill=TEXT)
            t = pct(item.score)
            d.text((x + cw - int(14 * s) - d.textlength(t, font=f), y), t, font=f, fill=colr if not item.passed else DIM)
    return overlay(im, model, "Все проверки", None, "ДЕТАЛИ")
