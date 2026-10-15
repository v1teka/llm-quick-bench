"""Визуальный тест: подсчёт окон и поиск отличий на двух рендерах эталонного дома.

Картинки одинаковы для всех моделей (фиксированный seed), ответы проверяются точно.
"""
from __future__ import annotations

import copy
import json
import os
import random

from . import prompts
from .reference import GLASS, reference_house, _box
from .scene import Scene, parse_response  # noqa: F401
from .render.video import render_still

OPTIONS = {
    "a": "Убрана дымовая труба",
    "b": "Кровля перекрашена в зелёный цвет",
    "c": "Убран балкон вместе с ограждением",
    "d": "Добавлено окно на восточном фасаде 1 этажа",
    "e": "Убрано одно окно на южном фасаде 1 этажа",
    "f": "Входная дверь перенесена на восточный фасад",
    "g": "Дерево возле дома убрано",
}
SEED = 7


def _apply(d, key):
    objs = d["objects"]
    if key == "a":
        d["objects"] = [o for o in objs if o["tag"] != "chimney"]
    elif key == "b":
        for o in objs:
            if o["tag"] == "roof":
                o["color"] = "#3f7a3a"
    elif key == "c":
        d["objects"] = [o for o in objs if not (o["id"].startswith("bal") or o["id"] == "door_balcony")]
    elif key == "d":
        objs.append(_box("win_e1_new", "window", 9.67, 10.03, 4.0, 5.2, 1.35, 2.75, GLASS))
    elif key == "e":
        d["objects"] = [o for o in objs if o["id"] != "win_s1_2"]
    elif key == "g":
        d["objects"] = [o for o in objs if not o["id"].startswith("tree")]
    return d


def make_task(out_dir, quality="normal"):
    rng = random.Random(SEED)
    changed = sorted(rng.sample(["a", "b", "c", "d", "e"], 3))
    base = reference_house()
    mod = copy.deepcopy(base)
    for k in changed:
        mod = _apply(mod, k)
    os.makedirs(out_dir, exist_ok=True)
    pa, pb = os.path.join(out_dir, "vision_A.png"), os.path.join(out_dir, "vision_B.png")
    render_still(Scene.from_data(base), pa, azim=312, elev=16, quality=quality)
    render_still(Scene.from_data(mod), pb, azim=312, elev=16, quality=quality)
    opts = "\n".join(f"{k}) {v}" for k, v in OPTIONS.items())
    truth = {"floor1": 3, "floor2": 2, "changes": changed}
    return {"prompt": prompts.VISION.format(options=opts), "images": [pa, pb], "truth": truth}


def score(text, truth):
    pr_text = text or ""
    data = None
    for cand in _json_candidates(pr_text):
        try:
            data = json.loads(cand)
            break
        except Exception:  # noqa: BLE001
            continue
    if not isinstance(data, dict):
        return 0.0, "ответ не разобран"
    sw = data.get("south_windows") or {}
    try:
        f1, f2 = int(sw.get("floor1")), int(sw.get("floor2"))
    except (TypeError, ValueError):
        f1 = f2 = None
    q1 = ((f1 == truth["floor1"]) + (f2 == truth["floor2"])) / 2
    ch = data.get("changes") or []
    got = {str(c).strip().lower()[:1] for c in ch if str(c).strip()}
    want = set(truth["changes"])
    q2 = len(got & want) / len(got | want) if (got | want) else 1.0
    s = (q1 + q2) / 2
    detail = (f"окна юг: {f1}/{f2} (верно {truth['floor1']}/{truth['floor2']}); "
              f"отличия: {','.join(sorted(got)) or '—'} (верно {','.join(sorted(want))}), Жаккар {q2:.2f}")
    return s, detail


def _json_candidates(text):
    import re
    out = re.findall(r"```(?:json)?\s*\n(.*?)```", text, flags=re.S)
    i, j = text.find("{"), text.rfind("}")
    if i != -1 and j > i:
        out.append(text[i:j + 1])
    return out
