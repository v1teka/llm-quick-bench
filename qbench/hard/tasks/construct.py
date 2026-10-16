"""Набор «Конструкции»: открытые задачи без единственного ответа. Модель сдаёт конструкцию,
код проверяет её корректность и меряет качество; балл растёт плавно от простой конструкции
до лучшей известной. Потолка «решил — не решил» нет, поэтому набор различает сильные модели.
"""
from __future__ import annotations

import itertools
import math

from .common import Task, extract_json, get_item

N_GRID = 24          # поле для «Ни трёх на прямой»
TAMMES_N = 13        # точек на сфере
PACK_N = 10          # точек в квадрате
CODE_LEN, CODE_D = 10, 4

ITEMS = [
    {
        "name": f"Ни трёх на прямой {N_GRID}×{N_GRID}",
        "text": f"Выбери как можно больше узлов решётки {N_GRID}×{N_GRID} (координаты — целые 0…{N_GRID - 1}) так, "
                "чтобы никакие три выбранных узла не лежали на одной прямой (любого наклона). "
                "Ответ — список пар `[x, y]`. Если найдётся тройка на одной прямой или повтор — 0 баллов.",
        # лучшее известное — 2N (больше невозможно: в каждой строке не более двух точек)
        "best": 2 * N_GRID, "base": 0,
    },
    {
        "name": f"{TAMMES_N} точек на сфере",
        "text": f"Расставь {TAMMES_N} точек на единичной сфере так, чтобы наименьший из углов между ними (если смотреть "
                "из центра) был как можно больше (задача Таммеса). Ответ — список троек `[x, y, z]` (не меньше 9 знаков "
                "после запятой); каждая тройка нормируется на длину, нулевой вектор — 0 баллов.",
        # лучшее известное (и доказанный оптимум, Мусин и Тарасов, 2012): 57.1367031°; база — 50°
        "best": 57.1367031, "base": 50.0,
    },
    {
        "name": f"{PACK_N} точек в квадрате",
        "text": f"Расставь {PACK_N} точек в единичном квадрате [0, 1]×[0, 1] так, чтобы наименьшее из попарных "
                "расстояний было как можно больше. Ответ — список пар чисел `[x, y]` (не меньше 9 знаков после запятой). "
                "Точки вне квадрата (допуск 10⁻⁹) — 0 баллов.",
        # лучшее известное (Packomania): 0.421279543983903; база — решётка 4×3 с шагом 1/3
        "best": 0.421279543983903, "base": 1 / 3,
    },
    {
        "name": f"Код длины {CODE_LEN}, расстояние {CODE_D}",
        "text": f"Найди как можно больше двоичных слов длины {CODE_LEN}, любые два из которых различаются не менее "
                f"чем в {CODE_D} позициях (кодовое расстояние Хэмминга ≥ {CODE_D}). Ответ — список строк из 0 и 1. "
                "Если найдётся пара слов на расстоянии меньше требуемого или повтор — 0 баллов.",
        # A(10, 4) = 40 (нелинейный код); база 20, лучший линейный код [10, 5, 4] даёт 32
        "best": 40, "base": 20,
    },
]


def _clamp(x):
    return max(0.0, min(1.0, x))


def check_grid(ans):
    pts = [(int(p[0]), int(p[1])) for p in ans]
    if any(not (0 <= x < N_GRID and 0 <= y < N_GRID) for x, y in pts):
        return 0, "точка вне поля"
    if len(set(pts)) != len(pts):
        return 0, "повтор точки"
    for a, b, c in itertools.combinations(pts, 3):
        if (b[0] - a[0]) * (c[1] - a[1]) == (b[1] - a[1]) * (c[0] - a[0]):
            return 0, f"три точки на прямой: {list(a)}, {list(b)}, {list(c)}"
    return len(pts), f"{len(pts)} точек"


def check_sphere(ans):
    pts = [tuple(float(c) for c in p[:3]) for p in ans]
    if len(pts) != TAMMES_N:
        return None, f"точек {len(pts)}, нужно {TAMMES_N}"
    norm = []
    for p in pts:
        r = math.sqrt(sum(c * c for c in p))
        if r < 1e-9:
            return None, "нулевой вектор"
        norm.append(tuple(c / r for c in p))
    cos = max(sum(x * y for x, y in zip(a, b)) for a, b in itertools.combinations(norm, 2))
    ang = math.degrees(math.acos(max(-1.0, min(1.0, cos))))
    return ang, f"мин. угол {ang:.4f}°"


def check_packing(ans):
    pts = [(float(p[0]), float(p[1])) for p in ans]
    if len(pts) != PACK_N:
        return None, f"точек {len(pts)}, нужно {PACK_N}"
    if any(not (-1e-9 <= x <= 1 + 1e-9 and -1e-9 <= y <= 1 + 1e-9) for x, y in pts):
        return None, "точка вне квадрата"
    d = min(math.dist(a, b) for a, b in itertools.combinations(pts, 2))
    return d, f"мин. расстояние {d:.6f}"


def check_code(ans):
    words = [str(w).strip() for w in ans]
    if any(len(w) != CODE_LEN or set(w) - {"0", "1"} for w in words):
        return 0, "слово не из 0/1 нужной длины"
    if len(set(words)) != len(words):
        return 0, "повтор слова"
    ints = [int(w, 2) for w in words]
    for a, b in itertools.combinations(range(len(ints)), 2):
        if bin(ints[a] ^ ints[b]).count("1") < CODE_D:
            return 0, f"слова {words[a]} и {words[b]} на расстоянии {bin(ints[a] ^ ints[b]).count('1')}"
    return len(words), f"{len(words)} слов"


def item_score(k, ans):
    """-> (балл 0..1, подробности). k — номер с 0."""
    it = ITEMS[k]
    if not isinstance(ans, list) or not ans:
        return 0.0, "нет ответа"
    try:
        if k == 0:
            v, det = check_grid(ans)
            return _clamp(v / it["best"]), det + (f" (максимум {it['best']})" if v else "")
        if k == 1:
            v, det = check_sphere(ans)
            if v is None:
                return 0.0, det
            return _clamp((v - it["base"]) / (it["best"] - it["base"])), det + f" (лучшее известное {it['best']:.4f}°)"
        if k == 2:
            v, det = check_packing(ans)
            if v is None:
                return 0.0, det
            return _clamp((v - it["base"]) / (it["best"] - it["base"])), det + f" (лучшее известное {it['best']:.6f})"
        v, det = check_code(ans)
        if not v:
            return 0.0, det
        return _clamp((v - it["base"]) / (it["best"] - it["base"])), det + f" (лучшее известное {it['best']})"
    except (TypeError, ValueError, IndexError) as e:
        return 0.0, f"неверный формат: {e}"[:120]


class ConstructTask(Task):
    key = "construct"
    title = "Конструкции"
    channel = 6
    ability = "construction"

    def generate(self, seed):
        return {"seed": seed, "items": [{"name": it["name"], "best": it["best"], "base": it["base"]} for it in ITEMS]}

    def n_items(self, inst):
        return len(ITEMS)

    def prompt(self, inst, only=None):
        idx = range(len(ITEMS)) if only is None else [only]
        L = ["# Открытые конструкции", "",
             "У этих задач нет единственного ответа: сдай конструкцию, код проверит её корректность и измерит качество. "
             "Балл растёт плавно: простая корректная конструкция даёт немного, полный балл — результат на уровне лучшего "
             "известного. Некорректная конструкция — 0 баллов, поэтому лучше сдать чуть более слабую, но проверенную.", ""]
        for j, i in enumerate(idx, 1):
            L += [f"## Задача {j}. {ITEMS[i]['name']}", "", ITEMS[i]["text"], ""]
        keys = ", ".join(f'"{j}": [...]' for j in range(1, len(idx) + 1))
        L += ["## Формат ответа", f"ОДИН блок JSON без пояснений: {{{keys}}}"]
        return "\n".join(L)

    def prompt_item(self, inst, k):
        return self.prompt(inst, only=k - 1)

    def score(self, inst, text):
        ans = extract_json(text)
        items = []
        for k, it in enumerate(ITEMS):
            a = get_item(ans, k + 1) if ans else None
            s, det = item_score(k, a)
            items.append({"ok": s, "label": it["name"], "detail": det, "got": a})
        return sum(x["ok"] for x in items) / len(items), items
