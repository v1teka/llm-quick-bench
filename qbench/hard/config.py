"""Константы задания «Дом-3D Pro». Метры; X — восток, Y — север, Z — вверх."""

X0, X1 = 0.0, 13.0
Y0, Y1 = 0.0, 10.0
FFL0, FFL1 = 0.45, 3.45
EAVE, RIDGE = 6.45, 9.6
SLAB1_BOTTOM_MAX = 3.25

PLOT = (-8.0, 22.0, -8.0, 16.0)
STREET_Y = (16.0, 23.0)

MAX_RISER = 0.18
MIN_TREAD = 0.27
MIN_STAIR_WIDTH = 1.0
HEADROOM = 2.1
MAX_BALUSTER_GAP = 0.10
HANDRAIL = (0.85, 1.0)
MAX_RAMP_SLOPE = 1 / 12
MIN_RAMP_WIDTH = 0.9
MIN_OVERHANG = 0.4
CHIMNEY_ABOVE_RIDGE = 0.5
BALCONY_DEPTH, BALCONY_WIDTH = 1.4, 3.0
BALCONY_RAIL = (1.0, 1.15)

MIN_PRIMS = 10000
MIN_COMPONENTS = 15
MIN_TILES = 1000

ROOMS_REQUIRED = {
    # тип: (этаж, мин. площадь)
    "hall": [(0, 8.0), (1, 6.0)],
    "living": [(0, 20.0)],
    "kitchen": [(0, 12.0)],
    "bedroom_mom": [(0, 12.0)],
    "bath_access": [(0, 5.0)],
    "bedroom_kids": [(1, 10.0)],
    "bedroom_master": [(1, 14.0)],
    "bath": [(1, 4.0)],
}
HABITABLE = {"living", "kitchen", "bedroom_mom", "bedroom_kids", "bedroom_master", "study"}

TVS = {
    # id: (комната, место просмотра (тег мебели), мин. диагональ)
    "tv1": ("living", "sofa", 1.2),
    "tv2": ("kitchen", "table", 1.0),
    "tv3": ("bedroom_mom", "bed", 1.0),
    "tv4": ("bedroom_kids", "bed", 1.0),
    "tv5": ("bedroom_master", "bed", 1.0),
}
TV_DIAG_MAX = 1.65

ABILITIES = [
    ("format", "Формат и DSL"),
    ("spec", "Следование ТЗ"),
    ("spatial", "Пространство"),
    ("inference", "Неявные выводы"),
    ("physics", "Физика"),
    ("detail", "Детализация"),
    ("planning", "План ↔ дело"),
    ("self_fix", "Самокоррекция"),
    ("logic", "Логика (ARC)"),
    ("simulation", "Симуляция"),
    ("plan_search", "Поиск плана"),
    ("rotation", "Вращение 3D"),
]
ABILITY_NAMES = dict(ABILITIES)
