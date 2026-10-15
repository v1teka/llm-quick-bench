"""Константы задания «Дом-3D». Все размеры в метрах.

Система координат: X — восток, Y — север, Z — вверх, земля z = 0.
"""

# Пятно застройки дома (наружные грани стен)
X0, X1 = 0.0, 10.0
Y0, Y1 = 0.0, 8.0

# Отметки
FFL0 = 0.45  # чистовой пол 1 этажа
FFL1 = 3.45  # чистовой пол 2 этажа
EAVE = 6.45  # верх стен 2 этажа
RIDGE = 9.0  # верх конька

# Участок; за северной границей — улица
PLOT = (-6.0, 20.0, -8.0, 14.0)
STREET_Y = (14.0, 21.0)

# Нормы
MAX_RISER = 0.19
MIN_TREAD = 0.26
MIN_STAIR_WIDTH = 0.9
HEADROOM = 2.0
MAX_RAMP_SLOPE = 1 / 12  # самый мягкий из распространённых стандартов (ADA / EN)
MIN_OVERHANG = 0.3
CHIMNEY_ABOVE_RIDGE = 0.5
MAX_OBJECTS = 600

# Гараж (фаза «изменение ТЗ»)
GX0, GX1 = 10.0, 14.0
GY0, GY1 = 3.0, 8.0
GARAGE_ROOF_TOP = 3.2

TOL = 0.15  # допуск по габаритам

TAGS = {
    "foundation", "floor_slab", "wall", "window", "door", "stair_step", "ramp",
    "landing", "railing", "balcony", "roof", "chimney", "garage_door",
    "furniture", "decor", "other",
}
TYPES = {"box", "cylinder", "sphere", "wedge", "gable"}

# Способности (оси радара). vision — только при --vision
ABILITIES = [
    ("format", "Формат"),
    ("spec", "Следование ТЗ"),
    ("spatial", "Пространство"),
    ("math", "Математика"),
    ("inference", "Неявные выводы"),
    ("physics", "Физика"),
    ("planning", "План ↔ дело"),
    ("change", "Адаптация"),
    ("self_fix", "Самокоррекция"),
    ("vision", "Зрение"),
]
ABILITY_NAMES = dict(ABILITIES)
