import json

import numpy as np

from qbench.hard import config as C
from qbench.hard.dsl import build_scene, parse_house
from qbench.hard.house_checks import run_house
from qbench.hard.reference import as_response, flawed_house, reference_house
from qbench.hard.runner import TASKS, HardRun, mock_factory, task_instance


def test_reference_house_passes_everything():
    res = run_house(parse_house(as_response(reference_house())))
    failed = [(c.id, c.detail) for c in res if not c.passed]
    assert not failed, failed


def test_reference_is_detailed():
    sc = build_scene(reference_house())
    assert len(sc.objects) >= C.MIN_PRIMS
    assert sc.n_components >= C.MIN_COMPONENTS
    assert set(sc.tvs) == {"tv1", "tv2", "tv3", "tv4", "tv5"}


def test_flawed_house_fails_expected():
    res = {c.id: c for c in run_house(parse_house(as_response(flawed_house())))}
    for cid in ("H7", "H22", "H26", "H27", "H32"):
        assert not res[cid].passed, cid
    for cid in ("H2", "H4", "H12", "H16"):
        assert res[cid].passed, cid


def test_dsl_nested_rotation_and_grid():
    data = {
        "components": {
            "leg": {"objects": [{"type": "box", "tag": "furniture", "center": [0, 0, 0.5], "size": [0.1, 0.1, 1], "color": "#000"}]},
            "table": {"objects": [{"type": "box", "tag": "table", "center": [1, 0, 1], "size": [2, 1, 0.1], "color": "#fff"}],
                      "instances": [{"use": "leg", "at": [2, 0, 0]}]},
        },
        "instances": [{"use": "table", "at": [10, 10, 0], "rot": 90, "grid": {"count": [2, 3], "step": [[0, 5, 0], [5, 0, 0]]}}],
    }
    sc = build_scene(data)
    assert len(sc.objects) == 12
    tops = [o for o in sc.objects if o.tag == "table"]
    # поворот на 90°: локальный +x становится мировым +y
    assert np.allclose(tops[0].center, [10, 11, 1])
    legs = [o for o in sc.objects if o.tag == "furniture"]
    assert np.allclose(legs[0].center, [10, 12, 0.5])
    assert len(set(sc.groups)) == 6


def test_task_instances_perfect_answers():
    for key in TASKS:
        inst = task_instance(key, 2026)
        t = TASKS[key]()
        if key == "arc":
            ans = {str(i + 1): p["answer"] for i, p in enumerate(inst["puzzles"])}
        elif key == "optics":
            ans = {str(i + 1): ({"targets": b["targets"], "cells": b["cells"]} if b["kind"] == "trace" else {"mirrors": b["solution"]})
                   for i, b in enumerate(inst["boards"])}
        elif key == "nav":
            ans = {str(i + 1): l["solution"] for i, l in enumerate(inst["levels"]) if l["solution"]}
        else:
            ans = {str(i + 1): ({"stamps": p["stamps"], **p["final"]} if p["kind"] == "sim" else {"moves": p["solution"]})
                   for i, p in enumerate(inst["puzzles"])}
        s, items = t.score(inst, json.dumps(ans))
        expected = [it for it, lv in zip(items, inst.get("levels", items)) if not (isinstance(lv, dict) and lv.get("kind") == "postman")]
        assert all(it["ok"] >= 0.999 for it in expected), (key, [it["detail"] for it in items])


def test_mock_pro_run(tmp_path):
    s = HardRun(mock_factory(), "Mock", str(tmp_path), log=lambda *a: None).execute()
    assert s["abilities"]["self_fix"] is not None
    assert 0.5 < s["overall"] <= 1.0
    assert s["phases"][-1]["score"] > 0.99
