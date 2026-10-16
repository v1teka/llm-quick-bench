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
        if key in ("code", "code1", "construct"):  # свои тесты ниже
            continue
        inst = task_instance(key, 2026)
        t = TASKS[key]()
        if key == "arc":
            ans = {str(i + 1): p["answer"] for i, p in enumerate(inst["puzzles"])}
        elif key == "optics":
            ans = {str(i + 1): ({"targets": b["targets"], "cells": b["cells"]} if b["kind"] == "trace" else {"mirrors": b["solution"]})
                   for i, b in enumerate(inst["boards"])}
        elif key == "nav":
            ans = {str(i + 1): l["solution"] for i, l in enumerate(inst["levels"]) if l["solution"]}
        elif key.startswith("matryoshka"):
            ch = inst["chain"]
            text = "\n".join(json.dumps({k: ch[k]}) for k in ("L5", "L4", "L3", "L2", "L1"))
            s, items = t.score(inst, text + "\n" + json.dumps({"L0": {"moves": ch["L0"]}}))
            assert s > 0.999 and all(it["ok"] >= 0.999 for it in items), (key, [it["detail"] for it in items])
            continue
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


def test_merge_doc_replace_remove_reset():
    from qbench.hard.dsl import merge_doc
    a = {"objects": [{"id": "x", "v": 1}, {"id": "y"}], "instances": [{"use": "c", "id": "i1"}], "plan": {"a": 1}}
    b = merge_doc(a, {"objects": [{"id": "x", "v": 2}, {"id": "z"}], "remove": ["y", "i1"], "plan": {"b": 2}})
    assert [o["id"] for o in b["objects"]] == ["x", "z"] and b["objects"][0]["v"] == 2
    assert b["instances"] == [] and b["plan"] == {"a": 1, "b": 2}
    assert merge_doc(b, {"reset": True, "objects": [{"id": "q"}]})["objects"] == [{"id": "q"}]


def test_house_in_parts_equals_whole():
    from qbench.hard.dsl import parse_house_parts
    d = reference_house()
    objs, inst = d.get("objects", []), d.get("instances", [])
    p1 = {k: v for k, v in d.items() if k not in ("objects", "instances")}
    p1.update(objects=objs[: len(objs) // 2], instances=inst[: len(inst) // 2])
    p2 = {"objects": objs[len(objs) // 2:], "instances": inst[len(inst) // 2:]}
    pr, _ = parse_house_parts([json.dumps(p1), json.dumps(p2)])
    assert len(pr.scene.objects) == len(build_scene(d).objects)
    assert all(c.passed for c in run_house(pr))


def test_puzzle_modes(tmp_path):
    s = HardRun(mock_factory(), "Mock", str(tmp_path / "a"), log=lambda *a: None, puzzles="sets", item_diag=True).execute()
    single = {k: t for k, t in s["tasks"].items() if t.get("mode") != "layers"}  # матрёшка — диалог по слоям
    assert single and all(t["answered"] and t.get("items_res") for t in single.values())
    s2 = HardRun(mock_factory(), "Mock", str(tmp_path / "b"), log=lambda *a: None, puzzles="items").execute()
    assert {k: round(t["res"]["score"], 3) for k, t in s2["tasks"].items() if k in single} == \
           {k: round(t["items_res"]["score"], 3) for k, t in single.items()}


def test_matryoshka_v2_layers():
    from qbench.hard.tasks import olymp2026 as O
    # слой 5: структура последовательности совпадает с прямым перебором
    pre, _ = O.p6_prefix(1925, 4000)
    assert O.p6_count(1925, pre[-1]) == len(pre) and O.p6_term(1925, 3999) == pre[3998]
    # слой 4: критерий коллапса и точный счёт ударов
    assert [A for A in range(100, 500) if (O.sim4_float(A, 0.8, 200000) is None) != O.collapse4(A)] == []
    assert O.solve4(250) == 105 and O.solve4(1000) == 0
    # слой 3: c_1 = 2/3, c_2 = 4/7
    assert O.solve3(1) == 5 and O.solve3(2) == 11
    # слой 2: порядок тасовки = перебор циклов
    assert all(O.monge_order(D) == O.monge_order_brute(D) for D in range(2, 400, 2))
    # слой 1: θ = 180°/n ⇔ k | m
    assert O.solve1(358) == 23  # m = 360
    # оценка «по слоям»: ошибка в слое 4 не обнуляет внешние слои
    from qbench.hard.runner import TASKS, task_instance
    inst = task_instance("matryoshka", 2026)
    t = TASKS["matryoshka"]()
    ch = dict(inst["chain"])
    wrong4 = ch["L4"] + 1
    a3 = O.solve3(O.param3(wrong4)); a2 = O.solve2(a3); a1 = O.solve1(a2)
    from qbench.hard.tasks.matryoshka import solve0
    _, _, mv = solve0(inst["board"], a1)
    text = "\n".join(json.dumps(x) for x in ({"L5": ch["L5"]}, {"L4": wrong4}, {"L3": a3}, {"L2": a2}, {"L1": a1},
                                               {"L0": {"moves": mv}}))
    s, items = t.score(inst, text)
    summ = items[0]["summary"]
    assert summ["chain"] == 1 and abs(summ["by_layer"] - 5 / 6) < 1e-9


def test_code_reference_matches_brute_and_scores_full():
    import random
    from qbench.hard.tasks import code as K
    for k in range(4):
        inp, want = K.SAMPLES[k]
        assert K.same(K.BRUTE[k](inp), want)
        rng = random.Random(k)
        for _ in range(15):
            inp = K.GENS[k](rng, "small")
            st, out = K.run_code(K.REF[k], inp, 30)
            assert st == "OK" and K.same(out, K.BRUTE[k](inp)), (k, inp[:120])
    t = TASKS["code1"]()
    inst = task_instance("code1", 2026)
    text = "\n\n".join(f"```python\n# задача {k}\n{c}```" for k, c in enumerate(K.REF, 1))
    s, items = t.score(inst, text)
    assert s == 1.0, [it["detail"] for it in items]
    naive = "```python\n# задача 3\nn=int(input());from math import gcd\n" \
            "print(sum(gcd(i,j)**2 for i in range(1,n+1) for j in range(1,n+1))%998244353)\n```"
    s, items = t.score(inst, naive)
    assert 0 < items[2]["ok"] < 1 and "TL" in items[2]["detail"] and items[0]["ok"] == 0


def test_constructions_scoring():
    from qbench.hard.runner import mock_constructions
    from qbench.hard.tasks.construct import item_score
    m = mock_constructions()
    assert abs(item_score(0, m["1"])[0] - 23 / 48) < 1e-9
    assert 0 < item_score(1, m["2"])[0] < 0.2                       # спираль Фибоначчи: 50.7°
    assert item_score(1, m["2"][:-1])[0] == 0                       # 12 точек вместо 13
    assert item_score(2, m["3"])[0] == 0 and abs(item_score(3, m["4"])[0] - 0.6) < 1e-9
    assert item_score(0, m["1"] + [[23, 23]])[0] == 0              # (0,0), (1,1), (23,23) на одной прямой


def test_opt_evaluators_and_reference():
    from qbench.hard.tasks import opt as O
    insts = O.make_instances(2026)
    tsp, col, strip = insts[0][0], insts[1][0], insts[2][0]
    n = 2000
    assert O.eval_tsp(tsp, " ".join(map(str, range(n))))[0] > 0
    assert O.eval_tsp(tsp, " ".join(map(str, [0] * n)))[0] is None
    assert O.eval_color(col, " ".join(map(str, range(250))))[0] == 250
    assert O.eval_color(col, " ".join(["0"] * 250))[0] is None
    d = O._ints(strip)
    assert O.eval_strip(strip, "\n".join("0 0" for _ in range(d[0])))[0] is None    # все в углу — наложение
    ref = O.load_ref(2026)
    assert ref and all(len(ref[p["key"]]) == len(p["instances"]) for p in O.PROBLEMS)
    t = TASKS["code"]()
    inst = task_instance("code", 2026)
    assert all(pr["ref"] and all(b > r for b, r in zip(pr["base"], pr["ref"])) for pr in inst["problems"])
