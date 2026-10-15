import json

from qbench import checks as K
from qbench import reference as R
from qbench.llm import MockClient, ManualClient, NeedResponse
from qbench.runner import Run
from qbench.scene import Scene, parse_response


def test_reference_passes_all_base_checks():
    res = K.run_base(parse_response(R.as_response(R.reference_house())))
    failed = [(c.id, c.detail) for c in res if not c.passed]
    assert not failed, failed


def test_reference_garage_passes_change_checks():
    pr0 = parse_response(R.as_response(R.reference_house()))
    base, extra = K.run_change(parse_response(R.as_response(R.reference_with_garage())), pr0.scene, K.run_base(pr0))
    failed = [(c.id, c.detail) for c in base + extra if not c.passed]
    assert not failed, failed


def test_flawed_house_fails_expected_checks():
    res = {c.id: c for c in K.run_base(parse_response(R.as_response(R.flawed_house())))}
    for cid in ("S5", "S7", "S12", "S15", "S17", "S21"):
        assert not res[cid].passed, cid
    for cid in ("S1", "S3", "S8", "S11", "S14"):
        assert res[cid].passed, cid


def test_parse_lenient_json_is_not_strict():
    txt = R.as_response(R.reference_house()).replace('"objects": [', '"objects": [ // комментарий\n', 1)
    pr = parse_response(txt)
    assert pr.scene is not None and not pr.strict


def test_garbage_response():
    pr = parse_response("извините, не могу")
    assert pr.scene is None
    res = K.run_base(pr)
    assert all(not c.passed for c in res)


def test_primitives_geometry():
    sc = Scene.from_data({"objects": [
        {"id": "w", "type": "wedge", "tag": "ramp", "center": [0, 0, 0.5], "size": [2, 1, 1], "slope": "+x", "color": "#999"},
        {"id": "g", "type": "gable", "tag": "roof", "center": [0, 0, 1], "size": [2, 2, 2], "ridge": "x", "color": "#900"},
    ]})
    w, g = sc.objects
    assert abs(float(w.top_at(1.0, 0)) - 1.0) < 1e-9 and abs(float(w.top_at(-1.0, 0))) < 1e-9
    assert abs(float(g.top_at(0, 0)) - 2.0) < 1e-9 and abs(float(g.top_at(0, 1.0))) < 1e-9
    assert len(w.triangles()) == 8 and len(g.triangles()) == 8


def test_mock_run_end_to_end(tmp_path):
    s = Run(MockClient("flawed"), "Mock", str(tmp_path), do_vision=True, quality="fast", log=lambda *a: None).execute()
    assert s["abilities"]["self_fix"] == 1.0
    assert s["abilities"]["change"] == 1.0
    assert 0.5 < s["overall"] <= 1.0


def test_manual_mode_stops_and_resumes(tmp_path):
    d = tmp_path / "dialog"
    run = Run(ManualClient(str(d)), "Manual", str(tmp_path), change=False, log=lambda *a: None)
    try:
        run.execute()
        raise AssertionError("ожидался NeedResponse")
    except NeedResponse as e:
        assert e.prompt_path.endswith("01_build.prompt.md")
        with open(e.response_path, "w", encoding="utf-8") as f:
            f.write(R.as_response(R.reference_house()))
    s = Run(ManualClient(str(d)), "Manual", str(tmp_path), change=False, log=lambda *a: None).execute()
    assert s["phases"][0]["score"] > 0.99
    json.dumps(s["abilities"])
