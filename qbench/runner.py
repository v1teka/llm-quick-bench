"""Оркестрация прогона: build → fix → change → vision → отчёт и видео."""
from __future__ import annotations

import datetime as dt
import json
import os
import re

from . import checks as K
from . import config as C
from . import prompts, vision
from .llm import NeedResponse
from .scene import Scene, parse_response


def slug(s):
    return re.sub(r"[^a-zA-Z0-9._-]+", "-", s).strip("-").lower()[:60] or "model"


class Run:
    def __init__(self, client, model_label, out_dir, fix_rounds=1, change=True, change_fix_rounds=0,
                 do_vision=False, quality="normal", log=print):
        self.client, self.model = client, model_label
        self.out = out_dir
        self.fix_rounds, self.change, self.change_fix_rounds = fix_rounds, change, change_fix_rounds
        self.do_vision, self.quality, self.log = do_vision, quality, log
        self.phases, self.usage = [], {"in": 0, "out": 0, "seconds": 0.0, "calls": 0}
        os.makedirs(os.path.join(out_dir, "responses"), exist_ok=True)

    # --------------------------------------------------------------
    def _ask(self, msgs, phase):
        text, usage, sec = self.client.chat(msgs, phase=phase)
        self.usage["in"] += usage.get("in", 0) or 0
        self.usage["out"] += usage.get("out", 0) or 0
        self.usage["seconds"] += sec or 0
        self.usage["calls"] += 1
        with open(os.path.join(self.out, "responses", f"{phase}.md"), "w", encoding="utf-8") as f:
            f.write(text)
        self.log(f"  · {phase}: {len(text)} симв., {sec:.0f} с")
        return text

    def _record(self, kind, label, step, pr, checks_, extra=None):
        ph = {"kind": kind, "label": label, "step": step, "scene": pr.scene, "error": pr.error,
              "checks": checks_, "score": K.weighted(checks_)}
        if extra:
            ph["extra_checks"] = extra
            ph["score"] = K.weighted(checks_ + extra)
        if pr.scene is not None:
            name = f"scene_{len(self.phases) + 1:02d}_{kind}.json"
            with open(os.path.join(self.out, name), "w", encoding="utf-8") as f:
                json.dump(pr.scene.to_data(), f, ensure_ascii=False, indent=1)
            ph["scene_file"] = name
        self.phases.append(ph)
        self.log(f"  → {label}: {ph['score']:.0%} (пройдено {sum(c.passed for c in checks_ + (extra or []))}"
                 f"/{len(checks_ + (extra or []))})")
        return ph

    @staticmethod
    def _feedback(pr, cs):
        if pr.scene is None:
            return prompts.FEEDBACK_UNPARSED.format(error=pr.error or "нет JSON")
        return prompts.FEEDBACK.format(issues=K.feedback_text(cs))

    # --------------------------------------------------------------
    def execute(self):
        msgs = [{"role": "system", "content": prompts.SYSTEM}, {"role": "user", "content": prompts.BUILD}]
        self.log("Фаза 1: постройка по ТЗ")
        text = self._ask(msgs, "build")
        msgs.append({"role": "assistant", "content": text})
        pr = parse_response(text)
        first = K.run_base(pr)
        self._record("build", "Попытка 1", "1", pr, first)
        cur_pr, cur = pr, first

        fixes_done = 0
        for r in range(self.fix_rounds):
            if all(c.passed for c in cur):
                break
            self.log(f"Фаза 2.{r + 1}: самокоррекция по отчёту валидатора")
            msgs.append({"role": "user", "content": self._feedback(cur_pr, cur)})
            text = self._ask(msgs, f"fix{r + 1}")
            msgs.append({"role": "assistant", "content": text})
            pr2 = parse_response(text)
            cs2 = K.run_base(pr2)
            self._record("fix", f"Исправление {r + 1}", "FIX", pr2, cs2)
            cur_pr, cur = pr2, cs2
            fixes_done += 1
        self.self_fix = K.self_fix_score(first, cur) if fixes_done else None
        self.base_final = cur

        self.change_checks = None
        if self.change:
            self.log("Фаза 3: изменение ТЗ (гараж)")
            prev_scene, prev_checks = cur_pr.scene, cur
            msgs.append({"role": "user", "content": prompts.CHANGE})
            text = self._ask(msgs, "change")
            msgs.append({"role": "assistant", "content": text})
            pr3 = parse_response(text)
            b3, g3 = K.run_change(pr3, prev_scene, prev_checks)
            self._record("change", "Изменение ТЗ: гараж", "+ГАРАЖ", pr3, b3, g3)
            self.change_checks = g3
            for r in range(self.change_fix_rounds):
                if all(c.passed for c in b3 + g3):
                    break
                msgs.append({"role": "user", "content": self._feedback(pr3, b3 + g3)})
                text = self._ask(msgs, f"change_fix{r + 1}")
                msgs.append({"role": "assistant", "content": text})
                pr3 = parse_response(text)
                b3, g3 = K.run_change(pr3, prev_scene, prev_checks)
                self._record("change_fix", f"Гараж: исправление {r + 1}", "FIX", pr3, b3, g3)

        self.vision_score, self.vision_detail = None, ""
        if self.do_vision:
            self.log("Фаза 4: зрение")
            task = vision.make_task(os.path.join(self.out, "vision"), self.quality)
            vm = [{"role": "system", "content": prompts.VISION_SYSTEM},
                  {"role": "user", "content": task["prompt"], "images": task["images"]}]
            text = self._ask(vm, "vision")
            self.vision_score, self.vision_detail = vision.score(text, task["truth"])
            self.log(f"  → зрение: {self.vision_score:.0%} ({self.vision_detail})")
        return self.summary()

    # --------------------------------------------------------------
    def summary(self):
        first = self.phases[0]["checks"]
        ab = K.ability_scores(first, self.change_checks, self.self_fix, self.vision_score)
        total = K.overall(ab)
        by_kind = {}
        for p in self.phases:
            by_kind.setdefault(p["kind"], p)
        last_fix = [p for p in self.phases if p["kind"] == "fix"]
        stats = [
            f"Попытка 1: {self.phases[0]['score']:.0%}"
            + (f" · после исправлений: {last_fix[-1]['score']:.0%}" if last_fix else "")
            + (f" · с гаражом: {by_kind['change']['score']:.0%}" if "change" in by_kind else ""),
            f"Запросов: {self.usage['calls']} · токены вх/вых: {self.usage['in']}/{self.usage['out']}"
            + (f" · {self.usage['seconds']:.0f} с" if self.usage["seconds"] >= 1 else ""),
            f"{dt.date.today().isoformat()} · режим: {getattr(self.client, 'kind', 'api')}",
        ]
        groups = [("Попытка 1", first)]
        if last_fix:
            groups.append((f"После исправлений ({len(last_fix)})", [c for c in last_fix[-1]["checks"] if not c.id.startswith("F")]))
        if "change" in by_kind:
            groups.append(("Изменение ТЗ", by_kind["change"].get("extra_checks", [])))
        n_checks = len(K.BASE) + (len(K.CHANGE) if self.change else 0) + (1 if self.do_vision else 0)
        return {
            "model": self.model, "date": dt.datetime.now().strftime("%Y-%m-%d %H:%M"),
            "phases": self.phases, "abilities": ab, "overall": total, "self_fix": self.self_fix,
            "vision": {"score": self.vision_score, "detail": self.vision_detail},
            "usage": self.usage, "stats_lines": stats, "checks_groups": groups, "n_checks": n_checks,
            "mode": getattr(self.client, "kind", "api"),
        }


# ----------------------------------------------------------------------
# сохранение / загрузка
# ----------------------------------------------------------------------

def save_run(summary, out_dir):
    data = dict(summary)
    data["phases"] = [{k: v for k, v in p.items() if k not in ("scene", "checks", "extra_checks")}
                      | {"checks": [c.to_dict() for c in p["checks"]],
                         "extra_checks": [c.to_dict() for c in p.get("extra_checks", [])]}
                      for p in summary["phases"]]
    data["checks_groups"] = [(t, [c.id + "@" + c.phase for c in cs]) for t, cs in summary["checks_groups"]]
    with open(os.path.join(out_dir, "run.json"), "w", encoding="utf-8") as f:
        json.dump(data, f, ensure_ascii=False, indent=1)
    with open(os.path.join(out_dir, "report.md"), "w", encoding="utf-8") as f:
        f.write(report_md(summary))


def load_run(out_dir):
    with open(os.path.join(out_dir, "run.json"), encoding="utf-8") as f:
        data = json.load(f)
    for p in data["phases"]:
        p["checks"] = [K.CheckResult(**{**c, "abilities": tuple(c["abilities"])}) for c in p["checks"]]
        p["extra_checks"] = [K.CheckResult(**{**c, "abilities": tuple(c["abilities"])}) for c in p.get("extra_checks", [])]
        p["scene"] = Scene.load(os.path.join(out_dir, p["scene_file"])) if p.get("scene_file") else None
    groups = []
    for (title, ids), ph in zip(data["checks_groups"], _groups_source(data["phases"])):
        groups.append((title, ph))
    data["checks_groups"] = groups
    return data


def _groups_source(phases):
    first = phases[0]["checks"]
    fixes = [p for p in phases if p["kind"] == "fix"]
    change = next((p for p in phases if p["kind"] == "change"), None)
    out = [first]
    if fixes:
        out.append([c for c in fixes[-1]["checks"] if not c.id.startswith("F")])
    if change:
        out.append(change["extra_checks"])
    return out


def report_md(s):
    L = [f"# llm-quick-bench · Дом-3D — {s['model']}", "", f"Дата: {s['date']} · режим: {s['mode']}", "",
         f"## Итог: **{s['overall']:.0%}**", "", "| Способность | Оценка |", "|---|---|"]
    for key, name in C.ABILITIES:
        v = s["abilities"].get(key)
        L.append(f"| {name} | {'—' if v is None else f'{v:.0%}'} |")
    L += ["", *[f"- {x}" for x in s["stats_lines"]], ""]
    if s["vision"]["score"] is not None:
        L += [f"**Зрение:** {s['vision']['score']:.0%} — {s['vision']['detail']}", ""]
    for p in s["phases"]:
        L += [f"## {p['label']} — {p['score']:.0%}", ""]
        if p["scene"] is None:
            L += [f"Сцена не получена: {p['error']}", ""]
        elif p["scene"].plan.get("notes"):
            L += [f"> {p['scene'].plan.get('notes')}", ""]
        L += ["| | Проверка | Оценка | Детали |", "|---|---|---|---|"]
        for c in p["checks"] + p.get("extra_checks", []):
            L.append(f"| {'✅' if c.passed else '❌'} | {c.id} {c.name} | {c.score:.0%} | {c.detail.replace('|', '/')} |")
        L.append("")
    return "\n".join(L)


def update_leaderboard(results_dir):
    rows = []
    for root, _, files in os.walk(results_dir):
        if "run.json" in files:
            try:
                with open(os.path.join(root, "run.json"), encoding="utf-8") as f:
                    d = json.load(f)
            except Exception:  # noqa: BLE001
                continue
            rows.append((d["overall"], d, os.path.relpath(root, results_dir)))
    rows.sort(key=lambda r: -r[0])
    from .hard import config as HC
    L = ["# Лидерборд llm-quick-bench · Дом-3D", ""]
    from .hard.runner import PROTOCOL, PROTOCOL_NAMES, protocol_of
    sections = (
        (f"Pro: детальный дом + наборы задач (протокол {PROTOCOL}: {PROTOCOL_NAMES[PROTOCOL]})", HC.ABILITIES,
         lambda d: d.get("track") == "pro" and protocol_of(d) == PROTOCOL),
        ("Pro: прежние версии протокола", HC.ABILITIES, lambda d: d.get("track") == "pro" and protocol_of(d) != PROTOCOL),
        ("Базовый трек", C.ABILITIES, lambda d: d.get("track") != "pro"))
    for title, abil, sel in sections:
        part = [r for r in rows if sel(r[1])]
        if not part:
            continue
        old = title.startswith("Pro: прежние")
        L += [f"## {title}", "", "| # | Модель | Итог | " + ("Протокол | " if old else "") + " | ".join(n for _, n in abil)
              + " | Дата | Видео |", "|" + "---|" * (len(abil) + 5 + old)]
        for i, (tot, d, rel) in enumerate(part, 1):
            vals = " | ".join("—" if d["abilities"].get(k) is None else f"{d['abilities'][k]:.0%}" for k, _ in abil)
            video = f"[gif]({rel}/video.gif)" if os.path.exists(os.path.join(results_dir, rel, "video.gif")) else "—"
            proto = f"{protocol_of(d)} | " if old else ""
            L.append(f"| {i} | {d['model']} | **{tot:.0%}** | {proto}{vals} | {d['date'][:10]} | {video} |")
        L.append("")
    os.makedirs(results_dir, exist_ok=True)
    path = os.path.join(results_dir, "LEADERBOARD.md")
    with open(path, "w", encoding="utf-8") as f:
        f.write("\n".join(L).rstrip() + "\n")
    return path


__all__ = ["Run", "NeedResponse", "save_run", "load_run", "report_md", "update_leaderboard", "slug"]
