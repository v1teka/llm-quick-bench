"""Прогон «Дом-3D Pro»: дом (постройка + самокоррекция) и 4 задачи для телевизоров.

Каждая задача — отдельный диалог. В ручном режиме недостающие ответы не останавливают
остальные диалоги: все ожидающие промпты пишутся сразу (их можно решать параллельно).
"""
from __future__ import annotations

import datetime as dt
import json
import os

import numpy as np

from ..checks import CheckResult, weighted, self_fix_score
from ..llm import ManualClient, NeedResponse
from . import config as C
from . import prompts as P
from .dsl import parse_house_parts
from .house_checks import run_house, feedback, CHECKS
from .tasks.arc import ArcTask
from .tasks.cube import CubeTaskHarder
from .tasks.nav import NavTaskHard
from .tasks.optics import OpticsTaskHarder

SEED = 2026
TASKS = {
    "arc": lambda: ArcTask(difficulty=3),
    "optics": lambda: OpticsTaskHarder(difficulty=3),
    "nav": lambda: NavTaskHard(difficulty=3),
    "cube": lambda: CubeTaskHarder(difficulty=3),
}
DATA = os.path.join(os.path.dirname(__file__), "data")


def task_instance(key, seed):
    """Экземпляры задач фиксированы и кэшируются в qbench/hard/data (генерация сокобана долгая)."""
    path = os.path.join(DATA, f"{key}_{seed}.json")
    if os.path.exists(path):
        with open(path, encoding="utf-8") as f:
            return json.load(f)
    inst = TASKS[key]().generate(seed)
    os.makedirs(DATA, exist_ok=True)
    with open(path, "w", encoding="utf-8") as f:
        json.dump(inst, f, ensure_ascii=False)
    return inst
TASK_ABILITY = {"arc": "logic", "optics": "simulation", "nav": "plan_search", "cube": "rotation"}
TASK_SYSTEM = "Ты внимательно решаешь задачи и отвечаешь строго в требуемом формате. " + P.BUDGET


class Pending(Exception):
    pass


class HardRun:
    def __init__(self, client_factory, model_label, out_dir, fix_rounds=1, tasks=("arc", "optics", "nav", "cube"),
                 seed=SEED, log=print, puzzles="sets", item_diag=False):
        self.factory = client_factory  # conv_name -> client
        self.model = model_label
        self.out = out_dir
        self.fix_rounds = fix_rounds
        self.task_keys = list(tasks)
        self.seed = seed
        self.log = log
        self.pending = []
        # "sets" — набор головоломок одним запросом (официальный счёт); "items" — по головоломке на запрос
        self.puzzles = puzzles
        # при puzzles="sets" дополнительно спросить каждую головоломку отдельно (диагностика, в счёт не идёт)
        self.item_diag = bool(item_diag) and puzzles == "sets"
        self.last_usage = {}
        self.usage = {"in": 0, "out": 0, "seconds": 0.0, "calls": 0, "overruns": 0}
        os.makedirs(os.path.join(out_dir, "responses"), exist_ok=True)

    def _ask(self, client, msgs, phase):
        text, usage, sec = client.chat(msgs, phase=phase)
        self.last_usage = usage
        self.usage["in"] += usage.get("in", 0) or 0
        self.usage["out"] += usage.get("out", 0) or 0
        self.usage["seconds"] += sec or 0
        self.usage["calls"] += 1
        self.usage["overruns"] += usage.get("overruns", 0) or 0
        with open(os.path.join(self.out, "responses", f"{phase}.md"), "w", encoding="utf-8") as f:
            f.write(text)
        self.log(f"  · {phase}: {len(text)} симв., {sec:.0f} с")
        return text

    # ------------------------------------------------------------------
    def run_house(self):
        """Постройка в P.N_PARTS ответах (каждый — свой лимит вывода), затем раунды исправления патчами."""
        client = self.factory("house")
        msgs = [{"role": "system", "content": P.SYSTEM}, {"role": "user", "content": P.BUILD}]
        phases, texts = [], []
        for k in range(1, P.N_PARTS + 1):
            if k > 1:
                msgs.append({"role": "user", "content": P.PARTS[k]})
            try:
                text = self._ask(client, msgs, f"house_part{k}")
            except NeedResponse as e:
                self.pending.append(e)
                return phases
            msgs.append({"role": "assistant", "content": text})
            texts.append(text)
        pr, doc = parse_house_parts(texts)
        self._save_doc(doc, "build")
        cs = run_house(pr)
        phases.append(self._phase("build", "Попытка 1", pr, cs))
        for r in range(self.fix_rounds):
            if all(c.passed for c in cs):
                break
            fb = P.FEEDBACK.format(issues=feedback(cs)) if pr.scene is not None else P.FEEDBACK_UNPARSED.format(error=pr.error)
            msgs.append({"role": "user", "content": fb})
            try:
                text = self._ask(client, msgs, f"house_fix{r + 1}")
            except NeedResponse as e:
                self.pending.append(e)
                break
            msgs.append({"role": "assistant", "content": text})
            texts.append(text)
            pr, doc = parse_house_parts(texts)
            self._save_doc(doc, f"fix{r + 1}")
            cs = run_house(pr)
            phases.append(self._phase("fix", f"Исправление {r + 1}", pr, cs))
        return phases

    def _save_doc(self, doc, name):
        with open(os.path.join(self.out, f"scene_{name}.json"), "w", encoding="utf-8") as f:
            json.dump(doc, f, ensure_ascii=False)

    def _phase(self, kind, label, pr, cs):
        ph = {"kind": kind, "label": label, "scene": pr.scene, "error": pr.error, "checks": cs, "score": weighted(cs)}
        if pr.scene is not None:
            self.log(f"  → {label}: {ph['score']:.0%} ({sum(c.passed for c in cs)}/{len(cs)}), примитивов {len(pr.scene.objects)}")
        else:
            self.log(f"  → {label}: сцена не разобрана ({pr.error})")
        return ph

    def _ask_items(self, key, task, inst):
        """Каждая головоломка набора — отдельный запрос; ответы склеиваются и оцениваются вместе."""
        texts, waiting, overruns = [], False, 0
        for k in range(1, task.n_items(inst) + 1):
            client = self.factory(f"{key}/{k}")
            msgs = [{"role": "system", "content": TASK_SYSTEM}, {"role": "user", "content": task.prompt_item(inst, k)}]
            try:
                texts.append(self._ask(client, msgs, f"task_{key}_{k}"))
                overruns += self.last_usage.get("overruns", 0) or 0
            except NeedResponse as e:
                self.pending.append(e)
                texts.append("")
                waiting = True
        if waiting:
            return None
        text = task.merge_items(texts)
        with open(os.path.join(self.out, "responses", f"task_{key}_items.json"), "w", encoding="utf-8") as f:
            f.write(text)
        score, items = task.score(inst, text)
        return {"score": score, "items": items, "overruns": overruns}

    def run_task(self, key):
        task = TASKS[key]()
        inst = task_instance(key, self.seed)
        with open(os.path.join(self.out, f"task_{key}.json"), "w", encoding="utf-8") as f:
            json.dump(inst, f, ensure_ascii=False)
        out = {"inst": inst, "title": task.title, "mode": self.puzzles}
        if self.puzzles == "items":
            r = self._ask_items(key, task, inst)
            if r is None:
                return None
            out.update(res={"score": r["score"], "items": r["items"]}, answered=True, overruns=r["overruns"])
        else:
            msgs = [{"role": "system", "content": TASK_SYSTEM}, {"role": "user", "content": task.prompt(inst)}]
            text, ov = None, 0
            try:
                text = self._ask(self.factory(key), msgs, f"task_{key}")
                ov = self.last_usage.get("overruns", 0) or 0
            except NeedResponse as e:
                self.pending.append(e)
            diag = self._ask_items(key, task, inst) if self.item_diag else None
            if text is None or (self.item_diag and diag is None):
                return None
            score, items = task.score(inst, text)
            answered = bool(text.strip())
            if not answered:
                for it in items:
                    it["detail"] = f"нет ответа: лимит вывода исчерпан рассуждениями ({ov} из {ov} попыток)" if ov else "пустой ответ"
            out.update(res={"score": score, "items": items}, answered=answered, overruns=ov)
            if diag is not None:
                out["items_res"] = diag
        self.log(f"  → {task.title}: {out['res']['score']:.0%}" + ("" if out["answered"] else " (нет ответа)")
                 + (f"; по одной: {out['items_res']['score']:.0%}" if out.get("items_res") else ""))
        return out

    # ------------------------------------------------------------------
    def execute(self):
        self.log("Дом: постройка и самокоррекция")
        phases = self.run_house()
        tasks = {}
        for k in self.task_keys:
            self.log(f"Задача {k}")
            r = self.run_task(k)
            if r is not None:
                tasks[k] = r
        if self.pending:
            raise Pending(self.pending)
        return self.summary(phases, tasks)

    def summary(self, phases, tasks):
        first = phases[0]["checks"] if phases else []
        final = phases[-1]["checks"] if phases else []
        ab = {}
        for key, _ in C.ABILITIES:
            if key == "self_fix":
                ab[key] = self_fix_score(first, final) if len(phases) > 1 else None
            elif key in TASK_ABILITY.values():
                k = next(t for t, a in TASK_ABILITY.items() if a == key)
                ab[key] = tasks[k]["res"]["score"] if k in tasks else None
            else:
                cs = [c for c in first if key in c.abilities]
                ab[key] = weighted(cs) if cs else None
        vals = [v for v in ab.values() if v is not None]
        overall = float(np.mean(vals)) if vals else 0.0
        groups = []
        if phases:
            groups.append(("Дом: попытка 1", first))
            if len(phases) > 1:
                groups.append(("Дом: после исправления", [c for c in final if not c.passed][:12] or final[:6]))
        for k, t in tasks.items():
            groups.append((f"{t['title']}: {t['res']['score']:.0%}" + ("" if t.get("answered", True) else " — нет ответа"), [
                CheckResult(f"{k[:1].upper()}{i + 1}", it["label"], (TASK_ABILITY[k],), 1.0, it["ok"], it["ok"] >= 0.999, it["detail"], k)
                for i, it in enumerate(t["res"]["items"])]))
        hs0 = phases[0]["score"] if phases else None
        hs1 = phases[-1]["score"] if phases else None
        stats = [
            f"Дом: попытка 1 — {hs0:.0%}" + (f", после исправления — {hs1:.0%}" if len(phases) > 1 else "") if phases else "Дом: нет",
            ("Наборы задач (каждый — одним ответом): " if self.puzzles == "sets" else "Задачи (по одной на запрос): ")
            + ", ".join(f"{t['title']} {t['res']['score']:.0%}" + ("" if t.get("answered", True) else " (нет ответа)")
                        for t in tasks.values()),
            *(["По одной головоломке на запрос (диагностика): "
               + ", ".join(f"{t['title']} {t['items_res']['score']:.0%}" for t in tasks.values() if t.get("items_res"))]
              if any(t.get("items_res") for t in tasks.values()) else []),
            f"Запросов: {self.usage['calls']}"
            + (f" · токенов вывода: {self.usage['out'] / 1000:.0f} тыс." if self.usage["out"] else "")
            + (f" · превышений лимита: {self.usage['overruns']}" if self.usage["overruns"] else "")
            + f" · {dt.date.today().isoformat()}",
        ]
        return {"model": self.model, "date": dt.datetime.now().strftime("%Y-%m-%d %H:%M"), "phases": phases,
                "tasks": tasks, "abilities": ab, "overall": overall, "house_score": hs1, "house_first": hs0,
                "checks_groups": groups, "n_house_checks": len(CHECKS), "stats_lines": stats, "usage": self.usage}


def manual_factory(folder):
    clients = {}

    def make(name):
        if name not in clients:
            clients[name] = ManualClient(os.path.join(folder, "dialog", name))
        return clients[name]
    return make


def save_summary(s, out_dir):
    data = {k: v for k, v in s.items() if k not in ("phases", "tasks", "checks_groups")}
    data["track"] = "pro"
    data["phases"] = []
    for i, p in enumerate(s["phases"]):
        d = {k: v for k, v in p.items() if k not in ("scene", "checks")}
        d["checks"] = [c.to_dict() for c in p["checks"]]
        d["n_prims"] = len(p["scene"].objects) if p["scene"] is not None else 0
        data["phases"].append(d)
    data["tasks"] = {k: {"title": t["title"], "mode": t.get("mode"), "answered": t.get("answered", True),
                         "overruns": t.get("overruns", 0), "score": t["res"]["score"], "items": t["res"]["items"],
                         "items_diag": t.get("items_res")} for k, t in s["tasks"].items()}
    with open(os.path.join(out_dir, "run.json"), "w", encoding="utf-8") as f:
        json.dump(data, f, ensure_ascii=False, indent=1, default=str)
    with open(os.path.join(out_dir, "report.md"), "w", encoding="utf-8") as f:
        f.write(report(s))


def report(s):
    L = [f"# llm-quick-bench · Дом-3D Pro — {s['model']}", "", f"Дата: {s['date']}", "",
         f"## Итог: **{s['overall']:.0%}**", "", "| Способность | Оценка |", "|---|---|"]
    for key, name in C.ABILITIES:
        v = s["abilities"].get(key)
        L.append(f"| {name} | {'—' if v is None else f'{v:.0%}'} |")
    L += ["", *[f"- {x}" for x in s["stats_lines"]], ""]
    for p in s["phases"]:
        L += [f"## Дом: {p['label']} — {p['score']:.0%}", ""]
        if p["scene"] is None:
            L += [f"Сцена не получена: {p['error']}", ""]
        else:
            L += [f"Примитивов после развёртки: {len(p['scene'].objects)}", ""]
            if p["scene"].plan.get("notes"):
                L += [f"> {p['scene'].plan.get('notes')}", ""]
        L += ["| | Проверка | Оценка | Детали |", "|---|---|---|---|"]
        for c in p["checks"]:
            L.append(f"| {'✅' if c.passed else '❌'} | {c.id} {c.name} | {c.score:.0%} | {c.detail.replace('|', '/')} |")
        L.append("")
    for k, t in s["tasks"].items():
        mode = "набор одним ответом" if t.get("mode") == "sets" else "по одной головоломке на запрос"
        L += [f"## {t['title']} — {t['res']['score']:.0%} ({mode})", ""]
        if not t.get("answered", True):
            L += [f"Ответа нет: все попытки ({t.get('overruns', 0)}) исчерпали лимит вывода рассуждениями.", ""]
        L += ["| | Пункт | Оценка | Детали |", "|---|---|---|---|"]
        for it in t["res"]["items"]:
            L.append(f"| {'✅' if it['ok'] >= 0.999 else ('🟡' if it['ok'] > 0 else '❌')} | {it['label']} | {it['ok']:.0%} | {it['detail'].replace('|', '/')} |")
        L.append("")
        if t.get("items_res"):
            d = t["items_res"]
            L += [f"Диагностика — каждая головоломка отдельным запросом: **{d['score']:.0%}** (в счёт не идёт)", "",
                  "| | Пункт | Оценка | Детали |", "|---|---|---|---|"]
            for it in d["items"]:
                L.append(f"| {'✅' if it['ok'] >= 0.999 else ('🟡' if it['ok'] > 0 else '❌')} | {it['label']} | {it['ok']:.0%} | {it['detail'].replace('|', '/')} |")
            L.append("")
    return "\n".join(L)


class MockHard:
    """Детерминированная «модель» для демо и тестов Pro-набора."""

    kind = "mock"

    def __init__(self, conv, seed=SEED):
        self.conv = conv
        self.seed = seed

    def chat(self, messages, phase=""):
        from .reference import reference_house, flawed_house, as_response
        if phase == "house_part1":
            return as_response(flawed_house()), {"in": 0, "out": 0}, 0.0
        if phase.startswith("house_part"):
            return "{}", {"in": 0, "out": 0}, 0.0
        if phase.startswith("house_fix"):
            return as_response(dict(reference_house(), reset=True)), {"in": 0, "out": 0}, 0.0
        key, _, item = phase.replace("task_", "").partition("_")
        inst = task_instance(key, self.seed)
        if key == "arc":
            ans = {str(i + 1): p["answer"] for i, p in enumerate(inst["puzzles"])}
            ans["2"] = ["0"]  # одна ошибка для наглядности
        elif key == "optics":
            ans = {str(i + 1): ({"targets": b["targets"], "cells": b["cells"]} if b["kind"] == "trace" else {"mirrors": b["solution"]})
                   for i, b in enumerate(inst["boards"])}
        elif key == "nav":
            ans = {str(i + 1): (l["solution"] or "") for i, l in enumerate(inst["levels"])}
        else:
            ans = {str(i + 1): ({"stamps": p["stamps"], **p["final"]} if p["kind"] == "sim" else {"moves": p["solution"]})
                   for i, p in enumerate(inst["puzzles"])}
        if not item:  # набор одним ответом
            return json.dumps(ans, ensure_ascii=False), {"in": 0, "out": 0}, 0.0
        return json.dumps({"1": ans.get(item, "")}, ensure_ascii=False), {"in": 0, "out": 0}, 0.0


def mock_factory():
    return lambda name: MockHard(name)


def api_factory(model, **kw):
    from ..llm import OpenAIClient
    client = OpenAIClient(model, **kw)
    return lambda name: client
