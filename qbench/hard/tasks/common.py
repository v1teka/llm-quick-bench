"""Общие помощники для задач «телеканалов»."""
from __future__ import annotations

import json
import re


def extract_json(text):
    """Последний разбираемый JSON-объект из ответа модели (или None)."""
    text = text or ""
    cands = re.findall(r"```(?:json|JSON)?\s*\n(.*?)```", text, flags=re.S)
    i, j = text.find("{"), text.rfind("}")
    if i != -1 and j > i:
        cands.append(text[i:j + 1])
    for c in reversed(sorted(cands, key=len)):
        for variant in (c, re.sub(r",\s*([}\]])", r"\1", c)):
            try:
                d = json.loads(variant)
                if isinstance(d, dict):
                    return d
            except Exception:  # noqa: BLE001
                continue
    return None


def get_item(answer, k):
    """answer['1'] / answer[1] / answer['task1'] …"""
    if not isinstance(answer, dict):
        return None
    for key in (str(k), k, f"task{k}", f"задача{k}", f"level{k}", f"board{k}", f"puzzle{k}"):
        if key in answer:
            return answer[key]
    return None


ITEM_KEYS = ("puzzles", "boards", "levels")
SINGLE_NOTE = "\n\nВ этом запросе одна задача из набора — ответь JSON с единственным ключом \"1\"."


class Task:
    """Набор однотипных задач. Каждая задача набора отдаётся модели отдельным запросом
    (prompt_item), ответы склеиваются в общий JSON (merge_items) и оцениваются score()."""

    key = ""
    title = ""
    channel = 0
    ability = ""

    def items_key(self, inst):
        return next(k for k in ITEM_KEYS if k in inst)

    def n_items(self, inst):
        return len(inst[self.items_key(inst)])

    def prompt_item(self, inst, k):
        """Тот же промпт, но только с k-й задачей набора (k с 1); в ответе она идёт под ключом "1"."""
        key = self.items_key(inst)
        return self.prompt(dict(inst, **{key: [inst[key][k - 1]]})) + SINGLE_NOTE

    @staticmethod
    def merge_items(texts):
        """Ответы на отдельные задачи -> JSON-текст общего ответа {"1": ..., "2": ...}."""
        merged = {}
        for k, t in enumerate(texts, 1):
            d = extract_json(t)
            if not d:
                continue
            v = get_item(d, 1)
            if v is None and len(d) == 1:
                v = next(iter(d.values()))
            if v is None:
                v = get_item(d, k)
            if v is not None:
                merged[str(k)] = v
        return json.dumps(merged, ensure_ascii=False)

    def generate(self, seed: int) -> dict:  # pragma: no cover - интерфейс
        raise NotImplementedError

    def prompt(self, inst: dict) -> str:  # pragma: no cover
        raise NotImplementedError

    def score(self, inst: dict, text: str):
        """-> (score 0..1, items: list[dict(ok: float, label: str, detail: str)])"""
        raise NotImplementedError
