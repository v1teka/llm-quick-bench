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


class Task:
    key = ""
    title = ""
    channel = 0
    ability = ""

    def generate(self, seed: int) -> dict:  # pragma: no cover - интерфейс
        raise NotImplementedError

    def prompt(self, inst: dict) -> str:  # pragma: no cover
        raise NotImplementedError

    def score(self, inst: dict, text: str):
        """-> (score 0..1, items: list[dict(ok: float, label: str, detail: str)])"""
        raise NotImplementedError
