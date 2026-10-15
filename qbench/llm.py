"""Клиенты моделей: OpenAI-совместимый API, ручной режим через файлы, mock."""
from __future__ import annotations

import base64
import json
import mimetypes
import os
import time

from . import reference as R


class NeedResponse(Exception):
    """Ручной режим: нужен ответ модели на сохранённый промпт."""

    def __init__(self, prompt_path, response_path):
        super().__init__(f"нужен ответ: {response_path}")
        self.prompt_path, self.response_path = prompt_path, response_path


def _data_uri(path):
    mt = mimetypes.guess_type(path)[0] or "image/png"
    with open(path, "rb") as f:
        return f"data:{mt};base64,{base64.b64encode(f.read()).decode()}"


class OpenAIClient:
    """Любой OpenAI-совместимый /chat/completions: OpenRouter, OpenAI, vLLM, Ollama, LM Studio…"""

    kind = "api"

    def __init__(self, model, base_url=None, api_key=None, max_tokens=32000, temperature=None,
                 extra=None, timeout=900, retries=4):
        import requests  # noqa: F401  (проверка зависимости)
        self.model = model
        self.base_url = (base_url or os.environ.get("QBENCH_BASE_URL") or "https://openrouter.ai/api/v1").rstrip("/")
        self.api_key = api_key or os.environ.get("QBENCH_API_KEY") or os.environ.get("OPENROUTER_API_KEY") \
            or os.environ.get("OPENAI_API_KEY") or ""
        self.max_tokens, self.temperature = max_tokens, temperature
        self.extra = extra or {}
        self.timeout, self.retries = timeout, retries

    def _convert(self, messages):
        out = []
        for m in messages:
            if m.get("images"):
                parts = [{"type": "text", "text": m["content"]}]
                parts += [{"type": "image_url", "image_url": {"url": _data_uri(p)}} for p in m["images"]]
                out.append({"role": m["role"], "content": parts})
            else:
                out.append({"role": m["role"], "content": m["content"]})
        return out

    def chat(self, messages, phase=""):
        import requests
        body = {"model": self.model, "messages": self._convert(messages)}
        if self.max_tokens:
            body["max_tokens"] = self.max_tokens
        if self.temperature is not None:
            body["temperature"] = self.temperature
        body.update(self.extra)
        headers = {"Content-Type": "application/json", "HTTP-Referer": "https://github.com/v1teka/llm-quick-bench",
                   "X-Title": "llm-quick-bench"}
        if self.api_key:
            headers["Authorization"] = f"Bearer {self.api_key}"
        t0, err = time.time(), None
        for attempt in range(self.retries + 1):
            try:
                r = requests.post(f"{self.base_url}/chat/completions", headers=headers, json=body, timeout=self.timeout)
                if r.status_code in (429, 500, 502, 503, 504) and attempt < self.retries:
                    time.sleep(min(60, 5 * 2 ** attempt))
                    continue
                r.raise_for_status()
                data = r.json()
                msg = data["choices"][0]["message"]
                text = msg.get("content") or ""
                if isinstance(text, list):
                    text = "".join(p.get("text", "") for p in text if isinstance(p, dict))
                u = data.get("usage") or {}
                usage = {"in": u.get("prompt_tokens", 0), "out": u.get("completion_tokens", 0)}
                return text, usage, time.time() - t0
            except requests.RequestException as e:
                err = e
                if attempt < self.retries:
                    time.sleep(min(60, 5 * 2 ** attempt))
        raise RuntimeError(f"API недоступен: {err}")


class ManualClient:
    """Промпты пишутся в папку, ответы берутся из файлов NN_phase.response.md.

    Удобно для моделей без API (чат-интерфейс) и для прогонов через агента.
    Повторный запуск воспроизводит все шаги и останавливается на первом отсутствующем ответе.
    """

    kind = "manual"

    def __init__(self, folder):
        self.folder = folder
        os.makedirs(folder, exist_ok=True)
        self.n = 0

    def chat(self, messages, phase=""):
        self.n += 1
        stem = os.path.join(self.folder, f"{self.n:02d}_{phase or 'step'}")
        resp = stem + ".response.md"
        if os.path.exists(resp):
            with open(resp, encoding="utf-8") as f:
                text = f.read()
            meta = {}
            if os.path.exists(stem + ".meta.json"):
                with open(stem + ".meta.json", encoding="utf-8") as f:
                    meta = json.load(f)
            return text, {"in": meta.get("in", 0), "out": meta.get("out", 0)}, float(meta.get("seconds", 0))
        last = messages[-1]
        with open(stem + ".prompt.md", "w", encoding="utf-8") as f:
            if len([m for m in messages if m["role"] == "user"]) == 1:
                sys = [m for m in messages if m["role"] == "system"]
                if sys:
                    f.write(f"<!-- system -->\n{sys[0]['content']}\n\n<!-- user -->\n")
            else:
                f.write("<!-- продолжение того же диалога -->\n")
            f.write(last["content"])
            for p in last.get("images") or []:
                f.write(f"\n\n[изображение: {os.path.abspath(p)}]")
        raise NeedResponse(stem + ".prompt.md", resp)


class MockClient:
    """Детерминированная «модель» для демо и тестов: ideal | flawed | broken."""

    kind = "mock"

    def __init__(self, variant="flawed"):
        self.variant = variant
        self.calls = 0

    def chat(self, messages, phase=""):
        self.calls += 1
        time.sleep(0.01)
        if phase.startswith("vision"):
            text = '{"south_windows": {"floor1": 3, "floor2": 2}, "changes": ["a", "c", "d"]}'
        elif phase == "build":
            text = {"ideal": R.as_response(R.reference_house()),
                    "flawed": R.as_response(R.flawed_house()),
                    "broken": "Вот дом: {objects: [ {id: 1, ...} ]}"}[self.variant]
        elif phase.startswith("fix"):
            text = R.as_response(R.reference_house())
        else:
            text = R.as_response(R.reference_with_garage())
        return text, {"in": 0, "out": len(text) // 4}, 0.01


def make_client(model, **kw):
    if model.startswith("mock"):
        return MockClient(model.split(":", 1)[1] if ":" in model else "flawed")
    if kw.get("manual"):
        return ManualClient(kw["manual"])
    kw.pop("manual", None)
    return OpenAIClient(model, **kw)
