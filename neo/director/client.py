"""OpenAI-compatible DeepSeek director client (stdlib-only transport)."""
from __future__ import annotations

import json
import os
import urllib.error
import urllib.request
from typing import Any

from .schema import DirectorPlan

SYSTEM_PROMPT = """You are NEO's film director. Return only a JSON object matching
the supplied DirectorPlan schema. Plan one coherent next generation interval.
Respect established character, prop, geography and action continuity. Give
explicit visual, motion, camera and audio direction. Do not claim that a
reference image or event exists unless supplied in the production state."""


class DeepSeekDirectorError(RuntimeError):
    pass


class DeepSeekDirector:
    """Calls a DeepSeek-compatible chat-completions endpoint.

    Configure DEEPSEEK_API_KEY. Optional overrides:
    DEEPSEEK_BASE_URL (default https://api.deepseek.com),
    DEEPSEEK_MODEL (default deepseek-chat).
    """

    def __init__(self, api_key: str | None = None, model: str | None = None,
                 base_url: str | None = None, timeout: int = 120):
        self.api_key = api_key or os.getenv("DEEPSEEK_API_KEY")
        if not self.api_key:
            raise DeepSeekDirectorError("Set DEEPSEEK_API_KEY before requesting a plan.")
        self.model = model or os.getenv("DEEPSEEK_MODEL", "deepseek-chat")
        self.base_url = (base_url or os.getenv("DEEPSEEK_BASE_URL", "https://api.deepseek.com")).rstrip("/")
        self.timeout = timeout

    def plan(self, brief: str, production_state: dict[str, Any]) -> DirectorPlan:
        payload = {
            "model": self.model,
            "temperature": 0.3,
            "response_format": {"type": "json_object"},
            "messages": [
                {"role": "system", "content": SYSTEM_PROMPT},
                {"role": "user", "content": json.dumps({
                    "brief": brief,
                    "production_state": production_state,
                    "required_schema": DirectorPlan.model_json_schema(),
                }, ensure_ascii=False)},
            ],
        }
        request = urllib.request.Request(
            self.base_url + "/chat/completions",
            data=json.dumps(payload).encode("utf-8"),
            headers={"Authorization": f"Bearer {self.api_key}",
                     "Content-Type": "application/json"},
            method="POST",
        )
        try:
            with urllib.request.urlopen(request, timeout=self.timeout) as response:
                body = json.loads(response.read().decode("utf-8"))
            content = body["choices"][0]["message"]["content"]
            return DirectorPlan.model_validate_json(content)
        except (urllib.error.URLError, KeyError, IndexError, ValueError) as exc:
            raise DeepSeekDirectorError(f"DeepSeek planning failed: {exc}") from exc
