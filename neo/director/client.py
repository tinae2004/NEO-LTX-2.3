"""OpenAI-compatible DeepSeek director client (stdlib-only transport)."""
from __future__ import annotations

import base64
import json
import mimetypes
import os
import urllib.error
import urllib.request
from pathlib import Path
from typing import Any

from .schema import DirectorPlan

SYSTEM_PROMPT = """You are NEO's film director. Return only a JSON object matching
the supplied DirectorPlan schema. Plan one coherent next generation interval.
Respect established character, prop, geography and action continuity. Give
explicit visual, motion, camera and audio direction. When a preview image is
attached, inspect the actual pixels and compare visible continuity with the
production state. Do not claim that an event exists unless visible or supplied
in the production state."""


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
        self.model = model or os.getenv("DEEPSEEK_MODEL", "deepseek-flash")
        self.base_url = (base_url or os.getenv("DEEPSEEK_BASE_URL", "https://api.deepseek.com")).rstrip("/")
        self.timeout = timeout

    def plan(self, brief: str, production_state: dict[str, Any]) -> DirectorPlan:
        user_payload = {
            "brief": brief,
            "production_state": production_state,
            "required_schema": DirectorPlan.model_json_schema(),
        }
        user_text = json.dumps(user_payload, ensure_ascii=False)
        user_content: str | list[dict[str, Any]] = user_text

        checkpoint = production_state.get("denoising_checkpoint", {})
        preview_path = (
            checkpoint.get("preview_path")
            if isinstance(checkpoint, dict)
            else None
        ) or production_state.get("last_preview_path")
        if preview_path:
            image_path = Path(str(preview_path))
            if image_path.is_file():
                image_size = image_path.stat().st_size
                if image_size > 32 * 1024 * 1024:
                    raise DeepSeekDirectorError(
                        "Preview image exceeds DeepSeek's 32 MiB inline image limit."
                    )
                mime_type = mimetypes.guess_type(image_path.name)[0] or "image/png"
                if mime_type not in {"image/jpeg", "image/png", "image/gif", "image/webp"}:
                    raise DeepSeekDirectorError(
                        f"Unsupported preview image type for DeepSeek vision: {mime_type}"
                    )
                encoded = base64.b64encode(image_path.read_bytes()).decode("ascii")
                user_content = [
                    {"type": "text", "text": user_text},
                    {
                        "type": "image_url",
                        "image_url": {
                            "url": f"data:{mime_type};base64,{encoded}",
                            "detail": "low",
                        },
                    },
                ]

        payload = {
            "model": self.model,
            "temperature": 0.3,
            "response_format": {"type": "json_object"},
            "messages": [
                {"role": "system", "content": SYSTEM_PROMPT},
                {"role": "user", "content": user_content},
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
