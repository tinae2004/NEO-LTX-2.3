"""DeepSeek failure-analysis helper for NEO GPU runs.

This agent is advisory only: it cannot execute shell commands, change rental limits,
or create/destroy Vast instances. It receives a redacted error summary and returns
diagnostic suggestions for the human/operator.
"""
from __future__ import annotations

import argparse
import os
import re
import sys
from typing import Any

import requests

API_URL = "https://api.deepseek.com/chat/completions"
MODEL = "deepseek-flash"  # Official DeepSeek V4.1 Flash API model


def redact(value: str) -> str:
    value = re.sub(r"(?i)(Bearer\s+)[A-Za-z0-9._-]+", r"\1[REDACTED]", value)
    value = re.sub(r"(?i)(api[_-]?key\s*[:=]\s*)[^\s,;]+", r"\1[REDACTED]", value)
    value = re.sub(r"sk-[A-Za-z0-9_-]{12,}", "[REDACTED]", value)
    return value[-12000:]


def analyze_failure(error: str, context: str = "") -> str:
    key = os.environ.get("DEEPSEEK_API_KEY", "").strip()
    if not key:
        return (
            "DeepSeek advisor skipped: DEEPSEEK_API_KEY is not configured. "
            "Set it as a GitHub Actions repository secret to enable AI diagnosis."
        )

    system = (
        "You are DeepSeek Flash, the diagnostic engineer for NEO VIDEO GEN's HunyuanVideo "
        "and LTX video-generation workflows running on Vast.ai. Analyze only the supplied "
        "logs and context. Identify the exact failing stage, root cause, evidence, "
        "and a minimal safe code/configuration fix. Distinguish proven facts from "
        "hypotheses. Explicitly check whether the GPU destruction was requested and flag "
        "any evidence it may still be running. Never ask for or reveal secrets. Never recommend exceeding "
        "the configured USD budget, hourly cap, or runtime. You are advisory only: "
        "do not claim to execute commands or modify infrastructure. Keep the answer "
        "under 350 words."
    )
    user = (
        "GPU workflow failure:\n" + redact(error) +
        ("\n\nRelevant context:\n" + redact(context) if context else "")
    )
    response = requests.post(
        API_URL,
        headers={"Authorization": f"Bearer {key}", "Content-Type": "application/json"},
        json={
            "model": MODEL,
            "messages": [
                {"role": "system", "content": system},
                {"role": "user", "content": user},
            ],
            "temperature": 0.1,
            "max_tokens": 700,
        },
        timeout=40,
    )
    response.raise_for_status()
    payload: dict[str, Any] = response.json()
    return payload["choices"][0]["message"]["content"].strip()


def main() -> int:
    parser = argparse.ArgumentParser()
    parser.add_argument("--error", required=True)
    parser.add_argument("--context", default="")
    args = parser.parse_args()
    try:
        print("=== DeepSeek diagnostic agent ===", flush=True)
        print(analyze_failure(args.error, args.context), flush=True)
        return 0
    except Exception as exc:
        print(f"DeepSeek advisor unavailable: {type(exc).__name__}: {exc}", file=sys.stderr, flush=True)
        return 0  # Diagnosis must never hide the original training failure.


if __name__ == "__main__":
    raise SystemExit(main())
