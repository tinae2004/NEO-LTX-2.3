"""Crash-safe JSON production state for resumable NEO jobs."""
from __future__ import annotations

import json
import os
import tempfile
from pathlib import Path
from typing import Any


class ProductionState:
    def __init__(self, path: str | Path):
        self.path = Path(path)
        self.data: dict[str, Any] = self._load()

    def _load(self) -> dict[str, Any]:
        if not self.path.exists():
            return {"schema": "neo-production-state-v1", "status": "new",
                    "completed_intervals": [], "character_state": {},
                    "scene_state": {}, "last_frame_path": None}
        return json.loads(self.path.read_text(encoding="utf-8"))

    def save(self) -> None:
        self.path.parent.mkdir(parents=True, exist_ok=True)
        fd, temp_name = tempfile.mkstemp(prefix=self.path.name, dir=self.path.parent)
        try:
            with os.fdopen(fd, "w", encoding="utf-8") as stream:
                json.dump(self.data, stream, ensure_ascii=False, indent=2)
                stream.flush()
                os.fsync(stream.fileno())
            os.replace(temp_name, self.path)
        finally:
            if os.path.exists(temp_name):
                os.unlink(temp_name)

    def record_interval(self, interval_id: str, output_path: str,
                        last_frame_path: str | None, metadata: dict[str, Any] | None = None) -> None:
        completed = self.data.setdefault("completed_intervals", [])
        if any(item["interval_id"] == interval_id for item in completed):
            raise ValueError(f"Interval already recorded: {interval_id}")
        completed.append({"interval_id": interval_id, "output_path": output_path,
                          "metadata": metadata or {}})
        self.data["last_frame_path"] = last_frame_path
        self.data["status"] = "in_progress"
        self.save()
