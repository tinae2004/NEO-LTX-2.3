"""Validate NEO scene manifests and referenced files."""
from __future__ import annotations

import json
from pathlib import Path
from typing import Any

REQUIRED = {
    "scene_id", "engine", "engine_version", "schema_version",
    "capture_profile", "coordinate_system", "units", "frames",
}


def validate_manifest(path: str | Path, check_files: bool = True) -> dict[str, Any]:
    manifest_path = Path(path)
    data = json.loads(manifest_path.read_text(encoding="utf-8"))
    errors: list[str] = []
    warnings: list[str] = []
    missing = sorted(REQUIRED - data.keys())
    if missing:
        errors.append("Missing fields: " + ", ".join(missing))
    frames = data.get("frames", [])
    if not isinstance(frames, list) or not frames:
        errors.append("frames must be a non-empty list")
        frames = []
    seen: set[int] = set()
    for frame in frames:
        index = frame.get("frame_index")
        if index in seen:
            errors.append(f"Duplicate frame_index: {index}")
        seen.add(index)
        if "timestamp_seconds" not in frame:
            errors.append(f"Frame {index} has no timestamp_seconds")
        if not frame.get("available_channels"):
            warnings.append(f"Frame {index} reports no available channels")
        if check_files:
            candidates = list(frame.get("passes", {}).values())
            if frame.get("state_file"):
                candidates.append(frame["state_file"])
            for relative in candidates:
                if not (manifest_path.parent / relative).exists():
                    errors.append(f"Frame {index}: missing file {relative}")
    return {
        "valid": not errors,
        "manifest": str(manifest_path),
        "frame_count": len(frames),
        "errors": errors,
        "warnings": warnings,
    }
