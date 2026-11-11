"""Engine-independent scene capture schema."""
from __future__ import annotations

from dataclasses import asdict, dataclass, field
from typing import Any


@dataclass
class FrameRecord:
    frame_index: int
    timestamp_seconds: float
    passes: dict[str, str] = field(default_factory=dict)
    state_file: str | None = None
    available_channels: list[str] = field(default_factory=list)
    unavailable_channels: list[str] = field(default_factory=list)


@dataclass
class SceneManifest:
    scene_id: str
    engine: str
    engine_version: str
    schema_version: str
    capture_profile: str
    coordinate_system: str
    units: str
    frames: list[FrameRecord] = field(default_factory=list)
    assets: list[dict[str, Any]] = field(default_factory=list)
    scene_metadata: dict[str, Any] = field(default_factory=dict)
    notes: list[str] = field(default_factory=list)

    def to_dict(self) -> dict[str, Any]:
        return asdict(self)
