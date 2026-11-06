"""Structured DeepSeek director contract for NEO-LTX.

DeepSeek directs production; it does not replace the trained LTX Gemma
conditioning encoder in this initial scaffold.
"""
from __future__ import annotations

from typing import Literal
from pydantic import BaseModel, Field


class AudioDirection(BaseModel):
    dialogue: list[str] = Field(default_factory=list)
    sound_effects: list[str] = Field(default_factory=list)
    music: str | None = None
    ambience: str | None = None


class ReferenceGuide(BaseModel):
    image_path: str
    frame_index: int = Field(ge=0)
    strength: float = Field(default=1.0, ge=0.0, le=2.0)
    purpose: str = "continuity"


class DirectorPlan(BaseModel):
    version: Literal["neo-director-v1"] = "neo-director-v1"
    scene_id: str
    visual_prompt: str
    motion_direction: str
    camera_direction: str = ""
    audio: AudioDirection = Field(default_factory=AudioDirection)
    references: list[ReferenceGuide] = Field(default_factory=list, max_length=50)
    target_frames: int = Field(default=121, ge=9)
    fps: int = Field(default=24, ge=1, le=60)
    continuity_notes: list[str] = Field(default_factory=list)
    completion_criteria: list[str] = Field(default_factory=list)
