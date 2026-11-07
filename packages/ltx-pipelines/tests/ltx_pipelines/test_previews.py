"""Tests for safe decoded previews and the DeepSeek visual checkpoint adapter."""
from __future__ import annotations

import base64
import json
from pathlib import Path
from types import SimpleNamespace

import pytest
import torch
from PIL import Image

from neo.director.client import DeepSeekDirector
from neo.runtime.observer import ReasoningObserver, ReasoningPreviewAdapter
from neo.runtime.previews import VideoPreviewWriter
from neo.runtime.state import ProductionState


class _FakeVideoDecoder:
    def __init__(self) -> None:
        self.calls = 0

    def __call__(self, latent, tiling_config, generator, *, dtype=None):
        del tiling_config, generator, dtype
        self.calls += 1
        assert latent.ndim == 5
        # LTX playback tensors are frame-first, RGB-last.
        return iter([torch.full((3, 8, 10, 3), 0.5, dtype=torch.float32)])


def test_preview_writer_decodes_middle_rgb_frame_and_bounds_files(tmp_path: Path) -> None:
    decoder = _FakeVideoDecoder()
    writer = VideoPreviewWriter(decoder, tmp_path, max_previews=1)
    latent = torch.zeros((1, 128, 3, 4, 4), dtype=torch.float32)

    first = writer("stage_1/base", 1, latent)
    assert first.is_file()
    with Image.open(first) as image:
        assert image.size == (20, 8)
        assert image.mode == "RGB"
        assert image.getpixel((0, 0)) == (128, 128, 128)

    second = writer("stage_2_refine", 1, latent)
    assert second.is_file()
    assert not first.exists()
    assert len(list(tmp_path.glob("*.png"))) == 1
    assert decoder.calls == 2


def test_preview_writer_rejects_patchified_sampler_tokens(tmp_path: Path) -> None:
    writer = VideoPreviewWriter(_FakeVideoDecoder(), tmp_path)
    with pytest.raises(ValueError, match="unpatchified"):
        writer("stage_2_refine", 1, torch.zeros((1, 20, 128)))


def test_reasoning_preview_adapter_persists_plan_and_preview_path(tmp_path: Path) -> None:
    preview = tmp_path / "frame.png"
    Image.new("RGB", (8, 8), (12, 34, 56)).save(preview)

    class _FakeDirector:
        def __init__(self) -> None:
            self.calls = []

        def plan(self, brief, production_state):
            self.calls.append((brief, production_state))
            return SimpleNamespace(model_dump=lambda: {"scene_id": "scene-1"})

    director = _FakeDirector()
    state = ProductionState(tmp_path / "production.json")
    state.data["character_state"] = {"hero": "red jacket"}
    observer = ReasoningObserver(director)
    adapter = ReasoningPreviewAdapter(
        observer, "Continue the scene", state, every_n_chunks=2
    )

    assert adapter("stage_2_refine", 1, preview) is None
    assert not director.calls
    plan = adapter("stage_2_refine", 2, preview)

    assert plan == {"scene_id": "scene-1"}
    assert len(director.calls) == 1
    brief, submitted_state = director.calls[0]
    assert brief == "Continue the scene"
    assert submitted_state["character_state"] == {"hero": "red jacket"}
    assert submitted_state["denoising_checkpoint"]["preview_path"] == str(preview)
    assert state.data["last_director_plan"] == plan
    persisted = json.loads((tmp_path / "production.json").read_text())
    assert persisted["last_preview_path"] == str(preview)


def test_deepseek_client_attaches_preview_as_real_vision_input(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    preview = tmp_path / "preview.png"
    Image.new("RGB", (4, 4), (80, 90, 100)).save(preview)
    plan_json = json.dumps({
        "version": "neo-director-v1",
        "scene_id": "scene-1",
        "visual_prompt": "A red-jacketed hero turns toward the doorway.",
        "motion_direction": "Slow turn, then a step forward.",
        "camera_direction": "Medium tracking shot.",
        "audio": {},
        "references": [],
        "target_frames": 121,
        "fps": 24,
        "continuity_notes": [],
        "completion_criteria": [],
    })
    captured = {}

    class _Response:
        def __enter__(self):
            return self

        def __exit__(self, *_args):
            return False

        def read(self):
            return json.dumps({
                "choices": [{"message": {"content": plan_json}}]
            }).encode("utf-8")

    def fake_urlopen(request, timeout):
        captured["payload"] = json.loads(request.data.decode("utf-8"))
        captured["timeout"] = timeout
        return _Response()

    monkeypatch.setattr("urllib.request.urlopen", fake_urlopen)
    director = DeepSeekDirector(api_key="test-key")
    result = director.plan(
        "Continue the scene",
        {"denoising_checkpoint": {"preview_path": str(preview)}},
    )

    assert result.scene_id == "scene-1"
    payload = captured["payload"]
    assert payload["model"] == "deepseek-flash"
    user_message = payload["messages"][1]["content"]
    assert isinstance(user_message, list)
    image_part = next(part for part in user_message if part["type"] == "image_url")
    data_url = image_part["image_url"]["url"]
    encoded = data_url.split(",", 1)[1]
    assert base64.b64decode(encoded) == preview.read_bytes()
    assert image_part["image_url"]["detail"] == "low"
