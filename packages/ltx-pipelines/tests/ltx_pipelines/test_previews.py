"""Tests for safe decoded previews and the DeepSeek visual checkpoint adapter."""
from __future__ import annotations

import base64
import json
from pathlib import Path
from types import SimpleNamespace

import pytest
import torch
from PIL import Image

from ltx_core.types import DenoiseMask, LatentState
from ltx_pipelines.chunks.layout import ChunkLayout
from ltx_pipelines.chunks.operators import denoise_chunks
from ltx_pipelines.chunks.state import Chunk
from ltx_pipelines.distilled import _director_plan_prompt
from ltx_pipelines.utils.types import VideoAudio

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
        # LTX playback tensors are frame-first, RGB-last; endpoints differ.
        frames = torch.empty((3, 8, 10, 3), dtype=torch.float32)
        frames[0].fill_(0.25)
        frames[1].fill_(0.5)
        frames[2].fill_(0.75)
        return iter([frames])


def test_preview_writer_decodes_start_end_contact_sheet_and_bounds_files(tmp_path: Path) -> None:
    decoder = _FakeVideoDecoder()
    writer = VideoPreviewWriter(decoder, tmp_path, max_previews=1)
    latent = torch.zeros((1, 128, 3, 4, 4), dtype=torch.float32)

    first = writer("stage_1/base", 1, latent)
    assert first.is_file()
    with Image.open(first) as image:
        assert image.size == (20, 8)
        assert image.mode == "RGB"
        assert image.getpixel((0, 0)) == (64, 64, 64)
        assert image.getpixel((10, 0)) == (191, 191, 191)

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



def test_director_plan_prompt_carries_visual_motion_camera_and_audio() -> None:
    prompt = _director_plan_prompt({
        "visual_prompt": "A woman stands beside a red car.",
        "motion_direction": "She opens the driver's door.",
        "camera_direction": "Slow dolly from a medium shot.",
        "audio": {
            "dialogue": ["I am leaving now."],
            "sound_effects": ["Car door latch clicks."],
            "music": "Soft cinematic strings.",
            "ambience": "Quiet evening street.",
        },
        "continuity_notes": ["Keep the red car on screen left."],
    })

    assert "Visual direction: A woman stands beside a red car." in prompt
    assert "Action and motion: She opens the driver's door." in prompt
    assert "Camera direction: Slow dolly from a medium shot." in prompt
    assert "Dialogue: I am leaving now." in prompt
    assert "Sound effects: Car door latch clicks." in prompt
    assert "Music: Soft cinematic strings." in prompt
    assert "Ambience: Quiet evening street." in prompt
    assert "Keep the red car on screen left." in prompt


def test_director_plan_prompt_rejects_empty_plan() -> None:
    with pytest.raises(ValueError, match="no usable"):
        _director_plan_prompt({})



def test_context_provider_supplies_new_embeddings_at_each_chunk() -> None:
    class _IdentityStage:
        def __init__(self):
            self.contexts = []

        def __call__(self, *, modalities, **_kwargs):
            self.contexts.append(modalities.video.context)
            latent = modalities.video.latent
            state = LatentState(
                latent=latent,
                denoise_mask=DenoiseMask(tensor=torch.ones((1, 1, 1, 1, 1))),
                positions=torch.zeros_like(latent),
                clean_latent=latent.clone(),
            )
            return VideoAudio(video=state, audio=None)

    layout = ChunkLayout(
        pixel_frames=1,
        start_pixel_frame=0,
        prev_video_carry_frames=0,
        next_video_carry_frames=0,
    )
    chunks = [
        Chunk(layout=layout, video=torch.zeros((1, 2, 1, 1, 1)), audio=None),
        Chunk(layout=layout, video=torch.zeros((1, 2, 1, 1, 1)), audio=None),
    ]
    first_context = torch.tensor([[1.0]])
    second_context = torch.tensor([[2.0]])
    stage = _IdentityStage()

    outputs = list(
        denoise_chunks(
            iter(chunks),
            stage,
            sigmas=torch.tensor([1.0, 0.0]),
            noiser=object(),
            fps=24.0,
            context_provider=lambda index, _chunk: (
                first_context if index == 1 else second_context,
                None,
            ),
        )
    )

    assert len(outputs) == 2
    torch.testing.assert_close(stage.contexts[0], first_context)
    torch.testing.assert_close(stage.contexts[1], second_context)
