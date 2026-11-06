"""Sampler checkpoint callback contract tests for synchronized audio/video updates."""
from collections.abc import Callable

import pytest
import torch

from ltx_core.types import DenoiseMask, LatentState
from ltx_pipelines.distilled import _stage_step_callback, ancestral_sampler_kwargs
from ltx_pipelines.utils.samplers import euler_ancestral_denoising_loop, euler_denoising_loop
from ltx_pipelines.utils.types import DenoisedLatentResult, VideoAudio


def _latent_state(value: float = 0.0) -> LatentState:
    latent = torch.full((1, 2), value, dtype=torch.float32)
    return LatentState(
        latent=latent,
        denoise_mask=DenoiseMask(tensor=torch.ones_like(latent)),
        positions=torch.zeros_like(latent),
        clean_latent=torch.zeros_like(latent),
    )


class _ConstantDenoiser:
    def __call__(
        self,
        transformer: object,
        video_state: LatentState | None,
        audio_state: LatentState | None,
        sigmas: torch.Tensor,
        step_index: int,
        v_context: torch.Tensor | None = None,
        a_context: torch.Tensor | None = None,
    ) -> VideoAudio[DenoisedLatentResult]:
        del transformer, sigmas, step_index, v_context, a_context
        video = (
            DenoisedLatentResult(denoised=torch.ones_like(video_state.latent))
            if video_state is not None
            else None
        )
        audio = (
            DenoisedLatentResult(denoised=torch.ones_like(audio_state.latent))
            if audio_state is not None
            else None
        )
        return VideoAudio(video=video, audio=audio)


def _run_sampler(
    sampler: Callable[..., VideoAudio[LatentState]],
    video_state: LatentState | None,
    audio_state: LatentState | None,
    callback: Callable[[int, int, LatentState | None, LatentState | None], None],
) -> VideoAudio[LatentState]:
    kwargs: dict[str, object] = {}
    if sampler is euler_ancestral_denoising_loop:
        # Deterministic ancestral limit keeps this contract test CPU-only and repeatable.
        kwargs.update(eta=0.0, s_noise=0.0, model_dtype=torch.float32)
    return sampler(
        sigmas=torch.tensor([1.0, 0.5, 0.0], dtype=torch.float32),
        video_state=video_state,
        audio_state=audio_state,
        transformer=object(),
        denoiser=_ConstantDenoiser(),
        step_callback=callback,
        **kwargs,
    )


@pytest.mark.parametrize(
    "sampler",
    [euler_denoising_loop, euler_ancestral_denoising_loop],
    ids=["euler", "ancestral-euler"],
)
def test_callback_observes_both_modalities_after_each_joint_update(sampler: Callable[..., object]) -> None:
    events: list[tuple[int, int, LatentState | None, LatentState | None]] = []

    initial_video = _latent_state()
    initial_audio = _latent_state(2.0)
    original_video = initial_video.latent.clone()
    original_audio = initial_audio.latent.clone()

    _run_sampler(
        sampler,
        initial_video,
        initial_audio,
        lambda step, total, video, audio: events.append((step, total, video, audio)),
    )

    assert [(step, total) for step, total, _, _ in events] == [(1, 2), (2, 2)]
    assert all(video is not None and audio is not None for _, _, video, audio in events)
    assert all(not torch.equal(video.latent, original_video) for _, _, video, _ in events)
    assert all(not torch.equal(audio.latent, original_audio) for _, _, _, audio in events)


@pytest.mark.parametrize(
    ("sampler", "present_modality"),
    [
        (euler_denoising_loop, "video"),
        (euler_denoising_loop, "audio"),
        (euler_ancestral_denoising_loop, "video"),
        (euler_ancestral_denoising_loop, "audio"),
    ],
    ids=["euler-video-only", "euler-audio-only", "ancestral-video-only", "ancestral-audio-only"],
)
def test_callback_preserves_absent_modality_as_none(
    sampler: Callable[..., object], present_modality: str
) -> None:
    events: list[tuple[int, int, LatentState | None, LatentState | None]] = []
    video_state = _latent_state() if present_modality == "video" else None
    audio_state = _latent_state() if present_modality == "audio" else None

    result = _run_sampler(
        sampler,
        video_state,
        audio_state,
        lambda step, total, video, audio: events.append((step, total, video, audio)),
    )

    assert len(events) == 2
    assert all((video is None) == (present_modality != "video") for _, _, video, _ in events)
    assert all((audio is None) == (present_modality != "audio") for _, _, _, audio in events)
    assert (result.video is None) == (present_modality != "video")
    assert (result.audio is None) == (present_modality != "audio")



def test_ancestral_pipeline_adapter_forwards_callback(monkeypatch: pytest.MonkeyPatch) -> None:
    events: list[tuple[int, int, LatentState | None, LatentState | None]] = []

    def fake_ancestral_loop(
        sigmas: torch.Tensor,
        video_state: LatentState | None,
        audio_state: LatentState | None,
        transformer: object,
        denoiser: object,
        v_context: torch.Tensor | None = None,
        a_context: torch.Tensor | None = None,
        step_callback: Callable[[int, int, LatentState | None, LatentState | None], None] | None = None,
        **kwargs: object,
    ) -> VideoAudio[LatentState]:
        del sigmas, transformer, denoiser, v_context, a_context, kwargs
        assert step_callback is not None
        step_callback(1, 1, video_state, audio_state)
        return VideoAudio(video=video_state, audio=audio_state)

    monkeypatch.setattr("ltx_pipelines.distilled.euler_ancestral_denoising_loop", fake_ancestral_loop)
    loop = ancestral_sampler_kwargs(seed=7, dtype=torch.float32, noise_seed_offset=10000)["loop"]
    video_state = _latent_state()

    loop(
        sigmas=torch.tensor([1.0, 0.0]),
        video_state=video_state,
        audio_state=None,
        transformer=object(),
        denoiser=object(),
        step_callback=lambda step, total, video, audio: events.append((step, total, video, audio)),
    )

    assert len(events) == 1
    assert events[0][0:2] == (1, 1)
    assert events[0][2] is video_state
    assert events[0][3] is None


def test_pipeline_callback_assigns_stage_and_one_based_chunk_index() -> None:
    events: list[tuple[object, ...]] = []
    callback = _stage_step_callback(lambda *args: events.append(args), "stage_1_base")
    assert callback is not None

    callback(1, 2, None, None)
    callback(2, 2, None, None)
    callback(1, 2, None, None)

    assert [event[:4] for event in events] == [
        ("stage_1_base", 1, 1, 2),
        ("stage_1_base", 1, 2, 2),
        ("stage_1_base", 2, 1, 2),
    ]
