"""Sampler checkpoint callback contract tests for synchronized audio/video updates."""
from collections.abc import Callable

import pytest
import torch

from ltx_core.types import DenoiseMask, LatentState
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

    _run_sampler(
        sampler,
        _latent_state(),
        _latent_state(2.0),
        lambda step, total, video, audio: events.append((step, total, video, audio)),
    )

    assert [(step, total) for step, total, _, _ in events] == [(1, 2), (2, 2)]
    assert all(video is not None and audio is not None for _, _, video, audio in events)
    assert all(video.latent.shape == audio.latent.shape for _, _, video, audio in events)


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
