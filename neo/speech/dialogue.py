"""Timed Shona dialogue synthesis, soundtrack mixing and MP4 muxing.

This is an optional NEO orchestration layer. It keeps speech turns on disk,
mixes WAV audio on CPU, and invokes FFmpeg only for the final mux.
"""
from __future__ import annotations

import shutil
import subprocess
from dataclasses import dataclass
from pathlib import Path
from typing import Protocol


class ShonaSynthesizer(Protocol):
    def synthesize(self, text: str, output_path: str | Path) -> Path: ...
    def unload(self) -> None: ...


@dataclass(frozen=True, slots=True)
class DialogueTurn:
    text: str
    start_seconds: float
    speaker: str = "narrator"
    gain: float = 1.0


def render_shona_dialogue(
    turns: list[DialogueTurn],
    *,
    synthesizer: ShonaSynthesizer,
    output_dir: str | Path,
) -> list[tuple[DialogueTurn, Path]]:
    """Render each exact Shona transcript to its own WAV, then unload TTS.

    TTS is loaded lazily by the backend and released before the caller starts
    or resumes LTX. Turn order and timing are preserved in the returned list.
    """
    destination = Path(output_dir)
    destination.mkdir(parents=True, exist_ok=True)
    rendered: list[tuple[DialogueTurn, Path]] = []
    try:
        for index, turn in enumerate(turns):
            if not turn.text.strip():
                raise ValueError(f"Dialogue turn {index} has empty Shona text.")
            if turn.start_seconds < 0:
                raise ValueError(f"Dialogue turn {index} has a negative start time.")
            if turn.gain < 0:
                raise ValueError(f"Dialogue turn {index} has a negative gain.")
            wav_path = destination / f"shona_{index:04d}.wav"
            synthesizer.synthesize(turn.text, wav_path)
            if not wav_path.is_file():
                raise RuntimeError(f"Shona TTS did not create the expected file: {wav_path}")
            rendered.append((turn, wav_path))
    finally:
        synthesizer.unload()
    return rendered


def mix_dialogue_wav(
    rendered_turns: list[tuple[DialogueTurn, Path]],
    *,
    output_path: str | Path,
    soundtrack_path: str | Path | None = None,
    sample_rate: int = 24_000,
    duck_db: float = -8.0,
) -> Path:
    """CPU-mix Shona dialogue over optional LTX soundtrack into stereo PCM WAV.

    Dialogue is placed at its authored start time. The soundtrack is attenuated
    under the union of speech intervals; no time-stretching is performed.
    """
    try:
        import numpy as np
        import soundfile as sf
    except ImportError as exc:
        raise RuntimeError("Install numpy and soundfile for CPU dialogue mixing.") from exc

    decoded: list[tuple[DialogueTurn, object]] = []
    total_samples = 0
    speech_mask = np.zeros(0, dtype=np.bool_)
    for turn, wav_path in rendered_turns:
        audio, sr = sf.read(str(wav_path), dtype="float32", always_2d=True)
        if sr != sample_rate:
            raise ValueError(
                f"{wav_path} is {sr} Hz; expected {sample_rate} Hz. "
                "Resample explicitly rather than silently changing pitch."
            )
        if audio.shape[1] == 1:
            audio = np.repeat(audio, 2, axis=1)
        elif audio.shape[1] != 2:
            raise ValueError(f"{wav_path} must be mono or stereo.")
        start = round(turn.start_seconds * sample_rate)
        total_samples = max(total_samples, start + len(audio))
        decoded.append((turn, audio))
    if soundtrack_path is not None:
        bed, bed_sr = sf.read(str(soundtrack_path), dtype="float32", always_2d=True)
        if bed_sr != sample_rate:
            raise ValueError(f"Soundtrack is {bed_sr} Hz; expected {sample_rate} Hz.")
        if bed.shape[1] == 1:
            bed = np.repeat(bed, 2, axis=1)
        elif bed.shape[1] != 2:
            raise ValueError("Soundtrack must be mono or stereo.")
        total_samples = max(total_samples, len(bed))
    else:
        bed = None

    if total_samples == 0:
        raise ValueError("No dialogue or soundtrack audio was provided.")

    mix = np.zeros((total_samples, 2), dtype=np.float32)
    speech_mask = np.zeros(total_samples, dtype=np.bool_)
    for turn, audio in decoded:
        start = round(turn.start_seconds * sample_rate)
        end = start + len(audio)
        mix[start:end] += audio * float(turn.gain)
        speech_mask[start:end] = True

    if bed is not None:
        bed_mix = np.zeros_like(mix)
        bed_mix[: len(bed)] = bed
        duck_gain = 10.0 ** (duck_db / 20.0)
        bed_mix[speech_mask] *= duck_gain
        mix += bed_mix

    # Leave headroom; do not hard-clip dialogue peaks.
    peak = float(np.max(np.abs(mix)))
    if peak > 0.98:
        mix *= 0.98 / peak

    target = Path(output_path)
    target.parent.mkdir(parents=True, exist_ok=True)
    sf.write(str(target), mix, sample_rate, subtype="PCM_16")
    return target


def mux_video_with_shona_audio(
    video_path: str | Path,
    mixed_audio_path: str | Path,
    output_path: str | Path,
    *,
    ffmpeg: str = "ffmpeg",
) -> Path:
    """Copy the generated video stream and mux the finished Shona audio track."""
    executable = shutil.which(ffmpeg)
    if executable is None:
        raise RuntimeError("FFmpeg was not found. Install it before muxing Shona dialogue.")
    video = Path(video_path)
    audio = Path(mixed_audio_path)
    output = Path(output_path)
    if not video.is_file() or not audio.is_file():
        raise FileNotFoundError("Both the generated video and mixed Shona WAV must exist.")
    output.parent.mkdir(parents=True, exist_ok=True)
    command = [
        executable, "-y", "-i", str(video), "-i", str(audio),
        "-map", "0:v:0", "-map", "1:a:0",
        "-c:v", "copy", "-c:a", "aac", "-b:a", "192k",
        "-movflags", "+faststart", "-shortest", str(output),
    ]
    completed = subprocess.run(command, capture_output=True, text=True, check=False)
    if completed.returncode != 0:
        output.unlink(missing_ok=True)
        raise RuntimeError(f"FFmpeg mux failed: {completed.stderr[-4000:]}")
    return output
