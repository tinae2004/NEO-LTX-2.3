from __future__ import annotations

from pathlib import Path

import numpy as np
import soundfile as sf

from neo.speech.dialogue import (
    DialogueTurn,
    mix_dialogue_wav,
    mux_video_with_shona_audio,
    render_shona_dialogue,
)


class FakeSynthesizer:
    def __init__(self) -> None:
        self.unloaded = False

    def synthesize(self, text: str, output_path: str | Path) -> Path:
        assert text
        path = Path(output_path)
        sf.write(path, np.ones((2400, 1), dtype=np.float32) * 0.1, 24_000)
        return path

    def unload(self) -> None:
        self.unloaded = True


def test_render_preserves_turn_order_and_unloads(tmp_path: Path) -> None:
    backend = FakeSynthesizer()
    turns = [
        DialogueTurn("Mangwanani.", 0.0),
        DialogueTurn("Makadii?", 1.5, speaker="second"),
    ]
    rendered = render_shona_dialogue(turns, synthesizer=backend, output_dir=tmp_path)
    assert [turn for turn, _ in rendered] == turns
    assert all(path.is_file() for _, path in rendered)
    assert backend.unloaded


def test_mix_places_dialogue_at_authored_timestamp(tmp_path: Path) -> None:
    speech = tmp_path / "speech.wav"
    sf.write(speech, np.ones((2400, 1), dtype=np.float32) * 0.2, 24_000)
    output = mix_dialogue_wav(
        [(DialogueTurn("Mhoro.", 1.0), speech)],
        output_path=tmp_path / "mix.wav",
    )
    audio, sr = sf.read(output, dtype="float32", always_2d=True)
    assert sr == 24_000
    assert audio.shape == (26_400, 2)
    assert np.max(np.abs(audio[:20_000])) == 0.0
    assert np.max(np.abs(audio[24_000:])) > 0.0


def test_mix_ducks_soundtrack_under_dialogue(tmp_path: Path) -> None:
    speech = tmp_path / "speech.wav"
    bed = tmp_path / "bed.wav"
    sf.write(speech, np.ones((2400, 1), dtype=np.float32) * 0.2, 24_000)
    sf.write(bed, np.ones((24_000, 1), dtype=np.float32) * 0.1, 24_000)
    output = mix_dialogue_wav(
        [(DialogueTurn("Mhoro.", 0.25), speech)],
        output_path=tmp_path / "mix.wav",
        soundtrack_path=bed,
        duck_db=-8.0,
    )
    audio, _ = sf.read(output, dtype="float32", always_2d=True)
    # The bed is ducked during the spoken segment, then returns to its normal level.
    assert audio[7_000, 0] < audio[15_000, 0]
