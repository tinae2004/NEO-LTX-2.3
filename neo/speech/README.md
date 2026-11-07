# NEO Shona speech path

## Goal

Generate intelligible, natural Zimbabwean chiShona dialogue with pronunciation checked by native speakers. Do not treat an English voice reading Shona spelling as acceptable. The initial target is standard written Zimbabwean Shona; dialect-labelled variants (for example Zezuru, Karanga and Manyika) must remain explicit rather than being silently mixed.

## Audio architecture

1. DeepSeek writes dialogue in Shona and returns the exact Shona transcript, speaker intent, emotion and scene timing. It must not invent phonetic spellings as a substitute for correct Shona orthography.
2. LTX generates the video and non-dialogue soundtrack (ambience, effects and music).
3. A Shona-capable TTS backend synthesizes each dialogue turn from the approved Shona transcript. Keep turns separate with speaker, start time and duration metadata.
4. Fit each speech segment to its scene timing using pauses and edit decisions. Do not time-stretch aggressively to force a bad fit.
5. Mix the Shona dialogue with LTX ambience/effects/music, ducking the background under speech. Mux the final audio with the video and verify A/V sync.
6. Keep the original LTX audio track as an optional layer. Do not discard useful ambience when replacing generated dialogue.

This is a deliberate two-pass path: it avoids claiming that LTX's native audio model already produces reliable Shona pronunciation. It also allows the Shona TTS model to be unloaded before LTX video decoding on a 24 GB GPU, then loaded after video generation, rather than keeping both large models resident together.

## Model and data candidates (license-aware)

- Baseline to evaluate: `multilingual-tts/F5-TTS-OpenBible-Shona` (F5-TTS, 24 kHz, reference-audio conditioned). Its model card states that it was trained on Open Bible Shona and needs a clean 5–10 second reference clip with an exact transcript. Its license is CC BY-SA 4.0. This is a useful baseline, but Bible-style read speech is not enough to certify conversational Zimbabwean Shona.
- Fine-tuning/evaluation data candidate: `manassehzw/sna-dataset` (CC BY 4.0; 16,980 clips, about 86.2 hours, 133 speakers according to its dataset card). Review source terms, speaker metadata, transcript quality and dialect coverage before training.
- Do not silently ship `manassehzw/sna-f5-tts` in a commercial NEO Cloud service: its published license is CC BY-NC 4.0 and its model card positions it as a research checkpoint. Obtain separate written commercial permission first.

Before any model is bundled or deployed, record the exact model revision, license, dataset provenance and required attribution in the model manifest.

## Pronunciation and quality gates

- Preserve standard Shona orthography and punctuation. Never pass Shona through English phonetic respelling.
- Maintain a reviewed lexicon for names, places, loanwords, acronyms and domain vocabulary. Store approved spelling, dialect, pronunciation note, and reviewer provenance.
- Use native-speaker review for the target accent and dialect. Automated ASR/phoneme checks are diagnostics, not proof of natural pronunciation.
- Build a held-out evaluation set containing everyday conversation, questions, names/places, numbers/dates, code-switching and longer sentences.
- Score word error rate with a Shona-capable ASR model, but require native listeners to rate pronunciation, intelligibility, prosody and naturalness. Reject clips with wrong words or visibly unnatural prosody.
- Keep transcript-to-audio alignment per dialogue segment so the video/lip-sync stage can use exact timings.

## 24 GB GPU policy

The Shona TTS model must be loaded sequentially with the LTX transformer/decoder by default. Keep the transcript, timing metadata and final WAV on CPU/disk. Add explicit peak allocated/reserved VRAM telemetry for each stage. A TTS checkpoint fitting by itself does not prove the full LTX + TTS workflow fits in 24 GB.

## Current implementation status

## Implemented integration modules

- `f5_shona.py`: optional, lazy F5-TTS inference backend. It loads the published OpenBible Shona checkpoint only when requested and can unload it before LTX runs.
- `dialogue.py`: renders timestamped Shona dialogue turns, unloads the TTS backend, mixes WAV tracks on CPU with optional soundtrack ducking, and muxes the final WAV into an MP4 with FFmpeg.
- `test_dialogue.py`: unit tests for turn ordering, GPU-model unload invocation, timestamp placement and soundtrack ducking.
- `requirements.txt`: optional speech dependencies, intentionally separate from the core LTX install.

Example orchestration:

```python
from pathlib import Path
from neo.speech.f5_shona import F5ShonaTTS, ShonaVoice
from neo.speech.dialogue import (
    DialogueTurn, render_shona_dialogue, mix_dialogue_wav,
    mux_video_with_shona_audio,
)

tts = F5ShonaTTS(ShonaVoice(
    reference_audio=Path("voices/approved_shona_reference.wav"),
    reference_text="Mhinduro chaiyo yakanyorwa sezvayakataurwa.",
))
turns = [
    DialogueTurn("Mangwanani, makadii nhasi?", start_seconds=1.2),
    DialogueTurn("Tiri kufara kukuonai.", start_seconds=3.8),
]
rendered = render_shona_dialogue(turns, synthesizer=tts, output_dir="work/shona")
mixed = mix_dialogue_wav(
    rendered, output_path="work/shona_mix.wav",
    soundtrack_path="work/ltx_soundtrack.wav",
)
mux_video_with_shona_audio(
    "work/ltx_video.mp4", mixed, "output/neo_shona.mp4",
)
```

Install speech dependencies in a separate worker/environment with
`pip install -r neo/speech/requirements.txt`. FFmpeg must be installed on the
worker that performs final muxing. The voice reference must be a clean 5–10
second recording from a speaker who has consented to its use; its transcript
must match exactly.

This is code-level integration, not a completed runtime validation. The model
download, real inference, audio quality, FFmpeg mux, tests, and full RTX
3090/4090 24 GB run still need execution in the target environment. The
OpenBible model is read-speech trained, so native-speaker listening tests are
required before describing it as natural conversational Zimbabwean Shona.
