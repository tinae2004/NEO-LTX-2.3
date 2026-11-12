# NEO Shona Engine — data-first build plan

## Goal

Build a Shona speech engine with explicit support for Zezuru, Karanga, Manyika and Korekore. Ndau is included as a closely related but separately labelled language/variety; do not silently merge Ndau into standard Shona. Every sample can carry a variety label or remain `unknown` until a fluent reviewer confirms it.

The speech engine is separate from HunyuanVideo. Hunyuan handles visual generation; the speech model handles pronunciation, voice identity and spoken dialogue; a synchronization layer aligns speech with character motion.

## What the repository now provides

- `scripts/inspect_shona_sources.py`: inspect Hugging Face dataset configs, splits and fields before downloading.
- `scripts/ingest_shona_hf.py`: stream one selected HF config into FLAC clips plus a canonical JSONL manifest.
- `scripts/audit_shona_corpus.py`: audit manifest integrity, source attribution, transcript status and variety balance.
- `neo_video_gen/shona/manifest.py`: shared schema validation and SHA-256 support.

No model is trained by these scripts. They establish traceable, auditable training data first.

## Initial sources to inspect

1. `google/WaxalNLP`, config `sna_asr` — the official card names this exact Shona ASR config and uses the `transcription` and `audio` fields. Its Shona ASR source is attributed to Digital Umuganda and is listed as CC-BY-SA-4.0. This is ASR speech, not a Shona TTS set. The WAXAL TTS config examples are for other languages; the card does not list Shona in its TTS language table. Do not treat this as a ready-made Shona voice-cloning corpus.
2. `Teleagents/zimbabwe-all-speech` — source-attributed collection reporting 596+ hours of Shona data across `shona_waxal`, `shona_waxal_pseudo`, `shona_bible` and `shona_fleurs`. The pseudo transcripts are not human ground truth. Its aggregate license is marked `other`; inspect each upstream source and obtain written clearance for commercial training before ingestion.
3. `manassehzw/sna-dataset` — a cleaned WAXAL-derived corpus reporting 86.2 hours and 16,980 clips. It inherits WAXAL lineage; do not count it as independent additional audio, and verify the effective upstream license before commercial use. Its published card notes speaker overlap across splits, so do not use its default splits as a final leakage-free benchmark.
4. `Professor/shona-speech-data` — reports 83.7 hours from Afrivoice; verify the dataset card and upstream license before use.

## Critical data rules

- Do not download random YouTube/radio/TV material for training unless the rights holder grants suitable permission. Publicly viewable does not mean training-permitted.
- Keep a separate source license and attribution record for every sample.
- Never infer dialect solely from a speaker's name, gender, location guess, or ASR output. Use a source-provided label or a human reviewer; otherwise set `unknown`.
- Keep human transcripts, canonical text, pseudo-labels and unverified transcripts distinct. Pseudo-labels are not ground truth.
- Detect duplicates by audio hash and, later, acoustic fingerprint. Multiple dataset mirrors may contain the same original recordings.
- Keep speaker-disjoint train/validation/test partitions. Never split clips from one speaker across partitions.
- Keep Bible/read-speech, conversational speech, narration and studio TTS data tagged separately. They are not interchangeable voice styles.
- Do not claim dialect coverage until the dataset audit shows enough reviewed examples and native-speaker evaluation for each variety.

## First commands

Install optional data tooling:

    pip install -e ".[shona-data]"

Inspect sources without downloading the full corpus:

    python scripts/inspect_shona_sources.py

After checking the exact dataset card and license for the intended use, ingest one config. Example only — replace config and column names with those reported by the inspection command:

    python scripts/ingest_shona_hf.py \\
      --dataset google/WaxalNLP --config YOUR_VERIFIED_CONFIG --split train \\
      --text-column YOUR_TRANSCRIPT_COLUMN --source-license CC-BY-4.0 \\
      --license-reference https://huggingface.co/datasets/google/WaxalNLP \\
      --license-reviewed --max-samples 1000

Audit the resulting corpus:

    python scripts/audit_shona_corpus.py ./neo_data/shona/manifest.jsonl --root ./neo_data/shona

The ingestion command deliberately requires explicit license review. It does not silently approve a dataset's commercial use.

## Dialect and language policy

Use these controlled labels: `zezuru`, `karanga`, `manyika`, `korekore`, `ndau`, `unknown`. The first four are the core Shona variety targets. Ndau must retain its own label and may require a distinct language code/head depending on linguistic review. Do not fabricate a five-way balanced dataset from sources that do not expose those labels.

## ASR is not TTS\n\nThe WAXAL Shona `sna_asr` recordings are useful for speech recognition and pronunciation analysis. They do not automatically provide the clean, consistent, scripted, single-speaker data normally needed to adapt a high-quality TTS voice. For Shona TTS, NEO should recruit consenting native speakers and record a balanced script set with explicit commercial model-training consent, clean microphone technique, consistent recording conditions, and human-verified transcripts. Collect different speakers and varieties separately; do not claim a dialect is supported until native speakers evaluate it.\n\n## Training is a later milestone

After collecting a rights-cleared, speaker-disjoint and human-reviewed seed corpus, benchmark candidate TTS architectures. Compare pronunciation, intelligibility, naturalness, dialect fidelity, speaker consistency and inference speed. Do not choose a model family just because it is popular or because a tiny demo checkpoint exists. Hunyuan fine-tuning is a separate visual-language experiment and will not teach its video transformer to speak.
