#!/usr/bin/env python3
"""Stream a reviewed Hugging Face speech dataset into NEO's canonical audio manifest.

This tool intentionally requires explicit license review acknowledgement. It does not
infer dialect from accent, geography, or dataset name; set --variety only when supported
by source metadata or human review. Unlabelled dialects remain 'unknown'.
"""
from __future__ import annotations

import argparse
import hashlib
import json
import re
from pathlib import Path
from typing import Any

import numpy as np

from neo_video_gen.shona.manifest import append_jsonl, sha256_file


def clean_text(value: Any) -> str:
    return re.sub(r"\\s+", " ", str(value or "")).strip()


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--dataset", required=True, help="Hugging Face dataset repo ID")
    parser.add_argument("--config", required=True, help="Exact dataset configuration name")
    parser.add_argument("--split", default="train")
    parser.add_argument("--audio-column", default="audio")
    parser.add_argument("--text-column", required=True, help="Exact transcript column name")
    parser.add_argument("--speaker-column", default="speaker_id")
    parser.add_argument("--variety-column", default=None, help="Optional source column with reviewed variety labels")
    parser.add_argument("--variety", choices=["zezuru", "karanga", "manyika", "korekore", "ndau", "unknown"], default="unknown")
    parser.add_argument("--language", choices=["sna", "ndc"], default="sna")
    parser.add_argument("--transcript-type", choices=["human", "pseudo", "canonical", "unknown"], default="human")
    parser.add_argument("--source-license", required=True, help="License identifier copied from source terms")
    parser.add_argument("--license-reference", required=True, help="URL or written record of license review")
    parser.add_argument("--license-reviewed", action="store_true", help="Confirm terms permit this intended local training use")
    parser.add_argument("--output", type=Path, default=Path("./neo_data/shona"))
    parser.add_argument("--max-samples", type=int, default=0, help="0 means stream the full selected split")
    parser.add_argument("--token", default=None, help="Optional HF read token; prefer HF_TOKEN environment variable")
    args = parser.parse_args()
    if not args.license_reviewed:
        raise SystemExit("Refusing ingestion: review the dataset and its upstream terms, then pass --license-reviewed.")
    if args.max_samples < 0:
        raise SystemExit("--max-samples must be >= 0")

    try:
        from datasets import load_dataset
        import soundfile as sf
    except ImportError as exc:
        raise SystemExit("Install optional dependencies: pip install -e '.[shona-data]'") from exc

    audio_dir = args.output / "audio"
    audio_dir.mkdir(parents=True, exist_ok=True)
    manifest = args.output / "manifest.jsonl"
    dataset = load_dataset(args.dataset, args.config, split=args.split, streaming=True, token=args.token)
    written = skipped = 0

    for index, row in enumerate(dataset):
        if args.max_samples and written >= args.max_samples:
            break
        transcript = clean_text(row.get(args.text_column))
        audio = row.get(args.audio_column)
        if not transcript or not isinstance(audio, dict) or audio.get("array") is None:
            skipped += 1
            continue
        try:
            samples = np.asarray(audio["array"], dtype=np.float32)
            sample_rate = int(audio["sampling_rate"])
            if samples.ndim == 2:
                samples = samples.mean(axis=1) if samples.shape[0] < samples.shape[1] else samples.mean(axis=0)
            if samples.ndim != 1 or sample_rate <= 0 or samples.size == 0:
                skipped += 1
                continue
            source_id = str(row.get("id") or row.get("source_id") or index)
            stable = hashlib.sha256(f"{args.dataset}|{args.config}|{args.split}|{source_id}".encode()).hexdigest()[:20]
            relative_audio = Path("audio") / f"{stable}.flac"
            target = args.output / relative_audio
            if not target.exists():
                sf.write(target, samples, sample_rate, format="FLAC", subtype="PCM_16")
            speaker = row.get(args.speaker_column, "unknown")
            variety = args.variety
            if args.variety_column and row.get(args.variety_column):
                candidate = str(row[args.variety_column]).strip().lower()
                if candidate in {"zezuru", "karanga", "manyika", "korekore", "ndau"}:
                    variety = candidate
            record = {
                "sample_id": stable, "audio_path": str(relative_audio), "text": transcript,
                "language": args.language, "variety": variety,
                "speaker_id": str(speaker) if speaker not in (None, "") else "unknown",
                "source_dataset": args.dataset, "source_config": args.config, "source_split": args.split,
                "source_license": args.source_license, "license_reference": args.license_reference,
                "license_reviewed": True, "transcript_type": args.transcript_type,
                "transcript_verified": args.transcript_type in {"human", "canonical"},
                "duration_seconds": round(float(samples.size / sample_rate), 4),
                "sample_rate": sample_rate, "audio_sha256": sha256_file(target),
                "source_row_index": index,
            }
            append_jsonl(manifest, record)
            written += 1
            if written % 1000 == 0:
                print(json.dumps({"written": written, "skipped": skipped, "manifest": str(manifest)}), flush=True)
        except Exception as exc:
            skipped += 1
            print(f"Skipping row {index}: {exc}", flush=True)

    print(json.dumps({"written": written, "skipped": skipped, "manifest": str(manifest), "audio_dir": str(audio_dir)}, indent=2))


if __name__ == "__main__":
    main()
