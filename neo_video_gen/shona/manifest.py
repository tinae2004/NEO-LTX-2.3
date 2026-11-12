"""Canonical manifest schema and quality checks for NEO Shona speech data."""
from __future__ import annotations

import hashlib
import json
from pathlib import Path
from typing import Any

DIALECTS = {"zezuru", "karanga", "manyika", "korekore", "ndau", "unknown"}
LANGUAGES = {"sna", "ndc"}
REQUIRED_FIELDS = {
    "sample_id", "audio_path", "text", "language", "variety", "speaker_id",
    "source_dataset", "source_config", "source_split", "source_license", "license_reference",
    "license_reviewed", "transcript_type", "transcript_verified", "duration_seconds",
    "sample_rate", "audio_sha256",
}


def sha256_file(path: str | Path) -> str:
    digest = hashlib.sha256()
    with Path(path).open("rb") as stream:
        for chunk in iter(lambda: stream.read(1024 * 1024), b""):
            digest.update(chunk)
    return digest.hexdigest()


def validate_record(record: dict[str, Any], root: str | Path | None = None) -> list[str]:
    errors: list[str] = []
    missing = sorted(REQUIRED_FIELDS - record.keys())
    if missing:
        errors.append("missing fields: " + ", ".join(missing))
    if record.get("variety") not in DIALECTS:
        errors.append(f"unsupported variety: {record.get('variety')!r}")
    if record.get("language") not in LANGUAGES:
        errors.append(f"unsupported language code: {record.get('language')!r}")
    if not str(record.get("text", "")).strip():
        errors.append("empty transcript")
    if record.get("transcript_type") not in {"human", "pseudo", "canonical", "unknown"}:
        errors.append("transcript_type must be human, pseudo, canonical, or unknown")
    if record.get("transcript_type") == "pseudo" and record.get("transcript_verified") is True:
        errors.append("pseudo transcript cannot be marked verified without human review")
    if record.get("license_reviewed") is not True:
        errors.append("license has not been reviewed for this intended use")
    if not str(record.get("license_reference", "")).strip():
        errors.append("missing license_reference")
    checksum = str(record.get("audio_sha256", ""))
    if len(checksum) != 64 or any(ch not in "0123456789abcdef" for ch in checksum.lower()):
        errors.append("audio_sha256 must be a 64-character SHA-256 hex digest")
    try:
        if float(record.get("duration_seconds", 0)) <= 0:
            errors.append("duration_seconds must be positive")
        if int(record.get("sample_rate", 0)) <= 0:
            errors.append("sample_rate must be positive")
    except (TypeError, ValueError):
        errors.append("invalid duration_seconds or sample_rate")
    if root is not None and record.get("audio_path"):
        audio_path = Path(root) / str(record["audio_path"])
        if not audio_path.is_file():
            errors.append(f"audio file missing: {audio_path}")
    return errors


def append_jsonl(path: str | Path, record: dict[str, Any]) -> None:
    target = Path(path)
    target.parent.mkdir(parents=True, exist_ok=True)
    with target.open("a", encoding="utf-8") as stream:
        stream.write(json.dumps(record, ensure_ascii=False) + "\n")


def audit_jsonl(path: str | Path, root: str | Path | None = None) -> dict[str, Any]:
    counts: dict[str, int] = {}
    transcript_counts: dict[str, int] = {}
    speakers: set[str] = set()
    total_seconds = 0.0
    seen_ids: set[str] = set()
    seen_audio: set[str] = set()
    errors: list[dict[str, Any]] = []
    rows = 0
    with Path(path).open("r", encoding="utf-8") as stream:
        for line_number, line in enumerate(stream, 1):
            if not line.strip():
                continue
            rows += 1
            try:
                record = json.loads(line)
            except json.JSONDecodeError as exc:
                errors.append({"line": line_number, "errors": [f"invalid JSON: {exc}"]})
                continue
            problems = validate_record(record, root=root)
            sample_id = str(record.get("sample_id", ""))
            audio_hash = str(record.get("audio_sha256", ""))
            if sample_id and sample_id in seen_ids:
                problems.append("duplicate sample_id")
            if audio_hash and audio_hash in seen_audio:
                problems.append("duplicate audio_sha256; possible mirrored/repeated recording")
            if sample_id:
                seen_ids.add(sample_id)
            if audio_hash:
                seen_audio.add(audio_hash)
            if problems:
                errors.append({"line": line_number, "sample_id": record.get("sample_id"), "errors": problems})
            variety = str(record.get("variety", "unknown"))
            counts[variety] = counts.get(variety, 0) + 1
            transcript_type = str(record.get("transcript_type", "unknown"))
            transcript_counts[transcript_type] = transcript_counts.get(transcript_type, 0) + 1
            if record.get("speaker_id") not in (None, "", "unknown"):
                speakers.add(str(record["speaker_id"]))
            try:
                total_seconds += float(record.get("duration_seconds", 0))
            except (TypeError, ValueError):
                pass
    return {
        "manifest": str(path), "rows": rows, "valid_rows": rows - len(errors),
        "variety_counts": counts, "transcript_type_counts": transcript_counts,
        "unique_speakers": len(speakers), "total_hours": round(total_seconds / 3600, 3),
        "errors": errors,
    }
