from neo_video_gen.shona.manifest import validate_record


def sample_record():
    return {
        "sample_id": "sample-001",
        "audio_path": "audio/sample-001.flac",
        "text": "Mangwanani, makadii?",
        "language": "sna",
        "variety": "zezuru",
        "speaker_id": "speaker-001",
        "source_dataset": "example/corpus",
        "source_config": "shona",
        "source_split": "train",
        "source_license": "CC-BY-4.0",
        "license_reference": "https://example.org/license",
        "license_reviewed": True,
        "transcript_type": "human",
        "transcript_verified": True,
        "duration_seconds": 2.4,
        "sample_rate": 24000,
        "audio_sha256": "a" * 64,
    }


def test_valid_reviewed_record_passes():
    assert validate_record(sample_record()) == []


def test_unknown_dialect_is_allowed_until_review():
    record = sample_record()
    record["variety"] = "unknown"
    assert validate_record(record) == []


def test_pseudo_transcript_cannot_be_marked_verified():
    record = sample_record()
    record["transcript_type"] = "pseudo"
    record["transcript_verified"] = True
    assert any("pseudo transcript" in error for error in validate_record(record))


def test_unreviewed_license_is_rejected():
    record = sample_record()
    record["license_reviewed"] = False
    assert any("license has not been reviewed" in error for error in validate_record(record))


def test_ndau_is_kept_as_separate_language_label():
    record = sample_record()
    record["language"] = "ndc"
    record["variety"] = "ndau"
    assert validate_record(record) == []
