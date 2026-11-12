#!/usr/bin/env python3
"""Audit a NEO Shona JSONL manifest for completeness, quality labels and dialect balance."""
from __future__ import annotations

import argparse
import json
from pathlib import Path

from neo_video_gen.shona.manifest import audit_jsonl


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("manifest", type=Path)
    parser.add_argument("--root", type=Path, default=None, help="Root directory used to check audio files")
    parser.add_argument("--report", type=Path, default=None)
    args = parser.parse_args()
    report = audit_jsonl(args.manifest, root=args.root)
    rendered = json.dumps(report, indent=2, ensure_ascii=False)
    print(rendered)
    if args.report:
        args.report.parent.mkdir(parents=True, exist_ok=True)
        args.report.write_text(rendered + "\\n", encoding="utf-8")
    if report["errors"]:
        raise SystemExit(1)


if __name__ == "__main__":
    main()
