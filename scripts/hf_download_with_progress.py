#!/usr/bin/env python3
"""Resumable Hugging Face downloader with durable file-count progress logs."""
from __future__ import annotations

import argparse
import sys
import time
from pathlib import Path

from huggingface_hub import snapshot_download
from tqdm.asyncio import tqdm_asyncio


class LogProgress(tqdm_asyncio):
    """Log completed snapshot files even when CI cannot render tqdm bars."""

    def __init__(self, *args, **kwargs):
        kwargs.setdefault("mininterval", 2.0)
        kwargs.setdefault("miniters", 1)
        kwargs.setdefault("dynamic_ncols", False)
        kwargs.setdefault("leave", True)
        super().__init__(*args, **kwargs)
        self._last_logged_time = 0.0
        self._last_logged_count = -1

    def display(self, msg=None, pos=None):
        now = time.monotonic()
        total = int(self.total or 0)
        done = int(self.n or 0)
        if done != self._last_logged_count or now - self._last_logged_time >= 15:
            percent = f"{100.0 * done / total:.1f}" if total else "unknown"
            print(
                f"DOWNLOAD_FILE_PROGRESS completed={done} total={total} percent={percent}",
                file=sys.stdout,
                flush=True,
            )
            self._last_logged_count = done
            self._last_logged_time = now


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("repo_id")
    parser.add_argument("local_dir", type=Path)
    parser.add_argument(
        "--allow",
        nargs="+",
        default=None,
        help="Optional Hugging Face allow_patterns. Omit to download the full repository.",
    )
    parser.add_argument("--max-workers", type=int, default=8)
    args = parser.parse_args()

    if not 1 <= args.max_workers <= 16:
        raise SystemExit("--max-workers must be between 1 and 16")

    args.local_dir.mkdir(parents=True, exist_ok=True)
    print(f"DOWNLOAD_START repo={args.repo_id} destination={args.local_dir}", flush=True)
    if args.allow:
        for pattern in args.allow:
            print(f"DOWNLOAD_ALLOW_PATTERN {pattern}", flush=True)

    result = snapshot_download(
        repo_id=args.repo_id,
        local_dir=str(args.local_dir),
        allow_patterns=args.allow,
        max_workers=args.max_workers,
        tqdm_class=LogProgress,
    )
    print(f"DOWNLOAD_COMPLETE repo={args.repo_id} path={result}", flush=True)


if __name__ == "__main__":
    main()
