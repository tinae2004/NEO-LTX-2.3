#!/usr/bin/env python3
"""Resumable Hugging Face downloader with durable file-count progress logs."""
from __future__ import annotations

import argparse
import os
import sys
import threading
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
    parser.add_argument("--min-mbps", type=float, default=65.0)
    parser.add_argument("--speed-window-seconds", type=int, default=45)
    parser.add_argument("--stall-seconds", type=int, default=90)
    args = parser.parse_args()

    if not 1 <= args.max_workers <= 16:
        raise SystemExit("--max-workers must be between 1 and 16")

    args.local_dir.mkdir(parents=True, exist_ok=True)
    print(f"DOWNLOAD_START repo={args.repo_id} destination={args.local_dir}", flush=True)
    if args.allow:
        for pattern in args.allow:
            print(f"DOWNLOAD_ALLOW_PATTERN {pattern}", flush=True)

    if args.min_mbps <= 0 or args.speed_window_seconds < 15 or args.stall_seconds < args.speed_window_seconds:
        raise SystemExit("Invalid download watchdog settings.")

    # Track bytes actually written into this Hugging Face local_dir, including
    # .incomplete files. This measures the real model transfer, not Cloudflare.
    stop = threading.Event()
    failure = []
    def directory_bytes():
        total = 0
        try:
            for item in args.local_dir.rglob("*"):
                try:
                    if item.is_file():
                        total += item.stat().st_size
                except OSError:
                    pass
        except OSError:
            pass
        return total

    def watch_download():
        started = time.monotonic()
        last_change = started
        last_bytes = directory_bytes()
        samples = [(started, last_bytes)]
        last_report = started
        while not stop.wait(5):
            now = time.monotonic()
            current = directory_bytes()
            if current > last_bytes:
                last_change = now
            samples.append((now, current))
            cutoff = now - args.speed_window_seconds
            while len(samples) > 2 and samples[1][0] <= cutoff:
                samples.pop(0)
            elapsed = now - samples[0][0]
            delta = max(0, current - samples[0][1])
            mbps = (delta / 1_000_000) / elapsed if elapsed > 0 else 0.0
            if now - last_report >= 15:
                print(f"HF_REAL_DOWNLOAD_SPEED_MBPS={mbps:.2f} bytes_written={current} window_seconds={elapsed:.0f}", flush=True)
                last_report = now
            if now - last_change >= args.stall_seconds:
                failure.append(f"HF_DOWNLOAD_STALLED: no model bytes written for {args.stall_seconds}s")
                stop.set()
                return
            # Enforce only after the complete rolling window has elapsed and
            # actual file growth proves that a model transfer is underway.
            if elapsed >= args.speed_window_seconds and delta > 0 and mbps < args.min_mbps:
                failure.append(f"SLOW_HF_DOWNLOAD_HOST: {mbps:.2f} MB/s below {args.min_mbps:.0f} MB/s")
                stop.set()
                return
            last_bytes = current

    watcher = threading.Thread(target=watch_download, name="hf-download-watchdog", daemon=True)
    watcher.start()
    try:
        result = snapshot_download(
            repo_id=args.repo_id,
            local_dir=str(args.local_dir),
            allow_patterns=args.allow,
            max_workers=args.max_workers,
            tqdm_class=LogProgress,
            token=os.environ.get("HF_TOKEN") or None,
        )
    finally:
        stop.set()
        watcher.join(timeout=10)
    if failure:
        raise RuntimeError(failure[0])
    print(f"DOWNLOAD_COMPLETE repo={args.repo_id} path={result}", flush=True)


if __name__ == "__main__":
    main()
