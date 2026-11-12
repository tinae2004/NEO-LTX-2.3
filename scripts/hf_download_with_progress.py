#!/usr/bin/env python3
"""Hugging Face snapshot downloader with durable, live progress lines for CI logs."""
from __future__ import annotations

import os
import sys
import time
from huggingface_hub import snapshot_download
from tqdm.auto import tqdm


class LogProgress(tqdm):
    """Emit readable progress updates instead of carriage-return-only bars."""

    def __init__(self, *args, **kwargs):
        kwargs.setdefault("mininterval", 2.0)
        kwargs.setdefault("miniters", 1)
        kwargs.setdefault("dynamic_ncols", False)
        kwargs.setdefault("leave", True)
        super().__init__(*args, **kwargs)
        self._last_logged_percent = -1
        self._last_logged_time = 0.0

    def display(self, msg=None, pos=None):
        total = self.total or 0
        percent = int(self.n * 100 / total) if total else 0
        now = time.monotonic()
        if percent >= self._last_logged_percent + 5 or percent >= 100 or now - self._last_logged_time >= 15:
            name = str(self.desc or "Hugging Face file").replace("\\n", " ")
            total_text = f"{total:,}" if total else "unknown"
            print(
                f"DOWNLOAD_PROGRESS file={name} bytes={self.n:,}/{total_text} percent={percent if total else 'unknown'}",
                file=sys.stdout,
                flush=True,
            )
            self._last_logged_percent = percent
            self._last_logged_time = now


def main():
    if len(sys.argv) != 3:
        raise SystemExit("Usage: hf_download_with_progress.py REPO_ID LOCAL_DIR")
    repo_id, local_dir = sys.argv[1:]
    print(f"DOWNLOAD_START repo={repo_id} destination={local_dir}", flush=True)
    result = snapshot_download(
        repo_id=repo_id,
        local_dir=local_dir,
        tqdm_class=LogProgress,
        resume_download=True,
    )
    print(f"DOWNLOAD_COMPLETE repo={repo_id} path={result}", flush=True)


if __name__ == "__main__":
    main()
