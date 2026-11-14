#!/usr/bin/env python3
"""Authenticated browser-style Hugging Face downloader with HTTP Range resume."""
from __future__ import annotations
import argparse, fnmatch, os, time
from concurrent.futures import ThreadPoolExecutor, as_completed
from pathlib import Path
from urllib.parse import quote
import requests
from huggingface_hub import HfApi

def matches(name, patterns):
    return not patterns or any(fnmatch.fnmatch(name, p) for p in patterns)

def download_one(repo, revision, item, dest, token, session):
    name = item.rfilename
    expected = getattr(item, "size", None)
    target = dest / name
    target.parent.mkdir(parents=True, exist_ok=True)
    if target.is_file() and expected is not None and target.stat().st_size == expected:
        print(f"DOWNLOAD_SKIP_VERIFIED file={name} bytes={expected}", flush=True)
        return expected
    partial = target.with_name(target.name + ".incomplete")
    url = f"https://huggingface.co/{repo}/resolve/{revision}/{quote(name, safe='/')}?download=true"
    for attempt in range(4):
        offset = partial.stat().st_size if partial.exists() else 0
        headers = {"Authorization": f"Bearer {token}", "User-Agent": "NEO-Hunyuan-GPU/1.0"}
        if offset:
            headers["Range"] = f"bytes={offset}-"
        try:
            with session.get(url, headers=headers, stream=True, allow_redirects=True, timeout=(30, 180)) as response:
                if response.status_code not in (200, 206):
                    response.raise_for_status()
                    raise RuntimeError(f"Unexpected HTTP {response.status_code}")
                append = response.status_code == 206 and offset > 0
                if offset and response.status_code == 200:
                    offset = 0
                received = offset
                print(f"FILE_TRANSFER_START file={name} resume_bytes={offset} http={response.status_code}", flush=True)
                with partial.open("ab" if append else "wb") as out:
                    last_report = time.monotonic()
                    for chunk in response.iter_content(chunk_size=8 * 1024 * 1024):
                        if chunk:
                            out.write(chunk)
                            received += len(chunk)
                            now = time.monotonic()
                            if now - last_report >= 10:
                                print(f"FILE_TRANSFER_PROGRESS file={name} bytes={received}", flush=True)
                                last_report = now
                    out.flush()
                    os.fsync(out.fileno())
            actual = partial.stat().st_size
            if expected is not None and actual != expected:
                raise IOError(f"Size mismatch expected={expected} actual={actual}")
            partial.replace(target)
            print(f"FILE_TRANSFER_COMPLETE file={name} bytes={actual}", flush=True)
            return actual
        except Exception as exc:
            print(f"FILE_TRANSFER_RETRY file={name} attempt={attempt+1}/4 error={type(exc).__name__}:{exc}", flush=True)
            if attempt == 3:
                raise
            time.sleep(5 * (attempt + 1))
    raise RuntimeError(f"Download failed: {name}")

def main():
    p = argparse.ArgumentParser()
    p.add_argument("repo_id")
    p.add_argument("local_dir", type=Path)
    p.add_argument("--allow", nargs="+", default=None)
    p.add_argument("--max-workers", type=int, default=4)
    a = p.parse_args()
    if not 1 <= a.max_workers <= 8:
        raise SystemExit("--max-workers must be between 1 and 8")
    token = os.environ.get("HF_TOKEN", "").strip()
    if not token:
        raise SystemExit("HF_TOKEN is missing; refusing anonymous download.")
    a.local_dir.mkdir(parents=True, exist_ok=True)
    info = HfApi(token=token).model_info(a.repo_id, files_metadata=True)
    files = [f for f in (info.siblings or []) if f.rfilename and matches(f.rfilename, a.allow)]
    if not files:
        raise SystemExit("No files matched the requested Hugging Face allow patterns.")
    print("HF_TOKEN_PRESENT=1", flush=True)
    print(f"HF_AUTHENTICATED=1 repo={a.repo_id} revision={info.sha}", flush=True)
    print(f"DIRECT_RESOLVE_DOWNLOAD=1 selected_files={len(files)}", flush=True)
    for pattern in a.allow or []:
        print(f"DOWNLOAD_ALLOW_PATTERN {pattern}", flush=True)
    expected_total = sum(int(f.size) for f in files if getattr(f, "size", None) is not None)
    print(f"DOWNLOAD_MANIFEST files={len(files)} expected_bytes={expected_total}", flush=True)
    session = requests.Session()
    done = total = 0
    failures = []
    with ThreadPoolExecutor(max_workers=a.max_workers) as pool:
        futures = {pool.submit(download_one, a.repo_id, info.sha, f, a.local_dir, token, session): f.rfilename for f in files}
        for future in as_completed(futures):
            name = futures[future]
            try:
                size = future.result()
                done += 1
                total += size
                print(f"DOWNLOAD_FILE_PROGRESS completed={done} total={len(files)} bytes={total}", flush=True)
            except Exception as exc:
                failures.append((name, str(exc)))
                print(f"DOWNLOAD_FILE_FAILED file={name} error={exc}", flush=True)
    session.close()
    if failures:
        raise SystemExit(f"Download failures: {failures[:5]}")
    print(f"DOWNLOAD_COMPLETE repo={a.repo_id} files={done} bytes={total} path={a.local_dir}", flush=True)

if __name__ == "__main__":
    main()
