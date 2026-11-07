from __future__ import annotations

import asyncio
import json
import os
import secrets
import sqlite3
import subprocess
import sys
import time
import uuid
from pathlib import Path
from typing import Any

from fastapi import Depends, FastAPI, Header, HTTPException
from pydantic import BaseModel, Field

ROOT = Path(os.getenv("NEO_WORKER_ROOT", "/workspace/neo"))
STATE = Path(os.getenv("NEO_WORKER_STATE", "/workspace/neo-state"))
DB = STATE / "jobs.sqlite3"
SECRET = os.environ.get("NEO_WORKER_SECRET", "")
REPO_URL = os.getenv("NEO_REPO_URL", "https://github.com/tinae2004/NEO-LTX-2.3.git")
BRANCH = os.getenv("NEO_REPO_BRANCH", "neo-reasoning-dev")
app = FastAPI(title="NEO GPU Worker", version="1.0.0")
_tasks: dict[str, asyncio.Task] = {}


def db() -> sqlite3.Connection:
    STATE.mkdir(parents=True, exist_ok=True)
    conn = sqlite3.connect(DB)
    conn.row_factory = sqlite3.Row
    conn.execute("""CREATE TABLE IF NOT EXISTS jobs (
        id TEXT PRIMARY KEY, task TEXT NOT NULL, status TEXT NOT NULL,
        created REAL NOT NULL, updated REAL NOT NULL, result TEXT, error TEXT
    )""")
    return conn


def authorize(authorization: str | None = Header(default=None)) -> None:
    if not SECRET:
        raise HTTPException(503, "Worker authentication is not configured")
    supplied = authorization.removeprefix("Bearer ").strip() if authorization else ""
    if not secrets.compare_digest(supplied, SECRET):
        raise HTTPException(401, "Invalid worker token")


def run_checked(args: list[str], cwd: Path | None = None, timeout: int = 1800) -> dict[str, Any]:
    p = subprocess.run(args, cwd=str(cwd) if cwd else None, text=True,
                       stdout=subprocess.PIPE, stderr=subprocess.STDOUT,
                       timeout=timeout, check=False)
    if p.returncode:
        raise RuntimeError(f"Command failed ({p.returncode}): {' '.join(args)}\n{p.stdout[-12000:]}")
    return {"returncode": p.returncode, "output": p.stdout[-12000:]}


async def execute(job_id: str, task_name: str) -> None:
    now = time.time()
    with db() as conn:
        conn.execute("UPDATE jobs SET status='running', updated=? WHERE id=?", (now, job_id))
    try:
        if task_name == "gpu_diagnostics":
            result = run_checked(["bash", "-lc", "nvidia-smi && python -c 'import torch; assert torch.cuda.is_available(); print(torch.__version__); print(torch.cuda.get_device_name(0)); print(round(torch.cuda.get_device_properties(0).total_memory/1024**3,1), \"GiB\")'"], timeout=300)
        elif task_name == "shona_tests":
            if not (ROOT / ".git").exists():
                run_checked(["git", "clone", "--depth", "1", "--branch", BRANCH, REPO_URL, str(ROOT)], timeout=600)
            run_checked([sys.executable, "-m", "pip", "install", "-q", "pytest", "numpy", "soundfile"], timeout=900)
            result = run_checked([sys.executable, "-m", "pytest", "-q", "neo/speech/test_dialogue.py"], cwd=ROOT, timeout=900)
        elif task_name == "ltx_environment":
            result = run_checked(["bash", "-lc", "nvidia-smi; python - <<'PY'\nimport importlib.util\nfor m in ('torch','torchvision','transformers','safetensors','ltx_core','ltx_pipelines'):\n print(m, bool(importlib.util.find_spec(m)))\nPY"], timeout=300)
        else:
            raise ValueError("Unsupported task. Allowed: gpu_diagnostics, shona_tests, ltx_environment")
        with db() as conn:
            conn.execute("UPDATE jobs SET status='succeeded', result=?, updated=? WHERE id=?",
                         (json.dumps(result), time.time(), job_id))
    except Exception as exc:
        with db() as conn:
            conn.execute("UPDATE jobs SET status='failed', error=?, updated=? WHERE id=?",
                         (str(exc)[-12000:], time.time(), job_id))


class JobRequest(BaseModel):
    task: str = Field(pattern="^(gpu_diagnostics|shona_tests|ltx_environment)$")


@app.on_event("startup")
async def startup() -> None:
    db().close()


@app.get("/health")
async def health() -> dict[str, Any]:
    return {"ok": True, "service": "neo-gpu-worker", "time": time.time()}


@app.get("/v1/gpu", dependencies=[Depends(authorize)])
async def gpu() -> dict[str, Any]:
    p = subprocess.run(["nvidia-smi", "--query-gpu=name,memory.total,memory.used,utilization.gpu", "--format=csv,noheader"],
                       text=True, capture_output=True)
    return {"returncode": p.returncode, "gpu": p.stdout.strip(), "error": p.stderr.strip()}


@app.post("/v1/jobs", dependencies=[Depends(authorize)], status_code=202)
async def submit_job(request: JobRequest) -> dict[str, str]:
    job_id = uuid.uuid4().hex
    now = time.time()
    with db() as conn:
        conn.execute("INSERT INTO jobs(id,task,status,created,updated) VALUES(?,?,?,?,?)",
                     (job_id, request.task, "queued", now, now))
    _tasks[job_id] = asyncio.create_task(execute(job_id, request.task))
    return {"job_id": job_id, "status": "queued"}


@app.get("/v1/jobs/{job_id}", dependencies=[Depends(authorize)])
async def get_job(job_id: str) -> dict[str, Any]:
    with db() as conn:
        row = conn.execute("SELECT * FROM jobs WHERE id=?", (job_id,)).fetchone()
    if not row:
        raise HTTPException(404, "Job not found")
    return dict(row)


@app.get("/v1/jobs", dependencies=[Depends(authorize)])
async def list_jobs(limit: int = 20) -> list[dict[str, Any]]:
    with db() as conn:
        rows = conn.execute("SELECT * FROM jobs ORDER BY created DESC LIMIT ?", (min(max(limit, 1), 100),)).fetchall()
    return [dict(row) for row in rows]
