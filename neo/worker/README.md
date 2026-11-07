# NEO GPU Worker API

The worker runs inside the rented GPU machine. It exposes an authenticated API for health, GPU telemetry, and persistent job submission/status.

## Runtime configuration

- `NEO_WORKER_SECRET`: required bearer token. Never commit it.
- `NEO_WORKER_ROOT`: checkout path (default `/workspace/neo`).
- `NEO_WORKER_STATE`: persistent state path (default `/workspace/neo-state`).
- `NEO_REPO_BRANCH`: Git branch to clone (default `neo-reasoning-dev`).

Run with:

```bash
python -m pip install -r neo/worker/requirements.txt
NEO_WORKER_SECRET='...' uvicorn neo.worker.api:app --host 0.0.0.0 --port 8765
```

Health is unauthenticated at `/health`. All `/v1/*` routes require `Authorization: Bearer <NEO_WORKER_SECRET>`.

## API

- `GET /v1/gpu` — GPU name, memory and utilization.
- `POST /v1/jobs` with JSON `{"task":"gpu_diagnostics"}`, `{"task":"shona_tests"}`, or `{"task":"ltx_environment"}`.
- `GET /v1/jobs/{job_id}` — durable job state/result/error.
- `GET /v1/jobs` — recent jobs.

Jobs are recorded in SQLite under `NEO_WORKER_STATE`. This is a starter worker API; LTX generation handlers will be connected to the repository's actual pipeline after the first GPU smoke test. Keep the state directory on a persistent volume if jobs must survive instance replacement. An API process cannot survive destruction of the machine itself.

Do not expose this port publicly without the bearer token and network restrictions. Use TLS/reverse proxy or a private tunnel for production.
