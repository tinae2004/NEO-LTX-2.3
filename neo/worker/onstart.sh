#!/usr/bin/env bash
set -Eeuo pipefail
mkdir -p /workspace/neo-state
cd /workspace
if [ ! -d /workspace/NEO-LTX-2.3/.git ]; then
  git clone --depth 1 --branch neo-reasoning-dev https://github.com/tinae2004/NEO-LTX-2.3.git /workspace/NEO-LTX-2.3
fi
python -m pip install --upgrade pip
python -m pip install -r /workspace/NEO-LTX-2.3/neo/worker/requirements.txt
python -m pip install pytest numpy soundfile
cd /workspace/NEO-LTX-2.3
nohup python -m uvicorn neo.worker.api:app --host 0.0.0.0 --port 8765 --workers 1 > /workspace/neo-state/worker-api.log 2>&1 &
echo $! > /workspace/neo-state/worker-api.pid
