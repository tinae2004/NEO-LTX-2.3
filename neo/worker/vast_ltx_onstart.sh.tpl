#!/usr/bin/env bash
set -Eeuo pipefail
ROOT=/workspace/neo_gpu_worker
LTX_ROOT=/workspace/LTX-2
MODEL_ROOT=/workspace/models
CHECKPOINT="$MODEL_ROOT/ltx-2.3/ltx-2.3-22b-distilled-1.1.safetensors"
UPSAMPLER="$MODEL_ROOT/ltx-2.3/ltx-2.3-spatial-upscaler-x2-1.1.safetensors"
GEMMA_ROOT="$MODEL_ROOT/gemma-3-12b"
STATUS="$ROOT/bootstrap-status.json"
mkdir -p "$ROOT" "$ROOT/videos" "$MODEL_ROOT/ltx-2.3" "$GEMMA_ROOT"
exec >> "$ROOT/startup.log" 2>&1
# Install the original NEO GPU control agent from the working MovieGen app.py.
CONTROL_ROOT=/workspace/.neo_gpu_control
mkdir -p "$CONTROL_ROOT/jobs"
chmod 700 "$CONTROL_ROOT" "$CONTROL_ROOT/jobs"
python - <<'PY_CONTROL_CFG'
import os,json
p='/workspace/.neo_gpu_control/config.json'; tmp=p+'.tmp'
cfg={'token':os.environ.get('NEO_GPU_CONTROL_TOKEN',''),'instance_id':''}
if len(cfg['token'])<32: raise SystemExit('NEO_GPU_CONTROL_TOKEN missing/too short')
with open(tmp,'w') as f: json.dump(cfg,f)
os.chmod(tmp,0o600); os.replace(tmp,p)
PY_CONTROL_CFG
CONTROL_B64='__CONTROL_B64__'
printf '%s' "$CONTROL_B64" | base64 -d | python -c "import gzip,sys; sys.stdout.buffer.write(gzip.decompress(sys.stdin.buffer.read()))" > "$CONTROL_ROOT/control_api.py"
chmod 700 "$CONTROL_ROOT/control_api.py"
nohup bash -lc 'while true; do export NEO_GPU_CONTROL_TOKEN="$(python -c "import json;print(json.load(open(\"/workspace/.neo_gpu_control/config.json\"))[\"token\"])" )"; export NEO_GPU_CONTROL_INSTANCE_ID="$(python -c "import json;print(json.load(open(\"/workspace/.neo_gpu_control/config.json\"))[\"instance_id\"])" )"; python -u /workspace/.neo_gpu_control/control_api.py; rc=$?; echo "Control API exited rc=$rc; restarting"; sleep 2; done' >> "$CONTROL_ROOT/supervisor.log" 2>&1 </dev/null &
echo $! > "$CONTROL_ROOT/supervisor.pid"
for i in $(seq 1 60); do
  if curl -fsS --max-time 2 http://127.0.0.1:8191/health >/dev/null 2>&1; then say 'NEO_GPU_CONTROL_API_READY port=8191'; break; fi
  sleep 1
done
say(){ printf '[%s] %s\n' "$(date -u +%FT%TZ)" "$*"; }
write_status(){
  python - "$STATUS" "$1" "$2" "${3:-}" <<'PY_STATUS'
import json,os,sys,time
path,ready,phase,error=sys.argv[1:]
tmp=path+'.tmp'
os.makedirs(os.path.dirname(path),exist_ok=True)
with open(tmp,'w',encoding='utf-8') as f: json.dump({'ready':ready.lower()=='true','phase':phase,'error':error[:1000],'updated_at':int(time.time())},f)
os.replace(tmp,path)
PY_STATUS
}
on_error(){ rc=$?; trap - ERR; say "BOOTSTRAP_FAILED exit=$rc command=${BASH_COMMAND:0:300}"; write_status false failed "exit=$rc; command=${BASH_COMMAND:0:300}" || true; exit "$rc"; }
trap on_error ERR
write_status false starting ""
say "NEO LTX-2.3 worker bootstrap started."
python - <<'PY_CFG'
import os,json
label=os.environ.get('VAST_CONTAINERLABEL','')
iid=label.rsplit('.',1)[-1] if '.' in label else label
cfg={'auth_token':os.environ.get('NEO_WORKER_TOKEN',''),'hf_token':os.environ.get('HF_TOKEN',''),
     'bucket':os.environ.get('NEO_BUCKET','Tinae/neo-bucket'),'instance_id':iid}
if len(cfg['auth_token'])<32: raise SystemExit('NEO_WORKER_TOKEN missing/too short')
if not cfg['hf_token']: raise SystemExit('HF_TOKEN missing')
p='/workspace/.neo_gpu_worker.json'; tmp=p+'.tmp'
with open(tmp,'w',encoding='utf-8') as f: json.dump(cfg,f)
os.chmod(tmp,0o600); os.replace(tmp,p)
PY_CFG
WORKER_B64='__WORKER_B64__'
printf '%s' "$WORKER_B64" | base64 -d | gzip -d > "$ROOT/worker.py"
chmod 700 "$ROOT/worker.py"
# Start the persistent API before downloading models so Actions can stream the current phase.
if ! curl -fsS --max-time 2 http://127.0.0.1:8190/health >/dev/null 2>&1; then
  nohup python -u "$ROOT/worker.py" >> "$ROOT/worker.log" 2>&1 </dev/null &
  echo $! > "$ROOT/worker.pid"
fi
for i in $(seq 1 30); do
  curl -fsS --max-time 2 http://127.0.0.1:8190/health >/dev/null 2>&1 && break
  sleep 1
done
write_status false installing_ltx_requirements ""
export DEBIAN_FRONTEND=noninteractive
if command -v apt-get >/dev/null 2>&1; then apt-get update -qq; apt-get install -y -qq git ffmpeg curl ca-certificates; fi
python -m pip install --disable-pip-version-check --upgrade "huggingface_hub[hf_transfer]<2.0"
if [ ! -d "$LTX_ROOT/.git" ]; then
  rm -rf "$LTX_ROOT"
  git clone --depth=1 https://github.com/Lightricks/LTX-2.git "$LTX_ROOT"
fi
python -m pip install --disable-pip-version-check --upgrade uv
cd "$LTX_ROOT"
uv sync --extra natten
cd /
write_status false downloading_ltx_models ""
export HF_HUB_ENABLE_HF_TRANSFER=1
if [ ! -s "$CHECKPOINT" ]; then hf download Lightricks/LTX-2.3 ltx-2.3-22b-distilled-1.1.safetensors --local-dir "$MODEL_ROOT/ltx-2.3"; fi
test -s "$CHECKPOINT"
if [ ! -s "$UPSAMPLER" ]; then hf download Lightricks/LTX-2.3 ltx-2.3-spatial-upscaler-x2-1.1.safetensors --local-dir "$MODEL_ROOT/ltx-2.3"; fi
test -s "$UPSAMPLER"
if [ ! -f "$GEMMA_ROOT/config.json" ]; then hf download google/gemma-3-12b-it-qat-q4_0-unquantized --local-dir "$GEMMA_ROOT"; fi
test -f "$GEMMA_ROOT/config.json"
write_status false verifying_ltx_pipeline ""
"$LTX_ROOT/.venv/bin/python" -m ltx_pipelines.distilled --help > "$ROOT/ltx_cli_help.txt"
for flag in --distilled-checkpoint-path --spatial-upsampler-path --gemma-root --output-path --num-frames --frame-rate --quantization --offload; do
  grep -q -- "$flag" "$ROOT/ltx_cli_help.txt" || { say "CLI verification missing flag: $flag"; exit 43; }
done
write_status true ready ""
say "NEO_LTX_WORKER_READY: worker API, LTX-2.3 checkpoint, spatial upsampler, Gemma encoder, and CLI verified."
unset HF_TOKEN NEO_WORKER_TOKEN NEO_BUCKET
