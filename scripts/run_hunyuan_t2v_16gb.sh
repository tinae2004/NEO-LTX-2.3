#!/usr/bin/env bash
# NEO HunyuanVideo 1.5 — 16GB-class RTX text-to-video smoke test.
# Run on a rented Vast.ai GPU. This script never calls the Vast API.
set -Eeuo pipefail

PROMPT="${1:-}"
if [[ -z "$PROMPT" ]]; then
  echo "Usage: $0 'Describe the video you want to generate'"
  exit 2
fi

NEO_DATA_DIR="${NEO_DATA_DIR:-/workspace/neo-data}"
REPO_DIR="$NEO_DATA_DIR/repos/HunyuanVideo-1.5"
MODEL_DIR="$NEO_DATA_DIR/models/hunyuanvideo15/ckpts"
OUTPUT_DIR="$NEO_DATA_DIR/outputs/hunyuan"
VIDEO_LENGTH="${VIDEO_LENGTH:-145}"      # 145 frames ~= 6 seconds at 24 fps
INFERENCE_STEPS="${INFERENCE_STEPS:-50}"
SEED="${SEED:-123}"

mkdir -p "$NEO_DATA_DIR/repos" "$MODEL_DIR" "$OUTPUT_DIR"

# Use an isolated virtual environment: Debian's system pip is externally
# managed and must never be upgraded/uninstalled by this workflow. Preserve
# preinstalled GPU packages (especially torch) via --system-site-packages.
SYSTEM_PYTHON="$(command -v python3 || command -v python || true)"
if [[ -z "$SYSTEM_PYTHON" ]]; then
  echo "Python is missing; installing Python 3 and venv support..."
  apt-get update
  DEBIAN_FRONTEND=noninteractive apt-get install -y python3 python3-venv python3-pip
  SYSTEM_PYTHON="$(command -v python3)"
fi
VENV_DIR="$NEO_DATA_DIR/venv"
if [[ ! -x "$VENV_DIR/bin/python" ]]; then
  if ! "$SYSTEM_PYTHON" -m venv --system-site-packages "$VENV_DIR"; then
    echo "Installing missing Python venv support..."
    apt-get update
    DEBIAN_FRONTEND=noninteractive apt-get install -y python3-venv
    "$SYSTEM_PYTHON" -m venv --system-site-packages "$VENV_DIR"
  fi
fi
PYTHON_BIN="$VENV_DIR/bin/python"
echo "Using isolated Python environment: $PYTHON_BIN"

echo "== NEO HunyuanVideo 1.5 / 16GB T2V =="
echo "Data directory: $NEO_DATA_DIR"
echo "Output directory: $OUTPUT_DIR"

command -v nvidia-smi >/dev/null || { echo "ERROR: NVIDIA driver/nvidia-smi unavailable."; exit 1; }
GPU_NAME="$(nvidia-smi --query-gpu=name --format=csv,noheader | head -n1)"
GPU_VRAM_MB="$(nvidia-smi --query-gpu=memory.total --format=csv,noheader,nounits | head -n1)"
echo "GPU: $GPU_NAME"
echo "VRAM: $GPU_VRAM_MB MiB"

if [[ "$GPU_VRAM_MB" -lt 14336 ]]; then
  echo "ERROR: HunyuanVideo 1.5 documents a 14GB minimum with model offloading."
  echo "This GPU reports less than 14GiB. Refusing to launch."
  exit 1
fi

RAM_GB="$(awk '/MemTotal/ {printf "%d", $2/1024/1024}' /proc/meminfo)"
echo "System RAM: ${RAM_GB} GiB"
if [[ "$RAM_GB" -lt 48 ]]; then
  echo "WARNING: Less than 48GiB system RAM. CPU offloading may run out of memory."
fi

"$PYTHON_BIN" - <<'PY'
import sys
if sys.version_info < (3, 10):
    raise SystemExit("Python 3.10+ is required.")
print("Python:", sys.version.split()[0])
PY

if [[ ! -d "$REPO_DIR/.git" ]]; then
  echo "Cloning official Tencent HunyuanVideo-1.5 source..."
  git clone --depth 1 https://github.com/Tencent-Hunyuan/HunyuanVideo-1.5.git "$REPO_DIR"
else
  echo "Using existing HunyuanVideo source checkout: $REPO_DIR"
fi

cd "$REPO_DIR"

# Install runtime dependencies into the active worker environment.
"$PYTHON_BIN" -m pip install --upgrade pip
"$PYTHON_BIN" -m pip install -r requirements.txt
"$PYTHON_BIN" -m pip install -U "huggingface_hub==0.34.0" modelscope tencentcloud-sdk-python

# Download checkpoints only when needed. Hugging Face downloads resume if interrupted.
if [[ ! -d "$MODEL_DIR/transformer" && ! -f "$MODEL_DIR/.neo_hunyuan_base_downloaded" ]]; then
  echo "Downloading HunyuanVideo-1.5 checkpoints..."
  "$PYTHON_BIN" -c 'from huggingface_hub import snapshot_download; import sys; snapshot_download(repo_id=sys.argv[1], local_dir=sys.argv[2])' tencent/HunyuanVideo-1.5 "$MODEL_DIR"
  touch "$MODEL_DIR/.neo_hunyuan_base_downloaded"
else
  echo "Hunyuan base checkpoint folder already exists; skipping full download."
fi

mkdir -p "$MODEL_DIR/text_encoder/llm" "$MODEL_DIR/text_encoder/byt5-small" "$MODEL_DIR/text_encoder/Glyph-SDXL-v2"

if [[ ! -f "$MODEL_DIR/text_encoder/llm/config.json" ]]; then
  echo "Downloading Qwen2.5-VL-7B text encoder..."
  "$PYTHON_BIN" -c 'from huggingface_hub import snapshot_download; import sys; snapshot_download(repo_id=sys.argv[1], local_dir=sys.argv[2])' Qwen/Qwen2.5-VL-7B-Instruct "$MODEL_DIR/text_encoder/llm"
fi

if [[ ! -f "$MODEL_DIR/text_encoder/byt5-small/config.json" ]]; then
  echo "Downloading ByT5 text encoder..."
  "$PYTHON_BIN" -c 'from huggingface_hub import snapshot_download; import sys; snapshot_download(repo_id=sys.argv[1], local_dir=sys.argv[2])' google/byt5-small "$MODEL_DIR/text_encoder/byt5-small"
fi

if [[ ! -d "$MODEL_DIR/text_encoder/Glyph-SDXL-v2/checkpoints" ]]; then
  echo "Downloading Glyph-SDXL-v2 text encoder assets..."
  modelscope download --model AI-ModelScope/Glyph-SDXL-v2 --local_dir "$MODEL_DIR/text_encoder/Glyph-SDXL-v2"
fi

export PYTORCH_CUDA_ALLOC_CONF="${PYTORCH_CUDA_ALLOC_CONF:-expandable_segments:True,max_split_size_mb:128}"
export TOKENIZERS_PARALLELISM=false

OUTPUT_PATH="$OUTPUT_DIR/hunyuan_t2v_$(date -u +%Y%m%dT%H%M%S).mp4"

echo "Starting 480p text-to-video generation..."
echo "Prompt: $PROMPT"
echo "Frames: $VIDEO_LENGTH (~6 seconds at 24 fps) | Steps: $INFERENCE_STEPS"
echo "Offloading: enabled | Super-resolution: disabled | Prompt rewrite: disabled"

if command -v torchrun >/dev/null 2>&1; then
  TORCHRUN=(torchrun)
else
  TORCHRUN=("$PYTHON_BIN" -m torch.distributed.run)
fi

"${TORCHRUN[@]}" --nproc_per_node=1 generate.py \
  --prompt "$PROMPT" \
  --image_path none \
  --resolution 480p \
  --aspect_ratio 16:9 \
  --video_length "$VIDEO_LENGTH" \
  --num_inference_steps "$INFERENCE_STEPS" \
  --seed "$SEED" \
  --rewrite false \
  --cfg_distilled false \
  --offloading true \
  --group_offloading true \
  --overlap_group_offloading false \
  --sr false \
  --dtype bf16 \
  --output_path "$OUTPUT_PATH" \
  --model_path "$MODEL_DIR"

test -s "$OUTPUT_PATH" || { echo "ERROR: Generation finished but output file is missing/empty."; exit 1; }
echo "SUCCESS: $OUTPUT_PATH"
nvidia-smi
