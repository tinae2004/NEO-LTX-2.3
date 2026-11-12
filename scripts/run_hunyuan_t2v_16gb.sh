#!/usr/bin/env bash
# NEO HunyuanVideo 1.5 — 16GB-class RTX text-to-video smoke test.
# Run on a rented Vast.ai GPU. This script never calls the Vast API.
set -Eeuo pipefail

# Resolve helper scripts before changing into the upstream Hunyuan repository.
SCRIPT_DIR="$(cd -- "$(dirname -- "${BASH_SOURCE[0]}")" && pwd)"

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

# Reuse the CUDA-enabled PyTorch stack baked into the Vast PyTorch image
# when it is already compatible; fall back to the upstream requirements only
# when the base image does not provide a usable CUDA stack.
echo "[1/7] CHECKING PREINSTALLED PYTORCH/CUDA STACK..."
if "$PYTHON_BIN" - <<'PY'
import sys
try:
    import torch
    import torchvision
    import torchaudio
    from packaging.version import Version
    version = Version(torch.__version__.split("+")[0])
    if version < Version("2.6.0") or not torch.cuda.is_available():
        raise RuntimeError(f"PyTorch {torch.__version__}, CUDA available={torch.cuda.is_available()}")
    print(f"PREINSTALLED_TORCH={torch.__version__}", flush=True)
    print(f"PREINSTALLED_TORCHVISION={torchvision.__version__}", flush=True)
    print(f"PREINSTALLED_TORCHAUDIO={torchaudio.__version__}", flush=True)
    print(f"PYTORCH_CUDA={torch.version.cuda}", flush=True)
    print(f"GPU={torch.cuda.get_device_name(0)}", flush=True)
except Exception as exc:
    print(f"PREINSTALLED_CUDA_STACK_NOT_USABLE: {type(exc).__name__}: {exc}", file=sys.stderr, flush=True)
    sys.exit(1)
PY
then
  echo "PREINSTALLED_CUDA_STACK_VERIFIED — preserving Torch/CUDA/cuDNN wheels."
  FILTERED_REQUIREMENTS="$NEO_DATA_DIR/requirements-hunyuan-no-torch.txt"
  "$PYTHON_BIN" - "$REPO_DIR/requirements.txt" "$FILTERED_REQUIREMENTS" <<'PY'
import re
import sys
from pathlib import Path

source, target = map(Path, sys.argv[1:])
skip = {"torch", "torchvision", "torchaudio"}
kept = []
removed = []
for line in source.read_text().splitlines():
    stripped = line.strip()
    if not stripped or stripped.startswith("#"):
        kept.append(line)
        continue
    name = re.split(r"[<=>!~\[]", stripped, maxsplit=1)[0].strip().lower().replace("_", "-")
    if name in skip:
        removed.append(stripped)
    else:
        kept.append(line)
target.write_text("\n".join(kept) + "\n")
print("CUDA_PACKAGES_REUSED=" + ", ".join(removed), flush=True)
print(f"NON_CUDA_REQUIREMENTS={sum(bool(x.strip()) and not x.lstrip().startswith('#') for x in kept)}", flush=True)
PY
  "$PYTHON_BIN" -m pip install --progress-bar on -r "$FILTERED_REQUIREMENTS"
else
  echo "FALLBACK: Base image lacks a working compatible CUDA stack; installing upstream pinned requirements."
  "$PYTHON_BIN" -m pip install --progress-bar on -r requirements.txt
fi
"$PYTHON_BIN" -m pip install --progress-bar on tencentcloud-sdk-python
"$PYTHON_BIN" - <<'PY'
import torch
if not torch.cuda.is_available():
    raise SystemExit("ERROR: CUDA is still unavailable after dependency setup.")
print(f"CUDA_STACK_READY torch={torch.__version__} cuda={torch.version.cuda} gpu={torch.cuda.get_device_name(0)}", flush=True)
PY
echo "DEPENDENCIES_INSTALL_COMPLETE"

# Push Xet downloads harder on high-bandwidth Vast hosts.
# High-performance mode enables larger buffers and parallel transfers;
# fixed download concurrency bypasses Xet's conservative adaptive ramp-up.
export HF_XET_HIGH_PERFORMANCE=1
export HF_XET_FIXED_DOWNLOAD_CONCURRENCY=32
export HF_HUB_DOWNLOAD_TIMEOUT=180
echo "HF_XET_HIGH_PERFORMANCE=$HF_XET_HIGH_PERFORMANCE"
echo "HF_XET_FIXED_DOWNLOAD_CONCURRENCY=$HF_XET_FIXED_DOWNLOAD_CONCURRENCY"
"$PYTHON_BIN" - <<'PY'
try:
    import hf_xet
    print(f"HF_XET_VERSION={getattr(hf_xet, '__version__', 'installed')}", flush=True)
except Exception as exc:
    raise SystemExit(f"ERROR: hf_xet is not installed; refusing slow fallback downloader: {exc}")
PY

# Download only the model components used by this 480p text-to-video run.
# The complete upstream transformer directory is hundreds of GB and includes
# unrelated I2V, 720p, SR and distilled checkpoints.
T2V_WEIGHT="$MODEL_DIR/transformer/480p_t2v/diffusion_pytorch_model.safetensors"
VAE_WEIGHT="$MODEL_DIR/vae/diffusion_pytorch_model.safetensors"
if [[ ! -s "$T2V_WEIGHT" || ! -s "$VAE_WEIGHT" || ! -f "$MODEL_DIR/transformer/480p_t2v/config.json" ]]; then
  echo "[2/7] DOWNLOADING ONLY HUNYUANVIDEO-1.5 480P T2V + VAE CHECKPOINTS..."
  "$PYTHON_BIN" "$SCRIPT_DIR/hf_download_with_progress.py" tencent/HunyuanVideo-1.5 "$MODEL_DIR" \
    --max-workers 16 \
    --allow config.json model_index.json 'scheduler/**' 'transformer/480p_t2v/**' 'vae/**'
  test -s "$T2V_WEIGHT" || { echo "ERROR: 480p T2V transformer checkpoint is missing/empty."; exit 1; }
  test -s "$VAE_WEIGHT" || { echo "ERROR: Hunyuan VAE checkpoint is missing/empty."; exit 1; }
  test -f "$MODEL_DIR/transformer/480p_t2v/config.json" || { echo "ERROR: 480p T2V transformer config is missing."; exit 1; }
  touch "$MODEL_DIR/.neo_hunyuan_base_downloaded"
  echo "HUNYUAN_480P_T2V_BASE_ASSETS_VERIFIED"
else
  echo "Hunyuan 480p T2V transformer and VAE already exist; skipping download."
fi

mkdir -p "$MODEL_DIR/text_encoder/llm" "$MODEL_DIR/text_encoder/byt5-small" "$MODEL_DIR/text_encoder/Glyph-SDXL-v2"

if [[ ! -f "$MODEL_DIR/text_encoder/llm/config.json" ]]; then
  echo "[3/7] DOWNLOADING QWEN2.5-VL-7B TEXT ENCODER..."
  "$PYTHON_BIN" "$SCRIPT_DIR/hf_download_with_progress.py" Qwen/Qwen2.5-VL-7B-Instruct "$MODEL_DIR/text_encoder/llm"
fi

if [[ ! -f "$MODEL_DIR/text_encoder/byt5-small/config.json" ]]; then
  echo "[4/7] DOWNLOADING BYT5 TEXT ENCODER..."
  "$PYTHON_BIN" "$SCRIPT_DIR/hf_download_with_progress.py" google/byt5-small "$MODEL_DIR/text_encoder/byt5-small"
fi

if [[ ! -d "$MODEL_DIR/text_encoder/Glyph-SDXL-v2/checkpoints" ]]; then
  echo "[5/7] DOWNLOADING GLYPH-SDXL-V2 ASSETS..."
  modelscope download --model AI-ModelScope/Glyph-SDXL-v2 --local_dir "$MODEL_DIR/text_encoder/Glyph-SDXL-v2"
  echo "GLYPH_DOWNLOAD_COMPLETE"
fi

export PYTORCH_CUDA_ALLOC_CONF="${PYTORCH_CUDA_ALLOC_CONF:-expandable_segments:True,max_split_size_mb:128}"
export TOKENIZERS_PARALLELISM=false

OUTPUT_PATH="$OUTPUT_DIR/hunyuan_t2v_$(date -u +%Y%m%dT%H%M%S).mp4"

echo "[6/7] MODEL ASSETS READY — STARTING 480P VIDEO GENERATION..."
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

echo "[7/7] VERIFYING GENERATED VIDEO..."
test -s "$OUTPUT_PATH" || { echo "ERROR: Generation finished but output file is missing/empty."; exit 1; }
echo "SUCCESS: $OUTPUT_PATH"
stat -c 'VIDEO_FILE_SIZE_BYTES=%s' "$OUTPUT_PATH"
nvidia-smi
