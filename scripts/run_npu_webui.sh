#!/usr/bin/env bash
set -euo pipefail

SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
PROJECT_ROOT="$(cd "$SCRIPT_DIR/.." && pwd)"
LGM_DATA_DIR="${LGM_DATA_DIR:-/tmp/lgm}"

export ASCEND_RT_VISIBLE_DEVICES="${ASCEND_RT_VISIBLE_DEVICES:-4,5,6,7}"
export HF_HOME="${HF_HOME:-$LGM_DATA_DIR/cache/huggingface}"
export HUGGINGFACE_HUB_CACHE="${HUGGINGFACE_HUB_CACHE:-$HF_HOME/hub}"
export TRANSFORMERS_CACHE="${TRANSFORMERS_CACHE:-$HF_HOME/transformers}"
export TORCH_HOME="${TORCH_HOME:-$LGM_DATA_DIR/cache/torch}"
export XDG_CACHE_HOME="${XDG_CACHE_HOME:-$LGM_DATA_DIR/cache/xdg}"
export U2NET_HOME="${U2NET_HOME:-$LGM_DATA_DIR/cache/rembg}"
export GRADIO_TEMP_DIR="${GRADIO_TEMP_DIR:-$LGM_DATA_DIR/tmp/gradio}"
export TMPDIR="${TMPDIR:-$LGM_DATA_DIR/tmp}"
export HOME="${LGM_HOME:-$LGM_DATA_DIR/home}"
export PYTHONDONTWRITEBYTECODE="${PYTHONDONTWRITEBYTECODE:-1}"
export PYTHONPYCACHEPREFIX="${PYTHONPYCACHEPREFIX:-$LGM_DATA_DIR/pycache}"
export ASCEND_WORK_PATH="${ASCEND_WORK_PATH:-$LGM_DATA_DIR/ascend/work}"
export ASCEND_PROCESS_LOG_PATH="${ASCEND_PROCESS_LOG_PATH:-$LGM_DATA_DIR/ascend/log}"
export ASCEND_GLOBAL_LOG_LEVEL="${ASCEND_GLOBAL_LOG_LEVEL:-3}"
export LGM_MVDREAM_MODEL="${LGM_MVDREAM_MODEL:-$LGM_DATA_DIR/models/mvdream-sd2.1-diffusers}"
export LGM_IMAGEDREAM_MODEL="${LGM_IMAGEDREAM_MODEL:-$LGM_DATA_DIR/models/imagedream-ipmv-diffusers}"
export LGM_CHECKPOINT="${LGM_CHECKPOINT:-$LGM_DATA_DIR/models/model_fp16_fixrot.safetensors}"

mkdir -p \
  "$HOME" \
  "$TMPDIR" \
  "$HF_HOME" \
  "$HUGGINGFACE_HUB_CACHE" \
  "$TRANSFORMERS_CACHE" \
  "$TORCH_HOME" \
  "$XDG_CACHE_HOME" \
  "$U2NET_HOME" \
  "$GRADIO_TEMP_DIR" \
  "$PYTHONPYCACHEPREFIX" \
  "$ASCEND_WORK_PATH" \
  "$ASCEND_PROCESS_LOG_PATH" \
  "$LGM_DATA_DIR/workspace"

cd "$PROJECT_ROOT"

if [[ -d "$LGM_MVDREAM_MODEL/unet" ]]; then
  cp mvdream/mv_unet_text.py "$LGM_MVDREAM_MODEL/unet/mv_unet.py"
fi
if [[ -d "$LGM_IMAGEDREAM_MODEL/unet" ]]; then
  cp mvdream/mv_unet.py "$LGM_IMAGEDREAM_MODEL/unet/mv_unet.py"
fi
rm -rf "$HF_HOME/modules/diffusers_modules/local"

if [[ -n "${LGM_VENV:-}" && -f "$LGM_VENV/bin/activate" ]]; then
  source "$LGM_VENV/bin/activate"
fi

python app.py big \
  --resume "$LGM_CHECKPOINT" \
  --workspace "$LGM_DATA_DIR/workspace" \
  "$@"
