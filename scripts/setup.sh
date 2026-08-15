#!/bin/bash
# Download unsloth/Qwen3.8-27B-NVFP4 if not already present.
# Idempotent: skips silently when the model files already exist.
# Uses the same MODEL_DIR default as start.sh; override with MODEL_DIR env var.
#
# Usage: ./setup.sh [MODEL_DIR]
set -u

MODEL_DIR="${MODEL_DIR:-$HOME/models/unsloth/Qwen3.8-27B-NVFP4}"
[ $# -ge 1 ] && MODEL_DIR="$1"

if [ -f "$MODEL_DIR/config.json" ] && [ -f "$MODEL_DIR/model.safetensors" ]; then
  echo "model already present at $MODEL_DIR — nothing to do"
  exit 0
fi

mkdir -p "$MODEL_DIR" || { echo "cannot create $MODEL_DIR"; exit 1; }

if command -v hf >/dev/null 2>&1; then
  DOWNLOADER=(hf download unsloth/Qwen3.8-27B-NVFP4 --local-dir "$MODEL_DIR")
elif command -v huggingface-cli >/dev/null 2>&1; then
  DOWNLOADER=(huggingface-cli download unsloth/Qwen3.8-27B-NVFP4 --local-dir "$MODEL_DIR")
else
  echo "error: neither 'hf' nor 'huggingface-cli' found."
  echo "install the CLI first, e.g.:  uv pip install 'huggingface_hub[cli]'"
  exit 1
fi

echo "downloading unsloth/Qwen3.8-27B-NVFP4 (~22 GiB) to $MODEL_DIR ..."
"${DOWNLOADER[@]}" || { echo "download failed"; exit 1; }

if [ -f "$MODEL_DIR/config.json" ] && [ -f "$MODEL_DIR/model.safetensors" ]; then
  echo "done: $MODEL_DIR"
else
  echo "error: download reported success but files are missing — check $MODEL_DIR"
  exit 1
fi
