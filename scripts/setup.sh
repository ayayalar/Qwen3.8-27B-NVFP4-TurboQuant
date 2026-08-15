#!/bin/bash
# One-time bootstrap for this recipe: create the vLLM venv AND download the
# model, skipping anything that's already in place. Idempotent — safe to
# re-run; it only does work that hasn't been done yet.
#
# Defaults mirror start.sh ($HOME-relative, portable). Override with the
# ENV_DIR / MODEL_DIR env vars, or positional args.
#
# Usage: ./setup.sh [ENV_DIR [MODEL_DIR]]
#   e.g.  ./setup.sh
#         ./setup.sh ~/venvs/qwen38 ~/models/Qwen3.8-27B-NVFP4
set -u

ENV_DIR="${ENV_DIR:-$HOME/unsloth-nvfp4-env}"
MODEL_DIR="${MODEL_DIR:-$HOME/models/unsloth/Qwen3.8-27B-NVFP4}"
[ $# -ge 1 ] && ENV_DIR="$1"
[ $# -ge 2 ] && MODEL_DIR="$2"

VLLM_BIN="$ENV_DIR/bin/vllm"
REPO="unsloth/Qwen3.8-27B-NVFP4"
PINNED=( "vllm>=0.25.0,<0.28" "flashinfer-python>=0.6.13" "nvidia-cutlass-dsl>=4.5.2" )

# ---- 1. vLLM venv -----------------------------------------------------------
if [ -x "$VLLM_BIN" ]; then
  echo "env already present: $ENV_DIR — skipping install"
else
  echo "creating venv at $ENV_DIR ..."
  if command -v uv >/dev/null 2>&1; then
    uv venv "$ENV_DIR" || exit 1
    uv pip install --python "$ENV_DIR/bin/python" "${PINNED[@]}" --torch-backend=auto \
      || { echo "env install failed"; exit 1; }
  else
    echo "uv not found — falling back to python3 -m venv + pip"
    python3 -m venv "$ENV_DIR" || exit 1
    "$ENV_DIR/bin/pip" install -U pip >/dev/null || exit 1
    "$ENV_DIR/bin/pip" install "${PINNED[@]}" || { echo "env install failed"; exit 1; }
  fi
  [ -x "$VLLM_BIN" ] || { echo "error: $VLLM_BIN was not created"; exit 1; }
  echo "env ready at $ENV_DIR"
fi

# ---- 2. model weights -------------------------------------------------------
if [ -f "$MODEL_DIR/config.json" ] && [ -f "$MODEL_DIR/model.safetensors" ]; then
  echo "model already present at $MODEL_DIR — skipping download"
else
  mkdir -p "$MODEL_DIR" || { echo "cannot create $MODEL_DIR"; exit 1; }
  if command -v hf >/dev/null 2>&1; then
    DOWNLOADER=(hf download "$REPO" --local-dir "$MODEL_DIR")
  elif command -v huggingface-cli >/dev/null 2>&1; then
    DOWNLOADER=(huggingface-cli download "$REPO" --local-dir "$MODEL_DIR")
  else
    echo "error: neither 'hf' nor 'huggingface-cli' found."
    echo "install the CLI first, e.g.:  uv pip install 'huggingface_hub[cli]'"
    exit 1
  fi
  echo "downloading $REPO (~22 GiB) to $MODEL_DIR ..."
  "${DOWNLOADER[@]}" || { echo "download failed"; exit 1; }
  [ -f "$MODEL_DIR/config.json" ] && [ -f "$MODEL_DIR/model.safetensors" ] \
    || { echo "error: download reported success but weights are missing"; exit 1; }
  echo "model ready at $MODEL_DIR"
fi

echo
echo "bootstrap complete. start the server with:  ./scripts/start.sh"
