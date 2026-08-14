#!/bin/bash
# Serve unsloth/Qwen3.8-27B-NVFP4 at its full 262,144-token context on a single
# RTX 5090 (32 GiB). GPU-only — no CPU offloading.
# Validated 2026-08-14, vLLM 0.27.1. See README.md and CALIBRATION.md.
#
# Usage: MODEL_DIR=/abs/path/... ./serve_qwen38.sh   (defaults shown below)
set -e

MODEL_DIR="${MODEL_DIR:-/home/ayayalar/models/unsloth/Qwen3.8-27B-NVFP4}"
SERVED_NAME="${SERVED_NAME:-unsloth/Qwen3.8-27B-NVFP4}"
PORT="${PORT:-8000}"
VLLM_BIN="${VLLM_BIN:-vllm}"
LOGFILE="${LOGFILE:-/tmp/qwen38_vllm.log}"

exec "$VLLM_BIN" serve "$MODEL_DIR" \
  --host 0.0.0.0 \
  --port "$PORT" \
  --served-model-name "$SERVED_NAME" \
  --max-model-len 262144 \
  --kv-cache-dtype turboquant_4bit_nc \
  --kv-cache-memory-bytes 5368709120 \
  --max-num-seqs 4 \
  --max-num-batched-tokens 512 \
  --gpu-memory-utilization 0.98 \
  --enable-auto-tool-choice \
  --tool-call-parser qwen3_xml \
  --reasoning-parser qwen3 \
  >> "$LOGFILE" 2>&1
