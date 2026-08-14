#!/bin/bash
# Start Qwen3.8-27B-NVFP4 vLLM server in the background (start/stop lifecycle).
# Validated 2026-08-14, vLLM 0.27.1 — full 262,144-token context on one RTX 5090,
# GPU-only (no CPU offloading). See README.md and CALIBRATION.md.
#
# Usage:  ./start.sh [--model-dir PATH] [--port N] [--served-name NAME]
#   or short: ./start.sh "PATH [PORT]"
set -u

MODEL_DIR="${MODEL_DIR:-/home/ayayalar/models/unsloth/Qwen3.8-27B-NVFP4}"
SERVED_NAME="${SERVED_NAME:-unsloth/Qwen3.8-27B-NVFP4}"
PORT="${PORT:-8000}"
VLLM_BIN="${VLLM_BIN:-/home/ayayalar/unsloth-nvfp4-env/bin/vllm}"
LOGFILE="${LOGFILE:-/tmp/qwen38_vllm.log}"
PIDFILE="${PIDFILE:-/tmp/qwen38_vllm.pid}"

# Positional shorthand: ./start.sh /abs/model/dir [port]
[ $# -ge 1 ] && MODEL_DIR="$1"
[ $# -ge 2 ] && PORT="$2"

if [ -f "$PIDFILE" ] && kill -0 "$(cat "$PIDFILE")" 2>/dev/null; then
  echo "already running (pid $(cat "$PIDFILE")) — stop first if you want to restart"
  exit 0
fi

rm -f "$PIDFILE"
"$VLLM_BIN" serve "$MODEL_DIR" \
  --host 0.0.0.0 \
  --port "$PORT" \
  --served-model-name "$SERVED_NAME" \
  --max-model-len 262144 \
  --kv-cache-dtype turboquant_4bit_nc \
  --kv-cache-memory-bytes 5368709120 \
  --max-num-seqs 4 \
  --max-num-batched-tokens 512 \
  --gpu-memory-utilization 0.98 \
  --attention-config.flash_attn_version=2 \
  --enable-auto-tool-choice \
  --tool-call-parser qwen3_xml \
  --reasoning-parser qwen3 \
  >> "$LOGFILE" 2>&1 &
echo $! > "$PIDFILE"
echo "started pid $(cat "$PIDFILE"), log: $LOGFILE"
