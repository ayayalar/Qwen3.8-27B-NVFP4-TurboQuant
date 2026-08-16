#!/bin/bash
# Start Qwen3.8-27B-NVFP4 vLLM server in the background (start/stop lifecycle).
# Validated 2026-08-14, vLLM 0.27.1 — full 262,144-token context on one RTX 5090,
# GPU-only (no CPU offloading). See README.md and CALIBRATION.md.
#
# Usage:  ./start.sh [--model-dir PATH] [--port N] [--served-name NAME]
#   or short: ./start.sh "PATH [PORT]"
#
# Optional: MTP=1 enables speculative decoding (~3x single-stream decode speed).
#   It MUST be combined with dynamic speculation (num_speculative_tokens_per_batch_size)
#   so vLLM steps cudagraphs from FULL_AND_PIECEWISE to PIECEWISE; on 0.27.1 the
#   FULL-mode spec-verify path corrupts turboquant KV output (see CALIBRATION.md).
#   Requires a slightly larger KV pin (5.8 GiB) to boot at full 262K.
set -u

MODEL_DIR="${MODEL_DIR:-$HOME/models/unsloth/Qwen3.8-27B-NVFP4}"
SERVED_NAME="${SERVED_NAME:-unsloth/Qwen3.8-27B-NVFP4}"
PORT="${PORT:-8000}"
VLLM_BIN="${VLLM_BIN:-$HOME/unsloth-nvfp4-env/bin/vllm}"
LOGFILE="${LOGFILE:-/tmp/qwen38_vllm.log}"
PIDFILE="${PIDFILE:-/tmp/qwen38_vllm.pid}"
MTP="${MTP:-0}"

# Positional shorthand: ./start.sh /abs/model/dir [port]
[ $# -ge 1 ] && MODEL_DIR="$1"
[ $# -ge 2 ] && PORT="$2"

# MTP needs a larger KV pin than the non-spec config (verified empirically:
# K3 draft head needs ~5.02 GiB of KV at 262144, over the 5 GiB default).
KV_BYTES=5368709120
[ "$MTP" = "1" ] && KV_BYTES=5800000000

# The model must already be on disk — this recipe does not auto-fetch.
# Run scripts/setup.sh to download it.
if [ ! -f "$MODEL_DIR/config.json" ] || [ ! -f "$MODEL_DIR/model.safetensors" ]; then
  echo "error: model files not found in $MODEL_DIR" >&2
  echo "  run scripts/setup.sh to download unsloth/Qwen3.8-27B-NVFP4" >&2
  echo "  or set MODEL_DIR to an existing local copy" >&2
  exit 1
fi

if [ -f "$PIDFILE" ] && kill -0 "$(cat "$PIDFILE")" 2>/dev/null; then
  echo "already running (pid $(cat "$PIDFILE")) — stop first if you want to restart"
  exit 0
fi

rm -f "$PIDFILE"
SPEC_ARGS=()
if [ "$MTP" = "1" ]; then
  SPEC_ARGS=(--speculative-config '{"method":"mtp","num_speculative_tokens":3,"num_speculative_tokens_per_batch_size":[[1,4,3]]}')
fi
"$VLLM_BIN" serve "$MODEL_DIR" \
  --host 0.0.0.0 \
  --port "$PORT" \
  --served-model-name "$SERVED_NAME" \
  --max-model-len 262144 \
  --kv-cache-dtype turboquant_4bit_nc \
  --kv-cache-memory-bytes "$KV_BYTES" \
  --max-num-seqs 4 \
  --max-num-batched-tokens 512 \
  --gpu-memory-utilization 0.98 \
  --attention-config.flash_attn_version=2 \
  "${SPEC_ARGS[@]}" \
  --enable-auto-tool-choice \
  --tool-call-parser qwen3_xml \
  --reasoning-parser qwen3 \
  >> "$LOGFILE" 2>&1 &
echo $! > "$PIDFILE"
echo "started pid $(cat "$PIDFILE"), log: $LOGFILE"
