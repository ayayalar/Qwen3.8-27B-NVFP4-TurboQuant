#!/bin/bash
# A/B: MTP K3 vs K2 (dynamic spec), same everything else, official speed_test + acceptance.
set -u
Q=/tmp/qbench
cd $Q

run_arm() {
  local label="$1" spec_cfg="$2"
  echo "########## ARM: $label ##########"
  $Q/scripts/stop.sh >/dev/null 2>&1
  sleep 3
  # launch with explicit spec config (hand-built, not via start.sh MTP to vary K)
  PYTORCH_CUDA_ALLOC_CONF="expandable_segments:True" nohup ~/unsloth-nvfp4-env/bin/vllm serve ~/models/unsloth/Qwen3.8-27B-NVFP4 \
    --host 0.0.0.0 --port 8000 --served-model-name unsloth/Qwen3.8-27B-NVFP4 \
    --max-model-len 262144 --kv-cache-dtype turboquant_4bit_nc \
    --kv-cache-memory-bytes 5800000000 --max-num-seqs 4 --max-num-batched-tokens 512 \
    --gpu-memory-utilization 0.98 --attention-config.flash_attn_version=2 \
    --speculative-config "$spec_cfg" \
    --enable-auto-tool-choice --tool-call-parser qwen3_xml --reasoning-parser qwen3 \
    > /tmp/ab_$label.log 2>&1 &
  local PID=$!
  for i in $(seq 1 90); do sleep 2; curl -s -o /dev/null -w "%{http_code}" --max-time 4 localhost:8000/v1/models 2>/dev/null | grep -q 200 && break; done
  curl -s -o /dev/null -w "http:%{http_code}\n" --max-time 4 localhost:8000/v1/models || { echo DOWN; return; }
  grep -m1 "GPU KV cache size" /tmp/ab_$label.log
  echo "--- warmup + official speed_test:"
  cd $Q/benchmark && python3 speed_test.py 2>&1 | grep -E "measured|CONCURRENT"
  echo "--- acceptance ($label):"
  grep -oE "Mean acceptance length: [0-9.]+" /tmp/ab_$label.log | tail -1
  grep -o "Accepted: [0-9]* tokens, Drafted: [0-9]*" /tmp/ab_$label.log | tail -1
  kill $PID 2>/dev/null; sleep 2
  echo ""
}

run_arm "K3" '{"method":"mtp","num_speculative_tokens":3,"num_speculative_tokens_per_batch_size":[[1,4,3]]}'
run_arm "K2" '{"method":"mtp","num_speculative_tokens":2,"num_speculative_tokens_per_batch_size":[[1,4,2]]}'

echo "=== RESTORE DEFAULT ==="
$Q/scripts/stop.sh >/dev/null 2>&1; sleep 2
$Q/scripts/start.sh >/dev/null 2>&1; echo restore-issued
