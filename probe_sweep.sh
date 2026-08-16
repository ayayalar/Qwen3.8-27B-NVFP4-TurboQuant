#!/bin/bash
# Sweep decode-acceleration knobs under MTP: max-num-batched-tokens and K.
# Every arm: boot -> speed_test (official harness) -> acceptance -> math sanity.
set -u
Q=/tmp/qbench
cd $Q

arm() {
  local label="$1" mnbt="$2" K="$3"
  echo "########## ARM $label: mnbt=$mnbt K=$K ##########"
  $Q/scripts/stop.sh >/dev/null 2>&1; sleep 3
  PYTORCH_CUDA_ALLOC_CONF=expandable_segments:True nohup ~/unsloth-nvfp4-env/bin/vllm serve ~/models/unsloth/Qwen3.8-27B-NVFP4 \
    --host 0.0.0.0 --port 8000 --served-model-name unsloth/Qwen3.8-27B-NVFP4 \
    --max-model-len 262144 --kv-cache-dtype turboquant_4bit_nc \
    --kv-cache-memory-bytes 5800000000 --max-num-seqs 4 --max-num-batched-tokens $mnbt \
    --gpu-memory-utilization 0.98 --attention-config.flash_attn_version=2 \
    --speculative-config "{\"method\":\"mtp\",\"num_speculative_tokens\":$K,\"num_speculative_tokens_per_batch_size\":[[1,4,$K]]}" \
    --enable-auto-tool-choice --tool-call-parser qwen3_xml --reasoning-parser qwen3 \
    > /tmp/arm_$label.log 2>&1 &
  local PID=$!
  for i in $(seq 1 90); do sleep 2; curl -s -o /dev/null -w "%{http_code}" --max-time 4 localhost:8000/v1/models 2>/dev/null | grep -q 200 && break; done
  curl -s -o /dev/null -w "http:%{http_code}\n" --max-time 4 localhost:8000/v1/models || { echo "  BOOT FAIL"; tail -2 /tmp/arm_$label.log; kill $PID 2>/dev/null; return; }
  grep -oE "Overriding cudagraph[^\n]*" /tmp/arm_$label.log | head -1
  grep -m1 "GPU KV cache size" /tmp/arm_$label.log
  # official single-stream (counting prompt) + warm = the repo harness
  cd $Q/benchmark && python3 speed_test.py 2>&1 | grep -E "measured|CONCURRENT"
  cd $Q
  # acceptance
  grep -oE "Mean acceptance length: [0-9.]+" /tmp/arm_$label.log | tail -1
  # correctness sanity (math) — catch a garble regression
  python3 - <<'EOF'
import json, urllib.request
p={"model":"unsloth/Qwen3.8-27B-NVFP4","messages":[{"role":"user","content":"What is 2+2? Answer with just the number."}],"max_tokens":80,"temperature":0.0}
r=urllib.request.Request("http://localhost:8000/v1/chat/completions",data=json.dumps(p).encode(),headers={"Content-Type":"application/json"})
try:
    d=json.loads(urllib.request.urlopen(r,timeout=120).read())
    m=d["choices"][0]["message"]
    print("  math:", repr((m.get("content") or "")[-20:]))
except Exception as e: print("  math ERR", e)
EOF
  kill $PID 2>/dev/null; sleep 2
  echo ""
}

# baseline = shipped default
arm "default_k3_m512" 512 3
# mnbt sweep at K3 (vLLM warned 512 throttles spec decode)
arm "k3_m1024" 1024 3
arm "k3_m2048" 2048 3
arm "k3_m4096" 4096 3
# K sweeps at the best-looking mnbt
arm "k4_m2048" 2048 4
arm "k5_m2048" 2048 5

echo "=== RESTORE ==="
$Q/scripts/stop.sh >/dev/null 2>&1; sleep 2
$Q/scripts/start.sh >/dev/null 2>&1; echo restore-to-shipped-default
