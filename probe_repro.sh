#!/bin/bash
# Reproduction: mnbt=1024 K3 (best sweep arm) — 3x official harness + token detail.
set -u
Q=/tmp/qbench
$Q/scripts/stop.sh >/dev/null 2>&1; sleep 3
PYTORCH_CUDA_ALLOC_CONF=expandable_segments:True nohup ~/unsloth-nvfp4-env/bin/vllm serve ~/models/unsloth/Qwen3.8-27B-NVFP4 \
  --host 0.0.0.0 --port 8000 --served-model-name unsloth/Qwen3.8-27B-NVFP4 \
  --max-model-len 262144 --kv-cache-dtype turboquant_4bit_nc \
  --kv-cache-memory-bytes 5800000000 --max-num-seqs 4 --max-num-batched-tokens 1024 \
  --gpu-memory-utilization 0.98 --attention-config.flash_attn_version=2 \
  --speculative-config '{"method":"mtp","num_speculative_tokens":3,"num_speculative_tokens_per_batch_size":[[1,4,3]]}' \
  --enable-auto-tool-choice --tool-call-parser qwen3_xml --reasoning-parser qwen3 \
  > /tmp/arm_rep.log 2>&1 &
PID=$!
for i in $(seq 1 90); do sleep 2; curl -s -o /dev/null -w "%{http_code}" --max-time 4 localhost:8000/v1/models 2>/dev/null | grep -q 200 && break; done
curl -s -o /dev/null -w "http:%{http_code}\n" --max-time 4 localhost:8000/v1/models || { echo BOOTFAIL; exit 1; }
grep -m1 "GPU KV cache size" /tmp/arm_rep.log
python3 - <<'EOF'
import json, urllib.request, time
URL="http://localhost:8000/v1/chat/completions"
SYS="You are terse. Respond only with repeating numbers separated by spaces, like: 1 2 3 4 5 6 ... Always continue the sequence with the next integer (increment by 1). Start and stay in the sequence, do not stop."
def one():
    p={"model":"unsloth/Qwen3.8-27B-NVFP4","messages":[{"role":"system","content":SYS},{"role":"user","content":"Go"}],"max_tokens":400,"temperature":0.0}
    r=urllib.request.Request(URL,data=json.dumps(p).encode(),headers={"Content-Type":"application/json"})
    t0=time.time()
    with urllib.request.urlopen(r,timeout=600) as resp: d=json.loads(resp.read().decode())
    dt=time.time()-t0
    u=d["usage"]; m=d["choices"][0]["message"]
    return u, dt, m
# warmup
one()
for i in range(3):
    u, dt, m = one()
    comp=u["completion_tokens"]
    print(f"run{i}: {comp} ct in {dt:.2f}s -> {comp/dt:.1f} tok/s | content_head={repr((m.get('content') or '')[:25])} reason_len={len(m.get('reasoning') or '')}")
EOF
echo "=== acceptance:"
grep -oE "Mean acceptance length: [0-9.]+" /tmp/arm_rep.log | tail -2
echo "=== RESTORE ==="
kill $PID 2>/dev/null; sleep 2
$Q/scripts/start.sh >/dev/null 2>&1; echo restored