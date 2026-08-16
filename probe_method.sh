#!/bin/bash
# Isolate the discrepancy: same prompt, same server (no allocator), two computation methods.
# Method 1 (what I did before): wall-time / assumed 500.
# Method 2 (official): wall-time / actual usage.completion_tokens.
set -u
Q=/tmp/qbench
$Q/scripts/stop.sh >/dev/null 2>&1; sleep 3
PYTORCH_CUDA_ALLOC_CONF="" MTP=1 $Q/scripts/start.sh >/dev/null 2>&1
for i in $(seq 1 90); do sleep 2; curl -s -o /dev/null -w "%{http_code}" --max-time 4 localhost:8000/v1/models 2>/dev/null | grep -q 200 && break; done
echo "boot ok (MTP=1, no allocator)"
python3 - <<'EOF'
import json, urllib.request, time
URL="http://localhost:8000/v1/chat/completions"
# EXACT prompt I used for the 166.9 measurement
p={"model":"unsloth/Qwen3.8-27B-NVFP4","messages":[{"role":"user","content":"You are terse. Respond with only numbers separated by spaces: 1 2 3 4 ... continue incrementing. Start now."}],"max_tokens":500,"temperature":0.0}
r=urllib.request.Request(URL,data=json.dumps(p).encode(),headers={"Content-Type":"application/json"})
t0=time.time()
with urllib.request.urlopen(r,timeout=600) as resp: d=json.loads(resp.read().decode())
dt=time.time()-t0
u=d["usage"]; n_actual=u["completion_tokens"]
print(f"actual completion_tokens={n_actual} (max_tokens was 500)")
print(f"Method1 (assume 500): {500/dt:.1f} tok/s")
print(f"Method2 (actual {n_actual}): {n_actual/dt:.1f} tok/s")
print(f"reasoning_tokens={u.get('completion_tokens_details',{}).get('reasoning_tokens','n/a')}")
print(f"content_head={repr(d['choices'][0]['message'].get('content'))[:60]}")
EOF
echo "=== RESTORE ==="
$Q/scripts/stop.sh >/dev/null 2>&1; sleep 2
$Q/scripts/start.sh >/dev/null 2>&1; echo restore-issued
