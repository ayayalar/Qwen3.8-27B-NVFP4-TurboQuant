#!/bin/bash
# Test expandable_segments workaround for the 196K-window OOM under MTP=1.
# Config: MTP=1 (dynamic spec) + PYTORCH_CUDA_ALLOC_CONF=expandable_segments:True
set -u
Q=/tmp/qbench
# stop any running server (default recipe)
$Q/scripts/stop.sh >/dev/null 2>&1
sleep 3
# Start MTP=1 with the allocator env var; keep the exact K3 dynamic config from start.sh
MTP=1 PYTORCH_CUDA_ALLOC_CONF=expandable_segments:True ~/unsloth-nvfp4-env/bin/vllm serve ~/models/unsloth/Qwen3.8-27B-NVFP4 \
  --host 0.0.0.0 --port 8000 --served-model-name unsloth/Qwen3.8-27B-NVFP4 \
  --max-model-len 262144 --kv-cache-dtype turboquant_4bit_nc \
  --kv-cache-memory-bytes 5800000000 --max-num-seqs 4 --max-num-batched-tokens 512 \
  --gpu-memory-utilization 0.98 --attention-config.flash_attn_version=2 \
  --speculative-config '{"method":"mtp","num_speculative_tokens":3,"num_speculative_tokens_per_batch_size":[[1,4,3]]}' \
  --enable-auto-tool-choice --tool-call-parser qwen3_xml --reasoning-parser qwen3 \
  > /tmp/mtp_exp.log 2>&1 &
MTPPID=$!
for i in $(seq 1 90); do sleep 2; curl -s -o /dev/null -w "%{http_code}" --max-time 4 localhost:8000/v1/models 2>/dev/null | grep -q 200 && break; done
echo "=== boot:"
curl -s -o /dev/null -w "http:%{http_code}\n" --max-time 4 localhost:8000/v1/models || { echo DOWN; exit 1; }
grep -m1 "GPU KV cache size" /tmp/mtp_exp.log
echo "=== needles: 131K then 196K (the OOM trigger) ==="
python3 - <<'EOF'
import json, urllib.request, random, time
URL="http://localhost:8000/v1/chat/completions"
def call(content, mt=150, temp=0.0, label=""):
    p={"model":"unsloth/Qwen3.8-27B-NVFP4","messages":[{"role":"user","content":content}],"max_tokens":mt,"temperature":temp}
    r=urllib.request.Request(URL,data=json.dumps(p).encode(),headers={"Content-Type":"application/json"})
    t0=time.time()
    try:
        with urllib.request.urlopen(r,timeout=600) as resp: d=json.loads(resp.read().decode())
    except Exception as e:
        print(f"[{label}] REQ ERR {e}"); return ""
    m=d["choices"][0]["message"]
    c=(m.get("content") or ""); rr=(m.get("reasoning") or "")
    print(f"[{label}] {time.time()-t0:.1f}s content={c[:60]!r} reason_len={len(rr)}")
    return c+rr
random.seed(7)
w="alpha beta gamma delta epsilon".split()
words=lambda n:" ".join(random.choice(w) for _ in range(n))
doc=words(65536)+" NEEDLE-XY78 engaged "+words(65536)
print("=== 131K needle ===")
print("  XY78 found:", "NEEDLE-XY78" in call(doc+"\n\nReturn ONLY the marker sentence starting with NEEDLE-.", mt=150, label="131K"))
doc=words(98304)+" NEEDLE-ZZ99 locked "+words(98304)
print("=== 196K needle (was OOM) ===")
print("  ZZ99 found:", "NEEDLE-ZZ99" in call(doc+"\n\nReturn ONLY the marker sentence starting with NEEDLE-.", mt=150, label="196K"))
EOF
echo "=== engine still alive after 196K? ==="
curl -s -o /dev/null -w "http:%{http_code}\n" --max-time 4 localhost:8000/v1/models || echo DOWN
grep -iE "OutOfMemory|OOM|illegal memory" /tmp/mtp_exp.log | head -3
echo "=== RESTORE DEFAULT ==="
kill $MTPPID 2>/dev/null; sleep 2
cd $Q && ./scripts/start.sh >/dev/null 2>&1; echo restore-issued
