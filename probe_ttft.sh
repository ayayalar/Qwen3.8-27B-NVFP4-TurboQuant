#!/bin/bash
# TTFT measurement: MTP=1 vs MTP=0, streaming first-token latency at multiple prompt sizes.
set -u
Q=/tmp/qbench
cd $Q

run_arm() {
  local label="$1"; shift
  echo "############ ARM: $label ############"
  $Q/scripts/stop.sh >/dev/null 2>&1; sleep 3
  if [ "$label" = "MTP1" ]; then
    $Q/scripts/start.sh >/dev/null 2>&1
  else
    MTP=0 $Q/scripts/start.sh >/dev/null 2>&1
  fi
  for i in $(seq 1 90); do sleep 2; curl -s -o /dev/null -w "%{http_code}" --max-time 4 localhost:8000/v1/models 2>/dev/null | grep -q 200 && break; done
  echo "boot ok; warmup..."
  # warmup request
  python3 - <<'EOF'
import json, urllib.request
p={"model":"unsloth/Qwen3.8-27B-NVFP4","messages":[{"role":"user","content":"hi"}],"max_tokens":5,"temperature":0.0}
urllib.request.urlopen(urllib.request.Request("http://localhost:8000/v1/chat/completions",data=json.dumps(p).encode(),headers={"Content-Type":"application/json"}),timeout=300).read()
print("warmup done")
EOF
  python3 - "$label" <<'EOF'
import sys, json, urllib.request, time, random
label=sys.argv[1]
URL="http://localhost:8000/v1/chat/completions"
MODEL="unsloth/Qwen3.8-27B-NVFP4"
random.seed(5)
w="alpha beta gamma delta epsilon zeta eta theta".split()
words=lambda n:" ".join(random.choice(w) for _ in range(n))
def ttft(prompt_tokens, max_tokens=50):
    prompt = words(prompt_tokens)
    payload={"model":MODEL,"messages":[{"role":"user","content":prompt}],"max_tokens":max_tokens,"temperature":0.0,"stream":True}
    req=urllib.request.Request(URL,data=json.dumps(payload).encode(),headers={"Content-Type":"application/json"})
    t0=time.time()
    with urllib.request.urlopen(req,timeout=900) as r:
        for line in r:
            line=line.decode().strip()
            if not line.startswith("data:"): continue
            data=line[5:].strip()
            if data=="[DONE]": break
            try: obj=json.loads(data)
            except Exception: continue
            delta=(obj.get("choices") or [{}])[0].get("delta") or {}
            # First emitted token of ANY kind (reasoning or content) = engine TTFT.
            # Chunk 1 is a role marker (content=""): ignore empty values.
            if delta.get("content") or delta.get("reasoning") or delta.get("reasoning_content"):
                return time.time()-t0
        return None  # no delta seen
print(f"[{label}] TTFT (time to FIRST emitted token — reasoning or content):")
for n in [1024, 8192, 32768, 65536]:
    t=ttft(n); print(f"  prompt {n:>6}: {'n/a' if t is None else f'{t:.2f}s'}", flush=True)
print(f"[{label}] TTFT done")
EOF
  echo ""
}

# MTP=1
run_arm "MTP1"
# MTP=0
run_arm "MTP0"

echo "=== RESTORE DEFAULT (MTP=1) ==="
$Q/scripts/stop.sh >/dev/null 2>&1; sleep 2
$Q/scripts/start.sh >/dev/null 2>&1; echo restore-issued
