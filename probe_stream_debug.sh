#!/bin/bash
# Capture the raw first ~800 bytes of a streamed response to debug the parser.
set -u
Q=/tmp/qbench
$Q/scripts/stop.sh >/dev/null 2>&1; sleep 2
MTP=0 $Q/scripts/start.sh >/dev/null 2>&1
for i in $(seq 1 90); do sleep 2; curl -s -o /dev/null -w "%{http_code}" --max-time 4 localhost:8000/v1/models 2>/dev/null | grep -q 200 && break; done
echo "boot ok"
python3 - <<'EOF'
import json, urllib.request, time
p={"model":"unsloth/Qwen3.8-27B-NVFP4","messages":[{"role":"user","content":"hi"}],"max_tokens":20,"temperature":0.0,"stream":True}
req=urllib.request.Request("http://localhost:8000/v1/chat/completions",data=json.dumps(p).encode(),headers={"Content-Type":"application/json"})
t0=time.time()
with urllib.request.urlopen(req,timeout=120) as r:
    count=0
    for line in r:
        count+=1
        if count>3: break
        print(f"RAW[{count}]: {line!r}")
EOF
echo "=== restore ==="
$Q/scripts/stop.sh >/dev/null 2>&1; sleep 2
$Q/scripts/start.sh >/dev/null 2>&1; echo restored
