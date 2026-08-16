#!/bin/bash
# Investigate why thinking-off looks 4x slower: check actual tokens, acceptance, and repetition.
set -u
Q=/tmp/qbench
$Q/scripts/stop.sh >/dev/null 2>&1; sleep 3
$Q/scripts/start.sh >/dev/null 2>&1
for i in $(seq 1 90); do sleep 2; curl -s -o /dev/null -w "%{http_code}" --max-time 4 localhost:8000/v1/models 2>/dev/null | grep -q 200 && break; done
echo "boot ok"
python3 - <<'EOF'
import json, urllib.request, time
URL="http://localhost:8000/v1/chat/completions"
MODEL="unsloth/Qwen3.8-27B-NVFP4"
SYS="You are terse. Respond only with repeating numbers separated by spaces, like: 1 2 3 4 5 6 ... Always continue the sequence with the next integer (increment by 1). Start and stay in the sequence, do not stop."
def inspect(label, kwargs=None):
    p={"model":MODEL,"messages":[{"role":"system","content":SYS},{"role":"user","content":"Go"}],"max_tokens":150,"temperature":0.0}
    if kwargs: p["chat_template_kwargs"]=kwargs
    req=urllib.request.Request(URL,data=json.dumps(p).encode(),headers={"Content-Type":"application/json"})
    t0=time.time()
    with urllib.request.urlopen(req,timeout=300) as r: d=json.loads(r.read().decode())
    dt=time.time()-t0
    m=d["choices"][0]["message"]
    print(f"\n[{label}] wall={dt:.1f}s")
    print(f"  usage: {d['usage']}")
    print(f"  content head: {repr((m.get('content') or '')[:80])}")
    print(f"  reasoning head: {repr((m.get('reasoning') or '')[:80])}")
    print(f"  finish_reason:", d["choices"][0].get("finish_reason"))
inspect("think-on")
inspect("think-off", {"enable_thinking": False})
# one more with max_tokens=400 to see if think-off actually fills 400 tokens
p={"model":MODEL,"messages":[{"role":"system","content":SYS},{"role":"user","content":"Go"}],"max_tokens":400,"temperature":0.0,"chat_template_kwargs":{"enable_thinking":False}}
req=urllib.request.Request(URL,data=json.dumps(p).encode(),headers={"Content-Type":"application/json"})
t0=time.time()
with urllib.request.urlopen(req,timeout=300) as r: d=json.loads(r.read().decode())
dt=time.time()-t0
print(f"\n[think-off 400] wall={dt:.1f}s tok/s(usage)={d['usage']['completion_tokens']/dt:.1f} usage={d['usage']}")
EOF
echo "=== acceptance during think-off (draft 400 run):"
grep -oE "Mean acceptance length: [0-9.]+" /tmp/qwen38_vllm.log | tail -2
echo "=== RESTORE ==="
$Q/scripts/stop.sh >/dev/null 2>&1; sleep 2
$Q/scripts/start.sh >/dev/null 2>&1; echo restored