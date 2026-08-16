#!/bin/bash
# Measure decode speed with thinking ON vs OFF on the shipped MTP=1 default.
# Uses the same counting prompt + usage-based math as speed_test.py.
set -u
Q=/tmp/qbench
cd $Q
$Q/scripts/stop.sh >/dev/null 2>&1; sleep 3
$Q/scripts/start.sh >/dev/null 2>&1
for i in $(seq 1 90); do sleep 2; curl -s -o /dev/null -w "%{http_code}" --max-time 4 localhost:8000/v1/models 2>/dev/null | grep -q 200 && break; done
echo "boot ok (MTP=1 default)"

python3 - <<'EOF'
import json, urllib.request, time
URL="http://localhost:8000/v1/chat/completions"
MODEL="unsloth/Qwen3.8-27B-NVFP4"
SYS="You are terse. Respond only with repeating numbers separated by spaces, like: 1 2 3 4 5 6 ... Always continue the sequence with the next integer (increment by 1). Start and stay in the sequence, do not stop."

def run(label, extra_kwargs=None):
    p={"model":MODEL,"messages":[{"role":"system","content":SYS},{"role":"user","content":"Go"}],
       "max_tokens":400,"temperature":0.0}
    if extra_kwargs: p["chat_template_kwargs"]=extra_kwargs
    # warmup
    req=urllib.request.Request(URL,data=json.dumps(p).encode(),headers={"Content-Type":"application/json"})
    urllib.request.urlopen(req,timeout=300).read()
    results=[]
    for _ in range(3):
        req=urllib.request.Request(URL,data=json.dumps(p).encode(),headers={"Content-Type":"application/json"})
        t0=time.time()
        with urllib.request.urlopen(req,timeout=600) as r: d=json.loads(r.read().decode())
        dt=time.time()-t0
        u=d["usage"]; comp=u["completion_tokens"]
        det=u.get("completion_tokens_details",{})
        results.append(comp/dt)
    best=max(results)
    avg=sum(results)/len(results)
    print(f"[{label}] runs(tok/s): {[round(x,1) for x in results]} avg={avg:.1f} best={best:.1f}")

print("=== A) THINKING ON (default, as speed_test runs) ===")
run("think-on")
print("=== B) THINKING OFF (enable_thinking:false) ===")
run("think-off", {"enable_thinking": False})
EOF
echo "=== RESTORE ==="
$Q/scripts/stop.sh >/dev/null 2>&1; sleep 2
$Q/scripts/start.sh >/dev/null 2>&1; echo restored-to-default
