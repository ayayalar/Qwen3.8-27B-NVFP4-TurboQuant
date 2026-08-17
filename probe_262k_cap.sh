#!/bin/bash
# Reproduce the verified 262K single-stream cap test (CALIBRATION §7).
# Boots the KV de-over-provisioned profile (KV_BYTES=5500000000), then sends a
# /tokenize-calibrated prompt ~262,122 tokens (99.99% of the 262,144 cap),
# confirms usage.prompt_tokens, and checks the engine survived.
# Expected (verified 2026-08-17 on the reference box): request completes in
# ~120-125s, engine HTTP 200 after. Requires ~35s boot + ~2.5 min test.
set -u
Q="$(cd "$(dirname "$0")" && pwd)"
# stop any running server
"$Q/scripts/stop.sh" >/dev/null 2>&1
sleep 3
# shutdown any orphaned engine cores (vLLM subprocess holds VRAM if parent is killed)
pkill -f "[V]LLM::EngineCore" 2>/dev/null
pkill -f "[m]ultiprocessing.resource_tracker" 2>/dev/null
sleep 5
KV_BYTES=5500000000 "$Q/scripts/start.sh" 2>&1 | tail -1
echo "waiting for boot..."
for i in $(seq 1 150); do
  sleep 2
  curl -s -o /dev/null -w "%{http_code}" --max-time 4 localhost:8000/v1/models 2>/dev/null | grep -q 200 && break
done
curl -s -o /dev/null -w "boot: http:%{http_code}\n" --max-time 4 localhost:8000/v1/models || { echo BOOTFAIL; exit 1; }
grep -m1 "GPU KV cache size" /tmp/qwen38_vllm.log 2>/dev/null
echo "=== genuine 262K cap prefill ==="
python3 - <<'PYEOF'
import json, time, urllib.request
HOST="http://localhost:8000"
FILLER=("The quick brown fox jumps over the lazy dog while the sun sets behind the hills. "
        "Program managers coordinate schedules, dependencies, and deliverables across teams. "
        "The server allocates GPU memory for key-value caches proportional to sequence length. ")
SUFFIX="\n\nSay DONE and stop."
def tok(text):
    b=json.dumps({"model":"unsloth/Qwen3.8-27B-NVFP4","prompt":text}).encode()
    r=urllib.request.Request(HOST+"/tokenize",data=b,headers={"Content-Type":"application/json"})
    with urllib.request.urlopen(r,timeout=120) as resp:
        return len(json.loads(resp.read().decode())["token_ids"])
n=tok(FILLER*200); cpt=len(FILLER*200)/n
want=int((262143-tok(SUFFIX)-4)*cpt)
prompt=(FILLER*(want//len(FILLER)+2))[:want]
print(f"exact prompt (via /tokenize): {tok(prompt+SUFFIX)} tokens")
body=json.dumps({"model":"unsloth/Qwen3.8-27B-NVFP4",
    "messages":[{"role":"user","content":prompt+SUFFIX}],
    "max_tokens":1,"temperature":0.0,"stream":True,
    "chat_template_kwargs":{"enable_thinking":False,"thinking":False},
    "stream_options":{"include_usage":True}}).encode()
req=urllib.request.Request(HOST+"/v1/chat/completions",data=body,headers={"Content-Type":"application/json"})
t0=time.time(); usage=None
try:
    with urllib.request.urlopen(req,timeout=1500) as resp:
        for raw in resp:
            line=raw.decode().strip()
            if not line.startswith("data:"): continue
            d=line[5:].strip()
            if d=="[DONE]": break
            try:
                o=json.loads(d)
                if o.get("usage"): usage=o["usage"]
            except Exception: pass
    pt=(usage or {}).get("prompt_tokens",-1)
    print(f"REQUEST OK wall={time.time()-t0:.1f}s usage.prompt_tokens={pt} ({pt/262144*100:.1f}% of cap)")
except Exception as e:
    print(f"REQUEST FAILED: {e}")
try:
    urllib.request.urlopen(HOST+"/v1/models",timeout=4)
    print("engine after: HTTP 200 (SURVIVED)")
except Exception as e:
    print("engine after: DOWN -", e)
PYEOF
echo "=== restore shipped default ==="
"$Q/scripts/stop.sh" >/dev/null 2>&1; sleep 2
"$Q/scripts/start.sh" >/dev/null 2>&1
echo "restored"
