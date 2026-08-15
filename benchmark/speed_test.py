import json, os, urllib.request, time, threading

URL = os.environ.get("BENCH_URL", "http://localhost:8000/v1/chat/completions")
MODEL = os.environ.get("BENCH_MODEL", "unsloth/Qwen3.8-27B-NVFP4")

def run(payload, results, idx):
    t0=time.time()
    req=urllib.request.Request(URL, data=json.dumps(payload).encode(), headers={"Content-Type":"application/json"})
    with urllib.request.urlopen(req, timeout=900) as r:
        d=json.loads(r.read().decode())
    dt=time.time()-t0
    u=d["usage"]
    results[idx]={"dt":dt,"prompt":u["prompt_tokens"],"completion":u["completion_tokens"],
                  "body":(d["choices"][0]["message"].get("content") or "")[:40],
                  "reasoning_len":len(d["choices"][0]["message"].get("reasoning") or "")}

# Single-stream decode speed: force long output, no thinking variability
for label, system in [("warmup","You are terse. Respond only with repeating numbers separated by spaces, like: 1 2 3 4 5 6 ... Always continue the sequence with the next integer (increment by 1). Start and stay in the sequence, do not stop."),
                      ("measured", "You are terse. Respond only with repeating numbers separated by spaces, like: 1 2 3 4 5 6 ... Always continue the sequence with the next integer (increment by 1). Start and stay in the sequence, do not stop.")]:
    p={"model":MODEL,"messages":[{"role":"system","content":system},{"role":"user","content":"Go"}],"max_tokens":400,"temperature":0.0}
    r=run(p,[None],0)
    if r is None:
        # run inline
        t0=time.time(); req=urllib.request.Request(URL, data=json.dumps(p).encode(), headers={"Content-Type":"application/json"})
        with urllib.request.urlopen(req, timeout=900) as rsp: d=json.loads(rsp.read().decode())
        dt=time.time()-t0; u=d["usage"]
        print(label, "tokens:", u["completion_tokens"], "time:", round(dt,2), "-> tok/s:", round(u["completion_tokens"]/dt,1), "| content head:", repr((d["choices"][0]["message"].get("content") or "")[:30]))
    else:
        rr=r[0]
        print(label, "tokens:", rr["completion"], "time:", round(rr["dt"],2), "-> tok/s:", round(rr["completion"]/rr["dt"],1))

# Concurrent: 4 simultaneous requests
t0=time.time()
res={}
threads=[]
for i in range(4):
    p={"model":MODEL,"messages":[{"role":"user","content":"Give me a list of exactly 300 five-letter English words separated by newlines, nothing else."}],"max_tokens":400,"temperature":0.7}
    th=threading.Thread(target=run,args=(p,res,i)); th.start(); threads.append(th)
for th in threads: th.join()
wall=time.time()-t0
tot=sum(r["completion"] for r in res.values())
print("CONCURRENT x4:", tot, "tokens in", round(wall,1), "s -> aggregate tok/s:", round(tot/wall,1), "(per-request ~", round(tot/wall/4,1), "tok/s)")
