#!/usr/bin/env python3
"""Minimal control: identical prompts, identical scoring, compare configs.
Endpoint is env-overridable (BENCH_URL), default localhost:8000."""
import json, os, urllib.request, random, sys

URL = os.environ.get("BENCH_URL", "http://localhost:8000/v1/chat/completions")
MODEL = os.environ.get("BENCH_MODEL", "unsloth/Qwen3.8-27B-NVFP4")

def call(payload, timeout=300):
    req = urllib.request.Request(URL, data=json.dumps(payload).encode(),
                                 headers={"Content-Type": "application/json"})
    with urllib.request.urlopen(req, timeout=timeout) as r:
        return json.loads(r.read().decode())

def content_of(d):
    m = d["choices"][0]["message"]
    return (m.get("content") or "").strip()

print("=== BASIC ===")
for q in ["What is 2+2? Answer with just the number.",
          "Say hello in exactly two words.",
          "What is the capital of France? One word."]:
    try:
        d = call({"model": MODEL, "messages": [{"role": "user", "content": q}],
                  "max_tokens": 80, "temperature": 0.2})
        print(repr(q), "->", repr(content_of(d)))
    except Exception as e:
        print("ERR", e)

print("=== TOOL ===")
t = {"type": "function", "function": {"name": "web_search", "description": "search the web",
     "parameters": {"type": "object", "properties": {"query": {"type": "string"}}, "required": ["query"]}}}
for i in range(2):
    try:
        d = call({"model": MODEL, "messages": [{"role": "user", "content": "Search for quantum computing articles"}],
                  "tools": [t], "max_tokens": 200, "temperature": 0.0})
        m = d["choices"][0]["message"]
        print("tool_calls:", m.get("tool_calls"), "| content:", repr(content_of(d)))
    except Exception as e:
        print("ERR", e)

print("=== NEEDLE 4K ===")
WORDS = ("alpha beta gamma delta").split()
random.seed(42)
filler = " ".join(random.choice(WORDS) for _ in range(4000))
doc = filler + " NEEDLE-7F3A is active " + filler
G = "Return ONLY the marker sentence starting with NEEDLE-."
try:
    d = call({"model": MODEL, "messages": [{"role": "user", "content": doc + "\n\n" + G}],
              "max_tokens": 80, "temperature": 0.0})
    c = content_of(d)
    if not c:
        c = "(EMPTY) " + (d["choices"][0]["message"].get("reasoning") or "")[:200]
    print("content:", repr(c))
    print("recall:", "NEEDLE-7F3A is active" in c)
except Exception as e:
    print("ERR", e)
