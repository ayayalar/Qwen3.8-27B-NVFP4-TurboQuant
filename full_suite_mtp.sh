#!/bin/bash
# Full official benchmark suite against the MTP=1 config (shipped start.sh).
# Runs: control_test, speed_test, bench_framework.py t4 (README suite).
set -u
Q=/tmp/qbench
mkdir -p $Q
$Q/scripts/stop.sh >/dev/null 2>&1
sleep 3
MTP=1 $Q/scripts/start.sh 2>&1
echo "waiting for boot..."
for i in $(seq 1 90); do sleep 2; curl -s -o /dev/null -w "%{http_code}" --max-time 4 localhost:8000/v1/models 2>/dev/null | grep -q 200 && break; done
echo "=== boot state:"
curl -s -o /dev/null -w "http:%{http_code}\n" --max-time 4 localhost:8000/v1/models || { echo DOWN; exit 1; }
grep -oE "Overriding cudagraph[^\n]*" /tmp/qwen38_vllm.log | tail -1
grep -E "GPU KV cache size" /tmp/qwen38_vllm.log | tail -1

echo ""
echo "############ 1/3 control_test.sh ############"
cd $Q/benchmark && python3 control_test.py 2>&1 | tail -12

echo ""
echo "############ 2/3 speed_test.py ############"
python3 speed_test.py 2>&1 | grep -E "warmup|measured|CONCURRENT"

echo ""
echo "############ 3/3 bench_framework.py t4 (FULL README SUITE) ############"
python3 bench_framework.py t4 2>&1 | tail -30

echo ""
echo "=== acceptance (whole run) ==="
grep -o "Accepted: [0-9]* tokens, Drafted: [0-9]*" /tmp/qwen38_vllm.log | tail -3
echo "=== RESTORE DEFAULT (no MTP) ==="
cd $Q && ./scripts/stop.sh >/dev/null 2>&1; sleep 2
./scripts/start.sh >/dev/null 2>&1; echo restore-issued
