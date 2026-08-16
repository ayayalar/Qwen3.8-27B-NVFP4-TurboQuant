#!/bin/bash
# Validate the NEW shipped default (MNBT=1024, MTP=1): full official suite.
set -u
Q=/tmp/qbench
mkdir -p $Q
$Q/scripts/stop.sh >/dev/null 2>&1; sleep 3
$Q/scripts/start.sh >/dev/null 2>&1   # stock = new default (mnbt 1024, MTP on)
for i in $(seq 1 90); do sleep 2; curl -s -o /dev/null -w "%{http_code}" --max-time 4 localhost:8000/v1/models 2>/dev/null | grep -q 200 && break; done
echo "=== boot state:"
curl -s -o /dev/null -w "http:%{http_code}\n" --max-time 4 localhost:8000/v1/models || { echo DOWN; exit 1; }
pgrep -af "vllm serve" | grep -oE -- "--max-num-batched-tokens [0-9]+|--speculative-config [^ ]+" | head -2
grep -E "max_num_batched_tokens" /tmp/qwen38_vllm.log | tail -1 | grep -oE "max_num_batched_tokens.: [0-9]+"
echo ""
echo "### 1/3 speed_test ###"
cd $Q/benchmark && python3 speed_test.py 2>&1 | grep -E "measured|CONCURRENT"
echo ""
echo "### 2/3 control_test ###"
python3 control_test.py 2>&1 | tail -8
echo ""
echo "### 3/3 bench_framework t4 (FULL: tools + needles incl 196K + code) ###"
python3 bench_framework.py t4 2>&1 | tail -25
echo ""
echo "### acceptance ###"
grep -o "Accepted: [0-9]* tokens, Drafted: [0-9]*" /tmp/qwen38_vllm.log | tail -2
echo "### server alive after suite? ###"
curl -s -o /dev/null -w "%{http_code}\n" --max-time 4 localhost:8000/v1/models || echo DOWN
echo "### done (leave running on new default) ###"
