#!/bin/bash
# Verify shipped default (start.sh, now MTP=1 default). Then verify MTP=0 restores original config.
set -u
Q=/tmp/qbench
echo "########## A) stock ./start.sh (default should now be MTP=1) ##########"
$Q/scripts/stop.sh >/dev/null 2>&1; sleep 3
$Q/scripts/start.sh 2>&1 | tail -1
echo "waiting boot..."
for i in $(seq 1 90); do sleep 2; curl -s -o /dev/null -w "%{http_code}" --max-time 4 localhost:8000/v1/models 2>/dev/null | grep -q 200 && break; done
echo "http check:"; curl -s -o /dev/null -w "%{http_code}\n" --max-time 4 localhost:8000/v1/models
echo "spec enabled? (expect speculative_config + PIECEWISE override):"
grep -oE "speculative_config=[^ ]{0,60}" /tmp/qwen38_vllm.log | tail -1
grep -oE "Overriding cudagraph[^\n]*" /tmp/qwen38_vllm.log | tail -1
echo "KV bytes (expect 5800000000):"
grep -oE "kv_cache_memory_bytes?: [0-9]+|[0-9]{8,10} memory for KV" /tmp/qwen38_vllm.log | tail -1
echo "--- official speed_test:"
cd $Q/benchmark && python3 speed_test.py 2>&1 | grep -E "measured|CONCURRENT"

echo ""
echo "########## B) MTP=0 ./start.sh (should be ORIGINAL no-spec config) ##########"
$Q/scripts/stop.sh >/dev/null 2>&1; sleep 3
MTP=0 $Q/scripts/start.sh >/dev/null 2>&1
for i in $(seq 1 90); do sleep 2; curl -s -o /dev/null -w "%{http_code}" --max-time 4 localhost:8000/v1/models 2>/dev/null | grep -q 200 && break; done
echo "spec disabled? (expect no speculative_config, no override, 5368709120 pin):"
grep -c "speculative_config" /tmp/qwen38_vllm.log | tail -1
grep -oE "reserved [0-9.]+ GiB memory for KV" /tmp/qwen38_vllm.log | tail -1
echo "--- official speed_test (MTP=0):"
cd $Q/benchmark && python3 speed_test.py 2>&1 | grep -E "measured|CONCURRENT"
echo "=== leave server on DEFAULT (MTP on) ==="
$Q/scripts/stop.sh >/dev/null 2>&1; sleep 2
$Q/scripts/start.sh >/dev/null 2>&1; echo "restored to shipped default"
