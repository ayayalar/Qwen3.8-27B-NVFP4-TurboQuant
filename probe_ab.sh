#!/bin/bash
# A/B the exact factor: official speed_test.py WITH vs WITHOUT expandable_segments,
# both using the SAME MTP=1 start.sh config. Captures acceptance too.
set -u
Q=/tmp/qbench

# ---- RUN A: no expandable_segments (PYTORCH_CUDA_ALLOC_CONF set to empty so start.sh keeps it) ----
echo "########## RUN A: MTP=1 WITHOUT expandable_segments ##########"
$Q/scripts/stop.sh >/dev/null 2>&1
sleep 3
cd $Q
env -u PYTORCH_CUDA_ALLOC_CONF MTP=1 ./scripts/start.sh >/dev/null 2>&1  # no allocator setting
# start.sh only sets it if MTP=1 and unset -> so with it unset it WOULD set it. Use empty export instead:
$Q/scripts/stop.sh >/dev/null 2>&1; sleep 2
PYTORCH_CUDA_ALLOC_CONF="" MTP=1 $Q/scripts/start.sh >/dev/null 2>&1
for i in $(seq 1 90); do sleep 2; curl -s -o /dev/null -w "%{http_code}" --max-time 4 localhost:8000/v1/models 2>/dev/null | grep -q 200 && break; done
echo "boot ok"
echo "--- official speed_test (no allocator):"
cd $Q/benchmark && python3 speed_test.py 2>&1 | grep -E "measured|CONCURRENT"
echo "--- acceptance (run A):"
grep -o "Accepted: [0-9]* tokens, Drafted: [0-9]*" /tmp/qwen38_vllm.log | tail -2
grep -oE "Mean acceptance length: [0-9.]+" /tmp/qwen38_vllm.log | tail -1

echo ""
echo "########## RUN B: MTP=1 WITH expandable_segments (shipped default) ##########"
$Q/scripts/stop.sh >/dev/null 2>&1; sleep 3
MTP=1 $Q/scripts/start.sh >/dev/null 2>&1
for i in $(seq 1 90); do sleep 2; curl -s -o /dev/null -w "%{http_code}" --max-time 4 localhost:8000/v1/models 2>/dev/null | grep -q 200 && break; done
echo "boot ok"
echo "--- official speed_test (with allocator):"
cd $Q/benchmark && python3 speed_test.py 2>&1 | grep -E "measured|CONCURRENT"
echo "--- acceptance (run B):"
grep -o "Accepted: [0-9]* tokens, Drafted: [0-9]*" /tmp/qwen38_vllm.log | tail -2
grep -oE "Mean acceptance length: [0-9.]+" /tmp/qwen38_vllm.log | tail -1

echo ""
echo "=== RESTORE DEFAULT ==="
$Q/scripts/stop.sh >/dev/null 2>&1; sleep 2
$Q/scripts/start.sh >/dev/null 2>&1; echo restore-issued
