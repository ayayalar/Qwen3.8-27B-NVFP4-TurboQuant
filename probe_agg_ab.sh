#!/bin/bash
# A/B clean-boot: fresh-boot aggregate for MNBT=512 vs MNBT=1024 (4-way agg only, official harness).
set -u
Q=/tmp/qbench
cd $Q
arm() {
  local label="$1" mnbt="$2"
  echo "===== $label: mnbt=$mnbt ====="
  $Q/scripts/stop.sh >/dev/null 2>&1; sleep 4
  MNBT=$mnbt $Q/scripts/start.sh >/dev/null 2>&1
  for i in $(seq 1 90); do sleep 2; curl -s -o /dev/null -w "%{http_code}" --max-time 4 localhost:8000/v1/models 2>/dev/null | grep -q 200 && break; done
  pgrep -af "vllm serve" | grep -oE -- "--max-num-batched-tokens [0-9]+" | head -1
  cd $Q/benchmark && python3 speed_test.py 2>&1 | grep -E "measured|CONCURRENT"
  cd $Q
  echo ""
}
# warmup both fresh after their own boot
arm "A" 512
arm "B" 1024
echo "=== leave on DEFAULT (1024) ==="
$Q/scripts/stop.sh >/dev/null 2>&1; sleep 2
$Q/scripts/start.sh >/dev/null 2>&1; echo restored-to-1024-default
