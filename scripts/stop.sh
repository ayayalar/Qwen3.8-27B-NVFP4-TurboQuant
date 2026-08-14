#!/bin/bash
# Stop the Qwen3.8-27B-NVFP4 vLLM server started by start.sh.
# Graceful SIGTERM first (vLLM drains), then SIGKILL after a timeout.
set -u

PIDFILE="${PIDFILE:-/tmp/qwen38_vllm.pid}"
LOGFILE="${LOGFILE:-/tmp/qwen38_vllm.log}"
WAIT_SECONDS="${WAIT_SECONDS:-30}"

stop_pid() {
  local pid="$1"
  if kill -0 "$pid" 2>/dev/null; then
    kill -TERM "$pid"
  fi
  # Also signal the EngineCore children if the launcher already exited
  for cpid in $(pgrep -P "$pid" 2>/dev/null); do
    kill -TERM "$cpid" 2>/dev/null
  done
  for _ in $(seq 1 "$WAIT_SECONDS"); do
    if ! kill -0 "$pid" 2>/dev/null; then
      break
    fi
    sleep 1
  done
  if kill -0 "$pid" 2>/dev/null; then
    echo "pid $pid did not exit after ${WAIT_SECONDS}s — sending SIGKILL"
    kill -KILL "$pid" 2>/dev/null
    sleep 1
  fi
}

stopped=""
if [ -f "$PIDFILE" ]; then
  pid="$(cat "$PIDFILE")"
  if kill -0 "$pid" 2>/dev/null; then
    echo "stopping pid $pid (SIGTERM)"
    stop_pid "$pid"
    stopped="$pid"
  fi
  rm -f "$PIDFILE"
fi

# Fallback: any stray vllm serve process (e.g. from an old manual launch)
for pid in $(pgrep -f "vllm serve .*Qwen3.8-27B-NVFP4" 2>/dev/null); do
  if [ "$pid" != "$stopped" ] && kill -0 "$pid" 2>/dev/null; then
    echo "stopping stray vllm pid $pid"
    stop_pid "$pid"
    stopped="$pid"
  fi
done

if [ -z "$stopped" ]; then
  echo "no running server found (pidfile:$PIDFILE)"
else
  echo "stopped ($stopped). last log lines:"
  [ -f "$LOGFILE" ] && tail -3 "$LOGFILE"
fi
