#!/usr/bin/env bash
# run_watchdog.sh <out_dir> <idle_seconds> <retries> -- <command ...>
# Runs a resumable sharded command (zeroshot_diagnose.py skips finished shards). If no new shard_*.csv appears in <out_dir>
# for <idle_seconds> (counted from the later of the start and the newest shard), the whole process group (main process,
# vLLM EngineCore and workers) is killed, GPU memory is awaited, and the command is retried (up to <retries> attempts).
# Stopping the watchdog (Ctrl-C, kill) stops the command's whole process group as well. Before a retry it waits only for the
# GPUs of the command (CUDA_VISIBLE_DEVICES; all GPUs when unset), so other jobs on a shared machine cannot block it.
OUT=$1; IDLE=$2; N=$3; shift 3; [ "$1" = "--" ] && shift
POLL=${WATCHDOG_POLL:-30}  # seconds between checks
pid=""
stop() {
  if [ -n "$pid" ]; then
    kill -TERM -- -"$pid" 2>/dev/null
    for _ in $(seq 1 30); do kill -0 -- -"$pid" 2>/dev/null || break; sleep 1; done
    kill -KILL -- -"$pid" 2>/dev/null
  fi
  exit 130
}
trap stop INT TERM HUP
for attempt in $(seq 1 $N); do
  setsid "$@" &
  pid=$!
  last=$(date +%s)
  hung=0
  while kill -0 $pid 2>/dev/null; do
    sleep "$POLL" & wait $!   # interruptible, so a stop signal is handled at once
    newest=$(stat -c %Y "$OUT"/shard_*.csv 2>/dev/null | sort -n | tail -n 1)
    [ -n "$newest" ] && [ "$newest" -gt "$last" ] && last=$newest
    if [ $(( $(date +%s) - last )) -gt "$IDLE" ]; then
      echo "watchdog $(date '+%m-%d %H:%M:%S'): no new shard in $OUT for ${IDLE}s, killing process group $pid (attempt $attempt)"
      kill -9 -- -$pid 2>/dev/null; kill -9 $pid 2>/dev/null; hung=1; break
    fi
  done
  wait $pid; rc=$?
  [ $hung -eq 0 ] && [ $rc -eq 0 ] && exit 0
  echo "watchdog: attempt $attempt ended rc=$rc hung=$hung"
  ids=${CUDA_VISIBLE_DEVICES:+--id=$CUDA_VISIBLE_DEVICES}
  until [ "$(nvidia-smi $ids --query-gpu=memory.used --format=csv,noheader,nounits | awk '$1 > 3000' | wc -l)" -eq 0 ]; do
    sleep "${WATCHDOG_POLL:-15}"
  done
done
exit 1
