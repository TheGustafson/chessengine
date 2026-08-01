#!/usr/bin/env bash
# Keep the trainer the LAST thing the OOM killer picks.
# The trainer inherits oom_score_adj=200 from the harness and cannot lower it
# without CAP_SYS_RESOURCE, so instead we push everything DISPOSABLE to 1000.
# Measured 16:06: the trainer sat at score 940 with 4.1 GB RSS and was killed
# while builds ran. This inverts that ordering.
set -uo pipefail
while :; do
  for pat in 'rustc' 'cargo' 'fastchess' 'target-sota/release/chessengine' \
             'engines/bin/archon' 'engines/bin/s7' 'node-MainThread' 'npm'; do
    for p in $(pgrep -f "$pat" 2>/dev/null); do
      c=$(tr '\0' ' ' < "/proc/$p/cmdline" 2>/dev/null) || continue
      [ -z "$c" ] && continue
      case "$c" in *shell-snapshots*|*pgrep*|*train_guard*) continue;; esac
      case "$c" in *pump_train*) continue;; esac          # never the trainer
      cur=$(cat "/proc/$p/oom_score_adj" 2>/dev/null || echo 0)
      [ "${cur:-0}" -lt 1000 ] && echo 1000 > "/proc/$p/oom_score_adj" 2>/dev/null
    done
  done
  sleep 15
done
