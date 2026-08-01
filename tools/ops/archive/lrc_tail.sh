#!/usr/bin/env bash
# STANDING RULE: every training run gets tb telemetry armed at launch.
# The LR cycles run as their own units and log to JOURNALD, so the existing
# cyc.log shim cannot see them. Bridge the journal into a file the shim reads.
set -uo pipefail
OUT=/home/nick/projects/chessengine/data/nnue/experiments/d49-1280/lrcycles.log
for c in 1 2 3; do
  journalctl --user -u "d49lrc${c}" -f -n 0 --no-pager 2>/dev/null \
    | stdbuf -oL grep -E 'components sb|heldout sb' >> "$OUT" &
done
wait
