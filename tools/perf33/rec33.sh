#!/usr/bin/env bash
# perf study 33 -- record one event on the study-29/31 frame.
#
#   bash tools/perf33/rec33.sh <event-tag> <perf-event> [period] [binary]
#
# Frame (identical to studies 29 and 31 so the numbers compose):
#   bench 16 1 100000 tools/nnue/calib/wl50.fens nodes engines/canonical/net.cenn
#   = 50 FENs x 100k nodes = 5,001,562 nodes, one thread, 16 MB TT, D48X.
#
# Default binary is bin33/prof -- the line-table build, proven codegen-identical
# to bin33/base by tools/perf33/textnull33.py.  perf.data lands on DISK, never
# /tmp: /tmp is tmpfs (RAM) on this box and a 15 GB machine with a live trainer
# cannot afford it.
#
# CPU 2 (SMT sibling 10) is deliberate: perf study 32 is measuring on CPU 6
# (sibling 14) in a neighbouring worktree.
set -euo pipefail
cd "$(dirname "$0")/../.."
ROOT=$PWD
TAG=${1:?event tag}
EV=${2:?perf event}
PERIOD=${3:-50021}
BIN=${4:-bin33/prof}
CPU=${P33_CPU:-2}
NET=$ROOT/engines/canonical/net.cenn
FENS=$ROOT/tools/nnue/calib/wl50.fens
OUT=$ROOT/perf33data
mkdir -p "$OUT"

nice -n 19 perf record -e "$EV" -c "$PERIOD" --no-inherit -q \
  -o "$OUT/$TAG.data" -- \
  taskset -c "$CPU" "$ROOT/$BIN" bench 16 1 100000 "$FENS" nodes "$NET" \
  > "$OUT/$TAG.bench" 2>"$OUT/$TAG.err" || {
    echo "perf record failed:"; tail -5 "$OUT/$TAG.err"; exit 1; }

grep -E 'Nodes searched|Total time' "$OUT/$TAG.bench" || true
echo "[rec33] $TAG: $(perf report -i "$OUT/$TAG.data" --stdio -q 2>/dev/null | head -1)"
