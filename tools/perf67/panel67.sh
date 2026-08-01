#!/usr/bin/env bash
# perf study 67 — the paired panel. Runs every label once per ROUND, in the
# same round, so box load (a GPU trainer and a continuous 1500-game match share
# this machine) hits every arm equally and cancels in the within-round contrast.
#
#   bash tools/perf67/panel67.sh <rounds> <label> [label...]
#
# Four hardware events + task-clock: Zen 4 has six general PMCs, so nothing
# multiplexes and every number is a full count. `cycles:u` is deliberately
# ABSENT (studies 43/63: +49.9% under load, not load-robust).
set -euo pipefail
cd "$(dirname "$0")/../.."
R=${1:?rounds}; shift
CPU=${P67_CPU:-2}
EV=instructions:u,branch-misses:u,ic_cache_fill_l2:u,de_op_queue_empty:u,task-clock
NET=engines/canonical/net.cenn
FENS=tools/nnue/calib/wl50.fens
echo "round,label,instructions,branch_misses,ic_fill_l2,opq_empty,task_clock_ms"
for r in $(seq 1 "$R"); do
  for L in "$@"; do
    # `perf stat` writes its counters to STDERR and the bench writes its node
    # report to STDOUT, so the redirection order matters: dup stderr onto the
    # capture pipe FIRST, then send stdout to /dev/null.
    out=$(taskset -c "$CPU" perf stat -x, -e "$EV" -- \
      "engines/bin/$L" bench 16 1 100000 "$FENS" nodes "$NET" 2>&1 >/dev/null || true)
    i=$(printf '%s\n' "$out" | awk -F, '/instructions:u/{print $1}')
    b=$(printf '%s\n' "$out" | awk -F, '/branch-misses:u/{print $1}')
    f=$(printf '%s\n' "$out" | awk -F, '/ic_cache_fill_l2:u/{print $1}')
    q=$(printf '%s\n' "$out" | awk -F, '/de_op_queue_empty:u/{print $1}')
    t=$(printf '%s\n' "$out" | awk -F, '/task-clock/{print $1}')
    echo "$r,$L,$i,$b,$f,$q,$t"
  done
done
