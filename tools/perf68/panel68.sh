#!/usr/bin/env bash
# perf study 68 — the paired panel, study 67's `panel67.sh` plus the op-cache
# counter that study 67 named as unmeasured.
#
#   bash tools/perf68/panel68.sh <rounds> <label> [label...]
#
# Every label runs once per ROUND, in the same round, so the box's other
# tenants (a GPU trainer and a continuous 1500-game match) hit every arm
# equally and cancel in the within-round contrast.
#
# FIVE hardware events. Zen 4 has six general PMCs, so nothing multiplexes and
# every number is a full count; `task-clock` is a software event and takes no
# PMC. `cycles:u` is deliberately ABSENT (studies 43/63: +49.9 % under load).
#
# `de_src_op_disp.decoder` is the new one: ops delivered to dispatch by the
# DECODER rather than the op cache. It is the direct counter for "the frontend
# is decoding 40.5 % of this engine's ops" (study 67 §1.3) and it is the one a
# code-LAYOUT arm is supposed to move.
set -euo pipefail
cd "$(dirname "$0")/../.."
R=${1:?rounds}; shift
CPU=${P68_CPU:-2}
EV=instructions:u,branch-misses:u,ic_cache_fill_l2:u,de_op_queue_empty:u,de_src_op_disp.decoder:u,task-clock
NET=engines/canonical/net.cenn
FENS=tools/nnue/calib/wl50.fens
echo "round,label,instructions,branch_misses,ic_fill_l2,opq_empty,dec_ops,task_clock_ms"
for r in $(seq 1 "$R"); do
  for L in "$@"; do
    # `perf stat` writes counters to STDERR and the bench writes its node report
    # to STDOUT, so the redirection order matters: dup stderr onto the capture
    # pipe FIRST, then send stdout to /dev/null.
    out=$(taskset -c "$CPU" perf stat -x, -e "$EV" -- \
      "engines/bin/$L" bench 16 1 100000 "$FENS" nodes "$NET" 2>&1 >/dev/null || true)
    i=$(printf '%s\n' "$out" | awk -F, '/instructions:u/{print $1}')
    b=$(printf '%s\n' "$out" | awk -F, '/branch-misses:u/{print $1}')
    f=$(printf '%s\n' "$out" | awk -F, '/ic_cache_fill_l2:u/{print $1}')
    q=$(printf '%s\n' "$out" | awk -F, '/de_op_queue_empty:u/{print $1}')
    d=$(printf '%s\n' "$out" | awk -F, '/de_src_op_disp.decoder:u/{print $1}')
    t=$(printf '%s\n' "$out" | awk -F, '/task-clock/{print $1}')
    echo "$r,$L,$i,$b,$f,$q,$d,$t"
  done
done
