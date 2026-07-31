#!/usr/bin/env bash
# perf study 61 — where do this engine's L1 line fills COME FROM?
#
# Study 51 measured that removing the fold's weight-row streaming (`probe-wpin`)
# is worth −13.19% cycles and −19.7% of all L1 fills.  It never asked which
# LEVEL those fills were served by.  That is the question that decides whether
# the remaining route is an L1 problem (arrival/ordering — nine dead arms) or an
# L2/L3/DRAM problem (footprint), and the two have disjoint levers.
#
#   bash tools/perf61/fills61.sh <label> [<label> ...]
#
# Zen 4 `ls_any_fills_from_sys` umasks, in latency order:
#   local_l2     — filled from this core's own L2   (~14 cy)
#   local_ccx    — filled from the CCX's L3         (~50 cy)
#   near_cache   — another CCX's cache
#   dram_io_near — filled from DRAM                 (~250-400 cy)
#
# Emits one TSV row per (label, run): label, run, nodes, then per-node rates.
# `cycles:u` is collected but is NOT load-robust (study 60 §6) — it is here for
# its sign only, and the adjudication lives in a panel, not in this script.
set -euo pipefail
cd "$(dirname "$0")/../.."
CPU=${P61_CPU:-2}
REPS=${P61_REPS:-3}
NET=engines/canonical/net.cenn
FRAME=(bench 16 1 100000 tools/nnue/calib/wl50.fens nodes "$NET")
EV=ls_any_fills_from_sys.all:u,ls_any_fills_from_sys.local_l2:u,ls_any_fills_from_sys.local_ccx:u,ls_any_fills_from_sys.dram_io_near:u,instructions:u,cycles:u

printf 'label\trun\tnodes\tfills\tl2\tl3\tdram\tinsn\tcycles\n'
for b in "$@"; do
  for r in $(seq 1 "$REPS"); do
    out=$(taskset -c "$CPU" perf stat -x, -e "$EV" ./bin61/"$b" "${FRAME[@]}" 2>&1)
    nodes=$(printf '%s\n' "$out" | sed -n 's/^Nodes searched  *: *//p')
    get() { printf '%s\n' "$out" | awk -F, -v e="$1" '$3==e {print $1}'; }
    printf '%s\t%s\t%s' "$b" "$r" "$nodes"
    for e in ls_any_fills_from_sys.all:u ls_any_fills_from_sys.local_l2:u \
             ls_any_fills_from_sys.local_ccx:u ls_any_fills_from_sys.dram_io_near:u \
             instructions:u cycles:u; do
      v=$(get "$e")
      printf '\t%.4f' "$(python3 -c "print(${v:-0}/${nodes})")"
    done
    printf '\n'
  done
done
