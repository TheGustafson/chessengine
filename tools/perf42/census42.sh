#!/usr/bin/env bash
# perf study 42 -- per-node engine census at a chosen Hash size.
#
#   bash tools/perf42/census42.sh <hashMB> [binary] [budget] [reps]
#
# Five event groups, each its own run so nothing multiplexes (Zen 4 has 6
# general PMCs and `perf stat` will happily time-slice past that in silence).
# Everything is reported PER NODE against the run's own `Nodes searched`, which
# is what makes two Hash sizes comparable at all: the trees differ, the node
# BUDGET does not.
set -euo pipefail
cd "$(dirname "$0")/../.."
MB=${1:-16}
BIN=${2:-bin34/b42}
BUDGET=${3:-100000}
REPS=${4:-1}
CPU=${P42_CPU:-2}
NET=engines/canonical/net.cenn
FENS=tools/nnue/calib/wl50.fens
tmp=$(mktemp -d); trap 'rm -rf "$tmp"' EXIT

groups=(
  "instructions:u,cycles:u,ex_no_retire.load_not_complete:u,ex_no_retire.empty:u,ex_no_retire.thread_not_selected:u"
  "ls_l1_d_tlb_miss.all:u,ls_l1_d_tlb_miss.all_l2_miss:u,L1-dcache-loads:u,L1-dcache-load-misses:u"
  "ls_any_fills_from_sys.all:u,ls_any_fills_from_sys.local_l2:u,ls_any_fills_from_sys.local_ccx:u,ls_any_fills_from_sys.dram_io_all:u"
)

for rep in $(seq "$REPS"); do
  for ev in "${groups[@]}"; do
    nice -n 19 perf stat -x, -e "$ev" -o "$tmp/stat" -- \
      taskset -c "$CPU" "./$BIN" bench "$MB" 1 "$BUDGET" "$FENS" nodes "$NET" \
      >/dev/null 2>"$tmp/bench"
    nodes=$(sed -n 's/^Nodes searched  : //p' "$tmp/bench" | tail -1)
    awk -F, -v n="$nodes" -v mb="$MB" -v r="$rep" '
      $1 ~ /^[0-9]+$/ && $3 != "" { printf "mb=%s rep=%s nodes=%s %s=%.6f\n", mb, r, n, $3, $1 / n }
    ' "$tmp/stat"
  done
done
