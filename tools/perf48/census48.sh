#!/usr/bin/env bash
# perf study 48 -- the LOAD census. Re-derives study 44's 2280.76 ld_dispatch/node
# and splits the memory-op population into loads / stores / load-store.
#
#   bash tools/perf48/census48.sh <reps> <label>
#
# Groups are UNMULTIPLEXED (<= 6 events each; Zen 4 has 6 core PMCs). Every
# group carries cycles:u AND instructions:u so a share can be computed WITHIN a
# run (study 44 Section 0.3) rather than against a pooled level from another slot.
set -euo pipefail
cd "$(dirname "$0")/../.."
REPS=${1:-5}
LABEL=${2:-base48}
CPU=${P48_CPU:-2}
NET=${P48_NET:-engines/canonical/net.cenn}
FENS=${P48_FENS:-tools/nnue/calib/wl50.fens}
BIN=${P48_BIN:-bin48}
OUT=${P48_OUT:-docs/research/perf-artifacts-48/census.txt}
mkdir -p "$(dirname "$OUT")"

EVGROUPS=(
  'cycles:u,instructions:u,ls_dispatch.ld_dispatch:u,ls_dispatch.store_dispatch:u,ls_dispatch.ld_st_dispatch:u'
  'cycles:u,instructions:u,L1-dcache-loads:u,ls_mab_alloc.load_store_allocations:u,ls_mab_alloc.hardware_prefetcher_allocations:u'
  'cycles:u,instructions:u,ex_no_retire.load_not_complete:u,ex_no_retire.not_complete:u,ex_no_retire.all:u'
  'cycles:u,instructions:u,de_dis_dispatch_token_stalls1.store_queue_rsrc_stall:u,de_dis_dispatch_token_stalls1.load_queue_rsrc_stall:u'
  'cycles:u,instructions:u,ls_alloc_mab_count:u,ls_any_fills_from_sys.all:u,ls_any_fills_from_sys.dram_io_all:u'
)
: > "$OUT"
for r in $(seq 1 "$REPS"); do
  for g in "${EVGROUPS[@]}"; do
    echo "### rep=$r label=$LABEL group=$g" >> "$OUT"
    perf stat -x, -e "$g" -- taskset -c "$CPU" "./$BIN/$LABEL" \
      bench 16 1 100000 "$FENS" nodes "$NET" 2>>"$OUT" >/dev/null
  done
done
echo "[census48] -> $OUT"
