#!/usr/bin/env bash
# perf study 41 — the FRONT-END census.
#
#   bash tools/perf41/census41.sh <reps> <binary-label>...
#   P41_CENSUS=  P41_BUDGET=  P41_CPU=  P41_NET=  P41_FENS=  P41_GROUPS=
#
# Study 40 measured `ex_no_retire.empty` = 192.91 cycles/node (4.7% of the
# engine) and attributed only a sliver of it to branch mispredicts. Nobody has
# ever measured this engine's front end, so this walks the whole Zen 4
# front-end event set over the SAME fixed-node frame.
#
# Zen 4 has 6 general-purpose PMCs. Every group below is <= 6 events so perf
# never multiplexes; the reducer still checks the enabled/running ratio that
# `perf stat -x,` prints in field 5-6 and refuses a scaled row.
#
# Every group carries `cycles:u` so the groups can be stitched into one
# per-node frame without assuming run-to-run stability of the denominator.
set -euo pipefail
cd "$(dirname "$0")/../.."
REPS=${1:?reps}; shift
NET=${P41_NET:-engines/canonical/net.cenn}
FENS=${P41_FENS:-tools/nnue/calib/wl50.fens}
CPU=${P41_CPU:-2}
BUDGET=${P41_BUDGET:-100000}
OUT=${P41_CENSUS:-docs/research/perf-artifacts-41/census.txt}
mkdir -p "$(dirname "$OUT")"
tmp=$(mktemp -d); trap 'rm -rf "$tmp"' EXIT

# --- the groups -------------------------------------------------------------
# g_retire  where do cycles go when nothing retires (the 192.91 and its family)
# g_slot    the dispatch-slot view of the same question (6 slots/cycle)
# g_ic      L1 instruction cache: accesses, misses, and where the fill came from
# g_tlb     instruction-side TLB: L1 ITLB hits by page size, L2 hits, walks
# g_oc      op cache vs decoder vs loop buffer — the delivery path
# g_br      branches and mispredicts, with the empty/SMT pair for the exchange
declare -A GROUP
GROUP[g_retire]="cycles:u,instructions:u,ex_no_retire.empty:u,ex_no_retire.thread_not_selected:u,ex_no_retire.not_complete:u,ex_no_retire.other:u"
GROUP[g_slot]="cycles:u,instructions:u,de_no_dispatch_per_slot.no_ops_from_frontend:u,de_no_dispatch_per_slot.backend_stalls:u,de_no_dispatch_per_slot.smt_contention:u"
GROUP[g_ic]="cycles:u,instructions:u,ic_tag_hit_miss.all_instruction_cache_accesses:u,ic_tag_hit_miss.instruction_cache_miss:u,ic_cache_fill_l2:u,ic_cache_fill_sys:u"
GROUP[g_tlb]="cycles:u,instructions:u,bp_l1_tlb_fetch_hit.if4k:u,bp_l1_tlb_fetch_hit.if2m:u,bp_l1_tlb_miss_l2_tlb_hit:u,bp_l1_tlb_miss_l2_tlb_miss.all:u"
GROUP[g_oc]="cycles:u,instructions:u,de_src_op_disp.op_cache:u,de_src_op_disp.decoder:u,de_src_op_disp.loop_buffer:u,op_cache_hit_miss.op_cache_miss:u"
GROUP[g_br]="cycles:u,instructions:u,branches:u,ex_ret_brn_misp:u,ex_no_retire.empty:u,ex_no_retire.thread_not_selected:u"
EVGROUPS=${P41_GROUPS:-g_retire g_slot g_ic g_tlb g_oc g_br}

for r in $(seq "$REPS"); do
  for label in "$@"; do
    for g in $EVGROUPS; do
      nice -n 19 perf stat -x, -e "${GROUP[$g]}" -o "$tmp/stat" -- \
        taskset -c "$CPU" "./bin34/$label" \
        bench 16 1 "$BUDGET" "$FENS" nodes "$NET" \
        >/dev/null 2>"$tmp/bench"
      nodes=$(sed -n 's/^Nodes searched  : //p' "$tmp/bench" | tail -1)
      # perf -x, fields: value,unit,event,run_time,enabled_pct,...
      awk -F, -v l="$label" -v r="$r" -v g="$g" -v n="$nodes" '
        $1 ~ /^[0-9]+$/ && $3 != "" {
          pct = ($6 == "" ? 100 : $6)
          printf "%s=%.6f ", $3, $1 / n
          if (pct + 0 < 99.5) printf "SCALED[%s@%s%%] ", $3, pct
        }
        END { print "" }
      ' "$tmp/stat" | sed "s|^|r$r $label $g nodes=$nodes |" >> "$OUT"
    done
  done
done
echo "[census41] $REPS reps x $# binaries -> $OUT"
