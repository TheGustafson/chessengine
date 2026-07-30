#!/usr/bin/env bash
# perf study 44 — the BACK-END census.
#
#   bash tools/perf44/census44.sh <reps> <binary-label>...
#   P44_CENSUS=  P44_BUDGET=  P44_CPU=  P44_NET=  P44_FENS=  P44_GROUPS=  P44_BIN=
#
# Study 41 closed the front end (19.9% of dispatch slots, ~8.5 Elo if abolished
# entirely) and found the engine is BACK-END bound: 49.0% of slots, and
# `ex_no_retire.not_complete` = 2815 cycles/node = 64.5% of all cycles. Nobody
# has decomposed that. This walks the Zen 4 back-end event set over the same
# fixed-node frame study 41 used, so the two censuses stitch.
#
# Zen 4 has 6 general-purpose PMCs. Every group below is <= 6 events so perf
# never multiplexes; the reducer still checks the enabled/running ratio that
# `perf stat -x,` prints and refuses a scaled row.
#
# Every group carries cycles:u and instructions:u so groups can be stitched
# into one per-node frame without assuming run-to-run stability, and so the
# frame gate (instructions/node) is re-checked on every single run.
set -euo pipefail
cd "$(dirname "$0")/../.."
REPS=${1:?reps}; shift
NET=${P44_NET:-engines/canonical/net.cenn}
FENS=${P44_FENS:-tools/nnue/calib/wl50.fens}
CPU=${P44_CPU:-2}
BUDGET=${P44_BUDGET:-100000}
BIN=${P44_BIN:-bin44}
OUT=${P44_CENSUS:-docs/research/perf-artifacts-44/census.txt}
mkdir -p "$(dirname "$OUT")"
tmp=$(mktemp -d); trap 'rm -rf "$tmp"' EXIT

# --- the groups -------------------------------------------------------------
# g_ret    THE PARTITION. Zen 4's ex_no_retire.* buckets tile every cycle in
#          which nothing retired; load_not_complete is the memory SUBSET of
#          not_complete, given by hardware. cycles - all = retiring cycles.
# g_ret2   the closure check: ex_no_retire.all against its own parts.
# g_slot   study 41's slot view, re-derived here rather than inherited (rule 8).
# g_disp   the other half of the slot sum: ops actually dispatched.
# g_tok1   WHICH structure blocks dispatch: load queue, store queue, FP sched,
#          FP register file. An FP-scheduler stall is the port-pressure
#          signature; a load-queue stall is the memory signature.
# g_tok2   integer schedulers (4 of them) — int-side port pressure.
# g_tok3   the remaining token stalls incl. ROB-full (retire_token_stall).
# g_fill   WHERE demand loads are satisfied: L2, sibling cache, DRAM.
# g_any    all fills including hardware prefetch, plus MAB allocation source.
# g_mab    memory-level parallelism: MAB occupancy per cycle, load/store mix,
#          store-to-load forwarding conflicts.
# g_fp     execution mix: 512-bit vs 256-bit packed uops, total macro-ops.
# g_l2     L2 hit/miss for data, and how much of L2 traffic is HW prefetch.
declare -A GROUP
GROUP[g_ret]="cycles:u,instructions:u,ex_no_retire.not_complete:u,ex_no_retire.load_not_complete:u,ex_no_retire.other:u,ex_no_retire.empty:u"
GROUP[g_ret2]="cycles:u,instructions:u,ex_no_retire.all:u,ex_no_retire.thread_not_selected:u,ex_no_retire.not_complete:u,ex_no_retire.load_not_complete:u"
GROUP[g_slot]="cycles:u,instructions:u,de_no_dispatch_per_slot.no_ops_from_frontend:u,de_no_dispatch_per_slot.backend_stalls:u,de_no_dispatch_per_slot.smt_contention:u"
GROUP[g_disp]="cycles:u,instructions:u,de_src_op_disp.op_cache:u,de_src_op_disp.decoder:u,de_src_op_disp.loop_buffer:u,ex_ret_ops:u"
GROUP[g_tok1]="cycles:u,instructions:u,de_dis_dispatch_token_stalls1.load_queue_rsrc_stall:u,de_dis_dispatch_token_stalls1.store_queue_rsrc_stall:u,de_dis_dispatch_token_stalls1.fp_sch_rsrc_stall:u,de_dis_dispatch_token_stalls1.fp_reg_file_rsrc_stall:u"
GROUP[g_tok2]="cycles:u,instructions:u,de_dis_dispatch_token_stalls2.int_sch0_token_stall:u,de_dis_dispatch_token_stalls2.int_sch1_token_stall:u,de_dis_dispatch_token_stalls2.int_sch2_token_stall:u,de_dis_dispatch_token_stalls2.int_sch3_token_stall:u"
GROUP[g_tok3]="cycles:u,instructions:u,de_dis_dispatch_token_stalls2.retire_token_stall:u,de_dis_dispatch_token_stalls1.int_phy_reg_file_rsrc_stall:u,de_dis_dispatch_token_stalls1.taken_brnch_buffer_rsrc:u,de_dis_dispatch_token_stalls1.fp_flush_recovery_stall:u"
GROUP[g_fill]="cycles:u,instructions:u,ls_dmnd_fills_from_sys.local_l2:u,ls_dmnd_fills_from_sys.local_ccx:u,ls_dmnd_fills_from_sys.dram_io_near:u,ls_dmnd_fills_from_sys.near_cache:u"
GROUP[g_any]="cycles:u,instructions:u,ls_any_fills_from_sys.all:u,ls_any_fills_from_sys.local_l2:u,ls_mab_alloc.load_store_allocations:u,ls_mab_alloc.hardware_prefetcher_allocations:u"
GROUP[g_mab]="cycles:u,instructions:u,ls_alloc_mab_count:u,ls_dispatch.ld_dispatch:u,ls_dispatch.store_dispatch:u,ls_bad_status2.stli_other:u"
GROUP[g_fp]="cycles:u,instructions:u,fp_ops_retired_by_width.pack_512_uops_retired:u,fp_ops_retired_by_width.pack_256_uops_retired:u,fp_ops_retired_by_width.all:u,fp_ops_retired_by_width.scalar_uops_retired:u"
GROUP[g_l2]="cycles:u,instructions:u,l2_cache_req_stat.dc_access_in_l2:u,l2_cache_req_stat.dc_hit_in_l2:u,l2_request_g1.all_dc:u,l2_request_g1.l2_hw_pf:u"
EVGROUPS=${P44_GROUPS:-g_ret g_ret2 g_slot g_disp g_tok1 g_tok2 g_tok3 g_fill g_any g_mab g_fp g_l2}

for r in $(seq "$REPS"); do
  for label in "$@"; do
    for g in $EVGROUPS; do
      nice -n 19 perf stat -x, -e "${GROUP[$g]}" -o "$tmp/stat" -- \
        taskset -c "$CPU" "./$BIN/$label" \
        bench 16 1 "$BUDGET" "$FENS" nodes "$NET" \
        >/dev/null 2>"$tmp/bench"
      nodes=$(sed -n 's/^Nodes searched  : //p' "$tmp/bench" | tail -1)
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
echo "[census44] $REPS reps x $# binaries -> $OUT"
