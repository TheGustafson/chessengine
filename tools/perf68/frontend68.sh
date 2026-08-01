#!/usr/bin/env bash
# perf study 68 — the frontend LEVEL panel: one binary, three counter groups,
# printed as raw totals for a per-node derivation.
#
#   bash tools/perf68/frontend68.sh <label> [nodes]
#
# Three groups because six PMCs cannot hold all of it at once, and multiplexing
# a frontend counter on a shared box is how a study invents a number. Each
# group is a separate full run of the same deterministic frame, so the totals
# are directly comparable within a group and only approximately across groups.
set -euo pipefail
cd "$(dirname "$0")/../.."
L=${1:?binary label under engines/bin/}
N=${2:-100000}
CPU=${P68_CPU:-2}
NET=engines/canonical/net.cenn
FENS=tools/nnue/calib/wl50.fens
run() {
  echo "### $1"
  taskset -c "$CPU" perf stat -e "$1" -- \
    "engines/bin/$L" bench 16 1 "$N" "$FENS" nodes "$NET" 2>&1 >/dev/null \
    | grep -E '^\s+[0-9,]+|seconds'
}
# group 1: op-cache vs decoder delivery, and the op cache's own hit rate
run 'de_src_op_disp.all:u,de_src_op_disp.decoder:u,de_src_op_disp.op_cache:u,op_cache_hit_miss.op_cache_hit:u,op_cache_hit_miss.op_cache_miss:u'
# group 2: instruction fetch
run 'instructions:u,ic_cache_fill_l2:u,ic_cache_fill_sys:u,ic_tag_hit_miss.instruction_cache_miss:u,ic_tag_hit_miss.all_instruction_cache_accesses:u'
# group 3: dispatch slots and branches
run 'de_no_dispatch_per_slot.no_ops_from_frontend:u,de_no_dispatch_per_slot.backend_stalls:u,de_op_queue_empty:u,ex_ret_brn_misp:u,bp_de_redirect:u'
