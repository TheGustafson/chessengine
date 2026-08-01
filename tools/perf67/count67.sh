#!/usr/bin/env bash
# perf study 67 — aggregate counters for one binary on the study frame.
#   bash tools/perf67/count67.sh <binary> [events]
set -euo pipefail
cd "$(dirname "$0")/../.."
BIN=${1:?binary}
EV=${2:-instructions:u,branches:u,branch-misses:u,cycles:u,L1-dcache-loads:u,L1-dcache-load-misses:u,ic_tag_hit_miss.instruction_cache_miss:u,ic_tag_hit_miss.all_instruction_cache_accesses:u}
CPU=${P67_CPU:-2}
taskset -c "$CPU" perf stat -x, -e "$EV" -- \
  "$BIN" bench 16 1 100000 tools/nnue/calib/wl50.fens nodes engines/canonical/net.cenn 2>&1 >/dev/null \
  | grep -v '^#'
