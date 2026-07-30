#!/usr/bin/env bash
# perf study 37 — did the weight tables actually get 2 MiB pages?
#
#   bash tools/perf37/thpcheck37.sh <binary>
#
# `advise_huge` returning true only says the kernel RECORDED the hint. Two
# independent observations of whether it acted on it:
#
#   1. /proc/<pid>/smaps_rollup's AnonHugePages, read while the engine sits at
#      the UCI prompt with its net loaded. This is the allocation-side fact.
#   2. `ls_l1_d_tlb_miss.tlb_reload_2m_l2_hit` over a real search — a DISCRETE
#      outcome (0 vs not-0) rather than a percentage, which is what CLAUDE.md's
#      rule 3 asks for before any cp-style mean.
#
# Both are printed, because they can disagree: fragmented memory can leave the
# hint recorded and unhonoured, and that shows up as AnonHugePages far below the
# table size.
set -euo pipefail
cd "$(dirname "$0")/../.."
BIN=${1:?binary}
NET=engines/canonical/net.cenn
FENS=tools/nnue/calib/wl50.fens

fifo=$(mktemp -u); mkfifo "$fifo"
"$BIN" < "$fifo" > /dev/null 2>&1 &
pid=$!
exec 9> "$fifo"
printf 'uci\nsetoption name EvalFile value %s\nisready\n' "$NET" >&9
# `isready` cannot be awaited on a discarded stdout, so give the load a moment.
sleep 5
echo "== /proc/$pid/smaps_rollup"
grep -E 'AnonHugePages|^Rss|^Pss' "/proc/$pid/smaps_rollup" | sed 's/^/   /'
printf 'quit\n' >&9
exec 9>&-
wait "$pid" 2>/dev/null || true
rm -f "$fifo"

echo "== TLB reload split over the study frame (10k nodes)"
nice -n 19 perf stat -x, \
  -e ls_l1_d_tlb_miss.tlb_reload_4k_l2_hit:u,ls_l1_d_tlb_miss.tlb_reload_2m_l2_hit:u,ls_l1_d_tlb_miss.all_l2_miss:u \
  -- taskset -c "${P37_CPU:-2}" "$BIN" bench 16 1 10000 "$FENS" nodes "$NET" 2>&1 >/dev/null \
  | awk -F, '$1 ~ /^[0-9]+$/ { printf "   %-46s %s\n", $3, $1 }'
