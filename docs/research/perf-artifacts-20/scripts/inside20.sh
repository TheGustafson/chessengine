#!/usr/bin/env bash
# perf study 20 — WHERE INSIDE ONE SYMBOL the instructions land.
#
#   bash docs/research/perf-artifacts-20/scripts/inside20.sh <symbol-substring> [event]
#
# Region attribution only.  Per-OPCODE attribution on this box is wrong by
# ~1.44x (perf study 18), so nothing here may be quoted per instruction; the
# point is to say "the mass is in the sort" vs "the mass is in the scoring
# loop" before spending a build on the wrong one.
set -euo pipefail
SYM=${1:?symbol substring}
EV=${2:-instructions:u}
BIN=target-fast20/release/chessengine
D=docs/research/perf-artifacts-20
nice -n 19 perf record -q -e "$EV" -F 3999 -o "$D/perf-inside.data" -- \
  taskset -c 6 "$BIN" bench 16 1 100000 tools/nnue/calib/wl50.fens nodes \
  engines/canonical/net.cenn > /dev/null 2>&1
full=$(perf report -i "$D/perf-inside.data" --stdio -q --no-children --percent-limit 0.2 \
  | grep -oP '\[\.\] \K\S*'"$SYM"'\S*' | head -1)
echo "symbol: $full"
perf annotate -i "$D/perf-inside.data" --stdio --symbol="$full" --percent-type global 2>/dev/null \
  | grep -oP '^\s*\K[0-9]+\.[0-9]+(?=\s*:\s*[0-9a-f]+:)|^\s*:\s*\K[0-9a-f]+(?=:)' > /dev/null 2>&1 || true
perf annotate -i "$D/perf-inside.data" --stdio --symbol="$full" 2>/dev/null > "$D/anno-$SYM.txt"
rm -f "$D/perf-inside.data"
wc -l "$D/anno-$SYM.txt"
