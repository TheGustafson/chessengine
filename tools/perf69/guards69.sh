#!/usr/bin/env bash
# perf study 69 — find the `LazyLock`/`OnceLock` GUARD BRANCHES inside a hot
# symbol and print each one's branch-site offset.
#
#   bash tools/perf69/guards69.sh <label> <symbol-substring> <static+stateoff>
#     e.g. ... s69-b0 'search::<chessengine::search::NonPv>' 'TABLES+0x1040'
#
# Study 68's `statics68.sh` counts REFERENCES to a named static; `refoffs69.sh`
# splits those references by BYTE OFFSET, which is what tells the `Once` state
# word apart from the data it guards. This takes the state word's offset and
# emits, for each load of it, the offset of the conditional branch that
# consumes it (within the next few instructions -- the scheduler may separate
# them). Pair those offsets with `mispagg69.py`'s `seen` column and the guard
# stops being "22 references" and becomes a measured executions-per-node
# number, which is the only form an arm may be priced in (study 68 §4.3:
# counting a sequence is not measuring it).
set -euo pipefail
cd "$(dirname "$0")/../.."
L=${1:?binary label under engines/bin/}
S=${2:?symbol substring}
ST=${3:?static+offset substring, e.g. TABLES+0x1040}
B="engines/bin/$L"
read -r START SIZE <<<"$(nm -SC "$B" | grep -F -- "$S" | grep -v '\.cold' \
  | awk '{print $1, $2}' | head -1)"
[ -n "${START:-}" ] || { echo "no symbol matching '$S' in $B" >&2; exit 1; }
END=$((0x$START + 0x$SIZE))
objdump -d --start-address="0x$START" --stop-address="$END" "$B" \
  | awk -v base=$((0x$START)) -v st="$ST" '
    index($0, st) > 0 { look = 6; split($1, a, ":"); ldoff = strtonum("0x" a[1]) - base; next }
    look > 0 {
      look--
      if ($0 ~ /\tj(ne|e|a|b|ae|be|g|l|ge|le|s|ns)? /) {
        split($1, a, ":"); addr = strtonum("0x" a[1]) - base
        printf "load +0x%-6x  branch +0x%-6x  %s\n", ldoff, addr, substr($0, index($0, "\t") + 1)
        look = 0
      }
    }'
