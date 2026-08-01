#!/usr/bin/env bash
# perf study 68 — every RIP-relative reference to a NAMED static inside a hot
# symbol, counted.
#
#   bash tools/perf68/statics68.sh <label> <symbol-substring>
#
# The `qtrace` class has a third form after "the argument does the work"
# (study 67) and "the state does the work" (study 68 §3): **the guard does the
# work**. A `LazyLock`/`OnceLock` over an environment variable is a process
# constant, but its deref reads an atomic, so a recursive function reloads it at
# every node and no optimiser may hoist it. Those loads are invisible in source
# review and obvious in the disassembly, which is what this prints:
# `objdump`'s `# <addr> <symbol>` annotation on every RIP-relative operand,
# rolled up by symbol.
set -euo pipefail
cd "$(dirname "$0")/../.."
L=${1:?binary label under engines/bin/}
S=${2:?symbol substring}
B="engines/bin/$L"
read -r START SIZE <<<"$(nm -SC "$B" | grep -F -- "$S" | grep -v '\.cold' \
  | awk '{print $1, $2}' | head -1)"
[ -n "${START:-}" ] || { echo "no symbol matching '$S' in $B" >&2; exit 1; }
END=$(printf '%d' $((0x$START + 0x$SIZE)))
echo "# $S in $L: 0x$START + 0x$SIZE bytes"
objdump -d --start-address="0x$START" --stop-address="$END" "$B" \
  | grep -oE '# [0-9a-f]+ <[^>]+>' \
  | sed -E 's/^# [0-9a-f]+ <//; s/>$//; s/\+0x[0-9a-f]+$//' \
  | sort | uniq -c | sort -rn | head -40
