#!/usr/bin/env bash
# perf study 11 — total i/n of ONE opcode across the WHOLE profile.
#
#   bash opcount11.sh <perf.data> <opcode> <engine-i-per-node>
#
# Study 10 measured `vpmovsxbw` inside ONE symbol per engine and compared the
# two.  That is only valid if neither engine executes the opcode anywhere else
# — an assumption study 10 listed as a threat to validity (§6.5) and never
# discharged.  This walks EVERY symbol above 0.05% and sums the opcode's
# sample weight, so the comparison rests on a whole-engine count.
set -euo pipefail
DATA=${1:?perf.data}
OP=${2:?opcode}
PERNODE=${3:?engine instructions per node}

rep=$(mktemp)
perf report -i "$DATA" --stdio --no-children --percent-limit 0.05 -s sym 2>/dev/null > "$rep"

# "  12.34%  [.] symbol name" -> "12.34<TAB>symbol name"
syms=$(sed -n 's/^ *\([0-9.]*\)% *\[\.\] *\(.*\)$/\1\t\2/p' "$rep")
rm -f "$rep"

total=0
printf '%-9s %-9s %s\n' "sym-share" "op-i/n" "symbol"
while IFS=$'\t' read -r share sym; do
  [ -n "$sym" ] || continue
  ann=$(mktemp)
  perf annotate -i "$DATA" --stdio -s "$sym" --percent-limit 0 2>/dev/null > "$ann" || true
  # local percent of this opcode within the symbol, and the symbol's local total
  read -r loc tot < <(awk -v op="$OP" '
    match($0, /^ *([0-9]+\.[0-9]+) *: *[0-9a-f]+: +([a-z0-9.]+)/, m) {
      t += m[1]; if (m[2] == op) l += m[1]
    } END { printf "%.6f %.6f\n", l+0, t+0 }' "$ann")
  rm -f "$ann"
  if [ "$(echo "$tot > 0 && $loc > 0" | bc -l)" = "1" ]; then
    ipn=$(echo "scale=4; $share / 100 * $PERNODE * $loc / $tot" | bc -l)
    total=$(echo "$total + $ipn" | bc -l)
    printf '%-9s %-9.1f %s\n' "$share%" "$ipn" "${sym:0:88}"
  fi
done <<< "$syms"

printf '\nTOTAL %s = %.1f i/n\n' "$OP" "$total"
