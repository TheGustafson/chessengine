#!/usr/bin/env bash
# perf study 56 -- SMALL HOT OUT-OF-LINE helpers, mechanism A's licensed
# population.
#
# Study 55 measured mechanism A (`#[inline(always)]` on a single-call-site or
# duplicated helper) as a function of BODY SIZE: a win under ~450 B, a loss
# above ~800 B (`do_move` at 3516 B is +25.7 i/n). So the candidate set is
# exactly "still out of line, small, and hot" -- and study 55 named it by hand.
# This joins the three facts mechanically: `nm` for size, `perf` for weight.
#
#   bash tools/perf56/small56.sh <binary> <perf.data>
set -euo pipefail
cd "$(dirname "$0")/../.."
BIN=${1:?binary}
DATA=${2:?perf.data}
MAXB=${P56_MAXB:-460}

perf report -q -i "$DATA" --percent-limit 0.02 --sort symbol 2>/dev/null \
  | sed -E 's/^\s*([0-9.]+)%\s+\[\.\]\s+//; s/\s+-\s+-\s*$//' \
  | awk '{pct=$0; sub(/ .*/,"",pct); $1=""; sub(/^ /,""); print pct"\t"$0}' \
  > /tmp/p56_prof.$$ || true

# `nm -S` gives the size of every symbol that is still a real function.
nm -C -S --size-sort "$BIN" 2>/dev/null \
  | awk '{sz=strtonum("0x"$2); $1="";$2="";$3=""; sub(/^ +/,""); print sz"\t"$0}' \
  > /tmp/p56_size.$$

echo "still-out-of-line helpers under ${MAXB} B, ranked by profile weight:"
printf '%8s %7s  %s\n' "instr%" "bytes" "symbol"
while IFS=$'\t' read -r pct sym; do
  sz=$(awk -F'\t' -v s="$sym" '$2==s {print $1; exit}' /tmp/p56_size.$$)
  [ -z "${sz:-}" ] && continue
  [ "$sz" -gt "$MAXB" ] && continue
  printf '%8s %7s  %s\n' "$pct" "$sz" "${sym:0:96}"
done < /tmp/p56_prof.$$
rm -f /tmp/p56_prof.$$ /tmp/p56_size.$$
