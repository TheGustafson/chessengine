#!/usr/bin/env bash
# perf study 69 — THE WHOLE-BINARY COST OF ONE GUARD, in ops per node.
#
#   bash tools/perf69/guardcost69.sh <label> <static+off> <ibs.txt> <ops-per-node>
#
# Disassembles the ENTIRE `.text`, finds every instruction that references the
# given `static+offset` (a `LazyLock`'s `Once` state word), maps each to
# `symbol+offset`, and sums the IBS op samples landing exactly there. The
# result is the guard's total dispatched-op rate per node across every inlined
# copy in the engine -- the arm's budget, measured before the arm is built
# (study 68 §1: price it first; §4.3: a sequence's length is not its cost).
set -euo pipefail
cd "$(dirname "$0")/../.."
L=${1:?label}; ST=${2:?static+offset}; IBS=${3:?ibs dump}; OPN=${4:?ops per node}
B="engines/bin/$L"
tmp=$(mktemp); trap 'rm -f "$tmp"' EXIT
# `objdump -d` prints each function's own `<sym+0xoff>` header, so track the
# current symbol and emit `sym+0xoff` for every hit -- the exact key IBS uses.
objdump -dC "$B" | awk -v st="$ST" '
  /^[0-9a-f]+ <.*>:$/ {
    sym = $2; gsub(/^</, "", sym); gsub(/>:$/, "", sym)
    if (match(sym, /\+0x[0-9a-f]+$/)) {          # objdump splits big symbols
      base = strtonum("0x" substr(sym, RSTART + 3)); sym = substr(sym, 1, RSTART - 1)
      base = strtonum("0x" $1) - base
    } else { base = strtonum("0x" $1) }
    next
  }
  index($0, st) > 0 && /^ *[0-9a-f]+:/ {
    split($1, a, ":"); printf "%s+0x%x\n", sym, strtonum("0x" a[1]) - base
  }' > "$tmp"
echo "# $(wc -l < "$tmp") instructions reference $ST in $L"
python3 - "$tmp" "$IBS" "$OPN" <<'PY'
import collections, sys
sites = [l.strip() for l in open(sys.argv[1]) if l.strip()]
hist = collections.Counter(); tot = 0
for line in open(sys.argv[2], errors="replace"):
    p = line.split(None, 1)
    if len(p) < 2: continue
    tot += 1; hist[p[1].strip()] += 1
per = float(sys.argv[3]) / tot
bysym = collections.Counter()
n = 0
for s in sites:
    c = hist[s]; n += c
    bysym[s.rsplit("+0x", 1)[0]] += c
print(f"{'ops/node':>10}{'sites':>7}  symbol")
for sym, c in bysym.most_common(20):
    k = sum(1 for s in sites if s.rsplit('+0x',1)[0] == sym)
    print(f"{c*per:>10.4f}{k:>7}  {sym}")
print(f"{n*per:>10.4f}{len(sites):>7}  TOTAL  ({n} samples of {tot})")
PY
