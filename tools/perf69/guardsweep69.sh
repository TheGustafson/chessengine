#!/usr/bin/env bash
# perf study 69 — THE WHOLE-BINARY GUARD CENSUS. The third dead-work form,
# swept exhaustively instead of one static at a time.
#
#   bash tools/perf69/guardsweep69.sh <label> <ibs.txt> <ops-per-node>
#
# Study 67 found "the argument does the work", study 68 "the state does the
# work" and "the guard does the work". Each was found by reading one symbol's
# disassembly. This finds ALL of them: every `mov <static>(%rip),%reg` (or
# `cmp $imm, <static>(%rip)`) whose value is consumed by a conditional branch
# within three instructions -- the machine shape of a `LazyLock`/`OnceLock`
# deref or a process-constant re-test -- rolled up by the static it reads and
# weighted by IBS dispatched-op samples, so the output is ORDERED BY MEASURED
# COST PER NODE rather than by reference count.
#
# A guard that never executes costs nothing and must not be "fixed"; a guard at
# 5 ops/node is worth a diff. Only this instrument can tell them apart, because
# an LBR profile cannot see a never-taken branch at all.
set -euo pipefail
cd "$(dirname "$0")/../.."
L=${1:?label}; IBS=${2:?ibs dump}; OPN=${3:?ops per node}
B="engines/bin/$L"
tmp=$(mktemp); trap 'rm -f "$tmp"' EXIT
objdump -dC "$B" | awk '
  /^[0-9a-f]+ <.*>:$/ {
    sym = $2; gsub(/^</, "", sym); gsub(/>:$/, "", sym)
    if (match(sym, /\+0x[0-9a-f]+$/)) {
      off = strtonum("0x" substr(sym, RSTART + 3)); sym = substr(sym, 1, RSTART - 1)
      base = strtonum("0x" $1) - off
    } else { base = strtonum("0x" $1) }
    next
  }
  # candidate: a RIP-relative read of a NAMED static into a register
  /^ *[0-9a-f]+:/ && /%rip/ && /# [0-9a-f]+ </ && (/\tmov/ || /\tcmp/) {
    st = $0; sub(/.*# [0-9a-f]+ </, "", st); sub(/>.*/, "", st)
    split($1, a, ":"); cand = strtonum("0x" a[1]) - base
    csym = sym; cst = st; look = 3; next
  }
  look > 0 {
    look--
    if ($0 ~ /\tj(ne|e|a|b|ae|be|g|l|ge|le|s|ns) /) {
      printf "%s\t%s+0x%x\n", cst, csym, cand; look = 0
    } else if ($0 !~ /\ttest/ && $0 !~ /\tcmp/ && $0 !~ /\tset/) { look = 0 }
  }' > "$tmp"
echo "# $(wc -l < "$tmp") guard-shaped static reads in $L"
python3 - "$tmp" "$IBS" "$OPN" <<'PY'
import collections, sys
rows = [l.rstrip("\n").split("\t") for l in open(sys.argv[1]) if "\t" in l]
hist = collections.Counter(); tot = 0
for line in open(sys.argv[2], errors="replace"):
    p = line.split(None, 1)
    if len(p) < 2: continue
    tot += 1; hist[p[1].strip()] += 1
per = float(sys.argv[3]) / tot
cost = collections.Counter(); n = collections.Counter()
for st, site in rows:
    cost[st] += hist[site]; n[st] += 1
print(f"{'ops/node':>10}{'sites':>7}  guarded static")
tc = 0
for st, c in cost.most_common(25):
    if c == 0 and tc > 0: break
    tc += c
    print(f"{c*per:>10.4f}{n[st]:>7}  {st}")
print(f"{sum(cost.values())*per:>10.4f}{len(rows):>7}  TOTAL")
PY
