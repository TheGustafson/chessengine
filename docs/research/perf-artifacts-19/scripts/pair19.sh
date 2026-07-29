#!/usr/bin/env bash
# perf study 19 — THE SECOND METRIC AXIS: cycles:u and cache-misses:u, PAIRED.
#
#   bash pair19.sh <labelA> <labelB> [rounds]
#
# Both binaries must already be stashed in bin/ by probe19.sh.
#
# WHY THIS EXISTS. The commission requires cycles and cache-misses alongside
# instructions, because instruction parity alone still leaves us ~1.16x slower.
# But study 19's null panel showed a SEQUENTIAL cycles measurement cannot carry
# a verdict on this box:
#
#     null-1        (3 runs, min)   4425.7 cyc/n
#     null-2-rebuild(7 runs, min)   4810.5 cyc/n     +384.8, +8.7%
#
# — two builds of BIT-IDENTICAL source, with instructions:u identical to the
# digit. A GPU trainer shares the box and its load drifts on a timescale longer
# than one probe run, so taking more repeats does not help: null-2 had MORE
# repeats and a worse minimum. That is rule 1's gate firing, and the fix is to
# change instrument rather than to quote the number with a caveat.
#
# THE FIX: interleave. Round r runs A then B back to back, so both see the same
# ambient load; the statistic is the per-round PAIRED RATIO B/A, and the report
# is its median across rounds plus the full spread. Drift that is common to the
# pair divides out; only drift WITHIN a round survives, and that is bounded by
# the round-to-round spread this script prints — which is the null for it.
#
# Order is alternated (A,B then B,A) on odd/even rounds so a systematic
# "first run of a pair is colder" effect cannot masquerade as a delta.
set -euo pipefail
A=${1:?labelA}
B=${2:?labelB}
ROUNDS=${3:-8}
BINDIR=docs/research/perf-artifacts-19/bin
OUT=docs/research/perf-artifacts-19/pairs.txt
BA="$BINDIR/$A"
BB="$BINDIR/$B"
[ -x "$BA" ] || { echo "missing $BA (run probe19.sh $A)"; exit 1; }
[ -x "$BB" ] || { echo "missing $BB (run probe19.sh $B)"; exit 1; }

tmp=$(mktemp -d)
run() {  # run <bin> <tag>
  nice -n 19 perf stat -x, -e instructions:u,cycles:u,cache-misses:u -o "$tmp/$2" -- \
    taskset -c 6 "$1" bench 16 1 100000 tools/nnue/calib/wl50.fens nodes \
    engines/canonical/net.cenn > /dev/null 2> "$tmp/$2.eng"
}
rows=""
for r in $(seq 1 "$ROUNDS"); do
  if [ $((r % 2)) -eq 1 ]; then run "$BA" "a.$r"; run "$BB" "b.$r"
  else                          run "$BB" "b.$r"; run "$BA" "a.$r"; fi
  rows="$rows $tmp/a.$r:$tmp/b.$r"
done

python3 - "$A" "$B" "$OUT" $rows <<'PY'
import statistics, sys
A, B, OUT = sys.argv[1:4]
def read(p):
    d = {}
    for ln in open(p):
        f = ln.strip().split(",")
        if len(f) > 2 and f[2]:
            d[f[2]] = int(f[0])
    n = int([l for l in open(p + ".eng") if "Nodes searched" in l][-1].split(":")[1])
    return d, n
rc, rm, ri = [], [], []
for pair in sys.argv[4:]:
    pa, pb = pair.split(":")
    da, na = read(pa); db, nb = read(pb)
    assert na == nb, f"node counts differ: {na} vs {nb}"
    rc.append((db["cycles:u"] / nb) / (da["cycles:u"] / na))
    rm.append((db["cache-misses:u"] / nb) / (da["cache-misses:u"] / na))
    ri.append((db["instructions:u"] / nb) / (da["instructions:u"] / na))
def s(v):
    return f"med={statistics.median(v):.4f} min={min(v):.4f} max={max(v):.4f} spread={100*(max(v)-min(v)):.2f}pp"
out = (f"pair {B} / {A}  n={len(rc)} rounds\n"
       f"  cycles/node      {s(rc)}\n"
       f"  cache-miss/node  {s(rm)}\n"
       f"  instr/node       {s(ri)}   <- must be flat; it is the within-pair control\n")
print(out, end="")
open(OUT, "a").write(out)
PY
rm -rf "$tmp"
