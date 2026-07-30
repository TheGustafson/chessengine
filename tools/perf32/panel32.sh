#!/usr/bin/env bash
# perf study 32 — the counter panel, PAIRED WITHIN ROUND, with a null.
#
#   bash tools/perf32/panel32.sh <base> <arm> [rounds]
#
# Study 29 paid for the shape of this: its prefetch arm read -1.67% cycles at
# n=13 on a ROTATED round-robin and it nearly landed; the paired within-round
# form then showed a byte-DIFFERENT null moving +1.5% while the arms moved
# +1.1% — it was position-in-round, not the arm. So every round here runs
# base, a byte-identical copy of base (the NULL), and the arm back to back, in
# a randomised order, and the reported figure is the MEDIAN over rounds of the
# per-round ratio. A claim is admissible only when it clears the null.
#
# THE GATE ARMS ON CYCLES. Study 20 landed -5.99% cycles with instructions UP
# 0.19%; this change moves BYTES, so instructions are expected to rise.
set -euo pipefail
cd "$(dirname "$0")/../.."
BASE=${1:?base label}
ARM=${2:?arm label}
ROUNDS=${3:-8}
CPU=${P32_CPU:-6}
NET=$PWD/engines/canonical/net.cenn
FENS=$PWD/tools/nnue/calib/wl50.fens
OUT=$PWD/docs/research/perf-artifacts-32
mkdir -p "$OUT" bin32
cp -f "bin32/$BASE" "bin32/${BASE}-null"

EVENTS=cycles:u,instructions:u,ex_no_retire.load_not_complete:u,L1-dcache-loads:u,L1-dcache-load-misses:u
tmp=$(mktemp -d); trap 'rm -rf "$tmp"' EXIT

run() {  # $1 = label -> prints "cyc ins stall loads misses nodes"
  nice -n 19 perf stat -x, -e "$EVENTS" -o "$tmp/s" -- \
    taskset -c "$CPU" "bin32/$1" bench 16 1 100000 "$FENS" nodes "$NET" \
    >/dev/null 2>"$tmp/b"
  local n; n=$(grep -oP 'Nodes searched\s*:\s*\K[0-9]+' "$tmp/b" | tail -1)
  awk -F, -v n="$n" '
    $3=="cycles:u"{c=$1} $3=="instructions:u"{i=$1}
    $3=="ex_no_retire.load_not_complete:u"{s=$1}
    $3=="L1-dcache-loads:u"{l=$1} $3=="L1-dcache-load-misses:u"{m=$1}
    END{printf "%.4f %.4f %.4f %.4f %.4f %d\n", c/n, i/n, s/n, l/n, m/n, n}' "$tmp/s"
}

: > "$tmp/rows"
for r in $(seq "$ROUNDS"); do
  for label in $(printf '%s\n%s\n%s\n' "$BASE" "${BASE}-null" "$ARM" | shuf); do
    echo "$r $label $(run "$label")" >> "$tmp/rows"
  done
  echo "[panel32] round $r done" >&2
done
cp "$tmp/rows" "$OUT/panel-raw.txt"

python3 - "$tmp/rows" "$BASE" "${BASE}-null" "$ARM" <<'PY' | tee "$OUT/panel.txt"
import sys, statistics as st
rows, base, null, arm = sys.argv[1], sys.argv[2], sys.argv[3], sys.argv[4]
per = {}
for line in open(rows):
    f = line.split()
    per.setdefault(int(f[0]), {})[f[1]] = [float(x) for x in f[2:7]] + [int(f[7])]
names = ["cycles/node", "instructions/node", "load-stall/node", "L1d loads/node", "L1d misses/node"]
print(f"{'counter':<20s} {'base':>12s} {'NULL':>10s} {'ARM':>10s} {'arm-neg':>8s}")
print("-" * 64)
for k, nm in enumerate(names):
    b = [v[base][k] for v in per.values() if base in v and arm in v]
    dn = [100*(v[null][k]/v[base][k]-1) for v in per.values() if null in v and base in v]
    da = [100*(v[arm][k]/v[base][k]-1) for v in per.values() if arm in v and base in v]
    neg = sum(1 for x in da if x < 0)
    print(f"{nm:<20s} {st.median(b):12.2f} {st.median(dn):+9.3f}% {st.median(da):+9.3f}% "
          f"{neg:>4d}/{len(da)}")
nb = {v[base][5] for v in per.values() if base in v}
na = {v[arm][5] for v in per.values() if arm in v}
print(f"\nnodes: base {nb} arm {na}  {'(NODE-IDENTICAL)' if nb == na else '(NOT node-identical -- expected, the eval changed)'}")
PY
