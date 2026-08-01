#!/usr/bin/env bash
# tools/perf70/screen70.sh — screen ordering arms on NODES TO A FIXED DEPTH.
#
# One `arm70` build, many arms: the metric is a deterministic integer, so the
# runtime cost of the env read is invisible to it and a per-arm build would buy
# nothing. Prints each arm's three totals and its percentage against the null
# (the empty arm, which MUST reproduce the shipped counts).
set -euo pipefail
cd "$(dirname "$0")/../.."

BIN=${BIN:-target/arm/release/chessengine}
NET=${NET:-engines/canonical/net.cenn}

run() { # spec fens depth
    ARCHON_A70="$1" "$BIN" bench 16 1 "$3" "$2" depth "$NET" 2>&1 > /dev/null \
        | sed -n 's/^Nodes searched  : //p'
}

triple() { # spec
    local w u10 u11
    w=$(run "$1" tools/nnue/calib/wl50.fens 13)
    u10=$(run "$1" tools/perf70/uho300.fens 10)
    u11=$(run "$1" tools/perf70/uho300.fens 11)
    echo "$w $u10 $u11"
}

read -r BW BU10 BU11 <<< "$(triple '')"
printf '%-28s %10s %10s %10s   %7s %7s %7s   %s\n' \
    ARM wl50d13 uho300d10 uho300d11 d13% d10% d11% MEAN%
printf '%-28s %10s %10s %10s   %7s %7s %7s   %s\n' \
    '(null)' "$BW" "$BU10" "$BU11" 0.000 0.000 0.000 0.000

for spec in "$@"; do
    read -r w u10 u11 <<< "$(triple "$spec")"
    python3 - "$spec" "$BW" "$BU10" "$BU11" "$w" "$u10" "$u11" <<'PY'
import sys
spec, bw, b10, b11, w, u10, u11 = sys.argv[1:8]
b = [int(bw), int(b10), int(b11)]
a = [int(w), int(u10), int(u11)]
p = [100.0 * (x - y) / y for x, y in zip(a, b)]
print(f"{spec:<28} {a[0]:>10} {a[1]:>10} {a[2]:>10}   "
      f"{p[0]:>+7.3f} {p[1]:>+7.3f} {p[2]:>+7.3f}   {sum(p)/3:>+7.3f}")
PY
done
