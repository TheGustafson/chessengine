#!/usr/bin/env bash
# tools/perf70/nodes70.sh — the study-70 primary metric: NODES TO A FIXED DEPTH.
#
# Discrete and load-immune (design/14 §3.8 reasoning): the number does not
# care that the box is busy, so it needs no null panel and no repetition.
# Two sets, deliberately different in character:
#   wl50  @ d13 — the NNUE gate's own 50 positions (the pinned d13 total)
#   uho300 @ d10/d11 — 300 UHO opening positions, sampled with seed 70
#
# Usage: tools/perf70/nodes70.sh <binary> [label]
set -euo pipefail
cd "$(dirname "$0")/../.."

BIN=${1:?usage: nodes70.sh <binary> [label]}
LABEL=${2:-$(basename "$BIN")}
NET=${NET:-engines/canonical/net.cenn}

run() { # fens depth
    "$BIN" bench 16 1 "$2" "$1" depth "$NET" 2>&1 > /dev/null \
        | sed -n 's/^Nodes searched  : //p'
}

w13=$(run tools/nnue/calib/wl50.fens 13)
u10=$(run tools/perf70/uho300.fens 10)
u11=$(run tools/perf70/uho300.fens 11)
printf '%s\twl50_d13=%s\tuho300_d10=%s\tuho300_d11=%s\n' "$LABEL" "$w13" "$u10" "$u11"
