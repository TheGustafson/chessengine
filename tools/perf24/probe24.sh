#!/usr/bin/env bash
# tools/perf24/probe24.sh <label> [frame]
#
# Perf study 24's screening instrument: ONE run under `perf stat`, on a
# fixed-node tree, reporting instructions/node to four decimals plus the node
# total and the eval hash (the two identity checks the commission requires on
# every arm, rejects included).
#
# frames (all --features sota,arch-d48x, canonical net):
#   bench16   bench 16 1 16 default depth   -- the OWNER'S RANKING FRAME
#   wl100k    bench 16 1 100000 wl50 nodes  -- study 19's probe frame
#
# Instructions at fixed nodes are EXACT on this box (study 19 null: 0.00016
# i/n). Cycles from this script are NOT usable -- use pair24.sh for cycles.
set -euo pipefail
cd "$(dirname "$0")/../.."
ROOT=$PWD

label=${1:?usage: probe24.sh <label> [bench16|wl100k]}
frame=${2:-bench16}
BIN=$ROOT/perf24/bin/$label
NET=$ROOT/engines/canonical/net.cenn
FENS=$ROOT/tools/nnue/calib/wl50.fens

[ -x "$BIN" ] || { echo "no binary $BIN" >&2; exit 1; }

case $frame in
    bench16) args=(bench 16 1 16 default depth "$NET") ;;
    wl100k)  args=(bench 16 1 100000 "$FENS" nodes "$NET") ;;
    *) echo "unknown frame $frame" >&2; exit 2 ;;
esac

tmp=$(mktemp -d)
trap 'rm -rf "$tmp"' EXIT

perf stat -x, -e instructions:u,cycles:u -o "$tmp/stat" \
    "$BIN" "${args[@]}" >/dev/null 2>"$tmp/bench"

nodes=$(sed -n 's/^Nodes searched  : //p' "$tmp/bench")
ins=$(awk -F, '$3=="instructions:u"{print $1}' "$tmp/stat")
cyc=$(awk -F, '$3=="cycles:u"{print $1}' "$tmp/stat")
hash=$("$BIN" nnue-eval "$NET" < "$FENS" | sha256sum | cut -d' ' -f1)

awk -v l="$label" -v f="$frame" -v n="$nodes" -v i="$ins" -v c="$cyc" -v h="$hash" \
    'BEGIN{printf "%-22s %-8s nodes=%s  i=%s  i/n=%.4f  c/n=%.4f  hash=%s\n",
           l, f, n, i, i/n, c/n, substr(h,1,16)}'
