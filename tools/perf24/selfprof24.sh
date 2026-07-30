#!/usr/bin/env bash
# tools/perf24/selfprof24.sh <label> <event> [frame]
#
# ARCHON'S OWN PROFILE -- the ranking instrument this study is required to use
# (the commission forbids ranking by diff against Stockfish).
set -euo pipefail
cd "$(dirname "$0")/../.."
ROOT=$PWD
label=${1:?}; ev=${2:-cycles:u}; frame=${3:-bench16}
NET=$ROOT/engines/canonical/net.cenn
FENS=$ROOT/tools/nnue/calib/wl50.fens
case $frame in
    bench16) args=(bench 16 1 16 default depth "$NET") ;;
    wl100k)  args=(bench 16 1 100000 "$FENS" nodes "$NET") ;;
esac
out=$ROOT/perf24/$label-${ev%%:*}-$frame.data
taskset -c 3 perf record -F 3999 -e "$ev" --call-graph=none -o "$out" -- \
    "$ROOT/perf24/bin/$label" "${args[@]}" >/dev/null 2>&1
perf report -i "$out" --stdio --no-children --percent-limit 0.30 2>/dev/null \
    | grep -E '^\s+[0-9]' | head -40
