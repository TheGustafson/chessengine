#!/usr/bin/env bash
# tools/perf27/probe27.sh <label> [reps]
#
# ONE exact probe on the study-27 frame:
#   bench 16 1 100000 tools/nnue/calib/wl50.fens nodes engines/canonical/net.cenn
# = 50 FENs x 100k nodes, 1 thread, 16 MB TT, canonical net, sota,arch-d48x.
#
# GATE COUNTERS: instructions:u AND branches:u -- both exact at fixed nodes
# (null on byte-identical binaries: +0.000%). branch-misses:u and
# L1-dcache-load-misses:u are reported as DIRECTIONAL mechanism evidence; their
# own nulls are measured by nullpanel27.sh before any delta is quoted.
#
# CYCLES ARE NOT MEASURABLE ON THIS BOX (four independent confirmations, up to
# a 44-point spread on byte-identical binaries). This script does not print
# them; do not add them.
set -euo pipefail
cd "$(dirname "$0")/../.."
ROOT=$PWD

label=${1:?usage: probe27.sh <label> [reps]}
reps=${2:-1}
BIN=$ROOT/perf27/bin/$label
NET=$ROOT/engines/canonical/net.cenn
FENS=$ROOT/tools/nnue/calib/wl50.fens
OUT=$ROOT/docs/research/perf-artifacts-27/probes.txt
CPU=${P27_CPU:-7}

[ -x "$BIN" ] || { echo "no binary $BIN" >&2; exit 1; }

tmp=$(mktemp -d); trap 'rm -rf "$tmp"' EXIT
hash=$("$BIN" nnue-eval "$NET" < "$FENS" | sha256sum | cut -c1-16)

for r in $(seq "$reps"); do
  nice -n 19 perf stat -x, \
    -e instructions:u,branches:u,branch-misses:u,L1-dcache-load-misses:u \
    -o "$tmp/stat" -- \
    taskset -c "$CPU" "$BIN" bench 16 1 100000 "$FENS" nodes "$NET" \
    >/dev/null 2>"$tmp/bench"
  nodes=$(grep -oP 'Nodes searched\s*:\s*\K[0-9]+' "$tmp/bench" | tail -1)
  ins=$(awk -F, '$3~/^instructions/{print $1}' "$tmp/stat")
  br=$(awk -F, '$3~/^branches:/{print $1}' "$tmp/stat")
  bm=$(awk -F, '$3~/^branch-misses/{print $1}' "$tmp/stat")
  l1=$(awk -F, '$3~/^L1-dcache-load-misses/{print $1}' "$tmp/stat")
  python3 -c "
print(f'probe={\"$label\":22s} r=$r nodes={$nodes:>9,d} i_n={$ins/$nodes:9.3f} br_n={$br/$nodes:8.3f} bmiss_n={$bm/$nodes:7.4f} l1m_n={$l1/$nodes:8.3f} hash=$hash')" \
    | tee -a "$OUT"
done
