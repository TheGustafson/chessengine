#!/usr/bin/env bash
# perf study 15 — screening probe.
#
#   bash probe15.sh <label>
#
# Inherited from probe14lm.sh. Fixed 5,001,562-node tree, instructions:u, wl50
# eval hash + node count printed so a tree/eval change cannot hide inside a
# speed claim. thin LTO + 4 codegen units: ~20x faster to build than the pinned
# profile and a fraction of the peak RSS.
#
# BUILD DISCIPLINE (study 14 §1.4, and the study-15 commission): every build
# runs inside its OWN memory-capped cgroup, so a runaway build is killed inside
# that cgroup instead of tripping the global OOM killer that has now taken out
# two training runs. The `free` floor is a courtesy; the cgroup is the guard.
#
# study 12 §0.3 stands: NO probe delta may be quoted as a result. Screening
# only; the PGO panel decides.
set -euo pipefail
LABEL=${1:?label}
OUT=docs/research/perf-artifacts-15/probes-fast.txt
free -m | awk 'NR==2 {if ($7 < 3200) {print "REFUSING: only " $7 " MB available (need 3200)"; exit 1}}'

systemd-run --user --scope -q -p MemoryMax=4G -p MemorySwapMax=0 -- \
  env RUSTFLAGS="-C target-cpu=native" CARGO_BUILD_JOBS=2 \
  CARGO_PROFILE_RELEASE_LTO=thin CARGO_PROFILE_RELEASE_CODEGEN_UNITS=4 \
  nice -n 10 cargo build --release --features sota,arch-d48x --target-dir target-fast15 -q
BIN=target-fast15/release/chessengine

tmp=$(mktemp)
nice -n 19 perf stat -x, -e instructions:u,branches:u -o "$tmp.perf" -- \
  taskset -c 6 "$BIN" bench 16 1 100000 tools/nnue/calib/wl50.fens nodes \
  engines/canonical/net.cenn > /dev/null 2> "$tmp.eng"
ins=$(awk -F, '$3=="instructions:u"{print $1}' "$tmp.perf")
brs=$(awk -F, '$3=="branches:u"{print $1}' "$tmp.perf")
nodes=$(grep -oP 'Nodes searched\s*:\s*\K[0-9]+' "$tmp.eng" | tail -1)
hash=$("$BIN" nnue-eval engines/canonical/net.cenn < tools/nnue/calib/wl50.fens | sha256sum | cut -c1-16)
rm -f "$tmp" "$tmp.perf" "$tmp.eng"

line=$(python3 -c "print(f'fast={\"$LABEL\":32s} instr={$ins:>14,d} nodes={$nodes:>10,d} per_node={$ins/$nodes:8.1f} brs_per_node={$brs/$nodes:7.1f} hash=$hash')")
echo "$line" | tee -a "$OUT"
