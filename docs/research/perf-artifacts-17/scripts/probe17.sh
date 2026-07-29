#!/usr/bin/env bash
# perf study 17 — screening probe.
#
#   bash probe17.sh <label>
#
# Inherited from probe16.sh; only the target dir and the output file change.
# Fixed 5,001,562-node tree, instructions:u + branches:u, wl50 eval hash and node
# count printed on every line so a tree or eval change cannot hide inside a speed
# claim. thin LTO + 4 codegen units.
#
# BUILD DISCIPLINE (study 17 commission): NO lto="fat" while the GPU trainer is
# live. Every build runs inside its OWN memory-capped cgroup AND through
# tools/oomlast.sh, which raises oom_score_adj to 1000 before exec so children
# inherit it and the kernel picks THIS build rather than the trainer.
# `systemd-run -p OOMScoreAdjust=` does not work for user scopes.
#
# study 12 §0.3 stands: NO probe delta may be quoted as a result. Screening
# only; the PGO panel decides.
set -euo pipefail
LABEL=${1:?label}
OUT=docs/research/perf-artifacts-17/probes-fast.txt
free -m | awk 'NR==2 {if ($7 < 2600) {print "REFUSING: only " $7 " MB available (need 2600)"; exit 1}}'

systemd-run --user --scope -q -p MemoryMax=3G -p MemorySwapMax=0 -- \
  tools/oomlast.sh env RUSTFLAGS="-C target-cpu=native" CARGO_BUILD_JOBS=2 \
  CARGO_PROFILE_RELEASE_LTO=thin CARGO_PROFILE_RELEASE_CODEGEN_UNITS=4 \
  nice -n 10 cargo build --release --features sota,arch-d48x --target-dir target-fast17 -q
BIN=target-fast17/release/chessengine

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
