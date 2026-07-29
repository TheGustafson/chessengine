#!/usr/bin/env bash
# perf study 14 — LOW-MEMORY screening probe.
#
# Identical instrument to probe14fast.sh (same flags, same tree, same 5,001,562
# node bench, same eval-hash print). Two differences, both about the BOX, not
# the measurement:
#
#   1. `-j 1` instead of `-j 2`  — roughly halves the compiler's peak RSS.
#   2. the compile runs inside a `systemd-run --user --scope -p MemoryMax=2G`
#      cgroup — so a runaway build is killed by the kernel *inside its own
#      cgroup* instead of triggering the GLOBAL OOM killer that took out a
#      training run on 2026-07-27. This is the guard, not the `free` check:
#      the free check can race, a MemoryMax cannot.
#
# With the hard cap in place the availability floor is 2800 MB (>= 3x the
# measured peak of a -j1 thin-LTO build of this crate). The D48Y trainer held
# 5.3 GB for this whole session, which is why this variant exists at all.
#
# Deltas are comparable to probe14fast.sh's ONLY if the codegen is identical;
# `-j 1` vs `-j 2` does NOT change codegen (codegen-units is pinned at 4 by
# the env var, and cargo's job count only schedules crates). The rebuild null
# in probes-fast.txt spans both spellings and is the evidence for that.
set -euo pipefail
LABEL=${1:?label}
OUT=docs/research/perf-artifacts-14/probes-fast.txt
free -m | awk 'NR==2 {if ($7 < 2800) {print "REFUSING: only " $7 " MB available (need 2800)"; exit 1}}'

systemd-run --user --scope -q -p MemoryMax=2G -p MemorySwapMax=0 -- \
  env RUSTFLAGS="-C target-cpu=native" CARGO_BUILD_JOBS=1 \
  CARGO_PROFILE_RELEASE_LTO=thin CARGO_PROFILE_RELEASE_CODEGEN_UNITS=4 \
  nice -n 10 cargo build --release --features sota,arch-d48x --target-dir target-fast14 -q
BIN=target-fast14/release/chessengine

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
