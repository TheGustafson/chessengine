#!/usr/bin/env bash
# perf study 19 — screening probe (PRIMARY AXIS: instructions:u).
#
#   bash probe19.sh <label>
#
# Inherited from probe18.sh (study 17 -> 18 -> 19). The frame is unchanged:
# fixed 5,001,562-node tree, thin LTO + 4 codegen units, `-C target-cpu=native`,
# non-PGO, wl50 eval hash and node count printed on every line so a tree or eval
# change cannot hide inside a speed claim.
#
# TWO CHANGES FROM probe18.sh, both forced by study 19's own null panel:
#
#  1. i/n is printed to FOUR decimals. Study 18 quoted its null at ±0.00017 i/n;
#     one decimal cannot state that.
#  2. The built binary is STASHED at bin/<label> so `pair19.sh` can come back and
#     measure the contended axes (cycles, cache-misses) INTERLEAVED against
#     another arm. It cannot be done here: the study-19 null measured two builds
#     of BIT-IDENTICAL SOURCE at 4425.7 and 4810.5 cyc/n — a 384.8 (8.7%) gap
#     from session drift alone, larger than any arm this study can deliver. A
#     sequential cycles number IS NOT A MEASUREMENT on this box; only a paired
#     interleaved one is. `instructions:u` is unaffected (identical to the digit
#     across the same two builds), which is why it stays primary.
#
# BUILD DISCIPLINE (study 17 commission, still in force): NO lto="fat" while the
# GPU trainer is live. Every build runs inside its OWN memory-capped cgroup AND
# through tools/oomlast.sh, which raises oom_score_adj to 1000 before exec so
# children inherit it and the kernel picks THIS build rather than the trainer.
#
# study 12 §0.3 stands: NO probe delta may be quoted as a result. Screening
# only; the PGO panel decides.
set -euo pipefail
LABEL=${1:?label}
OUT=docs/research/perf-artifacts-19/probes-fast.txt
BINDIR=docs/research/perf-artifacts-19/bin
mkdir -p "$BINDIR"
free -m | awk 'NR==2 {if ($7 < 2600) {print "REFUSING: only " $7 " MB available (need 2600)"; exit 1}}'

systemd-run --user --scope -q -p MemoryMax=3G -p MemorySwapMax=0 -- \
  tools/oomlast.sh env RUSTFLAGS="-C target-cpu=native" CARGO_BUILD_JOBS=2 \
  CARGO_PROFILE_RELEASE_LTO=thin CARGO_PROFILE_RELEASE_CODEGEN_UNITS=4 \
  nice -n 10 cargo build --release --features sota,arch-d48x --target-dir target-fast19 -q
BIN=target-fast19/release/chessengine

tmp=$(mktemp -d)
nice -n 19 perf stat -x, -e instructions:u,branches:u -o "$tmp/perf" -- \
  taskset -c 6 "$BIN" bench 16 1 100000 tools/nnue/calib/wl50.fens nodes \
  engines/canonical/net.cenn > /dev/null 2> "$tmp/eng"
ins=$(awk -F, '$3=="instructions:u"{print $1}' "$tmp/perf")
brs=$(awk -F, '$3=="branches:u"{print $1}' "$tmp/perf")
nodes=$(grep -oP 'Nodes searched\s*:\s*\K[0-9]+' "$tmp/eng" | tail -1)
hash=$("$BIN" nnue-eval engines/canonical/net.cenn < tools/nnue/calib/wl50.fens | sha256sum | cut -c1-16)
rm -rf "$tmp"
cp "$BIN" "$BINDIR/$LABEL"

line=$(python3 - "$LABEL" "$ins" "$brs" "$nodes" "$hash" <<'PY'
import sys
lab, ins, brs, nodes, h = sys.argv[1:6]
ins, brs, nodes = int(ins), int(brs), int(nodes)
print(f"fast={lab:34s} instr={ins:>15,d} i/n={ins/nodes:11.4f} br/n={brs/nodes:9.3f} "
      f"nodes={nodes:>10,d} hash={h}")
PY
)
echo "$line" | tee -a "$OUT"
