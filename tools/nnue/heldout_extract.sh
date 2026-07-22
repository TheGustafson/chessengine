#!/usr/bin/env bash
# One-time held-out set extraction (design: policy roadmap §5b, 2026-07-22).
# Streams the training binpack through the TRAINER's own loader + quiet filter
# with the hash BAND [0,5) replacing the keep-test (hash%100 >= 75) — the
# shadow population no run in this family has ever trained on.
# CPU-only (the dump path exits before device creation) but RAM-guard it if a
# trainer is running (loader buffer + chunks; keep >= 1.5G available).
#   bash tools/nnue/heldout_extract.sh [rows] [out.bin]
set -euo pipefail
cd "$(dirname "$0")/../.."
ROWS="${1:-1000000}"
OUT="${2:-data/nnue/heldout/uho-band0-5.bin}"
mkdir -p "$(dirname "$OUT")"
D40_DATA=data/nnue/binpacks/nodes5000pv2_UHO.binpack D40_FORMAT=sfbinpack \
D40_SF_BUFFER_MB=128 SFBINPACK_SCORE_DIV=1.9 D40_THREADS=4 \
D40_HELDOUT_BAND=0:5 D40_HELDOUT_DUMP="$OUT" D40_HELDOUT_ROWS="$ROWS" \
D40_VERIFY_ROWS=64 CUDA_PATH=/opt/cuda \
.archon-state/toolchains/bullet/target/release/examples/d40_features
ls -la "$OUT"
