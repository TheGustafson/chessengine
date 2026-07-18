#!/usr/bin/env bash
# Overnight capture exact-node SCALE-UP (2026-07-18): the in-search capture is
# ~99.6% aspiration bounds, so the clean (Bound::Exact) subset is ~0.36% of raw
# and the raw .ceobs cannot be kept at scale (~9.4 GB / 600 games). This loop
# keeps disk bounded: per batch -> harvest self-play capture -> export the
# exact-node corpus (~73 MB) -> DELETE the raw shards -> next batch. Train on
# D41_DATA = comma-list of all batch dirs. Per-batch dedup only (diverse UHO
# openings make cross-batch dups rare). NEVER run beside a TC gate.
# Usage: bash tools/obs/capture_scale_batches.sh [batches=12] [games=600]
set -euo pipefail
cd "$(dirname "$0")/../.."
BATCHES=${1:-12}
GAMES=${2:-600}
NET=data/nnue/nets/stage2-v1-s3-2048-farseer-120sb.cenn
ENG=engines/bin/probe-l2048-datagen
TOOL=./target/release/obs-tool
OUTROOT=data/obs/train/capture-batches
mkdir -p "$OUTROOT" logs/overnight
TOTAL=0
for b in $(seq 1 "$BATCHES"); do
  seed=$((20260900 + b))
  raw=data/obs/harvest/batch$b
  mkdir -p "$raw" data/nnue/experiments/capture-batch$b
  echo "[batch $b/$BATCHES] harvest $GAMES games seed $seed  $(date +%H:%M:%S)"
  uv run python tools/nnue/eval_match.py \
    --candidate "$NET" --baseline "$NET" \
    --candidate-engine "$ENG" --baseline-engine "$ENG" \
    --nodes 50000 --games "$GAMES" --concurrency 14 --seed "$seed" \
    --openings data/nnue/openings/UHO_Lichess_4852_v1.epd \
    --each-option ObsCapturePath=$raw/run.ceobs \
    --output data/nnue/experiments/capture-batch$b > logs/overnight/capture-batch$b.log 2>&1
  echo "[batch $b/$BATCHES] export  $(date +%H:%M:%S)"
  $TOOL export-train "$OUTROOT/batch$b" "$raw"/*.ceobs --all-tiers > logs/overnight/export-batch$b.log 2>&1
  rows=$(python3 -c "import json;print(json.load(open('$OUTROOT/batch$b/manifest.json'))['stats']['arms_out'])" 2>/dev/null || echo 0)
  TOTAL=$((TOTAL + rows))
  rm -rf "$raw"
  free=$(df -h . | tail -1 | awk '{print $4" ("$5")"}')
  echo "[batch $b/$BATCHES] exact arms $rows | cumulative $TOTAL | disk free $free  $(date +%H:%M:%S)"
done
echo "ALL $BATCHES BATCHES DONE — cumulative exact arms: $TOTAL  $(date +%H:%M:%S)"
