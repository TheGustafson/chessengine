#!/usr/bin/env bash
# First real Phase-1 labeling run (obs sample v1): the SAME 1000 UHO
# positions through BOTH candidate teachers, then the M4 audits.
#   - Reckless d445b8b, child-negate arms (it ignores searchmoves)
#   - Stockfish 18, true searchmoves arms
# Budget tiers 4k/16k/64k nodes; deterministic (seeded sample, pinned
# created-unix). Teachers run SEQUENTIALLY (one engine process at a time —
# the 720sb trainer owns the GPU and ~2 cores until ~03:50).
set -euo pipefail
cd "$(dirname "$0")/../.."
FENS=data/nnue/experiments/obs-sample-v1-fens.txt
BIN="cargo run -q --features datagen"
R=data/nnue/experiments/chain-v2-results.log
say() { echo "[obslabel $(date +%H:%M:%S)] $*" | tee -a "$R"; }

say "labeling obs-sample-v1: Reckless child-mode arms (1000 pos x 3 tiers)"
$BIN --bin root-analysis -- \
  --engine engines/bin/reckless-d445b8b --arm-mode child \
  --fens "$FENS" --out data/obs/sample-v1-reckless.ceobs \
  --nodes 4000,16000,64000 --seed 20260717 --created-unix 1784600000 \
  || { say "Reckless labeling FAILED"; exit 1; }
say "Reckless shard done"

say "labeling obs-sample-v1: SF18 searchmoves arms"
$BIN --bin root-analysis -- \
  --engine engines/stockfish-18-src/src/stockfish --arm-mode searchmoves \
  --fens "$FENS" --out data/obs/sample-v1-sf18.ceobs \
  --nodes 4000,16000,64000 --seed 20260717 --created-unix 1784600000 \
  || { say "SF18 labeling FAILED"; exit 1; }
say "SF18 shard done"

say "M4 audits"
$BIN --bin obs-tool -- verify data/obs/sample-v1-reckless.ceobs data/obs/sample-v1-sf18.ceobs | tee -a "$R"
$BIN --bin obs-tool -- stability data/obs/sample-v1-reckless.ceobs | tee -a "$R"
$BIN --bin obs-tool -- stability data/obs/sample-v1-sf18.ceobs | tee -a "$R"
$BIN --bin obs-tool -- disagree data/obs/sample-v1-reckless.ceobs data/obs/sample-v1-sf18.ceobs | tee -a "$R"
say "OBS SAMPLE V1 COMPLETE"
