#!/usr/bin/env bash
# Big Phase-1 labeling run (obs big v1): 10,000 fresh UHO positions (seed
# 20260718, sample-v1's 1000 excluded — sample-v1 stays a clean held-out
# shard) through BOTH teachers, then the M4 audits.
#   - Reckless d445b8b, child-negate arms (it ignores searchmoves)
#   - Stockfish 18, true searchmoves arms
# Teachers run in PARALLEL: fixed-node searches are load-immune (T4
# concurrency invariance), so this needs cores, not a quiet box. The guard
# below still waits for every TC job (night-chain gate, S1 SPRT) because
# THOSE need wall-clock fairness, and for the trainer (GPU box RAM).
# ETA ~10.5h from start (Reckless is the long pole at ~63 min/1000 pos).
set -euo pipefail
cd "$(dirname "$0")/../.."
FENS=data/nnue/experiments/obs-big-v1-fens.txt
BIN="cargo run -q --features datagen"
R=data/nnue/experiments/chain-v2-results.log
say() { echo "[obsbig $(date +%H:%M:%S)] $*" | tee -a "$R"; }

say "armed: waiting for the box to go quiet (night chain + S1 gate + trainer)"
while true; do
  chain_done=$(grep -c "D40 NIGHT CHAIN COMPLETE" "$R" || true)
  # `|| true`: pgrep exits 1 on zero matches, which pipefail would otherwise
  # propagate through wc and set -e would kill the guard (bit us at 03:09).
  s1_alive=$(pgrep -f "tools/nnue/s1_gate_v1.sh" | wc -l || true)
  train_alive=$(pgrep -x "d40_features" | wc -l || true)
  match_alive=$(pgrep -x "fastchess" | wc -l || true)
  if [ "$chain_done" -ge 1 ] && [ "$s1_alive" -eq 0 ] \
     && [ "$train_alive" -eq 0 ] && [ "$match_alive" -eq 0 ]; then
    break
  fi
  sleep 120
done
say "box is quiet; building datagen bins"

# Build once up front so the two parallel runs don't race a cargo lock.
cargo build -q --features datagen --bin root-analysis --bin obs-tool

say "labeling obs-big-v1: BOTH teachers in parallel (10000 pos x 3 tiers)"
target/debug/root-analysis \
  --engine engines/bin/reckless-d445b8b --arm-mode child \
  --fens "$FENS" --out data/obs/big-v1-reckless.ceobs \
  --nodes 4000,16000,64000 --seed 20260718 --created-unix 1784700000 \
  > data/nnue/experiments/obs-big-v1-reckless.log 2>&1 &
RECK_PID=$!
target/debug/root-analysis \
  --engine engines/stockfish-18-src/src/stockfish --arm-mode searchmoves \
  --fens "$FENS" --out data/obs/big-v1-sf18.ceobs \
  --nodes 4000,16000,64000 --seed 20260718 --created-unix 1784700000 \
  > data/nnue/experiments/obs-big-v1-sf18.log 2>&1 &
SF_PID=$!

wait "$SF_PID" || { say "SF18 labeling FAILED"; kill "$RECK_PID" 2>/dev/null || true; exit 1; }
say "SF18 shard done"
wait "$RECK_PID" || { say "Reckless labeling FAILED"; exit 1; }
say "Reckless shard done"

say "M4 audits"
target/debug/obs-tool verify data/obs/big-v1-reckless.ceobs data/obs/big-v1-sf18.ceobs | tee -a "$R"
target/debug/obs-tool stability data/obs/big-v1-reckless.ceobs | tee -a "$R"
target/debug/obs-tool stability data/obs/big-v1-sf18.ceobs | tee -a "$R"
target/debug/obs-tool disagree data/obs/big-v1-reckless.ceobs data/obs/big-v1-sf18.ceobs | tee -a "$R"
say "OBS BIG V1 COMPLETE"
