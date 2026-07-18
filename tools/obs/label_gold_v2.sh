#!/usr/bin/env bash
# Gold-v2 PARALLEL labeling (flywheel cycle-1 overnight gold run, 2026-07-18):
#   Faithful copy of label_gold_v1.sh retargeted to FEN pool #2
#   (obs-pool2-fens.txt, 100k fresh openings-derived positions, seed 20260800,
#   deduped vs obs-gold-v1-fens + big-v1). Same single 16k production tier +
#   10% dual-tier (16k+64k) drift-monitor head slice. Disjoint FEN slices per
#   worker (fixed-node = load-immune, T4). Output shards need no merging
#   (obs-tool / export-train take multiple shards, dedup by canonical key).
# Usage: bash tools/obs/label_gold_v2.sh <positions> [workers-per-teacher]
#   e.g. 100000 6  → 12 worker processes/teacher-pair + 2 monitor, ~5-6h at
#   full free box (2 teachers × 6 = 12 + 2 monitor ≈ 14 of 16 cores).
# NEVER run beside TC gates (contends for CPU); coexists fine with itself.
set -euo pipefail
cd "$(dirname "$0")/../.."
POS=${1:?positions}
W=${2:-6}
FENS=data/nnue/experiments/obs-pool2-fens.txt
R=data/nnue/experiments/chain-v2-results.log
say() { echo "[obsgoldv2 $(date +%H:%M:%S)] $*" | tee -a "$R"; }
mkdir -p data/obs/gold-v2 data/nnue/experiments/obs-gold-v2

cargo build -q --features datagen --bin root-analysis --bin obs-tool

# Slice the fen file: monitor slice = first 10%, then W equal production slices.
python3 - "$POS" "$W" "$FENS" <<'EOF'
import sys
pos, w, fenpath = int(sys.argv[1]), int(sys.argv[2]), sys.argv[3]
fens = [l for l in open(fenpath)
        if l.strip() and not l.startswith("#")][:pos]
mon = fens[: max(1, pos // 10)]
rest = fens[len(mon):]
open("data/nnue/experiments/obs-gold-v2/monitor.fens", "w").writelines(mon)
per = (len(rest) + w - 1) // w
for i in range(w):
    open(f"data/nnue/experiments/obs-gold-v2/slice{i}.fens", "w").writelines(
        rest[i * per : (i + 1) * per])
print(f"monitor {len(mon)} + {w} slices of ~{per}")
EOF

say "gold-v2: $POS positions, $W workers/teacher, single 16k tier + 10% monitor (pool2)"
PIDS=()
for i in $(seq 0 $((W - 1))); do
  target/debug/root-analysis \
    --engine engines/bin/reckless-d445b8b --arm-mode child \
    --fens data/nnue/experiments/obs-gold-v2/slice$i.fens \
    --out data/obs/gold-v2/reckless-w$i.ceobs \
    --nodes 16000 --seed 2026080$i --created-unix 1784800000 \
    > data/nnue/experiments/obs-gold-v2/reckless-w$i.log 2>&1 &
  PIDS+=($!)
  target/debug/root-analysis \
    --engine engines/stockfish-18-src/src/stockfish --arm-mode searchmoves \
    --fens data/nnue/experiments/obs-gold-v2/slice$i.fens \
    --out data/obs/gold-v2/sf18-w$i.ceobs \
    --nodes 16000 --seed 2026080$i --created-unix 1784800000 \
    > data/nnue/experiments/obs-gold-v2/sf18-w$i.log 2>&1 &
  PIDS+=($!)
done
# Drift-monitor slice: dual tier, one worker per teacher.
target/debug/root-analysis \
  --engine engines/bin/reckless-d445b8b --arm-mode child \
  --fens data/nnue/experiments/obs-gold-v2/monitor.fens \
  --out data/obs/gold-v2/reckless-monitor.ceobs \
  --nodes 16000,64000 --seed 20260861 --created-unix 1784800000 \
  > data/nnue/experiments/obs-gold-v2/reckless-monitor.log 2>&1 &
PIDS+=($!)
target/debug/root-analysis \
  --engine engines/stockfish-18-src/src/stockfish --arm-mode searchmoves \
  --fens data/nnue/experiments/obs-gold-v2/monitor.fens \
  --out data/obs/gold-v2/sf18-monitor.ceobs \
  --nodes 16000,64000 --seed 20260861 --created-unix 1784800000 \
  > data/nnue/experiments/obs-gold-v2/sf18-monitor.log 2>&1 &
PIDS+=($!)

FAIL=0
for pid in "${PIDS[@]}"; do
  wait "$pid" || FAIL=$((FAIL + 1))
done
[ "$FAIL" -gt 0 ] && { say "gold-v2: $FAIL worker(s) FAILED"; exit 1; }

say "workers done; audits"
target/debug/obs-tool verify data/obs/gold-v2/*.ceobs | tail -3 | tee -a "$R"
target/debug/obs-tool stability data/obs/gold-v2/reckless-monitor.ceobs | tee -a "$R"
say "OBS GOLD V2 COMPLETE"
