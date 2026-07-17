#!/usr/bin/env bash
# Gold-v1 PARALLEL labeling (datagen scalability levers 1+2, 2026-07-17):
#   - N workers per teacher on disjoint FEN slices (fixed-node = load-immune,
#     T4, so parallelism is free quality-wise; per-worker output shards need
#     no merging — obs-tool and export-train take multiple shards and dedup
#     by canonical key).
#   - SINGLE production tier at 16k nodes (the 3-tier design was for the M4
#     audits, which proved 10cp median cross-tier value drift; we train on
#     values/margins, so 16k labels sit at the label-noise floor) PLUS a 10%
#     dual-tier (16k+64k) drift-monitor slice from the head of the file.
# Usage: bash tools/obs/label_gold_v1.sh <positions> [workers-per-teacher]
#   e.g. 20000 4  → 8 worker processes total, ~2.5h at full box.
# WAIT for a quiet-box-optional window: this contends for CPU with TC games
# (never run beside gates) but coexists fine with itself.
set -euo pipefail
cd "$(dirname "$0")/../.."
POS=${1:?positions}
W=${2:-4}
FENS=data/nnue/experiments/obs-gold-v1-fens.txt
R=data/nnue/experiments/chain-v2-results.log
say() { echo "[obsgold $(date +%H:%M:%S)] $*" | tee -a "$R"; }
mkdir -p data/obs/gold-v1 data/nnue/experiments/obs-gold-v1

cargo build -q --features datagen --bin root-analysis --bin obs-tool

# Slice the fen file: monitor slice = first 10%, then W equal production slices.
python3 - "$POS" "$W" <<'EOF'
import sys
pos, w = int(sys.argv[1]), int(sys.argv[2])
fens = [l for l in open("data/nnue/experiments/obs-gold-v1-fens.txt")
        if l.strip() and not l.startswith("#")][:pos]
mon = fens[: max(1, pos // 10)]
rest = fens[len(mon):]
open("data/nnue/experiments/obs-gold-v1/monitor.fens", "w").writelines(mon)
per = (len(rest) + w - 1) // w
for i in range(w):
    open(f"data/nnue/experiments/obs-gold-v1/slice{i}.fens", "w").writelines(
        rest[i * per : (i + 1) * per])
print(f"monitor {len(mon)} + {w} slices of ~{per}")
EOF

say "gold-v1: $POS positions, $W workers/teacher, single 16k tier + 10% monitor"
PIDS=()
for i in $(seq 0 $((W - 1))); do
  target/debug/root-analysis \
    --engine engines/bin/reckless-d445b8b --arm-mode child \
    --fens data/nnue/experiments/obs-gold-v1/slice$i.fens \
    --out data/obs/gold-v1/reckless-w$i.ceobs \
    --nodes 16000 --seed 2026075$i --created-unix 1784800000 \
    > data/nnue/experiments/obs-gold-v1/reckless-w$i.log 2>&1 &
  PIDS+=($!)
  target/debug/root-analysis \
    --engine engines/stockfish-18-src/src/stockfish --arm-mode searchmoves \
    --fens data/nnue/experiments/obs-gold-v1/slice$i.fens \
    --out data/obs/gold-v1/sf18-w$i.ceobs \
    --nodes 16000 --seed 2026075$i --created-unix 1784800000 \
    > data/nnue/experiments/obs-gold-v1/sf18-w$i.log 2>&1 &
  PIDS+=($!)
done
# Drift-monitor slice: dual tier, one worker per teacher.
target/debug/root-analysis \
  --engine engines/bin/reckless-d445b8b --arm-mode child \
  --fens data/nnue/experiments/obs-gold-v1/monitor.fens \
  --out data/obs/gold-v1/reckless-monitor.ceobs \
  --nodes 16000,64000 --seed 20260760 --created-unix 1784800000 \
  > data/nnue/experiments/obs-gold-v1/reckless-monitor.log 2>&1 &
PIDS+=($!)
target/debug/root-analysis \
  --engine engines/stockfish-18-src/src/stockfish --arm-mode searchmoves \
  --fens data/nnue/experiments/obs-gold-v1/monitor.fens \
  --out data/obs/gold-v1/sf18-monitor.ceobs \
  --nodes 16000,64000 --seed 20260760 --created-unix 1784800000 \
  > data/nnue/experiments/obs-gold-v1/sf18-monitor.log 2>&1 &
PIDS+=($!)

FAIL=0
for pid in "${PIDS[@]}"; do
  wait "$pid" || FAIL=$((FAIL + 1))
done
[ "$FAIL" -gt 0 ] && { say "gold-v1: $FAIL worker(s) FAILED"; exit 1; }

say "workers done; audits"
target/debug/obs-tool verify data/obs/gold-v1/*.ceobs | tail -3 | tee -a "$R"
target/debug/obs-tool stability data/obs/gold-v1/reckless-monitor.ceobs | tee -a "$R"
say "OBS GOLD V1 COMPLETE"
