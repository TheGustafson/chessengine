#!/usr/bin/env bash
set -uo pipefail
cd /home/nick/projects/chessengine
TRAIN_PID="${1:?train pid}"
while kill -0 "$TRAIN_PID" 2>/dev/null; do sleep 120; done
echo "[twin] launching d47c-twin-nopol"
mkdir -p data/nnue/experiments/d47c-twin-nopol-240sb/ckpt
D40_DATA=data/nnue/binpacks/nodes5000pv2_UHO.binpack D40_FORMAT=sfbinpack D40_SF_BUFFER_MB=128 \
D40_HALFKA_ONLY=0 D40_FACTORIZE=0 \
D40_ID=d47c-twin-nopol-240sb D40_OUT=data/nnue/experiments/d47c-twin-nopol-240sb/ckpt D40_SEED=198273612 \
D40_SUPERBATCHES=240 D40_LR_END_SB=240 D40_SAVE_RATE=20 \
D40_BATCH=8192 D40_BPS=3040 D40_LR=0.001 D40_WDL=0.3 \
SFBINPACK_SCORE_DIV=1.9 D40_RANDOM_SKIP_PCT=75 \
D40_EVAL_TARGET_SCALE=400 D40_SCORE_MULTIPLIER=1.0 \
D40_THREADS=6 D40_VERIFY_ROWS=4096 CUDA_PATH=/opt/cuda \
.archon-state/toolchains/bullet/target/release/examples/d40_features
echo "[twin] TRAIN EXIT $?"
