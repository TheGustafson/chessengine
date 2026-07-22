#!/usr/bin/env bash
# d47c-stage2: warm retrain from the finished d47c-1280-240sb per roadmap §5b.
# PREPARED 2026-07-22 00:30 (pre-fork). DO NOT LAUNCH until:
#   1. the main run finishes (sb240) and gate_d47.sh has run,
#   2. the twin gap + end-slope reads happen (they calibrate the two OPEN knobs),
#   3. the GPU is free (twin done or consciously pre-empted).
# Launch: copy to /tmp and run from there (never run the repo copy of a live script).
#
# OPEN KNOB 1 — D40_SUPERBATCHES/D40_LR_END_SB: 120 if end-slope at sb240 is
#   flattening, 240 if value is still falling steeply.
# OPEN KNOB 2 — D40_DATA: same UHO stream (default) vs farseer fold-in per the
#   incumbent stage-2 recipe (B4: coverage beats epochs). Verify D46 gates on
#   any new stream before pointing at it.
#
# PINNED (decided at the sb120 fork discussion, do not re-litigate at launch):
#   warm from sb240-raw; LR restarts at HALF stage-1 init (0.0005) with a fresh
#   cosine — the "pump the LR, flatten the decay" move; POL_LAMBDA = lambda/4
#   = 0.00125, NOT zero (frozen-policy trunk drift rots accuracy silently);
#   everything else identical to stage-1 (seed/batch/BPS/filters/scale).

set -uo pipefail
cd /home/nick/projects/chessengine
D40_DATA=data/nnue/binpacks/nodes5000pv2_UHO.binpack D40_FORMAT=sfbinpack D40_SF_BUFFER_MB=128 \
D40_HALFKA_ONLY=0 D40_FACTORIZE=0 \
D40_ID=d47c-stage2 D40_OUT=data/nnue/experiments/d47c-stage2/ckpt D40_SEED=198273612 \
D40_SUPERBATCHES=240 D40_LR_END_SB=240 D40_SAVE_RATE=1 \
D40_INIT_RAW=data/nnue/experiments/d47c-1280-240sb/ckpt/d47c-1280-240sb-sb240-raw.bin \
D40_BATCH=8192 D40_BPS=3040 D40_LR=0.0005 D40_WDL=0.3 \
SFBINPACK_SCORE_DIV=1.9 D40_RANDOM_SKIP_PCT=75 \
D40_EVAL_TARGET_SCALE=400 D40_SCORE_MULTIPLIER=1.0 \
D40_POL_LAMBDA=0.00125 D40_POL_LEGAL_MASK=1 D40_THREADS=6 \
D40_VERIFY_ROWS=4096 CUDA_PATH=/opt/cuda \
.archon-state/toolchains/bullet/target/release/examples/d40_features
echo "[d47c-stage2] TRAIN EXIT $?"
