#!/usr/bin/env bash
# Held-out evaluation of one checkpoint: exact training graph + loss nodes over
# the shadow-population dump; prints "heldout  rows N  value_mse X  pol_ce Y".
# USES THE GPU (ModelEvaluator) — do not run concurrently with a trainer
# unless you accept the contention; RAM ~2G peak. pol=1 only for nets whose
# quant/raw contains the policy head (d47c family); pol=0 for the twin/value-only.
#   bash tools/nnue/heldout_eval.sh <ckpt.bin> <raw|quant> <pol 0|1> [dump.bin]
set -euo pipefail
cd "$(dirname "$0")/../.."
CKPT="${1:?checkpoint path}"
KIND="${2:?raw|quant}"
POL="${3:?0|1}"
DUMP="${4:-data/nnue/heldout/uho-band0-5.bin}"
[ -s "$CKPT" ] || { echo "missing ckpt: $CKPT"; exit 2; }
[ -s "$DUMP" ] || { echo "missing dump: $DUMP (run heldout_extract.sh)"; exit 2; }
ENVS=(
  D40_DATA=data/nnue/binpacks/nodes5000pv2_UHO.binpack D40_FORMAT=sfbinpack
  D40_SF_BUFFER_MB=128 SFBINPACK_SCORE_DIV=1.9 D40_THREADS=4
  D40_RANDOM_SKIP_PCT=75 D40_VERIFY_ROWS=64
  D40_EVAL_TARGET_SCALE=400 D40_WDL=0.3 D40_SCORE_MULTIPLIER=1.0
  D40_HELDOUT_EVAL="$DUMP" CUDA_PATH=/opt/cuda
)
if [ "$POL" = 1 ]; then ENVS+=(D40_POL_LAMBDA=0.005 D40_POL_LEGAL_MASK=1); fi
case "$KIND" in
  raw)   ENVS+=(D40_PROBE_RAW="$CKPT") ;;
  quant) ENVS+=(D40_PROBE_QUANT="$CKPT") ;;
  *) echo "kind must be raw|quant"; exit 2 ;;
esac
env "${ENVS[@]}" .archon-state/toolchains/bullet/target/release/examples/d40_features | grep -a "^heldout"
