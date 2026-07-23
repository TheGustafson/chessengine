#!/usr/bin/env bash
# DECISIVE DIAGNOSTIC (2026-07-20): reproduce the WORKING incumbent
# d40-uho-v1-240sb from its exact recipe on sfbinpack UHO. If this reproduces a
# competitive net (gate ~0 vs the shipped d40-uho-v1-240sb), the trainer
# environment is sound and the canon −582 failure is in the canon DATA path
# (bf-convert/bulletformat, disagreement subset, SF18 cp, or WDL=0). If this
# ALSO fails, the trainer/environment itself is broken.
#   recipe = d40_uho_v1.sh: sfbinpack, SCORE_DIV=1.9, skip75, WDL=0.3, 240sb, seed 198273612
set -uo pipefail
cd "$(dirname "$0")/../.."
DATA=data/nnue/binpacks/nodes5000pv2_UHO.binpack
[ -s "$DATA" ] || { echo "UHO binpack not present: $DATA"; exit 1; }
TRAINER=.archon-state/toolchains/bullet/target/release/examples/d40_features
ID="${NET_ID:-repro-uho-240sb}"
CKPT="data/nnue/experiments/$ID/ckpt"; mkdir -p "$CKPT"
SB="${SB:-240}"
echo "[repro] id=$ID sfbinpack UHO, SCORE_DIV=1.9, skip75, WDL=0.3, ${SB}sb"

D40_DATA="$DATA" \
D40_FORMAT=sfbinpack \
SFBINPACK_SCORE_DIV=1.9 \
D40_RANDOM_SKIP_PCT=75 \
D40_ID="$ID" \
D40_OUT="$CKPT" \
D40_SEED=198273612 \
D40_SUPERBATCHES="$SB" D40_LR_END_SB="$SB" D40_SAVE_RATE="$SB" \
D40_BATCH=16384 D40_BPS=1520 \
D40_LR=0.001 D40_WDL=0.3 \
D40_EVAL_TARGET_SCALE=400 D40_SCORE_MULTIPLIER=1.0 \
D40_HALFKA_ONLY=0 D40_SF_BUFFER_MB=128 D40_VERIFY_ROWS=4096 \
CUDA_PATH=/opt/cuda LD_LIBRARY_PATH=/opt/cuda/lib64:${LD_LIBRARY_PATH:-} \
"$TRAINER"
rc=$?; echo "[repro] trainer exit=$rc"; [ $rc -ne 0 ] && exit $rc
QUANT="$CKPT/${ID}-sb${SB}-quant.bin"; NET="data/nnue/nets/${ID}.cenn"
uv run --no-project --with numpy python tools/nnue/netpack.py "$QUANT" "$NET" d40 2>&1 | tail -2
ls -la "$NET"; echo "[repro] DONE — now gate vs data/nnue/nets/d40-uho-v1-240sb.cenn (expect ~0)"
