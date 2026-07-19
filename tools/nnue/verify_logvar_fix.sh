#!/usr/bin/env bash
# EXP-UNC-1: verify the logvar-save fix, box-friendly. Hypothesis: exposing
# `logvar` as a saved output makes l3_var land in the raw checkpoint (the raw
# save persists only output-reachable weights). Sequence, each stage gated to
# avoid disturbing higher-priority jobs:
#   1. wait for the Integral match DONE (frees CPU / equal-time constraint),
#   2. BUILD the fixed trainer (CPU, capped) — verifies the edit compiles,
#   3. wait for the twin train to finish (frees the GPU) — then a 2-SB smoke,
#   4. check the raw ckpt for l3_varw/l3_varb.
set -uo pipefail
cd "$(dirname "$0")/../.."
LOG=data/nnue/experiments/d40-uho-unc-v1/logvar-verify.log
SM=/tmp/claude-0/-root-projects-learnrust-chessengine/9162696c-0351-4e70-854a-ccb48a50c724/scratchpad/logvar_verify
BULLET=.archon-state/toolchains/bullet
D=/root/projects/learnrust/chessengine/data/corpora/official-stockfish-master-binpacks/main/raw/nodes5000pv2_UHO.binpack
TWIN_PID="${1:-430823}"
say(){ echo "[logvar-verify $(date -u +%H:%M:%S)] $*" | tee -a "$LOG"; }
mkdir -p "$SM"

say "waiting for the Integral match to free the box"
until [ -f data/nnue/experiments/integral-match-v1/DONE ]; do sleep 30; done
sleep 5
say "match done; staging + building fixed trainer (CARGO_BUILD_JOBS=2)"
cp tools/nnue/bullet/d40_features.rs "$BULLET/examples/d40_features.rs"
if ! ( cd "$BULLET" && CUDA_PATH=/usr CARGO_BUILD_JOBS=2 cargo build --release --features cuda --example d40_features ) >> "$LOG" 2>&1; then
    say "BUILD FAILED — the NLL+logvar edit does not compile; see log"; exit 1
fi
say "build OK (edit compiles)"

say "waiting for the twin train (pid $TWIN_PID) to free the GPU before the smoke"
while kill -0 "$TWIN_PID" 2>/dev/null; do sleep 60; done
sleep 5
say "GPU free; 2-SB smoke with D40_HETERO_NLL=1 (Gate-1 match is fixed-node/load-immune)"
( cd "$BULLET" && CUDA_PATH=/usr D40_HETERO_NLL=1 \
    D40_DATA="$D" D40_FORMAT=sfbinpack \
    D40_SUPERBATCHES=2 D40_BPS=40 D40_BATCH=16384 D40_LR=0.001 D40_LR_END_SB=2 \
    D40_WDL=0.3 SFBINPACK_SCORE_DIV=1.9 D40_RANDOM_SKIP_PCT=75 \
    D40_SF_BUFFER_MB=128 D40_VERIFY_ROWS=64 \
    D40_EVAL_TARGET_SCALE=400 D40_SCORE_MULTIPLIER=1.0 D40_HALFKA_ONLY=0 \
    D40_OUT="$SM/ckpt" D40_ID=lv \
    cargo run --release --features cuda --example d40_features ) >> "$LOG" 2>&1 || say "smoke exited nonzero (check log)"

say "checking raw ckpt for the saved variance head"
.venv/bin/python3 -c "
import sys; sys.path.insert(0,'tools/nnue')
from extract_unc_head import parse_raw
t = sorted(parse_raw('$SM/ckpt/lv-final-raw.bin').keys())
ok = ('l3_varw' in t) and ('l3_varb' in t)
print('[logvar-verify] RESULT:', 'FIX WORKS — l3_varw/l3_varb ARE in the raw ckpt' if ok
      else 'FIX FAILED — l3_var still missing; need another save mechanism')
print('[logvar-verify] raw tensors:', t)
" 2>&1 | tee -a "$LOG"
touch data/nnue/experiments/d40-uho-unc-v1/LOGVAR-VERIFY-DONE
say "done"
