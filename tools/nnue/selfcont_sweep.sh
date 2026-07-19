#!/usr/bin/env bash
# PROOF sweep (user demand 2026-07-19): continue the CONVERGED incumbent on its
# OWN UHO training data (exact recipe: div 1.9, skip 75%, WDL 0.3) across a range
# of LRs. If continuing on own data at a gentle LR degrades -> the finetune SETUP
# is broken (a bug), not the data. If it holds ~0 -> setup is competent and this
# maps the safe-LR envelope for real finetunes. Airtight: no scale/overfit/drift
# confound (same data, same scale, full corpus).
set -uo pipefail
cd "$(dirname "$0")/../.."
REPO=$(pwd)
INC=$REPO/data/nnue/experiments/d40-long-v1/ckpt/d40-uho-720sb-final-raw.bin
UHO=$REPO/data/corpora/official-stockfish-master-binpacks/main/raw/nodes5000pv2_UHO.binpack
mkdir -p data/nnue/fleet-relabel/selfcont-ckpt
say(){ echo "[selfcont $(date -u +%H:%M:%S)] $*"; }
say "SELF-CONTINUATION SWEEP: incumbent continued on its OWN UHO data"
for LR in 0.00001 0.00003 0.0001 0.0003; do
  ID=selfcont_lr${LR}
  say "training $ID (LR=$LR, 6 SB, div 1.9, skip 75, WDL 0.3)"
  ( cd .archon-state/toolchains/bullet && \
    D40_DATA=$UHO D40_FORMAT=sfbinpack SFBINPACK_SCORE_DIV=1.9 D40_RANDOM_SKIP_PCT=75 \
    D40_INIT_RAW=$INC D40_OUT=$REPO/data/nnue/fleet-relabel/selfcont-ckpt D40_ID=$ID \
    D40_SUPERBATCHES=6 D40_LR_END_SB=6 D40_LR=$LR D40_WDL=0.3 CUDA_PATH=/usr \
    nice -n 5 cargo run --release --features cuda --example d40_features > $REPO/data/nnue/fleet-relabel/sc_${ID}.log 2>&1 )
  uv run python tools/nnue/netpack.py data/nnue/fleet-relabel/selfcont-ckpt/${ID}-sb6-quant.bin data/nnue/nets/${ID}.cenn d40 >/dev/null 2>&1
  # free the raw ckpt immediately (disk discipline)
  rm -f data/nnue/fleet-relabel/selfcont-ckpt/${ID}-sb*-raw.bin
  say "$ID packed; short gate (200g fixed-node) vs incumbent"
  engines/bin/fastchess -engine cmd=engines/bin/probe-sota name=$ID option.EvalFile=data/nnue/nets/${ID}.cenn \
    -engine cmd=engines/bin/probe-sota name=inc option.EvalFile=data/nnue/nets/d40-uho-720sb.cenn \
    -each nodes=25000 option.Threads=1 option.Hash=16 proto=uci \
    -openings file=data/nnue/openings/UHO_Lichess_4852_v1.epd format=epd order=random \
    -rounds 100 -games 2 -repeat -concurrency 6 -recover \
    -pgnout file=data/nnue/fleet-relabel/scgate_${ID}.pgn > data/nnue/fleet-relabel/scgate_${ID}.log 2>&1
  E=$(grep -E "Elo:" data/nnue/fleet-relabel/scgate_${ID}.log | tail -1)
  say "RESULT $ID -> ${E:-no-elo}"
done
say "SWEEP DONE"
