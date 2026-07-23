#!/usr/bin/env bash
# ============================================================================
# D41 "drop-threats + 2x-accumulator" net: 1536-wide HalfKA-only (NO
# FullThreats), L2 head bottleneck widened 16->32.
#
# Executive-decision arch (2026-07-21): drop FullThreats (the NPS tax), spend
# the freed NPS on a 2x accumulator, and widen the L1->L2 bottleneck 16->32
# (the leading suspect for why wider nets stopped learning).  Arch dims are
# COMPILED into the trainer (HL=1536, L2=32) + engine (sota_head::L1=1536,
# L2=32); D40_HALFKA_ONLY=1 makes the saved quant.bin genuinely threatless.
#
# Two-stage proven recipe (the B4 winner, retargeted to the new arch):
#   Stage 1: UHO (sfbinpack), div 1.9, skip75, WDL 0.3, 240 SB, LR 1e-3, scratch.
#   Stage 2: farseerT76 warm-continue (D40_INIT_RAW), div 1.9, WDL 0.25,
#            120 SB, LR 5e-4.
#
# Usage:
#   bash tools/nnue/train_1536_halfka.sh          # full run (~3.6h GPU)
#   SMOKE=1 bash tools/nnue/train_1536_halfka.sh  # 3+3 SB end-to-end validation
# Requires: CUDA at /opt/cuda; trainer built --features cuda; free GPU.
# ============================================================================
set -uo pipefail
cd "$(dirname "$0")/../.."

TRAINER=.archon-state/toolchains/bullet/target/release/examples/d40_features
UHO=data/nnue/binpacks/nodes5000pv2_UHO.binpack
FARSEER=data/nnue/binpacks/farseerT76.binpack
[ -x "$TRAINER" ] || { echo "trainer not built: $TRAINER"; exit 1; }
[ -s "$UHO" ]     || { echo "UHO binpack missing: $UHO"; exit 1; }
[ -s "$FARSEER" ] || { echo "farseer binpack missing: $FARSEER"; exit 1; }

# new-arch netpack env (must match compiled trainer/engine consts)
PACK_ENV=(D40_ACC=1536 D40_L2=32 D40_L3=32 D40_BUCKETS=8 D40_HALFKA_ONLY=1)

# Superbatches: the 1536 net has 2x the feature-transformer capacity of the 768
# incumbent, so the proven 240/120 schedule would UNDER-train it — the classic
# width false-negative ([[no-verdicts-from-weak-tests]]: "the wide net is badly
# under-trained"). Bump ~1.25x to give the doubled capacity data to fill; LR_END
# tracks SUPERBATCHES so the decay stretches to match. Env-overridable.
SMOKE="${SMOKE:-0}"
if [ "$SMOKE" = 1 ]; then S1_SB=3; S2_SB=3; TAG=smoke-;
else S1_SB="${S1_SB:-300}"; S2_SB="${S2_SB:-140}"; TAG=; fi

say() { echo "[1536 $(date +%H:%M:%S)] $*"; }

pack() { # ckpt id sb out
  local ckpt="$1" id="$2" sb="$3" out="$4"
  local q="$ckpt/${id}-sb${sb}-quant.bin"
  [ -s "$q" ] || { say "PACK ERR: quant missing $q"; ls -la "$ckpt"; return 1; }
  env "${PACK_ENV[@]}" uv run --no-project --with numpy \
    python tools/nnue/netpack.py "$q" "$out" d40 2>&1 | tail -5
  [ -s "$out" ] || { say "PACK ERR: netpack produced nothing"; return 1; }
  # prove the engine loads it
  printf 'uci\nsetoption name EvalFile value %s\nisready\nquit\n' "$out" \
    | engines/bin/archon-sota-1536 2>&1 | grep -q readyok \
    && say "engine LOADS $out (readyok)" || { say "PACK ERR: engine cannot load $out"; return 1; }
  ls -la "$out"
}

# --------------------------------------------------------------------------
# STAGE 1 — UHO, from scratch
# --------------------------------------------------------------------------
S1_ID="h1536-${TAG}uho-${S1_SB}sb"
S1_CKPT="data/nnue/experiments/$S1_ID/ckpt"; mkdir -p "$S1_CKPT"
S1_NET="data/nnue/nets/$S1_ID.cenn"
S1_RAW="$S1_CKPT/${S1_ID}-final-raw.bin"
if [ ! -s "$S1_NET" ]; then
  say "STAGE 1: $S1_ID  (1536 HalfKA-only, UHO, div1.9 skip75 WDL0.3 ${S1_SB}sb LR1e-3)"
  D40_DATA="$UHO" D40_FORMAT=sfbinpack D40_SF_BUFFER_MB=256 \
  D40_HALFKA_ONLY=1 D40_FACTORIZE=0 \
  D40_ID="$S1_ID" D40_OUT="$S1_CKPT" D40_SEED=198273612 \
  D40_SUPERBATCHES="$S1_SB" D40_LR_END_SB="$S1_SB" D40_SAVE_RATE="$S1_SB" \
  D40_BATCH=16384 D40_BPS=1520 \
  D40_LR=0.001 D40_WDL=0.3 \
  SFBINPACK_SCORE_DIV=1.9 D40_RANDOM_SKIP_PCT=75 \
  D40_EVAL_TARGET_SCALE=400 D40_SCORE_MULTIPLIER=1.0 \
  D40_VERIFY_ROWS=4096 CUDA_PATH=/opt/cuda \
  "$TRAINER" || { say "STAGE 1 trainer FAILED"; exit 1; }
  pack "$S1_CKPT" "$S1_ID" "$S1_SB" "$S1_NET" || exit 1
  say "STAGE 1 packed OK -> $S1_NET"
else
  say "STAGE 1 net exists ($S1_NET) — skipping"
fi
[ -s "$S1_RAW" ] || { say "STAGE 1 raw missing (needed for warm-start): $S1_RAW"; exit 1; }

# --------------------------------------------------------------------------
# STAGE 2 — farseerT76, warm-continue from stage 1
# --------------------------------------------------------------------------
S2_ID="h1536-${TAG}s2-farseer-${S2_SB}sb"
S2_CKPT="data/nnue/experiments/$S2_ID/ckpt"; mkdir -p "$S2_CKPT"
S2_NET="data/nnue/nets/$S2_ID.cenn"
if [ ! -s "$S2_NET" ]; then
  say "STAGE 2: $S2_ID  (warm from $S1_RAW, farseer, div1.9 WDL0.25 ${S2_SB}sb LR5e-4)"
  D40_DATA="$FARSEER" D40_FORMAT=sfbinpack D40_SF_BUFFER_MB=256 \
  D40_HALFKA_ONLY=1 D40_FACTORIZE=0 \
  D40_INIT_RAW="$S1_RAW" \
  D40_ID="$S2_ID" D40_OUT="$S2_CKPT" D40_SEED=198273612 \
  D40_SUPERBATCHES="$S2_SB" D40_LR_END_SB="$S2_SB" D40_SAVE_RATE="$S2_SB" \
  D40_BATCH=16384 D40_BPS=1520 \
  D40_LR=0.0005 D40_WDL=0.25 \
  SFBINPACK_SCORE_DIV=1.9 \
  D40_EVAL_TARGET_SCALE=400 D40_SCORE_MULTIPLIER=1.0 \
  D40_VERIFY_ROWS=4096 CUDA_PATH=/opt/cuda \
  "$TRAINER" || { say "STAGE 2 trainer FAILED"; exit 1; }
  pack "$S2_CKPT" "$S2_ID" "$S2_SB" "$S2_NET" || exit 1
  say "STAGE 2 packed OK -> $S2_NET"
else
  say "STAGE 2 net exists ($S2_NET) — skipping"
fi

say "DONE.  stage1=$S1_NET"
say "        SHIP CANDIDATE (stage2)=$S2_NET"
