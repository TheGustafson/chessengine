#!/usr/bin/env bash
# grow_net.sh — widen a trained D40 net to a bigger accumulator + warm fine-tune it,
# with a HARD function-preservation gate so a transform bug can't slip through.
#
#   grow_net.sh <src_raw.bin> <new_hl> <parent.cenn> <finetune_sb> [out_id]
#
# Net2WiderNet (tools/nnue/net2wider.py) copies the trained narrow net into a wider
# one whose new neurons are zero-init, so it starts function-IDENTICAL to the parent;
# a short fine-tune then develops the new capacity. Much cheaper than from-scratch.
#
# PREREQUISITES (both must be built AT new_hl — the width is a compile-time const):
#   trainer:  set HL=<new_hl> in .archon-state/toolchains/bullet/examples/d40_features.rs,
#             then (cd .archon-state/toolchains/bullet && CUDA_PATH=/opt/cuda \
#               cargo build --release --features cuda --example d40_features)
#   engine:   set L1=<new_hl> in src/eval/nnue/sota_head.rs + the ACC assert in
#             sota_features.rs, then RUSTFLAGS="-C target-cpu=native" cargo build \
#               --release --features sota  ->  cp target/release/chessengine \
#               engines/bin/archon-sota-<new_hl>
#
# Training recipe is env-overridable (defaults = the UHO mix-4ep recipe):
#   D40_DATA, SFBINPACK_SCORE_DIV, D40_MIX_DATA, D40_MIX_PCT, D40_LR, D40_SEED, ...
set -uo pipefail
cd "$(dirname "$0")/../.."

SRC="${1:?src raw .bin}"; NEW_HL="${2:?new HL width}"; PARENT="${3:?parent .cenn for the preservation check}"
FT_SB="${4:?fine-tune superbatches}"; OUT_ID="${5:-grown-${NEW_HL}-ft${4}}"
TRAINER=.archon-state/toolchains/bullet/target/release/examples/d40_features
ENG_NEW="engines/bin/archon-sota-${NEW_HL}"; ENG_PARENT="${ENG_PARENT:-engines/bin/archon-sota-native}"
FENS="${FENS:-data/nnue/canon-v1/inputs/tasks/000000.fens}"
say(){ echo "[grow $(date -u +%H:%M:%S)] $*"; }
for f in "$SRC" "$PARENT" "$TRAINER" "$ENG_NEW"; do [ -e "$f" ] || { echo "missing: $f"; exit 1; }; done

# recipe defaults (override via env)
export D40_DATA="${D40_DATA:-data/nnue/binpacks/nodes5000pv2_UHO.binpack}"
export D40_FORMAT="${D40_FORMAT:-sfbinpack}" SFBINPACK_SCORE_DIV="${SFBINPACK_SCORE_DIV:-1.9}"
export D40_RANDOM_SKIP_PCT="${D40_RANDOM_SKIP_PCT:-75}" D40_SEED="${D40_SEED:-198273612}"
export D40_BATCH="${D40_BATCH:-16384}" D40_BPS="${D40_BPS:-1520}" D40_WDL="${D40_WDL:-0.3}"
export D40_EVAL_TARGET_SCALE="${D40_EVAL_TARGET_SCALE:-400}" D40_SCORE_MULTIPLIER="${D40_SCORE_MULTIPLIER:-1.0}"
export D40_HALFKA_ONLY="${D40_HALFKA_ONLY:-0}" D40_SF_BUFFER_MB="${D40_SF_BUFFER_MB:-128}" D40_THREADS="${D40_THREADS:-6}"
export CUDA_PATH=/opt/cuda LD_LIBRARY_PATH=/opt/cuda/lib64
MIX_ENV=(); [ -n "${D40_MIX_DATA:-}" ] && MIX_ENV=(D40_MIX_DATA="$D40_MIX_DATA" D40_MIX_PCT="${D40_MIX_PCT:-0.231}")

GROWN="data/nnue/experiments/$OUT_ID/ckpt/grown-init-raw.bin"; mkdir -p "$(dirname "$GROWN")"

# 1) widen
say "widen $SRC -> HL=$NEW_HL"
uv run --no-project --with numpy python tools/nnue/net2wider.py "$SRC" "$GROWN" "$NEW_HL"

# 2) HARD preservation gate: frozen (LR=0) save -> netpack -> eval must match parent
say "validating function-preservation (frozen save)"
VCK="data/nnue/experiments/$OUT_ID/validate"; mkdir -p "$VCK"
env "${MIX_ENV[@]}" D40_INIT_RAW="$GROWN" D40_ID=grown-validate D40_OUT="$VCK" \
  D40_SUPERBATCHES=1 D40_LR_END_SB=1 D40_SAVE_RATE=1 D40_LR=0.0 "$TRAINER" >/dev/null 2>&1
D40_ACC=$NEW_HL uv run --no-project --with numpy python tools/nnue/netpack.py \
  "$VCK/grown-validate-sb1-quant.bin" "$VCK/grown-validate.cenn" d40 >/dev/null 2>&1
head -400 "$FENS" > /tmp/grow_cmp.fens
RMSE=$(paste <(cat /tmp/grow_cmp.fens | "$ENG_NEW"    nnue-eval "$VCK/grown-validate.cenn" 2>/dev/null) \
             <(cat /tmp/grow_cmp.fens | "$ENG_PARENT" nnue-eval "$PARENT"                  2>/dev/null) \
  | awk 'NF==2{d=$1-$2;s+=d*d;n++} END{if(n)printf "%.3f",sqrt(s/n); else print "nan"}')
say "preservation rmse=${RMSE}cp vs parent"
awk "BEGIN{exit !($RMSE < 6.0)}" || { echo "ABORT: grown net is NOT function-identical (rmse=$RMSE) — transform bug"; exit 2; }
say "✓ function-identical — transform correct"

# 3) warm fine-tune
say "warm fine-tune $FT_SB sb (LR ${D40_LR:-0.0005})"
FCK="data/nnue/experiments/$OUT_ID/ckpt"
env "${MIX_ENV[@]}" D40_INIT_RAW="$GROWN" D40_ID="$OUT_ID" D40_OUT="$FCK" \
  D40_SUPERBATCHES="$FT_SB" D40_LR_END_SB="$FT_SB" D40_SAVE_RATE="$FT_SB" D40_LR="${D40_LR:-0.0005}" \
  "$TRAINER" 2>&1 | grep -aE "FOLD-IN|DONE|error|panic" | tail -3
D40_ACC=$NEW_HL uv run --no-project --with numpy python tools/nnue/netpack.py \
  "$FCK/$OUT_ID-sb$FT_SB-quant.bin" "data/nnue/nets/$OUT_ID.cenn" d40 >/dev/null 2>&1
say "DONE -> data/nnue/nets/$OUT_ID.cenn  (gate it with $ENG_NEW; parent uses $ENG_PARENT)"
