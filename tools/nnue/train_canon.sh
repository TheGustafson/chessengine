#!/usr/bin/env bash
# Train the first SF18-canonical D40 eval net (native-Arch, 2026-07-20).
# Twin of the incumbent d40-uho-v1-240sb recipe, retargeted to the canon-v1
# curated SF18 corpus.
#
# SCALE CORRECTION (2026-07-21, docs/research/sf18-canon-data-diagnosis-2026-07-21.md
# + own re-derivation): the original header claim "canon UCI cp == binpack_cp/1.9"
# is FALSE and caused the -582/-400 collapse (half-scale net breaks every
# co-adapted search margin). It is NOT a broken label.
#
# Derivation (verified): engine value = l3 * network_scale, netpack bakes
# network_scale=400 for EVERY recipe, and l3 ~= training_score * MULT / 400, so
# value ~= training_score * MULT. Binpack path: training_score = binpack_internal/1.9
# (SFBINPACK_SCORE_DIV, applied in sfbinpack.rs), MULT=1.0 => value ~= binpack_internal/1.9.
# Text/canon path: training_score = raw SF18-UCI cp, so value ~= SF18_UCI * MULT.
# Scale-match the incumbent => MULT = K/1.9 where K = binpack_internal/SF18_UCI cp,
# measured K = 2.64-2.76 (r=0.94, n=798) => MULT ~= 1.42.
#
# *** MULT IS COUPLED TO WDL ***: 1.42 is for the TARGET recipe WDL=0.3 (blend
# cancels in the incumbent ratio). The 1.94 exp-0111 found is the WDL=0 value
# (no blend => a bigger MULT needed to match the WDL=0.3 incumbent's sharpened
# output). This legacy script still runs WDL=0 (0.5 placeholder result byte, no
# game-WDL join); the TIGHT recipe adds a real WDL join + WDL=0.3 + MULT=1.42.
# AUTHORITY: a post-train calibration self-check (net output-magnitude vs the
# incumbent on a fixed position set) is what actually pins MULT; the numbers
# above are the starting point, not the final word.
#
#   train_canon.sh                 # full 240sb (~2.4h on RTX 4060 Ti)
#   SMOKE=1 train_canon.sh         # 3 SB smoke (~2min) to validate end-to-end
#   FACTORIZE=1 train_canon.sh     # + king-bucket factorization (#27)
# Requires: CUDA at /opt/cuda, corpus.shuf.bin built, trainer built for cuda.
set -uo pipefail
cd "$(dirname "$0")/../.."

DATA="${DATA:-data/nnue/canon-v1/build/corpus.shuf.bin}"
[ -s "$DATA" ] || { echo "corpus not built: $DATA (run build_canon_corpus.sh)"; exit 1; }
TRAINER=.archon-state/toolchains/bullet/target/release/examples/d40_features
[ -x "$TRAINER" ] || { echo "trainer not built (need CUDA): $TRAINER"; exit 1; }

FACT="${FACTORIZE:-0}"
SB="${SB:-240}"
ID="canon-v1-${SB}sb"; [ "$FACT" = 1 ] && ID="canon-v1-fac-${SB}sb"
if [ "${SMOKE:-0}" = 1 ]; then SB=3; ID="canon-v1-smoke"; fi
ID="${NET_ID:-$ID}"
CKPT="data/nnue/experiments/$ID/ckpt"; mkdir -p "$CKPT"
echo "[train_canon] id=$ID data=$DATA factorize=$FACT superbatches=$SB"

# Warm-start / finetune: INIT_RAW=<base -final-raw.bin> loads a base net's weights
# (D40_INIT_RAW) and continues training from there — the mechanism from
# d40_s2_gate_v1.sh. Pair with a reduced LR (e.g. LR=0.0003). Default off = from-scratch.
if [ -n "${INIT_RAW:-}" ]; then
  [ -s "$INIT_RAW" ] || { echo "INIT_RAW not found: $INIT_RAW"; exit 1; }
  export D40_INIT_RAW="$INIT_RAW"
  echo "[train_canon] WARM-START from $INIT_RAW  (LR=${LR:-0.001})"
fi
# WD overrides AdamW weight decay (default 0.01); WD=0 for a scale-preserving finetune.
[ -n "${WD:-}" ] && { export D40_WEIGHT_DECAY="$WD"; echo "[train_canon] weight_decay=$WD"; }

# WDL-coupled SCORE_MULT default (see the SCALE CORRECTION header): the text/canon
# path carries raw SF18-UCI cp, so MULT alone maps it to the incumbent's scale.
# WDL=0 (legacy, no game-WDL join) needs ~1.94 to match the WDL=0.3 incumbent;
# WDL>0 (tight recipe, real join) needs ~1.42 = K/1.9 (K=2.7). Explicit SCORE_MULT
# always wins. The post-train calibration check (tools/nnue/net_scale_check.sh) is
# the final authority on this number.
WDL="${WDL:-0}"
if [ -z "${SCORE_MULT:-}" ]; then
  if awk "BEGIN{exit !($WDL>0)}"; then SCORE_MULT=1.42; else SCORE_MULT=1.94; fi
  echo "[train_canon] SCORE_MULT auto-set to $SCORE_MULT for WDL=$WDL (override with SCORE_MULT=)"
fi

D40_DATA="$DATA" \
D40_FORMAT=bulletformat \
D40_FACTORIZE="$FACT" \
D40_ID="$ID" \
D40_OUT="$CKPT" \
D40_SEED=198273612 \
D40_SUPERBATCHES="$SB" D40_LR_END_SB="$SB" D40_SAVE_RATE="${SAVE_RATE:-$SB}" \
D40_BATCH=16384 D40_BPS=1520 \
D40_LR="${LR:-0.001}" D40_WDL="$WDL" \
D40_EVAL_TARGET_SCALE=400 D40_SCORE_MULTIPLIER="$SCORE_MULT" \
D40_HALFKA_ONLY=0 D40_SF_BUFFER_MB=128 D40_VERIFY_ROWS=4096 \
CUDA_PATH=/opt/cuda \
"$TRAINER"
rc=$?
echo "[train_canon] trainer exit=$rc"
[ $rc -ne 0 ] && exit $rc

# pack the merged quant checkpoint -> drop-in .cenn
QUANT="$CKPT/${ID}-sb${SB}-quant.bin"
NET="data/nnue/nets/${ID}.cenn"
if [ -s "$QUANT" ]; then
  echo "[train_canon] netpack $QUANT -> $NET"
  uv run --no-project --with numpy python tools/nnue/netpack.py "$QUANT" "$NET" d40 2>&1 | tail -3
  ls -la "$NET"
else
  echo "[train_canon] WARN: quant checkpoint not found at $QUANT — inspect $CKPT"
  ls -la "$CKPT"
fi
echo "[train_canon] DONE id=$ID"
