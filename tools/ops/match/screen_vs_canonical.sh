#!/usr/bin/env bash
# tools/ops/match/screen_vs_canonical.sh — pack the newest checkpoint and screen
# it against the canonical net at equal nodes.
#
# PURPOSE
#   The standing "is this run ahead of what we ship?" measurement. Packs the
#   newest checkpoint of a run and plays it against `engines/canonical/net.cenn`
#   at FIXED NODES — because the result must be immune to a busy box, and this
#   machine usually carries a GPU trainer and often a build.
#
# USAGE
#   systemd-run --user --unit=d49cs -p MemoryMax=3G -p MemorySwapMax=0 \
#       /path/to/tools/ops/match/screen_vs_canonical.sh
#
#   Environment:
#     EXP_DIR      checkpoint dir     (default data/nnue/experiments/d49-1280)
#     CKPT_PREFIX  run prefix         (default d49sfb)
#     SB           superbatch to pack (default: newest on disk)
#     SCALE        network_scale used when packing (default 756)
#     GAMES        default 1500     NODES default 25000     CONC default 3
#     RUN_SPEC     spec supplying requant arch params (default d49_sfblend_500.json)
#
# PRODUCES
#   $ARCHON_D49/cs-sb<SB>.out   full fastchess output
#   the verdict block appended to the run's cyc.log
#
# WHY 1500 GAMES
#   The error bar is 383/sqrt(n): 1500 games = +/-10 Elo. **No net verdict is
#   read under ~1500 games.** Interim readings off this harness have twice been
#   over-read and then contradicted by the final number (scale 1068 read as
#   "identical, 0.03 Elo" at n=529 and finished -3.01; scale 756 read as -6.77
#   at n=1027 and finished -1.39).
#
# WHY CONC=3 AND NOT HIGHER
#   Screen concurrency is MEMORY-bound at roughly 2 x conc x 138 MB of nets, and
#   this box has OOM-killed the trainer repeatedly. Because the games are fixed
#   node, a lower concurrency costs wall-clock only — never correctness.
#
# WHY THE EVAL-STDEV LINE COMES BEFORE THE GAMES
#   A large spread difference between candidate and reference means the two are
#   in different eval FRAMES, which confounds the Elo with a scale effect. That
#   is much cheaper to learn before spending 1500 games than after.

source "$(dirname "${BASH_SOURCE[0]}")/../lib/common.sh"

EXP_DIR=${EXP_DIR:-$ARCHON_ROOT/data/nnue/experiments/d49-1280}
CKPT_PREFIX=${CKPT_PREFIX:-d49sfb}
SCALE=${SCALE:-756}
GAMES=${GAMES:-1500}
NODES=${NODES:-25000}
CONC=${CONC:-3}
RUN_SPEC=${RUN_SPEC:-d49_sfblend_500.json}
OPS_LOG="$EXP_DIR/cyc.log"; export OPS_LOG

SB=${SB:-$(ls "$EXP_DIR/$CKPT_PREFIX"-sb*-ema-raw.bin 2>/dev/null \
           | sed -E 's/.*-sb([0-9]+)-ema-raw\.bin/\1/' | sort -n | tail -1)}
[ -n "$SB" ] || { log "FATAL: no checkpoint for prefix '$CKPT_PREFIX' in $EXP_DIR"; exit 2; }

CKPT="$EXP_DIR/$CKPT_PREFIX-sb${SB}-ema-raw.bin"
[ -s "$CKPT" ] || { log "FATAL: checkpoint missing: $CKPT"; exit 2; }

OUT="$ARCHON_ROOT/data/nnue/tmp/canon"; mkdir -p "$OUT"
CAND="$OUT/$CKPT_PREFIX-sb${SB}-${SCALE}.cenn"
BIN=./target-sota/release/chessengine

log "packing sb${SB} at network_scale=${SCALE}"
uv run --no-project --with numpy==2.5.1 python tools/nnue/requant_threats.py \
    "$CKPT" "$OUT/q${SB}.bin" --scale 256 \
    --run-spec "tools/nnue/runs/$RUN_SPEC" >/dev/null 2>&1 \
    || { log "FATAL: requant failed"; exit 3; }
# Pack dims come from the RUN SPEC, never hardcoded: a 1280 default silently
# mispacks any other arch, and D40_HALFKA_ONLY must be set for a threatless net
# or netpack demands a threat table that does not exist. Both bit the D50-HKA768
# sb100 gate on 2026-08-02 -- the net trained fine and simply could not be packed.
SPEC_JSON="$ARCHON_ROOT/tools/nnue/runs/$RUN_SPEC"
D40_ACC=$(python3 -c "import json,sys;print(json.load(open(sys.argv[1])).get('arch_hl',1280))" "$SPEC_JSON")
D40_L2=$(python3 -c "import json,sys;print(json.load(open(sys.argv[1])).get('arch_l2',16))" "$SPEC_JSON")
D40_L3=$(python3 -c "import json,sys;print(json.load(open(sys.argv[1])).get('arch_l3',32))" "$SPEC_JSON")
D40_HALFKA_ONLY=$(python3 -c "import json,sys;print(1 if json.load(open(sys.argv[1])).get('arch_halfka_only') else 0)" "$SPEC_JSON")
export D40_ACC D40_L2 D40_L3 D40_HALFKA_ONLY
log "  pack dims from spec: acc=$D40_ACC l2=$D40_L2 l3=$D40_L3 halfka_only=$D40_HALFKA_ONLY"
D40_POLICY=1 D40_POL_DIM=16 \
D40_THREAT_QUANT=256 D40_NETWORK_SCALE="$SCALE" \
  uv run --no-project --with numpy==2.5.1 python tools/nnue/netpack.py \
    "$OUT/q${SB}.bin" "$CAND" d40 >/dev/null 2>&1 \
    || { log "FATAL: netpack failed"; exit 3; }

if [ -x "$BIN" ]; then
    sd=$("$BIN" nnue-eval "$(readlink -f "$CAND")" < tools/nnue/calib/wl50.fens 2>/dev/null \
         | grep -oE '^-?[0-9]+' \
         | python3 -c "import sys,statistics as st;v=[int(x) for x in sys.stdin];print(f'{st.pstdev(v):.1f}')" 2>/dev/null)
    log "sb${SB} eval stdev=${sd:-unavailable}"
fi

log "SCREEN sb${SB} (scale $SCALE) vs CANONICAL — $GAMES games @ ${NODES}n conc=$CONC"
CS="$ARCHON_D49/cs-sb${SB}.out"
ENGINE="$BIN" CONC="$CONC" bash tools/screen.sh \
    "$CAND" engines/canonical/net.cenn "$GAMES" "$NODES" > "$CS" 2>&1
{ echo "=== SCREEN sb${SB} vs CANONICAL ==="; grep -A5 'Results of' "$CS" | tail -5; } >> "$OPS_LOG"
log "done -> $CS"
