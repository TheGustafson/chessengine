#!/usr/bin/env bash
# D49 endpoint verdict — the protocol, encoded so the decisions survive.
#
# WHY THIS EXISTS. Three separate traps were identified during the run and each
# would silently corrupt the verdict:
#
#  1. COMPARE EACH NET AT ITS OWN OPTIMUM. Canonical ships network_scale=600
#     because @400 measured 14.78 +/- 5.45 Elo WORSE over 5880 games. Screening
#     D49 against canonical@400 would hand D49 a ~15 Elo head start unrelated to
#     its recipe.
#  2. D49's OWN optimum is net-specific. The "matching eval stdev transfers"
#     hypothesis was REFUTED: D49@1111 (stdev 600) lost -14.47 +/- 11.09 to
#     D49@756 (stdev 415), while D48X prefers stdev 610. So sweep, do not
#     assume -- but the frame was stable across sb140-250 (235.0/229.9/234.1),
#     so a NARROW ladder around 756 suffices unless sb400 has drifted.
#  3. DEPLOY THE EMA. The heldout log's ema column is the deployed net and it
#     has run 200-500e-6 below the raw column all run. Export *-ema-raw.bin.
#
# Usage:  bash tools/d49_verdict.sh <SB>        e.g. bash tools/d49_verdict.sh 400
set -euo pipefail
SB="${1:?superbatch, e.g. 400}"
ROOT="$(cd "$(dirname "${BASH_SOURCE[0]}")/.." && pwd)"; cd "$ROOT"
CK="data/nnue/experiments/d49-1280/d49-1280-sb${SB}-ema-raw.bin"
[ -r "$CK" ] || { echo "no checkpoint: $CK" >&2; exit 1; }
OUT=data/nnue/tmp/verdict; mkdir -p "$OUT"
BIN=./target-sota/release/chessengine
[ -x "$BIN" ] || { echo "build first: cargo build --release --features sota,arch-d48x (CARGO_TARGET_DIR=target-sota)" >&2; exit 1; }

echo "== 1. requant + pack the EMA checkpoint at the ladder scales =="
uv run --no-project --with numpy==2.5.1 python tools/nnue/requant_threats.py \
    "$CK" "$OUT/q.bin" --scale 256 --run-spec tools/nnue/runs/d49_1280.json
for s in 640 756 880; do
  D40_ACC=1280 D40_L2=16 D40_L3=32 D40_POLICY=1 D40_POL_DIM=16 \
  D40_THREAT_QUANT=256 D40_NETWORK_SCALE=$s \
    uv run --no-project --with numpy==2.5.1 python tools/nnue/netpack.py \
      "$OUT/q.bin" "$OUT/d49-sb${SB}-${s}.cenn" d40 >/dev/null
done

echo "== 2. re-measure the frame (has the anneal moved it off 756?) =="
for s in 640 756 880; do
  printf "  scale %-4s stdev=" "$s"
  $BIN nnue-eval "$(readlink -f "$OUT/d49-sb${SB}-${s}.cenn")" \
      < tools/nnue/calib/wl50.fens 2>/dev/null | grep -oE '^-?[0-9]+' \
    | python3 -c "import sys,statistics as st;v=[int(x) for x in sys.stdin];print(f'{st.pstdev(v):.1f}')"
done
echo "  (canonical D48X@600 is 610.3; D49 measured 415-422 at 756 through sb270)"

echo
echo "== 3. LADDER: DISCHARGED -- SKIP unless the frame above has moved =="
echo "  Trap 2 in this file's header said 'sweep, do not assume -- but the frame"
echo "  was stable across sb140-250, so a NARROW ladder suffices unless sb400 has"
echo "  drifted.' It has NOT drifted, measured three times at scale 756:"
echo "      through sb270 : 415-422"
echo "      sb900         : 423.5"
echo "      sb1060        : 426.1"
echo "  Stable across 790 superbatches, through the anneal AND eleven LR cycles."
echo "  756 stands. SKIP to step 4 -- this saves two 1466-game screens (~1 h)."
echo
echo "  Run the ladder ONLY if the stdev printed above is outside ~415-430:"
echo "    ENGINE=$BIN bash tools/screen.sh $OUT/d49-sb${SB}-880.cenn $OUT/d49-sb${SB}-756.cenn 1466 25000"
echo "    ENGINE=$BIN bash tools/screen.sh $OUT/d49-sb${SB}-640.cenn $OUT/d49-sb${SB}-756.cenn 1466 25000"
echo
echo "== 4. VERDICT: D49 at its optimum vs canonical AT ITS OWN OPTIMUM =="
echo "  ENGINE=$BIN bash tools/screen.sh $OUT/d49-sb${SB}-<best>.cenn engines/canonical/net.cenn 5880 25000"
echo
echo
echo "ENGINE= IS NOT OPTIONAL. engines/canonical/archon (screen.sh's default)"
echo "predates CENN-v5 and CANNOT load a netpack-with-THREAT_QUANT container."
echo "On 2026-07-30 that silently disabled NNUE on one side and returned"
echo "-1106.98 Elo over 1466 games with FIVE draws. screen.sh now refuses to"
echo "start in that case, but the invocations above name the right engine so it"
echo "never gets that far."
echo "Run each as a systemd SERVICE (a --scope dies with the session and killed"
echo "a 5880-game screen at 1043). ENGINE=$BIN CONC=<=10."
