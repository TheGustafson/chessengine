#!/usr/bin/env bash
# The gate every SPSA vector must clear before a single number moves in
# params.rs: does the tuned vector actually beat the defaults at a real
# time control?
#
#   bash tools/spsa/verify_theta.sh [theta.json] [games] [tc]
#
# theta.json defaults to the newest completed stage-A run's theta_final.json,
# and falls back to state/theta-latest.json on the exchange if the run is
# still going (an interim read is legitimate — just say so when quoting it).
#
# Both sides are the SAME tune-feature binary and the SAME net. One gets the
# tuned options, the other gets nothing and therefore the compiled defaults.
# That isolates the vector: no build difference, no net difference, and the
# tune build's option-read overhead is paid identically by both, so it cannot
# bias an equal-time result.
#
# fn tuning is load-immune, but this is EQUAL TIME and the box may be
# training. Load hits both engines symmetrically (they alternate colours and
# share the machine), so it adds variance, not bias — still, prefer a quiet
# box when the margin matters, and always quote games + error bars.
set -eu
cd "$(dirname "$0")/../.."
ENGINE=engines/bin/archon-d47v-tune-pgo
NET=data/nnue/nets/screens/d47v-a4b2-cal400.cenn
THETA=${1:-}
GAMES=${2:-600}
TC=${3:-10+0.1}
CONC=${4:-6}

if [ -z "$THETA" ]; then
  THETA=$(ls -t tools/spsa/out/s2-stageA-*/theta_final.json 2>/dev/null | head -1 || true)
fi
[ -n "$THETA" ] && [ -e "$THETA" ] || {
  THETA="$HOME/.cache/archon-spsa-stageA/state/theta-latest.json"
  echo "[verify] no theta_final.json — falling back to the INTERIM vector $THETA"
}
[ -e "$THETA" ] || { echo "[verify] no theta anywhere; nothing to verify"; exit 1; }

# theta_final.json carries {"final": {...}}; theta-latest.json carries
# {"theta": {...}}. Accept either, and round to ints the way the engine will.
OPTS=$(python3 - "$THETA" <<'PY'
import json, sys
d = json.load(open(sys.argv[1]))
vals = d.get("final") or d.get("theta") or {}
print(" ".join(f"option.{k}={int(round(float(v)))}" for k, v in sorted(vals.items())))
PY
)
[ -n "$OPTS" ] || { echo "[verify] theta had no parameters"; exit 1; }
echo "[verify] theta   : $THETA"
echo "[verify] tuned   : $OPTS"
echo "[verify] $GAMES games at $TC, concurrency $CONC, both sides $ENGINE"

OUT=data/nnue/experiments/spsa-stageA-verify
mkdir -p "$OUT"
# shellcheck disable=SC2086
engines/bin/fastchess \
  -engine cmd="$ENGINE" name=tuned option.EvalFile="$NET" $OPTS \
  -engine cmd="$ENGINE" name=default option.EvalFile="$NET" \
  -each proto=uci tc="$TC" option.Threads=1 option.Hash=64 \
  -openings file=engines/books/UHO_Lichess_4852_v1.epd format=epd order=random \
  -repeat -games 2 -rounds $((GAMES / 2)) -concurrency "$CONC" -recover \
  -draw movenumber=40 movecount=8 score=15 -resign movecount=4 score=700 -maxmoves 250 \
  -srand 20260801 -pgnout file="$OUT/games.pgn" notation=lan 2>&1 | tail -12
echo "[verify] done — a positive, error-bar-clearing result is what licenses"
echo "[verify] editing src/search/params.rs; anything else stays unlanded."
