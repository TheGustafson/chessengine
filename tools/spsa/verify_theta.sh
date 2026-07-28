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
# Defaults follow the CANONICAL pointer (CLAUDE.md: scripts reference
# engines/canonical/net.cenn, never a bare filename). They were pinned to the
# D47V pair, which silently became WRONG when D48X was promoted on 2026-07-28:
# a vector tuned on arch-d48x would have been verified on a 768-wide D47V
# engine+net, i.e. a different engine entirely, and the mismatch is invisible
# because both sides of the A/B would have used it.
#
# archon-trace is the only tune-feature build for the canonical arch. It is
# non-PGO, which is a deviation from the standing PGO ruling but not a bias:
# both sides are the SAME binary, so at equal time both simply search fewer
# nodes. Rebuild a PGO tune binary if absolute strength matters here.
ENGINE=${ARCHON_VERIFY_ENGINE:-engines/bin/archon-trace}
NET=${ARCHON_VERIFY_NET:-engines/canonical/net.cenn}
THETA=${1:-}
GAMES=${2:-600}
TC=${3:-10+0.1}
CONC=${4:-6}

if [ -z "$THETA" ]; then
  THETA=$(ls -t tools/spsa/out/*/theta_final.json 2>/dev/null | head -1 || true)
fi
[ -n "$THETA" ] && [ -e "$THETA" ] || {
  THETA=$(ls -t "$HOME"/spsa-ex-*/state/theta-latest.json "$HOME"/.cache/archon-spsa-stageA/state/theta-latest.json 2>/dev/null | head -1 || true)
  echo "[verify] no theta_final.json — falling back to the INTERIM vector $THETA"
}
[ -e "$THETA" ] || { echo "[verify] no theta anywhere; nothing to verify"; exit 1; }

# theta_final.json carries {"final": {...}}; theta-latest.json carries
# {"theta": {...}}. Accept either. Integer-valued params render as ints the
# way the engine will read them, but genuinely fractional ones (the reduction
# table's REDUCTION_BASE / REDUCTION_DEN_COEFF are f64) must NOT be rounded —
# rounding 1.48 to 1 would be a large silent change to the reduction shape,
# and the verify would then measure something the tuner never proposed.
OPTS=$(python3 - "$THETA" <<'PY'
import json, sys
d = json.load(open(sys.argv[1]))
vals = d.get("final") or d.get("theta") or {}
out = []
for k, v in sorted(vals.items()):
    f = float(v)
    out.append(f"option.{k}={int(round(f))}" if f == int(f) else
               f"option.{k}={f:.6f}")
print(" ".join(out))
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
