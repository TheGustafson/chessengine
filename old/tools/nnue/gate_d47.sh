#!/usr/bin/env bash
# Gate a D47 net (1536/256 multihead) — LEARNABILITY-FIRST framing (user
# directive: "we learn if we can learn; NPS optimization comes later").
#   1. hard scale-check (cross-arch: candidate in archon-d47, ref in -native)
#   2. policy top-1 accuracy (the primary learnability metric)
#   3. fixed-node 5k vs the 768 incumbent (eval-quality readout; NPS-blind)
#   4. NPS measurement (REPORTED, not gated — the accepted D47 cost)
# Equal-time is deliberately NOT part of this gate (post-optimization only).
#   bash tools/nnue/gate_d47.sh <net.cenn> [games]
set -uo pipefail
cd "$(dirname "$0")/../.."
CAND="${1:?d47 net.cenn}"
GAMES="${2:-300}"
DENG=engines/bin/archon-d47
NENG=engines/bin/archon-sota-native
INC=data/nnue/nets/d40-uho-v1-240sb.cenn

echo "=== [1] scale check (hard) ==="
REF_ENGINE="$NENG" bash tools/nnue/net_scale_check.sh "$CAND" "$INC" "$DENG" || {
  [ "${FORCE:-0}" = 1 ] || { echo "[gate-d47] ABORT: scale FAIL (FORCE=1 to override)"; exit 1; }
  echo "[gate-d47] scale FAIL but FORCE=1 — continuing"; }

echo "=== [2] policy learnability ==="
ENG="$DENG" bash tools/nnue/policy_accuracy_probe.sh "$CAND"

echo "=== [3] fixed-node 5k vs incumbent (${GAMES}g) ==="
OUT="data/nnue/experiments/gate-$(basename "$CAND" .cenn)"
python3 tools/nnue/eval_match.py \
  --candidate "$CAND" --baseline "$INC" \
  --nodes 5000 --games "$GAMES" --concurrency 8 \
  --candidate-engine "$DENG" --baseline-engine "$NENG" \
  --openings engines/books/UHO_Lichess_4852_v1.epd \
  --output "$OUT" 2>/dev/null
python3 -c "import json;d=json.load(open('$OUT/summary.json'));print(f'  RESULT: score={d[\"candidate_score\"]:.3f}  elo={d[\"candidate_elo\"]:.1f} +/- {d[\"candidate_elo_se\"]:.1f}')"

echo "=== [4] NPS (reported, not gated) ==="
for lbl in "D47 $CAND $DENG" "INCUMBENT $INC $NENG"; do
  set -- $lbl
  ( printf 'uci\nsetoption name EvalFile value %s\nisready\nposition startpos\ngo movetime 4000\n' "$2"; sleep 5 ) \
    | timeout 12 "$3" 2>&1 | tr '\r' '\n' | grep -aoE "nps [0-9]+" | tail -1 | sed "s/^/  $1: /"
done
echo "[gate-d47] COMPLETE"
