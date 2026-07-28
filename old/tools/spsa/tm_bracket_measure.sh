#!/usr/bin/env bash
# Phase-6b TM-SPSA bracket (user mandate 2026-07-23): equal-time 10+0.1
# vs SF16, run PRE-SPSA (baseline) and POST-SPSA (tuned) — same net, same
# binary, only the tuned constants differ; the pair isolates the SPSA delta.
#
#   bash tools/spsa/tm_bracket_measure.sh pre  <net.cenn> <engine-bin> [games]
#   bash tools/spsa/tm_bracket_measure.sh post <net.cenn> <engine-bin> [games]
#
# post mode additionally requires TM_OPTS='name=value;name=value;...' (the
# SPSA winner θ) exported — passed to the candidate as UCI setoptions.
# HARD GUARD: refuses to run unless the box is idle (equal-time results are
# corrupt under CPU contention — trainer, builds, other matches).
set -euo pipefail
cd "$(dirname "$0")/../.."

TAG="${1:?pre|post}"; NET="${2:?net.cenn}"; ENG="${3:?engine binary}"
GAMES="${4:-300}"
SF=engines/bin/stockfish16
OUT="data/nnue/experiments/tm-bracket-${TAG}-$(basename "$NET" .cenn)"

[ -x "$ENG" ] && [ -x "$SF" ] && [ -s "$NET" ] || { echo "missing engine/net"; exit 2; }
if [ "$TAG" = "post" ] && [ -z "${TM_OPTS:-}" ]; then
  echo "post mode requires TM_OPTS='NAME=V;NAME=V;...' (SPSA winner theta)"; exit 2
fi

# Idle-box guard: no trainer, no cargo, no other fastchess/matches, load sane.
busy=$(pgrep -cf "stage3_train|d40_features|cargo|fastchess|eval_match" || true)
if [ "${busy:-0}" -gt 0 ]; then
  echo "BOX NOT IDLE ($busy competing processes) — equal-time would be corrupt. Refusing."; exit 3
fi
load=$(awk '{print ($1 > 4.0)}' /proc/loadavg)
[ "$load" = "0" ] || { echo "load too high for equal-time; refusing"; exit 3; }

EXTRA=()
if [ -n "${TM_OPTS:-}" ]; then
  # eval_match passes candidate UCI options via --candidate-option NAME=V
  IFS=';' read -ra KV <<< "$TM_OPTS"
  for kv in "${KV[@]}"; do EXTRA+=(--candidate-option "$kv"); done
fi

echo "[tm-bracket:$TAG $(date +%H:%M:%S)] ${GAMES}g 10+0.1 $(basename "$NET") vs SF16 -> $OUT"
python3 tools/nnue/eval_match.py \
  --candidate "$NET" --baseline default \
  --tc "10+0.1" --games "$GAMES" --concurrency 1 \
  --candidate-engine "$ENG" --baseline-engine "$SF" \
  --openings engines/books/UHO_Lichess_4852_v1.epd \
  --output "$OUT" "${EXTRA[@]}"
python3 - "$OUT/summary.json" "$TAG" <<'EOF'
import json, sys
d = json.load(open(sys.argv[1]))
print(f"[tm-bracket:{sys.argv[2]}] RESULT elo {d['candidate_elo']:+.1f} +/- "
      f"{d['candidate_elo_se']:.1f} vs SF16 (10+0.1) "
      f"W-D-L {d.get('candidate_wins','?')}-{d.get('draws','?')}-{d.get('candidate_losses','?')}")
EOF
