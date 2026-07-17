#!/usr/bin/env bash
# S1 (correction history) EQUAL-TIME gate: c10680f (bench 701605) vs
# pre-S1 df2b746 (bench 548978), same s3 incumbent net both sides, UHO
# book, SPRT [0,10] logistic with 600-game cap at 10+0.1 conc 3.
# Fires only when the box is QUIET: the d40 night chain has completed AND
# the obs labeling run has completed AND no trainer is resident.
# Fixed-node prior: +18.5 +- 17.4 (exp-0071); this decides deployment.
set -uo pipefail
cd "$(dirname "$0")/../.."
R=data/nnue/experiments/chain-v2-results.log
say() { echo "[s1gate $(date +%H:%M:%S)] $*" | tee -a "$R"; }

say "waiting for a quiet box (night chain + labeling + trainer done)"
while true; do
  grep -q "D40 NIGHT CHAIN COMPLETE" "$R" && \
  grep -q "OBS SAMPLE V1 COMPLETE" "$R" && \
  ! pgrep -f 'examples/d40_features' >/dev/null && break
  sleep 300
done
say "box quiet; starting the S1 equal-time SPRT (600g cap, early stop)"
uv run python tools/nnue/eval_match.py \
  --candidate data/nnue/nets/stage2-v1-s3-2048-farseer-120sb.cenn \
  --baseline data/nnue/nets/stage2-v1-s3-2048-farseer-120sb.cenn \
  --games 600 --tc 10+0.1 --concurrency 3 --seed 20260760 \
  --sprt-elo0 0 --sprt-elo1 10 \
  --openings data/nnue/openings/UHO_Lichess_4852_v1.epd \
  --candidate-engine "/tmp/claude-0/-root-projects-learnrust-chessengine/9162696c-0351-4e70-854a-ccb48a50c724/scratchpad/engine-s1" --baseline-engine "/tmp/claude-0/-root-projects-learnrust-chessengine/9162696c-0351-4e70-854a-ccb48a50c724/scratchpad/engine-pre-s1" \
  --output data/nnue/experiments/s1-gate/tc10 \
  || { say "S1 gate FAILED to run"; exit 1; }
python3 -c "
import json
s = json.load(open('data/nnue/experiments/s1-gate/tc10/summary.json'))
sprt = s.get('sprt') or {}
print(f'[S1 EQUAL-TIME GATE] RESULT: {s[\"pgn_games\"]}g elo={s[\"candidate_elo\"]:.1f} se={s[\"candidate_elo_se\"]:.1f} verdict={sprt.get(\"verdict\")} llr={sprt.get(\"llr\")}')" | tee -a "$R"
say "S1 GATE COMPLETE"
