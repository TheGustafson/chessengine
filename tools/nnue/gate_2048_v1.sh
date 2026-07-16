#!/usr/bin/env bash
# Cross-width equal-time promotion gate: uf2048-240sb vs incumbent
# u1024-120sb, 200g 10+0.1, QUIET BOX (waits for the stage-2 train to
# finish so nothing else runs). Completion detected via the results-log
# line stage2_v1.sh writes after its whole pipeline (train+screen) ends.
set -uo pipefail
cd "$(dirname "$0")/../.."

R=data/nnue/experiments/chain-v2-results.log
say() { echo "[gate2048 $(date +%H:%M:%S)] $*" | tee -a "$R"; }

say "waiting for stage2 attempt 1 to release the box"
until grep -aqE "^\[stage2 [0-9:]+\] STAGE2 ATTEMPT1 DONE" "$R"; do sleep 120; done
sleep 30

d=data/nnue/experiments/uho-wave4/uf2048-240sb/gate-tc10
if [ ! -f "$d/summary.json" ] || ! grep -q '"status": "complete"' "$d/summary.json" 2>/dev/null; then
  say "equal-time gate: uf2048-240sb vs u1024-120sb (200g 10+0.1, conc 3)"
  uv run python tools/nnue/eval_match.py \
    --candidate data/nnue/nets/uho-wave4-uf2048-240sb.cenn \
    --baseline data/nnue/nets/uho-ladder-v1-u1024-120sb.cenn \
    --games 200 --tc 10+0.1 --concurrency 3 --seed 20260722 \
    --candidate-engine engines/bin/probe-l2048 --baseline-engine engines/bin/probe-l1024 \
    --output "$d" > /dev/null 2>&1 || { say "gate FAILED to run"; exit 1; }
fi
python3 -c "import json;s=json.load(open('$d/summary.json'));print(f'[GATE uf2048 vs incumbent u1024-120sb EQUAL-TIME] RESULT: elo={s[\"candidate_elo\"]:.1f} se={s[\"candidate_elo_se\"]:.1f} score={s[\"candidate_score\"]:.3f}')" | tee -a "$R"
say "GATE2048 DONE"
