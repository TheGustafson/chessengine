#!/usr/bin/env bash
# Rerun of the killed d40 manual screen; last in the CPU queue (RAM-heavy:
# three D40 engine processes). Gates on E1's rescreen.
set -euo pipefail
cd "$(dirname "$0")/../.."
until [ -f data/nnue/experiments/fishpack-sf16-relabel/RESCREEN-DONE ]; do sleep 60; done
uv run python tools/nnue/eval_match.py \
  --candidate data/nnue/nets/sf100-matrix-v1-sf100-d40-full.cenn \
  --baseline data/nnue/nets/net-0.cenn \
  --games 300 --nodes 5000 --concurrency 2 --seed 20260715 \
  --candidate-engine engines/bin/probe-sota --baseline-engine engines/bin/probe-default \
  --output data/nnue/campaigns/sf100-matrix-v1/matches/d40-manual-screen-n5k
python3 -c "import json;s=json.load(open('data/nnue/campaigns/sf100-matrix-v1/matches/d40-manual-screen-n5k/summary.json'));print(f'[d40-rescreen] RESULT: score={s[\"candidate_score\"]:.3f} elo={s[\"candidate_elo\"]:.1f} se={s[\"candidate_elo_se\"]:.1f}')"
touch data/nnue/campaigns/sf100-matrix-v1/D40-RESCREEN-DONE
