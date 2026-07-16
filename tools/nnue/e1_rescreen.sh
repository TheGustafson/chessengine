#!/usr/bin/env bash
# Clean rerun of E1's screen (original was contaminated: its fastchess was
# killed 16s in during an unrelated RAM-freeing pkill). Gates on the E1 size
# control finishing so CPUs are free.
set -euo pipefail
cd "$(dirname "$0")/../.."
until [ -f data/nnue/experiments/e1-size-control/DONE ]; do sleep 60; done
uv run python tools/nnue/eval_match.py \
  --candidate data/nnue/nets/e1-fishpackpos-sf16labels-classic256.cenn \
  --baseline data/nnue/nets/net-0.cenn \
  --games 300 --nodes 5000 --concurrency 3 --seed 20260715 \
  --candidate-engine engines/bin/probe-default --baseline-engine engines/bin/probe-default \
  --output data/nnue/experiments/fishpack-sf16-relabel/screen-n5k
python3 -c "import json;s=json.load(open('data/nnue/experiments/fishpack-sf16-relabel/screen-n5k/summary.json'));print(f'[e1-rescreen] RESULT: score={s[\"candidate_score\"]:.3f} elo={s[\"candidate_elo\"]:.1f} se={s[\"candidate_elo_se\"]:.1f}')"
touch data/nnue/experiments/fishpack-sf16-relabel/RESCREEN-DONE
