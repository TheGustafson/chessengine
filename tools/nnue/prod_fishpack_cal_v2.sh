#!/usr/bin/env bash
# v2 isolation ladder vs the matrix-v2 anchor (fishpack classic256, div 1.0,
# no skip, 30 SB = -163 @5k): (a) divisor 2.0 alone; (b) divisor 2.0 + 75%
# deterministic random skip (official-SF-style stream decorrelation). Each
# 30 SB + 80g @5k screen. Self-gates on prod-fishpack-cal-v1 DONE (GPU).
set -euo pipefail
cd "$(dirname "$0")/../.."

V1=data/nnue/campaigns/prod-fishpack-cal-v1
V2=data/nnue/campaigns/prod-fishpack-cal-v2

echo "[v2 $(date +%H:%M:%S)] waiting for prod-v1 DONE"
until [ -f "$V1/DONE" ]; do sleep 60; done
bash tools/nnue/apply_bullet_patches.sh

for tag in div2-noskip div2-skip75; do
  id="prod-fishpack-cal-v2-$tag-classic256"
  echo "[v2 $(date +%H:%M:%S)] training $id"
  uv run python tools/nnue/train_run.py --config "$V2/resolved-configs/$id.json" \
    --only preflight,corpus,train,verify,pack,parity,sanity
  echo "[v2 $(date +%H:%M:%S)] screening $id (80g @5k)"
  uv run python tools/nnue/eval_match.py \
    --candidate "data/nnue/nets/$id.cenn" --baseline data/nnue/nets/net-0.cenn \
    --games 80 --nodes 5000 --concurrency 4 --seed 20260715 \
    --candidate-engine engines/bin/probe-default --baseline-engine engines/bin/probe-default \
    --output "$V2/matches/$id-screen-n5k"
  python3 -c "import json;s=json.load(open('$V2/matches/$id-screen-n5k/summary.json'));print(f'[v2] RESULT $id: score={s[\"candidate_score\"]:.3f} elo={s[\"candidate_elo\"]:.1f} se={s[\"candidate_elo_se\"]:.1f}')"
done

touch "$V2/DONE"
echo "[v2 $(date +%H:%M:%S)] V2 ISOLATION COMPLETE"
