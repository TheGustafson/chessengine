#!/usr/bin/env bash
# Production track run 1 (docs/research/value-foundation-state.md §Production
# track): calibrated fishpack (score/3.5, H3b fix), WDL 0.3, 480 SB full
# decay. classic256 first (fast calibration read vs the -163/-328
# miscalibrated anchors), then classic1024 (the first real candidate,
# 200-game endpoint screen).
set -euo pipefail
cd "$(dirname "$0")/../.."

C=data/nnue/campaigns/prod-fishpack-cal-v1
bash tools/nnue/apply_bullet_patches.sh

for width in 256 1024; do
  id="prod-fishpack-cal-v1-classic$width"
  echo "[prod $(date +%H:%M:%S)] training $id"
  uv run python tools/nnue/train_run.py --config "$C/resolved-configs/$id.json" \
    --only preflight,corpus,train,verify,pack,parity,sanity
  games=80; cand_engine=engines/bin/probe-default
  [ "$width" = 1024 ] && games=200 && cand_engine=engines/bin/probe-l1024
  echo "[prod $(date +%H:%M:%S)] screening $id (${games}g @5k)"
  uv run python tools/nnue/eval_match.py \
    --candidate "data/nnue/nets/$id.cenn" --baseline data/nnue/nets/net-0.cenn \
    --games "$games" --nodes 5000 --concurrency 4 --seed 20260715 \
    --candidate-engine "$cand_engine" --baseline-engine engines/bin/probe-default \
    --output "$C/matches/$id-screen-n5k"
  python3 -c "import json;s=json.load(open('$C/matches/$id-screen-n5k/summary.json'));print(f'[prod] RESULT $id: score={s[\"candidate_score\"]:.3f} elo={s[\"candidate_elo\"]:.1f} se={s[\"candidate_elo_se\"]:.1f}')"
done

touch "$C/DONE"
echo "[prod $(date +%H:%M:%S)] PRODUCTION RUN 1 COMPLETE"
