#!/usr/bin/env bash
# v3: full sf-100 frame normalization of fishpack (div 2.0 + random skip 75%
# + draw-keep 32% -> sf-100 marginals). 30 SB + 80g @5k screen vs net-0.
# If it clearly beats the -163 anchor, immediately run the 480-SB version
# with the same env and a 200g screen (the first serious candidate from
# normalized foreign data). Self-gates on prod-fishpack-cal-v2 DONE.
set -euo pipefail
cd "$(dirname "$0")/../.."

V2=data/nnue/campaigns/prod-fishpack-cal-v2
V3=data/nnue/campaigns/prod-fishpack-cal-v3

echo "[v3 $(date +%H:%M:%S)] waiting for v2 DONE"
until [ -f "$V2/DONE" ]; do sleep 60; done
bash tools/nnue/apply_bullet_patches.sh

id="prod-fishpack-cal-v3-sf100frame-classic256"
echo "[v3 $(date +%H:%M:%S)] training $id (30 SB)"
uv run python tools/nnue/train_run.py --config "$V3/resolved-configs/$id.json" \
  --only preflight,corpus,train,verify,pack,parity,sanity
uv run python tools/nnue/eval_match.py \
  --candidate "data/nnue/nets/$id.cenn" --baseline data/nnue/nets/net-0.cenn \
  --games 80 --nodes 5000 --concurrency 4 --seed 20260715 \
  --candidate-engine engines/bin/probe-default --baseline-engine engines/bin/probe-default \
  --output "$V3/matches/$id-screen-n5k"
elo=$(python3 -c "import json;print(json.load(open('$V3/matches/$id-screen-n5k/summary.json'))['candidate_elo'])")
python3 -c "print(f'[v3] RESULT $id: elo={$elo:.1f}')"

if python3 -c "exit(0 if $elo >= -100 else 1)"; then
  long_id="prod-fishpack-cal-v3-sf100frame-long-classic256"
  echo "[v3 $(date +%H:%M:%S)] anchor beaten decisively; training 480-SB version $long_id"
  python3 - <<EOF
import json
cfg = json.load(open('$V3/resolved-configs/$id.json'))
cfg['net_id'] = '$long_id'
cfg['train']['checkpoint_dir'] = '/root/projects/learnrust/chessengine/data/nnue/campaigns/prod-fishpack-cal-v3/checkpoints/$long_id'
cfg['train']['env']['NET0_SUPERBATCHES'] = '480'
cfg['train']['env']['NET0_LR_END_SB'] = '480'
cfg['train']['env']['NET0_SAVE_RATE'] = '120'
json.dump(cfg, open('$V3/resolved-configs/$long_id.json','w'), indent=2)
EOF
  uv run python tools/nnue/train_run.py --config "$V3/resolved-configs/$long_id.json" \
    --only preflight,corpus,train,verify,pack,parity,sanity
  uv run python tools/nnue/eval_match.py \
    --candidate "data/nnue/nets/$long_id.cenn" --baseline data/nnue/nets/net-0.cenn \
    --games 200 --nodes 5000 --concurrency 4 --seed 20260715 \
    --candidate-engine engines/bin/probe-default --baseline-engine engines/bin/probe-default \
    --output "$V3/matches/$long_id-screen-n5k"
  python3 -c "import json;s=json.load(open('$V3/matches/$long_id-screen-n5k/summary.json'));print(f'[v3] RESULT $long_id: score={s[\"candidate_score\"]:.3f} elo={s[\"candidate_elo\"]:.1f} se={s[\"candidate_elo_se\"]:.1f}')"
else
  echo "[v3] RESULT: 30-SB normalized cell did not beat -100; long run not triggered"
fi

touch "$V3/DONE"
echo "[v3 $(date +%H:%M:%S)] V3 COMPLETE"
