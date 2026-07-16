#!/usr/bin/env bash
# Stage-2 wave 2 (auto-research loop), gated on the 2048 equal-time gate:
#   1. s2b: second 300g h2h block for s2-1024-test80-120sb (pool to 600g)
#   2. s2-2048-test80-120sb: warm retrain the BEST stage-1 net (uf2048) on
#      test80, same stage-2 recipe; h2h vs uf2048-240sb (same width).
set -uo pipefail
cd "$(dirname "$0")/../.."

R=data/nnue/experiments/chain-v2-results.log
say() { echo "[s2w2 $(date +%H:%M:%S)] $*" | tee -a "$R"; }

say "waiting for the 2048 equal-time gate to release the box"
until grep -aqE "^\[gate2048 [0-9:]+\] GATE2048 DONE" "$R"; do sleep 120; done
sleep 30

d1=data/nnue/experiments/stage2-v1/s2-1024-test80-120sb
if [ ! -f "$d1/h2h-b/summary.json" ]; then
  say "s2b: second h2h block for s2-1024-test80-120sb"
  uv run python tools/nnue/eval_match.py \
    --candidate data/nnue/nets/stage2-v1-s2-1024-test80-120sb.cenn \
    --baseline data/nnue/nets/uho-full-v1-uf1024-240sb.cenn \
    --games 300 --nodes 5000 --concurrency 4 --seed 20260723 \
    --candidate-engine engines/bin/probe-l1024 --baseline-engine engines/bin/probe-l1024 \
    --output "$d1/h2h-b" > /dev/null 2>&1 || say "s2b screen FAILED"
fi
python3 - <<'EOF' | tee -a data/nnue/experiments/chain-v2-results.log
import json, math
try:
    a = json.load(open('data/nnue/experiments/stage2-v1/s2-1024-test80-120sb/h2h/summary.json'))
    b = json.load(open('data/nnue/experiments/stage2-v1/s2-1024-test80-120sb/h2h-b/summary.json'))
    w = a['candidate_wins']+b['candidate_wins']; l = a['candidate_losses']+b['candidate_losses']; d = a['draws']+b['draws']; g = w+l+d
    s = (w+0.5*d)/g
    elo = 400*math.log10(s/(1-s))
    var = (w*(1-s)**2 + l*(0-s)**2 + d*(0.5-s)**2)/(g-1)
    se = 400/math.log(10)*math.sqrt(var/g)/(s*(1-s))
    print(f'[s2-1024 POOLED {g}g vs uf1024-240sb] RESULT: elo={elo:.1f} se={se:.1f} score={s:.3f}')
except Exception as e:
    print(f'[s2 pool] FAILED: {e}')
EOF

name=s2-2048-test80-120sb
d=data/nnue/experiments/stage2-v1/$name
mkdir -p "$d"
python3 - <<'EOF'
import json, copy, os
base = json.load(open('data/nnue/experiments/stage2-v1/s2-1024-test80-120sb/config.json'))
cfg = copy.deepcopy(base)
cfg['net_id'] = 'stage2-v1-s2-2048-test80-120sb'
cfg['parent'] = 'uho-wave4-uf2048-240sb'
cfg['arch'] = '768x2x2048x1-screlu'
cfg['pack']['l1'] = 2048
cfg['engine_features'] = 'nnue-l1-2048'
cfg['notes'] = 'Stage-2 on the best stage-1 net: uf2048-240sb + test80, LR/2, WDL 0.25, div 1.45.'
ckpt = '/root/projects/learnrust/chessengine/data/nnue/experiments/stage2-v1/s2-2048-test80-120sb/ckpt'
os.makedirs(ckpt, exist_ok=True)
cfg['train']['checkpoint_dir'] = ckpt
env = cfg['train']['env']
env['NET0_HIDDEN'] = '2048'
env['NET0_INIT_CENN'] = 'data/nnue/nets/uho-wave4-uf2048-240sb.cenn'
json.dump(cfg, open('data/nnue/experiments/stage2-v1/s2-2048-test80-120sb/config.json', 'w'), indent=2)
EOF
if [ ! -f "data/nnue/nets/stage2-v1-$name.cenn" ]; then
  say "training $name (warm start from uf2048-240sb)"
  uv run python tools/nnue/train_run.py --config "$d/config.json" \
    --only preflight,train,verify,pack,sanity >> "$d/run.log" 2>&1 \
    || { say "$name train FAILED (see $d/run.log)"; exit 1; }
  say "$name packed OK"
fi
if [ ! -f "$d/h2h/summary.json" ]; then
  say "h2h screening $name vs uf2048-240sb (300g @5k)"
  uv run python tools/nnue/eval_match.py \
    --candidate data/nnue/nets/stage2-v1-$name.cenn \
    --baseline data/nnue/nets/uho-wave4-uf2048-240sb.cenn \
    --games 300 --nodes 5000 --concurrency 4 --seed 20260724 \
    --candidate-engine engines/bin/probe-l2048 --baseline-engine engines/bin/probe-l2048 \
    --output "$d/h2h" > /dev/null 2>&1 || { say "$name screen FAILED"; exit 1; }
fi
python3 -c "import json;s=json.load(open('$d/h2h/summary.json'));print(f'[$name vs uf2048-240sb] RESULT: elo={s[\"candidate_elo\"]:.1f} se={s[\"candidate_elo_se\"]:.1f} score={s[\"candidate_score\"]:.3f}')" | tee -a "$R"
say "S2W2 DONE"
