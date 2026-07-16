#!/usr/bin/env bash
# Stage-2 wave 4: farseerT74 hop on the incumbent (gated on wave5's GPU).
#   s4-2048-t74-240sb — warm from s3-2048-farseer-120sb, T74 (20.1GB, 3x
#   T76), div 1.66 (audit farseert74-label-semantics-v1), 240 SB.
# h2h vs the incumbent (same width => transfers to equal time).
set -uo pipefail
cd "$(dirname "$0")/../.."

R=data/nnue/experiments/chain-v2-results.log
T74=data/corpora/official-stockfish-master-binpacks/main/raw/farseerT74.binpack
say() { echo "[s2w4 $(date +%H:%M:%S)] $*" | tee -a "$R"; }

say "waiting for wave5 to release the GPU"
until grep -aqE "^\[wave5 [0-9:]+\] WAVE5 DONE" "$R"; do sleep 120; done

name=s4-2048-t74-240sb
d=data/nnue/experiments/stage2-v1/$name
mkdir -p "$d"
python3 - <<'EOF'
import json, copy, os
base = json.load(open('data/nnue/experiments/stage2-v1/s3-2048-farseer-120sb/config.json'))
cfg = copy.deepcopy(base)
cfg['net_id'] = 'stage2-v1-s4-2048-t74-240sb'
cfg['parent'] = 'stage2-v1-s3-2048-farseer-120sb'
cfg['corpus'] = {'id': 'farseert74-full', 'raw_dir': None,
                 'train_bin': 'data/corpora/official-stockfish-master-binpacks/main/raw/farseerT74.binpack'}
cfg['notes'] = 'Stage-2 hop 2 on the incumbent: T74 (3x T76 data), div 1.66, LR/2, WDL 0.25, 240 SB.'
ckpt = '/root/projects/learnrust/chessengine/data/nnue/experiments/stage2-v1/s4-2048-t74-240sb/ckpt'
os.makedirs(ckpt, exist_ok=True)
cfg['train']['checkpoint_dir'] = ckpt
env = cfg['train']['env']
env['NET0_DATA'] = cfg['corpus']['train_bin']
env['NET0_INIT_CENN'] = 'data/nnue/nets/stage2-v1-s3-2048-farseer-120sb.cenn'
env['SFBINPACK_SCORE_DIV'] = '1.66'
env['NET0_SUPERBATCHES'] = '240'
env['NET0_LR_END_SB'] = '240'
env['NET0_SAVE_RATE'] = '240'
json.dump(cfg, open('data/nnue/experiments/stage2-v1/s4-2048-t74-240sb/config.json', 'w'), indent=2)
EOF

if [ ! -f "data/nnue/nets/stage2-v1-$name.cenn" ]; then
  say "training $name (warm from the s3 incumbent, ~50 min)"
  uv run python tools/nnue/train_run.py --config "$d/config.json" \
    --only preflight,train,verify,pack,sanity >> "$d/run.log" 2>&1 \
    || { say "$name train FAILED (see $d/run.log)"; exit 1; }
  say "$name packed OK"
fi
if [ ! -f "$d/h2h/summary.json" ]; then
  say "h2h screening $name vs the s3 incumbent (300g @5k)"
  uv run python tools/nnue/eval_match.py \
    --candidate data/nnue/nets/stage2-v1-$name.cenn \
    --baseline data/nnue/nets/stage2-v1-s3-2048-farseer-120sb.cenn \
    --games 300 --nodes 5000 --concurrency 4 --seed 20260727 \
    --candidate-engine engines/bin/probe-l2048 --baseline-engine engines/bin/probe-l2048 \
    --output "$d/h2h" > /dev/null 2>&1 || { say "$name screen FAILED"; exit 1; }
fi
python3 -c "import json;s=json.load(open('$d/h2h/summary.json'));print(f'[$name vs s3-incumbent] RESULT: elo={s[\"candidate_elo\"]:.1f} se={s[\"candidate_elo_se\"]:.1f} score={s[\"candidate_score\"]:.3f}')" | tee -a "$R"
say "S2W4 DONE"
