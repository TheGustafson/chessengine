#!/usr/bin/env bash
# Stage-2 warm retrain, attempt 1 (official recipe shape, docs
# stockfish_reference/05-nnue-pytorch-training-recipes.md):
#   s2-1024-test80-120sb — warm start from uf1024-240sb (NET0_INIT_CENN),
#   rich Lc0-derived data (test80 2022-08), LR halved (0.0005), WDL 0.25
#   (~official stage-2 lambda 0.75), SCORE_DIV 1.45 (test80 units 2.18x
#   SF18-UCI -> proven SF16 frame ~1.5x; audit test80-label-semantics-v1).
# Gated on wave4 finishing (GPU + target/release ownership).
# Screen: h2h vs the warm-start parent uf1024-240sb (same width => NPS-free).
set -uo pipefail
cd "$(dirname "$0")/../.."

R=data/nnue/experiments/chain-v2-results.log
say() { echo "[stage2 $(date +%H:%M:%S)] $*" | tee -a "$R"; }

say "waiting for wave4 to finish"
until grep -aqE "^\[wave4 [0-9:]+\] WAVE4 COMPLETE" "$R"; do sleep 120; done

name=s2-1024-test80-120sb
d=data/nnue/experiments/stage2-v1/$name
mkdir -p "$d"
python3 - <<'EOF'
import json, copy, os
base = json.load(open('data/nnue/experiments/uho-full-v1/uf1024-240sb/config.json'))
cfg = copy.deepcopy(base)
cfg['net_id'] = 'stage2-v1-s2-1024-test80-120sb'
cfg['parent'] = 'uho-full-v1-uf1024-240sb'
cfg['corpus'] = {'id': 'test80-2022-08-lc0-min', 'raw_dir': None,
                 'train_bin': 'data/corpora/official-stockfish-master-binpacks/main/raw/test80-2022-08-aug-16tb7p.v6-dd.min.binpack'}
cfg['notes'] = ('Stage-2 warm retrain: uf1024-240sb + test80 rich data, LR/2, '
                'WDL 0.25, div 1.45. Official two-stage recipe shape.')
ckpt = '/root/projects/learnrust/chessengine/data/nnue/experiments/stage2-v1/s2-1024-test80-120sb/ckpt'
os.makedirs(ckpt, exist_ok=True)
cfg['train']['checkpoint_dir'] = ckpt
env = cfg['train']['env']
env['NET0_DATA'] = cfg['corpus']['train_bin']
env['NET0_INIT_CENN'] = 'data/nnue/nets/uho-full-v1-uf1024-240sb.cenn'
env['NET0_LR'] = '0.0005'
env['NET0_WDL'] = '0.25'
env['SFBINPACK_SCORE_DIV'] = '1.45'
env['NET0_SUPERBATCHES'] = '120'
env['NET0_LR_END_SB'] = '120'
env['NET0_SAVE_RATE'] = '120'
json.dump(cfg, open(f'data/nnue/experiments/stage2-v1/s2-1024-test80-120sb/config.json', 'w'), indent=2)
EOF

if [ ! -f "data/nnue/nets/stage2-v1-$name.cenn" ]; then
  say "training $name (warm start from uf1024-240sb)"
  uv run python tools/nnue/train_run.py --config "$d/config.json" \
    --only preflight,train,verify,pack,sanity >> "$d/run.log" 2>&1 \
    || { say "$name train FAILED (see $d/run.log)"; exit 1; }
  say "$name packed OK"
fi
if [ ! -f "$d/h2h/summary.json" ]; then
  say "h2h screening $name vs uf1024-240sb (300g @5k)"
  uv run python tools/nnue/eval_match.py \
    --candidate data/nnue/nets/stage2-v1-$name.cenn \
    --baseline data/nnue/nets/uho-full-v1-uf1024-240sb.cenn \
    --games 300 --nodes 5000 --concurrency 4 --seed 20260721 \
    --candidate-engine engines/bin/probe-l1024 --baseline-engine engines/bin/probe-l1024 \
    --output "$d/h2h" > /dev/null 2>&1 || { say "$name screen FAILED"; exit 1; }
fi
python3 -c "import json;s=json.load(open('$d/h2h/summary.json'));print(f'[$name vs uf1024-240sb] RESULT: elo={s[\"candidate_elo\"]:.1f} se={s[\"candidate_elo_se\"]:.1f} score={s[\"candidate_score\"]:.3f}')" | tee -a "$R"
say "STAGE2 ATTEMPT1 DONE"
