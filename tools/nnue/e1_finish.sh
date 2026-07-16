#!/usr/bin/env bash
# E1 finisher (docs/research/value-foundation-state.md): once the SF16 relabel
# completes AND the sf100-matrix frees the GPU, build the corpus from the
# relabeled shards, train with the exact net-0 recipe, and screen 300g @5k
# vs net-0. The corpus is fishpack positions + SF16 labels + fishpack results.
set -euo pipefail
cd "$(dirname "$0")/../.."

E=data/nnue/experiments/fishpack-sf16-relabel
M=data/nnue/campaigns/sf100-matrix-v1

echo "[e1 $(date +%H:%M:%S)] waiting for relabel COMPLETE + matrix DONE"
until grep -aq COMPLETE "$E/relabel.log" 2>/dev/null; do sleep 60; done
echo "[e1 $(date +%H:%M:%S)] labels ready: $(cat $E/labels/relabel_*.txt | wc -l) rows"
until [ -f "$M/DONE" ]; do sleep 60; done

echo "[e1 $(date +%H:%M:%S)] building corpus"
mkdir -p "$E/corpus-raw"
i=0
for f in "$E"/labels/relabel_*.txt; do
  ln -sf "$(realpath "$f")" "$E/corpus-raw/shard_$i.txt"; i=$((i+1))
done
BULLETFORMAT_CONVERT="tools/nnue/bf-convert/target/release/nnue_convert convert" \
BULLET_UTILS_SHUFFLE="tools/nnue/bullet_shuffle_wrap.sh" \
BULLET_UTILS_ILEAVE="tools/nnue/bullet_ileave_wrap.sh" \
bash tools/nnue/build_corpus.sh --raw "$E/corpus-raw" --out "$E/corpus" --seed 1 --require-convert

echo "[e1 $(date +%H:%M:%S)] training (exact net-0 recipe)"
python3 - <<'EOF'
import json
cfg = json.load(open('data/nnue/campaigns/sf100-matrix-v1/resolved-configs/sf100-matrix-v1-sf100-classic256.json'))
net_id = 'e1-fishpackpos-sf16labels-classic256'
cfg['net_id'] = net_id
cfg['notes'] = 'E1: fishpack positions + SF16 5000-node labels (gen_sf_data-faithful) + fishpack results; exact net-0 recipe (30 SB).'
cfg['corpus'] = {'id': 'e1-fishpackpos-sf16labels', 'raw_dir': None,
                 'train_bin': 'data/nnue/experiments/fishpack-sf16-relabel/corpus/train.bin'}
cfg['train']['checkpoint_dir'] = '/root/projects/learnrust/chessengine/data/nnue/experiments/fishpack-sf16-relabel/ckpt'
env = cfg['train']['env']
env['NET0_DATA'] = 'data/nnue/experiments/fishpack-sf16-relabel/corpus/train.bin'
env['NET0_FORMAT'] = 'bulletformat'
env['NET0_SUPERBATCHES'] = '30'
env['NET0_LR_END_SB'] = '30'
env['NET0_SAVE_RATE'] = '30'
json.dump(cfg, open('data/nnue/experiments/fishpack-sf16-relabel/e1-config.json', 'w'), indent=2)
EOF
uv run python tools/nnue/train_run.py --config "$E/e1-config.json" \
  --only preflight,train,verify,pack,sanity

echo "[e1 $(date +%H:%M:%S)] screening 300g @5k vs net-0"
uv run python tools/nnue/eval_match.py \
  --candidate data/nnue/nets/e1-fishpackpos-sf16labels-classic256.cenn \
  --baseline data/nnue/nets/net-0.cenn \
  --games 300 --nodes 5000 --concurrency 4 --seed 20260715 \
  --candidate-engine engines/bin/probe-default --baseline-engine engines/bin/probe-default \
  --output "$E/screen-n5k"
python3 -c "import json;s=json.load(open('$E/screen-n5k/summary.json'));print(f'[e1] RESULT: score={s[\"candidate_score\"]:.3f} elo={s[\"candidate_elo\"]:.1f} se={s[\"candidate_elo_se\"]:.1f}')"
touch "$E/DONE"
echo "[e1 $(date +%H:%M:%S)] E1 COMPLETE"
