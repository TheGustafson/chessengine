#!/usr/bin/env bash
# Stage-2 wave 3: farseerT76 hop (cleaner labels: Pearson 0.934 vs test80's
# 0.784) on both widths. GPU serial; h2h vs each warm-start parent.
#   s3-2048-farseer-120sb — warm from incumbent uf2048-240sb, div 1.9
#   s3-1024-farseer-120sb — warm from uf1024-240sb (test80 hop gave +19
#                           there; direct corpus-quality comparison)
set -uo pipefail
cd "$(dirname "$0")/../.."

R=data/nnue/experiments/chain-v2-results.log
FARSEER=data/corpora/official-stockfish-master-binpacks/main/raw/farseerT76.binpack
say() { echo "[s2w3 $(date +%H:%M:%S)] $*" | tee -a "$R"; }

python3 - <<'EOF'
import json, copy, os
base = json.load(open('data/nnue/experiments/stage2-v1/s2-1024-test80-120sb/config.json'))
FARSEER = 'data/corpora/official-stockfish-master-binpacks/main/raw/farseerT76.binpack'
cells = [
    ('s3-2048-farseer-120sb', 2048, 'uho-wave4-uf2048-240sb'),
    ('s3-1024-farseer-120sb', 1024, 'uho-full-v1-uf1024-240sb'),
]
for name, hidden, parent in cells:
    cfg = copy.deepcopy(base)
    cfg['net_id'] = f'stage2-v1-{name}'
    cfg['parent'] = parent
    cfg['arch'] = f'768x2x{hidden}x1-screlu'
    cfg['pack']['l1'] = hidden
    cfg['engine_features'] = f'nnue-l1-{hidden}'
    cfg['corpus'] = {'id': 'farseert76-full', 'raw_dir': None, 'train_bin': FARSEER}
    cfg['notes'] = f'Stage-2 farseer hop: {name}, warm from {parent}, LR/2, WDL 0.25, div 1.9.'
    ckpt = f'/root/projects/learnrust/chessengine/data/nnue/experiments/stage2-v1/{name}/ckpt'
    os.makedirs(ckpt, exist_ok=True)
    cfg['train']['checkpoint_dir'] = ckpt
    env = cfg['train']['env']
    env['NET0_DATA'] = FARSEER
    env['NET0_HIDDEN'] = str(hidden)
    env['NET0_INIT_CENN'] = f'data/nnue/nets/{parent}.cenn'
    env['SFBINPACK_SCORE_DIV'] = '1.9'
    json.dump(cfg, open(f'data/nnue/experiments/stage2-v1/{name}/config.json', 'w'), indent=2)
EOF

for spec in "s3-2048-farseer-120sb probe-l2048 uho-wave4-uf2048-240sb" "s3-1024-farseer-120sb probe-l1024 uho-full-v1-uf1024-240sb"; do
  set -- $spec; name=$1; eng=$2; parent=$3
  d=data/nnue/experiments/stage2-v1/$name
  if [ ! -f "data/nnue/nets/stage2-v1-$name.cenn" ]; then
    say "training $name"
    uv run python tools/nnue/train_run.py --config "$d/config.json" \
      --only preflight,train,verify,pack,sanity >> "$d/run.log" 2>&1 \
      || { say "$name train FAILED (see $d/run.log)"; continue; }
    say "$name packed OK"
  fi
  if [ ! -f "$d/h2h/summary.json" ]; then
    say "h2h screening $name vs $parent (300g @5k)"
    uv run python tools/nnue/eval_match.py \
      --candidate data/nnue/nets/stage2-v1-$name.cenn \
      --baseline data/nnue/nets/$parent.cenn \
      --games 300 --nodes 5000 --concurrency 4 --seed 20260725 \
      --candidate-engine engines/bin/$eng --baseline-engine engines/bin/$eng \
      --output "$d/h2h" > /dev/null 2>&1 || { say "$name screen FAILED"; continue; }
  fi
  python3 -c "import json;s=json.load(open('$d/h2h/summary.json'));print(f'[$name vs $parent] RESULT: elo={s[\"candidate_elo\"]:.1f} se={s[\"candidate_elo_se\"]:.1f} score={s[\"candidate_score\"]:.3f}')" | tee -a "$R"
done
say "S2W3 DONE"
