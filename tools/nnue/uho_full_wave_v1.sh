#!/usr/bin/env bash
# Full-UHO scale wave (auto-research loop): once the u1024 equal-time gate
# frees the box, register the 40.3GB corpus, then train+screen, serial:
#   uf512-120sb   — width 512, ~1 pass of the full file (3B presentations)
#   uf1024-240sb  — width 1024, ~2 passes (6B presentations)
# e3 recipe otherwise (div 1.9, skip75, WDL 0.3). Screens: 300g @5k conc 4.
set -uo pipefail
cd "$(dirname "$0")/../.."

R=data/nnue/experiments/chain-v2-results.log
UHO=data/corpora/official-stockfish-master-binpacks/main/raw/nodes5000pv2_UHO.binpack
say() { echo "[ufull $(date +%H:%M:%S)] $*" | tee -a "$R"; }

say "waiting for u1024 equal-time gate to finish"
until [ -f data/nnue/experiments/uho-ladder-v1/u1024-120sb/gate-tc10/summary.json ]; do sleep 120; done

say "registering full corpus in ledger (sha256 of 40GB, ~minutes)"
python3 tools/ledger/ledger.py add-data --id uho-stage1-full \
  --path "$UHO" --source sf-datagen --format sfbinpack \
  --gen-recipe '{"origin":"official-stockfish nodes5000pv2_UHO.binpack","fetch":"tools/nnue/fetch_official_stockfish_binpack.py --file nodes5000pv2_UHO.binpack --full","verified":"38546 chunks, 0 ragged bytes"}' \
  --notes "Full official stage-1 UHO corpus (40.3GB). Supersedes uho-stage1-head10gib for scale runs." \
  || say "ledger add-data failed (continuing; maybe already registered)"

python3 - <<'EOF'
import json, copy, os
base = json.load(open('data/nnue/experiments/uho-transplant/e3-config.json'))
UHO = 'data/corpora/official-stockfish-master-binpacks/main/raw/nodes5000pv2_UHO.binpack'
for name, hidden, sb in [('uf512-120sb', 512, 120), ('uf1024-240sb', 1024, 240)]:
    cfg = copy.deepcopy(base)
    cfg['net_id'] = f'uho-full-v1-{name}'
    cfg['arch'] = f'768x2x{hidden}x1-screlu'
    cfg['pack']['l1'] = hidden
    cfg['engine_features'] = f'nnue-l1-{hidden}'
    cfg['corpus'] = {'id': 'uho-stage1-full', 'raw_dir': None, 'train_bin': UHO}
    cfg['notes'] = f'Full-UHO scale wave: {name}; e3 recipe (div 1.9, skip75, WDL 0.3).'
    ckpt = f'/root/projects/learnrust/chessengine/data/nnue/experiments/uho-full-v1/{name}/ckpt'
    os.makedirs(ckpt, exist_ok=True)
    cfg['train']['checkpoint_dir'] = ckpt
    env = cfg['train']['env']
    env['NET0_DATA'] = UHO
    env['NET0_HIDDEN'] = str(hidden)
    env['NET0_SUPERBATCHES'] = str(sb)
    env['NET0_LR_END_SB'] = str(sb)
    env['NET0_SAVE_RATE'] = str(sb)
    json.dump(cfg, open(f'data/nnue/experiments/uho-full-v1/{name}/config.json', 'w'), indent=2)
EOF

for spec in "uf512-120sb archon-classic512" "uf1024-240sb probe-l1024"; do
  set -- $spec; name=$1; eng=$2
  d=data/nnue/experiments/uho-full-v1/$name
  if [ ! -f "data/nnue/nets/uho-full-v1-$name.cenn" ]; then
    say "training $name"
    uv run python tools/nnue/train_run.py --config "$d/config.json" \
      --only preflight,train,verify,pack,sanity >> "$d/run.log" 2>&1 \
      || { say "$name train FAILED (see $d/run.log)"; continue; }
    say "$name packed OK"
  fi
  if [ ! -f "$d/screen-n5k/summary.json" ]; then
    say "screening $name (300g @5k)"
    uv run python tools/nnue/eval_match.py \
      --candidate data/nnue/nets/uho-full-v1-$name.cenn \
      --baseline data/nnue/nets/net-0.cenn \
      --games 300 --nodes 5000 --concurrency 4 --seed 20260715 \
      --candidate-engine engines/bin/$eng --baseline-engine engines/bin/probe-default \
      --output "$d/screen-n5k" > /dev/null 2>&1 || { say "$name screen FAILED"; continue; }
  fi
  python3 -c "import json;s=json.load(open('$d/screen-n5k/summary.json'));print(f'[$name] RESULT: elo={s[\"candidate_elo\"]:.1f} se={s[\"candidate_elo_se\"]:.1f} score={s[\"candidate_score\"]:.3f}')" | tee -a "$R"
done
say "FULL-UHO WAVE COMPLETE"
