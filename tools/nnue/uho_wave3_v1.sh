#!/usr/bin/env bash
# Wave 3 (auto-research loop): width-1024 axes on the full 40.3GB UHO file.
#   uf1024-480sb   — schedule doubling (12B presentations, ~4 passes)
#   uf1024-240sb-wdl0 — official stage-1 lambda: pure score (WDL 0.0)
# Screens: 300g @5k HEAD-TO-HEAD vs uf1024-240sb (same width/NPS => clean).
set -uo pipefail
cd "$(dirname "$0")/../.."

R=data/nnue/experiments/chain-v2-results.log
UHO=data/corpora/official-stockfish-master-binpacks/main/raw/nodes5000pv2_UHO.binpack
say() { echo "[wave3 $(date +%H:%M:%S)] $*" | tee -a "$R"; }

python3 - <<'EOF'
import json, copy, os
base = json.load(open('data/nnue/experiments/uho-full-v1/uf1024-240sb/config.json'))
cells = [('uf1024-480sb', {'NET0_SUPERBATCHES': '480', 'NET0_LR_END_SB': '480', 'NET0_SAVE_RATE': '480'}),
         ('uf1024-240sb-wdl0', {'NET0_WDL': '0.0'})]
for name, envops in cells:
    cfg = copy.deepcopy(base)
    cfg['net_id'] = f'uho-wave3-{name}'
    cfg['notes'] = f'Wave 3 {name}: width-1024 axis probe on full UHO.'
    ckpt = f'/root/projects/learnrust/chessengine/data/nnue/experiments/uho-wave3/{name}/ckpt'
    os.makedirs(ckpt, exist_ok=True)
    cfg['train']['checkpoint_dir'] = ckpt
    cfg['train']['env'].update(envops)
    json.dump(cfg, open(f'data/nnue/experiments/uho-wave3/{name}/config.json', 'w'), indent=2)
EOF

for name in uf1024-480sb uf1024-240sb-wdl0; do
  d=data/nnue/experiments/uho-wave3/$name
  if [ ! -f "data/nnue/nets/uho-wave3-$name.cenn" ]; then
    say "training $name"
    uv run python tools/nnue/train_run.py --config "$d/config.json" \
      --only preflight,train,verify,pack,sanity >> "$d/run.log" 2>&1 \
      || { say "$name train FAILED (see $d/run.log)"; continue; }
    say "$name packed OK"
  fi
  if [ ! -f "$d/h2h/summary.json" ]; then
    say "h2h screening $name vs uf1024-240sb (300g @5k)"
    uv run python tools/nnue/eval_match.py \
      --candidate data/nnue/nets/uho-wave3-$name.cenn \
      --baseline data/nnue/nets/uho-full-v1-uf1024-240sb.cenn \
      --games 300 --nodes 5000 --concurrency 4 --seed 20260718 \
      --candidate-engine engines/bin/probe-l1024 --baseline-engine engines/bin/probe-l1024 \
      --output "$d/h2h" > /dev/null 2>&1 || { say "$name screen FAILED"; continue; }
  fi
  python3 -c "import json;s=json.load(open('$d/h2h/summary.json'));print(f'[$name vs uf1024-240sb] RESULT: elo={s[\"candidate_elo\"]:.1f} se={s[\"candidate_elo_se\"]:.1f} score={s[\"candidate_score\"]:.3f}')" | tee -a "$R"
done
say "WAVE3 COMPLETE"
