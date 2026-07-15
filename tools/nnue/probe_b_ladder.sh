#!/usr/bin/env bash
# Probe B (docs/research/value-foundation-state.md): budget-ladder
# decomposition. Paired 80-game matches vs net-0 at 32/256/2k/20k fixed nodes
# for the three strongest failed nets. Curve shape separates "genuinely worse
# net" (uniform loss) from "search-consumer coupling" (strong at tiny budgets,
# degrading as pruning margins bite).
set -euo pipefail
cd "$(dirname "$0")/../.."

OUT=data/nnue/campaigns/probe-b-ladder-v1
mkdir -p "$OUT"
SEED=20260716

run_ladder() { # tag cenn engine
  local tag=$1 cenn=$2 engine=$3
  for nodes in 32 256 2000 20000; do
    local dir="$OUT/$tag-n$nodes"
    if [ -f "$dir/summary.json" ]; then
      echo "[probe-b] skip $tag n$nodes (done)"
      continue
    fi
    echo "[probe-b $(date +%H:%M:%S)] $tag @ $nodes nodes"
    uv run python tools/nnue/eval_match.py \
      --candidate "$cenn" --baseline data/nnue/nets/net-0.cenn \
      --games 80 --nodes "$nodes" --concurrency 2 --seed "$SEED" \
      --candidate-engine "$engine" --baseline-engine engines/bin/probe-default \
      --output "$dir"
    python3 -c "import json;s=json.load(open('$dir/summary.json'));print(f'[probe-b] RESULT $tag n$nodes: score={s[\"candidate_score\"]:.3f} elo={s[\"candidate_elo\"]:.1f} se={s[\"candidate_elo_se\"]:.1f}')"
  done
}

run_ladder lichess-classic1024 data/nnue/nets/value-bulk-matrix-v2-lichess-classic1024.cenn engines/bin/probe-l1024
run_ladder d40-lichess-mid-sb15 data/nnue/nets/value-bulk-d40-lichess-mid-v1-lichess-d40-full.cenn engines/bin/probe-sota
run_ladder binpack-classic256 data/nnue/nets/value-bulk-matrix-v2-binpack-classic256.cenn engines/bin/probe-default

touch "$OUT/DONE"
echo "[probe-b $(date +%H:%M:%S)] LADDER COMPLETE"
