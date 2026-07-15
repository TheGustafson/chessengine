#!/usr/bin/env bash
# Probe A2 (H3b target-scale test — docs/research/value-foundation-state.md):
# after the Probe A campaign frees the GPU, apply the env-gated
# sfbinpack score-divisor patch and rerun the exact matrix-v2 fishpack
# classic256 anchor recipe (30 SB, full decay) at score divisors 3.03
# (measured OLS) and 3.50 (SF pawn normalization), then screen each 80g @5k
# vs net-0 on the prebuilt default binary. Anchor to beat: -163.
set -euo pipefail
cd "$(dirname "$0")/../.."

A1=data/nnue/campaigns/value-scale-probe-a-v1
A2=data/nnue/campaigns/value-scale-probe-a2-v1

echo "[probe-a2 $(date +%H:%M:%S)] waiting for probe-a campaign DONE (GPU + toolchain must be free)"
until [ -f "$A1/DONE" ]; do sleep 60; done
echo "[probe-a2 $(date +%H:%M:%S)] campaign done; applying bullet patches"
bash tools/nnue/apply_bullet_patches.sh

for tag in div303 div350; do
  id="value-scale-probe-a2-$tag-binpack-classic256"
  cfg="$A2/resolved-configs/$id.json"
  echo "[probe-a2 $(date +%H:%M:%S)] training $id"
  uv run python tools/nnue/train_run.py --config "$cfg" \
    --only preflight,corpus,train,verify,pack,parity,sanity
  echo "[probe-a2 $(date +%H:%M:%S)] screening $id"
  uv run python tools/nnue/eval_match.py \
    --candidate "data/nnue/nets/$id.cenn" --baseline data/nnue/nets/net-0.cenn \
    --games 80 --nodes 5000 --concurrency 4 --seed 20260715 \
    --candidate-engine engines/bin/probe-default --baseline-engine engines/bin/probe-default \
    --output "$A2/matches/$id-screen-n5k"
  python3 -c "import json;s=json.load(open('$A2/matches/$id-screen-n5k/summary.json'));print(f'[probe-a2] RESULT $id: score={s[\"candidate_score\"]:.3f} elo={s[\"candidate_elo\"]:.1f} se={s[\"candidate_elo_se\"]:.1f}')"
done

touch "$A2/DONE"
echo "[probe-a2 $(date +%H:%M:%S)] PROBE A2 COMPLETE"
