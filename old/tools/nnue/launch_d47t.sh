#!/usr/bin/env bash
# launch_d47t.sh — inject warm/heldout from the finished D47S run and launch
# the D47T policy resync under systemd (memory-capped, tb-shimmed).
# Run AFTER d47s_pipeline.sh calibrate + the q1 screen pass verdict.
set -euo pipefail
cd "$(dirname "$0")/../.."

WARM=data/nnue/experiments/d47s-v15/d47s-v15-sb240-ema-raw.bin
[ -s "$WARM" ] || WARM=$(ls -t data/nnue/experiments/d47s-v15/d47s-v15-*-ema-raw.bin | head -1)
# heldout_expect = D47S's FINAL old-frame heldout EMA (the gate function is
# transform-invariant by design — pump_train computes it with TrainConfig
# defaults, so the value in the log is directly comparable).
EXPECT=$(grep "heldout sb" /tmp/d47s_v15_train.log | tail -1 | grep -oP 'ema \K[0-9.]+')
[ -n "$EXPECT" ] || { echo "no heldout in the d47s log"; exit 1; }

CFG=tools/nnue/runs/d47t_policy_resync.json
python3 - "$CFG" "$WARM" "$EXPECT" <<'EOF'
import json, sys
p, warm, expect = sys.argv[1:4]
c = json.load(open(p))
c["warm"] = warm
c["heldout_expect"] = float(expect)
json.dump(c, open(p, "w"), indent=2)
print(f"injected: warm={warm} heldout_expect={expect}")
EOF

mv /tmp/d47t_train.log /tmp/d47t_train.log.old 2>/dev/null || true
systemd-run --user --unit=archon-d47t --working-directory="$PWD/jax" \
  -p MemoryHigh=5G -p MemoryMax=7G -p MemorySwapMax=0 --collect \
  -E PYTHONUNBUFFERED=1 -E ARCHON_JAX_ALLOW_GPU=1 -E ARCHON_JAX_FT_IMPL=pallas \
  /bin/sh -c 'exec uv run --no-project --with numpy==2.5.1 --with "jax[cuda12]==0.11.0" --with optax==0.2.8 python tools/pump_train.py ../tools/nnue/runs/d47t_policy_resync.json >> /tmp/d47t_train.log 2>&1'
# tb telemetry armed at launch (standing rule)
systemd-run --user --unit=archon-tb-shim-d47t --working-directory="$PWD" --collect \
  /bin/sh -c 'exec uv run --no-project --with tensorboardX python tools/nnue/tb_shim.py /tmp/d47t_train.log data/nnue/tb/d47t-presync/train' 2>/dev/null || true
echo "D47T launched (unit archon-d47t); watch /tmp/d47t_train.log"
echo "INVARIANT: heldout must reproduce $EXPECT EXACTLY at every checkpoint (frozen value path)."
