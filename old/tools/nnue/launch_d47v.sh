#!/usr/bin/env bash
# Launch D47V (768·32·32·8 + skip, teacher-distilled MTL bet) from the
# hold-best d47s EMA. EXECUTIVE ORDER 2026-07-26 ~21:1x: V runs right after
# the hold leg; U is the fallback ("go back bigger if we need to").
#   bash tools/nnue/launch_d47v.sh <hold-best-ema-raw.bin>
set -euo pipefail
cd "$(dirname "$0")/../.."
BEST="${1:?hold-best ema raw path}"
[ -s "$BEST" ] || { echo "no such warm: $BEST"; exit 1; }

WARM=data/nnue/experiments/d47v-warm.bin
JAX_PLATFORMS=cpu uv run --no-project --with numpy==2.5.1 --with jax==0.11.0 \
  --with optax==0.2.8 python tools/nnue/assemble_d47u_warm.py \
  "$BEST" "$WARM" --l2 32 --l3 32 --slice-hl 768

python3 - "$BEST" <<'PYEOF'
import json, sys
p = "tools/nnue/runs/d47v_mtl768.json"
c = json.load(open(p))
c["warm"] = "data/nnue/experiments/d47v-warm.bin"
c["teacher_warm"] = sys.argv[1]
c["heldout_expect"] = 0.0
json.dump(c, open(p, "w"), indent=1)
print(f"injected: warm={c['warm']} teacher={sys.argv[1]}")
PYEOF

mv /tmp/d47v_train.log /tmp/d47v_train.log.old 2>/dev/null || true
systemd-run --user --unit=archon-d47v --working-directory="$PWD/jax" \
  -p MemoryHigh=5G -p MemoryMax=7G -p MemorySwapMax=0 --collect \
  -E PYTHONUNBUFFERED=1 -E ARCHON_JAX_ALLOW_GPU=1 -E ARCHON_JAX_FT_IMPL=pallas \
  /bin/sh -c 'exec uv run --no-project --with numpy==2.5.1 --with "jax[cuda12]==0.11.0" --with optax==0.2.8 python tools/pump_train.py ../tools/nnue/runs/d47v_mtl768.json >> /tmp/d47v_train.log 2>&1'
systemd-run --user --unit=archon-tb-shim-d47v --working-directory="$PWD" --collect \
  /bin/sh -c 'exec uv run --no-project --with tensorboardX python tools/nnue/tb_shim.py /tmp/d47v_train.log data/nnue/tb/d47v-mtl768/train' 2>/dev/null || true
echo "D47V launched (unit archon-d47v); log /tmp/d47v_train.log; tb armed."
echo "GATES: in-frame heldout falling; Spearman within 0.003 of hold-best;"
echo "policy top-1 vs 26.1; WAC; DECISIVE: equal-time vs hold-best >= 0."
