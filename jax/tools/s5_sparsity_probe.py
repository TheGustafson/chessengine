"""S5: sparsity-lambda probe — 3sb warm runs at candidate lambdas.

Warm-starts the stage-3 config (all features) from d47c-stage2-sb240 and
trains 3 superbatches per lambda, printing per-sb value / ft_l1 /
ft_density. Pick: the largest lambda whose VALUE component stays inside
the lambda=0 run's envelope while density falls. ~6 min per lambda at
216k pos/s.

  cd jax && PYTHONUNBUFFERED=1 ARCHON_JAX_ALLOW_GPU=1 ARCHON_JAX_FT_IMPL=pallas \
    XLA_PYTHON_CLIENT_PREALLOCATE=false uv run --no-project \
    --with "jax[cuda12]==0.11.0" --with optax==0.2.8 --with numpy==2.5.1 -- \
    python tools/s5_sparsity_probe.py 0.0 0.001 0.003
"""

from __future__ import annotations

import dataclasses
import sys
import time
from pathlib import Path

import numpy as np

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))

from archon_jax import checkpoint as ck                      # noqa: E402
from archon_jax.data import BulletBin, make_mixed_batches    # noqa: E402
from archon_jax.spec import ArchSpec, TrainConfig            # noqa: E402
from archon_jax.train import make_update_step                # noqa: E402

ROOT = Path(__file__).resolve().parents[2]
WARM = ROOT / "data/nnue/experiments/d47c-stage2/ckpt/d47c-stage2-sb240-raw.bin"
MIX = [(ROOT / "data/nnue/jaxcorpora/uho-band75-100.shuf.bin", 0.40),
       (ROOT / "data/nnue/jaxcorpora/fishpack32.shuf.bin", 0.35),
       (ROOT / "data/nnue/jaxcorpora/farseerT74-band75-100.shuf.bin", 0.25)]
PROBE_SB = 3


def probe(lam: float) -> None:
    import jax

    arch = dataclasses.replace(ArchSpec.full(), factorize=True, psqt=True)
    cfg = dataclasses.replace(
        TrainConfig(), lr=0.00025, lr_end_superbatch=331, superbatches=PROBE_SB,
        pol_lambda=0.00125, ft_sparsity_lambda=lam)
    params = {k: jax.numpy.asarray(v) for k, v in ck.load_warm(WARM, arch).items()}
    mix = [(BulletBin(str(p)), w) for p, w in MIX]
    batches = make_mixed_batches(mix, cfg, policy=True, seed=0x00D47D03, workers=5)
    opt, step = make_update_step(arch, cfg)
    opt_state = opt.init(params)
    bps = cfg.batches_per_superbatch
    running: dict = {}
    t0 = time.time()
    for i, batch in enumerate(batches):
        params, opt_state, metrics = step(params, opt_state, batch)
        for k, v in metrics.items():
            running[k] = running.get(k, 0.0) + float(v)
        if i % bps == bps - 1:
            sb = i // bps + 1
            c = {k: v / bps for k, v in running.items()}
            print(f"probe lam={lam} sb {sb} | value {c.get('value', 0):.6f} | "
                  f"ft_l1 {c.get('ft_l1', 0):.6f} | ft_density {c.get('ft_density', 0):.6f} | "
                  f"{(i + 1) * cfg.batch_size / (time.time() - t0):,.0f} pos/s", flush=True)
            running = {}
        if i + 1 >= bps * PROBE_SB:
            break


if __name__ == "__main__":
    for lam_s in sys.argv[1:] or ["0.0", "0.001", "0.003"]:
        probe(float(lam_s))
    print("S5 PROBE EXIT 0", flush=True)
