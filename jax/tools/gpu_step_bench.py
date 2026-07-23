"""Phase-2 go/no-go: full-arch jit'd train-step throughput on the GPU.

Measures pos/s for the exact D47c recipe shape (batch 8192, full 22528+60720
feature space, policy head, bullet-AdamW) against bullet's ~210k pos/s on the
same GPU. Gate: >=0.8x (168k). Uses REAL heldout rows; 8 distinct batches
cycled to defeat cache-residency flattery. Note: this pipeline carries
pol_target_id (sparse) rather than bullet's dense 1858-target+mask uploads —
a designed improvement counted in its favor.

  cd jax && XLA_PYTHON_CLIENT_PREALLOCATE=false \
    uv run --no-project --with jax[cuda12] --with optax -- \
    python tools/gpu_step_bench.py [n_batches=8] [timed_steps=40]
"""

import os
import sys
import time
from pathlib import Path

import numpy as np

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))
os.environ.setdefault("XLA_PYTHON_CLIENT_PREALLOCATE", "false")

from archon_jax.features import map_batch  # noqa: E402
from archon_jax.model import init_params  # noqa: E402
from archon_jax.spec import RECORD_DTYPE, ArchSpec, TrainConfig  # noqa: E402
from archon_jax.train import make_update_step  # noqa: E402


def main(n_batches=8, timed_steps=40, batch_size=0):
    import jax

    dev = jax.devices()[0]
    print(f"device: {dev} ({dev.platform})")
    if dev.platform != "gpu":
        print("FATAL: not on GPU — aborting (bench is meaningless on CPU)")
        return 2

    arch, cfg = ArchSpec.full(), TrainConfig()
    if batch_size:
        cfg = type(cfg)(**{**cfg.__dict__, 'batch_size': int(batch_size)})
    dump = Path(__file__).resolve().parents[2] / "data/nnue/heldout/uho-band0-5.bin"
    recs = np.memmap(dump, dtype=RECORD_DTYPE, mode="r")
    need = n_batches * cfg.batch_size
    idx = np.linspace(0, len(recs) - 1, num=need, dtype=np.int64)
    rows = np.array(recs[idx])

    batches = []
    t0 = time.time()
    for i in range(n_batches):
        b = map_batch(rows[i * cfg.batch_size:(i + 1) * cfg.batch_size],
                      wdl=cfg.wdl, score_multiplier=cfg.score_multiplier,
                      scale=cfg.scale, policy=True)
        batches.append({k: np.asarray(v) for k, v in b.items()})
    prep = time.time() - t0
    print(f"host map_batch: {need / prep:,.0f} pos/s (prep only, single thread)")

    opt, step = make_update_step(arch, cfg)
    params = init_params(arch, seed=0x00D40FEA)
    opt_state = opt.init(params)

    # compile + warmup
    t0 = time.time()
    params, opt_state, m = step(params, opt_state, batches[0])
    jax.block_until_ready(m["loss"])
    print(f"compile+first step: {time.time() - t0:.1f}s  loss {float(m['loss']):.5f}")
    for i in range(1, 3):
        params, opt_state, m = step(params, opt_state, batches[i % n_batches])
    jax.block_until_ready(m["loss"])

    t0 = time.time()
    for i in range(timed_steps):
        params, opt_state, m = step(params, opt_state, batches[i % n_batches])
    jax.block_until_ready(m["loss"])
    dt = time.time() - t0
    pos_s = timed_steps * cfg.batch_size / dt
    print(f"timed: {timed_steps} steps in {dt:.2f}s = {pos_s:,.0f} pos/s "
          f"(bullet ref ~210k; 0.8x gate = 168k)")
    print(f"verdict: {'GO' if pos_s >= 168_000 else 'BELOW GATE'} "
          f"({pos_s / 210_000:.2f}x bullet)")
    return 0


if __name__ == "__main__":
    a = [int(x) for x in sys.argv[1:4]]
    raise SystemExit(main(*a))
