"""Loader benchmark: numpy structured-dtype reader + vectorized extraction.

Reports pos/s for each pipeline stage over real corpus data, single-thread and
with the threaded prefetch loader, against the incumbent reference point
(Rust sfbinpack streaming + feature mapping: ~200k pos/s at 6 threads).

Run:
  cd jax && uv run --no-project --with numpy -- python tools/bench_loader.py \
      <corpus.bin> [batch_size] [n_batches]
"""

import sys
import time
from pathlib import Path

import numpy as np

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))

from archon_jax.data import BulletBin, make_batches  # noqa: E402
from archon_jax.features import DecodedBatch, halfka_batch, map_batch, threat_batch  # noqa: E402
from archon_jax.spec import TrainConfig  # noqa: E402


def bench(name, fn, n_pos, reps=3):
    times = []
    for _ in range(reps):
        t0 = time.perf_counter()
        fn()
        times.append(time.perf_counter() - t0)
    best = min(times)
    print(f"{name:42s} {n_pos / best:12,.0f} pos/s   ({best * 1e3:8.1f} ms)")
    return n_pos / best


def main(corpus, batch_size=8192, n_batches=8):
    bin_ = BulletBin(corpus)
    print(f"corpus: {corpus} ({len(bin_):,} records)")
    total = batch_size * n_batches
    slabs = [bin_.slice(i * batch_size, batch_size) for i in range(n_batches)]

    # Stage timings (single thread)
    bench("read+decode (occ/pcs -> piece lists)",
          lambda: [DecodedBatch(s) for s in slabs], total)
    decs = [DecodedBatch(s) for s in slabs]
    bench("halfka indices (both perspectives)",
          lambda: [halfka_batch(d) for d in decs], total)
    bench("threat indices (both perspectives)",
          lambda: [threat_batch(d) for d in decs], total)
    r = bench("FULL map_batch (halfka+threats+targets)",
              lambda: [map_batch(s, wdl=0.3, score_multiplier=1.0, scale=400.0,
                                 policy=True) for s in slabs], total)
    bench("halfka-only map_batch (no threats)",
          lambda: [map_batch(s, wdl=0.3, score_multiplier=1.0, scale=400.0,
                             policy=True, threats=False) for s in slabs], total)

    # Threaded loader end-to-end
    for workers in (2, 4, 6):
        cfg = TrainConfig(batch_size=batch_size)
        n = 24
        t0 = time.perf_counter()
        cnt = 0
        for b in make_batches(bin_, cfg, policy=True, workers=workers,
                              max_batches=n, seed=1):
            cnt += len(b["target"])
        dt = time.perf_counter() - t0
        print(f"threaded loader end-to-end ({workers} workers)   "
              f"{cnt / dt:12,.0f} pos/s")

    print(f"\nreference: Rust sfbinpack stream+map ~200,000 pos/s (6 threads)")
    print(f"single-thread full map = {r:,.0f} pos/s")


if __name__ == "__main__":
    main(sys.argv[1], *(int(a) for a in sys.argv[2:4]))
