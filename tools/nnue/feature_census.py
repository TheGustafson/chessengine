"""Tier-0 pre-screen prototype: active-feature frequency census.

Streams N shuffled bulletformat records through the trainer's own vectorized
feature mapper (jax/archon_jax/features.py -- index-exact vs the engine) and
bincounts HalfKA + FullThreats active indices, stm perspective.

Outputs: mean active counts, never-fired row counts, and coverage curves
(fraction of total fire mass carried by the top-K most frequent rows).

CPU-only numpy; chunked; safe beside the live GPU trainer.
"""
import sys
import numpy as np

sys.path.insert(0, "/home/nick/projects/chessengine/jax")
from archon_jax.data import BulletBin  # noqa: E402
from archon_jax.features import map_batch  # noqa: E402
from archon_jax.spec import HALFKA_DIMS, THREAT_DIMS  # noqa: E402

PACK = "/home/nick/projects/chessengine/data/nnue/jaxcorpora/uho-band75-100.shuf.bin"
N = 4_000_000
CHUNK = 65_536

bin_ = BulletBin(PACK)
n = min(N, len(bin_))
hk_counts = np.zeros(HALFKA_DIMS, dtype=np.int64)
th_counts = np.zeros(THREAT_DIMS, dtype=np.int64)
hk_active = 0
th_active = 0

done = 0
while done < n:
    recs = bin_.slice(done, min(CHUNK, n - done))
    out = map_batch(recs, wdl=0.0, score_multiplier=1.0, scale=400.0,
                    policy=False, threats=True)
    hk = out["hk_stm"].ravel()
    th = out["th_stm"].ravel()
    hk = hk[hk >= 0]
    th = th[th >= 0]
    hk_counts += np.bincount(hk, minlength=HALFKA_DIMS)
    th_counts += np.bincount(th, minlength=THREAT_DIMS)
    hk_active += hk.size
    th_active += th.size
    done += len(recs)

print(f"pack={PACK}")
print(f"positions={done}")
print(f"halfka: mean active/pos = {hk_active/done:.2f}")
print(f"threat: mean active/pos = {th_active/done:.2f}")


def coverage(counts, name, ks):
    order = np.sort(counts)[::-1]
    total = order.sum()
    csum = np.cumsum(order)
    nz = int((counts > 0).sum())
    print(f"{name}: rows ever fired = {nz}/{len(counts)} "
          f"({100.0*nz/len(counts):.1f}%)")
    for k in ks:
        if k <= len(order):
            print(f"  top-{k:>6} rows carry {100.0*csum[k-1]/total:6.2f}% of fire mass")
    for q in (0.90, 0.99, 0.999):
        kq = int(np.searchsorted(csum, q * total) + 1)
        print(f"  {100*q:.1f}% of mass in top-{kq} rows")


coverage(hk_counts, "halfka", [256, 1024, 4096, 8192, 16384])
coverage(th_counts, "threat", [1024, 2048, 4096, 8192, 16384, 32768])

np.savez("/tmp/claude-1000/-home-nick-projects-chessengine/48ee83a2-f33c-4ab2-8e12-b12ca79a3d94/scratchpad/census_counts.npz",
         hk=hk_counts, th=th_counts, n=done)
