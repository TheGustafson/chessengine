#!/usr/bin/env python3
"""Learned-representation study: LEARNED king-context partitions on the
measured in-tree transition matrix, with DATA-SIDE occupancy balance.

The census script's annealer balances on in-tree ARRIVAL mass, which
over-weights endgame king activity. Conditioning value lives in the training
distribution, so this tool re-runs the min-flip search with the balance
constraint on DATA-side king occupancy (from the heldout pack's `ksq` /
`opp_ksq` fields — both perspectives, own-frame) while the flip objective
stays IN-TREE (the engine pays flips in search, not in the corpus). For each
partition it prints: in-tree flips/node, exact addition increment
(union-with-mirror minus mirror) and replacement delta (vs the d/e mirror),
data-side and arrival-side mass splits, and the board diagram.

Frames: records are stm-canonicalized by vertical flip only (files raw), so
`ksq` is the own king in own-frame and `opp_ksq ^ 56` is the enemy king in
ITS own-frame — the same color-relative frame as the KTRANS counter.

Usage:
  .venv/bin/python tools/updloc/updloc_partition.py <foldcount-dump> \
      [heldout.bin] [n-records]
"""

import sys
from pathlib import Path

import numpy as np

sys.path.insert(0, str(Path(__file__).resolve().parent))
sys.path.insert(0, str(Path(__file__).resolve().parents[2] / "jax"))
from updloc_census import anneal_bipartition, flip_rate, ktrans, parse  # noqa: E402
from archon_jax.spec import RECORD_DTYPE  # noqa: E402

ROOT = Path(__file__).resolve().parents[2]


def data_occupancy(heldout, n_rec):
    rec = np.fromfile(heldout, dtype=RECORD_DTYPE, count=n_rec)
    own = np.bincount(rec["ksq"], minlength=64).astype(float)
    enemy = np.bincount(rec["opp_ksq"] ^ 56, minlength=64).astype(float)
    return (own + enemy) / (own + enemy).sum(), len(rec)


def describe(name, p, t, n, occ, arrivals, mirror_rate):
    fr = flip_rate(t, p, n)
    mir = lambda f, g: ((f & 7) <= 3) != ((g & 7) <= 3)
    union = sum(t[f][g] for f in range(64) for g in range(64)
                if p[f] != p[g] or mir(f, g)) / n
    d_mass = min(sum(occ[s] for s in range(64) if p[s]),
                 sum(occ[s] for s in range(64) if not p[s]))
    a_tot = sum(arrivals)
    a_mass = min(sum(arrivals[s] for s in range(64) if p[s]),
                 sum(arrivals[s] for s in range(64) if not p[s])) / a_tot
    print(f"{name:26s} flips={fr:8.5f}  add={union - mirror_rate:8.5f}  "
          f"repl={fr - mirror_rate:+8.5f}  data-min-mass={d_mass:6.2%}  "
          f"arrival-min-mass={a_mass:6.2%}")
    return fr


def diagram(p):
    for rank in range(7, -1, -1):
        print("      " + "".join(".#"[p[rank * 8 + f]] for f in range(8))
              + f"   rank {rank + 1}")


def main():
    dump = sys.argv[1]
    heldout = sys.argv[2] if len(sys.argv) > 2 else str(
        ROOT / "data/nnue/heldout/uho-band0-5.bin")
    n_rec = int(sys.argv[3]) if len(sys.argv) > 3 else 1_000_000

    kv = parse(dump)
    n = kv["nodes"]
    t = ktrans(kv)
    arrivals = [sum(t[x][s] for x in range(64)) for s in range(64)]
    occ, used = data_occupancy(heldout, n_rec)
    mirror_rate = kv["kmove.mirror"] / n
    print(f"nodes={n:,}  heldout records={used:,}  "
          f"mirror flips/node={mirror_rate:.5f}")

    hand = {
        "mirror(file a-d)": [0 if (s & 7) <= 3 else 1 for s in range(64)],
        "rank1(back rank)": [0 if (s >> 3) == 0 else 1 for s in range(64)],
        "rank12(back two)": [0 if (s >> 3) <= 1 else 1 for s in range(64)],
        "rankhalf(own half)": [0 if (s >> 3) <= 3 else 1 for s in range(64)],
    }
    print("\n-- hand bits --")
    for name, p in hand.items():
        describe(name, p, t, n, occ, arrivals, mirror_rate)

    print("\n-- learned min-flip partitions (balance on DATA occupancy) --")
    occ_list = list(occ)
    for share in (0.10, 0.20, 0.30, 0.40, 0.48):
        best = None
        for seed in (1, 2, 3, 4):
            p, fr = anneal_bipartition(t, n, occ_list, share, seed=seed)
            if best is None or fr < best[1]:
                best = (p, fr)
        p, fr = best
        if sum(occ[s] for s in range(64) if p[s]) > 0.5:
            p = [1 - x for x in p]
        describe(f"learned(>={share:.0%} data mass)", p, t, n, occ, arrivals,
                 mirror_rate)
        diagram(p)


if __name__ == "__main__":
    main()
