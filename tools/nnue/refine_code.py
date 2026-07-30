"""Build context-REFINED threat codes (richer-representation study, Tier-1).

A refined code is a (THREAT_DIMS, 2) int map: (tuple, region-bit) -> row.
Two modes:

  refine  — the deployable-shaped candidate: CONSTANT row count (60,720).
            Split rows are funded by the dead-row budget: tuples that never
            fire on the training mix NOR heldout are merged into per-attacker
            scrap rows, and their freed rows become the second (bit=1) copy of
            the split tuples. Split tuples are chosen by the gradient probe's
            debiased split gain (region_probe.py) where measured, then by the
            joint census's min-cell count — both artifacts are inputs, so the
            build is deterministic and auditable.
  oracle  — the upper-bound arm: rows grow to THREAT_DIMS + K; the top-K
            tuples by census mass (min-cell >= threshold) get a dedicated
            second row, everything else is identity. Trainer-only (a >65,535
            row space does not fit the engine's u16 TRM table; it exists to
            measure how much value king-conditioned threat semantics carry at
            this finetune budget, not to ship).

The same refined .npy serves the candidate arm (threat_region = the real bit)
and the control arm (threat_region = "null"): identical structure, identical
init, content-free conditioning — the null that isolates the bit's CONTENT
from the extra capacity and optimizer effects of splitting.

Usage:
  python refine_code.py refine <bit> [--splits N]   -> refine-<bit>-r60720.npy
  python refine_code.py oracle <bit> [--k K]        -> oracle-<bit>-r<rows>.npy
"""
import hashlib
import json
import sys
from pathlib import Path

import numpy as np

sys.path.insert(0, str(Path(__file__).resolve().parents[2] / "jax"))
from archon_jax.features import (ALL_PIECES, NUM_VALID_TARGETS,  # noqa: E402
                                 THREAT_CUM_OFFSET, THREAT_PIECE_OFFSET)
from archon_jax.spec import THREAT_DIMS                          # noqa: E402

ROOT = Path("/home/nick/projects/chessengine")
SCREENS = ROOT / "data/nnue/nets/screens/recode"
CENSUS = SCREENS / "region-census-mix8M.npz"
PROBE = SCREENS / "region-probe.npz"
MIN_CELL = 30          # a split needs at least this many firings in EACH cell


def block_ids() -> np.ndarray:
    """(THREAT_DIMS,) attacker-block id per tuple (12 blocks, SF piece-code
    enumeration order) — scrap rows are per block so a dead tuple's fallback
    value stays within its own attacker family (blunder-tail guard)."""
    out = np.zeros(THREAT_DIMS, dtype=np.int64)
    for i, code in enumerate(ALL_PIECES):
        lo = int(THREAT_CUM_OFFSET[code])
        hi = lo + int(NUM_VALID_TARGETS[code] * THREAT_PIECE_OFFSET[code])
        out[lo:hi] = i
    return out


def load_census(bit: str):
    jc = np.load(CENSUS)
    bits = [b.decode() if isinstance(b, bytes) else str(b) for b in jc["bits"]]
    b = bits.index(bit)
    joint = jc["joint"][b] + jc["joint_held"][b]         # (T, 2)
    dead = ~(jc["fired_mix"] | jc["fired_held"])
    return joint, dead


def probe_scores(bit: str) -> tuple[np.ndarray, np.ndarray]:
    """(tuple_ids, scores) for tuples the probe measured on `bit`, restricted
    to positive debiased gain, descending."""
    if not PROBE.exists():
        return np.zeros(0, np.int64), np.zeros(0)
    pr = np.load(PROBE)
    bits = [b.decode() if isinstance(b, bytes) else str(b) for b in pr["bits"]]
    b = bits.index(bit)
    K = int(pr["K"])
    top = pr["top_th"]
    n = pr["N_th"][b].reshape(K, 2).astype(np.float64)
    g = pr["G_th"][b].reshape(K, 2, -1).astype(np.float64)
    s2 = pr["S2_th"][b].reshape(K, 2)
    n0, n1 = n[:, 0], n[:, 1]
    ok = (n0 >= 20) & (n1 >= 20)
    mu0 = g[:, 0] / np.maximum(n0, 1)[:, None]
    mu1 = g[:, 1] / np.maximum(n1, 1)[:, None]
    d2 = np.sum((mu0 - mu1) ** 2, axis=1)
    var = ((s2[:, 0] + s2[:, 1]) - n0 * np.sum(mu0 ** 2, axis=1)
           - n1 * np.sum(mu1 ** 2, axis=1)) / np.maximum(n0 + n1 - 2, 1)
    deb = d2 - var * (1 / np.maximum(n0, 1) + 1 / np.maximum(n1, 1))
    score = n0 * n1 / np.maximum(n0 + n1, 1) * deb
    keep = ok & (score > 0)
    order = np.argsort(score[keep])[::-1]
    return top[keep][order], score[keep][order]


def build_refine(bit: str, n_splits: int | None):
    joint, dead = load_census(bit)
    blocks = block_ids()
    dead_ids = np.nonzero(dead)[0]

    # scrap rows: one dead row per attacker block that has any dead tuples
    scrap_row = {}
    for b in np.unique(blocks[dead_ids]):
        scrap_row[int(b)] = int(dead_ids[blocks[dead_ids] == b][0])
    freed = np.array([t for t in dead_ids if t not in scrap_row.values()],
                     dtype=np.int64)

    # split ranking: probe first, census min-cell fallback
    splittable = (np.minimum(joint[:, 0], joint[:, 1]) >= MIN_CELL) & ~dead
    p_ids, _ = probe_scores(bit)
    p_ids = p_ids[splittable[p_ids]]
    rest_mask = splittable.copy()
    rest_mask[p_ids] = False
    rest = np.nonzero(rest_mask)[0]
    rest = rest[np.argsort(np.minimum(joint[rest, 0], joint[rest, 1]))[::-1]]
    ranked = np.concatenate([p_ids, rest])

    budget = len(freed) if n_splits is None else min(n_splits, len(freed))
    split = ranked[:budget]

    code = np.repeat(np.arange(THREAT_DIMS, dtype=np.int64)[:, None], 2, axis=1)
    for t in dead_ids:
        code[t, :] = scrap_row[int(blocks[t])]
    code[split, 1] = freed[:len(split)]

    total = joint.sum()
    info = {
        "mode": "refine", "bit": bit, "rows": THREAT_DIMS,
        "dead": int(dead.sum()), "scrap_rows": len(scrap_row),
        "splits": int(len(split)), "probe_ranked": int(len(p_ids)),
        "split_mass": float(joint[split].sum() / total),
        "min_cell": MIN_CELL,
    }
    return code, info


def build_oracle(bit: str, k: int):
    joint, dead = load_census(bit)
    splittable = np.minimum(joint[:, 0], joint[:, 1]) >= MIN_CELL
    order = np.argsort(joint.sum(axis=1))[::-1]
    top = order[splittable[order]][:k]
    code = np.repeat(np.arange(THREAT_DIMS, dtype=np.int64)[:, None], 2, axis=1)
    code[top, 1] = THREAT_DIMS + np.arange(len(top))
    info = {
        "mode": "oracle", "bit": bit, "rows": THREAT_DIMS + len(top),
        "splits": int(len(top)),
        "split_mass": float(joint[top].sum() / joint.sum()),
        "min_cell": MIN_CELL,
    }
    return code, info


def main() -> int:
    mode, bit = sys.argv[1], sys.argv[2]
    if mode == "refine":
        n = None
        if "--splits" in sys.argv:
            n = int(sys.argv[sys.argv.index("--splits") + 1])
        code, info = build_refine(bit, n)
    elif mode == "oracle":
        k = 16384
        if "--k" in sys.argv:
            k = int(sys.argv[sys.argv.index("--k") + 1])
        code, info = build_oracle(bit, k)
    else:
        raise SystemExit(f"unknown mode {mode}")

    out = SCREENS / f"{info['mode']}-{bit}-r{info['rows']}.npy"
    np.save(out, code.astype(np.int64))
    info["sha256"] = hashlib.sha256(out.read_bytes()).hexdigest()
    info["path"] = str(out)
    man = out.with_suffix(".json")
    man.write_text(json.dumps(info, indent=2) + "\n")
    print(json.dumps(info, indent=2))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
