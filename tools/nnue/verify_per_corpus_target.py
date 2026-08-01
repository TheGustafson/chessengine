#!/usr/bin/env python3
"""VERIFY the corpus <-> `per_corpus_target` index mapping. Do not assume it.

`d49_1280.json` carries twelve `(a, b)` draw-model pairs and twelve corpora, and
nothing in the file says which goes with which. `data.py:176-183` consumes them
positionally (`bins = [b for b, _ in mix]`, `_off[src]` with `src` the mix index,
plus a hard length check), so the CODE says the mapping is `mix[i] <-> pct[i]`.
This script confirms that against the DATA: refit each corpus's own draw model
and check that the identity permutation is the assignment that matches.

The model is the one `features.py:644-651` applies:
    P(win) = sig((s-a)/b),  P(loss) = sig((-s-a)/b),  P(draw) = 1 - Pw - Pl
fitted by maximum likelihood on (score, result), which is what
`fit_target_transform.py` does -- reimplemented here on a numpy-only
coarse-to-fine grid because scipy is not installed in the project venv and a
grid is entirely adequate to *identify* a stored pair.

Usage:
  .venv/bin/python tools/nnue/verify_per_corpus_target.py --sample 1000000
"""
from __future__ import annotations

import argparse
import json
import os
from pathlib import Path

import numpy as np

ROOT = Path(__file__).resolve().parents[2]
MAIN = Path(os.environ.get("ARCHON_MAIN", "/home/nick/projects/chessengine"))

RECORD_DTYPE = np.dtype([
    ("occ", "<u8"), ("pcs", "u1", (16,)), ("score", "<i2"), ("result", "u1"),
    ("ksq", "u1"), ("opp_ksq", "u1"), ("extra", "u1", (3,)),
])


def sig(x):
    return 1.0 / (1.0 + np.exp(-np.clip(x, -60, 60)))


def nll(a, b, s, res):
    pw = sig((s - a) / b)
    pl = sig((-s - a) / b)
    p = np.where(res == 2, pw, np.where(res == 0, pl, 1.0 - pw - pl))
    return -float(np.mean(np.log(np.clip(p, 1e-6, 1.0))))


def fit_grid(s, res, a_rng=(50.0, 400.0), b_rng=(10.0, 200.0), steps=25, passes=4):
    a_lo, a_hi = a_rng
    b_lo, b_hi = b_rng
    best = (np.nan, np.nan, np.inf)
    for _ in range(passes):
        for a in np.linspace(a_lo, a_hi, steps):
            for b in np.linspace(b_lo, b_hi, steps):
                v = nll(a, b, s, res)
                if v < best[2]:
                    best = (float(a), float(b), v)
        da = (a_hi - a_lo) / steps
        db = (b_hi - b_lo) / steps
        a_lo, a_hi = best[0] - 2 * da, best[0] + 2 * da
        b_lo, b_hi = max(5.0, best[1] - 2 * db), best[1] + 2 * db
    return best


def main() -> int:
    ap = argparse.ArgumentParser()
    spec_default = ROOT / "tools/nnue/runs/d49_1280.json"
    if not spec_default.exists():
        spec_default = MAIN / "tools/nnue/runs/d49_1280.json"
    ap.add_argument("--spec", default=str(spec_default))
    ap.add_argument("--sample", type=int, default=1_000_000,
                    help="records per corpus; the LEADING window, matching "
                         "fit_target_transform.py's r['score'][:n]")
    ap.add_argument("--out", default="")
    a = ap.parse_args()

    spec = json.load(open(a.spec))
    mix, pct = spec["mix"], spec["per_corpus_target"]
    print(f"{len(mix)} corpora, {len(pct)} target pairs\n")

    fits = []
    for i, (path, _w) in enumerate(mix):
        r = np.memmap(MAIN / path, dtype=RECORD_DTYPE, mode="r")
        n = min(len(r), a.sample)
        s = np.asarray(r["score"][:n], dtype=np.float64)
        res = np.asarray(r["result"][:n], dtype=np.int8)
        # fit_target_transform.py fits on a random HALF and this reproduction
        # uses the same split so a match is a match on like terms.
        idx = np.random.default_rng(0).permutation(n)[: n // 2]
        fa, fb, v = fit_grid(s[idx], res[idx])
        fits.append((fa, fb))
        sa, sb = pct[i]
        da = abs(fa - sa) / sa * 100
        db = abs(fb - sb) / sb * 100
        print(f"[{i:2d}] {Path(path).name.replace('.shuf.bin',''):24s} "
              f"n={n:>9,}  fitted a={fa:7.1f} b={fb:6.1f}   "
              f"stored a={sa:7.1f} b={sb:6.1f}   "
              f"delta {da:5.1f}% / {db:5.1f}%", flush=True)
        del r

    # Assignment check: for every stored pair, which fitted corpus is nearest?
    # Distance is relative so `a` (~150-250) and `b` (~30-120) count equally.
    #
    # Nearest-neighbour alone is NOT a sound verdict: two stored pairs can be
    # byte-identical (d49 slot 11 is a deliberate copy of slot 0 -- hardneg is
    # 0.013% of the mix and was never fitted, docs/research/d48y-wdl-fix-plan.md
    # line 234), and a duplicate makes one of them lose its own slot by a tie.
    # So the positional claim is tested DIRECTLY -- does corpus i's own refit
    # reproduce stored[i]? -- and duplicates are reported as duplicates.
    TOL = 0.05
    print("\nnearest-fitted-corpus for each stored pair:")
    matched, dup = 0, []
    for j, (sa, sb) in enumerate(pct):
        d = [abs(fa - sa) / sa + abs(fb - sb) / sb for fa, fb in fits]
        k = int(np.argmin(d))
        own = d[j]
        is_dup = any(pct[z] == pct[j] for z in range(len(pct)) if z != j)
        if is_dup:
            dup.append(j)
        ok = own <= TOL
        matched += ok
        flag = ("" if ok else
                ("   <-- stored value is a DUPLICATE of another slot; this "
                 "corpus was never fitted" if is_dup else "   <-- MISMATCH"))
        print(f"  stored[{j:2d}] a={sa:7.1f} b={sb:6.1f}  nearest corpus[{k:2d}] "
              f"{Path(mix[k][0]).name.replace('.shuf.bin',''):24s}  "
              f"own-slot distance {own:.3f}{flag}")
    print(f"\n{matched}/{len(pct)} corpora reproduce their OWN slot within "
          f"{TOL:.0%} relative distance.")
    if dup:
        print(f"duplicate stored pairs at slots {dup} -- a duplicate cannot be "
              f"attributed by value, so those slots are confirmed by the "
              f"consumer code (data.py:176-183) only.")
    perm_ok = matched + len(dup) >= len(pct)
    print("VERDICT:", "POSITIONAL mapping mix[i] <-> per_corpus_target[i] CONFIRMED"
          if perm_ok else "positional mapping NOT confirmed")

    if a.out:
        json.dump({"fits": fits, "stored": pct,
                   "corpora": [p for p, _ in mix], "identity_ok": bool(perm_ok)},
                  open(a.out, "w"), indent=1)
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
