#!/usr/bin/env python3
"""Fit the eval->target transform to OUR training data, rigorously.

The trainer maps a label score to a [0,1] target and blends it with the game
result:  pt = transform(score)*lambda + result*(1-lambda).  If transform() is
miscalibrated the two halves of the target disagree on every sample.

We currently use  sigmoid(score/400)  with no draw model. SF uses a
draw-aware double sigmoid, 0.5*(1 + sig((s-a)/b) - sig((-s-a)/b)), with
a=270,b=340 for the label side. This script asks what OUR data wants.

Rigour (each addresses a way the first pass could have lied):
  * MAXIMUM LIKELIHOOD on raw (score,result) samples — not least squares on
    bin midpoints, which weights a 99k bin the same as a 629k bin.
  * HELD-OUT split — fit on half, score on the other half.
  * BOOTSTRAP CIs on the fitted parameters.
  * MULTIPLE CORPORA — a transform that only fits one corpus is a corpus
    artifact, not a property of our data.
  * CONSTRAINED vs UNCONSTRAINED — with offset<0 the WDL decomposition is
    invalid (P(win)+P(loss)>1). Report both: unconstrained is legitimate for
    an expected-score TARGET; constrained is required for a real WDL head.
  * Reports LOG-LOSS (the quantity that actually matters for a probabilistic
    target) alongside curve MAE.

    python3 tools/nnue/fit_target_transform.py --corpus data/nnue/jaxcorpora/uho-band75-100.shuf.bin
"""
from __future__ import annotations

import argparse
import sys
from pathlib import Path

import numpy as np

ROOT = Path(__file__).resolve().parents[2]
sys.path.insert(0, str(ROOT / "jax"))


def sig(x):
    return 1.0 / (1.0 + np.exp(-np.clip(x, -60, 60)))


def wdl_probs(s, a, b):
    """P(win), P(draw), P(loss) under the double-sigmoid model."""
    pw = sig((s - a) / b)
    pl = sig((-s - a) / b)
    return pw, 1.0 - pw - pl, pl


def expected_score(s, a, b):
    pw, _, pl = wdl_probs(s, a, b)
    return 0.5 * (1.0 + pw - pl)


def neg_loglik(params, s, res, floor=1e-6):
    """3-way multinomial NLL. res: 0=loss 1=draw 2=win (stm)."""
    a, b = params
    if b <= 1.0:
        return 1e18
    pw, pd, pl = wdl_probs(s, a, b)
    p = np.where(res == 2, pw, np.where(res == 0, pl, pd))
    return -np.mean(np.log(np.clip(p, floor, 1.0)))


def brier(params, s, res):
    """Expected-score squared error — the quantity the TARGET feeds."""
    e = expected_score(s, *params)
    actual = np.where(res == 2, 1.0, np.where(res == 0, 0.0, 0.5))
    return float(np.mean((e - actual) ** 2))


def fit(s, res, constrained: bool):
    from scipy.optimize import minimize
    best, bestv = None, np.inf
    for a0 in (-300, -150, 0, 150, 270, 500):
        for b0 in (60, 120, 200, 340, 500):
            if constrained and a0 < 0:
                continue
            bnds = [(0, 3000) if constrained else (-3000, 3000), (5, 5000)]
            r = minimize(neg_loglik, [a0, b0], args=(s, res), method="L-BFGS-B", bounds=bnds)
            if r.fun < bestv:
                best, bestv = r.x, r.fun
    return best, bestv


def report(name, params, s, res, single=False):
    if single:
        k = params
        e = sig(s / k)
        actual = np.where(res == 2, 1.0, np.where(res == 0, 0.0, 0.5))
        # a single sigmoid has no draw model; score it only on expected score
        print(f"  {name:44s} brier {np.mean((e-actual)**2):.5f}   nll    n/a")
        return
    print(f"  {name:44s} brier {brier(params, s, res):.5f}   "
          f"nll {neg_loglik(params, s, res):.5f}")


def main() -> int:
    ap = argparse.ArgumentParser()
    ap.add_argument("--corpus", action="append", required=True)
    ap.add_argument("--sample", type=int, default=2_000_000)
    ap.add_argument("--bootstrap", type=int, default=30)
    a = ap.parse_args()
    from archon_jax.spec import RECORD_DTYPE

    for path in a.corpus:
        p = ROOT / path if not Path(path).is_absolute() else Path(path)
        r = np.memmap(p, dtype=RECORD_DTYPE, mode="r")
        n = min(len(r), a.sample)
        s = np.asarray(r["score"][:n], dtype=np.float64)
        res = np.asarray(r["result"][:n], dtype=np.int8)
        rng = np.random.default_rng(0)
        idx = rng.permutation(n)
        tr, te = idx[: n // 2], idx[n // 2:]
        print(f"\n{'='*78}\n{Path(path).name}   n={n:,}  "
              f"(W {int((res==2).sum()):,} D {int((res==1).sum()):,} L {int((res==0).sum()):,})")

        u_par, _ = fit(s[tr], res[tr], constrained=False)
        c_par, _ = fit(s[tr], res[tr], constrained=True)
        # best single sigmoid on the same training half, by brier
        from scipy.optimize import minimize_scalar
        ks = minimize_scalar(
            lambda k: np.mean((sig(s[tr] / max(k, 1e-3))
                               - np.where(res[tr] == 2, 1.0, np.where(res[tr] == 0, 0.0, 0.5))) ** 2),
            bounds=(20, 2000), method="bounded").x

        print("  --- scored on the HELD-OUT half ---")
        report("CURRENT: single sigmoid, scale 400", 400.0, s[te], res[te], single=True)
        report(f"best single sigmoid, scale {ks:6.1f}", ks, s[te], res[te], single=True)
        report("SF label transform (offset 270, scale 380)", (270.0, 380.0), s[te], res[te])
        report("SF net transform  (offset 270, scale 340)", (270.0, 340.0), s[te], res[te])
        report(f"FITTED unconstrained (a {u_par[0]:7.1f}, b {u_par[1]:6.1f})", u_par, s[te], res[te])
        report(f"FITTED constrained   (a {c_par[0]:7.1f}, b {c_par[1]:6.1f})", c_par, s[te], res[te])

        if a.bootstrap:
            boots = []
            for i in range(a.bootstrap):
                bs = np.random.default_rng(i).integers(0, len(tr), size=min(len(tr), 200_000))
                boots.append(fit(s[tr][bs], res[tr][bs], constrained=False)[0])
            boots = np.array(boots)
            lo, hi = np.percentile(boots, [2.5, 97.5], axis=0)
            print(f"  bootstrap 95% CI: a [{lo[0]:.1f}, {hi[0]:.1f}]   b [{lo[1]:.1f}, {hi[1]:.1f}]")

        pw, pd, pl = wdl_probs(np.array([0.0]), *u_par)
        print(f"  sanity @score=0 (unconstrained): P(w) {pw[0]:.3f} P(d) {pd[0]:.3f} "
              f"P(l) {pl[0]:.3f}  {'<-- INVALID (P(d)<0)' if pd[0] < 0 else ''}")
        pw, pd, pl = wdl_probs(np.array([0.0]), *c_par)
        print(f"  sanity @score=0 (constrained):   P(w) {pw[0]:.3f} P(d) {pd[0]:.3f} P(l) {pl[0]:.3f}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
