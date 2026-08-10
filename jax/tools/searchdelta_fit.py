#!/usr/bin/env python3
"""Fit search-vs-static disagreement from the frozen trunk, and null it.

THE QUESTION
    D57/D58 established that the trunk cannot predict the value head's own
    residual (r^2 = 0.0017 signed / 0.0266 magnitude on 1M heldout, unchanged
    when the residual switched to the score-only column). That is structural:
    a converged model's residual against its own features is empty.

    SEARCH-vs-STATIC disagreement is a different quantity -- the net is not
    trained on it, it is a property of the tree below the position. This
    script asks whether it is linearly extractable from hl1, the exact vector
    the disagreement head reads.

THE NULL, AND WHY IT IS NOT OPTIONAL
    |search - static| is large in sharp middlegames and small in dead endgames,
    and the trunk encodes PHASE trivially (piece count is nearly a linear
    readout of the accumulator). So a headline correlation here can be almost
    entirely "predicted the phase", which is worth nothing to a search that
    already knows the piece count for free.

    Two nulls are therefore reported next to every number:
      piececount  -- fit on popcount(occ) alone, the free feature
      residual    -- fit on hl1 AFTER regressing piece count out of the target
    Only the RESIDUAL column is signal a search consumer could not get for
    nothing. A trunk r that is high while the residual r is ~0 means the probe
    rediscovered material, and the honest verdict is negative.

WHY RIDGE AND NOT AN MLP
    A linear probe is a LOWER bound on what a 768->32->1 head could extract,
    it is closed-form (no LR, no seed, no early stopping to argue about), and
    it cannot memorise 48k rows with 768 features the way an MLP can. If ridge
    finds nothing, an MLP finding a lot would be a claim about capacity that
    needs its own generalisation test -- which is exactly the trap D57 fell
    into. Start with the instrument that cannot lie in that direction.
"""
import pathlib
import sys

import numpy as np

sys.path.insert(0, str(pathlib.Path(__file__).resolve().parents[1]))

from archon_jax.spec import RECORD_DTYPE  # noqa: E402


def static_basis(static: np.ndarray) -> np.ndarray:
    """A FLEXIBLE function of the static eval, not a linear one.

    `[static, |static|]` is too weak a control here. Search regresses extreme
    evals toward the mean, and in probability space that is SIGMOIDAL, so a
    linear-in-static null leaves curvature on the table that the trunk can
    then "explain" -- while telling the search nothing it did not already
    have. This basis (odd powers, |.|, and three tanh scales spanning the
    eval range) can absorb any smooth monotone shape, so whatever the trunk
    adds ON TOP of it is genuinely not a function of the eval alone.
    """
    s = static / 600.0                      # ~ the network_scale unit
    return np.stack([
        s, np.abs(s), s ** 2, s ** 3, np.abs(s) ** 3,
        np.tanh(s), np.tanh(s * 3.0), np.tanh(s / 3.0),
        np.abs(np.tanh(s)), np.sign(s),
    ], 1)


def ridge_fit(X: np.ndarray, y: np.ndarray, lam: float) -> np.ndarray:
    """Closed-form ridge with an intercept (column of ones prepended)."""
    n, d = X.shape
    Xb = np.concatenate([np.ones((n, 1), np.float64), X.astype(np.float64)], 1)
    A = Xb.T @ Xb
    A[np.arange(1, d + 1), np.arange(1, d + 1)] += lam      # never the bias
    return np.linalg.solve(A, Xb.T @ y.astype(np.float64))


def apply_fit(w: np.ndarray, X: np.ndarray) -> np.ndarray:
    Xb = np.concatenate([np.ones((len(X), 1), np.float64), X.astype(np.float64)], 1)
    return Xb @ w


def report(name: str, yhat: np.ndarray, y: np.ndarray) -> float:
    r = float(np.corrcoef(yhat, y)[0, 1]) if np.std(yhat) > 1e-12 else 0.0
    mse = float(np.mean((y - yhat) ** 2))
    null = float(np.mean((y - np.mean(y)) ** 2))
    print(f"  {name:<22} r {r:+.4f}  r^2 {r*r:.4f}   "
          f"MSE {mse:.1f} / null {null:.1f}  ({100*(mse-null)/null:+.2f}%)")
    return r


def main(bin_path: str, static_txt: str, search_txt: str, hl1_npy: str,
         lam: float = 100.0) -> int:
    static = np.loadtxt(static_txt, dtype=np.float64)
    search = np.loadtxt(search_txt, dtype=np.float64)
    hl1 = np.load(hl1_npy)
    n = min(len(static), len(search), len(hl1))
    static, search, hl1 = static[:n], search[:n], hl1[:n]

    recs = np.memmap(bin_path, dtype=RECORD_DTYPE, mode="r")
    pc = np.bitwise_count(np.ascontiguousarray(recs["occ"][:n])).astype(np.float64)

    delta = search - static
    mag = np.abs(delta)
    print(f"n {n:,}   |delta| mean {mag.mean():.1f} median {np.median(mag):.1f} "
          f"p90 {np.percentile(mag,90):.1f} max {mag.max():.1f}")
    print(f"corr(|delta|, piececount) {np.corrcoef(mag, pc)[0,1]:+.4f}")

    cut = int(n * 0.8)
    tr, te = slice(0, cut), slice(cut, n)
    print(f"train {cut:,}  test {n-cut:,}  ridge lambda {lam}")

    # ** THE CONTROL THAT DECIDES THIS. ** Search systematically pulls extreme
    # static evals back toward zero, so delta ~ -k*static -- and the trunk
    # predicts `static` PERFECTLY by construction, because hl1 IS the eval's
    # own input layer. Without this column, "trunk predicts delta" can be a
    # restatement of "trunk predicts its own eval", which buys a search
    # exactly nothing: it already has the static eval in hand at every node
    # where it would consult the head.
    print(f"\ncorr(delta, static)  {np.corrcoef(delta, static)[0,1]:+.4f}"
          f"   corr(|delta|, |static|) {np.corrcoef(mag, np.abs(static))[0,1]:+.4f}")
    # Mate scores (|score| near 32000) are a different population: there delta
    # is "search found a forced mate", not "the eval is unstable". Reported
    # both ways rather than silently clipped.
    keep = (np.abs(search) < 10000) & (np.abs(static) < 10000)
    print(f"non-mate rows: {keep.sum():,} / {n:,}")

    for label, y in (("|delta| (magnitude)", mag), ("delta (signed)", delta)):
        print(f"\n{label}")
        # 1. the free feature on its own
        wpc = ridge_fit(pc[tr, None], y[tr], lam)
        yhat_pc = apply_fit(wpc, pc[te, None])
        report("piececount null", yhat_pc, y[te])
        # 2. the trunk
        w = ridge_fit(hl1[tr], y[tr], lam)
        report("hl1 (trunk)", apply_fit(w, hl1[te]), y[te])
        # 3. the trunk on what piece count could NOT explain -- the only
        #    column that is signal a search does not already have for free.
        resid_tr = y[tr] - apply_fit(wpc, pc[tr, None])
        resid_te = y[te] - yhat_pc
        wr = ridge_fit(hl1[tr], resid_tr, lam)
        report("hl1 | piececount out", apply_fit(wr, hl1[te]), resid_te)
        # 4. the static eval alone -- the feature search ALREADY HAS.
        feat = np.stack([static, np.abs(static)], 1)
        wse = ridge_fit(feat[tr], y[tr], lam)
        yhat_se = apply_fit(wse, feat[te])
        report("static eval null", yhat_se, y[te])
        # 5. the trunk on what the static eval could NOT explain. THIS is the
        #    only number that means "the head knows something the search does
        #    not already have for free".
        rtr = y[tr] - apply_fit(wse, feat[tr])
        rte = y[te] - yhat_se
        wsr = ridge_fit(hl1[tr], rtr, lam)
        report("hl1 | static out", apply_fit(wsr, hl1[te]), rte)

    # Same two columns on the NON-MATE population.
    print("\n=== non-mate rows only ===")
    idx = np.nonzero(keep)[0]
    c2 = int(len(idx) * 0.8)
    tr2, te2 = idx[:c2], idx[c2:]
    for label, y in (("|delta| (magnitude)", mag), ("delta (signed)", delta)):
        print(f"\n{label}")
        feat = static_basis(static)
        wse = ridge_fit(feat[tr2], y[tr2], lam)
        yhat_se = apply_fit(wse, feat[te2])
        report("static eval null", yhat_se, y[te2])
        w = ridge_fit(hl1[tr2], y[tr2], lam)
        report("hl1 (trunk)", apply_fit(w, hl1[te2]), y[te2])
        rtr = y[tr2] - apply_fit(wse, feat[tr2])
        rte = y[te2] - yhat_se
        wsr = ridge_fit(hl1[tr2], rtr, lam)
        report("hl1 | static out", apply_fit(wsr, hl1[te2]), rte)
    return 0


if __name__ == "__main__":
    if len(sys.argv) < 5:
        print(__doc__)
        raise SystemExit(2)
    raise SystemExit(main(*sys.argv[1:5],
                          lam=float(sys.argv[5]) if len(sys.argv) > 5 else 100.0))
