"""The honest castling decode test: AUC on the PLACEMENT-CONSISTENT subset.

Unconditionally, castling rights are proxy-decodable from placement (K/R on
home squares correlates with rights). The L2 claim is about the residual:
restrict evaluation to rows where own K sits on e1 AND own R on h1
(cast_consist=1, computed from record bytes = exactly what the net sees) and
ask whether any probe can still tell rights from no-rights. Structural
expectation: AUC ~= 0.5 up to whatever correlated placement signal (e.g.
early-game piece patterns) remains — which is itself informative, and the
shuffled-label null calibrates it.

  .venv/bin/python tools/nnue/reprgap/castle_conditional.py
"""
from __future__ import annotations

import sys
from pathlib import Path

import numpy as np

ROOT = Path(__file__).resolve().parents[3]
sys.path.insert(0, str(ROOT / "tools/nnue/reprgap"))
sys.path.insert(0, str(ROOT / "jax"))

import probes  # noqa: E402
from probes import (DIR, build_X, layout, make_rf, ridge_w,  # noqa: E402
                    stream_shards, target_columns)

TRAIN, EVAL = "tsvfilt_tr", "tsvfilt_ev"
TARGETS = ["castle_own_k", "castle_any", "rule50_hi"]


def main() -> int:
    gz = np.load(DIR / f"gram-{TRAIN}.npz", allow_pickle=True)
    G = gz["G"]
    mom = {k[4:]: gz[k] for k in gz.files if k.startswith("mom_")}
    tcols_tr = target_columns(TRAIN)
    lay = layout(tcols_tr)
    names = lay["names"]
    R, rb = make_rf(2560 + 1280)
    tcols_ev = target_columns(EVAL)
    shuf_ev = {k: v[np.random.default_rng(99).permutation(len(v))]
               for k, v in tcols_ev.items()}

    def cols_of(*blocks):
        idx = []
        for b in blocks:
            o, w = lay[b]
            idx.extend(range(o, o + w))
        idx.append(lay["one"][0])
        return np.asarray(sorted(set(idx)))

    ycol = {k: lay["t"][0] + i for i, k in enumerate(names)}
    ycol_s = {k: lay["tshuf"][0] + i for i, k in enumerate(names)}
    pr = {"P1_acc": cols_of("acc"), "P2_acch": cols_of("acc", "h"),
          "P3_rf": cols_of("rf")}
    solves = {}
    for pname, cols in pr.items():
        for t in TARGETS:
            for kind, yc in (("real", ycol[t]), ("null", ycol_s[t])):
                solves[(pname, t, kind)] = (
                    cols, ridge_w(G, cols, G[:, yc], 1e-2))

    scores = {k: [] for k in solves}
    consist, yvals = [], {t: [] for t in TARGETS}
    nrow = 0
    for acs, acn, out, psqt, targ in stream_shards(EVAL):
        for b0 in range(0, len(acs), 4096):
            sl = slice(b0, min(b0 + 4096, len(acs)))
            X = build_X(acs[sl], acn[sl], out[sl], psqt[sl], mom, R, rb,
                        tcols_ev, shuf_ev, nrow + b0)
            for key, (cols, w) in solves.items():
                scores[key].append((X[:, cols] @ w).astype(np.float32))
        consist.append(tcols_ev["cast_consist"][nrow:nrow + len(acs)])
        for t in TARGETS:
            yvals[t].append(tcols_ev[t][nrow:nrow + len(acs)])
        nrow += len(acs)
    m = np.concatenate(consist) > 0.5
    print(f"eval rows {nrow}, cast_consist rows {m.sum()} "
          f"({m.mean():.4f})")

    def auc(score, yy):
        r = np.argsort(np.argsort(score)) + 1.0
        pos = yy > 0.5
        if pos.sum() == 0 or (~pos).sum() == 0:
            return float("nan")
        return float((r[pos].sum() - pos.sum() * (pos.sum() + 1) / 2)
                     / (pos.sum() * (~pos).sum()))

    for t in TARGETS:
        yy = np.concatenate(yvals[t])
        mask = m if t.startswith("castle") else np.ones_like(m)
        print(f"\n{t}: subset n={mask.sum()}, base rate "
              f"{yy[mask].mean():.4f}")
        for pname in pr:
            s = np.concatenate(scores[(pname, t, "real")])
            sn = np.concatenate(scores[(pname, t, "null")])
            print(f"  {pname:8s} AUC {auc(s[mask], yy[mask]):.4f} "
                  f"(uncond {auc(s, yy):.4f})  null {auc(sn[mask], yy[mask]):.4f}")
    return 0


if __name__ == "__main__":
    sys.exit(main())
