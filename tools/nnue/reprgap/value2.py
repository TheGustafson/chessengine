"""Value probes v2: baseline INCLUDES the net's own output.

v1's value arms used rf(acc,h) alone as the base — measured label MSE
0.0154 vs the net's own 0.0081, so a candidate T could "gain" merely by
recovering probe weakness; the material control exposed this (-6.8e-4 fake
gain for a quantity the net decodably has at R^2 0.995). v2 rebases every
arm on [rf, netout, psqt]: a gain now means T improves the LABEL prediction
beyond what the net's own readout plus a rich acc summary achieves — with
the shuffled twin as the null, same solve, same Gram.

  .venv/bin/python tools/nnue/reprgap/value2.py <train_tag> <eval_tag>
"""
from __future__ import annotations

import json
import sys
from pathlib import Path

import numpy as np

ROOT = Path(__file__).resolve().parents[3]
sys.path.insert(0, str(ROOT / "tools/nnue/reprgap"))
sys.path.insert(0, str(ROOT / "jax"))

from probes import (DIR, build_X, layout, make_rf, ridge_w,  # noqa: E402
                    stream_shards, target_columns)

LAMS = [1e-3, 1e-1]


def main(train_tag: str, eval_tag: str) -> int:
    gz = np.load(DIR / f"gram-{train_tag}.npz", allow_pickle=True)
    G = gz["G"]
    xty_l = gz["Xty_label"]
    mom = {k[4:]: gz[k] for k in gz.files if k.startswith("mom_")}
    tcols_tr = target_columns(train_tag)
    lay = layout(tcols_tr)
    names = lay["names"]
    R, rb = make_rf(2560 + 1280)
    tcols_ev = target_columns(eval_tag)
    shuf_ev = {k: v[np.random.default_rng(43).permutation(len(v))]
               for k, v in tcols_ev.items()}

    def cols_of(*blocks, extra=()):
        idx = []
        for b in blocks:
            o, w = lay[b]
            idx.extend(range(o, o + w))
        idx.extend(extra)
        idx.append(lay["one"][0])
        return np.asarray(sorted(set(idx)))

    ycol = {k: lay["t"][0] + i for i, k in enumerate(names)}
    ycol_s = {k: lay["tshuf"][0] + i for i, k in enumerate(names)}
    base_cols = cols_of("rf", "netout", "psqt")
    varms = {}
    for lam in LAMS:
        varms[("base", "-", lam)] = (base_cols, ridge_w(G, base_cols, xty_l, lam))
    for t in names:
        for kind, yc in (("real", ycol[t]), ("null", ycol_s[t])):
            cols = cols_of("rf", "netout", "psqt", extra=(yc,))
            for lam in LAMS:
                varms[(t, kind, lam)] = (cols, ridge_w(G, cols, xty_l, lam))
    # the ALL-real arm: every candidate T at once (the joint headroom)
    all_real = cols_of("rf", "netout", "psqt",
                       extra=tuple(ycol[t] for t in names))
    all_null = cols_of("rf", "netout", "psqt",
                       extra=tuple(ycol_s[t] for t in names))
    for lam in LAMS:
        varms[("ALL", "real", lam)] = (all_real, ridge_w(G, all_real, xty_l, lam))
        varms[("ALL", "null", lam)] = (all_null, ridge_w(G, all_null, xty_l, lam))

    vmse = {k: 0.0 for k in varms}
    nrow = 0
    for acs, acn, out, psqt, targ in stream_shards(eval_tag):
        for b0 in range(0, len(acs), 4096):
            sl = slice(b0, min(b0 + 4096, len(acs)))
            X = build_X(acs[sl], acn[sl], out[sl], psqt[sl], mom, R, rb,
                        tcols_ev, shuf_ev, nrow + b0)
            lab = targ[sl].astype(np.float64)
            for key, (cols, w) in varms.items():
                p = X[:, cols] @ w
                vmse[key] += float(np.sum((p - lab) ** 2))
        nrow += len(acs)
        print(f"[v2 {eval_tag}] {nrow}", flush=True)
    rows = [{"arm": t, "kind": k, "lam": lam, "label_mse": m / nrow}
            for (t, k, lam), m in vmse.items()]
    (DIR / f"value2-{train_tag}-{eval_tag}.json").write_text(
        json.dumps({"rows": rows, "n_eval": nrow}, indent=1))
    base = min(r["label_mse"] for r in rows if r["arm"] == "base")
    print(f"\n=== VALUE v2 (base = rf+netout+psqt) {train_tag}->{eval_tag} "
          f"n={nrow} ===")
    print(f"{'base':14s} {base:.6f}")
    for t in names + ["ALL"]:
        try:
            re_ = min(r["label_mse"] for r in rows
                      if r["arm"] == t and r["kind"] == "real")
            nu = min(r["label_mse"] for r in rows
                     if r["arm"] == t and r["kind"] == "null")
        except ValueError:
            continue
        print(f"{t:14s} +T {re_ - base:+.2e}  +shufT {nu - base:+.2e}")
    return 0


if __name__ == "__main__":
    sys.exit(main(sys.argv[1], sys.argv[2]))
