"""Representation-gap study: capacity-stratified decodability + value probes.

One Gram matrix per dataset over the feature blocks

  acc   (2560)  pre-CReLU accumulators, both perspectives  [linear class]
  h     (1280)  CReLU(0,1) pairwise products, both persp   [the head's
                product slots / out_skip lane function class]
  rf    (3072)  relu(R [acc_std; h_std] + b), R,b frozen   [shallow-MLP
                class upper bound, closed-form trainable]
  netout, psqt  the net's own value logit + skip term      [reference row:
                ridge(netout) must reproduce ~panel heldout MSE]
  T     target columns + per-column shuffled twins          [value-probe
                arms + the shuffle nulls]
  1     intercept

Every probe is an exact column-subset ridge solve from the same Gram:
  DECODE  P1 acc -> T, P2 [acc,h] -> T, P3 rf -> T   (+ shuffled-label null)
  VALUE   [rf] -> label  vs  [rf, T] -> label  vs  [rf, shuf(T)] -> label
Train Gram on the TRAIN dataset, evaluate predictions on the EVAL dataset.
Binary targets report AUC; continuous report R^2. All CPU, all numpy.

Run:
  .venv/bin/python tools/nnue/reprgap/probes.py gram <tag>       # heavy pass
  .venv/bin/python tools/nnue/reprgap/probes.py report <train_tag> <eval_tag>
"""
from __future__ import annotations

import json
import sys
from pathlib import Path

import numpy as np

ROOT = Path(__file__).resolve().parents[3]
DIR = ROOT / "data/nnue/nets/screens/reprgap"
RF_DIM = 3072
RF_SEED = 220480777
SHUF_SEED = 41
HL = 1280
DECODE_EVAL_CAP = 131072   # rows of stored decode scores (AUC needs vectors;
                           # capped for the ~3 GB box budget)

# tag -> (acc shard prefix, targets npz, sidecar npz | None, row_start, row_end)
SETS = {
    "mix": ("acc-mix-5000000-350000-", "targets-mix-5000000-350000.npz",
            None, 0, None),
    "heldout": ("acc-heldout-0-500000-", "targets-heldout-0-500000.npz",
                None, 0, None),
    "tsvfilt_tr": ("acc-recs-tsvfilt-", "targets-recs-tsvfilt.npz",
                   "recs-tsvfilt.npz", 0, 350_000),
    "tsvfilt_ev": ("acc-recs-tsvfilt-", "targets-recs-tsvfilt.npz",
                   "recs-tsvfilt.npz", 350_000, None),
    "tsvraw": ("acc-recs-tsvraw-", "targets-recs-tsvraw.npz",
               "recs-tsvraw.npz", 0, None),
}

BATTERY = ["material", "pcount", "hashbit", "has_capture", "see_best",
           "see_pos", "opp_see_pos", "in_check", "hang_cnt", "atk_lower",
           "pin_own", "pin_opp", "passed_own", "passed_opp"]
BINARY = {"hashbit", "has_capture", "see_pos", "opp_see_pos", "in_check",
          "castle_own_k", "castle_own_q", "castle_opp_k", "castle_opp_q",
          "castle_any", "ep_any", "rule50_hi", "cast_consist"}


def target_columns(tag: str) -> dict[str, np.ndarray]:
    _, tfile, side, r0, r1 = SETS[tag]
    t = np.load(DIR / tfile)
    cols = {k: t[k].astype(np.float64) for k in BATTERY}
    cols["see_best"] = np.clip(cols["see_best"], -900, 900) / 100.0
    if side:
        s = np.load(DIR / side)
        for k in ("castle_own_k", "castle_own_q", "castle_opp_k",
                  "castle_opp_q"):
            cols[k] = s[k].astype(np.float64)
        cols["castle_any"] = (cols["castle_own_k"] + cols["castle_own_q"]
                              + cols["castle_opp_k"] + cols["castle_opp_q"]
                              > 0).astype(np.float64)
        cols["ep_any"] = (s["ep_file"] >= 0).astype(np.float64)
        cols["rule50"] = s["rule50"].astype(np.float64) / 50.0
        cols["rule50_hi"] = (s["rule50"] >= 20).astype(np.float64)
        cols["cast_consist"] = consistency_flags(s["recs"])
    return {k: v[r0:r1] for k, v in cols.items()}


def consistency_flags(recs_u8: np.ndarray) -> np.ndarray:
    """own K on e1 AND own R on h1 (kingside home shape), from record bytes."""
    from archon_jax.spec import RECORD_DTYPE  # numpy-pure
    recs = recs_u8.view(RECORD_DTYPE).reshape(-1)
    occ = recs["occ"]
    n = len(recs)
    out = np.zeros(n, np.float64)
    ksq_ok = recs["ksq"] == 4
    h1 = (occ >> np.uint64(7)) & np.uint64(1)
    for i in np.nonzero(ksq_ok & (h1 == 1))[0]:
        below = int(occ[i]) & 0x7F
        j = bin(below).count("1")
        nib = (int(recs["pcs"][i][j // 2]) >> (4 * (j % 2))) & 0xF
        out[i] = float(nib == 3)  # color 0, rook (pt=3)
    return out


def stream_shards(tag: str):
    """Yield (acc_stm, acc_ntm, out, psqt, target) already row-sliced."""
    prefix, _, _, r0, r1 = SETS[tag]
    files = sorted(DIR.glob(prefix + "*.npz"))
    assert files, f"no acc shards for {tag}"
    off = 0
    for f in files:
        z = np.load(f)
        nb = len(z["out"])
        lo = max(r0 - off, 0)
        hi = nb if r1 is None else min(max(r1 - off, 0), nb)
        if lo < hi:
            sl = slice(lo, hi)
            yield (z["acc_stm"][sl], z["acc_ntm"][sl], z["out"][sl],
                   z["psqt"][sl], z["target"][sl])
        off += nb


def _feat(acs, acn):
    acc = np.concatenate([acs, acn], axis=1).astype(np.float32)
    c = np.clip(acc, 0.0, 1.0)  # the network's _crelu
    hh = np.concatenate([c[:, :HL // 2] * c[:, HL // 2:HL],
                         c[:, HL:HL + HL // 2] * c[:, HL + HL // 2:]], axis=1)
    return acc, hh


def moments(tag: str) -> dict:
    """Chunked at 8192 rows: whole-shard f64 copies were a measured ~2.7 GB
    peak (coordinator flag 2026-07-30) — the box budget is ~3 GB total."""
    n = 0
    s1a = np.zeros(2560); s2a = np.zeros(2560)
    s1h = np.zeros(1280); s2h = np.zeros(1280)
    for acs, acn, *_ in stream_shards(tag):
        for b0 in range(0, len(acs), 8192):
            sl = slice(b0, b0 + 8192)
            acc, hh = _feat(acs[sl], acn[sl])
            a64, h64 = acc.astype(np.float64), hh.astype(np.float64)
            s1a += a64.sum(0); s2a += (a64 ** 2).sum(0)
            s1h += h64.sum(0); s2h += (h64 ** 2).sum(0)
            n += len(acc)
    ma, mh = s1a / n, s1h / n
    va = np.maximum(s2a / n - ma ** 2, 1e-8)
    vh = np.maximum(s2h / n - mh ** 2, 1e-8)
    return {"ma": ma, "sa": np.sqrt(va), "mh": mh, "sh": np.sqrt(vh),
            "n": np.float64(n)}


def make_rf(nin: int):
    rng = np.random.default_rng(RF_SEED)
    R = (rng.standard_normal((nin, RF_DIM)) / np.sqrt(nin)).astype(np.float32)
    rb = (0.5 * rng.standard_normal(RF_DIM)).astype(np.float32)
    return R, rb


def build_X(acs, acn, out, psqt, mom, R, rb, tcols, shufs, row0) -> np.ndarray:
    acc, hh = _feat(acs, acn)
    za = (acc - mom["ma"].astype(np.float32)) / mom["sa"].astype(np.float32)
    zh = (hh - mom["mh"].astype(np.float32)) / mom["sh"].astype(np.float32)
    z = np.concatenate([za, zh], axis=1)
    rf = np.maximum(z @ R + rb, 0.0)
    nb = len(acc)
    names = sorted(tcols)
    tb = np.stack([tcols[k][row0:row0 + nb] for k in names], axis=1)
    ts = np.stack([shufs[k][row0:row0 + nb] for k in names], axis=1)
    ones = np.ones((nb, 1), np.float32)
    return np.concatenate(
        [za, zh, rf, out[:, None], psqt[:, None], tb.astype(np.float32),
         ts.astype(np.float32), ones], axis=1).astype(np.float64)


def layout(tcols) -> dict:
    names = sorted(tcols)
    k = len(names)
    off_rf = 2560 + 1280
    off_out = off_rf + RF_DIM
    off_t = off_out + 2
    off_ts = off_t + k
    dim = off_ts + k + 1
    return {"names": names, "acc": (0, 2560), "h": (2560, 1280),
            "rf": (off_rf, RF_DIM), "netout": (off_out, 1),
            "psqt": (off_out + 1, 1), "t": (off_t, k), "tshuf": (off_ts, k),
            "one": (dim - 1, 1), "dim": dim}


def gram(tag: str) -> None:
    mom = moments(tag)
    print(f"[{tag}] moments over n={int(mom['n'])}", flush=True)
    tcols = target_columns(tag)
    lay = layout(tcols)
    rng = np.random.default_rng(SHUF_SEED)
    shufs = {k: v[rng.permutation(len(v))] for k, v in tcols.items()}
    R, rb = make_rf(2560 + 1280)
    G = np.zeros((lay["dim"], lay["dim"]), np.float64)
    Xty_label = np.zeros(lay["dim"], np.float64)
    n = 0
    for acs, acn, out, psqt, targ in stream_shards(tag):
        for b0 in range(0, len(acs), 4096):
            sl = slice(b0, min(b0 + 4096, len(acs)))
            X = build_X(acs[sl], acn[sl], out[sl], psqt[sl], mom, R, rb,
                        tcols, shufs, n + b0)
            G += X.T @ X
            Xty_label += X.T @ targ[sl].astype(np.float64)
        n += len(acs)
        print(f"[{tag}] gram {n}", flush=True)
    np.savez(DIR / f"gram-{tag}.npz", G=G, Xty_label=Xty_label, n=n,
             names=np.array(lay["names"]),
             **{f"mom_{k}": v for k, v in mom.items()})
    print(f"[{tag}] gram saved dim={lay['dim']}")


def ridge_w(G, cols, xty, lam):
    Gs = G[np.ix_(cols, cols)]
    d = len(cols)
    A = Gs + lam * np.eye(d) * (np.trace(Gs) / d)
    return np.linalg.solve(A, xty[cols])


def report(train_tag: str, eval_tag: str) -> None:
    gz = np.load(DIR / f"gram-{train_tag}.npz", allow_pickle=True)
    G, n_tr = gz["G"], int(gz["n"])
    mom = {k[4:]: gz[k] for k in gz.files if k.startswith("mom_")}
    tcols_tr = target_columns(train_tag)
    lay = layout(tcols_tr)
    names = lay["names"]
    R, rb = make_rf(2560 + 1280)

    tcols_ev = target_columns(eval_tag)
    assert sorted(tcols_ev) == names, "train/eval target sets differ"
    rng = np.random.default_rng(SHUF_SEED + 1)
    shuf_ev = {k: v[rng.permutation(len(v))] for k, v in tcols_ev.items()}

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
    LAMS = [1e-3, 1e-1]
    probes = {"P0_netout": cols_of("netout", "psqt"), "P1_acc": cols_of("acc"),
              "P2_acch": cols_of("acc", "h"), "P3_rf": cols_of("rf")}

    solves = {}
    for pname, cols in probes.items():
        for t in names:
            for kind, yc in (("real", ycol[t]), ("null", ycol_s[t])):
                for lam in LAMS:
                    solves[(pname, t, kind, lam)] = (
                        cols, ridge_w(G, cols, G[:, yc], lam))
    xty_l = gz["Xty_label"]
    varms = {}
    for t in names:
        for kind, yc in (("real", ycol[t]), ("null", ycol_s[t])):
            cols = cols_of("rf", extra=(yc,))
            for lam in LAMS:
                varms[(t, kind, lam)] = (cols, ridge_w(G, cols, xty_l, lam))
    for lam in LAMS:
        varms[("base", "-", lam)] = (cols_of("rf"),
                                     ridge_w(G, cols_of("rf"), xty_l, lam))
        varms[("netout", "-", lam)] = (
            cols_of("netout", "psqt"),
            ridge_w(G, cols_of("netout", "psqt"), xty_l, lam))

    dec_scores = {k: [] for k in solves}
    vmse = {k: 0.0 for k in varms}
    ys = {t: [] for t in names}
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
            if nrow + b0 < DECODE_EVAL_CAP:
                for key, (cols, w) in solves.items():
                    dec_scores[key].append((X[:, cols] @ w).astype(np.float32))
        for t in names:
            ys[t].append(tcols_ev[t][nrow:nrow + len(acs)])
        nrow += len(acs)
        print(f"[eval {eval_tag}] {nrow}", flush=True)
    y = {t: np.concatenate(ys[t]) for t in names}

    def auc(score, yy):
        r = np.argsort(np.argsort(score)) + 1.0
        pos = yy > 0.5
        np_, nn = pos.sum(), (~pos).sum()
        if np_ == 0 or nn == 0:
            return float("nan")
        return float((r[pos].sum() - np_ * (np_ + 1) / 2) / (np_ * nn))

    out_rows = []
    for pname in probes:
        for t in names:
            for kind in ("real", "null"):
                vals = []
                for lam in LAMS:
                    p = np.concatenate(dec_scores[(pname, t, kind, lam)])
                    yy = y[t][:len(p)]
                    if t in BINARY:
                        vals.append(auc(p, yy))
                    else:
                        vals.append(1.0 - np.mean((p - yy) ** 2)
                                    / max(np.var(yy), 1e-12))
                out_rows.append({"probe": pname, "target": t, "kind": kind,
                                 "metric": "auc" if t in BINARY else "r2",
                                 "value": round(float(np.nanmax(vals)), 4)})
    vrows = [{"arm": t, "kind": kind, "lam": lam,
              "label_mse": round(m / nrow, 7)}
             for (t, kind, lam), m in vmse.items()]
    outp = DIR / f"probe-report-{train_tag}-{eval_tag}.json"
    outp.write_text(json.dumps({"decode": out_rows, "value": vrows,
                                "n_train": n_tr, "n_eval": nrow}, indent=1))
    print(f"wrote {outp}\n")
    print(f"=== DECODE {train_tag}->{eval_tag} (AUC bin / R2 cont; "
          f"eval n={min(nrow, DECODE_EVAL_CAP)}) ===")
    print(f"{'target':14s} {'P0_net':>7s} {'P1_acc':>7s} {'P2_acch':>7s} "
          f"{'P3_rf':>7s} {'P3null':>7s}")
    for t in names:
        def g(pn, kind="real"):
            return max(r["value"] for r in out_rows
                       if r["probe"] == pn and r["target"] == t
                       and r["kind"] == kind)
        print(f"{t:14s} {g('P0_netout'):7.3f} {g('P1_acc'):7.3f} "
              f"{g('P2_acch'):7.3f} {g('P3_rf'):7.3f} "
              f"{g('P3_rf', 'null'):7.3f}")
    base = min(r["label_mse"] for r in vrows if r["arm"] == "base")
    netr = min(r["label_mse"] for r in vrows if r["arm"] == "netout")
    print(f"\n=== VALUE (label MSE on {eval_tag}, n={nrow}) ===")
    print(f"{'netout ref':14s} {netr:.6f}   rf-base {base:.6f}")
    for t in names:
        re_ = min(r["label_mse"] for r in vrows
                  if r["arm"] == t and r["kind"] == "real")
        nu = min(r["label_mse"] for r in vrows
                 if r["arm"] == t and r["kind"] == "null")
        print(f"{t:14s} +T {re_:.6f} ({re_ - base:+.2e})  "
              f"+shufT {nu:.6f} ({nu - base:+.2e})")


if __name__ == "__main__":
    sys.path.insert(0, str(ROOT / "jax"))
    if sys.argv[1] == "gram":
        gram(sys.argv[2])
    elif sys.argv[1] == "report":
        report(sys.argv[2], sys.argv[3])
    else:
        sys.exit(2)
