#!/usr/bin/env python3
"""PER-CORPUS calibration of the Stockfish-net TEACHER against our stored LABELS.

Why (2026-08-01). The distillation blend (60% stored label / 40% SF-net teacher)
mixes two numbers that are NOT in the same frame. Getting the conversion wrong
trains toward a corrupted target and stays invisible for many superbatches, so
the conversion is measured here, PER CORPUS, before any training run.

THREE FRAMES, and only one of them is obvious
---------------------------------------------
teacher  `engines/bin/archon-sfswapbin sf-eval <sf.nnue>` prints
         `sf_swap::Net::evaluate_raw` -- Stockfish's RAW NNUE output in THEIR
         INTERNAL UNITS, side-to-move relative. It is NOT in ARCHON's frame
         (that is `evaluate()`, which multiplies by FRAME = 0.5126) and it is
         NOT printed cp (that is `to_cp = 100*v/p_a(material)`).

label    `recs["score"]`: a Stockfish SEARCH score, mover-relative, divided at
         convert time by that pack's own `SFBINPACK_SCORE_DIV` (the ledger and
         `tools/nnue/runs/stage5_data_chain.sh` record a DIFFERENT divisor per
         pack, 1.49-2.25; the two house corpora are SF18 *UCI cp* rescaled by
         0.674/0.699 instead, which is a different frame family entirely).

material SF's `evaluate()` scales the raw net by material before the search
         ever sees it (eca43a97 evaluate.cpp:80-81):
             material = 534*count<PAWN>() + non_pawn_material()
             v = (nnue*(77871+material) + optimism*(7191+material)) / 77871
         so every LABEL carries a `(77871+material)/77871` factor that
         `evaluate_raw` does not. That factor runs 1.05 (bare endgame) to 1.32
         (startpos) -- a swing a lone scalar buries in the residual.

MODELS (OLS; the label is regressed ON the teacher -- the teacher is
deterministic and the label is the noisy response, so this is both the
unbiased direction and the direction we need, teacher -> OUR frame):

  M1  y = b0 + b1*t                              scalar (what a naive impl uses)
  M2  y = b0 + b1*t + b2*t*(mat/77871)           free material slope; the ratio
                                                 gamma = b2/b1 is 1.0 exactly if
                                                 SF's own blend is the whole law
  MB  y = b0 + b1*(t*blend)                      SF-blend corrected. b1 here is
                                                 the pack's frame constant and
                                                 should equal 1/SCORE_DIV
  MA  y = b0 + b1*(t*blend*100/p_a(mat))         the UCI-cp path, for corpora
                                                 whose labels went through
                                                 `to_cp` (canon-tight, hardneg)

Every model is reported with its R2, its residual's material trend, and the
median label/teacher ratio in the LOW and HIGH material terciles. The tercile
spread is the load-bearing diagnostic: a multiplicative material leak is the
failure mode here and an additive residual-vs-material regression can miss it.

CONVENTIONS. Records are stm-canonicalized (the mover is White on a vertically
mirrored board), so the decoded FEN is white-to-move and `sf-eval`'s stm-relative
output is already in the label's POV. Castling rights and the ep square are not
stored -- irrelevant, because HalfKAv2_hm + threats are blind to both (measured
2026-07-30, 800/800 bit-identical twins). Sign agreement is reported in bands:
the near-zero band is coin-flippy by construction and a pooled figure hides a
real convention bug behind it.

Usage:
  .venv/bin/python tools/nnue/sf_teacher_calibration.py --n 20000 \
      --out docs/research/sf-teacher-calibration.json
"""
from __future__ import annotations

import argparse
import json
import os
import subprocess
import sys
import time
from pathlib import Path

import numpy as np

ROOT = Path(__file__).resolve().parents[2]
# data/ and engines/bin/ are gitignored, so they live only in the main checkout;
# an agent worktree must still be able to reach them.
MAIN = Path(os.environ.get("ARCHON_MAIN", "/home/nick/projects/chessengine"))
SFSWAP = str(MAIN / "engines/bin/archon-sfswapbin")
SFNET = os.environ.get("SF_TEACHER_NET", "/home/nick/sf-eca43a97/src/nn-0ee0657fb25e.nnue")

RECORD_DTYPE = np.dtype([
    ("occ", "<u8"),
    ("pcs", "u1", (16,)),
    ("score", "<i2"),
    ("result", "u1"),
    ("ksq", "u1"),
    ("opp_ksq", "u1"),
    ("extra", "u1", (3,)),
])
assert RECORD_DTYPE.itemsize == 32

PIECE_CHARS = "PNBRQK"
# eca43a97 types.h piece values, indexed by bulletformat pt code 0..5 = P N B R Q K.
# Pawns are counted separately by SF's blend, so PawnValue is not used here.
NPM_VALUE = (0, 781, 825, 1276, 2538, 0)
# uci.cpp:503 win_rate_params material counter (a DIFFERENT weighting from the
# evaluate() blend above -- do not merge the two).
WDL_VALUE = (1, 3, 3, 5, 9, 0)
PAWN_W = 534          # evaluate.cpp:80
BLEND_BASE = 77871    # evaluate.cpp:81
WDL_AS = (-72.32565836, 185.93832038, -144.58862193, 416.44950446)  # uci.cpp:510

# The divisor each pack was converted with. Sources, per corpus:
#   ledger/data.jsonl gen_recipe (rescale/convert rows) and
#   tools/nnue/runs/stage5_data_chain.sh's `do_pack` menu.
# `kind` says which frame family the label lives in and therefore which model
# is expected to fit: "internal" = SF search Value / DIV, "ucicp" = SF18 UCI cp
# (already through to_cp) / DIV.
SCORE_DIV = {
    "uho-band75-100":       (1.90, "internal"),
    "fishpack32":           (1.90, "internal"),
    "farseerT74-band75-100": (1.66, "internal"),
    "datapv2-band75-88":    (2.25, "internal"),
    "farseerT75-band75-83": (1.94, "internal"),
    "multinet-band75-88":   (1.98, "internal"),
    "t60t70-band75-85":     (1.97, "internal"),
    "pylon-band75-88":      (1.95, "internal"),
    "wisr-band75-95":       (1.85, "internal"),
    "wrongnnue-band75-95":  (1.49, "internal"),
    "canon-tight-frame":    (0.674, "ucicp"),
    "hardneg-frame":        (0.699, "ucicp"),
}


# --------------------------------------------------------------------------
# record decoding
# --------------------------------------------------------------------------

def decode(recs: np.ndarray):
    """records -> (FENs, blend material, WDL material count, piece count).

    `occ` set bits in ASCENDING square order pair with `pcs` nibbles in order;
    each nibble is (color << 3) | piece_type, with color 0 = the mover.
    """
    n = len(recs)
    fens: list[str] = []
    material = np.zeros(n, dtype=np.int64)
    wdlmat = np.zeros(n, dtype=np.int64)
    npieces = np.zeros(n, dtype=np.int32)
    occ_all = recs["occ"]
    pcs_all = recs["pcs"]
    for i in range(n):
        occ = int(occ_all[i])
        pcs = pcs_all[i]
        board = [0] * 64
        k = pawns = npm = wm = 0
        while occ:
            sq = (occ & -occ).bit_length() - 1
            occ &= occ - 1
            code = (int(pcs[k >> 1]) >> (4 * (k & 1))) & 0xF
            color, pt = code >> 3, code & 7
            board[sq] = (color << 3) | pt | 0x40
            if pt == 0:
                pawns += 1
            else:
                npm += NPM_VALUE[pt]
            wm += WDL_VALUE[pt]
            k += 1
        npieces[i] = k
        material[i] = PAWN_W * pawns + npm
        wdlmat[i] = wm
        rows = []
        for rank in range(7, -1, -1):
            row, empty = [], 0
            base = rank * 8
            for f in range(8):
                v = board[base + f]
                if not v:
                    empty += 1
                    continue
                if empty:
                    row.append(str(empty))
                    empty = 0
                ch = PIECE_CHARS[v & 7]
                row.append(ch if not (v & 8) else ch.lower())
            if empty:
                row.append(str(empty))
            rows.append("".join(row))
        fens.append("/".join(rows) + " w - - 0 1")
    return fens, material, wdlmat, npieces


def p_a(wdlmat: np.ndarray) -> np.ndarray:
    """SF's `to_cp` divisor a = p_a(material) (uci.cpp:501-517)."""
    m = np.clip(wdlmat, 17, 78).astype(float) / 58.0
    return ((WDL_AS[0] * m + WDL_AS[1]) * m + WDL_AS[2]) * m + WDL_AS[3]


def sample(path: str, n: int, seed: int, chunk: int = 1) -> np.ndarray:
    """n records drawn INDEPENDENTLY at uniform random over the whole corpus.

    `chunk` > 1 reads contiguous windows instead: faster, and fine for pilots.

    Why iid is the default, stated accurately (an earlier version of this
    docstring overstated it and the study doc §6 carries the retraction): a
    record-level bootstrap *assumes* iid draws, and neighbouring records in a
    `*.shuf.bin` still share a game, so windows satisfy that assumption only
    approximately. That is a reason to draw iid on principle -- it costs 3.5 s
    per 20 000 records on the 81 GB corpus -- NOT evidence that windows measurably
    broke anything. Tested properly (pairwise z against the bootstrap SE inflated
    by sqrt(2), because the difference of two estimates carries sqrt(2) the SE),
    window and iid give sd(z) 1.30 vs 1.19 on k and 1.12 vs 1.12 on the scalar
    slope -- indistinguishable at these n. Point estimates moved < 2 %.

    What IS real, in both regimes, is a mild ~1.1-1.2x under-statement by the
    percentile bootstrap -- ordinary heavy-tail behaviour for an OLS slope whose
    leverage sits in a few extreme-|teacher| positions. For bars you intend to
    lean on, average independent seeds rather than trusting one bootstrap.
    """
    mm = np.memmap(path, dtype=RECORD_DTYPE, mode="r")
    total = len(mm)
    rng = np.random.default_rng(seed)
    if chunk <= 1:
        # sorted indices keep the page walk monotone; the sample is still iid
        out = np.asarray(mm[np.sort(rng.integers(0, total, size=n))])
    else:
        chunk = min(chunk, max(1, total // 64))
        starts = rng.integers(0, max(1, total - chunk), size=max(1, n // chunk))
        out = np.concatenate([np.asarray(mm[s:s + chunk]) for s in starts])[:n]
    del mm
    return out


# --------------------------------------------------------------------------
# teacher
# --------------------------------------------------------------------------

def teacher_evals(fens: list[str], net: str = SFNET) -> np.ndarray:
    """sf-eval over FENs, line-aligned with `fens` (NaN where unparsed).

    A silent drop would shift every later pair by one and destroy the fit while
    still producing a plausible-looking slope, so the line count is checked.
    """
    r = subprocess.run([SFSWAP, "sf-eval", net], input="\n".join(fens) + "\n",
                       capture_output=True, text=True, timeout=7200)
    lines = r.stdout.splitlines()
    if len(lines) != len(fens):
        raise RuntimeError(f"sf-eval returned {len(lines)} lines for {len(fens)} fens; "
                           f"stderr: {r.stderr[:400]}")
    out = np.full(len(fens), np.nan)
    for i, ln in enumerate(lines):
        s = ln.strip()
        if s.lstrip("-").isdigit():
            out[i] = float(s)
    return out


# --------------------------------------------------------------------------
# fitting
# --------------------------------------------------------------------------

def ols(X: np.ndarray, y: np.ndarray):
    beta, *_ = np.linalg.lstsq(X, y, rcond=None)
    resid = y - X @ beta
    ss_res = float(resid @ resid)
    ss_tot = float(((y - y.mean()) ** 2).sum())
    return beta, (1.0 - ss_res / ss_tot if ss_tot else float("nan")), resid


def boot_ci(X: np.ndarray, y: np.ndarray, idx: int, reps: int, seed: int = 7):
    rng = np.random.default_rng(seed)
    n = len(y)
    vals = np.empty(reps)
    for r in range(reps):
        s = rng.integers(0, n, n)
        b, _, _ = ols(X[s], y[s])
        vals[r] = b[idx]
    return [float(np.percentile(vals, 2.5)), float(np.percentile(vals, 97.5))]


def terciles(y: np.ndarray, x: np.ndarray, mat: np.ndarray, floor: float = 100.0):
    """median y/x in the low and high material terciles, and their ratio.

    This is the multiplicative-leak test. `floor` keeps out the near-zero band
    where a ratio is numerically meaningless.
    """
    g = np.abs(x) > floor
    if g.sum() < 200:
        return {}
    mm, rr = mat[g], y[g] / x[g]
    lo_c, hi_c = np.percentile(mm, [33.333, 66.667])
    lo = float(np.median(rr[mm <= lo_c]))
    hi = float(np.median(rr[mm >= hi_c]))
    return {"lo": lo, "hi": hi, "spread": (hi / lo if lo else float("nan"))}


def resid_trend(resid: np.ndarray, mat: np.ndarray) -> dict:
    A = np.column_stack([np.ones(len(mat)), mat.astype(float)])
    b, _, _ = ols(A, resid)
    lo_c, hi_c = np.percentile(mat, [33.333, 66.667])
    return {"slope_per_mat": float(b[1]),
            "mean_lowmat": float(resid[mat <= lo_c].mean()),
            "mean_highmat": float(resid[mat >= hi_c].mean())}


def fit_corpus(label, teacher, mat, wdlmat, reps: int = 300) -> dict:
    y = label.astype(float)
    t = teacher.astype(float)
    m = mat.astype(float) / BLEND_BASE
    blend = 1.0 + m
    tb = t * blend                       # teacher pushed through SF's own blend
    ta = tb * 100.0 / p_a(wdlmat)        # ... and then through to_cp
    one = np.ones(len(y))

    res: dict = {"n": int(len(y))}

    # sign agreement, in bands. A pooled number hides a convention bug.
    res["sign"] = {}
    for lo in (0, 50, 100, 200):
        k = (np.abs(y) > lo) & (np.abs(t) > lo)
        res["sign"][f">{lo}"] = [int(k.sum()),
                                 float(np.mean(np.sign(y[k]) == np.sign(t[k]))) if k.sum() else float("nan")]

    for name, x in (("M1", t), ("MB", tb), ("MA", ta)):
        X = np.column_stack([one, x])
        b, r2, rr = ols(X, y)
        d = {"slope": float(b[1]), "intercept": float(b[0]), "r2": float(r2),
             "slope_zero_int": float(x @ y / (x @ x)) if (x @ x) else float("nan"),
             "resid": resid_trend(rr, mat), "ratio": terciles(y, x, mat)}
        if reps:
            d["slope_ci95"] = boot_ci(X, y, 1, reps)
        res[name] = d

    X2 = np.column_stack([one, t, t * m])
    b2, r2_2, rr2 = ols(X2, y)
    gamma = float(b2[2] / b2[1]) if b2[1] else float("nan")
    # M2's own material-corrected teacher, using the FITTED gamma
    t2 = t * (1.0 + gamma * m)
    res["M2"] = {"slope": float(b2[1]), "slope_mat": float(b2[2]),
                 "intercept": float(b2[0]), "r2": float(r2_2),
                 "gamma": gamma, "resid": resid_trend(rr2, mat),
                 "ratio": terciles(y, t2, mat)}
    if reps:
        res["M2"]["slope_ci95"] = boot_ci(X2, y, 1, reps)
        res["M2"]["slope_mat_ci95"] = boot_ci(X2, y, 2, reps)

    # MP: label = k * blend^p * teacher, fitted in LOG space on the decisive,
    # sign-agreeing subpopulation. Two reasons this is the shippable form and
    # M2 is not:
    #   * (1 + gamma*m) goes NEGATIVE for gamma < -3 (wrongnnue fits -4), which
    #     is a sign flip in the endgame -- unusable as a conversion.
    #   * a level-space LS slope is dominated by the |eval| tail; the trainer's
    #     target is a saturating sigmoid, so the TYPICAL position is what the
    #     loss actually sees. A log-ratio fit weights positions, not centipawns.
    # p == 1 reproduces SF's own evaluate() blend exactly; p == 0 says the
    # corpus shows no material dependence at all.
    good = (np.abs(t) > 150) & (np.sign(y) == np.sign(t)) & (y != 0)
    res["MP"] = {"n_fit": int(good.sum())}
    if good.sum() > 500:
        lr = np.log(np.abs(y[good]) / np.abs(t[good]))
        lb = np.log(blend[good])
        Xp = np.column_stack([np.ones(int(good.sum())), lb])
        bp, r2p, rp = ols(Xp, lr)
        k, p = float(np.exp(bp[0])), float(bp[1])
        tp = t * k * blend ** p
        res["MP"].update(k=k, p=p, r2_log=float(r2p),
                         ratio=terciles(y, tp, mat),
                         resid=resid_trend(y - tp, mat))
        if reps:
            ci = boot_ci(Xp, lr, 0, reps)
            res["MP"]["k_ci95"] = [float(np.exp(ci[0])), float(np.exp(ci[1]))]
            res["MP"]["p_ci95"] = boot_ci(Xp, lr, 1, reps)

    res["label_std"] = float(y.std())
    res["teacher_std"] = float(t.std())
    res["pearson"] = float(np.corrcoef(y, t)[0, 1])
    res["material_mean"] = float(mat.mean())
    res["blend_mean"] = float(blend.mean())
    res["p_a_mean"] = float(p_a(wdlmat).mean())
    return res


# --------------------------------------------------------------------------

def main() -> int:
    ap = argparse.ArgumentParser(description=__doc__,
                                 formatter_class=argparse.RawDescriptionHelpFormatter)
    # d49_1280.json is untracked, so in an agent worktree it only exists in the
    # main checkout -- fall back there rather than failing.
    spec_default = ROOT / "tools/nnue/runs/d49_1280.json"
    if not spec_default.exists():
        spec_default = MAIN / "tools/nnue/runs/d49_1280.json"
    ap.add_argument("--spec", default=str(spec_default))
    ap.add_argument("--n", type=int, default=20000)
    ap.add_argument("--seed", type=int, default=20260801)
    ap.add_argument("--clip", type=int, default=12000,
                    help="drop |label| above this (mate/tb scores are not a frame)")
    ap.add_argument("--reps", type=int, default=300, help="bootstrap reps (0 = off)")
    ap.add_argument("--chunk", type=int, default=1,
                    help="1 = independent random records (correct bars); >1 = "
                         "contiguous windows, faster but clustered -- pilots only")
    ap.add_argument("--out", default=str(ROOT / "docs/research/sf-teacher-calibration.json"))
    ap.add_argument("--only", default="", help="substring filter on corpus path")
    ap.add_argument("--null", action="store_true",
                    help="NULL PANEL: permute the teacher against the labels "
                         "within each corpus. Every slope must collapse to ~0 "
                         "and sign agreement to ~0.5. If it does not, the "
                         "instrument -- not the arm -- is producing the signal.")
    a = ap.parse_args()

    spec = json.load(open(a.spec))
    mix = spec["mix"]
    pct = spec.get("per_corpus_target")
    print(f"spec {a.spec}: {len(mix)} corpora, per_corpus_target "
          f"{len(pct) if pct else 0} entries", flush=True)
    print(f"teacher: {SFSWAP} sf-eval {SFNET}\n", flush=True)

    rows = []
    for i, (path, weight) in enumerate(mix):
        if a.only and a.only not in path:
            continue
        full = str(MAIN / path)
        name = Path(path).name.replace(".shuf.bin", "")
        if not os.path.exists(full):
            print(f"  !! {name}: {full} missing -- SKIPPED", file=sys.stderr)
            continue
        t0 = time.time()
        recs = sample(full, a.n, a.seed + i, chunk=a.chunk)
        fens, mat, wdlmat, npc = decode(recs)
        lab = recs["score"].astype(np.int64)
        tv = teacher_evals(fens)
        if a.null:
            tv = np.random.default_rng(1000 + i).permutation(tv)
        keep = np.isfinite(tv) & (np.abs(lab) <= a.clip)
        div, kind = SCORE_DIV.get(name, (float("nan"), "?"))
        r = fit_corpus(lab[keep], tv[keep], mat[keep], wdlmat[keep], reps=a.reps)
        r.update(idx=i, corpus=name, path=path, weight=weight,
                 per_corpus_target=(pct[i] if pct else None),
                 score_div=div, frame_kind=kind,
                 predicted_slope=(1.0 / div if div == div else float("nan")),
                 dropped=int((~keep).sum()), n_sampled=int(len(recs)),
                 piece_count_mean=float(npc[keep].mean()),
                 secs=round(time.time() - t0, 1))
        rows.append(r)
        mp = r["MP"]
        print(f"[{i:2d}] {name:24s} n={r['n']:6d}  "
              f"M1 {r['M1']['slope']:.4f} (R2 {r['M1']['r2']:.3f}, terc "
              f"{r['M1']['ratio'].get('spread', float('nan')):.3f})  "
              f"MB {r['MB']['slope']:.4f} (terc "
              f"{r['MB']['ratio'].get('spread', float('nan')):.3f})  "
              f"MP k={mp.get('k', float('nan')):.4f} p={mp.get('p', float('nan')):+.2f} "
              f"(terc {mp.get('ratio', {}).get('spread', float('nan')):.3f})  "
              f"pred 1/{div:.3f}={1.0 / div:.4f}  "
              f"sign>100 {r['sign']['>100'][1]:.3f}  ({r['secs']}s)", flush=True)

    out = {"generated": time.strftime("%Y-%m-%dT%H:%M:%S"),
           "spec": a.spec, "n_per_corpus": a.n, "seed": a.seed, "clip": a.clip,
           "teacher_net": SFNET, "teacher_binary": SFSWAP,
           "teacher_units": "Stockfish RAW NNUE internal (sf_swap::evaluate_raw), stm-relative",
           "rows": rows}
    Path(a.out).parent.mkdir(parents=True, exist_ok=True)
    json.dump(out, open(a.out, "w"), indent=1)
    print(f"\nwrote {a.out}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
