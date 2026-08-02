#!/usr/bin/env python3
"""tools/ops/data/study_label_quality.py — pick the oracle node count.

THE QUESTION
  How many nodes should the labelling oracle get? Deeper is NOT automatically
  better: a shallow NNUE cannot represent what a deep search knows, and a label
  it cannot represent is noise. So we need the point where extra nodes stop
  buying agreement with the game outcome, and we need to know what they start
  costing instead.

WHAT IT MEASURES (discrete first, per the method rules)

  1. EVAL-OUTCOME DISAGREEMENT — the owner's named metric, and the primary one.
     Among positions the oracle calls decisive (|cp| >= T), the fraction where
     the game went to the OTHER side. A label that is individually accurate but
     contradicts the outcome is noise to a WDL-blended target.
       Read it like this: at low nodes the rate is dominated by ORACLE error and
     falls as nodes rise. It cannot fall to zero, because the rest is PLAYER
     error -- the game was played by engines that throw advantages away. Where
     the curve FLATTENS, extra nodes are no longer buying outcome consistency,
     and that flattening point is the knee we are looking for.

  2. TIER AGREEMENT — best-move agreement and |delta cp| against the deepest
     tier run. Frame-free and immune to the cp scale.

  3. THE NULL, RUN FIRST AND REPORTED FIRST. Two independent nulls:
       (a) instrument null: label_positions.py --null showed 200/200 labels
           bit-identical, so there is no run-to-run noise to confuse a tier
           difference with.
       (b) SAMPLING null: the position set is split into two disjoint halves by
           FEN hash and every statistic is computed on each half separately.
           The half-vs-half spread at a FIXED node count is the noise floor of
           this instrument. Any tier-to-tier movement smaller than that floor is
           NOT RESOLVABLE and is reported as such rather than read as a trend.

  4. SLICES — by piece count (phase), by |cp| band, by termination
     (adjudication vs played out), and by distance to the end of the game.

USAGE
  .venv/bin/python tools/ops/data/study_label_quality.py \
      --positions data/nnue/selfplay/pilot-self/positions.tsv \
      --oracle engines/bin/stockfish18 \
      --nodes 2000,5000,10000,25000,50000,100000 \
      --workers 6 --limit 3000 --out data/nnue/selfplay/study/nodes-self

  Reuses cached tier TSVs if present, so it is resumable and cheap to re-read.

FRAME
  Every Stockfish binary here prints real cp, so tiers are directly comparable.
  An ARCHON oracle (--net given) prints ~2.07x that; the tool records the frame
  per tier and REFUSES to mix frames in one comparison rather than quietly
  dividing, because a hand-applied 2.07 is exactly the error the boundary
  helper exists to prevent.
"""
from __future__ import annotations

import argparse
import hashlib
import json
import subprocess
import sys
from pathlib import Path

import chess

ROOT = Path(__file__).resolve().parents[3]
LABELLER = ROOT / "tools/ops/data/label_positions.py"
PY = ROOT / ".venv/bin/python"


def half_of(fen: str) -> int:
    return hashlib.blake2b(fen.encode(), digest_size=8).digest()[0] & 1


def load_tier(path: Path) -> dict:
    rows = {}
    with open(path) as fh:
        hdr = fh.readline().rstrip("\n").split("\t")
        ix = {c: i for i, c in enumerate(hdr)}
        for line in fh:
            f = line.rstrip("\n").split("\t")
            if len(f) < len(hdr):
                continue
            fen = f[ix["fen"]]
            raw = f[ix["raw_cp"]]
            rows[fen] = dict(
                raw=int(raw) if raw not in ("", None) else None,
                best=f[ix["best"]],
                result=f[ix["result"]],
                ply=int(f[ix["ply"]] or 0),
                pte=int(f[ix["plies_to_end"]] or 0),
                term=f[ix["termination"]],
                reason=f[ix["drop_reason"]] if "drop_reason" in ix else "",
            )
    return rows


def piece_count(fen: str) -> int:
    return sum(1 for c in fen.split()[0] if c.isalpha())


def disagreement(items, T):
    """items: list of (cp_white, result_float). Returns (n_decisive, n_contra)."""
    dec = contra = 0
    for cp, r in items:
        if cp is None:
            continue
        if abs(cp) < T:
            continue
        dec += 1
        if (cp >= T and r == 0.0) or (cp <= -T and r == 1.0):
            contra += 1
    return dec, contra


def wilson(k, n):
    """95% Wilson interval half-width, in percent. Small-n honest."""
    if n == 0:
        return 0.0
    p = k / n
    z = 1.96
    denom = 1 + z * z / n
    centre = (p + z * z / (2 * n)) / denom
    half = z * ((p * (1 - p) / n + z * z / (4 * n * n)) ** 0.5) / denom
    return round(100 * half, 2), round(100 * centre, 2)


def main() -> int:
    ap = argparse.ArgumentParser()
    ap.add_argument("--positions", required=True)
    ap.add_argument("--oracle", default="engines/bin/stockfish18")
    ap.add_argument("--net", default=None)
    ap.add_argument("--nodes", default="2000,5000,10000,25000,50000,100000")
    ap.add_argument("--workers", type=int, default=6)
    ap.add_argument("--limit", type=int, default=3000)
    ap.add_argument("--threshold", type=int, default=150,
                    help="|cp| >= T counts as a decisive label (oracle frame)")
    ap.add_argument("--out", required=True)
    ap.add_argument("--tag", default="")
    args = ap.parse_args()

    tiers = [int(x) for x in args.nodes.split(",")]
    outdir = Path(args.out)
    outdir.mkdir(parents=True, exist_ok=True)
    frame = "archon-internal" if args.net else "sf-printed-cp"

    # --- label at every tier (cached, audit mode so the position SET is fixed)
    tsvs = {}
    thr = {}
    for n in tiers:
        p = outdir / f"tier-{n}.tsv"
        if not p.exists():
            cmd = [str(PY), str(LABELLER), args.positions, "--out", str(p),
                   "--oracle", args.oracle, "--nodes", str(n),
                   "--workers", str(args.workers), "--no-filter"]
            if args.limit:
                cmd += ["--limit", str(args.limit)]
            if args.net:
                cmd += ["--net", args.net]
            sys.stderr.write(f"[tier {n}] labelling...\n")
            r = subprocess.run(cmd, capture_output=True, text=True)
            if r.returncode != 0:
                sys.stderr.write(r.stderr[-3000:])
                return 2
        st = json.loads(Path(str(p) + ".stats.json").read_text())
        thr[n] = st["throughput"]
        tsvs[n] = load_tier(p)
        sys.stderr.write(f"[tier {n}] {len(tsvs[n])} rows, "
                         f"{thr[n]['pos_per_s']} pos/s\n")

    deepest = max(tiers)

    # --- paired set: quiet at EVERY tier, and scored at every tier ------------
    common = set(tsvs[deepest])
    for n in tiers:
        common &= {f for f, d in tsvs[n].items()
                   if d["raw"] is not None and not d["reason"]}
    common = sorted(common)
    sys.stderr.write(f"paired set: {len(common)} positions quiet+scored at all "
                     f"{len(tiers)} tiers\n")

    rmap = {"1.0": 1.0, "0.5": 0.5, "0.0": 0.0}
    report = dict(
        positions_file=args.positions, tag=args.tag,
        oracle=args.oracle, net=args.net, label_frame=frame,
        threshold_cp=args.threshold, tiers=tiers,
        paired_positions=len(common), tier_rowcounts={n: len(tsvs[n]) for n in tiers},
        results={}, null={}, slices={},
    )

    for n in tiers:
        d = tsvs[n]
        items = [(d[f]["raw"], rmap[d[f]["result"]]) for f in common
                 if d[f]["result"] in rmap]
        dec, contra = disagreement(items, args.threshold)
        hw, centre = wilson(contra, dec)

        # tier agreement vs deepest
        agree = sum(1 for f in common if d[f]["best"] == tsvs[deepest][f]["best"])
        deltas = sorted(abs(d[f]["raw"] - tsvs[deepest][f]["raw"]) for f in common)
        med = deltas[len(deltas) // 2] if deltas else 0

        # filter drop rate AT THIS TIER (moves with the tier: a deeper search
        # can return a capture where a shallow one returned a quiet move)
        allrows = tsvs[n]
        dropped = sum(1 for v in allrows.values() if v["reason"])

        report["results"][n] = dict(
            decisive=dec, contradicted=contra,
            disagreement_pct=round(100 * contra / max(dec, 1), 2),
            disagreement_ci95=hw,
            bestmove_agree_with_deepest_pct=round(100 * agree / max(len(common), 1), 2),
            median_abs_delta_vs_deepest_cp=med,
            d46_drop_pct_at_tier=round(100 * dropped / max(len(allrows), 1), 2),
            throughput=thr[n],
        )

        # --- SAMPLING NULL: same tier, two disjoint halves -------------------
        h0 = [(d[f]["raw"], rmap[d[f]["result"]]) for f in common
              if d[f]["result"] in rmap and half_of(f) == 0]
        h1 = [(d[f]["raw"], rmap[d[f]["result"]]) for f in common
              if d[f]["result"] in rmap and half_of(f) == 1]
        d0, c0 = disagreement(h0, args.threshold)
        d1, c1 = disagreement(h1, args.threshold)
        r0 = 100 * c0 / max(d0, 1)
        r1 = 100 * c1 / max(d1, 1)
        report["null"][n] = dict(
            half0=dict(decisive=d0, pct=round(r0, 2)),
            half1=dict(decisive=d1, pct=round(r1, 2)),
            half_spread_pp=round(abs(r0 - r1), 2),
        )

    # --- slices at the deepest and a mid tier --------------------------------
    for n in tiers:
        d = tsvs[n]
        sl = {}
        for name, pred in (
            ("pieces_le12", lambda f: piece_count(f) <= 12),
            ("pieces_13_20", lambda f: 13 <= piece_count(f) <= 20),
            ("pieces_ge21", lambda f: piece_count(f) >= 21),
            ("term_normal", lambda f: d[f]["term"] == "normal"),
            ("term_adjudication", lambda f: d[f]["term"] == "adjudication"),
            ("near_end_le20", lambda f: d[f]["pte"] <= 20),
            ("far_from_end_gt60", lambda f: d[f]["pte"] > 60),
        ):
            items = [(d[f]["raw"], rmap[d[f]["result"]]) for f in common
                     if d[f]["result"] in rmap and pred(f)]
            dec, contra = disagreement(items, args.threshold)
            hw, _ = wilson(contra, dec)
            sl[name] = dict(n=len(items), decisive=dec,
                            disagreement_pct=round(100 * contra / max(dec, 1), 2),
                            ci95=hw)
        report["slices"][n] = sl

    outp = outdir / "report.json"
    outp.write_text(json.dumps(report, indent=2))

    # --- human-readable table -------------------------------------------------
    print(f"\n=== label quality vs oracle nodes  ({args.tag or args.oracle}) ===")
    print(f"paired positions: {len(common)}   threshold |cp|>={args.threshold} "
          f"({frame})\n")
    print(f"{'nodes':>8} {'disagree%':>10} {'+-95%':>7} {'nullspread':>11} "
          f"{'bmAgree%':>9} {'medD':>6} {'drop%':>6} {'pos/s':>8}")
    for n in tiers:
        r = report["results"][n]
        print(f"{n:>8} {r['disagreement_pct']:>10.2f} {r['disagreement_ci95']:>7.2f} "
              f"{report['null'][n]['half_spread_pp']:>11.2f} "
              f"{r['bestmove_agree_with_deepest_pct']:>9.1f} "
              f"{r['median_abs_delta_vs_deepest_cp']:>6} "
              f"{r['d46_drop_pct_at_tier']:>6.1f} "
              f"{r['throughput']['pos_per_s']:>8.1f}")
    print("\nnullspread = |half0 - half1| at the SAME tier = sampling noise floor.")
    print("A tier-to-tier change smaller than the local nullspread is NOT resolvable.")
    print(f"\nreport -> {outp}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
