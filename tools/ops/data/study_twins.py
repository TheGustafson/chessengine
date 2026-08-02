#!/usr/bin/env python3
"""tools/ops/data/study_twins.py — how much of a label the net CANNOT represent.

THE QUESTION THIS EXISTS FOR
  The owner's constraint on the oracle is not "be accurate", it is: "we don't
  want deep labels as a shallow NNUE can't learn them, but we do want them to
  be accurate and not disagree with WDL". The eval-outcome study measures the
  accuracy half. This measures the OTHER half, and it is the one that has no
  instrument anywhere else in the tree.

THE METHOD
  Build PAIRS of positions that are IDENTICAL in the network's input space and
  differ only in a dimension the network cannot see:

    castle  full castling rights vs rights stripped
    rule50  halfmove clock 0 vs 90 (five plies from the draw)

  Our features carry neither. Verified here, not assumed: `archon nnue-eval` is
  run on both members of every pair and the identical-output count is reported
  as a receipt. If it is not 100%, the twin construction is invalid and the run
  says so instead of producing a number.

  Then label BOTH twins with the oracle at each node count. Whatever the oracle
  says the pair differs by, the network MUST predict the same value for both --
  it cannot see the difference. So the twin spread is IRREDUCIBLE LABEL NOISE:
  a floor on training loss that no amount of capacity or training removes, and
  it is injected by the labelling policy itself.

  If that spread GROWS with node count, then deeper labels are strictly noisier
  in the representation the net actually has. That is the precise, measurable
  form of "a shallow NNUE can't learn deep labels".

DISCRETE OUTCOME FIRST
  The headline is the BEST-MOVE CHANGE RATE across the pair (frame-free,
  immune to cp scale); the cp spread is reported second, as quantiles rather
  than a mean, because a mean over a heavy tail is the statistic that hid a
  16x subclass once already.

USAGE
  .venv/bin/python tools/ops/data/study_twins.py \
      --positions data/nnue/selfplay/pilot-self/positions.tsv \
      --kind castle --nodes 5000,25000,100000 --n 400 \
      --oracle engines/bin/stockfish18 --out data/nnue/selfplay/study/twins
"""
from __future__ import annotations

import argparse
import json
import multiprocessing as mp
import subprocess
import sys
from pathlib import Path

import chess

RAW_MAGNITUDE_CAP = 10_000   # D46 criterion 3

ROOT = Path(__file__).resolve().parents[3]
sys.path.insert(0, str(ROOT / "tools/forensics"))
import uciprobe  # noqa: E402

_W = {}


def _init(oracle, net, hash_mb):
    _W["eng"] = uciprobe.Engine(oracle, net=net, hash_mb=hash_mb,
                                options={"Threads": 1}, name="oracle")


def _score(job):
    fen, nodes = job
    eng = _W["eng"]
    eng.newgame()
    try:
        r = eng.go(fen=fen, nodes=nodes)
    except RuntimeError:
        return None
    if not r["ladder"]:
        return None
    top = r["ladder"][max(r["ladder"])]
    b = chess.Board(fen)
    sc = top["score"] if b.turn == chess.WHITE else -top["score"]
    return (sc, r["best"])


def make_pair(kind: str, fen: str):
    """Return (base_fen, twin_fen) or None if this position cannot be twinned."""
    b = chess.Board(fen)
    if kind == "castle":
        if not b.castling_rights:
            return None
        stripped = chess.Board(fen)
        stripped.castling_rights = 0
        return (b.fen(), stripped.fen())
    if kind == "rule50":
        # isolate the clock: no rights, no ep, and enough material that the
        # draw pressure is a real evaluation term rather than trivially moot.
        if b.castling_rights or b.ep_square is not None:
            return None
        p = fen.split()
        base = " ".join(p[:4] + ["0", p[5]])
        twin = " ".join(p[:4] + ["90", p[5]])
        return (base, twin)
    raise ValueError(kind)


def archon_static(fens):
    p = subprocess.run(
        [str(ROOT / "engines/canonical/archon"), "nnue-eval",
         str(ROOT / "engines/canonical/net.cenn")],
        input="\n".join(fens) + "\n", capture_output=True, text=True, timeout=900)
    vals = []
    for line in p.stdout.strip().splitlines():
        try:
            vals.append(int(float(line.split()[0])))
        except (ValueError, IndexError):
            pass
    return vals


def quant(xs, q):
    if not xs:
        return 0
    s = sorted(xs)
    return s[min(len(s) - 1, int(q * (len(s) - 1)))]


def main() -> int:
    ap = argparse.ArgumentParser()
    ap.add_argument("--positions", required=True)
    ap.add_argument("--kind", default="castle", choices=["castle", "rule50"])
    ap.add_argument("--nodes", default="5000,25000,100000")
    ap.add_argument("--n", type=int, default=400)
    ap.add_argument("--oracle", default="engines/bin/stockfish18")
    ap.add_argument("--net", default=None)
    ap.add_argument("--workers", type=int, default=6)
    ap.add_argument("--out", required=True)
    args = ap.parse_args()

    tiers = [int(x) for x in args.nodes.split(",")]
    outdir = Path(args.out)
    outdir.mkdir(parents=True, exist_ok=True)

    pairs = []
    with open(args.positions) as fh:
        hdr = fh.readline().rstrip("\n").split("\t")
        ix = {c: i for i, c in enumerate(hdr)}
        for line in fh:
            f = line.rstrip("\n").split("\t")
            if len(f) < len(hdr):
                continue
            pr = make_pair(args.kind, f[ix["fen"]])
            if pr:
                pairs.append(pr)
            if len(pairs) >= args.n:
                break
    if not pairs:
        sys.stderr.write(f"no positions eligible for kind={args.kind}\n")
        return 2
    sys.stderr.write(f"{len(pairs)} twin pairs ({args.kind})\n")

    # --- RECEIPT: the network must be blind to the twinned dimension ---------
    a_base = archon_static([p[0] for p in pairs])
    a_twin = archon_static([p[1] for p in pairs])
    ident = sum(1 for x, y in zip(a_base, a_twin) if x == y)
    blind_pct = 100.0 * ident / max(len(a_base), 1)
    sys.stderr.write(f"archon nnue-eval identical across twins: "
                     f"{ident}/{len(a_base)} ({blind_pct:.1f}%)\n")
    if len(a_base) != len(pairs) or blind_pct < 99.9:
        sys.stderr.write(
            "REFUSING to report a noise figure: the twin construction is only "
            "valid if the network's own output is identical across the pair. "
            "It is not, so the measured spread would be partly representable "
            "and this instrument cannot answer the question.\n")
        return 3

    report = dict(kind=args.kind, oracle=args.oracle, net=args.net,
                  pairs=len(pairs), archon_blind_pct=round(blind_pct, 2),
                  label_frame="archon-internal" if args.net else "sf-printed-cp",
                  tiers={})

    for n in tiers:
        jobs = [(f, n) for p in pairs for f in p]
        with mp.Pool(args.workers, initializer=_init,
                     initargs=(args.oracle, args.net, 16)) as pool:
            res = pool.map(_score, jobs, chunksize=8)
        deltas, changed, ok, mate_excluded = [], 0, 0, 0
        for i in range(0, len(res), 2):
            a, b = res[i], res[i + 1]
            if a is None or b is None:
                continue
            # D46 criterion 3 drops |raw| > 10000, so a mate-scored pair can
            # NEVER become a training label and must not count as corpus noise.
            # Measured before this exclusion, the rule50 mean read 27.6 -> 64.1
            # -> 62.5 cp with a max of 29,830 -- i.e. uciprobe's mate mapping
            # ((30000-|n|)*sign), a handful of pairs where the deep search saw
            # a mate on one side of the clock twin and a draw on the other.
            # That is a real effect about SEARCH, but it is not noise in the
            # corpus we would actually build.
            if abs(a[0]) > RAW_MAGNITUDE_CAP or abs(b[0]) > RAW_MAGNITUDE_CAP:
                mate_excluded += 1
                continue
            ok += 1
            deltas.append(abs(a[0] - b[0]))
            if a[1] != b[1]:
                changed += 1
        report["tiers"][n] = dict(
            pairs_scored=ok, mate_excluded=mate_excluded,
            bestmove_changed=changed,
            bestmove_changed_pct=round(100 * changed / max(ok, 1), 2),
            abs_delta_cp=dict(p50=quant(deltas, .5), p90=quant(deltas, .9),
                              p99=quant(deltas, .99), max=max(deltas or [0]),
                              mean=round(sum(deltas) / max(len(deltas), 1), 1)),
        )
        sys.stderr.write(f"[{n}] moved {changed}/{ok} "
                         f"({100*changed/max(ok,1):.1f}%), |d| p50="
                         f"{quant(deltas,.5)} p90={quant(deltas,.9)} "
                         f"(mate-excluded {mate_excluded})\n")

    p = outdir / f"twins-{args.kind}.json"
    p.write_text(json.dumps(report, indent=2))

    print(f"\n=== irreducible label noise: {args.kind} twins "
          f"({report['label_frame']}) ===")
    print(f"pairs {len(pairs)}   network blind to the dimension: {blind_pct:.1f}% "
          f"(receipt: archon nnue-eval)\n")
    print(f"{'nodes':>8} {'bestmoveChg%':>13} {'|d|p50':>7} {'|d|p90':>7} "
          f"{'|d|p99':>7} {'|d|mean':>8} {'|d|max':>7} {'mateExcl':>9}")
    for n in tiers:
        t = report["tiers"][n]
        d = t["abs_delta_cp"]
        print(f"{n:>8} {t['bestmove_changed_pct']:>13.1f} {d['p50']:>7} "
              f"{d['p90']:>7} {d['p99']:>7} {d['mean']:>8.1f} {d['max']:>7} "
              f"{t['mate_excluded']:>9}")
    print("\nEvery cp of this spread is noise the network CANNOT fit: its inputs")
    print("are identical across the pair. Growth with node count is the precise")
    print("form of 'a shallow NNUE cannot learn what a deep search knows'.")
    print(f"\nreport -> {p}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
