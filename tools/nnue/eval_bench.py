#!/usr/bin/env python3
"""eval_bench — how does a NET ALONE stack up against other evaluators?

Scores static evaluators against a deep-search oracle on a shared position
set. The point is to measure EVAL QUALITY independent of search, and
independent of each engine's arbitrary internal units.

    # one-shot: extract positions, label them, compare six evaluators
    python3 tools/nnue/eval_bench.py \
      --positions-from-pgn 'data/nnue/experiments/*/games.pgn' --count 3000 \
      --oracle engines/bin/stockfish18 --oracle-nodes 50000 \
      --eval 'd47h=archon:data/nnue/nets/screens/d47h-anneal-sb150-nocpol.cenn' \
      --eval 'SF18=sf:engines/bin/stockfish18' \
      --eval 'SF16=sf:engines/bin/stockfish16' \
      --eval 'Reckless=reckless:engines/bin/reckless' \
      --worst 5

Everything is cached under --cache (default data/nnue/evalbench/) keyed by
content hash, so re-running with one new net only pays for that net.

WHY THE RESCALED COLUMN IS THE ONE THAT MATTERS
-----------------------------------------------
Every engine reports on its own internal units (measured 2026-07-25: a rook
advantage reads ~1326 in ARCHON, ~1283 in Reckless, ~498 UCI cp in SF18 —
SF's UCI cp is win-rate-normalized, `NormalizeToPawnValue`, and even differs
between SF16 and SF18). So RAW MAE across engines mostly measures unit
choice, not quality. We therefore fit least squares `target ~= a*eval + b`
over the whole set and report the MAE of the rescaled prediction. That
removes any linear scale/offset difference and leaves how well the evaluator
RANKS AND SHAPES positions. The fitted slope `a` is how much you must
multiply that engine's output by to reach the oracle's scale (a < 1 => its
evals are LARGER in magnitude than the oracle's).

POV: oracle labels are UCI cp from the side to move. `sf` and `reckless`
adapters emit white-POV and are negated on black-to-move; `archon`
(nnue-eval) is already stm-relative. Getting this wrong silently halves
every correlation, so the adapters declare it explicitly.

READ THE CAVEATS the report prints. Two matter most: positions drawn from
our own games flatter our nets, and using SF18's search as the oracle
flatters SF18's static eval (same network's knowledge). They partly offset;
neither is neutral. Run with several --positions sources before believing a
small gap. And note the standing finding (2026-07-25) that across four
ARCHON nets spanning 86 Elo, this class of metric ordered them WRONG — so
treat it as "is this eval in the right class", never as an Elo predictor.
"""

from __future__ import annotations

import argparse
import glob
import hashlib
import json
import os
import random
import re
import subprocess
import sys
from pathlib import Path

ROOT = Path(__file__).resolve().parents[2]
CLIP = 800.0


# --------------------------------------------------------------------------
# evaluator adapters: name -> (build stdin, parse stdout, white_pov?)
# --------------------------------------------------------------------------
def _run(binary: str, script: str, timeout: int = 7200) -> list[str]:
    p = subprocess.run([binary], input=script, capture_output=True, text=True,
                       timeout=timeout)
    return p.stdout.splitlines()


def eval_archon(spec: str, fens: list[str]) -> list[float | None]:
    """archon:<net.cenn> — engine `nnue-eval`, one value per line, stm-POV."""
    net = spec
    engine = os.environ.get("ARCHON_BIN", str(ROOT / "engines/bin/archon-d47"))
    p = subprocess.run([engine, "nnue-eval", net], input="\n".join(fens) + "\n",
                       capture_output=True, text=True, timeout=7200)
    out = []
    for line in p.stdout.splitlines():
        tok = line.split("\t")[0].strip()
        out.append(float(tok) if re.fullmatch(r"-?\d+", tok) else None)
    return out


def eval_sf(spec: str, fens: list[str]) -> list[float | None]:
    """sf:<binary> — `eval`, reads 'Final evaluation' in pawns, white-POV.
    In-check positions print 'none' and yield None."""
    script = "uci\n" + "".join(f"position fen {f}\neval\nisready\n" for f in fens) + "quit\n"
    out, pending = [], None
    for line in _run(spec, script):
        if line.startswith("Final evaluation"):
            m = re.search(r"([-+]?\d+\.\d+)", line)
            pending = float(m.group(1)) * 100 if m else None
        elif line.startswith("readyok"):
            out.append(pending)
            pending = None
    return out


def eval_reckless(spec: str, fens: list[str]) -> list[float | None]:
    """reckless:<binary> — `eval`, bare integer, white-POV. NOTE: Reckless
    does not answer `isready`, so we rely on strict output ordering."""
    script = "uci\n" + "".join(f"position fen {f}\neval\n" for f in fens) + "quit\n"
    return [float(l) for l in _run(spec, script) if re.fullmatch(r"-?\d+", l.strip())]


ADAPTERS = {
    "archon": (eval_archon, False),    # (fn, white_pov)
    "sf": (eval_sf, True),
    "reckless": (eval_reckless, True),
}


# --------------------------------------------------------------------------
# positions + oracle
# --------------------------------------------------------------------------
def positions_from_pgn(pattern: str, count: int, min_ply: int, seed: int) -> list[str]:
    import chess
    import chess.pgn
    rng = random.Random(seed)
    files = sorted(glob.glob(pattern))
    if not files:
        sys.exit(f"no PGNs matched {pattern!r}")
    keep, rate = [], 0.06
    for path in files:
        with open(path, errors="ignore") as fh:
            while len(keep) < count * 12:
                g = chess.pgn.read_game(fh)
                if g is None:
                    break
                b, ply = g.board(), 0
                for mv in g.mainline_moves():
                    b.push(mv)
                    ply += 1
                    if (ply >= min_ply and not b.is_check()
                            and b.legal_moves.count() > 2 and rng.random() < rate):
                        keep.append(b.fen())
        if len(keep) >= count * 12:
            break
    rng.shuffle(keep)
    return keep[:count]


def label_with_oracle(binary: str, nodes: int, fens: list[str]) -> list[float | None]:
    """Deep-search labels, stm-relative UCI cp. Mates map to +/-(30000-10*n)."""
    script = ("uci\nsetoption name Threads value 1\n"
              + "".join(f"position fen {f}\ngo nodes {nodes}\n" for f in fens)
              + "quit\n")
    out, last = [], None
    for line in _run(binary, script):
        m = re.search(r"score cp (-?\d+)", line)
        if m:
            last = float(m.group(1))
        else:
            m = re.search(r"score mate (-?\d+)", line)
            if m:
                n = int(m.group(1))
                last = 30000.0 - n * 10 if n > 0 else -30000.0 - n * 10
        if line.startswith("bestmove"):
            out.append(last)
            last = None
    return out


# --------------------------------------------------------------------------
def cache_path(cache: Path, kind: str, key: str) -> Path:
    return cache / f"{kind}-{hashlib.sha256(key.encode()).hexdigest()[:16]}.json"


def cached(cache: Path, kind: str, key: str, produce):
    p = cache_path(cache, kind, key)
    if p.exists():
        return json.loads(p.read_text())
    val = produce()
    p.parent.mkdir(parents=True, exist_ok=True)
    p.write_text(json.dumps(val))
    return val


def main() -> int:
    ap = argparse.ArgumentParser(description=__doc__,
                                 formatter_class=argparse.RawDescriptionHelpFormatter)
    src = ap.add_mutually_exclusive_group(required=True)
    src.add_argument("--positions", help="file of FENs, one per line")
    src.add_argument("--positions-from-pgn", help="glob of PGNs to sample from")
    ap.add_argument("--count", type=int, default=3000)
    ap.add_argument("--min-ply", type=int, default=12)
    ap.add_argument("--seed", type=int, default=20260725)
    ap.add_argument("--oracle", default=str(ROOT / "engines/bin/stockfish18"))
    ap.add_argument("--oracle-nodes", type=int, default=50000)
    ap.add_argument("--labels", help="pre-labelled 'FEN<TAB>cp' file; skips the oracle "
                                     "entirely. Use for corpora already labelled in bulk.")
    ap.add_argument("--labels-white-pov", action="store_true",
                    help="the --labels file is WHITE-relative, not side-to-move. "
                         "canon-v1 corpus.quiet.txt IS white-POV (verified 2026-07-25): "
                         "reading it as stm collapses every correlation to ~0.04.")
    ap.add_argument("--eval", action="append", required=True,
                    metavar="NAME=KIND:SPEC", help="repeatable; kind in " + "/".join(ADAPTERS))
    ap.add_argument("--cache", default=str(ROOT / "data/nnue/evalbench"))
    ap.add_argument("--clip", type=float, default=CLIP)
    ap.add_argument("--worst", type=int, default=0, help="dump N worst offenders per evaluator")
    ap.add_argument("--json-out")
    a = ap.parse_args()

    import numpy as np
    cache = Path(a.cache)

    # ---- positions ----------------------------------------------------
    if a.positions:
        fens = [l.strip() for l in open(a.positions) if l.strip()]
    else:
        key = f"pgn|{a.positions_from_pgn}|{a.count}|{a.min_ply}|{a.seed}"
        fens = cached(cache, "pos", key,
                      lambda: positions_from_pgn(a.positions_from_pgn, a.count,
                                                 a.min_ply, a.seed))
    fens = fens[:a.count]
    pos_key = hashlib.sha256("\n".join(fens).encode()).hexdigest()[:16]
    print(f"positions: {len(fens)}  (set {pos_key})", file=sys.stderr)

    # ---- oracle labels (the expensive part; always cached) --------------
    if a.labels:
        tbl = {}
        for line in open(a.labels):
            parts = line.rstrip("\n").split("\t")
            if len(parts) >= 2 and parts[1]:
                tbl[parts[0]] = float(parts[1])
        lab = [tbl.get(f) for f in fens]
        if a.labels_white_pov:
            lab = [None if v is None else (-v if f.split()[1] == "b" else v)
                   for f, v in zip(fens, lab)]
        miss = sum(x is None for x in lab)
        print(f"oracle: pre-labelled {Path(a.labels).name}"
              + (f"  ({miss} unlabelled)" if miss else ""), file=sys.stderr)
    else:
        lab = cached(cache, "lab", f"{pos_key}|{a.oracle}|{a.oracle_nodes}",
                     lambda: label_with_oracle(a.oracle, a.oracle_nodes, fens))
        print(f"oracle: {Path(a.oracle).name} @{a.oracle_nodes} nodes", file=sys.stderr)

    # ---- evaluators -----------------------------------------------------
    black = np.array([f.split()[1] == "b" for f in fens])
    series = {}
    for item in a.eval:
        name, _, rest = item.partition("=")
        kind, _, spec = rest.partition(":")
        if kind not in ADAPTERS:
            sys.exit(f"unknown adapter {kind!r}; have {list(ADAPTERS)}")
        fn, white_pov = ADAPTERS[kind]
        vals = cached(cache, "ev", f"{pos_key}|{kind}|{spec}", lambda: fn(spec, fens))
        if len(vals) != len(fens):
            print(f"  !! {name}: got {len(vals)} values for {len(fens)} positions — SKIPPED",
                  file=sys.stderr)
            continue
        v = np.array([np.nan if x is None else float(x) for x in vals])
        if white_pov:                      # white-POV -> side-to-move relative
            v = np.where(black, -v, v)
        series[name] = v

    y = np.array([np.nan if x is None else float(x) for x in lab])
    ok = ~np.isnan(y)
    for v in series.values():
        ok &= ~np.isnan(v)
    print(f"usable: {int(ok.sum())}/{len(fens)}\n", file=sys.stderr)

    # ---- basis split (2026-07-26 trap, third strike) ---------------------
    # The value head is NEVER consulted in check during play
    # (src/search/mod.rs:2327: raw_eval=VALUE_NONE, static_eval inherited).
    # A metric over a set that mixes in-check rows is NOT comparable to one
    # over a non-check set — quoting either without basis+n produced a
    # phantom "sharp erosion" verdict once. When in-check rows exist, this
    # report prints BOTH tables, each labeled; the NON-CHECK one is the
    # play-relevant basis.
    try:
        import chess as _chess
        chk = np.array([_chess.Board(f).is_check() for f in fens])
    except Exception:
        chk = np.zeros(len(fens), bool)
        print("!! python-chess unavailable — in-check basis split SKIPPED; "
              "treat every number below as mixed-basis", file=sys.stderr)
    n_chk = int((chk & ok).sum())
    bases = [("all", ok)]
    if n_chk:
        bases = [("non-check (PLAY-RELEVANT)", ok & ~chk), ("all (mixed!)", ok)]
        print(f"!! BASIS: {n_chk}/{int(ok.sum())} usable positions are IN CHECK, where\n"
              f"   the value head is never consulted in play. Never quote a number\n"
              f"   from this report without its basis label and n.\n", file=sys.stderr)

    # ---- metrics ---------------------------------------------------------
    rng = np.random.default_rng(0)
    out = {}
    for basis, bm in bases:
        nb = int(bm.sum())
        c = np.clip(y[bm], -a.clip, a.clip)
        boot = rng.integers(0, nb, size=(400, nb))
        rows = []
        primary = basis == bases[0][0]
        for name, v in series.items():
            e = np.clip(v[bm], -a.clip, a.clip)
            slope, icpt = np.polyfit(e, c, 1)
            r = slope * e + icpt
            resc = np.abs(r - c)
            ci = np.percentile([resc[b].mean() for b in boot], [2.5, 97.5])
            rows.append((name, np.mean(np.abs(e - c)), resc.mean(), ci, slope,
                         float(np.corrcoef(e, c)[0, 1]),
                         100 * float(np.mean(np.sign(e) == np.sign(c))),
                         float(np.percentile(resc, 90)), float(np.percentile(resc, 99))))
            key = name if primary else f"{name}@{basis.split()[0]}"
            out[key] = dict(basis=basis, n=nb, raw_mae=rows[-1][1],
                            rescaled_mae=resc.mean(), ci95=list(ci), slope=slope,
                            pearson=rows[-1][5], sign_pct=rows[-1][6],
                            p90=rows[-1][7], p99=rows[-1][8])
            if a.worst and primary:
                idx = np.argsort(-resc)[:a.worst]
                out[key]["worst"] = [
                    dict(fen=[f for f, k in zip(fens, bm) if k][int(i)],
                         label=float(c[i]), eval=float(e[i]), rescaled=float(r[i]))
                    for i in idx]

        if primary and rows and max(abs(r[5]) for r in rows) < 0.30:
            print("\n!! WARNING: EVERY evaluator correlates < 0.30 with the labels.\n"
                  "   That is a data-alignment bug, not six bad evaluators. Check:\n"
                  "   (a) label POV — try --labels-white-pov (canon-v1 IS white-POV);\n"
                  "   (b) FEN/label row alignment; (c) label units.\n", file=sys.stderr)

        rows.sort(key=lambda t: t[2])
        print(f"── basis: {basis}  n={nb} ──")
        print(f"{'evaluator':22s} {'rawMAE':>7s} {'RESCALED':>9s} {'95% CI':>15s} "
              f"{'slope':>7s} {'pearson':>8s} {'sign%':>6s} {'p90':>6s} {'p99':>6s}")
        for n, raw, rs, ci, sl, pr, sg, p90, p99 in rows:
            print(f"{n:22s} {raw:7.1f} {rs:9.1f} {f'[{ci[0]:.1f},{ci[1]:.1f}]':>15s} "
                  f"{sl:7.3f} {pr:8.4f} {sg:6.1f} {p90:6.1f} {p99:6.1f}")
        print()

    print("\nCAVEATS: (1) positions from our own games flatter our nets; an oracle")
    print("that is one engine's search flatters that engine's static eval. Run")
    print("several position sources. (2) This metric ordered four ARCHON nets")
    print("spanning 86 Elo WRONG (2026-07-25) — it says 'right class', not 'stronger'.")

    if a.worst:
        for name in out:
            print(f"\nworst {a.worst} for {name}:")
            for w in out[name]["worst"]:
                print(f"  label {w['label']:+7.0f}  pred {w['rescaled']:+7.0f}   {w['fen']}")
    if a.json_out:
        Path(a.json_out).write_text(json.dumps(out, indent=2))
        print(f"\nwrote {a.json_out}", file=sys.stderr)
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
