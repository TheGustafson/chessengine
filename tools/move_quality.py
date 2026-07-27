#!/usr/bin/env python3
"""Mean centipawn loss against an independent oracle — the exploration screen.

WHY THIS EXISTS. Screening tunables on (nodes at fixed depth, tactical
solutions at fixed nodes) produced a configuration that measured +32/1277
at 3.5 sigma and then read NEGATIVE in games. The screen was wrong, and it
was wrong for a specific reason worth writing down:

  * A mate suite scores BINARY hit/miss on positions that contain a forced
    win. Real games are overwhelmingly quiet positions where several moves
    are playable and the question is which is best by a little. Tactical
    hit-rate has both the wrong content and terrible variance -- one position
    is worth 1 bit.
  * Node count is not a strength measure at all. It is one input to strength
    (time-to-depth) with the other input (decision quality) unmeasured.

THE INSTRUMENT. For each position, ask an ORACLE (Stockfish, deep, MultiPV)
for the score of every plausible move. Then run the candidate at a fixed node
budget and look up what its chosen move was worth. The loss is

    loss = oracle_score(oracle's best move) - oracle_score(candidate's move)

in the ORACLE's centipawns, which is the right unit: it is real advantage
given up, not something denominated in our own compressed eval frame. Average
that over a few hundred positions and you have a continuous, low-variance
number where every position contributes a real value rather than one bit.

Three properties that make it a good exploration screen:

  * INDEPENDENT. The oracle is Stockfish, so this does not measure agreement
    with our own eval's biases -- which is exactly what using our own engine
    as its own oracle would do.
  * AMORTISED. The oracle pass is the expensive part and it runs ONCE per
    position set, cached to JSON. Every subsequent arm costs only its own
    fast searches, so exploration is cheap after the first run.
  * DETERMINISTIC. Fixed node budgets, so it reproduces exactly under load --
    which matters because the box usually has a trainer on it.

CENSORING, stated honestly. The oracle returns the top K moves. If the
candidate plays something outside that set, its true loss is unknown and only
bounded below by the K-th move's loss. Those are counted and reported
separately, and the headline mean uses the K-th score as a floor, so the
reported loss is an UNDERSTATEMENT whenever the censored rate is non-trivial.
Compare arms on censored rate as well as mean loss.

    # once per FEN set (slow):
    python3 tools/move_quality.py --build-oracle --positions 300 --oracle-depth 16
    # then per arm (fast):
    python3 tools/move_quality.py 'combo:NMP_EVAL_BASE=168,RAZOR_BASE=3000'
"""

from __future__ import annotations

import argparse
import json
import math
import os
import statistics
import sys

import chess
import chess.engine

NET = "data/nnue/nets/screens/d47v-a4b2-cal400.cenn"
ENGINE = "target/release/chessengine"
ORACLE = "engines/bin/stockfish18"
CACHE = "tools/nnue/calib/oracle-multipv.json"
FENS = "tools/nnue/calib/scale-check-800.fens"

# Families shared with arm_screen.py so an arm spec means the same thing in
# both tools.
sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
from arm_screen import FAMILIES, round_away  # noqa: E402


def parse_arm(spec: str) -> tuple[str, dict[str, int]]:
    name, _, body = spec.partition(":")
    ov: dict[str, int] = {}
    for tok in filter(None, (t.strip() for t in body.split(","))):
        if "*" in tok:
            fam, _, f = tok.partition("*")
            for k, v in FAMILIES[fam.strip()].items():
                ov[k] = round_away(v * float(f))
        else:
            k, _, v = tok.partition("=")
            ov[k.strip()] = int(v)
    return name, ov


def normalise(line: str) -> str:
    p = line.split()
    if len(p) >= 6 and p[4].lstrip("-").isdigit() and p[5].isdigit():
        return " ".join(p[:6])
    return " ".join(p[:4] + ["0", "1"])


def load_fens(path: str, limit: int) -> list[str]:
    raw = [l.strip() for l in open(path) if l.strip() and not l.startswith("#")]
    out = []
    for line in raw:
        f = normalise(line)
        b = chess.Board(f)
        # A position with one legal move measures nothing, and a finished game
        # has no move to score.
        if b.is_game_over() or len(list(b.legal_moves)) < 2:
            continue
        out.append(f)
        if len(out) >= limit:
            break
    return out


def build_oracle(fens: list[str], depth: int, multipv: int) -> dict:
    eng = chess.engine.SimpleEngine.popen_uci(ORACLE)
    table: dict[str, dict] = {}
    try:
        for i, f in enumerate(fens, 1):
            board = chess.Board(f)
            infos = eng.analyse(board, chess.engine.Limit(depth=depth), multipv=multipv)
            moves = {}
            for info in infos:
                pv = info.get("pv")
                if not pv:
                    continue
                sc = info["score"].relative
                # Mate scores get a large finite value: their exact magnitude
                # is meaningless as a cp loss, but "hugely better/worse" is
                # the correct ordering and clipping keeps one position from
                # dominating the mean.
                cp = sc.score(mate_score=2000)
                moves[pv[0].uci()] = cp
            if len(moves) >= 2:
                table[f] = moves
            if i % 25 == 0:
                print(f"  oracle {i}/{len(fens)}", flush=True)
    finally:
        eng.quit()
    return table


def score_arm(table: dict, nodes: int, ov: dict[str, int]) -> dict:
    eng = chess.engine.SimpleEngine.popen_uci(ENGINE)
    losses: list[float] = []
    censored = 0
    top1 = 0
    try:
        eng.configure({"EvalFile": NET})
        for k, v in ov.items():
            eng.configure({k: v})
        for fen, moves in table.items():
            board = chess.Board(fen)
            res = eng.play(board, chess.engine.Limit(nodes=nodes))
            if res.move is None:
                continue
            uci = res.move.uci()
            best = max(moves.values())
            worst_known = min(moves.values())
            if uci in moves:
                losses.append(best - moves[uci])
                top1 += int(moves[uci] == best)
            else:
                # Censored: only bounded below by the worst move the oracle
                # bothered to report. Understates the true loss.
                censored += 1
                losses.append(best - worst_known)
    finally:
        eng.quit()
    n = len(losses)
    return {
        "n": n,
        "mean": statistics.fmean(losses) if n else float("nan"),
        "median": statistics.median(losses) if n else float("nan"),
        "sem": (statistics.stdev(losses) / math.sqrt(n)) if n > 1 else float("nan"),
        "top1": top1 / n if n else float("nan"),
        "censored": censored / n if n else float("nan"),
    }


def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("arms", nargs="*", help="name:K=V,FAMILY*FACTOR")
    ap.add_argument("--build-oracle", action="store_true")
    ap.add_argument("--positions", type=int, default=300)
    ap.add_argument("--oracle-depth", type=int, default=16)
    ap.add_argument("--multipv", type=int, default=6)
    ap.add_argument("--nodes", type=int, default=50000)
    ap.add_argument(
        "--budgets",
        default="",
        help="comma-separated candidate node budgets -> a LOSS CURVE instead "
        "of one point. A single budget is one sample of a curve that two "
        "configs can cross: the config that loses least at 50k can lose most "
        "at 200k, and a screen that reports only 50k cannot see it. This is "
        "the failure mode that let a +32-tactics arm read negative in games.",
    )
    ap.add_argument("--fens", default=FENS)
    ap.add_argument("--cache", default=CACHE)
    args = ap.parse_args()

    if args.build_oracle:
        fens = load_fens(args.fens, args.positions)
        print(f"building oracle over {len(fens)} positions "
              f"(depth {args.oracle_depth}, MultiPV {args.multipv})")
        table = build_oracle(fens, args.oracle_depth, args.multipv)
        json.dump({"depth": args.oracle_depth, "multipv": args.multipv,
                   "table": table}, open(args.cache, "w"))
        print(f"wrote {args.cache}: {len(table)} positions")
        if not args.arms:
            return

    blob = json.load(open(args.cache))
    table = blob["table"]
    print(f"oracle: {len(table)} positions, depth {blob['depth']}, "
          f"MultiPV {blob['multipv']}; arms at {args.nodes} nodes\n")

    budgets = ([int(b) for b in args.budgets.split(",")] if args.budgets
               else [args.nodes])
    arms = [("control", {})] + [parse_arm(a) for a in args.arms]

    if len(budgets) > 1:
        # Curve mode: mean cp loss per budget, deltas vs control at the SAME
        # budget. Reading across a row shows how an arm SCALES, which is what
        # decides whether it survives a longer time control.
        base_by_budget = {}
        print(f"{'arm':<24}" + "".join(f"{b//1000:>10}k" for b in budgets))
        for name, ov in arms:
            row = f"{name:<24}"
            for b in budgets:
                r = score_arm(table, b, ov)
                if not ov:
                    base_by_budget[b] = r["mean"]
                    row += f"{r['mean']:>11.1f}"
                else:
                    row += f"{r['mean'] - base_by_budget[b]:>+11.1f}"
            print(row, flush=True)
        print("\nControl row is absolute mean cp loss; arm rows are deltas vs "
              "control at the\nSAME budget. LOWER is better, so NEGATIVE "
              "deltas are improvements. An arm whose\ndelta worsens as the "
              "budget grows is buying its gain from shallow search and\nwill "
              "not survive a longer time control.")
        return

    base = score_arm(table, args.nodes, {})
    print(f"{'arm':<24}{'mean loss':>11}{'+-sem':>8}{'delta':>9}"
          f"{'median':>8}{'top1':>8}{'cens':>7}")
    print(f"{'control':<24}{base['mean']:>11.1f}{base['sem']:>8.1f}{'':>9}"
          f"{base['median']:>8.1f}{base['top1']:>8.1%}{base['censored']:>7.1%}")
    for spec in args.arms:
        name, ov = parse_arm(spec)
        r = score_arm(table, args.nodes, ov)
        # LOWER loss is better, so a negative delta is an improvement.
        print(f"{name:<24}{r['mean']:>11.1f}{r['sem']:>8.1f}"
              f"{r['mean']-base['mean']:>+9.1f}"
              f"{r['median']:>8.1f}{r['top1']:>8.1%}{r['censored']:>7.1%}",
              flush=True)
    print("\nLOWER mean loss is better; delta is vs control. The paired sem "
          "understates\nsignificance (both arms see the same positions) -- "
          "prefer the sign and size of\ndelta over its ratio to sem, and "
          "confirm anything promising with an SPRT.")


if __name__ == "__main__":
    main()
