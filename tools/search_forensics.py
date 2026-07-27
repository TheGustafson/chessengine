#!/usr/bin/env python3
"""Search-internals forensics: why is our tree wider and our value noisier?

Every probe here started as a throwaway shell one-liner during the
2026-07-27 investigation. They are collected because each answered a question
that recurred, and because several of the answers CONTRADICTED the obvious
first reading — which is exactly the kind of thing that must be re-runnable
rather than remembered.

Subcommands
  ladder POS      per-iteration trace for both engines: eval, best move,
                  nodes, PV. Reads the divergence as a concrete line.
  research        aspiration re-search rate, ours (from ARCHON_SEARCH_STATS)
                  and Stockfish's (inferred from its info lines, since it
                  publishes one per fail and we do not).
  ebf             effective branching factor per ply, both engines.
  qshare          qsearch share of nodes by depth.
  aspiration      sweep the aspiration window and watch re-search rate,
                  node count and first-move cutoff move together.

TRAPS THIS TOOL ENCODES, all of which cost time on 2026-07-27:

  * Our engine treats stdin EOF as a STOP, so piping `go depth N` and closing
    the pipe yields a bestmove with no info lines at all. stdin must be held
    open past the search.
  * The UCI `nodes` field is CUMULATIVE. Reading consecutive iterations as
    per-iteration counts turns a normal branching factor of ~2.5 into an
    apparent "3.5x node explosion". It is not one.
  * We do not publish info lines on aspiration fails; Stockfish does. So
    counting info lines measures re-searches for SF and nothing for us, while
    ARCHON_SEARCH_STATS measures it for us and nothing for SF. Each engine
    needs its own instrument.
  * Our UCI cp is RAW INTERNAL (~0.64x SF18's frame). Any cross-engine eval
    comparison must convert first.

    python3 tools/search_forensics.py research
    python3 tools/search_forensics.py ladder "FEN" --max-depth 14
"""

from __future__ import annotations

import argparse
import os
import re
import subprocess
import sys

ARCHON = "target/release/chessengine"
NET = "data/nnue/nets/screens/d47v-a4b2-cal400.cenn"
SF18 = "engines/bin/stockfish18"
FENS = "tools/nnue/calib/scale-check-800.fens"
OUR_SCALE = 0.64


def bench_stats(depth: int, ov: dict[str, int] | None = None, fens: str = FENS) -> dict:
    """Aggregate ARCHON_SEARCH_STATS counters over a whole bench."""
    lines = [f"setoption name EvalFile value {NET}"]
    lines += [f"setoption name {k} value {v}" for k, v in (ov or {}).items()]
    lines += [f"bench 16 1 {depth} {fens} depth", "quit"]
    r = subprocess.run([ARCHON], input="\n".join(lines) + "\n", text=True,
                       capture_output=True,
                       env={**os.environ, "ARCHON_SEARCH_STATS": "1"})
    tot: dict[str, int] = {}
    for key in ("nodes", "qnodes", "depth", "fh", "fhf", "research", "ttcut"):
        tot[key] = sum(int(x) for x in re.findall(rf"\b{key}=(\d+)", r.stderr))
    return tot


def hold_open(binary: str, script: str, wait: int) -> str:
    """Run a UCI script with stdin held open past the search.

    Closing stdin stops our search immediately, which silently yields an
    empty info stream — the single most time-wasting trap in this file.
    """
    return subprocess.run(
        ["bash", "-c", f'{{ printf %s "$SCRIPT"; sleep {wait}; }} | timeout {wait + 30} {binary}'],
        capture_output=True, text=True, env={**os.environ, "SCRIPT": script},
    ).stdout


def cmd_research(args):
    print("aspiration re-searches per ID iteration\n")
    t = bench_stats(args.depth)
    print(f"  ARCHON (counters, depth {args.depth} bench): "
          f"{t['research'] / max(t['depth'], 1):.2f}")
    print(f"    over {t['depth']:,} iterations, {t['nodes']:,} nodes\n")
    if args.fen:
        out = hold_open(SF18,
                        f"uci\nisready\nposition fen {args.fen}\ngo depth {args.sf_depth}\n",
                        args.wait)
        per: dict[str, int] = {}
        for line in out.splitlines():
            if line.startswith("info depth") and "currmove" not in line:
                d = line.split()[2]
                per[d] = per.get(d, 0) + 1
        if per:
            extra = sum(v - 1 for v in per.values()) / len(per)
            print(f"  SF18 (info lines, one position): {extra:.2f}")
            print("    Stockfish publishes an info line per aspiration fail;")
            print("    we do not, so the two must be measured differently.")


def cmd_ebf(args):
    depths = [int(d) for d in args.depths.split(",")]
    print(f"{'depth':>6}{'ARCHON':>12}{'SF18':>12}{'ours/theirs':>13}")
    prev = {}
    import chess
    import chess.engine
    fens = [l.strip() for l in open(args.positions) if l.strip()][: args.limit]
    for d in depths:
        tot = {}
        for name, binary, opts in (("a", ARCHON, {"EvalFile": NET}), ("s", SF18, {})):
            eng = chess.engine.SimpleEngine.popen_uci(binary)
            try:
                for k, v in opts.items():
                    eng.configure({k: v})
                n = 0
                for f in fens:
                    info = eng.analyse(chess.Board(f), chess.engine.Limit(depth=d))
                    n += info.get("nodes") or 0
            finally:
                eng.quit()
            tot[name] = n
        r = tot["a"] / tot["s"] if tot["s"] else float("nan")
        print(f"{d:>6}{tot['a']:>12,}{tot['s']:>12,}{r:>13.2f}")
        if prev:
            dd = d - prev["d"]
            print(f"{'':>6}{'  EBF/ply':>12}"
                  f"{(tot['a']/prev['a'])**(1/dd):>7.2f}"
                  f"{(tot['s']/prev['s'])**(1/dd):>12.2f}")
        prev = {"d": d, "a": tot["a"], "s": tot["s"]}
    print("\nEqual EBF with a ratio > 1 means a CONSTANT overhead, not a")
    print("compounding one — look at shallow depths and qsearch, not pruning.")


def cmd_qshare(args):
    print(f"{'depth':>6}{'nodes':>13}{'qnodes':>13}{'qshare':>9}")
    for d in [int(x) for x in args.depths.split(",")]:
        t = bench_stats(d)
        print(f"{d:>6}{t['nodes']:>13,}{t['qnodes']:>13,}"
              f"{100 * t['qnodes'] / max(t['nodes'], 1):>8.1f}%")


def cmd_aspiration(args):
    print(f"{'BASE_DELTA':>11}{'MSS_DIV':>9}{'nodes':>13}{'research/it':>13}{'fhf%':>8}")
    for spec in args.grid.split(","):
        base, _, div = spec.partition("/")
        ov = {"ASPIRATION_BASE_DELTA": int(base)}
        if div:
            ov["ASPIRATION_MSS_DIVISOR"] = int(div)
        t = bench_stats(args.depth, ov)
        print(f"{base:>11}{div or '-':>9}{t['nodes']:>13,}"
              f"{t['research'] / max(t['depth'], 1):>13.2f}"
              f"{100 * t['fhf'] / max(t['fh'], 1):>8.2f}")
    print("\nA wider window trades re-searches for a looser first search.")
    print("If nodes FALL as the window widens, aspiration is costing more")
    print("than it saves at that setting.")


def cmd_ladder(args):
    sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
    from tree_walk import ladder  # noqa: E402
    ours = ladder(ARCHON, args.pos, args.max_depth, {"EvalFile": NET}, OUR_SCALE)
    theirs = ladder(SF18, args.pos, args.max_depth)
    print(f"{'d':>3} | {'ARCHON':>7} {'move':>6} {'nodes':>9} | "
          f"{'SF18':>7} {'move':>6} {'nodes':>9}")
    for d in sorted(ours):
        oe, om, _, on = ours[d]
        te, tm, _, tn = theirs.get(d, (None, None, "", None))
        print(f"{d:>3} | {'' if oe is None else f'{oe:+7.2f}'} {om or '-':>6} "
              f"{on or 0:>9,} | {'' if te is None else f'{te:+7.2f}'} "
              f"{tm or '-':>6} {tn or 0:>9,}")
    print("\nNOTE: `nodes` is CUMULATIVE. Consecutive-iteration ratios are the")
    print("branching factor (~2.5 is normal), NOT node explosions.")


def main() -> None:
    ap = argparse.ArgumentParser()
    sub = ap.add_subparsers(dest="cmd", required=True)

    p = sub.add_parser("research"); p.set_defaults(fn=cmd_research)
    p.add_argument("--depth", type=int, default=11)
    p.add_argument("--fen"); p.add_argument("--sf-depth", type=int, default=16)
    p.add_argument("--wait", type=int, default=25)

    p = sub.add_parser("ebf"); p.set_defaults(fn=cmd_ebf)
    p.add_argument("positions", nargs="?", default=FENS)
    p.add_argument("--depths", default="8,10,12,14"); p.add_argument("--limit", type=int, default=80)

    p = sub.add_parser("qshare"); p.set_defaults(fn=cmd_qshare)
    p.add_argument("--depths", default="4,6,8,10,12")

    p = sub.add_parser("aspiration"); p.set_defaults(fn=cmd_aspiration)
    p.add_argument("--grid", default="5,10,20,40,200,100000")
    p.add_argument("--depth", type=int, default=11)

    p = sub.add_parser("ladder"); p.set_defaults(fn=cmd_ladder)
    p.add_argument("pos"); p.add_argument("--max-depth", type=int, default=14)

    args = ap.parse_args()
    args.fn(args)


if __name__ == "__main__":
    main()
