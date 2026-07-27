#!/usr/bin/env python3
"""One-command tunable screen: cp loss (tiered) + nodes + tactics.

    python3 tools/screen.py 'RED2600:RED_TABLE_NUM=2600' 'RFPup:RFP*1.25'
    python3 tools/screen.py --base RED_TABLE_NUM=2600 'RFPup:RFP*1.25'   # chain

THREE INSTRUMENTS, RANKED BY HOW MUCH THEY DESERVE TO BE BELIEVED.

  cp@TIER  Mean centipawn loss against SF18 root-analysis labels, per oracle
           tier. THE PRIMARY METRIC. Continuous, ~10k positions, every legal
           move scored so nothing is censored, and denominated in real
           advantage rather than our own compressed eval frame. Lower is
           better; arm rows are deltas vs control at the same tier.

           The tiers are diagnostic, not decoration: an arm that improves
           against 4k labels while worsening against 64k is trading deep
           mistakes for shallow ones. A single-tier screen calls that a wash.

  nodes    Tree size at fixed depth. A COST, not a strength. Useful only
           alongside cp loss -- on its own it rewards pruning everything.

  tactics  mate_in_3 solved at a fixed node budget. THE WEAKEST INSTRUMENT,
           kept only because it is cheap and occasionally catches a
           catastrophe. It is what produced a "+32 solutions at 3.5 sigma"
           configuration that then measured -30 Elo in games: a mate suite is
           made of positions containing a forced win, so pruning harder looks
           free there while it discards the right move in the quiet positions
           that decide games. Never promote an arm on this number.

CHAINING. `--base K=V,FAMILY*F` applies a fixed configuration to the control
AND to every arm, so each round measures the marginal effect of a change ON
TOP of what has already been accepted. That is what makes a reasoned sequence
of combined ablations possible: accept a winner, fold it into --base, and the
next round is automatically measured against it rather than against stock.

Setup, once (both cached and gitignored):
    ./target/release/obs-tool export data/obs/big-v1-sf18.ceobs \\
        | python3 tools/obs_to_oracle.py --out tools/nnue/calib/oracle-obs.json
"""

from __future__ import annotations

import argparse
import json
import os
import re
import math
import statistics
import subprocess
import sys

import chess
import chess.engine

sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
from arm_screen import FAMILIES, round_away  # noqa: E402

NET = "data/nnue/nets/screens/d47v-a4b2-cal400.cenn"
ENGINE = "target/release/chessengine"
ORACLE = "tools/nnue/calib/oracle-obs.json"
BENCH_FENS = "tools/nnue/calib/scale-check-800.fens"
SUITE = "engines/suites/mate_in_3.epd"
NODES_RE = re.compile(r"^Nodes searched  : (\d+)", re.M)


def parse_ov(body: str) -> dict[str, int]:
    ov: dict[str, int] = {}
    for tok in filter(None, (t.strip() for t in body.split(","))):
        if "*" in tok:
            fam, _, f = tok.partition("*")
            for k, v in FAMILIES[fam.strip()].items():
                ov[k] = round_away(v * float(f))
        else:
            k, _, v = tok.partition("=")
            ov[k.strip()] = int(v)
    return ov


def bench(depth: int, ov: dict[str, int]) -> int:
    lines = [f"setoption name EvalFile value {NET}"]
    lines += [f"setoption name {k} value {v}" for k, v in ov.items()]
    lines += [f"bench 16 1 {depth} {BENCH_FENS} depth", "quit"]
    r = subprocess.run([ENGINE], input="\n".join(lines) + "\n", text=True,
                       capture_output=True)
    m = NODES_RE.search(r.stdout + r.stderr)  # bench reports on stderr
    return int(m.group(1)) if m else 0


def tactics(nodes: int, ov: dict[str, int]) -> int:
    cmd = ["python3", "tools/tactics.py", ENGINE, SUITE, "--nodes", str(nodes),
           "--option", f"EvalFile={NET}"]
    for k, v in ov.items():
        cmd += ["--option", f"{k}={v}"]
    out = subprocess.run(cmd, text=True, capture_output=True).stdout
    m = re.search(r": (\d+)/(\d+)", out)
    return int(m.group(1)) if m else 0


def play_all(fens: list[str], nodes: int, ov: dict[str, int]) -> dict[str, str]:
    eng = chess.engine.SimpleEngine.popen_uci(ENGINE)
    out: dict[str, str] = {}
    try:
        eng.configure({"EvalFile": NET})
        for k, v in ov.items():
            eng.configure({k: v})
        for fen in fens:
            res = eng.play(chess.Board(fen), chess.engine.Limit(nodes=nodes))
            if res.move is not None:
                out[fen] = res.move.uci()
    finally:
        eng.quit()
    return out


def cp_loss(tier_tables: dict, fens: list[str], chosen: dict[str, str]):
    """Per-tier mean loss, per-tier PER-POSITION losses, and top-1 rate.

    The per-position vector is what makes a delta readable. Control and arm
    see identical positions, so the paired difference
    `loss_arm[i] - loss_control[i]` cancels position difficulty, and its
    standard error is far smaller than either arm's own spread. Reading an
    unpaired sem instead is how a set of 1-3 cp deltas at n=300 got mistaken
    for signal on 2026-07-27; the same arms measured the OPPOSITE sign at
    n=1500.
    """
    per_tier, per_pos, top1 = {}, {}, float("nan")
    keys = sorted(tier_tables, key=int)
    for t in keys:
        tbl, losses, hits = tier_tables[t], {}, 0
        for fen, uci in chosen.items():
            moves = tbl.get(fen)
            if not moves or uci not in moves:
                continue
            best = max(moves.values())
            losses[fen] = best - moves[uci]
            hits += int(moves[uci] == best)
        vals = list(losses.values())
        per_tier[t] = statistics.fmean(vals) if vals else float("nan")
        per_pos[t] = losses
        if t == keys[-1] and vals:
            top1 = hits / len(vals)
    return per_tier, per_pos, top1


def paired(arm: dict[str, float], ctrl: dict[str, float]) -> tuple[float, float]:
    """(mean paired delta, its standard error) over positions both scored."""
    d = [arm[f] - ctrl[f] for f in arm if f in ctrl]
    if len(d) < 2:
        return float("nan"), float("nan")
    return statistics.fmean(d), statistics.stdev(d) / math.sqrt(len(d))


def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("arms", nargs="*", help="name:K=V,FAMILY*FACTOR")
    ap.add_argument("--base", default="", help="config applied to control AND every arm")
    ap.add_argument("--limit", type=int, default=800, help="oracle positions")
    ap.add_argument("--nodes", type=int, default=50000, help="candidate budget")
    ap.add_argument(
        "--budgets",
        default="",
        help="comma-separated candidate budgets -> a SCALING check. A real "
        "bullet move at 8+0.08 gets ~1.3M nodes on this box, 26x the 50k "
        "default, and depth-gated mechanisms (RFP's depth<17, razoring's d^2, "
        "LMP's depth^2, the NMP bar) behave differently at depth 13 and depth "
        "19. An arm whose gain shrinks or flips as the budget grows was tuned "
        "for a search shallower than the one it will play in.",
    )
    ap.add_argument("--depth", type=int, default=10, help="bench depth")
    ap.add_argument("--no-nodes", action="store_true")
    ap.add_argument("--no-tactics", action="store_true")
    ap.add_argument("--oracle", default=ORACLE)
    args = ap.parse_args()

    base_ov = parse_ov(args.base) if args.base else {}
    tier_tables = json.load(open(args.oracle))["tiers"]
    keys = sorted(tier_tables, key=int)
    if args.limit:
        tier_tables = {t: dict(list(tbl.items())[: args.limit])
                       for t, tbl in tier_tables.items()}
    fens = list(tier_tables[keys[0]])

    arms = [("control", {})]
    for spec in args.arms:
        name, _, body = spec.partition(":")
        arms.append((name, parse_ov(body)))

    if base_ov:
        print(f"base: {args.base}")
    # Power warning. Paired sem falls as 1/sqrt(n) and the effects worth
    # chasing here are ~1 cp, so a small --limit cannot resolve them however
    # confident the mean looks. Measured on this corpus: sem ~2.0 cp at 250
    # positions, ~1.1 at 800, ~0.8 at 1500, ~0.3 at the full 9992.
    if len(fens) < 3000:
        approx = 2.0 * (250 / max(len(fens), 1)) ** 0.5
        print(f"{len(fens)} positions, arms at {args.nodes} nodes  "
              f"[EXPLORATORY: paired sem ~{approx:.1f} cp -- use --limit 0 "
              f"for a verdict]\n")
    else:
        print(f"{len(fens)} positions, arms at {args.nodes} nodes\n")
    hdr = f"{'arm':<20}" + "".join(f"{'cp@' + t[:-3] + 'k':>13}" for t in keys) + f"{'top1':>8}"
    if not args.no_nodes:
        hdr += f"{'nodes':>10}"
    if not args.no_tactics:
        hdr += f"{'tact':>8}"
    print(hdr)

    if args.budgets:
        budgets = [int(b) for b in args.budgets.split(",")]
        print(f"{'arm':<20}" + "".join(f"{str(b//1000)+'k':>12}" for b in budgets)
              + "    (cp@64k tier, paired delta)")
        ref_by_budget: dict[int, dict[str, float]] = {}
        for name, ov in arms:
            full = {**base_ov, **ov}
            row = f"{name:<20}"
            for b in budgets:
                _, per_pos, _ = cp_loss(tier_tables, fens, play_all(fens, b, full))
                deep = per_pos[keys[-1]]
                if name == "control":
                    ref_by_budget[b] = deep
                    row += f"{statistics.fmean(list(deep.values())):>12.2f}"
                else:
                    d, se = paired(deep, ref_by_budget[b])
                    row += f"{d:>+8.2f}+-{se:<3.1f}"
            print(row, flush=True)
        print("\nControl row is absolute cp loss at the deepest oracle tier; "
              "arm rows are paired\ndeltas at the SAME budget. A gain that "
              "shrinks with budget will not survive a\nreal time control.")
        return

    ref_cp, ref_pos, ref_nodes, ref_tact = None, None, None, None
    for name, ov in arms:
        full = {**base_ov, **ov}
        per_tier, per_pos, top1 = cp_loss(
            tier_tables, fens, play_all(fens, args.nodes, full)
        )
        row = f"{name:<20}"
        if ref_cp is None:
            ref_cp, ref_pos = per_tier, per_pos
            row += "".join(f"{per_tier[t]:>9.2f}" for t in keys)
        else:
            # Paired delta and its sem: "+-" is one standard error of the
            # PAIRED difference, so |delta| < sem is noise regardless of how
            # large the absolute losses are.
            for t in keys:
                d, se = paired(per_pos[t], ref_pos[t])
                row += f"{d:>+7.2f}+-{se:<4.2f}"
        row += f"{top1:>8.1%}"
        if not args.no_nodes:
            n = bench(args.depth, full)
            if ref_nodes is None:
                ref_nodes = n
                row += f"{'(base)':>10}"
            else:
                row += f"{n / ref_nodes - 1:>+9.1%} "
        if not args.no_tactics:
            h = tactics(args.nodes, full)
            if ref_tact is None:
                ref_tact = h
                row += f"{h:>8}"
            else:
                row += f"{h - ref_tact:>+8}"
        print(row, flush=True)

    print("\ncp = mean centipawn loss vs SF18 labels; LOWER is better. Arm rows "
          "are PAIRED\ndeltas +- one standard error of the paired difference: "
          "|delta| < sem is noise.\nRequire the sign to hold across all three "
          "tiers. Promote on cp, sanity-check\nwith nodes, never promote on "
          "tactics alone.")


if __name__ == "__main__":
    main()
