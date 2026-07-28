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
import random
import re
import math
import statistics
import subprocess
import sys

# python-chess lives in the repo venv, not the system interpreter, and
# `python3 tools/screen.py` is what everyone types. Re-exec instead of dying
# on a ModuleNotFoundError that has nothing to do with the task at hand.
try:
    import chess
    import chess.engine
except ModuleNotFoundError:  # pragma: no cover
    from pathlib import Path as _P
    _venv = _P(__file__).resolve().parent.parent / ".venv" / "bin" / "python3"
    if _venv.exists() and _P(sys.executable).resolve() != _venv.resolve():
        os.execv(str(_venv), [str(_venv), *sys.argv])
    raise

LEDGER = "docs/research/screen-results.jsonl"


def record(row: dict) -> None:
    """Append one measured arm to the results ledger.

    The point is compounding: a screening session produces numbers that are
    otherwise lost to scrollback, so the same arm gets re-measured weeks later
    and, worse, a REFUTED arm gets re-proposed because nobody remembers it
    died. Every row here carries the config, the metric, the sample size and
    the instrument settings, so a later run can be compared rather than
    repeated.

    Append-only JSONL, one row per arm per run, git-tracked. Cheap to grep,
    impossible to silently overwrite.
    """
    import datetime as _dt
    row = dict(row)
    row["utc"] = _dt.datetime.now(_dt.timezone.utc).isoformat(timespec="seconds")
    with open(LEDGER, "a") as fh:
        fh.write(json.dumps(row, sort_keys=True) + "\n")


def prior(name: str, ov: dict) -> list[dict]:
    """Past measurements of an identical config, newest first."""
    if not os.path.exists(LEDGER):
        return []
    key = json.dumps(ov, sort_keys=True)
    out = [r for line in open(LEDGER)
           if (r := json.loads(line)).get("config") == key]
    return list(reversed(out))

sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
from arm_screen import FAMILIES, round_away  # noqa: E402

NET = "data/nnue/nets/screens/d47v-a4b2-cal400.cenn"
# target/release/chessengine silently IGNORES EvalFile unless built
# with sota,arch-d47v -- see assert_net_live below.
ENGINE = os.environ.get("ARCHON_BIN", "target/qtrace/release/chessengine")
ORACLE = "tools/nnue/calib/oracle-obs.json"
BENCH_FENS = "tools/nnue/calib/scale-check-800.fens"
SUITE = "engines/suites/mate_in_3.epd"
def assert_net_live(binary: str, net: str) -> None:
    """Refuse to run unless the net actually changes the search.

    Twice on 2026-07-27 an entire day of measurement was invalidated because
    `setoption name EvalFile` silently did nothing and the engine kept using
    PSQT. The first time it was a bench argument the UCI path ignored; the
    second, a binary built without `sota,arch-d47v`, where the CEN2 reader is
    absent and the load fails without a word.

    A missing net does not error, does not warn, and produces perfectly
    plausible numbers -- so the only defence is to prove the net is live
    before trusting anything. Two identical searches with and without it mean
    it is not.
    """
    import subprocess as _sp
    def probe(with_net: bool) -> str:
        cmd = "uci\n"
        if with_net:
            cmd += f"setoption name EvalFile value {net}\n"
        cmd += "isready\nposition startpos\ngo depth 8\n"
        p = _sp.Popen(["bash", "-c", f'{{ printf %s "$S"; sleep 4; }} | timeout 20 {binary}'],
                      stdout=_sp.PIPE, text=True, env={**os.environ, "S": cmd})
        out, _ = p.communicate()
        for line in reversed(out.splitlines()):
            if line.startswith("info depth 8 "):
                return line.split(" nodes ")[1].split()[0] if " nodes " in line else line
        return ""
    off, on = probe(False), probe(True)
    if off and off == on:
        raise SystemExit(
            f"REFUSING TO RUN: {binary} ignores EvalFile ({net}).\n"
            f"  depth-8 nodes identical with and without the net: {off}\n"
            f"  build one that can read it:\n"
            f"    cargo build --release --features tune,sota,arch-d47v "
            f"--target-dir target/qtrace\n"
            f"  every measurement from a net-less binary is a PSQT measurement."
        )


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


def fhf(depth: int, ov: dict[str, int]) -> float:
    """First-move cutoff rate: of all fail-highs, the share taken on move 1.

    The cheapest meaningful search-quality metric we have -- one deterministic
    bench, seconds. A cutoff taken on move 5 instead of move 1 searched four
    subtrees for nothing, so higher fhf is strictly better ordering.

    Ours is 85.6% on a net-live binary. Do NOT compare that to the "93-95%"
    this docstring used to quote for Stockfish: that figure is folklore, was
    never measured with our definition, and the 89.4% it was compared against
    was a PSQT measurement. The old claim that fhf and a "1.6-3.6x wider tree"
    were the same fact died with the width gap itself -- net-live, our tree is
    ~1.4x SF18's at depth 10 and ~0.93x at depth 14. Treat fhf as a RELATIVE
    instrument: arm vs control on the same binary, never against a remembered
    cross-engine constant.

    Higher is better, and unlike cp loss it moves on runs that take seconds,
    so it is the right first filter for move-ORDERING parameters.
    """
    lines = [f"setoption name EvalFile value {NET}"]
    lines += [f"setoption name {k} value {v}" for k, v in ov.items()]
    lines += [f"bench 16 1 {depth} {BENCH_FENS} depth", "quit"]
    r = subprocess.run([ENGINE], input="\n".join(lines) + "\n", text=True,
                       capture_output=True, env={**os.environ, "ARCHON_SEARCH_STATS": "1"})
    fh = sum(int(x) for x in re.findall(r"\bfh=(\d+)", r.stderr))
    hit = sum(int(x) for x in re.findall(r"\bfhf=(\d+)", r.stderr))
    return 100.0 * hit / fh if fh else float("nan")


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


def play_reference(binary: str, fens: list[str], nodes: int) -> dict[str, str]:
    """Same position set, same node budget, a DIFFERENT engine.

    Why this row belongs on the table. Mean cp loss is a relative instrument:
    it tells you arm-vs-control and nothing about how much is left to win.
    Our control loses ~4.6 cp per move at 50k nodes -- is that close to what a
    50k-node search can achieve, or is half of it recoverable? Without an
    opponent at the same budget the number has no scale, and a screen with no
    scale cannot tell "we tuned away the last 5%" from "we are leaving 3 cp on
    the table".

    BIAS, stated plainly: the oracle labels are Stockfish's, so scoring
    Stockfish against them measures partly its agreement with ITSELF. An SF
    reference is therefore an optimistic floor, not a fair target. Run a
    non-Stockfish engine (Reckless) alongside it -- where the two disagree,
    the truth is between them, and their SPREAD is the honest error bar on
    "how much headroom exists".
    """
    eng = chess.engine.SimpleEngine.popen_uci(binary)
    out: dict[str, str] = {}
    try:
        eng.configure({"Threads": 1, "Hash": 16})
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


def tail_stats(losses: dict[str, float]) -> tuple[float, float]:
    """(blunder rate, p90 loss) for one arm at one tier.

    The mean is almost entirely tail: the control's MEDIAN loss is ~1-2 cp
    against a mean of ~17. So "mean cp loss" is really a blunder-weighted
    statistic wearing a misleading name, and the tail deserves to be visible
    rather than inferred.

    It also matters differently in games. A blunder loses a game outright; a
    string of 3 cp inaccuracies usually does not. Two configs with the same
    mean are not equally good if one gets there via a fatter tail, and mean
    alone cannot separate them.

    100 cp is the blunder line -- roughly a pawn, i.e. large enough that no
    amount of subsequent accuracy reliably recovers it.
    """
    v = list(losses.values())
    if not v:
        return float("nan"), float("nan")
    blunders = sum(1 for x in v if x >= 100) / len(v)
    return blunders, sorted(v)[int(0.90 * (len(v) - 1))]


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
    ap.add_argument("--fhf", action="store_true",
                    help="also report first-move cutoff rate (cheap, targets "
                    "the measured ordering defect)")
    ap.add_argument("--oracle", default=ORACLE)
    ap.add_argument("--reference", action="append", default=[],
                    metavar="[NAME=]BINARY",
                    help="score another engine at the same node budget, to give "
                         "cp loss an absolute scale (repeatable)")
    ap.add_argument("--fast", action="store_true",
                    help="fhf + nodes only, skipping the cp oracle. Both are "
                    "deterministic bench measurements taking SECONDS, against "
                    "minutes for cp, so this is the right first pass over a "
                    "large family. Neither is a strength measure on its own -- "
                    "survivors must still face cp before an SPRT.")
    ap.add_argument("--no-record", action="store_true",
                    help="do not append results to the ledger")
    args = ap.parse_args()
    assert_net_live(ENGINE if 'ENGINE' in globals() else ARCHON, NET)

    base_ov = parse_ov(args.base) if args.base else {}

    if args.fast:
        arms_f = [("control", {})]
        for spec in args.arms:
            nm, _, body = spec.partition(":")
            arms_f.append((nm, parse_ov(body)))
        if args.base:
            print(f"base: {args.base}")
        print(f"{'arm':<24}{'nodes':>10}{'fhf%':>9}{'d(fhf)':>9}")
        ref_n = ref_f = None
        for nm, ov in arms_f:
            full = {**base_ov, **ov}
            n, f = bench(args.depth, full), fhf(args.depth, full)
            if ref_n is None:
                ref_n, ref_f = n, f
                print(f"{nm:<24}{'(base)':>10}{f:>9.2f}{'':>9}")
            else:
                print(f"{nm:<24}{n/ref_n-1:>+9.1%} {f:>9.2f}{f-ref_f:>+9.2f}",
                      flush=True)
        print("\nfhf = first-move cutoff rate, HIGHER is better (ours ~85.6% "
              "net-live).\nRead it arm-vs-control only -- there is no verified "
              "cross-engine number to\ncompare against. Want fhf UP and nodes "
              "DOWN together; either alone can be gamed.")
        return
    tier_tables = json.load(open(args.oracle))["tiers"]
    keys = sorted(tier_tables, key=int)
    if args.limit:
        # RANDOM sample, not a prefix. The corpus is ordered, and taking the
        # first N systematically over-weights hard positions: the control's
        # absolute mean loss fell monotonically with the prefix length --
        # 19.50 / 19.10 / 18.51 / 17.81 / 17.29 cp at n = 250 / 1500 / 2000 /
        # 4000 / 9992. That is BIAS, not noise, and it inflated every
        # exploratory delta measured before 2026-07-27, because effects are
        # larger on harder positions.
        #
        # Fixed seed so an exploratory run and its full-corpus follow-up
        # remain comparable, and so two arms always see identical positions.
        rng = random.Random(20260727)
        keys0 = sorted(tier_tables[sorted(tier_tables, key=int)[0]])
        pick = set(rng.sample(keys0, min(args.limit, len(keys0))))
        tier_tables = {t: {f: mv for f, mv in tbl.items() if f in pick}
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
    hdr = f"{'arm':<20}" + "".join(f"{'cp@' + t[:-3] + 'k':>13}" for t in keys) + f"{'top1':>8}{'blund':>8}{'p90':>7}"
    if not args.no_nodes:
        hdr += f"{'nodes':>10}"
    if args.fhf:
        hdr += f"{'fhf%':>8}"
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
        for spec in args.reference:
            nm, _, path = spec.rpartition("=")
            nm = nm or os.path.basename(path)
            row = f"{'* ' + nm:<20}"
            for b in budgets:
                _, per_pos, _ = cp_loss(
                    tier_tables, fens, play_reference(path, fens, b))
                deep = per_pos[keys[-1]]
                row += f"{statistics.fmean(list(deep.values())):>12.2f}"
            print(row, flush=True)

        print("\nControl row is absolute cp loss at the deepest oracle tier; "
              "arm rows are paired\ndeltas at the SAME budget. A gain that "
              "shrinks with budget will not survive a\nreal time control.")
        if args.reference:
            print("\nA REFERENCE engine swept across budgets is an eval-vs-search")
            print("decomposition, and it is the only instrument here that can do it.")
            print("At a tiny budget a search barely improves on its own static eval, so")
            print("the gap to the reference is mostly EVAL QUALITY. As the budget grows,")
            print("search contributes more, so a gap that WIDENS with budget is search")
            print("and a gap that stays FLAT is eval.")
            print("Read the SLOPE, not the level: the oracle labels are Stockfish's, so")
            print("a Stockfish reference is partly scoring its agreement with itself and")
            print("its level is flattered at every budget. That bias is roughly constant")
            print("across budgets, which is exactly why the slope survives it.")
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
        blunder, p90 = tail_stats(per_pos[keys[-1]])
        row += f"{top1:>8.1%}{blunder:>8.1%}{p90:>7.0f}"
        if not args.no_nodes:
            n = bench(args.depth, full)
            if ref_nodes is None:
                ref_nodes = n
                row += f"{'(base)':>10}"
            else:
                row += f"{n / ref_nodes - 1:>+9.1%} "
        if args.fhf:
            row += f"{fhf(args.depth, full):>8.2f}"
        if not args.no_tactics:
            h = tactics(args.nodes, full)
            if ref_tact is None:
                ref_tact = h
                row += f"{h:>8}"
            else:
                row += f"{h - ref_tact:>+8}"
        print(row, flush=True)
        if not args.no_record and ov:
            hist = prior(name, full)
            if hist:
                h = hist[0]
                print(f"{'':<20}  (seen {h['utc'][:10]}: "
                      f"cp@deep {h.get('cp_deep'):+.2f} at n={h.get('n')})")
            record({
                "arm": name, "config": json.dumps(full, sort_keys=True),
                "base": args.base, "n": len(fens), "nodes": args.nodes,
                "cp_deep": round(per_tier[keys[-1]] - ref_cp[keys[-1]], 3),
                "cp_by_tier": {t: round(per_tier[t] - ref_cp[t], 3) for t in keys},
                "top1": round(top1, 4), "oracle": os.path.basename(args.oracle),
            })

    for spec in args.reference:
        nm, _, path = spec.rpartition("=")
        nm = nm or os.path.basename(path)
        per_tier, per_pos, top1 = cp_loss(
            tier_tables, fens, play_reference(path, fens, args.nodes))
        row = f"{'* ' + nm:<20}" + "".join(f"{per_tier[t]:>9.2f}" for t in keys)
        blunder, p90 = tail_stats(per_pos[keys[-1]])
        row += f"{top1:>8.1%}{blunder:>8.1%}{p90:>7.0f}"
        print(row, flush=True)
        d, se = paired(per_pos[keys[-1]], ref_pos[keys[-1]])
        print(f"{'':<20}  headroom vs control at {args.nodes} nodes: "
              f"{-d:+.2f} +- {se:.2f} cp")

    if args.reference:
        print("\n* reference rows are ABSOLUTE cp loss, same positions, same node")
        print("budget, no tunables. They give the screen a scale: 'headroom' is how")
        print("much a stronger engine saves per move at OUR budget, i.e. the most")
        print("that search tuning could possibly recover. CAVEAT: the oracle labels")
        print("are Stockfish's, so an SF reference is partly scoring its agreement")
        print("with itself and reads optimistically. Use a non-SF engine too and")
        print("treat the spread between them as the error bar on the headroom.")

    print("\nblund = share of moves losing >=100 cp; p90 = 90th-percentile loss, "
          "both at the\ndeepest tier. The mean is mostly tail (median loss is "
          "~1-2 cp), and a blunder\nloses a game where small inaccuracies "
          "usually do not -- so prefer an arm that\nimproves the mean WITHOUT "
          "fattening the tail.")
    print("\ncp = mean centipawn loss vs SF18 labels; LOWER is better. Arm rows "
          "are PAIRED\ndeltas +- one standard error of the paired difference: "
          "|delta| < sem is noise.\nRequire the sign to hold across all three "
          "tiers. Promote on cp, sanity-check\nwith nodes, never promote on "
          "tactics alone.")


if __name__ == "__main__":
    main()
