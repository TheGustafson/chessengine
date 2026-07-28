#!/usr/bin/env python3
"""rematch — play ARCHON vs a reference engine and run the whole forensic
chain in one command.

WHY THIS EXISTS. Loss mining is the highest-information instrument we have
(playbook section 6), but assembling it by hand takes a dozen steps: build the
right binary, prove the net is live, size concurrency around whatever else is
on the box, invoke fastchess with matched options, then run match_report,
loss_positions and the taxonomy diff over the result. Every one of those steps
has been got wrong at least once. The most expensive was the first: on
2026-07-27 an entire day of ARCHON-vs-SF18 games was played by a binary that
silently ignored `setoption name EvalFile` and evaluated with PSQT.

So this refuses to run until the net demonstrably changes the search, and then
does the rest without a human in the loop.

TWO LEGS, AND WHY BOTH. `--legs time,nodes` plays the same match twice:

  time    the honest matchup. Includes our NPS, our time management, and any
          time-pressure mistakes. This is what Elo means.
  nodes   equal nodes per move, both sides. Removes speed entirely, so what
          remains is decision quality per node.

The taxonomy DIFF between the two legs is the point. A loss class that appears
in the timed leg and vanishes at equal nodes is a SPEED problem (we did not
get deep enough in time); one that survives both is a JUDGEMENT problem (we
got there and still chose wrong). Those two findings send you to completely
different tunables, and no single leg can tell them apart.

    python3 tools/rematch.py --games 120
    python3 tools/rematch.py --games 60 --legs time --our-options QS_FUTILITY_MARGIN=260
    python3 tools/rematch.py --games 120 --loss-oracle     # label the losses too

Outputs land in engines/runs/<label>/<leg>/ : games.pgn, fastchess.log,
report.txt, losses.json and loss-positions.fens — the last being a corpus of
the exact positions where games were decided.

To screen tunables on that corpus you must LABEL it first; `screen.py` takes
an oracle, not a position list (there is no `--positions` flag):

    python3 tools/move_quality.py --build-oracle --fens <loss-positions.fens> \
        --cache oracle-losses.json --multipv 20 --oracle-depth 18
    python3 tools/screen.py --oracle oracle-losses.json 'arm:PARAM=V'

`--loss-oracle` does both steps for you.
"""

from __future__ import annotations

import argparse
import json
import math
import os
import re
import shutil
import subprocess
import sys
import time
from pathlib import Path

REPO = Path(__file__).resolve().parent.parent

# python-chess lives in the repo venv, not the system interpreter. Re-exec
# rather than fail: `python3 tools/rematch.py` is what everyone types, and a
# ModuleNotFoundError three tools deep is a waste of a turn.
try:
    import chess  # noqa: F401
except ModuleNotFoundError:  # pragma: no cover
    venv = REPO / ".venv" / "bin" / "python3"
    if venv.exists() and Path(sys.executable).resolve() != venv.resolve():
        os.execv(str(venv), [str(venv), *sys.argv])
    raise

sys.path.insert(0, str(Path(__file__).resolve().parent))
from screen import assert_net_live  # noqa: E402  (also the net-live guard)
ARCHON = os.environ.get("ARCHON_MATCH_BIN", "engines/bin/archon-d47v-pgo")
NET = os.environ.get("ARCHON_NET", "engines/canonical/net.cenn")
SF18 = "engines/bin/stockfish18"
FASTCHESS = "engines/bin/fastchess"
BOOK = "engines/books/UHO_Lichess_4852_v1.epd"

# reported-cp -> oracle-cp slope per engine, for match_report's cross-engine
# comparisons. Ours is the measured 0.64 eval frame; SF18's is its own.
OUR_SLOPE, SF_SLOPE = 0.64, 1.07


def physical_free(reserve: int) -> int:
    """Physical cores we can take without fighting the trainer.

    Timed games and hyperthread siblings do not mix -- contention shows up as
    clock noise, which reads as a time-management bug that is not there. So
    this counts PHYSICAL cores only and subtracts whatever is already running
    (the NNUE trainer holds ~3.5 of them for days at a time).
    """
    phys = os.cpu_count() // 2 if os.cpu_count() else 4
    try:
        # the 5-minute average, not the 1-minute one: our own net-live probe
        # spikes the latter and would throttle the whole match to one game.
        load = os.getloadavg()[1]
    except OSError:
        load = 0.0
    return max(2, phys - math.ceil(load) - reserve)


def build_cmd(leg: str, args, out: Path, ours_label: str) -> list[str]:
    our_opts = ["-engine", f"cmd={args.engine}", f"name={ours_label}",
                f"option.EvalFile={args.net}", "option.Threads=1",
                f"option.Hash={args.hash}"]
    for kv in (args.our_options or "").split(",") :
        if kv.strip():
            k, _, v = kv.partition("=")
            our_opts.append(f"option.{k.strip()}={v.strip()}")

    them = ["-engine", f"cmd={args.opponent}", f"name={args.opponent_name}",
            "option.Threads=1", f"option.Hash={args.hash}"]
    # A self-play A/B (same binary, one option changed) is the normal way to
    # measure an engine change, and it walks straight into the net-not-loaded
    # trap: the opponent block carries no EvalFile, so the BASELINE would play
    # PSQT while the candidate plays NNUE and the match would measure the net,
    # not the change. --opponent-options is how the net gets onto that side.
    for kv in (args.opponent_options or "").split(","):
        if kv.strip():
            k, _, v = kv.partition("=")
            them.append(f"option.{k.strip()}={v.strip()}")

    if leg == "time":
        tc = ["-each", f"tc={args.tc}"]
        conc = args.concurrency or physical_free(args.reserve)
    else:
        # fixed nodes is immune to machine load and to SMT, so it may use every
        # logical thread that is not already busy.
        tc = ["-each", "tc=inf", f"nodes={args.nodes}"]
        conc = args.concurrency or max(1, (os.cpu_count() or 8)
                                       - math.ceil(os.getloadavg()[0]) - args.reserve)

    return [args.fastchess, *our_opts, *them, *tc,
            "-openings", f"file={args.book}", "format=epd", "order=random",
            "-repeat", "-games", "2", "-rounds", str(max(1, args.games // 2)),
            "-concurrency", str(conc),
            # twosided is MANDATORY here, not a preference. Adjudication
            # thresholds are in centipawns of whichever engine is reporting,
            # and our cp is RAW INTERNAL (~247 units/pawn) while Stockfish's
            # is real centipawns. One-sided `score=700` would mean "resign at
            # 2.8 pawns down" for us and "resign at 7 pawns down" for SF18 --
            # we would concede games it would still be playing, and the loss
            # taxonomy would inherit the asymmetry. Requiring both engines to
            # agree removes the frame from the decision entirely.
            "-resign", "movecount=4", "score=700", "twosided=true",
            "-draw", "movenumber=40", "movecount=8", "score=20",
            "-maxmoves", "400",
            "-pgnout", f"file={out/'games.pgn'}", "-log", f"file={out/'fastchess.log'}",
            "-recover"]


def run_leg(leg: str, args, root: Path, ours_label: str) -> Path:
    out = root / leg
    out.mkdir(parents=True, exist_ok=True)
    cmd = build_cmd(leg, args, out, ours_label)
    (out / "cmd.txt").write_text(" ".join(cmd) + "\n")
    print(f"\n=== leg: {leg} ===\n{' '.join(cmd[:6])} ...  "
          f"concurrency={cmd[cmd.index('-concurrency') + 1]}", flush=True)
    t0 = time.time()
    p = subprocess.run(cmd, cwd=REPO, capture_output=True, text=True)
    (out / "stdout.log").write_text(p.stdout)
    res = parse_result(p.stdout)
    if res:
        print(f"  {res['games']} games  {res['pct']:.2f}%  "
              f"Elo {res['elo']:+.1f} +/- {res['err']:.1f}   "
              f"(+{res['w']} ={res['d']} -{res['l']})")
    else:
        print(p.stdout[-800:])
    print(f"  ({time.time() - t0:.0f}s)")
    LEG_RESULT[leg] = res
    return out / "games.pgn"


LEG_RESULT: dict[str, dict | None] = {}
# (?<!n) matters: fastchess prints "Elo: 9.27 +/- 17.73, nElo: 14.55 +/- 27.80"
# on ONE line, and a bare `Elo:` matches the nElo too -- findall then hands back
# the normalized figure last, so the reported Elo is silently the wrong
# statistic. Caught by unit-checking the parser against a real log line.
RE_ELO = re.compile(r"(?<!n)Elo:\s*([-\d.]+)\s*\+/-\s*([-\d.naN]+)")
RE_NELO = re.compile(r"nElo:\s*([-\d.]+)\s*\+/-\s*([-\d.naN]+)")
RE_GAMES = re.compile(r"Games:\s*(\d+), Wins:\s*(\d+), Losses:\s*(\d+), "
                      r"Draws:\s*(\d+), Points:\s*[\d.]+\s*\(([\d.]+)\s*%\)")


def parse_result(stdout: str) -> dict | None:
    """Last reported score block. fastchess reprints it every few games, so
    only the final one is the whole match."""
    g = RE_GAMES.findall(stdout)
    if not g:
        return None
    n, w, l, d, pct = g[-1]

    def last(rx):
        m = rx.findall(stdout)
        try:
            return float(m[-1][0]), float(m[-1][1])
        except (IndexError, ValueError):
            return float("nan"), float("nan")

    elo, err = last(RE_ELO)
    nelo, nerr = last(RE_NELO)
    return {"games": int(n), "w": int(w), "l": int(l), "d": int(d),
            "pct": float(pct), "elo": elo, "err": err,
            "nelo": nelo, "nerr": nerr}


def forensics(pgn: Path, out: Path, ours_label: str, args) -> dict:
    """match_report + loss_positions, saved and summarised."""
    rep = subprocess.run(
        [sys.executable, "tools/match_report.py", str(pgn),
         "--a", ours_label, "--b", args.opponent_name,
         "--a-slope", str(OUR_SLOPE), "--b-slope", str(SF_SLOPE)],
        cwd=REPO, capture_output=True, text=True)
    (out / "report.txt").write_text(rep.stdout + rep.stderr)

    lp = subprocess.run(
        [sys.executable, "tools/loss_positions.py", str(pgn),
         "--us", ours_label, "--our-scale", str(OUR_SLOPE),
         "--fens", str(out / "loss-positions.fens"),
         "--json", str(out / "losses.json")],
        cwd=REPO, capture_output=True, text=True)
    (out / "loss_positions.txt").write_text(lp.stdout + lp.stderr)
    print(lp.stdout or lp.stderr[-1500:])

    try:
        blob = json.loads((out / "losses.json").read_text())
    except Exception:
        return {}
    tax: dict[str, int] = {}
    # a loss can carry several tags (grind+timepressure); count each, and
    # normalise by game count so the two legs stay comparable if one played
    # fewer games than the other.
    tax["_games"] = blob.get("tally", {}).get("games", 0)
    tax["_losses"] = blob.get("tally", {}).get("losses", 0)
    for L in blob.get("losses", []):
        for t in L.get("tags") or ["unclassified"]:
            tax[t] = tax.get(t, 0) + 1
    return tax


def build_loss_oracle(fens: Path, cache: Path, args) -> None:
    """Label the loss corpus with SF18 so tunables can be screened on it.

    MultiPV is raised well above move_quality's default of 6 because this
    corpus is small and adversarial: a move outside the oracle's top-K has an
    unknown loss that gets floored, and the moves most likely to fall outside
    are the bad ones -- so censoring flatters exactly the arms that deserve to
    look worst.
    """
    n = sum(1 for _ in open(fens))
    print(f"\nlabelling {n} loss positions with SF18 "
          f"(multipv {args.loss_multipv}, depth {args.loss_depth}) ...")
    subprocess.run(
        [sys.executable, "tools/move_quality.py", "--build-oracle",
         "--fens", str(fens), "--cache", str(cache),
         "--multipv", str(args.loss_multipv),
         "--oracle-depth", str(args.loss_depth)],
        cwd=REPO, check=False)
    print(f"screen on it:  python3 tools/screen.py --oracle {cache} 'arm:PARAM=V'")


def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("--games", type=int, default=100)
    ap.add_argument("--legs", default="time,nodes")
    ap.add_argument("--tc", default="10+0.1")
    ap.add_argument("--nodes", type=int, default=200000)
    ap.add_argument("--engine", default=ARCHON)
    ap.add_argument("--net", default=NET)
    ap.add_argument("--opponent", default=SF18)
    ap.add_argument("--opponent-name", default="sf18")
    ap.add_argument("--our-options", default="",
                    help="comma-separated K=V applied to ARCHON only")
    ap.add_argument("--opponent-options", default="",
                    help="comma-separated K=V applied to the OPPONENT only. For a "
                         "self-play A/B you MUST pass EvalFile here or the baseline "
                         "plays PSQT and the match measures the net, not the change.")
    ap.add_argument("--label", default=None)
    ap.add_argument("--hash", type=int, default=16)
    ap.add_argument("--book", default=BOOK)
    ap.add_argument("--fastchess", default=FASTCHESS)
    ap.add_argument("--concurrency", type=int, default=0)
    ap.add_argument("--reserve", type=int, default=1,
                    help="physical cores to leave free for the box")
    ap.add_argument("--loss-oracle", action="store_true",
                    help="label the loss corpus with SF18 so screen.py can use it")
    ap.add_argument("--loss-multipv", type=int, default=20)
    ap.add_argument("--loss-depth", type=int, default=18)
    ap.add_argument("--skip-net-check", action="store_true",
                    help="DO NOT USE except when the opponent is not ARCHON-shaped")
    args = ap.parse_args()

    os.chdir(REPO)
    if not shutil.which(args.fastchess) and not Path(args.fastchess).exists():
        raise SystemExit(f"fastchess not found at {args.fastchess}")

    # The guard that would have saved a day. Two searches, with and without the
    # net; identical node counts mean the binary cannot read it.
    if not args.skip_net_check:
        assert_net_live(args.engine, args.net)
        print(f"net live: {args.engine} + {Path(args.net).name}")
        if "archon" in Path(args.opponent).name.lower() \
           and "EvalFile" not in (args.opponent_options or ""):
            raise SystemExit(
                f"REFUSING TO RUN: opponent {args.opponent} looks like an ARCHON "
                f"binary but --opponent-options sets no EvalFile.\n"
                f"  It would play PSQT while our side plays NNUE, so the match "
                f"would measure the net rather than the change.\n"
                f"  add:  --opponent-options EvalFile={args.net}")

    label = args.label or f"rematch-{time.strftime('%m%d-%H%M')}"
    root = REPO / "engines" / "runs" / label
    ours = "archon"
    tax_by_leg = {}
    for leg in [l.strip() for l in args.legs.split(",") if l.strip()]:
        pgn = run_leg(leg, args, root, ours)
        if pgn.exists() and pgn.stat().st_size:
            tax_by_leg[leg] = forensics(pgn, root / leg, ours, args)

    rt, rn = LEG_RESULT.get("time"), LEG_RESULT.get("nodes")
    if rt and rn:
        # The headline. Our tree shape now measures close to Stockfish's at
        # equal depth, so the ~190 Elo gap has to come from somewhere else,
        # and there are only two candidates: nodes per second, or the quality
        # of the decision made with each node. These two legs separate them
        # cleanly, because the nodes leg holds throughput constant by fiat.
        gap = rn["elo"] - rt["elo"]
        print("\n=== equal-time vs equal-nodes: where the gap lives ===")
        print(f"  timed  {rt['pct']:6.2f}%   Elo {rt['elo']:+7.1f} +/- {rt['err']:.1f}")
        print(f"  nodes  {rn['pct']:6.2f}%   Elo {rn['elo']:+7.1f} +/- {rn['err']:.1f}")
        print(f"  gap    {gap:+7.1f} Elo recovered by removing the speed difference")
        thresh = 2 * math.hypot(rt["err"], rn["err"])
        if gap > thresh:
            print("  -> SPEED. We make comparable decisions per node and lose because")
            print("     we get fewer of them. The lever is NPS and time management,")
            print("     not search quality. At ~120 Elo per doubling, price any")
            print("     candidate by its throughput cost before its node count.")
        elif gap < -thresh:
            print("  -> Stockfish gains MORE from equal nodes than we do, i.e. its")
            print("     node budget buys more. Look at what it does per node that")
            print("     we do not, starting with move ordering.")
        else:
            # NOT a judgement verdict. An unresolved gap means this comparison
            # alone cannot attribute the difference, and labelling it
            # "JUDGEMENT" (as an earlier version did) turns a null result into
            # a positive claim.
            print(f"  -> UNRESOLVED at this sample size: |{gap:+.1f}| < {thresh:.1f} Elo.")
            print(f"     The POINT ESTIMATE splits the gap as ~{gap:.0f} Elo speed and")
            print(f"     ~{abs(rn['elo']):.0f} Elo per-node judgement, but this leg cannot")
            print("     separate them on its own. Corroborate with the NPS ratio and")
            print("     with cp loss across budgets before believing either share.")
        print(f"  (threshold {thresh:.1f} Elo = 2x the combined error. More games narrow")
        print("   it as 1/sqrt(n); theory does not.)")

    if len(tax_by_leg) == 2:
        a, b = tax_by_leg["time"], tax_by_leg["nodes"]
        ga, gb = max(a.get("_games", 1), 1), max(b.get("_games", 1), 1)
        print("\n=== taxonomy diff: timed vs equal-nodes "
              f"({ga} vs {gb} games; rates per 100) ===")
        print(f"{'class':<18}{'timed':>9}{'nodes':>9}{'delta':>9}")
        for k in sorted(k for k in set(a) | set(b) if not k.startswith("_")):
            ra, rb = 100 * a.get(k, 0) / ga, 100 * b.get(k, 0) / gb
            print(f"{k:<18}{ra:>9.1f}{rb:>9.1f}{ra - rb:>+9.1f}")
        print(f"{'ALL LOSSES':<18}{100*a.get('_losses',0)/ga:>9.1f}"
              f"{100*b.get('_losses',0)/gb:>9.1f}"
              f"{100*a.get('_losses',0)/ga - 100*b.get('_losses',0)/gb:>+9.1f}")
        print("\nA class that SHRINKS at equal nodes was bought by speed --")
        print("look at NPS, time management and pruning throughput. One that")
        print("SURVIVES equal nodes is a judgement error and lives in the")
        print("evaluation or the ordering, not the clock.")

    print(f"\nartifacts: {root}")
    leg0 = next(iter(tax_by_leg), "time")
    fens = root / leg0 / "loss-positions.fens"
    if args.loss_oracle and fens.exists():
        build_loss_oracle(fens, root / leg0 / "oracle-losses.json", args)
    else:
        print("\nto screen tunables where the games were actually decided:")
        print(f"  python3 tools/move_quality.py --build-oracle --fens {fens} \\")
        print(f"      --cache {root/leg0/'oracle-losses.json'} --multipv 20 --oracle-depth 18")
        print(f"  python3 tools/screen.py --oracle {root/leg0/'oracle-losses.json'} 'arm:PARAM=V'")
        print("  (--multipv 20, not the default 6: on a corpus this small every")
        print("   censored move is a floored loss, and the censoring lands hardest")
        print("   exactly on the arms that behave worst.)")


if __name__ == "__main__":
    main()
