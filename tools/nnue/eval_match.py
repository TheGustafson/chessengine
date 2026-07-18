#!/usr/bin/env python3
"""Reproducible paired match between two evaluator files on one engine build."""

from __future__ import annotations

import argparse
import json
import math
import re
import subprocess
from datetime import datetime, timezone
from pathlib import Path

from policy_sweep import paired_bootstrap


REPO = Path(__file__).resolve().parents[2]
SUMMARY_RE = re.compile(
    r"Games:\s*(\d+),\s*Wins:\s*(\d+),\s*Losses:\s*(\d+),"
    r"\s*Draws:\s*(\d+),\s*Points:"
)
# e.g. "LLR: 1.42 (48.2%) (-2.94, 2.94) [0.00, 10.00]" (the % is progress).
LLR_RE = re.compile(
    r"LLR:\s*(-?[\d.]+)\s*(?:\([\d.]+%\)\s*)?\((-?[\d.]+),\s*(-?[\d.]+)\)"
)
SPRT_ALPHA = 0.05
SPRT_BETA = 0.05


def absolute(value: str) -> Path:
    path = Path(value)
    return path if path.is_absolute() else (REPO / path).resolve()


def pgn_games(path: Path) -> int:
    return path.read_text(errors="replace").count('[Result "') if path.exists() else 0


def estimate(games: int, wins: int, losses: int, draws: int) -> dict:
    score = (wins + 0.5 * draws) / games
    elo = 400.0 * math.log10(score / (1.0 - score)) if 0.0 < score < 1.0 else (
        math.inf if score == 1.0 else -math.inf
    )
    values = [1.0] * wins + [0.0] * losses + [0.5] * draws
    variance = sum((value - score) ** 2 for value in values) / (games - 1)
    score_se = math.sqrt(variance / games)
    elo_se = (
        400.0 / math.log(10.0) * score_se / (score * (1.0 - score))
        if 0.0 < score < 1.0 else math.inf
    )
    return {"candidate_score": score, "candidate_elo": elo, "candidate_elo_se": elo_se}


def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--candidate", required=True, type=absolute)
    parser.add_argument("--baseline", required=True, type=absolute)
    parser.add_argument("--games", type=int, default=40)
    limit = parser.add_mutually_exclusive_group()
    limit.add_argument("--nodes", type=int)
    limit.add_argument("--tc")
    parser.add_argument("--concurrency", type=int, default=1)
    parser.add_argument("--sprt-elo0", type=float,
                        help="SPRT H0 bound (logistic Elo); early-stops a decided "
                             "match, with --games as the cap. Requires --sprt-elo1.")
    parser.add_argument("--sprt-elo1", type=float,
                        help="SPRT H1 bound (logistic Elo). Requires --sprt-elo0.")
    parser.add_argument("--seed", type=int, default=20260714)
    parser.add_argument("--openings", type=absolute,
                        default=absolute("data/nnue/openings/policy-balanced-256.epd"))
    parser.add_argument("--output", required=True, type=absolute)
    parser.add_argument("--engine", type=absolute, default=absolute("target/release/chessengine"))
    parser.add_argument("--candidate-engine", type=absolute,
                        help="candidate binary (defaults to --engine)")
    parser.add_argument("--baseline-engine", type=absolute,
                        help="baseline binary (defaults to --engine)")
    parser.add_argument("--fastchess", type=absolute, default=absolute("engines/bin/fastchess"))
    parser.add_argument("--each-option", action="append", default=[], metavar="NAME=VALUE",
                        help="extra UCI option applied to BOTH engines (repeatable), e.g. "
                             "--each-option ObsCapturePath=data/obs/harvest/run1.ceobs for "
                             "fixed-node capture harvests (engines pid-suffix their shards; "
                             "engine must be a datagen-feature build; NEVER use in TC gates)")
    args = parser.parse_args()
    args.candidate_engine = args.candidate_engine or args.engine
    args.baseline_engine = args.baseline_engine or args.engine
    if args.nodes is None and args.tc is None:
        args.nodes = 2000
    if args.games <= 0 or args.games % 2:
        parser.error("--games must be a positive even number")
    if (args.sprt_elo0 is None) != (args.sprt_elo1 is None):
        parser.error("--sprt-elo0 and --sprt-elo1 must be given together")
    sprt = args.sprt_elo1 is not None
    if sprt and args.sprt_elo1 <= args.sprt_elo0:
        parser.error("--sprt-elo1 must be greater than --sprt-elo0")
    if args.nodes is not None and args.nodes <= 0:
        parser.error("--nodes must be positive")
    for path in (args.candidate, args.baseline, args.openings, args.candidate_engine,
                 args.baseline_engine, args.fastchess):
        if not path.exists():
            parser.error(f"missing required path: {path}")

    args.output.mkdir(parents=True, exist_ok=True)
    state_path = args.output / "summary.json"
    pgn = args.output / "games.pgn"
    stdout_log = args.output / "stdout.log"
    expected = {
        "candidate": str(args.candidate), "baseline": str(args.baseline),
        "games": args.games, "nodes": args.nodes, "tc": args.tc,
        "concurrency": args.concurrency, "seed": args.seed,
        "sprt_elo0": args.sprt_elo0, "sprt_elo1": args.sprt_elo1,
        "openings": str(args.openings), "engine": str(args.engine),
        "candidate_engine": str(args.candidate_engine),
        "baseline_engine": str(args.baseline_engine),
        "each_option": args.each_option,
    }
    if state_path.exists():
        state = json.loads(state_path.read_text())
        mismatches = {key: (state.get(key), value) for key, value in expected.items()
                      if state.get(key) != value}
        if mismatches:
            raise RuntimeError(f"output directory has another configuration: {mismatches}")
        # SPRT runs legitimately finish with fewer games than the cap, so a
        # completed state records how many it actually played.
        if (state.get("status") == "complete"
                and pgn_games(pgn) == state.get("pgn_games", args.games)):
            print(json.dumps(state, indent=2, sort_keys=True))
            return 0
        if pgn.exists():
            pgn.unlink()
    else:
        state = {**expected, "created_utc": datetime.now(timezone.utc).isoformat(),
                 "status": "created"}

    search_limit = [f"tc={args.tc}"] if args.tc else ["tc=inf", f"nodes={args.nodes}"]
    command = [
        str(args.fastchess),
        "-engine", f"cmd={args.candidate_engine}", "name=candidate", f"option.EvalFile={args.candidate}",
        "-engine", f"cmd={args.baseline_engine}", "name=baseline", f"option.EvalFile={args.baseline}",
        "-each", "proto=uci", *search_limit, "option.Threads=1", "option.Hash=16",
        *[f"option.{opt}" for opt in args.each_option],
        "-openings", f"file={args.openings}", "format=epd", "order=random",
        "-repeat", "-games", "2", "-rounds", str(args.games // 2),
        "-concurrency", str(args.concurrency), "-recover",
        "-draw", "movenumber=40", "movecount=8", "score=15",
        "-resign", "movecount=4", "score=700", "-maxmoves", "250",
        "-srand", str(args.seed), "-pgnout", f"file={pgn}", "notation=lan",
        "-log", f"file={args.output / 'fastchess.log'}", "level=warn",
    ]
    if sprt:
        # logistic model: bounds are in the same regular-Elo units every other
        # number in the ledger uses (normalized would report nElo).
        command += ["-sprt", f"elo0={args.sprt_elo0}", f"elo1={args.sprt_elo1}",
                    f"alpha={SPRT_ALPHA}", f"beta={SPRT_BETA}", "model=logistic"]
    state.update(status="running", command=command)
    state_path.write_text(json.dumps(state, indent=2, sort_keys=True) + "\n")
    result = subprocess.run(command, cwd=REPO, text=True, stdout=subprocess.PIPE,
                            stderr=subprocess.STDOUT)
    stdout_log.write_text(result.stdout)
    matches = SUMMARY_RE.findall(result.stdout)
    count = pgn_games(pgn)
    # In SPRT mode a decided match legitimately stops before the --games cap:
    # accept a short run only when the final LLR actually crossed a bound.
    sprt_state = None
    if sprt:
        llr_matches = LLR_RE.findall(result.stdout)
        if llr_matches:
            llr, lower, upper = map(float, llr_matches[-1])
            verdict = ("H1" if llr >= upper else
                       "H0" if llr <= lower else "max-games")
            sprt_state = {"elo0": args.sprt_elo0, "elo1": args.sprt_elo1,
                          "alpha": SPRT_ALPHA, "beta": SPRT_BETA,
                          "llr": llr, "lower": lower, "upper": upper,
                          "verdict": verdict}
    early_stop_ok = sprt_state is not None and sprt_state["verdict"] != "max-games"
    if result.returncode or not matches or (count != args.games and not early_stop_ok):
        state.update(status="failed", returncode=result.returncode, pgn_games=count,
                     output_tail=result.stdout[-4000:])
        state_path.write_text(json.dumps(state, indent=2, sort_keys=True) + "\n")
        raise RuntimeError(f"match failed: rc={result.returncode}, games={count}/{args.games}")
    games, wins, losses, draws = map(int, matches[-1])
    state.update(status="complete", pgn_games=count,
                 candidate_wins=wins, candidate_losses=losses, draws=draws,
                 **estimate(games, wins, losses, draws),
                 paired_bootstrap=paired_bootstrap(pgn, "candidate", args.seed))
    if sprt_state is not None:
        state["sprt"] = sprt_state
    state_path.write_text(json.dumps(state, indent=2, sort_keys=True, allow_nan=True) + "\n")
    print(json.dumps(state, indent=2, sort_keys=True))
    return 0


if __name__ == "__main__":
    try:
        raise SystemExit(main())
    except (OSError, RuntimeError) as error:
        raise SystemExit(f"eval_match: {error}")
