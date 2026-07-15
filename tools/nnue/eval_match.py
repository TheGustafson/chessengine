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
    args = parser.parse_args()
    args.candidate_engine = args.candidate_engine or args.engine
    args.baseline_engine = args.baseline_engine or args.engine
    if args.nodes is None and args.tc is None:
        args.nodes = 2000
    if args.games <= 0 or args.games % 2:
        parser.error("--games must be a positive even number")
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
        "openings": str(args.openings), "engine": str(args.engine),
        "candidate_engine": str(args.candidate_engine),
        "baseline_engine": str(args.baseline_engine),
    }
    if state_path.exists():
        state = json.loads(state_path.read_text())
        mismatches = {key: (state.get(key), value) for key, value in expected.items()
                      if state.get(key) != value}
        if mismatches:
            raise RuntimeError(f"output directory has another configuration: {mismatches}")
        if state.get("status") == "complete" and pgn_games(pgn) == args.games:
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
        "-openings", f"file={args.openings}", "format=epd", "order=random",
        "-repeat", "-games", "2", "-rounds", str(args.games // 2),
        "-concurrency", str(args.concurrency), "-recover",
        "-draw", "movenumber=40", "movecount=8", "score=15",
        "-resign", "movecount=4", "score=700", "-maxmoves", "250",
        "-srand", str(args.seed), "-pgnout", f"file={pgn}", "notation=lan",
        "-log", f"file={args.output / 'fastchess.log'}", "level=warn",
    ]
    state.update(status="running", command=command)
    state_path.write_text(json.dumps(state, indent=2, sort_keys=True) + "\n")
    result = subprocess.run(command, cwd=REPO, text=True, stdout=subprocess.PIPE,
                            stderr=subprocess.STDOUT)
    stdout_log.write_text(result.stdout)
    matches = SUMMARY_RE.findall(result.stdout)
    count = pgn_games(pgn)
    if result.returncode or not matches or count != args.games:
        state.update(status="failed", returncode=result.returncode, pgn_games=count,
                     output_tail=result.stdout[-4000:])
        state_path.write_text(json.dumps(state, indent=2, sort_keys=True) + "\n")
        raise RuntimeError(f"match failed: rc={result.returncode}, games={count}/{args.games}")
    games, wins, losses, draws = map(int, matches[-1])
    state.update(status="complete", candidate_wins=wins, candidate_losses=losses, draws=draws,
                 **estimate(games, wins, losses, draws),
                 paired_bootstrap=paired_bootstrap(pgn, "candidate", args.seed))
    state_path.write_text(json.dumps(state, indent=2, sort_keys=True, allow_nan=True) + "\n")
    print(json.dumps(state, indent=2, sort_keys=True))
    return 0


if __name__ == "__main__":
    try:
        raise SystemExit(main())
    except (OSError, RuntimeError) as error:
        raise SystemExit(f"eval_match: {error}")
