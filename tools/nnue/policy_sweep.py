#!/usr/bin/env python3
"""Reproducible fixed-node policy-weight sweep against an unchanged evaluator.

The first FastChess engine is always the policy-off control.  The second uses
POLICY_NET and, when supplied, POLICY_TRUNK.  This makes it possible to retain
classic net-0 evaluation while attributing any change solely to move ordering.
Runs are resumable, but an incomplete PGN is deleted because FastChess appends.
"""

from __future__ import annotations

import argparse
import json
import math
import os
from pathlib import Path
import re
import random
import shlex
import subprocess
import sys
from datetime import datetime, timezone


REPO = Path(__file__).resolve().parents[2]
SUMMARY_RE = re.compile(
    r"Games:\s*(\d+),\s*Wins:\s*(\d+),\s*Losses:\s*(\d+),"
    r"\s*Draws:\s*(\d+),\s*Points:"
)


def absolute(value: str) -> Path:
    path = Path(value)
    return path if path.is_absolute() else (REPO / path).resolve()


def run_checked(command: list[str] | list[Path], **kwargs) -> subprocess.CompletedProcess[str]:
    printable = " ".join(shlex.quote(str(part)) for part in command)
    print(f"$ {printable}", flush=True)
    result = subprocess.run([str(part) for part in command], **kwargs)
    if result.returncode:
        output = (result.stdout or "") + (result.stderr or "")
        raise RuntimeError(f"command failed ({result.returncode}): {printable}\n{output}")
    return result


def preflight(args: argparse.Namespace, wrapper: Path) -> None:
    trunk = args.trunk or args.eval
    probe = (
        "rnbqkbnr/pppppppp/8/8/8/8/PPPPPPPP/RNBQKBNR w KQkq - 0 1"
        "|e2e4 d2d4 g1f3\n"
    )
    oracle = run_checked(
        [args.engine, "policy-eval", trunk, args.policy],
        input=probe,
        text=True,
        capture_output=True,
        cwd=REPO,
        timeout=60,
    )
    try:
        scores = [float(value) for value in oracle.stdout.split()]
    except ValueError as error:
        raise RuntimeError(f"policy oracle returned non-numeric output: {oracle.stdout}") from error
    if len(scores) != 3 or not all(math.isfinite(value) for value in scores):
        raise RuntimeError(f"policy oracle returned invalid scores: {oracle.stdout}")

    uci = (
        f"uci\nsetoption name EvalFile value {args.eval}\nisready\n"
        "position startpos\ngo nodes 200\nquit\n"
    )
    smoke = run_checked(
        [wrapper], input=uci, text=True, capture_output=True, cwd=REPO, timeout=60
    )
    if "readyok" not in smoke.stdout or "bestmove " not in smoke.stdout:
        raise RuntimeError(f"policy UCI smoke did not complete:\n{smoke.stdout}\n{smoke.stderr}")
    if "failed to load POLICY_" in smoke.stderr:
        raise RuntimeError(f"policy UCI smoke rejected an artifact:\n{smoke.stderr}")


def pgn_games(path: Path) -> int:
    if not path.exists():
        return 0
    return path.read_text(errors="replace").count('[Result "')


def estimate(games: int, control_wins: int, control_losses: int, draws: int) -> dict:
    policy_score = (control_losses + 0.5 * draws) / games
    if 0.0 < policy_score < 1.0:
        elo = 400.0 * math.log10(policy_score / (1.0 - policy_score))
    else:
        elo = math.inf if policy_score == 1.0 else -math.inf
    values = [0.0] * control_wins + [1.0] * control_losses + [0.5] * draws
    if games > 1:
        variance = sum((value - policy_score) ** 2 for value in values) / (games - 1)
        score_se = math.sqrt(variance / games)
    else:
        score_se = math.inf
    if 0.0 < policy_score < 1.0:
        elo_se = 400.0 / math.log(10.0) * score_se / (policy_score * (1.0 - policy_score))
    else:
        elo_se = math.inf
    return {
        "policy_wins": control_losses,
        "policy_losses": control_wins,
        "draws": draws,
        "policy_score": policy_score,
        "policy_elo": elo,
        "policy_elo_se": elo_se,
    }


def paired_bootstrap(path: Path, policy_name: str, seed: int, samples: int = 20_000) -> dict:
    """Bootstrap opening-pair scores, preserving the colour-swapped unit."""
    headers: list[dict[str, str]] = []
    current: dict[str, str] = {}
    for line in path.read_text(errors="replace").splitlines():
        match = re.match(r'^\[([^ ]+) "(.*)"\]$', line)
        if match:
            key, value = match.groups()
            if key == "Event" and current:
                headers.append(current)
                current = {}
            current[key] = value
    if current:
        headers.append(current)
    if len(headers) % 2:
        raise RuntimeError(f"paired bootstrap requires an even PGN game count: {len(headers)}")

    def policy_points(game: dict[str, str]) -> float:
        result = game.get("Result")
        if result == "1/2-1/2":
            return 0.5
        if result not in ("1-0", "0-1"):
            raise RuntimeError(f"paired bootstrap found invalid result: {result}")
        policy_is_white = game.get("White") == policy_name
        policy_won = (result == "1-0") == policy_is_white
        return float(policy_won)

    pairs = [
        (policy_points(headers[i]) + policy_points(headers[i + 1])) / 2.0
        for i in range(0, len(headers), 2)
    ]
    rng = random.Random(seed)
    means = sorted(
        sum(pairs[rng.randrange(len(pairs))] for _ in pairs) / len(pairs)
        for _ in range(samples)
    )

    def quantile(p: float) -> float:
        return means[round(p * (len(means) - 1))]

    def elo(score: float) -> float:
        if score <= 0.0:
            return -math.inf
        if score >= 1.0:
            return math.inf
        return 400.0 * math.log10(score / (1.0 - score))

    low, high = quantile(0.025), quantile(0.975)
    return {
        "pairs": len(pairs),
        "samples": samples,
        "score_mean": sum(pairs) / len(pairs),
        "score_ci95": [low, high],
        "elo_ci95": [elo(low), elo(high)],
    }


def save(path: Path, state: dict) -> None:
    path.write_text(json.dumps(state, indent=2, sort_keys=True, allow_nan=True) + "\n")


def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--eval", required=True, type=absolute, help="unchanged evaluator CENN")
    parser.add_argument("--policy", required=True, type=absolute, help="CPOL policy head")
    parser.add_argument("--trunk", type=absolute, help="independent D40 policy trunk")
    parser.add_argument("--weights", nargs="+", required=True, type=int)
    parser.add_argument("--games", type=int, default=40)
    limit = parser.add_mutually_exclusive_group()
    limit.add_argument("--nodes", type=int, help="fixed nodes per move (default: 2000)")
    limit.add_argument("--tc", help="FastChess time control, e.g. 1+0.01")
    parser.add_argument("--concurrency", type=int, default=1)
    parser.add_argument("--min-depth", type=int, help="POLICY_MIN_DEPTH search gate")
    parser.add_argument("--max-ply", type=int, help="POLICY_MAX_PLY root-distance gate")
    parser.add_argument("--root-policy", action="store_true", help="cache one policy prior at root")
    parser.add_argument("--captures", action="store_true", help="also order captures with policy")
    parser.add_argument("--capture-weight", type=int, help="POLICY_CAPTURE_W (defaults to weight)")
    parser.add_argument("--seed", type=int, default=20260714)
    parser.add_argument(
        "--openings", type=absolute,
        default=absolute("tests/data/openings/openings-8ply.epd"),
    )
    parser.add_argument("--timeout", type=int, default=900)
    parser.add_argument("--output", required=True, type=absolute)
    parser.add_argument("--engine", type=absolute, default=absolute("target/release/chessengine"))
    parser.add_argument("--fastchess", type=absolute, default=absolute("engines/bin/fastchess"))
    args = parser.parse_args()
    if args.nodes is None and args.tc is None:
        args.nodes = 2000
    if args.games <= 0 or args.games % 2:
        parser.error("--games must be a positive even number")
    if args.nodes is not None and args.nodes <= 0:
        parser.error("--nodes must be positive")
    if args.max_ply is not None and args.max_ply < 0:
        parser.error("--max-ply must be non-negative")
    for path in [args.eval, args.policy, args.openings, args.engine, args.fastchess]:
        if not path.exists():
            parser.error(f"missing required path: {path}")
    if args.trunk is not None and not args.trunk.exists():
        parser.error(f"missing policy trunk: {args.trunk}")

    args.output.mkdir(parents=True, exist_ok=True)
    state_path = args.output / "summary.json"
    if state_path.exists():
        state = json.loads(state_path.read_text())
        state.setdefault("min_depth", None)
        state.setdefault("max_ply", None)
        state.setdefault("root_policy", False)
        state.setdefault("tc", None)
        state.setdefault("captures", False)
        state.setdefault("capture_weight", None)
        expected = {
            "eval": str(args.eval), "policy": str(args.policy),
            "trunk": str(args.trunk) if args.trunk else None,
            "games": args.games, "nodes": args.nodes, "tc": args.tc,
            "concurrency": args.concurrency, "seed": args.seed,
            "openings": str(args.openings), "min_depth": args.min_depth,
            "max_ply": args.max_ply,
            "root_policy": args.root_policy,
            "captures": args.captures, "capture_weight": args.capture_weight,
        }
        mismatches = {
            key: (state.get(key), value)
            for key, value in expected.items()
            if state.get(key) != value
        }
        if mismatches:
            raise RuntimeError(f"output directory belongs to a different configuration: {mismatches}")
    else:
        state = {
            "created_utc": datetime.now(timezone.utc).isoformat(),
            "eval": str(args.eval),
            "policy": str(args.policy),
            "trunk": str(args.trunk) if args.trunk else None,
            "games": args.games,
            "nodes": args.nodes,
            "tc": args.tc,
            "concurrency": args.concurrency,
            "min_depth": args.min_depth,
            "max_ply": args.max_ply,
            "root_policy": args.root_policy,
            "captures": args.captures,
            "capture_weight": args.capture_weight,
            "seed": args.seed,
            "openings": str(args.openings),
            "results": {},
        }
        save(state_path, state)

    for weight in args.weights:
        key = str(weight)
        run_dir = args.output / f"w{weight}"
        run_dir.mkdir(exist_ok=True)
        pgn = run_dir / "games.pgn"
        log = run_dir / "fastchess.log"
        stdout_log = run_dir / "stdout.log"
        wrapper = run_dir / "engine-policy"
        exports = [f"export POLICY_NET={shlex.quote(str(args.policy))}"]
        if args.trunk:
            exports.append(f"export POLICY_TRUNK={shlex.quote(str(args.trunk))}")
        exports.append(f"export POLICY_W={weight}")
        if args.min_depth is not None:
            exports.append(f"export POLICY_MIN_DEPTH={args.min_depth}")
        if args.max_ply is not None:
            exports.append(f"export POLICY_MAX_PLY={args.max_ply}")
        if args.root_policy:
            exports.append("export POLICY_ROOT=1")
        if args.captures:
            exports.append("export POLICY_CAPTURES=1")
        if args.capture_weight is not None:
            exports.append(f"export POLICY_CAPTURE_W={args.capture_weight}")
        wrapper.write_text(
            "#!/usr/bin/env bash\n" + "\n".join(exports) + "\n" +
            f"exec {shlex.quote(str(args.engine))} \"$@\"\n"
        )
        wrapper.chmod(0o755)

        record = state["results"].get(key, {})
        if record.get("status") == "complete" and pgn_games(pgn) == args.games:
            print(f"skip complete weight {weight}: {record}", flush=True)
            continue
        if pgn.exists():
            print(f"deleting incomplete PGN for weight {weight} ({pgn_games(pgn)} games)", flush=True)
            pgn.unlink()

        preflight(args, wrapper)
        search_limit = [f"tc={args.tc}"] if args.tc else ["tc=inf", f"nodes={args.nodes}"]
        command = [
            args.fastchess,
            "-engine", f"cmd={args.engine}", "name=policy-off", f"option.EvalFile={args.eval}",
            "-engine", f"cmd={wrapper}", f"name=policy-w{weight}", f"option.EvalFile={args.eval}",
            "-each", "proto=uci", *search_limit, "option.Threads=1", "option.Hash=16",
            "-openings", f"file={args.openings}", "format=epd", "order=random",
            "-repeat", "-games", "2", "-rounds", str(args.games // 2),
            "-concurrency", str(args.concurrency), "-recover",
            "-draw", "movenumber=40", "movecount=8", "score=15",
            "-resign", "movecount=4", "score=700", "-maxmoves", "250",
            "-srand", str(args.seed),
            "-pgnout", f"file={pgn}", "notation=lan",
            "-log", f"file={log}", "level=warn",
        ]
        printable = " ".join(shlex.quote(str(part)) for part in command)
        print(f"$ {printable}", flush=True)
        record = {"status": "running", "command": printable}
        state["results"][key] = record
        save(state_path, state)
        try:
            result = subprocess.run(
                [str(part) for part in command], cwd=REPO, text=True,
                stdout=subprocess.PIPE, stderr=subprocess.STDOUT, timeout=args.timeout,
            )
        except subprocess.TimeoutExpired as error:
            stdout_log.write_text(error.stdout or "")
            record.update(status="timeout", pgn_games=pgn_games(pgn))
            save(state_path, state)
            raise RuntimeError(f"weight {weight} timed out after {args.timeout}s") from error
        stdout_log.write_text(result.stdout)
        matches = SUMMARY_RE.findall(result.stdout)
        count = pgn_games(pgn)
        if result.returncode or not matches or count != args.games:
            record.update(
                status="failed", returncode=result.returncode, pgn_games=count,
                output_tail=result.stdout[-4000:],
            )
            save(state_path, state)
            raise RuntimeError(
                f"weight {weight} failed: rc={result.returncode}, summaries={matches[-1:]}, "
                f"PGN games={count}/{args.games}; see {stdout_log}"
            )
        games, wins, losses, draws = map(int, matches[-1])
        if games != args.games or wins + losses + draws != games:
            raise RuntimeError(f"weight {weight} malformed FastChess summary: {matches[-1]}")
        record.update(
            status="complete", control_wins=wins, control_losses=losses,
            **estimate(games, wins, losses, draws),
            paired_bootstrap=paired_bootstrap(pgn, f"policy-w{weight}", args.seed + weight),
        )
        save(state_path, state)
        print(f"weight {weight}: {record}", flush=True)
    return 0


if __name__ == "__main__":
    try:
        raise SystemExit(main())
    except (RuntimeError, OSError) as error:
        print(f"error: {error}", file=sys.stderr)
        raise SystemExit(1)
