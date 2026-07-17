#!/usr/bin/env python3
"""Local SPSA tuner for ARCHON search parameters (D45, design/14 §9.6).

Classic simultaneous-perturbation SPSA with fishtest/OpenBench-style
schedules, run as paired fixed-node fastchess mini-matches between one tune
build playing both sides (`option.<NAME>` per side). Fixed-node games are
load-immune (T4), so tuning runs at high concurrency on a busy box; the
tuned candidate still needs the equal-time SPRT ship gate (D39) before its
values become defaults.

Schedules (alpha=0.602, gamma=0.101, A=0.1*T):
    c_k = c_end * (T/k)^gamma          (perturbation radius, ends at c_end)
    a_k = r_end*c_end^2 * ((A+T)/(A+k))^alpha
    theta_i += (a_k / c_k_i) * z * delta_i,  z = (wins-losses)/games of the
    theta+ engine over the batch, delta_i in {-1,+1} (Rademacher).

Config JSON:
    {"name": "...", "params": [{"name": "CORR_PAWN_WEIGHT", "min": 4000,
     "max": 28000, "c_end": 550, "r_end": 0.002, "float": false}, ...]}
  c_end defaults to (max-min)/20, r_end to 0.002. Start values come from the
  engine's own `tunables` listing (preflight also catches name typos loudly —
  a typo'd name would otherwise tune nothing while reporting success).
"""

import argparse
import json
import math
import random
import re
import subprocess
import sys
import time
from pathlib import Path

REPO = Path(__file__).resolve().parents[2]
SUMMARY_RE = re.compile(
    r"Games:\s*(\d+),\s*Wins:\s*(\d+),\s*Losses:\s*(\d+),\s*Draws:\s*(\d+)"
)
ALPHA, GAMMA = 0.602, 0.101


def preflight(engine: Path, params: list[dict]) -> dict[str, float]:
    out = subprocess.run([str(engine), "tunables"], text=True,
                         capture_output=True, check=True).stdout
    listed = {}
    for line in out.splitlines():
        name, _, value = line.partition("\t")
        if name:
            listed[name] = value
    theta = {}
    for p in params:
        if p["name"] not in listed:
            sys.exit(f"preflight: '{p['name']}' is not a tunable in {engine}")
        raw = listed[p["name"]]
        theta[p["name"]] = float(raw)
        if not (p["min"] <= theta[p["name"]] <= p["max"]):
            sys.exit(f"preflight: {p['name']} default {raw} outside "
                     f"[{p['min']}, {p['max']}] — fix the config range")
    return theta


def engine_options(values: dict[str, float], params: list[dict]) -> list[str]:
    opts = []
    for p in params:
        v = values[p["name"]]
        rendered = f"{v:.6f}" if p.get("float") else str(int(round(v)))
        opts.append(f"option.{p['name']}={rendered}")
    return opts


def play_batch(args, plus_opts, minus_opts, seed) -> tuple[int, int, int]:
    command = [
        str(args.fastchess),
        "-engine", f"cmd={args.engine}", "name=plus", *plus_opts,
        "-engine", f"cmd={args.engine}", "name=minus", *minus_opts,
        "-each", "proto=uci", "tc=inf", f"nodes={args.nodes}",
        "option.Threads=1", "option.Hash=16",
        f"option.EvalFile={args.eval_file}",
        "-openings", f"file={args.book}", "format=epd", "order=random",
        "-repeat", "-games", "2", "-rounds", str(args.pairs),
        "-concurrency", str(args.concurrency), "-recover",
        "-draw", "movenumber=40", "movecount=8", "score=15",
        "-resign", "movecount=4", "score=700", "-maxmoves", "250",
        "-srand", str(seed),
    ]
    result = subprocess.run(command, cwd=REPO, text=True,
                            stdout=subprocess.PIPE, stderr=subprocess.STDOUT)
    matches = SUMMARY_RE.findall(result.stdout)
    if result.returncode or not matches:
        raise RuntimeError(
            f"batch failed rc={result.returncode}:\n{result.stdout[-2000:]}")
    games, wins, losses, _draws = map(int, matches[-1])
    return games, wins, losses


def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("--config", type=Path, required=True)
    ap.add_argument("--engine", type=Path, required=True,
                    help="tune-feature build of chessengine")
    ap.add_argument("--fastchess", type=Path,
                    default=REPO / "engines/bin/fastchess")
    ap.add_argument("--book", type=Path,
                    default=REPO / "data/nnue/openings/UHO_Lichess_4852_v1.epd")
    ap.add_argument("--out", type=Path, required=True)
    ap.add_argument("--eval-file", type=Path, required=True,
                    help="net BOTH sides play (D39: tune on the shipped net)")
    ap.add_argument("--iters", type=int, default=1000)
    ap.add_argument("--pairs", type=int, default=4, help="game pairs per batch")
    ap.add_argument("--nodes", type=int, default=20000)
    ap.add_argument("--concurrency", type=int, default=10)
    ap.add_argument("--seed", type=int, default=20260717)
    ap.add_argument("--override", action="append", default=[],
                    metavar="NAME=VALUE",
                    help="start theta override (repeatable) — for detuned-"
                         "start positive controls of the harness itself")
    ap.add_argument("--r-end", type=float, default=None,
                    help="override every param's r_end (default from config "
                         "or 0.002; larger = coarser but more decisive steps)")
    args = ap.parse_args()

    config = json.loads(args.config.read_text())
    params = config["params"]
    for p in params:
        p.setdefault("c_end", (p["max"] - p["min"]) / 20.0)
        p.setdefault("r_end", 0.002)
        if args.r_end is not None:
            p["r_end"] = args.r_end

    theta = preflight(args.engine, params)
    for spec in args.override:
        name, _, raw = spec.partition("=")
        if name not in theta:
            sys.exit(f"--override: '{name}' is not in this campaign")
        value = float(raw)
        matching = next(p for p in params if p["name"] == name)
        if not (matching["min"] <= value <= matching["max"]):
            sys.exit(f"--override: {name}={value} outside "
                     f"[{matching['min']}, {matching['max']}]")
        theta[name] = value
        print(f"[spsa {config['name']}] start override: {name} = {value}",
              flush=True)
    start = dict(theta)
    args.out.mkdir(parents=True, exist_ok=True)
    history = (args.out / "history.jsonl").open("a")
    rng = random.Random(args.seed)
    big_a = 0.1 * args.iters
    t0 = time.time()
    total_games = 0

    for k in range(1, args.iters + 1):
        delta = {p["name"]: rng.choice((-1, 1)) for p in params}
        c_k = {p["name"]: p["c_end"] * (args.iters / k) ** GAMMA for p in params}
        plus, minus = {}, {}
        for p in params:
            n = p["name"]
            plus[n] = min(p["max"], max(p["min"], theta[n] + c_k[n] * delta[n]))
            minus[n] = min(p["max"], max(p["min"], theta[n] - c_k[n] * delta[n]))
        games, wins, losses = play_batch(
            args, engine_options(plus, params), engine_options(minus, params),
            seed=args.seed + k)
        total_games += games
        z = (wins - losses) / games
        for p in params:
            n = p["name"]
            a_k = p["r_end"] * p["c_end"] ** 2 * \
                ((big_a + args.iters) / (big_a + k)) ** ALPHA
            theta[n] = min(p["max"], max(p["min"],
                           theta[n] + (a_k / c_k[n]) * z * delta[n]))
        history.write(json.dumps({"k": k, "z": z, "games": games,
                                  "theta": theta}) + "\n")
        history.flush()
        if k % 25 == 0 or k == args.iters:
            rate = total_games / (time.time() - t0)
            drift = {n: round(theta[n] - start[n], 1) for n in theta
                     if abs(theta[n] - start[n]) >= 1}
            print(f"[spsa {config['name']}] k={k}/{args.iters} "
                  f"games={total_games} ({rate:.0f}/s) drift={drift}",
                  flush=True)

    final = {p["name"]: (theta[p["name"]] if p.get("float")
                         else int(round(theta[p["name"]])))
             for p in params}
    (args.out / "theta_final.json").write_text(json.dumps(
        {"name": config["name"], "start": start, "final": final,
         "iters": args.iters, "pairs": args.pairs, "nodes": args.nodes,
         "total_games": total_games, "seed": args.seed}, indent=2) + "\n")
    print(f"[spsa {config['name']}] DONE: {json.dumps(final)}", flush=True)


if __name__ == "__main__":
    main()
