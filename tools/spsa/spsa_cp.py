#!/usr/bin/env python3
"""SPSA over mean centipawn loss instead of game score.

WHY THIS EXISTS. `spsa_tune.py` estimates its gradient from paired fixed-node
mini-matches, so each iteration's signal is `z = (wins-losses)/games`. That is
an extremely noisy estimator, and the noise is what makes SPSA campaigns take
days. Now that a deterministic, paired, load-immune objective exists -- mean cp
loss against an SF18 root-analysis oracle -- the same SPSA math can run on a
vastly better signal:

    instrument            sd of one batch      cost
    32 fixed-node games      ~55 Elo           1.02 G nodes
    cp loss, n=1500          ~18 Elo           0.037 G nodes
    cp loss, n=4000          ~11 Elo           0.100 G nodes

Three times the precision at a twenty-seventh of the cost, which is roughly
250x the signal per node. It is also deterministic and paired: both arms see
identical positions, so position difficulty cancels instead of being averaged
away, and the whole run can be niced onto a busy box without corrupting it.

WHAT IT IS NOT. This minimises **agreement-with-SF18-labels at a fixed node
budget**, which is not Elo. Three specific ways that can mislead, and what is
done about each:

  * the ceiling is Stockfish's own judgement, so this can at best tune us
    toward playing SF18's moves at low nodes -- fine as a direction-finder,
    useless as a ship gate. Everything that survives still needs an SPRT.
  * optimising against a fixed label set overfits it. A held-out split is
    scored every `--holdout-every` iterations and printed alongside the train
    objective; if train falls while holdout does not, stop.
  * the tuning budget is not the game budget. Confirm any winner with
    `screen.py --budgets` before believing it transfers, and prefer a node
    budget with real oracle headroom (`screen.py --oracle-audit`).

    python3 tools/spsa/spsa_cp.py tools/spsa/campaigns/<cfg>.json \
        --iters 300 --positions 1500 --nodes 12500

Config JSON is the same shape `spsa_tune.py` reads, so campaigns are shared:
    {"name": "...", "params": [{"name": "NMP_EVAL_BASE", "min": 100,
     "max": 600, "c_end": 25}, ...]}
"""

from __future__ import annotations

import argparse
import json
import math
import os
import random
import statistics
import sys
from pathlib import Path

REPO = Path(__file__).resolve().parents[2]
sys.path.insert(0, str(REPO / "tools"))

try:
    import chess
    import chess.engine
except ModuleNotFoundError:  # pragma: no cover
    _venv = REPO / ".venv" / "bin" / "python3"
    if _venv.exists() and Path(sys.executable).resolve() != _venv.resolve():
        os.execv(str(_venv), [str(_venv), *sys.argv])
    raise

from screen import ENGINE, NET, assert_net_live  # noqa: E402

ALPHA, GAMMA = 0.602, 0.101


class Evaluator:
    """One long-lived engine process, reconfigured per arm.

    Spawning a process per evaluation would reload an 84 MB net every time --
    two spawns an iteration over hundreds of iterations is a large fraction of
    the run spent on startup rather than search.
    """

    def __init__(self, binary: str, net: str, nodes: int):
        self.eng = chess.engine.SimpleEngine.popen_uci(binary)
        self.eng.configure({"EvalFile": net})
        self.nodes = nodes
        self._arm = 0

    def loss(self, boards, tier, theta: dict[str, float]) -> dict[str, float]:
        """Evaluate one arm. `game=` forces a `ucinewgame` at every arm switch.

        Without it the transposition table and history tables carry over from
        the previous arm, so results depend on evaluation ORDER and two
        identical configurations do not agree. Caught by the no-op check: with
        a persistent engine and no reset, theta == base measured +3.5 cp
        instead of exactly 0. screen.py avoids this by spawning a fresh process
        per arm; a long-lived process has to clear state explicitly.
        """
        for k, v in theta.items():
            self.eng.configure({k: int(round(v))})
        self._arm += 1
        token = object()
        out: dict[str, float] = {}
        for fen, board in boards:
            moves = tier.get(fen)
            if not moves:
                continue
            res = self.eng.play(board, chess.engine.Limit(nodes=self.nodes),
                                game=token)
            if res.move is None:
                continue
            uci = res.move.uci()
            if uci in moves:
                out[fen] = max(moves.values()) - moves[uci]
        return out

    def close(self):
        self.eng.quit()


def paired_mean(a: dict[str, float], b: dict[str, float]) -> float:
    """Mean of (a - b) over positions BOTH scored. Pairing is the whole point:
    unpaired, position difficulty dominates a ~0.2 cp effect."""
    d = [a[f] - b[f] for f in a if f in b]
    return statistics.fmean(d) if d else 0.0


def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("config")
    ap.add_argument("--oracle", default="tools/nnue/calib/oracle-obs.json")
    ap.add_argument("--iters", type=int, default=300)
    ap.add_argument("--positions", type=int, default=1500)
    ap.add_argument("--nodes", type=int, default=12500)
    ap.add_argument("--holdout", type=float, default=0.25)
    ap.add_argument("--holdout-every", type=int, default=25)
    ap.add_argument("--seed", type=int, default=20260727)
    ap.add_argument("--step-scale", type=float, default=3.0,
                    help="multiplies the cp gradient. r_end in the shared campaign "
                         "configs was calibrated for game score z in [-1,1]; a cp "
                         "difference of the same Elo size is numerically smaller, "
                         "so without this the steps are invisible. Watch theta "
                         "actually move before trusting a run.")
    ap.add_argument("--out", default="tools/spsa/out/spsa_cp.jsonl")
    args = ap.parse_args()

    os.chdir(REPO)
    cfg = json.load(open(args.config))
    params = cfg["params"]
    assert_net_live(ENGINE, NET)

    tiers = json.load(open(args.oracle))["tiers"]
    deep = tiers[sorted(tiers, key=int)[-1]]
    rng = random.Random(args.seed)
    fens = sorted(deep)
    rng.shuffle(fens)
    fens = fens[: args.positions]
    cut = int(len(fens) * (1 - args.holdout))
    train = [(f, chess.Board(f)) for f in fens[:cut]]
    hold = [(f, chess.Board(f)) for f in fens[cut:]]
    print(f"{len(train)} train / {len(hold)} holdout positions, "
          f"arms at {args.nodes} nodes, {args.iters} iterations")

    ev = Evaluator(ENGINE, NET, args.nodes)
    try:
        theta = {p["name"]: float(p.get("start", (p["min"] + p["max"]) / 2))
                 for p in params}
        # start from the engine's own defaults unless the config overrides
        import subprocess
        listed = dict(l.split("\t", 1) for l in subprocess.run(
            [ENGINE, "tunables"], text=True, capture_output=True).stdout.splitlines()
            if "\t" in l)
        for p in params:
            if "start" not in p and p["name"] in listed:
                theta[p["name"]] = float(listed[p["name"]])
            if p["name"] not in listed:
                raise SystemExit(f"preflight: '{p['name']}' is not a tunable")
        base = dict(theta)
        T, A = args.iters, 0.1 * args.iters
        Path(args.out).parent.mkdir(parents=True, exist_ok=True)
        log = open(args.out, "a")

        for k in range(1, T + 1):
            plus, minus, delta = {}, {}, {}
            for p in params:
                n = p["name"]
                c_end = p.get("c_end", (p["max"] - p["min"]) / 20)
                c_k = c_end * (T / k) ** GAMMA
                d = rng.choice((-1.0, 1.0))
                delta[n] = (d, c_k, p)
                plus[n] = min(p["max"], max(p["min"], theta[n] + c_k * d))
                minus[n] = min(p["max"], max(p["min"], theta[n] - c_k * d))

            lp = ev.loss(train, deep, plus)
            lm = ev.loss(train, deep, minus)
            y = paired_mean(lp, lm)          # >0 means theta+ is WORSE
            for n, (d, c_k, p) in delta.items():
                r_end = p.get("r_end", 0.002)
                a_k = r_end * c_k ** 2 * ((A + T) / (A + k)) ** ALPHA
                # minimisation: step AGAINST the gradient of cp loss
                theta[n] = min(p["max"], max(p["min"],
                                             theta[n] - (a_k / c_k) * y * args.step_scale * d))

            if k % args.holdout_every == 0 or k == T:
                lh = ev.loss(hold, deep, theta)
                lb = ev.loss(hold, deep, base)
                hd = paired_mean(lh, lb)
                row = {"iter": k, "train_y": round(y, 4),
                       "holdout_vs_base": round(hd, 4),
                       "theta": {n: round(v, 1) for n, v in theta.items()}}
                log.write(json.dumps(row) + "\n")
                log.flush()
                print(f"  it {k:>4}  y {y:+.3f}  holdout vs base {hd:+.3f} cp   "
                      + " ".join(f"{n}={v:.0f}" for n, v in theta.items()),
                      flush=True)
    finally:
        ev.close()

    print("\nfinal theta as a screen.py arm:")
    print("  " + ",".join(f"{n}={int(round(v))}" for n, v in theta.items()))
    print("\nHOLDOUT is the number that matters. If train improves while holdout")
    print("does not, the run is fitting the label set rather than the engine.")
    print("Confirm any winner with `screen.py --budgets` (does the gain survive a")
    print("bigger budget?) and then an SPRT. cp loss is a direction-finder; only")
    print("games decide.")


if __name__ == "__main__":
    main()
