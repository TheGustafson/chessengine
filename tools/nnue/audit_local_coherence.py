#!/usr/bin/env python3
"""Measure evaluator smoothness across legal parent/child transitions.

Random-position fit can improve while alpha-beta strength collapses.  This
audit measures the local quantity search consumes: after a legal move, the
child side-to-move score should approximately negate the parent score.  It
reports the residual ``-E(child) - E(parent)`` by move class and piece, plus
candidate-vs-reference disagreement on that residual.
"""

from __future__ import annotations

import argparse
import json
from pathlib import Path
import subprocess

import chess
import numpy as np


def evaluate(engine: Path, net: Path, fens: list[str]) -> np.ndarray:
    proc = subprocess.run(
        [str(engine.resolve()), "nnue-eval", str(net.resolve())],
        input="\n".join(fens) + "\n",
        text=True,
        stdout=subprocess.PIPE,
        stderr=subprocess.PIPE,
    )
    if proc.returncode:
        raise RuntimeError(f"evaluation failed for {net}: {proc.stderr[-2000:]}")
    values = np.fromstring(proc.stdout, sep=" ", dtype=np.float64)
    if len(values) != len(fens):
        raise RuntimeError(f"{net} returned {len(values)} scores for {len(fens)} FENs")
    return values


def stats(values: list[float] | np.ndarray) -> dict[str, float | int]:
    a = np.abs(np.asarray(values, dtype=np.float64))
    if not len(a):
        return {"count": 0}
    return {
        "count": int(len(a)),
        "abs_mean_cp": float(a.mean()),
        "abs_p50_cp": float(np.quantile(a, 0.50)),
        "abs_p95_cp": float(np.quantile(a, 0.95)),
        "abs_p99_cp": float(np.quantile(a, 0.99)),
        "abs_max_cp": float(a.max()),
    }


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--engine", type=Path, default=Path("target/release/chessengine"))
    parser.add_argument("--fens", type=Path, required=True)
    parser.add_argument("--positions", type=int, default=1000)
    parser.add_argument("--net", action="append", nargs=2, metavar=("NAME", "CENN"), required=True)
    parser.add_argument("--reference", help="net NAME used for residual-disagreement metrics")
    parser.add_argument("--output", type=Path)
    args = parser.parse_args()

    roots: list[chess.Board] = []
    for raw in args.fens.read_text(encoding="utf-8").splitlines():
        raw = raw.strip()
        if not raw or raw.startswith("#"):
            continue
        try:
            board = chess.Board(raw)
        except ValueError:
            continue
        if board.is_valid() and not board.is_game_over(claim_draw=True):
            roots.append(board)
        if len(roots) == args.positions:
            break
    if not roots:
        raise RuntimeError("no valid nonterminal positions")

    all_fens: list[str] = []
    transitions: list[dict[str, object]] = []
    for board in roots:
        parent_idx = len(all_fens)
        all_fens.append(board.fen())
        for move in board.legal_moves:
            piece = board.piece_at(move.from_square)
            capture = board.is_capture(move)
            board.push(move)
            child_idx = len(all_fens)
            all_fens.append(board.fen())
            board.pop()
            transitions.append({
                "parent": parent_idx,
                "child": child_idx,
                "class": "capture" if capture else "quiet",
                "piece": chess.piece_name(piece.piece_type) if piece else "unknown",
            })

    predictions = {
        name: evaluate(args.engine, Path(path), all_fens) for name, path in args.net
    }
    residuals: dict[str, np.ndarray] = {}
    report: dict[str, object] = {
        "root_count": len(roots),
        "transition_count": len(transitions),
        "score_semantics": "side-to-move centipawns; residual=-child-parent",
        "nets": {},
    }
    for name, values in predictions.items():
        residual = np.asarray([
            -values[int(t["child"])] - values[int(t["parent"])] for t in transitions
        ])
        residuals[name] = residual
        by_class = {}
        for cls in ("quiet", "capture"):
            mask = np.asarray([t["class"] == cls for t in transitions])
            by_class[cls] = stats(residual[mask])
        by_piece = {}
        for piece in ("pawn", "knight", "bishop", "rook", "queen", "king"):
            mask = np.asarray([t["piece"] == piece for t in transitions])
            by_piece[piece] = stats(residual[mask])
        report["nets"][name] = {
            "all": stats(residual), "by_class": by_class, "by_piece": by_piece,
        }

    reference = args.reference or args.net[0][0]
    if reference not in residuals:
        raise RuntimeError(f"reference {reference!r} is not a --net name")
    disagreements = {}
    for name, residual in residuals.items():
        if name == reference:
            continue
        delta = residual - residuals[reference]
        quiet = np.asarray([t["class"] == "quiet" for t in transitions])
        disagreements[name] = {
            "all": stats(delta), "quiet": stats(delta[quiet]), "capture": stats(delta[~quiet]),
        }
    report["reference"] = reference
    report["residual_disagreement_vs_reference"] = disagreements

    rendered = json.dumps(report, indent=2, sort_keys=True) + "\n"
    if args.output:
        args.output.parent.mkdir(parents=True, exist_ok=True)
        args.output.write_text(rendered, encoding="utf-8")
    print(rendered, end="")


if __name__ == "__main__":
    main()
