#!/usr/bin/env python3
"""Audit classic CENN evaluators on matched FEN-text validation corpora."""

from __future__ import annotations

import argparse
import json
import math
from pathlib import Path
import subprocess

import numpy as np


def read_texts(directory: Path) -> tuple[list[str], np.ndarray, np.ndarray]:
    fens: list[str] = []
    scores: list[int] = []
    results: list[float] = []
    for path in sorted(directory.glob("shard-*.txt")):
        with path.open(encoding="utf-8") as handle:
            for line in handle:
                fen, score, result = (part.strip() for part in line.split("|"))
                black = fen.split()[1] == "b"
                fens.append(fen)
                scores.append(-int(score) if black else int(score))
                white_result = float(result)
                results.append(1.0 - white_result if black else white_result)
    if not fens:
        raise RuntimeError(f"no rows in {directory}")
    return fens, np.asarray(scores, dtype=np.float64), np.asarray(results, dtype=np.float64)


def sigmoid(values: np.ndarray, scale: float = 400.0) -> np.ndarray:
    clipped = np.clip(values / scale, -40.0, 40.0)
    return 1.0 / (1.0 + np.exp(-clipped))


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--engine", type=Path, default=Path("target/release/chessengine"))
    parser.add_argument("--corpus-root", type=Path, required=True)
    parser.add_argument("--net", action="append", nargs=2, metavar=("NAME", "CENN"), required=True)
    parser.add_argument("--output", type=Path)
    args = parser.parse_args()

    corpora: dict[str, tuple[list[str], np.ndarray, np.ndarray]] = {}
    for teacher in ("sf16", "reckless", "stockfish18"):
        corpora[teacher] = read_texts(args.corpus_root / teacher / "val")
    reference_fens = corpora["sf16"][0]
    for teacher, (fens, _, _) in corpora.items():
        if fens != reference_fens:
            raise RuntimeError(f"FEN order differs for {teacher}")

    fen_input = "\n".join(reference_fens) + "\n"
    report: dict[str, object] = {"rows": len(reference_fens), "nets": {}}
    for name, net_string in args.net:
        net = Path(net_string)
        proc = subprocess.run(
            [str(args.engine.resolve()), "nnue-eval", str(net.resolve())],
            input=fen_input, text=True, stdout=subprocess.PIPE, stderr=subprocess.PIPE,
        )
        if proc.returncode:
            raise RuntimeError(f"{name} evaluation failed: {proc.stderr[-2000:]}")
        prediction = np.fromstring(proc.stdout, sep=" ", dtype=np.float64)
        if len(prediction) != len(reference_fens):
            raise RuntimeError(f"{name} returned {len(prediction)} values")
        probability = sigmoid(prediction)
        metrics: dict[str, object] = {}
        for teacher, (_, label, result) in corpora.items():
            target = 0.3 * result + 0.7 * sigmoid(label)
            metrics[teacher] = {
                "training_target_mse": float(np.mean((probability - target) ** 2)),
                "outcome_brier": float(np.mean((probability - result) ** 2)),
                "cp_mae": float(np.mean(np.abs(prediction - label))),
                "cp_rmse": float(np.sqrt(np.mean((prediction - label) ** 2))),
                "cp_pearson": float(np.corrcoef(prediction, label)[0, 1]),
            }
        report["nets"][name] = {
            "path": str(net.resolve()),
            "prediction_mean": float(prediction.mean()),
            "prediction_std": float(prediction.std()),
            "metrics": metrics,
        }
    rendered = json.dumps(report, indent=2, sort_keys=True) + "\n"
    if args.output:
        args.output.parent.mkdir(parents=True, exist_ok=True)
        args.output.write_text(rendered, encoding="utf-8")
    print(rendered, end="")


if __name__ == "__main__":
    main()
