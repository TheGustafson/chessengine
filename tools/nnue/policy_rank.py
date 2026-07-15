#!/usr/bin/env python3
"""Measure CPOL ordering quality against sparse Lc0 policy targets.

This is the fail-fast gate between training loss and chess games. It evaluates
the packed runtime artifacts through ``chessengine policy-eval`` and reports
metrics only over moves present in the teacher's legal policy support, matching
how move ordering consumes the logits more closely than 4096-way training CE.

Example:
  uv run --no-project --with numpy --with python-chess python \
    tools/nnue/policy_rank.py DATA EVAL.cenn HEAD.cpol ENGINE --rows 2000
"""

from __future__ import annotations

import argparse
import math
import subprocess
from pathlib import Path

import chess
import numpy as np


def board_from_features(row: np.ndarray) -> chess.Board:
    board = chess.Board.empty()
    for raw in row:
        feature = int(raw)
        if feature < 0:
            break
        colour, rem = divmod(feature, 384)
        piece_type, square = divmod(rem, 64)
        board.set_piece_at(square, chess.Piece(piece_type + 1, colour == 0))
    board.turn = chess.WHITE
    return board


def aggregate_targets(moves: np.ndarray, probabilities: np.ndarray) -> tuple[list[int], np.ndarray]:
    combined: dict[int, float] = {}
    for move, probability in zip(moves, probabilities):
        move = int(move)
        if move < 0:
            break
        combined[move] = combined.get(move, 0.0) + float(probability)
    indices = list(combined)
    target = np.asarray([combined[index] for index in indices], dtype=np.float64)
    target /= target.sum()
    return indices, target


def from_to_uci(index: int) -> str:
    return chess.square_name(index // 64) + chess.square_name(index % 64)


def log_softmax(values: np.ndarray) -> np.ndarray:
    maximum = values.max()
    return values - maximum - math.log(np.exp(values - maximum).sum())


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("data", type=Path)
    parser.add_argument("eval", type=Path)
    parser.add_argument("policy", type=Path)
    parser.add_argument("engine", type=Path)
    parser.add_argument("--rows", type=int, default=2000)
    parser.add_argument("--seed", type=int, default=20260714)
    parser.add_argument("--side", choices=("white", "black", "both"), default="both")
    parser.add_argument(
        "--transpose-scores",
        action="store_true",
        help="diagnostic: query runtime score(to,from) for each target score(from,to)",
    )
    args = parser.parse_args()

    sides = ("white", "black") if args.side == "both" else (args.side,)
    rng = np.random.default_rng(args.seed)
    examples: list[tuple[np.ndarray, np.ndarray, np.ndarray]] = []
    per_side = max(1, args.rows // len(sides))
    for side in sides:
        root = args.data / side
        features = np.load(root / "feats.npy", mmap_mode="r")
        moves = np.load(root / "moves.npy", mmap_mode="r")
        probabilities = np.load(root / "probs.npy", mmap_mode="r")
        chosen = rng.choice(len(features), size=min(per_side, len(features)), replace=False)
        examples.extend((features[i], moves[i], probabilities[i]) for i in chosen)

    lines: list[str] = []
    prepared: list[tuple[chess.Board, list[int], np.ndarray]] = []
    for features, moves, probabilities in examples:
        board = board_from_features(features)
        indices, target = aggregate_targets(moves, probabilities)
        if len(indices) < 2:
            continue
        queried = [((index % 64) * 64 + index // 64) for index in indices] if args.transpose_scores else indices
        lines.append(board.fen() + "|" + " ".join(from_to_uci(index) for index in queried))
        prepared.append((board, indices, target))

    command = [str(args.engine), "policy-eval", str(args.eval), str(args.policy)]
    result = subprocess.run(
        command,
        input="\n".join(lines) + "\n",
        text=True,
        stdout=subprocess.PIPE,
        stderr=subprocess.PIPE,
        check=True,
    )
    output = result.stdout.splitlines()
    if len(output) != len(prepared):
        raise SystemExit(f"runtime returned {len(output)} rows for {len(prepared)} inputs")

    top1 = top3 = top5 = 0.0
    teacher_top1 = 0
    reciprocal_rank = 0.0
    cross_entropy = 0.0
    quiet_top1 = quiet_count = 0
    score_ranges: list[float] = []
    support_sum = 0
    teacher_entropy = 0.0
    teacher_best_mass = 0.0
    uniform_cross_entropy = 0.0
    for (board, indices, target), line in zip(prepared, output):
        logits = np.fromstring(line, sep="\t", dtype=np.float64)
        if len(logits) != len(indices) or not np.isfinite(logits).all():
            raise SystemExit("runtime emitted wrong-sized or non-finite logits")
        order = np.argsort(-logits)
        support_sum += len(indices)
        teacher_entropy -= float(np.dot(target, np.log(np.maximum(target, 1e-300))))
        teacher_best_mass += float(target.max())
        uniform_cross_entropy += math.log(len(indices))
        teacher = int(np.argmax(target))
        rank = int(np.flatnonzero(order == teacher)[0]) + 1
        teacher_top1 += rank == 1
        reciprocal_rank += 1.0 / rank
        top1 += float(target[order[:1]].sum())
        top3 += float(target[order[:3]].sum())
        top5 += float(target[order[:5]].sum())
        cross_entropy -= float(np.dot(target, log_softmax(logits)))
        score_ranges.append(float(logits.max() - logits.min()))

        quiet = []
        for position, index in enumerate(indices):
            src, dst = index // 64, index % 64
            piece = board.piece_at(src)
            capture = board.piece_at(dst) is not None
            pawn_diagonal = piece is not None and piece.piece_type == chess.PAWN and src % 8 != dst % 8
            if not capture and not pawn_diagonal:
                quiet.append(position)
        if len(quiet) >= 2:
            quiet_count += 1
            qteacher = quiet[int(np.argmax(target[quiet]))]
            qbest = quiet[int(np.argmax(logits[quiet]))]
            quiet_top1 += qteacher == qbest

    count = len(prepared)
    print(f"rows={count}")
    print(f"mean_legal_support={support_sum / count:.3f}")
    print(f"teacher_entropy={teacher_entropy / count:.6f}")
    print(f"teacher_best_mass={teacher_best_mass / count:.6f}")
    print(f"uniform_cross_entropy={uniform_cross_entropy / count:.6f}")
    print(f"teacher_top1_accuracy={teacher_top1 / count:.6f}")
    print(f"teacher_top1_mrr={reciprocal_rank / count:.6f}")
    print(f"target_mass_at_1={top1 / count:.6f}")
    print(f"target_mass_at_3={top3 / count:.6f}")
    print(f"target_mass_at_5={top5 / count:.6f}")
    print(f"legal_support_cross_entropy={cross_entropy / count:.6f}")
    print(f"quiet_teacher_top1_accuracy={quiet_top1 / max(1, quiet_count):.6f} rows={quiet_count}")
    print(f"median_legal_logit_range={np.median(score_ranges):.6f}")


if __name__ == "__main__":
    main()
