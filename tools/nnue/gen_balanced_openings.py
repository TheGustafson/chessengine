#!/usr/bin/env python3
"""Generate a deterministic, diverse, engine-balanced EPD opening book.

At each ply, sample among Stockfish's near-best MultiPV moves, then retain only
final positions within a configurable evaluation window. The result is intended
for local A/B testing where the tiny hand-written 43-line smoke book would
otherwise repeat openings in longer confirmations.
"""

from __future__ import annotations

import argparse
import hashlib
import json
from pathlib import Path
import random

import chess
import chess.engine


def sha256(path: Path) -> str:
    h = hashlib.sha256()
    with path.open("rb") as stream:
        while chunk := stream.read(8 * 1024 * 1024):
            h.update(chunk)
    return h.hexdigest()


def score_cp(info: dict, turn: chess.Color) -> int:
    value = info["score"].pov(turn).score(mate_score=10_000)
    return value if value is not None else 10_000


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("output", type=Path)
    parser.add_argument("--stockfish", type=Path, default=Path("engines/stockfish-datagen"))
    parser.add_argument("--count", type=int, default=512)
    parser.add_argument("--plies", type=int, default=8)
    parser.add_argument("--multipv", type=int, default=8)
    parser.add_argument("--move-nodes", type=int, default=500)
    parser.add_argument("--gate-nodes", type=int, default=5000)
    parser.add_argument("--max-move-loss", type=int, default=100)
    parser.add_argument("--balance", type=int, default=120)
    parser.add_argument("--seed", type=int, default=20260714)
    parser.add_argument("--max-attempts", type=int, default=100_000)
    args = parser.parse_args()
    if args.count <= 0 or args.plies <= 0 or args.multipv <= 0:
        parser.error("count, plies, and multipv must be positive")

    rng = random.Random(args.seed)
    engine = chess.engine.SimpleEngine.popen_uci(str(args.stockfish), timeout=30)
    engine.configure({"Threads": 1, "Hash": 64})
    positions: list[str] = []
    seen: set[str] = set()
    attempts = 0
    rejected_terminal = rejected_balance = rejected_duplicate = 0
    try:
        while len(positions) < args.count and attempts < args.max_attempts:
            attempts += 1
            board = chess.Board()
            valid = True
            for _ in range(args.plies):
                infos = engine.analyse(
                    board, chess.engine.Limit(nodes=args.move_nodes),
                    multipv=args.multipv, game=object(),
                )
                if not isinstance(infos, list):
                    infos = [infos]
                candidates = []
                for rank, info in enumerate(infos):
                    pv = info.get("pv") or []
                    if pv:
                        candidates.append((rank, score_cp(info, board.turn), pv[0]))
                if not candidates:
                    valid = False
                    break
                best = max(score for _, score, _ in candidates)
                candidates = [row for row in candidates if best - row[1] <= args.max_move_loss]
                # Favor main lines while still exploring all near-best moves.
                weights = [1.0 / (1.0 + rank) for rank, _, _ in candidates]
                move = rng.choices([row[2] for row in candidates], weights=weights, k=1)[0]
                board.push(move)
                if board.is_game_over(claim_draw=True):
                    valid = False
                    break
            if not valid:
                rejected_terminal += 1
                continue
            gate = engine.analyse(board, chess.engine.Limit(nodes=args.gate_nodes), game=object())
            white_score = score_cp(gate, chess.WHITE)
            if abs(white_score) > args.balance:
                rejected_balance += 1
                continue
            fen = board.fen()
            key = " ".join(fen.split()[:4])
            if key in seen:
                rejected_duplicate += 1
                continue
            seen.add(key)
            positions.append(fen)
            if len(positions) % 50 == 0:
                print(f"accepted {len(positions)}/{args.count} after {attempts} attempts", flush=True)
    finally:
        try:
            engine.quit()
        except chess.engine.EngineTerminatedError:
            pass
    if len(positions) != args.count:
        raise SystemExit(f"only generated {len(positions)}/{args.count} in {attempts} attempts")

    args.output.parent.mkdir(parents=True, exist_ok=True)
    args.output.write_text("\n".join(positions) + "\n")
    meta = {
        "schema": "deterministic Stockfish MultiPV balanced EPD",
        "output": str(args.output.resolve()),
        "output_sha256": sha256(args.output),
        "stockfish": str(args.stockfish.resolve()),
        "stockfish_sha256": sha256(args.stockfish),
        "count": args.count,
        "plies": args.plies,
        "multipv": args.multipv,
        "move_nodes": args.move_nodes,
        "gate_nodes": args.gate_nodes,
        "max_move_loss_cp": args.max_move_loss,
        "balance_cp": args.balance,
        "seed": args.seed,
        "attempts": attempts,
        "rejected_terminal": rejected_terminal,
        "rejected_balance": rejected_balance,
        "rejected_duplicate": rejected_duplicate,
    }
    args.output.with_suffix(args.output.suffix + ".meta.json").write_text(
        json.dumps(meta, indent=2, sort_keys=True) + "\n"
    )
    print(json.dumps(meta, indent=2, sort_keys=True))


if __name__ == "__main__":
    main()
