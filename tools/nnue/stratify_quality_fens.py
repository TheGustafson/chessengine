#!/usr/bin/env python3
"""Stratify a hash-ranked FEN candidate pool for SF18 label-quality audits."""

from __future__ import annotations

import argparse
import hashlib
import json
from pathlib import Path
from typing import Any

import chess

from relabel_fen_static_paired import RawEvalEngine, engine_identity, sha256


SCHEMA = "archon.sf18-quality-audit-fens.v1"


def rank(seed: int, fen: str) -> int:
    return int.from_bytes(
        hashlib.blake2b(seed.to_bytes(8, "little") + fen.encode(), digest_size=8).digest(),
        "little",
    )


def phase(board: chess.Board) -> str:
    non_pawn_non_king = sum(
        len(board.pieces(piece, colour))
        for colour in chess.COLORS
        for piece in (chess.KNIGHT, chess.BISHOP, chess.ROOK, chess.QUEEN)
    )
    pieces = len(board.piece_map())
    if board.fullmove_number <= 15 and pieces >= 24:
        return "opening"
    if non_pawn_non_king <= 4 or pieces <= 10:
        return "endgame"
    return "middlegame"


def score_band(cp: int) -> str:
    absolute = abs(cp)
    if absolute < 80:
        return "balanced"
    if absolute < 250:
        return "advantage"
    return "decisive"


def flags(board: chess.Board) -> list[str]:
    result: list[str] = []
    if board.is_check():
        result.append("in_check")
    if board.castling_rights:
        result.append("castling_rights")
    if board.ep_square is not None:
        result.append("en_passant")
    if board.halfmove_clock >= 50:
        result.append("rule50_ge_50")
    if board.fullmove_number >= 60:
        result.append("late_game")
    return result


def counts(rows: list[dict[str, Any]]) -> dict[str, Any]:
    return {
        "phase": {name: sum(row["phase"] == name for row in rows)
                  for name in ("opening", "middlegame", "endgame")},
        "score_band": {name: sum(row["score_band"] == name for row in rows)
                       for name in ("balanced", "advantage", "decisive")},
        "primary_stratum": {
            name: sum(row["stratum"] == name for row in rows)
            for name in sorted({row["stratum"] for row in rows})
        },
        "flags": {
            name: sum(name in row["flags"] for row in rows)
            for name in ("in_check", "castling_rights", "en_passant", "rule50_ge_50", "late_game")
        },
    }


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("input", type=Path)
    parser.add_argument("--output", type=Path, required=True)
    parser.add_argument("--stockfish", type=Path, required=True)
    parser.add_argument("--count", type=int, default=256)
    parser.add_argument("--seed", type=int, default=0xD40B3)
    parser.add_argument("--special-per-flag", type=int, default=8)
    args = parser.parse_args()
    if args.count <= 0 or args.special_per_flag < 0:
        parser.error("invalid count")
    for path in (args.input, args.stockfish):
        if not path.is_file():
            parser.error(f"missing input: {path}")

    fens = [line.strip() for line in args.input.read_text(encoding="utf-8").splitlines() if line.strip()]
    if len(set(fens)) != len(fens):
        raise RuntimeError("candidate pool contains duplicate FEN strings")
    engine = RawEvalEngine(str(args.stockfish.resolve()))
    rows: list[dict[str, Any]] = []
    try:
        for index, fen in enumerate(fens):
            board = chess.Board(fen)
            if board.is_game_over(claim_draw=False):
                raise RuntimeError(f"terminal candidate {index}")
            value = engine.evaluate(fen)
            # Stockfish deliberately has no static value in check.  Keep such
            # positions as a special search-audit stratum and use an unknown
            # static score band rather than silently dropping them.
            cp = None if value is None else value["final_uci_cp"]
            band = "unknown" if cp is None else score_band(cp)
            item = {
                "candidate_index": index,
                "fen": fen,
                "rank": rank(args.seed, fen),
                "phase": phase(board),
                "score_band": band,
                "static_final_uci_cp": cp,
                "flags": flags(board),
            }
            item["stratum"] = f"{item['phase']}:{band}"
            rows.append(item)
    finally:
        engine.close()

    selected: dict[str, dict[str, Any]] = {}
    special_names = ("in_check", "castling_rights", "en_passant", "rule50_ge_50", "late_game")
    for flag in special_names:
        candidates = sorted((row for row in rows if flag in row["flags"]), key=lambda row: row["rank"])
        for row in candidates[:args.special_per_flag]:
            selected[row["fen"]] = row

    strata = sorted({row["stratum"] for row in rows})
    quota = max(1, args.count // len(strata))
    for stratum in strata:
        candidates = sorted((row for row in rows if row["stratum"] == stratum), key=lambda row: row["rank"])
        already = sum(row["stratum"] == stratum for row in selected.values())
        for row in candidates[:max(0, quota - already)]:
            selected[row["fen"]] = row

    for row in sorted(rows, key=lambda row: row["rank"]):
        if len(selected) >= args.count:
            break
        selected[row["fen"]] = row
    if len(selected) < args.count:
        raise RuntimeError(f"only selected {len(selected)} of {args.count}")

    final = sorted(selected.values(), key=lambda row: row["rank"])[:args.count]
    args.output.parent.mkdir(parents=True, exist_ok=True)
    args.output.write_text("\n".join(row["fen"] for row in final) + "\n", encoding="utf-8")
    manifest = {
        "schema": SCHEMA,
        "input": {"path": str(args.input.resolve()), "sha256": sha256(args.input), "rows": len(rows)},
        "output": {"path": str(args.output.resolve()), "sha256": sha256(args.output), "rows": len(final)},
        "seed": args.seed,
        "special_per_flag": args.special_per_flag,
        "stockfish": engine_identity(args.stockfish),
        "static_field_for_stratification": "final_uci_cp only; never used as the deep oracle",
        "selection": "reserve rare-state flags, balance observed phase:score strata, fill by deterministic rank",
        "candidate_composition": counts(rows),
        "selected_composition": counts(final),
        "selected_rows": final,
    }
    args.output.with_suffix(args.output.suffix + ".json").write_text(
        json.dumps(manifest, indent=2, sort_keys=True) + "\n", encoding="utf-8"
    )
    print(json.dumps({"output": manifest["output"], "composition": manifest["selected_composition"]}, indent=2))


if __name__ == "__main__":
    main()
