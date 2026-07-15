#!/usr/bin/env python3
"""Measure deep completed-iteration label stability for pinned UCI teachers.

This is a quality-frontier benchmark, not a bulk labeler.  One natural
iterative-deepening search is run per engine/position and its completed
iterations are sampled at requested depths.  The output is resumable and
records exact score semantics, binary hashes, state limitations, nodes, time,
best moves, bounds, and a conservative last-three-depth stability decision.
"""

from __future__ import annotations

import argparse
import hashlib
import json
import os
from pathlib import Path
import tempfile
import time
from typing import Any

import chess
import chess.engine


SCHEMA = "archon.search-label-stability.v2"
MATE_SCORE = 30_000


def sha256(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as handle:
        for block in iter(lambda: handle.read(8 * 1024 * 1024), b""):
            digest.update(block)
    return digest.hexdigest()


def canonical_hash(value: Any) -> str:
    encoded = json.dumps(value, sort_keys=True, separators=(",", ":")).encode()
    return hashlib.sha256(encoded).hexdigest()


def engine_identity(path: Path) -> dict[str, Any]:
    engine = chess.engine.SimpleEngine.popen_uci(str(path), timeout=30)
    try:
        return {
            "path": str(path.resolve()),
            "sha256": sha256(path),
            "uci_name": engine.id.get("name"),
            "uci_author": engine.id.get("author"),
            "uci_options": sorted(engine.options),
        }
    finally:
        engine.quit()


def load_fens(path: Path, limit: int | None) -> list[str]:
    result: list[str] = []
    with path.open("r", encoding="utf-8") as handle:
        for line_number, line in enumerate(handle, 1):
            if not line.strip():
                continue
            fen = line.partition("|")[0].strip()
            try:
                board = chess.Board(fen)
            except ValueError as error:
                raise ValueError(f"{path}:{line_number}: {error}") from error
            if board.is_game_over(claim_draw=False):
                raise ValueError(f"{path}:{line_number}: terminal position")
            result.append(fen)
            if limit is not None and len(result) >= limit:
                break
    if not result:
        raise ValueError("no positions")
    return result


def encode_info(info: dict[str, Any], board: chess.Board) -> dict[str, Any] | None:
    score = info.get("score")
    if score is None:
        return None
    pov = score.pov(board.turn)
    cp = pov.score(mate_score=MATE_SCORE)
    if cp is None:
        return None
    pv = info.get("pv", [])
    move = pv[0] if pv else None
    if move is not None and move not in board.legal_moves:
        raise RuntimeError(f"illegal PV move {move} for {board.fen()}")
    record: dict[str, Any] = {
        "depth": int(info.get("depth", 0)),
        "seldepth": int(info.get("seldepth", 0)),
        "score_cp_with_mate_30000": int(cp),
        "mate_plies": pov.mate(),
        "best_move_uci": None if move is None else move.uci(),
        "bound": (
            "lower" if info.get("lowerbound")
            else "upper" if info.get("upperbound")
            else "exact"
        ),
        "nodes": int(info.get("nodes", 0)),
        "time_seconds": float(info.get("time", 0.0)),
    }
    if "nps" in info:
        record["nps"] = int(info["nps"])
    if "hashfull" in info:
        record["hashfull_per_mille"] = int(info["hashfull"])
    if "wdl" in info:
        wdl = info["wdl"].pov(board.turn)
        record["wdl_stm"] = [int(wdl.wins), int(wdl.draws), int(wdl.losses)]
    return record


def stability(
    iterations: list[dict[str, Any]], requested_depths: list[int], window: int, tolerance: int,
) -> dict[str, Any]:
    tail = iterations[-window:]
    result: dict[str, Any] = {
        "required_iterations": window,
        "available_iterations": len(iterations),
        "score_tolerance_cp": tolerance,
        "depths": [row["depth"] for row in tail],
        "stable": False,
        "value_stable": False,
        "move_stable": False,
        "policy_stable": False,
        "joint_stable": False,
    }
    if len(tail) < window or [row["depth"] for row in tail] != requested_depths[-window:]:
        result["reason"] = "missing_requested_depth"
        return result
    if any(row["bound"] != "exact" for row in tail):
        result["reason"] = "bounded_iteration"
        return result
    moves = {row["best_move_uci"] for row in tail}
    result["move_stable"] = len(moves) == 1 and None not in moves
    result["move_reason"] = "stable" if result["move_stable"] else "best_move_changed"
    mates = [row["mate_plies"] for row in tail]
    if any(mate is not None for mate in mates):
        if any(mate is None for mate in mates) or len({mate > 0 for mate in mates}) != 1:
            result["reason"] = "mate_status_changed"
            return result
        # Mate distance is allowed to tighten as depth rises; sign and move
        # stability are the meaningful oracle condition.
        result.update({"value_stable": True, "value_reason": "stable_mate", "score_span_cp": None})
    else:
        scores = [row["score_cp_with_mate_30000"] for row in tail]
        span = max(scores) - min(scores)
        result["score_span_cp"] = span
        result["value_stable"] = span <= tolerance
        result["value_reason"] = "stable" if result["value_stable"] else "score_drift"
    result["policy_stable"] = result["value_stable"] and result["move_stable"]
    result["joint_stable"] = result["policy_stable"]
    result["stable"] = result["joint_stable"]  # Backward-readable strict gate.
    result["reason"] = (
        result["value_reason"] if not result["value_stable"]
        else result["move_reason"] if not result["move_stable"]
        else "stable_mate" if result["value_reason"] == "stable_mate"
        else "stable"
    )
    return result


def search_position(
    engine: chess.engine.SimpleEngine,
    board: chess.Board,
    depths: list[int],
    stable_window: int,
    tolerance: int,
    max_nodes: int,
) -> dict[str, Any]:
    requested = set(depths)
    captured: dict[int, dict[str, Any]] = {}
    began = time.monotonic()
    # A unique game token forces ucinewgame and a cold TT for every position.
    with engine.analysis(
        board,
        # Use a single UCI limit.  Some engines prioritize `depth` and ignore a
        # simultaneous `nodes` token, which defeats the hard cost ceiling.
        chess.engine.Limit(nodes=max_nodes),
        game=object(),
        info=chess.engine.INFO_ALL,
    ) as analysis:
        for info in analysis:
            depth = int(info.get("depth", 0))
            if depth in requested:
                encoded = encode_info(info, board)
                if encoded is not None:
                    captured[depth] = encoded
                    # Aspiration retries can emit lower/upper records at the
                    # target depth before the completed exact iteration.
                    if depth == max(depths) and encoded["bound"] == "exact":
                        analysis.stop()
                        break
    rows = [captured[depth] for depth in depths if depth in captured]
    return {
        "fen": board.fen(),
        "iterations": rows,
        "wall_seconds": time.monotonic() - began,
        "reached_max_depth": bool(rows and rows[-1]["depth"] == max(depths)),
        "stability": stability(rows, depths, stable_window, tolerance),
    }


def atomic_json(path: Path, value: dict[str, Any]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    with tempfile.NamedTemporaryFile("w", dir=path.parent, prefix=f".{path.name}.", delete=False) as f:
        staged = Path(f.name)
        json.dump(value, f, indent=2, sort_keys=True)
        f.write("\n")
        f.flush()
        os.fsync(f.fileno())
    os.replace(staged, path)


def summarize(results: dict[str, list[dict[str, Any]]]) -> dict[str, Any]:
    summary: dict[str, Any] = {}
    for teacher, rows in results.items():
        stable = sum(bool(row["stability"]["joint_stable"]) for row in rows)
        value_stable = sum(bool(row["stability"]["value_stable"]) for row in rows)
        policy_stable = sum(bool(row["stability"]["policy_stable"]) for row in rows)
        final = [row["iterations"][-1] for row in rows if row["iterations"]]
        summary[teacher] = {
            "positions": len(rows),
            "stable_positions": stable,
            "unstable_positions": len(rows) - stable,
            "stability_rate": stable / len(rows) if rows else None,
            "gold_value_positions": value_stable,
            "gold_value_rate": value_stable / len(rows) if rows else None,
            "gold_policy_positions": policy_stable,
            "gold_policy_rate": policy_stable / len(rows) if rows else None,
            "final_exact_positions": sum(row["bound"] == "exact" for row in final),
            "reached_max_depth_positions": sum(row["reached_max_depth"] for row in rows),
            "total_final_nodes": sum(row["nodes"] for row in final),
            "total_wall_seconds": sum(row["wall_seconds"] for row in rows),
            "failure_reasons": {
                reason: sum(row["stability"]["reason"] == reason for row in rows)
                for reason in sorted({row["stability"]["reason"] for row in rows})
            },
        }
    return summary


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("fens", type=Path)
    parser.add_argument("--output", type=Path, required=True)
    parser.add_argument("--reckless", type=Path,
                        help="optional sparse audit teacher; Stockfish 18 is the production default")
    parser.add_argument("--stockfish", type=Path, required=True)
    parser.add_argument("--depths", type=int, nargs="+", default=[16, 18, 20, 22])
    parser.add_argument("--stable-window", type=int, default=3)
    parser.add_argument("--score-tolerance-cp", type=int, default=25)
    parser.add_argument("--hash-mb", type=int, default=64)
    parser.add_argument("--max-nodes-per-position", type=int, required=True)
    parser.add_argument("--max-positions", type=int)
    parser.add_argument("--position-indices", type=int, nargs="+",
                        help="zero-based source positions to benchmark (for escalation tiers)")
    parser.add_argument("--escalate-from", type=Path,
                        help="prior stability manifest whose failed gate selects source positions")
    parser.add_argument("--escalate-gate", choices=("value", "policy", "joint"), default="value")
    parser.add_argument("--selection-manifest", type=Path,
                        help="JSON manifest containing source_position_indices")
    args = parser.parse_args()
    depths = sorted(set(args.depths))
    if not depths or min(depths) < 1 or depths != args.depths:
        parser.error("depths must be positive, unique, and increasing")
    if not 2 <= args.stable_window <= len(depths):
        parser.error("stable-window must be between 2 and the number of depths")
    if args.score_tolerance_cp < 0 or args.hash_mb <= 0 or args.max_nodes_per_position <= 0:
        parser.error("invalid tolerance, hash size, or node ceiling")
    if args.max_positions is not None and args.max_positions <= 0:
        parser.error("max-positions must be positive")
    selectors = sum(value is not None for value in (
        args.max_positions, args.position_indices, args.escalate_from, args.selection_manifest,
    ))
    if selectors > 1:
        parser.error("max-positions, position-indices, and escalate-from are mutually exclusive")
    for path in (args.fens, args.stockfish, *([args.reckless] if args.reckless else [])):
        if not path.is_file():
            parser.error(f"missing input: {path}")

    all_fens = load_fens(
        args.fens,
        args.max_positions if selectors == 1 and args.max_positions is not None else None,
    )
    escalation: dict[str, Any] | None = None
    selection: dict[str, Any] | None = None
    if args.selection_manifest is not None:
        if not args.selection_manifest.is_file():
            parser.error(f"missing selection manifest: {args.selection_manifest}")
        selected = json.loads(args.selection_manifest.read_text(encoding="utf-8"))
        if selected.get("input", {}).get("sha256") != sha256(args.fens):
            raise RuntimeError("selection manifest has a different FEN input")
        source_indices = [int(index) for index in selected["source_position_indices"]]
        if len(set(source_indices)) != len(source_indices):
            raise RuntimeError("selection manifest has duplicate positions")
        fens = [all_fens[index] for index in source_indices]
        selection = {"path": str(args.selection_manifest.resolve()),
                     "sha256": sha256(args.selection_manifest), "positions": len(source_indices)}
    elif args.escalate_from is not None:
        if not args.escalate_from.is_file():
            parser.error(f"missing escalation manifest: {args.escalate_from}")
        previous = json.loads(args.escalate_from.read_text(encoding="utf-8"))
        if previous.get("input", {}).get("sha256") != sha256(args.fens):
            raise RuntimeError("escalation manifest has a different FEN input")
        gate_field = {
            "value": "value_stable", "policy": "policy_stable", "joint": "joint_stable",
        }[args.escalate_gate]
        previous_rows = previous.get("results", {}).get("stockfish18", [])
        if not previous_rows:
            raise RuntimeError("escalation manifest has no Stockfish results")
        source_indices = [
            int(row["position_index"])
            for row in previous_rows if not bool(row["stability"].get(gate_field, False))
        ]
        escalation = {
            "path": str(args.escalate_from.resolve()),
            "sha256": sha256(args.escalate_from),
            "gate": args.escalate_gate,
            "gate_field": gate_field,
            "prior_positions": len(previous_rows),
            "selected_failures": len(source_indices),
        }
        fens = [all_fens[index] for index in source_indices]
    elif args.position_indices is None:
        source_indices = list(range(len(all_fens)))
        fens = all_fens
    else:
        if len(set(args.position_indices)) != len(args.position_indices):
            parser.error("position-indices must be unique")
        if any(index < 0 or index >= len(all_fens) for index in args.position_indices):
            parser.error("position-index is outside the input")
        source_indices = args.position_indices
        fens = [all_fens[index] for index in source_indices]
    identities = {"stockfish18": engine_identity(args.stockfish)}
    if args.reckless:
        identities["reckless"] = engine_identity(args.reckless)
    config = {
        "schema": SCHEMA,
        "input": {"path": str(args.fens.resolve()), "sha256": sha256(args.fens)},
        "position_count": len(fens),
        "source_position_indices": source_indices,
        "escalation": escalation,
        "selection": selection,
        "position_fen_hash": canonical_hash(fens),
        "state_completeness": "full FEN except repetition history",
        "score_perspective": "side_to_move",
        "score_semantics": "UCI search score; mate mapped only for numeric diagnostics",
        "search_mode": "one cold node-capped iterative-deepening search per engine/position; client stops after observing target completed depth",
        "depths": depths,
        "stable_window": args.stable_window,
        "score_tolerance_cp": args.score_tolerance_cp,
        "hash_mb": args.hash_mb,
        "max_nodes_per_position": args.max_nodes_per_position,
        "threads": 1,
        "teachers": identities,
    }
    config_hash = canonical_hash(config)
    if args.output.exists():
        output = json.loads(args.output.read_text(encoding="utf-8"))
        if output.get("config_hash") != config_hash:
            raise RuntimeError("existing output has a different configuration")
    else:
        output = {**config, "config_hash": config_hash, "results": {name: [] for name in identities}}

    paths = {"stockfish18": args.stockfish}
    if args.reckless:
        paths["reckless"] = args.reckless
    for teacher, path in paths.items():
        completed = output["results"][teacher]
        engine = chess.engine.SimpleEngine.popen_uci(str(path), timeout=30)
        try:
            settings: dict[str, Any] = {}
            if "Threads" in engine.options:
                settings["Threads"] = 1
            if "Hash" in engine.options:
                settings["Hash"] = args.hash_mb
            if "UCI_ShowWDL" in engine.options:
                settings["UCI_ShowWDL"] = True
            if settings:
                engine.configure(settings)
            for index in range(len(completed), len(fens)):
                result = search_position(
                    engine, chess.Board(fens[index]), depths,
                    args.stable_window, args.score_tolerance_cp,
                    args.max_nodes_per_position,
                )
                result["position_index"] = source_indices[index]
                completed.append(result)
                output["summary"] = summarize(output["results"])
                atomic_json(args.output, output)
                print(json.dumps({
                    "teacher": teacher,
                    "position": index,
                    "stability": result["stability"],
                    "final": result["iterations"][-1] if result["iterations"] else None,
                    "wall_seconds": result["wall_seconds"],
                }, sort_keys=True), flush=True)
        finally:
            engine.quit()
    output["summary"] = summarize(output["results"])
    atomic_json(args.output, output)
    print(json.dumps({"output": str(args.output), "summary": output["summary"]}, indent=2))


if __name__ == "__main__":
    main()
