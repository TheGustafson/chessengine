#!/usr/bin/env python3
"""Cold-replay selected Archon LMR observations at alternative reductions.

This bootstrap deliberately does not claim exact in-tree counterfactuals: each
move/action is replayed from the stored full FEN with a cold engine, TT, and
histories. It does preserve the observed alpha/beta context for labeling which
side of the original bound each action returns. Repetition history is absent.
"""

from __future__ import annotations

import argparse
from concurrent.futures import ProcessPoolExecutor, as_completed
from dataclasses import asdict
import hashlib
import heapq
import json
import os
from pathlib import Path
import shutil
from typing import Any

import chess

from root_move_oracle import OracleConfig, analyse_cold_arm, sha256


TRACE_SCHEMA = "archon.search-lmr-observation.v3"
SCHEMA = "archon.search-counterfactual-cold-v1"


def canonical_hash(value: object) -> str:
    return hashlib.sha256(
        json.dumps(value, sort_keys=True, separators=(",", ":")).encode()
    ).hexdigest()


def move_uci(raw: int) -> str:
    source = chess.square_name(raw & 63)
    destination = chess.square_name((raw >> 6) & 63)
    kind = raw >> 14
    promotion = ""
    if kind == 1:
        promotion = "nbrq"[(raw >> 12) & 3]
    return source + destination + promotion


def importance(row: dict, line_number: int) -> tuple[int, int, int]:
    if row["lmr_research"]:
        category = 0
    elif row["alpha_raise"] or row["beta_cutoff"]:
        category = 1
    elif abs(row["scout_score"] - row["alpha"]) <= 50:
        category = 2
    else:
        category = 3
    tie = int.from_bytes(
        hashlib.blake2b(
            f"{line_number}:{row['position_key']}:{row['raw_move']}".encode(), digest_size=8
        ).digest(),
        "little",
    )
    return category, tie, line_number


def select(trace: Path, count: int, mode: str) -> list[dict]:
    # Keep the lexicographically smallest active-mining priorities in bounded
    # memory. Exact replay duplicates are removed before they consume budget.
    heaps: list[list[tuple[tuple[int, int, int], dict]]] = [[], [], [], []]
    seen: set[tuple[Any, ...]] = set()
    current_root = None
    with trace.open(encoding="utf-8") as source:
        header = json.loads(next(source))
        if header.get("schema") != TRACE_SCHEMA or header.get("kind") != "header":
            raise ValueError("counterfactual replay requires v3 full-FEN telemetry")
        for line_number, line in enumerate(source, 2):
            row = json.loads(line)
            if row.get("kind") == "go":
                current_root = int(row["root_position_key"])
                continue
            if row.get("kind") != "row":
                continue
            nominal = int(row["nominal_child_depth"])
            if nominal < 2 or int(row["scout_depth"]) >= nominal:
                continue
            key = (
                row["fen"], row["raw_move"], nominal, row["alpha"], row["beta"],
                row["static_eval"], row["move_index"],
            )
            if key in seen:
                continue
            seen.add(key)
            item = {
                "observation_line": line_number,
                "root_position_key": current_root,
                "observation": row,
            }
            rank = importance(row, line_number)
            inverted = tuple(-part for part in rank)
            heap = heaps[rank[0]]
            if len(heap) < count:
                heapq.heappush(heap, (inverted, item))
            elif inverted > heap[0][0]:
                heapq.heapreplace(heap, (inverted, item))
    candidates = [
        sorted((item for _, item in heap), key=lambda item: importance(item["observation"], item["observation_line"]))
        for heap in heaps
    ]
    if mode == "active":
        return sorted(
            (item for category in candidates for item in category),
            key=lambda item: importance(item["observation"], item["observation_line"]),
        )[:count]
    quotas = [count // 2, count // 5, count * 15 // 100]
    quotas.append(count - sum(quotas))
    selected = [item for category, quota in zip(candidates, quotas) for item in category[:quota]]
    selected_lines = {item["observation_line"] for item in selected}
    if len(selected) < count:
        remainder = sorted(
            (
                item
                for category in candidates
                for item in category
                if item["observation_line"] not in selected_lines
            ),
            key=lambda item: importance(item["observation"], item["observation_line"]),
        )
        selected.extend(remainder[: count - len(selected)])
    return sorted(selected, key=lambda item: item["observation_line"])


def relation(score: int, alpha: int, beta: int) -> str:
    if score <= alpha:
        return "upper"
    if score >= beta:
        return "lower"
    return "interior"


def replay(item: dict, config: OracleConfig, max_reduction: int, config_hash: str) -> dict:
    row = item["observation"]
    board = chess.Board(row["fen"])
    uci = move_uci(int(row["raw_move"]))
    move = next((candidate for candidate in board.legal_moves if candidate.uci() == uci), None)
    if move is None:
        raise RuntimeError(f"stored move {uci} is not legal in {row['fen']}")
    nominal = int(row["nominal_child_depth"])
    reductions = sorted({r for r in range(max_reduction + 1) if nominal - r >= 1})
    actions = []
    for reduction in reductions:
        child_depth = nominal - reduction
        result = analyse_cold_arm(config, board, move, child_depth + 1)
        actions.append(
            {
                "reduction": reduction,
                "child_depth": child_depth,
                "root_depth": child_depth + 1,
                "score_cp": result["score_cp"],
                "bound": result["bound"],
                "nodes": result["nodes"],
                "time_ms": result["time_ms"],
                "seldepth": result["seldepth"],
                "pv": result["pv"],
            }
        )
    reference = next(action for action in actions if action["reduction"] == 0)
    reference_relation = relation(reference["score_cp"], row["alpha"], row["beta"])
    for action in actions:
        action_relation = relation(action["score_cp"], row["alpha"], row["beta"])
        action["window_relation"] = action_relation
        action["wrong_window_relation"] = action_relation != reference_relation
        action["score_error_cp"] = action["score_cp"] - reference["score_cp"]
        action["absolute_score_error_cp"] = abs(action["score_error_cp"])
        action["node_fraction_of_reference"] = action["nodes"] / max(1, reference["nodes"])
    return {
        "schema": SCHEMA,
        "config_hash": config_hash,
        "observation_line": item["observation_line"],
        "root_position_key": item["root_position_key"],
        "observation": row,
        "replay_state": {
            "full_fen": row["fen"],
            "repetition_history": "unavailable",
            "tt_and_histories": "cold_per_action",
            "move": uci,
        },
        "reference": {"reduction": 0, "window_relation": reference_relation},
        "actions": actions,
    }


def atomic_json(path: Path, value: object) -> None:
    temporary = path.with_suffix(path.suffix + ".tmp")
    temporary.write_text(json.dumps(value, indent=2, sort_keys=True) + "\n", encoding="utf-8")
    temporary.replace(path)


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--trace", type=Path, required=True)
    parser.add_argument("--output", type=Path, required=True)
    parser.add_argument("--engine", type=Path, required=True)
    parser.add_argument("--eval", type=Path, required=True)
    parser.add_argument("--count", type=int, default=128)
    parser.add_argument("--selection-mode", choices=("stratified", "active"), default="stratified")
    parser.add_argument("--max-reduction", type=int, default=3)
    parser.add_argument("--hash-mb", type=int, default=16)
    parser.add_argument("--workers", type=int, default=2)
    parser.add_argument("--timeout-seconds", type=float, default=120.0)
    args = parser.parse_args()
    if min(args.count, args.max_reduction, args.hash_mb, args.workers) <= 0:
        parser.error("count, max-reduction, hash-mb, and workers must be positive")
    trace = args.trace.resolve()
    engine = args.engine.resolve()
    eval_file = args.eval.resolve()
    for label, path in (("trace", trace), ("engine", engine), ("eval", eval_file)):
        if not path.is_file():
            parser.error(f"missing {label}: {path}")
    if not os.access(engine, os.X_OK):
        parser.error(f"engine is not executable: {engine}")

    selected = select(trace, args.count, args.selection_mode)
    if not selected:
        raise RuntimeError("no eligible reduced LMR observations")
    generator = Path(__file__).resolve()
    identities = {
        "generator": {"path": str(generator), "sha256": sha256(generator)},
        "trace": {"path": str(trace), "sha256": sha256(trace)},
        "engine": {"path": str(engine), "sha256": sha256(engine)},
        "eval": {"path": str(eval_file), "sha256": sha256(eval_file)},
    }
    config_payload = {
        "schema": SCHEMA,
        "identities": identities,
        "selection": {
            "mode": args.selection_mode,
            "categories": "actual re-search, alpha/cutoff, within-50cp, deterministic uniform",
            "stratified_quotas": "50%, 20%, 15%, 15% with deterministic deficit fill",
        },
        "selected_observation_lines": [item["observation_line"] for item in selected],
        "max_reduction": args.max_reduction,
        "search": {"threads": 1, "hash_mb": args.hash_mb, "cold_per_action": True},
        "state_limit": "FEN has no repetition history; TT and histories cold",
    }
    config_hash = canonical_hash(config_payload)
    output = args.output.resolve()
    records_dir = output / "records"
    manifest_path = output / "manifest.json"
    if output.exists():
        manifest = json.loads(manifest_path.read_text(encoding="utf-8"))
        if manifest.get("config_hash") != config_hash:
            parser.error("existing output has a different configuration")
    else:
        records_dir.mkdir(parents=True)
        shutil.copy2(generator, output / "generator.snapshot.py")
        atomic_json(manifest_path, {**config_payload, "config_hash": config_hash})

    oracle = OracleConfig(
        engine=str(engine), teacher="archon-cold-counterfactual",
        teacher_sha256=identities["engine"]["sha256"], limit_type="depth",
        low=1, high=2, hash_mb=args.hash_mb, timeout_seconds=args.timeout_seconds,
        tie_cp=0, arm_mode="searchmoves", engine_options=(("EvalFile", str(eval_file)),),
        option_artifacts=(("EvalFile", str(eval_file), identities["eval"]["sha256"]),),
    )
    pending = []
    for ordinal, item in enumerate(selected):
        path = records_dir / f"{ordinal:06d}.json"
        if path.exists():
            record = json.loads(path.read_text(encoding="utf-8"))
            if record.get("config_hash") != config_hash or record.get("observation_line") != item["observation_line"]:
                raise RuntimeError(f"invalid completed record: {path}")
        else:
            pending.append((ordinal, item, path))
    with ProcessPoolExecutor(max_workers=args.workers) as pool:
        futures = {
            pool.submit(replay, item, oracle, args.max_reduction, config_hash): (ordinal, path)
            for ordinal, item, path in pending
        }
        for future in as_completed(futures):
            ordinal, path = futures[future]
            atomic_json(path, future.result())
            print(f"wrote {ordinal + 1}/{len(selected)} {path}")
    if len(list(records_dir.glob("*.json"))) != len(selected):
        raise RuntimeError("record coverage incomplete")
    print(f"complete: {len(selected)} observations in {output}")


if __name__ == "__main__":
    main()
