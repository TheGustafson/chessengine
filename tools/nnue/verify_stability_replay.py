#!/usr/bin/env python3
"""Require deterministic semantic equality between stability audit and replay."""

from __future__ import annotations

import argparse
import hashlib
import json
from pathlib import Path


SEMANTIC_ITERATION_FIELDS = (
    "depth", "seldepth", "score_cp_with_mate_30000", "mate_plies",
    "best_move_uci", "bound", "nodes", "hashfull_per_mille", "wdl_stm",
)


def sha256(path: Path) -> str:
    return hashlib.sha256(path.read_bytes()).hexdigest()


def semantic_iteration(row: dict) -> dict:
    return {field: row.get(field) for field in SEMANTIC_ITERATION_FIELDS}


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("original", type=Path)
    parser.add_argument("replay", type=Path)
    parser.add_argument("--output", type=Path, required=True)
    args = parser.parse_args()
    for path in (args.original, args.replay):
        if not path.is_file():
            parser.error(f"missing input: {path}")
    original = json.loads(args.original.read_text(encoding="utf-8"))
    replay = json.loads(args.replay.read_text(encoding="utf-8"))
    for field in ("schema", "input", "depths", "stable_window", "score_tolerance_cp",
                  "hash_mb", "max_nodes_per_position"):
        if original[field] != replay[field]:
            raise RuntimeError(f"configuration mismatch: {field}")
    if original["teachers"]["stockfish18"]["sha256"] != replay["teachers"]["stockfish18"]["sha256"]:
        raise RuntimeError("teacher binary mismatch")

    reference = {
        int(row["position_index"]): row for row in original["results"]["stockfish18"]
    }
    checks: list[dict] = []
    for row in replay["results"]["stockfish18"]:
        index = int(row["position_index"])
        expected = reference.get(index)
        if expected is None:
            raise RuntimeError(f"replay position {index} absent from original")
        if row["fen"] != expected["fen"]:
            raise RuntimeError(f"FEN mismatch at {index}")
        actual_iterations = [semantic_iteration(item) for item in row["iterations"]]
        expected_iterations = [semantic_iteration(item) for item in expected["iterations"]]
        if actual_iterations != expected_iterations:
            raise RuntimeError(f"semantic iteration mismatch at {index}")
        if row["stability"] != expected["stability"]:
            raise RuntimeError(f"stability decision mismatch at {index}")
        checks.append({
            "source_position_index": index,
            "semantic_iterations_equal": True,
            "stability_equal": True,
            "original_wall_seconds": expected["wall_seconds"],
            "replay_wall_seconds": row["wall_seconds"],
        })
    result = {
        "schema": "archon.search-label-stability-replay.v1",
        "original": {"path": str(args.original.resolve()), "sha256": sha256(args.original)},
        "replay": {"path": str(args.replay.resolve()), "sha256": sha256(args.replay)},
        "teacher_sha256": replay["teachers"]["stockfish18"]["sha256"],
        "excluded_nondeterministic_fields": ["time_seconds", "nps", "wall_seconds"],
        "positions_checked": len(checks),
        "semantic_mismatches": 0,
        "checks": checks,
    }
    args.output.parent.mkdir(parents=True, exist_ok=True)
    args.output.write_text(json.dumps(result, indent=2, sort_keys=True) + "\n", encoding="utf-8")
    print(json.dumps({"positions_checked": len(checks), "semantic_mismatches": 0,
                      "output": str(args.output)}, indent=2))


if __name__ == "__main__":
    main()
