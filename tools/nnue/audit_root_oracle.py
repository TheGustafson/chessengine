#!/usr/bin/env python3
"""Audit one or more ARCHON root-move oracle datasets."""

from __future__ import annotations

import argparse
from collections import Counter
import json
from pathlib import Path

import numpy as np


def load(root: Path) -> tuple[dict, list[dict]]:
    manifest = json.loads((root / "manifest.json").read_text())
    shards = sorted((root / "shards").glob("*.json"))
    rows: list[dict] = []
    for path in shards:
        shard = json.loads(path.read_text())
        if shard["config_hash"] != manifest["config_hash"]:
            raise RuntimeError(f"config mismatch: {path}")
        rows.extend(shard["records"])
    ids = [row["source_id"] for row in rows]
    if len(ids) != len(set(ids)) or len(rows) != manifest["total_positions"]:
        raise RuntimeError(f"coverage mismatch: {root}")
    return manifest, rows


def percentile(values: list[float]) -> dict[str, float | None]:
    if not values:
        return {key: None for key in ("min", "p25", "median", "p75", "max")}
    out = np.percentile(values, [0, 25, 50, 75, 100])
    return {key: float(value) for key, value in zip(("min", "p25", "median", "p75", "max"), out)}


def audit(root: Path, manifest: dict, rows: list[dict]) -> dict:
    arms = [arm for row in rows for arm in row["arms"]]
    low = np.asarray([arm["low"]["score_cp"] for arm in arms], dtype=np.float64)
    high = np.asarray([arm["high"]["score_cp"] for arm in arms], dtype=np.float64)
    ordinary = (np.abs(low) < 25_000) & (np.abs(high) < 25_000)
    mixed_rows = 0
    modes = Counter()
    for row in rows:
        row_modes = {
            arm[budget]["execution_mode"]
            for arm in row["arms"]
            for budget in ("low", "high")
        }
        mixed_rows += len(row_modes) > 1
        modes.update(row_modes)
    if mixed_rows:
        raise RuntimeError(f"{root}: {mixed_rows} rows mix arm execution semantics")
    margins = [
        row["summary"]["high_best_second_margin_cp"]
        for row in rows
        if row["summary"]["high_best_second_margin_cp"] is not None
    ]
    return {
        "root": str(root.resolve()),
        "config_hash": manifest["config_hash"],
        "positions": len(rows),
        "arms": len(arms),
        "acceptance": dict(Counter(row["acceptance"]["status"] for row in rows)),
        "accepted_fraction": float(np.mean([row["acceptance"]["accepted"] for row in rows])),
        "best_stable_fraction": float(np.mean([row["summary"]["best_stable"] for row in rows])),
        "high_unique_fraction": float(np.mean([len(row["summary"]["high_tie_moves"]) == 1 for row in rows])),
        "legal_moves": percentile([len(row["arms"]) for row in rows]),
        "high_best_second_margin_cp": percentile(margins),
        "execution_modes_by_position": dict(modes),
        "ordinary_low_high": {
            "rows": int(ordinary.sum()),
            "pearson": float(np.corrcoef(low[ordinary], high[ordinary])[0, 1]),
            "mae_cp": float(np.mean(np.abs(low[ordinary] - high[ordinary]))),
        },
        "nodes": {
            "low_total": sum(arm["low"]["nodes"] for arm in arms),
            "high_total": sum(arm["high"]["nodes"] for arm in arms),
            "low_per_arm": percentile([arm["low"]["nodes"] for arm in arms]),
            "high_per_arm": percentile([arm["high"]["nodes"] for arm in arms]),
        },
        "search_time_ms": {
            "low_total": sum(arm["low"].get("time_ms", 0) for arm in arms),
            "high_total": sum(arm["high"].get("time_ms", 0) for arm in arms),
            "low_per_arm": percentile([arm["low"].get("time_ms", 0) for arm in arms]),
            "high_per_arm": percentile([arm["high"].get("time_ms", 0) for arm in arms]),
        },
    }


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("datasets", nargs="+", type=Path)
    parser.add_argument("--output", type=Path)
    args = parser.parse_args()
    loaded = [(root, *load(root)) for root in args.datasets]
    report: dict[str, object] = {"schema": "archon.root-oracle-audit.v1", "datasets": []}
    report["datasets"] = [audit(root, manifest, rows) for root, manifest, rows in loaded]
    if len(loaded) > 1:
        baseline = {row["source_id"]: row for row in loaded[0][2]}
        comparisons = []
        for root, _manifest, rows in loaded[1:]:
            candidate = {row["source_id"]: row for row in rows}
            common = sorted(set(baseline) & set(candidate))
            agreement = sum(
                baseline[key]["summary"]["best_move_high"]
                == candidate[key]["summary"]["best_move_high"]
                for key in common
            )
            comparisons.append({"baseline": str(loaded[0][0]), "candidate": str(root),
                                "common_positions": len(common),
                                "high_best_agreement_fraction": agreement / len(common) if common else None})
        report["comparisons"] = comparisons
    rendered = json.dumps(report, indent=2, sort_keys=True) + "\n"
    if args.output:
        args.output.parent.mkdir(parents=True, exist_ok=True)
        args.output.write_text(rendered)
    print(rendered, end="")


if __name__ == "__main__":
    main()
