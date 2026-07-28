#!/usr/bin/env python3
"""Merge ordered SF18 stability tiers into gold and quarantine manifests."""

from __future__ import annotations

import argparse
import hashlib
import json
import os
from pathlib import Path
import tempfile
from typing import Any

import numpy as np


SCHEMA = "archon.sf18-adaptive-label-audit.v1"


def sha256(path: Path) -> str:
    return hashlib.sha256(path.read_bytes()).hexdigest()


def percentile(values: list[float], q: float) -> float | None:
    return None if not values else float(np.percentile(np.asarray(values), q))


def atomic_json(path: Path, value: dict[str, Any]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    with tempfile.NamedTemporaryFile("w", dir=path.parent, prefix=f".{path.name}.", delete=False) as f:
        staged = Path(f.name)
        json.dump(value, f, indent=2, sort_keys=True)
        f.write("\n")
        f.flush()
        os.fsync(f.fileno())
    os.replace(staged, path)


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("tiers", type=Path, nargs="+")
    parser.add_argument("--sample-manifest", type=Path, required=True)
    parser.add_argument("--output", type=Path, required=True)
    args = parser.parse_args()
    if len(args.tiers) < 1:
        parser.error("at least one tier is required")
    for path in (*args.tiers, args.sample_manifest):
        if not path.is_file():
            parser.error(f"missing input: {path}")

    tier_docs = [json.loads(path.read_text(encoding="utf-8")) for path in args.tiers]
    input_identity = tier_docs[0]["input"]
    teacher_hash = tier_docs[0]["teachers"]["stockfish18"]["sha256"]
    source_count = len(Path(input_identity["path"]).read_text(encoding="utf-8").splitlines())
    for ordinal, doc in enumerate(tier_docs):
        if doc.get("schema") != "archon.search-label-stability.v2":
            raise RuntimeError(f"tier {ordinal} is not stability schema v2")
        if doc["input"] != input_identity:
            raise RuntimeError(f"tier {ordinal} input mismatch")
        if doc["teachers"]["stockfish18"]["sha256"] != teacher_hash:
            raise RuntimeError(f"tier {ordinal} teacher mismatch")

    sample = json.loads(args.sample_manifest.read_text(encoding="utf-8"))
    if sample.get("output", {}).get("sha256") != input_identity["sha256"]:
        raise RuntimeError("sample manifest does not describe tier input")
    sample_rows = sample["selected_rows"]
    if len(sample_rows) != source_count:
        raise RuntimeError("sample row count mismatch")

    histories: dict[int, list[dict[str, Any]]] = {index: [] for index in range(source_count)}
    total_nodes = 0
    total_wall = 0.0
    tier_reports: list[dict[str, Any]] = []
    for tier_index, (path, doc) in enumerate(zip(args.tiers, tier_docs, strict=True), 1):
        rows = doc["results"]["stockfish18"]
        seen: set[int] = set()
        for row in rows:
            source_index = int(row["position_index"])
            if source_index in seen or not 0 <= source_index < source_count:
                raise RuntimeError(f"duplicate/invalid source index in tier {tier_index}")
            seen.add(source_index)
            final = row["iterations"][-1] if row["iterations"] else None
            nodes = 0 if final is None else int(final["nodes"])
            total_nodes += nodes
            total_wall += float(row["wall_seconds"])
            histories[source_index].append({
                "tier": tier_index,
                "tier_manifest": str(path.resolve()),
                "tier_manifest_sha256": sha256(path),
                "stability": row["stability"],
                "final": final,
                "wall_seconds": row["wall_seconds"],
                "nodes": nodes,
            })
        tier_reports.append({
            "tier": tier_index,
            "path": str(path.resolve()),
            "sha256": sha256(path),
            "depths": doc["depths"],
            "max_nodes_per_position": doc["max_nodes_per_position"],
            "positions": len(rows),
            "summary": doc["summary"]["stockfish18"],
        })

    gold_value: list[dict[str, Any]] = []
    gold_policy: list[dict[str, Any]] = []
    quarantine: list[dict[str, Any]] = []
    cumulative_nodes: list[float] = []
    cumulative_wall: list[float] = []
    accepted_per_tier = {str(i): 0 for i in range(1, len(tier_docs) + 1)}
    policy_per_tier = {str(i): 0 for i in range(1, len(tier_docs) + 1)}
    for source_index, history in histories.items():
        cumulative_nodes.append(float(sum(row["nodes"] for row in history)))
        cumulative_wall.append(float(sum(row["wall_seconds"] for row in history)))
        value_choice = next((row for row in history if row["stability"]["value_stable"]), None)
        policy_choice = next((row for row in history if row["stability"]["policy_stable"]), None)
        identity = {
            "source_position_index": source_index,
            "fen": sample_rows[source_index]["fen"],
            "phase": sample_rows[source_index]["phase"],
            "score_band": sample_rows[source_index]["score_band"],
            "flags": sample_rows[source_index]["flags"],
        }
        if value_choice is not None:
            accepted_per_tier[str(value_choice["tier"])] += 1
            gold_value.append({**identity, "accepted_tier": value_choice["tier"],
                               "label": value_choice["final"], "history": history})
        else:
            quarantine.append({**identity, "history": history,
                               "final_reason": history[-1]["stability"]["reason"]})
        if policy_choice is not None:
            policy_per_tier[str(policy_choice["tier"])] += 1
            gold_policy.append({**identity, "accepted_tier": policy_choice["tier"],
                                "label": policy_choice["final"]})

    result = {
        "schema": SCHEMA,
        "input": input_identity,
        "sample_manifest": {"path": str(args.sample_manifest.resolve()),
                            "sha256": sha256(args.sample_manifest)},
        "stockfish": tier_docs[0]["teachers"]["stockfish18"],
        "quality_contract": {
            "value": "first tier with exact terminal window and <=25cp span, or stable mate sign/status",
            "policy": "gold value plus stable legal best move across terminal window",
            "quarantine": "no gold value after final tier",
        },
        "tiers": tier_reports,
        "positions": source_count,
        "gold_value_positions": len(gold_value),
        "gold_value_rate": len(gold_value) / source_count,
        "gold_policy_positions": len(gold_policy),
        "gold_policy_rate": len(gold_policy) / source_count,
        "quarantined_positions": len(quarantine),
        "quarantine_rate": len(quarantine) / source_count,
        "accepted_value_per_tier": accepted_per_tier,
        "accepted_policy_per_tier": policy_per_tier,
        "cost": {
            "total_final_nodes": total_nodes,
            "total_wall_seconds": total_wall,
            "per_source_cumulative_nodes": {
                "p50": percentile(cumulative_nodes, 50), "p90": percentile(cumulative_nodes, 90),
                "p95": percentile(cumulative_nodes, 95), "p99": percentile(cumulative_nodes, 99),
                "max": max(cumulative_nodes), "mean": float(np.mean(cumulative_nodes)),
            },
            "per_source_cumulative_wall_seconds": {
                "p50": percentile(cumulative_wall, 50), "p90": percentile(cumulative_wall, 90),
                "p95": percentile(cumulative_wall, 95), "p99": percentile(cumulative_wall, 99),
                "max": max(cumulative_wall), "mean": float(np.mean(cumulative_wall)),
            },
        },
        "gold_value": gold_value,
        "gold_policy": gold_policy,
        "quarantine": quarantine,
    }
    atomic_json(args.output, result)
    print(json.dumps({key: result[key] for key in (
        "positions", "gold_value_positions", "gold_value_rate", "gold_policy_positions",
        "gold_policy_rate", "quarantined_positions", "quarantine_rate",
        "accepted_value_per_tier", "cost",
    )}, indent=2))


if __name__ == "__main__":
    main()
