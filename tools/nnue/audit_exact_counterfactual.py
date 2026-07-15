#!/usr/bin/env python3
"""Audit exact deterministic Archon intervention counterfactuals."""

from __future__ import annotations

import argparse
from collections import Counter, defaultdict
import json
from pathlib import Path


SCHEMA = "archon.search-counterfactual-exact-v1"


def relation(score: int, alpha: int, beta: int) -> int:
    return -1 if score <= alpha else (1 if score >= beta else 0)


def distribution(values: list[float]) -> dict:
    ordered = sorted(values)
    def q(fraction: float) -> float:
        at = fraction * (len(ordered) - 1)
        lo = int(at)
        hi = min(lo + 1, len(ordered) - 1)
        return ordered[lo] + (ordered[hi] - ordered[lo]) * (at - lo)
    return {"median": q(0.5), "p90": q(0.9), "p99": q(0.99), "max": ordered[-1]}


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("dataset", type=Path)
    parser.add_argument("--output", type=Path)
    args = parser.parse_args()
    manifest = json.loads((args.dataset / "manifest.json").read_text(encoding="utf-8"))
    config_hash = manifest["config_hash"]
    status = Counter()
    occurrences = Counter()
    by_reduction: dict[int, Counter] = defaultdict(Counter)
    node_fractions: dict[int, list[float]] = defaultdict(list)
    root_score_errors: dict[int, list[float]] = defaultdict(list)
    lines = set()
    roots = set()
    for path in sorted((args.dataset / "records").glob("*.json")):
        record = json.loads(path.read_text(encoding="utf-8"))
        if record.get("schema") != SCHEMA or record.get("config_hash") != config_hash:
            raise ValueError(f"{path}: schema/config mismatch")
        line = int(record["observation_line"])
        if line in lines:
            raise ValueError(f"{path}: duplicate observation line")
        lines.add(line)
        roots.add(int(record["go_ordinal"]))
        status[record["status"]] += 1
        if record["status"] != "accepted_exact_identity":
            if record["actions"]:
                raise ValueError(f"{path}: rejected identity has action labels")
            continue
        occurrence = int(record["matched_occurrence"])
        occurrences[occurrence] += 1
        references = [action for action in record["actions"] if action["reduction"] == 0]
        if len(references) != 1 or references[0]["wrong_window_relation"]:
            raise ValueError(f"{path}: invalid exact reference")
        reference_relation = relation(
            references[0]["local"]["final_score"], record["observation"]["alpha"], record["observation"]["beta"]
        )
        for action in record["actions"]:
            reduction = int(action["reduction"])
            if action["root"]["nodes"] < 0 or action["local"]["total_nodes"] < 0:
                raise ValueError(f"{path}: negative cost")
            by_reduction[reduction]["actions"] += 1
            by_reduction[reduction]["wrong_window"] += bool(action["wrong_window_relation"])
            scout_wrong = action.get("scout_wrong_window_relation")
            if scout_wrong is None:
                scout_wrong = relation(
                    action["local"]["scout_score"], record["observation"]["alpha"], record["observation"]["beta"]
                ) != reference_relation
            by_reduction[reduction]["scout_wrong_window"] += bool(scout_wrong)
            by_reduction[reduction]["root_best_changed"] += bool(action["root_best_changed"])
            by_reduction[reduction]["root_score_changed"] += action["root_score_error_cp"] != 0
            by_reduction[reduction]["root_more_nodes"] += action["root_node_fraction"] > 1.0
            node_fractions[reduction].append(float(action["root_node_fraction"]))
            root_score_errors[reduction].append(abs(float(action["root_score_error_cp"])))
    expected = len(manifest["selection"]["lines"])
    if len(lines) != expected:
        raise ValueError(f"expected {expected} records, found {len(lines)}")
    result = {
        "schema": "archon.search-counterfactual-exact-audit.v1",
        "dataset": str(args.dataset.resolve()),
        "config_hash": config_hash,
        "records": len(lines),
        "unique_root_ordinals": len(roots),
        "status": dict(status),
        "matched_occurrences": dict(sorted(occurrences.items())),
        "by_reduction": {
            str(reduction): {
                "actions": counts["actions"],
                "wrong_window_fraction": counts["wrong_window"] / counts["actions"],
                "scout_wrong_window_fraction": counts["scout_wrong_window"] / counts["actions"],
                "root_best_changed_fraction": counts["root_best_changed"] / counts["actions"],
                "root_score_changed_fraction": counts["root_score_changed"] / counts["actions"],
                "root_more_nodes_fraction": counts["root_more_nodes"] / counts["actions"],
                "root_node_fraction": distribution(node_fractions[reduction]),
                "absolute_root_score_error_cp": distribution(root_score_errors[reduction]),
            }
            for reduction, counts in sorted(by_reduction.items())
        },
    }
    rendered = json.dumps(result, indent=2, sort_keys=True) + "\n"
    if args.output:
        args.output.write_text(rendered, encoding="utf-8")
    print(rendered, end="")


if __name__ == "__main__":
    main()
