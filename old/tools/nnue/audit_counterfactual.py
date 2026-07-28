#!/usr/bin/env python3
"""Audit Archon cold counterfactual reduction records."""

from __future__ import annotations

import argparse
from collections import Counter, defaultdict
import json
from pathlib import Path


SCHEMA = "archon.search-counterfactual-cold-v1"


def quantiles(values: list[float]) -> dict:
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
    counts = Counter()
    by_reduction: dict[int, Counter] = defaultdict(Counter)
    absolute_errors: dict[int, list[float]] = defaultdict(list)
    node_fractions: dict[int, list[float]] = defaultdict(list)
    observation_lines = set()
    root_keys = set()
    for path in sorted((args.dataset / "records").glob("*.json")):
        record = json.loads(path.read_text(encoding="utf-8"))
        if record.get("schema") != SCHEMA or record.get("config_hash") != config_hash:
            raise ValueError(f"{path}: schema/config mismatch")
        line = int(record["observation_line"])
        if line in observation_lines:
            raise ValueError(f"{path}: duplicate observation line {line}")
        observation_lines.add(line)
        root_keys.add(int(record["root_position_key"]))
        reference = [action for action in record["actions"] if action["reduction"] == 0]
        if len(reference) != 1 or reference[0]["wrong_window_relation"]:
            raise ValueError(f"{path}: invalid reference action")
        counts["records"] += 1
        counts["observed_lmr_research"] += bool(record["observation"]["lmr_research"])
        for action in record["actions"]:
            reduction = int(action["reduction"])
            if action["nodes"] < 0 or not 0 <= action["node_fraction_of_reference"]:
                raise ValueError(f"{path}: invalid action cost")
            counts["actions"] += 1
            by_reduction[reduction]["actions"] += 1
            by_reduction[reduction]["wrong_window_relation"] += bool(action["wrong_window_relation"])
            absolute_errors[reduction].append(float(action["absolute_score_error_cp"]))
            node_fractions[reduction].append(float(action["node_fraction_of_reference"]))
    expected = len(manifest["selected_observation_lines"])
    if counts["records"] != expected:
        raise ValueError(f"expected {expected} records, found {counts['records']}")
    result = {
        "schema": "archon.search-counterfactual-audit.v1",
        "dataset": str(args.dataset.resolve()),
        "config_hash": config_hash,
        "records": counts["records"],
        "actions": counts["actions"],
        "unique_root_keys": len(root_keys),
        "observed_lmr_research_fraction": counts["observed_lmr_research"] / counts["records"],
        "by_reduction": {
            str(reduction): {
                "actions": counter["actions"],
                "wrong_window_relation": counter["wrong_window_relation"],
                "wrong_window_fraction": counter["wrong_window_relation"] / counter["actions"],
                "absolute_score_error_cp": quantiles(absolute_errors[reduction]),
                "node_fraction_of_reference": quantiles(node_fractions[reduction]),
            }
            for reduction, counter in sorted(by_reduction.items())
        },
    }
    rendered = json.dumps(result, indent=2, sort_keys=True) + "\n"
    if args.output:
        args.output.write_text(rendered, encoding="utf-8")
    print(rendered, end="")


if __name__ == "__main__":
    main()
