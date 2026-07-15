#!/usr/bin/env python3
"""Validate and summarize Archon observational LMR JSONL telemetry."""

from __future__ import annotations

import argparse
import json
import math
from collections import Counter
from pathlib import Path


SCHEMAS = {"archon.search-lmr-observation.v2", "archon.search-lmr-observation.v3"}


def percentile(values: list[float], q: float) -> float:
    if not values:
        return math.nan
    ordered = sorted(values)
    at = q * (len(ordered) - 1)
    lo = int(at)
    hi = min(lo + 1, len(ordered) - 1)
    return ordered[lo] + (ordered[hi] - ordered[lo]) * (at - lo)


def summary(path: Path) -> dict:
    with path.open(encoding="utf-8") as source:
        try:
            header = json.loads(next(source))
        except StopIteration as error:
            raise ValueError("empty trace") from error
        trace_schema = header.get("schema")
        if trace_schema not in SCHEMAS or header.get("kind") != "header":
            raise ValueError("invalid trace header")
        latent_dim = int(header["latent_dim"])
        counts: Counter[str] = Counter()
        depths: Counter[int] = Counter()
        scout_nodes: list[float] = []
        total_nodes: list[float] = []
        reductions: list[float] = []
        keys: set[int] = set()
        root_keys: set[int] = set()
        go_rows_declared = 0
        go_records = 0
        rows = 0
        for line_number, line in enumerate(source, 2):
            row = json.loads(line)
            if row.get("schema") != trace_schema:
                raise ValueError(f"line {line_number}: schema/kind mismatch")
            if row.get("kind") == "go":
                root_keys.add(int(row["root_position_key"]))
                go_rows_declared += int(row["rows"])
                go_records += 1
                continue
            if row.get("kind") != "row":
                raise ValueError(f"line {line_number}: schema/kind mismatch")
            latent = row.get("latent")
            if not isinstance(latent, list) or len(latent) != latent_dim:
                raise ValueError(f"line {line_number}: latent dimension mismatch")
            if any(not isinstance(value, int) or not -32768 <= value <= 32767 for value in latent):
                raise ValueError(f"line {line_number}: invalid latent value")
            if trace_schema.endswith(".v3"):
                fen = row.get("fen")
                if not isinstance(fen, str) or len(fen.split()) != 6:
                    raise ValueError(f"line {line_number}: invalid full-FEN replay state")
            if row["depth"] < 2 or row["move_index"] <= 1:
                raise ValueError(f"line {line_number}: not an LMR-eligible action")
            if row["side_to_move"] not in (0, 1) or not 0 <= row["moving_piece"] < 6:
                raise ValueError(f"line {line_number}: invalid mover features")
            if not 0 <= row["captured_piece"] <= 6 or not 0 <= row["promotion_piece"] <= 6:
                raise ValueError(f"line {line_number}: invalid captured/promotion features")
            if row["scout_depth"] <= 0 or row["scout_nodes"] < 0 or row["total_nodes"] < row["scout_nodes"]:
                raise ValueError(f"line {line_number}: invalid search cost")
            # A re-search can return from TT without making another move, so
            # invocation=true with zero newly counted nodes is legitimate.
            if row["research_nodes"] > 0 and not row["lmr_research"]:
                raise ValueError(f"line {line_number}: LMR re-search cost without invocation")
            if row["pv_research_nodes"] > 0 and not row["pv_research"]:
                raise ValueError(f"line {line_number}: PV re-search cost without invocation")
            for target in ("lmr_research", "pv_research", "alpha_raise", "beta_cutoff", "capture", "gives_check", "tt_move"):
                counts[target] += bool(row[target])
            depths[int(row["depth"])] += 1
            keys.add(int(row["position_key"]))
            scout_nodes.append(float(row["scout_nodes"]))
            total_nodes.append(float(row["total_nodes"]))
            reductions.append((row["nominal_child_depth"] - row["scout_depth"]))
            rows += 1
    if rows == 0:
        raise ValueError("trace contains no observations")
    if go_records and go_rows_declared != rows:
        raise ValueError(f"go records declare {go_rows_declared} rows, found {rows}")

    def distribution(values: list[float]) -> dict:
        return {
            "median": percentile(values, 0.5),
            "p90": percentile(values, 0.9),
            "p99": percentile(values, 0.99),
            "max": max(values),
        }

    return {
        "schema": "archon.search-lmr-audit.v1",
        "path": str(path.resolve()),
        "header": header,
        "rows": rows,
        "go_records": go_records,
        "unique_root_position_keys": len(root_keys),
        "unique_position_keys": len(keys),
        "depths": dict(sorted(depths.items())),
        "fractions": {name: count / rows for name, count in sorted(counts.items())},
        "effective_reduction_plies": distribution(reductions),
        "scout_nodes": distribution(scout_nodes),
        "total_nodes": distribution(total_nodes),
    }


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("trace", type=Path)
    parser.add_argument("--output", type=Path)
    args = parser.parse_args()
    result = summary(args.trace)
    rendered = json.dumps(result, indent=2, sort_keys=True) + "\n"
    if args.output:
        args.output.write_text(rendered, encoding="utf-8")
    print(rendered, end="")


if __name__ == "__main__":
    main()
