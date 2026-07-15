#!/usr/bin/env python3
"""Materialize grouped, mover-normalized tensors from Archon LMR telemetry."""

from __future__ import annotations

import argparse
import json
from pathlib import Path

import numpy as np


SCHEMAS = {"archon.search-lmr-observation.v2", "archon.search-lmr-observation.v3"}


def clipped(value: int, scale: float, limit: float = 32.0) -> float:
    return max(-limit, min(limit, value / scale))


def context(row: dict) -> list[float]:
    node_kind = int(row["node_kind"])
    tt_bound = int(row["tt_bound"])
    raw = int(row["raw_move"])
    source = raw & 63
    destination = (raw >> 6) & 63
    if int(row["side_to_move"]) == 1:
        source ^= 56
        destination ^= 56
    return [
        row["depth"] / 16.0,
        row["ply"] / 64.0,
        clipped(row["alpha"], 1000.0),
        clipped(row["beta"], 1000.0),
        clipped(row["static_eval"], 1000.0),
        clipped(row["best_value_before"], 1000.0),
        *(float(node_kind == kind) for kind in range(4)),
        float(row["in_check"]),
        float(row["improving"]),
        float(row["cut_node"]),
        float(row["tt_hit"]),
        row["tt_depth"] / 16.0,
        *(float(tt_bound == bound) for bound in range(4)),
        float(row["tt_move"]),
        min(row["move_index"], 128) / 64.0,
        float(row["capture"]),
        float(row["gives_check"]),
        clipped(row["stat_score"], 20000.0),
        clipped(row["base_reduction"], 4096.0),
        clipped(row["final_reduction"], 4096.0),
        row["extension"] / 3.0,
        row["nominal_child_depth"] / 16.0,
        row["scout_depth"] / 16.0,
        (row["nominal_child_depth"] - row["scout_depth"]) / 8.0,
        source / 63.0,
        destination / 63.0,
        row["moving_piece"] / 5.0,
        row["captured_piece"] / 6.0,
        row["promotion_piece"] / 6.0,
        ((raw >> 14) & 3) / 3.0,
    ]


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("trace", type=Path)
    parser.add_argument("output", type=Path)
    parser.add_argument("--validation-per-mille", type=int, default=200)
    args = parser.parse_args()
    if not 1 <= args.validation_per_mille <= 999:
        parser.error("--validation-per-mille must be in 1..999")

    latents = []
    contexts = []
    root_keys = []
    position_keys = []
    lmr_research = []
    alpha_raise = []
    beta_cutoff = []
    log_scout_nodes = []
    log_total_nodes = []
    score_gain = []
    current_root = None
    declared = remaining = 0
    with args.trace.open(encoding="utf-8") as source:
        header = json.loads(next(source))
        trace_schema = header.get("schema")
        if trace_schema not in SCHEMAS or header.get("kind") != "header":
            raise ValueError("invalid trace header")
        for line_number, line in enumerate(source, 2):
            row = json.loads(line)
            if row.get("schema") != trace_schema:
                raise ValueError(f"line {line_number}: schema mismatch")
            if row.get("kind") == "go":
                if remaining:
                    raise ValueError(f"line {line_number}: prior go is short by {remaining} rows")
                current_root = int(row["root_position_key"])
                declared += int(row["rows"])
                remaining = int(row["rows"])
                continue
            if row.get("kind") != "row" or current_root is None or remaining <= 0:
                raise ValueError(f"line {line_number}: row outside declared go block")
            remaining -= 1
            latents.append(row["latent"])
            contexts.append(context(row))
            root_keys.append(current_root)
            position_keys.append(int(row["position_key"]))
            lmr_research.append(row["lmr_research"])
            alpha_raise.append(row["alpha_raise"])
            beta_cutoff.append(row["beta_cutoff"])
            log_scout_nodes.append(np.log1p(row["scout_nodes"]))
            log_total_nodes.append(np.log1p(row["total_nodes"]))
            score_gain.append(np.clip(row["final_score"] - row["alpha"], -2000, 2000) / 1000.0)
    if remaining:
        raise ValueError(f"final go is short by {remaining} rows")
    if declared != len(latents) or not latents:
        raise ValueError(f"declared {declared} rows but materialized {len(latents)}")

    root = np.asarray(root_keys, dtype=np.uint64)
    validation = root % 1000 < args.validation_per_mille
    if not validation.any() or validation.all():
        raise ValueError("grouped split produced an empty partition")
    metadata = {
        "schema": "archon.search-lmr-tensor.v1",
        "source": str(args.trace.resolve()),
        "source_schema": trace_schema,
        "rows": len(latents),
        "latent_dim": len(latents[0]),
        "context_dim": len(contexts[0]),
        "split": f"validation iff root_position_key % 1000 < {args.validation_per_mille}",
        "feature_contract": "protected raw signed [stm;ntm] classic accumulator (consumer must apply net activation) + pre-action context + mover-normalized move",
        "target_contract": "observational classical LMR outcomes; not counterfactual bound-error labels",
    }
    args.output.parent.mkdir(parents=True, exist_ok=True)
    np.savez_compressed(
        args.output,
        latent=np.asarray(latents, dtype=np.int16),
        context=np.asarray(contexts, dtype=np.float32),
        root_key=root,
        position_key=np.asarray(position_keys, dtype=np.uint64),
        validation=validation,
        y_lmr_research=np.asarray(lmr_research, dtype=np.uint8),
        y_alpha_raise=np.asarray(alpha_raise, dtype=np.uint8),
        y_beta_cutoff=np.asarray(beta_cutoff, dtype=np.uint8),
        y_log_scout_nodes=np.asarray(log_scout_nodes, dtype=np.float32),
        y_log_total_nodes=np.asarray(log_total_nodes, dtype=np.float32),
        y_score_gain=np.asarray(score_gain, dtype=np.float32),
        metadata=np.asarray(json.dumps(metadata, sort_keys=True)),
    )
    print(json.dumps(metadata, indent=2, sort_keys=True))


if __name__ == "__main__":
    main()
