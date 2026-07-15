#!/usr/bin/env python3
"""Build grouped tensors for cold counterfactual reduction consequences."""

from __future__ import annotations

import argparse
import json
from pathlib import Path

import numpy as np

from build_search_trace_npz import context


SCHEMA = "archon.search-counterfactual-cold-v1"


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("dataset", type=Path)
    parser.add_argument("output", type=Path)
    args = parser.parse_args()
    manifest = json.loads((args.dataset / "manifest.json").read_text(encoding="utf-8"))
    config_hash = manifest["config_hash"]
    latent = []
    features = []
    roots = []
    wrong = []
    log_nodes = []
    absolute_error = []
    relation_lower = []
    for path in sorted((args.dataset / "records").glob("*.json")):
        record = json.loads(path.read_text(encoding="utf-8"))
        if record.get("schema") != SCHEMA or record.get("config_hash") != config_hash:
            raise ValueError(f"{path}: schema/config mismatch")
        observation = record["observation"]
        base = context(observation)
        for action in record["actions"]:
            latent.append(observation["latent"])
            features.append(
                base
                + [
                    action["reduction"] / 8.0,
                    action["child_depth"] / 16.0,
                    action["root_depth"] / 16.0,
                ]
            )
            roots.append(int(record["root_position_key"]))
            wrong.append(action["wrong_window_relation"])
            log_nodes.append(np.log1p(action["nodes"]))
            absolute_error.append(np.log1p(action["absolute_score_error_cp"]))
            relation_lower.append(action["window_relation"] == "lower")
    if not latent:
        raise ValueError("no counterfactual actions")
    root = np.asarray(roots, dtype=np.uint64)
    validation = root % 1000 < 200
    if not validation.any() or validation.all():
        raise ValueError("empty grouped split")
    metadata = {
        "schema": "archon.search-counterfactual-tensor.v1",
        "dataset": str(args.dataset.resolve()),
        "config_hash": config_hash,
        "rows": len(latent),
        "latent_dim": len(latent[0]),
        "context_action_dim": len(features[0]),
        "split": "validation iff root_position_key % 1000 < 200",
        "contract": "cold full-FEN replay; no repetition history; fresh TT/histories per action",
    }
    args.output.parent.mkdir(parents=True, exist_ok=True)
    np.savez_compressed(
        args.output,
        latent=np.asarray(latent, dtype=np.int16),
        context_action=np.asarray(features, dtype=np.float32),
        root_key=root,
        validation=validation,
        y_wrong_window=np.asarray(wrong, dtype=np.uint8),
        y_log_nodes=np.asarray(log_nodes, dtype=np.float32),
        y_log_absolute_score_error=np.asarray(absolute_error, dtype=np.float32),
        y_relation_lower=np.asarray(relation_lower, dtype=np.uint8),
        metadata=np.asarray(json.dumps(metadata, sort_keys=True)),
    )
    print(json.dumps(metadata, indent=2, sort_keys=True))


if __name__ == "__main__":
    main()
