#!/usr/bin/env python3
"""Root-paired bootstrap for independently fitted exact-target baselines."""

from __future__ import annotations

import argparse
import json
from pathlib import Path

import numpy as np

from bootstrap_exact_head_compare import scores


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("predictions", type=Path)
    parser.add_argument("--target", required=True)
    parser.add_argument("--left", required=True)
    parser.add_argument("--right", required=True)
    parser.add_argument("--replicates", type=int, default=10000)
    parser.add_argument("--seed", type=int, default=4101)
    parser.add_argument("--output", type=Path)
    args = parser.parse_args()
    data = np.load(args.predictions, allow_pickle=False)
    roots = data["root_ordinal"]
    y = data[f"y__{args.target}"]
    left = data[f"{args.left}__{args.target}"]
    right = data[f"{args.right}__{args.target}"]
    unique = np.unique(roots)
    rows = {int(root): np.flatnonzero(roots == root) for root in unique}
    rng = np.random.default_rng(args.seed)
    delta = []
    for _ in range(args.replicates):
        sampled = rng.choice(unique, len(unique), replace=True)
        index = np.concatenate([rows[int(root)] for root in sampled])
        if len(np.unique(y[index])) < 2:
            continue
        ls = scores(y[index], left[index])
        rs = scores(y[index], right[index])
        delta.append((rs[0] - ls[0], rs[1] - ls[1]))
    delta = np.asarray(delta)
    ls = scores(y, left)
    rs = scores(y, right)
    result = {
        "schema": "archon.independent-baseline-paired-bootstrap.v1",
        "predictions": str(args.predictions.resolve()),
        "target": args.target,
        "comparison": f"{args.right} minus {args.left}",
        "held_roots": len(unique),
        "replicates": len(delta),
        "left": {"roc_auc": ls[0], "average_precision": ls[1]},
        "right": {"roc_auc": rs[0], "average_precision": rs[1]},
        "delta": {
            "roc_auc": rs[0] - ls[0],
            "roc_auc_ci95": np.quantile(delta[:, 0], [0.025, 0.975]).tolist(),
            "average_precision": rs[1] - ls[1],
            "average_precision_ci95": np.quantile(delta[:, 1], [0.025, 0.975]).tolist(),
        },
    }
    rendered = json.dumps(result, indent=2, sort_keys=True) + "\n"
    if args.output:
        args.output.write_text(rendered, encoding="utf-8")
    print(rendered, end="")


if __name__ == "__main__":
    main()
