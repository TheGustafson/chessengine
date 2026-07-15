#!/usr/bin/env python3
"""Root-paired bootstrap comparisons for exact consequence-head predictions."""

from __future__ import annotations

import argparse
import json
from pathlib import Path

import numpy as np


TARGETS = ("scout_wrong_window", "wrong_window", "root_best_changed", "root_score_changed", "local_research")


def sigmoid(x: np.ndarray) -> np.ndarray:
    return 1.0 / (1.0 + np.exp(-np.clip(x, -30, 30)))


def scores(y: np.ndarray, prediction: np.ndarray) -> tuple[float, float]:
    # Keep these metrics local to avoid sklearn's repeated validation/import
    # overhead in a root bootstrap.  Group equal predictions explicitly:
    # tree baselines produce many ties, and arbitrary within-tie ordering would
    # bias both AUC and average precision.
    order = np.argsort(prediction)
    sorted_y = y[order].astype(bool)
    sorted_prediction = prediction[order]
    npos = int(sorted_y.sum())
    nneg = len(y) - npos
    boundaries = np.r_[0, np.flatnonzero(np.diff(sorted_prediction)) + 1, len(y)]
    favorable = 0.0
    negatives_before = 0
    groups = []
    for lo, hi in zip(boundaries[:-1], boundaries[1:]):
        positives = int(sorted_y[lo:hi].sum())
        negatives = hi - lo - positives
        favorable += positives * negatives_before + 0.5 * positives * negatives
        negatives_before += negatives
        groups.append((positives, negatives))
    auc = favorable / (npos * nneg)
    true_positive = false_positive = 0
    ap = 0.0
    for positives, negatives in reversed(groups):
        true_positive += positives
        false_positive += negatives
        if positives:
            ap += (positives / npos) * (true_positive / (true_positive + false_positive))
    return float(auc), float(ap)


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("predictions", type=Path)
    parser.add_argument("--left", default="context_action")
    parser.add_argument("--right", default="lowrank")
    parser.add_argument("--replicates", type=int, default=10000)
    parser.add_argument("--seed", type=int, default=4101)
    parser.add_argument("--output", type=Path)
    args = parser.parse_args()
    data = np.load(args.predictions, allow_pickle=False)
    roots = data["root_ordinal"]
    target = data["target"][:, : len(TARGETS)]
    left = sigmoid(data[args.left][:, : len(TARGETS)])
    right = sigmoid(data[args.right][:, : len(TARGETS)])
    unique = np.unique(roots)
    rows = {int(root): np.flatnonzero(roots == root) for root in unique}
    rng = np.random.default_rng(args.seed)
    result = {
        "schema": "archon.exact-head-paired-bootstrap.v1",
        "predictions": str(args.predictions.resolve()),
        "comparison": f"{args.right} minus {args.left}",
        "held_roots": len(unique),
        "replicates_requested": args.replicates,
        "targets": {},
    }
    for column, name in enumerate(TARGETS):
        full_left = scores(target[:, column], left[:, column])
        full_right = scores(target[:, column], right[:, column])
        differences = []
        for _ in range(args.replicates):
            sampled = rng.choice(unique, len(unique), replace=True)
            index = np.concatenate([rows[int(root)] for root in sampled])
            y = target[index, column]
            if len(np.unique(y)) < 2:
                continue
            ls = scores(y, left[index, column])
            rs = scores(y, right[index, column])
            differences.append((rs[0] - ls[0], rs[1] - ls[1]))
        delta = np.asarray(differences)
        result["targets"][name] = {
            "left": {"roc_auc": full_left[0], "average_precision": full_left[1]},
            "right": {"roc_auc": full_right[0], "average_precision": full_right[1]},
            "delta": {
                "roc_auc": float(full_right[0] - full_left[0]),
                "roc_auc_ci95": np.quantile(delta[:, 0], [0.025, 0.975]).tolist(),
                "average_precision": float(full_right[1] - full_left[1]),
                "average_precision_ci95": np.quantile(delta[:, 1], [0.025, 0.975]).tolist(),
            },
            "replicates": len(delta),
        }
    rendered = json.dumps(result, indent=2, sort_keys=True) + "\n"
    if args.output:
        args.output.write_text(rendered, encoding="utf-8")
    print(rendered, end="")


if __name__ == "__main__":
    main()
