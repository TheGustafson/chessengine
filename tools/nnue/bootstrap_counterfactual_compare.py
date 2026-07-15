#!/usr/bin/env python3
"""Paired root-bootstrap comparison for counterfactual error-risk models."""

from __future__ import annotations

import argparse
import json
from pathlib import Path

import numpy as np
from sklearn.decomposition import PCA
from sklearn.ensemble import HistGradientBoostingClassifier
from sklearn.metrics import average_precision_score, brier_score_loss, roc_auc_score


def fit(train_x, train_y):
    return HistGradientBoostingClassifier(
        max_iter=160, learning_rate=0.08, max_leaf_nodes=31, min_samples_leaf=20,
        l2_regularization=1.0, random_state=4101,
    ).fit(train_x, train_y)


def metrics(y, probability) -> dict:
    return {
        "roc_auc": float(roc_auc_score(y, probability)),
        "average_precision": float(average_precision_score(y, probability)),
        "brier": float(brier_score_loss(y, probability)),
    }


def calibration(y, probability, bins=10) -> list[dict]:
    order = np.argsort(probability)
    result = []
    for indices in np.array_split(order, bins):
        if len(indices):
            result.append(
                {
                    "rows": len(indices),
                    "mean_prediction": float(probability[indices].mean()),
                    "observed_fraction": float(y[indices].mean()),
                }
            )
    return result


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("tensor", type=Path)
    parser.add_argument("--output", type=Path)
    parser.add_argument("--bootstrap", type=int, default=2000)
    args = parser.parse_args()
    data = np.load(args.tensor, allow_pickle=False)
    validation = data["validation"].astype(bool)
    y = data["y_wrong_window"].astype(np.uint8)
    context = data["context_action"].astype(np.float32)
    latent = data["latent"].astype(np.float32) / 255.0
    pca = PCA(n_components=min(32, (~validation).sum()), svd_solver="randomized", random_state=4101).fit(latent[~validation])
    train_latent = pca.transform(latent[~validation])
    val_latent = pca.transform(latent[validation])
    feature_sets = {
        "action_only": (context[~validation, -3:], context[validation, -3:]),
        "context_action": (context[~validation], context[validation]),
        "protected_latent_plus_context_action": (
            np.concatenate((train_latent, context[~validation]), axis=1),
            np.concatenate((val_latent, context[validation]), axis=1),
        ),
    }
    predictions = {
        name: fit(train_x, y[~validation]).predict_proba(val_x)[:, 1]
        for name, (train_x, val_x) in feature_sets.items()
    }
    val_y = y[validation]
    roots = data["root_key"][validation]
    unique_roots = np.unique(roots)
    rng = np.random.default_rng(4101)
    deltas = {metric: [] for metric in ("roc_auc", "average_precision", "brier")}
    for _ in range(args.bootstrap):
        sampled = rng.choice(unique_roots, size=len(unique_roots), replace=True)
        indices = np.concatenate([np.flatnonzero(roots == root) for root in sampled])
        sample_y = val_y[indices]
        if len(np.unique(sample_y)) < 2:
            continue
        context_metrics = metrics(sample_y, predictions["context_action"][indices])
        latent_metrics = metrics(sample_y, predictions["protected_latent_plus_context_action"][indices])
        for metric in deltas:
            deltas[metric].append(latent_metrics[metric] - context_metrics[metric])
    comparison = {}
    for metric, values in deltas.items():
        array = np.asarray(values)
        comparison[metric] = {
            "latent_minus_context": float(
                metrics(val_y, predictions["protected_latent_plus_context_action"])[metric]
                - metrics(val_y, predictions["context_action"])[metric]
            ),
            "root_bootstrap_95": [float(np.quantile(array, 0.025)), float(np.quantile(array, 0.975))],
            "replicates": len(array),
        }
    result = {
        "schema": "archon.search-counterfactual-bootstrap.v1",
        "data": json.loads(str(data["metadata"])),
        "validation": {
            "rows": len(val_y), "roots": len(unique_roots),
            "positives": int(val_y.sum()), "positive_fraction": float(val_y.mean()),
        },
        "models": {
            name: {**metrics(val_y, probability), "calibration": calibration(val_y, probability)}
            for name, probability in predictions.items()
        },
        "paired_comparison": comparison,
    }
    rendered = json.dumps(result, indent=2, sort_keys=True) + "\n"
    if args.output:
        args.output.write_text(rendered, encoding="utf-8")
    print(rendered, end="")


if __name__ == "__main__":
    main()
