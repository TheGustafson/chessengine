#!/usr/bin/env python3
"""Fit small leakage-safe baselines to a root-action consequence tensor."""

from __future__ import annotations

import argparse
import json
from pathlib import Path

import numpy as np
from sklearn.linear_model import LogisticRegression, Ridge
from sklearn.metrics import brier_score_loss, log_loss, mean_absolute_error, r2_score, roc_auc_score
from sklearn.decomposition import PCA
from sklearn.ensemble import HistGradientBoostingClassifier, HistGradientBoostingRegressor
from sklearn.pipeline import make_pipeline
from sklearn.preprocessing import StandardScaler


def classification(x_train, y_train, x_val, y_val, balanced: bool = False):
    model = make_pipeline(
        StandardScaler(),
        LogisticRegression(C=0.1, max_iter=1000, class_weight="balanced" if balanced else None),
    )
    model.fit(x_train, y_train)
    probability = model.predict_proba(x_val)[:, 1]
    return model, probability, {
        "rows": len(y_val),
        "positive_fraction": float(np.mean(y_val)),
        "roc_auc": float(roc_auc_score(y_val, probability)),
        "brier": float(brier_score_loss(y_val, probability)),
        "log_loss": float(log_loss(y_val, probability)),
    }


def regression(x_train, y_train, x_val, y_val):
    model = make_pipeline(StandardScaler(), Ridge(alpha=10.0))
    model.fit(x_train, y_train)
    prediction = model.predict(x_val)
    return {
        "rows": len(y_val),
        "r2": float(r2_score(y_val, prediction)),
        "mae_log": float(mean_absolute_error(y_val, prediction)),
        "median_multiplicative_error": float(np.median(np.exp(np.abs(y_val - prediction)))),
    }


def tree_classification(x_train, y_train, x_val, y_val, balanced: bool = False):
    model = HistGradientBoostingClassifier(
        learning_rate=0.05, max_iter=100, max_leaf_nodes=15, l2_regularization=1.0,
        random_state=20260715,
    )
    sample_weight = None
    if balanced:
        positive = max(1, int(y_train.sum()))
        negative = max(1, len(y_train) - positive)
        sample_weight = np.where(y_train, len(y_train) / (2 * positive), len(y_train) / (2 * negative))
    model.fit(x_train, y_train, sample_weight=sample_weight)
    probability = model.predict_proba(x_val)[:, 1]
    return probability, {
        "rows": len(y_val), "positive_fraction": float(np.mean(y_val)),
        "roc_auc": float(roc_auc_score(y_val, probability)),
        "brier": float(brier_score_loss(y_val, probability)),
        "log_loss": float(log_loss(y_val, probability)),
    }


def tree_regression(x_train, y_train, x_val, y_val):
    model = HistGradientBoostingRegressor(
        learning_rate=0.05, max_iter=100, max_leaf_nodes=15, l2_regularization=1.0,
        random_state=20260715,
    )
    model.fit(x_train, y_train)
    prediction = model.predict(x_val)
    return {
        "rows": len(y_val), "r2": float(r2_score(y_val, prediction)),
        "mae_log": float(mean_absolute_error(y_val, prediction)),
        "median_multiplicative_error": float(np.median(np.exp(np.abs(y_val - prediction)))),
    }


def ranking(groups, target, probability):
    correct = 0
    reciprocal = 0.0
    count = 0
    random_top1 = 0.0
    for group in np.unique(groups):
        indices = np.flatnonzero(groups == group)
        truths = target[indices]
        if not truths.any():
            continue
        order = indices[np.argsort(-probability[indices])]
        winning = set(indices[truths].tolist())
        rank = next(position for position, index in enumerate(order, 1) if int(index) in winning)
        correct += rank == 1
        reciprocal += 1.0 / rank
        random_top1 += len(winning) / len(indices)
        count += 1
    return {"positions": count, "top1": correct / count, "mrr": reciprocal / count,
            "random_tie_aware_top1": random_top1 / count}


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("data", type=Path)
    parser.add_argument("--output", type=Path)
    args = parser.parse_args()
    with np.load(args.data, allow_pickle=False) as values:
        full = values["features"].astype(np.float64)
        context = values["context_only"].astype(np.float64)
        val = values["is_validation"]
        error = values["error"].astype(bool)
        best = values["best"].astype(bool)
        groups = values["group"]
        log_low_nodes = values["log_low_nodes"]
        log_high_nodes = values["log_high_nodes"]
        metadata = json.loads(values["metadata_json"].tobytes().decode())
    train = ~val
    pca_components = min(32, int(train.sum()) - 1, full.shape[1] - context.shape[1])
    pca = PCA(n_components=pca_components, whiten=True, random_state=20260715)
    latent_train = pca.fit_transform(full[train, :512])
    latent_val = pca.transform(full[val, :512])
    pca_full = np.empty((len(full), pca_components + context.shape[1]), dtype=np.float64)
    pca_full[train] = np.concatenate([latent_train, context[train]], axis=1)
    pca_full[val] = np.concatenate([latent_val, context[val]], axis=1)
    report = {"schema": "archon.root-consequence-baseline.v1", "data": metadata, "models": {}}
    probabilities = {}
    for name, features in (
        ("context_only", context),
        ("protected_latent_plus_context", full),
        (f"protected_latent_pca{pca_components}_plus_context", pca_full),
    ):
        error_model, error_probability, error_metrics = classification(
            features[train], error[train], features[val], error[val]
        )
        _best_model, best_probability, best_metrics = classification(
            features[train], best[train], features[val], best[val], balanced=True
        )
        probabilities[name] = best_probability
        report["models"][name] = {
            "error50": error_metrics,
            "best_move": {**best_metrics, "ranking": ranking(groups[val], best[val], best_probability)},
            "low_node_cost": regression(features[train], log_low_nodes[train],
                                        features[val], log_low_nodes[val]),
            "high_node_cost": regression(features[train], log_high_nodes[train],
                                         features[val], log_high_nodes[val]),
        }
    error_probability, error_metrics = tree_classification(
        pca_full[train], error[train], pca_full[val], error[val]
    )
    best_probability, best_metrics = tree_classification(
        pca_full[train], best[train], pca_full[val], best[val], balanced=True
    )
    report["models"][f"protected_latent_pca{pca_components}_histgb"] = {
        "error50": error_metrics,
        "best_move": {**best_metrics, "ranking": ranking(groups[val], best[val], best_probability)},
        "low_node_cost": tree_regression(pca_full[train], log_low_nodes[train],
                                         pca_full[val], log_low_nodes[val]),
        "high_node_cost": tree_regression(pca_full[train], log_high_nodes[train],
                                          pca_full[val], log_high_nodes[val]),
    }
    rendered = json.dumps(report, indent=2, sort_keys=True) + "\n"
    if args.output:
        args.output.parent.mkdir(parents=True, exist_ok=True)
        args.output.write_text(rendered)
    print(rendered, end="")


if __name__ == "__main__":
    main()
