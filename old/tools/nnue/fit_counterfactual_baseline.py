#!/usr/bin/env python3
"""Fit context-only and protected-latent cold-counterfactual baselines."""

from __future__ import annotations

import argparse
import json
from pathlib import Path

import numpy as np
from sklearn.decomposition import PCA
from sklearn.ensemble import HistGradientBoostingClassifier, HistGradientBoostingRegressor
from sklearn.metrics import average_precision_score, brier_score_loss, mean_absolute_error, r2_score, roc_auc_score


def classify(x_train, y_train, x_val, y_val) -> dict:
    result = {"rows": len(y_val), "positive_fraction": float(y_val.mean()), "positives": int(y_val.sum())}
    if len(np.unique(y_train)) < 2 or len(np.unique(y_val)) < 2:
        return {**result, "error": "both classes required"}
    model = HistGradientBoostingClassifier(
        max_iter=120, learning_rate=0.08, max_leaf_nodes=15, min_samples_leaf=10,
        l2_regularization=1.0, random_state=4101,
    ).fit(x_train, y_train)
    probability = model.predict_proba(x_val)[:, 1]
    return {
        **result,
        "roc_auc": float(roc_auc_score(y_val, probability)),
        "average_precision": float(average_precision_score(y_val, probability)),
        "brier": float(brier_score_loss(y_val, probability)),
    }


def regress(x_train, y_train, x_val, y_val) -> dict:
    model = HistGradientBoostingRegressor(
        max_iter=120, learning_rate=0.08, max_leaf_nodes=15, min_samples_leaf=10,
        l2_regularization=1.0, random_state=4101,
    ).fit(x_train, y_train)
    prediction = model.predict(x_val)
    return {
        "rows": len(y_val),
        "r2": float(r2_score(y_val, prediction)),
        "mae_log": float(mean_absolute_error(y_val, prediction)),
        "median_multiplicative_error": float(np.exp(np.median(np.abs(y_val - prediction)))),
    }


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("tensor", type=Path)
    parser.add_argument("--output", type=Path)
    args = parser.parse_args()
    data = np.load(args.tensor, allow_pickle=False)
    validation = data["validation"].astype(bool)
    context = data["context_action"].astype(np.float32)
    latent = data["latent"].astype(np.float32) / 255.0
    components = min(32, latent[~validation].shape[0], latent.shape[1])
    pca = PCA(n_components=components, svd_solver="randomized", random_state=4101).fit(latent[~validation])
    latent_train = pca.transform(latent[~validation])
    latent_val = pca.transform(latent[validation])
    feature_sets = {
        "context_action_only": (context[~validation], context[validation]),
        f"protected_latent_pca{components}_plus_context_action": (
            np.concatenate((latent_train, context[~validation]), axis=1),
            np.concatenate((latent_val, context[validation]), axis=1),
        ),
    }
    models = {}
    for name, (train_x, val_x) in feature_sets.items():
        models[name] = {
            "wrong_window": classify(train_x, data["y_wrong_window"][~validation], val_x, data["y_wrong_window"][validation]),
            "relation_lower": classify(train_x, data["y_relation_lower"][~validation], val_x, data["y_relation_lower"][validation]),
            "log_nodes": regress(train_x, data["y_log_nodes"][~validation], val_x, data["y_log_nodes"][validation]),
            "log_absolute_score_error": regress(train_x, data["y_log_absolute_score_error"][~validation], val_x, data["y_log_absolute_score_error"][validation]),
        }
    result = {
        "schema": "archon.search-counterfactual-baseline.v1",
        "data": json.loads(str(data["metadata"])),
        "split_counts": {"train": int((~validation).sum()), "validation": int(validation.sum())},
        "pca_explained_variance": float(pca.explained_variance_ratio_.sum()),
        "models": models,
    }
    rendered = json.dumps(result, indent=2, sort_keys=True) + "\n"
    if args.output:
        args.output.write_text(rendered, encoding="utf-8")
    print(rendered, end="")


if __name__ == "__main__":
    main()
