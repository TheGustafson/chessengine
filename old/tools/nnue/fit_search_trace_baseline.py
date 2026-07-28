#!/usr/bin/env python3
"""Fit context and protected-latent baselines to observational LMR targets."""

from __future__ import annotations

import argparse
import json
from pathlib import Path

import numpy as np
from sklearn.decomposition import PCA
from sklearn.ensemble import HistGradientBoostingClassifier, HistGradientBoostingRegressor
from sklearn.metrics import (
    average_precision_score,
    brier_score_loss,
    log_loss,
    mean_absolute_error,
    r2_score,
    roc_auc_score,
)


def classification(y_train, y_val, x_train, x_val) -> dict:
    if len(np.unique(y_train)) != 2 or len(np.unique(y_val)) != 2:
        return {"error": "target lacks both classes", "positive_fraction": float(y_val.mean())}
    model = HistGradientBoostingClassifier(
        learning_rate=0.08, max_iter=160, max_leaf_nodes=31, min_samples_leaf=30,
        l2_regularization=1.0, random_state=4101,
    ).fit(x_train, y_train)
    probability = model.predict_proba(x_val)[:, 1]
    return {
        "rows": len(y_val),
        "positive_fraction": float(y_val.mean()),
        "roc_auc": float(roc_auc_score(y_val, probability)),
        "average_precision": float(average_precision_score(y_val, probability)),
        "brier": float(brier_score_loss(y_val, probability)),
        "log_loss": float(log_loss(y_val, probability)),
    }


def regression(y_train, y_val, x_train, x_val) -> dict:
    model = HistGradientBoostingRegressor(
        learning_rate=0.08, max_iter=160, max_leaf_nodes=31, min_samples_leaf=30,
        l2_regularization=1.0, random_state=4101,
    ).fit(x_train, y_train)
    prediction = model.predict(x_val)
    return {
        "rows": len(y_val),
        "r2": float(r2_score(y_val, prediction)),
        "mae_log": float(mean_absolute_error(y_val, prediction)),
        "median_multiplicative_error": float(np.exp(np.median(np.abs(y_val - prediction)))),
    }


def evaluate(data, features: dict[str, tuple[np.ndarray, np.ndarray]]) -> dict:
    validation = data["validation"].astype(bool)
    result = {}
    for name, (train_x, val_x) in features.items():
        tasks = {}
        for target in ("lmr_research", "alpha_raise", "beta_cutoff"):
            y = data[f"y_{target}"]
            tasks[target] = classification(y[~validation], y[validation], train_x, val_x)
        for target in ("log_scout_nodes", "log_total_nodes", "score_gain"):
            y = data[f"y_{target}"]
            tasks[target] = regression(y[~validation], y[validation], train_x, val_x)
        result[name] = tasks
    return result


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("tensor", type=Path)
    parser.add_argument("--output", type=Path)
    parser.add_argument("--pca-components", type=int, default=32)
    args = parser.parse_args()
    data = np.load(args.tensor, allow_pickle=False)
    validation = data["validation"].astype(bool)
    context = data["context"].astype(np.float32)
    latent = data["latent"].astype(np.float32) / 255.0
    components = min(args.pca_components, latent[~validation].shape[0], latent.shape[1])
    pca = PCA(n_components=components, svd_solver="randomized", random_state=4101).fit(
        latent[~validation]
    )
    train_latent = pca.transform(latent[~validation])
    val_latent = pca.transform(latent[validation])
    features = {
        "context_only_histgb": (context[~validation], context[validation]),
        f"protected_latent_pca{components}_plus_context_histgb": (
            np.concatenate((train_latent, context[~validation]), axis=1),
            np.concatenate((val_latent, context[validation]), axis=1),
        ),
    }
    result = {
        "schema": "archon.search-lmr-baseline.v1",
        "data": json.loads(str(data["metadata"])),
        "split_counts": {"train": int((~validation).sum()), "validation": int(validation.sum())},
        "pca_explained_variance": float(pca.explained_variance_ratio_.sum()),
        "models": evaluate(data, features),
    }
    rendered = json.dumps(result, indent=2, sort_keys=True) + "\n"
    if args.output:
        args.output.write_text(rendered, encoding="utf-8")
    print(rendered, end="")


if __name__ == "__main__":
    main()
