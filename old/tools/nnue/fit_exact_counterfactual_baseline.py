#!/usr/bin/env python3
"""Fit baseline consequence models to exact Archon intervention labels."""

from __future__ import annotations

import argparse
import json
from pathlib import Path

import numpy as np
from sklearn.decomposition import PCA
from sklearn.ensemble import HistGradientBoostingClassifier, HistGradientBoostingRegressor
from sklearn.metrics import average_precision_score, brier_score_loss, mean_absolute_error, r2_score, roc_auc_score


def classify(train_x, train_y, val_x, val_y, max_iter: int) -> tuple[dict, np.ndarray | None]:
    result = {"rows": len(val_y), "positives": int(val_y.sum()), "positive_fraction": float(val_y.mean())}
    if len(np.unique(train_y)) < 2 or len(np.unique(val_y)) < 2:
        return {**result, "error": "both classes required"}, None
    model = HistGradientBoostingClassifier(
        max_iter=max_iter, learning_rate=0.08, max_leaf_nodes=31, min_samples_leaf=20,
        l2_regularization=1.0, random_state=4101,
    ).fit(train_x, train_y)
    probability = model.predict_proba(val_x)[:, 1]
    return {
        **result,
        "roc_auc": float(roc_auc_score(val_y, probability)),
        "average_precision": float(average_precision_score(val_y, probability)),
        "brier": float(brier_score_loss(val_y, probability)),
    }, probability


def regress(train_x, train_y, val_x, val_y, max_iter: int) -> dict:
    model = HistGradientBoostingRegressor(
        max_iter=max_iter, learning_rate=0.08, max_leaf_nodes=31, min_samples_leaf=20,
        l2_regularization=1.0, random_state=4101,
    ).fit(train_x, train_y)
    prediction = model.predict(val_x)
    return {
        "rows": len(val_y),
        "r2": float(r2_score(val_y, prediction)),
        "mae": float(mean_absolute_error(val_y, prediction)),
        "median_multiplicative_error": float(np.exp(np.median(np.abs(val_y - prediction)))),
    }


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("tensor", type=Path)
    parser.add_argument("--output", type=Path)
    parser.add_argument("--max-iter", type=int, default=80)
    parser.add_argument("--skip-regression", action="store_true")
    parser.add_argument("--targets", default="scout_wrong_window,wrong_window,root_best_changed,root_score_changed,local_research")
    parser.add_argument("--predictions", type=Path)
    args = parser.parse_args()
    data = np.load(args.tensor, allow_pickle=False)
    validation = data["validation"].astype(bool)
    context = data["context_action"].astype(np.float32)
    # Exact tensors retain the raw signed accumulator.  Net-0 exposes SCReLU,
    # so downstream heads consume clamp(a, 0, 255)^2 rather than raw values.
    latent = np.square(np.clip(data["latent"].astype(np.float32), 0.0, 255.0) / 255.0)
    components = min(32, int((~validation).sum()), latent.shape[1])
    pca = PCA(n_components=components, svd_solver="randomized", random_state=4101).fit(latent[~validation])
    train_latent = pca.transform(latent[~validation])
    val_latent = pca.transform(latent[validation])
    feature_sets = {
        "action_only": (context[~validation, -3:], context[validation, -3:]),
        "context_action": (context[~validation], context[validation]),
        f"protected_latent_pca{components}_plus_context_action": (
            np.concatenate((train_latent, context[~validation]), axis=1),
            np.concatenate((val_latent, context[validation]), axis=1),
        ),
    }
    models = {}
    prediction_arrays = {"root_ordinal": data["root_ordinal"][validation]}
    allowed_targets = {"scout_wrong_window", "wrong_window", "root_best_changed", "root_score_changed", "local_research"}
    classification_targets = args.targets.split(",")
    if not classification_targets or any(target not in allowed_targets for target in classification_targets):
        parser.error(f"--targets must be a comma-separated subset of {sorted(allowed_targets)}")
    for name, (train_x, val_x) in feature_sets.items():
        tasks = {}
        for target in classification_targets:
            y = data[f"y_{target}"]
            tasks[target], probability = classify(train_x, y[~validation], val_x, y[validation], args.max_iter)
            prediction_arrays[f"y__{target}"] = y[validation]
            if probability is not None:
                prediction_arrays[f"{name}__{target}"] = probability
        if not args.skip_regression:
            for target in (
                "log_root_nodes", "log_root_node_fraction", "signed_log_root_node_delta",
                "log_local_nodes", "log_absolute_root_score_error",
            ):
                y = data[f"y_{target}"]
                tasks[target] = regress(train_x, y[~validation], val_x, y[validation], args.max_iter)
        models[name] = tasks
    result = {
        "schema": "archon.search-counterfactual-exact-baseline.v1",
        "data": json.loads(str(data["metadata"])),
        "split_counts": {"train": int((~validation).sum()), "validation": int(validation.sum())},
        "pca_explained_variance": float(pca.explained_variance_ratio_.sum()),
        "max_iter": args.max_iter,
        "skip_regression": args.skip_regression,
        "classification_targets": classification_targets,
        "models": models,
    }
    rendered = json.dumps(result, indent=2, sort_keys=True) + "\n"
    if args.output:
        args.output.write_text(rendered, encoding="utf-8")
    if args.predictions:
        np.savez_compressed(args.predictions, **prediction_arrays)
    print(rendered, end="")


if __name__ == "__main__":
    main()
