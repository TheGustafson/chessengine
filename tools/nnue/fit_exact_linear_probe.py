#!/usr/bin/env python3
"""Independent regularized probes for protected exact-intervention signal."""

from __future__ import annotations

import argparse
import json
from pathlib import Path

import numpy as np
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import average_precision_score, brier_score_loss, log_loss, roc_auc_score
from sklearn.preprocessing import StandardScaler


def evaluate(y: np.ndarray, probability: np.ndarray) -> dict:
    return {
        "rows": len(y),
        "positives": int(y.sum()),
        "roc_auc": float(roc_auc_score(y, probability)),
        "average_precision": float(average_precision_score(y, probability)),
        "brier": float(brier_score_loss(y, probability)),
        "log_loss": float(log_loss(y, probability)),
    }


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("tensor", type=Path)
    parser.add_argument("--target", default="scout_wrong_window")
    parser.add_argument("--c-grid", default="0.0001,0.0003,0.001,0.003,0.01,0.03,0.1,0.3,1")
    parser.add_argument("--output", type=Path)
    parser.add_argument("--predictions", type=Path)
    args = parser.parse_args()
    data = np.load(args.tensor, allow_pickle=False)
    target_key = f"y_{args.target}"
    if target_key not in data:
        parser.error(f"tensor has no {target_key}")
    held = data["validation"].astype(bool)
    roots = data["root_ordinal"]
    calibration = (~held) & (roots % 10 == 1)
    fit = (~held) & (~calibration)
    y = data[target_key].astype(np.uint8)
    context = data["context_action"].astype(np.float32)
    latent = np.square(np.clip(data["latent"].astype(np.float32), 0.0, 255.0) / 255.0)
    feature_sets = {
        "action_only": context[:, -3:],
        "context_action": context,
        "protected_latent_full_plus_context_action": np.concatenate((latent, context), axis=1),
    }
    if "board" in data:
        board = data["board"].astype(np.float32)
        feature_sets["board_plus_context_action"] = np.concatenate((board, context), axis=1)
        feature_sets["board_plus_protected_latent_plus_context_action"] = np.concatenate(
            (board, latent, context), axis=1
        )
    c_grid = [float(value) for value in args.c_grid.split(",")]
    result = {
        "schema": "archon.exact-linear-probe.v1",
        "tensor": str(args.tensor.resolve()),
        "target": args.target,
        "split_rows": {"fit": int(fit.sum()), "calibration": int(calibration.sum()), "held": int(held.sum())},
        "selection": "C minimizes log loss on inner root-grouped calibration; refit non-held; final held untouched",
        "models": {},
    }
    predictions = {"root_ordinal": roots[held], f"y__{args.target}": y[held]}
    for name, features in feature_sets.items():
        scaler = StandardScaler().fit(features[fit])
        fit_x = scaler.transform(features[fit])
        calibration_x = scaler.transform(features[calibration])
        trials = []
        for c in c_grid:
            model = LogisticRegression(C=c, solver="lbfgs", max_iter=500, random_state=4101).fit(fit_x, y[fit])
            probability = model.predict_proba(calibration_x)[:, 1]
            trials.append({"C": c, "calibration_log_loss": float(log_loss(y[calibration], probability))})
        best = min(trials, key=lambda item: item["calibration_log_loss"])
        scaler = StandardScaler().fit(features[~held])
        model = LogisticRegression(C=best["C"], solver="lbfgs", max_iter=500, random_state=4101).fit(
            scaler.transform(features[~held]), y[~held]
        )
        probability = model.predict_proba(scaler.transform(features[held]))[:, 1]
        result["models"][name] = {
            "features": features.shape[1],
            "selected_C": best["C"],
            "calibration_trials": trials,
            "held": evaluate(y[held], probability),
        }
        predictions[f"{name}__{args.target}"] = probability
    rendered = json.dumps(result, indent=2, sort_keys=True) + "\n"
    if args.output:
        args.output.write_text(rendered, encoding="utf-8")
    if args.predictions:
        np.savez_compressed(args.predictions, **predictions)
    print(rendered, end="")


if __name__ == "__main__":
    main()
