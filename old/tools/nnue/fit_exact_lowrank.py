#!/usr/bin/env python3
"""Fit small move-conditioned heads to exact Archon interventions.

This is a research trainer, not a deployment format.  It deliberately keeps
the protected NNUE accumulator fixed and compares three otherwise-identical
models: action-only, context/action, and a low-rank interaction between the
protected accumulator and context/action.  Epoch selection uses only an inner
root-grouped calibration split; the tensor's validation roots stay held out.
"""

from __future__ import annotations

import argparse
import json
from pathlib import Path

import numpy as np
from sklearn.metrics import average_precision_score, mean_absolute_error, r2_score, roc_auc_score


CLASS_TARGETS = ("scout_wrong_window", "wrong_window", "root_best_changed", "root_score_changed", "local_research")
REG_TARGETS = (
    "log_root_nodes", "log_root_node_fraction", "signed_log_root_node_delta",
    "log_local_nodes", "log_absolute_root_score_error",
)
NCLASS = len(CLASS_TARGETS)
NREG = len(REG_TARGETS)
NOUT = NCLASS + NREG


def sigmoid(x: np.ndarray) -> np.ndarray:
    x = np.clip(x, -30.0, 30.0)
    return 1.0 / (1.0 + np.exp(-x))


def standardise(train: np.ndarray, *others: np.ndarray) -> tuple[np.ndarray, ...]:
    mean = train.mean(axis=0, keepdims=True)
    scale = train.std(axis=0, keepdims=True)
    scale[scale < 1e-5] = 1.0
    return tuple((value - mean) / scale for value in (train, *others))


class Adam:
    def __init__(self, params: dict[str, np.ndarray], learning_rate: float):
        self.params = params
        self.learning_rate = learning_rate
        self.m = {key: np.zeros_like(value) for key, value in params.items()}
        self.v = {key: np.zeros_like(value) for key, value in params.items()}
        self.step = 0

    def update(self, grads: dict[str, np.ndarray]) -> None:
        self.step += 1
        for key, grad in grads.items():
            self.m[key] = 0.9 * self.m[key] + 0.1 * grad
            self.v[key] = 0.999 * self.v[key] + 0.001 * grad * grad
            m = self.m[key] / (1.0 - 0.9**self.step)
            v = self.v[key] / (1.0 - 0.999**self.step)
            self.params[key] -= self.learning_rate * m / (np.sqrt(v) + 1e-8)


class Head:
    def __init__(self, latent_dim: int, context_dim: int, rank: int, hidden: int, seed: int, use_latent: bool):
        rng = np.random.default_rng(seed)
        self.use_latent = use_latent
        self.rank = rank
        scale = lambda fan: np.float32(np.sqrt(2.0 / fan))
        self.params: dict[str, np.ndarray] = {
            "wc": rng.normal(0, scale(context_dim), (context_dim, rank)).astype(np.float32),
            "bc": np.zeros(rank, np.float32),
        }
        zdim = rank + context_dim
        if use_latent:
            self.params.update(
                wp=rng.normal(0, scale(latent_dim), (latent_dim, rank)).astype(np.float32),
                bp=np.zeros(rank, np.float32),
            )
            zdim += 2 * rank
        self.params.update(
            wh=rng.normal(0, scale(zdim), (zdim, hidden)).astype(np.float32),
            bh=np.zeros(hidden, np.float32),
            wo=rng.normal(0, scale(hidden), (hidden, NOUT)).astype(np.float32),
            bo=np.zeros(NOUT, np.float32),
        )

    def forward(self, latent: np.ndarray, context: np.ndarray) -> tuple[np.ndarray, tuple]:
        p = None
        c = np.tanh(context @ self.params["wc"] + self.params["bc"])
        parts = [c, context]
        if self.use_latent:
            p = np.tanh(latent @ self.params["wp"] + self.params["bp"])
            parts = [p, c, p * c, context]
        z = np.concatenate(parts, axis=1)
        h = np.tanh(z @ self.params["wh"] + self.params["bh"])
        out = h @ self.params["wo"] + self.params["bo"]
        return out, (latent, context, p, c, z, h)

    def gradients(self, cache: tuple, dout: np.ndarray, weight_decay: float) -> dict[str, np.ndarray]:
        latent, context, p, c, z, h = cache
        grads = {
            "wo": h.T @ dout + weight_decay * self.params["wo"],
            "bo": dout.sum(axis=0),
        }
        dh = (dout @ self.params["wo"].T) * (1.0 - h * h)
        grads["wh"] = z.T @ dh + weight_decay * self.params["wh"]
        grads["bh"] = dh.sum(axis=0)
        dz = dh @ self.params["wh"].T
        if self.use_latent:
            dp = dz[:, : self.rank] + dz[:, 2 * self.rank : 3 * self.rank] * c
            dc = dz[:, self.rank : 2 * self.rank] + dz[:, 2 * self.rank : 3 * self.rank] * p
            dp *= 1.0 - p * p
            grads["wp"] = latent.T @ dp + weight_decay * self.params["wp"]
            grads["bp"] = dp.sum(axis=0)
        else:
            dc = dz[:, : self.rank]
        dc *= 1.0 - c * c
        grads["wc"] = context.T @ dc + weight_decay * self.params["wc"]
        grads["bc"] = dc.sum(axis=0)
        return grads


class ProtectedResidual:
    """Position-conditioned residual on a frozen context consequence head."""

    def __init__(self, base: Head, latent_dim: int, context_dim: int, rank: int, seed: int):
        rng = np.random.default_rng(seed ^ 0xA4C04101)
        self.base = base
        self.params = {
            "wp": rng.normal(0, np.sqrt(2.0 / latent_dim), (latent_dim, rank)).astype(np.float32),
            "bp": np.zeros(rank, np.float32),
            "wc": rng.normal(0, np.sqrt(2.0 / context_dim), (context_dim, rank)).astype(np.float32),
            "bc": np.zeros(rank, np.float32),
            # Zero initialization makes the first prediction bit-identical to
            # the protected context model while still allowing gradients into
            # the output interaction.  Position/context projections begin
            # learning on the following update.
            "wo": np.zeros((rank, NOUT), np.float32),
        }

    def forward(self, latent: np.ndarray, context: np.ndarray) -> tuple[np.ndarray, tuple]:
        base_output, _ = self.base.forward(latent, context)
        p = np.tanh(latent @ self.params["wp"] + self.params["bp"])
        c = np.tanh(context @ self.params["wc"] + self.params["bc"])
        interaction = p * c
        return base_output + interaction @ self.params["wo"], (latent, context, p, c, interaction)

    def gradients(self, cache: tuple, dout: np.ndarray, weight_decay: float) -> dict[str, np.ndarray]:
        latent, context, p, c, interaction = cache
        grads = {"wo": interaction.T @ dout + weight_decay * self.params["wo"]}
        di = dout @ self.params["wo"].T
        dp = di * c * (1.0 - p * p)
        dc = di * p * (1.0 - c * c)
        grads.update(
            wp=latent.T @ dp + weight_decay * self.params["wp"],
            bp=dp.sum(axis=0),
            wc=context.T @ dc + weight_decay * self.params["wc"],
            bc=dc.sum(axis=0),
        )
        return grads


def objective(
    output: np.ndarray, target: np.ndarray, reg_weight: float,
    class_indices: list[int], reg_indices: list[int],
) -> tuple[float, np.ndarray]:
    probability = sigmoid(output[:, :NCLASS])
    eps = 1e-7
    classification = 0.0
    regression = 0.0
    dout = np.zeros_like(output)
    if class_indices:
        p = probability[:, class_indices]
        y = target[:, class_indices]
        classification = -np.mean(y * np.log(p + eps) + (1.0 - y) * np.log(1.0 - p + eps))
        dout[:, class_indices] = (p - y) / (output.shape[0] * len(class_indices))
    if reg_indices:
        columns = [NCLASS + index for index in reg_indices]
        regression_error = output[:, columns] - target[:, columns]
        regression = np.mean(regression_error * regression_error)
        dout[:, columns] = reg_weight * 2.0 * regression_error / (output.shape[0] * len(reg_indices))
    return float(classification + reg_weight * regression), dout


def metrics(output: np.ndarray, target: np.ndarray, reg_mean: np.ndarray, reg_scale: np.ndarray) -> dict:
    result: dict[str, dict] = {}
    probability = sigmoid(output[:, :NCLASS])
    for index, name in enumerate(CLASS_TARGETS):
        y = target[:, index]
        item = {"rows": len(y), "positives": int(y.sum()), "positive_fraction": float(y.mean())}
        if len(np.unique(y)) == 2:
            p = probability[:, index]
            ece = 0.0
            for lower in np.linspace(0.0, 0.9, 10):
                member = (p >= lower) & (p < lower + 0.1 if lower < 0.9 else p <= 1.0)
                if member.any():
                    ece += member.mean() * abs(float(p[member].mean() - y[member].mean()))
            item.update(
                roc_auc=float(roc_auc_score(y, p)),
                average_precision=float(average_precision_score(y, p)),
                brier=float(np.mean(np.square(p - y))),
                expected_calibration_error_10bin=float(ece),
            )
        result[name] = item
    truth = target[:, NCLASS:] * reg_scale + reg_mean
    prediction = output[:, NCLASS:] * reg_scale + reg_mean
    for index, name in enumerate(REG_TARGETS):
        result[name] = {
            "r2": float(r2_score(truth[:, index], prediction[:, index])),
            "mae": float(mean_absolute_error(truth[:, index], prediction[:, index])),
        }
    return result


def platt_calibrate(calibration_output: np.ndarray, calibration_target: np.ndarray, held_output: np.ndarray):
    """Fit one conservative affine-logit calibrator per classification task."""
    adjusted = held_output.copy()
    parameters = []
    prior = np.asarray([1.0, 0.0], dtype=np.float64)
    for column in range(NCLASS):
        y = calibration_target[:, column].astype(np.float64)
        x = np.column_stack((calibration_output[:, column], np.ones(len(y))))
        weight = prior.copy()
        if len(np.unique(y)) == 2:
            for _ in range(50):
                probability = sigmoid(x @ weight).astype(np.float64)
                curvature = np.maximum(probability * (1.0 - probability), 1e-6)
                gradient = x.T @ (probability - y) + (weight - prior)
                hessian = x.T @ (x * curvature[:, None]) + np.eye(2)
                step = np.linalg.solve(hessian, gradient)
                weight -= step
                if np.max(np.abs(step)) < 1e-7:
                    break
        adjusted[:, column] = held_output[:, column] * weight[0] + weight[1]
        parameters.append({"slope": float(weight[0]), "intercept": float(weight[1])})
    return adjusted, parameters


def train_one(
    latent: np.ndarray,
    context: np.ndarray,
    target: np.ndarray,
    fit: np.ndarray,
    calibration: np.ndarray,
    held: np.ndarray,
    mode: str,
    seed: int,
    args: argparse.Namespace,
    reg_mean: np.ndarray,
    reg_scale: np.ndarray,
) -> tuple[dict, np.ndarray, np.ndarray]:
    if mode == "action_only":
        context = context[:, -3:]
    def fit_head(head, run_seed: int):
        optimiser = Adam(head.params, args.learning_rate)
        rng = np.random.default_rng(run_seed)
        best_loss = float("inf")
        best_epoch = 0
        best_params = None
        stale = 0
        for epoch in range(1, args.epochs + 1):
            order = rng.permutation(np.flatnonzero(fit))
            for start in range(0, len(order), args.batch_size):
                batch = order[start : start + args.batch_size]
                output, cache = head.forward(latent[batch], context[batch])
                _, dout = objective(
                    output, target[batch], args.regression_weight,
                    args.class_loss_indices, args.reg_loss_indices,
                )
                optimiser.update(head.gradients(cache, dout, args.weight_decay))
            calibration_output, _ = head.forward(latent[calibration], context[calibration])
            calibration_loss, _ = objective(
                calibration_output, target[calibration], args.regression_weight,
                args.class_loss_indices, args.reg_loss_indices,
            )
            if calibration_loss < best_loss - 1e-5:
                best_loss = calibration_loss
                best_epoch = epoch
                best_params = {key: value.copy() for key, value in head.params.items()}
                stale = 0
            else:
                stale += 1
            if stale >= args.patience:
                break
        assert best_params is not None
        head.params = best_params
        return head, best_epoch, best_loss

    if mode == "residual":
        base = Head(latent.shape[1], context.shape[1], args.rank, args.hidden, seed, False)
        base, base_epoch, _ = fit_head(base, seed)
        head = ProtectedResidual(base, latent.shape[1], context.shape[1], args.rank, seed)
        head, best_epoch, best_loss = fit_head(head, seed ^ 0x51D0)
    else:
        use_latent = mode == "lowrank"
        head = Head(latent.shape[1], context.shape[1], args.rank, args.hidden, seed, use_latent)
        head, best_epoch, best_loss = fit_head(head, seed)
        base_epoch = None
    held_output, _ = head.forward(latent[held], context[held])
    calibration_output, _ = head.forward(latent[calibration], context[calibration])
    return {
        "seed": seed,
        "best_epoch": best_epoch,
        "base_epoch": base_epoch,
        "calibration_loss": best_loss,
        "held": metrics(held_output, target[held], reg_mean, reg_scale),
    }, held_output, calibration_output


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("tensor", type=Path)
    parser.add_argument("--output", type=Path)
    parser.add_argument("--predictions", type=Path)
    parser.add_argument("--rank", type=int, default=16)
    parser.add_argument("--hidden", type=int, default=32)
    parser.add_argument("--epochs", type=int, default=250)
    parser.add_argument("--patience", type=int, default=25)
    parser.add_argument("--batch-size", type=int, default=256)
    parser.add_argument("--learning-rate", type=float, default=2e-3)
    parser.add_argument("--weight-decay", type=float, default=1e-4)
    parser.add_argument("--regression-weight", type=float, default=0.25)
    parser.add_argument("--seeds", default="4101,4102,4103")
    parser.add_argument("--modes", default="action_only,context_action,lowrank,residual")
    parser.add_argument("--latent-activation", choices=("screlu", "crelu", "raw"), default="screlu")
    parser.add_argument("--position-input", choices=("latent", "board", "latent_board"), default="latent")
    parser.add_argument("--structured-move", action="store_true")
    parser.add_argument("--loss-targets", default=",".join(CLASS_TARGETS + REG_TARGETS))
    args = parser.parse_args()
    loss_targets = args.loss_targets.split(",")
    allowed_targets = set(CLASS_TARGETS + REG_TARGETS)
    if not loss_targets or any(target not in allowed_targets for target in loss_targets):
        parser.error(f"--loss-targets must be a comma-separated subset of {sorted(allowed_targets)}")
    args.class_loss_indices = [CLASS_TARGETS.index(target) for target in loss_targets if target in CLASS_TARGETS]
    args.reg_loss_indices = [REG_TARGETS.index(target) for target in loss_targets if target in REG_TARGETS]

    data = np.load(args.tensor, allow_pickle=False)
    held = data["validation"].astype(bool)
    roots = data["root_ordinal"].astype(np.uint32)
    calibration = (~held) & (roots % 10 == 1)
    fit = (~held) & (~calibration)
    if min(fit.sum(), calibration.sum(), held.sum()) == 0:
        raise ValueError("fit/calibration/held root-grouped split is empty")

    latent = data["latent"].astype(np.float32)
    if args.latent_activation == "screlu":
        latent = np.square(np.clip(latent, 0.0, 255.0) / 255.0)
    elif args.latent_activation == "crelu":
        latent = np.clip(latent, 0.0, 255.0) / 255.0
    else:
        latent /= 255.0
    if args.position_input != "latent":
        if "board" not in data:
            parser.error("--position-input board/latent_board requires a tensor with board features")
        board = data["board"].astype(np.float32)
        latent = board if args.position_input == "board" else np.concatenate((latent, board), axis=1)
    context = data["context_action"].astype(np.float32)
    if args.structured_move:
        if "structured_move_action" not in data:
            parser.error("--structured-move requires structured_move_action tensor features")
        context = np.concatenate((context, data["structured_move_action"].astype(np.float32)), axis=1)
    latent[fit], latent[calibration], latent[held] = standardise(latent[fit], latent[calibration], latent[held])
    context[fit], context[calibration], context[held] = standardise(context[fit], context[calibration], context[held])
    classification = np.column_stack([data[f"y_{name}"] for name in CLASS_TARGETS]).astype(np.float32)
    regression = np.column_stack([data[f"y_{name}"] for name in REG_TARGETS]).astype(np.float32)
    reg_mean = regression[fit].mean(axis=0)
    reg_scale = regression[fit].std(axis=0)
    reg_scale[reg_scale < 1e-5] = 1.0
    target = np.concatenate((classification, (regression - reg_mean) / reg_scale), axis=1)

    seeds = [int(value) for value in args.seeds.split(",")]
    result = {
        "schema": "archon.exact-lowrank-fit.v1",
        "tensor": str(args.tensor.resolve()),
        "split_rows": {"fit": int(fit.sum()), "calibration": int(calibration.sum()), "held": int(held.sum())},
        "split_contract": "outer held from tensor; inner calibration iff non-held root ordinal % 10 == 1",
        "configuration": {key: getattr(args, key) for key in ("rank", "hidden", "epochs", "patience", "batch_size", "learning_rate", "weight_decay", "regression_weight")},
        "latent_activation": args.latent_activation,
        "position_input": args.position_input,
        "structured_move": args.structured_move,
        "loss_targets": loss_targets,
        "models": {},
    }
    prediction_arrays = {"root_ordinal": roots[held], "target": target[held]}
    modes = args.modes.split(",")
    allowed_modes = {"action_only", "context_action", "lowrank", "residual"}
    if not modes or any(mode not in allowed_modes for mode in modes):
        parser.error(f"--modes must be a comma-separated subset of {sorted(allowed_modes)}")
    for mode in modes:
        runs = []
        outputs = []
        calibration_outputs = []
        for seed in seeds:
            run, output, calibration_output = train_one(
                latent, context, target, fit, calibration, held, mode, seed, args, reg_mean, reg_scale
            )
            runs.append(run)
            outputs.append(output)
            calibration_outputs.append(calibration_output)
        mean_output = np.mean(outputs, axis=0)
        mean_calibration_output = np.mean(calibration_outputs, axis=0)
        calibrated_output, calibration_parameters = platt_calibrate(
            mean_calibration_output, target[calibration], mean_output
        )
        result["models"][mode] = {
            "runs": runs,
            "ensemble_held_raw": metrics(mean_output, target[held], reg_mean, reg_scale),
            "ensemble_held": metrics(calibrated_output, target[held], reg_mean, reg_scale),
            "platt_calibration": calibration_parameters,
        }
        prediction_arrays[mode] = calibrated_output

    rendered = json.dumps(result, indent=2, sort_keys=True) + "\n"
    if args.output:
        args.output.write_text(rendered, encoding="utf-8")
    if args.predictions:
        np.savez_compressed(args.predictions, **prediction_arrays)
    print(rendered, end="")


if __name__ == "__main__":
    main()
