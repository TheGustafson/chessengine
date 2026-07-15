#!/usr/bin/env python3
"""Audit matched SF16/Reckless/SF18 labels in completed immutable chunks."""

from __future__ import annotations

import argparse
import hashlib
import json
import os
from pathlib import Path
import tempfile
from typing import Any

import numpy as np


NO_MATE = np.int16(32767)


def sha256(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as handle:
        for block in iter(lambda: handle.read(8 * 1024 * 1024), b""):
            digest.update(block)
    return digest.hexdigest()


def ranks(values: np.ndarray) -> np.ndarray:
    order = np.argsort(values, kind="stable")
    ranked = np.empty(len(values), dtype=np.float64)
    ordered = values[order]
    starts = np.r_[0, np.flatnonzero(ordered[1:] != ordered[:-1]) + 1]
    stops = np.r_[starts[1:], len(values)]
    for start, stop in zip(starts, stops, strict=True):
        ranked[order[start:stop]] = 0.5 * (start + stop - 1)
    return ranked


def comparison(reference: np.ndarray, predicted: np.ndarray) -> dict[str, Any]:
    reference = reference.astype(np.float64, copy=False)
    predicted = predicted.astype(np.float64, copy=False)
    residual = predicted - reference
    slope, intercept = np.polyfit(reference, predicted, 1)
    decisive = (np.abs(reference) >= 50) & (np.abs(predicted) >= 50)
    return {
        "rows": int(len(reference)),
        "pearson": float(np.corrcoef(reference, predicted)[0, 1]),
        "spearman": float(np.corrcoef(ranks(reference), ranks(predicted))[0, 1]),
        "predicted_on_reference_slope": float(slope),
        "predicted_on_reference_intercept": float(intercept),
        "mae_cp": float(np.mean(np.abs(residual))),
        "median_abs_error_cp": float(np.median(np.abs(residual))),
        "p90_abs_error_cp": float(np.quantile(np.abs(residual), 0.9)),
        "rmse_cp": float(np.sqrt(np.mean(residual * residual))),
        "reference_std_cp": float(np.std(reference)),
        "predicted_std_cp": float(np.std(predicted)),
        "sign_agreement": float(np.mean(np.sign(reference) == np.sign(predicted))),
        "decisive_rows": int(decisive.sum()),
        "decisive_sign_agreement": (
            float(np.mean(np.sign(reference[decisive]) == np.sign(predicted[decisive])))
            if decisive.any() else None
        ),
    }


def target_calibration(scores: np.ndarray, outcome: np.ndarray) -> dict[str, Any]:
    scores = scores.astype(np.float64, copy=False)
    scales = {}
    for scale in (100, 150, 200, 250, 300, 400, 500, 600, 800):
        probability = 1.0 / (1.0 + np.exp(-np.clip(scores / scale, -50, 50)))
        scales[str(scale)] = {
            "value_only_brier": float(np.mean((probability - outcome) ** 2)),
            "wdl03_blended_brier": float(
                np.mean((0.3 * outcome + 0.7 * probability - outcome) ** 2)
            ),
        }
    best = min(scales, key=lambda key: scales[key]["value_only_brier"])
    return {"scales": scales, "best_value_only_scale": int(best)}


def atomic_json(path: Path, value: Any) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    with tempfile.NamedTemporaryFile(
        mode="w", encoding="utf-8", prefix=path.name + ".",
        suffix=".tmp", dir=path.parent, delete=False,
    ) as handle:
        temporary = Path(handle.name)
        json.dump(value, handle, indent=2, sort_keys=True)
        handle.write("\n")
        handle.flush()
        os.fsync(handle.fileno())
    try:
        os.replace(temporary, path)
    finally:
        temporary.unlink(missing_ok=True)


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("data", type=Path)
    parser.add_argument("--chunks", type=int, help="audit only the first N completed chunks")
    parser.add_argument("--output", type=Path, required=True)
    args = parser.parse_args()
    manifest_path = args.data / "manifest.json"
    manifest_bytes = manifest_path.read_bytes()
    manifest = json.loads(manifest_bytes)
    chunks = list(manifest["chunks"])
    if args.chunks is not None:
        if args.chunks <= 0:
            parser.error("--chunks must be positive")
        chunks = chunks[:args.chunks]
    if not chunks:
        raise SystemExit("manifest contains no completed chunks")

    collected: dict[str, list[np.ndarray]] = {
        name: [] for name in (
            "sf16_score_stm", "result_stm",
            "reckless_score_cp", "reckless_mate_plies", "reckless_best_move",
            "stockfish_score_cp", "stockfish_mate_plies", "stockfish_best_move",
            "stockfish_wdl",
        )
    }
    chunk_records = []
    for chunk in chunks:
        path = args.data / chunk["path"]
        actual_sha = sha256(path)
        if actual_sha != chunk["sha256"]:
            raise RuntimeError(f"chunk SHA mismatch: {path}")
        with np.load(path, allow_pickle=False) as data:
            for name in collected:
                collected[name].append(data[name].copy())
            rows = len(data["source_line"])
            if rows != int(chunk["rows"]):
                raise RuntimeError(f"chunk row mismatch: {path}")
        chunk_records.append({"path": chunk["path"], "rows": rows, "sha256": actual_sha})
    values = {name: np.concatenate(parts) for name, parts in collected.items()}
    rows = len(values["result_stm"])
    outcome = values["result_stm"].astype(np.float64) / 2.0
    reckless_nonmate = values["reckless_mate_plies"] == NO_MATE
    stockfish_nonmate = values["stockfish_mate_plies"] == NO_MATE
    ordinary = reckless_nonmate & stockfish_nonmate

    sf16 = values["sf16_score_stm"]
    reckless = values["reckless_score_cp"]
    stockfish = values["stockfish_score_cp"]
    central = ordinary & (np.abs(reckless.astype(np.int32)) <= 1000) & (
        np.abs(stockfish.astype(np.int32)) <= 1000
    )
    sf_on_reckless_slope, sf_on_reckless_intercept = np.polyfit(
        reckless[central].astype(np.float64), stockfish[central].astype(np.float64), 1
    )
    wdl_valid = np.all(values["stockfish_wdl"] != np.uint16(65535), axis=1)
    stockfish_wdl_expectation = (
        values["stockfish_wdl"][:, 0].astype(np.float64)
        + 0.5 * values["stockfish_wdl"][:, 1].astype(np.float64)
    ) / 1000.0
    report = {
        "schema": "archon.paired-value-audit.v1",
        "data": str(args.data),
        "manifest_sha256": hashlib.sha256(manifest_bytes).hexdigest(),
        "config_hash": manifest["config_hash"],
        "chunks": chunk_records,
        "rows": rows,
        "mate_incidence": {
            "reckless_rows": int((~reckless_nonmate).sum()),
            "reckless_fraction": float(np.mean(~reckless_nonmate)),
            "stockfish_rows": int((~stockfish_nonmate).sum()),
            "stockfish_fraction": float(np.mean(~stockfish_nonmate)),
            "either_teacher_rows": int((~ordinary).sum()),
            "either_teacher_fraction": float(np.mean(~ordinary)),
        },
        "ordinary_cp_comparisons": {
            "sf18_vs_sf16": comparison(sf16[stockfish_nonmate], stockfish[stockfish_nonmate]),
            "reckless_vs_sf16": comparison(sf16[reckless_nonmate], reckless[reckless_nonmate]),
            "sf18_vs_reckless": comparison(reckless[ordinary], stockfish[ordinary]),
        },
        "best_move_agreement": {
            "rows": rows,
            "fraction": float(np.mean(
                values["reckless_best_move"] == values["stockfish_best_move"]
            )),
        },
        "outcome_calibration_nonmate": {
            "sf16": target_calibration(sf16.astype(np.float64), outcome),
            "reckless": target_calibration(reckless[reckless_nonmate], outcome[reckless_nonmate]),
            "stockfish18": target_calibration(stockfish[stockfish_nonmate], outcome[stockfish_nonmate]),
        },
        "stockfish_wdl": {
            "valid_rows": int(wdl_valid.sum()),
            "valid_fraction": float(np.mean(wdl_valid)),
            "expectation_outcome_brier": (
                float(np.mean((stockfish_wdl_expectation[wdl_valid] - outcome[wdl_valid]) ** 2))
                if wdl_valid.any() else None
            ),
        },
        "consensus_calibration_hint": {
            "central_rows": int(central.sum()),
            "sf18_on_reckless_slope": float(sf_on_reckless_slope),
            "sf18_on_reckless_intercept": float(sf_on_reckless_intercept),
            "note": "diagnostic only; freeze mapping on train split before constructing consensus",
        },
    }
    atomic_json(args.output, report)
    print(json.dumps(report, indent=2, sort_keys=True))


if __name__ == "__main__":
    main()
