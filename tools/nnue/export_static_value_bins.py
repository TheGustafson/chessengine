#!/usr/bin/env python3
"""Calibrate precisely named static-teacher fields and export verified Bullet bins."""

from __future__ import annotations

import argparse
import hashlib
import json
import os
from pathlib import Path
import shutil
import subprocess
import tempfile
from typing import Any

import numpy as np


NO_VALUE = np.int32(-(2**31))
RECORD_BYTES = 32


def sha256(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as handle:
        for block in iter(lambda: handle.read(8 * 1024 * 1024), b""):
            digest.update(block)
    return digest.hexdigest()


def fen_hash64(fen: str) -> int:
    return int.from_bytes(hashlib.blake2b(fen.encode(), digest_size=8).digest(), "little")


def parse_source(line: str) -> tuple[str, int, float]:
    parts = [part.strip() for part in line.rstrip("\n").split("|")]
    if len(parts) != 3:
        raise ValueError("invalid source row")
    return parts[0], int(parts[1]), float(parts[2])


def load_manifest(root: Path, chunks: int | None) -> tuple[dict[str, Any], list[dict[str, Any]]]:
    manifest = json.loads((root / "manifest.json").read_text())
    if manifest.get("schema") != "archon.paired-fen-static-values.v2":
        raise RuntimeError("input is not a semantics-explicit v2 paired static-value corpus")
    if not manifest.get("field_semantics"):
        raise RuntimeError("input is missing field semantics")
    entries = list(manifest["chunks"])
    if chunks is not None:
        entries = entries[:chunks]
    if not entries:
        raise RuntimeError("no input chunks")
    return manifest, entries


def fit_linear(entries: list[dict[str, Any]], root: Path, teacher: str, limit: int) -> dict[str, float | int]:
    xs: list[np.ndarray] = []
    ys: list[np.ndarray] = []
    for entry in entries:
        path = root / entry["path"]
        if sha256(path) != entry["sha256"]:
            raise RuntimeError(f"chunk SHA mismatch: {path}")
        with np.load(path, allow_pickle=False) as data:
            x = data[teacher].astype(np.float64)
            y = data["sf16_score_stm"].astype(np.float64)
        mask = (x != NO_VALUE) & (np.abs(x) < limit) & (np.abs(y) < limit)
        xs.append(x[mask]); ys.append(y[mask])
    x = np.concatenate(xs); y = np.concatenate(ys)
    design = np.column_stack([x, np.ones(len(x))])
    slope, intercept = np.linalg.lstsq(design, y, rcond=None)[0]
    prediction = slope * x + intercept
    return {
        "rows": int(len(x)), "slope": float(slope), "intercept": float(intercept),
        "correlation": float(np.corrcoef(x, y)[0, 1]),
        "calibrated_mae_cp": float(np.mean(np.abs(prediction - y))),
    }


def join(paths: list[Path], output: Path) -> str:
    with tempfile.NamedTemporaryFile(dir=output.parent, prefix=f".{output.name}.", delete=False) as f:
        staged = Path(f.name)
        for path in paths:
            with path.open("rb") as source:
                shutil.copyfileobj(source, f, 8 * 1024 * 1024)
        f.flush(); os.fsync(f.fileno())
    os.replace(staged, output)
    return sha256(output)


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("labels", type=Path)
    parser.add_argument("--output", type=Path, required=True)
    parser.add_argument("--chunks", type=int)
    parser.add_argument("--score-limit", type=int, default=2500)
    parser.add_argument("--val-per-mille", type=int, default=10)
    parser.add_argument("--target", choices=(
        "consensus", "stockfish18_final", "stockfish18_big",
        "reckless_static", "reckless_raw",
    ),
                        default="consensus")
    parser.add_argument("--converter", type=Path,
                        default=Path("tools/nnue/bf-convert/target/release/nnue_convert"))
    args = parser.parse_args()
    if not args.converter.is_file() or not 0 < args.score_limit <= 32767:
        parser.error("missing converter or invalid score limit")
    if not 0 <= args.val_per_mille < 1000:
        parser.error("invalid validation fraction")
    if args.output.exists() and any(args.output.iterdir()):
        parser.error("output must be absent or empty")

    manifest, entries = load_manifest(args.labels, args.chunks)
    target_fields = {
        "stockfish18_final": "stockfish18_final_uci_cp",
        "stockfish18_big": "stockfish18_big_sum_uci_cp",
        "reckless_static": "reckless_static_internal",
        "reckless_raw": "reckless_raw_nnue_units",
    }
    active_teachers = set(manifest.get("active_teachers", []))
    target_fields = {
        name: field for name, field in target_fields.items()
        if not name.startswith("reckless") or "reckless" in active_teachers
    }
    required_names = (
        {"stockfish18_final", "reckless_static"}
        if args.target == "consensus" else {args.target}
    )
    missing_targets = required_names.difference(target_fields)
    if missing_targets:
        parser.error(f"target requires absent teacher fields: {sorted(missing_targets)}")
    calibration = {
        name: fit_linear(entries, args.labels, field, args.score_limit)
        for name, field in target_fields.items()
    }
    args.output.mkdir(parents=True, exist_ok=True)
    shard_root = args.output / "shards"; shard_root.mkdir()
    records: list[dict[str, Any]] = []

    for ordinal, entry in enumerate(entries):
        chunk = args.labels / entry["path"]
        if sha256(chunk) != entry["sha256"]:
            raise RuntimeError(f"chunk SHA mismatch: {chunk}")
        with np.load(chunk, allow_pickle=False) as data:
            metadata = json.loads(data["metadata_json"].tobytes().decode())
            source_line = data["source_line"].copy()
            hashes = data["fen_hash64"].copy()
            results = data["result_stm"].copy()
            raw = {name: data[field].astype(np.float64) for name, field in target_fields.items()}
        calibrated = {
            name: calibration[name]["slope"] * value + calibration[name]["intercept"]
            for name, value in raw.items()
        }
        selected_names = (
            ("stockfish18_final", "reckless_static")
            if args.target == "consensus" else (args.target,)
        )
        valid = np.logical_and.reduce([raw[name] != NO_VALUE for name in selected_names])
        selected = np.mean([calibrated[name] for name in selected_names], axis=0)
        target = np.rint(selected).astype(np.int64)
        valid &= np.abs(target) < args.score_limit
        is_val = hashes.astype(np.uint64) % np.uint64(1000) < np.uint64(args.val_per_mille)

        source = Path(metadata["source"]["path"])
        if sha256(source) != metadata["source"]["sha256"]:
            raise RuntimeError(f"source SHA mismatch: {source}")
        start, stop = metadata["source_start_line"], metadata["source_stop_line"]
        outputs: dict[str, Path] = {}; counts: dict[str, int] = {}
        with tempfile.TemporaryDirectory(prefix=f"archon-static-export-{ordinal:06d}-") as temporary:
            temp = Path(temporary)
            handles = {
                split: (temp / f"{split}.txt").open("w", encoding="utf-8")
                for split in ("train", "val")
            }
            try:
                with source.open("r", encoding="utf-8") as rows:
                    for _ in range(start):
                        if not rows.readline(): raise RuntimeError("source ended before chunk")
                    for row in range(stop - start):
                        raw = rows.readline()
                        if not raw: raise RuntimeError("source ended within chunk")
                        fen, _, _ = parse_source(raw)
                        if int(source_line[row]) != start + row or int(hashes[row]) != fen_hash64(fen):
                            raise RuntimeError(f"source alignment failure: {chunk}:{row}")
                        if not valid[row]: continue
                        black = fen.split()[1] == "b"
                        score_white = -int(target[row]) if black else int(target[row])
                        result_stm = float(results[row]) / 2.0
                        result_white = 1.0 - result_stm if black else result_stm
                        split = "val" if is_val[row] else "train"
                        handles[split].write(f"{fen} | {score_white} | {result_white}\n")
            finally:
                for handle in handles.values(): handle.close()
            for split in ("train", "val"):
                text_path = temp / f"{split}.txt"
                count = sum(1 for _ in text_path.open("r", encoding="utf-8"))
                staged = temp / f"{split}.bin"
                subprocess.run([str(args.converter.resolve()), "convert", str(text_path), str(staged)], check=True)
                if staged.stat().st_size != count * RECORD_BYTES:
                    raise RuntimeError("converter record mismatch")
                output = shard_root / f"{ordinal:06d}-{split}.bin"
                os.replace(staged, output); outputs[split] = output; counts[split] = count
        records.append({
            "input": entry["path"], "input_sha256": entry["sha256"],
            "input_rows": len(source_line), "train_records": counts["train"],
            "val_records": counts["val"], "excluded_rows": int(len(valid) - valid.sum()),
            "train_sha256": sha256(outputs["train"]), "val_sha256": sha256(outputs["val"]),
        })
        print(json.dumps(records[-1], sort_keys=True), flush=True)

    train = [shard_root / f"{i:06d}-train.bin" for i in range(len(entries))]
    val = [shard_root / f"{i:06d}-val.bin" for i in range(len(entries))]
    result = {
        "schema": "archon.calibrated-static-value-bins.v2",
        "source": str(args.labels.resolve()), "source_manifest_sha256": sha256(args.labels / "manifest.json"),
        "teachers": manifest["teachers"], "field_semantics": manifest["field_semantics"],
        "calibration_target": "original SF16 side-to-move score; affine scale reference only",
        "calibration": calibration, "target": args.target,
        "target_rule": (
            "equal mean after affine calibration" if args.target == "consensus"
            else f"{args.target} after affine scale calibration"
        ),
        "score_limit": args.score_limit, "val_per_mille": args.val_per_mille,
        "result_field": "original game outcome", "records": records,
        "train_records": sum(x["train_records"] for x in records),
        "val_records": sum(x["val_records"] for x in records),
    }
    result["train_sha256"] = join(train, args.output / "train.bin")
    result["val_sha256"] = join(val, args.output / "val.bin")
    (args.output / "manifest.json").write_text(json.dumps(result, indent=2, sort_keys=True) + "\n")
    print(json.dumps({"schema": result["schema"], "train_records": result["train_records"],
                      "val_records": result["val_records"], "calibration": calibration}, indent=2))


if __name__ == "__main__":
    main()
