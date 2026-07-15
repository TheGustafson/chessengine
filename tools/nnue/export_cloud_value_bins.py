#!/usr/bin/env python3
"""Build aligned Bullet bins from downloaded cloud position/teacher shards.

The cloud LC0 corpus deliberately retains only mover-normalized piece state.
This exporter joins that durable position state with paired Reckless/SF18
labels, forms a scale-calibrated score consensus, and uses bulletformat's own
converter for the 32-byte record layout.  Game results are unavailable for
these positions, so records carry a neutral result and consumers must train
with WDL=0 (or supply a separate outcome target in a future format).
"""

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


NO_MATE = np.int16(32767)
RECORD_BYTES = 32
PIECES = "PNBRQKpnbrqk"


def sha256(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as handle:
        for block in iter(lambda: handle.read(8 * 1024 * 1024), b""):
            digest.update(block)
    return digest.hexdigest()


def atomic_json(path: Path, value: Any) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    with tempfile.NamedTemporaryFile("w", dir=path.parent, prefix=f".{path.name}.", delete=False) as f:
        temporary = Path(f.name)
        json.dump(value, f, indent=2, sort_keys=True)
        f.write("\n")
        f.flush()
        os.fsync(f.fileno())
    os.replace(temporary, path)


def parse_ordinals(value: str) -> list[int]:
    result: set[int] = set()
    for item in value.split(","):
        item = item.strip()
        if not item:
            continue
        if "-" in item:
            first, last = (int(x) for x in item.split("-", 1))
            if first > last:
                raise argparse.ArgumentTypeError(f"reversed ordinal range: {item}")
            result.update(range(first, last + 1))
        else:
            result.add(int(item))
    if not result or min(result) < 0:
        raise argparse.ArgumentTypeError("ordinals must be non-negative")
    return sorted(result)


def fen_from_features(row: np.ndarray) -> str:
    board: list[str | None] = [None] * 64
    for raw in row:
        feature = int(raw)
        if feature < 0:
            break
        colour, remainder = divmod(feature, 384)
        piece_type, square = divmod(remainder, 64)
        if colour not in (0, 1) or piece_type >= 6 or board[square] is not None:
            raise ValueError(f"invalid position feature {feature}")
        board[square] = PIECES[colour * 6 + piece_type]
    ranks: list[str] = []
    for rank in range(7, -1, -1):
        encoded = ""
        empty = 0
        for file in range(8):
            piece = board[rank * 8 + file]
            if piece is None:
                empty += 1
            else:
                if empty:
                    encoded += str(empty)
                    empty = 0
                encoded += piece
        if empty:
            encoded += str(empty)
        ranks.append(encoded)
    return "/".join(ranks) + " w - - 0 1"


def join_bins(paths: list[Path], destination: Path) -> str:
    with tempfile.NamedTemporaryFile(dir=destination.parent, prefix=f".{destination.name}.", delete=False) as out:
        temporary = Path(out.name)
        for path in paths:
            with path.open("rb") as source:
                shutil.copyfileobj(source, out, length=8 * 1024 * 1024)
        out.flush()
        os.fsync(out.fileno())
    os.replace(temporary, destination)
    return sha256(destination)


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--positions", type=Path, required=True,
                        help="directory containing <ordinal>.npz position shards")
    parser.add_argument("--labels", type=Path, required=True,
                        help="directory containing aligned <ordinal>.npz label shards")
    parser.add_argument("--position-manifest", type=Path, required=True)
    parser.add_argument("--output", type=Path, required=True)
    parser.add_argument("--ordinals", type=parse_ordinals, required=True)
    parser.add_argument("--converter", type=Path,
                        default=Path("tools/nnue/bf-convert/target/release/nnue_convert"))
    parser.add_argument("--sf18-factor", type=float, default=1.514839773)
    parser.add_argument("--reckless-factor", type=float, default=1.663689293)
    parser.add_argument("--score-limit", type=int, default=2500)
    parser.add_argument("--val-per-mille", type=int, default=10)
    parser.add_argument(
        "--allow-bounds",
        action="store_true",
        help="retain root lower/upper bounds as scalar targets (unsafe by default)",
    )
    args = parser.parse_args()
    if not args.converter.is_file():
        parser.error(f"missing converter: {args.converter}")
    if args.sf18_factor <= 0 or args.reckless_factor <= 0:
        parser.error("teacher factors must be positive")
    if not 0 < args.score_limit <= 32767 or not 0 <= args.val_per_mille < 1000:
        parser.error("invalid score-limit or val-per-mille")

    source_manifest = json.loads(args.position_manifest.read_text())
    expected = {int(item["ordinal"]): item for item in source_manifest["shards"]}
    missing = set(args.ordinals).difference(expected)
    if missing:
        parser.error(f"ordinals absent from position manifest: {sorted(missing)[:10]}")
    args.output.mkdir(parents=True, exist_ok=True)
    shard_root = args.output / "shards"
    shard_root.mkdir(exist_ok=True)
    records: list[dict[str, Any]] = []

    for ordinal in args.ordinals:
        position_path = args.positions / f"{ordinal:06d}.npz"
        label_path = args.labels / f"{ordinal:06d}.npz"
        if not position_path.is_file() or not label_path.is_file():
            raise SystemExit(f"missing aligned local shard {ordinal:06d}")
        position_hash = sha256(position_path)
        if position_hash != expected[ordinal]["sha256"]:
            raise RuntimeError(f"position hash mismatch for ordinal {ordinal}")
        label_hash = sha256(label_path)
        with np.load(position_path, allow_pickle=False) as position_data, np.load(
            label_path, allow_pickle=False
        ) as label_data:
            features = position_data["features"]
            position_index = position_data["source_index"]
            label_index = label_data["source_index"]
            metadata = json.loads(label_data["metadata_json"].tobytes().decode())
            if metadata["input_sha256"] != position_hash:
                raise RuntimeError(f"label input hash mismatch for ordinal {ordinal}")
            if not np.array_equal(position_index, label_index):
                raise RuntimeError(f"source-index mismatch for ordinal {ordinal}")
            reckless = label_data["reckless_score_cp"].astype(np.float64)
            stockfish = label_data["stockfish_score_cp"].astype(np.float64)
            nonmate = (label_data["reckless_mate_plies"] == NO_MATE) & (
                label_data["stockfish_mate_plies"] == NO_MATE
            )
            exact = (label_data["reckless_bound"] == 0) & (
                label_data["stockfish_bound"] == 0
            )
            consensus = np.rint(
                0.5 * (args.reckless_factor * reckless + args.sf18_factor * stockfish)
            ).astype(np.int64)
            retained = nonmate & (np.abs(consensus) < args.score_limit)
            if not args.allow_bounds:
                retained &= exact
            # SplitMix-style deterministic mixing of global corpus ordinal.
            # source_index is local to the original source shard and is not a
            # corpus-global row number.  Use the position-manifest offset plus
            # the verified row order so every exported position gets one stable,
            # globally unique split key.
            global_index = (
                np.arange(len(features), dtype=np.uint64)
                + np.uint64(expected[ordinal]["global_start"])
            )
            mixed = global_index * np.uint64(0x9E3779B97F4A7C15)
            is_val = mixed % np.uint64(1000) < np.uint64(args.val_per_mille)

            outputs: dict[str, Path] = {}
            counts: dict[str, int] = {}
            with tempfile.TemporaryDirectory(prefix=f"archon-cloud-bin-{ordinal:06d}-") as temporary:
                temporary_root = Path(temporary)
                for split, mask in (("train", retained & ~is_val), ("val", retained & is_val)):
                    text_path = temporary_root / f"{split}.txt"
                    indices = np.flatnonzero(mask)
                    with text_path.open("w", encoding="utf-8") as stream:
                        for row in indices:
                            stream.write(
                                f"{fen_from_features(features[row])} | {int(consensus[row])} | 0.5\n"
                            )
                    output = shard_root / f"{ordinal:06d}-{split}.bin"
                    staged = temporary_root / f"{split}.bin"
                    subprocess.run(
                        [str(args.converter.resolve()), "convert", str(text_path), str(staged)],
                        check=True,
                    )
                    if staged.stat().st_size != len(indices) * RECORD_BYTES:
                        raise RuntimeError(f"converted record count mismatch for {ordinal} {split}")
                    os.replace(staged, output)
                    outputs[split] = output
                    counts[split] = len(indices)

        record = {
            "ordinal": ordinal,
            "position": str(position_path.resolve()),
            "position_sha256": position_hash,
            "labels": str(label_path.resolve()),
            "labels_sha256": label_hash,
            "input_rows": int(len(position_index)),
            "joint_exact_rows": int(exact.sum()),
            "excluded_rows": int(len(position_index) - retained.sum()),
            "train_records": counts["train"],
            "val_records": counts["val"],
            "train_sha256": sha256(outputs["train"]),
            "val_sha256": sha256(outputs["val"]),
        }
        records.append(record)
        print(json.dumps(record, sort_keys=True), flush=True)

    train_paths = [shard_root / f"{ordinal:06d}-train.bin" for ordinal in args.ordinals]
    val_paths = [shard_root / f"{ordinal:06d}-val.bin" for ordinal in args.ordinals]
    train_hash = join_bins(train_paths, args.output / "train.bin")
    val_hash = join_bins(val_paths, args.output / "val.bin")
    manifest = {
        "schema": "archon.cloud-paired-value-bins.v1",
        "state_completeness": "piece_state_bootstrap_only",
        "result_field": "neutral draw placeholder; train with WDL=0",
        "target": "equal SF18/Reckless consensus after scale calibration to net-0/SF16",
        "sf18_factor": args.sf18_factor,
        "reckless_factor": args.reckless_factor,
        "score_limit": args.score_limit,
        "allow_bounds": args.allow_bounds,
        "val_per_mille": args.val_per_mille,
        "source_manifest": str(args.position_manifest.resolve()),
        "source_manifest_sha256": sha256(args.position_manifest),
        "ordinals": args.ordinals,
        "shards": records,
        "train_records": sum(item["train_records"] for item in records),
        "val_records": sum(item["val_records"] for item in records),
        "train_sha256": train_hash,
        "val_sha256": val_hash,
    }
    atomic_json(args.output / "manifest.json", manifest)
    print(json.dumps({k: manifest[k] for k in ("schema", "train_records", "val_records")}, indent=2))


if __name__ == "__main__":
    main()
