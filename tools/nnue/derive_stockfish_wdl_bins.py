#!/usr/bin/env python3
"""Derive matched Bullet corpora from Stockfish's WDL expectation.

The proven value trainer predicts a sigmoid-domain expected outcome.  Modern
Stockfish UCI centipawns are not on the historical SF16 scale, but its WDL
triple directly supplies the relevant target: ``p = (W + 0.5 D) / 1000``.
This tool replaces only each 32-byte Bullet record's i16 score with
``round(400 * logit(p))``; board state and terminal game result remain exact
copies of the already-audited matched Stockfish corpus.
"""

from __future__ import annotations

import argparse
import hashlib
import json
from pathlib import Path
import shutil

import numpy as np


RECORD_BYTES = 32
SCORE_OFFSET = 24


def sha256(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as stream:
        for block in iter(lambda: stream.read(8 * 1024 * 1024), b""):
            digest.update(block)
    return digest.hexdigest()


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("labels", type=Path)
    parser.add_argument("matched", type=Path)
    parser.add_argument("output", type=Path)
    parser.add_argument("--chunks", type=int)
    parser.add_argument("--val-per-mille", type=int, default=100)
    parser.add_argument("--score-limit", type=int, default=2500)
    parser.add_argument("--output-clamp", type=int, default=32000)
    parser.add_argument("--scale", type=float, default=400.0)
    parser.add_argument(
        "--probability-floor",
        type=float,
        default=0.0005,
        help="midpoint bound for integer-per-mille WDL values (default: 0.5/1000)",
    )
    args = parser.parse_args()
    if args.output.exists() and any(args.output.iterdir()):
        parser.error(f"output must be absent or empty: {args.output}")

    source_manifest = json.loads((args.labels / "manifest.json").read_text())
    chunks = list(source_manifest["chunks"])
    if args.chunks is not None:
        chunks = chunks[: args.chunks]
    expected_manifest = json.loads((args.matched / "manifest.json").read_text())
    if len(expected_manifest["exported"]) != len(chunks):
        raise RuntimeError("matched export and selected label chunk counts differ")

    by_split: dict[str, list[np.ndarray]] = {"train": [], "val": []}
    clipped = {"train": 0, "val": 0}
    for ordinal, (entry, exported) in enumerate(
        zip(chunks, expected_manifest["exported"], strict=True)
    ):
        if entry["path"] != exported["input"]:
            raise RuntimeError(f"chunk order mismatch at ordinal {ordinal}")
        path = args.labels / entry["path"]
        if sha256(path) != entry["sha256"]:
            raise RuntimeError(f"label chunk hash mismatch: {path}")
        with np.load(path, allow_pickle=False) as values:
            scores = np.column_stack(
                [
                    values["sf16_score_stm"],
                    values["reckless_score_cp"],
                    values["stockfish_score_cp"],
                ]
            ).astype(np.int64)
            retained = np.all(np.abs(scores) < args.score_limit, axis=1)
            is_val = (
                values["fen_hash64"].astype(np.uint64) % np.uint64(1000)
            ) < args.val_per_mille
            wdl = values["stockfish_wdl"].astype(np.float64)
            if np.any(wdl == 65535) or np.any(wdl.sum(axis=1) != 1000):
                raise RuntimeError(f"invalid/missing Stockfish WDL in {path}")
            probability = (wdl[:, 0] + 0.5 * wdl[:, 1]) / 1000.0
            probability = np.clip(
                probability,
                args.probability_floor,
                1.0 - args.probability_floor,
            )
            raw = np.rint(args.scale * np.log(probability / (1.0 - probability)))
            derived = np.clip(raw, -args.output_clamp, args.output_clamp).astype("<i2")
            for split, selection in (
                ("train", retained & ~is_val),
                ("val", retained & is_val),
            ):
                selected_raw = raw[selection]
                clipped[split] += int(np.count_nonzero(np.abs(selected_raw) > args.output_clamp))
                part = derived[selection].copy()
                if len(part) != int(exported[split]):
                    raise RuntimeError(f"row mismatch in {entry['path']} {split}")
                by_split[split].append(part)

    args.output.mkdir(parents=True, exist_ok=True)
    outputs = {}
    for split in ("train", "val"):
        source = args.matched / "stockfish18" / f"{split}.bin"
        destination = args.output / f"{split}.bin"
        shutil.copyfile(source, destination)
        data = np.memmap(destination, mode="r+", dtype=np.uint8)
        record_scores = np.ndarray(
            shape=(len(data) // RECORD_BYTES,),
            dtype="<i2",
            buffer=data,
            offset=SCORE_OFFSET,
            strides=(RECORD_BYTES,),
        )
        derived = np.concatenate(by_split[split])
        if len(record_scores) != len(derived):
            raise RuntimeError(f"binary/derived count mismatch for {split}")
        record_scores[:] = derived
        data.flush()
        del record_scores, data
        outputs[split] = {
            "source": str(source.resolve()),
            "source_sha256": sha256(source),
            "output": str(destination.resolve()),
            "output_sha256": sha256(destination),
            "rows": int(len(derived)),
            "score_min": int(derived.min()),
            "score_max": int(derived.max()),
            "clipped": clipped[split],
        }

    provenance = {
        "schema": "archon.stockfish-wdl-bullet-corpus.v1",
        "labels": str(args.labels.resolve()),
        "label_config_hash": source_manifest["config_hash"],
        "matched_manifest_sha256": sha256(args.matched / "manifest.json"),
        "chunks": len(chunks),
        "formula": "round(scale * logit((wins + 0.5*draws)/1000))",
        "scale": args.scale,
        "probability_floor": args.probability_floor,
        "score_limit": args.score_limit,
        "output_clamp": args.output_clamp,
        "score_offset": SCORE_OFFSET,
        "result_field": "unchanged terminal game result; trainer WDL blend remains configurable",
        "outputs": outputs,
    }
    (args.output / "manifest.json").write_text(
        json.dumps(provenance, indent=2, sort_keys=True) + "\n"
    )
    print(json.dumps(provenance, indent=2, sort_keys=True))


if __name__ == "__main__":
    main()
