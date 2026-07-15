#!/usr/bin/env python3
"""Blend aligned Bullet ChessBoard score labels without changing positions.

The inputs must contain the same 32-byte records in the same order.  Bytes
24..26 (the little-endian i16 score) may differ; every other byte is verified
before the output is committed.  The first input supplies the board and result
fields and the rounded weighted mean supplies the new score.
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
DEFAULT_CHUNK_RECORDS = 1_000_000


def digest(path: Path) -> str:
    hasher = hashlib.sha256()
    with path.open("rb") as stream:
        while chunk := stream.read(8 * 1024 * 1024):
            hasher.update(chunk)
    return hasher.hexdigest()


def score_view(raw: np.memmap, records: int) -> np.ndarray:
    return np.ndarray(
        shape=(records,),
        dtype="<i2",
        buffer=raw,
        offset=SCORE_OFFSET,
        strides=(RECORD_BYTES,),
    )


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("first", type=Path)
    parser.add_argument("second", type=Path)
    parser.add_argument("output", type=Path)
    parser.add_argument("--first-weight", type=float, default=0.5)
    parser.add_argument("--second-weight", type=float, default=0.5)
    parser.add_argument("--clamp", type=int, default=32000)
    parser.add_argument("--chunk-records", type=int, default=DEFAULT_CHUNK_RECORDS)
    args = parser.parse_args()

    first_size = args.first.stat().st_size
    second_size = args.second.stat().st_size
    if first_size != second_size or first_size % RECORD_BYTES:
        parser.error(
            f"inputs must have equal 32-byte-aligned sizes: {first_size} != {second_size}"
        )
    if args.first.resolve() == args.output.resolve() or args.second.resolve() == args.output.resolve():
        parser.error("output must not overwrite an input")
    if args.first_weight < 0 or args.second_weight < 0:
        parser.error("weights must be non-negative")
    weight_sum = args.first_weight + args.second_weight
    if weight_sum <= 0:
        parser.error("at least one weight must be positive")
    if not 0 < args.clamp <= 32767 or args.chunk_records <= 0:
        parser.error("clamp must be 1..32767 and chunk-records must be positive")

    args.output.parent.mkdir(parents=True, exist_ok=True)
    shutil.copyfile(args.first, args.output)
    records = first_size // RECORD_BYTES
    first = np.memmap(args.first, mode="r", dtype=np.uint8, shape=(records, RECORD_BYTES))
    second = np.memmap(args.second, mode="r", dtype=np.uint8, shape=(records, RECORD_BYTES))
    output = np.memmap(args.output, mode="r+", dtype=np.uint8, shape=(records, RECORD_BYTES))
    first_scores = score_view(first, records)
    second_scores = score_view(second, records)
    output_scores = score_view(output, records)

    score_min = 32767
    score_max = -32768
    clipped = 0
    for start in range(0, records, args.chunk_records):
        stop = min(records, start + args.chunk_records)
        # Score bytes are the sole permitted difference between aligned inputs.
        if not np.array_equal(first[start:stop, :SCORE_OFFSET], second[start:stop, :SCORE_OFFSET]):
            raise SystemExit(f"record mismatch before score field in records {start}:{stop}")
        if not np.array_equal(first[start:stop, SCORE_OFFSET + 2 :], second[start:stop, SCORE_OFFSET + 2 :]):
            raise SystemExit(f"record mismatch after score field in records {start}:{stop}")
        blended = np.rint(
            (
                args.first_weight * first_scores[start:stop].astype(np.float64)
                + args.second_weight * second_scores[start:stop].astype(np.float64)
            )
            / weight_sum
        )
        clipped += int(np.count_nonzero((blended < -args.clamp) | (blended > args.clamp)))
        blended = np.clip(blended, -args.clamp, args.clamp).astype("<i2")
        output_scores[start:stop] = blended
        score_min = min(score_min, int(blended.min()))
        score_max = max(score_max, int(blended.max()))

    output.flush()
    del first_scores, second_scores, output_scores, first, second, output
    provenance = {
        "schema": "archon.bullet-score-blend.v1",
        "record_bytes": RECORD_BYTES,
        "score_offset": SCORE_OFFSET,
        "records": records,
        "first": str(args.first.resolve()),
        "first_sha256": digest(args.first),
        "first_weight": args.first_weight,
        "second": str(args.second.resolve()),
        "second_sha256": digest(args.second),
        "second_weight": args.second_weight,
        "normalised_weights": [args.first_weight / weight_sum, args.second_weight / weight_sum],
        "clamp": args.clamp,
        "clipped_records": clipped,
        "transformed_score_min": score_min,
        "transformed_score_max": score_max,
        "output": str(args.output.resolve()),
        "output_sha256": digest(args.output),
        "reason": (
            "Consensus target from independently searched, scale-calibrated teachers; "
            "all non-score record bytes were verified identical."
        ),
    }
    meta = args.output.with_suffix(args.output.suffix + ".transform.json")
    meta.write_text(json.dumps(provenance, indent=2, sort_keys=True) + "\n")
    print(json.dumps(provenance, indent=2, sort_keys=True))


if __name__ == "__main__":
    main()
