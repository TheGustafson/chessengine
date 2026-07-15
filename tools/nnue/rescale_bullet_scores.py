#!/usr/bin/env python3
"""Copy a bulletformat ChessBoard corpus while rescaling its i16 score.

ChessBoard is a pinned 32-byte repr(C) record and its score occupies bytes
24..26. This utility is for controlled label-calibration experiments; it never
mutates the source corpus and writes provenance beside the derived output.
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


def digest(path: Path) -> str:
    hasher = hashlib.sha256()
    with path.open("rb") as stream:
        while chunk := stream.read(8 * 1024 * 1024):
            hasher.update(chunk)
    return hasher.hexdigest()


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("source", type=Path)
    parser.add_argument("output", type=Path)
    parser.add_argument("--factor", type=float, required=True)
    parser.add_argument("--clamp", type=int, default=32000)
    args = parser.parse_args()
    size = args.source.stat().st_size
    if size % RECORD_BYTES:
        parser.error(f"source size {size} is not divisible by {RECORD_BYTES}")
    if args.factor <= 0 or not 0 < args.clamp <= 32767:
        parser.error("factor and clamp must be positive, clamp <= 32767")
    args.output.parent.mkdir(parents=True, exist_ok=True)
    shutil.copyfile(args.source, args.output)
    raw = np.memmap(args.output, mode="r+", dtype=np.uint8, shape=(size,))
    scores = np.ndarray(
        shape=(size // RECORD_BYTES,), dtype="<i2", buffer=raw,
        offset=SCORE_OFFSET, strides=(RECORD_BYTES,),
    )
    original_min = int(scores.min())
    original_max = int(scores.max())
    scaled = np.clip(np.rint(scores.astype(np.float64) * args.factor), -args.clamp, args.clamp)
    scores[:] = scaled.astype(np.int16)
    raw.flush()
    del scores, raw
    provenance = {
        "schema": "bulletformat ChessBoard 32-byte score-label transform",
        "source": str(args.source.resolve()),
        "source_sha256": digest(args.source),
        "output": str(args.output.resolve()),
        "output_sha256": digest(args.output),
        "records": size // RECORD_BYTES,
        "score_offset": SCORE_OFFSET,
        "factor": args.factor,
        "clamp": args.clamp,
        "original_score_min": original_min,
        "original_score_max": original_max,
        "transformed_score_min": int(max(-args.clamp, round(original_min * args.factor))),
        "transformed_score_max": int(min(args.clamp, round(original_max * args.factor))),
        "reason": (
            "Lc0 Q expected score: (1+Q)/2 = sigmoid(2*atanh(Q)); source labels "
            "are 400*atanh(Q), so factor 2 matches the trainer's sigmoid(cp/400)"
        ),
    }
    meta = args.output.with_suffix(args.output.suffix + ".transform.json")
    meta.write_text(json.dumps(provenance, indent=2, sort_keys=True) + "\n")
    print(json.dumps(provenance, indent=2, sort_keys=True))


if __name__ == "__main__":
    main()
