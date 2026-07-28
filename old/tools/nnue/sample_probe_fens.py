#!/usr/bin/env python3
"""Select a deterministic hash-ranked FEN parity corpus from NNUE text shards."""

from __future__ import annotations

import argparse
import hashlib
import heapq
import json
from pathlib import Path


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("inputs", nargs="+", type=Path)
    parser.add_argument("--output", required=True, type=Path)
    parser.add_argument("--count", type=int, default=4096)
    parser.add_argument("--seed", type=int, default=0xD40B1)
    args = parser.parse_args()
    if args.count <= 0:
        parser.error("--count must be positive")

    # Python's heap is a min-heap. Store negative ranks so the largest retained
    # rank is evicted whenever a better (smaller) deterministic rank arrives.
    heap: list[tuple[int, int, int, str]] = []
    rows = 0
    seed = args.seed.to_bytes(8, "little", signed=False)
    for source_number, path in enumerate(args.inputs):
        with path.open("r", encoding="utf-8") as handle:
            for line_number, line in enumerate(handle):
                rows += 1
                fen = line.partition("|")[0].strip()
                if len(fen.split()) != 6:
                    raise ValueError(f"{path}:{line_number + 1}: expected six-field FEN")
                digest = hashlib.blake2b(
                    seed
                    + source_number.to_bytes(4, "little")
                    + line_number.to_bytes(8, "little")
                    + fen.encode("utf-8"),
                    digest_size=8,
                ).digest()
                rank = int.from_bytes(digest, "little")
                item = (-rank, source_number, line_number, fen)
                if len(heap) < args.count:
                    heapq.heappush(heap, item)
                elif item > heap[0]:
                    heapq.heapreplace(heap, item)

    selected = sorted(heap, key=lambda item: (-item[0], item[1], item[2]))
    args.output.parent.mkdir(parents=True, exist_ok=True)
    args.output.write_text("\n".join(item[3] for item in selected) + "\n", encoding="utf-8")
    manifest = {
        "schema": "archon.nnue-parity-fens.v1",
        "inputs": [str(path.resolve()) for path in args.inputs],
        "rows_scanned": rows,
        "selected": len(selected),
        "seed": args.seed,
        "ranking": "smallest blake2b-64(seed,source,line,fen)",
        "output": str(args.output.resolve()),
        "sha256": hashlib.sha256(args.output.read_bytes()).hexdigest(),
    }
    args.output.with_suffix(args.output.suffix + ".json").write_text(
        json.dumps(manifest, indent=2, sort_keys=True) + "\n", encoding="utf-8"
    )
    print(json.dumps(manifest, indent=2, sort_keys=True))


if __name__ == "__main__":
    main()
