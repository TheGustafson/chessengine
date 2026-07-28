#!/usr/bin/env python3
"""Create a score-rescaled copy of a Bullet FEN-text corpus."""

from __future__ import annotations

import argparse
import json
from pathlib import Path


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("source", type=Path)
    parser.add_argument("output", type=Path)
    parser.add_argument("--scale", type=float, required=True)
    parser.add_argument("--limit", type=int, default=2499)
    args = parser.parse_args()
    if args.scale <= 0 or args.limit <= 0:
        parser.error("scale and limit must be positive")
    if args.output.exists() and any(args.output.iterdir()):
        parser.error(f"output must be absent or empty: {args.output}")

    totals = {"rows": 0, "clipped": 0}
    files = []
    for split in ("train", "val"):
        sources = sorted((args.source / split).glob("shard-*.txt"))
        if not sources:
            raise RuntimeError(f"no source shards in {args.source / split}")
        destination = args.output / split
        destination.mkdir(parents=True, exist_ok=True)
        for source in sources:
            output = destination / source.name
            rows = clipped = 0
            with source.open(encoding="utf-8") as inp, output.open("x", encoding="utf-8") as out:
                for line in inp:
                    fen, score, result = (part.strip() for part in line.split("|"))
                    scaled = round(int(score) * args.scale)
                    bounded = max(-args.limit, min(args.limit, scaled))
                    clipped += bounded != scaled
                    out.write(f"{fen} | {bounded} | {result}\n")
                    rows += 1
            totals["rows"] += rows
            totals["clipped"] += clipped
            files.append({"split": split, "source": str(source), "output": str(output),
                          "rows": rows, "clipped": clipped})
    manifest = {"schema": "archon.rescaled-value-text.v1", "source": str(args.source.resolve()),
                "scale": args.scale, "limit": args.limit, "totals": totals, "files": files}
    (args.output / "manifest.json").write_text(json.dumps(manifest, indent=2) + "\n")
    print(json.dumps({"scale": args.scale, "totals": totals}, indent=2))


if __name__ == "__main__":
    main()
