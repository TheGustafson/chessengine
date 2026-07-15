#!/usr/bin/env python3
"""Partition a position manifest into disjoint Batch task manifests."""

from __future__ import annotations

import argparse
import hashlib
import json
from pathlib import Path
from typing import Any


def canonical_hash(value: Any) -> str:
    return hashlib.sha256(
        json.dumps(value, sort_keys=True, separators=(",", ":")).encode()
    ).hexdigest()


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("manifest", type=Path)
    parser.add_argument("--output", type=Path, required=True)
    parser.add_argument("--partitions", type=int, required=True)
    args = parser.parse_args()
    if args.partitions <= 0:
        parser.error("--partitions must be positive")
    source = json.loads(args.manifest.read_text(encoding="utf-8"))
    shards = source.get("shards", [])
    if not shards or len({int(shard["ordinal"]) for shard in shards}) != len(shards):
        raise SystemExit("source manifest has no shards or duplicate ordinals")
    args.output.mkdir(parents=True, exist_ok=True)
    summary = []
    for partition in range(args.partitions):
        selected = shards[partition::args.partitions]
        derived = {
            key: value for key, value in source.items()
            if key not in {"shards", "total_rows", "total_shards", "manifest_hash"}
        }
        derived.update({
            "parent_manifest_hash": source.get("manifest_hash"),
            "partition_method": "ordinal_modulo",
            "partition_index": partition,
            "partition_count": args.partitions,
            "shards": selected,
            "total_rows": sum(int(shard["rows"]) for shard in selected),
            "total_shards": len(selected),
        })
        derived["manifest_hash"] = canonical_hash(derived)
        path = args.output / f"part-{partition:02d}.json"
        path.write_text(json.dumps(derived, indent=2, sort_keys=True) + "\n", encoding="utf-8")
        summary.append({
            "partition": partition,
            "path": str(path),
            "shards": len(selected),
            "rows": derived["total_rows"],
            "manifest_hash": derived["manifest_hash"],
            "first_ordinal": selected[0]["ordinal"],
            "last_ordinal": selected[-1]["ordinal"],
        })
    if sum(item["rows"] for item in summary) != source["total_rows"]:
        raise RuntimeError("partition row total does not match source")
    print(json.dumps(summary, indent=2, sort_keys=True))


if __name__ == "__main__":
    main()
