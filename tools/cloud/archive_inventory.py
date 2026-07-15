#!/usr/bin/env python3
"""Create a deterministic size and SHA-256 inventory for durable archives."""

from __future__ import annotations

import argparse
import hashlib
import json
from pathlib import Path
import tempfile


SCHEMA = "archon.archive-inventory.v1"


def sha256(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as handle:
        for block in iter(lambda: handle.read(8 * 1024 * 1024), b""):
            digest.update(block)
    return digest.hexdigest()


def canonical_hash(value: object) -> str:
    return hashlib.sha256(
        json.dumps(value, sort_keys=True, separators=(",", ":")).encode()
    ).hexdigest()


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("paths", nargs="+", type=Path)
    parser.add_argument("--root", type=Path, required=True)
    parser.add_argument("--output", type=Path, required=True)
    parser.add_argument("--archive-id", required=True)
    args = parser.parse_args()
    root = args.root.resolve()
    files: list[Path] = []
    for raw in args.paths:
        path = raw.resolve()
        if path.is_dir():
            files.extend(candidate for candidate in path.rglob("*") if candidate.is_file())
        elif path.is_file():
            files.append(path)
        else:
            raise SystemExit(f"missing archive input: {path}")
    files = sorted(set(files), key=lambda path: str(path.relative_to(root)))
    records = []
    for ordinal, path in enumerate(files):
        relative = str(path.relative_to(root))
        record = {"path": relative, "bytes": path.stat().st_size, "sha256": sha256(path)}
        records.append(record)
        print(f"{ordinal + 1}/{len(files)} {relative} {record['sha256']}", flush=True)
    inventory = {
        "schema": SCHEMA,
        "archive_id": args.archive_id,
        "root": str(root),
        "files": records,
        "total_files": len(records),
        "total_bytes": sum(record["bytes"] for record in records),
    }
    inventory["inventory_hash"] = canonical_hash(inventory)
    destination = args.output.resolve()
    destination.parent.mkdir(parents=True, exist_ok=True)
    with tempfile.NamedTemporaryFile(
        mode="w", encoding="utf-8", dir=destination.parent,
        prefix=f".{destination.name}.", delete=False,
    ) as handle:
        json.dump(inventory, handle, indent=2, sort_keys=True)
        handle.write("\n")
        staged = Path(handle.name)
    staged.replace(destination)
    print(json.dumps({
        "output": str(destination),
        "files": len(records),
        "bytes": inventory["total_bytes"],
        "inventory_hash": inventory["inventory_hash"],
    }, sort_keys=True))


if __name__ == "__main__":
    main()
