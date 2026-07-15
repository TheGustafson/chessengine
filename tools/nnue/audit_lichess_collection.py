#!/usr/bin/env python3
"""Fast metadata and shard-boundary audit for the Lichess Parquet collection."""

from __future__ import annotations

import argparse
import hashlib
import json
from pathlib import Path

import pyarrow.parquet as pq


EXPECTED_COLUMNS = {
    "fen": "string",
    "line": "string",
    "depth": "uint8",
    "knodes": "int32",
    "cp": "int16",
    "mate": "int8",
}


def sha256(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as handle:
        while chunk := handle.read(8 * 1024 * 1024):
            digest.update(chunk)
    return digest.hexdigest()


def edge_fen(parquet: pq.ParquetFile, row_group: int, tail: bool) -> str:
    table = parquet.read_row_group(row_group, columns=["fen"])
    values = table.column("fen")
    return values[-1 if tail else 0].as_py()


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("root", type=Path)
    parser.add_argument("--glob", default="train-*.parquet")
    parser.add_argument("--hash", action="store_true", help="also SHA-256 every shard")
    parser.add_argument("--output", type=Path)
    args = parser.parse_args()

    paths = sorted(args.root.glob(args.glob))
    if not paths:
        parser.error(f"no files matched {args.root / args.glob}")
    files = []
    total_rows = 0
    total_row_groups = 0
    schema_reference = None
    boundaries = []
    prior_last = None
    for path in paths:
        parquet = pq.ParquetFile(path)
        schema = {field.name: str(field.type) for field in parquet.schema_arrow}
        if schema_reference is None:
            schema_reference = schema
        first = edge_fen(parquet, 0, False)
        last = edge_fen(parquet, parquet.num_row_groups - 1, True)
        if prior_last is not None:
            boundaries.append({
                "left": files[-1]["name"],
                "right": path.name,
                "same_fen_across_boundary": prior_last == first,
                "left_last_fen": prior_last,
                "right_first_fen": first,
            })
        record = {
            "name": path.name,
            "bytes": path.stat().st_size,
            "rows": parquet.metadata.num_rows,
            "row_groups": parquet.num_row_groups,
            "schema": schema,
            "schema_matches_first": schema == schema_reference,
            "first_fen": first,
            "last_fen": last,
        }
        if args.hash:
            record["sha256"] = sha256(path)
        files.append(record)
        total_rows += parquet.metadata.num_rows
        total_row_groups += parquet.num_row_groups
        prior_last = last

    result = {
        "schema": "archon.lichess-evaluations.collection-audit.v1",
        "root": str(args.root.resolve()),
        "file_count": len(files),
        "total_bytes": sum(row["bytes"] for row in files),
        "total_rows": total_rows,
        "total_row_groups": total_row_groups,
        "expected_schema": EXPECTED_COLUMNS,
        "schema_matches_contract": schema_reference == EXPECTED_COLUMNS,
        "all_schemas_identical": all(row["schema_matches_first"] for row in files),
        "cross_shard_fen_groups": sum(row["same_fen_across_boundary"] for row in boundaries),
        "boundaries": boundaries,
        "files": files,
    }
    encoded = json.dumps(result, indent=2, sort_keys=True) + "\n"
    if args.output:
        args.output.parent.mkdir(parents=True, exist_ok=True)
        temporary = args.output.with_suffix(args.output.suffix + ".tmp")
        temporary.write_text(encoded)
        temporary.replace(args.output)
    print(encoded, end="")


if __name__ == "__main__":
    main()
