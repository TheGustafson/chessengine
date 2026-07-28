#!/usr/bin/env python3
"""Resumably download and verify the immutable Kaggle Lichess eval shards.

Run with a transient dependency if kagglehub is not installed in the project:

  uv run --with kagglehub python tools/nnue/download_lichess_evaluations.py

The script deliberately downloads individual files.  The complete Kaggle ZIP
would temporarily require space for both the archive and extracted Parquet.
"""

from __future__ import annotations

import argparse
import hashlib
import json
import os
import shutil
import sys
import time
import urllib.request
from concurrent.futures import ThreadPoolExecutor, as_completed
from pathlib import Path

import kagglehub


DATASET = "lichess/chess-evaluations"
VERSION = 14
FILES_URL = f"https://www.kaggle.com/api/v1/datasets/list/{DATASET}"
VIEW_URL = f"https://www.kaggle.com/api/v1/datasets/view/{DATASET}"


def fetch_json(url: str) -> dict:
    with urllib.request.urlopen(url, timeout=60) as response:
        return json.load(response)


def sha256(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as handle:
        while chunk := handle.read(8 * 1024 * 1024):
            digest.update(chunk)
    return digest.hexdigest()


def atomic_json(path: Path, value: object) -> None:
    temporary = path.with_suffix(path.suffix + ".tmp")
    temporary.write_text(json.dumps(value, indent=2, sort_keys=True) + "\n")
    os.replace(temporary, path)


def download_one(output: Path, entry: dict, force: bool) -> dict:
    name = entry["name"]
    expected = int(entry["totalBytes"])
    destination = output / name
    if destination.exists() and destination.stat().st_size == expected and not force:
        status = "already_complete"
    else:
        if destination.exists():
            destination.unlink()
        resolved = Path(
            kagglehub.dataset_download(
                f"{DATASET}/versions/{VERSION}",
                path=name,
                output_dir=str(output),
                force_download=force,
            )
        )
        if resolved.resolve() != destination.resolve():
            raise RuntimeError(f"unexpected Kaggle path for {name}: {resolved}")
        status = "downloaded"
    actual = destination.stat().st_size
    if actual != expected:
        raise RuntimeError(f"size mismatch for {name}: expected {expected}, got {actual}")
    return {
        "name": name,
        "bytes": actual,
        "sha256": sha256(destination),
        "status": status,
    }


def main() -> int:
    parser = argparse.ArgumentParser()
    parser.add_argument(
        "--output",
        type=Path,
        default=Path("data/corpora/lichess-chess-evaluations/kaggle-v14/raw"),
    )
    parser.add_argument("--workers", type=int, default=3)
    parser.add_argument("--force", action="store_true")
    parser.add_argument("--skip", type=int, default=0)
    args = parser.parse_args()
    if args.workers < 1:
        parser.error("--workers must be positive")

    args.output.mkdir(parents=True, exist_ok=True)
    file_response = fetch_json(FILES_URL)
    metadata = fetch_json(VIEW_URL)
    entries = file_response["datasetFiles"]
    entries.sort(key=lambda item: item["name"])
    if len(entries) != 17:
        raise RuntimeError(f"expected 17 files, Kaggle reports {len(entries)}")

    required = sum(int(item["totalBytes"]) for item in entries)
    existing = sum(
        path.stat().st_size
        for path in args.output.glob("*.parquet")
        if path.is_file()
    )
    free = shutil.disk_usage(args.output).free
    remaining = max(0, required - existing)
    if free - remaining < 8 * 1024**3:
        raise RuntimeError(
            f"insufficient safety margin: {free / 1024**3:.1f} GiB free, "
            f"{remaining / 1024**3:.1f} GiB remaining"
        )

    snapshot = {
        "schema_version": 1,
        "dataset": DATASET,
        "version": VERSION,
        "retrieved_unix": int(time.time()),
        "license": metadata.get("licenseName"),
        "last_updated": metadata.get("lastUpdated"),
        "description": metadata.get("description"),
        "expected_total_bytes": required,
        "expected_files": entries,
    }
    atomic_json(args.output.parent / "source-metadata.json", snapshot)

    selected = entries[args.skip :]
    results: list[dict] = []
    with ThreadPoolExecutor(max_workers=args.workers) as executor:
        futures = {
            executor.submit(download_one, args.output, entry, args.force): entry
            for entry in selected
        }
        for future in as_completed(futures):
            result = future.result()
            results.append(result)
            results.sort(key=lambda item: item["name"])
            atomic_json(
                args.output.parent / "download-manifest.json",
                {
                    "schema_version": 1,
                    "dataset": DATASET,
                    "version": VERSION,
                    "expected_file_count": len(entries),
                    "verified": results,
                },
            )
            print(
                f"verified {result['name']} {result['bytes']} bytes "
                f"sha256={result['sha256']}",
                flush=True,
            )
    return 0


if __name__ == "__main__":
    sys.exit(main())
