#!/usr/bin/env python3
"""Snapshot and optionally fetch one official Stockfish master binpack.

The repository is 287 GB.  Listing is the default; use ``--file`` and either
``--head-bytes`` for an audit slice or ``--full`` for an intentional full pull.
"""

from __future__ import annotations

import argparse
import hashlib
import json
import os
import socket
import time
import urllib.error
import urllib.request
from pathlib import Path


REPO = "official-stockfish/master-binpacks"
REVISION = "main"
API = f"https://huggingface.co/api/datasets/{REPO}/tree/{REVISION}?recursive=true&expand=true&limit=100"


def fetch_manifest() -> list[dict]:
    with urllib.request.urlopen(API, timeout=60) as response:
        rows = json.load(response)
    return [row for row in rows if row["path"].endswith(".binpack")]


def atomic_json(path: Path, value: object) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    temporary = path.with_suffix(path.suffix + ".tmp")
    temporary.write_text(json.dumps(value, indent=2, sort_keys=True) + "\n")
    os.replace(temporary, path)


def sha256(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as handle:
        while chunk := handle.read(8 * 1024 * 1024):
            digest.update(chunk)
    return digest.hexdigest()


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--file")
    mode = parser.add_mutually_exclusive_group()
    mode.add_argument("--head-bytes", type=int)
    mode.add_argument("--full", action="store_true")
    parser.add_argument(
        "--root",
        type=Path,
        default=Path("data/corpora/official-stockfish-master-binpacks/main"),
    )
    args = parser.parse_args()
    entries = fetch_manifest()
    snapshot = {
        "schema": "archon.official-stockfish-master-binpacks.remote-manifest.v1",
        "repository": REPO,
        "revision": REVISION,
        "license": "odbl",
        "total_bytes": sum(int(row["size"]) for row in entries),
        "files": entries,
    }
    atomic_json(args.root / "remote-manifest.json", snapshot)
    if args.file is None:
        print(json.dumps(snapshot, indent=2, sort_keys=True))
        return
    matches = [row for row in entries if row["path"] == args.file]
    if not matches:
        parser.error(f"unknown file: {args.file}")
    if not args.full and args.head_bytes is None:
        parser.error("select --head-bytes N or --full")
    entry = matches[0]
    expected = int(entry["size"])
    requested = expected if args.full else min(expected, args.head_bytes)
    suffix = "" if args.full else f".head{requested}"
    output = args.root / ("raw" if args.full else "samples") / f"{args.file}{suffix}"
    output.parent.mkdir(parents=True, exist_ok=True)
    temporary = output.with_suffix(output.suffix + ".tmp")
    url = f"https://huggingface.co/datasets/{REPO}/resolve/{REVISION}/{args.file}?download=true"
    offset = temporary.stat().st_size if temporary.exists() else 0
    if offset > requested:
        temporary.unlink()
        offset = 0
    failures = 0
    while offset < requested:
        request = urllib.request.Request(url)
        request.add_header("Range", f"bytes={offset}-{requested - 1}")
        try:
            with urllib.request.urlopen(request, timeout=45) as response:
                status = getattr(response, "status", None)
                if offset and status != 206:
                    print("server ignored resume range; restarting temporary file", flush=True)
                    temporary.unlink(missing_ok=True)
                    offset = 0
                    continue
                mode = "ab" if offset else "wb"
                with temporary.open(mode) as handle:
                    while offset < requested:
                        chunk = response.read(min(8 * 1024 * 1024, requested - offset))
                        if not chunk:
                            break
                        handle.write(chunk)
                        offset += len(chunk)
                        if offset % (256 * 1024 * 1024) < len(chunk) or offset == requested:
                            print(f"fetched {offset}/{requested} bytes", flush=True)
            failures = 0
        except (TimeoutError, socket.timeout, urllib.error.URLError) as error:
            failures += 1
            if failures > 20:
                raise RuntimeError(f"download repeatedly stalled at byte {offset}") from error
            print(f"download retry {failures} at byte {offset}: {error}", flush=True)
            time.sleep(min(failures, 5))
    actual = temporary.stat().st_size
    if actual != requested:
        temporary.unlink(missing_ok=True)
        raise RuntimeError(f"expected {requested} bytes, received {actual}")
    digest = sha256(temporary)
    if args.full and entry.get("lfs", {}).get("oid") != digest:
        temporary.unlink(missing_ok=True)
        raise RuntimeError("full-file SHA-256 differs from Hugging Face LFS oid")
    os.replace(temporary, output)
    result = {
        "schema": "archon.official-stockfish-master-binpack-fetch.v1",
        "repository": REPO,
        "revision": REVISION,
        "source": entry,
        "output": str(output.resolve()),
        "bytes": actual,
        "sha256": digest,
        "complete_file": args.full,
    }
    atomic_json(output.with_suffix(output.suffix + ".json"), result)
    print(json.dumps(result, indent=2, sort_keys=True))


if __name__ == "__main__":
    main()
