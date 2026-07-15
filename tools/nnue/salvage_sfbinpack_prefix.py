#!/usr/bin/env python3
"""Copy all complete BINP chunks from a byte-range prefix into a valid file."""

from __future__ import annotations

import argparse
import hashlib
import json
import os
import struct
from pathlib import Path


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("input", type=Path)
    parser.add_argument("output", type=Path)
    args = parser.parse_args()
    args.output.parent.mkdir(parents=True, exist_ok=True)
    temporary = args.output.with_suffix(args.output.suffix + ".tmp")
    chunks = 0
    source_bytes = 0
    written = 0
    digest = hashlib.sha256()
    with args.input.open("rb") as source, temporary.open("wb") as destination:
        while True:
            header = source.read(8)
            source_bytes += len(header)
            if not header:
                break
            if len(header) != 8 or header[:4] != b"BINP":
                break
            size = struct.unpack("<I", header[4:])[0]
            if size > 100 * 1024 * 1024:
                raise RuntimeError(f"implausible BINP chunk size: {size}")
            payload = source.read(size)
            source_bytes += len(payload)
            if len(payload) != size:
                break
            destination.write(header)
            destination.write(payload)
            digest.update(header)
            digest.update(payload)
            chunks += 1
            written += 8 + size
    if chunks == 0:
        temporary.unlink(missing_ok=True)
        raise RuntimeError("input contained no complete BINP chunks")
    os.replace(temporary, args.output)
    report = {
        "schema": "archon.sfbinpack-complete-prefix.v1",
        "input": str(args.input.resolve()),
        "input_bytes": args.input.stat().st_size,
        "complete_chunks": chunks,
        "output": str(args.output.resolve()),
        "output_bytes": written,
        "sha256": digest.hexdigest(),
        "discarded_tail_bytes": args.input.stat().st_size - written,
    }
    metadata = args.output.with_suffix(args.output.suffix + ".json")
    metadata_tmp = metadata.with_suffix(metadata.suffix + ".tmp")
    metadata_tmp.write_text(json.dumps(report, indent=2, sort_keys=True) + "\n")
    os.replace(metadata_tmp, metadata)
    print(json.dumps(report, indent=2, sort_keys=True))


if __name__ == "__main__":
    main()
