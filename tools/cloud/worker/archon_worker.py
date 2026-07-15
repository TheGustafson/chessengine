#!/usr/bin/env python3
"""Small, auditable entrypoint for Archon Batch worker images."""

from __future__ import annotations

import argparse
import hashlib
import json
import os
from pathlib import Path
import subprocess
import sys


ROOT = Path("/opt/archon")


def sha256(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as handle:
        for block in iter(lambda: handle.read(1024 * 1024), b""):
            digest.update(block)
    return digest.hexdigest()


def verify() -> None:
    manifest = json.loads((ROOT / "provenance.json").read_text(encoding="utf-8"))
    failures: list[str] = []
    for item in manifest["artifacts"]:
        path = ROOT / item["image_path"]
        if not path.is_file():
            failures.append(f"missing: {path}")
            continue
        actual = sha256(path)
        if actual != item["sha256"]:
            failures.append(f"hash mismatch: {path}: {actual}")
    if failures:
        raise SystemExit("image verification failed:\n" + "\n".join(failures))
    print(json.dumps({
        "status": "ok",
        "schema": manifest["schema"],
        "source_commit": manifest["source"]["git_commit"],
        "artifacts": len(manifest["artifacts"]),
    }, sort_keys=True))


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    subparsers = parser.add_subparsers(dest="command", required=True)
    subparsers.add_parser("verify", help="verify every packaged artifact hash")
    run = subparsers.add_parser("run", help="execute a command after verification")
    run.add_argument("argv", nargs=argparse.REMAINDER)
    args = parser.parse_args()
    verify()
    if args.command == "run":
        if not args.argv:
            parser.error("run requires a command")
        completed = subprocess.run(args.argv, env=os.environ.copy(), check=False)
        raise SystemExit(completed.returncode)


if __name__ == "__main__":
    main()
