#!/usr/bin/env python3
"""Build a minimal Archon worker context from explicitly pinned artifacts."""

from __future__ import annotations

import argparse
import hashlib
import json
from pathlib import Path
import shutil
import subprocess
import tempfile


HERE = Path(__file__).resolve().parent
ROOT = HERE.parents[1]
WORKER = HERE / "worker"


def sha256(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as handle:
        for block in iter(lambda: handle.read(1024 * 1024), b""):
            digest.update(block)
    return digest.hexdigest()


def command_output(*argv: str) -> str:
    return subprocess.check_output(argv, cwd=ROOT, text=True).strip()


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--tag", required=True, help="complete local or registry image tag")
    parser.add_argument("--push", action="store_true")
    args = parser.parse_args()

    specification = json.loads(
        (WORKER / "image_manifest.json").read_text(encoding="utf-8")
    )
    with tempfile.TemporaryDirectory(prefix="archon-worker-context-") as temporary:
        context = Path(temporary)
        shutil.copy2(WORKER / "Dockerfile", context / "Dockerfile")
        shutil.copy2(WORKER / "requirements.lock", context / "requirements.lock")

        artifacts = []
        for item in specification["artifacts"]:
            source = ROOT / item["source"]
            if not source.is_file():
                raise SystemExit(f"missing image input: {source}")
            actual_hash = sha256(source)
            expected_hash = item.get("sha256")
            if expected_hash is not None and actual_hash != expected_hash:
                raise SystemExit(
                    f"pinned hash mismatch for {source}: expected {expected_hash}, got {actual_hash}"
                )
            destination = context / item["image_path"]
            destination.parent.mkdir(parents=True, exist_ok=True)
            shutil.copy2(source, destination)
            artifacts.append({**item, "sha256": actual_hash})

        entrypoint = context / "bin" / "archon-worker"
        entrypoint.parent.mkdir(parents=True, exist_ok=True)
        shutil.copy2(WORKER / "archon_worker.py", entrypoint)
        entrypoint_hash = sha256(entrypoint)
        artifacts.append({
            "source": "tools/cloud/worker/archon_worker.py",
            "image_path": "bin/archon-worker",
            "sha256": entrypoint_hash,
        })

        diff = subprocess.run(
            ["git", "diff", "--binary", "HEAD"],
            cwd=ROOT,
            check=True,
            capture_output=True,
        ).stdout
        provenance = {
            "schema": "archon.worker-image-provenance.v1",
            "source": {
                "git_commit": command_output("git", "rev-parse", "HEAD"),
                "tracked_diff_sha256": hashlib.sha256(diff).hexdigest(),
                "worktree_clean": not bool(command_output("git", "status", "--porcelain")),
            },
            "artifacts": artifacts,
        }
        (context / "provenance.json").write_text(
            json.dumps(provenance, indent=2, sort_keys=True) + "\n", encoding="utf-8"
        )

        subprocess.run(
            ["docker", "build", "--platform", "linux/amd64", "--tag", args.tag, "."],
            cwd=context,
            check=True,
        )
        if args.push:
            subprocess.run(["docker", "push", args.tag], check=True)


if __name__ == "__main__":
    main()
