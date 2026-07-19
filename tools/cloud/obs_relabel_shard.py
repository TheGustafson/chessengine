#!/usr/bin/env python3
"""Deep-relabel one curated FEN task into an immutable obs `.ceobs` shard.

The obs-era replacement for `dual_label_shard.py` (kept untouched as a
historical control). Instead of attaching flat paired scalar labels to an
LC0-feature shard, this worker drives the PACKAGED `root-analysis` binary over
a per-task FEN file and publishes the resulting Observation-Store shard, so the
Tier-F layers (L1 `PositionState` + L3 `SearchObservations` incl. `id_snapshots`
trajectory + `pv` + `completion` + `Provenance`) ride along for free
(data-schema-and-storage.md §5, §7 step 1).

Governance is preserved VERBATIM from `dual_label_shard.py`: sha256 object
metadata, `if_generation_match=0` no-overwrite upload, validate-existing-shard
resume, atomic local write, and deterministic output. The campaign
discriminators `{oracle, budget, date}` (§6.5) are stamped into both the config
hash and the GCS object metadata.

    obs-relabel = root-analysis --engine <SF18|reckless|archon> \
        --fens task.fens --nodes 100000 --reference-only --out shard.ceobs

`--reference-only` (Tier-F) is the affordable bulk-relabel shape added by M2 to
`root-analysis`: one Observation per tier, SKIPPING the ~35x per-legal-move
`searchmoves` arm loop. `--tier rich` omits `--reference-only` and drives the
existing per-arm Tier-R shape for the high-value gate.

Determinism / byte-exact resume: `root-analysis` embeds the pinned
`--created-unix` AND `hardware_identity` (CPU model from /proc/cpuinfo) in the
shard `Provenance`. With a pinned `--created-unix` (the submit pins it to the
campaign date) and the c2d-highcpu-16 machine lock (a single CPU family, so the
model string is stable across every Spot VM in the campaign), a retry re-emits
byte-identical shards, so the validate-existing resume holds. The machine-lock
rail therefore underpins byte determinism here, not only cost.
"""

from __future__ import annotations

import argparse
import hashlib
import json
import os
from pathlib import Path
import re
import shutil
import subprocess
import tempfile
import time
from typing import Any

from google.cloud import storage


SCHEMA = "archon.obs-relabel.v1"
OBS_MAGIC = b"CEOBS\0"
ORACLE_ENGINES = ("sf18", "reckless", "archon")
_EPISODE_SUMMARY = re.compile(r"(\d+)\s+episode")


def sha256(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as handle:
        for block in iter(lambda: handle.read(1024 * 1024), b""):
            digest.update(block)
    return digest.hexdigest()


def canonical_hash(value: Any) -> str:
    return hashlib.sha256(
        json.dumps(value, sort_keys=True, separators=(",", ":")).encode()
    ).hexdigest()


def parse_gs(uri: str) -> tuple[str, str]:
    if not uri.startswith("gs://"):
        raise ValueError(uri)
    bucket, _, name = uri[5:].partition("/")
    if not bucket or not name:
        raise ValueError(uri)
    return bucket, name


def download(client: storage.Client, uri: str, destination: Path) -> storage.Blob:
    bucket_name, name = parse_gs(uri)
    blob = client.bucket(bucket_name).blob(name)
    blob.reload()
    blob.download_to_filename(destination, checksum="crc32c")
    return blob


def read_json(client: storage.Client, uri: str) -> dict[str, Any]:
    bucket_name, name = parse_gs(uri)
    return json.loads(client.bucket(bucket_name).blob(name).download_as_text())


def format_budget(nodes: int) -> str:
    """The `{budget}` campaign discriminator, e.g. 100000 -> '100k' (§6.5)."""
    if nodes % 1_000_000 == 0:
        return f"{nodes // 1_000_000}m"
    if nodes % 1_000 == 0:
        return f"{nodes // 1_000}k"
    return str(nodes)


def campaign_id(oracle: str, nodes: int, date: str) -> str:
    """`{oracle}-{budget}-{date}`, e.g. 'sf18-100k-2026-07-18' (§6.5)."""
    return f"{oracle}-{format_budget(nodes)}-{date}"


def count_fens(path: Path) -> int:
    """Positions `root-analysis` will read: non-blank, non-comment lines
    (matches `root_analysis.rs::run`'s own line filter)."""
    rows = 0
    with path.open("r", encoding="utf-8") as handle:
        for line in handle:
            stripped = line.strip()
            if stripped and not stripped.startswith("#"):
                rows += 1
    return rows


def validate_ceobs(path: Path) -> None:
    """Confirm root-analysis produced a real obs shard, not a truncated stub."""
    with path.open("rb") as handle:
        magic = handle.read(len(OBS_MAGIC))
    if magic != OBS_MAGIC:
        raise RuntimeError(f"output is not an obs shard (bad magic): {path}")
    if path.stat().st_size <= len(OBS_MAGIC):
        raise RuntimeError(f"output shard is empty: {path}")


def run_root_analysis(
    root_analysis: Path,
    engine: Path,
    fens: Path,
    out: Path,
    nodes: int,
    tier: str,
    arm_mode: str,
    seed: int,
    created_unix: int | None,
    uci_options: list[str],
) -> int:
    """Invoke the packaged root-analysis; return the reported episode count."""
    argv = [
        str(root_analysis),
        "--engine", str(engine),
        "--fens", str(fens),
        "--out", str(out),
        "--nodes", str(nodes),
        "--arm-mode", arm_mode,
        "--seed", str(seed),
    ]
    if tier == "reference-only":
        # M2's Tier-F flag: L1+L3+Provenance, no per-arm searchmoves loop.
        argv.append("--reference-only")
    if created_unix is not None:
        argv += ["--created-unix", str(created_unix)]
    for option in uci_options:
        argv += ["--uci-option", option]
    completed = subprocess.run(argv, capture_output=True, text=True, check=False)
    if completed.returncode != 0:
        raise RuntimeError(
            f"root-analysis failed ({completed.returncode}): {completed.stderr.strip()}"
        )
    match = _EPISODE_SUMMARY.search(completed.stdout)
    return int(match.group(1)) if match else 0


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    source = parser.add_mutually_exclusive_group(required=True)
    source.add_argument("--input")
    source.add_argument("--manifest")
    parser.add_argument("--task-index", type=int)
    parser.add_argument("--output", required=True)
    parser.add_argument("--root-analysis", type=Path, required=True,
                        help="packaged root-analysis binary (image: /opt/archon/bin/root-analysis)")
    parser.add_argument("--engine", type=Path, required=True,
                        help="UCI engine to drive (SF18 / Reckless / ARCHON)")
    parser.add_argument("--oracle", choices=ORACLE_ENGINES, required=True,
                        help="oracle discriminator stamped into the campaign id + metadata")
    parser.add_argument("--nodes", type=int, required=True, help="node budget per position")
    parser.add_argument("--date", required=True, help="campaign date (YYYY-MM-DD)")
    parser.add_argument("--tier", choices=("reference-only", "rich"),
                        default="reference-only")
    parser.add_argument("--arm-mode", choices=("searchmoves", "child"),
                        default="searchmoves")
    parser.add_argument("--seed", type=int, default=0)
    parser.add_argument("--created-unix", type=int,
                        help="pin for byte-reproducible shards (submit pins to the campaign date)")
    parser.add_argument("--uci-option", action="append", default=[], metavar="NAME=VALUE",
                        help="passthrough to root-analysis (e.g. Threads=1, Hash=16, EvalFile=…)")
    args = parser.parse_args()

    if args.nodes <= 0:
        parser.error("node budget must be positive")
    if args.seed < 0:
        parser.error("--seed cannot be negative")
    if args.created_unix is not None and args.created_unix < 0:
        parser.error("--created-unix cannot be negative")
    if not re.fullmatch(r"\d{4}-\d{2}-\d{2}", args.date):
        parser.error("--date must be YYYY-MM-DD")
    for binary in (args.root_analysis, args.engine):
        if not binary.is_file() or not os.access(binary, os.X_OK):
            parser.error(f"not an executable binary: {binary}")
    for option in args.uci_option:
        if "=" not in option:
            parser.error(f"--uci-option wants NAME=VALUE, got '{option}'")

    client = storage.Client() if any(
        value and value.startswith("gs://")
        for value in (args.input, args.manifest, args.output)
    ) else None

    shard_detail: dict[str, Any] = {}
    input_uri = args.input
    if args.manifest:
        if client is None:
            raise AssertionError("GCS client missing")
        manifest = read_json(client, args.manifest)
        task_index = args.task_index
        if task_index is None:
            task_index = int(os.environ.get("BATCH_TASK_INDEX", "-1"))
        if not 0 <= task_index < len(manifest["shards"]):
            raise SystemExit(f"invalid task index {task_index}")
        shard_detail = manifest["shards"][task_index]
        input_uri = shard_detail["fen_file_uri"]
    if input_uri is None:
        raise AssertionError("input URI missing")

    campaign = campaign_id(args.oracle, args.nodes, args.date)
    config = {
        "schema": SCHEMA,
        "oracle": args.oracle,
        "campaign": campaign,
        "date": args.date,
        "search_limit": {"kind": "nodes", "value": args.nodes},
        "tier": args.tier,
        "reference_only": args.tier == "reference-only",
        "arm_mode": args.arm_mode,
        "seed": args.seed,
        "created_unix": args.created_unix,
        "score_perspective": "side_to_move",
        "root_analysis": {
            "path": str(args.root_analysis.resolve()),
            "sha256": sha256(args.root_analysis.resolve()),
        },
        "engine": {
            "path": str(args.engine.resolve()),
            "sha256": sha256(args.engine.resolve()),
            "oracle": args.oracle,
        },
        "uci_options": sorted(args.uci_option),
    }
    config_hash = canonical_hash(config)

    with tempfile.TemporaryDirectory(prefix="archon-obs-relabel-") as temporary:
        root = Path(temporary)
        input_path = root / "task.fens"
        input_blob = None
        if input_uri.startswith("gs://"):
            if client is None:
                raise AssertionError("GCS client missing")
            input_blob = download(client, input_uri, input_path)
        else:
            input_path = Path(input_uri).resolve()
        input_hash = sha256(input_path)
        if shard_detail and input_hash != shard_detail["sha256"]:
            raise RuntimeError("input hash does not match manifest")
        if input_blob and input_blob.metadata and input_blob.metadata.get("sha256") != input_hash:
            raise RuntimeError("input hash does not match object metadata")
        rows = count_fens(input_path)
        if shard_detail and rows != int(shard_detail["rows"]):
            raise RuntimeError(
                f"fen row count {rows} does not match manifest {shard_detail['rows']}"
            )
        if rows == 0:
            raise RuntimeError("task fen file has no positions")

        output_path = root / "shard.ceobs"
        started = time.monotonic()
        episodes = run_root_analysis(
            args.root_analysis.resolve(),
            args.engine.resolve(),
            input_path,
            output_path,
            args.nodes,
            args.tier,
            args.arm_mode,
            args.seed,
            args.created_unix,
            args.uci_option,
        )
        elapsed = time.monotonic() - started
        validate_ceobs(output_path)
        output_hash = sha256(output_path)

        expected_metadata = {
            "sha256": output_hash,
            "schema": SCHEMA,
            "config_hash": config_hash,
            "input_sha256": input_hash,
            "rows": str(rows),
            "episodes": str(episodes),
            "oracle": args.oracle,
            "budget": format_budget(args.nodes),
            "budget_nodes": str(args.nodes),
            "date": args.date,
            "campaign": campaign,
            "tier": args.tier,
        }

        if args.output.startswith("gs://"):
            if client is None:
                raise AssertionError("GCS client missing")
            bucket_name, prefix = parse_gs(args.output)
            ordinal = shard_detail.get("ordinal", args.task_index or 0)
            object_name = f"{prefix.rstrip('/')}/shards/{int(ordinal):06d}.ceobs"
            blob = client.bucket(bucket_name).blob(object_name)
            if blob.exists():
                blob.reload()
                if blob.metadata != expected_metadata or blob.size != output_path.stat().st_size:
                    raise RuntimeError(f"existing output differs: gs://{bucket_name}/{object_name}")
                status = "validated_existing"
            else:
                blob.metadata = expected_metadata
                blob.upload_from_filename(output_path, if_generation_match=0, checksum="crc32c")
                blob.reload()
                status = "uploaded"
            output_uri = f"gs://{bucket_name}/{object_name}"
        else:
            destination = Path(args.output).resolve()
            destination.parent.mkdir(parents=True, exist_ok=True)
            if destination.exists() and sha256(destination) != output_hash:
                raise RuntimeError(f"existing local output differs: {destination}")
            if not destination.exists():
                with tempfile.NamedTemporaryFile(
                    dir=destination.parent,
                    prefix=f".{destination.name}.",
                    delete=False,
                ) as handle:
                    staged = Path(handle.name)
                try:
                    shutil.copy2(output_path, staged)
                    os.replace(staged, destination)
                finally:
                    staged.unlink(missing_ok=True)
            output_uri = str(destination)
            status = "written"

    print(json.dumps({
        "status": status,
        "output": output_uri,
        "campaign": campaign,
        "rows": rows,
        "episodes": episodes,
        "seconds": elapsed,
        "rows_per_second": rows / elapsed if elapsed else None,
        "config_hash": config_hash,
        "sha256": output_hash,
    }, sort_keys=True))


if __name__ == "__main__":
    main()
