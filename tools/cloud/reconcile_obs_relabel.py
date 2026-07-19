#!/usr/bin/env python3
"""Reconcile immutable obs-relabel `.ceobs` shards, then register the campaign.

The obs-era counterpart of `reconcile_dual_labels.py`: inventory the shards a
wave published, enforce oracle/config identity, verify counts + SHA-256
metadata, optionally download a spread and run `obs-tool verify` (the full
structural + replay audit) on it. When a wave reconciles CLEAN and `--ledger-add`
is set, it materializes the campaign locally, `obs-tool verify`s the whole
campaign, then wires `tools/ledger/ledger.py add-data` (campaign id
`{oracle}-{budget}-{date}`, sha256 over the shard dir, `--source` = the GCS
prefix, the submit invocation as the regen recipe) and runs `ledger.py verify`
as the accept gate (data-schema-and-storage.md §6.5, §7 step 5).
"""

from __future__ import annotations

import argparse
import hashlib
import json
from pathlib import Path
import re
import subprocess
import sys
import tempfile

from google.cloud import storage

# Same keyless-client cross-import reconcile_dual_labels.py uses.
from shard_lc0_positions import parse_gs, storage_client


OUTPUT_SCHEMA = "archon.obs-relabel.v1"
OUTPUT_NAME = re.compile(r"^(?P<prefix>.+)/shards/(?P<ordinal>[0-9]{6})\.ceobs$")
OBS_MAGIC = b"CEOBS\0"


def sha256(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as handle:
        for block in iter(lambda: handle.read(8 * 1024 * 1024), b""):
            digest.update(block)
    return digest.hexdigest()


def read_json(client: storage.Client, uri: str) -> dict:
    bucket_name, name = parse_gs(uri)
    return json.loads(client.bucket(bucket_name).blob(name).download_as_text())


def obs_tool_verify(obs_tool: Path, shards: list[Path]) -> None:
    """Full structural + replay audit; exit 0 only if every shard replays."""
    result = subprocess.run(
        [str(obs_tool), "verify", *[str(shard) for shard in shards]],
        capture_output=True, text=True, check=False,
    )
    if result.returncode != 0:
        raise RuntimeError(
            f"obs-tool verify failed:\n{result.stdout}\n{result.stderr}".strip()
        )


def validate_download(path: Path, shard: dict, blob: storage.Blob, obs_tool: Path | None) -> None:
    if sha256(path) != blob.metadata["sha256"]:
        raise RuntimeError(f"download hash mismatch for {blob.name}")
    with path.open("rb") as handle:
        if handle.read(len(OBS_MAGIC)) != OBS_MAGIC:
            raise RuntimeError(f"downloaded object is not an obs shard: {blob.name}")
    if obs_tool is not None:
        obs_tool_verify(obs_tool, [path])


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--manifest", required=True)
    parser.add_argument("--output", required=True, help="GCS campaign prefix")
    parser.add_argument("--project", default="archon-502500")
    parser.add_argument("--expected-config-hash")
    parser.add_argument("--download-samples", type=int, default=0)
    parser.add_argument("--obs-tool", type=Path, default=Path("target/release/obs-tool"),
                        help="obs-tool binary for structural + replay verify")
    # Ledger wiring (M6): register the campaign after a clean wave.
    parser.add_argument("--ledger-add", action="store_true")
    parser.add_argument("--campaign-id", help="ledger id, e.g. sf18-100k-2026-07-18")
    parser.add_argument("--campaign-dir", type=Path,
                        help="local dir to materialize + hash the campaign shards")
    parser.add_argument("--submit-cmd", help="exact submit invocation, recorded as the regen recipe")
    parser.add_argument("--ledger", type=Path, default=Path("tools/ledger/ledger.py"))
    parser.add_argument("--python", default=sys.executable)
    args = parser.parse_args()
    if args.download_samples < 0:
        parser.error("--download-samples cannot be negative")
    if args.ledger_add and not (args.campaign_id and args.campaign_dir and args.submit_cmd):
        parser.error("--ledger-add requires --campaign-id, --campaign-dir, and --submit-cmd")
    obs_tool = args.obs_tool if args.obs_tool.is_file() else None

    client = storage_client(args.project)
    manifest = read_json(client, args.manifest)
    expected = {int(shard["ordinal"]): shard for shard in manifest["shards"]}
    bucket_name, prefix = parse_gs(args.output)
    object_prefix = prefix.rstrip("/") + "/shards/"
    blobs: dict[int, storage.Blob] = {}
    config_hashes: set[str] = set()
    campaigns: set[str] = set()
    rows = 0
    bytes_total = 0
    for blob in client.list_blobs(bucket_name, prefix=object_prefix):
        match = OUTPUT_NAME.match(blob.name)
        if not match or match.group("prefix") != prefix.rstrip("/"):
            raise RuntimeError(f"unexpected output object: gs://{bucket_name}/{blob.name}")
        ordinal = int(match.group("ordinal"))
        if ordinal not in expected or ordinal in blobs:
            raise RuntimeError(f"unexpected or duplicate ordinal {ordinal}")
        blob.reload()
        metadata = blob.metadata or {}
        shard = expected[ordinal]
        required_metadata = {
            "schema": OUTPUT_SCHEMA,
            "input_sha256": shard["sha256"],
            "rows": str(shard["rows"]),
        }
        for key, value in required_metadata.items():
            if metadata.get(key) != value:
                raise RuntimeError(f"metadata {key} mismatch for ordinal {ordinal}")
        for key in ("sha256", "config_hash", "oracle", "budget_nodes", "date", "campaign"):
            if not metadata.get(key):
                raise RuntimeError(f"missing {key} metadata for ordinal {ordinal}")
        config_hashes.add(metadata["config_hash"])
        campaigns.add(metadata["campaign"])
        blobs[ordinal] = blob
        rows += int(metadata["rows"])
        bytes_total += int(blob.size)

    if len(config_hashes) > 1:
        raise RuntimeError(f"multiple relabel config hashes: {sorted(config_hashes)}")
    if len(campaigns) > 1:
        raise RuntimeError(f"multiple campaigns in one prefix: {sorted(campaigns)}")
    if args.expected_config_hash and config_hashes and config_hashes != {args.expected_config_hash}:
        raise RuntimeError(f"unexpected relabel config hash: {sorted(config_hashes)}")
    campaign = next(iter(campaigns), None)

    sampled: list[int] = []
    if args.download_samples and blobs:
        available = sorted(blobs)
        count = min(args.download_samples, len(available))
        sampled = sorted({available[round(i * (len(available) - 1) / max(1, count - 1))]
                          for i in range(count)})
        with tempfile.TemporaryDirectory(prefix="archon-obs-audit-") as temporary:
            root = Path(temporary)
            for ordinal in sampled:
                path = root / f"{ordinal:06d}.ceobs"
                blobs[ordinal].download_to_filename(path, checksum="crc32c")
                validate_download(path, expected[ordinal], blobs[ordinal], obs_tool)

    missing = sorted(set(expected).difference(blobs))
    complete = not missing
    report = {
        "schema": "archon.obs-relabel-reconciliation.v1",
        "manifest": args.manifest,
        "output": args.output,
        "campaign": campaign,
        "expected_shards": len(expected),
        "present_shards": len(blobs),
        "missing_shards": len(missing),
        "first_missing": missing[:20],
        "present_rows": rows,
        "present_bytes": bytes_total,
        "config_hashes": sorted(config_hashes),
        "download_validated_ordinals": sampled,
        "obs_tool_verified": obs_tool is not None and bool(sampled),
        "complete": complete,
    }

    if args.ledger_add:
        if not complete:
            raise SystemExit("refusing to ledger an incomplete wave (missing shards)")
        report["ledger"] = ledger_register(
            args, prefix, bucket_name, blobs, expected, rows, campaign, obs_tool, client,
        )

    print(json.dumps(report, indent=2, sort_keys=True))


def ledger_register(args, prefix, bucket_name, blobs, expected, rows, campaign, obs_tool, client):
    """Materialize the campaign locally, verify it, then add-data + verify."""
    if campaign and args.campaign_id != campaign:
        raise SystemExit(
            f"--campaign-id {args.campaign_id!r} != shard-metadata campaign {campaign!r}"
        )
    campaign_dir: Path = args.campaign_dir
    campaign_dir.mkdir(parents=True, exist_ok=True)
    # Download any shard whose local copy is missing or drifted (verify sha256).
    for ordinal, blob in sorted(blobs.items()):
        local = campaign_dir / f"{ordinal:06d}.ceobs"
        want = blob.metadata["sha256"]
        if local.is_file() and sha256(local) == want:
            continue
        blob.download_to_filename(local, checksum="crc32c")
        if sha256(local) != want:
            raise RuntimeError(f"materialized shard hash mismatch: {local}")
    shard_paths = sorted(campaign_dir.glob("*.ceobs"))
    if len(shard_paths) != len(expected):
        raise SystemExit(
            f"campaign dir has {len(shard_paths)} shards, expected {len(expected)}"
        )
    if obs_tool is not None:
        obs_tool_verify(obs_tool, shard_paths)

    source_prefix = f"gs://{bucket_name}/{prefix.rstrip('/')}"
    gen_recipe = json.dumps({
        "submit_cmd": args.submit_cmd,
        "source_prefix": source_prefix,
        "manifest": args.manifest,
    }, sort_keys=True)
    add_data = [
        args.python, str(args.ledger), "add-data",
        "--id", args.campaign_id,
        "--path", str(campaign_dir),
        "--glob", "*.ceobs",
        "--source", source_prefix,
        "--positions", str(rows),
        "--format", "obs .ceobs Tier-F relabel (root-analysis)",
        "--gen-recipe", gen_recipe,
        "--notes", f"obs-relabel campaign {campaign} reconciled from gs://{bucket_name}/{prefix}",
    ]
    added = _ledger_add_idempotent(args, add_data)
    verify = subprocess.run(
        [args.python, str(args.ledger), "verify"],
        capture_output=True, text=True, check=False,
    )
    if verify.returncode != 0:
        raise SystemExit(f"ledger verify FAILED (accept gate):\n{verify.stdout}\n{verify.stderr}")
    return {
        "campaign_id": args.campaign_id,
        "campaign_dir": str(campaign_dir),
        "shards": len(shard_paths),
        "add_data": added,
        "verify": "ok",
    }


def _ledger_add_idempotent(args, add_data_argv) -> str:
    """add-data, but a re-reconcile of an already-registered campaign is fine
    (the ledger is append-only and dies on a duplicate id — that is expected)."""
    # ledger.py lives at <repo>/tools/ledger/ledger.py; its registries are at
    # <repo>/ledger/*.jsonl (ledger.py: REPO = parents[2], LEDGER = REPO/ledger).
    ledger_data = Path(args.ledger).resolve().parents[2] / "ledger" / "data.jsonl"
    if ledger_data.is_file():
        for line in ledger_data.read_text().splitlines():
            if line.strip() and json.loads(line).get("id") == args.campaign_id:
                return "already-registered"
    result = subprocess.run(add_data_argv, capture_output=True, text=True, check=False)
    if result.returncode != 0:
        raise SystemExit(f"ledger add-data failed:\n{result.stdout}\n{result.stderr}")
    return "registered"


if __name__ == "__main__":
    main()
