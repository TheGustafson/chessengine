#!/usr/bin/env python3
"""Stream normalized LC0 feature arrays into immutable GCS recovery shards."""

from __future__ import annotations

import argparse
from datetime import datetime, timedelta
import hashlib
import json
import os
from pathlib import Path
import tempfile
from typing import Any
import subprocess

import google.auth
from google.auth import exceptions as google_auth_exceptions
from google.auth.credentials import Credentials
from google.cloud import storage
import numpy as np


SCHEMA = "archon.position-shards.v1"


class GcloudCredentials(Credentials):
    """Refreshable local credentials without creating an ADC or key file."""

    def refresh(self, request: Any) -> None:
        self.token = subprocess.check_output(
            ["gcloud", "auth", "print-access-token"], text=True
        ).strip()
        self.expiry = datetime.now() + timedelta(minutes=50)

    @property
    def requires_scopes(self) -> bool:
        return False


def storage_client(project: str) -> storage.Client:
    try:
        credentials, detected_project = google.auth.default()
        return storage.Client(project=project or detected_project, credentials=credentials)
    except google_auth_exceptions.DefaultCredentialsError:
        credentials = GcloudCredentials()
        credentials.refresh(None)
        return storage.Client(project=project, credentials=credentials)


def sha256(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as handle:
        for block in iter(lambda: handle.read(1024 * 1024), b""):
            digest.update(block)
    return digest.hexdigest()


def canonical_hash(value: Any) -> str:
    encoded = json.dumps(value, sort_keys=True, separators=(",", ":")).encode()
    return hashlib.sha256(encoded).hexdigest()


def parse_gs(uri: str) -> tuple[str, str]:
    if not uri.startswith("gs://"):
        raise ValueError(f"expected gs:// URI: {uri}")
    bucket, _, prefix = uri[5:].partition("/")
    if not bucket or not prefix:
        raise ValueError(f"URI must contain bucket and prefix: {uri}")
    return bucket, prefix.rstrip("/")


def parse_input(value: str) -> tuple[str, Path]:
    name, separator, raw_path = value.partition("=")
    if not separator or not name or not raw_path:
        raise argparse.ArgumentTypeError("input must be NAME=DATASET_DIRECTORY")
    return name, Path(raw_path).resolve()


def upload_immutable(
    bucket: storage.Bucket,
    object_name: str,
    source: Path,
    metadata: dict[str, str],
) -> tuple[int, str]:
    blob = bucket.blob(object_name)
    if blob.exists():
        blob.reload()
        if blob.metadata != metadata or blob.size != source.stat().st_size:
            raise RuntimeError(f"existing object differs: gs://{bucket.name}/{object_name}")
        print(f"validated gs://{bucket.name}/{object_name}", flush=True)
        return int(blob.generation), metadata["sha256"]
    blob.metadata = metadata
    blob.upload_from_filename(source, if_generation_match=0, checksum="crc32c")
    blob.reload()
    if blob.metadata != metadata or blob.size != source.stat().st_size:
        raise RuntimeError(f"uploaded object failed validation: {object_name}")
    print(f"uploaded gs://{bucket.name}/{object_name}", flush=True)
    return int(blob.generation), metadata["sha256"]


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--input", action="append", type=parse_input, required=True)
    parser.add_argument("--corpus-id", required=True)
    parser.add_argument("--output", required=True)
    parser.add_argument("--project", default="archon-502500")
    parser.add_argument("--shard-rows", type=int, default=250_000)
    parser.add_argument("--max-shards", type=int)
    parser.add_argument("--compression", action=argparse.BooleanOptionalAction, default=True)
    args = parser.parse_args()

    if args.shard_rows <= 0 or (args.max_shards is not None and args.max_shards <= 0):
        parser.error("--shard-rows and --max-shards must be positive")
    bucket_name, prefix = parse_gs(args.output)
    client = storage_client(args.project)
    bucket = client.bucket(bucket_name)

    sources = []
    for dataset_name, root in args.input:
        meta_path = root / "meta.json"
        if not meta_path.is_file():
            raise SystemExit(f"missing metadata: {meta_path}")
        for side in ("white", "black"):
            path = root / side / "feats.npy"
            if not path.is_file():
                raise SystemExit(f"missing features: {path}")
            array = np.load(path, mmap_mode="r")
            if array.ndim != 2 or array.shape[1] != 32 or array.dtype != np.int16:
                raise SystemExit(f"unexpected feature array: {path}: {array.shape} {array.dtype}")
            sources.append({
                "dataset": dataset_name,
                "side": side,
                "path": str(path),
                "rows": len(array),
                "shape": list(array.shape),
                "dtype": str(array.dtype),
                "file_bytes": path.stat().st_size,
                "meta_sha256": sha256(meta_path),
            })

    config = {
        "schema": SCHEMA,
        "corpus_id": args.corpus_id,
        "state_completeness": "piece_state_bootstrap_only",
        "reconstruction": {
            "side_to_move": "white (mover-normalized source)",
            "castling": "unavailable; forced none",
            "en_passant": "unavailable; forced none",
            "rule50": "unavailable; forced zero",
            "repetition": "unavailable",
        },
        "shard_rows": args.shard_rows,
        "compressed_npz": args.compression,
        "max_shards": args.max_shards,
        "sources": sources,
    }
    config_hash = canonical_hash(config)
    manifest: dict[str, Any] = {**config, "config_hash": config_hash, "shards": []}

    with tempfile.TemporaryDirectory(prefix="archon-position-shard-") as temporary:
        temporary_root = Path(temporary)
        global_ordinal = 0
        for source in sources:
            if args.max_shards is not None and len(manifest["shards"]) >= args.max_shards:
                break
            features = np.load(source["path"], mmap_mode="r")
            for start in range(0, len(features), args.shard_rows):
                if args.max_shards is not None and len(manifest["shards"]) >= args.max_shards:
                    break
                stop = min(start + args.shard_rows, len(features))
                shard_number = len(manifest["shards"])
                filename = f"{shard_number:06d}.npz"
                local_path = temporary_root / filename
                arrays = {
                    "features": np.asarray(features[start:stop]),
                    "source_index": np.arange(start, stop, dtype=np.uint64),
                }
                if args.compression:
                    np.savez_compressed(local_path, **arrays)
                else:
                    np.savez(local_path, **arrays)
                digest = sha256(local_path)
                object_name = f"{prefix}/shards/{filename}"
                metadata = {
                    "sha256": digest,
                    "schema": SCHEMA,
                    "config_hash": config_hash,
                    "dataset": source["dataset"],
                    "normalized_side": source["side"],
                    "source_start": str(start),
                    "source_stop": str(stop),
                    "rows": str(stop - start),
                }
                generation, _ = upload_immutable(bucket, object_name, local_path, metadata)
                manifest["shards"].append({
                    "ordinal": shard_number,
                    "uri": f"gs://{bucket_name}/{object_name}",
                    "generation": generation,
                    "sha256": digest,
                    "bytes": local_path.stat().st_size,
                    "rows": stop - start,
                    "global_start": global_ordinal,
                    "dataset": source["dataset"],
                    "normalized_side": source["side"],
                    "source_start": start,
                    "source_stop": stop,
                })
                global_ordinal += stop - start
                local_path.unlink()

    manifest["total_rows"] = global_ordinal
    manifest["total_shards"] = len(manifest["shards"])
    manifest_hash = canonical_hash(manifest)
    manifest["manifest_hash"] = manifest_hash
    with tempfile.TemporaryDirectory(prefix="archon-position-manifest-") as temporary:
        manifest_path = Path(temporary) / "manifest.json"
        manifest_path.write_text(
            json.dumps(manifest, indent=2, sort_keys=True) + "\n", encoding="utf-8"
        )
        digest = sha256(manifest_path)
        upload_immutable(
            bucket,
            f"{prefix}/manifest.json",
            manifest_path,
            {
                "sha256": digest,
                "schema": SCHEMA,
                "config_hash": config_hash,
                "manifest_hash": manifest_hash,
                "rows": str(global_ordinal),
                "shards": str(len(manifest["shards"])),
            },
        )
    print(json.dumps({
        "manifest": f"gs://{bucket_name}/{prefix}/manifest.json",
        "rows": global_ordinal,
        "shards": len(manifest["shards"]),
        "config_hash": config_hash,
    }, sort_keys=True))


if __name__ == "__main__":
    main()
