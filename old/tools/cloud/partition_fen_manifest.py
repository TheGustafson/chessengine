#!/usr/bin/env python3
"""Chunk a curated FEN subset into immutable per-task shards + a task manifest.

The obs-era, TARGETED-relabel replacement for the relabel-everything
`shard_lc0_positions.py` + `partition_position_manifest.py` path. It ingests the
LOCALLY disagreement-selected subset (`|SF18_moderate − existing_label|` ranked;
data-quality-strategy Dimension 1) rather than a whole LC0-feature corpus, so
the cloud's job is exactly "here is a curated subset, deep-relabel it" (cloud
scale-up assessment gap 2.a).

Input is a text file of `FEN|cp|result` (or plain FEN) lines: the `cp`/`result`
fields are the EXISTING labels used locally for disagreement selection and are
dropped here — the relabel worker (`root-analysis` via `obs_relabel_shard.py`)
produces fresh labels, and the old label is rejoined later at export by
`canonical_key`. Each per-task shard is a plain-FEN file (one FEN per line) that
`root-analysis --fens` reads directly.

Emits a task manifest of `{fen_file_uri, rows, sha256, ordinal}` records
(top-level `shards`/`total_shards`/`total_rows`/`manifest_hash` identical in
shape to the position manifest, so `submit_obs_relabel.py`'s validation and the
reconcile 1:1 accounting reuse it verbatim). Uploads are immutable
(no-overwrite, validate-existing), inheriting `shard_lc0_positions.upload_immutable`.
"""

from __future__ import annotations

import argparse
import json
from pathlib import Path
import tempfile
from typing import Any

# Reuse the VERBATIM keyless GCS client + immutable no-overwrite uploader (the
# same cross-import reconcile_dual_labels.py uses; shard_lc0_positions.py stays
# in the repo as the shared GCS helper library even though its scalar SHARDING
# path is retired from the worker image).
from shard_lc0_positions import (
    canonical_hash,
    parse_gs,
    sha256,
    storage_client,
    upload_immutable,
)


SCHEMA = "archon.fen-task-manifest.v1"


def extract_fen(line: str) -> str | None:
    """The FEN field of a `FEN|cp|result` or plain-FEN line, or None to skip.

    A plain FEN contains no `|`, so `split('|', 1)[0]` is a no-op for it. Light
    structural validation only (a board field with ranks + a side-to-move
    token); full legality is `Board::from_fen`'s job inside root-analysis.
    """
    stripped = line.strip()
    if not stripped or stripped.startswith("#"):
        return None
    fen = stripped.split("|", 1)[0].strip()
    fields = fen.split()
    if len(fields) < 2 or "/" not in fields[0] or fields[1] not in ("w", "b"):
        raise SystemExit(f"malformed FEN: {line.strip()!r}")
    return fen


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("input", type=Path, help="curated FEN|cp|result (or plain FEN) subset")
    parser.add_argument("--corpus-id", required=True, help="the disagreement-selection id")
    parser.add_argument("--output", required=True, help="GCS prefix for the task inputs")
    parser.add_argument("--project", default="archon-502500")
    parser.add_argument("--rows-per-task", type=int, default=2_000,
                        help="FENs per task shard (small: deep relabel is slow, cf. Appendix C)")
    parser.add_argument("--max-tasks", type=int)
    args = parser.parse_args()

    if args.rows_per_task <= 0 or (args.max_tasks is not None and args.max_tasks <= 0):
        parser.error("--rows-per-task and --max-tasks must be positive")
    if not args.input.is_file():
        parser.error(f"missing input file: {args.input}")
    bucket_name, prefix = parse_gs(args.output)
    client = storage_client(args.project)
    bucket = client.bucket(bucket_name)

    # Pass 1: count valid FENs by STREAMING the input (O(1) memory). A
    # 168M-FEN list would need ~17GB; this box has ~7GB. source_rows feeds
    # config_hash, so we need the count before sharding.
    source_rows = 0
    with args.input.open("r", encoding="utf-8") as handle:
        for line in handle:
            if extract_fen(line) is not None:
                source_rows += 1
    if source_rows == 0:
        raise SystemExit("curated subset has no positions")

    config = {
        "schema": SCHEMA,
        "corpus_id": args.corpus_id,
        "selection": "locally disagreement-selected subset (data-quality-strategy Dimension 1)",
        "rows_per_task": args.rows_per_task,
        "source_file": str(args.input.resolve()),
        "source_sha256": sha256(args.input.resolve()),
        "source_rows": source_rows,
    }
    config_hash = canonical_hash(config)
    manifest: dict[str, Any] = {**config, "config_hash": config_hash, "shards": []}

    import concurrent.futures

    total_shards_est = (source_rows + args.rows_per_task - 1) // args.rows_per_task
    if args.max_tasks is not None:
        total_shards_est = min(total_shards_est, args.max_tasks)

    with tempfile.TemporaryDirectory(prefix="archon-fen-task-") as temporary:
        temporary_root = Path(temporary)

        def _write_shard(ordinal: int, batch: list[str], start: int, stop: int):
            filename = f"{ordinal:06d}.fens"
            local_path = temporary_root / filename
            local_path.write_text("\n".join(batch) + "\n", encoding="utf-8")
            digest = sha256(local_path)
            object_name = f"{prefix}/tasks/{filename}"
            metadata = {
                "sha256": digest,
                "schema": SCHEMA,
                "config_hash": config_hash,
                "corpus_id": args.corpus_id,
                "source_start": str(start),
                "source_stop": str(stop),
                "rows": str(stop - start),
            }
            return (ordinal, local_path, digest, object_name, metadata, start, stop)

        def _upload(spec):
            ordinal, local_path, digest, object_name, metadata, start, stop = spec
            generation, _ = upload_immutable(bucket, object_name, local_path, metadata)
            size = local_path.stat().st_size
            local_path.unlink(missing_ok=True)  # bounded disk: drop after upload
            return {
                "ordinal": ordinal,
                "fen_file_uri": f"gs://{bucket_name}/{object_name}",
                "generation": generation,
                "sha256": digest,
                "bytes": size,
                "rows": stop - start,
                "source_start": start,
                "source_stop": stop,
            }

        # Pass 2: stream shards out in batches. Each batch of shard files is
        # uploaded in PARALLEL (per-object round-trip dominates) then unlinked,
        # so at most UPLOAD_BATCH shard files exist on disk at once. The keyless
        # client + immutable no-overwrite semantics are unchanged.
        UPLOAD_BATCH = 256
        done = 0

        def _flush(batch):
            nonlocal done
            if not batch:
                return
            with concurrent.futures.ThreadPoolExecutor(max_workers=32) as pool:
                for shard in pool.map(_upload, batch):
                    manifest["shards"].append(shard)
                    done += 1
                    if done % 500 == 0:
                        print(f"uploaded {done}/{total_shards_est} shards", flush=True)

        pending: list = []
        buf: list[str] = []
        ordinal = 0
        start = 0
        with args.input.open("r", encoding="utf-8") as handle:
            for line in handle:
                fen = extract_fen(line)
                if fen is None:
                    continue
                buf.append(fen)
                if len(buf) >= args.rows_per_task:
                    stop = start + len(buf)
                    pending.append(_write_shard(ordinal, buf, start, stop))
                    ordinal += 1
                    start = stop
                    buf = []
                    if args.max_tasks is not None and ordinal >= args.max_tasks:
                        break
                    if len(pending) >= UPLOAD_BATCH:
                        _flush(pending)
                        pending = []
            if buf and (args.max_tasks is None or ordinal < args.max_tasks):
                stop = start + len(buf)
                pending.append(_write_shard(ordinal, buf, start, stop))
        _flush(pending)
        manifest["shards"].sort(key=lambda item: item["ordinal"])

    manifest["total_rows"] = sum(int(shard["rows"]) for shard in manifest["shards"])
    manifest["total_shards"] = len(manifest["shards"])
    manifest_hash = canonical_hash(manifest)
    manifest["manifest_hash"] = manifest_hash
    with tempfile.TemporaryDirectory(prefix="archon-fen-manifest-") as temporary:
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
                "rows": str(manifest["total_rows"]),
                "shards": str(manifest["total_shards"]),
            },
        )
    print(json.dumps({
        "manifest": f"gs://{bucket_name}/{prefix}/manifest.json",
        "corpus_id": args.corpus_id,
        "rows": manifest["total_rows"],
        "shards": manifest["total_shards"],
        "config_hash": config_hash,
        "manifest_hash": manifest_hash,
    }, sort_keys=True))


if __name__ == "__main__":
    main()
