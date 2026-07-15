#!/usr/bin/env python3
"""Stream a collection-aware Lichess value projection directly to Bulletformat."""

from __future__ import annotations

import argparse
import hashlib
import json
import os
import subprocess
from collections import Counter
from pathlib import Path

import numpy as np
import pyarrow as pa
import pyarrow.compute as pc
import pyarrow.csv as csv
import pyarrow.parquet as pq


REPO = Path(__file__).resolve().parents[2]


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
    parser.add_argument("root", type=Path)
    parser.add_argument("output", type=Path)
    parser.add_argument("--min-depth", type=int, default=20)
    parser.add_argument("--min-knodes", type=int, default=1_000)
    parser.add_argument("--max-abs-cp", type=int, default=2_000)
    parser.add_argument("--max-positions", type=int)
    parser.add_argument("--converter", type=Path,
                        default=REPO / "tools/nnue/bf-convert/target/release/nnue_convert")
    args = parser.parse_args()

    paths = sorted(args.root.glob("train-*.parquet"))
    if len(paths) != 17:
        parser.error(f"expected 17 Parquets, found {len(paths)}")
    if not args.converter.exists():
        parser.error(f"build converter first: missing {args.converter}")
    args.output.parent.mkdir(parents=True, exist_ok=True)
    temporary = args.output.with_suffix(args.output.suffix + ".tmp")
    temporary.unlink(missing_ok=True)
    command = [str(args.converter), "convert-stdin", str(temporary)]
    process = subprocess.Popen(command, stdin=subprocess.PIPE)
    assert process.stdin is not None
    stream = pa.output_stream(process.stdin)
    csv_options = csv.WriteOptions(include_header=False, delimiter="|", quoting_style="none")

    counts = Counter()
    depth_counts = Counter()
    last_fen = None
    stop = False
    try:
        for shard, path in enumerate(paths):
            parquet = pq.ParquetFile(path)
            for batch in parquet.iter_batches(
                batch_size=131_072,
                columns=["fen", "depth", "knodes", "cp", "mate"],
            ):
                counts["rows_scanned"] += batch.num_rows
                fen = batch.column("fen")
                n = batch.num_rows
                first = np.empty(n, dtype=np.bool_)
                first[0] = fen[0].as_py() != last_fen
                if n > 1:
                    first[1:] = pc.not_equal(fen.slice(1), fen.slice(0, n - 1)).to_numpy(
                        zero_copy_only=False
                    )
                last_fen = fen[-1].as_py()
                indices = np.flatnonzero(first)
                counts["position_groups"] += len(indices)
                if not len(indices):
                    continue
                selected = batch.take(pa.array(indices, type=pa.int64()))
                depth = selected.column("depth")
                knodes = selected.column("knodes")
                cp_values = selected.column("cp")
                mate = selected.column("mate")
                board_field = pc.list_element(pc.split_pattern(selected.column("fen"), " "), 0)
                piece_count = pc.count_substring_regex(
                    board_field, pattern="[prnbqkPRNBQK]"
                )
                structurally_packable = pc.and_kleene(
                    pc.less_equal(piece_count, 32),
                    pc.and_kleene(
                        pc.equal(pc.count_substring_regex(board_field, pattern="K"), 1),
                        pc.and_kleene(
                            pc.equal(pc.count_substring_regex(board_field, pattern="k"), 1),
                            pc.and_kleene(
                                pc.less_equal(pc.count_substring_regex(board_field, pattern="P"), 8),
                                pc.less_equal(pc.count_substring_regex(board_field, pattern="p"), 8),
                            ),
                        ),
                    ),
                )
                eligible = pc.and_kleene(
                    pc.and_kleene(pc.greater_equal(depth, args.min_depth),
                                  pc.greater_equal(knodes, args.min_knodes)),
                    pc.and_kleene(
                        structurally_packable,
                        pc.and_kleene(pc.is_null(mate),
                                      pc.and_kleene(pc.is_valid(cp_values),
                                                    pc.less_equal(pc.abs(cp_values), args.max_abs_cp))),
                    ),
                )
                accepted = selected.filter(eligible)
                rejected = selected.filter(pc.invert(eligible))
                counts["accepted"] += accepted.num_rows
                counts["rejected"] += rejected.num_rows
                for value, count in zip(*np.unique(
                    accepted.column("depth").to_numpy(zero_copy_only=False),
                    return_counts=True,
                )):
                    depth_counts[int(value)] += int(count)
                if args.max_positions is not None and counts["accepted"] > args.max_positions:
                    excess = counts["accepted"] - args.max_positions
                    accepted = accepted.slice(0, accepted.num_rows - excess)
                    counts["accepted"] -= excess
                    counts["rejected_by_limit"] += excess
                    stop = True
                if accepted.num_rows:
                    export = pa.table({
                        "fen": accepted.column("fen"),
                        "cp": accepted.column("cp"),
                        "result": pa.array([0.5] * accepted.num_rows, type=pa.float32()),
                    })
                    csv.write_csv(export, stream, write_options=csv_options)
                if stop:
                    break
            print(
                f"shard {shard + 1}/17 rows={counts['rows_scanned']} "
                f"groups={counts['position_groups']} accepted={counts['accepted']}",
                flush=True,
            )
            if stop:
                break
    finally:
        stream.close()
    returncode = process.wait()
    if returncode:
        temporary.unlink(missing_ok=True)
        raise RuntimeError(f"converter exited {returncode}")
    expected_bytes = counts["accepted"] * 32
    if temporary.stat().st_size != expected_bytes:
        temporary.unlink(missing_ok=True)
        raise RuntimeError("Bullet output byte count differs from accepted record count")
    os.replace(temporary, args.output)
    manifest = {
        "schema": "archon.lichess-evaluations.bullet-projection.v1",
        "source_root": str(args.root.resolve()),
        "source_files": [path.name for path in paths],
        "selection": {
            "row": "first flattened row per globally contiguous FEN group",
            "rationale": "source-recommended first PV of deepest evaluation; audited ordering",
            "min_depth": args.min_depth,
            "min_knodes": args.min_knodes,
            "max_abs_cp": args.max_abs_cp,
            "mates": "excluded",
            "result": "neutral storage placeholder; training MUST use WDL=0",
        },
        "counts": dict(counts),
        "accepted_depth_counts": dict(sorted(depth_counts.items())),
        "output": str(args.output.resolve()),
        "bytes": expected_bytes,
        "sha256": sha256(args.output),
        "record_bytes": 32,
    }
    atomic_json(args.output.with_suffix(args.output.suffix + ".json"), manifest)
    print(json.dumps(manifest, indent=2, sort_keys=True))


if __name__ == "__main__":
    main()
