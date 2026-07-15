#!/usr/bin/env python3
"""Export a provenance-aligned, eval-only Bullet smoke corpus from Parquet.

The source score is White-relative.  The text result is a neutral placeholder
and the corresponding trainer MUST use WDL weight zero.
"""

from __future__ import annotations

import argparse
import hashlib
import json
import os
from collections import Counter
from pathlib import Path

import chess
import pyarrow as pa
import pyarrow.parquet as pq


INDEX_SCHEMA = pa.schema(
    [
        ("fen4", pa.string()),
        ("source_file", pa.string()),
        ("source_row", pa.int64()),
        ("position_hash64", pa.uint64()),
        ("depth", pa.uint8()),
        ("knodes", pa.int32()),
        ("cp_white", pa.int16()),
        ("best_move_uci_chess960", pa.string()),
        ("split", pa.string()),
    ]
)


def hash64(fen: str, seed: int) -> int:
    return int.from_bytes(
        hashlib.blake2b(seed.to_bytes(8, "little") + fen.encode(), digest_size=8).digest(),
        "little",
    )


def standard_position(fen4: str) -> bool:
    try:
        board = chess.Board(f"{fen4} 0 1")
    except ValueError:
        return False
    return board.is_valid() and not board.has_chess960_castling_rights()


class Exporter:
    def __init__(self, args: argparse.Namespace, handles: dict[str, object], writer: pq.ParquetWriter) -> None:
        self.args = args
        self.handles = handles
        self.writer = writer
        self.current_fen: str | None = None
        self.current_key: tuple[int, int] | None = None
        self.current_block_first: dict | None = None
        self.blocks: list[dict] = []
        self.counts: Counter[str] = Counter()
        self.index_buffer: list[dict] = []
        self.done = False

    def finish_block(self) -> None:
        if self.current_block_first is not None:
            self.blocks.append(self.current_block_first)
        self.current_block_first = None

    def finish_position(self) -> None:
        if self.current_fen is None:
            return
        self.finish_block()
        self.counts["source_positions"] += 1
        selected = max(self.blocks, key=lambda row: (row["depth"], row["knodes"]))
        reason = self.reject_reason(self.current_fen, selected)
        if reason is not None:
            self.counts[f"rejected_{reason}"] += 1
        else:
            rank = hash64(self.current_fen, self.args.seed)
            split = "val" if rank % 1000 < self.args.val_per_mille else "train"
            fen6 = f"{self.current_fen} 0 1"
            self.handles[split].write(f"{fen6} | {selected['cp']} | 0.5\n")
            first_move = selected["line"].partition(" ")[0]
            self.index_buffer.append(
                {
                    "fen4": self.current_fen,
                    "source_file": selected["source_file"],
                    "source_row": selected["source_row"],
                    "position_hash64": rank,
                    "depth": selected["depth"],
                    "knodes": selected["knodes"],
                    "cp_white": selected["cp"],
                    "best_move_uci_chess960": first_move,
                    "split": split,
                }
            )
            self.counts[f"accepted_{split}"] += 1
            if len(self.index_buffer) >= 65_536:
                self.flush_index()
            if self.counts["accepted_train"] + self.counts["accepted_val"] >= self.args.positions:
                self.done = True
        self.blocks = []

    def reject_reason(self, fen: str, row: dict) -> str | None:
        if row["cp"] is None:
            return "mate"
        if abs(int(row["cp"])) > self.args.max_abs_cp:
            return "score"
        if int(row["depth"]) < self.args.min_depth:
            return "depth"
        if int(row["knodes"]) < self.args.min_knodes:
            return "nodes"
        if not row["line"]:
            return "missing_pv"
        if not standard_position(fen):
            return "invalid_or_chess960"
        return None

    def add(self, row: dict, source_file: str, source_row: int) -> None:
        fen = row["fen"]
        if self.current_fen != fen:
            self.finish_position()
            if self.done:
                return
            self.current_fen = fen
            self.current_key = None
        key = (row["depth"], row["knodes"])
        if self.current_key != key:
            self.finish_block()
            self.current_key = key
            self.current_block_first = {
                **row,
                "source_file": source_file,
                "source_row": source_row,
            }

    def flush_index(self) -> None:
        if self.index_buffer:
            self.writer.write_table(pa.Table.from_pylist(self.index_buffer, schema=INDEX_SCHEMA))
            self.index_buffer = []

    def finish(self) -> None:
        if not self.done:
            self.finish_position()
        self.flush_index()


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("inputs", nargs="+", type=Path)
    parser.add_argument("--output", required=True, type=Path)
    parser.add_argument("--positions", type=int, default=1_000_000)
    parser.add_argument("--min-depth", type=int, default=20)
    parser.add_argument("--min-knodes", type=int, default=1_000)
    parser.add_argument("--max-abs-cp", type=int, default=2000)
    parser.add_argument("--val-per-mille", type=int, default=10)
    parser.add_argument("--seed", type=int, default=0xA4C40)
    args = parser.parse_args()
    if not 0 <= args.val_per_mille <= 1000:
        parser.error("--val-per-mille must be in [0,1000]")

    args.output.mkdir(parents=True, exist_ok=True)
    paths = {
        "train": args.output / "train.txt",
        "val": args.output / "val.txt",
        "index": args.output / "index.parquet",
    }
    temporary = {name: path.with_suffix(path.suffix + ".tmp") for name, path in paths.items()}
    handles = {name: temporary[name].open("w") for name in ("train", "val")}
    writer = pq.ParquetWriter(temporary["index"], INDEX_SCHEMA, compression="zstd")
    exporter = Exporter(args, handles, writer)
    try:
        for path in args.inputs:
            parquet = pq.ParquetFile(path)
            source_row = 0
            for batch in parquet.iter_batches(batch_size=65_536):
                for row in batch.to_pylist():
                    exporter.add(row, path.name, source_row)
                    source_row += 1
                    if exporter.done:
                        break
                if exporter.done:
                    break
            if exporter.done:
                break
        exporter.finish()
    finally:
        writer.close()
        for handle in handles.values():
            handle.close()

    accepted = exporter.counts["accepted_train"] + exporter.counts["accepted_val"]
    if accepted == 0:
        raise RuntimeError("no positions passed the filters")
    for name, path in paths.items():
        os.replace(temporary[name], path)
    manifest = {
        "schema": "archon.lichess-browser-stockfish-value-smoke.v1",
        "source": "lichess/chess-evaluations/versions/14",
        "inputs": [str(path.resolve()) for path in args.inputs],
        "selection": "first PV of max(depth, then knodes) evaluation block",
        "score": "cp_white; exact source semantics; mates excluded",
        "result": "neutral placeholder 0.5; trainer MUST set WDL weight to zero",
        "fen_state": "source four-field FEN plus synthetic halfmove=0 fullmove=1; no repetition",
        "filters": {
            "positions": args.positions,
            "min_depth": args.min_depth,
            "min_knodes": args.min_knodes,
            "max_abs_cp": args.max_abs_cp,
            "standard_chess_only_when_castling_rights_identify_variant": True,
        },
        "split": f"val iff blake2b64(seed || fen4) % 1000 < {args.val_per_mille}",
        "seed": args.seed,
        "counts": dict(exporter.counts),
        "artifacts": {name: str(path.resolve()) for name, path in paths.items()},
    }
    manifest_path = args.output / "manifest.json"
    manifest_tmp = manifest_path.with_suffix(".json.tmp")
    manifest_tmp.write_text(json.dumps(manifest, indent=2, sort_keys=True) + "\n")
    os.replace(manifest_tmp, manifest_path)
    print(json.dumps(manifest, indent=2, sort_keys=True))


if __name__ == "__main__":
    main()
