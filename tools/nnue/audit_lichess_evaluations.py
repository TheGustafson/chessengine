#!/usr/bin/env python3
"""Streaming structural audit for the flattened Lichess evaluation Parquet."""

from __future__ import annotations

import argparse
import json
from collections import Counter
from pathlib import Path

import chess
import pyarrow.parquet as pq


def percentile(counter: Counter[int], fraction: float) -> int | None:
    total = sum(counter.values())
    if not total:
        return None
    target = max(1, int(total * fraction + 0.999999))
    cumulative = 0
    for value, count in sorted(counter.items()):
        cumulative += count
        if cumulative >= target:
            return value
    raise AssertionError("unreachable")


def try_line(fen4: str, line: str, chess960: bool) -> tuple[bool, int, str | None]:
    try:
        board = chess.Board(f"{fen4} 0 1", chess960=chess960)
    except ValueError as error:
        return False, 0, f"fen:{error}"
    plies = 0
    for token in line.split():
        try:
            move = board.parse_uci(token)
        except ValueError as error:
            return False, plies, f"move:{token}:{error}"
        board.push(move)
        plies += 1
    return True, plies, None


def line_is_legal(fen4: str, line: str) -> tuple[bool, int, str | None]:
    standard = try_line(fen4, line, chess960=False)
    if standard[0]:
        return standard
    chess960 = try_line(fen4, line, chess960=True)
    if chess960[0]:
        return chess960
    return False, max(standard[1], chess960[1]), f"standard={standard[2]};chess960={chess960[2]}"


class Audit:
    def __init__(self, legal_sample_limit: int) -> None:
        self.rows = 0
        self.positions = 0
        self.eval_blocks = 0
        self.cp_rows = 0
        self.mate_rows = 0
        self.both_null = 0
        self.both_present = 0
        self.fen_field_counts: Counter[int] = Counter()
        self.depths: Counter[int] = Counter()
        self.knodes_log2: Counter[int] = Counter()
        self.selected_depths: Counter[int] = Counter()
        self.selected_knodes_log2: Counter[int] = Counter()
        self.selected_abs_cp_buckets: Counter[int] = Counter()
        self.rows_per_position: Counter[int] = Counter()
        self.evals_per_position: Counter[int] = Counter()
        self.pvs_per_eval: Counter[int] = Counter()
        self.first_eval_is_max_depth = 0
        self.first_eval_is_selected = 0
        self.max_depth_ties = 0
        self.repeated_eval_key_blocks = 0
        self.cp_order_blocks = Counter()
        self.position_eval_blocks: list[dict] = []
        self.current_fen: str | None = None
        self.current_eval_key: tuple[int, int] | None = None
        self.current_eval_rows: list[dict] = []
        self.position_rows = 0
        self.seen_eval_keys: set[tuple[int, int]] = set()
        self.legal_sample_limit = legal_sample_limit
        self.legal_sampled = 0
        self.legal_lines = 0
        self.illegal_lines = 0
        self.illegal_examples: list[dict] = []
        self.castling_positions = 0
        self.en_passant_positions = 0
        self.side_to_move: Counter[str] = Counter()

    def finish_eval(self) -> None:
        if not self.current_eval_rows:
            return
        rows = self.current_eval_rows
        self.eval_blocks += 1
        self.pvs_per_eval[len(rows)] += 1
        cps = [row["cp"] for row in rows if row["cp"] is not None]
        if len(cps) >= 2:
            side = self.current_fen.split()[1]  # type: ignore[union-attr]
            expected = cps == sorted(cps, reverse=(side == "w"))
            self.cp_order_blocks["perspective_consistent" if expected else "other"] += 1
        self.position_eval_blocks.append(
            {
                "depth": rows[0]["depth"],
                "knodes": rows[0]["knodes"],
                "first": rows[0],
            }
        )
        self.current_eval_rows = []

    def finish_position(self) -> None:
        if self.current_fen is None:
            return
        self.finish_eval()
        blocks = self.position_eval_blocks
        self.positions += 1
        self.rows_per_position[self.position_rows] += 1
        self.evals_per_position[len(blocks)] += 1
        maximum_depth = max(block["depth"] for block in blocks)
        maximum_blocks = [block for block in blocks if block["depth"] == maximum_depth]
        if blocks[0]["depth"] == maximum_depth:
            self.first_eval_is_max_depth += 1
        if len(maximum_blocks) > 1:
            self.max_depth_ties += 1
        selected = max(blocks, key=lambda block: (block["depth"], block["knodes"]))
        selected_row = selected["first"]
        self.selected_depths[selected["depth"]] += 1
        self.selected_knodes_log2[max(0, int(selected["knodes"]).bit_length() - 1)] += 1
        if selected_row["cp"] is not None:
            self.selected_abs_cp_buckets[min(abs(int(selected_row["cp"])) // 100, 30)] += 1
        if blocks[0] is selected:
            self.first_eval_is_selected += 1

        fields = self.current_fen.split()
        self.fen_field_counts[len(fields)] += 1
        if len(fields) >= 4:
            self.side_to_move[fields[1]] += 1
            self.castling_positions += fields[2] != "-"
            self.en_passant_positions += fields[3] != "-"
        if self.legal_sampled < self.legal_sample_limit:
            row = selected["first"]
            valid, plies, error = line_is_legal(self.current_fen, row["line"])
            self.legal_sampled += 1
            self.legal_lines += valid
            self.illegal_lines += not valid
            if not valid and len(self.illegal_examples) < 20:
                self.illegal_examples.append(
                    {"fen": self.current_fen, "line": row["line"], "plies": plies, "error": error}
                )

        self.position_eval_blocks = []
        self.position_rows = 0
        self.seen_eval_keys = set()

    def add(self, row: dict) -> None:
        fen = row["fen"]
        if self.current_fen != fen:
            self.finish_position()
            self.current_fen = fen
            self.current_eval_key = None
        key = (row["depth"], row["knodes"])
        if self.current_eval_key != key:
            self.finish_eval()
            if key in self.seen_eval_keys:
                self.repeated_eval_key_blocks += 1
            self.seen_eval_keys.add(key)
            self.current_eval_key = key
        self.current_eval_rows.append(row)
        self.position_rows += 1
        self.rows += 1
        self.depths[row["depth"]] += 1
        self.knodes_log2[max(0, int(row["knodes"]).bit_length() - 1)] += 1
        self.cp_rows += row["cp"] is not None
        self.mate_rows += row["mate"] is not None
        self.both_null += row["cp"] is None and row["mate"] is None
        self.both_present += row["cp"] is not None and row["mate"] is not None

    def summary(self, path: Path, max_rows: int | None) -> dict:
        self.finish_position()
        return {
            "schema_version": 1,
            "input": str(path),
            "max_rows": max_rows,
            "rows": self.rows,
            "positions": self.positions,
            "evaluation_blocks": self.eval_blocks,
            "cp_rows": self.cp_rows,
            "mate_rows": self.mate_rows,
            "both_label_fields_null": self.both_null,
            "both_label_fields_present": self.both_present,
            "fen_field_counts": dict(self.fen_field_counts),
            "side_to_move": dict(self.side_to_move),
            "castling_positions": self.castling_positions,
            "en_passant_positions": self.en_passant_positions,
            "first_evaluation_is_max_depth": self.first_eval_is_max_depth,
            "first_evaluation_is_selected_by_depth_then_nodes": self.first_eval_is_selected,
            "max_depth_tie_positions": self.max_depth_ties,
            "repeated_noncontiguous_eval_key_blocks": self.repeated_eval_key_blocks,
            "cp_multipv_order_blocks": dict(self.cp_order_blocks),
            "legal_line_sample": {
                "sampled": self.legal_sampled,
                "legal": self.legal_lines,
                "illegal": self.illegal_lines,
                "examples": self.illegal_examples,
            },
            "row_depth_percentiles": {
                str(percent): percentile(self.depths, percent / 100)
                for percent in (1, 5, 25, 50, 75, 95, 99)
            },
            "selected_depth_percentiles": {
                str(percent): percentile(self.selected_depths, percent / 100)
                for percent in (1, 5, 25, 50, 75, 95, 99)
            },
            "rows_per_position_percentiles": {
                str(percent): percentile(self.rows_per_position, percent / 100)
                for percent in (50, 75, 95, 99, 100)
            },
            "evaluations_per_position_percentiles": {
                str(percent): percentile(self.evals_per_position, percent / 100)
                for percent in (50, 75, 95, 99, 100)
            },
            "pvs_per_evaluation_percentiles": {
                str(percent): percentile(self.pvs_per_eval, percent / 100)
                for percent in (50, 75, 95, 99, 100)
            },
            "knodes_floor_log2_counts": dict(self.knodes_log2),
            "selected_knodes_floor_log2_counts": dict(self.selected_knodes_log2),
            "selected_abs_cp_100cp_bucket_counts": dict(self.selected_abs_cp_buckets),
        }


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("input", type=Path)
    parser.add_argument("--max-rows", type=int)
    parser.add_argument("--legal-sample", type=int, default=10_000)
    parser.add_argument("--output", type=Path)
    args = parser.parse_args()

    parquet = pq.ParquetFile(args.input)
    audit = Audit(args.legal_sample)
    stop = False
    for batch in parquet.iter_batches(batch_size=65_536):
        for row in batch.to_pylist():
            audit.add(row)
            if args.max_rows is not None and audit.rows >= args.max_rows:
                stop = True
                break
        if stop:
            break
    result = audit.summary(args.input, args.max_rows)
    encoded = json.dumps(result, indent=2, sort_keys=True) + "\n"
    if args.output:
        args.output.parent.mkdir(parents=True, exist_ok=True)
        temporary = args.output.with_suffix(args.output.suffix + ".tmp")
        temporary.write_text(encoded)
        temporary.replace(args.output)
    print(encoded, end="")


if __name__ == "__main__":
    main()
