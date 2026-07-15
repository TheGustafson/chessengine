#!/usr/bin/env python3
"""Collect bounded Archon LMR telemetry on deterministic full-FEN positions."""

from __future__ import annotations

import argparse
import hashlib
import json
import os
from pathlib import Path
import shutil

import chess
import chess.engine

from root_move_oracle import read_fens


SCHEMA = "archon.search-trace-corpus.v1"


def sha256(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as source:
        for block in iter(lambda: source.read(1 << 20), b""):
            digest.update(block)
    return digest.hexdigest()


def atomic_json(path: Path, value: object) -> None:
    temporary = path.with_suffix(path.suffix + ".tmp")
    temporary.write_text(json.dumps(value, indent=2, sort_keys=True) + "\n", encoding="utf-8")
    temporary.replace(path)


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--fen-file", type=Path, required=True)
    parser.add_argument("--output", type=Path, required=True)
    parser.add_argument("--engine", type=Path, required=True)
    parser.add_argument("--eval", type=Path, required=True)
    parser.add_argument("--max-positions", type=int, default=64)
    parser.add_argument("--seed", type=int, default=0xA4C0_4101)
    parser.add_argument("--depth", type=int, default=8)
    parser.add_argument("--hash-mb", type=int, default=16)
    parser.add_argument("--timeout-seconds", type=float, default=120.0)
    parser.add_argument("--sample-log2", type=int, default=6)
    parser.add_argument("--min-depth", type=int, default=3)
    parser.add_argument("--max-rows-per-go", type=int, default=512)
    args = parser.parse_args()
    for name in ("max_positions", "depth", "hash_mb", "max_rows_per_go"):
        if getattr(args, name) <= 0:
            parser.error(f"--{name.replace('_', '-')} must be positive")
    if not 0 <= args.sample_log2 <= 63:
        parser.error("--sample-log2 must be in 0..63")
    if args.min_depth < 2:
        parser.error("--min-depth must be at least 2")

    engine_path = args.engine.resolve()
    eval_path = args.eval.resolve()
    source_path = args.fen_file.resolve()
    for label, path in (("engine", engine_path), ("eval", eval_path), ("source", source_path)):
        if not path.is_file():
            parser.error(f"{label} file does not exist: {path}")
    if not os.access(engine_path, os.X_OK):
        parser.error(f"engine is not executable: {engine_path}")
    output = args.output.resolve()
    if output.exists():
        parser.error(f"refusing to overwrite output: {output}")
    output.mkdir(parents=True)
    shutil.copy2(Path(__file__).resolve(), output / "generator.snapshot.py")
    trace_path = output / "observations.jsonl"
    positions = read_fens(source_path, args.max_positions, args.seed)

    environment = dict(os.environ)
    environment.update(
        {
            "ARCHON_SEARCH_TRACE_PATH": str(trace_path),
            "ARCHON_SEARCH_TRACE_SAMPLE_LOG2": str(args.sample_log2),
            "ARCHON_SEARCH_TRACE_MIN_DEPTH": str(args.min_depth),
            "ARCHON_SEARCH_TRACE_MAX_ROWS": str(args.max_rows_per_go),
        }
    )
    records = []
    try:
        engine = chess.engine.SimpleEngine.popen_uci(
            str(engine_path), timeout=args.timeout_seconds, env=environment
        )
        try:
            for required in ("Threads", "Hash", "EvalFile"):
                if required not in engine.options:
                    raise RuntimeError(f"engine does not expose required option {required}")
            engine.configure({"Threads": 1, "Hash": args.hash_mb, "EvalFile": str(eval_path)})
            for position in positions:
                board = chess.Board(position.fen)
                info = engine.analyse(
                    board,
                    chess.engine.Limit(depth=args.depth),
                    game=object(),
                    info=chess.engine.INFO_ALL,
                )
                score = info.get("score")
                score_cp = None if score is None else score.pov(board.turn).score(mate_score=30000)
                records.append(
                    {
                        "source_id": position.source_id,
                        "source_detail": position.source_detail,
                        "fen": position.fen,
                        "depth": int(info.get("depth", 0)),
                        "seldepth": int(info.get("seldepth", 0)),
                        "nodes": int(info.get("nodes", 0)),
                        "time_ms": int(round(float(info.get("time", 0.0)) * 1000)),
                        "score_cp": score_cp,
                        "pv": [move.uci() for move in info.get("pv", [])],
                    }
                )
        finally:
            engine.quit()

        go_records = 0
        trace_rows = 0
        with trace_path.open(encoding="utf-8") as source:
            for line in source:
                item = json.loads(line)
                go_records += item.get("kind") == "go"
                trace_rows += item.get("kind") == "row"
        if go_records != len(positions):
            raise RuntimeError(f"expected {len(positions)} trace go records, found {go_records}")
        generator = Path(__file__).resolve()
        manifest = {
            "schema": SCHEMA,
            "generator": {"path": str(generator), "sha256": sha256(generator)},
            "generator_snapshot": {
                "path": "generator.snapshot.py",
                "sha256": sha256(output / "generator.snapshot.py"),
            },
            "engine": {"path": str(engine_path), "sha256": sha256(engine_path)},
            "eval": {"path": str(eval_path), "sha256": sha256(eval_path)},
            "source": {"path": str(source_path), "sha256": sha256(source_path)},
            "selection": {
                "seed": args.seed,
                "max_positions": args.max_positions,
                "method": "smallest blake2b-64(seed:line_number:canonical_fen)",
            },
            "search": {"depth": args.depth, "threads": 1, "hash_mb": args.hash_mb},
            "trace": {
                "path": trace_path.name,
                "sha256": sha256(trace_path),
                "rows": trace_rows,
                "go_records": go_records,
                "sample_log2": args.sample_log2,
                "min_depth": args.min_depth,
                "max_rows_per_go": args.max_rows_per_go,
            },
            "positions": records,
        }
        atomic_json(output / "manifest.json", manifest)
        print(f"wrote {trace_rows} observations from {go_records} positions to {output}")
    except BaseException:
        shutil.rmtree(output, ignore_errors=True)
        raise


if __name__ == "__main__":
    main()
