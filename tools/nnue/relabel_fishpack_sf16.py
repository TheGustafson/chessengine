#!/usr/bin/env python3
"""Positions-vs-labels experiment: relabel fishpack positions with SF16.

Reproduces gen_sf_data.py's labeling EXACTLY (same SF16 binary, Threads=1,
Hash=16, fixed 5000-node analyse, White-relative UCI cp, and the same
curation: past-opening ply, not in check, quiet SF bestmove, |cp| < 2500,
non-mate, dedup). Input: TSV `FEN\\tresult_white\\tply` from bulk_fens.
Output: sf-100-format text shards `FEN | cp | result` under --out.

Resumable: input is split round-robin by worker; each worker writes 20k-line
chunks with a .done marker and skips completed chunks on restart.
"""

from __future__ import annotations

import argparse
import json
import multiprocessing as mp
import os
import sys
import time
from pathlib import Path

import chess
import chess.engine
import chess.polyglot

CHUNK_LINES = 20_000
NODES = 5000
SCORE_CAP = 2500
MIN_PLY = 12


def worker_chunks(path: str, workers: int, wid: int):
    """Stream this worker's lines (row_index % workers == wid) grouped into
    CHUNK_LINES-sized chunks, without holding the file in memory."""
    chunk, chunk_idx, local_idx = [], 0, 0
    with open(path) as fh:
        for i, line in enumerate(fh):
            if i % workers != wid:
                continue
            chunk.append(line.rstrip("\n"))
            local_idx += 1
            if len(chunk) == CHUNK_LINES:
                yield chunk_idx, chunk
                chunk, chunk_idx = [], chunk_idx + 1
    if chunk:
        yield chunk_idx, chunk


def worker(wid: int, args, counters) -> None:
    os.nice(args.nice)
    out_dir = Path(args.out)
    engine = chess.engine.SimpleEngine.popen_uci(args.sf)
    engine.configure({"Threads": 1, "Hash": 16})
    seen: set[int] = set()
    for chunk_idx, chunk in worker_chunks(args.input, args.workers, wid):
        done_marker = out_dir / f"relabel_{wid}_{chunk_idx}.done"
        shard = out_dir / f"relabel_{wid}_{chunk_idx}.txt"
        if done_marker.exists():
            continue
        rows = []
        for line in chunk:
            parts = line.split("\t")
            if len(parts) != 3:
                continue
            fen, result, ply = parts[0], parts[1], int(parts[2])
            if ply < MIN_PLY:
                continue
            try:
                board = chess.Board(fen)
            except ValueError:
                continue
            if board.is_check() or not board.is_valid():
                continue
            key = chess.polyglot.zobrist_hash(board)
            if key in seen:
                continue
            seen.add(key)
            try:
                info = engine.analyse(board, chess.engine.Limit(nodes=NODES))
            except chess.engine.EngineError:
                engine.quit()
                engine = chess.engine.SimpleEngine.popen_uci(args.sf)
                engine.configure({"Threads": 1, "Hash": 16})
                continue
            score = info["score"].white()
            if score.is_mate():
                continue
            cp = score.score()
            if cp is None or abs(cp) >= SCORE_CAP:
                continue
            pv = info.get("pv")
            if not pv:
                continue
            best = pv[0]
            if board.is_capture(best) or best.promotion is not None:
                continue
            rows.append(f"{fen} | {cp} | {result}")
            with counters.get_lock():
                counters[wid] += 1
        tmp = shard.with_suffix(".tmp")
        tmp.write_text("\n".join(rows) + ("\n" if rows else ""))
        tmp.rename(shard)
        done_marker.touch()
    engine.quit()


def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--input", required=True)
    parser.add_argument("--sf", default="engines/stockfish-datagen")
    parser.add_argument("--out", required=True)
    parser.add_argument("--workers", type=int, default=12)
    parser.add_argument("--nice", type=int, default=10)
    args = parser.parse_args()
    out_dir = Path(args.out)
    out_dir.mkdir(parents=True, exist_ok=True)
    manifest = {
        "experiment": "positions-vs-labels (value-foundation-state.md)",
        "input": args.input, "sf": args.sf, "nodes": NODES,
        "score_cap": SCORE_CAP, "min_ply": MIN_PLY, "workers": args.workers,
        "labeling": "gen_sf_data.py-faithful: Threads=1 Hash=16 nodes=5000, "
                    "white-POV UCI cp, quiet-bestmove/check/mate/cap filters, "
                    "per-worker zobrist dedup; bounds NOT filtered (faithful)",
        "started_utc": time.strftime("%Y-%m-%dT%H:%M:%SZ", time.gmtime()),
    }
    (out_dir / "manifest.json").write_text(json.dumps(manifest, indent=2))
    counters = mp.Array("l", args.workers)
    procs = [mp.Process(target=worker, args=(w, args, counters)) for w in range(args.workers)]
    for p in procs:
        p.start()
    start = time.time()
    while any(p.is_alive() for p in procs):
        time.sleep(30)
        total = sum(counters)
        rate = total / max(1.0, time.time() - start)
        print(f"[relabel {time.strftime('%H:%M:%S')}] labeled={total} rate={rate:.0f}/s", flush=True)
    for p in procs:
        p.join()
    print(f"[relabel] COMPLETE labeled={sum(counters)}", flush=True)
    return 0


if __name__ == "__main__":
    sys.exit(main())
