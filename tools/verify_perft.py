#!/usr/bin/env python3
"""Perft oracle via python-chess — the movegen debugging tool for this repo.

python-chess is an independent legal-move implementation that shares no code
with our engine, so it serves two jobs (design/06-perft-validation.md):

  1. Debugging oracle for future movegen work: print perft or divide counts
     for any position, to compare against `chessengine perft` / `go perft N`
     output while descending into a disagreement (§3.7 context — the
     engine-vs-engine bisection tool `tools/perft_diff.py` lands with the
     UCI loop; until then, this is the reference side of that loop).
  2. Validator for the oracle data itself: `--suite` recomputes the counts in
     tests/data/perft.epd. Every count the suite gates on must be
     independently verified before it can gate the engine (D27: counts are
     facts and are cited exactly).

Requires: python-chess (`pip install chess`; the checked-in counts were
verified with 1.11.2). Everything else is stdlib.

Usage:
  # perft count for one position
  python3 tools/verify_perft.py "<fen>" <depth>

  # per-root-move subtree counts, Stockfish `go perft` format
  # (sorted by move; sort the engine's output before diffing)
  python3 tools/verify_perft.py "<fen>" <depth> --divide

  # re-verify tests/data/perft.epd, skipping counts above the node budget
  python3 tools/verify_perft.py --suite [--max-nodes N] [--jobs J] \
      [--line L] [--epd PATH]

Exit status: 0 iff every computed count matches (suite mode) / always 0 for
one-off counts. Suite counts above --max-nodes are reported as SKIP, never
silently trusted — raise the budget (and --jobs) to check them; big jobs are
split at the root across a process pool, so even the 100M-node entries verify
in minutes on a multicore box. The multi-billion extended tier is still
impractical in Python; those counts carry published provenance in the EPD.
"""

from __future__ import annotations

import argparse
import multiprocessing
import os
import sys
import time
from pathlib import Path

try:
    import chess
except ImportError:  # pragma: no cover
    sys.exit("verify_perft.py needs python-chess: pip install chess")

REPO_ROOT = Path(__file__).resolve().parent.parent
DEFAULT_EPD = REPO_ROOT / "tests" / "data" / "perft.epd"

# Suite mode splits any expected count above this into per-subtree pool tasks
# (see split_frontier). ~20M nodes ≈ 15-20 s per task in python-chess.
SPLIT_TARGET = 20_000_000


def perft(board: chess.Board, depth: int) -> int:
    """Bulk-counted perft: at the horizon the answer is the move-list length."""
    if depth == 0:
        return 1
    if depth == 1:
        return board.legal_moves.count()
    nodes = 0
    for move in board.legal_moves:
        board.push(move)
        nodes += perft(board, depth - 1)
        board.pop()
    return nodes


def divide(board: chess.Board, depth: int) -> list[tuple[str, int]]:
    """Per-root-move subtree counts, sorted by UCI move text."""
    entries = []
    for move in board.legal_moves:
        board.push(move)
        entries.append((move.uci(), perft(board, depth - 1)))
        board.pop()
    return sorted(entries)


# ---------------------------------------------------------------------------
# EPD parsing (mirror of tests/common/mod.rs::parse_perft_epd)
# ---------------------------------------------------------------------------


def parse_epd(text: str, path: Path):
    """Parse the perft EPD format:

        <FEN> ;D1 20 ;D2 400 ;D3 8902

    '#' comments and blank lines skipped; 'D'/'d' accepted; flexible
    whitespace; trailing/doubled semicolons tolerated. Yields
    (line_number, fen, [(depth, nodes), ...]).
    """
    for lineno, raw in enumerate(text.splitlines(), start=1):
        line = raw.strip()
        if not line or line.startswith("#"):
            continue
        fen, *segments = [part.strip() for part in line.split(";")]
        if not fen:
            sys.exit(f"{path}:{lineno}: missing FEN before the first ';'")
        counts = []
        for segment in segments:
            if not segment:
                continue  # trailing or doubled semicolon
            tokens = segment.split()
            if len(tokens) != 2 or tokens[0][:1] not in ("D", "d"):
                sys.exit(f"{path}:{lineno}: bad segment {segment!r}")
            counts.append((int(tokens[0][1:]), int(tokens[1])))
        if not counts:
            sys.exit(f"{path}:{lineno}: no 'D<depth> <nodes>' entries")
        yield lineno, fen, counts


# ---------------------------------------------------------------------------
# Suite verification with a process pool
# ---------------------------------------------------------------------------


def split_frontier(fen: str, depth: int, expected: int) -> list[tuple[str, int]]:
    """Split one perft job into subtree jobs for the pool.

    Expands the position one ply at a time until the *average* expected
    subtree size drops to SPLIT_TARGET (perft totals are exactly the sum of
    the frontier subtree counts, so this is loss-free). Children with no
    legal moves simply contribute nothing, matching perft's recursion.
    """
    board = chess.Board(fen)
    frontier: list[tuple[chess.Board, int]] = [(board, depth)]
    while expected / len(frontier) > SPLIT_TARGET and frontier[0][1] > 2:
        expanded: list[tuple[chess.Board, int]] = []
        for parent, d in frontier:
            for move in parent.legal_moves:
                parent.push(move)
                expanded.append((parent.copy(stack=False), d - 1))
                parent.pop()
        if not expanded:
            break  # no legal moves anywhere: nothing to count
        frontier = expanded
    return [(b.fen(), d) for b, d in frontier]


def _pool_task(task: tuple[int, str, int]) -> tuple[int, int]:
    """(job id, fen, depth) -> (job id, nodes). Top level for pickling."""
    job, fen, depth = task
    return job, perft(chess.Board(fen), depth)


def run_suite(args: argparse.Namespace) -> int:
    text = args.epd.read_text()
    jobs = []  # (job id, lineno, fen, depth, expected)
    skipped = bad_fens = 0
    for lineno, fen, counts in parse_epd(text, args.epd):
        if args.line is not None and lineno != args.line:
            continue
        try:
            chess.Board(fen)
        except ValueError as exc:
            print(f"FAIL line {lineno}: python-chess rejects FEN {fen!r}: {exc}")
            bad_fens += 1
            continue
        for depth, expected in counts:
            if expected > args.max_nodes:
                print(f"SKIP line {lineno} d{depth} {expected:>13,}  (> --max-nodes)")
                skipped += 1
            else:
                jobs.append((len(jobs), lineno, fen, depth, expected))

    # Fan every job out into subtree tasks; biggest first so the pool drains
    # evenly instead of ending on one huge straggler.
    tasks: list[tuple[int, str, int]] = []
    for job, _lineno, fen, depth, expected in sorted(
        jobs, key=lambda j: j[4], reverse=True
    ):
        tasks.extend((job, sub_fen, sub_depth)
                     for sub_fen, sub_depth in split_frontier(fen, depth, expected))

    totals = [0] * len(jobs)
    start = time.perf_counter()
    if tasks:
        with multiprocessing.Pool(args.jobs) as pool:
            for job, nodes in pool.imap_unordered(_pool_task, tasks, chunksize=1):
                totals[job] += nodes
    elapsed = time.perf_counter() - start

    failures = bad_fens
    for job, lineno, fen, depth, expected in jobs:
        if totals[job] == expected:
            print(f"OK   line {lineno} d{depth} {expected:>13,}  {fen}")
        else:
            failures += 1
            print(f"FAIL line {lineno} d{depth} expected {expected:,} "
                  f"got {totals[job]:,}  {fen}")

    total_nodes = sum(j[4] for j in jobs)
    nps = total_nodes / elapsed if elapsed > 0 else 0.0
    print(f"\n{len(jobs)} counts checked ({total_nodes:,} nodes in {elapsed:.1f}s, "
          f"{nps / 1e6:.1f} Mnps aggregate), {skipped} skipped (> --max-nodes), "
          f"{failures} failed")
    return 1 if failures else 0


# ---------------------------------------------------------------------------
# CLI
# ---------------------------------------------------------------------------


def main() -> int:
    parser = argparse.ArgumentParser(
        description="perft/divide oracle and perft.epd verifier (python-chess)",
    )
    parser.add_argument("fen", nargs="?", help="position to count (FEN)")
    parser.add_argument("depth", nargs="?", type=int, help="perft depth")
    parser.add_argument("--divide", action="store_true",
                        help="print per-root-move counts (SF `go perft` format)")
    parser.add_argument("--suite", action="store_true",
                        help="re-verify the perft EPD instead of a single position")
    parser.add_argument("--epd", type=Path, default=DEFAULT_EPD, metavar="PATH",
                        help="EPD file for --suite (default: tests/data/perft.epd)")
    parser.add_argument("--max-nodes", type=int, default=1_000_000, metavar="N",
                        help="--suite: skip counts above N (default: 1,000,000)")
    parser.add_argument("--jobs", type=int, default=os.cpu_count(), metavar="J",
                        help="--suite: process-pool size (default: all CPUs)")
    parser.add_argument("--line", type=int, default=None, metavar="L",
                        help="--suite: only verify the EPD record on this line")
    args = parser.parse_args()

    if args.suite:
        if args.fen is not None or args.depth is not None:
            parser.error("--suite does not take a FEN/depth")
        if args.divide:
            parser.error("--divide needs a FEN and depth, not --suite")
        return run_suite(args)

    if args.fen is None or args.depth is None:
        parser.error('need "<fen>" <depth>, or --suite')
    if args.depth < 0:
        parser.error("depth must be >= 0")
    try:
        board = chess.Board(args.fen)
    except ValueError as exc:
        sys.exit(f"bad FEN {args.fen!r}: {exc}")

    start = time.perf_counter()
    if args.divide and args.depth >= 1:
        entries = divide(board, args.depth)
        total = sum(nodes for _, nodes in entries)
        for uci, nodes in entries:
            print(f"{uci}: {nodes}")
        print(f"\nNodes searched: {total}")
    else:
        print(perft(board, args.depth))
    elapsed = time.perf_counter() - start
    print(f"[{elapsed:.2f}s]", file=sys.stderr)
    return 0


if __name__ == "__main__":
    sys.exit(main())
