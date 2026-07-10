#!/usr/bin/env python3
"""Differential perft testing with divide-descend bisection (design/06 §3.7).

Compares our engine's per-root-move perft counts against an independent
reference at every level of a disagreement, descending into the first move
whose subtree count differs until it reaches a position where the *move
sets* differ (or depth 1). The report is a minimal reproducer: a FEN plus
the move sequence reaching the exact divergence — paste it into a unit test.

Engine faces:

  ours      — the `chessengine perft <depth> [fen]` CLI subcommand
              (design/06 §3.2 SF-format output). P4 note: once the UCI loop
              grows `go perft N`, this switches to the same UCI driver as
              the reference so both sides run over one protocol, per the
              doc's original §3.7 shape.
  reference — python-chess by default (no external binary needed; same
              independent oracle as tools/verify_perft.py), or any UCI
              engine supporting SF-style `go perft` via --ref/$PERFT_REF
              (e.g. a locally built Stockfish).

Requires: python-chess (`pip install chess`); everything else is stdlib
(D15: tooling only, never a crate dependency).

Usage:
  # one position
  python3 tools/perft_diff.py --fen "FEN" --depth N

  # the whole EPD suite at every depth whose count fits the node budget
  python3 tools/perft_diff.py --suite [--max-nodes N]

  # random-walk fuzzing: diff at depth D, play a random legal move, repeat
  python3 tools/perft_diff.py --walk 200 --depth 4 [--seed S] [--fen "FEN"]

Exit status: 0 iff every comparison agreed.
"""

from __future__ import annotations

import argparse
import os
import random
import subprocess
import sys
from pathlib import Path

try:
    import chess
except ImportError:  # pragma: no cover
    sys.exit("perft_diff.py needs python-chess: pip install chess")

REPO_ROOT = Path(__file__).resolve().parent.parent
DEFAULT_EPD = REPO_ROOT / "tests" / "data" / "perft.epd"
DEFAULT_ENGINE = REPO_ROOT / "target" / "release" / "chessengine"
STARTPOS = chess.STARTING_FEN


def position_after(fen: str, moves: list[str]) -> chess.Board:
    board = chess.Board(fen)
    for uci in moves:
        board.push_uci(uci)
    return board


def parse_divide(text: str, context: str) -> dict[str, int]:
    """Parse SF-format divide output: `move: count` lines + total line."""
    counts: dict[str, int] = {}
    total = None
    for line in text.splitlines():
        line = line.strip()
        if not line:
            continue
        if line.startswith("Nodes searched:"):
            total = int(line.split(":", 1)[1])
        elif ":" in line:
            move, count = line.split(":", 1)
            counts[move.strip()] = int(count)
    if total is None:
        sys.exit(f"{context}: no 'Nodes searched:' line in output:\n{text}")
    if sum(counts.values()) != total:
        sys.exit(f"{context}: divide entries sum to {sum(counts.values())}, "
                 f"total says {total}")
    return counts


class CliEngine:
    """Our engine through the `chessengine perft` argv subcommand."""

    def __init__(self, binary: Path):
        if not binary.is_file():
            sys.exit(f"engine binary not found: {binary}\n"
                     f"build it first: cargo build --release")
        self.binary = binary

    def divide(self, fen: str, moves: list[str], depth: int) -> dict[str, int]:
        # The CLI takes a bare FEN; replay the move prefix here. python-chess
        # writes the ep field only when an ep capture is legal, matching our
        # D9 recording rule, so the FEN round-trip is faithful.
        position_fen = position_after(fen, moves).fen()
        result = subprocess.run(
            [str(self.binary), "perft", str(depth), position_fen],
            capture_output=True, text=True, check=True,
        )
        return parse_divide(result.stdout, f"{self.binary} perft {depth}")


class UciEngine:
    """A reference engine over UCI `position fen ... moves ...` + `go perft`."""

    def __init__(self, binary: str):
        self.proc = subprocess.Popen(
            [binary], stdin=subprocess.PIPE, stdout=subprocess.PIPE, text=True,
        )
        self._send("uci")
        self._read_until("uciok")

    def _send(self, line: str) -> None:
        assert self.proc.stdin is not None
        self.proc.stdin.write(line + "\n")
        self.proc.stdin.flush()

    def _read_until(self, token: str) -> list[str]:
        assert self.proc.stdout is not None
        lines = []
        while True:
            line = self.proc.stdout.readline()
            if not line:
                sys.exit("reference engine closed its pipe")
            lines.append(line.rstrip("\n"))
            if lines[-1].startswith(token):
                return lines

    def divide(self, fen: str, moves: list[str], depth: int) -> dict[str, int]:
        position = f"position fen {fen}"
        if moves:
            position += " moves " + " ".join(moves)
        self._send(position)
        self._send(f"go perft {depth}")
        lines = self._read_until("Nodes searched:")
        return parse_divide("\n".join(lines), "reference go perft")

    def close(self) -> None:
        try:
            self._send("quit")
            self.proc.wait(timeout=5)
        except Exception:
            self.proc.kill()


class PythonChessEngine:
    """python-chess as the reference: an independent legal-move oracle."""

    def divide(self, fen: str, moves: list[str], depth: int) -> dict[str, int]:
        board = position_after(fen, moves)
        counts = {}
        for move in board.legal_moves:
            board.push(move)
            counts[move.uci()] = self._perft(board, depth - 1)
            board.pop()
        return counts

    def _perft(self, board: chess.Board, depth: int) -> int:
        if depth == 0:
            return 1
        if depth == 1:
            return board.legal_moves.count()
        nodes = 0
        for move in board.legal_moves:
            board.push(move)
            nodes += self._perft(board, depth - 1)
            board.pop()
        return nodes


def diff(ours, ref, fen: str, moves: list[str], depth: int) -> bool:
    """The design/06 §3.7 recursion. Returns True iff the engines agree;
    on disagreement prints a minimal reproducer and returns False."""
    our_counts = ours.divide(fen, moves, depth)
    ref_counts = ref.divide(fen, moves, depth)
    if our_counts == ref_counts:
        return True

    missing = sorted(set(ref_counts) - set(our_counts))
    extra = sorted(set(our_counts) - set(ref_counts))
    if missing or extra or depth == 1:
        print("DIVERGENCE — minimal reproducer:")
        print(f"  fen:   {fen}")
        print(f"  moves: {' '.join(moves) if moves else '(none)'}")
        print(f"  position: {position_after(fen, moves).fen()}")
        print(f"  depth: {depth}")
        if missing:
            print(f"  we fail to generate: {missing}")
        if extra:
            print(f"  we generate illegally: {extra}")
        deltas = {m: (our_counts.get(m), ref_counts.get(m))
                  for m in sorted(set(our_counts) & set(ref_counts))
                  if our_counts[m] != ref_counts[m]}
        for move, (got, want) in deltas.items():
            print(f"  count delta {move}: ours {got}, reference {want}")
        return False

    # Same move set, some subtree count off: descend into the divergence.
    culprit = next(m for m in sorted(our_counts)
                   if our_counts[m] != ref_counts[m])
    print(f"  descending: {' '.join(moves + [culprit])} "
          f"(ours {our_counts[culprit]}, reference {ref_counts[culprit]})")
    return diff(ours, ref, fen, moves + [culprit], depth - 1)


def parse_epd(path: Path):
    """(fen, [(depth, nodes)]) records; format as in tools/verify_perft.py."""
    for raw in path.read_text().splitlines():
        line = raw.strip()
        if not line or line.startswith("#"):
            continue
        fen, *segments = [part.strip() for part in line.split(";")]
        counts = [(int(seg.split()[0][1:]), int(seg.split()[1]))
                  for seg in segments if seg]
        yield fen, counts


def main() -> int:
    parser = argparse.ArgumentParser(
        description="differential perft bisection (ours vs reference)")
    parser.add_argument("--engine", type=Path,
                        default=Path(os.environ.get("CHESSENGINE", DEFAULT_ENGINE)),
                        help="our binary (default: target/release/chessengine)")
    parser.add_argument("--ref", default=os.environ.get("PERFT_REF"),
                        help="reference UCI engine with SF-style `go perft` "
                             "(default: python-chess in-process)")
    parser.add_argument("--fen", help="position to diff (default: startpos)")
    parser.add_argument("--depth", type=int, default=4, help="divide depth")
    parser.add_argument("--suite", action="store_true",
                        help="diff every EPD record at every in-budget depth")
    parser.add_argument("--epd", type=Path, default=DEFAULT_EPD)
    parser.add_argument("--max-nodes", type=int, default=1_000_000,
                        help="--suite: skip depths whose expected count "
                             "exceeds this (default: 1,000,000)")
    parser.add_argument("--walk", type=int, metavar="PLIES",
                        help="random-walk fuzzing for PLIES plies")
    parser.add_argument("--seed", type=int, default=0, help="--walk RNG seed")
    args = parser.parse_args()

    ours = CliEngine(args.engine)
    ref = UciEngine(args.ref) if args.ref else PythonChessEngine()
    failures = 0
    try:
        if args.suite:
            for fen, counts in parse_epd(args.epd):
                for depth, expected in counts:
                    if expected > args.max_nodes:
                        continue
                    print(f"suite: d{depth} {fen}")
                    if not diff(ours, ref, fen, [], depth):
                        failures += 1
        elif args.walk is not None:
            rng = random.Random(args.seed)
            fen = args.fen or STARTPOS
            moves: list[str] = []
            for ply in range(args.walk):
                print(f"walk ply {ply}: {' '.join(moves) or '(root)'}")
                if not diff(ours, ref, fen, moves, args.depth):
                    failures += 1
                    break
                legal = sorted(ref.divide(fen, moves, 1))
                if not legal:
                    print("  no legal moves (mate/stalemate); walk ends")
                    break
                moves.append(rng.choice(legal))
        else:
            fen = args.fen or STARTPOS
            if not diff(ours, ref, fen, [], args.depth):
                failures += 1
            else:
                print(f"AGREE: d{args.depth} {fen}")
    finally:
        if isinstance(ref, UciEngine):
            ref.close()

    if failures:
        print(f"\n{failures} divergence(s) found")
        return 1
    print("\nall comparisons agree")
    return 0


if __name__ == "__main__":
    sys.exit(main())
