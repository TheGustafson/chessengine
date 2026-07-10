#!/usr/bin/env python3
"""Tier-1.5 tactical sanity runner (design/14 §3.1) — logic behind tools/tactics.sh.

Feeds an EPD suite to a UCI engine (`position fen` + `go movetime`/`go depth`
via python-chess), scores bm/am hits, and optionally appends the score to
docs/strength-log.md. This tier is a SANITY harness: it never gates a merge,
but it is required before milestone declarations (M2+), where an unexplained
drop against the tracked baseline blocks the milestone (design/14 §3.9 item 8).

Suites:
  tests/data/tactics-smoke.epd    committed, 28 verified positions (default)
  engines/suites/*.epd            WAC + mate-in-N, fetched by
                                  `tools/tactics.sh --fetch-wac` (never
                                  vendored, D15)

Requires: pip install chess  (same dependency as tools/verify_perft.py).

EPD parsing is deliberately tolerant: lines starting with '#' are comments,
and records whose opcodes python-chess cannot parse strictly (e.g. the
`ce +M2;` fields in some published mate suites) fall back to a regex parse
of just the bm/am/id opcodes.
"""

from __future__ import annotations

import argparse
import datetime
import re
import subprocess
import sys
from pathlib import Path

import chess
import chess.engine

REPO = Path(__file__).resolve().parent.parent


def parse_epd_line(line: str):
    """Return (board, bm_moves, am_moves, ident) or None for non-records."""
    line = line.strip()
    if not line or line.startswith("#"):
        return None
    fields = line.split()
    if len(fields) < 4:
        return None
    board = chess.Board(" ".join(fields[:4]) + " 0 1")
    ops = line.split(maxsplit=4)[4] if len(fields) > 4 else ""

    def moves_for(opcode: str) -> list[chess.Move]:
        m = re.search(rf"(?:^|;)\s*{opcode}\s+([^;]+);", " " + ops)
        if not m:
            return []
        out = []
        for tok in m.group(1).split():
            try:
                out.append(board.parse_san(tok))
            except ValueError:
                print(f"warning: unparsable {opcode} move {tok!r}", file=sys.stderr)
        return out

    ident_m = re.search(r'id\s+"([^"]*)"', ops)
    ident = ident_m.group(1) if ident_m else fields[0]
    return board, moves_for("bm"), moves_for("am"), ident


def git_rev() -> str:
    try:
        return subprocess.run(
            ["git", "-C", str(REPO), "rev-parse", "--short", "HEAD"],
            capture_output=True, text=True, check=True,
        ).stdout.strip()
    except (subprocess.CalledProcessError, FileNotFoundError):
        return "unknown"


def append_log(log_path: Path, row: str) -> None:
    header = (
        "| date | commit | kind | detail | result |\n"
        "|---|---|---|---|---|\n"
    )
    if log_path.exists():
        text = log_path.read_text()
        if not text.endswith("\n"):
            text += "\n"
        log_path.write_text(text + row)
    else:
        log_path.write_text(
            "# Strength log\n\n"
            "Audit trail behind every strength claim (design/14 §5, D25):\n"
            "SPRT verdicts, gauntlet ratings, and tier-1.5 tactical-suite\n"
            "scores, one row per run, appended by tools/sprt.sh,\n"
            "tools/gauntlet.sh, and tools/tactics.sh --log. Never edit or\n"
            "delete history; corrections get a new row.\n\n" + header + row
        )
    print(f"logged to {log_path}")


def main() -> int:
    p = argparse.ArgumentParser(description=__doc__.splitlines()[0])
    p.add_argument("engine", help="path to the UCI engine binary")
    p.add_argument("suite", nargs="?",
                   default=str(REPO / "tests/data/tactics-smoke.epd"),
                   help="EPD suite (default: the committed smoke suite)")
    lim = p.add_mutually_exclusive_group()
    lim.add_argument("--movetime", type=int, default=1000, metavar="MS",
                     help="ms per position (default 1000)")
    lim.add_argument("--depth", type=int, metavar="D",
                     help="fixed depth instead of movetime")
    p.add_argument("--hash", type=int, default=16, metavar="MB",
                   help="Hash option in MiB (default 16)")
    p.add_argument("--log", action="store_true",
                   help="append the score to docs/strength-log.md")
    p.add_argument("-v", "--verbose", action="store_true",
                   help="print every position, not just failures")
    args = p.parse_args()

    suite = Path(args.suite)
    records = []
    for n, line in enumerate(suite.read_text().splitlines(), 1):
        try:
            rec = parse_epd_line(line)
        except ValueError as e:
            print(f"warning: {suite.name}:{n}: skipped ({e})", file=sys.stderr)
            continue
        if rec is not None:
            records.append(rec)
    if not records:
        print(f"error: no EPD records in {suite}", file=sys.stderr)
        return 2

    limit = (chess.engine.Limit(depth=args.depth) if args.depth is not None
             else chess.engine.Limit(time=args.movetime / 1000))
    limit_desc = (f"depth {args.depth}" if args.depth is not None
                  else f"movetime {args.movetime}ms")

    engine = chess.engine.SimpleEngine.popen_uci(args.engine)
    try:
        engine.configure({"Hash": args.hash})
    except chess.engine.EngineError:
        pass  # engines without a Hash option are fine
    hits = 0
    try:
        for i, (board, bm, am, ident) in enumerate(records):
            # A fresh `game` per position => ucinewgame between positions.
            result = engine.play(board, limit, game=i)
            move = result.move
            ok = (move in bm if bm else True) and move not in am
            hits += ok
            if args.verbose or not ok:
                status = "ok  " if ok else "FAIL"
                want = " ".join(board.san(m) for m in bm) or "(am-only)"
                print(f"{status} {ident:28s} played {board.san(move):8s} "
                      f"expected {want}")
    finally:
        engine.quit()

    total = len(records)
    pct = 100.0 * hits / total
    print(f"\n{suite.name}: {hits}/{total} ({pct:.0f}%) at {limit_desc}")

    if args.log:
        date = datetime.date.today().isoformat()
        row = (f"| {date} | {git_rev()} | tactics | {suite.name}, "
               f"{limit_desc}, Hash={args.hash} | {hits}/{total} ({pct:.0f}%) |\n")
        append_log(REPO / "docs/strength-log.md", row)
    return 0


if __name__ == "__main__":
    sys.exit(main())
