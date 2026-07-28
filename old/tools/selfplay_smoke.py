#!/usr/bin/env python3
"""Full fixed-movetime self-play game over the real binary (P5/M3 exit gate).

design/00-master-plan.md P5 exit + design/12 step 5 name a full game under
cutechess with zero stalls / illegal moves / duplicate bestmoves. Where
cutechess-cli/fastchess are not installed, this driver is the sanctioned
fallback: python-chess's SimpleEngine plays the engine against ITSELF at a
fixed movetime, python-chess independently verifies the legality of every
move, and a per-move watchdog turns a stall into a hard failure instead of
a hang. Any UCI protocol violation (malformed/duplicate `bestmove`, illegal
move token) raises inside python-chess and fails the run.

Usage:
    python3 tools/selfplay_smoke.py [binary] [movetime_ms]

Defaults: target/release/chessengine, 100 ms. Exit 0 iff a full game
reached a real result (mate/stalemate/draw-by-rule) with every move legal.
"""

import signal
import sys
from pathlib import Path

import chess
import chess.engine

MAX_PLIES = 600  # a legal game claiming 3-fold/50-move ends far below this
WATCHDOG_SECS = 15  # per-move stall guard; movetime is 0.1 s


class MoveStall(Exception):
    pass


def _alarm(_sig, _frame):
    raise MoveStall(f"engine produced no bestmove within {WATCHDOG_SECS}s")


def main() -> int:
    binary = sys.argv[1] if len(sys.argv) > 1 else "target/release/chessengine"
    movetime_ms = int(sys.argv[2]) if len(sys.argv) > 2 else 100
    binary = str(Path(binary).resolve())

    engine = chess.engine.SimpleEngine.popen_uci(binary, timeout=30)
    board = chess.Board()
    san_line = []
    signal.signal(signal.SIGALRM, _alarm)
    try:
        while not board.is_game_over(claim_draw=True):
            if len(board.move_stack) >= MAX_PLIES:
                print(f"FAIL: no result after {MAX_PLIES} plies", file=sys.stderr)
                return 1
            signal.alarm(WATCHDOG_SECS)
            result = engine.play(board, chess.engine.Limit(time=movetime_ms / 1000))
            signal.alarm(0)
            move = result.move
            if move is None or move not in board.legal_moves:
                print(f"FAIL: illegal bestmove {move} in {board.fen()}", file=sys.stderr)
                return 1
            san_line.append(board.san(move))
            board.push(move)
    except MoveStall as e:
        print(f"FAIL: stall at ply {len(board.move_stack) + 1}: {e}", file=sys.stderr)
        engine.close()
        return 1
    finally:
        signal.alarm(0)

    engine.quit()  # raises if the engine desynced the protocol

    outcome = board.outcome(claim_draw=True)
    assert outcome is not None
    moves = " ".join(
        f"{i // 2 + 1}.{'..' if i % 2 else ''} {san}" if i % 2 == 0 or i == 0 else san
        for i, san in enumerate(san_line)
    )
    print(f"plies : {len(san_line)}")
    print(f"result: {board.result(claim_draw=True)} ({outcome.termination.name})")
    print(f"game  : {moves}")
    print("OK: full game, every move verified legal, no stalls, no protocol errors")
    return 0


if __name__ == "__main__":
    sys.exit(main())
