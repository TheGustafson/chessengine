#!/usr/bin/env python3
"""Regenerate tests/data/openings/openings-8ply.epd from the SAN lines below.

The committed opening book for local strength testing (design/14 §3.8): a
small, balanced set of standard theory lines, exactly 8 plies each, written
by hand (opening moves are public-domain facts, D27). Each line is replayed
through python-chess, so an illegal or mistyped move aborts generation —
the committed EPD can never contain an unreachable position.

Output format: one FEN per line (all six fields), the shape fastchess
consumes with `-openings file=... format=epd`.

This book is deliberately tame: near-balanced classical openings, good for
smoke SPRTs and gauntlets. For serious SPRT runs prefer an UHO
(unbalanced human openings) book — see tools/sprt.sh for the download
pointer.

Usage: python3 tools/gen_openings.py > tests/data/openings/openings-8ply.epd
Requires: pip install chess
"""

import sys

import chess

# (name, SAN line — exactly 8 plies). Names are comments for maintainers;
# they are not emitted (fastchess wants bare EPD lines).
LINES = [
    ("Ruy Lopez, Morphy", "e4 e5 Nf3 Nc6 Bb5 a6 Ba4 Nf6"),
    ("Ruy Lopez, Berlin", "e4 e5 Nf3 Nc6 Bb5 Nf6 O-O Be7"),
    ("Italian, Giuoco Piano", "e4 e5 Nf3 Nc6 Bc4 Bc5 c3 Nf6"),
    ("Italian, Two Knights", "e4 e5 Nf3 Nc6 Bc4 Nf6 d3 Be7"),
    ("Scotch", "e4 e5 Nf3 Nc6 d4 exd4 Nxd4 Nf6"),
    ("Four Knights", "e4 e5 Nf3 Nc6 Nc3 Nf6 Bb5 Bb4"),
    ("Petroff, Classical", "e4 e5 Nf3 Nf6 Nxe5 d6 Nf3 Nxe4"),
    ("Vienna Gambit", "e4 e5 Nc3 Nf6 f4 d5 fxe5 Nxe4"),
    ("King's Gambit Declined", "e4 e5 f4 Bc5 Nf3 d6 Nc3 Nf6"),
    ("Sicilian, Najdorf move order", "e4 c5 Nf3 d6 d4 cxd4 Nxd4 Nf6"),
    ("Sicilian, Open ...Nc6", "e4 c5 Nf3 Nc6 d4 cxd4 Nxd4 Nf6"),
    ("Sicilian, Taimanov", "e4 c5 Nf3 e6 d4 cxd4 Nxd4 Nc6"),
    ("Sicilian, Kan", "e4 c5 Nf3 e6 d4 cxd4 Nxd4 a6"),
    ("Sicilian, Alapin", "e4 c5 c3 Nf6 e5 Nd5 d4 cxd4"),
    ("Sicilian, Rossolimo", "e4 c5 Nf3 Nc6 Bb5 g6 O-O Bg7"),
    ("Sicilian, Moscow", "e4 c5 Nf3 d6 Bb5+ Bd7 Bxd7+ Qxd7"),
    ("French, Tarrasch", "e4 e6 d4 d5 Nd2 Nf6 e5 Nfd7"),
    ("French, Winawer", "e4 e6 d4 d5 Nc3 Bb4 e5 c5"),
    ("French, Classical", "e4 e6 d4 d5 Nc3 Nf6 Bg5 Be7"),
    ("Caro-Kann, Classical", "e4 c6 d4 d5 Nc3 dxe4 Nxe4 Bf5"),
    ("Caro-Kann, Advance", "e4 c6 d4 d5 e5 Bf5 Nf3 e6"),
    ("Scandinavian, ...Qa5", "e4 d5 exd5 Qxd5 Nc3 Qa5 d4 Nf6"),
    ("Pirc, Classical", "e4 d6 d4 Nf6 Nc3 g6 Nf3 Bg7"),
    ("Modern", "e4 g6 d4 Bg7 Nc3 d6 f4 Nf6"),
    ("Alekhine, Modern", "e4 Nf6 e5 Nd5 d4 d6 Nf3 Bg4"),
    ("QGD, Orthodox", "d4 d5 c4 e6 Nc3 Nf6 Bg5 Be7"),
    ("QGD, Exchange", "d4 d5 c4 e6 Nc3 Nf6 cxd5 exd5"),
    ("QGA, Classical", "d4 d5 c4 dxc4 Nf3 Nf6 e3 e6"),
    ("Slav, Main line", "d4 d5 c4 c6 Nf3 Nf6 Nc3 dxc4"),
    ("Semi-Slav", "d4 d5 c4 c6 Nf3 Nf6 Nc3 e6"),
    ("Nimzo-Indian, Rubinstein", "d4 Nf6 c4 e6 Nc3 Bb4 e3 O-O"),
    ("Queen's Indian", "d4 Nf6 c4 e6 Nf3 b6 g3 Ba6"),
    ("King's Indian, Classical", "d4 Nf6 c4 g6 Nc3 Bg7 e4 d6"),
    ("Gruenfeld, Exchange", "d4 Nf6 c4 g6 Nc3 d5 cxd5 Nxd5"),
    ("Catalan, Closed", "d4 Nf6 c4 e6 g3 d5 Bg2 Be7"),
    ("Benoni, Modern", "d4 Nf6 c4 c5 d5 e6 Nc3 exd5"),
    ("London System", "d4 d5 Bf4 Nf6 e3 c5 Nf3 Nc6"),
    ("Trompowsky", "d4 Nf6 Bg5 e6 e4 h6 Bxf6 Qxf6"),
    ("Dutch, Leningrad", "d4 f5 g3 Nf6 Bg2 g6 Nf3 Bg7"),
    ("English, Four Knights", "c4 e5 Nc3 Nf6 Nf3 Nc6 g3 d5"),
    ("English, Symmetrical", "c4 c5 Nf3 Nf6 d4 cxd4 Nxd4 e6"),
    ("Reti", "Nf3 d5 c4 e6 g3 Nf6 Bg2 Be7"),
    ("King's Indian Attack", "Nf3 Nf6 g3 g6 Bg2 Bg7 O-O O-O"),
]


def main() -> int:
    seen = set()
    for name, san_line in LINES:
        board = chess.Board()
        moves = san_line.split()
        if len(moves) != 8:
            print(f"{name}: {len(moves)} plies, want 8", file=sys.stderr)
            return 1
        for san in moves:
            board.push_san(san)  # raises on an illegal/ambiguous move
        fen = board.fen()
        if fen in seen:
            print(f"{name}: duplicate position", file=sys.stderr)
            return 1
        seen.add(fen)
        print(fen)
    print(f"{len(LINES)} openings written", file=sys.stderr)
    return 0


if __name__ == "__main__":
    sys.exit(main())
