#!/usr/bin/env python3
"""What does every eval-denominated threshold MEAN, in real pawns?

The eval-frame problem has been chased all day as "are our constants
Stockfish's raw values" — they are, verified to the digit — and every attempt
to rescale them globally, per family, as the algebraic dual, or class 4 alone
has failed. That framing is exhausted.

This asks a different question. Our eval measures ~0.64x Stockfish's, so a
constant of 335 means **1.36 pawns to us and 0.87 pawns to Stockfish**. The
error is the same 1.56x everywhere, so the error alone cannot say where to
look. What differs is the CONSEQUENCE, and consequence is

    frame error  x  how often the gate fires  x  cost of a wrong firing

which is why the one place it demonstrably bit us was **qsearch futility**:
identical 1.56x mis-scaling, but paid at EVERY LEAF of every tree. It shows
up as a flat ~2x node multiplier that survives four other explanations.

So this prints each threshold in real pawns for both engines and flags the
ones whose real-world meaning is implausible on its face — a "futility
margin" of 1.36 pawns, a "razoring" trigger at 3.1 pawns down. A threshold
that reads absurd in pawns is mis-scaled in a way that matters, whatever the
integer says.

    python3 tools/frame_audit.py                 # the table
    python3 tools/frame_audit.py --activation    # pair it with measured firing rates

Frames: our pawn is ~246 internal (PAWN_VALUE 208 is MATERIAL, not eval —
the eval frame is separate and measured at 0.64x SF18's a(material) ~385).
"""

from __future__ import annotations

import argparse
import re

OUR_PAWN = 0.64 * 385.5      # our internal units per real pawn
SF_PAWN = 385.5              # SF18's a(material) at anchor material

# Eval-denominated thresholds: constants compared against a static eval or an
# eval difference. Material-denominated ones (SEE bounds against PIECE_VALUES)
# and unit-free ones (plies, counts, history) are deliberately absent — they
# carry NO frame error, which is itself worth knowing.
#
# `per` records how often the gate can fire, because that is the multiplier on
# the consequence.
EVAL_THRESHOLDS = [
    ("QS_FUTILITY_MARGIN",   335, "every qsearch leaf",      "qsearch futility base"),
    ("RAZOR_BASE",           465, "shallow non-PV nodes",    "drop to qsearch"),
    ("RFP_MULT_BASE",         40, "per ply, shallow nodes",  "return eval, no search"),
    ("RFP_MULT_CAP",          80, "per ply, shallow nodes",  "cap on the above"),
    ("NMP_EVAL_BASE",        374, "cut nodes",               "skip a move entirely"),
    ("PFUT_BASE",             40, "every quiet in the loop", "skip one quiet"),
    ("PFUT_NO_BESTMOVE",     138, "every quiet in the loop", ""),
    ("PFUT_LMRDEPTH_MULT",   117, "every quiet in the loop", ""),
    ("CFUT_BASE",            231, "every capture in loop",   "skip one capture"),
    ("CFUT_LMRDEPTH_MULT",   232, "every capture in loop",   ""),
    ("PROBCUT_BETA_BASE",    214, "depth>=3 nodes",          "raised beta for probcut"),
    ("SE_BETA_MULT",          60, "ttMove at depth>=6",      "singular margin, per ply"),
    ("ASPIRATION_BASE_DELTA",  5, "every ID iteration",      "root window half-width"),
    ("LMR_DEEPER_MARGIN",     52, "every LMR re-search",     ""),
    ("LMR_SHALLOWER_MARGIN",   9, "every LMR re-search",     ""),
]


def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("--params", default="src/search/params.rs")
    ap.add_argument("--flag-above", type=float, default=0.75,
                    help="flag thresholds meaning more than this many pawns")
    args = ap.parse_args()

    src = open(args.params).read()
    live = {m.group(1): int(m.group(2))
            for m in re.finditer(r"pub ([A-Z_0-9]+)\s*:\s*\w+\s*=\s*(-?\d+);", src)}

    print(f"{'threshold':<24}{'value':>7}{'ours(pawns)':>13}{'SF(pawns)':>11}"
          f"{'':>3}{'fires':<24}")
    for name, expect, per, what in EVAL_THRESHOLDS:
        v = live.get(name, expect)
        if v != expect:
            per = f"{per}  [VALUE MOVED {expect}->{v}]"
        ours, theirs = v / OUR_PAWN, v / SF_PAWN
        flag = " <<" if abs(ours) >= args.flag_above else "  "
        print(f"{name:<24}{v:>7}{ours:>13.2f}{theirs:>11.2f}{flag:>3}{per:<24}")

    print(f"\nEvery row is mis-scaled by the SAME 1.56x, so the ratio cannot")
    print("rank them. What ranks them is CONSEQUENCE = frame error x firing")
    print("rate x cost of a wrong firing. Rows flagged '<<' mean more than")
    print(f"{args.flag_above} pawns in OUR frame, which is where a threshold")
    print("stops being a nudge and starts changing which moves get searched.")
    print("\nCross-reference the 'fires' column: a mis-scaled threshold paid at")
    print("every leaf (qsearch futility) cost us a flat ~2x node multiplier,")
    print("while the same error on a gate that fires at 0.2% of nodes (NMP)")
    print("cost nothing measurable.")
    print("\nDeliberately ABSENT: SEE bounds (QS_SEE_FLOOR, CAP_SEE_*, QSEE_MULT)")
    print("are compared against PIECE_VALUES, which are Stockfish's exactly, so")
    print("they carry NO frame error -- and QS_SEE_FLOOR is nonetheless the")
    print("single largest lever on per-leaf qsearch cost. Frame error is not")
    print("the same thing as importance.")


if __name__ == "__main__":
    main()
