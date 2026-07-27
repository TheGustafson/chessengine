#!/usr/bin/env python3
"""Three-way eval-scale study: ARCHON vs SF16 vs SF18, in each engine's OWN frame.

WHY. Every cp-denominated search constant — razoring, RFP/futility margins,
aspiration windows, history bonuses — is compared against the engine's static
eval. A constant is therefore meaningless except relative to the scale of the
eval it is compared against. We adopted a great many of these constants from
Stockfish, so the question "are our margins right?" reduces to "is our eval on
Stockfish's scale?", and that has never been measured directly.

There are hints that it is not. `src/search/params.rs` records, next to
OPTIMISM_NUM, that "our engine frame is ~2x compressed vs SF's reported cp"
from a 2026-07-26 winner-slope measurement — and that correction was applied
to the optimism priors ONLY (75/45 = SF's 137/81 x 0.55). If the compression
is real and general, every OTHER SF-derived constant is mis-scaled by the same
factor and nobody has adjusted them.

THE THREE FRAMES, which are genuinely different and must not be conflated:

  ARCHON   internal value, reported RAW as `cp` over UCI (no normalisation).
           Material/SEE scale: PAWN_VALUE = 208.
  SF16     internal value; UCI divides by a FIXED NormalizeToPawnValue = 328.
           Material: PawnValueMg 126 / PawnValueEg 208.
  SF18     internal value; UCI divides by a MATERIAL-DEPENDENT a(material)
           from the WDL model (a ~= 385.5 at the anchor material 58), so SF18's
           displayed cp is not a fixed multiple of its internal value at all.
           Material: PawnValue = 208 (mg/eg split dropped).

So SF16 and SF18 differ from each other by ~17.5% in internal-per-displayed at
the anchor, before we even get to our own frame. Comparing raw constants across
versions without this correction is exactly the mistake this tool exists to
prevent.

METHOD. Both Stockfish binaries print a pre-blend "NNUE evaluation" in PAWNS,
i.e. v/a. Multiplying back by a(material) recovers their internal v. Ours is
`nnue-eval`, already internal and pre-blend. Comparing pre-blend on both sides
keeps it apples-to-apples: the optimism/material blend that follows has its own
separate constants and would confound the scale question.

Robust slope by median-of-ratios rather than least squares — eval
distributions are heavy-tailed and a couple of tactical outliers would
otherwise set the answer.

    python3 tools/eval_scale_study.py [fens] [--net N] [--limit K]
"""

from __future__ import annotations

import argparse
import re
import statistics
import subprocess
import sys
from pathlib import Path

ARCHON = "engines/bin/archon-d47v-pgo"
NET = "data/nnue/nets/screens/d47v-a4b2-cal400.cenn"
SF16 = "engines/bin/stockfish16"
SF18 = "engines/bin/stockfish18"

NNUE_RE = re.compile(r"NNUE evaluation\s+([+-]?[\d.]+)")
FINAL_RE = re.compile(r"Final evaluation\s+([+-]?[\d.]+)")

# SF18 WDL model (uci.cpp win_rate_params). a(material) converts internal->cp.
AS = (-72.32565836, 185.93832038, -144.58862193, 416.44950446)
SF16_NORM = 328.0  # constant NormalizeToPawnValue


def sf18_a(material: int) -> float:
    m = min(max(material, 17), 78) / 58.0
    return ((AS[0] * m + AS[1]) * m + AS[2]) * m + AS[3]


def material_of(fen: str) -> int:
    """SF's material count: P + 3N + 3B + 5R + 9Q, both colours."""
    board = fen.split()[0]
    w = {c: board.count(c) for c in "PNBRQpnbrq"}
    return (
        (w["P"] + w["p"])
        + 3 * (w["N"] + w["n"] + w["B"] + w["b"])
        + 5 * (w["R"] + w["r"])
        + 9 * (w["Q"] + w["q"])
    )


def sf_evals(binary: str, fens: list[str]) -> list[tuple[float, float] | None]:
    """Return (nnue_pawns, final_pawns) per FEN, from one engine session."""
    cmd = "".join(f"position fen {f}\neval\n" for f in fens) + "quit\n"
    r = subprocess.run([binary], input=cmd, text=True, capture_output=True)
    out = r.stdout + r.stderr
    # Each `eval` emits one NNUE line and (usually) one Final line, in order.
    nnue = [float(m) for m in NNUE_RE.findall(out)]
    final = [float(m) for m in FINAL_RE.findall(out)]
    res: list[tuple[float, float] | None] = []
    for i in range(len(fens)):
        n = nnue[i] if i < len(nnue) else None
        f = final[i] if i < len(final) else n
        res.append((n, f) if n is not None else None)
    return res


def archon_evals(fens: list[str], net: str) -> list[int | None]:
    r = subprocess.run([ARCHON, "nnue-eval", net], input="\n".join(fens) + "\n",
                       text=True, capture_output=True)
    vals = []
    for line in r.stdout.splitlines():
        line = line.strip()
        vals.append(int(line) if re.fullmatch(r"[+-]?\d+", line) else None)
    return vals + [None] * (len(fens) - len(vals))


def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("fens", nargs="?", default="tools/nnue/calib/wl50.fens")
    ap.add_argument("--net", default=NET)
    ap.add_argument("--limit", type=int, default=0)
    args = ap.parse_args()

    raw = [l.strip() for l in Path(args.fens).read_text().splitlines()
           if l.strip() and not l.startswith("#")]

    def normalise(line: str) -> str:
        """Accept both FEN and EPD.

        EPD carries only 4 mandatory fields (board stm castling ep) and then
        OPCODES such as `bm Rg3; id "WAC.003";`. Slicing the first 6
        whitespace fields therefore swallows `bm Rg3;` into the FEN and every
        position fails to parse — which is exactly what happened, silently,
        yielding zero usable rows rather than an error.
        """
        f = line.split()
        head = f[:4]
        # Fields 5/6 are the clocks only when they are numeric; in EPD they
        # are the first opcode.
        if len(f) >= 6 and f[4].isdigit() and f[5].isdigit():
            head += [f[4], f[5]]
        else:
            head += ["0", "1"]
        return " ".join(head)

    fens = [normalise(l) for l in raw]
    if args.limit:
        fens = fens[: args.limit]
    print(f"positions: {len(fens)}  ({args.fens})\n")

    a_vals = archon_evals(fens, args.net)
    s16 = sf_evals(SF16, fens)
    s18 = sf_evals(SF18, fens)

    rows = []
    for fen, av, e16, e18 in zip(fens, a_vals, s16, s18):
        if av is None or e16 is None or e18 is None:
            continue
        mat = material_of(fen)
        # PERSPECTIVE. Ours is SIDE-TO-MOVE relative (eval/mod.rs "Inner eval,
        # stm-relative"); Stockfish's `eval` prints "(white side)". Without
        # this flip every black-to-move position contributes a negative ratio
        # and the median lands near zero — which is what the first run of this
        # tool produced, and it looks like a scale finding rather than a units
        # bug. Convert SF to stm-relative.
        stm_sign = 1.0 if fen.split()[1] == "w" else -1.0
        i16 = e16[0] * SF16_NORM * stm_sign
        i18 = e18[0] * sf18_a(mat) * stm_sign
        rows.append((fen, mat, av, i16, i18))

    print(f"usable: {len(rows)}\n")
    print(f"{'material':>8} {'ARCHON':>9} {'SF16 int':>10} {'SF18 int':>10} "
          f"{'A/SF16':>8} {'A/SF18':>8}")
    r16, r18 = [], []
    for _, mat, av, i16, i18 in rows[:15]:
        c1 = av / i16 if abs(i16) > 1e-9 else float("nan")
        c2 = av / i18 if abs(i18) > 1e-9 else float("nan")
        print(f"{mat:8d} {av:9d} {i16:10.1f} {i18:10.1f} {c1:8.3f} {c2:8.3f}")
    # Two filters, both necessary.
    #
    # Magnitude: a near-zero denominator makes the ratio explode and says
    # nothing about scale.
    #
    # Sign agreement: on some positions our net and SF's genuinely disagree
    # about who stands better. That is an eval-QUALITY difference, not a
    # units difference, and it produces negative ratios that wreck a median —
    # the unfiltered median came out at 0.16 while every well-behaved row sat
    # between 0.4 and 1.1. Scale can only be read off positions where both
    # engines agree on the sign; the disagreements are counted and reported
    # separately rather than silently dropped, because their RATE is itself
    # interesting.
    disagree16 = disagree18 = 0
    for _, _, av, i16, i18 in rows:
        if abs(i16) >= 40 and abs(av) >= 20:
            if (av > 0) == (i16 > 0):
                r16.append(av / i16)
            else:
                disagree16 += 1
        if abs(i18) >= 40 and abs(av) >= 20:
            if (av > 0) == (i18 > 0):
                r18.append(av / i18)
            else:
                disagree18 += 1

    def summarise(name: str, rs: list[float]) -> None:
        if len(rs) < 5:
            print(f"\n{name}: too few usable points ({len(rs)})")
            return
        rs = sorted(rs)
        med = statistics.median(rs)
        q1, q3 = rs[len(rs) // 4], rs[3 * len(rs) // 4]
        print(f"\n{name}: median ratio {med:.3f}   IQR [{q1:.3f}, {q3:.3f}]   n={len(rs)}")
        print(f"    -> our frame is {1/med:.2f}x COMPRESSED vs theirs"
              if med < 1 else
              f"    -> our frame is {med:.2f}x EXPANDED vs theirs")

    summarise("ARCHON / SF16 (internal)", r16)
    print(f"    sign disagreements excluded: {disagree16}"
          f" of {len(r16) + disagree16}")
    summarise("ARCHON / SF18 (internal)", r18)
    print(f"    sign disagreements excluded: {disagree18}"
          f" of {len(r18) + disagree18}")
    print(
        "\nRead with care: this is the scale of the NET OUTPUT, pre-blend. It does\n"
        "not by itself say our margins are wrong — margins were partly re-derived\n"
        "for our frame. It says what factor any SF-sourced constant must carry to\n"
        "mean the same thing here, and it is the input a margin model needs."
    )


if __name__ == "__main__":
    main()
