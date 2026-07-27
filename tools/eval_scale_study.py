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


def drop_in_check(fens: list[str]) -> list[str]:
    """Remove positions where the side to move is in check.

    Stockfish's `eval` prints NO "NNUE evaluation" line for a position in
    check -- it prints only a "Final evaluation: none (in check)". Since the
    per-position values are recovered by INDEX from one batched session, a
    single in-check position silently shifts every SF value after it onto the
    wrong FEN. Measured on scale-check-800: 100 FENs, 7 in-check, 93 NNUE
    lines. That corrupted an earlier run of this study badly enough to give a
    7x wrong answer, and it corrupts any paired statistic computed the same
    way.

    Dropping them is also correct on the merits: a static eval in check is
    not a quantity either engine uses -- search generates evasions instead --
    so it has no bearing on how a margin should be scaled.
    """
    import chess
    return [f for f in fens if not chess.Board(f).is_check()]


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


def quantile_report(rows: list[tuple[str, int, int, float, float]]) -> None:
    """Quantile-match ARCHON's eval distribution against SF18's.

    This is the robust form of the band question, and it exists because the
    obvious method is a trap.

    Slicing positions by |SF eval| and regressing ARCHON on SF *within* each
    slice attenuates every slope toward zero: conditioning on a narrow x-range
    strips out the x-variance while leaving the y-noise, and with ~1/3 of
    positions carrying an outright sign disagreement that noise is large. The
    per-band slopes it produces are not comparable to each other, because the
    attenuation differs with band width.

    Quantile matching sidesteps both problems. Sort each engine's evals
    independently and pair them by RANK: the p-th percentile of ours against
    the p-th percentile of theirs. That compares the two marginal
    DISTRIBUTIONS, so a position where the engines disagree about who stands
    better contributes to each distribution's shape without poisoning a
    paired residual, and there is no regression to attenuate.

    The result is the calibration map T: their-frame -> our-frame. What a
    margin needs is its LOCAL DERIVATIVE T' at the operating point, which is
    the ratio of successive quantile gaps. A single global ratio is the right
    conversion only if T is a straight line through the origin.
    """
    ours = sorted(av for _, _, av, _, _ in rows)
    theirs = sorted(i18 for _, _, _, _, i18 in rows)
    n = min(len(ours), len(theirs))
    if n < 40:
        print(f"\nquantile match: too few positions ({n})")
        return
    ours, theirs = ours[:n], theirs[:n]
    pct = [2, 5, 10, 20, 30, 40, 50, 60, 70, 80, 90, 95, 98]
    print(f"\n{'pct':>5} {'SF18 int':>10} {'ARCHON':>9} {'T(x)/x':>8} {'local T′':>9}")
    prev = None
    for p in pct:
        i = min(n - 1, max(0, round(p / 100 * (n - 1))))
        x, y = theirs[i], ours[i]
        ratio = y / x if abs(x) > 1e-9 else float("nan")
        slope = float("nan")
        if prev is not None:
            dx, dy = x - prev[0], y - prev[1]
            slope = dy / dx if abs(dx) > 1e-9 else float("nan")
        print(f"{p:>5} {x:>10.1f} {y:>9.1f} {ratio:>8.3f} {slope:>9.3f}")
        prev = (x, y)
    # The operating band: margins compare eval against alpha/beta, and those
    # sit near the eval of a roughly balanced position. Take the local slope
    # across the middle of the distribution, where nearly every pruning
    # decision is actually made.
    lo = ours[round(0.30 * (n - 1))], theirs[round(0.30 * (n - 1))]
    hi = ours[round(0.70 * (n - 1))], theirs[round(0.70 * (n - 1))]
    central = (hi[0] - lo[0]) / (hi[1] - lo[1]) if abs(hi[1] - lo[1]) > 1e-9 else float("nan")
    print(
        f"\ncentral slope (30th->70th pct): {central:.3f}\n"
        "This is the number a margin conversion would have to use. Compare it\n"
        "with the global median ratio below: if they disagree, the global\n"
        "ratio was measuring the tails, not the region margins operate in."
    )


def band_report(rows: list[tuple[str, int, int, float, float]]) -> None:
    """Local slope d(ARCHON)/d(SF18), conditioned on |SF18 eval|.

    Why this is the number that matters, and the global median is not.

    A search margin never asks "how big is the eval". It asks whether a
    DIFFERENCE clears a threshold: `eval - beta >= rfp_margin`,
    `eval + margin <= alpha`. So the quantity a margin must be denominated in
    is the local derivative d(ours)/d(theirs) in the band where that
    comparison happens — a few hundred cp around alpha/beta — not the global
    regression slope over the whole distribution.

    The two coincide only if the relation is a straight line through the
    origin. It need not be: a net trained toward a WDL-shaped target
    saturates in the tails, which compresses large evals hard while leaving
    the near-zero region close to linear. If that is what ours does, the
    global 0.58 is dominated by the tails and the local slope where the
    margins actually bite could be near 1.0 — which would explain why
    Stockfish's raw constants measure as near-optimal for us and why every
    attempt to rescale them lost Elo.

    Reported per band: an ordinary least-squares slope WITH intercept (a
    forced-through-origin fit would just re-derive the ratio), the median
    ratio for comparison, and n. The intercept is printed because a large one
    is itself the signal that ratio-thinking was wrong.
    """
    edges = [(0, 50), (50, 100), (100, 200), (200, 400), (400, 800), (800, 10**9)]
    print(f"\n{'|SF18| band':>14} {'n':>5} {'slope':>7} {'intercept':>10} "
          f"{'median ratio':>13}")
    for lo, hi in edges:
        pts = [(i18, av) for _, _, av, _, i18 in rows if lo <= abs(i18) < hi]
        if len(pts) < 8:
            print(f"{f'{lo}-{hi}':>14} {len(pts):>5}   (too few)")
            continue
        n = len(pts)
        mx = sum(p[0] for p in pts) / n
        my = sum(p[1] for p in pts) / n
        sxx = sum((p[0] - mx) ** 2 for p in pts)
        sxy = sum((p[0] - mx) * (p[1] - my) for p in pts)
        slope = sxy / sxx if sxx > 1e-9 else float("nan")
        icept = my - slope * mx
        ratios = sorted(p[1] / p[0] for p in pts if abs(p[0]) >= 40)
        med = statistics.median(ratios) if len(ratios) >= 5 else float("nan")
        label = f"{lo}-{hi}" if hi < 10**9 else f"{lo}+"
        print(f"{label:>14} {n:>5} {slope:>7.3f} {icept:>10.1f} {med:>13.3f}")
    print(
        "\nRead: if the slope is ~flat across bands, the eval really is a\n"
        "uniform rescale and the global ratio is the right conversion. If it\n"
        "RISES toward the low bands, the compression lives in the tails, the\n"
        "margins operate in a near-1.0 region, and SF's raw constants are the\n"
        "right ones for us — which is what every rescale experiment measured."
    )


def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("fens", nargs="?", default="tools/nnue/calib/wl50.fens")
    ap.add_argument("--net", default=NET)
    ap.add_argument("--limit", type=int, default=0)
    ap.add_argument(
        "--bands",
        action="store_true",
        help="local slope d(ARCHON)/d(SF18) conditioned on |SF18 eval| "
        "instead of one global ratio (see band_report)",
    )
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
    before = len(fens)
    fens = drop_in_check(fens)
    if before != len(fens):
        print(f"dropped {before - len(fens)} in-check positions (see drop_in_check)")
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

    if args.bands:
        quantile_report(rows)
        band_report(rows)
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
