#!/usr/bin/env python3
"""What does every eval-denominated constant MEAN, in real pawns -- and is any
of them unaccounted for?

WHY IT WAS REWRITTEN (2026-07-27). The first version carried a HAND-WRITTEN
list of 15 thresholds. On re-derivation from the type system it turned out to
be missing eight genuinely eval-denominated constants, one of which
(`RFP_THREAT_MARGIN`) sits inside a mechanism that a "per-mechanism" rescale
had supposedly covered. The eval-frame rescale is still refuted -- five
formulations, none of which produced Elo -- but the honest statement of that
refutation is "over 30 of ~38 eval-denominated constants", not "over all of
them", and a hand-list is exactly how the gap stayed invisible.

So the list is no longer hand-written. `Value` IS the eval type in
`src/search/params.rs`, so every `pub NAME: Value = N;` is enumerated
automatically and must appear in CLASS below. A new tunable that nobody
classified makes this tool EXIT NON-ZERO rather than silently skipping it.
That is the same guard shape as screen.py's net-live check: the failure mode
is omission, so the defence has to be exhaustiveness, not care.

Being typed `Value` does not make a constant frame-carrying. Four kinds share
the type and only the first is affected by our 0.64x eval frame:

  eval      compared against a static eval, an eval difference, alpha or beta
            -- FRAME-CARRYING, means something different to us than to SF
  material  compared against PIECE_VALUES / SEE, which are Stockfish's exact
            values -- carries NO frame error
  ratio     a weight or denominator in a blend (x*a + y*b)/den -- dimensionless
  history   history/statScore units -- a separate frame, verified identical
            to eca43a97 (see docs/research/structural-audit-2026-07-27.md)

    python3 tools/frame_audit.py              # the table + coverage check
    python3 tools/frame_audit.py --all        # include the non-frame classes

Frames: our pawn is ~247 internal units. PAWN_VALUE 208 is MATERIAL, not eval
-- the eval frame is separate and measured at 0.64x SF18's a(material) ~385.
"""

from __future__ import annotations

import argparse
import re
import sys
from pathlib import Path

REPO = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(REPO / "tools"))

OUR_PAWN = 0.64 * 385.5      # our internal units per real pawn
SF_PAWN = 385.5              # SF18's a(material) at anchor material

# Every `pub NAME: Value` in params.rs must appear here. `fires` records how
# often the gate can fire, because consequence = frame error x firing rate x
# cost of a wrong firing, and the frame error alone is the SAME 1.56x for
# every row -- so it cannot rank them.
CLASS: dict[str, tuple[str, str]] = {
    # ---- eval-denominated: frame-carrying ----
    "QS_FUTILITY_MARGIN":          ("eval", "every qsearch leaf"),
    "RAZOR_BASE":                  ("eval", "shallow non-PV nodes"),
    "RAZOR_DEPTH_MULT":            ("eval", "shallow non-PV nodes, per d^2"),
    "RFP_MULT_BASE":               ("eval", "per ply, shallow nodes"),
    "RFP_MULT_CAP":                ("eval", "per ply, shallow nodes"),
    "RFP_MULT_SLOPE":              ("eval", "per ply, shallow nodes"),
    "RFP_MULT_NO_TTHIT":           ("eval", "shallow nodes, TT miss"),
    "RFP_THREAT_MARGIN":           ("eval", "RFP when nothing of ours hangs"),
    "NMP_EVAL_BASE":               ("eval", "cut nodes"),
    "NMP_EVAL_DEPTH":              ("eval", "cut nodes, per ply"),
    "NMP_EVAL_IMPROVING":          ("eval", "cut nodes, improving"),
    "PFUT_BASE":                   ("eval", "every quiet in the loop"),
    "PFUT_NO_BESTMOVE":            ("eval", "every quiet in the loop"),
    "PFUT_LMRDEPTH_MULT":          ("eval", "every quiet in the loop"),
    "PFUT_EVAL_ABOVE_ALPHA":       ("eval", "every quiet in the loop"),
    "CFUT_BASE":                   ("eval", "every capture in the loop"),
    "CFUT_LMRDEPTH_MULT":          ("eval", "every capture in the loop"),
    "PROBCUT_BETA_BASE":           ("eval", "depth>=3 nodes"),
    "PROBCUT_BETA_IMPROVING":      ("eval", "depth>=3 nodes, improving"),
    "PROBCUT_CHECK_BETA":          ("eval", "probcut in check"),
    "SE_BETA_MULT":                ("eval", "ttMove at depth>=6, per ply"),
    "SE_BETA_TTPV_MULT":           ("eval", "ttPv singular, per ply"),
    # the double/triple margins are compared against `value - singularBeta`,
    # an eval DIFFERENCE -- eval-denominated, not the dimensionless weights
    # their arithmetic shape suggests.
    "SE_DOUBLE_BASE":              ("eval", "singular double-extension margin"),
    "SE_DOUBLE_PV":                ("eval", "double margin, PV"),
    "SE_DOUBLE_NON_TTCAP":         ("eval", "double margin, quiet ttMove"),
    "SE_DOUBLE_DEEP":              ("eval", "double margin, deep"),
    "SE_TRIPLE_BASE":              ("eval", "singular triple-extension margin"),
    "SE_TRIPLE_PV":                ("eval", "triple margin, PV"),
    "SE_TRIPLE_NON_TTCAP":         ("eval", "triple margin, quiet ttMove"),
    "SE_TRIPLE_TTPV":              ("eval", "triple margin, ttPv"),
    "SE_TRIPLE_DEEP":              ("eval", "triple margin, deep"),
    "ASPIRATION_BASE_DELTA":       ("eval", "every ID iteration"),
    "LMR_DEEPER_MARGIN":           ("eval", "every LMR re-search"),
    "LMR_SHALLOWER_MARGIN":        ("eval", "every LMR re-search"),
    "HINDSIGHT_EVAL_SUM":          ("eval", "prior-reduced nodes -- changes DEPTH"),
    "FL_BELOW_EVAL_MARGIN":        ("eval", "fail-low history credit"),
    "FL_BELOW_PARENT_EVAL_MARGIN": ("eval", "fail-low history credit"),
    "EVALDIFF_CLAMP_LO":           ("eval", "eval-difference bonus, clamp"),
    "EVALDIFF_CLAMP_HI":           ("eval", "eval-difference bonus, clamp"),
    "EVALDIFF_OFFSET":             ("eval", "eval-difference bonus, offset"),
    "OPTIMISM_NUM":                ("eval", "root optimism, output scale"),
    "OPTIMISM_DEN_BASE":           ("eval", "root optimism, softening knee"),

    # ---- material / SEE: PIECE_VALUES are Stockfish's exactly, NO frame error ----
    "QS_SEE_FLOOR":                ("material", "every qsearch capture"),
    "QSEE_MULT":                   ("material", "every qsearch capture"),
    "CAP_SEE_DEPTH_MULT":          ("material", "every capture in the loop"),
    "GOOD_CAPTURE_SEE_DIVISOR":    ("material", "every capture, ordering"),
    "CAPTURE_MVV_FACTOR":          ("material", "every capture, ordering"),
    "QUIET_CHECK_SEE_MARGIN":      ("material", "checking quiets, ordering"),

    # ---- dimensionless ratios: numerator/denominator of a blend ----
    "ASPIRATION_WIDEN_NUM":        ("ratio", ""),
    "ASPIRATION_WIDEN_DEN":        ("ratio", ""),
    "QS_FAILHIGH_BEST_W":          ("ratio", ""),
    "QS_FAILHIGH_BETA_W":          ("ratio", ""),
    "QS_FAILHIGH_DEN":             ("ratio", ""),
    "QS_STANDPAT_BEST_W":          ("ratio", ""),
    "QS_STANDPAT_BETA_W":          ("ratio", ""),
    "QS_STANDPAT_DEN":             ("ratio", ""),
    "RFP_RET_BETA_W":              ("ratio", ""),
    "RFP_RET_EVAL_W":              ("ratio", ""),
    "RFP_RET_DEN":                 ("ratio", ""),
    "RFP_IMPROVING":               ("ratio", "scaled by futility_mult / 1024"),
    "RFP_OPP_WORSENING":           ("ratio", "scaled by futility_mult / 1024"),
    "RFP_TREND_DEN":               ("ratio", ""),
    "FRAME_S":                     ("ratio", "fixed-point scale"),
    "FH_SOFTEN_BETA_W":            ("ratio", "weighted against DEPTH, not eval"),
    "SE_BETA_DIV":                 ("ratio", "divides a depth-scaled margin"),
    "TTCUT_BOUND_SLACK":           ("depth", "Depth-typed; meets beta only via a predicate"),
    "TTCUT_AGREE_DEPTH":           ("depth", "Depth-typed, compared against depth"),

    # ---- history units: a separate frame, verified identical to eca43a97 ----
    "QUIET_MAIN_HISTORY_WEIGHT":   ("history", ""),
    "QUIET_PAWN_HISTORY_WEIGHT":   ("history", ""),
    "QUIET_LOW_PLY_WEIGHT":        ("history", ""),
    "QUIET_CHECK_BONUS":           ("history", ""),
    "QUIET_THREAT_FACTOR":         ("history", "x piece_value, ordering only"),
    "QUIET_OFFENSE_FACTOR":        ("history", "ARCHON original, x piece_value"),
    "QUIET_SORT_DEPTH_FACTOR":     ("history", ""),
    "GOOD_QUIET_THRESHOLD":        ("history", ""),
}


# CROSS-UNIT constants: they meet an eval quantity but are NOT typed `Value`,
# because they are typed by their OWN unit (a divisor, a converter, a rate).
# The Value-typed table above is structurally blind to them, which is exactly
# how ASPIRATION_MSS_DIVISOR survived five eval-frame rescale campaigns.
#
# The correction is r^k where k is set by the constant's DIMENSIONAL ROLE, not
# +1 everywhere:
#     k = +1  divisor of a SQUARED eval quantity (the term then scales r^2)
#     k = -1  coefficient converting eval into another unit (history, time)
#     k =  0  ratio, or a converter whose input is already in our frame
#
# name -> (k, what it meets, status)
CROSS_UNIT: dict[str, tuple[int, str, str]] = {
    "ASPIRATION_MSS_DIVISOR":      (+1, "divides mean_squared_score (eval^2)",
                                    "SHIPPED 10588 -> 6776, 2026-07-27"),
    "EVALDIFF_MAIN_MULT":          (-1, "eval_diff -> main history bonus",
                                    "tested 10 -> 16: 0.0 Elo over 140 games"),
    "EVALDIFF_PAWN_MULT":          (-1, "eval_diff -> pawn history bonus",
                                    "tested 13 -> 20: same arm, 0.0 Elo"),
    "FALLING_EVAL_PREV_AVG_COEFF": (-1, "eval drop -> time multiplier",
                                    "tested 2.21 -> 3.45: null at bullet"),
    "FALLING_EVAL_ITER_COEFF":     (-1, "eval drop -> time multiplier",
                                    "tested 1.0 -> 1.5625: same arm"),
    "FALLING_EVAL_BASE":           (0, "constant term beside the coeffs above",
                                    "dimensionless once the coeffs are corrected"),
    "FALLING_EVAL_DIV":            (0, "scales the whole falling_eval expression",
                                    "equivalent to scaling the coeffs; do one or the other"),
    "CORR_BONUS_DIVISOR":          (0, "eval error -> correction units",
                                    "corr is read back as cv/131072, so round-trip is consistent"),
    "RFP_CORR_DIV":                (0, "|correction_value| -> eval margin",
                                    "correction_value is already in OUR frame"),
    "RED_CORR_DIV":                (0, "|correction_value| -> reduction", ""),
    "SE_CORR_DIV":                 (0, "|correction_value| -> singular margin", ""),
    "RAZOR_CORR_DIV":              (0, "|correction_value| -> razor margin",
                                    "ARCHON original, inert at 1e9"),
    "NMP_CORR_DIV":                (0, "|correction_value| -> NMP margin",
                                    "ARCHON original, inert at 1e9"),
    "STATSCORE_CAPT_NUM":          (0, "captured PIECE VALUE -> statScore",
                                    "material carries no frame error"),
    "STATSCORE_CAPT_DEN":          (0, "same", ""),
    # Found by this tool's own guard on its first run, 2026-07-27.
    "SEE_THRESH_MULT":             (-1, "eval margin -> SEE threshold (MATERIAL)",
                                    "1024 identity; our eval is 0.64x SF's while "
                                    "PIECE_VALUES are identical, so we prune 36% "
                                    "tighter. Derived value 1600 -- UNDER TEST"),
}


def cross_unit_report(repo: Path) -> int:
    """Find constants that MEET an eval quantity, and check each is classified.

    The guard the Value-typed table cannot provide: greps the search for every
    `params::X` appearing on a line with an eval-derived identifier, and reports
    any name classified in NEITHER table. Returns the count of unclassified.
    """
    import re as _re
    evalish = _re.compile(r"static_eval|\beval\b|eval_diff|correction_value|"
                          r"mean_squared|best_value|\bbeta\b|\balpha\b")
    seen: set[str] = set()
    for rel in ("src/search/mod.rs", "src/eval/mod.rs", "src/history.rs",
                "src/timeman.rs"):
        f = repo / rel
        if not f.exists():
            continue
        for line in f.read_text().splitlines():
            st = line.strip()
            if st.startswith("//") or st.startswith("///"):
                continue
            if evalish.search(line):
                seen.update(_re.findall(r"params::([A-Z_0-9]+)", line))
    print("\n=== CROSS-UNIT constants (meet an eval quantity, not typed Value) ===")
    print(f"{'constant':<30}{'k':>4}  {'meets':<40}")
    for n, (k, meets, status) in sorted(CROSS_UNIT.items()):
        flag = "" if k == 0 else "  <-- frame-corrected"
        print(f"{n:<30}{k:>+4}  {meets:<40}{flag}")
        if status:
            print(f"{'':<36}{status}")
    unknown = sorted(n for n in seen if n not in CLASS and n not in CROSS_UNIT)
    if unknown:
        print("\nUNCLASSIFIED constants that meet an eval quantity:")
        for n in unknown:
            print(f"    {n}")
        print("Classify each in CLASS (if eval-denominated) or CROSS_UNIT (if it")
        print("meets eval across a unit boundary). k=+1 for a squared-quantity")
        print("divisor, k=-1 for an eval->other-unit converter, k=0 for a ratio.")
    return len(unknown)


def live_params(path: Path) -> dict[str, int]:
    src = path.read_text()
    return {m.group(1): int(m.group(2).replace("_", ""))
            for m in re.finditer(r"pub ([A-Z_0-9]+)\s*:\s*Value\s*=\s*(-?[\d_]+);", src)}


def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("--params", default="src/search/params.rs")
    ap.add_argument("--all", action="store_true", help="show non-frame classes too")
    ap.add_argument("--flag-above", type=float, default=0.75,
                    help="flag thresholds meaning more than this many pawns")
    args = ap.parse_args()

    live = live_params(REPO / args.params)

    # THE GUARD. An unclassified Value tunable is the exact failure that let
    # eight constants sit outside every rescale for a day.
    unknown = sorted(set(live) - set(CLASS))
    if unknown:
        print("UNCLASSIFIED eval-typed tunables -- add them to CLASS in this file:")
        for n in unknown:
            print(f"    {n:<30} = {live[n]}")
        print("\nUntil then no rescale can claim to be exhaustive. Exiting non-zero.")
        raise SystemExit(2)
    stale = sorted(set(CLASS) - set(live))
    if stale:
        print(f"note: classified but no longer in params.rs: {', '.join(stale)}\n")

    try:
        from arm_screen import FAMILIES
        covered = {p for v in FAMILIES.values() for p in v}
    except Exception:
        covered = set()

    show = ("eval",) if not args.all else ("eval", "material", "ratio", "history")
    print(f"{'constant':<30}{'value':>8}{'ours(pw)':>10}{'SF(pw)':>9}"
          f"{'resc':>6}  {'fires':<38}")
    n_eval = n_uncovered = 0
    for name in sorted(live, key=lambda n: (CLASS[n][0], n)):
        kind, fires = CLASS[name]
        if kind not in show:
            continue
        v = live[name]
        if kind == "eval":
            n_eval += 1
            ours, theirs = f"{v / OUR_PAWN:>10.2f}", f"{v / SF_PAWN:>9.2f}"
            flag = "" if name in covered else "  NO"
            n_uncovered += int(name not in covered)
            big = " <<" if abs(v / OUR_PAWN) >= args.flag_above else ""
        else:
            ours = theirs = f"{'-':>9} "
            flag, big = "", ""
        print(f"{name:<30}{v:>8}{ours}{theirs}{flag:>6}  {fires + big:<38}")

    if not args.all:
        others = {}
        for n in live:
            k = CLASS[n][0]
            if k != "eval":
                others[k] = others.get(k, 0) + 1
        print("\nhidden (not frame-carrying): "
              + ", ".join(f"{k} {c}" for k, c in sorted(others.items()))
              + "   -- use --all to show")

    print(f"\n{n_eval} eval-denominated constants; {n_uncovered} are in NO rescale "
          f"family (the 'NO' column).")
    print("Those are the ones the five refuted frame formulations never touched, so")
    print("'the frame rescale is refuted' means 'over the other "
          f"{n_eval - n_uncovered}'. The prior is still low --")
    print("five formulations found nothing -- but HINDSIGHT_EVAL_SUM is different in")
    print("kind from the rest: it changes DEPTH by a ply rather than moving a margin,")
    print("and a depth change is a much larger lever than a margin change.")
    n_unknown = cross_unit_report(REPO)
    print("\nEvery row is mis-scaled by the SAME 1.56x, so the ratio cannot rank them.")
    print("What ranks them is CONSEQUENCE = frame error x firing rate x cost of a")
    print("wrong firing -- read the 'fires' column, not the pawn columns.")


if __name__ == "__main__":
    main()
