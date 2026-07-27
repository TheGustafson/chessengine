#!/usr/bin/env python3
"""Compare two SPSA campaigns over their shared parameters — the replication test.

Why this exists. We cannot verify small SPSA gains by playing matches: a
200-game verification has roughly a +-13 Elo error bar, which is several times
larger than the effect SPSA produces, and that is exactly why stage A's verify
came back 0.00 +- 13.14 and told us nothing at all. Buying resolution with
games is not an option either — halving the bar costs 4x the games.

So the evidence has to come from the campaigns themselves. Two campaigns that
start from the SAME engine defaults and run at DIFFERENT depths are independent
experiments on the same question. If both push a parameter the same direction,
that is a replicated result drawn from the combined game count. If they
disagree, the parameter is depth-sensitive or noise, and either way it should
not be shipped.

The statistic. drift_report gives each parameter a displacement and a sigma —
the displacement measured against that parameter's own random-walk scale, so
sigma is already a signal-to-noise ratio. Signed, it is a z-score. Two
independent z-scores combine as (z1 + z2) / sqrt(2) (Stouffer's method with
equal weights), which is the right way to pool them: agreement reinforces,
disagreement cancels, and a large value from one campaign alone cannot carry a
parameter on its own.

A caveat this tool cannot check for you: the two campaigns must actually be
independent and comparable. If one was warm-started from the other's output, or
they ran against different engine builds, agreement is manufactured rather than
observed. c2 and c3 differ by the upcoming-repetition feature, so a
DISAGREEMENT between them has two possible causes (depth or that feature), and
should not be read as depth-sensitivity without checking.

    python3 tools/spsa/replication_report.py <run-dir-A> <run-dir-B>
"""

from __future__ import annotations

import re
import subprocess
import sys
from pathlib import Path

# drift_report row: NAME start end moved path directed sigma
# NOTE the explicit [+-]? on `moved`: drift_report renders positives with a
# leading '+'. A '-?' here silently drops every parameter that moved UPWARD,
# which does not error — it just returns half the rows and looks like a clean
# result. Caught by self-comparing a run and getting 7 params instead of 15.
ROW = re.compile(
    r"^([A-Z][A-Z0-9_]+)\s+([+-]?\d+)\s+([+-]?\d+)\s+([+-]?\d+)\s+(\d+)\s+([\d.]+)\s+([\d.]+)\s*$"
)

# Below this the parameter never moved meaningfully in either run; reporting it
# as "agreeing" would be noise dressed as evidence.
FLOOR = 0.5


def read_drift(run: Path) -> dict[str, tuple[int, float]]:
    here = Path(__file__).resolve().parent
    out = subprocess.run(["python3", str(here / "drift_report.py"), str(run)],
                         text=True, capture_output=True).stdout
    rows = {}
    candidates = 0
    for line in out.splitlines():
        # Anything shaped like a parameter row, whether or not the full
        # pattern matches. A silent shortfall here is the dangerous failure:
        # it does not raise, it just quietly halves the evidence.
        if re.match(r"^[A-Z][A-Z0-9_]+\s+[+-]?\d", line):
            candidates += 1
        m = ROW.match(line)
        if m:
            rows[m.group(1)] = (int(m.group(4)), float(m.group(7)))
    if not rows:
        sys.exit(f"no parameter rows parsed from drift_report for {run}")
    if len(rows) != candidates:
        sys.exit(f"parsed {len(rows)} rows but saw {candidates} parameter-shaped "
                 f"lines in {run} — the ROW pattern is out of step with "
                 f"drift_report's output; fix it rather than trusting a subset")
    return rows


def main() -> None:
    if len(sys.argv) != 3:
        sys.exit(__doc__.strip().splitlines()[-1].strip())
    a_dir, b_dir = Path(sys.argv[1]), Path(sys.argv[2])
    if a_dir.resolve() == b_dir.resolve():
        print("!! SAME RUN ON BOTH SIDES — this exercises the plumbing, it is not\n"
              "!! evidence. A run always agrees with itself, so every verdict below\n"
              "!! is meaningless as replication. Pass two DIFFERENT campaigns.\n")
    a, b = read_drift(a_dir), read_drift(b_dir)

    shared = sorted(set(a) & set(b))
    if not shared:
        sys.exit("the two campaigns share no parameters — nothing to replicate")
    print(f"A: {a_dir.name}   ({len(a)} params)")
    print(f"B: {b_dir.name}   ({len(b)} params)")
    print(f"shared: {len(shared)}\n")

    print(f"{'parameter':26s} {'A moved':>9} {'A sig':>6} {'B moved':>9} {'B sig':>6} "
          f"{'combined z':>11}  verdict")
    results = []
    for name in shared:
        am, asig = a[name]
        bm, bsig = b[name]
        # Signed z-scores, then Stouffer with equal weights.
        za = asig if am > 0 else (-asig if am < 0 else 0.0)
        zb = bsig if bm > 0 else (-bsig if bm < 0 else 0.0)
        z = (za + zb) / (2 ** 0.5)
        if max(asig, bsig) < FLOOR:
            verdict = "inert (neither moved)"
        elif (am > 0) == (bm > 0) and min(asig, bsig) >= FLOOR:
            verdict = "REPLICATED" if abs(z) >= 2.0 else "agrees, weak"
        else:
            verdict = "CONTRADICTS — do not ship"
        results.append((abs(z) if verdict.startswith("REPLICATED") else -1, name, verdict))
        print(f"{name:26s} {am:+9d} {asig:6.1f} {bm:+9d} {bsig:6.1f} {z:+11.2f}  {verdict}")

    strong = [n for score, n, v in results if v == "REPLICATED"]
    contra = [n for _, n, v in results if v.startswith("CONTRADICTS")]
    print(f"\nREPLICATED (agree, |combined z| >= 2): {strong if strong else 'none'}")
    print(f"CONTRADICTING: {contra if contra else 'none'}")
    print(
        "\nOnly the REPLICATED set has evidence worth acting on, and even that is\n"
        "evidence about the DIRECTION of a parameter, not proof of an Elo gain.\n"
        "Contradictions are informative too: they mark parameters whose optimum\n"
        "moves with depth, which is a reason to leave them at their defaults\n"
        "rather than to tune them at the wrong depth."
    )


if __name__ == "__main__":
    main()
