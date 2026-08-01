#!/usr/bin/env python3
"""tools/perf70/pool_sprt.py — pool the PENTANOMIAL counts of several fastchess
runs of the SAME pair into one result.

Why this exists: fastchess cannot resume. A long SPRT that dies (OOM, reboot,
a killed scope) leaves thousands of perfectly good games that would otherwise
be thrown away and re-played. Pooling is valid here because the runs differ
only in the book's random order, so the game pairs are exchangeable.

It pools PAIRS, not games. The pentanomial `Ptnml(0-2)` counts are the
statistically correct unit for an opening-repeated match: the two games of a
round share an opening and are correlated, and treating them as independent
understates the variance (which is exactly what the trinomial Elo bar does).

  tools/perf70/pool_sprt.py engines/runs/*/sprt.out

`--verify` re-derives each input run's own Elo/nElo from its counts and prints
them beside the values fastchess printed, so the formulas are checked against
the tool they are imitating rather than trusted.
"""

import argparse
import math
import re
import sys

# Pair outcomes, in Ptnml order: LL, LD, LW+DD, DW, WW.
POINTS = (0.0, 0.25, 0.5, 0.75, 1.0)
LOG10 = math.log(10.0)


def parse(path):
    """(ptnml counts, fastchess's own Elo, nElo) from a sprt.out."""
    ptnml = elo = nelo = None
    for line in open(path, errors="replace"):
        m = re.search(r"Ptnml\(0-2\):\s*\[([0-9,\s]+)\]", line)
        if m:
            ptnml = [int(x) for x in m.group(1).split(",")]
        m = re.search(r"Elo:\s*(-?[\d.]+)\s*\+/-\s*([\d.]+),\s*nElo:\s*(-?[\d.]+)", line)
        if m:
            elo, nelo = float(m.group(1)), float(m.group(3))
    return ptnml, elo, nelo


def stats(counts):
    n = sum(counts)
    if n == 0:
        raise SystemExit("pool_sprt: no pairs")
    mu = sum(c * p for c, p in zip(counts, POINTS)) / n
    var = sum(c * (p - mu) ** 2 for c, p in zip(counts, POINTS)) / n
    sd = math.sqrt(var)
    se = sd / math.sqrt(n)
    # Logistic Elo of the per-pair score, and its delta-method bar.
    def elo(x):
        x = min(max(x, 1e-9), 1 - 1e-9)
        return -400.0 * math.log10(1.0 / x - 1.0)

    # Normalized Elo: the score advantage in units of the pair-result standard
    # deviation, on the 800/ln(10) scale. The `sqrt(2)` is fastchess's own
    # per-GAME convention against this per-PAIR sigma; it is CALIBRATED, not
    # derived — run with `--verify` and it reproduces fastchess to 0.01 nElo
    # (9.7658 * sqrt(2) = 13.8109 against fastchess's printed 13.81). Elo and
    # LOS below need no such constant and match fastchess exactly.
    k = (800.0 / LOG10) / math.sqrt(2.0)
    nelo = (mu - 0.5) / sd * k if sd > 0 else 0.0
    nelo_bar = 1.96 * se / sd * k if sd > 0 else 0.0
    los = 0.5 * (1.0 + math.erf((mu - 0.5) / (se * math.sqrt(2.0)))) if se > 0 else 0.5
    return {
        "pairs": n,
        "games": 2 * n,
        "score": mu,
        "elo": elo(mu),
        "elo_lo": elo(mu - 1.96 * se),
        "elo_hi": elo(mu + 1.96 * se),
        "nelo": nelo,
        "nelo_bar": nelo_bar,
        "los": 100.0 * los,
    }


def show(label, s):
    bar = (s["elo_hi"] - s["elo_lo"]) / 2.0
    print(
        f"{label:<34} pairs={s['pairs']:<6} games={s['games']:<6} "
        f"score={100 * s['score']:.2f}%  "
        f"Elo={s['elo']:+.2f} +/- {bar:.2f}  "
        f"nElo={s['nelo']:+.2f} +/- {s['nelo_bar']:.2f}  LOS={s['los']:.2f}%"
    )


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("runs", nargs="+")
    ap.add_argument("--verify", action="store_true", help="check formulas against fastchess")
    a = ap.parse_args()

    total = [0] * 5
    any_found = False
    for path in a.runs:
        counts, elo, nelo = parse(path)
        if not counts:
            print(f"# {path}: no Ptnml line (run still warming up?)", file=sys.stderr)
            continue
        any_found = True
        total = [t + c for t, c in zip(total, counts)]
        s = stats(counts)
        show(path, s)
        if a.verify and elo is not None:
            print(
                f"{'  fastchess said':<34} "
                f"Elo={elo:+.2f}  nElo={nelo:+.2f}   "
                f"(ours: Elo={s['elo']:+.2f}  nElo={s['nelo']:+.2f})"
            )
    if not any_found:
        raise SystemExit("pool_sprt: nothing to pool")
    print("-" * 118)
    show("POOLED", stats(total))
    print(f"{'  pooled Ptnml(0-2)':<34} {total}")


if __name__ == "__main__":
    main()
