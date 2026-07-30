#!/usr/bin/env python3
"""perf study 40 — the arm-minus-null contrast, with a standard error.

    python3 tools/perf40/contrast40.py <panel.txt> <ref>[,<ref2>] <null> <arm>... [--metric M]

Three things this does that `med40.py` does not, all of them because the arm
this study has to decide (`insertwin34`) is a ~0.2%-of-cycles trade:

1. **The reference may be an AVERAGE of two byte-identical runs.** Half of a
   paired delta's variance is the reference run's. Averaging two reference runs
   from the same round cuts the reference's contribution in half — free, since
   the byte-identical null has to be in the panel anyway as the gate.
2. **A 20% TRIMMED mean, with its own standard error.** The paired noise here is
   heavy-tailed (raw sd 3.4%, trimmed sd 1.3%), so a trimmed statistic converges
   ~3x faster than the raw mean and far faster than a sign count.
3. **The contrast `arm − null` and its se.** The null is a binary computing the
   identical tree, so anything it reads other than zero is the panel's own bias;
   subtracting it is what makes the arm's number mean something. The verdict is
   this row, not the arm's raw delta.
"""

import re
import statistics
import sys
from collections import defaultdict

ROW = re.compile(r"^r(\d+)\s+(\S+)\s+nodes=(\d+)\s+(.*)$")


def load(path: str, metric: str) -> dict[int, dict[str, float]]:
    data: dict[int, dict[str, float]] = defaultdict(dict)
    for line in open(path):
        m = ROW.match(line.strip())
        if not m:
            continue
        for tok in m.group(4).split():
            if tok.startswith(metric + "="):
                data[int(m.group(1))][m.group(2)] = float(tok.split("=", 1)[1])
    return data


def trimmed(d: list[float], frac: float = 0.2) -> tuple[float, float, int]:
    d = sorted(d)
    t = int(len(d) * frac)
    core = d[t : len(d) - t] if len(d) - 2 * t >= 2 else d
    return statistics.fmean(core), statistics.stdev(core) / len(core) ** 0.5, len(core)


def main() -> None:
    path = sys.argv[1]
    refs = sys.argv[2].split(",")
    null = sys.argv[3]
    arms = [a for a in sys.argv[4:] if not a.startswith("--")]
    args = sys.argv
    metric = args[args.index("--metric") + 1] if "--metric" in args else "cycles:u"
    data = load(path, metric)

    per: dict[str, list[float]] = defaultdict(list)
    for _, cols in sorted(data.items()):
        have = [cols[r] for r in refs if r in cols]
        if len(have) < len(refs):
            continue
        base = statistics.fmean(have)
        for lab in [null] + arms:
            if lab in cols:
                per[lab].append(100.0 * (cols[lab] - base) / base)

    print(f"{path}  [{metric}]   reference = mean({','.join(refs)})")
    print(f"{'label':<8} {'n':>5} {'trimmed':>10} {'se':>8} {'raw mean':>10} {'median':>9} {'neg/n':>10}")
    stats = {}
    for lab in [null] + arms:
        d = per[lab]
        if len(d) < 8:
            continue
        tm, se, nc = trimmed(d)
        stats[lab] = (tm, se)
        neg = sum(1 for x in d if x < 0)
        print(
            f"{lab:<8} {len(d):>5} {tm:>+9.3f}% {se:>7.3f}% "
            f"{statistics.fmean(d):>+9.3f}% {statistics.median(d):>+8.3f}% {neg:>5}/{len(d):<4}"
        )
    print()
    if null in stats:
        nm, nse = stats[null]
        for lab in arms:
            if lab not in stats:
                continue
            am, ase = stats[lab]
            diff = am - nm
            se = (ase**2 + nse**2) ** 0.5
            print(
                f"CONTRAST {lab} - {null}: {diff:+.3f}% +/- {se:.3f}%  "
                f"({abs(diff) / se:.2f} se)"
            )


if __name__ == "__main__":
    main()
