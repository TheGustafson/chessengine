#!/usr/bin/env python3
"""perf study 41 — price a front-end term by its DOSE SLOPE, not a datasheet.

    python3 tools/perf41/slope41.py <panel.txt> <ref> <y-metric> <x-metric>

Reads a `panel40.sh`-format panel that contains a layout DOSE SERIES (several
binaries differing only in code placement) and fits, over the per-label paired
medians, the ordinary least-squares line

    y_per_node  =  a + b * x_per_node

so `b` is "how many `y` does one more `x` cost, on THIS engine". The point of
doing it this way rather than quoting a manual's L1i-miss latency is that the
engine's front end runs ahead: most of a miss is hidden, and only the measured
slope says how much is not.

`b * x_ref` is then the share of `y` attributable to `x` at the shipping
operating point, extrapolating the dose line back to x = 0. Printed with the
fit's R^2 and the residual at the reference, because a dose line that does not
actually fit is not a price.

Every label is paired against `<ref>` WITHIN a round first (common-mode noise
cancels), then the per-label medians of the paired deltas are what the line is
fit through — so the fit inherits the panel's pairing, not its raw spread.
"""

import re
import statistics
import sys
from collections import defaultdict

ROW = re.compile(r"^r(\d+)\s+(\S+)\s+nodes=(\d+)\s+(.*)$")


def load(path: str) -> dict[int, dict[str, dict[str, float]]]:
    data: dict[int, dict[str, dict[str, float]]] = defaultdict(dict)
    for line in open(path):
        m = ROW.match(line.strip())
        if not m:
            continue
        row: dict[str, float] = {}
        for tok in m.group(4).split():
            if "=" in tok:
                k, v = tok.split("=", 1)
                try:
                    row[k] = float(v)
                except ValueError:
                    pass
        data[int(m.group(1))][m.group(2)] = row
    return data


def paired(data, ref: str, lab: str, k: str) -> list[float]:
    """Per-round (lab - ref) for metric k, only where both are present."""
    out = []
    for cols in data.values():
        if lab in cols and ref in cols and k in cols[lab] and k in cols[ref]:
            out.append(cols[lab][k] - cols[ref][k])
    return out


def main() -> None:
    path, ref, ymet, xmet = sys.argv[1:5]
    data = load(path)
    labels = []
    for cols in data.values():
        for lab in cols:
            if lab not in labels:
                labels.append(lab)

    yref = statistics.median(
        [c[ref][ymet] for c in data.values() if ref in c and ymet in c[ref]]
    )
    xref = statistics.median(
        [c[ref][xmet] for c in data.values() if ref in c and xmet in c[ref]]
    )
    print(f"reference {ref}:  {ymet} = {yref:.3f}/node   {xmet} = {xref:.3f}/node")
    print()
    print(f"{'label':<8} {'n':>4} {'d'+xmet:>22} {'d'+ymet:>22}")
    pts = []
    for lab in labels:
        dx, dy = paired(data, ref, lab, xmet), paired(data, ref, lab, ymet)
        if not dx or not dy:
            continue
        mx, my = statistics.median(dx), statistics.median(dy)
        print(f"{lab:<8} {len(dx):>4} {mx:>+22.3f} {my:>+22.3f}")
        pts.append((mx, my))

    n = len(pts)
    if n < 3:
        print("\nneed >= 3 dose points for a slope")
        return
    mx = statistics.fmean(p[0] for p in pts)
    my = statistics.fmean(p[1] for p in pts)
    sxx = sum((p[0] - mx) ** 2 for p in pts)
    sxy = sum((p[0] - mx) * (p[1] - my) for p in pts)
    if sxx == 0:
        print("\ndose did not move x — no slope")
        return
    b = sxy / sxx
    a = my - b * mx
    ss_tot = sum((p[1] - my) ** 2 for p in pts)
    ss_res = sum((p[1] - (a + b * p[0])) ** 2 for p in pts)
    r2 = 1 - ss_res / ss_tot if ss_tot else float("nan")
    # se(b) from the residuals, n-2 df
    se_b = ((ss_res / (n - 2)) / sxx) ** 0.5 if n > 2 else float("nan")
    print()
    print(f"slope  d({ymet}) / d({xmet}) = {b:+.4f} +/- {se_b:.4f}   R^2 = {r2:.4f}   n = {n}")
    print(f"share of {ymet} attributable to {xmet} at the reference:")
    print(f"    b * x_ref = {b * xref:+.2f} of {yref:.2f}  =  {100.0 * b * xref / yref:+.1f}%")


if __name__ == "__main__":
    main()
