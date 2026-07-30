#!/usr/bin/env python3
"""perf study 41 — attribute a stall counter across several causes at once.

    python3 tools/perf41/attrib41.py <panel.txt> <ref> <y> <x1> [x2 ...]

`slope41.py` fits one cause. That is not enough here: a code-layout dose moves
the icache miss rate AND the branch predictor's aliasing at the same time, so a
one-variable fit hands the whole effect to whichever cause you happened to
regress on. This fits them together.

Method, and its two deliberate choices:

* **Paired medians, one point per label.** Every label is differenced against
  `<ref>` inside the SAME round first, so common-mode box noise cancels
  (study 40 §1.1); the per-label median of those deltas is one data point.
* **No intercept.** A byte-identical null must map to the origin by
  construction, so forcing the fit through it is not an assumption, it is the
  null. The fit's residual AT the null is printed as a check on exactly that.

Then `beta_i * x_i(ref)` is cause i's share of `y` at the shipping operating
point. Read those as UPPER bounds per cause: a dose that raises the footprint
also fragments the fetch stream, and any effect collinear with the dose is
attributed to whichever regressor carries it. The printed pairwise correlation
between the regressors is how collinear the design actually was — if it is near
1 the split between those two causes is not identified and only their SUM is.
"""

import statistics
import sys

sys.path.insert(0, __file__.rsplit("/", 1)[0])
from slope41 import load, paired  # noqa: E402


def solve(a: list[list[float]], b: list[float]) -> list[float]:
    """Gaussian elimination with partial pivoting on a small dense system."""
    n = len(b)
    m = [row[:] + [b[i]] for i, row in enumerate(a)]
    for c in range(n):
        p = max(range(c, n), key=lambda r: abs(m[r][c]))
        if abs(m[p][c]) < 1e-15:
            raise ValueError("singular normal equations — regressors are collinear")
        m[c], m[p] = m[p], m[c]
        for r in range(n):
            if r == c:
                continue
            f = m[r][c] / m[c][c]
            for k in range(c, n + 1):
                m[r][k] -= f * m[c][k]
    return [m[i][n] / m[i][i] for i in range(n)]


def main() -> None:
    path, ref, ymet = sys.argv[1], sys.argv[2], sys.argv[3]
    xmets = sys.argv[4:]
    data = load(path)

    labels: list[str] = []
    for cols in data.values():
        for lab in cols:
            if lab not in labels and lab != ref:
                labels.append(lab)

    def refmed(k: str) -> float:
        return statistics.median(
            [c[ref][k] for c in data.values() if ref in c and k in c[ref]]
        )

    rows, ys, names = [], [], []
    for lab in labels:
        dy = paired(data, ref, lab, ymet)
        dxs = [paired(data, ref, lab, x) for x in xmets]
        if not dy or any(not d for d in dxs):
            continue
        names.append(lab)
        ys.append(statistics.median(dy))
        rows.append([statistics.median(d) for d in dxs])

    n, p = len(rows), len(xmets)
    print(f"y = {ymet}   regressors = {xmets}")
    print(f"reference {ref}: y = {refmed(ymet):.3f}/node, "
          + ", ".join(f"{x} = {refmed(x):.3f}/node" for x in xmets))
    print()
    hdr = f"{'label':<8} {'dy':>10} " + " ".join(f"{('d' + x)[:14]:>15}" for x in xmets)
    print(hdr)
    for lab, y, r in zip(names, ys, rows):
        print(f"{lab:<8} {y:>+10.3f} " + " ".join(f"{v:>+15.4f}" for v in r))
    if n <= p:
        print("\nnot enough dose points")
        return

    # pairwise correlation between regressors — the identifiability check
    if p > 1:
        print()
        for i in range(p):
            for j in range(i + 1, p):
                ci = [r[i] for r in rows]
                cj = [r[j] for r in rows]
                mi, mj = statistics.fmean(ci), statistics.fmean(cj)
                num = sum((a - mi) * (b - mj) for a, b in zip(ci, cj))
                den = (sum((a - mi) ** 2 for a in ci) * sum((b - mj) ** 2 for b in cj)) ** 0.5
                print(f"corr(d{xmets[i]}, d{xmets[j]}) = {num / den if den else float('nan'):+.4f}")

    ata = [[sum(r[i] * r[j] for r in rows) for j in range(p)] for i in range(p)]
    atb = [sum(r[i] * y for r, y in zip(rows, ys)) for i in range(p)]
    beta = solve(ata, atb)

    fitted = [sum(b * v for b, v in zip(beta, r)) for r in rows]
    res = [y - f for y, f in zip(ys, fitted)]
    dof = n - p
    s2 = sum(e * e for e in res) / dof
    inv = [solve(ata, [1.0 if j == i else 0.0 for j in range(p)])[i] for i in range(p)]
    ss_tot = sum(y * y for y in ys)  # through the origin
    print()
    print(f"n = {n} dose points, {dof} dof, R^2(through origin) = "
          f"{1 - sum(e*e for e in res) / ss_tot:.4f}")
    yref = refmed(ymet)
    total = 0.0
    for i, x in enumerate(xmets):
        se = (s2 * inv[i]) ** 0.5
        share = beta[i] * refmed(x)
        total += share
        print(f"  beta[{x}] = {beta[i]:+.4f} +/- {se:.4f}  "
              f"->  {share:+8.2f} of {yref:.2f}  ({100 * share / yref:+.1f}%)")
    print(f"  UNATTRIBUTED residual                  ->  {yref - total:+8.2f} "
          f"({100 * (yref - total) / yref:+.1f}%)")
    print()
    print("per-point fit residuals (a byte-identical null must sit near 0):")
    for lab, e in zip(names, res):
        print(f"  {lab:<8} {e:>+9.3f}")


if __name__ == "__main__":
    main()
