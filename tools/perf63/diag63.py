#!/usr/bin/env python3
"""Decompose a `wallclock_panel.py --dump` TSV — perf study 63.

WHY THIS EXISTS
---------------
`wallclock_panel.py` refused to license three arms across studies 62 and 63, and
each time the refusal was read as "the box is noisy, come back later". It was
not. A byte-identical null that reads p < 1e-4 is a BUG SIGNATURE, and the only
way to tell a bug from noise is to ask WHERE the null's excursion lives:

  * split by ROTATION PHASE  -> a position artifact shows up in one phase only
  * split by QUARTER of run  -> drift, or an unrelated job, shows up in one block
  * per-POSITION medians     -> names the position effect directly
  * position-CORRECTED redo  -> what is left after the position main effect

Study 63's 400-round panel was positive in all three phases (+0.407 / +0.661 /
+0.864 %) and all four quarters, with the three position medians inside 0.25 %
of each other — so not position, not drift, therefore a property of the null
BINARY. That pointed at the two asymmetries `wallclock_panel.py`'s banner now
documents: the null copy lived on tmpfs while the arms lived on the NVMe, and
the runs were not pinned.

USAGE
-----
    python3 tools/perf63/diag63.py <dump.tsv>

The rotation is the one `wallclock_panel.py` uses:
    round % 3 == 0 -> (base, null, arm)
    round % 3 == 1 -> (null, arm, base)
    round % 3 == 2 -> (arm, base, null)
"""
import statistics as st
import sys
from math import comb

ORDER = {0: ("base", "null", "arm"),
         1: ("null", "arm", "base"),
         2: ("arm", "base", "null")}
POS = {ph: {name: i for i, name in enumerate(names)} for ph, names in ORDER.items()}


def rep(v, tag):
    neg, n = sum(1 for x in v if x < 0), len(v)
    k = max(neg, n - neg)
    p = min(1.0, 2 * sum(comb(n, i) for i in range(k, n + 1)) / 2 ** n)
    print(f"{tag}: median {st.median(v):+.3f}%  sd {st.stdev(v):.3f}  "
          f"sign {neg}/{n} neg  p={p:.4f}")


def main():
    rows = []
    for line in open(sys.argv[1]):
        if line.startswith("#") or line.startswith("round"):
            continue
        r, tb, tn, ta = line.split()
        rows.append((int(r), float(tb), float(tn), float(ta)))
    n = len(rows)
    print(f"rounds {n}")
    print("absolute medians (s): base %.4f  null %.4f  arm %.4f"
          % (st.median(r[1] for r in rows), st.median(r[2] for r in rows),
             st.median(r[3] for r in rows)))

    bypos = {0: [], 1: [], 2: []}
    for r, tb, tn, ta in rows:
        m = POS[r % 3]
        bypos[m["base"]].append(tb)
        bypos[m["null"]].append(tn)
        bypos[m["arm"]].append(ta)
    print("position medians (s): "
          + "  ".join(f"pos{p}={st.median(bypos[p]):.4f}" for p in (0, 1, 2)))
    print()

    for phase in (0, 1, 2):
        sub = [x for x in rows if x[0] % 3 == phase]
        if not sub:
            continue
        m = POS[phase]
        nul = [100 * (tn - tb) / tb for _, tb, tn, _ in sub]
        arm = [100 * (ta - tb) / tb for _, tb, _, ta in sub]
        print(f"phase {phase} (base pos {m['base']}, null pos {m['null']}, "
              f"arm pos {m['arm']}): n={len(sub)}  "
              f"null median {st.median(nul):+.3f}  arm median {st.median(arm):+.3f}")
    print()
    q = max(1, n // 4)
    for i in range(4):
        sub = rows[i * q:(i + 1) * q]
        if not sub:
            continue
        print("quarter %d: median t_base %.4f   null %+.3f%%   arm %+.3f%%"
              % (i, st.median(x[1] for x in sub),
                 st.median(100 * (x[2] - x[1]) / x[1] for x in sub),
                 st.median(100 * (x[3] - x[1]) / x[1] for x in sub)))
    print()

    posmed = {p: st.median(bypos[p]) for p in (0, 1, 2)}
    adj_null, adj_arm = [], []
    for r, tb, tn, ta in rows:
        m = POS[r % 3]
        b = tb / posmed[m["base"]]
        adj_null.append(100 * (tn / posmed[m["null"]] - b) / b)
        adj_arm.append(100 * (ta / posmed[m["arm"]] - b) / b)
    print("--- POSITION-CORRECTED (each timing divided by its position's median) ---")
    rep(adj_null, "NULL")
    rep(adj_arm, "ARM ")


if __name__ == "__main__":
    main()
