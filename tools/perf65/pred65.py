#!/usr/bin/env python3
"""perf study 65 — read the eval-ELISION predictor census.

    python3 tools/perf65/pred65.py <foldcount dump>

Study 64 measured that 61.41% of every fresh NNUE forward pass in this engine
is consumed by ONE one-sided threshold comparison (reverse futility pruning, or
the qsearch stand pat) and by nothing else.  Removing an evaluation at such a
node removes its whole fold, which is ~45% of the engine's L1-miss latency at
the free-predictor ceiling — but only if something CHEAP can answer the
comparison before the fold runs.

The census records, per node, the true slack `value - threshold` and each
candidate's `pred - threshold`, cross-tabbed.  Sweeping a safety band `b` over
`pred - threshold >= b` reads the whole ROC off that table:

  * ELIDED  — the candidate says "cut" and the truth agrees: a fold saved.
  * WRONG   — the candidate says "cut" and the truth disagrees: a FUNCTIONAL
              error, a node pruned that the search wanted to look at.
  * MISSED  — the candidate stays silent on a node that would have cut: the
              full fold is paid anyway, plus the candidate's own cost.

`const0` is the null: a constant predictor.  A candidate that does not beat it
is reading the base rate.  `truth` is the self-test: at band 0 it must be 100%
elided / 0 wrong, or the census is mis-wired.
"""

import sys
from collections import defaultdict

B = 130
TARGETS = (0.0, 0.01, 0.02, 0.05, 0.10, 0.20)
CNAMES = ["<=-992", "-992..-480", "-480..-224", "-224..-96", "-96..-32",
          "-32..0", "0..32", "32..96", "96..224", "224..480", "480..992",
          ">=992"]
CEDGE_LO = [-(1 << 30), -992, -480, -224, -96, -32, 0, 32, 96, 224, 480, 992]
PC = 12


def bucket_lo(b):
    if b == 0:
        return -(1 << 30)
    if b == B - 1:
        return 1024
    return -1024 + (b - 1) * 16


def parse(path):
    d = {"nodes": {}, "elide": {}, "slack": defaultdict(dict),
         "roc": defaultdict(dict), "err": defaultdict(dict),
         "joint": defaultdict(dict), "avail": {}}
    for line in open(path):
        f = line.split()
        if not f:
            continue
        if f[0] == "PREDNODES":
            ch = f[1]
            d["nodes"][ch] = int(f[2].split("=")[1])
            d["elide"][ch] = int(f[3].split("=")[1])
        elif f[0] == "PREDSLACK":
            d["slack"][f[1]] = {int(k): int(v)
                                for k, v in (t.split(":") for t in f[2:])}
        elif f[0] == "PREDAVAIL":
            d["avail"][(f[1], f[2])] = int(f[3].split("=")[1])
        elif f[0] == "PREDROC":
            e = int(f[3].split("=")[1])
            d["roc"][(f[1], f[2], e)] = {int(k): int(v)
                                         for k, v in (t.split(":") for t in f[4:])}
        elif f[0] == "PREDJOINT":
            s = int(f[3].split("=")[1])
            d["joint"][(f[1], f[2], s)] = {int(k): int(v)
                                           for k, v in (t.split(":") for t in f[4:])}
        elif f[0] == "PREDERR":
            d["err"][(f[1], f[2])] = {int(k): int(v)
                                      for k, v in (t.split(":") for t in f[3:])}
    return d


def pct(a, b):
    return 100.0 * a / b if b else 0.0


def quantiles(hist, qs):
    tot = sum(hist.values())
    if not tot:
        return {q: None for q in qs}
    out, run, i = {}, 0, 0
    for b in sorted(hist):
        run += hist[b]
        while i < len(qs) and run >= qs[i] * tot:
            out[qs[i]] = bucket_lo(b)
            i += 1
    while i < len(qs):
        out[qs[i]] = bucket_lo(B - 1)
        i += 1
    return out


def eqs(v):
    return "-inf" if v is not None and v < -(1 << 20) else str(v)


def band_at(pos, neg, target):
    """Tightest band whose wrong-elision rate is <= target."""
    for b in range(B):
        good = sum(c for k, c in pos.items() if k >= b)
        bad = sum(c for k, c in neg.items() if k >= b)
        if good + bad and bad / (good + bad) <= target:
            return bucket_lo(b), good, bad, b
    return None


def main():
    d = parse(sys.argv[1])
    names = []
    for (_, name) in d["avail"]:
        if name not in names:
            names.append(name)

    for ch in d["nodes"]:
        n, el = d["nodes"][ch], d["elide"][ch]
        print(f"\n=== channel {ch}: {n:,} decision nodes, {el:,} elide "
              f"({pct(el, n):.2f}%)")
        sq = quantiles(d["slack"][ch], [0.05, 0.25, 0.5, 0.75, 0.95])
        print("    true slack (value - threshold) quantiles: "
              + "  ".join(f"p{int(q * 100)}={eqs(sq[q])}" for q in sorted(sq)))
        near = sum(c for b, c in d["slack"][ch].items() if 0 <= bucket_lo(b) < 128)
        print(f"    elisions clearing the bar by < 128 units: {near:,} "
              f"({pct(near, el):.2f}% of elisions)")

        hdr = (f"    {'candidate':<10} {'avail%':>7} {'errp5':>7} {'errp50':>7} "
               f"{'errp95':>7} |" + "".join(f"{t:>8.0%}" for t in TARGETS))
        print(hdr)
        for name in names:
            av = d["avail"].get((ch, name), 0)
            if not av:
                continue
            eq = quantiles(d["err"].get((ch, name), {}), [0.05, 0.5, 0.95])
            pos = d["roc"].get((ch, name, 1), {})
            neg = d["roc"].get((ch, name, 0), {})
            row = (f"    {name:<10} {pct(av, n):>6.1f}% {eqs(eq[0.05]):>7} "
                   f"{eqs(eq[0.5]):>7} {eqs(eq[0.95]):>7} |")
            for t in TARGETS:
                p = band_at(pos, neg, t)
                row += "     -- " if p is None else f" {pct(p[1], el):>6.2f}%"
            print(row)
        print("      err pN = quantiles of (pred - value); '-inf' = the "
              "under-range bucket.")
        print("      the six right-hand columns are COVERAGE — % of this "
              "channel's TRUE elisions")
        print("      converted — at the tightest band whose WRONG-elision rate "
              "is <= that header.")

        print("    cost of being wrong, at each candidate's <=5%-wrong band:")
        for name in names:
            if not d["avail"].get((ch, name), 0):
                continue
            p = band_at(d["roc"].get((ch, name, 1), {}),
                        d["roc"].get((ch, name, 0), {}), 0.05)
            if p is None:
                continue
            lo = p[0]
            tot = defaultdict(int)
            for s in range(6):          # coarse slack < 0 ⇒ a wrong elision
                for pcol, c in d["joint"].get((ch, name, s), {}).items():
                    if CEDGE_LO[pcol] >= lo:
                        tot[s] += c
            nb = sum(tot.values())
            if not nb:
                continue
            worst = "  ".join(f"{CNAMES[s]}:{pct(c, nb):.0f}%"
                              for s, c in sorted(tot.items()) if c)
            print(f"      {name:<10} band={lo:>5} n={nb:>8,}  true slack of "
                  f"the wrongly elided: {worst}")


if __name__ == "__main__":
    main()
