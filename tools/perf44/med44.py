#!/usr/bin/env python3
"""perf study 44 — reduce a panel44.sh dump to PAIRED within-round contrasts.

    python3 tools/perf44/med44.py <panel.txt> <ref> <label>... [--metric E]

For every label and counter, the per-round percentage difference against the
SAME round's reference, reported as median, 20%-trimmed mean, a standard error
on the trimmed mean, and a SIGN COUNT with an exact binomial p.

Why all four: study 40 measured this box's `cycles:u` noise as heavy-tailed --
raw sd 3.404% but 20%-trimmed sd 1.263% over 60 rounds of a byte-identical
null -- so a trimmed statistic converges ~3x faster than the range suggests,
while the sign count stays valid whatever the tail does.
"""

import math
import statistics
import sys
from collections import defaultdict


def load(path):
    rows = defaultdict(dict)  # rows[round][label] = {event: per-node}
    for line in open(path):
        if "SCALED" in line:
            raise SystemExit(f"refusing multiplexed row: {line.strip()}")
        t = line.split()
        if len(t) < 3:
            continue
        rnd, label = t[0], t[1]
        d = {}
        for tok in t[2:]:
            if "=" in tok:
                k, v = tok.split("=", 1)
                if k != "nodes":
                    d[k] = float(v)
        rows[rnd][label] = d
    return rows


def trimmed(xs, frac=0.2):
    xs = sorted(xs)
    k = int(len(xs) * frac)
    core = xs[k:len(xs) - k] if len(xs) - 2 * k >= 2 else xs
    m = statistics.fmean(core)
    sd = statistics.stdev(core) if len(core) > 1 else float("nan")
    return m, sd / math.sqrt(len(core)) if len(core) > 1 else float("nan")


def binom_p(k, n):
    """Two-sided exact binomial p at q=0.5 (stdlib only)."""
    k = min(k, n - k)
    tail = sum(math.comb(n, i) for i in range(k + 1)) / 2 ** n
    return min(1.0, 2 * tail)


def main():
    args = [a for a in sys.argv[1:] if not a.startswith("--")]
    path, ref, labels = args[0], args[1], args[2:]
    rows = load(path)

    # Study 44 §4.2: `panel44.sh` balances position-in-round with a Latin
    # square whose block length is the LABEL COUNT, so the design is only
    # exactly balanced at a multiple of that. Reading a partial block
    # reintroduces the very confound the square exists to remove -- a +3.9%
    # first-slot-to-last-slot drift on this frame -- so truncate to whole
    # blocks rather than quietly report a lopsided n.
    nlab = len(labels) + 1
    complete = (len(rows) // nlab) * nlab
    if complete < len(rows):
        keep = sorted(rows, key=lambda s: int(s[1:]))[:complete]
        dropped = len(rows) - complete
        rows = {k: rows[k] for k in keep}
        print(f"# truncated to {complete} rounds = {complete // nlab} whole "
              f"Latin-square blocks of {nlab} ({dropped} partial-block "
              f"round(s) dropped)")
    if not rows:
        raise SystemExit(f"not one complete block of {nlab} rounds yet")
    # Study 40 §3.2: `ex_no_retire.thread_not_selected` is the SMT sibling
    # stealing the core. It ranges -56% to +203% between rounds and is pure
    # noise for a single-thread arm, so subtracting it gives a cleaner cycle
    # counter -- measured there to cut a byte-identical null's trimmed sd from
    # 0.888% to 0.624%, i.e. 2.0x fewer rounds for the same resolution.
    for per in rows.values():
        for d in per.values():
            if "cycles:u" in d and "ex_no_retire.thread_not_selected:u" in d:
                d["cycclean:u"] = d["cycles:u"] - d["ex_no_retire.thread_not_selected:u"]
    events = sorted({e for r in rows.values() for d in r.values() for e in d})

    print(f"# {path}\n# reference: {ref}\n")
    for label in labels:
        deltas = defaultdict(list)
        for rnd, per in rows.items():
            if ref not in per or label not in per:
                continue
            for e in events:
                if e in per[ref] and e in per[label] and per[ref][e]:
                    deltas[e].append((per[label][e] / per[ref][e] - 1) * 100)
        n = max((len(v) for v in deltas.values()), default=0)
        print(f"## {label}  vs  {ref}   (n = {n} paired rounds)")
        print(f"{'counter':<40}{'median%':>10}{'trim20%':>10}{'se':>9}"
              f"{'neg/n':>10}{'p':>9}")
        print("-" * 88)
        for e in events:
            v = deltas.get(e)
            if not v:
                continue
            m, se = trimmed(v)
            neg = sum(1 for x in v if x < 0)
            print(f"{e:<40}{statistics.median(v):>10.3f}{m:>10.3f}{se:>9.3f}"
                  f"{f'{neg}/{len(v)}':>10}{binom_p(neg, len(v)):>9.4f}")
        print()


if __name__ == "__main__":
    main()
