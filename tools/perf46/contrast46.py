#!/usr/bin/env python3
"""Per-round arm-minus-null contrast (the study-40 form).

Differencing two medians is NOT the same statistic as the median of per-round
differences, and only the latter cancels the round's shared load. This computes
the contrast the honest way: for each round, (arm - ref) - (null - ref), then
median / trimmed mean / se / sign test over rounds.
"""
import sys, statistics as st
from collections import defaultdict

path, ref, null = sys.argv[1], sys.argv[2], sys.argv[3]
arms = sys.argv[4:]
rows = defaultdict(dict)
for line in open(path):
    f = line.split()
    if len(f) < 3: continue
    r, lab = f[0], f[1]
    d = {}
    for kv in f[2:]:
        if "=" in kv:
            k, v = kv.split("=", 1)
            try: d[k] = float(v)
            except ValueError: pass
    if "cycles:u" in d and "ex_no_retire.thread_not_selected:u" in d:
        d["cycclean:u"] = d["cycles:u"] - d["ex_no_retire.thread_not_selected:u"]
    rows[r][lab] = d

def trimmed(xs, p=0.2):
    xs = sorted(xs); k = int(len(xs) * p)
    return st.mean(xs[k:len(xs)-k]) if len(xs) - 2*k > 0 else st.mean(xs)

counters = ["cycclean:u", "cycles:u", "instructions:u",
            "de_dis_dispatch_token_stalls1.store_queue_rsrc_stall:u",
            "ex_no_retire.load_not_complete:u"]
for arm in arms:
    print(f"\n## CONTRAST  {arm} - {null}   (both vs same-round {ref})")
    print(f"{'counter':<52}{'median%':>9}{'trim20%':>9}{'se':>8}{'neg/n':>8}")
    print("-" * 86)
    for c in counters:
        vals = []
        for r, labs in rows.items():
            if all(l in labs and c in labs[l] for l in (ref, arm, null)):
                a = (labs[arm][c] - labs[ref][c]) / labs[ref][c] * 100
                b = (labs[null][c] - labs[ref][c]) / labs[ref][c] * 100
                vals.append(a - b)
        if not vals: continue
        n = len(vals)
        se = st.stdev(vals) / (n ** 0.5) if n > 1 else 0.0
        neg = sum(1 for v in vals if v < 0)
        print(f"{c:<52}{st.median(vals):>9.3f}{trimmed(vals):>9.3f}{se:>8.3f}{neg:>5}/{n}")
