#!/usr/bin/env python3
"""perf study 66 — the per-node arithmetic behind the doc's tables, in one
place so every quoted figure is reproducible from the two raw artefacts
(`fc66.txt`'s census lines and `all4.tsv`'s panel medians)."""

N = 5_001_560  # nodes in the study frame

ROWS = {
    "hka_into": 11_305_854,
    "hka_dual": 4_792_984,
    "hka_inplace": 622,
    "thr_into": 19_714_419,
    "thr_refresh": 402_368,
}
TOTAL = sum(ROWS.values())
print(f"folded weight rows  {TOTAL:>14,}   {TOTAL / N:8.4f}/node")
for k, v in ROWS.items():
    print(f"  {k:14s} {v:>12,}   {v / N:8.4f}/node  {100 * v / TOTAL:5.2f}%")
print(f"weight-row bytes/node {TOTAL * 1280 / N:10.0f}")
print(f"weight-row lines/node {TOTAL * 20 / N:10.1f}")
print()

BASE = {
    "instructions:u": 27_791_774_374,
    "task-clock (ms)": 4152,
    "L1-dcache-load-misses": 2_793_889_052,
    "l2 dc_access_in_l2": 2_904_556_864,
    "l2 ic_dc_miss_in_l2": 215_652_762,
    "l2_pf_hit_l2.all": 1_114_009_309,
}
for k, v in BASE.items():
    print(f"{k:24s} {v:>16,}   {v / N:10.2f}/node")
print()

# Temporal reuse distance, from FOLDROWREUSE's log2 buckets. A fully
# associative cache holding the last W DISTINCT rows gets at least CDF(W)
# hits, because a window of W touches contains at most W distinct rows.
DIST = {
    "thr": [33362, 267772, 931303, 2148905, 3190593, 2909642, 2545991, 2029396,
            1508241, 1088435, 807569, 619521, 473533, 351937, 254949, 172494,
            105351, 61309, 35708, 31041, 35400, 35582, 24426, 7532, 266],
    "hka": [54071, 109245, 1458336, 1657816, 1669979, 1795763, 1795918, 1633128,
            1392391, 1147782, 919795, 730638, 561509, 404768, 276277, 172790,
            94027, 49151, 48680, 44453, 33414, 20499, 6654, 1795],
}
# 32 KB L1 / 1280 B = 25 rows; 1 MB L2 = 819; 32 MB L3 = 26214.
for name, h in DIST.items():
    tot, cum, out = sum(h), 0, []
    for i, x in enumerate(h):
        cum += x
        if i in (4, 9, 14):
            out.append(f"W<2^{i + 1} ({2 ** (i + 1)} rows): {100 * cum / tot:5.1f}%")
    print(f"{name} reuse CDF   touches={tot:,}   " + "   ".join(out))
print()

# The transpose arm's own locality census, on PHYSICAL offsets, both modes
# (`FOLDREDUND thr` under ARCHON_PERM=id and ARCHON_PERM=1).
SPAN = {
    "identity": dict(near=1_727_734, pairs=61_834_924,
                     ge24=539_774 + 547_620 + 1_340_780 + 175_412,
                     calls=4_316_728, asc=1_328_344),
    "transposed": dict(near=520_399, pairs=61_834_924,
                       ge24=404_868 + 498_858 + 1_316_701 + 208_934,
                       calls=4_316_728, asc=1_330_092),
}
for name, d in SPAN.items():
    print(f"{name:11s} pairs within 4 KB {100 * d['near'] / d['pairs']:6.3f}%   "
          f"folds spanning >=2^24 {100 * d['ge24'] / d['calls']:5.2f}%   "
          f"ascending {100 * d['asc'] / d['calls']:5.1f}%")
print(f"same-page row pairs fell "
      f"{SPAN['identity']['near'] / SPAN['transposed']['near']:.2f}x under the transpose")
