#!/usr/bin/env python3
"""perf study 8 — measure the EVAL FIDELITY of a deliberately eval-changing arm.

    python3 evaldiff.py <baseline-cp-file> <arm-cp-file>

Each file is the raw `nnue-eval <net> < fens` output: one integer score per
line, in our internal frame.  A bit-exact change produces an identical file; an
arm that trades a little accuracy for speed does not, and then the question is
HOW MUCH — which is a distribution, not a hash.

Reports the delta distribution in BOTH frames: ours, and Stockfish's printed cp
(ours / 2.07, the standing normalisation) so the numbers can be read against
the ~1 cp intuition everyone actually has.
"""

import statistics
import sys

FRAME = 2.07  # our printed score / SF printed cp (CLAUDE.md, rule 2)

a = [int(x) for x in open(sys.argv[1]) if x.strip()]
b = [int(x) for x in open(sys.argv[2]) if x.strip()]
assert len(a) == len(b), f"{len(a)} vs {len(b)} scores"

d = [y - x for x, y in zip(a, b)]
ad = [abs(v) for v in d]
n = len(d)
same = sum(1 for v in d if v == 0)

print(f"positions              : {n}")
print(f"identical              : {same} ({same / n * 100:.1f}%)")
print(f"mean signed delta      : {statistics.mean(d):+.3f}  (ours)  "
      f"{statistics.mean(d) / FRAME:+.3f} (SF cp)")
print(f"mean |delta|           : {statistics.mean(ad):.3f}   (ours)  "
      f"{statistics.mean(ad) / FRAME:.3f} (SF cp)")
print(f"median |delta|         : {statistics.median(ad):.1f}")
for q in (90, 99):
    k = sorted(ad)[min(n - 1, int(n * q / 100))]
    print(f"p{q} |delta|            : {k}   (ours)  {k / FRAME:.2f} (SF cp)")
print(f"max |delta|            : {max(ad)}   (ours)  {max(ad) / FRAME:.2f} (SF cp)")
# correlation is what actually matters for move ORDERING
mean_a, mean_b = statistics.mean(a), statistics.mean(b)
cov = sum((x - mean_a) * (y - mean_b) for x, y in zip(a, b))
va = sum((x - mean_a) ** 2 for x in a)
vb = sum((y - mean_b) ** 2 for y in b)
print(f"Pearson r              : {cov / (va * vb) ** 0.5:.6f}")
print(f"baseline |score| mean  : {statistics.mean(abs(x) for x in a):.1f}")
