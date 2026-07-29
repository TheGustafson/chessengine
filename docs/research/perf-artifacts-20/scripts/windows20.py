"""perf study 20 — bucket a `perf annotate` dump into 64-byte address windows.

Region attribution only. Per-opcode attribution on this box is wrong by ~1.44x
(perf study 18), so this exists to answer "which LOOP holds the mass", never
"which instruction costs what".

    python3 windows20.py docs/research/perf-artifacts-20/anno-<sym>.txt [top]
"""

import re
import sys
from collections import defaultdict

path = sys.argv[1]
top = int(sys.argv[2]) if len(sys.argv) > 2 else 10

rows = []
for line in open(path):
    m = re.match(r"\s*([0-9]+\.[0-9]+)\s*:\s*([0-9a-f]+):\s*(.*)", line)
    if m:
        rows.append((float(m.group(1)), int(m.group(2), 16), m.group(3).strip()))
        continue
    m = re.match(r"\s*:\s*([0-9a-f]+):\s*(.*)", line)
    if m:
        rows.append((0.0, int(m.group(1), 16), m.group(2).strip()))

buckets = defaultdict(float)
members = defaultdict(list)
for pct, addr, text in rows:
    buckets[addr >> 6] += pct
    members[addr >> 6].append((pct, addr, text))

print(f"{path}: {len(rows)} instructions, {sum(b for b in buckets.values()):.1f}% total")
for key, share in sorted(buckets.items(), key=lambda kv: -kv[1])[:top]:
    print(f"--- {hex(key << 6)}  {share:5.1f}%")
    for pct, addr, text in sorted(members[key], key=lambda x: -x[0])[:5]:
        print(f"      {pct:5.2f}  {hex(addr)}  {text[:70]}")
