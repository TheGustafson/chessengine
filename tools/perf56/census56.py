#!/usr/bin/env python3
"""perf study 56 -- the MECHANISM B CENSUS.

Study 55's largest single win (`bcbuf55`, -84.6 i/n) was three `& BUF_MASK`
insertions on a fixed-power-of-two array whose index LLVM could not prove in
range.  Study 55 §7 item 4 says the method is disassembly, and leaves the script
unwritten.  This is it.

For every symbol in the binary, count the machine sites that exist ONLY because
a bounds check could not be folded:

  * `panic_bounds_check` calls           -- an unfoldable slice/array index
  * `panic::<...>` / `unwrap_failed`     -- an unfoldable Option/Result
  * `slice_start_index_len_fail` etc.    -- an unfoldable range

and rank symbols by count.  A `call` to a panic helper is a cold edge, but each
one is PRECEDED by a `cmp`/`j<cc>` on the hot path -- that pair is the cost, and
it is what the mask deletes.  Ranking by panic-edge count therefore ranks the
hot-path compare-and-branch pairs the mask can delete, per symbol.

  python3 tools/perf56/census56.py bin56/ctl56 [top-n]
"""
import re
import subprocess
import sys
from collections import Counter, defaultdict

BIN = sys.argv[1]
TOPN = int(sys.argv[2]) if len(sys.argv) > 2 else 30

# The cold-edge helpers Rust emits for a bounds/range/unwrap failure.
PANIC = re.compile(
    r"panic_bounds_check|slice_index_order_fail|slice_start_index_len_fail"
    r"|slice_end_index_len_fail|unwrap_failed|panic_misaligned|panic_cannot_unwind"
    r"|core\.\.option\.\.expect_failed|core\.\.panicking\.\.panic\b"
)
KIND = [
    ("bounds", re.compile(r"panic_bounds_check")),
    ("range", re.compile(r"slice_index_order_fail|slice_start_index_len_fail|slice_end_index_len_fail")),
    ("unwrap", re.compile(r"unwrap_failed|expect_failed")),
    ("panic", re.compile(r"core\.\.panicking\.\.panic\b")),
]

out = subprocess.run(
    ["objdump", "-C", "-d", "--no-show-raw-insn", BIN],
    capture_output=True, text=True, check=True,
).stdout

sym = None
per_sym = defaultdict(Counter)
sizes = {}
start = None
last = None
for line in out.splitlines():
    m = re.match(r"^([0-9a-f]+) <(.+)>:$", line)
    if m:
        if sym is not None and start is not None and last is not None:
            sizes[sym] = last - start
        sym = m.group(2)
        start = int(m.group(1), 16)
        per_sym.setdefault(sym, Counter())
        continue
    m = re.match(r"^\s+([0-9a-f]+):\s", line)
    if m:
        last = int(m.group(1), 16)
    if sym is None:
        continue
    if "call" in line and PANIC.search(line):
        for name, rx in KIND:
            if rx.search(line):
                per_sym[sym][name] += 1
                break
if sym is not None and start is not None and last is not None:
    sizes[sym] = last - start

rows = []
for s, c in per_sym.items():
    tot = sum(c.values())
    if tot:
        rows.append((tot, c, sizes.get(s, 0), s))
rows.sort(reverse=True, key=lambda r: r[0])

grand = sum(r[0] for r in rows)
print(f"{BIN}: {grand} panic edges over {len(rows)} symbols\n")
print(f"{'edges':>6}{'bnd':>6}{'rng':>6}{'unw':>6}{'pan':>6}{'bytes':>8}  symbol")
for tot, c, size, s in rows[:TOPN]:
    print(f"{tot:>6}{c['bounds']:>6}{c['range']:>6}{c['unwrap']:>6}{c['panic']:>6}"
          f"{size:>8}  {s[:110]}")
