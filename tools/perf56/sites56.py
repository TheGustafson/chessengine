#!/usr/bin/env python3
"""perf study 56 -- resolve every bounds-check panic edge to its SOURCE LINE,
with no debug-info build.

Study 55 §7 item 4 left the census method as "objdump, count call targets".
That names the symbol but not the line, and a 16 KB `search::<NT>` with eight
bounds checks tells you nothing about which index to mask.  This resolves them
exactly, and it is free: `core::panicking::panic_bounds_check` takes a
`&'static Location`, and rustc materialises that argument with a
`lea <addr>(%rip), %rdx` a few instructions ahead of the call.  A `Location` is
`{ file: &str (ptr,len), line: u32, col: u32 }` -- 24 bytes in `.rodata` -- so
reading it back gives file:line:col for every single check in the binary.

Usage:
  python3 tools/perf56/sites56.py bin56/ctl56 [symbol-substring]
"""
import re
import struct
import subprocess
import sys
from collections import Counter

BIN = sys.argv[1]
WANT = sys.argv[2] if len(sys.argv) > 2 else None

# ---- vaddr -> file offset, from the section table.
secs = []
for line in subprocess.run(["objdump", "-h", BIN], capture_output=True, text=True,
                           check=True).stdout.splitlines():
    f = line.split()
    if len(f) >= 7 and re.fullmatch(r"\d+", f[0]):
        size, vma, fileoff = int(f[2], 16), int(f[3], 16), int(f[5], 16)
        secs.append((vma, size, fileoff))
blob = open(BIN, "rb").read()


def read(vaddr, n):
    for vma, size, off in secs:
        if vma <= vaddr < vma + size:
            o = off + (vaddr - vma)
            return blob[o:o + n]
    return None


# The binary is a PIE, so the `file: &str` pointer inside every static
# `Location` is stored as ZERO and supplied at load time by an R_X86_64_RELATIVE
# relocation whose addend is the target vaddr. Without this table every Location
# decodes to a null file pointer -- which is exactly how the first version of
# this script reported "1 distinct source site" for 284 edges.
RELOC = {}
for line in subprocess.run(["readelf", "-r", "-W", BIN], capture_output=True, text=True,
                           check=True).stdout.splitlines():
    f = line.split()
    if len(f) >= 4 and f[2] == "R_X86_64_RELATIVE":
        try:
            RELOC[int(f[0], 16)] = int(f[3], 16)
        except ValueError:
            pass


def location(vaddr):
    """Decode a core::panic::Location at `vaddr` -> 'file:line:col'."""
    raw = read(vaddr, 24)
    if not raw or len(raw) < 24:
        return None
    ptr, ln, line, col = struct.unpack("<QQII", raw)
    if ptr == 0:
        ptr = RELOC.get(vaddr, 0)
    if not (0 < ln < 512) or ptr == 0:
        return None
    s = read(ptr, ln)
    if not s:
        return None
    try:
        f = s.decode("utf-8")
    except UnicodeDecodeError:
        return None
    if not f.endswith(".rs"):
        return None
    return f"{f}:{line}:{col}"


PANIC = re.compile(r"panic_bounds_check|slice_(start|end)_index_len_fail|slice_index_order_fail")
LEA = re.compile(r"lea\s+-?0x[0-9a-f]+\(%rip\),%r\w+\s+#\s+([0-9a-f]+)")

dis = subprocess.run(["objdump", "-C", "-d", "--no-show-raw-insn", BIN],
                     capture_output=True, text=True, check=True).stdout.splitlines()

sym, window, hits = None, [], []
for line in dis:
    m = re.match(r"^([0-9a-f]+) <(.+)>:$", line)
    if m:
        sym, window = m.group(2), []
        continue
    if sym is None:
        continue
    window.append(line)
    if len(window) > 14:
        window.pop(0)
    if "call" in line and PANIC.search(line):
        loc = None
        # rustc materialises the &Location a few instructions ahead of the call.
        for prev in reversed(window[:-1]):
            lm = LEA.search(prev)
            if lm:
                loc = location(int(lm.group(1), 16))
                if loc:
                    break
        hits.append((sym, loc or "<unresolved>"))

if WANT:
    hits = [h for h in hits if WANT in h[0]]

per_site = Counter(loc for _, loc in hits)
print(f"{BIN}: {len(hits)} bounds/range panic edges, "
      f"{len(per_site)} distinct source sites"
      + (f"  (filtered on '{WANT}')" if WANT else ""))
print()
own = [(n, s) for s, n in per_site.items() if "/src/" in s or s.startswith("src/")]
own.sort(reverse=True)
print(f"{'edges':>6}  ARCHON source site")
for n, s in own:
    syms = sorted({sym for sym, loc in hits if loc == s})
    print(f"{n:>6}  {s}")
    for y in syms:
        print(f"{'':>8}  in {y[:104]}")
