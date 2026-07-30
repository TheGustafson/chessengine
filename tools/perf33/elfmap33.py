#!/usr/bin/env python3
"""perf study 33 -- runtime IP -> ELF virtual address, done correctly.

Shared by every IBS tool here.  It exists because getting this wrong is silent:
an off-by-one-page mapping still resolves every sample to *a* symbol, still
produces a plausible table, and still sums to 100%.  Study 33 shipped exactly
that error for one commit (see the retraction in the write-up) and caught it
only because `ibsnull33.py` cross-checked the home-made resolver against
`perf report`.

THE TRAP.  `perf script`'s MMAP2 record gives the runtime base of a mapping and
its FILE OFFSET.  `nm` reports ELF VIRTUAL addresses.  For this binary the
executable LOAD segment sits at file offset 0x3e150 but virtual address
0x3f150 -- they differ by 0x1000, because the linker page-aligns the segment's
vaddr independently of its offset.  So

    elf_vaddr = ip - mmap_vaddr + mmap_pgoff + (p_vaddr - p_offset)

and the last term is NOT zero and NOT constant across binaries.  Read it from
the program headers; never assume it.
"""

from __future__ import annotations

import re
import subprocess

MMAP = re.compile(r"\[0x([0-9a-f]+)\((0x[0-9a-f]+)\) @ (0x[0-9a-f]+|0)[^]]*\]: r-xp")


def text_delta(binary: str) -> int:
    """`p_vaddr - p_offset` for the executable LOAD segment."""
    out = subprocess.run(["readelf", "-lW", binary], capture_output=True, text=True).stdout
    for line in out.splitlines():
        f = line.split()
        if len(f) >= 8 and f[0] == "LOAD" and "E" in f[-2]:
            return int(f[2], 16) - int(f[1], 16)
    raise SystemExit(f"no executable LOAD segment in {binary}")


def ip_to_elf(mmap_path: str, binary: str):
    """Return a function mapping a runtime IP to an ELF virtual address."""
    m = MMAP.search(open(mmap_path).read())
    if not m:
        raise SystemExit(f"no r-xp mapping in {mmap_path}")
    vbase, pgoff = int(m.group(1), 16), int(m.group(3), 16)
    delta = text_delta(binary)
    return lambda ip: ip - vbase + pgoff + delta, vbase, pgoff, delta
