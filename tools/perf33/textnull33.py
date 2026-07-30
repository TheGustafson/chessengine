#!/usr/bin/env python3
"""perf study 33 -- THE NULL for the source-attribution instrument.

    python3 tools/perf33/textnull33.py bin33/base bin33/prof [<symbol regex>]

Study 33 reads its decomposition off a build carrying DWARF line tables
(`--profile profiling`, which inherits release and adds
`debug = "line-tables-only"`).  Every share it reports is a share of the
SHIPPED binary only if the line tables are codegen-inert.  Asserting that from
the LLVM documentation is not a null; this script measures it.

Raw `.text` bytes do NOT match, and that is expected for a benign reason: the
two cargo profiles get different `-C metadata`, so every v0 symbol carries a
different crate disambiguator, and the resulting symbol-table size difference
shifts `.text` by a constant.  The test that matters is therefore:

  1. identical instruction COUNT in the symbol,
  2. identical instruction LENGTHS, in order,
  3. identical MNEMONICS, in order,
  4. every differing byte confined to a displacement/immediate field and equal
     to the constant whole-symbol address shift.

If all four hold, the two builds execute the same instruction stream and a
per-instruction profile of one is a per-instruction profile of the other.
"""

from __future__ import annotations

import re
import subprocess
import sys

LINE = re.compile(r"^\s+([0-9a-f]+):\t([0-9a-f ]+?)\t+([a-z0-9.]+)\s*(.*)$")


def find_symbol(binary: str, pattern: str) -> str:
    out = subprocess.run(["nm", binary], capture_output=True, text=True).stdout
    for line in out.splitlines():
        name = line.split()[-1]
        if re.search(pattern, name):
            return name
    raise SystemExit(f"no symbol matching /{pattern}/ in {binary}")


def disasm(binary: str, sym: str) -> tuple[int, list[tuple[int, str, str, bytes]]]:
    out = subprocess.run(
        ["objdump", "-d", f"--disassemble={sym}", binary],
        capture_output=True, text=True,
    ).stdout
    base, insns = None, []
    for line in out.splitlines():
        m = LINE.match(line)
        if not m:
            continue
        addr = int(m.group(1), 16)
        if base is None:
            base = addr
        raw = bytes.fromhex(m.group(2).replace(" ", ""))
        insns.append((addr - base, m.group(3), m.group(4), raw))
    return base or 0, insns


def main() -> int:
    a_bin, b_bin = sys.argv[1], sys.argv[2]
    pat = sys.argv[3] if len(sys.argv) > 3 else r"6Worker6search.*5NonPv"

    a_sym, b_sym = find_symbol(a_bin, pat), find_symbol(b_bin, pat)
    a_base, a = disasm(a_bin, a_sym)
    b_base, b = disasm(b_bin, b_sym)
    shift = b_base - a_base

    print(f"  A {a_bin:12s} {a_sym[:44]}... @ 0x{a_base:x}  {len(a)} insns")
    print(f"  B {b_bin:12s} {b_sym[:44]}... @ 0x{b_base:x}  {len(b)} insns")
    print(f"  whole-symbol address shift: {shift:+d} bytes")

    ok = True
    if len(a) != len(b):
        print(f"  FAIL instruction count {len(a)} vs {len(b)}")
        return 1

    bad_len = bad_mn = bad_byte = 0
    shifted = 0
    deltas: dict[int, int] = {}
    for (oa, mna, opa, ra), (ob, mnb, opb, rb) in zip(a, b):
        if oa != ob or len(ra) != len(rb):
            bad_len += 1
            continue
        if mna != mnb:
            bad_mn += 1
            continue
        if ra == rb:
            continue
        # Bytes differ.  Accept ONLY if every differing byte is confined to ONE
        # 4-byte little-endian field -- a RIP-relative displacement -- and the
        # delta is a multiple of the section shift.  A .text->.text reference
        # keeps its displacement (both ends moved); a .text->.rodata reference
        # loses exactly `shift` (only the referrer moved).  Anything else is a
        # real codegen difference.
        diff = [i for i in range(len(ra)) if ra[i] != rb[i]]
        lo, hi = min(diff), max(diff)
        # x86 displacement/immediate fields are 1, 2, 4 or 8 bytes at an offset
        # this script does not decode, so try every window that covers exactly
        # the differing bytes and accept the one whose signed delta is +/- the
        # section shift.  0 => both ends moved (.text -> .text); -shift => only
        # the referrer moved (.text -> .rodata/.data).
        delta = None
        for width in (1, 2, 4, 8):
            for start in range(len(ra) - width + 1):
                if not (start <= lo and hi < start + width):
                    continue
                va = int.from_bytes(ra[start:start + width], "little", signed=True)
                vb = int.from_bytes(rb[start:start + width], "little", signed=True)
                if vb - va in (shift, -shift):
                    delta = vb - va
                    break
            if delta is not None:
                break
        if delta is not None:
            shifted += 1
            deltas[delta] = deltas.get(delta, 0) + 1
        else:
            bad_byte += 1
            if bad_byte <= 5:
                print(f"    UNEXPLAINED @+0x{oa:x} {mna:8s} {ra.hex()} vs {rb.hex()}")

    print(f"  offsets/lengths differing : {bad_len}")
    print(f"  mnemonics differing       : {bad_mn}")
    hist = ", ".join(f"{d:+d}:{n}" for d, n in sorted(deltas.items()))
    print(f"  bytes differing, all inside ONE 4-byte displacement field : {shifted}")
    print(f"    displacement deltas (each +/- the {shift:+d} section shift): {hist}")
    print(f"  bytes differing, UNEXPLAINED                     : {bad_byte}")
    ok = (bad_len == 0 and bad_mn == 0 and bad_byte == 0)
    print("  NULL PASSES -- identical instruction stream; line tables are codegen-inert."
          if ok else "  NULL FAILS -- the profiling build is not the release build.")
    return 0 if ok else 1


if __name__ == "__main__":
    raise SystemExit(main())
