#!/usr/bin/env python3
"""perf study 52 -- decompose ONE symbol into its inlined callees.

    python3 tools/perf52/inline33.py <perf.data> <binary> [symbol-regex]

`search::<NonPv>` is 19.9% of ARCHON's profile and 15,898 bytes of machine
code, because LTO inlines the TT probe, the move picker's callers, make/unmake,
the history updates and the accumulator push straight into it.  A symbol-level
profile therefore reports one number for a dozen different mechanisms, which is
exactly why study 28 could measure +303.4 cycles/node at this site and study 31
could still find no mechanism.

This tool splits it.  For every sample that lands in the symbol it reconstructs
the FULL INLINE STACK from DWARF (`addr2line -i`), then reports:

  * the INNERMOST frame -- the function whose code the sample is actually in,
    which is the decomposition into callees; and
  * the OUTERMOST-below-search frame -- which phase of the node the sample
    belongs to, aggregating a callee's whole subtree to its entry point.

Both are needed: `Board::make_move` inlining `put_piece` shows up as
`put_piece` in the first view and `make_move` in the second, and only the
second answers "how much of the node is make/unmake".

Addresses are handled via `sym+offset` from `perf script`, so PIE/ASLR never
enters: the file address is (symbol's static address) + offset.
"""

from __future__ import annotations

import collections
import re
import subprocess
import sys

SAMPLE = re.compile(r"^\s*([0-9a-f]+)\s+(\S+)\+(0x[0-9a-f]+)")


def sym_addrs(binary: str) -> dict[str, int]:
    out = subprocess.run(["nm", binary], capture_output=True, text=True).stdout
    d = {}
    for line in out.splitlines():
        f = line.split()
        if len(f) == 3:
            d[f[2]] = int(f[0], 16)
    return d


def demangle(names: list[str]) -> dict[str, str]:
    if not names:
        return {}
    # DWARF inline frames already carry readable DW_AT_name values; only the
    # ELF symbol names are mangled.  `rustfilt` is not installed on this box,
    # so fall back to binutils' Rust demangler and then to identity.
    for cmd in (["rustfilt"], ["c++filt", "--no-strip-underscore"]):
        try:
            out = subprocess.run(cmd, input="\n".join(names),
                                 capture_output=True, text=True)
        except FileNotFoundError:
            continue
        if out.returncode == 0 and len(out.stdout.splitlines()) == len(names):
            return dict(zip(names, out.stdout.splitlines()))
    return {n: n for n in names}


def short(name: str) -> str:
    """Trim a demangled Rust path to something readable, keeping the tail."""
    name = re.sub(r"::h[0-9a-f]{16}$", "", name)
    name = name.replace("chessengine::", "")
    name = re.sub(r"<(\w+) as [^>]+>", r"\1", name)
    return name


def main() -> int:
    data, binary = sys.argv[1], sys.argv[2]
    pat = sys.argv[3] if len(sys.argv) > 3 else r"6Worker6search.*5NonPv"

    addrs = sym_addrs(binary)
    target = [s for s in addrs if re.search(pat, s)]
    if not target:
        raise SystemExit(f"no symbol matching /{pat}/")
    target_base = {s: addrs[s] for s in target}

    script = subprocess.run(
        ["perf", "script", "-i", data, "-F", "ip,sym,symoff", "--no-demangle"],
        capture_output=True, text=True).stdout

    total = 0
    hits: collections.Counter[int] = collections.Counter()
    per_sym: collections.Counter[str] = collections.Counter()
    for line in script.splitlines():
        m = SAMPLE.match(line)
        if not m:
            continue
        total += 1
        sym, off = m.group(2), int(m.group(3), 16)
        per_sym[sym] += 1
        if sym in target_base:
            hits[target_base[sym] + off] += 1

    insym = sum(hits.values())
    print(f"\n  samples total {total:,}   in target symbol {insym:,} "
          f"({100.0 * insym / max(total, 1):.2f}%)")
    print(f"  distinct addresses in target: {len(hits):,}")

    # Resolve every distinct address once, with the full inline stack.
    uniq = sorted(hits)
    a2l = subprocess.run(
        ["addr2line", "-i", "-f", "-e", binary] + [f"0x{a:x}" for a in uniq],
        capture_output=True, text=True).stdout.splitlines()

    # addr2line -i -f emits alternating func/file:line, one PAIR per inline
    # frame, and the frames for one address run innermost-first.  There is no
    # blank separator, so the boundaries are recovered by re-querying without
    # -i to learn each address's frame COUNT... instead, ask for frame counts
    # cheaply: run again with a sentinel between addresses.
    frames: list[list[tuple[str, str]]] = []
    per_addr = subprocess.run(
        ["addr2line", "-i", "-f", "-e", binary, "-p"] + [f"0x{a:x}" for a in uniq],
        capture_output=True, text=True).stdout.splitlines()
    cur: list[tuple[str, str]] = []
    for ln in per_addr:
        if ln.startswith(" (inlined by) "):
            body = ln[len(" (inlined by) "):]
        else:
            if cur:
                frames.append(cur)
            cur = []
            body = ln
        fn, _, loc = body.partition(" at ")
        cur.append((fn.strip(), loc.strip()))
    if cur:
        frames.append(cur)
    if len(frames) != len(uniq):
        print(f"  WARNING: {len(frames)} resolved frame stacks for {len(uniq)} "
              f"addresses -- attribution below is unreliable")

    inner: collections.Counter[str] = collections.Counter()
    outer: collections.Counter[str] = collections.Counter()
    lines: collections.Counter[str] = collections.Counter()
    for addr, stack in zip(uniq, frames):
        n = hits[addr]
        # -p prints innermost FIRST, then "(inlined by)" callers outward.
        innermost_fn, innermost_loc = stack[0]
        inner[innermost_fn] += n
        lines[innermost_loc] += n
        # The frame just BELOW the search function itself: last element is the
        # outermost (search); take the one before it when there is one.
        outer[stack[-2][0] if len(stack) >= 2 else stack[-1][0]] += n

    names = set(inner) | set(outer)
    dm = demangle(sorted(names))

    def report(title: str, c: collections.Counter[str], n: int = 30) -> None:
        print(f"\n  {title}")
        print(f"  {'share':>7s} {'samples':>9s}  function")
        for name, cnt in c.most_common(n):
            print(f"  {100.0 * cnt / insym:6.2f}% {cnt:9,d}  "
                  f"{short(dm.get(name, name))[:96]}")

    report("INNERMOST inline frame -- which callee's code the sample is in", inner)
    report("OUTERMOST frame below search -- which PHASE of the node", outer)

    print(f"\n  TOP SOURCE LINES")
    print(f"  {'share':>7s} {'samples':>9s}  file:line")
    for loc, cnt in lines.most_common(25):
        print(f"  {100.0 * cnt / insym:6.2f}% {cnt:9,d}  {loc}")

    print(f"\n  TOP SYMBOLS ENGINE-WIDE (context)")
    for sym, cnt in per_sym.most_common(12):
        print(f"  {100.0 * cnt / total:6.2f}% {cnt:9,d}  "
              f"{short(demangle([sym]).get(sym, sym))[:96]}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
