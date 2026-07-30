#!/usr/bin/env python3
"""perf study 41 — instruction DENSITY of the hot functions.

    python3 tools/perf41/density41.py <binary>

The front end's currency is 64-byte instruction FETCHES, not instructions, so
what a function costs the fetcher is its BYTES. AVX-512 code is long: an
EVEX-prefixed instruction runs 6-10 bytes where the scalar equivalent is 3-4.
Bytes per instruction is what converts "fewer instructions" into "fewer
fetches", and it is where the two diverge.

Symbol ranges come from `nm` (address + size) and instructions are counted by
address containment, so a `.llvm.` clone or a `_generic` sibling with a similar
name cannot be folded into the wrong function — which is exactly what an
earlier substring-matching version of this did, reporting `forward_into` as 271
bytes holding 850 instructions.
"""

import bisect
import re
import subprocess
import sys

HOT = [
    "<chessengine::search::Worker>::search::<chessengine::search::NonPv>",
    "<chessengine::search::Worker>::qsearch::<chessengine::search::NonPv>",
    "<chessengine::eval::nnue::sota_threats::ThreatContext>::at_full",
    "chessengine::eval::nnue::sota_features::refresh_threat_into_at",
    "<chessengine::movepick::MovePicker>::init_quiets",
    "<chessengine::movepick::MovePicker>::init_captures",
    "chessengine::movegen::generate::<2>",
    "chessengine::movegen::generate::<3>",
    "<chessengine::eval::nnue::sota_accumulator::HalfkaContext>::finny_refresh_into",
    "chessengine::eval::nnue::sota_simd::avx512::i8_deltas_into",
    "chessengine::eval::nnue::sota_simd::avx512::i8_deltas_into_generic",
    "<chessengine::eval::nnue::sota_head::SotaHead>::forward_into",
]


def main() -> None:
    path = sys.argv[1]
    nm = subprocess.run(
        ["nm", "-C", "--print-size", "--numeric-sort", path],
        capture_output=True, text=True, check=True,
    ).stdout
    ranges: dict[str, tuple[int, int]] = {}
    for ln in nm.splitlines():
        p = ln.split(None, 3)
        if len(p) == 4 and p[2] in ("t", "T"):
            name = p[3].strip()
            if name in HOT and name not in ranges:
                ranges[name] = (int(p[0], 16), int(p[1], 16))

    dis = subprocess.run(
        ["objdump", "-d", "--no-show-raw-insn", path],
        capture_output=True, text=True, check=True,
    ).stdout
    addrs = []
    pat = re.compile(r"^\s+([0-9a-f]+):\t")
    for ln in dis.splitlines():
        m = pat.match(ln)
        if m:
            addrs.append(int(m.group(1), 16))
    addrs.sort()

    print(f"{'symbol':<58} {'bytes':>7} {'insns':>7} {'B/insn':>7}")
    tb = ti = 0
    for name in HOT:
        if name not in ranges:
            continue
        lo, size = ranges[name]
        n = bisect.bisect_left(addrs, lo + size) - bisect.bisect_left(addrs, lo)
        tb += size
        ti += n
        short = name.replace("chessengine::", "")
        print(f"{short[:58]:<58} {size:>7} {n:>7} {size / n:>7.2f}" if n else "")
    print(f"{'HOT TOTAL':<58} {tb:>7} {ti:>7} {tb / ti:>7.2f}")
    print()
    print(f"at 64 B per instruction fetch, {64 * ti / tb:.2f} instructions fit in one fetch block")


if __name__ == "__main__":
    main()
