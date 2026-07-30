#!/usr/bin/env python3
"""perf study 41 — the HOT FOOTPRINT of a binary, read off its symbol table.

    python3 tools/perf41/hotsize41.py <binary>...

The L1 instruction cache on this box is 32 KB. This sums the byte sizes of the
functions the bench frame actually spends its time in, so "the hot set is N
times L1i" is a measurement rather than an impression. It is the independent
variable every layout arm in this study moves, and it is what decides whether
the icache miss rate is a capacity problem or a layout problem.

Matching is on the WHOLE demangled name, deliberately: a substring match on
`search::<...NonPv>` also matches `qsearch::<...NonPv>`, which silently reports
qsearch's size for both and makes the hot set look 20 KB smaller than it is.
"""

import subprocess
import sys

HOT = [
    "<chessengine::search::Worker>::search::<chessengine::search::NonPv>",
    "<chessengine::search::Worker>::search::<chessengine::search::Pv>",
    "<chessengine::search::Worker>::qsearch::<chessengine::search::NonPv>",
    "<chessengine::search::Worker>::qsearch::<chessengine::search::Pv>",
    "<chessengine::eval::nnue::sota_threats::ThreatContext>::at_full",
    "chessengine::eval::nnue::sota_features::refresh_threat_into_at",
    "<chessengine::movepick::MovePicker>::init_quiets",
    "<chessengine::movepick::MovePicker>::init_captures",
    "chessengine::movegen::generate::<2>",
    "chessengine::movegen::generate::<3>",
    "<chessengine::eval::nnue::sota_accumulator::HalfkaContext>::finny_refresh_into",
    "chessengine::eval::nnue::sota_simd::avx512::i8_deltas_into",
    "<chessengine::eval::nnue::sota_head::SotaHead>::forward_into",
]
L1I = 32 * 1024


def sizes(path: str) -> dict[str, int]:
    out = subprocess.run(
        ["nm", "-C", "--size-sort", "-td", path],
        capture_output=True, text=True, check=True,
    ).stdout
    table: dict[str, int] = {}
    for line in out.splitlines():
        parts = line.split(None, 2)
        if len(parts) != 3 or parts[1] not in ("t", "T"):
            continue
        table[parts[2].strip()] = max(int(parts[0]), table.get(parts[2].strip(), 0))
    return table


def main() -> None:
    for path in sys.argv[1:]:
        table = sizes(path)
        print(f"=== {path} ===")
        total = 0
        for name in HOT:
            sz = table.get(name, 0)
            total += sz
            short = name.replace("chessengine::", "").replace("<>", "")
            print(f"  {short[:62]:<62} {sz:>7}{'   MISSING' if sz == 0 else ''}")
        print(f"  {'TOTAL':<62} {total:>7}   = {total / L1I:.2f} x 32 KB L1i")


if __name__ == "__main__":
    main()
