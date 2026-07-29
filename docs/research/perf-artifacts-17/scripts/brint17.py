#!/usr/bin/env python3
"""Exact per-function CONDITIONAL BRANCH counts, SIMD kernels and start-up removed.

Study 17 §6.2: the callgrind frame runs the AVX2 fallback kernels because our
SIMD dispatch is runtime and valgrind's CPUID hides AVX-512, so any branch count
attributed to `avx2::*` describes code the engine does not ship.  Everything
ELSE in the binary is the same compiled code either way (up to the measured 4.8%
v3-vs-native codegen difference), so the integer rows ARE usable — and the
branch ratio is the one open question they can speak to.

Usage: brint17.py <archon-br.out> <archon-nodes> <sf-br.out> <sf-nodes>
"""

import re
import sys

sys.path.insert(0, __file__.rsplit("/", 1)[0])
from cgsum17 import parse, demangle  # noqa: E402

# `sse` as a bare substring matches "che-sse-ngine" and silently deletes EVERY
# ARCHON symbol -- the first run of this filter reported an integer subtotal of
# 3.5 branches per node, which is how it was caught. Anchor the module path.
SKIP = re.compile(
    r"avx2::|avx512::|::sse\d?::|read_leb_128|load_v2_bytes|init_magics|FromIterator|"
    r"Worker::clear|Worker>::clear|NNUE::permute|hash_bytes|get_process_affinity|"
    r"Network::load|apply_combined|update_accumulator_refresh_cache|"
    r"NetworkArchitecture::propagate|streambuf|istream|__mem|read_i16s|read_i32s|"
    r"iterative_deepening|SearchManager|_dl_|do_lookup|tolower|clock_gettime|"
    r"Rb_tree|cuckoo::build|core::fmt|Display>::fmt|String as|RawVec|"
    r"evaluate_side|update_accumulator_incremental|current_full|finny_refresh"
)


def rows(path, nodes):
    self_b, calls, _, _ = parse(path, event=1)  # event 1 = Bc
    out, tot = [], 0
    for fn, b in self_b.items():
        if not fn:
            continue
        tot += b
        if SKIP.search(fn):
            continue
        out.append((b / nodes, calls.get(fn, 0) / nodes, demangle(fn)))
    out.sort(reverse=True)
    return out, tot / nodes


BUCKETS = [
    ("threat gen", r"sota_threats|ThreatContext|refresh_threat|ensure_threats|ripple|"
                   r"record_|update_piece_threats|append_changed_indices|FullThreats"),
    ("movepick", r"MovePicker|partial_insertion_sort"),
    ("search body", r"Worker>::search|Worker::search|Worker>::qsearch|Worker::qsearch"),
    ("make/unmake", r"make_move_with|unmake_move|Position::do_move|Position::undo_move|"
                    r"set_check_info|refresh_check_info|from_move|update_slider_blockers|"
                    r"Worker>::do_move|Worker::do_move"),
    ("board queries", r"see_ge|gives_check|::legal|attackers_to|attacks_bb|is_draw|"
                      r"AttackTables>::attacks|upcoming_repetition|pseudo_legal"),
    # NOT `Stockfish::\(anonymous` -- that matches every SF anonymous-namespace
    # function, including `update_continuation_histories` and `correction_value`,
    # and silently reported SF's history branches as movegen (caught by SF's
    # `histories` row coming out at exactly 0.0, which is this study's own
    # "a block is not a symbol list" rule catching this study).
    ("movegen", r"movegen::generate|generate_moves|Stockfish::generate<"),
    ("histories", r"histor|Histories|correction_value|update_all_stats|update_quiet"),
    ("tt", r"TranspositionTable|TTEntry"),
]


def bucket(rr):
    acc = {n: 0.0 for n, _ in BUCKETS}
    other = 0.0
    for b, _c, fn in rr:
        for n, pat in BUCKETS:
            if re.search(pat, fn):
                acc[n] += b
                break
        else:
            other += b
    return acc, other


def main():
    a, atot = rows(sys.argv[1], float(sys.argv[2]))
    s, stot = rows(sys.argv[3], float(sys.argv[4]))
    ab, ao = bucket(a)
    sb, so = bucket(s)
    print("# branch buckets, integer code only, EXACT")
    print(f"{'block':<16}{'ARCHON':>9}{'SF':>9}{'delta':>9}")
    for n, _ in BUCKETS:
        print(f"{n:<16}{ab[n]:>9.1f}{sb[n]:>9.1f}{ab[n] - sb[n]:>+9.1f}")
    print(f"{'other':<16}{ao:>9.1f}{so:>9.1f}{ao - so:>+9.1f}")
    print(f"{'TOTAL':<16}{sum(ab.values()) + ao:>9.1f}{sum(sb.values()) + so:>9.1f}"
          f"{sum(ab.values()) + ao - sum(sb.values()) - so:>+9.1f}")
    print()
    print("# EXACT conditional branches per node (callgrind Bc); SIMD kernels and")
    print("# start-up EXCLUDED -- see perf study 17 section 6.2 for why the SIMD")
    print("# rows in this frame are an artifact and must not be read.")
    print(f"# all rows, including the excluded ones: ARCHON {atot:.1f}  SF {stot:.1f}")
    print(f"# integer-only subtotal:                 ARCHON {sum(r[0] for r in a):.1f}"
          f"  SF {sum(r[0] for r in s):.1f}")
    for label, rr in (("ARCHON", a), ("STOCKFISH", s)):
        print(f"\n== {label} ==")
        print(f"{'br/node':>9} {'calls/node':>11} {'br/call':>9}  function")
        for b, c, fn in rr[:24]:
            print(f"{b:9.2f} {c:11.4f} {(b / c if c else float('nan')):9.1f}  {fn[:88]}")


if __name__ == "__main__":
    main()
