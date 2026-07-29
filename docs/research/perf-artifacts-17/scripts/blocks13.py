#!/usr/bin/env python3
"""Bucket two `perf report --stdio` flat profiles into comparable BLOCKS.

Study 12 compared ARCHON and Stockfish one symbol at a time and had to correct
itself twice for mixing frames.  This does the bucketing once, from a table of
regexes, and prints both engines' i/n side by side with the difference — so a
block that is at parity cannot be quoted as a gap.

Usage: blocks13.py <archon-flat> <archon-total-i/n> <sf-flat> <sf-total-i/n>
"""

import re
import sys

BLOCKS = [
    ("search", [r"Worker>::search", r"Worker::search", r"Worker>::qsearch", r"Worker::qsearch"]),
    (
        "movepick",
        [r"MovePicker", r"partial_insertion_sort", r"movepick"],
    ),
    ("eval head", [r"forward_into", r"::propagate", r"FeatureTransformer::transform"]),
    (
        "acc fold",
        [
            r"i8_deltas",
            r"i16_deltas",
            r"apply_combined",
            r"update_accumulator_incremental",
            r"update_accumulator_refresh_cache",
            r"finny_refresh_into",
            r"HalfkaContext>::current_full",
            r"AccumulatorStack::evaluate_side",
        ],
    ),
    (
        "threat gen",
        [
            r"sota_threats",
            r"update_piece_threats",
            r"append_changed_indices",
            r"ThreatContext",
            r"refresh_threat",
            r"ensure_threats",
            r"compute_threats",
        ],
    ),
    (
        "eval glue",
        [r"SotaContext", r"NNUE::Network::evaluate", r"EvalContext", r"Worker::evaluate"],
    ),
    (
        "make/unmake",
        [
            r"make_move_with",
            r"unmake_move",
            r"Worker>::do_move",
            r"Worker::do_move",
            r"Position::do_move",
            r"Position::undo_move",
            r"set_check_info",
            r"DirtyPiece>::from_move",
            r"update_slider_blockers",
            r"remove_piece",
            r"put_piece",
        ],
    ),
    (
        "board queries",
        [
            r"attackers_to",
            r">::legal",
            r"::legal\(",
            r"see_ge",
            r"gives_check",
            r"pseudo_legal",
            r"both_attacks_bb",
            r"attacks_bb",
            r"AttackTables>::attacks",
            r"upcoming_repetition",
            r"is_draw",
        ],
    ),
    ("movegen", [r"movegen::generate", r"generate_moves", r"Stockfish::generate<"]),
    # `histor` catches BOTH `update_continuation_histories` (SF) and
    # `Histories`/`history` (ARCHON). It also matches SF's
    # `do_move(..., const SharedHistories*)` signature, which is why this row
    # sits BELOW make/unmake: the first matching bucket wins.
    ("histories", [r"histor", r"correction_value", r"update_all_stats"]),
    ("tt", [r"TranspositionTable", r"TTEntry"]),
    (
        "startup/io",
        [r"load_v2_bytes", r"init_magics", r"read_leb_128", r"Display>::fmt", r"hash_bytes", r"Worker::clear", r"Worker>::clear", r"malloc", r"memset", r"memcpy", r"free"],
    ),
]

ROW = re.compile(r"^\s+([0-9.]+)%\s+\[\.\]\s+(.*)$")


def load(path: str, total: float) -> list[tuple[float, str]]:
    out = []
    with open(path, encoding="utf-8", errors="replace") as fh:
        for line in fh:
            m = ROW.match(line)
            if m:
                out.append((float(m.group(1)) / 100.0 * total, m.group(2).strip()))
    return out


def bucket(rows: list[tuple[float, str]]) -> tuple[dict[str, float], list[tuple[float, str]]]:
    acc = {name: 0.0 for name, _ in BLOCKS}
    other: list[tuple[float, str]] = []
    for val, sym in rows:
        for name, pats in BLOCKS:
            if any(re.search(p, sym) for p in pats):
                acc[name] += val
                break
        else:
            other.append((val, sym))
    return acc, other


def main() -> None:
    a_rows = load(sys.argv[1], float(sys.argv[2]))
    b_rows = load(sys.argv[3], float(sys.argv[4]))
    a, a_other = bucket(a_rows)
    b, b_other = bucket(b_rows)
    print(f"{'block':<16}{'ARCHON':>9}{'SF':>9}{'delta':>9}")
    tot_a = tot_b = 0.0
    for name, _ in BLOCKS:
        print(f"{name:<16}{a[name]:>9.0f}{b[name]:>9.0f}{a[name]-b[name]:>+9.0f}")
        tot_a += a[name]
        tot_b += b[name]
    oa = sum(v for v, _ in a_other)
    ob = sum(v for v, _ in b_other)
    print(f"{'unbucketed':<16}{oa:>9.0f}{ob:>9.0f}{oa-ob:>+9.0f}")
    print(f"{'TOTAL':<16}{tot_a+oa:>9.0f}{tot_b+ob:>9.0f}{tot_a+oa-tot_b-ob:>+9.0f}")
    print("\n-- ARCHON unbucketed, top 12 --")
    for v, s in sorted(a_other, reverse=True)[:12]:
        print(f"  {v:8.1f}  {s[:88]}")
    print("-- SF unbucketed, top 12 --")
    for v, s in sorted(b_other, reverse=True)[:12]:
        print(f"  {v:8.1f}  {s[:88]}")


if __name__ == "__main__":
    main()
