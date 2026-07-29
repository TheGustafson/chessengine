#!/usr/bin/env python3
"""Study 13's block table, re-derived from EXACT callgrind self-costs.

Study 13 built its block table from two `perf record` flat profiles of two
DIFFERENT (stage-split, `noinline`) builds and warned in the same paragraph that
"split frames are not comparable in absolute terms".  This does the same
bucketing over callgrind's exact per-function self-Ir, from ONE build of each
engine, on the SAME 50 positions and the SAME node budget -- so the block
differences mean what they say.

The frame is AVX2 (valgrind cannot run AVX-512) and shallow (3000 nodes per
position), so ABSOLUTE totals are not the PGO headline's.  What survives the
frame is (a) any function whose call rate is ~1.0 per node, and (b) the
composition -- which block owns which share.

Usage: blocks17.py <archon.out> <archon-nodes> <sf.out> <sf-nodes>
"""

import re
import sys

sys.path.insert(0, __file__.rsplit("/", 1)[0])
from cgsum18 import parse  # noqa: E402

BLOCKS = [
    # startup first: it must not be stolen by a later pattern
    ("startup/io", [r"load_v2_bytes", r"init_magics", r"read_leb_128", r"FromIterator<i16>",
                    r"hash_bytes", r"Worker::clear", r"Worker>::clear", r"NNUE::permute",
                    r"Network::load", r"get_process_affinity", r"read_i16s", r"read_i8s",
                    r"streambuf", r"istream", r"build_threat_tables", r"AttackTables>::new",
                    r"Cuckoo", r"::main", r"lang_start", r"bench", r"from_fen", r"Board>::new",
                    r"init_state", r"TranspositionTable>::resize", r"TranspositionTable::resize",
                    # NOT a bare `tt::` -- it shadowed the whole `tt` bucket, which
                    # came out at exactly 0.0 for ARCHON. A bucket at zero is a bug in
                    # the filter, not a finding (D33 reviewer C).
                    r"tt::tests", r"::clear\(\)"]),
    ("search", [r"Worker>::search", r"Worker::search", r"Worker>::qsearch", r"Worker::qsearch",
                r"iterative_deepening", r"stat_score", r"update_all_stats", r"value_to_tt",
                r"Worker>::do_move", r"Worker::do_move\(", r"undo_move\(Stockfish::Move\)$"]),
    ("movepick", [r"MovePicker", r"partial_insertion_sort", r"movepick",
                  r"ensure_threats", r"compute_threats"]),
    ("eval head", [r"forward_into", r"::propagate", r"FeatureTransformer::transform",
                   r"l1_dot", r"l2_tail", r"find_nnz", r"activate", r"skip_dot",
                   r"Network::evaluate", r"SotaHead"]),
    # WARNING: SF's `update_accumulator_refresh_cache` also contains the INLINED
    # `FullThreats::append_active_indices`
    # (eca43a97:src/nnue/nnue_accumulator.cpp:577) -- a from-scratch enumeration
    # of every active threat feature, i.e. SF's counterpart of our
    # `refresh_threat`. It lands in "acc fold" here and NO regex can split it.
    # SF's "threat gen" row is therefore a LOWER BOUND (D33 reviewer B).
    ("acc fold", [r"i8_deltas", r"i16_deltas", r"apply_combined",
                  r"update_accumulator_incremental", r"update_accumulator_refresh_cache",
                  r"finny_refresh_into", r"HalfkaContext", r"AccumulatorStack",
                  r"halfka", r"HalfKA"]),
    # `ensure_threats`/`compute_threats` are the Wave-G MOVE-ORDERING attack map
    # (D30), consumed by `MovePicker::init_quiets` -- NOT the NNUE FullThreats
    # feature set. Stockfish's counterpart is `attacks_by<>` inlined into
    # `MovePicker::score`, which lands in "movepick". Bucketing ours into
    # "threat gen" was a like-for-unlike comparison and inflated our threat
    # column by 74.1 i/n (D33 reviewer B). They now go to movepick, which is
    # listed above this row so it wins.
    ("threat gen", [r"sota_threats", r"ThreatContext", r"refresh_threat",
                    r"update_piece_threats", r"append_changed_indices",
                    r"FullThreats", r"record_square_touch", r"record_targets",
                    r"ThreatPosition"]),
    ("eval glue", [r"SotaContext", r"EvalContext", r"Worker::evaluate", r"Worker>::evaluate",
                   r"eval::", r"NNUE::"]),
    ("make/unmake", [r"make_move_with", r"unmake_move", r"Position::do_move",
                     r"Position::undo_move", r"set_check_info", r"refresh_check_info",
                     r"DirtyPiece>::from_move", r"update_slider_blockers",
                     r"remove_piece", r"put_piece", r"move_piece", r"make_null", r"undo_null"]),
    ("board queries", [r"attackers_to", r">::legal", r"::legal\(", r"see_ge", r"gives_check",
                       r"pseudo_legal", r"both_attacks_bb", r"attacks_bb",
                       r"AttackTables>::attacks", r"upcoming_repetition", r"is_draw",
                       r"has_repeated", r"capture_stage", r"piece_on", r"Attacks::"]),
    ("movegen", [r"movegen::generate", r"generate_moves", r"Stockfish::generate<", r"MoveList"]),
    ("histories", [r"histor", r"Histories", r"correction_value", r"History"]),
    ("tt", [r"TranspositionTable", r"TTEntry", r"tt::"]),
    ("libc", [r"^__mem", r"^_int_", r"malloc", r"free$", r"^__mem.*_erms$"]),
]


def bucket(self_ir, nodes):
    acc = {name: 0.0 for name, _ in BLOCKS}
    other = []
    for fn, ir in self_ir.items():
        if not fn:
            continue
        for name, pats in BLOCKS:
            if any(re.search(p, fn) for p in pats):
                acc[name] += ir / nodes
                break
        else:
            other.append((ir / nodes, fn))
    return acc, other


def main():
    a_self, _, _, _ = parse(sys.argv[1])
    a_nodes = float(sys.argv[2])
    b_self, _, _, _ = parse(sys.argv[3])
    b_nodes = float(sys.argv[4])
    a, a_other = bucket(a_self, a_nodes)
    b, b_other = bucket(b_self, b_nodes)
    print(f"{'block':<16}{'ARCHON':>10}{'SF':>10}{'delta':>10}")
    ta = tb = 0.0
    for name, _ in BLOCKS:
        print(f"{name:<16}{a[name]:>10.1f}{b[name]:>10.1f}{a[name]-b[name]:>+10.1f}")
        ta += a[name]; tb += b[name]
    oa = sum(v for v, _ in a_other); ob = sum(v for v, _ in b_other)
    print(f"{'unbucketed':<16}{oa:>10.1f}{ob:>10.1f}{oa-ob:>+10.1f}")
    print(f"{'TOTAL':<16}{ta+oa:>10.1f}{tb+ob:>10.1f}{ta+oa-tb-ob:>+10.1f}")
    for label, rows in (("ARCHON", a_other), ("SF", b_other)):
        print(f"\n-- {label} unbucketed, top 10 --")
        for v, s in sorted(rows, reverse=True)[:10]:
            print(f"  {v:8.2f}  {s[:96]}")


if __name__ == "__main__":
    main()
