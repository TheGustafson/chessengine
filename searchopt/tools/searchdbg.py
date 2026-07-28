#!/usr/bin/env python3
"""ARCHON search debugger — walk the ACTUAL search on a failing position.

WHY this exists.  Every broad instrument tried on the 2026-07-28 loss corpus
died the same way: the argmax of a search is a chaotic function of its pruning
constants (an inert +-1 flips 25% of positions; disabling ALL gates recovers
FEWER positions than disabling one), so differential black-box testing cannot
localise anything.  This reads the search's own decisions instead of inferring
them from outside.

Requires a binary built with `--features search-control`:

    RUSTFLAGS="-C target-cpu=x86-64-v3" cargo build --release \
        --features sota,arch-d47v,search-control

TWO TRAPS, both of which cost hours before this tool worked:

1. **stdin EOF is a STOP.**  `printf ... | engine` aborts the search the instant
   the pipe closes, so it returns a fallback move and the trace stays EMPTY.
   Every depth returns the same bestmove, which is the tell.  This module drives
   the engine through the persistent-stdin harness in `uciprobe`.
2. **The trace used to be coupled to the D41 policy latent.**  Rows were pushed
   only when `classic_policy_accumulators` returned Some; policy was ditched
   (pol_lambda 0), so on sota-v2 nets every row was silently dropped while the
   file still got its header — armed-looking and empty.  Fixed in
   `search/mod.rs` by separating selection from latent capture.

WHAT IT CAN AND CANNOT SEE.  Rows are pushed after a move is searched, so the
trace contains only moves that SURVIVED the shallow-pruning block.  A move
pruned by razoring/futility/SEE/LMP never appears.  Absence at a node where
siblings are present is therefore the signature of a prune, and `--absent`
reports exactly that.
"""

from __future__ import annotations

import argparse
import json
import os
import subprocess
import sys
from collections import defaultdict

sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
from uciprobe import Engine, NET, REPO  # noqa: E402

FILES = "abcdefgh"


def sq(i: int) -> str:
    return f"{FILES[i & 7]}{(i >> 3) + 1}"


def decode(raw: int) -> str:
    """u16 -> UCI.  Encoding (design/05): 0-5 from, 6-11 to, 12-13 promo,
    14-15 kind (0 normal, 1 promotion, 2 en-passant, 3 castling).  Castling is
    encoded as the KING's move (e1g1), so no rook-square special case."""
    frm, to = raw & 63, (raw >> 6) & 63
    s = sq(frm) + sq(to)
    if (raw >> 14) & 3 == 1:
        s += "nbrq"[(raw >> 12) & 3]
    return s


def run_trace(fen, depth, engine_path, net, hash_mb, out_path, nodes=None,
              min_depth=2, max_rows=200_000, options=None):
    """Run one search with tracing armed and return the parsed rows."""
    if os.path.exists(out_path):
        os.remove(out_path)
    env = dict(os.environ)
    env.update(
        ARCHON_SEARCH_TRACE_PATH=out_path,
        ARCHON_SEARCH_TRACE_SAMPLE_LOG2="0",
        ARCHON_SEARCH_TRACE_MIN_DEPTH=str(min_depth),
        ARCHON_SEARCH_TRACE_MAX_ROWS=str(max_rows),
        ARCHON_SEARCH_TRACE_NO_LATENT="1",
        ARCHON_SEARCH_TRACE_ALL_MOVES="1",
    )
    # The harness spawns the engine itself; push the env through os.environ so
    # the child inherits it (Engine deliberately owns process creation, because
    # that is where the stdin-EOF trap lives).
    saved = {k: os.environ.get(k) for k in env if k.startswith("ARCHON_SEARCH_TRACE")}
    os.environ.update({k: v for k, v in env.items() if k.startswith("ARCHON_SEARCH_TRACE")})
    try:
        e = Engine(engine_path, net=net, hash_mb=hash_mb, options=options, name="dbg")
        e.assert_net_live()
        g = e.go(fen=fen, depth=depth, nodes=nodes)
        # NOT e.close(): it waits only 5 s then SIGKILLs.  The trace flush is
        # one unbuffered write per field, so ~70k rows takes far longer than
        # that and the engine dies mid-line, leaving a truncated final record.
        e.send("quit")
        try:
            e.p.wait(timeout=600)
        except subprocess.TimeoutExpired:
            e.p.kill()
    finally:
        for k, v in saved.items():
            if v is None:
                os.environ.pop(k, None)
            else:
                os.environ[k] = v

    rows, meta, bad = [], {}, 0
    with open(out_path) as fh:
        for line in fh:
            try:
                r = json.loads(line)
            except json.JSONDecodeError:
                bad += 1          # tolerate a truncated tail; never silently
                continue          # (the count is printed) — see close() above
            if r.get("kind") == "row":
                r["uci"] = decode(r["raw_move"])
                rows.append(r)
            elif r.get("kind") == "go":
                meta = r
    if bad:
        print(f"  [warn] {bad} malformed trace line(s) skipped (flush truncated)")
    return g, rows, meta


NODE_KIND = {0: "root", 1: "pv", 2: "nonpv"}


def report_root(rows, interest, top=6):
    """One line per (depth, move): the LAST decision at that depth.

    Aspiration re-searches emit several rows per move per depth; printing them
    all buries the decision in noise. The last row is the one that stood."""
    root = [r for r in rows if r["ply"] == 0]
    if not root:
        print("  (no root rows)")
        return
    last = {}
    for r in root:
        last[(r["depth"], r["uci"])] = r        # later rows overwrite earlier
    depths = sorted({d for d, _ in last})
    print(f"\n{'d':>3} {'move':>7} {'idx':>4} {'reduc':>6} {'ext':>3} {'scoutD':>6} "
          f"{'score':>7} {'re':>3} {'a+':>3} {'cut':>3} {'nodes':>9}")
    for d in depths:
        at_d = sorted([r for (dd, _), r in last.items() if dd == d],
                      key=lambda r: r["move_index"])
        shown = [r for r in at_d if r["uci"] in interest]
        shown += [r for r in at_d if r["uci"] not in interest][:top]
        for r in shown:
            mark = "*" if r["uci"] in interest else " "
            print(f"{d:>3} {r['uci']:>6}{mark} {r['move_index']:>4} "
                  f"{r['final_reduction']:>6} {r['extension']:>3} {r['scout_depth']:>6} "
                  f"{r['final_score']:>7} "
                  f"{'Y' if r['lmr_research'] or r['pv_research'] else '.':>3} "
                  f"{'Y' if r['alpha_raise'] else '.':>3} "
                  f"{'Y' if r['beta_cutoff'] else '.':>3} {r['total_nodes']:>9,}")


def verdict(rows, mv):
    """METHOD STEP 3: the one sentence, or an explicit statement of what is missing."""
    hits = [r for r in rows if r["uci"] == mv]
    print(f"\n=== VERDICT for {mv} ===")
    if not hits:
        print(f"  {mv} is NEVER SEARCHED anywhere in this trace. It is pruned before the")
        print(f"  move loop's push at every node — a gate deleted it, not a reduction.")
        return
    deep = max(hits, key=lambda r: (r["depth"], -r["move_index"]))
    RU = 1024
    print(f"  At depth {deep['depth']} (ply {deep['ply']}), {mv} is ordered "
          f"#{deep['move_index']}, reduced by {deep['final_reduction']/RU:+.2f} plies,")
    print(f"  scouted at depth {deep['scout_depth']} of {deep['depth']}, returns "
          f"{deep['final_score']} against alpha {deep['alpha']},")
    researched = deep["lmr_research"] or deep["pv_research"]
    print(f"  and is {'RE-SEARCHED' if researched else 'NOT re-searched'}.")
    n_ext = sum(1 for r in hits if r["extension"] != 0)
    print(f"  Seen at {len(hits)} nodes; extended at {n_ext}; "
          f"median order #{sorted(r['move_index'] for r in hits)[len(hits)//2]}.")
    if deep["move_index"] > 6 and deep["final_reduction"] > RU:
        print(f"  >> ordered late AND heavily reduced — ordering is upstream of the reduction")
        print(f"     (reduction scales with ln(move_index)); look at why it sorts there.")
    elif deep["final_reduction"] <= 0:
        print(f"  >> NOT reduced (extended, if anything). Ordering and LMR are innocent here;")
        print(f"     the error is inside this move's SUBTREE — walk one ply deeper.")


def report_move(rows, mv, limit=14):
    """Everywhere a move of interest was searched, deepest-first."""
    hits = [r for r in rows if r["uci"] == mv]
    if not hits:
        print(f"\n  {mv}: NEVER SEARCHED anywhere in the tree "
              f"(pruned before the move loop's push, at every node)")
        return
    hits.sort(key=lambda r: (-r["depth"], r["ply"]))
    print(f"\n  {mv}: searched at {len(hits)} nodes")
    print(f"    {'d':>3} {'ply':>3} {'kind':>5} {'#':>3} {'r_fin':>5} {'ext':>3} "
          f"{'scout':>7} {'final':>7} {'alpha':>7} {'beta':>7} {'LMRre':>5} {'b!':>2}")
    for r in hits[:limit]:
        print(f"    {r['depth']:>3} {r['ply']:>3} {NODE_KIND.get(r['node_kind'],'?'):>5} "
              f"{r['move_index']:>3} {r['final_reduction']:>5} {r['extension']:>3} "
              f"{r['scout_score']:>7} {r['final_score']:>7} {r['alpha']:>7} {r['beta']:>7} "
              f"{str(r['lmr_research'])[0]:>5} {str(r['beta_cutoff'])[0]:>2}")
    if len(hits) > limit:
        print(f"    ... {len(hits)-limit} more")


def report_absent(rows, mv):
    """Nodes where this move was legal-and-considered elsewhere but absent here.

    A move missing from a node whose siblings are present was pruned before the
    push — this is the only way the trace can show a prune, since pruned moves
    `continue` out of the loop before any row is written."""
    nodes = defaultdict(set)
    for r in rows:
        nodes[(r["position_key"], r["depth"])].add(r["uci"])
    present = sum(1 for v in nodes.values() if mv in v)
    print(f"\n  {mv}: present at {present}/{len(nodes)} traced (position,depth) nodes")


def main():
    ap = argparse.ArgumentParser(description=__doc__,
                                 formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--fen", required=True)
    ap.add_argument("--depth", type=int, default=14)
    ap.add_argument("--nodes", type=int)
    ap.add_argument("--moves", default="", help="comma-separated moves of interest")
    ap.add_argument("--engine", default="engines/bin/archon-trace",
                    help="MUST be built --features search-control; a default-feature\n"
                         "rebuild silently overwrites target/release/chessengine")
    ap.add_argument("--net", default=NET)
    ap.add_argument("--hash", type=int, default=64)
    ap.add_argument("--out", default="/tmp/archon_searchdbg.jsonl")
    ap.add_argument("--setoption", action="append", default=[],
                    help="NAME=VALUE, repeatable — for A/B against an arm")
    ap.add_argument("--top", type=int, default=6)
    ap.add_argument("--absent", action="store_true")
    ap.add_argument("--verbose", action="store_true",
                    help="also dump every node the move was searched at")
    a = ap.parse_args()

    interest = [m for m in a.moves.split(",") if m]
    opts = dict(kv.split("=", 1) for kv in a.setoption)
    os.chdir(REPO)
    g, rows, meta = run_trace(a.fen, a.depth, a.engine, a.net, a.hash, a.out,
                              nodes=a.nodes, options=opts)

    print(f"bestmove {g['best']}   reached depth {max(g['ladder']) if g['ladder'] else 0}"
          f"   trace rows {len(rows)} / considered {meta.get('considered', '?')}")
    if not rows:
        print("EMPTY TRACE — binary lacks --features search-control, or stdin EOF "
              "aborted the search (see module docstring).")
        return 1
    if g["ladder"]:
        top = g["ladder"][max(g["ladder"])]
        print(f"final pv {' '.join(top['pv'][:8])}  score {top['score']}")

    report_root(rows, set(interest), top=a.top)
    for mv in interest:
        verdict(rows, mv)
        if a.verbose:
            report_move(rows, mv)
        if a.absent:
            report_absent(rows, mv)
    return 0


if __name__ == "__main__":
    sys.exit(main())
