#!/usr/bin/env python3
"""ARCHON vs Stockfish on the SAME positions: exact eval bias and tree shape.

Two questions the game logs raise but cannot answer, because a PGN only ever
has one engine's opinion of any given position.

**1. Is our eval biased, exactly?** `loss_positions.py --eval-bias` compares
our eval at ply N with the opponent's at ply N+1. Across a long match that is
informative, but it conflates eval bias with move quality (if our move was
bad, the position really did get worse) and it is contaminated by clipped
mate scores in already-lost endgames. Running both engines on the *identical*
position removes both problems.

**2. Are our plies too cheap?** In real games we report depth 20 where SF18
reports 18 — and lose. Depth is not comparable across engines: we prune
harder, so a ply of ours contains less. The honest measure is **nodes spent
to reach a given depth**, which is tree WIDTH. If ours is far smaller at the
same depth, we are buying depth we have not earned, and the fix is to trade
some back for width.

Both engines are run to FIXED DEPTH, not fixed nodes or time, because that is
the only setting where "nodes per ply" is a meaningful comparison. Fixed
depth is also load-immune, which matters on a box with a trainer on it.

Frames: ARCHON's UCI `cp` is raw internal (~0.64x SF18's); Stockfish's is
normalised. `--our-scale` converts ours before comparison. Mate scores are
excluded rather than clipped — a mate is not an eval and averaging it
corrupts a bias estimate.

    python3 tools/engine_compare.py positions.txt --depth 12 --limit 200
"""

from __future__ import annotations

import argparse
import math
import statistics as st

import chess
import chess.engine

ARCHON = "target/release/chessengine"
NET = "data/nnue/nets/screens/d47v-a4b2-cal400.cenn"
SF18 = "engines/bin/stockfish18"
# The static path needs a binary whose `nnue-eval` can READ this net. The
# plain `--features tune` build rejects it ("CENN bad magic ... CEN2"): the
# CEN2 reader lives behind the `sota` feature, so a tune-only build silently
# has no static eval at all. This PGO binary carries it.
ARCHON_STATIC = "engines/bin/archon-d47v-pgo"


def probe(binary: str, fens: list[str], depth: int, options: dict | None = None):
    """(eval in pawns or None for mate, nodes) per FEN, at fixed depth."""
    eng = chess.engine.SimpleEngine.popen_uci(binary)
    out = []
    try:
        for k, v in (options or {}).items():
            eng.configure({k: v})
        for f in fens:
            info = eng.analyse(chess.Board(f), chess.engine.Limit(depth=depth))
            sc = info["score"].relative
            cp = sc.score()          # None when it is a mate score
            out.append((None if cp is None else cp / 100.0, info.get("nodes")))
    finally:
        eng.quit()
    return out


AS = (-72.32565836, 185.93832038, -144.58862193, 416.44950446)
NNUE_RE = __import__("re").compile(r"NNUE evaluation\s+([+-]?[\d.]+)")


def sf18_a(material: int) -> float:
    m = min(max(material, 17), 78) / 58.0
    return ((AS[0] * m + AS[1]) * m + AS[2]) * m + AS[3]


def material_of(fen: str) -> int:
    b = fen.split()[0]
    c = {k: b.count(k) for k in "PNBRQpnbrq"}
    return ((c["P"] + c["p"]) + 3 * (c["N"] + c["n"] + c["B"] + c["b"])
            + 5 * (c["R"] + c["r"]) + 9 * (c["Q"] + c["q"]))


def static_compare(fens: list[str], our_scale: float):
    """Raw NET output, no search, both engines, same positions.

    This is the question search cannot answer: when we misjudge a position,
    is it the NET that is wrong or the SEARCH that spoils a sound eval?
    Comparing searched scores conflates the two -- a good net can be dragged
    off by a bad tree and a weak net can be rescued by a good one.

    Both sides are taken PRE-BLEND and converted to a common frame. SF18's
    `eval` prints its NNUE evaluation in PAWNS (v/a), so multiplying by
    a(material) recovers internal v; ours is `nnue-eval`, already internal.
    In-check positions are dropped: Stockfish emits no NNUE line for them,
    and recovering values by index across a batch would silently misalign
    every position after the first one (this bit us once already).
    """
    import subprocess
    keep = [f for f in fens if not chess.Board(f).is_check()]
    ours = subprocess.run([ARCHON_STATIC, "nnue-eval", NET], input="\n".join(keep) + "\n",
                          text=True, capture_output=True).stdout.split()
    ev = subprocess.run([SF18], text=True, capture_output=True,
                        input="".join(f"position fen {f}\neval\n" for f in keep) + "quit\n")
    theirs = [float(x) for x in NNUE_RE.findall(ev.stdout + ev.stderr)]
    if len(ours) != len(keep):
        print(f"  WARNING: {len(ours)} static evals for {len(keep)} FENs -- "
              "is this binary built with the CEN2 (sota) reader? skipping")
        return
    if len(theirs) != len(keep):
        print(f"  WARNING: {len(theirs)} NNUE lines for {len(keep)} FENs -- "
              "index pairing unsafe, skipping static compare")
        return
    diffs = []
    for f, o, t in zip(keep, ours, theirs):
        try:
            oi = int(o)
        except ValueError:
            continue
        a = sf18_a(material_of(f))
        ti = t * a
        if f.split()[1] == "b":
            ti = -ti           # their `eval` is White-POV; ours is stm-relative
        # Convert both to SF18 pawns for a readable number.
        diffs.append((oi / our_scale - ti) / a)
    if diffs:
        print(f"\nSTATIC NET vs NET ({len(diffs)} positions, no search):")
        print(f"  median {st.median(diffs):+.2f} pawns   mean {st.fmean(diffs):+.2f}")
        print(f"  |diff| median {st.median([abs(d) for d in diffs]):.2f} pawns")
        print("  positive = our NET is more optimistic than SF18's, before any search")


OUR_OPTS: dict[str, int] = {}


def stability(fens: list[str], lo: int, hi: int, our_scale: float):
    """Iteration-to-iteration eval oscillation, both engines, same positions.

    Found by walking one losing position by hand: on it, our eval wandered
    over a 1.0-pawn range across depths 7-14 (-2.03 .. -1.06) while SF18 held
    within 0.35 -- and both engines ended on the SAME move with nearly the
    same PV. So the divergence there was not disagreement about the move, it
    was NOISE in how we score it.

    Instability is not cosmetic. A search whose value jumps between
    iterations re-searches more (wider tree for the same depth), and it
    corrupts time management, because a transient high reads as a reason to
    move or to keep thinking.

    Reported as the mean absolute change between consecutive iterations,
    which is what a time manager and an aspiration window actually see --
    not the standard deviation, which a single monotone trend would inflate.
    """
    def wobble(binary, options):
        eng = chess.engine.SimpleEngine.popen_uci(binary)
        per = []
        try:
            for k, v in (options or {}).items():
                eng.configure({k: v})
            for f in fens:
                seq = []
                for d in range(lo, hi + 1):
                    info = eng.analyse(chess.Board(f), chess.engine.Limit(depth=d))
                    cp = info["score"].relative.score()
                    if cp is not None:
                        seq.append(cp / 100.0)
                if len(seq) >= 3:
                    per.append(st.fmean(abs(b - a) for a, b in zip(seq, seq[1:])))
        finally:
            eng.quit()
        return per

    o = wobble(ARCHON, {"EvalFile": NET, **OUR_OPTS})
    t = wobble(SF18, None)
    if o and t:
        om = st.median(o) / our_scale        # our cp is raw internal
        tm = st.median(t)
        print(f"\nITERATION STABILITY, depths {lo}-{hi} ({len(o)} positions):")
        print(f"  ARCHON  mean |eval change| per iteration: {om:.3f} pawns")
        print(f"  SF18    mean |eval change| per iteration: {tm:.3f} pawns")
        print(f"  ratio {om / tm:.2f}x  -- higher means our search value is noisier,")
        print("  which costs re-searches (tree width) and misleads time management.")


def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("positions", help="one FEN per line")
    ap.add_argument("--depth", type=int, default=12)
    ap.add_argument("--depths", default="", help="comma list -> nodes-per-ply curve")
    ap.add_argument("--limit", type=int, default=200)
    ap.add_argument("--our-scale", type=float, default=0.64)
    ap.add_argument("--stability", action="store_true",
                    help="iteration-to-iteration eval oscillation vs SF18")
    ap.add_argument("--our-options", default="",
                    help="K=V,K=V applied to ARCHON only -- lets a candidate "
                    "be re-measured against the SAME forensics that diagnosed "
                    "the defect, which is the point of having them")
    ap.add_argument("--static", action="store_true",
                    help="also compare raw NET output with no search")
    args = ap.parse_args()

    fens = []
    for line in open(args.positions):
        line = line.strip()
        if not line or line.startswith("#"):
            continue
        b = chess.Board(line)
        if not b.is_game_over() and len(list(b.legal_moves)) >= 2:
            fens.append(line)
        if len(fens) >= args.limit:
            break
    print(f"{len(fens)} positions\n")

    our_ov = {}
    for tok in filter(None, (t.strip() for t in args.our_options.split(","))):
        k, _, v = tok.partition("=")
        our_ov[k.strip()] = int(v)
    if our_ov:
        print(f"ARCHON options: {our_ov}\n")

    depths = [int(d) for d in args.depths.split(",")] if args.depths else [args.depth]

    print(f"{'depth':>6}{'ARCHON nodes':>15}{'SF18 nodes':>13}{'width ratio':>13}"
          f"{'eval bias':>12}")
    for d in depths:
        a = probe(ARCHON, fens, d, {"EvalFile": NET, **our_ov})
        s = probe(SF18, fens, d)
        an = [n for _, n in a if n]
        sn = [n for _, n in s if n]
        # Bias only over positions where BOTH returned a real eval: a mate
        # score is not an eval, and clipping it corrupts the mean.
        both = [(ae * (1.0 / args.our_scale), se)
                for (ae, _), (se, _) in zip(a, s) if ae is not None and se is not None]
        bias = st.median([x - y for x, y in both]) if both else float("nan")
        ratio = (st.median(an) / st.median(sn)) if an and sn else float("nan")
        print(f"{d:>6}{st.median(an):>15,.0f}{st.median(sn):>13,.0f}"
              f"{ratio:>13.2f}{bias:>+12.2f}")

    if args.stability:
        OUR_OPTS.update(our_ov)
        stability(fens, 6, 14, args.our_scale)
    if args.static:
        static_compare(fens, args.our_scale)

    print("\nwidth ratio = our median nodes / theirs at the SAME depth.")
    print("  < 1 means our tree is THINNER: we reach that depth on fewer nodes,")
    print("  so our plies contain less and the depth number flatters us.")
    print("eval bias  = median(our eval - their eval), our POV, common frame,")
    print("  mate scores EXCLUDED. Positive = we think we are better than we are.")


if __name__ == "__main__":
    main()
