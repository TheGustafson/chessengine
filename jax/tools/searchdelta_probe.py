#!/usr/bin/env python3
"""Can the trunk predict SEARCH-vs-STATIC disagreement? (probe, no trainer)

WHY THIS EXISTS
    D57 and D58 both asked the head to predict the value head's own residual
    against its own training target, from the value head's own (stop-gradient)
    features. Measured on 1M heldout positions that is r^2 = 0.0017 signed /
    0.0266 magnitude -- uncorrelated, not mis-scaled, and unchanged when the
    residual was switched to the score-only column (D58: +0.0428 vs D57's
    +0.0446, with a control in the same job).

    That is not a tuning failure, it is structural: a CONVERGED model's
    residual against its own features is approximately unpredictable, because
    any generalisable error structure visible in those features would already
    have been absorbed by the value tail. No target inside that framing
    escapes it.

    SEARCH-vs-STATIC disagreement is a genuinely different quantity. The net
    is NOT trained on `search(pos) - static(pos)`; it is a property of the
    game tree below the position, not of the label. Tactical positions are
    recognisable from the features (hanging pieces, checks, loose sliders), so
    there is a real function to learn -- and it is exactly what an RFP margin
    or an LMR reduction should key on: "how much will search move this eval?"

WHAT THIS DOES (and deliberately does NOT do)
    A SELF-CONTAINED FEASIBILITY PROBE. It fits a small head on the frozen
    trunk activations and reports held-out correlation. It does NOT touch the
    trainer, the run specs, or the engine. If the correlation is not there,
    nothing further is worth building; if it is, the trainer plumbing is a
    known quantity (D58 proved `dis_score_only` end to end).

    Train/test split is WITHIN ONE CORPUS, unlike the D57/D58 heldout test
    which trained on band75-100 and scored on band0-5. That confound is
    removed on purpose: this asks "is the signal there at all", which must be
    answered before "does it transfer across bands".

STAGES
    1  decode N bulletformat records -> FENs
    2  (external) root-analysis  -> searched score per FEN
    2b (external) nnue-eval      -> static eval per FEN
    3  this script: trunk activations, fit, report held-out r

FEN RECONSTRUCTION, AND ITS ONE HONEST LIMITATION
    The record carries occupancy + 4-bit piece codes only: NO castling rights,
    NO en passant, NO side-to-move flag. `color = nib >> 3` is 0 = SIDE TO
    MOVE, 1 = opponent (bulletformat stores from the mover's view), so the
    faithful reconstruction is white-to-move with `- -`. That is a legal
    position and BOTH sides of the delta see the identical FEN, so the delta
    stays meaningful. It also matches what the net itself sees -- this engine
    and Stockfish are both measured blind to castling and ep (800/800
    bit-identical twins), so nothing is lost that the features ever had.
"""
import sys
import pathlib

import numpy as np

sys.path.insert(0, str(pathlib.Path(__file__).resolve().parents[1]))

from archon_jax.spec import RECORD_DTYPE  # noqa: E402

PT_CHAR = "pnbrqk"


def records_to_fens(recs: np.ndarray) -> list[str]:
    """Bulletformat records -> FEN strings, mover-to-move, no castling/ep.

    Mirrors `features.py`'s decoder exactly (occ little-endian bit order,
    nibble per occupied square in occ order, color = nib>>3, pt = nib&7) so
    the two cannot drift: a private re-derivation of that layout is how a
    corpus tool silently produces garbage.
    """
    n = len(recs)
    occ = np.ascontiguousarray(recs["occ"])
    bits = np.unpackbits(occ.view(np.uint8).reshape(n, 8), axis=1,
                         bitorder="little")
    row, sq = np.nonzero(bits)
    counts = bits.sum(axis=1)
    starts = np.zeros(n, dtype=np.int64)
    np.cumsum(counts[:-1], out=starts[1:])
    ordinal = np.arange(len(row)) - starts[row]
    pcs = recs["pcs"]
    nib = (pcs[row, ordinal >> 1] >> ((ordinal & 1) << 2)) & 0xF
    color = (nib >> 3).astype(np.int64)
    pt = (nib & 7).astype(np.int64)

    board = np.full((n, 64), ".", dtype="<U1")
    ch = np.array([PT_CHAR[p] for p in pt], dtype="<U1")
    upper = color == 0                      # 0 = side to move -> "white"
    ch = np.where(upper, np.char.upper(ch), ch)
    board[row, sq] = ch

    fens = []
    for i in range(n):
        b = board[i]
        ranks = []
        for r in range(7, -1, -1):          # FEN is rank 8 first; a1 = 0
            run, out = 0, []
            for f in range(8):
                c = b[r * 8 + f]
                if c == ".":
                    run += 1
                else:
                    if run:
                        out.append(str(run))
                        run = 0
                    out.append(c)
            if run:
                out.append(str(run))
            ranks.append("".join(out))
        fens.append("/".join(ranks) + " w - - 0 1")
    return fens


def main(bin_path: str, out_fens: str, n: int = 60000, offset: int = 0) -> int:
    recs = np.memmap(bin_path, dtype=RECORD_DTYPE, mode="r")
    sub = np.array(recs[offset:offset + n])
    fens = records_to_fens(sub)
    # Two kings or it is not a position the engine will accept. A record that
    # fails this is a decoder bug, not a bad record, so it is fatal rather
    # than filtered -- silently dropping them would hide the bug.
    bad = [i for i, f in enumerate(fens)
           if f.count("K") != 1 or f.count("k") != 1]
    if bad:
        print(f"FATAL: {len(bad)} FENs lack exactly one king each; "
              f"first={fens[bad[0]]!r}", file=sys.stderr)
        return 3
    pathlib.Path(out_fens).write_text("\n".join(fens) + "\n")
    np.save(out_fens + ".idx.npy", np.arange(offset, offset + n))
    print(f"wrote {len(fens):,} FENs -> {out_fens}")
    print(f"example: {fens[0]}")
    return 0


if __name__ == "__main__":
    if len(sys.argv) < 3:
        print(__doc__)
        raise SystemExit(2)
    raise SystemExit(main(sys.argv[1], sys.argv[2],
                          int(sys.argv[3]) if len(sys.argv) > 3 else 60000))
