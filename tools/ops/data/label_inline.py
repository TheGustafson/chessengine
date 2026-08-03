#!/usr/bin/env python3
"""Stage 3 WITHOUT running the oracle — use the score it already produced in play.

THE IDEA
    In `MODE=oracle` both players ARE the labelling engine, so every ply of
    every game carries a search the oracle already paid for: the score is in
    the PGN comment (`own_cp`) and the move it chose is the move it played
    (`played`). Re-searching those positions afterwards buys a second copy of
    an answer we already have. Skipping it removes the entire labelling stage
    and roughly DOUBLES corpus throughput (measured 5.85 -> 2.90 core-h per
    1M kept positions).

WHY THIS IS SAFE, MEASURED RATHER THAN ARGUED
    In-game scores are path-dependent in a way clean re-labels are not: the
    engine carries a warm transposition table, search history and killers
    across moves, and its root score can be a bound rather than an exact value.
    `label_positions.py` issues `ucinewgame` per position precisely to kill all
    of that, and that is what buys its 200/200 bit-identical null.

    So the question is whether path-dependence BIASES the label. Measured on
    3,437 paired positions from a 60-game oracle run at 5k/5k — every position
    scored both ways:

        signed mean (clean - in-game)   -2.16 cp
        |mean| / stdev                   0.027
        median |delta|                   19 cp
        stdev                            79.3
        within 10 cp                     34.1 %

    The study accepted shallow labels on exactly this test: unbiased noise,
    signed mean under 0.08 sigma, median deviation ~20 cp (§4.3 of
    selfplay-data-design-2026-08-02.md, where 5k-vs-200k read 0.059 sigma and
    median 20). In-game eval lands at 0.027 sigma and median 19 — inside the
    bar that was already judged harmless. A regression trained on zero-mean
    noise converges to the same conditional mean; it needs more data, not
    different data, and skipping the label pass is what buys the extra data.

WHY IT REFUSES ANY OTHER MODE
    In `self`/`sf18`/`sf16` the two sides are different engines, so at ARCHON's
    turns `played` is ARCHON's move and `own_cp` is ARCHON's eval in ARCHON's
    frame (~2.07x Stockfish's printed cp). Labelling from those would mix two
    frames into one column and call ARCHON's own opinion an oracle label — the
    exact self-distillation trap the teacher-blend result argues against. This
    tool reads the run manifest and exits non-zero unless mode == oracle.

THE FILTER IS IMPORTED, NOT COPIED
    `d46_verdict` comes from label_positions.py so the two stages cannot drift.
    That function is itself a line-by-line mirror of the normative Rust
    (tools/nnue/bullet/d40_features.rs::sfbinpack_filter).

USAGE
    python3 tools/ops/data/label_inline.py <run-dir> [--out labelled.tsv]
"""

from __future__ import annotations

import argparse
import json
import pathlib
import sys

sys.path.insert(0, str(pathlib.Path(__file__).resolve().parent))
import chess  # noqa: E402
from label_positions import (  # noqa: E402
    VALUE_SCORE_CAP,
    d46_verdict,
)

COLUMNS = ["fen", "label_cp", "raw_cp", "best", "result", "ply",
           "plies_to_end", "termination", "game_id", "drop_reason"]


def main() -> int:
    ap = argparse.ArgumentParser()
    ap.add_argument("rundir", type=pathlib.Path)
    ap.add_argument("--out", default=None)
    ap.add_argument("--positions", default=None)
    a = ap.parse_args()

    rundir = a.rundir
    manifest = rundir / "manifest.json"
    if not manifest.exists():
        print(f"FATAL: no manifest at {manifest}", file=sys.stderr)
        return 2
    m = json.loads(manifest.read_text())
    if m.get("mode") != "oracle":
        print(f"FATAL: mode is '{m.get('mode')}', not 'oracle'. In-game scores are "
              "only oracle labels when the oracle played BOTH sides; in any other "
              "mode half the rows carry ARCHON's own eval, in ARCHON's frame.",
              file=sys.stderr)
        return 2

    src = pathlib.Path(a.positions) if a.positions else rundir / "positions.tsv"
    dst = pathlib.Path(a.out) if a.out else rundir / "labelled.tsv"
    drop_path = dst.with_suffix(dst.suffix + ".dropped.tsv")

    n_in = kept = 0
    reasons: dict[str, int] = {}
    with src.open() as f, dst.open("w") as out, drop_path.open("w") as dropf:
        header = f.readline().rstrip("\n").split("\t")
        ix = {c: i for i, c in enumerate(header)}
        for need in ("fen", "ply", "result", "played", "own_cp"):
            if need not in ix:
                print(f"FATAL: positions.tsv lacks column '{need}'", file=sys.stderr)
                return 2
        out.write("\t".join(COLUMNS) + "\n")
        dropf.write("\t".join(COLUMNS) + "\n")

        for line in f:
            row = line.rstrip("\n").split("\t")
            if len(row) < len(header):
                continue
            n_in += 1
            fen = row[ix["fen"]]
            raw_s = row[ix["own_cp"]]
            best = row[ix["played"]]
            ply = int(row[ix["ply"]] or 0)

            # No score in the comment means a book move or a MATE score, which
            # pgn_positions deliberately refuses to turn into a number. Both are
            # non-labels, not zero-valued labels.
            if raw_s == "":
                reason = "noscore"
            else:
                raw = int(round(float(raw_s)))
                board = chess.Board(fen)
                reason = d46_verdict(board, best, raw, ply)

            fields = [fen, "", raw_s, best,
                      row[ix["result"]], str(ply),
                      row[ix.get("plies_to_end", 0)] if "plies_to_end" in ix else "",
                      row[ix["termination"]] if "termination" in ix else "",
                      row[ix["game_id"]] if "game_id" in ix else "",
                      reason or ""]
            if reason:
                reasons[reason] = reasons.get(reason, 0) + 1
                dropf.write("\t".join(fields) + "\n")
                continue
            # CLAMP LAST — the |raw| <= 10000 criterion was tested on the raw
            # score inside d46_verdict; only survivors are clamped.
            fields[1] = str(max(-VALUE_SCORE_CAP, min(VALUE_SCORE_CAP, raw)))
            out.write("\t".join(fields) + "\n")
            kept += 1

    stats = dict(positions_in=n_in, kept=kept, dropped=n_in - kept,
                 drop_rate_pct=round(100.0 * (n_in - kept) / n_in, 2) if n_in else 0.0,
                 drops_by_reason=reasons,
                 oracle=dict(path=m.get("engine", {}).get("path"),
                             sha256=m.get("engine", {}).get("sha256"),
                             nodes=m.get("nodes"),
                             label_frame="sf-printed-cp",
                             source="in-game (no re-search)"),
                 filter="D46 five-criterion (enforced, imported from label_positions)")
    (dst.parent / (dst.name + ".stats.json")).write_text(json.dumps(stats, indent=2) + "\n")
    print(json.dumps(stats))
    return 0


if __name__ == "__main__":
    sys.exit(main())
