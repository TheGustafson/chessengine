#!/usr/bin/env python3
"""Is our static eval's ERROR the same absolute size as Stockfish's?

THE QUESTION THIS SETTLES. Our eval is a uniform ~0.65x rescale of SF18's
(quantile-matched, `eval_scale_study.py --bands`: T(x)/x is flat at 0.65 from
-1750 to +1550, so there is no tail saturation and no "margins operate in a
near-1.0 region" escape hatch). Every cp-denominated margin we own is
Stockfish's raw value. So the margins are ~1.5x larger relative to our eval
than SF's are relative to theirs -- and correcting that, in both directions,
lost Elo (docs/research/eval-frame-scale-model-2026-07-27.md).

The reconciliation on the table: **a margin tracks the eval's ERROR, not its
MAGNITUDE.** RFP asks "is `eval - beta` big enough that this node almost
certainly fails high?" That is a question about how far static eval typically
sits from the true search value. Stockfish tuned its constants against its
own eval's error. If our error, measured in OUR frame, is about the same
ABSOLUTE size as SF's error in THEIRS, then SF's raw constants are already
the right ones for us -- which is exactly what every rescale experiment
measured, and would explain a negative that currently has no mechanism.

That requires our eval to be ~1/0.65 = 1.54x noisier in relative terms. Not
implausible: it is the same deficit the equal-node Elo gap reports.

METHOD. For each position, take the engine's own static eval and its own
deep search score, both in that engine's internal frame, and call the
difference the error. Compare the two error distributions. No cross-engine
pairing is involved, so the ~1/3 sign-disagreement rate that wrecks paired
regressions is irrelevant here -- each engine is measured against its own
search, which is the only ground truth a margin can be tuned against.

Frames, which must not be conflated:
  ARCHON  `nnue-eval` and UCI `score cp` are both raw internal. No conversion.
  SF18    `eval` prints NNUE in PAWNS (v/a) and UCI `score cp` is normalised
          by the same material-dependent a(material). Both are multiplied
          back by a(material) to recover internal v.

WHAT THE ANSWER MEANS.
  errors comparable in absolute terms -> SF's raw margins are right for us,
      the frame ratio is a red herring for tuning, and margins should be
      retuned only AFTER the net's error drops (i.e. after D48X), not before.
  our error ~0.65x SF's (i.e. proportional) -> the frame argument was right
      after all and the negative needs a different explanation entirely.

    python3 tools/eval_error_study.py [fens] [--depth D] [--limit K]
"""

from __future__ import annotations

import argparse
import re
import statistics
import subprocess
from pathlib import Path

import chess
import chess.engine

ARCHON = "engines/bin/archon-d47v-pgo"
NET = "engines/canonical/net.cenn"
SF18 = "engines/bin/stockfish18"

AS = (-72.32565836, 185.93832038, -144.58862193, 416.44950446)
NNUE_RE = re.compile(r"NNUE evaluation\s+([+-]?[\d.]+)")
SCORE_RE = re.compile(r"score cp (-?\d+)")
BESTMOVE_RE = re.compile(r"^bestmove", re.M)


def drop_in_check(fens: list[str]) -> list[str]:
    """Remove positions where the side to move is in check.

    Stockfish's `eval` prints NO "NNUE evaluation" line for a position in
    check -- it prints only a "Final evaluation: none (in check)". Since the
    per-position values are recovered by INDEX from one batched session, a
    single in-check position silently shifts every SF value after it onto the
    wrong FEN. Measured on scale-check-800: 100 FENs, 7 in-check, 93 NNUE
    lines. That corrupted an earlier run of this study badly enough to give a
    7x wrong answer, and it corrupts any paired statistic computed the same
    way.

    Dropping them is also correct on the merits: a static eval in check is
    not a quantity either engine uses -- search generates evasions instead --
    so it has no bearing on how a margin should be scaled.
    """
    import chess
    return [f for f in fens if not chess.Board(f).is_check()]


def sf18_a(material: int) -> float:
    m = min(max(material, 17), 78) / 58.0
    return ((AS[0] * m + AS[1]) * m + AS[2]) * m + AS[3]


def material_of(fen: str) -> int:
    board = fen.split()[0]
    c = {k: board.count(k) for k in "PNBRQpnbrq"}
    return (
        (c["P"] + c["p"])
        + 3 * (c["N"] + c["n"] + c["B"] + c["b"])
        + 5 * (c["R"] + c["r"])
        + 9 * (c["Q"] + c["q"])
    )


def run(binary: str, script: str, extra: list[str] | None = None) -> str:
    r = subprocess.run(
        [binary] + (extra or []), input=script, text=True, capture_output=True
    )
    return r.stdout + r.stderr


def last_score(block: str) -> int | None:
    """The deepest `score cp` in one search's output.

    Mate scores have no `cp` field, so a mate-terminated search yields None
    and is dropped: its error is unbounded and would dominate any spread
    statistic. That is a real exclusion, not a convenience -- margins are not
    tuned against mate scores either.
    """
    m = SCORE_RE.findall(block)
    return int(m[-1]) if m else None


def search_scores(binary: str, fens: list[str], depth: int,
                  options: dict[str, str] | None = None) -> list[int | None]:
    """Deep search score per FEN, side-to-move relative, in the engine's own
    reported units.

    Driven through python-chess rather than a piped script: our engine treats
    `quit` as a stop, so feeding every command down one pipe cancels each
    search the moment it starts and yields a bestmove with no info lines at
    all (measured -- `go depth 8` from startpos returned `bestmove a2a3`).
    The UCI protocol needs a reader that waits for `bestmove`, which is what
    SimpleEngine does.

    A mate-terminated search returns None: its error is unbounded and would
    dominate any spread statistic, and margins are not tuned against mate
    scores either.
    """
    eng = chess.engine.SimpleEngine.popen_uci(binary)
    out: list[int | None] = []
    try:
        if options:
            for k, v in options.items():
                eng.configure({k: v})
        for f in fens:
            board = chess.Board(f)
            info = eng.analyse(board, chess.engine.Limit(depth=depth))
            sc = info["score"].relative
            out.append(sc.score() if sc.score() is not None else None)
    finally:
        eng.quit()
    return out


def archon_errors(fens: list[str], depth: int, net: str) -> list[float]:
    statics = run(ARCHON, "\n".join(fens) + "\n", ["nnue-eval", net]).splitlines()
    sv: list[int | None] = []
    for line in statics:
        line = line.strip()
        sv.append(int(line) if re.fullmatch(r"[+-]?\d+", line) else None)
    scores = search_scores(ARCHON, fens, depth, {"EvalFile": net})
    errs = []
    for i, f in enumerate(fens):
        if i >= len(sv) or sv[i] is None or i >= len(scores) or scores[i] is None:
            continue
        # Both raw internal, both side-to-move relative. No conversion.
        errs.append(abs(scores[i] - sv[i]))
    return errs


def sf_errors(fens: list[str], depth: int) -> list[float]:
    ev = run(SF18, "".join(f"position fen {f}\neval\n" for f in fens) + "quit\n")
    nnue = [float(x) for x in NNUE_RE.findall(ev)]
    # After drop_in_check there must be exactly one NNUE line per FEN. If not,
    # some other position class also suppresses the line and every value below
    # is misaligned -- fail loudly rather than report a plausible wrong number.
    assert len(nnue) == len(fens), (
        f"SF18 emitted {len(nnue)} NNUE lines for {len(fens)} FENs -- "
        "index-based pairing would be misaligned"
    )
    scores = search_scores(SF18, fens, depth)
    errs = []
    for i, f in enumerate(fens):
        if i >= len(nnue) or i >= len(scores) or scores[i] is None:
            continue
        a = sf18_a(material_of(f))
        # SF's `eval` prints NNUE in pawns from WHITE's point of view; its
        # `score cp` is side-to-move relative and normalised by the same
        # a(material). Recover internal v on both sides, then align to stm.
        static_int = nnue[i] * a
        if f.split()[1] == "b":
            static_int = -static_int
        search_int = scores[i] * a / 100.0
        errs.append(abs(search_int - static_int))
    return errs


def summarise(name: str, e: list[float]) -> tuple[float, float]:
    if len(e) < 10:
        print(f"{name}: too few points ({len(e)})")
        return float("nan"), float("nan")
    e = sorted(e)
    med = statistics.median(e)
    p90 = e[int(0.9 * (len(e) - 1))]
    print(
        f"{name:>10}  n={len(e):<4} median |err| {med:8.1f}   "
        f"p90 {p90:8.1f}   mean {statistics.fmean(e):8.1f}"
    )
    return med, p90


def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("fens", nargs="?", default="tools/nnue/calib/scale-check-800.fens")
    ap.add_argument("--depth", type=int, default=10)
    ap.add_argument("--limit", type=int, default=120)
    ap.add_argument("--net", default=NET)
    args = ap.parse_args()

    raw = [
        l.strip()
        for l in Path(args.fens).read_text().splitlines()
        if l.strip() and not l.startswith("#")
    ]

    def norm(line: str) -> str:
        parts = line.split()
        if len(parts) >= 6 and parts[4].lstrip("-").isdigit() and parts[5].isdigit():
            return " ".join(parts[:6])
        return " ".join(parts[:4] + ["0", "1"])

    fens = [norm(l) for l in raw]
    kept = drop_in_check(fens)
    dropped = len(fens) - len(kept)
    fens = kept[: args.limit or None]
    print(f"positions: {len(fens)}   depth {args.depth}   "
          f"(dropped {dropped} in-check)\n")

    a_err = archon_errors(fens, args.depth, args.net)
    s_err = sf_errors(fens, args.depth)
    a_med, _ = summarise("ARCHON", a_err)
    s_med, _ = summarise("SF18", s_err)

    if a_med == a_med and s_med == s_med and s_med > 1e-9:
        ratio = a_med / s_med
        print(f"\nerror ratio (ours/theirs, each in its OWN frame): {ratio:.3f}")
        print(
            "\ncompare against the eval-MAGNITUDE ratio 0.65:\n"
            "  ratio ~1.0  -> our absolute error matches SF's despite the\n"
            "                 compressed frame, so SF's raw margins are right\n"
            "                 for us and the frame ratio is a red herring.\n"
            "                 Retune margins AFTER the net improves, not before.\n"
            "  ratio ~0.65 -> error scales with the frame, the rescale argument\n"
            "                 was sound, and the measured negative needs a\n"
            "                 different explanation."
        )


if __name__ == "__main__":
    main()
