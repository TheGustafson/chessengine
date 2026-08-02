#!/usr/bin/env python3
"""tools/ops/data/label_positions.py — stage 3: score positions with an oracle.

PURPOSE
  Attach a value label to every candidate position and apply the D46 quiet
  filter. THIS IS THE FILTER SITE. The contract (data-recipe-contract.md §1)
  says the filter runs "at the layer that still has raw scores + best moves" —
  that layer is here and nowhere later: the exported text has lost the move, so
  a downstream filter is structurally partial (the `canon_filter` lesson).

  It is also the CLAMP site, and the order is load-bearing: the |raw score| <=
  10000 criterion is tested on the RAW score, and only survivors are clamped to
  the +/-2499 the bulletformat converter requires. A clamp-first pipeline once
  hid 86,320 mate labels inside an aux corpus.

USAGE
  .venv/bin/python tools/ops/data/label_positions.py \
      positions.tsv --out labelled.tsv --oracle engines/bin/stockfish18 \
      --nodes 25000 --workers 6

  # null control: relabel the same input twice and diff (see --null below)
  .venv/bin/python tools/ops/data/label_positions.py positions.tsv \
      --out a.tsv --oracle ... --nodes 25000 --null

INPUT   stage-2 TSV (header line, `fen` first column) or a bare FEN per line.
OUTPUT  TSV, header line, one row per SURVIVING position:
    fen  label_cp  raw_cp  best  result  ply  plies_to_end  termination  game_id
  plus <out>.stats.json with per-criterion drop counts and throughput, and
  <out>.dropped.tsv (the rejects with their reason) so a filter change is
  auditable instead of invisible.

LABEL FRAME — stated, never guessed
  label_cp is the ORACLE'S OWN PRINTED cp, WHITE-relative, and the manifest
  records which engine produced it. Stockfish prints real cp; ARCHON prints raw
  internal values ~2.07x larger (uciprobe.ARCHON_PER_SF_PRINTED). Those are
  DIFFERENT UNITS and the trainer's SCORE_MULT is per-source (contract §2:
  1.42 for SF-UCI cp at WDL=0.3, 1.94 at WDL=0). This tool therefore refuses to
  mix oracles into one output file and writes the frame into the stats, so the
  MULT is derived from a recorded fact rather than remembered.

BOUND QUARANTINE (contract §6)
  A node-limited search's last info line is usually a fail-low/high BOUND, not a
  value; uciprobe drops bound lines, and we take the score at max(ladder), i.e.
  the deepest EXACT line. A position whose ladder is empty after that filtering
  has no exact reference and is dropped, not guessed. Recording the bound as a
  score is how a 100k-node probe once read "-40" where the truth was -228.

REPRODUCIBILITY
  `ucinewgame` before every position gives every label a cold TT, so a label is
  a function of (engine, net, nodes, position) alone and not of what happened to
  be searched before it. That costs a hash clear per position (~5% at 25k nodes
  with Hash=16) and buys exact reproducibility, which --null verifies.
"""
from __future__ import annotations

import argparse
import hashlib
import json
import multiprocessing as mp
import os
import sys
import time
from pathlib import Path

import chess

ROOT = Path(__file__).resolve().parents[3]
sys.path.insert(0, str(ROOT / "tools/forensics"))
import uciprobe  # noqa: E402

VALUE_SCORE_CAP = 2499      # bulletformat converter asserts |score| < 2500
RAW_MAGNITUDE_CAP = 10_000  # D46 criterion 3, tested on the RAW score
MIN_PLY = 16                # D46 criterion 1

_W = {}


def _init(oracle, net, nodes, hash_mb, threads):
    _W["eng"] = uciprobe.Engine(oracle, net=net, hash_mb=hash_mb,
                                options={"Threads": threads}, name="oracle")
    _W["nodes"] = nodes


def d46_verdict(board: chess.Board, best_uci: str, raw_cp: int, ply: int):
    """The five trainer-parity criteria, in the normative order.

    Mirrors sfbinpack_filter (tools/nnue/bullet/d40_features.rs) exactly,
    including the subtle pair at the end: criterion 4 rejects en passant via the
    move KIND, and criterion 5 tests DESTINATION-EMPTY. Those are not redundant
    — an en-passant capture lands on an EMPTY square, so destination-empty alone
    would let it through. (relabel_depth.py uses board.is_capture(), which
    catches ep by a different route; this copy follows the Rust.)
    """
    if ply < MIN_PLY:
        return "ply"
    if board.is_check():
        return "check"
    if abs(raw_cp) > RAW_MAGNITUDE_CAP:
        return "score"
    try:
        mv = chess.Move.from_uci(best_uci)
    except ValueError:
        return "badmove"
    if mv not in board.legal_moves:
        return "badmove"
    if mv.promotion is not None or board.is_castling(mv) or board.is_en_passant(mv):
        return "kind"
    if board.piece_at(mv.to_square) is not None:
        return "capture"
    return None


def _work(row):
    fen, ply = row[0], row[1]
    eng = _W["eng"]
    try:
        board = chess.Board(fen)
    except ValueError:
        return (row, None, None, "badfen")
    eng.newgame()
    try:
        r = eng.go(fen=fen, nodes=_W["nodes"])
    except RuntimeError:
        return (row, None, None, "enginedied")
    if not r["ladder"]:
        # every info line was a bound -> no exact reference (contract §6)
        return (row, None, None, "nobound")
    top = r["ladder"][max(r["ladder"])]
    raw_stm = top["score"]                       # side-to-move relative
    raw_white = raw_stm if board.turn == chess.WHITE else -raw_stm
    best = r["best"]
    reason = d46_verdict(board, best, raw_white, ply)
    return (row, raw_white, best, reason)


def read_rows(path, min_ply):
    with open(path) as fh:
        first = fh.readline()
        has_header = first.startswith("fen\t") or first.startswith("fen ")
        cols = first.rstrip("\n").split("\t") if has_header else None
        if not has_header:
            fh.seek(0)
        for line in fh:
            line = line.rstrip("\n")
            if not line:
                continue
            f = line.split("\t")
            if cols:
                d = dict(zip(cols, f))
                fen = d["fen"]
                ply = int(d.get("ply") or 0)
                yield (fen, ply, d.get("result", ""), d.get("plies_to_end", ""),
                       d.get("termination", ""), d.get("game_id", ""))
            else:
                fen = f[0]
                b = chess.Board(fen)
                ply = 2 * (b.fullmove_number - 1) + (0 if b.turn == chess.WHITE else 1)
                yield (fen, ply, "", "", "", "")


def sha(p):
    if not p or not os.path.exists(p):
        return None
    h = hashlib.sha256()
    with open(p, "rb") as f:
        for b in iter(lambda: f.read(1 << 20), b""):
            h.update(b)
    return h.hexdigest()


def main() -> int:
    ap = argparse.ArgumentParser()
    ap.add_argument("positions")
    ap.add_argument("--out", required=True)
    ap.add_argument("--oracle", required=True)
    ap.add_argument("--net", default=None,
                    help="EvalFile for an ARCHON oracle; omit for Stockfish")
    ap.add_argument("--nodes", type=int, required=True)
    ap.add_argument("--workers", type=int, default=4)
    ap.add_argument("--hash", type=int, default=16)
    ap.add_argument("--threads", type=int, default=1)
    ap.add_argument("--limit", type=int, default=0)
    ap.add_argument("--no-filter", action="store_true",
                    help="AUDIT ONLY. Emits every position with its verdict "
                         "recorded but not enforced. Never train on this.")
    ap.add_argument("--null", action="store_true",
                    help="null control: label twice, assert bit-identical")
    args = ap.parse_args()

    if args.no_filter:
        sys.stderr.write(
            "WARNING: --no-filter output is AUDIT ONLY and violates D46. "
            "Never bf-convert this file.\n")

    rows = list(read_rows(args.positions, MIN_PLY))
    if args.limit:
        rows = rows[:args.limit]
    if not rows:
        sys.stderr.write("no input rows\n")
        return 2

    def run_pass():
        t0 = time.time()
        with mp.Pool(args.workers, initializer=_init,
                     initargs=(args.oracle, args.net, args.nodes,
                               args.hash, args.threads)) as pool:
            res = pool.map(_work, rows, chunksize=16)
        return res, time.time() - t0

    res, elapsed = run_pass()

    if args.null:
        res2, _ = run_pass()
        a = [(r[1], r[2]) for r in res]
        b = [(r[1], r[2]) for r in res2]
        diff = sum(1 for x, y in zip(a, b) if x != y)
        sys.stderr.write(
            f"NULL CONTROL: {len(a)} positions relabelled identically="
            f"{len(a) - diff} differing={diff} "
            f"({100.0 * diff / max(len(a), 1):.3f}%)\n")
        if diff:
            sys.stderr.write(
                "  ^ non-zero means labels are NOT a pure function of "
                "(engine,net,nodes,position); any node-count study on this "
                "instrument is confounded by that much run-to-run noise.\n")

    drops: dict[str, int] = {}
    kept = 0
    out = open(args.out, "w")
    out.write("fen\tlabel_cp\traw_cp\tbest\tresult\tply\tplies_to_end\t"
              "termination\tgame_id\n")
    dropf = open(args.out + ".dropped.tsv", "w")
    dropf.write("fen\traw_cp\tbest\treason\n")

    for row, raw, best, reason in res:
        fen, ply, result, pte, term, gid = row
        if reason is not None:
            drops[reason] = drops.get(reason, 0) + 1
            dropf.write(f"{fen}\t{'' if raw is None else raw}\t{best or ''}\t{reason}\n")
            if not args.no_filter:
                continue
        # CLAMP LAST — only after the raw magnitude criterion has been applied.
        label = max(-VALUE_SCORE_CAP, min(VALUE_SCORE_CAP, raw)) if raw is not None else ""
        out.write(f"{fen}\t{label}\t{'' if raw is None else raw}\t{best or ''}\t"
                  f"{result}\t{ply}\t{pte}\t{term}\t{gid}\n")
        kept += 1
    out.close()
    dropf.close()

    n = len(rows)
    frame = "archon-internal" if args.net else "sf-printed-cp"
    stats = dict(
        positions_in=n, kept=kept, dropped=n - kept,
        drop_rate_pct=round(100.0 * (n - kept) / max(n, 1), 2),
        drops_by_reason=drops,
        oracle=dict(path=args.oracle, sha256=sha(args.oracle),
                    net=args.net, net_sha256=sha(args.net) if args.net else None,
                    nodes=args.nodes, hash_mb=args.hash, threads=args.threads,
                    label_frame=frame),
        throughput=dict(elapsed_s=round(elapsed, 1), workers=args.workers,
                        pos_per_s=round(n / max(elapsed, 1e-9), 2),
                        pos_per_hour=round(3600 * n / max(elapsed, 1e-9))),
        filter="D46 five-criterion (enforced)" if not args.no_filter
               else "D46 recorded but NOT enforced (AUDIT ONLY)",
    )
    Path(args.out + ".stats.json").write_text(json.dumps(stats, indent=2))
    sys.stderr.write(json.dumps(stats, indent=2) + "\n")

    # Self-check (contract §5a): a drop rate near 0% means the filter did not
    # run; near 100% means a field is broken. Warn loudly either way.
    if not args.no_filter:
        r = stats["drop_rate_pct"]
        if r < 1.0:
            sys.stderr.write(f"WARN: drop rate {r}% ~ 0 — did the filter run?\n")
        elif r > 90.0:
            sys.stderr.write(f"WARN: drop rate {r}% ~ 100 — is a field broken?\n")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
