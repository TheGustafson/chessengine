"""Representation-gap study: oracle twins for the invisible state dimensions.

For a state dimension the network cannot see (L2+L3: castling rights,
rule50), build PAIRS of positions that are IDENTICAL to the network and
differ only in that dimension, and measure the true value difference with a
fixed-node oracle (SF eca43a97 — the project's pinned constants reference).
Discrete outcome first (bestmove changes), cp distribution second.

Twin constructions:
  castle   full-state FEN with rights (own side, K+R home) vs rights stripped
  rule50   same FEN at halfmove 0 vs halfmove 90 (5 plies from the draw)

Also the ARCHON blindness receipts:
  archon-static   `chessengine nnue-eval` on both twins — castling twins MUST
                  be bit-identical (features carry no rights); rule50 twins
                  MUST be bit-identical at the NET level (rule50 enters only
                  the outer damping, which nnue-eval's forward pass excludes
                  if it prints raw net output — whichever way it prints, the
                  receipt documents where the sensitivity lives).

Run (SF ~200 MB, single thread — light enough beside one probe pass):
  .venv/bin/python tools/nnue/reprgap/twins.py castle 1200
  .venv/bin/python tools/nnue/reprgap/twins.py rule50 1200
"""
from __future__ import annotations

import json
import subprocess
import sys
from pathlib import Path

import chess

ROOT = Path(__file__).resolve().parents[3]
sys.path.insert(0, str(ROOT / "tools/forensics"))
import uciprobe  # noqa: E402

DIR = ROOT / "data/nnue/nets/screens/reprgap"
TSV = DIR / "tsv-filt-450k.tsv"
SF = "/home/nick/sf-eca43a97/src/stockfish"
NODES = 60_000


def sf_score(eng, fen: str) -> tuple[int, str]:
    r = eng.go(fen=fen, nodes=NODES)
    d = max(r["ladder"])
    return r["ladder"][d]["score"], r["best"]


def pick_positions(kind: str, n: int) -> list[str]:
    """FENs from the full-state sample. castle: stm still owns a home-shape
    right. rule50: quiet-ish, no rights (isolate the clock), |material| any."""
    out = []
    with open(TSV) as fh:
        for line in fh:
            fen = line.split("\t")[0]
            b = chess.Board(fen)
            if kind == "castle":
                if b.has_castling_rights(b.turn):
                    out.append(fen)
            elif kind == "rule50":
                if b.castling_rights == 0 and b.ep_square is None:
                    out.append(fen)
            if len(out) >= n:
                break
    return out


def twin(kind: str, fen: str) -> str:
    b = chess.Board(fen)
    if kind == "castle":
        b.castling_rights = 0
        return b.fen()
    parts = fen.split()
    parts[4] = "90" if kind == "rule50" else parts[4]
    return " ".join(parts)


def base_variant(kind: str, fen: str) -> str:
    if kind != "rule50":
        return fen
    parts = fen.split()
    parts[4] = "0"
    return " ".join(parts)


def archon_static(fens: list[str]) -> list[float]:
    p = subprocess.run(
        [str(ROOT / "engines/canonical/archon"), "nnue-eval",
         str(ROOT / "engines/canonical/net.cenn")],
        input="\n".join(fens) + "\n", capture_output=True, text=True,
        timeout=600)
    vals = []
    for line in p.stdout.strip().splitlines():
        try:
            vals.append(float(line.split()[0]))
        except (ValueError, IndexError):
            pass
    return vals


def main() -> int:
    kind = sys.argv[1]
    n = int(sys.argv[2]) if len(sys.argv) > 2 else 1200
    fens = pick_positions(kind, n)
    print(f"[{kind}] {len(fens)} base positions")
    bases = [base_variant(kind, f) for f in fens]
    twins = [twin(kind, f) for f in fens]

    # ARCHON static receipts first (cheap)
    a_base = archon_static(bases)
    a_twin = archon_static(twins)
    ident = sum(1 for x, y in zip(a_base, a_twin) if x == y)
    print(f"[{kind}] archon nnue-eval identical twins: {ident}/{len(a_base)}")

    eng = uciprobe.Engine(SF, net=None, hash_mb=64, name="sf")
    rows = []
    for i, (fb, ft) in enumerate(zip(bases, twins)):
        sb, mb = sf_score(eng, fb)
        st, mt = sf_score(eng, ft)
        rows.append({"fen": fb, "d": sb - st, "s_base": sb, "s_twin": st,
                     "move_changed": int(mb != mt)})
        if (i + 1) % 100 == 0:
            print(f"  {i + 1}/{len(bases)}", flush=True)
    eng.close()
    ds = [r["d"] for r in rows]
    mc = sum(r["move_changed"] for r in rows)
    ds_a = sorted(abs(d) for d in ds)
    q = lambda p: ds_a[int(p * (len(ds_a) - 1))]
    print(f"[{kind}] SF@{NODES}n twins n={len(rows)}: "
          f"mean d {sum(ds)/len(ds):+.1f} cp, |d| p50 {q(.5)} p90 {q(.9)} "
          f"p99 {q(.99)}, bestmove changed {mc}/{len(rows)} "
          f"({100*mc/len(rows):.1f}%)")
    (DIR / f"twins-{kind}.json").write_text(json.dumps(
        {"kind": kind, "nodes": NODES, "rows": rows,
         "archon_static_identical": ident, "n_archon": len(a_base)}, indent=1))
    return 0


if __name__ == "__main__":
    sys.exit(main())
