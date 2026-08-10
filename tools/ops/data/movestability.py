#!/usr/bin/env python3
"""Best-move INSTABILITY per position: how often the PV move changes with depth.

WHY THIS TARGET, AFTER FOUR NULLS
    The disagreement head was trained on `search - static` in CENTIPAWNS and
    tested through four search consumers (RFP margin, eval correction naive and
    orthogonalised, LMR reduction) over ~10,500 games. Offline the signal is
    large and well controlled (r^2 = 0.611 held out, beyond a flexible function
    of the static eval). In games it produced nothing.

    The suspicion the ledger ends on is a UNITS problem: centipawns are not
    something search consumes directly. Added to the eval they break the
    calibration every margin is tuned against; scaling a margin by them is
    indirect and clamped.

    Best-move instability is different in kind. It is dimensionless, it is a
    property of the position rather than of the eval's frame, and it is what
    actually costs Elo — this project's standing bar is NO ACCEPTABLE BLUNDERS,
    and a blunder is a wrong MOVE. A wrong score that picks the right move
    costs nothing.

WHAT IT MEASURES
    One `go nodes N` per position, parsing `info depth D ... pv <move> ...`.
    Emits per position:
        changes   how many times the depth-1..D best move differed from the
                  previous depth (0 = the first move found is the final one)
        depths    how many depths reported (normaliser; a shallow search has
                  fewer chances to change its mind)
        final     the final best move
    The trainable target is `changes / max(depths-1, 1)` — a rate in [0,1], so
    it does not smuggle in "how deep did this search get", which is mostly a
    function of position complexity AND node budget.

HONEST LIMITATION
    Instability at 4000 nodes is a proxy for instability at the depths a real
    search reaches, and the two need not agree. It is the same proxy caveat the
    cp target had; what changes is the units, not the proxy status.
"""
import subprocess
import sys
import pathlib


def run(engine: str, net: str, fens: list[str], nodes: int = 4000):
    p = subprocess.Popen([engine], stdin=subprocess.PIPE, stdout=subprocess.PIPE,
                         text=True, bufsize=1)
    w, r = p.stdin, p.stdout
    w.write("uci\n"); w.flush()
    while True:
        line = r.readline()
        if not line or line.startswith("uciok"):
            break
    w.write(f"setoption name EvalFile value {net}\n")
    w.write("setoption name Hash value 16\nisready\n"); w.flush()
    while True:
        line = r.readline()
        if not line or line.startswith("readyok"):
            break

    out = []
    for fen in fens:
        w.write(f"position fen {fen}\ngo nodes {nodes}\n"); w.flush()
        best_by_depth, last_depth = [], -1
        while True:
            line = r.readline()
            if not line:
                break
            if line.startswith("bestmove"):
                break
            if line.startswith("info depth") and " pv " in line:
                t = line.split()
                try:
                    d = int(t[t.index("depth") + 1])
                    mv = t[t.index("pv") + 1]
                except (ValueError, IndexError):
                    continue
                # One entry per DEPTH: multipv/aspiration re-searches emit
                # several lines per depth and only the last is that depth's
                # answer. Counting them all would measure re-search churn,
                # not depth-to-depth mind changes.
                if d == last_depth and best_by_depth:
                    best_by_depth[-1] = mv
                else:
                    best_by_depth.append(mv)
                    last_depth = d
        ch = sum(1 for a, b in zip(best_by_depth, best_by_depth[1:]) if a != b)
        nd = len(best_by_depth)
        out.append((ch, nd, best_by_depth[-1] if best_by_depth else "-"))
    w.write("quit\n"); w.flush()
    p.wait(timeout=10)
    return out


def main(engine: str, net: str, fen_file: str, out_file: str,
         nodes: int = 4000) -> int:
    fens = [ln.strip() for ln in pathlib.Path(fen_file).read_text().splitlines()
            if ln.strip()]
    res = run(engine, net, fens, nodes)
    with open(out_file, "w") as f:
        for ch, nd, mv in res:
            f.write(f"{ch}\t{nd}\t{mv}\n")
    ch = [c for c, _, _ in res]
    nd = [d for _, d, _ in res]
    rate = [c / max(d - 1, 1) for c, d, _ in res]
    print(f"positions {len(res):,}")
    print(f"  changes: mean {sum(ch)/len(ch):.2f}  zero {100*sum(1 for c in ch if c==0)/len(ch):.1f}%")
    print(f"  depths : mean {sum(nd)/len(nd):.2f}  min {min(nd)} max {max(nd)}")
    print(f"  rate   : mean {sum(rate)/len(rate):.4f}")
    return 0


if __name__ == "__main__":
    if len(sys.argv) < 5:
        print(__doc__)
        raise SystemExit(2)
    raise SystemExit(main(sys.argv[1], sys.argv[2], sys.argv[3], sys.argv[4],
                          int(sys.argv[5]) if len(sys.argv) > 5 else 4000))
