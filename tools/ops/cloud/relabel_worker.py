#!/usr/bin/env python3
"""Relabel a slice of a 32-byte bulletformat corpus with fixed-node SF search.

PURE STDLIB by requirement: the cloud payload ships the public python image
with no numpy (see build_payload.sh — vendoring compiled wheels is unsafe
there). The 5k-node searches dominate runtime, so pure-python decode is fine.

Record layout (archon_jax.spec.RECORD_DTYPE, 32 B):
    occ u64 LE (bytes 0-7) | pcs 16 B (bytes 8-23) | score i16 LE (BYTES
    24-25 — the first smoke wrote offset 12, mid-pcs, corrupting pieces and
    leaving scores untouched; the offset below is now load-bearing tested) |
    result u8 | ksq u8 | opp_ksq u8 | extra 3 B
Piece nibble: bit3 = colour (0 = side to move), bits 0-2 = P N B R Q K.
FEN reconstruction is STM-as-white with `w - - 0 1` — the same contract the
audit measured; rows where EITHER side still has K+R on home squares are
DROPPED (castling rights unknowable; ~2.8 % on fishpack32), never mislabeled.

DETERMINISM CONTRACT (retry safety): shard bytes must be reproducible because
st_put treats a byte mismatch on retry as fatal. Positions are assigned to
engines round-robin (i % CONC), each engine labels its list sequentially with
TT carry-over and no ucinewgame — deterministic for a fixed CONC, so CONC is
part of the campaign config and recorded in stats.json.

Output: input records verbatim with ONLY bytes 12-13 (score) replaced by the
new label — SF printed cp from the side to move, clamped to ±30000, mate
mapped to ±(30000 - plies). Result byte untouched (the WDL blend needs it).
"""
import json
import struct
import subprocess
import sys
import time

PT = "pnbrqk"


def decode(rec: bytes):
    """-> (fen or None-if-castlepossible, ) for one 32-B record."""
    occ = struct.unpack_from("<Q", rec, 0)[0]
    pcs = rec[8:24]
    board = [""] * 64
    idx = 0
    for sq in range(64):
        if not (occ >> sq) & 1:
            continue
        nib = (pcs[idx >> 1] >> ((idx & 1) * 4)) & 0xF
        idx += 1
        c, p = nib >> 3, nib & 7
        ch = PT[p]
        board[sq] = ch.upper() if c == 0 else ch
    # castle-possible filter: K and a R on home squares for either side
    def home(base, off):
        return (board[base + 4] == PT[5].upper() if off == 0 else board[base + 4] == PT[5]) and (
            (board[base] == PT[3].upper() if off == 0 else board[base] == PT[3])
            or (board[base + 7] == PT[3].upper() if off == 0 else board[base + 7] == PT[3]))
    if home(0, 0) or home(56, 1):
        return None
    ranks = []
    for r in range(7, -1, -1):
        run, out = 0, []
        for f in range(8):
            ch = board[r * 8 + f]
            if not ch:
                run += 1
            else:
                if run:
                    out.append(str(run)); run = 0
                out.append(ch)
        if run:
            out.append(str(run))
        ranks.append("".join(out))
    return "/".join(ranks) + " w - - 0 1"


def label_chunk(engine: str, fens, nodes: int, hash_mb: int):
    p = subprocess.Popen([engine], stdin=subprocess.PIPE, stdout=subprocess.PIPE,
                         text=True, bufsize=1)
    w, r = p.stdin, p.stdout
    w.write("uci\n"); w.flush()
    while not r.readline().startswith("uciok"):
        pass
    w.write(f"setoption name Hash value {hash_mb}\nisready\n"); w.flush()
    while not r.readline().startswith("readyok"):
        pass
    out = []
    for fen in fens:
        w.write(f"position fen {fen}\ngo nodes {nodes}\n"); w.flush()
        score = 0
        while True:
            ln = r.readline()
            if not ln or ln.startswith("bestmove"):
                break
            if " score cp " in ln:
                t = ln.split(); score = int(t[t.index("cp") + 1])
            elif " score mate " in ln:
                t = ln.split(); m = int(t[t.index("mate") + 1])
                score = (30000 - abs(m)) if m > 0 else -(30000 - abs(m))
        out.append(max(-30000, min(30000, score)))
    w.write("quit\n"); w.flush()
    p.wait(timeout=10)
    return out


def main(slice_path, out_path, stats_path, engine, nodes, conc):
    import concurrent.futures as cf
    t0 = time.time()
    raw = open(slice_path, "rb").read()
    n = len(raw) // 32
    fens, keep_idx, old = [], [], []
    dropped = 0
    for i in range(n):
        rec = raw[i * 32:(i + 1) * 32]
        fen = decode(rec)
        if fen is None:
            dropped += 1
            continue
        fens.append(fen); keep_idx.append(i)
        old.append(struct.unpack_from("<h", rec, 24)[0])
    # deterministic round-robin split
    chunks = [fens[c::conc] for c in range(conc)]
    with cf.ThreadPoolExecutor(max_workers=conc) as ex:
        results = list(ex.map(lambda ch: label_chunk(engine, ch, nodes, 16), chunks))
    new = [0] * len(fens)
    for c in range(conc):
        for j, s in enumerate(results[c]):
            new[j * conc + c] = s
    out = bytearray()
    for k, i in enumerate(keep_idx):
        rec = bytearray(raw[i * 32:(i + 1) * 32])
        struct.pack_into("<h", rec, 24, new[k])
        out += rec
    open(out_path, "wb").write(out)
    # pure-python pearson r as the shard-level sanity stat
    m = len(new)
    mo, mn = sum(old) / m, sum(new) / m
    co = sum((a - mo) * (b - mn) for a, b in zip(old, new))
    vo = sum((a - mo) ** 2 for a in old) ** 0.5
    vn = sum((b - mn) ** 2 for b in new) ** 0.5
    corr = co / (vo * vn) if vo and vn else 0.0
    stats = dict(records_in=n, dropped_castle=dropped, records_out=m,
                 corr_old_new=round(corr, 4), nodes=nodes, conc=conc,
                 seconds=round(time.time() - t0, 1),
                 pos_per_sec=round(m / max(time.time() - t0, 1), 1))
    open(stats_path, "w").write(json.dumps(stats) + "\n")
    print(json.dumps(stats))


if __name__ == "__main__":
    a = sys.argv
    main(a[1], a[2], a[3], a[4], int(a[5]), int(a[6]))
