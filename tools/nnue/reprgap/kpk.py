"""Representation-gap study: KPK exact game-theoretic result as a probe target.

Retrograde-solves KPK (white K+P vs black K) exactly — promotion evaluated
through the resulting KQK (queen-capture / mate / stalemate handled), pawn
mates and stalemates handled — then cross-validates against SF eca43a97 at
fixed nodes on a random sample, encodes all legal white-to-move states as
canonical records, and (after acc extraction) probes the frozen accumulator
for the exact result with the study's capacity ladder.

The exact result of a 3-man position is the archetype L1 target: a
deterministic placement function whose computation is a minimax fixpoint —
nothing a linear map + 640 products can iterate. Controls: pawn rank (must
decode — it is literally an input atom), occ-hash bit (must not).

Distribution note (honest): KPK states are off the training mix's
distribution; the probe measures what the frozen representation encodes
there. The engine must nevertheless evaluate such positions in play.

  .venv/bin/python tools/nnue/reprgap/kpk.py solve      # bitbase + SF check
  (extract.py recs kpk-recs.npz)                        # acc extraction
  .venv/bin/python tools/nnue/reprgap/kpk.py probe      # ridge ladder
"""
from __future__ import annotations

import json
import sys
from pathlib import Path

import numpy as np

ROOT = Path(__file__).resolve().parents[3]
DIR = ROOT / "data/nnue/nets/screens/reprgap"
sys.path.insert(0, str(ROOT / "tools/nnue/reprgap"))
sys.path.insert(0, str(ROOT / "jax"))

WIN, DRAW, UNK = 2, 1, 0

KATT = [0] * 64
for s in range(64):
    f, r = s & 7, s >> 3
    m = 0
    for df in (-1, 0, 1):
        for dr in (-1, 0, 1):
            if df == dr == 0:
                continue
            f2, r2 = f + df, r + dr
            if 0 <= f2 <= 7 and 0 <= r2 <= 7:
                m |= 1 << (r2 * 8 + f2)
    KATT[s] = m


def patt(p: int) -> int:
    f, r = p & 7, p >> 3
    m = 0
    for df in (-1, 1):
        if 0 <= f + df <= 7 and r + 1 <= 7:
            m |= 1 << ((r + 1) * 8 + f + df)
    return m


def qatt(q: int, occ: int) -> int:
    m = 0
    for df, dr in ((1, 0), (-1, 0), (0, 1), (0, -1), (1, 1), (1, -1),
                   (-1, 1), (-1, -1)):
        f, r = (q & 7) + df, (q >> 3) + dr
        while 0 <= f <= 7 and 0 <= r <= 7:
            s = r * 8 + f
            m |= 1 << s
            if occ & (1 << s):
                break
            f, r = f + df, r + dr
    return m


def promo_result(wk: int, bk: int, q: int) -> int:
    """KQK, black to move, queen just appeared on q (rank 8)."""
    if KATT[bk] & (1 << q) and not (KATT[wk] & (1 << q)):
        return DRAW  # queen falls, KK
    occ = (1 << wk) | (1 << bk) | (1 << q)
    qa = qatt(q, occ ^ (1 << bk))  # queen sees through nothing; bk removed so
    # squares behind bk count as attacked (standard king-move legality)
    moves = KATT[bk] & ~KATT[wk] & ~qa & ~(1 << wk) & ~(1 << q)
    # capture the queen if defended was excluded above; undefended handled
    if moves:
        return WIN  # KQvK with a safe queen is won
    return WIN if qa & (1 << bk) else DRAW  # mate : stalemate


def idx(wk: int, bk: int, p: int) -> int:
    return (wk * 64 + bk) * 48 + (p - 8)


def solve() -> tuple[np.ndarray, np.ndarray]:
    """Returns (res_w, res_b): result arrays for wtm/btm over idx space."""
    N = 64 * 64 * 48
    res_w = np.zeros(N, np.int8)
    res_b = np.zeros(N, np.int8)
    legal_w = np.zeros(N, bool)
    legal_b = np.zeros(N, bool)
    for wk in range(64):
        for bk in range(64):
            if wk == bk or KATT[wk] & (1 << bk):
                continue
            for p in range(8, 56):
                if p == wk or p == bk:
                    continue
                i = idx(wk, bk, p)
                legal_w[i] = not (patt(p) & (1 << bk))  # btm king capturable
                legal_b[i] = True

    def black_moves(wk, bk, p):
        out = []
        cap_ok = KATT[bk] & (1 << p) and not (KATT[wk] & (1 << p))
        for t in bits(KATT[bk] & ~KATT[wk] & ~(1 << wk)):
            if t == p:
                continue  # capture handled separately
            if patt(p) & (1 << t):
                continue
            out.append(t)
        return out, bool(cap_ok)

    def bits(m):
        while m:
            yield (m & -m).bit_length() - 1
            m &= m - 1

    changed = True
    sweeps = 0
    while changed:
        changed = False
        sweeps += 1
        for wk in range(64):
            for bk in range(64):
                base = (wk * 64 + bk) * 48
                for p in range(8, 56):
                    i = base + (p - 8)
                    # --- btm ---
                    if legal_b[i] and res_b[i] == UNK:
                        moves, cap = black_moves(wk, bk, p)
                        if cap:
                            res_b[i] = DRAW
                        elif not moves:
                            res_b[i] = WIN if patt(p) & (1 << bk) else DRAW
                        else:
                            vals = [res_w[idx(wk, t, p)] for t in moves]
                            if any(v == DRAW for v in vals):
                                res_b[i] = DRAW
                            elif all(v == WIN for v in vals):
                                res_b[i] = WIN
                        if res_b[i] != UNK:
                            changed = True
                    # --- wtm ---
                    if legal_w[i] and res_w[i] == UNK:
                        vals = []
                        for t in bits(KATT[wk] & ~KATT[bk]):
                            if t == p or t == bk:
                                continue
                            vals.append(res_b[idx(t, bk, p)])
                        pushes = []
                        t = p + 8
                        if t < 64 and t != wk and t != bk:
                            pushes.append(t)
                            t2 = p + 16
                            if p >> 3 == 1 and t2 != wk and t2 != bk:
                                pushes.append(t2)
                        for t in pushes:
                            if t >= 56:
                                vals.append(promo_result(wk, bk, t))
                            else:
                                j = idx(wk, bk, t)
                                vals.append(res_b[j] if not (patt(t) & (1 << bk))
                                            or True else UNK)
                                # pawn push giving check is fine btm-side
                        if not vals:
                            res_w[i] = DRAW  # white stalemated (rare)
                        elif any(v == WIN for v in vals):
                            res_w[i] = WIN
                        elif all(v == DRAW for v in vals):
                            res_w[i] = DRAW
                        if res_w[i] != UNK:
                            changed = True
    # closure: remaining UNK cells are draw-by-repetition cycles
    res_w[legal_w & (res_w == UNK)] = DRAW
    res_b[legal_b & (res_b == UNK)] = DRAW
    print(f"solved in {sweeps} sweeps; wtm legal {legal_w.sum()}, "
          f"win {(res_w[legal_w] == WIN).sum()} "
          f"({(res_w[legal_w] == WIN).mean():.3f})")
    np.savez(DIR / "kpk-bitbase.npz", res_w=res_w, res_b=res_b,
             legal_w=legal_w, legal_b=legal_b)
    return res_w, legal_w


def fen_of(wk: int, bk: int, p: int) -> str:
    board = [""] * 64
    board[wk], board[bk], board[p] = "K", "k", "P"
    rows = []
    for r in range(7, -1, -1):
        row, run = "", 0
        for f in range(8):
            c = board[r * 8 + f]
            if c:
                row += (str(run) if run else "") + c
                run = 0
            else:
                run += 1
        rows.append(row + (str(run) if run else ""))
    return "/".join(rows) + " w - - 0 1"


def sf_check(n: int = 300) -> None:
    sys.path.insert(0, str(ROOT / "tools/forensics"))
    import uciprobe
    z = np.load(DIR / "kpk-bitbase.npz")
    res_w, legal_w = z["res_w"], z["legal_w"]
    rng = np.random.default_rng(5)
    cand = np.nonzero(legal_w)[0]
    pick = rng.choice(cand, n, replace=False)
    eng = uciprobe.Engine("/home/nick/sf-eca43a97/src/stockfish", net=None,
                          hash_mb=64, name="sf")
    agree = dis = skip = 0
    for i in pick:
        wk, bk, p = i // (64 * 48), (i // 48) % 64, (i % 48) + 8
        r = eng.go(fen=fen_of(wk, bk, p), nodes=60_000)
        d = max(r["ladder"])
        sc = r["ladder"][d]["score"]
        mine = res_w[i]
        if abs(sc) < 40:
            sfv = DRAW
        elif sc > 250:
            sfv = WIN
        else:
            skip += 1
            continue
        if sfv == mine:
            agree += 1
        else:
            dis += 1
            if dis <= 5:
                print(f"  DISAGREE {fen_of(wk, bk, p)} mine={mine} sf={sc}")
    eng.close()
    print(f"SF cross-check: {agree} agree, {dis} disagree, {skip} ambiguous "
          f"of {n}")


def make_recs() -> None:
    sys.path.insert(0, str(ROOT / "tools/nnue/reprgap"))
    import chess
    from encode import board_to_record
    z = np.load(DIR / "kpk-bitbase.npz")
    res_w, legal_w = z["res_w"], z["legal_w"]
    from archon_jax.spec import RECORD_DTYPE
    ids = np.nonzero(legal_w)[0]
    recs = np.zeros(len(ids), dtype=RECORD_DTYPE)
    win = np.zeros(len(ids), np.uint8)
    for k, i in enumerate(ids):
        wk, bk, p = i // (64 * 48), (i // 48) % 64, (i % 48) + 8
        b = chess.Board(fen_of(wk, bk, p))
        recs[k] = board_to_record(b)[0][0]
        recs[k]["result"] = 2 if res_w[i] == WIN else 1
        win[k] = res_w[i] == WIN
    np.savez(DIR / "recs-kpk.npz", recs=recs.view(np.uint8).reshape(-1, 32),
             kpk_win=win)
    print(f"encoded {len(ids)} KPK wtm states -> recs-kpk.npz "
          f"(win rate {win.mean():.3f})")


def probe() -> None:
    from probes import _feat, make_rf
    files = sorted(DIR.glob("acc-recs-kpk-*.npz"))
    accs, accn = [], []
    for f in files:
        z = np.load(f)
        accs.append(z["acc_stm"]); accn.append(z["acc_ntm"])
    acs = np.concatenate(accs); acn = np.concatenate(accn)
    y = np.load(DIR / "recs-kpk.npz")["kpk_win"].astype(np.float64)
    z = np.load(DIR / "kpk-bitbase.npz")
    ids = np.nonzero(z["legal_w"])[0]
    # memory envelope (<=3 GB): subsample 60k of the 163k states
    sub = np.random.default_rng(3).choice(len(y), 60_000, replace=False)
    acs, acn, y, ids = acs[sub], acn[sub], y[sub], ids[sub]
    prank = ((ids % 48) + 8) // 8  # control target: pawn rank
    occb = np.zeros(len(ids))     # hash-bit control from state index
    occb = ((ids.astype(np.uint64) * np.uint64(0x9E3779B97F4A7C15))
            >> np.uint64(63)).astype(np.float64)
    n = len(y)
    rng = np.random.default_rng(9)
    perm = rng.permutation(n)
    tr, ev = perm[: int(n * 0.7)], perm[int(n * 0.7):]
    acc, hh = _feat(acs, acn)
    ma, sa = acc[tr].mean(0), acc[tr].std(0) + 1e-6
    mh, sh = hh[tr].mean(0), hh[tr].std(0) + 1e-6
    za = (acc - ma) / sa
    zh = (hh - mh) / sh
    R, rb = make_rf(2560 + 1280)
    rf = np.maximum(np.concatenate([za, zh], 1) @ R + rb, 0)

    def ridge_auc(X, yy, lam=1e-2):
        Xt = np.concatenate([X, np.ones((len(X), 1), np.float32)], 1)
        A = Xt[tr].T.astype(np.float64) @ Xt[tr]
        d = len(A)
        w = np.linalg.solve(A + lam * np.eye(d) * np.trace(A) / d,
                            Xt[tr].T.astype(np.float64) @ yy[tr])
        s = Xt[ev] @ w
        r = np.argsort(np.argsort(s)) + 1.0
        pos = yy[ev] > 0.5
        return float((r[pos].sum() - pos.sum() * (pos.sum() + 1) / 2)
                     / max(pos.sum() * (~pos).sum(), 1))

    out = {}
    for tname, yy in (("kpk_win", y), ("pawn_rank_ge5", (prank >= 5).astype(float)),
                      ("hashbit", occb),
                      ("kpk_win_SHUF", y[rng.permutation(n)])):
        out[tname] = {
            "P1_acc": round(ridge_auc(za, yy), 4),
            "P2_acch": round(ridge_auc(np.concatenate([za, zh], 1), yy), 4),
            "P3_rf": round(ridge_auc(rf.astype(np.float32), yy), 4),
        }
        print(tname, out[tname])
    (DIR / "kpk-probe.json").write_text(json.dumps(out, indent=1))


if __name__ == "__main__":
    cmd = sys.argv[1]
    if cmd == "solve":
        solve()
        sf_check()
        make_recs()
    elif cmd == "probe":
        probe()
