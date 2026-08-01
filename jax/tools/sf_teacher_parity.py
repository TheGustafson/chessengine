"""GATE 1 for the SF-net teacher: does the converted checkpoint reproduce
Stockfish?

Three quantities on identical positions:

  A. ``archon-sfswapbin sf-eval`` -- THE ORACLE. Stockfish's raw internal NNUE
     units, verified 36/36 against `stockfish-eca43a97`'s own `eval`.
  B. ``sf_to_archon_warm.evaluate_int`` -- an exact numpy transcription of
     `sf_swap::evaluate_raw`, fed ARCHON's OWN sparse feature indices. B == A
     to the integer proves the FILE LAYOUT and the FEATURE MAPPING; there is
     no float slack for an error to hide in.
  C. ``archon_jax.model.forward`` on the converted checkpoint -- the actual
     teacher. C - A is the float-vs-fixed-point gap and is REPORTED, not
     assumed: it is the only error the trainer will ever see.

Run (from the `jax/` directory -- the repo root has a `jax/` dir that shadows
the real package):

  cd jax && JAX_PLATFORMS=cpu XLA_PYTHON_CLIENT_PREALLOCATE=false \
    uv run --no-project --with numpy==2.5.1 --with jax==0.11.0 \
       --with optax==0.2.8 -- python tools/sf_teacher_parity.py \
       ../data/nnue/teachers/sf-eca43a97-teacher-raw.bin \
       --fens ../tools/nnue/calib/wl50.fens \
       --corpus ../data/nnue/jaxcorpora/uho-band75-100.shuf.bin --n 2000
"""

from __future__ import annotations

import argparse
import importlib.util
import os
import subprocess
import sys
from pathlib import Path

import numpy as np

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))
os.environ.setdefault("JAX_PLATFORMS", "cpu")
os.environ.setdefault("XLA_PYTHON_CLIENT_PREALLOCATE", "false")

ROOT = Path(__file__).resolve().parents[2]
SFSWAP = str(ROOT / "engines/bin/archon-sfswapbin")
SFNET = "/home/nick/sf-eca43a97/src/nn-0ee0657fb25e.nnue"

PIECE_CODE = {"P": 0, "N": 1, "B": 2, "R": 3, "Q": 4, "K": 5}
PIECE_CHARS = "PNBRQK"


def _load(name: str, path: Path):
    spec = importlib.util.spec_from_file_location(name, path)
    mod = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(mod)
    return mod


# `tools/nnue/` is not a package and the repo root cannot go on sys.path here
# (its `jax/` directory would shadow the real jax), so load by file path.
sfconv = _load("sf_to_archon_warm", ROOT / "tools/nnue/sf_to_archon_warm.py")


# ---------------------------------------------------------------------------
# positions
# ---------------------------------------------------------------------------

def canonicalize(fen: str) -> str:
    """Any FEN -> the stm-canonical white-to-move placement the corpus stores
    (mover becomes White on a vertically mirrored board). Castling rights and
    the ep square are dropped, which is exactly what bulletformat does and is
    harmless: HalfKAv2_hm + FullThreats are blind to both (800/800 identical
    twins, 2026-07-30)."""
    placement, stm = fen.split()[:2]
    if stm == "w":
        return placement + " w - - 0 1"
    rows = placement.split("/")[::-1]
    return "/".join(r.swapcase() for r in rows) + " w - - 0 1"


def fen_to_record(fen: str, dtype) -> np.ndarray:
    """White-to-move FEN -> a bulletformat record (score/result unused here)."""
    placement = fen.split()[0]
    rec = np.zeros(1, dtype=dtype)
    pieces = []
    for r, row in enumerate(placement.split("/")):
        rank = 7 - r
        f = 0
        for ch in row:
            if ch.isdigit():
                f += int(ch)
                continue
            pieces.append((rank * 8 + f, 0 if ch.isupper() else 1,
                           PIECE_CODE[ch.upper()]))
            f += 1
    occ = 0
    for i, (sq, color, pt) in enumerate(sorted(pieces)):
        occ |= 1 << sq
        rec["pcs"][0][i // 2] |= ((color << 3) | pt) << (4 * (i % 2))
        if color == 0 and pt == 5:
            rec["ksq"] = sq
        if color == 1 and pt == 5:
            rec["opp_ksq"] = sq
    rec["occ"] = occ
    rec["extra"][0] = (64, 64, 0)
    rec["result"] = 1
    return rec


def sf_eval(fens: list[str], net: str) -> np.ndarray:
    r = subprocess.run([SFSWAP, "sf-eval", net], input="\n".join(fens) + "\n",
                       capture_output=True, text=True, timeout=7200)
    lines = r.stdout.splitlines()
    if len(lines) != len(fens):
        raise RuntimeError(f"sf-eval gave {len(lines)} lines for {len(fens)} "
                           f"fens; stderr: {r.stderr[:400]}")
    return np.array([int(x.strip()) for x in lines], dtype=np.int64)


def records_to_fens(recs: np.ndarray) -> list[str]:
    """Inverse of fen_to_record over a batch (the stored frame IS the board)."""
    out = []
    for rec in recs:
        occ, pcs, board = int(rec["occ"]), rec["pcs"], [0] * 64
        k = 0
        while occ:
            sq = (occ & -occ).bit_length() - 1
            occ &= occ - 1
            code = (int(pcs[k >> 1]) >> (4 * (k & 1))) & 0xF
            board[sq] = 0x40 | code
            k += 1
        rows = []
        for rank in range(7, -1, -1):
            row, empty = [], 0
            for f in range(8):
                v = board[rank * 8 + f]
                if not v:
                    empty += 1
                    continue
                if empty:
                    row.append(str(empty))
                    empty = 0
                ch = PIECE_CHARS[v & 7]
                row.append(ch.lower() if v & 8 else ch)
            if empty:
                row.append(str(empty))
            rows.append("".join(row))
        out.append("/".join(rows) + " w - - 0 1")
    return out


# ---------------------------------------------------------------------------

def report(name: str, a: np.ndarray, b: np.ndarray) -> float:
    """|a - b| summary PLUS the OLS of a on b.

    The absolute gap alone cannot decide anything here: what the trainer
    actually consumes is the teacher after a fitted (k, p) that was regressed
    against the ORACLE. A gap that is pure zero-mean noise passes straight
    through that fit; a gap with slope != 1 or a nonzero intercept does not.
    So slope/intercept/residual-sd are the load-bearing columns."""
    d = np.abs(a - b)
    slope, icept = np.polyfit(b.astype(float), a.astype(float), 1)
    resid = a - (slope * b + icept)
    print(f"  {name:<28} n={len(d)}  exact={int((d == 0).sum())}/{len(d)}  "
          f"max={d.max():.2f}  median={np.median(d):.2f}\n"
          f"  {'':<28} slope={slope:.6f}  intercept={icept:+.3f}  "
          f"resid_sd={resid.std():.2f}  bias(mean a-b)={np.mean(a - b):+.3f}")
    return float(d.max())


def run_panel(label: str, recs: np.ndarray, fens: list[str], warm: str,
              net_path: str, chunk: int) -> int:
    import jax.numpy as jnp
    from archon_jax import checkpoint as ck
    from archon_jax.features import map_batch
    from archon_jax.model import forward

    arch = sfconv.archon_arch()
    params = {k: jnp.asarray(v) for k, v in ck.load_warm(warm, arch).items()}
    sfnet = sfconv.SFNet(net_path)

    b = map_batch(recs, wdl=0.0, score_multiplier=1.0, scale=400.0,
                  policy=False, threats=True, sf_bucket=True)
    t_bucket = np.asarray(b["t_bucket"])

    oracle = sf_eval(fens, net_path)
    ints = sfconv.evaluate_int(sfnet, np.asarray(b["hk_stm"]),
                               np.asarray(b["hk_ntm"]), np.asarray(b["th_stm"]),
                               np.asarray(b["th_ntm"]), t_bucket)
    del sfnet

    def run(arc):
        out = np.empty(len(recs), np.float64)
        for i in range(0, len(recs), chunk):
            sl = slice(i, min(i + chunk, len(recs)))
            sub = {"hk_stm": jnp.asarray(b["hk_stm"][sl]),
                   "hk_ntm": jnp.asarray(b["hk_ntm"][sl]),
                   "th_stm": jnp.asarray(b["th_stm"][sl]),
                   "th_ntm": jnp.asarray(b["th_ntm"][sl]),
                   "bucket": jnp.asarray(t_bucket[sl])}
            out[sl] = np.asarray(forward(params, arc, sub)[0], np.float64)
        return out

    floats = run(arch)

    print(f"\n{label}  (oracle range {oracle.min()} .. {oracle.max()})")
    bad = report("B int reference - A oracle", ints, oracle)
    report("C jax teacher   - A oracle", floats, oracle)

    # CONTROL, and the null for `sf_act`: the same weights read through OUR
    # SqrCReLU tail instead of SF's. If sf_act were cosmetic this arm would
    # land in the same place; it must be far worse, or the flag is not doing
    # what its docstring claims.
    import dataclasses
    ctl = run(dataclasses.replace(arch, sf_act=False))
    report("  null: sf_act=False", ctl, oracle)

    if bad != 0.0:
        i = int(np.argmax(np.abs(ints - oracle)))
        print(f"  !! LAYOUT ERROR. worst row {i}: int={ints[i]} oracle="
              f"{oracle[i]} bucket={t_bucket[i]}\n     {fens[i]}")
        return 1
    return 0


def main() -> int:
    ap = argparse.ArgumentParser()
    ap.add_argument("warm")
    ap.add_argument("--fens", default=None)
    ap.add_argument("--corpus", default=None)
    ap.add_argument("--n", type=int, default=2000)
    ap.add_argument("--seed", type=int, default=20260801)
    ap.add_argument("--net", default=SFNET)
    ap.add_argument("--chunk", type=int, default=64)
    a = ap.parse_args()

    from archon_jax.spec import RECORD_DTYPE
    rc = 0

    if a.fens:
        raw = [ln.strip() for ln in Path(a.fens).read_text().splitlines()
               if ln.strip()]
        canon = [canonicalize(f) for f in raw]
        # Control: the canonicalization must be eval-invariant, or the whole
        # stm-canonical corpus convention is unsound. Free, so take it.
        if any(f.split()[1] == "b" for f in raw):
            d = np.abs(sf_eval(raw, a.net) - sf_eval(canon, a.net))
            print(f"control: sf-eval(raw) vs sf-eval(canonical) "
                  f"max|diff|={d.max()} over {len(d)}")
        recs = np.concatenate([fen_to_record(f, RECORD_DTYPE) for f in canon])
        rc |= run_panel(f"FENS {a.fens}", recs, canon, a.warm, a.net, a.chunk)

    if a.corpus:
        mm = np.memmap(a.corpus, dtype=RECORD_DTYPE, mode="r")
        rng = np.random.default_rng(a.seed)
        recs = np.asarray(mm[np.sort(rng.integers(0, len(mm), size=a.n))])
        del mm
        rc |= run_panel(f"CORPUS {Path(a.corpus).name} n={a.n}", recs,
                        records_to_fens(recs), a.warm, a.net, a.chunk)
    return rc


if __name__ == "__main__":
    raise SystemExit(main())
