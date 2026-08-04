"""JAX↔Rust parity for the DISAGREEMENT HEAD, end to end through the real
export path.

What it actually exercises, in order: `checkpoint.dis_section_bytes` writes a
DIS1 section onto a real quant.bin → `netpack.py` passes it through into a
`.cenn` → `sota_loader::parse_dis_section` reads it back → `SotaHead::
disagreement` computes it → `nnue-eval` prints both columns. Against that, the
JAX graph's own `dis_pred` on the same positions.

**The engine cannot equal the trainer's float graph and it is not a bug that it
does not** — the accumulator is int8/int16 and `hl1` is a u8 with 127 levels.
So the tool runs THREE arms and states which difference is which, instead of
choosing a tolerance that makes one number look acceptable:

  * `jax(f32)`      — the trainer's own graph. The reference the head was fitted
                      in, and NOT the thing the engine should reproduce exactly.
  * `engine-model`  — the engine's documented fixed-point pipeline, reproduced
                      here in exact int64: the per-column i8 re-quantisation of
                      the HalfKA table at load (`sota_loader::quantise_halfka`,
                      perf study 32), the threat merge shift, the [0,255] clamp
                      and the `(a·b) >> FT_SHIFT` pairwise truncation
                      (`sota_simd::scalar::activate`). This IS what the engine
                      should reproduce, to f32 rounding.
  * `engine`        — the deployed binary, through the real export path.

Measured 2026-08-03 on `d50w768-sb1030-ema` @ wl50, proj 32:
`engine − engine-model` is **2.98e-07** while `engine-model − jax(f32)` is
**2.4e-02**. Read that second number before using this head for anything: the
quantity a search consumer sees is the ENGINE's, and it differs from the
trainer's by ~6 % of the head's own mean output. That is the cost of the
int8 substrate, not of this plumbing, and it is exactly as large for the value
head — but a consumer calibrated on training-time residuals will be off by it.

And against two NULLS, because "close enough" means nothing without one:

  * `column swap`      — the sgn and mag columns exchanged.
  * `proj_w transposed`— the classic layout bug.

A PASS is `|engine − engine-model| ≤ tol` AND that residual at least 20×
smaller than the swap null. Anything else is printed as a number, not smoothed
away.

Run (from jax/):
  JAX_PLATFORMS=cpu XLA_PYTHON_CLIENT_PREALLOCATE=false \
    uv run --no-project --with numpy==2.5.1 --with jax==0.11.0 \
      --with optax==0.2.8 -- python tools/dis_parity.py \
      --quant <quant.bin> --run-spec tools/nnue/runs/d50_w768.json \
      --fens tools/nnue/calib/wl50.fens [--dis-from-raw <raw.bin>] [--proj 32]

Env: ARCHON_ENGINE (an engine built with the run's arch features), ARCHON_TMP.
"""

from __future__ import annotations

import argparse
import dataclasses
import json
import os
import subprocess
import sys
import tempfile
from pathlib import Path

import numpy as np

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))
os.environ.setdefault("JAX_PLATFORMS", "cpu")
os.environ.setdefault("XLA_PYTHON_CLIENT_PREALLOCATE", "false")

from archon_jax import checkpoint as ck            # noqa: E402
from archon_jax.features import map_batch          # noqa: E402
from archon_jax.spec import (                      # noqa: E402
    RECORD_DTYPE, ArchSpec, TrainConfig,
)

ROOT = Path(__file__).resolve().parents[2]
ENGINE = os.environ.get("ARCHON_ENGINE", str(ROOT / "target/nnue-gate/release/chessengine"))
PIECE_CODE = {"P": 0, "N": 1, "B": 2, "R": 3, "Q": 4, "K": 5}

# The engine's u8 pairwise grid: ft_u8 = floor(hl1 · FT_QUANT² / 2^FT_SHIFT).
FT_QUANT, FT_SHIFT = 255, 9
FT_GRID = FT_QUANT * FT_QUANT / (1 << FT_SHIFT)     # 127.001953125


# ---------------------------------------------------------------------------
# positions (the stm-canonical frame; sf_teacher_parity.py's contract)
# ---------------------------------------------------------------------------

def canonicalize(fen: str) -> str:
    """Any FEN -> the white-to-move placement the corpus frame stores. Castling
    and ep are dropped, which is what bulletformat does and is harmless here:
    HalfKAv2_hm + FullThreats are blind to both (800/800 identical twins,
    2026-07-30) — so the ENGINE may read the original FEN and JAX the stripped
    record and both see the same features."""
    placement, stm = fen.split()[:2]
    if stm == "w":
        return placement
    return "/".join(r.swapcase() for r in placement.split("/")[::-1])


def fen_to_record(placement: str) -> np.ndarray:
    rec = np.zeros(1, dtype=RECORD_DTYPE)
    pieces = []
    for r, row in enumerate(placement.split("/")):
        rank, f = 7 - r, 0
        for chx in row:
            if chx.isdigit():
                f += int(chx)
                continue
            pieces.append((rank * 8 + f, 0 if chx.isupper() else 1,
                           PIECE_CODE[chx.upper()]))
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


# ---------------------------------------------------------------------------
# weights
# ---------------------------------------------------------------------------

def load_ft_raw(quant: Path, arch: ArchSpec) -> dict:
    """The two feature transformers as the INTEGERS the file stores.

    Only the FT is read: `hl1` is all the disagreement head consumes, so the
    value tail, the policy rows and the PSQ1/SKP1 trailers are irrelevant here
    and reading them would multiply the resident set for nothing. Integers
    rather than floats because the engine model below needs the exact stored
    values, and the float reference is one divide away."""
    raw = np.memmap(quant, dtype=np.uint8, mode="r")
    off, out = 0, {}
    for name, kind, mult, transpose in arch.quant_layout():
        shape = arch.param_shapes()[name]
        n = int(np.prod(shape))
        width = {"i16": 2, "i8": 1, "f32": 4}[kind]
        if name in ("l0_pstw", "l0_pstb", "l0_threatw"):
            assert not transpose, name
            view = raw[off:off + width * n].view({"i16": "<i2", "i8": "<i1"}[kind])
            out[name] = np.array(view).reshape(shape)
            out[name + "_mult"] = mult
        off += width * n
    return out


def float_ft(ftq: dict) -> dict:
    return {k: (v.astype(np.float32) / ftq[k + "_mult"])
            for k, v in ftq.items() if not k.endswith("_mult")}


# ---------------------------------------------------------------------------
# The engine's integer pipeline, reproduced exactly (sota_loader.rs
# halfka_column_scales / quantise_halfka; sota_simd.rs scalar::activate).
#
# This is NOT a tolerance-widening device. It is the difference between
# "the engine disagrees with the trainer by 2e-2 and we do not know why" and
# "the engine reproduces its own documented fixed-point pipeline to f32
# rounding" — the second is a parity result, the first is not.
# ---------------------------------------------------------------------------

def _round_half_away(n: np.ndarray, d: np.ndarray) -> np.ndarray:
    q = (2 * np.abs(n).astype(np.int64) + d) // (2 * d)
    return np.where(n < 0, -q, q)


def engine_ft_u8(ftq: dict, arch: ArchSpec, batch: dict) -> np.ndarray:
    """`out: [u8; L1]` for each position — the vector the engine's dis head
    consumes, computed in int64 so every step is exact."""
    hkw = ftq["l0_pstw"].astype(np.int64)                    # (HALFKA, hl) i16
    bias = ftq["l0_pstb"].astype(np.int64)                   # (hl,) i16
    thw = ftq.get("l0_threatw")
    # Per-column i8 scale, then the quantised table (loader lines: peak over
    # HalfKA rows only — the bias is rebased onto it but does not set it).
    peak = np.abs(hkw).max(axis=0)
    scale = np.clip(-(-peak // 127), 1, 255)                 # div_ceil, clamped
    hk_q = np.clip(_round_half_away(hkw, scale), -127, 127)  # i8
    bias_q = _round_half_away(bias, scale)                   # i16, NOT clamped
    shift = 8 - int(arch.threat_quant()).bit_length() + 1

    n, hl = batch["hk_stm"].shape[0], arch.hl
    h = hl // 2
    out = np.zeros((n, hl), dtype=np.int64)
    for side, (hk_key, th_key) in enumerate([("hk_stm", "th_stm"),
                                             ("hk_ntm", "th_ntm")]):
        hk = batch[hk_key]
        pst = np.stack([bias_q + hk_q[r[r >= 0]].sum(axis=0) for r in hk])
        if thw is not None and not arch.halfka_only:
            th = batch[th_key]
            tacc = np.stack([thw[r[r >= 0]].astype(np.int64).sum(axis=0)
                             for r in th])
        else:
            tacc = np.zeros_like(pst)
        merged = np.clip(pst * scale + (tacc << shift), 0, 255)
        out[:, side * h:(side + 1) * h] = (merged[:, :h] * merged[:, h:]) >> 9
    return out


def dis_from_raw(raw_path: Path, hl: int, proj: int) -> dict | None:
    """Pull the four dis tensors out of a bullet raw checkpoint WITHOUT loading
    the other 260 MB, applying the same shape rules `checkpoint.load_warm` does.
    Returns None when the checkpoint's head does not match the requested proj."""
    idx = ck.read_raw_index(raw_path)
    if "dis_projw" not in idx:
        return None
    got = {}
    for name in ck.DIS_TENSORS:
        o, c = idx[name]
        got[name] = np.array(np.memmap(raw_path, dtype="<f4", mode="r",
                                       offset=o, shape=(c,)))
    if got["dis_projw"].size != hl * proj:
        return None
    out = {"dis_projw": got["dis_projw"].reshape(hl, proj),
           "dis_projb": got["dis_projb"].reshape(proj)}
    if got["dis_outw"].size == proj:            # the pre-d7582485 single lane
        rng = np.random.default_rng(0xD15)
        col0 = np.sqrt(2.0 / proj) * rng.standard_normal(proj)
        out["dis_outw"] = np.stack([col0, got["dis_outw"]], 1).astype(np.float32)
        out["dis_outb"] = np.array([0.0, float(got["dis_outb"][0])], np.float32)
    else:
        out["dis_outw"] = got["dis_outw"].reshape(proj, ck.DIS_OUTS)
        out["dis_outb"] = got["dis_outb"].reshape(ck.DIS_OUTS)
    return out


def dis_synthetic(hl: int, proj: int, seed: int) -> dict:
    """A deterministic head with NO symmetry anywhere — which is what makes a
    transposed or column-swapped read detectable. The scale is chosen so the
    CReLU is genuinely active (some units clip at 1, some at 0): a head that
    saturated everywhere would agree with any wrong implementation."""
    rng = np.random.default_rng(seed)
    return {
        "dis_projw": (rng.standard_normal((hl, proj)) * 0.05).astype(np.float32),
        "dis_projb": (rng.standard_normal(proj) * 0.10).astype(np.float32),
        "dis_outw": (rng.standard_normal((proj, ck.DIS_OUTS)) * 0.50).astype(np.float32),
        "dis_outb": np.array([0.02, 0.11], np.float32),
    }


# ---------------------------------------------------------------------------
# the two implementations
# ---------------------------------------------------------------------------

def jax_hl1(ft_params, arch, batch):
    """`hl1` from the trainer's OWN graph — the reference implementation."""
    import jax.numpy as jnp
    from archon_jax.model import _crelu, _ft_accumulate_pair, _pairwise

    p = {k: jnp.asarray(v) for k, v in ft_params.items()}
    b = {k: jnp.asarray(v) for k, v in batch.items() if k != "target"}
    # psqt off for the ACCUMULATOR call only: the PSQ1 channel joins the value
    # logit outside every nonlinearity and never reaches `hl1`, so the head
    # cannot see it and loading its 2.7 MB table would be pure cost.
    ft_arch = dataclasses.replace(arch, psqt=False)
    acc_stm, acc_ntm, _, _ = _ft_accumulate_pair(p, b, ft_arch, "reference")
    return np.asarray(jnp.concatenate([_pairwise(_crelu(acc_stm), arch.hl),
                                       _pairwise(_crelu(acc_ntm), arch.hl)],
                                      axis=-1))


def dis_forward(hl1: np.ndarray, dis: dict) -> np.ndarray:
    """The head itself, in f32 — the identical expression on both sides, so a
    difference between two arms is a difference in `hl1` and nothing else."""
    h = np.clip(hl1.astype(np.float32) @ dis["dis_projw"] + dis["dis_projb"], 0.0, 1.0)
    return h @ dis["dis_outw"] + dis["dis_outb"]


def engine_dis(cenn: Path, fens: list[str]) -> np.ndarray:
    proc = subprocess.run([ENGINE, "nnue-eval", str(cenn)],
                          input="\n".join(fens) + "\n", text=True,
                          capture_output=True, timeout=1800)
    rows = []
    for line in proc.stdout.splitlines():
        if "\tdis " not in line:
            continue
        _, tail = line.split("\tdis ", 1)
        a, b = tail.split()[:2]
        rows.append((float(a), float(b)))
    if len(rows) != len(fens):
        raise RuntimeError(
            f"engine printed {len(rows)} dis rows for {len(fens)} fens.\n"
            f"stdout head: {proc.stdout[:400]!r}\nstderr: {proc.stderr[:400]!r}")
    return np.array(rows)


def build_cenn(quant: Path, dis: dict, arch: ArchSpec, spec: dict,
               tmp: Path, net_scale: int = 400) -> tuple[Path, Path]:
    """quant.bin + a DIS1 section -> .cenn, through netpack.py itself."""
    qd = tmp / "quant-with-dis.bin"
    with open(qd, "wb") as f:
        f.write(quant.read_bytes())
        f.write(ck.dis_section_bytes(dis, arch))
    cenn = tmp / "parity.cenn"
    env = dict(os.environ)
    env.update({
        "D40_ACC": str(arch.hl), "D40_L2": str(arch.l2), "D40_L3": str(arch.l3),
        "D40_BUCKETS": str(arch.num_buckets),
        "D40_HALFKA_ONLY": "1" if arch.halfka_only else "0",
        "D40_POLICY": "1" if arch.policy else "0",
        "D40_POL_DIM": str(arch.pol_proj or arch.l2),
        "D40_THREAT_QUANT": str(arch.threat_quant()),
        "D40_NETWORK_SCALE": str(net_scale),
    })
    r = subprocess.run([sys.executable, str(ROOT / "tools/nnue/netpack.py"),
                        str(qd), str(cenn), "d40"], env=env, text=True,
                       capture_output=True, timeout=1800)
    if r.returncode != 0 or not cenn.exists():
        raise RuntimeError(f"netpack failed:\n{r.stdout}\n{r.stderr}")
    print(r.stdout.rstrip())
    return qd, cenn


# ---------------------------------------------------------------------------

def report(name: str, a: np.ndarray, b: np.ndarray) -> float:
    d = a - b
    scale = float(np.abs(b).mean()) + 1e-12
    print(f"  {name:22s} "
          f"col0 max|d| {np.abs(d[:, 0]).max():.3e}  rms {np.sqrt((d[:, 0]**2).mean()):.3e} | "
          f"col1 max|d| {np.abs(d[:, 1]).max():.3e}  rms {np.sqrt((d[:, 1]**2).mean()):.3e} | "
          f"rel {np.abs(d).max() / scale:.3e}")
    return float(np.abs(d).max())


def main() -> int:
    ap = argparse.ArgumentParser()
    ap.add_argument("--quant", required=True)
    ap.add_argument("--run-spec", required=True)
    ap.add_argument("--fens", default=str(ROOT / "tools/nnue/calib/wl50.fens"))
    ap.add_argument("--dis-from-raw", default=None)
    ap.add_argument("--proj", type=int, default=0,
                    help="dis_proj override; default = the run spec's")
    ap.add_argument("--seed", type=int, default=20260803)
    ap.add_argument("--network-scale", type=int, default=400,
                    help="header network_scale for the packed .cenn. Affects "
                         "ONLY the printed cp value, never the dis columns "
                         "(the head branches off hl1, upstream of it). The "
                         "canonical net's 600 is a post-hoc header rewrite "
                         "(tools/nnue/rescale_net_header.py), so a net packed "
                         "here at the default prints a different cp than "
                         "engines/canonical/net.cenn on the same position — "
                         "expected, and not a parity failure.")
    ap.add_argument("--tol", type=float, default=1e-5,
                    help="max |engine - jax(qgrid)| accepted, in probability "
                         "units. Stated up front; do NOT move it to pass.")
    args = ap.parse_args()

    spec = json.loads(Path(args.run_spec).read_text())
    arch = dataclasses.replace(ArchSpec.from_run_spec(spec), factorize=False)
    if args.proj:
        arch = dataclasses.replace(arch, dis_proj=args.proj)
    if not arch.dis_proj:
        print("run spec has arch_dis_proj = 0; pass --proj N", file=sys.stderr)
        return 2
    print(f"arch: hl={arch.hl} l2={arch.l2} l3={arch.l3} buckets={arch.num_buckets} "
          f"dis_proj={arch.dis_proj} threat_quant={arch.threat_quant()}")

    dis = None
    if args.dis_from_raw:
        dis = dis_from_raw(Path(args.dis_from_raw), arch.hl, arch.dis_proj)
        print(f"dis head: {'from ' + args.dis_from_raw if dis else 'raw did not match, synthesising'}")
    if dis is None:
        dis = dis_synthetic(arch.hl, arch.dis_proj, args.seed)
        print(f"dis head: synthetic (seed {args.seed})")

    fens = [ln.strip() for ln in Path(args.fens).read_text().splitlines() if ln.strip()]
    recs = np.concatenate([fen_to_record(canonicalize(f)) for f in fens])
    cfg = TrainConfig()
    batch = map_batch(recs, wdl=cfg.wdl, score_multiplier=1.0, scale=cfg.scale,
                      policy=False)

    tmpdir = os.environ.get("ARCHON_TMP")
    with tempfile.TemporaryDirectory(dir=tmpdir) as td:
        tmp = Path(td)
        _, cenn = build_cenn(Path(args.quant), dis, arch, spec, tmp,
                             args.network_scale)
        eng = engine_dis(cenn, fens)

    ftq = load_ft_raw(Path(args.quant), arch)
    hl1_float = jax_hl1(float_ft(ftq), arch, batch)
    hl1_model = engine_ft_u8(ftq, arch, batch).astype(np.float64) / FT_GRID

    jf = dis_forward(hl1_float, dis)
    jm = dis_forward(hl1_model, dis)
    swap = {**dis, "dis_outw": np.ascontiguousarray(dis["dis_outw"][:, ::-1]),
            "dis_outb": np.ascontiguousarray(dis["dis_outb"][::-1])}
    js = dis_forward(hl1_model, swap)
    # `proj_w` read with its two axes exchanged. Only expressible when the
    # reshape is legal; when it is not, the swap null carries the section.
    transp = None
    if arch.hl == arch.dis_proj or arch.dis_proj > 1:
        tw = dis["dis_projw"].reshape(arch.dis_proj, arch.hl).T \
            if dis["dis_projw"].size == arch.hl * arch.dis_proj else None
        if tw is not None:
            transp = dis_forward(hl1_model, {**dis,
                                             "dis_projw": np.ascontiguousarray(tw)})

    print(f"\nn={len(fens)} positions, both columns (probability units)")
    print("  ARMS (engine minus the named reference)")
    report("engine - jax(f32)", eng, jf)
    d_model = report("engine - engine-model", eng, jm)
    print("  MODEL (size of the fixed-point pipeline the model reproduces)")
    report("engine-model - jax(f32)", jm, jf)
    print("  NULLS (what a WRONG wiring reads like)")
    d_swap = report("column swap", js, jm)
    if transp is not None:
        report("proj_w transposed", transp, jm)

    print(f"\nmean |dis| col0 {np.abs(jm[:, 0]).mean():.4f}  "
          f"col1 {np.abs(jm[:, 1]).mean():.4f}")
    ok = d_model <= args.tol and d_model * 20.0 < d_swap
    print(f"\n{'PASS' if ok else 'FAIL'}: "
          f"max|engine - engine-model| = {d_model:.3e} (tol {args.tol:.1e}), "
          f"null max|column swap| = {d_swap:.3e} "
          f"({d_swap / max(d_model, 1e-30):.1f}x the residual)")
    return 0 if ok else 1


if __name__ == "__main__":
    raise SystemExit(main())
