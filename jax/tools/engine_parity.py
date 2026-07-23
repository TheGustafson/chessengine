"""PARITY PROOF 3 (deploy-level, distributional): engine int8 inference vs the
JAX float graph on real corpus positions.

Pipeline: corpus records -> reconstructed FENs (the stored stm-canonicalized
frame: white to move, no castling/ep) -> prebuilt engine `eval` with the
OFFICIAL sb240 quant packed to .cenn by netpack.py; versus -> JAX forward with
the dequantised SAME quant + the engine's outer shape replicated exactly
(apply_outer_shape, src/eval/mod.rs:419-436: v*(77871+material)/77871 with
material = 534*pawns + npm over PIECE_VALUES {N781,B825,R1276,Q2538},
optimism=0, rule50=0, i64 truncating division).

The residual is then PURELY the engine's fixed-point pipeline (u8 FT >>9,
i8 L1) vs f32 — the quantity to characterize for the study.

Run:
  cd jax && JAX_PLATFORMS=cpu XLA_PYTHON_CLIENT_PREALLOCATE=false \
    uv run --no-project --with jax --with optax -- python tools/engine_parity.py \
    <quant.bin> <cenn> <corpus.bin> [n]
"""

import os
import subprocess
import sys
from pathlib import Path

import numpy as np

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))
os.environ.setdefault("JAX_PLATFORMS", "cpu")
os.environ.setdefault("XLA_PYTHON_CLIENT_PREALLOCATE", "false")

from archon_jax import checkpoint  # noqa: E402
from archon_jax.features import DecodedBatch, map_batch  # noqa: E402
from archon_jax.spec import RECORD_DTYPE, ArchSpec, TrainConfig  # noqa: E402

ENGINE = "/home/nick/projects/chessengine/target/release/chessengine"
PIECE_CHARS = "PNBRQK"
PIECE_VALUES = np.array([0, 781, 825, 1276, 2538, 0])  # types.rs:636-644 (P/K excl)
MATERIAL_PAWN_WEIGHT = 534   # eval/params.rs:20
EVAL_BLEND_BASE = 77871      # eval/params.rs:27


def rec_to_fen(dec: DecodedBatch, i: int) -> str:
    rows = []
    for rank in range(7, -1, -1):
        row, empty = "", 0
        for file in range(8):
            code = dec.mailbox_code[i][rank * 8 + file]
            if code == 0:
                empty += 1
                continue
            if empty:
                row += str(empty)
                empty = 0
            ch = PIECE_CHARS[(code & 7) - 1]
            row += ch if code < 8 else ch.lower()
        if empty:
            row += str(empty)
        rows.append(row)
    return "/".join(rows) + " w - - 0 1"


def material(dec: DecodedBatch, i: int) -> int:
    sel = dec.row == i
    pts = dec.pt[sel]
    pawns = int((pts == 0).sum())
    npm = int(PIECE_VALUES[pts[(pts >= 1) & (pts <= 4)]].sum())
    return MATERIAL_PAWN_WEIGHT * pawns + npm


def outer_shape(cp: float, mat: int) -> int:
    # i64 truncating division (Rust `/` truncates toward zero)
    v = int(cp) * (EVAL_BLEND_BASE + mat)
    return int(v / EVAL_BLEND_BASE) if v >= 0 else -int(-v // EVAL_BLEND_BASE)


def engine_evals(cenn: str, fens: list[str]) -> np.ndarray:
    cmds = ["uci", f"setoption name EvalFile value {cenn}", "isready"]
    for fen in fens:
        cmds += [f"position fen {fen}", "eval"]
    cmds.append("quit")
    proc = subprocess.run([ENGINE], input="\n".join(cmds) + "\n", text=True,
                          capture_output=True, timeout=600)
    vals = [int(line.split()[2]) for line in proc.stdout.splitlines()
            if line.startswith("Final evaluation:")]
    assert len(vals) == len(fens), (len(vals), proc.stdout[-500:])
    return np.array(vals)


def main(quant_path, cenn, corpus, n=64):
    arch = ArchSpec.full()
    cfg = TrainConfig()
    params = checkpoint.dequantise_quant(quant_path, arch)
    recs = np.fromfile(corpus, dtype=RECORD_DTYPE, count=n)
    dec = DecodedBatch(recs)
    fens = [rec_to_fen(dec, i) for i in range(n)]
    mats = [material(dec, i) for i in range(n)]

    eng = engine_evals(cenn, fens)

    batch = map_batch(recs, wdl=cfg.wdl, score_multiplier=1.0, scale=cfg.scale,
                      policy=False)
    import jax.numpy as jnp

    from archon_jax.model import forward
    jparams = {k: jnp.asarray(v) for k, v in params.items()}
    jbatch = {k: jnp.asarray(v) for k, v in batch.items() if k != "target"}
    logit, _, _ = forward(jparams, arch, jbatch)
    cp = np.asarray(logit) * 400.0
    ours = np.array([outer_shape(float(c), m) for c, m in zip(cp, mats)])

    d = eng - ours
    print(f"n={n}  engine-vs-JAX (outer-shaped, cp): "
          f"mean {d.mean():+.1f}  std {d.std():.1f}  max|d| {np.abs(d).max()}")
    q = np.percentile(np.abs(d), [50, 90, 100])
    print(f"|d| percentiles p50 {q[0]:.0f}  p90 {q[1]:.0f}  p100 {q[2]:.0f}")
    print(f"sign agreement: {(np.sign(eng) == np.sign(ours)).mean() * 100:.0f}%")
    worst = np.argsort(-np.abs(d))[:5]
    for i in worst:
        print(f"  d={d[i]:+4d} eng={eng[i]:+5d} jax={ours[i]:+5d} "
              f"pieces={int((dec.row == i).sum()):2d} {fens[i][:40]}")


if __name__ == "__main__":
    main(*sys.argv[1:4], int(sys.argv[4]) if len(sys.argv) > 4 else 64)
