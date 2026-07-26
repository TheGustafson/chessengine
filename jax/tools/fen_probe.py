"""PARITY PROOF 3 (deploy-level): JAX forward vs the ENGINE's int8 inference.

Builds bulletformat records from white-to-move FENs, runs the full JAX graph
with the dequantised OFFICIAL quant weights, and prints cp = value_logit * 400
(the network_scale the data contract §2 pins). Compare with the engine's
`eval` output on the same FENs with the same quant packed via netpack.py.

Expected agreement: within the engine's fixed-point pipeline error (the i16
accumulator + ft_shift + i8 L1 path truncates; a few cp), NOT bit-exact — the
engine computes in integers, we in f32.

Run:
  cd jax && JAX_PLATFORMS=cpu XLA_PYTHON_CLIENT_PREALLOCATE=false \
    uv run --no-project --with jax --with optax -- python tools/fen_probe.py \
    <quant.bin> <fens.txt>
"""

import os
import sys
from pathlib import Path

import numpy as np

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))
os.environ.setdefault("JAX_PLATFORMS", "cpu")
os.environ.setdefault("XLA_PYTHON_CLIENT_PREALLOCATE", "false")

from archon_jax import checkpoint  # noqa: E402
from archon_jax.features import map_batch  # noqa: E402
from archon_jax.spec import RECORD_DTYPE, ArchSpec, TrainConfig  # noqa: E402

PIECE_CODE = {"P": 0, "N": 1, "B": 2, "R": 3, "Q": 4, "K": 5}


def fen_to_record(fen: str) -> np.ndarray:
    """White-to-move FEN -> stm-canonicalized bulletformat record (score 0)."""
    placement, stm = fen.split()[:2]
    assert stm == "w", "probe FENs must be white to move (stored frame == board)"
    rec = np.zeros(1, dtype=RECORD_DTYPE)
    pieces = []
    for r, row in enumerate(placement.split("/")):
        rank = 7 - r
        f = 0
        for ch in row:
            if ch.isdigit():
                f += int(ch)
                continue
            color = 0 if ch.isupper() else 1
            pt = PIECE_CODE[ch.upper()]
            pieces.append((rank * 8 + f, color, pt))
            f += 1
    occ = 0
    for i, (sq, color, pt) in enumerate(sorted(pieces)):
        occ |= 1 << sq
        rec["pcs"][0][i // 2] |= ((color << 3) | pt) << (4 * (i % 2))
        if color == 0 and pt == 5:
            rec["ksq"] = sq
    rec["occ"] = occ
    rec["extra"][0] = (64, 64, 0)
    rec["result"] = 1
    return rec


def main(quant_path, fen_path):
    # D47V parity: env-driven arch (quant layout = post-factorizer-fold, so
    # factorize stays False here). PROBE_HL=768 PROBE_L2=32 PROBE_L3=32
    # PROBE_SKIP=1 PROBE_PSQT=1 for the d47v family; unset = classic full().
    import dataclasses, os
    arch = dataclasses.replace(
        ArchSpec.full(),
        hl=int(os.environ.get("PROBE_HL", ArchSpec.full().hl)),
        l2=int(os.environ.get("PROBE_L2", ArchSpec.full().l2)),
        l3=int(os.environ.get("PROBE_L3", ArchSpec.full().l3)),
        out_skip=bool(int(os.environ.get("PROBE_SKIP", "0"))),
        psqt=bool(int(os.environ.get("PROBE_PSQT", "0"))))
    cfg = TrainConfig()
    params = checkpoint.dequantise_quant(quant_path, arch)

    fens = [l.strip() for l in open(fen_path)
            if l.strip() and not l.startswith("#")]
    recs = np.concatenate([fen_to_record(f) for f in fens])
    batch = map_batch(recs, wdl=cfg.wdl, score_multiplier=1.0, scale=cfg.scale,
                      policy=False)

    import jax.numpy as jnp

    from archon_jax.model import forward
    jparams = {k: jnp.asarray(v) for k, v in params.items()}
    jbatch = {k: jnp.asarray(v) for k, v in batch.items() if k != "target"}
    value_logit, _, _ = forward(jparams, arch, jbatch)
    for fen, v in zip(fens, np.asarray(value_logit)):
        print(f"{v * 400.0:8.1f}\t{fen}")


if __name__ == "__main__":
    main(*sys.argv[1:3])
