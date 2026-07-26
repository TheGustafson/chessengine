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
    # Trailer-aware dequant (D47V): the bullet stream carries the classic
    # layout; PSQ1/SKP1 ride as trailers. Parse the stream with a
    # trailer-less arch, then reconstruct psqt/skip params from the trailers
    # so the jax forward matches what the ENGINE actually computes.
    stream_arch = dataclasses.replace(arch, psqt=False, out_skip=False)
    params = checkpoint.dequantise_quant(quant_path, stream_arch)
    if arch.psqt or arch.out_skip:
        raw = open(quant_path, "rb").read()
        i = raw.find(b"PSQ1")
        if arch.psqt:
            assert i >= 0, "psqt requested but no PSQ1 trailer"
            import struct as _s
            nb, rows = _s.unpack_from("<II", raw, i + 4)
            q = np.frombuffer(raw, dtype="<i4", count=rows * nb,
                              offset=i + 12).reshape(rows, nb)
            from archon_jax.checkpoint import PSQT_FIXED_MULT
            from archon_jax.spec import HALFKA_DIMS
            w = q.astype(np.float64) / PSQT_FIXED_MULT
            params["l0_psqt_hkw"] = w[:HALFKA_DIMS].astype(np.float32)
            params["l0_psqt_thw"] = w[HALFKA_DIMS:].astype(np.float32)
        if arch.out_skip:
            j = raw.find(b"SKP1")
            assert j >= 0, "out_skip requested but no SKP1 trailer"
            import struct as _s
            nb, hl = _s.unpack_from("<II", raw, j + 4)
            assert hl == arch.hl, (hl, arch.hl)
            params["skipw"] = np.frombuffer(
                raw, dtype="<f4", count=hl * nb, offset=j + 12
            ).reshape(hl, nb).copy()

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
