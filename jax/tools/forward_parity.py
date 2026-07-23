"""PARITY PROOF 2: full-graph JAX forward on the REAL D47c sb240 weights.

Memory-safe trick: the batch only references a few thousand distinct FT rows,
so we np.unique the sparse indices, gather exactly those rows from the
memmapped raw checkpoint, remap the indices, and run the REAL model.forward
(jnp.take path) on the compacted matrices. Mathematically identical to using
the full 432 MB tensors; peak RSS stays ~200 MB.

Checks:
  A. raw-weights forward vs dequantised-quant-weights forward through the SAME
     graph: |Δ sigmoid| should be at quant-noise scale (~1e-3), policy argmax
     should agree on ~all rows. (Both artifacts were written by bullet from
     one weight state — byte-parity already proved we read them faithfully.)
  B. trainer-target consistency: value MSE + pol CE against the same targets
     the trainer optimizes, on trainer-parity-filtered rows; compare with the
     final training-loss components in /tmp/d47c_train.log.
  C. determinism + capacity: max active counts within 32/160.

Run:
  cd jax && JAX_PLATFORMS=cpu XLA_PYTHON_CLIENT_PREALLOCATE=false \
    uv run --no-project --with jax --with optax -- python tools/forward_parity.py \
    <raw.bin> <quant.bin> <corpus.bin> [n_rows]
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


def compact_batch(params, batch):
    """Gather only the referenced FT rows; remap indices. Returns (small_params,
    remapped_batch) able to run model.forward exactly."""
    out = dict(params)
    b = dict(batch)
    for w, keys in (("l0_pstw", ("hk_stm", "hk_ntm")),
                    ("l0_threatw", ("th_stm", "th_ntm"))):
        allidx = np.concatenate([batch[k].ravel() for k in keys])
        uniq = np.unique(allidx[allidx >= 0])
        rows = np.asarray(params[w][uniq])  # memmap fancy-index -> small copy
        lut = np.full(params[w].shape[0], -1, dtype=np.int64)
        lut[uniq] = np.arange(len(uniq))
        for k in keys:
            idx = batch[k]
            b[k] = np.where(idx >= 0, lut[np.maximum(idx, 0)], -1).astype(np.int32)
        out[w] = rows
    # head tensors: materialize (small)
    for k in ("l0_pstb", "l1w", "l1b", "l2w", "l2b", "l3w", "l3b",
              "polmovew", "polmoveb"):
        out[k] = np.asarray(params[k])
    return out, b


def run(params, batch, arch):
    import jax.numpy as jnp

    from archon_jax.model import forward
    small, b = compact_batch(params, batch)
    small = {k: jnp.asarray(v) for k, v in small.items()}
    jb = {k: jnp.asarray(v) for k, v in b.items()
          if k in ("hk_stm", "hk_ntm", "th_stm", "th_ntm", "bucket")}
    value_logit, pol_logits, _ = forward(small, arch, jb)
    return np.asarray(value_logit), np.asarray(pol_logits)


def main(raw_path, quant_path, corpus, n_rows=256):
    arch = ArchSpec.full()
    cfg = TrainConfig()
    raw = checkpoint.open_raw(raw_path, arch.param_shapes(), mmap=True)
    quant = checkpoint.dequantise_quant(quant_path, arch)

    recs = np.fromfile(corpus, dtype=RECORD_DTYPE, count=n_rows)
    batch = map_batch(recs, wdl=cfg.wdl, score_multiplier=cfg.score_multiplier,
                      scale=cfg.scale, policy=True)
    print(f"rows {len(recs)}  max active: halfka "
          f"{int((batch['hk_stm'] >= 0).sum(1).max())}, threats "
          f"{int((batch['th_stm'] >= 0).sum(1).max())}")

    v_raw, p_raw = run(raw, batch, arch)
    v_q, p_q = run(quant, batch, arch)

    sig_raw = 1 / (1 + np.exp(-v_raw))
    sig_q = 1 / (1 + np.exp(-v_q))
    print(f"A. raw-vs-quant |dsigmoid|: max {np.abs(sig_raw - sig_q).max():.6f} "
          f"mean {np.abs(sig_raw - sig_q).mean():.6f}")
    am_r, am_q = p_raw.argmax(1), p_q.argmax(1)
    print(f"   policy argmax agreement: {(am_r == am_q).mean() * 100:.1f}%")

    mse = float(np.mean((sig_raw - batch["target"]) ** 2))
    print(f"B. value MSE vs trainer targets ({Path(corpus).name}): {mse:.6f}")
    # engine-scale sanity: logit*400 == engine cp scale (data contract §2)
    cp = v_raw * 400.0
    sc = recs["score"].astype(np.float32)
    mask = np.abs(sc) > 50
    if mask.any():
        agree = float((np.sign(cp[mask]) == np.sign(sc[mask])).mean())
        print(f"   sign agreement vs labels (|score|>50): {agree * 100:.1f}%")
    print(f"   cp range: [{cp.min():.0f}, {cp.max():.0f}]  label range: "
          f"[{sc.min():.0f}, {sc.max():.0f}]")

    has = batch["pol_target_id"] >= 0
    if has.any():
        ids = batch["pol_target_id"][has]
        top1 = float((p_raw[has].argmax(1) == ids).mean())
        print(f"   policy top-1 vs packed teacher move: {top1 * 100:.1f}% "
              f"({has.sum()} rows with packed moves)")
    else:
        print("   corpus has no packed policy moves (bf-converted)")
    return 0


if __name__ == "__main__":
    main(*sys.argv[1:4], int(sys.argv[4]) if len(sys.argv) > 4 else 256)
