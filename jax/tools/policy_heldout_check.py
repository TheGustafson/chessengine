"""Policy-head parity on the heldout dump: compute the MASKED softmax-CE with
the exact training semantics and compare against the trainer's reported
pol_ce (~2.424 at sb240), plus masked top-1 vs the packed teacher move.

Run:
  cd jax && JAX_PLATFORMS=cpu XLA_PYTHON_CLIENT_PREALLOCATE=false \
    uv run --no-project --with jax --with optax --with chess -- \
    python tools/policy_heldout_check.py <raw.bin> <heldout.bin> [n]
"""

import os
import sys
from pathlib import Path

import numpy as np

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))
os.environ.setdefault("JAX_PLATFORMS", "cpu")
os.environ.setdefault("XLA_PYTHON_CLIENT_PREALLOCATE", "false")

from archon_jax import checkpoint  # noqa: E402
from archon_jax.data import legal_move_masks  # noqa: E402
from archon_jax.features import map_batch  # noqa: E402
from archon_jax.spec import RECORD_DTYPE, ArchSpec, TrainConfig  # noqa: E402
from tools.forward_parity import run  # noqa: E402


def main(raw_path, heldout, n=512):
    import dataclasses, os
    arch = dataclasses.replace(
        ArchSpec.full(),
        factorize=bool(int(os.environ.get("PROBE_FACT", "0"))),
        psqt=bool(int(os.environ.get("PROBE_PSQT", "0"))),
        hl=int(os.environ.get("PROBE_HL", ArchSpec.full().hl)),
        l2=int(os.environ.get("PROBE_L2", ArchSpec.full().l2)),
        l3=int(os.environ.get("PROBE_L3", ArchSpec.full().l3)),
        out_skip=bool(int(os.environ.get("PROBE_SKIP", "0"))))
    cfg = TrainConfig()
    params = checkpoint.open_raw(raw_path, arch.param_shapes(), mmap=True)
    recs = np.fromfile(heldout, dtype=RECORD_DTYPE, count=n)
    batch = map_batch(recs, wdl=cfg.wdl, score_multiplier=cfg.score_multiplier,
                      scale=cfg.scale, policy=True)
    ids = batch["pol_target_id"]
    print(f"rows {n}, rows with packed move: {(ids >= 0).sum()}")
    masks = legal_move_masks(recs, ids)  # raises on any frame-tripwire failure
    print("legal masks built; frame tripwire passed for every packed target")

    _, p = run(params, batch, arch)
    logits = p + masks
    logp = logits - np.log(np.exp(logits - logits.max(1, keepdims=True)).sum(1, keepdims=True)) \
        - logits.max(1, keepdims=True)
    has = ids >= 0
    ce = -logp[np.arange(n), np.maximum(ids, 0)]
    ce = ce[has]
    print(f"masked pol_ce (mean over rows with target): {ce.mean():.4f}  "
          f"(trainer sb240 reported ~2.424)")
    top1 = (logits[has].argmax(1) == ids[has]).mean()
    legal_counts = (masks[has] == 0).sum(1)
    print(f"masked top-1 vs teacher move: {top1 * 100:.1f}%  "
          f"(avg legal moves/pos {legal_counts.mean():.1f} -> random {100 / legal_counts.mean():.1f}%)")


if __name__ == "__main__":
    main(sys.argv[1], sys.argv[2], int(sys.argv[3]) if len(sys.argv) > 3 else 512)
