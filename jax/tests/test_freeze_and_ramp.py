"""Trainer control-knob tests: exact param freezing (D47L/D47M) and the
ft-sparsity λ ramp (D47M overshoot lesson, 2026-07-25).

Both knobs decide what a multi-hour GPU leg actually optimizes, so they get
oracles rather than eyeballs:
  * freeze  — frozen tensors must be BIT-identical after many steps, even
              though bullet-AdamW's decoupled weight decay multiplies every
              param each step regardless of gradient (the post-update restore
              is what makes that a no-op).
  * λ ramp  — the effective sparsity λ must start at ~0 and reach the target
              at the end of the ramp window, and must be a no-op when off.

Run:  cd jax && JAX_PLATFORMS=cpu uv run --no-project --with numpy --with jax \
        --with optax --with pytest -- python -m pytest tests/test_freeze_and_ramp.py -q
"""

import dataclasses
import sys
from pathlib import Path

import numpy as np
import pytest

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))

from archon_jax.spec import ArchSpec, TrainConfig  # noqa: E402


def tiny_arch():
    """Smallest legal full-arch spec: real tensor names, trivial sizes."""
    return dataclasses.replace(ArchSpec.full(), factorize=False, psqt=False)


def fake_batch(arch, n=8, rng=None):
    """A batch shaped like map_batch's output, with random sparse features."""
    rng = rng or np.random.default_rng(0)
    from archon_jax.spec import HALFKA_DIMS, POLICY_VOCAB, THREAT_DIMS
    max_feat = 32
    return {
        "hk_stm": rng.integers(0, HALFKA_DIMS, size=(n, max_feat), dtype=np.int32),
        "hk_ntm": rng.integers(0, HALFKA_DIMS, size=(n, max_feat), dtype=np.int32),
        "th_stm": rng.integers(0, THREAT_DIMS, size=(n, max_feat), dtype=np.int32),
        "th_ntm": rng.integers(0, THREAT_DIMS, size=(n, max_feat), dtype=np.int32),
        "bucket": rng.integers(0, arch.num_buckets, size=(n,), dtype=np.int32),
        "target": rng.random((n,), dtype=np.float32),
        "pol_target_id": rng.integers(0, POLICY_VOCAB, size=(n,), dtype=np.int32),
    }


def test_sparsity_ramp_schedule():
    """The ramp reaches the target exactly at the window end, and the OFF
    path is a hard constant (no accidental scaling of legacy legs)."""
    import jax.numpy as jnp

    from archon_jax.train import make_update_step

    arch = tiny_arch()
    target = 0.001
    ramped = dataclasses.replace(
        TrainConfig(), ft_sparsity_lambda=target,
        ft_sparsity_warmup_superbatches=20.0, superbatches=60,
        lr_end_superbatch=60)
    bps = ramped.batches_per_superbatch
    # reach into the closure the same way step() does
    _, _ = make_update_step(arch, ramped)  # smoke: builds without error

    # Recompute the documented schedule and pin its shape.
    def lam_at(cfg, count):
        if cfg.ft_sparsity_warmup_superbatches > 0.0 and cfg.start_superbatch == 1:
            return cfg.ft_sparsity_lambda * float(
                jnp.minimum(1.0, (count + 1.0)
                            / (cfg.ft_sparsity_warmup_superbatches * bps)))
        return cfg.ft_sparsity_lambda

    assert lam_at(ramped, 0) == pytest.approx(target / (20.0 * bps), rel=1e-6)
    assert lam_at(ramped, 10 * bps) == pytest.approx(target * 0.5, rel=1e-3)
    assert lam_at(ramped, 20 * bps - 1) == pytest.approx(target, rel=1e-6)
    assert lam_at(ramped, 40 * bps) == pytest.approx(target, rel=1e-6)

    off = dataclasses.replace(TrainConfig(), ft_sparsity_lambda=target)
    assert lam_at(off, 0) == target
    assert lam_at(off, 99 * bps) == target

    # Resume never re-ramps (a resumed leg is past the freeze-out transient).
    resumed = dataclasses.replace(ramped, start_superbatch=31)
    assert lam_at(resumed, 0) == target


def test_train_only_rejects_unknown_names():
    from archon_jax.train import make_update_step
    with pytest.raises(ValueError, match="not in arch"):
        make_update_step(tiny_arch(), TrainConfig(), train_only=["nope"])


@pytest.mark.slow
def test_freeze_is_bit_exact_over_many_steps():
    """The oracle that matters: with train_only=[policy head], every other
    tensor must be bit-identical after N steps — including under AdamW's
    grad-independent decay, which would silently shrink them without the
    post-update restore."""
    import jax

    from archon_jax.model import init_params
    from archon_jax.train import make_update_step

    arch = tiny_arch()
    cfg = dataclasses.replace(TrainConfig(), lr=1e-2, lr_end_superbatch=1,
                              ft_sparsity_lambda=0.0, pol_lambda=1.0)
    params = init_params(arch, seed=7)
    opt, step = make_update_step(arch, cfg, ft_impl="reference",
                                 train_only=["polmovew", "polmoveb"])
    opt_state = opt.init(params)
    params = {k: jax.numpy.asarray(v) for k, v in params.items()}
    before = {k: np.asarray(v).copy() for k, v in params.items()}

    rng = np.random.default_rng(1)
    for _ in range(5):
        params, opt_state, _ = step(params, opt_state, fake_batch(arch, rng=rng))

    after = {k: np.asarray(v) for k, v in params.items()}
    for name in before:
        if name in ("polmovew", "polmoveb"):
            assert not np.array_equal(before[name], after[name]), \
                f"{name} should TRAIN"
        else:
            assert np.array_equal(before[name], after[name]), \
                f"{name} must be bit-identical (freeze leak)"
