"""ft_kernel vs the reference FT path: forward + gradient equivalence.

The scan impl is pure XLA and is tested everywhere (CPU included). The
pallas impl needs a CUDA device AND the explicit ARCHON_JAX_ALLOW_GPU=1
opt-in (train.py's guard philosophy: no silent GPU grabs); it is exercised
by the GPU bench + these tests during GPU sessions:

  ARCHON_JAX_ALLOW_GPU=1 uv run --no-project --with "jax[cuda12]" \
    --with pytest -- python -m pytest tests/test_ft_kernel.py -q
"""

import os

import numpy as np
import pytest

jax = pytest.importorskip("jax")
import jax.numpy as jnp  # noqa: E402

from archon_jax.ft_kernel import ft_accumulate_pallas, ft_accumulate_scan  # noqa: E402
from archon_jax.model import _ft_accumulate  # noqa: E402

ROWS_W, ROWS_T, H = 512, 1024, 128
S_W, S_T = 32, 160


def _gpu_available() -> bool:
    if os.environ.get("ARCHON_JAX_ALLOW_GPU") != "1":
        return False
    try:
        return jax.devices()[0].platform == "gpu"
    except RuntimeError:
        return False


def _case(m=64, seed=0):
    rng = np.random.default_rng(seed)
    pstw = jnp.asarray(rng.normal(size=(ROWS_W, H)).astype(np.float32) * 0.1)
    thw = jnp.asarray(rng.normal(size=(ROWS_T, H)).astype(np.float32) * 0.1)
    pstb = jnp.asarray(rng.normal(size=(H,)).astype(np.float32) * 0.1)

    def mk(rows, s, lo, hi):
        n = rng.integers(lo, hi + 1, size=m)
        idx = np.full((m, s), -1, np.int32)
        for r in range(m):  # front-packed, like features.py emits
            idx[r, :n[r]] = rng.integers(0, rows, size=n[r])
        return jnp.asarray(idx)

    return pstw, thw, pstb, mk(ROWS_W, S_W, 2, S_W), mk(ROWS_T, S_T, 0, S_T)


def _reference(pstw, thw, pstb, hk, th):
    params = {"l0_pstw": pstw, "l0_threatw": thw, "l0_pstb": pstb}
    return _ft_accumulate(params, hk, th)


def _check_impl(fn, atol_fwd=1e-5, atol_bwd=1e-4):
    pstw, thw, pstb, hk, th = _case()
    ref = _reference(pstw, thw, pstb, hk, th)
    out = jax.jit(fn)(pstw, thw, pstb, hk, th)
    assert float(jnp.max(jnp.abs(out - ref))) < atol_fwd

    cot = jnp.asarray(
        np.random.default_rng(1).normal(size=ref.shape).astype(np.float32))

    def loss(f):
        return lambda a, b, c: jnp.vdot(f(a, b, c, hk, th), cot)

    g_ref = jax.grad(loss(_reference), argnums=(0, 1, 2))(pstw, thw, pstb)
    g_new = jax.jit(jax.grad(loss(fn), argnums=(0, 1, 2)))(pstw, thw, pstb)
    for r, n in zip(g_ref, g_new):
        # atomics reorder the adds -> float-noise differences only
        assert float(jnp.max(jnp.abs(r - n))) < atol_bwd


def test_scan_matches_reference():
    _check_impl(ft_accumulate_scan)


@pytest.mark.skipif(not _gpu_available(),
                    reason="needs CUDA + ARCHON_JAX_ALLOW_GPU=1")
def test_pallas_matches_reference():
    _check_impl(ft_accumulate_pallas)


@pytest.mark.skipif(not _gpu_available(),
                    reason="needs CUDA + ARCHON_JAX_ALLOW_GPU=1")
def test_pallas_empty_bags():
    """All-padding rows (edge: nb == 0 loops) must yield exactly pstb."""
    pstw, thw, pstb, hk, th = _case()
    hk = jnp.full_like(hk, -1)
    th = jnp.full_like(th, -1)
    out = jax.jit(ft_accumulate_pallas)(pstw, thw, pstb, hk, th)
    assert float(jnp.max(jnp.abs(out - pstb[None, :]))) == 0.0
