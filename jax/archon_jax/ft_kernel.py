"""GPU feature-transformer kernels: the sparse embedding-bag that XLA can't do.

The D47c input layer sums ~14-200 active rows (of 83,248) from two FT tables
into a 1280-wide accumulator, per perspective. The reference path in model.py
(`jnp.take` -> mask -> `sum`) is what XLA materializes as a B x S x H f32
intermediate: 6.7 GB for threats alone at batch 8192 -> OOM at batch >= 2048,
and 0.05x bullet throughput at batch 1024 (measured 2026-07-23,
tools/gpu_step_bench.py). This module provides two alternatives:

- ``ft_accumulate_pallas`` — a Pallas-Triton embedding-bag with a custom VJP:
  forward accumulates rows in registers (nothing materialized), backward
  scatter-adds into dense grad buffers with f32 atomics (the same shape the
  optimiser consumes; bullet's CUDA backward is the same construction).
  Indices are front-packed with -1 padding (features.py fills slots 0..n-1),
  so both kernels run per-bag *dynamic* trip counts and never touch padding.
- ``ft_accumulate_scan`` — a pure-XLA `lax.scan` over slots (one (B, H) gather
  per slot instead of one (B, S, H) monolith). No custom code, works anywhere,
  but the backward lowers to S sequential scatter-adds; kept as the measured
  route-2 baseline and as a portability fallback.

Both compute EXACTLY the same function as ``model._ft_accumulate`` (the CPU
reference, byte-faithful to production); float-order differences only.
Numerical-parity + grad-parity tests: ``jax/tests/test_ft_kernel.py``;
measured throughput: ``docs/research/jax-gpu-kernel-2026-07-23.md``.

The Triton backend whitelists data-center device kinds; consumer cards (the
RTX 4060 Ti, Ada sm_89) are not on it even though the arch is identical to
the whitelisted L4/RTX 4090. ``register_gpu()`` fills the public registry
hook from the device's own reported compute capability. jax is PINNED
(0.11.0, jax/pyproject.toml); revisit the private-module import on bumps.

Tuning knobs (defaults measured best on the 4060 Ti, see the research doc;
the end-to-end spread across {64,128,256} x {2,4} was only ~2%):
  ARCHON_JAX_FT_BLOCK  column-block width, must divide HL   (default 128)
  ARCHON_JAX_FT_WARPS  Triton num_warps per program         (default 2)
"""

from __future__ import annotations

import os
from functools import partial

import jax
import jax.numpy as jnp
import numpy as np
from jax.experimental import pallas as pl
from jax.experimental.pallas import triton as plt

_FT_BLOCK = int(os.environ.get("ARCHON_JAX_FT_BLOCK", "128"))
_FT_WARPS = int(os.environ.get("ARCHON_JAX_FT_WARPS", "2"))
_INTERPRET = os.environ.get("ARCHON_JAX_FT_INTERPRET") == "1"  # CPU tests

_registered = False


def register_gpu() -> None:
    """Register the current CUDA device with the Pallas-Triton backend if its
    device kind is not on the built-in whitelist (consumer GPUs)."""
    global _registered
    if _registered:
        return
    _registered = True
    dev = jax.devices()[0]
    cc = getattr(dev, "compute_capability", None)
    if dev.platform != "gpu" or not cc:
        return
    from jax._src.pallas.triton import gpu_info as _gi  # pinned jax 0.11.0

    if dev.device_kind not in _gi.registry:
        _gi.registry[dev.device_kind] = lambda: _gi.GpuInfo(
            gpu_version=None, arch_name=cc,
            compute_capability=int(cc.replace(".", "")))


def _block(h: int) -> int:
    bh = min(_FT_BLOCK, h)
    while h % bh:  # need an exact divisor; fall back to smaller powers of two
        bh //= 2
    return bh


# ---------------------------------------------------------------------------
# Forward kernel: out[b] = pstb + sum_s pstw[hk[b,s]] + sum_s thw[th[b,s]]
# grid = (H/bh, M); batch is the LAST grid dim so consecutive programs share a
# column block (hot table rows hit L2). Trip counts na/nb skip the -1 padding.
# ---------------------------------------------------------------------------

def _fwd_kernel(hk_ref, th_ref, hkc_ref, thc_ref, pstw_ref, thw_ref, pstb_ref,
                out_ref, *, bh: int):
    cb = pl.program_id(0)
    b = pl.program_id(1)
    col = pl.dslice(cb * bh, bh)
    acc = pstb_ref[col]
    na = hkc_ref[b]
    nb = thc_ref[b]

    def body_hk(s, acc):
        return acc + pstw_ref[hk_ref[b, s], col]

    acc = jax.lax.fori_loop(0, na, body_hk, acc)

    def body_th(s, acc):
        return acc + thw_ref[th_ref[b, s], col]

    acc = jax.lax.fori_loop(0, nb, body_th, acc)
    out_ref[b, col] = acc


def _bag2_fwd(pstw, thw, pstb, hk, th):
    register_gpu()
    m, h = hk.shape[0], pstw.shape[1]
    bh = _block(h)
    hkc = jnp.sum(hk >= 0, axis=1, dtype=jnp.int32)
    thc = jnp.sum(th >= 0, axis=1, dtype=jnp.int32)
    return pl.pallas_call(
        partial(_fwd_kernel, bh=bh),
        grid=(h // bh, m),
        out_shape=jax.ShapeDtypeStruct((m, h), pstw.dtype),
        compiler_params=plt.CompilerParams(num_warps=_FT_WARPS),
        interpret=_INTERPRET,
    )(hk, th, hkc, thc, pstw, thw, pstb)


# ---------------------------------------------------------------------------
# Backward kernel: dtable[idx[b,s]] += dacc[b] (f32 atomics into the zeroed
# dense grad buffers the optimiser consumes anyway). Nondeterministic add
# order — differences are ~1 ulp, same property as bullet's CUDA backward.
# ---------------------------------------------------------------------------

def _bwd_kernel(hk_ref, th_ref, hkc_ref, thc_ref, dacc_ref, zw_ref, zt_ref,
                dpstw_ref, dthw_ref, *, bh: int):
    del zw_ref, zt_ref  # aliased zero-initialized outputs
    cb = pl.program_id(0)
    b = pl.program_id(1)
    col = pl.dslice(cb * bh, bh)
    g = dacc_ref[b, col]
    na = hkc_ref[b]
    nb = thc_ref[b]
    zero = jnp.minimum(na, jnp.int32(0))  # traced zero: scf.yield needs a Value

    def body_hk(s, c):
        plt.atomic_add(dpstw_ref, (hk_ref[b, s], col), g)
        return c

    jax.lax.fori_loop(0, na, body_hk, zero)

    def body_th(s, c):
        plt.atomic_add(dthw_ref, (th_ref[b, s], col), g)
        return c

    jax.lax.fori_loop(0, nb, body_th, zero)


def _bag2_bwd_tables(hk, th, hkc, thc, dacc, rows_w: int, rows_t: int):
    m, h = dacc.shape
    bh = _block(h)
    zw = jnp.zeros((rows_w, h), dacc.dtype)
    zt = jnp.zeros((rows_t, h), dacc.dtype)
    return pl.pallas_call(
        partial(_bwd_kernel, bh=bh),
        grid=(h // bh, m),
        out_shape=[jax.ShapeDtypeStruct((rows_w, h), dacc.dtype),
                   jax.ShapeDtypeStruct((rows_t, h), dacc.dtype)],
        input_output_aliases={5: 0, 6: 1},
        compiler_params=plt.CompilerParams(num_warps=_FT_WARPS),
        interpret=_INTERPRET,
    )(hk, th, hkc, thc, dacc, zw, zt)


# ---------------------------------------------------------------------------
# custom_vjp wiring
# ---------------------------------------------------------------------------

@jax.custom_vjp
def ft_accumulate_pallas(pstw, thw, pstb, hk, th):
    """acc[b] = pstb + Σ pstw[hk[b]>=0] + Σ thw[th[b]>=0]  — (M, H) f32.

    ``hk`` (M, 32) / ``th`` (M, 160) int32, front-packed, -1 padded. Callers
    stack the stm and ntm perspectives along the batch axis (M = 2B) so one
    kernel launch serves both and the backward scatters into shared buffers.
    """
    return _bag2_fwd(pstw, thw, pstb, hk, th)


def _ft_fwd_rule(pstw, thw, pstb, hk, th):
    out = _bag2_fwd(pstw, thw, pstb, hk, th)
    hkc = jnp.sum(hk >= 0, axis=1, dtype=jnp.int32)
    thc = jnp.sum(th >= 0, axis=1, dtype=jnp.int32)
    return out, (hk, th, hkc, thc, pstw.shape[0], thw.shape[0])


def _ft_bwd_rule(res, dacc):
    hk, th, hkc, thc, rows_w, rows_t = res
    dpstw, dthw = _bag2_bwd_tables(hk, th, hkc, thc, dacc, rows_w, rows_t)
    dpstb = jnp.sum(dacc, axis=0)
    zeros_hk = np.zeros(hk.shape, jax.dtypes.float0)  # int inputs: float0
    zeros_th = np.zeros(th.shape, jax.dtypes.float0)
    return dpstw, dthw, dpstb, zeros_hk, zeros_th


ft_accumulate_pallas.defvjp(_ft_fwd_rule, _ft_bwd_rule)


# ---------------------------------------------------------------------------
# Route 2: pure-XLA scan over slots (no custom code, portable, slower).
# One (M, H) gather+add per slot; autodiff gives S sequential scatter-adds.
# ---------------------------------------------------------------------------

def ft_accumulate_scan(pstw, thw, pstb, hk, th):
    def bag(table, idx):
        def body(acc, idx_s):
            rows = jnp.take(table, jnp.maximum(idx_s, 0), axis=0)
            return acc + jnp.where((idx_s >= 0)[:, None], rows, 0.0), None

        acc0 = jnp.zeros((idx.shape[0], table.shape[1]), table.dtype)
        acc, _ = jax.lax.scan(body, acc0, idx.T)
        return acc

    return pstb + bag(pstw, hk) + bag(thw, th)
