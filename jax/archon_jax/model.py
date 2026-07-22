"""The D47c graph in JAX — a faithful transcription of the bullet
ModelDefinition in d40_features.rs lines 948-1119.

Design choice: **pure functional JAX over a flat params dict**, not flax.linen
or equinox. Justification:

1. The params dict keys ARE the bullet tensor names, so checkpoint load/export
   is an identity mapping (no name translation layer to drift — the class of
   bug that byte-parity proved absent). flax/equinox both interpose their own
   module trees + param naming.
2. optax works on any pytree, so the whole optimiser/schedule ecosystem is
   available regardless (see optim.py).
3. Arch exploration = editing one ~40-line pure function with visible shapes,
   which is the entire point of the migration. A module framework adds
   init/apply indirection and buys us nothing at this model size.
If a team later wants flax, wrapping these functions is trivial; the reverse
migration is not.

Semantics pinned against the bullet compiler sources:
  - crelu   = clamp(x, 0, 1)            (pointwise.rs CReLU, lines 76-97)
  - screlu  = clamp(x, 0, 1)^2          (pointwise.rs SCReLU, lines 99-118)
  - pairwise_mul = first_half * second_half (builder.rs:390-393:
        slice_rows(0, n/2) * slice_rows(n/2, n))  ← ordering landmine
  - concat  = [stm_hidden; ntm_hidden]  (d40:979, stm first)
  - select  = contiguous chunk b of size rows/divisor (index.rs SelectRows)
  - the two-FT sum: HalfKA affine (with THE bias) + bias-free threat matmul
    (d40:963-973; the threat tensor is zero-init and has NO bias — d40:951-956)
  - policy head reads hl2 = the bucket-selected, POST-screlu L2 bottleneck
    (d40:1106-1107), logits = affine(hl2) + additive mask
"""

from __future__ import annotations

from functools import partial

import jax
import jax.numpy as jnp
import numpy as np

from .spec import ArchSpec, HALFKA_DIMS, THREAT_DIMS, POLICY_VOCAB


def init_params(arch: ArchSpec, seed: int) -> dict:
    """Match bullet's InitSettings (builder.rs:104-119, weights.rs:52-74):
    affine weights ~ Normal(0, sqrt(2/fan_in)), biases zero, threat FT zero.
    l0_pst uses effective input size = 32 active features, NOT 22528
    (init_with_effective_input_size, d40:964) → stdev = sqrt(2/32) = 0.25.

    NOTE: distribution-parity only. Bullet draws from Xoroshiro128Plus in
    BTreeMap tensor order; bit-identical init reproduction is possible but
    deliberately out of scope (resume/export parity does not need it).
    """
    key = jax.random.PRNGKey(seed)
    shapes = arch.param_shapes()
    params = {}
    stdevs = {
        "l0_pstw": np.sqrt(2.0 / 32.0),
        "l1w": np.sqrt(2.0 / arch.hl),
        "l2w": np.sqrt(2.0 / arch.l2),
        "l3w": np.sqrt(2.0 / arch.l3),
        "polmovew": np.sqrt(2.0 / arch.l2),
    }
    for name, shape in sorted(shapes.items()):
        key, sub = jax.random.split(key)
        if name in stdevs:
            params[name] = stdevs[name] * jax.random.normal(sub, shape, jnp.float32)
        else:  # biases and l0_threatw: zero (InitSettings::Zeroed)
            params[name] = jnp.zeros(shape, jnp.float32)
    return params


def _ft_accumulate(params: dict, hk_idx: jnp.ndarray, th_idx: jnp.ndarray | None):
    """acc = l0_pstb + Σ l0_pstw[active_halfka] + Σ l0_threatw[active_threat].

    Sparse indices are -1-padded (the bullet sentinel, d40:872-875); padding
    rows are gathered at index 0 and masked to zero.
    """
    hk_valid = (hk_idx >= 0)[..., None]
    hk_rows = jnp.take(params["l0_pstw"], jnp.maximum(hk_idx, 0), axis=0)
    acc = params["l0_pstb"] + jnp.sum(hk_rows * hk_valid, axis=-2)
    if th_idx is not None:
        th_valid = (th_idx >= 0)[..., None]
        th_rows = jnp.take(params["l0_threatw"], jnp.maximum(th_idx, 0), axis=0)
        acc = acc + jnp.sum(th_rows * th_valid, axis=-2)
    return acc


def _crelu(x):
    return jnp.clip(x, 0.0, 1.0)


def _screlu(x):
    c = jnp.clip(x, 0.0, 1.0)
    return c * c


def _select(x: jnp.ndarray, bucket: jnp.ndarray, divisor: int) -> jnp.ndarray:
    """SelectRows: pick contiguous chunk `bucket` of size rows/divisor."""
    b, rows = x.shape
    chunk = rows // divisor
    x = x.reshape(b, divisor, chunk)
    return jnp.take_along_axis(x, bucket[:, None, None], axis=1)[:, 0, :]


def head_forward(params: dict, arch: ArchSpec, acc_stm: jnp.ndarray,
                 acc_ntm: jnp.ndarray, bucket: jnp.ndarray,
                 pol_mask: jnp.ndarray | None = None):
    """Everything after the accumulators (shared by training + parity probes).

    Returns (value_logit (B,), pol_logits (B, 1858) or None, hl2 (B, L2)).
    """
    stm_hidden = _pairwise(_crelu(acc_stm), arch.hl)
    ntm_hidden = _pairwise(_crelu(acc_ntm), arch.hl)
    hl1 = jnp.concatenate([stm_hidden, ntm_hidden], axis=-1)      # (B, HL)

    l1 = hl1 @ params["l1w"] + params["l1b"]                      # (B, 8*L2)
    hl2 = _screlu(_select(l1, bucket, arch.num_buckets))          # (B, L2)

    l2 = hl2 @ params["l2w"] + params["l2b"]                      # (B, 8*L3)
    hl3 = _screlu(_select(l2, bucket, arch.num_buckets))          # (B, L3)

    l3 = hl3 @ params["l3w"] + params["l3b"]                      # (B, 8)
    out = _select(l3, bucket, arch.num_buckets)[:, 0]             # (B,)

    pol_logits = None
    if arch.policy:
        pol_logits = hl2 @ params["polmovew"] + params["polmoveb"]  # (B, 1858)
        if pol_mask is not None:
            pol_logits = pol_logits + pol_mask
    return out, pol_logits, hl2


def _pairwise(c: jnp.ndarray, hl: int) -> jnp.ndarray:
    return c[..., : hl // 2] * c[..., hl // 2:]


def forward(params: dict, arch: ArchSpec, batch: dict):
    """Full graph: sparse features -> (value_logit, pol_logits)."""
    th_stm = batch.get("th_stm") if not arch.halfka_only else None
    th_ntm = batch.get("th_ntm") if not arch.halfka_only else None
    acc_stm = _ft_accumulate(params, batch["hk_stm"], th_stm)
    acc_ntm = _ft_accumulate(params, batch["hk_ntm"], th_ntm)
    return head_forward(params, arch, acc_stm, acc_ntm, batch["bucket"],
                        batch.get("pol_mask"))


@partial(jax.jit, static_argnames=("arch",))
def forward_jit(params: dict, arch: ArchSpec, batch: dict):
    return forward(params, arch, batch)[:2]
