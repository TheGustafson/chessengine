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

from .spec import POLICY_MASK_ILLEGAL, ArchSpec, HALFKA_DIMS, THREAT_DIMS, POLICY_VOCAB, PS_NB


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
        "polmovew": np.sqrt(2.0 / (arch.pol_proj or arch.l2)),
        "pol_projw": np.sqrt(2.0 / arch.hl),
        "dis_projw": np.sqrt(2.0 / arch.hl),
        "dis_outw": np.sqrt(2.0 / max(arch.dis_proj, 1)),
    }
    for name, shape in sorted(shapes.items()):
        key, sub = jax.random.split(key)
        if name in stdevs:
            params[name] = stdevs[name] * jax.random.normal(sub, shape, jnp.float32)
        else:  # biases and l0_threatw: zero (InitSettings::Zeroed)
            params[name] = jnp.zeros(shape, jnp.float32)
    return params


def effective_pstw(params: dict, arch: ArchSpec) -> jnp.ndarray:
    """The HalfKA table the FT actually gathers from. With T0 factorization
    the real table and the PS_NB-wide bucket-agnostic factor table are summed
    (merged[f] = real[f] + factor[f % PS_NB], the merge_factoriser contract);
    tiling reproduces f % PS_NB exactly because the HalfKA layout is
    king-bucket-major. Zero factor => identical to l0_pstw."""
    if not arch.factorize:
        return params["l0_pstw"]
    reps = HALFKA_DIMS // PS_NB
    return params["l0_pstw"] + jnp.tile(params["l0_factw"], (reps, 1))


def _masked_gather_sum(table: jnp.ndarray, idx: jnp.ndarray) -> jnp.ndarray:
    """Σ table[idx] over the -1-padded active-index axis."""
    valid = (idx >= 0)[..., None]
    rows = jnp.take(table, jnp.maximum(idx, 0), axis=0)
    return jnp.sum(rows * valid, axis=-2)


def _psqt_side_sums(params: dict, arch: ArchSpec, hk_idx, th_idx):
    """One perspective's raw PSQT bucket sums (B, num_buckets), XLA path."""
    s = _masked_gather_sum(params["l0_psqt_hkw"], hk_idx)
    if th_idx is not None and not arch.halfka_only:
        s = s + _masked_gather_sum(params["l0_psqt_thw"], th_idx)
    return s


def psqt_combine(psqt_stm: jnp.ndarray, psqt_ntm: jnp.ndarray,
                 bucket: jnp.ndarray) -> jnp.ndarray:
    """T6c skip: 0.5 * (psqt_stm - psqt_ntm)[bucket] — added to the value
    logit OUTSIDE every nonlinearity (stm-canonical frame => SF's (us-0.5)
    is the constant +0.5)."""
    diff = psqt_stm - psqt_ntm
    return 0.5 * jnp.take_along_axis(diff, bucket[:, None], axis=1)[:, 0]


def _ft_accumulate(params: dict, hk_idx: jnp.ndarray, th_idx: jnp.ndarray | None):
    """acc = l0_pstb + Σ l0_pstw[active_halfka] + Σ l0_threatw[active_threat].

    Sparse indices are -1-padded (the bullet sentinel, d40:872-875); padding
    rows are gathered at index 0 and masked to zero.

    This is the REFERENCE path (and the only one exercised by default): XLA
    materializes the B x S x H gather, which is fine on CPU / small batches
    but OOMs the GPU at batch >= 2048 (6.7 GB for threats at 8192). GPU
    training selects ``ft_impl="pallas"`` in ``forward`` (see ft_kernel.py).
    """
    pstw = params.get("_pstw_eff", params["l0_pstw"])
    hk_valid = (hk_idx >= 0)[..., None]
    hk_rows = jnp.take(pstw, jnp.maximum(hk_idx, 0), axis=0)
    acc = params["l0_pstb"] + jnp.sum(hk_rows * hk_valid, axis=-2)
    if th_idx is not None:
        th_valid = (th_idx >= 0)[..., None]
        th_rows = jnp.take(params["l0_threatw"], jnp.maximum(th_idx, 0), axis=0)
        acc = acc + jnp.sum(th_rows * th_valid, axis=-2)
    return acc


def _ft_accumulate_pair(params: dict, batch: dict, arch: ArchSpec, ft_impl: str):
    """Both perspectives' accumulators under the selected FT implementation.

    ``ft_impl``: "reference" (default everywhere; the proven CPU path),
    "pallas" (GPU embedding-bag kernel, ft_kernel.py) or "scan" (pure-XLA
    slot scan, the measured fallback). The GPU impls stack stm+ntm along the
    batch axis: one launch, shared grad buffers in the backward.
    """
    th_stm = batch.get("th_stm") if not arch.halfka_only else None
    th_ntm = batch.get("th_ntm") if not arch.halfka_only else None
    pstw = effective_pstw(params, arch)
    if ft_impl != "reference" and th_stm is not None:
        from .ft_kernel import ft_accumulate_pallas, ft_accumulate_scan

        fn = ft_accumulate_pallas if ft_impl == "pallas" else ft_accumulate_scan
        b = batch["hk_stm"].shape[0]
        hk = jnp.concatenate([batch["hk_stm"], batch["hk_ntm"]], axis=0)
        th = jnp.concatenate([th_stm, th_ntm], axis=0)
        acc = fn(pstw, params["l0_threatw"], params["l0_pstb"], hk, th)
        if arch.psqt:
            # T6c via a SECOND kernel invocation on the 8-wide psqt tables —
            # the kernel is generic over width, and this dodges XLA's slow
            # narrow scatter (measured -32% step throughput on the XLA path).
            ps = fn(params["l0_psqt_hkw"], params["l0_psqt_thw"],
                    jnp.zeros((arch.num_buckets,), jnp.float32), hk, th)
            return acc[:b], acc[b:], ps[:b], ps[b:]
        return acc[:b], acc[b:], None, None
    # reference (also the halfka_only fallback: GPU impls are two-table only)
    p = {**params, "_pstw_eff": pstw}
    acc_pair = (_ft_accumulate(p, batch["hk_stm"], th_stm),
                _ft_accumulate(p, batch["hk_ntm"], th_ntm))
    if arch.psqt:
        return (*acc_pair,
                _psqt_side_sums(params, arch, batch["hk_stm"], th_stm),
                _psqt_side_sums(params, arch, batch["hk_ntm"], th_ntm))
    return (*acc_pair, None, None)


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
                 pol_mask: jnp.ndarray | None = None,
                 pol_legal: jnp.ndarray | None = None):
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
    if arch.out_skip:
        # D47U output-skip: per-bucket linear lane on hl1, outside every
        # nonlinearity (zero-init => bit-exact no-op at leg start).
        sk = hl1 @ params["skipw"]                                # (B, 8)
        out = out + jnp.take_along_axis(sk, bucket[:, None], axis=1)[:, 0]

    dis_pred = None
    if arch.dis_proj:
        # PURE PROBE (2026-07-27 ruling): stop_gradient on the trunk input —
        # the disagreement head observes the substrate, never steers it. The
        # face-off's value/policy results stay uncontaminated; only
        # dis_projw/dis_outw train from the dis loss.
        h_in = jax.lax.stop_gradient(hl1)
        dis_h = _crelu(h_in @ params["dis_projw"] + params["dis_projb"])
        dis_pred = dis_h @ params["dis_outw"] + params["dis_outb"][0]  # (B,)
    pol_logits = None
    if arch.policy:
        if arch.pol_proj:
            # D48W: dedicated projection off the accumulator (hl1), not hl2.
            pol_h = _crelu(hl1 @ params["pol_projw"] + params["pol_projb"])
            pol_logits = pol_h @ params["polmovew"] + params["polmoveb"]
        else:
            pol_logits = hl2 @ params["polmovew"] + params["polmoveb"]  # (B, 1858)
        if pol_legal is not None:
            # Legal-move IDS -> the SAME dense additive mask, scattered HERE on
            # device (2026-07-26). The host-side (N,1858) f32 mask was 61 MB per
            # 8192-batch and ~25% of trainer throughput; ids are (N,128) i16 =
            # 2.1 MB, 29x less, and 24x faster to build. Verified to reproduce
            # the dense mask EXACTLY, so the loss is bit-identical.
            # Rows with no packed move are all -1 -> all-zero mask (zero CE
            # either way), matching legal_move_masks.
            # Scatter into VOCAB+1 columns and drop the last: padded (-1)
            # entries go to that DUMMY column instead of colliding with real
            # ones. (First draft clipped -1 to column 0 and wrote ILLEGAL
            # there, clobbering a genuinely legal move id 0 — caught by the
            # bit-parity test, max|diff| was exactly 30.0.)
            n, k = pol_legal.shape
            rows = jnp.repeat(jnp.arange(n), k)
            flat = pol_legal.reshape(-1).astype(jnp.int32)
            cols = jnp.where(flat >= 0, flat, POLICY_VOCAB)
            dense = jnp.full((n, POLICY_VOCAB + 1), POLICY_MASK_ILLEGAL, jnp.float32)
            dense = dense.at[rows, cols].set(0.0)[:, :POLICY_VOCAB]
            allpad = (pol_legal < 0).all(axis=1)[:, None]
            dense = jnp.where(allpad, 0.0, dense)
            pol_logits = pol_logits + dense
        elif pol_mask is not None:
            pol_logits = pol_logits + pol_mask
    return out, pol_logits, hl2, dis_pred


def _pairwise(c: jnp.ndarray, hl: int) -> jnp.ndarray:
    return c[..., : hl // 2] * c[..., hl // 2:]


def forward(params: dict, arch: ArchSpec, batch: dict,
            ft_impl: str = "reference", with_acc: bool = False,
            with_dis: bool = False):
    """Full graph: sparse features -> (value_logit, pol_logits).

    ``ft_impl`` selects the feature-transformer implementation (see
    ``_ft_accumulate_pair``); everything downstream is identical. The default
    stays "reference" — the GPU path is opt-in (train.make_update_step wires
    ARCHON_JAX_FT_IMPL through; the head math never forks).
    ``with_acc=True`` appends the raw accumulator pair to the return (the
    FT-sparsity regularizer reads it; plain callers are unchanged).
    ``with_dis=True`` appends the disagreement-head prediction (D48W).
    """
    acc_stm, acc_ntm, psqt_stm, psqt_ntm = _ft_accumulate_pair(
        params, batch, arch, ft_impl)
    out, pol_logits, hl2, dis_pred = head_forward(
        params, arch, acc_stm, acc_ntm,
        batch["bucket"], batch.get("pol_mask"), batch.get("pol_legal"))
    if arch.psqt:
        out = out + psqt_combine(psqt_stm, psqt_ntm, batch["bucket"])
    extras = []
    if with_dis:
        extras.append(dis_pred)
    if with_acc:
        extras.append((acc_stm, acc_ntm))
    if extras:
        return (out, pol_logits, hl2, *extras)
    return out, pol_logits, hl2


@partial(jax.jit, static_argnames=("arch", "ft_impl"))
def forward_jit(params: dict, arch: ArchSpec, batch: dict,
                ft_impl: str = "reference"):
    return forward(params, arch, batch, ft_impl)[:2]
