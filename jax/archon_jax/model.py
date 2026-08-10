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
        # fan_in from the SHAPE, not from arch.l2/l3 — identical without
        # dual_tail, correct with it (fan-ins 2*L2 and 2*L2+2*L3).
        "l2w": np.sqrt(2.0 / shapes["l2w"][0]),
        "l3w": np.sqrt(2.0 / shapes["l3w"][0]),
        "polmovew": np.sqrt(2.0 / (arch.pol_proj
                                   or shapes.get("polmovew", (arch.l2,))[0])),
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


def fake_quant(w: jnp.ndarray, scale: float, qmax: float) -> jnp.ndarray:
    """Straight-through fake quantisation: forward returns the value the ENGINE
    will actually use, backward passes gradient through as identity.

    Why (2026-07-29). We train in float and quantise afterwards, so the loss
    never sees quantisation damage — and it is not small. Measured on
    d48y2-sb190: the threat channel is 72.9% of the input space and carries
    **86% of all accumulator quantisation noise** (5.58% relative error, 25.1 dB
    SNR, against HalfKA's 0.51% / 45.8 dB). Worse, the scale is not tuned: the
    exporter RAISES on overflow rather than clipping, our max |threat w| is
    1.978, and 127/1.978 = 64.2 — so **one outlier out of 77.7 M weights sets
    the precision for all of them**.

    Post-hoc rescaling can only trade resolution against clipping on a
    distribution that was never chosen with quantisation in mind (measured
    optimum: scale 128, 1.65x better than 64; Stockfish's 255 is *worse* for our
    weights at 0.82x, because their trainer produces a distribution that fits
    +/-0.5 and ours does not). QAT removes that trade: the network learns to
    place mass where the levels are, which is how Stockfish uses a 255-frame
    productively (`nnue-training-gap-study-2026-07-28.md` §2.5).

    STE identity: `w + stop_gradient(q(w) - w)` is exactly `q(w)` in the forward
    pass and exactly `w` in the backward, which is what lets gradients reach
    weights whose quantised value did not move.
    """
    q = jnp.clip(jnp.round(w * scale), -qmax, qmax) / scale
    return w + jax.lax.stop_gradient(q - w)


def _ft_accumulate_pair(params: dict, batch: dict, arch: ArchSpec, ft_impl: str):
    """Both perspectives' accumulators under the selected FT implementation.

    ``ft_impl``: "reference" (default everywhere; the proven CPU path),
    "pallas" (GPU embedding-bag kernel, ft_kernel.py) or "scan" (pure-XLA
    slot scan, the measured fallback). The GPU impls stack stm+ntm along the
    batch axis: one launch, shared grad buffers in the backward.

    QAT (``arch.qat_threat_scale`` / ``arch.qat_ft_scale``, 0 = off) is applied
    HERE, to the effective tables, so BOTH the reference and the Pallas paths
    get quantised weights from one site and cannot drift apart.
    """
    th_stm = batch.get("th_stm") if not arch.halfka_only else None
    th_ntm = batch.get("th_ntm") if not arch.halfka_only else None
    pstw = effective_pstw(params, arch)
    # QAT: quantise the FOLDED HalfKA table, not l0_pstw — the factoriser is
    # merged at export, so the deployed weight is the sum and that is what must
    # be made representable.
    if arch.qat_ft_scale > 0.0:
        pstw = fake_quant(pstw, arch.qat_ft_scale, 32767.0)   # i16
    if arch.qat_threat_scale > 0.0 and not arch.halfka_only:
        params = {**params,
                  "l0_threatw": fake_quant(params["l0_threatw"],
                                           arch.qat_threat_scale, 127.0)}  # i8
    if ft_impl != "reference" and (th_stm is not None or arch.halfka_only):
        from .ft_kernel import ft_accumulate_pallas, ft_accumulate_scan

        fn = ft_accumulate_pallas if ft_impl == "pallas" else ft_accumulate_scan
        b = batch["hk_stm"].shape[0]
        hk = jnp.concatenate([batch["hk_stm"], batch["hk_ntm"]], axis=0)
        if arch.halfka_only:
            # The bag kernel is two-table by construction, and dropping to
            # the reference path for a HalfKA-only net would cost ~24x
            # throughput (9.5k pos/s against Pallas's 225k) — which would
            # make the whole no-threats experiment unaffordable.
            #
            # Instead hand it a single all-zero threat row that nothing ever
            # indexes. The kernel counts valid slots per row
            # (`nb = thc_ref[b]`) and loops `fori_loop(0, nb, ...)`, so an
            # all -1 index array makes the threat loop run ZERO iterations:
            # no work, no contribution, and a gradient w.r.t. that row that
            # is identically zero. Nothing is hardcoded to 160 slots — the
            # kernel takes its dimensions from the shapes it is given.
            thw = jnp.zeros((1, pstw.shape[1]), pstw.dtype)
            th = jnp.full((hk.shape[0], 1), -1, jnp.int32)
        else:
            thw = params["l0_threatw"]
            th = jnp.concatenate([th_stm, th_ntm], axis=0)
        acc = fn(pstw, thw, params["l0_pstb"], hk, th)
        if arch.psqt:
            # T6c via a SECOND kernel invocation on the 8-wide psqt tables —
            # the kernel is generic over width, and this dodges XLA's slow
            # narrow scatter (measured -32% step throughput on the XLA path).
            ps_thw = (jnp.zeros((1, arch.num_buckets), jnp.float32)
                      if arch.halfka_only else params["l0_psqt_thw"])
            ps = fn(params["l0_psqt_hkw"], ps_thw,
                    jnp.zeros((arch.num_buckets,), jnp.float32), hk, th)
            return acc[:b], acc[b:], ps[:b], ps[b:]
        return acc[:b], acc[b:], None, None
    # reference path (and the CPU fallback for halfka_only)
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


# SF's square lane, normalised. Their int form is `(v*v) >> (2*(WSB+1)+7)`
# capped at 127, against a clipped lane of `(v >> (WSB+1))` clamped to
# [0, 127]. Writing x = v / (2^(WSB+1) * 127) makes the clipped lane exactly
# clamp(x, 0, 1); the same substitution turns the square lane into
# min(x^2 * 127/128, 1), at BOTH layers (the WSB+1 / WSB pair cancels).
SF_SQR_GAIN = 127.0 / 128.0


def _dual(x, sf_act: bool = False):
    """SF `eca43a97`'s dual activation: hand BOTH SqrCReLU(x) and CReLU(x)
    forward, concatenated (nnue_architecture.h:56-67 — ac_sqr_N writes the
    low half of fc_{N+1}'s input buffer, ac_N the high half, so sqr comes
    FIRST). Doubles what each unit contributes at one extra multiply per
    unit; adds routes rather than width (study §4.3/§4.4).

    ``sf_act=False`` is SHAPE ONLY — SF's `WeightScaleBits+1` /
    `WeightScaleBits` shifts are int8 requantisation bookkeeping and our tail
    is f32. ``sf_act=True`` is the exact float analogue of their integer
    activation, needed only to REPRODUCE their net as a teacher: the square
    lane reads the UNCLIPPED pre-activation (so negative units contribute
    positively) and carries the 127/128 gain. See ArchSpec.sf_act.
    """
    c = jnp.clip(x, 0.0, 1.0)
    if sf_act:
        return jnp.concatenate([jnp.minimum(SF_SQR_GAIN * x * x, 1.0), c],
                               axis=-1)
    return jnp.concatenate([c * c, c], axis=-1)


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
    # QAT, activation side: the engine truncates the pairwise product to u8
    # (>> FT_SHIFT leaves 127 levels). This is the single largest unmodelled
    # quantisation step in the whole forward pass — everything downstream of
    # L1's dequant is f32 and needs no emulation.
    if arch.qat_act_levels > 0.0:
        hl1 = fake_quant_act(hl1, arch.qat_act_levels)

    l1w = params["l1w"]
    if arch.qat_l1_scale > 0.0:
        l1w = fake_quant(l1w, arch.qat_l1_scale, 127.0)           # i8 @ L1_QUANT
    l1 = hl1 @ l1w + params["l1b"]                                # (B, 8*L2)
    if arch.dual_tail:
        # R2: dual activation + inter-layer concat skip (study §4.4). L1/L2
        # unchanged, so `l1` above — the expensive part — is untouched.
        sfa = getattr(arch, "sf_act", False)
        hl2 = _dual(_select(l1, bucket, arch.num_buckets), sfa)   # (B, 2*L2)
        l2 = hl2 @ params["l2w"] + params["l2b"]                  # (B, 8*L3)
        hl3 = _dual(_select(l2, bucket, arch.num_buckets), sfa)   # (B, 2*L3)
        # fc_2 reads BOTH layers' pairs: a neck direction reaches the output
        # without passing through fc_1's nonlinearity.
        l3 = jnp.concatenate([hl2, hl3], axis=-1) @ params["l3w"] + params["l3b"]
    else:
        hl2 = _screlu(_select(l1, bucket, arch.num_buckets))      # (B, L2)

        l2 = hl2 @ params["l2w"] + params["l2b"]                  # (B, 8*L3)
        hl3 = _screlu(_select(l2, bucket, arch.num_buckets))      # (B, L3)

        l3 = hl3 @ params["l3w"] + params["l3b"]                  # (B, 8)
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
        dis_pred = dis_h @ params["dis_outw"] + params["dis_outb"]     # (B, 2)
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
    return out, pol_logits, hl2, dis_pred, hl1


def _pairwise(c: jnp.ndarray, hl: int) -> jnp.ndarray:
    return c[..., : hl // 2] * c[..., hl // 2:]


def fake_quant_act(x: jnp.ndarray, levels: float) -> jnp.ndarray:
    """Straight-through quantisation of an ACTIVATION onto a `levels`-step grid
    over [0, 1]. Same STE identity as `fake_quant`.

    Weight-only QAT is the weak version. The lossy steps in NNUE inference are
    the ACTIVATION boundaries, and the engine's are enumerable
    (`sota_head.rs:3` — "int8 L1 -> f32 L2/L3 tail"):

      * FT accumulator clamps to [0, FT_QUANT=255]. Already implicit: a sum of
        quantised weights lands on the grid, so no extra error.
      * **pairwise multiply then `>> FT_SHIFT(9)` -> u8 in [0, 127].** Float
        computes the exact product; the engine keeps **127 levels**. Nothing in
        training has ever seen this, and it is the largest unmodelled step.
      * L1 weights i8 @ L1_QUANT=64, i32 dot, then a single f32 dequant by
        `(1<<9)/(255^2 * 64)`.
      * L2/L3 are f32 in our engine — no analogue to quantise.

    So the emulation is bounded: FT weights, the pairwise grid, and L1 weights.

    **FLOOR, not round.** The engine's `>> FT_SHIFT` is an integer shift, i.e. a
    floor, and Stockfish's own QAT floors activations for exactly this reason
    (`nnue-pytorch/model/quantize.py:36-56` — "activations *floored* to match
    the inference bitshift"). Rounding here would train against a pipeline the
    engine does not have. Weights still use round-to-nearest, which is what both
    their quantiser and ours do.
    """
    q = jnp.clip(jnp.floor(x * levels), 0.0, levels) / levels
    return x + jax.lax.stop_gradient(q - x)


def clip_qat_weights(params: dict, arch: ArchSpec) -> dict:
    """Clip weights into their quantised range — Stockfish's QAT CALLBACK half.

    This is the mechanism that makes a fine scale usable, and we do not have it.
    `nnue-pytorch/trainer/callbacks.py:67-79` clips layer stacks **after every
    batch** and the feature transformer **once per epoch**, to ranges
    l1 +/-0.9922 @128, l2 +/-1.9844 @64, output +/-0.9922 @128 and
    **threat FT +/-0.4961 @256**. Export is then hard-gated (`_safe_convert`
    RAISES on any clipping) — the callback guarantees it never fires.

    Why it matters here (measured 2026-07-29): our threat weights drift to
    +/-1.978 because NOTHING constrains them, and since our exporter also raises
    on overflow, the scale is forced down to 127/1.978 = 64.2 -> 64. One outlier
    out of 77.7 M weights sets the precision for all of them. With this callback
    the weights stay in range and the 256-class scale Stockfish uses becomes
    available; without it, post-hoc rescaling can only trade resolution against
    clipping (measured optimum 128, and SF's 256 is *worse* at 0.82x).

    SF's justification, `docs/nnue.md`: "Because threat feature weights are
    typically not high in absolute value..., we can clip them during training
    with negligible loss in evaluation quality" — worth +10% ARM / +5% x86 from
    the reduced accumulation bandwidth.

    Call between optimiser steps; a no-op when the corresponding scale is 0.
    """
    out = dict(params)
    if arch.qat_threat_scale > 0.0 and not arch.halfka_only:
        lim = 127.0 / arch.qat_threat_scale
        out["l0_threatw"] = jnp.clip(out["l0_threatw"], -lim, lim)
    if arch.qat_l1_scale > 0.0:
        lim = 127.0 / arch.qat_l1_scale
        out["l1w"] = jnp.clip(out["l1w"], -lim, lim)
    return out


def forward(params: dict, arch: ArchSpec, batch: dict,
            ft_impl: str = "reference", with_acc: bool = False,
            with_dis: bool = False, with_hl1: bool = False):
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
    out, pol_logits, hl2, dis_pred, hl1 = head_forward(
        params, arch, acc_stm, acc_ntm,
        batch["bucket"], batch.get("pol_mask"), batch.get("pol_legal"))
    if arch.psqt:
        out = out + psqt_combine(psqt_stm, psqt_ntm, batch["bucket"])
    extras = []
    if with_dis:
        extras.append(dis_pred)
    if with_acc:
        extras.append((acc_stm, acc_ntm))
    # hl1 = the POST-QAT trunk vector the disagreement head reads. Exposed so
    # an offline probe fits the identical representation instead of
    # re-deriving it from the accumulators and silently drifting from the
    # pairwise/crelu/fake_quant_act chain above.
    if with_hl1:
        extras.append(hl1)
    if extras:
        return (out, pol_logits, hl2, *extras)
    return out, pol_logits, hl2


@partial(jax.jit, static_argnames=("arch", "ft_impl"))
def forward_jit(params: dict, arch: ArchSpec, batch: dict,
                ft_impl: str = "reference"):
    return forward(params, arch, batch, ft_impl)[:2]
