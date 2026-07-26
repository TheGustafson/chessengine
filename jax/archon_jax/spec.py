"""The ARCHON D47c architecture + data contract, transcribed for the JAX pipeline.

Every constant here is pinned against the normative sources:

- ``.archon-state/toolchains/bullet/examples/d40_features.rs`` (the trainer; "d40" below)
- ``src/eval/nnue/sota_features.rs`` (the engine feature spec; "sota" below)
- ``src/eval/nnue/policy_vocab.rs`` (the 1858-class move vocabulary)
- ``.archon-state/toolchains/bullet/crates/trainer/src/model/weights.rs``
  (raw-checkpoint serialization + quantisation; "weights.rs" below)
- ``.archon-state/toolchains/bullet/crates/trainer/src/optimiser/adam.rs`` (AdamW kernel)
- ``.archon-state/toolchains/bullet/crates/trainer/src/run.rs`` (grad scale, LR schedule host)
- ``tools/nnue/netpack.py`` (quant.bin consumer)

Do NOT change values here without re-deriving them from those files.
"""

from __future__ import annotations

import dataclasses

import numpy as np

# ---------------------------------------------------------------------------
# Feature spaces (d40 lines 213-218; sota lines 49-53, 132)
# ---------------------------------------------------------------------------
HALFKA_DIMS = 22528          # 64 * PS_NB / 2       (sota:53)
THREAT_DIMS = 60720          # FullThreats          (sota:132)
TOTAL_INPUTS = HALFKA_DIMS + THREAT_DIMS
HALFKA_MAX_ACTIVE = 32       # d40:216
THREAT_MAX_ACTIVE = 160      # d40:217
PS_NB = 704                  # sota:49
PS_KING = 640                # sota:51

# ---------------------------------------------------------------------------
# D47c architecture (d40 lines 221-229; confirmed against the live run's
# checkpoint byte sizes: quant 137,089,088 B == layout WITH policy head + 24 B
# 'bullet' padding; raw 432,573,893 B == 11 named f32 tensors + 157 B headers)
# ---------------------------------------------------------------------------
HL = 1280                    # accumulator width / perspective (d40:229)
L2 = 128                     # shared bottleneck (d40:227)
L3 = 32                      # value tail hidden (d40:228)
NUM_BUCKETS = 8              # piece-count output buckets (d40:221)
POLICY_VOCAB = 1858          # policy_vocab.rs:40
GEOMETRIC_CLASSES = 1792     # policy_vocab.rs:43

# Piece-count -> output bucket (d40:288-298). Index = popcount(occ), min 32.
OUTPUT_BUCKETS_LAYOUT = np.array(
    [0, 0, 0, 0, 0, 0, 0, 0, 0,
     1, 1, 1, 1,
     2, 2, 2, 2,
     3, 3, 3,
     4, 4, 4,
     5, 5, 5,
     6, 6, 6,
     7, 7, 7, 7], dtype=np.int32)
assert OUTPUT_BUCKETS_LAYOUT.shape == (33,)

# ---------------------------------------------------------------------------
# Targets / loss (d40 lines 219-220, 796-801, 877-879)
# ---------------------------------------------------------------------------
DEFAULT_SCALE = 400.0        # D40_EVAL_TARGET_SCALE default (d40:219)
DEFAULT_WDL = 0.3            # D40_WDL default (d40:220)
POLICY_MASK_ILLEGAL = -30.0  # additive mask value (d40:920; chosen so exp()
                             # never underflows to exact 0 -- -1e4 NaN'd the CE)

# ---------------------------------------------------------------------------
# Optimiser (adam.rs:31 + the CUDA kernel at adam.rs:35-52; run.rs:163 makes
# the gradient adj = 1/batch_size, i.e. gradients of the MEAN loss)
# ---------------------------------------------------------------------------
ADAM_DECAY = 0.01            # applied as p *= 1 - decay*lr BEFORE the step
ADAM_BETA1 = 0.9
ADAM_BETA2 = 0.999
ADAM_EPS = 1e-8              # added to sqrt(v) (NOT inside the sqrt)
WEIGHT_CLIP = 1.98           # applied to EVERY tensor AFTER the step
# NOTE: bullet's kernel has NO bias correction (no m_hat/v_hat) -- optax's
# default adamw does NOT match; use archon_jax.optim.bullet_adamw.

# ---------------------------------------------------------------------------
# Quantisation (d40 fmt push order lines 1279-1314; weights.rs QuantTarget)
# ---------------------------------------------------------------------------
QUANT_FT = 255               # l0_pstw / l0_pstb -> i16, round
QUANT_THREAT = 64            # l0_threatw -> i8, round
QUANT_L1 = 64                # l1w -> i8, round (transposed)
QUANT_POL = 64               # polmovew -> i8, round (transposed)
QUANT_PAD = b"bullet"        # cyclic padding to a 64-byte multiple (weights.rs:165-174)

# The raw checkpoint is a concatenation of `name\n` + u64le(count) + f32le[count]
# records in BTreeMap (i.e. lexicographic-by-name) order (weights.rs:95-110,
# utils::write_to_byte_buffer). Tensor list for D47c with the policy head on:
RAW_TENSOR_NAMES = [
    "l0_pstb", "l0_pstw", "l0_threatw",
    "l1b", "l1w", "l2b", "l2w", "l3b", "l3w",
    "polmoveb", "polmovew",
]

# ---------------------------------------------------------------------------
# Bulletformat ChessBoard record (bulletformat-1.8.0 src/chess.rs, 32-byte POD)
# Stored stm-canonicalized: the mover is "White" on a vertically-normalized
# board; score is mover-relative; result in {0,1,2} = {loss,draw,win} for the
# mover; ksq = mover king square; extra[0..3] = packed oriented policy move
# (from, to, promo-code) or zeros.
# ---------------------------------------------------------------------------
RECORD_DTYPE = np.dtype([
    ("occ", "<u8"),
    ("pcs", "u1", (16,)),
    ("score", "<i2"),
    ("result", "u1"),
    ("ksq", "u1"),
    ("opp_ksq", "u1"),
    ("extra", "u1", (3,)),
])
assert RECORD_DTYPE.itemsize == 32


def clip_map_for_arch(arch) -> dict[str, float]:
    """Per-tensor weight clip DERIVED FROM EACH TENSOR'S EXPORT ENCODING (v1.5).

    The historical single WEIGHT_CLIP=1.98 is exactly 127/64 — correct for an
    i8 tensor quantized at 64, and wrong for everything else:
      * f32-exported tensors have NO quantization limit at all
      * the i16 FT at scale 255 can hold |w| < 32767/255 = 128.5, not 1.98
    Measured on d47p-vrepair2-sb80: l0_pstw / l0_threatw / polmovew / polmoveb
    all pinned AT 1.98, i.e. the clip was an active capacity ceiling.

    A tensor absent from this map keeps the caller's default clip.
    `None` means "no clip".
    """
    m: dict[str, float | None] = {}
    for name, kind, quant, _t in arch.quant_layout():
        if kind == "f32":
            m[name] = None                      # no quantization => no clip
        elif kind == "i16":
            # 32767/255 = 128.5 is the exact ceiling; leave ~2% headroom so the
            # FACTORIZER MERGE (l0_factw folded into l0_pstw at export) cannot
            # push the sum over it. netpack's clip gate is the backstop.
            m[name] = 32000.0 / float(quant)    # FT @255 -> 125.5
        else:                                   # i8
            m[name] = 127.0 / float(quant)      # @64 -> 1.984 == the old 1.98
    # Tensors outside quant_layout(): the factorizer shares the FT's i16
    # budget (it is folded in); PSQT columns ship as f32 in the PSQ1 section.
    if getattr(arch, "factorize", False):
        m["l0_factw"] = m.get("l0_pstw", 125.0)
    if getattr(arch, "psqt", False):
        m["l0_psqt_hkw"] = None
        if not getattr(arch, "halfka_only", False):
            m["l0_psqt_thw"] = None
    return m


def decay_map_no_bias(arch, decay: float = ADAM_DECAY) -> dict[str, float]:
    """AdamW decay on weights only. SF puts biases in their own param groups
    and never decays them (and defaults weight decay to 0 entirely); we have
    been decaying every tensor including the FT bias."""
    return {n: (0.0 if n.endswith("b") and not n.endswith("w") else decay)
            for n in arch.param_shapes()}


@dataclasses.dataclass(frozen=True)
class ArchSpec:
    """Parameterized architecture so scaled-down smokes reuse the exact graph.

    ``full()`` is the live D47c-1280 config; anything else is for CPU smokes /
    arch exploration. All layout conventions are identical at every size.
    """

    hl: int = HL
    l2: int = L2
    l3: int = L3
    num_buckets: int = NUM_BUCKETS
    policy: bool = True          # POL_ON (D40_POL_LAMBDA > 0) -- gates head + quant layout
    halfka_only: bool = False    # D40_HALFKA_ONLY (drops the threat tensor entirely)
    # T0 (task #27): king-bucket factorization. Training-time only: a
    # PS_NB-wide bucket-agnostic factor table added to every real HalfKA row
    # (real f shares factor f % PS_NB). Folded into l0_pstw at quant export
    # (bullet merge_factoriser semantics, d40:271-283) -> deploy format
    # UNCHANGED. Zero-init => exact no-op, warm-start safe.
    factorize: bool = False
    # T6c: FT PSQT skip connection (SF-style unclamped linear channel).
    # Per-feature per-output-bucket columns summed straight into the output:
    # out += 0.5 * (psqt_stm - psqt_ntm)[bucket]  (stm-canonical frame =>
    # SF's (us - 0.5) is the constant +0.5). Zero-init => exact no-op,
    # warm-start safe. Deploy: a versioned PSQ1 section after the bullet
    # stream (see quant export) — old nets/engines unaffected.
    psqt: bool = False
    # D47U (2026-07-26): SF-style OUTPUT-SKIP lane — a per-bucket learned
    # linear read of hl1 (the pairwise features, the same vector l1 consumes)
    # added to the value logit outside every nonlinearity. Zero-init => exact
    # no-op, warm-start safe. Deploy: SKP1 section after the bullet stream
    # (PSQ1 pattern); old nets/engines unaffected. SF-verified enabler of
    # thin value heads (their fc_0 +1 lane; nnue_architecture.h L2Big=15).
    out_skip: bool = False
    # Accumulator-branched head projections (D48W / gen-2 sketch 2026-07-27).
    pol_proj: int = 0            # 0 = legacy polmove-on-L2
    dis_proj: int = 0            # 0 = no disagreement head

    @classmethod
    def full(cls) -> "ArchSpec":
        return cls()

    # Parameter shapes in OUR storage convention: every affine weight is
    # (fan_in, fan_out) row-major, which is byte-identical to bullet's internal
    # column-major (fan_out rows x fan_in cols) flat layout -- so raw-checkpoint
    # IO is a plain reshape, no transposes (weights.rs stores col-major; the
    # merge_factoriser comment in d40:262-266 pins "feature-major
    # w[feature*HL + neuron]").
    def param_shapes(self) -> dict[str, tuple[int, ...]]:
        shapes = {
            "l0_pstw": (HALFKA_DIMS, self.hl),
            "l0_pstb": (self.hl,),
            "l1w": (self.hl, self.num_buckets * self.l2),
            "l1b": (self.num_buckets * self.l2,),
            "l2w": (self.l2, self.num_buckets * self.l3),
            "l2b": (self.num_buckets * self.l3,),
            "l3w": (self.l3, self.num_buckets),
            "l3b": (self.num_buckets,),
        }
        if not self.halfka_only:
            # No bias: the shared FT bias lives on the HalfKA tensor only
            # (d40:951-956) -- landmine: do NOT give the threat FT a bias.
            shapes["l0_threatw"] = (THREAT_DIMS, self.hl)
        if self.factorize:
            shapes["l0_factw"] = (PS_NB, self.hl)
        if self.psqt:
            shapes["l0_psqt_hkw"] = (HALFKA_DIMS, self.num_buckets)
            if not self.halfka_only:
                shapes["l0_psqt_thw"] = (THREAT_DIMS, self.num_buckets)
        if self.policy:
            if self.pol_proj:
                # D48W: policy branches OFF THE ACCUMULATOR through its own
                # bottleneck — never through the value L2 (the measured V
                # mistake: top-1 22.7->19 through the 32-wide chokepoint).
                shapes["pol_projw"] = (self.hl, self.pol_proj)
                shapes["pol_projb"] = (self.pol_proj,)
                shapes["polmovew"] = (self.pol_proj, POLICY_VOCAB)
            else:
                shapes["polmovew"] = (self.l2, POLICY_VOCAB)
            shapes["polmoveb"] = (POLICY_VOCAB,)
        if self.dis_proj:
            # T11 seed: disagreement/uncertainty head off the accumulator,
            # predicts |sigmoid(value) - target| (its own error magnitude).
            shapes["dis_projw"] = (self.hl, self.dis_proj)
            shapes["dis_projb"] = (self.dis_proj,)
            shapes["dis_outw"] = (self.dis_proj,)
            shapes["dis_outb"] = (1,)
        if self.out_skip:
            shapes["skipw"] = (self.hl, self.num_buckets)
        return shapes

    def quant_layout(self):
        """The quant.bin tensor order + encodings (d40 fmt vec, lines 1295-1314).

        Yields (name, kind, quant, transpose) where kind is 'i16'/'i8'/'f32'.
        The policy head rides BETWEEN the threat FT and l1 (d40:1305-1308).
        """
        out = [("l0_pstw", "i16", QUANT_FT, False),
               ("l0_pstb", "i16", QUANT_FT, False)]
        if not self.halfka_only:
            out.append(("l0_threatw", "i8", QUANT_THREAT, False))
        if self.policy:
            out.append(("polmovew", "i8", QUANT_POL, True))
            out.append(("polmoveb", "f32", None, False))
        out += [("l1w", "i8", QUANT_L1, True),
                ("l1b", "f32", None, False),
                ("l2w", "f32", None, True),
                ("l2b", "f32", None, False),
                ("l3w", "f32", None, True),
                ("l3b", "f32", None, False)]
        return out


@dataclasses.dataclass(frozen=True)
class TrainConfig:
    """The d47c recipe knobs (from /tmp/d47c_train.sh + d40 defaults)."""

    batch_size: int = 8192           # D40_BATCH
    batches_per_superbatch: int = 3040  # D40_BPS
    superbatches: int = 240          # D40_SUPERBATCHES
    lr: float = 0.001                # D40_LR
    lr_end_superbatch: int = 240     # D40_LR_END_SB
    wdl: float = DEFAULT_WDL         # D40_WDL
    scale: float = DEFAULT_SCALE     # D40_EVAL_TARGET_SCALE
    score_multiplier: float = 1.0    # D40_SCORE_MULTIPLIER
    # Draw-aware double-sigmoid target (2026-07-26 calibration). 0.0 = the
    # historical plain sigmoid(s/scale). a>0 gives P(win)=sig((s-a)/scale),
    # P(loss)=sig((-s-a)/scale), remainder draw, value = EXPECTED SCORE — the
    # same quantity as `result`, so the wdl blend finally mixes like with like.
    # MLE fits on our corpora: uho a=175.5 scale=71.9, fishpack a=160.2
    # scale=33.2 (tools/nnue/fit_target_transform.py).
    target_offset: float = 0.0
    # Frame of a DISTILL TEACHER trained with a different target transform
    # (set them to the TEACHER run's target_offset/scale). When offset > 0,
    # the teacher's predicted expected score is mapped through the inverse
    # of its own transform back to cp and re-projected into THIS run's
    # plain-sigmoid(s/scale) frame before the α-blend, so both blend terms
    # live in one frame. Without this the student settles BETWEEN frames —
    # the 2026-07-26 D47V 1.378× output-scale gotcha.
    teacher_frame_offset: float = 0.0
    teacher_frame_scale: float = 0.0
    # Linear ramp distill_alpha -> distill_alpha_end over the whole schedule
    # (nnue-pytorch --start-lambda/--end-lambda precedent: fold ground truth
    # in as training proceeds). Negative (default) = constant alpha.
    distill_alpha_end: float = -1.0
    # Piecewise-linear alpha over ABSOLUTE superbatch: ((sb, alpha), ...).
    # Takes precedence over distill_alpha_end. Clamps flat outside the
    # breakpoints. The 2026-07-26 D47V leg-2 schedule: ((70, 0.7),
    # (100, 0.10), (240, 0.0)) — teacher to 10% by sb100, slow to 0.
    distill_alpha_ramp: tuple = ()
    # ---- v1.5 warm-refinement knobs (2026-07-26). All default to the
    # historical behaviour, so every existing config is untouched. ----
    # Per-tensor weight clips derived from each tensor's export encoding
    # (clip_map_for_arch) instead of one global 1.98. MEASURED: the uniform
    # clip is binding on nearly every tensor of a live net.
    per_tensor_clip: bool = False
    # SF never decays biases (and defaults weight decay to 0 entirely).
    no_bias_decay: bool = False
    # Global gradient-norm clip; 0.0 = off. SF hard-codes 2.0.
    grad_clip_norm: float = 0.0
    # Value-loss exponent |sigmoid(out) - target|^p. SF uses 2.5 (docs say 2.6
    # also worked); higher = more weight on precision, less on outliers.
    pow_exp: float = 2.0
    pol_lambda: float = 0.005        # D40_POL_LAMBDA
    weight_decay: float = ADAM_DECAY
    # FT-sparsity regularizer (perf study #6 #1 NPS lever): L1 on the
    # post-CReLU accumulators, driving deploy density down (density-linear
    # l1_dot cost). 0.0 = OFF = graph identical to the incumbent recipe.
    dis_lambda: float = 0.0          # disagreement-head loss weight (D48W)
    ft_sparsity_lambda: float = 0.0
    # Resume: 1-based superbatch this run STARTS at (bullet's
    # D40_START_SUPERBATCH). The LR schedule keys on
    # start_superbatch + steps/bps so a resumed run continues the cosine
    # instead of restarting it (fresh optimizer state = the accepted ~2sb
    # transient, the d47c OOM-resume precedent).
    start_superbatch: int = 1
    # EMA shadow weights (stage-4+ pump-chain upgrade, 2026-07-23): the
    # SHIPPABLE net is the exponential moving average, which barely
    # bounces through a warm-restart pump and typically ends better than
    # the live weights. 0.0 = off. Default horizon when on: ~2sb of
    # steps (decay = 1 - 1/6080 ≈ 0.99984 at BPS 3040).
    ema_decay: float = 0.0
    # LR warmup ramp (stage-4+): linear per-step ramp over the first N
    # superbatches of a FRESH run (never re-applied on resume — gated on
    # start_superbatch == 1). Softens the step-0 shock of pumping an
    # annealed warm net at full peak LR. 0.0 = off (bullet parity).
    warmup_superbatches: float = 0.0
    # FT-sparsity λ ramp (D47M lesson, 2026-07-25): a leg that REMOVES a
    # gradient source which had been resisting the L1 (head freeze-out,
    # trunk freeze) leaves the penalty unopposed on the features that
    # source had recruited — D47M pruned density 0.29→0.19, overshooting
    # the incumbent's 0.26 equilibrium, then spent ~60sb rebuilding.
    # Ramps ft_sparsity_lambda linearly 0→target over the first N
    # superbatches (fresh runs only, same gate as the LR warmup).
    # 0.0 = off = the step-change behavior every pre-D47M leg used.
    ft_sparsity_warmup_superbatches: float = 0.0
    # D47V teacher distillation: value target = alpha*sigmoid(teacher) +
    # (1-alpha)*label; policy CE against the teacher softmax when a teacher
    # is passed to make_update_step. 0 = off (default, bit-identical).
    distill_alpha: float = 0.0


def cosine_lr(cfg: TrainConfig, superbatch: int) -> float:
    """The d40 LR schedule (d40:1251-1259): cosine from lr to lr*0.3^3,
    parameterized by the 1-based superbatch index only (constant within a
    superbatch)."""
    initial, end_sb = cfg.lr, cfg.lr_end_superbatch
    final = initial * 0.3 ** 3
    if end_sb <= 1:
        return initial
    t = min(max((superbatch - 1.0) / (end_sb - 1.0), 0.0), 1.0)
    return final + 0.5 * (initial - final) * (1.0 + np.cos(np.pi * t))
