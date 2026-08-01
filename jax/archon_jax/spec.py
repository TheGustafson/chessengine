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
# l0_threatw -> i8, round. The DEFAULT only; the scale a given net is exported
# at comes from `ArchSpec.threat_quant()`, which returns `qat_threat_scale`
# whenever QAT is on. 64 is the largest scale a NON-QAT net's +-1.98 weights
# survive; a QAT net clipped to +-127/256 wants 256 and loses two bits at 64
# (perf study 30). Engine side: CENN-v5 carries it in the header.
QUANT_THREAT = 64
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

# ---------------------------------------------------------------------------
# Stockfish-side constants, needed only when SF's own net is the live teacher
# (sf-teacher-blend-2026-08-01). Kept here so the trainer, the converter and
# the calibration tool cannot drift apart on them.
# ---------------------------------------------------------------------------
SF_BUCKETS = 8               # their layer stacks; index = (pieces-1)/4
# `material` in SF's evaluate() blend: 534*count<PAWN>() + non_pawn_material().
# Piece values are ARCHON's transcription of eca43a97 (types.rs:636-644);
# index by PieceType 0=P 1=N 2=B 3=R 4=Q 5=K.
PIECE_VALUES = np.array([0, 781, 825, 1276, 2538, 0], dtype=np.int64)
MATERIAL_PAWN_WEIGHT = 534   # eval/params.rs:20
EVAL_BLEND_BASE = 77871      # eval/params.rs:27 — v*(BASE+material)/BASE


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
    # --- QAT (2026-07-29, owner ruling: "QAT should happen between every
    # operation"). Straight-through fake quantisation of the ACTUAL integer
    # pipeline, not just the weights. 0.0 = off everywhere => bit-exact no-op,
    # warm-start safe. Engine reference: sota_head.rs:118-133.
    #   qat_ft_scale     FT HalfKA table -> i16 @ FT_QUANT     (255)
    #   qat_threat_scale FT threat table -> i8  @ the EXPORT threat scale.
    #                    **This value IS the export scale** — see
    #                    `threat_quant()` below, which every quant path reads,
    #                    and CENN-v5, which writes it into the net header. It
    #                    used to be free to disagree with `QUANT_THREAT`, and
    #                    D49 trained 100 superbatches at 256 while the packer
    #                    exported at 64: max|w| 32/127, 65 of 256 i8 levels,
    #                    -128.7 +- 9.6 Elo over 1600 games, invisible to every float metric
    #                    (perf study 30).
    #   qat_act_levels   pairwise product -> u8, 127 levels    (the >>9 step;
    #                    the largest previously unmodelled loss)
    #   qat_l1_scale     L1 weights -> i8 @ L1_QUANT           (64)
    # L2/L3 are f32 in our engine, so there is nothing below L1 to emulate.
    qat_ft_scale: float = 0.0
    qat_threat_scale: float = 0.0
    qat_act_levels: float = 0.0
    qat_l1_scale: float = 0.0
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
    # R2 (nnue-architecture-study-2026-07-28 §4.4/§7): SF `eca43a97`'s tail —
    # DUAL ACTIVATION + inter-layer concat skip. Our tail is the `sf_18`-era
    # shape: one activation per layer, and the neck's only route to the output
    # is fc_1's CReLU. SF master hands every hidden unit forward as BOTH
    # SqrCReLU(x) and CReLU(x), and lets fc_2 read layer 0's pair directly
    # (eca43a97:src/nnue/nnue_architecture.h:56-67 —
    #   ac_sqr_0/ac_0 <FC_0_OUTPUTS>, fc_1 <FC_0_OUTPUTS*2, FC_1_OUTPUTS>,
    #   ac_sqr_1/ac_1 <FC_1_OUTPUTS>, fc_2 <FC_0_OUTPUTS*2 + FC_1_OUTPUTS*2, 1>).
    # Shape only: SF's per-layer WeightScaleBits(+1) shifts are int8
    # quantisation bookkeeping and have no f32 analogue.
    # L1 and L2 are UNCHANGED (l1_dot untouched); only l2w/l3w grow:
    #   l2w (l2,   B*l3) -> (2*l2,        B*l3)
    #   l3w (l3,   B)    -> (2*l2 + 2*l3, B)
    # False = today's single-activation tail, byte-for-byte.
    dual_tail: bool = False
    # SF-net TEACHER only (sf-teacher-blend-2026-08-01): the exact float
    # analogue of SF's integer dual activation, which `dual_tail` deliberately
    # does NOT reproduce (it is "shape only" — see above). Two differences,
    # both load-bearing when the goal is to REPRODUCE their net rather than to
    # borrow its topology:
    #   * SF squares the UNCLIPPED pre-activation: `(x*x) >> 21` has no lower
    #     clamp, so a NEGATIVE unit contributes POSITIVELY to the sqr lane,
    #     where our SqrCReLU sends it to 0. Ignoring this is not a rounding
    #     difference, it changes the function.
    #   * the sqr lane's gain is 127/128, not 1: at the normalisation that
    #     makes the clipped lane `clamp(x,0,1)` (divide by 2^(WSB+1)*127), the
    #     square lands on `min(x^2 * 127/128, 1)`.
    # Derivation in docs/research/sf-teacher-blend-2026-08-01.md §2.
    # Requires dual_tail; False everywhere else, so no trained net moves.
    sf_act: bool = False
    # Recode POC (learned-feature-code study §6.1, 2026-07-30): the threat
    # embedding's ROW COUNT. 0 = the identity code (THREAT_DIMS rows, every
    # net before the study). When a run trains a re-coded threat channel,
    # this is the compressed row count R'; the u16 code table itself rides
    # in TrainConfig.threat_remap (a data-pipeline concern — the arch only
    # needs the shape). A net is now (weights, code): the code's sha256
    # belongs in the ledger row beside the net's.
    threat_rows: int = 0

    @classmethod
    def full(cls) -> "ArchSpec":
        return cls()

    @classmethod
    def from_run_spec(cls, spec: dict) -> "ArchSpec":
        """The arch a `tools/nnue/runs/*.json` describes — ONE definition.

        `pump_train` used to build this inline, so any other tool that wanted
        to re-export a run's checkpoint had to re-derive it and could silently
        get a different arch (`tools/nnue/requant_threats.py` used the bare
        default, whose `l2` is 128 where every gen-2 run's is 16). Both read
        this now.
        """
        import dataclasses as _dc
        return _dc.replace(
            cls.full(), factorize=True, psqt=True,
            hl=int(spec.get("arch_hl", cls.full().hl)),
            l2=int(spec.get("arch_l2", cls.full().l2)),
            l3=int(spec.get("arch_l3", cls.full().l3)),
            out_skip=bool(spec.get("arch_out_skip", False)),
            halfka_only=bool(spec.get("arch_halfka_only", False)),
            pol_proj=int(spec.get("arch_pol_proj", 0)),
            dis_proj=int(spec.get("arch_dis_proj", 0)),
            qat_ft_scale=float(spec.get("qat_ft_scale", 0.0)),
            qat_threat_scale=float(spec.get("qat_threat_scale", 0.0)),
            qat_act_levels=float(spec.get("qat_act_levels", 0.0)),
            qat_l1_scale=float(spec.get("qat_l1_scale", 0.0)),
            threat_rows=int(spec.get("arch_threat_rows", 0)),
            num_buckets=int(spec.get("arch_num_buckets", NUM_BUCKETS)))

    # Parameter shapes in OUR storage convention: every affine weight is
    # (fan_in, fan_out) row-major, which is byte-identical to bullet's internal
    # column-major (fan_out rows x fan_in cols) flat layout -- so raw-checkpoint
    # IO is a plain reshape, no transposes (weights.rs stores col-major; the
    # merge_factoriser comment in d40:262-266 pins "feature-major
    # w[feature*HL + neuron]").
    def param_shapes(self) -> dict[str, tuple[int, ...]]:
        # dual_tail widens only the two tail fan-ins (see the field comment):
        # fc_1 reads the neck's dual activation, fc_2 reads BOTH layers' pairs.
        l2_fan_in = 2 * self.l2 if self.dual_tail else self.l2
        l3_fan_in = 2 * self.l2 + 2 * self.l3 if self.dual_tail else self.l3
        shapes = {
            "l0_pstw": (HALFKA_DIMS, self.hl),
            "l0_pstb": (self.hl,),
            "l1w": (self.hl, self.num_buckets * self.l2),
            "l1b": (self.num_buckets * self.l2,),
            "l2w": (l2_fan_in, self.num_buckets * self.l3),
            "l2b": (self.num_buckets * self.l3,),
            "l3w": (l3_fan_in, self.num_buckets),
            "l3b": (self.num_buckets,),
        }
        # Recode POC: 0 = identity code = the full geometric space.
        threat_rows = self.threat_rows or THREAT_DIMS
        if not self.halfka_only:
            # No bias: the shared FT bias lives on the HalfKA tensor only
            # (d40:951-956) -- landmine: do NOT give the threat FT a bias.
            shapes["l0_threatw"] = (threat_rows, self.hl)
        if self.factorize:
            shapes["l0_factw"] = (PS_NB, self.hl)
        if self.psqt:
            shapes["l0_psqt_hkw"] = (HALFKA_DIMS, self.num_buckets)
            if not self.halfka_only:
                shapes["l0_psqt_thw"] = (threat_rows, self.num_buckets)
        if self.policy:
            if self.pol_proj:
                # D48W: policy branches OFF THE ACCUMULATOR through its own
                # bottleneck — never through the value L2 (the measured V
                # mistake: top-1 22.7->19 through the 32-wide chokepoint).
                shapes["pol_projw"] = (self.hl, self.pol_proj)
                shapes["pol_projb"] = (self.pol_proj,)
                shapes["polmovew"] = (self.pol_proj, POLICY_VOCAB)
            else:
                # legacy policy-on-L2 reads hl2, which dual_tail widens to 2*l2
                shapes["polmovew"] = (l2_fan_in, POLICY_VOCAB)
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

    def threat_quant(self):
        """The scale the threat FT must be EXPORTED at — derived, never chosen.

        **This is the one place the trainer's QAT scale and the export scale
        meet, and it is a derivation so they cannot disagree.** QAT clips the
        threat table to +-127/`qat_threat_scale`; exporting at any smaller
        scale throws away exactly the range QAT reserved. D49 trained at 256
        while every deployment path hardcoded 64, and the resulting net used
        32 of 127 i8 levels and lost 125 Elo with a float heldout R2 within
        0.008 of the incumbent (perf study 30).

        Falls back to the module default when QAT is off, which is what every
        pre-QAT net was exported at, so nothing about them changes.
        """
        scale = self.qat_threat_scale if self.qat_threat_scale > 0.0 else float(QUANT_THREAT)
        q = int(round(scale))
        if q != scale or q <= 0 or q > 256 or (q & (q - 1)):
            raise ValueError(
                f"qat_threat_scale={scale} cannot be an export scale: the engine "
                f"carries FT_QUANT/threat_quant as one shift, so it must be an "
                f"integral power of two in 1..=256 "
                f"(sota_head::threat_shift_for; CENN-v5 header)")
        return q

    def quant_layout(self):
        """The quant.bin tensor order + encodings (d40 fmt vec, lines 1295-1314).

        Yields (name, kind, quant, transpose) where kind is 'i16'/'i8'/'f32'.
        The policy head rides BETWEEN the threat FT and l1 (d40:1305-1308).
        """
        out = [("l0_pstw", "i16", QUANT_FT, False),
               ("l0_pstb", "i16", QUANT_FT, False)]
        if not self.halfka_only:
            out.append(("l0_threatw", "i8", self.threat_quant(), False))
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
    # Recode POC: path to a .npy int array (len THREAT_DIMS, entries <
    # arch.threat_rows) re-coding the threat index space at the loader.
    # "" = identity. Pairs with ArchSpec.threat_rows; data.py loads it once
    # per stream and passes it into map_batch — zero trainer-graph changes.
    threat_remap: str = ""
    # Richer-representation study (2026-07-31): CONTEXT-REFINED threat code.
    # threat_code2d names a .npy int array of shape (THREAT_DIMS, 2) mapping
    # (tuple, region-bit) -> row (< arch.threat_rows); threat_region names the
    # king-region bit (archon_jax.features.REGION_BIT_NAMES) that indexes the
    # second axis, computed per perspective at the loader. Mutually exclusive
    # with threat_remap. "" = off.
    threat_code2d: str = ""
    threat_region: str = ""
    # Head-side refinement: comma-separated region-bit names that extend the
    # OUTPUT bucket index (each doubles the bucket space; arch.num_buckets
    # must be 8 * 2^n). "" = the plain piece-count buckets.
    bucket_regions: str = ""
    # Draw-aware double-sigmoid target (2026-07-26 calibration). 0.0 = the
    # historical plain sigmoid(s/scale). a>0 gives P(win)=sig((s-a)/scale),
    # P(loss)=sig((-s-a)/scale), remainder draw, value = EXPECTED SCORE — the
    # same quantity as `result`, so the wdl blend finally mixes like with like.
    # MLE fits on our corpora: uho a=175.5 scale=71.9, fishpack a=160.2
    # scale=33.2 (tools/nnue/fit_target_transform.py).
    target_offset: float = 0.0
    # D48Y: [[offset, scale], ...] aligned with the mix order. When set it
    # OVERRIDES target_offset/scale per record. None = unchanged behaviour.
    per_corpus_target: tuple | None = None
    # Frame of a DISTILL TEACHER trained with a different target transform
    # (set them to the TEACHER run's target_offset/scale). When offset > 0,
    # the teacher's predicted expected score is mapped through the inverse
    # of its own transform back to cp and re-projected into THIS run's
    # plain-sigmoid(s/scale) frame before the α-blend, so both blend terms
    # live in one frame. Without this the student settles BETWEEN frames —
    # the 2026-07-26 D47V 1.378× output-scale gotcha.
    teacher_frame_offset: float = 0.0
    teacher_frame_scale: float = 0.0
    # SF-NET TEACHER frame (sf-teacher-blend-2026-08-01). The two scalars above
    # cannot express it and no scalar can: an SF net emits raw internal units,
    # and the measured conversion into our label frame is
    #     label_frame = k_i * (1 + material/77871)^p_i * teacher_raw
    # per corpus AND per position. Over the (corpus x material) grid the
    # multiplier runs 0.319-0.604 (1.90x); the weight-averaged constant 0.4182
    # is wrong by up to +45%, and uho (25% of the mix) is off -20% at low
    # material and +19% at high BY THE SAME CONSTANT. So this is a list of
    # (k, p), indexed POSITIONALLY against `mix` exactly like
    # per_corpus_target (verified against the data, not assumed — that study
    # §3, 11/12 corpora reproduce their own slot to <=1.6%).
    # None = no SF teacher, and every code path it touches stays inert.
    teacher_kp: tuple | None = None
    # The teacher's ArchSpec kwargs (tools/nnue/sf_to_archon_warm.py
    # --dump-arch). Empty = the historical ArchSpec.full() teacher.
    teacher_arch: tuple = ()
    # --- D49 (2026-07-29). All default to the historical behaviour, so an
    # untouched config trains bit-identically to before.
    # Loss: SF applies the offset two-sigmoid to BOTH sides; D48Y2 did the
    # LABEL side only, which is the leading candidate for its R2-without-
    # strength result. net_scale > 0 enables the NETWORK side.
    net_offset: float = 0.0
    net_scale: float = 0.0
    qp_asymmetry: float = 0.0        # SF ships 0.2287
    # Data shaping. SF's loader REWEIGHTS (probabilistic skip); ours was a pure
    # accept/reject predicate with no shaping at all.
    piece_rebalance: bool = False    # SF PR #4942 spline; our mix is 3x endgame-skewed
    wdl_filter: bool = False         # down-weight eval/result disagreement
    wdl_filter_floor: float = 0.0
    # QAT scales (0 = off). Engine constants: sota_head.rs:118-133.
    qat_ft_scale: float = 0.0
    qat_threat_scale: float = 0.0
    qat_act_levels: float = 0.0
    qat_l1_scale: float = 0.0
    # Linear ramp distill_alpha -> distill_alpha_end over the whole schedule
    # (nnue-pytorch --start-lambda/--end-lambda precedent: fold ground truth
    # in as training proceeds). Negative (default) = constant alpha.
    distill_alpha_end: float = -1.0
    # Piecewise-linear alpha over ABSOLUTE superbatch: ((sb, alpha), ...).
    # Takes precedence over distill_alpha_end. Clamps flat outside the
    # breakpoints. The 2026-07-26 D47V leg-2 schedule: ((70, 0.7),
    # (100, 0.10), (240, 0.0)) — teacher to 10% by sb100, slow to 0.
    distill_alpha_ramp: tuple = ()
    # Piecewise-linear LR on the absolute superbatch (same mechanism as
    # distill_alpha_ramp; clamped flat outside the breakpoints). Takes
    # priority over the lr/lr_end_superbatch cosine when non-empty — lets
    # a warm-up/main/anneal curve live in ONE config so long runs never
    # stop for an LR change (D48W schedule, user design 2026-07-27).
    lr_ramp: tuple = ()
    # Superbatch export cadence. 10 = the historical default. The LR
    # jump->anneal cycles set this to the cycle length so checkpoints land
    # only at the BOTTOM of the cycle (LR -> 0).
    export_every: int = 10
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
