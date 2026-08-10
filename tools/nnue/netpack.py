#!/usr/bin/env python3
"""Pack a bullet ``quantised.bin`` into our CENN container (design/11 §3.4.5).

P13 step 3 deliverable. bullet exports a flat little-endian ``i16`` blob in the
save-format order pinned by ``tools/nnue/train_net0.rs`` (== the canonical
bullet ``1_simple`` layout):

    l0w : i16[768*256]   feature-transformer weights, feature-major
    l0b : i16[256]       feature-transformer bias
    l1w : i16[512]       output weights, stm half first (stm.concat(ntm))
    l1b : i16[1]         output bias, scale QA*QB = 16320
    (then zero-or-more padding bytes to a 64-byte multiple)

CENN wants exactly the same payload order, EXCEPT the output bias is an
``i32`` (design/11 §3.4.5). bullet emits ``l1b`` as a single ``i16`` at the
``QA*QB`` scale, so we **sign-extend** it into the i32 field (the §3.4.5 packer
note; ``int()`` on a numpy ``<i2`` element is already the signed value, so
writing it as an ``i32`` is a faithful sign-extension — never zero-extend).

The engine loader (src/eval/nnue/loader.rs) HARD-REJECTS any net whose
quantized ``|out_weight| > 127`` (``NnueError::OutWeightOutOfClip``), because
the AVX2 ``mullo_epi16`` SCReLU trick is bit-exact vs the scalar reference only
under that clip (§3.4.3). bullet's default AdamW clip is float ±1.98
(``1.98 * QB=64 = 126.72 -> round 127``), so a correctly-trained net satisfies
it — but we VERIFY here before packing and refuse to emit an out-of-clip net,
so a bad net fails loudly at pack time rather than at load time.

Run under uv (design/11 §11.1):
    uv run python tools/nnue/netpack.py <quantised.bin> <out.cenn> [screlu|crelu]

Discardable-scaffold-adjacent tooling (D15/D28): reads a gitignored bullet
artifact, writes a gitignored net. No engine (`src/`) code is touched.
"""

import os
import sys
from pathlib import Path

import numpy as np

# Architecture + quantization — must match src/eval/nnue/mod.rs and the
# loader header (design/11 §3.4.1, §3.4.3, §3.4.5).
INPUT = 768
# Hidden width. 256 default; set NET_L1=512 to pack a 768x512 net (must match the
# bullet NET0_HIDDEN the net was trained with, and the engine build's L1).
L1 = int(os.environ.get("NET_L1", "256"))
QA = 255
QB = 64
SCALE = 400
MAGIC = b"CENN"
CENN_VERSION = 1
HEADER_LEN = 48

# Payload element counts, in CENN order.
N_FT_W = INPUT * L1  # 196608
N_FT_B = L1          # 256
N_OUT_W = 2 * L1     # 512
# ... then one i16 l1b -> i32 out_bias.

# The AVX2 SCReLU clip the loader enforces (design/11 §3.4.3).
OUT_WEIGHT_CLIP = 127


def die(msg: str):
    print(f"netpack: {msg}", file=sys.stderr)
    sys.exit(2)


def fnv1a64(data: bytes) -> int:
    """FNV-1a 64-bit (design/11 §3.4.5 payload checksum). Matches loader.rs."""
    h = 0xCBF29CE484222325
    for byte in data:
        h ^= byte
        h = (h * 0x100000001B3) & 0xFFFFFFFFFFFFFFFF
    return h


def activation_id(name: str) -> int:
    return {"crelu": 0, "screlu": 1}.get(name.lower(), None)


def pack(quantised_path: str, out_path: str, activation: str) -> None:
    act_id = activation_id(activation)
    if act_id is None:
        die(f"bad activation '{activation}' (want screlu or crelu)")

    raw = Path(quantised_path).read_bytes()
    n_i16 = N_FT_W + N_FT_B + N_OUT_W + 1  # 197377
    need_bytes = n_i16 * 2                 # 394754 (before bullet's 64B pad)
    if len(raw) < need_bytes:
        die(f"{quantised_path}: {len(raw)} bytes < {need_bytes} needed "
            f"({n_i16} i16 for {INPUT}->2x{L1}->1)")

    a = np.frombuffer(raw, dtype="<i2", count=n_i16)
    off = 0
    ft_w = a[off:off + N_FT_W]; off += N_FT_W
    ft_b = a[off:off + N_FT_B]; off += N_FT_B
    out_w = a[off:off + N_OUT_W]; off += N_OUT_W
    l1b_i16 = int(a[off])  # signed i16 -> Python int (already sign-extended)

    # --- the pre-pack clip gate (design/11 §3.4.3) ---------------------------
    max_abs_out = int(np.abs(out_w.astype(np.int32)).max())
    if max_abs_out > OUT_WEIGHT_CLIP:
        bad = int(np.argmax(np.abs(out_w.astype(np.int32))))
        die(f"out_weight[{bad}] = {int(out_w[bad])} exceeds the AVX2 SCReLU clip "
            f"|w| <= {OUT_WEIGHT_CLIP} (design/11 §3.4.3). The loader would reject "
            f"this net (NnueError::OutWeightOutOfClip). Tighten the training clip "
            f"(bullet AdamW min/max_weight, currently ±1.98) and retrain — NOT "
            f"packing a net our engine cannot load.")

    # --- build payload in CENN order (all little-endian) ---------------------
    payload = bytearray()
    payload += ft_w.tobytes()   # i16[768*256] feature-major
    payload += ft_b.tobytes()   # i16[256]
    payload += out_w.tobytes()  # i16[512] stm half first
    # sign-extend the i16 output bias into the i32 field (§3.4.5 packer note).
    payload += np.int32(l1b_i16).tobytes()

    expected_payload = (N_FT_W + N_FT_B + N_OUT_W) * 2 + 4
    assert len(payload) == expected_payload, (len(payload), expected_payload)
    checksum = fnv1a64(bytes(payload))

    # --- header (design/11 §3.4.5) -------------------------------------------
    header = bytearray()
    header += MAGIC
    header += int(CENN_VERSION).to_bytes(4, "little")
    header += int(INPUT).to_bytes(4, "little")
    header += int(L1).to_bytes(4, "little")
    header += bytes([act_id])
    header += b"\x00\x00\x00"                       # padding
    header += int(QA).to_bytes(2, "little")
    header += int(QB).to_bytes(2, "little")
    header += int(SCALE).to_bytes(2, "little")
    header += b"\x00" * 6                            # padding
    header += len(payload).to_bytes(8, "little")
    header += checksum.to_bytes(8, "little")
    assert len(header) == HEADER_LEN, len(header)

    with open(out_path, "wb") as f:
        f.write(header)
        f.write(payload)

    total = HEADER_LEN + len(payload)
    print(f"netpack: wrote {out_path} ({total} bytes)")
    print(f"  arch        {INPUT} -> 2x{L1} -> 1   activation={activation.lower()}")
    print(f"  quant       QA={QA} QB={QB} SCALE={SCALE}")
    print(f"  out_weights max|w| = {max_abs_out}  (clip {OUT_WEIGHT_CLIP}: "
          f"{'OK' if max_abs_out <= OUT_WEIGHT_CLIP else 'FAIL'})")
    print(f"  out_bias    l1b i16 {l1b_i16} -> i32 (sign-extended)")
    print(f"  payload_len {len(payload)}   checksum {checksum:#018x}")


# ============================================================================
# D40 CENN-v2 packer (design/11 §12.6)
#
# Transcodes bullet d40_features.rs quantised.bin → CENN-v2 (.cenn).
#
# Bullet's d40_features SavedFormat (from research):
#   0: l0_pstw  i16  (ACC, HALFKA)  neuron-major (scale 255)
#   1: l0_pstb  i16  (ACC, 1)       shared FT bias (scale 255)
#   2: l0_threatw i8 (ACC, THREAT)  neuron-major (scale 64)
#   3: l1w      i8   (ACC, BUCKETS*L2)  TRANSPOSED → on disk (BUCKETS*L2, ACC)
#   4: l1b      f32  (BUCKETS*L2, 1)
#   5: l2w      f32  (L2, BUCKETS*L3)   TRANSPOSED → on disk (BUCKETS*L3, L2)
#   6: l2b      f32  (BUCKETS*L3, 1)
#   7: l3w      f32  (L3, BUCKETS)      TRANSPOSED → on disk (BUCKETS, L3)
#   8: l3b      f32  (BUCKETS, 1)
#
# The engine wants feature-major FTs and a group-interleaved L1 layout.
# ============================================================================

MAGIC_V2 = b"CEN2"
CENN_V2_VERSION = 2
HEADER_V2_LEN = 64
# CENN-v5 (perf study 30): the v2 header plus `threat_quant: i32` at offset 64
# and a zero `reserved: u32` at 68; the payload then starts at 72. FIVE, not
# three: CEN3/CEN4 are the simple/protected HalfKA containers
# (src/eval/nnue/simple_halfka.rs, protected_halfka.rs) and uci.rs routes on
# them. Loader: src/eval/nnue/sota_loader.rs.
MAGIC_V5 = b"CEN5"
CENN_V5_VERSION = 5
HEADER_V5_LEN = 72
FLAG_HAS_THREATS = 1

# D40 architecture constants — must match src/eval/nnue/sota_head.rs +
# sota_features.rs.
HALFKA_DIMS = 22528
THREAT_DIMS = 60720
# Arch dims are env-overridable so a single netpack tracks the trainer/engine
# consts without a code edit (defaults = the original 768+threats D40 arch, so
# packing an old net is unchanged). The D41 "drop threats, widen" arch sets
# D40_ACC=1536 D40_L2=32 D40_HALFKA_ONLY=1 to match sota_head.rs.
D40_ACC = int(os.environ.get("D40_ACC", "768"))      # Accumulator width (= SotaHead::L1)
D40_L2 = int(os.environ.get("D40_L2", "16"))
D40_L3 = int(os.environ.get("D40_L3", "32"))
D40_BUCKETS = int(os.environ.get("D40_BUCKETS", "8"))
# When set, the net has NO FullThreats channel (HalfKA-only): skip the threat
# tensor entirely and clear FLAG_HAS_THREATS so the engine loader uses its
# zero-threat accumulator path.
D40_HALFKA_ONLY = os.environ.get("D40_HALFKA_ONLY", "0") not in ("0", "", "false", "False")
# D47 (2026-07-22): the shared-bottleneck multihead quant carries the 1858-row
# policy head BETWEEN l0_threatw and l1w (trainer fmt order). D40_POLICY=1
# parses it and writes the .cpol companion (the CENN itself stays value-only,
# so every scale-check/gate tool keeps working unchanged).
D40_POLICY = os.environ.get("D40_POLICY", "0") not in ("0", "", "false", "False")
POLICY_VOCAB = 1858
# The policy head projects from its OWN bottleneck width, which is only equal
# to L2 by coincidence in the archs shipped so far (d48w/d48x both have
# pol_proj == l2 == 16). arch-hka2048 is the first with l2=32 and pol_proj=16,
# and assuming L2 here made its checkpoints unpackable under EVERY value of
# D40_POLICY: the stream over- or under-shot by exactly POLICY_VOCAB*|l2-dim|
# bytes and the PSQ1 trailer split then failed with "unrecognized D40 trailing
# payload". Defaults to L2, so every existing invocation is unchanged.
D40_POL_DIM = int(os.environ.get("D40_POL_DIM", str(D40_L2)))
D40_FT_QUANT = 255
# The scale the FullThreats table was quantised at. **A property of the NET**
# (perf study 30): a non-QAT net's weights span +-1.98 and 64 is the largest
# scale that does not clip them, while a net trained with
# `qat_threat_scale = 256` is clipped to +-127/256 and loses two bits of the i8
# if it is exported at 64 (measured on D49-sb100: max|w| 32/127, 65 of 256
# levels used). Env-overridable exactly like D40_NETWORK_SCALE; anything other
# than 64 emits a CENN-v5 container so the engine can read it back.
# MUST equal the training spec's `qat_threat_scale` when QAT is on —
# `jax/archon_jax/spec.py: ArchSpec.threat_quant()` is the single source, and
# `tools/nnue/pack_ckpt.sh`/the export path derive this from it.
D40_THREAT_QUANT = int(os.environ.get("D40_THREAT_QUANT", "64"))
D40_L1_QUANT = 64
D40_FT_SHIFT = 9
# The trainer's target is sigmoid(score / 400), so the inverse-logit runtime
# scale is 400 unless an explicitly calibrated successor overrides it.
D40_NETWORK_SCALE = 400


def _split_psq1_trailer(raw: bytes, stream_end: int):
    """Validate the quant stream's trailer starting at ``stream_end``:
    bullet's cyclic b"bullet" pad to the next 64-byte boundary, optionally
    followed by a T6c ``PSQ1`` section (jax/archon_jax/checkpoint.py:
    magic | u32 num_buckets | u32 rows | i32[rows*num_buckets]).

    Returns the PSQ1 section bytes (``b""`` when absent) or ``None`` when the
    trailer is malformed. Legacy quants (pad only) return ``b""`` — packing
    them stays byte-identical."""
    pad_len = (64 - stream_end % 64) % 64
    pad = raw[stream_end:stream_end + pad_len]
    if len(pad) != pad_len or pad != bytes(b"bullet"[i % 6] for i in range(pad_len)):
        return None
    rest = raw[stream_end + pad_len:]
    if not rest:
        return b""
    tail = rest
    if len(tail) >= 12 and tail[:4] == b"PSQ1":
        nb = int.from_bytes(tail[4:8], "little")
        rows = int.from_bytes(tail[8:12], "little")
        psq_len = 12 + rows * nb * 4
        if len(tail) < psq_len:
            return None
        tail = tail[psq_len:]
    # D47V: optional SKP1 section (output-skip lane, f32) after PSQ1 —
    # b"SKP1" | u32 num_buckets | u32 hl | f32[hl*num_buckets]. Passed
    # through verbatim with PSQ1; absent on pre-skip nets.
    if len(tail) >= 12 and tail[:4] == b"SKP1":
        snb = int.from_bytes(tail[4:8], "little")
        shl = int.from_bytes(tail[8:12], "little")
        sk_len = 12 + snb * shl * 4
        if len(tail) < sk_len:
            return None
        tail = tail[sk_len:]
    # 2026-08 audit item 1: optional OBL1 section (per-net output-bucket
    # layout, the LAST trailer) — b"OBL1" | u32 bins | u8[bins]. Passed
    # through verbatim (jax/archon_jax/checkpoint.py bucket_section_bytes;
    # engine: sota_loader.rs BUCKETS_MAGIC).
    if len(tail) >= 8 and tail[:4] == b"OBL1":
        bins = int.from_bytes(tail[4:8], "little")
        if len(tail) < 8 + bins:
            return None
        tail = tail[8 + bins:]
    if tail:
        return None
    return bytes(rest)


def pack_d40(quantised_path: str, out_path: str, network_scale: int = None,
             threat_quant: int = None) -> None:
    """Pack a bullet d40_features quantised.bin into a CENN container.

    Emits CENN-v2 at the legacy threat scale of 64 (so every existing
    invocation writes a byte-identical file) and CENN-v5 at any other, which is
    the only container that can state the scale.
    """
    if network_scale is None:
        network_scale = int(os.environ.get("D40_NETWORK_SCALE", str(D40_NETWORK_SCALE)))
    if threat_quant is None:
        threat_quant = D40_THREAT_QUANT
    # The runtime carries FT_QUANT/threat_quant as 256/threat_quant, one shift
    # (sota_head::threat_shift_for), so the scale must be a power of two <= 256.
    if threat_quant <= 0 or threat_quant > 256 or (threat_quant & (threat_quant - 1)):
        die(f"D40_THREAT_QUANT={threat_quant} must be a power of two in 1..=256")
    v5 = threat_quant != 64

    raw = Path(quantised_path).read_bytes()

    acc = int(os.environ.get("D40_ACC", str(D40_ACC)))  # width-override for grown nets (e.g. 1024)
    buckets = D40_BUCKETS

    # ── Parse bullet tensors ───────────────────────────────────────────
    # Tensor 0: l0_pstw i16 (HALFKA_DIMS × ACC) feature-major. Bullet's
    # untransposed affine storage is already input/feature-major.
    n_pst_w = acc * HALFKA_DIMS
    # Tensor 1: l0_pstb i16 (ACC)
    n_pst_b = acc
    # Tensor 2: l0_threatw i8 (THREAT_DIMS × ACC) feature-major.
    # HalfKA-only nets carry no threat tensor at all.
    n_threat_w = 0 if D40_HALFKA_ONLY else acc * THREAT_DIMS
    # Tensor 3: l1w i8 (BUCKETS*L2 × ACC) — bullet stores TRANSPOSED
    n_l1w = buckets * D40_L2 * acc
    # Tensor 4: l1b f32 (BUCKETS*L2)
    n_l1b = buckets * D40_L2
    # Tensor 5: l2w f32 (BUCKETS*L3 × L2) — TRANSPOSED
    n_l2w = buckets * D40_L3 * D40_L2
    # Tensor 6: l2b f32 (BUCKETS*L3)
    n_l2b = buckets * D40_L3
    # Tensor 7: l3w f32 (BUCKETS × L3) — TRANSPOSED
    n_l3w = buckets * D40_L3
    # Tensor 8: l3b f32 (BUCKETS)
    n_l3b = buckets

    # Byte offsets
    # D47 policy head (mid-stream, between threats and l1w when D40_POLICY=1)
    n_polw = POLICY_VOCAB * D40_POL_DIM if D40_POLICY else 0
    n_polb = POLICY_VOCAB if D40_POLICY else 0
    need_bytes = (n_pst_w + n_pst_b) * 2 + n_threat_w + n_polw + n_l1w + (n_polb + n_l1b + n_l2w + n_l2b + n_l3w + n_l3b) * 4
    if len(raw) < need_bytes:
        die(f"{quantised_path}: {len(raw)} bytes < {need_bytes} needed for D40 arch")
    # Dual-target checkpoints append one policy head after the eval tensors.
    # CENN contains eval only; a CPOL companion is produced by policy_pack.py.
    bilinear_bytes = 2 * (64 * 8 * acc + 64 * 8) * 4
    full_bytes = (4096 * acc + 4096) * 4
    psq1 = None
    policy_bytes = None
    for size in (0, bilinear_bytes, full_bytes):
        if len(raw) < need_bytes + size:
            continue
        sec = _split_psq1_trailer(raw, need_bytes + size)
        if sec is not None:
            policy_bytes = size
            psq1 = sec
            break
    if policy_bytes is None:
        die(f"{quantised_path}: unrecognized D40 trailing payload ({len(raw)-need_bytes} bytes)")
    if psq1 and psq1[:4] == b"PSQ1":
        nb = int.from_bytes(psq1[4:8], "little")
        rows = int.from_bytes(psq1[8:12], "little")
        want_rows = HALFKA_DIMS + (0 if D40_HALFKA_ONLY else THREAT_DIMS)
        if nb != D40_BUCKETS or rows != want_rows:
            die(f"{quantised_path}: PSQ1 section buckets={nb} rows={rows} "
                f"(want buckets={D40_BUCKETS} rows={want_rows})")

    off = 0

    # l0_pstw: i16 (HALFKA, ACC), already feature-major in Bullet.
    pst_w = np.frombuffer(raw, dtype="<i2", count=n_pst_w, offset=off).reshape(HALFKA_DIMS, acc)
    off += n_pst_w * 2

    # l0_pstb: i16 (ACC)
    pst_b = np.frombuffer(raw, dtype="<i2", count=n_pst_b, offset=off)
    off += n_pst_b * 2

    # l0_threatw: i8 (THREAT, ACC), also already feature-major. Absent for
    # HalfKA-only nets.
    if D40_HALFKA_ONLY:
        threat_w = None
    else:
        threat_w = np.frombuffer(raw, dtype="<i1", count=n_threat_w, offset=off).reshape(THREAT_DIMS, acc)
        off += n_threat_w

    # D47 policy head: polmovew i8 (POLICY_VOCAB, L2) row-major (TRANSPOSED),
    # then polmoveb f32 (POLICY_VOCAB) — trainer fmt order places them here.
    pol_w = pol_b = None
    if D40_POLICY:
        pol_w = np.frombuffer(raw, dtype="<i1", count=n_polw, offset=off).reshape(POLICY_VOCAB, D40_POL_DIM)
        off += n_polw
        pol_b = np.frombuffer(raw, dtype="<f4", count=n_polb, offset=off)
        off += n_polb * 4

    # l1w: i8 — bullet stores as (BUCKETS*L2, ACC) row-major (TRANSPOSED flag)
    l1w_raw = np.frombuffer(raw, dtype="<i1", count=n_l1w, offset=off).reshape(buckets * D40_L2, acc)
    off += n_l1w

    # l1b: f32 (BUCKETS*L2)
    l1b = np.frombuffer(raw, dtype="<f4", count=n_l1b, offset=off)
    off += n_l1b * 4

    # l2w: f32 — bullet stores as (BUCKETS*L3, L2) row-major (TRANSPOSED)
    l2w_raw = np.frombuffer(raw, dtype="<f4", count=n_l2w, offset=off).reshape(buckets * D40_L3, D40_L2)
    off += n_l2w * 4

    # l2b: f32 (BUCKETS*L3)
    l2b = np.frombuffer(raw, dtype="<f4", count=n_l2b, offset=off)
    off += n_l2b * 4

    # l3w: f32 — bullet stores as (BUCKETS, L3) row-major (TRANSPOSED)
    l3w_raw = np.frombuffer(raw, dtype="<f4", count=n_l3w, offset=off).reshape(buckets, D40_L3)
    off += n_l3w * 4

    # l3b: f32 (BUCKETS)
    l3b = np.frombuffer(raw, dtype="<f4", count=n_l3b, offset=off)
    off += n_l3b * 4

    assert pst_w.shape == (HALFKA_DIMS, acc), pst_w.shape
    if not D40_HALFKA_ONLY:
        assert threat_w.shape == (THREAT_DIMS, acc), threat_w.shape

    # ── Rearrange L1 weights to group-interleaved layout ───────────────
    # Engine's propagate_l1 accesses: wb[group * (L2*4) + neuron * 4 + k]
    # where group = input/4, k = input%4, neuron = output.
    # This is [L1/4][L2][4] per bucket.
    #
    # Bullet has: l1w_raw[bucket*L2 + neuron][input] (row-major TRANSPOSED).
    l1w_engine = np.zeros(buckets * D40_L2 * acc, dtype=np.int8)
    for b in range(buckets):
        for j in range(D40_L2):  # output neuron
            for i in range(acc):  # input
                g = i // 4
                k = i % 4
                src = l1w_raw[b * D40_L2 + j, i]
                dst_idx = b * D40_L2 * acc + g * (D40_L2 * 4) + j * 4 + k
                l1w_engine[dst_idx] = src

    # ── Rearrange L2 weights ───────────────────────────────────────────
    # Engine accesses: l2_weights[bucket * L2 * L3 + i * L3 + j]
    #   where i=L2 input, j=L3 output.
    # Bullet has: l2w_raw[bucket*L3 + j][i] (TRANSPOSED).
    l2w_engine = np.zeros(buckets * D40_L2 * D40_L3, dtype=np.float32)
    for b in range(buckets):
        for i in range(D40_L2):
            for j in range(D40_L3):
                src = l2w_raw[b * D40_L3 + j, i]
                dst_idx = b * D40_L2 * D40_L3 + i * D40_L3 + j
                l2w_engine[dst_idx] = src

    # ── Rearrange L3 weights ───────────────────────────────────────────
    # Engine accesses: l3_weights[bucket * L3 + i] (one output per bucket).
    # Bullet has: l3w_raw[bucket][i] (TRANSPOSED: output=bucket, input=i).
    # This is already the right layout when flattened: [bucket*L3 + i].
    l3w_engine = l3w_raw.ravel()

    # ── Threat i8 range gate (perf study 30) ───────────────────────────
    # The bug this catches, in full: D49 trained 100 superbatches with
    # `qat_threat_scale = 256`, which CLIPS the threat table to +-127/256, and
    # every deployment path then exported it at 64. The i8 weights reached
    # max|w| = 32 of 127 and used 65 of 256 levels — two bits thrown away — for
    # a measured -128.7 +- 9.6 Elo over 1600 calibrated games, while the FLOAT heldout R2
    # stayed within 0.008 of the incumbent. Nothing except games could see it.
    #
    # An i8 table that does not reach even half the range was quantised at
    # least 2x too coarsely. That is not a judgement call, it is arithmetic, so
    # this refuses rather than warns.
    if not D40_HALFKA_ONLY:
        peak = int(np.abs(threat_w.astype(np.int32)).max())
        if peak * 2 <= 127 and os.environ.get("D40_ALLOW_THREAT_UNDERRANGE", "0") in (
                "0", "", "false", "False"):
            want = threat_quant
            while want < 256 and peak * 2 <= 127:
                want *= 2
                peak *= 2
            die(f"{quantised_path}: threat i8 peaks at "
                f"{int(np.abs(threat_w.astype(np.int32)).max())}/127 at "
                f"threat_quant={threat_quant} — at least one bit of the i8 is "
                f"unused, which is what a QAT net exported at the legacy scale "
                f"looks like (perf study 30: -128.7 Elo, invisible to R2).\n"
                f"  The quant.bin ITSELF is the wrong one: re-export it at the "
                f"trainer's qat_threat_scale (jax ArchSpec.threat_quant(); "
                f"tools/nnue/requant_threats.py --scale N --run-spec ...) and "
                f"pack that with D40_THREAT_QUANT=N. N is at least {want} on "
                f"this table's range alone; the RUN SPEC decides the real "
                f"value. Set D40_ALLOW_THREAT_UNDERRANGE=1 if the table really "
                f"is this small.")

    # ── Build CENN-v2 payload ──────────────────────────────────────────
    payload = bytearray()
    payload += pst_w.astype("<i2").tobytes()         # halfka_w feature-major i16
    payload += pst_b.astype("<i2").tobytes()          # ft_bias i16
    if not D40_HALFKA_ONLY:
        payload += threat_w.astype("<i1").tobytes()   # threat_w feature-major i8
    payload += l1w_engine.tobytes()                   # l1_w i8 group-interleaved
    payload += l1b.astype("<f4").tobytes()             # l1_b f32
    payload += l2w_engine.astype("<f4").tobytes()      # l2_w f32
    payload += l2b.astype("<f4").tobytes()             # l2_b f32
    payload += l3w_engine.astype("<f4").tobytes()      # l3_w f32
    payload += l3b.astype("<f4").tobytes()             # l3_b f32

    checksum = fnv1a64(bytes(payload))

    # ── Build the CENN header (64 bytes v2, 72 bytes v5) ───────────────
    # v5 keeps bytes 0..64 identical in meaning and APPENDS, so this is one
    # emitter with a two-field tail.
    flags = 0 if D40_HALFKA_ONLY else FLAG_HAS_THREATS
    header = bytearray()
    header += MAGIC_V5 if v5 else MAGIC_V2
    header += int(CENN_V5_VERSION if v5 else CENN_V2_VERSION).to_bytes(2, "little")
    header += int(flags).to_bytes(2, "little")
    header += int(HALFKA_DIMS).to_bytes(4, "little")
    header += int(0 if D40_HALFKA_ONLY else THREAT_DIMS).to_bytes(4, "little")
    header += int(acc).to_bytes(4, "little")           # acc_width (width-override aware)
    header += int(D40_L2).to_bytes(4, "little")
    header += int(D40_L3).to_bytes(4, "little")
    header += int(D40_BUCKETS).to_bytes(4, "little")
    header += int(D40_FT_QUANT).to_bytes(4, "little", signed=True)
    header += int(D40_L1_QUANT).to_bytes(4, "little", signed=True)
    header += int(D40_FT_SHIFT).to_bytes(4, "little", signed=True)
    header += int(network_scale).to_bytes(4, "little", signed=True)
    header += len(payload).to_bytes(8, "little")
    header += checksum.to_bytes(8, "little")
    if v5:
        header += int(threat_quant).to_bytes(4, "little", signed=True)
        header += (0).to_bytes(4, "little")  # reserved, MUST be zero
    want_header = HEADER_V5_LEN if v5 else HEADER_V2_LEN
    assert len(header) == want_header, f"header {len(header)} != {want_header}"

    with open(out_path, "wb") as f:
        f.write(header)
        f.write(payload)
        # T6c/OBL1: pass the trailer chain (PSQ1 + SKP1 + OBL1, whichever
        # exist) through VERBATIM, after the checksummed
        # payload (mirroring its position after the padded bullet stream).
        # Absent from the source ⇒ absent here ⇒ legacy .cenn byte-identical.
        if psq1:
            f.write(psq1)

    total = want_header + len(payload) + (len(psq1) if psq1 else 0)
    print(f"netpack-d40: wrote {out_path} ({total:,} bytes, "
          f"{'CENN-v5' if v5 else 'CENN-v2'})")
    if psq1 and psq1[:4] == b"PSQ1":
        print(f"  psqt        PSQ1 section passed through ({len(psq1):,} bytes, "
              f"{int.from_bytes(psq1[8:12], 'little')} rows x "
              f"{int.from_bytes(psq1[4:8], 'little')} buckets)")
    if psq1 and b"OBL1" == psq1[-41:][:4]:
        print(f"  buckets     OBL1 output-bucket layout passed through "
              f"({list(psq1[-33:])})")

    # D47: emit the .cpol companion (policy rows OUTSIDE the CENN so every
    # value-net tool — scale-check, gates, loaders — is untouched).
    # Format CPL1: magic + u32 vocab + u32 dim + i8 rows[vocab][dim] (row-major,
    # canonical-frame move_id order per policy_vocab.rs) + f32 bias[vocab].
    if D40_POLICY:
        cpol_path = os.environ.get("D40_CPOL_OUT") or str(Path(out_path).with_suffix(".cpol"))
        with open(cpol_path, "wb") as f:
            f.write(b"CPL1")
            f.write(POLICY_VOCAB.to_bytes(4, "little"))
            f.write(int(D40_POL_DIM).to_bytes(4, "little"))
            f.write(pol_w.astype("<i1").tobytes())
            f.write(pol_b.astype("<f4").tobytes())
        print(f"netpack-d40: wrote {cpol_path} ({12 + pol_w.size + pol_b.size*4:,} bytes, vocab {POLICY_VOCAB} x {D40_POL_DIM})")
    threats_desc = "OFF (HalfKA-only)" if D40_HALFKA_ONLY else THREAT_DIMS
    print(f"  arch        halfka={HALFKA_DIMS} threats={threats_desc} acc={acc}")
    print(f"              L2={D40_L2} L3={D40_L3} buckets={D40_BUCKETS}")
    print(f"  quant       ft_quant={D40_FT_QUANT} threat_quant={threat_quant} "
          f"(merge shift {8 - threat_quant.bit_length() + 1}) "
          f"l1_quant={D40_L1_QUANT} "
          f"ft_shift={D40_FT_SHIFT} scale={network_scale}")
    print(f"  payload     {len(payload):,} bytes   checksum {checksum:#018x}")


def strip_d40_threats(cenn_path: str, out_path: str) -> None:
    """Create a diagnostic HalfKA-only CENN-v2 while preserving its head.

    This is an inference-cost isolation tool, not a strength network: removing
    a trained threat transformer changes its evaluations. The resulting file
    remains a fully checksummed, loader-valid CENN-v2 container.
    """
    raw = bytearray(Path(cenn_path).read_bytes())
    if len(raw) < HEADER_V2_LEN or raw[:4] not in (MAGIC_V2, MAGIC_V5):
        die(f"{cenn_path}: not a CENN-v2/v5 container")
    hdr = HEADER_V5_LEN if raw[:4] == MAGIC_V5 else HEADER_V2_LEN
    if len(raw) < hdr:
        die(f"{cenn_path}: truncated CENN-v5 header")
    flags = int.from_bytes(raw[6:8], "little")
    threat_dims = int.from_bytes(raw[12:16], "little")
    if not flags & FLAG_HAS_THREATS or threat_dims != THREAT_DIMS:
        die(f"{cenn_path}: expected a {THREAT_DIMS}-feature threat channel")
    payload_len = int.from_bytes(raw[48:56], "little")
    payload = raw[hdr:]
    if len(payload) != payload_len:
        die(f"{cenn_path}: payload length mismatch")
    prefix = HALFKA_DIMS * D40_ACC * 2 + D40_ACC * 2
    threat_bytes = THREAT_DIMS * D40_ACC
    payload = payload[:prefix] + payload[prefix + threat_bytes:]
    raw[6:8] = (flags & ~FLAG_HAS_THREATS).to_bytes(2, "little")
    raw[12:16] = (0).to_bytes(4, "little")
    raw[48:56] = len(payload).to_bytes(8, "little")
    raw[56:64] = fnv1a64(bytes(payload)).to_bytes(8, "little")
    Path(out_path).write_bytes(raw[:hdr] + payload)
    print(f"netpack-d40: wrote diagnostic HalfKA-only {out_path} ({hdr + len(payload):,} bytes)")


def main():
    if len(sys.argv) == 4 and sys.argv[1] == "--strip-d40-threats":
        strip_d40_threats(sys.argv[2], sys.argv[3])
        return
    if len(sys.argv) < 3:
        die("usage: netpack.py <quantised.bin> <out.cenn> [screlu|crelu|d40]\n"
            "       netpack.py --strip-d40-threats <in.cenn> <out.cenn>")
    mode = sys.argv[3] if len(sys.argv) >= 4 else "screlu"

    if mode.lower() == "d40":
        pack_d40(sys.argv[1], sys.argv[2])
    else:
        pack(sys.argv[1], sys.argv[2], mode)


if __name__ == "__main__":
    main()
