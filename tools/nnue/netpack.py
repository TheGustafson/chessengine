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
FLAG_HAS_THREATS = 1

# D40 architecture constants — must match src/eval/nnue/sota_head.rs +
# sota_features.rs.
HALFKA_DIMS = 22528
THREAT_DIMS = 60720
D40_ACC = 768      # Accumulator width (= SotaHead::L1)
D40_L2 = 16
D40_L3 = 32
D40_BUCKETS = 8
D40_FT_QUANT = 255
D40_THREAT_QUANT = 64
D40_L1_QUANT = 64
D40_FT_SHIFT = 9
# The trainer's target is sigmoid(score / 400), so the inverse-logit runtime
# scale is 400 unless an explicitly calibrated successor overrides it.
D40_NETWORK_SCALE = 400


def pack_d40(quantised_path: str, out_path: str, network_scale: int = None) -> None:
    """Pack a bullet d40_features quantised.bin into CENN-v2."""
    if network_scale is None:
        network_scale = int(os.environ.get("D40_NETWORK_SCALE", str(D40_NETWORK_SCALE)))

    raw = Path(quantised_path).read_bytes()

    acc = D40_ACC
    buckets = D40_BUCKETS

    # ── Parse bullet tensors ───────────────────────────────────────────
    # Tensor 0: l0_pstw i16 (HALFKA_DIMS × ACC) feature-major. Bullet's
    # untransposed affine storage is already input/feature-major.
    n_pst_w = acc * HALFKA_DIMS
    # Tensor 1: l0_pstb i16 (ACC)
    n_pst_b = acc
    # Tensor 2: l0_threatw i8 (THREAT_DIMS × ACC) feature-major.
    n_threat_w = acc * THREAT_DIMS
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
    need_bytes = (n_pst_w + n_pst_b) * 2 + n_threat_w + n_l1w + (n_l1b + n_l2w + n_l2b + n_l3w + n_l3b) * 4
    if len(raw) < need_bytes:
        die(f"{quantised_path}: {len(raw)} bytes < {need_bytes} needed for D40 arch")
    padding = raw[need_bytes:]
    expected_padding = bytes(b"bullet"[i % 6] for i in range(len(padding)))
    if len(padding) > 63 or padding != expected_padding:
        die(f"{quantised_path}: invalid trailing data after D40 tensors "
            f"({len(padding)} bytes; expected Bullet's <=63-byte padding)")

    off = 0

    # l0_pstw: i16 (HALFKA, ACC), already feature-major in Bullet.
    pst_w = np.frombuffer(raw, dtype="<i2", count=n_pst_w, offset=off).reshape(HALFKA_DIMS, acc)
    off += n_pst_w * 2

    # l0_pstb: i16 (ACC)
    pst_b = np.frombuffer(raw, dtype="<i2", count=n_pst_b, offset=off)
    off += n_pst_b * 2

    # l0_threatw: i8 (THREAT, ACC), also already feature-major.
    threat_w = np.frombuffer(raw, dtype="<i1", count=n_threat_w, offset=off).reshape(THREAT_DIMS, acc)
    off += n_threat_w

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

    # ── Build CENN-v2 payload ──────────────────────────────────────────
    payload = bytearray()
    payload += pst_w.astype("<i2").tobytes()         # halfka_w feature-major i16
    payload += pst_b.astype("<i2").tobytes()          # ft_bias i16
    payload += threat_w.astype("<i1").tobytes()       # threat_w feature-major i8
    payload += l1w_engine.tobytes()                   # l1_w i8 group-interleaved
    payload += l1b.astype("<f4").tobytes()             # l1_b f32
    payload += l2w_engine.astype("<f4").tobytes()      # l2_w f32
    payload += l2b.astype("<f4").tobytes()             # l2_b f32
    payload += l3w_engine.astype("<f4").tobytes()      # l3_w f32
    payload += l3b.astype("<f4").tobytes()             # l3_b f32

    checksum = fnv1a64(bytes(payload))

    # ── Build CENN-v2 header (64 bytes) ────────────────────────────────
    flags = FLAG_HAS_THREATS  # always has threats for d40_features
    header = bytearray()
    header += MAGIC_V2
    header += int(CENN_V2_VERSION).to_bytes(2, "little")
    header += int(flags).to_bytes(2, "little")
    header += int(HALFKA_DIMS).to_bytes(4, "little")
    header += int(THREAT_DIMS).to_bytes(4, "little")
    header += int(D40_ACC).to_bytes(4, "little")       # acc_width
    header += int(D40_L2).to_bytes(4, "little")
    header += int(D40_L3).to_bytes(4, "little")
    header += int(D40_BUCKETS).to_bytes(4, "little")
    header += int(D40_FT_QUANT).to_bytes(4, "little", signed=True)
    header += int(D40_L1_QUANT).to_bytes(4, "little", signed=True)
    header += int(D40_FT_SHIFT).to_bytes(4, "little", signed=True)
    header += int(network_scale).to_bytes(4, "little", signed=True)
    header += len(payload).to_bytes(8, "little")
    header += checksum.to_bytes(8, "little")
    assert len(header) == HEADER_V2_LEN, f"header {len(header)} != {HEADER_V2_LEN}"

    with open(out_path, "wb") as f:
        f.write(header)
        f.write(payload)

    total = HEADER_V2_LEN + len(payload)
    print(f"netpack-d40: wrote {out_path} ({total:,} bytes)")
    print(f"  arch        halfka={HALFKA_DIMS} threats={THREAT_DIMS} acc={D40_ACC}")
    print(f"              L2={D40_L2} L3={D40_L3} buckets={D40_BUCKETS}")
    print(f"  quant       ft_quant={D40_FT_QUANT} threat_quant={D40_THREAT_QUANT} "
          f"l1_quant={D40_L1_QUANT} "
          f"ft_shift={D40_FT_SHIFT} scale={network_scale}")
    print(f"  payload     {len(payload):,} bytes   checksum {checksum:#018x}")


def main():
    if len(sys.argv) < 3:
        die("usage: netpack.py <quantised.bin> <out.cenn> [screlu|crelu|d40]")
    mode = sys.argv[3] if len(sys.argv) >= 4 else "screlu"

    if mode.lower() == "d40":
        pack_d40(sys.argv[1], sys.argv[2])
    else:
        pack(sys.argv[1], sys.argv[2], mode)


if __name__ == "__main__":
    main()
