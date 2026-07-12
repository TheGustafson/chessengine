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

import sys

import numpy as np

# Architecture + quantization — must match src/eval/nnue/mod.rs and the
# loader header (design/11 §3.4.1, §3.4.3, §3.4.5).
INPUT = 768
L1 = 256
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

    raw = open(quantised_path, "rb").read()
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


def main():
    if len(sys.argv) < 3:
        die("usage: netpack.py <quantised.bin> <out.cenn> [screlu|crelu]")
    activation = sys.argv[3] if len(sys.argv) >= 4 else "screlu"
    pack(sys.argv[1], sys.argv[2], activation)


if __name__ == "__main__":
    main()
