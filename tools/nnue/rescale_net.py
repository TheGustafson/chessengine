#!/usr/bin/env python3
"""Rescale a CENN net's output layer by a constant factor k (hot-scale recalib).

The NNUE forward output is LINEAR in the output-layer weights (design/11
§3.4.3, §3.4.6):

    v = (Σ screlu(acc)·out_weight / QA + out_bias) · SCALE / (QA·QB)

so multiplying every ``out_weight`` AND ``out_bias`` by a factor ``k`` scales
the reported eval by ``k`` — a pure data transform, no retrain, no code change.
The header ``SCALE`` stays 400 (the engine uses the compile-time constant); the
whole recalibration lives in the smaller output weights.

net-0 evaluates ~2.5-3x "hot" vs a normal cp scale (startpos +125, a knight up
~+995 where ~300 is typical). Our search's pruning margins (RFP / futility /
razoring / delta) are cp-tuned for the PSQT scale, so a calibrated (lower) eval
scale MIGHT gain Elo. ``k < 1`` shrinks the eval; it also keeps the already-
maxed ``|out_weight| = 127`` inside the AVX2 SCReLU clip ``|w| <= 127`` the
loader enforces (loader.rs ``OutWeightOutOfClip``), so the rescaled net still
loads.

CENN format (design/11 §3.4.5, mirrors tools/nnue/netpack.py):
    48-byte header, then payload (all little-endian):
        ft_weights : i16[768*256]   (unchanged)
        ft_bias    : i16[256]       (unchanged)
        out_weights: i16[512]       *= k, round-nearest, clamp to +-127
        out_bias   : i32            *= k, round-nearest
    header[40:48] = FNV-1a-64 of the payload (recomputed here).

Only ``out_weights`` and ``out_bias`` change; every other byte (header padding,
ft_* payload) is preserved verbatim, and the checksum is recomputed over the
edited payload.

PRECISION CAVEAT: the rescaled ``out_weights`` are re-rounded integers, so the
transform is k-scaling *up to* a per-weight rounding error of <= 0.5 quant unit
(bounded, but the calibrated net is NOT a bit-exact k-multiple of net-0).

Stdlib only (no deps). Run under uv or plain python3 (design/11 §11.1):
    uv run python tools/nnue/rescale_net.py <in.cenn> <out.cenn> <k>

Data/tooling only (D15/D28): reads a gitignored net, writes a gitignored net.
No engine (``src/``) code is touched.
"""

import struct
import sys

# Architecture + quantization — must match src/eval/nnue/mod.rs and the loader
# header (design/11 §3.4.1, §3.4.3, §3.4.5).
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
N_OUT_W = 2 * L1     # 512  (stm half first)
# ... then one i32 out_bias.

# Byte offsets into the payload (i16 = 2 bytes each).
OFF_FT_W = 0
OFF_FT_B = OFF_FT_W + N_FT_W * 2       # 393216
OFF_OUT_W = OFF_FT_B + N_FT_B * 2      # 393728
OFF_OUT_B = OFF_OUT_W + N_OUT_W * 2    # 394752
PAYLOAD_LEN = OFF_OUT_B + 4            # 394756

# The AVX2 SCReLU clip the loader enforces (design/11 §3.4.3): |out_weight|<=127.
OUT_WEIGHT_CLIP = 127


def die(msg: str):
    print(f"rescale_net: {msg}", file=sys.stderr)
    sys.exit(2)


def fnv1a64(data: bytes) -> int:
    """FNV-1a 64-bit (design/11 §3.4.5 payload checksum). Matches loader.rs."""
    h = 0xCBF29CE484222325
    for byte in data:
        h ^= byte
        h = (h * 0x100000001B3) & 0xFFFFFFFFFFFFFFFF
    return h


def round_half_away(x: float) -> int:
    """Round to nearest integer, ties away from zero (symmetric for +/-)."""
    import math

    return math.floor(x + 0.5) if x >= 0.0 else math.ceil(x - 0.5)


def rescale(in_path: str, out_path: str, k: float) -> None:
    raw = open(in_path, "rb").read()

    # --- header sanity (design/11 §3.4.5) ------------------------------------
    if len(raw) < HEADER_LEN + PAYLOAD_LEN:
        die(f"{in_path}: {len(raw)} bytes < {HEADER_LEN + PAYLOAD_LEN} "
            f"(header + payload for {INPUT}->2x{L1}->1)")
    if raw[0:4] != MAGIC:
        die(f"bad magic {raw[0:4]!r} (expected {MAGIC!r})")
    version = struct.unpack_from("<I", raw, 4)[0]
    if version != CENN_VERSION:
        die(f"unsupported version {version} (this tool writes v{CENN_VERSION})")
    input_dims = struct.unpack_from("<I", raw, 8)[0]
    l1 = struct.unpack_from("<I", raw, 12)[0]
    if input_dims != INPUT or l1 != L1:
        die(f"arch {input_dims}->2x{l1} != {INPUT}->2x{L1} (this tool's arch)")
    payload_len = struct.unpack_from("<Q", raw, 32)[0]
    if payload_len != PAYLOAD_LEN:
        die(f"payload_len {payload_len} != {PAYLOAD_LEN} (arch mismatch)")

    header = bytearray(raw[:HEADER_LEN])
    payload = bytearray(raw[HEADER_LEN:HEADER_LEN + PAYLOAD_LEN])

    # --- scale out_weights (i16[512]) — round-nearest, clamp to +-127 --------
    out_w = list(struct.unpack_from(f"<{N_OUT_W}h", payload, OFF_OUT_W))
    clamped = 0
    max_abs_before = max(abs(w) for w in out_w)
    new_out_w = []
    for w in out_w:
        v = round_half_away(w * k)
        if v > OUT_WEIGHT_CLIP:
            v = OUT_WEIGHT_CLIP
            clamped += 1
        elif v < -OUT_WEIGHT_CLIP:
            v = -OUT_WEIGHT_CLIP
            clamped += 1
        new_out_w.append(v)
    struct.pack_into(f"<{N_OUT_W}h", payload, OFF_OUT_W, *new_out_w)
    max_abs_after = max(abs(w) for w in new_out_w)

    # --- scale out_bias (i32) — round-nearest --------------------------------
    old_bias = struct.unpack_from("<i", payload, OFF_OUT_B)[0]
    new_bias = round_half_away(old_bias * k)
    lo, hi = -(1 << 31), (1 << 31) - 1
    if not (lo <= new_bias <= hi):
        die(f"out_bias {new_bias} overflows i32 (should never happen for k<1)")
    struct.pack_into("<i", payload, OFF_OUT_B, new_bias)

    # --- recompute checksum + patch header -----------------------------------
    checksum = fnv1a64(bytes(payload))
    struct.pack_into("<Q", header, 40, checksum)

    with open(out_path, "wb") as f:
        f.write(header)
        f.write(payload)

    print(f"rescale_net: wrote {out_path} ({HEADER_LEN + PAYLOAD_LEN} bytes)")
    print(f"  k            {k}")
    print(f"  out_weights  max|w| {max_abs_before} -> {max_abs_after} "
          f"(clip {OUT_WEIGHT_CLIP}; {clamped} clamped)")
    print(f"  out_bias     {old_bias} -> {new_bias}")
    print(f"  payload_len  {PAYLOAD_LEN}   checksum {checksum:#018x}")


def main():
    if len(sys.argv) != 4:
        die("usage: rescale_net.py <in.cenn> <out.cenn> <k>")
    try:
        k = float(sys.argv[3])
    except ValueError:
        die(f"bad k '{sys.argv[3]}' (want a float, e.g. 0.33)")
    if k <= 0.0:
        die(f"k must be > 0 (got {k})")
    rescale(sys.argv[1], sys.argv[2], k)


if __name__ == "__main__":
    main()
