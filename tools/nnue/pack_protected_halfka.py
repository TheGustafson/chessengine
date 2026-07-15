#!/usr/bin/env python3
"""Pack a frozen CENN-v1 base and exact-HalfKA residual into CENN-v4."""

from __future__ import annotations

import argparse
from pathlib import Path
import struct


HALFKA = 22_528
ACC = 256
HEADER = 64


def fnv(data: bytes) -> int:
    value = 0xCBF29CE484222325
    for byte in data:
        value = ((value ^ byte) * 0x100000001B3) & 0xFFFFFFFFFFFFFFFF
    return value


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("base", type=Path, help="validated CENN-v1 frozen evaluator")
    parser.add_argument("residual", type=Path, help="Bullet quantised residual tensors")
    parser.add_argument("output", type=Path)
    parser.add_argument(
        "--residual-scale",
        type=float,
        default=1.0,
        help="multiply the quantised residual output layer and bias (default: 1)",
    )
    args = parser.parse_args()
    if not 0.0 <= args.residual_scale <= 1.0:
        parser.error("--residual-scale must be between zero and one")

    base = args.base.read_bytes()
    if base[:4] != b"CENN":
        parser.error("base must be a CENN-v1 classic evaluator")
    raw = args.residual.read_bytes()
    i16_count = HALFKA * ACC + ACC + 2 * ACC + 1
    body_bytes = 2 * i16_count
    if len(raw) < body_bytes or len(raw) - body_bytes > 63:
        parser.error(f"unexpected residual length {len(raw)} (body {body_bytes})")
    padding = raw[body_bytes:]
    if any(byte != b"bullet"[i % 6] for i, byte in enumerate(padding)):
        parser.error("invalid Bullet alignment padding")

    out_offset = 2 * (HALFKA * ACC + ACC)
    out_weights = struct.unpack_from(f"<{2 * ACC}h", raw, out_offset)
    scaled_weights = tuple(round(value * args.residual_scale) for value in out_weights)
    raw = raw[:out_offset] + struct.pack(f"<{2 * ACC}h", *scaled_weights) + raw[
        out_offset + 2 * (2 * ACC):
    ]
    out_weights = scaled_weights
    max_out = max(map(abs, out_weights))
    if max_out > 127:
        parser.error(f"residual output |weight| {max_out} exceeds 127")
    bias_offset = out_offset + 2 * (2 * ACC)
    out_bias = round(struct.unpack_from("<h", raw, bias_offset)[0] * args.residual_scale)
    residual = raw[:bias_offset] + struct.pack("<i", out_bias)
    payload = base + residual
    header = bytearray(HEADER)
    struct.pack_into(
        "<4sHHIIiiiIQQQ",
        header,
        0,
        b"CEN4",
        4,
        0,
        HALFKA,
        ACC,
        255,
        64,
        400,
        0,
        len(base),
        len(residual),
        fnv(payload),
    )
    args.output.parent.mkdir(parents=True, exist_ok=True)
    args.output.write_bytes(header + payload)
    print(
        f"wrote {args.output} ({HEADER + len(payload)} bytes), "
        f"base={len(base)} residual={len(residual)} max|out|={max_out} "
        f"residual_scale={args.residual_scale:g}"
    )


if __name__ == "__main__":
    main()
