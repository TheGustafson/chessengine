#!/usr/bin/env python3
"""Pack Bullet KingBucket768->2x256->1 SCReLU weights into CENN-v3."""
from __future__ import annotations

import argparse
from pathlib import Path
import struct

HALFKA = 64 * 768
ACC = 256
HEADER = 48


def fnv(data: bytes) -> int:
    value = 0xCBF29CE484222325
    for byte in data:
        value = ((value ^ byte) * 0x100000001B3) & 0xFFFFFFFFFFFFFFFF
    return value


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("input", type=Path)
    parser.add_argument("output", type=Path)
    args = parser.parse_args()
    raw = args.input.read_bytes()
    body_bytes = 2 * (HALFKA * ACC + ACC + 2 * ACC + 1)
    if len(raw) < body_bytes or len(raw) - body_bytes > 63:
        raise SystemExit(f"unexpected Bullet length {len(raw)} (body {body_bytes})")
    padding = raw[body_bytes:]
    if any(byte != b"bullet"[i % 6] for i, byte in enumerate(padding)):
        raise SystemExit("invalid Bullet alignment padding")
    offset = 2 * (HALFKA * ACC + ACC)
    out_weights = struct.unpack_from(f"<{2 * ACC}h", raw, offset)
    if max(map(abs, out_weights)) > 127:
        raise SystemExit("output weight exceeds classic AVX2 clip")
    offset += 4 * ACC
    out_bias_i16 = struct.unpack_from("<h", raw, offset)[0]
    payload = raw[:offset] + struct.pack("<i", out_bias_i16)
    header = bytearray(HEADER)
    struct.pack_into("<4sHHIIiiiIQQ", header, 0, b"CEN3", 3, 0, HALFKA, ACC,
                     255, 64, 400, 0, len(payload), fnv(payload))
    args.output.parent.mkdir(parents=True, exist_ok=True)
    args.output.write_bytes(header + payload)
    print(f"wrote {args.output} ({len(header) + len(payload)} bytes), max|out|={max(map(abs, out_weights))}")


if __name__ == "__main__":
    main()
