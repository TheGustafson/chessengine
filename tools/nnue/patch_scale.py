#!/usr/bin/env python3
"""Patch a CENN container's `network_scale` in place-by-copy.

WHY THIS EXISTS (2026-07-30). `tools/nnue/rescale_net.py` refuses current
containers: it asserts `len(file) == header + payload` and dies on a gen-2 net
because netpack APPENDS a PSQ1 section (2,704,920 bytes) after the payload.
That check is right for what it does -- it rescales WEIGHTS -- but it blocks the
much cheaper thing: `network_scale` is a single i32 in the header, and changing
it needs no weight touch at all.

Measured 2026-07-30: on D49 sb180, scale 400 -> 756 is worth **+32.09 +/- 11.38
Elo over 1466 games** with a byte-identical payload. The frame is a real,
large, and until now completely unswept lever.

Layout (design/11; CEN2 and CEN5 share 0..64):
    0   magic  CEN2 | CEN5
    32  ft_quant     36  l1_quant     40  ft_shift
    44  network_scale (i32 LE)        48  payload_len
"""
import shutil, struct, sys

OFF = 44

def main() -> int:
    if len(sys.argv) not in (3, 4):
        print("usage: patch_scale.py <in.cenn> <out.cenn> <new_scale>   "
              "(omit new_scale to just READ it)", file=sys.stderr)
        return 2
    src, dst = sys.argv[1], sys.argv[2]
    with open(src, "rb") as f:
        head = f.read(64)
    magic, old = head[0:4], struct.unpack_from("<i", head, OFF)[0]
    if magic not in (b"CEN2", b"CEN5"):
        print(f"error: bad magic {magic!r} (want CEN2/CEN5)", file=sys.stderr)
        return 1
    if len(sys.argv) == 3:
        print(f"{src}: magic={magic.decode()} network_scale={old}")
        return 0
    new = int(sys.argv[3])
    shutil.copyfile(src, dst)
    with open(dst, "r+b") as f:
        f.seek(OFF); f.write(struct.pack("<i", new))
    with open(dst, "rb") as f:
        chk = struct.unpack_from("<i", f.read(64), OFF)[0]
    if chk != new:
        print(f"error: readback {chk} != {new}", file=sys.stderr)
        return 1
    print(f"{dst}: magic={magic.decode()} network_scale {old} -> {new} "
          f"(payload untouched)")
    return 0

if __name__ == "__main__":
    raise SystemExit(main())
