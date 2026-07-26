#!/usr/bin/env python3
"""Patch a CENN net's `network_scale` header field (offset 44, i32).

The engine converts the head's final activation to centipawns with
`l3 * network_scale` (sota_head.rs:350; header layout sota_loader.rs:25).
Changing it rescales the entire eval output — which is exactly what the
2026-07-25 calibration finding calls for: our evals measured 0.587x
SF18's cp scale, so every cp-denominated search margin (futility, RFP,
delta, aspiration) is effectively ~1.7x wider than the constants intend.

  python3 tools/nnue/rescale_net_header.py in.cenn out.cenn 681

NOTE: our shipped nets carry network_scale = 400 (the 380 in sota_head.rs:431
is only the code default). Scale-matched to SF18 => 400 / 0.587 ~= 681.

Weights are untouched — only the output scale changes. Copy the sibling
.cpol yourself if the policy head is wanted (the head reads the L2
bottleneck and is unaffected by this constant).
"""
import shutil
import struct
import sys

OFFSET = 44


def main() -> int:
    if len(sys.argv) != 4:
        print(__doc__)
        return 2
    src, dst, new = sys.argv[1], sys.argv[2], int(sys.argv[3])
    shutil.copyfile(src, dst)
    with open(dst, "r+b") as f:
        f.seek(OFFSET)
        old = struct.unpack("<i", f.read(4))[0]
        f.seek(OFFSET)
        f.write(struct.pack("<i", new))
    print(f"{dst}: network_scale {old} -> {new}  (ratio {new/old:.3f})")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
