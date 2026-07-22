"""Exporter CLI: raw <-> quant, byte-compatible with bullet (proven byte-identical
on the real D47c sb240 checkpoint — tools/quant_byte_parity.py).

Usage:
  python -m archon_jax.export quant  <raw.bin> <quant.bin>   # raw f32 -> quant
  python -m archon_jax.export dequant <quant.bin> <raw.bin>  # quant -> raw f32
                                                             # (lossy source; for probes)
The output quant.bin is a drop-in netpack.py input:
  D40_ACC=1280 D40_L2=128 D40_L3=32 D40_POLICY=1 python tools/nnue/netpack.py \
      out-quant.bin out.cenn d40
"""

from __future__ import annotations

import sys

from . import checkpoint
from .spec import ArchSpec


def main(argv):
    if len(argv) != 4 or argv[1] not in ("quant", "dequant"):
        print(__doc__)
        return 2
    arch = ArchSpec.full()
    if argv[1] == "quant":
        params = checkpoint.open_raw(argv[2], arch.param_shapes(), mmap=True)
        n = checkpoint.write_quant(argv[3], params, arch)
        print(f"wrote {argv[3]} ({n:,} bytes)")
    else:
        params = checkpoint.dequantise_quant(argv[2], arch)
        checkpoint.write_raw(argv[3], params)
        print(f"wrote {argv[3]}")
    return 0


if __name__ == "__main__":
    sys.exit(main(sys.argv))
