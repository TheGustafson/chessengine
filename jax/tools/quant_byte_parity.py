"""THE core parity proof: re-quantise the real D47c raw checkpoint through the
JAX pipeline's exporter and compare byte-for-byte against the quant.bin that
bullet itself wrote from the same weights at the same superbatch.

If every byte matches, then ALL of the following are proven simultaneously:
  - the raw-checkpoint parse (names, order, offsets, counts),
  - the (fan_in, fan_out) storage convention == bullet's internal col-major,
  - the SavedFormat tensor order incl. the mid-stream policy head,
  - the transpose semantics for l1w/l2w/l3w/polmovew,
  - the i16(255)/i8(64) round-half-away-from-zero quantisation,
  - the 64-byte 'bullet' padding.

Memory-safe: raw tensors stay memmapped; the exporter streams in chunks.

Run (read-only on the experiment dir):
  cd jax && uv run --no-project --with numpy -- python tools/quant_byte_parity.py \
      /path/to/...-sbN-raw.bin /path/to/...-sbN-quant.bin
"""

import sys
from pathlib import Path

import numpy as np

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))

from archon_jax.checkpoint import open_raw, quant_bytes_iter  # noqa: E402
from archon_jax.spec import ArchSpec  # noqa: E402


def main(raw_path: str, quant_path: str) -> int:
    arch = ArchSpec.full()
    params = open_raw(raw_path, arch.param_shapes(), mmap=True)
    print(f"raw tensors: {sorted(params)}")

    official = np.memmap(quant_path, dtype=np.uint8, mode="r")
    pos = 0
    mismatches = 0
    for buf in quant_bytes_iter(params, arch):
        chunk = np.frombuffer(buf, dtype=np.uint8)
        ref = official[pos:pos + len(chunk)]
        if len(ref) != len(chunk) or not np.array_equal(chunk, ref):
            bad = int(np.argmax(chunk[: len(ref)] != ref)) if len(ref) else 0
            print(f"MISMATCH at byte {pos + bad} "
                  f"(ours {chunk[bad]:#x} vs official {ref[bad]:#x})")
            mismatches += 1
            if mismatches > 5:
                return 1
        pos += len(chunk)
    if pos != len(official):
        print(f"LENGTH MISMATCH: ours {pos} vs official {len(official)}")
        return 1
    if mismatches:
        return 1
    print(f"BYTE-IDENTICAL: {pos:,} bytes ({pos / 1e6:.1f} MB) "
          f"— exporter == bullet to_quantised_buffer")
    return 0


if __name__ == "__main__":
    sys.exit(main(sys.argv[1], sys.argv[2]))
