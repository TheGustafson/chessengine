#!/usr/bin/env python3
"""Pack a D40 or frozen-classic policy checkpoint into CPOL v2."""

import argparse
import struct
from pathlib import Path

HALFKA, THREATS, TRUNK, L2, L3, BUCKETS = 22528, 60720, 768, 16, 32, 8
DEFAULT_RANK = 8
HEADER = 32
BASE = ((HALFKA * TRUNK + TRUNK) * 2 + THREATS * TRUNK
        + BUCKETS * L2 * TRUNK
        + (BUCKETS * L2 + BUCKETS * L3 * L2 + BUCKETS * L3 + BUCKETS * L3 + BUCKETS) * 4)
def fnv1a64(data: bytes) -> int:
    value = 0xCBF29CE484222325
    for byte in data:
        value = ((value ^ byte) * 0x100000001B3) & 0xFFFFFFFFFFFFFFFF
    return value


def die(message: str) -> None:
    raise SystemExit(f"policy_pack: {message}")


def pack(source: Path, output: Path, kind: str, rank: int = DEFAULT_RANK,
         base_override: int | None = None) -> None:
    if kind not in ("bilinear", "full", "classic", "classic-factorized"):
        die("head must be bilinear, full, classic, or classic-factorized")
    if not 1 <= rank <= 64:
        die("bilinear rank must be in 1..=64")
    raw = source.read_bytes()
    classic_trunk = 512
    if kind == "classic":
        base = 0
        size = 2 * (64 * rank * classic_trunk + 64 * rank) * 4
    elif kind == "classic-factorized":
        base = 0
        size = (rank * classic_trunk + rank + 4096 * rank + 4096) * 4
    else:
        base = BASE if base_override is None else base_override
        size = (2 * (64 * rank * TRUNK + 64 * rank) * 4
                if kind == "bilinear" else (4096 * TRUNK + 4096) * 4)
    padding = raw[base + size:]
    expected = bytes(b"bullet"[i % 6] for i in range(len(padding)))
    if len(raw) < base + size or len(padding) > 63 or padding != expected:
        die(f"checkpoint size {len(raw):,} does not match {kind} layout")
    payload = raw[base:base + size]
    kind_id = {"bilinear": 1, "full": 2, "classic": 3,
               "classic-factorized": 4}[kind]
    # 0 = generation-0 bilinear target/matmul transpose (legacy recovery in
    # runtime), 1 = canonical from-to layout produced by the corrected trainer.
    # Full heads are already canonical and retain reserved byte 0.
    layout = 1 if kind in ("bilinear", "classic") else 0
    rank = rank if kind in ("bilinear", "classic", "classic-factorized") else 0
    input_size = classic_trunk if kind.startswith("classic") else TRUNK
    header = (b"CPOL" + struct.pack("<HBBIIQQ", 2, kind_id, layout, input_size, rank,
                                     len(payload), fnv1a64(payload)))
    assert len(header) == HEADER
    output.parent.mkdir(parents=True, exist_ok=True)
    output.write_bytes(header + payload)
    print(f"policy_pack: wrote {output} ({len(header)+len(payload):,} bytes, {kind})")


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("source", type=Path)
    parser.add_argument("output", type=Path)
    parser.add_argument("head", choices=("bilinear", "full", "classic", "classic-factorized"))
    parser.add_argument("--rank", type=int, default=DEFAULT_RANK)
    parser.add_argument("--base", type=int, default=None, help=(
        "override the head-tensor byte offset (checkpoints whose layout "
        "omits the value head, e.g. d41_alloc: trunk-only base 81237504)"))
    args = parser.parse_args()
    pack(args.source, args.output, args.head, args.rank, args.base)
