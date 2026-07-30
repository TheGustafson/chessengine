#!/usr/bin/env python3
"""Recode POC net packer (learned-feature-code study §6.1, deploy modes i/ii).

Reads a CEN2/CEN5 SOTA net, builds a threat re-coding `remap: u16[60720] -> R`
plus the projected small tables, and writes arm nets:

  --arm scatter     mode (i)-shaped twin for the speed A/B: FULL-size threat
                    tables with W'[t] = Wsmall[remap[t]] scattered back over
                    60,720 rows, plus an IDENTITY TRM1 so the engine executes
                    the exact same remap-lookup instruction stream as the
                    compressed arm. Eval is bit-identical to `compress`.
  --arm compress    mode (ii): R-row threat table + the real TRM1. This is the
                    arm whose weight table shrinks 77.7 MB -> R*ACC bytes.
  --arm tied        mode (i) proper: full-size tables, NO TRM1 — zero engine
                    deviation from a stock net (for strength A/Bs on the
                    unmodified loader path).

Partitions (--code):
  random     balanced pseudo-random partition: remap[t] = perm[t] % R.
             The content-free null — and sufficient for the phase-1 speed
             question, where only the access-set size matters.
  identity   rows must equal 60720 (instrumentation arm).
  file:PATH  load a remap from a .npy (int array, len 60720, entries < R).

Projection (T0b): Wsmall[r] = round(sum_t f_t W[t] / sum_t f_t) over the
class, f from --census (census_counts.npz, key 'th'), +1-smoothed so empty
census classes fall back to the unweighted mean. PSQ1 threat rows are
projected the same way; SKP1 passes through untouched.

Frame notes: threat weights are i8 in the CENN and the loader consumes them
raw (no requantisation), so scatter/compress arms are eval-BIT-IDENTICAL by
construction; the engine test `recode_pair_is_eval_identical_over_a_game`
pins the same property in-tree.
"""

import argparse
import struct
import sys
from pathlib import Path

import numpy as np

HALFKA_DIMS = 22528
THREAT_DIMS = 60720


def fnv1a64(data: bytes) -> int:
    """FNV-1a 64-bit, chunk-unrolled (matches sota_loader.rs fnv1a64)."""
    h = 0xCBF29CE484222325
    prime = 0x100000001B3
    mask = 0xFFFFFFFFFFFFFFFF
    for byte in data:
        h = ((h ^ byte) * prime) & mask
    return h


def parse_net(path: Path):
    b = path.read_bytes()
    magic = b[:4]
    if magic not in (b"CEN2", b"CEN5"):
        sys.exit(f"bad magic {magic!r}")
    hlen = 64 if magic == b"CEN2" else 72
    (halfka_dims, threat_dims, acc, l2, l3, buckets) = struct.unpack("<6I", b[8:32])
    if halfka_dims != HALFKA_DIMS or threat_dims != THREAT_DIMS:
        sys.exit(f"expected a stock net (halfka {halfka_dims}, threat {threat_dims})")
    plen, _ck = struct.unpack("<QQ", b[48:64])
    payload = b[hlen : hlen + plen]

    off = 0
    n_hk = HALFKA_DIMS * acc * 2
    halfka = payload[off : off + n_hk]
    off += n_hk
    bias = payload[off : off + acc * 2]
    off += acc * 2
    threat = np.frombuffer(payload[off : off + THREAT_DIMS * acc], dtype=np.int8).reshape(
        THREAT_DIMS, acc
    )
    off += THREAT_DIMS * acc
    head = payload[off:]

    rest = b[hlen + plen :]
    psqt_hk = psqt_th = None
    skp1 = b""
    if rest[:4] == b"PSQ1":
        nb, rows = struct.unpack("<II", rest[4:12])
        assert nb == buckets and rows == HALFKA_DIMS + THREAT_DIMS, (nb, rows)
        table = np.frombuffer(rest[12 : 12 + rows * nb * 4], dtype=np.int32).reshape(rows, nb)
        psqt_hk = table[:HALFKA_DIMS]
        psqt_th = table[HALFKA_DIMS:]
        rest = rest[12 + rows * nb * 4 :]
    if rest[:4] == b"SKP1":
        nb, hl = struct.unpack("<II", rest[4:12])
        skp1 = bytes(rest[: 12 + nb * hl * 4])
        rest = rest[12 + nb * hl * 4 :]
    if rest:
        sys.exit(f"unparsed trailer bytes: {len(rest)} starting {rest[:4]!r}")

    hdr = {
        "magic": magic,
        "hlen": hlen,
        "acc": acc,
        "buckets": buckets,
        "raw_header": b[:hlen],
    }
    return hdr, halfka, bias, threat, head, psqt_hk, psqt_th, skp1


def build_remap(code: str, rows: int, seed: int) -> np.ndarray:
    if code == "identity":
        if rows != THREAT_DIMS:
            sys.exit("identity code requires --rows 60720")
        return np.arange(THREAT_DIMS, dtype=np.uint16)
    if code == "random":
        rng = np.random.default_rng(seed)
        perm = rng.permutation(THREAT_DIMS)
        return (perm % rows).astype(np.uint16)
    if code.startswith("file:"):
        remap = np.load(code[5:]).astype(np.int64)
        if remap.shape != (THREAT_DIMS,) or remap.min() < 0 or remap.max() >= rows:
            sys.exit(f"bad remap file: shape {remap.shape} range "
                     f"[{remap.min()},{remap.max()}] vs rows {rows}")
        return remap.astype(np.uint16)
    sys.exit(f"unknown --code {code}")


def project(table: np.ndarray, remap: np.ndarray, rows: int, freq: np.ndarray,
            out_dtype, clip=None) -> np.ndarray:
    """Frequency-weighted per-class mean of `table` rows, rounded half away."""
    w = freq.astype(np.float64) + 1.0  # +1 smoothing: empty classes -> plain mean
    num = np.zeros((rows, table.shape[1]), dtype=np.float64)
    den = np.zeros(rows, dtype=np.float64)
    # Chunked so the float64 temporaries stay ~100 MB (15 GB box, live trainer).
    step = 4096
    for lo in range(0, table.shape[0], step):
        hi = min(lo + step, table.shape[0])
        np.add.at(num, remap[lo:hi], table[lo:hi].astype(np.float64) * w[lo:hi, None])
        np.add.at(den, remap[lo:hi], w[lo:hi])
    den[den == 0.0] = 1.0
    mean = num / den[:, None]
    out = np.sign(mean) * np.floor(np.abs(mean) + 0.5)  # round half away from zero
    if clip is not None:
        out = np.clip(out, -clip, clip)
    return out.astype(out_dtype)


def trm1_bytes(remap: np.ndarray, rows: int) -> bytes:
    return (b"TRM1" + struct.pack("<II", THREAT_DIMS, rows)
            + remap.astype("<u2").tobytes())


def psq1_bytes(hk: np.ndarray, th: np.ndarray, buckets: int) -> bytes:
    table = np.concatenate([hk, th], axis=0)
    return (b"PSQ1" + struct.pack("<II", buckets, table.shape[0])
            + table.astype("<i4").tobytes())


def emit(out_path: Path, hdr, halfka, bias, threat_bytes, head, threat_rows,
         psq1: bytes, skp1: bytes, trm1: bytes):
    payload = halfka + bias + threat_bytes + head
    header = bytearray(hdr["raw_header"])
    header[12:16] = struct.pack("<I", threat_rows)
    header[48:56] = struct.pack("<Q", len(payload))
    header[56:64] = struct.pack("<Q", fnv1a64(payload))
    out_path.write_bytes(bytes(header) + payload + psq1 + skp1 + trm1)
    print(f"wrote {out_path}  ({(len(header)+len(payload)+len(psq1)+len(skp1)+len(trm1))/1e6:.1f} MB, "
          f"threat rows {threat_rows})")


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("net", type=Path)
    ap.add_argument("out", type=Path)
    ap.add_argument("--arm", required=True, choices=["scatter", "compress", "tied"])
    ap.add_argument("--code", default="random")
    ap.add_argument("--rows", type=int, default=8192)
    ap.add_argument("--seed", type=int, default=1)
    ap.add_argument("--census", type=Path, default=None,
                    help="census_counts.npz with key 'th' (frequency weights)")
    args = ap.parse_args()

    hdr, halfka, bias, threat, head, psqt_hk, psqt_th, skp1 = parse_net(args.net)
    acc, buckets = hdr["acc"], hdr["buckets"]
    remap = build_remap(args.code, args.rows, args.seed)

    freq = np.zeros(THREAT_DIMS, dtype=np.int64)
    if args.census is not None:
        freq = np.load(args.census)["th"]
        assert freq.shape == (THREAT_DIMS,)

    w_small = project(threat, remap, args.rows, freq, np.int8, clip=127)
    p_small = (project(psqt_th, remap, args.rows, freq, np.int64).astype(np.int32)
               if psqt_th is not None else None)

    if args.arm == "compress":
        psq1 = psq1_bytes(psqt_hk, p_small, buckets) if p_small is not None else b""
        emit(args.out, hdr, halfka, bias, w_small.tobytes(), head, args.rows,
             psq1, skp1, trm1_bytes(remap, args.rows))
    else:  # scatter / tied: full-size tables reading the same values
        w_full = w_small[remap]  # (60720, acc)
        p_full = p_small[remap] if p_small is not None else None
        psq1 = psq1_bytes(psqt_hk, p_full, buckets) if p_full is not None else b""
        identity = np.arange(THREAT_DIMS, dtype=np.uint16)
        trm1 = trm1_bytes(identity, THREAT_DIMS) if args.arm == "scatter" else b""
        emit(args.out, hdr, halfka, bias, w_full.tobytes(), head, THREAT_DIMS,
             psq1, skp1, trm1)


if __name__ == "__main__":
    main()
