#!/usr/bin/env python3
"""CENN-v2 reader/writer for perf study 32's offline i8 fidelity work.

Why this exists: the study needs to answer "what does an i8 HalfKA feature
transformer do to the EVAL" before any engine work.  The trick that makes that
answerable with zero Rust is that quantising a weight to i8 with a scale `s`
and multiplying by `s` on the way into an i16 accumulator is *numerically
identical* to storing `s * round(w / s)` in the existing i16 table.  So we can
simulate any i8 scheme exactly by writing a normal CENN-v2 net whose HalfKA
weights happen to lie on the i8 grid, and score it with the shipped engine.

The only fiddly part is the payload checksum (FNV-1a-64 over 135 MB), which a
Python loop cannot do in reasonable time -- so it is compiled to a tiny C
shared object on first use and called through `ctypes`.

Header layout: src/eval/nnue/sota_loader.rs (module docs).
"""

from __future__ import annotations

import ctypes
import hashlib
import struct
import subprocess
import tempfile
from dataclasses import dataclass
from pathlib import Path

import numpy as np

HEADER_LEN = 64
MAGIC_V2 = b"CEN2"
PSQT_MAGIC = b"PSQ1"
SKIP_MAGIC = b"SKP1"

_FNV_SRC = r"""
#include <stdint.h>
#include <stddef.h>
uint64_t fnv1a64(const unsigned char *p, size_t n) {
    uint64_t h = 0xCBF29CE484222325ULL;
    for (size_t i = 0; i < n; i++) { h ^= p[i]; h *= 0x100000001B3ULL; }
    return h;
}
"""

_fnv_lib = None


def _fnv_fn():
    """Compile (once) and return the C FNV-1a-64 entry point."""
    global _fnv_lib
    if _fnv_lib is None:
        so = Path(tempfile.gettempdir()) / "cennq_fnv.so"
        if not so.exists():
            with tempfile.NamedTemporaryFile("w", suffix=".c", delete=False) as f:
                f.write(_FNV_SRC)
                csrc = f.name
            subprocess.run(
                ["cc", "-O2", "-shared", "-fPIC", "-o", str(so), csrc], check=True
            )
        _fnv_lib = ctypes.CDLL(str(so))
        _fnv_lib.fnv1a64.restype = ctypes.c_uint64
        _fnv_lib.fnv1a64.argtypes = [ctypes.c_char_p, ctypes.c_size_t]
    return _fnv_lib.fnv1a64


def fnv1a64(buf) -> int:
    b = bytes(buf) if not isinstance(buf, bytes) else buf
    return int(_fnv_fn()(b, len(b)))


@dataclass
class Cenn:
    raw: bytearray
    flags: int
    halfka_dims: int
    threat_dims: int
    acc: int
    l2: int
    l3: int
    buckets: int
    ft_quant: int
    l1_quant: int
    ft_shift: int
    network_scale: int
    payload_len: int
    checksum: int
    # payload section byte offsets (absolute, into `raw`)
    off_halfka: int
    off_ft_bias: int
    off_threat: int
    off_rest: int

    @property
    def n_halfka(self) -> int:
        return self.halfka_dims * self.acc

    def halfka(self) -> np.ndarray:
        """HalfKA FT weights as an (halfka_dims, acc) int16 view into `raw`."""
        a = np.frombuffer(
            self.raw, dtype="<i2", count=self.n_halfka, offset=self.off_halfka
        )
        return a.reshape(self.halfka_dims, self.acc)

    def ft_bias(self) -> np.ndarray:
        return np.frombuffer(self.raw, dtype="<i2", count=self.acc, offset=self.off_ft_bias)

    def threats(self) -> np.ndarray:
        n = self.threat_dims * self.acc
        if n == 0:
            return np.zeros((0, self.acc), dtype=np.int8)
        a = np.frombuffer(self.raw, dtype="<i1", count=n, offset=self.off_threat)
        return a.reshape(self.threat_dims, self.acc)


def load(path) -> Cenn:
    raw = bytearray(Path(path).read_bytes())
    if raw[:4] != MAGIC_V2:
        raise ValueError(f"not a CEN2 file: {bytes(raw[:4])!r}")
    (version, flags) = struct.unpack_from("<HH", raw, 4)
    if version != 2:
        raise ValueError(f"unsupported CENN-v2 version {version}")
    (halfka_dims, threat_dims, acc, l2, l3, buckets) = struct.unpack_from("<6I", raw, 8)
    (ft_quant, l1_quant, ft_shift, network_scale) = struct.unpack_from("<4i", raw, 32)
    (payload_len, checksum) = struct.unpack_from("<QQ", raw, 48)

    off = HEADER_LEN
    off_halfka = off
    off += halfka_dims * acc * 2
    off_ft_bias = off
    off += acc * 2
    off_threat = off
    off += threat_dims * acc * 1
    off_rest = off  # l1_w ... l3_b, then optional PSQ1 / SKP1 trailers

    c = Cenn(
        raw=raw, flags=flags, halfka_dims=halfka_dims, threat_dims=threat_dims,
        acc=acc, l2=l2, l3=l3, buckets=buckets, ft_quant=ft_quant,
        l1_quant=l1_quant, ft_shift=ft_shift, network_scale=network_scale,
        payload_len=payload_len, checksum=checksum, off_halfka=off_halfka,
        off_ft_bias=off_ft_bias, off_threat=off_threat, off_rest=off_rest,
    )
    got = fnv1a64(memoryview(raw)[HEADER_LEN:HEADER_LEN + payload_len])
    if got != checksum:
        raise ValueError(f"checksum mismatch: header {checksum:#x}, computed {got:#x}")
    return c


def write_variant(src: Cenn, out_path, halfka_new: np.ndarray, ft_bias_new=None) -> str:
    """Write a CENN-v2 net identical to `src` except for the HalfKA weights
    (and optionally the FT bias).  Recomputes the payload checksum; the PSQ1 /
    SKP1 trailers ride outside it and are copied verbatim."""
    raw = bytearray(src.raw)  # copy
    assert halfka_new.shape == (src.halfka_dims, src.acc), halfka_new.shape
    assert halfka_new.dtype == np.int16
    raw[src.off_halfka:src.off_halfka + src.n_halfka * 2] = (
        halfka_new.astype("<i2", copy=False).tobytes()
    )
    if ft_bias_new is not None:
        assert ft_bias_new.shape == (src.acc,) and ft_bias_new.dtype == np.int16
        raw[src.off_ft_bias:src.off_ft_bias + src.acc * 2] = (
            ft_bias_new.astype("<i2", copy=False).tobytes()
        )
    new_ck = fnv1a64(memoryview(raw)[HEADER_LEN:HEADER_LEN + src.payload_len])
    struct.pack_into("<Q", raw, 56, new_ck)
    Path(out_path).write_bytes(raw)
    return hashlib.sha256(raw).hexdigest()[:16]
