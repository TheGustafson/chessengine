"""bullet raw-checkpoint IO: name-tagged f32 tensor blobs.

Format (weights.rs ``ModelWeights::write_into`` + ``utils::write_to_byte_buffer``,
lines 95-110 / 404-431): for each tensor, in BTreeMap (lexicographic name)
order::

    <ascii name> b"\\n"  u64le(count)  f32le[count]

The values are bullet's INTERNAL layout: dense matrices are column-major over
(shape.rows = fan_out, shape.cols = fan_in), i.e. the flat array is
``w[fan_in_index * fan_out + fan_out_index]`` -- which is exactly a row-major
(fan_in, fan_out) numpy array. So our storage convention (spec.ArchSpec
docstring) round-trips with plain reshape/ravel and no transposes.
"""

from __future__ import annotations

import struct
from pathlib import Path

import numpy as np


def read_raw_index(path: str | Path) -> dict[str, tuple[int, int]]:
    """Scan a raw checkpoint, returning {name: (payload_offset_bytes, count)}.

    Only reads the tiny headers; pair with ``np.memmap`` for zero-copy access
    to the big tensors (the 432 MB D47c raw never needs to be resident).
    """
    out: dict[str, tuple[int, int]] = {}
    size = Path(path).stat().st_size
    with open(path, "rb") as f:
        while f.tell() < size:
            name = bytearray()
            while True:
                ch = f.read(1)
                if not ch:
                    raise EOFError("truncated raw checkpoint (name)")
                if ch == b"\n":
                    break
                name += ch
            (count,) = struct.unpack("<Q", f.read(8))
            out[bytes(name).decode("ascii")] = (f.tell(), count)
            f.seek(count * 4, 1)
    return out


def open_raw(path: str | Path, shapes: dict[str, tuple[int, ...]] | None = None,
             mmap: bool = True) -> dict[str, np.ndarray]:
    """Load (or memory-map) every tensor of a raw checkpoint.

    If ``shapes`` is given, tensors are reshaped to it (and the set of names
    must match exactly); otherwise they stay flat.
    """
    index = read_raw_index(path)
    if shapes is not None:
        missing = set(shapes) - set(index)
        extra = set(index) - set(shapes)
        if missing or extra:
            raise ValueError(f"tensor set mismatch: missing={missing} extra={extra}")
    out = {}
    for name, (offset, count) in index.items():
        if mmap:
            arr = np.memmap(path, dtype="<f4", mode="r", offset=offset, shape=(count,))
        else:
            with open(path, "rb") as f:
                f.seek(offset)
                arr = np.fromfile(f, dtype="<f4", count=count)
        if shapes is not None:
            want = shapes[name]
            if int(np.prod(want)) != count:
                raise ValueError(f"{name}: file count {count} != shape {want}")
            arr = arr.reshape(want)
        out[name] = arr
    return out


def write_raw(path: str | Path, params: dict[str, np.ndarray]) -> None:
    """Write a bullet-compatible raw checkpoint (BTreeMap = sorted name order)."""
    with open(path, "wb") as f:
        for name in sorted(params):
            arr = np.ascontiguousarray(params[name], dtype="<f4").ravel()
            f.write(name.encode("ascii") + b"\n")
            f.write(struct.pack("<Q", arr.size))
            f.write(arr.tobytes())


# ---------------------------------------------------------------------------
# Quantisation (weights.rs QuantTarget::quantise, lines 292-334)
# ---------------------------------------------------------------------------

def _round_half_away(x: np.ndarray) -> np.ndarray:
    """Rust f64::round: round half AWAY from zero (numpy rounds half to even)."""
    return np.sign(x) * np.floor(np.abs(x) + 0.5)


def quantise(values: np.ndarray, multiplier: int, dtype: str,
             round_: bool = True) -> np.ndarray:
    """f32 -> i8/i16 exactly like bullet: qf = round|trunc(q * f64(w)); error
    out (like bullet's io::Error) if any value doesn't fit the target type."""
    qf = np.float64(multiplier) * values.astype(np.float64)
    qf = _round_half_away(qf) if round_ else np.trunc(qf)
    info = np.iinfo(dtype)
    if qf.min() < info.min or qf.max() > info.max:
        bad = int(np.argmax(np.abs(qf)))
        raise OverflowError(
            f"quantisation overflow: value {values.ravel()[bad]} * {multiplier} "
            f"= {qf.ravel()[bad]} outside {dtype}")
    return qf.astype(dtype)


def quant_bytes_iter(params: dict[str, np.ndarray], arch, chunk_rows: int = 4096):
    """Yield the quant.bin byte stream for ``params`` under ``arch``
    (spec.ArchSpec), tensor by tensor, chunked so the 137 MB D47c export never
    materializes more than ~chunk_rows*HL floats at once (params may be
    memmaps). Layout = arch.quant_layout() == the d40 fmt vec (lines 1295-1314),
    padded to a 64-byte multiple with cyclic b"bullet" (weights.rs:165-174)."""
    total = 0
    for name, kind, mult, transpose in arch.quant_layout():
        arr = params[name]
        if transpose:
            # SavedFormat::transpose (weights.rs:260-274): internal col-major
            # (out, in) -> row-major (out, in); with our (in, out) storage
            # that is arr.T.
            arr = arr.T
        arr = np.atleast_2d(arr)
        for i in range(0, arr.shape[0], chunk_rows):
            chunk = np.ascontiguousarray(arr[i:i + chunk_rows])
            if kind == "f32":
                buf = chunk.astype("<f4").tobytes()
            else:
                buf = quantise(chunk, mult, {"i8": "<i1", "i16": "<i2"}[kind]).tobytes()
            total += len(buf)
            yield buf
    if total % 64:
        yield bytes(spec_pad(64 - total % 64))


def spec_pad(n: int) -> bytes:
    pad = b"bullet"
    return bytes(pad[i % 6] for i in range(n))


def write_quant(path: str | Path, params: dict[str, np.ndarray], arch) -> int:
    """Write a byte-exact bullet quant.bin; returns total size."""
    total = 0
    with open(path, "wb") as f:
        for buf in quant_bytes_iter(params, arch):
            f.write(buf)
            total += len(buf)
    return total


def dequantise_quant(path: str | Path, arch) -> dict[str, np.ndarray]:
    """Parse a quant.bin back into float params (the load_quantised_weights
    path, d40:598-653): i16/255, i8/64, f32 passthrough; transposed tensors
    are un-transposed back into our (in, out) storage."""
    raw = np.fromfile(path, dtype=np.uint8)
    shapes = arch.param_shapes()
    out = {}
    off = 0
    for name, kind, mult, transpose in arch.quant_layout():
        shape = shapes[name]
        n = int(np.prod(shape))
        if kind == "i16":
            vals = raw[off:off + 2 * n].view("<i2").astype(np.float32) / mult
            off += 2 * n
        elif kind == "i8":
            vals = raw[off:off + n].view("<i1").astype(np.float32) / mult
            off += n
        else:
            vals = raw[off:off + 4 * n].view("<f4").astype(np.float32)
            off += 4 * n
        if transpose:
            fan_in, fan_out = shape
            vals = vals.reshape(fan_out, fan_in).T.copy()
        else:
            vals = vals.reshape(shape)
        out[name] = vals
    padding = raw[off:]
    if len(padding) > 63 or bytes(padding) != spec_pad(len(padding)):
        raise ValueError(f"invalid quant padding ({len(padding)} bytes)")
    return out
