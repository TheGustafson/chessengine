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


def load_warm(path: str | Path, arch) -> dict[str, np.ndarray]:
    """Warm-start load: raw checkpoint -> params for a possibly EXTENDED arch.

    New additive tensors the checkpoint predates (l0_factw, l0_psqt_*) are
    zero-filled — zero-init is an exact no-op for all three T0/T6c/sparsity
    features, so warm-starting an extended arch from a bullet-era raw is
    function-preserving at step 0. Tensors in the file but not the arch are
    an error (that's a wrong-arch load, not a warm start)."""
    shapes = arch.param_shapes()
    flat = open_raw(path, shapes=None, mmap=False)
    extra = set(flat) - set(shapes)
    # Dropping the FullThreats channel is a deliberate arch change, not a
    # wrong-arch load: a halfka_only run warm-starting from a net that HAS
    # threats is exactly the no-threats experiment, and the trained HalfKA
    # rows are worth keeping. Only those two tensors may be discarded, and
    # only when the arch has actually dropped the channel.
    if extra and getattr(arch, "halfka_only", False):
        droppable = {"l0_threatw", "l0_psqt_thw"}
        if extra <= droppable:
            print(f"warm-start: halfka_only — dropping {sorted(extra)}")
            extra = set()
    if extra & set(DIS_TENSORS) and not getattr(arch, "dis_proj", 0):
        # Same argument as the fresh-init below, in the other direction: the
        # dis head cannot influence the value function, so a checkpoint that
        # carries one and an arch that does not want one is a probe being
        # switched OFF, not a wrong-arch load.
        print(f"warm-start: dis_proj=0 — dropping {sorted(extra & set(DIS_TENSORS))}")
        extra = extra - set(DIS_TENSORS)
    if extra:
        raise ValueError(f"checkpoint has tensors the arch lacks: {extra}")
    # ---- Is the checkpoint's dis head THE SAME HEAD this arch wants? --------
    # Decided ONCE, from `dis_projw`, before any tensor is read — because
    # element COUNTS collide and a per-tensor size check silently accepts a
    # different head. Caught on the real canonical checkpoint, 2026-08-03:
    # it carries the pre-d7582485 head at `dis_proj = 64`, so its `dis_outw` is
    # 64 values (one lane per hidden unit); an arch asking for `dis_proj = 32`
    # wants `(32, 2)` — ALSO 64 values. `prod(want) == arr.size` passed, and the
    # load reshaped one net's 64-unit magnitude lane into another net's 32x2
    # output matrix without a word. `dis_projw` cannot collide the same way:
    # its count is `hl · proj` and `hl` is fixed by the arch, so it pins `proj`.
    dis_proj = int(getattr(arch, "dis_proj", 0) or 0)
    dis_reuse = False
    if dis_proj:
        pw, ow = flat.get("dis_projw"), flat.get("dis_outw")
        dis_reuse = (pw is not None and pw.size == arch.hl * dis_proj
                     and ow is not None
                     and ow.size in (dis_proj, dis_proj * DIS_OUTS))
        if pw is not None and not dis_reuse:
            print(f"warm-start: checkpoint dis head is a DIFFERENT head "
                  f"(dis_projw {pw.size} values vs {arch.hl}x{dis_proj}) — "
                  f"re-initialising all of {sorted(DIS_TENSORS)}")

    out = {}
    dis_rng = np.random.default_rng(0xD15)
    for name, want in shapes.items():
        if name in DIS_TENSORS and not dis_reuse:
            # THE DISAGREEMENT HEAD IS FRESH-INIT, NEVER ZERO-INIT, and the
            # distinction is functional. `dis_h = crelu(hl1 @ W + b)`: with W
            # and b both zero every unit sits exactly on the clamp's lower
            # kink, so the head emits a constant and its gradient is at best
            # ill-defined there. Zero-init is the right answer for l0_factw /
            # the PSQT columns because zero makes them an exact NO-OP on a
            # function that must not move; the dis head has no such duty — it
            # is stop_gradient'ed away from the trunk on both sides
            # (model.py h_in, losses.py resid), so re-initialising it CANNOT
            # change the value function, which is precisely what makes it
            # warm-start safe at any shape.
            print(f"warm-start: fresh-init dis tensor {name} {want}")
            out[name] = _init_dis(name, want, dis_rng)
            continue
        if name in flat:
            arr = flat[name]
            if int(np.prod(want)) == arr.size:
                out[name] = arr.reshape(want)
                continue
            # A dis tensor whose SHAPE moved is not a wrong-arch load — see the
            # fresh-init note above for why the head is re-initialisable at
            # will. Every other tensor keeps the hard error.
            if name not in DIS_TENSORS:
                raise ValueError(f"{name}: file count {arr.size} != shape {want}")
            if (name == "dis_outw" and arr.size == want[0]
                    and want[1] == DIS_OUTS):
                # The 1-column -> 2-column widening (d7582485). The old lane
                # predicted |residual|, which is TODAY'S COLUMN 1, so it is
                # carried there and column 0 (signed) starts fresh. No net in
                # the tree has a trained one — every run so far set
                # dis_lambda = 0 — but the migration is six lines and putting
                # a magnitude lane in the signed column would be silent.
                col0 = np.sqrt(2.0 / max(want[0], 1)) * dis_rng.standard_normal(want[0])
                mig = np.stack([col0, arr.reshape(want[0])], axis=1)
                print(f"warm-start: dis_outw widened 1 -> {DIS_OUTS} columns "
                      f"(old lane -> col 1 = magnitude, col 0 re-init)")
                out[name] = mig.astype(np.float32)
                continue
            if name == "dis_outb" and arr.size == 1 and want == (DIS_OUTS,):
                out[name] = np.array([0.0, float(arr.reshape(1)[0])], np.float32)
                continue
            print(f"warm-start: dis tensor {name} re-init "
                  f"(checkpoint {arr.size} values, arch wants {want})")
            out[name] = _init_dis(name, want, dis_rng)
        elif name in DIS_TENSORS:
            # Reuse was on (`dis_projw` matched) but this tensor is absent —
            # a half-written head. Fresh, same rationale as above.
            print(f"warm-start: fresh-init dis tensor {name} {want}")
            out[name] = _init_dis(name, want, dis_rng)
        else:
            if name not in ("l0_factw", "l0_psqt_hkw", "l0_psqt_thw"):
                raise ValueError(f"checkpoint missing non-additive tensor {name}")
            print(f"warm-start: zero-init new tensor {name} {want}")
            out[name] = np.zeros(want, dtype=np.float32)
    return out


def _init_dis(name: str, shape: tuple[int, ...], rng) -> np.ndarray:
    """model.init_params' distribution for one dis tensor, on numpy.

    Weights ~ Normal(0, sqrt(2/fan_in)) with fan_in = shape[0]; biases zero.
    Kept here rather than calling model.init_params because load_warm is the
    one checkpoint path that must not import jax (tools read raw checkpoints
    on hosts with no accelerator, and jax's import is the expensive part)."""
    if name.endswith("b"):
        return np.zeros(shape, dtype=np.float32)
    return (np.sqrt(2.0 / max(shape[0], 1))
            * rng.standard_normal(shape)).astype(np.float32)


def fold_for_export(params: dict[str, np.ndarray], arch) -> dict[str, np.ndarray]:
    """Collapse training-only parameterization for the deploy stream:
    merge the T0 factoriser into l0_pstw (merged[f] = real[f] + fact[f % PS_NB],
    d40 merge_factoriser) so the bullet-format tensors are shape-identical to
    an unfactorized net. PSQT tensors pass through untouched (they live in
    their own PSQ1 section, not the bullet stream)."""
    if not getattr(arch, "factorize", False) or "l0_factw" not in params:
        return params
    from .spec import HALFKA_DIMS, PS_NB
    fact = np.asarray(params["l0_factw"], dtype=np.float32)
    merged = np.asarray(params["l0_pstw"], dtype=np.float32) + np.tile(
        fact, (HALFKA_DIMS // PS_NB, 1))
    out = {k: v for k, v in params.items() if k != "l0_factw"}
    out["l0_pstw"] = merged
    return out


# PSQ1 section (T6c deploy format, appended AFTER the padded bullet stream):
#   b"PSQ1" | u32 num_buckets | u32 rows (HALFKA_DIMS + THREAT_DIMS; HalfKA
#   rows first) | i32[rows * num_buckets] row-major.
# Stored value = round_half_away(w * 0.5 * SCALE * 256) — the 0.5
# stm-canonical factor and the cp scale (400) are folded in, 8 fractional
# bits. Engine: value_cp += (psqt_stm[b] - psqt_ntm[b]) >> 8 (arithmetic).
PSQT_MAGIC = b"PSQ1"
PSQT_FIXED_MULT = 0.5 * 400.0 * 256.0

# SKP1 section (D47U output-skip deploy format, appended after PSQ1 / the
# padded bullet stream): b"SKP1" | u32 num_buckets | u32 hl |
# f32[hl * num_buckets] row-major — the skipw lane in raw f32 (the engine
# tail is f32; no quant gymnastics, mirrors SF's fwdOut semantics).
SKIP_MAGIC = b"SKP1"

# DIS1 section (the disagreement head, 2026-08-03), appended after SKP1:
#
#   b"DIS1" | u32 hl | u32 dis_proj | u32 outs
#           | f32[hl · dis_proj]      dis_projw, row-major (hl, dis_proj)
#           | f32[dis_proj]           dis_projb
#           | f32[dis_proj · outs]    dis_outw,  row-major (dis_proj, outs)
#           | f32[outs]               dis_outb
#
# The head reads `hl1` — the concatenated pairwise activation, stm half first
# — so the engine's `[u8; L1]` FT vector IS its input, in the reduced frame
# `FT_QUANT² / 2^FT_SHIFT`. Column 0 is the SIGNED residual, column 1 the
# MAGNITUDE (losses.py).
#
# **f32, NOT quantised, and that is a decision with a reason.** Everything in
# the CENN that is integer is integer because it is on the per-node eval path;
# this head is polled SELECTIVELY and by design never runs there. Three
# consequences: the section is 98.7 kB at hl=768/proj=32 against an 84 MB net
# (0.12 %), one evaluation is ~25 k MACs against the ~640 k the value path
# already pays, and — the reason that actually decides it — the engine then
# computes the SAME function the trainer fitted, so a JAX↔Rust parity check
# measures the deployment plumbing instead of a quantisation frame nobody has
# calibrated. Identical reasoning and identical encoding to the SKP1 lane.
DIS_MAGIC = b"DIS1"
DIS_HEADER_LEN = 16
# Tensor names, in section order. `dis_outw` is (dis_proj, DIS_OUTS).
DIS_TENSORS = ("dis_projw", "dis_projb", "dis_outw", "dis_outb")
DIS_OUTS = 2


def dis_section_bytes(params: dict[str, np.ndarray], arch) -> bytes:
    import struct as _s
    proj = int(arch.dis_proj)
    pw = np.asarray(params["dis_projw"], dtype=np.float32).reshape(arch.hl, proj)
    pb = np.asarray(params["dis_projb"], dtype=np.float32).reshape(proj)
    ow = np.asarray(params["dis_outw"], dtype=np.float32).reshape(proj, DIS_OUTS)
    ob = np.asarray(params["dis_outb"], dtype=np.float32).reshape(DIS_OUTS)
    return (DIS_MAGIC + _s.pack("<III", arch.hl, proj, DIS_OUTS)
            + pw.astype("<f4").tobytes() + pb.astype("<f4").tobytes()
            + ow.astype("<f4").tobytes() + ob.astype("<f4").tobytes())


def parse_dis_section(sec: bytes) -> dict[str, np.ndarray]:
    """Inverse of :func:`dis_section_bytes`, starting at the DIS1 magic.

    Tolerant of trailing bytes (future sections); raises on anything it cannot
    read. Returns the four tensors in OUR storage convention (in, out)."""
    if len(sec) < DIS_HEADER_LEN or sec[:4] != DIS_MAGIC:
        raise ValueError("not a DIS1 section")
    hl = int.from_bytes(sec[4:8], "little")
    proj = int.from_bytes(sec[8:12], "little")
    outs = int.from_bytes(sec[12:16], "little")
    counts = [hl * proj, proj, proj * outs, outs]
    need = DIS_HEADER_LEN + 4 * sum(counts)
    if len(sec) < need:
        raise ValueError(f"DIS1 section truncated: need {need}, got {len(sec)}")
    flat = np.frombuffer(sec, dtype="<f4", count=sum(counts), offset=DIS_HEADER_LEN)
    off, out = 0, {}
    for name, n, shape in zip(DIS_TENSORS, counts,
                              [(hl, proj), (proj,), (proj, outs), (outs,)]):
        out[name] = flat[off:off + n].reshape(shape).astype(np.float32)
        off += n
    return out


def dis_section_len(sec: bytes) -> int:
    """Byte length of the DIS1 section at the head of ``sec`` (0 if absent)."""
    if len(sec) < DIS_HEADER_LEN or sec[:4] != DIS_MAGIC:
        return 0
    hl = int.from_bytes(sec[4:8], "little")
    proj = int.from_bytes(sec[8:12], "little")
    outs = int.from_bytes(sec[12:16], "little")
    return DIS_HEADER_LEN + 4 * (hl * proj + proj + proj * outs + outs)


def skip_section_bytes(params: dict[str, np.ndarray], arch) -> bytes:
    import struct as _s
    w = np.asarray(params["skipw"], dtype=np.float32)
    return (SKIP_MAGIC + _s.pack("<II", arch.num_buckets, arch.hl)
            + w.astype("<f4").tobytes())


def psqt_section_bytes(params: dict[str, np.ndarray], arch) -> bytes:
    from .spec import HALFKA_DIMS, THREAT_DIMS
    import struct as _s
    hk = np.asarray(params["l0_psqt_hkw"], dtype=np.float64)
    rows = [hk]
    total_rows = HALFKA_DIMS
    if not arch.halfka_only:
        rows.append(np.asarray(params["l0_psqt_thw"], dtype=np.float64))
        total_rows += THREAT_DIMS
    table = np.concatenate(rows, axis=0) * PSQT_FIXED_MULT
    q = _round_half_away(table)
    if np.abs(q).max() >= 2 ** 31:
        raise OverflowError("psqt quantisation overflow")
    return (PSQT_MAGIC + _s.pack("<II", arch.num_buckets, total_rows)
            + q.astype("<i4").tobytes())


def quant_bytes_iter(params: dict[str, np.ndarray], arch, chunk_rows: int = 4096):
    """Yield the quant.bin byte stream for ``params`` under ``arch``
    (spec.ArchSpec), tensor by tensor, chunked so the 137 MB D47c export never
    materializes more than ~chunk_rows*HL floats at once (params may be
    memmaps). Layout = arch.quant_layout() == the d40 fmt vec (lines 1295-1314),
    padded to a 64-byte multiple with cyclic b"bullet" (weights.rs:165-174).
    T0 factoriser tensors are folded into l0_pstw first (fold_for_export), so
    the emitted stream is always shape-identical to an unfactorized net."""
    params = fold_for_export(params, arch)
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
    """Write a byte-exact bullet quant.bin; returns total size. When the arch
    carries the T6c PSQT skip, the versioned PSQ1 section follows the padded
    bullet stream (absent otherwise — legacy byte parity is untouched)."""
    total = 0
    with open(path, "wb") as f:
        for buf in quant_bytes_iter(params, arch):
            f.write(buf)
            total += len(buf)
        if getattr(arch, "psqt", False):
            sec = psqt_section_bytes(params, arch)
            f.write(sec)
            total += len(sec)
        if getattr(arch, "out_skip", False):
            sec = skip_section_bytes(params, arch)
            f.write(sec)
            total += len(sec)
        if getattr(arch, "dis_proj", 0):
            sec = dis_section_bytes(params, arch)
            f.write(sec)
            total += len(sec)
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
    # Pad to the next 64-byte boundary, then OPTIONAL trailers (PSQ1 then
    # SKP1) — dequant parses the STREAM tensors only; trailer reconstruction
    # is the caller's job (they carry folded/f32 encodings, not stream quant).
    pad_len = (64 - off % 64) % 64
    padding = raw[off:off + pad_len]
    if bytes(padding) != spec_pad(pad_len):
        raise ValueError(f"invalid quant padding ({len(padding)} bytes)")
    rest = bytes(raw[off + pad_len:])
    if rest and rest[:4] not in (PSQT_MAGIC, SKIP_MAGIC, DIS_MAGIC):
        raise ValueError(f"unrecognized quant trailer ({len(rest)} bytes)")
    return out
