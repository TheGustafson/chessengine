#!/usr/bin/env python3
"""Stockfish `.nnue` -> an ARCHON raw warm checkpoint, so SF's net can be a
LIVE TEACHER inside the JAX trainer (`teacher_warm` in a run spec).

**Why this is a converter and not a relabelling pass.** The two nets share
their entire input space: HalfKAv2_hm (22,528) + FullThreats (60,720) = 83,248,
proven index-for-index identical (`sf_swap.rs` header, 40/40 x 2 perspectives).
Only the layer widths and the tail's routing differ. So SF's weights can be
poured into ARCHON's own parameter dict and evaluated by ARCHON's own
`model.forward`, sharing the student's sparse-index gather batch for batch.

**The topology maps exactly, with no approximation, once three existing arch
flags are set together** (derivation in
`docs/research/sf-teacher-blend-2026-08-01.md` §2):

  hl=1024 l2=32 l3=32   SF's L1 / FC_0_OUTPUTS / FC_1_OUTPUTS
  dual_tail=True        their SqrCReLU+CReLU pair at both layers, and fc_2
                        reading BOTH layers' pairs (the 128-wide concat)
  sf_act=True           the exact float form of that activation
  out_skip=True         carries their `skip_0` term: fc0[30]-fc0[31] is an
                        affine function of the pairwise features, which is
                        precisely what the out_skip lane computes
  psqt=True             their PSQT buckets; psqt_combine's 0.5*(stm-ntm)[b]
                        IS their `(psqt[stm][b]-psqt[ntm][b])/2`
  policy=False          they have no policy head

The only thing that does NOT map is the OUTPUT BUCKET INDEX: SF uses
`(pieces-1)/4`, ARCHON uses `OUTPUT_BUCKETS_LAYOUT[pieces]`, and they are not
the same partition (ARCHON's bucket 0 spans popcounts 0-8, which is SF's
buckets 0, 1 and 2). The teacher is therefore handed its own `t_bucket`
column; see `features.map_batch(sf_bucket=True)`.

Quantisation frames folded into the emitted floats (WSB=6, FT_MAX=255):

    l0_pst*/l0_threatw  / 255          so ARCHON's crelu == their clamp(0,255)
    l0_psqt_*           / 16           so psqt_combine == their psqt_v/16
    l1w  * (255^2/512) / (2^7 * 127)   pairwise frame, then the layer-0 shift
    l2w  * 127 / (2^6 * 127)           the layer-1 shift
    l3w  * 127 * (600/16384)           their OutputScale/HIDDEN_ONE tail
    skipw, l3b                          skip_0, split into its linear and
                                        constant halves

Usage:
    python3 tools/nnue/sf_to_archon_warm.py <net.nnue> <out-raw.bin>
    python3 tools/nnue/sf_to_archon_warm.py --dump-arch

The exact-integer reference `evaluate_int()` is exported for the parity gate
(`jax/tools/sf_teacher_parity.py`) — it reproduces `sf_swap::evaluate_raw`
including every truncation, so a layout error cannot hide behind float noise.
"""

from __future__ import annotations

import argparse
import struct
import sys
from pathlib import Path

import numpy as np

# --- SF `eca43a97` architecture constants (sf_swap.rs:34-45) ---------------
L1 = 1024
L2 = 32
L3 = 32
BUCKETS = 8
PSQ_DIMS = 22528
THR_DIMS = 60720
FT_MAX = 255
WSB = 6
OUT_SCALE = 16
HIDDEN_ONE = 128
VERSION = 0x6A44_8AFA

# --- the float frames the conversion lands in ------------------------------
FTG = FT_MAX * FT_MAX / 512.0          # pairwise: (lo*hi)/512 from crelu^2
DEN0 = float(2 ** (WSB + 1) * 127)     # layer-0 activation normaliser (16256)
DEN1 = float(2 ** WSB * 127)           # layer-1 activation normaliser (8128)
OUTQ = 600.0 / 16384.0                 # fwd -> their internal output units
SKIPQ = OUTQ * DEN0                    # skip_0 rides unnormalised (595.3125)

ARCH_KWARGS = dict(
    hl=L1, l2=L2, l3=L3, num_buckets=BUCKETS,
    dual_tail=True, sf_act=True, out_skip=True, psqt=True,
    policy=False, factorize=False, halfka_only=False,
)


# ---------------------------------------------------------------------------
# SF's LEB128 container (nnue_common.h:226-238), decoded with numpy
# ---------------------------------------------------------------------------

def _leb128_chunk(b: np.ndarray) -> np.ndarray:
    """Decode one self-contained run of LEB128 bytes (ends on a terminator)."""
    term = (b & 0x80) == 0
    # value index of every byte: bytes before the k-th terminator belong to k
    grp = np.empty(b.size, np.int64)
    np.cumsum(term, out=grp)
    grp -= term                                   # terminator joins its own group
    n = int(grp[-1]) + 1
    # position of each byte within its value
    starts = np.searchsorted(grp, np.arange(n))
    k = np.arange(b.size, dtype=np.int64) - starts[grp]
    kmax = int(k.max())
    if kmax > 4:
        raise ValueError(f"leb128 value spans {kmax + 1} bytes (i32 takes <= 5)")
    out = np.zeros(n, np.int64)
    payload = (b & 0x7F).astype(np.int64)
    for i in range(kmax + 1):
        sel = k == i
        # SF shifts by `shift % 32`; with <= 5 bytes the shift never reaches 32
        out[grp[sel]] |= payload[sel] << (7 * i)
    # sign extension: only when the terminator has 0x40 set AND shift < 32
    tail = np.flatnonzero(term)
    bits = 7 * (k[tail] + 1)
    neg = (bits < 32) & ((b[tail] & 0x40) != 0)
    idx = grp[tail][neg]
    out[idx] |= ~((np.int64(1) << bits[neg]) - 1)
    return out


class Reader:
    """Byte cursor over the .nnue file, mirroring sf_swap.rs's `Reader`."""

    MAGIC = b"COMPRESSED_LEB128"

    def __init__(self, buf: memoryview):
        self.buf = buf
        self.pos = 0

    def u32(self) -> int:
        v = struct.unpack_from("<I", self.buf, self.pos)[0]
        self.pos += 4
        return v

    def raw_i8(self, n: int) -> np.ndarray:
        a = np.frombuffer(self.buf, dtype=np.int8, count=n, offset=self.pos)
        self.pos += n
        return a

    def raw_i32(self, n: int) -> np.ndarray:
        a = np.frombuffer(self.buf, dtype="<i4", count=n, offset=self.pos)
        self.pos += 4 * n
        return a

    def leb128(self, n: int, chunk: int = 1 << 23) -> np.ndarray:
        if bytes(self.buf[self.pos:self.pos + len(self.MAGIC)]) != self.MAGIC:
            raise ValueError(f"leb128 magic missing at {self.pos}")
        self.pos += len(self.MAGIC)
        nbytes = self.u32()
        raw = np.frombuffer(self.buf, dtype=np.uint8, count=nbytes,
                            offset=self.pos)
        self.pos += nbytes
        # Decode in chunks that END on a terminator, so peak temporaries stay
        # ~chunk-sized rather than ~90 MB x 8 (this box has 15 GB, shared).
        parts, start, got = [], 0, 0
        while start < nbytes and got < n:
            stop = min(start + chunk, nbytes)
            if stop < nbytes:
                seg = raw[start:stop]
                last = np.flatnonzero((seg & 0x80) == 0)
                if last.size == 0:
                    raise ValueError("leb128 chunk holds no terminator")
                stop = start + int(last[-1]) + 1
            vals = _leb128_chunk(raw[start:stop])
            parts.append(vals)
            got += vals.size
            start = stop
        out = np.concatenate(parts) if len(parts) > 1 else parts[0]
        if out.size < n:
            raise ValueError(f"leb128 produced {out.size} of {n} values")
        return out[:n]


class SFNet:
    """The raw integer tensors of a Stockfish net, in their own layout."""

    def __init__(self, path: str | Path):
        raw = memoryview(Path(path).read_bytes())
        r = Reader(raw)
        version = r.u32()
        if version != VERSION:
            raise ValueError(f"version {version:#x} != {VERSION:#x}")
        r.u32()                       # net hash
        desc = r.u32()                # description length -- NOT `r.pos +=
        r.pos += desc                 # r.u32()`: augmented assignment loads
        r.pos += 4                    # r.pos BEFORE the call advances it

        self.bias = r.leb128(L1).astype(np.int32)
        # The one raw little-endian array, and 69% of the file.
        self.thr_w = r.raw_i8(THR_DIMS * L1).reshape(THR_DIMS, L1)
        self.thr_psqt = r.leb128(THR_DIMS * BUCKETS).astype(np.int32) \
            .reshape(THR_DIMS, BUCKETS)
        self.psq_w = r.leb128(PSQ_DIMS * L1).astype(np.int16) \
            .reshape(PSQ_DIMS, L1)
        self.psq_psqt = r.leb128(PSQ_DIMS * BUCKETS).astype(np.int32) \
            .reshape(PSQ_DIMS, BUCKETS)

        self.fc0_b = np.empty((BUCKETS, L2), np.int32)
        self.fc0_w = np.empty((BUCKETS, L2, L1), np.int8)
        self.fc1_b = np.empty((BUCKETS, L3), np.int32)
        self.fc1_w = np.empty((BUCKETS, L3, 64), np.int8)
        self.fc2_b = np.empty(BUCKETS, np.int32)
        self.fc2_w = np.empty((BUCKETS, 128), np.int8)
        for b in range(BUCKETS):
            r.pos += 4                # layer-stack sub-hash
            self.fc0_b[b] = r.raw_i32(L2)
            self.fc0_w[b] = r.raw_i8(L2 * L1).reshape(L2, L1)
            self.fc1_b[b] = r.raw_i32(L3)
            self.fc1_w[b] = r.raw_i8(L3 * 64).reshape(L3, 64)
            self.fc2_b[b] = r.raw_i32(1)[0]
            self.fc2_w[b] = r.raw_i8(128)
        if r.pos != len(raw):
            raise ValueError(f"consumed {r.pos} of {len(raw)} bytes")


# ---------------------------------------------------------------------------
# The exact integer reference (sf_swap.rs:184-269), vectorised over a batch
# ---------------------------------------------------------------------------

def _trunc_div(a: np.ndarray, b: int) -> np.ndarray:
    """C/Rust integer division: truncate toward zero. numpy's // FLOORS, which
    differs on negatives — and psqt_v, positional and the final sum are all
    signed, so this is not a nicety."""
    q = np.abs(a) // abs(b)
    return np.where((a < 0) != (b < 0), -q, q)


def _bag(table: np.ndarray, idx: np.ndarray, dtype) -> np.ndarray:
    """Sum table[idx] over the -1-padded active-index axis (N, S) -> (N, W).

    Accumulated SLOT BY SLOT rather than by materialising (N, S, W): the
    threat channel is 160 slots x 1024 wide, so the dense form is 1.3 GB at
    N=2000 and this box has 15 GB shared with a live trainer.
    """
    out = np.zeros((idx.shape[0], table.shape[1]), dtype)
    for s in range(idx.shape[1]):
        col = idx[:, s]
        keep = col >= 0
        if keep.any():
            out[keep] += table[col[keep]].astype(dtype)
    return out


def evaluate_int(net: SFNet, hk_stm, hk_ntm, th_stm, th_ntm,
                 bucket: np.ndarray) -> np.ndarray:
    """`sf_swap::evaluate_raw` for a batch, bit-exact including truncations.

    Indices are ARCHON's own (-1-padded) sparse columns — proven identical to
    SF's — and `bucket` is SF's `(pieces-1)/4`.
    """
    accs, psqts = [], []
    for hk, th in ((hk_stm, th_stm), (hk_ntm, th_ntm)):
        acc = (net.bias.astype(np.int32)
               + _bag(net.psq_w, hk, np.int32) + _bag(net.thr_w, th, np.int32))
        accs.append(acc)
        psqts.append(_bag(net.psq_psqt, hk, np.int64)
                     + _bag(net.thr_psqt, th, np.int64))
    n = hk_stm.shape[0]
    rows = np.arange(n)
    psqt_v = _trunc_div(psqts[0][rows, bucket] - psqts[1][rows, bucket], 2)

    ft = np.empty((n, L1), np.int64)
    for p, acc in enumerate(accs):                       # stm first
        lo = np.clip(acc[:, :L1 // 2], 0, FT_MAX).astype(np.int64)
        hi = np.clip(acc[:, L1 // 2:], 0, FT_MAX).astype(np.int64)
        ft[:, p * (L1 // 2):(p + 1) * (L1 // 2)] = (lo * hi) // 512

    fc0 = (net.fc0_b[bucket].astype(np.int64)
           + np.einsum("nj,noj->no", ft, net.fc0_w[bucket].astype(np.int64)))
    concat = np.empty((n, 128), np.int64)
    concat[:, :L2] = np.minimum((fc0 * fc0) >> (2 * (WSB + 1) + 7), 127)
    concat[:, L2:2 * L2] = np.clip(fc0 >> (WSB + 1), 0, 127)
    fc1 = (net.fc1_b[bucket].astype(np.int64)
           + np.einsum("nj,noj->no", concat[:, :64],
                       net.fc1_w[bucket].astype(np.int64)))
    concat[:, 2 * L2:2 * L2 + L3] = np.minimum((fc1 * fc1) >> (2 * WSB + 7), 127)
    concat[:, 2 * L2 + L3:] = np.clip(fc1 >> WSB, 0, 127)

    fwd = (net.fc2_b[bucket].astype(np.int64)
           + np.einsum("nj,nj->n", concat, net.fc2_w[bucket].astype(np.int64))
           + fc0[:, L2 - 2] - fc0[:, L2 - 1])            # skip_0
    positional = _trunc_div(fwd * (600 * OUT_SCALE), HIDDEN_ONE * (1 << WSB) * 2)
    return (_trunc_div(psqt_v, OUT_SCALE)
            + _trunc_div(positional, OUT_SCALE)).astype(np.int64)


# ---------------------------------------------------------------------------
# The conversion
# ---------------------------------------------------------------------------

def to_archon_params(net: SFNet, debias: bool = False) -> dict[str, np.ndarray]:
    """SF's integer tensors -> ARCHON's parameter dict, in ARCHON's frames.

    ``debias`` (OFF, and it is off because it was MEASURED and LOST) folds a
    truncation-bias correction into the biases. Three of SF's steps discard a
    fractional part -- ``ft[j] = (lo*hi)/512`` and the two dual activations,
    all floors -- so the obvious model is "each drops 1/2 on average", which
    is a constant and therefore an exact, free fold:

        l1b -= 0.5 * sum_j fc0_w[o][j]  (/DEN0)   and the l2b / l3b analogues

    **On wl50 that made the float-vs-oracle gap 4.7x WORSE** (median 6.19 ->
    27.28, max 58.7 -> 277.5), which refutes the uniform-fractional-part
    premise rather than the arithmetic. The mechanism is ACTIVATION SPARSITY:
    a CReLU'd accumulator is mostly zeros, `lo*hi` is then exactly 0, and a
    truncation that discards nothing has mean 0, not 1/2. The true mean loss
    is some data-dependent fraction of 1/2, so the correct correction is not a
    constant we can derive from the weights alone.

    Kept, off, and documented because the reasoning is the tempting one and
    the next reader deserves the receipt. The uncorrected residual is
    measured, small and near-unbiased in the regression sense -- see
    docs/research/sf-teacher-blend-2026-08-01.md §3.
    """
    p: dict[str, np.ndarray] = {}
    p["l0_pstb"] = net.bias.astype(np.float32) / FT_MAX
    p["l0_pstw"] = net.psq_w.astype(np.float32) / FT_MAX
    p["l0_threatw"] = net.thr_w.astype(np.float32) / FT_MAX
    p["l0_psqt_hkw"] = net.psq_psqt.astype(np.float32) / OUT_SCALE
    p["l0_psqt_thw"] = net.thr_psqt.astype(np.float32) / OUT_SCALE

    # (fan_in, buckets*fan_out) row-major, buckets-major on the output axis --
    # exactly what _select(x, bucket, num_buckets) slices.
    l1w = net.fc0_w.astype(np.float64) * (FTG / DEN0)      # (B, L2, L1)
    p["l1w"] = np.ascontiguousarray(
        l1w.transpose(2, 0, 1).reshape(L1, BUCKETS * L2)).astype(np.float32)
    l1b = net.fc0_b.astype(np.float64)                     # (B, L2)
    if debias:
        l1b = l1b - 0.5 * net.fc0_w.astype(np.float64).sum(axis=2)
    l1b = l1b / DEN0
    p["l1b"] = l1b.reshape(BUCKETS * L2).astype(np.float32)

    # fc_1 reads 127*hl2, and hl2 is already in the [0,1] frame, so the 127
    # rides here: 127/DEN1 == 1/64.
    l2w = net.fc1_w.astype(np.float64) * (127.0 / DEN1)    # (B, L3, 64)
    p["l2w"] = np.ascontiguousarray(
        l2w.transpose(2, 0, 1).reshape(64, BUCKETS * L3)).astype(np.float32)
    l2b = net.fc1_b.astype(np.float64)                     # (B, L3)
    if debias:
        l2b = l2b - 0.5 * net.fc1_w.astype(np.float64).sum(axis=2)
    p["l2b"] = (l2b / DEN1).reshape(BUCKETS * L3).astype(np.float32)

    # fc_2 reads 127*[hl2; hl3] and its output is scaled into their internal
    # units by OUTQ; ARCHON's l3 is (fan_in, buckets) with one output each.
    p["l3w"] = np.ascontiguousarray(
        (net.fc2_w.astype(np.float64) * (127.0 * OUTQ)).T).astype(np.float32)

    # skip_0 = fc0[L2-2] - fc0[L2-1] is affine in the pairwise features, so it
    # splits exactly into the out_skip lane (linear part) and l3b (constant).
    dw = l1w[:, L2 - 2, :] - l1w[:, L2 - 1, :]             # (B, L1)
    p["skipw"] = np.ascontiguousarray(
        (dw * SKIPQ).T).astype(np.float32)                 # (L1, B)
    db = l1b[:, L2 - 2] - l1b[:, L2 - 1]                   # (B,)
    l3b = net.fc2_b.astype(np.float64)
    if debias:
        l3b = l3b - 0.5 * net.fc2_w.astype(np.float64).sum(axis=1)
    p["l3b"] = (l3b * OUTQ + db * SKIPQ).astype(np.float32)
    return p


def archon_arch():
    """The ArchSpec this checkpoint loads under (imported lazily: this tool is
    numpy-only so it runs without the jax venv)."""
    import dataclasses
    sys.path.insert(0, str(Path(__file__).resolve().parents[2] / "jax"))
    from archon_jax.spec import ArchSpec
    return dataclasses.replace(ArchSpec.full(), **ARCH_KWARGS)


def main() -> int:
    ap = argparse.ArgumentParser(description=__doc__)
    ap.add_argument("net", nargs="?", help="Stockfish .nnue")
    ap.add_argument("out", nargs="?", help="output raw warm checkpoint")
    ap.add_argument("--dump-arch", action="store_true",
                    help="print the ArchSpec kwargs as JSON and exit")
    ap.add_argument("--debias", action="store_true",
                    help="fold the naive 0.5-per-truncation correction "
                         "(MEASURED WORSE — see to_archon_params)")
    a = ap.parse_args()
    if a.dump_arch:
        import json
        print(json.dumps(ARCH_KWARGS, indent=2))
        return 0
    if not a.net or not a.out:
        ap.error("need <net.nnue> <out-raw.bin> (or --dump-arch)")

    print(f"reading {a.net}", flush=True)
    net = SFNet(a.net)
    print(f"  psq_w  {net.psq_w.shape} |max| {np.abs(net.psq_w).max()}\n"
          f"  thr_w  {net.thr_w.shape} |max| {np.abs(net.thr_w).max()}\n"
          f"  bias   {net.bias.shape} |max| {np.abs(net.bias).max()}", flush=True)
    params = to_archon_params(net, debias=a.debias)
    del net

    arch = archon_arch()
    want = arch.param_shapes()
    if set(want) != set(params):
        raise SystemExit(f"tensor set mismatch: arch wants {sorted(want)}, "
                         f"converter made {sorted(params)}")
    for k, shape in want.items():
        if params[k].shape != shape:
            raise SystemExit(f"{k}: made {params[k].shape}, arch wants {shape}")

    sys.path.insert(0, str(Path(__file__).resolve().parents[2] / "jax"))
    from archon_jax.checkpoint import write_raw
    Path(a.out).parent.mkdir(parents=True, exist_ok=True)
    write_raw(a.out, params)
    print(f"wrote {a.out} ({Path(a.out).stat().st_size / 1e6:.1f} MB), "
          f"{len(params)} tensors", flush=True)
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
