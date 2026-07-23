#!/usr/bin/env python3
"""net2wider — grow a trained D40 sota net's accumulator width, function-preservingly.

Net2WiderNet (Chen et al. 2016) for our arch: copy a trained narrow net into a
wider one with the new neurons ZERO-initialised, so the wide net computes the
*identical* function until you fine-tune it. Warm-starting a wider net this way
costs a short fine-tune instead of a full from-scratch run, and inherits all the
parent's training.

    net2wider.py <src_raw.bin> <dst_raw.bin> <new_hl>

<src_raw.bin>  a bullet D40 checkpoint (`*-raw.bin`, the INIT_RAW / weights.load_from
               format: repeated [name\\n][u64 LE count][count f32 LE]).
<new_hl>       target accumulator width (must be even and > source width; the source
               width and every other arch dim are auto-detected from the raw).

Then warm-start the trainer with D40_INIT_RAW=<dst_raw.bin> (built for the new HL),
and netpack the result with D40_ACC=<new_hl>.

── Why the reindex is not a naive append ──────────────────────────────────────
The trunk is `crelu().pairwise_mul()`, which multiplies element i by element
i+HL/2 (inference: "first half × second half", STM half in [0,HL/2)). A naive
append (old neurons 0..OLD, new OLD..NEW) would RE-PAIR the old neurons and change
the function. So the old first half stays in the new first half and the old second
half moves to the start of the new second half; the freed slots
[OLD/2, OLD/2+PAD) and [NEW-PAD, NEW) (PAD=(NEW-OLD)/2) are the zero-init capacity:

    R(x) = x            for x <  OLD/2
    R(x) = x + PAD      for x >= OLD/2

── HL-axis tensors (layout confirmed from d40_features.rs's own loader) ────────
    l0_pstw    [features, HL]  feature-major -> reindex HL cols
    l0_pstb    [HL]                          -> reindex
    l0_threatw [features, HL]  feature-major -> reindex HL cols
    l1w        [HL, BUCKETS*L2] (transpose_saved) -> reindex HL rows (head input)
    l1b,l2w,l2b,l3w,l3b                       -> copied verbatim (don't touch HL)

── SYMMETRY-BREAKING (critical — else the new capacity is DEAD) ────────────────
Pure zero-init new neurons receive ZERO gradient forever: the pairwise-mul pairs a
new neuron only with another new neuron, so new_product = 0·0 = 0 and its head
weight (also 0) gets grad ∝ product = 0, while the new FT weights get grad ∝ head
weight = 0. The net would train identically to the narrow one, wasting the width.
Fix: init the new FT weights (l0_pstw/l0_threatw new cols) with SMALL NOISE so the
new products are nonzero, but keep the new HEAD weights (l1w new rows) at ZERO.
The output is still IDENTICAL (nonzero product × zero head weight = 0, rmse 0.00),
but now the head weight gets a real gradient → grows off zero → then feeds gradient
back to the FT weights: the capacity bootstraps itself alive from a function-
identical start. `--noise 0` restores pure-zero (dead) init for the exact-copy check.

Proven function-preserving: a frozen (LR=0) trainer save netpacked to CENN evals
bit-identically to the parent (rmse 0.00cp) — tools/nnue/grow_net.sh runs that check.
"""
import struct
import sys
import numpy as np


def read_tensors(path):
    data = open(path, "rb").read()
    off, out = 0, []
    while off < len(data):
        nl = data.index(b"\n", off)
        name = data[off:nl].decode()
        off = nl + 1
        (count,) = struct.unpack_from("<Q", data, off)
        off += 8
        arr = np.frombuffer(data, dtype="<f4", count=count, offset=off).copy()
        off += count * 4
        out.append((name, arr))
    return out


def widen(src, dst, new_hl, noise=0.1, seed=1234):
    rng = np.random.default_rng(seed)
    tensors = read_tensors(src)
    by_name = dict(tensors)
    if "l0_pstb" not in by_name:
        sys.exit(f"{src}: not a D40 raw (no l0_pstb tensor; found {list(by_name)})")

    old_hl = by_name["l0_pstb"].size                       # bias is [HL]
    if new_hl <= old_hl or new_hl % 2 or old_hl % 2 or (new_hl - old_hl) % 2:
        sys.exit(f"new_hl={new_hl} must be even, > old_hl={old_hl}, with even difference")
    pad = (new_hl - old_hl) // 2
    half = old_hl // 2
    # the new HL slots (freed by the pairing-preserving reindex): first-half tail
    # and second-half tail.
    new_cols = np.r_[half:half + pad, new_hl - pad:new_hl]

    def reindex_last(m, fill_noise=False):    # reindex the LAST axis (…,OLD) -> (…,NEW)
        out = np.zeros(m.shape[:-1] + (new_hl,), np.float32)
        out[..., :half] = m[..., :half]
        out[..., half + pad: old_hl + pad] = m[..., half:]
        if fill_noise and noise > 0:          # symmetry-break the new INPUT (FT) neurons
            sigma = noise * float(m.std())
            out[..., new_cols] = rng.normal(0.0, sigma, out[..., new_cols].shape).astype(np.float32)
        return out

    # infer the other axis of each HL-tensor from its element count / old_hl
    out_tensors = []
    for name, arr in tensors:
        if name in ("l0_pstw", "l0_threatw"):              # [features, HL] -> cols (FT: noise)
            feats = arr.size // old_hl
            arr = reindex_last(arr.reshape(feats, old_hl), fill_noise=True).reshape(-1)
        elif name == "l0_pstb":                             # [HL] bias -> zero new (FT weights break symmetry)
            arr = reindex_last(arr.reshape(old_hl))
        elif name == "l1w":                                 # [HL, out] head -> rows stay ZERO (preserve output)
            out_dim = arr.size // old_hl
            m = arr.reshape(old_hl, out_dim)
            wide = np.zeros((new_hl, out_dim), np.float32)
            wide[:half] = m[:half]
            wide[half + pad: old_hl + pad] = m[half:]
            arr = wide.reshape(-1)
        out_tensors.append((name, arr))

    with open(dst, "wb") as f:
        for name, arr in out_tensors:
            f.write(name.encode() + b"\n")
            f.write(struct.pack("<Q", arr.size))
            f.write(np.ascontiguousarray(arr, dtype="<f4").tobytes())

    kind = "zero (DEAD — exact-copy check only)" if noise == 0 else f"noise σ={noise}×wstd (trainable)"
    print(f"net2wider: HL {old_hl} -> {new_hl} (pad {pad}/side, {2*pad} new neurons, FT init: {kind}; head weights zero => rmse 0)")
    print("  " + ", ".join(f"{n}={a.size}" for n, a in out_tensors))
    print(f"  wrote {dst}")


if __name__ == "__main__":
    if len(sys.argv) not in (4, 5):
        sys.exit("usage: net2wider.py <src_raw.bin> <dst_raw.bin> <new_hl> [noise=0.1]")
    nz = float(sys.argv[4]) if len(sys.argv) == 5 else 0.1
    widen(sys.argv[1], sys.argv[2], int(sys.argv[3]), noise=nz)
