#!/usr/bin/env python3
"""EXP-UNC-1: extract the value-uncertainty (log-variance) head from a bullet
raw checkpoint into the runtime sidecar consumed by D40_UNC_HEAD.

The heteroscedastic-NLL trainer (D40_HETERO_NLL=1) adds a parallel `l3_var`
affine (L3 -> NUM_BUCKETS) whose weights live in the RAW checkpoint (bullet
saves every trainable tensor there) but NOT in the quantised bin / CENN. This
reads the raw ckpt, pulls `l3_varw` + `l3_varb`, and writes the sidecar:

    [ l3_var_weights : BUCKETS*L3 f32, BUCKETS-major (b*L3 + j) ]
    [ l3_var_biases  : BUCKETS      f32 ]

matching SotaHead.l3_var_weights / propagate_l3_var (`vw[bucket*L3 + j]`).

Raw ckpt format (bullet ModelWeights::write_into): a concatenation of tensors,
each `<ascii id>\n` + `<u64 LE value-count>` + `<count * f32 LE>`, in BTreeMap
(sorted-by-name) order.

Layout note: bullet stores the `l3` affine weight as (L3, NUM_BUCKETS) and the
save-format applies `.transpose()` so the CENN's l3w is (NUM_BUCKETS, L3)
bucket-major — exactly the runtime layout. `l3_var` has the identical shape, so
we apply the SAME transpose here. (Validated against the mean head: pass
--validate <cenn> to assert transpose(raw l3w) == the CENN's l3w.)

Usage:
    extract_unc_head.py <raw.bin> <out.sidecar> [--validate <net.cenn>]
"""
import struct
import sys

L3 = 32
BUCKETS = 8


def parse_raw(path):
    """Return {name: [f32, ...]} for every tensor in a bullet raw checkpoint."""
    with open(path, "rb") as f:
        buf = f.read()
    tensors, off = {}, 0
    while off < len(buf):
        nl = buf.index(b"\n", off)
        name = buf[off:nl].decode("ascii")
        off = nl + 1
        (count,) = struct.unpack_from("<Q", buf, off)
        off += 8
        vals = list(struct.unpack_from(f"<{count}f", buf, off))
        off += count * 4
        tensors[name] = vals
    return tensors


def transpose_l3(native):
    """bullet-native (L3, BUCKETS) [j*BUCKETS + b] -> runtime (BUCKETS, L3)
    [b*L3 + j], the exact transform the save-format's .transpose() applies to
    l3w before it becomes the CENN's bucket-major block."""
    assert len(native) == L3 * BUCKETS, f"l3 weight is {len(native)}, expected {L3*BUCKETS}"
    out = [0.0] * (BUCKETS * L3)
    for j in range(L3):
        for b in range(BUCKETS):
            out[b * L3 + j] = native[j * BUCKETS + b]
    return out


def main():
    if len(sys.argv) < 3:
        sys.exit("usage: extract_unc_head.py <raw.bin> <out.sidecar> [--validate <net.cenn>]")
    raw_path, out_path = sys.argv[1], sys.argv[2]
    tensors = parse_raw(raw_path)

    for needed in ("l3_varw", "l3_varb"):
        if needed not in tensors:
            sys.exit(f"{raw_path}: no '{needed}' tensor — was this trained with D40_HETERO_NLL=1? "
                     f"(present: {sorted(tensors)})")

    var_w = transpose_l3(tensors["l3_varw"])
    var_b = tensors["l3_varb"]
    assert len(var_b) == BUCKETS, f"l3_varb is {len(var_b)}, expected {BUCKETS}"

    with open(out_path, "wb") as f:
        f.write(struct.pack(f"<{len(var_w)}f", *var_w))
        f.write(struct.pack(f"<{len(var_b)}f", *var_b))
    print(f"[extract-unc] wrote {out_path}: {len(var_w)} weights + {len(var_b)} biases "
          f"({(len(var_w)+len(var_b))*4} bytes)")

    if "--validate" in sys.argv:
        # Self-check the transpose on the MEAN head: transpose(raw l3w) must
        # equal the CENN's l3w block (netpack reads it (BUCKETS, L3) bucket-major).
        cenn = sys.argv[sys.argv.index("--validate") + 1]
        _validate_against_cenn(tensors["l3w"], cenn)


def _validate_against_cenn(raw_l3w, cenn_path):
    """Assert transpose(raw l3w) matches the CENN's l3w (dequantised f32 block).
    Locates the l3w block from the end: the D40 CENN ends with l3w (BUCKETS*L3
    f32) then l3b (BUCKETS f32), before <=63B of bullet padding."""
    import numpy as np
    with open(cenn_path, "rb") as f:
        raw = f.read()
    n_l3w, n_l3b = BUCKETS * L3, BUCKETS
    # Scan trailing padding (<=63 bytes of the ascii "bullet" cycle) off the end.
    pad = 0
    while pad <= 63 and raw[len(raw) - 1 - pad] == b"bullet"[(len(raw) - 1 - pad) % 6]:
        pad += 1
    end = len(raw) - pad
    l3b_off = end - n_l3b * 4
    l3w_off = l3b_off - n_l3w * 4
    cenn_l3w = np.frombuffer(raw[l3w_off:l3b_off], dtype="<f4")
    ours = np.array(transpose_l3(raw_l3w), dtype="<f4")
    err = float(np.max(np.abs(cenn_l3w - ours)))
    print(f"[extract-unc] validate: max|transpose(raw l3w) - cenn l3w| = {err:.3e} "
          f"(l3w_off={l3w_off}, pad={pad})")
    assert err < 1e-3, "transpose mismatch — the l3 layout assumption is wrong; fix transpose_l3"
    print("[extract-unc] transpose VALIDATED against the mean head")


if __name__ == "__main__":
    main()
