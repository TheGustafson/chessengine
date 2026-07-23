#!/usr/bin/env python3
"""extract_head — pull an optional co-trained head out of a bullet raw checkpoint
into a flat-f32 sidecar for the engine's optional-head loader. Generalizes
extract_unc_head.py to ANY head added via `new_affine("<name>", ...)`:

    extract_head.py <raw.bin> <head_name> <sidecar.out>

    head_name = l3_var  (uncertainty, L3->BUCKETS)
              = l3_wdl  (WDL, L3->3)
              = l_polfrom / l_polto  (policy, HL->64)

Bullet names an affine's tensors `<name>w` (weights, [out,in] out-major) and
`<name>b` (biases, [out]). The raw format is repeated [name\\n][u64 LE count]
[count f32 LE] (same reader as net2wider.py). Output sidecar is flat little-endian
f32: all weights (out-major, row o = weights[o*in .. o*in+in]) then all biases —
exactly the layout the engine's propagate_l3_* / policy loaders index. The head is
kept OUT of the CENN (the value net stays byte-identical); a net with no sidecar
no-ops the head.
"""
import struct
import sys

import numpy as np


def read_tensors(path):
    data = open(path, "rb").read()
    off, out = 0, {}
    while off < len(data):
        nl = data.index(b"\n", off)
        name = data[off:nl].decode("ascii")
        off = nl + 1
        (count,) = struct.unpack_from("<Q", data, off)
        off += 8
        out[name] = np.frombuffer(data, dtype="<f4", count=count, offset=off).copy()
        off += count * 4
    return out


VALUE_TENSORS = {"l0_pstw", "l0_pstb", "l0_threatw", "l1w", "l1b", "l2w", "l2b", "l3w", "l3b"}


def main():
    if len(sys.argv) != 4:
        sys.exit("usage: extract_head.py <raw.bin> <head_name> <sidecar.out>")
    raw, head, out = sys.argv[1], sys.argv[2], sys.argv[3]
    tensors = read_tensors(raw)
    w, b = head + "w", head + "b"
    if w not in tensors or b not in tensors:
        heads = sorted(k for k in tensors if k not in VALUE_TENSORS)
        sys.exit(f"{raw}: missing {w}/{b}. Head tensors present: {heads or '(none — was λ>0 set?)'}")
    with open(out, "wb") as f:
        f.write(np.ascontiguousarray(tensors[w], dtype="<f4").tobytes())
        f.write(np.ascontiguousarray(tensors[b], dtype="<f4").tobytes())
    print(f"extract_head: {head} -> {out}  ({tensors[w].size} weights + {tensors[b].size} biases f32)")


if __name__ == "__main__":
    main()
