#!/usr/bin/env python3
"""Assemble the D47U warm file: d47s trunk (frozen) + FRESH 64-wide value
head + zero skip lane + fresh polmove on the 64 substrate.

GPU-minimal by design (user 2026-07-26 ~20:1x): the trunk — 99% of the
parameters and all of the knowledge — transfers verbatim; only the thin
head retrains (D47Q-precedent: heads converge in ~30-60sb).

Usage: assemble_d47u_warm.py <d47s-final-ema-raw.bin> <out-raw.bin> [--l2 64]
Run from repo root with the jax venv deps (JAX_PLATFORMS=cpu).
"""
import sys, dataclasses
sys.path.insert(0, "jax")
import numpy as np
from archon_jax.spec import ArchSpec
from archon_jax import checkpoint as ck
from archon_jax.model import init_params

src, out = sys.argv[1], sys.argv[2]
def argval(flag, default):
    return int(sys.argv[sys.argv.index(flag) + 1]) if flag in sys.argv else default
l2 = argval("--l2", 64)
l3 = argval("--l3", 32)
hl = argval("--slice-hl", 0)         # 0 = keep full width (D47U mode)

old_arch = dataclasses.replace(
    ArchSpec.full(), factorize=True, psqt=True,
    hl=argval("--src-hl", ArchSpec.full().hl),
    l2=argval("--src-l2", ArchSpec.full().l2),
    l3=argval("--src-l3", ArchSpec.full().l3))
new_arch = dataclasses.replace(old_arch, l2=l2, l3=l3, out_skip=True,
                               hl=hl or old_arch.hl)

old = ck.load_warm(src, old_arch)
fresh = init_params(new_arch, seed=0xD470)
TRUNK = {"l0_pstw", "l0_pstb", "l0_threatw", "l0_factw",
         "l0_psqt_hkw", "l0_psqt_thw"}

def slice_cols(w):
    """Pair-respecting width slice (D47V): our pairwise multiplies column i
    with i+hl/2, so keeping pairs means taking [0:new/2] + [old/2:old/2+new/2].
    Applies to every hl-COLUMN tensor (FT tables + bias + factorizer)."""
    oh, nh = old_arch.hl, new_arch.hl
    return np.concatenate([w[..., :nh // 2], w[..., oh // 2: oh // 2 + nh // 2]],
                          axis=-1)

params = {}
for name, shape in new_arch.param_shapes().items():
    if name in TRUNK:
        w = old[name]
        if hl and w.shape[-1] == old_arch.hl:
            w = slice_cols(np.asarray(w))
        assert w.shape == shape, (name, w.shape, shape)
        params[name] = np.ascontiguousarray(w)
    elif name == "skipw":
        params[name] = np.zeros(shape, np.float32)   # exact no-op lane
    else:
        params[name] = np.asarray(fresh[name])       # fresh head/polmove
ck.write_raw(out, params)
mode = f"slice-hl {new_arch.hl} (pair-respecting)" if hl else "full trunk (frozen)"
print(f"assembled {out}: {mode} from {src}, head l2={l2} l3={l3}, zero skip")
