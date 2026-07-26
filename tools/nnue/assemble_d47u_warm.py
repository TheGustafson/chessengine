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
l2 = int(sys.argv[sys.argv.index("--l2") + 1]) if "--l2" in sys.argv else 64

old_arch = dataclasses.replace(ArchSpec.full(), factorize=True, psqt=True)
new_arch = dataclasses.replace(old_arch, l2=l2, out_skip=True)

old = ck.load_warm(src, old_arch)
fresh = init_params(new_arch, seed=0xD470)
TRUNK = {"l0_pstw", "l0_pstb", "l0_threatw", "l0_factw",
         "l0_psqt_hkw", "l0_psqt_thw"}
params = {}
for name, shape in new_arch.param_shapes().items():
    if name in TRUNK:
        assert old[name].shape == shape, (name, old[name].shape, shape)
        params[name] = old[name]
    elif name == "skipw":
        params[name] = np.zeros(shape, np.float32)   # exact no-op lane
    else:
        params[name] = np.asarray(fresh[name])       # fresh head/polmove
ck.write_raw(out, params)
print(f"assembled {out}: trunk from {src}, fresh head l2={l2}, zero skip")
print("train_only for the leg:",
      sorted(set(new_arch.param_shapes()) - TRUNK))
