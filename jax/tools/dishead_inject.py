#!/usr/bin/env python3
"""Inject an offline-fitted disagreement head into a raw checkpoint.

WHY THIS IS SAFE TO DO OUTSIDE THE TRAINER
    `head_forward` puts a `stop_gradient` on the trunk input, so the head can
    never influence the value path. Fitting it on cached `hl1` and writing the
    four tensors back is therefore not an approximation of a trainer run -- it
    optimises the same parameters against the same objective. The value
    tensors are copied through untouched, so the packed net must be
    BIT-IDENTICAL in play to its parent; that is the property to verify with
    `bench`, and it is what makes the head testable as a pure addition.

FRAME (checked, not assumed)
    The engine's `DisHead::forward` multiplies each u8 lane by `FT_DEQUANT`,
    whose doc states it returns "the trainer's hl1 in [0,1]", and applies
    `clamp(h, 0, 1)` as its hidden activation. The offline fit used exactly
    that: `jnp.clip(hl1 @ W + b, 0, 1)` on post-`fake_quant_act` hl1. So the
    weights transfer with NO rescaling.

    The head's outputs are `(search - static) / SCALE` with SCALE = 600, i.e.
    the network_scale unit. A consumer in search must multiply by 600 to get
    internal centipawns. That factor lives in the npz and is echoed here so it
    cannot be lost between the fit and the engine.
"""
import pathlib
import sys

import numpy as np

sys.path.insert(0, str(pathlib.Path(__file__).resolve().parents[1]))

from archon_jax import checkpoint as ck  # noqa: E402
from archon_jax.spec import ArchSpec  # noqa: E402


def main(parent_raw: str, npz: str, out_raw: str, run_spec: str) -> int:
    import json
    spec = json.loads(pathlib.Path(run_spec).read_text())
    arch = ArchSpec.from_run_spec(spec)
    assert arch.dis_proj, "run spec has no dis head"

    params = ck.load_warm(pathlib.Path(parent_raw), arch)
    w = np.load(npz)

    want = {
        "dis_projw": (arch.hl, arch.dis_proj),
        "dis_projb": (arch.dis_proj,),
        "dis_outw": (arch.dis_proj, 2),
        "dis_outb": (2,),
    }
    for k, shape in want.items():
        a = np.asarray(w[k], dtype=np.float32)
        # Shape-check every tensor. A silent reshape here would produce a net
        # that loads, plays identically (the value path is untouched) and
        # emits garbage from the head -- the hardest possible bug to notice,
        # because every cheap check still passes.
        if a.shape != shape:
            print(f"FATAL: {k} is {a.shape}, arch wants {shape}", file=sys.stderr)
            return 3
        if not np.isfinite(a).all():
            print(f"FATAL: {k} has non-finite values", file=sys.stderr)
            return 3
        params[k] = a

    ck.write_raw(pathlib.Path(out_raw), params)
    scale = float(w["scale"]) if "scale" in w else float("nan")
    print(f"wrote {out_raw}")
    print(f"  dis_projw {want['dis_projw']}  dis_outw {want['dis_outw']}")
    print(f"  head predicts (search - static) / {scale:g}  "
          f"-> multiply by {scale:g} for internal cp")
    print("  value tensors copied through: packed net must bench IDENTICALLY")
    return 0


if __name__ == "__main__":
    if len(sys.argv) < 5:
        print(__doc__)
        raise SystemExit(2)
    raise SystemExit(main(*sys.argv[1:5]))
