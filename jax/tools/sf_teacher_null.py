"""GATE 3 for the SF-net teacher: the alpha=0 BIT-IDENTITY null.

Everything this change touches must be inert until `distill_alpha > 0`. Three
arms, from the same warm checkpoint over the same batches:

  A  no teacher at all                       -- today's code, the reference
  B  teacher wired, distill_alpha = 0.0      -- must be BIT-IDENTICAL to A
  C  teacher wired, distill_alpha = 0.4      -- must DIFFER

C is not decoration. A null that passes because the harness never actually
wired the teacher proves nothing, and that failure is invisible: arm B would
match arm A for exactly the wrong reason. C is the evidence that this harness
CAN move the numbers, so B's match means inertness and not disconnection.

The config comes from `pump_train.build_cfg` rather than being rebuilt here —
a null that tests a configuration the trainer would never construct is worth
nothing.

  cd jax && ARCHON_JAX_ALLOW_GPU=1 ARCHON_JAX_FT_IMPL=pallas \
    uv run --no-project --with numpy==2.5.1 --with 'jax[cuda12]==0.11.0' \
       --with optax==0.2.8 -- python tools/sf_teacher_null.py \
       ../tools/nnue/runs/d49_sfblend_probe.json --steps 8
"""

from __future__ import annotations

import argparse
import dataclasses
import json
import sys
from pathlib import Path

import numpy as np

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))
sys.path.insert(0, str(Path(__file__).resolve().parent))

ROOT = Path(__file__).resolve().parents[2]


def compare(a: dict, b: dict) -> tuple[int, float, str]:
    """Bit-identity over the whole tree, plus the worst tensor if not."""
    diff, worst, wname = 0, 0.0, ""
    for k in sorted(a):
        if not np.array_equal(a[k], b[k]):
            diff += 1
            d = float(np.abs(a[k].astype(np.float64)
                             - b[k].astype(np.float64)).max())
            if d > worst:
                worst, wname = d, k
    return diff, worst, wname


def main() -> int:
    ap = argparse.ArgumentParser()
    ap.add_argument("spec")
    ap.add_argument("--steps", type=int, default=8)
    ap.add_argument("--resume-sb", type=int, default=1630)
    a = ap.parse_args()

    import jax
    import pump_train as pt
    from archon_jax import checkpoint as ck
    from archon_jax.data import BulletBin, make_mixed_batches
    from archon_jax.spec import ArchSpec
    from archon_jax.train import make_update_step

    spec = json.loads(Path(a.spec).read_text())
    arch = ArchSpec.from_run_spec(spec)
    base = pt.build_cfg(spec, a.resume_sb + 1)
    print(f"cfg: distill_alpha={base.distill_alpha}  "
          f"teacher_kp={'set' if base.teacher_kp else 'None'}  "
          f"per_corpus_target={'set' if base.per_corpus_target else 'None'}  "
          f"batch={base.batch_size}", flush=True)
    if not base.teacher_kp:
        raise SystemExit("spec carries no teacher_kp — nothing to null")

    mixbins = [(BulletBin(str(ROOT / p)), w) for p, w in spec["mix"]]
    warm = ROOT / spec["out"] / f"{spec['net_id']}-sb{a.resume_sb}-raw.bin"
    t_arch = dataclasses.replace(ArchSpec.full(),
                                 **{"factorize": True, "psqt": True,
                                    **spec["teacher_arch"]})
    t_params = ck.load_warm(ROOT / spec["teacher_warm"], t_arch)

    def run_arm(name, alpha, with_teacher):
        cfg = dataclasses.replace(base, distill_alpha=alpha)
        if not with_teacher:
            # Arm A is TODAY'S run: no teacher object, and no teacher_kp — so
            # the data path does not even emit t_bucket / teacher_mult.
            cfg = dataclasses.replace(cfg, teacher_kp=None)
        params = ck.load_warm(warm, arch)
        opt, step = make_update_step(
            arch, cfg, teacher=(t_params, t_arch) if with_teacher else None)
        opt_state = opt.init(params)
        params = {k: jax.numpy.asarray(v) for k, v in params.items()}
        ema = {k: jax.numpy.asarray(v) for k, v in params.items()}
        stream = make_mixed_batches(
            mixbins, cfg, policy=True,
            legal_mask=(spec.get("pol_mask")
                        if isinstance(spec.get("pol_mask"), str)
                        else bool(spec.get("pol_mask", False))),
            seed=int(spec["seed"]),
            workers=int(spec.get("workers", 4)),
            prefetch=int(spec.get("prefetch", 4)),
            shuffle_chunk=int(spec.get("shuffle_chunk", 0)),
            total_batches=cfg.batches_per_superbatch * cfg.superbatches,
            skip_batches=cfg.batches_per_superbatch * a.resume_sb)
        keys = None
        for i, batch in enumerate(stream):
            if i >= a.steps:
                break
            if keys is None:
                keys = sorted(batch)
            params, ema, opt_state, _ = step(params, ema, opt_state, batch)
        out = {k: np.asarray(v) for k, v in params.items()}
        del params, ema, opt_state, step, opt, stream
        print(f"  arm {name}: alpha={alpha} teacher={'yes' if with_teacher else 'no'}"
              f"  batch keys={keys}", flush=True)
        return out

    print(f"\nwarm {warm.name}, {a.steps} steps", flush=True)
    A = run_arm("A", 0.0, False)
    B = run_arm("B", 0.0, True)
    n, worst, wname = compare(A, B)
    print(f"\nNULL  A vs B (alpha=0):   {n} of {len(A)} tensors differ"
          + (f", worst |d|={worst:.3e} on {wname}" if n else "  -> BIT-IDENTICAL"))

    C = run_arm("C", 0.4, True)
    m, cworst, cname = compare(A, C)
    print(f"CTRL  A vs C (alpha=0.4): {m} of {len(A)} tensors differ"
          + (f", worst |d|={cworst:.3e} on {cname}" if m
             else "  -> IDENTICAL, so the null proves NOTHING"))

    ok = (n == 0) and (m > 0)
    print(f"\nGATE 3 {'PASS' if ok else 'FAIL'}", flush=True)
    return 0 if ok else 1


if __name__ == "__main__":
    raise SystemExit(main())
