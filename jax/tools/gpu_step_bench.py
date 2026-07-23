"""Phase-2 go/no-go: full-arch jit'd train-step throughput on the GPU.

Measures pos/s for the exact D47c recipe shape (batch 8192, full 22528+60720
feature space, policy head, bullet-AdamW) against bullet's ~210k pos/s on the
same GPU. Gate: >=0.8x (168k). Uses REAL heldout rows; 8 distinct batches
cycled to defeat cache-residency flattery. Note: this pipeline carries
pol_target_id (sparse) rather than bullet's dense 1858-target+mask uploads —
a designed improvement counted in its favor.

  cd jax && XLA_PYTHON_CLIENT_PREALLOCATE=false ARCHON_JAX_ALLOW_GPU=1 \
    uv run --no-project --with jax[cuda12] --with optax -- \
    python tools/gpu_step_bench.py [n_batches=8] [timed_steps=40] [batch_size]
                                   [impl=reference|pallas|scan] [mode=step|report]

Positional int args are unchanged from the original interface. New keyword
tokens:
  impl=...     feature-transformer implementation for the timed step
               (default reference — the original jnp.take path; pallas = the
               ft_kernel.py embedding-bag kernel; scan = the pure-XLA
               fallback)
  mode=report  the full go/no-go report: train-step + forward-only pos/s at
               batch {1024, 4096, 8192} for every impl, numerical parity vs
               the CPU reference forward, training-step equivalence, and
               peak GPU memory. (mode=step is the original single
               measurement.)
"""

import os
import sys
import time
from pathlib import Path

import numpy as np

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))
os.environ.setdefault("XLA_PYTHON_CLIENT_PREALLOCATE", "false")

from archon_jax.features import map_batch  # noqa: E402
from archon_jax.model import forward_jit, init_params  # noqa: E402
from archon_jax.spec import RECORD_DTYPE, ArchSpec, TrainConfig  # noqa: E402
from archon_jax.train import make_update_step  # noqa: E402

BULLET_REF = 210_000
GATE = 168_000  # 0.8x


def _cfg_with_batch(cfg, batch_size):
    return type(cfg)(**{**cfg.__dict__, "batch_size": int(batch_size)})


def load_batches(cfg, n_batches, *, quiet=False):
    dump = Path(__file__).resolve().parents[2] / "data/nnue/heldout/uho-band0-5.bin"
    recs = np.memmap(dump, dtype=RECORD_DTYPE, mode="r")
    need = n_batches * cfg.batch_size
    idx = np.linspace(0, len(recs) - 1, num=need, dtype=np.int64)
    rows = np.array(recs[idx])

    batches = []
    t0 = time.time()
    for i in range(n_batches):
        b = map_batch(rows[i * cfg.batch_size:(i + 1) * cfg.batch_size],
                      wdl=cfg.wdl, score_multiplier=cfg.score_multiplier,
                      scale=cfg.scale, policy=True)
        batches.append({k: np.asarray(v) for k, v in b.items()})
    prep = time.time() - t0
    if not quiet:
        act = np.mean([(b["hk_stm"] >= 0).sum(1) + (b["th_stm"] >= 0).sum(1)
                       for b in batches])
        print(f"host map_batch: {need / prep:,.0f} pos/s (prep only, single "
              f"thread); avg active rows/perspective {act:.1f}")
    return batches


def _is_oom(e) -> bool:
    return "RESOURCE_EXHAUSTED" in str(e) or "Out of memory" in str(e)


def bench_step(arch, cfg, batches, timed_steps, impl, *, quiet=False):
    """Timed full train step (fwd+bwd+bullet-AdamW). pos/s, or None on OOM."""
    import jax

    n_batches = len(batches)
    try:
        opt, step = make_update_step(arch, cfg, ft_impl=impl)
        params = init_params(arch, seed=0x00D40FEA)
        opt_state = opt.init(params)

        t0 = time.time()
        params, opt_state, m = step(params, opt_state, batches[0])
        jax.block_until_ready(m["loss"])
        if not quiet:
            print(f"  [{impl}] compile+first step: {time.time() - t0:.1f}s  "
                  f"loss {float(m['loss']):.5f}")
        for i in range(1, 3):
            params, opt_state, m = step(params, opt_state, batches[i % n_batches])
        jax.block_until_ready(m["loss"])

        t0 = time.time()
        for i in range(timed_steps):
            params, opt_state, m = step(params, opt_state, batches[i % n_batches])
        jax.block_until_ready(m["loss"])
        dt = time.time() - t0
        return timed_steps * cfg.batch_size / dt
    except Exception as e:  # noqa: BLE001 — OOM is a reportable result here
        if _is_oom(e):
            return None
        raise


def bench_forward(arch, cfg, batches, timed_steps, impl):
    """Forward-only (value+policy logits) pos/s — the datagen/eval number."""
    import jax

    try:
        params = init_params(arch, seed=0x00D40FEA)
        out = forward_jit(params, arch, batches[0], impl)
        jax.block_until_ready(out[0])
        t0 = time.time()
        for i in range(timed_steps):
            out = forward_jit(params, arch, batches[i % len(batches)], impl)
        jax.block_until_ready(out[0])
        dt = time.time() - t0
        return timed_steps * cfg.batch_size / dt
    except Exception as e:  # noqa: BLE001
        if _is_oom(e):
            return None
        raise


def perturbed_params(arch, seed=0x00D40FEA):
    """init_params + noise on every tensor so the zero-init threat table (and
    biases) actually contribute — parity must exercise both FT tables."""
    import jax

    params = init_params(arch, seed)
    key = jax.random.PRNGKey(seed ^ 0xA5A5)
    out = {}
    for name in sorted(params):
        key, sub = jax.random.split(key)
        p = params[name]
        out[name] = p + 0.02 * jax.random.normal(sub, p.shape, p.dtype)
    return out


def parity_report(arch, cfg, n_rows=512):
    """GPU impls vs the CPU reference forward (the path proven byte-faithful
    to production) on real heldout rows + perturbed params."""
    import jax
    import jax.numpy as jnp

    batch = load_batches(_cfg_with_batch(cfg, n_rows), 1, quiet=True)[0]
    params = perturbed_params(arch)
    host_params = {k: np.asarray(v) for k, v in params.items()}

    cpu = jax.devices("cpu")[0]
    params_cpu = jax.device_put(host_params, cpu)
    batch_cpu = jax.device_put(batch, cpu)
    with jax.default_device(cpu):
        v_ref, p_ref = forward_jit(params_cpu, arch, batch_cpu, "reference")
    v_ref, p_ref = np.asarray(v_ref), np.asarray(p_ref)

    results = {}
    for impl in ("pallas", "scan"):
        v_gpu, p_gpu = forward_jit(params, arch, batch, impl)
        dv = float(jnp.max(jnp.abs(jnp.asarray(v_ref) - v_gpu)))
        flips = int(np.sum(np.argmax(np.asarray(p_gpu), -1) != np.argmax(p_ref, -1)))
        results[impl] = (dv, flips)
        print(f"  {impl:9s} vs CPU reference ({n_rows} real rows): "
              f"max|dvalue_logit| {dv:.2e} "
              f"({'PASS' if dv < 1e-3 else 'FAIL'} vs 1e-3), "
              f"policy argmax flips {flips}/{n_rows}")
    return results


def step_equivalence(arch, cfg, n_steps=6, batch_size=1024):
    """A few real optimiser steps, reference vs pallas FT, same init + data:
    the loss curves must agree to float noise (atomics reorder the adds)."""
    import jax

    cfg_s = _cfg_with_batch(cfg, batch_size)
    batches = load_batches(cfg_s, n_steps, quiet=True)
    curves = {}
    for impl in ("reference", "pallas"):
        opt, step = make_update_step(arch, cfg_s, ft_impl=impl)
        params = init_params(arch, seed=0x00D40FEA)
        opt_state = opt.init(params)
        losses = []
        for b in batches:
            params, opt_state, m = step(params, opt_state, b)
            losses.append(float(m["loss"]))
        jax.block_until_ready(m["loss"])
        curves[impl] = losses
    worst = max(abs(a - b) / max(abs(a), 1e-12)
                for a, b in zip(curves["reference"], curves["pallas"]))
    print(f"  loss curve  reference: {['%.6f' % x for x in curves['reference']]}")
    print(f"  loss curve  pallas   : {['%.6f' % x for x in curves['pallas']]}")
    print(f"  worst relative step-loss delta over {n_steps} steps: {worst:.2e}")
    return worst


def report(n_batches, timed_steps):
    """The full go/no-go report (docs/research/jax-gpu-kernel-2026-07-23.md)."""
    import jax

    arch, cfg = ArchSpec.full(), TrainConfig()
    dev = jax.devices()[0]

    print("== numerical parity (gate: max|dvalue_logit| < 1e-3) ==")
    parity_report(arch, cfg)

    print("== training-step equivalence (batch 1024, same data+init) ==")
    step_equivalence(arch, cfg)

    print("== throughput: train step (fwd+bwd+bullet-AdamW) ==")
    peak = None
    results = {}
    for batch_size in (8192, 4096, 1024):
        cfg_b = _cfg_with_batch(cfg, batch_size)
        batches = load_batches(cfg_b, n_batches, quiet=(batch_size != 8192))
        # reference last: its OOM at >=2048 must not fragment the pool under
        # the kernel measurements.
        for impl in ("pallas", "scan", "reference"):
            pos_s = bench_step(arch, cfg_b, batches, timed_steps, impl, quiet=True)
            results[(impl, batch_size)] = pos_s
            note = f"{pos_s:,.0f} pos/s" if pos_s else "OOM"
            print(f"  batch {batch_size:5d}  {impl:9s}: {note}")
            if impl == "pallas" and batch_size == 8192:
                stats = dev.memory_stats() or {}
                peak = stats.get("peak_bytes_in_use")
                fwd = bench_forward(arch, cfg_b, batches, timed_steps, impl)
                print(f"  batch {batch_size:5d}  pallas forward-only: "
                      f"{fwd:,.0f} pos/s")
        del batches

    if peak is not None:
        print(f"peak GPU bytes in use (through pallas@8192): {peak / 2**30:.2f} GiB")

    best = results.get(("pallas", 8192))
    if best:
        print(f"verdict: {best:,.0f} pos/s = {best / BULLET_REF:.2f}x bullet "
              f"({'GO — >=0.8x gate' if best >= GATE else 'BELOW GATE'})")
    return 0


def main(n_batches=8, timed_steps=40, batch_size=0, impl="reference",
         mode="step"):
    import jax

    dev = jax.devices()[0]
    print(f"device: {dev} ({dev.platform})")
    if dev.platform != "gpu":
        print("FATAL: not on GPU — aborting (bench is meaningless on CPU)")
        return 2

    if mode == "report":
        return report(n_batches, timed_steps)

    arch, cfg = ArchSpec.full(), TrainConfig()
    if batch_size:
        cfg = _cfg_with_batch(cfg, batch_size)
    batches = load_batches(cfg, n_batches)

    pos_s = bench_step(arch, cfg, batches, timed_steps, impl)
    if pos_s is None:
        print(f"[{impl}] OOM at batch {cfg.batch_size}")
        return 1
    print(f"timed: {timed_steps} steps = {pos_s:,.0f} pos/s "
          f"(bullet ref ~{BULLET_REF // 1000}k; 0.8x gate = {GATE // 1000}k)")
    print(f"verdict: {'GO' if pos_s >= GATE else 'BELOW GATE'} "
          f"({pos_s / BULLET_REF:.2f}x bullet)")
    return 0


if __name__ == "__main__":
    ints, kw = [], {}
    for tok in sys.argv[1:]:
        if "=" in tok:
            k, v = tok.split("=", 1)
            kw[k] = v
        else:
            ints.append(int(tok))
    raise SystemExit(main(*ints, **kw))
