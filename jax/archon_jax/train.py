"""The training loop: bullet run.rs semantics on the JAX stack.

- gradients of the MEAN loss (== bullet's summed loss x 1/batch_size adj)
- bullet_adamw (optim.py) with the ±1.98 post-clip
- cosine LR keyed on the 1-based superbatch (constant within a superbatch)
- per-superbatch loss/metric reporting (mean per position, like run.rs:276)
- checkpoints every save_rate superbatches: raw f32 (resume source) + quant
  (netpack input), byte-format-identical to bullet's

CPU-only by policy: import jax only under JAX_PLATFORMS=cpu.
"""

from __future__ import annotations

import os
import time
from functools import partial
from pathlib import Path

import numpy as np

from . import checkpoint
from .spec import ArchSpec, TrainConfig


def _require_cpu():
    # Safe-by-default: JAX grabs the GPU unless told otherwise, and for most
    # of this package's life a bullet trainer owned it. Explicit opt-in for
    # GPU sessions (benchmarks, shadow training) when the GPU is free:
    if os.environ.get("ARCHON_JAX_ALLOW_GPU") == "1":
        return
    if os.environ.get("JAX_PLATFORMS") != "cpu":
        raise RuntimeError(
            "refusing to import jax without JAX_PLATFORMS=cpu "
            "(set ARCHON_JAX_ALLOW_GPU=1 for a deliberate GPU session)")


def make_update_step(arch: ArchSpec, cfg: TrainConfig, ft_impl: str | None = None):
    """``ft_impl``: feature-transformer implementation for model.forward —
    "reference" (default), "pallas" (the GPU embedding-bag kernel,
    ft_kernel.py) or "scan". Defaults to $ARCHON_JAX_FT_IMPL so GPU sessions
    opt in by environment without touching call sites."""
    _require_cpu()
    import jax
    import jax.numpy as jnp

    from .losses import total_loss
    from .model import forward
    from .optim import bullet_adamw

    if ft_impl is None:
        ft_impl = os.environ.get("ARCHON_JAX_FT_IMPL", "reference")

    bps = cfg.batches_per_superbatch

    def lr_fn(count):
        sb = count // bps + cfg.start_superbatch  # 1-based superbatch
        initial = cfg.lr
        final = initial * 0.3 ** 3
        if cfg.lr_end_superbatch <= 1:
            return jnp.asarray(initial)
        t = jnp.clip((sb - 1.0) / (cfg.lr_end_superbatch - 1.0), 0.0, 1.0)
        lr = final + 0.5 * (initial - final) * (1.0 + jnp.cos(jnp.pi * t))
        if cfg.warmup_superbatches > 0.0 and cfg.start_superbatch == 1:
            # Per-step linear ramp over the first warmup_superbatches;
            # deviates from constant-within-sb only during the ramp.
            lr = lr * jnp.minimum(1.0, (count + 1.0) / (cfg.warmup_superbatches * bps))
        return lr

    opt = bullet_adamw(lr_fn)

    def loss_fn(params, batch):
        if cfg.ft_sparsity_lambda > 0.0:
            value_logit, pol_logits, _, (acc_stm, acc_ntm) = forward(
                params, arch, batch, ft_impl, with_acc=True)
            loss, metrics = total_loss(value_logit, pol_logits, batch["target"],
                                       batch.get("pol_target_id"), cfg.pol_lambda)
            # L1 on the post-CReLU activations = what the engine pays for
            # (density-linear l1_dot). Density metric = deploy-relevant
            # fraction of nonzero clamped activations.
            act = 0.5 * (jnp.mean(jnp.clip(acc_stm, 0.0, 1.0))
                         + jnp.mean(jnp.clip(acc_ntm, 0.0, 1.0)))
            density = 0.5 * (jnp.mean((acc_stm > 0.0).astype(jnp.float32))
                             + jnp.mean((acc_ntm > 0.0).astype(jnp.float32)))
            loss = loss + cfg.ft_sparsity_lambda * act
            metrics["ft_l1"] = act
            metrics["ft_density"] = density
            metrics["loss"] = loss
            return loss, metrics
        value_logit, pol_logits, _ = forward(params, arch, batch, ft_impl)
        return total_loss(value_logit, pol_logits, batch["target"],
                          batch.get("pol_target_id"), cfg.pol_lambda)

    import optax

    @partial(jax.jit, donate_argnums=(0, 1))
    def step(params, opt_state, batch):
        (_, metrics), grads = jax.value_and_grad(loss_fn, has_aux=True)(params, batch)
        updates, opt_state = opt.update(grads, opt_state, params)
        params = optax.apply_updates(params, updates)
        return params, opt_state, metrics

    if cfg.ema_decay <= 0.0:
        return opt, step

    d = cfg.ema_decay

    @partial(jax.jit, donate_argnums=(0, 1, 2))
    def step_ema(params, ema, opt_state, batch):
        (_, metrics), grads = jax.value_and_grad(loss_fn, has_aux=True)(params, batch)
        updates, opt_state = opt.update(grads, opt_state, params)
        params = optax.apply_updates(params, updates)
        ema = jax.tree.map(lambda e, p: d * e + (1.0 - d) * p, ema, params)
        return params, ema, opt_state, metrics

    return opt, step_ema


def train(arch: ArchSpec, cfg: TrainConfig, batches, *, out_dir: str,
          net_id: str, seed: int = 0x00D40FEA, save_rate: int = 10,
          log_every: int = 128, params=None):
    """``batches``: iterable of mapped numpy batch dicts (see data.make_batches).
    Runs until the iterator ends or superbatches complete."""
    _require_cpu()
    import jax

    from .model import init_params

    Path(out_dir).mkdir(parents=True, exist_ok=True)
    opt, step = make_update_step(arch, cfg)
    if params is None:
        params = init_params(arch, seed)
    opt_state = opt.init(params)

    bps = cfg.batches_per_superbatch
    running = {}
    t0 = time.time()
    for i, batch in enumerate(batches):
        params, opt_state, metrics = step(params, opt_state, batch)
        for k, v in metrics.items():
            running[k] = running.get(k, 0.0) + float(v)
        bidx, sb = i % bps, i // bps + 1
        if bidx % log_every == 0:
            print(f"  sb {sb} batch {bidx}: loss {float(metrics['loss']):.6f}")
        if bidx == bps - 1:
            parts = " | ".join(f"{k} {v / bps:.6f}" for k, v in sorted(running.items()))
            rate = (sb * bps * cfg.batch_size) / (time.time() - t0)
            print(f"superbatch {sb} done | {parts} | {rate:,.0f} pos/s")
            running = {}
            if sb % save_rate == 0 or sb == cfg.superbatches:
                save_checkpoint(params, arch, out_dir, f"{net_id}-sb{sb}")
        if i + 1 >= bps * cfg.superbatches:
            break
    host = jax.tree.map(np.asarray, params)
    save_checkpoint(host, arch, out_dir, f"{net_id}-final")
    return host


def save_checkpoint(params, arch: ArchSpec, out_dir: str, tag: str):
    import jax

    host = {k: np.asarray(v) for k, v in params.items()}
    raw = Path(out_dir) / f"{tag}-raw.bin"
    quant = Path(out_dir) / f"{tag}-quant.bin"
    checkpoint.write_raw(raw, host)
    n = checkpoint.write_quant(quant, host, arch)
    print(f"  saved {raw} + {quant} ({n} bytes)")
