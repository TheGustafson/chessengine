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


def make_update_step(arch: ArchSpec, cfg: TrainConfig, ft_impl: str | None = None,
                     train_only: list[str] | None = None,
                     teacher: tuple | None = None):
    """``ft_impl``: feature-transformer implementation for model.forward —
    "reference" (default), "pallas" (the GPU embedding-bag kernel,
    ft_kernel.py) or "scan". Defaults to $ARCHON_JAX_FT_IMPL so GPU sessions
    opt in by environment without touching call sites.

    ``train_only``: param names allowed to update (frozen-trunk head training,
    D47L). Everything else is exactly frozen: grads are zeroed (Adam moments
    stay 0) AND the post-update values are restored to the pre-step tensors —
    the restore is what makes AdamW's grad-independent weight decay a no-op
    on frozen params. None (default) = train everything, today's behavior."""
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
        if cfg.warmup_superbatches > 0.0:
            # Per-step linear ramp over the first warmup_superbatches OF THIS
            # LEG (count is within-run, so a resumed leg warms up too when
            # configured — the fresh-Adam transient at sb31 showed resumes
            # hitting full LR with empty moments; ~0.25sb smooths it).
            # Default 0.0 = no-op; fresh-run behavior unchanged.
            lr = lr * jnp.minimum(1.0, (count + 1.0) / (cfg.warmup_superbatches * bps))
        return lr

    clip_map = decay_map = None
    if getattr(cfg, "per_tensor_clip", False):
        from .spec import clip_map_for_arch
        clip_map = clip_map_for_arch(arch)
    if getattr(cfg, "no_bias_decay", False):
        from .spec import decay_map_no_bias
        decay_map = decay_map_no_bias(arch, cfg.weight_decay)
    opt = bullet_adamw(lr_fn, clip_map=clip_map, decay_map=decay_map)

    def sparsity_lambda(count):
        """ft_sparsity_lambda for this step, with the D47M ramp: linear
        0→target over the first N superbatches of a FRESH run. Constant at
        the target when the ramp is off or on resume (same gate as the LR
        warmup — a resumed leg is past the freeze-out transient)."""
        lam = cfg.ft_sparsity_lambda
        if cfg.ft_sparsity_warmup_superbatches > 0.0 and cfg.start_superbatch == 1:
            ramp = jnp.minimum(
                1.0, (count + 1.0) / (cfg.ft_sparsity_warmup_superbatches * bps))
            return lam * ramp
        return jnp.asarray(lam)

    a_end = float(getattr(cfg, "distill_alpha_end", -1.0))

    def distill_alpha_at(count):
        """Per-step distill α: linear ramp distill_alpha → distill_alpha_end
        over the WHOLE schedule (nnue-pytorch precedent: --start-lambda 1.0
        --end-lambda 0.75 — start on the trusted denoised signal, fold the
        ground truth in as training proceeds). Anchored on the ABSOLUTE
        superbatch (start_superbatch + within-run count) so a resumed leg
        lands on the same global ramp despite the fresh Adam count.
        a_end < 0 (default) = constant α — every existing config unchanged."""
        if a_end < 0.0 or d_alpha <= 0.0:
            return jnp.asarray(d_alpha)
        sb_abs = (cfg.start_superbatch - 1.0) + count / bps
        frac = jnp.clip(sb_abs / float(cfg.superbatches), 0.0, 1.0)
        return d_alpha + (a_end - d_alpha) * frac

    t_params = t_arch = None
    t_frame = None
    d_alpha = float(getattr(cfg, "distill_alpha", 0.0) or 0.0)
    if teacher is not None:
        t_params, t_arch = teacher
        t_params = {k: jnp.asarray(v) for k, v in t_params.items()}
        f_off = float(getattr(cfg, "teacher_frame_offset", 0.0) or 0.0)
        if f_off > 0.0:
            # The teacher was trained in a DIFFERENT target frame (draw-aware
            # double sigmoid, offset f_off / scale f_scl). Its sigmoid output
            # estimates ITS transform of the eval, not ours; blending it raw
            # with our plain-sigmoid(s/scale) label term mixes frames and the
            # student settles between them (the 1.378x scale gotcha,
            # 2026-07-26). Map teacher LOGIT -> implied cp (inverse of its
            # transform) -> our frame, via a 1cp lookup table over ±12k cp.
            # The table is indexed by LOGIT, not probability: logit(T(s)) is
            # asymptotically linear in s (slope 1/f_scl), so the grid stays
            # STRICTLY increasing in float32 across the whole range, where a
            # probability-space grid saturates and develops ties beyond
            # ~±1400cp. jnp.interp clamps beyond both ends = correct
            # saturation. T(s) = 0.5(sig((s-a)/b) + sig((s+a)/b)) and
            # 1-T(s) = 0.5(sig(-(s-a)/b) + sig(-(s+a)/b)) — both sums of
            # positive stable terms, so log(T)-log(1-T) is float64-exact.
            f_scl = float(cfg.teacher_frame_scale)
            assert f_scl > 0.0, "teacher_frame_offset set without teacher_frame_scale"
            s_grid = np.arange(-12000.0, 12001.0, dtype=np.float64)

            def _sig(x):
                return 1.0 / (1.0 + np.exp(-x))

            t_hi = 0.5 * (_sig((s_grid - f_off) / f_scl)
                          + _sig((s_grid + f_off) / f_scl))
            t_lo = 0.5 * (_sig(-(s_grid - f_off) / f_scl)
                          + _sig(-(s_grid + f_off) / f_scl))
            x_grid = np.log(t_hi) - np.log(t_lo)
            p_grid = _sig(s_grid / float(cfg.scale))
            t_frame = (jnp.asarray(x_grid, jnp.float32),
                       jnp.asarray(p_grid, jnp.float32))

    def _teacher_targets(batch):
        """Teacher forward (no grad): blended value target + policy logits.
        The teacher shares the batch's sparse indices — its FT gather is the
        same cost class as the student's; total step ~2x, paid only on
        distill legs (D47V)."""
        tv, tp, _ = forward(t_params, t_arch, batch, ft_impl)
        if t_frame is not None:
            # interp on the raw logit — see the frame-LUT construction above
            tv = jnp.interp(tv, t_frame[0], t_frame[1])
        else:
            tv = jax.nn.sigmoid(tv)
        tv = jax.lax.stop_gradient(tv)
        tp = None if tp is None else jax.lax.stop_gradient(tp)
        return tv, tp

    def loss_fn(params, batch, lam, da):
        if t_params is not None and d_alpha > 0.0:
            tv, tp = _teacher_targets(batch)
            batch = dict(batch)
            batch["target"] = da * tv + (1.0 - da) * batch["target"]
            if tp is not None:
                # policy distill: soft targets via teacher softmax -> CE
                # against student logits == KL + const. Reuses pol_lambda.
                batch["_pol_soft"] = jax.nn.softmax(tp)
        if cfg.ft_sparsity_lambda > 0.0:
            value_logit, pol_logits, _, (acc_stm, acc_ntm) = forward(
                params, arch, batch, ft_impl, with_acc=True)
            loss, metrics = total_loss(value_logit, pol_logits, batch["target"],
                                       batch.get("pol_target_id"), cfg.pol_lambda,
                                       getattr(cfg, "pow_exp", 2.0),
                                       pol_soft=batch.get("_pol_soft"))
            # L1 on the post-CReLU activations = what the engine pays for
            # (density-linear l1_dot). Density metric = deploy-relevant
            # fraction of nonzero clamped activations.
            act = 0.5 * (jnp.mean(jnp.clip(acc_stm, 0.0, 1.0))
                         + jnp.mean(jnp.clip(acc_ntm, 0.0, 1.0)))
            density = 0.5 * (jnp.mean((acc_stm > 0.0).astype(jnp.float32))
                             + jnp.mean((acc_ntm > 0.0).astype(jnp.float32)))
            loss = loss + lam * act
            metrics["ft_l1"] = act
            metrics["ft_density"] = density
            metrics["ft_lambda"] = lam
            metrics["loss"] = loss
            return loss, metrics
        value_logit, pol_logits, _ = forward(params, arch, batch, ft_impl)
        return total_loss(value_logit, pol_logits, batch["target"],
                          batch.get("pol_target_id"), cfg.pol_lambda,
                          getattr(cfg, "pow_exp", 2.0),
                          pol_soft=batch.get("_pol_soft"))

    import optax

    allow = set(train_only) if train_only is not None else None
    if allow is not None:
        unknown = allow - set(arch.param_shapes())
        if unknown:
            raise ValueError(f"train_only names not in arch: {sorted(unknown)}")

    gnorm = float(getattr(cfg, "grad_clip_norm", 0.0) or 0.0)

    def masked_update(params, opt_state, grads):
        if gnorm > 0.0:
            # global-norm clip (SF hard-codes 2.0). Applied BEFORE masking so
            # the norm reflects the real gradient, and before the optimizer so
            # Adam moments see the clipped values.
            sq = sum(jnp.sum(g * g) for g in jax.tree.leaves(grads))
            scale = jnp.minimum(1.0, gnorm / (jnp.sqrt(sq) + 1e-12))
            grads = jax.tree.map(lambda g: g * scale, grads)
        if allow is not None:
            grads = {k: (g if k in allow else jnp.zeros_like(g))
                     for k, g in grads.items()}
        updates, opt_state = opt.update(grads, opt_state, params)
        new = optax.apply_updates(params, updates)
        if allow is not None:
            new = {k: (v if k in allow else params[k]) for k, v in new.items()}
        return new, opt_state

    @partial(jax.jit, donate_argnums=(0, 1))
    def step(params, opt_state, batch):
        # opt_state = (BulletAdamWState, count) — the same step counter the
        # LR schedule keys on (optim.py init/update).
        lam = sparsity_lambda(opt_state[1])
        da = distill_alpha_at(opt_state[1])
        (_, metrics), grads = jax.value_and_grad(loss_fn, has_aux=True)(
            params, batch, lam, da)
        params, opt_state = masked_update(params, opt_state, grads)
        return params, opt_state, metrics

    if cfg.ema_decay <= 0.0:
        return opt, step

    d = cfg.ema_decay

    @partial(jax.jit, donate_argnums=(0, 1, 2))
    def step_ema(params, ema, opt_state, batch):
        lam = sparsity_lambda(opt_state[1])
        da = distill_alpha_at(opt_state[1])
        (_, metrics), grads = jax.value_and_grad(loss_fn, has_aux=True)(
            params, batch, lam, da)
        params, opt_state = masked_update(params, opt_state, grads)
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
