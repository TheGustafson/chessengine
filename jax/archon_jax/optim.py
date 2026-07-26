"""bullet's AdamW as an optax GradientTransformation — EXACT kernel parity.

The bullet kernel (adam.rs:35-52, params at adam.rs:31) per weight, with
g = grad_of_mean_loss (run.rs:163) and rate = lr(step):

    p *= 1 - decay*rate                  # decoupled decay, BEFORE the step
    m  = b1*m + (1-b1)*g
    v  = b2*v + (1-b2)*g^2
    p -= rate * m / (sqrt(v) + 1e-8)     # NO bias correction; eps OUTSIDE sqrt
    p  = clip(p, -1.98, 1.98)            # EVERY tensor, weights AND biases

Differences from optax.adamw that make the stock optimiser WRONG here:
  - optax applies bias correction (m_hat/v_hat); bullet does not.
  - optax adds eps inside/outside differently per variant; bullet is
    sqrt(v)+eps.
  - optax's weight_decay is added to the update; bullet multiplies the param
    before the Adam step.
  - the ±1.98 post-clip has no stock optax equivalent (it is what guarantees
    i8 quantisation at 64x never overflows: 1.98*64 = 126.72 → round 127,
    netpack.py:22-26).

The clip is implemented here as part of the update: the transform returns
``new_p - p`` computed from the clipped target, so ``optax.apply_updates``
lands exactly on the bullet trajectory.
"""

from __future__ import annotations

from typing import NamedTuple

import jax
import jax.numpy as jnp
import optax

from .spec import ADAM_BETA1, ADAM_BETA2, ADAM_DECAY, ADAM_EPS, WEIGHT_CLIP


class BulletAdamWState(NamedTuple):
    momentum: optax.Params
    velocity: optax.Params


def bullet_adamw(lr_schedule, decay: float = ADAM_DECAY,
                 beta1: float = ADAM_BETA1, beta2: float = ADAM_BETA2,
                 eps: float = ADAM_EPS, clip: float = WEIGHT_CLIP,
                 clip_map: dict[str, float] | None = None,
                 decay_map: dict[str, float] | None = None,
                 ) -> optax.GradientTransformation:
    """``lr_schedule``: step_index -> learning rate (see spec.cosine_lr; bullet
    holds LR constant within a superbatch).

    ``clip_map`` / ``decay_map``: OPTIONAL per-tensor overrides keyed by param
    name (v1.5, 2026-07-26). Defaults keep the historical single ±1.98 clip on
    every tensor and decay on everything — bullet parity.

    WHY THE OVERRIDES EXIST: the uniform 1.98 was inherited from bullet and is
    MEASURED to be binding on nearly every tensor of a live net, including
    tensors exported as f32 (no quantization reason to clip at all) and the
    i16 FT whose real ceiling is ~128, not 1.98. SF derives its clip per layer
    from that layer's own quantization scale. See spec.clip_map_for_arch().
    SF also never decays biases; decay_map lets us match."""

    def init(params):
        z = jax.tree.map(jnp.zeros_like, params)
        return (BulletAdamWState(z, jax.tree.map(jnp.zeros_like, params)),
                jnp.zeros([], jnp.int32))

    def update(grads, state, params):
        inner, count = state
        rate = lr_schedule(count)

        def one(g, m, v, p, c, d):
            m = beta1 * m + (1 - beta1) * g
            v = beta2 * v + (1 - beta2) * g * g
            p_decayed = p * (1 - d * rate)
            p_new = p_decayed - rate * m / (jnp.sqrt(v) + eps)
            if c is not None:
                p_new = jnp.clip(p_new, -c, c)
            return p_new - p, m, v

        names = sorted(grads) if isinstance(grads, dict) else None
        flat_g, treedef = jax.tree.flatten(grads)
        flat_m = treedef.flatten_up_to(inner.momentum)
        flat_v = treedef.flatten_up_to(inner.velocity)
        flat_p = treedef.flatten_up_to(params)
        # jax.tree.flatten on a dict yields values in sorted-key order, so
        # `names` lines up with the flattened lists.
        if names is None:
            cs = [clip] * len(flat_g)
            ds = [decay] * len(flat_g)
        else:
            cs = [(clip_map or {}).get(n, clip) for n in names]
            ds = [(decay_map or {}).get(n, decay) for n in names]
        out = [one(g, m, v, p, c, d)
               for g, m, v, p, c, d in zip(flat_g, flat_m, flat_v, flat_p, cs, ds)]
        updates = treedef.unflatten([o[0] for o in out])
        new_m = treedef.unflatten([o[1] for o in out])
        new_v = treedef.unflatten([o[2] for o in out])
        return updates, (BulletAdamWState(new_m, new_v), count + 1)

    return optax.GradientTransformation(init, update)
