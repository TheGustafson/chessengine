"""Loss functions matching bullet's graph exactly (d40_features.rs 1059-1116).

bullet computes SUMMED-over-batch losses and then scales gradients by
1/batch_size (run.rs:163: ``tgf = 1/batch_size``), which is mathematically the
gradient of the MEAN loss. We therefore compute means directly; the reported
numbers match bullet's per-superbatch console output (run.rs:276: loss/B).

Value:  (sigmoid(out) - target)^2, target = wdl*result + (1-wdl)*sigmoid(score*mult/scale)
        (the blend is precomputed in the loader, d40:877-879).
Policy: softmax-CE over 1858 logits with the ADDITIVE legal mask already in
        the logits: CE = -Σ target·log softmax(logits + mask). Rows with no
        packed move have an all-zero target AND all-zero mask -> exactly zero
        CE contribution (d40:891-895). Mask value -30.0 (NOT -1e4/-inf): soft
        enough that exp() never underflows — with a one-hot target,
        -inf-masking would be safe in jax's log_softmax, but we keep -30.0 for
        exact loss-value parity with the incumbent trainer.
Total:  value + λ_pol · policy (additive, d40:1097-1116); the WDL head
        (λ_wdl, d40:1082-1093) is not carried in D47c (λ=0 ⇒ tensor absent).
"""

from __future__ import annotations

import jax
import jax.numpy as jnp


def value_loss(value_logit: jnp.ndarray, target: jnp.ndarray) -> jnp.ndarray:
    """Mean sigmoid-MSE (bullet: sig.squared_error(targets).reduce_sum_batch(),
    d40:1069, then /B at reporting)."""
    sig = jax.nn.sigmoid(value_logit)
    return jnp.mean((sig - target) ** 2)


def policy_loss(pol_logits: jnp.ndarray, pol_target_id: jnp.ndarray) -> jnp.ndarray:
    """Mean masked softmax-CE. ``pol_logits`` already include the additive
    mask; ``pol_target_id`` is -1 for rows without a packed move (zero CE,
    matching the all-zero-target rows in bullet).

    bullet's op is elementwise -t·log(softmax(x)) summed over rows+batch
    (softmax.rs forward), == one-hot CE for our targets.
    """
    logp = jax.nn.log_softmax(pol_logits, axis=-1)
    has = pol_target_id >= 0
    picked = jnp.take_along_axis(
        logp, jnp.maximum(pol_target_id, 0)[:, None], axis=-1)[:, 0]
    ce = jnp.where(has, -picked, 0.0)
    return jnp.mean(ce)


def total_loss(value_logit, pol_logits, target, pol_target_id, pol_lambda: float):
    v = value_loss(value_logit, target)
    metrics = {"value": v}
    loss = v
    if pol_logits is not None and pol_lambda > 0.0:
        p = policy_loss(pol_logits, pol_target_id)
        metrics["pol_ce"] = p
        loss = loss + pol_lambda * p
    metrics["loss"] = loss
    return loss, metrics
