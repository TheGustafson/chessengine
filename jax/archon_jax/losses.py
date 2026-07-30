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


def expected_score(x: jnp.ndarray, offset: float, scale: float) -> jnp.ndarray:
    """Stockfish's OFFSET TWO-SIGMOID: a score -> an expected score in [0,1].

        0.5 * (1 + sig((x-offset)/scale) - sig((-x-offset)/scale))

    The offset is what creates the draw band; a plain `sigmoid(x/k)` is
    draw-blind and pins every quiet position near 0.5.

    **SF applies this on BOTH sides of the loss and we only did the label side**
    (`nnue-pytorch/model/nnue.py:39-70`):

        qf = 0.5*(1 + sig((scorenet-in_offset)/in_scaling) - sig(...))   # NETWORK
        pf = 0.5*(1 + sig((score-out_offset)/out_scaling) - sig(...))    # LABEL
        pt = pf*lambda + t*(1-lambda)

    D48Y2 fitted per-corpus constants for `pf` and left `qf` a plain sigmoid.
    That is very likely why its heldout R² improved monotonically (0.8815 ->
    0.8906) while its strength did not move: **the network-output frame stayed
    uncorrected.** The training-gap study is explicit that the network side is
    what absorbs a systematic frame mismatch between the net's output units and
    the labelling engine's — i.e. exactly our 2.07x frame problem.

    SF's shipped constants (HPO-tuned, `nettest/threats.yaml:78-102`):
    in_scaling 295.65 / in_offset 285.27; out_scaling 379.99 / out_offset
    289.13. Ours must be fitted in OUR units — `value_logit` is not cp — so
    these are a shape reference, not values to copy.
    """
    return 0.5 * (1.0 + jax.nn.sigmoid((x - offset) / scale)
                      - jax.nn.sigmoid((-x - offset) / scale))


def value_loss(value_logit: jnp.ndarray, target: jnp.ndarray,
               pow_exp: float = 2.0,
               net_offset: float = 0.0, net_scale: float = 0.0,
               qp_asymmetry: float = 0.0, weight=None) -> jnp.ndarray:
    """Mean sigmoid power-error (bullet: sig.squared_error(targets)
    .reduce_sum_batch(), d40:1069, then /B at reporting).

    ``pow_exp`` 2.0 is the historical MSE. SF uses 2.5 (their docs report 2.6
    also training well): a higher exponent weights precision more and outlier
    accuracy less — the axis on which we trail SF18. Implemented on |.| so
    non-integer exponents are well defined; eps keeps the gradient finite at
    exactly zero error.

    ``net_scale`` > 0 switches the NETWORK side from a plain sigmoid to the
    offset two-sigmoid (see ``expected_score``) — the half of SF's transform we
    were missing. 0.0 keeps the historical behaviour bit-exact.

    ``qp_asymmetry`` reproduces SF's `qp_asymmetry = 0.2287`: overestimation is
    penalised ~23% harder than underestimation. 0.0 = symmetric = unchanged.
    """
    if net_scale > 0.0:
        sig = expected_score(value_logit, net_offset, net_scale)
    else:
        sig = jax.nn.sigmoid(value_logit)
    if qp_asymmetry == 0.0 and pow_exp == 2.0 and weight is None:
        return jnp.mean((sig - target) ** 2)
    err = (jnp.abs(sig - target) + 1e-12) ** pow_exp
    if qp_asymmetry != 0.0:
        err = err * ((sig > target) * qp_asymmetry + 1.0)
    if weight is not None:
        # SF's shaping is a probabilistic SKIP; a fixed-batch pipeline gets the
        # same thing in expectation as a weighted mean (their loss is likewise
        # `(loss*weights).sum()/weights.sum()`).
        return jnp.sum(err * weight) / (jnp.sum(weight) + 1e-12)
    return jnp.mean(err)


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


def total_loss(value_logit, pol_logits, target, pol_target_id, pol_lambda: float,
               pow_exp: float = 2.0, pol_soft=None,
               dis_pred=None, dis_lambda: float = 0.0,
               net_offset: float = 0.0, net_scale: float = 0.0,
               qp_asymmetry: float = 0.0, weight=None):
    v = value_loss(value_logit, target, pow_exp,
                   net_offset=net_offset, net_scale=net_scale,
                   qp_asymmetry=qp_asymmetry, weight=weight)
    metrics = {"value": v}
    loss = v
    if pol_logits is not None and pol_lambda > 0.0:
        if pol_soft is not None:
            # D47V distillation: CE against the teacher's softmax (== KL up
            # to the teacher's constant entropy). Masking rides in the
            # logits exactly as with hard targets.
            logp = jax.nn.log_softmax(pol_logits)
            p = -(pol_soft * logp).sum(axis=-1).mean()
        else:
            p = policy_loss(pol_logits, pol_target_id)
        metrics["pol_ce"] = p
        loss = loss + pol_lambda * p
    if dis_pred is not None and dis_lambda > 0.0:
        # D48W disagreement head: predict the value head's own error
        # magnitude in probability space. Target is stop-gradient so the
        # head observes, never steers, the value fit.
        err = jax.lax.stop_gradient(
            jnp.abs(jax.nn.sigmoid(value_logit) - target))
        d = jnp.mean((dis_pred - err) ** 2)
        metrics["dis"] = d
        loss = loss + dis_lambda * d
    metrics["loss"] = loss
    return loss, metrics
