#!/usr/bin/env python3
"""Score the disagreement head on the HELDOUT set.

WHY THIS EXISTS
    `dis_sgn` in the training log is not just an uncertainty metric -- it IS
    the corrected eval's mean squared error, and `dis_sgn_null` is the raw
    eval's. Algebraically, with r = target - sigmoid(value_logit):

        dis_sgn      = E[(dis_pred - r)^2] = E[((sigmoid(v) + dis_pred) - t)^2]
        dis_sgn_null = E[r^2]              = E[( sigmoid(v)             - t)^2]

    So D57's -56% on the training stream literally says "adding the head's
    output to the eval more than halves squared error against the label".
    That is an extraordinary claim for a 24k-parameter head reading a
    stop-gradiented trunk, and it is measured IN-SAMPLE on the training
    stream. This script asks the only question that matters: does it hold on
    data the head never saw?

WHAT IT PRINTS
    raw / corrected / corrected_shifted MSE, and the magnitude column against
    its own null. `corrected_shifted` subtracts the heldout residual MEAN
    before correcting, which separates "the head learned a constant offset"
    from "the head learned a function of the position" -- the same trap the
    dis_sgn zero-null had.

    A head that generalises shows corrected << raw on BOTH. A head that
    memorised shows corrected ~ raw here while the training log still reads
    -56%.

USAGE
    uv run --no-project --with numpy==2.5.1 --with jax[cuda12]==0.11.0 \
        --with optax==0.2.8 python tools/dis_heldout.py <raw.bin> <run.json>
"""
import json
import pathlib
import sys

import numpy as np

sys.path.insert(0, str(pathlib.Path(__file__).resolve().parents[1]))

from archon_jax import checkpoint as ck  # noqa: E402
from archon_jax.spec import ArchSpec, TrainConfig  # noqa: E402
from pump_train import HELDOUT, RECORD_DTYPE, map_batch, build_cfg  # noqa: E402


def main(ckpt: str, spec_path: str, batch_size: int = 256,
         ft_impl: str = "reference") -> int:
    import jax
    from archon_jax.model import forward
    from archon_jax.losses import expected_score

    spec = json.loads(pathlib.Path(spec_path).read_text())
    cfg: TrainConfig = build_cfg(spec, 1)
    # ArchSpec.from_run_spec, NOT a hand-built ArchSpec: the run spec drives
    # factoriser/psqt/policy tensors that a hand construction silently omits,
    # and load_warm then rejects the checkpoint for "tensors the arch lacks".
    arch = ArchSpec.from_run_spec(spec)
    assert arch.dis_proj, "this net has no disagreement head"

    params = ck.load_warm(pathlib.Path(ckpt), arch)
    jparams = {k: jax.numpy.asarray(v) for k, v in params.items()}

    # The run's OWN frame. Scoring a calibrated net against the uncalibrated
    # target reads as a regression by construction -- the trap documented on
    # heldout_value_loss.
    a, b_ = spec["heldout_target"]
    net_offset = spec.get("net_offset", 0.0)
    net_scale = spec.get("net_scale", 0.0)
    # Which residual this run's head was trained against.
    score_only = bool(spec.get("dis_score_only", False))
    print(f"residual column   {'target_score' if score_only else 'target'}"
          f"   (dis_score_only={score_only})")

    recs = np.memmap(HELDOUT, dtype=RECORD_DTYPE, mode="r")
    n = (len(recs) // batch_size) * batch_size
    # Cap the sample. The heldout set is 1,000,000 positions; the quantity
    # being measured is a mean over i.i.d. positions, so 200k already pins it
    # to ~0.2% relative and the rest is wall clock. Overridable because the
    # FULL set is what a headline number should quote.
    limit = int(__import__("os").environ.get("DIS_LIMIT", "200000"))
    if limit:
        n = min(n, (limit // batch_size) * batch_size)

    resid_all, mag_pred_all, sgn_pred_all = [], [], []
    for off in range(0, n, batch_size):
        batch = map_batch(np.array(recs[off:off + batch_size]), wdl=cfg.wdl,
                          score_multiplier=cfg.score_multiplier,
                          scale=b_, policy=False, target_offset=a)
        batch = {k: np.asarray(v) for k, v in batch.items()}
        out, _pol, _hl2, dis_pred = forward(
            jparams, arch, batch, ft_impl, with_dis=True)
        if net_scale > 0.0:
            q = np.asarray(expected_score(out, net_offset, net_scale))
        else:
            q = np.asarray(jax.nn.sigmoid(out))
        # SCORE THE HEAD AGAINST THE RESIDUAL IT WAS TRAINED ON. A run with
        # `dis_score_only` fits (target_score - q); scoring it against the
        # wdl-blended `target` measures a DIFFERENT QUANTITY and understates
        # the correlation by roughly sqrt(1 - outcome_share) -- the same
        # class of frame error as scoring a calibrated net against the
        # uncalibrated target. Caught 2026-08-06 after an early D58 read came
        # back at corr +0.046 and was nearly written up as a refutation.
        tcol = "target_score" if score_only else "target"
        resid_all.append(q - batch[tcol])              # NOTE sign: q - t
        dis_pred = np.asarray(dis_pred)
        sgn_pred_all.append(dis_pred[:, 0])
        mag_pred_all.append(dis_pred[:, 1])

    # The trainer's residual is (target - q); ours above is (q - target), so
    # flip once here rather than silently comparing two different signs.
    r = -np.concatenate(resid_all)
    sgn = np.concatenate(sgn_pred_all)
    mag = np.concatenate(mag_pred_all)

    raw = float(np.mean(r ** 2))
    corrected = float(np.mean((r - sgn) ** 2))
    rbar = float(np.mean(r))
    corrected_shift = float(np.mean(((r - rbar) - (sgn - np.mean(sgn))) ** 2))
    raw_shift = float(np.mean((r - rbar) ** 2))

    am = np.abs(r)
    mag_mse = float(np.mean((mag - am) ** 2))
    mag_null = float(np.mean((am - np.mean(am)) ** 2))
    # Rank quality is what a SEARCH consumer actually uses: it scales a margin
    # by relative uncertainty, so what matters is whether high-|r| positions
    # score high, not the absolute calibration.
    corr = float(np.corrcoef(mag, am)[0, 1])

    print(f"n                  {len(r):,}")
    print(f"resid mean         {rbar:+.6f}")
    print(f"raw   MSE          {raw:.6f}")
    print(f"corr  MSE          {corrected:.6f}   ({100*(corrected-raw)/raw:+.2f}% vs raw)")
    print(f"raw   MSE (demean) {raw_shift:.6f}")
    print(f"corr  MSE (demean) {corrected_shift:.6f}   "
          f"({100*(corrected_shift-raw_shift)/raw_shift:+.2f}% vs raw-demean)")
    print(f"dis_mag            {mag_mse:.6f} / null {mag_null:.6f}   "
          f"({100*(mag_mse-mag_null)/mag_null:+.2f}%)")
    print(f"corr(mag, |r|)     {corr:+.4f}")

    # ---- IS IT MIS-SCALED, OR IS IT UNCORRELATED? -------------------------
    # These separate a FIXABLE fault from a FATAL one, and without them a big
    # positive "corrected MSE" is unreadable. The head is trained on a
    # population whose residual sigma is ~3x the heldout set's (mix is
    # band75-100, heldout is uho-band0-5), so its outputs are expected to come
    # out too large in scale even if the RANKING is perfect. Correlation is
    # scale-free and answers the question the MSE cannot.
    #
    # best_* is the MSE after the optimal least-squares affine rescale, i.e.
    # the floor any deployment-time recalibration could reach:
    #   min_k E[(r - k*s)^2] = Var(r) * (1 - corr^2)   (plus the mean term)
    # If best_sgn is not meaningfully below raw, no rescale saves it.
    csgn = float(np.corrcoef(sgn, r)[0, 1])
    k_sgn = float(np.dot(sgn - np.mean(sgn), r - np.mean(r))
                  / max(np.dot(sgn - np.mean(sgn), sgn - np.mean(sgn)), 1e-12))
    best_sgn = float(np.mean((r - (np.mean(r) + k_sgn * (sgn - np.mean(sgn)))) ** 2))
    k_mag = float(np.dot(mag - np.mean(mag), am - np.mean(am))
                  / max(np.dot(mag - np.mean(mag), mag - np.mean(mag)), 1e-12))
    best_mag = float(np.mean((am - (np.mean(am) + k_mag * (mag - np.mean(mag)))) ** 2))
    print(f"corr(sgn, r)       {csgn:+.4f}   (r^2 = {csgn**2:.4f})")
    print(f"  best rescale k   {k_sgn:+.4f}")
    print(f"  best sgn MSE     {best_sgn:.6f}   ({100*(best_sgn-raw)/raw:+.2f}% vs raw)")
    print(f"  best mag MSE     {best_mag:.6f}   ({100*(best_mag-mag_null)/mag_null:+.2f}% vs null)")
    print(f"sigma(r) heldout   {float(np.std(r)):.6f}")
    print(f"sigma(sgn) pred    {float(np.std(sgn)):.6f}")
    return 0


if __name__ == "__main__":
    if len(sys.argv) < 3:
        print(__doc__)
        raise SystemExit(2)
    raise SystemExit(main(sys.argv[1], sys.argv[2]))
