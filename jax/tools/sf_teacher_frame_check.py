"""GATE 2 for the SF-net teacher: does (k, p) actually land the teacher in the
LABEL frame — through THIS pipeline?

`sf-teacher-calibration-2026-08-01` fitted (k, p) with its own tooling. This
re-measures the finished product end to end: the converted checkpoint, our
`material` decode, our positional corpus indexing, our transcription of the
table. Any of those can be wrong in a way the study cannot see, and each
failure mode is silent — a mis-indexed corpus still produces a plausible
slope, just the wrong one.

**Test the statistic the table was FITTED to, not the one that reads best.**
`(k, p)` come from the study's model MP, which is a LOG-SPACE fit of
log(|label|/|teacher|) on log(blend), restricted to the decisive sign-agreeing
subpopulation (|teacher| > 150). That is a deliberate choice with a stated
reason — a level-space least-squares slope is dominated by the |eval| tail,
while the trainer's target is a saturating sigmoid, so what the loss actually
sees is the TYPICAL position. So the gate is:

  ratio   geometric mean of |label| / |converted| over that same
          subpopulation. 1.000 = the conversion landed.
  terc    median-ratio high-material tercile / low, |converted| > 100.
          1.000 = the exponent removed the material dependence.
  null    the same numbers with the teacher column PERMUTED. Without it a
          ratio near 1 proves nothing: a wrong constant times a right shape
          still lands near 1 on a symmetric distribution.

`ols0` is reported too and is NOT the gate — it is the tail-weighted view the
study rejected, and it reads 0.94-1.67 on a table that is correct on its own
terms. Kept visible because the divergence is real information about WHERE
the teacher and the labels disagree (it tracks R2), not an error.

  cd jax && JAX_PLATFORMS=cpu XLA_PYTHON_CLIENT_PREALLOCATE=false \
    uv run --no-project --with numpy==2.5.1 --with jax==0.11.0 \
       --with optax==0.2.8 -- python tools/sf_teacher_frame_check.py \
       ../tools/nnue/runs/d49_sfblend_probe.json \
       ../data/nnue/teachers/sf-eca43a97-teacher-raw.bin --n 4000
"""

from __future__ import annotations

import argparse
import json
import os
import sys
from pathlib import Path

import numpy as np

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))
os.environ.setdefault("JAX_PLATFORMS", "cpu")
os.environ.setdefault("XLA_PYTHON_CLIENT_PREALLOCATE", "false")

ROOT = Path(__file__).resolve().parents[2]


def _load_conv():
    import importlib.util
    p = ROOT / "tools/nnue/sf_to_archon_warm.py"
    spec = importlib.util.spec_from_file_location("sf_to_archon_warm", p)
    mod = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(mod)
    return mod


def ols0(y: np.ndarray, x: np.ndarray) -> float:
    """Slope through the origin — the TAIL-WEIGHTED view, reported not gated."""
    return float((x @ y) / (x @ x))


def geo_ratio(y: np.ndarray, x: np.ndarray, raw: np.ndarray) -> float:
    """Geometric mean of |y|/|x| on the study's own MP subpopulation
    (|raw teacher| > 150, signs agree, y != 0). This is the quantity the
    log-space fit set to 1 by construction, so it is the quantity that
    verifies the table survived the trip through this pipeline."""
    g = (np.abs(raw) > 150.0) & (np.sign(y) == np.sign(raw)) & (y != 0)
    if g.sum() < 500:
        return float("nan")
    return float(np.exp(np.mean(np.log(np.abs(y[g]) / np.abs(x[g])))))


def terc_ratio(y: np.ndarray, x: np.ndarray, mat: np.ndarray) -> float:
    """median(y/x) in the high material tercile over the low — the study's
    `terciles()`, floor 100 on |x|."""
    g = np.abs(x) > 100.0
    if g.sum() < 200:
        return float("nan")
    mm, rr = mat[g], y[g] / x[g]
    lo_c, hi_c = np.percentile(mm, [33.333, 66.667])
    lo = float(np.median(rr[mm <= lo_c]))
    return float(np.median(rr[mm >= hi_c]) / lo) if lo else float("nan")


def main() -> int:
    ap = argparse.ArgumentParser()
    ap.add_argument("spec", help="run json carrying mix + teacher_kp")
    ap.add_argument("warm", help="converted SF teacher checkpoint")
    ap.add_argument("--n", type=int, default=4000)
    ap.add_argument("--seed", type=int, default=20260801)
    ap.add_argument("--chunk", type=int, default=64)
    a = ap.parse_args()

    import jax.numpy as jnp
    from archon_jax import checkpoint as ck
    from archon_jax.features import map_batch
    from archon_jax.model import forward
    from archon_jax.spec import EVAL_BLEND_BASE, RECORD_DTYPE

    conv = _load_conv()
    spec = json.loads(Path(a.spec).read_text())
    kp = spec["teacher_kp"]
    mix = spec["mix"]
    if len(kp) != len(mix):
        raise SystemExit(f"teacher_kp {len(kp)} != mix {len(mix)}")
    smul = float(spec.get("score_multiplier", 1.0))

    arch = conv.archon_arch()
    params = {k: jnp.asarray(v) for k, v in ck.load_warm(a.warm, arch).items()}
    rng = np.random.default_rng(a.seed)

    print(f"{'#':>2} {'corpus':<26} {'k':>7} {'p':>6} | {'RATIO':>6} "
          f"{'null':>6} {'terc':>6} | {'ols0':>6} {'R2':>6}")
    ratios, weights = [], []
    for i, ((path, wt), (k, p)) in enumerate(zip(mix, kp)):
        mm = np.memmap(ROOT / path, dtype=RECORD_DTYPE, mode="r")
        recs = np.asarray(mm[np.sort(rng.integers(0, len(mm), size=a.n))])
        del mm
        b = map_batch(recs, wdl=0.0, score_multiplier=1.0, scale=400.0,
                      policy=False, threats=True, sf_bucket=True)
        mat = np.asarray(b["material"], np.float64)
        tb = np.asarray(b["t_bucket"])
        raw = np.empty(len(recs), np.float64)
        for j in range(0, len(recs), a.chunk):
            sl = slice(j, min(j + a.chunk, len(recs)))
            raw[sl] = np.asarray(forward(params, arch, {
                "hk_stm": jnp.asarray(b["hk_stm"][sl]),
                "hk_ntm": jnp.asarray(b["hk_ntm"][sl]),
                "th_stm": jnp.asarray(b["th_stm"][sl]),
                "th_ntm": jnp.asarray(b["th_ntm"][sl]),
                "bucket": jnp.asarray(tb[sl])})[0], np.float64)

        conv_t = k * (1.0 + mat / EVAL_BLEND_BASE) ** p * raw
        label = recs["score"].astype(np.float64) * smul
        ratio = geo_ratio(label, conv_t, raw)
        perm = rng.permutation(conv_t)
        null = geo_ratio(label, perm, raw)
        slope = ols0(label, conv_t)
        r2 = 1.0 - (label - slope * conv_t).var() / label.var()
        terc = terc_ratio(label, conv_t, mat)
        print(f"{i:>2} {Path(path).stem.replace('.shuf', ''):<26} {k:>7.4f} "
              f"{p:>6.2f} | {ratio:>6.3f} {null:>6.3f} {terc:>6.3f} | "
              f"{slope:>6.3f} {r2:>6.3f}")
        ratios.append(ratio)
        weights.append(wt)

    w = np.asarray(weights) / np.sum(weights)
    s = np.asarray(ratios)
    print(f"\nGATE 2 statistic (geometric ratio, want 1.000):"
          f"\n  weighted mean {float(w @ s):.4f}   range {s.min():.4f}-"
          f"{s.max():.4f}   worst |1-ratio| {np.abs(1 - s).max() * 100:.1f}%")
    ok = np.abs(1 - s).max() < 0.10
    print(f"GATE 2 {'PASS' if ok else 'FAIL'}")
    return 0 if ok else 1


if __name__ == "__main__":
    raise SystemExit(main())
