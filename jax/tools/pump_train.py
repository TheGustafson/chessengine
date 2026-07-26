"""Generalized pump-stage driver (stage-4/5/6 of the release campaign).

Everything stage3_train.py proved, plus the pump-chain upgrades:
EMA shadow weights (the shippable net), LR warmup ramp, config-file mix.

Config JSON (tools/nnue/runs/stageN_pump.json):
{
  "net_id": "d47e-stage4",
  "warm": "data/nnue/experiments/d47d-stage3/d47d-stage3-sb240-raw.bin",
  "heldout_expect": 0.0,            // 0 = skip the reproduce assert value
  "lr": 0.0005, "lr_end_superbatch": 331, "superbatches": 240,
  "warmup_superbatches": 2.0, "ema_decay": 0.99984,
  "pol_lambda": 0.00125, "ft_sparsity_lambda": 0.001,
  "seed": 123, "out": "data/nnue/experiments/d47e-stage4",
  "mix": [["data/nnue/jaxcorpora/a.shuf.bin", 0.30], ...]
}

Run (from jax/):
  PYTHONUNBUFFERED=1 ARCHON_JAX_ALLOW_GPU=1 ARCHON_JAX_FT_IMPL=pallas \
    uv run ... python tools/pump_train.py <config.json> [resume_sb]

Ship-anneal note (stage-6): set lr_end_superbatch == superbatches and a
small lr — the cosine then lands at 0.027x, the one sanctioned near-zero.
"""

from __future__ import annotations

import dataclasses
import json
import sys
import time
from pathlib import Path

import numpy as np

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))

from archon_jax import checkpoint as ck                      # noqa: E402
from archon_jax.data import BulletBin, make_mixed_batches    # noqa: E402
from archon_jax.features import map_batch                    # noqa: E402
from archon_jax.spec import ArchSpec, RECORD_DTYPE, TrainConfig  # noqa: E402
from archon_jax.train import make_update_step                # noqa: E402

ROOT = Path(__file__).resolve().parents[2]
HELDOUT = ROOT / "data/nnue/heldout/uho-band0-5.bin"
HELDOUT_TOL = 0.00012


def heldout_value_loss(params, arch, ft_impl="pallas", batch_size=8192):
    import jax
    from archon_jax.model import forward

    cfg = TrainConfig()
    recs = np.memmap(HELDOUT, dtype=RECORD_DTYPE, mode="r")
    n = (len(recs) // batch_size) * batch_size
    total = 0.0
    jparams = {k: jax.numpy.asarray(v) for k, v in params.items()}
    for off in range(0, n, batch_size):
        b = map_batch(np.array(recs[off:off + batch_size]), wdl=cfg.wdl,
                      score_multiplier=cfg.score_multiplier, scale=cfg.scale,
                      policy=False)
        b = {k: np.asarray(v) for k, v in b.items()}
        out = forward(jparams, arch, b, ft_impl)[0]
        total += float(np.sum((np.asarray(jax.nn.sigmoid(out)) - b["target"]) ** 2))
    return total / n


def save_pair(host, arch, out: Path, name: str) -> None:
    ck.write_raw(out / f"{name}-raw.bin", host)
    ck.write_quant(out / f"{name}-quant.bin", host, arch)


def main(cfg_path: str, resume_sb: int = 0) -> int:
    import jax

    spec = json.loads(Path(cfg_path).read_text())
    # D47U arch overrides: spec-driven head width / output-skip so a head
    # rebuild rides the standard driver (config keys arch_l2 / arch_out_skip).
    arch = dataclasses.replace(
        ArchSpec.full(), factorize=True, psqt=True,
        l2=int(spec.get("arch_l2", ArchSpec.full().l2)),
        out_skip=bool(spec.get("arch_out_skip", False)))
    start_sb = resume_sb + 1 if resume_sb else 1
    cfg = dataclasses.replace(
        TrainConfig(),
        lr=spec["lr"], lr_end_superbatch=spec["lr_end_superbatch"],
        superbatches=spec["superbatches"], pol_lambda=spec["pol_lambda"],
        ft_sparsity_lambda=spec["ft_sparsity_lambda"],
        scale=float(spec.get("scale", TrainConfig().scale)),
        target_offset=float(spec.get("target_offset", 0.0)),
        per_tensor_clip=bool(spec.get("per_tensor_clip", False)),
        no_bias_decay=bool(spec.get("no_bias_decay", False)),
        grad_clip_norm=float(spec.get("grad_clip_norm", 0.0)),
        pow_exp=float(spec.get("pow_exp", 2.0)),
        ft_sparsity_warmup_superbatches=spec.get(
            "ft_sparsity_warmup_superbatches", 0.0),
        warmup_superbatches=spec.get("warmup_superbatches", 0.0),
        ema_decay=spec.get("ema_decay", 0.0),
        start_superbatch=start_sb)
    out = ROOT / spec["out"]
    out.mkdir(parents=True, exist_ok=True)
    net_id = spec["net_id"]
    ema_on = cfg.ema_decay > 0.0

    if resume_sb:
        print(f"[pump {net_id}] RESUME from sb{resume_sb} (fresh Adam transient)",
              flush=True)
        params = ck.load_warm(out / f"{net_id}-sb{resume_sb}-raw.bin", arch)
        ema = (ck.load_warm(out / f"{net_id}-sb{resume_sb}-ema-raw.bin", arch)
               if ema_on else None)
    else:
        print(f"[pump {net_id}] warm load {spec['warm']}", flush=True)
        params = ck.load_warm(ROOT / spec["warm"], arch)
        expect = float(spec.get("heldout_expect", 0.0))
        if expect > 0.0:
            hv = heldout_value_loss(params, arch)
            ok = abs(hv - expect) <= HELDOUT_TOL
            print(f"[pump {net_id}] warm heldout {hv:.6f} "
                  f"(expect {expect}±{HELDOUT_TOL}) {'PASS' if ok else 'FAIL'}",
                  flush=True)
            if not ok:
                print(f"[pump {net_id}] ABORT: warm net mismatch", flush=True)
                return 2
        ema = dict(params) if ema_on else None

    mix = [(BulletBin(str(ROOT / p)), w) for p, w in spec["mix"]]
    # Loader parallelism. Trimmed 5/8 -> 4/4 during the 2026-07-25 OOM loop
    # because each in-flight MASKED batch held a dense (batch,1858) f32 mask.
    # That constraint is GONE: legal_mask="ids" ships (batch,128) i16 = 2.1 MB
    # instead of 61 MB (29x) and scatters the identical mask on device. So the
    # default goes back up; both are config-overridable.
    batches = make_mixed_batches(
        mix, cfg, policy=True,
        legal_mask=(spec.get("pol_mask") if isinstance(spec.get("pol_mask"), str)
                    else bool(spec.get("pol_mask", False))),
        seed=int(spec["seed"]),
        workers=int(spec.get("workers", 6)), prefetch=int(spec.get("prefetch", 8)),
        shuffle_chunk=int(spec.get("shuffle_chunk", 0)),
        total_batches=cfg.batches_per_superbatch * cfg.superbatches,
        skip_batches=cfg.batches_per_superbatch * resume_sb)

    # Freeze control: "train_only" (allowlist) or "freeze" (deny-list — the
    # complement against arch.param_shapes()); at most one may be present.
    train_only = spec.get("train_only")
    if spec.get("freeze"):
        assert train_only is None, "config: use train_only OR freeze, not both"
        train_only = sorted(set(arch.param_shapes()) - set(spec["freeze"]))
    opt, step = make_update_step(arch, cfg, train_only=train_only)
    opt_state = opt.init(params)
    params = {k: jax.numpy.asarray(v) for k, v in params.items()}
    if ema_on:
        ema = {k: jax.numpy.asarray(v) for k, v in ema.items()}

    bps = cfg.batches_per_superbatch
    running: dict = {}
    t0 = time.time()
    remaining = bps * (cfg.superbatches - resume_sb)
    for i, batch in enumerate(batches):
        if ema_on:
            params, ema, opt_state, metrics = step(params, ema, opt_state, batch)
        else:
            params, opt_state, metrics = step(params, opt_state, batch)
        for k, v in metrics.items():
            running[k] = running.get(k, 0.0) + float(v)
        bidx, sb = i % bps, i // bps + start_sb
        if bidx == bps - 1:
            c = {k: v / bps for k, v in running.items()}
            rate = ((i + 1) * cfg.batch_size) / (time.time() - t0)
            print(f"components sb {sb} | value {c.get('value', 0):.6f} | "
                  f"pol_ce {c.get('pol_ce', 0):.6f} | "
                  f"ft_l1 {c.get('ft_l1', 0):.6f} | "
                  f"ft_density {c.get('ft_density', 0):.6f} | "
                  f"{rate:,.0f} pos/s", flush=True)
            running = {}
            if sb % 10 == 0 or sb == cfg.superbatches:
                host = jax.tree.map(np.asarray, params)
                save_pair(host, arch, out, f"{net_id}-sb{sb}")
                hv = heldout_value_loss(host, arch)
                if ema_on:
                    ema_host = jax.tree.map(np.asarray, ema)
                    save_pair(ema_host, arch, out, f"{net_id}-sb{sb}-ema")
                    he = heldout_value_loss(ema_host, arch)
                    print(f"heldout sb {sb} | value {hv:.6f} | ema {he:.6f}",
                          flush=True)
                else:
                    print(f"heldout sb {sb} | value {hv:.6f}", flush=True)
        if i + 1 >= remaining:
            break
    print(f"[pump {net_id}] TRAIN EXIT 0", flush=True)
    return 0


if __name__ == "__main__":
    res = int(sys.argv[2]) if len(sys.argv) > 2 else 0
    raise SystemExit(main(sys.argv[1], res))
