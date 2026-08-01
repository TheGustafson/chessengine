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

from archon_jax import checkpoint as ck
from archon_jax import model                      # noqa: E402
from archon_jax.data import BulletBin, make_mixed_batches    # noqa: E402
from archon_jax.features import map_batch                    # noqa: E402
from archon_jax.spec import ArchSpec, RECORD_DTYPE, TrainConfig  # noqa: E402
from archon_jax.train import make_update_step                # noqa: E402

ROOT = Path(__file__).resolve().parents[2]
HELDOUT = ROOT / "data/nnue/heldout/uho-band0-5.bin"
HELDOUT_TOL = 0.00012
# (offset, scale) for the HELDOUT corpus's own fitted transform. Set from the
# run spec so the heldout is scored in the same frame the run trains in.
HELDOUT_TARGET = (0.0, None)


def heldout_value_loss(params, arch, ft_impl="pallas", batch_size=8192,
                       target_offset=0.0, scale=None,
                       net_offset=0.0, net_scale=0.0):
    """Heldout value MSE.

    **The frame matters and this is a trap.** This metric was pinned at
    `wdl 0.3 / scale 400 / offset 0`, which is the UNCALIBRATED target. A net
    trained on the calibrated draw-aware target predicts a different (correct)
    quantity, so scoring it against the old target reads as a REGRESSION BY
    CONSTRUCTION. Pass the run's own transform; cross-frame comparison of this
    number between runs is meaningless.
    """
    import jax
    from archon_jax.model import forward

    cfg = TrainConfig()
    if scale is None:
        scale = cfg.scale
    recs = np.memmap(HELDOUT, dtype=RECORD_DTYPE, mode="r")
    n = (len(recs) // batch_size) * batch_size
    total = 0.0
    jparams = {k: jax.numpy.asarray(v) for k, v in params.items()}
    for off in range(0, n, batch_size):
        b = map_batch(np.array(recs[off:off + batch_size]), wdl=cfg.wdl,
                      score_multiplier=cfg.score_multiplier, scale=scale,
                      policy=False, target_offset=target_offset)
        b = {k: np.asarray(v) for k, v in b.items()}
        out = forward(jparams, arch, b, ft_impl)[0]
        # THE SAME TRAP AS THE LABEL FRAME, ON THE NETWORK SIDE. D49 trains
        # with the offset two-sigmoid on the network output; scoring it with a
        # plain sigmoid measures a DIFFERENT FUNCTION than training optimises
        # and reads as a regression by construction.
        if net_scale > 0.0:
            from archon_jax.losses import expected_score
            q = np.asarray(expected_score(out, net_offset, net_scale))
        else:
            q = np.asarray(jax.nn.sigmoid(out))
        total += float(np.sum((q - b["target"]) ** 2))
    return total / n


def save_pair(host, arch, out: Path, name: str) -> None:
    ck.write_raw(out / f"{name}-raw.bin", host)
    ck.write_quant(out / f"{name}-quant.bin", host, arch)


def build_cfg(spec: dict, start_sb: int) -> TrainConfig:
    """The TrainConfig a run json describes — ONE definition, for the same
    reason ArchSpec.from_run_spec exists. Every field is listed EXPLICITLY
    (a spec key with no line here is silently dropped, which is how D48Y2
    trained the old target while its config claimed the new one), so any tool
    that wants to reason about a run — the alpha=0 null, in particular — must
    be able to get the identical object rather than re-deriving it."""
    return dataclasses.replace(
        TrainConfig(),
        distill_alpha=float(spec.get("distill_alpha", 0.0)),
        dis_lambda=float(spec.get("dis_lambda", 0.0)),
        distill_alpha_end=float(spec.get("distill_alpha_end", -1.0)),
        distill_alpha_ramp=tuple(
            (float(a), float(b)) for a, b in spec.get("distill_alpha_ramp", [])),
        lr_ramp=tuple(
            (float(a), float(b)) for a, b in spec.get("lr_ramp", [])),
        export_every=int(spec.get("export_every", 10)),
        lr=spec["lr"], lr_end_superbatch=spec["lr_end_superbatch"],
        superbatches=spec["superbatches"], pol_lambda=spec["pol_lambda"],
        ft_sparsity_lambda=spec["ft_sparsity_lambda"],
        scale=float(spec.get("scale", TrainConfig().scale)),
        target_offset=float(spec.get("target_offset", 0.0)),
        # D48Y: [[offset, scale], ...] aligned with the mix. Without this line
        # the field is silently dropped and the run trains the OLD target while
        # the config claims the new one.
        per_corpus_target=(tuple((float(a), float(b))
                                 for a, b in spec["per_corpus_target"])
                           if spec.get("per_corpus_target") else None),
        net_offset=float(spec.get("net_offset", 0.0)),
        net_scale=float(spec.get("net_scale", 0.0)),
        qp_asymmetry=float(spec.get("qp_asymmetry", 0.0)),
        piece_rebalance=bool(spec.get("piece_rebalance", False)),
        wdl_filter=bool(spec.get("wdl_filter", False)),
        wdl_filter_floor=float(spec.get("wdl_filter_floor", 0.0)),
        qat_ft_scale=float(spec.get("qat_ft_scale", 0.0)),
        qat_threat_scale=float(spec.get("qat_threat_scale", 0.0)),
        qat_act_levels=float(spec.get("qat_act_levels", 0.0)),
        qat_l1_scale=float(spec.get("qat_l1_scale", 0.0)),
        teacher_frame_offset=float(spec.get("teacher_frame_offset", 0.0)),
        teacher_frame_scale=float(spec.get("teacher_frame_scale", 0.0)),
        # SF-net teacher: [[k, p], ...] aligned with the mix, same positional
        # contract as per_corpus_target. Dropping this line silently trains
        # the WRONG teacher frame while the config claims the right one —
        # exactly the class of bug that sank D48Y2.
        teacher_kp=(tuple((float(k), float(p))
                          for k, p in spec["teacher_kp"])
                    if spec.get("teacher_kp") else None),
        per_tensor_clip=bool(spec.get("per_tensor_clip", False)),
        no_bias_decay=bool(spec.get("no_bias_decay", False)),
        grad_clip_norm=float(spec.get("grad_clip_norm", 0.0)),
        pow_exp=float(spec.get("pow_exp", 2.0)),
        ft_sparsity_warmup_superbatches=spec.get(
            "ft_sparsity_warmup_superbatches", 0.0),
        warmup_superbatches=spec.get("warmup_superbatches", 0.0),
        ema_decay=spec.get("ema_decay", 0.0),
        start_superbatch=start_sb)


def main(cfg_path: str, resume_sb: int = 0) -> int:
    import jax

    spec = json.loads(Path(cfg_path).read_text())
    # D47U arch overrides: spec-driven head width / output-skip so a head
    # rebuild rides the standard driver (config keys arch_l2 / arch_out_skip).
    # The mapping lives in ArchSpec so re-export tools cannot derive a
    # different arch from the same run json (perf study 30).
    arch = ArchSpec.from_run_spec(spec)
    start_sb = resume_sb + 1 if resume_sb else 1
    cfg = build_cfg(spec, start_sb)
    global HELDOUT_TARGET
    if spec.get("heldout_target"):
        HELDOUT_TARGET = (float(spec["heldout_target"][0]),
                          float(spec["heldout_target"][1]))
        print(f"[pump] heldout scored in the RUN frame "
              f"a={HELDOUT_TARGET[0]} b={HELDOUT_TARGET[1]}", flush=True)
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
    elif not spec.get("warm"):
        # FROM SCRATCH (D49). Every recent run was warm-started, so this branch
        # did not exist and `spec["warm"]` raised KeyError. Bullet-parity init:
        # affine ~ Normal(0, sqrt(2/fan_in)) with l0_pstw using the EFFECTIVE
        # input size (32 active features, not 22528), biases and the threat FT
        # zeroed. See model.init_params.
        seed = int(spec.get("init_seed", 0))
        print(f"[pump {net_id}] FROM SCRATCH (init_params seed={seed})", flush=True)
        # np.asarray so params and the EMA shadow get SEPARATE device buffers.
        # init_params returns jax arrays, and `dict(params)` would share them —
        # the step fn donates both (donate_argnums=(0,1)) and JAX rejects
        # donating one buffer twice. The warm path is safe only because
        # load_warm returns numpy.
        params = {k: np.asarray(v) for k, v in model.init_params(arch, seed).items()}
        ema = {k: v.copy() for k, v in params.items()} if ema_on else None
    else:
        print(f"[pump {net_id}] warm load {spec['warm']}", flush=True)
        params = ck.load_warm(ROOT / spec["warm"], arch)
        expect = float(spec.get("heldout_expect", 0.0))
        if expect > 0.0:
            hv = heldout_value_loss(params, arch,
                                    target_offset=HELDOUT_TARGET[0],
                                    scale=HELDOUT_TARGET[1])
            ok = abs(hv - expect) <= HELDOUT_TOL
            print(f"[pump {net_id}] warm heldout {hv:.6f} "
                  f"(expect {expect}±{HELDOUT_TOL}) {'PASS' if ok else 'FAIL'}",
                  flush=True)
            if not ok:
                print(f"[pump {net_id}] ABORT: warm net mismatch", flush=True)
                return 2
        ema = dict(params) if ema_on else None

    teacher = None
    if spec.get("teacher_warm"):
        # `teacher_arch` lets the teacher be a DIFFERENT architecture from the
        # student — which is the whole point once the teacher is Stockfish's
        # own net (1024/32/32 with their dual tail) rather than a previous
        # ARCHON leg. Absent => the historical full() teacher, unchanged.
        # teacher_arch is merged LAST and wins: SF's checkpoint has no
        # factoriser tensor, so `factorize` must be able to go back to False.
        t_arch = dataclasses.replace(
            ArchSpec.full(),
            **{"factorize": True, "psqt": True, **spec.get("teacher_arch", {})})
        t_params = ck.load_warm(ROOT / spec["teacher_warm"], t_arch)
        teacher = (t_params, t_arch)
        print(f"[pump {net_id}] teacher loaded: {spec['teacher_warm']} "
              f"(alpha {cfg.distill_alpha}, hl={t_arch.hl} l2={t_arch.l2} "
              f"l3={t_arch.l3} dual_tail={t_arch.dual_tail} "
              f"sf_act={t_arch.sf_act})", flush=True)
        if cfg.teacher_kp is not None and len(cfg.teacher_kp) != len(spec["mix"]):
            print(f"[pump {net_id}] ABORT: teacher_kp has "
                  f"{len(cfg.teacher_kp)} entries for {len(spec['mix'])} "
                  f"corpora", flush=True)
            return 2

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
    opt, step = make_update_step(arch, cfg, train_only=train_only, teacher=teacher)
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
            # Export cadence. Default 10, overridable per-run with
            # "export_every". The LR jump->anneal cycles set it to the cycle
            # length so checkpoints land ONLY at the BOTTOM of the cycle
            # (LR -> 0): measured 2026-08-01, a mid-cycle checkpoint taken just
            # after a 5x LR jump was 13 Elo weaker than the anneal-end one
            # (sb1580 -3.94 vs sb1570 +9.27, 1500 games each).
            _every = int(getattr(cfg, 'export_every', 10) or 10)
            if sb % _every == 0 or sb == cfg.superbatches:
                host = jax.tree.map(np.asarray, params)
                save_pair(host, arch, out, f"{net_id}-sb{sb}")
                hv = heldout_value_loss(host, arch,
                                        target_offset=HELDOUT_TARGET[0],
                                        scale=HELDOUT_TARGET[1],
                                        net_offset=cfg.net_offset,
                                        net_scale=cfg.net_scale)
                if ema_on:
                    ema_host = jax.tree.map(np.asarray, ema)
                    save_pair(ema_host, arch, out, f"{net_id}-sb{sb}-ema")
                    he = heldout_value_loss(ema_host, arch,
                                            target_offset=HELDOUT_TARGET[0],
                                            scale=HELDOUT_TARGET[1],
                                            net_offset=cfg.net_offset,
                                            net_scale=cfg.net_scale)
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
