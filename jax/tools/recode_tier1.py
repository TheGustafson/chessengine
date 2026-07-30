"""Tier-1 Recode panel driver: project-and-finetune (study §5.2).

Per arm: load the incumbent raw (full 60,720-row arch), PROJECT the threat
tensors through the arm's code (l0_threatw rows -> frequency-weighted class
means, or a supplied centroid file; l0_psqt_thw likewise), then finetune the
compressed arch for a few superbatches with the data loader re-coding
indices via cfg.threat_remap. Heldout (the +/-4-Elo trend instrument) is
scored EVERY superbatch — and once at sb0, which IS the T0b projection
readout (the identity arm's sb0 must reproduce the incumbent's heldout).

Run json = a pump_train-style spec plus:
  "arch_threat_rows": R'          (0 = identity)
  "threat_remap":     path.npy    ("" = identity)
  "recode_centroids": path.npy    ("" = project the warm rows instead)

Pre-registered decision rule (study §5.2 — do not change after readout):
  (1) random-8k ~ identity within run noise => instrument can't see codes,
      change instrument; (2) learned must beat random by >=3x run noise AND
      land within ~<=10 fixed-node Elo of identity; (3) 8k pass => 7.4x
      compressible; 8k fail + 16k pass => shallower but real; all fail =>
      report and move to the discovery arms.

Usage (from jax/, GPU free — check D49 first):
  uv run ... python tools/recode_tier1.py ../tools/nnue/runs/recode_t1_<arm>.json
"""

from __future__ import annotations

import dataclasses
import json
import os
import sys
import time
from pathlib import Path

import numpy as np

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))

from archon_jax import checkpoint as ck                      # noqa: E402
from archon_jax.data import BulletBin, make_mixed_batches    # noqa: E402
from archon_jax.features import map_batch                    # noqa: E402
from archon_jax.spec import (ArchSpec, RECORD_DTYPE, THREAT_DIMS,  # noqa: E402
                             TrainConfig)
from archon_jax.train import make_update_step                # noqa: E402

ROOT = Path(__file__).resolve().parents[2]
HELDOUT = ROOT / "data/nnue/heldout/uho-band0-5.bin"


def heldout_value_loss(params, arch, remap, batch_size=8192):
    """Heldout MSE in the D48X frame (wdl .3, sigmoid s/400, offset 0) —
    remap-aware: a re-coded arch's indices go through the arm's own code."""
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
                      policy=False, threat_remap=remap)
        b = {k: np.asarray(v) for k, v in b.items()}
        out = forward(jparams, arch, b,
                      os.environ.get("ARCHON_JAX_FT_IMPL", "reference"))[0]
        q = np.asarray(jax.nn.sigmoid(out))
        total += float(np.sum((q - b["target"]) ** 2))
    return total / n


def project_threat(params: dict, remap: np.ndarray, rows: int,
                   counts: np.ndarray, centroids: np.ndarray | None) -> dict:
    """Tied warm init: threat tensors -> the compressed row space."""
    out = dict(params)
    w = counts.astype(np.float64) + 1.0

    def _proj(t: np.ndarray) -> np.ndarray:
        num = np.zeros((rows, t.shape[1]), dtype=np.float64)
        den = np.zeros(rows, dtype=np.float64)
        np.add.at(num, remap, t.astype(np.float64) * w[:, None])
        np.add.at(den, remap, w)
        den[den == 0.0] = 1.0
        return (num / den[:, None]).astype(np.float32)

    out["l0_threatw"] = (centroids if centroids is not None
                         else _proj(np.asarray(params["l0_threatw"])))
    assert out["l0_threatw"].shape[0] == rows
    if "l0_psqt_thw" in params:
        out["l0_psqt_thw"] = _proj(np.asarray(params["l0_psqt_thw"]))
    return out


def main(cfg_path: str) -> int:
    import jax

    spec = json.loads(Path(cfg_path).read_text())
    rows = int(spec.get("arch_threat_rows", 0))
    remap_path = spec.get("threat_remap", "")
    arch = ArchSpec.from_run_spec(spec)          # reads arch_threat_rows
    full_arch = dataclasses.replace(arch, threat_rows=0)

    cfg = dataclasses.replace(
        TrainConfig(),
        lr=spec["lr"], lr_end_superbatch=spec["lr_end_superbatch"],
        superbatches=spec["superbatches"], pol_lambda=spec["pol_lambda"],
        ft_sparsity_lambda=spec.get("ft_sparsity_lambda", 0.0),
        per_tensor_clip=bool(spec.get("per_tensor_clip", False)),
        no_bias_decay=bool(spec.get("no_bias_decay", False)),
        grad_clip_norm=float(spec.get("grad_clip_norm", 0.0)),
        pow_exp=float(spec.get("pow_exp", 2.0)),
        warmup_superbatches=spec.get("warmup_superbatches", 0.0),
        ema_decay=spec.get("ema_decay", 0.0),
        threat_remap=remap_path,
        start_superbatch=1)
    out = ROOT / spec["out"]
    out.mkdir(parents=True, exist_ok=True)
    net_id = spec["net_id"]

    print(f"[t1 {net_id}] warm load {spec['warm']} (full arch)", flush=True)
    params = ck.load_warm(ROOT / spec["warm"], full_arch)

    if remap_path:
        remap = np.load(ROOT / remap_path).astype(np.int64)
        assert remap.shape == (THREAT_DIMS,) and 0 <= remap.min() \
            and remap.max() < rows, "bad remap"
        counts = np.load(ROOT / spec["census"])["th"]
        cent = (np.load(ROOT / spec["recode_centroids"]).astype(np.float32)
                if spec.get("recode_centroids") else None)
        params = project_threat(params, remap, rows, counts, cent)
        print(f"[t1 {net_id}] projected threat tensors -> {rows} rows "
              f"(centroids={'yes' if cent is not None else 'projected'})",
              flush=True)
        remap_arr = remap
    else:
        remap_arr = None
        assert rows in (0, THREAT_DIMS), "identity arm must keep full rows"

    hv0 = heldout_value_loss(params, arch, remap_arr)
    print(f"heldout sb 0 | value {hv0:.6f}   <- T0b projection readout",
          flush=True)

    mix = [(BulletBin(str(ROOT / p)), w) for p, w in spec["mix"]]
    batches = make_mixed_batches(
        mix, cfg, policy=False, seed=int(spec["seed"]),
        workers=int(spec.get("workers", 4)),
        prefetch=int(spec.get("prefetch", 4)),
        shuffle_chunk=int(spec.get("shuffle_chunk", 128)),
        total_batches=cfg.batches_per_superbatch * cfg.superbatches)

    opt, step = make_update_step(arch, cfg, train_only=None, teacher=None)
    opt_state = opt.init(params)
    params = {k: jax.numpy.asarray(v) for k, v in params.items()}
    ema_on = cfg.ema_decay > 0.0
    ema = {k: jax.numpy.asarray(np.asarray(v).copy()) for k, v in params.items()} \
        if ema_on else None

    bps = cfg.batches_per_superbatch
    running: dict = {}
    t0 = time.time()
    for i, batch in enumerate(batches):
        if ema_on:
            params, ema, opt_state, metrics = step(params, ema, opt_state, batch)
        else:
            params, opt_state, metrics = step(params, opt_state, batch)
        for k, v in metrics.items():
            running[k] = running.get(k, 0.0) + float(v)
        bidx, sb = i % bps, i // bps + 1
        if bidx == bps - 1:
            c = {k: v / bps for k, v in running.items()}
            rate = ((i + 1) * cfg.batch_size) / (time.time() - t0)
            print(f"components sb {sb} | value {c.get('value', 0):.6f} | "
                  f"{rate:,.0f} pos/s", flush=True)
            running = {}
            host = jax.tree.map(np.asarray, params)
            hv = heldout_value_loss(host, arch, remap_arr)
            line = f"heldout sb {sb} | value {hv:.6f}"
            if ema_on:
                ema_host = jax.tree.map(np.asarray, ema)
                line += f" | ema {heldout_value_loss(ema_host, arch, remap_arr):.6f}"
            print(line, flush=True)
            if sb == cfg.superbatches:
                ck.write_raw(out / f"{net_id}-sb{sb}-raw.bin", host)
                if ema_on:
                    ck.write_raw(out / f"{net_id}-sb{sb}-ema-raw.bin", ema_host)
        if i + 1 >= bps * cfg.superbatches:
            break
    print(f"[t1 {net_id}] EXIT 0", flush=True)
    return 0


if __name__ == "__main__":
    raise SystemExit(main(sys.argv[1]))
