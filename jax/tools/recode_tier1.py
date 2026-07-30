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

REFINEMENT arms (richer-representation study, 2026-07-31) — same driver,
three additional spec keys:
  "threat_code2d":  path.npy   (THREAT_DIMS, 2) int: (tuple, region-bit)->row
  "threat_region":  bit name   (archon_jax.features.REGION_BIT_NAMES)
  "region_census":  path.npz   the JOINT census (region_census.py) whose
                               `joint` array weights the projection init
  "bucket_regions": "b1[,b2]"  head-side arm: region-conditioned OUTPUT
                               buckets; needs arch_num_buckets = 8 * 2^n and
                               warm-start duplicates the head tensors.
The refinement projection init is CONSERVATIVE: both region cells of a tuple
start at the incumbent row, so an arm whose merged rows never fire on heldout
must reproduce the identity sb0 heldout EXACTLY — that reproduction is the
arm's own validity gate, printed at sb0. Its pre-registered decision rule
lives in docs/research/richer-representation-study-2026-07-31.md §6 and is
NOT restated loosely here to avoid two normative copies.

Usage (from jax/, GPU free — check the queue first):
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


def heldout_value_loss(params, arch, remap, batch_size=8192, region_kw=None):
    """Heldout MSE in the D48X frame (wdl .3, sigmoid s/400, offset 0) —
    remap-aware: a re-coded arch's indices go through the arm's own code
    (1-D remap, 2-D region-conditioned code, or region output buckets)."""
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
                      policy=False, threat_remap=remap, **(region_kw or {}))
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


def project_threat_code2d(params: dict, code2d: np.ndarray, rows: int,
                          joint_counts: np.ndarray) -> dict:
    """Refinement warm init: source cell (t, r) carries the incumbent row t,
    weighted by the JOINT census count of that cell; a target row's init is
    the weighted mean of the cells mapped onto it.

    For split rows (one (t, r) cell each) this reduces to W[t] exactly, and
    for untouched rows to the identity — so a code that only splits hot
    tuples and merges never-fired ones starts BIT-EQUAL to the incumbent on
    any position that fires no merged tuple.
    """
    out = dict(params)
    n_t = code2d.shape[0]
    w2 = joint_counts.astype(np.float64) + 1.0        # (T, 2)

    def _proj(t: np.ndarray) -> np.ndarray:
        num = np.zeros((rows, t.shape[1]), dtype=np.float64)
        den = np.zeros(rows, dtype=np.float64)
        for lo in range(0, n_t, 4096):                 # tuple chunks: no 1.2 GB f64 repeat
            hi = min(lo + 4096, n_t)
            cells = code2d[lo:hi].reshape(-1)          # (t, r) pairs, r-minor
            src = np.repeat(t[lo:hi].astype(np.float64), 2, axis=0)
            w = w2[lo:hi].reshape(-1)
            np.add.at(num, cells, src * w[:, None])
            np.add.at(den, cells, w)
        den[den == 0.0] = 1.0
        return (num / den[:, None]).astype(np.float32)

    out["l0_threatw"] = _proj(np.asarray(params["l0_threatw"]))
    assert out["l0_threatw"].shape[0] == rows
    if "l0_psqt_thw" in params:
        out["l0_psqt_thw"] = _proj(np.asarray(params["l0_psqt_thw"]))
    return out


def widen_head_buckets(params: dict, base: int, target: int) -> dict:
    """Head-side arm warm start: tile every per-bucket head tensor from
    `base` to `target` buckets by duplication. Duplicated buckets compute the
    identical function, so the widened net's sb0 heldout must equal the
    incumbent's — the arm's own validity gate."""
    assert target % base == 0 and target > base
    reps = target // base
    out = dict(params)
    for name in ("l1w", "l1b", "l2w", "l2b", "l3w", "l3b", "skipw"):
        if name not in params:
            continue
        t = np.asarray(params[name])
        cols = t.shape[-1]
        assert cols % base == 0, (name, t.shape)
        t2 = t.reshape(*t.shape[:-1], base, cols // base)
        t2 = np.tile(t2, (1,) * (t2.ndim - 2) + (reps, 1))
        out[name] = np.ascontiguousarray(
            t2.reshape(*t.shape[:-1], cols * reps))
    for name in ("l0_psqt_hkw", "l0_psqt_thw"):
        if name in params:
            out[name] = np.ascontiguousarray(
                np.tile(np.asarray(params[name]), (1, reps)))
    return out


def main(cfg_path: str) -> int:
    import jax

    spec = json.loads(Path(cfg_path).read_text())
    rows = int(spec.get("arch_threat_rows", 0))
    remap_path = spec.get("threat_remap", "")
    code2d_path = spec.get("threat_code2d", "")
    threat_region = spec.get("threat_region", "")
    bucket_regions = spec.get("bucket_regions", "")
    arch = ArchSpec.from_run_spec(spec)          # reads arch_threat_rows (+buckets)
    full_arch = dataclasses.replace(arch, threat_rows=0,
                                    num_buckets=ArchSpec.full().num_buckets)

    cfg = dataclasses.replace(
        TrainConfig(),
        lr=spec["lr"], lr_end_superbatch=spec["lr_end_superbatch"],
        superbatches=spec["superbatches"], pol_lambda=spec["pol_lambda"],
        ft_sparsity_lambda=spec.get("ft_sparsity_lambda", 0.0),
        # D47M lesson: a penalty ADDED to a converged warm start ramps, never
        # steps (updloc study ftsparse arms; train.py applies the ramp on
        # fresh runs, which a finetune leg is).
        ft_sparsity_warmup_superbatches=spec.get(
            "ft_sparsity_warmup_superbatches", 0.0),
        per_tensor_clip=bool(spec.get("per_tensor_clip", False)),
        no_bias_decay=bool(spec.get("no_bias_decay", False)),
        grad_clip_norm=float(spec.get("grad_clip_norm", 0.0)),
        pow_exp=float(spec.get("pow_exp", 2.0)),
        warmup_superbatches=spec.get("warmup_superbatches", 0.0),
        ema_decay=spec.get("ema_decay", 0.0),
        threat_remap=remap_path,
        threat_code2d=code2d_path,
        threat_region=threat_region,
        bucket_regions=bucket_regions,
        start_superbatch=1)
    out = ROOT / spec["out"]
    out.mkdir(parents=True, exist_ok=True)
    net_id = spec["net_id"]

    print(f"[t1 {net_id}] warm load {spec['warm']} (full arch)", flush=True)
    params = ck.load_warm(ROOT / spec["warm"], full_arch)

    region_kw: dict = {}
    if bucket_regions:
        names = tuple(b.strip() for b in bucket_regions.split(",") if b.strip())
        base = ArchSpec.full().num_buckets
        assert arch.num_buckets == base * (2 ** len(names)), \
            (arch.num_buckets, names)
        params = widen_head_buckets(params, base, arch.num_buckets)
        region_kw["bucket_regions"] = names
        print(f"[t1 {net_id}] head buckets {base} -> {arch.num_buckets} "
              f"(regions {','.join(names)}; duplicated => sb0 must equal "
              f"identity)", flush=True)

    remap_arr = None
    if remap_path:
        assert not code2d_path, "threat_remap and threat_code2d are exclusive"
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
    elif code2d_path:
        code2d = np.load(ROOT / code2d_path).astype(np.int64)
        assert code2d.shape == (THREAT_DIMS, 2) and 0 <= code2d.min() \
            and code2d.max() < (rows or THREAT_DIMS), "bad code2d"
        assert threat_region, "threat_code2d needs threat_region"
        jc = np.load(ROOT / spec["region_census"])
        bit_idx = list(jc["bits"]).index(threat_region.encode()
                                         if isinstance(jc["bits"][0], bytes)
                                         else threat_region)
        joint = jc["joint"][bit_idx] + jc["joint_held"][bit_idx]
        params = project_threat_code2d(params, code2d,
                                       rows or THREAT_DIMS, joint)
        print(f"[t1 {net_id}] refined threat code -> {rows or THREAT_DIMS} "
              f"rows on bit '{threat_region}' (conservative init)", flush=True)
        region_kw["threat_code2d"] = code2d
        region_kw["threat_region"] = threat_region
    else:
        assert rows in (0, THREAT_DIMS), "identity arm must keep full rows"

    hv0 = heldout_value_loss(params, arch, remap_arr, region_kw=region_kw)
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
            hv = heldout_value_loss(host, arch, remap_arr, region_kw=region_kw)
            line = f"heldout sb {sb} | value {hv:.6f}"
            if ema_on:
                ema_host = jax.tree.map(np.asarray, ema)
                ema_hv = heldout_value_loss(ema_host, arch, remap_arr,
                                            region_kw=region_kw)
                line += f" | ema {ema_hv:.6f}"
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
