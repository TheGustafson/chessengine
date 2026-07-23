"""Stage-3 driver (campaign phase 2): the first REAL JAX training run.

Recipe (pinned 2026-07-23, post user rulings — see archon-v1-release-campaign.md):
  warm  : d47c-stage2-sb240 raw (load_warm zero-fills the new tensors)
  arch  : d47c 1280 + T0 factorization + T6c PSQT skip (both zero-init no-ops)
  loss  : + FT-sparsity L1 (ft_sparsity_lambda from argv; probed pre-launch)
  data  : make_mixed_batches over unit-normalized *.shuf.bin corpora
          (UHO band75-100 0.40 / fishpack32 0.35 / farseerT74 band75-100 0.25
          — coverage-forward per B4, UHO as the stability anchor)
  LR    : peak 0.00025 (stage-2's restart halved again), cosine STRETCHED via
          lr_end_superbatch=331 so sb240 lands at ~0.2x peak — the
          anneal-to-FLOOR doctrine knob, NOT a to-zero ship anneal.
  pol   : lambda 0.00125 (stage-2 continuation)
  gates : pre-flight heldout reproduce (S1) hard-aborts a bad warm load;
          heldout eval at every save (the +/-4-Elo instrument, live).

Run (from jax/): see tools/nnue/runs/stage3_jax_train.sh
"""

from __future__ import annotations

import dataclasses
import sys
from pathlib import Path

import numpy as np

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))

from archon_jax import checkpoint as ck                      # noqa: E402
from archon_jax.data import BulletBin, make_mixed_batches    # noqa: E402
from archon_jax.features import map_batch                    # noqa: E402
from archon_jax.spec import ArchSpec, RECORD_DTYPE, TrainConfig  # noqa: E402
from archon_jax.train import make_update_step                # noqa: E402

ROOT = Path(__file__).resolve().parents[2]
WARM = ROOT / "data/nnue/experiments/d47c-stage2/ckpt/d47c-stage2-sb240-raw.bin"
HELDOUT = ROOT / "data/nnue/heldout/uho-band0-5.bin"
OUT = ROOT / "data/nnue/experiments/d47d-stage3"
MIX = [(ROOT / "data/nnue/jaxcorpora/uho-band75-100.shuf.bin", 0.40),
       (ROOT / "data/nnue/jaxcorpora/fishpack32.shuf.bin", 0.35),
       (ROOT / "data/nnue/jaxcorpora/farseerT74-band75-100.shuf.bin", 0.25)]
NET_ID = "d47d-stage3"
SEED = 0x00D47D03
# S1 pre-flight: full-heldout value loss of the warm net (trainer-graph
# instrument read 0.006094 at stage-2 sb240; envelope covers instrument noise).
HELDOUT_EXPECT = 0.006094
HELDOUT_TOL = 0.00012


def heldout_value_loss(params, arch, ft_impl="pallas", batch_size=8192):
    import jax
    from archon_jax.model import forward

    cfg = TrainConfig()
    recs = np.memmap(HELDOUT, dtype=RECORD_DTYPE, mode="r")
    n = (len(recs) // batch_size) * batch_size
    total, count = 0.0, 0
    jparams = {k: jax.numpy.asarray(v) for k, v in params.items()}
    for off in range(0, n, batch_size):
        b = map_batch(np.array(recs[off:off + batch_size]), wdl=cfg.wdl,
                      score_multiplier=cfg.score_multiplier, scale=cfg.scale,
                      policy=False)
        b = {k: np.asarray(v) for k, v in b.items()}
        out = forward(jparams, arch, b, ft_impl)[0]
        sig = jax.nn.sigmoid(out)
        total += float(np.sum((np.asarray(sig) - b["target"]) ** 2))
        count += batch_size
    return total / count


def main(sparsity_lambda: float, superbatches: int = 240,
         resume_sb: int = 0) -> int:
    import jax

    arch = dataclasses.replace(ArchSpec.full(), factorize=True, psqt=True)
    start_sb = resume_sb + 1 if resume_sb else 1
    cfg = dataclasses.replace(
        TrainConfig(), lr=0.00025, lr_end_superbatch=331,
        superbatches=superbatches, pol_lambda=0.00125,
        ft_sparsity_lambda=sparsity_lambda, start_superbatch=start_sb)
    OUT.mkdir(parents=True, exist_ok=True)

    if resume_sb:
        src = OUT / f"{NET_ID}-sb{resume_sb}-raw.bin"
        print(f"[stage3] RESUME from {src.name} (start sb {start_sb}; "
              f"fresh Adam = accepted ~2sb transient)", flush=True)
        params = ck.load_warm(src, arch)
    else:
        print(f"[stage3] warm load {WARM.name}", flush=True)
        params = ck.load_warm(WARM, arch)

        print("[stage3] S1 pre-flight: heldout reproduce on the warm net", flush=True)
        hv = heldout_value_loss(params, arch)
        ok = abs(hv - HELDOUT_EXPECT) <= HELDOUT_TOL
        print(f"[stage3] warm heldout {hv:.6f} (expect {HELDOUT_EXPECT}±{HELDOUT_TOL}) "
              f"{'PASS' if ok else 'FAIL'}", flush=True)
        if not ok:
            print("[stage3] ABORT: warm net does not reproduce heldout", flush=True)
            return 2

    mix = [(BulletBin(str(p)), w) for p, w in MIX]
    batches = make_mixed_batches(
        mix, cfg, policy=True, seed=SEED, workers=5,
        total_batches=cfg.batches_per_superbatch * superbatches,
        skip_batches=cfg.batches_per_superbatch * resume_sb)

    opt, step = make_update_step(arch, cfg)
    opt_state = opt.init(params)
    params = {k: jax.numpy.asarray(v) for k, v in params.items()}

    bps = cfg.batches_per_superbatch
    running: dict = {}
    import time
    t0 = time.time()
    remaining = bps * (cfg.superbatches - resume_sb)
    for i, batch in enumerate(batches):
        params, opt_state, metrics = step(params, opt_state, batch)
        for k, v in metrics.items():
            running[k] = running.get(k, 0.0) + float(v)
        bidx, sb = i % bps, i // bps + start_sb
        if bidx == bps - 1:
            comp = {k: v / bps for k, v in running.items()}
            rate = ((i + 1) * cfg.batch_size) / (time.time() - t0)
            print(f"components sb {sb} | value {comp.get('value', 0):.6f} | "
                  f"pol_ce {comp.get('pol_ce', 0):.6f} | "
                  f"ft_l1 {comp.get('ft_l1', 0):.6f} | "
                  f"ft_density {comp.get('ft_density', 0):.6f} | "
                  f"{rate:,.0f} pos/s", flush=True)
            running = {}
            if sb % 10 == 0 or sb == cfg.superbatches:
                host = jax.tree.map(np.asarray, params)
                ck.write_raw(OUT / f"{NET_ID}-sb{sb}-raw.bin", host)
                ck.write_quant(OUT / f"{NET_ID}-sb{sb}-quant.bin", host, arch)
                hv = heldout_value_loss(host, arch)
                print(f"heldout sb {sb} | value {hv:.6f}", flush=True)
        if i + 1 >= remaining:
            break
    print("[stage3] TRAIN EXIT 0", flush=True)
    return 0


if __name__ == "__main__":
    lam = float(sys.argv[1]) if len(sys.argv) > 1 else 0.001
    sbs = int(sys.argv[2]) if len(sys.argv) > 2 else 240
    res = int(sys.argv[3]) if len(sys.argv) > 3 else 0
    raise SystemExit(main(lam, sbs, res))
