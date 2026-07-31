"""Representation-gap study: frozen-accumulator extraction.

Computes, for a window of positions, everything the trained D48X network's
readout has access to — the pre-CReLU accumulators for both perspectives,
the output bucket, the PSQT skip term — plus the network's own value logit
and the training target, and writes them as npz shards for the probe stage.

VALIDITY GATE (asserted on the first 3 batches of every run): the value
logit recomposed from the stored pieces (head_forward(acc) + psqt_combine)
must equal model.forward's logit EXACTLY (same graph, same weights). A probe
suite built on accumulators that do not reproduce the model's own output
would be measuring a different network.

Sources:
  mix      uho-band75-100.shuf.bin  [5.0M, 5.0M+count)   (probe TRAIN split;
           window disjoint from every prior study's: census [0,2M),
           conj probe [2M,3M), confirm [3M,3.5M), region probe [4M,4.51M))
  heldout  uho-band0-5.bin          [0, count)           (probe EVAL split —
           the same file and target semantics as the Tier-1 panels, so
           value-probe deltas read in panel heldout-MSE units)
  recs     an .npz with a "recs" RECORD_DTYPE array (TSV-encoded positions
           from encode.py; state sidecar rides in the same npz untouched)

Run (CPU; the GPU is the D49 trainer's):
  JAX_PLATFORMS=cpu uv run --no-project --with numpy==2.5.1 --with jax==0.11.0 \
    python tools/nnue/reprgap/extract.py <mix|heldout|recs> [count|path] [start]
"""
from __future__ import annotations

import json
import os
import sys
import time
from pathlib import Path

os.environ.setdefault("JAX_PLATFORMS", "cpu")

import numpy as np

ROOT = Path(__file__).resolve().parents[3]
sys.path.insert(0, str(ROOT / "jax"))

import jax  # noqa: E402
import jax.numpy as jnp  # noqa: E402
from archon_jax import checkpoint as ck  # noqa: E402
from archon_jax.features import map_batch  # noqa: E402
from archon_jax.model import (_ft_accumulate, _psqt_side_sums,  # noqa: E402
                              effective_pstw, forward, head_forward,
                              psqt_combine)
from archon_jax.spec import RECORD_DTYPE, ArchSpec, TrainConfig  # noqa: E402

SPEC = ROOT / "data/nnue/experiments/recode-t1/recode_t1_identity.json"
OUT = ROOT / "data/nnue/nets/screens/reprgap"
SHARD = 65536
BATCH = 512  # (B,160,1280) f32 gather intermediates: 512 -> ~420MB peak per
             # channel; 4096 was a measured 9.7GB anon-RSS global OOM kill

SOURCES = {
    "mix": (ROOT / "data/nnue/jaxcorpora/uho-band75-100.shuf.bin", 5_000_000),
    "heldout": (ROOT / "data/nnue/heldout/uho-band0-5.bin", 0),
}


def main() -> int:
    src = sys.argv[1]
    print("[startup] imports done", flush=True)
    spec = json.loads(SPEC.read_text())
    arch = ArchSpec.from_run_spec(spec)
    cfg = TrainConfig()
    params = ck.load_warm(ROOT / spec["warm"], arch)
    print("[startup] checkpoint loaded", flush=True)
    jparams = {k: jnp.asarray(v) for k, v in params.items()}
    jparams["_pstw_eff"] = effective_pstw(jparams, arch)

    if src == "recs":
        path = Path(sys.argv[2])
        recs = np.load(path)["recs"].view(RECORD_DTYPE).reshape(-1)
        start, count, tag = 0, len(recs), path.stem
    else:
        path, base = SOURCES[src]
        count = int(sys.argv[2]) if len(sys.argv) > 2 else 350_000
        start = base + (int(sys.argv[3]) if len(sys.argv) > 3 else 0)
        recs = np.memmap(path, dtype=RECORD_DTYPE, mode="r")
        recs = recs[start:start + count]
        tag = f"{src}-{start}-{count}"

    @jax.jit
    def batch_fwd(b):
        acc_s = _ft_accumulate(jparams, b["hk_stm"], b["th_stm"])
        acc_n = _ft_accumulate(jparams, b["hk_ntm"], b["th_ntm"])
        ps_s = _psqt_side_sums(jparams, arch, b["hk_stm"], b["th_stm"])
        ps_n = _psqt_side_sums(jparams, arch, b["hk_ntm"], b["th_ntm"])
        psqt = psqt_combine(ps_s, ps_n, b["bucket"])
        out = head_forward(jparams, arch, acc_s, acc_n, b["bucket"])[0] + psqt
        return acc_s, acc_n, psqt, out

    OUT.mkdir(parents=True, exist_ok=True)
    n = len(recs)
    checked = 0
    t0 = time.time()
    for s0 in range(0, n, SHARD):
        cnt = min(SHARD, n - s0)
        accs = np.zeros((cnt, arch.hl), np.float16)
        accn = np.zeros((cnt, arch.hl), np.float16)
        psqt = np.zeros(cnt, np.float32)
        outv = np.zeros(cnt, np.float32)
        targ = np.zeros(cnt, np.float32)
        buck = np.zeros(cnt, np.uint8)
        for b0 in range(0, cnt, BATCH):
            rb = np.array(recs[s0 + b0:s0 + b0 + min(BATCH, cnt - b0)])
            b = map_batch(rb, wdl=cfg.wdl, score_multiplier=cfg.score_multiplier,
                          scale=cfg.scale, policy=False)
            jb = {k: jnp.asarray(v) for k, v in b.items() if k != "target"}
            a_s, a_n, ps, out = batch_fwd(jb)
            if checked < 3:
                ref = forward(jparams, arch, jb)[0]
                d = float(jnp.max(jnp.abs(out - ref)))
                # not bit-exact: forward() computes _pstw_eff inside its own
                # jit, ours is eager -> different f32 summation order (measured
                # 7.2e-7). 1e-5 on a logit whose sigmoid feeds a [0,1] target
                # is a strong witness of same-representation.
                assert d < 1e-5, f"PARITY FAIL: recomposed logit != forward ({d})"
                print(f"[parity] max|recomposed - forward| = {d:.2e}", flush=True)
                checked += 1
            sl = slice(b0, b0 + len(rb))
            accs[sl] = np.asarray(a_s, np.float16)
            accn[sl] = np.asarray(a_n, np.float16)
            psqt[sl] = np.asarray(ps)
            outv[sl] = np.asarray(out)
            targ[sl] = b["target"]
            buck[sl] = b["bucket"].astype(np.uint8)
        fn = OUT / f"acc-{tag}-{s0:08d}.npz"
        np.savez(fn, acc_stm=accs, acc_ntm=accn, psqt=psqt, out=outv,
                 target=targ, bucket=buck)
        done = s0 + cnt
        print(f"[{tag}] {done}/{n}  {done / (time.time() - t0):.0f} pos/s  -> {fn.name}",
              flush=True)
    print(f"[{tag}] DONE (parity gate passed on {checked} batches)")
    return 0


if __name__ == "__main__":
    sys.exit(main())
