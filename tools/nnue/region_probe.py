"""Tier-0 value probe for context REFINEMENT (richer-representation study).

The joint census (region_census.py) answers whether hot tuples CO-OCCUR with
both values of a candidate king-context bit. This probe answers the value
question one inference deeper, still without any training: with the incumbent
net FROZEN, does the loss gradient at a tuple's accumulator row POINT IN
DIFFERENT DIRECTIONS under the two values of the bit?

Mechanism. The accumulator is additive, so for one position the gradient of
the loss w.r.t. EVERY active feature row equals the gradient g w.r.t. the
accumulator itself (d acc / d W[row] = 1 for active rows). One vjp through
``head_forward`` therefore yields, per position and perspective, the shared
per-row gradient g in R^ACC. Scattering g by (tuple, bit-value) gives the
cell mean gradients mu_0, mu_1; if the incumbent's single shared row were
split by the bit, the first-order loss reduction is proportional to

    gain(t, b) = (n0*n1)/(n0+n1) * ||mu_0 - mu_1||^2   (debiased, see below)

summed over tuples. Under the null (the bit carries nothing) mu_0 - mu_1 is
pure sampling noise with E||mu_0-mu_1||^2 = tr Var(g|t) * (1/n0 + 1/n1); we
subtract exactly that (estimated from the per-cell second moments), and ALSO
run a content-free position-hash bit through the identical pipeline — the
empirical null every candidate must beat (NULL FIRST).

The same g serves BOTH channels (HalfKA rows and threat rows receive the
identical per-position gradient), so the probe prices threat-side and
halfka-side conditioning in one pass; the head/value path is asserted equal
to model.forward on the first batch, so the probe cannot drift from the
model.

CPU-only (JAX_PLATFORMS=cpu enforced below): the GPU is spoken for.
"""
from __future__ import annotations

import json
import os
import sys
import time
from pathlib import Path

os.environ.setdefault("JAX_PLATFORMS", "cpu")

import numpy as np

ROOT = Path("/home/nick/projects/chessengine")
WT = Path(__file__).resolve().parents[2]
sys.path.insert(0, str(WT / "jax"))

import jax                                                    # noqa: E402
import jax.numpy as jnp                                       # noqa: E402
from archon_jax import checkpoint as ck                       # noqa: E402
from archon_jax.data import BulletBin                         # noqa: E402
from archon_jax.features import DecodedBatch, halfka_batch, threat_batch  # noqa: E402
from archon_jax.model import (_ft_accumulate, _psqt_side_sums,  # noqa: E402
                              effective_pstw, forward, head_forward,
                              psqt_combine)
from archon_jax.spec import (ArchSpec, HALFKA_DIMS, THREAT_DIMS,  # noqa: E402
                             OUTPUT_BUCKETS_LAYOUT, TrainConfig)

# Reuse the census's region-bit definitions verbatim (single source of truth).
sys.path.insert(0, str(WT / "tools/nnue"))
from region_census import BITS, region_bits                   # noqa: E402

SPEC = ROOT / "data/nnue/experiments/recode-t1/recode_t1_identity.json"
CENSUS = ROOT / "data/nnue/nets/screens/recode/region-census-mix8M.npz"
OLD_CENSUS = ROOT / "data/nnue/nets/screens/recode/census-uho-band75-100-4M.npz"
OUT = ROOT / "data/nnue/nets/screens/recode/region-probe.npz"
K = 4096          # top-K tuples per channel carried at full gradient resolution
BATCH = 256
N_POS = int(sys.argv[1]) if len(sys.argv) > 1 else 512_000


def main() -> int:
    spec = json.loads(SPEC.read_text())
    arch = ArchSpec.from_run_spec(spec)
    cfg = TrainConfig()
    params = ck.load_warm(ROOT / spec["warm"], arch)
    jparams = {k: jnp.asarray(v) for k, v in params.items()}
    jparams["_pstw_eff"] = effective_pstw(jparams, arch)

    # top-K tuple selection: threat from the mix census; halfka from the
    # band75-100 census (marginal ranking only — noted in the study doc).
    th_counts = np.load(CENSUS)["joint"][0].sum(axis=1)
    hk_counts = np.load(OLD_CENSUS)["hk"]
    top = {"th": np.argsort(th_counts)[::-1][:K].copy(),
           "hk": np.argsort(hk_counts)[::-1][:K].copy()}
    local = {"th": np.full(THREAT_DIMS, -1, np.int64),
             "hk": np.full(HALFKA_DIMS, -1, np.int64)}
    for ch in ("th", "hk"):
        local[ch][top[ch]] = np.arange(K)

    nbits = len(BITS)
    # Accumulators: sum of g, count, and sum ||g||^2 per (channel, bit, tuple, r)
    G = {ch: np.zeros((nbits, K * 2, arch.hl), np.float32) for ch in ("th", "hk")}
    N = {ch: np.zeros((nbits, K * 2), np.int64) for ch in ("th", "hk")}
    S2 = {ch: np.zeros((nbits, K * 2), np.float64) for ch in ("th", "hk")}

    def loss_vec(acc_stm, acc_ntm, psqt_term, bucket, target):
        # psqt_term is the T6c skip (psqt_combine), passed as a CONSTANT: it
        # shifts q (and hence the residual/gradient) but its own gradient goes
        # to the 8-wide psqt tables, not the accumulator rows this probe prices.
        out = head_forward(jparams, arch, acc_stm, acc_ntm, bucket)[0] + psqt_term
        q = jax.nn.sigmoid(out)
        return (q - target) ** 2

    grad_fn = jax.jit(
        lambda a_s, a_n, ps, bucket, target: jax.vjp(
            lambda x, y: loss_vec(x, y, ps, bucket, target), a_s, a_n
        )[1](jnp.ones(target.shape, jnp.float32)))

    @jax.jit
    def segsum(g, pos_of_ev, cell_of_ev):
        return jax.ops.segment_sum(g[pos_of_ev], cell_of_ev,
                                   num_segments=K * 2)

    src = BulletBin(str(ROOT / "data/nnue/jaxcorpora/uho-band75-100.shuf.bin"))
    # sample AFTER the census window (census used [0, 2.01M) of this pack)
    base = 4_000_000
    checked = False
    t0 = time.time()
    done = 0
    while done < N_POS:
        recs = src.slice(base + done, min(BATCH, N_POS - done))
        dec = DecodedBatch(recs)
        hk = halfka_batch(dec)
        th = threat_batch(dec)
        bucket = OUTPUT_BUCKETS_LAYOUT[np.minimum(dec.piece_count, 32)]
        score = recs["score"].astype(np.float32)
        result = recs["result"].astype(np.float32) / 2.0
        s = score * cfg.score_multiplier
        eval_t = 1.0 / (1.0 + np.exp(-s / cfg.scale))
        target = (cfg.wdl * result + (1.0 - cfg.wdl) * eval_t).astype(np.float32)

        acc = [_ft_accumulate(jparams, jnp.asarray(hk[:, p]), jnp.asarray(th[:, p]))
               for p in (0, 1)]
        if arch.psqt:
            ps = psqt_combine(
                _psqt_side_sums(jparams, arch, jnp.asarray(hk[:, 0]),
                                jnp.asarray(th[:, 0])),
                _psqt_side_sums(jparams, arch, jnp.asarray(hk[:, 1]),
                                jnp.asarray(th[:, 1])),
                jnp.asarray(bucket))
        else:
            ps = jnp.zeros(len(recs), jnp.float32)
        if not checked:
            # drift guard: the probe's value path must equal model.forward
            batch = {"hk_stm": hk[:, 0], "hk_ntm": hk[:, 1],
                     "th_stm": th[:, 0], "th_ntm": th[:, 1],
                     "bucket": bucket.astype(np.int32), "target": target}
            ref = np.asarray(forward(jparams, arch, batch, "reference")[0])
            mine = np.asarray(head_forward(jparams, arch, acc[0], acc[1],
                                           jnp.asarray(bucket))[0] + ps)
            assert np.allclose(ref, mine, atol=1e-4), (
                f"probe forward drift: max|d|={np.abs(ref - mine).max()}")
            print(f"[probe] forward parity OK (max|d|="
                  f"{np.abs(ref - mine).max():.2e})", flush=True)
            checked = True

        gs = grad_fn(acc[0], acc[1], ps, jnp.asarray(bucket), jnp.asarray(target))
        g = [np.asarray(x) for x in gs]              # per-persp (B, hl)
        g2 = [np.sum(x.astype(np.float64) ** 2, axis=1) for x in g]

        for p in (0, 1):
            bits = region_bits(dec, p)               # (nbits, B)
            for ch, idx in (("th", th[:, p]), ("hk", hk[:, p])):
                pos_ev, slot = np.nonzero(idx >= 0)
                tl = local[ch][idx[pos_ev, slot]]
                keep = tl >= 0
                pos_ev, tl = pos_ev[keep], tl[keep]
                gj = jnp.asarray(g[p])
                for b in range(nbits):
                    cell = tl * 2 + bits[b][pos_ev]
                    G[ch][b] += np.asarray(segsum(gj, jnp.asarray(pos_ev),
                                                  jnp.asarray(cell)))
                    np.add.at(N[ch][b], cell, 1)
                    np.add.at(S2[ch][b], cell, g2[p][pos_ev])
        done += len(recs)
        if done % 25600 == 0:
            print(f"[probe] {done}/{N_POS} pos, {done / (time.time() - t0):,.0f} pos/s",
                  flush=True)

    np.savez(OUT, bits=np.array(BITS), K=K, n_pos=done,
             **{f"G_{c}": G[c] for c in G}, **{f"N_{c}": N[c] for c in N},
             **{f"S2_{c}": S2[c] for c in S2},
             **{f"top_{c}": top[c] for c in top})

    # ---- readout ----------------------------------------------------------
    print(f"\n[probe] {done} positions; per-channel per-bit first-order "
          f"split gain over top-{K} tuples (debiased; 'null' is the bar):")
    for ch in ("th", "hk"):
        for b, bname in enumerate(BITS):
            n = N[ch][b].reshape(K, 2).astype(np.float64)
            g_ = G[ch][b].reshape(K, 2, arch.hl).astype(np.float64)
            s2 = S2[ch][b].reshape(K, 2)
            n0, n1 = n[:, 0], n[:, 1]
            ok = (n0 >= 20) & (n1 >= 20)
            mu0 = g_[:, 0] / np.maximum(n0, 1)[:, None]
            mu1 = g_[:, 1] / np.maximum(n1, 1)[:, None]
            d2 = np.sum((mu0 - mu1) ** 2, axis=1)
            # per-cell variance of g around its own mean, traced
            var = ((s2[:, 0] + s2[:, 1])
                   - n0 * np.sum(mu0 ** 2, axis=1)
                   - n1 * np.sum(mu1 ** 2, axis=1)) / np.maximum(n0 + n1 - 2, 1)
            w = n0 * n1 / np.maximum(n0 + n1, 1)
            debiased = d2 - var * (1 / np.maximum(n0, 1) + 1 / np.maximum(n1, 1))
            gain = float(np.sum((w * debiased)[ok]))
            raw = float(np.sum((w * d2)[ok]))
            print(f"  {ch} {bname:6s} gain={gain:12.4e} (raw {raw:.4e}) "
                  f"tuples_ok={int(ok.sum())}", flush=True)
    print(f"[probe] saved {OUT}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
