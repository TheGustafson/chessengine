#!/usr/bin/env python3
"""Fit the REAL disagreement head (768->32->2) offline on cached trunk vectors.

WHY OFFLINE IS EXACT, NOT AN APPROXIMATION
    `model.py::head_forward` computes the head as

        dis_h    = crelu(stop_gradient(hl1) @ dis_projw + dis_projb)
        dis_pred = dis_h @ dis_outw + dis_outb

    The `stop_gradient` means the head NEVER influences the trunk, so training
    it inside the trainer and fitting it here on cached `hl1` optimise exactly
    the same objective over exactly the same parameters. The only thing the
    trainer adds is a data pipeline we do not need, because the target is not
    in the corpus records at all -- it comes from the engine.

    The output is therefore directly packable into a DIS1 section: same four
    tensors, same shapes, same order as `checkpoint.py::dis_section_bytes`.

WHAT IT PREDICTS
    col 0  signed  (search - static) / SCALE
    col 1  magnitude |search - static| / SCALE

    NOT the value head's own residual -- that quantity is structurally
    unpredictable and two runs (D57, D58) measured it at r^2 = 0.0017 on 1M
    heldout positions. Search-vs-static is a property of the tree below the
    position, which the net is not trained on, and a linear probe already
    extracts r^2 = 0.363 of it beyond a flexible function of the static eval.

THE BAR
    The linear ridge probe IS the null. An MLP that does not beat it has only
    demonstrated that it can memorise; that is precisely how D57 looked good
    in-sample and died out of sample. Held-out r for both columns is printed
    against the probe's numbers, and the split is SEQUENTIAL because the
    corpus is game-ordered -- a random split leaks positions from the same
    game across the boundary.
"""
import pathlib
import sys

import numpy as np

SCALE = 600.0          # network_scale unit; keeps the head's output O(1)


def main(hl1_npy: str, static_txt: str, search_txt: str, out_npz: str,
         hidden: int = 32, epochs: int = 60, lr: float = 3e-3,
         wd: float = 1e-4, seed: int = 0) -> int:
    import jax
    import jax.numpy as jnp
    import optax

    hl1 = np.load(hl1_npy)
    static = np.loadtxt(static_txt, dtype=np.float64)
    search = np.loadtxt(search_txt, dtype=np.float64)
    n = min(len(hl1), len(static), len(search))
    hl1, static, search = hl1[:n], static[:n], search[:n]

    # Mate rows are a different population and dominate MSE (six rows carried
    # >50% of the variance at n=60k). Excluded, and the count is printed so
    # the exclusion is never silent.
    keep = (np.abs(search) < 10000) & (np.abs(static) < 10000)
    print(f"n {n:,}  non-mate {keep.sum():,}  dropped {n - keep.sum():,}")
    hl1, static, search = hl1[keep], static[keep], search[keep]
    m = len(hl1)

    delta = (search - static) / SCALE
    y = np.stack([delta, np.abs(delta)], 1).astype(np.float32)
    X = hl1.astype(np.float32)

    cut = int(m * 0.8)                       # SEQUENTIAL: corpus is game-ordered
    Xtr, ytr, Xte, yte = X[:cut], y[:cut], X[cut:], y[cut:]
    print(f"train {cut:,}  test {m - cut:,}  hidden {hidden}  epochs {epochs}")

    key = jax.random.PRNGKey(seed)
    k1, k2 = jax.random.split(key)
    d = X.shape[1]
    params = {
        # He-ish init on a CReLU fan-in; the trainer's fresh-init path does the
        # same thing, so a comparison against a trainer-trained head is fair.
        "dis_projw": jax.random.normal(k1, (d, hidden)) * np.sqrt(2.0 / d),
        "dis_projb": jnp.zeros((hidden,)),
        "dis_outw": jax.random.normal(k2, (hidden, 2)) * np.sqrt(2.0 / hidden),
        "dis_outb": jnp.zeros((2,)),
    }

    def fwd(p, x):
        h = jnp.clip(x @ p["dis_projw"] + p["dis_projb"], 0.0, 1.0)  # crelu
        return h @ p["dis_outw"] + p["dis_outb"]

    def loss_fn(p, x, t):
        return jnp.mean((fwd(p, x) - t) ** 2)

    opt = optax.adamw(lr, weight_decay=wd)
    st = opt.init(params)

    @jax.jit
    def step(p, st, x, t):
        l, g = jax.value_and_grad(loss_fn)(p, x, t)
        upd, st = opt.update(g, st, p)
        return optax.apply_updates(p, upd), st, l

    BS = 4096
    rng = np.random.default_rng(seed)
    for ep in range(epochs):
        perm = rng.permutation(cut)
        tot = 0.0
        for off in range(0, cut - BS + 1, BS):
            idx = perm[off:off + BS]
            params, st, l = step(params, st, Xtr[idx], ytr[idx])
            tot += float(l)
        if ep % 10 == 9 or ep == 0:
            pr = np.asarray(fwd(params, jnp.asarray(Xte)))
            rs = np.corrcoef(pr[:, 0], yte[:, 0])[0, 1]
            rm = np.corrcoef(pr[:, 1], yte[:, 1])[0, 1]
            print(f"  ep {ep+1:3d}  train {tot/max(1,(cut//BS)):.5f}  "
                  f"test r signed {rs:+.4f}  magnitude {rm:+.4f}")

    pr = np.asarray(fwd(params, jnp.asarray(Xte)))
    rs = float(np.corrcoef(pr[:, 0], yte[:, 0])[0, 1])
    rm = float(np.corrcoef(pr[:, 1], yte[:, 1])[0, 1])
    print(f"\nFINAL held-out   signed r {rs:+.4f} (r^2 {rs*rs:.4f})   "
          f"magnitude r {rm:+.4f} (r^2 {rm*rm:.4f})")
    print("linear ridge probe (the bar): signed r +0.6745 raw / +0.6024 "
          "beyond a flexible static-eval basis; magnitude +0.4395 / +0.3526")

    np.savez(out_npz, **{k: np.asarray(v) for k, v in params.items()},
             scale=np.float32(SCALE))
    print(f"wrote {out_npz}")
    return 0


if __name__ == "__main__":
    if len(sys.argv) < 5:
        print(__doc__)
        raise SystemExit(2)
    raise SystemExit(main(*sys.argv[1:5]))
