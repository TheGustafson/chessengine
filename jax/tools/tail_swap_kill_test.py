"""R2 frozen-trunk tail-swap KILL TEST (nnue-architecture-study-2026-07-28 §7 R2).

Question: does SF `eca43a97`'s DUAL-ACTIVATION tail (§4.4) beat our `sf_18`-era
single-activation tail, on a FIXED trunk? If it cannot win with the trunk held
still, it will not win co-trained — and the 50-60 GPU-hour full run is dead.

Protocol (pre-registered before any number was looked at):
  * trunk  = D48X's feature transformer + PSQT/factoriser tables, from
             the canonical net's own raw checkpoint. FROZEN via train.train_only
             (D47L: grads zeroed AND post-update values restored, so AdamW's
             grad-independent decay is a no-op on frozen params).
  * head   = l1w/l1b/l2w/l2b/l3w/l3b/skipw, FRESH-INIT from `--head-seed`,
             retrained from scratch. Both arms start fresh because arm B's
             tail tensors have different shapes and CANNOT be warm-started;
             warm-starting only arm A would hand it a trained optimum.
  * arm A  = --arm A = today's tail (dual_tail=False), bit-identical code path.
  * arm B  = --arm B = dual_tail=True.
  * metric = heldout value MSE, mean over data/nnue/heldout/uho-band0-5.bin
             of (sigmoid(out) - target)^2 — the same statistic every campaign
             leg reports (pump_train.heldout_value_loss).
  * BAR    = kill unless (A - B) / A >= 0.01, i.e. arm B improves >= 1.0%
             relative.

NULL PANEL FIRST (CLAUDE.md rule 1): run arm A twice with different
--head-seed. If the seed-to-seed spread is comparable to 1.0%, the instrument
cannot resolve the bar and no verdict may be read off it.

This writes NO checkpoints by default (a raw pair is 570 MB) — the deliverable
is the heldout number. --save-head dumps just the head tensors (~700 KB).

Run (from jax/):
  ARCHON_JAX_ALLOW_GPU=1 ARCHON_JAX_FT_IMPL=pallas uv run --no-project \\
    --with numpy==2.5.1 --with "jax[cuda12]==0.11.0" --with optax==0.2.8 \\
    python tools/tail_swap_kill_test.py --arm A --head-seed 1 --batches 20000
"""

from __future__ import annotations

import argparse
import dataclasses
import json
import sys
import time
from pathlib import Path

import numpy as np

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))

from archon_jax import checkpoint as ck                       # noqa: E402
from archon_jax.data import BulletBin, make_mixed_batches     # noqa: E402
from archon_jax.features import map_batch                     # noqa: E402
from archon_jax.spec import ArchSpec, RECORD_DTYPE, TrainConfig  # noqa: E402
from archon_jax.train import make_update_step                 # noqa: E402

ROOT = Path(__file__).resolve().parents[2]
HELDOUT = ROOT / "data/nnue/heldout/uho-band0-5.bin"
# The canonical net's own raw checkpoint (engines/canonical/net.cenn ->
# d48x-1280-sb520-ema.cenn; the .cenn is the deploy pack, this is the f32
# source it was packed from). READ-ONLY here.
TRUNK = ROOT / "data/nnue/experiments/d48x-1280/d48x-1280-sb520-ema-raw.bin"
D48X_CFG = ROOT / "tools/nnue/runs/d48x_1280.json"

# Everything after the accumulator. The complement is the frozen trunk.
HEAD_TENSORS = ("l1w", "l1b", "l2w", "l2b", "l3w", "l3b", "skipw")


def d48x_arch(dual: bool) -> ArchSpec:
    """D48X's architecture, minus the two dead heads.

    pol_lambda and dis_lambda are 0 in D48X's final recipe (its _policy_ruling:
    "POLICY DITCHED"), so pol_*/dis_* carry exactly zero gradient and cannot
    reach the value output — dropping them from the arch is value-path
    equivalent and saves the loader the legal-move mask entirely.
    """
    return dataclasses.replace(
        ArchSpec.full(), hl=1280, l2=16, l3=32,
        factorize=True, psqt=True, out_skip=True,
        policy=False, pol_proj=0, dis_proj=0,
        dual_tail=dual)


def build_params(arch: ArchSpec, head_seed: int) -> dict:
    """Frozen D48X trunk + a fresh-init head."""
    from archon_jax.model import init_params

    shapes = arch.param_shapes()
    fresh = init_params(arch, head_seed)
    flat = ck.open_raw(TRUNK, shapes=None, mmap=True)
    out, from_file, from_init = {}, [], []
    for name, want in shapes.items():
        if name in HEAD_TENSORS:
            out[name] = np.asarray(fresh[name])
            from_init.append(name)
            continue
        if name not in flat:
            raise KeyError(f"trunk checkpoint lacks {name}")
        arr = flat[name]
        if int(np.prod(want)) != arr.size:
            raise ValueError(f"{name}: file count {arr.size} != shape {want}")
        out[name] = np.asarray(arr).reshape(want)
        from_file.append(name)
    print(f"[trunk] {TRUNK.name}: {len(from_file)} frozen tensors "
          f"({sum(int(np.prod(shapes[n])) for n in from_file):,} params) | "
          f"head fresh-init seed {head_seed}: {sorted(from_init)} "
          f"({sum(int(np.prod(shapes[n])) for n in from_init):,} params)",
          flush=True)
    return out


def _fingerprint(x) -> str:
    """sha256 of a tensor's raw bytes, streamed in 16 MB chunks.

    Bit-exact and memory-safe: the naive `sum(abs(x.astype(f64)))` doubles a
    311 MB tensor into 622 MB, which this 15 GB box cannot spare.
    """
    import hashlib

    a = np.asarray(x)
    flat = np.ascontiguousarray(a).reshape(-1)
    h = hashlib.sha256()
    step = 4 << 20  # 4M float32 = 16 MB
    for i in range(0, flat.size, step):
        h.update(flat[i:i + step].tobytes())
    return h.hexdigest()


def heldout_value_loss(params, arch, ft_impl="pallas", batch_size=8192) -> float:
    """Mean (sigmoid(value) - target)^2 over the heldout corpus.

    Byte-for-byte the statistic pump_train.heldout_value_loss reports, so the
    numbers here are directly comparable to every campaign leg.
    """
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


def main(argv=None) -> int:
    ap = argparse.ArgumentParser()
    ap.add_argument("--arm", choices=("A", "B"), required=True,
                    help="A = current tail (control), B = dual-activation tail")
    ap.add_argument("--head-seed", type=int, default=1)
    ap.add_argument("--batches", type=int, default=20000)
    ap.add_argument("--bps", type=int, default=1000,
                    help="batches per superbatch (report/heldout cadence)")
    ap.add_argument("--lr", type=float, default=1e-3)
    ap.add_argument("--data-seed", type=int, default=220480001,
                    help="D48X's own mix seed; IDENTICAL across arms")
    # D48X's own loader settings (tools/nnue/runs/d48x_1280.json)
    ap.add_argument("--workers", type=int, default=4)
    ap.add_argument("--prefetch", type=int, default=4)
    ap.add_argument("--shuffle-chunk", type=int, default=128)
    ap.add_argument("--tag", default="")
    ap.add_argument("--save-head", default="")
    ap.add_argument("--reference-only", action="store_true",
                    help="just print D48X's own heldout (trunk + ITS head) and exit")
    a = ap.parse_args(argv)

    import jax

    arch = d48x_arch(dual=(a.arm == "B"))
    tag = a.tag or f"arm{a.arm}-s{a.head_seed}"

    if a.reference_only:
        ref_arch = dataclasses.replace(d48x_arch(False), policy=True, pol_proj=16,
                                       dis_proj=64)
        ref = ck.load_warm(TRUNK, ref_arch)
        print(json.dumps({"tag": "d48x-own-head",
                          "heldout": heldout_value_loss(ref, ref_arch)}),
              flush=True)
        return 0

    nsb = a.batches // a.bps
    cfg = dataclasses.replace(
        TrainConfig(),
        batches_per_superbatch=a.bps, superbatches=nsb,
        lr=a.lr, lr_end_superbatch=nsb,
        warmup_superbatches=0.25,
        pol_lambda=0.0, ft_sparsity_lambda=0.0, dis_lambda=0.0,
        # D48X's recipe knobs, verbatim (tools/nnue/runs/d48x_1280.json)
        pow_exp=2.5, per_tensor_clip=True, no_bias_decay=True,
        grad_clip_norm=2.0)

    spec = json.loads(D48X_CFG.read_text())
    mix = [(BulletBin(str(ROOT / p)), w) for p, w in spec["mix"]]
    batches = make_mixed_batches(
        mix, cfg, policy=False, seed=a.data_seed,
        workers=a.workers, prefetch=a.prefetch,
        shuffle_chunk=a.shuffle_chunk, total_batches=a.batches)

    params = build_params(arch, a.head_seed)
    frozen = sorted(set(arch.param_shapes()) - set(HEAD_TENSORS))
    fp0 = {n: _fingerprint(params[n]) for n in frozen}

    opt, step = make_update_step(arch, cfg, train_only=list(HEAD_TENSORS))
    opt_state = opt.init(params)
    params = {k: jax.numpy.asarray(v) for k, v in params.items()}

    print(f"[{tag}] arm {a.arm} dual_tail={arch.dual_tail} | "
          f"{a.batches:,} batches x {cfg.batch_size} = "
          f"{a.batches * cfg.batch_size:,} positions | {nsb} sb x {a.bps} | "
          f"lr {a.lr} cosine->{a.lr * 0.3 ** 3:.2e} | frozen {len(frozen)} tensors",
          flush=True)
    hv0 = heldout_value_loss(params, arch)
    print(f"[{tag}] heldout sb 0 (fresh head) | value {hv0:.6f}", flush=True)

    running: dict = {}
    t0 = time.time()
    hv = hv0
    for i, batch in enumerate(batches):
        params, opt_state, metrics = step(params, opt_state, batch)
        for k, v in metrics.items():
            running[k] = running.get(k, 0.0) + float(v)
        bidx, sb = i % a.bps, i // a.bps + 1
        if bidx == a.bps - 1:
            c = {k: v / a.bps for k, v in running.items()}
            rate = ((i + 1) * cfg.batch_size) / (time.time() - t0)
            # device params straight in: jnp.asarray is a no-op on them, so
            # this avoids a 434 MB host copy per eval (15 GB box, ~3 GB free)
            hv = heldout_value_loss(params, arch)
            print(f"[{tag}] sb {sb}/{nsb} | train_value {c.get('value', 0):.6f} | "
                  f"heldout {hv:.6f} | {rate:,.0f} pos/s | "
                  f"{(time.time() - t0) / 60:.1f} min", flush=True)
            running = {}
        if i + 1 >= a.batches:
            break

    hv = heldout_value_loss(params, arch)
    wall = time.time() - t0

    # FREEZE ORACLE: the trunk must be BIT-identical after every step (the
    # D47L contract — zeroed grads alone would NOT be enough, because AdamW's
    # decoupled decay multiplies every param regardless of gradient).
    moved = [n for n, v in fp0.items() if _fingerprint(params[n]) != v]
    frozen_ok = not moved
    print(f"[{tag}] freeze oracle ({len(fp0)} trunk tensors): "
          f"{'OK' if frozen_ok else 'FAIL, moved: ' + str(moved)}", flush=True)

    if a.save_head:
        p = Path(a.save_head)
        p.parent.mkdir(parents=True, exist_ok=True)
        np.savez(p, **{k: np.asarray(params[k]) for k in HEAD_TENSORS})
        print(f"[{tag}] head saved -> {p}", flush=True)

    print("RESULT " + json.dumps({
        "tag": tag, "arm": a.arm, "dual_tail": bool(arch.dual_tail),
        "head_seed": a.head_seed, "data_seed": a.data_seed,
        "batches": a.batches, "positions": a.batches * cfg.batch_size,
        "lr": a.lr, "bps": a.bps,
        "heldout_init": hv0, "heldout_final": hv,
        "wall_s": round(wall, 1), "gpu_hours": round(wall / 3600.0, 3),
        "frozen_ok": frozen_ok,
        "head_params": sum(int(np.prod(arch.param_shapes()[n]))
                           for n in HEAD_TENSORS),
    }), flush=True)
    return 0 if frozen_ok else 3


if __name__ == "__main__":
    raise SystemExit(main())
