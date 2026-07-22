"""CPU training smoke: overfit ~10k real rows on a scaled-down D47c graph.

Demonstrates the full loop end-to-end on CPU within the RAM budget:
  data -> mapped batches (incl. synthesized policy targets + legal masks,
  exercising the canonical-frame tripwire) -> jit'd bullet-AdamW steps ->
  falling value MSE + policy CE -> raw & quant export -> raw round-trip ->
  netpack.py parses the quant (run separately; prints the env line).

The arch is ArchSpec(hl=256, l2=32, l3=16) — same graph code, same layout
conventions, ~23M params, fits CPU. The FULL-size graph is proven by the
byte-parity + forward-parity tools; the training loop is size-agnostic.

Run:
  cd jax && JAX_PLATFORMS=cpu XLA_PYTHON_CLIENT_PREALLOCATE=false \
    uv run --no-project --with jax --with optax --with chess -- \
    python tools/train_smoke.py <corpus.bin> <out_dir>
"""

import os
import sys
import time
from pathlib import Path

import numpy as np

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))
os.environ.setdefault("JAX_PLATFORMS", "cpu")
os.environ.setdefault("XLA_PYTHON_CLIENT_PREALLOCATE", "false")

from archon_jax import checkpoint  # noqa: E402
from archon_jax.data import legal_move_masks  # noqa: E402
from archon_jax.features import canonical_orient, map_batch  # noqa: E402
from archon_jax.spec import RECORD_DTYPE, ArchSpec, TrainConfig  # noqa: E402

N_ROWS = 10240
BATCH = 1024
STEPS = 400


def synthesize_teacher_moves(recs: np.ndarray, seed: int = 7) -> np.ndarray:
    """Give each row a pseudo teacher move (a random legal move, packed in the
    canonical frame exactly like the trainer's loader would: oriented from/to
    + promo code in extra[0..3])."""
    import chess
    rng = np.random.default_rng(seed)
    recs = recs.copy()
    orient = canonical_orient(recs["ksq"].astype(np.int64) & 63)
    promo_code = {None: 0, chess.KNIGHT: 1, chess.BISHOP: 2, chess.ROOK: 3,
                  chess.QUEEN: 0}
    kept = np.ones(len(recs), dtype=bool)
    for i in range(len(recs)):
        board = chess.Board(None)
        occ = int(recs["occ"][i])
        pcs = recs["pcs"][i]
        idx = 0
        while occ:
            sq = (occ & -occ).bit_length() - 1
            occ &= occ - 1
            nib = (pcs[idx // 2] >> (4 * (idx % 2))) & 0xF
            idx += 1
            board.set_piece_at(
                sq, chess.Piece((nib & 7) + 1,
                                chess.WHITE if (nib & 8) == 0 else chess.BLACK))
        board.turn = chess.WHITE
        moves = list(board.legal_moves)
        if not moves:
            kept[i] = False
            continue
        mv = moves[rng.integers(len(moves))]
        o = int(orient[i])
        recs["extra"][i] = (mv.from_square ^ o, mv.to_square ^ o,
                            promo_code.get(mv.promotion, 0))
    return recs[kept]


def main(corpus: str, out_dir: str):
    import jax
    import jax.numpy as jnp

    from archon_jax.model import init_params
    from archon_jax.train import make_update_step, save_checkpoint

    arch = ArchSpec(hl=256, l2=32, l3=16)
    cfg = TrainConfig(batch_size=BATCH, batches_per_superbatch=100,
                      superbatches=4, lr_end_superbatch=4, pol_lambda=0.05)

    print(f"loading {N_ROWS} rows + synthesizing teacher moves...")
    recs = np.fromfile(corpus, dtype=RECORD_DTYPE, count=N_ROWS)
    recs = synthesize_teacher_moves(recs)
    print(f"  {len(recs)} rows kept")

    # Pre-map all batches once (overfit: fixed set, reused each epoch).
    batches = []
    for i in range(0, len(recs) - BATCH + 1, BATCH):
        chunk = recs[i:i + BATCH]
        b = map_batch(chunk, wdl=cfg.wdl, score_multiplier=cfg.score_multiplier,
                      scale=cfg.scale, policy=True)
        b["pol_mask"] = legal_move_masks(chunk, b["pol_target_id"])
        batches.append(b)
    print(f"  {len(batches)} batches mapped (masks passed the frame tripwire)")

    params = init_params(arch, seed=0x00D40FEA)
    opt, step = make_update_step(arch, cfg)
    opt_state = opt.init(params)

    t0 = time.time()
    history = []
    for s in range(STEPS):
        b = batches[s % len(batches)]
        jb = {k: jnp.asarray(v) for k, v in b.items()}
        params, opt_state, metrics = step(params, opt_state, jb)
        if s % 25 == 0 or s == STEPS - 1:
            m = {k: float(v) for k, v in metrics.items()}
            history.append((s, m))
            print(f"step {s:4d}  loss {m['loss']:.6f}  value {m['value']:.6f}"
                  f"  pol_ce {m.get('pol_ce', float('nan')):.4f}")
    dt = time.time() - t0
    print(f"{STEPS} steps in {dt:.1f}s ({STEPS * BATCH / dt:,.0f} pos/s)")

    first, last = history[0][1], history[-1][1]
    assert last["value"] < first["value"] * 0.8, "value loss must fall"
    assert last["pol_ce"] < first["pol_ce"] * 0.8, "policy CE must fall"
    print(f"LOSS FALLS: value {first['value']:.4f} -> {last['value']:.4f}, "
          f"pol_ce {first['pol_ce']:.3f} -> {last['pol_ce']:.3f}")

    host = {k: np.asarray(v) for k, v in params.items()}
    Path(out_dir).mkdir(parents=True, exist_ok=True)
    save_checkpoint(host, arch, out_dir, "smoke")

    # raw round-trip
    back = checkpoint.open_raw(Path(out_dir) / "smoke-raw.bin",
                               arch.param_shapes(), mmap=False)
    for k in host:
        assert np.array_equal(host[k].astype(np.float32), back[k]), k
    print("raw round-trip: byte-faithful")

    # quant round-trip sanity through our own dequantiser
    deq = checkpoint.dequantise_quant(Path(out_dir) / "smoke-quant.bin", arch)
    err = max(float(np.abs(host[k] - deq[k]).max()) for k in host)
    print(f"quant round-trip max |err|: {err:.6f} (<= 1/(2*64) + f32 eps expected)")

    print("\nnetpack check (run this, pure python+numpy):\n"
          f"  D40_ACC=256 D40_L2=32 D40_L3=16 D40_POLICY=1 "
          f"uv run --no-project --with numpy -- python tools/nnue/netpack.py "
          f"{out_dir}/smoke-quant.bin {out_dir}/smoke.cenn d40")


if __name__ == "__main__":
    main(sys.argv[1], sys.argv[2])
