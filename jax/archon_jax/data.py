"""Loader candidate (a): numpy structured-dtype reader over bulletformat .bin.

The corpus is memory-mapped (32-byte PODs, spec.RECORD_DTYPE) so the OS page
cache does all buffering — the same property the Rust ``Mixed`` reader exploits
(d40_features.rs:56-59). Feature extraction is the vectorized numpy path in
features.py; batches are mapped in worker threads (numpy releases the GIL in
the hot ops) with a small prefetch queue, mirroring bullet's
ReadMapLoader(threads=6) shape.

IMPORTANT data contract (D46): this path applies NO load-time filter, exactly
like the trainer's bulletformat branch (d40:663-673) — the corpus MUST be
pre-filtered to trainer parity (docs/research/data-recipe-contract.md). The
sfbinpack streaming path stays in Rust by design; see the study doc's loader
section (sfbinpack decode + filter + skip is a solved, fast, tested Rust
component — the JAX pipeline consumes either a pre-converted .bin or a
Rust-side conversion step).
"""

from __future__ import annotations

import concurrent.futures as cf
import queue
import threading

import numpy as np

from .features import canonical_orient, map_batch, policy_move_id
from .spec import POLICY_MASK_ILLEGAL, POLICY_VOCAB, RECORD_DTYPE, TrainConfig


class BulletBin:
    """Memory-mapped bulletformat corpus."""

    def __init__(self, path: str):
        self.path = path
        self.recs = np.memmap(path, dtype=RECORD_DTYPE, mode="r")

    def __len__(self) -> int:
        return len(self.recs)

    def slice(self, start: int, count: int) -> np.ndarray:
        return np.asarray(self.recs[start:start + count])


def make_batches(bin_: BulletBin, cfg: TrainConfig, *, policy: bool,
                 threats: bool = True, legal_mask: bool = False,
                 seed: int = 0, workers: int = 4, prefetch: int = 8,
                 max_batches: int | None = None):
    """Yield mapped training batches (dicts of numpy arrays).

    Batches are CONTIGUOUS windows at shuffled offsets (the corpus is expected
    pre-shuffled, like every *.shuf.bin — same contract as the Rust fold-in
    reader). Set workers=0 for a synchronous loop (benchmarking).
    """
    n_batches = len(bin_) // cfg.batch_size
    rng = np.random.default_rng(seed)
    order = rng.permutation(n_batches)
    if max_batches is not None:
        order = order[:max_batches]

    def map_one(i: int) -> dict:
        recs = bin_.slice(int(i) * cfg.batch_size, cfg.batch_size)
        out = map_batch(recs, wdl=cfg.wdl, score_multiplier=cfg.score_multiplier,
                        scale=cfg.scale, policy=policy, threats=threats,
                        target_offset=getattr(cfg, "target_offset", 0.0))
        if policy and legal_mask:
            out["pol_mask"] = legal_move_masks(recs, out["pol_target_id"])
        return out

    if workers == 0:
        for i in order:
            yield map_one(i)
        return

    q: queue.Queue = queue.Queue(maxsize=prefetch)
    stop = threading.Event()

    def producer():
        with cf.ThreadPoolExecutor(workers) as pool:
            for batch in _stream(pool, order, prefetch, map_one, stop):
                if stop.is_set():
                    break
                q.put(batch)
        q.put(None)

    t = threading.Thread(target=producer, daemon=True)
    t.start()
    try:
        while (item := q.get()) is not None:
            yield item
    finally:
        stop.set()


def make_mixed_batches(mix, cfg: TrainConfig, *, policy: bool,
                       threats: bool = True, legal_mask: bool = False,
                       seed: int = 0, workers: int = 4, prefetch: int = 8,
                       total_batches: int | None = None,
                       skip_batches: int = 0):
    """Multi-corpus batch stream: ``mix`` = [(BulletBin, weight), ...].

    Each batch's corpus is drawn ~ weights; windows within a corpus follow an
    independent shuffled permutation that re-permutes on exhaustion (a new
    epoch of that corpus). The schedule is fully determined by ``seed`` —
    reproducible and resumable by skipping consumed indices.

    D46: every corpus must be pre-shuffled (*.shuf.bin) AND unit-normalized
    to the SAME score frame before mixing (per-pack SFBINPACK_SCORE_DIV at
    convert time, or rescale_records.py) — mixing frames silently corrupts
    the value target.
    """
    bins = [b for b, _ in mix]
    weights = np.asarray([wt for _, wt in mix], dtype=np.float64)
    weights = weights / weights.sum()
    counts = [len(b) // cfg.batch_size for b in bins]
    total = (total_batches if total_batches is not None
             else cfg.batches_per_superbatch * cfg.superbatches)
    rng = np.random.default_rng(seed)
    choice = rng.choice(len(bins), size=total, p=weights)
    orders = [rng.permutation(c) for c in counts]
    cursors = [0] * len(bins)
    sched = []
    for c in choice:
        c = int(c)
        if cursors[c] >= counts[c]:
            orders[c] = rng.permutation(counts[c])
            cursors[c] = 0
        sched.append((c, int(orders[c][cursors[c]])))
        cursors[c] += 1
    # Resume support: the schedule is a pure function of (seed, sizes), so a
    # restarted run regenerates it and skips what the crashed run consumed.
    sched = sched[skip_batches:]

    def map_one(job) -> dict:
        c, i = job
        recs = bins[c].slice(i * cfg.batch_size, cfg.batch_size)
        out = map_batch(recs, wdl=cfg.wdl, score_multiplier=cfg.score_multiplier,
                        scale=cfg.scale, policy=policy, threats=threats,
                        target_offset=getattr(cfg, "target_offset", 0.0))
        if policy and legal_mask:
            out["pol_mask"] = legal_move_masks(recs, out["pol_target_id"])
        return out

    if workers == 0:
        for job in sched:
            yield map_one(job)
        return

    q: queue.Queue = queue.Queue(maxsize=prefetch)
    stop = threading.Event()

    def producer():
        with cf.ThreadPoolExecutor(workers) as pool:
            for batch in _stream(pool, sched, prefetch, map_one, stop):
                if stop.is_set():
                    break
                q.put(batch)
        q.put(None)

    t = threading.Thread(target=producer, daemon=True)
    t.start()
    try:
        while (item := q.get()) is not None:
            yield item
    finally:
        stop.set()


def _stream(pool, order, prefetch, fn, stop):
    """Submit-ahead window of `prefetch` futures, yielding results in order."""
    futs = [pool.submit(fn, i) for i in order[:prefetch]]
    nxt = prefetch
    while futs:
        if stop.is_set():
            return
        f = futs.pop(0)
        if nxt < len(order):
            futs.append(pool.submit(fn, order[nxt]))
            nxt += 1
        yield f.result()


# ---------------------------------------------------------------------------
# Legal-move policy mask (trainer parity: d40_features.rs 913-941)
# ---------------------------------------------------------------------------
# The Rust mapper reconstructs the STORED (stm-canonicalized) board — no
# castling rights, no ep square (bulletformat drops them; d40:162-168 documents
# why that is harmless) — runs the engine's generate_legal, and writes an
# additive mask (0 legal / -30 illegal) in the canonical policy frame.
#
# Prototype implementation: python-chess with castling/ep cleared reproduces
# exactly that conservative mask. It is the SLOW path (~5-15k pos/s/thread);
# the production design is a PyO3 binding of the engine movegen or a
# preprocessed sidecar — see the study doc. Only needed when POL_LAMBDA>0.

_CHESS = None


def _chess():
    global _CHESS
    if _CHESS is None:
        import chess as _c
        _CHESS = _c
    return _CHESS


_PROMO_CODE = {None: 0, 2: 1, 3: 2, 4: 3, 5: 0}  # chess.KNIGHT..QUEEN -> our code


def legal_move_masks(recs: np.ndarray, pol_target_id: np.ndarray) -> np.ndarray:
    """(N, 1858) float32 additive masks. Rows without a packed move keep an
    all-zero mask (their CE is identically zero either way — d40:891-895).

    Fast path: the PyO3 engine-movegen masker (pymask/, bit-exact vs this
    reference, >1M pos/s). Falls back to the python-chess prototype if the
    module is absent. NOTE the fast path fixed a real bug the prototype
    exposed: Board::empty+put_piece reconstruction leaks pinned-piece moves
    (stale StateInfo); pymask reconstructs via Board::from_fen."""
    try:
        import archon_mask
        flat = np.ascontiguousarray(recs).view(np.uint8).reshape(-1)
        return np.asarray(
            archon_mask.legal_move_masks(flat, pol_target_id.astype(np.int64)),
            dtype=np.float32)
    except ImportError:
        pass
    chess = _chess()
    n = len(recs)
    masks = np.zeros((n, POLICY_VOCAB), dtype=np.float32)
    orient = canonical_orient(recs["ksq"].astype(np.int64) & 63)
    for i in range(n):
        if pol_target_id[i] < 0:
            continue
        board = chess.Board(None)
        occ = int(recs["occ"][i])
        pcs = recs["pcs"][i]
        idx = 0
        while occ:
            sq = (occ & -occ).bit_length() - 1
            occ &= occ - 1
            nib = (pcs[idx // 2] >> (4 * (idx % 2))) & 0xF
            idx += 1
            color = chess.WHITE if (nib & 8) == 0 else chess.BLACK
            board.set_piece_at(sq, chess.Piece((nib & 7) + 1, color))
        board.turn = chess.WHITE
        row = masks[i]
        row[:] = POLICY_MASK_ILLEGAL
        o = int(orient[i])
        froms, tos, promos = [], [], []
        for mv in board.legal_moves:
            froms.append(mv.from_square ^ o)
            tos.append(mv.to_square ^ o)
            promos.append(_PROMO_CODE.get(mv.promotion, 0))
        ids = policy_move_id(np.array(froms), np.array(tos), np.array(promos))
        row[ids[ids >= 0]] = 0.0
        # frame tripwire (d40:933-941): the packed target must be legal
        if row[pol_target_id[i]] != 0.0:
            raise AssertionError(
                f"policy frame bug: target {pol_target_id[i]} not legal (row {i})")
    return masks
