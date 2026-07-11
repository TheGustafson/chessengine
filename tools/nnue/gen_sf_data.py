#!/usr/bin/env python3
"""Bootstrap NNUE datagen teacher: Stockfish self-play -> filtered text shards.

design/15-nnue-datagen.md (D28 Tier-A scaffold). This is DISCARDABLE
development tooling that drives a *locally-compiled* Stockfish (GPLv3, never
shipped) to label positions for the first NNUE net. It emits bulletformat's
documented text interchange line

    <FEN> | <score> | <result>

one position per line (§3.2.2): `score` is White-relative centipawns (SF's
UCI-normalised `score cp`, sign-flipped for Black-to-move, §3.2.4), `result`
is the game's White-relative outcome 1.0 / 0.5 / 0.0. bulletformat's
converter (invoked later by build_corpus.sh) turns these into the 32-byte
`ChessBoard` records bullet trains on -- we deliberately do NOT hand-pack the
binary (§3.2.1); the text line is correct-by-construction.

Curation (§3.5): a position is written only if it (1) is past the opening
plies, (2) is not in check, (3) has a QUIET SF bestmove (no capture / ep /
promotion -- the highest-leverage filter, since the net is a *static* eval),
(4) has |score| < SCORE_CAP and is not a mate score, (5) is not a within-run
duplicate (64-bit position key), and (6) is one of at most P_cap positions
sampled evenly across a game. A piece-count / game-phase histogram is
tracked for the meta.json balance report (filter 7, monitor-only here).

Games are fixed-node SF self-play (§3.3.2, default 5000 nodes/move) started
from a varied opening (random R_open plies + a ±600cp balance gate, or a
supplied opening book), adjudicated by explicit rules (§3.4). python-chess
provides all rules/FEN/adjudication (§3.7); SF's search dominates runtime.

Parallel / sharded / resumable / throttleable (§3.7):
  * --workers N processes, each owning ONE single-threaded SF (Threads=1);
  * each worker writes data/nnue/raw/shard_<seed>_<worker>.txt and only ever
    contains COMPLETE games (a game's lines are buffered and flushed
    atomically with a progress checkpoint), so a killed run resumes cleanly;
  * determinism/regenerability: each game's RNG is derived purely from
    (--seed, global_game_index) -- INDEPENDENT of --workers -- so the same
    --seed + settings + SF binary reproduce a byte-identical corpus, and the
    game content does not shift when you change the worker count. (This
    strengthens the doc §3.7 per-worker-stream clause into a per-game
    stream, which is what makes resume-without-replay exact.)
  * --nice lowers priority; pin cores with `taskset -c ...` (the bulk run is
    THROTTLED so it never starves the concurrent P9 SPRT -- and is DEFERRED
    pending human review of the sample; nothing here launches it).

Usage (SMOKE -- a handful of games, the only thing this workflow runs):
    python3 tools/nnue/gen_sf_data.py --seed 1 --sf engines/stockfish-datagen \
        --workers 2 --games 8 --nodes 5000 --out data/nnue/raw

All filter / adjudication thresholds are CLI knobs (retuned by the human
before the bulk run). Requires: python-chess, a locally-built SF.
"""

from __future__ import annotations

import argparse
import json
import multiprocessing as mp
import os
import random
import sys
import time
from dataclasses import asdict, dataclass, field
from pathlib import Path

import chess
import chess.engine
import chess.polyglot

# Odd 64-bit constant (fractional part of the golden ratio) for seed mixing;
# same constant the doc names for worker-seed spacing (§3.7).
GAMMA = 0x9E3779B97F4A7C15
MASK64 = (1 << 64) - 1
MATE_SENTINEL = 100_000  # stands in for a mate score in adjudication math


@dataclass
class Config:
    sf: str
    out: str
    seed: int
    workers: int
    games: int  # total games across all workers (0 => unbounded, use max_positions)
    nodes: int
    hash_mb: int
    gate_nodes: int
    gate_cp: int
    ropen_min: int
    ropen_max: int
    open_retries: int
    book: str  # path to an EPD opening book, or "" for random openings
    score_cap: int
    win_cp: int
    win_plies: int
    draw_cp: int
    draw_after: int
    draw_plies: int
    max_plies: int
    p_cap: int
    max_positions: int  # 0 => unbounded
    nice: int


@dataclass
class Stats:
    games_played: int = 0
    games_aborted_open: int = 0  # opening balance gate never satisfied
    positions_written: int = 0
    # rejection tallies (a position may be rejected by the first failing gate)
    rej_in_check: int = 0
    rej_noisy_bestmove: int = 0  # capture / ep / promotion
    rej_score_cap: int = 0
    rej_mate: int = 0
    rej_dup: int = 0
    rej_pcap: int = 0  # dropped by the per-game cap
    result_w: int = 0
    result_d: int = 0
    result_l: int = 0
    piece_hist: dict = field(default_factory=dict)  # piece-count 2..32 -> n
    phase_hist: dict = field(default_factory=dict)  # phase 0..24 -> n

    def bump(self, d: dict, key: int) -> None:
        d[str(key)] = d.get(str(key), 0) + 1


def phase_of(board: chess.Board) -> int:
    """Tapered game phase 0..24 (SF-style non-pawn material weight)."""
    w = {chess.KNIGHT: 1, chess.BISHOP: 1, chess.ROOK: 2, chess.QUEEN: 4}
    ph = 0
    for pt, wt in w.items():
        ph += wt * len(board.pieces(pt, chess.WHITE))
        ph += wt * len(board.pieces(pt, chess.BLACK))
    return min(24, ph)


def is_noisy(board: chess.Board, move: chess.Move) -> bool:
    """§3.5(3): bestmove is a capture / en-passant / promotion."""
    return board.is_capture(move) or move.promotion is not None


def pos_key(board: chess.Board) -> int:
    """§3.5(5): 64-bit identity over placement+stm+castling+ep (not clocks)."""
    return chess.polyglot.zobrist_hash(board)


def load_book(path: str) -> list[str]:
    fens = []
    for line in Path(path).read_text().splitlines():
        line = line.strip()
        if line:
            fens.append(line)
    if not fens:
        raise SystemExit(f"opening book {path} is empty")
    return fens


def make_opening(cfg: Config, rng, engine, book, game_token) -> chess.Board | None:
    """A varied, roughly balanced starting position (§3.4). None => give up."""
    if book is not None:
        # Book lines are pre-balanced; no balance gate needed (§3.4 alt).
        board = chess.Board(rng.choice(book))
        return board if not board.is_game_over() else None
    for _ in range(cfg.open_retries):
        board = chess.Board()
        r_open = rng.randint(cfg.ropen_min, cfg.ropen_max)
        ok = True
        for _ in range(r_open):
            moves = list(board.legal_moves)
            if not moves:
                ok = False
                break
            board.push(rng.choice(moves))
            if board.is_game_over():
                ok = False
                break
        if not ok:
            continue
        info = engine.analyse(board, chess.engine.Limit(nodes=cfg.gate_nodes),
                              game=game_token)
        sw = info["score"].white().score(mate_score=MATE_SENTINEL)
        if abs(sw) <= cfg.gate_cp:
            return board
    return None


def play_game(cfg: Config, rng, engine, book, game_index: int, stats: Stats):
    """One fixed-node SF self-play game. Returns (lines, result_char)."""
    game_token = object()  # forces `ucinewgame` -> fresh TT -> determinism
    board = make_opening(cfg, rng, engine, book, game_token)
    if board is None:
        stats.games_aborted_open += 1
        return [], None

    samples = []  # dicts: fen, score, in eligible (quiet) positions only
    result = None  # White-relative float once adjudicated / terminal
    win_streak, win_side, draw_streak = 0, 0, 0

    while True:
        ply = len(board.move_stack)
        # --- terminal / adjudication checks that need no search ---
        if board.is_checkmate():
            result = 0.0 if board.turn == chess.WHITE else 1.0
            break
        if (board.is_stalemate() or board.is_insufficient_material()
                or board.is_repetition(3) or board.halfmove_clock >= 100):
            result = 0.5
            break
        if ply >= cfg.max_plies:
            result = 0.5
            break

        info = engine.analyse(board, chess.engine.Limit(nodes=cfg.nodes),
                              game=game_token)
        sc = info["score"].white()
        is_mate = sc.is_mate()
        sw = sc.score(mate_score=MATE_SENTINEL)
        pv = info.get("pv") or []
        bestmove = pv[0] if pv else next(iter(board.legal_moves))

        # --- curation filters (§3.5) applied to THIS position ---
        if board.is_check():
            stats.rej_in_check += 1
        elif is_noisy(board, bestmove):
            stats.rej_noisy_bestmove += 1
        elif is_mate:
            stats.rej_mate += 1
        elif abs(sw) >= cfg.score_cap:
            stats.rej_score_cap += 1
        else:
            samples.append({
                "fen": board.fen(),
                "score": sw,
                "pc": chess.popcount(board.occupied),
                "ph": phase_of(board),
            })

        # --- adjudication streak bookkeeping (§3.4) ---
        if sw >= cfg.win_cp:
            win_streak = win_streak + 1 if win_side == 1 else 1
            win_side = 1
        elif sw <= -cfg.win_cp:
            win_streak = win_streak + 1 if win_side == -1 else 1
            win_side = -1
        else:
            win_streak, win_side = 0, 0
        if win_streak >= cfg.win_plies:
            result = 1.0 if win_side == 1 else 0.0
            break

        if ply >= cfg.draw_after and abs(sw) <= cfg.draw_cp:
            draw_streak += 1
        else:
            draw_streak = 0
        if draw_streak >= cfg.draw_plies:
            result = 0.5
            break

        board.push(bestmove)

    # --- per-game cap: keep <= P_cap eligible positions, evenly spread ---
    if len(samples) > cfg.p_cap:
        n = len(samples)
        keep_idx = sorted({round(i * (n - 1) / (cfg.p_cap - 1))
                           for i in range(cfg.p_cap)})
        stats.rej_pcap += n - len(keep_idx)
        samples = [samples[i] for i in keep_idx]

    return samples, result


def result_char(result: float) -> str:
    return {1.0: "1.0", 0.5: "0.5", 0.0: "0.0"}[result]


def worker_main(worker_id: int, cfg: Config, shared_written, lock):
    if cfg.nice:
        try:
            os.nice(cfg.nice)  # SF child inherits the niceness
        except OSError:
            pass

    out_dir = Path(cfg.out)
    out_dir.mkdir(parents=True, exist_ok=True)
    shard = out_dir / f"shard_{cfg.seed}_{worker_id}.txt"
    prog = out_dir / f"shard_{cfg.seed}_{worker_id}.progress"
    statf = out_dir / f"shard_{cfg.seed}_{worker_id}.stats.json"

    # Resume: recover last completed global game index and the dedup set.
    last_done = -1
    if prog.exists():
        last_done = int(prog.read_text().strip() or "-1")
    seen: set[int] = set()
    stats = Stats()
    if shard.exists() and last_done >= 0:
        for line in shard.read_text().splitlines():
            if " | " in line:
                fen = line.split(" | ", 1)[0]
                seen.add(pos_key(chess.Board(fen)))
        stats.positions_written = len(seen)  # coarse resume of the counter

    book = load_book(cfg.book) if cfg.book else None

    # SF engine: single-threaded, fixed hash for reproducibility.
    engine = chess.engine.SimpleEngine.popen_uci(cfg.sf, timeout=30)
    engine.configure({"Threads": 1, "Hash": cfg.hash_mb})

    fout = shard.open("a", buffering=1)

    def flush_stats():
        statf.write_text(json.dumps(asdict(stats), indent=2, sort_keys=True))

    # Round-robin over global game indices: worker w owns g where g%W == w.
    gi = worker_id
    interrupted = False
    try:
        while cfg.games == 0 or gi < cfg.games:
            if cfg.max_positions and shared_written.value >= cfg.max_positions:
                break
            if gi <= last_done:
                gi += cfg.workers
                continue

            game_seed = (cfg.seed + gi * GAMMA) & MASK64
            rng = random.Random(game_seed)

            samples, result = play_game(cfg, rng, engine, book, gi, stats)
            wrote = 0
            if result is not None:
                stats.games_played += 1
                if result == 1.0:
                    stats.result_w += 1
                elif result == 0.5:
                    stats.result_d += 1
                else:
                    stats.result_l += 1
                rc = result_char(result)
                buf = []
                for s in samples:
                    k = pos_key(chess.Board(s["fen"]))
                    if k in seen:
                        stats.rej_dup += 1
                        continue
                    seen.add(k)
                    buf.append(f"{s['fen']} | {s['score']} | {rc}")
                    stats.bump(stats.piece_hist, s["pc"])
                    stats.bump(stats.phase_hist, s["ph"])
                    wrote += 1
                if buf:
                    fout.write("\n".join(buf) + "\n")
                    fout.flush()
                stats.positions_written += wrote

            # Atomic checkpoint: shard already flushed, now advance progress.
            prog.write_text(str(gi))
            flush_stats()
            if wrote and cfg.max_positions:
                with lock:
                    shared_written.value += wrote
            gi += cfg.workers
    except KeyboardInterrupt:
        interrupted = True
    finally:
        fout.close()
        flush_stats()
        try:
            engine.quit()
        except Exception:
            engine.close()

    tag = "interrupted" if interrupted else "done"
    print(f"[worker {worker_id}] {tag}: games={stats.games_played} "
          f"positions={stats.positions_written} -> {shard.name}",
          file=sys.stderr)


def parse_args(argv=None) -> Config:
    p = argparse.ArgumentParser(description="SF self-play NNUE datagen (design/15)")
    p.add_argument("--sf", default="engines/stockfish-datagen",
                   help="path to the locally-compiled SF teacher")
    p.add_argument("--out", default="data/nnue/raw", help="text shard output dir")
    p.add_argument("--seed", type=int, required=True, help="master RNG seed")
    p.add_argument("--workers", type=int, default=2)
    p.add_argument("--games", type=int, default=0,
                   help="total games across workers (0 => bounded by --max-positions)")
    p.add_argument("--nodes", type=int, default=5000, help="SF nodes/move (label budget)")
    p.add_argument("--hash", dest="hash_mb", type=int, default=16, help="SF Hash MB/worker")
    p.add_argument("--gate-nodes", type=int, default=5000, help="balance-gate eval budget")
    p.add_argument("--gate-cp", type=int, default=600, help="opening balance |cp| gate")
    p.add_argument("--ropen-min", type=int, default=8)
    p.add_argument("--ropen-max", type=int, default=10)
    p.add_argument("--open-retries", type=int, default=16)
    p.add_argument("--book", default="", help="EPD opening book (overrides random openings)")
    p.add_argument("--score-cap", type=int, default=2500, help="|cp| exclusion cap")
    p.add_argument("--win-cp", type=int, default=2500)
    p.add_argument("--win-plies", type=int, default=4)
    p.add_argument("--draw-cp", type=int, default=8)
    p.add_argument("--draw-after", type=int, default=80)
    p.add_argument("--draw-plies", type=int, default=10)
    p.add_argument("--max-plies", type=int, default=400)
    p.add_argument("--p-cap", type=int, default=10, help="max positions/game")
    p.add_argument("--max-positions", type=int, default=0, help="global stop (0 => off)")
    p.add_argument("--nice", type=int, default=10, help="niceness (throttle)")
    a = p.parse_args(argv)
    if a.games == 0 and a.max_positions == 0:
        p.error("supply --games and/or --max-positions (else unbounded)")
    return Config(
        sf=str(Path(a.sf)), out=a.out, seed=a.seed, workers=a.workers,
        games=a.games, nodes=a.nodes, hash_mb=a.hash_mb, gate_nodes=a.gate_nodes,
        gate_cp=a.gate_cp, ropen_min=a.ropen_min, ropen_max=a.ropen_max,
        open_retries=a.open_retries, book=a.book, score_cap=a.score_cap,
        win_cp=a.win_cp, win_plies=a.win_plies, draw_cp=a.draw_cp,
        draw_after=a.draw_after, draw_plies=a.draw_plies, max_plies=a.max_plies,
        p_cap=a.p_cap, max_positions=a.max_positions, nice=a.nice,
    )


def main(argv=None) -> int:
    cfg = parse_args(argv)
    if not Path(cfg.sf).exists():
        raise SystemExit(f"SF teacher not found: {cfg.sf} (build it, design/15 §3.3)")
    Path(cfg.out).mkdir(parents=True, exist_ok=True)

    t0 = time.perf_counter()
    ctx = mp.get_context("spawn")
    shared_written = ctx.Value("q", 0)
    lock = ctx.Lock()
    procs = [ctx.Process(target=worker_main, args=(w, cfg, shared_written, lock))
             for w in range(cfg.workers)]
    for pr in procs:
        pr.start()
    try:
        for pr in procs:
            pr.join()
    except KeyboardInterrupt:
        for pr in procs:
            pr.terminate()
        for pr in procs:
            pr.join()

    # Aggregate a run-level summary from the per-worker stats sidecars.
    total = 0
    for w in range(cfg.workers):
        sf = Path(cfg.out) / f"shard_{cfg.seed}_{w}.stats.json"
        if sf.exists():
            total += json.loads(sf.read_text()).get("positions_written", 0)
    dt = time.perf_counter() - t0
    print(f"[gen] seed={cfg.seed} workers={cfg.workers} wrote ~{total} positions "
          f"in {dt:.1f}s ({total / dt:.0f} pos/s)", file=sys.stderr)
    return 0


if __name__ == "__main__":
    sys.exit(main())
