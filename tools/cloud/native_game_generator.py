#!/usr/bin/env python3
"""Generate deterministic, full-history native games for durable Archon data."""

from __future__ import annotations

import argparse
from dataclasses import asdict, dataclass
import gzip
import hashlib
import json
import math
import os
from pathlib import Path
import random
import shutil
import tempfile
import time
from typing import Any

import chess
import chess.engine
import chess.polyglot
from google.cloud import storage

from shard_lc0_positions import parse_gs, storage_client


SCHEMA = "archon.native-games.v1"
GAMMA = 0x9E3779B97F4A7C15
MASK64 = (1 << 64) - 1
MATE_SCORE = 30_000


@dataclass(frozen=True)
class SearchRecord:
    ply: int
    generator: str
    score_cp: int
    mate_plies: int | None
    best_move: str
    chosen_move: str
    chosen_rank: int
    depth: int
    seldepth: int
    nodes: int


def sha256(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as handle:
        for block in iter(lambda: handle.read(8 * 1024 * 1024), b""):
            digest.update(block)
    return digest.hexdigest()


def canonical_hash(value: Any) -> str:
    return hashlib.sha256(
        json.dumps(value, sort_keys=True, separators=(",", ":")).encode()
    ).hexdigest()


def engine_identity(path: Path) -> dict[str, Any]:
    engine = chess.engine.SimpleEngine.popen_uci(str(path), timeout=30)
    try:
        return {
            "path": str(path.resolve()),
            "sha256": sha256(path),
            "uci_name": engine.id.get("name"),
            "uci_author": engine.id.get("author"),
            "uci_options": sorted(engine.options),
        }
    finally:
        engine.quit()


def configure(engine: chess.engine.SimpleEngine, hash_mb: int) -> None:
    options: dict[str, Any] = {}
    if "Threads" in engine.options:
        options["Threads"] = 1
    if "Hash" in engine.options:
        options["Hash"] = hash_mb
    if options:
        engine.configure(options)


def score_info(info: dict[str, Any], turn: chess.Color) -> tuple[int, int | None]:
    score = info["score"].pov(turn)
    mate = score.mate()
    cp = score.score(mate_score=MATE_SCORE)
    if cp is None:
        raise RuntimeError("engine returned no score")
    return int(cp), mate


def analyse_one(
    engine: chess.engine.SimpleEngine,
    board: chess.Board,
    nodes: int,
    game_token: object,
) -> tuple[chess.Move, dict[str, Any]]:
    info = engine.analyse(
        board, chess.engine.Limit(nodes=nodes), game=game_token,
        info=chess.engine.INFO_ALL,
    )
    pv = info.get("pv") or []
    if not pv or pv[0] not in board.legal_moves:
        raise RuntimeError(f"engine returned no legal PV for {board.fen()}")
    return pv[0], info


def choose_multipv(
    engine: chess.engine.SimpleEngine,
    board: chess.Board,
    nodes: int,
    count: int,
    temperature_cp: float,
    rng: random.Random,
    game_token: object,
) -> tuple[chess.Move, dict[str, Any], int, chess.Move]:
    count = min(count, board.legal_moves.count())
    infos = engine.analyse(
        board, chess.engine.Limit(nodes=nodes), game=game_token,
        multipv=count, info=chess.engine.INFO_ALL,
    )
    if not isinstance(infos, list):
        infos = [infos]
    candidates: list[tuple[chess.Move, dict[str, Any], int]] = []
    for rank, info in enumerate(infos):
        pv = info.get("pv") or []
        if pv and pv[0] in board.legal_moves:
            candidates.append((pv[0], info, rank))
    if not candidates:
        raise RuntimeError(f"MultiPV returned no legal candidate for {board.fen()}")
    scores = [score_info(info, board.turn)[0] for _, info, _ in candidates]
    best = max(scores)
    weights = [math.exp(max(-50.0, (score - best) / temperature_cp)) for score in scores]
    choice = rng.choices(range(len(candidates)), weights=weights, k=1)[0]
    chosen_move, chosen_info, chosen_rank = candidates[choice]
    return chosen_move, chosen_info, chosen_rank, candidates[0][0]


def random_balanced_opening(
    stockfish: chess.engine.SimpleEngine,
    cfg: dict[str, Any],
    rng: random.Random,
) -> tuple[chess.Board, list[str]] | None:
    for attempt in range(cfg["opening_retries"]):
        board = chess.Board()
        moves: list[str] = []
        plies = rng.randint(cfg["opening_min_plies"], cfg["opening_max_plies"])
        for _ in range(plies):
            legal = list(board.legal_moves)
            if not legal:
                break
            move = rng.choice(legal)
            moves.append(move.uci())
            board.push(move)
        if len(moves) != plies or board.is_game_over(claim_draw=True):
            continue
        _, info = analyse_one(stockfish, board, cfg["gate_nodes"], object())
        cp, mate = score_info(info, board.turn)
        if mate is None and abs(cp) <= cfg["gate_cp"]:
            return board, moves
    return None


def generator_for(
    stratum: str,
    game_index: int,
    turn: chess.Color,
) -> str:
    if stratum == "reckless_self":
        return "reckless"
    if stratum == "sf18_self":
        return "stockfish18"
    reckless_white = game_index % 2 == 0
    return "reckless" if turn == reckless_white else "stockfish18"


def state_key(board: chess.Board, repetition_count: int) -> str:
    material = f"{board.fen(en_passant='fen')}|rep={repetition_count}".encode()
    return hashlib.blake2b(material, digest_size=16).hexdigest()


def evenly_spaced(items: list[dict[str, Any]], limit: int) -> list[dict[str, Any]]:
    if len(items) <= limit:
        return items
    indices = sorted({round(i * (len(items) - 1) / (limit - 1)) for i in range(limit)})
    return [items[index] for index in indices]


def play_game(
    cfg: dict[str, Any],
    task_index: int,
    game_index: int,
    engines: dict[str, chess.engine.SimpleEngine],
    rng: random.Random,
    seen_states: set[str],
) -> dict[str, Any] | None:
    opening = random_balanced_opening(engines["stockfish18"], cfg, rng)
    if opening is None:
        return None
    board, opening_moves = opening
    all_moves = opening_moves.copy()
    game_tokens = {name: object() for name in engines}
    repetition_counts: dict[int, int] = {}
    replay = chess.Board()
    repetition_counts[chess.polyglot.zobrist_hash(replay)] = 1
    for move_uci in opening_moves:
        replay.push_uci(move_uci)
        key = chess.polyglot.zobrist_hash(replay)
        repetition_counts[key] = repetition_counts.get(key, 0) + 1
    if replay.fen() != board.fen():
        raise RuntimeError("opening replay mismatch")

    candidates: list[dict[str, Any]] = []
    searches: list[SearchRecord] = []
    win_side = 0
    win_streak = 0
    draw_streak = 0
    termination = "max_plies"
    result = 0.5

    while board.ply() < cfg["max_plies"]:
        outcome = board.outcome(claim_draw=True)
        if outcome is not None:
            termination = outcome.termination.name.lower()
            result = 0.5 if outcome.winner is None else float(outcome.winner == chess.WHITE)
            break
        repetition_count = repetition_counts.get(chess.polyglot.zobrist_hash(board), 1)
        key = state_key(board, repetition_count)
        candidates.append({
            "ply": board.ply(),
            "fen": board.fen(en_passant="fen"),
            "state_key": key,
            "repetition_count": repetition_count,
            "legal_moves": board.legal_moves.count(),
            "in_check": board.is_check(),
        })

        generator = generator_for(cfg["stratum"], game_index, board.turn)
        engine = engines[generator]
        use_exploration = (
            cfg["stratum"] == "exploratory"
            and board.ply() < len(opening_moves) + cfg["explore_plies"]
        )
        if use_exploration:
            move, info, rank, best = choose_multipv(
                engine, board, cfg["move_nodes"], cfg["multipv"],
                cfg["temperature_cp"], rng, game_tokens[generator],
            )
        else:
            move, info = analyse_one(
                engine, board, cfg["move_nodes"], game_tokens[generator]
            )
            rank = 0
            best = move
        cp, mate = score_info(info, board.turn)
        searches.append(SearchRecord(
            ply=board.ply(), generator=generator, score_cp=cp,
            mate_plies=mate, best_move=best.uci(), chosen_move=move.uci(),
            chosen_rank=rank, depth=int(info.get("depth", 0)),
            seldepth=int(info.get("seldepth", 0)), nodes=int(info.get("nodes", 0)),
        ))

        white_cp = cp if board.turn == chess.WHITE else -cp
        side = 1 if white_cp >= cfg["win_cp"] else (-1 if white_cp <= -cfg["win_cp"] else 0)
        if side:
            win_streak = win_streak + 1 if side == win_side else 1
            win_side = side
        else:
            win_side = win_streak = 0
        draw_streak = draw_streak + 1 if (
            board.ply() >= cfg["draw_after_ply"] and abs(white_cp) <= cfg["draw_cp"]
        ) else 0
        all_moves.append(move.uci())
        board.push(move)
        history_key = chess.polyglot.zobrist_hash(board)
        repetition_counts[history_key] = repetition_counts.get(history_key, 0) + 1
        if win_streak >= cfg["win_plies"]:
            termination = "eval_adjudication"
            result = 1.0 if win_side == 1 else 0.0
            break
        if draw_streak >= cfg["draw_plies"]:
            termination = "draw_adjudication"
            result = 0.5
            break

    selected = evenly_spaced(candidates, cfg["positions_per_game"])
    unique = []
    for sample in selected:
        if sample["state_key"] not in seen_states:
            seen_states.add(sample["state_key"])
            unique.append(sample)
    if not unique:
        return None
    return {
        "schema": SCHEMA,
        "task_index": task_index,
        "game_index": game_index,
        "stratum": cfg["stratum"],
        "start_fen": chess.STARTING_FEN,
        "opening_plies": len(opening_moves),
        "moves": all_moves,
        "searches": [asdict(record) for record in searches],
        "samples": unique,
        "result_white": result,
        "termination": termination,
    }


def validate_games(path: Path, expected_positions: int, expected_games: int) -> None:
    positions = games = 0
    with gzip.open(path, "rt", encoding="utf-8") as handle:
        for line in handle:
            game = json.loads(line)
            if game.get("schema") != SCHEMA or not game.get("samples"):
                raise RuntimeError("invalid native game record")
            board = chess.Board(game["start_fen"])
            samples = {int(sample["ply"]): sample for sample in game["samples"]}
            repetitions = {chess.polyglot.zobrist_hash(board): 1}
            for move_uci in game["moves"]:
                if board.ply() in samples:
                    sample = samples[board.ply()]
                    if board.fen(en_passant="fen") != sample["fen"]:
                        raise RuntimeError("sample FEN does not match replay")
                    repetition_count = repetitions[chess.polyglot.zobrist_hash(board)]
                    if repetition_count != sample["repetition_count"]:
                        raise RuntimeError("sample repetition count does not match replay")
                    if state_key(board, repetition_count) != sample["state_key"]:
                        raise RuntimeError("sample state key does not match replay")
                move = chess.Move.from_uci(move_uci)
                if move not in board.legal_moves:
                    raise RuntimeError("illegal stored game move")
                board.push(move)
                key = chess.polyglot.zobrist_hash(board)
                repetitions[key] = repetitions.get(key, 0) + 1
            positions += len(samples)
            games += 1
    if (positions, games) != (expected_positions, expected_games):
        raise RuntimeError(
            f"native output count mismatch: {(positions, games)} != {(expected_positions, expected_games)}"
        )


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--reckless", type=Path, required=True)
    parser.add_argument("--stockfish", type=Path, required=True)
    parser.add_argument("--output", required=True)
    parser.add_argument("--corpus-id", required=True)
    parser.add_argument("--stratum", choices=(
        "reckless_self", "sf18_self", "cross", "exploratory",
    ), required=True)
    parser.add_argument("--task-index", type=int)
    parser.add_argument("--seed", type=int, default=20260715)
    parser.add_argument("--target-positions", type=int, default=100_000)
    parser.add_argument("--positions-per-game", type=int, default=40)
    parser.add_argument("--max-games", type=int, default=10_000)
    parser.add_argument("--move-nodes", type=int, default=5_000)
    parser.add_argument("--gate-nodes", type=int, default=500)
    parser.add_argument("--gate-cp", type=int, default=500)
    parser.add_argument("--opening-min-plies", type=int, default=8)
    parser.add_argument("--opening-max-plies", type=int, default=16)
    parser.add_argument("--opening-retries", type=int, default=50)
    parser.add_argument("--max-plies", type=int, default=180)
    parser.add_argument("--win-cp", type=int, default=1_500)
    parser.add_argument("--win-plies", type=int, default=8)
    parser.add_argument("--draw-cp", type=int, default=20)
    parser.add_argument("--draw-after-ply", type=int, default=80)
    parser.add_argument("--draw-plies", type=int, default=12)
    parser.add_argument("--explore-plies", type=int, default=20)
    parser.add_argument("--multipv", type=int, default=4)
    parser.add_argument("--temperature-cp", type=float, default=80.0)
    parser.add_argument("--hash-mb", type=int, default=16)
    parser.add_argument("--project", default="archon-502500")
    args = parser.parse_args()
    task_index = args.task_index
    if task_index is None:
        task_index = int(os.environ.get("BATCH_TASK_INDEX", "-1"))
    if task_index < 0:
        parser.error("--task-index or BATCH_TASK_INDEX is required")
    numeric = (
        args.target_positions, args.positions_per_game, args.max_games,
        args.move_nodes, args.gate_nodes, args.opening_retries, args.max_plies,
        args.win_plies, args.draw_plies, args.multipv, args.hash_mb,
        args.temperature_cp,
    )
    if any(value <= 0 for value in numeric):
        parser.error("budgets, counts, and temperature must be positive")
    if not 0 <= args.opening_min_plies <= args.opening_max_plies < args.max_plies:
        parser.error("invalid opening or maximum ply range")
    for engine in (args.reckless, args.stockfish):
        if not engine.is_file() or not os.access(engine, os.X_OK):
            parser.error(f"engine is not executable: {engine}")

    started = time.monotonic()
    identities = {
        "reckless": engine_identity(args.reckless),
        "stockfish18": engine_identity(args.stockfish),
    }
    cfg = {
        key.replace("-", "_"): value
        for key, value in vars(args).items()
        if key not in {"reckless", "stockfish", "output", "project", "task_index"}
    }
    # The configuration hash identifies one reproducible campaign contract, not
    # one task.  Task identity and its derived seed are recorded separately.
    # Including the generator source hash ensures code changes cannot masquerade
    # as the same data configuration even when the JSON schema is unchanged.
    cfg.update({
        "schema": SCHEMA,
        "generator_sha256": sha256(Path(__file__)),
        "engines": identities,
    })
    config_hash = canonical_hash(cfg)
    task_seed = (args.seed ^ (task_index * GAMMA)) & MASK64

    expected_resume_metadata = {
        "schema": SCHEMA,
        "config_hash": config_hash,
        "corpus_id": args.corpus_id,
        "stratum": args.stratum,
        "task_index": str(task_index),
        "positions": str(args.target_positions),
    }
    if args.output.startswith("gs://"):
        client = storage_client(args.project)
        bucket_name, prefix = parse_gs(args.output)
        name = f"{prefix.rstrip('/')}/tasks/{task_index:06d}.jsonl.gz"
        existing_blob = client.bucket(bucket_name).blob(name)
        if existing_blob.exists():
            existing_blob.reload()
            metadata = existing_blob.metadata or {}
            if any(metadata.get(key) != value for key, value in expected_resume_metadata.items()):
                raise RuntimeError(f"existing output metadata differs: gs://{bucket_name}/{name}")
            with tempfile.TemporaryDirectory(prefix="archon-native-resume-") as temporary:
                existing_path = Path(temporary) / f"{task_index:06d}.jsonl.gz"
                existing_blob.download_to_filename(existing_path, checksum="crc32c")
                if sha256(existing_path) != metadata.get("sha256"):
                    raise RuntimeError("existing native output hash mismatch")
                validate_games(existing_path, args.target_positions, int(metadata["games"]))
            seconds = time.monotonic() - started
            print(json.dumps({
                "status": "validated_existing",
                "output": f"gs://{bucket_name}/{name}",
                "positions": args.target_positions,
                "games": int(metadata["games"]),
                "seconds": seconds,
                "positions_per_second": args.target_positions / seconds,
                "config_hash": config_hash,
                "sha256": metadata["sha256"],
            }, sort_keys=True))
            return
    else:
        existing_path = Path(args.output) / f"{task_index:06d}.jsonl.gz"
        existing_metadata_path = existing_path.with_suffix(existing_path.suffix + ".metadata.json")
        if existing_path.exists() and existing_metadata_path.is_file():
            metadata = json.loads(existing_metadata_path.read_text(encoding="utf-8"))
            if any(metadata.get(key) != value for key, value in expected_resume_metadata.items()):
                raise RuntimeError(f"existing local output metadata differs: {existing_path}")
            if sha256(existing_path) != metadata.get("sha256"):
                raise RuntimeError("existing local native output hash mismatch")
            validate_games(existing_path, args.target_positions, int(metadata["games"]))
            seconds = time.monotonic() - started
            print(json.dumps({
                "status": "validated_existing", "output": str(existing_path),
                "positions": args.target_positions, "games": int(metadata["games"]),
                "seconds": seconds, "positions_per_second": args.target_positions / seconds,
                "config_hash": config_hash, "sha256": metadata["sha256"],
            }, sort_keys=True))
            return

    engines = {
        "reckless": chess.engine.SimpleEngine.popen_uci(str(args.reckless.resolve()), timeout=30),
        "stockfish18": chess.engine.SimpleEngine.popen_uci(str(args.stockfish.resolve()), timeout=30),
    }
    for engine in engines.values():
        configure(engine, args.hash_mb)
    records: list[dict[str, Any]] = []
    seen_states: set[str] = set()
    positions = 0
    try:
        for game_index in range(args.max_games):
            game_seed = (task_seed ^ ((game_index + 1) * GAMMA)) & MASK64
            game_rng = random.Random(game_seed)
            game = play_game(cfg, task_index, game_index, engines, game_rng, seen_states)
            if game is None:
                continue
            game["game_seed"] = game_seed
            game["config_hash"] = config_hash
            remaining = args.target_positions - positions
            if len(game["samples"]) > remaining:
                game["samples"] = game["samples"][:remaining]
            records.append(game)
            positions += len(game["samples"])
            if positions >= args.target_positions:
                break
    finally:
        for engine in engines.values():
            try:
                engine.quit()
            except (chess.engine.EngineError, OSError):
                pass
    if positions < args.target_positions:
        raise RuntimeError(
            f"generated only {positions}/{args.target_positions} unique positions in {args.max_games} games"
        )

    with tempfile.TemporaryDirectory(prefix="archon-native-games-") as temporary:
        root = Path(temporary)
        data_path = root / f"{task_index:06d}.jsonl.gz"
        with data_path.open("wb") as raw:
            with gzip.GzipFile(fileobj=raw, mode="wb", filename="", mtime=0, compresslevel=6) as compressed:
                for record in records:
                    compressed.write(
                        (json.dumps(record, sort_keys=True, separators=(",", ":")) + "\n").encode()
                    )
        validate_games(data_path, positions, len(records))
        digest = sha256(data_path)
        metadata = {
            "sha256": digest,
            "schema": SCHEMA,
            "config_hash": config_hash,
            "corpus_id": args.corpus_id,
            "stratum": args.stratum,
            "task_index": str(task_index),
            "positions": str(positions),
            "games": str(len(records)),
        }
        if args.output.startswith("gs://"):
            client = storage_client(args.project)
            bucket_name, prefix = parse_gs(args.output)
            name = f"{prefix.rstrip('/')}/tasks/{task_index:06d}.jsonl.gz"
            blob = client.bucket(bucket_name).blob(name)
            if blob.exists():
                blob.reload()
                if blob.metadata != metadata or blob.size != data_path.stat().st_size:
                    raise RuntimeError(f"existing output differs: gs://{bucket_name}/{name}")
                status = "validated_existing"
            else:
                blob.metadata = metadata
                blob.upload_from_filename(data_path, if_generation_match=0, checksum="crc32c")
                blob.reload()
                status = "uploaded"
            output = f"gs://{bucket_name}/{name}"
        else:
            output_root = Path(args.output)
            output_root.mkdir(parents=True, exist_ok=True)
            output_path = output_root / data_path.name
            metadata_path = output_path.with_suffix(output_path.suffix + ".metadata.json")
            if output_path.exists():
                if sha256(output_path) != digest or not metadata_path.is_file():
                    raise RuntimeError(f"existing local output differs: {output_path}")
                if json.loads(metadata_path.read_text()) != metadata:
                    raise RuntimeError(f"existing local metadata differs: {metadata_path}")
                status = "validated_existing"
            else:
                with tempfile.NamedTemporaryFile(
                    prefix=output_path.name + ".", suffix=".tmp",
                    dir=output_root, delete=False,
                ) as temporary_output:
                    temporary_output_path = Path(temporary_output.name)
                try:
                    shutil.copyfile(data_path, temporary_output_path)
                    if sha256(temporary_output_path) != digest:
                        raise RuntimeError("copied local native output hash mismatch")
                    os.replace(temporary_output_path, output_path)
                finally:
                    temporary_output_path.unlink(missing_ok=True)
                temporary_metadata = metadata_path.with_suffix(metadata_path.suffix + ".tmp")
                temporary_metadata.write_text(
                    json.dumps(metadata, indent=2, sort_keys=True) + "\n", encoding="utf-8"
                )
                os.replace(temporary_metadata, metadata_path)
                status = "written"
            output = str(output_path)
    seconds = time.monotonic() - started
    print(json.dumps({
        "status": status,
        "output": output,
        "positions": positions,
        "games": len(records),
        "seconds": seconds,
        "positions_per_second": positions / seconds,
        "config_hash": config_hash,
        "sha256": digest,
    }, sort_keys=True))


if __name__ == "__main__":
    main()
