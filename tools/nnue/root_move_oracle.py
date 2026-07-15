#!/usr/bin/env python3
"""Generate independent cold-arm root-move oracle observations.

For every legal move, this tool launches a fresh UCI engine process and runs a
forced arm at two fixed limits. It uses ``searchmoves <move>`` when supported;
for engines such as pinned Reckless that ignore ``searchmoves``, an explicit
mode searches the already-forced child and converts the score back to the root
side. Process-per-arm is intentionally expensive: it prevents TT, history, and
first-search state from leaking between moves or budgets. The result is a clean
bootstrap corpus for ARCHON-native best-move and pairwise targets, not a
high-throughput production labeler.

Input is either a text file containing complete six-field FENs, or a
deterministic subset of the existing side-normalized LC0 ``feats.npy`` corpus.
LC0 reconstruction preserves piece placement only and is explicitly marked
``piece_state_bootstrap_only`` in every record.

Output consists of an immutable manifest and atomically published JSON shards.
An identical rerun validates and skips completed shards; a changed engine,
source, limit, or selection schedule requires a new output directory.

Example:

    uv run --no-project --with numpy --with python-chess python \
      tools/nnue/root_move_oracle.py \
      --lc0-source data/lc0_nnue/policy_eval_ds_100m \
      --rows-per-side 8 --output data/root_oracle_smoke \
      --engine reckless_reference/Reckless/target/release/reckless \
      --teacher reckless-d445b8bf --limit-type depth --low 4 --high 6
"""

from __future__ import annotations

import argparse
from concurrent.futures import ProcessPoolExecutor
from dataclasses import dataclass, replace
import hashlib
import heapq
import json
import math
import os
from pathlib import Path
import tempfile
from typing import Any

import chess
import chess.engine
import numpy as np


SCHEMA = "archon.root-move-oracle.v1"
MATE_SCORE = 30_000
DEFAULT_SEED = 20260714
SELF_TEST_FEN = "1Qk1r1n1/p6r/4p3/P1pPBp1p/3q2pb/8/3PPPB1/4K1NR b K - 1 43"


def sha256(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as handle:
        for block in iter(lambda: handle.read(1024 * 1024), b""):
            digest.update(block)
    return digest.hexdigest()


def canonical_hash(value: Any) -> str:
    encoded = json.dumps(value, sort_keys=True, separators=(",", ":")).encode()
    return hashlib.sha256(encoded).hexdigest()


def atomic_json(path: Path, value: Any) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    with tempfile.NamedTemporaryFile(
        mode="w",
        encoding="utf-8",
        dir=path.parent,
        prefix=f".{path.name}.",
        delete=False,
    ) as handle:
        json.dump(value, handle, indent=2, sort_keys=True)
        handle.write("\n")
        temp = Path(handle.name)
    try:
        os.replace(temp, path)
    finally:
        temp.unlink(missing_ok=True)


def engine_identity(path: Path) -> dict[str, Any]:
    engine = chess.engine.SimpleEngine.popen_uci(str(path), timeout=30)
    try:
        identity = dict(engine.id)
        option_names = sorted(engine.options)
    finally:
        engine.quit()
    name = identity.get("name")
    if not name:
        raise RuntimeError(f"engine did not report an id name: {path}")
    return {
        "path": str(path),
        "sha256": sha256(path),
        "uci_name": name,
        "uci_author": identity.get("author"),
        "uci_options": option_names,
    }


def board_from_features(row: np.ndarray) -> chess.Board:
    board = chess.Board.empty()
    for raw in row:
        feature = int(raw)
        if feature < 0:
            break
        colour, rem = divmod(feature, 384)
        piece_type, square = divmod(rem, 64)
        if colour not in (0, 1) or piece_type >= 6:
            raise ValueError(f"invalid normalized feature {feature}")
        board.set_piece_at(square, chess.Piece(piece_type + 1, colour == 0))
    board.turn = chess.WHITE
    board.castling_rights = chess.BB_EMPTY
    board.ep_square = None
    board.halfmove_clock = 0
    board.fullmove_number = 1
    if not board.is_valid():
        raise ValueError(f"invalid reconstructed board: {board.fen()}")
    return board


def affine_parameters(rows: int, seed: int) -> tuple[int, int]:
    if rows <= 0:
        raise ValueError("source contains no rows")
    if rows == 1:
        return 1, 0
    rng = np.random.default_rng(seed)
    multiplier = int(rng.integers(1, rows))
    while math.gcd(multiplier, rows) != 1:
        multiplier = (multiplier + 1) % rows or 1
    return multiplier, int(rng.integers(0, rows))


@dataclass(frozen=True)
class PositionInput:
    ordinal: int
    source_id: str
    fen: str
    state_completeness: str
    source_detail: dict[str, Any]


@dataclass(frozen=True)
class OracleConfig:
    engine: str
    teacher: str
    teacher_sha256: str
    limit_type: str
    low: int
    high: int
    hash_mb: int
    timeout_seconds: float
    tie_cp: int
    arm_mode: str
    engine_options: tuple[tuple[str, str], ...]
    option_artifacts: tuple[tuple[str, str, str], ...]


def read_fens(path: Path, max_positions: int | None, seed: int) -> list[PositionInput]:
    # Keep the smallest deterministic hash keys. This gives a bounded sample
    # without loading a potentially 25M-line historical corpus into memory.
    selected: list[tuple[int, int, PositionInput]] = []
    with path.open(encoding="utf-8") as source:
        for line_number, raw in enumerate(source, 1):
            text = raw.strip()
            if not text or text.startswith("#"):
                continue
            if "|" in text:
                text = text.split("|", 1)[0].strip()
            fields = text.split()
            if len(fields) != 6:
                raise ValueError(
                    f"{path}:{line_number}: expected a complete six-field FEN, got {len(fields)} fields"
                )
            try:
                board = chess.Board(text)
            except ValueError as error:
                raise ValueError(f"{path}:{line_number}: invalid FEN: {error}") from error
            if not board.is_valid():
                raise ValueError(f"{path}:{line_number}: illegal board state: {text}")
            item = PositionInput(
                ordinal=-1,
                source_id=f"fen:{line_number}",
                fen=board.fen(),
                state_completeness="full_fen_no_repetition_history",
                source_detail={"line_number": line_number},
            )
            key = int.from_bytes(
                hashlib.blake2b(
                    f"{seed}:{line_number}:{item.fen}".encode(), digest_size=8
                ).digest(),
                "little",
            )
            if max_positions is None:
                selected.append((-key, -line_number, item))
            elif len(selected) < max_positions:
                heapq.heappush(selected, (-key, -line_number, item))
            elif (key, line_number) < (-selected[0][0], -selected[0][1]):
                heapq.heapreplace(selected, (-key, -line_number, item))
    if not selected:
        raise ValueError(f"no FEN positions found in {path}")
    ordered = sorted(((-neg_key, -neg_line, item) for neg_key, neg_line, item in selected))
    return [
        PositionInput(
            ordinal=ordinal,
            source_id=item.source_id,
            fen=item.fen,
            state_completeness=item.state_completeness,
            source_detail={**item.source_detail, "selection_hash64": key},
        )
        for ordinal, (key, _line, item) in enumerate(ordered)
    ]


def lc0_positions(
    root: Path, rows_per_side: int, seed: int
) -> tuple[list[PositionInput], dict[str, Any]]:
    by_side: dict[str, list[PositionInput]] = {}
    source_detail: dict[str, Any] = {"root": str(root), "sides": {}}
    for side_index, side in enumerate(("white", "black")):
        feats_path = root / side / "feats.npy"
        if not feats_path.is_file():
            raise ValueError(f"missing LC0 feature array: {feats_path}")
        features = np.load(feats_path, mmap_mode="r")
        available = len(features)
        count = min(rows_per_side, available)
        multiplier, offset = affine_parameters(available, seed + side_index)
        selected: list[PositionInput] = []
        for selection_ordinal in range(count):
            source_index = (selection_ordinal * multiplier + offset) % available
            board = board_from_features(features[source_index])
            selected.append(
                PositionInput(
                    ordinal=-1,
                    source_id=f"lc0:{side}:{source_index}",
                    fen=board.fen(),
                    state_completeness="piece_state_bootstrap_only",
                    source_detail={
                        "normalized_side": side,
                        "source_index": source_index,
                        "selection_ordinal": selection_ordinal,
                    },
                )
            )
        by_side[side] = selected
        stat = feats_path.stat()
        source_detail["sides"][side] = {
            "feats_path": str(feats_path),
            "available_rows": available,
            "dtype": str(features.dtype),
            "shape": list(features.shape),
            "file_bytes": stat.st_size,
            "permutation": {"multiplier": multiplier, "offset": offset},
            "selected_rows": count,
        }

    # Interleave normalized sides so every early shard covers both sources.
    positions: list[PositionInput] = []
    for selection_ordinal in range(max(map(len, by_side.values()))):
        for side in ("white", "black"):
            if selection_ordinal < len(by_side[side]):
                item = by_side[side][selection_ordinal]
                positions.append(
                    PositionInput(
                        ordinal=len(positions),
                        source_id=item.source_id,
                        fen=item.fen,
                        state_completeness=item.state_completeness,
                        source_detail=item.source_detail,
                    )
                )
    return positions, source_detail


def limit_for(config: OracleConfig, value: int) -> chess.engine.Limit:
    if config.limit_type == "depth":
        return chess.engine.Limit(depth=value)
    if config.limit_type == "nodes":
        return chess.engine.Limit(nodes=value)
    raise AssertionError(config.limit_type)


def bound_name(info: dict[str, Any]) -> str:
    lower = bool(info.get("lowerbound", False))
    upper = bool(info.get("upperbound", False))
    if lower and upper:
        raise RuntimeError("engine marked one result as both lowerbound and upperbound")
    if lower:
        return "lower"
    if upper:
        return "upper"
    return "exact"


def run_engine_arm(
    config: OracleConfig,
    board: chess.Board,
    limit_value: int,
    root_move: chess.Move | None,
) -> dict[str, Any]:
    # Fresh process = cold TT, cold histories, and no dependence on another arm.
    engine = chess.engine.SimpleEngine.popen_uci(
        config.engine,
        timeout=config.timeout_seconds,
    )
    try:
        configurable = engine.options
        if "Threads" not in configurable or "Hash" not in configurable:
            raise RuntimeError("engine must expose Threads and Hash UCI options")
        settings: dict[str, Any] = {"Threads": 1, "Hash": config.hash_mb}
        for name, value in config.engine_options:
            if name not in configurable:
                raise RuntimeError(f"engine does not expose requested option {name}")
            settings[name] = value
        engine.configure(settings)
        info = engine.analyse(
            board,
            limit_for(config, limit_value),
            root_moves=None if root_move is None else [root_move],
            game=object(),
            info=chess.engine.INFO_ALL,
        )
    finally:
        try:
            engine.quit()
        except (chess.engine.EngineTerminatedError, TimeoutError):
            pass

    return info


def format_arm_result(
    info: dict[str, Any],
    score_colour: chess.Color,
    move: chess.Move,
    requested_limit: int,
    engine_limit: int,
    execution_mode: str,
    prepend_move: bool,
) -> dict[str, Any]:
    if "score" not in info:
        raise RuntimeError(f"no score for forced move {move.uci()}")
    pov = info["score"].pov(score_colour)
    score_cp = pov.score(mate_score=MATE_SCORE)
    if score_cp is None:
        raise RuntimeError(f"unusable score for forced move {move.uci()}")
    child_pv = list(info.get("pv", []))
    pv = ([move] + child_pv) if prepend_move else child_pv
    if not pv or pv[0] != move:
        rendered = [candidate.uci() for candidate in pv]
        raise RuntimeError(
            f"engine did not honor searchmoves {move.uci()}; pv={rendered}"
        )
    return {
        "requested_root_limit": requested_limit,
        "engine_limit": engine_limit,
        "execution_mode": execution_mode,
        "score_cp": int(score_cp),
        "mate_plies": pov.mate(),
        "bound": bound_name(info),
        "depth": int(info.get("depth", 0)),
        "seldepth": int(info.get("seldepth", 0)),
        "nodes": int(info.get("nodes", 0)),
        "time_ms": int(round(float(info.get("time", 0.0)) * 1000.0)),
        "nps": int(info.get("nps", 0)),
        "pv": [candidate.uci() for candidate in pv],
        "best_move": pv[0].uci(),
    }


def analyse_cold_arm(
    config: OracleConfig,
    board: chess.Board,
    move: chess.Move,
    limit_value: int,
) -> dict[str, Any]:
    if config.arm_mode in ("auto", "searchmoves"):
        info = run_engine_arm(config, board, limit_value, move)
        pv = list(info.get("pv", []))
        if pv and pv[0] == move:
            return format_arm_result(
                info,
                board.turn,
                move,
                limit_value,
                limit_value,
                "uci_searchmoves",
                prepend_move=False,
            )
        if config.arm_mode == "searchmoves":
            rendered = [candidate.uci() for candidate in pv]
            raise RuntimeError(
                f"engine ignored searchmoves {move.uci()} for {board.fen()}; pv={rendered}"
            )

    # Some UCI engines (including pinned Reckless d445b8bf) do not implement
    # `searchmoves`.  A fresh search of the forced child is the causal fallback:
    # the root move is made outside the engine, its score is converted back to
    # the original side's POV, and the move is prepended to the child PV.  For
    # depth limits, subtract the forced root ply so requested root depths retain
    # their intended meaning.  The clamped depth-1 case is explicitly recorded.
    child = board.copy(stack=True)
    original_turn = board.turn
    child.push(move)
    engine_limit = (
        max(1, limit_value - 1) if config.limit_type == "depth" else limit_value
    )
    info = run_engine_arm(config, child, engine_limit, None)
    return format_arm_result(
        info,
        original_turn,
        move,
        limit_value,
        engine_limit,
        "forced_child_position",
        prepend_move=True,
    )


def ranking(arms: list[dict[str, Any]], budget: str) -> list[dict[str, Any]]:
    return sorted(
        ({"move": arm["move"], "score_cp": arm[budget]["score_cp"]} for arm in arms),
        key=lambda item: (-item["score_cp"], item["move"]),
    )


def process_position(position: PositionInput, config: OracleConfig) -> dict[str, Any]:
    board = chess.Board(position.fen)
    legal = sorted(board.legal_moves, key=lambda move: move.uci())
    if not legal:
        return {
            "schema": SCHEMA,
            "source_id": position.source_id,
            "source_ordinal": position.ordinal,
            "source_detail": position.source_detail,
            "fen": board.fen(),
            "state_completeness": position.state_completeness,
            "teacher_sha256": config.teacher_sha256,
            "teacher": config.teacher,
            "legal_moves": [],
            "arms": [],
            "acceptance": {"accepted": False, "status": "terminal_no_legal_moves"},
        }

    arms: list[dict[str, Any]] = []
    for move in legal:
        low = analyse_cold_arm(config, board, move, config.low)
        high = analyse_cold_arm(config, board, move, config.high)
        arms.append(
            {
                "move": move.uci(),
                "low": low,
                "high": high,
                "stability": {
                    "score_delta_cp": high["score_cp"] - low["score_cp"],
                    "bound_stable": low["bound"] == high["bound"],
                    "pv_first_stable": low["best_move"]
                    == high["best_move"]
                    == move.uci(),
                },
            }
        )

    # `searchmoves` has no UCI capability advertisement. An engine that ignores
    # it can still naturally choose the requested move, making that individual
    # arm look supported. Never allow the resulting root-search/child-search
    # depth semantics to mix within one position: one demonstrated fallback
    # forces every arm through the causal child-position path.
    if config.arm_mode == "auto" and any(
        arm[budget]["execution_mode"] == "forced_child_position"
        for arm in arms
        for budget in ("low", "high")
    ):
        child_config = replace(config, arm_mode="child")
        for arm in arms:
            move = chess.Move.from_uci(arm["move"])
            for budget, limit in (("low", config.low), ("high", config.high)):
                if arm[budget]["execution_mode"] != "forced_child_position":
                    arm[budget] = analyse_cold_arm(child_config, board, move, limit)
            arm["stability"] = {
                "score_delta_cp": arm["high"]["score_cp"] - arm["low"]["score_cp"],
                "bound_stable": arm["low"]["bound"] == arm["high"]["bound"],
                "pv_first_stable": arm["low"]["best_move"]
                == arm["high"]["best_move"]
                == move.uci(),
            }

    low_rank = ranking(arms, "low")
    high_rank = ranking(arms, "high")
    low_best = low_rank[0]["move"]
    high_best = high_rank[0]["move"]
    high_top = high_rank[0]["score_cp"]
    high_ties = [
        item["move"]
        for item in high_rank
        if high_top - item["score_cp"] <= config.tie_cp
    ]
    low_top = low_rank[0]["score_cp"]
    low_ties = [
        item["move"] for item in low_rank if low_top - item["score_cp"] <= config.tie_cp
    ]
    margin = None if len(high_rank) == 1 else high_top - high_rank[1]["score_cp"]
    best_stable = low_best == high_best
    unique = len(high_ties) == 1
    accepted = best_stable and unique
    if len(high_rank) == 1:
        status = "accepted_stable_singleton"
        accepted = True
    elif not best_stable:
        status = "rejected_unstable_best"
    elif not unique:
        status = "rejected_ambiguous_tie"
    else:
        status = "accepted_stable_unique"

    return {
        "schema": SCHEMA,
        "source_id": position.source_id,
        "source_ordinal": position.ordinal,
        "source_detail": position.source_detail,
        "fen": board.fen(),
        "state_completeness": position.state_completeness,
        "teacher_sha256": config.teacher_sha256,
        "teacher": config.teacher,
        "score_perspective": "side_to_move",
        "mate_score_cp": MATE_SCORE,
        "legal_moves": [move.uci() for move in legal],
        "arms": arms,
        "summary": {
            "best_move_low": low_best,
            "best_move_high": high_best,
            "best_stable": best_stable,
            "low_tie_moves": low_ties,
            "high_tie_moves": high_ties,
            "tie_tolerance_cp": config.tie_cp,
            "high_best_second_margin_cp": margin,
        },
        "acceptance": {"accepted": accepted, "status": status},
    }


def expected_ids(positions: list[PositionInput]) -> list[str]:
    return [position.source_id for position in positions]


def validate_shard(
    path: Path,
    config_hash: str,
    teacher: str,
    teacher_sha256: str,
    positions: list[PositionInput],
) -> None:
    try:
        shard = json.loads(path.read_text(encoding="utf-8"))
    except (OSError, json.JSONDecodeError) as error:
        raise RuntimeError(f"invalid completed shard {path}: {error}") from error
    if shard.get("schema") != SCHEMA or shard.get("config_hash") != config_hash:
        raise RuntimeError(f"{path}: schema/config mismatch")
    if shard.get("teacher") != {"label": teacher, "sha256": teacher_sha256}:
        raise RuntimeError(f"{path}: teacher identity mismatch")
    if shard.get("teacher_sha256") != teacher_sha256:
        raise RuntimeError(f"{path}: teacher mismatch")
    records = shard.get("records")
    if not isinstance(records, list) or [
        row.get("source_id") for row in records
    ] != expected_ids(positions):
        raise RuntimeError(f"{path}: source schedule mismatch")
    for row in records:
        if (
            row.get("schema") != SCHEMA
            or row.get("teacher") != teacher
            or row.get("teacher_sha256") != teacher_sha256
        ):
            raise RuntimeError(f"{path}: incomplete record provenance")


def build_shard(
    shard_number: int,
    positions: list[PositionInput],
    config: OracleConfig,
    config_hash: str,
) -> tuple[int, dict[str, Any]]:
    if sha256(Path(config.engine)) != config.teacher_sha256:
        raise RuntimeError("engine binary changed after manifest identity was frozen")
    for name, path, expected_sha in config.option_artifacts:
        if sha256(Path(path)) != expected_sha:
            raise RuntimeError(
                f"engine option artifact changed after manifest freeze: {name}={path}"
            )
    records = [process_position(position, config) for position in positions]
    return shard_number, {
        "schema": SCHEMA,
        "config_hash": config_hash,
        "teacher": {"label": config.teacher, "sha256": config.teacher_sha256},
        "teacher_sha256": config.teacher_sha256,
        "shard_number": shard_number,
        "source_ids": expected_ids(positions),
        "records": records,
    }


def run_generation(
    output: Path,
    positions: list[PositionInput],
    manifest_config: dict[str, Any],
    oracle_config: OracleConfig,
    shard_rows: int,
    workers: int,
) -> None:
    config_hash = canonical_hash(manifest_config)
    manifest_path = output / "manifest.json"
    manifest = {
        "schema": SCHEMA,
        "config_hash": config_hash,
        "config": manifest_config,
        "total_positions": len(positions),
        "total_shards": math.ceil(len(positions) / shard_rows),
    }
    if manifest_path.exists():
        existing = json.loads(manifest_path.read_text(encoding="utf-8"))
        if (
            existing.get("config_hash") != config_hash
            or existing.get("config") != manifest_config
        ):
            raise RuntimeError(
                "existing manifest configuration differs; use a new output path"
            )
    else:
        atomic_json(manifest_path, manifest)

    shards = [
        positions[start : start + shard_rows]
        for start in range(0, len(positions), shard_rows)
    ]
    pending: list[tuple[int, list[PositionInput]]] = []
    for shard_number, shard_positions in enumerate(shards):
        path = output / "shards" / f"{shard_number:08d}.json"
        if path.exists():
            validate_shard(
                path,
                config_hash,
                oracle_config.teacher,
                oracle_config.teacher_sha256,
                shard_positions,
            )
            print(f"validated {path} ({len(shard_positions)} positions)", flush=True)
        else:
            pending.append((shard_number, shard_positions))

    def publish(result: tuple[int, dict[str, Any]]) -> None:
        shard_number, shard = result
        path = output / "shards" / f"{shard_number:08d}.json"
        atomic_json(path, shard)
        validate_shard(
            path,
            config_hash,
            oracle_config.teacher,
            oracle_config.teacher_sha256,
            shards[shard_number],
        )
        print(f"wrote {path} ({len(shards[shard_number])} positions)", flush=True)

    if workers == 1:
        for shard_number, shard_positions in pending:
            publish(
                build_shard(shard_number, shard_positions, oracle_config, config_hash)
            )
    else:
        with ProcessPoolExecutor(max_workers=workers) as executor:
            futures = [
                executor.submit(build_shard, number, rows, oracle_config, config_hash)
                for number, rows in pending
            ]
            # Submission order is shard order, so publication/logging is deterministic.
            for future in futures:
                publish(future.result())

    manifest["completed_shards"] = len(shards)
    manifest["completed_positions"] = len(positions)
    atomic_json(manifest_path, manifest)


def self_test(engine_path: Path, teacher: str, hash_mb: int, timeout: float) -> None:
    identity = engine_identity(engine_path)
    config = OracleConfig(
        engine=str(engine_path),
        teacher=teacher,
        teacher_sha256=identity["sha256"],
        limit_type="depth",
        low=2,
        high=3,
        hash_mb=hash_mb,
        timeout_seconds=timeout,
        tie_cp=0,
        arm_mode="auto",
        engine_options=(),
        option_artifacts=(),
    )
    position = PositionInput(
        ordinal=0,
        source_id="self-test:forced-single-move",
        fen=SELF_TEST_FEN,
        state_completeness="full_fen_no_repetition_history",
        source_detail={"fixture": "one legal move"},
    )
    first = process_position(position, config)
    second = process_position(position, config)
    if first != second:
        raise RuntimeError(
            "cold-arm self-test was not bit-deterministic across repeated runs"
        )
    if first["legal_moves"] != ["c8d7"]:
        raise RuntimeError(f"self-test fixture changed: {first['legal_moves']}")
    if first["acceptance"] != {"accepted": True, "status": "accepted_stable_singleton"}:
        raise RuntimeError(f"unexpected self-test acceptance: {first['acceptance']}")
    print(
        json.dumps(
            {
                "self_test": "passed",
                "engine": identity,
                "source_id": first["source_id"],
                "move": first["legal_moves"][0],
                "low": first["arms"][0]["low"],
                "high": first["arms"][0]["high"],
            },
            indent=2,
            sort_keys=True,
        )
    )


def positive(parser: argparse.ArgumentParser, name: str, value: int) -> None:
    if value <= 0:
        parser.error(f"{name} must be positive")


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    source = parser.add_mutually_exclusive_group()
    source.add_argument("--fen-file", type=Path)
    source.add_argument("--lc0-source", type=Path)
    parser.add_argument("--output", type=Path)
    parser.add_argument("--engine", type=Path, required=True)
    parser.add_argument("--teacher", required=True)
    parser.add_argument("--limit-type", choices=("depth", "nodes"), default="depth")
    parser.add_argument("--low", type=int, default=6)
    parser.add_argument("--high", type=int, default=8)
    parser.add_argument("--hash-mb", type=int, default=16)
    parser.add_argument("--timeout-seconds", type=float, default=120.0)
    parser.add_argument("--tie-cp", type=int, default=0)
    parser.add_argument(
        "--arm-mode",
        choices=("auto", "searchmoves", "child"),
        default="auto",
        help="force UCI searchmoves, force child-position arms, or fall back automatically",
    )
    parser.add_argument("--rows-per-side", type=int, default=8)
    parser.add_argument(
        "--max-positions",
        type=int,
        help="deterministic bounded sample size for --fen-file (default: all)",
    )
    parser.add_argument("--seed", type=int, default=DEFAULT_SEED)
    parser.add_argument("--shard-rows", type=int, default=4)
    parser.add_argument("--workers", type=int, default=1)
    parser.add_argument("--self-test", action="store_true")
    parser.add_argument(
        "--engine-option",
        action="append",
        default=[],
        metavar="NAME=VALUE",
        help="additional UCI option, recorded in provenance (repeatable)",
    )
    args = parser.parse_args()

    engine = args.engine.resolve()
    if not engine.is_file() or not os.access(engine, os.X_OK):
        parser.error(f"engine is not executable: {engine}")
    if args.self_test:
        self_test(engine, args.teacher, args.hash_mb, args.timeout_seconds)
        return
    if args.fen_file is None and args.lc0_source is None:
        parser.error("one of --fen-file or --lc0-source is required")
    if args.output is None:
        parser.error("--output is required outside --self-test")
    for name in ("low", "high", "hash-mb", "rows-per-side", "shard-rows", "workers"):
        positive(parser, name, getattr(args, name.replace("-", "_")))
    if args.high <= args.low:
        parser.error("--high must be greater than --low")
    if args.timeout_seconds <= 0:
        parser.error("--timeout-seconds must be positive")
    if args.max_positions is not None and args.max_positions <= 0:
        parser.error("--max-positions must be positive")
    if args.tie_cp < 0:
        parser.error("--tie-cp must be non-negative")
    engine_options = []
    for item in args.engine_option:
        if "=" not in item:
            parser.error(f"engine-option must be NAME=VALUE: {item}")
        name, value = item.split("=", 1)
        if not name or not value:
            parser.error(f"engine-option must have nonempty name/value: {item}")
        engine_options.append((name, value))
    option_artifacts = tuple(
        (name, str(Path(value).resolve()), sha256(Path(value).resolve()))
        for name, value in engine_options
        if Path(value).is_file()
    )

    identity = engine_identity(engine)
    if args.fen_file is not None:
        fen_file = args.fen_file.resolve()
        positions = read_fens(fen_file, args.max_positions, args.seed)
        source_config = {
            "kind": "full_fen_or_bullet_text_file",
            "path": str(fen_file),
            "sha256": sha256(fen_file),
            "max_positions": args.max_positions,
            "selection": "smallest blake2b-64(seed:line_number:canonical_fen)",
            "seed": args.seed,
            "state_completeness": "full_fen_no_repetition_history",
        }
    else:
        lc0_source = args.lc0_source.resolve()
        positions, detail = lc0_positions(lc0_source, args.rows_per_side, args.seed)
        source_config = {
            "kind": "lc0_normalized_features",
            "detail": detail,
            "seed": args.seed,
            "rows_per_side": args.rows_per_side,
            "state_completeness": "piece_state_bootstrap_only",
            "reconstruction": {
                "side_to_move": "white (source normalized)",
                "castling": "unavailable; forced none",
                "en_passant": "unavailable; forced none",
                "rule50": "unavailable; forced zero",
                "repetition_history": "unavailable",
            },
        }

    oracle_config = OracleConfig(
        engine=str(engine),
        teacher=args.teacher,
        teacher_sha256=identity["sha256"],
        limit_type=args.limit_type,
        low=args.low,
        high=args.high,
        hash_mb=args.hash_mb,
        timeout_seconds=args.timeout_seconds,
        tie_cp=args.tie_cp,
        arm_mode=args.arm_mode,
        engine_options=tuple(engine_options),
        option_artifacts=option_artifacts,
    )
    manifest_config = {
        "schema": SCHEMA,
        "generator": {
            "path": str(Path(__file__).resolve()),
            "sha256": sha256(Path(__file__).resolve()),
        },
        "teacher": args.teacher,
        "engine": identity,
        "search": {
            "threads": 1,
            "hash_mb": args.hash_mb,
            "limit_type": args.limit_type,
            "low": args.low,
            "high": args.high,
            "cold_mode": "fresh_process_per_move_per_budget",
            "arm_mode": args.arm_mode,
            "score_perspective": "side_to_move",
            "mate_score_cp": MATE_SCORE,
            "tie_cp": args.tie_cp,
            "engine_options": dict(engine_options),
            "engine_option_artifacts": [
                {"name": name, "path": path, "sha256": digest}
                for name, path, digest in option_artifacts
            ],
        },
        "source": source_config,
        "selection_source_ids_sha256": canonical_hash(expected_ids(positions)),
        "shard_rows": args.shard_rows,
    }
    run_generation(
        args.output.resolve(),
        positions,
        manifest_config,
        oracle_config,
        args.shard_rows,
        args.workers,
    )


if __name__ == "__main__":
    main()
