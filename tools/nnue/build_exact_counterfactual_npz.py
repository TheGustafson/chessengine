#!/usr/bin/env python3
"""Build grouped tensors from exact Archon intervention counterfactuals."""

from __future__ import annotations

import argparse
import json
from pathlib import Path

import numpy as np

from build_search_trace_npz import context


SCHEMA = "archon.search-counterfactual-exact-v1"


def board_features(fen: str) -> list[int]:
    """Mover-normalized 12x64 relative-color piece-square occupancy."""
    placement, side = fen.split()[:2]
    result = [0] * (12 * 64)
    ranks = placement.split("/")
    pieces = {piece: index for index, piece in enumerate("pnbrqk")}
    for fen_rank, text in enumerate(ranks):
        file = 0
        for symbol in text:
            if symbol.isdigit():
                file += int(symbol)
                continue
            absolute_color = 0 if symbol.isupper() else 1
            relative_color = absolute_color if side == "w" else 1 - absolute_color
            square = (7 - fen_rank) * 8 + file
            if side == "b":
                square ^= 56
            result[(relative_color * 6 + pieces[symbol.lower()]) * 64 + square] = 1
            file += 1
    return result


def one_hot(index: int, size: int) -> list[int]:
    result = [0] * size
    if 0 <= index < size:
        result[index] = 1
    return result


def structured_move_action(observation: dict, action: dict) -> list[int]:
    """Categorical move/action encoding for factorized consequence heads."""
    raw = int(observation["raw_move"])
    source = raw & 63
    destination = (raw >> 6) & 63
    if int(observation["side_to_move"]) == 1:
        source ^= 56
        destination ^= 56
    return (
        one_hot(source, 64)
        + one_hot(destination, 64)
        + one_hot(int(observation["moving_piece"]), 6)
        + one_hot(int(observation["captured_piece"]), 7)
        + one_hot(int(observation["promotion_piece"]), 7)
        + one_hot((raw >> 14) & 3, 4)
        + one_hot(min(int(action["reduction"]), 8), 9)
    )


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("dataset", type=Path)
    parser.add_argument("output", type=Path)
    args = parser.parse_args()
    manifest = json.loads((args.dataset / "manifest.json").read_text(encoding="utf-8"))
    config_hash = manifest["config_hash"]
    latent = []
    features = []
    boards = []
    structured_moves = []
    roots = []
    wrong_window = []
    scout_wrong_window = []
    root_best_changed = []
    root_score_changed = []
    local_research = []
    log_root_nodes = []
    log_root_node_fraction = []
    signed_log_root_node_delta = []
    log_local_nodes = []
    log_absolute_root_score_error = []
    rejected = 0
    for path in sorted((args.dataset / "records").glob("*.json")):
        record = json.loads(path.read_text(encoding="utf-8"))
        if record.get("schema") != SCHEMA or record.get("config_hash") != config_hash:
            raise ValueError(f"{path}: schema/config mismatch")
        if record["status"] != "accepted_exact_identity":
            rejected += 1
            continue
        observation = record["observation"]
        base = context(observation)
        board = board_features(observation["fen"])
        reference = next(action for action in record["actions"] if action["reduction"] == 0)
        alpha, beta = observation["alpha"], observation["beta"]
        def relation(score: int) -> int:
            return -1 if score <= alpha else (1 if score >= beta else 0)
        reference_relation = relation(reference["local"]["final_score"])
        reference_root_nodes = int(reference["root"]["nodes"])
        for action in record["actions"]:
            latent.append(observation["latent"])
            features.append(
                base
                + [
                    action["reduction"] / 8.0,
                    action["scout_depth"] / 16.0,
                    (action["scout_depth"] - observation["scout_depth"]) / 8.0,
                ]
            )
            boards.append(board)
            structured_moves.append(structured_move_action(observation, action))
            roots.append(int(record["go_ordinal"]))
            wrong_window.append(action["wrong_window_relation"])
            scout_wrong_window.append(
                action.get("scout_wrong_window_relation", relation(action["local"]["scout_score"]) != reference_relation)
            )
            root_best_changed.append(action["root_best_changed"])
            root_score_changed.append(action["root_score_error_cp"] != 0)
            local_research.append(action["local"]["lmr_research"] or action["local"]["pv_research"])
            log_root_nodes.append(np.log1p(action["root"]["nodes"]))
            log_root_node_fraction.append(np.log(max(action["root_node_fraction"], 1e-9)))
            root_node_delta = int(action["root"]["nodes"]) - reference_root_nodes
            signed_log_root_node_delta.append(np.sign(root_node_delta) * np.log1p(abs(root_node_delta)))
            log_local_nodes.append(np.log1p(action["local"]["total_nodes"]))
            log_absolute_root_score_error.append(np.log1p(abs(action["root_score_error_cp"])))
    if not latent:
        raise ValueError("no accepted exact actions")
    root = np.asarray(roots, dtype=np.uint32)
    validation = root % 5 == 0
    if not validation.any() or validation.all():
        raise ValueError("empty grouped split")
    metadata = {
        "schema": "archon.search-counterfactual-exact-tensor.v1",
        "dataset": str(args.dataset.resolve()),
        "config_hash": config_hash,
        "rows": len(latent),
        "rejected_observations": rejected,
        "latent_dim": len(latent[0]),
        "context_action_dim": len(features[0]),
        "board_dim": len(boards[0]),
        "structured_move_action_dim": len(structured_moves[0]),
        "split": "validation iff source go ordinal % 5 == 0",
        "contract": "exact Threads=1 prefix identity; one scout-depth intervention per fresh root replay; latent is raw signed accumulator and consumer must apply net activation",
    }
    args.output.parent.mkdir(parents=True, exist_ok=True)
    np.savez_compressed(
        args.output,
        latent=np.asarray(latent, dtype=np.int16),
        context_action=np.asarray(features, dtype=np.float32),
        board=np.asarray(boards, dtype=np.uint8),
        structured_move_action=np.asarray(structured_moves, dtype=np.uint8),
        root_ordinal=root,
        validation=validation,
        y_wrong_window=np.asarray(wrong_window, dtype=np.uint8),
        y_scout_wrong_window=np.asarray(scout_wrong_window, dtype=np.uint8),
        y_root_best_changed=np.asarray(root_best_changed, dtype=np.uint8),
        y_root_score_changed=np.asarray(root_score_changed, dtype=np.uint8),
        y_local_research=np.asarray(local_research, dtype=np.uint8),
        y_log_root_nodes=np.asarray(log_root_nodes, dtype=np.float32),
        y_log_root_node_fraction=np.asarray(log_root_node_fraction, dtype=np.float32),
        y_signed_log_root_node_delta=np.asarray(signed_log_root_node_delta, dtype=np.float32),
        y_log_local_nodes=np.asarray(log_local_nodes, dtype=np.float32),
        y_log_absolute_root_score_error=np.asarray(log_absolute_root_score_error, dtype=np.float32),
        metadata=np.asarray(json.dumps(metadata, sort_keys=True)),
    )
    print(json.dumps(metadata, indent=2, sort_keys=True))


if __name__ == "__main__":
    main()
