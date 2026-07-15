#!/usr/bin/env python3
"""Generate exact Archon LMR counterfactuals by deterministic tree replay.

Every observation first replays at its recorded scout depth and must reproduce
both the local telemetry and the complete root result exactly. Only then are
alternative scout depths accepted as causal labels. Each action uses a fresh
Threads=1 process, preserving the original cold root search prefix up to the
single explicit intervention.
"""

from __future__ import annotations

import argparse
from concurrent.futures import ProcessPoolExecutor, as_completed
import hashlib
import heapq
import json
import os
from pathlib import Path
import shutil
import tempfile
from collections import Counter

import chess
import chess.engine


TRACE_SCHEMA = "archon.search-lmr-observation.v3"
INTERVENTION_SCHEMA = "archon.lmr-intervention.v1"
SCHEMA = "archon.search-counterfactual-exact-v1"


def sha256(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as source:
        for block in iter(lambda: source.read(1 << 20), b""):
            digest.update(block)
    return digest.hexdigest()


def canonical_hash(value: object) -> str:
    return hashlib.sha256(json.dumps(value, sort_keys=True, separators=(",", ":")).encode()).hexdigest()


def atomic_json(path: Path, value: object) -> None:
    temporary = path.with_suffix(path.suffix + ".tmp")
    temporary.write_text(json.dumps(value, indent=2, sort_keys=True) + "\n", encoding="utf-8")
    temporary.replace(path)


def category(row: dict) -> int:
    if row["lmr_research"]:
        return 0
    if row["alpha_raise"] or row["beta_cutoff"]:
        return 1
    if abs(row["scout_score"] - row["alpha"]) <= 50:
        return 2
    return 3


def select(trace: Path, source_manifest: dict, count: int, max_per_root: int) -> list[dict]:
    heaps = [[], [], [], []]
    # Applying a root cap only *after* global category heaps lets prolific roots
    # crowd independent roots out of the candidate pool.  Keep the best
    # `max_per_root` hash-ranked rows for each (category, root) first so the
    # later quota pass sees the independent coverage that actually exists.
    root_heaps = [dict(), dict(), dict(), dict()] if max_per_root else None
    go = -1
    with trace.open(encoding="utf-8") as source:
        header = json.loads(next(source))
        if header.get("schema") != TRACE_SCHEMA or header.get("kind") != "header":
            raise ValueError("exact replay requires v3 full-FEN telemetry")
        for line_number, line in enumerate(source, 2):
            row = json.loads(line)
            if row.get("kind") == "go":
                go += 1
                continue
            if row.get("kind") != "row":
                continue
            if row["nominal_child_depth"] < 2 or row["scout_depth"] > row["nominal_child_depth"]:
                continue
            root = source_manifest["positions"][go]
            tie = int.from_bytes(
                hashlib.blake2b(
                    f"{line_number}:{row['position_key']}:{row['raw_move']}".encode(), digest_size=8
                ).digest(),
                "little",
            )
            rank = (tie, line_number)
            inverted = (-tie, -line_number)
            item = {"observation_line": line_number, "go_ordinal": go, "root": root, "observation": row}
            group = category(row)
            if root_heaps is not None:
                heap = root_heaps[group].setdefault(go, [])
                if len(heap) < max_per_root:
                    heapq.heappush(heap, (inverted, item))
                elif inverted > heap[0][0]:
                    heapq.heapreplace(heap, (inverted, item))
            else:
                heap = heaps[group]
                if len(heap) < count:
                    heapq.heappush(heap, (inverted, item))
                elif inverted > heap[0][0]:
                    heapq.heapreplace(heap, (inverted, item))
    if root_heaps is not None:
        heaps = [
            [entry for root_heap in category_heaps.values() for entry in root_heap]
            for category_heaps in root_heaps
        ]
    candidates = [
        [
            item
            for _rank, item in sorted(
                (((-inverted[0], -inverted[1]), item) for inverted, item in heap),
                key=lambda entry: entry[0],
            )
        ]
        for heap in heaps
    ]
    quotas = [count // 2, count // 5, count * 15 // 100]
    quotas.append(count - sum(quotas))
    selected = []
    root_counts: Counter[int] = Counter()
    for items, quota in zip(candidates, quotas):
        taken = 0
        for item in items:
            root = item["go_ordinal"]
            if max_per_root and root_counts[root] >= max_per_root:
                continue
            selected.append(item)
            root_counts[root] += 1
            taken += 1
            if taken == quota:
                break
    used = {item["observation_line"] for item in selected}
    if len(selected) < count:
        remainder = sorted(
            (item for items in candidates for item in items if item["observation_line"] not in used),
            key=lambda item: item["observation_line"],
        )
        for item in remainder:
            root = item["go_ordinal"]
            if max_per_root and root_counts[root] >= max_per_root:
                continue
            selected.append(item)
            root_counts[root] += 1
            if len(selected) == count:
                break
    return sorted(selected, key=lambda item: item["observation_line"])


def root_result(info: dict, board: chess.Board) -> dict:
    score = info.get("score")
    return {
        "score_cp": None if score is None else score.pov(board.turn).score(mate_score=30000),
        "nodes": int(info.get("nodes", 0)),
        "depth": int(info.get("depth", 0)),
        "seldepth": int(info.get("seldepth", 0)),
        "pv": [move.uci() for move in info.get("pv", [])],
    }


def run_action(item: dict, config: dict, occurrence: int, scout_depth: int) -> dict:
    row = item["observation"]
    root = item["root"]
    with tempfile.TemporaryDirectory(prefix="archon-exact-") as temporary:
        result_path = Path(temporary) / "intervention.json"
        environment = dict(os.environ)
        environment.update(
            {
                "ARCHON_LMR_OVERRIDE_RESULT": str(result_path),
                "ARCHON_LMR_OVERRIDE_KEY": str(row["position_key"]),
                "ARCHON_LMR_OVERRIDE_MOVE": str(row["raw_move"]),
                "ARCHON_LMR_OVERRIDE_DEPTH": str(row["depth"]),
                "ARCHON_LMR_OVERRIDE_PLY": str(row["ply"]),
                "ARCHON_LMR_OVERRIDE_MOVE_INDEX": str(row["move_index"]),
                "ARCHON_LMR_OVERRIDE_OCCURRENCE": str(occurrence),
                "ARCHON_LMR_OVERRIDE_SCOUT_DEPTH": str(scout_depth),
            }
        )
        engine = chess.engine.SimpleEngine.popen_uci(
            config["engine"], timeout=config["timeout_seconds"], env=environment
        )
        board = chess.Board(root["fen"])
        try:
            engine.configure({"Threads": 1, "Hash": config["hash_mb"], "EvalFile": config["eval"]})
            info = engine.analyse(
                board,
                chess.engine.Limit(depth=config["root_depth"]),
                game=object(),
                info=chess.engine.INFO_ALL,
            )
        finally:
            engine.quit()
        if not result_path.is_file():
            raise RuntimeError("engine did not write intervention result")
        intervention = json.loads(result_path.read_text(encoding="utf-8"))
        if intervention.get("schema") != INTERVENTION_SCHEMA:
            raise RuntimeError("intervention result schema mismatch")
        return {"root": root_result(info, board), "intervention": intervention}


def identity_matches(item: dict, result: dict) -> bool:
    row = item["observation"]
    root = item["root"]
    intervention = result["intervention"]
    if not intervention.get("applied") or "outcome" not in intervention:
        return False
    outcome = intervention["outcome"]
    local_fields = (
        "alpha", "beta", "scout_score", "final_score", "scout_nodes", "research_nodes",
        "pv_research_nodes", "total_nodes", "lmr_research", "pv_research",
    )
    if any(outcome[field] != row[field] for field in local_fields):
        return False
    if outcome["classical_scout_depth"] != row["scout_depth"] or outcome["applied_scout_depth"] != row["scout_depth"]:
        return False
    replay_root = result["root"]
    return (
        replay_root["nodes"] == root["nodes"]
        and replay_root["score_cp"] == root["score_cp"]
        and replay_root["pv"] == root["pv"]
    )


def relation(score: int, alpha: int, beta: int) -> str:
    if score <= alpha:
        return "upper"
    if score >= beta:
        return "lower"
    return "interior"


def process(item: dict, config: dict, max_occurrence: int, max_reduction: int, config_hash: str) -> dict:
    row = item["observation"]
    identity_attempts = []
    occurrence = None
    identity = None
    for candidate in range(max_occurrence + 1):
        result = run_action(item, config, candidate, row["scout_depth"])
        identity_attempts.append(
            {
                "occurrence": candidate,
                "applied": result["intervention"].get("applied", False),
                "matched": identity_matches(item, result),
            }
        )
        if identity_attempts[-1]["matched"]:
            occurrence = candidate
            identity = result
            break
    base = {
        "schema": SCHEMA,
        "config_hash": config_hash,
        "observation_line": item["observation_line"],
        "go_ordinal": item["go_ordinal"],
        "observation": row,
        "source_root": item["root"],
        "identity_attempts": identity_attempts,
    }
    if occurrence is None:
        return {**base, "status": "identity_rejected", "actions": []}
    nominal = int(row["nominal_child_depth"])
    scout_depths = sorted({max(1, nominal - reduction) for reduction in range(max_reduction + 1)} | {int(row["scout_depth"])})
    actions = []
    for depth in scout_depths:
        result = identity if depth == row["scout_depth"] else run_action(item, config, occurrence, depth)
        intervention = result["intervention"]
        if not intervention.get("applied") or "outcome" not in intervention:
            raise RuntimeError("matched intervention disappeared under alternative action")
        actions.append(
            {
                "scout_depth": depth,
                "reduction": nominal - depth,
                "local": intervention["outcome"],
                "root": result["root"],
            }
        )
    reference = next(action for action in actions if action["reduction"] == 0)
    ref_relation = relation(reference["local"]["final_score"], row["alpha"], row["beta"])
    for action in actions:
        scout_relation = relation(action["local"]["scout_score"], row["alpha"], row["beta"])
        action_relation = relation(action["local"]["final_score"], row["alpha"], row["beta"])
        action["scout_window_relation"] = scout_relation
        action["scout_wrong_window_relation"] = scout_relation != ref_relation
        action["window_relation"] = action_relation
        action["wrong_window_relation"] = action_relation != ref_relation
        action["root_score_error_cp"] = action["root"]["score_cp"] - reference["root"]["score_cp"]
        action["root_best_changed"] = action["root"]["pv"][:1] != reference["root"]["pv"][:1]
        action["root_node_fraction"] = action["root"]["nodes"] / max(1, reference["root"]["nodes"])
    return {
        **base,
        "status": "accepted_exact_identity",
        "matched_occurrence": occurrence,
        "reference": {"scout_depth": nominal, "window_relation": ref_relation},
        "actions": actions,
    }


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--trace", type=Path, required=True)
    parser.add_argument("--source-manifest", type=Path, required=True)
    parser.add_argument("--output", type=Path, required=True)
    parser.add_argument("--engine", type=Path, required=True)
    parser.add_argument("--eval", type=Path, required=True)
    parser.add_argument("--count", type=int, default=16)
    parser.add_argument("--max-reduction", type=int, default=3)
    parser.add_argument("--max-occurrence", type=int, default=3)
    parser.add_argument("--max-per-root", type=int, default=0, help="zero means unlimited")
    parser.add_argument("--workers", type=int, default=2)
    parser.add_argument("--hash-mb", type=int, default=16)
    parser.add_argument("--timeout-seconds", type=float, default=120.0)
    args = parser.parse_args()
    if min(args.count, args.max_reduction, args.workers, args.hash_mb) <= 0 or min(args.max_occurrence, args.max_per_root) < 0:
        parser.error("invalid positive count/reduction/workers/hash or negative occurrence")
    trace = args.trace.resolve()
    source_manifest_path = args.source_manifest.resolve()
    engine = args.engine.resolve()
    eval_file = args.eval.resolve()
    for label, path in (("trace", trace), ("source manifest", source_manifest_path), ("engine", engine), ("eval", eval_file)):
        if not path.is_file():
            parser.error(f"missing {label}: {path}")
    source_manifest = json.loads(source_manifest_path.read_text(encoding="utf-8"))
    root_depth = int(source_manifest["search"]["depth"])
    selected = select(trace, source_manifest, args.count, args.max_per_root)
    if len(selected) != args.count:
        raise RuntimeError(f"requested {args.count} observations, selected {len(selected)}")
    generator = Path(__file__).resolve()
    identities = {
        "generator": {"path": str(generator), "sha256": sha256(generator)},
        "trace": {"path": str(trace), "sha256": sha256(trace)},
        "source_manifest": {"path": str(source_manifest_path), "sha256": sha256(source_manifest_path)},
        "source_engine": source_manifest["engine"],
        "oracle_engine": {"path": str(engine), "sha256": sha256(engine)},
        "eval": {"path": str(eval_file), "sha256": sha256(eval_file)},
    }
    config = {
        "engine": str(engine), "eval": str(eval_file), "root_depth": root_depth,
        "hash_mb": args.hash_mb, "timeout_seconds": args.timeout_seconds,
    }
    payload = {
        "schema": SCHEMA,
        "identities": identities,
        "search": {"root_depth": root_depth, "threads": 1, "hash_mb": args.hash_mb, "fresh_process_per_action": True},
        "selection": {
            "quotas": "50% re-search, 20% alpha/cutoff, 15% near-window, 15% uniform",
            "ranking": "smallest blake2b-64(line:position_key:move), per-root cap before quota",
            "max_per_root": args.max_per_root,
            "lines": [item["observation_line"] for item in selected],
        },
        "actions": {"max_reduction": args.max_reduction, "reference": "nominal child depth (reduction 0)"},
        "identity_gate": {"max_occurrence": args.max_occurrence, "local_fields": "exact", "root_score_nodes_pv": "exact"},
    }
    config_hash = canonical_hash(payload)
    output = args.output.resolve()
    records_dir = output / "records"
    manifest_path = output / "manifest.json"
    if output.exists():
        manifest = json.loads(manifest_path.read_text(encoding="utf-8"))
        if manifest.get("config_hash") != config_hash:
            parser.error("existing output configuration mismatch")
    else:
        records_dir.mkdir(parents=True)
        shutil.copy2(generator, output / "generator.snapshot.py")
        atomic_json(manifest_path, {**payload, "config_hash": config_hash})
    pending = []
    for ordinal, item in enumerate(selected):
        path = records_dir / f"{ordinal:06d}.json"
        if path.exists():
            record = json.loads(path.read_text(encoding="utf-8"))
            if record.get("config_hash") != config_hash or record.get("observation_line") != item["observation_line"]:
                raise RuntimeError(f"invalid completed record {path}")
        else:
            pending.append((ordinal, item, path))
    with ProcessPoolExecutor(max_workers=args.workers) as pool:
        futures = {
            pool.submit(process, item, config, args.max_occurrence, args.max_reduction, config_hash): (ordinal, path)
            for ordinal, item, path in pending
        }
        for future in as_completed(futures):
            ordinal, path = futures[future]
            atomic_json(path, future.result())
            print(f"wrote {ordinal + 1}/{len(selected)} {path}")
    records = [json.loads(path.read_text(encoding="utf-8")) for path in sorted(records_dir.glob("*.json"))]
    if len(records) != len(selected):
        raise RuntimeError("record coverage incomplete")
    accepted = sum(record["status"] == "accepted_exact_identity" for record in records)
    print(f"complete: accepted exact identity {accepted}/{len(records)} in {output}")


if __name__ == "__main__":
    main()
