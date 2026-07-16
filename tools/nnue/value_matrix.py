#!/usr/bin/env python3
"""Resumable source-by-architecture value matrix for ARCHON."""

from __future__ import annotations

import argparse
import json
import os
import shutil
import subprocess
from datetime import datetime, timezone
from pathlib import Path


REPO = Path(__file__).resolve().parents[2]


def atomic_json(path: Path, value: object) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    temporary = path.with_suffix(path.suffix + ".tmp")
    temporary.write_text(json.dumps(value, indent=2, sort_keys=True) + "\n")
    os.replace(temporary, path)


def run(command: list[str], *, dry_run: bool = False) -> None:
    print("$", " ".join(command), flush=True)
    if not dry_run:
        subprocess.run(command, cwd=REPO, check=True)


def candidate_config(spec: dict, source_name: str, cell: str, root: Path) -> dict:
    source = spec["sources"][source_name]
    train = spec["train"]
    net_id = f"{spec['campaign_id']}-{source_name}-{cell}"
    checkpoint = root / "checkpoints" / net_id
    common = {
        "net_id": net_id,
        "provenance": source["provenance"],
        "parent": None,
        "notes": f"Bulk matrix source={source_name} cell={cell}",
        "corpus": {"id": source["id"], "raw_dir": None, "train_bin": source["path"]},
        "sprt": {"enable": False},
    }
    schedule = {
        "SUPERBATCHES": str(train["superbatches"]),
        "BPS": str(train["batches_per_superbatch"]),
        "BATCH": str(train["batch_size"]),
        "LR": str(train["lr"]),
        "LR_END_SB": str(train["lr_end_superbatch"]),
        "SAVE_RATE": str(train["save_rate"]),
        "SEED": str(train["seed"]),
    }
    if cell.startswith("classic"):
        width = 256 if cell == "classic256" else 1024
        common.update(
            arch=f"768x2x{width}x1-screlu",
            engine_features=None if width == 256 else "nnue-l1-1024",
            train={
                "bullet_dir": ".archon-state/toolchains/bullet",
                "example": "train_net0.rs",
                "checkpoint_dir": str(checkpoint),
                "env": {
                    "NET0_FORMAT": source["format"], "NET0_SF_BUFFER_MB": "256",
                    "NET0_HIDDEN": str(width), "NET0_WDL": str(source["wdl"]),
                    **{f"NET0_{key}": value for key, value in schedule.items()},
                },
            },
            verify_out_clip=True,
            pack={"activation": "screlu", "l1": width},
            parity={"enable": False},
        )
    elif cell == "king768x256":
        common.update(
            arch="KingBucket768-49152x2x256x1-screlu",
            engine_features="sota",
            train={
                "bullet_dir": ".archon-state/toolchains/bullet",
                "example": "train_halfka_classic.rs",
                "checkpoint_dir": str(checkpoint),
                "env": {
                    "HKA_DATA": source["path"], "HKA_OUT": str(checkpoint),
                    "HKA_ID": net_id, "HKA_FORMAT": source["format"],
                    "HKA_SF_BUFFER_MB": "256", "HKA_HIDDEN": "256",
                    "HKA_WDL": str(source["wdl"]),
                    **{f"HKA_{key}": value for key, value in schedule.items()},
                },
            },
            verify_out_clip=True,
            pack={"format": "halfka-classic", "activation": "screlu",
                  "input": 49152, "l1": 256},
            parity={"enable": False},
        )
    elif cell == "d40-full":
        common.update(
            arch="sf-halfkav2-hm-fullthreats-768x2-8bkt-hybrid",
            engine_features="sota",
            train={
                "bullet_dir": ".archon-state/toolchains/bullet",
                "example": "bullet/d40_features.rs", "example_name": "d40_features",
                "checkpoint_dir": str(checkpoint),
                "quantised": f"{net_id}-sb{train['superbatches']}-quant.bin",
                "env": {
                    "D40_DATA": source["path"], "D40_OUT": str(checkpoint),
                    "D40_ID": net_id, "D40_FORMAT": source["format"],
                    "D40_SF_BUFFER_MB": "256", "D40_HALFKA_ONLY": "0",
                    "D40_WDL": str(source["wdl"]), "D40_EVAL_TARGET_SCALE": "400",
                    "D40_SCORE_MULTIPLIER": "1.0", "D40_VERIFY_ROWS": "4096",
                    **({
                        "D40_START_SUPERBATCH": str(train["start_superbatch"]),
                        "D40_INIT_RAW": str(REPO / train["init_raw"]),
                    } if "start_superbatch" in train else {}),
                    **{f"D40_{key}": value for key, value in schedule.items()},
                },
            },
            verify_out_clip=False,
            pack={"activation": "d40", "strip_d40_threats": False},
            parity={
                "enable": True, "fens": "data/nnue/parity/d40-local-4096.fens",
                "raw": str(checkpoint / f"{net_id}-final-raw.bin"),
                "max_abs_quantization_logit_error": 2.0,
                "mean_abs_quantization_logit_error": 0.3,
                "max_abs_runtime_logit_error": 0.2,
                "mean_abs_runtime_logit_error": 0.03,
            },
        )
    else:
        raise ValueError(cell)
    return common


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--config", type=Path,
                        default=REPO / "tools/nnue/configs/value-bulk-matrix-v1.json")
    parser.add_argument("--source", default="all",
                        help="a source key from the config's `sources`, or 'all'")
    parser.add_argument("--cell", action="append")
    parser.add_argument("--dry-run", action="store_true")
    args = parser.parse_args()
    spec = json.loads(args.config.read_text())
    root = REPO / "data/nnue/campaigns" / spec["campaign_id"]
    state_path = root / "state.json"
    state = json.loads(state_path.read_text()) if state_path.exists() else {
        "schema": "archon.value-matrix-state.v1", "campaign_id": spec["campaign_id"],
        "created_utc": datetime.now(timezone.utc).isoformat(), "jobs": {},
    }
    sources = list(spec["sources"]) if args.source == "all" else [args.source]
    cells = args.cell or spec["cells"]

    baseline_engine = root / "engines" / "baseline"
    if not baseline_engine.exists() and not args.dry_run:
        run(["cargo", "build", "--release"])
        baseline_engine.parent.mkdir(parents=True, exist_ok=True)
        shutil.copy2(REPO / "target/release/chessengine", baseline_engine)

    for source_name in sources:
        source_path = REPO / spec["sources"][source_name]["path"]
        if not source_path.exists() and not args.dry_run:
            raise SystemExit(f"source is not ready: {source_path}")
        for cell in cells:
            config = candidate_config(spec, source_name, cell, root)
            net_id = config["net_id"]
            job = state["jobs"].setdefault(net_id, {})
            config_path = root / "resolved-configs" / f"{net_id}.json"
            atomic_json(config_path, config)
            if job.get("status") == "complete":
                continue
            if not args.dry_run:
                job.update(status="training", config=str(config_path),
                           updated_utc=datetime.now(timezone.utc).isoformat())
                atomic_json(state_path, state)
            run(["uv", "run", "python", "tools/nnue/train_run.py", "--config", str(config_path),
                 "--only", "preflight,corpus,train,verify,pack,parity,sanity"],
                dry_run=args.dry_run)
            if args.dry_run:
                continue
            candidate_engine = root / "engines" / net_id
            candidate_engine.parent.mkdir(parents=True, exist_ok=True)
            shutil.copy2(REPO / "target/release/chessengine", candidate_engine)
            screen_dir = root / "matches" / net_id / "screen-n5k"
            run([
                "uv", "run", "python", "tools/nnue/eval_match.py",
                "--candidate", f"data/nnue/nets/{net_id}.cenn",
                "--baseline", spec["baseline_net"], "--games", str(spec["screen"]["games"]),
                "--nodes", str(spec["screen"]["nodes"]), "--concurrency",
                str(spec["screen"]["concurrency"]), "--candidate-engine", str(candidate_engine),
                "--baseline-engine", str(baseline_engine), "--output", str(screen_dir),
            ])
            screen = json.loads((screen_dir / "summary.json").read_text())
            job["screen"] = screen
            if screen["candidate_elo"] >= spec["screen"]["survivor_elo_floor"]:
                job["curve"] = {}
                for nodes in spec["screen"]["curve_nodes"]:
                    match_dir = root / "matches" / net_id / f"curve-n{nodes}"
                    run([
                        "uv", "run", "python", "tools/nnue/eval_match.py",
                        "--candidate", f"data/nnue/nets/{net_id}.cenn",
                        "--baseline", spec["baseline_net"], "--games",
                        str(spec["screen"]["curve_games"]), "--nodes", str(nodes),
                        "--concurrency", str(spec["screen"]["concurrency"]),
                        "--candidate-engine", str(candidate_engine), "--baseline-engine",
                        str(baseline_engine), "--output", str(match_dir),
                    ])
                    job["curve"][str(nodes)] = json.loads((match_dir / "summary.json").read_text())
                    atomic_json(state_path, state)
            job.update(status="complete", updated_utc=datetime.now(timezone.utc).isoformat())
            atomic_json(state_path, state)


if __name__ == "__main__":
    main()
