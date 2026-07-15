#!/usr/bin/env python3
"""Select a deterministic stratified sample of early gold labels for deep audit."""

from __future__ import annotations

import argparse
import hashlib
import json
from pathlib import Path


def sha256(path: Path) -> str:
    return hashlib.sha256(path.read_bytes()).hexdigest()


def rank(seed: int, row: dict) -> int:
    payload = f"{seed}:{row['source_position_index']}:{row['fen']}".encode()
    return int.from_bytes(hashlib.blake2b(payload, digest_size=8).digest(), "little")


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("audit", type=Path)
    parser.add_argument("--output", type=Path, required=True)
    parser.add_argument("--count", type=int, default=64)
    parser.add_argument("--seed", type=int, default=0xD40B4)
    parser.add_argument("--accepted-tiers", type=int, nargs="+", default=[1, 2])
    args = parser.parse_args()
    if args.count <= 0:
        parser.error("count must be positive")
    doc = json.loads(args.audit.read_text(encoding="utf-8"))
    candidates = [
        row for row in doc["gold_value"] if int(row["accepted_tier"]) in args.accepted_tiers
    ]
    if len(candidates) < args.count:
        raise RuntimeError("not enough candidate labels")
    for row in candidates:
        row["audit_rank"] = rank(args.seed, row)
        row["audit_stratum"] = (
            f"tier{row['accepted_tier']}:{row['phase']}:{row['score_band']}"
        )
    strata = sorted({row["audit_stratum"] for row in candidates})
    quota = max(1, args.count // len(strata))
    selected: dict[int, dict] = {}
    for stratum in strata:
        rows = sorted((row for row in candidates if row["audit_stratum"] == stratum),
                      key=lambda row: row["audit_rank"])
        for row in rows[:quota]:
            selected[row["source_position_index"]] = row
    for row in sorted(candidates, key=lambda row: row["audit_rank"]):
        if len(selected) >= args.count:
            break
        selected[row["source_position_index"]] = row
    final = sorted(selected.values(), key=lambda row: row["audit_rank"])[:args.count]
    result = {
        "schema": "archon.sf18-gold-cross-depth-selection.v1",
        "source_audit": {"path": str(args.audit.resolve()), "sha256": sha256(args.audit)},
        "input": doc["input"],
        "seed": args.seed,
        "accepted_tiers": args.accepted_tiers,
        "selection": "balanced accepted-tier:phase:score strata, filled by deterministic rank",
        "source_position_indices": [row["source_position_index"] for row in final],
        "selected_rows": [{key: row[key] for key in (
            "source_position_index", "accepted_tier", "phase", "score_band", "flags",
            "audit_stratum", "audit_rank",
        )} for row in final],
    }
    args.output.parent.mkdir(parents=True, exist_ok=True)
    args.output.write_text(json.dumps(result, indent=2, sort_keys=True) + "\n", encoding="utf-8")
    composition = {
        str(tier): sum(row["accepted_tier"] == tier for row in final)
        for tier in args.accepted_tiers
    }
    print(json.dumps({"positions": len(final), "accepted_tier_composition": composition,
                      "output": str(args.output)}, indent=2))


if __name__ == "__main__":
    main()
