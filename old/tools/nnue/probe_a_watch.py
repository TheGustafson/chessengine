#!/usr/bin/env python3
"""Probe A checkpoint watcher: pack + screen intermediate checkpoints.

Polls the value-scale-probe-a-v1 checkpoint roots; for every stable
intermediate bullet checkpoint (<net_id>-<sb>/quantised.bin, sb % SAVE == 0),
packs a CENN into the campaign dir and runs an 80-game 5k-node paired screen
vs net-0 with the prebuilt per-width engine binaries. Appends one CSV row per
screen to <campaign>/probe-a-curve.csv. Resumable: existing screen summaries
are trusted and skipped. Exits when <campaign>/DONE exists and no work is
pending.
"""

from __future__ import annotations

import csv
import json
import os
import subprocess
import sys
import time
from pathlib import Path

REPO = Path(__file__).resolve().parents[2]
CAMPAIGN = REPO / "data/nnue/campaigns/value-scale-probe-a-v1"
CKPT_ROOT = CAMPAIGN / "checkpoints"
CURVE_CSV = CAMPAIGN / "probe-a-curve.csv"
SAVE_RATE = 30
FINAL_SB = 480
SEED = 20260715  # one seed => identical paired openings across every checkpoint

CELLS = {
    "value-scale-probe-a-v1-binpack-classic256": {"l1": 256, "engine": "engines/bin/probe-default"},
    "value-scale-probe-a-v1-binpack-classic1024": {"l1": 1024, "engine": "engines/bin/probe-l1024"},
    "value-scale-probe-a-v1-lichess-classic1024": {"l1": 1024, "engine": "engines/bin/probe-l1024"},
}
BASELINE_ENGINE = REPO / "engines/bin/probe-default"
BASELINE_NET = REPO / "data/nnue/nets/net-0.cenn"


def log(msg: str) -> None:
    print(f"[watch {time.strftime('%H:%M:%S')}] {msg}", flush=True)


def stable(path: Path) -> bool:
    try:
        st = path.stat()
    except FileNotFoundError:
        return False
    if time.time() - st.st_mtime < 30:
        return False
    size = st.st_size
    time.sleep(3)
    try:
        return path.stat().st_size == size and size > 0
    except FileNotFoundError:
        return False


def screen_checkpoint(net_id: str, sb: int, quant: Path) -> None:
    cell = CELLS[net_id]
    out_dir = CAMPAIGN / "matches" / net_id / f"curve-sb{sb}-n5k"
    summary = out_dir / "summary.json"
    if summary.exists():
        return
    cenn = CAMPAIGN / "cenn" / f"{net_id}-sb{sb}.cenn"
    cenn.parent.mkdir(parents=True, exist_ok=True)
    if not cenn.exists():
        log(f"packing {net_id} sb{sb}")
        env = dict(os.environ, NET_L1=str(cell["l1"]))
        subprocess.run(
            ["uv", "run", "--with", "numpy", "python", "tools/nnue/netpack.py",
             str(quant), str(cenn), "screlu"],
            cwd=REPO, env=env, check=True, capture_output=True, text=True)
    log(f"screening {net_id} sb{sb} (80g @5k)")
    subprocess.run(
        ["uv", "run", "python", "tools/nnue/eval_match.py",
         "--candidate", str(cenn), "--baseline", str(BASELINE_NET),
         "--games", "80", "--nodes", "5000", "--concurrency", "2",
         "--seed", str(SEED),
         "--candidate-engine", str(REPO / cell["engine"]),
         "--baseline-engine", str(BASELINE_ENGINE),
         "--output", str(out_dir)],
        cwd=REPO, check=True, capture_output=True, text=True)
    row = json.loads(summary.read_text())
    write_header = not CURVE_CSV.exists()
    with CURVE_CSV.open("a", newline="") as fh:
        writer = csv.writer(fh)
        if write_header:
            writer.writerow(["net_id", "sb", "games", "score", "elo", "elo_se"])
        writer.writerow([net_id, sb, row.get("games"), row.get("candidate_score"),
                         row.get("candidate_elo"), row.get("candidate_elo_se")])
    log(f"RESULT {net_id} sb{sb}: score={row.get('candidate_score')} "
        f"elo={row.get('candidate_elo')} se={row.get('candidate_elo_se')}")


def pending_work() -> list[tuple[str, int, Path]]:
    work = []
    for net_id in CELLS:
        root = CKPT_ROOT / net_id
        if not root.is_dir():
            continue
        for sub in sorted(root.iterdir()):
            name = sub.name
            if not name.startswith(net_id + "-"):
                continue
            tail = name[len(net_id) + 1:]
            if not tail.isdigit():
                continue
            sb = int(tail)
            if sb % SAVE_RATE or sb >= FINAL_SB:
                continue  # final SB is screened by the campaign driver itself
            summary = CAMPAIGN / "matches" / net_id / f"curve-sb{sb}-n5k" / "summary.json"
            if summary.exists():
                continue
            quant = sub / "quantised.bin"
            if quant.exists():
                work.append((net_id, sb, quant))
    return work


def main() -> int:
    log("watcher started")
    while True:
        work = pending_work()
        for net_id, sb, quant in work:
            if not stable(quant):
                continue
            try:
                screen_checkpoint(net_id, sb, quant)
            except subprocess.CalledProcessError as err:
                log(f"ERROR {net_id} sb{sb}: {err}\n{err.stderr[-2000:] if err.stderr else ''}")
        if (CAMPAIGN / "DONE").exists() and not pending_work():
            log("campaign DONE and no pending work; exiting")
            return 0
        time.sleep(60)


if __name__ == "__main__":
    sys.exit(main())
