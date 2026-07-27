#!/usr/bin/env python3
"""Distributed-SPSA worker: poll the exchange, claim a unit, play it,
upload the result. Stateless — safe to preempt at any moment (the
coordinator reissues stale units; duplicate results are deduped by
unit id, first one wins).

Runs identically on the local box (file:// exchange, N processes) and on
spot VMs (gs:// exchange). Fixed-node games make heterogeneous/loaded
hosts measurement-safe by design (see spsa_tune.py's header).

    python3 worker.py --exchange file:///tmp/spsa-ex --id 3
"""

from __future__ import annotations

import argparse
import json
import re
import subprocess
import sys
import time
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parent))
import exchange as ex  # noqa: E402

SUMMARY_RE = re.compile(
    r"Games:\s*(\d+),\s*Wins:\s*(\d+),\s*Losses:\s*(\d+),\s*Draws:\s*(\d+)")
CLAIM_FRESH_S = 120.0


def fetch_spec(uri: str, cache: Path) -> dict:
    meta = json.loads(ex.get_bytes(uri, "spec/meta.json"))
    for name in meta["files"]:
        dest = cache / name
        if not dest.exists():
            print(f"[worker] fetching spec/{name}", flush=True)
            if not ex.get_file(uri, f"spec/{name}", dest):
                raise RuntimeError(f"spec/{name} missing on exchange")
    for name in ("engine", "fastchess"):
        (cache / meta[name]).chmod(0o755)
    return meta


def play_unit(meta: dict, cache: Path, unit: dict) -> tuple[int, int, int]:
    cmd = [
        str(cache / meta["fastchess"]),
        "-engine", f"cmd={cache / meta['engine']}", "name=plus",
        *unit["plus_opts"],
        "-engine", f"cmd={cache / meta['engine']}", "name=minus",
        *unit["minus_opts"],
        "-each", "proto=uci", "tc=inf", f"nodes={unit['nodes']}",
        "option.Threads=1", "option.Hash=16",
        f"option.EvalFile={cache / meta['net']}",
        "-openings", f"file={cache / meta['book']}", "format=epd",
        "order=random",
        "-repeat", "-games", "2", "-rounds", str(unit["pairs"]),
        "-concurrency", str(unit.get("concurrency", 1)), "-recover",
        "-draw", "movenumber=40", "movecount=8", "score=15",
        "-resign", "movecount=4", "score=700", "-maxmoves", "250",
        "-srand", str(unit["seed"]),
    ]
    r = subprocess.run(cmd, text=True, stdout=subprocess.PIPE,
                       stderr=subprocess.STDOUT)
    m = SUMMARY_RE.findall(r.stdout)
    if r.returncode or not m:
        raise RuntimeError(f"unit failed rc={r.returncode}:"
                           f"\n{r.stdout[-1500:]}")
    games, wins, losses, _ = map(int, m[-1])
    return games, wins, losses


def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("--exchange", required=True)
    ap.add_argument("--id", required=True)
    ap.add_argument("--cache", type=Path,
                    default=Path("/tmp/spsa-worker-cache"))
    ap.add_argument("--poll", type=float, default=3.0)
    args = ap.parse_args()

    args.cache.mkdir(parents=True, exist_ok=True)
    meta = None
    idle_since = time.time()
    while True:
        if ex.exists(args.exchange, "control/STOP"):
            print("[worker] STOP seen — exiting", flush=True)
            return
        units = ex.list_names(args.exchange, "units/")
        done = set(ex.list_names(args.exchange, "results/"))
        claimed = None
        for uname in units:
            uid = uname.split("/", 1)[1].removesuffix(".json")
            if f"results/{uid}.json" in done:
                continue
            age = ex.mtime_age(args.exchange, f"claims/{uid}.claim")
            if age is not None and age < CLAIM_FRESH_S:
                continue  # someone live has it
            ex.put_bytes(args.exchange, f"claims/{uid}.claim",
                         args.id.encode())
            claimed = (uname, uid)
            break
        if claimed is None:
            if time.time() - idle_since > 1800:
                print("[worker] idle 30min — exiting", flush=True)
                return
            time.sleep(args.poll)
            continue
        idle_since = time.time()
        uname, uid = claimed
        if meta is None:
            meta = fetch_spec(args.exchange, args.cache)
        unit = json.loads(ex.get_bytes(args.exchange, uname))
        t0 = time.time()
        try:
            games, wins, losses = play_unit(meta, args.cache, unit)
        except RuntimeError as err:
            print(f"[worker {args.id}] {uid}: {err}", flush=True)
            ex.put_bytes(args.exchange, f"results/{uid}.json", json.dumps(
                {"unit": uid, "error": str(err)[:500],
                 "worker": args.id}).encode())
            continue
        ex.put_bytes(args.exchange, f"results/{uid}.json", json.dumps(
            {"unit": uid, "games": games, "wins": wins, "losses": losses,
             "worker": args.id, "secs": round(time.time() - t0, 1)}).encode())
        print(f"[worker {args.id}] {uid}: {games}g +{wins}-{losses} "
              f"({time.time() - t0:.0f}s)", flush=True)


if __name__ == "__main__":
    main()
