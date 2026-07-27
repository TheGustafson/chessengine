#!/usr/bin/env python3
"""Distributed SPSA coordinator: spsa_tune.py's exact math, with the
per-iteration batch fanned out to workers through the exchange instead
of one local fastchess call. The brain stays on this box; workers are
stateless (worker.py — local processes or spot VMs, mixed freely).

Per iteration k: one Rademacher delta, theta±c_k rendered to
option.NAME strings, split into --units work units of --pairs game
pairs each. Units are reissued if no result lands within
--unit-timeout (preempted/hung workers); duplicate results are deduped
(first wins). z = (wins-losses)/games pooled over the WHOLE iteration —
identical estimator to the local tuner at equal total pairs.

Resume: state/theta-latest.json on the exchange is written after every
iteration; restarting the coordinator with --resume picks up from it
(history continues in the same --out).
"""

from __future__ import annotations

import argparse
import json
import random
import sys
import time
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parent))
import exchange as ex  # noqa: E402
import spsa_tune  # noqa: E402  (ALPHA/GAMMA, preflight, engine_options)


def stage_spec(args, uri: str) -> None:
    files = {
        "engine": Path(args.engine), "fastchess": Path(args.fastchess),
        "book": Path(args.book), "net": Path(args.eval_file),
    }
    meta = {k: p.name for k, p in files.items()}
    meta["files"] = [p.name for p in files.values()]
    sidecar = Path(args.eval_file).with_suffix(".cpol")
    if sidecar.exists():
        meta["files"].append(sidecar.name)
        ex.put_file(uri, f"spec/{sidecar.name}", sidecar)
    for _, p in files.items():
        print(f"[dist] staging spec/{p.name}", flush=True)
        ex.put_file(uri, f"spec/{p.name}", p)
    # The GCP worker_startup.sh bootstraps worker.py + exchange.py from
    # spec/ so spot VMs need nothing but python3 + gcloud.
    here = Path(__file__).resolve().parent
    ex.put_file(uri, "spec/worker.py", here / "worker.py")
    ex.put_file(uri, "spec/exchange.py", here / "exchange.py")
    ex.put_bytes(uri, "spec/meta.json", json.dumps(meta).encode())


def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("--config", type=Path, required=True)
    ap.add_argument("--engine", type=Path, required=True)
    ap.add_argument("--fastchess", type=Path,
                    default=spsa_tune.REPO / "engines/bin/fastchess")
    ap.add_argument("--book", type=Path, default=spsa_tune.REPO /
                    "data/nnue/openings/UHO_Lichess_4852_v1.epd")
    ap.add_argument("--eval-file", type=Path, required=True)
    ap.add_argument("--exchange", required=True,
                    help="file:///dir or gs://bucket/prefix")
    ap.add_argument("--out", type=Path, required=True)
    ap.add_argument("--iters", type=int, default=400)
    ap.add_argument("--units", type=int, default=8,
                    help="work units per iteration (max useful workers)")
    ap.add_argument("--pairs", type=int, default=4,
                    help="game pairs PER UNIT (batch = units*pairs*2 games)")
    ap.add_argument("--nodes", type=int, default=20000)
    ap.add_argument("--unit-timeout", type=float, default=300.0)
    ap.add_argument("--seed", type=int, default=20260727)
    ap.add_argument("--r-end", type=float, default=None)
    ap.add_argument("--resume", action="store_true")
    ap.add_argument("--stage-only", action="store_true",
                    help="upload spec + exit (pre-stage a gs:// exchange)")
    args = ap.parse_args()

    config = json.loads(args.config.read_text())
    params = config["params"]
    for p in params:
        p.setdefault("c_end", (p["max"] - p["min"]) / 20.0)
        p.setdefault("r_end", 0.002)
        if args.r_end is not None:
            p["r_end"] = args.r_end

    stage_spec(args, args.exchange)
    if args.stage_only:
        print("[dist] spec staged; exiting (--stage-only)", flush=True)
        return

    theta = spsa_tune.preflight(args.engine, params)
    start_k = 1
    if args.resume:
        blob = ex.get_bytes(args.exchange, "state/theta-latest.json")
        if blob:
            saved = json.loads(blob)
            theta.update(saved["theta"])
            start_k = saved["k"] + 1
            print(f"[dist] resumed at k={start_k}", flush=True)
    start = dict(theta)
    args.out.mkdir(parents=True, exist_ok=True)
    history = (args.out / "history.jsonl").open("a")
    rng = random.Random(args.seed)
    big_a = 0.1 * args.iters
    t0, total_games = time.time(), 0

    for k in range(start_k, args.iters + 1):
        delta = {p["name"]: rng.choice((-1, 1)) for p in params}
        c_k = {p["name"]: p["c_end"] * (args.iters / k) ** spsa_tune.GAMMA
               for p in params}
        plus, minus = {}, {}
        for p in params:
            n = p["name"]
            plus[n] = min(p["max"], max(p["min"], theta[n] + c_k[n] * delta[n]))
            minus[n] = min(p["max"], max(p["min"], theta[n] - c_k[n] * delta[n]))
        plus_opts = spsa_tune.engine_options(plus, params)
        minus_opts = spsa_tune.engine_options(minus, params)

        issued = {}
        for u in range(args.units):
            uid = f"k{k:05d}-u{u:02d}"
            unit = {"unit": uid, "iter": k, "pairs": args.pairs,
                    "nodes": args.nodes, "seed": args.seed + k * 1000 + u,
                    "plus_opts": plus_opts, "minus_opts": minus_opts}
            ex.put_bytes(args.exchange, f"units/{uid}.json",
                         json.dumps(unit).encode())
            issued[uid] = time.time()

        got, g_w_l = {}, [0, 0, 0]
        while len(got) < args.units:
            if ex.exists(args.exchange, "control/STOP"):
                print("[dist] STOP — exiting", flush=True)
                return
            for uid, t_issue in list(issued.items()):
                if uid in got:
                    continue
                blob = ex.get_bytes(args.exchange, f"results/{uid}.json")
                if blob:
                    res = json.loads(blob)
                    got[uid] = res
                    if "error" in res:
                        print(f"[dist] {uid} ERROR: {res['error'][:200]}",
                              flush=True)
                    else:
                        g_w_l[0] += res["games"]
                        g_w_l[1] += res["wins"]
                        g_w_l[2] += res["losses"]
                elif time.time() - t_issue > args.unit_timeout:
                    unit = {"unit": uid, "iter": k, "pairs": args.pairs,
                            "nodes": args.nodes,
                            "seed": args.seed + k * 1000 + 777,
                            "plus_opts": plus_opts,
                            "minus_opts": minus_opts}
                    ex.put_bytes(args.exchange, f"units/{uid}.json",
                                 json.dumps(unit).encode())
                    issued[uid] = time.time()
                    print(f"[dist] reissued {uid}", flush=True)
            time.sleep(2.0)

        games, wins, losses = g_w_l
        if games == 0:
            print(f"[dist] k={k}: every unit errored — aborting", flush=True)
            return
        total_games += games
        z = (wins - losses) / games
        for p in params:
            n = p["name"]
            a_k = p["r_end"] * p["c_end"] ** 2 * \
                ((big_a + args.iters) / (big_a + k)) ** spsa_tune.ALPHA
            theta[n] = min(p["max"], max(p["min"],
                           theta[n] + (a_k / c_k[n]) * z * delta[n]))
        history.write(json.dumps({"k": k, "z": z, "games": games,
                                  "theta": theta}) + "\n")
        history.flush()
        ex.put_bytes(args.exchange, "state/theta-latest.json", json.dumps(
            {"k": k, "theta": theta, "total_games": total_games}).encode())
        if k % 10 == 0 or k == args.iters:
            rate = total_games / max(1e-9, time.time() - t0)
            drift = {n: round(theta[n] - start[n], 1) for n in theta
                     if abs(theta[n] - start[n]) >= 1}
            print(f"[dist {config['name']}] k={k}/{args.iters} "
                  f"games={total_games} ({rate:.1f}/s) drift={drift}",
                  flush=True)

    final = {p["name"]: (theta[p["name"]] if p.get("float")
                         else int(round(theta[p["name"]])))
             for p in params}
    (args.out / "theta_final.json").write_text(json.dumps(
        {"name": config["name"], "start": start, "final": final,
         "iters": args.iters, "units": args.units, "pairs": args.pairs,
         "nodes": args.nodes, "total_games": total_games,
         "seed": args.seed}, indent=2) + "\n")
    ex.put_bytes(args.exchange, "control/STOP", b"done")
    print(f"[dist {config['name']}] DONE: {json.dumps(final)}", flush=True)


if __name__ == "__main__":
    main()
