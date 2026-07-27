#!/usr/bin/env python3
"""netonly_eval — the STRATIFIED net-alone comparison table (user directive
2026-07-26: every 10sb from leg-2 sb80, candidate vs the old nets and the
stockfishes, stratified by positions; "I want that tool robust").

Method (pinned by the 2026-07-26 artifact hunts — do not vary casually):
  - Oracle: cached SF18@50k labels on the two persisted position sets
    (data/nnue/evalbench/sets/). Labels clipped to ±800 for iso-MAE.
  - Evals NEVER clipped (isotonic is monotone-invariant; clipping evals
    creates tie-blocks — the raw-frame teacher artifact).
  - Spearman per stratum; iso-MAE (5-fold cross-fitted PAV) on headline
    bases only. Strata with n < 100 are suppressed (weak-test rule).
  - Bases always labeled with n. Sharp headline basis = NON-CHECK (the
    value head is never consulted in check: search/mod.rs:2327).
  - SF adapters emit white-POV -> negated on black-to-move rows.

Usage:
  .venv/bin/python tools/nnue/netonly_eval.py NAME=KIND:SPEC[:ENGINE] ...
  # e.g. the leg-2 checkpoint:
  ... d47v-sb80=archon:/tmp/panel-sb80.cenn:engines/bin/archon-d47v
References (teacher/d47p/SF18/SF16) are always included; --no-refs drops
them. Sweeps run through eval_bench (per-net ARCHON_BIN) and are cached,
so references cost nothing after the first ever run.
"""
from __future__ import annotations

import argparse
import hashlib
import json
import os
import subprocess
import sys
from pathlib import Path

import chess
import numpy as np

ROOT = Path(__file__).resolve().parents[2]
CACHE = ROOT / "data/nnue/evalbench"
SETS = [(ROOT / "data/nnue/evalbench/sets/evalbench_mixed.fens", 3000, "QUIET"),
        (ROOT / "data/nnue/evalbench/sets/leafduty.fens", 2400, "SHARP")]
ORACLE = "engines/bin/stockfish18"
ORACLE_NODES = 50000
CLIP = 800.0
MIN_N = 100

REFS = [
    ("teacher-sb230", "archon",
     "data/nnue/nets/screens/d47s-v15-sb230-ema-rawframe-nocpol.cenn",
     "engines/bin/archon-d47"),
    ("d47p", "archon",
     "data/nnue/nets/screens/d47p-vrepair2-sb120-ema-nocpol.cenn",
     "engines/bin/archon-d47"),
    ("SF18", "sf", "engines/bin/stockfish18", None),
    ("SF16", "sf", "engines/bin/stockfish16", None),
]


def cache_get(kind: str, key: str):
    p = CACHE / f"{kind}-{hashlib.sha256(key.encode()).hexdigest()[:16]}.json"
    return json.loads(p.read_text()) if p.exists() else None


def vec(v):
    return np.array([np.nan if e is None else float(e) for e in v])


def ensure_sweep(name, kind, spec, engine, set_path, count):
    """Sweep via eval_bench if not cached (per-net ARCHON_BIN)."""
    fens = [l.strip() for l in open(set_path) if l.strip()][:count]
    pk = hashlib.sha256("\n".join(fens).encode()).hexdigest()[:16]
    if cache_get("ev", f"{pk}|{kind}|{spec}") is not None:
        return pk
    env = dict(os.environ)
    if engine:
        env["ARCHON_BIN"] = str(ROOT / engine)
    subprocess.run(
        [str(ROOT / ".venv/bin/python"), str(ROOT / "tools/nnue/eval_bench.py"),
         "--positions", str(set_path), "--count", str(count),
         "--oracle", str(ROOT / ORACLE), "--oracle-nodes", str(ORACLE_NODES),
         "--eval", f"{name}={kind}:{spec}"],
        cwd=ROOT, env=env, capture_output=True, check=False)
    if cache_get("ev", f"{pk}|{kind}|{spec}") is None:
        print(f"!! sweep FAILED for {name} on {set_path.name}", file=sys.stderr)
    return pk


def pav(y):
    y = y.astype(float).copy()
    out = []
    for i in range(len(y)):
        out.append([y[i], 1.0, i, i])
        while len(out) > 1 and out[-2][0] > out[-1][0]:
            v2, w2, s2, e2 = out.pop()
            v1, w1, s1, e1 = out.pop()
            out.append([(v1 * w1 + v2 * w2) / (w1 + w2), w1 + w2, s1, e2])
    fit = np.empty_like(y)
    for v, _, s, e in out:
        fit[s:e + 1] = v
    return fit


def iso_mae(x, y, folds=5, seed=0):
    rng = np.random.default_rng(seed)
    idx = rng.permutation(len(x))
    mae = np.empty(len(x))
    for f in range(folds):
        te = idx[f::folds]
        tr = np.setdiff1d(idx, te)
        o = np.argsort(x[tr], kind="stable")
        xt, yt = x[tr][o], pav(y[tr][o])
        ux, inv = np.unique(xt, return_index=True)
        mae[te] = np.abs(np.interp(x[te], ux, yt[inv]) - y[te])
    return float(mae.mean())


def spear(a, b, m):
    if m.sum() < MIN_N:
        return float("nan")
    xr = np.argsort(np.argsort(a[m])).astype(float)
    yr = np.argsort(np.argsort(b[m])).astype(float)
    return float(np.corrcoef(xr, yr)[0, 1])


def main() -> int:
    ap = argparse.ArgumentParser()
    ap.add_argument("nets", nargs="+",
                    help="NAME=KIND:SPEC[:ENGINE]  (kind archon|sf)")
    ap.add_argument("--no-refs", action="store_true")
    a = ap.parse_args()

    entries = []
    for s in a.nets:
        name, rest = s.split("=", 1)
        parts = rest.split(":")
        kind, spec = parts[0], parts[1]
        engine = parts[2] if len(parts) > 2 else (
            "engines/bin/archon-d47" if kind == "archon" else None)
        entries.append((name, kind, spec, engine))
    if not a.no_refs:
        entries += [r for r in REFS if r[0] not in {e[0] for e in entries}]

    results_for_log = []
    for set_path, count, tag in SETS:
        fens = [l.strip() for l in open(set_path) if l.strip()][:count]
        pk = None
        for name, kind, spec, engine in entries:
            pk = ensure_sweep(name, kind, spec, engine, set_path, count)
        L = vec(cache_get("lab", f"{pk}|{ORACLE}|{ORACLE_NODES}"))
        if np.all(np.isnan(L)):
            print(f"!! no oracle labels for {tag} — label the set first")
            continue
        Lc = np.clip(L, -CLIP, CLIP)
        boards = [chess.Board(f) for f in fens]
        chk = np.array([b.is_check() for b in boards])
        pieces = np.array([chess.popcount(b.occupied) for b in boards])
        black = np.array([f.split()[1] == "b" for f in fens])

        series = {}
        for name, kind, spec, engine in entries:
            x = vec(cache_get("ev", f"{pk}|{kind}|{spec}") or [])
            if x.size == 0:
                continue
            series[name] = np.where(black, -x, x) if kind == "sf" else x

        nc = ~chk
        strata = [
            ("non-check (HEADLINE)", nc),
            ("in-check", chk),
            ("pieces<=8 nc", nc & (pieces <= 8)),
            ("pieces 9-16 nc", nc & (pieces >= 9) & (pieces <= 16)),
            ("pieces 17-24 nc", nc & (pieces >= 17) & (pieces <= 24)),
            ("pieces>=25 nc", nc & (pieces >= 25)),
            ("|label|<100 nc", nc & (np.abs(L) < 100)),
            ("|label| 100-400 nc", nc & (np.abs(L) >= 100) & (np.abs(L) < 400)),
            ("|label|>=400 nc", nc & (np.abs(L) >= 400)),
        ]
        names = list(series)
        print(f"\n════ {tag} — Spearman by stratum (n<{MIN_N} suppressed) ════")
        print(f"{'stratum':22s} {'n':>5s}  " + "  ".join(f"{n:>13s}" for n in names))
        for label, m0 in strata:
            m = m0.copy()
            for x in series.values():
                m &= ~np.isnan(x) & ~np.isnan(L)
            if m.sum() < MIN_N:
                continue
            row = "  ".join(f"{spear(x, Lc, m):13.4f}" for x in series.values())
            print(f"{label:22s} {int(m.sum()):5d}  {row}")
        # headline iso-MAE (common non-check basis)
        m = nc.copy()
        for x in series.values():
            m &= ~np.isnan(x) & ~np.isnan(L)
        print(f"{'iso-MAE (headline nc)':22s} {int(m.sum()):5d}  " +
              "  ".join(f"{iso_mae(x[m], Lc[m]):13.1f}" for x in series.values()))
        for nm, x in series.items():
            results_for_log.append({
                "net": nm, "set": tag, "n": int(m.sum()),
                "spearman_nc": round(float(spear(x, Lc, nc & m)), 6),
                "iso_mae_nc": round(float(iso_mae(x[m], Lc[m])), 3)})
    print("\n(labels clipped ±800; evals unclipped; PAV 5-fold; "
          "never quote a number without its stratum + n)")
    _append_metrics_log(results_for_log)
    return 0


# The project has no cheap proxy that predicts fixed-node game strength — every
# promotion needs games, which is the single biggest brake on the research loop.
# Fitting one needs (proxy metrics, fn Elo) pairs across many nets, and we
# cannot fit it retroactively: the panels print their numbers and discard them,
# and the eval cache is keyed by content hash with no net mapping. So start
# accumulating the dataset now — every future panel run adds a row, and in a few
# weeks there is enough to regress fn Elo (from the ledger) on these columns and
# find out which proxy, if any, actually tracks strength.
#
# Deliberately defensive: this runs inside the every-30sb watcher, and a metrics
# logger must never be the reason an instrument round dies.
def _append_metrics_log(rows) -> None:
    try:
        import datetime
        if not rows:
            return
        out = ROOT / "data/nnue/evalbench/proxy_metrics.jsonl"
        out.parent.mkdir(parents=True, exist_ok=True)
        stamp = datetime.datetime.now().isoformat(timespec="seconds")
        with out.open("a") as f:
            for r in rows:
                f.write(json.dumps({"ts": stamp, **r}) + "\n")
    except Exception as err:  # noqa: BLE001 — never break the caller
        print(f"(metrics log skipped: {type(err).__name__}: {err})")


if __name__ == "__main__":
    sys.exit(main())
