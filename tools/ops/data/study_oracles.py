#!/usr/bin/env python3
"""tools/ops/data/study_oracles.py — WHICH oracle, and does the choice matter?

THE QUESTION
  We have stockfish18, stockfish16 and stockfish-eca43a97 (our pinned constants
  reference) plus our own binaries. Two separate things must be established and
  they are easy to conflate:

    1. Which is STRONGEST at the node count we will actually use?
    2. Does the choice CHANGE THE LABELS enough to matter?

  (2) is the decision-relevant one. If three oracles produce labels that agree
  on the best move 97% of the time and sit 6 cp apart, then picking the
  "strongest" is not worth a rebuild, and the honest recommendation is to take
  whichever is fastest. Only a measured disagreement justifies a preference.

METHOD — discrete outcomes first
  Label the SAME positions with every oracle at the SAME node count, then:
    * pairwise BEST-MOVE agreement          (frame-free, scale-free)
    * pairwise |delta cp| quantiles         (all SF binaries print real cp, so
                                             these are directly comparable; an
                                             ARCHON oracle is refused rather
                                             than silently divided by 2.07)
    * each oracle's own EVAL-OUTCOME DISAGREEMENT against the game result --
      the same statistic study_label_quality.py uses, so the two studies can be
      read against each other.
    * throughput, because a 3% label change bought at half the speed is a bad
      trade and the table should make that visible.

NULL
  A self-vs-self arm is included automatically: the first oracle is labelled
  TWICE and compared with itself. That pair must show 100.00% agreement and a
  zero cp delta. If it does not, the comparison instrument has run-to-run noise
  and every cross-oracle number below it is inside that noise.

USAGE
  .venv/bin/python tools/ops/data/study_oracles.py \
      --positions data/nnue/selfplay/study/combined.tsv \
      --oracles engines/bin/stockfish18,engines/bin/stockfish16,engines/bin/stockfish-eca43a97 \
      --nodes 25000 --limit 3000 --workers 5 \
      --out data/nnue/selfplay/study/oracles
"""
from __future__ import annotations

import argparse
import json
import subprocess
import sys
from pathlib import Path

ROOT = Path(__file__).resolve().parents[3]
LABELLER = ROOT / "tools/ops/data/label_positions.py"
PY = ROOT / ".venv/bin/python"


def load(path: Path):
    rows = {}
    with open(path) as fh:
        hdr = fh.readline().rstrip("\n").split("\t")
        ix = {c: i for i, c in enumerate(hdr)}
        for line in fh:
            f = line.rstrip("\n").split("\t")
            if len(f) < len(hdr):
                continue
            raw = f[ix["raw_cp"]]
            rows[f[ix["fen"]]] = dict(
                raw=int(raw) if raw != "" else None,
                best=f[ix["best"]], result=f[ix["result"]],
                reason=f[ix["drop_reason"]] if "drop_reason" in ix else "")
    return rows


def quant(xs, q):
    if not xs:
        return 0
    s = sorted(xs)
    return s[min(len(s) - 1, int(q * (len(s) - 1)))]


def main() -> int:
    ap = argparse.ArgumentParser()
    ap.add_argument("--positions", required=True)
    ap.add_argument("--oracles", required=True)
    ap.add_argument("--nodes", type=int, default=25000)
    ap.add_argument("--limit", type=int, default=3000)
    ap.add_argument("--workers", type=int, default=5)
    ap.add_argument("--threshold", type=int, default=150)
    ap.add_argument("--out", required=True)
    args = ap.parse_args()

    oracles = args.oracles.split(",")
    outdir = Path(args.out)
    outdir.mkdir(parents=True, exist_ok=True)

    # arms: every oracle, plus a self-vs-self NULL on the first one
    arms = [(Path(o).name, o) for o in oracles]
    arms.append((Path(oracles[0]).name + "__NULLCOPY", oracles[0]))

    data, thr = {}, {}
    for name, path in arms:
        p = outdir / f"{name}.tsv"
        if not p.exists():
            cmd = [str(PY), str(LABELLER), args.positions, "--out", str(p),
                   "--oracle", path, "--nodes", str(args.nodes),
                   "--workers", str(args.workers), "--no-filter",
                   "--limit", str(args.limit)]
            sys.stderr.write(f"[{name}] labelling @{args.nodes}n...\n")
            r = subprocess.run(cmd, capture_output=True, text=True)
            if r.returncode != 0:
                sys.stderr.write(r.stderr[-3000:])
                return 2
        st = json.loads(Path(str(p) + ".stats.json").read_text())
        if st["oracle"]["label_frame"] != "sf-printed-cp":
            sys.stderr.write(
                f"REFUSING: {name} labels are in {st['oracle']['label_frame']}, "
                "not sf-printed-cp. Mixing frames in one cp comparison is the "
                "error the tool boundary exists to prevent.\n")
            return 3
        thr[name] = st["throughput"]
        data[name] = load(p)

    common = None
    for name, _ in arms:
        s = {f for f, d in data[name].items()
             if d["raw"] is not None and not d["reason"]}
        common = s if common is None else (common & s)
    common = sorted(common)
    sys.stderr.write(f"paired set: {len(common)} positions\n")

    rmap = {"1.0": 1.0, "0.5": 0.5, "0.0": 0.0}
    rep = dict(nodes=args.nodes, paired_positions=len(common),
               threshold_cp=args.threshold, per_oracle={}, pairwise={})

    for name, _ in arms:
        d = data[name]
        dec = contra = 0
        for f in common:
            if d[f]["result"] not in rmap:
                continue
            cp, r = d[f]["raw"], rmap[d[f]["result"]]
            if abs(cp) < args.threshold:
                continue
            dec += 1
            if (cp >= args.threshold and r == 0.0) or \
               (cp <= -args.threshold and r == 1.0):
                contra += 1
        rep["per_oracle"][name] = dict(
            decisive=dec, contradicted=contra,
            disagreement_pct=round(100 * contra / max(dec, 1), 2),
            throughput=thr[name])

    names = [n for n, _ in arms]
    for i, a in enumerate(names):
        for b in names[i + 1:]:
            da, db = data[a], data[b]
            agree = sum(1 for f in common if da[f]["best"] == db[f]["best"])
            deltas = [abs(da[f]["raw"] - db[f]["raw"]) for f in common]
            rep["pairwise"][f"{a} vs {b}"] = dict(
                bestmove_agree_pct=round(100 * agree / max(len(common), 1), 2),
                abs_delta_cp=dict(p50=quant(deltas, .5), p90=quant(deltas, .9),
                                  p99=quant(deltas, .99),
                                  mean=round(sum(deltas) / max(len(deltas), 1), 1)))

    (outdir / "report.json").write_text(json.dumps(rep, indent=2))

    print(f"\n=== oracle comparison @ {args.nodes} nodes "
          f"({len(common)} paired positions) ===\n")
    print(f"{'oracle':<34} {'disagree%':>10} {'decisive':>9} {'pos/s':>8}")
    for n in names:
        r = rep["per_oracle"][n]
        print(f"{n:<34} {r['disagreement_pct']:>10.2f} {r['decisive']:>9} "
              f"{r['throughput']['pos_per_s']:>8.1f}")
    print(f"\n{'pair':<50} {'bmAgree%':>9} {'|d|p50':>7} {'|d|p90':>7} {'|d|mean':>8}")
    for k, v in rep["pairwise"].items():
        d = v["abs_delta_cp"]
        print(f"{k:<50} {v['bestmove_agree_pct']:>9.2f} {d['p50']:>7} "
              f"{d['p90']:>7} {d['mean']:>8.1f}")
    print("\nThe __NULLCOPY row/pair is the SAME binary against itself: it must")
    print("read 100.00% agreement and 0 cp. Any cross-oracle difference smaller")
    print("than that pair's is inside the instrument's own noise.")
    print(f"\nreport -> {outdir/'report.json'}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
