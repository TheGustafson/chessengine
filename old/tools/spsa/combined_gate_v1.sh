#!/usr/bin/env bash
# Combined-config ship SPRT (D45 season closer, 2026-07-17).
# Takes the TENTATIVE-ACCEPTED campaigns from gate night, unions their moved
# params (same 1%-of-range materiality rule as night_gates_v1.sh), and runs
# ONE TC SPRT combined-vs-defaults. Ship rule: H1 or strong positive -> edit
# params.rs defaults; ambiguous/negative -> ship nothing.
# Usage: ACCEPTED="c3-rfp-razor c5-futility-see" bash tools/spsa/combined_gate_v1.sh
#        ACCEPTED="..." DRY=1 bash ...   # print the union, run nothing
set -euo pipefail
cd "$(dirname "$0")/../.."
R=data/nnue/experiments/chain-v2-results.log
say() { echo "[combined $(date +%H:%M:%S)] $*" | tee -a "$R"; }
S3=data/nnue/nets/stage2-v1-s3-2048-farseer-120sb.cenn
: "${ACCEPTED:?space-separated campaign names, e.g. \"c3-rfp-razor c5-futility-see\"}"

OPTS=$(python3 - "$ACCEPTED" <<'EOF'
import json, sys
opts = {}
for camp in sys.argv[1].split():
    t = json.load(open(f"data/nnue/experiments/spsa/{camp}/theta_final.json"))
    f, st = t["final"], t["start"]
    ranges = {p["name"]: (p["min"], p["max"])
              for p in json.load(open(f"tools/spsa/campaigns/{camp}.json"))["params"]}
    moved = {k: v for k, v in f.items()
             if (lambda lo, hi: hi > lo and abs(v - st[k]) / (hi - lo) >= 0.01)(*ranges[k])}
    if not moved:
        print(f"WARNING {camp}: nothing moved >=1% of range", file=sys.stderr)
    # A param tuned in two campaigns would collide; ranges are disjoint by
    # design (one bundle per heuristic family) — assert anyway.
    dup = set(moved) & set(opts)
    assert not dup, f"param collision across campaigns: {dup}"
    opts.update(moved)
assert opts, "union is empty — nothing to gate"
print(" ".join(f"option.{k}={v}" for k, v in sorted(opts.items())))
EOF
)
# DRY runs print to stdout only — the results log stays verdicts-only.
[ "${DRY:-0}" = "1" ] && { echo "[combined DRY] union (${ACCEPTED}): $OPTS"; exit 0; }
say "combined union (${ACCEPTED}): $OPTS"

busy=$(pgrep -f "spsa_tune.py|root-analysis|night_gates" | wc -l || true)
[ "$busy" -gt 0 ] && { say "ABORT: box not quiet ($busy procs)"; exit 1; }

OUT=data/nnue/experiments/spsa/combined-tc-gate.log
say "combined ship SPRT: union vs defaults (10+0.1, SPRT [0,10], cap 600g, srand 20260790)"
# shellcheck disable=SC2086
engines/bin/fastchess \
  -engine cmd=engines/bin/tune-l2048 name=combined $OPTS \
  -engine cmd=engines/bin/tune-l2048 name=default \
  -each proto=uci tc=10+0.1 option.Threads=1 option.Hash=16 \
    "option.EvalFile=$S3" \
  -openings file=data/nnue/openings/UHO_Lichess_4852_v1.epd format=epd order=random \
  -repeat -games 2 -rounds 300 -concurrency 3 -recover \
  -draw movenumber=40 movecount=8 score=15 \
  -resign movecount=4 score=700 -maxmoves 250 \
  -sprt elo0=0 elo1=10 alpha=0.05 beta=0.05 model=logistic \
  -srand 20260790 > "$OUT" 2>&1 || true
echo "[GATE RESULT combined] $(grep -aE 'LLR:' "$OUT" | tail -1) | $(grep -aE 'Elo:' "$OUT" | tail -1) | $(grep -aE 'Games:' "$OUT" | tail -1)" | tee -a "$R"
