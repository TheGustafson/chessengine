#!/usr/bin/env bash
# H-AUX effective-dose tripwire (2026-07-21): how far did an aux-λ run's VALUE
# output move from the λ=0 recipe baseline (d40-uho-v1-240sb)?
#
#   bash tools/nnue/haux_divergence_probe.sh <candidate.cenn>
#
# Reads: same-recipe-different-run noise floor ≈ the foldin-tight-240sb line
# (its aux provably contributed ~nothing — absorption probe 48.3%, NIGHT-3).
#   * divergence ≈ the floor  -> aux DID NOT ENGAGE (λ lost / dose washed out):
#     the run measured nothing — check env before trusting its gate.
#   * divergence >> floor     -> trunk was shaped; gate Elo is meaningful.
#   * correlation collapsing  -> λ too hot (aux dominated the value loss).
set -uo pipefail
cd "$(dirname "$0")/../.."
CAND="${1:?candidate .cenn}"
BASE=data/nnue/nets/d40-uho-v1-240sb.cenn
ENG=engines/bin/archon-sota-native
FENS=tools/nnue/calib/scale-check-800.fens
T="$(mktemp -d "${TMPDIR:-/tmp}/haux-probe.XXXXXX")"; trap 'rm -rf "$T"' EXIT
"$ENG" nnue-eval "$CAND" < "$FENS" > "$T/c.ev"
"$ENG" nnue-eval "$BASE" < "$FENS" > "$T/b.ev"
python3 - "$T/c.ev" "$T/b.ev" "$CAND" <<'PY'
import sys, math
c=[int(l) for l in open(sys.argv[1])]; b=[int(l) for l in open(sys.argv[2])]
d=sorted(abs(x-y) for x,y in zip(c,b)); n=len(d)
mc=sum(c)/n; mb=sum(b)/n
r=sum((x-mc)*(y-mb) for x,y in zip(c,b))/math.sqrt(sum((x-mc)**2 for x in c)*sum((y-mb)**2 for y in b))
print(f"[haux-probe] {sys.argv[3]}")
print(f"[haux-probe] |Δ| vs λ=0 baseline: mean {sum(d)/n:.1f}  p50 {d[n//2]}  p90 {d[9*n//10]}  r={r:.4f}")
print(f"[haux-probe] MEASURED noise floor (foldin-tight vs v1-240sb, aux-inert, same recipe/seed,")
print(f"[haux-probe] different run): mean 47.3  p50 30  p90 115  r=0.9871. Divergence at ~the floor")
print(f"[haux-probe] = indistinguishable from run-noise; the engagement signature is divergence")
print(f"[haux-probe] SCALING WITH λ across the sweep (pol4 >> pol1), not any single number.")
PY
