#!/usr/bin/env bash
# ============================================================================
# net_scale_check.sh — the OUTPUT-SCALE calibration authority (2026-07-21).
#
# Every cp-threshold in search (RFP/razoring/futility margins, aspiration
# windows, history bonuses) is co-adapted to the incumbent net's output
# magnitude. A net trained with a wrong SCORE_MULT fits its labels perfectly
# yet outputs at the wrong scale — the exact failure that produced the canon
# -582 (a HALF-scale net; see docs/research/sf18-canon-data-diagnosis-2026-07-21.md).
# Loss curves cannot catch this. This check can, in seconds, with no games:
# it compares static evals (nnue-eval) of a candidate vs the reference net on
# a fixed 800-FEN probe set and reports the robust output-scale ratio.
#
#   bash tools/nnue/net_scale_check.sh <candidate.cenn> [reference.cenn] [engine]
#
# Defaults: reference = the incumbent d40-uho-720sb, engine = archon-sota-native.
# For a new-arch candidate pass its own engine (e.g. engines/bin/archon-sota-1536)
# — the reference is always evaluated with the engine that loads it.
# PASS band [0.85, 1.18]: generous vs measurement noise, tight vs the 2x/0.5x
# class of recipe bug. Exit 0 = PASS, 1 = FAIL, 2 = usage/infra error.
#
# RULE: every newly trained net passes this check BEFORE any gate match.
# ============================================================================
set -uo pipefail
cd "$(dirname "$0")/../.."

CAND="${1:?usage: net_scale_check.sh <candidate.cenn> [reference.cenn] [cand-engine]}"
REF="${2:-data/nnue/nets/d40-uho-720sb.cenn}"
CAND_ENGINE="${3:-engines/bin/archon-sota-native}"
REF_ENGINE="${REF_ENGINE:-engines/bin/archon-sota-native}"
FENS=tools/nnue/calib/scale-check-800.fens

for f in "$CAND" "$REF" "$FENS"; do
  [ -s "$f" ] || { echo "[scale-check] missing: $f"; exit 2; }
done
for e in "$CAND_ENGINE" "$REF_ENGINE"; do
  [ -x "$e" ] || { echo "[scale-check] engine not built: $e"; exit 2; }
done

TMP="$(mktemp -d "${TMPDIR:-/tmp}/scale-check.XXXXXX")"
trap 'rm -rf "$TMP"' EXIT

"$CAND_ENGINE" nnue-eval "$CAND" < "$FENS" > "$TMP/cand.ev" 2>"$TMP/cand.err" \
  || { echo "[scale-check] nnue-eval failed on candidate:"; tail -3 "$TMP/cand.err"; exit 2; }
"$REF_ENGINE" nnue-eval "$REF" < "$FENS" > "$TMP/ref.ev" 2>"$TMP/ref.err" \
  || { echo "[scale-check] nnue-eval failed on reference:"; tail -3 "$TMP/ref.err"; exit 2; }

python3 - "$TMP/cand.ev" "$TMP/ref.ev" "$CAND" "$REF" <<'PY'
import sys
cand = [int(l.split()[0]) for l in open(sys.argv[1]) if l.strip()]
ref  = [int(l.split()[0]) for l in open(sys.argv[2]) if l.strip()]
assert len(cand) == len(ref) and len(cand) >= 700, f"row mismatch {len(cand)} vs {len(ref)}"
# Robust ratio: median of cand/ref where the reference is confidently nonzero.
pairs = [(c, r) for c, r in zip(cand, ref) if abs(r) > 100]
ratios = sorted(c / r for c, r in pairs)
med = ratios[len(ratios) // 2]
q1, q3 = ratios[len(ratios) // 4], ratios[3 * len(ratios) // 4]
# Zero-intercept regression slope as the second estimator.
slope = sum(c * r for c, r in pairs) / sum(r * r for _, r in pairs)
# Sign agreement guards against a POV/sign bug masquerading as scale ~ -1.
sign = sum(1 for c, r in pairs if c * r > 0) / len(pairs)
print(f"[scale-check] candidate={sys.argv[3]}")
print(f"[scale-check] reference={sys.argv[4]}")
print(f"[scale-check] n={len(pairs)} (|ref|>100)  median(cand/ref)={med:.3f}  IQR=({q1:.2f},{q3:.2f})  slope={slope:.3f}  sign-agree={100*sign:.1f}%")
ok = 0.85 <= med <= 1.18 and 0.85 <= slope <= 1.18 and sign >= 0.95
print(f"[scale-check] {'PASS' if ok else 'FAIL'} (band 0.85-1.18, sign>=95%)")
sys.exit(0 if ok else 1)
PY
