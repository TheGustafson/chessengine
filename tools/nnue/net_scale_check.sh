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
#
# ---------------------------------------------------------------------------
# THE RELATIVE CHECK ABOVE IS SELF-REFERENTIAL (found 2026-07-28).
#
# It compares each candidate to OUR OWN INCUMBENT, so it certifies that a net
# reproduces its ancestor's scale — it cannot see a scale that the whole lineage
# shares. And the whole lineage does share one: our eval measures **0.538x
# Stockfish's INTERNAL frame** (n=600), while every cp-threshold this header
# names was transcribed verbatim from Stockfish and is therefore ~1.86x too
# large in our units. Net-0's arbitrary scale became the lineage's scale, and
# this gate has been actively enforcing it ever since. The tighter it passes,
# the more perfectly the mismatch is preserved.
#
# So the check now ALSO reports an ABSOLUTE ratio against Stockfish's internal
# frame. Note that Stockfish does not PRINT that frame: `UCIEngine::to_cp`
# (eca43a97 uci.cpp:554) returns `100 * v / a` with `a = p_a(material)` ~ 385,
# so its printed cp must be multiplied by a/100 to recover the units its search
# constants actually live in. Comparing printed-cp to printed-cp measures
# nothing and inverts the apparent direction — the mistake that cost most of a
# night on 2026-07-28.
#
# ADVISORY, not fatal, and deliberately so: every net in the lineage fails it
# today, so hard-failing would block all training work. It is here to make the
# number impossible to not see. Set ABSOLUTE_STRICT=1 to make it fatal once the
# lineage has been rescaled.  Skipped silently if $SF_BIN is absent.
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
REL_RC=$?

# --- ABSOLUTE: against Stockfish's INTERNAL frame ---------------------------
SF_BIN="${SF_BIN:-engines/bin/stockfish18}"
if [ -x "$SF_BIN" ]; then
  # SF prints one "Final evaluation" line per `eval`, from WHITE's point of view.
  awk '{print "position fen "$0"\neval"} END{print "quit"}' "$FENS" \
    | "$SF_BIN" 2>/dev/null | grep -a "Final evaluation" > "$TMP/sf.ev" || true
  python3 - "$TMP/cand.ev" "$TMP/sf.ev" "$FENS" "$CAND" <<'PY'
import sys
ours = [int(l.split()[0]) for l in open(sys.argv[1]) if l.strip()]
sfl  = [l for l in open(sys.argv[2]) if "Final evaluation" in l]
fens = [l.strip() for l in open(sys.argv[3]) if l.strip()]

# a = p_a(material), the divisor UCIEngine::to_cp applies before printing.
AS = (-72.32565836, 185.93832038, -144.58862193, 416.44950446)
W = {"p": 1, "n": 3, "b": 3, "r": 5, "q": 9}
def a_of(fen):
    board = fen.split()[0]
    mat = sum(W.get(ch.lower(), 0) for ch in board if ch.isalpha())
    m = max(17, min(78, mat)) / 58.0
    return ((AS[0] * m + AS[1]) * m + AS[2]) * m + AS[3]

pairs = []
for fen, o, line in zip(fens, ours, sfl):
    tok = line.split()[2]
    try:
        white_cp = float(tok) * 100.0
    except ValueError:      # "none" for in-check positions
        continue
    # our nnue-eval is side-to-move relative; SF's printout is white-relative
    stm_cp = white_cp if fen.split()[1] == "w" else -white_cp
    internal = stm_cp * a_of(fen) / 100.0
    if abs(internal) > 100:
        pairs.append((o, internal))

if len(pairs) < 100:
    print(f"[scale-check] ABSOLUTE skipped (only {len(pairs)} usable rows)")
    sys.exit(0)
ratios = sorted(o / s for o, s in pairs)
med = ratios[len(ratios) // 2]
slope = sum(o * s for o, s in pairs) / sum(s * s for _, s in pairs)
print(f"[scale-check] ABSOLUTE vs SF INTERNAL frame (printed cp x a/100, a~385):")
print(f"[scale-check]   n={len(pairs)}  median(ours/sf)={med:.3f}  slope={slope:.3f}")
print(f"[scale-check]   -> transcribed cp constants are {1/slope:.2f}x too large "
      f"in this net's units; EVAL_SCALE parity point = {round(1024/slope)}")
good = 0.80 <= slope <= 1.25
if good:
    print("[scale-check]   ABSOLUTE PASS (band 0.80-1.25)")
else:
    print("[scale-check]   ABSOLUTE FAIL (band 0.80-1.25) — the net's frame does NOT")
    print("[scale-check]   match the frame our search constants were transcribed for.")
sys.exit(0 if good else 1)
PY
  ABS_RC=$?
  if [ "${ABSOLUTE_STRICT:-0}" = "1" ] && [ "$ABS_RC" -ne 0 ]; then
    echo "[scale-check] ABSOLUTE_STRICT=1 — failing on the absolute band."
    exit 1
  fi
fi
exit "$REL_RC"
