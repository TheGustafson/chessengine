#!/usr/bin/env bash
# evolution_panel.sh — the ONE standard per-checkpoint read for a training run:
# pack -> sweep -> Spearman table WITH EXPLICIT BASES -> policy top-1.
#
# Exists because of the 2026-07-26 basis trap (third strike): a sharp-set
# Spearman quoted without its basis (non-check n=1817 vs all n=2399) produced
# a phantom "erosion" verdict. This tool prints every basis separately and
# refuses to emit an unlabeled number. The play-relevant basis is NON-CHECK
# (src/search/mod.rs:2327 — the value head is never consulted in check).
#
#   bash tools/nnue/evolution_panel.sh <ema-quant.bin | packed.cenn> <tag>
#
# Env (defaults = the D47V arc):
#   PANEL_ARCH   pack env string        (default "D40_ACC=768 D40_L2=32 D40_L3=32 D40_BUCKETS=8 D40_POLICY=1")
#   PANEL_ENGINE engine for the net     (default engines/bin/archon-d47v)
#   PANEL_TEACH  teacher pack for rank agreement (default the sb230 scratchpad pack; RAW frame is fine — rank-invariant)
#   PANEL_TEACH_ENGINE                  (default engines/bin/archon-d47)
#   Oracle labels: cached SF18@50k keyed by position-set content hash.
# Position sets (persisted): data/nnue/evalbench/sets/{evalbench_mixed,leafduty}.fens
set -uo pipefail
cd "$(dirname "$0")/../.."

SRC="${1:?usage: evolution_panel.sh <quant.bin|.cenn> <tag>}"
TAG="${2:?tag, e.g. sb30}"
PANEL_ARCH="${PANEL_ARCH:-D40_ACC=768 D40_L2=32 D40_L3=32 D40_BUCKETS=8 D40_POLICY=1}"
PANEL_ENGINE="${PANEL_ENGINE:-engines/bin/archon-d47v}"
PANEL_TEACH="${PANEL_TEACH:-/tmp/claude-1000/-home-nick-projects-chessengine/61b8a705-638b-4a89-a73b-e4fc2a5170a2/scratchpad/d47s-newest.cenn}"
PANEL_TEACH_ENGINE="${PANEL_TEACH_ENGINE:-engines/bin/archon-d47}"
QUIET=data/nnue/evalbench/sets/evalbench_mixed.fens
SHARP=data/nnue/evalbench/sets/leafduty.fens
ORACLE=engines/bin/stockfish18

CENN="$SRC"
if [[ "$SRC" != *.cenn ]]; then
  CENN="/tmp/panel-$TAG.cenn"
  env $PANEL_ARCH .venv/bin/python tools/nnue/netpack.py "$SRC" "$CENN" d40 2>&1 | tail -1
fi

# Sweeps (content-hash cached; a re-sweep of an IDENTICAL path with different
# bytes must purge first — new tags use new paths, so the default is safe).
for SET in "$QUIET" "$SHARP"; do
  ARCHON_BIN="$PANEL_ENGINE" .venv/bin/python tools/nnue/eval_bench.py \
    --positions "$SET" --count 3000 --oracle "$ORACLE" --oracle-nodes 50000 \
    --eval "panel-$TAG=archon:$CENN" >/dev/null 2>&1
done
# Teacher sweeps ride the same cache (pays once, ever).
for SET in "$QUIET" "$SHARP"; do
  ARCHON_BIN="$PANEL_TEACH_ENGINE" .venv/bin/python tools/nnue/eval_bench.py \
    --positions "$SET" --count 3000 --oracle "$ORACLE" --oracle-nodes 50000 \
    --eval "teacher=archon:$PANEL_TEACH" >/dev/null 2>&1
done

CENN="$CENN" TAG="$TAG" TEACH="$PANEL_TEACH" QUIET="$QUIET" SHARP="$SHARP" \
.venv/bin/python - <<'PY'
import hashlib, json, os
import numpy as np, chess
from pathlib import Path

cache = Path("data/nnue/evalbench")
def get(kind, key):
    p = cache / f"{kind}-{hashlib.sha256(key.encode()).hexdigest()[:16]}.json"
    return json.loads(p.read_text()) if p.exists() else None
def vec(v):
    return np.array([np.nan if e is None else float(e) for e in v])
def spear(a, b, m0):
    m = m0 & ~(np.isnan(a) | np.isnan(b))
    if m.sum() < 10:
        return float("nan"), int(m.sum())
    xr = np.argsort(np.argsort(a[m])).astype(float)
    yr = np.argsort(np.argsort(b[m])).astype(float)
    return float(np.corrcoef(xr, yr)[0, 1]), int(m.sum())

def err_corr(x, t, L, m0):
    """corr(student_err, teacher_err) in rank space (transform-invariant).
    The Layer-2 exhaustion tripwire (distillation endgame protocol,
    2026-07-26): oracle metrics PLATEAUED while this keeps CLIMBING means
    the remaining distillation gradient is pure teacher-bias absorption —
    time for the alpha ablation. While this sits high (~0.85+) the student
    cannot meaningfully exceed the teacher vs the oracle."""
    m = m0 & ~(np.isnan(x) | np.isnan(t) | np.isnan(L))
    if m.sum() < 10:
        return float("nan")
    def zr(a):
        r = np.argsort(np.argsort(a[m])).astype(float)
        return (r - r.mean()) / r.std()
    eS = zr(x) - zr(L)
    eT = zr(t) - zr(L)
    return float(np.corrcoef(eS, eT)[0, 1])

tag, cenn, teach = os.environ["TAG"], os.environ["CENN"], os.environ["TEACH"]
print(f"═══ EVOLUTION PANEL {tag} ═══  (net {cenn})")
for path, name in ((os.environ["QUIET"], "QUIET"), (os.environ["SHARP"], "SHARP")):
    fens = [l.strip() for l in open(path) if l.strip()][:3000]
    pk = hashlib.sha256("\n".join(fens).encode()).hexdigest()[:16]
    L = vec(get("lab", f"{pk}|engines/bin/stockfish18|50000"))
    if np.all(np.isnan(L)):
        print(f"{name}: NO CACHED LABELS for this set hash — label first"); continue
    x = vec(get("ev", f"{pk}|archon|{cenn}"))
    t = vec(get("ev", f"{pk}|archon|{teach}"))
    boards = [chess.Board(f) for f in fens]
    chk = np.array([b.is_check() for b in boards])
    pieces = np.array([chess.popcount(b.occupied) for b in boards])
    cuts = [("all", np.ones(len(fens), bool))]
    if chk.any():
        cuts = [("non-check", ~chk), ("in-check(NOT play-relevant)", chk), ("all", np.ones(len(fens), bool))]
    # the worst slice (2026-07-26 analysis): late endgame, capacity-triage
    # canary for the D47U fork — non-check basis
    cuts.append(("endgame<=8pc non-chk", (~chk) & (pieces <= 8)))
    for cut, m0 in cuts:
        ro, n = spear(x, L, m0)
        rt, _ = spear(x, t, m0)
        to, _ = spear(t, L, m0)
        ec = err_corr(x, t, L, m0)
        print(f"  {name} [{cut:>26s} n={n:4d}]  vs-oracle {ro:.4f}   vs-teacher {rt:.4f}   err-corr {ec:.4f}   (teacher-vs-oracle {to:.4f})")
PY

# Policy top-1 (engine probe; needs the .cpol sidecar the pack emits).
ENG="$PANEL_ENGINE" bash tools/nnue/policy_accuracy_probe.sh "$CENN" || true
echo "═══ panel $TAG done — quote NOTHING from this without its basis+n ═══"
