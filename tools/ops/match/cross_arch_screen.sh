#!/usr/bin/env bash
# tools/ops/match/cross_arch_screen.sh — screen two nets that need DIFFERENT builds.
#
# PURPOSE
#   `screen.sh` drives one engine binary with two `EvalFile` values. That is
#   correct only while both nets share an architecture. A `.cenn` records its
#   `acc_width` and the loader REFUSES a mismatch:
#
#       CENN acc_width 768 != 1280 (architecture mismatch)
#
#   So comparing the 768 HalfKA-only net against a 1280 canonical needs one
#   binary per arch. This runs that comparison.
#
# USAGE
#   CAND_BIN=target/hka768/release/chessengine CAND_NET=...cenn \
#   REF_BIN=target-sota/release/chessengine   REF_NET=engines/canonical/net.cenn \
#   GAMES=1500 bash tools/ops/match/cross_arch_screen.sh
#
#   Environment:
#     CAND_BIN / CAND_NET   candidate build + net (REQUIRED)
#     REF_BIN               default target-sota/release/chessengine
#     REF_NET               default engines/canonical/net.cenn
#     GAMES 1500   NODES 25000   CONC 3   TAG (labels the output file)
#
# WHY FIXED NODES MAKES THIS FAIR DESPITE DIFFERENT BINARIES
#   Different builds have different SPEED (PGO, target-cpu, inlining), which at a
#   real clock would confound the net comparison with a build comparison. At a
#   fixed node budget both engines get the same number of nodes and speed drops
#   out entirely — so a non-PGO candidate can be compared to a PGO reference
#   without bias. **This tool therefore refuses to run with a TC.**
#
#   The corollary matters: a fixed-node win here says the NET is better per node.
#   It says NOTHING about wall-clock strength, and the 768 net's whole case is
#   that it is ~2x faster. That has to be measured separately, at a real clock,
#   with matched build types.

source "$(dirname "${BASH_SOURCE[0]}")/../lib/common.sh"

CAND_BIN=${CAND_BIN:?CAND_BIN required}
CAND_NET=${CAND_NET:?CAND_NET required}
REF_BIN=${REF_BIN:-target-sota/release/chessengine}
REF_NET=${REF_NET:-engines/canonical/net.cenn}
GAMES=${GAMES:-1500}
NODES=${NODES:-25000}
CONC=${CONC:-3}
TAG=${TAG:-$(basename "$CAND_NET" .cenn)}
FASTCHESS=$(asset engines/bin/fastchess)
BOOK=$(asset engines/books/UHO_Lichess_4852_v1.epd)
OPS_LOG=${OPS_LOG:-$ARCHON_D49/cross_arch.log}; export OPS_LOG

for v in CAND_BIN CAND_NET REF_BIN REF_NET; do
    p=${!v}; case "$p" in /*) ;; *) p="$ARCHON_ROOT/$p";; esac
    printf -v "$v" '%s' "$p"
done
[ -x "$CAND_BIN" ] || { log "FATAL: candidate binary not executable: $CAND_BIN"; exit 2; }
[ -x "$REF_BIN" ]  || { log "FATAL: reference binary not executable: $REF_BIN"; exit 2; }
[ -s "$CAND_NET" ] || { log "FATAL: candidate net missing: $CAND_NET"; exit 2; }
[ -s "$REF_NET" ]  || { log "FATAL: reference net missing: $REF_NET"; exit 2; }

# Prove each binary can actually LOAD its own net before spending games. The
# failure mode this catches is silent: fastchess reports 0 games and exits 0.
for pair in "CAND:$CAND_BIN:$CAND_NET" "REF:$REF_BIN:$REF_NET"; do
    which=${pair%%:*}; rest=${pair#*:}; b=${rest%%:*}; n=${rest#*:}
    if "$b" nnue-eval "$n" < tools/nnue/calib/wl50.fens 2>&1 | grep -qiE 'ERROR|mismatch|cannot'; then
        log "FATAL: $which binary cannot load its net"
        "$b" nnue-eval "$n" < tools/nnue/calib/wl50.fens 2>&1 | grep -iE 'ERROR|mismatch' | head -1 | sed 's/^/    /'
        exit 3
    fi
    sd=$("$b" nnue-eval "$n" < tools/nnue/calib/wl50.fens 2>/dev/null | grep -oE '^-?[0-9]+' \
         | python3 -c "import sys,statistics as st;v=[int(x) for x in sys.stdin];print(f'{st.pstdev(v):.1f}')" 2>/dev/null)
    log "  $which ok: $(basename "$b") + $(basename "$n")  eval stdev ${sd:-?}"
done

log "CROSS-ARCH SCREEN: $TAG vs $(basename "$REF_NET") — $GAMES games @ ${NODES}n conc=$CONC"
OUT="$ARCHON_D49/xarch-${TAG}.out"
"$FASTCHESS" \
  -engine cmd="$CAND_BIN" name=cand proto=uci option.EvalFile="$CAND_NET" \
  -engine cmd="$REF_BIN"  name=ref  proto=uci option.EvalFile="$REF_NET" \
  -each nodes="$NODES" tc=inf \
  -openings file="$BOOK" format=epd order=random \
  -games 2 -rounds $((GAMES / 2)) -repeat -concurrency "$CONC" \
  -recover -ratinginterval 100 > "$OUT" 2>&1

{ echo "=== CROSS-ARCH $TAG vs $(basename "$REF_NET") (${NODES}n) ==="
  grep -A5 'Results of' "$OUT" | tail -5; } | tee -a "$OPS_LOG"
log "done -> $OUT"
