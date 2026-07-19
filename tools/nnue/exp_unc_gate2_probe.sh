#!/usr/bin/env bash
# EXP-UNC-1 Gate 2: does consuming the value-uncertainty σ̂ help? A/B the SAME
# variance net (g2) with consumers ON vs OFF. Both engines load the net + the
# sidecar (D40_UNC_HEAD, process-wide); only the CANDIDATE gets nonzero coeffs
# (via a wrapper — fastchess has no per-engine env), so the baseline is
# consumers-off = bit-identical eval. Run AFTER g2 Gate 1 passes.
#
#   $1 net.cenn   $2 sidecar   $3 c1 (RFP)   $4 c2 (TM)   [$5 tag]
# Fixed-node isolates the RFP consumer's per-node effect (TM is time-only, ~noop
# at fixed nodes); equal-time (10+0.1) is the ship metric for both. The
# equal-time half needs a QUIET box — pause the relabel first (this script does
# NOT, so the caller controls timing).
set -uo pipefail
cd "$(dirname "$0")/../.."
NET="${1:?net.cenn}"; SIDE="${2:?sidecar}"; C1="${3:-0.3}"; C2="${4:-0.3}"; TAG="${5:-c1_${3}_c2_${4}}"
ENG=target/release/chessengine
OUT="data/nnue/experiments/d40-uho-unc-g2/gate2-$TAG"
BOOK=data/nnue/openings/policy-balanced-256.epd
mkdir -p "$OUT"
say(){ echo "[gate2-probe $(date -u +%H:%M:%S)] $*" | tee -a "$OUT/run.log"; }

# Candidate wrapper: same engine binary, consumers ARMED via env.
WRAP="$OUT/probe-unc.sh"
cat > "$WRAP" <<SH
#!/usr/bin/env bash
export ARCHON_UNC_RFP_C1=$C1
export ARCHON_UNC_TM_C2=$C2
exec $(pwd)/$ENG "\$@"
SH
chmod +x "$WRAP"

export D40_UNC_HEAD="$(pwd)/$SIDE"   # both engines load the variance head
say "net=$NET sidecar=$SIDE c1=$C1 c2=$C2 (candidate=consumers ON, baseline=OFF)"

run_match() { # $1 limit-flag  $2 tag  $3 games
    say "$2 match ($3 games)"
    engines/bin/fastchess \
        -engine cmd="$WRAP" name=unc-on option.EvalFile="$NET" \
        -engine cmd="$(pwd)/$ENG" name=unc-off option.EvalFile="$NET" \
        -each proto=uci $1 option.Threads=1 option.Hash=16 \
        -openings file="$BOOK" format=epd order=random \
        -repeat -games 2 -rounds $(($3 / 2)) -concurrency 1 -recover \
        -draw movenumber=40 movecount=8 score=15 -resign movecount=4 score=700 -maxmoves 250 \
        -srand 20260719 -pgnout file="$OUT/$2.pgn" \
        -log file="$OUT/$2-fc.log" level=warn > "$OUT/$2.log" 2>&1
    grep -E "Elo:|Score of|Games:" "$OUT/$2.log" | tail -3 | tee -a "$OUT/run.log"
}

run_match "nodes=5000" fixednode 600   # isolates the RFP consumer per-node
run_match "tc=10+0.1"  equaltime 300   # ship metric (BOTH consumers); QUIET box
say "GATE 2 DONE ($TAG). unc-on positive Elo = the uncertainty consumers help."
touch "$OUT/DONE"
