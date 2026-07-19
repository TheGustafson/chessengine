#!/usr/bin/env bash
# Faithful n-gate (t-shirt S/M/L) auto research loop — PHASE 1: the SMALL tier
# (the immediately-testable speed lever). Self-firing: waits for the cheap
# HalfKA-only net, then sweeps (thresh × mat_gate) and measures, per cell:
#   - NPS (bench, gate ON vs OFF): does the gate actually cut eval cost?
#   - equal-time 10+0.1 Elo vs uniform-medium (the SHIP metric).
# Baseline is ALWAYS uniform-medium (all tiers off) = bit-identical eval, so a
# positive result is the ROUTING, not a net swap. The equal-time + NPS arms need
# a quiet box, so we SIGSTOP the relabel workers for the sweep and SIGCONT after.
# Design + honest caveats: docs/research/n-gate-tshirt-design.md
set -uo pipefail
cd "$(dirname "$0")/../.."

MEDIUM="${ARCHON_NGATE_MEDIUM:-data/nnue/nets/d40-uho-720sb.cenn}"
SMALL="${ARCHON_NGATE_SMALL:-data/nnue/nets/d40-uho-halfka-cheap-240sb.cenn}"
EXP=data/nnue/experiments/n-gate
JOURNAL="$EXP/journal.md"
ENG=target/release/chessengine
BOOK=data/nnue/openings/policy-balanced-256.epd
GAMES="${ARCHON_NGATE_GAMES:-200}"
mkdir -p "$EXP"
say(){ echo "[n-gate $(date -u +%H:%M:%S)] $*" | tee -a "$EXP/loop.log"; }
jrn(){ echo "$*" >> "$JOURNAL"; }

# --- prerequisites -----------------------------------------------------------
[ -f "$JOURNAL" ] || { echo "# N-gate research journal"; echo; echo "Baseline = uniform-medium ($MEDIUM), gate OFF. Positive = the routing helps."; echo; } > "$JOURNAL"
say "waiting for the cheap net: $SMALL"
for _ in $(seq 1 720); do [ -f "$SMALL" ] && break; sleep 60; done
if [ ! -f "$SMALL" ]; then say "ERROR: cheap net never appeared — is the cheap-net train done? Aborting."; exit 1; fi
if [ ! -f "$MEDIUM" ]; then say "ERROR: medium net $MEDIUM missing."; exit 1; fi
# Build the sota engine fresh (the gate lives behind --features sota).
say "building the sota engine"
RUSTFLAGS="-C target-cpu=x86-64-v3" cargo build --release --features sota >/dev/null 2>&1 || { say "build failed"; exit 1; }

# --- quiet the box: pause the relabel's SF18 workers for clean timing ---------
pause_relabel(){ pkill -STOP -f relabel_depth 2>/dev/null; pkill -STOP stockfish 2>/dev/null; say "paused relabel workers (quiet box)"; sleep 2; }
resume_relabel(){ pkill -CONT stockfish 2>/dev/null; pkill -CONT -f relabel_depth 2>/dev/null; say "resumed relabel workers"; }
trap resume_relabel EXIT

nps_of(){ # $1=extra env string  -> prints NPS. NB: bench prints to STDERR, so 2>&1.
    env $1 "$ENG" bench 16 1 8 default depth "$MEDIUM" 2>&1 | awk '/Nodes\/second/{print $3}'
}

equal_time_elo(){ # $1=thresh $2=mat_gate $3=tag  -> appends Elo to journal
    local out="$EXP/$3"; mkdir -p "$out"
    local wrap="$out/cand.sh"
    cat > "$wrap" <<SH
#!/usr/bin/env bash
export ARCHON_SMALLNET=$(pwd)/$SMALL
export ARCHON_SMALLNET_THRESH=$1
export ARCHON_SMALLNET_MAT_GATE=$2
exec $(pwd)/$ENG "\$@"
SH
    chmod +x "$wrap"
    engines/bin/fastchess \
        -engine cmd="$wrap" name=gate-on option.EvalFile="$MEDIUM" \
        -engine cmd="$(pwd)/$ENG" name=gate-off option.EvalFile="$MEDIUM" \
        -each proto=uci tc=10+0.1 option.Threads=1 option.Hash=16 \
        -openings file="$BOOK" format=epd order=random \
        -repeat -games 2 -rounds $((GAMES/2)) -concurrency 6 -recover \
        -draw movenumber=40 movecount=8 score=15 -resign movecount=4 score=700 -maxmoves 250 \
        -srand 20260719 -pgnout file="$out/games.pgn" > "$out/match.log" 2>&1
    grep -E "Elo:|Games:" "$out/match.log" | tail -2
}

# --- the sweep ---------------------------------------------------------------
say "SMALL-tier sweep on medium=$MEDIUM small=$SMALL ($GAMES games/cell equal-time)"
jrn "## SMALL-tier sweep ($(date -u +%Y-%m-%dT%H:%MZ))"; jrn ""
jrn "| thresh | mat_gate | NPS on/off | equal-time Elo vs uniform |"
jrn "|---|---|---|---|"
BASE_NPS=$(nps_of "")
say "baseline (gate off) NPS=$BASE_NPS"

pause_relabel
for thresh in 300 500 700; do
  for mat in 0 300 500; do
    tag="t${thresh}_m${mat}"
    on_nps=$(nps_of "ARCHON_SMALLNET=$(pwd)/$SMALL ARCHON_SMALLNET_THRESH=$thresh ARCHON_SMALLNET_MAT_GATE=$mat")
    say "cell $tag: NPS on=$on_nps off=$BASE_NPS"
    elo=$(equal_time_elo "$thresh" "$mat" "$tag" | grep "Elo:" | head -1)
    ratio=$(awk "BEGIN{printf \"%.2f\", $on_nps/$BASE_NPS}")
    jrn "| $thresh | $mat | ${on_nps}/${BASE_NPS} (${ratio}x) | ${elo:-n/a} |"
    say "cell $tag DONE: NPS ${ratio}x  $elo"
  done
done
resume_relabel; trap - EXIT

say "PHASE 1 sweep DONE. Journal: $JOURNAL"
jrn ""; jrn "Next: refine around any cell that is BOTH faster AND >= +8 Elo; ledger it. If all flat at equal time despite higher NPS -> cheap net too weak, retrain distilled-from-medium (design doc abort rule)."
touch "$EXP/PHASE1-DONE"
