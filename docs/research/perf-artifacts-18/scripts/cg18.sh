#!/usr/bin/env bash
# perf study 18 (instrument inherited from 17) — the EXACT per-function instruction counter.
#
#   bash cg17.sh <label> <nodes-per-pos> <n-fens> [binary] [extra-args...]
#
# WHY THIS EXISTS. Every previous study attributed instructions to functions
# with `perf record -e instructions:u -c <period>` (`flat8.sh`) — a SAMPLING
# instrument, and study 16 §11 measured its per-opcode form wrong by 1.44x here.
# `callgrind` counts every executed instruction exactly and deterministically,
# per function, with no sampling and no skid, and it runs on ANY binary — so
# ARCHON and Stockfish can be put in the same frame without either engine being
# rebuilt or instrumented.
#
# THE FRAME. valgrind cannot execute AVX-512, so the ARCHON binary this script
# measures is built `-C target-cpu=x86-64-v3` (AVX2) into its own target dir.
# That is NOT the probe frame and NOT the PGO frame: use it for ATTRIBUTION
# (which function, how many instructions, how many calls) and never for a delta.
# Deltas come from probe17.sh.
#
# Exclusive Ir per function is what `callgrind_annotate` prints; call counts
# come from the `calls=` records in the raw file (cgcalls17.py).
set -euo pipefail
LABEL=${1:?label}
NODES=${2:?nodes per position}
NFENS=${3:?number of fens}
BIN=${4:-target-cg18/release/chessengine}
shift 3 || true
shift 1 || true

OUTDIR=docs/research/perf-artifacts-18
mkdir -p "$OUTDIR/cg"
FENS=$OUTDIR/cg/fens-$NFENS.txt
head -"$NFENS" tools/nnue/calib/wl50.fens > "$FENS"

OUT="$OUTDIR/cg/callgrind-$LABEL.out"
ENG="$OUTDIR/cg/callgrind-$LABEL.eng"

# taskset OUTSIDE valgrind: putting it inside makes valgrind profile `taskset`
# and exec the engine natively (6000 nodes for 412k Ir gave that away).
nice -n 19 taskset -c 6 valgrind --tool=callgrind --callgrind-out-file="$OUT" \
  --cache-sim=no --branch-sim=no --separate-threads=no \
  "$BIN" "$@" > "$ENG" 2>"$OUT.log"

# `bench` prints its summary on STDERR, which valgrind's log also captures.
nodes=$(grep -oP 'Nodes searched\s+:\s*\K[0-9]+' "$OUT.log" | tail -1)
total=$(grep -oP 'Collected\s*:\s*\K[0-9]+' "$OUT.log" | tail -1)
echo "label=$LABEL nodes=$nodes total_Ir=$total per_node=$(python3 -c "print(f'{$total/$nodes:.1f}')")"
callgrind_annotate --threshold=99.5 "$OUT" > "$OUTDIR/cg/annot-$LABEL.txt"
echo "wrote $OUTDIR/cg/annot-$LABEL.txt"
