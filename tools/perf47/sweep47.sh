#!/usr/bin/env bash
# perf study 47 -- reciprocal-throughput sweep over the ops the hot kernels
# actually issue.  For each instruction, run the generated microbenchmark at
# 1..N independent chains and report cycles/op.
#
#     bash tools/perf47/sweep47.sh [insn ...]
#
#   NCH = 1      -> latency
#   NCH large    -> reciprocal throughput  (1 / effective pipes)
#   the knee     -> chains the machine wants = latency x throughput
#
# Method is study 46 s1.2's, with its two instructions kept as CONTROLS: this
# harness must reproduce 1.00 c/op for vfmadd231ps and ~0.55 for vpaddd or its
# new rows mean nothing.  min-of-REPS, pinned, nice'd.
set -euo pipefail
cd "$(dirname "$0")/../.."

S=${P47_SCRATCH:-/tmp/claude-1000/-home-nick-projects-chessengine/48ee83a2-f33c-4ab2-8e12-b12ca79a3d94/scratchpad}
BIN=$S/ops47
CPU=${P47_CPU:-0}
REPS=${P47_REPS:-5}
ITERS=${P47_ITERS:-2000000}
MAXCH=${P47_MAXCH:-10}

[ -x "$BIN" ] || { echo "build $BIN first (gen_ops47.py + gcc)"; exit 1; }

INSNS=${*:-"vfmadd231ps vpaddd vpmovsxbw vpaddw vpsubw"}

printf '%-16s' "insn"
for n in $(seq 1 "$MAXCH"); do printf '%8d' "$n"; done
printf '\n'

for insn in $INSNS; do
  printf '%-16s' "$insn"
  for n in $(seq 1 "$MAXCH"); do
    best=""
    for _ in $(seq 1 "$REPS"); do
      # perf prints cycles on stderr (-x,), the program prints the op count.
      out=$(nice -n 19 perf stat -x, -e cycles:u -- \
              taskset -c "$CPU" "$BIN" "$insn" "$n" "$ITERS" 2>&1 || true)
      ops=$(echo "$out" | grep -E '^[0-9]+$' | tail -1)
      cyc=$(echo "$out" | grep -F 'cycles:u' | cut -d, -f1)
      [ -n "$ops" ] && [ -n "$cyc" ] || continue
      cpo=$(awk -v c="$cyc" -v o="$ops" 'BEGIN{printf "%.4f", c/o}')
      if [ -z "$best" ] || awk -v a="$cpo" -v b="$best" 'BEGIN{exit !(a<b)}'; then best=$cpo; fi
    done
    printf '%8s' "${best:-  --}"
  done
  printf '\n'
done
