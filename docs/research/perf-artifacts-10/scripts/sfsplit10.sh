#!/usr/bin/env bash
# perf study 10 — annotate SF18's big-net `evaluate` and split it by OPCODE
# CLASS, which is what makes the SF-side accounting possible at all.
#
#   bash sfsplit10.sh <perf.data> <out.txt>
#
# SF inlines its whole accumulator chain into `Network<1024>::evaluate`, so a
# per-symbol profile cannot say how much of that 28% is the threat fold, the
# HalfKA fold, or the head.  The opcodes separate them cleanly, because each
# stage has a signature instruction:
#
#   vpmovsxbw / vpdpbusd on an i8 column  -> FullThreats fold (i8 weights)
#   vpaddw / vpsubw on i16 columns        -> HalfKA fold + the FT merge
#   vpdpbusd                              -> fc_0 sparse matmul (the head)
#   vpmaddubsw / vpmulhrsw / vpmaxsw      -> the transform's clamp/pairwise
#
# The threat fold is the only consumer of i8 -> i16 widening in SF's eval, so
# `vpmovsxbw`'s share is a lower bound on it that needs no source reading.
set -euo pipefail
DATA=${1:?perf.data}
OUT=${2:?out}
rep=$(mktemp)
perf report -i "$DATA" --stdio --no-children -s sym 2>/dev/null > "$rep"
# `grep -m1 | sed` down a pipe from perf makes perf die on SIGPIPE under
# `set -o pipefail` (exit 141) — materialise the report first.
SYM=$(grep 'NetworkArchitecture<1024u' "$rep" | grep evaluate | head -1 \
      | sed 's/^ *[0-9.]*% *\[\.\] *//')
rm -f "$rep"
[ -n "$SYM" ] || { echo "big-net evaluate symbol not found" >&2; exit 1; }
perf annotate -i "$DATA" --stdio -s "$SYM" --percent-limit 0 2>/dev/null > "$OUT"
wc -l < "$OUT"
