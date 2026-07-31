#!/usr/bin/env bash
# perf study 61 — identity gate for a set of labelled binaries.
#
#   bash tools/perf61/ident61.sh <label> [<label> ...]
#
# Prints, per binary: node total on the study frame, the wl50 eval hash, the
# `.text` byte size, and `instructions:u`.
#
# WHY `.text` SIZE AND NOT ITS md5: study 55's md5 trick strips objdump's
# filename banner, but the disassembly listing ALSO carries symbol names in its
# branch-target annotations, and a symbol name embeds the crate metadata hash —
# which changes whenever the Cargo FEATURE SET changes, even for byte-identical
# code. Promoting an arm from `#[cfg(feature = "x")]` to unconditional therefore
# moves the md5 while the code does not. Size plus an exact `instructions:u`
# comparison is the check that survives that (perf study 61).
set -euo pipefail
cd "$(dirname "$0")/../.."
CPU=${P61_CPU:-2}
NET=engines/canonical/net.cenn
printf '%-10s %10s  %-64s %9s %12s\n' label nodes eval-hash text-bytes instructions
for b in "$@"; do
  out=$(taskset -c "$CPU" perf stat -x, -e instructions:u ./bin61/"$b" \
        bench 16 1 100000 tools/nnue/calib/wl50.fens nodes "$NET" 2>&1)
  nodes=$(printf '%s\n' "$out" | sed -n 's/^Nodes searched  *: *//p')
  insn=$(printf '%s\n' "$out" | awk -F, '$3=="instructions:u"{print $1}')
  hash=$(./bin61/"$b" nnue-eval "$NET" < tools/nnue/calib/wl50.fens | sha256sum | cut -d' ' -f1)
  text=$(size -A "bin61/$b" | awk '$1==".text"{print $2}')
  printf '%-10s %10s  %-64s %9s %12s\n' "$b" "$nodes" "$hash" "$text" "$insn"
done
