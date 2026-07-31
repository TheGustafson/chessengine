#!/usr/bin/env bash
# perf study 47 -- WHICH `fp_ops_retired_by_type` BUCKET DOES EACH OP LAND IN?
#
# Study 46 closed study 44's assumption A4 by issuing a KNOWN number of 512-bit
# instructions and reading `pack_512_uops_retired` (ratio 1.0000, one event per
# instruction).  The same trick answers the question study 46 left open -- the
# engine's 512-bit OP MIX -- because Zen 4 also exposes the retired vector ops
# split BY TYPE (add/sub/mul/mac/cmp/cvt/shuffle/logical/blend/other).  If each
# hot instruction can be shown to land in a known bucket, then reading those
# buckets on the engine gives the dynamic mix EXACTLY, with no static modelling
# and no sampling.
#
#     bash tools/perf47/typesem47.sh [insn ...]
#
# Prints, per instruction: issued count, every non-zero bucket, and the ratio.
# A clean result is one bucket at ratio ~1.0000 and the rest at ~0.
set -euo pipefail
cd "$(dirname "$0")/../.."

S=${P47_SCRATCH:-/tmp/claude-1000/-home-nick-projects-chessengine/48ee83a2-f33c-4ab2-8e12-b12ca79a3d94/scratchpad}
BIN=$S/ops47
CPU=${P47_CPU:-0}
NCH=${P47_NCH:-8}
ITERS=${P47_ITERS:-1000000}

EV_A=fp_ops_retired_by_type.vector_add:u,fp_ops_retired_by_type.vector_sub:u,fp_ops_retired_by_type.vector_mul:u,fp_ops_retired_by_type.vector_mac:u,fp_ops_retired_by_type.vector_cmp:u,fp_ops_retired_by_type.vector_cvt:u
EV_B=fp_ops_retired_by_type.vector_shuffle:u,fp_ops_retired_by_type.vector_logical:u,fp_ops_retired_by_type.vector_blend:u,fp_ops_retired_by_type.vector_other:u,fp_ops_retired_by_type.vector_all:u,fp_ops_retired_by_width.pack_512_uops_retired:u

INSNS=${*:-"vfmadd231ps vpdpbusd vpaddd vpmovsxbw vpaddw vpsubw vpsllw vpmullw vpaddsw vpmaxsw vpminsw vpmovwb"}

for insn in $INSNS; do
  ops=""
  echo "=== $insn"
  for ev in "$EV_A" "$EV_B"; do
    out=$(nice -n 19 perf stat -x, -e "$ev" -- \
            taskset -c "$CPU" "$BIN" "$insn" "$NCH" "$ITERS" 2>&1 || true)
    [ -n "$ops" ] || ops=$(echo "$out" | grep -E '^[0-9]+$' | tail -1)
    echo "$out" | grep -F 'fp_' | while IFS=, read -r cnt _ name _; do
      awk -v n="$name" -v c="$cnt" -v o="$ops" \
        'BEGIN{ r = (o>0 ? c/o : 0); if (r > 0.001) printf "    %-52s %14d  ratio %.4f\n", n, c, r }'
    done
  done
  echo "    (issued $ops)"
done
