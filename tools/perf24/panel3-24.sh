#!/usr/bin/env bash
# tools/perf24/panel3-24.sh <baseline> <identical-copy-of-baseline> <arm> [rounds] [frame]
#
# THE ROUND-ROBIN PANEL. The null runs INTERLEAVED WITH THE ARM, not before it.
#
# Why (coordinator correction, 2026-07-29, corroborated independently here):
#   * study 22 ran four nulls of one binary against its own byte-identical copy
#     in a single session and got -3.40 / -0.07 / -1.51 / +1.42 percent -- a
#     4.8-point spread on a change that does not exist -- and retracted a
#     committed -1.16% headline over it;
#   * this study's own fresh null (`base` vs `nullB`, bit-identical source,
#     n=8, fixed A,B interleave) read -2.43% / -3.80%, which is the SAME size
#     as both arms it had just "measured";
#   * the coordinator's verification panel read a 35.8% cycles spread.
#
# A null taken before the arm characterises a DIFFERENT MACHINE STATE than the
# arm runs in: a live 400-superbatch trainer plus three sibling studies move
# this box several percent over tens of minutes. So all three binaries are
# cycled TOGETHER, one round each, and the B->B' ratio is the live null for the
# B->X ratio measured in the very same rounds.
#
# Order rotates each round (ABX, BXA, XAB, ...) so no slot is systematically
# first. MEDIAN only -- never min-of-N: study 22 measured min-of-N at -2.07% on
# two identical binaries.
#
# `instructions:u` and `branches:u` are EXACT at a fixed node count (this
# study's null: 0.0000% on both) and are the gate for everything in this block:
# study 19 measured `at_full` at only 4.69% of cache misses, so it is
# instruction-dominated, not stall-dominated.
set -euo pipefail
cd "$(dirname "$0")/../.."
ROOT=$PWD

A=${1:?usage: panel3-24.sh <base> <base-copy> <arm> [rounds] [frame]}
B=${2:?}
X=${3:?}
rounds=${4:-9}
frame=${5:-bench16}
CPU=${P24_CPU:-3}

NET=$ROOT/engines/canonical/net.cenn
FENS=$ROOT/tools/nnue/calib/wl50.fens
case $frame in
    bench16) args=(bench 16 1 16 default depth "$NET") ;;
    wl100k)  args=(bench 16 1 100000 "$FENS" nodes "$NET") ;;
    *) echo "unknown frame $frame" >&2; exit 2 ;;
esac

EV=instructions:u,cycles:u,branches:u,L1-dcache-loads:u
tmp=$(mktemp -d); trap 'rm -rf "$tmp"' EXIT

run() { # <label> -> "cyc ins br ld nodes"
    taskset -c "$CPU" perf stat -x, -e "$EV" -o "$tmp/s" \
        "$ROOT/perf24/bin/$1" "${args[@]}" >/dev/null 2>"$tmp/b"
    awk -F, '$3~/^cycles/{c=$1} $3~/^instructions/{i=$1} $3~/^branches/{b=$1}
             $3~/^L1-dcache-loads/{l=$1} END{printf "%s %s %s %s", c,i,b,l}' "$tmp/s"
    sed -n 's/^Nodes searched  : / /p' "$tmp/b" | tr -d '\n'
    echo
}

: > "$tmp/rows"
for r in $(seq 1 "$rounds"); do
    case $((r % 3)) in
        1) order="$A $B $X" ;;
        2) order="$B $X $A" ;;
        0) order="$X $A $B" ;;
    esac
    declare -A cyc ins br ld
    nodes=
    for lbl in $order; do
        read -r c i b l n <<<"$(run "$lbl")"
        cyc[$lbl]=$c; ins[$lbl]=$i; br[$lbl]=$b; ld[$lbl]=$l
        if [ -z "$nodes" ]; then nodes=$n
        elif [ "$nodes" != "$n" ]; then echo "NODE MISMATCH round $r ($lbl): $n vs $nodes" >&2; exit 3; fi
    done
    echo "${cyc[$A]} ${cyc[$B]} ${cyc[$X]} ${ins[$A]} ${ins[$B]} ${ins[$X]} ${br[$A]} ${br[$B]} ${br[$X]} ${ld[$A]} ${ld[$B]} ${ld[$X]} $nodes" >> "$tmp/rows"
    awk -v r="$r" -v a="${cyc[$A]}" -v b="${cyc[$B]}" -v x="${cyc[$X]}" \
        'BEGIN{printf "  round %2d   NULL %+7.3f%%   ARM %+7.3f%%\n", r, (b/a-1)*100, (x/a-1)*100}'
done
cp "$tmp/rows" "$ROOT/perf24/panel3-$A-$X-$frame.txt"

awk -v A="$A" -v B="$B" -v X="$X" -v f="$frame" '
function med(v, k,   mid) { asort(v); mid=int((k+1)/2);
    return (k%2) ? v[mid] : (v[k/2]+v[k/2+1])/2 }
{ n++; nu[n]=$2/$1; ar[n]=$3/$1;
  ia=$4; ib=$5; ix=$6; ba=$7; bb=$8; bx=$9; la=$10; lb=$11; lx=$12; nodes=$13 }
END{
  printf "\nROUND-ROBIN PANEL  %s (base) / %s (identical copy) / %s (arm)  [%s, n=%d, nodes=%s]\n",
         A, B, X, f, n, nodes;
  printf "  CYCLES  live NULL  median %+7.3f%%   spread %+7.3f%% .. %+7.3f%%\n",
         (med(nu,n)-1)*100, (nu[1]-1)*100, (nu[n]-1)*100;
  printf "  CYCLES  ARM        median %+7.3f%%   spread %+7.3f%% .. %+7.3f%%\n",
         (med(ar,n)-1)*100, (ar[1]-1)*100, (ar[n]-1)*100;
  printf "  INSTR   null %+8.4f i/n     ARM %+8.4f i/n   (%.4f -> %.4f)\n",
         (ib-ia)/nodes, (ix-ia)/nodes, ia/nodes, ix/nodes;
  printf "  BRANCH  null %+8.4f b/n     ARM %+8.4f b/n   (%.4f -> %.4f)\n",
         (bb-ba)/nodes, (bx-ba)/nodes, ba/nodes, bx/nodes;
  printf "  L1LOAD  null %+8.4f l/n     ARM %+8.4f l/n   (%.4f -> %.4f)\n",
         (lb-la)/nodes, (lx-la)/nodes, la/nodes, lx/nodes;
}' "$tmp/rows"
