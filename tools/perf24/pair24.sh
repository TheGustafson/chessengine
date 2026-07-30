#!/usr/bin/env bash
# tools/perf24/pair24.sh <baseline-label> <arm-label> [rounds] [frame]
#
# THE CYCLES GATE (study 24's primary axis, per the commission) plus the two
# DETERMINISTIC microarchitectural axes.
#
# Why it is shaped like this -- the null panel, run first (§ doc 2):
#   * a SEQUENTIAL cycles probe of three bit-identical binaries read
#     4421 / 4493 / 4595 / 4708 c/n (+6.5%). Unusable, exactly as study 19 §2
#     found (+8.7% there).
#   * interleaving A,B,A,B + `taskset` to one core cut the n=6 null to
#     -0.36% / +0.02% / -0.55% ... but the SAME null at n=10 read +1.38%.
#     A wandering estimator, and study 19's rule 3 ("a null at a different n
#     is not a null") applies.
#   * the residual is FAST noise: consecutive A and B rounds are uncorrelated,
#     so the drift is sub-run-scale contention (three sibling studies build on
#     this box and a trainer holds 143% CPU), not slow thermal drift.
#
# Contention at fixed instructions is strictly ADDITIVE to cycles -- a stolen
# SMT sibling, an evicted line, a migrated page never make a run faster. So
# the informative rounds are the CHEAPEST ones, and the estimator reported
# first is the median over the least-contended half of the rounds
# ("clean-half median"). The plain median is printed beside it; when they
# disagree the arm is not resolvable on this axis and the doc says so.
#
# The last three axes are architectural events, exact at a fixed node count,
# and are the ones this study actually reads: study 19 measured that `at_full`
# owns only 4.69% of cache misses, so this block is instruction-bound.
set -euo pipefail
cd "$(dirname "$0")/../.."
ROOT=$PWD

base=${1:?usage: pair24.sh <base> <arm> [rounds] [frame]}
arm=${2:?usage: pair24.sh <base> <arm> [rounds] [frame]}
rounds=${3:-8}
frame=${4:-bench16}
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

one() { # <bin> -> "cyc ins br ld nodes"
    taskset -c "$CPU" perf stat -x, -e "$EV" -o "$tmp/s" \
        "$1" "${args[@]}" >/dev/null 2>"$tmp/b"
    awk -F, '$3~/^cycles/{c=$1} $3~/^instructions/{i=$1} $3~/^branches/{b=$1}
             $3~/^L1-dcache-loads/{l=$1} END{printf "%s %s %s %s", c,i,b,l}' "$tmp/s"
    sed -n 's/^Nodes searched  : / /p' "$tmp/b" | tr -d '\n'
    echo
}

: > "$tmp/rows"
for r in $(seq 1 "$rounds"); do
    read -r cA iA bA lA nA <<<"$(one "$ROOT/perf24/bin/$base")"
    read -r cB iB bB lB nB <<<"$(one "$ROOT/perf24/bin/$arm")"
    [ "$nA" = "$nB" ] || { echo "NODE MISMATCH round $r: $nA vs $nB" >&2; exit 3; }
    echo "$cA $cB $iA $iB $bA $bB $lA $lB $nA" >> "$tmp/rows"
    awk -v r="$r" -v a="$cA" -v b="$cB" -v n="$nA" \
        'BEGIN{printf "  round %2d  A %.1f c/n  B %.1f c/n   cyc %.5f\n", r, a/n, b/n, b/a}'
done
cp "$tmp/rows" "$ROOT/perf24/rows-$base-$arm-$frame.txt"

awk -v base="$base" -v arm="$arm" -v f="$frame" '
function med(v, k,   mid) { asort(v); mid=int((k+1)/2);
    return (k%2) ? v[mid] : (v[k/2]+v[k/2+1])/2 }
{ n++; ca[n]=$1; cb[n]=$2; ia=$3; ib=$4; ba=$5; bb=$6; la=$7; lb=$8; nodes=$9
  tot[n]=$1+$2; cr[n]=$2/$1 }
END{
  # clean-half: the rounds whose A+B cycle total is in the lowest half
  for (i=1;i<=n;i++) s[i]=tot[i]; asort(s); half=s[int((n+1)/2)];
  k=0; for (i=1;i<=n;i++) if (tot[i]<=half) { k++; clean[k]=cr[i] }
  printf "\n%s -> %s  [%s, n=%d, nodes=%s]\n", base, arm, f, n, nodes;
  printf "  CYCLES  clean-half median %.5f (%+.3f%%)  [k=%d]\n", med(clean,k), (med(clean,k)-1)*100, k;
  printf "  CYCLES  all-round  median %.5f (%+.3f%%)   spread %.5f..%.5f\n",
         med(cr,n), (med(cr,n)-1)*100, cr[1], cr[n];
  printf "  INSTR   %.4f -> %.4f i/n   ratio %.6f (%+.4f%%)   delta %+.2f i/n\n",
         ia/nodes, ib/nodes, ib/ia, (ib/ia-1)*100, (ib-ia)/nodes;
  printf "  BRANCH  %.4f -> %.4f b/n   ratio %.6f (%+.4f%%)   delta %+.2f b/n\n",
         ba/nodes, bb/nodes, bb/ba, (bb/ba-1)*100, (bb-ba)/nodes;
  printf "  L1LOAD  %.4f -> %.4f l/n   ratio %.6f (%+.4f%%)   delta %+.2f l/n\n",
         la/nodes, lb/nodes, lb/la, (lb/la-1)*100, (lb-la)/nodes;
}' "$tmp/rows"
