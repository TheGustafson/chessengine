#!/usr/bin/env bash
# SMP scaling — TIME-TO-DEPTH, which is the metric that actually matters.
#
# Why this exists. Perf study 7 §6 lists SMP scaling as an open,
# release-relevant unknown and says plainly: needs an idle box, do not attempt
# under load. The only numbers we have are from 2026-07-12 (strength-log row
# 24), and they are (a) PSQT-era, from before NNUE, S0 and the pruning work,
# and (b) mostly the wrong metric.
#
# NPS SCALING IS NOT SPEEDUP. Lazy SMP works by having threads search
# overlapping trees with different move orders and share a TT. Most of the
# extra nodes are DUPLICATED work, so nodes/second scales far better than
# time-to-depth does — 6.94x NPS at 8 threads (the number we have) is entirely
# compatible with a 2-3x real speedup. Quoting NPS scaling as SMP strength is
# the classic lazy-SMP error.
#
# What matters for Elo is how much faster the engine REACHES A GIVEN DEPTH,
# because that is what converts through the ~120 Elo/doubling constant
# (exp-0154). So this measures wall-clock to complete a fixed-depth bench at
# T threads and reports speedup against T=1.
#
# Wall clock is the one instrument we cannot avoid here — the whole question is
# about time — which is exactly why the box must be idle. The guard below is
# not advisory: a contended run produces a plausible-looking wrong curve, and
# an SMP curve that is wrong in the optimistic direction would send us shipping
# a bad default thread count.
#
#   bash tools/smp_scaling.sh [depth] [threads-list] [repeats]
#   bash tools/smp_scaling.sh 12 "1 2 4 8" 3
set -eu
cd "$(dirname "$0")/.."

DEPTH=${1:-12}
THREADS=${2:-"1 2 4 8"}
REPEATS=${3:-3}
NET=${NET:-data/nnue/nets/screens/d47v-a4b2-cal400.cenn}
FENS=${FENS:-tools/nnue/calib/wl50.fens}
BIN=${BIN:-engines/bin/archon-d47v-pgo}
HASH=${HASH:-256}   # generous: a too-small TT throttles SMP and understates it

# ── the idle-box guard ────────────────────────────────────────────────────
fail() { printf 'REFUSING: %s\n' "$*" >&2; exit 1; }
[ -f "$BIN" ] || fail "engine $BIN missing"
[ -f "$NET" ] || fail "net $NET missing"
systemctl --user is-active archon-d48x.service >/dev/null 2>&1 &&
    fail "the D48X trainer is running — SMP timing needs an idle box"
pgrep -f 'spsa_dist.py' >/dev/null 2>&1 &&
    fail "an SPSA campaign is running — SMP timing needs an idle box"
pgrep -f 'pump_train.py' >/dev/null 2>&1 &&
    fail "a trainer is running — SMP timing needs an idle box"
LOAD=$(cut -d' ' -f1 /proc/loadavg)
awk -v l="$LOAD" 'BEGIN { exit !(l > 2.0) }' &&
    fail "load average is $LOAD — wait for the box to settle below 2.0"

NCPU=$(nproc)
printf 'SMP scaling: depth %s, %s repeats, hash %sMB, %s logical cpus\n' \
    "$DEPTH" "$REPEATS" "$HASH" "$NCPU"
printf 'engine %s\nnet    %s\n\n' "$BIN" "$NET"

baseline=""
printf '%8s %12s %12s %10s %10s\n' threads "median ms" "spread" "speedup" "efficiency"
for t in $THREADS; do
    if [ "$t" -gt "$NCPU" ]; then
        printf '%8s  skipped (more threads than logical cpus)\n' "$t"
        continue
    fi
    times=""
    for _ in $(seq "$REPEATS"); do
        # bench's 2nd argument is the thread count. Report goes to stderr.
        ms=$("$BIN" bench "$HASH" "$t" "$DEPTH" "$FENS" depth "$NET" 2>&1 >/dev/null \
             | sed -n 's/^Total time (ms) : //p')
        times="$times $ms"
    done
    # median + spread; lazy SMP is non-deterministic across threads, so a
    # single run means nothing and the spread is part of the result.
    read -r med lo hi <<EOF
$(printf '%s\n' $times | sort -n | awk '{a[NR]=$1} END {printf "%d %d %d", a[int((NR+1)/2)], a[1], a[NR]}')
EOF
    [ -n "$baseline" ] || baseline=$med
    awk -v t="$t" -v m="$med" -v lo="$lo" -v hi="$hi" -v b="$baseline" 'BEGIN {
        sp = b / m
        printf "%8s %12d %12s %9.2fx %9.0f%%\n", t, m, lo "-" hi, sp, 100*sp/t
    }'
done

cat <<'EOF'

Reading this: "speedup" is time-to-depth against 1 thread, which is the number
that converts to Elo at ~120 per doubling. "efficiency" is speedup/threads —
lazy SMP is expected to fall well short of 100%, and 50-60% at 8 threads would
be healthy. Compare against NPS scaling (6.94x at 8t, PSQT era) to see how much
of that was duplicated work rather than progress.
EOF
