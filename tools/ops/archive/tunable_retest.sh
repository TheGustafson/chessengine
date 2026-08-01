#!/usr/bin/env bash
# THE TUNABLE RE-TEST, done properly this time.
#
# The first attempt (sprt68, -29.49 +/- 14.88) went through tools/sprt.sh when
# that tool passed NO EvalFile, so BOTH SIDES RAN PSQT. It measured an engine we
# never play. The revert d5f89245 rests on it, so the values have in fact never
# been tested.
#
# Two failures of that arm, both fixed here:
#   1. PSQT instead of NNUE  -> sprt.sh now defaults NET=engines/canonical/net.cenn
#   2. the pair differed by MORE than the tunables (study 68's SearchStats fix)
#      -> both binaries are built from TODAY'S main, differing ONLY in the 154 values
set -uo pipefail
MW=/home/nick/projects/chessengine-mainwt
LOG=/home/nick/projects/chessengine/data/nnue/experiments/d49-1280/cyc.log
cd "$MW"

waitmem() { for _ in $(seq 1 60); do
  A=$(awk '/^MemAvailable/{print $2}' /proc/meminfo); [ "$((A/1024))" -ge 3000 ] && return 0; sleep 60; done; return 0; }

# Don't build while a match is running.
while systemctl --user is-active --quiet d49len 2>/dev/null; do sleep 60; done

echo "[retest] building BASELINE (main, no tunables) $(date -Is)" >> "$LOG"
waitmem
git checkout -q main 2>/dev/null
CARGO_BUILD_JOBS=2 bash tools/build_pgo.sh sota,arch-d48x archon-notun engines/canonical/net.cenn >/dev/null 2>&1

echo "[retest] building CANDIDATE (main + 154 ebcea3ef tunables) $(date -Is)" >> "$LOG"
git branch -f tunables-retest main >/dev/null 2>&1
git checkout -q tunables-retest
git revert --no-edit -m 1 d5f89245 >/dev/null 2>&1 || git revert --no-edit d5f89245 >/dev/null 2>&1
waitmem
CARGO_BUILD_JOBS=2 bash tools/build_pgo.sh sota,arch-d48x archon-tun engines/canonical/net.cenn >/dev/null 2>&1
git checkout -q main

for b in archon-notun archon-tun; do
  printf "[retest] %s bench=" "$b" >> "$LOG"
  "$MW/engines/bin/$b" bench 16 1 6 default depth 2>&1 | grep 'Nodes searched  :' | awk '{print $4}' >> "$LOG"
done

waitmem
echo "[retest] SPRT with NNUE net, 1500 games @ 25k $(date -Is)" >> "$LOG"
OUT=/home/nick/.local/share/archon-d49/retest.out
NODES=25000 ROUNDS=750 CONCURRENCY=2 BOOK=engines/books/UHO_Lichess_4852_v1.epd \
  bash tools/sprt.sh "$MW/engines/bin/archon-tun" "$MW/engines/bin/archon-notun" > "$OUT" 2>&1
printf "[retest] RESULT (tunables vs none): " >> "$LOG"
grep -E '^Elo:' "$OUT" | tail -1 >> "$LOG"
grep -E '^Games:' "$OUT" | tail -1 >> "$LOG"
echo "[retest] done $(date -Is)" >> "$LOG"
