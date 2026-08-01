#!/bin/bash
# Basket-A autonomous pipeline (simplification plan stage 2).
# Runs as a detached systemd unit; every phase logs to this file's stdout.
#
# Phases:
#   1. WAIT for d49train inactive (the coordinator's hard build-freeze rule).
#   2. BUILD from the basket-A export (~/.cache/archon-simplify/a-src):
#      cargo test, default-release d6 bench (new Bench trailer number),
#      sota d13 wl50 + eval hash (hash MUST equal 2e1ede95..79c4 — search-only
#      change; abort loudly if it moved).
#   3. PGO candidate build (archon-simp-a) + d13 identity check vs phase 2.
#   4. WAIT for the box's matches to clear (pgrep -c fastchess == 0, two
#      consecutive polls) — the coordinator's d49tv Elo verdict runs first.
#   5. MATCH: stop d49xml if active; chain70 segments CAND vs BASE at fn25k
#      (UHO book, conc 2, MIN_AVAIL 2500); pool + GSPRT [-3, +1]; restart xml.
set -u
A=/home/nick/.cache/archon-simplify/a-src
WT=/home/nick/projects/chessengine/.claude/worktrees/agent-abed05f2117dec14a
NET=/home/nick/projects/chessengine/data/nnue/nets/screens/d48x-1280-sb520-ema-s600.cenn
FENS=tools/nnue/calib/wl50.fens
EXPECT_HASH=2e1ede9521624dd89eab4eb1bae0a1ea568cd6cf7e85b668179a7b5e93cf79c4

echo "== phase 1: waiting for d49train inactive $(date -Is)"
while [ "$(systemctl --user is-active d49train 2>/dev/null)" = "active" ]; do sleep 120; done
echo "== d49train inactive $(date -Is); settling 180s"
sleep 180
# Re-check: a relaunch loop may bring it back; require two consecutive reads.
while [ "$(systemctl --user is-active d49train 2>/dev/null)" = "active" ]; do
    echo "== trainer came back; waiting again $(date -Is)"; sleep 120
done

cd "$A" || exit 9
echo "== phase 2: build + pins $(date -Is)"
cargo test --quiet > /dev/null 2>&1 || { echo "BASKET-A TESTS FAILED"; exit 2; }
echo "tests green"
cargo build --release --quiet || exit 3
echo "-- new Bench (d6 default):"
./target/release/chessengine bench 16 1 6 default depth 2>&1 >/dev/null | grep 'Nodes searched'
cargo build --release --features sota,arch-d48x --target-dir target/nnue-gate --quiet 2>/dev/null || exit 4
echo "-- new d13 wl50+net:"
./target/nnue-gate/release/chessengine bench 16 1 13 "$FENS" depth "$NET" 2>&1 >/dev/null | grep 'Nodes searched'
hash=$(./target/nnue-gate/release/chessengine nnue-eval "$NET" < "$FENS" | sha256sum | cut -d' ' -f1)
echo "-- eval hash: $hash"
if [ "$hash" != "$EXPECT_HASH" ]; then
    echo "EVAL HASH MOVED — basket A touched eval. ABORT before any match."
    exit 5
fi
d13_plain=$(./target/nnue-gate/release/chessengine bench 16 1 13 "$FENS" depth "$NET" 2>&1 >/dev/null | sed -n 's/^Nodes searched  : //p')

echo "== phase 3: PGO candidate $(date -Is)"
export PGO_PROF_DIR=/home/nick/.cache/archon-simplify/pgo-prof-a
mkdir -p "$PGO_PROF_DIR" engines/bin
bash tools/build_pgo.sh sota,arch-d48x archon-simp-a "$NET" || exit 6
d13_pgo=$(./engines/bin/archon-simp-a bench 16 1 13 "$FENS" depth "$NET" 2>&1 >/dev/null | sed -n 's/^Nodes searched  : //p')
echo "-- PGO d13: $d13_pgo (non-PGO: $d13_plain)"
[ "$d13_pgo" = "$d13_plain" ] || { echo "PGO/non-PGO d13 mismatch — PGO build is not node-identical. ABORT."; exit 7; }
install -m755 engines/bin/archon-simp-a "$WT/engines/bin/archon-simp-a"
echo "candidate installed: $WT/engines/bin/archon-simp-a"

echo "== phase 4: waiting for the box's matches to clear $(date -Is)"
clear_count=0
while [ "$clear_count" -lt 2 ]; do
    n=$(pgrep -c fastchess 2>/dev/null || echo 0)
    if [ "${n:-0}" -eq 0 ]; then clear_count=$((clear_count+1)); else clear_count=0; fi
    sleep 90
done
echo "== fastchess clear $(date -Is)"

echo "== phase 5: match $(date -Is)"
cd "$WT" || exit 9
xml_was_active=0
if [ "$(systemctl --user is-active d49xml 2>/dev/null)" = "active" ]; then
    xml_was_active=1
    systemctl --user stop d49xml
    echo "d49xml stopped for the match window"
fi
MIN_AVAIL=2500 CONCURRENCY=2 NODES=25000 \
    BOOK=/home/nick/projects/chessengine/engines/books/UHO_Lichess_4852_v1.epd \
    bash tools/perf70/chain70.sh engines/bin/archon-simp-a engines/bin/archon-simp-base simp-a 16 125
echo "== pooled result:"
python3 tools/perf70/pool_sprt.py engines/runs/simp-a-*/sprt.out | tail -3
python3 /home/nick/.cache/archon-simplify/llr_pool.py --elo0 -3 --elo1 1 engines/runs/simp-a-*/sprt.out
if [ "$xml_was_active" = "1" ]; then
    bash /home/nick/.local/share/archon-d49/rearm_xml.sh 2>/dev/null \
        || systemctl --user start d49xml 2>/dev/null \
        || echo "NOTE: could not restart d49xml — restart it manually"
fi
echo "== pipeline complete $(date -Is)"
