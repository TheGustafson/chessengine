#!/usr/bin/env bash
# tools/nnue/verify_forward.sh — the P13 step-1 forward-pass bit-oracle
# (design/11 §3.4.3, §11 step 1 GATE).
#
# Builds deterministic CENN test nets, runs BOTH the Rust engine's `nnue-eval`
# subcommand and the independent Python reference (tools/nnue/nnue_ref.py) over
# 100+ varied positions, and asserts bit-for-bit equality on every one. Exits
# non-zero on any mismatch. Covers both activations (SCReLU + CReLU).
#
# The Rust serializer is the single source of net bytes (both readers load the
# same .cenn file), so this checks the two forward-pass *implementations*, not
# two serializers. The vs-`bullet` cross-check is a separate, later oracle
# (step 3) that folds in once a real trained net exists — this script never
# fabricates one.
#
# Python runs under uv (design/11 §11.1). Nets/FENs live in a temp dir and are
# never committed (binary nets stay out of git, design/11 §3.4.5).
set -euo pipefail
cd "$(dirname "$0")/.."/..   # repo root

BIN=target/release/chessengine
REF=tools/nnue/nnue_ref.py

echo "==> building release engine"
cargo build --release >/dev/null 2>&1
[ -x "$BIN" ] || { echo "FAIL: $BIN missing" >&2; exit 1; }

WORK=$(mktemp -d)
trap 'rm -rf "$WORK"' EXIT
FENS="$WORK/fens.txt"

# Clean FEN corpus: the tracked corpus (comments/blanks stripped) plus a
# supplement of known-legal positions (drawn from the bench suite) so the set
# clears 100 with endgames, in-check nodes, promotions, and castling variety.
grep -v '^[[:space:]]*#' tests/data/fens.txt | grep -v '^[[:space:]]*$' > "$FENS"
cat >> "$FENS" <<'EOF'
r3k2r/p1ppqpb1/bn2pnp1/3PN3/1p2P3/2N2Q1p/PPPBBPPP/R3K2R w KQkq - 0 10
8/2p5/3p4/KP5r/1R3p1k/8/4P1P1/8 w - - 0 11
4rrk1/pp1n3p/3q2pQ/2p1pb2/2PP4/2P3N1/P2B2PP/4RRK1 b - - 7 19
r3r1k1/2p2ppp/p1p1bn2/8/1q2P3/2NPQN2/PPP3PP/R4RK1 b - - 2 15
r1bbk1nr/pp3p1p/2n5/1N4p1/2Np1B2/8/PPP2PPP/2KR1B1R w kq - 0 13
r1bq1rk1/ppp1nppp/4n3/3p3Q/3P4/1BP1B3/PP1N2PP/R4RK1 w - - 1 16
4r1k1/r1q2ppp/ppp2n2/4P3/5Rb1/1N1BQ3/PPP3PP/R5K1 w - - 1 17
2rqkb1r/ppp2p2/2npb1p1/1N1Nn2p/2P1PP2/8/PP2B1PP/R1BQK2R b KQ - 0 11
r1bq1r1k/b1p1npp1/p2p3p/1p6/3PP3/1B2NN2/PP3PPP/R2Q1RK1 w - - 1 16
3r1rk1/p5pp/bpp1pp2/8/q1PP1P2/b3P3/P2NQRPP/1R2B1K1 b - - 6 22
4k2r/1pb2ppp/1p2p3/1R1p4/3P4/2r1PN2/P4PPP/1R4K1 b - - 3 22
3q2k1/pb3p1p/4pbp1/2r5/PpN2N2/1P2P2P/5PP1/Q2R2K1 b - - 4 26
6k1/6p1/6Pp/ppp5/3pn2P/1P3K2/1PP2P2/3N4 b - - 0 1
3b4/5kp1/1p1p1p1p/pP1PpP1P/P1P1P3/3KN3/8/8 w - - 0 1
8/6pk/1p6/8/PP3p1p/5P2/4KP1q/3Q4 w - - 0 1
7k/3p2pp/4q3/8/4Q3/5Kp1/P6b/8 w - - 0 1
8/2p5/8/2kPKp1p/2p4P/2P5/3P4/8 w - - 0 1
8/1p3pp1/7p/5P1P/2k3P1/8/2K2P2/8 w - - 0 1
8/pp2r1k1/2p1p3/3pP2p/1P1P1P1P/P5KR/8/8 w - - 0 1
5k2/7R/4P2p/5K2/p1r2P1p/8/8/8 b - - 0 1
6k1/6p1/P6p/r1N5/5p2/7P/1b3PP1/4R1K1 w - - 0 1
6k1/4pp1p/3p2p1/P1pPb3/R7/1r2P1PP/3B1P2/6K1 w - - 0 1
8/3p3B/5p2/5P2/p7/PP5b/k7/6K1 w - - 0 1
5rk1/q6p/2p3bR/1pPp1rP1/1P1Pp3/P3B1Q1/1K3P2/R7 w - - 93 90
r3k2r/3nnpbp/q2pp1p1/p7/Pp1PPPP1/4BNN1/1P5P/R2Q1RK1 w kq - 0 16
4k3/3q1r2/1N2r1b1/3ppN2/2nPP3/1B1R2n1/2R1Q3/3K4 w - - 5 1
8/8/8/8/5kp1/P7/8/1K1N4 w - - 0 1
8/8/8/5N2/8/p7/8/2NK3k w - - 0 1
8/3k4/8/8/8/4B3/4KB2/2B5 w - - 0 1
8/8/1P6/5pr1/8/4R3/7k/2K5 w - - 0 1
8/2p4P/8/kr6/6R1/8/8/1K6 w - - 0 1
8/8/3P3k/8/1p6/8/1P6/1K3n2 b - - 0 1
8/R7/2q5/8/6k1/8/1P5p/K6R w - - 0 124
6k1/3b3r/1p1p4/p1n2p2/1PPNpP1q/P3Q1p1/1R1RB1P1/5K2 b - - 0 1
r2r1n2/pp2bk2/2p1p2p/3q4/3PN1QP/2P3R1/P4PP1/5RK1 w - - 0 1
EOF

N=$(wc -l < "$FENS")
echo "==> $N positions"
if [ "$N" -lt 100 ]; then
    echo "FAIL: need >= 100 positions, have $N" >&2
    exit 1
fi

status=0
for act in screlu crelu; do
    NET="$WORK/net_$act.cenn"
    case "$act" in
        screlu) seed=20260712 ;;
        crelu)  seed=424242 ;;
    esac
    echo "==> activation=$act seed=$seed"
    "$BIN" nnue-testnet "$NET" "$seed" "$act"

    "$BIN" nnue-eval "$NET" < "$FENS" > "$WORK/rust_$act.txt"
    uv run python "$REF" "$NET" "$FENS" > "$WORK/py_$act.txt"

    rn=$(wc -l < "$WORK/rust_$act.txt")
    pn=$(wc -l < "$WORK/py_$act.txt")
    if [ "$rn" != "$N" ] || [ "$pn" != "$N" ]; then
        echo "FAIL[$act]: line counts differ (fens=$N rust=$rn py=$pn)" >&2
        status=1
        continue
    fi

    if diff -u "$WORK/rust_$act.txt" "$WORK/py_$act.txt" > "$WORK/diff_$act.txt"; then
        echo "    OK: $N/$N positions match (rust == python), 0 mismatches"
    else
        mism=$(grep -c '^[<>]' "$WORK/diff_$act.txt" || true)
        echo "FAIL[$act]: $mism differing lines (rust vs python):" >&2
        head -20 "$WORK/diff_$act.txt" >&2
        status=1
    fi
done

if [ "$status" -eq 0 ]; then
    echo "==> ALL GATES PASS: forward pass bit-identical (rust == python) over $N positions, both activations"
else
    echo "==> MISMATCHES FOUND" >&2
fi
exit "$status"
