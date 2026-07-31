#!/usr/bin/env bash
# R3 (D45a speed program): PGO build pipeline for machine-local binaries.
#   bash tools/build_pgo.sh <features> <out-binary-name> [eval-file]
#   e.g. bash tools/build_pgo.sh sota probe-sota-pgo data/nnue/nets/d40-uho-720sb.cenn
#        bash tools/build_pgo.sh nnue-l1-2048 probe-l2048-pgo data/nnue/nets/stage2-v1-s3-2048-farseer-120sb.cenn
# Produces engines/bin/<out> built with target-cpu=native + profile-use.
# The training workload MUST exercise the real inference path: bench runs the
# DEFAULT embedded eval only (its 4th arg is a FEN file), so a bench-only
# profile leaves the feature build's NNUE kernels unprofiled and PGO inert
# (measured 2026-07-18: pgo == native on the sota path). Pass the eval-file
# the binary will actually serve; the workload is bench (search coverage,
# default eval) + a UCI session on that net (deep startpos + kiwipete).
# STANDARD BUILD RULING (user, 2026-07-26; perf study 3 §5.4 verification:
# native+PGO = +8.1% NPS, 8/8 ABBA blocks, 800-FEN hash + fixed-depth node
# totals bit-identical): PGO via this script is THE standard way match and
# release engine binaries are built. The release build is a PGO'd arch
# ladder (x86-64-v3 portable + native machine-local; set PGO_CPU=x86-64-v3
# for the portable rung). Study 3 also moved P4's PGO AHEAD of the small
# SPSA — the SPSA baseline must be the PGO'd binary (>5%-NPS freeze rule).
# The Bench: trailer stays on the portable DEFAULT build (D24 — trailer
# numbers are node counts and never come from PGO binaries) and SPRT pairs
# must both be PGO or both not (D16).
set -euo pipefail
cd "$(dirname "$0")/.."
FEATURES=${1:?features (e.g. sota | nnue-l1-2048)}
OUT=${2:?output binary name under engines/bin/}
EVALFILE=${3:-}
TD=target-pgo-$FEATURES
# `/tmp` on this box is TMPFS, i.e. RAM (standing rule; a full tmpfs makes bash
# return a bare exit 1 and kills the job). A `profile-generate` run of the
# 5 M-node bench frame writes tens to hundreds of MB of raw profile, and a study
# that PGO-builds several configurations back to back keeps one such directory
# per feature set. `PGO_PROF_DIR` moves them onto real disk; the default is
# unchanged so nothing that already works changes behaviour. (Perf study 63.)
PROF=${PGO_PROF_DIR:-/tmp}/pgo-prof-$FEATURES
SYSROOT=$(rustc --print sysroot)
PROFDATA=$(find "$SYSROOT" -name llvm-profdata | head -1)
[ -n "$PROFDATA" ] || { echo "llvm-profdata not found (rustup component add llvm-tools)"; exit 1; }

rm -rf "$PROF"; mkdir -p "$PROF"
echo "[pgo] phase 1: instrumented build ($FEATURES)"
RUSTFLAGS="-C target-cpu=${PGO_CPU:-native} -C profile-generate=$PROF" \
  nice -n 10 cargo build --release --features "$FEATURES" --target-dir "$TD" -q
# PERF STUDY 60 — THE SILENT-PGO GUARD, AND THE NUMBER THAT BOUGHT IT.
#
# `engines/canonical/net.cenn` is a symlink into `data/`, which is gitignored,
# so it DANGLES in every fresh git worktree. Before this guard the UCI training
# session below still ran (the `setoption` just failed, PSQT eval), phase 3
# still merged a profile, phase 4 still produced a binary, and nothing said a
# word — the failure mode this script's own header documents in capitals, with
# no check behind it.
#
# Measured 2026-07-31, `sota,arch-d48x` + D48X, 5,001,562-node frame, median
# of 3, non-PGO native as the reference:
#
#   PGO trained WITH the net     5728.98 i/node   -1.559%
#   PGO trained WITHOUT the net  6040.90 i/node   +3.802%   <- WORSE than no PGO
#
# A silently-untrained PGO binary is not "PGO inert", it is 5.4% worse than the
# correctly-trained one, and any study that measures an arm on one is measuring
# the profile, not the arm. `[ -r ]` follows symlinks, so it is exactly the
# dangling-symlink test; the post-session grep catches an EvalFile that exists
# but does not parse.
if [ -n "$EVALFILE" ] && [ ! -r "$EVALFILE" ]; then
  echo "[pgo] REFUSING: eval-file '$EVALFILE' is missing or unreadable." >&2
  echo "[pgo] In a git worktree, symlink data/ first:" >&2
  echo "[pgo]   ln -s /path/to/chessengine/data data" >&2
  exit 1
fi
echo "[pgo] phase 2: training workload (bench depth 8 + real-net UCI searches)"
"./$TD/release/chessengine" bench 16 1 8 default depth > /dev/null
if [ -n "$EVALFILE" ]; then
  TRAINLOG=$(mktemp)
  { printf 'uci\nsetoption name EvalFile value %s\nisready\nposition startpos\ngo depth 14\n' "$EVALFILE"
    sleep 25
    printf 'position fen r3k2r/p1ppqpb1/bn2pnp1/3PN3/1p2P3/2N2Q1p/PPPBBPPP/R3K2R w KQkq -\ngo depth 12\n'
    sleep 25; echo quit; } | "./$TD/release/chessengine" > "$TRAINLOG"
  if grep -q 'ERROR EvalFile' "$TRAINLOG"; then
    echo "[pgo] REFUSING: the training session did not load '$EVALFILE':" >&2
    grep 'ERROR EvalFile' "$TRAINLOG" >&2
    rm -f "$TRAINLOG"
    exit 1
  fi
  rm -f "$TRAINLOG"
else
  { printf 'uci\nisready\nposition startpos\ngo depth 15\n'; sleep 20; echo quit; } \
    | "./$TD/release/chessengine" > /dev/null
fi
echo "[pgo] phase 3: merge profiles"
"$PROFDATA" merge -o "$PROF/merged.profdata" "$PROF"
echo "[pgo] phase 4: optimized rebuild"
RUSTFLAGS="-C target-cpu=${PGO_CPU:-native} -C profile-use=$PROF/merged.profdata" \
  nice -n 10 cargo build --release --features "$FEATURES" --target-dir "$TD" -q
cp "$TD/release/chessengine" "engines/bin/$OUT"
echo "[pgo] done: engines/bin/$OUT"
