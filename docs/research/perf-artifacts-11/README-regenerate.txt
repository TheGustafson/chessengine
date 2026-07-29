perf study 11 artifacts — how each file was produced
=====================================================
Write-up: ../perf-study-11-fair-target-2026-07-29.md
Baseline: main @ 51fe78e.  Target: engines/bin/stockfish-eca43a97 (PGO,
nn-0ee0657fb25e.nnue embedded, verified single-net).

All ARCHON builds:
  RUSTFLAGS="-C target-cpu=native" CARGO_BUILD_JOBS=2 \
    cargo build --release --features sota,arch-d48x --target-dir target-probe11

counters-headline.txt   §0.1 three-arm panel
  bash scripts/pstat8.sh 6 300000 tools/nnue/calib/wl50.fens <out> \
    "s10final:./engines/bin/archon-s10-final @BENCH@ engines/canonical/net.cenn" \
    "sfeca:./engines/bin/stockfish-eca43a97 @BENCH@" \
    "sf18:./engines/bin/stockfish18 @BENCH@"
  parse: python3 scripts/parse8.py <out> sfeca

probes.txt              §5 per-change pricing (append-only)
  bash scripts/probe11.sh <label>          # run on each working tree state

flat-*-instructions.txt / perf-*.data      §3 flat profiles
  bash scripts/flat8.sh s11base 2000003 . \
    ./target-probe11/release/chessengine bench 16 1 200000 \
    tools/nnue/calib/wl50.fens nodes engines/canonical/net.cenn
  bash scripts/flat8.sh sfeca   2000003 . \
    ./engines/bin/stockfish-eca43a97 bench 16 1 200000 \
    tools/nnue/calib/wl50.fens nodes

annot-*.txt             §4 the retracted attributions (kept as evidence)
  perf annotate -i perf-<x>-instructions.data --stdio -s "<symbol>" \
    --percent-limit 0 > annot-<x>.txt
  split:  python3 scripts/opsplit11.py annot-<x>.txt <sym-share%> <engine-i/n>
  totals: python3 scripts/opcount11.py perf-<x>.data <engine-i/n> vpmovsxbw ...

counters11.patch        §2 ARCHON deterministic row counters (REVERTED)
  git apply counters11.patch
  cargo build --release --features sota,arch-d48x --target-dir target-count11
  ./target-count11/release/chessengine bench 16 1 30000 \
    tools/nnue/calib/wl50.fens nodes engines/canonical/net.cenn 2>&1 >/dev/null
  git checkout -- src/ && rm src/perfcount.rs
  (perfcount11.rs.txt is the module the patch adds, kept separately because
   `git diff` does not include untracked files.)

sfcount11-nnue_accumulator.cpp.txt   §2 Stockfish-side counters
  Scratch copy, so the user's /home/nick/sf-eca43a97 is never modified:
    cp -r /home/nick/sf-eca43a97/src   /home/nick/sfcount11/src
    cp -r /home/nick/sf-eca43a97/scripts /home/nick/sfcount11/scripts
    ln -s .../nn-0ee0657fb25e.nnue /home/nick/sfcount11/src/   # don't copy 90 MB
  Then the three edits shown in the .txt (counter defs + dump in
  nnue_accumulator.cpp, externs in nnue_accumulator.h, one dump call after
  uci.cpp's "Nodes/second" line), and:
    cd /home/nick/sfcount11/src && make -j2 build ARCH=x86-64-vnni512
  Row counts do not depend on codegen, so a non-PGO build is sufficient here.

nulls11.patch           §5 both reverted implementations

NOT reproducible on this box: `perf record -e instructions:pu` (precise
sampling) exits 255.  §4 exists because of that.
