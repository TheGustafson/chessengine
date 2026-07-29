perf study 10 — how to regenerate everything here
=================================================

Binaries this study produced (both PGO, sota,arch-d48x, canonical net):
  engines/bin/archon-s10-final   sha256 fc73c9a7c2004cd24ad814fb365280bf7390d491a3df6baffc5549f35cc6854a
    bash tools/build_pgo.sh sota,arch-d48x archon-s10-final engines/canonical/net.cenn
  The pipeline is DETERMINISTIC: two invocations minutes apart produced
  byte-identical binaries (study §1, the PGO null).  Baseline for the panel is
  engines/bin/archon-s9-final (study 9's deliverable = this branch's parent).

Identity pins (both unchanged by everything this study landed):
  ./engines/bin/archon-s10-final nnue-eval engines/canonical/net.cenn \
    < tools/nnue/calib/wl50.fens | sha256sum          -> 5ca978e0...8b38
  taskset -c 4 ./engines/bin/archon-s10-final bench 16 1 13 \
    tools/nnue/calib/wl50.fens depth engines/canonical/net.cenn  -> 1,922,947
  CHECK_NNUE_GATE=1 bash tools/check.sh                          -> Bench: 587683

The matched counter panels (counters-base.txt, counters-final.txt):
  bash scripts/pstat8.sh 6 300000 tools/nnue/calib/wl50.fens /tmp/c.txt \
    "s9final:./engines/bin/archon-s9-final @BENCH@ engines/canonical/net.cenn" \
    "s10final:./engines/bin/archon-s10-final @BENCH@ engines/canonical/net.cenn" \
    "sf18:./engines/bin/stockfish18 @BENCH@"
  python3 scripts/parse8.py /tmp/c.txt sf18
  instructions:u is the carrier; cycles/NPS on this box were bands all session
  (load 5-10, 1-3 GB free) and NOTHING in the study rests on them.

Per-change pricing (probes.txt) — non-PGO pairs, exact because identical
source + identical flags gives an identical binary:
  bash scripts/probe10.sh <label>
Each row prints the wl50 eval hash, so an arm that perturbs the eval (and
therefore the tree) cannot be silently compared per node.

Flat instruction profiles (flat-*.txt):
  bash scripts/flat8.sh <label> 500000 . <binary> \
    bench 16 1 100000 tools/nnue/calib/wl50.fens nodes engines/canonical/net.cenn
The .data files were deleted after the annotations were taken (6 MB of them);
re-record if a new annotation is needed.

Per-symbol annotations (annot-*.txt):
  perf annotate -i <perf.data> --stdio -s "<symbol>" --percent-limit 0
  python3 scripts/annot8.py annot-X.txt [split-addr ...]
Note: addresses shift between builds, so a split-address list is only valid
for the annotation it was derived from.

NEW INSTRUMENTS THIS STUDY ADDED
  scripts/panicedge10.py <binary> [top-n]
    Every conditional jump into a panic landing pad, by enclosing symbol.
    Study 8's bcheck8.py keys on the jump's TARGET SYMBOL and therefore misses
    the majority of them, because LLVM sinks the landing pad into the cold tail
    of the enclosing hot function.  CANDIDATE LIST ONLY — rank by
    (edges in the loop x trip count) or by a probe, never by annotate weight
    (study 8 §7.1 measured that class at -0.045% after it scored 14%).

  scripts/sfsplit10.sh <perf.data> <out.txt>
    Annotate SF18's big-net evaluate.  SF inlines its whole accumulator chain
    into that one symbol, so the only way to cost its stages is by OPCODE:
    vpmovsxbw runs if and only if a FullThreats i8 row is being folded
    (ThreatWeightType = int8_t is the only i8 weight table).  That is how §3.2
    established the threat fold is at parity.

  counters.patch  (src/perfcount.rs + 8 call sites; REVERTED after use)
    git apply counters.patch
    cargo build --release --features sota,arch-d48x --target-dir target-probe10
    taskset -c 4 ./target-probe10/release/chessengine \
      bench 16 1 30000 tools/nnue/calib/wl50.fens nodes engines/canonical/net.cenn
    Prints the per-node rate table to stderr (counters-perfcount.txt).
    This is the instrument the study's method rests on: every candidate was
    sized as loop-body x trip-count from these rates, and two beliefs died on
    them before any code was written (§1.1).
