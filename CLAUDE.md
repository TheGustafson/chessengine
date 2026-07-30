# ARCHON

A UCI chess engine in Rust (crate/dir name `chessengine`, engine name
**ARCHON**), aiming at Stockfish-class architecture. This is also a Rust
learning project: **prefer the idiomatic Rust way** over a transliteration of
C++ patterns, and briefly explain non-obvious idioms when introducing them.

## Standing quality bar: NO ACCEPTABLE BLUNDERS (user, 2026-07-28)

**Work backwards from where we want to be — a top-tier engine — not forward
from where we are. Top-tier engines almost never blunder big moves. Do not
accept blunders; root them out with aggressive investigation.**

Never triage a catastrophic loss by its aggregate Elo share. On 2026-07-28 I
computed that five genuine move errors in a 120-game match were worth ~5.5 Elo
against ~32 Elo of per-move drift, and used that ratio to rank blunders as the
lesser lever. Both halves of that reasoning were wrong:

- **The detector had ~9% recall.** `tools/blunder_forensics.py` only fires when
  OUR OWN eval collapses within 2 plies, so a move whose refutation lands ten
  plies later is invisible. A detector-independent oracle audit of moves from
  LOST games (SF18 @120k, `searchmoves`, every move scored) found **1.5% of our
  moves lose >=100 cp — about ONE per lost game**, and 0.40 game-equivalents of
  win probability per lost game. Five in 120 games was a floor, not a count.
- **Centipawns are not additive into outcomes.** Summing per-move cp loss
  assumes small inaccuracies accumulate linearly; they are mostly absorbed,
  while one 126 cp error at a decision point converts a result. Use win
  probability.


**And they are the cheapest defects to fix.** A catastrophic loss leaves a
complete trace: a position, the move played, the move that was right, a
measurable gap, and a reproducible search you can bisect gate by gate.
Cumulative drift leaves none of that — there is nothing to load into a
debugger. Rank work by TRACTABILITY x Elo, not by Elo share. Every blunder not
dived into is free, high-resolution diagnostic information thrown away, and
each one fixed removes a CLASS of failure rather than one position.

When a blunder is found, drive it to a mechanism with the WHY ladder in
`searchopt/README.md`, and validate any fix on at least
THREE independent losses of the same mode — one position is a patch, three is
a fix.

## Standing research-method bar: NULL FIRST, GAMES ARBITRATE (user, 2026-07-28)

**Live session state: `searchopt/LEDGER.md`** — running matches,
what is established, what is REFUTED (several headline facts from that night's
own notes are retracted later in the same file), and what to do next.


**On 2026-07-28 I spent eight hours producing ~130 measurement scripts, ZERO
completed game results, zero strength changes, and NINE retracted claims.**
The full post-mortem is
`searchopt/README.md`; read it before any
long forensics session. Root cause: a slow arbiter (games) was available the
whole time and I kept choosing fast instruments that could not resolve the
question, then chained inferences on top of their noise.

The eight rules, in priority order:

1. **NULL FIRST, ALWAYS.** Run the null panel *before* the arm. If an inert
   perturbation moves the statistic as much as the arm plausibly could, **the
   instrument cannot answer this question — change instrument.** A gate, not a
   footnote. (The 44-position forced-move instrument moves 13/44 → 20/44 under
   an inert ±1; I read verdicts off it twice, in opposite directions.)
2. **NORMALISE AT THE TOOL BOUNDARY, not in prose.** Our search scores are
   **2.07× Stockfish's PRINTED cp** — SF's `to_cp` divides its internal value by
   `a = p_a(material) ≈ 385` (`eca43a97:src/uci.cpp:554`), and we print raw
   internal values. Use `uciprobe.to_sf_frame()` / `Engine(..., frame=...)`;
   never hand-compare. A fact in a doc gets forgotten, a constructor cannot.
3. **PREFER DISCRETE OUTCOMES** — nodes-to-find, does-it-find-it, argmax
   agreement. Immune to frame, bounds, scale and chaos. Reach for a cp mean only
   after a discrete statistic has shown the effect exists.
4. **ONE INFERENCE DEEP, THEN MEASURE.** Never chain two unmeasured inferences.
   Every rung of a causal chain gets its own measurement *before* the next rung
   is written down.
5. **GAMES ARBITRATE, AND THEY START EARLY.** A queued match costs nothing while
   you think. Launch the best available candidate at hour 0. **A session that
   ends with zero completed game results produced nothing**, however much was
   learned.
6. **WHEN THE STORY GETS BEAUTIFUL, ATTACK IT HARDEST.** Narrative coherence is
   the warning sign, not the reward. An arm labelled "falsification control"
   that *wins* is telling you your model's sign is wrong — this happened twice
   in one night before I believed it.
7. **PER-CLASS *AND* PER-POSITION, TOGETHER.** Never a lone median (a 1.0×
   median hid a 16× subclass), never a lone position (one 190 cp gap did not
   generalise: median −17 over 44). Both directions lie.
8. **RE-DERIVE ANY NUMBER BEFORE BUILDING ON IT**, especially one whose `n` you
   did not set. "Our EBF 1.49 vs SF 1.65" was an n=14 artifact; at n=44 it is
   1.551 vs 1.497 and a proper geometric EBF is 1.629 vs 1.620 —
   indistinguishable. It was load-bearing for hours.

## Design documents (read before implementing anything)

`design/` holds the reviewed design suite that governs all implementation:

- `design/decisions.md` — pinned decisions D1-D44. **Decisions change here
  first; docs follow; code follows docs.**
- `design/00-master-plan.md` — phase ordering (P0-P15), milestone gates,
  current phase status.
- `design/README.md` — index and process rules for the suite.
- `design/01-*.md` … `design/19-*.md` — one contract per component (14
  numbered + `15` NNUE datagen + `16` research data platform + `17` NNUE
  research predecessor, D34/D40 + `18` experimental ledger, D38 + `19` ARCHON
  learned search allocation, D41). Check code against the
  owning doc's acceptance criteria (§8) before calling a phase done.

## Live research state (read before any research/training loop)

`docs/research/value-foundation-state.md` is the live research log — its
TAKEOVER CHECKPOINT at the top is the authoritative current state (incumbent
chain, external anchors, in-flight jobs). Read it before any research/training
loop and update it after every completed, rejected, or redirected trial. Its
predecessor `old/docs/research/policy-automl-state.md` covers the D41 policy era;
that doc's header invariants (bounded-label quarantine, quality-first flywheel
rule, SF18 tiered-oracle plan) still stand. Companion docs:
`docs/research/archon-execution-plan.md`,
`docs/research/archon-self-improvement-roadmap.md`,
`docs/research/sf18-label-quality-plan.md`.

## Experimental ledger (D38, design/18)

`ledger/` is the content-addressed provenance DAG + results ledger tracking the
engine's evolution (nets, corpora, experiments) so any version is reproducible
and any net's lineage is provable. Append-only JSONL registries
(`nets`/`data`/`experiments`) + `milestones.md`, driven by the stdlib-only
`tools/ledger/ledger.py` CLI. **After training a net, building a corpus, or
running a match/SPRT/SPSA, add a ledger row** (`add-net`/`add-data`/`add-exp`);
identity is the artifact's sha256; every row carries a regen recipe. Metadata is
git-tracked; the net/corpus bytes stay gitignored (GCS mirror + OpenBench come
later, D36). `ledger.py verify` checks hash drift + referential integrity.

When implementing a component, read its design doc first and follow it; if
reality forces a deviation, update the doc (and decisions.md if a pinned
decision changes) in the same commit.

**Scalar-value training data is governed by D46** (design/16 §3.10): the
trainer-parity quiet filter is default-ON at every export/extraction layer,
scale multipliers are derived + verified (`tools/nnue/net_scale_check.sh` is a
hard pre-game gate), filters run before clamps, WDL columns are real or
WDL=0-with-matching-MULT. Before building or modifying ANY data pipeline, read
`docs/research/data-recipe-contract.md` (one page; §7 = live tool status).

## Commands

- `cargo test` — run all tests (the default check after any change);
  includes the fast perft tier + checked mini-tier (design/06 §3.5 tier 1)
- `cargo clippy --all-targets` — must stay clean
- `cargo run` — the UCI loop (P0 subset: `uci`/`isready`/`d`/`quit`;
  argv subcommands `bench`, `perft <depth> [fen]`, `d`)
- `bash tools/check.sh` — the full merge gate (fmt, clippy, tests, release
  perft tier, bench fingerprint; prints the `Bench:` trailer line to paste)
- **The `Bench:` commit trailer (D24, design/14 §3.2)** — every commit that
  can affect search behavior (search, eval, movegen/make-unmake, move
  ordering, TT) ends with a `Bench: <nodes>` trailer: the `Nodes searched`
  total from the pinned fingerprint invocation
  `./target/release/chessengine bench 16 1 6 default depth`
  (check.sh runs exactly this — see `tools/check.sh` — and prints the
  ready-to-paste line). The depth is pinned at **6** (~34 s in the fast
  config): the TT landed in P6 and flattened the node curve enough to
  raise it from the pre-TT depth 3; it keeps climbing toward the SF-default
  13 as the pruning waves shrink the tree, staying under ~60 s per run.
  Trailers from before this pin (P9 Waves C–G and earlier) were recorded at
  depth 3 and are NOT reproducible by today's depth-6 check.sh — reproduce a
  commit's number with the depth its own era used. A patch that changes the
  number is *functional* by
  definition and the trailer states the **new** count; intended
  non-functional patches (refactors, speedups) must reproduce the old count
  bit-for-bit — verify by rerunning check.sh on both sides. Commits that
  cannot affect search (docs, UCI I/O, build scripts) may omit the trailer.
  Details: TESTING.md, design/14 §3.2.
- **The NNUE behavioural gate** — `CHECK_NNUE_GATE=1 bash tools/check.sh`,
  **mandatory before merging anything under `src/eval/nnue/`.** The `Bench:`
  pin above is BLIND to NNUE work: `Cargo.toml` has no `default = [...]`, so
  the pinned `bench 16 1 6 default depth` is a default-feature build and the
  whole SOTA path (accumulator, SIMD kernels, `sota_threats.rs`) is behind
  `sota` — verified 2026-07-27 by `nm` finding zero threat symbols in the
  default release binary. A no-net `bench` is a PSQT run (empty `EvalFile`
  ⇒ NNUE off). So an incremental-accumulator rewrite can be arbitrarily
  wrong and leave 587683 untouched. The gate reads the CANONICAL net and arch
  from `engines/canonical/` (see below) and pins two numbers, both reproduced
  by a from-source non-PGO build: the **eval hash**
  `eb981b11…4f4f37` (sha256 of the forward-pass score for all 50
  `tools/nnue/calib/wl50.fens`) and the **d13 node total 1985398**
  (`bench 16 1 13 <fens> depth <net>` — the net is the optional **6th arg**;
  omit it and you silently measure PSQT, 3441859). Both were re-pinned on
  2026-07-30 by **perf study 32's i8 HalfKA feature transformer** — the 57.7 MB
  i16 weight table is quantised to i8 at load with a per-column integer scale,
  measured −6.8% cycles/node and +3.6 ± 7.7 Elo over 2932 fixed-node games
  (i.e. no detectable fidelity cost). The immediately preceding pins were
  `5ca978e0…8b38` / `1922947` (perf study 8's threat-frame change, §4.6 — the
  FT merge carries the 255/64 frame ratio as 4, a measured 0.14 SF cp mean
  shift), before that `07993850…41bd` / `1945095`, and the D47V ones
  `cd190d1b…7f3d` / `2068701`.
  Same update rule as the bench pin: an intentional change moves the pin in the
  same commit. Default OFF (needs a second release build + the gitignored net).
- **The canonical net and engine — `engines/canonical/`.** ONE place names the
  release pair, so switching nets is a symlink change with no stragglers:
  `net.cenn` -> the release net, `archon` -> the release PGO binary, `arch.txt`
  -> the cargo features the binary MUST be built with (a mismatched net/arch
  pair fails the CENN section-size check at load). **Scripts and docs must
  reference `engines/canonical/net.cenn`, never a bare filename.**
  `uciprobe.NET` already resolves to it. Current: **D48X** (gen-2, 1280
  accumulator, `sota,arch-d48x`), promoted 2026-07-28 at a measured
  **-9.1 [-34.4, +16.1] Elo over 382 games vs D47V** — deliberately, on the
  ruling that D48X is the stronger eval and loses only because the search
  cannot yet exploit it. Rationale, evidence and the falsifiable prediction it
  makes are in `engines/canonical/MANIFEST.md`.
- **The D24 perft ritual** — any change touching `movegen`, `board`
  mutation, `zobrist`, or `attacks` must pass, before landing:
  `cargo test` (fast tier, automatic) AND
  `cargo test --release -- --ignored perft_full_suite` (< 30 s). The
  extended tier (`--ignored perft_extended`, ~18 G nodes, < 4 min in the
  fast configuration) runs before releases and after movegen/make-unmake
  rewrites. Debugging a wrong count: TESTING.md (divide →
  `tools/verify_perft.py "FEN" N --divide` → `tools/perft_diff.py`).
- **Fast builds** (D16, design/01 §3.7) — the supported configuration for
  strength testing, bench signatures, and SPRT; SPRT pairs must use
  identical flags:
  `RUSTFLAGS="-C target-cpu=x86-64-v3" cargo build --release`
  (or `target-cpu=native` for a machine-local binary). Default builds stay
  portable — we deliberately don't commit these flags in `.cargo/config.toml`.
  **PGO is the standard for match/release engine binaries (user ruling
  2026-07-26; perf study 3: +8.1% NPS, proof-gated):** build them with
  `bash tools/build_pgo.sh <features> <out> <eval-file>` — the eval-file
  argument is REQUIRED for NNUE binaries (bench-only training leaves the
  NNUE kernels unprofiled and PGO inert, measured 2026-07-18). SPRT pairs
  both-PGO or both-not; the `Bench:` trailer stays on the portable default
  build; the SPSA baseline is the PGO'd binary.
- Profiling: `RUSTFLAGS="-C target-cpu=x86-64-v3 -C force-frame-pointers=yes"
  cargo build --profile profiling` (release codegen + line tables for
  perf/flamegraph)
- **NEVER `pkill -f <pattern>`** (user instruction 2026-07-27, after it
  happened twice in one session). `pkill -f` matches every process's FULL
  command line — including the shell running the very command that typed the
  pattern — so it kills its own caller. The turn dies with a bare exit 143/144
  and no output, which then looks like an unrelated failure. The bracket dodge
  (`'fastches[s]'`) is NOT sufficient: it only works while the name appears
  once, and it silently fails the moment the same command line also mentions
  the plain name (e.g. a kill followed by a relaunch). Kill by PID instead:
  - a recorded PID file — `kill "$(cat run.pid)"` (the SPSA/trainer scripts
    write them), or
  - `systemctl --user stop <unit>` for anything from `systemd-run` (the
    trainer, tb shims and watchers all are), or
  - `pgrep -f 'PATTERN' | grep -vE "^($$|$PPID)$" | xargs -r kill` —
    excluding `$$` and `$PPID` is what makes it safe.

  And never put a kill and a relaunch of the same name in ONE command line.

  **The `$$`/`$PPID` filter is NOT sufficient for DETECTION** (found
  2026-07-27, after it twice reported a dead SPSA run as "running"): a
  `pgrep` inside a shell function or `$(...)` runs in a SUBSHELL whose pid is
  neither `$$` nor `$PPID`, so the wrapper still matches itself and the count
  comes back non-zero for a process that does not exist. Every Claude-run
  command sources `shell-snapshots`, which is a reliable signature to exclude:

  ```bash
  pgrep -f 'PATTERN' | while read -r p; do
      tr '\0' ' ' < "/proc/$p/cmdline" 2>/dev/null | grep -q shell-snapshots && continue
      echo "$p"
  done
  ```

  Better still, don't pattern-match at all: `systemctl --user is-active <unit>`
  for anything from `systemd-run`, or a PID file. When a count looks
  surprising, confirm by reading `/proc/<pid>/cmdline` directly — that is what
  finally settled it both times.

## Search optimization — `searchopt/`

**All search / loss-forensics work happens in `searchopt/`** — the method guide
(`README.md`), the claim ledger (`LEDGER.md`, every surviving claim with the
number that decides it, plus the refuted list), the tools, and the position
sets. Reading and changing `src/` is expected; what you should NOT do is go
shopping in `docs/research/` for context — that is how 2026-07-28 lost a day.
Three standing rules from that day, each paid for: **eval is an AXIOM, not a
hypothesis**; **null the METRIC (run the opposite-direction arm) BEFORE the
arm**; **stay on ONE position, no aggregates during diagnosis**.

## `old/` — archived, not deleted

2026-07-28: 371 files (206 one-shot campaign scripts, 129 per-session research
docs, raw perf artifacts) were moved to `old/` because nothing live referenced
them. Verified: no live file references any moved path, and
`CHECK_NNUE_GATE=1 bash tools/check.sh` still exits 0 with `Bench: 587683`.
**Nothing in `old/` is loaded or trusted**; several of its documents contain
conclusions that were later refuted.

## Layout

Library crate + thin binary. Modules (re-exported at the crate root by
`src/lib.rs`):

- `types` — `Color`, `PieceType`, `Piece`, `Square` (+ `Display`/`FromStr`)
- `bitboard` — `Bitboard` newtype over `u64`, set ops, iteration, constants
- `board` — `Board` (hybrid bitboards + mailbox), `CastlingRights`
- `fen` — `Board::from_fen`/`to_fen`, `FenError`, `START_FEN`
- `moves` — `Move` (packed `u16`), `MoveKind`, `MoveList` (stack-allocated)
- `attacks` — const-built leaper/geometry tables (`knight_attacks`,
  `pawn_attacks`, `between`, `line`, `aligned`, ...) + fancy magic slider
  lookups behind `LazyLock` (`attacks(pt, sq, occ)`; hoist `tables()` in hot
  loops). Criterion baseline: `benches/attacks.rs` → `docs/perf-baseline.md`.
- `movegen` — `generate::<GT>` (const-generic GenType), `generate_legal`,
  `has_legal_moves`; `Board::{legal, pseudo_legal, capture_stage,
  is_capture, attackers_to_exist}` live here next to the generator.
- `perft` — `perft` (bulk-counted), `perft_checked` (per-node invariant
  oracle), `divide` (SF-format output). Oracle data: `tests/data/perft.epd`.

## Core encoding decisions (do not change casually)

- **Square mapping**: little-endian rank-file. a1 = 0, b1 = 1, ..., h8 = 63;
  `file = index & 7`, `rank = index >> 3`.
- **Board representation**: bitboards are primary (`pieces[color][piece_type]`,
  per-color occupancy, all occupancy); a redundant `mailbox: [Option<Piece>; 64]`
  gives O(1) piece-on-square. `put_piece`/`remove_piece` — plus the internal
  `move_piece` fast path used by make/unmake — are the ONLY ways to mutate
  placement; they keep every layer in sync but are placement-only:
  Zobrist key deltas live in `make_move`, NOT in the placement mutators (the
  five-key family updates differently per piece type and move kind, and the
  ep/castling/side keys never route through placement — design/04 §3.3, §9.1).
  From-scratch key computation happens once at setup (`from_fen`) and in the
  debug oracle.
- **Game state** (castling rights, ep square, halfmove clock, all Zobrist
  keys, check info) lives in a `StateInfo` stack owned by `Board`
  (the live ply inline as `state: StateInfo` plus a `history: Vec<StateInfo>`
  of everything below it; D8 as amended by perf study 14) — not in `Board`
  fields. The
  public accessors delegate to the top state.
- **Move encoding** (Stockfish-style, must fit u16): bits 0-5 from, 6-11 to,
  12-13 promotion piece (0 = knight ... 3 = queen), 14-15 kind
  (normal/promotion/en-passant/castling). Castling is the king's move (e1g1).

## Conventions

- Newtypes with private fields; constructors enforce invariants
  (`Square::new` asserts index < 64). Expose behavior, not representation.
- No heap allocation in anything that will be hot (movegen, search):
  `MoveList` is a fixed array + len that `Deref`s to `[Move]`.
- Sets don't implement `Iterator`; they implement `IntoIterator` with a
  separate iterator struct (see `Bitboard`/`BitboardIter`).
- Text conversions go through `Display`/`FromStr` with real error types that
  implement `std::error::Error`. Fallible parsing returns `Result`, never
  panics.
- `const fn` and `const` tables wherever possible; zero runtime dependencies
  so far — keep it that way unless there's a strong reason.
- Tests: build positions with `Board::empty()` + `put_piece` or `from_fen`,
  never struct literals. Board-mutating tests should call
  `board.assert_consistent()` — the full invariant oracle (placement layers,
  all six Zobrist keys vs from-scratch recompute, structural sanity, check
  bookkeeping; doc 04 §3.12). It is compiled under
  `cfg(any(test, debug_assertions))`, and make/unmake run it automatically
  at every node in debug builds; release compiles it out.
  Use the named square constants (`Square::E4`) for readability.

## Stockfish reference

`stockfish_reference/Stockfish/` holds a clone of Stockfish (gitignored).
When unsure how to do something optimally, read the corresponding C++ source,
then write a distilled markdown note in `stockfish_reference/` (tracked; index
in `NOTES.md`) so the insight isn't lost. **Never copy Stockfish code** — it's
GPLv3; take ideas, reimplement idiomatically in Rust.

**For CONSTANTS, the reference is `eca43a97` — never the checkout, and never
a release tag.** Our tunables were transcribed from Stockfish master at
`eca43a97` (2026-07-06), which is **223 commits NEWER than the `sf_18` tag**
(2026-01-31). Verified 2026-07-27: a full screen of all 260 tunables found
every value ≥20 present at eca43a97 except two known ARCHON originals, and
in-place spot checks (razoring, both ProbCut betas, the whole movepick
scoring family) match exactly. **The transcription is faithful; there is no
drift Elo to harvest.**

Razoring is the canary, and it moved in BOTH directions:

| revision | razoring margin |
|---|---|
| `sf_16` | `alpha − 456 − 252·d²` |
| `sf_18` (tag) | `alpha − 485 − 281·d²` |
| **`eca43a97` (our source)** | **`alpha − 465 − 300·d²`** ← what ARCHON ships |

So reading a release tag for a parameter comparison is the SAME mistake as
the old `sf_16` checkout, just in the other direction: the clone sat at
`sf_16` until 2026-07-27 (bare greps silently returned SF16 constants), and
it now sits at `sf_18`, which is *older* than our source. On 2026-07-27 this
made the frame study report "razor 465 vs 485" as a deviation when 465 is an
exact match to our real reference. **Read the reference explicitly:**
`git -C stockfish_reference/Stockfish show eca43a97:src/search.cpp`.

`python3 tools/spsa/param_provenance.py` checks all of this — reference
canaries, the transcription screen, and SPSA campaign coverage (121 of 260
tunables are in no campaign) — and exits non-zero on a mismatch or a stale
campaign entry. Run it before quoting an SF constant as "ours" or "theirs".

**Any distilled note written before 2026-07-27 was read against SF16** —
check the era before quoting one as current. Shape/mechanism questions ("does
SF do X") are fine against the checkout; only VALUES need eca43a97.

`reckless_reference/Reckless/` similarly holds a clone of Reckless (a top-2
Rust engine, ~3833 SPCC; **AGPLv3**, gitignored). Its distilled study notes
live in `reckless_reference/` (tracked; index `NOTES.md`, adaptation list
`00-design-implications.md`) — same rule: **never copy code**, take ideas and
reimplement in our own safe Rust (D27/D31).

`integral_reference/integral/` holds a clone of Integral (a top-6 solo-dev C++
engine, aronpetko; **NO license → all-rights-reserved**, gitignored — ideas-only
even harder than SF/Reckless). Distilled notes in `integral_reference/` (tracked;
index `NOTES.md`, headline `00-success-factors.md`, adaptation list
`00-design-implications.md`) — the study of how a solo dev reached top-6
(D36/D37). **Never copy code.**

## Multi-agent review loop (D33)

For substantial batched or one-shot implementations, **before game testing**
run an iterative multi-persona review: 3 parallel reviewer subagents with
distinct personas — (A) correctness/safety, (B) reference-fidelity vs
`stockfish_reference/` + `reckless_reference/`, (C) Rust-idiom/perf/test-quality
— each reporting structured findings; then a synthesis agent verifies + applies
the warranted fixes + re-runs the correctness gate + commits; loop
A/B/C → adjust until a round has no material findings (convergence, ~4-round
cap), **then** game testing. Reviewers are read-only; only the adjust agent
mutates. Full mechanics: `design/14` §10.

## Status / roadmap

**CURRENT (2026-07-27, fourth refresh): D48X TRAINING + STAGE-A SPSA,
BOTH LIVE.** ET testing stopped mid-SF18 by user order ("I know how
they will go… GOGO"); the fn pentagon stands (V−Spost −35, Spost =
SF16 parity, exp-0149..0153) and so does the headline: **V beat S-post
+38 at equal time — 1.51× speed ≈ +73 Elo ≈ ~120 Elo/doubling, THE
calibration constant (exp-0154)**. **The pre-registered NPS gate FIRED
and D48W became D48X: width 1536 → 1280, recipe untouched.** Measured
idle-box, one protocol, all PGO, all on the AVX-512 VBMI2 path: V-768
1051k, **X-1280 893k**, W-1536 804k, S-1280/128 724k ⇒ 1536 was only
1.11× S and 0.77× V (−45 Elo of equal-time speed, vs −28 at 1280), and
1280 is additionally the A3 trunk's OWN width so the warm is a
VERBATIM transfer instead of a net2net expansion. Standing finding:
**inside 768–1536 width is ~Elo-NEUTRAL at equal time** (+44 fn per
doubling vs −45 speed per doubling) — gen-2 must win on the RECIPE.
Running: `archon-d48x` (380sb, ~178k pos/s, ~3.3 min/sb) + tb shim +
every-30sb watcher vs a frozen V yardstick; POLICY DITCHED
(pol_lambda 0; user 35%-top-1 bar), dis head deferred to
post-finalization ("derived from itself"); NO automated stops.
Concurrently: **stage-A SPSA** (12 params, V net + PGO tune binary,
fn20k, 8 games × 2400 iters, exchange on DISK not tmpfs) — measured
zero training interference.
Engine: S0 search program COMPLETE (all six audit items; **depth-6
bench pin now 587683**; tripwires clean); **PGO is THE
standard build** (tools/build_pgo.sh WITH eval-file; pairs both-PGO;
Bench: trailer stays portable-default); freebies landed (+~6% both
arches); distributed-SPSA kit built+parked (D36 amended — no
OpenBench). **Landed 2026-07-27:** (a) `perf-impl-1` — threat events
from bitboards instead of a mailbox replay + movepick stage-init
outlining: **−6.17% instructions, −8.44% L1d loads**, node-identical
and eval bit-exact. Both wins were mechanisms perf study 7 named
WRONGLY (threats = representation not row count, overfold was 1.034;
movepick = register pressure not icache) — see that study's two
correction banners. (b) **upcoming-repetition** (P9 item 19's last
piece): the cuckoo tables finally have a caller. Bench 385329 →
445174, which is +15.5% ONLY because bench position 37 is the
artificial pawnless max-mobility position where rule50 never resets;
across the other 48 it is **−28.8% nodes**. Strength testing still
owed on both — they need a quiet box. **The authoritative live state is the TAKEOVER CHECKPOINT
atop `docs/research/value-foundation-state.md` — read it before
anything.** The paragraph below is the pre-distillation-era history
(2026-07-16 snapshot).
