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
`docs/research/LOSS-FORENSICS-PLAYBOOK.md`, and validate any fix on at least
THREE independent losses of the same mode — one position is a patch, three is
a fix.

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
predecessor `docs/research/policy-automl-state.md` covers the D41 policy era;
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
  wrong and leave 587683 untouched. The gate pins two V-arch numbers, both
  reproduced by a from-source non-PGO build: the **eval hash**
  `cd190d1b…7f3d` (sha256 of the forward-pass score for all 50
  `tools/nnue/calib/wl50.fens`) and the **d13 node total 2068701**
  (`bench 16 1 13 <fens> depth <net>` — the net is the optional **6th arg**;
  omit it and you silently measure PSQT, 3441859). Same update rule as the
  bench pin: an intentional change moves the pin in the same commit. Default
  OFF (needs a second release build + the gitignored 84 MB net).
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
  (`states: Vec<StateInfo>`, never empty; D8) — not in `Board` fields. The
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

**HISTORICAL (2026-07-16): D41 ARCHON era — value-foundation phase; net-0
dethroned.**
Classical engine complete (P0–P11, ~2720 Elo PSQT; the PSQT-era depth-6 bench was 548978 — the pin moved to **701605** when S1 correction history landed 2026-07-17).
P13 NNUE landed **net-0** (768→2x256 SCReLU, SF16-teacher, **+696 vs PSQT**,
~3400 anchored); net-1..11, the 13-net D40 policy campaign, and the bulk
value matrix are historical controls. **The B4 question is answered
(2026-07-16): training-data volume was the constraint.** Every failed
challenger trained on 25M–265M positions; nets trained on the official
Stockfish binpacks (nodes5000pv2_UHO stage-1 → farseer stage-2 warm retrain,
3–9B position presentations, official two-stage recipe) beat net-0
decisively. This supersedes the earlier target/search-contract-mismatch
root-cause hypothesis. Current incumbent:
`stage2-v1-s3-2048-farseer-120sb` (L1=2048), ≈ **+265 composed at equal time
(10+0.1)** over net-0 via the gated chain u1024 (+166±18) → uf2048 (+44±14)
→ s3-farseer (+55±17 h2h); the width ladder saturated at 2048. External
anchors (100g, 10+0.1): SF16 −78±18, SF18 −191±23, Integral v6 −70±22
(cross-match triangle closes within 1.3σ). The D40 full-feature arch (SF
HalfKAv2_hm + FullThreats + Reckless-shaped head) is implemented,
SIMD-optimized, parity-proven, and — at full data scale — beats the best
classic-feature net at fixed nodes (stage-1 only, +32.5±17.3, exp-0065); its
equal-time promotion gate is tracked in the live log. **D41 renamed the
engine ARCHON; the learned-search-allocation program (design/19, P15/M11)
resumes after the eval foundation settles.** A ranked search-improvement
program distilled from the SF14→16 patch study
(`stockfish_reference/06-sf14-sf16-search-patch-study.md` §6b) is queued
behind the eval arc: S0 unbalanced testing book + SF16-late mechanics audit,
S1 wire the existing correction-history tables, S2 gainer-season global SPSA
retune. Standing facts: labels from OUR hard-node-limit (5k) datagen are
frequently aspiration-window **bounds** and stay quarantined from scalar
losses (the official binpacks are empirically clean — they trained the nets
that won); the deep value oracle is the tiered SF18 protocol
(`docs/research/sf18-label-quality-plan.md`); the one confirmed policy
positive remains gen-0 rank-8 bilinear head + unchanged net-0 = +68.6±23 at
2k **fixed nodes** (not equal-time; large NPS tax). Live leaderboard +
in-flight jobs: `docs/research/value-foundation-state.md`. The narrative
below is the phase-by-phase history.

Done: types, bitboards, board + FEN, `Move`/`MoveList`; attack tables (P1:
const leapers + geometry, fancy magic sliders, exhaustively validated
against the naive ray oracle, criterion-benchmarked); P2 (SEE pending) —
`zobrist` module (five-key family, all SF tricks), the
`CopiedState`/`StateInfo` split with the `Vec<StateInfo>` state stack,
from-scratch key computation at FEN setup, `attackers_to` + check/pin
bookkeeping, legality-exact FEN ep validation, `make_move(m,
gives_check)`/`unmake_move`/`gives_check(m)` for every move kind with all
five Zobrist keys maintained incrementally, null moves, repetition
detection + `is_draw`/`has_repeated`, the full `assert_consistent`
invariant oracle wired into make/unmake in debug builds, and the
bench-neutral `MakeMoveHooks` seam (`make_move_with`, `prefetch_key`) for
the future TT/NNUE layers.

P3: `movegen` — type-parameterized pseudo-legal generation
(CAPTURES/QUIETS/EVASIONS/NON_EVASIONS with the SF promotion split), the
lazy legality layer (`Board::legal`, `generate_legal`,
`has_legal_moves`), the TT-move validator (`pseudo_legal` + 64k fuzz),
`capture_stage`/`is_capture`; `perft`/`perft_checked`/`divide`
(design/06) with the full 23-position suite green at every depth —
fast tier in `cargo test`, full tier (~1.6 G nodes, ~7 s) and extended
tier (~18 G nodes, ~58 s) in release — and kiwipete d5 bulk perft
measured at ~340 Mnps in the fast configuration (floor 100, target 250).
`tools/perft_diff.py` bisects any divergence against python-chess (or a
UCI reference via `--ref`).

P4–P5: `see_ge`; `eval` phase A (material + PSQT, incremental
`EvalContext`); `search` (iterative-deepening negamax + alpha-beta,
stand-pat qsearch, mate/draw scores, `Worker`/`Stack`/`RootMoves`);
`timeman` `Limits`; `uci` (full loop — position/go/stop on a search
thread, exactly-once bestmove, info lines, setoption); `bench` (perft +
depth modes; the `Bench:` trailer + `tools/check.sh` D24 ritual);
strength harness (`tools/sprt.sh`/`gauntlet.sh`/`tactics.sh`, fastchess +
ordo + SF16 in gitignored `engines/`). M3 passed: complete legal games at
fixed movetime, mate-in-1..3 exact, bit-reproducible fingerprints.

P6: `tt` — 10-byte relaxed-atomic entries in 32-byte clusters, `mul_hi64`
indexing, SF replacement/aging rules, `probe`/`TtWriter`, generations,
fallible `resize` + `clear`, `hashfull`, and the prefetch intrinsic (the
crate's single sanctioned `unsafe`, D14a); `Hash`/`Clear Hash` UCI
options; search integration (probe at node entry, eval-cache reuse,
ttMove-first ordering, store at node exit, prefetch hook in `do_move`) —
bench −46.6% nodes at equal depth, TT-on/off value equality pinned by
tests; miri-clean. M4 passed (audited 2026-07-10, evidence in
`docs/strength-log.md`): gauntlet places the engine between the SF16
UCI_Elo 2000 and 2200 rungs; WAC-revised tactics 154/200 at 100 ms.

P7 (`be3adfa`): PVS zero-window + aspiration windows + full qsearch spec +
hygiene pass + basic clock TM. +65 Elo vs P6, zero time forfeits, ~33× node
reduction at equal depth. P8 (`3227593`): staged `MovePicker` + the full
history/correction stack (butterfly/capture/continuation/pawn/low-ply,
gravity updates, no killers — D17); +76 Elo vs naive MVV-LVA, borrow-safe
`ContHistId` (no unsafe).

P9 pruning waves IN PROGRESS: Wave A (reduction table + RFP + NMP + LMR)
landed ~+570 Elo vs pre-pruning at fixed time (92% smaller tree); **Wave B**
(razoring / LMP / parent- & capture-futility + SEE) PASSED its gate
(`81538f7`, real 10+0.1 **+237 Elo** vs pre-Wave-B, 0 forfeits; bench 350190;
item 9 IIR reverted — regressed ~−15 Elo, parked on
`wip/p9-item9-iir-followpv`). **Waves C–G are now BATCHED (D32, per user):**
item 10 (statScore, `2436669`) landed; items 11–21 + the Wave-G threat map
(Reckless, D30) are implemented + unit-tested in one pass (no per-item games),
then a full correctness gate, then a SINGLE bug-tripwire SPRT vs pre-C — small
regressions are kept and micro-tuned later via SPSA; only a large/surprising
regression (= an implementation bug) triggers a bisect. Waves A/B used the
older per-heuristic proxy-SPRT rhythm (doc 14 §9.4).

Reckless study (2026-07-11, D31): a distilled study of the Reckless engine
(`reckless_reference/`) fed vetted refinements into the P9 waves, added Wave G
(threat map, D30), and annotated the P10/P11/P13 docs; the declined set
(Move-kind capture bits, unsafe TT/Stack, Syzygy FFI, RNG jitter) is recorded
against D2/D14/D15/D19/D24. Adaptation list:
`reckless_reference/00-design-implications.md`.

Integral study (2026-07-12, D36/D37): a distilled study of Integral (top-6
solo-dev C++ engine) — how one dev reached top-6. Headline: ~half the story is
free DISTRIBUTED TEST COMPUTE (D36 — resolved via GCE-spot OpenBench workers at
the tuning grind, post-first-NNUE; proactive SPSA scaffold + OpenBench-parseable
bench built now). Validated our D24/D25/D28/D32/D35 + bullet + UHO book calls;
declined killers (affirms D17); sharpened the P13 loop-first lesson. Notes:
`integral_reference/00-success-factors.md` + `00-design-implications.md`.

Next / plan amended (D21/D28-D37, 2026-07-11..12): eval Phase B (HCE + texel)
DESCOPED — Stockfish is now the NNUE bootstrap data generator, so HCE is off
the critical path. Order: **P9 → P10** (time-mgmt completion + full UCI) **→
P11** (lazy SMP) **→ P13** (NNUE), with PSQT as the eval fallback. NNUE is
trained AlphaGo→AlphaZero style: distill from a compiled SF (discardable
scaffold), then retrain the *shipped* net from self-play for clean
provenance. Datagen is a research data platform (D29, `design/16`); the
bootstrap pipeline + `bullet` format contract live in `design/15`. Perft
remains the correctness oracle — no movegen/make-unmake change lands
without the D24 ritual above.
