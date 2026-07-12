# Further study — coverage/accuracy audit findings

Written by a coverage-and-accuracy pass over `reckless_reference/00-*.md` ..
`14-*.md` and `NOTES.md` against the vendored source in
`reckless_reference/Reckless`. Everything below is either a small, fixed-cost
gap worth a future pass, or a correction to double-check next time a note in
this suite is touched. Nothing here blocks the notes' current usefulness —
the spot-checked material (see §2) was found to be accurate to the line.

## 1. Coverage

Every `.rs` file under `reckless_reference/Reckless/{src,build}` is cited by
literal filename in at least one note, **with one exception**:

- **`src/main.rs`** (4 lines: `fn main() { ...; reckless::run(buffer); }`) has
  no literal citation anywhere in the suite. It's trivial glue (parses argv
  into a `VecDeque<String>` and hands it to `reckless::run`, which is
  `lookup::initialize(); nnue::initialize(); uci::message_loop(buffer);` —
  see `src/lib.rs:26-30`), so this is *not* a missing-content gap — but there
  is one small idea worth one sentence in `09-uci-time-management.md`'s
  `message_loop` discussion: Reckless feeds **argv and stdin through the
  exact same message queue**
  (`main.rs` → `run(buffer: VecDeque<String>)` → `message_loop`), so CLI
  subcommands and UCI commands share one dispatch path with no special-casing
  at the entry point. That's directly relevant to this project's own
  `src/main.rs` (`cargo run` argv subcommands `bench`/`perft <depth>
  [fen]`/`d`, per `design/12-uci-time-management.md` and `CLAUDE.md`'s
  Commands section) as a possible simplification: route argv subcommands
  through the same loop as stdin `uci` input instead of a separate `match` on
  `std::env::args()`, if that isn't already how ours is structured. Low
  effort (one sentence + citation), not done here to keep this pass surgical
  — flagging for whoever next revises note 09.
- `src/tools/mod.rs` and `src/types/mod.rs` are likewise pure `mod`/`pub use`
  wiring with no literal-filename citation, but every symbol they re-export
  (`bench`/`perft`/`speedtest`, and all of `types::*`) is independently cited
  at its defining file — no content gap, nothing to add.

No other file is uncited or under-covered. The two build-script-generated
directories (`build/*.rs`) and every `src/nnue/**` leaf (including the
AVX2/AVX-512 threat-vectorization internals, which are cited via
`vectorized/avx2.rs:NN-MM`-style line ranges rather than a bare filename
match — easy to miss with a naive grep, confirmed present on inspection) are
covered.

## 2. Accuracy spot-checks performed

Verified line-for-line against `search.rs`, `movepick.rs`, `history.rs`,
`stack.rs`, `types/score.rs`, `types/piece.rs`, `nnue.rs`,
`nnue/accumulator/psq.rs`, and `nnue/accumulator/threats/threat_index.rs`:

- **07-search-node-pruning.md**: RFP margin formula (`search.rs:547-563`),
  NMP gate + reduction formula + TT-capture veto (`search.rs:565-629`),
  singular-extension margins/multicut/LDSE (`search.rs:687-736`), `Stack`/
  `StackEntry` layout (`stack.rs`), `Score` constants and `draw()`
  (`types/score.rs`), piece values (`types/piece.rs:138-148`) — **all exact
  matches**, formulas and constants reproduced correctly to the literal
  integer.
- **05-movepick-history.md**: `Stage` enum and `MovePicker` struct/`new`/
  `next` dispatch (`movepick.rs:1-70`), the full `score_noisy`/`score_quiet`
  formulas including every named constant (`movepick.rs:128-213`),
  `QuietHistory`/`PawnHistory`/`NoisyHistory`/`CorrectionHistory` shapes and
  `apply_bonus` gravity operator (`history.rs`), the fifty-move-clock bucket
  formula (`board.rs:75-77`) — **all exact matches**. Two very minor nits
  (neither changes any cited constant):
  1. §2's prose description of the `offense[pt]` bitboard ("squares where
     moving a piece of type `pt` there would let it attack a *vulnerable*
     enemy piece") is accurate for pawn/knight/bishop/queen but doesn't call
     out that the **rook** term is actually `Bitboard::file(enemy_king.file())
     & !threats` (`movepick.rs:181`) — lining up on the enemy king's file, not
     targeting a "vulnerable rook-attackable piece" bitboard the way the
     other four terms do. Worth a one-clause fix next time §2 is touched;
     doesn't affect the reproduced formula (`+ 3446 · offense[pt].contains(..)`)
     which is correct as written.
  2. The `NoisyHistory` size estimate in the §3 table ("12×64×7×2×2B ≈ 21
     KiB") uses 12 piece slots; the actual backing type is
     `PieceToHistory<T> = [[T; 64]; 13]` (13 = 12 real pieces + `Piece::None`,
     `history.rs:6`), so the true footprint is ≈22.75 KiB, not ≈21 KiB. Cosmetic
     (off by ~8% in a rough size estimate, not a formula or behavior claim).
- **12-nnue-accumulator-psq.md**: every named constant in §1
  (`NETWORK_SCALE=380`, `INPUT_BUCKETS=10`, `OUTPUT_BUCKETS=8`,
  `L1/L2/L3=768/16/32`, `FT_QUANT=255`, `L1_QUANT=64`, `FT_SHIFT=9`,
  `DEQUANT_MULTIPLIER`), `INPUT_BUCKETS_LAYOUT`/`OUTPUT_BUCKETS_LAYOUT`
  contents (`nnue.rs:82-123`), the `pst_index` flip/bucket/offset formula
  (`nnue/accumulator/psq.rs:204-211`), and the 66,864-row threat feature table
  + `PIECE_TARGET_COUNT = [6,10,8,8,10,8]` (`nnue.rs:358`,
  `threats/threat_index.rs:44`) — **all exact matches**.

No misstatement found that would mislead an implementer; the notes are safe
to build against as-is.

## 3. Links

- `NOTES.md` lists all 14 numbered notes (`01-core-types-move-encoding.md`
  through `14-tablebases-tooling-build.md`); confirmed every listed link
  target exists and every note file on disk is listed. No stray or missing
  entries either direction.
- `00-design-implications.md`'s `design/NN` cross-references were checked
  against the real `design/` suite (`00-master-plan.md`, `01-architecture.md`,
  ... `16-research-data-platform.md`). Findings:
  - Every citation of the form `design/NN §X.Y` resolves to a real doc and,
    spot-checked for `design/06 §3.5/§3.6/§9`, `design/12 §3.6-3.14/§9.6`,
    `design/13 §3.3/§3.4/§3.7/§3.8/§3.10/§9`, `design/14 §3.6/§3.7/§3.8`, a
    real, topically-matching section (not just a doc that happens to exist).
  - `design/17-tablebases.md` is cited twice (lines ~436, and in
    `14-tablebases-tooling-build.md`'s adapt-to-us list) — **this doc does
    not exist** (the suite currently runs `00`-`16`). Not a defect: both
    citations explicitly say "a **future** `design/17-tablebases.md`", i.e.
    they're proposing a name for a doc that doesn't exist yet, not asserting
    one does. Tablebases are presently tracked only as scattered mentions in
    `00-master-plan.md`/`07-search-core.md`/`10-pruning-extensions.md`/
    `12-uci-time-management.md`/`13-threading-smp.md`/`02-core-types.md`
    (`VALUE_TB` score-band plumbing, mostly) with no owning component doc.
    **Flag for whoever schedules P-post-P13 tablebase work**: Reckless's
    two-tier-probing note (`14-tablebases-tooling-build.md` §on Fathom FFI)
    is a ready-made starting point once that doc gets written; no action
    needed before then.
  - One citation is loose rather than wrong: `00-design-implications.md`'s
    P10 item 6 cites `design/12 §3.14 open Q5` for the WDL/`to_cp` question,
    but `design/12-uci-time-management.md` §3.14 ("Score formatting and the
    normalization hook") doesn't itself number an "open Q5" — the actual
    open question 5 lives in the doc's global `## 9. Open questions` list
    ("WDL model timing: `to_cp` stays identity until phase C..."). The
    *topic* match is correct, just the locator conflates "§3.14 is the
    relevant section" with "§9 item 5 is the open question." Cosmetic;
    worth tightening to `design/12 §3.14 / §9 open Q5` next edit.

## 4. Suggested follow-ups (not done here, kept out of scope for a surgical pass)

1. Add the argv-into-message-loop one-liner to `09-uci-time-management.md`
   (§1.a above).
2. Fix the `offense[pt]` rook-term description and the `NoisyHistory` size
   estimate in `05-movepick-history.md` (§2.b/§2.c above).
3. Tighten the `design/12 §3.14 open Q5` locator in
   `00-design-implications.md` (§3 above).
4. None of the above changes any cited constant, formula, or the "Adapt to
   us" action list — no re-prioritization needed.
