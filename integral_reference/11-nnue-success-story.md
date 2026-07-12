# Integral's eval success story: data, training, and the net-iteration loop

**Provenance/licensing note** (same as every file in this directory): Integral
has no license file → all-rights-reserved by default, stricter than
Stockfish's GPLv3 or Reckless's AGPLv3 (D27 applies harder still). Nothing
below reproduces Integral source. Every mechanism is stated as prose (with a
file:line or commit hash to re-find it), a constant *value* (facts/numbers
are not copyrightable expression), or my own paraphrase/pseudocode.

**Sources**: `README.md` (the "Data Generation Process"/"Training Process"
section); `src/data_gen/data_gen.{h,cc}`; `src/data_gen/format/{format.h,
binpack.h,fens.h}`; `preprocess/net_processing.cc`; `shared/nnue/
definitions.h`; the full 669-commit `main` history plus **945 remote
branches** (`git branch -a`, `git log --all`) — the branch graveyard is a
primary source of its own and is mostly untouched by this repo's other
Integral notes.

**Scope vs. this directory's other notes** — this note is deliberately
narrow: it is the *data + training* story, not the architecture or the
general-ecosystem story. Cross-references, not duplication:
- `integral_reference/04-nnue-architecture.md` and
  `05-nnue-inference-accumulator.md` already own the network-*shape*
  timeline (hidden-layer sizes, bucket counts, quantization) in depth — §5
  below only recaps enough of that arc to anchor the data story in time,
  and flags one small reconciliation (a pre-main prototype commit note 04
  cites vs. the mainline merge this note cites).
- `integral_reference/10-ecosystem-force-multipliers.md` already owns
  OpenBench/SPSA/testing-infrastructure evidence and the "net swap is a
  one-line commit" mechanism (its §"comparison table" row + "Adapt" item 3,
  citing PR #212) — §6 below adds only what that note does *not* cover: the
  offline weight-preprocessing tool that makes the one-line commit possible.
- Our own design docs this note is calibrated against: `design/decisions.md`
  D28 (bootstrap strategy), D29 (research data platform), D34
  (search-integrated NNUE bet); `design/15-nnue-datagen.md` (format/curation
  spec); `design/16-research-data-platform.md` (three-store model);
  `design/17-nnue-research.md` (the research thesis).

---

## 0. The headline, up front

Integral's NNUE did not happen as one bootstrap event followed by
maintenance. It happened as a **closed loop, run continuously for at least
19 months and still running** at the end of this clone's history
(2026-04-02): generate self-play data with the current best engine → train
a candidate net with `bullet` → gate it (OpenBench SPRT, per note 10) → land
the survivor as a near-zero-diff commit → repeat. The distinctive thing this
note adds evidence for is that **the data/training *knobs* — not just the
network *shape* — were iterated the same disposable-branch way** as
everything else in this project: the WDL-vs-eval blend, the eval-scale
constant, the opening-randomization ply count, and even the *source* of the
bootstrap data (own self-play vs. real Lichess games) were each tried on
throwaway branches and mostly discarded. The mainline `git log` (19 net/arch
commits, README §"NNUE") is the tip of an iceberg; the **945-branch,
669-commit** ratio (§5) is the base.

---

## 1. Origin: from a texel-tuned HCE to first NNUE in five days

Tag `v4` (`d4359007`, 2024-09-01, "Data Generation (#51)") is the last
purely hand-crafted-eval (HCE) tagged release. That single PR introduces
`src/data_gen/` **wholesale** — both output formats at once (`format/
binpack.h` *and* `format/fens.h`, §3) — by wiring a self-play loop directly
on top of the already-mature board/search code (no new chess logic needed,
only a game-driver and two serializers). Five days later, `99b7b22e`
("NNUE (#52)", 2024-09-06, `Bench: 3382771`) lands the first trained
network: `768 → 256`, single hidden layer, 8 output buckets, no king
buckets, no horizontal mirror yet (`nnue/arch.h` at that commit:
`kHiddenLayerSize=256`, `kOutputBucketCount=8`,
`kHiddenLayerQuantization=255`, `kOutputQuantization=64`, `kEvalScale=200`).

**Reconciling with note 04's timeline**: note 04 §8 cites `cd3f7acb`
("first nnue", 2024-08-27) as "first NNUE at all". That commit is real but
is **not an ancestor of `main`** (`git merge-base --is-ancestor cd3f7acb
main` fails) — it lives only on prototype branches (`256_400s`, `accum_fix`,
`beta_95`, `nnue2`, `nnue_16ply`, …). So the accurate sequence is: a ~10-day
prototype incubated off-branch (Aug 27) → cleaned up and merged to `main`
as the "official" first NNUE nine days later (Sep 6). Both dates are
correct; they describe different points in the same short incubation.

**README, verbatim in spirit**: *"The first iteration of Integral's neural
network was trained on data from version 4, which had a powerful
hand-crafted evaluation."* This is the single most important sentence in
this note for calibrating our own plan: **Integral bootstrapped its first
NNUE from its own texel-tuned HCE, self-labeling its own self-play games —
no foreign teacher engine was ever used.** Compare directly against our
**D28(a)**: we deliberately add a *locally-compiled Stockfish* as a teacher
to de-risk the NNUE-infrastructure phase before ever training on our own
labels. Integral is proof that step is not strictly necessary to reach
top-tier — it is an *optional accelerant/insurance policy* we've chosen to
buy, not a precondition the one successful precedent we're studying
actually required. (See item 1 of §8 below.)

**The data-source decision was itself tested, not assumed.** Two branches,
both dated 2024-08-08 — three weeks before the prototype NNUE and 24 days
before the merged one — represent a direct A/B on bootstrap data source:
`origin/own_hce_data` (own self-play, HCE-labeled) vs. `origin/
lichess_hce_data` (real Lichess games, HCE-labeled). The shipped pipeline
(README, and the absence of any Lichess-derived corpus anywhere in the final
`src/`) shows **self-play won**: no filtered-real-game corpus survived into
the production pipeline. This is a second, independent confirmation of the
self-play-centric design our own **D29** already commits to — a real
competitor tried the "mine real human games" alternative and did not ship
it.

The precursor branch to `#51` (visible via `git log --oneline
origin/own_hce_data ^main`) carries commit messages `"stuff"`, `"yuh"`,
`"bruh"`, `"Working data generation"` before being squashed into the
presentable, `Bench:`-trailered PR that actually landed. Worth remembering
when reading a polished mainline history: the real process is messier than
the merged artifact, on every project.

---

## 2. The self-play engine (`data_gen.cc`/`data_gen.h`)

### 2.1 Configuration surface

`data_gen::Config` (`data_gen.h:17-25`) is a flat struct: soft/hard
per-move node limits, a target game count, a thread count, a `[min, max]`
random-opening-ply range, an output path, and an optional opening-book
(FEN list) path. It is exposed as a **UCI-shaped `datagen` command baked
directly into the engine binary** (`uci.cc:124-146`) — `datagen
soft_limit=X hard_limit=Y games=N threads=T min_moves=A max_moves=B
out=path [book=file]` — not a separate tool, not an external
orchestration script. `stop datagen` (`uci.cc:148-155`) sets the same
atomic flag Ctrl-C's `SIGINT` handler sets (`data_gen.cc:321-323`), so a
running bulk job can be halted cleanly from either source.

**Own-words restatement of the threading/output model**
(`data_gen.cc:325-441`): split `num_games` evenly across `num_threads`;
each worker thread opens its own temp file, runs its full share of games
end-to-end (own `Searcher`, own 16 MB TT, own RNG seeded by thread id +
wall clock), and on completion the main thread concatenates every worker's
temp file into one final output and deletes the temps. No shared queue, no
cross-thread synchronization during play at all beyond two relaxed atomics
for progress counters (`positions_written`, `games_completed`) — the
parallelism is embarrassingly-parallel by construction, which is exactly
what unlimited (or contributed) CPU cores want.

### 2.2 Opening diversification — two modes, one goal

**No book** (`FindStartingPosition`, `data_gen.cc:52-134`, no-book branch,
my own restatement): pick a random target ply count uniformly from
`[min_move_plies, max_move_plies]`, then for each ply, bucket the legal
moves by moving-piece type and pick a piece type from a **fixed discrete
distribution weighted `{35, 25, 25, 5, 5, 5}`** over
`{pawn, knight, bishop, rook, queen, king}` (`data_gen.cc:106-107`,
`kPieceProbabilities`), then among that piece's moves pick a destination
square weighted toward the board's center-ish squares (`kCenterScore`, a
hand-authored 64-entry table peaking at 9 around e4/d5-ish squares,
`data_gen.cc:18-27`; `SelectPreferredMove`, `data_gen.cc:30-50`). If the
game runs out of legal moves before the target ply (checkmate/stalemate),
restart from the start position. This is a deliberately *biased* random
walk — heavily pawn/minor-weighted, centrally-weighted — not a uniform
random legal-move walk, presumably to avoid the kind of degenerate,
edge-hugging openings a uniform walk would occasionally produce.

**With a book** (`fens_file` set, e.g. `UHO_Lichess_4852_v1`, README): pick
a random FEN line from the book, then play only **3-4** further uniformly
-random legal plies on top (README's stated number; the same
`min_move_plies`/`max_move_plies` config field does double duty for both
modes). The book already supplies well-distributed, roughly-balanced
midgame-ish diversity, so far fewer extra random plies are needed than the
no-book mode's from-scratch walk requires. `Generate` (`data_gen.cc:349-359`)
loads and shuffles the whole book once at startup and each worker draws
from it independently.

**The balance gate** (`GameLoop`, `data_gen.cc:243-249`): after finding a
starting position, run one *real* search — depth 10, capped at 1,000,000
nodes — and **reject-and-retry the entire opening** if
`|initial_score| ≥ 300` cp (`kInitialScoreThreshold`, `data_gen.cc:221`).
This is a full-search-based sanity check, not a static-eval heuristic —
expensive per rejected attempt, cheap in aggregate since only openings that
are already meaningfully unbalanced get thrown away.

### 2.3 The game itself — fixed node budgets, not depth or time

Every recorded game runs at a **fixed per-move node budget**
(`config.soft_node_limit`/`hard_node_limit` → `search::TimeConfig`,
`data_gen.cc:223-224`), not depth or wall-clock time. This is the concrete,
practical reason self-play datagen scales across heterogeneous, even
volunteer-contributed, hardware for free: a "5000-soft/N-hard-node game" is
the same amount of *work* on a laptop and a server, whereas a
depth-or-time-based game would run at different real strengths (and
different wall-clock costs) on different machines. (Our own `design/15`
§3.3.2 independently arrives at the identical convention — "fixed nodes,
default 5000" — for the *teacher* side of Tier-A; Integral applies the same
idea on the *generator* side too.)

### 2.4 Adjudication — the exact thresholds

`GameLoop` (`data_gen.cc:217-314`), constants at the top of the function:

| Constant | Value | Effect |
|---|---|---|
| `kWinThreshold` | 2500 cp | score magnitude that starts counting toward an early win |
| `kWinPliesThreshold` | 5 | consecutive plies at/above `kWinThreshold` (same side) before adjudicating a win |
| `kDrawThreshold` | 2 cp | score magnitude that starts counting toward an early draw |
| `kDrawPliesThreshold` | 8 | consecutive plies at/below `kDrawThreshold`, **and** only once `half_moves ≥ 200` (i.e. past move 100) |
| `kInitialScoreThreshold` | 300 cp | the opening-balance gate, §2.2 |

Mate scores (`|score| ≥ kMateInMaxPlyScore`) are adjudicated immediately as
a win/loss for the mating side — no ply-streak needed. Draws by
repetition/insufficient-material are checked **directly on the board**
after every move (`data_gen.cc:291-294`), not inferred from the search's
internal draw score, "since search doesn't terminate with an adjudicated
draw score at root" (the code's own comment) — a small but important
precision: the *label* comes from ground-truth board state, the *decision
to keep playing* comes from the search.

Compare to our own `design/15` §3.4: `|score| ≥ 2500 cp` for 4 consecutive
plies (win), `|score| ≤ 8 cp` for 10 consecutive plies after ply 80 (draw).
**These are strikingly close to Integral's own numbers** (2500 cp win
threshold identical; draw-streak length and post-ply gate in the same
neighborhood, 8 vs 10 plies / ply 80 vs ply 100) despite being designed
independently — good convergent validation that these are sane defaults,
not a number to second-guess. The one place we are *tighter* is the balance
gate: our doc's `±600 cp` opening-reject threshold is roughly double
Integral's `300 cp` — worth revisiting downward once our own review sample
(§3.6) is in hand, since Integral's real production pipeline runs stricter.

### 2.5 Syzygy reuse — "build once, benefit twice"

Syzygy tablebase probing (`Syzygy TB Support`, PR #27, `e21ffa3b`,
2024-07-27) landed over a year before any NNUE work, as ordinary in-search
plumbing (`search.cc:587-634`: any node with ≤ 7 men on the board and
sufficient remaining depth gets probed, and the returned WDL tightens
`alpha`/`beta` or the final score). Because `data_gen`'s `GameLoop` runs
self-play through the exact same `Searcher`, **every self-play game
automatically gets TB-accurate endgame play and scoring with zero
datagen-specific code** — the README's "5-man Syzygy endgame tablebases are
used to guide the data generation search" is not a special integration, it
is a side effect of TB support already existing generically. This is a
clean instance of a general lesson: infrastructure built for *play*
strength pays a second, unplanned dividend in *data* quality, for free,
the moment self-play reuses the same search — and it is specifically the
kind of dividend our own project currently forgoes by design (Syzygy is
deferred past P13 per **D31(c)**, on `unsafe`-FFI/zero-runtime-deps
grounds, D14/D15/D19). Endgame data quality is a real, named cost of that
choice, not a hypothetical one — Integral shows concretely what it buys.

---

## 3. Two on-disk formats, one lesson: match the format to the era

`src/data_gen/format/format.h` defines a tiny abstract interface (my own
restatement, not copied):

```
trait OutputFormatter {
    fn set_position(&mut self, state: &BoardState);
    fn push_move(&mut self, mv: Move, turn: Color, score: Score);
    fn write_outcome(&mut self, wdl: f64) -> u64; // -> positions written
}
```

Both eras of Integral's datagen — the pre-NNUE, HCE/texel-tuning era and
the NNUE era — plug a *different implementation of this same three-method
interface* into an otherwise **unchanged** self-play loop. That is the
whole lesson of this section: the format is a strategy-pattern seam, not a
rewrite, and it cost Integral near-zero extra engineering to switch data
eras. Both formatters were introduced in the *same* PR (`d4359007`, §1) —
not added sequentially as the need arose.

### 3.1 `FenFormatter` (`format/fens.h`) — the HCE/texel-tuning-era format

Filters at **write time**: skip any position where the move about to be
made is a capture/en-passant/promotion, or the position itself is in check
(`fens.h:20-27`) — i.e. record the position *before* a quiet move, never a
position on the edge of a tactic or currently in check. Caps at **50**
positions per game, chosen by shuffling the whole game's eligible FEN list
and taking the first 50 (`fens.h:36-41`) — an intra-game-correlation limiter
almost identical in spirit to our own `design/15` §3.5 item 6
(`P_cap`, default 10). Output is plain text: `<FEN> [<wdl>]` per line.

### 3.2 `BinPackFormatter` (`format/binpack.h`, the "Marlin" format) — the NNUE-era format

A **game-oriented**, not position-oriented, packed binary format:

- One `MarlinChessBoard` header per game (`binpack.h:67-76`, `__attribute__
  ((packed))`): a 64-bit occupancy bitboard, a 4-bit-per-occupied-square
  piece/color nibble array (`U4Array<32>`, `binpack.h:41-65` — a small
  packed-nibble accessor type with a `const`/mutable split via a private
  proxy object, itself a nice tiny idiom: read returns a plain `uint8_t`,
  write goes through an `IndexedU4` proxy that knows which nibble it owns),
  turn+en-passant-file in one byte, halfmove clock, fullmove number, and
  *placeholder* evaluation/WDL fields filled in only once the game ends.
  Castling rights are folded into the piece encoding itself via an
  **"unmoved rook" piece ID** (`binpack.h:141-153`): a rook that still has
  its corresponding castling right gets piece-type id 6 instead of the
  normal rook id, so castling-rights state needs no separate field at all —
  a compact, if slightly cheeky, encoding trick.
- One `(move: u16, score: i16)` pair per ply after that (`binpack.h:78,
  98-115`), terminated by a sentinel null-move pair (`binpack.h:80,125`).
  The move re-derives the packed rank/file/promotion/kind bits from
  scratch per ply rather than storing full board snapshots — the whole
  game is reconstructible by replaying moves from the header position.
- One trailing WDL byte per game (`0`=loss, `1`=draw, `2`=win, White
  -relative, `binpack.h:117-129`).

Crucially: **`PushMove` records every ply unconditionally** — no
quiet/no-check filter analogous to `FenFormatter`'s (contrast `binpack.h:98
-115` with `fens.h:18-28`). Filtering for this format is deferred
downstream, to whatever consumes the binpack (a training-time replay pass,
or — per our own `design/15` §3.2.3's characterization of this whole format
family — the trainer's own binpack loader, which can re-derive quiet
positions, dedupe, and even re-score at load time because it has the *whole
game*, not a pre-filtered slice).

### 3.3 The design trade-off, and which side we're on

This is a genuine trade-off, not a strict improvement in either direction:

| | Filter-at-write (flat record, `FenFormatter` / our bootstrap) | Filter-at-load (game-oriented binpack) |
|---|---|---|
| Storage density | Larger (full FEN text, or 32 B/position even packed) *per retained position*, but only retained positions are stored at all | Much denser per *ply* (a few bytes), but **every** ply is stored, filtered or not |
| Simplicity to emit correctly | Trivial — one filter check, one line | More state (whole-game replay, sentinel handling) |
| Flexibility to re-derive labels/filters later | None — what you filtered out is gone | Full — can re-run a different filter or a different teacher's rescoring over the same corpus without regenerating games |
| What we chose (bootstrap) | **This one** — `bulletformat::ChessBoard`, `design/15` §3.2.1/§3.2.3 | Named explicitly as the fallback "if corpus storage or relabelling throughput ever dominates" (§3.2.3, open question 4) |

Integral is a real, shipping, production-scale confirmation that the
"fallback" side of this trade-off is a genuinely viable, load-bearing path
once volume/relabeling needs justify it — **and** that you do not need to
start there: their own first bootstrap (the `FenFormatter`, HCE era) used
the *simpler*, filter-at-write, flat-text shape, and only adopted the
denser game-oriented format once the NNUE era's data volumes made it worth
the added complexity. That maps directly onto our own staged plan (`design/
15` §3.6: smoke → review sample → ~100 M bootstrap corpus) — start flat,
graduate to binpack-shaped only if/when the numbers say so, exactly the
order Integral actually used.

---

## 4. Training-signal purity: search heuristics vs. the training label

`AdjustStaticEval` (`search.cc:220-245`) applies three corrections to the
static eval on top of the raw NNUE output, in order: (1) a material-phase
scale (`kMaterialScaleBase` + a weighted piece count, `/32768`), (2) a
correction-history adjustment (`CorrectStaticEval`), (3) a fifty-move-rule
decay (`* (220 - fifty_moves_clock) / 220`). Steps **(1) and (3) are
wrapped in `#ifndef DATAGEN`** (`search.cc:225,239`) — compiled out of any
binary built with the `-DDATAGEN` flag (`option(DATAGEN OFF)`,
`CMakeLists.txt:161-164`; `Makefile:22-27`). Step (2), correction history,
is **not** gated — it applies identically in both build modes.

**Own-words statement of the principle this encodes**: a heuristic that
exists to shape the *search's decision policy* — "don't let the static eval
drift toward a draw-avoidant or draw-seeking bias as the fifty-move clock
approaches" (3), "discount an eval swing if material has already been
mostly traded off" (1) — is a form of contempt/risk management. It is
legitimate for choosing which *move* to play, but if it leaks into the
*score* that a self-play game then writes into tomorrow's training corpus,
the net is handed a label that encodes the search's risk policy as if it
were a fact about the position, and will learn to reproduce that bias as
"evaluation." A heuristic that instead exists to make the static eval a
*more accurate estimate of the position* (2, correction history — pattern
-based bias correction learned from search-vs-static disagreement) has no
such contamination risk and stays on in both modes. The dividing line is
**"does this term change what move I'd play, or does it change what I
believe the position is worth" — only the latter belongs in a training
label.**

**This was not a day-one design decision.** The `DATAGEN` build flag and
its two `#ifndef` guards were added by PR #174, "Disable evaluation scaling
during data generation" (`45ca6386`, 2025-02-26) — roughly **5.5 months
after** the first NNUE net shipped (Sep 2024) and after several net
iterations had already trained on data generated *without* this guard. This
is worth remembering precisely because it is reassuring, not alarming: even
a project that reached top-6 discovered and fixed this class of bug
mid-stream rather than anticipating it from the start. The discipline that
matters is noticing the problem and being able to *cheaply iterate a fix*
(a build flag + a two-line guard, one PR) — not never making the mistake.

**What this means for our own pipeline.** Our Tier-A bootstrap (`design/15`,
D28a) is immune to this specific bug by construction: the label comes from
a wholly separate Stockfish-teacher process, so none of *our* search's
decision-policy heuristics ever touch a Tier-A label. But the moment Tier-B
begins — our own engine's self-play, self-labeled, per D28(b)/D29 — this
exact class of bug becomes live risk for us for the first time, and
Integral's PR #174 is the concrete template for the audit to run *before*
that first Tier-B corpus is generated: enumerate every place a heuristic
adjusts the score that a self-play game would record, and classify each one
as "decision policy" (gate it out of the recorded label) or "evaluation
accuracy" (keep it in). See item 2 of §8.

---

## 5. The retraining loop and the branch graveyard

### 5.1 The loop, as stated and as observed

README, restated: each network iteration after the first is trained on a
**fresh** dataset generated by the **prior** network. This is a literal,
running instance of the AlphaGo→AlphaZero-style bootstrap our own **D28**
names as the target shape — except Integral runs the *entire* arc
self-referentially (HCE self-play → net 1 → net-1 self-play → net 2 → …)
with no foreign-teacher step anywhere in it. The 19 net/architecture
commits on `main` across ~10 months (README's arc, cross-referenced in
depth by `04-nnue-architecture.md` §8 and `05-nnue-inference-accumulator.md`
§12) are the visible, *merged* instances of this loop.

### 5.2 The branch census — the actual unit of "one experiment"

`git branch -a` on this clone lists **945 remote branches** against **669**
commits reachable from `main` — a ratio of roughly **1.4 experimental
branches per merged mainline commit**. This number does not appear in any
of this directory's other notes and is, in this note's reading, the single
most important piece of *process* evidence specifically about the
data/training side of the project (as distinct from note 10's
testing-infrastructure evidence): **the branch, not the commit, is where
almost all of the actual data/training research happens**, and the
overwhelming majority of it never reaches `main` at all.

Curated, dated sample of NNUE/data-relevant branch names (`git branch -a`,
`git log -1 --format="%ad %s" --date=short origin/<branch>` per row) —
every date below was independently pulled from the branch tip's own commit,
not inferred:

| Branch | Date | What it evidently swept |
|---|---|---|
| `nnue_zero_wdl`, `nnue_full_wdl` | 2024-09-03 | WDL-vs-eval blend, 0% and 100% game-result weight — 3 days *before* the merged first net |
| `0.8wdl`, `100wdl`, `100wdl_2`, `100wdl_scaled` | (undated in this sample) | further WDL-blend percentages |
| `nnue_200_scale`, `300_scale`, `400_scale` | (undated) | the final centipawn-scale constant |
| `3ply`, `4ply_conthist`, `6ply`, `6ply_ch`, `6ply_mp`, `nnue_16ply`, `ply_history` | (undated) | the random-opening-ply count (our own `design/15` §3.4 `R_open`) |
| `128hl_300`, `1280hl_500sb`, `nnue_256`, `nnue_256hl` | — | hidden-layer-width alternatives around the shipped 256→1280→1536 arc (04 §8) |
| `7buckets` | 2024-10-17 | king-bucket-count alternative (shipped: 9-factorized, then 12) |
| `8bucket_factorized` | 2024-10-08 | ditto |
| `12kb_net` | 2024-11-16 | 12 king buckets tried standalone, ~6 weeks before it landed bundled into `581cebc1` |
| `13kb_net` | 2024-11-15 | 13 king buckets — tried, **not** adopted (shipped settled at 12) |
| `own_hce_data`, `lichess_hce_data` | 2024-08-08 | bootstrap data *source* (§1) |
| `leela_net`, `vine_net`, `vine_net_scaled`, `yoshie_net` | 2024-09-10 .. 2025-12-04 | named alternative net/architecture experiments, at least one (`yoshie_net`) plausibly tied to outside contributor `Yoshie2000` (note 10 §5) |
| `hardswishswiglu` | **2026-04-02** | a gated-activation (SwiGLU-family) experiment — the single most recent commit in this entire clone |

Sample WIP commit-message texture on these branches (not cleaned up, since
they never became a PR): `"ok"`, `"inference"`, `"stuff"`, `"yuh"`,
`"bruh"`. Same pattern as §1's precursor branch — real solo-dev process is
informal on the branch and only gets polished at the point something is
promoted.

### 5.3 The loop has no finish line

The `hardswishswiglu` branch (2026-04-02) is dated to the exact same day as
the most recent commit reachable from `main` in this clone — i.e., as of
the end of this repository's recorded history, the developer is *still*
actively branch-experimenting with a new activation function, **19 months**
after the first shipped net and **7 mainline architecture generations**
(the "Zekrom v2"..."v7" cadence, `04`/`05`/`10`) into the project. There is
no visible "NNUE is done" commit or milestone anywhere in this history —
the loop of §5.1 simply keeps running. This matters for how we should
*budget* P13: not as a milestone with a defined exit, but as the kind of
open-ended research thread our own **D34** already frames it as.

---

## 6. The one-line net swap — the missing piece from note 10

`10-ecosystem-force-multipliers.md` already documents the headline
mechanism precisely: once `NETWORK_NAME`/`EVALFILE`/an external
`aronpetko/integral-networks` GitHub Releases repo exist (`CMakeLists.txt:
23-48`), shipping a new net becomes a **one-line `NETWORK_NAME` diff**
(verified directly: `git show --stat` on `f51c7200`, `8215e459`, `ecb262bd`,
`733036df` each shows exactly 1-4 changed lines in `CMakeLists.txt`/
`uci.h`, §"the timeline" table in §1 above). What that note does not cover,
and what belongs squarely in this note's data/training remit, is **how the
raw trained weights get from `bullet`'s output into that exact one-line
-swappable, SIMD-ready shape**: a small, separate, disposable build tool.

`preprocess/net_processing.cc` (introduced by PR #190, `576bef0b`,
2025-04-24, "Pre-process the network and use it directly from the binary")
is a standalone `main()` — `preprocess <input.nnue> <output.nnue>` — wired
into the main `CMakeLists.txt` as a build-time custom command
(`CMakeLists.txt:70-83`: `add_custom_command` running `preprocess` on
`EVALFILE`, output feeding `-DEVALFILE=<processed path>`). Own-words summary
of what it does to the raw trained network before it is ever embedded in
the engine binary:

1. Feature-transformer weights/biases: copied as-is, except on
   SIMD-capable, non-sparse-permute builds, where 128-bit blocks are
   reordered via a fixed `kPackusOrder` permutation table — pre-arranging
   the packed-int8 conversion order the runtime FT update will need, so the
   hot path never has to reorder at inference time.
2. L1 weights: **transposed** from `[bucket][l2][l1]` (the natural shape a
   trainer emits, one row per output neuron) to `[bucket][l1][l2]` (the
   shape the engine's dot-product loop actually walks).
3. On SIMD builds, a **second** permutation of the (now-transposed) L1
   weights, grouped in 4-wide chunks, specifically to match the operand
   layout `_mm_dpbusd_epi32`-style int8 dot-product instructions expect.
4. L2 weights: transposed `[bucket][l3][l2] → [bucket][l2][l3]`, same
   reasoning as (2).

**Why this matters for the "one-line commit" story**: it means none of
this format-adaptation work happens in `bullet`'s own export step (which
stays a generic, trainer-agnostic exporter), and **none of it happens at
engine-process startup** (which would otherwise cost real load-time work
every single run, on every user's machine, for a transform that is the same
every time for a given network). It happens **once, offline, per network
release**, as its own tiny standalone binary — which is exactly what lets
the actual engine-repo commit that "ships" a new net stay a one-line
pointer change. The lesson generalizes past NNUE specifically: *any*
foreign-artifact reshaping that is expensive-but-deterministic belongs in a
disposable offline tool, not in the hot load path and not in the training
tool.

**One thing to notice, not to imitate**: like Reckless (`reckless_reference/
13-nnue-inference-simd.md` §7, "no format validation at all") Integral's
net loading is a raw, untagged struct read (`RawNetwork`/`Network`,
`shared/nnue/definitions.h:27-50`; `net_processing.cc:94-98` reads straight
`ifstream::read` into a `RawNetwork*`) — no magic number, no version tag, no
dimension check, no checksum. Any struct-layout drift between the training
repo's exporter and this repo's `RawNetwork` definition is silently
undetectable. Our own plan is already stricter here: `design/11` §3.4.5's
CENN container (magic + version + dims + activation id + checksum) is a
real, deliberate improvement over what *either* C++ engine we've studied
actually ships — worth stating plainly as "we made the right call, keep
it," not something to loosen for parity.

---

## 7. Comparison table

| Aspect | Stockfish | Reckless | Integral | Us (planned — D28/D29, `design/15-17`) |
|---|---|---|---|---|
| First-net bootstrap teacher | its own strong search, at a scale (billions of positions, years of accreted data) no small project can replicate | not covered in our snapshot | **its own texel-tuned HCE (v4)**, self-labeling its own self-play — no foreign teacher at all | a **locally-compiled Stockfish** as teacher (D28a) — an *optional accelerant* Integral's own history shows is not strictly required |
| Ongoing-net data source | own engine self-play (further "zero"-style iterations reported in the community) | not covered here | own engine self-play **exclusively** — a real Lichess-derived alternative was branch-tested (§1) and not shipped | own engine self-play, Tier-B (D28b/D29) — same conclusion, arrived at independently |
| Corpus format, over time | its own binpack-family formats (not examined here) | not covered here | two eras: flat, filtered, textual FEN+result (HCE era, `fens.h`) → dense, **unfiltered**, game-oriented binpack ("Marlin", NNUE era, `binpack.h`) | flat 32-byte `bulletformat::ChessBoard` for the bootstrap (`design/15` §3.2.1); binpack explicitly named as the fallback "if storage or relabelling throughput ever dominates" (§3.2.3) — same order Integral actually used |
| Write-time filtering | (not examined) | (not examined) | yes for the flat FEN format (quiet + not-in-check, ≤ 50/game); **no** filtering at write time for the binpack format (deferred downstream) | yes, at write time, for the flat bootstrap format (quiet/no-check/score-cap/dedup/≤10-per-game, `design/15` §3.5) — closer to Integral's flat-era discipline than its binpack-era laxness |
| Search-heuristic/training-label purity | (not examined) | (not examined) | an explicit `#ifndef DATAGEN` gate, added **5.5 months** into the NNUE era (PR #174) after training on unguarded data | not yet a live risk (Tier-A's label comes from a separate SF process); becomes live at Tier-B — Integral's PR #174 is the concrete pre-emptive checklist item |
| WDL-blend / eval-scale tuning | (not examined) | (not examined) | swept via dedicated, mostly-discarded branches (`nnue_zero_wdl` … `100wdl_scaled`; `200_scale`/`300_scale`/`400_scale`), shipped value `kEvalScale = 200` | the same knobs are named (`design/15` §3.2.4/§3.2.5, bullet's `--scale`/`--wdl`) but **not yet swept** — currently assumed `SCALE=400`; treat as unverified, not inherited (§8) |
| Opening-randomization tuning | (not examined) | (not examined) | swept across at least 5 distinct ply counts on branches (3/4/6/16 plies) before landing on "book + 3-4 plies" | `design/15` §3.4 currently plans `[8,10]` plies, no book — a reasonable *starting* guess, not yet empirically tuned |
| Net file format validation | own versioned format (not examined here) | none — raw `transmute`, no magic/version/checksum (`reckless_reference/13` §7) | none — raw `ifstream::read` into a struct, no magic/version/checksum | **CENN**: magic + version + dims + activation id + checksum (`design/11` §3.4.5) — stricter than either C++ engine studied |
| Net-swap mechanics | (not examined) | `build.rs` fetch + `include_bytes!` + `transmute`, same shape as Integral | offline preprocessing tool (§6) + external GitHub Releases repo + a 1-line version-name diff to ship (note 10) | open question in `design/00-master-plan.md` P13 row — both C++ engines converge on the same answer |
| Retraining cadence | continuous, large team, decades | not covered | ~19 net/arch commits to `main` over ~10 months, but **945 branches** total and still actively branching 19 months later | not yet begun (P13); D34 already frames this as perpetual, matching what's actually observed here |

---

## What this means for us as a small project

Ordered roughly by leverage-per-effort, cross-referenced to our own
`design/`, `CLAUDE.md`, and the roadmap.

1. **Treat the SF-teacher bootstrap (D28a) as insurance, not a
   prerequisite — and budget a cheap self-labeled fallback alongside it.**
   Integral's entire NNUE arc started from its own texel-tuned HCE labeling
   its own self-play, with no foreign engine involved at all, and reached
   top-6 that way. If our Stockfish-teacher pipeline (`design/15`) hits
   friction — build issues, licensing-review delays, throughput problems —
   a "self-play labeled by our current PSQT + search" bootstrap is a
   validated fallback path with real precedent, not a downgrade. Doc:
   `design/15` §3.1/§9 (open questions), D28.

2. **Write the "search-heuristic vs. training-label purity" audit into
   `design/15`/`design/16`'s acceptance criteria *now*, before Tier-B's
   first corpus, not after.** Integral discovered this bug class 5.5
   months into its NNUE era (PR #174, §4). We can pre-empt it: when Tier-B
   self-play begins, enumerate every place our own eval/search pipeline
   adjusts a score before it would be recorded (correction history and any
   future contempt-like, fifty-move, or material-scaling term) and classify
   each as "decision policy" (must be excluded from the recorded label,
   e.g. via a `cfg`/feature-gated datagen build) or "evaluation accuracy"
   (keep it). Doc: `design/16` §3.4/§8 (acceptance criteria) — add this as
   a named checklist item.

3. **Don't inherit Integral's WDL-blend or eval-scale numbers — inherit the
   sweep methodology.** Their shipped `kEvalScale = 200` sits at half our
   own `design/15` §3.2.4/§3.4.7's assumed `SCALE = 400`, and their final
   WDL blend was reached only after multiple dedicated, mostly-discarded
   branch experiments (`nnue_zero_wdl` … `100wdl_scaled`), not by
   inspection. Budget an equivalent small sweep (a handful of short,
   throwaway training runs at different `--wdl`/`--scale` values, gated by
   whatever proxy-SPRT or eval-quality metric is cheapest at that stage)
   before locking either constant for our own first net.

4. **Treat the opening-randomization ply count the same way.** Integral
   tried at least five distinct values (3/4/6/16 plies, plus book-vs-no
   -book) on branches before settling on "book + 3-4 plies." Our
   `design/15` §3.4 `[8, 10]`-plies-no-book default is a reasonable
   starting point but should be revisited once the "review sample" step
   (§3.6, ~0.5-1 M positions inspected before the bulk run) is in hand —
   exactly the point our own doc already schedules a human-in-the-loop
   check.

5. **Wire the same opening book into datagen, not just into gauntlet/SPRT.**
   Per note 10's "we already made the right call on the book" item, we
   already reference `UHO_Lichess_4852_v1` in `tools/sprt.sh`/gauntlet — the
   same book Integral's README names for datagen. We currently only use it
   for one of the two purposes; Integral uses one book for both. Doc:
   `design/15` §3.4 (currently plans a random-walk-only default, book
   "selectable by CLI... once wired").

6. **Give our own datagen tool the same pluggable-formatter seam,** even
   though we start with only one output shape. Integral's tiny
   `OutputFormatter` interface (`set_position`/`push_move`/`write_outcome`,
   §3) let it add a second, denser data format later (binpack) without
   touching the self-play loop at all. If/when our own bootstrap corpus
   ever needs the binpack-shaped fallback `design/15` §3.2.3 already names
   as a possibility, having this seam from day one means adding a second
   trait implementation, not a rewrite. Doc: `design/15` §3.7 (tooling &
   module map) — worth baking the trait boundary in even for the single
   -format v1.

7. **Consider a self-hosted `datagen` engine subcommand for our own Tier-B
   self-play**, mirroring the existing `bench`/`perft` argv subcommands
   (`CLAUDE.md` "Commands"), once P13's own-engine self-play starts.
   Integral never leaves its own binary to generate its bulk corpus — no
   external orchestration harness, just threads + temp files + a
   concatenation step. This is simpler and faster than round-tripping
   through a scripting-language harness *once the "teacher" is our own
   engine* — it does **not** apply to Tier-A, where the teacher genuinely
   is a separate Stockfish binary and our current python-chess-based
   orchestration (`design/15` §3.7) is correctly scoped there.

8. **File the offline-weight-preprocessing pattern into `design/11`'s P13
   planning explicitly**, even though we've already chosen the stricter
   CENN-with-validation path over Integral's/Reckless's raw-transmute
   approach (keep that choice — §6 above). The *shape* worth adopting
   regardless is: any one-time, deterministic, SIMD-layout-specific weight
   transform (transpose, byte-permutation for a specific dot-product
   instruction) belongs in its own small offline tool that runs once per
   net release, not in the trainer's export step and not in the runtime
   loader's hot path. Doc: `design/11`'s module map (`netpack`/`loader.rs`
   line) — note this explicitly as the tool's responsibility boundary.

9. **Expect (and don't be alarmed by) a branch-to-mainline-commit ratio
   well above 1:1 once our own net/data experimentation begins.**
   Integral's observed ratio is roughly 1.4 discarded-or-parked branches
   per merged mainline commit, sustained for the entire NNUE era. That is
   a healthy signal of real experimentation, not wasted effort — plan
   review/synthesis time (D33's multi-agent loop, or a lighter equivalent
   for data/training-only changes) accordingly rather than expecting most
   experiments to land.

10. **Budget P13 (and its successors) as an open-ended research thread,
    not a milestone with a defined finish line.** The most recent commit
    in this entire clone (2026-04-02) is a branch experimenting with a new
    activation function, 19 months and 7 architecture generations after
    the first shipped net — there is no visible "NNUE is done" point
    anywhere in this history. This matches, and is independent supporting
    evidence for, our own **D34** framing of search-integrated NNUE as a
    perpetual co-evolution rather than a phase to complete once.
