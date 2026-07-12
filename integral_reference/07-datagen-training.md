# Integral — datagen + NNUE training pipeline

Source: `integral_reference/integral` (aronpetko/integral, top-6 C++ engine,
solo dev, gitignored clone). **No explicit license → all-rights-reserved by
default.** Stricter than our GPL/AGPL treatment of Stockfish/Reckless (D27):
nothing below is copied code, not even short snippets — every algorithm is
retold in our own words / own-notation pseudocode, every data layout is a
prose table of field *meanings*, and every number is a fact (facts aren't
copyrightable) cited back to a `file:line` or a `git` rev so a future reader
can re-open the exact source. Primary files: `src/data_gen/{data_gen.h,
data_gen.cc}`, `src/data_gen/format/{format.h,fens.h,binpack.h}`,
`src/engine/search/search.cc` (the `DataGenStart` entry point + the
in-search Syzygy probe), `src/engine/search/syzygy/{syzygy.h,syzygy.cc}`,
`src/engine/uci/uci.cc` (the `datagen` UCI command + `SyzygyPath`/
`SyzygyProbeDepth` options), `preprocess/net_processing.cc`, `CMakeLists.txt`
(net download/versioning), `README.md`. Git history explored read-only via
`git -C integral_reference/integral log`/`show` (669 commits on `main`, 946
remote branches — see §8).

Cross-refs into our own tree: `design/15-nnue-datagen.md` (our bullet
contract + curation spec — this is the doc every finding below is measured
against), `design/16-research-data-platform.md` (D29, the observation-store
umbrella), `design/11-evaluation.md` §3.4.7 (our own clean self-play
generator spec), `design/decisions.md` D24/D25/D27/D28/D29,
`stockfish_reference/NOTES.md`, `reckless_reference/NOTES.md` +
`00-design-implications.md`.

---

## 1. What the README claims vs. what the code does

`README.md:21-26` gives the public one-paragraph recipe: "tens of millions
of self-play games", each starting from "3-4 randomly selected moves off a
randomly selected opening from the UHO_Lichess_4852_v1 book", "5-man Syzygy
endgame tablebases... guide the data generation search", trained with
`bullet`, first iteration from HCE (v4) data, "each iteration since then...
trained on a fresh dataset using the prior network."

The code confirms every clause but adds precision the README omits, and one
outright contradiction worth flagging:

- The **book/random-ply numbers are not compiled-in constants.** `Config`
  (`data_gen.h:21-24`) has *fallback* defaults `min_move_plies = 8,
  max_move_plies = 9`, but the `datagen` UCI command (`uci.cc:124-146`)
  declares `min_moves`/`max_moves` as `ArgumentType::kRequired` — every real
  invocation overrides them explicitly. The README's "3-4" is therefore the
  actual recipe used for the published nets, but it lives **only** in the
  README/the dev's own shell history, not in git-tracked config. There is no
  `meta.json`-style record of which exact CLI flags produced any specific
  `zekrom-vN` net (contrast with our own §3.6 discipline — see §9).
- The **"5-man Syzygy" claim is a hardware/storage choice, not a code
  constant.** The in-search probe gate (`search.cc:590`) is
  `state.Occupied().PopCount() <= 7` — a generic "≤7 total men on the board"
  early-exit before calling the real Syzygy probe (7 total pieces = 5
  non-king pieces = what tablebase folks call "5-man" tables, since Syzygy
  naming counts the two kings). The code would happily probe up to 7-man
  coverage if the files existed; "5-man" in the README just says which
  `SyzygyPath` directory (~950 MB) the dev actually pointed at — 7-man
  Syzygy tables are multi-terabyte and clearly out of a solo dev's storage
  budget. This is a nice small data point about *right-sizing infra to your
  actual hardware* rather than chasing the theoretical maximum.

## 2. The self-play game loop (`data_gen.cc`)

### 2.1 Per-game flow, own-words pseudocode

One `GameLoop` runs per worker thread (`data_gen.cc:211-319`); each owns a
private 16 MB TT (`searcher.ResizeHash(16)`, line 230) and its own
`std::mt19937` seeded from `RandomSeed(thread_id, wall_clock)` (line 215) —
no shared RNG or shared TT across workers, so a slow worker never stalls a
fast one and there's no lock contention on the hot path.

```
fn datagen_worker(cfg, thread_id, book_fens):
    seed_rng(thread_id, now())
    tt = Tt::new(16 MiB)                       // private, small, per worker
    games_share = cfg.num_games / cfg.num_threads

    for i in 0..games_share:
        loop {                                  // "was this opening already decided?" retry
            board = random_opening(cfg, book_fens)      // §3
            (initial_score, _) = search(board, depth <= 10, nodes <= 1_000_000)
            if |initial_score| < 300 { break }           // accept
            // else: silently redo this game index — reroll the whole opening
        }
        formatter.start_game(board)

        win_run, loss_run, draw_run = 0, 0, 0
        loop {
            (score, best_move) = search(board, hard_nodes = cfg.hard_limit,
                                                soft_nodes = cfg.soft_limit)
            if best_move.is_none() {
                result = if board.in_check() { win_for(!board.side_to_move) } else { DRAW }
                break
            }
            if |score| is a mate-range score {
                result = win_for(if score > 0 { White } else { Black }); break
            }
            // streak-based adjudication, §4
            bump_streaks(score, &mut win_run, &mut loss_run, &mut draw_run, board.ply)
            if win_run  >= 5 { result = decisive(winning_side); break }
            if loss_run >= 5 { result = decisive(losing_side);  break }
            if draw_run >= 8 { result = DRAW; break }

            board.make_move(best_move)
            if board.is_repetition(0) || board.is_insufficient_material() {
                result = DRAW; break
            }
            formatter.push_move(best_move, side_to_move_before_move, score)
        }
        formatter.write_outcome(result)   // per-format sampling, §5
```

Every move is chosen by the engine's **own search** under a **node budget**,
not a depth or time budget: `TimeConfig{ nodes: hard_limit, soft_nodes:
soft_limit }` (`data_gen.cc:223-224`). This is the same reasoning our own
design/15 §3.3.2 gives independently (node budgets are hardware-independent
and have near-constant per-move work, unlike depth or wall-clock) — good
convergent validation that node-limited self-play is the right default, not
just an SF-teacher-specific trick.

### 2.2 The "opening was already lost" gate

Before recording any game, a quick sanity search (depth ≤ 10 **or** 1M
nodes, whichever first — `data_gen.cc:243-244`) checks the position reached
after the random opening plies. If `|score| ≥ 300` the *entire game index is
retried* (`--i; continue;`, line 247-249) — a fresh random opening is drawn,
not just a fresh continuation from the same skewed position. This is
Integral's version of our design/15 §3.4 "balance gate" (we use `±600 cp`
against an external SF-teacher eval; Integral uses `±300 cp` against **its
own** engine's quick search) — tighter, and self-refereed rather than
teacher-refereed (see §7, this matters for the provenance story).

### 2.3 Datagen-only search behavior (`#ifndef DATAGEN`)

A dedicated CMake option `DATAGEN` (`CMakeLists.txt:161-163` sets
`-DDATAGEN`, gated in `Makefile:28` default `OFF`) compiles a *different*
search binary for datagen than for play. Inside `AdjustStaticEval`
(`search.cc:220-245`), two score adjustments are **skipped** when built with
`DATAGEN`:

1. the material-phase scaling of the static eval (`search.cc:225-233`), and
2. the fifty-move-rule proximity damping (`search.cc:239-242`).

Rationale (not stated in-repo, but the only sensible one): those two
adjustments exist purely to make the *search's decisions* more human/
practical (trust the eval less near a 50-move shuffle, weight it by
material left) — they are search-time heuristics, not claims about the
"true" static value of the position. Baking them into the *label* a network
trains to reproduce would teach the net to imitate a search-side fudge
factor instead of a clean positional judgment, and the correction would
compound every time a stronger net is retrained on labels produced by a
search that includes it. Our own design/15 doesn't have an equivalent
concern yet because Phase A/B PSQT+HCE eval doesn't have these adjustments,
but design/11 §3.6's `evaluate()` shape **does** include a rule-50 damping
term (`v -= v * halfmove_clock / 199`) and a material-scaled optimism blend
— once we're generating *our own* labels with `bin/datagen.rs` (Tier B),
this is a concrete trap to avoid (§9 Adapt list).

### 2.4 Multi-threaded generation, not a search-thread pool

`Generate()` (`data_gen.cc:325-441`) spawns `num_threads` independent
`std::thread`s, each writing to its **own temp file**
(`<path>_temp<i>`, line 364), then joins and **concatenates** the temp files
into one output (lines 403-434), deleting the shards after. No shared
output stream, no interleaving/locking during generation — the only
serialization point is progress-bar printing, guarded by a `display_mutex`
(line 137) and throttled to print at most every `num_games/50` completions
(line 309). This is a pragmatic, low-engineering-effort approach: threads
never coordinate except at the very end, and the concatenation step is where
our own "shuffle + interleave" step (design/15 §3.2.2) would also plug in.

## 3. Opening diversity — two mechanisms, one guard

`FindStartingPosition` (`data_gen.cc:52-134`) has two modes selected by
whether a `--book` file was passed (added later — `4816a902 "Allow data
generation from books"`; book support did **not** exist in the original
`d4359007` PR):

**Book mode** (`fens` non-empty): pick a uniformly random FEN line from the
shuffled book file (`Generate()` shuffles the whole book once at startup,
`std::ranges::shuffle`, `data_gen.cc:357-358`), then play a uniformly random
**legal** move for `target_plies ∈ [min_move_plies, max_move_plies]` plies.
If a mid-walk position has no legal moves, redraw a fresh book line and
restart the walk from ply 0 (lines 73-83) — never continue a random walk
that dead-ends into a mate/stalemate the walk didn't intend.

**Organic mode** (no book): starts from the normal start position and
biases the random walk twice, not once:

1. **Which piece moves** — a `std::discrete_distribution` over piece type
   with weights `{Pawn 35, Knight 25, Bishop 25, Rook 5, Queen 5, King 5}`
   (`data_gen.cc:106-114`; enum order confirmed at `src/utils/types.h:24-30`)
   — i.e. roughly a coin-flip between "push a pawn" and "develop a minor",
   rooks/queen/king moves are rare in the opening walk exactly as a human's
   intuition would want.
2. **Which square it goes to** — among the legal moves of the chosen piece
   type, `SelectPreferredMove` (`data_gen.cc:30-50`) weights each candidate
   by a hand-tuned 8×8 "center desirability" grid (`kCenterScore`,
   `data_gen.cc:18-27`): highest weight on the four true-center squares,
   tapering outward, near-uniform low weight on the back two ranks (a couple
   of squares get a small bump — clearly hand-tuned by eyeballing normal
   opening development squares, not derived from any formula).

Both modes reroll the whole opening if the piece-type bucket is empty or if
the final random ply happens to land on checkmate/stalemate (lines
127-132) — an opening walk is never allowed to accidentally *be* the whole
game.

**Comparison with our plan.** Our design/15 §3.4 opening spec ("uniformly
random legal moves, `R_open ∈ [8,10]`, reject via a balance gate") is
closer to Integral's plain **book mode** (uniform random moves) than its
organic mode. Integral's *organic* mode is strictly more work (two levels of
weighting) to buy *plausibility* of the resulting positions without an
external book at all — worth knowing as a fallback if we ever want opening
diversity without maintaining/curating a book file (see §9).

## 4. Adjudication — the exact thresholds

All constants live at `data_gen.cc:217-221`; the full decision table:

| condition | threshold | citation |
|---|---|---|
| mate-range score at any ply | immediate win/loss for the mating side | `data_gen.cc:264-267` |
| decisive-score streak | `\|score\| ≥ 2500 cp` for **5** consecutive plies from the same side | `kWinThreshold=2500, kWinPliesThreshold=5`, lines 217-218, 268-271, 277-280 |
| near-zero streak | `\|score\| ≤ 2 cp` for **8** consecutive plies, **and** `half_moves ≥ 200` (i.e. only after ply 200 of the *whole game*, not the search) | `kDrawThreshold=2, kDrawPliesThreshold=8`, lines 219-220, 272-274, 281-283 |
| repetition / insufficient material | immediate draw, checked **after** `make_move`, every ply | lines 291-294 — added later, `6c970780` "Add insufficient material check to datagen adjudication (#207)", 2025-06-18 |
| no legal moves | checkmate → win for mover-not-to-move; stalemate → draw | lines 260-262 |

**Vs. our design/15 §3.4**: our spec is *looser* on both axes — win at
`±2500` for **4** plies (Integral: 5), draw at `±8 cp` for **10** plies
**after ply 80** (Integral: `±2 cp` for 8 plies after ply **200**). Integral
is markedly more conservative about calling a draw early (100 fewer
half-moves' difference is not a rounding gap — it means Integral basically
never adjudicates a draw before deep middlegame/endgame, only lets truly
dead-flat long games end that way), and slightly more conservative on the
score band (±2 vs our ±8). Two live commits show this wasn't first-try:
`9e81ccf4` "Make data generation adjudication less aggressive" and
`edc56a0d` "Don't adjudicate when TB score is found in data generation" —
i.e. the dev shipped an aggressive adjudicator, watched it mislabel some
positions (plausibly near a just-discovered TB win/draw), and walked the
thresholds back. Concrete, if soft, evidence that our own thresholds should
be treated as a first guess to be tuned against real label-quality
evidence, not a value to defend past the point the data says otherwise.

## 5. Output formats — every ply recorded vs. capped-and-filtered

Two `OutputFormatter` implementations (`format/format.h:8-15` is the trait:
`set_position`, `push_move`, `write_outcome`) ship side by side:

**`BinPackFormatter`** (`format/binpack.h`, the one actually used at scale —
this is the format the README's "bullet" claim refers to): writes one fixed
header record per game (occupancy bitboard, one 4-bit nibble per occupied
square encoding `(colour, piece-or-unmoved-rook-flag)`, side to move + en
passant file, fifty-move clock, full-move number, and the game's final WDL
as `0/1/2`), followed by a `(move, score)` pair **for every single ply of
the game, unfiltered**, terminated by a null-move sentinel
(`format/binpack.h:67-129`). This is functionally the community "Marlin
format" (the struct is even named `MarlinChessBoard`,
`format/binpack.h:67`) — a `bullet`-native binpack layout, so **no quiet/
in-check/noisy filtering happens in the generator at all**; it's deferred
entirely to whatever reads the binpack later (bullet's own binpack loaders
support in-trainer filtering/rescoring — our design/15 §3.2.3 already notes
`SfBinpackLoader`/`MontyBinpackLoader`/`ViriBinpackLoader` support this).

**`FenFormatter`** (`format/fens.h`, the simpler/older path): filters as it
goes — skips any move that is a capture, en-passant, or promotion, or where
the position *before* the move was in check (`format/fens.h:20-27`,
mirroring exactly our design/15 §3.5 quiet-position filter) — then at game
end **randomly shuffles the accepted FENs and keeps only the first 50**
(`format/fens.h:33-41`), discarding the rest of that game's eligible
positions. This is a **hard per-game cap implemented as reservoir-style
shuffle-then-truncate**, conceptually identical to our design/15 §3.5(6)
`P_cap` (we default `P_cap = 10`).

**The load-bearing comparison.** Our own design/15 §10 (validation-run
results, 2026-07-11) measured that *our* `P_cap = 10` throws away **88.8%**
of an already-fully-labeled game's eligible quiet positions (§10.4 —
`rej_pcap` 792,080 of 892,151 searched-and-eligible positions) and flagged
raising `p_cap` as the single highest-leverage lever before the bulk run.
Integral's production path (binpack, no cap at all — effectively
"`p_cap = ∞`, filter later") is **direct evidence that a top-6 engine chose
the "record everything, filter downstream" side of exactly this tradeoff**
for its at-scale pipeline, and reserved the capped/filtered `FenFormatter`
for what looks like a simpler/earlier or debug-oriented path. This
strongly reinforces our own doc's pending recommendation (§9 Adapt list).

It's also a small, concrete instance of our D29 research-data-platform
philosophy ("store observations, not labels") arriving independently at
small scale: Integral doesn't decide at generation time which positions are
"good training data" — it stores the raw game (every ply, every score) and
lets the *trainer* decide, which is cheap to redo differently later without
regenerating any search. Worth citing to the team as validation that D29
isn't over-engineering — a solo dev backed into a lighter version of the
same idea under the same cost pressure (search is expensive, filtering is
free, so don't destroy information the search already paid for).

## 6. Syzygy-guided search — not datagen-specific machinery

There is no separate "TB-guided datagen mode." Syzygy probing
(`syzygy::ProbePosition`, gated by `syzygy::enabled`/`syzygy::probe_depth`,
`search.cc:587-638`) is simply **always active in the normal search** once
`SyzygyPath` is set (`uci.cc:29-34`), at any node — root or interior — with
`≤ 7` men, no castling rights either side, zero fifty-move clock, and
`depth ≥ probe_depth` (default `SyzygyProbeDepth = 1`, i.e. essentially
every qualifying node once past the first ply, `uci.cc:32-34`). A hit
returns an exact/bound score that is TT-stored and can raise `alpha` at PV
nodes (`search.cc:628-636`), same as it would during a rated game. Because
`DataGenStart` (`search.cc:1399-1417`) runs the *exact same*
`IterativeDeepening<SearchType::kBench>` path used for the `bench` command,
self-play games automatically inherit whatever tablebase the datagen host
has configured — "guiding the data generation search" is just "the engine
plays better and more truthfully near 5-piece endgames, so the labels near
there are TB-exact instead of search-approximate." No extra code exists to
special-case it for datagen.

**Comparison.** Syzygy/Fathom integration is explicitly **declined** in our
own roadmap (D31c: "vendored C + `cc`/`bindgen` + runtime libc + unsafe
FFI... affirms D15 (zero runtime deps); deferred past P13"), so we won't
get this "for free" the way Integral's single always-on search integration
does. If/when we revisit that decision, Integral's structure (TB probing is
a *generic search feature*, not a datagen add-on) is the right shape to
copy: it means the *same* code path that helps rated games also
automatically raises training-label quality near the tablebase boundary,
with zero datagen-specific plumbing.

## 7. The provenance story: Integral never used a foreign teacher

This is the single most important structural fact for comparing against
our own D28. Re-reading `README.md:24-26` precisely: *"The first iteration
of Integral's neural network was trained on data from version 4, which had
a powerful hand-crafted evaluation (HCE). Each iteration... since then has
been trained on a fresh dataset using the prior network."* There is **no
external engine** anywhere in this loop — not Stockfish, not any other
program. Integral's very first net was distilled from **its own** HCE via
self-play (`data_gen.cc`'s self-play loop, run with the pre-NNUE HCE build),
and every subsequent net is that same self-play loop run with the
*previous* NNUE build as both the move-chooser and the label source. The
`DataGenStart` search that produces training labels (§2.1, §6) is
*identical* in kind to the search that plays rated games — same pruning,
same TT, same Syzygy — just running under a node cap instead of a clock.

Compare our D28: we deliberately introduce a **foreign, stronger teacher**
(a locally-compiled Stockfish) for Tier A specifically to *de-risk the NNUE
machinery* faster than our own (currently PSQT-only) eval could, with Tier B
(clean self-play, our own labels) as the eventual convergence point. Both
strategies are legitimate and D28's own rationale (decouple machinery risk
from data-quality risk) is sound — but Integral is concrete, load-bearing
**proof that the foreign-teacher step is not required to reach top-6**: a
pure self-bootstrap loop (weak HCE → net-1 → net-2 → ... → net-7) got there
unaided. The git evidence for *how fast* that climb was is in §8.

## 8. Net iteration cadence, naming, and hosting (git archaeology)

### 8.1 Timeline reconstructed from commit dates

| date | event | rev |
|---|---|---|
| 2023-10-25 | initial commit | `8d6ed33b` |
| 2024-04-23 | OpenBench (OB) SPRT support added — **5 months before any NNUE work** | `5f45ebc6` "ob support stuff" |
| 2024-08-03 | first working datagen prototype | `601d120e` "Working data generation" |
| 2024-08-27 | first NNUE code (still pre-merge) | `cd3f7acb` "first nnue" |
| 2024-09-01 | `Data Generation (#51)` merged | `d4359007` |
| 2024-09-06 | `NNUE (#52)` merged — **5 days after datagen** | `99b7b22e` |
| 2024-09-08 | `Datagen Fixes (#54)` | `018f5df3` |
| 2024-09-18/19 | horizontal mirroring, phase-scaled NNUE | `796c4dca`, `5fa0c7e1` |
| 2024-10-20 | first LTC SPSA tune, 25k iters/100k games | `5348ae81` |
| 2024-10-23 | OB per-network result caching | `c822a25f` |
| 2025-02-13 | engine "v7.0.0" tag commit | `6eb962bd` |
| 2025-04-25/26 → 07-02 | `zekrom` v1 → v2 → v3 → v4 → v5 → v6 → v7, **six accepted net upgrades in ~10 weeks** | see §8.2 table |
| 2025-06-18 | insufficient-material adjudication fix | `6c970780` |
| 2025-07-02 → 2026-04 (latest examined) | ~20 named net *experiments*, main branch still on `zekrom-v7` | see §8.3 |

Datagen infra and NNUE landed as a single tightly-coupled push (≈5 weeks:
Aug 3 → Sep 6, 2024), roughly **10 months** into the project — the engine
spent its first year as pure HCE alpha-beta, exactly matching the "HCE
first, NNUE later" order every doc/decision of ours already assumes (D28,
design/11 Phase B→C). Also notable: **OpenBench predates NNUE by 5
months** — the SPRT harness was already load-bearing for plain search
tuning well before it had to carry net-swap testing too, so net swaps never
needed *new* testing infrastructure, only a new artifact to feed the
existing one.

### 8.2 Net naming — themed codename + linear version, not content-hash

The shipped-network build mechanism (`CMakeLists.txt:24-48`) is a CMake
**auto-download at configure time**: `NETWORK_NAME` is a plain string
constant, the `.nnue` file is fetched from
`github.com/aronpetko/integral-networks/releases/download/<NAME>/<NAME>.nnue`
(a *separate* GitHub repo used purely as a binary-artifact CDN via GitHub
Releases) if not already present locally, and the whole thing runs through
the offline `preprocess` tool (§8.4) before the main build embeds it via
`INCBIN`. Reconstructing every `NETWORK_NAME` edit in `CMakeLists.txt`'s
history (`git log -p -- CMakeLists.txt`) gives the full lineage:

```
a → zekrom → zekrom-sparser → zekrom → zekrom-v2 → zekrom-v3 → zekrom-v4
  → zekrom-v5 → zekrom-v6 → dual_act → zekrom-v6 (reverted) → zekrom-v7
  → 90wdl-sparse → 100wdl-sparse → 100wdl-redo-sparse → zekrom-v7 (reverted)
  → vine-256 → integral-256 → vine-256 (reverted) → vine → vine_small
  → vine_small_adjudicated → zekrom-v7 (reverted) → swish-swiglu
  → swish-swiglu-permuted → swish-swiglu-permuted1
```

Two naming registers coexist: a **themed main line** (`zekrom`, a legendary
Pokémon — v1 through v7 are the accepted, shipped progression, each commit
literally titled `"Switch to Zekrom vN network"`), and **descriptive
experiment names** for everything that branches off it (`dual_act` = a
dual-activation-function trial, `90wdl-sparse`/`100wdl-sparse` = WDL-blend-
weight sweeps at differing sparsity, `swish-swiglu*` = activation-function
swaps, `vine`/`vine-256`/`vine_small` = architecture-size variants). Every
non-`zekrom` name above except the final `swish-swiglu-permuted1` was
eventually **reverted back to `zekrom-v7`** on the branch this history was
taken from — i.e. most net experiments in this window did not beat the
incumbent. There is no content-hash naming scheme (unlike Stockfish's
`nn-<sha256prefix>.nnue`) — a human-readable, memorable name is enough at
this project's scale and makes the CMake diff/revert story trivially
readable.

### 8.3 The v7 plateau — progress is not monotonic

The `zekrom` line moved v1→v7 in about **ten weeks** (2025-04-25 to
2025-07-02, one accepted upgrade roughly every 12 days — see the exact
per-version dates recovered from `git show -s --format='%ci %s'` on each
`"Switch to Zekrom vN"` commit). Every `NETWORK_NAME` edit *after* v7 that
this history captures (Jul 2025 → Apr 2026, **~9 months**, spanning `Bench:`
commits as late as `3f15ba6b` (2026-04-02, the tip of `main` at the time of
this study — `1840658` in an earlier draft of this note was not a valid
object in the repo) reverts back to `zekrom-v7` — none of the
dual-activation, WDL-sweep, SwiGLU, or smaller-net experiments replaced it
as the shipped default in the window examined. Meanwhile the *search* side
kept shipping real gains in that same window (`Play quiet moves in QS...`,
`History Pruning before SEE Pruning`, `SF-style aspiration window
adjustments`, etc. — all dated Feb-Apr 2026). The lesson: **a single strong
net can carry an engine for the better part of a year while search-side
work keeps compounding on top of it** — net churn is not a prerequisite for
continued Elo growth, and a long stretch of failed net experiments is normal
and doesn't block progress elsewhere. This matches our own D32 prioritization
("NNUE, SIMD, concurrency are the priorities" over incremental heuristic
tuning) from the opposite direction — Integral shows the inverse also
holds: once you have *a* net, search-side iteration can dominate for a long
time before the next net upgrade matters again.

### 8.4 The offline net-preprocessing step

`preprocess/net_processing.cc` is a **separate small binary** (its own
`CMakeLists.txt` in `preprocess/`, invoked automatically as a build step,
`CMakeLists.txt:71-77`) that takes bullet's raw trained-network dump and
transforms it into the engine's fastest inference layout before `INCBIN`
embeds it: transposing the two-hidden-layer weight matrices from the
trainer's row-major `[layer_in][layer_out]` order into the transposed order
the SIMD dot-product code wants, and (when the target ISA supports it)
permuting weight blocks into the exact lane order the packed-integer SIMD
kernel consumes so no runtime shuffle is needed (`net_processing.cc:14-36,
53-69`, `#if BUILD_HAS_SIMD`). The insight worth lifting is the **separation
of concerns**: `bullet` emits whatever layout is natural for its own
training math; the engine's runtime format is a *distinct* concern solved by
one small offline tool, not by asking the trainer to match the engine or by
paying a transpose cost on every load. Our design/11 §3.4.7's `netpack`
helper is already scoped as exactly this kind of tool ("wrap bullet's raw
quantized output in the CENN container, compute checksum") — Integral is
independent confirmation this is the right shape, and a reminder to make
`netpack` also own any SIMD-lane permutation our own inference backend
needs (design/11 §3.4.6), not bake it into the hot load path.

### 8.5 OpenBench + the ~946-branch experimentation culture (pointer only)

Beyond the scope of this datagen-focused note, but too relevant to omit a
pointer: `git branch -r | wc -l` on this clone returns **946** remote
branches (against 669 commits on `main`), the overwhelming majority
single-purpose experiment branches (`aggro_cut_node_lmr`, `asp_window_tweak`,
`7buckets`, `1536l1`, `dual_activation`, `eval_policy_new_net_ahhahaha`,
...) — the vast majority never merged. Combined with the explicit OpenBench
integration commits (`5f45ebc6`, `a36224d6`/`c994293e`/`0851665d` "Fixed OB
bench command crashing [again]", `c822a25f` "Allow OB network caching"),
this is strong circumstantial evidence that **every** net/architecture
experiment in §8.2-8.3, not just search heuristics, goes through the same
branch → OpenBench SPRT → merge-or-abandon pipeline. This is squarely a
"how did a solo dev do it" finding (process/tooling, not datagen mechanics)
and is flagged here for whichever note in this study owns that question —
it belongs next to the OpenBench/branch-discipline analysis, not duplicated
in depth here.

## 9. Comparison table

| aspect | Stockfish | Reckless | Integral | Us (design/15 + D28/D29) |
|---|---|---|---|---|
| teacher for first net | community fishtest + nnue-pytorch, no single "teacher" engine | not in vendored source (no datagen tool available to study) | **its own HCE**, pure self-bootstrap | a **foreign** teacher (local SF) for Tier A, own labels for Tier B |
| move-selection budget | community-scale, effectively unlimited compute | unknown (not vendored) | fixed **nodes** (soft+hard caps) | fixed **nodes** (~5000/move) — independently converges with Integral |
| opening diversity | large curated book ecosystem | unknown | book **or** organic (piece-weighted + center-weighted) random walk | uniform random walk + optional balance-gate book |
| record granularity | binpack, every ply, filtered by nnue-pytorch data loader | n/a | **binpack, every ply, unfiltered** (bullet-native); a separate capped/filtered FEN path also exists | flat 32-byte `bulletformat::ChessBoard`, **capped at P_cap=10/game**, filtered in the generator |
| TB use in datagen | yes (community convention) | Fathom FFI vendored, declined by us (D31c) | **always-on generic search feature** (≤7 men), not datagen-specific | not planned before P13 (deferred, D31c) |
| net naming | content hash `nn-<sha>.nnue` | unknown | **themed codename + linear version** (`zekrom-v7`), descriptive names for side experiments | not yet designed (design/11 §3.4.5 owns the CENN container/versioning) |
| provenance recording | fishtest's own DB | unknown | **none committed** — README prose is the only record | `meta.json` per corpus (design/15 §3.6) — stricter than Integral |
| SPRT gate for a new net | fishtest | unknown | OpenBench, same pipeline as search tuning | our own SPRT harness (D25), analogous gate (design/15 §4 step 7) |

## 10. Adapt to us

1. **Raise our `P_cap` toward "record everything, filter downstream"** —
   design/15 §3.5(6)/§10.4. Integral's at-scale path records every ply
   unfiltered and defers curation to the trainer; our own §10 validation run
   already flagged `P_cap=10` as discarding 88.8% of paid-for labels. This
   note is corroborating evidence from a shipped top-6 engine, not just our
   own measurement — raise `P_cap` materially (or drop the per-game cap
   entirely in favor of a binpack-style record) before the real bootstrap
   bulk run. Impact: near-linear throughput win, i.e. the difference between
   a multi-week and a multi-day bulk run at our target corpus size.
2. **Plan for a pure self-bootstrap path (no foreign teacher) as a real
   fallback for D28**, not just the SF-distillation Tier A. Integral is
   concrete proof the HCE→NNUE→retrain loop reaches top-6 unaided. If the
   local-SF-teacher setup (build, licensing bookkeeping, D27/D28 scaffold
   discipline) ever becomes the bottleneck, our own PSQT/HCE self-play
   (Tier B machinery, run early) is a legitimate Tier-A substitute, not a
   downgrade. Impact: a risk-reducing alternate path, low cost to keep open
   (design/15 §3.1 already frames Tier B as "may be a full self-play zero
   restart" — this just says Tier A could be that too).
3. **Guard against baking search-only score adjustments into training
   labels** (design/11 §3.6's rule-50 damping and material-scaled optimism
   terms). Integral's `#ifndef DATAGEN` split (`search.cc:220-245`) shows
   this is a real, previously-hit footgun, not a hypothetical one. When
   `bin/datagen.rs` (design/11 §3.4.7) is built, add an explicit `datagen`
   cargo feature/flag that disables the rule-50 and optimism terms in the
   *label-producing* eval call specifically, even if the *move-choosing*
   search still uses them. Impact: correctness of the training signal itself
   — cheap to add now, expensive to discover after a net trains on
   self-inflicted noise.
4. **Adopt a themed-codename + linear-version net naming scheme**
   (`design/11 §3.4.5` CENN container is the natural place to pin this),
   e.g. `<codename>-v<N>` for the accepted main line and a short descriptive
   slug for every side experiment (`wdl-sweep-90`, `swiglu-a`, ...), hosted
   as build-time-downloaded release artifacts (a small sibling repo or
   release assets on our own remote) rather than committed binaries — mirrors
   design/11 §3.4.7 step 5 ("nets are build artifacts... embed at build
   time") and gives free, readable git-diffable history of which net is
   live, exactly like `CMakeLists.txt`'s one-line `NETWORK_NAME` diffs.
   Impact: organizational only, but cheap and directly reusable once P13
   starts producing more than one net.
5. **Expect (and budget for) a long stretch of failed net experiments after
   the first strong net** — §8.3's ~9-month `zekrom-v7` plateau against
   continuous search-side gains is a realistic prior for our own post-P13
   trajectory. Don't treat a run of SPRT-losing net retrains as a signal
   the NNUE bet (D34) isn't working; treat search/SMP/eval-integration work
   as the expected majority of Elo gain between net upgrades. Impact:
   expectation-setting for roadmap planning, not a code change.
6. **If we ever revisit Syzygy (past P13, D31c)**, copy Integral's shape —
   a single generic in-search probe gated on piece count/depth/rule50/
   castling that both rated play *and* datagen inherit automatically
   (`search.cc:587-638`) — rather than a datagen-only special case. Impact:
   deferred/low-priority (D31c already declines TB for now), but worth
   recording so the eventual design doesn't re-derive this from scratch.
7. **Node-budget self-play labeling at ~5000 nodes/move is validated
   independently**, not just an SF-teacher convention: design/11 §3.4.7 and
   design/15 §3.3.2 both already specify "fixed nodes, not depth/time," and
   Integral's own `TimeConfig{ nodes, soft_nodes }` self-play loop (no depth
   or clock in the hot loop at all) reaches top-6 on the same premise.
   Impact: confidence, not a design change — keep the node-budget contract
   as-is for `bin/datagen.rs`.

## 11. Citations (re-find index)

- `README.md:11-37` — public architecture/datagen/training summary.
- `src/data_gen/data_gen.h` (30 lines) — `Config` struct, defaults.
- `src/data_gen/data_gen.cc` — `kCenterScore` 18-27; `SelectPreferredMove`
  30-50; `FindStartingPosition` 52-134; adjudication constants 217-221;
  `GameLoop` 211-319; `Generate` 325-441.
- `src/data_gen/format/format.h:8-15` — `OutputFormatter` trait shape.
- `src/data_gen/format/fens.h` — quiet filter 20-27, 50-cap 33-41.
- `src/data_gen/format/binpack.h` — `MarlinChessBoard` 67-76,
  `BinPackFormatter` 84-177, `ConvertBoardState` 132-171.
- `src/engine/search/search.cc` — datagen-only eval adjustments 220-245;
  Syzygy in-search probe 587-638; `DataGenStart` 1399-1417.
- `src/engine/search/syzygy/syzygy.h` — `enabled`/`probe_depth` atomics.
- `src/engine/uci/uci.cc` — options `Initialize` 18-36 (`SyzygyPath`,
  `SyzygyProbeDepth`); `datagen` UCI command 124-146.
- `preprocess/net_processing.cc` (113 lines) — offline net layout transform.
- `CMakeLists.txt:23-48` — `NETWORK_NAME`/auto-download/versioning.
- `src/utils/types.h:24-33` — `PieceType` enum order (for §3's weight table).
- Revs cited: `8d6ed33b` (initial commit), `5f45ebc6` (OB support),
  `601d120e` (working datagen prototype), `cd3f7acb` (first nnue),
  `d4359007` (Data Generation #51), `99b7b22e` (NNUE #52), `018f5df3`
  (Datagen Fixes #54), `796c4dca`/`5fa0c7e1` (mirroring/phase scaling),
  `5348ae81` (first LTC SPSA), `c822a25f` (OB network caching), `6eb962bd`
  (v7.0.0 tag), `4816a902` (book support), `9e81ccf4`/`edc56a0d` (adjudication
  walk-back), `3c09d3fc` (cuckoo repetition), `6c970780` (#207 insufficient
  material), `45ca6386` (#174 disable eval scaling during datagen),
  `ffb2c1c8` (credit Bullet), and the full `NETWORK_NAME` lineage in
  `git log -p --all -- CMakeLists.txt` (§8.2's commit chain).
