# Generating ARCHON's own NNUE training data — design and operating point

**Written 2026-08-02.** Tools live in `tools/ops/data/` (indexed in
`tools/ops/README.md`). Everything below is measured on this box unless it is
explicitly flagged as unestablished.

**Nothing here has been run at production scale.** The owner's constraint was
tooling plus a measured operating point, ready to launch on their word. The
pilots below are ~450 games total.

---

## 0. Why this exists

Over ~3,000 games we measured that **our current labels point the wrong way**:
blending 40 % of a Stockfish net into the training target gave
**+30.42 ± 11.02 Elo vs canonical** while the label-heldout loss rose
monotonically (`d48y-wdl-fix-plan.md`, 2026-08-01 18:00). Moving *away* from our
labels gains Elo. Label quality is therefore the binding constraint, and that is
why data is being regenerated rather than the trainer retuned.

The one-line design consequence: **this pipeline's job is not to produce many
positions, it is to produce labels that do not fight the outcome.**

---

## 1. THE RECOMMENDED OPERATING POINT

*(section filled in below once every parameter's measurement is in — see §2 ff.)*

---

## 2. The pipeline

Four stages, each resumable, each leaving a stats file. `run_pipeline.sh` chains
them.

| stage | tool | what it decides |
|---|---|---|
| 1 play | `gen_games.sh` | mode, node budget, book, adjudication |
| 2 extract | `pgn_positions.py` | which positions, how many per game |
| 3 label | `label_positions.py` | oracle, node count, **the D46 filter** |
| 4 export | `export_corpus.sh` | trainer bytes + the contract's self-checks |

### Where the D46 filter runs, and why it can only run there

The contract (`data-recipe-contract.md` §1) requires the five-criterion quiet
filter at "the layer that still has raw scores + best moves". In this pipeline
that layer is **stage 3 and nowhere else** — the exported text has lost the move,
so a downstream filter is structurally partial (the `canon_filter` lesson).

Stage 2 applies the two criteria that need neither a score nor a move (ply ≥ 16,
not in check) purely so we do not buy an oracle search for a position that is
already disqualified. It records those counts so the **full** five-criterion rate
stays reconstructable; otherwise stage 3's rate looks deceptively low and its
"did the filter run?" self-check loses its meaning.

Two details that are easy to get wrong and are implemented deliberately:

* **Criteria 4 and 5 are not redundant.** An en-passant capture lands on an
  *empty* square, so "destination empty" alone lets it through; the move-kind
  test is what catches it. `label_positions.py` follows the normative Rust
  (`d40_features.rs::sfbinpack_filter`) rather than `relabel_depth.py`, which
  reaches the same place by a different route.
* **Filter first, clamp last.** `|raw| ≤ 10000` is tested on the raw score;
  only survivors are clamped to ±2499. Stage 4 re-asserts the clamp and *dies*
  rather than clamping again — a second clamp is exactly the bug that put
  86,320 mate labels into an aux corpus.

### The WDL column is real by construction

This is the pipeline's main structural advantage over the shard path.
`build_canon_tight.sh` has to inner-join a result column onto positions labelled
from a different sample, and ships `JOIN_MIN=33` because its measured overlap was
**35.6 %** — two thirds of its labelled positions had no result and were
discarded. Here the result rides *with* the position from the game that produced
it, so the join hit-rate is **100 % by construction**; `export_corpus.sh` asserts
it and treats a miss as an upstream bug, not an acceptable loss.

---

## 3. NULL FIRST — what the instruments can and cannot resolve

Run before any arm was read.

### 3.1 Instrument null: are labels reproducible at all?

`label_positions.py --null` labels the input twice and diffs.

```
.venv/bin/python tools/ops/data/label_positions.py <200 pilot positions> \
    --out lab200.tsv --oracle engines/bin/stockfish18 --nodes 25000 \
    --workers 4 --null
→ NULL CONTROL: 200 positions relabelled identically=200 differing=0 (0.000%)
```

**200/200 bit-identical.** Labels are a pure function of
(engine, net, nodes, position); `ucinewgame` per position is what buys it. So no
node-count difference below can be run-to-run noise. This was worth checking
first: without it every tier comparison would have an unknown noise floor.

### 3.2 Sampling null: how big a difference is even resolvable?

Every statistic in the node study is also computed on two disjoint halves of the
position set **at the same node count**. The half-vs-half spread is the sampling
noise floor and is printed beside every tier as `nullspread`. **A tier-to-tier
movement smaller than the local nullspread is reported as not resolvable**, and
in §4 several of them are.

### 3.3 Oracle-comparison null

`study_oracles.py` automatically adds a `__NULLCOPY` arm: the first oracle
labelled a second time and compared with itself. It must read 100.00 %
best-move agreement and 0 cp. Any cross-oracle difference smaller than that is
inside the instrument.

### 3.4 A null that is NOT clean, stated plainly

**Throughput numbers are contaminated by box load.** `promogate` (a 1500-game
match) ran throughout, and the pilots and studies overlapped each other; load
average reached 17.98 on 16 cores. Specifically, `pilot-sf16` ran while the node
study was saturating the box and reported 1475 games/h against 2857 games/h for
`pilot-sf18`, which ran on a quieter box. **That is not evidence that SF16 is
slower than SF18** — it is the load difference, and it would be a wrong
conclusion drawn from a dirty instrument. Every wall-clock number in §7 is a
**lower bound**, and the mode-to-mode comparison of games/h is void.
Re-measuring the throughput table needs a quiet box.

---

## 4. How many nodes should the oracle get?

*(pending — §4 table)*

---

## 5. Which oracle?

*(pending)*

---

## 6. Diversity and WDL meaningfulness

### 6.1 Sampling policy: fixed count per game, not every ply

Positions are sampled evenly across each game's eligible plies. The receipt for
caring is `tools/forensics/build_oracle.py`, whose predecessor sampled without
this discipline, came out 100 % ≥21-piece positions, and was blind to 82 % of
our damage.

Games yield **98.6 quiet plies each** (150 self-play games, 14,795 quiet plies),
of which **99.8 % are distinct piece placements** — so the cost of taking more
per game is *correlation*, not duplication.

Taking every ply is not free, and the bias is measurable:

| sampling | n | W | D | L |
|---|---|---|---|---|
| every quiet ply | 14,795 | 32.3 % | **43.0 %** | 24.7 % |
| even 14/game | 2,099 | 33.3 % | **40.6 %** | 26.0 % |
| **per-GAME (unbiased reference)** | 150 | 33.3 % | **40.7 %** | 26.0 % |

**A fixed count per game reproduces the true per-game result distribution
exactly; taking every ply inflates draws by +2.4 pp.** The mechanism is direct:
drawn games run **104.3** quiet plies against **93.7** for losses and 95.5 for
wins, so per-ply sampling over-weights them. Fixed-count sampling is
self-normalising and costs nothing to keep.

```
.venv/bin/python tools/ops/data/pgn_positions.py \
    data/nnue/selfplay/pilot-self/games.pgn --out all.tsv --per-game 0
```

### 6.2 Per-mode distributions

150 games per mode, 25k nodes both sides, UHO book, 14 positions/game:

| mode | n | W | D | L | mean pieces | ≤12 | 13–20 | ≥21 |
|---|---|---|---|---|---|---|---|---|
| self | 2,099 | 33.3 % | 40.6 % | 26.0 % | 19.0 | 25.1 % | 29.0 % | 45.9 % |
| sf18 | 2,056 | 31.9 % | 48.7 % | 19.4 % | 17.5 | 33.6 % | 26.4 % | 40.0 % |
| sf16 | 2,061 | 29.6 % | 37.6 % | 32.8 % | 18.3 | 28.5 % | 29.2 % | 42.3 % |

All three are non-degenerate (contract §4 requires no value above 70 %). Phase
coverage is broad in every mode; vs-SF18 skews most toward endgames (33.6 %),
which is the mode most likely to supply conversion technique.

### 6.3 Both Stockfish anchors are genuinely contested

WDL is only meaningful if the games were in doubt. At 25k nodes both sides:

| mode | ARCHON score | Elo | draw rate |
|---|---|---|---|
| vs Stockfish 18 | 41.67 % | **−58.45 ± 34.65** | 32.0 % |
| vs Stockfish 16 | 61.67 % | **+82.59 ± 37.57** | 41.3 % |

We sit *between* SF16 and SF18 at equal nodes, so neither matchup is a blowout
and neither produces a corpus of foregone conclusions. (150 games each: these
place us on a ladder, they are not verdicts — the bar at n=150 is ±31 Elo.)

### 6.4 Adjudication: a throughput lever that costs endgames

| mode | played out | resign-adjudicated | draw-adjudicated |
|---|---|---|---|
| self | 25 % | **57 %** | 18 % |
| sf18 | 42 % | **42 %** | 16 % |
| sf16 | 29 % | **57 %** | 13 % |

Two consequences, and they pull in opposite directions:

* **Draw adjudication is cheap and honest.** A position dead level for 8
  consecutive moves after move 40 is a draw; adjudicating it costs nothing in
  label quality and saves real time.
* **Resign adjudication truncates the conversion phase.** 57 % of self-play
  games ended when both engines agreed on a ≥1600-ARCHON-cp (~773 SF-cp)
  advantage for 4 moves. Those games never played out the technique that
  converts a won position, so the corpus is systematically short of exactly the
  endgames a value net needs to price. It also makes eval-outcome agreement
  **partly circular at high |cp|**: the result was defined by an eval. The
  circularity does not reach the decision threshold used in §4 (150 SF-cp is far
  below the 773 SF-cp resign bar), and the `term_normal` slice is reported
  separately so the honest number is visible.

`ADJ=0` disables both. It is exposed precisely because this is a real trade the
owner may want to make differently.

### 6.5 The adjudication frame trap (fixed, and worth knowing)

fastchess compares its resign/draw thresholds against **each engine's printed
score**, and ARCHON prints ~2.07× Stockfish's printed cp. A one-sided
`score=1600` would therefore fire for ARCHON at ~773 SF-cp and for Stockfish at
1600 SF-cp — a **colour-correlated truncation bias landing directly in the WDL
column**. `gen_games.sh` uses `twosided=true`, which requires both engines to
agree and removes the asymmetry.

### 6.6 Self-play must not use `-repeat`

Two identical deterministic engines replay the same game with colours reversed.
Measured: 8 games under `-games 2 -rounds 4 -repeat` produced **4 distinct
(opening, move-list) signatures — every pair bit-identical** (127/140/87/143
plies). In `self` mode `-repeat` is an exact 2× CPU waste contributing zero new
positions; `gen_games.sh` uses it only in the vs-Stockfish modes, where the two
engines differ and colour balance is real.

### 6.7 Book

`engines/books/UHO_Lichess_4852_v1.epd`, **2,632,036 positions** at ~move 4.
Deliberately unbalanced openings, which is what makes games decisive and the WDL
column informative. Diversity is not a constraint at our scale: a 20,000-game run
consumes 0.8 % of the book. `gen_games.sh` derives a reproducible non-overlapping
book slice from the run tag (`BOOKSTART`, `SRAND`), so successive runs never mine
the same openings twice.

---

## 7. Throughput

*(pending — §7 table)*

---

## 8. What I could not establish

*(pending)*
