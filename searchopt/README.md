# Greedy Search-Tree Optimization

Everything needed for this effort lives in this folder — guide, ledger, tools, positions.

**Scope rule.** Reading and changing **ARCHON's own source** (`src/`) is expected and
encouraged: the whole point is to find the line of code that discards the right move, and
you cannot do that from outside the engine. Read `src/search/mod.rs`, add a tunable, add
instrumentation — that is the work.

What you should *not* do is wander the wider **docs/research** tree looking for context.
It is a large pile of half-refuted narrative from earlier sessions, and going shopping in
it is how the 2026-07-28 session lost a day. `LEDGER.md` here holds everything from it
that survived.

---

## 0. The one-sentence problem

**The search discards the right move too early.** The move is present, it is generated,
and it is valued correctly once it is actually searched — it just is not given the search
it needs to show its point, so a wrong move keeps the root.

Everything below exists to find *which node and which decision* does the discarding, on
one position at a time.

---

## 1. AXIOMS — not hypotheses. Do not re-test these.

**A1. The eval is at par with Stockfish or better. Eval is never the answer.**

This is a standing ruling from the project owner, and it is now also measured: our net's
correlation with SF's deep scores is **+0.803 vs SF's own static eval at +0.722** on 2912
positions from lost games. On 2026-07-28 I accused eval three times anyway. Every time it
was at a moment when the search analysis had become hard and I wanted an exit, and every
time it was wrong — the last time in direct contradiction of a static-eval comparison I
had run myself ten minutes earlier.

*If you find yourself about to write "the net misvalues X", stop. You are avoiding work.*

**A2. Our search constants are faithful to Stockfish `eca43a97`.** Verified line by line:
reduction table `int(2834/128·log(i))`, the reduction formula
`scale − delta·617/rootDelta + !improving·scale·194/512 + 1027`, statScore
`2·main + contHist[0] + contHist[1]`, the coupling `r -= statScore·445/4096`, history clamp
`7183`, bonus `min(134d−79,1572)+382·ttMove+statScore/30`, malus `min(1005d−205,2218)`,
and the whole quiet ordering key. **Blunders are not mis-transcribed constants.**

**A3. Every global pruning/reduction constant is at a local optimum at equal nodes.**
Measured across `RED_TABLE_NUM`, `RED_BASE`, `RED_CUTNODE`, `RED_TABLE_OFFSET`, RFP,
razoring, capture/parent/qsearch futility, and the whole quiet-ordering weight family.
Global level changes are dead. Only *new conditioning axes* can move anything.

---

## 2. THE METHOD

One position. One question. No aggregate statistics anywhere in the turn.

### Step 1 — pick a position that indicts us
From `positions/ours.json`: SF plays a sound move at our in-game budget (110k nodes) and
we play the losing one. If SF also blunders there, it is not our defect — skip it.

### Step 2 — walk the tree to the discarding node
`tools/searchdbg.py` on the root, then follow the divergence down one ply at a time. At
each node ask only: **does the right move get searched, and with what?** Record
`move_index`, `final_reduction`, `scout_depth`, `lmr_research`, alpha/beta.

Descend while the engines disagree. Stop at the first node where the right move gets a
full-depth search and is *still* rejected — the discarding happens at the ply above.

### Step 3 — state the decision in one sentence
"At `<FEN>`, move `<X>` is ordered `<n>`th, reduced by `<r>`, scouted at depth `<d>` of
`<D>`, returns `<v>` against alpha `<a>`, and is not re-searched."

If you cannot write that sentence, you have not found it yet. **Go back to step 2. Do not
start measuring things.**

### Step 4 — BEFORE building an arm, break your own metric
Write down the metric you would judge the arm on, then run the **opposite-direction arm
first**. If reducing *more* helps as much as reducing *less*, your metric is measuring
noise and the whole experiment is void.

This is not optional and it is not a formality. On 2026-07-28 five arms cleared a
seven-sample null band on "nodes to stop playing the blunder" — and then
`RED_GIVES_CHECK = −512` (reduce checks **more**) cleared it too. The metric rewarded any
perturbation that shook the search off the losing move. A whole day's headline result
evaporated on one control that should have been run first.

### Step 5 — only proof-grade outcomes count
Admissible: **a forced mate found**, or a value independently confirmed correct by SF at
high depth. Inadmissible: "plays a different move", "score collapsed", "AUC improved",
"collapse depth moved". All of those are satisfiable by noise.

### Step 6 — games arbitrate
Equal binaries differing only by `setoption`. 1200 games at 10+0.1. **Never read a
partial match**: `rfp-threat0` read −15 Elo at n=600 and finished at −3.5 ± 14 at n=1185.

---

## 3. TRAPS — every one of these cost hours

| trap | the tell | the rule |
|---|---|---|
| **stdin EOF is a STOP** | every depth returns the same bestmove; trace file empty | drive through `uciprobe`'s persistent stdin, never `printf \| engine` |
| **fail-soft zero-window returns are BOUNDS** | a suspiciously tiny margin ("fails low by 2 cp") | re-measure bound-free with `searchmoves`; the real margin was 402 |
| **probe ceilings look like blindness** | "never finds it" where your loop stops | always extend the ladder past the claim before saying "never" |
| **fixed depth is tautological for reduction knobs** | any "reduce less" arm wins | equal **nodes**, always |
| **field semantics** | `tt_move` means "*this row's move is* the TT move", not "entry has a move" | read the struct definition before using a field |
| **self-selected PV leaves** | comparing two engines' static eval at their own PV ends | circular; deep-search both fixed positions instead |
| **`--features tune` missing** | every arm returns byte-identical results | params are inlined constants without it |
| **default-feature rebuild** | tracer suddenly writes no rows | it overwrites `target/release/chessengine`; keep `engines/bin/archon-trace` |
| **arm contamination** | a repeated arm gives two different answers | restore every touched param to its engine-reported default between arms |

---

## 4. WHAT THE CORPUS IS

44 decidable losing moves, split at the in-game budget (110k nodes), null-bounded:

| share | class | ours? |
|---|---|---|
| 14% | SF blunders too | **no** — position hardness |
| ~41% | near-ties for *both* engines (separation 23 vs 28 cp) | **no** — anyone errs |
| ~14% | warm history/aspiration flips it (warm TT *helps*) | yes, narrow |
| **32%** | **SF avoids at 110k, we don't** | **yes — work here** |

**Character of the 32%: we are confidently wrong, not uncertain.** Top1−top2 separation
**55 cp for us vs 10.5 cp for SF**, with SF near-tied in 64% of them. We commit conviction
to a losing move; SF sees enough danger to decline to commit.

⇒ **Sharpening the eval would make this worse.** The lever is making refuting lines
cheaper to resolve so the danger reaches the root and flattens a confidence that should
never have formed.

---

## 4b. BUILDING A FRESH CORPUS (do this after any net change)

The corpus is net-specific: positions mined from a D47V engine say nothing about
where D48X discards moves. Full loop, all inside this folder:

```bash
# 1. play the canonical engine against the oracle (~700 games, 10+0.1, UHO book)
engines/canonical/archon  vs  engines/bin/stockfish18       # see results/ for the cmd

# 2. oracle-score EVERY one of our moves in every LOST game
uv run --with chess python searchopt/tools/mine_losses.py \
    --pgn engines/runs/<run>/games.pgn --out searchopt/positions/losses.json

# 3. split by whose defect it is
python3 searchopt/tools/classify_losses.py --losses searchopt/positions/losses.json
```

`mine_losses.py` scores every move with SF18 rather than triggering on our own
eval collapsing: the previous detector did that and had **~9% recall**, because a
refutation landing ten plies later is invisible to it. It ranks by **win
probability**, not centipawns — cp does not sum into outcomes, and one large
error at a decision point converts a result while many small ones are absorbed.

`classify_losses.py` produces `ours.json` / `both.json` / `neither.json`.
**Work `ours.json`.** `both.json` is position hardness (SF blunders there too at
our budget) and `neither.json` is context or instability; neither is a search
defect you can fix.

## 5. FILES

```
searchopt/
  README.md            this guide
  LEDGER.md            what is established / refuted, with the number that decides it
  tools/uciprobe.py    persistent-stdin UCI harness + frame conversion
  tools/searchdbg.py   the search debugger (needs archon-trace)
  tools/build.sh       build the instrumented binary correctly
  tools/mine_losses.py     oracle-score every move in every lost game
  tools/classify_losses.py split into OURS / BOTH / NEITHER
  positions/ours.json      the actionable set  <- work here
  positions/*.json         both/neither/losses/classified, plus the D47V-era sets
  results/             one file per position walked, named by FEN slug
```

Binary: `engines/bin/archon-trace` — build with `tools/build.sh`.
Net: `data/nnue/nets/screens/d47v-a4b2-cal400.cenn`. Oracle: `engines/bin/stockfish18`.

---

## 6. RETIRED — do not resurrect

- `shapeopt.py` — its metrics (AUC, collapse depth, budget-to-fix) are all
  noise-satisfiable. Deleted, not moved.
- Any aggregate screen over the corpus during *diagnosis*. Aggregates are for
  characterising the corpus once, not for finding or judging mechanisms.
- `RED_GIVES_CHECK` as a validated arm — the parameter still exists (default 0) but its
  evidence was destroyed by the opposite-direction control (§2 step 4).
