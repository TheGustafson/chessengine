# Search optimization — method

Everything for this effort lives in this folder: guide, ledger, tools, positions.

**Scope.** Reading and changing ARCHON's own source (`src/`) is the work. What you should
NOT do is wander `docs/research/` for context — it is half-refuted narrative, and shopping
in it is how 2026-07-28 lost a day. `LEDGER.md` holds everything that survived.

**The net is out of scope.** NNUE training, architecture, width, retraining — human calls
made outside this folder.

---

## 0. The problem — RE-FOUNDED 2026-07-28

This folder was built on *"the search discards the right move too early."* That premise
was tested to destruction and **does not describe how we lose**. One actionable blunder
position in 404 games; the real failure is that we hold +1.0 and bleed it ~5 cp/move for
forty moves with no identifiable error anywhere. See `START-HERE.md` §1.

**Work the degradation, not the blunder.**

---

## 1. AXIOMS — do not re-test these

**A1. Eval is at par with Stockfish or better. Eval is never the answer.**

This is a standing ruling, and on 2026-07-28 it was independently re-measured *because an
agent kept relitigating it*: static eval vs a deep oracle over 630 positions spanning all
404 corpus games gives **MAE 97.1 (ours) vs 97.2 (SF18)**, correlation **0.899 vs 0.905**,
and **our p90 error is LOWER** (149.5 vs 176) — a fatter middle, a thinner tail. The
owner's training-time benches, on far more data, put D48X ahead of SF18's net.

### The eval trap, in full — because knowing the axiom is demonstrably not enough

On 2026-07-28 an agent was given this axiom in its brief, read it here and in
`START-HERE.md`, and was made by the owner to restate the commitment **in its own words
five separate times** before starting. It then blamed eval **twice**, hours apart, in two
different disguises, and was wrong both times:

1. **"The bottleneck is eval cost."** Measured a 1.92× NPS gap, found 75% of node time in
   evaluation, concluded the deficit was eval expense and therefore out of scope —
   *committed it to git*. Equal-node matches then showed cost explains **at most half**.
2. **"Therefore eval," by elimination.** After four hypotheses came back null, it declared
   the residual must be eval because nothing else was left. The owner pointed out that
   elimination is not measurement; the agent agreed and repeated the move two messages
   later.

The direct measurement — 630 positions, both engines' static eval against a deep oracle,
**fifteen minutes of compute** — came back **at par or better for us**, exactly as the
axiom always said. It could have been run in hour one instead of hour six.

Three rules fall out, and they are the actual content of this axiom:

* **The pull to eval arrives at one specific moment: when the search analysis runs out of
  road.** It does not feel like quitting. It feels like a rigorous conclusion, arrived at
  reluctantly, supported by numbers.
* **A conclusion that ends "and therefore it is not my problem" requires MORE evidence
  than one that doesn't.** Convenience is the tell. Both failures above had that shape and
  neither was challenged by the person who wrote it.
* **Rigour around a wrong conclusion makes it worse.** Both came with NPS decompositions,
  bootstrap CIs and pre-registered thresholds, which made them harder to argue with. The
  owner had to push back three separate times and was right on every one.

If you are about to write *"the net misvalues X"* or *"this is an eval problem"*: run
`staticeval.py` against a deep oracle first. Fifteen minutes. Then you will know instead
of inferring.

**A2. Our search constants are faithful to `eca43a97`.** Re-verified this session on the
reduction family (`r -= moveCount * 62` matches exactly; SF later moved it to 73, which is
a version difference, not drift). Blunders are not mis-transcribed constants.

**A3. Every global pruning/reduction constant is at a local optimum at equal nodes.**
Only *new conditioning axes* can move anything. Note SPSA has already swept more than the
README used to list — check `tools/spsa/out/*/coordinator.log` for a parameter's drift
before proposing it. `RED_MOVECOUNT` drifts ~0 over 3,280 games while its neighbours move
100+, so it is not an opportunity.

---

## 2. THE METHOD, AS REVISED BY WHAT FAILED

The old method was: pick one loss, walk it to the discarding node, arm it, game-test.
**Steps 1–4 of that no longer apply** to the degradation failure — there is no discarding
node. What survives:

### Characterise from the corpus, for free
`lossshape.py` reads both engines' per-move evals out of the match PGN annotations. No
engine time, all 404 games, **including the 204 draws the miner never scored — which is
where the Elo actually is.** Use it before spending a single node.

### Prefer equal-NODES game matches over any proxy
Fixed-node matches (`-each nodes=N`) neutralise the speed deficit, are contention-immune
(so several can share the box), and — unlike every per-position proxy tried — actually
resolve a 5 cp/move effect. A 400-game rung gives ±30 Elo.

### Null the metric, not just the arm, and do it FIRST
Non-negotiable. This session killed **five** instruments on their own nulls:

| instrument | how it died |
|---|---|
| free-search argmax | flips **2/6** under inert ±1 |
| `searchmoves` head-to-head | flips **2/6**; and 2 root moves vs 16 hands each ~8× the budget anyway |
| root node allocation | ratio swings **8.1× → 0.1×** across the same inert panel |
| annotation-based eval drop | claimed 141/174 lost games contain a blunder; **156/174 of those drops start from an already-lost position** |
| pruning-bias cp-loss | the inert null arm captured **57%** of the candidate arm's effect |

### Test the MECHANISM before you test the outcome
The strongest discipline found this session. Build an arm from a measured mechanism, then
**verify the arm acts on that mechanism** before spending games. A root-reduction arm was
killed this way in minutes: dropping a starved move's reduction from 1020 to **3** moved
its node allocation 440 → **414**. The reduction was never what starved it — the move was
cheap because it failed low, and low cost was the *symptom*, not the cause.

### Only proof-grade outcomes count
A forced mate, or a value an independent engine confirms at high depth. "Plays a different
move", "score collapsed", "AUC improved" are all satisfiable by noise.

---

## 3. TRAPS — every one cost hours, several cost them twice

| trap | tell | rule |
|---|---|---|
| **eval perspective** | absurd correlation (0.03) | ARCHON's `eval` is **side-to-move POV and the label changes**; SF's is always white POV. Use `staticeval.py`, never a hand-rolled regex. This trap has now been hit twice |
| **frame** | a suspiciously large cross-engine ratio | ours ≈ **2.07×** SF's printed cp. `classify_losses.py` itself printed the two side by side and the docs read a ratio off it — the "55.3 vs 10.5" headline is **2.54×, not 5.27×** |
| **stdin EOF is a STOP** | every count comes back 0 | never `printf \| engine`; persistent stdin only |
| **partial matches** | a tempting point estimate | 10k read **−113 at n=194** and finished **−92 at n=406**. `match.py` refuses to over-read |
| **fail-soft returns are BOUNDS** | a suspiciously small margin | re-measure bound-free with `searchmoves` |
| **cheap oracles manufacture defects** | "blunders" that vanish on inspection | a 120k oracle gave a **44% false-positive rate**; 2 of 3 OURS positions were artifacts. Verify `best` at ≥1M before spending a walk |
| **already-lost positions** | huge eval drops | at worse than −2.5 the game is gone; those drops are death throes (156/174) |
| **one trace row ≠ a total** | a startling ratio | "a1b1 got 1 node" was one row at one depth; its real total was **17,478** |
| **`--features tune` missing** | every arm identical | params are inlined constants without it |
| **in-game evals ≠ cold probes** | disagreement with a fresh search | in-game numbers come from a warm TT after 60 moves |

---

## 4. WHAT THE CORPUS IS

404 games vs SF18 on colour-paired UHO openings (203 distinct, 201 played twice).
`mine_losses.py` oracle-scores every ARCHON move in every LOST game; `classify_losses.py`
splits OURS / BOTH / NEITHER **and now runs an oracle-verification gate** (a candidate's
`best` is re-checked forced at 1M; without it 2 of 3 OURS positions were artifacts).

**The miner only ever scored LOST games. The 204 draws — 172 of them squandered wins —
were never touched, and that is where the deficit lives.** Use `lossshape.py` for those.

---

## 5. FILES

```
searchopt/
  START-HERE.md          the problem, the eliminated hypotheses, what to work on
  README.md              this guide
  LEDGER.md              every claim + what killed the dead ones
  tools/lossshape.py     per-game eval trajectories from the PGN — free, all 404 games
  tools/staticeval.py    perspective-safe static eval, both engines, one frame
  tools/oracle_cache.py  persistent deep-search cache (oracle work is the expensive part)
  tools/mine_losses.py   oracle-score every move in every lost game
  tools/classify_losses.py  OURS/BOTH/NEITHER + the oracle-verification gate
  tools/match.py         read a match; refuses to over-read a partial one
  tools/walk.py          descend one loss (LEGACY — see §0, the premise it serves is dead)
  tools/searchdbg.py     read the search's own decisions at a node
  tools/arm.py           test a tunable with a mandatory anti-arm control
  positions/             corpus + plateau + search_failures sets
  results/               one file per investigation, including the negative ones
engines/books/d48x-corpus-openings.epd   all 203 corpus openings — the test book
```

---

## 6. RETIRED — do not resurrect

- `shapeopt.py` — every metric noise-satisfiable. Deleted.
- **Per-position argmax, head-to-head, and node-allocation instruments** — all died to
  inert nulls (§2). They cannot resolve a 5 cp/move effect; nothing can, one position at
  a time.
- `RED_GIVES_CHECK` as a validated arm — its evidence died to the opposite-direction control.
- **Blunder-hunting as the organising frame.** 0.46% of moves, 44% of those artifacts,
  one actionable position in 404 games. It is not how we lose.

---

## 7. Regenerating the test book

`engines/books/d48x-corpus-openings.epd` is the right book for any arm test — all 203
corpus openings, wins included, so there is no selection bias. `engines/books/` is
gitignored, so if it is missing:

```bash
python3 - <<'PY'
import re
txt=open('engines/runs/corpus-d48x/games.pgn',errors='ignore').read()
fens=sorted({m.group(1) for m in re.finditer(r'\[FEN "([^"]+)"\]', txt)})
open('engines/books/d48x-corpus-openings.epd','w').write('\n'.join(fens)+'\n')
print(len(fens), 'openings')
PY
```

Equal-nodes match template (contention-immune — several rungs can share the box):

```bash
fastchess -engine cmd=engines/canonical/archon name=archon \
            option.EvalFile=engines/canonical/net.cenn option.Hash=64 option.Threads=1 \
          -engine cmd=engines/bin/stockfish18 name=sf18 option.Hash=64 option.Threads=1 \
          -each nodes=110000 proto=uci \
          -openings file=engines/books/d48x-corpus-openings.epd format=epd order=sequential \
          -rounds 203 -repeat -games 2 -concurrency 12 \
          -pgnout file=engines/runs/<name>/games.pgn
```

Baselines to beat, same book, same opponent: **−134 Elo at 10+0.1**, **−68 at 110k nodes**.
