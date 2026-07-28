# Structural audit: where ARCHON actually differs from Stockfish

2026-07-27, net-live era. Source of truth for Stockfish is master `eca43a97`,
read explicitly — never the checkout, never the `sf_18` tag.

The playbook's rule is *structure first, constants second*: a structural
difference never shows up in a parameter sweep, so no amount of SPSA will find
it. This is that pass over the move-ordering and threat machinery. It is
mostly a list of things that are NOT different, which is the point — a
verified negative costs one read and saves a campaign.

---

## 1. Verified identical to `eca43a97`

**History table geometry.** Every dimension and every bound matches:

| table | ARCHON | SF `eca43a97` |
|---|---|---|
| butterfly | `[2][65536]`, D=7183 | `Stats<i16, 7183, COLOR_NB, UINT_16_HISTORY_SIZE>` |
| low-ply | `[5][65536]`, D=7183 | `LOW_PLY_HISTORY_SIZE=5`, same |
| capture | `[12][64][7]`, D=10692 | `[PIECE_NB][SQUARE_NB][PIECE_TYPE_NB]`, same |
| continuation | D=30000 | same |
| pawn | 8192 slots, D=8192 | `PAWN_HISTORY_BASE_SIZE = 8192` |
| correction | 65536 slots, limit 1024 | `CORRHIST_BASE_SIZE = 65536`, `CORRECTION_HISTORY_LIMIT = 1024` |
| tt-move | D=8192 | `TTMoveHistory = StatsEntry<i16, 8192>` |

I went looking for a *history frame* problem — the hypothesis that our tables
are sized differently, so each entry sees a different number of updates and
every `statScore / DIVISOR` transcribed from Stockfish means something else.
**Refuted.** Indexing by `Move::raw16` is not an approximation of Stockfish's
butterfly either; Stockfish indexes by `m.raw()` into a 65536-entry table too.

One real difference, with no effect at our test settings: Stockfish's pawn and
correction histories are `DynStats`, sized `numThreads × BASE_SIZE` and
**shared across threads** with atomic entries. Ours are per-thread. At
`Threads=1` — every SPRT we run — the geometry is identical. This only becomes
a divergence when SMP testing resumes.

**Move-ordering structure.** The stage machine is stage-for-stage identical
(`MAIN_TT … QCAPTURE`, same fallthroughs, same pointer juggling for bad
captures). `select()` applies the same `!= ttMove` filter. The scoring
formulas match term for term, including the continuation slabs actually read
(0, 1, 2, 3, 5 — slot 4 is written but never read for ordering, in both
engines), and every constant: `CAPTURE_MVV_FACTOR 7`, main/pawn history
weights 2 and 2, `QUIET_CHECK_BONUS 16384` at SEE margin −75, threat factor
20, low-ply weight 8, `GOOD_QUIET_THRESHOLD −14000`, good-capture SEE divisor
18, quiet sort limit −3560·depth. `threatByLesser` is built with the same
cumulative unions.

So the ordering path is a faithful transcription. If our first-move cutoff
rate is lower than Stockfish's, it is **not** because we order moves by a
different rule.

(And the "Stockfish's fhf is 93–95%" figure that used to justify that
comparison is folklore — never measured with our definition, and the 89.4%
it was compared against was a PSQT run. Both have been removed from
`screen.py`.)

---

## 2. The two real deviations

### 2.1 `QUIET_OFFENSE_FACTOR` — an ARCHON original, never individually tested

Stockfish's quiet score rewards *escaping* a threat and penalises *stepping
into* one. We add the mirror term (D30, from the Reckless study): a quiet that
moves to a square attacking an undefended enemy piece worth at least as much
as the mover earns `10 × piece_value`.

Two facts about it:

- It landed inside the **batched Wave C–G pass** (D32), which was validated by
  a single bug-tripwire SPRT over items 11–21 together. Nothing has ever
  measured this term on its own.
- It costs a **magic attack lookup per quiet move**, at every main-search node
  where any undefended enemy piece exists — which is most of them.

That combination (unvalidated, in the hot loop, ours to change freely) makes
it the highest-value single knob in this file. The test is two-stage and the
stages measure different things:

1. `QUIET_OFFENSE_FACTOR=0` — measures whether the **bonus** is worth
   anything. Cheap, deterministic, already tunable.
2. Deleting the code — measures the **NPS** the computation costs. Step 1
   cannot see this, because setting the factor to zero still runs the
   `attacks()` call and still builds the `vulnerable[]` table.

Do not skip to step 2, and do not read step 1 as pricing the removal.

### 2.2 The threat map is eager where Stockfish's is lazy

`Board::set_threats` runs from `set_check_info`, i.e. on **every
`make_move`** — every node, including every qsearch node. It computes, for the
side not to move: pawn fill, knight attacks, bishop/rook/queen slider attacks
through our king, the king ring, six per-type sets and their union. In a
normal middlegame that is roughly **six magic slider lookups per node**, into
tables far larger than L1.

Stockfish computes the equivalent inside `score<QUIETS>` — so only at nodes
that actually generate quiets. It also computes strictly less: no queen set,
no king set, no union.

Our consumers are exactly two:

- `MovePicker::init_quiets` (main search, and only when quiets are generated)
- the RFP threat-coupled relaxation at `src/search/mod.rs:2792`

**qsearch never reads it.** Its picker runs `QSEARCH_TT → QCAPTURE_INIT →
QCAPTURE`, or the evasion pipeline in check; neither path calls `init_quiets`.
Since qsearch is a large share of all nodes, a large share of this work is
provably discarded.

Two opposite responses, and which is right is an empirical question, not a
matter of taste:

- **Make it lazy** — memoise per `StateInfo`, compute on first read. Pure
  speed, node-identical, eval-identical.
- **Keep it eager and use it more** — see below.

---

## 3. The ARCHON-original angle

The eager threat map looks like an inefficiency, but it is also an *asset that
Stockfish does not have*, and the asymmetry is worth stating precisely:

> Because we maintain the threat map in `StateInfo`, the marginal cost of
> consulting it at a **new** decision point is zero. For Stockfish, each new
> use site costs a recompute, so it can only afford threat-conditioned logic
> where it already pays for the sets.

That inverts the usual copy-Stockfish economics. A family of
threat-conditioned gates — threat-aware LMR (reduce less for a move that
escapes a threat on a valuable piece), threat-aware futility (do not prune a
quiet that unpins or evacuates a hanging piece), threat-conditioned
extensions — is **cheap for us and expensive for them**. That is a defensible
place to be original rather than faithful, and it is original by architecture
rather than by parameter.

The decision between "make it lazy" and "amortise it harder" should be settled
by the equal-time vs equal-nodes rematch: if the Elo gap is dominated by
speed, take the lazy win first; if it is dominated by judgement, spend the
sunk cost on more use sites.

---

## 3b. The eval side: two things Stockfish has that we do not run

### `nnueComplexity` is dead code in ARCHON

Stockfish derives a free uncertainty signal from its architecture — its
feature transformer emits a PSQT head alongside the positional head, so
`complexity = |psqt − positional|` costs nothing — and spends it twice:

```
optimism += optimism * complexity / 476;
nnue     -= nnue     * complexity / 18236;
```

We have both constants (`COMPLEXITY_OPTIMISM_DIV = 476`,
`COMPLEXITY_EVAL_DIV = 18236`, exact matches) and **neither term ever fires**:
`complexity` is hardcoded 0 because our CENN-v2 net has no psqt/positional
split. So a signal Stockfish uses at every evaluated node is simply absent
from ours.

We do have a substitute Stockfish does not: a **WDL head**, already computed
per eval and already consumed at the root for time management
(`wdl_tm_factor`, behind `ARCHON_WDL_TM_C3`). Draw probability is a calibrated
uncertainty measure rather than a proxy for one.

Honest prior, split by term:

- The **eval damping** half is probably near-neutral. `|eval|` and draw
  probability are strongly inversely related, so `eval × (1 − k·p_draw)` is
  close to a monotone squash of `eval`, and a monotone reshape cannot change
  move order within a node. It only moves pruning thresholds — which is the
  same class of change the five frame formulations already failed to profit
  from.
- The **optimism** half is not a reshape of eval at all, so that objection
  does not apply to it. Test the two separately; a combined arm would confound
  them.

### Stockfish deleted its dual net; we still have the machinery

At `eca43a97` there is a single `Network` — no `NetworkBig`/`NetworkSmall`, no
`smallNet` threshold anywhere in `evaluate.cpp`. Stockfish removed the
adaptive-eval routing it used to have.

ARCHON still has it, twice over, both inert:

- `active_smallnet()` — a cheap-net gate (perf study #3 item #8), env-driven
  via `ARCHON_SMALLNET` / `ARCHON_SMALLNET_THRESH`, `None` unless armed. Runs
  the cheap net's HalfKA channel and returns early on decided positions,
  skipping the big net's FullThreats fold. That skip *is* the win.
- a "t-shirt" S/M/L tier gate for critical positions.

Both are blocked on the same missing artifact: **no cheap net has been
trained**. That is a training task, not a tuning one, and the box is committed
to D48X.

Worth stating plainly: Stockfish removing this is evidence it was not worth
the complexity *at their operating point*. Ours differs — we carry an 84 MB
net and, if the rematch says the gap is speed, a cheaper eval on decided
positions is aimed exactly at that. It is a hypothesis their deletion weakens
but does not settle.

---

## 4. Queued experiments

| # | experiment | instrument | why it is worth a slot |
|---|---|---|---|
| 1 | `QUIET_OFFENSE_FACTOR=0` | `screen.py` cp + fhf + nodes | unvalidated ARCHON original in the hot loop |
| 2 | lazy threat map | perf + bench NPS, node-identical | provably discarded work at every qsearch node |
| 3 | threat-conditioned LMR / futility | `screen.py`, then SPRT | free for us, costly for Stockfish |
| 4 | `RAZOR_CORR_DIV` / `NMP_CORR_DIV` at 182069 | `screen.py` | SF applies `\|correctionValue\|` at 3 sites; we have plumbing for 5, inert at 1e9 |
| 5 | `FRAME12` ×0.64, and `HINDSIGHT_EVAL_SUM=111` alone | `screen.py` | the 12 eval constants no frame formulation ever moved |
| 6 | optimism boosted by WDL uncertainty | `screen.py` | replaces a dead SF term with a signal only we have |

Experiment 1 is deterministic under load and can run on a busy box — verified
2026-07-27, where a `nice 19` screen alongside a timed match reproduced the
control row exactly (cp@64k 4.64, top-1 66.5%). Fixed-node cp loss, fixed-depth
bench and fhf counters do not depend on wall clock. Experiment 2 is the
exception: its whole claim is wall-clock, so it needs a quiet box.

That distinction is worth keeping as a rule. **Deterministic instruments can
be niced onto a busy box; wall-clock instruments cannot run at all.** It is the
difference between waiting an hour and not.
