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

## 4. Queued experiments

| # | experiment | instrument | why it is worth a slot |
|---|---|---|---|
| 1 | `QUIET_OFFENSE_FACTOR=0` | `screen.py` cp + fhf + nodes | unvalidated ARCHON original in the hot loop |
| 2 | lazy threat map | perf + bench NPS, node-identical | provably discarded work at every qsearch node |
| 3 | threat-conditioned LMR / futility | `screen.py`, then SPRT | free for us, costly for Stockfish |

Experiment 1 is deterministic under load and can run on a busy box.
Experiment 2 needs a quiet box, because its whole claim is wall-clock.
