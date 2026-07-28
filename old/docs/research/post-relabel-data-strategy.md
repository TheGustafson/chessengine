# Post-Relabel Data Strategy

**Created:** 2026-07-19 · owner: user + agent · governs epic #40 Phase A–E data design.
Answers the strategy questions raised once the full `official-stockfish/master-binpacks`
set (12 standard files, ~250 GB, mirrored to GCS) became available. Companion to
`sf18-canonical-epic.md` (the plan) and `data-index.md` (the catalog).

---

## 0. The reframe: we now have far MORE positions than we can relabel

**Measured (fast_fens chunk-header estimate, 2026-07-19):** the 3 *local* binpacks
alone = **~20.1 B positions** — UHO **14.84 B** (38546 chunks × ~385k), wrongIsRight
**2.51 B**, farseerT76 **2.74 B**. At ~0.4 B pos/GB the full 12-file set (~250 GB) is
**~100 BILLION positions**. (Context: prior training used a **188.5 M** UHO sample =
**1.3 %** of the full UHO pool.)

| Asset | Positions | Labels | Scale | Cost to use |
|-------|-----------|--------|-------|-------------|
| 12 raw standard binpacks | **~100 BILLION** (measured ~20 B in 3 local files) | native 5000-node | per-file (÷1.9…÷2.82, stm-POV) | **free (already labeled)** |
| SF18 warm@10k relabel | our **budget** (0.25–2 B) | SF18 @10k, White-rel | **one scale (SF18-UCI)** | ~$11 / 250M … ~$91 / 2B |

**Coverage reality:** even a 1 B relabel budget = **~1 %** of the raw pool. We can
never relabel a meaningful fraction — which is *why* the raw native pool must carry
breadth (train-on-everything, §7) and the relabel budget must be spent where a clean
scale earns the most (calibration stage-2, §1).

**Relabeling "everything" is off the table** — 42–114 B positions at warm@10k
would cost thousands of dollars and run for weeks even at the 96-vCPU cap. And we
**don't need to**: the raw native labels already trained every winning net (the B4
finding). So the strategy is **selection**, not exhaustion — relabel a strategic
*budget* where a clean unified scale buys the most, and keep the raw billions as a
free breadth substrate.

### The two assets play two roles
1. **Raw binpacks → COVERAGE (breadth).** Billions of unique positions, free native
   labels. This is what the proven two-stage recipe consumed. Weakness: mixed
   per-file scales and 5k-bound noise. Mitigation: apply the per-file divisors
   (data-index §0) to bring them to an approximate common frame; use for the broad
   stage-1.
2. **SF18-relabeled budget → CALIBRATION (precision).** A clean, single-scale
   (SF18-UCI) set on the positions that matter most for the eval↔search co-adaptation
   fix (Phase C). This is where relabel dollars go.

This layering is the answer to "relabel all in a strategic order": relabel a
**budget**, ordered by value, and let the raw set carry breadth.

---

## 1. Relabel scope + order (Phase A, revised)

**Budget recommendation: 500 M – 1 B unique positions** (~16–32 GB relabeled;
~$23–46; ~21–41 h). Rationale: big enough for a real stage-1+stage-2 on clean
labels, small enough to fit locally and to iterate in days, not weeks. Start at
the low end and grow only if Phase C shows the calibration set is coverage-starved.

**Strategic order (relabel highest-value first so Phase C can start before the whole
run finishes):**
1. **farseer family** (T76 → T74 → T75) — the stage-2 clean family that trains the
   incumbent; relabeling these first unblocks the Phase-C warm-retrain immediately.
2. **UHO + wrongIsRight** — the stage-1 breadth core (already partly done as c1@50k).
3. **New coverage** (T60T70, data_pv-2, multinet, pylon) — sampled to fill the budget,
   deduped against 1–2.
4. **Hard negatives** (Archon-vs-SF18 losses) — Phase D, upsampled.

Within each source: **game-order** (native binpack order) so the warm TT reuses
trees (Fable study; decided). Dedup on (board, stm) at manifest build (~1% dupes).
Sampling within an over-large source: per-game stride cap (as `fast_fens` already
does) to spread coverage rather than take one contiguous prefix.

---

## 2. Organization: the canonical dataset layout

**One scale, versioned, source-tagged, tiered by role.** Everything SF18-UCI cp,
White-relative, bulletformat 32 B/pos (features derived at train time — §4).

```
GCS  gs://archon-main-bucket/corpora/canonical/v1/
  manifest.json                      # provenance: sources, counts, sha, scale=SF18-UCI, relabel config
  t0-breadth/{source}/shard-*.bin    # UHO, wrongIsRight, (raw-native or relabeled breadth)
  t1-clean/{farseerT76,74,75}/*.bin  # stage-2 clean calibration
  t2-coverage/{T60T70,data_pv2,...}/*.bin
  t3-hardneg/archon-loss/*.bin       # Phase D, upsampled
  t4-selfplay/*.bin                  # Phase E
local  data/corpora/canonical/v1/    # only the tiers a given training stage needs
```

- **Tier = role**, not just source: T0 breadth, T1 clean calibration, T2 extra
  coverage, T3 hard negatives, T4 self-play. Training stages pull specific tiers.
- **manifest.json** is the contract: exact source→shard map, per-shard sha + count,
  the single scale constant, and the relabel config (warm@10k, bmi2-PGO SF18). Every
  tier also gets a `ledger/data.jsonl` row (D38).
- **Immutable + versioned** (`v1`, `v2`): a retrain is reproducible from the manifest.

---

## 3. Storage feasibility — can we train locally?

**Yes.** The key: we never need the raw 250 GB locally — those stay in GCS and are
streamed/sampled *in the cloud* during relabel. Locally we hold only the **relabeled
canonical set + the training working set**.

| What | Size | Where |
|------|------|-------|
| Raw binpacks (12) | ~250 GB | **GCS only** (stream in cloud for relabel) |
| Canonical relabeled @ 500 M | 16 GB | local + GCS |
| Canonical relabeled @ 1 B | 32 GB | local + GCS |
| Shuffled training .bin (bullet consumes) | ≈ same as tier size | local, transient |
| Net checkpoints per run (interruptible saves) | ~2–4 GB/run | local, prune to finals |
| Working headroom (shuffle buffer, temp) | ~20–30 GB | local |

**Current disk:** 251 GB total, ~16 GB free — **too tight as-is.** Plan:
- Reclaim now: drop `uho-stage1-head10gib` (10.7 GB, regen by truncation), dead-run
  intermediate checkpoints, superseded lc0/mix corpora (data-index §7) → frees ~25–40 GB.
- With a 1 B canonical set (32 GB) + ~30 GB working headroom we want **~65 GB free**;
  after reclaim + optionally dropping the raw binpacks we no longer need locally
  (they're in GCS now), that's comfortable. **Verdict: a 500 M–1 B canonical set
  trains locally on this box; a 2 B set (64 GB) needs the raw binpacks off-disk first.**

---

## 4. Search-augmented features (D41 and beyond)

Two very different meanings — both supported without bloating the canonical set:

1. **Eval FEATURES (HalfKAv2_hm + FullThreats, the D40/D41 arch).** These are
   **derived from the position at train time** by the bullet feature extractor — the
   canonical 32 B bulletformat row stores the board, and classic-768, D40-HalfKA, and
   any future feature set are all computed from it. **No extra storage; the same
   canonical dataset feeds every architecture.** This is why we relabel to positions+cp,
   not to features.
2. **Learned-search / helm data (D41 RootMoveAnalysis).** A *different, much larger*
   data type (per-move root arms × tiers; the obs factory measured ~9 MB/game). Storing
   it for 1 B positions would be TBs — infeasible and unnecessary. Strategy: the SF18
   relabeler (`root-analysis`) can emit RootMoveAnalysis, so capture obs on a **curated
   subset** (decisive / hard / Archon-loss positions) in the *same* relabel pass — one
   expensive search, two products: scalar cp for the bulk eval set + root-move obs for
   the D41 track. Bulk stays scalar; D41 gets its targeted obs. (Gated behind the
   label-value pilot per the path-to-1 reframe.)

---

## 5. Presentation / training order — does it matter?

**Two levels; opposite answers.**

- **Within a training stage: NO game-order — SHUFFLE.** SGD needs near-iid batches;
  consecutive plies of one game are highly correlated and would bias/destabilize
  training. bullet/sfbinpack already shuffle; we keep that. **Do not train in game
  order.** (Game-order is a *relabel*-time trick for warm-TT reuse only — §1.)
- **Across stages: YES — curriculum ordering matters.** The proven winner is
  **two-stage warm-retrain**: broad stage-1 → warm-start → clean/rich stage-2 at
  lower LR. Generalize to a curriculum of tiers:
  1. **Stage-1 (breadth):** T0 (+ raw-native breadth) — many unique positions, general
     eval, higher LR.
  2. **Stage-2 (calibration):** warm-start from stage-1, train on T1 clean SF18-UCI +
     upsampled T3 hard-negatives, lower LR, tuned WDL weight. This is where the single
     clean scale earns its keep (and where Phase C pairs the search SPSA retune).
  3. **Stage-3+ (self-play):** T4, the compounding loop.
- **Mixing ratios within a stage matter:** upsample hard negatives (T3) and any
  under-covered source; keep WDL weight and scale=400 constant per stage. These are
  SPSA/ablation knobs, not order.

**Summary:** relabel in game order (warm-TT); train shuffled within a stage; order
the *stages/tiers* as a breadth→calibration→self-play curriculum with warm-starts.

---

## 6. Recommended plan + open decisions

**Recommended:**
- Relabel budget **500 M → grow to 1 B** if Phase C is coverage-starved; farseer-first order.
- Canonical layout as §2 (tiered, one scale, versioned, manifested + ledgered).
- Train: two-stage warm-retrain, shuffled within stage, breadth→calibration curriculum;
  pair every eval with a search SPSA retune (Principle 3).
- Keep raw binpacks in GCS; reclaim local disk to ~65 GB free before the 1 B retrain.
- D41: capture root-move obs on a curated subset during the same relabel pass.

**Open decisions (for the user):**
1. **Budget:** 500 M (safe/fast, ~$23) vs 1 B (~$46) vs 2 B (needs raw off-disk, ~$91)?
2. **Breadth source:** relabel T0 too (clean but costs budget) vs keep T0 as raw-native
   (free, mixed-scale) and spend the budget on T1/T2/T3?
3. **Exact per-file position counts:** run a cheap chunk-count audit to size sampling
   precisely (recommended before finalizing the manifest).
4. **Capture obs-in-relabel now** (one pass, two products) vs defer D41 until the pilot pays?

---

## 7. Reasoning: "train on EVERY position we have"

We have ~42–114 B raw positions across 12 binpacks with **heterogeneous native
labels**. Can we train on all of them, and is it even the right goal? Reasoning
through the three walls:

### Wall 1 — scale heterogeneity (the labels don't share a unit)
Each binpack's score is *that generator's internal eval*, and the generators differ
(SF versions, Lc0-farseer rescores) → slopes ÷1.9 / ÷1.66 / ÷2.82, some stm-POV,
lichess White-POV (data-index §0). Two ways to reconcile, use **both**:
- **(a) Per-file rescale pass** — apply each file's audited divisor + POV fix so every
  score lands in one frame (SF18-UCI). One streaming pass over the concat; cheap
  relative to training. Removes the heterogeneity at the source.
- **(b) Train in sigmoid space with a WDL blend** — target
  `(1−λ)·σ(score/S) + λ·game_result`. This is **how Stockfish itself trains on these
  very files despite their scale drift**: σ saturates so moderate scale differences
  wash out in the loss's active region, and the **game-result term is scale-invariant**
  — a free anchor that doesn't care about any generator's units. Our trainer already
  exposes this (`D40_WDL`, `SCORE_MULTIPLIER`, scale=400). λ≈0.25–0.5 lets WDL carry
  the files whose cp scale we trust least.
> Verdict: rescale (a) to get everyone into one frame, then rely on σ+WDL (b) to
> absorb the residual. Perfect per-file calibration is unnecessary — that's the
> insight that makes "everything" trainable at all.

### Wall 2 — logistics (250 GB won't fit on an 18 GB-free box)
- **bullet/sfbinpack already stream a multi-file concat** from disk (comma-separated
  `NET0_DATA`), with an internal shuffle buffer — no need to hold it in RAM, but it
  does need the bytes on a fast local disk.
- **The data already lives in GCS.** So full-corpus training belongs **in the cloud**,
  next to the data, on a box with a big cheap disk + a GPU (or a strong CPU trainer) —
  the same D36 direction as distributed testing. Bring the resulting stage-1 net back
  local for the clean stage-2 + eval. Alternatives (worse): resize this box's disk to
  ~400 GB; or gcsfuse-stream from GCS (network-bound, slow); or sharded local passes
  (download shard → N superbatches → delete → next), which fights the shuffle and
  hurts iid — acceptable only with a large shuffle buffer.

### Wall 3 — diminishing returns ("all positions × many epochs" is overkill)
You never epoch over 100 B positions. Training is a **fixed presentation budget**
(superbatches × batch size) sampled from the pool; the winning nets saw **3–9 B
presentations**. At 100 B unique positions, even one pass = 100 B presentations —
far past where Elo/position flattens. So "train on everything" really means **"use
the full pool as the sampling source for a single-digit-billion presentation budget"**
— which is exactly what SF does. Beyond coverage of the position *distribution*, more
raw positions buy sharply diminishing Elo; label *quality* and *curriculum* matter more.

### De-dup
Global dedup across 100 B keys is its own huge job (TB-scale external sort) and SF
doesn't bother — duplicates are mild upweighting the sigmoid tolerates. Skip global
dedup; optional per-file dedup only.

### The concrete recipe for "everything"
1. **Rescale pass** (cloud): stream each binpack → apply its audited divisor + POV
   fix → write a re-scored copy in one SF18-UCI frame (or carry a per-file scale into
   the loader). Ledger each output.
2. **Stage-1 breadth** (cloud, big disk): bullet over the **full multi-file concat**,
   native rescaled labels, σ+WDL (λ≈0.3), high LR, a fixed budget (~2–4 B presentations),
   shuffle buffer across files. This is the "train on everything" pass.
3. **Stage-2 calibration** (local): warm-start from stage-1, retrain on the clean
   **SF18-relabeled** budget set (§1) + upsampled hard-negatives, low LR — the precision
   layer where one exact scale matters (and where Phase C pairs the search SPSA retune).
4. **Gate** at equal time; attribute Elo to (breadth vs calibration) honestly.

> Bottom line: "train on everything" is viable and is essentially the proven SF
> recipe — **full-pool native-label breadth stage in the cloud (σ+WDL absorbs the
> scale mess), then a clean SF18 stage-2**. The relabel budget is best spent on
> stage-2, not on trying to relabel the whole pool. This unifies the two questions:
> the raw pool gives breadth for free; the relabel gives calibration where it counts.
