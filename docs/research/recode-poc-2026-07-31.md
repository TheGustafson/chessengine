# Recode POC — the learned threat re-coding, built and measured (2026-07-30/31)

**Status:** working proof of concept, landed. The engine half (deploy mode
(ii)) is implemented, gate-verified behaviourally invisible for every existing
net, and measured; the trainer half (phase 2) is implemented and CPU-verified;
the Tier-1 panel (phase 3) is built, smoke-tested, and armed to self-launch
when the GPU frees. Predecessor design:
`docs/research/learned-feature-code-study-2026-07-30.md` (a4d5aec).

## Phase-1 verdict up front: does the smaller table buy measurable cycles?

**YES — the cache win is real, directionally robust, and small-to-moderate:
−1.5% cycles (paired median) at R'=8192 on a quiet-ish box, growing to −3…−16%
under memory pressure. It is not the −11.5% class of win; it prices at ~+2.5
Elo of speed quiet, more under load — plus a 69 MB smaller net and a ~316M-cycle
cheaper process launch.** Frame-validity evidence sits beside every number
below.

### The frame (commissioning recipe, followed exactly)

Same binary (native `sota,arch-d48x` release, `target/native-recode/`), two
**eval-bit-identical** nets differing ONLY in threat-table size:

- **Arm A "scatter"**: full 60,720-row threat table (77.7 MB at ACC=1280),
  `W'[t] = Wsmall[remap[t]]` scattered back, + an **identity** TRM1 so the
  instruction stream includes the same remap lookup.
- **Arm B "compress"**: the 8,192-row table (10.5 MB) + the real TRM1.

Both were built by `tools/nnue/recode_pack.py` from the canonical D48X with a
balanced random partition (`perm % R'`, seed 1) and census-weighted projection
(T0b). Eval identity verified per battery by `nnue-eval` sha256 over wl50:
both arms hash `591b92e1…`; the canonical net through the same binary still
hashes the pin `2e1ede95…`. Fixed **node** budget
(`bench 16 1 100000 wl50.fens nodes <net>`), arm order alternated per round,
`cycles:u` + `instructions:u`.

### Frame validity, measured

- **Node identity**: one node total (5,001,626) across all 28 runs of the
  14-round battery — the arms search literally the same tree.
- **Instruction parity**: raw medians differed −0.79%, which decomposes
  ENTIRELY into net LOAD (the loader FNV-checksums + copies a 135.6 MB vs
  68.3 MB payload; a nodes=1 battery measures the startup medians directly:
  A 1.215 G cycles / 1.610 G instr, B 0.898 G / 1.375 G). **After subtracting
  per-arm startup medians, the instruction medians agree to 14 instructions
  in 28.2 billion (−0.0000%).** The remaining free variable is cycles.

### The cycles verdict (R'=8192, 14 rounds, startup-subtracted)

```
paired rounds where B faster : 12/14        (binomial p ≈ 0.006)
pairwise (14×14)             : 131/196 for B
median paired Δ              : −1.49%
best / worst round           : −7.32% / +5.66%
min-vs-min (least interference): −0.35%
```

The effect is **load-dependent** with a consistent sign: splitting the battery
at its visible load boundary, the busy half's paired median is ≈ −2.6% and the
quiet half's ≈ −0.7%. An independent 8-round battery with miss counters
reproduced the direction (6/8 rounds, and −16.0%/−11.5% in its two
heavily-loaded rounds). Match conditions run concurrent games, so the loaded
regime is arguably the more deployment-relevant one.

### Mechanism (8-round battery, `cache-references/misses`)

Median cache-misses fell only −1.75% (A 289.4M → B 284.4M, and ~3M of that is
startup page traffic) while cycles fell more — because the **residual miss
population is dominated by the tables the recode does not touch**: HalfKA
(28.8 MB), the TT (16 MB at this hash size), and PSQ1 (2.7 MB). Even with the
threat table at 10.5 MB the engine's combined hot data (~58 MB) still exceeds
the 32 MB L3; under external cache pressure the big table's misses get
dramatically more expensive, which is exactly where B's advantage explodes.
This **partially confirms the predecessor's §8.1 suspicion**: the quiet-box
miss profile is substantially a resident-hot-subset story, so shrinking one
table buys a fraction of the naive bound — and it says the next speed lever of
this kind is the **HalfKA table** (R2a coarsening: 28.8 → ~10 MB), which
together with Recode-T would bring the whole input machinery near L3 reach.

### Dose response (R'=4096, 5.2 MB table)

An interleaved 10-round battery (both pairs run inside the same rounds, so
both doses see the same box conditions; startup-subtracted; instruction
parity +0.0000% for both pairs; node identity single-total for both):

| dose | table | rounds B faster | pairwise | paired median Δ | worst / best round |
|---|---|---|---|---|---|
| R'=8192 | 10.5 MB | 6/10 | 56/100 | −0.45% | +12.46% / −3.18% |
| R'=4096 | 5.2 MB | 8/10 | 79/100 | −1.85% | +4.27% / −8.99% |

The R'=8192 column is dragged by one +12.5% interference outlier (excluding
it, the paired median is −1.90%); R'=4096 trends slightly better, consistent
with its table fitting deeper into L3, but the two doses are within each
other's noise — **most of the win is already collected at 8192**. A separate
non-interleaved 10-round battery at R'=4096, which happened to run while a
k-means job was hammering the box (a natural high-pressure regime), showed
**10/10 rounds for B, paired median −4.81%** raw.

Across every 100k-node battery in this study the sign count is **32/42
rounds for the smaller table (sign-test p ≈ 4·10⁻⁴)**, with per-battery
paired medians between −0.45% and −4.8% depending on load.

### What phase 1 does NOT show

- No strength claim: both arms are the same (projected) eval by construction.
  Quality is phase 3's question.
- The Elo pricing (~120/doubling) applies to quiet single-engine NPS; the
  loaded-regime numbers translate less directly.
- One box, one binary, one bench suite. The 16 MB TT is bench's default; a
  bigger hash (typical match setting) would raise cache pressure and likely
  the win.

## What landed (all committed on this branch)

1. **Engine — TRM1 threat re-coding (deploy mode ii).** CENN nets may carry
   `TRM1` (`u16[60720] → threat_rows`) + a compressed threat table; the remap
   applies at the single resolver chokepoint
   (`ThreatResolver::index_decoded`), shared by the incremental fold, the
   from-scratch refresh, and the PSQ1 skip channel — the three paths cannot
   drift. `None` = identity: **a net without TRM1 loads and evaluates
   bit-identically** (verified: `CHECK_NNUE_GATE=1 tools/check.sh` exits 0
   with `Bench: 587683`, eval hash `2e1ede95…`, d13 `1739588` — all three
   pins bit-for-bit). Correctness pins: TRM1 round-trip + rejection tests,
   and `recode_pair_is_eval_identical_over_a_game` (compressed vs scattered
   twin, bit-identical evals over a random game, with the debug
   incremental-vs-refresh oracle running at every node in the re-coded
   space).
2. **Tools.** `tools/nnue/recode_pack.py` (arm-net packer: scatter / compress
   / tied; random / identity / file codes; census-weighted projection),
   `recode_speed_ab.sh` + `recode_speed_report.py` (the phase-1 frame with
   its validity gates), `recode_kmeans.py` (frequency-weighted k-means over
   the incumbent's threat rows; heavy-hitter init).
3. **Trainer (phase 2).** `TrainConfig.threat_remap` + `ArchSpec.threat_rows`;
   `map_batch(..., threat_remap=)` re-codes indices at the loader with -1
   padding preserved; both stream builders pass it through. **Zero
   trainer-graph changes**, exactly the seam the study pre-registered.
   Verified on real records: remapped indices equal `remap[raw]` everywhere.
   (Worktree `jax/` only; the live D49 trainer reads main's copy, untouched.)
4. **Tier-1 panel (phase 3), built and armed.** `jax/tools/recode_tier1.py`
   (project-and-finetune: warm from D48X sb520-EMA raw, threat tensors
   projected — freq-weighted means or k-means centroids — heldout every
   superbatch, sb0 = the T0b readout), three arm specs
   (`tools/nnue/runs/recode_t1_{identity,random8k,kmeans8k}.json`, 12
   superbatches at LR 1.1e-4 on the D48X mix/frame), and a **polite
   launcher** (`recode-t1.service`) that waits for `archon-d49` to exit AND
   10 consecutive idle-GPU minutes before running the arms sequentially. The
   k-means code artifact exists (R'=8192, 18 iters, converging, no empty
   classes; heavy hitters keep singleton classes, the never-fires collapse
   into one ~25.6k-row zero class — the learned-sketch allocation). CPU
   smoke: warm→project→forward runs; one-batch heldout identity 0.006328 vs
   kmeans8k-projected 0.006661 (+5.3% relative before any finetuning — the
   quantity the finetune is supposed to recover, and the panel's question is
   how much of it, relative to the random-code null).

The **pre-registered decision rule** (study §5.2) is restated verbatim in the
driver's docstring and will not be changed after readout. Deviation from the
pre-registered panel: the fly-hash k=2 control is NOT in this first panel —
k=2 needs a wider active-index padding in the loader (2× THREAT_MAX_ACTIVE)
and is deferred; the three arms that the decision rule actually consumes
(identity, random-8k, learned-8k) are all present. The 16k dose point is a
one-json addition if rule (3) needs it.

## Least sure about, in order

1. **Whether the Tier-1 finetune recovers the projection cost.** +5.3%
   relative one-batch MSE at sb0 is a real dent; 12 superbatches at low LR
   may be too short for the tail rows the code merged. If all arms fail rule
   (2), the pre-registered answer is to report it and move to the
   splits/vocabulary arms — not to stretch the rule.
2. **The k-means metric** (study §8.4): weighted L2 in row space produced a
   heavy-hitter/singleton structure that looks right, but rows interact
   through the nonlinear head; if kmeans-8k ≈ random-8k at readout, suspect
   the metric before the thesis.
3. **The load-dependence of the speed win.** The quiet-box number (−0.7%…
   −1.5%) and the loaded number (−3%…−16%) bracket reality; which applies to
   real matches depends on concurrency and hash size, and I did not measure
   a games-running-concurrently regime directly.
4. **The identity-arm reproduce.** The panel's instrument null (identity sb0
   must equal the incumbent's heldout, and random-8k must be distinguishable
   from run noise) is exactly rule (1); if it fails, the whole panel is
   void and says so itself.
5. **k-means ran on one corpus band's census** (uho-band75-100). The mix
   matters (study §5.4); a mix-weighted census is a cheap follow-up before
   any Tier-2 retrain.
