# What the Integral study means for our Rust design

The code/technique payoff of the 11-note Integral study: a prioritized,
roadmap-ordered list of what we could **adapt** (never copy — Integral has
**no LICENSE ⇒ all-rights-reserved**, stricter than D27's SF/Reckless rule;
ideas, algorithms, constants, and formulas only). Distilled from
`integral_reference/01-*.md` … `11-*.md`. The *process/success-factor* payoff
— the headline the user cares about most — is the sibling note
**[00-success-factors.md](00-success-factors.md)**; this file is the
technique-adaptation companion, modelled on
`reckless_reference/00-design-implications.md`.

Integral is `aronpetko/integral`, C++, top-6 / TCEC-competitive, ~10,500 LOC,
~98.5% one solo developer. Its internal eval scale (QA=255, eval scale 200) and
constants are its own; compare shapes, not raw numbers.

Every item names the exact owning doc/section, the phase, rough effort
(S/M/L), expected impact (lo/med/hi), and a status tag: **[novel]** (new idea
for us) · **[already-planned]** (our docs already name it; Integral is
confirmation or a worked example) · **[already-have]** (we implement it;
recorded for completeness) · **[conflicts-Dxx]** (bumps a pinned decision —
needs a decisions.md amendment first, per README process rules).

**The meta-point first:** unlike the Reckless study, Integral surfaces
*relatively few genuinely novel search techniques we lack* — because we source
from *current* SF (D20/D27) and Integral is consistently the **cheaper,
less-hardened variant** at nearly every search step (looser TT-cutoff, IIR gate
two plies shallower, SF-classic 2-D LMR table, SF-flat ProbCut, a
singular/negative-extension shape it took months to reach what we ship from the
start). The Integral value is therefore mostly **(a) validation** that our
SF-fidelity strategy is sound, **(b) a handful of cheap, cheaply-testable
search micro-refinements**, **(c) strong convergent confirmation of our NNUE
upgrade-path and datagen plans**, and **(d) the process/tooling gap** (SPSA +
distributed testing) that dominates — owned by the success-factors note.

---

## Executive summary — the highest-value code/technique takeaways

1. **The genuinely novel search idea worth a real look: `eval_complexity`.**
   Integral stashes `|corrected_static_eval − raw_static_eval|` — a free
   by-product of correction history — and reuses it as a "trust this node less"
   signal in *both* the RFP margin (widen it) and the LMR reduction (reduce
   less). Present in **none** of SF, Reckless, or our current design. Cheap
   (reuses two values search already computes), plausibly a free win. `design/09`
   §3.10 + `design/10` RFP/LMR. **S / lo-med. [novel]**

2. **A cluster of one-line, cheaply-SPRT-able P9 search refinements**, each a
   small addition Integral's own tuning measured positive: shallow-ttMove IIR
   disjunct (`!tt_move || tt.depth+4 < depth` — may explain our reverted Wave-B
   IIR regression); LMP `|| eval >= beta` disjunct; exempt direct-check quiets
   from parent futility (we already compute the direct-check proxy for Wave-G).
   Each isolated-SPRT `[0,3]`. **[novel]**

3. **Strong convergent confirmation of our NNUE plan is the biggest cluster.**
   Integral is a *third* independent engine (after Reckless) confirming: the
   split-halves pairwise-multiply activation (FT_SHIFT=9, **identical** in both),
   the int8-sparse-FT→f32-tail hybrid quant, NNZ sparse propagation, Finny-table
   refresh cache, ply-indexed lazy accumulator, and — an **exact** match to a
   formula we had only chosen, not verified — the uniform output-bucket formula
   `(popcount(occ)−2)/4`. And it confirms our *eager DirtyPiece* choice (contra
   Reckless's lazy one). These are P13 upgrade-path de-riskers, not new work.

4. **One real conflict worth an honest flag, not adoption: killer moves.**
   Integral is a second modern top-6 engine that *keeps* 2 killers/ply feeding
   LMR, actively tuned — direct counter-evidence to D17's "killers are subsumed
   by continuation history." Recorded as **[conflicts-D17]**: not adopted, but
   the "SF removed them ⇒ settled" framing is weaker than D17 assumes.

5. **The dominant lesson is not code — it is SPSA + distributed testing.** The
   `TUNABLE`/`TUNABLE_STEP` scaffold (126 tunables, OpenBench-format print) is a
   cheap, mechanical, one-time build worth doing *now, before* the compute to
   feed it exists. Full treatment in [00-success-factors.md](00-success-factors.md);
   the code shape is item X-1 below.

---

## P9 — search: pruning / reductions / ordering (NOW; `design/10`, `09`, `05`, `04`)

The current phase. Integral diverges from our SF-faithful shape mostly by being
*simpler*, so the adoptable set is small and cheap. All search changes carry a
`Bench:` trailer (D24) and go through an isolated proxy SPRT `[0,3]` (or fold
into the D32 batch's tripwire).

### Genuinely novel, worth measuring

1. **`eval_complexity` distrust signal.** Stash
   `|corrected_static_eval − raw_static_eval|` once per node; feed it into the
   RFP margin (widen when correction disagreed a lot) and the LMR reduction
   (reduce less). Free reuse of values already computed; in none of SF/Reckless/
   us. `design/09` §3.10 (correction_value), `design/10` RFP/LMR formulas.
   **S / lo-med. [novel]**

### Cheap one-liners Integral's own SPRTs measured positive

2. **Shallow-ttMove IIR disjunct.** Fire IIR when `!tt_move || tt.depth+4 <
   depth`, not only `!tt_move` (Integral #131). May explain part of why our
   Wave-B item 9 IIR regressed (it never covered the shallow-but-present ttMove
   case). `design/10` Wave B item 9 (parked on `wip/p9-item9-iir-followpv`);
   isolated SPRT `[0,3]`. **S / lo-med. [novel]**
3. **LMP `|| eval >= beta` disjunct** on the improving-divisor term (Integral
   #203). One-line addition to an existing formula. `design/10` §3.5.
   **S / lo. [novel]**
4. **Exempt direct-check-giving quiets from parent futility pruning** (Integral
   #236). We already compute a cheap direct-check proxy per the Wave-G nps-micro
   note, so this is nearly free to wire. `design/10` §3.5 parent futility.
   **S / lo. [novel]**

### Compute-once / reuse-everywhere hygiene (verify we don't already do it)

5. **One blended move-history scalar per move, computed once pre-make**, reused
   by every pruning guard at the node (LMR / LMP / FP / history-prune /
   SEE-prune) instead of re-derived per site. Architecturally tidy; **verify
   `src/search/mod.rs` doesn't already do this** before assuming a gap.
   `design/09` §3.9 / `design/10` pruning call sites. Refactor-only,
   bench-neutral. **S / lo. [novel-or-already-have]**
6. **One check-zone bitboard per node, reused twice** — the quiet check-bonus
   (ordering) AND the futility-exemption (item 4). Cheap to verify whether our
   SEE-gated check bonus already exempts checks from FP. `design/09` §3.7 /
   `design/10` FP guard. **S / lo. [novel]**

### Correction-history experiments (Wave C tail / P9 tail)

7. **Add a `major_key` (rooks+queens) correction-history table alongside our
   `minor_key`, and measure.** Integral tried minor+major together and dropped
   minor (the *opposite* of SF's/our minor-only choice) — worth the direct
   "add major, keep minor, see what dominates" experiment. Needs a new
   incremental key. `design/09` §3.10, `design/04` **D9** (new key).
   **M / lo-med. [novel] (touches D9 — a new incremental key, record before the
   commit).**
8. **Three-way continuation-history write-side correlation test**: our
   consistency-multiplier vs Reckless's flat/unweighted vs Integral's shared
   "total-score" gravity (sum the scores at all write-offsets, use that sum as
   the gravity reference for each write). Three engines picked three mechanisms
   for the same intuition; genuinely undetermined. `design/09` §3.9
   `update_continuation_histories`. **S / lo. [novel]**

### Micro-op

9. **Pre-filter pinned knights out of full-legal generation**
   (`pieces(us, Knight) & !blockers_for_king(us)`) — a pinned knight can never
   have a legal destination, so this saves a `Board::legal` probe + MovePicker
   score per generated pinned-knight move at the cost of one Bitboard AND per
   node. `design/05` §3 per-piece loop (`movegen.rs`). **Must be perft- and
   bench-neutral (D24 ritual).** **S / lo (nps). [novel]**

### Confirmations (no action — Integral validates a call we already made)

10. **statScore ordering before SEE-pruning** — Integral converged on this via
    its own SPRT (#235), matching a call our design already made. **[already-have]**
11. **RFP threat-coupling / from-scratch per-node threat map** — Integral
    independently built an `opponent_easy_capture` RFP term *and* recomputes its
    whole threat map from scratch every node (scalar loop). Third independent
    convergence (after Reckless's setwise fill) validating **D30 / Wave G**
    (already landed). **[already-have]**
12. **SF-flat ProbCut with no `improving` coupling** — Integral ships the
    classic flat two-step (not Reckless's dynamic scheme). Negative evidence that
    Reckless's extra ProbCut complexity is not required for top-10; we already
    landed the SF-flat version. **[already-have]**
13. **Do NOT adopt Integral's classic 2-D LMR table** (quiet/tactical split,
    `ln(depth)·ln(moveCount)`). It differs from *both* current SF (continuous
    two 1-D tables + aspiration term — our shape) and Reckless (`ilog2(depth)`
    only). No shape is shown to dominate; a top-6 engine using a different one
    does not refute ours. `design/10` §3.3. **[already-have]**
14. **Material-phase eval scaling in the *search* layer** (eval ×
    (base+material)/scale) — near-identical shape in Integral **and** Reckless
    (three engines with SF converging). A validated, transferable pattern for
    post-P13 eval/search-boundary work (D21 HCE deferred, but the shape
    transfers). `future eval/search boundary`. **[already-planned (pattern)]**

### The one real conflict (flag, don't adopt)

15. **Killer moves.** Integral keeps 2 killers/ply feeding move-ordering **and**
    LMR (`kLmrKillerMoves`), still actively tuned (commit `8b017b02`, 2025) — a
    second modern top-6 engine with a full continuation-history stack that did
    *not* drop killers. Direct counter-evidence to **D17**'s premise. **Not
    recommended for adoption** (our own note 02 analysis: none of the three
    studied engines' shapes dominates, so this is not a refutation, only a
    reminder the "modern engines removed killers" framing is "some," not "all").
    Recorded so the assumption isn't treated as more settled than it is; an
    actual test would need a **D17 amendment first**. `design/decisions.md` D17,
    `design/09`. **[conflicts-D17].**

---

## P10 — time management completion + full UCI (`design/12`)

Integral's TM is materially *simpler* than our SF-ported formula (no ply/phase
term, no `movestogo` branch; formula borrowed once from Alexandria then
SPSA-refit 6×) — a data point, not an adoption target. One reusable trick:

1. **Fixed power-of-two node-effort array** indexed by `move.data() & 4095`
   (4096 slots) in lieu of a `HashMap<Move,u64>` for per-root-move node-effort
   accounting: zero heap, no hashing, harmless collisions at this scale — fits
   our no-heap-in-hot-paths convention. `design/12` §3.9 node-effort factor.
   **S / lo. [novel]**
2. **Even a top-6 engine needed real TM iteration** (a full rewrite #98 + 3
   time-loss bugfixes *after* the formula was shipped and SPSA-tuned) — process
   calibration that TM is iteration-heavy, not a one-shot. **[process, no code]**

---

## P11 — threading: lazy SMP + NUMA (`design/13`)

Integral is the *minimalist* of the three studied engines (no ThreadPool/Worker
split, no best-thread voting, no jitter, no NUMA replication until very late) —
and we already independently match or exceed it. Almost everything here is
validation + one cautionary bug class.

1. **First-touch NUMA by construction/reset on the owning thread — for BOTH
   one-time construction AND every per-`go` re-touch.** Integral took one
   abandoned false start (an `emplace_back` data race, apparently caught
   before merge — it's not even an ancestor of `main`, and its PR number is
   a clean gap in `main`'s merged sequence) plus 3 commits landing over
   ~2.5 months (including a stack-lifetime bug Rust's borrow checker would
   reject) to reach exactly what our `src/threads.rs` does from day one of
   P11 — corrected here from an earlier "3 attempts over ~15 months,
   10-week-live data race" claim in this doc, which didn't survive a
   git-log date/ancestry check (see `06-smp-numa-scaling.md` §6 for the
   full correction). **No code change**; add as the citation validating
   that a future NUMA revisit (`design/13` §3.15) is already low-risk.
   **[already-have]**
2. **Keep the TSan / loom / stress-test plan (`design/13` §5).** Integral's
   NUMA work produced two real concurrency bug shapes: an `emplace_back`
   data race (never merged, apparently caught pre-merge — *not* evidence
   the SPRT gate missed it) and a barrier-lifetime bug (which did land on
   `main` before a next-day follow-up fix). Both are bug classes a
   single-threaded OpenBench SPRT gate structurally *cannot* detect by
   itself — still concrete evidence our concurrency-test investment earns
   its keep; just don't cite the first one as something that "shipped."
   **[already-planned]**
3. **Verify VNNI/dpbusd kernels actually emit the native instruction**, don't
   just gate a feature flag: Integral ships a `BUILD_VNNI512` tier whose dpbusd
   kernel never uses real `_mm512_dpbusd_epi32` (always the emulation ladder) — a
   "feature flag threaded through a whole build tier but never consumed" bug to
   test against when we add that tier. `design/11` §3.4.6 future AVX-512/VNNI.
   **S / lo (a test). [novel-cautionary]**
4. **Single-writer node counters beat Integral's atomic `fetch_add` per node** —
   Integral does the more expensive thing; our load-then-store single-writer
   scheme is already ahead. **[already-have]**
5. **Barrier-rendezvous orchestration** (one ~50-line cyclic `Barrier`, no
   per-thread addressability) is a third valid paradigm (vs our per-thread
   job-queue and Reckless's channel) — but **not adoptable**, since we have
   concrete near-term needs (parallel TT clear, `run_on_main` test hook) that
   require the addressability `Job::Custom` gives. `design/13` §3.4/§9.
   **[record only]**
6. **No voting / no jitter still earned "scales well"** — Integral's minimal set
   (lock-free TT + two atomic flags + a barrier) had the scaling reputation
   ~10 months before any NUMA code, and prints thread-0's result unconditionally
   (no vote, no aspiration/reduction jitter). Confirms voting/jitter are
   refinements, not prerequisites, for good lazy-SMP scaling. **[already-have]**

---

## P13 — NNUE (`design/11`, plus `15`/`16`/`04`)

The richest confirmation cluster in the study. Integral is a mature, wider net
(L1 1536, 12 king buckets, 8 output buckets) whose *architecture* independently
converges with Reckless and with our `design/11` upgrade path — three
implementations agreeing is strong evidence the shape is right.

### Activation / quantization / inference (post-v1 upgrade path)

1. **Split-halves pairwise-multiply activation** (clip both 768-lane halves of
   the accumulator to `[0,QA]`, multiply elementwise, `>>9`) instead of literal
   SCReLU-square. Now confirmed by **two** top engines with the *identical*
   `FT_SHIFT=9` (Reckless + Integral); strictly more expressive at the same
   one-multiply cost since the two halves carry independently-trained weights.
   `design/11` §3.4.3. **S-M / med. [novel] (Reckless-corroborated).**
2. **NNZ sparse-propagation table** (256-entry nonzero-lane-bitmask → set-bit
   lookup) to skip zeroed FT-output groups in the next layer's matmul. Now
   convergent across **three** studied engines (SF, Reckless, Integral). Cheap
   once a sparse int8 L1 exists. `design/11` §3.4 upgrade path. **S / lo-med.
   [novel] (three-way corroborated).**
3. **Int8-sparse-FT/L1 → f32-tail hybrid quantization** with dpbusd + NNZ
   sparsity, bias-as-initial-accumulator, mulhi activation — structurally
   identical to Reckless. Target this shape for our deeper post-v1 net instead of
   SF's all-integer cascade. `design/11` §3.4.6. **M / med. [already-planned
   (Reckless-sourced), Integral-corroborated].**
4. **Two-stage offline net processing: quantize, THEN lay out for SIMD** —
   Integral's standalone `preprocess/net_processing.cc` does pure
   transpose/permute (no numeric change) after quantization, keeping each step
   independently diffable. `design/11` §3.4.7 (netpack helper). **S-M / lo.
   [code-technique].**

### Buckets / accumulator (upgrade path)

5. **Uniform output-bucket formula `bucket = min((popcount(occ)−2)/4, 7)`, 8
   buckets** — an **exact** match to what `design/11` §3.4.1 already pins,
   upgrading it from "a formula we chose" to "shipped by a top-6 engine."
   Reckless's non-uniform 33-entry table is the documented alternative if we
   ever refine. **[already-planned (confirmed)].**
6. **Grow king-bucket count incrementally** (Integral 4→9→12 over ~5 weeks, each
   SPRT/bench-gated, finest granularity near the back rank first) rather than
   hand-designing a final N-bucket table. Integral's 12-bucket asymmetric layout
   is a *second* concrete precedent alongside Reckless's 10-bucket scheme.
   `design/11` §3.4.1 upgrade path. **M / med-hi. [already-planned (two
   precedents)].**
7. **Finny-table refresh cache** keyed by (mirror, king-bucket), diff current
   board vs cached snapshot, refresh only on a bucket/mirror crossing — third
   independent instance (SF/Reckless/Integral) of our `design/11` §3.4.4 plan.
   **[already-planned (confirmed)].**
8. **Eager, fully-materialized DirtyPiece record at move-make** (not Reckless's
   lazy `(mv,piece,captured)`-reconstruct-later) — Integral matches **our** plan,
   a second independent precedent for it. `design/11` §3.4.4. **[already-planned
   (confirmed)].**
9. **Skip the accumulator stack push/pop entirely on null moves** — nothing
   about active features changes, so no slot is needed. Free
   correctness-preserving simplification our current plan doesn't yet take (it
   implies pushing an empty dirty record). `design/11` §3.4.4 `do_null_move`.
   **S / lo. [novel].**
10. **(Speculative) SPARSE_PERMUTE**: profile-guided neuron reordering
    (bench-instrumented activation counting → offline stable-sort + weight/bias
    permutation). Novel research tool in neither SF nor Reckless; **no before/
    after numbers even in Integral**. `design/11` §3.4.1 upgrade path.
    **Flag as research idea only. [novel, speculative].**

### Datagen / training pipeline (`design/15`/`16`)

11. **Raise `P_cap` materially, or move to unfiltered binpack-style recording.**
    Integral's production path records *every ply unfiltered* and defers curation
    to the trainer; combined with our own `design/15` §10.4 measurement that
    `P_cap=10` discards 88.8% of already-paid-for labels, this is corroborating
    evidence from a shipped top-6 engine to fix before the bootstrap bulk run.
    `design/15` §3.5(6)/§10.4, `bin/datagen.rs`. **S / med (near-linear
    throughput). [already-planned (corroborated)].**
12. **DATAGEN-mode flag that disables search-only score adjustments** (rule-50
    damping, material-scaled optimism) when producing training labels — Integral
    hit and fixed exactly this footgun (`#ifndef DATAGEN`, PR #174, ~5.5 months
    into its NNUE era). Pre-empt it since our `design/11` §3.6 `evaluate()` has
    equivalent terms. `bin/datagen.rs`, `design/11` §3.6, `design/16` §3.4
    acceptance criteria. **S / med (footgun-prevention). [novel].**
13. **Pluggable `OutputFormatter`/`GameRecorder` trait seam from day one**, even
    with one format — Integral's tiny `SetPosition/PushMove/WriteOutcome`
    interface let it add a dense binpack exporter later without touching the
    self-play loop (exactly `design/15` §3.2.3's named fallback path).
    `design/15` §3.7. **S / lo. [code-technique].**
14. **Self-hosted `datagen` engine subcommand** (mirroring `bench`/`perft`) for
    Tier-B once our own engine is the teacher — Integral never leaves its binary
    to generate bulk data (threads + temp files + concat), simpler/faster than a
    scripting harness. Does *not* apply to Tier-A where SF is genuinely external.
    `CLAUDE.md` Commands, post-P13. **M / med. [code-technique].**
15. **Themed-codename + linear-version net naming, hosted as build-time-
    downloaded release artifacts** (Integral's `NETWORK_NAME` one-line CMake
    diffs auto-fetching from a separate `integral-networks` Releases repo) —
    directly answers our `design/00` P13 open question ("release assets vs git
    LFS"): recommend **release assets**. `design/11` §3.4.5. **S / lo. [novel
    (resolves open Q)].**
16. **If Syzygy is ever revisited, wire it as one generic always-on in-search
    probe** (rated play and datagen both inherit it), not a datagen-only special
    case — matches Integral's structure exactly. `post-P13, D31c`. **[code-
    technique, deferred] (Fathom itself conflicts-D14/D15 — opt-in feature only).**

### Input features / architecture calibration

17. **Threat/relational input features are a later, non-blocking lever, not a
    prerequisite.** Integral (top-6, beat SF in a TCEC game) ships **zero** threat
    features — a second data point (Reckless being the counter-direction)
    supporting base-net-first sequencing. `design/17` §3.5 (already lists this
    near-term-but-not-v1). **[already-planned (confirmed)].**
18. **Decouple "add depth" and "add width" into separate SPRT-gated steps**
    (Integral's L2/L3-tail commit transiently *shrank* L1 rather than growing
    width+depth at once) — keeps SPRT attribution clean, per our D25/one-change
    discipline. **[process].**
19. **Pure self-bootstrap (no foreign teacher) as a real D28-Tier-A fallback** —
    Integral's *only* teacher was ever its own HCE; concrete proof the SF-teacher
    shortcut is optional insurance, and a fallback worth keeping ready.
    `design/15` §3.1/§9, **D28**. **[process, already-planned].**

### Where our plan is already stricter (record)

20. **CENN net format (magic + version + checksum)** beats both Integral's and
    Reckless's raw, unvalidated `ifstream` struct read. Confirmed the right call
    given we ship an external `EvalFile`/net option. `design/11` §3.4.5.
    **[already-have (plan)].**

---

## Cross-cutting — tooling, build, attack tables

### Tooling (the dominant gap — full treatment in the success-factors note)

X-1. **SPSA tunable scaffold** (`cargo feature = "spsa"`: a `TUNABLE`/
`TUNABLE_STEP`-style macro over `search/params.rs`, zero-cost `const` in normal
builds, UCI-option + OpenBench-format print under the feature). Integral's ~126
tunables were shaped by 12,000+-iteration / 100,000-game SPSA runs; our
identically-shaped constants are all untuned SF starting points. Mechanical,
low-risk, build **before** the compute to feed it exists. `design/10` §9.1 /
`design/00` P14. **M / hi (unlocks tuning). [already-planned (build ahead of
need)].**

X-2. **OpenBench-shaped distributed testing** — the single highest-leverage item
in the whole study; owned by [00-success-factors.md](00-success-factors.md) #1.
`design/14` §3.8. **[novel infra, P14].**

### Attack tables (`design/03`)

X-3. **If a PEXT variant is ever explored, gate it on CPU-family exclusion**
(Zen1/2 have slow microcoded `pext`/`pdep`), not on `target_feature="bmi2"`
alone — our D16 x86-64-v3 build implies BMI2 presence but Zen1/2 support it only
*slowly*, so a naive feature-flag gate would silently regress there. Integral's
`CMakeLists.txt` writes 45 lines of CMake to route around exactly this
(Zen1/2 disabled, Zen3/4+Intel enabled, a hardcoded "AMD EPYC 7B" carve-out for
its own OpenBench fleet) — a shipped, working confirmation of the concern that
**already justified our D12 no-PEXT decision.** `design/03` §3.8 (append caveat),
**D12/D16**. **S / lo (doc caveat). [novel — confirms D12].**

---

## Explicitly declined / recorded as prior art (no action)

- **Copy-full-`BoardState` undo** (a `List<BoardState,2048>` history stack, zero
  unmake-reversal logic — "undo cannot be wrong by construction"). The
  polar-opposite of our D8 incremental StateInfo stack; trades ~600KB+/thread
  (worse under P11 lazy SMP) for zero unmake bugs. Considered, **declined** —
  affirms **D8**. **[record only]**
- **Plain/dense magics** (`[[BB;512];64]` + `[[BB;4096];64]`, ~2.25 MB) vs our
  fancy compact tables (~841 KB). A real but performance-immaterial divergence on
  modern hardware; we keep fancy. **[record only]**
- **Move encoding with the two 2-bit fields in the opposite order** — cosmetic,
  no impact; affirms **D2**. **[record only]**
- **Growable (`head_idx` cursor into a `Vec`) accumulator stack** vs our + our
  planned fixed `MAX_PLY` array — minor; no reason to change. **[record only]**
- **Genuine atomic `fetch_add` per node** for counters — Integral's more
  expensive choice; our single-writer scheme is ahead. **[record only]**

## Where we're already ahead (recorded so we don't "fix" it)

- **Legality-exact FEN ep validation (D9)** vs Integral's unconditional,
  unvalidated ep recording (`fen.cc`) — a smaller robustness margin against
  fuzzed/adversarial FEN.
- **EVASIONS generation type** (ANDs targets with `between(king, checker)` at
  generation time) vs Integral generating every non-king piece's full target set
  and filtering afterward on single check — we (and SF) are strictly more
  efficient here.
- **Per-node invariant oracle** (`assert_consistent`, all six Zobrist keys
  recomputed) at every make/unmake in debug — Integral has only perft/SEE/bench
  suites and zero threading-specific or sanitizer builds.
- **Validated CENN net format**, **conservative draw adjudication + committed
  provenance metadata** (`design/15` §3.6 `meta.json`), and the **persistent
  design/decision layer + D33 review loop** (our substitute for a solo human's
  continuous memory and free self-review) — none of which Integral has.
