# What is holding ARCHON back from SF16? — a decomposition study

Date: 2026-07-18. Method: decompose the ~63–78 Elo equal-time gap to SF16 into
eval / search / speed / compute, each backed by our own experiment ledger.
Two parallel evidence sweeps (search-side, eval-side) + the anchor triangle.
Citations: `exp-NNNN` (ledger), `file:line`, VFS = value-foundation-state.md.

## The gap, calibrated

ARCHON vs SF16 at equal time (100g, 10+0.1): **−78 ± 18** (2026-07-16 anchor),
**−63 ± 29** on the post-S1 rematch (exp-0091, not a significant move). vs SF18
−191 ± 23, vs Integral-v6 −70 ± 22. The measured SF16↔SF18 gap (113 ± 29) ≈
the known ~90 Elo, so the anchors are roughly calibrated — we are genuinely
~63–78 behind SF16, needing ~+95–110 to reach SF16 +30.

## Finding 1 — Eval is NOT the primary problem (it's competitive with SF16)

- vs **full-strength SF16**, ARCHON scores **39% with a 70% draw rate, 0 time
  losses** (exp-0060) — "the eval is competitive; the deficit is now plausibly
  search-era as much as eval" (VFS:743-750). A quality-capped eval does not draw
  70% against SF16.
- The D40 arch (HalfKAv2_hm + FullThreats + Reckless head) **wins per node**
  vs our own classic-2048 incumbent (+30.2 ± 17.4, exp-0067) and reached
  **equal-time parity** (−8.7 ± 12.9, CI straddles zero, exp-0093) — the recovery
  came entirely from the speed program (PGO +10.5%), not any quality change. So
  D40 is **NPS-limited, not quality-limited**: there is real eval headroom
  sitting *behind* inference speed, not behind a quality wall.
- Width saturated at 2048 (4096 only +15, exp-0058) — the arch platform is set.

Capturable eval headroom on one box is therefore **modest and concrete**
(~30–40 fixed-node), ranked:
1. **Bank the D40 per-node edge at equal time** via longer stage-1 (720sb beats
   240sb by +30.2/node, exp-0072; curve not flat; official runs 40–60B
   presentations vs our 17.9B). Now capturable since D40 is at speed parity.
2. **King-bucket factorization (#27)** — we train the 32-bucket HalfKA tensor
   **UNfactorized** (TMS:80, G3:155-172): every (piece,square) weight is learned
   32× with no gradient sharing. Official nnue-pytorch / Integral / Viridithas
   all factorize. Expected +10–30 fixed-node, **zero inference cost**,
   one-box-verifiable. The single most under-exploited eval lever.
3. Stage-2 farseer — SPECULATIVE for D40: classic banked +84 across two hops,
   D40 has banked ≤0 (exp-0066 −23.2, diag1 exp-0094 −12.7; the LR-5e-4
   warm-from-strong cell, diag2, is being screened now). If diag2 also ≤0,
   stage-2 is dry for the threat arch.

## Finding 2 — Search IS measurably behind, but the gap is un-localizable on one box

- **The decisive number (design/14 §3.11, D39):** ARCHON's search is
  **−86.68 ± 22.97 Elo/node behind Reckless, same net both sides, fixed 100k
  nodes.** Eval and speed controlled — this is a pure per-node search-quality
  deficit vs a top-2 Rust engine (~3833 SPCC, ≈ SF16 class). Real,
  second-order, "a tuning-maturity gap deferred behind the net."
- Yet the **machinery is essentially all present** (search agent, confirmed in
  `src/search/mod.rs`): singular/double/triple extensions (354-414), multicut
  (2942), negative extensions (2323), ProbCut (2509-2537), do_deeper/do_shallower
  (3066-3111), cutoff_cnt + cut_node (1978/2023/2202), the full correction-history
  stack (S1, 121-127), RFP/NMP/LMR/LMP/SEE + butterfly/capture/continuation/pawn
  histories. Only small slices are missing: IIR-narrow (0–8, parked), optimism
  (+5–15, flat in SPSA c7), the SF16-late mechanics bundle (+5–15, unaudited),
  nodesEffort TM. The SF14→16 port list is "mostly DONE" (06-study §6a).
- **So the −87/node is in the CONSTANTS, not the algorithms** — and one box
  cannot mine it: the D45 SPSA season shipped **8/8 nothing**, and a **+50%
  detune of the base LMR reduction costs only −8.1 ± 12.9** (exp-0078). The
  search-param landscape is a high plateau; re-search safety nets (PVS,
  do_deeper/shallower, TT) absorb gross mistuning. Sub-10-Elo per-param mining
  is fishtest-scale (~48k games per +2 Elo, design/14 §3.8).

## Finding 3 — The measurement/deflation wall makes local gains non-transferable

- **Internal gains deflate 2–3× externally**, self-play-tuned worst. S1
  correction history: **+72.9 ± 16.2 internal** (exp-0074) → the SF16 anchor
  moved ≤+15, not significant (exp-0091). The internal ladder is **+265 over
  net-0** yet we are still −63 to −78 vs SF16.
- One box resolves ~**±13 Elo** at its practical budget; 100g "proves nothing
  inside ±30" (design/14 §3.8). We cannot verify a sub-30-Elo fix against SF16
  even if we make one.

## The diagnosis (ranked)

The engine is **architecturally complete** — SF-class search machinery + an eval
that is *competitive* with SF16 (70% draws). It is not held back by a missing
algorithm or a broken component. The ~63–78 Elo is a blend, in order of size:

1. **A real but un-localizable search per-node deficit (~87/node vs Reckless),
   which is a tuning-maturity gap requiring fishtest-scale compute to close.**
   The Elo lives in hundreds of SF-default constants (`params.rs:233-240`) never
   tuned for our tree, and one-box SPSA has hit its resolution floor (exp-0078).
2. **The compute/measurement wall (D36) as the binding meta-constraint.** Both
   the search fine-tuning AND the remaining eval refinements (SF18-relabeled data
   at official presentation volume, factorization verification, data selection)
   need testing throughput we do not have locally, and internal gains can't be
   verified against SF16 at one-box game counts.
3. **A modest, real, one-box-capturable eval headroom (~30–40 fixed-node,
   less after 2–3× external deflation)** — banking D40's per-node edge via longer
   stage-1 + king-bucket factorization.
4. **Speed is largely banked** (PGO; D40 at equal-time parity) — no longer the
   constraint.

**Bottom line:** ARCHON is a complete, SF-class engine that is *under-tuned and
under-scaled* relative to a target with 10 years of fishtest. One box, on its
own, most likely **cannot** close +100 to reach SF16+30: the two largest levers
(search SPSA at scale, data at official volume+quality) are both gated on the
D36 distributed-compute decision. Eval-quality is not the wall — it is
competitive; the wall is testing/tuning throughput plus external deflation.

## The decisive confirming experiment (built, ready to run)

The **firewalled Reckless-net instrument** (rebased today, compiles clean):
run ARCHON-search + Reckless-net vs Reckless itself at **fixed nodes**.
- If ≈ Reckless → the −87/node was a stale/eval artifact and our search is
  actually sound → the gap is eval + compute + deflation (Findings 1&3 dominate).
- If ARCHON+Reckless-net reproduces ~−87/node → the search deficit is **real and
  isolated**, and the instrument becomes a one-box-verifiable, deflation-free
  test bed for search changes (measure fixes against a top engine with eval
  controlled, at fixed nodes) — the single most valuable local search tool we
  could have, and the one scenario with a large one-box win.
This is the highest-value next measurement and should run as soon as the box is
free.

## Recommendation

- **Now, one-box (positive-EV, verifiable):** (1) the Reckless-net A/B — it
  decides whether search is a real local lever; (2) longer D40 stage-1 to bank
  the per-node eval edge at equal time; (3) king-bucket factorization (#27).
- **Strategic unlock:** stand up the D36 GCE-spot OpenBench workers. The
  convergent conclusion of both evidence sweeps is that the path to SF16 runs
  through **testing throughput** — it is the only thing that converts the
  un-minable sub-10-Elo search ladder and the un-verifiable eval gains into
  external Elo. Without it, +30-over-SF16 is likely out of reach on this box.
- **Honesty guardrail:** stop banking internal Elo into external projections
  (exp-0091). Track progress against the SF16 anchor, not the net-0 ladder.
