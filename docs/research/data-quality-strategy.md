# ARCHON data-quality strategy — two dimensions (2026-07-18, user framework)

The user's decomposition, which supersedes the ad-hoc "more/better data" framing:

- **Dimension 1 — EVAL quality:** the accuracy of the value LABEL on a position.
- **Dimension 2 — POSITION quality:** the coverage/relevance of the positions
  themselves.

Governing principles (user, 2026-07-18):
1. **Never throw positions away.** We have a lot of positions; a low-value
   position with a GOOD eval label is still net-positive. No position pruning.
   We improve labels IN PLACE and ADD new positions — the corpus only grows.
2. **Improve eval quality by relabeling existing positions** with our strongest
   evaluator (SF18 today; ARCHON eventually).
3. **Resources are limited → target the relabel.** Prioritize the positions
   where the strong evaluator DISAGREES most with the existing label — relabel
   those first. (This is active-learning / importance-sampling: spend deep-label
   compute where current labels are most wrong = maximum label-quality gain per
   compute.)
4. **Position quality comes from strong-engine games at deep search**
   (SF18 v SF18, SF18 v Reckless, SF18 v ARCHON, Reckless v ARCHON) — these
   reach positions our net doesn't reflect.
5. **ARCHON-included games do double duty:** they add new positions AND expose
   *why ARCHON loses* — the positions ARCHON mis-evaluates are its blind spots,
   the highest-value training targets (learn from your losses).

## How our current work aligns

- The **label-value pilot** (running) is exactly Dimension 1: does relabeling
  existing positions with deeper (SF18 @100k) labels make a stronger net? Its
  pre-gate already MAPS the disagreement: on the training distribution, cheap
  vs deep labels are Pearson 0.959 / RMS 42cp / 15% sign-flips, and the gap is
  CONCENTRATED in decisive positions (RMS 26cp balanced → 136cp in >400cp). So
  we already know *where* the label error lives — principle 3's targeting data
  is in hand.
- We have the **machinery** for Dimension 2 already: fastchess for engine games,
  SF18/Reckless/ARCHON binaries, the Reckless-net instrument, obs-capture for
  in-game position extraction, and the SF18 relabel tooling.
- The gap vs the user's framework: we've done UNIFORM relabeling (all positions)
  and OUR-engine self-play datagen — not TARGETED relabeling (disagreement-first)
  nor STRONG-engine game generation with the learn-from-losses signal. This
  strategy sharpens both.

## Dimension 1 plan — TARGETED eval-quality relabeling (do FIRST; cheaper)

Rationale: relabeling existing positions costs no game-play, and we have a huge
corpus. Coarse-to-fine to spend deep search only where it pays.

- **Gate (running):** the label-value pilot. If deep-label net beats cheap-label
  net (≥+15 fixed-node), the whole relabel program is validated. If null, labels
  aren't the bottleneck — stop here, redirect to Dimension 2 + search.
- **Step 1 — disagreement scoring (cheap):** one MODERATE SF18 pass (~25k nodes,
  ~5× faster than 100k) over a corpus subset; score each position by
  `|SF18_moderate − existing_label|`. This is the label-error estimate. The
  calibration already says decisive positions dominate, so `|existing_cp|` is a
  free coarse prior — start there, refine with the moderate pass.
- **Step 2 — targeted deep relabel:** relabel ONLY the top-disagreement subset at
  deep budget (100k–250k). Keep the low-disagreement majority at their existing
  (cheap-but-adequate) labels — they're fine (Pearson 0.98 on balanced
  positions). This concentrates the expensive search where labels are wrong.
- **Step 3 — retrain + gate** the corpus with the improved labels vs the current
  net (fixed-node then equal-time).

## Dimension 2 plan — POSITION quality via strong-engine games (do SECOND)

Rationale: adds genuinely new coverage; more expensive (deep games). The
ARCHON-loss subset is the highest-value part.

- **Step 4 — strong-engine game generation** at a deep TC (or high fixed nodes):
  SF18 v SF18, SF18 v Reckless, **SF18 v ARCHON**, Reckless v ARCHON. Diverse
  openings (UHO). Extract non-book quiet positions from the PGNs.
- **Step 5 — the learn-from-losses signal (the key part):** in the ARCHON games,
  compare ARCHON's in-game eval to SF18-deep on the same positions; the large
  ARCHON-error positions (especially where ARCHON was losing) are its blind
  spots — weight these heavily as training targets.
- **Step 6 — label the new positions** with SF18-deep (they're new → need labels;
  this reuses Dimension 1's relabel machinery) and ADD to the corpus (never
  replace). Retrain + gate.

## Resource prioritization + the honest constraint

- Order: **Dimension 1 targeted relabel FIRST** (cheaper per position, huge
  existing corpus), **Dimension 2 games SECOND** (focused on ARCHON-loss
  positions).
- Both are compute-heavy AT SCALE. On one box we run PILOTS + targeted subsets
  (which is what the label-value pilot is). The FULL program — relabel millions
  of high-disagreement positions + generate millions of strong-game positions +
  deep-label them — is the **D36 distributed-compute investment**. The user's
  targeting is what makes it affordable: relabel/generate only where it helps,
  not blindly. This is precisely the right way to spend eval compute when we have
  it, and the efficiency case for standing up D36.
- Self-improvement endgame: once ARCHON is strong, ARCHON's OWN
  disagreement-with-labels guides relabeling and ARCHON-loss games guide
  position generation — the AlphaZero-style flywheel, bootstrapped by SF18/Reckless.

## VERIFIED (2026-07-18): the obs schema ALREADY captures the superset

Confirmed in `src/obs/schema.rs` — no new schema needed, only route relabeling
through it (answers "capture now what future helm/beyond nets need"):
- **`IdSnapshot`** (in `SearchObservations.id_snapshots`) = the eval TRAJECTORY:
  `{depth, score, best, nodes, time_ms, pv}` at EVERY iterative-deepening depth —
  the #1 capture-now / helm-needs-later signal, already in the schema, ~free.
- **`PerMoveObservation`** = per-move `{returned_score, nodes, searched_depth,
  order_index, pruned_by, ...}` — policy/move-ordering + helm-effort target.
- **`root-analysis`** emits one Observation PER TIER (node_budget) — multi-budget
  per-move capture is native (tiers configurable).
- **`Provenance`** = full versioning `{engine_version, engine_bench_sig,
  net_version, teacher_identity(rev+binary sha256), search_config+hash,
  generator_version, rng_seed, dataset_source, opening_source, hardware_identity}`
  — the versioned/additive-label requirement, fully covered.

**THE ONLY GAP:** the relabel PILOT path (`relabel_depth.py`) bypasses all of it
(python-chess `analyse` → scalar cp). The PRODUCTION relabel must route through
`root-analysis`/obs so trajectory + per-move + provenance ride along for free;
the scalar-cp corpus stays a cheap EXPORT VIEW (`obs-tool export-train`). So the
forward-looking data plan is de-risked: the machinery to "capture the superset
once" exists — we just stop using the scalar shortcut for production relabeling.

## Value-tiered capture (recommended default, pending user steer)
- EVERY deep search → the free layers (trajectory + PV + metadata + provenance).
- HIGH-VALUE positions (high-disagreement, game-derived, ARCHON-loss) → also the
  rich per-move analysis + effort (searchmoves, ~35x cost).
- Bulk low-disagreement → free layers only.
- Far-future (counterfactual/intervention search traces, for true search-augmentation):
  keep the obs hooks, sample a thin slice, do NOT scale on one box.
