# EPIC: SF18-Canonical Training Foundation

**Owner:** user + agent · **Created:** 2026-07-19 · **Status:** codified, ready to run

A clean rebuild of ARCHON's training foundation around a single canonical,
SF18-labeled dataset — the substrate that lets a better eval + a retuned search +
hard-negative and self-play data compound into real Elo. This supersedes the
patch-the-incumbent (finetune) approach, which the evidence proved is the wrong
vehicle (see §Evidence).

**North star (unchanged):** beat SF16 by +30 Elo at equal time (10+0.1). Current
gap ≈ −41.5. This epic is the vehicle to close it and keep going.

---

## Why this plan (the evidence that shaped it)

1. **B4 — volume, not cleverness, was the constraint.** The nets that beat net-0
   trained on the official SF binpacks at 3–9B presentations. Data quantity +
   quality of a real teacher is the lever.
2. **Label DEPTH is a MODEST lever.** SF18@50k vs the 5k labels agree on the
   *winner* 99.8% of the time (r=0.93); they differ in magnitude/calibration
   (mean |cp| 515→189), not verdict. Don't expect big Elo from label depth alone.
3. **Eval and search are CO-ADAPTED.** Finetuning a recalibrated eval onto a
   search tuned for the old calibration degrades play (ft50k gated −196 not
   because the labels are bad, but because cooling the eval broke the search's
   cp-denominated margins). Diagnosed directly this session.
4. **The finetune MACHINERY works** — continuing the converged incumbent on its
   OWN data at LR 1e-5 gated ≈0 (+15±34). So the failures were *confounds we
   control* (scale bug, subset-overfit, distribution drift), not a dead end.
5. **Scale is a real trap.** Fleet cp = exact raw SF18 UCI cp (verified, ratio
   1.000); the binpack needs ÷1.9 to reach the same units. Mixing scales silently
   wrecks training. → **unify on one scale.**
6. **Warm relabel is cheap.** warm@20k = 4.0× faster than cold@50k, labels median
   Δ14cp, 0 winner-flips → the full relabel is ~$20 / ~18h at the permanent 96-vCPU
   cap.

## The five principles (non-negotiable)

1. **RETRAIN, don't finetune.** A natively-SF18-calibrated eval from scratch, not
   a patch onto the incumbent. This dissolves the co-adaptation wall.
2. **One scale.** Everything on SF18 UCI-cp. Kill the binpack ÷1.9 mixing. One
   dataset, one target convention.
3. **Pair the eval with a search retune.** Every new eval gets a search SPSA pass
   BEFORE it's judged. Never gate a new eval against the old search's tuning.
4. **Two axes.** Label quality (SF18) is the base; POSITION VALUE (Archon-loss
   hard negatives, then self-play) is where the big Elo lives. Weight effort there.
5. **Gate on real games, on large-corpus nets.** Sanity fens don't catch a broken
   net (the −636 passed them). Fixed-node for isolation, equal-time for the ship gate.

---

## Phases

### Phase A — Build the canonical dataset (the full relabel)
- **warm@10k SF18 relabel of EVERYTHING we have, EXCLUDING the ~20M c1 already did**
  (c1 stays at its 50k cold labels). Corpus = UHO(188.5M)+wrongIsRight(~32M)+
  farseer(~30M) − c1 ≈ **230M new + 20M c1 = ~250M canonical positions**.
- **Game-order** (native binpack order, no stratify) so the warm TT reuses trees;
  history-dependent labels **accepted** (speed > purity, decided).
- **Config finalized by the Fable relabel-speed study** (`docs/research/relabel-speed-study-2026-07.md`,
  2026-07-19), which measured every lever:
  - **warm@10k, not 20k** — 1.81× faster, labels statistically identical
    (dq 0.0097 vs 0.0101; 1 flip/988). The extra nodes bought nothing. Config-only.
  - **SF18 rebuilt `make profile-build ARCH=x86-64-bmi2`** (bmi2+PGO) — +6.7% NPS,
    labels **bit-identical** (r=1.0000). Ready binaries staged; the image rebuild
    was already required for `--warm`, so re-pinning the sha is zero marginal ops.
  - **Hash=16, Threads=1, 1 proc/vCPU, deduped manifest** — all measured optimal;
    bigger hash / more threads / cheap-screen cascade / incremental feed all
    measured *worse or null* and rejected.
  - **Syzygy ≤7-man split (18.6% of corpus): +1.2× — DEFERRED to a Phase-B TB
    label-semantics decision.** A c3d/Genoa vnni512-pgo spot probe (~$0.20) may add ~0.9×.
- Prereqs: (a) rebuild+push the worker image with warm `root_analysis` **+ the
  bmi2-PGO SF18 binary**; (b) FEN manifest that excludes c1's FENs, game-ordered +
  deduped; (c) game-ordered partition.
- **Cost ~$10.5, ~9.5h** at 96 vCPU (permanent cap) — 1.9× better than the original
  warm@20k plan (5.35× over cold@50k). Aggressive fallback warm@5k → ~5h/$5.5 at
  dq 0.0114 (still 32% cleaner than the binpack-native labels the winning nets trained on).
- **Gate A:** coverage + label sanity + scale consistency (all SF18 UCI-cp);
  spot-check vs live SF18 (ratio 1.000).

### Phase B — Rewrite the training pipeline for the canonical set
- Unify format + scale: the canonical corpus as ONE source on the SF18 UCI-cp
  scale (mult=1, scale=400, NO ÷1.9). WDL = game result.
- Rewrite the run config/recipe: from-scratch (or the proven multi-stage) tuned
  for SF18 labels — arch, LR schedule, WDL weight, superbatch budget.
- Fold in `train_run.py`'s corpus→train→verify→pack→sanity→sprt→ledger flow.
- **Gate B:** the new pipeline, from scratch on the canonical data, REPRODUCES or
  exceeds the incumbent's strength — proves the rewrite + scale are sound before
  we build on it.

### Phase C — Eval retrain + PAIRED search retune
- Retrain the shipped eval from scratch on the canonical SF18 dataset.
- **SPSA-retune the search** (RFP/futility/LMR/etc. margins) for the new eval's
  calibration — the co-adaptation fix.
- **Gate C:** (eval + retuned search) vs incumbent at EQUAL TIME; and vs SF16
  (the north-star anchor). This is the first real "did the canonical base help" read.

### Phase D — Incorporate Archon-vs-SF18 position-value data
- Scale the mining (pipeline built: `mine_archon_losses.sh` + `pgn_mine_losses.py`
  + parallel `root-analysis` deep-label): bulk ARCHON-vs-SF18 games → mine
  losing/blind-spot positions → SF18-label them → fold into the canonical set,
  **upsampled** (the hard negatives where our eval is provably wrong).
- Retrain + re-retune search.
- **Gate D:** with-hard-negatives vs without, equal time. Expect the bigger jump here.

### Phase E — Self-play datagen (future / the compounding engine)
- AlphaZero-style loop: best net generates self-play games → label (warm SF18 or
  the net itself) → retrain → iterate. The long-run Elo engine; the SF18 relabel
  is the bridge to this.

---

## Honest Elo expectation

Label depth alone: **modest** (99.8% winner-agreement). The big Elo, if it comes,
comes from **Phase C's clean rebuild + search retune** (finally letting a better
eval express itself), **Phase D's hard negatives**, and **Phase E's self-play** —
not from "SF18 labels are magic." Attribute wins honestly by phase gate.

## Risks + mitigations
- **Co-adaptation** → always pair eval + search retune (Principle 3).
- **Scale bugs** → one scale, verified against live SF18 (Principle 2 + Gate A).
- **Overfit / weak from-scratch nets** → full canonical corpus, proven schedule;
  Gate B must reproduce the incumbent before trusting downstream gates.
- **Modest label lever underdelivers** → the plan's value is front-loaded onto
  position-value (D/E), not label depth; if C is null, that's expected, push to D.
- **96-vCPU cap** → permanent; warm@20k already makes the relabel sub-day, and
  local training/self-play isn't quota-bound.

## Open decisions (resolve as we go)
- Arch for the canonical retrain (keep D40 HalfKA+threats-768x2 vs revisit).
- From-scratch vs multi-stage (stage-1 broad → stage-2 curated) for the canonical set.
- Search-retune scope (which params, what book, SPSA budget).
- Self-play label source in Phase E (warm SF18 teacher vs self-distillation).
- Whether to keep the old binpack 5k data at all once canonical exists (probably drop).

## Status pointers
- Warm mode: implemented (`--warm`) + validated (4× / Δ14cp / 0 flips).
- Finetune campaign (task #39): recast as a DIAGNOSTIC ("does SF18 data help, done
  right"), not the product. Finish for signal; the product is Phase C.
- Live relabel: c1 (~20M @50k cold) finishing; c2 held; the full warm@20k relabel
  (Phase A) replaces the c2-blanket plan.
- Live research log: `docs/research/value-foundation-state.md`. Data strategy:
  `docs/research/data-quality-curriculum.md`.
