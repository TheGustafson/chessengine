# Finetuning question for an expert (2026-07-20)

A domain-general statement of a problem we hit, for a finetuning/ML expert. The
domain is a chess evaluation network but nothing below depends on chess.

## Setup
A small, quantized MLP that regresses one scalar value. A very large sparse first
layer (embedding-style feature transformer, tens of millions of params) → two small
dense layers → scalar output; int8/int16-quantized for fast inference. Trained
**from scratch** on ~**billions** of (input, target) examples with **AdamW
(weight_decay = 0.01)** and a **cosine LR schedule from 1e-3 decaying to ~2.7e-5**
(peak × 0.027), run to convergence. It is a strong, well-calibrated network.

## Load-bearing constraint
The output's *absolute magnitude* matters, not just its ranking — a downstream
decision procedure has thresholds calibrated in the network's output units. So we
care about the whole output distribution, not just relative accuracy.

## Goal
Add coverage of a small set of "hard" cases (a few million examples the deployed net
handles poorly, mixed with in-distribution retention data) by **finetuning** the
converged net — not retraining from scratch on billions.

## The core finding (the puzzle)
Sweeping the finetune peak-LR (same cosine-to-near-zero schedule, ~3 epochs,
warm-started from the converged net), measuring end-task strength vs the untouched
base (Δ in a log-odds win-rate metric, ±~22 noise):

| finetune peak-LR | end-task Δ vs base | output scale (vs fixed reference) |
|---|---|---|
| 1e-6 | 0.0  | 0.962× |
| 3e-6 | +1.7 | 0.959× |
| 1e-5 | −12  | 0.953× |
| 3e-5 | −40  | 0.950× |
| 5e-5 | −37  | 0.946× |

**There is no LR that improves the net.** Below the base's own *final* LR (~2.7e-5)
it is unchanged (no learning); above it, it monotonically degrades.

## Two hypotheses we tested and REFUTED (measured, not guessed)
1. **AdamW weight decay shrinking the weights.** We exposed the knob and reran with
   `weight_decay = 0`. Result: **identical** to 0.01 at both a gentle schedule
   (−34.9 vs −36.6) and an aggressive one (−212.4 vs −209.9). Weight decay is not
   the cause. (Consistent with the arithmetic: over a 3-epoch finetune,
   `w·(1−0.01·LR)` accumulates to only ~0.23%.)
2. **Output-scale compression decalibrating the downstream.** REFUTED by an
   anti-correlation: the *gentle* net has a *compressed* scale (0.946×) but the
   *better* Elo (−35); the *aggressive* net (10 epochs) has a *near-perfect* scale
   (0.993×) but the *worse* Elo (−212). Scale and play strength move in **opposite**
   directions — restoring the scale made play worse.

## What actually seems to be happening
The regression tracks the **amount of finetuning** (LR × epochs), i.e. classic
**overfitting the small set / forgetting the billions** — and, strikingly, it is
**invisible in static-eval metrics**: the aggressive net looks excellent statically
(correlation 0.966 to a strong reference, near-perfect scale 0.993×, very few large
errors) yet plays ~212 Elo worse. Static accuracy does not predict end-task quality
here; end-task quality is governed by behavior on the (different) distribution of
inputs the downstream search actually visits.

## Questions for a finetuning expert
1. Is this expected — is small-set finetuning of a from-scratch-converged regressor
   simply the wrong tool for adding a few million targeted corrections, vs. folding
   the new examples into a large mixed stream and re-running a full-scale schedule?
2. If finetuning *can* work here, what makes it work: replay ratio (retention vs
   new), freezing the output layer / adapting only some layers, explicit
   regularization toward the base weights (EWC / L2-to-init), warmup, a *tiny* LR
   with *many more* epochs, or something else?
3. How do you *measure* finetune quality cheaply when static accuracy metrics
   mislead (good static signature, bad end-task)? Any proxy that predicts the
   deployed behavior short of a full end-task eval?
4. Is "the base finished at LR ≈ 2.7e-5, so any finetune LR below does nothing and
   any above degrades" a fundamental tension, or a symptom of an over-converged base
   / too-small a retention set?
