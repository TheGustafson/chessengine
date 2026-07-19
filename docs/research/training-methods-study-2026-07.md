> **★ CLOSING VERDICT (2026-07-19): the one-box training-method well is DRY.**
> Every ranked experiment ran and none cleared the +15 fixed-node gate:
> - X1 checkpoint/seed harvest → NULL (exp-0084: the final ckpt IS the best of a
>   run; intra-run harvest adds nothing; multi-seed selection left as a future
>   GPU-budget question, not a method gain).
> - X2 loss exponent 2.5 → NEUTRAL **+5.0 ± 9.7** (exp-0077, 900g).
> - X3 king-bucket factorization → WEAK **+6.9, CI [−29,+43]** (exp-0099), below
>   gate — implemented + parity-verified (`D40_FACTORIZE`), banked-not-shipped.
> - Stage-2 warm-retrain → **DEAD for the D40 threat arch** (exp-0095 −72.9;
>   3-cell verdict); width → **saturated at 2048**; LR-shape / warmup / constant-WDL
>   / batch-for-strength / optimizer → CLOSED (§D).
>
> So the ONLY eval levers left on one box are (a) **longer stage-1** (same UHO
> distribution — a DATA-scale fact, +30/node proven but diminishing) and
> (b) **better data** (SF18 relabel — the running label-value pilot — or the
> fleet). This is the same wall the **SF16 gap diagnosis**
> (`sf16-gap-diagnosis-2026-07-19.md`) hits from the game side: the −41.5 is a
> fine-grained per-node eval deficit (0.2% blindspot rate, +1 ply depth), not a
> method bug. **Two independent lines → the fleet + beyond-SF16 data is the
> transformational lever; every one-box method knob is spent.** Task #27 is
> answered: nothing material is being left on the table by method choice.
>
> **Reconciliation note (added at review, 2026-07-17 09:00):** Experiment #4
> ("quant-tail robustness") was drafted before the same-day packing-tail
> diagnosis (docs/research/d40-packing-tail-2026-07-17.md) landed. That
> diagnosis PROVED: no tensor overflows its quant range in any D40 net;
> rescale_net.py is impotent against the tail (integer-vs-float arithmetic
> divergence, not quantization); the parity gate was recalibrated to
> distribution bounds and d40-uho-720sb is already admitted to gating.
> Experiment #4's diagnosis step is therefore DONE and its tighter-clip
> variant is demoted to low priority. Experiments #1 (checkpoint
> harvesting) and #2 (loss power 2.5) launched together at 09:00 as
> d40-uho-p25-240sb — an exact twin A/B against d40-uho-v1-240sb.

# Training-methods study — are we leaving Elo on the table? (2026-07-17)

Status: research deliverable for review. NOT committed. No code changed.
Scope: the NNUE **value-net training method** — hyperparameters, loss,
schedules, filters, selection discipline — for the D40-class full-feature
nets (HalfKAv2_hm + FullThreats + multi-layer head, the strategic
architecture) and the classic-net line, trained with bullet on the official
Stockfish binpacks.

Method: (1) extracted every effective hyperparameter from our actual
trainers and run configs (file:line cited); (2) web-researched the official
nnue-pytorch recipe (source files, PRs, wiki), bullet-community practice
(bullet source/docs/examples, top-engine repos, Cosmo's ablation), and
public NNUE ablations — three parallel research passes, each claim
URL-cited with a confidence tag; (3) cross-checked everything against our
own empirical record in `docs/research/value-foundation-state.md` so we do
not re-propose experiments we already ran flat.

Governing context (do not lose this):
- **Our own B4 answer stands: data volume was the dominant constraint.**
  Nothing below claims otherwise. These are second-order method levers to
  harvest AFTER the data lever, sized honestly.
- **Promotion bar is equal-time vs the incumbent.** D40 currently wins at
  fixed nodes (+32.5 ± 17.3 stage-1-only, exp-0065; 720sb adds +30.2,
  exp-0072) but LOST the equal-time gate (−57.9 ± 12.6, exp-0073) on
  inference tax. Training-method gains on D40 bank at fixed nodes until the
  NPS work (R2 VNNI etc.) pays the tax. That does not make them worthless —
  it makes them stockpiled — but rank-order any "ship now" reasoning
  through the classic line.

---

## A. Effective-hyperparameter table: US vs official nnue-pytorch vs bullet community

"US (D40)" = the production D40 recipe as actually run
(`tools/nnue/bullet/d40_features.rs` + run env in
`data/nnue/experiments/d40-long-v1/config.json` and
`tools/nnue/runs/d40-uho-v1-240sb/run.json`). "US (classic)" noted where it
differs (`tools/nnue/train_net0.rs`). Official = current
official-stockfish/nnue-pytorch master + shipping-net PRs. Community =
bullet norms per bullet source/docs/examples + top-engine repos.

**DIVERGE** = material difference worth attention. (match) = fine.

| Knob | US (D40 production) | Official nnue-pytorch | bullet community | Verdict |
|---|---|---|---|---|
| Trainer | bullet, low-level `ModelDefinition` path (d40_features.rs:604) | nnue-pytorch (Lightning) | bullet high-level `ValueTrainerBuilder` | (match — bullet is the top-engine norm) |
| Optimizer | AdamW: β1 0.9, β2 0.999, decay 0.01, weight clip ±1.98 (bullet `adam.rs:31` default; d40_features.rs:686) | config-driven wrapper; historically Ranger-family; **per-tensor weight-decay groups** (`ft_weight_decay`, `dense_weight_decay`, `factorized_weight_decay` in `model/lightning_module.py`) | AdamW 0.9/0.999/0.01/±1.98 — every top bullet engine | (match community; see D-list re: optimizer swap) |
| Gradient clip | none (weight clip only) | not observed in lightning_module excerpt (unverified) | none exposed | (match; ignore) |
| LR (stage 1) | 1e-3 (`D40_LR`) | 8.75e-4 | 1e-3 | (match) |
| LR schedule | cosine 1e-3 → 2.7e-5 over the run (d40_features.rs:691-698, final = lr·0.3³) | StepLR-style gamma 0.992 **per 100M-position epoch** (≈ exponential) | cosine to ~1e-5..8e-6; Cosmo: schedules "essentially indistinguishable, cosine potentially edging out" | (match — and OUR probes were flat too; closed) |
| LR warmup | none | none in standard recipe | `Warmup` exists in bullet, rarely used | (match; closed) |
| Stage-2 LR | halved, 5e-4 (stage2_v1.sh:38) | halved, 4.375e-4, gamma 0.995 | same shape | (match) |
| Batch size | 16,384 | 16,384 ("advised") | 16,384 | (match) |
| Presentations, stage 1 | 6.0B (240 SB × 1520 × 16384); best 720 SB = 17.9B | 40–60B (400–600 epochs × 100M) | ~80B total not unusual (Alexandria ≈ 800 SB × ~100M) | **DIVERGE (scale — known lever; D40 curve NOT yet flat: +30.2 at 3×)** |
| Presentations, stage 2 | **3.0B (120 SB)** | **80B (800 epochs)** — stage 2 is LONGER than stage 1 | — | **DIVERGE (27× short; our hop plateau may be budget, not a dry well → X5)** |
| Loss | `sigmoid(out).squared_error(target)` — plain one-sided sigmoid, **power 2** (d40_features.rs:633; train_net0.rs:181) | `|pt − qf|^2.5`, **two-sided sigmoids with 270-internal offset**, in_scaling 340 / out_scaling 380 (lightning_module.py + config.py; docs: "good results with 2.6", code is 2.5) | sigmoid MSE power 2 (bullet default); `power_error`/`AbsPower` op exists | **DIVERGE (power + shape → X2/X7)** |
| Target temperature | sigmoid(score/400) after ÷1.9 units bridge ⇒ ~760 internal-cp per logit (~267 UCI-cp-equiv) | out_scaling 380 internal + 270 offset (~113–127 UCI-equiv) — ~2× sharper, but offset-flattened near 0 | scale 240–400 by engine (Viri 240, Reckless 380, Obsidian 400) | **DIVERGE — but naive sharpening ALONE is empirically refuted here (H3b′: ÷3.5 → −470). Only test as part of the full official shape (X7)** |
| λ / WDL, stage 1 | WDL 0.3 (= λ 0.7) constant | λ 1.0 (pure eval; "WDL-skipping works better with λ=1.0", PR #3927) | WDL(result-weight) ~0.2–0.5 | (our own probe: wdl0 flat at width 1024, −6.9 ± 17.4 — closed at stage 1) |
| λ / WDL, stage 2 | WDL 0.25 **constant** (stage2_v1.sh:39) | λ **scheduled 1.0 → 0.75** linear over the retrain (PRs #4020/#4100; #4351 ends 0.7) | `LinearWDL{start,end}` exists in bullet | **DIVERGE (schedule never tested; endpoint matches → X5 rider)** |
| Random skip | 75% (`D40_RANDOM_SKIP_PCT`, FNV-deterministic, d40_features.rs:68-75) | random-fen-skipping 3 (~75%) | used | (match — adopted 2026-07-15) |
| Smart-fen-skip | equivalent: best move must be quiet Normal + not in check (sfbinpack_filter, d40_features.rs:92-101) | ON by default | common | (match) |
| Early-ply skip | hard `ply >= 16` | early-ply skipping (soft variant exists) | similar | (match, minor variant) |
| **WLD-consistency filter** | **none** | **wld-fen-skipping ON by default** — skip positions whose eval contradicts the game result | rare | **DIVERGE → X6** |
| Draw down-sampling | classic only (`NET0_DRAW_KEEP_PCT`), inert in D40 runs | not used (filters instead) | not standard | (ours, keep as-is) |
| Mate/TB scores | dropped (|score| ≤ 10,000) | remapped to finite 7000 + 20000·0.8^plies band and KEPT | mostly dropped | minor DIVERGE (stage-2 Lc0 data only; bundle into X6 if touching the loader) |
| Units bridge | `SFBINPACK_SCORE_DIV` 1.9 (UHO/fishpack internal units → proven SF16 frame; tools/nnue/patches/bullet-sfbinpack-score-div.patch) | trains natively in SF internal units with matched loss constants | n/a (self-play data in own units) | (ours; empirically calibrated, proven by E3 → keep) |
| **Feature factorization** | **none** — 32-king-bucket HalfKA tensor trained raw (threat tensor has no king buckets, nothing to factorize there) | **HalfKAv2_hm^ factorizer** (virtual king-bucket-free features, merged at export; dedicated `factorized_weight_decay` in the optimizer config) | supported (`Factorised<A,B>`, `Factorises` in bullet inputs; example 4_multi_layer uses it with stricter ±0.99 clip); **Integral ships 768×12-factorized, Viridithas factorized king buckets** | **DIVERGE → X3** |
| Init | HalfKA `init_with_effective_input_size(32)`; threats zero-seeded, no bias (d40_features.rs:612-615) | FT random + virtual-zero style | default | (match in spirit) |
| Output buckets | 8, non-uniform popcount layout (d40_features.rs:123-132) | 8 by (piece_count−1)/4 | 8 is the norm | (match; Cosmo: buckets = "modest but clear improvement" — already have) |
| Activation / head | crelu·pairwise_mul FT → per-bucket 768→16→32→1, i8 L1 + f32 tail | SqrClippedReLU-family, int8 stack | SCReLU or pairwise-mul; SCReLU > CReLU (Cosmo) | (match — Reckless-shaped, parity-proven, leave alone) |
| QAT | none beyond the global ±1.98 clip; post-hoc quantise at save; parity gate at pack | continuous train-time weight clamping to the representable range is their whole "QAT"; quantization error "negligible" for shallow nets *when weights stay in range* | bullet example clips factorized layer tighter (±0.99) | partial DIVERGE — our recurring OOD quant tails (0.241/0.2517 vs 0.20 gate; barred the hop AND 720sb) say our weights/activations wander where theirs are boxed → X4 |
| Save cadence | production runs save FINAL ONLY (`D40_SAVE_RATE` 240/720 = end) + raw at saves (T5) | save every epoch-window; **checkpoints game-tested DURING training** (run_games.py, c-chess-cli, ordo ranking, delete_bad_nets.py) | varies | **DIVERGE → X1** |
| Run selection | 1 seed, endpoint screened | **~16 parallel runs, winners picked by GAMES, never loss** | varies | **DIVERGE → X1 (partial)** |

Sources for the official/community columns (fetched 2026-07-17):
- Loss/λ/constants: `model/lightning_module.py`, `model/config.py`,
  `model/lambda_utils.py` (official repo, master);
  loss lineage commits `b50c70d91` (2→2.6) and `52fc070f2` (2.6→2.5,
  "slightly stronger net").
- Recipes: Stockfish PRs #3927 (SFNNv4), #4020 (SFNNv5), #4100, #4351,
  #4795 (SFNNv8, 6-stage); nnue-pytorch wiki (Training datasets, Basic
  training procedure); our distilled note
  `stockfish_reference/05-nnue-pytorch-training-recipes.md`.
- bullet: `crates/trainer/src/optimiser/adam.rs` (defaults verified in OUR
  vendored checkout at `.archon-state/toolchains/bullet/`),
  `crates/bullet_lib/src/trainer/schedule/{lr,wdl}.rs` (CosineDecay/Step/
  Exponential/Warmup/Sequence; ConstantWDL/LinearWDL/Warmup),
  `crates/bullet_lib/src/game/inputs/factorised.rs` (Factorised/Factorises,
  merge at export), `crates/compiler/src/model/builder.rs`
  (`power_error(targets, p)`, `abs_pow`, `clip_pass_through_grad`),
  `docs/1-basics.md`, example `progression/4_multi_layer.rs`.
- Community ablation: Cosmo (Viridithas), "Experimenting with NNUE
  configurations" — https://asteri.sm/files/2024-06-25-nnue-research-01.html
  (LR schedules indistinguishable; SCReLU>CReLU ~50% size-efficiency;
  output buckets modest-clear; small-batch gains did not transfer to the
  long regime).
- Threats/arch context: SF SFNNv10 threat-inputs commit (8e5392d7, fishtest
  patch ≈ +13.9 ± 1.9 isolated); PlentyChess/Stormphrax/Viridithas release
  notes (threat-era releases +40..+94, always bundled with pawn-pair
  inputs). Confirms D40's feature bet; no action needed.
- Superbatch caveat: current bullet canonical is `batches_per_superbatch:
  6104` ≈ 100M/SB; OUR 1520 ≈ 25M/SB. All comparisons above are in
  PRESENTATIONS, not SB counts.

---

## B. The gaps that plausibly matter

Ordered by (mechanism strength × evidence quality × fit to our observed
failure modes). Each: what, why it matters, evidence grade.

**G1 — We throw away every checkpoint and every seed but one.**
Production D40 runs save the final superbatch only; one seed; endpoint
screened. The official pipeline saves continuously, game-tests checkpoints
DURING training, ranks by ordo, and deletes bad nets — because "training
exhibits variance; multiple runs needed for meaningful comparisons" and
loss never picks the winner. We have INDEPENDENTLY proven the premise: our
loss-strength dissociation is on record twice (lowest loss coincided with a
−881 screen, Probe A; the E2 d40 cell hit 0.000633 loss and screened −478).
Variance we already paid for is left unharvested at the exact place it is
cheapest to collect. Evidence: official practice (high) + our own logs
(high). Cost of fixing: ≈ zero GPU.

**G2 — Loss shape: power 2 plain sigmoid vs power 2.5 offset two-sided.**
Official loss = |pt−qf|^2.5 on two-sided sigmoids with a ~90-UCI-cp offset
and deliberate net/teacher temperature asymmetry (340 vs 380). Ours is the
bullet default: plain sigmoid, squared error. Mechanism: the exponent
shifts gradient mass toward large errors (better ordering of clearly
won/lost positions — exactly what search margins consume); the offset
flattens the gradient in the drawish band (less capacity spent
micro-fitting ±30cp noise); the in/out asymmetry trains the net ~0.9×
cooler than its teacher. Evidence: official code (high); 2.6→2.5 documented
only as "slightly stronger net" behind a green SPRT (medium); no public
isolated Elo for the whole shape vs plain MSE (honest gap). Our note 05
already prepared this ("bullet accepts custom loss_fn closures"). Bullet
has `power_error(targets, 2.5)` native — the exponent alone is a
ONE-WORD change. Caution from our own record: sharpening the TARGET
temperature alone (÷3.5) lost −470; the official shape's sharpness comes
packaged with the offset and exponent, so test the pieces in the right
order (exponent first, full shape second), never bare sharpening again.

**G3 — The 32-king-bucket HalfKA tensor trains unfactorized.**
Every (piece, square) HalfKA weight is learned 32 separate times — once per
king bucket — with no gradient sharing. The official trainer factorizes
exactly this (HalfKAv2_hm^: virtual king-bucket-free features whose
gradients update on every occurrence, merged into the real rows at export —
zero inference cost); their optimizer config carries a dedicated
`factorized_weight_decay`, i.e. it is a first-class recipe component.
bullet supports it (`Factorised`/`Factorises` + `merge_factoriser`; its own
multi-layer example uses it with a stricter ±0.99 clip on the factorized
tensor), and Integral (768×12 factorized) and Viridithas ship it.
Mechanism-fit for us is unusually good: D40 is our most data-hungry arch
(every historical D40 failure was data starvation), and factorization is
precisely a data-efficiency device for bucketed inputs — official docs:
"accelerates the early stages of training and reduces the sparsity of the
input." FullThreats is king-bucket-free by design (design/11 §12.2), so
the HalfKA tensor is the only candidate — which also keeps the change
small. Evidence: universal practice among bucketed-input trainers (high);
no public isolated Elo (the honest gap); mechanism fit (high).

**G4 — Our stage 2 is 27× shorter than official and unscheduled.**
Official stage 2 = 800 epochs ≈ 80B presentations with λ annealed
1.0 → 0.75; ours = 120 SB ≈ 3B at constant WDL 0.25. Two confounds in our
"stage-2 plateau" conclusion (+55 → +29 → +15.6): each hop was
budget-capped at 3B, and gains tracked LABEL quality — we never tested
whether ONE long stage-2 run at the proven label source (farseer family,
0.93 Pearson) keeps paying where three short hops flattened. Supporting
signal from our own data: schedule extension is arch-dependent — classic
1024 was flat at 480 SB, but D40 stage-1 720 SB paid +30.2 over 240 SB. The
λ schedule itself has no public isolated Elo (endorsed default, medium
evidence) — ride it along, don't test it alone.

**G5 — Nothing keeps the quantization tail in-distribution during training.**
Three separate D40 nets now show the same signature: mean/p95 parity
excellent, max barred by OOD endgames (hop 0.241, 720sb 0.2517 vs 0.20
gate; quant-vs-raw tracks runtime → the error is created at quantise time).
Official practice treats train-time range control as the whole point of
their "QAT" (continuous clamping so the int8/int16 mapping is lossless
where it matters); bullet's own example clips its factorized tensor
tighter. Our global ±1.98 keeps weights REPRESENTABLE (±1.98·64 = 126.7 ≤
i8 127) but nothing bounds accumulated activation/rounding drift on
positions far outside the data (bare-kings endgames after ply-16-quiet
filtering). This gap is currently COSTING us finished nets: 720sb's +30.2
is trained and unshippable without a waiver. Evidence: our own parity
records (high); the specific fix path (tighter i8-tensor clip vs STE
clamp vs post-hoc rescale_net.py) is unproven (low-medium) — treat as a
diagnosis-first experiment.

**G6 — No eval/outcome-consistency (WLD) filter.**
Official trains with wld-fen-skipping ON: positions whose eval does not
correlate with the final game result are skipped — a label-noise filter on
exactly the axis our own audits measure (farseer 0.93 vs test80 0.78
Pearson, and stage-2 gains tracked it). We adopted the random-skip and
smart-skip equivalents on 2026-07-15 but never this one. Mechanism: same
family as our draw-keep knob — distribution shaping — but keyed on
label-result coherence rather than draw rate. Expected effect largest on
noisy stage-2 corpora, small on clean UHO. Evidence: official default
(high that it's used); no isolated Elo (medium-low).

**G7 (minor) — Mate scores dropped vs remapped.** Official remaps mate/TB
to a finite 7000..27000 band and keeps them; we drop |score| > 10,000.
Irrelevant for quiet-filtered stage-1 UHO; mildly relevant for Lc0-derived
stage-2 mixes. Bundle with G6 loader work if it happens; never alone.

---

## C. Ranked experiment list

Score = expected Elo × confidence ÷ GPU-hours (RTX-4060-class, observed
costs: D40 240 SB ≈ 2.4 h, 720 SB ≈ 7 h; classic-2048 stage-2 120 SB ≈
1–2 h; 300g @5k screens ≈ seconds–minutes; equal-time gate 200g ≈ 3 h
quiet box). Twin baselines already exist for most cells (same seed 198273612,
same data, same recipe), so most A/Bs cost ONE new run.
All fixed-node screens are attribution instruments; the promotion bar
stays the 200g 10+0.1 equal-time gate per the live log.

**X1 — Checkpoint + seed harvesting (selection by games).** RANK 1.
- Knobs: `D40_SAVE_RATE` 240/720 → 30 (config env, e.g.
  `data/nnue/experiments/d40-long-v1/config.json` train.env); reuse the
  existing watcher pattern (`tools/nnue/probe_a_watch.py`) to screen each
  checkpoint 80–300g @5k vs the run's own endpoint; for promotion
  candidates only, add a second seed (`D40_SEED`) and pick the winner.
- Cost: ~0 extra GPU (saves are I/O; screens are minutes; second seed = one
  extra run ONLY for gate candidates). Expected: +5–15 fixed-node from
  variance harvesting; confidence medium-high (official institutionalizes
  it; our loss-strength dissociation is proven in-house).
- Success: any non-final checkpoint or second seed beats the endpoint by
  ≥ 2σ at 300g @5k, sustained on a 600g pool.
- Abort: after 3 campaigns, if the endpoint is always within noise of the
  best checkpoint AND seeds agree within noise, drop the ritual (keep the
  denser saves — they're free resume insurance, T5).

**X2 — Loss exponent 2.5 (one-word change).** RANK 2.
- Knob: `tools/nnue/bullet/d40_features.rs:633`
  `out.sigmoid().squared_error(targets)` →
  `out.sigmoid().power_error(targets, 2.5)` (op verified present in our
  vendored bullet: `crates/compiler/src/model/builder.rs:356`; classic twin
  at `tools/nnue/train_net0.rs:181`). 1-SB smoke first (backward-pass of
  `AbsPower` unexercised by us so far).
- Cost: 2.4 h (one 240 SB run; d40-uho-v1-240sb is the ready-made twin).
  Expected: +0–15; confidence medium (official lineage: 2→2.6 "good
  results", 2.6→2.5 "slightly stronger"; no public magnitude).
- Success: ≥ +15 ± noise vs twin at 300g @5k (same binary, same seed).
- Abort: |Δ| < 10 at 600g pooled → exponent alone is flat for us; do NOT
  iterate exponent values; proceed to X7 decision instead.

**X3 — Factorize the HalfKA tensor (virtual king-bucket-free features).** RANK 3.
- Knobs: in `d40_features.rs`, add a 704-dim virtual HalfKA input (the
  king-bucket-free, mirror-free (piece,square) factor of
  `halfka_index`), third summed tensor in the graph
  (`builder.new_weights("l0_virtw", (HL, 704), Zeroed)`; forward becomes
  `l0_pst.forward(hk) + l0_virt.matmul(hk_virt) + threats`), mapper emits
  the derived pair alongside each HalfKA pair (the derivation is a pure
  function of the existing index math in
  `src/eval/nnue/sota_features.rs`); at save, fold virtual rows into the
  22,528 real rows before quantise (mirrors bullet's `merge_factoriser`;
  netpack and the engine see an UNCHANGED format — zero inference cost).
  Optionally clip the virtual tensor tighter (±0.99, bullet's own example
  precedent).
- Cost: ~1 day engineering (index math already unit-tested engine-side;
  parity scan `D40_VERIFY_ONLY=1` extends naturally) + 2.4 h per A/B run.
  Expected: +10–30 at FIXED 240 SB budget (data-efficiency mechanism, our
  most data-hungry arch); confidence medium (universal practice among
  bucketed trainers + dedicated optimizer support in official config, but
  no public isolated Elo anywhere — we would be producing that number).
- Success: ≥ +15 vs unfactorized twin at 240 SB, 300g @5k; secondary
  signal: the 60 SB checkpoint of the factorized run matches the 240 SB
  unfactorized endpoint (the "accelerates early training" claim made
  measurable).
- Abort: flat at BOTH 60 SB (early) and 240 SB (final) → factorization is
  not our binding constraint at current data scale; park with a note to
  re-try only if a future arch multiplies bucket count.

**X4 — Quant-tail robustness (ship the already-trained +30).** RANK 4.
- Knobs, in diagnosis-first order: (a) post-hoc: `tools/nnue/rescale_net.py`
  pass on the barred 720sb checkpoint + parity re-run (zero GPU — do this
  regardless); (b) train-time: tighter clip on the two i8-quantised tensors
  (threats ×64, L1 ×64) — bullet's low-level path takes one global
  `AdamWParams` (d40_features.rs:686), so either global
  `{min_weight:-1.5, max_weight:1.5}` or locate bullet's per-weight
  override used by its example 4; (c) if (b) costs strength, STE clamp via
  `clip_pass_through_grad` on the accumulator pre-quant ranges.
- Cost: (a) minutes; (b) 7 h (needs the LONG schedule to reproduce the
  failure — the 240 SB net passed parity). Expected Elo: indirect —
  unblocks the +30.2 720sb class and every future long-schedule D40 net;
  confidence medium-high that the tail is fixable, low-medium on which fix.
- Success: parity max ≤ 0.20 with strength within noise of the default-clip
  twin (h2h 300g @5k).
- Abort: clip variant loses ≥ 15 Elo h2h → accept the rescale/waiver path
  and record the gate as distribution-bounded (the run configs already
  moved that way: mean/p99 primary, max backstop).

**X5 — One LONG scheduled stage 2 (the official shape at real budget).** RANK 5.
- Knobs: warm-retrain the best D40 stage-1 (or s3-2048 classic for a
  tax-free read) on farseer-family data (best label quality, 0.93 Pearson):
  480 SB (≈ 12B, 4× our standard hop), LR 5e-4 cosine, WDL scheduled
  0.0 → 0.25 (classic path: `wdl::LinearWDL{start, end}` at
  `tools/nnue/train_net0.rs:299`, one line; D40 path: the mapper closure at
  `d40_features.rs:566-601` already receives `_step` — interpolate wdl on
  `step.superbatch()`, three lines).
- Cost: 5–8 h. Expected: +10–30 over the 120 SB constant-WDL hop;
  confidence medium (official stage-2-longer-than-stage-1 practice is
  universal; our "plateau" was measured only at 3B/hop; the λ schedule
  rider is endorsed-default evidence only).
- Success: beats the existing 120 SB hop twin ≥ +15 h2h @5k; then
  equal-time gate as usual.
- Abort: intermediate 240 SB checkpoint (X1 makes it free) ≤ +5 over the
  120 SB twin → the stage-2 well is genuinely dry at this label quality;
  close the axis and leave data-quality (new corpora) as the only stage-2
  lever.

**X6 — WLD-consistency filter.** RANK 6.
- Knob: add the official-style skip to `sfbinpack_filter`
  (d40_features.rs:92-101 / train_net0.rs:109-121): skip when the stored
  score strongly contradicts the stored result (e.g. |score| ≥ K
  internal-cp with the result on the wrong side; calibrate K on the
  existing audit JSONs under `data/nnue/audits/` before training anything).
  Deterministic like the other filters.
- Cost: 0.5 day + 2.4 h A/B. Expected: +0–10 on clean UHO stage 1
  (probably nothing), +5–20 on noisy stage-2 corpora (test80-class);
  confidence low-medium.
- Success/abort: run it on a stage-2 hop, not stage 1; abort the axis if
  the noisy-corpus A/B is flat — do not tune K beyond one retry.
- Rider: if this loader work happens, add the mate-remap (G7) in the same
  patch behind an env knob, default off.

**X7 — Full official loss shape (two-sided offset sigmoids + temperature
asymmetry + λ-on-target).** RANK 7, GATED on X2's outcome.
- Knobs: loader-side pf computation (score and outcome separately — the
  mapper already has both at d40_features.rs:598-600), graph-side qf =
  0.5·(1 + σ((out−o)/s_in) − σ((−out−o)/s_in)) built from existing builder
  ops; constants transposed into our units through the proven ÷1.9 bridge
  (i.e. tune s_in/s_out/offset as UCI-frame equivalents, NOT a blind copy
  of 340/380/270 internal).
- Cost: ~2 days engineering + 2.4 h per run. Expected: +5–25; confidence
  medium-low standalone, medium if X2 showed any signal (the pieces are
  claimed to work together).
- Success: ≥ +15 vs the X2 winner at 300g @5k.
- Abort: flat → close the loss-shape program permanently (matches the live
  log's "prepared, unneeded so far" stance, now with the experiment
  actually run).

Explicitly NOT scheduled as method experiments (they are the data program,
already queued in the live log): longer D40 stage-1 (720 SB+ paying is a
DATA-scale fact), bigger/better stage-2 corpora (T60T70/farseer downloads),
self-play datagen flywheel. This study does not double-book them.

---

## D. Not worth it / already optimal for us

Respecting our own empirical record — these are closed, do not re-run:

1. **LR schedule shape ladders (cosine vs step vs exponential).** Our
   probes: flat (width-256 30/60/120 SB all ≈ +30–50; width-1024 480 SB
   −10.4). Cosmo's independent ablation: "essentially indistinguishable,
   cosine potentially edging out." Cosine stays. CLOSED.
2. **LR warmup.** Exists in bullet, absent from the official recipe,
   rarely used in the community, no evidence anywhere. Skip.
3. **Stage-1 λ/WDL constant ladders.** Our wdl0 (= official stage-1 λ 1.0)
   probe: −6.9 ± 17.4, flat. WDL 0.3 stays. CLOSED (schedule-in-stage-2 is
   X5's rider, a different question).
4. **Batch size for STRENGTH.** Official advice = 16,384; Cosmo found
   small-batch gains do not transfer to long runs; no public NNUE
   batch-vs-Elo ablation exists. 32,768 remains interesting ONLY as a
   VRAM-throughput experiment (3.4/8 GB used, GPU-bound 99%) and changes
   recipe semantics — the live log already classifies it correctly.
5. **Optimizer swap (Ranger/Lookahead port into bullet).** High engineering
   cost, zero community evidence it beats AdamW in a bullet setting, and
   every top bullet engine ships on AdamW 0.9/0.999/0.01/±1.98 — including
   engines above our level. The one official nuance worth remembering
   (per-tensor weight decay) arrives naturally with X3's factorized tensor
   if ever needed. Skip.
6. **Width ladder.** Saturated at 2048 classic (4096: +15 ± 17, eaten by
   the time tax). CLOSED by exp-0058.
7. **Naive target-scale "unit correction."** ÷3.5 UCI-calibration lost
   −470; strength was monotone in output amplitude; the consumer is
   hot-tolerant/cold-fragile (H3b′). The ÷1.9 bridge is the proven
   operating point. Never re-run bare sharpening (X7 is the only sanctioned
   revisit, packaged with offset + exponent).
8. **king768-style alternate king-bucket probes on classic.** Superseded:
   D40's HalfKAv2_hm at billions-scale data IS the king-bucket bet, and it
   works (+32.5 fixed-node). The scheduled "king768 probe" in the old
   next-levers list should be considered absorbed by the D40 line.
9. **Gradient-norm clipping.** Not exposed in bullet's value path; no
   engine reports needing it; our runs don't show instability (loss curves
   smooth). Skip.
10. **Superbatch-size convention (1520 vs 6104 BPS).** Cosmetic — only
    presentations matter; changing it would break every historical SB-count
    comparison in the ledger for zero benefit. Keep 1520, keep reporting
    presentations.
11. **Threat-feature validation.** The field converged on our bet after we
    made it (SF SFNNv10 threat inputs ≈ +14 isolated; PlentyChess-lineage
    releases +40..+94 bundled). Nothing to change — the remaining threat
    work is NPS (R2), not training.
12. **Draw down-sampling / distribution shaping beyond current knobs.**
    NET0_DRAW_KEEP_PCT exists, default-inert, and the UHO family already
    matches the proven temperature signature (~50% draws, std ≈ 312
    UCI-equiv). Leave inert unless a new corpus audits cold.

---

## Appendix: primary sources

Official: github.com/official-stockfish/nnue-pytorch —
`model/lightning_module.py`, `model/config.py`, `model/lambda_utils.py`,
`docs/nnue.md`, wiki (Training datasets; Basic training procedure
train.py/easy_train.py); commits b50c70d91, 52fc070f2, b6865d59a;
Stockfish PRs #3927, #4020, #4100, #4351, #4795; SFNNv10 threat-inputs
commit 8e5392d79a36ab. Bullet: github.com/jw1912/bullet — docs/1-basics.md,
docs/3-data.md, crates as cited in §A (all verified in our vendored
checkout `.archon-state/toolchains/bullet/`). Community: Cosmo's NNUE
config ablation (asteri.sm/files/2024-06-25-nnue-research-01.html);
Viridithas/Obsidian/Alexandria/Stormphrax/PlentyChess repos + releases;
chessprogramming.org/NNUE; arXiv 2412.17948 ("Study of the Proper NNUE
Dataset" — quiet-position filtering; weak strength anchor). Local:
`stockfish_reference/05-nnue-pytorch-training-recipes.md`,
`docs/research/value-foundation-state.md`,
`tools/nnue/bullet/d40_features.rs`, `tools/nnue/train_net0.rs`,
`tools/nnue/train_run.py`, run configs under `tools/nnue/runs/` and
`data/nnue/experiments/d40-long-v1/`,
`tools/nnue/patches/bullet-sfbinpack-score-div.patch`.

Honesty flags carried over from the research pass: no public source
isolates stage-2-vs-single-stage Elo, λ-schedule-vs-constant, batch-size
effects, factorization Elo, or quantization cost — the field documents
these as best practice, not measured deltas. Where we run X1–X7 we will be
generating numbers the community does not have; ledger them (D38).
