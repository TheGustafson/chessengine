# JAX workstream plan (activated 2026-07-22 evening, user directive)

**DIRECTION DECISION (user, 2026-07-23 ~02:0x): the JAX-GPU kernel fix is
THE next direction** ("obviously we have to fix the jax problem") — chosen
over another-training-push-now and the search arc, because every queued
research thread (sparsity retrain, doctrine run, T10, soft-policy heads,
spike-1) converges on a GPU-speed JAX trainer. Kernel agent commissioned
(Pallas embedding-bag / segment-sum / bf16 routes; gate ≥168k pos/s at
batch 8192; park <0.5×). Parallel cheap threads: data acquisition
(downloads) + TM SPSA as first quiet-night job. Crown rematch parked
until user calls it (protocol: 400g sidecar-ON + 400g OFF).

Source of truth for WHAT: `docs/research/jax-pipeline-study.md` (§7 phases,
§8 go/no-go). This doc is the concrete task ledger for executing it.
Standing user context: JAX is a committed direction (career + ambition +
TPU path); PyO3 bindings are part of this workstream; WASM/ARM tabled.

## Done (2026-07-22)
- [x] Study branch merged to main (23 files, additive; `jax/` package,
      parity tools, notebook, PyO3 design).
- [x] 7% heldout-readout discrepancy RECONCILED analytically:
      `forward_parity.py` reads a FILE-PREFIX subset (`np.fromfile
      count=N`) of the stream-ordered dump ⇒ easy-segment bias (same
      gradient as the training sawtooth). Trainer-graph instrument used
      all 999,424 rows. Confirm empirically in the quiet window (full-dump
      run); until then JAX absolute-loss numbers carry this caveat, while
      the byte/oracle parity proofs stand.

## Quiet-window queue A — RESEQUENCED (user, 22:0x): the RR is CANCELLED
## (rerun later "when we are stronger": tools/run_rr_tournament.sh), and
## this whole queue now waits behind perf study #4's report + the
## PROFILING RUNS on the freed box (clean measurements outrank JAX bench).
- [x] **RECONCILIATION CLOSED (23:0x, stronger than planned): same 4096
      stride-sampled rows through BOTH instruments — JAX 0.005771 vs
      trainer-graph 0.005771343, six-significant-figure agreement.**
      The pipelines are numerically identical; the 7%/-5% readings were
      prefix bias + stride-vs-full composition (row- vs game-weighted
      difficulty). JAX absolute numbers now fully trusted WHEN the
      sample is stated. TODO (cosmetic): forward_parity's top-1 print
      is UNMASKED argmax (reads ~2%, meaningless for a mask-trained
      head) — apply legal_move_masks before argmax.
- [ ] Pin the environment: lockfile for jax/jaxlib(+cuda) versions in
      `jax/pyproject.toml` (§8 ops gate).
- [x] **Phase-2 GPU benchmark FIRST RESULT (23:2x, gpu_step_bench.py):
      9,481 pos/s at batch 1024 = 0.05× bullet — far below gate, and
      batch 8192/2048 OOM (XLA materializes the gather: threats =
      B×160×1280×f32; 6.7GB at 8192).** The pre-registered sparse-gather
      risk is REAL and is a memory-materialization problem, not a
      compute one. Host loader confirmed: 217k pos/s single-thread ✓.
      **Gate NOT final until "reasonable tuning" runs:** (1) segment_sum
      /embedding-bag reformulation of the FT forward+grad; (2) a Pallas
      embedding-bag kernel (the known pattern for exactly this — ~50
      lines, keeps everything else JAX); (3) mixed precision (bf16
      gather halves traffic). If all three fail <0.5× ⇒ park per the
      abort framework (JAX stays the experimentation sidecar — which
      needs no GPU — and bullet remains the trainer). This tuning is a
      clean bounded agent task; commission when the box frees.
      ARCHON_JAX_ALLOW_GPU=1 opt-in added to the _require_cpu guard.
- [x] **Phase-2 gate PASSED (2026-07-23, kernel agent): Pallas-Triton
      embedding-bag + atomic scatter-add backward (custom VJP,
      `jax/archon_jax/ft_kernel.py`) = 225k pos/s full train step at
      batch 8192 = 1.07× bullet (gate was 0.8× = 168k); forward-only
      1.26M pos/s; peak 2.93 GiB of 8 GB; parity vs CPU reference
      3.5e-4 (< 1e-3 gate), 0/512 policy argmax flips. Route 2
      (pure-XLA scan) measured dead: 11.7k @1024, OOM ≥4096; bf16 not
      needed. Full numbers + go/no-go verdict:
      `docs/research/jax-gpu-kernel-2026-07-23.md`. CPU path untouched
      and default (`ARCHON_JAX_FT_IMPL=pallas` / `ft_impl=` opt-in).**
- [ ] Wire `jax/tools/quant_byte_parity.py` + `engine_parity.py` into the
      gate-script family (every future net passes them pre-game).

## Build-window queue B (cargo allowed; same window)
- [x] **D40_CONVERT_DUMP landed (2026-07-23, commit 2446fac):** one-pass
      trainer-parity sfbinpack→32B-record converter (bullet loader gained
      `with_single_pass` + tail flush; cumulative toolchain diff now
      tracked at `tools/nnue/bullet/bullet-toolchain.patch`). First
      corpus: fishpack32 → **1,612,358,018 rows / 49G** (63.0% keep =
      filter × skip-5 arithmetic, single pass verified). Global shuffle
      = `tools/nnue/shuffle_records.py` (two-pass bucket shuffle,
      multiset-checksum gate) → `fishpack32.shuf.bin` in flight.
      Convention: converts run with D40_RANDOM_SKIP_PCT=5 so the
      held-out shadow band [0,5) is excluded from EVERY corpus.
- [ ] PyO3 loader crate first build (`jax/rust-loader-design/` skeleton):
      exposes movegen/legal-mask + activation dumps at Rust speed (fixes
      the one below-reference loader component, 10k→fast legal masks).
- [ ] Top-k logit print + multi-head sidecar loader extension (policy
      roadmap build-window items — same window, shared rebuild).

## Phase 3 — ~~shadow training~~ CANCELLED (user ruling, 2026-07-23):
**"It isn't worth the time cost. Smoke test it aggressively then trust it
with the real thing. If it fails we only lost the same amount of time."**
The 7.4h replay is replaced by the smoke battery below (~45 min total);
the first real JAX training IS stage-3. Efficiency is a standing
directive.

## Phase 3' — aggressive smoke battery (each item minutes, not hours)
- [~] **S1 warm-start integrity**: quant re-export of d47c-stage2-sb240
      through JAX = **BYTE-IDENTICAL PASS** (137,089,088 B). Heldout
      reproduce + one-step sanity are wired as HARD PRE-FLIGHT GATES in
      the stage-3 driver itself (stage3_train.py aborts on mismatch).
- [~] **S2 trajectory replay — GATE AMENDED ON EVIDENCE (01:4x):** the
      train-loss comparison is CONFOUNDED: bullet's sb1-10 train loss is
      measured on its own stream (file-head, window-shuffled) while the
      JAX run samples the corpus uniformly — S2 read +8→11% on train
      value, but on the FIXED heldout population at MATCHED sb5 the JAX
      net is **+1.9%** from bullet's actual sb5 checkpoint (0.010949 vs
      0.010747; quant-vs-quant, common-mode error). Heldout-at-matched-sb
      is the correct instrument (axis law: Elo tracks heldout, ±4).
      Residual +1.9% ≈ unmasked-policy pull @λ=0.005 + init/run noise —
      S2b (value-only twin replay, RUNNING) pins the λ=0 residual.
      REVISED GATE: S2b heldout at sb5+sb10 within 2.5% of the twin's
      matched checkpoints. (First S2 attempt was killed by my own broad
      pgrep kill — ops rule added to memory.)
      **S2b VERDICT (02:0x): PASS-WITH-FINDING — JAX BEATS bullet on
      heldout at matched sb: sb5 −3.38%, sb10 −3.89%** (0.010384/
      0.009345 vs 0.010747/0.009724). The symmetric band tripped in the
      GOOD direction; the deficiency gate this test exists for passes
      decisively. Leak ruled out by construction (heldout band [0,5) is
      hash-disjoint from every corpus band). Two isolated effects:
      (1) S2-vs-S2b delta ⇒ UNMASKED policy @λ=0.005 drags heldout ~5%
      at sb5 (stage-3's λ/4 ⇒ ~1.3% early, endpoint-neutral per twin);
      (2) iid sampling over the globally-shuffled corpus outperforms
      bullet's windowed stream as a curriculum — a FREE structural win
      of the migration (pin: candidate explanation for future
      better-than-bullet curves; don't double-count it as a recipe win).
- [x] **S3 crash-resume — PASS (02:2x):** driver fresh-2sb → resume+1sb
      vs straight-3sb: heldout IDENTICAL to 4 sig figs (0.006734 both);
      train value shows only the known fresh-Adam transient. Also
      banked: S1 pre-flight fired live (warm heldout exactly 0.006094
      PASS); same-seed run-pair noise floor ~0.02% (A-vs-C sb1/sb2) —
      λ probes are effectively paired. NOTE: sb3 heldout 0.006734 >
      warm 0.006094 = the expected warm-restart bounce (SGDR arc, the
      stage-2 signature) — do NOT panic-abort stage-3 on early heldout
      regression; judge the arc.
- [x] **S4 export chain — GREEN (01:5x, on the JAX-trained S2 sb5):**
      netpack → .cenn (136.8MB) → engine loads → scale-check PASS
      (median 0.871 in band, sign 97.1% vs incumbent) → engine oracle
      p50 1cp / p90 4cp / 99% sign (gate ≤5cp p50; tail = the known
      ≤7-piece quant-edge pattern). Candidate engine for 1280 nets =
      engines/bin/archon-d47 (width is compile-time; target/release
      is a 768 build — same fact the crown-rematch protocol flagged).
- [x] S0 banked: kernel parity (3.53e-4, 0/512 flips), 400-step soak,
      corpus smoke at 4 offsets, 6-sig-fig loss reconciliation.
- [x] **S5 sparsity-λ probe — λ=0.001 PINNED (02:5x):** paired 3sb warm
      probes: λ=3e-3 = real GROWING value drag (+0.09→+0.23% by sb3,
      ~10× noise) — too hot; λ=1e-3 = value flat vs baseline (+0.02%,
      within noise) with density falling 0.399→0.366 and accelerating.
      **STAGE-3 LAUNCHED 02:5x** with λ=0.001: pre-flight PASS (0.006094
      exact), tb live (d47d-stage3/train incl. loss/heldout + ft/*),
      monitor armed (heldout/10sb + tripwires), GPU 93%, ~8.5h to sb240.
      BUILD SILENCE IN EFFECT until TRAIN EXIT.
On S1-S5 green → **launch stage-3** (driver: jax/tools/stage3_train.py,
wrapper: tools/nnue/runs/stage3_jax_train.sh; mix UHO .40 / fishpack32
.35 / farseerT74 .25 = 5.4B unique rows ≈ 1.1 epochs at 5.98B pres.;
~7.7h at 216k pos/s all-features).

## 2026-07-23 BAKED-IN WINS (user: "do those three", commit 3640f5c)
T0 factorization (-3% step), T6c PSQT skip (via 2nd pallas invocation —
XLA narrow-scatter route measured -32%, kernel route ~-1%), FT-sparsity
regularizer (λ knob + ft_density metric). All zero-init no-op proven,
warm-safe; ALL-ON = 216.6k pos/s (96.6% of plain kernel). Engine-side
PSQ1 consumer commissioned (worktree agent; format pinned in
checkpoint.py). D46 incident caught+fixed: converts made WITHOUT
SFBINPACK_SCORE_DIV carried raw units — rescale_records.py (f32-exact
LUT) repaired UHO+fishpack in place (frame verified vs heldout p50/p99);
T74 converted WITH DIV=1.66 at dump. Skip-75 dumps = the EXACT fixed
25% subpopulation bullet trained on (deterministic hash keep-test).

## (superseded spec kept for reference)
- [ ] **Data prerequisite: convert nodes5000pv2_UHO at skip-75**
      (`uho-band75-100.bin` → shuffle `--unlink-input` → ~84G final).
      SIZING (2026-07-23): UHO is ~16B raw positions — a skip-5 dump
      would be ~322G and does NOT fit this disk. But skip-75 is BETTER,
      not a compromise: the keep-test `hash%100 >= 75` is deterministic
      per position, so bullet's d47c stage-1 trained on a FIXED 25%
      subpopulation (~2.6B filtered rows, cycled ~2.3 epochs) — the
      skip-75 dump reproduces that EXACT multiset. Shadow-twin data
      equivalence is exact (same unique-position set); only sampling
      order differs. Heldout band [0,5) sits inside the skipped mass.
      [IN FLIGHT: conversion launched 2026-07-23 ~00:5x.]
- [ ] **Run: exact TrainConfig defaults** (= d47c stage-1: batch 8192,
      BPS 3040, 240sb, LR 0.001 cosine→0.3³, λ=0.005 policy ON, WDL 0.3,
      scale 400, seed pinned) with `ARCHON_JAX_FT_IMPL=pallas`. 5.98B
      presentations ≈ 7.4h at 225k pos/s. Full-state checkpoints
      (params+opt+step) every 10sb to disk (NEVER /tmp); tb telemetry
      armed AT LAUNCH (standing rule); heldout eval per checkpoint.
- [ ] **Gates, in order of cost:**
      1. Heldout-curve tracking: JAX heldout loss vs bullet d47c's
         logged curve at matching presentation counts — the ±4-Elo
         instrument; divergence beyond the run-to-run noise envelope
         fails fast (check at sb30/60/120, don't wait for sb240).
      2. netpack byte-path → quant → `net_scale_check.sh` PASS (D46).
      3. Engine oracle ≤5cp p50 (`engine_parity.py`).
      4. 300g fixed-node match vs d47c-1280-240sb endpoint — expect
         ~0±17 (trainer equivalence, not superiority).
      5. D38 ledger row (net + corpus rows for the converted corpora).
- [ ] On PASS → Phase 4 promotion per study §8; bullet inverts to
      reference oracle; campaign phase 2 (stage-3, warm from
      d47c-stage2-sb240, WSD/anneal-to-floor + new-pack folds) unblocks.

## Phase 4 — promotion + the "do it properly" run
- [ ] Promote per §8 (all four gates) → bullet's role inverts to
      reference oracle.
- [ ] First new-recipe run carries the full doctrine: WSD schedule with
      cooldown branches, share-locked λ, orbax full-state checkpoints,
      EMA, QAT trial, adaptive termination via the held-out instrument.
- [ ] Research queue unblocked thereafter: spike-1 (Gumbel sparse
      embeddings), §2b/§2c/§2d heads, T1/T5 screens in the notebook,
      T10 smallnet companion distillation (perfect first JAX training job).

## Sequencing constraints
- NO cargo builds / benchmarks / JAX execution while TIME-BASED matches
  run (crown gauntlet + RR tonight) — equal-time results corrupt under
  CPU contention. Fixed-node matches are immune.
- GPU work needs `XLA_PYTHON_CLIENT_PREALLOCATE=false` habits until the
  box stops hosting mixed workloads.

## Stage-3 live-arc calibration (03:5x)
Stage-2 precedent heldout arc (quant ckpts, 102k-row instrument):
sb10 0.007193 → sb30 0.007354 (STILL RISING) → sb60 0.007277 (turn) →
sb90 0.007132 (descending) → endpoint 0.006094 (+82.6 Elo). Stage-3:
sb10 0.006759 → sb40 0.006806 — SHALLOWER + FLATTER than the winning
run at matched phase. Gate recalibrated: intervene only if sb60 >
0.00696 (+3% over sb10, beyond stage-2's peak drift); primary judgment
= the TURN at sb90-120.

## STAGE-3 CLOSED (11:0x 2026-07-23)
TRAIN EXIT 0; 240sb @ 200.6k pos/s, zero interventions. Heldout
0.006094(warm) -> 0.006850(bounce peak sb50) -> 0.006330(sb240).
**300g endpoint gate: −4.6 ± 17.1 vs stage-2 = statistical parity at
FIXED NODES.** Honest ledger: pump-2 bought +60, pump-3 bought ~0 on
this instrument — BUT fixed nodes is blind to the run's real dividend:
FT density 0.384→0.313 ≈ +6-10% deploy NPS net of the PSQ1 tax, which
only equal-time shows (measured post-chain per protocol). Stage-3 also
banked 2.8B-row novel coverage (fishpack/T74) and repositioned the
trunk for the 5e-4 stage-4 pump. Chain doctrine satisfied: bank
coverage, hand off warm, judge strength at the settle. Screens
−17/−31/+14/−42/−31/0.0 (±30 each) = flat-at-parity series, matching
the gate. Bench 701605 bit-exact through the timeman promotion;
tune build lists 254 tunables; trainer rebuilt with D40_CONVERT_BAND.

## STAGE-5 PREP (18:2x 2026-07-23, drafted while stage-4 trained)
Endpoint step-8 is now execute-only:
- `tools/nnue/runs/stage5_data_chain.sh` (tracked DRAFT): fetch→convert→
  shuffle→rm-raw, one pack at a time, largest first, disk guards +
  refuses while training is live. Menu with ledger-audited DIVs:
  data_pv-2 2.25 (band 75:85), multinet 1.98 (75:85), pylon 1.95
  (75:90), wrongIsRight 1.85 (75:95), wrongNNUE 1.49 (75:95); T76
  optional (raw already local). Pre-cleanup deletes UHO+fishpack raws
  (converted corpora ledgered; HF re-fetchable). Bands finalize at
  endpoint (hash-band width = disk knob; lo>=5 heldout guard).
- **House-corpora units finding (verified today):** canon-tight-v1 +
  hardneg-tight-v2 carry SF18-UCI cp clamped ±2499 (canon audit slope
  1.000; percentile + clamp check). Training frame is 1.484×UCI →
  rescale-a-copy wired into the chain: canon-tight --div 0.674,
  hardneg --div 0.699 (its OWN audit, 18:4x: slope 1.038 vs SF18-UCI,
  Pearson 0.9936, sign 0.992 — PASS, ledger ann-0032/0033). No i16
  overflow (clamp 2499 → max ~3708). Originals stay canonical.
- `tools/nnue/runs/stage5_pump.json` skeleton: warm/heldout_expect/mix
  weights FILL AT ENDPOINT; placeholder split ≈59% new / 41% rehearsal.
- Ops note: stage-4 janitor was found dead (another silent kill);
  swept manually at sb170 — keeps sb60/120 + last two, disk 19G→33G.

## Screens-protocol finding (19:2x 2026-07-23, pre-endpoint)
The engine auto-loads a sibling `.cpol` unconditionally (nnue/mod.rs:296;
no option gate), and every packed candidate HAS one — so the v2 screen
series AND the stage-3 300g gate ran candidate-WITH-sidecar vs
base-WITHOUT. Not value-vs-value: each number carries the candidate's
own sidecar state, which ROTS across pumps (top-1 23.4→20.0%), so the
confound's sign is unknown and stage-dependent. Implications: (1) the
"pump-3 bought ~0" read has an unquantified sidecar term; (2) tonight's
endpoint battery strips .cpol from ALL sides (copies) — EMA-vs-LIVE,
vs-stage3, vs-stage2 are the first clean cross-stage value-only deltas;
(3) don't naively compare tonight's yardstick number to the old screen
series; (4) sidecar worth stays a phase-5 question, measured by ON/OFF
ablation + the top-1 probe, never mixed into value promotion gates.

## THE UNION RUN (user ruling 20:4x 2026-07-23) — supersedes staged 5/6
"Shuffle everything from every stage together and run one big stage over
all of it." Design (campaign doc has the governing text): Leg A = WSD
plateau, constant 5e-4 (lr_end_superbatch=0 hits the driver's constant
branch — zero code changes), warm from the stage-4 battery winner, EMA
0.99984, proportional-to-rows mix over the FULL union (auto-epoch-guard),
sb cap 720 (~1.1 epochs of a ~10B-row union), extend by resume. Leg B =
ship anneal config (cosine end==len -> 0.027x floor ~1.4e-5, warmup 0),
warm from the chosen plateau ckpt, ~15-20% of plateau length. Saturation
instrument = in-run heldout/EMA + tools/nnue/runs/d47f_union_watch.sh
(30sb value-only 100g EMA screens vs stage4-winner-nocpol). Tools:
make_union_mix.py (proportional weights from file sizes),
union_plateau.json / union_anneal.json (FILL marks), stage5_pump.json
superseded. Ops prep landed tonight: freed 148G (already-converted +
waived raws; all 5 stage-5 packs were ALREADY in raw/ so the chain skips
fetches) + 30G (d47c per-sb quant sweep to runbook keeps) -> 202G free;
chain bands widened to 75:88 for the big three; chain training-guard
pattern fixed (unanchored pump_train.py). Tomorrow: battery winner ->
make_union_mix -> fill leg-A config -> launch plateau + watcher + janitor.

## STAGE-4 CLOSED (20:5x 2026-07-23) — the EMA night
TRAIN EXIT 0, 240sb @183k pos/s, zero interventions. Heldout 0.00633
(warm) -> live 0.006557 / EMA 0.006452 (EMA monotone sb30->end);
density 0.313->0.254. Endpoint battery (VALUE-ONLY protocol, all in
~4 min on the idle box — exp-0130..0132, nets d47e-stage4-sb240[-ema]):
- **EMA vs LIVE +40.7±17.2** (127-81-92): EMA doctrine validated at
  play level; the shadow harvests ~40 Elo of hot-endpoint bounce.
  WINNER = d47e-stage4-sb240-ema (union warm start;
  screens/stage4-winner-nocpol.cenn is the watcher baseline).
- **vs stage-3 +5.8±16.9**: shippable-vs-shippable parity; by
  composition the live trunk sits ~-35 vs stage-3 — the EMA carried
  the stage. Banked instead: coverage + density + the EMA mechanism.
- **vs stage-2 +52.5±17.0 — THE CHAIN REHABILITATED**: composition
  puts stage-3 ~+47 value-only over stage-2, vs the old CONFOUNDED
  -4.6 gate: the rotting sidecar was hiding ~50 Elo of real value
  progress. The pump chain has been ascending all along.
- **Top-1 probe (EMA net): 23.2%** vs the 23.4->20.4->20.0 live-era
  rot series — EMA averaging repaired the policy head essentially for
  free. Phase-5 recovery baseline moves up accordingly.
Scale gates 0.988/0.997 (EMA quant+pack path clean). Data chain fired
20:55 (fetch-skip: packs pre-staged; 207G free at start).

## UNION RUN LAUNCHED (21:17 2026-07-23) — d47f-union LIVE
Auto-pipeline end-to-end in 32 min: train exit 20:49 → battery 4 min →
data chain 20 min (5 packs, fetch-skipped) → config auto-fill → launch.
One incident: rescale_records LUT overflow for expanding divisors
(div<1) — the table iterates all 65536 i16 inputs and ±32767/0.674
overflows; fixed by i16 saturation (7b0f44c), both frames re-verified
exactly (canon p50 153 = predicted 153, max 3708; hardneg p50 205).
**UNION: 10,188,262,722 rows / 12 corpora, epoch ≈ 617sb**; proportional
weights (uho .251 / fishpack .158 / T74 .121 / datapv2 .097 / T75 .090
/ multinet .090 / t60t70 .082 / pylon .047 / wisr .035 / wrongnnue .026
/ canon .0027 / hardneg .00013); new-coverage ≈ 31%. Warm gate EXACT
PASS (0.006452). Ledger: 7 new corpus rows (datapv2/multinet/pylon/
wisr/wrongnnue band corpora + canon-tight-frame/hardneg-frame).
Watcher (30sb value-only screens vs stage4-winner-nocpol), janitor
(sb%60+last2), tb shim, 3 monitors — all armed. Cap 720sb ≈ 18h;
saturation rule decides the anneal-leg trigger (union_anneal.json).

## ANNEAL PROBE (03:1x-04:4x 2026-07-24) — plateau HEALTHY, harvest proven
Screens sat flat at ≈−70 through sb150 (−74/−81/−56/−67/−74) with
heldout-EMA netting zero (0.006716→0.006708 over 110sb) → fired the
overnight-doctrine probe: killed plateau at sb155, 30sb mini-anneal
(d47g-probe) from sb150-ema, warm gate exact. The anneal DOVE:
heldout 0.006708 → 0.006512 live / 0.006500 ema at sb30. Screens:
**live −37.2±17.2 (300g)**, ema −38.4±30.4 (100g) vs stage4-winner.
VERDICT: +37 Elo harvested on demand from 25% of one union epoch —
the flat plateau screens are hot-LR tax, not absent learning; too-hot
hypothesis REFUTED; predictor validated again (forecast −20..−35).
Calibrations banked: anneal-harvest offset ≈ +35-40 at this depth;
EMA≈live at cold endpoints (leg-B pick becomes a coin-flip/cheap
check). RESUMED plateau from sb150 at 5e-4 (judgment: middle-zone
result + proven consolidation → stable phase stays hot). Cost: ~95
min GPU + 5sb re-run. Screens now read as (level + ~37) when
annealed; saturation rule still reads the TREND.

## Saturation instrument v2: PROBE-VS-PROBE (08:1x 2026-07-24)
The 100g plateau screens are a flat high-variance band (8 points,
−74/−81/−56/−67/−74/−35/−63/−85 ≈ −67±noise) — constant-LR theory's
prediction; they stay as a coarse tripwire only. THE saturation
measurement: at the epoch boundary (~sb617) fire probe-2 (same 30sb
snap-anneal config, warm from sb600/610-ema), then match probe-2 vs
probe-1's artifact (d47g-probe-sb30, −37.2±17.2 vs stage4-winner)
head-to-head 300g value-only. Probe-vs-probe IS the pump delta at ±17
on annealed artifacts. Clear probe-2 win → plateau still paying,
consider extending past sb720; flat → saturation, launch leg B (real
120-180sb ship anneal) from the best checkpoint. Probe cost ~95 min
each; decision quality worth it.

## PLATEAU ENDED BY RULING → SHIP ANNEAL LIVE (14:1x-14:4x 2026-07-24)
User call: "time we start prepping this network for release and
annealing it" — plateau ended at sb410/720 (66% of one union epoch).
RESUME POINT preserved + ledgered (d47f-union-sb410; keeps sb60..410;
resume: pump_train union_plateau.json 410). Plateau final state: EMA
0.006688 (floor arc 0.006727→0.006679), live band ~0.00700-0.00714,
density 0.2079, pol_ce 3.40→3.28. d47h-anneal LIVE: 150sb cosine
5e-4→0.027x, warm sb410-ema, warm gate EXACT PASS, same union mix.
Settle battery armed (EMA-vs-live → vs stage4-winner → vs probe-1
annealed-to-annealed → top-1 → finals) = the RELEASE VALUE NET
decision, ~6h out. Ops lesson (bit twice today): kill/pgrep commands
must never contain target patterns literally — heredoc/script text in
argv self-matches; use [b]racket patterns in FRESH commands only.

## UNION SETTLED + JOINT PIVOT TOOLED (19:5x-20:1x 2026-07-24)
d47h-anneal-sb150 = VALUE CHAMPION: +34.9±17.4 vs stage-4 (300g
value-only, exp-0133), EMA-vs-live −3.5 tie (cold-endpoint convergence
as probe-predicted; live wins tiebreak), +41.9±16.9 vs probe-1,
heldout 0.006262, top-1 20.0% (rot floor — the joint baseline), gates
0.981/0.984. Clean chain since stage-2 ≈ +87 fixed-node value-only.
d47h = release fallback + joint warm. PIVOT TOOLING: pymask crate
(bit-exact vs python-chess on 3 corpora incl. pin positions; >1M
pos/s; fixed the pinned-move leak that silently afflicts the bullet
d40 mask block; canon-tight has NO packed moves), data.py fast path,
pump_train pol_mask flag. Smoke ladder RUNNING (control whisper-
unmasked / lam0.05 masked / lam0.25 masked; 10sb each from the
champion at 2.5e-4 const): gate = sb10 heldout drag vs control +
pol_ce slope + tripwire silence + throughput + top-1 movement.
joint_plateau.json staged (d47j-joint, champion warm, 240sb cap,
adaptive; FILL pol_lambda from ladder). Loss-scale caution: pol term
≈40% of value term even at whisper λ — dominance risk at 0.25.

## LAMBDA LADDER + JOINT LAUNCH (20:0x-22:0x 2026-07-24)
4-rung ladder from the champion (10sb, 2.5e-4 const): control 0.006636
/ lam0.01 +0.00006 PASS / lam0.05 +0.0004 / lam0.25 +0.00136 (clean
monotone dominance curve — the loss-scale caution was right). Masked
learning is dramatic: pol_ce 3.11→2.43 in 10sb; TOP-1 20.0→**25.2%**
in 10sb at lam0.01 — beats the whisper era's all-time best (23.4) from
1/70th of a pump. Mask tax ~20% throughput (146k pos/s; mitigations
known, deferred). d47j-joint LIVE (champion warm exact, lam0.01
masked, 240sb cap, EMA, full union mix); watcher screens vs
d47h-champion-nocpol; janitor+shim+monitors armed. Cutover to the 2nd
ship anneal on value-stability + pol_ce flattening. d47h remains the
untouchable release fallback.

## JOINT RUN: going-long doctrine (23:5x 2026-07-24)
User: net cooks as long as needed; post-anneal expectation = strongest
net + ~arch/data ceiling → then T11 → TM-SPSA+bracket → profiling/
external battery → comprehensive SPSA → release engineering (~2wk to
tag). EXTENSION RULE: never edit superbatches+resume — the mix
scheduler draws corpus-choice THEN permutations from one seeded
stream, so changing the total reshuffles orders and breaks the
skipped-prefix correspondence. Extend via warm-handoff leg
(joint_ext.json staged). Cutover = top-1 flat 60sb+ AND value stable.
Settle battery v2 adds the sidecar ON/OFF ablation (fixed-node AND
equal-time) — first trained-head deployment test. sb40 state: value
drag ~nil (live 0.006683/EMA 0.006491), pol_ce 2.40, top-1 EMA 24.4%
@sb30, density 0.258↑, 146k pos/s (mask tax ~20%).

## D47J VERDICT + THE VALUE-TAX FINDING (2026-07-25 morning)
d47j ran clean to sb240 (top-1 ~20→25% pooled, pol_ce 3.11→2.29, EMA
heldout flat 0.0065 all run) and the cutover chain fired flawlessly
(d47k anneal launched 60s after exit, warm PASS bit-exact 0.006506).
Then the endpoint screen read −96 and forensics rewrote the story:
**masked joint training at λ0.01 costs ~30-40 Elo of value-only play
vs d47h from EARLY in the run** — invisible to quiet-heldout MSE, and
the in-run 100g screens were too noisy to see it (sb150 −3.5 was
noise; 200g rechecks: sb120 −38±18 pooled, sb210 −27±17, sb230
−40±21, sb240 −86±18 — endpoint carries extra damage; density/ft_l1
growth is the visible correlate). Pack determinism, quant/netpack clip
gates, game logs, measurement env (sb210 control recheck) ALL ruled
out. LESSONS: (1) 100g screens never carry narrative weight — 200g+
or silence; (2) play-strength drift needs its own tripwire — MSE-flat
is not play-flat.

## HEAD AUDIT + P-ORD (2026-07-25 morning)
User worry "25% could be dumb type-priors" REFUTED empirically
(scratchpad/policy_skeptic.py, 1201 SF18-labeled positions): the head
is a QUIET-MOVE SPECIALIST — quiet top-1 29.6%, checks 40.0%,
captures 3.2% (below random; D46 quiet filter removed capture targets
from training). Dumb baselines: random-legal 7.0%, MVV-LVA-else-random
16.8%, phase/type-prior (hindsight-fit) 6.0%. Complementarity thesis:
search already owns captures (SEE/MVV-LVA); history is weakest on
quiets — point the head exactly there. **P-ORD landed (403bc18)**:
QuietPrior = selected-row SIMD dot vs the picker-construction copy of
the L2 stash, quiets only, CPL_ORD_W env (0=OFF bit-identical,
verified at search level incl. MIN_DEPTH gate identity; ~6% NPS).
W-sweep @5k fixed nodes, same net both sides: W=10 **+11.6±11.7
(600g)**, W=32 −1.7±20.4, W=100 −13.9±20.6 — monotone: gentle nudge
positive, override negative. Promising-not-proven → equal-time SPRT
queued for the idle-box day.

## D47K REPAIR TRAJECTORY + D47L PIVOT (2026-07-25 midday)
d47k (anneal from the damaged sb240) is REPAIRING on schedule: screens
−86 warm → −70±31 @sb30 → −52±32 @sb60 (+17/30sb, cold tail still
ahead → parity-extrapolation at sb150), head SHARPENING through the
anneal (pooled top-1 25.8% @sb60, uho800 26.1% best-ever @sb30), EMA
heldout descending 0.006733→0.006642. KEEP ruling at sb60; kill case
was flat. **D47L staged (95ce4d3)**: train_only param freezing in the
JAX trainer (grads zeroed AND post-update restore — AdamW decay-proof;
freeze oracle = non-head tensors byte-identical + heldout exactly
0.006262) + frozen_head.json (masked head, λ1.0, lr 1e-3 cosine 90sb,
champion-trunk warm = zero value tax by construction). Launches on GPU
free (~17:15). Endgame: settle battery v2 compares (d47k joint net) vs
(d47h + D47L head) vs d47h floor — value-only 300g + sidecar ON/OFF +
P-ORD, fixed-node today, EQUAL-TIME tomorrow on the idle box. uho800
added as second probe set everywhere (canon400 alone churns ±3pts).

## λ-RAMP RULE FOR FREEZE-OUT LEGS (user ruling 2026-07-25 ~16:2x)
D47M overshoot: full ft_sparsity_lambda from step 0 of a frozen-head
leg let the L1 prune ex-policy features unresisted — density
0.29→0.19 (below champion 0.26 equilibrium), then slow self-recovery;
candidate cause of the residual −30 screen gap at heldout parity.
RULE: any leg that removes a gradient source (head freeze-out, trunk
freeze) starts ft_sparsity_lambda at 0 and ramps to target over
~20-30sb. Needs a small trainer knob (ft_sparsity_warmup_superbatches,
mirror lr warmup) — add before the next freeze-out leg. Meanwhile
D47M crossed the champion's heldout at sb70 (EMA 0.006232 <
0.006262), first net ever to do so; play verdict rides on sb90/120
screens + settle.

## SETTLE Q1 + THE POLICY-SUBSTRATE VERDICT + D47P (2026-07-25 evening)
**Q1 SETTLED — D47M IS THE VALUE LEADER: +26.7±17.0 vs d47h champion
(300g, 5k nodes, value-only, nocpol both sides; W-D-L 119-85-96)**,
confirming the cold-endpoint screens (+26±21 pooled 200g). Heldout EMA
0.006120 vs champion 0.006262; density 0.2007 (~24% sparser = unmeasured
equal-time speed). Ledgered (net row + exp-0135).

**BUT THE REPAIRED TRUNK WENT POLICY-HOSTILE.** d47n (head-only resync on
D47M, train_only polmove*, λ1.0 masked) crawled: pol_ce 3.01→2.94 in
10sb, top-1 15.6% pooled @sb10 (from the 9.7% rot floor). User correction
that reframed it: with legal-move masking the CE scale is ln(~30)≈3.4,
NOT ln(1858) — so 2.94 is only modestly better than random-legal, and the
gap to the joint head's 2.29 is large. Freeze oracles both PASSED (byte
level: 12 frozen tensors identical, only polmove* moved; metric level:
heldout reproduced 0.006120 EXACTLY). d47n KILLED at sb10 by user call
("dead to me") — the ceiling on D47M's trunk is not worth 3 more hours.

**USER RULING: "it isn't a trade imo, we could have both."** Mechanism
agrees: the value gain came from removing the gradient CONFLICT, not from
the pruning; pruning policy-recruited features is ~neutral for value and
fatal for policy. But a λ ramp ALONE is insufficient — with the head
frozen AND pol_lambda 0, policy features have no gradient support at all;
the L1 only accelerates a death that disuse would cause anyway.

**D47P = VALUE REPAIR v2 (live 18:3x, value_repair_v2.json).** Same warm
point as D47M (d47k-sb90-ema, warm gate PASS) = controlled comparison.
Two changes: (1) ft_sparsity_lambda RAMPS 0→0.001 over 25sb (the new
trainer knob, 949e065); (2) **pol_lambda 0.002 MASKED with the head still
PARAM-FROZEN = a DISTILLATION ANCHOR** — the trunk keeps a gradient
saying "preserve the features this good head uses" while the head cannot
chase or annex capacity (that was the joint tax at λ0.01 with a TRAINABLE
head). Anchor is 1/5 the joint strength.
**KEY INSTRUMENT: with the head frozen, top-1 measures FEATURE
PRESERVATION directly.** D47M control curve: 26.5% (warm) → 10.7 (sb30)
→ 9.4 (sb60) → 9.7 (sb90). D47P read at sb30: >22% = anchor working;
<18% = too weak, reassess rather than burn the run. Value instruments
unchanged (heldout/10sb, 100g value-only screens/30sb vs champion) so an
anchor that costs value shows up immediately.

**RELEASE PATH STRUCTURE (user, 18:4x):** if we ever abandon the policy
head for v1, **D47M is the release net**. For now we make the head work.
Fallback ladder: D47P (both) → D47M (value-only, +26.7 banked) → d47h
(untouchable original). Nothing promotes without settle_v2 q2/q3.

**OPS INCIDENT:** a CPU linear-probe job OOM-killed the trainer at
18:16 (15GB box; violated our own no-heavy-CPU-during-training rule).
sb10 ckpt was already written → resumed with zero loss. Re-run of that
probe now uses `systemd-run --user --scope -p MemoryMax=... -p
MemorySwapMax=0`, which contains the kill to the probe's own cgroup
(verified: "Memory cgroup out of memory" kills only the probe). Use that
pattern for ANY analysis job that shares the box with a trainer.

## OWED OPTIMIZATION: kill the dense policy mask (2026-07-25 ~19:1x)
The masked-policy path costs ~25% throughput (145k pos/s vs 197k unmasked)
and it is ALL redundancy, not legality computation (pymask is >1M pos/s).
We materialize a dense (batch, 1858) float32 mask = **58 MB per batch** at
batch 8192 — memset to -30, scatter ~30 zeros, copy over PCIe, add
elementwise on GPU. At 17.7 batches/s that is ~1 GB/s of mask traffic.
Measured: 41.6 ms/batch unmasked vs 56.5 ms masked = ~15 ms for 58 MB
(~4 GB/s effective) — memory-bandwidth bound, on a box whose RAM also
feeds the trainer. 1858 entries carrying ~30 bits of information.
**FIX (do before ANY future policy run):** pass legal move INDICES as
(batch, max_legal) int16 (~0.65 MB, 90x less) and compute CE by gathering
logits at those indices + log-sum-exp over them; or build the mask
on-device. Expect masked runs back to ~190k pos/s — hours per run.
NOT done mid-D47P on purpose: changing the data path would void the
controlled comparison vs D47M (identical schedule/mix), and a rushed edit
to the loss path is where a silent correctness bug would hide.

## ANCHOR SEQUENCING RULING (user, 2026-07-25 ~20:5x)
No external anchoring until the known-broken things are fixed — measuring
a defect wastes the measurement and forces a redo. ORDER:
  1. finish the net (D47P -> D47Q -> duel -> release candidate)
  2. POLICY HEAD REFINEMENT (P-ORD equal-time SPRT, sidecar ON/OFF, any
     further head work the duel motivates)
  3. INITIAL TM FIX — user reports from MANUAL GUI PLAY that time
     management "seems like a big problem", corroborating the analytic
     finding (~19s burned on move 1). Manual play surfaces TM pathologies
     that fixed-node testing structurally cannot see. Staged: tm-v1.json
     (28 params), tm_bracket_measure.sh.
  4. LIGHT SPSA
  5. **BASELINE vs SF16 AND SF18** <- the first external anchor, on a
     clean engine. Last real SF18 number (-191+/-23) is STALE: it was the
     older 2048-wide farseer line, not the D47 architecture; current gap
     is genuinely unknown (chain arithmetic suggests ~-90..-110 but
     fixed-node value-only deltas do not compose into equal-time Elo).
  6. first optimization/profiling pass
  7. comprehensive SPSA
  8. release engineering
Rationale for 5 sitting BEFORE 6/7: the anchor becomes the yardstick that
prices the two biggest remaining levers instead of being confounded by
them.

## MULTI-HEAD TRAINING: POST-RELEASE RESEARCH QUEUE (user, 2026-07-25 ~22:0x)
Ruling: note these, do NOT chase before release — go through the rest of
the plan first to calibrate how strong the engine already is. Today's
gains came from MECHANISM insight, not tuning, so more mechanism likely
remains; but the TM fix alone probably outweighs all of it.

Sorted by expected value per hour:
**Cheap, probably real**
1. ANCHOR STRENGTH UNSWEPT. pol_lambda 0.002 was intuition (1/5 of the
   joint value), never measured. The λ ladder proved this knob is SHARP
   (0.01 taxed value ~30-40 Elo; 0.25 dominated). Optimum could be
   0.001 or 0.005. ~3 short legs to find out.
2. ALTERNATION FREQUENCY. We ran ONE long value phase + ONE head phase.
   True BCD alternates repeatedly (value 20sb → head 10sb → …), which
   keeps the anchor FRESH instead of pinning the trunk to an
   increasingly stale head. Same total compute — pure scheduling.
3. L1 TARGET ITSELF. 0.19 density killed policy; ~0.27 preserves it.
   Nobody has asked what density is jointly optimal for value+policy+NPS.
**Structurally different**
4. ANCHOR ON OUTPUTS, NOT LABELS: KL against the warm net's own policy
   distribution ("don't change your policy behavior") instead of CE
   against true moves through the frozen head.
5. SEPARATE LR FOR THE HEAD during joint phases, so the head tracks the
   trunk rather than the trunk being dragged toward the head.
6. GRADIENT SURGERY (PCGrad / GradNorm / gradient vaccine) — the
   literature's answer to task conflict; often loses to good weighting,
   but never tried here.
7. DEDICATED CAPACITY: give policy its own bottleneck slice so the
   competition vanishes by construction (costs NPS).
**The big unexplored region**
8. FROM-SCRATCH JOINT TRAINING AT SCALE. Every joint experiment so far
   fine-tuned a value-SATURATED trunk. AlphaZero-style co-training from
   scratch is a different optimization problem and may reach a joint
   optimum none of the above can. Flywheel era, not now.

This list is also prime technical-report material — it turns the
write-up from a changelog into a study of multi-head NNUE training.

## FT PERMUTATION (SCHEDULED) + GROUP-AWARE PENALTY (RESEARCH ONLY)
User ruling 2026-07-25 ~22:3x: **the permutation WILL happen — it is free.
Slot: AFTER light SPSA, BEFORE the SF16/SF18 baselines** (so the first
external anchor measures the engine we intend to keep).

**Why permutation is free:** permuting FT output neurons and applying the
inverse permutation to L1 input rows computes an IDENTICAL function —
pure reindexing, no retraining, verifiable bit-exact. Our sparse L1
(`sota_head.rs:163-183` find_nnz → l1_dot) skips a 4-byte group only if
all 4 activations are zero, so which neurons share a group matters.
Theory: grouping SIMILAR-rate neurons dominates mixing them — for rates
{0.9,0.9,0.1,0.1}, {0.9,0.9}+{0.1,0.1} gives 1.18 expected active groups
vs 1.82 for the mixed pairing (35% less work, same neurons).
Procedure: activation stats over a large sample → clustering solve
(co-occurrence-aware beats sort-by-rate) → permute exported net →
assert bit-identical eval.

**FIRST, THE CONTROLLED A/B** (~1h): permute ONE net, measure NPS before
vs after. Today's cross-net evidence says this may be worth ~nothing:
D47M at ~57% group occupancy and d47h at ~70% benchmarked IDENTICALLY
(657.9k vs 655.7k nps, 3 positions x 2 passes under training load). If a
13-point occupancy swing bought 0%, permutation may too. Likely the cost
is dominated by the O(L1) find_nnz scan + accumulator update + non-eval
node work, not by group count. NOTE: this also undercuts the ORIGINAL
justification for the ft_sparsity penalty (adopted as "perf study #6 #1
NPS lever") — revisit whether we should pay capacity for density at all.

**GROUP-AWARE SPARSITY PENALTY — research queue, low priority.** Group
lasso (L2 over each 4-block, summed) instead of element-wise L1 makes
whole blocks die together. It is NOT redundant with permutation:
permutation cannot create correlation, and correlation is what makes
blocks skippable (independent neurons at rate 0.5 → 93.75% active groups
under ANY permutation; block-correlated at the same rate → 50%). But it
buys ONLY inference speed and it COSTS representational capacity (a
strictly stronger constraint than L1 at equal sparsity). Given the A/B
evidence that occupancy does not drive our NPS, this is probably a bad
trade for us. Verdict: park unless the permutation A/B surprises.

**WHAT STOCKFISH DOES (checked against the local clone):** VERIFIED —
SF uses the same find_nnz sparse-input affine
(`affine_transform_sparse_input.h:95-137`), and its `get_weight_index_
scrambled` (:177-192) is pure SIMD LAYOUT arithmetic (dpbusd chunk
interleave), NOT activation clustering. BELIEVED-NOT-VERIFIED — the
activation-clustering permutation lives trainer-side in nnue-pytorch,
which is not in our clone. NO EVIDENCE SF uses any group-aware sparsity
penalty; their sparsity is emergent (clipped ReLU + pairwise multiply).
Signal worth weighing: the strongest team took the free post-hoc lunch
and did not pay capacity for structured sparsity.

## VERIFIED AGAINST nnue-pytorch (user cloned it, 2026-07-25 ~22:5x)
`../nnue-pytorch/ftperm.py` (762 lines) — SF DOES do FT permutation, full
pipeline: `gather` activation matrix (~1M positions) -> `find_perm` ->
`eval_perm` -> apply via `serialize.py --ft_perm` (or one-shot
`--ft_optimize`). Directly portable; our 4-grouping is identical.
  * OBJECTIVE (`eval_act_mat`, :664): reshape (rows, L1/4, 4), `all()`
    over the group, maximize the FRACTION OF ALL-ZERO GROUPS.
  * SEARCH (`find_perm_impl`, :425): deterministic iterative local
    search, stage 1 pairwise swaps (`make_swaps_2`), stage 2 3-cycles
    (`make_swaps_3`) once swaps stall; <=50 rounds, a set of INDEPENDENT
    improving moves applied per round.
  * LAYOUT DETAIL TO COPY: they reshape to (rows*2, L1/2) — the two
    perspectives become separate rows and only L1/2 neurons are
    permuted, because the FT is shared. Ours is the same shape.

**THE BIGGER FINDING: SF'S TRAINER HAS NO SPARSITY PENALTY AT ALL.**
Grepped model/ trainer/ train.py for group-lasso / L1 / activation
penalties — nothing. Loss is purely eval-fitting (`calculate_sf_loss`,
model/nnue.py:39: weighted |pt-qf|^pow_exp + optional asymmetry). SF's
sparsity is 100% EMERGENT (clipped ReLU + pairwise multiply) and they
harvest it POST-HOC with permutation.
=> We pay ft_sparsity_lambda=0.001 on every leg as an "NPS lever", but
(a) our own A/B says density does not move NPS (57% vs 70% group
occupancy benchmarked identically), and (b) SF, running the SAME sparse-
L1 kernel, does not pay it. And that penalty is exactly what destroyed
the policy substrate in D47M when left unopposed.
**ACTION — CORRECTED (user pushback 2026-07-25 ~23:0x: "we can't assume
just because they don't have the L1 penalty that we shouldn't — they
aren't perfect and haven't tried everything").** SF's absence of the
penalty is an APPEAL TO AUTHORITY, not an argument; retracted as a
reason. The real state of the argument:
  FOR keeping it: the mechanism is real; sparsity may also REGULARIZE
  (entirely untested by us); and our NPS evidence is weak (3 positions,
  2 passes, under training load, CROSS-NET so confounded by different
  weights).
  AGAINST: the only benefit we can articulate does not show up in the
  one measurement we have; and the cost is demonstrated, not
  hypothetical — D47M is what this penalty does to the policy substrate
  when nothing opposes it.
  STRUCTURAL DETAIL THAT SHARPENS BOTH SIDES: the engine's find_nnz
  scans the POST-pairwise vector (`activate_ft` -> `ft:[u8;1280]` ->
  `find_nnz`, sota_head.rs:338-339) but our penalty/metric act on the
  PRE-pairwise accumulators. Post-pairwise density ~= pre-pairwise
  SQUARED, so our reported 0.27 is ~0.07 where the hardware skips.
  => the penalty has SQUARED LEVERAGE (argues FOR efficiency), but it
  also means D47M (~15% active groups) vs champion (~25%) was a 40%
  difference in real L1 work that produced ZERO measurable NPS delta
  (argues the bottleneck is the O(L1) find_nnz scan + non-eval node
  cost, not group count).
**VERDICT: "unjustified by evidence", NOT "wrong". Settle it with an
experiment, post-release:** two short legs from a shared warm point,
ft_sparsity_lambda 0.001 vs 0.0, comparing heldout value + screens +
the packed nets' NPS measured properly on an IDLE box. That tests
capacity, speed, and the regularization story at once. Do NOT change
mid-D47P.

## SPARSITY vs WIDTH: WHAT THE PROFILE MUST ANSWER (user Q, 2026-07-25 ~23:1x)
"Does the sparse incentive just make our network smaller? Is it an
argument for a smaller net?" — No, but the way it is no is actionable.
  * Sparsity = CONDITIONAL COMPUTATION, not a smaller net. 1280-wide at
    27% density and dense-345-wide do the same average L1 work, but the
    sparse-wide net has a far larger DICTIONARY and picks a different
    subset per position (endgame features dormant in middlegames). More
    expressive at equal average FLOPs.
  * BUT the equivalence covers only the L1 matmul. The INCREMENTAL
    ACCUMULATOR UPDATE costs in proportion to accumulator WIDTH,
    independent of density — sparsity buys nothing there.
    - sparsity: cheaper L1, same accumulator, MORE capacity
    - smaller net: cheaper L1 AND accumulator, LESS capacity
  * So "smaller net" only wins if L1 matmul is the bottleneck. Our
    evidence says it is NOT (D47M ~40% less real L1 work than the
    champion, identical NPS). Uncomfortable implication: if L1 is not
    the bottleneck, WIDTH (which also cuts accumulator cost) is the real
    speed knob — and width is exactly where capacity lives.
**=> WE ARE ARGUING FROM A PROFILE WE DO NOT HAVE.** One ~1h perf
profile (accumulator update vs find_nnz scan vs L1 matmul vs L2/L3 vs
non-eval node cost) settles THREE open questions at once: the
ft_sparsity penalty, the FT permutation payoff, and the width question.
RECOMMENDATION: pull this profile EARLY in the optimization phase — it
is the measurement three decisions are blocked on.

## WHY SPARSITY DOES NOT HELP THE ACCUMULATOR — AND THE VERSION THAT DOES
(user Q 2026-07-25 ~23:2x: "seems like we could take advantage of it")
  * The accumulator holds PRE-activation i16 sums; sparsity is a
    POST-activation property. An entry at -800 contributes 0 after the
    clipped ReLU but must still be maintained exactly — the next feature
    delta may push it positive. You cannot skip maintaining a value
    whose future sign is unknown.
  * Conditional skipping would cost MORE than the work saved: the update
    is `acc[i] +/- W[f][i]` over a contiguous 1280-elem i16 array (~40
    AVX2 instructions, pure sequential SIMD). A per-element margin test
    kills vectorization and needs per-entry bookkeeping.
**THE VERSION THAT WORKS: dead-neuron (structured) pruning.** What cuts
the accumulator update is WEIGHT sparsity — neurons dead across ALL
positions. If accumulator entry i never fires, its whole weight column
is useless: delete it, shrink the accumulator, save on BOTH the update
AND the L1 matmul, at ZERO capacity cost.
**HYPOTHESIS WE HAVE NEVER TESTED: our ft_sparsity penalty may already
be producing always-dead neurons.** L1 penalties are known to kill units
outright rather than merely quiet them. If ~10% of our 1280 entries
never fire, that is a free 10% width cut targeting the cost that
actually dominates — and it REFRAMES the penalty entirely: not an
L1-skip lever (which our NPS data says is worthless) but an implicit
STRUCTURAL PRUNER (which targets the real bottleneck).
**FREE TO MEASURE:** the activation-matrix gather needed for the FT
permutation yields per-neuron firing rates as a byproduct. ONE pass
gives us the permutation AND the dead-neuron count. Do it in the same
work item.

## ~~EVAL CALIBRATION FINDING~~ — RETRACTED, SEE BELOW (2026-07-25 ~23:5x)
An earlier version of this section claimed our eval was COMPRESSED ~0.59x
vs SF and that cp-denominated margins were therefore ~1.7x too wide.
**THAT WAS WRONG — two compounding errors:**
  1. Misread the regression direction. `polyfit(eval, target)` slope
     0.587 means target ~= 0.587*eval, i.e. our evals are ~1.7x LARGER
     than the labels, not smaller.
  2. Compared our output against SF's **UCI cp** when our margins live in
     SF's **INTERNAL** units. SF internal = 3.28 x UCI cp
     (uci.h:38 NormalizeToPawnValue=328).

**WHAT THE DATA ACTUALLY SHOWS — WE ARE ALREADY WELL CALIBRATED.**
Material probes (ARCHON static vs SF18 static, white-POV):
    up a pawn   ARCHON  321   SF18 154 cp    (our PAWN_VALUE   208)
    up a rook   ARCHON 1326   SF18 498 cp    (our ROOK_VALUE  1276)
    up a knight ARCHON 1118   SF18 480 cp    (our KNIGHT_VALUE 781)
A rook advantage reads 1326 against our own ROOK_VALUE of 1276 — our
eval and our search constants are on the SAME scale, as intended.
Median |value| over 1108 positions: labels 92.0, SF18 static 82.5 UCI cp
(x0.90 of labels), ARCHON 181.5 (x1.97 of labels). Converting SF18 to
internal (x3.28) gives 270.6, so ARCHON/SF18-internal = 0.67 — same
frame, ~1.5x apart, and our margins were tuned in OUR frame through
Waves A-G. No action indicated.

**WHAT SURVIVES AND IS STILL USEFUL:**
  * SF16 vs SF18 static differ in scale by 0.893 — there is NO canonical
    "SF cp scale"; SF changes NormalizeToPawnValue between versions and
    retunes. Matching an external scale is a category error; only
    INTERNAL consistency matters, and SPSA measures that empirically.
  * Both SF versions have near-identical static accuracy (MAE 63.5 vs
    64.3 on our 1201-position set) despite different scales => scale
    does not affect ranking quality.
  * THE ACCURACY COMPARISON ITSELF STANDS and is the real result: after
    an affine fit, ARCHON 66.1 MAE vs SF18 62.4 (correlation 0.913 vs
    0.918; our SIGN agreement is BETTER, 82.3% vs 79.5%). Our static
    eval is within ~6% of SF18's on SF18-search targets. That is a
    genuinely strong position and worth keeping in the write-up.
  * tools/nnue/rescale_net_header.py (commit e56fd86) still works and is
    a cheap 1-D probe of collective margin tightness IF we ever want it
    — but it is NOT a correction of a known error, and SPSA over the
    margins is the principled version. Deprioritized.

METHOD LESSON: check regression direction, and never compare against an
engine's UI-normalized units when the question is about internal ones.
Two sign/frame errors in the same direction produced a confident wrong
answer that survived until a 3-position material probe killed it.

## [SUPERSEDED — read the retraction above] original entry, kept for the record
## EVAL CALIBRATION (2026-07-25 ~23:3x) — user Q: "how does our
## value loss compare to SF18's?"
Ran SF18's STATIC eval (`eval` cmd, "Final evaluation", white-POV ->
flipped to stm) over the same 1201 SF18-search-labelled positions we
score our nets on. 1108 usable (93 in-check return "none").

  net                raw MAE   RESCALED MAE   slope   pearson
  SF18 static           64.3        62.4      1.146    0.9175
  ARCHON d47m sb60     135.1        66.1      0.587    0.9131
  ARCHON d47h champ    137.1        67.2      0.582    0.9059
  ARCHON d47j joint    138.2        69.3      0.580    0.9026
  (RESCALED = after optimal affine a*eval+b fit to the target)

**VIRTUALLY THE ENTIRE EVAL-ACCURACY GAP VS SF18 IS CALIBRATION, NOT
KNOWLEDGE.** Correlation is near-parity (0.913 vs 0.918) and our SIGN
agreement is BETTER (82.3% vs 79.5%); rescaling closes 135->66 vs SF18's
62. SF18 gains only 1.9 from rescaling (already calibrated); we gain 69.
The slope is IDENTICAL across d47h/d47j/d47m (0.580-0.587) => baked into
the TARGET DEFINITION, not a per-net accident.

MECHANISM: training target = sigmoid(score/DEFAULT_SCALE=400) blended
30% with outcome (DEFAULT_WDL=0.3, spec.py:61-62); engine maps net
output back to cp with **network_scale = 380** (sota_head.rs:431,
header offset 44). Empirically our cp land at 0.587x SF18's cp scale —
so the mapping constant is ~1.7x too small RELATIVE TO SF'S SCALE.
(Most likely our training labels' score scale differs from SF18's
win-rate-normalized cp; the net faithfully reproduces ITS labels.)

**WHY IT MAY COST ELO:** our pruning machinery is cp-denominated
(futility, RFP, delta, aspiration) and those constants were ADAPTED
FROM STOCKFISH, i.e. tuned against SF's cp scale. At 0.59x, every margin
is effectively ~1.7x wider than designed — we prune systematically less
than intended, and no eval improvement fixes it.
**CAVEAT:** Wave A-G tuning may have already partially adapted the
constants to OUR scale, in which case "correcting" the scale would break
that calibration. Unknown — must be tested, not assumed.
**EXPERIMENT (cheap, ONE constant):** A/B network_scale 380 vs ~650.
Visible at FIXED NODES too (pruning changes reshape the tree). Comprehensive
SPSA would eventually find this, but finding it FIRST means TM-SPSA and
light SPSA tune against a correctly-scaled eval instead of compensating
for a miscalibration. PROPOSED SLOT: right after the duel, before TM.

## ★ V2 NET PROGRAM — FROM SCRATCH WITH THE SF-TRAINER LESSONS
(user ruling 2026-07-25 ~23:5x, after the nnue-pytorch study
`stockfish_reference/09-nnue-pytorch-trainer-study.md`)
"All these make sense for us to implement as a SECOND RELEASE NET where we
START FROM SCRATCH." Correct framing: most of these are from-scratch
changes (data filters, loss shape, quant scales, init) that cannot be
retrofitted into the D47 warm chain without voiding its comparability.
**This is a post-v1 program. v1 ships from the D47P/D47M line.**

INCLUDED (from the study's ranked list):
 * per-tensor weight clips matched to export encoding (drop on f32
   tensors, widen on i16 FT, ±1.98 only for i8@64) — MEASURED: our
   uniform clip is saturating l0_pstw/l0_threatw/polmovew/polmoveb today
 * mate/TB label REMAP (4000 + 4000·0.85^ply) instead of hard-dropping
   |score|>10000 — recovers the "very winning" supervision we delete
 * group lasso over the deploy 4-groups replacing flat activation L1
   (current penalty is misaligned twice: charges per activation not per
   group, and penalizes magnitude ⇒ capacity tax)
 * position-level shuffle buffer (~1M) replacing frozen contiguous
   windows — our batch composition is currently fixed for the life of a
   corpus, on every leg, forever
 * WLD-coherence probabilistic thinning + soft ply ramp replacing the
   hard ply>=16 cut; random_fen_skipping for intra-game decorrelation
 * threat quantization 64 → 128/255 + training-time clip on l0_threatw
 * grad-norm clip 2.0, NaN/Inf abort, stop decaying biases
 * λ ramp (wdl 0 → 0.3), pow_exp 2.5
 * exponential LR decay for chained legs (schedule-length-agnostic)
 * identical init across layer-stack buckets; HCE-value PSQT init
 * fit our own eval→WDL scaling from data (their perf_sigmoid_fitter
   approach) instead of inheriting 400
UNCERTAIN, ON THE LIST ANYWAY (user: "I'm not sure about the draw aware
double sigmoid, but lets add it to the list"):
 * draw-aware DOUBLE SIGMOID target `0.5(1+σ(q)−σ(qm))` with independent
   in/out scaling (SF: 340/380, net×nnue2score 600) replacing plain
   σ(score/400). Run as a paired screen — it changes the loss scale, so
   heldout stops comparing to the incumbent chain.
DECLINED / DEPRIORITIZED:
 * **Concurrent UCB gauntlet — user: "I'm not sure I want to slow down
   training that much. I doubt it makes a big difference."** NOTE the
   objection targets the wrong cost: we ALREADY run 100g screens every
   30sb on CPU during training at no measurable GPU cost. The actual
   upgrade is only (a) SHIP THE ARGMAX rather than the final/EMA
   checkpoint, and (b) UCB — concentrate EXTRA games on leading
   checkpoints instead of spreading evenly (same budget, better spent).
   Our ±30 screens are too noisy to select on naively, which is exactly
   what UCB fixes. **User accepted this reframing ("fair about the
   gauntlet").** REVISED VERDICT: the concurrency objection is moot (we
   already pay it, for free); what stays on the list is the cheap half —
   SHIP THE ARGMAX of checkpoint screens rather than the final/EMA net,
   with UCB concentrating extra games on the leaders. Modest EV, zero
   extra GPU cost, folds naturally into settle-battery practice. Not a
   priority; do it when a settle battery next needs designing.
 * RangerLite/PNM/Lookahead (EMA fills the role), SWA, L1-layer
   factorization, adaptive piece-count rebalancing, their head topology
   ([x²,x] concat + linear bypass — interesting but needs new SIMD +
   deploy format + full retrain), PP_3Wide.

## ★ EVAL_BENCH — HOW OUR NET ALONE STACKS UP (tool: tools/nnue/eval_bench.py)
Built 2026-07-25 at user request ("make this a reusable style test... a
powerful indication of how a net alone stands up to others"). Scores STATIC
evaluators against a deep-search oracle, independent of search and of each
engine's arbitrary units. Pluggable adapters (archon/sf/reckless, each
declaring POV), content-hash caching of positions/labels/evals, bootstrap
95% CIs, p90/p99 tails, worst-offender dumps, --labels for bulk-labelled
corpora, and a guard that shouts when all evaluators correlate <0.30.

**METHOD (why RESCALED is the comparable column):** every engine reports in
its own units — a rook advantage reads ~1326 ARCHON / ~1283 Reckless / ~498
UCI cp SF18 (SF's UCI cp is win-rate-normalized and even differs SF16 vs
SF18, slope 0.893 between them). So we fit least squares
`target ≈ a·eval + b` and report MAE of the rescaled prediction; `a` is the
multiplier onto the oracle's scale (a<1 ⇒ that engine's evals are LARGER).
All series converted to stm-relative; clipped ±800cp.

**RUN 1 — 2999 positions from OUR OWN GAMES, SF18@50k oracle:**
ARCHON d47m 69.9 [67.1,73.1] ≈ SF18 71.1 [68.2,74.3] < SF16 76.9 ≈ Reckless
77.7. Read at the time as "parity with SF18".
**RUN 2 — 20000 positions from the canon corpus (SF18 relabel), the
robustness check: THE PARITY WAS HOME-FIELD ADVANTAGE.**
  SF18            61.9 [60.8,63.0]  pearson .9278  sign 81.6%  p99 400.7
  ARCHON d47m     68.2 [67.1,69.3]  pearson .9192  sign 84.2%  p99 370.8
  SF16            68.7 [67.4,69.9]  pearson .9059  sign 81.5%  p99 448.3
  ARCHON d47h     69.4  / d47p90 69.7
  Reckless        91.5 [90.2,92.9]  pearson .8676  sign 83.5%  p99 438.1
=> On a broad distribution SF18's static eval is ~10% better than ours (CIs
clear). **We are TIED WITH SF16 and clearly ahead of RECKLESS.** But ARCHON
has the BEST SIGN AGREEMENT of any evaluator (84.2% vs SF18 81.6%) and the
BEST p99 TAIL (370.8 vs 400.7): we get direction right more often and blow
up less on hard positions; SF18 beats us on fine magnitude in the bulk.
That is a specific, actionable profile for where eval work remains.

**INCIDENT + PERMANENT LESSON: canon-v1 `corpus.quiet.txt` labels are
WHITE-POV, not stm-relative.** Read as stm, every evaluator collapsed to
pearson ~0.04 / sign ~47% (worse than chance). Flipping on black-to-move
restored pearson .9192. Added `--labels-white-pov` + the <0.30 guard.
**OPEN ITEM (assumption, not verified):** our trainer consumes the BINARY
path, not this text dump, and the nets plainly work — so the .txt→.bin
conversion must apply the flip. That is now an explicit assumption worth a
5-minute verification rather than a silent belief.

**CAVEATS the tool prints every run:** own-game positions flatter our nets;
a single-engine oracle flatters that engine's static eval; and this metric
ordered four ARCHON nets spanning 86 Elo WRONG — it answers "is this eval in
the right class", never "is this net stronger".

## ★★ WHERE SF18 BEATS US — AND WHY IT MOSTLY DOESN'T (2026-07-25 ~00:3x)
Conditional error analysis on the 20k canon set, both evaluators affinely
rescaled onto the oracle first (scale-free, about SHAPE).

**FIRST: "their net is bigger" is FALSE.** stockfish18's binary is 107.8 MB
with BOTH nets embedded (EvalFile + EvalFileSmall); ours is a 1 MB binary +
133 MB .cenn. Same ballpark, ours likely larger. (An earlier claim that SF
nets are ~3x smaller was WRONG — it measured SF15.1 and our older SF clone,
not SF18. Retracted.)

**WHERE THE GAP LIVES (all positions, clip ±800):**
  phase: endgame <8 pieces  ARCHON 48.4 vs SF18 49.0 = **we win**
         8-11  +10.8 | 12-15 +11.8 | 16-19 +8.3 | 24+ ~+2
  one queen on board (asymmetric)      +20.1   (n=1050)
  material imbalance 4+ pawns          +24.6   (n=2129)
  3+ passed pawns                      +11.9

**THEN THE RESTRICTION THAT CHANGES EVERYTHING (user's call: "I wonder if
it even matters to be good at those, or just good enough... I almost like
our decision to exclude them"):**
  all positions      n=20000  ARCHON 68.2  SF18 61.9   gap **+6.3**
  |label| <= 400     n=16780  ARCHON 52.1  SF18 50.0   gap +2.1
  |label| <= 200     n=13095  ARCHON 35.6  SF18 36.5   gap **-0.9**
  |label| <= 100     n= 9519  ARCHON 22.1  SF18 23.2   gap **-1.2**
  SIGN AGREEMENT: ARCHON WINS AT EVERY LEVEL, widening as positions close
  (81.1/80.9 → 78.4/77.2 → 73.8/72.0 → 66.7/63.9).
  And material-imbalance 4+ pawns restricted to |label|<=400 (the
  COMPENSATION cases — sacrifices, fortresses, genuinely hard): **ARCHON is
  15.2 cp BETTER than SF18** (n=676).
=> **SF18's entire advantage is in ALREADY-DECIDED positions.** In the range
where the number can change a move, WE BEAT SF18. The "10% behind" headline
was an artifact of blowout positions dominating unrestricted MAE.

**CONSEQUENCES:**
 * The training decision to clamp (±2499) and drop (|raw|>10000) blowouts
   looks CORRECT — capacity went to the discriminating range.
 * **V2 item 6 (remap mates/TB instead of dropping) is DOWNGRADED** from
   near-top to low priority. Its rationale was "we truncate supervision
   where we are weak"; that theory is dead — the excluded positions are not
   even in this test set (canon labels clamp at 2499, nothing >10000), and
   blowout precision demonstrably does not help. Only revisit if remapping
   is argued to teach TRANSFERABLE concepts.
 * ERROR STRUCTURE: corr(ARCHON err, SF18 err) = 0.747 vs corr(SF16, SF18)
   = 0.813 — our mistakes are LESS SF-like than SF16's. A 50/50 rescaled
   ensemble of ARCHON+SF18 scores 60.6 MAE, BEATING SF18's 61.9 alone. We
   hold genuinely complementary knowledge.
 * Remaining honest caveat: static-eval accuracy still does not rank nets
   by playing strength (the 86-Elo null). This says our eval is well-SHAPED
   for the decisions search faces, not that we would win the match.

## ★★ NET SIZE / ARCHITECTURE — SETTLED FROM THE BINARY (2026-07-26 ~00:5x)
User: "chase down who's net is actually bigger and why. I thought SF18 had
a much bigger accumulator." Read from stockfish18's mangled symbols
(`FeatureTransformerIXT_EE` instantiated at `Lj1024E` and `Lj128E`,
HalfKAv2_hm) — not inferred from file size (SF nets are leb128-compressed,
so bytes lie):

              ARCHON      SF18 big    SF18 small
  L1 (acc)      1280         1024         128
  L2             128           32          -
  L3              32           32          -
  HalfKAv2_hm  22528        22528       22528
  FullThreats  60720     yes(refined)     -
**OUR ACCUMULATOR IS 25% WIDER THAN SF18's AND OUR L2 IS 4x THEIRS.** We
are bigger in every dimension. "Their net is bigger" is FALSE (and my
earlier "SF nets are ~3x smaller" was also wrong — that measured SF15.1
and our stale SF clone; both claims retracted).

**THE REASON THE ASSUMPTION FELT TRUE — and the real finding.** SF's
architecture history (docs/nnue.md:2995-3121): SFNNv5 1024 → v6 1536 →
v7 2048 → v8 2560 → **v9 3072**, then **SFNNv10: "Added FullThreats input
features, L1 size reduced to 1024, and used 255 as the FT quantization
scale."** SF scaled width to 3072 and then TRADED TWO-THIRDS OF IT AWAY to
buy threat features. v11 removed piece-to-king threats, v12 removed
king-to-piece threats, v13 raised L2 to 32, v14 added opposed-pawn
features, v14.1 requantized, v16 added PP_3Wide.
=> **Independent validation of our D40 FullThreats bet** — SF made the same
call, harder. It also says the frontier is now FEATURE REFINEMENT, not
width: every version since v10 changes the threat set, never L1.

**CAPABILITY WE LACK: SF18 ships TWO nets with a switching policy** (big
1024 for close positions, small 128 for already-decided ones). A speed
lever we have no equivalent of — and a measurement caveat: SF's `eval` may
have been scoring the SMALL net in exactly the lopsided positions where it
"beat" us.

**NEUTRAL-SET ROBUSTNESS CHECK (1960 positions from SF18-played UHO book
lines, freshly labelled SF18@50k — no net trained on these):**
  Reckless 47.8 [45.8,49.8] | SF18 49.5 [47.2,51.8] | ARCHON 51.1
  [48.8,53.3] | SF16 61.8 [59.4,64.6]
Top three statistically indistinguishable; SF16 clearly behind. **The canon
finding that we WIN the discriminating range did NOT clearly replicate
here** — treat it as promising-but-not-robust, not banked. Note these are
quiet SF-chosen positions, a different distribution again.

## ★★ SMALL-NET CONFOUNDER RESOLVED + THE REGIME-SCALE FINDING (2026-07-26 ~01:1x)
User: "I suspect that small net is a confounder. It is hyper specialized and
probably good at the imbalanced ones." Read SF18's EXACT rule from the live
clone (`git show sf_18:src/evaluate.cpp:43-72`; the clone's master is at
ebcea3ef/dev-20260716 — HEAD was merely detached at sf_16):
    simple_eval = PawnValue*(pawns_stm - pawns_opp) + npm_stm - npm_opp
    use_smallnet = |simple_eval| > 962            (PawnValue=208 => ~4.6 pawns)
    ...and if the small net then reports |nnue| < 277, SF RE-EVALUATES with
    the big net. So small-net use is lopsided-material AND confirmed-decisive.

**SPLIT BY THAT EXACT RULE:**
  CANON 20k (small net on 2052 = 10.3%)
    all                       n=20000  ARCHON  68.2  SF18  61.9  gap +6.3
    SF BIG net                n=17948  ARCHON  55.7  SF18  53.6  gap +2.1
    SF SMALL net              n= 2052  ARCHON 136.8  SF18 132.5  gap +4.3
    BIG + |label|<=400        n=16103  ARCHON  48.5  SF18  44.5  gap +4.1
    BIG + |label|<=200        n=12737  ARCHON  34.1  SF18  33.7  gap **+0.4**
  NEUTRAL 1960 (small net on 44 = 2.2%)
    all                       n= 1960  ARCHON  51.1  SF18  49.5  gap +1.7
    SF BIG net                n= 1916  ARCHON  49.1  SF18  46.3  gap +2.8
    SF SMALL net              n=   44  ARCHON  94.7  SF18 108.0  gap -13.3
    BIG + |label|<=400        n= 1915  ARCHON  46.6  SF18  45.8  gap +0.7
    BIG + |label|<=200        n= 1801  ARCHON  40.0  SF18  41.9  gap **-1.9**

**VERDICT ON THE HYPOTHESIS: half right.** The small net is NOT beating us —
in those positions both engines are bad (136.8 vs 132.5) and on the neutral
set WE beat it. What lopsided positions actually do is DOMINATE the
unrestricted mean with huge errors.
**CORRECTION TO AN EARLIER CLAIM:** I said the "we win the discriminating
range" result did not replicate on neutral positions. That was the
UNRESTRICTED comparison. Restricted properly (SF big net, |label|<=200) it
DOES replicate: canon +0.4 (tied), neutral **-1.9 (we win)**.

**★ NEW, ACTIONABLE: OUR EVAL IS MORE REGIME-DEPENDENT THAN SF18's.** The
overall canon gap is +6.3, but the weighted average of the two sub-gaps is
only ~+2.4. The difference is the AFFINE FIT: when one global (a,b) must
serve every regime, WE LOSE MORE THAN SF18 DOES. Fit each regime separately
and most of our deficit evaporates. So our eval→target map is less
uniformly linear across regimes, not merely differently scaled.
=> Points straight at the loss/target design: our single fixed
`scale=400` sigmoid vs SF's INDEPENDENTLY TUNED in_scaling 340 /
out_scaling 380 / nnue2score 600, and their DRAW-AWARE DOUBLE SIGMOID
`0.5(1+σ(q)−σ(qm))` with the ±270 offset — which is exactly a mechanism for
making the eval↔target relationship uniform across regimes.
**This PROMOTES the double-sigmoid item in the V2 program** from "uncertain,
on the list anyway" to a specific, motivated hypothesis with a measurable
prediction: it should flatten the regime-dependence of our affine fit.
Cheap pre-test available before any retrain: check whether a per-regime
affine fit still beats a global one after the change.

## ★★★ TARGET-TRANSFORM CALIBRATION — FITTED TO OUR OWN DATA (2026-07-26 ~01:3x)
User: "lets test [the double sigmoid]... my instinct is we can do EVEN
better than them." Confirmed. Fitted the expected-score curve on **3M
records of uho-band75-100** (our largest training corpus; bulletformat
carries the real game `result`, unlike canon whose wdl column is all 0.5).

  transform                                        curve MAE
  SF's double sigmoid, offset 270 / scale 340        0.1147
  OUR CURRENT single sigmoid, scale 400              0.1183
  best single sigmoid, scale 148                     0.0148
  **double sigmoid fitted to us, offset -162.5 / scale 91   0.0057**

**(1) OUR scale=400 IS BADLY MISCALIBRATED FOR OUR OWN DATA.** At score
+250 our target says expected score 0.65; the empirical curve says 0.80.
Best single sigmoid is ~148 — our target curve is ~3x too flat.
**(2) SF's CONSTANTS ARE NO BETTER FOR US** (0.1147 vs our 0.1183) —
copying 270/340 would buy nothing. Fit our own.
**(3) THE DOUBLE-SIGMOID FORM DOES EARN ITS PARAMETER** (0.0057 vs 0.0148,
2.6x) — BUT the fitted offset is NEGATIVE, i.e. our data wants a curve
STEEPER than a logistic, the OPPOSITE of SF's draw-band shape. At a<0 it is
no longer a valid WDL decomposition (P(win)+P(loss)>1 near zero); it is a
flexible 2-param shape. For a future real WDL head, constrain offset>=0 and
accept the worse fit.

**★ POSSIBLE LIVE TRAINING DEFECT:** we blend 30% game outcome with 70%
σ(s/400). If that sigmoid says 0.65 where the outcome distribution says
0.80, the two halves of the target DISAGREE SYSTEMATICALLY on every sample
— a confused gradient present in every net we have trained. Worth an
explicit experiment: (a) refit scale to ~148, (b) fitted double sigmoid,
(c) wdl=0 control. Cheap: short legs from a common warm point, compare
heldout + screens.
CAVEAT: the corpus is quiet- AND band-filtered, so P(result|score) is
selection-affected, not natural. For setting a TRAINING target that is
arguably the right conditional — it is the distribution the net sees.
=> Promotes the loss-shape items to the top of the V2 program, and makes
`perf_sigmoid_fitter`-style empirical fitting a prerequisite, not a nicety.
