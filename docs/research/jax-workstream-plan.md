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
