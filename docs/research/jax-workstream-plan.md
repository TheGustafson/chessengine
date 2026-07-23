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
- [ ] **S3 checkpoint/kill/resume**: 2sb → ckpt → kill → resume 1sb;
      metrics continuous, params within atomic-noise envelope of a
      straight-through 3sb run.
- [x] **S4 export chain — GREEN (01:5x, on the JAX-trained S2 sb5):**
      netpack → .cenn (136.8MB) → engine loads → scale-check PASS
      (median 0.871 in band, sign 97.1% vs incumbent) → engine oracle
      p50 1cp / p90 4cp / 99% sign (gate ≤5cp p50; tail = the known
      ≤7-piece quant-edge pattern). Candidate engine for 1280 nets =
      engines/bin/archon-d47 (width is compile-time; target/release
      is a 768 build — same fact the crown-rematch protocol flagged).
- [x] S0 banked: kernel parity (3.53e-4, 0/512 flips), 400-step soak,
      corpus smoke at 4 offsets, 6-sig-fig loss reconciliation.
- [ ] **S5 sparsity-λ probe (pre-launch)**: 2×3sb warm at λ∈{1e-3,3e-3};
      pick the strongest λ with flat value component; density metric
      live in tb thereafter.
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
