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
- [ ] Wire `jax/tools/quant_byte_parity.py` + `engine_parity.py` into the
      gate-script family (every future net passes them pre-game).

## Build-window queue B (cargo allowed; same window)
- [ ] D40_CONVERT dump mode in d40_features.rs (band-filter minus band =
      full-corpus sfbinpack→bulletformat converter; study sized it ~5
      lines) → convert fishpack32 (5.6GB) as the first JAX-native corpus.
      **MUST include a shuffle pass** (data.py's contract: contiguous
      windows at shuffled offsets over a PRE-SHUFFLED corpus — the
      *.shuf.bin convention; game-ordered input would correlate batches).
- [ ] PyO3 loader crate first build (`jax/rust-loader-design/` skeleton):
      exposes movegen/legal-mask + activation dumps at Rust speed (fixes
      the one below-reference loader component, 10k→fast legal masks).
- [ ] Top-k logit print + multi-head sidecar loader extension (policy
      roadmap build-window items — same window, shared rebuild).

## Phase 3 — shadow training (needs ~8h GPU; schedule after gauntlet
      results digested)
- [ ] Full 240sb D47c-recipe twin on JAX-GPU. Gates: netpack parses,
      scale check PASS, engine oracle ≤5cp p50, equal-time SPRT vs the
      bullet twin, D38 ledger row.

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
