# JAX-GPU sparse-FT kernel: gate PASSED at 1.07x bullet (2026-07-23)

**Verdict (study §8 line): 225,400 pos/s full train step at batch 8192 =
1.07x bullet (~210k) — GO; the 0.8x gate (168k) is cleared with margin,
and >1.0x means the judgement band doesn't even apply.**

Hardware: RTX 4060 Ti 8GB (Ada sm_89), jax 0.11.0 / optax 0.2.8 /
numpy 2.5.1 (`jax/pyproject.toml` pins), `jax[cuda12]`,
`XLA_PYTHON_CLIENT_PREALLOCATE=false`. Shape: the exact D47c recipe —
HalfKAv2_hm 22,528 + FullThreats 60,720 -> HL 1280 x 2 perspectives, policy
head on, bullet-AdamW, real heldout rows (`data/nnue/heldout/uho-band0-5.bin`),
8 distinct batches cycled.

## 1. The problem (recap of the 2026-07-22 finding)

`model._ft_accumulate`'s `jnp.take -> mask -> sum` makes XLA materialize a
B x S x 1280 f32 intermediate (threats: 6.7 GB at batch 8192): OOM at batch
>= 2048, and 9.5k pos/s (0.05x) at 1024. A memory-materialization problem,
not a compute one — exactly the pre-registered sparse-gather risk.

## 2. Routes tried, with numbers

### Route 1 — Pallas-Triton embedding-bag + scatter-add backward (WINNER)

`jax/archon_jax/ft_kernel.py`: `ft_accumulate_pallas` (custom VJP).

- **Forward**: grid `(HL/block, 2B)` — stm+ntm stacked along batch so one
  launch serves both. Each program accumulates its column block in
  registers over the bag's rows; nothing is ever materialized. Indices are
  front-packed with -1 padding (a features.py invariant), so per-bag
  *dynamic* trip counts skip padding entirely.
- **Backward**: `dtable[idx] += dacc` via f32 `atomic_add` into the two
  zero-initialized dense grad buffers (the exact tensors the optimiser
  consumes; bullet's CUDA backward is the same construction). dpstb is an
  XLA column-sum. Atomics reorder float adds -> run-to-run noise ~1e-4 on
  grads, same property bullet has.
- Two jax-0.11 API landmines, recorded for the next person:
  `pl.load/pl.store` are gone (ref indexing + `plt.load/plt.store/
  plt.atomic_add` from `jax.experimental.pallas.triton`); passing
  `plt.CompilerParams` is what routes `pallas_call` to the Triton backend
  (default is Mosaic-GPU, which tries to stage the 426 MB tables into
  SMEM and dies). Consumer device kinds are not on the Triton backend's
  whitelist — `ft_kernel.register_gpu()` fills the public
  `gpu_info.registry` hook from the device's own compute capability
  (4060 Ti -> 8.9/89, identical arch to the whitelisted L4/4090).
- Tuning: block x warps swept {64,128,256} x {1,2,4,8} standalone and
  end-to-end; end-to-end spread was ~2% (the step is no longer
  FT-dominated). Defaults 128 x 2 (`ARCHON_JAX_FT_BLOCK/_WARPS`).

### Route 2 — pure-XLA reformulation (scan over slots): measured dead

`ft_accumulate_scan` (`lax.scan`, one (B,1280) gather+add per slot; the
backward autodiffs to S sequential scatter-adds): **11.7k pos/s at batch
1024** (barely above the 9.5k reference), **OOM at >= 4096** — XLA's
autotuner materializes 3.1 GiB reduce fusions in the backward. The
`segment_sum` variant of this route is strictly worse by construction: it
*explicitly* materializes the (B*S, 1280) gather the reference path dies
on. Kept in-tree as the portability fallback and the honest baseline.

### Route 3 — bf16 gathers: not needed, deliberately skipped

The gate passed in pure f32 with 2.9 GiB peak and ~2x headroom to DRAM
bandwidth on the real access pattern. bf16 table reads would risk the
1e-3 parity bound and the +-1.98/i8-quant contract for a speedup the
verdict doesn't need. Revisit only if a future arch (HL >> 1280 or dense
threat encodings) moves the FT back to the critical path.

## 3. The numbers (tools/gpu_step_bench.py mode=report, 40 timed steps)

Train step (fwd + bwd + bullet-AdamW), pos/s:

| batch | reference (old) | scan | pallas |
|------:|----------------:|-----:|-------:|
| 1024  | 9,470 | 11,654 | 60,501 |
| 4096  | OOM | OOM | 163,555 |
| 8192  | OOM | OOM | **225,400** |

- Soaks at 8192: 224,374 pos/s over 400 steps, 231,350 over a separate
  200-step run (run-to-run range ~224-231k) — no fade, no leak; peak host
  RSS 1.6 GB for the whole process; GPU back to idle after.
- **Forward-only at 8192: 1,257,226 pos/s** (the future GPU
  datagen/eval/distill number).
- Peak GPU memory through the 8192 run: **2.93 GiB** of 8 GB (headroom for
  wider nets or batch 16k+).
- Host loader unchanged: ~210k pos/s single thread mapping — now the
  next bottleneck at scale (PyO3 loader remains the queued fix).

## 4. Parity evidence (the hard gate)

- **Forward parity** (512 real heldout rows, perturbed full-arch params so
  BOTH FT tables contribute; GPU pallas vs the CPU reference forward that
  is byte-faithful to production): max |d value_logit| = **3.53e-04**
  (< 1e-3 gate, PASS); **policy argmax flips 0/512**. The scan impl shows
  the identical 3.53e-04, i.e. the residual is GPU-vs-CPU float ordering
  in the shared head, not the kernel.
- **Training-step equivalence** (6 real optimiser steps, batch 1024, same
  init + data, reference vs pallas): losses identical to 6 decimals for
  the first 4 steps, worst relative delta 1.6e-04 by step 6 — pure float
  noise from atomic add ordering (bullet's backward is equally
  nondeterministic).
- Unit tests: `jax/tests/test_ft_kernel.py` (scan tested everywhere;
  pallas fwd+grad vs reference + empty-bag edge under
  ARCHON_JAX_ALLOW_GPU=1). Full suite: 11 passed on GPU; 9 passed +
  2 skipped in the CPU-only default.

## 5. Density honesty note

The brief's "62% avg slot density" was wrong for real data: the heldout
averages **26.5 active rows/perspective** (p50 21, p95 62, max 94 of 192
capacity) and the trainer-parity-filtered hardneg corpus averages **29.6**
— quiet filtered positions just don't carry many threats. So the bench
density IS the recipe density, and bullet's ~210k was measured on the same
data family (its kernel time scales with density too): the 1.07x ratio is
apples-to-apples. Synthetic worst case, which no chess position produces
(all 192 slots filled): 44.0k pos/s skewed-index / 38.4k uniform — even a
hypothetical corpus at the real p95 density (~62 rows) interpolates to
~120k = 0.57x, still above the 0.5x park line. At any real corpus density
measured, the verdict is GO.

## 6. What changed (all additive; CPU path untouched and default)

- `jax/archon_jax/ft_kernel.py` — the kernels + custom VJP + GPU registry
  shim (new).
- `jax/archon_jax/model.py` — `forward(..., ft_impl="reference")` dispatch
  (`_ft_accumulate_pair`); reference path byte-identical, still default
  everywhere; GPU impls are two-table only (halfka_only falls back).
- `jax/archon_jax/train.py` — `make_update_step(..., ft_impl=None)` reads
  `$ARCHON_JAX_FT_IMPL` (default "reference").
- `jax/tools/gpu_step_bench.py` — same CLI, new `impl=` / `mode=report`
  tokens (batch sweep old-vs-new, forward-only, parity, step equivalence,
  peak memory). Original invocation reproduces the original 9.5k baseline
  exactly (9,470 measured post-refactor).
- `jax/tests/test_ft_kernel.py` — kernel unit tests (new).

## 7. Consequences / next

- Phase 3 (shadow training, the full 240sb D47c twin on JAX-GPU) is
  unblocked: 240 x 3040 x 8192 at 225k pos/s ≈ 7.4 h — the ~8 h window the
  plan budgeted. Then netpack -> scale check -> engine oracle -> SPRT per
  §8 gates 2-4.
- The single-thread host mapper (~210k pos/s) is now at parity with the
  GPU step consumer — it must be overlapped (data.py already threads) and
  the PyO3 loader keeps its priority for policy-mask runs.
- GPU forward at 1.26M pos/s opens the cheap-eval doors early: held-out
  instruments per superbatch, activation stats, distill targets.
