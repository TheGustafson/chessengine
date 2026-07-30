# Perf study 25 — the NNUE SIMD kernels and head, on ARCHON's own terms

**Date:** 2026-07-29/30 · **Box:** 7700X (Zen 4), AVX-512 VBMI2, 32 KB L1i /
32 KB L1d, 1 MB L2/core, 32 MB L3, THP `always` · **Net/arch:** D48X,
`sota,arch-d48x`, ACC 1280 · **Base commit:** `cf7cf46`

**Scope:** `src/eval/nnue/{sota_simd,sota_head,sota_loader,sota_features}.rs`.
Three sibling studies own the accumulator/context (22), search/movepick (23) and
threats/board/movegen (24); nothing outside the four files above is touched.

**Brief:** land four optimisations in the NNUE SIMD kernels and head. Gate on
**cycles**, median of ≥6 interleaved rounds, instructions reported alongside.
Do not rank by diffing against Stockfish.

---

## 0. Frames, and the instrument

Two frames, quoted with every number (a symbol share is meaningless without
one — study 21 §0.2):

- **Screening / accept frame ("wl50"):** `bench 16 1 100000
  tools/nnue/calib/wl50.fens nodes engines/canonical/net.cenn` — a fixed
  **5,001,562-node** tree.
- **Ranking frame ("d16"):** `bench 16 1 16 default depth
  engines/canonical/net.cenn` — **9,444,932** nodes.

Two builds, and they are NOT interchangeable:

- **fast:** `target-cpu=native`, thin LTO, 4 CGUs. Used for A/B iteration.
- **PGO:** `tools/build_pgo.sh` pipeline (native + profile-use, eval-file
  supplied). The shipping configuration, and **the only one in which the
  question the brief asked even arises** — see §1.

### 0.1 The instrument had a positional bias, and ABBA fixed it

Study 21's harness alternates `A,B,A,B,…`. Run first, at n=9, my null panel
(the same binary against itself under two names) came back at:

| null, AB order, wl50, n=9 | value |
|---|---|
| Δcyc_n (medians) | **+0.97%** |
| paired median | **+176.0** |
| negative | 3/9 |
| spread | −97.6 … **+1163.6** |

**Rule 1 fired.** An inert perturbation moved the statistic by as much as any
arm here plausibly could, and the spread is heavily right-skewed — with a
trainer and three sibling studies on the box, load *ramps inside a round*, and
whichever arm runs second eats it. That is a positional bias, not noise.

Fix: **ABBA rounds** (`A,B,B,A`), pair delta = mean(B) − mean(A) within the
round, so a monotone drift across a round cancels to first order. Re-nulled:

| null, ABBA, wl50, n=7 | value |
|---|---|
| paired median | **−6.6** (−0.15%) |
| negative | **4/7** |
| spread | −268.2 … +414.5 |
| Δins_n | 0.00% (exact) |

A coin flip, as it should be. **The paired ABBA median is the statistic; the
medians-of-arms are not** — they drifted +1.02% across that same null panel,
and `cyc_n` moved 5264 → 4415 between two panels an hour apart on the same
binary. Every headline below is the paired median.

---

## 1. P0 — the inline/exclusive split, settled by measurement

The brief's target table (`i16_deltas_into` 12.54%, `forward_into` 8.66%) is a
`perf --no-children` profile: cycles in a **symbol's code region, including
callees the compiler inlined into it**. Study 21's table is cycles in a
**source function's own body**. The brief asked which is which here, and told
me to measure it rather than infer it. Built with `#[inline(never)]` on
`apply_i16_deltas{,_into,_dual}`, `apply_i8_deltas_into`, the head's dispatched
kernels and `propagate_l1/l2/l3`, PGO, re-profiled on d16:

| PGO d16, `--no-children` | baseline (region) | noinline (own body) |
|---|---|---|
| `at_full` | **23.53%** | `at_full` 7.64% + **`apply_i8_deltas_into` 16.20%** |
| `i16_deltas_into` | 11.84% | **`apply_i16_deltas_into` 11.91%** — unchanged |
| `finny_refresh_into` | 7.03% | `finny_refresh_into` 1.15% + **`apply_i16_deltas_dual` 4.76%** |
| `forward_into` | 8.31% | **falls below the 0.3% cut**; `l1_dot` 2.67 + `activate` 1.39 + `l2_tail` 1.05 + `find_nnz` 0.80 + `skip_dot_i8` 0.39 = 6.30% |
| `search::<NonPv>` | 18.84% | 19.39% |

**Both prior claims were right about their own question.** Study 21's
"`apply_i8_deltas_into` is the largest single cost in the engine" reproduces at
**16.20%** and it genuinely does not appear in `nm` — PGO inlines it into
`at_full`. The brief's `i16_deltas_into` 12.5% also reproduces, and that one
was never inlined: the symbol *is* the body.

`forward_into` is the interesting case: **its own body is essentially free.**
The 8.31% is entirely its five inlined kernels, so "optimise `forward_into`"
means "optimise `l1_dot`, `activate`, `l2_tail`, `find_nnz`, `skip_dot_i8`",
none of which individually clears 3%.

### 1.1 The consequence for scope

Summed over function bodies, **the four files this study owns hold ~39% of PGO
d16 cycles**, not the 21% briefed — because the two biggest bodies
(`apply_i8_deltas_into` 16.20%, `apply_i16_deltas_dual` 4.76%) were hidden
inside symbols named after files owned by other studies.

### 1.2 Where the cycles are *not* going

Per-symbol attribution, fast build, d16, three events on the same binary:

| symbol | cycles% | instructions% | ratio | L1d-miss% | br-miss% |
|---|---|---|---|---|---|
| `i8_deltas_into` | 15.14 | 11.71 | 1.29 | **27.64** | 6.33 |
| `i16_deltas_into` | 11.16 | 6.93 | **1.61** | **19.05** | 2.87 |
| `i16_deltas_dual` | 4.70 | — | — | 5.34 | 2.93 |
| `at_full` | 7.67 | 8.44 | 0.91 | 6.78 | 10.06 |
| `init_quiets` | 5.84 | 7.34 | 0.80 | 1.45 | 11.58 |

(br-miss column is from the PGO noinline build, which is why `at_full` and
`init_quiets` appear there at their inclusive size.)

`i16_deltas_into` burns **11.16% of cycles for 6.93% of instructions** — the
most stall-bound symbol in the engine. It and the i8 fold together own **47% of
all L1d misses**. Those misses are compulsory: 2.1 HalfKA rows/fold × 2560 B
from a 57.7 MB table, 7.75 threat rows/fold × 1280 B from a 77.7 MB table,
against a 32 MB L3. **Prefetch is settled dead** (study 21 §2, wrong-address
control at +0.01%) and is not re-run here.

So the misses are not the lever. What *is* the lever is everything the kernels
do around them — and that turned out to be a lot more than it looked.
