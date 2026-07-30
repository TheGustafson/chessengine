# Perf study 25 — the NNUE SIMD kernels and head, on ARCHON's own terms

**Date:** 2026-07-29/30 · **Box:** 7700X (Zen 4), AVX-512 VBMI2, 32 KB L1i /
32 KB L1d, 1 MB L2/core, 32 MB L3, THP `always` · **Net/arch:** D48X,
`sota,arch-d48x`, ACC 1280, L2 16, L3 32 · **Base commit:** `cf7cf46`

**Scope:** `src/eval/nnue/{sota_simd,sota_head,sota_loader,sota_features}.rs`.
Three sibling studies own the accumulator/context (22), search/movepick (23) and
threats/board/movegen (24); nothing outside those four files is touched.

**Brief:** land four optimisations in the NNUE SIMD kernels and head. Gate on
cycles; report instructions alongside. Do not rank by diffing against
Stockfish.

---

## ★ Headline, and a retraction of my own

**Landed:** one arm — shape-specialising the HalfKA fold — worth **−79.3
instructions/node (−1.30%)** and **−14.7 branches/node (−2.28%)**, both exact,
node-identical and eval-hash-identical, with the kernel's hot code cut 30%.

**Retracted:** that arm's *cycles* claim. I first measured it at −2.33% and
−1.70% across two panels against a null taken beforehand, and wrote that into
the commit message. Re-measured with the null **interleaved into the same
rounds**, the byte-identical-copy null came back at **+2.06%**, then at
**−1.93%** under a third harness — both larger than the arm. **There is no
defensible cycles number for this arm, or for any arm, tonight** (§0.3).

**Rejected, each with the measurement that killed it:** the same shape trick on
the FullThreats fold (§3), and a sized-array view of the head's weight blocks
(§4).

**Not padded to four.** Two rejections and one landed instruction win is what
the evidence supports.

---

## 0. Frames, and the instrument

Two frames, quoted with every number (a symbol share is meaningless without
one — study 21 §0.2):

- **Screening / accept frame ("wl50"):** `bench 16 1 100000
  tools/nnue/calib/wl50.fens nodes engines/canonical/net.cenn` — a fixed
  **5,001,562-node** tree.
- **Ranking frame ("d16"):** `bench 16 1 16 default depth
  engines/canonical/net.cenn` — **9,444,932** nodes.

Two builds, not interchangeable:

- **fast:** `target-cpu=native`, thin LTO, 4 CGUs — A/B iteration.
- **PGO:** the `build_pgo.sh` pipeline with the eval-file supplied — the
  shipping configuration, and **the only one in which the brief's first
  question even arises** (§1).

### 0.1 The instrument had a positional bias, and then a curvature bias

Study 21's harness alternates `A,B,A,B,…`. My first null — the same binary
against itself under two names — at n=9:

| null, AB order, wl50, n=9 | value |
|---|---|
| Δcyc_n (medians) | **+0.97%** |
| paired median | **+176.0** |
| negative | 3/9 |
| spread | −97.6 … **+1163.6** |

**Rule 1 fired.** An inert perturbation moved the statistic by as much as any
arm here could, and the spread is heavily right-skewed: with a trainer and three
sibling studies on the box, load *ramps inside a round*, and whichever arm runs
second eats it. Positional bias, not noise.

Fix 1 — **ABBA rounds**, pair delta = mean(B) − mean(A) within a round, which
cancels monotone drift. Re-nulled: paired median **−6.6 (−0.15%), 4/7
negative**. A coin flip.

That was still not enough. See §0.2.

**Standing consequence, independent of everything else in this document: the
medians-of-arms are not a statistic here.** They drifted +1.02% across a null
panel, and `cyc_n` on one unchanged binary moved 5264 → 4415 → 4310 across three
panels within two hours. Only within-round paired deltas mean anything.

### 0.2 ★ A null taken before the arm is not evidence

Mid-study the coordinator relayed study 22's result: four nulls on a binary
against its own byte-identical copy, one session, **−3.40% / −0.07% / −1.51% /
+1.42%** — a 4.8-point spread on a change that does not exist. Study 22 had
already committed a −1.16% headline against a null taken 40 minutes earlier and
had to retract it.

So the null must run **inside the same rounds as the arm**. Fix 2 — a three-way
round-robin, `base` / byte-identical `copy` / `arm`, six slots per round.

My first cut of that ran the fixed symmetric sequence `base copy arm arm copy
base`, reasoning that positions 1&6, 2&5, 3&4 cancel drift. They cancel
*monotone* drift. They do not cancel **curvature**, and that sequence hands
`base` the two extreme slots and `arm` the two central ones. Measured:

| fixed-order round-robin, wl50, n=8 | paired median | negative |
|---|---|---|
| **NULL** (copy − base) | **+88.7 (+2.06%)** | 2/8 |
| ARM (a1 − base) | +30.2 (+0.70%) | 3/8 |
| Δins_n | null 0.00, arm −79.34 | — |

**The null is bigger than the arm.** That is the whole verdict on cycles
tonight, and it is also what killed my earlier −2.33%/−1.70% numbers.

Fix 3 — **randomise the slot order each round** (`shuf` over
`BASE COPY ARM BASE COPY ARM`), so no position is systematically owned by any
binary and a position effect becomes noise that `n` averages down rather than
bias that it does not. That is the harness in
`docs/research/perf-artifacts-25/scripts/panel3-25.sh`, and its header carries
this reasoning so the next study inherits it.

It did not rescue the measurement:

| randomised round-robin, wl50, n=12 | paired median | negative |
|---|---|---|
| **NULL** (copy − base) | **−83.5 (−1.93%)** | **10/12** |
| ARM (a1 − base) | +25.5 (+0.59%) | 6/12 |
| Δins_n | null **0.00**, arm **−79.34** | — |

### 0.3 ★ The verdict on cycles, and it is not a hedge

Three nulls tonight, all on **byte-identical binaries**, all on this box:

| harness | null |
|---|---|
| ABAB, n=9 | **+0.97%** |
| symmetric round-robin, n=8 | **+2.06%** |
| randomised round-robin, n=12 | **−1.93%** |

**A ~4-point swing on a change that does not exist** — matching study 22's
−3.40 / −0.07 / −1.51 / +1.42 exactly in character. Worse, the last null was
**10/12 sign-consistent**, so the sign count is no safer than the median: an
inert change can look like a strong, repeatable win.

**No cycles claim in this document, for any arm, in either direction.** Every
number claimed below is `instructions:u` or `branches:u` per node, whose noise
floor is 0.0 — re-verified in every panel here, where the byte-identical copy
reproduced 6082.2 i/n to the digit. That is not a lowered bar; it is the only
counter on this box tonight that can carry a sign.

**`instructions:u` per node is exact** — 0.0 noise, verified again here: the
byte-identical copy reproduced 6082.2 i/n to the digit in every panel. Every
number this study *claims* is an instruction or branch count.

---

## 1. P0 — the inline/exclusive split, settled by measurement

The brief's table (`i16_deltas_into` 12.54%, `forward_into` 8.66%) is
`perf --no-children`: cycles in a **symbol's code region, including callees the
compiler inlined into it**. Study 21's table is cycles in a **source function's
own body**. The brief asked which is which, and said to measure rather than
infer. Built with `#[inline(never)]` on `apply_i16_deltas{,_into,_dual}`,
`apply_i8_deltas_into`, the head's dispatched kernels and `propagate_l1/l2/l3`;
PGO; re-profiled on d16. **Treated purely as an instrument, not a candidate** —
study 22 found outlining can move cycles by itself.

| PGO d16, `--no-children` | baseline (region) | noinline (own body) |
|---|---|---|
| `at_full` | **23.53%** | `at_full` 7.64% + **`apply_i8_deltas_into` 16.20%** |
| `i16_deltas_into` | 11.84% | **`apply_i16_deltas_into` 11.91%** — unchanged |
| `finny_refresh_into` | 7.03% | `finny_refresh_into` 1.15% + **`apply_i16_deltas_dual` 4.76%** |
| `forward_into` | 8.31% | **falls below the 0.3% cut**; `l1_dot` 2.67 + `activate` 1.39 + `l2_tail` 1.05 + `find_nnz` 0.80 + `skip_dot_i8` 0.39 = 6.30% |
| `search::<NonPv>` | 18.84% | 19.39% |

**Both prior claims were right about their own question.** Study 21's
"`apply_i8_deltas_into` is the largest single body in the engine" reproduces at
**16.20%**, and it genuinely does not appear in `nm` — PGO inlines it into
`at_full`. The brief's `i16_deltas_into` ≈12.5% also reproduces, and that one
was never inlined: the symbol *is* the body.

`forward_into` is the instructive case. **Its own body is essentially free**;
the 8.31% is entirely its five inlined kernels. "Optimise `forward_into`" means
"optimise `l1_dot`, `activate`, `l2_tail`, `find_nnz`, `skip_dot_i8`", none of
which individually clears 3%.

### 1.1 Scope is ~39% of cycles, not 21%

Summed over function bodies, the four files this study owns hold **~39% of PGO
d16 cycles** — because the second and third biggest bodies
(`apply_i8_deltas_into` 16.20%, `apply_i16_deltas_dual` 4.76%) were hidden
inside symbols named after files owned by other studies.

### 1.2 Where the cycles are *not* going

Per-symbol attribution, fast build, d16, same binary, three events:

| symbol | cycles% | instr% | cyc/instr | L1d-miss% | br-miss%¹ |
|---|---|---|---|---|---|
| `i8_deltas_into` | 15.14 | 11.71 | 1.29 | **27.64** | 6.33 |
| `i16_deltas_into` | 11.16 | 6.93 | **1.61** | **19.05** | 2.87 |
| `i16_deltas_dual` | 4.70 | — | — | 5.34 | 2.93 |
| `forward_into` | 3.59 | 3.61 | 0.99 | 7.40 | — |
| `l1_dot` | 2.92 | 3.33 | 0.88 | 5.70 | 1.33 |
| `skip_dot_i8` | 0.39² | — | — | — | **1.82** |

¹ from the PGO noinline build. ² PGO figure.

`i16_deltas_into` burns **11.16% of cycles for 6.93% of instructions** — the
worst ratio in the engine. It and the i8 fold own **47% of all L1d misses**.
Those misses are compulsory: 2.1 HalfKA rows/fold × 2560 B out of a 57.7 MB
table, 7.75 threat rows/fold × 1280 B out of 77.7 MB, against a 32 MB L3.
**Prefetch is settled dead** (study 21 §2, wrong-address control at +0.01%) and
was not re-run.

**Alignment, verified not re-hunted, as instructed.** All five large tables
(`halfka_weights`, `ft_bias`, `threat_weights`, `psqt_weights`, and the head's
`l1_weights`/`l1_biases`/`l2_weights`/`l2_biases`/`l3_weights`) are `Aligned<T>`
or `Aligned`-backed, i.e. 64-aligned by construction, and
`head_f32_blocks_start_on_a_cache_line` is a live gate on the three fields study
21 arm B4 fixed. Nothing regressed. (§4 finds that `Aligned` has a *different*
cost, unrelated to alignment.)

So the misses are not the lever, and neither is the vector arithmetic — see
§5.1 on why the fold kernels' ALU is already minimal. What was left was
everything the kernels do *around* them.

---

## 2. LANDED — arm 1: the HalfKA fold always has ≤2 adds and ≤2 subs

`src/eval/nnue/sota_simd.rs`, commit `2703512`.

`apply_dirty_directed` (`sota_accumulator.rs`) builds both feature lists in
`[usize; 2]` arrays, and its own comment pins the lower bound: every real move
emits at least one add and one sub, and a null move pushes no entry at all. **A
HalfKA fold is therefore always one of four shapes.** LLVM cannot see that
through `&[usize]`, and emitted a **6231-byte** function: both row loops
unrolled ×2 with an odd/even remainder branch, cloned across the empty-list
cases, **480 `vpaddw`/`vpsubw` in 24 copies of the tile body**.

Fix: four const-generic instantiations `i16_into_shaped::<NA, NS>`,
`(NA, NS) ∈ {1,2}²`, dispatched on the slice lengths. Each is a branch-free
straight line — no loop control, no unroll remainder, no empty-list clone — and
`feature * ACC` is computed once per call instead of once per tile. The tile
walk became `for t in 0..TILES` with `TILES` constant, so the provably-dead
`while off < ACC` tail is gone too. The generic path stays as
`#[cold] #[inline(never)]` for long lists and for the **empty** shape, which
must remain a copy (`sota_accumulator.rs` deleted its own copy special-case and
relies on it).

**This is not perf study 12 §4's rejected offset hoist.** That built a
`[MaybeUninit<usize>; 96]` for a runtime-length list and paid 91 instructions of
prologue to save four multiplies. Here the array is `[usize; NA]` with `NA ≤ 2`
known at compile time: two registers, no prologue.

**The dispatch was measured, not assumed.** Instrumenting 4,194,304 real folds
(wl50):

| shape | share |
|---|---|
| (1,1) | **72.93%** |
| (1,2) | **26.02%** |
| (2,1) | 1.05% |
| (2,2) / (0,0) | not observed |

**99.99% of folds hit a shaped instantiation.** That number is the arm's
mechanism, and it is why §3 fails.

| wl50 | baseline | a1-shaped | Δ |
|---|---|---|---|
| instructions/node | 6082.1 | **6002.8** | **−1.30%** (exact) |
| branches/node | 643.8 | **629.1** | **−2.28%** (exact) |
| hot code, `i16_deltas_into` | 6231 B | **4351 B** | −30% |
| nodes | 5,001,562 | 5,001,562 | identical |
| eval hash | `5ca978e012da1bb0` | same | identical |

**Cycles: unresolved.** See §0.2 — the interleaved null exceeded the arm. New
test `every_dispatched_i16_fold_shape_matches_scalar` covers all four shapes
plus the empty one; nothing did before, because every other kernel test in the
module runs 4–5 rows or 200+, all of which land in the generic fallback.

---

## 3. REJECTED — the same trick on the FullThreats fold

`apply_i8_deltas_into` is the biggest body in the engine (16.20%), so the
obvious move is arm 1 again. **Measure the shape space first** — the discipline
study 21 used to kill the cancellation idea. Over 4,194,304 real folds (wl50):
mean **1.98 adds / 2.66 subs**, spread over ~100 distinct shapes with a tail to
(12,13). Cumulative coverage:

| shaped set | coverage |
|---|---|
| `{0,1}²` | 27.0% |
| **`{0,1,2}²`** | **52.0%** |
| `{0..3}²` | 68.0% |

24.4% of folds have an **empty** list on one side ((0,1) 10.0%, (1,0) 8.5%),
which is why 0 belongs in the set. Each i8 instantiation costs
`2 tiles × (40 + 40(NA+NS))` vector instructions — i8 rows are two ops per block
(`vpmovsxbw` + `vpaddw`), twice arm 1's body — so `{0..3}²` would add ~22 KB to
a 32 KB L1i. `{0,1,2}²` was the cut taken: eight instantiations, ~11 KB, half
the calls.

**Pre-registered falsifier:** if cycles do not improve while instructions fall,
the kernel is I-cache limited and shaping is the wrong lever at this coverage.

**What actually happened is worse than the falsifier: instructions rose.**

| wl50 | a1-shaped | a2 (`match` on the pair) | a2b (packed key) |
|---|---|---|---|
| instructions/node | 6002.8 | 6011.0 (**+0.14%**) | 6017.1 (**+0.24%**) |
| branches/node | 629.1 | 636.3 (+1.15%) | 635.8 |
| hot code | 3336 B | 11674 B | 11674 B |

The first form's `match (adds.len(), subs.len())` over eight pairs compiled to a
**compare chain**, and the 48% of calls that fall through pay every link before
reaching the generic path. Rewriting it as one bounds check plus a dense
`na * 3 + ns` key did not recover it either. Cycles, ABBA-paired n=8: **paired
median −5.5 (−0.13%), 5/8 negative** — inside the null band, i.e. nothing.

**Why arm 1 works and this does not: coverage.** At ~100% every call amortises
the dispatch; at 52% half the calls pay it for nothing, and the shapes that fall
through are exactly the row-heavy ones where the generic loop's overhead was
already amortised. Reverted.

---

## 4. REJECTED — sized-array views of the head's weight blocks

Between the head's SIMD kernels, `forward_into`'s disassembly is dense with
scalar work. Each of the five bucket-selected blocks is taken as
`self.field[base..base + N].try_into().expect(..)`, which is a fat-pointer load
plus a range check plus a `try_into` length check — roughly ten scalar
instructions and three unhoistable panic branches, five blocks deep, on every
eval. The coordinator relayed study 22's matching result: `Box<[T]>`
fat-pointer indexing cost **+58.7 i/n** there until the arrays became
`Box<[T; N]>`, which turned it into **−9.3**.

Built it as `as_chunks::<N>()` views — the idiomatic sized-array re-type.

| wl50 | a1-shaped | a3 (as_chunks) | a3b (+ non-pow2 reverted) | a3c (+ inline fix) |
|---|---|---|---|---|
| instructions/node | 6002.8 | 6009.6 | 6009.6 | **6002.3** |

**−0.5 i/n. A null.** Three things were learned by measuring instead of
shipping it:

1. `as_chunks::<N>()` divides the slice length by `N`. For `L2 * L1` = 20480 =
   2¹²·5 and `L1` = 1280 = 2⁸·5 that is a real divide-by-constant, not a shift.
   Reverting those two to plain range slices changed nothing, so this was not
   the cost either.
2. Giving `skip_dot_i8` sized-array arguments **flipped LLVM's inlining
   decision** — it went out of line into a 361-byte symbol and the call cost
   +7 i/n, which is the whole of a3's regression. `#[inline(always)]` recovered
   it exactly.
3. **The real reason there was nothing to win: `Aligned<T>` is not a
   `Box<[T]>`.** It is a `Vec<T>` plus an offset, and `Deref` is
   `&self.buf[self.off..]` — a *bounds-checked range slice on every access*,
   before any indexing. `as_chunks` therefore traded one check for another and
   left the expensive part untouched. Killing it needs the deref itself to go,
   which means changing `Aligned` — `src/eval/nnue/aligned.rs`, **outside this
   study's four files**. Priced in §7.1 rather than reached across for.

Reverted. One piece was kept, on correctness grounds only and explicitly not
counted as a win: commit `074f875` fixes
`dispatched_skip_dot_i8_matches_scalar_exactly`, whose fixtures were
`vec![...; 768]` hardcoded from the D47V era. At ACC = 1280 that tested a width
the engine never passes, and since 768 is not a multiple of 256 it drove the
avx512 backend's tail loop while every real call drives the 256-wide four-chain
body study 21 arm B2 introduced. Test-only, so identity is untouched by
construction.

---

## 4a. LANDED (not a win) — four provably-dead remainder loops

`src/eval/nnue/sota_simd.rs`, commit `d157fd3`.

Every avx512 fold kernel ended its tile walk with a one-block `while off < ACC`
remainder, under a comment saying that remainder is dead: `pick_regs` returns an
exact divisor of `ACC_BLOCKS` at every shipped width. It really is dead; LLVM
cannot see it, because `off` is a runtime value, so it emitted the remainder
anyway and cloned it through the empty-list cases. `TILES * TILE == ACC` is now
the machine-checked statement of the invariant and the four remainders are gone.

**The obvious rewrite was the wrong one**, which is the part worth keeping:

| shape | instructions/node | Δ vs arm 1 |
|---|---|---|
| `for t in 0..TILES { let off = t * TILE; … }` | 6012.4 | **+9.6** |
| `let mut off; for _ in 0..TILES { … off += TILE }` | 6010.7 | **+7.9**, branches +3.6 |
| `while off + TILE <= ACC`, remainder deleted | **5999.6** | **−3.2** |

A compile-time trip count reads like the strictly better shape and is not: LLVM
declines to unroll the counted loop and keeps a real induction variable, while
with nothing following the `while` it unrolls both tiles outright. All three are
node- and eval-hash-identical, and the counter's floor is 0.0, so the signs are
real.

**−3.2 i/n is 0.05% and is not claimed as a win.** This lands as a dead-code
deletion — 74 lines of source that every future reader would have had to reason
about — which happens to measure the right way.

---

## 5. What did NOT turn out to be available

### 5.1 The fold kernels' vector arithmetic is already minimal

Worth recording so it is not re-derived. The i8 fold's inner op is
`vpmovsxbw zmm, ymmword ptr [mem]` (the load folds into the sign-extend) then
`vpaddw` — two instructions per 32 lanes per row, and there is no cheaper
sequence on this ISA:

- widening the threat table to i16 halves the ops but doubles a 77.7 MB table
  against a 32 MB L3;
- the `vpsllw/vpsraw` even/odd trick costs four ops per 64 lanes instead of
  four, *and* deinterleaves the accumulator;
- `vpmaddubsw`/`vpdpbusd` sum *adjacent* lanes, which is not what a row fold
  needs.

`REGS = 20` (two tiles) is likewise not free to change: `REGS = 40` spills, and
every other divisor of `ACC_BLOCKS` costs more loop control.

### 5.2 `forward_into`'s dead-init class is clean

Study 15 arm 4's defect (`let mut ft = [0u8; L1]` compiling to a per-eval
`memset`) does not recur: `objdump` of `forward_into` on the native build shows
**no `memset` or `memcpy` call at all**. `pre`/`out` are either genuinely
required seeds or dead-store-eliminated through the direct kernel calls.

---

## 6. Scorecard, honestly

| | |
|---|---|
| **Landed as a win** | 1 — arm 1, on the instruction/branch axis, exact |
| **Landed, explicitly not a win** | 2 — the dead remainder loops (§4a, −3.2 i/n), the stale 768-lane kernel test (§4) |
| **Rejected with a measurement** | 2 — i8 shape dispatch (§3), head sized-array views (§4) |
| **Retracted** | 1 — arm 1's own cycles figure (§0.2/§0.3) |

Cumulative, wl50, `instructions:u` at a 0.0 noise floor: **6082.1 → 5999.6 per
node, −1.36%**, branches 643.8 → 629.0 (−2.30%), nodes 5,001,562 and eval hash
`5ca978e012da1bb0` unchanged throughout, on every arm including the rejects.

The brief asked for four. **One landed win is what the evidence supports**, and
the reason is §0.3 rather than any shortage of candidates: a byte-identical
binary measured +0.97%, +2.06% and −1.93% against itself tonight, so no
cycles-only arm can be adjudicated at all, and every verdict here — both wins
and both rejections — was decided on the exact instruction counter instead. I
would rather report one verified win, two controlled rejections and a
retraction of my own headline than four arms at a floor I have just
demonstrated, three times, is above them.

What actually moved: **making a compile-time fact visible to the compiler.**
The HalfKA fold's shape space is four, pinned by a `[usize; 2]` two files away,
and the whole win is telling LLVM so.

---

## 7. Two-sided prediction, for the next study

**Prediction.** The head's five per-eval weight-block acquisitions cost **40–55
instructions/node**, and almost all of it is `Aligned<T>`'s `Deref` — a
bounds-checked range slice executed five times per eval before any indexing.
Converting the `SotaHead` blocks to thin, compile-time-sized
`Box<[Block64<[f32; N]>; BUCKETS]>` (a `#[repr(align(64))]` newtype, exactly
study 21's `Line64` shape) will measure **between −35 and −60 i/n** at the 0.0
noise floor.

**What kills it.** A one-build probe that replaces the five acquisitions with
raw-pointer arithmetic — same loads, no checks, no `Deref` — measuring **less
than −20 i/n**. If the checks are not the cost, the whole line is dead and
nobody should spend the six-file refactor on it. **Run that probe first.**

---

## 8. Ranked remainder

1. **Thin, sized head weight blocks** — §7. Needs `aligned.rs`, `sota_head.rs`,
   `sota_loader.rs`, `sota_features.rs`, and the two integration builders
   `tests/cenn_v2_smoke.rs` + `tests/eval_file_reload.rs` (the exact fallout
   shape study 21 arm B4 hit). Estimated −35 to −60 i/n; probe it before
   building it.
2. **`Aligned::deref`'s bounds check, for the whole engine.** The same
   `&self.buf[self.off..]` runs on every access to *every* weight table,
   including the two FT tables the fold kernels read. One `get_unchecked` with
   a two-line safety argument (`off <= buf.len()` holds by construction — `buf`
   is resized to `off` and then extended) removes a branch from every one. Not
   in this study's scope; cheap and crate-wide.
3. **47% of the HalfKA fold's vector work is pure copy.** Per fold the shaped
   kernel issues 40 `zmm` loads of `src` and 40 stores to `dst` to apply 2.27
   rows — 80 of ~171 vector ops are the parent→child copy, not arithmetic. This
   is the same object as study 21 §9.1 (ARCHON maintains two 1280-wide
   accumulators) seen from the kernel side, and it is study 22's to move: it
   needs the fold to happen in place, which is the caller's design.
4. **`apply_i16_deltas_dual` (4.76%)** still has the runtime tile walk and the
   provably-dead `while off < ACC` tail that arm 1 removed from its twin.
   Bit-identical to fix, strictly less code — but its feature lists are long
   (Finny refresh), so loop control is already amortised. Expect well under
   0.2%; below the floor even on a quiet box.
5. **Threat-row locality.** The threat fold takes 155 cache-line misses per
   fold from a 77.7 MB table. A **permutation** of the threat feature index
   space — applied consistently in `sota_features.rs` (index) and
   `sota_loader.rs` (rows) — is bit-identical and could co-locate rows that
   co-occur. Both files are in scope, but choosing the permutation needs
   co-access statistics; a research task, not a perf arm.
6. **`load_v2_bytes` is 1.37% of the d16 frame** (`read_i16s`/`read_i8s_aligned`
   walk 135 MB element-by-element). This is **startup cost, not search speed** —
   an engine loads its net once — but it is real wall clock and it silently
   inflates every fixed-node measurement in this document by over a percent.
   Worth fixing for the instrument's sake; worth **nothing** for playing
   strength, and must not be reported as a cycles win.

## 9. Do not re-propose

- **Shape-specialising the FullThreats fold** at `{0,1,2}²` — §3, instructions
  +0.24% exact, cycles inside the null, hot code 3.3 → 11.7 KB.
- **`as_chunks` views of the head's weight blocks** — §4, −0.5 i/n. The
  `Aligned` deref is the cost, not the indexing.
- **Software prefetch of FT weight rows**, at any site — study 21 §2, settled
  with a wrong-address control at +0.01%.
- **FullThreats add/sub cancellation** — study 21 §3, 0.075 pairs/fold.
- **A cheaper widening sequence in the i8 fold** — §5.1.
- **A null panel run before the arm.** §0.2.
- **`for t in 0..TILES` as a "compile-time trip count" tile walk** — §4a,
  +9.6 i/n. LLVM does not unroll it.

---

## 10. For whoever measures cycles next

The four candidates in §8 are all instruction-axis items and can be settled on a
loaded box. **Anything cycles-only must wait for the trainer to finish**, and
must then re-establish the floor before believing anything: run
`panel3-25.sh <base> <copy-of-base> <copy-of-base>` — *both* comparison slots a
byte-identical copy — and require the two reported medians to agree with each
other and with zero to inside ±0.3% before running a real arm. Tonight that
check would have failed three times out of three, and it costs one panel.
