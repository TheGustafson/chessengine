# Perf study 46 — `forward_into`, and what the machine actually wants

**Date:** 2026-07-30 · **Box:** 7700X (Zen 4), AVX-512 VBMI2, SMT on, 32 KB L1d,
1 MB L2/core, 32 MB L3, 16 GB · **Base:** `main` at `17ff52b2` (studies 43/44/45
merged) · **Net/arch:** D48X @ `network_scale=600`, `sota,arch-d48x`, ACC 1280,
L1 1280 / L2 16 / L3 32 / BUCKETS 8 · **Instruments:** `tools/perf46/` +
`docs/research/perf-artifacts-46/` · **Worktree:** `perf-study-46-forward`

**Frame, quoted with every number below:**

```
bench 16 1 100000 tools/nnue/calib/wl50.fens nodes engines/canonical/net.cenn
```

50 FENs × 100 k nodes = **5,001,562 nodes**, one thread, 16 MB TT, shipped
release profile (fat LTO, CGU = 1), `-C target-cpu=native`, non-PGO. Every
binary here is node-identical at 5,001,562, d13-identical at 1,739,588 and
eval-hash-identical at `2e1ede95…79c4`, gated per binary by
`tools/perf46/ident46.sh` before any timing.

Denominators used for cycle conversions: the settled quiet-box **4179.7
cycles/node** and study 44's loaded **4668.94**; both are quoted where a
percentage is derived, because this study's own panel ran on a loaded box.

---

## ★ HEADLINE

**The commissioned hypothesis is false, and I can say so with a number instead
of an argument.** `CHAINS = 4` in `l1_dot_inner` is not "half the ILP the
machine wants" — it is **exactly** the ILP the machine wants.

Measured directly on this box (generated assembly, chain-count sweep, min-of-5,
`perf stat -e cycles:u`, `tools/perf46/ilp.c`):

| independent chains | 1 | 2 | 3 | **4** | 5 | 6 | 8 | 10 |
|---|---:|---:|---:|---:|---:|---:|---:|---:|
| `vpdpbusd` **zmm** | 4.007 | 2.002 | 1.338 | **1.004** | 1.002 | 1.002 | 1.002 | 1.002 |
| `vfmadd231ps` **zmm** | 4.005 | 2.002 | 1.338 | **1.006** | 1.009 | 1.002 | 1.003 | 1.002 |
| `vpdpbusd` *ymm* (control) | 4.005 | 2.003 | 1.335 | 1.002 | 0.802 | 0.668 | **0.538** | 0.527 |
| `vfmadd231ps` *ymm* (control) | 4.006 | 2.003 | 1.335 | 1.003 | 0.802 | 0.669 | **0.537** | 0.526 |

Latency is **4 cycles** and 512-bit reciprocal throughput is **1.00 c/op — one
effective 512-bit FMA/VNNI pipe, not two and not four.** The knee is at 4 and
the curve is flat from 4 to 10. The premise behind the commission ("~4–5 cycle
latency against 2 issue ports ⇒ 8–10 chains") is right about latency and wrong
about ports: Zen 4 double-pumps 512-bit ops over a 256-bit datapath. **The
control proves the mechanism rather than asserting it** — the *same two
instructions* at 256 bits give 0.53 c/op (2/cycle) with the knee at 8, at the
same 4-cycle latency. `latency × throughput` = 4 at 512-bit and 8 at 256-bit,
measured both ways.

**Consequences, in order of how much they cost the engine:**

1. **`l1_dot_inner` needs no change.** Widening `CHAINS` to 8 buys provably
   zero. Study 21's design constant is correct, and study 21's own cycles null
   was right for the right reason.
2. **The `l1_dot` tail is not a defect either — I predicted it was, and the
   measurement refuted me.** §3.2.
3. **`l2_tail_inner` is the one genuine ILP deficit in the head** — 2
   accumulator chains where the machine wants 4 — and it is **priced at
   −0.185 % of engine cycles and REJECTED on cost/benefit**, because it is not
   eval-bit-identical and the prize is below the panel's own resolution. §4.
4. **★ Study 44's port-pressure kill is under-stated by ~2×, and the reason is
   a divisor it did not measure.** §5. This is the most consequential thing in
   the study.

**The load-bearing number that makes all of §3–§5 arithmetic instead of
hand-waving: `evals/node = 0.43117`**, exact (§2). Every per-eval quantity in
this document is converted with it, and it is 2.3× away from the "≈1" a reader
would assume.

**Landed:** §6 — the `REGS_CAP` sweep study 44 left open (`r4` built, panelled
against `r5` and `r8`).

**Instrument failure, mine, reported first:** my initial panel read a
byte-identical null at **+0.878 % cycclean**. Cause: I ran the microbenchmarks
of §1 concurrently with it — precisely the mistake study 44 §7 warns about. The
panel was discarded, preserved as `panel-regs-CONTAMINATED.txt`, and re-run with
nothing else of mine on the box. §6.1.

---

## 0. WHAT THE COMMISSION ASKED, AND THE ORDER I DID IT IN

Study 44 handed over one anomaly: **`SotaHead::forward_into` takes 8.20 % of all
retirement-blocked cycles while only 39 % of its stalls are memory** — the only
inversion in an engine that is otherwise 83 % memory-bound. It named a suspect
(study 21's `CHAINS`, at half the needed width) and required three things be
established before anything was built. All three are answered below with
measurements, and two of them came out against the hypothesis.

| # | the commission asked | answer | where |
|---|---|---|---|
| 1 | What is `BLOCKS` actually, at the shipped build? | `L2 = 16`, `L2_BLK = 16` ⇒ **`BLOCKS = 1`** ⇒ **`CHAINS = 4`**. Derived from source, not assumed. | §1.1 |
| 2 | What is the real latency and port count on Zen 4? | **latency 4.007 c, throughput 1.002 c/op at 512-bit**, knee at 4. Validated by a 256-bit control at 0.527 c/op, knee at 8. | §1.2 |
| 3 | Is the 11.38 % dependency-chain figure localised to `forward_into`? | Yes, per study 44 §2 — but the residual is **not** a fixable dependency chain. It is 512-bit **issue throughput**. | §5 |

---

## 1. THE ILP AUDIT

### 1.1 `BLOCKS` and `CHAINS`, derived rather than assumed

`sota_head.rs:101` gives `L2 = 16` for `arch-d48x`; `sota_simd.rs:2074-2075`:

```rust
const BLOCKS: usize = L2 / L2_BLK;                          // 16 / 16 = 1
const CHAINS: usize = if BLOCKS >= 4 { 1 } else { 4 / BLOCKS };  // = 4
```

So the shipped kernel runs **4 independent `zmm` accumulator chains** over
`nnz.as_chunks::<4>()`. This reproduces study 44 §2.2's correction (which had to
retract a first draft asserting `L2 = 128 ⇒ CHAINS = 1`), and it is the input to
everything below.

### 1.2 ★ The instrument: measured latency and throughput, with a control

`tools/perf46/ilp.c` is generated by `tools/perf46/gen_ilp.py`: for a given
instruction and a given chain count `N`, it issues `N` accumulators round-robin,
8 deep per loop iteration, in inline assembly so no compiler decision is in the
loop. Cycles come from `perf stat -e cycles:u`; the op count is printed by the
program. Min-of-5 over reps, `taskset` to one logical CPU.

Reading the table in §HEADLINE:

* **`N = 1` reads the latency** — 4.007 and 4.005 cycles, i.e. exactly 4.
* **`N` large reads the reciprocal throughput** — 1.002 c/op at 512-bit.
* **The knee is the number of chains the machine wants**, and it is where the
  two meet: `4 × 1 = 4`.

**Why the control matters more than the headline number.** A single flat curve
could be an artifact of my loop, my measurement, or a frequency effect. The
256-bit forms of *the same two instructions* run the *same* code shape and give
a different, equally clean answer — 0.527 c/op, knee at 8, same 4-cycle latency.
A measurement error would have to move both curves in opposite directions by
exactly 2× to fake this. The double-pump model predicts precisely that, so the
instrument is validated by an internal control rather than by my confidence in
it.

**This is the null the commission asked for on this instrument**, and it passed.

---

## 2. ★ `evals/node = 0.43117` — the factor that rescales everything

Every kernel measurement in this study is naturally **per eval**. Every engine
budget is **per node**. The conversion is not 1.

Taken from the exact `foldcount` instrument (`--features foldcount`, atomic
counters, no sampling anywhere), on the identical frame, from two independent
prior studies' artifacts:

| artifact | nodes | `eval.calls` | evals/node |
|---|---:|---:|---:|
| `docs/research/perf-artifacts-40/foldcount-main.txt` | 5,001,562 | 2,156,392 | **0.43117** |
| `docs/research/perf-artifacts-43/foldcount43.txt` | 5,001,562 | 2,156,392 | **0.43117** |

Two studies, same frame, identical count — so this is a property of the frame,
not of a build.

**Why it matters, and why I went looking for it.** My first pass at `activate`'s
cost did not close: study 26 §7 puts `activate` at ~26 % of `forward_into`, and
study 25 puts `forward_into` at ~8.3 % of cycles ⇒ ~90 cycles/node — but
`activate`'s static shape is ~600 zmm ops per eval, which at the §1.2 measured
throughput cannot fit in 90 cycles. At 0.431 evals/node the arithmetic closes:
~600 ops/eval ≈ 300 cycles/eval ≈ **129 cycles/node**, against ~90 attributed.
Same order, and the residual is explained by the ops that are not the 1/cycle
class (§5).

**A per-eval prize is worth 0.43× as much per node as it looks.** That single
factor is what turns §4's arm from "worth building" into "not worth building".

---

## 3. THE KERNEL-BY-KERNEL AUDIT — measured, not modelled

`tools/perf46/kern.rs` replicates each kernel's exact intrinsic sequence in a
standalone binary and **chains the calls** (each call's output feeds the next
call's input), because `forward_into` is a strictly serial pipeline —
`activate → find_nnz → l1_dot → l1_dequant → l2_tail → l3_dot`, each stage
consuming the whole of the previous stage's output. Chaining makes this a
**latency** instrument, which is the quantity a dependency-chain question is
about. Min-of-7, `perf stat -e cycles:u`.

| kernel shape | cycles/call |
|---|---:|
| `l2_tail`, **2 accumulators (shipped)** | **97.63** |
| `l2_tail`, 4 accumulators | **79.75** |
| `l2_tail`, 8 accumulators | 90.90 |
| `l1_dot` count = 39, **tail → chain 0 (shipped)** | **92.95** |
| `l1_dot` count = 39, tail spread across chains | 96.81 |
| `l1_dot` count = 40 (zero tail), shipped shape | 95.88 |
| `l1_dot` count = 40 (zero tail), spread shape | 95.19 |

All variants carry the same chaining overhead, so **differences are the signal
and absolute levels are not** — the feedback path (store → load → broadcast) is
a constant added to every row.

### 3.1 `l2_tail_8` being worse than `l2_tail_4` re-confirms §1.2 independently

`l2tail8` (90.90) is **worse** than `l2tail4` (79.75) while being strictly more
unrolled. That is the §1.2 knee showing up in real kernel code measured a
completely different way: past 4 chains you pay extra reduction adds for a
latency benefit that does not exist. Two independent instruments, same answer.

### 3.2 ★ The `l1_dot` tail: I predicted a defect and the measurement refuted it

`nnz.as_chunks::<CHAINS>()` leaves a remainder of up to 3 groups, and the
shipped code folds **every one of them into `sums[0]`** (`sota_simd.rs:2103`),
lengthening chain 0 while chains 1–3 sit idle. At a typical count of 39 that is
9 quads + 3 tail, so chain 0 is 12 deep against 9 — a predicted ~9 cycles of
exposed latency, and the fix looked free because **i32 wrapping addition is
associative, so spreading the tail across chains is bit-identical**.

**It measured worse: 96.81 against 92.95, +3.9 cycles.** And at count = 40,
where the tail is empty and the two shapes are the same computation, they are
within 0.7 cycles of each other. The gain I predicted is not there, and the
"fix" costs. The plausible cause is that indexing `sums[chain]` with a loop
variable in the remainder defeats the register allocation that a fixed `sums[0]`
target gets — but I did not disassemble to confirm, so that is a hypothesis and
not a finding.

**No change to `l1_dot`. It is at its ILP optimum, tail included.**

---

## 4. `l2_tail` — the one real deficit, priced and REJECTED

`sota_simd.rs:2355-2362`: two `zmm` accumulators, 16 iterations, 2 FMAs per
iteration. Each accumulator carries a **16-deep FMA chain**. At the §1.2
measured numbers:

| | cycles |
|---|---:|
| throughput floor (32 FMAs at 1.00 c/op) | 32 |
| latency bound at 2 chains (16 deep × 4 c) | **64** |
| latency bound at 4 chains (8 deep × 4 c) | 32 |

So the shipped kernel runs at **2× its own throughput floor**, and §3 measures
the realisable part of that at **−17.9 cycles/call**. Study 21 §9.3 item 5
identified this shape and left it open; this study prices it.

**Converted with §2: −17.9 × 0.43117 = −7.72 cycles/node = −0.185 %** of the
4179.7 frame (−0.165 % of 4668.94).

**Rejected, for three reasons that compound:**

1. **It is below the instrument's resolution.** Study 44's null on the fixed
   panel carried se ≈ 0.26 % on `cycclean` at n = 40. Resolving −0.185 % at
   3 se needs several hundred paired rounds — more wall clock than the whole
   study has, for an effect one seventh the size of study 44's landed win.
2. **It is not eval-bit-identical, and there is no bit-identical route.** A
   left-to-right float sum *is* a serial dependency chain; any reassociation
   changes the rounding. `propagate_l2`'s own doc comment
   (`sota_head.rs:481-488`) states the i-sequential accumulation order is a
   pinned contract, and the **scalar backend is the bit-exact reference** that
   both SIMD backends are tested against.
3. **The blast radius is three backends plus the reference.** AVX-512 (2
   `zmm` chains), AVX2 (`sota_simd.rs:1437-1448`, 4 `ymm` chains that are 4
   disjoint output slices, each still 16 deep — the same 2× deficit) and the
   scalar reference would all have to change **identically**, plus the
   cross-backend equivalence tests, plus the eval-hash and d13 pins.

**A pin-moving, three-backend, contract-changing rewrite for −0.185 % that the
panel cannot even see is the wrong trade.** Recorded here with the number so
nobody re-derives it, and so that if `L2` ever grows (the chain deepens
linearly, and at `L2 = 128` the deficit would be ~8× larger) it can be
re-priced immediately.

---

## 5. ★ THE CORRECTION — study 44's port-pressure kill is under-stated ~2×

Study 44 §1.2 killed execution-port pressure for the whole engine:

> `fp_ops_retired_by_width.all` = **1399.34 uops/node** (1282.60 of them
> 512-bit). Over Zen 4's **4 FP pipes** that is **7.1 % – 14.4 % occupancy** …
> **DEAD at either end of the bracket. The FP pipes are ≥ 85 % idle. Do not
> build a port-pressure arm.**

Study 44 was careful about the counter's semantics — its assumption **A4**
explicitly refuses to assert whether `pack_512_uops_retired` counts one event
per 512-bit instruction or two, and brackets the verdict at both readings. **But
it took the divisor — "4 FP pipes" — as given, and never measured it.** §1.2
measures it:

| 512-bit op class (8 chains, sustained, min-of-5) | cycles/op | effective pipes |
|---|---:|---:|
| `vfmadd231ps` zmm | 1.0035 | **1** |
| `vpdpbusd` zmm | 1.0033 | **1** |
| `vaddps` zmm | 1.0022 | **1** |
| `vpaddd` zmm (simple integer) | 0.5516 | **~2** |

**There is no op class in this engine's hot path that retires 4 per cycle at
512 bits.** The FMA/VNNI/FP-add class — which is `l1_dot`, `l2_tail`, `l3_dot`,
`skip_dot_i8`, `l1_dequant` and both fold kernels' multiply-accumulate work —
runs at **exactly 1 per cycle**. Simple integer vector work runs at ~2.

Re-running study 44's own arithmetic with a measured divisor instead of an
assumed one, and keeping its A4 bracket intact:

| counter semantics | op class | 512-bit issue cycles/node | % of 4668.94 |
|---|---|---:|---:|
| 1 event per instruction | 1/cycle (FMA/VNNI) | 1282.60 | **27.5 %** |
| 1 event per instruction | 2/cycle (integer) | 707.5 | 15.2 % |
| 2 events per instruction | 1/cycle (FMA/VNNI) | 641.3 | 13.7 % |
| 2 events per instruction | 2/cycle (integer) | 353.8 | 7.6 % |

**Study 44's bracket was 7.1 % – 14.4 %. The correct bracket is 7.6 % – 27.5 %**
— the top roughly doubles, and the engine's real mix (both fold kernels and
`activate` are integer-class; the head's dots are FMA-class) sits in the middle
of it, not at the bottom.

**What survives of study 44's verdict, and what does not.**

* **Survives:** there is no arm that "adds ports", and `fp_sch_rsrc_stall` at
  4.35 % genuinely says dispatch is rarely blocked *waiting for an FP scheduler
  slot*. Nobody should build a scheduler-pressure arm.
* **Does not survive: "the FP pipes are ≥ 85 % idle."** At the measured
  divisors they are plausibly a quarter to a third busy, and `fp_sch_rsrc_stall`
  does not corroborate idleness — **a machine can be 100 % pipe-busy and never
  fill its scheduler**, if the front end delivers ops at exactly the issue rate.
  The two counters answer different questions and study 44 read one as
  confirming the other.

### 5.1 `forward_into` in per-EVAL units, which is where it becomes legible

Study 44's shares are per node. Divided by §2's 0.43117 they become per eval,
and the numbers stop looking small:

| quantity | per node | **per EVAL** |
|---|---:|---:|
| `forward_into` total cycles (study 25's 8.31 % of 4179.7) | 347.3 | **805.5** |
| `forward_into` `not_complete` (8.20 % of 3090.08) | 253.4 | **587.7** |
| `forward_into` `load_not_complete` (3.87 % of 2558.77) | 99.0 | **229.6** |
| **execution-only stall** (the difference) | **154.4** | **358.1** |

**One evaluation of this net costs ~805 cycles, and ~358 of them are cycles in
which the oldest instruction had not completed for a reason that is not
memory.** That is the quantity the commission pointed at, now in units where it
can be compared against a kernel measurement.

§3 accounts for the shape of it: `l2_tail` contributes ~32 cycles/eval of true
exposed latency (of which 17.9 is realisable), and every other head kernel is at
its ILP knee. The remainder is not latency at all — it is ~950 vector operations
per eval queueing through pipes that accept 1 (FMA/VNNI) or ~2 (simple integer)
per cycle. **The op-count estimate in that last sentence is an estimate and is
flagged as such in §10** — it is the one number in this section I did not
measure directly.

**And this, not a dependency chain, is the answer to the commissioned
question.** `forward_into` shows 8.20 % of `not_complete` against 3.87 % of
`load_not_complete` — 2.1× more total stall than load stall — because the head
is a dense stream of 512-bit ops issuing through **one** effective FMA/VNNI pipe
while touching relatively little memory. `ex_no_retire.not_complete` counts
cycles where the oldest op has not completed; an op waiting its turn at a
saturated 1-per-cycle pipe is exactly that, and it is **not** a dependency chain
you can shorten with more accumulators. The §3 audit is the direct evidence: 4
of the 5 head kernels are already at their ILP knee, and the one that is not is
worth 0.185 %.

**The actionable form of this finding: the lever on the head is the NUMBER of
512-bit ops, not their arrangement.** That is study 44 §1.3's own conclusion
("the prize is in reducing the NUMBER of memory operations") generalised from
memory ops to all ops — and it points at study 26 §8 item 1 (the merged
`pst + 4·threat` accumulator, which deletes 6 of `activate`'s 16 ops per
iteration and 2 of its 4 loads, estimated 60–120 i/n) as the correctly-aimed
next lever, rather than at anything inside `forward_into`'s arrangement.

---

## 6. `REGS_CAP` — finishing study 44's sweep

Study 44 shipped `REGS_CAP = 5` with the tile response **still falling** at
`r5`, and recorded in its §8 that `r4` was never built and `r8` was measured
only on the instrument it had already shown to be broken. Both are settled here.

Feature arms already existed (`sota_simd.rs:1680`). Built with
`tools/perf46/build46.sh`, one script, differing only in the feature list.

| binary | `REGS` | fold-kernel bytes | vs study 44 |
|---|---:|---:|---|
| `base46` (plain `main`) | 5 | **13 388** | **identical to study 44's `r5`** |
| `r4` (`regs4_44`) | 4 | 14 112 | never built before |
| `r8` (`regs8_44`) | 8 | **12 426** | **identical to study 44's `r8`** |

Two of the three byte sizes reproduce study 44's exactly — an independent check
that this study's base is the same code.

### 6.1 ★ MY INSTRUMENT FAILED FIRST, AND THE NULL CAUGHT IT

The first panel (24 rounds, cpu4) read the **byte-identical null at +0.878 %
`cycclean`** (trimmed +0.629 %, se 0.261, 9/24). A byte-identical binary cannot
be 0.9 % slower, and the null sat *between* the two arms (`r4` +0.334 %, `r8`
+1.066 %), so by rule 1 the panel could not answer the question.

**The cause was mine.** I ran §1's and §3's microbenchmarks — saturating a core
each, dozens of runs — concurrently with the panel. Study 44 §7 states the rule
("No panel was started while a build of mine was running") and I broke it with
measurement load rather than build load. `thread_not_selected` on the null read
+10.8 % median, which is the fingerprint.

The panel was discarded, **preserved as
`perf-artifacts-46/panel-regs-CONTAMINATED.txt` rather than deleted**, and
re-run on `cpu0` with nothing else of mine on the box. Core choice by
physical-core busy (summing SMT sibling pairs, study 44 §7's method): core2 was
quietest at 20.0 % but is the house default other agents pin to, so **cpu0**
(core0, 26.1 %) was taken instead.

The exact-counter rows are unaffected by load and are worth recording from the
contaminated panel anyway, since they are deterministic:
`instructions:u` for `r4` = **+1.125 %** (0/24, exact) and `r8` = **−1.656 %**
(24/24, exact) against `r5`. Combined with study 44's ladder against `REGS = 20`
(`r10` +1.261 %, `r8` +2.305 %, `r5` +4.076 %), the instruction cost is monotone
in tile narrowness across all five settings — `r4` is the most expensive tile on
`instructions:u` ever measured here, as designed.

### 6.2 THE VERDICT

*(panel in flight — filled in below)*

---

## 7. GATES

| gate | result |
|---|---|
| node identity, `base46` / `r4` / `r8` / `null46` | **5 001 562** — exact, all four |
| d13 node identity, all four | **1 739 588** — exact, all four |
| wl50 eval hash, all four | `2e1ede95…79c4` — exact, all four |
| `null46` byte-identical to `base46` | `cmp` clean |
| `base46` fold-kernel bytes vs study 44 `r5` | **13 388 both** |
| `r8` fold-kernel bytes vs study 44 `r8` | **12 426 both** |

---

## 8. ASSUMPTIONS

| # | assumption | why believed | what breaks if false |
|---|---|---|---|
| A1 | **The generated-asm sweep measures issue latency/throughput, not a frequency or loop artifact** | Validated by an internal control: the same instructions at 256 bits give a *different* clean answer (0.527, knee 8) with the same latency (4.005). Min-of-5 with no compiler in the loop | §1.2, and with it §3, §4 and §5 |
| A2 | **`evals/node = 0.43117`** | Exact atomic counters, not sampling; identical in two independent studies' artifacts on the identical frame | Every per-eval → per-node conversion; §4's rejection could flip if it were ~1 |
| A3 | **A chained-call microbenchmark of a kernel measures the latency the engine experiences** | `forward_into` is a strictly serial pipeline, so latency is the right axis; differences are quoted, never absolute levels | §3's rankings. §3.1's independent reproduction of the §1.2 knee argues it is sound |
| A4 | **`fp_ops_retired_by_width.all` counting semantics remain unresolved** | **NOT assumed — inherited as an open bracket from study 44 A4 and kept open.** §5 quotes all four combinations | Nothing; §5's conclusion (the bracket roughly doubles at the top) holds at either semantics |
| A5 | **Symbol-level attribution in study 44 §2 transfers** | Inherited; not re-derived here | §5's framing of *which* symbol is issue-bound, not the issue-rate measurement itself |
| A6 | **`l2_tail`'s reassociation is the only route to its ILP** | Argued from the definition of a left-to-right float sum, not measured | §4's rejection. If a bit-identical route existed the trade would change |

---

## 9. REPRODUCTION

```bash
# 0. worktree needs the gitignored net visible under its own path
mkdir -p data/nnue/nets/screens
ln -sfn /path/to/d48x-1280-sb520-ema-s600.cenn \
        data/nnue/nets/screens/d48x-1280-sb520-ema-s600.cenn

# 1. THE ILP SWEEP (no repo build needed; settles the commission's question 2)
python3 tools/perf46/gen_ilp.py /tmp/ilp.c
gcc -O2 -mavx512f -mavx512vnni -mavx512bw -mavx512vl -o /tmp/ilp /tmp/ilp.c
# cycles/op for <insn> at <nchains>; the knee is latency x throughput
perf stat -e cycles:u -- taskset -c 0 /tmp/ilp vpdpbusd 4 12500000

# 2. the kernel-shape microbenchmark (chained calls = a latency instrument)
rustc -O -C target-cpu=native --edition 2024 -o /tmp/kern tools/perf46/kern.rs
perf stat -e cycles:u -- taskset -c 0 /tmp/kern l2tail2 3000000

# 3. build + GATE the REGS arms
bash tools/perf46/build46.sh base46            # REGS = 5, what ships
bash tools/perf46/build46.sh r4  regs4_44
bash tools/perf46/build46.sh r8  regs8_44
cp bin46/base46 bin46/null46                   # the byte-identical null
P46_CPU=0 bash tools/perf46/ident46.sh base46 r4 r8 null46

# 4. the panel -- pick a quiet PHYSICAL core, and run NOTHING ELSE (§6.1)
P46_CPU=0 \
P46_EV='cycles:u,instructions:u,ex_no_retire.thread_not_selected:u,ex_no_retire.load_not_complete:u,de_dis_dispatch_token_stalls1.store_queue_rsrc_stall:u' \
P46_PANEL=docs/research/perf-artifacts-46/panel-regs.txt \
  bash tools/perf46/panel46.sh 40 base46 r4 r8 null46
python3 tools/perf46/med46.py docs/research/perf-artifacts-46/panel-regs.txt \
  base46 r4 r8 null46
```

---

## 10. WHAT I COULD NOT ESTABLISH

*(filled in with the panel verdict)*
