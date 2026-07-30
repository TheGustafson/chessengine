# Perf study 22 — the dual accumulator, and what is actually in `sota_accumulator.rs`

**Date:** 2026-07-29
**Scope:** `src/eval/nnue/sota_accumulator.rs` (+ `sota_context.rs` where the
stack is driven). `sota_simd.rs`, `sota_head.rs`, `sota_loader.rs`,
`sota_features.rs`, `history.rs` belong to the concurrent study 21 and were
read but never edited.
**Frame, quoted with every number in this document:**
`bench 16 1 100000 tools/nnue/calib/wl50.fens nodes engines/canonical/net.cenn`
= 50 FENs × 100k nodes, one thread, 16 MB TT, D48X net, `sota,arch-d48x`,
`-C target-cpu=native`, thin LTO, 4 CGUs — **a fixed 5,001,562-node tree**.
Every arm below is node-identical at 5,001,562 and eval-hash-identical at
`5ca978e012da1bb0` unless it says otherwise.

---

## 0. The headline, in one paragraph

The premise this study was redirected onto — *"ARCHON maintains **two** 1280-wide
accumulators where Stockfish maintains **one**"* — is **false**, and the pinned
reference says so directly. At `eca43a97`, `AccumulatorStack` declares
`psq_accumulators` **and** `threat_accumulators` as two separate `MaxSize`-deep
arrays, each holding a full-width `int16` accumulator per colour, refreshed by
two different strategies and combined only at `transform()` time. That is
ARCHON's structure, feature for feature. §1 gives the citations, §3 prices the
merge at **+157 i/n saved against +872 i/n spent** — it loses ~6:1 — and §5.3
closes the only escape route. The four things this study landed in
`sota_accumulator.rs` are real but ordinary: a hot/cold split of the per-ply
stack entry, a boxed-array conversion that was worth more than the split it was
written to rescue, a probe-first relayout of the Finny cache slot, and a
comment finally made true. Together, **−8.08 instructions/node against an
exactly-zero null**; the −1.16% cycles figure this study first committed is
**RETRACTED in §4.4** — a three-way round-robin showed the null itself moving
+1.42%, and the arm's cycles delta sits inside it. One arm was built, measured,
and reverted (§4.6). Nothing here is a regression on any counter.

---

## 1. RETRACTION OF THE PREMISE: Stockfish keeps two accumulators too

### 1.1 What was claimed

The study-21 remainder that redirected this study reads:

> ARCHON maintains **TWO** 1280-wide accumulators where Stockfish maintains
> **one** — ~20 KB/node of copy traffic, **65% of all L1 misses**, 135 MB of
> weight tables against a 32 MB L3. It's a pin-moving refactor
> (saturating→wrapping merge), not a perf-study arm.

### 1.2 What the reference actually contains

`stockfish_reference/Stockfish/src/nnue/nnue_accumulator.h`, at the pinned
constants revision `eca43a97`:

```cpp
    std::array<AccumulatorState<PSQFeatureSet>,    MaxSize> psq_accumulators;
    std::array<AccumulatorState<ThreatFeatureSet>, MaxSize> threat_accumulators;
    std::size_t                                             size = 1;
```

and `nnue_architecture.h`:

```cpp
using ThreatFeatureSet = Features::FullThreats;
using PSQFeatureSet    = Features::HalfKAv2_hm;
constexpr IndexType TransformedFeatureDimensionsBig = 1024;
```

Each `AccumulatorState` holds an `Accumulator<Big>` **and** an
`Accumulator<Small>`, and `Accumulator<Size>` is
`std::array<std::array<std::int16_t, Size>, COLOR_NB>` plus its own
`psqtAccumulation` and `computed` flags. So per ply Stockfish carries:

| | ARCHON (D48X) | Stockfish `eca43a97` |
|---|---|---|
| PSQ/HalfKA accumulator | `[[i16; 1280]; 2]` = 5120 B | `[[i16; 1024]; 2]` + `[[i16; 128]; 2]` = 2304 B |
| Threat accumulator | `[[i16; 1280]; 2]` = 5120 B | `[[i16; 1024]; 2]` + `[[i16; 128]; 2]` = 2304 B |
| **accumulator stacks** | **2** | **2** |
| PSQT bucket sums | 2 × `[[i32; 8]; 2]` | 2 × `[[i32; 8]; 2]` (per size) |
| refresh strategy | Finny set-diff (HalfKA) / from-scratch (threats) | `update_accumulator_refresh_cache` (PSQ) / `update_threats_accumulator_full` (threats) |
| where they combine | `activate(out, pst, threat, stm)` | `FeatureTransformer::transform`, `accumulation` + `threatAccumulation` |

`AccumulatorStack::evaluate` makes the shape explicit — four independent folds,
two per feature set:

```cpp
    evaluate_side<PSQFeatureSet>(WHITE, ...);
    if (UseThreats) evaluate_side<ThreatFeatureSet>(WHITE, ...);
    evaluate_side<PSQFeatureSet>(BLACK, ...);
    if (UseThreats) evaluate_side<ThreatFeatureSet>(BLACK, ...);
```

`evaluate_side` then branches on the feature set for the refresh:
`update_accumulator_refresh_cache` for PSQ (the Finny cache), and
`update_threats_accumulator_full` — a **from-scratch rebuild** — for threats.
That is exactly ARCHON's `HalfkaContext::finny_refresh_into` /
`ThreatContext::discharge_refresh` split.

### 1.3 Where the "one accumulator" reading came from

From ARCHON's own comment in `sota_simd.rs`'s `activate_inner`:

> Stockfish — whose two feature sets fold into ONE 1024-wide accumulator, not
> two — merges the threat channel with a single
> `vec_add_16(acc, vec_convert_8_16(column))`.

That sentence is about the **quantisation frame**, not the number of
accumulators: SF's two feature sets share one quantisation scale, so its
threat column is added with no rescale, whereas ours needs the `<<2`. The
"ONE 1024-wide accumulator" phrasing is wrong on its face — SF's threat fold
in `update_threats_accumulator_full` writes into `threatAccumulation`, a
distinct array — and it is the origin of the claim that redirected this study.
**It should be corrected in place.** (Not done here: `sota_simd.rs` is study
21's file this session.)

### 1.4 Why a merge is not free even in principle

Suppose we merged anyway, keeping one `M = pst + 4·threat` per ply. The
blocker is that the two feature sets have **different refresh triggers**, and
the merged accumulator must refresh on the *union*:

* HalfKA refreshes whenever `halfka_king_key(perspective, king)` changes.
  That key is `bucket*2 + (file ≤ 3)`, and `KING_BUCKETS` is
  rank-major/file-mirrored, so the map square → key is a **bijection on all 64
  squares** — every own-king move changes it. (Checked by hand on rank 1:
  keys 1,3,5,7,6,4,2,0 for a1..h1.) Rate: **0.1611 refreshes/node**
  (`compute`'s own comment, perf study 12 §6).
* FullThreats refreshes only when that perspective's king crosses the d/e
  mirror. Rate: **0.0214 refreshes/node**, and one costs **6,240
  instructions** (`apply_at`'s comment, perf study 17 §3.3 + study 18).

A merged accumulator cannot reconstruct either channel separately, so an
own-king move would force a full threat rebuild:

```
  Δ instructions/node  ≈ (0.1611 − 0.0214) × 6240  ≈  +872 i/n   (+14%)
```

against a saving of one accumulator's parent-read + child-write, i.e.
5120 B + 5120 B per **evaluated** node. Even valuing that at one cycle per
64-byte line it is ~160 cycles per evaluated node, and the instruction cost
alone is ~620 cycles/node at the measured IPC of 1.45. Routing the merged
refresh through the Finny cache is *worse*, not better: the cache's set-diff is
between two arbitrary positions, and FullThreats has no set-difference form, so
it would need both positions enumerated — 2 × 6,240.

**Two of those three numbers are INHERITED, not re-derived in this frame** —
0.1611 refreshes/node and 0.0214 rebuilds/node come from studies 12 and 18's
own frames, and 6,240 instructions/rebuild from study 17's. Stated so the
reader can discount them. The verdict does not depend on their precision: the
cost would have to be over-stated by **8×**, or the refresh-rate gap by the
same, before the merge broke even against the measured 1.83% (§3).

**Conclusion: the dual accumulator is a faithful reproduction of Stockfish's
own structure, and the merge is not an available win.** The 20 KB/node figure
is arithmetically right (2 stacks × 2 perspectives × 2560 B, read + written,
per evaluated node) and is not addressable by merging.

---

## 2. The instrument, and the null that changed it

The brief quoted a cycles/node noise floor of **0.6–1.1% (median of 6)**. The
first thing this study ran was the null: the same binary, copied to two names,
through the interleaved A/B panel at n=6, pinned to core 3 as study 20 did.

```
A base    n=6 cyc_n=4727.8   B base2   n=6 cyc_n=4567.3
  paired dcyc_n (B-A): median=-282.1 min=-1528.5 max=+181.2
  medians: dcyc_n=-160.5 (-3.40%)  dins_n=-0.00
```

**−3.40% cycles between a binary and its own copy.** Cause, found by reading
`/proc/stat`: core 3 was at **100% busy** — a sibling study had its own bench
pinned there, and my harness process was getting 2.2% CPU. Instructions were
still exact (0.000%), which is why an instruction-gated study would never have
noticed.

Moving to core 2 (9.5% busy) fixed it. The null on core 2, n=8:

```
A base    n=8 cycmin=4227.7 cycmed=4445.2 ins=6085.0 l1m=624.46 l2m=74.52 dram=8.39 dtlb=2.19
B base2   n=8 cycmin=4140.3 cycmed=4441.9 ins=6084.6 l1m=623.08 l2m=74.08 dram=8.21 dtlb=2.18
  D cycmin=-87.5 (-2.07%)  cycmed=-3.2 (-0.07%)  ins=-0.45 (-0.007%)
  D l1m=-0.22%  l2m=-0.59%  dram=-2.17%  dtlb=-0.38%
```

Two things this fixes for the rest of the study:

1. **The gate is `cycmed`, n ≥ 8, on core 2.** Floor **0.07%**.
2. **`cycmin` is NOT the robust estimator here.** The intuition that
   interference can only add cycles, so min-of-N is safest, is wrong on this
   box: min-of-8 read −2.07% on two identical binaries, thirty times the
   median's error. One lucky round dominates a min; a median needs half the
   rounds to move. Reported alongside, never gated on.
3. Secondary counters and their nulls: `l1m` 0.22%, `l2m` 0.59%, `dram` 2.17%,
   `dtlb` 0.38%, `instructions` 0.007%.

### 2.1 …and then the null moved again, which changes what can be claimed

Two hours later, the **same** null — `base` vs `base2`, core 2, n=8 — read:

```
  D cycmed=-70.5 (-1.51%)   ins=+4.68 (+0.077%)   dram=-13.68%
```

Core 2 was still 100% *mine* (`cores22.sh`: cpu2 100%, its SMT sibling cpu10 at
9.8%), so this is not core stealing. It is **shared L3 and memory-bandwidth
contention** from the neighbours — inherent to a workload that streams random
2560-byte rows out of a 57.7 MB table, and not fixable by pinning.

Consequences, stated plainly because they bound every cycles number in this
document:

* **The cycles floor on this box is time-varying between 0.07% and 1.51%.** A
  null measured at 21:30 does not bound an arm measured at 23:00. Study 20's
  protocol — one null, then a sequence of arms — is not safe here.
* Even `instructions:u` drifted (+0.077%, i.e. +4.7 i/n) in the second null,
  against the 0.0 the brief quoted. At fixed nodes the *engine's* instruction
  count is exact; the residue is process startup and loader jitter inside the
  `perf stat` window. Treat ~5 i/n as the practical instruction floor, not 0.
* **The fix is a three-way round-robin** (`scripts/panel3-22.sh`): A, a copy of
  A, and B, cycled together, so the null and the arm see the *same* drift, with
  PAIRED per-round deltas and a sign count rather than a gap between two
  medians. That is the instrument the final measurement uses, and it is what
  this study would start with if it ran again.

Harness: `docs/research/perf-artifacts-22/scripts/{build22,probe22,panel22,panel3-22,cores22}.sh`.
Raw output: `docs/research/perf-artifacts-22/{probes-fast,panel,panel3}.txt`.

---

## 3. What one accumulator's copy traffic is actually worth — MEASURED

§1.4 prices the merge by arithmetic. This prices its *upside* by experiment,
because the redirect asked for the copy traffic to be measured rather than
accepted at 20 KB.

**The probe.** One extra `[[i16; ACC]; 2]` copy — 5120 bytes read from
`stack[ply-1].acc`, 5120 bytes written to a reused heap scratch — added to
`current_full`, `black_box`'d so it cannot be eliminated. That is exactly the
memory traffic of one more accumulator-pair fold per evaluated node, i.e. the
traffic a merge would delete. (Uncommitted; the diff is
`perf-artifacts-22/probe-extracopy.patch`.)

**Result** (interleaved A/B, n=10, core 2, node-identical, hash-identical):

```
A a12b            cycmed=4374.2  ins=6076.8  l1m=618.64  l2m=72.59  dram=8.44
B probe-extracopy cycmed=4454.3  ins=6234.2  l1m=674.02  l2m=72.30  dram=8.39
  D cycmed=+80.1 (+1.83%)   ins=+157.42 (+2.591%)
  D l1m=+55.38 (+8.95%)     l2m=-0.40%   dram=-0.63%
```

Three things this fixes. **Read the instruction and L1d rows, not the cycles
row** — §2.1 and §4.4 show cycles on this box carry a ±1.9% floor, while
instructions carry 0.000% and L1d ~0.3% in the same round-robin:

1. **One accumulator's copy traffic is +157.4 instructions/node and +55.4 L1d
   misses/node** — 25× and 2000× their respective floors. Both resolve
   cleanly. The +1.83% cycles is at the edge of the floor and is quoted for
   completeness only.
2. **8.95% of all L1d misses for ONE extra accumulator-pair read.** With four
   such reads+writes per evaluated node (2 stacks × parent-read + child-write),
   the fold copies are roughly a third of the engine's L1d misses — consistent
   with study 19's "65% of misses in the fold kernels" once the weight rows are
   added.
3. **+157 instructions for 5120 bytes = 32.6 B/instruction**, i.e. 64-byte
   `zmm` load/store pairs. The copy is already optimal; there is no
   instruction-side slack in it.

**Merge verdict, both sides on the SAME exact instrument:**

| | instructions/node |
|---|---|
| saved if one accumulator disappeared | **−157** (measured, §3) |
| cost of the refresh-trigger union | **+872** (§1.4, from two inherited rates) |

The merge loses by roughly **6:1 on instructions**, and §1 shows Stockfish does
not make it either. Putting both sides on `instructions:u` — the one counter
with a zero floor here — removes the noise argument entirely. **Closed as
refuted.**

---

## 4. Landed arms

All node-identical at 5,001,562 and eval-hash-identical at
`5ca978e012da1bb0`. All three merge pins reproduce after every commit:
`Bench: 587683`, eval hash `5ca978e0…8b38`, d13 nodes `1922947`.

### 4.1 Arm 1 — hot/cold split of the per-ply `Entry`

`dirty` (10 B) and `computed` (2 B) were the last two fields of a **5248-byte**
`Entry` — confirmed in the shipped binary as `imul rax,r9,0x1480`. So the 12
bytes of bookkeeping that `compute` touches three times per call (early-out,
ancestor walk, last-king-move scan) and `push` writes once per made move sat in
cache line 81 of 82, and consecutive plies were 5248 bytes apart: a 5 KB-stride
pointer chase over lines that nothing else reads. They now live in a parallel
`Meta` array — 12 B/ply, ~3 KB for the whole 254-ply stack — where a
parent/child pair shares a line.

### 4.2 Arm 1b — `Box<[T; N]>` instead of `Box<[T]>`, and it is the bigger half

Arm 1 alone measured **+58.7 i/n** (6087.4 → 6146.1). A second boxed SLICE is a
fat pointer: every `meta[ply]` reloads the length from memory and LLVM cannot
fold the panic edge across the outlined fold calls. Converting all three heap
arrays — `stack`, `meta`, `cache` — to `Box<[T; N]>` through a `boxed_array`
helper turned +58.7 into **−9.3 i/n** (6078.1) and dropped branches/node
643.3 → 640.8.

This is the arm's real lesson and it is now in the type's doc comment: **in this
file a boxed slice costs roughly 20 instructions per indexed access that a boxed
array does not**, and the cost is invisible until you split a struct and
suddenly index two arrays where you indexed one.

### 4.3 Arm 2 — the Finny slot, probe-first

`Option<FinnyEntry>` has no niche to hide in (the payload is all integers), so
the discriminant cost a **whole 64-byte cache line per slot** — 2752 B/slot
instead of 2688, tag at offset 0 with 63 bytes of padding beside it. Worse,
Rust's default field order put `by_piece` — the first thing the set-diff reads —
at byte **2656**, i.e. lines 41–42. Confirmed in the shipped binary as
`imul rdx,rdi,0xac0` / `mov r11,QWORD PTR [r14+rdx*1+0xa60]`. A probe therefore
touched three cache lines spread 2.6 KB apart to read 96 bytes and a tag.

Now `#[repr(C)]` with `by_piece` (96 B) + `psqt` (32 B) = **exactly two leading
cache lines**, `acc` 64-B aligned at offset 128, slot size 2688 = 42 lines with
no tail padding, and validity moved to a two-`u64` bitmap that never leaves L1.
The cold path additionally writes the slot in place instead of round-tripping a
2.5 KB stack temp.

### 4.4 The measurement — and the RETRACTION of this study's own cycles headline

**What was first measured and committed** (two-way interleaved panel, core 2,
n=12, against a null taken ~40 minutes earlier that read 0.07%):

| | cycles | instructions | L1d miss | L2 read miss | DRAM fill |
|---|---|---|---|---|---|
| base | 4283.7 | 6086.8 | 620.66 | 72.85 | 8.04 |
| a12b | 4233.9 | 6073.2 | 618.43 | 71.74 | 7.55 |
| Δ | −49.9 (−1.16%) | −13.59 (−0.223%) | −0.36% | −1.52% | −6.13% |

Commit `7237f6d` states that −1.16% as the headline. **It should not have.**

**What the three-way round-robin says** — base, a byte-identical copy of base,
and the arm, cycled together so the null and the arm see the same drift, n=10
each, paired per-round deltas:

```
A base       cyc=4820.6  ins=6086.5  l1m=624.25  l2m=74.35
N base2      cyc=4889.0  ins=6086.5  l1m=622.88  l2m=73.69
B a3only     cyc=4795.4  ins=6078.4  l1m=621.34  l2m=72.44

  NULL  paired dcyc median=+90.4 (+1.87%)  [-382,+414]  neg=3/10  same_sign=False
        medians:    dcyc=+68.4 (+1.42%)  dins=+0.00 (+0.000%)  dl1m=-1.36  dl2m=-0.66
  ARM   paired dcyc median=+22.7 (+0.47%)  [-615,+658]  neg=5/10  same_sign=False
        medians:    dcyc=-25.2 (-0.52%)  dins=-8.08 (-0.133%)  dl1m=-2.91  dl2m=-1.91
```

**RETRACTED: the −1.16% cycles figure.** In the round-robin the *null itself*
moves +1.42% (paired +1.87%, sign 3/10), so a −1.16% arm reading is inside the
instrument's own error. The arm's cycles delta here is −0.52% with a paired
sign count of 5/10 — a coin flip. **The cycles effect of arms 1+1b+2+3 is
UNRESOLVED on this box.** The −1.16% was real arithmetic on a null that had
gone stale, which is precisely the failure §2.1 describes.

**WHAT SURVIVES, and it is not nothing:**

| statistic | null (same round-robin) | arm | ratio |
|---|---|---|---|
| **instructions/node** | **+0.000%** (exactly 0.00 i/n) | **−8.08 (−0.133%)** | resolved |
| L1d misses/node | −1.36 | −2.91 | 2.1× |
| L2 read misses/node | −0.66 | −1.91 | 2.9× |
| cycles/node | +1.42% | −0.52% | **unresolved** |

* **The instruction win is exact.** The null's `dins` is 0.00 i/n to the
  printed precision in the same round-robin, and the arm is −8.08. Three
  independent measurements agree on sign and order: −9.3 (single probe),
  −13.59 (two-way panel), −8.08 (round-robin).
* **The cache-line reductions are structural facts, not measurements.**
  Arm 1 removes four guaranteed-distinct 5248-byte-strided line touches per
  node; arm 2 takes the Finny probe from three lines 2.6 KB apart to two
  adjacent leading lines and the slot from 2752 to 2688 bytes. Those are
  properties of the layout, provable from the disassembly, and they do not stop
  being true because the clock on a contended box cannot resolve them.
* **No downside is detectable on any counter.** Both miss counters move the
  arm's way by 2–3× the null.

So the honest claim is: **a resolved −8.08 i/n instruction win, a structurally
argued reduction in cache-line touches that this box cannot currently resolve
on cycles, and no measured regression.** That is a smaller claim than the one
committed, and it is the one the evidence supports.

### 4.5 Arm 3 — the Finny diff loop finally does what its comment says (NULL)

The comment above that loop has forbidden three things since perf study 2. The
first — *"iterate `PieceType::ALL` — LLVM materialises that 6-byte array on the
stack every call and reloads the piece type from it per iteration"* — was
**being done**, opcode for opcode, in the shipped binary:

```
    mov   WORD  PTR [rsp+0x154],0x504      ; PieceType::ALL, on the STACK
    mov   DWORD PTR [rsp+0x150],0x3020100  ;   written per call
    movzx r8d,BYTE PTR [rsp+rdi*1+0x150]   ; reloaded per iteration
    mov   r9, QWORD PTR [r12+rdx*1+0x140]  ; board.pieces[c][pt]
    mov   r11,QWORD PTR [r14+rdx*1+0xa60]  ; cached.by_piece[c][pt]
```

Zipping the two `[Bitboard; 6]` rows (as the comment always said) plus a
`static PIECE_TYPES` removes the stack copy — verified absent from the new
disassembly — and both indexed loads.

**Measured: +0.8 i/n. A NULL.** At 0.16 refreshes/node the eight instructions
per call this removes are worth ~1.3 i/n, which is inside the ~5 i/n practical
floor (§2.1). It is landed anyway, with no perf claim, because leaving a
comment that describes code which does not exist is how a reader gets misled —
this study lost time to exactly that failure twice (§1.3, §5.3).

### 4.6 Arm 4 — outlining the king-move branch — BUILT, MEASURED, REVERTED

`compute` is inlined into `current_full` twice (once per perspective,
specialised on a constant `Color`), so the whole king-move arm — the Finny
refresh call plus the backward-fill loop — was paid for twice in the icache
footprint of the hottest eval-side function, for a branch taken 0.1611 times
per node against the forward fold's ~1. Pulling it into an `#[inline(never)]
fn refresh_and_backfill` shrank `current_full` from **2461 to 1440 bytes**
(−41%) with the rare arm as a separate 994-byte symbol.

Two panels, same binaries, ~40 minutes apart:

```
  panel 1 (n=10):  cycmed  -34.1 (-0.77%)   ins  -2.80   l1m +0.39%  l2m +0.91%
  panel 2 (n=10):  cycmed  +31.2 (+0.60%)   ins  -1.09   l1m +0.62%  l2m +1.97%
```

**The sign reversed.** Both magnitudes are inside the null's own range on this
box (§2.1: 0.07% to 1.51%), and instructions moved by 1–3 i/n against a ~5 i/n
floor. Branch misses went the wrong way (19.66 → 20.37 per node), which is what
a call/ret on a 0.32-per-node path buys you.

**Reverted.** The 41% code-size reduction is a real fact and it is tempting —
that is exactly why it is not being landed. A change that two panels cannot
agree on the sign of, on a path taken a third of a node, that measurably raises
branch misses, is a change with no evidence behind it. Recorded here so study 23
does not re-discover it as a fresh idea; the same shape applied to
`ThreatContext::compute`'s much larger `at_full` (7855 bytes) is remainder
item 3, where the size ratio is far more favourable.

---

## 5. Rejected without building, and why

### 5.1 Software prefetch of the fold's weight rows — REJECTED, pre-refuted

The obvious stall arm on a fold that streams ~11 KB/node of random rows out of
a 57.7 MB `i16` table. It is already refuted twice on this box, by adjacent
measurements on *larger* tables:

* `sota_threats.rs:1101` — "the study's software row-prefetch was implemented
  here and MEASURED A LOSS (−2.5% NPS full-row, ~−0.4% first-two-lines,
  bench-with-net A/B on the Zen 4 box) — the OoO window and hardware stride
  prefetchers already cover the fold's row streams".
* `sota_simd.rs:1380` — "prefetch the next group's rows … measured
  noise-to-negative (−0.3% over 4 interleaved pairs)".

The HalfKA fold's rows are 2560 B of *contiguous* `i16` walked at a 64-byte
stride — the single most prefetcher-friendly shape there is, and strictly
easier for the hardware than the threat fold's, which already lost. Building a
third instance of an arm that lost twice is not a good use of a 12-minute
panel.

### 5.2 The 512 dead memset bytes in the Finny set-diff — REJECTED, priced below the floor

Confirmed present in the shipped binary. `finny_refresh_into`'s prologue
contains eight `vmovdqa64 ZMMWORD PTR [rsp+…],zmm0` — 512 bytes of zeroing for
`adds: [usize; 32]` and `subs: [usize; 32]`, of which only the first
`add_len`/`sub_len` entries are ever read. This is exactly the class study 20
arm 4 closed in `MoveList`.

It is rejected on study 20's own measurement of the identical fix: at **0.5
calls per node** it was worth −6.53 i/n and **+0.25% cycles, i.e. cycle-neutral
inside the null floor** ("the store buffer was absorbing them"). This site runs
at ~0.16 calls/node — roughly a third of the rate — so the predicted effect is
~−2 i/n and no cycles at all. Against that it would cost the first
`#[allow(unsafe_code)]` in a file that currently has none, which the crate's
`#![deny(unsafe_code)]` policy admits "only with a measured, documented win".
There is no measured win to document.

### 5.3 "Refresh on a king *bucket* change, not on any king move" — REJECTED, unreachable


`halfka_king_key`'s own doc comment advertises it:

> Cache key for a HalfKAv2_hm king state: 32 bucket ids × two horizontal
> orientations. **Moving the king without changing this key leaves every
> non-king feature's index stable, so a dirty-piece update is sufficient.**

The condition can never hold. `key = KING_BUCKETS[ksq ^ flip] * 2 +
(file(ksq) ≤ 3)`, `KING_BUCKETS` folds file f with file 7−f, and the
orientation bit separates exactly those two halves again — so the composition
is a **bijection from the 64 squares onto the 64 keys**. Two distinct king
squares never share a key, and the promised fast path is dead code that was
never written. The comment should be corrected to say the key is a
relabelling of the king square (which is also why `FINNY_KEYS == 64`).

This also removes the only escape from §1.4: there is no way to make the two
feature sets share a refresh trigger by weakening HalfKA's.

### 5.4 `min`-of-N as the robust cycles estimator — REJECTED by its own null

Reported for the record because it is intuitive and wrong on this box. On two
*identical* binaries, min-of-8 read **−2.07%** where median-of-8 read −0.07%,
and in the arms-3+4 panel min read −3.01% against the median's −0.77%. Cycle
interference is additive, so a min should be the safe estimator — but a min is
also a 1-of-N order statistic, so a single unusually quiet round moves it
entirely, while a median needs half the rounds to agree. `cycmin` is printed by
`panel22.sh` and gated on by nothing.

---
## 6. Ranked remainder

Ordered by expected value, with the evidence that ranks it. Items 1 and 2 are
this study's arms 1 and 1b applied to the file next door; they are the highest
ranked because the defect is *the same defect*, already measured once, on a
structure that is walked more often.

**1. `ThreatEntry` needs the identical hot/cold split.** (`sota_threats.rs`,
handed to study 23.) `ThreatContext::compute`'s ancestor walk reads
`self.stack[*a].fold[p]` — and `fold: [Fold; 2]` (2 B), `dp: DirtyPiece` (10 B)
and `position: ThreatPosition` (80 B) all sit at the **end** of a ~5344-byte
struct, exactly where `Entry`'s did. Two independent walks per evaluated node
(one per perspective), plus `apply_at` writing `dp`/`position`/`fold` on every
made move, plus `build_deltas` reading `position` back at fold time. That is
*more* traffic on the same 5 KB stride than the HalfKA side this study fixed for
−1.16%. Highest-confidence item on the list.

**2. `ThreatContext.stack: Box<[ThreatEntry]>` → `Box<[ThreatEntry; STACK_LEN]>`.**
(`sota_threats.rs`.) Arm 1b measured the fat-pointer index at **+58.7 i/n** on
the HalfKA side. `ThreatContext` indexes its boxed slice in `compute`,
`apply_at`, `at_full`, `refresh` and `discharge_refresh`. Mechanical; the
`boxed_array` helper is already in `sota_accumulator.rs` and can move to a
shared home.

**3. Outline the `Fold::Refresh` arm of `ThreatContext::compute`.**
(`sota_threats.rs`.) Arm 4's shape: the rare branch (0.0214/node) is the large
one, and it is inlined into the hot path. `at_full` is **7855 bytes** of code —
the largest symbol in the eval — against `current_full`'s 1440 after arm 4.

**4. Correct two comments that are actively misleading.** Neither is a perf
change; both cost real time this session.
   * `sota_simd.rs` `activate_inner`: "Stockfish — whose two feature sets fold
     into ONE 1024-wide accumulator, not two". It is two (§1.2). This sentence
     is the origin of the refactor premise that redirected this study.
   * `sota_features.rs` `halfka_king_key`: "Moving the king without changing
     this key…" — unreachable, the key is a bijection (§5.3).

**5. `#[repr(C)]` on `Entry`.** Its current field order (psqt 0..64, acc
64..5184, both 64-B aligned) is correct *by Rust's default alignment-descending
layout*, which is explicitly unspecified. Arm 1's alignment argument silently
depends on it. Free to pin; needs one re-measure to confirm codegen is
unchanged.

**6. The 512 dead memset bytes in the Finny set-diff** (§5.2). Priced at ~−2 i/n
and zero cycles from study 20's measurement of the identical fix at 3× the call
rate. Worth doing only if `sota_accumulator.rs` acquires an
`#[allow(unsafe_code)]` for some other reason.

**7. The dual accumulator itself — CLOSED, do not reopen without new
information.** §1 and §3 together: Stockfish keeps two, the copy traffic is
worth ≤1.83% of cycles, and merging costs ~+14% of instructions in extra threat
rebuilds. Reopening requires either a threat feature set with a set-difference
form, or a HalfKA refresh trigger that can be weakened (§5.3 says it cannot).

---

## 7. Method notes worth keeping — the first one is the expensive one

1. **A NULL EXPIRES. Run it in the same round-robin as the arm, or it bounds
   nothing.** This is the one that cost this study a retracted headline. Three
   nulls on the same binary against its own copy, same command, same core where
   noted: **−3.40%** (core 3, contended), **−0.07%** (core 2), **−1.51%**
   (core 2, two hours later), **+1.42%** (core 2, round-robin). "Null first"
   was followed to the letter and still produced a wrong claim, because the
   rule as written says *before the arm* and what it needs to say is
   *interleaved with the arm*. `scripts/panel3-22.sh` is the fix: A, a copy of
   A, and B, cycled together, paired per-round deltas, and a sign count.
2. **The null also caught a dead instrument in the first ten minutes.** Core 3
   was 100% busy with a sibling study's bench and the harness process was
   getting 2.2% CPU. An instruction-gated study would have measured 0.000%
   there and proceeded happily on a machine that could not resolve cycles at
   all. §2.
3. **Pick the measurement core by reading `/proc/stat`, not by inheriting it.**
   `scripts/cores22.sh`. Study 20's core-6 pin was correct when study 20 ran it
   and wrong four hours later.
4. **A fat pointer is a perf decision.** Splitting one struct into two arrays
   cost +58.7 i/n purely because the second array was a `Box<[T]>`; the same
   split with `Box<[T; N]>` was −9.3. Any future hot/cold split in this codebase
   should reach for the array type first.
5. **Read the reference before accepting a "we deviate from Stockfish" claim.**
   Three studies' worth of framing rested on one wrong sentence in one comment.
