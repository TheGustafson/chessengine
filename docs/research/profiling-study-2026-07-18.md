# ARCHON profiling study — where the cycles go, benchmarked against SF16/SF18/Reckless/Integral

Date: 2026-07-18. Scope: the **inference-speed / efficiency leg** of the D39 Elo
decomposition (Elo ≈ f(net quality, search quality, inference speed)). The
algorithmic/search-quality leg is owned by
`docs/research/what-holds-archon-back-2026-07-18.md` and is NOT re-derived here.
This is a study: analysis + ranked recommendations, no engine code was modified.

**Headline:** on the D40 sota path (HalfKAv2_hm 768 + FullThreats + hybrid head),
**~70% of all cycles are NNUE inference/maintenance**, and at least a quarter of
that is *implementation waste*, not architecture cost: a scalar-compiled f32 head
layer (10–12% of total runtime, 82% of `forward()` self-time, 28% of all
icache misses), 1.5 KB accumulator `memmove`s that fused kernels would eliminate
(9.4%), and always-paid threat-observer work of which ~55% is never consumed.
SF18 runs the *same feature-set family at 1024-wide* in **7.4k instructions/node
(AVX2 build)** where ARCHON needs **16.9k (AVX-512 build)** — a ~2.3× constant-factor
gap with the architecture held roughly equal. A low-risk package of 4 pure-speed
fixes worth an estimated **+20–30% NPS** on the sota path is identified; that is
the difference between D40's current equal-time parity (−8.7±12.9, exp-0093) and
banking its proven +30/node eval edge as real Elo.

---

## 1. Methodology

### 1.1 Environment and its two big caveats

- Box: AMD Ryzen 7 7700X (Zen 4, 8C/16T, AVX-512+VNNI capable), WSL2 kernel
  5.15.167.4-microsoft.
- **PMU: available after all.** The stock `/usr/bin/perf` wrapper refuses to run
  (no linux-tools for the Microsoft kernel string), but installing
  `linux-tools-generic` (6.8.0-136) and invoking
  `/usr/lib/linux-tools-6.8.0-136/perf` directly gives **working hardware
  counters** — `cycles`, `instructions`, `branches`, `branch-misses`,
  `L1-dcache/icache-*`, verified against a `dd` microbenchmark. This WSL2 kernel
  has PMU passthrough enabled, so the study got real `perf stat` + sampled
  `cycles:P` profiles instead of the expected software-clock fallback. Valgrind
  (callgrind/cachegrind) was still used where it is strictly better: exact
  per-function instruction counts, exact call counts, and modeled cache/branch
  attribution.
- **Contention:** a user-greenlit 10-worker SF18 relabel ran throughout
  (nice 10), plus an intermittent 4-process fixed-node fastchess screen (nice 0,
  ended mid-study). All ARCHON/SF16/Integral measurement runs were pinned
  (`taskset -c 14`) at normal priority, which preempts the niced relabel on that
  core; residual SMT-sibling, L3, and memory-bandwidth contention remains.
  **Absolute NPS numbers are therefore lower bounds** (historic clean perft
  baseline is ~280–300 Mnps vs ~160–210 Mnps measured here); **relative
  breakdowns (profile %, instr/node, per-function shares) are robust.**
  Reckless could not be pinned — its NUMA layer asserts under `taskset`
  (`src/numa.rs:150` in its tree), so it ran self-pinned/unpinned and its
  absolute NPS is additionally disadvantaged by perhaps 10–20%.

### 1.2 Builds and workloads

- Perf-sampling binary: current source,
  `RUSTFLAGS="-C target-cpu=native -C force-frame-pointers=yes" cargo build
  --profile profiling --features sota` (and `--features nnue-l1-2048` for the
  classic lane). Runtime kernel dispatch selects the AVX-512+VNNI backend on
  this CPU regardless of `target-cpu`. Note this build is **not PGO'd**; the
  deployed `engines/bin/probe-sota-pgo` is ~10% faster, so profile *shares* are
  from the un-PGO'd binary (PGO mostly relayouts code; shares shift only
  slightly).
- Valgrind lane: same source built `-C target-cpu=x86-64-v3` — **valgrind 3.22
  cannot decode EVEX** (the native build SIGILLs with "unhandled instruction
  bytes 0x62 ..."), and valgrind's virtual CPUID hides AVX-512, so under
  callgrind/cachegrind the engine runs its **AVX2** kernels. Call counts and
  algorithmic ratios are ISA-independent; kernel instruction counts quoted from
  callgrind are AVX2 (≈2× the AVX-512 dynamic count for the 16-lane loops) and
  are labeled as such.
- Workloads:
  - Hot-path profile: `bench 16 1 12 default depth data/nnue/nets/d40-uho-720sb.cenn`
    (3,515,701 nodes, ~8.6 s) under
    `perf record -F 999 -g --call-graph fp` (frame pointers compiled in), 9,615
    samples.
  - Callgrind/cachegrind: same bench at depth 8 (147,691 nodes) —
    `valgrind --tool=callgrind`, and `--tool=cachegrind --cache-sim=yes
    --branch-sim=yes --I1=32768,8,64 --D1=32768,8,64 --LL=33554432,16,64`
    (Zen-4-like L1 geometry, LL modeled as the 32 MB L3; the 1 MB per-core L2 is
    NOT modeled — D1 misses land in "L2-or-worse").
  - Cross-engine: each engine's own `bench` under `perf stat`, plus a common
    `position startpos / go nodes 500000` UCI run, plus perft (movegen isolate).
  - Nets: sota = `d40-uho-720sb.cenn` (81,354,848 bytes: HalfKA 22528×768 i16 ≈
    34.6 MB + FullThreats 60720×768 i8 ≈ 46.6 MB + head); classic =
    `stage2-v1-s3-2048-farseer-120sb.cenn` (3.16 MB).
- Reference binaries: `engines/bin/probe-sota-pgo`, `probe-l2048-pgov3`
  (target-cpu=native + PGO), `reckless-d445b8b` (native, AVX-512+VNNI),
  `integral-v6`, `/usr/games/stockfish` (SF16 — **distro SSE2-only build**, per
  its `compiler` command: "64bit SSE2"), `engines/stockfish-18-src/src/stockfish`
  (SF18, local **x86-64-avx2** build). These ISA differences are called out
  wherever they matter.

---

## 2. ARCHON hot-path breakdown (sota, the shipping-candidate path)

perf, depth-12 bench with the D40 net, single thread, leaf-attributed with
`memmove`/`memset` folded into their callers. One-time net load (1.4%) excluded.

| share | subsystem | what it is |
|------:|-----------|------------|
| 19.3% | NNUE head forward | `SotaHead::forward` self 12.6% + `activate` 3.6% + `l1_dot` 2.6% + `find_nnz` 0.6% (`src/eval/nnue/sota_head.rs:194-208`, `sota_simd.rs:502-555, 477-499`) |
| 15.8% | NNUE threat fold | `avx512::i8_deltas` — folding resolved FullThreats deltas into the 768-lane i8-weight accumulator (`sota_simd.rs:450-474`, driven by `sota_threats.rs:700-730`) |
| 15.0% | NNUE threat push (observer) | `EvalContext::push_sota_dirty` self 11.0% + `record_square_touch` 2.9% + `record_outgoing`/`ripple` ~1.1% — event generation + `resolve_entry` at **every** `do_move` (`eval/mod.rs:204-216`, `sota_threats.rs:196-327, 655-692, 760-792`, called from `search/mod.rs:4139-4142`) |
| 10.7% | NNUE glue + accumulator copies | dominated by `__memmove_avx512_unaligned_erms` 9.4%, callers resolved to the parent→child 1.5 KB accumulator copies inside the lazy folds (`sota_threats.rs:711-712`, `sota_accumulator.rs:97-101`) and eval-boundary buffer moves |
| 5.8% | NNUE HalfKA fold | `avx512::i16_deltas` (`sota_simd.rs:424-447`, driven by `sota_accumulator.rs:95-102`) |
| 2.9% | NNUE Finny refresh | `HalfkaContext::finny_refresh` (`sota_accumulator.rs:105-160`) |
| 0.9% | NNUE misc | threat-fold glue, from-scratch refresh |
| **70.3%** | **NNUE total** | |
| 9.4% | move ordering (picker) | `MovePicker::next_move` 6.9% + `::new` 2.4% (`src/movepick.rs:346`) |
| 7.6% | make/unmake | `Board::make_move_with` 3.9% + `set_check_info` 3.3% + rest |
| 4.4% | search control | `Worker::search`/`qsearch` bodies, stack bookkeeping |
| 2.3% | movegen/attacks | `generate`, `attackers_to`, magic lookups |
| 1.9% | move ordering (history) | `Histories::correction_value` 1.5% + updates |
| 0.8% | SEE | `Board::see_ge` |
| 0.3% | TT | `probe`/store (prefetch working as designed — `search/mod.rs:4137-4138`) |
| **26.6%** | **search machinery total** | |

Narrative: the search scaffold itself is cheap and healthy — TT probe is
invisible (0.3%), movegen is 2.3%, SEE under 1%. The engine is, cycle-wise, an
NNUE evaluator with a search attached. Three of the four biggest NNUE blocks are
implementation overhead rather than irreducible math (§4, §6).

For contrast, the **classic-2048 path** (`probe`-equivalent profiling build,
depth-12 bench, 1,846 samples): NNUE ≈ 42% (`simd::avx2::add_sub_sub_w` 15.1%,
`add_sub_w` 12.1%, `simd::avx2::forward` 9.8%, `NnueContext::compute` 3.3%,
memmove 1.4%), search machinery ≈ 45% (**`MovePicker::next_move` 15.2% is its
single hottest function**, `MovePicker::new` 6.0%, make/unmake 11.2%, search
7.0%). Two classic-lane observations: (a) the v1 kernels are **AVX2-only** —
`src/eval/nnue/simd.rs:6-14` documents scalar+AVX2 backends and nothing wider,
so on Zen 4 the 2048-lane updates run at half the possible vector width; (b) the
classic kernels already implement **fused copy+apply**
(`add_sub_w(dst, src, add_row, sub_row)`, `simd.rs:466-473`) — exactly what the
sota kernels lack (§4.3).

---

## 3. Cache & branch analysis (cachegrind model, depth-8, AVX2 lane)

Totals (whole process; startup ≈ 1.29G of 3.95G Ir inflates the loader's share
but search dominates D-refs):

| metric | value | reading |
|---|---|---|
| I1 miss rate | **1.12%** (44.25M) | HIGH for a chess engine — code-footprint problem, see below |
| D1 miss rate | 5.2% (70.2M; 6.0% rd / 3.9% wr) | moderate; weight-row streaming |
| LL (32 MB) miss rate | 0.1% (5.05M) | **not DRAM-bound** — the 81 MB net's hot rows live in L2/L3 |
| cond. branch mispredict | 2.5% modeled (perf measured 2.17% native) | fine |
| indirect branch mispredict | **16.2%** | two specific culprits, below |

Where the misses actually are:

- **`SotaHead::forward` causes 27.9% of ALL I1 misses (12.35M — 246 per eval).**
  The function compiles to **5,726 instructions of straight-line code** (native
  build, `objdump`): the fully-unrolled scalar f32 tail (§4.2) is ~25–30 KB of
  code that blows through the 32 KB I1 on every evaluation and evicts the rest
  of the engine. This single function explains most of the abnormal 1.12% I1
  rate, and is also 12% of D1 write misses (its stack buffers). The known
  PGO +10.5% gain (R3) is consistent with this: PGO's main lever is code
  layout, and the biggest layout problem is here.
- **NNUE weight-row streaming is ~52% of D1 read misses**: `i8_deltas` row loads
  11.9M (23.4%), `i16_deltas` 8.8M (17.2%), `l1_dot` weight groups 6.0M (11.8%).
  The i8 threat rows also produce 8% of LL(=L3-model) read misses — ~15 rows ×
  768 B scattered across a 46.6 MB table per fold has no temporal locality by
  construction. This is latency exposure a software prefetch can hide (§6 #3).
- **Indirect-branch pathology in the threat observer**: the event loop in
  `push_sota_dirty` retires 320k indirect jumps with a **67% mispredict rate**
  (214k mispredicts — 39% of the whole program's) — the per-event
  piece-type `match` dispatch in `record_square_touch`/`threat_index`
  (`sota_threats.rs:196-242`); `AttackTables::attacks`' jump table adds 124k at
  31% (`src/attacks.rs`). Absolute cost is modest (~0.3–0.5% of cycles) but it
  is free money for a branchless recode of the dispatch.
- The **kernel-dispatch indirection** (`sota_simd.rs:26-43`): every kernel call
  site is `call *KERNELS+off(%rip)` **plus a LazyLock state load+branch**
  (disassembly shows the `mov KERNELS+0x28,%eax; cmp; jne` pattern before each) —
  ~7–10 per eval. SF/Reckless/Integral all use compile-time dispatch precisely
  to avoid this (reference survey, §5.3).

Compute-vs-memory verdict: the sota path is **latency/front-end bound, not
bandwidth bound** — IPC 1.39 (perf, native) with LL misses near zero. The
classic path (IPC 1.15) is more memory-flavored (2048-lane × i16 row streams);
Integral shows the same signature more extremely (IPC 0.61, 1536-wide + dense
2.25 MB magic tables). SF16 posts IPC 1.71 — the most compute-dense of the
group even in an SSE2 build.

---

## 4. NNUE inference cost — per-eval anatomy and the waste map

### 4.1 The shape of an eval

Callgrind exact counts, depth-8 bench (147,691 nodes):

| event | count | per node | per eval |
|---|---:|---:|---:|
| `SotaContext::evaluate` | 50,192 | **0.34** | 1 |
| `i8_deltas` (threat fold) | 134,199 | 0.91 | 2.67 |
| `i16_deltas` (HalfKA fold + Finny diff) | 125,869 | 0.85 | 2.51 |
| `finny_refresh` (king-move rebuild) | 18,165 | 0.123 | 0.36 |
| `refresh_threat` (d/e mirror cross) | 3,015 | 0.020 | 0.06 |
| `activate` / `find_nnz` | 50,192 each | 0.34 | exactly 1 — memoization works |
| `MovePicker::next_move` | 236,634 | 1.60 | — |

The laziness stack works as designed: only 34% of nodes evaluate (TT-eval reuse
at `search/mod.rs:2112-2116`, in-check skip, cutoffs before eval), each eval
folds only ~1.34 deferred plies per perspective, and `activate`+`find_nnz` run
exactly once per eval. **The eval-call rate is not the problem; the cost per
call is.**

Per-eval instruction budget (callgrind, AVX2 lane; AVX-512 halves the
lane-loop terms but not the scalar head):

| stage | Ir/eval (AVX2) | notes |
|---|---:|---|
| threat fold `i8_deltas` | ~8,400 | 2.67 calls × ~3,140 Ir → implies **~15 avg features/fold** (vs HalfKA's ≤4) |
| observer (amortized) | ~6,600 | ~2,230 Ir at *every* node (§4.4) |
| head `forward` | ~4,800 | ~55% of it the scalar L2 (§4.2) |
| HalfKA fold `i16_deltas` | ~3,600 | 2.51 calls × ~1,450 Ir — consistent with ≤4 fused rows |
| `activate` | ~2,500 | i32-domain merge (§4.5) |
| copies/glue | ~2,000 | the 1.5 KB parent→child memmoves (§4.3) |
| Finny + misc | ~1,700 | 0.36 refreshes/eval — expected for HalfKA (every own-king move) |
| `l1_dot` + `find_nnz` | ~1,400 | already optimal (§4.6) |
| **total** | **~31,000** | ≈ 10.6k Ir/node of the 18.0k/node total (AVX2) |

### 4.2 The scalar f32 head tail — the single worst offender

`SotaHead::propagate_l2` (`src/eval/nnue/sota_head.rs:148-164`) is a 16×32 f32
layer — 512 FMAs, nominally ~64 AVX-512 ops. It compiles to **fully unrolled
scalar code**: the native binary's `forward()` contains **432 `vfmadd231ss` +
16 `vfmadd213ss`** (and only 40 128-bit packed FMAs, mostly from `propagate_l3`,
`sota_head.rs:168-188`). perf line-attribution confirms it at runtime: 50.3% of
`forward()` samples on `let w = self.l2_weights[wbase + i * L3 + j]` and 31.9%
on `fmaf32` — i.e. **82% of forward's 12.6% self-time ≈ 10.4% of total runtime
in this one tiny layer**, plus the §3 icache damage.

Why LLVM gave up: the weights are flat `Box<[f32]>` indexed
`wbase + i * L3 + j` with a runtime `bucket` (`sota_head.rs:63-75, 153-157`) —
the bounds check on every load (visible as `cmp/jae` panic edges in the
disassembly) is not provably redundant, and per-element `mul_add` becomes a
scalar `fmaf32` call pattern the vectorizer won't touch across a checked load.

The fix is mechanical: slice per-bucket weight blocks once
(`<&[f32; L2*L3]>::try_into`) or store per-bucket arrays at load time, then
vectorize across `j` (each `out[j]` keeps its own sequential accumulation over
`i`, so a j-lane vectorization is **bit-identical** to the pinned scalar order —
the "load-bearing" comment at `sota_head.rs:146-147` survives intact).
Reckless and Integral both run this exact tail shape as explicit 16-lane f32
SIMD (reference notes `reckless_reference/13-nnue-inference-simd.md` §2,
`integral_reference/05` §6).

### 4.3 The 1.5 KB copies the kernels force

Both lazy folds do copy-then-apply-in-place:
`sota_accumulator.rs:97-101` (`let mut acc = before[child-1].acc[p]; apply;
after[0].acc[p] = acc`) and `sota_threats.rs:711-727` (same pattern). That is a
1.5 KB read + 1.5 KB write of pure `memmove` per perspective per folded ply
**before** the kernel reads it all again — measured at **9.4% of total runtime**
(`__memmove_avx512_unaligned_erms`, callers: `SotaContext::evaluate` fold sites
and `eval::evaluate`'s inlined HalfKA fold). The classic path already has the
right shape: `simd.rs:466-473` `add_sub_w(dst, src, add_row, sub_row)` reads
the parent and writes the child in one fused pass. Porting that signature to
`sota_simd.rs` (`apply_i16_deltas_into(dst, src, adds, subs)` /
`apply_i8_deltas_into`) removes the memmove entirely and halves fold memory
traffic. SF's `vec_add_16`-chain updates are exactly this single-pass form
(survey §1.1).

### 4.4 The always-paid threat observer

`push_sota_dirty` runs at **every** `do_move` (`search/mod.rs:4139-4142`):
`build_deltas` replays the move against the post-move board
(`sota_threats.rs:798-895`), and `resolve_entry` maps every raw event through
`threat_index` per perspective with an O(n²) cancel scan
(`sota_threats.rs:760-792`). Measured: **~2,230 Ir/node, 15.0% of runtime**.
With 0.34 evals/node and 1.34 folded plies/eval, only ~45% of resolved delta
lists are ever folded — **~55% of observer work is discarded**. The structure is
forced (the post-move board is destroyed by siblings — module docs
`sota_threats.rs:36-46`; SF's DirtyThreats and Reckless's observer are also
built at make-time, survey §1.1/§2.1), so the lever is the constant factor, not
the schedule: branchless piece-type dispatch (§3's 67%-mispredicting jump
table), cheaper `threat_index` (precomputed per-(piece,from) row bases), and a
sort-free cancel (events for the same square pair are adjacent by construction —
a small fixed-size hash or sorted merge beats the quadratic scan at ~15+ events).

### 4.5 `activate` pays an i32 tax for the dual-quant merge

Because FullThreats is exported at `THREAT_QUANT=64` vs HalfKA's `QA=255`
(`sota_head.rs:34-37`), the merge must rescale with signed rounding per lane, in
i32 (`sota_simd.rs:502-555`): widen both i16 accumulators, `mullo`, `abs`, shift,
masked-negate, clamp — ~13 ops per 16 lanes where SF/Reckless/Integral's
same-scale pairwise SCReLU is a couple of i16 `mulhi`/`min`/`max` ops (survey:
all three converged on the i16 `mulhi` trick). Cost: 3.6% of runtime + it forces
the head input through an extra 768-entry u8 buffer. Unifying the two
transformers' quantization scale (export threat weights at the FT scale — a
trainer/netpack change, not bit-exactly reachable by load-time rescale because
the rounding applies to summed activations) would let `activate` collapse to the
standard i16 form. One for the next net-format rev, not a hot patch.

### 4.6 What is already right

- `l1_dot` AVX-512 is textbook: `vpbroadcastd` + `vpdpbusd` per nonzero group,
  6 instructions — LLVM even merged the `from_le_bytes` into a single broadcast
  load (`sota_simd.rs:477-499`; disassembly verified). VNNI is actually used —
  which Integral, notably, fails to do in its "VNNI" tier (survey §3.1).
- `find_nnz` movemask+LUT matches Reckless/SF practice (`sota_simd.rs:352-377`).
- Laziness discipline (deferred folds, memoized `computed` flags, eval-time-only
  activation) matches SF/Reckless exactly; TT-eval reuse keeps evals at 0.34/node.
- Finny cache behavior is standard: 0.36 refreshes/eval is the unavoidable
  HalfKA king-move rate, and refreshes go through the 64-key diff cache rather
  than from scratch (`sota_accumulator.rs:105-160`).
- 768-wide is the right ballpark: Reckless ships 768, SF 1024(+threats),
  Integral 1536-no-threats. The *width* is not the problem; the constant
  factors are.

---

## 5. Cross-engine NPS / efficiency comparison

### 5.1 The numbers

Own-`bench` suites under `perf stat`, single thread, pinned core 14 (except
Reckless, self-pinned), contended box (§1.1 — treat absolutes as lower bounds,
ratios as solid):

| engine (build ISA) | eval arch | nodes | knps | instr/node | cycles/node | IPC | br-miss |
|---|---|---:|---:|---:|---:|---:|---:|
| ARCHON sota-pgo (native AVX-512) | 768 HalfKA+threats, hybrid head | 3,515,701 (d12) | **412** | 16,924¹ | 12,137 | 1.39 | 2.17% |
| ARCHON l2048-pgov3 (native, AVX2 kernels) | 2048 classic, 1-layer head | 1,735,391 (d12) | **1,085** | 5,178¹ | 4,499 | 1.15 | 3.05% |
| ARCHON PSQT (same binary, no net) | PSQT | 2,789,693 (d12) | **1,894** | 3,144 | 2,568 | 1.22 | 3.45% |
| Reckless d445b8b (native AVX-512+VNNI) | 768 PST+threats, same head shape | 2,929,550 | **616** | ≤13,186² | ≤15,153² | 0.87² | 5.79% |
| Integral v6 | 1536 HalfKA, no threats | 1,851,233 | **866** | 4,212 | 6,929 | 0.61 | 4.33% |
| SF16 (**distro SSE2-only**) | 2048 HalfKAv2_hm, int tail | 2,593,605 (d13) | **982** | 8,879 | 5,194 | **1.71** | 3.95% |
| SF18 (local **AVX2**) | 1024 HalfKAv2_hm + FullThreats | 2,050,811 | **697**³ | **7,378** | 8,073 | 0.91³ | 6.82%³ |

¹ includes ~1.3G (sota) / ~0.3G (classic) startup instructions — net-of-startup ≈ 16.5k / 5.0k.
² Reckless's counters include helper-thread spin-waits (instruction count varies ±6% run-to-run at identical node counts); its true per-node cost is somewhat lower.
³ SF18 shared the box with its own 10 relabel siblings; its cycles/IPC are the most contention-poisoned row.

Common position (startpos, `go nodes 500000`): ARCHON sota **431k**, classic-2048
**1,055k**, Reckless **536k**, Integral **984k**, SF16 **488k** nps. (Bench-suite
NPS runs higher than startpos NPS for engines whose suites include endgames.)

Perft (movegen isolate, pinned): startpos d6 = 119,060,324 — ARCHON **~160 Mnps**
(0.75–0.78 s incl. ~0.05 s init) vs SF16 **~135 Mnps** (0.86–0.90 s); kiwipete d5
= 193,690,690 — ARCHON **~200 Mnps** vs SF16 **~188 Mnps**. (Clean-box ARCHON
baseline is 280–297 Mnps, `docs/perf-baseline.md:62-63`.) **Movegen is faster
than Stockfish's and is not a bottleneck** — consistent with its 2.3% profile share.

### 5.2 Why ARCHON's NPS is "higher than Reckless" — and why that's now false on the sota path

The remembered comparison (ARCHON 2.36 Mnps ≈ 2.5× Reckless) was the **PSQT/net-0
era**. The decomposition above shows it was never a same-work comparison:

- **PSQT (1.89 Mnps, 3.1k instr/node)** and **classic-2048 (1.09 Mnps, 5.2k
  instr/node)** are doing *far less work per node* than Reckless (≤13.2k):
  no threat features, a 1-layer head, no buckets. Higher NPS bought by a cheaper
  eval function — exactly the speed/quality trade you'd expect, and exactly why
  per-node quality lags (−87±23/node vs Reckless, design/14 §3.11): Reckless
  spends ~2.5× more instructions per node *computing a better eval* and pairs it
  with maturely-tuned search constants.
- On the **sota path the trade inverts**: ARCHON runs the *same architecture
  shape* as Reckless (768 PST+threats, pairwise FT, sparse int8 16, f32 16→32→1,
  8 buckets — the head is our bit-exact reimplementation of that design,
  `sota_head.rs:1-20`) at **412–431 knps vs Reckless's 536–616 knps**, i.e.
  **~1.3–1.5× slower with the same net shape**, and at 16.9k vs ≤13.2k
  instr/node with *worse* per-instruction density once Reckless's spin-waste is
  discounted. The gap is precisely the waste catalogued in §4: scalar head tail
  (~10%), copies (~9%), observer constants (~5–7% recoverable), icache thrash.
- The **SF comparison bounds what maturity buys**: SF18 evaluates
  1024-wide + FullThreats + a deeper all-integer tail in **7.4k instr/node** —
  2.3× fewer than ARCHON's 768-wide equivalent — and SF16 posts 982 knps *in an
  SSE2-only build* with a 2048-wide net. There is nothing architectural about
  ARCHON's cost; it is constant-factor engineering headroom.

### 5.3 Divergences from reference best practice (cycle-costing)

| practice | SF | Reckless | Integral | ARCHON |
|---|---|---|---|---|
| SIMD dispatch | per-ISA binaries | compile-time `cfg` modules | 5-tier build matrix | **runtime `LazyLock` fn-pointers, per-call state check** (`sota_simd.rs:26-43`) |
| fold kernel form | fused src→dst single pass | fused, lazy re-derive | variadic fused single pass | **in-place, forcing 1.5 KB pre-copies** (§4.3) |
| f32/int tail | all-integer, SIMD | explicit 16-lane f32 SIMD | explicit f32 SIMD | **unrolled scalar** (§4.2) |
| FT activation | i16 `mulhi` pairwise | i16 `mulhi` pairwise | i16 `mulhi` pairwise | **i32 widen+rescale merge** (§4.5) |
| net bytes at load | mmap-friendly, preprocessed layouts (Integral pre-transposes) | baked | pre-permuted per tier | **per-element `Vec` push parse: 0.21 s/spawn** (`sota_loader.rs:512-529`) |
| TT | 10 B/32 B + huge pages | 8 B packed-tag SWAR + huge pages | 10 B/32 B | 10 B/32 B, no huge pages (fine at 0.3% today) |

---

## 6. Ranked optimization opportunities

"NPS" = pure speed, same nodes/values — verifiable by bench-node identity plus
wall-time; Elo estimates use the D39 house rule (~+9–10 internal Elo per +10%
speed at 10+0.1, deflate ~2× externally) and assume the sota path is the target
(it is the one that is NPS-gated at its promotion gate, exp-0093).

| # | what / where | expected gain | effort | risk |
|---|---|---|---|---|
| **1** | **Vectorize the f32 head tail.** Per-bucket weight arrays (kill the bounds checks) + j-lane SIMD for `propagate_l2` (`sota_head.rs:148-164`), same for `propagate_l3` (`:168-188`) and the L1 dequant loop (`:136-142`). Per-lane accumulation order is preserved ⇒ **bit-identical eval**. Kills ~10% direct compute **and** the 28%-of-I1-misses code bloat (5.7k-instr function → ~1k). | **+10–15% NPS** (sota) | LOW (half-day + parity test) | LOW |
| **2** | **Fused copy+apply fold kernels.** Add `apply_i16_deltas_into(dst, src, ...)` / `apply_i8_deltas_into` to `sota_simd.rs` (read parent lanes, add/sub rows, write child lanes — the classic path's `add_sub_w` shape, `simd.rs:466-473`); use in `sota_accumulator.rs:95-102` and `sota_threats.rs:709-729`. Eliminates the 9.4% memmove and halves fold traffic. Bit-identical. | **+7–9% NPS** | LOW-MED (1 day) | LOW |
| **3** | **Prefetch + pair the fold rows.** At fold entry, issue `_mm_prefetch` for all add/sub weight rows (15×768 B i8 rows scattered over 46.6 MB — 23% of D1 read misses, 8% of L3-model misses), and process two features per lane-pass for ILP (Reckless's `double_dpbusd` idea). `sota_simd.rs:424-499`. | **+2–5% NPS** | LOW | LOW |
| **4** | **Static kernel dispatch.** Replace the per-call `LazyLock` fn-pointer indirection (`sota_simd.rs:26-88`) with a one-time detected enum driving monomorphized eval entry points, or `cfg`-module dispatch like Reckless (we already rebuild per target for strength testing, D16). Removes the state-check+indirect-call at ~10 sites/eval and unlocks inlining into the fold loops. | **+1–3% NPS** | MED | LOW |
| **5** | **Observer constant factor.** Branchless piece-type dispatch in `record_square_touch` (67%-mispredicting jump table), precomputed `threat_index` row bases per (piece,from), replace `resolve_entry`'s O(n²) cancel scan (`sota_threats.rs:760-792`). 15% block; ~15–25% of it shavable. | **+2–4% NPS** | MED | LOW-MED (parity gate exists: `threats_replay_*`) |
| 6 | **AVX-512 backend for the classic v1 kernels** (`simd.rs` is AVX2-only; classic spends ~27% in `add_sub*_w`). Only worth it if the classic incumbent keeps shipping. | +8–12% NPS (classic only) | LOW-MED | LOW |
| 7 | **Bulk net parse.** `read_i8s`/`read_i16s` (`sota_loader.rs:512-529`) do ~10 instr/byte per-element pushes → 0.21 s per process spawn (measured d1-bench delta 0.29 s vs 0.16 s). Chunked LE conversion → ~20 ms. No Elo; big match-harness/`isready` QoL, and it de-noises every future short-workload profile. | startup −0.2 s | TRIVIAL | none |
| 8 | **Unify threat/FT quantization scale in the next net format rev** → `activate` collapses to the standard i16 `mulhi` pairwise form (§4.5). Requires trainer/netpack change + retrained or re-exported net; not bit-exact vs current nets. | +2–3% NPS | MED-HIGH | MED |
| 9 | **MovePicker constant factor** (`movepick.rs:346` — 26.6% of its self-time is prologue/spill, 17% stage dispatch; 9.4% of sota runtime, 21% of classic). Split per-stage fast paths, hoist history-plane pointers. Borderline algorithmic; coordinate with the search study. | +1–3% NPS | MED | MED |
| 10 | **Huge pages for TT + net weights** (SF reports "measurable Elo" for the TT; our TT is 0.3% today at 16 MB bench hash — relevant at real hash sizes and for the 81 MB net's TLB reach). | +0–2% | LOW | LOW |

**Out of scope (algorithmic, deferred to the search study):** eval-call rate
(0.34/node is already healthy), lazy-eval thresholds (SF removed theirs; not the
gap), any search-shape change. Movegen needs nothing (faster than SF16's,
§5.1).

**Strategic read:** items 1–4 are independent, low-risk, bench-node-preserving,
and sum to an estimated **+20–30% NPS** on the sota path — roughly the entire
remaining speed gap to Reckless's same-shape implementation. Given D40 already
holds +30/node at fixed nodes (exp-0067) and sits at equal-time parity purely
for speed reasons (exp-0093, recovered to parity by PGO alone), this package is
the highest-confidence route to converting the proven eval edge into equal-time
Elo — worth executing before any further net-quality spend on that arch.

---

## Appendix: raw data locations

Scratchpad (`/tmp/claude-0/-root-projects-learnrust-chessengine/9162696c-.../scratchpad/prof/`):
`sota-d12.perf.data` (+ `.folded.pkl`), `l2048-d12.perf.data`,
`cg-sota-d8.out` (callgrind), `ccg-sota-d8.out` (cachegrind),
`archon-sota-prof` / `archon-sota-v3` / `archon-l2048-prof` binaries.
perf binary: `/usr/lib/linux-tools-6.8.0-136/perf` (invoke directly; the
`/usr/bin/perf` wrapper refuses the WSL kernel string).
