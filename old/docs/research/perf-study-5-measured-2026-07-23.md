# ARCHON performance study #5 — MEASURED ground truth for the eval layer

Date: 2026-07-22/23. The measurement companion to study #4
(`perf-study-4-eval-2026-07-22.md`, code-read cost model, no measurements).
Everything here is a number from a real tool run on the production workload;
raw artifacts live in `perf-artifacts/` (this worktree).

**Workload** (tripwire-verified): `bench 16 1 <depth> <fens> depth
data/nnue/nets/d47c-stage2-sb240.cenn` — the D47c stage-2 net, `.cpol`
sidecar auto-loaded. FEN sets: `wl50` = every-16th line of
`tools/nnue/calib/scale-check-800.fens` (50 positions, realistic phase mix
8–27 pieces), `wl12` = every-66th (13 positions). Net-load tripwire passed:
485–611 kNPS single-thread (PSQT fallback would read ~3M).

**Source state measured:** ee12507 + the main tree's full uncommitted
d47c-era diff — including `src/search/mod.rs` + `src/uci.rs`, which the
study-#4 snapshot MISSED (see §7 surprise S1). Bit-verified equal to the
optimizing agent's baseline commit `32e0a9c`. Binaries: native+sota release
(perf/heaptrack), x86-64-v3+sota (callgrind; valgrind 3.25.1 cannot decode
AVX-512 — SIGILL — so callgrind sees the AVX2 kernel path via masked CPUID;
used ONLY for deterministic call counts and Ir shares, never cycle shares).

**Box state during Phase A:** shared with the study-#4 implementation agent
(intermittent cargo builds) + idle tensorboard shims; loadavg 2–4. Per
protocol, Phase A reports only contention-tolerant facts (profile shape,
deterministic counts, static analysis). Phase B (quiet-box precision: NPS
ladders, IPC/cache/dTLB rates, bandwidth) is in §8.

---

## 1. Q1 — hot-path ground truth (top symbols, two independent instruments)

perf = cycles, native binary, wl50/d16 (50,488 samples, consult-live v2
binary). callgrind = exact instruction counts (Ir), v3 binary, wl50/d12
(57.4 G Ir, 1,874,757 nodes) — deterministic, contention-immune.

| symbol | perf cycles % | callgrind Ir % | block |
|---|---:|---:|---|
| `sota_simd::avx512::l1_dot` | **34.4** | 30.9 | head |
| `SotaHead::forward_full` (l1-dequant + L2 + L3 + heads, inlined) | **7.6** | 13.0 | head |
| `sota_simd::avx512::i8_deltas_into` (threat fold) | **7.5** | 11.8 | threat |
| `sota_simd::avx512::activate` (split-quant FT merge) | **5.9** | 6.4 | head |
| `MovePicker::next_move` | 4.9 | 3.3 | search |
| `sota_simd::avx512::i16_deltas_into` (HalfKA fold) | 4.7 | 7.1 | halfka |
| `sota_simd::avx512::i16_deltas` (finny + halfka in-place) | 3.9 | 3.8 | halfka |
| `Worker::search::<NonPv>` | 3.1 | 1.8+1.75 | search |
| `ThreatContext::at` (fold driver + resolve) | 2.5 | 1.9 | threat |
| `HalfkaContext::finny_refresh` | 2.0 | 1.5 | halfka |
| `MovePicker::new` | 1.9 | 0.9 | search |
| `HalfkaContext::current` | 1.8 | — | halfka |
| `PolicyHead::best_move` (consult, incl. its generate_legal) | 1.5 | — | policy |
| `sota_threats::record_square_touch` | 1.5 | 0.9 | threat |
| `Board::make_move_with` | 1.2 | 0.8 | search |
| `__memcpy_avx_unaligned_erms` | ~1.0 (libc) | **2.45** | copies |
| `TranspositionTable::probe` | 0.9 | — | search |
| `Board::set_check_info` | 0.8 | — | search |
| `Board::see_ge` | 0.7 | — | search |
| `avx2::find_nnz` (AVX2 kernel on the AVX-512 box) | 0.5 | 0.8 | head |

Block totals (perf cycles, native):

| block | measured | study-#4 EST | verdict |
|---|---:|---:|---|
| **head** (l1_dot + forward_full + activate + find_nnz) | **≈ 48.5%** | 25–35% | **REFUTED low — head is half the engine** |
| threat channel (fold + gen/resolve + push) | ≈ 12.4% | 27–36% | **REFUTED high — ~2.5× overestimated** |
| HalfKA + finny | ≈ 12.4% | 10–14% | confirmed |
| policy machinery (consult + stash) | ≈ 2% | 3–7% | revised down (§5) |
| eval layer total | ≈ 76% | 62–78% | confirmed |
| search machinery | ≈ 24% | 22–38% | confirmed |

Study #4's central claim ("threat delta + L1 matmul dominate") is **half
right**: l1_dot dominates everything, but the threat channel is a *third*
the size the model said, and the head is *double*. The width bet (L2
16→128) moved the cost into exactly the block that was never hand-tuned.

### Deterministic schedule counts (callgrind v2, consult-live, wl50/d12)

1,874,757 nodes, 791,679 full evals ⇒ **0.422 evals/node** (study #4 anchor
0.34 — same regime, slightly higher on this set).

| event | calls | per eval | note |
|---|---:|---:|---|
| threat folds (`i8_deltas_into`) | 2,134,515 | 2.70 | matches #4's 2.67 |
| — of which degenerate pure-memcpy | 606,481 | 0.77 | **28% of threat folds fold zero deltas** (null-move / no-event plies) |
| HalfKA folds (`i16_deltas_into`) | 1,359,479 | 1.72 | |
| finny refreshes | 527,096 | 0.67 | = 0.33/perspective (#4's "0.36" was per-persp) |
| finny refresh memcpys | 2,108,384 | 4.00/refresh | the "4 round-trips" claim: **exact** |
| threat pushes (`apply_at`) | 1,879,597 | 1.00/node | eager remnant confirmed trivial |
| policy consults (`best_move`) | 18,267 | 0.023 | 1.0% of nodes |
| malloc / free | 803,726 / 812,790 | 1.015 / 1.027 | the Box stash, §3 |
| memcpy total | 7,447,712 | 9.4 | callers: forward_full 3.0, finny 2.7, evaluate (stash) 1.0, MovePicker::new 1.0, i8 degenerate 0.77, current 0.67 |

---

## 2. Q2 — codegen quality: the two smoking guns (llvm-mca -mcpu=znver4)

### 2a. `l1_dot` — study #4's loop-swap claim CONFIRMED and understated

Disassembly (`perf-artifacts/l1_dot-native.asm`): LLVM fully unrolled the
8-block outer loop into **8 separate inner loops, each re-walking the nnz
list**, and each block's `vpdpbusd` chain accumulates into a **single zmm0 —
a serial dependency chain**. Per (group, block): 9 instructions of which 1
is useful (`movzwl; cmp+ja bounds-panic; mov; shl; vpbroadcastd; add;
vpdpbusd; cmp+jne`). Bonus defect: **a bounds-check panic edge per group per
block** (8× redundant; from indexing `ft[group*4]` in the from_le_bytes
idiom).

llvm-mca (Zen 4 model, 300 iterations, `perf-artifacts/mca-l1dot-*.txt`):

| shape | cycles per group×all-8-blocks | bottleneck |
|---|---:|---|
| current (block-outer, 1 accumulator) | 8 × 3.05 = **24.4** | dep-chain + frontend (6 overhead instr per dpbusd) |
| swapped (group-outer, 8 zmm accumulators) | **8.06** | load ports (RThroughput 8.0 = the 1×512-bit-load/cycle ceiling) |

**Kernel speedup ≈ 3.0×.** At l1_dot's measured 34.4% of cycles this bounds
the NPS gain at ≈ **+29%**; realistically +15–25% after memory effects
(the swapped loop's 512 B/group are contiguous vs today's 8 strided passes —
prefetch-friendlier, so the mca number may even be conservative on misses).
i32 adds commute exactly ⇒ **BIT-IDENTICAL**. This is now the #1
engineering item in the codebase by a wide margin.

**Why l1_dot exploded — the FT is no longer sparse.** From callgrind's
deterministic Ir/call (22,417) and the exact 14-instruction AVX2 loop shape
(8 unrolled bodies): **average nnz ≈ 200 of 320 groups = 62% density**.
Study #4 (and the source comment "~85–95% are zero") carried the 768-era
anchor of ~10–15% density / ~40 groups; the d47c net's activations are
~5× denser. Consequences: (a) l1_dot streams ~200 × 512 B ≈ **100 KB of L1
weights per eval** (not 20 KB) — the entire 160 KB bucket table effectively
passes through cache every eval; (b) per-eval streamed bytes total ≈
**180 KB, ~2× the #4 model** (≈ 42 GB/s nominal at current NPS — Phase B
checks how much reaches DRAM); (c) the sparse-index machinery only saves
~1/3 of the dense cost at this density; (d) **FT sparsity is a trainer-side
lever nobody priced**: activation-sparsity regularization in the next
retrain would cut l1_dot roughly linearly with density. Verify density on
future nets before reusing any sparse-path cost intuition.

### 2b. f32 L2 tail — round-2 F4's "xmm-stuck" finding PERSISTS at d47c dims

`propagate_l2` (inlined into forward_full) compiled to: per i (128
iterations): `vbroadcastss …,%xmm16` + **16 × `vmovsd` (TWO floats each) +
16 × 4-wide `vfmadd231ps` with only 2 active lanes** — the 32 j-lanes are
held as 16 two-float xmm accumulators. Effective SIMD width: **2 of 16
lanes**. perf-annotate attribution: this loop = **84.5% of forward_full's
self time** ≈ 6.4% of all cycles (dequant loop is fine — 8 zmm FMAs, 1.4%).

llvm-mca: current shape **9.35 cycles/i-iteration**; a bit-exact 2×zmm
kernel (broadcast + 2 FMA, same j-lane order, sequential-i accumulation
preserved) = **4.06 c/iter** (FMA-latency-chained) ⇒ ~2.3× on this loop,
≈ +3–4% NPS, BIT-IDENTICAL. (A 4-way partial-sum variant would be
throughput-bound ~2× faster again but changes the reduction shape ⇒ not
bit-exact — functional-patch territory, don't bundle it.)

### 2c. the rest of the codegen audit

- `find_nnz`: **confirmed** running the AVX2 kernel under the AVX-512
  dispatch (symbol `avx2::find_nnz` live in the native profile, 0.5%);
  VBMI2 `vpcompressw` upgrade remains valid but small.
- Kernel dispatch: the LazyLock indirect call is **per fold-application,
  not per row** (`call *KERNELS+0x10(%rip)` once per `at()`/`current()`
  drive; ~8 dispatches/eval). Study #4 item 2c (static dispatch) **revised
  down** to ~0.2–0.5% — worth taking only as a rider on kernel work, the
  win being inlining not the indirect call itself.
- `activate` (5.9%): hand-written AVX-512, healthy; its entire existence is
  still the THREAT_QUANT≠FT_QUANT tax (#4 item 6, format-rev item,
  unchanged verdict).
- fold kernels: clean row-add loops, zero prefetch (unchanged since #2 —
  the resolve→prefetch→fold item stays live but its base is 7.5%, not 26%).

---

## 3. Q3 — allocation census (heaptrack, exact counts)

wl12/d14, consult-live binary, 931,621 nodes: **415,319 total allocation
calls, 99.72% of them temporary** (alloc→free churn).

- **414,189 (99.93% of temporaries) from `SotaContext::evaluate`** — the
  per-eval `Box::new(l2)` stash. Exactly **1.00 per full eval** (matches
  callgrind malloc 1.015/eval), ≈ 250k allocs/s at native NPS.
- Everything else combined: ~1,100 allocs/run (root movegen Vecs, UCI,
  loader one-shots). **There is no other allocation on the search path** —
  study #4 missed nothing, the crate's no-alloc discipline holds everywhere
  but this one site.
- Peak heap 275 MB = 137 MB file bytes + 138 MB parsed-net leak
  (loader double-footprint transient); leaked-by-design 138 MB = the
  `Box::leak`'d net.

**BUT the cycle cost is small:** malloc+free don't clear callgrind's 0.28%
threshold (~0.1–0.2% Ir incl. the 512 B copy; perf shows libc malloc at
0.06%). Study #4's "1–2% NPS" for the stash fix is **optimistic**; the real
fix value is ~0.3–0.7% + hygiene (glibc arena state touched 250k×/s is
also TLB/cache pollution the counters can't cleanly attribute). Still worth
doing — it's a 20-line change — just not a headline. The `want_l2`-hint
half of #4's fix shape is mooted anyway: the consult fires at only 2.3% of
evals, so stash-on-demand ≈ never-stash.

---

## 4. Q4 — memory-boundedness + THP (Phase A half; counters in §8)

- **THP: `enabled=[always]` but `defrag=madvise` on this box, and the
  engine gets ZERO hugepages** — measured live: `AnonHugePages: 0 kB`
  against 175 MB Pss_Anon during search. With defrag=madvise, an un-advised
  fault only gets a 2 MB page if one is lying around free; on this
  15 GB box there never is one. ⇒ ~44,800 4K dTLB entries to cover the
  weights vs Zen 4's 3,072-entry L2 dTLB. **Study #4 item 8 upgraded from
  "audit" to "live defect": `madvise(MADV_HUGEPAGE)` on the net + TT
  allocations** (defrag=madvise means advised VMAs DO get compaction).
  dTLB-miss rate quantified in Phase B.
- Streamed-bytes model sanity check (from §1 counts): threat rows
  2.70 × ~15 × 1280 B ≈ 52 KB + HalfKA 1.72 × ≤4 × 2560 B ≈ 15 KB +
  L1 weights ~40 × 512 B ≈ 20 KB + finny 0.67 × ~10 KB ≈ **~94 KB/eval** —
  #4's ~95–100 KB/eval EST confirmed. At 550 kNPS × 0.42 evals/node ≈
  22 GB/s nominal weight traffic; whether that meets DRAM or dies in L2/L3
  is a Phase B question (likwid-perfctr is broken on this box — missing
  likwid-lua — so Phase B uses perf stat + a bandwidth micro-ceiling).

---

## 5. Q5 — policy sidecar cost (C3), now measured on the REAL consult path

(Study #4's snapshot missed that the consult exists in `search/mod.rs`'s
uncommitted diff; the first half of this study measured a consult-less
binary — those runs are retained in perf-artifacts as `wl50-d16-*` v1.)

- **Consult**: fires at 18,267 of 791,679 evals = **2.3% of evals / 1.0% of
  nodes** (gate: tt-miss ∧ depth≥3). Cycle cost: `PolicyHead::best_move`
  (incl. its second `generate_legal` + 1858-row scalar dot) = **1.53%** of
  cycles.
- **Stash**: 1 Box alloc + 512 B copy per eval (§3), ~0.1–0.3%.
- **Tree effect (new finding):** wl50/d16 with sidecar = 14,105,504 nodes;
  without = 13,287,259. **The consult makes the tree 6.2% BIGGER at equal
  depth on this set** — consistent with the sb30 ON/OFF 0-Elo ablation, but
  sharper: today's head doesn't just buy nothing, its orderings are
  slightly worse than the classical stack it overrides, AND it pays for
  the privilege. C3 verdict (with §8 B1's paired NPS ladder): total cost =
  **+3.5% NPS + ~6% tree** — at the low edge of #4's cycle band, but the
  tree tax is new and worse than the cycle tax.
- Cheapest fix while accuracy climbs: keep the sidecar but gate the consult
  OFF (env) — zero cycle cost, zero tree distortion, stash skipped (the
  stash is `active_policy_head()`-gated already; no code change needed to
  ship nets). The #4 item-4 hygiene work only matters if/when the head
  starts winning its ablation.

---

## 6. C4 — the smallnet gate's p, measured (game-position-weighted)

Exact replica of `material_imbalance()` (SEE values, P=208) over three
game-representative sets:

| set | n | ≥400 | ≥600 | ≥800 | ≥1000 | ≥1200 |
|---|---:|---:|---:|---:|---:|---:|
| scale-check-800 | 800 | 31.0% | 17.4% | 12.4% | **7.8%** | 5.2% |
| canon_train sample | 10,867 | 35.0% | 21.2% | 14.8% | **10.4%** | 7.5% |
| indist800 | 800 | 34.4% | 20.5% | 13.6% | **9.2%** | 5.4% |

Study #4's "game-weighted 20–35% at rook-plus" band was **2–3× optimistic**:
at the 1000 cp gate, p(position) ≈ **0.08–0.10**; even at 800 cp it's
0.12–0.15. The search-tree-weighted p can exceed this (conversion subtrees
search deep at high imbalance) — the in-tree counter remains the decisive
measurement before training — but re-running #4's own yield formula at
p=0.10–0.15, c=0.12, E=0.70 gives **+7–10% NPS central (not +20%)**, and
the 400-cp gate (p≈0.33) only pays if the tiny net stays accurate at
moderate imbalance. T10 stays a real lever but drops below the l1_dot fix
in expected value; its go/no-go should ride the in-tree histogram, not the
#4 estimate.

---

## 7. Q6 — surprises the code-reading studies missed

- **S1 (process, important):** the main tree's uncommitted state included
  `src/search/mod.rs` + `src/uci.rs` changes (the D47 consult site) that
  post-date study #4's file snapshot. Any worktree study that syncs "the
  seven modified eval files" silently builds a consult-less engine.
  Symptom that caught it: cpol/nocpol node counts were bit-identical.
  Recommendation: sync-check by hashing ALL of src/ against the main tree
  (or better: the implementing agent's `32e0a9c`-style sync commit).
- **S2:** 28% of threat perspective-folds carry zero deltas and degenerate
  to a 2560 B memcpy (606,481/run) — the null-move alias + "no events for
  this perspective" cases. An alias/short-circuit (point the child at the
  parent accumulator instead of copying) is cheaper than any kernel work
  on the same bytes.
- **S3:** `MovePicker::new` memcpys once per construction (785 k/run,
  ~0.9% Ir incl. probcut variant) — copying the staged movelist state;
  search-lane item, noted for the search study.
- **S4:** net parse at spawn: `read_i8s` 2.2% + `read_i16s` 0.95% +
  `load_v2_bytes` 0.8% + memset share of a d12 run ⇒ ~0.4–0.5 s per process
  spawn at 130.5 MB (per-element `Vec::push` parsing), exactly as #4
  priced it. Every SPRT game pays it twice.
- **S5:** valgrind 3.25.1 cannot decode AVX-512 → all future callgrind work
  on this box must use the v3 build (and know that CPUID masking flips the
  runtime dispatch to AVX2 kernels).
- Nothing else: no hashing, no locks, no hidden parsing in the search path;
  TT probe 0.9%; movegen invisible (<0.3% outside picker). The engine's
  discipline outside the head is excellent.

---

## 8. Phase B — precision runs (pinned core 4, interleaved-paired design)

The box never went fully quiet (an SF16 gauntlet + the implementing agent's
test runs shared it throughout), so the ladder used an INTERLEAVED
round-robin: one run of every config per round, ratios computed per round —
load drift cancels in the pairing. Engine pinned `taskset -c 4`; per-round
raw table in `perf-artifacts/phaseb-ladder-raw.txt` (round 2's baseline
cell was hit by a 13-thread job and is excluded).

### B1. NPS ladder (wl50/d14, medians of 5 clean rounds)

| config | median NPS | vs baseline |
|---|---:|---:|
| baseline native + cpol (32e0a9c-equiv) | 591,802 | — |
| **9b695fb (copy elimination)** | 613,674 | **+3.7%** |
| **a3f892e (threat constants)** | 612,240 | **+3.5%** (≈ 0 over 9b695fb) |
| baseline, sidecar hidden (nocpol) | 612,699 | **+3.5%** |
| baseline x86-64-v3 build | 569,384 | −3.8% |
| PSQT (no net) | 3,280,984 | 5.54× |

Paired per-round ratios: 9b695fb/baseline = +2.4…+4.1% (mean +3.5%);
a3f892e/9b695fb = −0.6…+1.2% (mean ≈ +0.3%); nocpol/baseline =
+1.6…+4.7% (mean +3.1%).

Readings:
- **C3 settled**: the sidecar's runtime tax is ≈ **3–3.5% NPS** (consult
  1.5% cycles + stash + alloc + cache pollution), ON TOP of the +6.2%
  equal-depth tree inflation from §5. Combined effective cost while the
  head buys 0 Elo: ~9% — hiding the sidecar until it wins an ablation is
  the cheapest "optimization" in this study.
- **D16 note**: the strength-testing x86-64-v3 build is 3.8% slower than
  the machine-local native build on this box (kernels are runtime-dispatched
  AVX-512 either way; the delta is autovectorized + scalar code). SPRT
  pairs are internally consistent (same flags both sides) so no action,
  but absolute NPS quotes should name the build.
- The d47c net costs 5.54× NPS vs the PSQT shell.

### B3/B4. Counter rates (perf stat, wl50/d15, native + cpol, 14.0 s)

| metric | value |
|---|---|
| IPC | **2.26** (164.4 G instr / 72.9 G cycles @ 5.2 GHz) |
| L1-dcache miss rate | **25.9%** (17.8 G misses — the weight-row stream) |
| L1-icache miss rate | 2.9% |
| branch miss rate | 1.0% |
| frontend idle | 8% |
| L1D fills from local L2 | 17.3 G × 64 B ≈ **79 GB/s L2→L1** |
| L1D fills from L3/CCX | 0.60 G (3.3% of fills) |
| **fills from DRAM** | 72.1 M × 64 B ≈ **0.33 GB/s** |
| L2-dTLB accesses (L1-dTLB misses) | 450 M (≈ 0.9% of loads) |
| dTLB walks (L2-dTLB misses) | 17.0 M (3.8% of L2-dTLB accesses) |

**Memory-bound verdict (the #4 §2 open question): the workload is
L2-RESIDENT, not DRAM-bound.** The 77.7 MB threat table's *hot rows* and
the per-bucket L1 weights live in L2/L3; DRAM sees 0.33 GB/s against a
machine ceiling of tens of GB/s. int4/FT-compression stays REJECTED with
prejudice; prefetch items are about hiding ~14-cycle L2→L1 latency, not
bandwidth. The engine is compute/latency-bound with a healthy 2.26 IPC —
which is exactly why the l1_dot dependency-chain fix (pure ILP) is the
right #1.

dTLB pressure is real but bounded: ~450 M L1-dTLB misses + 17 M walks ≈
EST 2–5% of cycles. THP would reclaim most of that, BUT:

### B5. THP experiment — blocked on this box

`GLIBC_TUNABLES=glibc.malloc.hugetlb=1` (madvise path): **AnonHugePages
stayed 0** — under this box's memory pressure/fragmentation even
MADV_HUGEPAGE-at-fault gets no 2 MB blocks (defrag=madvise does direct
compaction, which evidently failed), and the dTLB counters confirm nothing
changed (17.5 M vs 17.0 M walks). Interleaved plain-vs-tunable rounds
(6 rounds): median +0.9% for the tunable run — ambiguous allocator side
effect, not THP. **Item #8 demoted**: an in-code aligned-alloc + madvise
would still be the right implementation, but on this 15 GB box with
training pipelines resident, expect hugepage allocation to fail routinely;
the realistic upside is the 2–5% dTLB band and only when memory is
unfragmented. Revisit after the l1_dot/L2-tail fixes.

### B6. Bandwidth ceiling (likwid-bench is unusable on this box — broken
AUR install, missing access daemon AND likwid-lua; used a gcc -O3
-march=native C triad instead)

Single-thread DRAM triad: **28.4 GB/s**. Workload DRAM demand: 0.33 GB/s =
**1.2% of a one-core ceiling** — the DRAM-bound hypothesis is dead on
arrival; see B4.

---

## 9. Study-#4 claims: scoreboard

| # | claim | verdict | data |
|---|---|---|---|
| C1 | threat channel 26–35% of cycles | **REFUTED** | 12.4% (perf), 15.6% (Ir/AVX2) |
| C1' | threat fold latency-bound, no prefetch | confirmed (shape); counters §8 | fold = 7.5%, zero prefetch in asm |
| C2 | head co-equal at 20–30% | **REFUTED (too low)** | 48.5% measured |
| C2' | f32 tail under-vectorized | **CONFIRMED, worse than claimed** | 2-lane xmm; 9.35 c/iter vs 4.06 ideal |
| C2'' | l1_dot loop-order redundancy | **CONFIRMED + new dep-chain finding** | mca 3.0× kernel headroom |
| C3 | policy machinery 3–7% cycles | **REVISED** | +3.5% NPS (B1 paired), plus +6.2% tree size (new) |
| C4 | smallnet p ∈ 0.25–0.40 | **REFUTED at position level** | 0.08–0.15 at 800–1000 cp gates |
| — | Box::new per eval | **CONFIRMED exactly** | 1.00/eval, 99.93% of temporaries |
| — | finny 4 copies/refresh | **CONFIRMED exactly** | 4.00 memcpys/refresh measured |
| — | evals/node ≈ 0.34 | confirmed (0.42 here) | set-dependent |
| — | huge pages worth checking | **CONFIRMED — 0 hugepages live** | AnonHugePages=0 vs 175 MB anon |

## 10. Re-ranked opportunity list (measured basis)

| # | item | measured base | expected NPS | effort | bit-exact |
|---|---|---:|---:|---|---|
| 1 | **l1_dot group-outer swap** (8 zmm accumulators, kills 7 nnz re-walks + 7/8 bounds checks + the serial chain) | 34.4% | **+15–25%** (mca bound +29%) | LOW (one function, parity tests exist) | YES |
| 2 | **explicit 2×zmm `propagate_l2` kernel** (preserve i-order) | 6.4% | +3–4% | LOW | YES |
| 3 | **consult gate review** (hide/env-off the sidecar until the head wins its ablation) — measured +3.5% NPS (B1) plus the 6.2% equal-depth tree tax | 3.5%+tree | **+3.5% NPS, ~+6% depth-speed** | TRIVIAL | n/a (tree changes) |
| 4 | **threat-fold degenerate-copy alias** (28% of folds) + resolve→prefetch→fold (#4 item 3, base now 7.5% not 26%; prefetch = L2→L1 latency, §8 B4) | 7.5%+2.45% memcpy | +2–4% | MED | YES |
| 5 | THP madvise on net+TT — **demoted** (§8 B5: hugepages unobtainable on this box under load; upside = the 2–5% dTLB band when memory permits) | dTLB 2–5% | +0–3% | LOW | YES |
| 6 | stash de-Box + finny in-place (with #4 item 5's other copies) — **largely LANDED as 9b695fb on the implementing branch, measured +3.5–4% (§11)** | ~1–2% residual | done/rider | LOW | YES |
| 7 | T10 smallnet — AFTER in-tree p counter; position-weighted p says +7–10% not +20% | — | +7–10%? | HIGH | n/a |
| 8 | quant unification (activate 5.9% → ~1%) — ride next format rev | 5.9% | +3–5% | MED-HIGH | NO |
| 9 | bulk net parse | 0.4–0.5 s/spawn | QoL | TRIVIAL | YES |
| 10 | find_nnz VBMI2 + static dispatch riders | 0.5%+0.3% | +0.5% | LOW | YES |

Items 1+2 alone target ~41% of all cycles with bit-exact, test-covered,
single-function changes — the study-#4 "package #2" was correctly aimed but
under-priced by ~2×: measured, it is worth **+20–30% NPS**, i.e. most of
the d47c width tax, before any training-side lever.

## 11. Verification of the implementing agent's branch (independent)

Branch `worktree-agent-a7e7069d489335126`, verified commits: `32e0a9c`
(baseline sync — bit-verified byte-equal to this study's tree) → `9b695fb`
(copy elimination, claimed "bit-identical, NPS ~neutral 371.4k→371.7k") →
`a3f892e` (threat-pipeline constants, claimed "bit-identical, +2.5%
interleaved 383.7k→391.3k"). Each commit built from `git archive` in an
isolated tree, same flags (native+sota release).

| check | 9b695fb | a3f892e |
|---|---|---|
| static-eval parity, 800 FENs (`eval` cmd, d47c-stage2-sb240) | **800/800 identical** | **800/800 identical** |
| search parity, node counts, wl50 × d13 | **50/50 identical** | **50/50 identical** |
| NPS vs baseline (B1 paired-interleaved, wl50/d14, core-pinned) | **+3.7%** | **+3.5%** |
| NPS vs previous commit | +3.7% | **+0.3% ± 1** |

Verdicts:
- **Bit-identical claims: CONFIRMED for both commits**, at both the eval
  and the search level.
- **9b695fb's "NPS ~neutral" UNDERSOLD ITSELF: it is +3.5–4% here.** Their
  371 k baseline says their measurement ran under heavy contention (this
  study measures the same binary at ~592 k pinned); the true copy-
  elimination win was inside their noise floor.
- **a3f892e's "+2.5%" is NOT REPRODUCED on this workload** (+0.3% ± 1 over
  9b695fb, paired over 5 clean rounds). Discrepancy flagged: different FEN
  set/depth and their contended environment are both plausible causes; the
  patch is bit-identical so it carries no risk, but its standalone NPS
  claim should not be quoted until re-measured on a quiet box. Their two
  commits COMBINED are worth ≈ +3.5–4% NPS — real, just distributed
  differently than their per-commit log claims.
- Their remaining packages (p2 head kernels, p3 — binaries named
  `engine-p2-*`/`engine-p3-*` were seen testing during this study) were not
  committed by study close; verify them with the same recipe
  (`scratchpad/verif/` + `parity800.uci` + B1 ladder).
