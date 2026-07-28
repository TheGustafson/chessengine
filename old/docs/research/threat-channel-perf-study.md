# ARCHON threat-channel performance study — ground truth, SF18 differential, ranked options

Date: 2026-07-26. Commissioned by the user ("the threat channel needs to be
faster"). **Study only — no engine source was modified.** Binaries were built
(allowed) and one net file was *copied and truncated* in scratch for an
ablation.

Predecessors this supersedes or corrects on specific points:
`perf-study-4-eval-2026-07-22.md`, `perf-study-5-measured-2026-07-23.md`,
`perf-study-6-reprofile-2026-07-23.md`,
`profiling-study-3-creative-2026-07-19.md`. Those measured the **1280-wide
d47c/d47s arch**; this is the first profile of the **768-wide D47V deploy
config** that now carries the SKP1 output-skip lane.

---

## 0. Headline — three corrections to the commission's premises

The brief asked me to verify three things. All three turned out to be wrong in
ways that change the recommendation.

**(1) "The engine pays ~24% of cycles in the threat channel."** NOT reproducible
at 768. Measured: **15.1%** on the endgame-heavy wl50 set, **18.0%** on the
middlegame-heavy default bench set. The 24% in `value-foundation-state.md:1595`
was measured on the **1280-wide d47p** net in a *game* config; both the width
and the position mix inflate it. At 1280 on the same wl50 set I measure 17.5%.

**(2) "The incremental observer was designed but never built."** It **was
built** — `src/eval/nnue/sota_threats.rs`, 1511 lines, the full ≤80-event
dirty-diff observer with a fully-lazy fold, a random-game replay correctness
gate, and prior optimization passes. `refresh_threat` survives only as (a) the
root seed, (b) the king-mirror-cross rebuild, and (c) the test oracle. It fires
**0.0346× per node / 0.0831× per eval** and costs **0.67% of cycles**. There is
no from-scratch-per-position problem to fix.

**(3) "768 is only 1.11× faster than 1280 because threats + shell dominate."**
The mechanism is wrong. The dominant cause is a **fully scalar,
branch-mispredicting 768-iteration f32 loop** — the D47V SKP1 output-skip lane
in `sota_head.rs:364-374` — that **only the 768 net carries** (the 1280 teacher
net has no SKP1 section). Direct A/B ablation: removing that lane raises the
768-vs-1280 speed edge from **1.19× to 1.45×** and costs **19.3% of all
cycles** — more than the entire threat channel, and it is **66% of every branch
mispredict in the engine**.

**Top recommendation: fix the SKP1 skip lane first (≈ +16% NPS bit-identical
for a one-line change, ≈ +24% fully vectorized), then the fold kernels'
address arithmetic (≈ +5–8%, bit-identical). The threat channel's own
remaining fat is third and much harder.**

---

## 1. GROUND TRUTH — a fresh profile of the 768 fast binary

### 1.1 Provenance and exact commands

Source: `main` @ `bf3ec90` (clean tree). The commission's designated binary
`engines/bin/archon-d47v-fast` was **reproduced from that revision** and proven
identical by node count, so every source citation below is exact:

```bash
CARGO_TARGET_DIR=$S/tg-768v3  RUSTFLAGS="-C target-cpu=x86-64-v3" \
  cargo build --release --features sota,arch-d47v
CARGO_TARGET_DIR=$S/tg-1280v3 RUSTFLAGS="-C target-cpu=x86-64-v3" \
  cargo build --release --features sota            # 1280/128/32 default consts
CARGO_TARGET_DIR=$S/tg-768prof RUSTFLAGS="-C target-cpu=x86-64-v3 -C force-frame-pointers=yes" \
  cargo build --profile profiling --features sota,arch-d47v
```

`tg-768v3` reproduces `archon-d47v-fast`'s d15 node total **7,354,557
bit-for-bit** and the same byte size (1,118,152). Note the binary is a **v3
build with *runtime* AVX-512 dispatch** — the `avx512::*_inner` kernels appear
as separate symbols rather than inlined, which is what makes the flat profile
below so cleanly attributable.

Workloads (both engines, identical FEN files):

```bash
# the study-5/6 continuity set: every 16th line of the 800-FEN calib file
awk 'NR%16==1' tools/nnue/calib/scale-check-800.fens > $S/wl50.fens   # 50 FENs

# depth-mode profile (share continuity with studies 5/6)
taskset -c 4 perf record -F 4999 -e cycles:u -o $S/p768.data -- \
  ./engines/bin/archon-d47v-fast bench 16 1 15 $S/wl50.fens depth /tmp/d47v-sb100-nocpol.cenn

# fixed-node profile (the SF18-matched dataset; 300k nodes x 50 positions)
taskset -c 4 perf record -F 3999 -e cycles:u -o $S/d-ours.data -- \
  ./engines/bin/archon-d47v-fast bench 16 1 300000 $S/wl50.fens nodes /tmp/d47v-sb100-nocpol.cenn
taskset -c 4 perf stat -e cycles:u,instructions:u,branches:u,branch-misses:u,\
L1-dcache-loads:u,L1-dcache-load-misses:u,dTLB-load-misses:u \
  ./engines/bin/archon-d47v-fast bench 16 1 300000 $S/wl50.fens nodes /tmp/d47v-sb100-nocpol.cenn

# deterministic call counts
taskset -c 6 valgrind --tool=callgrind --cache-sim=no --branch-sim=no \
  $S/tg-768v3/release/chessengine bench 16 1 20000 $S/wl50.fens nodes /tmp/d47v-sb100-nocpol.cenn
```

**Box caveat (honored):** an RTX 4060 Ti trainer was resident at 93% GPU the
whole session; load average 1.5–2.3. All engine runs are `taskset`-pinned to
core 4 (callgrind to core 6). Absolute NPS is depressed; **every conclusion
below rests on shares, per-node cycle counts, or interleaved paired ratios**,
never on an absolute.

`LLC-loads`/`LLC-load-misses` are `<not supported>` on this Zen 4 kernel — the
memory-boundedness question is answered with L1d + dTLB instead.

### 1.2 The cycle-share table (fixed-node dataset, 15,001,572 nodes)

`perf stat`: **98.997 G cycles / 15.0016 M nodes = 6599 cycles/node**,
IPC 1.706.

| # | symbol | cycles % | cycles/node | block |
|---|---|---:|---:|---|
| 1 | `SotaHead::forward_into` | **24.64** | 1626 | head (≈96% of it is the SKP1 lane — §1.4) |
| 2 | `MovePicker::next_move` | 8.16 | 539 | shell |
| 3 | `avx512::i8_deltas_into_inner` | **6.97** | 460 | **threat fold** |
| 4 | `Board::make_move_with::<TtPrefetch>` | 4.87 | 321 | shell |
| 5 | `HalfkaContext::finny_refresh_into` | 4.74 | 313 | HalfKA refresh |
| 6 | `avx512::i16_deltas_inner` | 4.71 | 311 | HalfKA refresh (in-place) |
| 7 | `ThreatContext::at_full` | **4.61** | 304 | **threat gen+resolve+psqt driver** |
| 8 | `Worker::search::<NonPv>` | 4.40 | 290 | shell |
| 9 | `avx512::l1_dot_inner` | 4.06 | 268 | head |
| 10 | `avx512::i16_deltas_into_inner` | 3.99 | 263 | HalfKA fold |
| 11 | `avx512::activate_inner` | 3.97 | 262 | head (FT merge + clamp + pairwise) |
| 12 | `MovePicker::new` | 2.90 | 191 | shell |
| 13 | `HalfkaContext::current_full` | 2.52 | 166 | HalfKA driver |
| 14 | `sota_threats::record_square_touch` | **2.18** | 144 | **threat generation** |
| 15 | `SotaContext::evaluate` | 1.36 | 90 | head driver |
| 16 | `avx512::l2_tail_inner` | 1.08 | 71 | head |
| … | `sota_features::refresh_threat` | **0.53** | 35 | **threat from-scratch** |
| … | `sota_threats::ripple` | **0.33** | 22 | **threat generation** |
| … | `sota_threats::record_outgoing` | **0.18** | 12 | **threat generation** |
| … | `sota_threats::refresh_psqt_threat` | **0.14** | 9 | **threat from-scratch** |
| … | `sota_loader::*` + libc | 1.33 | 88 | startup (net parse) |

### 1.3 Block totals — the four numbers the brief asked for

| block | cycles % | cycles/node | note |
|---|---:|---:|---|
| **head math** | **36.56** | 2413 | of which **SKP1 lane 19.3% / 1276 c/n** (§1.4); "real" head 17.2% / 1137 c/n |
| **search + movegen shell** | **28.12** | 1856 | 21 symbols; `next_move` alone is 8.16% |
| **HalfKA accumulator update** | **16.14** | 1065 | Finny refresh 9.45% ≫ fold 3.99% |
| **threat channel** | **15.12** | 998 | fold 6.97 + gen 2.69 + resolve/psqt/driver 4.61 + refresh 0.67 + push 0.18 |
| startup / loader | 1.33 | 88 | excluded from steady-state reasoning |

Threat-channel internals, and the **frequency × unit-cost** decomposition
(frequencies from callgrind, unit costs = share × cycles ÷ calls):

| threat sub-item | calls/node | calls/eval | cycles/call | cycles/node | share |
|---|---:|---:|---:|---:|---:|
| fold `apply_i8_deltas_into` | 0.7974 | **1.9150** | 577 | 460 | 6.97% |
| gen+resolve+psqt driver `at_full` | 0.4164 | 1.0000 | 730 | 304 | 4.61% |
| generation `record_square_touch` | 1.0180 | 2.4448 | 141 | 144 | 2.18% |
| generation `ripple` | 0.4439 | 1.0659 | 49 | 22 | 0.33% |
| generation `record_outgoing` | 0.2375 | 0.5704 | 50 | 12 | 0.18% |
| **from-scratch `refresh_threat` (+psqt)** | **0.0346** | **0.0831** | **1278** | **44** | **0.67%** |
| push `apply_at` | 1.0018 | 2.4060 | ~12 | 12 | 0.18% |

Reference frequencies: **0.4164 evals/node**; HalfKA fold 1.5896/eval; Finny
refresh 0.7434/eval at **2016 cycles/call** (624 c/n — the single most
expensive *feature-update* item in the engine, and it is HalfKA's, not
threats').

**The from-scratch refresh is not a performance problem.** It is expensive
per call (1278 cycles) but fires on 3.5% of nodes, totalling 0.67% of cycles.
Frequency × unit cost is the whole story and the profiler only shows the
product — which is why the brief's framing ("`refresh_threat` is a
FROM-SCRATCH rebuild") pointed at the wrong target.

### 1.4 The finding that reframes everything: the SKP1 skip lane

`SotaHead::forward_into` is 24.64% of cycles. Source-line annotation of the
line-tables build (`perf annotate -l`) puts **96.4% of that symbol** on four
lines — `sota_head.rs:366-371`, the D47V output-skip loop:

```
   69.48 sota_head.rs:368      ( sk[i * BUCKETS + bucket] indexing )
   20.70 sota_head.rs:367      ( if a != 0 )
    2.62 sota_head.rs:366      ( for (i, &a) in ft.iter().enumerate() )
    0.51 sota_head.rs:371
    3.10 iterator glue (non_null.rs:1714, macros.rs:180)
```

Whole-program source-line aggregation agrees independently:
`sota_head.rs:368` = 16.62%, `:367` = 5.02%, `:366` = 0.63% → 22.3% before
glue.

Its instruction mix is the diagnosis:

| `forward_into` opcode mix | share of symbol |
|---|---:|
| `vaddss` (**scalar** f32 add) | 30.81% |
| `test` (the `a != 0` predicate) | 18.00% |
| `add` (stride-8 index advance) | 15.16% |
| `vmulss` (**scalar** f32 multiply) | 13.06% |
| `cmp` / `je` | 15.53% |
| **register width: 0% zmm, 0% ymm, "xmm" = single-lane `*ss` ops** | |

It is a 768-iteration scalar dot product with an unpredictable data-dependent
branch and a 32-byte-strided f32 load. The weight table is `768 × 8` f32 =
24 KB in **lane-interleaved** order (`i * BUCKETS + bucket`), confirmed from
the net file: SKP1 section = 24,588 bytes = 12 B header + 768·8·4.

**Direct ablation** (the loader interprets a trailing section only if its magic
matches — `sota_loader.rs:459-479` — so truncating the net at the SKP1 offset
yields a valid, skip-free net):

```bash
head -c 84133996 /tmp/d47v-sb100-nocpol.cenn > $S/d47v-noskip.cenn
```

| counter (fixed 300k×50 nodes, pinned) | skip ON | skip OFF | Δ = the loop |
|---|---:|---:|---:|
| cycles | 99.945 G | 80.617 G | **19.33 G = 19.3% of ON** |
| instructions | 169.08 G | 150.93 G | 18.16 G (2907/eval, 3.8/iter) |
| branches | 26.45 G | 20.36 G | 6.09 G |
| **branch-misses** | **1.168 G** | **0.394 G** | **0.774 G = 66% of ALL engine mispredicts** |
| L1d loads | 67.91 G | 64.65 G | 3.25 G (520/eval) |
| L1d load-misses | 10.486 G | 9.920 G | 0.566 G |
| NPS (median of 3, interleaved) | 809,626 | 998,174 | **+23.3%** |

**Verdict: the loop is branch-mispredict-bound, not load-bound.** 0.774 G
mispredicts × ~18 cycles ≈ 13.9 G cycles ≈ **72% of the loop's 19.3 G**. The
`a != 0` test mispredicts at 16% over 768 iterations = 124 mispredicts/eval.
Cost per eval: **3093 cycles** (4.0 cycles/iteration) for work whose vector lower bound is ~50–100
cycles (48 zmm FMAs) — a **~35× overhead**.

**Reconciling the three estimates of the lane's size.** `perf annotate` puts it
at 96.4% of a 24.64% symbol = **23.8%** of cycles; the counter Δ above says
**19.3%**; a second, independently-multiplexed `perf stat` of the ON config
(98.997 G, the run used for §1.2/§5.2) gives **18.6%**. The counter Δ is the
trustworthy one and 19.3% is used throughout — the annotate over-attributes
because sampling skid from surrounding stalls piles into a tight scalar loop,
and because removing the loop also removes some downstream effects that
annotation cannot separate. Treat the lane as **18.6–19.3% of cycles, ~1250–1280
cycles/node**, and treat 23.8% as an upper bound only.

The ablation is a *price tag, not a proposal*: SKP1 is a learned term and
removing it changes the eval (800-FEN eval hashes differ, §7.1). The fix is to
compute the same value faster.

### 1.5 Position-mix sensitivity (do not quote one number)

Threat cost is a function of how many attack relations exist, so it moves with
material:

| workload | threat channel | i8 fold | `at_full` | `forward_into` |
|---|---:|---:|---:|---:|
| wl50 (endgame-heavy, 50 FENs) | **15.1%** | 6.97% | 4.61% | 24.64% |
| default bench FENs (middlegame-heavy) | **18.0%** | 9.41% | 5.63% | 21.79% |

Real games sit at or above the default-fens end. Quote the workload with any
threat-share number — this is exactly how the 24% claim got stranded.

---

## 2. CURRENT IMPLEMENTATION ANALYSIS

### 2.1 What `refresh_threat` actually is, and when it runs

`sota_features.rs:515-529`. Seeds a zero accumulator, then calls
`for_each_threat` (`:413-471`) and adds one 768-wide i8 weight column per
active threat via `add_column` (`:481-486`, a scalar-but-autovectorized
wrapping-i16 loop).

`for_each_threat` iterates: for each color, for each pawn — the two diagonal
captures masked to `P|N|R` victims, plus the push relation against a pawn
directly in front; then for each of N/B/R/Q, `attacks(pt, from, occ) & targets`
per from-square. Kings are never attackers. Cost is **O(active attack edges)**
— up to 128 features per perspective (the FullThreats `MaxActiveDimensions`),
each costing a 768-byte random row read plus 768 wrapping adds.

It is invoked from exactly three places:

1. `ThreatContext::refresh` (`sota_threats.rs:693-711`) — root seed, once per
   `position` command.
2. `ThreatContext::apply_at` (`:802-816`) — **only** when `king_crossed(dp,
   persp)` (`:566-585`), i.e. that perspective's own king crossed the d/e file
   boundary. Measured **0.0346/node**.
3. The test oracle (`refresh_pair` in the replay gates).

So: **not per-position, not per-eval, and not per-node.** The design's tight
trigger is the reason. A threat index depends on the perspective's king only
through `orient_threats` (`sota_features.rs:177-179`) — a **single mirror
bit** — never through a king bucket. So the refresh condition is a *strict
subset* of HalfKA's (which must refresh on any of 32 `KING_BUCKETS` changing).

### 2.2 Where the cost actually is: the incremental path

`ThreatContext` (`sota_threats.rs:646-948`) is a per-ply stack, fully lazy:

- **`apply_at`** stores only a ~140-byte `ThreatSnapshot` (mailbox + occupancy
  + both king squares) plus the `DirtyPiece`. Nothing is generated, resolved,
  or folded. 1.0/node, ~12 cycles.
- **`compute`** (`:828-928`), triggered by `at_full`, walks from the nearest
  computed ancestor and per child does: `build_deltas` (regenerate raw events
  from the snapshot by replaying the edits) → `resolve_into` **once per
  perspective** → `apply_i8_deltas_into`.

Measured effect of the laziness: 1.0018 pushes/node but only **0.4164 folded
plies/node** — essentially exactly one folded ply per eval, both perspectives
(1.915 kernel calls/eval). The lazy design is working at close to its
theoretical optimum; there is no scheduling waste left to harvest.

`at_full`'s 304 c/n splits by source line as:

| region | share of `at_full` |
|---|---:|
| `resolve_into` psqt-branch body (`sota_threats.rs:993-1002`) | 31.5% |
| `ThreatResolver::index` (`sota_features.rs:374-382`) | 10.9% |
| psqt delta apply (`sota_threats.rs:918-925`) | 4.2% |
| `build_deltas` + `DeltaBuf::push` + glue | remainder |

Note the samples land in the **non-empty `psqt_table`** branch, and both nets
do carry a `PSQ1` section (verified: d47v @ 81470048, d47s @ 136843872). Each
resolved event therefore also does a random 32-byte read from the 1.94 MB
threat-PSQ1 table (`psqt_add`, `sota_accumulator.rs:23-28`) — a cache-miss
generator riding along with every event.

### 2.3 The asymmetry vs HalfKA — and its real shape

The brief expected the asymmetry to be "threats rebuild, HalfKA is
incremental." That is not the current state. Both are incremental and lazy.
The real asymmetry is **event count**, and it is intrinsic to the feature set:

| | HalfKA | FullThreats |
|---|---|---|
| feature semantics | one feature per **piece** | one feature per **directed attack relation** |
| events per move | ≤ 2 sub / ≤ 2 add (from `DirtyPiece`) | **≤ 80** (design bound); x-ray ripple + short-range incoming |
| index depends on king via | 32 buckets **+** mirror | **1 mirror bit only** |
| refresh trigger | any own-king move | own king crosses d/e (strict subset) |
| refresh cost mitigation | **Finny cache** (0.7434/eval, 2016 c/call) | none needed (0.0831/eval) |
| weight width | i16 | i8 (half the row bytes) |
| measured cycles/node | 1065 | 998 |

Which means the two channels cost **the same** per node today, for reasons
that are mirror images: HalfKA pays a cheap fold but an expensive, frequent
Finny refresh (624 of its 1065 c/n); threats pay a rare refresh but a fold at
20–40× the event count (460 of its 998 c/n, plus 178 c/n generating those
events and 304 c/n resolving them).

### 2.4 The fold kernels are address-arithmetic-bound — a second big finding

Both fold kernels share one shape (`sota_simd.rs:823-851` avx512,
`:602-651` avx2):

```rust
for off in (0..ACC).step_by(LANES) {          // 24 blocks at ACC=768, LANES=32
    let mut value = load(src + off);
    for &feature in adds { let row = weights.as_ptr().add(feature * ACC + off); ... }
    for &feature in subs { ... }
    store(dst + off, value);
}
```

`feature * ACC + off` is re-derived **inside** the block loop — 24 times per
event. Measured consequences:

| kernel | dynamic: zmm | dynamic: scalar/GPR | static instrs | static touching zmm | hot mnemonics |
|---|---:|---:|---:|---:|---|
| `i8_deltas_into_inner` (threats) | 47.6% | **52.4%** | 363 | 78 (**21%**) | `vpaddw` 32.0, `inc` 19.6, `add` 15.2, `vpsubw` 10.7 |
| `i16_deltas_into_inner` (HalfKA) | 1.5% | **98.5%** | 3535 | 572 (**16%**) | `inc` 55.3, `add` 13.0, `lea` 7.3, `mov` 5.3 |

The dynamic `inc` pileup is partly sampling skid, but the **static** mix is
not: `i16_deltas_into_inner` contains 499 `shl` + 454 `lea` + 707 `mov` against
400 `vpaddw` + 50 `vpsubw` + 122 `vmovdqu64` — roughly **two scalar
address-math instructions per vector operation**. (The `shl $0x9` + `lea
(%rsi,%rsi,2)` pair visible in the disassembly is literally `feature × 512 × 3
= feature × 1536`, the i16 row stride, recomputed per block.)

The pathology is worst exactly where the event count is smallest: with ≤2 adds
and ≤2 subs, HalfKA's inner loops have trip count 1–2, so per block you pay
two loop preambles and two full index derivations for two vector adds. Hence
98.5% scalar. Contrast SF, which resolves this with **compile-time
specialization** — its binary contains
`AccumulatorUpdateContext<...>::apply<(UpdateOperation)0,0,1,1,unsigned,
unsigned,unsigned,unsigned,true>`, a template-instantiated 2-add/2-sub kernel
with no runtime list loop at all.

---

## 3. THE DESIGNED INCREMENTAL OBSERVER (§12.9.1, not §12.2)

The brief pointed at "§12.2". The ≤80-event observer is specified in
**`design/11-evaluation.md` §12.9.1** (`:1566-1608`), titled "Incremental
FullThreats update (the ≤80/move dirty-diff observer)". §12.2 (`:1347-1372`)
is the accumulator/refresh/cache contract that *states* the ≤80 figure and the
mirror-only refresh trigger; §12.9.1 is the design.

### 3.1 What §12.9.1 specifies

- **Core primitive `record_square_touch(P, s, add, occ)`** with three parts:
  (1) *outgoing* — `piece_attacks(P,s,occ) & occ`, pawns via
  `pawn_push_or_attacks`; (2) *x-ray ripple* — for each slider attacking `s`,
  the single first occupant beyond `s` on the ray via `ray_pass(S.sq,s) & occ &
  slider_attacks(s)`, emitted with **flipped sign**, plus the slider's direct
  relation to `P@s`; (3) *incoming* — short-range attackers (pawns by color
  symmetry, knights). Kings never emitted as attackers; for `P` = king, run
  **only** the ripple.
- **Three drivers**: `on_move` (shared occupancy `occ ^ to` for both endpoint
  touches, which makes the along-a-ray discovered victim cancel — the doc
  explicitly calls this "Reckless's elegant replacement for SF's
  `noRaysContaining` guard"), `on_change`, `on_mutate` (relabel with no
  ripple). Plus an explicitly-offered **"simpler first-cut": `on_change` every
  `DirtyPiece` sub then every add — auto-cancels, costs extra events.**
- **Buffers**: `[ThreatDelta; 128]` with a `debug_assert len ≤ 80`.
- **The bound's derivation** (matching SF's own): ≤8 outgoing + ≤16 incoming
  per touched square, ×3 touches for a capture, +8 discovered = `(8+16)·3+8 =
  80`.
- **Refresh trigger**: mirror-half only; "piggyback: refresh threats whenever
  the PST accumulator refreshes (superset-safe; a separate mirror flag is a
  later optimization)". No Finny cache for threats.
- **The correctness gate (non-negotiable)**: random-game replay asserting
  `incremental == from_scratch` at every node.

### 3.2 It was implemented — assess what shipped vs what was designed

| §12.9.1 item | status in `sota_threats.rs` |
|---|---|
| `record_square_touch` 3 parts | **built**, `:259-390` |
| x-ray ripple via `ray_pass` | **built**, `ripple()` `:399-426` |
| `on_mutate` (no-ripple relabel) | **built**, `record_mutate` `:432-436` |
| the three optimized drivers | **NOT built** — the **first-cut** shipped (`build_deltas` `:1014-1111`, module doc `:29-34`) |
| `noRaysContaining` / along-ray cancellation | **NOT built** — `ripple` emits unconditionally |
| `[ThreatDelta; 128]`, `len ≤ 80` | **relaxed** to `DELTA_CAP = 256` "and never rely on the tight bound holding here" (`:108-112`) — because the first-cut over-generates |
| mirror-only refresh (better than the doc's piggyback) | **built and tighter**, `king_crossed` `:566-585` |
| replay correctness gate | **built**, `threats_replay_matches_refresh` + `_large` `:1486-1510` |
| **fully-lazy generation + resolution** (beyond the design) | **built** — study-3's restructure, `:36-53` |

**So the honest answer to "why was it never built" is: it was, and then it was
optimized twice** (study 3's fully-lazy schedule, study 4's `ThreatResolver`
hoist + resolve/fold split + null-move aliasing). What was *never* built is the
**optimized 3-primitive driver** — the design deliberately offered the
first-cut as a correctness-first stepping stone (`:29-34` cites §12.9.1 for
exactly this), and the stepping stone became the shipped state.

The cost of stopping at the first-cut, and the reason it is not a trivial
follow-up:

1. The first-cut must **reconstruct** intermediate board states, because it
   runs after the move rather than inside the mutators. `build_deltas` rebuilds
   pre-move occupancy from the snapshot + `DirtyPiece`, then replays edits
   carrying an 8-slot override map whose lookups are a **linear scan**
   (`Replay::find`, `:220-221`) called from every identity resolution
   (`resolve`, `:243-251`) — i.e. once per emitted event. The correctness
   argument for this (the telescoping) occupies 16 lines of module doc
   (`:59-74`) and is the single subtlest thing in the file.
2. It over-generates: `record_outgoing` uses `& occ` where the victim mask
   could be `& (P|N|R)` for pawns and `& !kings` for sliders, so every
   king-victim and pawn→B/Q edge is generated, buffered, and resolved
   **twice** (once per perspective) before `threat_index` drops it.
3. No along-ray cancellation, so every sliding move emits a spurious sub/add
   pair that survives to the fold.

Effort to close: the 3-primitive driver requires wiring generation into the
`MakeMoveHooks` seam (so occupancy is naturally correct mid-move) or
equivalently threading exact per-edit occupancy through the replay — a
**2–4 day** change touching the most safety-critical file in the eval, gated
by the existing replay tests. It is real work, and §5 prices it accordingly.

---

## 4. HOW STOCKFISH DOES IT — distilled (ideas only, never copied)

**Source provenance note.** `stockfish_reference/Stockfish/`'s *worktree* is at
tag `sf_16`, which has no FullThreats. But the clone's **object store contains
post-SF18 dev refs**, and the sources were read first-hand from
`origin/master` = `ebcea3ef` (2026-07-16, `misc.cpp:57` version = "dev") via
`git show origin/master:<path>` without checking anything out. Line citations
below are primary. Two independent traps found while establishing this:
`/home/nick/projects/nnue-pytorch/.../full_threats.py` is a **stale revision**
(`NUM_INPUTS = 59808`, `HASH = 0x2E6B9D04` vs the C++ `Dimensions = 60720`,
`0x8f234cb8`) and must not be used as the feature-space authority; and that dev
ref has **removed eval bimodality entirely** (`evaluate.cpp:42-50` takes a
single `Network&`), whereas the **binary I profiled** (`Stockfish 18` release,
with an `EvalFileSmall` option and both `<1024u>` and `<128u>` network
symbols) still has it. Both facts are used below where relevant.

### 4.1 Feature space (confirms our port is faithful)

`Dimensions = 60720` (`full_threats.h:41`), tiled by `init_threat_offsets`
(`full_threats.cpp:117-151`) as `Σ numValidTargets[pc] · cumulativePieceOffset[pc]`:

| attacker | empty-board edges | `numValidTargets` | block | vs naive from×to |
|---|---:|---:|---:|---:|
| P | 132 | 6 | 792 | 31.0× compression |
| N | 336 | 10 | 3360 | 12.2× |
| B | 560 | 8 | 4480 | 7.3× |
| R | 896 | 8 | 7168 | 4.6× |
| Q | 1456 | 10 | 14560 | 2.8× |
| K | 420 | **0** | **0** | dead |
| | | | **30360 / color → 60720** | |

`make_index` (`full_threats.cpp:193-206`) is a branch-free 3-term sum:
`index_lut1[att_o][atd_o][from_o < to_o] + offsets[att_o][from_o] +
index_lut2[att_o][from_o][to_o]`, where `lut2` is the **dense edge ordinal**
`popcount(((1<<to)-1) & attacks)` — the whole compression trick. Exclusion uses
a **sentinel, not a branch**: excluded pairs get `feature = Dimensions`, offsets
are still added (landing in `[60720, 62203]`), and filtering is the branchless
`push_back_if_lt` (`misc.h:213-217`, unconditional store then `size_ += (v <
max)`).

The `map` exclusions are *not* arbitrary: every excluded pair is one whose
relation is already captured by the reverse direction (B→Q is redundant with
Q→B; P→B with B→P), while P→N and P→R are **kept** because the geometries
differ. `semi_excluded` (`:164-177`) drops one direction of same-type pairs via
the `from_o < to_o` slot, **except same-color pawn→pawn**, which encodes the
directional *blocked-push*.

Our `sota_features.rs` reproduces all of this correctly, including the
inverted mirror polarity (`FullThreats::OrientTBL` is a1→SQ_A1 for king files
a–d, the exact opposite of `HalfKAv2_hm::OrientTBL` — `full_threats.h:45-54`
vs `half_ka_v2_hm.h:83-92`). No index-math divergence was found.

### 4.2 The schedule — SF is eager-generate / lazy-resolve / lazy-fold

| stage | SF | citation | ARCHON |
|---|---|---|---|
| raw event generation | **EAGER**, fused into `put_piece`/`remove_piece`/`move_piece`/`swap_piece` | `position.h:383-423` | **LAZY**, reconstructive replay |
| per-perspective resolution | **LAZY** at eval | `nnue_accumulator.cpp:418-426` | **LAZY** at eval |
| accumulator fold | **LAZY** at eval | `:429`, `:158-244` | **LAZY** |
| refresh from scratch | LAZY, on demand | `:98` | eager at push (king cross) |

SF generates eagerly because for them it is **nearly free**: at the instant
`put_piece` runs, `pieces()` *is* the correct intermediate occupancy, so there
is nothing to reconstruct. We defer generation precisely because ours is
expensive (the replay + override-map scan). **These two design choices are not
independent** — if our generation were made cheap, our laziness would be worth
much less.

**Answering the brief's question directly: SF computes threats incrementally,
resolved lazily.** Their `update_threats_accumulator_full<1024u>` (the
from-scratch path) is **0.49% of cycles** in my profile — the same order as our
0.67%.

### 4.3 What bounds their cost

`types.h:334-341` derives the **80** bound exactly as our design doc does
(≤8 outgoing + ≤16 incoming, ×3 touches, +8 discovered; castling `(5+1+3+9)·2 =
36`). The container is `ValueList<DirtyThreat, 96>` — **the 16 extra slots
exist solely so the AVX-512 path's unmasked 64-byte store cannot run off the
end.** Index lists: `MaxActiveDimensions = 128` for threats vs 32 for HalfKA
(`full_threats.h:67`, `half_ka_v2_hm.h:96`).

The ripple (`position.cpp:1182-1322`) costs **exactly two magic probes per
touched square** — `attacks_bb<ROOK>(s,occ)` and `attacks_bb<BISHOP>(s,occ)`
(`:1192-1193`) — reused for forming the incoming-slider set *and* the
discovered query. `RayPassBB` is a 64×64 precomputed table; intersecting it
with `rAttacks|bAttacks` yields the first blocker strictly beyond `s`, hence
`assert(!more_than_one(discovered))` (`:1208`). Kings are excluded from the
blocker set via `occupiedNoK` (`:1194`).

Two suppression tricks we lack:

- **`noRaysContaining`** (`position.h:410-422` → `position.cpp:1209`):
  `move_piece` passes `from|to`, and the discovered event is skipped when the
  ray contains both endpoints — a piece sliding *along* its own ray changes
  nothing behind it. This is add/sub cancellation performed **at generation
  time**, so the cancelled pair never enters the list, is never resolved twice,
  and never touches the accumulator.
- **`ComputeRay = false`** (`position.h:431,436`): for same-square mutations
  (promotion, capture-replacement) the entire slider lambda is **compiled
  out**. We have the equivalent (`record_mutate`) — parity here.

**Type partitioning by construction** (`position.cpp:1236,1243-1254,1196-1197`):
each contributing attacker set is built already type-correct (knights via
`PseudoAttacks[KNIGHT] & knights`, sliders via `rookQueens & rAttacks`), so
popping the union bit-by-bit needs **zero per-bit type re-tests**. Our
`record_incoming_mutate` (`sota_threats.rs:487-500`) does a `match` on piece
type plus 1–2 `Bitboard::contains` probes per popped bit.

### 4.4 Their vectorization

**Event emission** (`write_multiple_dirties`, `position.cpp:1146-1175`, gated
on `USE_AVX512ICL`). Because `DirtyThreat` is exactly 4 bytes and the whole
64-square mailbox fits in one zmm, up to 16 events are manufactured
**entirely in-register**: broadcast the record template; bit-compress the
target bitboard's set bits into packed square bytes and sign-extend to 16×32;
use those bytes as a **byte-granular in-register permute index into the mailbox
vector** to gather 16 piece codes with no memory access; shift squares and
pieces into their bitfield slots; merge template ∨ squares ∨ pieces with a
**single three-input ternary-logic instruction**; one unmasked 64-byte store;
advance length by the true popcount. Called twice per touch — once with the
outgoing field shifts, once with the incoming pair.

**The delta fold** (`apply_combined`, `nnue_accumulator.cpp:158-391`): an outer
loop over register tiles of the 1024-wide accumulator; per tile, load the
parent's lanes into `NumRegs` vector registers, then apply **all four lists
into those same live registers** — `psqRemoved`/`psqAdded` as i16 sub/add,
`thrRemoved`/`thrAdded` as **i8 columns widened on the fly** (`vec_convert_8_16`,
`:208-240`) — then one store. Threat weights are i8 (`nnue_common.h:59`), half
the row traffic, which matters at 128 active features. NEON gets
`vaddw_s8`/`vsubw_high_s8` (widen-and-accumulate in one instruction); RVV gets
a separate scalable `vsetvl` path with `vwadd.wv`.

Measured, this shows up exactly as expected. `Network<1024u>::evaluate` is
**65.3% zmm / 33.0% scalar-GPR** (vs our fold kernels' 47.6% and 1.5% zmm),
and a 512-byte-region split of the symbol separates their pipeline cleanly:

| region class (by signature mnemonics) | % of symbol | % of all cycles |
|---|---:|---:|
| threat fold — i8→i16 widen (`vpmovsxbw` + `vmovdqa64` + `vpaddw`/`vpsubw`) | 35.65 | 12.29 |
| accumulator fold add/sub, i16 (`vpsubw`/`vpaddw`/`vmovdqa64`) | 23.90 | 8.24 |
| scalar index-resolve / control | 18.32 | 6.32 |
| head L1 matmul (`vpdpbusd`) | 13.91 | 4.79 |
| FT activation (`vpminsw`/`vpmulhw`/`vpsllw`) | 8.04 | 2.77 |

Note `vpmovsxbw` at 17.1% of the symbol: **SF pays a large i8→i16 widening tax
too.** This is not a place we are uniquely bad.

One reference fact worth recording: SF issues **software prefetch of the weight
rows during index resolution, one stage ahead of the fold**
(`full_threats.cpp:297-299` driven by `nnue_accumulator.cpp:413-419`). We
implemented and **measured a loss** for this (`sota_threats.rs:861-866`, −2.5%
NPS full-row). Their tiling and ours differ enough that the technique does not
transfer; do not retry it without a new mechanism.

### 4.5 Cross-ply optimizations (we have none of these)

- **Forward multi-ply fold** (`nnue_accumulator.cpp:121-136`) — we have this.
- **Backward fold** (`:138-154`): refresh the *leaf* from the Finny cache, then
  fold **backward** toward an earlier ply by simply **swapping the added/removed
  list arguments** (`:424-426`) — the same kernel, reversed, no separate code
  path. Confirmed live in the binary:
  `update_accumulator_incremental<false, FullThreats, 1024u>` = 0.14%,
  `<false, HalfKAv2_hm, 1024u>` = 0.40%.
- **2-ply fused updates**: `double_inc_update<1024u>` for FullThreats = **1.78%
  of cycles**, for HalfKA 0.64%.
- **No cross-ply add/sub cancellation** at the fold level — their only
  cancellation is the intra-move ray suppression.

### 4.6 One place we are structurally *better* than SF

SF fuses HalfKA and threats into **one** accumulator (`accumulation[COLOR_NB][1024]`,
`nnue_accumulator.h:47`). Consequently it has **one** refresh trigger, and it
is HalfKA's — `find_last_usable_accumulator` consults only
`PSQFeatureSet::requires_refresh` (`nnue_accumulator.cpp:114`), with a comment
at `:112-113` acknowledging that threats would need only a subset. And **SF has
no Finny table for threats** (`:538-539` says so; `:576-577,615-630` confirm the
cache entry carries only the HalfKA delta, with threat columns re-added on top
of the register tile on *every* refresh, up to 128 i8 columns).

So SF eats a genuinely expensive threat rebuild on **every king-bucket change**;
we eat it only on a **d/e mirror crossing**. That is a real, defensible design
advantage of ours — and it is the direct price of their fused-accumulator win
(one tile load/store covering both channels; we write 2×768×2×2 = 6144 B of
accumulator per folded ply vs their 1×1024×2×2 = 4096 B, plus we pay an extra
1536-lane i16 add at every eval in `activate` that they don't pay at all).

---

## 5. DIFFERENTIAL EXECUTION COMPARISON — ARCHON vs SF18, same positions

### 5.1 Harness

Both engines' own `bench` with a **fixed node limit** on the **same 50 FENs**,
single thread, 16 MB TT, pinned to core 4. Node totals land within **0.06%**,
so cycles/node is directly comparable:

```bash
taskset -c 4 ./engines/bin/archon-d47v-fast bench 16 1 300000 $S/wl50.fens nodes /tmp/d47v-sb100-nocpol.cenn
taskset -c 4 ./engines/bin/stockfish18     bench 16 1 300000 $S/wl50.fens nodes
```

`engines/bin/stockfish18` is **not stripped** (1235 symbols, richly templated),
so subsystem attribution came free — no rebuild was needed.

**Bimodality handling.** The binary *does* have two nets (`EvalFileSmall`;
`NetworkArchitecture<1024u,…>` and `<128u,…>`). I **segment by the symbol's
width parameter**: all `<128u>` symbols total **4.24% of cycles** (`Network<128u>
::evaluate` 2.81 + `refresh_cache<128u>` 1.06 + `incremental<false,HalfKA,128u>`
0.20 + `double_inc_update<128u>` 0.17). Contamination is small and is quoted
separately everywhere below. Critically, **the `<128u>` symbols carry no
`FullThreats` instantiation at all** — SF's small net is HalfKA-only, so SF pays
threat cost *only* on big-net evals. Also `evaluate.cpp:48` asserts
`!pos.checkers()`, so SF never evaluates in check.

Caveats that cannot be removed: the two search shapes differ (node mix, qsearch
share), and node definitions are not guaranteed identical. **Compare shares and
cycles/node; do not compare absolute layouts.**

### 5.2 Whole-engine counters

| metric | ARCHON 768 (+skip) | ARCHON 768 (−skip) | SF18 | ours(+skip) ÷ SF18 |
|---|---:|---:|---:|---:|
| nodes | 15,001,572 | 15,001,562 | 15,010,238 | 1.000 |
| cycles | 98.997 G | 80.617 G | 49.830 G | — |
| **cycles/node** | **6599** | **5374** | **3320** | **1.99×** (−skip: 1.62×) |
| instructions/node | 11,271 | 10,061 | 4,331 | 2.60× |
| IPC | 1.706 | 1.872 | 1.305 | 1.31× |
| branches/node | 1763 | 1357 | 395 | 4.46× |
| **branch-misses/node** | **77.9** | **26.28** | **26.34** | 2.96× (−skip: **1.00×**) |
| branch-miss rate | 4.42% | 1.94% | 6.67% | — |
| L1d loads/node | 4527 | 4310 | 2370 | 1.91× |
| L1d miss rate | 15.4% | 15.3% | 17.8% | 0.87× |
| dTLB load-misses/node | 1.43 | — | 2.21 | 0.65× |

Two things jump out.

**We are compute/front-end-bound, not memory-bound.** Our L1d miss *rate* is
*lower* than SF18's (15.4% vs 17.8%) and our dTLB misses per node are 35%
*fewer*, yet we burn 2.0× the cycles at 2.6× the instructions. Our IPC is
*higher* than theirs. The gap is **instruction volume**, not bytes. This
refutes any "compress the threat table / reduce the byte stream" framing as the
primary lever — consistent with study 3's SVD result that killed low-rank
factorization for a different reason.

**Strip the skip lane and our branch behavior becomes SF18's, to three digits:**
26.28 vs 26.34 mispredicts/node. That single loop is responsible for the entire
branch-prediction gap between the two engines.

### 5.3 The killer table — feature-update machinery, cycles/node

| subsystem | ARCHON | | SF18 | | ratio |
|---|---:|---:|---:|---:|---:|
| | **c/node** | **%** | **c/node** | **%** | ours ÷ SF |
| **threat: fold** | 460 | 6.97 | 408 | 12.29 | 1.13× |
| threat: 2-ply fused fold | — | — | 59 | 1.78 | — |
| **threat: event generation** | 178 | 2.69 | 162 | 4.89 | 1.10× |
| **threat: per-perspective resolution** | 304¹ | 4.61 | 168² | 5.06² | 1.81× |
| **threat: from-scratch refresh** | **44** | **0.67** | **16** | **0.49** | 2.75× |
| threat: push / bookkeeping | 12 | 0.18 | — | — | — |
| **THREAT SUBTOTAL** | **998** | **15.12** | **814**² | **24.5**² | **1.23×** |
| HalfKA: fold | 263 | 3.99 | 274 | 8.24 | 0.96× |
| HalfKA: refresh (Finny) | 624 | 9.45 | 231 | 6.96 | **2.70×** |
| HalfKA: 2-ply fused / backward | — | — | 39 | 1.18 | — |
| HalfKA: driver + resolution | 178 | 2.70 | 42² | 1.26² | 4.2× |
| **HALFKA SUBTOTAL** | **1065** | **16.14** | **586** | **17.6** | **1.82×** |
| **ALL FEATURE UPDATE** | **2063** | **31.26** | **1399** | **42.1** | **1.47×** |
| head math (excl. our skip lane) | 1137 | 17.23 | 251 | 7.56 | 4.53× |
| **our SKP1 skip lane** | **1276** | **19.34** | **0** | **0** | **∞** |
| small-net eval (segmented out) | — | — | 141 | 4.24 | — |
| search + movegen shell | 1856 | 28.12 | 1385 | 41.71 | **1.34×** |
| **eval layer total** | **4476** | 67.8 | **1791** | 53.9 | **2.50×** |
| | (−skip: **3200**) | | | | (−skip: **1.79×**) |

¹ our `at_full` = resolution + `build_deltas` glue + the threat-PSQ1 delta.
² SF's 6.32%-of-cycles scalar index-resolve region inside `Network<1024u>
::evaluate` is **shared** between the two feature sets; I assign 80% to threats
(they have ~80 events vs HalfKA's ≤4, and 128 vs 32 active features). Every
starred cell inherits that assumption; the unambiguous number is the
**ALL FEATURE UPDATE** row.

**Reading of the differential:**

1. **Our shell is at rough parity** (1.34×). Their `next_move` costs *more*
   per node than ours (430 vs 539 c/n is ours worse by 1.25×, but their
   `search<NonPv>` is 505 vs our 290 — they run more heuristics per node). The
   search gap is not where our 2× lives.
2. **Our threat channel is only 1.23× SF18's** — and their *share* is much
   larger than ours (24.5% vs 15.1%) because their total is half ours. Per
   node, our threat machinery is *close to reference quality*. The fold is
   1.13×; generation 1.10×. This is the study's most surprising result and it
   sharply lowers the ceiling on any threat-specific optimization.
3. **The two places we are genuinely 2.7–4.5× off are not threats**: HalfKA's
   Finny refresh (624 vs 231 c/n) and the head math (1137 vs 251 c/n) — plus
   the skip lane, which has no counterpart at all.
4. **`refresh_threat` at 2.75× their equivalent is a share of 0.67%.** Even a
   perfect fix returns <0.5% NPS. It is not a target.

### 5.4 The inner update loops, side by side (measurement, not code)

| | ARCHON `i8_deltas_into_inner` | ARCHON `i16_deltas_into_inner` | SF18 fold (inside `evaluate`) |
|---|---|---|---|
| dynamic zmm share | 47.6% | **1.5%** | **65.3%** |
| dynamic scalar/GPR | 52.4% | **98.5%** | 33.0% |
| static instrs / static touching zmm | 363 / 78 (21%) | 3535 / 572 (16%) | — |
| dominant vector ops | `vpaddw` 32.0, `vpsubw` 10.7, `vpmovsxbw` 2.8 | `vpaddw`, `vpsubw` (buried) | `vmovdqa64` 17.3, `vpmovsxbw` 17.1, `vpsubw` 12.9, `vpaddw` 4.6 |
| dominant scalar ops | `inc` 19.6, `add` 15.2 | `inc` 55.3, `add` 13.0, `lea` 7.3 | `cmp` 8.1, `mov` 5.2, `add` 4.7 |
| loop shape | block-outer, feature-inner, **row address `feature*ACC+off` re-derived per block** | same | tile-outer, **all four index lists applied into live registers**, one store; count **specialized at compile time** via template `UpdateOperation` parameters |
| accumulator writes per folded ply | 2 × 768 × 2 × 2 B = 6144 B | | 1 × 1024 × 2 × 2 B = 4096 B (fused) |

The mechanism of the gap is now unambiguous and it is **not SIMD width** — both
use 512-bit lanes for the actual arithmetic. It is that our vector work is
**diluted by scalar address arithmetic** (≈2 scalar address ops per vector op)
because the row-base derivation sits inside the accumulator-block loop, and
because nothing is specialized on the delta count. SF's 65% zmm is what our
kernels would approach with the addresses hoisted.

### 5.5 Frequency vs unit cost — when each path fires

Ours (callgrind, deterministic, 1,000,750 nodes / 416,706 evals) — §1.3 table.
Headline: **`refresh_threat` 0.0346/node at 1278 c/call = 44 c/n.**

SF's, from the code: their threat accumulator refreshes whenever
`find_last_usable_accumulator` finds no usable ancestor, and the trigger it
consults is **HalfKA's `requires_refresh`** (any of 32 king buckets changing) —
strictly *more often* than our mirror-crossing test, and with **no Finny cache
to absorb it** (up to 128 i8 columns rebuilt every time,
`nnue_accumulator.cpp:538-539`). Measured cost of that path:
`update_threats_accumulator_full<1024u>` = 0.49% of cycles = 16 c/n. They fire
more often and still pay less, because their per-refresh column loop is the
same well-tiled `apply_combined` machinery rather than a separate scalar
`add_column` walk.

---

## 6. OPTIONS RANKED

Slope convention used throughout: at fixed cycles, removing a fraction *f* of
cycles multiplies NPS by `1/(1−f)`. All NPS estimates derive from the measured
shares in §1.3 / §5.3. House rule from prior studies: ~1 Elo per 1% NPS at
10+0.1 (opt-1's measured slope was 2.1 Elo/1%, so these are conservative).

### Option S — SKP1 skip lane, made branchless (NOT a threat option; do it first)

`sota_head.rs:364-374`. Delete the `if a != 0` guard so the loop is a
straight-line multiply-accumulate.

- **Estimated gain: +14% of cycles → ≈ +16% NPS.** From §1.4: mispredicts are
  ~72% of the loop's 19.3% of cycles.
- **Effort: minimal** (one branch removed; hours including gates).
- **Correctness risk: very low, and bit-identity is *provable*.** For `a == 0`,
  `f32::from(0u8) * w == ±0.0` and `acc + ±0.0 == acc` exactly for every finite
  `acc`; the only representational difference is `+0.0` vs `−0.0` when `acc` is
  zero, which is numerically equal and is then consumed by
  `(l3 * network_scale) as i32`. Summation order is **unchanged**. No NaN/Inf
  path exists (weights come from a validated section).
- **Validation:** the 800-FEN eval hash must be **unchanged** (§6.1) and the
  depth-6 bench trailer must reproduce **701605** bit-for-bit.

### Option S2 — SKP1 lane vectorized (bucket-major relayout + zmm reduction)

Transpose the 24 KB SKP1 table to bucket-major **at load time** (`sota_loader.rs`
`parse_skip_section`) so the per-bucket lane is 768 contiguous f32, then do a
zmm u8→f32 widen + FMA reduction.

- **Estimated gain: up to +19.3% of cycles → ≈ +24% NPS** (the full ablation
  Δ; measured A/B on this exact workload was **+23.3%**).
- Sub-step S2a — relayout only, keeping the scalar sequential adds — is
  **bit-identical** (same order, same values, just contiguous loads) and worth
  perhaps +1–2% on its own; take it as the enabling commit.
- **Effort: low-medium** (1 day). The transpose is 6144 elements at load.
- **Correctness risk: medium — NOT bit-identical.** A tree reduction changes
  f32 summation order, so the 800-FEN hash *will* change. The gate must
  therefore be **integer-cp identity** on a large FEN battery plus a bounded
  logit delta, not byte identity. Retain the scalar path as the reference and
  add a scalar-vs-SIMD parity test (the existing §3.4.6 discipline).
- Honest uncertainty: I have not verified that the cp values survive
  reordering on all 800 FENs — that measurement is the gate, and if more than a
  handful of positions move by 1 cp the change must go through SPRT rather than
  claim neutrality.

### Option F — Hoist the fold kernels' row-address arithmetic (bit-identical)

`sota_simd.rs` (both avx2 and avx512, both `i8_*` and `i16_*` variants):
precompute `add` / `sub` row base pointers **once** into a small stack array
before the `off` loop, then index them inside. Optionally add compile-time
specialization for the 1-add/1-sub and 2-add/2-sub cases (SF's technique).

- **Estimated gain:** the three fold/refresh kernels are 6.97 + 3.99 + 4.71 =
  **15.67% of cycles**, of which the measured scalar/GPR fraction is 52% (i8)
  and ~98% (i16). Not all of that is removable — 24 blocks of loop control is
  irreducible, and part of the `inc` pileup is skid. Taking the recoverable
  portion as the redundant per-block index derivation (≈½ of the scalar share):
  **≈ 5–7% of cycles → +5–8% NPS.** Threat channel gets ~⅓ of that, HalfKA ~⅔.
- **Effort: low** (1 day, mechanical, 4 kernels).
- **Correctness risk: low, bit-identity provable.** Identical loads in
  identical order, identical wrapping-i16 arithmetic. Nothing about the
  operation order changes.
- **Validation:** 800-FEN eval hash unchanged; depth-6 bench 701605; the
  `threats_replay_*` gates; `cargo test --release --features sota -- --ignored
  threats_replay_large`.
- This is the highest-confidence *threat-relevant* item in the study, and it is
  the one place where §5.4 gives a named mechanism and a reference target
  (65% zmm).

### Option T1 — The designed 3-primitive driver (`on_move`/`on_change`/`on_mutate`)

§3, plus the two SF suppression tricks (`noRaysContaining`; type-partitioned
attacker sets so the per-bit `match` + `contains` re-tests vanish).

- **Estimated gain:** the addressable block is generation + resolution = 2.69 +
  4.61 = **7.30% of cycles** (998 − 460 fold − 44 refresh − 12 push). Closing
  the *whole* generation gap to SF's per-node cost would recover only 178 − 162
  = 16 c/n; the real prize is that fewer *events* shrink the **fold** too
  (fold is linear in events). If over-generation + spurious ray pairs are ~15%
  of events (SF's `noRaysContaining` fires on every sliding move; king-victim
  and pawn→B/Q edges are generated on every touch), the fold and the
  resolution both shrink ~15%: 0.15 × (460 + 304) ≈ 115 c/n ≈ **1.7% of cycles
  → +2% NPS**, plus perhaps another 1% from the cheaper generation itself.
  **Total ≈ +2–4% NPS.**
- **Effort: HIGH (2–4 days).** It requires either wiring generation into the
  `MakeMoveHooks` seam so occupancy is naturally correct mid-move, or threading
  exact per-edit occupancy through the replay. It touches the telescoping
  correctness argument (`sota_threats.rs:59-74`).
- **Correctness risk: the highest in this list.** Changing which events are
  emitted changes what must cancel. It IS still bit-identical *at the
  accumulator* if done right (wrapping adds cancel arithmetically), and the
  replay gate is a strong oracle — but a subtle miss produces a wrong eval on
  rare geometries that the 60-game replay tier may not hit. Run the `_large`
  4000-game tier plus a fresh 800-FEN hash.
- **Verdict: do not do this yet.** §5.3 measured our generation at 1.10× SF's
  and our fold at 1.13×. The upside is 2–4% for the riskiest change available.
  It is a *later* item, and its value rises only after S/S2/F land (at which
  point the threat channel's *share* rises).

### Option T2 — Cheap subset of T1: victim pre-masking only

Change `record_outgoing` / `record_square_touch`'s target masks from `& occ` to
the type-correct masks the from-scratch path already builds (`& (P|N|R)` for
pawn diagonals, `& !kings` for sliders/knights). No driver restructure, no
occupancy threading.

- **Estimated gain: ≈ +1–2% NPS.** It removes events that are generated,
  buffered, resolved **twice**, and then dropped by `threat_index` — every
  king-victim edge on every touch, and pawn→B/Q.
- **Effort: low (half a day).** It is a mask change in 2–3 places.
- **Correctness risk: low-medium.** Bit-identical *provided* the mask exactly
  matches what `threat_index` would drop. The one asymmetric case the module
  docs already flag (`sota_threats.rs:22-27`) is pawn-push onto a knight/rook,
  which is guarded separately and must stay guarded. The replay gate catches
  any slip.
- **This is the good first threat-channel move** — most of T1's fold-shrinking
  benefit at a fraction of the risk.

### Option C — Cross-ply span merge / 2-ply fused fold (SF's `double_inc_update`)

Merge a multi-ply span's add/sub multisets before folding.

- **Estimated gain: ≤ +1%.** §2.2 measured 1.915 i8 fold calls/eval ≈ exactly
  one folded ply per eval, both perspectives. Multi-ply spans are rare in the
  current fully-lazy schedule, so there is almost nothing to merge. SF spends
  1.78% here because their schedule produces more 2-ply opportunities.
- **Effort: medium. Risk: low. Verdict: not worth it now** — measure span-length
  distribution before revisiting.

### Option D — Threat-PSQ1 row loads inside `resolve_into`

31.5% of `at_full` sits on `sota_threats.rs:993-1002`, whose payload is
`psqt_add`/`psqt_sub`: a random 32-byte read from the 1.94 MB threat-PSQ1 table
per event.

- **Estimated gain: ≈ +1%** if the 8 i32 buckets were narrowed to i16 (halving
  the row to 16 B, one line per two rows) — but that is a **net-format change**,
  not an inference change.
- **Verdict: park it as a rider on the next CENN rev**, alongside the
  long-standing quant-unification item (study 6 item 5).

### Options explicitly rejected (with the measurement that kills each)

| option | why rejected |
|---|---|
| Low-rank factorization of the 46 MB threat table | study 3 §3: rank-256 captures 50.9% of energy, residual 11× the i8 quant noise floor. Dead. |
| Software prefetch of weight rows (SF's technique) | implemented and **measured a loss** (−2.5% NPS full-row) — `sota_threats.rs:861-866`. |
| Compress the threat table / reduce byte volume as the primary lever | §5.2: our L1d miss *rate* (15.4%) and dTLB misses/node (1.43) are both **better** than SF18's. We are instruction-bound, not memory-bound. |
| Making `refresh_threat` faster / SIMD-ifying `add_column` | 0.67% of cycles at 0.0346 calls/node. Ceiling <0.5% NPS. |
| Per-event add/sub cancellation scan | already tried: cancels 0.48% of events while taxing 100% — `sota_threats.rs:956-963`. |
| SIMD event emission (SF's ICL pattern) | requires a packed 4-byte event record; our `ThreatDelta` is a 5-field struct. Prerequisite is T1. Gain ≈ +1–2% at MED-HIGH effort on top of T1. Defer. |

---

## 7. RECOMMENDED SEQUENCE

The ordering is dictated by (gain × confidence) ÷ (effort × risk), and by the
fact that S and F are provably bit-identical while everything in the threat
channel is only *arguably* so.

**Step 1 — Option S: branchless SKP1 lane.** Expected **+16% NPS**, hours of
work, bit-identity provable by argument.
*Acceptance:* (a) 800-FEN eval hash **exactly**
`0615d75e5c9487b6f6c66255d3c1d67789f46471d4c8eaec41580e3c070c92eb`;
(b) `bash tools/check.sh` → depth-6 bench **701605**; (c) `perf stat` shows
branch-misses/node fall from 77.9 toward ~30; (d) paired interleaved NPS ≥
+12%.
*Abort if:* the eval hash moves at all.

**Step 2 — Option F: hoist fold-kernel row addresses (+ optional count
specialization).** Expected **+5–8% NPS**, ~1 day, bit-identical, benefits both
channels.
*Acceptance:* (a) 800-FEN hash unchanged; (b) depth-6 bench 701605;
(c) `cargo test --release --features sota -- --ignored threats_replay_large`
green; (d) re-annotate the kernels — `i16_deltas_into_inner`'s dynamic zmm share
must rise from 1.5% toward SF18's 65%, and static zmm-touching instructions from
16% toward ~40%; (e) paired NPS ≥ +4%.
*Abort if:* zmm share does not move (means the compiler was already hoisting and
the `inc` samples were pure skid — a real possibility I could not fully exclude).

**Step 3 — Option T2: victim pre-masking in the observer.** Expected
**+1–2% NPS**, half a day, low risk. First genuine threat-channel item.
*Acceptance:* (a) 800-FEN hash unchanged; (b) both replay tiers green;
(c) event count per fold falls (instrument via a temporary counter in a scratch
build, or infer from the `i8_deltas_into_inner` cycles/call falling from 577);
(d) depth-6 bench 701605.

**Step 4 — Option S2: vectorize the SKP1 lane.** Expected a further **+6–8%**
on top of Step 1 (the two are sub-additive; the pair caps at the ablation's
+23%). Take S2a (bucket-major relayout, bit-identical) as a separate commit
first, then the zmm reduction.
*Acceptance:* S2a — 800-FEN hash unchanged. S2 proper — **integer-cp identity**
on all 800 FENs (not byte identity of the hash pipeline: compare the cp columns
directly), a scalar-vs-SIMD parity unit test, and, if any position moves by
≥1 cp, a full 10+0.1 SPRT rather than a neutrality claim.

**Step 5 — re-measure before going further.** After Steps 1–4 the engine should
be near **~4400–4700 cycles/node** (from 6599), i.e. **1.33–1.42× SF18** rather
than 1.99×. At that point the threat channel's *share* rises to ~19–21% purely
by deflation of everything else, and the T1 decision should be re-taken on
fresh numbers — including a fresh middlegame-heavy profile, since §1.5 shows the
threat share is +3 points there.

**Do not start with T1.** The commission's instinct was that the threat channel
was the problem; the differential says our threat channel is already within
1.23× of Stockfish 18's, while a single un-vectorized scalar loop in the head
costs more than the entire threat channel and can be largely fixed in an
afternoon.

### 7.1 The D24 / eval-identity validation instrument

Every option above is gated on eval identity, and the instrument exists:

```bash
# eval-identity oracle: 800 integer centipawn values, one per line
./binary nnue-eval /tmp/d47v-sb100-nocpol.cenn \
  < tools/nnue/calib/scale-check-800.fens | sha256sum
```

Verified this session:
- skip ON, `engines/bin/archon-d47v-fast`:
  `0615d75e5c9487b6f6c66255d3c1d67789f46471d4c8eaec41580e3c070c92eb`
- skip ON, rebuilt-from-source `tg-768v3`: **identical hash** (the oracle is
  revision-stable, so a hash match is a real proof of eval identity)
- skip OFF (ablated net):
  `67d3a9afcc489ad506dba7accef2fd81182ef92cb40c9b5d9901c6da98b0e569`
  (the oracle discriminates)

Plus the standard gates: `bash tools/check.sh` for the depth-6 bench trailer
(**701605**), `cargo test` for the fast tier, and
`cargo test --release --features sota -- --ignored threats_replay_large` for the
4000-game incremental-vs-refresh oracle. Any change that touches the observer's
emitted event set must run the `_large` tier, not just the default one.

---

## 8. Uncertainties and threats to validity

Stated plainly, because several of these bound the recommendations.

1. **The box was not quiet.** A GPU trainer held 93% GPU throughout. All shares
   and per-node cycle counts are robust to this; the absolute NPS figures
   (~800k) are depressed and must not be compared to numbers from idle-box
   sessions.
2. **`inc`-pileup skid.** The dynamic 55% `inc` share in
   `i16_deltas_into_inner` is partly sampling skid from the preceding loads. I
   corroborated with *static* instruction counts (16% zmm-touching), which is
   skid-immune, and the two agree in direction — but the split between
   "redundant address math" and "load stall attributed to the next instruction"
   is not measured exactly. Option F's estimate (+5–8%) inherits that
   uncertainty; its abort criterion is written to catch the bad case.
3. **SF's shared scalar-resolve region.** The 80/20 threats/HalfKA split of
   SF's 6.32% in-`evaluate` scalar region is an assumption, not a measurement.
   Only the ALL-FEATURE-UPDATE row (1399 c/n) is assumption-free.
4. **Node definitions may differ** between the two engines' counters. The
   0.06% agreement in totals is a consequence of both honoring `go nodes`, not
   proof that a "node" means the same thing.
5. **SF18 binary vs SF dev sources.** The profiled binary is Stockfish 18
   release (bimodal, two nets); the read sources are a 2026-07-16 dev ref where
   bimodality was removed. Schedule/kernel facts were confirmed against the
   binary's own symbols and disassembly, so the mismatch affects only the
   bimodality discussion, which I segmented separately (4.24%).
6. **Position mix.** wl50 is endgame-heavy. Threat shares on middlegame-heavy
   positions are ~3 points higher (§1.5), so §5.3's threat rows are the
   *favorable* end for us; a game-config profile would show our threat channel
   somewhat worse relative to SF18 than 1.23×.
7. **The S2 cp-identity claim is unverified.** I priced the vectorized skip lane
   from the ablation but did not measure whether a reordered f32 reduction
   preserves the integer cp on all 800 FENs. That measurement is the gate, and
   the recommendation is written so that a failure routes to SPRT rather than to
   a false neutrality claim.
8. **Nothing here was Elo-tested.** Every Elo figure is the house
   1 Elo/1% NPS conversion applied to a measured cycle share. The prior
   measured slope was 2.1 Elo/1%, so these are conservative — but they are
   conversions, not results.

## 9. Artifacts

Scratch (session-local, not committed):
`p768.data` / `p1280.data` (d15 depth-mode flat profiles, matched builds),
`d-ours.data` / `d-sf.data` (the fixed-node SF18-matched pair),
`pprof.data` (line-tables build for source-line annotation),
`p768-def.data` (middlegame-heavy default-fens profile),
`cg-ours.out` (callgrind call counts), `sf-ann-eval.txt`
(SF18 `Network<1024u>::evaluate` annotation), `opc.py` (opcode/register-width
histogram helper), `wl50.fens`, `d47v-noskip.cenn` (SKP1-truncated ablation net).

Regenerate the FEN subset with
`awk 'NR%16==1' tools/nnue/calib/scale-check-800.fens`.
