# ARCHON perf study #2 — broad ground truth, SF18 differential, assembly, ranked options

Date: 2026-07-26 (evening). Commissioned by the user as the sequel to
`docs/research/threat-channel-perf-study.md` ("study 1"), with a broad mandate:
profile the post-optimization binary from scratch, rebuild the SF18
differential, compare assembly, and hunt for big gaps **anywhere** in the
engine.

**Study only — no engine source was modified.** Binaries were built (allowed).
Four *prototype* patches were written and measured in a throwaway `git archive`
copy of `HEAD` under the session scratchpad; the repository working tree was
never touched (verified: `git status` clean at start and end). Every number
below marked **[proto]** comes from those scratch builds and is a measurement,
not a landed change.

---

## 0. Headline

**(1) The optimization program worked, and it was almost entirely SKP1.**
Two order-opposed paired measurements on the fixed-node wl50 workload: the
study-1 binary (`engines/bin/archon-d47v-fast`, `bf3ec90`) at **7120 c/n** vs
the new baseline (`engines/bin/archon-d47v-S2`, `b1b9c6f`) at **5006 c/n**
(old first) and **6842 vs 5136 c/n** (new first) — **−25 to −30 % cycles =
+33 to +43 % NPS**. (The commission's "+29.6 % contended" is a different,
game-config measurement; both are real, this one is bigger because wl50 is
eval-heavy.) Symbol-level attribution says **87 % of that gain is options
S + S2** (the SKP1 lane: 1621 → 136 c/n). Option F (fold address hoist)
delivered ~2 % of the gain and **failed its own abort criterion**; option T2
delivered ~3 %.

**(2) The cycle map has inverted. The shell is now the biggest block.**

| block | study 1 (old binary) | now |
|---|---:|---:|
| head math | **36.6 %** | 17.4 % |
| search + movegen shell | 28.1 % | **37.8 %** |
| HalfKA accumulator | 16.1 % | 21.6 % |
| threat channel | 15.1 % | 20.6 % |

**(3) We are 1.41× SF18 on cycles/node (was 1.99×), and the three biggest
remaining gaps are none of study 1's three.** They are: the **HalfKA Finny
refresh** (621 vs 248 c/n, 2.5×), **MovePicker construction** (215 c/n, against
a Stockfish constructor that zeroes nothing and copies nothing — we zero 2 KB
and then copy ~2.1 KB per node for a buffer whose unwritten slots are never
read), and **`make_move`'s StateInfo plumbing** (a 224-byte record built on the
stack and then copied again into a `Vec`, 5.5 % of all cycles, vs SF's 64-byte
prefix `memcpy` into a caller-owned object). All three are pure bookkeeping —
no chess, no arithmetic, no learned term.

**(4) Prototypes, measured, bit-identical.** A scratch build carrying (a) an
SF-style bitboard set-diff in the Finny refresh, (b) a wide register tile in
the three fold kernels, and (c) an uninitialised MovePicker buffer produces
the **identical 800-FEN eval hash**, the **identical 15,001,555-node** bench
total, a green `cargo test` (660 lib + all integration + `threats_replay_large`),
**−24.5 % instructions/node** and **≈ −9 to −10 % cycles**. Single best next
move: **the Finny refresh diff (≈ +6–8 % NPS on its own, one file, bit-identical).**

---

## 1. GROUND TRUTH — provenance, harness, contamination

### 1.1 Provenance

Source: `main` @ `172c7dd` (working tree clean). The designated baseline binary
was **reproduced byte-for-byte** from that revision, so every source citation
below is exact:

```bash
CARGO_TARGET_DIR=$S/tg-768v3 RUSTFLAGS="-C target-cpu=x86-64-v3" \
  cargo build --release --features sota,arch-d47v
sha256  46eab56bd0d22afa69f89618984aa754fa6314c268db3e187035ac7796b22a3d
        == engines/bin/archon-d47v-S2                       (identical)
```

Line-tables build for source attribution:

```bash
CARGO_TARGET_DIR=$S/tg-768prof \
RUSTFLAGS="-C target-cpu=x86-64-v3 -C force-frame-pointers=yes" \
  cargo build --profile profiling --features sota,arch-d47v
```

Net: `/tmp/d47v-sb100-nocpol.cenn` (84,158,584 B). Reference:
`engines/bin/stockfish18` (unstripped, 1235 symbols). SF *source* citations come
from the reference clone's object store at `origin/master` =
`ebcea3efe9c1b8748e080111c727c33c544d7e06` (tag
`stockfish-dev-20260716-ebcea3ef`, **243 commits after `sf_18`**) via
`git show origin/master:<path>`; where master and `sf_18` differ the difference
is stated. **No Stockfish code was copied — mechanisms are described in prose.**

### 1.2 Commands (study-1 discipline, §1.1 continuity)

```bash
awk 'NR%16==1' tools/nnue/calib/scale-check-800.fens > $S/wl50.fens   # 50 FENs

# matched fixed-node workload, both engines, single thread, 16 MB TT, pinned
taskset -c 4 ./engines/bin/archon-d47v-S2 bench 16 1 300000 $S/wl50.fens nodes /tmp/d47v-sb100-nocpol.cenn
taskset -c 4 ./engines/bin/stockfish18     bench 16 1 300000 $S/wl50.fens nodes
# node totals 15,001,555 vs 15,010,238 — 0.06 % apart, so c/n is comparable

taskset -c 4 perf record -F 3999 -e cycles:u -o $S/d-ours.data -- <as above>
taskset -c 4 perf stat  -e cycles:u,instructions:u,branches:u,branch-misses:u,\
  L1-dcache-loads:u,L1-dcache-load-misses:u -x, <as above>
# Zen 4 memory events (LLC-* is <not supported> on this kernel):
#   ls_any_fills_from_sys.{dram_io_all,local_ccx,local_l2},
#   l2_cache_req_stat.{dc_access_in_l2,ic_dc_miss_in_l2},
#   ic_tag_hit_miss.instruction_cache_miss
nice -n 10 taskset -c 0 valgrind --tool=callgrind --cache-sim=no --branch-sim=no \
  $S/tg-768v3/release/chessengine bench 16 1 20000 $S/wl50.fens nodes <net>
```

### 1.3 Contamination — stated plainly

The box was **not** quiet, and was noisier than in study 1:

- an RTX 4060 Ti trainer at 96 % GPU / ~176 % CPU throughout;
- a **live fastchess equivalence match** (`archon-d47v-S2a` vs `-S2`, 4 engine
  processes, unpinned) for most of the session;
- load average 4.2 – 7.0 (study 1 saw 1.5 – 2.3).

Consequences, honestly:

1. **Absolute NPS is meaningless here.** The same binary measured 933 k – 1134 k
   NPS on the same workload within one hour.
2. **Even `cycles:u` drifts ±6 %** run to run, because the co-tenants evict L2/L3.
   SF18's cycles/node measured 3320 in study 1 and 3327–3374 today (+1.5 %); our
   study-1 binary measured 6599 then and 7120 now (+7.9 %) — the *eval-heavy*
   binary is more contention-sensitive, which is itself a finding.
3. **Every comparison below is either interleaved A/B on the same box state, a
   share of a single profile, or a deterministic counter** (`instructions:u`,
   callgrind `Ir`, node totals). Nothing rests on an absolute.
4. Where I quote a prototype's effect I give **instructions/node** (perfectly
   deterministic, reproduced to the last digit across runs) alongside cycles.

`perf stat` multiplexed at 71–83 % when 6–7 events were requested; the reported
values are perf's scaled estimates.

### 1.4 The eval-identity instrument (D24 gate) — still valid, one caveat

```bash
./binary nnue-eval /tmp/d47v-sb100-nocpol.cenn < tools/nnue/calib/scale-check-800.fens | sha256sum
```

The current binary hashes to
**`0615d75e5c9487b6f6c66255d3c1d67789f46471d4c8eaec41580e3c070c92eb`** — the
**same** value study 1 pinned for the *pre-optimization* binary. So S, F, T2 and
S2 are all integer-cp identical on the 800-FEN battery, exactly as their landing
gates claimed. The oracle is current, not stale.

**Caveat worth recording:** the fixed-node bench total moved from
**15,001,572** (study-1 binary) to **15,001,555** (current) — 17 nodes in 15 M,
1.1 × 10⁻⁶. So S2's f32 reduction reorder *does* change the eval on at least one
position outside the 800-FEN battery. The battery is a strong gate, not a proof;
say "battery-identical", not "bit-identical", for S2.

---

## 2. THE FRESH CYCLE MAP

Three profiles were recorded within minutes of each other under equal load
(totals from `perf script -F period`, summed):

| profile | total cycles | nodes | **cycles/node** |
|---|---:|---:|---:|
| study-1 binary `archon-d47v-fast` (`bf3ec90`) | 102.647 G | 15,001,572 | **6842** |
| current baseline `archon-d47v-S2` (`b1b9c6f`) | 77.044 G | 15,001,555 | **5136** |
| `stockfish18` | 53.801 G | 15,010,238 | **3584** |

Note the two ARCHON generations were also measured back-to-back with
`perf stat` in the *opposite* order (old first): 106.813 G vs 75.101 G cycles =
7120 vs 5006 c/n, NPS 715,928 vs 1,021,556. Because the two orderings bracket
the same comparison from both sides, the honest statement of the optimization
program's effect is **−25 % to −30 % cycles / +33 % to +43 % NPS**, not a point
estimate.

The tighter back-to-back `perf stat` pairs against SF18 (used for the headline
ratios):

| pair | ARCHON c/n | SF18 c/n | ratio |
|---|---:|---:|---:|
| run 1 | 4593.1 | 3327.2 | 1.38× |
| run 2 | 4832.9 | 3374.1 | 1.43× |
| **median** | **4713** | **3351** | **1.41×** |

Study 1's figure was 1.99×. **The gap closed by 29 %.**

### 2.1 Whole-engine counters, ours vs SF18 (same session, interleaved)

| metric | ARCHON (S2) | SF18 | ours ÷ SF |
|---|---:|---:|---:|
| cycles/node | 4713 | 3351 | **1.41×** |
| instructions/node | **9273.6** | **4332.4** | 2.14× |
| IPC | 1.92 – 2.02 | 1.28 – 1.30 | 1.53× |
| branches/node | 1296.9 | 394.8 | 3.28× |
| **branch-misses/node** | **23.1** | **26.6** | **0.87×** |
| L1d loads/node | 4020 | 2369 | 1.70× |
| L1d miss rate | 14.75 % | 17.92 % | 0.82× |
| L1←L2 fills/node | 575 | 389 | 1.48× |
| L2 misses/node | 35.6 | 55.3 | 0.64× |
| L3 (local-CCX) fills/node | 33.3 | 51.0 | 0.65× |
| **DRAM fills/node** | **6.56** | **8.70** | **0.75×** |
| icache misses/node | 181.9 | 86.2 | 2.11× |

Read this carefully, because it kills two tempting stories:

- **We are not memory-bound relative to SF18.** Fewer DRAM fills, fewer L3
  fills, fewer L2 misses, a lower L1d miss rate. SF18 touches *more* memory
  than we do and still runs 1.41× faster. Any "compress the weight tables"
  framing is dead — study 1 said this and it is still true, with better events.
- **We no longer mispredict more than SF18.** Study 1 measured 77.9
  mispredicts/node (the SKP1 lane was 66 % of them); we are now at **23.1 vs
  SF's 26.6** — we are *better*. Branch prediction is a closed front.

What remains is **instruction volume**: 2.14× SF's, at an IPC 1.53× theirs.
That is the whole story of the residual gap, and §5 shows most of the excess
is bookkeeping (zeroing, copying, list-diffing), not arithmetic.

Instruction-cache misses at 2.11× look alarming but are not a differential:
per *instruction* the rates are 1/51 (ours) vs 1/50 (SF's). We simply execute
more instructions. It is, however, a hint that our hottest functions are large
(`next_move` is 2591 static instructions).

### 2.2 Flat profile, current baseline (shares × the 4713 c/n median)

| # | symbol | cycles % | c/node | block |
|---|---|---:|---:|---|
| 1 | `MovePicker::next_move` | **10.39** | 490 | shell |
| 2 | `avx512::i8_deltas_into_inner` | **10.00** | 471 | threat fold |
| 3 | `avx512::i16_deltas_inner` | 6.68 | 315 | **Finny refresh fold** |
| 4 | `Board::make_move_with::<TtPrefetch>` | 6.04 | 285 | shell |
| 5 | `HalfkaContext::finny_refresh_into` | 5.89 | 278 | **Finny refresh driver** |
| 6 | `Worker::search::<NonPv>` | 5.84 | 275 | shell |
| 7 | `ThreatContext::at_full` | 5.34 | 252 | threat resolve |
| 8 | `avx512::l1_dot_inner` | 5.29 | 249 | head |
| 9 | `avx512::i16_deltas_into_inner` | 5.09 | 240 | HalfKA fold |
| 10 | `avx512::activate_inner` | 5.08 | 239 | head (FT merge) |
| 11 | `MovePicker::new` | 3.58 | 169 | shell |
| 12 | `HalfkaContext::current_full` | 3.32 | 156 | HalfKA driver |
| 13 | `sota_threats::record_square_touch` | 2.94 | 139 | threat gen |
| 14 | **`libc` `memcpy`/`memset`** | **2.93** | **138** | **bookkeeping** |
| 15 | `avx512::skip_dot_inner` | 1.81 | 85 | head (SKP1) |
| 16 | `SotaContext::evaluate` | 1.81 | 85 | eval driver |
| 17 | `avx512::l2_tail_inner` | 1.26 | 59 | head |
| 18 | `Histories::update_continuation_histories` | 1.06 | 50 | shell |
| 19 | `Worker::do_move` | 1.04 | 49 | shell |
| 20 | `Histories::correction_value` | 0.95 | 45 | shell |
| 21 | `AttackTables::attacks` | 0.93 | 44 | movegen |
| 22 | `avx512::find_nnz_vbmi2_inner` | 0.85 | 40 | head |
| 23 | `SotaHead::forward_into` | 0.84 | 40 | head driver |
| 24 | `Board::see_ge` | 0.79 | 37 | shell |
| 25 | `Board::legal` | 0.73 | 34 | movegen |
| … | `sota_features::refresh_threat` | 0.69 | 33 | threat refresh |
| … | `Board::attackers_to` | 0.66 | 31 | movegen |
| … | `Board::set_check_info` | 0.60 | 28 | make/unmake |
| … | `TranspositionTable::probe` | 0.54 | 25 | TT |
| … | `Board::unmake_move` | 0.50 | 24 | make/unmake |
| … | `sota_threats::ripple` | 0.48 | 23 | threat gen |
| … | loader / `init_magics` (startup) | ~0.80 | 38 | excluded below |

### 2.3 Block totals — and how they moved

| block | study 1 (old) | **now** | now c/n |
|---|---:|---:|---:|
| head math (incl. SKP1) | 36.56 % | **17.4 %** | 822 |
| search + movegen shell | 28.12 % | **37.8 %** | 1783 |
| HalfKA accumulator | 16.14 % | **21.6 %** | 1017 |
| threat channel | 15.12 % | **20.6 %** | 970 |
| startup / loader | 1.33 % | 0.8 % | 38 |

The threat channel's *share* rose from 15.1 % to 20.6 % exactly as study 1 §7
step 5 predicted ("~19–21 % purely by deflation of everything else"). Its
absolute cost barely moved.

### 2.4 What the four landed options actually bought (matched profiles, c/n)

| symbol | old (`bf3ec90`) | new (`b1b9c6f`) | Δ |
|---|---:|---:|---:|
| `SotaHead::forward_into` | **1621** | 43 | **−1578** |
| `avx512::skip_dot_inner` | — | 93 | +93 |
| → **SKP1 net (options S + S2)** | | | **−1485** |
| `ThreatContext::at_full` (option T2) | 328 | 274 | −54 |
| `i16_deltas_into_inner` (option F) | 293 | 261 | −32 |
| `i8_deltas_into_inner` (option F) | 517 | 514 | −3 |
| `MovePicker::new` | 196 | 184 | −12 |
| `make_move_with` | 340 | 310 | −30 |
| `next_move` | 558 | 534 | −24 |
| `finny_refresh_into` | 315 | 302 | −13 |
| `i16_deltas_inner` | 351 | 343 | −8 |
| `search::<NonPv>` | 289 | 300 | +11 |
| `l1_dot_inner` | 255 | 272 | +17 |
| `activate_inner` | 256 | 261 | +5 |
| `current_full` | 172 | 171 | −1 |
| **total** | **6842** | **5136** | **−1706** |

Unchanged-by-construction symbols move ±5 %, which is the noise floor. On that
scale:

- **S + S2 = −1485 c/n = 87 % of the whole gain.** Study 1's ranking was right.
- **T2 = −54 c/n (3 %).** Real, small, as priced (+1–2 % NPS).
- **F = −35 c/n (2 %), and it failed its own abort test.** Study 1 §7 step 2
  wrote: *"Abort if zmm share does not move"*. It did not: `i16_deltas_into_inner`'s
  **dynamic** zmm share is **1.45 %** today versus 1.5 % before (§3.2). What
  moved is the *static* share (16 % → 31.7 %) — the hoisted code exists, it just
  is not where the cycles are. The `inc`/`dec` pile-up study 1 suspected of
  being skid **was** skid, from the cache-missing weight-row load. Recording
  this so the next study does not re-derive it.

---

## 3. ASSEMBLY SIDE BY SIDE

Method: `perf script -F ip,sym,symoff` histogram × `objdump -d -C`, so both
dynamic (sample-weighted) and static (skid-immune) mixes are available.
Helpers: `$S/opc2.py`, `$S/ann.py`.

### 3.1 The reference shape — SF18 `Network<1024u>::evaluate` (35.03 % of SF's cycles, 1174 c/n)

| | value |
|---|---|
| dynamic register width | **zmm 67.5 %**, gpr 30.9 %, ymm 0.7 %, xmm 0.8 % |
| static register width | zmm 32.7 %, gpr 63.2 % |
| top opcodes | `vpmovsxbw` 19.6, `vmovdqa64` 17.2, `vpsubw` 13.9, `cmp` 8.3, `vpdpbusd` 6.7, `mov` 5.0, `add` 4.6, `vpaddw` 4.5 |
| hot regions | +0x1000 (11.8 %) and +0x600 (11.8 %) — i8→i16 threat widen; +0x200 (10.7 %) — i16 HalfKA fold; +0x1a00 (7.7 %) — `vptestmd`/`vpcompressd`/`popcnt` nnz; +0x1200 (5.8 %) — `vpminsw` FT activation; +0x1c00 (4.4 %) — `vpdpbusd` L1 matmul |

**Their tile is 16 zmm live.** `SIMDTiling` (`simd.h:477-528`) computes
`NumRegs = 16`, `TileHeight = 512` i16 lanes for L1 = 1024 on AVX-512 — the
whole 1024-lane accumulator in **2** tile iterations, with all four index lists
(psqAdded/psqRemoved/thrAdded/thrRemoved) applied into those 16 live registers
before a single store (`nnue_accumulator.cpp:183-244`).

Correction to study 1 §2.4/§5.4: **SF no longer compile-time-specializes the
fold on the add/sub count.** The `UpdateOperation` variadic machinery existed at
`sf_18` (`sf_18:src/nnue/simd.h:251-283`) and was **deleted** in `86beb0f5`
(2026-07-16) after the HalfKA/threat accumulator merge (`7c7fe322`). Current
master runs a plain runtime loop over `ssize()` for all four lists. The reading:
once the i8 threat columns dominated, count specialization stopped paying. Do
not chase it.

### 3.2 Our three fold kernels

| | `i8_deltas_into_inner` (threats) | `i16_deltas_into_inner` (HalfKA fold) | `i16_deltas_inner` (Finny) |
|---|---|---|---|
| cycles/node | **471** | 240 | **315** |
| dynamic zmm | 43.7 % | **1.45 %** | **2.97 %** |
| static zmm | 29.4 % | 31.7 % | 19.6 % |
| static instrs | 323 | 1273 | **2925** |
| top dynamic opcodes | `vpaddw` 32.7, `dec` 18.4, `add` 16.9, `vpsubw` 8.9, `cmp` 7.9 | `dec` 37.8, `cmp` 24.2, `add` 13.6, `lea` 6.3, `mov` 6.0 | **`inc` 72.7**, `add` 9.3, `lea` 4.8, `mov` 2.5, `shl` 2.2 |
| row-address math | **hoisted** (option F) | **hoisted** (option F) | **NOT hoisted** |
| accumulator registers live | **1** | **1** | **1** |

`i8_deltas_into_inner`'s hot loop, annotated (31.6 % of the symbol on one
instruction):

```
+450  mov    (%r9),%r11                    ; hoisted row base
+453  add    $0x8,%r9
+457  vpmovsxbw (%rsi,%r11,1),%zmm2        ; 32 i8 from a random row of a 46.6 MB table
+45e  vpaddw %zmm2,%zmm1,%zmm1             ; 31.59 %  <- one accumulator register
+464  dec    %r13                          ;  7.75 %
+467  jne    ...
```

`i16_deltas_inner` (Finny), where the hoist was never applied:

```
+e0   mov    (%rdx),%rcx
+e7   lea    (%rcx,%rcx,2),%rcx            ; ×3
+eb   shl    $0x9,%rcx                     ; ×512  => feature × 1536 (the i16 row stride)
+ef   vpaddw 0x0(%rbp,%rcx,1),%zmm0,%zmm0  ; random row of a 34.6 MB table
+f7   inc    %r9                           ; 46.86 %  <- skid from the load above
+fa   jne    ...
```

Both kernels also contain an 8-way-unrolled path (visible at +0x480 in the i8
kernel) that takes **< 0.1 % of samples**: it is guarded on ≥ 8 remaining list
entries, and our per-block list lengths are almost always shorter, so the
one-feature-at-a-time loop is what actually runs.

**Diagnosis, three parts:**

1. **Tile width, not SIMD width.** Both engines use 512-bit lanes for the
   arithmetic. SF holds **16** accumulator registers live per tile; we hold
   **1** (`const LANES: usize = 32` at `sota_simd.rs:949`, one zmm of i16). With
   one register the `vpaddw` chain is serialised *and* each feature costs a full
   loop iteration of scalar overhead *and* the feature's row is read 64 bytes at
   a time in 24 separate passes, defeating the L2 stride prefetcher.
2. **The scalar pile-up is skid, not work.** §2.4 proved this: option F removed
   the address arithmetic and bought 2 % of the program.
3. **Where the cycles really go is the load.** A DRAM-fill profile
   (`ls_any_fills_from_sys.dram_io_all`, `-c 200`) attributes **53.1 %** of all
   DRAM fills to these three kernels (`i16_deltas_inner` 22.5 %,
   `i8_deltas_into_inner` 21.0 %, `i16_deltas_into_inner` 9.6 %), and an L2-fill
   profile attributes 24.3 %. The weight tables are 46.6 MB (threat i8) + 34.6 MB
   (HalfKA i16) against a 32 MB L3 shared with the co-tenants.

Prototype §5.2 tests the tile-width fix directly.

### 3.3 `MovePicker::next_move` — 2591 static instructions, 99.9 % GPR

| region | % of symbol | content |
|---|---:|---|
| +0x200 | **26.7 %** | `mov` 56 / `and` 5 / `lea` 5 — object field reload + stage dispatch |
| +0x2600 | **23.1 %** | `mov` 34 / `add` 12 / `movswl` 8 — **the quiet scorer** |
| +0x0 | 15.5 % | `mov` 45 / `and` 17 / `movabs` 7 / `push` 6 — prologue, spills, tunable-param loads |
| +0x2a00 | 7.2 % | stage transition |
| +0x2800 | 6.3 % | good-capture SEE split |
| +0x3000 | 5.4 % | epilogue |

Annotated quiet scorer (`movepick.rs:598-611`):

```
+2603 movzbl 0x18(%r13,%r15,2),%eax   ; board mailbox: piece_on(m.from())
+2609 cmp    $0x2,%al                 ;  5.64 %   <- skid from the mailbox load
+264e movswl (%rax,%r8,2),%eax        ; main_plane[raw16]
+266d add    %eax,%ecx                ;  2.94 %   <- skid from the history load
+267a movswl (%rdx,%rax,1),%eax       ; pawn_plane[pci][to]
+267e lea    (%rax,%rcx,2),%eax       ;  2.08 %   <- skid
```

Seven history planes are read per quiet move, each a dependent, effectively
random load. `next_move` accounts for **6.0 % of the engine's DRAM fills**.
SF pays the same algorithmic cost (`movepick.cpp:228-250`: 2 main + 2 pawn +
5 continuation lookups, index 4 deliberately skipped) but issues **five
software prefetches of history/correction lines inside `do_move`**
(`position.cpp:1006-1010`) — one full move-make ahead of the consumer. We issue
**one** prefetch (the TT, `board.rs` `MakeMoveHooks`/`prefetch_key`).

The 42 % of the symbol sitting in the first 1 KB is not "the match statement is
slow" — it is a very large inlined function reloading its own fields and
re-materialising tunable parameters on every call (1.84 calls/node). SF's
`next_move` is 12.66 % of their cycles (424 c/n) against our 10.39 % (490 c/n):
**1.16×, not a headline gap** — but it is our single hottest symbol and the
prefetch idea below is cheap.

### 3.4 `Board::make_move_with` — 725 static instructions, 88 % of it is plumbing

Source-line attribution on the line-tables build, as **% of the whole program**:

| line | % of program | what |
|---|---:|---|
| `board.rs:1068` | **3.22** | `StateInfo { copied: prev.copied, ..StateInfo::EMPTY }` — building a 224-byte record on the stack |
| `alloc .../vec/mod.rs:1933` | **2.30** | `self.states.push(st)` — 7 × `vmovups ymm` load + 7 × store, i.e. **copying the same 224 bytes again** |
| everything else in the symbol | ~0.8 | the actual chess |

Disassembly confirms both: `imul $0xe0,%rdi,%rdi` fixes `size_of::<StateInfo>()`
at **0xe0 = 224 bytes**, and the +0xb0a region is a clean 7-register ymm
copy loop into the `Vec` slot.

Stockfish's shape (`position.cpp:815-831`): the child `StateInfo` is a
**caller-owned stack object** passed by reference from `Search::Worker::search`
(`search.cpp:735`), and `do_move` copies exactly
`offsetof(StateInfo, key)` = **64 bytes** of carried prefix into it —
`sizeof(StateInfo)` is 192 and the other 128 bytes are left uninitialised and
recomputed. **No push, no second copy, one cache line moved instead of 448
bytes.**

Our per-node cost: `make_move_with` 285 + `set_check_info` 28 + `unmake_move`
24 = **337 c/n**, against SF's `do_move` 154 + `swap_piece` 33 + `undo_move` 3 =
**190 c/n**. (Both exclude threat generation: perf's `--no-children` self-time
puts SF's fused `update_piece_threats` in its own symbols, 120 c/n, and ours is
lazy in `at_full`. Like for like.) **1.77×, and ~5.5 % of all our cycles are
state-record plumbing.**

### 3.5 `activate_inner` — the one place our SIMD is genuinely half-width

`sota_simd.rs:1122-1176`. The pst+threat merge, clamp and pairwise product run
entirely in **i32** (`_mm512_cvtepi16_epi32` → `_mm512_mullo_epi32` →
`_mm512_abs_epi32` → `_mm512_srai_epi32` → `_mm512_cvtepi32_epi8`), i.e. **16
lanes per zmm**. SF's FT activation is an i16 pipeline —
`vpsllw` / `vpmulhw` / `vpminsw` / `vpackuswb` — at **32 lanes per zmm**.

Measured: ours **239 c/n**, SF's FT-activation region **94 c/n** — **2.54×**.
Constants (`sota_head.rs:60-65`): `FT_QUANT = 255`, `THREAT_QUANT = 64`,
`FT_SHIFT = 9`, `ACC = 768`, `L2 = 32`.

Two separable pieces, with very different tractability:

- **The pairwise product is exactly expressible in i16 and is provably
  bit-identical.** The kernel computes `(l · r) >> 9` with `l, r ∈ [0, 255]`.
  A plain `vpmullw` would overflow (255² = 65025 > 32767), which is presumably
  why the i32 path exists — but `vpsllw $7` on one operand followed by
  `vpmulhw` gives `(l·128 · r) >> 16 = (l·r) >> 9` **exactly**, in i16, at 32
  lanes per register. This is precisely SF's `vpsllw`/`vpmulhw`/`vpackuswb`
  sequence (their FT activation region, +0x1200, dominated by `vpminsw`).
- **The pst+threat merge is the hard half**, and it exists only because we
  carry two accumulators at two different quantisations: `pst + round(threat ·
  255 / 64)`, clamped to [0, 255]. `4·|t|` can exceed i16 range, so a
  saturating i16 form (`vpaddsw` ×2 then `vpmulhrsw` by 32640, which is exactly
  `round(|t|·255/64)` where it does not saturate) diverges from the i32 result
  in a narrow band, `|t| ∈ [8192, 8291]`, where the clamped output can differ
  by up to ~400. Rare, but not provably absent.

The structural fix for the merge is a **net-format** change, not an inference
change: SF has no merge step at all because they fused HalfKA and threats into
one accumulator (`7c7fe322`, 2026-06-10). If a future CENN rev quantised the
threat transformer at `FT_QUANT` instead of 64, the whole rescale collapses to
one `vpaddsw` — which is the long-standing quant-unification rider (study 6
item 5, study 1 §6 option D).

This is the clearest "our source shape is worse than their handwritten
equivalent" finding in the study, and — unlike the fold kernels — this stage
streams two 1536-byte L1-resident accumulators sequentially, so it is
**arithmetic-bound** and removed instructions here *will* convert to cycles.

### 3.6 Where we are already at or better than reference

| item | ARCHON | SF18 | note |
|---|---:|---:|---|
| TT geometry | 10 B entry, 3/cluster, `#[repr(C, align(32))]` 32 B cluster, `mul_hi64` index, depth−8·age replacement | identical (`tt.cpp:62-161`, `:277-279`) | **exact parity**; our probe is 25 c/n |
| L1 matmul | `l1_dot_inner` 249 c/n for 768×32 = 24576 MACs | 163 c/n for 1024×16 = 16384 MACs | **1.53× cost for 1.50× the work — reference efficiency per MAC** |
| `see_ge` | 37 c/n @ 1.26 calls/node | 71 c/n | ours cheaper |
| qsearch | 17 c/n | 61 c/n | different shapes; ours cheaper |
| branch prediction | 23.1 miss/node | 26.6 | ours better |
| DRAM / L3 / L2 traffic | lower on every counter | — | ours better |
| threat from-scratch refresh | 33 c/n (0.0346 calls/node) | ~16 c/n | 0.7 % of cycles; still not a target |
| Finny **trigger** | mirror-crossing only | any of 32 king buckets | ours strictly tighter (study 1 §4.6) |

---

## 4. THE DIFFERENTIAL TABLE, REBUILT

ARCHON: shares of `d-ours.data` × the 4713 c/n paired median.
SF18: direct symbols from `d-sf.data` × 3351 c/n, plus study 1's
region decomposition of `Network<1024u>::evaluate` where an internal split is
needed (flagged ‡ — the SF profile is stable to 1.5 % between the two studies,
so this is a safe transplant, but it is not a fresh measurement).

Sorted by **absolute cycles/node gap**:

| # | subsystem | ARCHON c/n | SF18 c/n | **gap** | ratio | mechanism |
|---|---|---:|---:|---:|---:|---|
| 1 | **HalfKA Finny refresh** (driver + fold + copy-out) | **621** | 248 | **+373** | **2.50×** | sort + 64 binary searches + 1536 B memcpy vs SIMD mailbox diff + fused two-store tile |
| 2 | head math total | 822 | 253‡ | +569 | 3.25× | of which: SKP1 85 (no SF counterpart), `activate` +145 (§3.5), drivers +97, L2/L3 tail +112 |
| 3 | **MovePicker construction** | **215** (169 + 46 memcpy) | not separately visible | **≈ +200** | — | 2 KB zeroed + ~2.1 KB copied per node for a never-read buffer; SF's ctor is inlined into `search`, initialises 9 members, zeroes nothing and copies nothing |
| 4 | **make/unmake + state plumbing** | **337** | 190 | **+147** | 1.77× | 224 B built + 224 B pushed vs 64 B prefix memcpy into a caller object |
| 5 | HalfKA lazy driver `current_full` | 156 | 42‡ | +114 | 3.7× | ancestor walk + null-move alias scan per eval |
| 6 | threat resolution `at_full` | 280 | 170‡ | +110 | 1.65× | reconstructive replay + 8-slot linear override scan |
| 7 | `next_move` | 490 | 424 | +66 | 1.16× | near parity; ours has no prefetch of the history planes |
| 8 | threat fold | 471 | 412‡ | +59 | 1.14× | 1-register tile vs 16 |
| 9 | threat event generation | 173 | 164‡ | +9 | 1.05× | parity |
| 10 | HalfKA incremental fold | 240 | 276‡ | **−36** | 0.87× | **we win** |
| 11 | search recursion (`search`+`qsearch`) | 303 | 592 | **−289** | 0.51× | SF inlines far more per node; not comparable like-for-like |
| — | **THREAT SUBTOTAL** | **970** (20.6 %) | 821‡ (24.5 %) | +149 | **1.18×** | |
| — | **HALFKA SUBTOTAL** | **1017** (21.6 %) | 590‡ (17.6 %) | +427 | **1.72×** | |
| — | **ALL FEATURE UPDATE** | **1987** | 1411 | +576 | **1.41×** | |
| — | **SHELL** | **1783** (37.8 %) | 1398 (41.7 %) | +385 | **1.28×** | |
| — | small-net eval (SF only) | — | 142 | — | — | segmented out |

Reading:

1. **Study 1's headline gaps are half-resolved.** Head math was 4.53× — it is
   now 3.25×, and most of the residual is a term SF does not have (SKP1) plus
   one half-width kernel. Threats were 1.23× — now 1.18×. The shell was
   1.34× — now 1.28×.
2. **The Finny refresh survived unchanged and is now the single biggest named
   gap** (+373 c/n = 7.9 % of our total). Study 1 measured 2.70×; today 2.50×
   with a different decomposition and the same verdict.
3. **Two gaps study 1 never looked at are together worth +362 c/n
   (7.7 % of cycles): MovePicker construction and state plumbing.** Both are
   pure bookkeeping — no chess, no arithmetic, no learned term.
4. `current_full` at 3.7× is small in absolute terms but structurally
   interesting: it is a per-eval walk of the accumulator stack looking for the
   nearest computed ancestor plus a null-move alias scan
   (`sota_accumulator.rs:139-160`). SF's equivalent is a single index compare.

---

## 5. PROTOTYPES — measured, not estimated

All four were implemented in throwaway `git archive HEAD` copies under the
scratchpad (`$S/proto`, `proto2`, `proto3`, `proto4`, `protoALL`), built with
the same `x86-64-v3` flags, and validated against three oracles: the 800-FEN
eval hash, the 15,001,555-node fixed-node bench total, and `cargo test`.
**The repository was never modified.**

### 5.1 P4 — Finny refresh: bitboard set diff instead of sort + binary search

`sota_accumulator.rs:225-301`. Today the refresh, on **every** call
(0.299/node = 0.731/eval):

1. walks all ≤ 32 occupied squares, calling `piece_on` and `halfka_index`;
2. **`features[..len].sort_unstable()`** — a 32-element sort;
3. runs **≤ 64 binary searches** (each cached feature against the new sorted
   list, each new feature against the cached list) to build the add/sub lists;
4. folds with `apply_i16_deltas`;
5. **`self.stack[ply].acc[p] = cached.acc`** — a 1536-byte `memcpy`
   (0.2994/node, 51.8 M `Ir`, 25 % of all engine memcpy).

SF's equivalent (`nnue_accumulator.cpp:435-536, 540-674`) does none of this: it
byte-compares two 64-square mailboxes with `_mm256_cmpeq_epi8` +
`_mm256_movemask_epi8`, ANDs the resulting changed-square bitboard with the old
and new occupancies to get the removed/added sets, and then — in **one** tile
loop over 16 live zmm — stores the updated tile **twice**, once to the cache
entry and once to the destination accumulator. No sort, no search, no copy-out.
(Historical note: SF used the 12-way per-(colour,piece-type) `old & ~new`
bitboard diff until `67645619`, 2025-10-14, when it was replaced by the mailbox
compare.)

**Prototype:** the 12-way bitboard version (closer to our `Board`'s native
representation than a mailbox compare, and simpler in safe Rust):
`FinnyEntry.features/len` → `by_piece: [[u64; 6]; 2]`; the diff becomes 12
`old & !new` / `new & !old` pairs with `trailing_zeros` pops; `halfka_index` is
called only for squares that actually changed. Nothing else changed — the fold,
the PSQT rows, and the copy-out are untouched.

| | baseline | **P4** |
|---|---:|---:|
| instructions/node | 9273.9 | **8831.5** (−4.8 %) |
| branch-misses/node | 23.4 | **20.9** (−10.7 %) |
| cycles/node (interleaved) | see §5.5 | **−6 to −8 %** |
| 800-FEN hash | `0615d75e…c92eb` | **identical** |
| bench node total | 15,001,555 | **identical** |

The branch-miss drop is the tell: the 32-element sort and 64 binary searches are
data-dependent branch storms, and their cost was being skidded onto neighbouring
instructions (which is why source-line attribution only showed ~1.8 %).

Not yet prototyped, and additive: **fuse the copy-out** (store the tile to both
the cache entry and the ply slot inside one loop, SF-style) — that removes the
1536-byte `memcpy`, worth a further ~0.7 % of cycles.

### 5.2 P2 — wide register tile in all three fold kernels

`sota_simd.rs` avx512 module: replace the `for off in (0..ACC).step_by(LANES)`
single-register block loop with a `REGS`-register tile
(`TILE = REGS * LANES`, `ACC % TILE == 0`), loading `REGS` zmm accumulators,
applying every add and every sub into all of them, then storing. This is SF's
`NumRegs` shape. Per-lane arithmetic order is unchanged, so it is
**provably bit-identical** (wrapping i16 adds, same sequence per lane).

| REGS | ins/node | NPS vs base (median of 3) |
|---:|---:|---:|
| baseline (1) | 9273.9 | — |
| 4 | — | +3.2 % |
| **8** | **7770.1 (−16.2 %)** | **+2.1 % … +5.1 %** |
| 12 | — | +1.2 % |
| 24 (whole ACC in registers) | — | +1.3 % |

**A −16.2 % instruction reduction bought only ~2.5–3 % of cycles.** That is the
honest headline for this option and it is the direct proof that the fold kernels
are memory-bound, not issue-bound: the removed instructions were executing in
the shadow of the weight-row loads. REGS = 4–8 is the sweet spot; 24 spills.

Eval hash and node total identical.

### 5.3 P1 — MovePicker's 2 KB buffer

`movepick.rs:251` `buf: [ExtMove; MAX_MOVES]` (256 × 8 B = **2048 B**) is
initialised with `[ExtMove::NULL; MAX_MOVES]` at `:331`. The module's own doc
comment on `ExtMove::NULL` says it is *"an inert slot no emission stage reads
(region indices guarantee only written slots are ever consumed)"* — i.e. the
initialiser is admitted to be dead. Callgrind, deterministically:

- `MovePicker::new → memcpy` **0.3929/node**, 86.9 M `Ir` (**42 % of all engine
  memcpy**) — the whole ~2112-byte object copied out of the constructor;
- `MovePicker::next_move → memset` 0.4844/node, 24.2 M `Ir` — the per-stage
  `MoveList` (512 B) zero-fill;
- `MovePicker::new_probcut → memcpy` 0.0344/node.

Stockfish's `MovePicker::moves[]` (`movepick.h:73`, also 2048 B) has **no
initialiser and is never zeroed**, the object is constructed in place as a
`search()` local, and copy/assign are `= delete`d (`movepick.h:39-40`).

**Prototype** (uninitialised buffer only; the constructor's return copy also
disappears once the initialiser is gone):

| | baseline | **P1** |
|---|---:|---:|
| instructions/node | 9273.9 | **8946.5** (−3.5 %) |
| cycles/node | — | **−1.0 %** |
| hash / node total | — | identical |

**The same pattern appears in three more places** (all from the callgrind
call-site table, all deterministic):

| site | dead bytes | rate | why it survives |
|---|---:|---:|---|
| `MovePicker` `buf: [ExtMove::NULL; 256]` (`movepick.rs:331`) | 2048 B zero + ~2.1 KB copy | 0.393/node | the doc comment already says the slots are never read |
| `MoveList::new` `moves: [Move::NULL; 256]` (`moves.rs:146`) | 512 B zero | 0.484/node | one per generated stage |
| `SotaHead::forward_into` `ft: [0u8; 768]` + `nnz: [0u16; 192]` (`sota_head.rs:347-349`) | 1152 B zero | 0.409/node | `activate_ft`/`find_nnz` dispatch through a runtime kernel table, so LLVM cannot prove they fully write the buffers and the dead stores survive |

Total dead traffic: **≈ 1.5 KB zeroed + ≈ 0.83 KB copied per node**, against a
node that does ~4020 L1 loads. The three sites share one fix (typed
`MaybeUninit`) and one risk profile.

A fourth, unattributed site: **`ThreatContext::at_full → memcpy`, 0.305/node,
52.8 M `Ir` = 25.6 % of all engine memcpy** (~173 `Ir`/call ⇒ ~1.5 KB blocks).
Candidates are the null-move `entry.acc[p] = parent.acc[p]` at
`sota_threats.rs:917` and `ThreatSnapshot::from_board`'s mailbox copy
(`:182-188`). Worth 30 minutes with a callgrind line-level run; not sized as an
option here.

⚠ **The prototype is UB as written** — `MaybeUninit::uninit().assume_init()` on
`[ExtMove; 256]` draws an explicit `invalid_value` warning from rustc
("this code causes undefined behavior when executed"). The landed version must
use `[MaybeUninit<ExtMove>; MAX_MOVES]` with `write`/`assume_init_ref` at the
five region boundaries, which is a real (if mechanical) refactor of
`movepick.rs`. The measurement stands; the effort estimate must include that
refactor.

### 5.4 P3 — StateInfo zero-fill: **a dead end, killed by measurement**

Hypothesis: `StateInfo { copied: prev.copied, ..StateInfo::EMPTY }`
(`board.rs:1071-1074`) wastes ~160 bytes of zero-fill per move.

Result: replacing it with an uninitialised record changed instructions/node from
9273.9 to **9265.5 — 8 instructions per node, 0.09 %.** LLVM already
dead-stores the initialiser. **Do not spend time on this.**

The *other* half of §3.4 — the 224-byte `Vec::push` copy at 2.30 % of program —
is real and was **not** prototyped, because removing it means constructing the
child state in place through `states[n]` while the function still reads
`self.st()` as the parent throughout. That is a genuine refactor of a 240-line
safety-critical function, and it is priced as such in §6.

### 5.5 Combined, and the correctness evidence

`protoALL` = P4 + P2(REGS=8) + P1:

| | baseline | **protoALL** |
|---|---:|---:|
| instructions/node | 9273.9 | **7000.2 (−24.5 %)** |
| branch-misses/node | 23.4 | **19.4 (−17 %)** |
| cycles/node (interleaved pairs) | — | **−9 to −10 %** |
| 800-FEN eval hash | `0615d75e…c92eb` | **identical** |
| bench node total | 15,001,555 | **identical** |
| `cargo test --release --features sota,arch-d47v` | — | **660 lib + 24 UCI + all integration: 0 failed** |
| `-- --ignored threats_replay_large` | — | **ok** |

Sub-additivity is expected: P4 and P2 both target the same memory-bound region.

---

## 6. OPTIONS RANKED

Convention (study 1 §6): at fixed cycles, removing a fraction *f* of cycles
multiplies NPS by `1/(1−f)`; house rule ~1 Elo per 1 % NPS at 10+0.1
(conservative — the measured slope has been 2.1).

### Option A — Finny refresh: bitboard set diff  ★ do this first

- **Measured basis:** `finny_refresh_into` 278 c/n + `i16_deltas_inner` 315 c/n
  + 1536 B copy-out ≈ 28 c/n = **621 c/n = 13.2 % of cycles**; SF's equivalent
  is 248 c/n.
- **Measured gain [proto]: −4.8 % instructions/node, −10.7 % branch-misses/node,
  −6 to −8 % cycles → ≈ +6.5 to +8.5 % NPS.**
- **Effort:** low. One function (`sota_accumulator.rs:225-301`) plus the
  `FinnyEntry` field swap. Half a day.
- **Correctness risk: low.** The accumulator arithmetic is untouched; only the
  *derivation* of the add/sub sets changes, and set equality is checkable. The
  existing `nnue_accumulator` tests plus the eval hash cover it.
- **Validation class: bit-identical (sha).** 800-FEN hash must stay
  `0615d75e…c92eb`; fixed-node bench total must stay 15,001,555; `cargo test`
  green. All three verified on the prototype.
- **Follow-on (separate commit, additive ~+0.7 %):** fuse the copy-out — write
  the refreshed tile to both the cache entry and `stack[ply].acc` from the same
  live registers, SF's `nnue_accumulator.cpp:612-613 / :632-633` shape. Needs a
  new two-destination kernel.

### Option B — Wide register tile in the fold kernels (REGS = 8)

- **Measured basis:** three kernels, 1026 c/n = 21.8 % of cycles; SF's
  equivalent tile is 16 registers wide.
- **Measured gain [proto]: −16.2 % instructions/node, ≈ −2.5 to −3 % cycles →
  ≈ +2.5 to +3 % NPS.** The instruction saving does **not** convert, because the
  kernels are DRAM/L2-latency bound (53 % of all DRAM fills).
- **Effort:** low (≈ 1 day; three avx512 kernels, and the avx2 twins for
  portability parity).
- **Correctness risk: very low; bit-identity is provable** (per-lane wrapping
  i16 adds, identical order).
- **Validation class: bit-identical (sha).** Verified on the prototype.
- Note this **supersedes option F** from study 1: the address hoist is already
  in the codegen and bought ~2 % of the last program. Do not do more of F.

### Option C — MovePicker: uninitialised buffer + in-place construction

- **Measured basis:** `MovePicker::new` 169 c/n + 42 % of the engine's memcpy
  (≈ 46 c/n) = **215 c/n = 4.6 % of cycles**, all of it zeroing and copying
  2 KB that no stage reads.
- **Measured gain [proto, unsafe form]: −3.5 % instructions/node, −1.0 %
  cycles → ≈ +1.0 to +1.5 % NPS.** (The prototype removed the zero-fill; the
  return copy needs the in-place construction to fully vanish, so the landed
  ceiling is somewhat higher than measured.)
- **Effort: medium.** The correct implementation is
  `buf: [MaybeUninit<ExtMove>; MAX_MOVES]` with `write` at the fill sites and
  `assume_init_ref` at the five region reads — mechanical but it touches every
  stage in `movepick.rs`. 1 day. A `#[deny(unsafe_code)]` module exemption with
  a documented safety argument is required (D14a precedent).
- **Correctness risk: medium** — the safety argument *is* the region-index
  invariant. If any stage can read an unwritten slot, this is UB rather than a
  wrong score. Mitigation: a debug-only `[bool; MAX_MOVES]` written-tracker
  asserted on every read, compiled out in release (the `assert_consistent`
  precedent).
- **Validation class: bit-identical (sha) + node-identical.** Verified on the
  prototype. Also re-run the **default-build depth-6 bench** (`tools/check.sh`,
  trailer **701605**) — `movepick.rs` is not feature-gated, so this option
  changes the default build too.

### Option D — `activate_inner`: i32 → i16 pipeline (split in two)

- **Measured basis:** 239 c/n = 5.1 % of cycles, against SF's 94 c/n for the
  analogous stage. Ours runs 16 lanes/zmm, theirs 32.

**D1 — the pairwise product only (provably bit-identical).**
Replace `_mm512_mullo_epi32` + `_mm512_srai_epi32::<9>` + `_mm512_cvtepi32_epi8`
with `vpsllw $7` + `vpmulhw` + `vpackuswb` on i16 lanes (§3.5 derivation).
- **Estimated gain: +0.7 to +1 % NPS.** Effort: half a day.
- **Risk: low; exact by construction.** Validation: **bit-identical (sha)**.

**D2 — the pst+threat merge in i16.**
- **Estimated gain: +1.5 to +2 % NPS** (the merge is ~2/3 of the kernel).
- **Effort: medium (1–2 days). Risk: medium-high** — §3.5 shows a saturating
  i16 form is *not* exactly equivalent for `|threat_acc| ∈ [8192, 8291]`.
  Either prove that band is unreachable for real nets (measurable: instrument
  the max |threat accumulator lane| over the 800-FEN battery and a game corpus),
  or accept a non-identical change.
- **Validation class: bit-identical (sha) if the band is proved unreachable;
  otherwise int-cp on all 800 FENs plus a 10+0.1 SPRT.** Keep the i32 path as
  the scalar reference with a randomised scalar-vs-SIMD parity test.

**D3 (structural, net-format) — unify the threat quantisation to `FT_QUANT`**
so the merge becomes a single `vpaddsw`. Removes D2's whole problem and most of
its cost. This is a rider on the next CENN rev (study 6 item 5), not an
inference change, and it must be scheduled with the trainer, not the engine.

### Option E — `make_move`: construct the child StateInfo in place

- **Measured basis:** `alloc/vec/mod.rs:1933` = **2.30 % of all cycles**, the
  224-byte copy from stack temp into `states`. (The zero-fill half is already
  free — §5.4.)
- **Estimated gain: +2 to +2.5 % NPS.**
- **Effort: high (2–3 days).** `board.rs:1064-1300` must write through
  `self.states[n]` while still reading the parent at `n-1`; today the parent is
  read as `self.st()` in ~10 places. Options: pre-`reserve` + `spare_capacity_mut`
  with a split borrow, or hoisting the handful of parent reads into locals up
  front (the function already does this for `prev_blockers`).
- **Correctness risk: medium.** Any missed field becomes a stale value rather
  than a crash. The D24 perft ritual (`--release -- --ignored perft_full_suite`)
  plus `assert_consistent` in debug make/unmake are strong oracles here.
- **Validation class: search-identical nodes + D24 perft + bench 701605.**
- Cheaper partial: shrink `StateInfo` itself. `check_squares: [Bitboard; 6]`
  (48 B) + `piece_threats: [Bitboard; 6]` (48 B) + `all_threats` (8 B) are
  **104 of the 224 bytes** and are recomputed every move by `set_check_info`.
  Moving them out of the pushed record into a single non-stacked "current"
  scratch (they are only ever read for the current position) cuts the copy by
  46 % for a fraction of the effort. **Verify the "only current" claim first** —
  `unmake_move` must be able to restore them, so this needs a recompute-on-unmake
  or a one-deep shadow.

### Option F′ — Software-prefetch the history planes in `do_move`

- **Measured basis:** `next_move` is 6.0 % of all DRAM fills and 4.6 % of L2
  fills; its quiet scorer (region +0x2600, 23 % of the symbol ≈ 113 c/n) is a
  chain of dependent history-plane loads. SF prefetches five history/correction
  lines per `do_move` (`position.cpp:1006-1010`), one whole move-make ahead of
  the consumer; we prefetch only the TT.
- **Estimated gain: +0.5 to +1.5 % NPS.** Honest uncertainty: study 1 recorded
  a **measured loss** (−2.5 %) for prefetching NNUE weight rows, and the same
  could happen here. The difference is lead distance — SF's history prefetch
  leads by an entire `do_move` + child prologue, whereas the NNUE attempt
  prefetched a line consumed a few instructions later.
- **Effort: low (half a day).** The `MakeMoveHooks` seam already exists
  (`board.rs`, `TtPrefetch`); add the correction/pawn-history addresses to it.
- **Correctness risk: nil** (prefetch is architecturally inert).
- **Validation class: bit-identical (sha) + bench 701605** — a prefetch cannot
  change a value; if the node count moves, something else is wrong.

### Option G — `current_full`: cheap "nearest computed ancestor"

- **Measured basis:** 156 c/n (3.3 %) vs SF's ~42 c/n. The function re-walks the
  accumulator stack downward on every eval plus runs a null-move alias scan
  (`sota_accumulator.rs:139-160`).
- **Estimated gain: +1 to +1.5 % NPS** by maintaining a `last_computed[persp]`
  index updated at `push`/`pop` instead of scanning.
- **Effort: low-medium (1 day). Risk: low-medium** — the alias/laziness
  invariants are subtle and the debug oracle
  (`assert_matches_refresh`, `sota_context.rs:386-387`) is the guard.
- **Validation class: bit-identical (sha) + node-identical.**

### Option H — AVX-512 partial insertion sort (SF's `MoveSorter`)

- **Measured basis:** our `partial_insertion_sort` inner loop
  (`movepick.rs:101-114`) is 1.05 % of program by source line.
- SF added a 16-element in-register sorter under `USE_AVX512`
  (`movepick.cpp:67-106`, **new since `sf_18`**) keeping values and moves in two
  zmm and inserting with `_mm512_mask_expand_epi32`.
- **Estimated gain: +0.5 %. Effort: medium. Verdict: park.**

### Explicitly rejected, with the measurement that kills each

| option | why rejected |
|---|---|
| More of study-1 option F (address hoisting / count specialisation) | §2.4: F bought 2 % of the last program and failed its abort test; **and SF deleted their own count-specialisation machinery** in `86beb0f5` after the accumulator merge |
| `StateInfo` zero-fill removal | §5.4: **8 instructions per node**. LLVM already dead-stores it |
| Compressing / low-ranking the weight tables | §2.1: we have **fewer** DRAM fills, L3 fills and L2 misses per node than SF18, and a lower L1d miss rate. We are not memory-*volume* bound |
| Branch-prediction work | §2.1: 23.1 mispredicts/node vs SF's 26.6. Front closed by option S |
| TT geometry changes | §3.6: byte-for-byte the same design as SF18's, including `align(32)` and `mul_hi64`. Probe is 25 c/n |
| Threat from-scratch refresh | 33 c/n at 0.0346 calls/node. Ceiling < 0.5 % — unchanged from study 1 |
| Making `see_ge` / qsearch / movegen faster | all measured **cheaper than SF18's** per node |
| Threat option T1 (3-primitive driver) | generation is now 1.05× SF's and the fold 1.14×; the 2–4 day, highest-risk item in study 1 is worth less than ever. **Still: do not do this** |

---

## 7. RECOMMENDED SEQUENCE

Ordering is (measured gain × confidence) ÷ (effort × risk), and puts every
provably bit-identical item before every arguable one.

**Step 1 — Option A (Finny bitboard diff).** ≈ **+6.5–8.5 % NPS**, half a day,
bit-identical, already prototyped and green.
*Acceptance:* (a) 800-FEN hash exactly `0615d75e5c9487b6f6c66255d3c1d67789f46471d4c8eaec41580e3c070c92eb`;
(b) fixed-node bench total exactly **15,001,555** on
`bench 16 1 300000 wl50.fens nodes`; (c) `cargo test --release --features
sota,arch-d47v` green including `--ignored threats_replay_large`;
(d) `perf stat` shows branch-misses/node fall from ~23.1 to ~20.9 and
instructions/node from 9273.9 to ~8831.5 (both deterministic — they must match
to the digit); (e) paired interleaved cycles/node ≥ −5 %.
*Abort if:* the eval hash moves at all.

**Step 2 — Option B (wide fold tile, REGS = 8).** ≈ **+2.5–3 % NPS**, 1 day,
bit-identical, prototyped and green.
*Acceptance:* (a) hash unchanged; (b) node total unchanged;
(c) instructions/node falls to ~7770 (≈ 7000 stacked on step 1);
(d) `i8_deltas_into_inner`'s dynamic zmm share rises from 43.7 % — and note
explicitly that `i16_deltas_into_inner`'s will **not** rise much, because that
kernel is latency-bound; (e) paired cycles ≥ −2 %.
*Do not* re-litigate REGS: 4, 8, 12 and 24 were swept and 4–8 is the plateau.

**Step 3 — Option F′ (history prefetch in `do_move`).** +0.5–1.5 %, half a day,
architecturally inert. Cheap enough to try before the harder items, and its
failure mode is a clean revert.
*Acceptance:* hash unchanged, node total unchanged, paired NPS ≥ +0.5 %.
*Abort if:* NPS does not improve — study 1's NNUE-prefetch attempt lost 2.5 %,
so treat a null result as a real possibility and revert without argument.

**Step 4 — re-measure.** After steps 1–3 the engine should be near
**4200–4350 cycles/node** (from 4713), i.e. **1.25–1.30× SF18** rather than
1.41×. Re-run the flat profile and the differential; the shell's share will rise
past 40 % and the ranking below should be re-taken on fresh numbers.

**Step 5 — Option C (MovePicker `MaybeUninit`).** +1.0–1.5 %, 1 day, needs the
typed-`MaybeUninit` refactor and a `#[deny(unsafe_code)]` exemption with a
written safety argument plus a debug-only written-tracker.
*Acceptance:* hash + node total unchanged, **and the default-build depth-6 bench
trailer reproduces 701605** (`tools/check.sh`) — this option is not
feature-gated.

**Step 6 — Option D (`activate` i16 pipeline).** +2–2.5 %, 1–2 days, and the
only remaining item where our SIMD is genuinely half-width. Gate on a written
exact-integer-equivalence proof plus a scalar-vs-SIMD parity test; if the
800-FEN hash moves, route to a full 10+0.1 SPRT rather than claiming neutrality.

**Step 7 — Options G then E.** `current_full` (+1–1.5 %, low risk) before the
`make_move` in-place refactor (+2–2.5 %, high effort, D24 perft ritual
mandatory). Consider the cheaper `StateInfo`-shrink partial first.

Cumulative, sub-additively: steps 1–3 ≈ **+10 %**, steps 1–6 ≈ **+14–16 %**,
everything ≈ **+18 %**, which would put us at roughly **1.15–1.20× SF18
cycles/node** on this workload.

### 7.1 Interaction with the v1 release campaign

`docs/research/archon-v1-release-campaign.md`'s APPROVED SEQUENCE puts
**P4 — perf study / NPS characterization + PGO/native release build** *after*
**P3 — search fit** and *before* **P5 (T11 disagreement head)** and
**P6 (full SPSA)**. Implications:

1. **Steps 1–3 are safe to land during P2/P3.** All three are bit-identical or
   architecturally inert, so they change **no** game result at fixed nodes and
   cannot invalidate a running gate. They *do* change equal-time results — which
   is the point — so any equal-time match already in flight (the S2a-vs-S2
   equivalence match was live during this study) must be finished or restarted,
   not straddled.
2. **Do not land steps 5–7 between the SPSA baseline and the SPSA run.** Search
   parameters tuned at one NPS do not transfer cleanly to another; anything that
   moves NPS by > 5 % belongs *before* P6, not during it.
3. **The bench trailer.** Steps 1, 2, 4 and 6 touch `sota`-gated code only and
   cannot move the default-build depth-6 bench (**701605**); steps 3, 5 and 7
   touch `board.rs`/`movepick.rs` in the default build and **must** reproduce
   701605 bit-for-bit or be declared functional. Step 7 additionally triggers the
   **D24 perft ritual** (`cargo test --release -- --ignored perft_full_suite`).
4. **PGO is still unmeasured and is P4's own item.** Nothing here overlaps it;
   PGO should be measured *after* steps 1–3 so it optimises the shape we ship.
5. **The equal-time distillation gate benefits directly.** At 10+0.1 a +10 %
   NPS is worth roughly +10 Elo on the house rule (and ~+21 on the measured
   slope), which is a material fraction of the "vs teacher ≥ 0" bar.

---

## 8. UNCERTAINTIES AND THREATS TO VALIDITY

1. **The box was busier than in study 1** — GPU trainer plus a live 4-process
   fastchess match, load 4–7. NPS varied ±10 % and even `cycles:u` ±6 % for the
   *same* binary within the hour. All A/B claims are interleaved; all
   prototype claims are anchored on `instructions:u`, which is exactly
   reproducible. Absolute NPS figures here must not be compared to any other
   session's.
2. **One measurement round was silently contaminated** by an earlier background
   benchmark that had not exited; it showed P4 *slower* than baseline. It was
   discarded and re-run after killing the stray process. Mentioning it because
   it is exactly the failure mode that produces a false negative.
3. **SF's internal `evaluate` split (‡ rows in §4) is transplanted from study
   1's region analysis**, not re-measured at the same granularity. I verified
   the symbol-level shares are stable (evaluate 34.5 % → 35.03 %,
   refresh_cache 6.96 % → 7.39 %, next_move 12.95 % → 12.66 %), so the
   transplant is sound, but the sub-symbol numbers inherit study 1's 80/20
   threats-vs-HalfKA assumption for the shared scalar-resolve region. Only the
   **ALL FEATURE UPDATE** row is assumption-free.
4. **Node definitions may differ** between the engines. The totals agree to
   0.06 % because both honour `go nodes`, which is not proof that a "node"
   means the same thing.
5. **SF binary vs SF sources.** The profiled binary is the Stockfish 18 release
   (bimodal, `<1024u>` + `<128u>`; the small net is 4.2 % of cycles and is
   segmented out); the read sources are 243 commits *after* `sf_18`. Where the
   two diverge — the deleted `UpdateOperation` machinery, the new AVX-512
   `MoveSorter`, the mailbox-compare refresh diff — I said so at the citation.
   The tiling, TT geometry and `do_move` prefix-copy facts were confirmed
   identical at `sf_18`.
6. **Position mix.** wl50 is endgame-heavy (study 1 §1.5 measured the threat
   share ~3 points higher on middlegame-heavy positions). I did **not** re-run
   the middlegame-heavy profile this session; treat the threat-channel rows as
   the favourable end for us, and the Finny/MovePicker/state rows as roughly
   mix-independent (they scale with nodes, not with material).
7. **Prototype P1 is UB as written** and its measured gain is therefore a
   *ceiling on the mechanism*, not a promise about the safe implementation. The
   typed-`MaybeUninit` version may retain a little of the cost.
8. **Nothing here was Elo-tested.** Every Elo figure is the house 1 Elo/1 % NPS
   conversion applied to a measured cycle delta. The prior measured slope was
   2.1 Elo/1 %, so these are conservative — conversions, not results.
9. **The `activate` (option D) gain is estimated, not measured.** It is the only
   item in §6's top six without a prototype behind it.

---

## 9. ARTIFACTS

Session-local, not committed, under
`/tmp/claude-1000/-home-nick-projects-chessengine/61b8a705-.../scratchpad`:

- `wl50.fens` — regenerate with `awk 'NR%16==1' tools/nnue/calib/scale-check-800.fens`
- `d-old.data` / `d-ours.data` / `d-sf.data` — the three matched flat profiles
- `d-p2.data` — prototype-B profile; `pprof.data` — line-tables profile
- `dram-ours.data` / `l2-ours.data` — memory-event attribution profiles
- `cg-ours.out` — callgrind call counts (1,000,761 nodes, AVX2 path)
- `opc2.py` (opcode / register-width / 512 B-region mix), `ann.py`
  (instruction-level annotate), `flat.sh`, `pair.sh`, `pstat.sh`
- `proto/` (option B), `proto2/` (C), `proto3/` (the dead end), `proto4/` (A),
  `protoALL/` (A+B+C) — throwaway `git archive HEAD` trees; binaries
  `bin-p1/p2/p2r4/p2r12/p2r24/p3/p4/all`
- `eval800-S2.txt` — the 800 integer-cp reference column
