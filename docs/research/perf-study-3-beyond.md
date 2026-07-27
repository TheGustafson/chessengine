# ARCHON perf study #3 — where we are still behind SF18, and where we can go past it

Date: 2026-07-26 (night). Third commissioned study in the chain
(`threat-channel-perf-study.md` = study 1, `perf-study-2-broad.md` = study 2).
Two mandates: **(A)** rebuild the ground truth and the SF18 differential on the
post-A/B/C/D/E binary; **(B)** hunt for structural advantages our architecture
enables that Stockfish's constraints deny them.

**Study only — the repository working tree was never modified by this session.**
Binaries were built (allowed). Five *prototypes* were written into throwaway
`git archive 9542e93` copies under the session scratchpad and measured there;
every number marked **[proto]** is a measurement, not a landed change.

---

## 0. HEADLINE

**(1) We are now 1.28× SF18 on cycles/node** (was 1.41× in study 2, 1.99× in
study 1). Measured on four interleaved paired runs of the same matched
fixed-node workload: ARCHON **4536 c/n** vs SF18 **3538 c/n**. Instructions/node
fell from study 2's 9274 to **6780** (−26.9 %, deterministic to four digits).
The A+B+C+D+E program bought **−12.0 % cycles** in a same-session three-way
paired measurement (5069 → 4461 c/n, mean of two orderings).

**(2) The biggest single lever in this study is not in the source at all — it
is the build.** `-C target-cpu=native` plus PGO measures **+8.1 % NPS**
(median of 8 ABBA-balanced blocks, **positive in 8/8**, range +4.5 … +12.5 %),
at **−11.9 % instructions/node and −11.5 % L1d loads/node** (both
deterministic), with the 800-FEN eval hash and the depth-13 node total
**both identical**. The v1 campaign plans to ship an `x86-64-v3` binary and
lists PGO as "still unmeasured". It is measured now, it is the largest
zero-risk item available, and it is *also* the fix for our worst remaining
memory counter (icache misses, 2.25× SF18's).

**(3) The cycle map moved again, and the shell is now dominant.**

| block | study 1 | study 2 | **now** | now c/n |
|---|---:|---:|---:|---:|
| search + movegen shell | 28.1 % | 37.8 % | **38.0 %** | 1724 |
| threat channel | 15.1 % | 20.6 % | **20.4 %** | 927 |
| HalfKA accumulator | 16.1 % | 21.6 % | **18.9 %** | 859 |
| head math + eval driver | 36.6 % | 17.4 % | **18.6 %** | 844 |
| libc `memcpy`/`memset` | (in blocks) | 2.9 % | **3.1 %** | 139 |

`MovePicker::next_move` is our #1 symbol at **13.75 % / 624 c/n**. The Finny
refresh — study 2's biggest named gap at 621 c/n — is down to **467 c/n**
against SF18's 267 (2.50× → **1.76×**).

**(4) Three findings nobody has looked at before.** (a) A **`std::sync::RwLock`
read on the per-eval hot path** (`sota_context.rs:433` → `mod.rs:183`), costing
~1.2 % of cycles — and the slot's own doc comment says it is read "never on the
hot per-node path". (b) The accumulator stacks carry **no cache-line alignment**
(`Entry` is 3152 B, align 4 — every ply starts 16 B out of phase), and adding
`#[repr(align(64))]` cuts **L1d loads by 1.29 % at an identical instruction
count**. (c) The L1 matmul's group density is **66.8 %** (128.3 of 192 groups
live) — the nnz sparsity machinery is skipping only a third of the work.

**(5) The best beyond-SF18 bet is the in-net policy head as an ordering prior —
but it is worth far less than a 50-position probe suggests.** On 50 FENs at
depth 13 it looked like **−12.0 % nodes**. On **800 FENs at depths 10 and 12**
it is **−1 to −4 % nodes** for a measured **−2.2 % NPS**, i.e. roughly
effective-NPS-neutral today. The signal is real and SF has no analogue, but the
tax is concentrated in one avoidable implementation choice (an independent
`generate_legal` inside the argmax substitute). Fix that and the −3.7 % node
reduction becomes ~**+4 % effective NPS**.

**(6) One prototype refuted a clean-looking hypothesis.** `l1_dot_inner` runs
two loop-carried `vpdpbusd` chains at Zen 4's 4-cycle latency, which *looks*
like a 512-cycle latency bound against a 128-cycle throughput bound. A
four-chain rewrite was built, proven bit-identical, and moved the kernel's
share **not at all** (6.46 % → 6.53 %). A memory-event profile then explained
why: `l1_dot_inner` is **20.65 % of all L1←L2 fills** — it is load-bound.

---

## 1. GROUND TRUTH

### 1.1 Provenance, and the gate catalog

Baseline: `main` @ **`9542e93`** ("activate as a full i16 pipeline"), working
tree clean at session start.

```bash
CARGO_TARGET_DIR=$S/tg-v3 RUSTFLAGS="-C target-cpu=x86-64-v3" \
  cargo build --release --features sota,arch-d47v
# sha256 a4f423e42555be14066b0b0a4f4fcd916ddbe1640c0d9ba7237ff0adf20a24de
#     == engines/bin/archon-d47v-D                       (byte-for-byte)
```

Net `/tmp/d47v-sb100-nocpol.cenn` (84,158,584 B). Reference
`engines/bin/stockfish18` (unstripped, 1235 symbols).

All four gates verified on the baseline:

| gate | value | status |
|---|---|---|
| 800-FEN eval hash | `0615d75e5c9487b6f6c66255d3c1d67789f46471d4c8eaec41580e3c070c92eb` | ✅ |
| fixed-node bench (`16 1 300000 wl50 nodes`) | 15,001,555 | ✅ |
| **fixed-depth bench (`16 1 13 wl50 depth`)** | **2,940,900** | ✅ *(new pin — see §1.4)* |
| default build, depth-6 trailer | 701,605 | ✅ |

**New instrument — use the fixed-DEPTH total, not the fixed-node total, as the
search-identity gate.** The fixed-node total is *not* cadence-free: the node
limit is checked on a granularity, so the per-position overshoot past 300,000
moves with codegen. Two provably behaviour-preserving builds
(`#[repr(align(64))]` on the accumulator entries; a relaxed-atomic shadow of an
`RwLock`) produced 15,001,555 and 15,001,565 for the *same* search. The
depth-13 total 2,940,900 reproduced exactly across every behaviour-preserving
variant I built and diverged only when behaviour actually changed. Study 2
treated 15,001,555 as a hard identity gate; it is a strong gate with a ±10-node
noise floor, and the depth gate has none.

### 1.2 Contamination — stated plainly, including a real incident

- An RTX 4060 Ti trainer (`tools/pump_train.py`) held 93 % GPU / ~245 % CPU for
  the whole session. Load average 2.7 → 4.6 over ~90 minutes.
- **A parallel Claude session was working the same repository.** It landed two
  commits (`aabb406`, `8cd138d` — S0 search patches) mid-session and left
  `src/search/mod.rs` and `src/search/params.rs` dirty, and it ran a debug
  binary at 99 % CPU during the last measurement block. Two prototypes
  (alignment, PGO) were first built after `HEAD` moved and therefore measured a
  *different search*; both were **rebuilt from the pinned `9542e93`** and
  re-measured. Every number below comes from a `9542e93` build. This is
  recorded because for twenty minutes it looked exactly like a latent
  uninitialised-memory bug — it was not.
- Consequence: **cycles drifted badly in the second half.** The same baseline
  binary measured **4303 – 7530 c/n** across the session. Every conclusion rests
  on (a) deterministic counters (`instructions:u`, `L1-dcache-loads:u`, node
  totals, callgrind `Ir`), (b) shares of a single profile, or (c)
  ABBA-balanced interleaved ratios. **No absolute NPS from this session should
  be compared to any other session's.** The §1.3 counter table comes from the
  quiet first hour, where the same binary reproduced to ±1.2 %.

`perf stat` multiplexed at ~72–85 % with 6 events; values are perf's scaled
estimates. `LLC-*` remains `<not supported>`; Zen 4 `ls_any_fills_from_sys.*`
and `l2_cache_req_stat.*` are used instead.

### 1.3 Whole-engine counters — ARCHON vs SF18, interleaved

Four rounds, orders alternated (A,S,A,S then S,A,S,A). Node totals
15,001,555 vs 15,010,238 (0.06 % apart).

| metric | ARCHON @ `9542e93` | SF18 | ours ÷ SF |
|---|---:|---:|---:|
| **cycles/node** (median of 4) | **4536** | **3538** | **1.282×** |
| instructions/node | 6779.6 | 4331.4 | **1.565×** |
| IPC | 1.494 | 1.224 | 1.22× |
| branches/node | 842.7 | 394.8 | 2.13× |
| branch-misses/node | 19.36 | 26.56 | **0.73×** (we win) |
| L1d loads/node | 3366.6 | 2368.3 | 1.42× |
| L1d misses/node | 566.9 | 424.7 | 1.33× |
| L1d miss rate | 16.84 % | 17.93 % | **0.94×** (we win) |
| **L1←L2 fills/node** | **546.3** | 391.3 | **1.40×** |
| L2 accesses/node | 592.7 | 446.3 | 1.33× |
| L2 misses/node | 44.0 | 57.6 | **0.76×** (we win) |
| L3 (local-CCX) fills/node | 38.1 | 50.3 | **0.76×** (we win) |
| **DRAM fills/node** | **9.33** | 11.33 | **0.82×** (we win) |
| **icache misses/node** | **197.5** | 88.0 | **2.25×** |

The four SF18 runs reproduced study 2's deterministic counters to four digits
(branches/node 394.8, branch-misses/node 26.6, L1d loads/node 2369) — a strong
cross-session validation of the harness. SF18's *cycles*/node measured 3538 here
vs 3351 in study 2 and 3320 in study 1; that +5.6 % is box state, not Stockfish.

**Read the memory rows carefully, because they set the whole of Part B.** At
**L2, L3 and DRAM we are strictly better than SF18** — fewer misses, fewer
fills, at a lower L1d miss *rate*. At **L1 and the instruction cache we are
worse** — 1.40× the L1←L2 refill traffic and 2.25× the icache misses. The
surplus is at the far end of the hierarchy; the deficit is at the near end.
Concretely: 9.33 DRAM fills/node × 64 B × 1.16 M nps = **0.69 GB/s on one
thread**, roughly 1 % of this box's DDR5 bandwidth. We could afford to touch a
great deal more DRAM. We cannot afford to touch more L1 or more code.

The icache row is a **new** differential: study 2 found parity per instruction
(1/51 vs 1/50). We now execute 26.9 % fewer instructions but miss icache almost
as often, so the rate per instruction is 1/34 vs SF's 1/49 — **1.4× worse**.
Our hot functions are large: `next_move` 2592 static instructions,
`search::<NonPv>` 2689, `at_full` 1935.

### 1.4 The three-generation arc, measured in one session

Same workload, same core, interleaved, two rounds:

| binary | commit | c/n (r1 / r2) | instructions/node | NPS |
|---|---|---:|---:|---:|
| `archon-d47v-fast` (study 1) | `bf3ec90` | 6736 / 6887 | 11,263 | 764k / 744k |
| `archon-d47v-S2` (study 2) | `b1b9c6f` | 5120 / 5018 | 9,275 | 1004k / 1024k |
| **HEAD** | **`9542e93`** | **4498 / 4424** | **6,784** | 1146k / 1165k |

- **A+B+C+D+E = −12.0 % cycles, −26.9 % instructions** vs study 2's baseline
  (5069 → 4461 c/n; 9275 → 6784 i/n).
- **The whole arc = −34.5 % cycles, −39.8 % instructions** vs study 1's binary
  (6812 → 4461 c/n), i.e. **+53 % NPS** at fixed nodes on this workload
  (754 k → 1155 k).

---

## 2. THE FRESH CYCLE MAP

### 2.1 Flat profile (shares × the 4536 c/n paired median)

| # | symbol | cyc % | c/node | calls/node | c/call | block |
|---|---|---:|---:|---:|---:|---|
| 1 | `MovePicker::next_move` | **13.75** | 624 | 1.8375 | 340 | shell |
| 2 | `avx512::i8_deltas_into_inner` | **8.89** | 403 | 0.7824 | 515 | threat fold |
| 3 | `avx512::i16_deltas_inner` | 6.57 | 298 | 0.2990 | 997 | **Finny fold** |
| 4 | `avx512::l1_dot_inner` | 6.46 | 293 | 0.4089 | 717 | head |
| 5 | `Worker::search::<NonPv>` | 6.29 | 285 | — | — | shell |
| 6 | `ThreatContext::at_full` | 6.26 | 284 | 0.4089 | 695 | threat resolve |
| 7 | `avx512::i16_deltas_into_inner` | 5.00 | 227 | 0.6592 | 344 | HalfKA fold |
| 8 | `Board::make_move_with::<TtPrefetch>` | 4.73 | 215 | 1.0018 | 215 | shell |
| 9 | `HalfkaContext::current_full` | 4.27 | 194 | 0.4089 | 474 | HalfKA driver |
| 10 | `sota_threats::record_square_touch` | 3.17 | 144 | 1.0038 | 143 | threat gen |
| 11 | `HalfkaContext::finny_refresh_into` | 2.96 | 134 | 0.2990 | 448 | Finny driver |
| 12 | `avx512::activate_inner` | 2.19 | 99 | 0.4089 | 242 | head |
| 13 | `SotaContext::evaluate` | 2.17 | 98 | 0.4089 | 240 | eval driver |
| 14 | `avx512::skip_dot_inner` | 2.09 | 95 | 0.4089 | 232 | head (SKP1) |
| 15 | **`SotaContext::push`** | **1.91** | **87** | 1.0018 | **87** | acc bookkeeping |
| 16 | `avx512::l2_tail_inner` | 1.68 | 76 | 0.4089 | 186 | head |
| 17 | `Histories::correction_value` | 1.54 | 70 | 0.9871 | 71 | shell |
| 18 | `Board::set_check_info` | 1.32 | 60 | 1.0036 | 60 | shell |
| 19 | `AttackTables::attacks` | 1.21 | 55 | — | — | movegen |
| 20 | `Histories::update_continuation_histories` | 1.09 | 49 | — | — | shell |
| 21 | **libc `memcpy`/`memset`** | **3.06** | **139** | 1.606 | 87 | bookkeeping |
| 22 | `Board::see_ge` | 0.97 | 44 | 1.2624 | 35 | shell |
| 23 | `SotaHead::forward_into` | 0.93 | 42 | 0.4089 | 103 | head driver |
| 24 | `Board::legal` | 0.84 | 38 | 1.6167 | 24 | movegen |
| 25 | `Worker::do_move` | 0.79 | 36 | — | — | shell |
| … | `sota_features::refresh_threat` | 0.78 | 35 | 0.0348 | 1006 | threat refresh |
| … | `Board::attackers_to` | 0.78 | 35 | 0.9740 | 36 | movegen |
| … | `TranspositionTable::probe` | 0.77 | 35 | 1.0434 | 34 | TT |
| … | `Board::unmake_move` | 0.60 | 27 | 1.0018 | 27 | shell |
| … | `sota_threats::ripple` | 0.55 | 25 | 0.4280 | 58 | threat gen |
| … | `avx512::find_nnz_vbmi2_inner` | 0.49 | 22 | 0.4089 | 54 | head |
| … | loader / `init_magics` (startup) | 0.96 | 44 | — | — | excluded |

Call frequencies are deterministic (callgrind, 1,000,761 nodes, AVX2 path).
`MovePicker::new` **no longer exists as a symbol** — option C's in-place
construction inlined it away.

### 2.2 Block totals

| block | cyc % | c/node | vs SF18 share |
|---|---:|---:|---|
| search + movegen shell | **38.01** | 1724 | SF 41.6 % / 1470 c/n |
| threat channel | **20.43** | 927 | SF ≈19.7 % / ≈696 c/n |
| HalfKA accumulator | **18.93** | 859 | SF ≈19.1 % / ≈675 c/n |
| head math + eval driver | **18.60** | 844 | SF ≈11 % / ≈400 c/n |
| libc `memcpy`/`memset` | 3.06 | 139 | SF 0.66 % of a 0.66 % libc total |
| startup / loader | 0.96 | 44 | — |

### 2.3 Where the 3.06 % of libc goes (deterministic, callgrind)

| caller → callee | calls/node | bytes/node | note |
|---|---:|---:|---|
| `MovePicker::next_move → memset` | 0.4844 | 248 | `MoveList::new`'s `[Move::NULL; 256]` |
| `SotaHead::forward_into → memset` | 0.4089 | 471 | `ft: [0u8; 768]` + `nnz: [0u16; 192]` |
| `ThreatContext::at_full → memcpy` | 0.3050 | ~458 | ~1.5 KB blocks |
| `finny_refresh_into → memcpy` | 0.2992 | 460 | the 1536 B cache copy-out |
| `ThreatContext::apply_at → memcpy` | 0.0347 | ~5 | |

**Study 2's §5.3 table listed exactly two dead-zero sites that option C did not
touch — `MoveList::new` and `forward_into`'s `ft`/`nnz`. Both are still there,
at exactly the rates study 2 measured** (0.4844 and 0.4089 per node). Together
they are 0.893 `memset` calls/node moving 719 B/node that no stage ever reads.

### 2.4 Position-mix sensitivity

Same binary and protocol on the middlegame-heavy default bench FEN set:

| symbol / block | wl50 (endgame-heavy) | default FENs (middlegame) |
|---|---:|---:|
| `i8_deltas_into_inner` | 8.89 % | **12.01 %** |
| `ThreatContext::at_full` | 6.26 % | **7.52 %** |
| `record_square_touch` | 3.17 % | 3.30 % |
| **threat channel total** | **20.4 %** | **≈24.4 %** |
| `finny_refresh_into` + `i16_deltas_inner` | 9.53 % | **7.29 %** |
| `i16_deltas_into_inner` | 5.00 % | 5.66 % |
| `next_move` | 13.75 % | 13.29 % |
| `l1_dot_inner` | 6.46 % | 6.03 % |
| `correction_value` | 1.54 % | 1.71 % |

Same shape as studies 1 and 2: the threat channel is ~4 points dearer in
middlegames, the Finny refresh ~2 points cheaper, shell items mix-insensitive.
Real games sit at or past the default-FENs end, so §4's threat rows are the
favourable end for us.

### 2.5 What A–E actually bought (matched profiles, c/n)

| symbol | study 2 (`b1b9c6f`) | now (`9542e93`) | Δ |
|---|---:|---:|---:|
| `finny_refresh_into` (option A) | 278 | 134 | **−144** |
| `i16_deltas_inner` (A + B) | 315 | 298 | −17 |
| `i16_deltas_into_inner` (option B) | 240 | 227 | −13 |
| `i8_deltas_into_inner` (option B) | 471 | 403 | **−68** |
| `activate_inner` (option D) | 239 | 99 | **−140** |
| `make_move_with` (option E) | 285 | 215 | **−70** |
| `MovePicker::new` + `next_move` (option C) | 169 + 490 = 659 | 0 + 624 = 624 | −35 |
| `libc memcpy/memset` | 138 | 139 | 0 |
| `current_full` | 156 | 194 | +38 |
| `l1_dot_inner` | 249 | 293 | +44 |
| `at_full` | 252 | 284 | +32 |
| `search::<NonPv>` | 275 | 285 | +10 |
| `SotaContext::push` | (not listed) | 87 | — |
| **total** | **4713** | **4536** | **−177** |

Two honest notes. **Option C's saving is smaller than it looks and is partly a
migration**: the constructor's 2 KB `memset` is gone, but the buffer's cache
lines are now first touched inside `init_captures`/`init_quiets`, which are
inlined into `next_move` — so `next_move` rose 490 → 624 while the pair fell
659 → 624. **Option B's register tile finally shows up in the register mix**:
`i8_deltas_into_inner`'s dynamic zmm share went 43.7 % → **87.4 %** and
`i16_deltas_inner`'s 2.97 % → **70.6 %** (§3.2), which is now *better* than
SF18's 67.0 %.

---

## 3. ASSEMBLY SIDE BY SIDE — the new top five

Method as study 2: `perf script -F ip,sym,symoff` histogram × `objdump -d -C`,
so both the dynamic (sample-weighted) and static (skid-immune) mixes are
available. Helper `$S/opc2.py`.

### 3.1 `MovePicker::next_move` — ours 624 c/n vs SF18 471 c/n (1.33×)

| | ARCHON | SF18 |
|---|---|---|
| static instructions | **2592** | 1763 |
| dynamic register width | gpr 99.87 % | gpr 98.97 % |
| top opcodes | `mov` 44.5, `cmp` 8.2, `add` 7.1, `shl` 5.7, `and` 5.6, `lea` 4.3, `jae` 3.5 | `mov` 24.3, `add` 13.5, `lea` 11.8, `sub` 8.1, `cmp` 7.4, `jle` 6.9, `and` 4.6 |
| hot regions | +0x200 **31.9 %** (entry + `match self.stage`, `movepick.rs:415/417`), +0x2600 20.5 % (quiet scorer, `:659`), +0x0 19.8 % (prologue + tunable materialisation), +0x2a00 8.8 % | +0x800 32.1 %, +0x1200 22.1 %, +0x0 16.9 % |
| BMI2 in the mix | none | **`pext` 0.87 %, `blsr` 1.05 %, `shlx` 0.82 %** |

Source-line attribution against the whole program: `movepick.rs:415` (the
function-entry line) = **4.63 %** and `:417` (`match self.stage`) = **2.24 %**.
That is **6.87 % of all engine cycles, 312 c/n, 170 cycles per call**, spent
before any move is scored. It is not "the match statement is slow": it is a
2.6 KB function reloading a ~2.1 KB object's fields and re-materialising
tunable parameters on every one of 1.84 calls/node — and `next_move` is
**15.26 % of all icache misses** and 5.0 % of all L1←L2 fills.

`mov` at 44.5 % of a 100 %-GPR symbol against SF's 24.3 % is the shape of the
gap. Note SF's `next_move` inlines `pext`-indexed magic lookups; ours has no
BMI2 anywhere (§6.9).

### 3.2 The three fold kernels — near or past reference now

| | `i8_deltas_into_inner` | `i16_deltas_inner` (Finny) | `i16_deltas_into_inner` | SF18 `refresh_cache<1024u>` |
|---|---|---|---|---|
| c/node | **403** | **298** | 227 | 267 |
| c/call | 515 | 997 | 344 | — |
| dynamic zmm | **87.4 %** | **70.6 %** | — | 72.3 % |
| static zmm | 52.9 % | 61.2 % | — | 31.6 % |
| static instrs | 380 | 745 | — | 361 |
| top opcodes | `vpaddw` 37.6, `vmovdqu64` 26.4, `vpsubw` 14.9, `vpmovsxbw` 8.5 | `vpaddw` 34.9, `vmovdqu64` 24.8, **`lea` 19.9**, `vpsubw` 10.9, `add` 8.2 | — | `vpaddw` 34.4, `vmovdqa64` 19.8, `vpsubw` 18.0, `cmp` 8.9 |
| share of all L1←L2 fills | 11.6 % | 5.8 % | 7.1 % | — |

The register mix is no longer the story — **option B closed it, and
`i8_deltas_into_inner` at 87.4 % zmm is now the most vectorised hot symbol in
either engine.** Two residuals remain:

- `i16_deltas_inner` still shows **`lea` 19.9 % + `add` 8.2 %** of dynamic
  samples. Study 2 recorded that this kernel alone never got option F's
  address hoist. With the surrounding scalar noise now gone, that 28 % is much
  harder to dismiss as skid than it was at 2.97 % zmm.
- `vpmovsxbw` at 8.5 % is the i8→i16 widen. SF pays the same tax
  (`vpmovsxbw` 18.6 % of their `evaluate`). Not a differential.

### 3.3 `avx512::l1_dot_inner` — 293 c/n, 717 c/call, and the refuted hypothesis

The hot loop is exactly eight instructions:

```
movzwl (%rdx,%r9,1),%eax          ; nnz index
cmp    $0xbf,%eax  /  ja panic    ; a bounds check, in the hot loop
vpbroadcastd (%rsi,%rax,4),%zmm2  ; the 4-byte FT group
shl    $0x7,%eax
vpdpbusd (%r8,%rax,1),%zmm2,%zmm1 ; chain A
vpdpbusd 0x40(%r8,%rax,1),%zmm2,%zmm0 ; chain B
add/cmp/jne
```

`BLOCKS = L2 / L2_BLK = 32 / 16 = 2` under `arch-d47v`, so there are exactly
**two** accumulators, each a loop-carried `vpdpbusd` chain. On Zen 4 VPDPBUSD is
latency 4 / throughput 0.5, so ~128 iterations ⇒ a 512-cycle latency floor
against a 128-cycle throughput floor. Measured 717 c/call. The hypothesis
wrote itself. **It is wrong** — see §5.1.

**Derived group density.** The AVX2 twin's loop body is exactly 12 instructions
× 2 blocks, plus ~16 of prologue/epilogue; callgrind gives 3095 `Ir`/call, so
**nnz ≈ (3095 − 16)/24 ≈ 128.3 of 192 groups = 66.8 % density.** The head's own
doc comment claims "~85–95 % are zero after the clamp"; at 768 wide, with
4-byte groups, the measured per-lane zero rate implied by 66.8 % group density
is ~76 %, and study 6 measured ~62 % group density on the 1280 net. The
sparsity machinery is real but is skipping only a third of the matmul.

### 3.4 `ThreatContext::at_full` — 284 c/n, 695 c/call, 91 % GPR

| dynamic | value |
|---|---|
| register width | gpr **91.1 %**, xmm 4.5 %, ymm 4.4 %, zmm 0 % |
| top opcodes | `inc` 29.3, `mov` 13.7, `add` 7.2, `movb` 7.0, `cmp` 5.6, `movzbl` 4.0 |
| regions | +0x1400 **50.2 %**, +0x1600 12.6 %, +0x200 7.9 % |

Source lines, as % of the whole program: **`sota_threats.rs:1008` = 1.28 %** and
**`:1012` = 0.58 %** — those two lines are `psqt_add`/`psqt_sub`, a random 32-byte
read from the 1.94 MB threat-PSQ1 table *per resolved event*. **1.86 % of all
engine cycles, 84 c/n, sits on threat-PSQ1 row loads.** Study 1 estimated 31.5 %
of `at_full`; the fresh measurement says 30 %. This is the strongest surviving
case for the net-format rider (i32→i16 buckets halves the row to 16 B).

### 3.5 `Worker::search::<NonPv>` — 285 c/n, 2689 static instructions, 97.6 % GPR

`mov` 26.9, `cmp` 7.6, `lea` 6.6, `movzbl` 5.5, `je` 5.4. No region above
11.8 % — the cost is spread. Against SF18's `search<NodeType 0>` at **530 c/n**
we are **0.54×**, but that is not like-for-like: SF inlines far more per node
(their `search<0>` + `qsearch<0>` = 597 c/n vs our 305). It is 8.20 % of our
icache misses.

---

## 4. THE DIFFERENTIAL TABLE, REBUILT

ARCHON: shares of `d-ours.data` × 4536 c/n. SF18: direct symbols from
`d-sf.data` × 3538 c/n, plus a **freshly measured** 512-byte-region
decomposition of `Network<1024u>::evaluate` (34.61 %, 1224 c/n) — study 2
transplanted study 1's; this one is re-measured:

| region signature | % of `evaluate` | c/n |
|---|---:|---:|
| threat i8→i16 fold (`vpmovsxbw`-dominated: +0x600, +0x1000, +0xe00) | 29.58 | 362 |
| HalfKA i16 fold (`vpaddw`/`vpsubw`/`vmovdqa64`: +0x200, +0x800, +0xa00, +0x2200) | 27.47 | 336 |
| scalar index resolve (`mov`/`lea`: +0x400, +0xc00) | 10.32 | 126 |
| nnz (`vptestmd`/`vpcompressd`/`popcnt`: +0x1a00) | 8.11 | 99 |
| FT activation (`vpminsw`: +0x1200) | 6.26 | 77 |
| L1 matmul (`vpdpbusd`: +0x1c00) | 5.09 | 62 |
| unattributed (smaller regions) | 13.17 | 161 |

Sorted by **absolute cycles/node gap**:

| # | subsystem | ARCHON | SF18 | **gap** | ratio | mechanism |
|---|---|---:|---:|---:|---:|---|
| 1 | **`MovePicker::next_move`** | **624** | 471 | **+153** | **1.33×** | 2592-instr symbol; 6.87 % of program on entry + stage dispatch; no history prefetch |
| 2 | **threat resolution `at_full`** | **284** | 126‡ | **+158** | **2.25×** | reconstructive replay + 8-slot linear override scan + 84 c/n of threat-PSQ1 row loads |
| 3 | **HalfKA lazy driver `current_full`** | **194** | ~25‡ | **+169** | ~8× | per-eval ancestor walk + null-move alias scan; SF's is an index compare |
| 4 | **HalfKA Finny refresh** (driver 134 + fold 298 + copy-out ~35) | **467** | 267 | **+200** | **1.76×** | was 2.50× in study 2; the *fold* is now 63 % of it and barely moved |
| 5 | head math total | 644 | ~400‡ | +244 | 1.6× | of which `l1_dot` 293 (their 62 at 2/3 the MACs), `skip_dot` 95 (no SF counterpart), `activate` 99 vs 77, `l2_tail` 76 |
| 6 | make/unmake + state plumbing | 302 | 194 | +108 | 1.56× | was 1.77×; option E closed a third |
| 7 | **libc `memcpy`/`memset`** | **139** | ~4 | **+135** | ~35× | 0.89 dead-zero calls/node still live (§2.3) |
| 8 | **`SotaContext::push`** | **87** | (in `do_move`) | **≈+87** | — | 16 B + a 144 B snapshot written 3.2 KB ahead in a 1.6 MiB stack pair |
| 9 | threat fold | 403 | 362 | +41 | 1.11× | near parity; ours is *more* vectorised (87.4 % vs 67.0 % zmm) |
| 10 | threat event generation | 214 | 150 | +64 | 1.43× | `record_square_touch` 144 + `ripple` 25 + `refresh_threat` 35 |
| 11 | `correction_value` | 70 | (inlined) | — | — | 6 random reads into 3.2 MiB, once per node, before the TT probe |
| 12 | HalfKA incremental fold | 227 | 336 | **−109** | **0.68×** | **we win** |
| 13 | search recursion | 305 | 597 | **−292** | 0.51× | not like-for-like (SF inlines more) |
| 14 | `see_ge` | 44 | 71 | −27 | 0.62× | we win |
| 15 | TT probe | 35 | (inlined) | — | — | geometry is byte-identical to SF's |
| — | **THREAT SUBTOTAL** | **927** (20.4 %) | ≈696‡ (19.7 %) | +231 | **1.33×** | |
| — | **HALFKA SUBTOTAL** | **859** (18.9 %) | ≈675‡ (19.1 %) | +184 | **1.27×** | |
| — | **ALL FEATURE UPDATE** | **1786** | ≈1371 | +415 | **1.30×** | |
| — | **SHELL** | **1724** (38.0 %) | 1470 (41.6 %) | +254 | **1.17×** | |
| — | small-net eval (SF only) | — | 151 (4.3 %) | — | — | segmented out |

‡ SF's in-`evaluate` scalar index-resolve region (126 c/n) is shared between the
two feature sets; I assign 80 % to threats, as studies 1 and 2 did. Only the
**ALL FEATURE UPDATE** row is assumption-free.

**Reading:**

1. **The gap is no longer concentrated anywhere.** Study 1 had one 4.5× item
   (head math) and one ∞ item (SKP1). Study 2 had one 2.5× item (Finny). Now
   the largest single named gap is **+200 c/n = 4.4 % of our cycles**, and the
   top five gaps together are +924 c/n against a total gap of +998. There is no
   remaining "afternoon fix worth 16 %".
2. **Study 2's #1 gap is half closed.** Finny 621 → 467 c/n, 2.50× → 1.76×.
   Option A did what it promised on the *driver* (278 → 134 c/n, −52 %) and
   almost nothing on the *fold* (315 → 298), which is now 63 % of the item and
   is 5.8 % of all L1←L2 fills. What is left of the Finny gap is a
   memory-latency problem, not an algorithm problem.
3. **Two "driver" items rose into the top three: `at_full` (+158) and
   `current_full` (+169).** Both are pure per-eval bookkeeping — a stack walk
   and a replay — and both got *more* expensive in absolute c/n while
   everything around them shrank. `current_full` at ~8× SF's index compare is
   the worst ratio in the table.
4. **`libc` at 35× is embarrassing and cheap to fix**: 0.893 `memset` calls per
   node for 719 bytes nothing reads, plus 0.604 `memcpy`. Study 2 named the
   sites; the patch was never extended to them.
5. **We win on branch prediction (0.73×), L2/L3/DRAM traffic (0.76–0.82×), the
   L1d miss rate (0.94×), `see_ge` (0.62×), qsearch, and the HalfKA incremental
   fold (0.68×).** Those fronts are closed.

---

## 5. PROTOTYPES — measured, not estimated

All built into `git archive 9542e93` copies under the scratchpad, with the same
`x86-64-v3` flags, gated on the 800-FEN hash, the fixed-node total, **and the
new cadence-free depth-13 total**.

### 5.1 P1 — `l1_dot` four-chain split: **bit-identical, and a null result**

Replaced the two loop-carried `vpdpbusd` chains with four independent
accumulator sets over `nnz.chunks_exact(4)` (i32 wrapping adds are associative
*and* commutative, so any partition of the contribution multiset is exact),
summing at the end. Codegen verified: eight live zmm accumulators, four groups
per iteration, no spills.

| | baseline | **P1** |
|---|---:|---:|
| 800-FEN hash | `0615d75e…c92eb` | **identical** |
| depth-13 total | 2,940,900 | **identical** |
| instructions/node | 6779.6 | 6680.8 (−1.5 %, loop control only) |
| **`l1_dot_inner` cycle share** | **6.46 %** | **6.53 %** |
| NPS (median of 8 interleaved) | 1,151,234 | 1,159,227 (**+0.69 %**) |

**The hypothesis is refuted and the reason is measurable.** A memory-event
profile attributes **20.65 % of all L1←L2 fills** to `l1_dot_inner` — by far the
largest single consumer, ahead of `i8_deltas_into_inner` (11.6 %). The kernel is
**load-bound**, not latency-bound: 128 live groups × 3 loads = 384 loads/eval
streaming a 24 KB per-bucket weight slice out of L2. Adding accumulator chains
adds no load bandwidth.

This kills a whole family of ideas (chain-depth surgery on `l1_dot`) and
redirects to the one lever that *does* move a load-bound kernel: **fewer live
groups**, i.e. FT sparsity (§6.5).

### 5.2 P2 — kill the `RwLock` on the eval path

`sota_context.rs:433` calls `super::active_policy_head()` **once per eval**
(0.4089/node). That accessor is
`*ACTIVE_POLICY_HEAD.read().unwrap_or_else(…)` — a futex-backed
`std::sync::RwLock`, i.e. two locked read-modify-writes and a store-buffer
drain, on every eval, **even when no policy sidecar is loaded**. The slot's own
doc comment (`mod.rs:164-167`) says it is read "once per search at the eval
boundary … never on the hot per-node path". Source-line attribution puts
`futex.rs:111` + `futex.rs:97` inside `SotaContext::evaluate` at **1.53 % of all
cycles**.

Prototype: a relaxed `AtomicPtr` shadow published under the same write lock.

| | baseline | **P2** |
|---|---:|---:|
| 800-FEN hash (nocpol and cpol) | both | **identical** |
| depth-13 total | 2,940,900 | **identical** |
| instructions/node | 6779.6 | 6770.5 (−0.13 %) |
| **cycles/node** (paired) | 4575.9 | **4522.3 (−1.17 %)** |
| NPS (median of 8 interleaved) | 1,160,814 | 1,166,487 (+0.49 %) |

**The instruction count barely moves and the cycles do** — which is the
signature of locked RMWs: ~4 instructions per eval that cost ~130 cycles.
Honest band **+0.5 to +1.2 % NPS**, one hour of work, provably identical.

### 5.3 P3 — `#[repr(align(64))]` on the accumulator entries

`Entry` (`sota_accumulator.rs:79`), `ThreatEntry` (`sota_threats.rs:609`) and
`FinnyEntry` carry only `#[derive(Clone)]`. Under `arch-d47v`, `Entry` is
`[[i16;768];2]` + `[[i32;8];2]` + `DirtyPiece` + `[bool;2]` = **3152 bytes,
align 4**. 3152 mod 64 = 16, so **every ply's accumulator starts 16 bytes
further out of phase than the last**, and the `Vec<Entry>` base is only
4-aligned. Every `vmovdqu64` in the three fold kernels therefore splits a cache
line on 3 plies in 4. SF's accumulator is cache-line aligned by construction.

One attribute on three structs (Rust rounds the size up to the alignment
automatically):

| | baseline | **P3** |
|---|---:|---:|
| 800-FEN hash | | **identical** |
| depth-13 total | 2,940,900 | **identical** |
| instructions/node | 6779.7 | **6780.1 (+0.006 %)** |
| **L1d loads/node** | **3337.6** | **3294.4 (−1.29 %)** |
| L1d misses/node | 570.4 | 564.2 (−1.1 %) |
| NPS (8 ABBA blocks vs baseline) | — | median **+3.60 %**, mean +5.23 %, range −1.8 … +16.9 %, positive in 6/8 |

**An identical instruction stream issuing 1.29 % fewer L1 accesses is exactly
what removing line-splitting looks like** — a split 64-byte load costs two L1
accesses. The NPS band is wide because the box degraded; the deterministic
counter is the evidence and the mechanism is unambiguous. Honest band
**+1 to +4 % NPS**, one line of code, bit- and node-identical.

### 5.4 Build variants — native, and PGO

Both built from `9542e93`; PGO trained on
`bench 16 1 60000 wl50 nodes` + `bench 16 1 8 default depth`, merged with
`llvm-profdata`, rebuilt with `-Cprofile-use`.

| build | instructions/node | L1d loads/node | 800-FEN hash | depth-13 total | NPS vs v3 (8 ABBA blocks) |
|---|---:|---:|---|---|---|
| `x86-64-v3` (baseline) | 6778.8 | 3337.6 | ✅ | 2,940,900 | — |
| `native` | 6741.6 (−0.55 %) | 3273.0 (−1.94 %) | ✅ | 2,940,900 | median **+2.43 %**, 6/8 positive (one −19 % contention outlier) |
| **`native` + PGO** | **5970.0 (−11.9 %)** | **2953.7 (−11.5 %)** | ✅ | 2,940,900 | median **+8.14 %**, mean +8.10 %, range **+4.46 … +12.54 %, 8/8 positive** |

The PGO block is the **tightest and most consistent measurement in the entire
study** — eight consecutive ABBA blocks, every one positive, on a box where the
baseline itself swung 4303–7530 c/n. It is also the only item here that
attacks the icache differential directly (−11.9 % instructions on functions
that are 2.6 KB each).

The native-vs-v3 mechanism is documented in the source
(`sota_simd.rs:110-134`): a v3 build keeps the runtime kernel table and pays a
`LazyLock` deref plus an indirect call *per kernel invocation* with no
cross-kernel inlining; a build that statically carries
`avx512f+avx512bw+avx512vnni` expands `kernel_dispatch!` to a direct call and
inlines the kernels into `SotaHead::forward_into`.

### 5.5 Measured and declined — the eval-laziness thread

The commission seeded "positions where the eval call can be skipped or reused
entirely". I measured the three candidates and all three are closed:

- **TT-eval reuse already exists** at both eval sites
  (`search/mod.rs:2347-2351` and `:4198-4202`, identical two-line guard), plus
  the in-check skip (`:2327-2330`, grandparent's eval inherited) and the
  singular-verification skip (`:2338-2345`).
- **Step 6 runs before the TT cutoff** (deliberately —
  `search/mod.rs:2311-2315`). The waste is bounded by the TT-cut rate, measured
  at **5.2 %** of nodes (SSTATS, 800 FENs at depth 12: 1,429,007 cuts /
  27,444,830 nodes), and most of those hits carry a valid `eval16` so the guard
  already skips `evaluate()`. **Ceiling < 0.1 %.**
- **`correction_value` is computed unconditionally** at `:2247`, before the TT
  probe *and* before the in-check branch, at 70 c/n for six pseudo-random reads
  into a 2 MiB + 1.22 MiB table pair. It looks like dead work on in-check
  nodes — but LMR reads it at `:3350` on in-check nodes too, so it cannot move
  inside `!in_check`. Moving it past the TT cutoff saves 5.2 % of 1.54 % =
  **0.08 %**. Not worth a line.

Evals fire **0.4089 per node**; the theoretical maximum is 1.0. There is no
laziness left to harvest at the call-site level.

---

## 6. PART B — BEYOND SF18

The framing question: what can *we* do that Stockfish's constraints —
portability, a fused single-accumulator eval, and a very conservative codebase
— prevent them from doing? Ranked by measured or derivable value.

### 6.1 B1 — Ship a PGO'd, arch-specialised binary ladder ★ the biggest one

**The advantage.** Stockfish *does* ship an arch ladder, so this is not a
capability they lack — but they do **not** ship PGO'd release binaries built
against a pinned training corpus with a reproducibility statement, and their
Makefile's `profile-build` is an opt-in for people compiling from source. The
v1 campaign already promises a public ledger and a reproducibility statement;
a **pinned PGO profile shipped alongside the binary** turns a build trick into
a provenance artifact, which is squarely on-brand for the release narrative.

**Measurement.** §5.4: +8.14 % median over 8/8 positive ABBA blocks,
−11.9 % instructions/node, −11.5 % L1d loads/node, eval-hash and depth-13-node
identical.

**Ceiling.** +8 % on this Zen 4 box; less on machines without AVX-512 (the
native half of the gain disappears, leaving PGO's ~+5 %).

**Effort.** One day of build system: a `tools/pgo.sh`, a pinned training
invocation, a checked-in `.profdata` or a documented regeneration recipe, and
an arch ladder (`x86-64-v3`, `x86-64-v3+avx512vnni`, plus the portable
fallback).

**What could kill it.** (a) *Reproducibility*: two PGO builds from the same
`.profdata` are deterministic, but regenerating the profile on a different box
can change the binary — so the `.profdata` must be an artifact, not a step.
(b) *The bench trailer*: PGO does not change search here (depth-13 identical),
but that must be re-verified per profile, and it must be re-verified in the
**default (PSQT) build** against 701605. (c) The training corpus must cover
game-shaped search, not just `bench` — a corpus that over-weights endgames
would cold-code the middlegame paths.

**Validation.** 800-FEN hash + depth-13 wl50 total + default-build 701605 +
`cargo test`, then an ABBA NPS block. All four already pass on the prototype.

### 6.2 B2 — 64-byte alignment of the accumulator stacks

**The advantage.** Nothing SF can't do — they already do it — but it is a
place *we* are simply behind, it costs one line, and it interacts with our
wider accumulator (we carry two separate 768-lane stacks where they carry one
fused 1024-lane one, so we do twice as many tile loads per folded ply and pay
line-splitting twice as often).

**Measurement.** §5.3: −1.29 % L1d loads at identical instructions,
NPS median +3.60 % (6/8 positive), bit- and node-identical.

**Ceiling.** +1 to +4 %. **Effort:** minutes. **What could kill it:** the
padding grows each stack by 1.5 % (254 × 48 B = 12 KB per stack per thread) —
irrelevant. Nothing else.

### 6.3 B3 — The in-net policy head as an ordering prior (P-ORD) ★ the real "beyond"

**The advantage is structural and SF does not have it.** Stockfish's net emits
one scalar. Ours emits a 1858-class move distribution off the *same shared
bottleneck* (`policy_head.rs`, CPL1 sidecar, `rows: [i8; 1858 × L2]`, selected-row
dot via `sota_simd::policy_dot`), so a per-move ordering prior costs one i8×f32
row dot on an activation that already exists. There is no cheap way for SF to
manufacture that signal.

**Current state (source-verified).** Two seams:
- **LIVE, no switch**: policy-argmax fills an empty TT-move slot at
  `search/mod.rs:2923-2932` whenever `tt_data.mv == NULL && depth >= 3` and a
  `.cpol` sidecar is loaded. `PolicyHead::best_move` runs a **full
  `generate_legal`** plus one row dot per legal move.
- **DORMANT**: the per-quiet prior, `CPL_ORD_W` (default 0.0),
  `movepick.rs:712-715`, additive in history units.

**Measurement — and a correction to my own first probe.** On **50 FENs at
depth 13** the sweep looked spectacular: base 2,940,900 nodes → `W=100`
2,588,834 = **−12.0 %**. Scanning depths and mixes destroyed that reading:

| workload | W=32 | W=100 |
|---|---:|---:|
| wl50 d9 | +0.87 % | −11.13 % |
| wl50 d11 | +11.00 % | +8.25 % |
| wl50 d13 | −10.53 % | −11.97 % |
| wl50 d15 | +1.75 % | −8.01 % |
| default-FENs d11 | −2.31 % | +6.75 % |
| default-FENs d13 | +7.87 % | +1.89 % |

Fixed-depth node totals on 50 positions are a **±10 % instrument**. Re-run on
all **800 FENs**:

| arm | d10 nodes | Δ | d12 nodes | Δ | d12 fhf | d12 mv@cut | d12 NPS |
|---|---:|---:|---:|---:|---:|---:|---:|
| base (no `.cpol`) | 8,299,011 | — | 27,444,830 | — | 0.8518 | 1.4048 | 1,133,755 |
| `.cpol`, W=0 (argmax only) | 8,392,417 | +1.13 % | 26,758,377 | −2.50 % | 0.8459 | 1.3999 | 1,120,394 |
| `.cpol`, W=32 | 8,173,808 | **−1.51 %** | 26,420,113 | **−3.73 %** | 0.8467 | 1.3952 | 1,109,203 |
| `.cpol`, W=100 | 8,221,541 | −0.93 % | 26,494,412 | −3.46 % | 0.8463 | 1.3966 | 1,085,347 |

And the **cost, isolated at fixed nodes** (median of 3, so tree shape cannot
contaminate it): base 1,166,619 NPS → `.cpol` W=0 **1,139,503 (−2.32 %)** →
W=100 **1,141,670 (−2.14 %)**.

**So the honest picture is:**
- the ordering signal is real but small — `mv@cut` improves consistently
  (1.4048 → 1.3952) and node counts fall 1–4 % at depth, growing with depth;
- `fhf_rate` gets slightly *worse* (0.8518 → 0.8467), so the gain is not
  first-move cutoffs; it is better 2nd/3rd choices and better LMR ordering;
- the **whole 2.2 % NPS tax is the argmax-TT-substitute** — W=0 and W=100 cost
  the same, so the per-quiet prior is essentially free and `best_move`'s
  independent `generate_legal` is the entire bill;
- net effective NPS today, at d12/800: W=32 **+1.6 %**, W=100 **−0.9 %**;
- `W=320` is over-weighted and *costs* nodes.

**The actionable version, and the ceiling.** Make the consumption free:
`best_move` should score the MovePicker's already-generated list instead of
calling `generate_legal` itself (the picker generates at the very next stage
anyway), and the argmax should be skipped when the picker will not reach the
quiet stage. That removes ~2 of the 2.2 %, leaving **−3.7 % nodes at ~0 cost
≈ +3.9 % effective NPS**, with the weight still un-tuned. The head itself gets
better in P1 (V-LegB is the first real policy training since d47p).

**What could kill it.** (a) Node reduction at fixed depth is not Elo —
reordering can prune a good move; only an SPRT decides. (b) The head is trained
on teacher *best moves*, not on search-optimal ordering, and the 2026-07-25
audit found 29.6 % top-1 on quiets but **3.2 % on captures — below random**, so
it must stay quiet-only. (c) `CPL_ORD_W` needs SPSA; the depth-13 wl50 fiasco
above is exactly what tuning it on a small probe would produce.

**Validation.** Fixed-depth node totals on ≥ 800 FENs at two depths (never 50),
`mv@cut` from SSTATS as the mechanism check, fixed-node NPS for the tax, then a
10+0.1 SPRT. This is precisely the campaign's **P3 P-ORD gate**; this study
supplies its instrument and its prior.

### 6.4 B4 — Finish the dead-zero sweep

**Measurement.** §2.3: `MoveList::new`'s `[Move::NULL; 256]` at 0.4844
`memset`/node and `SotaHead::forward_into`'s `ft`/`nnz` at 0.4089/node =
719 B/node of stores nothing reads, inside a 3.06 %-of-cycles libc block.
SF's `MovePicker::moves[]` has no initialiser at all.

**Ceiling.** +1 to +2 %. **Effort.** 1 day — the same typed
`[MaybeUninit<T>; N]` pattern option C established, with the same
`#[deny(unsafe_code)]` exemption and the same debug-only written-slot tracker.
`forward_into`'s buffers are the easier half: `activate_ft` and `find_nnz`
fully write them, and the dead stores survive only because the runtime kernel
table hides that from LLVM — **which means B1's static dispatch may delete them
for free.** Measure B1 first and re-check.

**What could kill it.** UB, not a wrong score. Requires miri or a
debug-assertions full-suite run.

### 6.5 B5 — FT sparsity as a trainer-side NPS lever

**The advantage is ours alone in kind, not in principle**: SF could regularise
their FT too, but we own the trainer, the inference kernel and the deployment
gate in one repo, so we can co-optimise density against measured NPS in one
loop.

**Measurement.** §3.3: group density **66.8 %** (128.3 of 192 groups),
derived deterministically from the AVX2 kernel's exact 12-instruction body and
callgrind's 3095 `Ir`/call. `l1_dot_inner` is 6.46 % of cycles and **20.65 % of
all L1←L2 fills** — and §5.1 proved it is load-bound, so its cost is
**density-linear**.

**Ceiling.** 67 % → 33 % density halves the kernel: **+3.4 % NPS at zero
inference risk**. Study 6 priced the same lever at +12–14 % on the 1280 net
(where `l1_dot` was 34 % of `forward_into`); at 768 the prize is smaller because
the kernel is smaller.

**What could kill it.** The standing ruling is `ft_sparsity` default 0
(2026-07-26), and the frozen-head λ-ramp memory says the ramp applies only if
group-shaped sparsity re-enters. A λ that costs eval quality costs more Elo than
3.4 % NPS buys. **The cheap, non-committal action is the one study 6 already
asked for: add an avg-nnz probe to the net gate so every candidate reports its
density** — density already varies across otherwise-identical nets, so we may be
leaving NPS on the table by selecting nets blind to it.

### 6.6 B6 — The accumulator-stack metadata split

**Measurement.** `SotaContext::push` is **1.91 % / 87 c/n** for writing a 14-byte
`DirtyPiece`, a `[bool; 2]`, and a 144-byte `ThreatSnapshot` — 87 cycles for
~160 bytes. The cause is layout: those fields sit at the **tail** of a 3152-byte
`Entry` and a 3296-byte `ThreatEntry`, so each push touches ~5 cache lines
spaced 3.2 KB from the previous ply's, inside a **1.56 MiB** stack pair that
exceeds Zen 4's 1 MB L2. `current_full` (194 c/n) and `at_full` (284 c/n) then
walk the same strided metadata to find the nearest computed ancestor.

**The fix.** Hoist `dirty` / `computed` / `snapshot` into compact side arrays
(254 × ~160 B ≈ 40 KB, permanently L1/L2-resident) and leave the accumulators
in the big arrays. Add a `last_computed[persp]` index maintained at push/pop so
`current_full` stops scanning (study 2's option G, never done).

**Ceiling.** +1 to +2 %. **Effort.** 1–2 days; the alias/laziness invariants are
subtle and `assert_matches_refresh` is the guard. **Killer:** the null-move
alias loop and the `king_crossed` refresh trigger both read this metadata; a
missed update is a silently wrong eval, not a crash.

Do it **in the same commit as B2** — both are `Entry`/`ThreatEntry` layout.

### 6.7 B7 — Threat-PSQ1 row narrowing (net-format rider)

**Measurement.** §3.4: `sota_threats.rs:1008` + `:1012` = **1.86 % of all
cycles / 84 c/n**, a random 32-byte read into a 1.94 MB table per resolved
event. **Ceiling** ~+1 % if the 8 i32 buckets become i16 (16 B rows, two rows
per line, table halved to 970 KB). **Effort:** a CENN format rev, so it rides
with the trainer, not the engine — the same rider as the long-standing
quant-unification item. **Killer:** i16 buckets must not overflow; needs the
same measured-range argument option D used for the threat lanes (`|lane| ≤ 8191`
debug-asserted, measured max 431).

### 6.8 B8 — Spend the L3/DRAM surplus (and *don't* spend L1)

**Measurement.** §1.3. We use **0.69 GB/s of DRAM on one thread** (~1 % of this
box's bandwidth), 38 L3 fills/node and 44 L2 misses/node — all below SF18 — but
**1.40× their L1←L2 traffic and 2.25× their icache misses**.

What that licenses and forbids:

| idea | verdict | why |
|---|---|---|
| bigger TT | affordable, but **not a perf lever** — the probe is 35 c/n and the geometry is byte-identical to SF's; TT size buys tree quality, not cycles | |
| fatter TT clusters | **decline** — 32 B / 3 entries is already one cache line; wider clusters add L1 traffic, which is the *scarce* resource | |
| bigger history / correction tables | affordable on the memory side; must be justified by Elo, not NPS | |
| deeper prefetch chains | **closed** — F′ (SF's five history prefetches per `do_move`) was implemented, measured flat, reverted; study 1's NNUE-row prefetch measured −2.5 %; `l1_dot`'s own in-source note records −0.3 % for a group prefetch. Three independent negatives. | |
| **shrink code / icache footprint** | **this is the actual opportunity** — 197.5 icache misses/node, 1.4× SF's rate per instruction, on 2.6 KB hot functions. PGO's hot/cold splitting (B1) is the direct fix and measured −11.9 % instructions. | |

**The honest conclusion of this seed idea: the surplus is real but there is
nothing valuable to spend it on. The deficit is at L1 and L0i, and the fix is
B1.**

### 6.9 B9 — Zen 4 ISA we do not use: mostly a decline, with one exception

Full intrinsic census: we use **AVX512F + BW + VNNI**, plus **VBMI2/VL at
256-bit only** (`_mm256_maskz_compress_epi16` in `find_nnz_vbmi2_inner`; there
is no `_mm512_maskz_compress_*` anywhere). Absent crate-wide: **GFNI, BITALG,
VPOPCNTDQ, IFMA, AVX512-BF16, AVX512-FP16**, all gathers/scatters, all
`permutex2var`. And there is **no SIMD at all outside `src/eval/nnue/`** — the
sole exception is `_mm_prefetch` in `tt.rs:601`.

| candidate | verdict | measured basis |
|---|---|---|
| **static AVX-512 dispatch** (the *real* "specialize harder" item) | **TAKE — B1** | +2.4 % native alone; the v3 build pays a `LazyLock` + indirect call per kernel with no cross-kernel inlining (`sota_simd.rs:110-134`) |
| GFNI (`vgf2p8affineqb` for whole-bitboard flips/reversals) | **decline** | no hot consumer. Our mirrors are single XORs (`orient_threats`); there is no bitboard-transpose in any symbol above 0.2 % |
| BITALG / VPOPCNTDQ | **decline** | popcount is scalar `count_ones()` on single u64s; there is no vector of bitboards to popcount |
| IFMA / BF16 / FP16 | **decline** | our arithmetic is i8/i16/i32 + a small f32 tail; no 52-bit or reduced-precision consumer |
| **BMI2 `PEXT` magics** | **decline (record the number)** | Zen 4's PEXT is 3-cycle, and SF's `next_move` shows `pext` 0.87 % of its dynamic mix. But `AttackTables::attacks` is 1.21 %, `attackers_to` 0.78 %, `legal` 0.84 % — and study 2 measured every one of them *cheaper per node than SF18's*. Ceiling < 1 % for a rewrite of the most perft-critical code in the engine. |
| AVX-512 movegen (`vpcompressd` bit iteration, 512-bit `attackers_to`) | **decline** | same block, same ~4 % ceiling, and we already win it |
| AVX-512 partial insertion sort (SF's new `MoveSorter`) | **park** (study 2 option H) | 1.05 % of program by source line; +0.5 % at medium effort |
| 512-bit `find_nnz` compress | **park** | `find_nnz_vbmi2_inner` is 0.49 % / 22 c/n. Halving it is +0.25 % |

**The takeaway the commission asked for is a negative with a number: our
runtime dispatch does let us specialise harder, but every exotic AVX-512 subset
lands on code that is either already faster than Stockfish's or under 1 % of
cycles. The only ISA money is in dispatching the kernels we already have
statically — which is B1.**

### 6.10 B10 — The disagreement head as an allocation signal

Source state: the **EXP-UNC-1 seam is fully built and fully dormant** —
`load_uncertainty_sidecar` (`sota_loader.rs:596-634`, env `D40_UNC_HEAD`, 1056
bytes), `propagate_l3_var` (`sota_head.rs:263-268`), σ̂ returned from the
forward, `SotaContext.last_sigma` with staleness invalidation, and two
consumers in search — `unc_margin_scale` (RFP/futility margin scaling, clamped
[0.7, 1.5]) and `unc_tm_factor` (TM stretch, clamped [0.8, 1.25]) — all
`unwrap_or(0.0)` and skipped entirely when the coefficients are zero. A *true*
disagreement head (predicting `|sigmoid(v) − target|`) exists only as a D48W
trainer-config line; there is no CENN section and no engine consumer.

**As an effective-NPS tool:** the inference cost is one extra 32→1 f32 dot per
eval, i.e. ~the size of `l2_tail`'s scale — call it +0.3 % NPS. Everything else
is tree shape. **The bar is already correct and already set: P5's +30 Elo ET vs
the same build without it.** This study adds nothing except the cost figure and
the observation that the plumbing is genuinely zero-overhead when off.

### 6.11 B11 — What I looked for and did not find

Recorded so the next study does not re-derive them:

| hypothesis | outcome |
|---|---|
| `l1_dot` is VNNI-latency-bound; split the chains | **refuted by prototype** (§5.1): share unmoved, 20.65 % of L1←L2 fills |
| the eval call can be skipped on TT-cut / in-check nodes | **measured, < 0.1 %** (§5.5) |
| `correction_value` is dead work before the TT probe | **measured, 0.08 %** — and LMR needs it on in-check nodes |
| memory-volume compression (smaller weight tables) | dead since study 1; we use 0.82× SF's DRAM and 0.76× their L3 |
| branch prediction | closed since study 2; 19.4 vs SF's 26.6 mispredicts/node |
| software prefetch anywhere | three independent negatives (§6.8) |
| a latent uninitialised read from option C | **investigated and cleared** — the apparent nondeterminism was a parallel session moving `HEAD` (§1.2); memcheck clean, and stack-shifting via env padding does not move any node total |

`skip_dot_inner` (232 c/call for a strictly serial 48-deep f32 FMA chain,
4-cycle latency ⇒ a 192-cycle floor) and `l2_tail_inner` (186 c/call, two
32-deep chains) *are* still latency-shaped, and unlike `l1_dot` they stream
L1-resident data. A 4-accumulator rewrite is worth ~+1.5 % — but it reorders an
f32 reduction, so it is **not** bit-identical and inherits option S2's gate
(integer-cp identity on all 800 FENs, else SPRT). Parked, not rejected.

---

## 7. OPTIONS RANKED

Convention (studies 1 and 2): removing a fraction *f* of cycles multiplies NPS
by `1/(1−f)`; house rule ~1 Elo per 1 % NPS at 10+0.1 (the measured slope has
been 2.1, so these are conservative).

| # | option | gain | effort | risk | validation class |
|---|---|---|---|---|---|
| **1** | **B1 — native + PGO release ladder** | **+8.1 % measured** (8/8 ABBA blocks) | 1 day, build system | low (build reproducibility, not correctness) | **hash + depth-13 + 701605 + node-identical** — all four verified |
| **2** | **B2 — `#[repr(align(64))]` on `Entry`/`ThreatEntry`/`FinnyEntry`** | **+1–4 %** (−1.29 % L1d loads at equal instructions) | minutes | very low | **bit- and node-identical** — verified |
| **3** | **P2 — relaxed-atomic policy-head slot** | **+0.5–1.2 %** (−1.17 % cycles at equal instructions) | 1 hour | very low | **bit- and node-identical** — verified |
| 4 | B4 — finish the dead-zero sweep (`MoveList::new`, `forward_into`) | +1–2 % | 1 day | medium (UB class) | bit + node-identical, plus miri |
| 5 | **B3 — P-ORD with the argmax cost removed** | up to **+4 % effective** (−3.7 % nodes at ~0 tax) | 1–2 days | medium (search behaviour) | fixed-depth nodes on ≥800 FENs, then SPRT |
| 6 | B6 — accumulator metadata split + `last_computed` index | +1–2 % | 1–2 days | medium | bit + node-identical |
| 7 | B5 — FT-sparsity λ (trainer) / avg-nnz in the net gate | up to +3.4 % NPS | probe: hours; λ: a training run | low for the probe, high for λ (eval quality) | net-gate report; λ needs a full heldout + ET arc |
| 8 | `skip_dot` / `l2_tail` chain split | +1–1.5 % | half a day | medium (f32 reorder) | int-cp identity on 800 FENs, else SPRT |
| 9 | B7 — threat-PSQ1 i16 rows | +1 % | next CENN rev | medium | net-format gate |
| ~~10~~ | ~~`i16_deltas_inner` address hoist~~ | **REFUTED 2026-07-27: +0.47% instructions** | done | — | bit-identical verified; instruction count worse |
| — | AVX-512 `MoveSorter`, 512-bit `find_nnz` | +0.5 %, +0.25 % | medium | low | park |

**Item 10 status (2026-07-27): REFUTED. Implemented, proven correct, and measurably WORSE. Parked-and-dead on branch `perf/i16-deltas-hoist`; do not land it.** The hoist chunks the
feature lists at 96 like the `_into_` twin so `feature * ACC` is computed once
instead of once per tile per feature. Correctness is settled: bit-identical by
construction (the per-lane ops are wrapping i16 add/sub and wrapping addition
is associative, so re-associating across chunks cannot move a lane), and
bit-identical by measurement — 6 positions at fixed depth 13 and 6 runs of 4M
nodes reproduce byte-for-byte, and a new test crosses the chunk boundary at
240 adds / 200 subs.

What could not be settled is the only thing the change is for. The box was
running the D48X trainer plus a 4-worker SPSA, which puts about +-9 % of noise
on a fixed-node timing; the paired difference landed at +1.8 % in the hoist's
favour with t = 0.37, which is nothing. Separating a predicted 0.5-1 % from
that noise needs hundreds of paired reps or, far cheaper, an idle box.

**The measurement that killed it.** Wall-clock could not resolve 0.5-1 % under
a loaded box, but instructions-retired can: because the two binaries are
bit-identical they search exactly the same tree, so `perf stat -e instructions:u`
on a fixed 4M-node search is a *load-immune* comparison. It is also nearly
deterministic — two runs of the same binary differed by 58k instructions out of
38.7 billion (0.00015 %). The result:

| binary | nodes | instructions:u |
|---|---|---|
| main (no hoist) | 2,783,484 | 38,673,356,122 / 38,673,413,863 |
| hoisted | 2,783,484 | **38,855,170,276 / 38,855,226,199** |

**The hoist executes 181.8M MORE instructions, +0.47 %.** The chunked
`MaybeUninit` offset table plus its per-tile reloads costs more than the
`feature * ACC` multiplies it removes. Cycles were noise-dominated (27.8-28.5 G
across four runs) and said nothing either way, which is exactly why the
instruction counter was the right instrument.

**Why the prediction was wrong, and the lesson.** The study inferred a cost from
`lea` 19.9 % + `add` 8.2 % of dynamic samples in this kernel. That inference does
not hold for a *load-bound* kernel: the address arithmetic issues in parallel
with the memory accesses it feeds, so those samples are attribution/skid rather
than a real serial cost, and removing the arithmetic buys nothing while the
bookkeeping to remove it costs real instructions. **Dynamic sample share on
cheap ALU ops is not evidence of a removable cost when the surrounding kernel is
waiting on loads** — the same reasoning that already retired the `l1_dot` chain
split in §5.1.

Two things worth carrying forward. First, **fixed-node timing is the right
instrument here and movetime-NPS is not**: because the two binaries are
bit-identical they search exactly the same tree, so elapsed time at a fixed
node count compares pure speed with the search variance removed. The first
attempt used movetime and produced a 10 % "regression" that reversed when the
run order was flipped — pure ordering bias. Second, the reason to park rather
than land is that this is the hottest unsafe kernel in the engine, and perf
items here land with proof of the gain, not just proof of safety. Rerun the
paired fixed-node A/B when the box is idle and land it if it pays.

**Rejected, with the measurement that kills each:**

| option | why |
|---|---|
| `l1_dot` chain split | §5.1: prototyped, bit-identical, **share unmoved**; the kernel is load-bound (20.65 % of L1←L2 fills) |
| eval skipping / TT-cut reordering / `correction_value` hoist | §5.5: measured ceilings 0.08–0.1 % |
| any software prefetch | three independent negatives, including F′ landed-and-reverted |
| GFNI / BITALG / VPOPCNTDQ / IFMA / BF16 | §6.9: no hot consumer above 0.2 % |
| PEXT magics / AVX-512 movegen | §6.9: ~4 % block where we already beat SF18 per node; < 1 % ceiling for the highest-risk code in the engine |
| fatter TT clusters / TT geometry | §6.8: 35 c/n probe, byte-identical design to SF18's, and wider clusters spend the *scarce* resource (L1) |
| weight-table compression | 0.82× SF18's DRAM, 0.76× their L3 |
| threat option T1 (3-primitive driver) | generation is 1.43× SF's on 214 c/n; still the riskiest item available for < 2 % |

---

## 8. RECOMMENDED SEQUENCE

Ordered by (measured gain × confidence) ÷ (effort × risk), with every
provably-identical item before every arguable one.

**Step 1 — B1: the PGO + native release ladder.** +8.1 %, one day, no source
change. *Acceptance:* 800-FEN hash `0615d75e…c92eb`; depth-13 wl50 total
**2,940,900**; default-build depth-6 **701605**; `cargo test` green; an
8-block ABBA NPS median ≥ +5 %. *Ship shape:* `x86-64-v3`,
`x86-64-v3 + avx512vnni`, and a portable fallback, each PGO'd from a **checked-in
`.profdata`** trained on a pinned, game-shaped corpus. *Abort if:* the depth-13
total moves for any arch (then PGO is changing behaviour and needs its own
bench pin, not a free pass).

**Step 2 — B2 + P2 in one commit.** +1.5 to +5 %, an hour of work, both
verified bit- and node-identical. Same acceptance gates as step 1. These are
the cheapest measured wins in three studies.

**Step 3 — re-measure on a quiet box.** After steps 1–2 the engine should be
near **3900–4200 c/n**, i.e. **1.10–1.19× SF18**. *This is the single highest-value
follow-up available to the measurement itself*: every band in §5 is wide because
a GPU trainer and a parallel session held the box. Twenty idle minutes would
turn ±5-point bands into ±1-point ones.

**Step 4 — B4, the dead-zero sweep.** +1–2 %. Do it **after** step 1, because
static dispatch may let LLVM delete `forward_into`'s dead stores for free —
re-measure the callgrind `memset` rate first and only patch what survives.

**Step 5 — B3, the P-ORD gate (campaign P3).** Remove the argmax's independent
`generate_legal`, then run the gate with the instrument this study establishes:
fixed-depth node totals on **≥ 800 FENs at two depths**, `mv@cut` as the
mechanism check, fixed-node NPS for the tax, then a 10+0.1 SPRT with
`CPL_ORD_W` swept. **Do not gate this on 50 positions** — §6.3's table shows a
50-FEN probe swings ±11 %.

**Step 6 — B6 (metadata split + `last_computed`), then the parked items.**

Cumulative, sub-additively: **steps 1–2 ≈ +10 %**, steps 1–4 ≈ **+12 %**,
everything ≈ **+16–18 %**, which would put us at roughly **1.08–1.15× SF18
cycles/node** on this workload — from 1.99× three studies ago.

### 8.1 Interaction with the v1 release campaign

`docs/research/archon-v1-release-campaign.md`'s APPROVED SEQUENCE puts **P4 —
perf study / NPS characterization + PGO/native release build** after **P3 —
search fit** and before **P5 (T11)** and **P6 (full SPSA)**.

1. **The >5 %-NPS SPSA-freeze rule binds on step 1 alone.** PGO+native is
   **+8.1 %**, and steps 1–2 together are ~+10 %. Search parameters tuned at one
   NPS do not transfer. **Therefore: pull B1 and B2 forward — land them before
   the P3 `s4-searchgap` small SPSA, not after it, and certainly before P6.**
   This is a re-ordering request against the approved sequence and needs
   ratification; the alternative is tuning the priors at a NPS the release
   binary will not run at.
2. **Nothing in steps 1–4 changes a fixed-node result.** All four are
   eval-hash-identical and depth-13-node-identical, so P2's fixed-node gates
   (vs SF16 equal-nodes, the headline gate) are unaffected. They **do** change
   equal-time results — that is the point — so any equal-time match in flight
   must be finished or restarted, never straddled.
3. **The bench trailer.** Steps 1 and 2 and P2 touch only `sota`-gated code or
   the build, and must reproduce the default-build **701605** (verified on all
   three prototypes). Step 4 touches `movepick.rs`/`moves.rs`, which are **not**
   feature-gated, so 701605 is a hard gate there. Step 5 changes search and is
   *functional* by definition — it states a new trailer.
4. **B1 changes what "the release binary" means.** P7 names an
   `x86-64-v3 release build`. Shipping only v3 leaves 8 % on the table for every
   AVX-512 user and understates us against CCRL/CEGT, who will run whatever
   binary we publish. Recommend the ladder, and recommend that the ledger row for
   the release net records **which arch + which `.profdata`** produced the
   measured Elo.
5. **B5 collides with a standing ruling** (`ft_sparsity` default 0, 2026-07-26).
   Only the **avg-nnz probe in the net gate** is proposed here; the λ itself is
   a training decision that must clear a value-quality bar, not an NPS bar.
6. **B3 is already P3's P-ORD gate.** This study supplies its prior (−1 to
   −4 % nodes, −2.2 % tax, both fixable) and, more importantly, its
   *instrument* — the campaign should not accept a 50-position screen.

---

## 9. UNCERTAINTIES AND THREATS TO VALIDITY

1. **The box was contended and got worse.** A GPU trainer throughout, and a
   parallel Claude session running a debug binary at 99 % CPU during the last
   block. The same baseline binary measured **4303 – 7530 c/n**. All A/B claims
   are ABBA-balanced or anchored on deterministic counters. §1.3's counter table
   is from the quiet first hour and is the one I would defend; §5.3's and
   §5.4's NPS bands are from the noisy second half and are wide for a reason.
2. **`HEAD` moved mid-session** (§1.2). Two prototypes were rebuilt from the
   pinned `9542e93`; the SF18 numbers, the flat profiles, the callgrind run and
   the counter tables all predate the move or use the pinned tree. If any number
   here looks inconsistent with a later `main`, that is why.
3. **SF's in-`evaluate` split is 80/20-assumed** for the shared scalar
   index-resolve region, as in studies 1 and 2. Only the ALL-FEATURE-UPDATE row
   is assumption-free. The region decomposition itself *is* freshly measured
   this time, unlike study 2's transplant.
4. **`l1_dot`'s SF counterpart is not comparable like-for-like.** Their L1 is
   1024×16 dense-ish at 62 c/n; ours is 768×32 sparse at 293 c/n over 128 live
   groups. Per *live* MAC we are worse, but the shapes differ enough that the
   ratio in §4 row 5 should not be quoted as an efficiency verdict.
5. **The nnz density (66.8 %) is derived, not instrumented.** It comes from
   inverting the AVX2 kernel's exact instruction count against callgrind `Ir`.
   The arithmetic is tight (12 instructions × 2 blocks, verified in the
   disassembly) but a direct probe would be better, and §6.5 asks for one.
6. **Node definitions may differ** between the engines; the 0.06 % agreement
   follows from both honouring `go nodes`.
7. **Position mix.** wl50 is endgame-heavy; §2.4 gives the middlegame shift.
   §4's threat rows are the favourable end for us.
8. **Nothing here was Elo-tested.** Every Elo figure is the house 1 Elo/1 % NPS
   conversion. The measured slope has been 2.1, so these are conservative
   conversions, not results.
9. **B3's node-reduction figure is from two depths on 800 positions.** It is
   ~16× better sampled than the 50-FEN probe that produced −12 %, but it is
   still a fixed-depth proxy, and the depth trend (−1.5 % at d10, −3.7 % at d12)
   suggests the number keeps moving with depth. The SPRT is the answer.

---

## 10. ARTIFACTS

Session-local, under
`/tmp/claude-1000/-home-nick-projects-chessengine/61b8a705-.../scratchpad`:

- `wl50.fens` (`awk 'NR%16==1' tools/nnue/calib/scale-check-800.fens`),
  `f800.fens` (the full battery)
- `d-ours.data`, `d-sf.data`, `d-def.data` (wl50 + SF18 + middlegame flat
  profiles), `pprof.data` (line-tables build), `d-p1.data` (prototype P1)
- `l2f.data` (L1←L2 fill attribution), `icm.data` (icache-miss attribution),
  `mem-ours.csv` / `mem-sf.csv` (Zen 4 memory counters)
- `cg-ours.out` (callgrind, 1,000,761 nodes, AVX2 path), `mc.log` (memcheck)
- `ps-*.csv` (every `perf stat` run), `pord-*.txt` / `p2-*.txt` (the P-ORD
  sweeps with SSTATS)
- `pr0/` (pristine `9542e93`), `pr1/` (P1 l1_dot chains), `pr2/` (P2 atomic
  policy slot), `pr3/` (P3 alignment); target dirs `tg-v3`, `tg-prof`, `tg-def`,
  `tg-p1`, `tg-p2`, `tg-p3b`, `tg-nat`, `tg-pg2gen`, `tg-pg2use`;
  `pgo2.profdata`
- `opc2.py` (opcode / register-width / 512 B-region mix), `pstat.sh`,
  `pord.sh`, `pord2.sh`

**Reproduction one-liners.**

```bash
# the four gates
./binary nnue-eval /tmp/d47v-sb100-nocpol.cenn < tools/nnue/calib/scale-check-800.fens | sha256sum
#   -> 0615d75e5c9487b6f6c66255d3c1d67789f46471d4c8eaec41580e3c070c92eb
taskset -c 4 ./binary bench 16 1 300000 $S/wl50.fens nodes /tmp/d47v-sb100-nocpol.cenn
#   -> Nodes searched  : 15001555      (±10; cadence-sensitive)
taskset -c 4 ./binary bench 16 1 13     $S/wl50.fens depth /tmp/d47v-sb100-nocpol.cenn
#   -> Nodes searched  : 2940900       (exact; the identity gate to use)
taskset -c 4 ./target/release/chessengine bench 16 1 6 default depth
#   -> Nodes searched  : 701605

# the ordering instrument
CPL_ORD_W=32 ARCHON_SEARCH_STATS=1 taskset -c 4 ./binary \
  bench 16 1 12 tools/nnue/calib/scale-check-800.fens depth /tmp/d47v-sb100.cenn 2>&1 \
  | grep '^SSTATS'          # aggregate fh / fhf / avg_moves_at_cut over 800 lines

# the PGO ladder
RUSTFLAGS="-C target-cpu=native -Cprofile-generate=$S/pgo" cargo build --release --features sota,arch-d47v
<train>; llvm-profdata merge -o $S/pgo.profdata $S/pgo/*.profraw
RUSTFLAGS="-C target-cpu=native -Cprofile-use=$S/pgo.profdata" cargo build --release --features sota,arch-d47v
```
