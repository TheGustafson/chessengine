# Perf study 67 — the search side is FRONTEND-bound, the shipped build was formatting a move per qsearch node, and function size is not the instruction-cache cost

**Date:** 2026-08-01 · **Box:** 7700X (Zen 4), AVX-512 VBMI2, SMT on, **32 KB
L1i**, 32 KB L1d, 1 MB L2/core, 32 MB L3, 15 GB · **Base:** `main` at
**`42c339cf`** (post-study-66 handoff) · **Net/arch:** D48X @
`network_scale=600`, `sota,arch-d48x`, ACC 1280 · **Artifacts:**
`docs/research/perf-artifacts-67/` · **Instruments:** `tools/perf67/` ·
**Branch:** `worktree-agent-aa5fcc264b207c816`

**Frame, quoted with every number below:**

```
bench 16 1 100000 tools/nnue/calib/wl50.fens nodes engines/canonical/net.cenn
```

50 FENs × 100 k nodes = **5,001,560 nodes**, one thread, 16 MB TT, pinned to
core 2. **Every binary profiled or panelled here is PGO** — built by
`tools/build_pgo.sh sota,arch-d48x <name> engines/canonical/net.cenn` with the
eval-file argument, because a bench-only profile leaves the NNUE kernels
unprofiled and PGO inert, and because a non-PGO profile inlines differently and
has misled this project before. A GPU training run and a continuous 1500-game
match shared the box throughout; **no `cycles:u` appears anywhere in this
document** (studies 43/63's standing refusal), and every speed claim is a
within-round paired median with its sign count against a **byte-identical
binary null**.

---

## ★ 0. WHAT LANDED, AND THE THREE ANSWERS

### 0.1 Landed and gate-green

`CHECK_NNUE_GATE=1 CHECK_TEST_THREADS=2 bash tools/check.sh` → exit 0,
`Bench: 486897`, eval hash `2e1ede95…79c4`, d13 `1743308` — **all three pins
unmoved** (§7).

1. **`qtrace` takes a `Move`, not a `&str`.** The one landed strength change,
   and the study's headline: the shipped engine was building a heap `String`
   for a trace function compiled to `{}`. **−1.259 % instructions (14/14),
   −5.087 % L1i line fills (12/14), −2.168 % task-clock (11/14)**, node
   count and eval hash exact. §2.
2. **`tt_verify_child` and `check_time_slow`** — two extractions that are
   codegen-neutral without `icold67` (`.text` 777,013 either way) and read
   better as functions.
3. **`tools/perf67/`** — twelve scripts, four of which are instruments this
   project did not have: a no-skid **dispatched-op** IBS profile
   (`ibsops67.sh` + `ibsagg67.py`), an **inline-callee byte census** that opens
   a PGO symbol (`inlcensus67.py`), a **duplication census** across
   monomorphisations (`dupcensus67.py`), and a **code coldness map**
   (`coldmap67.py`). §1.
4. **Two measurement features, both default OFF and both recorded as losses
   with their numbers in `Cargo.toml`:** `icold67` (§4) and `icold67b` (§5).

### 0.2 The re-derived profile — deliverable 2

**The standing list's dead symbols are confirmed dead:** `nm -C` on the current
PGO binary finds **zero** occurrences of `i16_deltas_into` or `i16_deltas_dual`.
Study 32's i8 conversion replaced them and the profile they were quoted from is
stale. The live list is §1.2, and **the search side owns 49.4 % of instructions,
64.7 % of branch mispredicts and 58.9 % of L1i line fills.**

### 0.3 The three one-line answers

> **Is the search memory-, branch- or compute-bound? It is FRONTEND-bound, and
> nobody here had measured that.** 21.09 % of all dispatch slots are lost to the
> frontend; the op queue is **completely empty for 445.7 cycles per node**
> (11.5 % of the frame); the engine takes **128.0 L1i line fills per node** —
> 8.2 KB of instruction bytes per node — and **40.5 % of its ops are delivered by
> the DECODER rather than the op cache.** `search::<NonPv>` alone is 34.06 % of
> those fills and is **45,565 bytes against a 32 KB L1i**. §1.3.

> **The largest single defect found: the shipped build ran
> `Move::to_string()` — heap allocation, `Display` chain, drop — once per
> qsearch move, to pass it to a function whose body is `{}`.** 2,171 B and 3.6 %
> of `qsearch::<NonPv>`'s dispatched ops, twice over, plus `malloc`,
> `<Square as Display>::fmt` and `<char as Display>::fmt`, which vanish from the
> profile entirely when it is fixed. §2.

> **And the negative that prices the whole route: FUNCTION SIZE IS NOT THE
> INSTRUCTION-CACHE COST — the number of distinct lines the frame EXECUTES is.**
> Cutting `search::<NonPv>` by 20.9 % by outlining its coldest 9.5 KB *raised*
> L1i fills **+3.2 %** (§4). Deduplicating its *hot* duplicated callees lowered
> them **−1.5 %** (§5) — the mechanism split is real — but cost 45 instructions
> per fill removed against a measured ceiling of **~5**. §6 states that as a
> pre-registerable rule.

---

## ★★★ 1. THE RE-DERIVED PROFILE

### 1.1 The instruments, and why the existing ones could not do this

| tool | what it does | why it was needed |
|---|---|---|
| `ibsops67.sh` | IBS op sampling with **`cnt_ctl=0`** — the sampling interval counted in **dispatched ops**, not cycles | Studies 59/64 sample IBS with `cnt_ctl=1`, which counts CYCLES between samples and produces a *latency* profile. `instructions:u` per-INSTRUCTION is not believable on this box (study 36 §2.3). Nothing here could answer *"how many of this function's bytes does the frame execute?"* |
| `inlcensus67.py` | resolves **every instruction** of a symbol through `addr2line -i` and rolls the byte budget up by the **outermost inlined callee** | PGO fuses dozens of callees into `search::<NonPv>`; `nm` reports one 45 KB symbol and study 33 could only say "32.87 % of it is search's own code". This gives the whole table. |
| `dupcensus67.py` | the same rollup **summed across symbols** | `do_move` is one function in the source and the machine holds **three copies**. No instrument in this series could see that. |
| `coldmap67.py` | never-sampled 64-byte code lines and their contiguous runs | the outlining candidate list |
| `srcmap67.py` | code lines → source, with inline frames | |
| `startup67.sh` | the same counters at **1 node per FEN** | the startup constant to subtract (memory note: half a headline can be startup) |

**The line-tables binary is a valid proxy.** `addr2line` needs DWARF, which the
release profile omits. `s67-dbg` is the same PGO pipeline with
`CARGO_PROFILE_RELEASE_DEBUG=line-tables-only`; its `search::<NonPv>` is the
**same 0xb1fd bytes** and disassembles to the **same 9,280 instructions with
zero mnemonic differences** (341 differ only in encoded bytes, all relocations —
`.text` moved 32 bytes). Proved, not assumed.

**★ An instrument bug of my own, caught mid-study and corrected.** All four
tools matched symbols by SUBSTRING, and
`search::<chessengine::search::NonPv>` **is a substring of**
`qsearch::<chessengine::search::NonPv>` — so 18,451 qsearch samples were folded
into the search symbol's histogram. Every within-symbol share in this document
is re-derived with exact matching. The corrections all moved the *same way*
(the cold set is colder than first reported): `search::<NonPv>` 26.33 % →
**23.04 %** of engine ops, never-sampled 15.7 % → **24.2 %** of the symbol,
`make_move_with<TtPrefetch>` 0.2 % → **0.1 %** of its ops. No conclusion
changes. The four tools now take an exact name (trailing `*` for a deliberate
prefix).

### 1.2 The live hot list — `s67-base`, PGO, whole frame

Per-symbol `instructions:u` shares are the attribution that survives its own
null on this box (study 58 §8). `perf record -c 2000003`, `--percent-limit 0.15`
(`perf-artifacts-67/prof-*.txt`).

| symbol | **instructions:u** | **branch-misses** | **`ic_cache_fill_l2`** | IBS µops |
|---|---:|---:|---:|---:|
| **`search::<NonPv>`** | **30.87 %** | **37.05 %** | **34.06 %** | 23.04 % |
| `avx512::i8_deltas_into_inner` | 15.07 % | 7.72 % | 9.90 % | 23.36 % |
| `SotaHead::forward_into` | 12.01 % | 3.68 % | 7.09 % | 10.72 % |
| **`MovePicker::init_quiets`** | **9.84 %** | **15.89 %** | **8.44 %** | 7.47 % |
| `ThreatContext::at_full` | 6.38 % | 7.42 % | 7.19 % | 5.19 % |
| **`qsearch::<NonPv>`** | 3.36 % | 2.86 % | **6.92 %** | 3.29 % |
| `HalfkaContext::finny_refresh_into` | 2.89 % | 4.76 % | 4.71 % | 4.87 % |
| `sota_threats::record_square_touch` | 2.85 % | 3.54 % | 3.53 % | 3.00 % |
| `HalfkaContext::current_full` | 2.35 % | 1.31 % | 2.48 % | 4.22 % |
| `avx512::i8_deltas_into_generic` | 2.01 % | 1.03 % | 0.80 % | 2.48 % |
| `sota_loader::load_v2_bytes` *(startup)* | 1.69 % | — | — | 2.61 % |
| `MovePicker::init_captures` | 1.27 % | 2.63 % | 2.89 % | 1.32 % |
| `Board::see_ge` | 1.18 % | 1.71 % | 1.30 % | 0.92 % |
| `attacks::init_magics` *(startup)* | 1.06 % | 0.42 % | — | 0.48 % |
| `avx512::i8_deltas_dual_generic` | 0.94 % | 1.31 % | 0.75 % | 1.85 % |
| `MovePicker::init_evasions` | 0.71 % | 1.31 % | 0.88 % | 0.59 % |
| `Worker::do_move` *(out-of-line copy)* | 0.66 % | 0.99 % | 1.83 % | 0.46 % |
| `search::<Pv>` | 0.61 % | 1.36 % | 1.14 % | 0.66 % |
| `Histories::update_continuation_histories` | 0.52 % | 0.28 % | 0.60 % | 0.29 % |
| **`malloc`** | **0.17 %** | **0.34 %** | **0.57 %** | — |
| **`<Square as Display>::fmt`** | **0.33 %** | **0.34 %** | **0.54 %** | — |
| **`<char as Display>::fmt`** | **0.30 %** | **0.21 %** | **0.31 %** | — |

The last column is **dispatched micro-ops**, not instructions: Zen 4
double-pumps 512-bit instructions, so the SIMD kernels carry ~2 µops per
instruction and the scalar search ~1. That is why `i8_deltas_into_inner` reads
15.07 % of one and 23.36 % of the other. The µop column is the right denominator
for a frontend question (§1.3) and the wrong one for an instruction count; they
are never mixed below.

**Search-side totals** (the search, qsearch, movepick, `see_ge`, `do_move`,
movegen and history-update rows):

| | share |
|---|---:|
| `instructions:u` | **49.40 %** |
| **branch-misses** | **64.66 %** |
| **`ic_cache_fill_l2`** | **58.87 %** |

**The last three rows of the symbol table are §2.** They are the trace's
formatting, and no other caller in this frame reaches them.

**Startup is 3.23 % and can be subtracted.** The 1-node frame runs
873,241,971 instructions against the frame's 27,047,443,172 — but only
**479,179 of 796,040,204 instruction-cache misses (0.06 %)**. The frontend
numbers below are per-node search work, not the 138 MB net load.

### 1.3 ★★★ THE ANSWER TO "MEMORY, BRANCH OR COMPUTE": FRONTEND

Startup subtracted, over 5,001,560 nodes:

| counter | per node | note |
|---|---:|---|
| `instructions:u` | 5,233.2 | |
| `cycles:u` *(quoted once, for IPC only)* | 3,864.9 | IPC **1.354** |
| `branches:u` | 521.1 | |
| `branch-misses:u` | 21.72 | 4.17 % miss rate; ~325 cycles/node **if** the penalty is ~15 cy — an ASSUMED figure, not measured here = 8.4 % |
| `L1-dcache-loads` | 2,890.3 | |
| `L1-dcache-load-misses` | 548.2 | 19.0 % — the memory-bound half studies 39–66 attacked |
| **`ic_tag_hit_miss.instruction_cache_miss`** | **159.1** | **30.6 % of all L1i tag lookups miss** |
| **`ic_cache_fill_l2`** | **128.0** | **8.2 KB of code fetched per node from L2** |
| `ic_cache_fill_sys` | 5.13 | the L3/DRAM instruction tail |
| **`de_op_queue_empty`** | **445.7 cycles** | **11.53 % of the frame with NOTHING to dispatch** |

And the dispatch-slot split (6 slots/cycle on Zen 4):

| | share of all dispatch slots |
|---|---:|
| **lost to the FRONTEND** (`no_ops_from_frontend`) | **21.09 %** |
| lost to the backend (`backend_stalls`) | 48.02 % |
| **ops delivered by the DECODER, not the op cache** (`de_src_op_disp`) | **40.5 %** |
| op-cache miss rate | 40.2 % |

> ### The mechanism, measured rather than argued
>
> The engine's **hot code working set is 120.3 KB** — the number of distinct
> 64-byte code lines holding 90 % of its dispatched ops (6,030 lines = 376.9 KB
> are touched at all). **The L1i is 32 KB.** 80 % of ops still need 72.8 KB;
> only the top 50 % fits, at 20.1 KB. (The tail of this curve includes the
> ~3 % of ops that are process startup — magic init, the 138 MB net load — so
> the 90 % and 95 % rows are upper bounds on the per-node working set. The 50 %
> and 80 % rows are not: startup code cannot reach that far up the histogram.)
>
> | share of engine ops | code needed |
> |---|---:|
> | 50 % | 20.1 KB |
> | 80 % | 72.8 KB |
> | **90 %** | **120.3 KB** |
> | 95 % | 167.1 KB |
>
> 128 L1i fills/node at ~14 cycles of L2 latency is 1,792 cycles/node of raw
> fetch latency; the measured 445.7 implies an average of ~4 instruction fetches
> in flight, which is the right order for a branch-predictor-driven fetcher.
> **I did not measure that overlap and I am not offering the arithmetic as
> evidence** — it is quoted only to show that the two independent counters are
> mutually possible. What is measured is each of them, and neither had been
> measured in this project before.

The search's own symbols are why:

| symbol | bytes | of L1i fills |
|---|---:|---:|
| `search::<NonPv>` | **45,565** | 34.06 % |
| `avx512::i8_deltas_into_inner` | 33,126 | 9.90 % |
| `iterative_deepening` | 26,324 | — |
| `qsearch::<Pv>` | 23,041 | — |
| `qsearch::<NonPv>` | 22,725 | 6.92 % |
| `search::<Pv>` | 20,326 | 1.14 % |
| `finny_refresh_into` | 18,799 | 4.71 % |

**One function is larger than the entire L1 instruction cache.**

Note the contrast that makes this a *code* problem and not a size superstition:
`i8_deltas_into_inner` is **33 KB and 23.36 % of ops for only 9.90 % of fills** —
a big function with excellent code reuse. `qsearch::<NonPv>` is **22.7 KB and
3.29 % of ops for 6.92 % of fills** — an ops-per-fill ratio of 0.475, the worst
of any symbol above 3 % of ops and 2.1× worse than the engine average of 1.0.
Size alone predicts nothing; §4 and §5 turn that into a rule.

### 1.4 Inside `search::<NonPv>` — the byte budget nobody had

`inlcensus67.py`, exact symbol match, 129,389 IBS op samples
(`perf-artifacts-67/inlcensus-nonpv.txt`). `%B` is the share of the symbol's
45,565 bytes; `%ops` the share of its dispatched ops. **The ratio of the two is
the outlining criterion.**

| bytes | %B | %ops | outermost inlined callee | B/op ratio |
|---:|---:|---:|---|---:|
| 9,390 | 20.6 % | 32.1 % | *(search's own code)* | 0.64 |
| 5,467 | 12.0 % | 17.3 % | `do_move` | 0.69 |
| 4,786 | 10.5 % | 6.9 % | `next_move` | 1.52 |
| 2,894 | 6.4 % | 4.0 % | `update_all_stats` | 1.60 |
| **2,679** | **5.9 %** | **0.1 %** | **`make_move_with<TtPrefetch>`** | **59** |
| 2,605 | 5.7 % | 1.9 % | `evaluate` | 3.0 |
| 1,609 | 3.5 % | 3.5 % | `legal` | 1.00 |
| 1,424 | 3.1 % | 1.1 % | `MovePicker::new` | 2.8 |
| 1,223 | 2.7 % | 2.6 % | `see_ge` | 1.04 |
| 803 | 1.8 % | 1.2 % | `upcoming_repetition` | 1.5 |
| 503 | 1.1 % | 6.0 % | `TranspositionTable::probe` | 0.18 |
| 495 | 1.1 % | 0.2 % | `check_time` | 5.5 |
| 441 | 1.0 % | **0.0 %** | `gives_check` | ∞ |
| 400 / 284 / 237 / 288 / 88 | | **0.0 %** | `unmake_move`, `make_null_move`, `is_shuffling`, `last_l2_for`, `unmake_null_move` | ∞ |

**`make_move_with<TtPrefetch>` is a second whole copy of make-move** — 2,679
bytes for **0.1 %** of the symbol's ops — inlined for the N6b child-TT
verification probe. It is 59× the symbol's average bytes per op, and with
`gives_check` and `unmake_move` alongside it the probe is **3,520 bytes**. That
is arm A's headline candidate, and §4 is what happened to it.

`coldmap67.py`: **172 of the symbol's 712 code lines (24.2 %, 11,008 bytes) collect
zero samples in the whole 5 M-node frame**, 162 of them in contiguous runs of ≥2
lines; **90 % of its ops live in 246 lines = 15,744 bytes (34.6 % of it)**.

### 1.5 The duplication census

`dupcensus67.py` over the four search monomorphisations (111,657 B in total):

| total B | copies | callee |
|---:|---:|---|
| 21,482 | 4 | *(each symbol's own code)* |
| **16,638** | **3** | **`do_move`** |
| **10,871** | **4** | **`next_move`** |
| 7,614 | 3 | `evaluate` |
| 6,152 | 3 | `see_ge` |
| **4,355** | **2** | **`to_string<Move>`** ← §2 |
| 4,187 | 3 | `MovePicker::new` |
| 3,708 | 4 | `legal` |
| 2,894 | 1 | `update_all_stats` |
| 2,679 | 1 | `make_move_with<TtPrefetch>` |

---

## ★★★ 2. THE LANDED WIN: A DEBUG TRACE THAT ALLOCATED IN THE SHIPPED BUILD

### 2.1 The defect

`qtrace` is the feature-gated qsearch decision trace. Without
`--features search-control` its body is `{}` and its doc comment says
*"compiled out of every shipped build"*. It took `mv: &str`, and all four
move-carrying call sites passed **`&m.to_string()`**.

**Rust evaluates arguments before the call.** An empty body removes the *use*,
not the *work*. The shipped engine ran the whole `Display` chain, allocated a
`String` on the heap, and dropped it — once for every move qsearch searched or
pruned at three of its gates.

`inlcensus67.py` on `qsearch::<NonPv>` (22,725 B, exact match, uncontaminated):

| bytes | %B | %ops | callee |
|---:|---:|---:|---|
| 5,540 | 24.4 % | 11.6 % | `do_move` |
| 2,490 | 11.0 % | 8.1 % | `evaluate` |
| 2,488 | 10.9 % | 5.4 % | `see_ge` |
| 2,392 | 10.5 % | 22.1 % | *(qsearch's own code)* |
| **2,171** | **9.6 %** | **3.6 %** | **`to_string<Move>`** |
| … | | | |
| **112** | 0.5 % | **0.7 %** | **`drop_glue<String>`** |

**10.0 % of the symbol's code and 4.3 % of its dispatched ops**, and the same
2,184 bytes again in `qsearch::<Pv>`. `malloc`,
`<Square as Display>::fmt` and `<char as Display>::fmt` are all in the frame's
profile (§1.2) and this is their only caller in it.

### 2.2 The fix

`qtrace` takes the `Move` **by value** (2 bytes, `Copy`) and formats inside the
enabled arm; `Move::NULL` carries what the `"-"` literal did. The disabled arm
is `#[inline(always)] fn qtrace(_: usize, _: Value, _: Value, _: Value, _: Move,
_: &str) {}` — six register-resident values with no side effects, which LLVM
deletes entirely. **This is the idiomatic Rust point the codebase's own
conventions imply: a zero-cost debug hook must not take a type whose
construction is the cost.** (A `macro_rules!` that expands to nothing would be
the belt-and-braces spelling; taking the `Copy` value is sufficient here and
keeps the enabled arm a normal function.)

### 2.3 What it removed, in the binary

| | `s67-b0` (base) | `s67-qt` | Δ |
|---|---:|---:|---:|
| `qsearch::<NonPv>` | 22,725 B | **20,123 B** | **−11.5 %** |
| `qsearch::<Pv>` | 23,041 B | **20,337 B** | **−11.7 %** |
| `search::<NonPv>` | 45,565 B | 45,565 B | 0 (qtrace is qsearch-only) |
| `.text` | 777,013 B | 771,205 B | −0.75 % |

and `malloc`, `<Square as Display>::fmt` and `<char as Display>::fmt` **drop off
the `instructions:u` profile entirely** (below the 0.15 % cut, from 0.17 / 0.33 /
0.30 %).

### 2.4 ★ The panel — 14 shuffled rounds

Reference `s67-b0`, the same tree without the fix. Within-round paired medians
with the count of rounds in which the delta was negative
(`perf-artifacts-67/panel-main.csv`):

| counter | base median | **`s67-qt`** | sign |
|---|---:|---:|---:|
| **`instructions:u`** | 27,054,344,721 | **−1.259 %** | **14/14** |
| **`ic_cache_fill_l2`** | 658,042,488 | **−5.087 %** | 12/14 |
| `de_op_queue_empty` | 2,446,184,890 | −6.674 % | 11/14 |
| **`task-clock`** | 4,206 ms | **−2.168 %** | 11/14 |
| `branch-misses:u` | 110,271,376 | +0.386 % | 5/14 |

**Node count 5,001,560 and eval hash `2e1ede95…79c4` — exact, on both sides.**

**The counter that decides it is `instructions:u`**: it is exact at fixed nodes,
it has a measured 0.000 % null (§3), and it is layout-independent. −1.259 % is
**−68.1 instructions per node** (and −6.69 L1i fills per node), and it is 14/14.

**Honest caveat on the fill counter.** Removing 5,808 bytes of `.text` shifts
every downstream symbol's address and therefore its L1i set mapping, so part of
−5.087 % could be a layout lottery rather than the deleted fetches. Two pieces
of evidence say it is mostly not: (a) the instruction drop proves the removed
code was *executed*, so its fetches were real; (b) §4's arm shrinks `.text` by
**1.16 %** and moves the fill counter the **wrong way** (+3.2 %) while §5's
shrinks it **3.79 %** for only −1.5 % of fills — `.text` size by itself does not
drive this counter in either direction, and does not even fix its sign.

### 2.5 What it is worth

−2.168 % of a 4,206 ms frame is −91.2 ms. **Measured**, not assumed: the 1-node
frame puts unchanged process startup at **289.7 ms**, so the change is
**−2.33 % of the 3,916 ms of actual search**. At this project's own calibration
constant (**~120 Elo per doubling**, exp-0154) that is a derived **+4.1 Elo**.
Derived, not measured: **games arbitrate**, and this change has not played any.

---

## ★★★ 3. THE NULL — AND THE PGO PIPELINE IS DETERMINISTIC

Rule 1 says the null comes first. It is stronger here than expected:

> **`s67-qt2` — a full independent rerun of `tools/build_pgo.sh` on the same
> source — is BYTE-IDENTICAL to `s67-qt`.** `cmp` clean; `.text` md5
> `807770636a4ab5746fc10f5c447f991e` on both.

So the study's null is a byte-identical binary under a second name (study 47's
method), and it also establishes that **build-to-build variance is exactly zero**
— every difference between arms below is real codegen, not PGO noise. Both
panels carried it:

| counter | panel 1 (14 rounds) | panel 2 (14 rounds) |
|---|---:|---:|
| `instructions:u` | −0.000 % (10/14) | +0.000 % (7/14) |
| `branch-misses:u` | −0.217 % (8/14) | +0.170 % (6/14) |
| **`ic_cache_fill_l2`** | **+0.467 % (6/14)** | **+0.536 % (5/14)** |
| `de_op_queue_empty` | −1.479 % (8/14) | −0.991 % (8/14) |
| `task-clock` | −0.758 % (9/14) | +0.183 % (7/14) |

**`ic_cache_fill_l2` has a ~0.5 % null at a coin-flip sign count** — usable for
effects of a few per cent, which is what the arms produce. `de_op_queue_empty`
has a ~1–1.5 % null and `task-clock` up to 0.76 %; both are quoted only as
paired medians with sign counts, never as levels.

---

## ★★★ 4. ARM A (`icold67`) — COLD-PATH OUTLINING. IT LOSES.

### 4.1 What it is

`#[inline(never)]` (three of them also `#[cold]`) on the eight callees with the
worst bytes-per-op ratio in §1.4: `tt_verify_child` (a genuine extraction of the
N6b probe — 3,520 B / 0.1 %), `evaluate` (2,605 / 1.9 %), `update_all_stats`
(2,894 / 4.0 %), `check_time_slow` (495 / 0.2 %), `new_probcut`,
`make_null_move`, `unmake_null_move`, `is_shuffling` (237 / 0.0 %). It
deliberately inverts study 55's `#[inline(always)]` ruling on two of them —
that ruling was made on `instructions:u`, which is blind to code size.

It does what it says to the binary:

| | `s67-qt` | `s67-qtc` | Δ |
|---|---:|---:|---:|
| `search::<NonPv>` | 45,565 B | **36,033 B** | **−20.9 %** |
| `qsearch::<NonPv>` | 20,123 B | 18,643 B | −7.4 % |
| `.text` | 771,205 B | 762,245 B | −1.16 % |

### 4.2 And it loses, in two independent panels

Reference `s67-qt`, within-round paired medians:

| counter | panel 1 | panel 2 | null |
|---|---:|---:|---:|
| `instructions:u` | **+0.385 % (0/14)** | **+0.385 % (0/14)** | ±0.000 % |
| **`ic_cache_fill_l2`** | **+3.203 % (3/14)** | **+3.315 % (2/14)** | +0.5 % (coin flip) |
| `branch-misses:u` | −0.393 % (8/14) | −0.392 % (11/14) | ±0.2 % |
| `de_op_queue_empty` | +1.298 % (6/14) | +0.276 % (5/14) | −1.0 to −1.5 % |
| `task-clock` | +1.078 % (6/14) | +0.154 % (7/14) | ±0.8 % |

The instruction cost reproduces **to three decimal places across two panels at
0/14 both times**, and the fill counter is outside its null in the wrong
direction in both.

### 4.3 ★ The mechanism

> **The bytes it removes were never FETCHED.** A 64-byte line with zero sampled
> ops over 5 M nodes does not enter L1i in the first place — the fetch
> granularity is the line, and a line nobody executes is a line nobody fetches.
> Deleting it from the function removes address *span*, not fills.
>
> And two of the eight were not cold at all. `evaluate` (1.9 % of the symbol's
> ops) and `update_all_stats` (4.0 %) are *warm*: outlining them replaces one
> fetch stream with two — the caller's line plus the callee's, re-entered on
> every call — and adds a call and its argument setup, which is the +0.385 %.
>
> **Function size is not the instruction-cache cost. The number of distinct
> lines the frame EXECUTES is.**

This is the code-side twin of study 66's data-side result: there, moving a
fold's weight rows 3.32× closer together converted nothing because the fold is
miss-parallel; here, moving a function's dead bytes away converts nothing
because the fetch stream follows execution, not layout. **Both close a
"rearrange it" family with a number rather than an argument.**

`icold67` stays in `Cargo.toml`, default OFF, with these numbers in its entry.
`tt_verify_child` and `check_time_slow` stay extracted unconditionally: without
the feature LLVM inlines them straight back (`.text` 777,013 either way,
identical to the pre-study base) and they read better as functions.

---

## ★★★ 5. ARM B (`icold67b`) — DEDUPLICATING THE HOT COPIES. THE MECHANISM WORKS; THE PRICE DOES NOT.

### 5.1 The opposite population

§1.5 says `do_move` exists in **3 copies (16,638 B)** and `next_move` in
**4 (10,871 B)**, and §1.4 says both are *hot* (17.3 % and 6.9 % of
`search::<NonPv>`'s ops; 11.6 % and 6.4 % of `qsearch::<NonPv>`'s). Unlike arm
A's population, these lines **are** fetched — in several places. One out-of-line
copy lets every caller fetch the same lines. `#[inline(never)]` on `do_move`,
`undo_move` and `next_move`.

| | `s67-qt` | `s67-qtd` | Δ |
|---|---:|---:|---:|
| `search::<NonPv>` | 45,565 B | **34,302 B** | **−24.7 %** |
| `qsearch::<NonPv>` | 20,123 B | **12,209 B** | **−39.3 %** |
| `.text` | 771,205 B | **741,973 B** | **−3.79 %** |

### 5.2 The panel — 14 shuffled rounds, reference `s67-qt`

| counter | **arm B** | sign | arm A | null |
|---|---:|---:|---:|---:|
| `instructions:u` | **+1.517 %** | **0/14** | +0.385 % | +0.000 % |
| **`ic_cache_fill_l2`** | **−1.462 %** | **11/14** | +3.315 % | +0.536 % |
| `branch-misses:u` | −0.609 % | 10/14 | −0.392 % | +0.170 % |
| `de_op_queue_empty` | +0.666 % | 6/14 | +0.276 % | −0.991 % |
| **`task-clock`** | **+1.644 %** | 4/14 | +0.154 % | +0.183 % |

> **Arm B is the only thing in this study that lowers `ic_cache_fill_l2` by
> construction, and it confirms the §4.3 mechanism from the other side:**
> deduplicating code the frame *executes* removes fills (−1.462 %, 11/14, well
> outside the +0.5 % null); outlining code it does not execute adds them.
>
> **And it still loses.** +1.517 % instructions at 0/14 — the calls, the
> argument setup, and the spills across them — for a fill saving worth far less.

---

## ★★★ 6. THE PRICE OF A FILL — the reusable number

Per node, from §1.3 and §5:

| | |
|---|---:|
| L1i line fills | 128.0 /node |
| cycles with a completely empty op queue | 445.7 /node |
| **⇒ upper bound on the cost of one fill** | **≤ 3.48 cycles** |
| at the engine's IPC 1.354 | **≤ 4.7 instructions** |

The bound is an *upper* one because it charges **every** cycle of frontend
starvation to instruction fetch; taken-branch bubbles and op-cache misses on
resident lines are in there too.

Arm B is the direct check. It removed **1.79 fills/node** (−1.462 % of 128.0)
and added **81.0 instructions/node** (+1.517 % of 5,341) — **45 instructions per
fill removed, ten times the ceiling** — and the panel duly reports +1.644 %
task-clock.

> ### Pre-registerable rule, in the form my own notes asked for
>
> **An instruction-cache arm in this engine must remove an L1i line fill for
> under ~5 instructions. Measure `ic_cache_fill_l2` and `instructions:u`
> together, in the same rounds, against a byte-identical-binary null — and
> price the arm against this ratio BEFORE building it.**
>
> Corollary, from §4: **do not price it in bytes.** Both arms cut `search::<NonPv>`
> by ~20 %; one moved the fill counter +3.2 %, the other −1.5 %. Bytes predicted
> the sign wrong.
>
> The one class that passes trivially is **deleting code that is executed and
> whose result is discarded** — §2 removed 65.9 instructions/node *and* 5 % of
> the fills at zero cost, because it paid nothing for either.

---

## 7. GATES

| gate | result |
|---|---|
| node total, every binary in this study | **5,001,560** — exact |
| wl50 eval hash | **`2e1ede95…79c4`** — exact |
| PGO rebuild determinism (`s67-qt` vs `s67-qt2`) | **`cmp` clean, byte-identical** |
| line-tables build is codegen-identical | same symbol size, **9,280/9,280 mnemonics equal**, 341 relocation-only byte diffs |
| `cargo fmt --check` / `cargo clippy --all-targets` | clean |
| `cargo test` (fast tier + checked perft mini-tier) | pass |
| release perft suite | pass |
| **`Bench:`** | **486897** — unchanged |
| **NNUE eval hash** (from source, non-PGO) | **`2e1ede95…79c4`** — unchanged |
| **NNUE d13 nodes** (from source, non-PGO) | **1743308** — unchanged |
| `CHECK_NNUE_GATE=1 CHECK_TEST_THREADS=2 bash tools/check.sh` | **exit 0** |

**No pin moves and none should:** the only behavioural change is the deletion of
work whose result was discarded, which cannot move a node count.

---

## 8. ASSUMPTIONS

| # | assumption | why believed | what breaks if false |
|---|---|---|---|
| A1 | IBS with `cnt_ctl=0` gives an unbiased **dispatched-MICRO-OP** histogram with no skid | IBS tags an op at issue and records that op's own rip (AMD PPR); with `cnt_ctl=0` the sampling interval is dispatched ops rather than cycles. **It is NOT a retired-instruction count and does not reproduce one** — checked: `i8_deltas_into_inner` is 15.07 % of `instructions:u` and 23.36 % of IBS ops, `search::<NonPv>` 30.87 % against 23.04 %, because Zen 4 double-pumps every 512-bit instruction into 2 µops while scalar search code is ~1. That is the correct denominator for a frontend/dispatch question and the wrong one for an instruction count, and the two are never mixed in this document | §1.4/§1.5's `%ops` columns; **not** the byte columns, which come from `objdump` + `addr2line`, and **not** any `instructions:u` figure |
| A2 | the line-tables binary's inline frames describe the plain binary's code | proved instruction-by-instruction (§1.1) | all source attribution |
| A3 | `ic_cache_fill_l2 + ic_cache_fill_sys` counts L1i line fills | AMD Zen 4 core PMC definition; the two sum to 133.1/node against 159.1 tag-lookup misses, the difference being misses to lines already in flight | §1.3's 8.2 KB/node and §6's ratio |
| A4 | `de_op_queue_empty` counts cycles with **zero** ops available to dispatch | AMD definition. Consistency check, not identity: `no_ops_from_frontend` is 25.56 G **slots**, and 6 × `de_op_queue_empty` is 14.34 G of them — the remaining 11.2 G are cycles where the frontend delivered *some but not six* ops, which is exactly the residue the two definitions predict. The counters are consistent; they are not the same quantity | §6's ceiling — which is stated as a ceiling for exactly this reason |
| A5 | the qtrace `to_string` was executed, not merely present | three independent facts: 3.6 % of `qsearch::<NonPv>`'s **sampled ops** land in it, `malloc`/`Display` symbols disappear from the profile when it is removed, and `instructions:u` falls 1.259 % at 14/14 | §2 entirely |

---

## 9. WHAT THE NEXT SESSION SHOULD DO

Ranked by the §6 rule, not by size.

1. **The §2 defect class — swept once, and it was a single site.** The class is
   "a call whose arguments do work the callee throws away". `qtrace` was the
   only `&str`-taking no-op hook in `src/search`, `src/movepick` and
   `src/board`, and `dupcensus67.py` over **fourteen** symbols (the whole hot
   list plus `iterative_deepening` and both `Pv` monomorphisations) finds no
   second `to_string`/`Display`/`drop_glue<String>` frame in hot code. What it
   does NOT cover is `init_quiets`/`init_captures`/`init_evasions`' callees and
   anything under 1 % — **re-run it after any new debug or telemetry hook
   lands**, because this class passes the §6 rule for free and is invisible to
   every other instrument in this series.
2. **`init_quiets` is 15.89 % of the engine's branch mispredicts and 9.84 % of
   its instructions**, and study 34 measured that **only 32.9 % of the quiet
   scoring it does is ever looked at**. Untouched by this study. The mispredict
   pool is ~325 cycles/node engine-wide and the search side owns 64.7 % of it.
3. **`qsearch::<NonPv>` is 3.29 % of ops for 6.92 % of L1i fills** — the worst
   ops-per-fill ratio of any symbol above 3 % of ops, 2.1× worse than the
   engine average.
   After §2 it is 20,123 B. Something in it is fetched far more than it runs;
   `coldmap67.py` on it is one command.
4. **Do NOT re-propose:** cold-path outlining in the search (§4, +3.2 % fills),
   hot-callee deduplication at this price (§5, 45 instructions/fill), sub-byte
   FT weight formats or weight-row locality (studies 65/66), the fourteen
   issue-throughput rewrites (study 47), the history prefetch (study 44 §3), or
   the TT-cutoff/eval reordering (study 64 §1.2, worth exactly zero
   evaluations).
