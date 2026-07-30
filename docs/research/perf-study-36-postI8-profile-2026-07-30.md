# ARCHON perf study #36 — the post-i8 profile, and half a percent taken out of it

**Date:** 2026-07-30 · **Base:** `main` at `74c07da` (studies 29-35 all merged) ·
**Instruments:** `tools/perf36/` plus study 34's panel kit ·
**Artifacts:** `docs/research/perf-artifacts-36/`

**Landed, three arms, all node-identical and eval-hash-identical:**

| arm | what | `instructions:u`/node | `branches:u`/node |
|---|---|---:|---:|
| 1 `evgeo` | in-check `pseudo_legal` answered from the geometry | **−0.120%** (−7.120) | **−0.258%** (−1.600) |
| 2 `flathist` | one flat `[piece × to]` index shared across six history planes | **−0.329%** (−19.497) | +0.022% |
| 3 `flat3` | the same idiom at the three other multi-plane sites | **−0.051%** (−3.013) | −0.000% |
| **total, measured END-TO-END** | 12 more paired rounds, `base` vs the landed tree | **−0.499%** (5935.139 → 5905.511) | **−0.236%** (620.785 → 619.318) |

Every one of those is measured against a **byte-identical null that moved
+0.000% with zero spread across every round**, on the exact counters. The
end-to-end row is not the sum of the three arms arithmetically assumed — it is
its own 12-round panel against pure `main`, and it lands on the sum. No cycles
claim is made anywhere in this document and §2 is the evidence for why.

**Retracted (mine to retract, and it was a "priced and dropped without a
build"):** perf study 34 §4.2 found ~11 loop-invariant stack reloads per scored
quiet in `init_quiets` and wrote *"It is **not** removable: the loop needs seven
plane pointers plus the buffer, list, board, index and bound live at once, and
x86-64 has 15 usable GPRs … each address computation still needs its own offset
in a register, which is the same number of loads."* That is wrong, and arm 2 is
the counter-example: the fix is not more registers, it is **one fewer variable
term in the addressing mode**, and it is worth 5.0 instructions per scored quiet
— 2.7× arm 1 and 78% of perf study 35's landed win.

---

## 0. HEADLINE — the target list, replaced

The standing list (`i16_deltas_into 13.14%, finny_refresh_into 9.15%,
forward_into 8.77%, init_quiets 6.35%, current_full 3.28%`, "65.75% of cache
misses in the three fold kernels") describes an engine that no longer exists.
Here is the one that does.

Frame, identical to studies 29/31/33/34/35 so the numbers compose:

```
bench 16 1 100000 tools/nnue/calib/wl50.fens nodes engines/canonical/net.cenn
```

50 FENs × 100 k nodes = **5,001,624 nodes**, one thread, 16 MB TT, D48X,
`--features sota,arch-d48x`, `RUSTFLAGS=-C target-cpu=native`, the **shipped**
release profile (fat LTO, CGU = 1), pinned to CPU 2, `nice 19`.

### 0.1 The four profiles, side by side

Within-run symbol shares. These are attributions *inside* one recording, so
they are immune to the between-run drift that killed the cycles instrument
(§2); the per-node totals beside them are not, and are only comparable within
this document.

| symbol | %cycles | %instr | %L1d loads | %mispred |
|---|---:|---:|---:|---:|
| `avx512::i8_deltas_into` — HalfKA, shape-specialised | **13.77** | 7.77 | 10.37 | 3.44 |
| `avx512::i8_deltas_into_generic` — **the whole threat channel** | **12.38** | 7.12 | 7.93 | 3.24 |
| `search::<NonPv>` | 9.80 | **12.82** | 10.78 | **16.57** |
| `HalfkaContext::finny_refresh_into` | 8.75 | 3.24 | 3.44 | 4.99 |
| `MovePicker::init_quiets` | 6.65 | 8.58 | 6.56 | **12.09** |
| `SotaHead::forward_into` | 6.15 | 9.74 | **16.09** | 3.36 |
| `ThreatContext::at_full` | 5.50 | 6.24 | 6.16 | 6.03 |
| `Worker::do_move` | 3.12 | 1.95 | 2.73 | 1.70 |
| `Board::make_move_with` | 3.04 | 5.00 | 4.45 | 4.12 |
| `HalfkaContext::current_full` | 2.95 | 1.73 | 1.79 | 1.08 |
| `sota_threats::record_square_touch` | 1.73 | 1.71 | 1.66 | 2.11 |
| `sota_loader::quantise_halfka` — **STARTUP** | 1.37 | 2.71 | 0.98 | 9.68 |
| `Histories::correction_value` | 1.34 | 1.59 | 1.48 | 1.60 |
| `sota_loader::load_v2_bytes` — **STARTUP** | 1.26 | 1.54 | 0.91 | — |
| `sota_accumulator::psqt_fold` | 1.17 | <0.73 | 1.22 | — |
| `Board::attackers_to` | 1.07 | 1.73 | 1.38 | 1.64 |
| `ThreatContext::apply_at` | 1.03 | 2.54 | 1.62 | 1.35 |
| `Board::refresh_check_info` | 1.01 | 1.10 | 1.17 | — |
| `Board::see_ge` | 0.97 | 1.31 | 1.12 | 1.84 |
| `Board::ensure_threats` | 0.92 | 1.30 | 0.93 | 1.29 |
| `Histories::update_continuation_histories` | 0.90 | 1.77 | 0.90 | 1.36 |
| `Board::unmake_move` | 0.80 | 1.31 | 0.84 | 0.94 |
| `MovePicker::init_captures` | 0.75 | 1.06 | 0.75 | 1.91 |
| `TranspositionTable::probe` | 0.74 | 1.00 | 0.81 | 1.11 |
| `movegen::generate::<EVASIONS>` | 0.24 | — | — | — |

Per-node totals in this frame: 8262.9 cycles, 5935.2 instructions, 2977.0
`L1-dcache-loads` events, 27.3 branch mispredicts.

### 0.2 The five things that changed

1. **`i16_deltas_into` is gone and was replaced by TWO symbols, not one.**
   Study 32's i8 transformer unified the fold family, and the shape dispatch
   study 25 built for HalfKA now splits the family by *caller*: HalfKA takes the
   four shape-specialised instantiations (`i8_deltas_into`), and **the entire
   threat channel falls through to `i8_deltas_into_generic`**, which is marked
   `#[cold] #[inline(never)]` and is **12.38% of engine cycles**. It is the
   second-largest symbol in the engine and it is labelled cold.
2. **The top three cycle symbols — 34.9% of engine cycles — are all weight-row
   streaming**, and their cycles/instructions ratios say so: 1.77, 1.74 and
   2.70 against an engine-wide 1.0. That is the fresh profile's real headline
   and §3 prices it.
3. **`search::<NonPv>` is no longer the top cycles symbol** — it is third at
   9.80%, though it remains #1 on instructions (12.82%) and #1 on mispredicts
   (16.57%), reproducing study 33 §6 on a different tree.
4. **~5.2% of this frame's instructions are one-time STARTUP**, and study 32
   added the largest piece of it. `quantise_halfka` alone is 2.71% of
   instructions and **9.68% of all branch mispredicts** in the frame. §4.
5. **`instructions:u` per-INSTRUCTION attribution is not an instruction census
   on this box.** §2.3 — it cost me one nearly-published claim.

---

## 1. METHOD, AND THE BOX

### 1.1 The box was NOT quiet, and it was worse than the commission's snapshot

Stated up front because it decided which instrument this study is allowed to
quote. Throughout:

* the **D49 trainer** at ~143% CPU and 4.4 GB RSS, which must not be disturbed;
* a **1466-game fastchess screen** at `-concurrency 10`, i.e. ten
  `chessengine` processes at ~48% CPU each, launched 04:43 and running through
  most of the session. Load average at commissioning: 20.5.

Available memory ran 2.3–5.9 GB. Every build went through study 34's
`build34.sh` (`systemd-run --user --scope -p MemoryMax=6G -p MemorySwapMax=0`,
`oomlast.sh`, `CARGO_BUILD_JOBS=2`, `nice 10`), which itself refuses to start
below 3000 MB available. Nothing was OOM-killed and the trainer was not touched.

### 1.2 Instruments

Study 33 left the DWARF/inline kit, studies 29/31/32/34/35 left theirs, and the
commission said to look before writing new scripts. I reused:

| tool | used for |
|---|---|
| `tools/perf34/build34.sh` | every labelled binary, under the memory discipline above |
| `tools/perf34/ident34.sh` | the identity gate (node count + wl50 eval hash) |
| `tools/perf34/panel34.sh` + `med34.py` | the **paired within-round** panel and its reduction |
| `tools/perf34/regions34.py` | splitting `init_quiets` by innermost back-edge |
| `--features foldcount` (study 29) | the exact integer fold census |

New, and only where nothing existed:

| tool | what it does |
|---|---|
| `tools/perf36/prof36.sh` | records the four-event symbol census on the study frame |
| `tools/perf36/census36.py` | turns a `foldcount` dump into §3's tables **and re-checks study 29's three published controls** |
| `tools/perf36/waitpanel36.sh` | blocks on a panel reaching its row count (background wait, never a foreground sleep) |

### 1.3 Null first

Every arm ran against a **byte-identical copy of its own reference binary**,
recorded as its own label in the same panel, in the same rotation. Across all
three panels and 40 paired rounds the null moved:

| counter | null median | null spread |
|---|---:|---|
| `instructions:u`/node | **+0.000%** | **exactly zero, every round** |
| `branches:u`/node | **+0.000%** | **exactly zero, every round** |
| `L1-dcache-loads:u`/node | −0.017% / −0.024% / +0.081% | ±0.4 points |
| `ex_ret_brn_misp:u`/node | +0.263% / −0.284% / +0.091% | ~10 points |
| `cycles:u`/node | +0.216% / +1.196% / +1.269% | **11–14 points** |

So this study quotes instructions and branches, treats loads as a ±0.4%
instrument (which none of my arms cleared), and quotes cycles nowhere.

---

## 2. THE INSTRUMENT, AND ONE THING NOBODY HAD CHECKED

### 2.1 Cycles: a third and fourth independent confirmation

Studies 33, 34 and 35 each reported that `cycles:u` cannot resolve a sub-1%
change on this box. Two more independent panels, each 14 paired within-round
rounds with the order rotating:

```
panel 1, reference `base`      null cyc/n  +0.216% [-7.18, +6.98]   (14.2 points)
panel 2, reference `evgeo`     null cyc/n  +1.196% [-5.73, +5.28]   (11.0 points)
panel 3, reference `flathist`  null cyc/n  +1.269% [-3.62, +6.16]   ( 9.8 points)
```

Every one of those is a binary computing an identical tree — same nodes, same
eval hash — against a copy of itself. The arms in this study are worth 0.05% to
0.33%. **The instrument is between 30× and 250× too coarse**, and the honest
move is the one study 34 made: carry the arms on exact counters and a mechanism,
and say plainly which claims that permits.

### 2.2 Mispredicts, likewise

`ex_ret_brn_misp:u` nulls of +0.263%, −0.284% and +0.091% over ranges of ~10
points. The mispredict *profile* (a within-run attribution) is fine and is in
§0.1; the mispredict *counter* cannot adjudicate an arm.

### 2.3 NEW: `instructions:u` per-instruction attribution is not a census

This is the one I nearly published. Building a mnemonic histogram of
`SotaHead::forward_into` from a `perf record -e instructions:u` annotation
returned **`movzwl` at 25.46% of the symbol** — a scalar 16-bit load taking a
quarter of an AVX-512 kernel, which reads like a finding.

It is an artifact. The disassembly shows `l1_dot`'s sparse loop is an 8×-unrolled
body of exactly four instructions per non-zero group:

```asm
movzwl      (%r15),%r12d              ; g = nnz[i]
vpbroadcastd (%r11,%r12,4),%zmm4      ; the group's four ft bytes
shl         $0x6,%r12d                ; g * 64
vpdpbusd    (%rax,%r12,1),%zmm4,%zmmN ; the weight block
```

All four retire the same number of times. The annotation gives them 25.46%,
1.24%, 2.25% and (below the 0.3% cut) — a **20× skew** among instructions with
identical execution counts. `instructions:u` is not a precise event on Zen 4:
the sampled IP is wherever retirement happens to be when the counter overflows,
and it piles up at long-latency boundaries exactly like a cycles event does.

**Per-SYMBOL instruction shares survive** (the skid is a few instructions and
almost never leaves the symbol), which is what §0.1 uses. Per-instruction and
per-mnemonic ones do not. `perf stat -e instructions:u` remains exact — it is a
count, not a sample.

---

## 3. THE FOLD CENSUS, EXACT — and study 29's controls reproduce

`--features foldcount`, same frame, node- and hash-identical to the plain build.
`python3 tools/perf36/census36.py docs/research/perf-artifacts-36/census36-foldcount.txt`:

| site | calls/node | rows/node | rows/call |
|---|---:|---:|---:|
| `hka_into` — HalfKA parent→child | 0.9782 | 2.2276 | 2.277 |
| `hka_dual` — Finny **warm** refresh | 0.1602 | 0.9100 | 5.682 |
| `hka_inplace` — Finny **cold** refresh | 0.0000 | 0.0001 | 5.141 |
| `thr_into` — FullThreats parent→child | 0.8572 | 3.9930 | 4.658 |
| `thr_copy` — the threat fold's empty-delta 2560 B copy | 0.3300 | — | — |
| `activate` = `eval` | 0.4272 | — | — |

**The commission's verifiable control, checked before anything was built on the
census:**

| study 29 control | published | mine | dev |
|---|---:|---:|---:|
| HalfKA refresh rate/node | 0.16102 | 0.16018 | **−0.52%** |
| HalfKA refresh rows/node | 0.9132 | 0.91004 | **−0.35%** |
| rows/fold | 2.275 | 2.27725 | **+0.10%** |

All three inside study 32's 0.9% reproduction band. The instrument is the same
instrument.

### 3.1 Where the memory traffic is

**7.131 weight rows/node × 1280 B = 8.91 KiB/node**, drawn from a 28.8 MB
HalfKA table (i8 since study 32) and a **77.7 MB** FullThreats table.
Accumulator traffic on top: 2.325 fold calls/node × 5120 B (2560 read + 2560
written) = **11.6 KiB/node**.

Per row, the two channels cost very differently. Taking §0.1's cycle shares
against the row rates above:

| kernel | cycles/node (this frame) | rows/node | **cycles/row** |
|---|---:|---:|---:|
| `i8_deltas_into` (HalfKA) | 1138 | 2.228 | **511** |
| `i8_deltas_into_generic` (threats) | 1023 | 3.993 | **256** |

A HalfKA row costs 2.0× a threat row, from a table 2.7× *smaller*. The reason is
in the census, not in the kernels: the HalfKA into-fold pays 2560 B of source
and 2560 B of destination for only **2.277** rows, where the threat fold
amortises the same 5120 B over **4.658**. Per weight row that is 2249 B of
accumulator traffic for HalfKA against 1099 B for threats.

### 3.2 The shape histograms, and one dead instantiation

HalfKA into-folds, by `(adds, subs)`:

| shape | share |
|---|---:|
| (1, 1) | 72.28% |
| (1, 2) | 26.73% |
| (2, 1) | 0.99% |
| **(2, 2)** | **0.00% — n=0 in 5,001,624 nodes** |

`i8_deltas_into_inner` carries a `(2, 2) => i8_into_shaped::<2,2>` arm that is
never taken. It is correct and harmless (castling and the other two-add shapes
route to a Finny refresh, not to an into-fold), and I did **not** remove it:
deleting it would only move a never-executed case to the generic path, and the
icache it occupies is unmeasurable with the instruments this box allows.

Threat folds, by total rows: **18.66% fold exactly one row, 59.10% fold four or
fewer, 85.08% eight or fewer**, mean 4.658, max 27.

That distribution is the case *for* shape-specialising the threat channel too —
and §6 is why I did not, with the number that decided it.

---

## 4. STARTUP IS 5.2% OF THIS FRAME, AND STUDY 32 ADDED MOST OF IT

Three of the symbols in §0.1 run **once, at engine start**, not per node:

| symbol | %instructions | i/node in this frame |
|---|---:|---:|
| `sota_loader::quantise_halfka` (added by study 32) | 2.71% | 161 |
| `sota_loader::load_v2_bytes` | 1.54% | 91 |
| `attacks::init_magics` | 0.95% | 56 |
| **total** | **~5.2%** | **~308** |

`quantise_halfka` walks 22,528 × 1280 = 28.8 M i16 weights to build the i8
table, and it is **9.68% of every branch mispredict the frame records**.

**This costs strength nothing** — a UCI engine loads its net once per process,
and fastchess reuses the process across a whole match — and it is not a target.
It matters for measurement hygiene, in one direction:

> Every *delta* in this study, and in studies 34 and 35, is unaffected: startup
> is bit-identical between arms and cancels in the pairing. But the
> **denominator** those deltas are divided by is ~5.2% too large, so every
> percentage quoted on this frame is ~5% **conservative**. Arm 2's −0.329% is
> −0.347% of the engine's actual per-node work.

Nothing needs retracting; the sign of the bias is worth knowing.

---

## 5. THE ARMS

### 5.1 Arm 1 — in-check `pseudo_legal`, from the geometry

**Landed, commit `85d8cac`.**

`Board::pseudo_legal`'s check 6 asked "in check, would the evasion generator
emit this Normal move?" by running a full `generate::<EVASIONS>` into a stack
`MoveList` and scanning it with `contains` — to validate **one** move. Study 34
§9 measured the bill: **157,016 of the engine's 349,337 evasion generations —
45% — are that line.**

Once checks 3–5 have done the piece geometry, `generate_all::<EVASIONS>` is
three facts, and this is the whole of the replacement:

* a **King** move is unrestricted (`king_target = !occupancy(us)`), and check 4
  cleared `to` while check 5b ran `king_attacks(from)`; check 3 puts our piece on
  `from`, so a King there **is** `king_square(us)`. Nothing is left to test.
* under **double check** every non-king generator is skipped outright.
* under **single check** every remaining generator is masked by
  `target = between(ksq, checker)` — pushes and piece moves directly, and pawn
  captures by `enemies = checkers()`, which is the same set here because the
  squares strictly between a king and its checker are empty by construction, so
  `target & occupancy(them) == checkers()`.

**Measured**, 14 paired within-round rounds, reference `base`:

| counter | arm | null | spread |
|---|---:|---:|---|
| `instructions:u`/node | **−0.120%** (5935.139 → 5928.019) | +0.000% | **zero, both** |
| `branches:u`/node | **−0.258%** (620.784 → 619.185) | +0.000% | **zero, both** |
| `L1-dcache-loads:u`/node | −0.039% | −0.024% | no signal |
| `cycles:u`/node | +0.564% | +0.216% over 14.2 points | dead |

**Mechanism, divided out:** 7.120 i/n ÷ 0.031393 removed generations/node =
**227 instructions and 50.9 branches per generation-and-scan**. That is the size
of a small evasion generation plus a linear `contains`.

**Tests.** `pseudo_legal` already had an exhaustive differential oracle — all
65,536 encodings against a fresh generation, over a fuzz corpus, plus
`perft_via_predicate` which recurses using the predicate at every node. I added
**five in-check fixtures, one per shape of the single-check mask** (knight
checker; distant rook with one piece having both blocking and non-blocking
destinations; contact queen; pawn checker; and a bishop check blockable *only*
by a double push, which is the one geometry where check 5a's relay-square test
and the ray mask must both fire), and put **perft position 4 — whose root is in
check** — into `perft_predicate_oracle_deep`, so the oracle now recurses through
the rewritten branch from the root as well as at interior nodes. The D24 ritual
(`cargo test` + `--ignored perft_full_suite`) passes.

### 5.2 Arm 2 — six planes, one cell: the flat `[piece × to]` index

**This is the study's main result**, and it is a retraction of an item a
predecessor priced and dropped without building.

`init_quiets`' per-quiet scoring loop reads **seven** history planes at one
`(pci, to)` pair:

```rust
let mut value = MAIN * main_plane[raw].get()
    + PAWN * pawn_plane[pci][to].get()
    + c1[pci][to].get() + c2[pci][to].get() + c3[pci][to].get()
    + c4[pci][to].get() + c6[pci][to].get();
```

Study 34 §4.2 found "~11 stack reloads of loop-invariant plane base pointers per
scored quiet" and concluded they were structural. Study 34's own region split
(re-run here on today's build) puts this loop at **52.55% of `init_quiets`'
cycles, 75 instructions per quiet**. The shipped disassembly:

```asm
shl  $0x7,%ecx                    ; pci * 128, shared
lea  (%rbx,%rbx,1),%r10d          ; to * 2,    shared
mov  0x70(%rsp),%r8               ; c1 base — reloaded
lea  (%r8,%rcx,1),%rdx            ;   + pci*128            <- a lea per plane
movswl (%r10,%rdx,1),%edx
mov  0x68(%rsp),%r8               ; c2 base — reloaded
lea  (%r8,%rcx,1),%rsi            ;   + pci*128            <- and again
movswl (%r10,%rsi,1),%esi
...                                ; five times
```

**The register-pressure diagnosis was the wrong diagnosis.** The cost is not
that there are too few registers — it is that `base + pci*128 + to*2` is **three
variable terms and an x86 addressing mode holds two**, so each plane must
materialise its own sum before it can load. Make the shared part the *whole*
offset and each plane collapses to one instruction:

```rust
let q = history::piece_to(pci, to);           // pci * 64 + to, computed once
… + history::at_piece_to(c1, q) + history::at_piece_to(c2, q) + …
```

`PieceToHist` is `[[StatEntry; 64]; 12]`, which Rust lays out row-major and
contiguously, so `p[pci][to]` and `p.as_flattened()[pci * 64 + to]` are the same
cell — same values, same order, bit-identical sum. **And no bounds check comes
back in exchange, with no mask and no `unsafe`:** the loop has already indexed a
6-element array by the piece type, so LLVM carries `pci < 12`; `Move::to` is six
bits, so `to < 64`; range analysis then proves `q < 768` and drops the check on
the 768-element flattened view. (`as_flattened` is the idiomatic form —
`[[T; N]]::as_flattened`, stable since 1.80 — and it is why this needed no
`get_unchecked`.)

The resulting loop:

```asm
shl  $0x7,%ecx
lea  (%rcx,%rbx,2),%ecx           ; q*2, ONE shared lea for all six planes
movswl (%rsi,%rcx,1),%esi         ; plane, no lea
movswl (%r9,%rcx,1),%r9d          ; plane, no lea
movswl (%r10,%rcx,1),%r13d        ; plane, no lea
mov  0x60(%rsp),%rdx ; movswl (%rdx,%rcx,1),%edx
mov  0x58(%rsp),%rdx ; movswl (%rdx,%rcx,1),%edx
movswl (%r11,%rcx,1),%r9d         ; plane, no lea
```

Four of the six bases now stay in registers, because the pressure that spilled
them was the per-plane sums, not the bases.

**Measured**, 14 paired within-round rounds, reference `evgeo` (arm 1's tree):

| counter | arm | null | spread |
|---|---:|---:|---|
| `instructions:u`/node | **−0.329%** (5928.019 → 5908.522) | +0.000% | **zero, both** |
| `branches:u`/node | +0.022% | +0.000% | zero, both |
| `L1-dcache-loads:u`/node | −0.020% | −0.017% | no signal |
| `cycles:u`/node | −0.233% | +1.196% over 11.0 points | dead |

**Mechanism, divided out:** 19.497 i/n ÷ 3.927 scored quiets/node = **4.97
instructions per scored quiet**, against **five** continuation-history planes
each losing exactly one `lea`. The pawn plane's `add` is paid for by the one new
shared `lea`. That is as clean a mechanism check as this study got.

**Honest sub-result:** I predicted the three vanished stack reloads would show
as a load reduction of ~0.4%, and **they did not** — loads moved −0.020%
against a −0.017% null. Either LLVM reloads those bases elsewhere in the
function, or the reduction is below this box's ±0.4% load floor. The win is
address-generation instructions, not loads, and I am recording the failed
prediction rather than the tidy story.

### 5.3 Arm 3 — the same idiom, everywhere else it applies

Three more sites read several `[piece × to]` planes at one shared cell:

* `Histories::update_continuation_histories` — six loop iterations, one plane
  each, `pci`/`toi` invariant across all of them (write side, `at_piece_to_mut`);
* `Histories::correction_value` — the two `cont_corr` planes at offsets 2 and 4;
* `Worker::stat_score` — the two continuation planes at offsets 1 and 2.

**Measured**, 12 paired rounds, reference `flathist`: **`instructions:u`/node
−0.051%** (5908.524 → 5905.511), zero spread, null +0.000%; branches −0.000%.

Small, and stated at its size. It is here because it is the same three-line
idiom, it is free once the helper exists, and leaving three sites on the old
spelling would make the codebase teach the wrong thing.

### 5.4 End to end

A fourth panel, 12 paired rounds, pure `main` at `74c07da` against the landed
tree, to check that three arms measured against three different references add
up when measured in one go:

```
label      n            i/n              cyc/n             br/n          loads/n
base      12  +0.000%[+0.00,+0.00] +0.000%             +0.000%         +0.000%
null      12  +0.000%[-0.00,+0.00] +1.394%[-4.98,+8.90] +0.000%        +0.065%
flat3     12  -0.499%[-0.50,-0.50] +2.807%[-3.10,+7.40] -0.236%[-0.24,-0.24] -0.011%
```

**−0.499% instructions and −0.236% branches, zero spread on both, against a
+0.000% null.** 5935.139 → 5905.511 i/n; 620.785 → 619.318 br/n. The sum of the
three arms is −0.500% / −0.236%; the direct measurement is −0.499% / −0.236%.

The cycles column on that same panel is the study's closing argument about the
instrument: a byte-identical null read **+1.394% over a 13.9-point range**,
while the arm — which is unambiguously 0.5% less work — read **+2.807%**.

---

## 6. PRICED, AND NOT TAKEN — each with the number that decided it

| candidate | why not |
|---|---|
| **Shape-specialise the threat fold** (§3.2 says 59% of calls fold ≤4 rows) | The generic kernel's loop control is **4 instructions per row per tile** (`mov`, `add`, `lea`, `shl`) at TILES = 2, i.e. 8 × 3.993 = **32 i/n ≈ 0.54%** total — and shaping recovers only part of that, against ~1.8 KB of new icache per instantiation and a combinatorial `(adds, subs)` matrix up to 27 rows. The annotated kernel is already fully unrolled 20 zmm wide; its cycles sit on `vpmovsxbw` from the 77.7 MB weight table, not on the loop. |
| **Un-`#[cold]` `i8_deltas_into_generic`** — 12.38% of cycles is not cold | Real, and probably right. But `#[inline(never)]` keeps it a separate symbol either way, so the change is **pure code placement**, which moves cycles and nothing else — and §2.1 says cycles cannot be measured here to 30× the effect size. Left as a one-word arm for a quiet box. |
| **The accumulator/threat channel fusion** (11.6 KiB/node of accumulator traffic, §3.1) | Study 29 priced it at ~65 cycles/node ≈ 1.4% and said "do not build it"; the refresh-trigger union and the Finny-key problem are the blockers, not the kernel. Nothing in my census changes that. |
| **The `(2, 2)` shaped instantiation, never executed** (§3.2) | Removing it is unmeasurable with the instruments this box allows, and it would move a never-taken case to a different kernel for no counted gain. |
| **`quantise_halfka`, 2.71% of frame instructions** (§4) | Startup. A UCI process loads its net once. Not a strength lever, and optimising it would only shrink a denominator. |
| **`init_quiets`' scoring loop, beyond arm 2** | The remaining ~70 instructions per quiet are 9 semantically necessary loads plus the threat/offense/check terms, all of which study 12/13/23/27 already reduced to single bit tests. |
| **`insertwin34`** — study 34's branchless insertion window | §7. |

### 6.1 `insertwin34` stays OFF, and this is the number

The commission named adjudicating study 34's `--features insertwin34` as a good
landed deliverable. **I did not do it, and I believe attempting it tonight would
have produced a false verdict.**

The trade is **+0.702% instructions for −0.530% branches and −4% of all engine
branch mispredicts**; study 34's own op-count model puts it at **≈ −0.17%/node**
in cycles. Cycles is the only arbiter for an instructions-for-mispredicts trade,
and §2.1's three nulls put this box's cycles floor at **9.8 to 14.2 points of
spread around a median that itself moved +0.216% to +1.269%**. That is 30–250×
the effect. Games cannot arbitrate either: the arm is node-identical, so a
fixed-node match is a dead heat by construction, and 0.17% of speed is ~0.3 Elo
at ARCHON's own 122.9-Elo-per-doubling constant — six figures of games.

The instrument does not exist tonight. Reporting "we flipped it on and it looked
fine" would be the exact error studies 22, 29 and 34 each caught. It needs a
genuinely idle box and ~40 paired rounds, and it is still one panel away, not
one reimplementation away.

---

## 7. GATES

`CHECK_NNUE_GATE=1 bash tools/check.sh` — **exit 0, "All checks passed"** — on
the tree with all three arms:

| gate | result |
|---|---|
| `cargo fmt --check` | clean |
| `cargo clippy --all-targets -- -D warnings` | clean |
| feature-matrix `cargo check` | clean |
| `cargo test` (debug: unit + fast perft tier) | pass |
| release perft suite (`--ignored perft_full_suite`) | pass |
| `--ignored perft_predicate_oracle_deep` (64k predicate scan per node, now with an in-check root) | pass |
| `bench 16 1 6 default depth` | **587683** — the pin, bit-for-bit |
| wl50 eval hash | `eb981b11…c84f4f37` — the pin |
| d13 node total | **1985398** — the pin |
| node identity, study frame | **5,001,624** on `base`, `evgeo`, `flathist`, `flat3` and every null |
| wl50 eval hash, study frame | `eb981b119bd66d6e` on all of the above |

All three arms are pure refactors of how the same values are computed and read,
so node identity is the primary gate and it holds on every binary.

---

## 8. REPRODUCING

```bash
bash tools/perf34/build34.sh base                                   # shipped
bash tools/perf34/build34.sh fold "foldcount,sota,arch-d48x"        # the census
bash tools/perf34/ident34.sh base fold

bash tools/perf36/prof36.sh base                                    # §0.1
ARCHON_FOLDCOUNT=docs/research/perf-artifacts-36/foldcount.txt \
  ./bin34/fold bench 16 1 100000 tools/nnue/calib/wl50.fens nodes \
  engines/canonical/net.cenn
python3 tools/perf36/census36.py docs/research/perf-artifacts-36/census36-foldcount.txt

cp bin34/base bin34/null                                            # the null
P34_PANEL=docs/research/perf-artifacts-36/panel-evgeo.txt \
  bash tools/perf34/panel34.sh 14 base null evgeo
python3 tools/perf34/med34.py docs/research/perf-artifacts-36/panel-evgeo.txt base
```

Raw panels: `panel-evgeo.txt`, `panel-flathist.txt`, `panel-flat3.txt` — every
row, in order. `census36-foldcount.txt` is the census dump. The `perf.data`
files are 36 MB and gitignored; regenerate with `prof36.sh`.

---

## 9. WHAT I AM LEAST SURE ABOUT

In order.

1. **That arm 2's mechanism is only the `lea`s.** 4.97 instructions per quiet
   against five planes is a very tidy fit, and §0's rule 6 says to attack a tidy
   fit rather than enjoy it. The competing story is that removing the per-plane
   sums freed registers and LLVM then reorganised more than I looked at — the
   static count for the whole of `init_quiets` moved only 1142 → 1137
   instructions and 181 → 177 stack references, which is *not* what a purely
   local five-instruction saving predicts, and I did not chase the difference.
   The measured delta is exact; my account of it may be incomplete.
2. **The load column, which contradicted my prediction** (§5.2). Three reloads
   per quiet visibly leave the loop body and the counter does not move. I do not
   have an explanation I would defend, and the ±0.4% load floor on this box is
   wide enough to hide the effect without explaining it.
3. **§3.1's 511 vs 256 cycles per row.** It divides a *cycles* share — the one
   currency §2 says is unreliable — by an exact row rate. Within-run shares are
   much better behaved than between-run deltas, so I believe the 2× is real, but
   I would not defend either number to three digits, and no arm was built on it.
4. **That the threat channel is genuinely at a local optimum** (§6 row 1). I
   priced the loop control from a static reading of the disassembly, not from an
   arm. The census says the lists are short and the shaped-dispatch precedent
   (study 25 arm 1, 11.16% of engine cycles) is exactly this shape, so a
   determined implementation could still beat my estimate. What I am confident
   of is the *reason* it is expensive — 5.1 KB/node streamed from a 77.7 MB
   table — and that is a table-size problem, not a kernel problem.
5. **Whether `#[cold]` on a 12.38%-of-cycles symbol is actually costing
   anything.** It is obviously mislabelled. Whether that mislabelling has a
   price is a cycles question and I could not ask it.
6. **The startup figure of 5.2%** (§4) is read off symbol shares, not off a
   two-budget difference. The three symbols are unambiguously one-time, but I
   did not run the second budget that would have made the number exact.
