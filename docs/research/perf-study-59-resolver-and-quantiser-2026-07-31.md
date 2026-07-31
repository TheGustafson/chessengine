# Perf study 59 — one fused threat table, and a quantiser LLVM can vectorise

**2026-07-31. Branch `worktree-agent-ad040678fa3fc78bf`, commit `77c3aa2e` on
top of `main` at `a8a013c4`.**
**Box** 7700X (Zen 4), AVX-512 VBMI2, cpu1 (`P59_CPU=1`), shared throughout with
the live D49 GPU trainer (93% GPU, ~145% CPU).
**Frame, quoted with every number below:**
`bench 16 1 100000 tools/nnue/calib/wl50.fens nodes engines/canonical/net.cenn`
— **5,001,562 nodes on every binary in this document**, and a 1-node variant of
the same command for the startup panel.

---

## ★ 0. THE ANSWER, FIRST

### 0.1 What landed

Two arms, both node-identical at **5,001,562** and eval-bit-exact
(`2e1ede95…79c4`), `instructions:u` on three separate compilers:

| arm | what | **shipped (native+PGO)** | gate (`x86-64-v3`) | native non-PGO |
|---|---|---:|---:|---:|
| **`lutp`** | `ThreatResolver::index_decoded` reads ONE table where it read two | **−24.29 /node** | **−15.71 /node** | **−11.91 /node** |
| **`qrow`** | `quantise_halfka` becomes a row walk LLVM auto-vectorises | −152.48 † | **−11.25 /node** | **−158.27 /node** |
| **both** | | **−176.77 /node (−3.086%)** | **−26.97 /node (−0.462%)** | **−170.18 /node (−2.924%)** |

† the only derived figure in this table: the difference of two panels measured
against the same PGO base. Every other cell is a directly measured contrast.

Branches, both arms, shipped build: **−13.83 /node (−2.503%)**. Cycles on the
shipped build, 39 paired rounds against a null that passes at 18/39:
**−0.886% (35/39, p < 0.0001)**, of which essentially all is the startup frame
(§5.3: **−147.2 M cycles per engine process**, on top of study 57 arm B's
−263.2 M).

Gate: `CHECK_NNUE_GATE=1 bash tools/check.sh` → **All checks passed.**,
`Bench: 587683`, eval hash
`2e1ede9521624dd89eab4eb1bae0a1ea568cd6cf7e85b668179a7b5e93cf79c4`,
d13 `1739588`; `cargo test --release -- --ignored perft_full_suite` → **ok**.

**`qrow` is a fixed per-process cost and this study does not pretend
otherwise.** §5.3 proves the split rather than asserting it: startup
instructions fall **47.000%** on the 1-node frame, which is **158.3 per node of
the 5,001,562-node frame** — `qrow`'s entire native figure of 158.27, to three
significant digits. So the whole of `qrow` is startup and none of it is search.
It is worth a real fraction of every fixed-node bench, `tools/screen.sh`
invocation and datagen process, and approximately nothing in a game at a real
time control. **`lutp` is the only arm here that runs inside the search**, and
on the shipped binary it is −24.29 i/node with a −0.19% net load win.

Series position, on the same gate build the brief's series total names
(`x86-64-v3`): **5834.43 → 5807.47 i/node.**

### 0.2 ★ The commission's profile is right, study 58's is right, and they are not about the same binary

The brief asked me to resolve a disagreement — study 58 read `current_full` at
2.52% of instructions and `do_move` at 8.76%; the commission read 8.88% and
`do_move` nowhere in the top ten. **Both reproduce exactly on my own
from-source builds. The variable is PGO, and the commission's table is the one
that describes the shipped binary.**

`perf record -e instructions:u -c 2000000`, same frame, same source, two builds:

| symbol | commission (shipped) | **mine, native+PGO** | **mine, native non-PGO** | study 58 (non-PGO) |
|---|---:|---:|---:|---:|
| `search::<NonPv>` | 26.95 | **26.01** | 13.50 | 13.11 |
| `SotaHead::forward_into` | 11.03 | 11.36 | 10.22 | 11.03 |
| `i8_deltas_into_generic` | 9.47 | 10.12 | 9.26 | 9.07 |
| **`HalfkaContext::current_full`** | **8.88** | **8.73** | **2.36** | **2.52** |
| `MovePicker::init_quiets` | 8.56 | 8.64 | 8.49 | — |
| `ThreatContext::at_full` | 8.03 | 8.33 | 6.34 | 6.39 |
| `finny_refresh_into` | 4.25 | 3.97 | 3.94 | 4.18 |
| `qsearch::<NonPv>` | 3.32 | 3.22 | 1.36 | — |
| `quantise_halfka` | 2.90 | 2.94 | 3.00 | — |
| `record_square_touch` | 2.65 | 2.91 | 2.36 | — |
| **`Worker::do_move`** | **(absent)** | **0.87** | **8.35** | **8.76** |

`do_move` and `current_full` are the two symbols PGO relocates: it folds
`do_move`/`make_move_with` into `search::<NonPv>` (13.50 → 26.01) and the fold
kernels into `current_full` (2.36 → 8.73). Every other row agrees within
sampling noise. **Nobody measured anything wrong; study 58 named its table
"non-PGO `instructions:u`" and it is exactly that.**

Three of study 58's own per-node figures reproduce to the decimal on
independently built binaries, which is the strongest cross-check in this
document: its landed non-PGO **5819.69** (I read 5819.68), its landed PGO
**5729.02** (I read 5728.99). And the commission's series total
**"5834.4 i/n on the gate build"** is my `x86-64-v3` baseline, **5834.43** —
so the commission's "gate build" is v3 while `build58.sh`/`build59.sh` default
to `native`, and study 58's non-PGO column is a *native* column. That is the
same two-compilers confusion the brief opens with, one level up.

### 0.3 ★ An instrument the brief did not warn about, and it is the one every study here has been reading

**`perf record -e instructions:u` is usable per SYMBOL and is not usable per
instruction or per source line on this box.** Two receipts, both from the
shipped binary:

* one stack load, `mov 0x2e0(%rsp),%rax` at `+0x47ba3`, collects **13.32% of
  `search::<NonPv>`'s `instructions:u` samples**;
* one source line — `mod.rs:551`, which is `self.tt.prefetch(key)`, a single
  `prefetcht0` — collects **13.15% of the ENGINE's**.

Neither can be a retired-instruction count. Study 36 §2.3 killed per-*mnemonic*
attribution of this counter with a 20× skew among four instructions that retire
identically; this is the same defect at per-instruction and per-line
granularity, and a study that reads a line census off `instructions:u` will
aim at whatever the skid picked.

The replacement is already in the repo and it is the right one: **AMD IBS op
sampling** (`ibs_op/cnt_ctl=1,swfilt=1/`, study 52's `tools/perf52/ibsline52.py`),
which tags a micro-op at ISSUE and records that op's own rip. Its per-symbol
shares agree with `instructions:u`'s to a point (28.78 vs 26.01 for
`search::<NonPv>`, 9.42 vs 8.33 for `at_full`, 8.54 vs 8.64 for `init_quiets`),
and unlike it, its per-line shares survive inspection: `lutp` exists because IBS
put ONE source line at 0.90% of the whole engine, and rebuilding that line
moved the counter in the predicted direction.

`tools/perf59/ibs59.sh` records it and `tools/perf59/stack59.py` prints the
full inline stack per hit offset — the frame *between* `ibsline52.py`'s
innermost and outermost, which is what says *who asked for* the
`copy_nonoverlapping` you are looking at (§1.2 answers exactly that question,
and the answer is "not a memcpy").

### 0.4 ★ An arm's value is a property of the target-cpu, and this study has the mechanism in the disassembly

The brief's headline rule is that study 57's arms are worth −30.8 i/n on
`native` and −187.6 on `x86-64-v3`, a 6× spread. **This study reproduces the
phenomenon at the same magnitude and in the OPPOSITE direction, with the cause
read off `objdump`:**

| | native | `x86-64-v3` | ratio |
|---|---:|---:|---:|
| `lutp` | −11.91 | **−15.71** | 0.76× |
| `qrow` | **−158.27** | −11.25 | **14.1×** |
| both | −170.18 | −26.97 | 6.31× |
| both, `branches:u` | −2.351% | **+0.003%** | sign flip |

`qrow`'s inner loop needs a 64×64→64 packed multiply. `native` has AVX-512DQ and
LLVM emits **`vpmullq`: 21 instructions for 8 weights**. `x86-64-v3` caps at
AVX2, which has no such instruction, so LLVM decomposes it into two `vpmuludq`
+ `vpsllq` + `vpaddq` per 4 lanes and emits **~28 instructions for 4 weights**.
2.6 instructions per weight against ~7 — and the scalar baseline is 24.

**So "6× between compilers" is not a curiosity of study 57's arms. It is what
happens whenever an arm's benefit depends on an instruction one target has and
the other does not, and the only defence is to build both.** Note that `lutp`
goes the *other* way: it is 32% larger on v3, because the v3 build has more
pressure on the loads it deletes.

### 0.4b One claim in the brief that this study could not confirm

The brief says study 57's arms are **−30.8 i/n on `native`** and **−187.6 on
`x86-64-v3`**. I did not re-derive those two numbers — study 57 is already
merged into my base, so re-pricing it would need two more builds off an older
commit and it was not on the path to landing anything. **So the 6× receipt in
this document is my own, not a confirmation of the brief's.** They agree on the
shape and on the magnitude; they are independent measurements of different
arms, and one of them (mine) has the mechanism in the disassembly while the
other has "`native` has AVX-512 VBMI2 and already performs by hand what some
arms do explicitly", which is a hypothesis. If the brief's figure ever becomes
load-bearing, it needs its own `objdump`.

### 0.5 Priced and NOT built, each with its number

Reported prominently because a closed lever is a result. All from the IBS
census in §1, all against the shipped binary:

1. **`search::<NonPv>`, the commission's largest target (28.78% of engine ops):
   I did not land in it, and here is the map that says where its cost is.** Its
   own body is 34.33% of the symbol; the rest is inlined callees, and every one
   of the big ones already has a prior study's number against it:
   `do_move` 21.25% (study 55: `#[inline(always)]` costs +25.7 i/n; study 15:
   deriving the threat snapshot from the parent costs +106), `next_move` 5.88%
   (study 27 arms 3+5 landed the literal-stage dispatch and the force-inline),
   `legal` 5.05%, `update_all_stats` 3.88%, `undo_move` 3.03%, `see_ge` 2.86%,
   `correction_value` 2.51%, `reduction` 2.30%, `stat_score` 2.15%, `probe`
   2.11% (study 52: at its 32-byte one-line floor), `gives_check_by` 1.80%
   (studies 53/54). Study 52's class census reproduces: I read spill traffic at
   **19.0% of the symbol's ops** where it read 19.3%.
2. **`correction_value` made lazy — priced at ≈4 i/n and declined.** It is
   computed unconditionally at node entry (2.51% of the symbol = 0.72% of the
   engine ≈ 41 i/n) and the `rfp52` move is to sink it below the early returns.
   But it is read at `mod.rs:3588` (singular) and `mod.rs:3719` (LMR), so
   **every node that reaches the move loop needs it**; the savable set is
   Step 3's draw/abort return, Step 4's mate-distance return, and in-check nodes
   that TT-cut off. ~10% of a 41 i/n site. Recorded so the next study does not
   re-derive it.
3. **`reduction`'s `idiv` — the divisor is per-ITERATION, and it is still not
   worth it.** `scale - delta * RED_DELTA_MULT() / self.root_delta`
   (`mod.rs:4441`) is 1.02% of the symbol on one line, and `root_delta` changes
   only when the aspiration window does — so study 57 arm B's reciprocal trick
   applies directly. It is declined for the reason study 52 §0.5 gave and this
   study can only restate: the whole divide class in this symbol is ~0.40% of
   frame by `ex_div_busy`, a reciprocal costs MORE instructions than an `idiv`
   (so `instructions:u`, the counter with the 0.000 null, would read it as a
   loss), and 0.4% of cycles is under what 40 paired rounds resolve.
4. **`init_quiets`' sort and scoring loop are defended.** `partial_insertion_sort`
   is 16.75% of the symbol (1.43% of engine ops) and carries FOUR prior numbers
   against it — study 23's `copy_within` +91 i/n and `partition_point` +77.8,
   study 34's branchless window (parked), study 55's `bcsort55` **+14.7 i/n on
   the shipped tree**. `generate_quiets` is 14.63% and is movegen. The scoring
   loop's single hottest line, `movepick.rs:1259` at 10.32% of the symbol =
   **0.88% of the engine**, is the seven-plane history gather — load-bound by
   construction and the subject of study 44's prefetch arm.
5. **The threat shape dispatch stays closed** (studies 25 §3, 36 §6, 57 §6.1;
   real value ≈0.63%). Not re-opened. `pair58`/`quad58` not re-proposed.

### 0.6 The intermediate arm, reported because it is the finding

`qrow` was reached in two steps and **the first step is only 19% of the win**:

| | `instructions:u` | `branches:u` |
|---|---:|---:|
| `quant` — keep study 57's iterator, replace `i % ACC` with a cyclic counter and the `Vec` with a `[u64; ACC]` | **−29.31 /node** | **+8.53 /node** |
| `qrow` — drop the collector, walk rows into a pre-allocated buffer | **−158.27 /node** | −33.09 /node |

The extra 129 i/node is not arithmetic. It is that the second shape is a loop
LLVM can vectorise and the first is not. §4.3.

---

## 1. THE CENSUS — where the shipped binary's ops actually are

`bash tools/perf59/ibs59.sh base-pgoline 3 <out>` (3 × 5,001,562 nodes,
722,738 tagged ops), then `tools/perf52/ibs52.py` and
`tools/perf52/ibsline52.py`. `base-pgoline` is a PGO build **with DWARF line
tables** (`tools/perf59/pgoline59.sh`) — the diagnostic twin of the shipped
binary, never the one a number is quoted from.

### 1.1 `at_full` — 9.42% of engine ops, and one line is a tenth of it

| innermost frame | % of `at_full`'s ops | ⇒ % of engine |
|---|---:|---:|
| its own body (`compute`'s walk, the entry hand-out) | 28.65 | 2.70 |
| **`ThreatResolver::index_decoded`** | **21.68** | **2.04** |
| `resolve_both` | 15.53 | 1.46 |
| `copy_nonoverlapping<u8>` | 7.52 | 0.71 |
| `i8_deltas_into_inner` + `_mm512_add/sub_epi16` | 7.60 | 0.72 |
| `build_deltas` | 3.68 | 0.35 |
| `RowOffset::of` | 2.19 | 0.21 |
| `record_incoming_mutate` | 2.09 | 0.20 |
| `sf_code` | 1.55 | 0.15 |

By source line inside `index_decoded`:

```
9.56 %   sota_features.rs:549   idx = base + offsets[att][from_o] + lut2[att][from_o][to_o]
4.25 %   sota_features.rs:545   base = lut1[att][atd][dir]
2.35 %   sota_features.rs:543   dir  = (from_o < to_o)
1.86 %   sota_features.rs:540   to_o
1.21 %   sota_features.rs:539   from_o
```

**Line 549 is 0.90% of every op the engine issues** — the largest single source
line in the symbol, and the arm in §4.1.

### 1.2 The `copy_nonoverlapping` is not a memcpy, and this is why the tool exists

7.52% of `at_full` on `copy_nonoverlapping<u8>` reads like a `ThreatPosition`
snapshot copy waiting to be deleted. `stack59.py` prints the caller:

```
3.09 %  copy_nonoverlapping @mod.rs:551 <- write_unaligned @mod.rs:2019 <- _mm512_storeu_si512 @avx512f.rs:36078
2.32 %  copy_nonoverlapping @mod.rs:551 <- _mm256_loadu_si256  @avx.rs:1725  <- i8_into_shaped @sota_simd.rs:1931
2.11 %  copy_nonoverlapping @mod.rs:551 <- _mm256_loadu_si256  @avx.rs:1725  <- i8_into_shaped @sota_simd.rs:1924
```

It is `core::arch`'s own spelling of an unaligned SIMD load/store, inside the
fold kernel. **There is nothing to delete.** An arm aimed at "the snapshot
copy" would have found nothing and cost a build; the two-frames-deep view is
what stopped it.

### 1.3 `init_quiets` — 8.54% of engine ops

| phase | % of `init_quiets`'s ops | ⇒ % of engine |
|---|---:|---:|
| the scoring loop, its own body | 47.98 | 4.10 |
| `partial_insertion_sort` | 16.75 | 1.43 |
| `generate_quiets` | 14.63 | 1.25 |
| `set_slot` | 4.67 | 0.40 |
| `offense_from` (`array::from_fn`) | 4.42 | 0.38 |
| `at_piece_to<8192>` (the pawn plane) | 2.39 | 0.20 |

Hottest lines: `movepick.rs:1259` (the seven-plane history sum) 10.32%,
`history.rs:223` (`at_piece_to`) 11.82% spread over six call sites,
`movepick.rs:282/283` (the sort's shift loop) 6.41%.

### 1.4 `search::<NonPv>` — 28.78% of engine ops

Phase table in §0.5 item 1. Operand class, static instruction count vs
**dynamic sampled ops**:

```
    class          static  static%   dyn ops    dyn%
    load-rsp         1260    15.1%     23580   11.9%
    load-mem         1322    15.8%     51740   26.1%
    store-rsp         340     4.1%     14162    7.1%
    store-mem         182     2.2%      5258    2.7%
    reg-only         5249    62.8%    103461   52.2%
```

Spill traffic (`load-rsp` + `store-rsp`) is **19.0% of the symbol's dynamic
ops**, reproducing study 52's 19.3% on a different PGO build a week later.
Study 52's conclusion stands and I am not re-deriving it: this is the cost of
keeping a 3,383-instruction function's state in 16 registers, there is no
single hot slot, and the symbol's remaining wins are at call sites, not in
classes.

---

## 2. THE INSTRUMENT — and one warning the brief did not carry

§0.3 has the finding. The operational rules this study used:

* **`instructions:u` per symbol: yes.** `perf report -s symbol` shares are
  stable across builds and reproduce prior studies' tables.
* **`instructions:u` per instruction or per line: NO.** 13.32% on one stack
  load, 13.15% on one `prefetcht0`. Use IBS.
* **`instructions:u` as a paired A/B total: exact**, as studies 48/57/58 found.
  `tools/perf59/pair59.sh` reads run-to-run agreement of ~5×10⁻⁸ on this box —
  every arm figure in §0.1 is a difference of two numbers that repeat to nine
  significant digits.
* **`cycles:u`** needs ~40 paired rounds for the sign test at ~1%
  (study 57 §9.3); §5.2 obeys that and says so where it does not apply.
* Byte-identical nulls ride in every panel: `null` = `cp base`,
  `nullpgo-pgo` = `cp base-pgo`, `null-v3` = `cp base-v3`.

---

## 3. ARM `lutp` — the from-square base belongs in the table it is added to

### 3.1 The mechanism

`ThreatResolver::index_decoded` ran, once per threat event **per perspective**:

```rust
let idx = (base + t.offsets[att][from_o] + t.lut2[att][from_o][to_o] as u32) as usize;
```

Two table reads, two address computations and two adds for a sum whose two
operands are **functions of the same `(att, from_o)` prefix**. `offsets` is the
prefix sum of empty-board attack counts before `from`; `lut2` is the ordinal of
`to` among `from`'s targets. Nothing indexes `offsets` anywhere else in the
engine — it existed only to be added here.

Fused at build time into one `from_to[code][from][to]`:

```rust
let idx = (base + t.from_to[att][from_o][to_o] as u32) as usize;
```

### 3.2 Why `u16` is structural and not lucky

`offsets[code][from]` is bounded by the attacker's total empty-board edge count
— `piece_offset`, at most **1456** (queen) — and the ordinal is bounded by the
same quantity, so the sum cannot reach 2·1456. `build_threat_tables` carries a
hard `assert!` (not `debug_assert!`: the table is built once at startup and a
silent wrap is an arbitrarily wrong evaluation that still benches), and
`threats_from_to_is_the_fused_sum` re-derives **both summands from the
definitions the builder used** and checks all 65,536 entries — an independent
recomputation, not a read-back.

### 3.3 The cost, stated

The table goes **64 KiB of `u8` → 128 KiB of `u16`**. That is address space,
not touched lines: a lookup reads ONE `to`-ordinal, so a 128-byte row costs the
same single cache line a 64-byte row did. The panel's load counters (§5) are
where that claim is checked rather than asserted.

### 3.4 Result

**−11.91 i/node native, −15.71 i/node on `x86-64-v3`**, node-identical,
eval-bit-exact. Predicted from the static count as "one load and one add out of
~15 per call"; the measured 0.20% of engine instructions is consistent with
IBS's 0.90%-of-engine line and a ~22% cut of it.

---

## 4. ARM `qrow` — the quantiser, and the difference between a loop and an iterator

### 4.1 What it was spending, off `objdump`

`quantise_halfka` divides the 57.7 MB i16 HalfKA table into i8 by a per-column
scale — **28,835,840 weights at every engine start**. Study 57 replaced its
`idiv` with a reciprocal multiply for −22.5% of startup cycles. The shipped
disassembly of that version is 2×-unrolled scalar, **24 instructions per
weight**, of which:

```
cmp $0x4ff,%rcx / mov $0x4ff,%esi / cmovb %rcx,%rsi     ; the column's range clamp
inc %rcx / cmp $0x500,%rcx / cmovne %rcx,%rdx           ; ... and its wrap        = 6
movzbl (hi) / movzbl (lo) / shl $0x8 / or               ; assembling one i16      = 4
```

plus a bounds check on `recip`, which was a `Vec` whose length LLVM cannot
relate to the column index.

### 4.2 The change

An explicit row walk into a pre-allocated buffer:

```rust
let mut out: Aligned<i8> = Aligned::from_iter_len(n, core::iter::repeat_n(0i8, n));
for (dst_row, src_row) in out.chunks_mut(ACC).zip(raw.chunks_exact(2 * ACC)) {
    let (src, _) = src_row.as_chunks::<2>();
    for (((d, w), &s), &r) in dst_row.iter_mut().zip(src).zip(scale.iter()).zip(recip.iter()) {
        …
    }
}
```

Three ACC-long lists, so the trip count is a compile-time constant and every
index is discharged by construction. `as_chunks::<2>()` yields `&[[u8; 2]]`, so
the i16 is one 16-bit load instead of two byte loads and a shift-or. `recip` is
a `[u64; ACC]`.

### 4.3 ★ Study 57's banner said DO NOT RE-PROPOSE THIS. It is honoured, not ignored — and here is the receipt

`quantise_halfka` carries a ⚠ banner from study 57 §4.4: the row-major rewrite
was **built and lost on every counter** (+10.7% instructions, +19.0% branches,
+36.7% loads, +5.3% cycles, 0/40 on all five). That banner is why this study
built the intermediate arm first.

The banner's own §4.4 names the cause precisely — *"the mechanism is the
collector, not the arithmetic"*: `Aligned::from_iter_len` ends in
`Vec::extend`, and study 57's `FlatMap` is neither `TrustedLen` nor
`ExactSizeIterator`, so `extend` fell off its bulk specialisation onto a
capacity check per weight. **`qrow` does not collect at all.** The destination
is allocated once by a `repeat_n` fill (which *is* the specialised path) and
overwritten in place, so there is no `extend` to de-specialise and no `FlatMap`
anywhere.

The two arms therefore differ in exactly the thing study 57 identified as
causal, and the intermediate arm isolates it:

| | shape | `instructions:u` |
|---|---|---:|
| study 57 (shipped before this study) | `Map<Enumerate<ChunksExact>>` + `i % ACC` | (baseline) |
| **`quant`** | same iterator, cyclic column counter, `[u64; ACC]` | **−29.31 /node** |
| **`qrow`** | explicit row loop, no collector | **−158.27 /node** |

**A stale banner is worth re-checking, and re-checking means reading what it
says the cause was.** Study 58 re-priced a stale candidate and it lost at
+16.96; this one won at −158. Both outcomes came from building.

### 4.4 What LLVM did with it

```
vmovdqu   (%r14,%r10,2),%xmm6          ; 8 raw i16
vpmovzxbq (%rbx,%r10,1),%zmm4          ; 8 scales, widened to u64
vmovdqu64 0x8(%rsp,%r10,8),%zmm5       ; 8 reciprocals
vpaddw / vpmovw2m / vpsubw / vpmovzxwq ; |v| and the sign mask
vpaddq  ×3                             ; n = 2|v| + s
vpmullq %zmm5,%zmm7,%zmm7              ; ← AVX-512DQ: eight 64x64 multiplies
vpsrlq  $0x20 / vpmovqd                ; >> 32, narrow
vpsubd  %ymm4,%ymm1,%ymm7{%k1}         ; negate under the sign mask
vpmaxsd / vpminsd                      ; clamp(-127, 127)
vpmovdb %ymm4,0x0(%r13,%r10,1)         ; store 8 i8
```

**21 instructions for 8 weights = 2.6 per weight**, from 24. The auto-vectoriser
did all of it; there is no intrinsic in the source.

### 4.5 And on `x86-64-v3` it is worth a fourteenth as much

AVX2 has no packed 64×64→64 multiply, so the same source compiles to a
`vpmuludq`/`vpsllq`/`vpaddq` decomposition, 4 lanes wide, computing the positive
and negated results and blending — **~28 instructions for 4 weights ≈ 7 per
weight**. Measured: **−11.25 i/node on v3 against −158.27 on native, 14.1×.**

This is §0.4's rule with its mechanism fully visible, and it is why every
number in §0.1 is given three times.

---

## 5. THE PANELS

Reproduce:

```bash
bash tools/perf59/build59.sh <label>                 # native, non-PGO
P59_CPUFLAG=x86-64-v3 bash tools/perf59/build59.sh <label>
bash tools/perf59/pgo59.sh <label>                   # native + PGO (the shipped build)
bash tools/perf59/allpanels59.sh                     # all four panels, sequentially
python3 tools/perf59/med59.py docs/research/perf-artifacts-59/panel-native.txt base null lutp final
```

All four are blocked Latin squares with a **byte-identical null in the
rotation** (`null` = `cp base`, `nullpgo-pgo` = `cp base-pgo`, `null-v3` =
`cp base-v3`), `flock`ed write-once output, only parsed numbers in the file,
and the node total carried on every marker. Mean position 1.00 of 1.00 (or 1.50
of 1.50) for every label in every panel. Raw panels in
`docs/research/perf-artifacts-59/`.

### 5.1 ★ The shipped binary — native + PGO, 39 rounds

```
REFERENCE per node: L1-dcache-loads 2956.43  branches 552.50  cycles 3538.13
                    instructions 5728.99  ls_dispatch.ld_dispatch 2254.71

label       event                      median %    se     neg/n     p
nullpgo     instructions:u              -0.000   0.000    20/39   1.0000
nullpgo     branches:u                  -0.000   0.000    21/39   0.7493
nullpgo     cycles:u                    +0.104   0.302    18/39   0.7493   ← PASSES
nullpgo     L1-dcache-loads:u           -0.065   0.058    26/39   0.0533
nullpgo     ls_dispatch.ld_dispatch:u   -0.033   0.088    24/39   0.1996

final-pgo   instructions:u              -3.086   0.000    39/39   0.0000
final-pgo   branches:u                  -2.503   0.000    39/39   0.0000
final-pgo   cycles:u                    -0.886   0.207    35/39   0.0000
final-pgo   L1-dcache-loads:u           -1.167   0.023    39/39   0.0000
final-pgo   ls_dispatch.ld_dispatch:u   -1.682   0.037    39/39   0.0000
```

**This is the one panel in the study that carries a cycles verdict**, and it
carries it on exactly the terms study 57 §9.3 set: 39 paired rounds, a null
that passes at 18/39 with p = 0.7493, and an arm at 35/39 with p < 0.0001.
Net of the null, **cycles −0.99%**.

`L1-dcache-loads:u`'s null is marginal here (26/39, p = 0.0533), so the load
win is quoted arm-minus-null: **≈ −1.10% ≈ −32.5 loads/node**. `ls_dispatch`
behaves (24/39, p = 0.20) and reads −1.65% net.

**And `lutp` alone on the shipped build is 2.04× its non-PGO value** — the
amplification study 58 §7 measured at 1.99× on a different arm family, now
confirmed on a third:

```
lutp-pgo    instructions:u              -0.424   0.000    24/24   0.0000    (= -24.29 /node)
lutp-pgo    branches:u                  -0.032   0.000    24/24   0.0000
lutp-pgo    L1-dcache-loads:u           -0.173   0.043    18/24   0.0227
lutp-pgo    ls_dispatch.ld_dispatch:u   -0.225   0.097    18/24   0.0227
lutp-pgo    cycles:u                    -0.872   0.372    15/24   0.3075   ← not evidence
```

`lutp`'s cycles reading is **not quoted**: the null on that 24-round panel is
−0.540% at 15/24 and the arm is −0.872% at 15/24. A ~0.4%-of-instructions arm
cannot be adjudicated on cycles at 24 rounds and this study does not try.

### 5.2 The gate rung — `x86-64-v3`, non-PGO, 24 rounds

```
REFERENCE per node: L1-dcache-loads 2922.41  branches 600.01  cycles 3548.69
                    instructions 5834.43  ls_dispatch.ld_dispatch 2193.32

label      event                       median %    se     neg/n     p
null-v3    instructions:u               -0.000   0.000    13/24   0.8388
null-v3    branches:u                   -0.000   0.000    13/24   0.8388
null-v3    cycles:u                     -0.256   0.309    15/24   0.3075
null-v3    L1-dcache-loads:u            -0.017   0.032    13/24   0.8388
null-v3    ls_dispatch.ld_dispatch:u    -0.001   0.053    12/24   1.0000

lutp-v3    instructions:u               -0.269   0.000    24/24   0.0000   (= -15.71 /node)
lutp-v3    branches:u                   -0.004   0.000    24/24   0.0000
lutp-v3    L1-dcache-loads:u            -0.109   0.026    21/24   0.0003
lutp-v3    ls_dispatch.ld_dispatch:u    -0.282   0.032    23/24   0.0000

final-v3   instructions:u               -0.462   0.000    24/24   0.0000   (= -26.97 /node)
final-v3   branches:u                   +0.003   0.000     0/24   0.0000
final-v3   L1-dcache-loads:u            -0.076   0.039    19/24   0.0066
final-v3   ls_dispatch.ld_dispatch:u    +0.025   0.065     9/24   0.3075
```

The reference row is the receipt for §0.2's claim about which build the
commission's series total describes: **5834.43 i/node**, against the brief's
"5834.4 i/n on the gate build".

Note `final-v3`'s **branches +0.003% (0/24, exact)**. The vectorised quantiser
removes branches wholesale on native (−2.503%); on v3 the AVX2 decomposition
computes both the positive and the negated result and blends, so the branch
saving is spent. It is 0.02 branches/node and it is reported because it is a
sign flip between compilers on an exact counter.

No cycles claim from this panel: the arm reads −0.316% at 16/24 (p = 0.15)
against a null of −0.256% at 15/24. Both are inside the box's cycles floor.

### 5.3 ★ Where the win actually lives — the 1-node startup frame, 39 rounds

```
REFERENCE per 1-node run: L1-dcache-loads 372.4 M  branches 155.9 M
                          cycles 926.7 M  instructions 1684.3 M
                          ls_dispatch 314.3 M

label   event                       median %    se     neg/n     p
null    instructions:u               -0.000   0.000    24/39   0.1996
null    branches:u                   -0.000   0.000    23/39   0.3368
null    cycles:u                     +0.019   0.194    19/39   1.0000   ← PASSES
null    L1-dcache-loads:u            -0.009   0.060    23/39   0.3368
null    ls_dispatch.ld_dispatch:u    -0.020   0.078    23/39   0.3368

final   instructions:u              -47.000   0.000    39/39   0.0000
final   branches:u                  -43.792   0.000    39/39   0.0000
final   cycles:u                    -15.886   0.188    39/39   0.0000
final   L1-dcache-loads:u           -33.796   0.059    39/39   0.0000
final   ls_dispatch.ld_dispatch:u   -33.422   0.077    39/39   0.0000
```

**Startup instructions are nearly halved: −47.000%, −791.6 M per engine
process.** Divided by the pinned frame's 5,001,562 nodes that is **158.3
i/node** — which is `qrow`'s entire native figure of 158.27 to three
significant digits. **The internal consistency is the point: it says the whole
of `qrow` is startup and none of it is search**, without needing to be argued.

Startup **cycles −15.886% = −147.2 M per process**, on top of study 57 arm B's
−263.2 M. Instructions fall 47% and cycles only 16% because what was removed is
cheap ILP-friendly scalar arithmetic and what remains is the 57.7 MB read and
the FNV-1a-64 payload checksum (study 57 §6: ~45% of startup cycles, blocked on
a format ruling).

At this box's clock that is ~29 ms per process. **In a game it is worth
approximately nothing. For `tools/screen.sh`, datagen and every fixed-node
bench — which spawn processes by the thousand — it is the whole point.**

### 5.4 The native non-PGO panel, and why its cycles column is not quoted

```
REFERENCE per node: instructions 5819.68  branches 581.82  cycles 3575.13
                    L1-dcache-loads 2921.73  ls_dispatch 2193.18

label   event                       median %    se     neg/n     p
null    instructions:u                0.000   0.000    11/24   0.8388
null    branches:u                    0.000   0.000     9/24   0.3075
null    cycles:u                     -0.930   0.412    18/24   0.0227   ← FAILS its null
null    L1-dcache-loads:u            -0.100   0.039    17/24   0.0639
null    ls_dispatch.ld_dispatch:u    -0.172   0.080    18/24   0.0227

lutp    instructions:u               -0.205   0.000    24/24   0.0000   (= -11.91 /node)
lutp    branches:u                   -0.004   0.000    24/24   0.0000
lutp    L1-dcache-loads:u            -0.294   0.027    23/24   0.0000
lutp    ls_dispatch.ld_dispatch:u    -0.532   0.042    23/24   0.0000

final   instructions:u               -2.924   0.000    24/24   0.0000   (= -170.18 /node)
final   branches:u                   -2.351   0.000    24/24   0.0000
final   L1-dcache-loads:u            -1.200   0.025    24/24   0.0000
final   ls_dispatch.ld_dispatch:u    -1.477   0.055    24/24   0.0000
```

**The cycles null FAILED on this panel** (−0.930% on byte-identical binaries,
18/24, p = 0.0227) — the box shares a live GPU trainer — so nothing on that row
is evidence and none of it is quoted. The exact counters are unaffected:
`instructions:u` and `branches:u` are 0.000 median at 0.000 se on the null,
the fourth confirmation of that on this box after studies 48, 57 and 58.

**`lutp` answers §3.3's cache question by measurement rather than argument.**
Doubling the fused table from 64 KiB to 128 KiB *reduces* loads: −0.294%
against a null of −0.100%, i.e. **≈ −0.19% net ≈ −5.7 loads/node**, with
`ls_dispatch` agreeing at −0.36% net. The deleted `offsets` load is worth more
than the wider table costs, which is what "a lookup reads one `to`-ordinal
either way" predicted.

---

## 6. WHAT THIS STUDY DID NOT ESTABLISH

Most doubtful first.

1. **No game has been played, none is owed, and NO ELO IS CLAIMED.** Both arms
   are node-identical at 5,001,562 and eval-bit-exact, so a fixed-node match
   against the base is a dead heat by construction (study 49 §8 item 6, study
   52 §9.1). The temptation is to convert −0.886% of shipped cycles at ARCHON's
   ~122.9 Elo per speed doubling and quote ~1.6 Elo; **that conversion is wrong
   here**, because §5.3 shows essentially all of the cycles win is the fixed
   startup cost, which a real time control amortises to nothing. The honest
   statement is: `lutp` is −24.29 i/node of *search* work on the shipped binary
   whose cycles value this box cannot resolve at 24 rounds, and `qrow` is
   −147.2 M cycles per *process*.
2. **The `lutp` cache claim rests on a counter with a documented floor.**
   "A 128-byte row costs the same one line a 64-byte row did" is a statement
   about the access pattern; what falsifies it is `L1-dcache-loads:u`, which
   reads −0.294% against a null of −0.100% on the native panel and −0.109%
   against −0.017% on v3. Both point the same way and both are quoted
   arm-minus-null — but study 58 §6 caught this counter FAILING its own null at
   n = 24 and passing at n = 39, and my nulls here are −0.100% (17/24,
   p = 0.0639) and −0.065% (26/39, p = 0.0533). Neither quite fails and neither
   is clean. The load win is directionally established and its magnitude is
   soft.
3. **I did not land in `search::<NonPv>`, the commission's largest target.**
   §0.5 item 1 is a map and a list of prior numbers, not a proof that nothing
   is left. Two things I priced and declined (items 2 and 3) are ~4 i/n and
   ~0.4% of cycles respectively; a third (the `[Move::NULL; 32]` initialisation
   of `quiets_searched`/`captures_searched` at every node, 0.46% of the symbol
   ≈ 7.6 i/n) needs `MaybeUninit` and I judged the unsafe not worth 7.6
   instructions. Those are judgement calls and they are recorded as such.
4. **The `quant` intermediate is measured but not shipped**, so its +8.53
   branches/node is a number about a binary that does not exist. It is reported
   because the 129 i/n gap between it and `qrow` is the study's transferable
   finding, not because the arm itself matters.
5. **The startup share is a property of THIS frame.** `qrow` is worth
   158.27 i/node because the pinned bench is 5,001,562 nodes in one process. At
   half the nodes it is worth twice as much per node and exactly as much in
   absolute terms; in a game it is worth ~0.
6. **IBS's own skid is uncharacterised in this study.** It tags at issue and I
   did not run a known-answer probe (study 49's `skidprobe` shape). Its per-line
   shares were used to CHOOSE an arm, and the arm was then adjudicated by a
   paired build — which is the right dependency direction, but it does mean no
   number in §1 has a null.

---

## 7. FOR THE NEXT STUDY

1. **Build both compilers, every time.** This study and the brief now have two
   independent 6× receipts pointing opposite ways. `build59.sh` takes
   `P59_CPUFLAG`; there is no excuse left.
2. **Do not read a line census off `instructions:u`.** §0.3. `ibs59.sh` +
   `stack59.py` cost one extra binary and three bench runs.
3. **`quantise_halfka` is now 2.6 instructions per weight on native and ~7 on
   v3.** If the portable rung ever matters, an explicit AVX2 path that keeps the
   numerator in 32 bits would close most of that — the exactness condition
   (`N < 2³²/e`) is what blocks a straight 32-bit reciprocal, and it would need
   a per-column `k`, which is real work for a startup-only win.
4. **The largest single line left in `at_full` after `lutp` is
   `sota_features.rs:545`, `lut1[att][atd][dir]`, at 4.25% of the symbol.**
   The `dir` axis is only ever *different* from `[0]` for semi-excluded pairs,
   so a `[u64; 16][16]` packing both entries and selecting by a shift is one
   spelling worth pricing — study 58's rule applies (a control-instruction count
   is an estimate of what LLVM will charge you), so price it by building it.
5. **Startup is now ~4.6% of the pinned frame's instructions** (`load_v2_bytes`
   1.60%, `quantise_halfka` down from 2.94%, `init_magics` 1.01%). Study 57 §6
   named the FNV-1a-64 payload checksum at ~45% of startup CYCLES and it still
   needs a format ruling, not a measurement.
