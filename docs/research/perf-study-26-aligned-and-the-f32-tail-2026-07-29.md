# Perf study 26 — `Aligned`, the head's weight blocks, and the f32 tail

**Date:** 2026-07-29/30 · **Box:** 7700X (Zen 4), AVX-512 VBMI2, 32 KB L1i /
32 KB L1d, 1 MB L2/core, 32 MB L3 · **Net/arch:** D48X, `sota,arch-d48x`,
ACC 1280, L2 16, L3 32 · **Base commit:** `44d20b7`

**Scope:** `src/eval/nnue/{aligned,sota_head,sota_loader,sota_features,sota_simd}.rs`.
Sibling studies own `search/mod.rs`+`movepick.rs` and `board.rs`+`sota_threats.rs`;
nothing outside these five files is touched.

**Brief:** land four optimisations, starting with the crate-wide `Aligned`
defect study 25 measured but could not fix in its scope. Gate on
`instructions:u` AND `branches:u` together, at **CGU = 1**. Do not rank against
Stockfish.

---

## ★ Headline

**Four landed arms, `−55.61 instructions/node (−0.959%)` and `−6.31
branches/node (−1.05%)`, every one of them exact against a null that measured
EXACTLY ZERO on both counters.** Node count 5,001,562 and eval hash
`5ca978e012da1bb0` unchanged on every arm including the rejects, and both
NNUE-gate pins reproduce on the **AVX2** build (`5ca978e0…8b38` / `1922947`).

`forward_into` — the head's whole forward pass — went **6114 → 3655 bytes,
1274 → 774 static instructions, 69 → 19 conditional branches.**

**Four rejections, each with the number that killed it**, and **one refutation
of the prediction I inherited**.

| | wl50 i/n | Δ | br/n | Δ |
|---|---|---|---|---|
| base `44d20b7` | 5798.797 | — | 600.914 | — |
| **null** — inert source change, rebuilt | **5798.797** | **0.000** | **600.914** | **0.000** |
| *probe* — raw pointers, all checks gone | *5776.871* | *−21.93* | *594.528* | *−6.39* |
| **a1** `Aligned::deref` → `get_unchecked` | 5790.495 | **−8.30** | 597.934 | **−2.98** |
| **a2** head weight tables → `Blocks<T,N,B>` | 5772.560 | **−17.94** | 594.608 | **−3.33** |
| **a3** `l3_dot` becomes a dispatched kernel | 5750.210 | **−22.35** | 594.608 | 0.00 |
| **a4** the two dead heads go `#[cold]` | **5743.183** | **−7.03** | **594.608** | 0.00 |
| **cumulative** | **5743.183** | **−55.61 (−0.959%)** | **594.608** | **−6.31 (−1.05%)** |

---

## 0. The instrument, and why it is CGU = 1

**No cycles number appears in this document, in either direction.** Study 25
§0.3 settled that: a byte-identical binary measured +0.97%, +2.06% and −1.93%
against itself in one night, and my own coordinator measured −1.653% on a quiet
box. The axis is dead until the trainer is off the machine.

The gate is therefore `instructions:u` **and** `branches:u` per node, both
required to agree in sign, and it is run at **CGU = 1 with fat LTO — the
shipped release profile.** Every prior probe script
(`probe20/21/22/24/25.sh`, `build22.sh`, `perf24/build24.sh`) overrides that
with `CARGO_PROFILE_RELEASE_LTO=thin CARGO_PROFILE_RELEASE_CODEGEN_UNITS=4`,
and at 4 CGUs the instruction counter is *also* unreliable — study 23 saw an arm
read +48 i/n while its loop body shrank 189 → 102 static instructions, with
untouched code blamed for +382. `tools/perf26/probe26.sh` is `probe25.sh` with
that override deleted; nothing else changed.

**The null, run first, and it is exactly zero.** Not the byte-identical-copy
null (which is trivially zero on this counter) but an **inert source change**:
rename `Aligned`'s private `off` field to `off_`, rebuild the whole crate,
re-measure.

| | instructions | i/n | br/n |
|---|---|---|---|
| base | 29,003,410,773 | 5798.87 | 600.93 |
| null (renamed field, rebuilt) | 29,003,410,200 | 5798.87 | 600.93 |
| | **−573 in 29.0 G = −0.0000020%** | **0.000** | **0.000** |

Four further rebuilds of semantically-identical source (a `cargo fmt`
whitespace change, an intrinsic respelling that LLVM canonicalised away, and the
final HEAD rebuild) landed within **±700 instructions in 29 G** of their twin —
**0.000 i/n and 0.000 br/n at the precision reported throughout**. That is the
null band for this instrument: two decimal places on i/n, and it is three
orders of magnitude below the smallest arm claimed here.

Run-to-run spread on one binary, n = 4: **0.0002 i/n**. Every number below is a
median of ≥ 2 runs and the spread never exceeded 0.0004.

**Frame:** `bench 16 1 100000 tools/nnue/calib/wl50.fens nodes
engines/canonical/net.cenn` — a fixed 5,001,562-node tree, the same frame
studies 21–25 used, so the numbers compose with theirs.

---

## 1. ★ The kill-probe first, as instructed — and it kills the prediction, not the item

Study 25 §7 predicted that converting the head's five bucket-selected weight
blocks to thin, compile-time-sized boxes would measure **−35 to −60 i/n**, and
pre-registered a falsifier: *a one-build probe that replaces the acquisitions
with raw-pointer arithmetic — same loads, no checks, no `Deref` — measuring less
than −20 i/n kills the whole line.*

That probe was the first thing built. All six acquisitions
(`l1_weights`, `l1_biases`, `l2_weights`, `l2_biases`, `l3_weights`, and the
skip lane) were replaced with `&*(self.buf.as_ptr().add(self.off + base) as
*const [T; N])`.

| | i/n | br/n | nodes | eval hash |
|---|---|---|---|---|
| base | 5798.797 | 600.914 | 5,001,562 | `5ca978e012da1bb0` |
| **raw-pointer probe** | **5776.871** | **594.528** | 5,001,562 | `5ca978e012da1bb0` |
| | **−21.93** | **−6.39** | | |

**−21.93 i/n. The line survives its own kill test by 1.93 i/n, and the −35 to
−60 prediction is refuted by the same measurement.** The absolute ceiling on
removing *every* check from the head's acquisitions is 21.93 instructions per
node; the prediction was 1.6–2.7× over.

That should have been the whole story for §3. It was not, and the reason is
§3's punchline.

---

## 2. LANDED — a1: `Aligned::deref` stops re-checking a bound it owns

`src/eval/nnue/aligned.rs`, commit `06038a2`. **−8.30 i/n (−0.143%), −2.98
br/n (−0.496%).**

**`Aligned<T>` is not a `Box<[T]>`.** It is a `Vec<T>` plus an offset, and
`Deref` was `&self.buf[self.off..]` — *a bounds-checked range slice re-derived
on every access to every weight table in the engine, before any indexing.*
Study 25 §4 found this while investigating why `as_chunks` views of the head
blocks bought only −0.5 i/n: the change traded one bounds check for another and
left the expensive part untouched. `aligned.rs` was outside its four files.

Both derefs now use `get_unchecked`. The crate is `#![deny(unsafe_code)]`, so
this needed a real safety argument rather than a shrug. The invariant is named
**`OFF_LE_LEN`** (`self.off <= self.buf.len()`) and argued in the module docs in
three parts, none of which is an appeal to the caller:

1. **Both fields are private and this module is the whole surface** — `off` is
   written in exactly one function.
2. **That function establishes it by construction** — `buf.resize(off, …)` makes
   `buf.len() == off` exactly and `buf.extend(iter)` can only grow it. `Aligned`
   exposes no `truncate`/`pop`/`clear`/`into_boxed_slice`, and `DerefMut` hands
   out `&mut [T]`, which cannot change a `Vec`'s length.
3. **Debug builds re-check it on every access** — `Aligned::debug_check` runs in
   front of both derefs, so the entire test suite and every debug run of the
   engine assert the invariant millions of times at the exact point release
   assumes it. A new constructor that got it wrong could not survive one
   `cargo test`.

**Why only −8.30 crate-wide, when the head alone was worth up to −21.93?**
Because the four *big* tables are deref'd **once per context construction, not
per fold**: `HalfkaContext::new(&net.halfka_weights, &net.ft_bias, …)` and
`ThreatContext::new(&net.threat_weights, …)` take `&[T]` and hold it. The
per-eval derefs are the head's six. So "crate-wide" is true of the change and
not of the saving — worth stating, because the obvious reading of "every weight
table in the engine" over-promises.

---

## 3. LANDED — a2: the head's weight tables become `Blocks`, and beat the probe

`src/eval/nnue/{aligned,sota_head,sota_loader}.rs` + `tests/cenn_v2_smoke.rs`,
commit `d68c6be`. **−17.94 i/n (−0.310%), −3.33 br/n (−0.556%).**

Each of the five bucket-selected tables was an `Aligned<T>` sliced as
`self.field[base..base + N].try_into().expect(..)`. In the disassembly that is
eight scalar instructions and a branch per acquisition:

```
mov  rdx, [r14+0x30]   ; buf.len
mov  rcx, [r14+0x38]   ; off
sub  rdx, rcx          ; payload length
lea  rsi, [rax+0x1f]   ; base + N - 1
cmp  rsi, rdx
jae  panic             ; the range check and the try_into, merged
shl  rcx, 2
add  rcx, [r14+0x28]   ; buf.ptr
```

New `Blocks<T, N, B>` = `Box<[Line<T, N>; B]>` with `#[repr(align(64))]` on the
block. **No `unsafe`, and no offset trick:** alignment is a property of the
*type*, so the allocator gives it to every block rather than only the table
base, and `into_boxed_slice`'s shrink-to-fit — the trap `Aligned` has to dodge —
cannot lose it. `Blocks::LAYOUT_OK` asserts at compile time that
`N · size_of::<T>()` is a multiple of 64, which holds at every shipped arch
(d47v, d48w, d48x, hka2048 and the default 1280/128/32 — all five were
`cargo check`ed), so there is no interior padding and the on-disk block-major
layout is byte-for-byte what it was.

### ★ 3.1 It beats "no checks at all", and the reason is the mechanism

| | i/n | vs a1 |
|---|---|---|
| a1 | 5790.495 | — |
| raw-pointer probe (no checks, no `Deref`) | 5776.871 | −13.62 |
| **a2 `Blocks`** | **5772.560** | **−17.94** |

**`Blocks` is 4.31 i/n better than removing the checks by hand.** The
disassembly says why: `forward_into` now contains **no bucket compare at all**
— a scan of every `cmp` immediate in the function finds `0x7f800000` (an
`is_finite` test), `0x2e0`, `0x560`, `0x21`, `0xe`, `0x7f`, and nothing that
could be a bounds check.

Against a runtime-derived payload length LLVM had nothing to fold. Against the
*constant* `BUCKETS` it discharges the check statically, because
`bucket = OUTPUT_BUCKETS_LAYOUT[piece_count]` is a `const [usize; 33]` whose
every entry is < 8 — a fact the compiler already had and could not use.

**Stating the shape in the type did not remove a check; it let the compiler
prove one it had been re-testing five times per eval.** That is a better outcome
than `unsafe`, and it is the generalisable lesson: when a bound is a
compile-time constant, the safe form can be *faster* than the unchecked one.

Also landed with it: the head's short-table panic moves from every eval to load
time (where CENN section-size validation has already made it unreachable), and
`read_f32s_aligned` is deleted — its five callers are now the blocks.

**Static effect:** `forward_into` 6114 → 5711 bytes, 1274 → 1192 instructions,
69 → 60 conditional branches.

---

## 4. LANDED — a3: the head's last autovectorized f32 path was running 2 lanes wide

`src/eval/nnue/{sota_simd,sota_head}.rs`. **−22.35 i/n (−0.387%)**, the biggest
arm in the study.

Found by line-attributed profiling of a `--profile profiling` build, not by
reading the source: **`sota_head.rs:399` — one line, the FMA inside
`propagate_l3` — was 4.56% of `forward_into`'s executed instructions.** For a
**32-element dot product**. The static disassembly of that line: sixteen
`vmovsd` (8-byte, i.e. *two-float*, loads), four `vextractf32x4`, four
`vshufpd`, and sixteen 2-lane `vfmadd213ps`.

The source looked like it should vectorize — a `while` over 16-lane
`try_into()`'d sub-arrays with a per-lane `mul_add` — and it is the shape three
prior studies left in place. It does not. This is study 25's own lesson in a new
place: **read the disassembly before assuming an optimisation exists, and
before assuming an existing one does.**

Fix: `l3_dot` becomes a dispatched kernel like every other head kernel
(`l1_dot`, `activate`, `find_nnz`, `l1_dequant`, `l2_tail`, `policy_dot`,
`skip_dot_i8`). The scalar backend **is the loop that used to live in
`propagate_l3`, moved verbatim**, so the SIMD backends have a bit-exact
reference rather than a description.

- **avx512**: one `zmm` accumulator *is* the reference's 16 partial lanes, so
  the whole dot is two `vfmadd231ps` plus a four-step halving tree.
- **avx2**: two `ymm` accumulators are lanes 0–7 and 8–15; their sum is the
  reference's stride-8 step, then the same tree.
- The last three tree steps are one shared `halve8` helper, so the two SIMD
  backends cannot drift from each other or from the reference.

**Bit-identity is a proof, not a hope.** The reduction pairs the same lanes in
the same order, so f32 non-associativity is respected rather than assumed away;
`mul_add` and `_mm512_fmadd_ps` are the same correctly-rounded fused operation.
Confirmed three ways: the eval hash is unchanged, the new cross-backend test
compares every available backend against the scalar reference at exact f32
equality over four magnitude scales spanning 10⁻³ to 10⁶ (chosen so that a
*re-associated* tree would actually round differently), and **both NNUE-gate
pins reproduce on the AVX2 build** — a completely different code path.

**Static effect:** `forward_into` 5711 → 5444 bytes, 1192 → 1155 instructions.

---

## 5. LANDED — a4: two heads no shipped net has were inflating the hottest function

`src/eval/nnue/sota_head.rs`. **−7.03 i/n (−0.122%).**

The same line-attributed profile showed **`sota_head.rs:461` as the single
largest static line in `forward_into`: 99 instructions, 32 × `lea`/`cmp`/`jae`.**
That is `propagate_l3_wdl`'s `ww[base + j]` — thirty-two bounds checks on a
runtime-length `Box<[f32]>`, plus a 31-deep chain of **scalar** `vfmadd231ss`
because the accumulator is a serial `s`.

**It never executes.** `l3_wdl_weights` and `l3_var_weights` are `None` for
every net in `engines/`; both heads load from optional sidecars nothing ships.
Inlined into `forward_into` they contributed ~140 static instructions and 41
conditional branches to the hottest eval function for zero executed work.

`#[cold] #[inline(never)]` on both — the same treatment, and the same reason, as
`i16_deltas_into_generic`. The `Option` test that guards them is unchanged, so
this is behaviour-preserving by construction.

| | bytes | static instrs | cond. branches |
|---|---|---|---|
| `forward_into` at a3 | 5444 | 1155 | 60 |
| **`forward_into` at a4** | **3655** | **774** | **19** |

**−33% of the hot function's code and 41 of its 60 branches.** The −7.03 i/n is
register allocation in the hot path improving once the cold bodies stop
competing for it — the executed instruction count of unreachable code is zero by
definition, so nothing else can explain it. This is the smallest of the four
wins and the one with the strongest independent justification.

---

## 6. REJECTED, each with its killing number

### 6.1 `#[inline(never)]` on the avx512 `l2_tail` — **+4.26 i/n**

`l2_tail`'s 16 `_mm512_set1_ps(l1[i])` broadcasts were **not** compiling to
`vbroadcastss zmm, m32`. `l1` is the caller's `[f32; L2]`, which `l1_dequant`
has just written with a single `zmm` store; LLVM forwards that store, so each
broadcast became an extract-from-register — three `vextractf32x4`, twelve
`vshufpd`/`vpshldq`/`vmovshdup`, then sixteen `vbroadcastss`. 36 instructions
for 16 broadcasts, and 3.14% of `forward_into`.

Outlining the kernel forces the memory read, and **it worked as a mechanism** —
the out-of-line body has exactly 17 `vbroadcastss` and no shuffle chain — and
still **lost 4.26 i/n and 0.85 br/n**, because the caller then spills its live
state around the call. Reverted.

### 6.2 A `black_box`'d `l1` pointer in `l2_tail` — **−3.19 i/n, and rejected anyway**

The same fix without the call: read `l1` through `std::hint::black_box(l1.as_ptr())`
so LLVM cannot relate the loads to the store. Node-identical, hash-identical,
**−3.19 i/n (−0.055%)**.

**Rejected on grounds, not on the number.** `black_box` is documented as a hint
the compiler is free to ignore; landing 0.055% on an unguaranteed optimisation
barrier, in a kernel, with no test that could detect its silent loss, is
exactly the sort of thing that rots. Study 25 declined to claim −3.2 i/n as a
win for a change that *also* deleted 74 lines of dead code; this one adds a
line and explains it. **The whole `l2_tail` broadcast line is worth at most
−3.19 i/n — a third of what its profile share suggested.**

### 6.3 `_mm_load_ss` + `_mm512_broadcastss_ps` — **exactly nothing**

The canonical intrinsic spelling of "broadcast from memory", tried before 6.1.
28,871,814,594 → 28,871,814,416 instructions: **inside the ±573 null band.**
LLVM's store-forwarding happens at IR level and does not care which intrinsic
spelled the load.

### 6.4 `(0..H).step_by(32)` → a `while` walk in `activate` — **−0.43 i/n**

`step_by.rs:446` carried 2.75 profile units, and study 25 §4a had established
the direction (its `while off + TILE <= ACC` beat both counted forms). Applied
to `activate`'s tile walk — the last `step_by` in a hot head kernel — it
measured **−0.43 i/n**. A null. `step_by` on a constant-bound range was already
being folded here; study 25's result was about *unrolling*, not about `step_by`.

### 6.5 Masking `att`/`atd` in `ThreatResolver::index_decoded` — killed by disassembly, no build

`index_decoded` already masks `from`/`to` with `& 63` as "a NO-OP on the value
and a RANGE FACT for the optimizer". The obvious sibling is `& 15` on the two
piece codes, which index `lut1[16][16][2]`, `offsets[16][64]` and
`lut2[16][64][64]` — four potential bounds checks per event per perspective.

**There are none.** A scan of every `cmp` immediate in `at_full` finds `0xff`
(the event-buffer cursor, study 24's), `0xed2f` = 60719 = `THREAT_DIMS − 1` (the
real exclusion test, not a bounds check), `0x3f`, `0x37`, and nothing at `0xf`
or `0x10`. LLVM already derives `att < 16` through `sf_code`'s
`((color as usize) << 3) | (pt as usize + 1)` from the enum ranges. Cost of this
rejection: one `objdump`.

### 6.6 A pre-shifted `nnz` array to remove `l1_dot`'s `shl` — killed by analysis

`l1_dot`'s inner loop is exactly four instructions per non-zero group —
`movzx` the index, `vpbroadcastd` the four FT bytes at `group·4`, `shl` by 6,
`vpdpbusd` against the 64-byte weight row at `group·64`. The `shl` is 3.60% of
`forward_into` and looks free to remove by storing the scaled index.

It is not, and x86 addressing says why: the two strides differ by 16× and the
scale field only offers 1/2/4/8, so **no single stored value serves both
operands**. Storing two arrays replaces the `shl` with a second `movzx` — the
same instruction count — and costs `find_nnz` an extra store per chunk. The only
encodable alternative needs the FT groups at an 8-byte stride, which breaks
`activate`'s dense `vpmovwb` store and `skip_dot_i8`'s dense read. **`l1_dot` at
four instructions per group is minimal on this ISA.**

---

## 7. Things confirmed minimal, so nobody re-derives them

- **`activate`** (~26% of `forward_into`): per 32 outputs it is 4 loads, 2
  `vpsllw<2>`, 2 `vpaddsw`, 2 `vpmaxsw`, 2 `vpminsw`, 1 `vpsllw<7>`, 1
  `vpmulhw`, 1 fused `vpmovwb`-to-memory. The clamp is 2 ops per half and
  AVX-512 has no 3-operand clamp; the saturating add cannot become a wrapping
  one without changing results; the `<<2` cannot move into the fold (it would
  cost ~186 extra instructions per threat fold to save 40 per eval — measured
  ratio ~19:1 against) and cannot move into the table without doubling 77.7 MB.
  The store is already the fused `vpmovwb ymmword ptr [mem], zmm` form.
- **`find_nnz`** (6.16%): 2 loads, 2 `vptestmd`, 1 `kunpckwd`, 1 `vpcompressw`,
  1 `vpaddw`, 1 store per 32 groups. Pre-baking 10 offset iotas trades the
  `vpaddw` for a load; keeping one iota in a register is already better.
- **`i16_deltas_into`**: for a (1,1) fold the shaped kernel is `vmovdqu64` src,
  `vpaddw` [add row], `vpsubw` [sub row], `vmovdqu64` dst — **four memory
  operations for four unavoidable memory accesses.** There is nothing to
  remove inside the kernel.

---

## 8. ★ Ranked remainder

1. **ARCHON maintains TWO 1280-wide accumulators where one would do, and the
   head only ever reads their sum.** `activate` is the only consumer of `pst`
   and `threat`, and it consumes them as `clamp(pst + 4·threat, 0, 255)`. The
   sum is **linear** — only the clamp is not — so a single accumulator holding
   `pst + 4·threat` would be *behaviourally identical* and would delete: one of
   the two parent→child copies per fold (40 `zmm` loads + 40 stores), 2 of
   `activate`'s 4 loads per iteration, and both `vpsllw<2>` and both `vpaddsw`
   — 6 of `activate`'s 16 ops per iteration plus half the fold copy traffic.
   Stockfish does exactly this (`eca43a97:src/nnue/nnue_accumulator.cpp:237-238`
   merges its second feature set with one `vec_add_16`). Study 25's remainder
   item 3 saw the copy half of this and called it "study 22's, the caller's
   design"; the *merge* half is additional and larger. **Cost:** the threat fold
   would pay `vpmovsxbw` + `vpsllw<2>` + `vpaddw` (3 ops per row-register)
   instead of 2, and — the real blocker — the HalfKA **Finny cache** is keyed by
   `(perspective, king-bucket, mirror)` and cannot cache a threat-contaminated
   accumulator. Wants a design note before an arm. Spans
   `sota_accumulator.rs` + `sota_threats.rs` + `sota_context.rs`, all outside
   this study's files. **Estimated 60–120 i/n, by far the largest item left.**
2. **`Aligned<T>` should probably not exist any more.** `Blocks` (§3) obtains
   64-byte alignment from `#[repr(align(64))]` on the element type, with no
   padding offset, no `unsafe`, and per-*block* alignment rather than
   per-table. The four remaining `Aligned` tables (`halfka_weights`, `ft_bias`,
   `threat_weights`, `psqt_weights`) are row-major with a fixed row width of
   `ACC`, so they are exactly `Blocks<T, ACC, N_FEATURES>` — which would also
   give every FT *row* its own cache line, not just the table base. Zero
   instruction win expected (their deref is per-context, §2), but it retires
   the offset hack and its `unsafe`. **A tidy-up with a real safety payoff, not
   a perf arm.**
3. **`load_v2_bytes` is 1.57% of the wl50 frame** (`read_i16s`/`read_i8s_aligned`
   walk 135 MB element-by-element). **Worth ~91 i/n on the instrument and
   ZERO for playing strength** — an engine loads its net once. Landing it would
   move the frame by more than this entire study's four arms and mean nothing.
   If it is ever fixed, it must be reported as a startup-latency change and the
   wl50 baseline re-pinned in the same commit. **Do not let it into a perf
   headline.**
4. **`apply_i16_deltas_dual` (4.76% of PGO cycles)** still has the runtime tile
   walk study 25 arm 1 removed from its twin. Study 25 priced it at "well under
   0.2%" and I did not re-price it; its feature lists are long (Finny refresh),
   so loop control is already amortised.
5. **The `l2_tail` broadcast chain is worth −3.19 i/n and no more** (§6.2).
   Anyone who sees the 36-instruction shuffle storm in the disassembly and gets
   excited should read that section first — the profile share (3.14%) is triple
   the real cost, because the 16 *necessary* broadcasts are on the same lines.
6. **Threat-row locality** — unchanged from study 25 §8.5: a permutation of the
   threat feature index space is bit-identical and could co-locate co-occurring
   rows, but choosing it needs co-access statistics. A research task.

---

## 9. Do not re-propose

- **`#[inline(never)]` on a head SIMD kernel to force memory operands** — §6.1,
  +4.26 i/n. It fixes the codegen and loses on caller spills.
- **`black_box` to defeat store-forwarding into `l2_tail`** — §6.2, −3.19 i/n,
  rejected on grounds.
- **`_mm_load_ss` + `_mm512_broadcastss_ps` as a "load from memory" idiom** —
  §6.3, exactly zero. The forwarding is at IR level.
- **`step_by` → `while` in `activate`** — §6.4, −0.43 i/n.
- **`& 15` masks on the threat resolver's piece codes** — §6.5, the bounds
  checks do not exist.
- **A pre-scaled `nnz` index to remove `l1_dot`'s `shl`** — §6.6, x86 has no
  scale-16.
- Everything in study 25 §9 still stands.

---

## 10. For the next study

**Use `tools/perf26/probe26.sh`, not `probe25.sh`.** The only difference is that
it does not override the release profile, and that difference is the whole
reason the numbers above have a 0.000% null instead of a ±0.3% one. The build is
slower by about a minute.

**Build the AVX2 path at least once.** `tools/perf26/nnuepins26.sh` reproduces
both NNUE-gate pins from a `target-cpu=x86-64-v3` build in one command. The
gate builds v3, so *the merge gate checks the avx2 backend of every dispatched
kernel* — a study that only ever builds `target-cpu=native` has not run the code
that gates its own merge. It caught nothing here, but it is the difference
between "the eval hash matches" and "the eval hash matches on the path the
reviewer will run".

**Profile with line attribution, not just symbols.** Both of the two largest
findings in this study (§4, §5) are invisible in a per-symbol profile — they are
single source lines inside a function three prior studies had already read.
`--profile profiling` + `objdump -d -l` + `perf annotate`, joined on address; the
join script is fifteen lines and is reproduced in §4's commit message.
