# Perf study 58 — `current_full` on the binary we actually ship

**2026-07-31. Branch `worktree-agent-afe02e02380931853`.**

**Rebase check (the `bcmisc55` rule).** The study was measured on study 57's
branch; `main` merged study 57 and moved eleven commits while it ran. Rebased
onto `43ed471a` and **re-measured**: `git diff HEAD~2 247a0d4d -- src Cargo.toml`
is empty (main's new commits are docs and tools), the post-rebase baseline
builds to the **same `.text` md5 `8876f37d`** — verified by building, not by
mtime — and the landing set re-reads **−152,220,000 instructions = −30.43/node**
against it, the same figure to five significant digits. Gate re-run green on the
rebased tree.

## 0. What landed

Three arms, all node-identical at **5,001,562** and eval-bit-exact
(`2e1ede95…79c4`), all inside `HalfkaContext::compute` / `apply_dirty_directed`
/ `DirtyPiece`:

| arm | what | `instructions:u` | `branches:u` |
|---|---|---:|---:|
| **dpref** | `compute` reads the dirty record where it lives (disjoint field borrows) instead of copying 14 bytes to the stack and reading them back | **−21.70 /node** | −0.225 /node |
| **dpmin** | `DirtyPiece::adds()/subs()` bound the slice by the array length as well as the counter | **−9.30 /node** | **−9.79 /node** |
| **zip** | `apply_dirty_directed` zips the destination arrays instead of indexing them by an `enumerate` counter | **−2.67 /node** | 0.00 |
| **landed together** | | **−30.43 /node (−0.520%)** | **−9.88 /node (−1.670%)** |

Two arms were **built and lost**, and both are recorded in the ⚠ banner on
`i8_deltas_into_generic` where they would be re-proposed:

| arm | `instructions:u` | `branches:u` |
|---|---:|---:|
| **pair** — study 37 arm 3 re-priced at `TILES = 8` | **+16.96 /node (+0.290%)** | +3.35 /node |
| **quad** — hoist the row loop's unroll prologue out of the tile walk | **+63.35 /node (+1.083%)** | +16.22 /node |

One arm was an exact null and is not in the tree: masking `DirtyPiece::add`'s
write cursor with `& 1` measured **+690 instructions in 2.93 × 10¹⁰**. LLVM
already proves that bound; there was no check to delete.

**PGO pair — and it goes the opposite way from the commission's expectation:**

| build | i/node | branches/node |
|---|---:|---:|
| base, non-PGO (the gate build) | 5850.12 | 591.70 |
| **landed, non-PGO** | **5819.69** (**−30.43**) | 581.82 |
| base, PGO (`build_pgo.sh sota,arch-d48x` + canonical net) | 5789.63 | 561.01 |
| **landed, PGO** | **5729.02** (**−60.62, −1.047%**) | 552.50 (−8.51) |

The commission said PGO absorbs ~38% of a hand-optimised instruction win
(−263.1 i/n non-PGO → −162.4 PGO across studies 52–57). **Here it nearly
doubles it: 1.99×.** §7 argues that is the complement of the 38% rather than a
contradiction of it, and that the mechanism is exactly the one the commission
insisted on.

**And on the shipped binary the cycles win resolves.** 39 paired rounds with a
byte-identical PGO null in the rotation:

```
final-pgo  cycles:u  -1.272%   se 0.304   30/39   p = 0.0011
nullpgo    cycles:u  +0.395%   se 0.574   17/39   p = 0.52   (coin flip)
```

That is the first cycles number this study is willing to quote, and it is
quoted only because n = 39 is what study 57 §9.3's calibration says a ~1%
effect needs. **The 24-round non-PGO panel read −0.269% at 14/24 and is
explicitly NOT evidence** (§6). The same panel also shows a **loads regression**
under PGO — `L1-dcache-loads +0.273%`, 3/39 — which is real, has a mechanism
(§7), and does not cost the cycles win.

Gate: `CHECK_NNUE_GATE=1 bash tools/check.sh` → `All checks passed.`,
`Bench: 587683`, eval hash `2e1ede95…79c4`, d13 `1739588`, release
`perft_full_suite ok`.

---

## 1. The commission's profile replicates — with one correction

Taken on my own PGO build of post-57 main (`build_pgo.sh sota,arch-d48x
p58-base engines/canonical/net.cenn`), `perf record -F 6999`, the pinned
5,001,562-node frame:

| function | commission | this study | |
|---|---:|---:|---|
| `search::<NonPv>` | 23.53 | 23.04 | |
| `HalfkaContext::current_full` | 12.40 | 13.18 | |
| `avx512::i8_deltas_into_generic` | 10.21 | 9.76 | |
| `SotaHead::forward_into` | 9.30 | 9.88 | |
| `MovePicker::init_quiets` | 7.64 | 7.16 | |
| `ThreatContext::at_full` | 7.19 | 8.18 | |
| `HalfkaContext::finny_refresh_into` | 6.38 | 6.25 | |
| `qsearch::<NonPv>` | 4.07 | 4.03 | |
| `sota_loader::load_v2_bytes` | 2.89 | 3.00 | |
| `sota_threats::record_square_touch` | 2.51 | 2.73 | |
| **`sota_loader::quantise_halfka`** | **2.36** | **1.00** | ← study 57 |

Everything is inside sampling noise at 3.7 k–20 k samples **except the last
row**, and that one is not noise. `quantise_halfka` halving is the signature of
**study 57 arm B** (the `idiv`-free reciprocal quantiser, −22.5% startup
cycles). Study 57 is not merged into `main`; my base is its branch. So **the
commission's profile was taken on a binary WITHOUT study 57**, and the 1.36
points of difference is exactly where that study's second arm landed. No claim
in the brief depends on it, but the table should be read as pre-57.

## 2. `current_full` is 13.18% of cycles and 2.52% of instructions

The commission names `current_full` (12.40%) and `forward_into` (9.30%) as the
targets and those are **cycles** shares. The gate adjudicates on
`instructions:u`. Those are not the same ranking, and for this symbol they are
not even close:

| symbol | PGO cycles | non-PGO `instructions:u` |
|---|---:|---:|
| `search::<NonPv>` | 23.04 | 13.11 |
| `HalfkaContext::current_full` | **13.18** | **2.52** |
| `SotaHead::forward_into` | 9.88 | 11.03 |
| `avx512::i8_deltas_into_generic` | 9.76 | 9.07 |
| `Worker::do_move` | 0.82 | 8.76 |
| `avx512::i8_deltas_into` | (inlined) | 7.92 |
| `ThreatContext::at_full` | 8.18 | 6.39 |
| `finny_refresh_into` | 6.25 | 4.18 |

**Study 49 already explained the 13.18 : 2.52 gap and its explanation holds
here**: `current_full` is a scalar caller sitting on top of stall-heavy
callees, its IBS share of true miss latency is 0.478%, and its
`load_not_complete` attribution reads 131% of a quantity with a 100% physical
ceiling. Study 49 then concluded *"this is the last attack that should be
attempted on instruction-count grounds"* after three failures (`lk49`, `out48`,
study 22's outline).

That conclusion was about **outlining and memo-ing**, which is what those three
arms were. It does not cover the symbol's own instruction stream, and the
shipped instruction stream is not the one study 49 read: it was measured
non-PGO, where the fold is a separate symbol. §3 is what the PGO layout shows.

## 3. What the PGO disassembly of `current_full` actually opens with

Sorting `perf annotate` by percent instead of by address, `current_full`'s
cycles split **56.5% vector / 43.5% scalar**, and the two hottest instructions
in the whole 7.9 KB function are scalar:

```
4.68 %   cmp    $0x2,%r15
4.49 %   movzbl 0x1c(%rsp),%r15d
```

Reading backwards to find what feeds them:

```
mov    (%r8,%rbp,1),%rax      ; meta[child].dirty, bytes 0..8
mov    0x6(%r8,%rbp,1),%rcx   ;                    bytes 6..14
mov    %rcx,0x16(%rsp)        ; ... copied to the stack
mov    %rax,0x10(%rsp)
movzbl 0x1c(%rsp),%r15d       ; ... and n_add read straight back off it
cmp    $0x2,%r15
ja     <panic_bounds_check>
```

Two facts, one line of source each:

1. `let dirty = self.meta[child].dirty;` **copies the 14-byte record to the
   stack** and the very next thing that happens is a read back out of that
   copy — a store-to-load forward at the head of the fold's dependency chain.
   The copy exists only because `apply_dirty_into` wants a `&DirtyPiece` while
   `self.stack` is borrowed mutably.
2. `cmp $0x2 / ja panic` is `DirtyPiece::adds()`'s slice bound. `n_add` is a
   `u8`, so **its range is a runtime fact to LLVM**, and every caller of
   `adds()`/`subs()` in the engine carries that edge.

Then the index loop itself, `for (i, &(piece, square)) in
dirty.adds().iter().enumerate()`, compiled to a rotated loop carrying an
`enumerate` counter, a 3-byte-stride cursor, two `setne`, a `cmovne`, two
`test %r11,%r11`, and a `cmp $0x1,%rdi / ja panic` for the two-element
destination arrays — **more control than index arithmetic**, for a loop whose
trip count is one or two.

## 4. The three arms

### 4.1 `dpref` — disjoint field borrows (−21.70 i/n)

```rust
let Self { weights, psqt_weights, stack, meta, .. } = self;
let (weights, psqt_weights) = (*weights, *psqt_weights);
for child in (ancestor + 1)..=ply {
    let dirty = &meta[child].dirty;          // was: self.meta[child].dirty
    let (before, after) = stack.split_at_mut(child);
    …
    apply_dirty_into(weights, psqt_weights, entry, parent, dirty, …);
    meta[child].computed[p] = true;
}
```

This is the idiomatic Rust answer to "I need two disjoint pieces of `self` at
once", and `finny_refresh_into` in the same file already makes exactly this
move for exactly this reason. It is not a `Copy`-avoidance micro-trick: the
copy was never *wanted*, it was the borrow checker's shape leaking into codegen.

Measured on the forward loop alone: **−20.81 i/n**. The same treatment on the
**backward fill loop** (the king-move branch, 0.0532 folds/node) adds
**−0.89 i/n** — small, in proportion to how rarely it runs, and free.

### 4.2 `dpmin` — a range fact, not a guard (−9.30 i/n, −9.79 br/n)

```rust
&self.adds[..(self.n_add as usize).min(self.adds.len())]
```

Two characters, and **the largest branch win of the study: −1.655% of every
branch the engine retires.** The `min` cannot change behaviour — the
constructors call `add`/`sub` at most twice per record, which the two new
`debug_assert`s now state — but it tells LLVM the slice is at most two long, so
the `cmp $0x2 / ja panic_bounds_check` edge disappears at every call site.

There are more of those than the accumulator: `sota_threats.rs` walks
`dp.adds()`/`dp.subs()` at four sites plus two slice-pattern fast paths
(`if let ([(mover, from)], [(placed, to)]) = (dp.subs(), dp.adds())`), and
`compute`'s last-king-move scan reads `subs()` once per ancestor.

This is the family study 56 worked with `& M` masks, with the correction study
56 §0.4 paid 15.2 i/n to learn: **a mask is unconditional cost, so spell the
bound as the fact you actually have.** The fact here is "no longer than the
array", and `min` says that for free.

### 4.3 `zip` — zip the destinations (−2.67 i/n)

```rust
for ((row, feat), &(piece, square)) in adds.iter_mut().zip(&mut add_feat).zip(dirty.adds()) {
```

`adds`/`add_feat` have length 2 at compile time, so the trip count becomes
`min(2, 2, n)` and the two `[_; 2]` stores lose their bounds checks by
construction.

It is small, and the reason is worth recording: **the `enumerate` counter was
never the expensive half** — `DirtyPiece`'s own unbounded length was, and that
is what `dpmin` fixes. The two are landed together because separately they each
leave the other's panic edge in place; combined they are slightly
super-additive (−2.67 and −9.30 alone, −12.83 together at the time both were
measured on the pre-`dpref` substrate).

## 5. The two losers, with their numbers

### 5.1 `pair` — study 37 arm 3 at `TILES = 8`: **+16.96 i/n**

This was the study's flagship candidate and the commission's named pattern.
Study 37 built the paired row walk (`split_at(min(na, ns))`, one paired loop
plus whichever tail survives) and measured **+0.147% instructions** — and then
wrote, in §9 item 4, that the verdict was width-specific: *"§3.1's explanation
is that LLVM's cloning outweighs the control, which is a statement about **this**
tile shape… I did not test another width and the banner does not claim I did."*
Study 44 then took `REGS_CAP` 20 → 5, so `TILES` 2 → 8, and nobody re-ran it.
That is precisely the shape that made study 57 arm A worth −41 i/n after study
37 had priced it at −8.

Rebuilt against post-57 main:

| | `instructions:u` | `branches:u` |
|---|---:|---:|
| study 37, `TILES = 2` | +0.147% | +0.243% |
| **study 58, `TILES = 8`** | **+0.290% (+16.96 i/n)** | **+0.566% (+3.35 br/n)** |

**The width was not the reason, and it is twice as bad at eight tiles.** Study
37's stated mechanism — LLVM clones the tile body across the empty-list cases
*and unrolls both tiles*, and the cloning is worth more than the control — does
not even apply at `TILES = 8`, where the tile walk is one body in a real loop.
What costs is turning two row loops into three: each gets its own entry test and
its own unroll prologue, per tile, and the ×4 applies to that too.

### 5.2 `quad` — hoist the unroll prologue: **+63.35 i/n**, the study's worst

The generic fold's row loops are runtime-length, so LLVM 4×-unrolls them with a
1-row prologue. It hoists the prologue *arithmetic* to the function entry — but
it re-runs the *decision* inside the tile walk, once per list per tile:

```
cmpl $0x18,-0x8(%rsp)   ; is a prologue needed?
jne  ...                ;   yes: load its count, enter it
mov  %rcx,%r9           ;   no:  point at the list
jmp  ...
cmpq $0x18,-0x10(%rsp)  ; is the 4x body reachable?
jb   ...
```

~5 instructions × 2 lists × 8 tiles ≈ **80 per call ≈ 68 i/n** at 0.854
generic-fold calls/node, read off the shipped disassembly. Splitting the lists
once, outside the walk, into compile-time 4-row groups plus a ≤3 tail
(`as_chunks::<4>`) should have collected it.

It measured **+1.083% instructions (+63.35 i/n), +16.22 br/n, node-identical**.
Two loops became four; each new loop wants its own entry test and LLVM unrolls
the outer `chunks` loop anyway, adding a second prologue on top of the one I was
removing.

**The rule these two arms buy, and it is now paid for three times in this
kernel (37 §3, 58 §5.1, 58 §5.2): a loop-control instruction count is not a
prize. It is an estimate of what LLVM will charge you for taking the loop away
from it.**

### 5.3 `dpwr` — masking the write cursor: exact null

`DirtyPiece::add`/`sub` write `self.adds[self.n_add as usize]`, which looks like
the same defect as §4.2 on the once-per-made-move path inside `do_move`
(7.97% of instructions). Masked with `& 1` — bit-identical, since each is called
at most twice per record — it measured **+690 instructions in 2.93 × 10¹⁰**, i.e.
+0.00014 /node. `n_add` is only ever incremented from zero inside one `from_move`
call, so LLVM already proves the bound and there is no check to remove. Reverted;
the `debug_assert`s stay as documentation of the invariant §4.2 relies on.

## 6. The panel

`tools/perf58/panel58.sh 24 base base6 land` — blocked Latin square, `flock`ed
write-once output, only parsed numbers in the file. `base6` is a **byte-identical
null** (`.text` md5 `8876f37d`, same as `base`, banner stripped) riding in the
rotation. Mean position 1.00 of 1.00 for all three labels.

```
REFERENCE per node: L1-dcache-loads 2926.30  branches 591.70  cycles 3575.37
                    instructions 5850.12  ls_dispatch.ld_dispatch 2194.73

label   event                    median %   se     neg/n    p
base6   instructions:u             -0.000  0.000   14/24   0.5413
base6   branches:u                  0.000  0.000   11/24   0.8388
base6   L1-dcache-loads:u          -0.086  0.038   21/24   0.0003   ← FAILS its null
base6   ls_dispatch.ld_dispatch:u  -0.088  0.079   16/24   0.1516
base6   cycles:u                   +0.306  0.460   10/24   0.5413

land    instructions:u             -0.520  0.000   24/24   0.0000
land    branches:u                 -1.670  0.000   24/24   0.0000
land    L1-dcache-loads:u          -0.455  0.057   24/24   0.0000
land    ls_dispatch.ld_dispatch:u  -0.650  0.119   24/24   0.0000
land    cycles:u                   -0.269  0.489   14/24   0.5413
```

Three things to read off it honestly:

* `instructions:u` and `branches:u` are **exact** — 0.000 se on both the null
  and the arm, every round the same sign. Study 48 §6 and study 57 §7 said so;
  this is the third confirmation on this box.
* **`L1-dcache-loads:u` FAILED its own null at n = 24** (−0.086%, 21/24,
  p = 0.0003 on byte-identical binaries). So the load win is quoted
  arm-minus-null: **≈ −0.37% ≈ −10.8 loads/node**, not −0.455%. The
  `ls_dispatch` sibling behaved (16/24) and reads −0.562% net ≈ −12.3
  dispatches/node — the two counters disagree by 50%, which is study 57 §7's
  standing warning that they are not the same event.
* **`cycles:u` is not evidence here and is not quoted as any.** The arm reads
  −0.269% at 14/24 against a null of +0.306% at 10/24. Study 57 §9.3's
  calibration says a ~1% cycles effect needs ~40 paired rounds; this effect is
  ~0.5% and 24 rounds cannot see it. §7.1's PGO panel, at 39 rounds and a
  ~1.3% effect, does resolve it — which is the same rule, obeyed.

Reproduce: `bash tools/perf58/build58.sh <label> [features]`,
`bash tools/perf58/panel58.sh <rounds> <label>...`,
`python3 tools/perf58/med58.py <panel> <ref> <labels>...`. Raw panels are in
`docs/research/perf-artifacts-58/`. Baseline `.text` md5 `8876f37d`, reproduced
by five independently-built inert controls (`base2`…`base6`) as each `cfg!`
split landed — every one of them byte-identical, which is what makes each arm's
number a property of the arm and not of the edit.

## 7. PGO amplified the win 1.99×, and why that is not a contradiction

The commission's standing figure is that **PGO absorbs 38%** of a hand-optimised
instruction win (studies 52–57: −263.1 i/n non-PGO, −162.4 PGO), largest for
inlining and call-site arms, smallest for things PGO cannot express.

This study measures the opposite sign:

| | base | landed | delta |
|---|---:|---:|---:|
| non-PGO (the gate build) | 5850.12 i/n | 5819.69 | **−30.43** |
| PGO (the shipped build) | 5789.63 i/n | 5729.02 | **−60.62** |

PGO itself is worth −60.5 i/n on the base tree and −90.7 i/n on the landed one.
The arms are worth **1.99× more on the binary we ship than on the one the gate
adjudicates.**

The mechanism is the commission's own point, run forwards. `dpref` deletes a
stack round-trip in `compute`; in the non-PGO build `compute`'s callees are
separate symbols and the record is passed by reference across a call boundary,
so the round-trip is small. **PGO inlines the whole fold into `current_full`,
which is what makes the copy sit on the fold's critical path and be executed at
full frequency** — the same phenomenon the commission documented for
`i8_deltas_into_inner` (10.36% non-PGO, absent under PGO, its cost reappearing
inside `current_full` at 3.72% → 12.40%).

So the two numbers are complements, not contradictions:

> **An arm found on the wrong binary gets absorbed. An arm found on the right
> one gets amplified.** The 38% figure is a property of where studies 52–57
> looked, not of PGO.

`branches:u` behaves the other way (−9.88 non-PGO, −8.51 PGO): PGO's block
placement had already recovered part of the panic-edge cost. That is the 38%
pattern, on the one counter where the arms overlap with what PGO can express.

### 7.1 The PGO panel, and a loads regression I did not expect

`tools/perf58/panel58.sh 40 base-pgo nullpgo final-pgo`, 39 complete rounds,
`nullpgo` a literal byte-copy of `base-pgo`. Mean position 1.00 for all three.

```
REFERENCE per node: L1-dcache-loads 2947.93  branches 561.01  cycles 3582.17
                    instructions 5789.61  ls_dispatch.ld_dispatch 2248.26

label      event                     median %    se     neg/n     p
nullpgo    instructions:u              0.000   0.000    17/39   0.5224
nullpgo    branches:u                 -0.000   0.000    21/39   0.7493
nullpgo    L1-dcache-loads:u          +0.009   0.029    19/39   1.0000
nullpgo    ls_dispatch.ld_dispatch:u  +0.064   0.075    17/39   0.5224
nullpgo    cycles:u                   +0.395   0.574    17/39   0.5224

final-pgo  instructions:u             -1.047   0.000    39/39   0.0000
final-pgo  branches:u                 -1.516   0.000    39/39   0.0000
final-pgo  cycles:u                   -1.272   0.304    30/39   0.0011
final-pgo  L1-dcache-loads:u          +0.273   0.027     3/39   0.0000
final-pgo  ls_dispatch.ld_dispatch:u  +0.278   0.058     5/39   0.0000
```

The load counter **passes its null at n = 39** (+0.009%, 19/39, p = 1.0) where
it failed at n = 24 non-PGO, so this reading is usable — and it says the arms
**cost ~8 loads/node on the shipped binary**, sign-flipped from the −10.8 the
non-PGO panel measured.

That is `dpref`, and the mechanism is the point of the arm inverted. The old
code did **two loads from `meta` + two stores to the stack**, and then every
subsequent read of the record — `n_add`, `n_sub`, the six `(Piece, Square)`
bytes — hit the stack copy, which is one hot line. The new code reads each
field from the `meta` array where it lives, so the *consolidating* effect of
the copy is gone: fewer instructions and fewer stores, more individual loads
of a colder line.

It is the trade study 47 §5.2's op table predicts (a folded memory operand
costs an issue slot, and a copy that is *read many times* can be worth its
stores), and **cycles say the trade is favourable by 1.27%.** Recorded because
the next person to look at `compute` will see the loads go up and should know
it was measured, not missed.

## 8. `forward_into`, the second commissioned target: a decomposition, not an arm

`forward_into` is **11.03% of `instructions:u`** — the largest NNUE symbol on
the gate counter, above its 9.88% of cycles. Study 25 §1 decomposed it once, by
building with `#[inline(never)]` on the kernels, and that build predates the i8
HalfKA transformer, the const-generic threat shift, and studies 26/46/48/50's
rewrites of the head. It has never been re-run.

`--features split58` is the re-runnable form of that experiment (diagnostic
only, `cfg_attr`-gated, byte-identical when off — verified: `.text` md5
`8876f37d` unchanged). Per-symbol instruction shares are the only attribution
this box gives that survives its own null; study 36 §2.3 killed per-mnemonic
attribution of `instructions:u` with a 20× skew among four instructions that
retire identically.

Measured on the LANDED tree (`--features sota,arch-d48x,split58`, same frame).
Outlining costs **+20.93 i/n (+0.360%)** of call overhead, so the shares below
are diluted by that much and the per-node column is against the split build's
own 5840.62 i/n:

| kernel | % of engine `instructions:u` | i/node | % of `forward_into` |
|---|---:|---:|---:|
| `avx512::activate` | 3.53 | 206.2 | 35 |
| `avx512::l1_dot` | 2.99 | 174.6 | 30 |
| `avx512::l2_tail` | 1.26 | 73.6 | 13 |
| `avx512::find_nnz_vbmi2` | 1.21 | 70.7 | 12 |
| `forward_into` residual (glue + `l1_dequant` + `l3_dot` + `skip_dot_i8`) | 0.96 | 56.1 | 10 |
| **total** | **9.95** | **581.1** | |

For scale, the same build shows the fold family the commission pointed at:
`i8_deltas_into_generic` 8.83% (threats), `i8_deltas_into` 8.08% (the shaped
HalfKA path), `i8_deltas_dual` 2.74% (Finny) — **19.65% together, twice
`forward_into`.**



Every one of those sits on a documented floor:

* **`activate`** — study 48 landed the register-scale hoist (−17.65 i/n),
  study 50 the u8 `ft_scale`. Study 47 §5.2 priced six further alternatives and
  all lose; the emitted body is 13 ops per (32-lane group, perspective) against
  a hand count of 14 with the `pst` load folded into the `vpmullw`.
* **`l1_dot`** — 4 instructions per active group (`movzwl` index,
  `vpbroadcastd` ft, `shl $6`, `vpdpbusd` weights). The `shl` is not removable:
  the ft stride is 4 and the weight stride is 64, a ratio of 16, and x86 scale
  goes to 8 (study 26 §6.6). §9 prices the one variant study 26 did *not*
  consider and declines it.
* **`l2_tail`** — study 46 §4 priced the 2→4 accumulator rewrite at −0.185% and
  rejected it on three grounds including that it is **not eval-bit-identical**.
  Its own re-price trigger (`L2` growth) has not fired.

**There is no cheap instruction win left in `forward_into`, and this is the
first decomposition that says so about the shipped architecture rather than
about D47V.**

## 9. Priced and declined

**A 3-instruction `l1_dot` inner group.** Study 26 §6.6 killed "pre-scaled nnz
indices" because two *index* arrays trade the `shl` for a second `movzx`. It did
not consider a **value** array: have `find_nnz` emit the ft dword alongside a
`×64`-pre-scaled index (319 × 64 = 20416, still `u16`), and the group becomes
`vpbroadcastd (%rvals,%rj,4)` + `movzwl (%ridx,%rj,2)` + `vpdpbusd
(%rax,%r11,1)` — the ft index disappears because the value array is walked
sequentially.

Arithmetic, at ~105 active groups/eval (from `l1_dot` = 3.26% of instructions
÷ 0.43117 evals/node): saves **105 instructions/eval**, costs a `vpcompressd`
plus a store per 16-group block in `find_nnz` — **+80/eval** — for a net
**≈ +25/eval ≈ 11 i/n**, plus 420 bytes/eval of extra store-then-load traffic
into a symbol study 50 already measured at a 39.7% L1 miss rate. **Declined on
the arithmetic; recorded because the reason study 26 gave does not cover it.**

## 10. What the next study should know

1. **`current_full`'s remaining cycles are not its own.** After this study its
   instruction share is 2.12%. Study 49's IBS work (0.478% of true miss
   latency, ratio 131% against a 100% ceiling, skidprobe collapsing the
   attribution 4.5×) is the explanation and it survives on the PGO layout too.
   Do not aim a fourth arm at "`current_full`'s stall".
2. **The generic fold's loop control is closed.** Three arms have now attacked
   it (37 §3, 58 §5.1, 58 §5.2) and the sum of their measured cost is
   +80 i/n. The residual scalar control is ~130 i/n by static count, and every
   attempt to collect it has made LLVM emit more, not less.
3. **The range-fact family is not exhausted, but it is site-specific.**
   `dpmin` was −9.79 br/n for two characters; `dpwr`, twenty lines away and
   apparently the same defect, was an exact null. The difference is whether
   LLVM can already prove the bound. **Check the disassembly for the panic
   edge before writing the fix** — there is no shape rule here.
4. **Profile the PGO binary, and gate on non-PGO instructions.** They rank
   differently enough that `current_full` is #2 on one and #10 on the other,
   and `do_move` is 0.82% on one and 8.76% on the other. This study's largest
   arm is invisible on the gate profile and its cost is 2× larger on the
   shipped one.
5. **A cycles number needs ~40 rounds and it is worth paying for.** The same
   landing set read −0.269% at 24 rounds (14/24, meaningless) and −1.272% at 39
   (30/39, p = 0.0011). Study 57 §9.3 called this and it is now confirmed on a
   second effect size. Budget the rounds or do not quote the counter.
6. **Still open from the catalogues, best-priced first** (neither touched here):
   study 43 §9 item 2's subtractive Finny-line arm (every Finny price in the
   repo carries a 3.7× bracket until it exists); study 57 §6's FNV-1a-64
   payload checksum (~542 M cycles ≈ 45% of all startup, needs a format
   ruling); study 48 §9 item 3's AVX2 `activate` hoist (the `x86-64-v3` build
   is a real target and the transform has to be re-derived, not copied).
