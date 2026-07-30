# Perf study 27 — the search layer, aimed at branch misses

**Date:** 2026-07-29
**Scope (owned files):** `src/search/mod.rs`, `src/movepick.rs`, `src/board.rs`,
`src/movegen.rs`, `src/moves.rs`. Nothing outside them was edited.
**Frame, quoted with every number below:**

```
bench 16 1 100000 tools/nnue/calib/wl50.fens nodes engines/canonical/net.cenn
```

50 FENs × 100k nodes, one thread, 16 MB TT, canonical D48X net, features
`sota,arch-d48x`, `RUSTFLAGS=-C target-cpu=native`, thin LTO,
**CODEGEN_UNITS=1**, pinned to core 7. A fixed **5,001,562-node** tree. Every
arm in this document — landed and rejected — reproduces that node count exactly
and the wl50 eval hash `5ca978e012da1bb0`.

**Instrument.** The gate is `instructions:u` **and** `branches:u`, which must
agree in sign. Both are exact at fixed nodes: the byte-identical-binary null in
§6 reads `+0.000%` and `-0.000%` with **zero spread over twelve rounds**.
`branch-misses:u` and `L1-dcache-load-misses:u` are reported on every arm, each
against its own null, because neither is exact. Cycles are not measurable on
this box and are used exactly once (§5.3), for an arm whose counters disagreed,
with a byte-identical copy carried as a third arm to be the error bar.

---

## 0. TL;DR

Four arms landed, node-identical and eval-hash-identical, measured together
against the pre-study base with the null carried in the same twelve rounds:

| counter | base | +4 arms | **paired median delta** | **its own null** |
|---|---:|---:|---:|---:|
| instructions / node | 5819.828 | 5756.570 | **−1.087%** | +0.000%, zero spread |
| branches / node | 607.495 | 601.591 | **−0.972%** | −0.000%, zero spread |
| branch misses / node | — | — | +0.747% | −0.280% [−0.770, +0.892] |
| L1d load misses / node | — | — | −0.495% | −0.241% [−0.683, +0.282] |

The bottom two rows are quoted **only** as paired medians and their per-node
levels are deliberately left blank: single-run values for those counters swing
by more than the effect (§3), and reading levels off them is how this study
would have reported a branch-miss win it did not get.

| arm | file | alone |
|---|---|---:|
| 5 — `#[inline(always)]` on `MovePicker::next_move` | `movepick.rs` | **−37.61 i/n, −6.20 br/n** |
| 2 — `ExtMove` has no padding hole, `value` first | `movepick.rs` | **−21.11 i/n, −1.01 bmiss/n** |
| 3 — stage transitions are literals, not `bump()` | `movepick.rs` | **−2.97 i/n, −1.01 br/n** |
| 4 — the state stack is a slab + cursor, not a `Vec` | `board.rs` | **−1.50 i/n, −2.50 L1d miss/n** |

**And the study's most useful result is a negative one, in the fourth row of
the first table.** The commission aimed me at branch misses. I removed a
mispredicting branch from `do_move`, an indirect dispatch from `next_move`, and
a quarter of the memory traffic in the insertion sort — and the engine's
branch-miss rate did not improve. It is inside its own null band, on the
high side. §4 is the evidence, and it retracts two attributions the commission
handed me as established.

Three arms were built, run and rejected on measurement (§4), including both
attempts at the commission's item 3.

---

## 1. What the commission asked for, and what the profile actually says

Four items came with measurements already done and "do not re-derive". I did
re-derive the two that were load-bearing, because rule 8 says to, and **both
moved**.

A `perf record -e branch-misses:u -c 10007` over the whole study frame (10K
samples, 107 M events) on the shipped base binary:

| symbol | share of engine branch misses | commission's figure |
|---|---:|---|
| `MovePicker::init_quiets` | **14.43%** | (the sort inside it: 10.19%) |
| `Worker::search::<NonPv>` | 14.36% | 13.07% |
| `ThreatContext::at_full` | 10.16% | 10.94% |
| `HalfkaContext::finny_refresh_into` | 5.58% | — |
| **`Worker::do_move`** | **5.49%** | **7.22%** |
| `SotaHead::forward_into` | 4.59% | — |
| `MovePicker::next_move` | 4.44% | (dispatch: 4.62%) |
| `Board::see_ge` | 2.20% | — |
| `Board::attackers_to` | 2.12% | — |
| `MovePicker::init_captures` | 2.08% | — |

Two corrections fall straight out of this table and are developed in §4:

* **`do_move` is 5.49%, not 7.22%** — and §4.1 shows the branch the commission
  named inside it carries none of it.
* **The `next_move` stage dispatch is ~0.9% of engine misses, not 4.62%.**
  `next_move` *in total* is 4.44%, and `perf annotate` puts the two jump-table
  dispatches and their landing prologues at roughly a fifth of that.

### 1.1 Inside the sort: it is the WALK, not the filter

The commission's item 1 is the insertion sort, and study 23's §8 proposed
attacking the `tmp.value < limit` filter (its remainder item 2a). Annotating
`init_quiets` at instruction level says that is the wrong half. The sort is
inlined, so its instructions live at `init_quiets+0x1510..+0x15a7`; the
per-instruction branch-miss shares there, *within* `init_quiets`:

| offset | instruction | share of `init_quiets` misses | belongs to |
|---|---|---:|---|
| `+0x1592` | `ja` (bounds edge, never taken) | 14.66% | walk exit — skid |
| `+0x157e` | `mov %bp,(%rdx,%rdi,8)` | 7.91% | walk body — skid |
| `+0x1574` | `cmp %r9d,%ebx` | 7.59% | walk body — skid |
| `+0x1533` | `je` (loop exit) | 4.41% | filter — skid |
| `+0x1586` | `dec %rdi` | 3.83% | walk body |
| `+0x1546` | `lea -0x1(%rsi),%r11` | 3.70% | filter/hole-punch |
| `+0x1577` | `jge` (walk exit) | 2.01% | walk exit |

`branch-misses:u` without `:pp` skids onto the instruction after the branch, so
these must be read in blocks rather than per row. Read that way the **walk**
region carries ~41% of `init_quiets`' misses and the **filter** region ~11% —
about four to one, the opposite of the ranking the plan assumed. Study 23's
own arm 8b was already evidence for this (a `partition_point` insertion-point
search left misses *up*), and it read that result as refuting the walk
hypothesis; the annotate says it refutes the filter hypothesis instead.

The sort in total is ~52% of `init_quiets` = **~7.5% of all engine branch
misses**, against the 10.19% the commission carried forward.

**This re-ranking is what produced arm 2.** A branchless partition attacks the
filter — 11% — and costs instructions to do it (§4.3). Halving the *memory
traffic of the walk* attacks 41% and costs nothing.

---

## 2. Landed arms

### 2.1 Arm 2 — `ExtMove` has no padding hole, and `value` goes first

`movepick.rs`. design/02 §3.8 fixes `ExtMove` at 8 bytes and justifies it in as
many words: *"the picker's insertion sort swaps `ExtMove`s in its
`[ExtMove; MAX_MOVES]` buffer millions of times per second; 8-byte elements
move as single u64 register loads/stores."*

They did not. The layout was `mv: Move` (2 bytes), an **implicit two-byte
hole**, then `value: i32` — and a hole makes a struct copy illegal to widen,
because the padding bytes are `undef` and LLVM may neither read nor write them.
So the hottest three lines in movepick compiled to a 2-byte move plus a 4-byte
move. One shift step of the insertion sort, before and after:

```
  mov    -0x4(%rdx,%rdi,8),%ebx          cmp    %r9d,-0x8(%rdx,%rdi,8)
  cmp    %r9d,%ebx                       jge    out
  jge    out                             mov    -0x8(%rdx,%rdi,8),%rbx
  movzwl -0x8(%rdx,%rdi,8),%ebp          mov    %rbx,(%rdx,%rdi,8)
  mov    %bp,(%rdx,%rdi,8)               dec    %rdi
  mov    %ebx,0x4(%rdx,%rdi,8)           jne    loop
  dec    %rdi
  jne    loop                            6 instructions, 3 memory ops
  8 instructions, 4 memory ops
```

Two changes, each doing a distinct job:

* **Naming the pad as a field** (`pad: u16`, always zero) closes the hole, so a
  copy is one 8-byte move.
* **Putting `value` at offset 0** lets the compare that decides whether to
  shift fold into the element's own address — `cmp %r9d,-0x8(%rdx,%rdi,8)`,
  with no separate load at all.

The same halving applies to the hole-punch, to `head[q] = tmp`, to every
`slot`/`set_slot` in both scoring loops, and to `select`. `ExtMove::new` is now
the only constructor so `pad` cannot be forgotten.

| counter | before | after | delta |
|---|---:|---:|---:|
| instructions / node | 5818.333 | 5797.220 | **−21.11** |
| branches / node | 607.494 | 607.494 | −0.00 (flat to six figures) |
| branch misses / node | 21.53 | 20.52 | −1.01 |
| L1d load misses / node | 617.45 | 609.29 | −8.16 |

design/02 §3.8 is amended in the same commit, with the mechanism and a warning
that a future field must keep the type hole-free or this reverts silently.

*(The −1.01 branch misses here is a single-run figure and §6's twelve-round
panel does not confirm it survives into the combined tree. The instruction and
branch numbers are exact; that one is not.)*

### 2.2 Arm 5 — `#[inline(always)]` on `MovePicker::next_move`

`movepick.rs`, one attribute. Study 23 shipped plain `#[inline]` (−12.0 i/n),
measured `#[inline(always)]` at −40.9, called it "more than three times better
on the gate", and declined it: it looked like it cost +7.5 L1d misses/node and
that box could not adjudicate the trade. It is the top item on that study's
remainder, marked "needs ten quiet minutes rather than a new idea."

Three measurements, none of them cycles:

1. **The gate.** −37.61 instructions/node (−0.65%) and −6.20 branches/node
   (−1.02%), both exact, both against a null of exactly zero. `next_move` is
   called 2.335 times per node; the win is the prologue, the six-register
   spill and the epilogue at partially-inlined sites, not the call.
2. **The L1d objection was measured without its denominator.** Total
   `L1-dcache-loads` **fall** by 4.1/node while misses rise ~2.4/node — a miss
   rate of 19.42% → 19.55%. That is *less memory work at a marginally worse hit
   rate*, not more work, and it is a different thing to price. (The +7.5 figure
   also does not reproduce: my two-run pairs give +2.4, and §6's panel puts the
   combined tree's L1d miss rate slightly *below* its own null.) The stack
   frame of `search::<NonPv>` grows 0xaf8 → 0xb18, 32 bytes, exactly as study 23
   found — so the frame is not the mechanism for what rise there is; the
   plausible one is that `search` recurses ~20 deep, so 32 bytes/frame is ~640
   bytes more live stack against a 32 KB L1d that the search stack already
   overflows.
3. **A cycles panel, reported for what it is.** Fifteen rounds, three-way
   round-robin, byte-identical copy of the base carried as the third arm, order
   rotated within each round:

   ```
   NULL (byte-identical copy)   median= +0.401%  faster_in=6/15  min=-4.571%  max=+8.170%
   ARM  (a3inl)                 median= -0.963%  faster_in=9/15  min=-6.686%  max=+6.113%
   ```

   **This is directional only and no verdict is claimed from it.** The null
   spans 12.7 points on a change that does not exist; a 1.36-point separation
   of medians inside that is not a measurement. What it does do is answer the
   question it was run for — whether cycles point the *other* way from the gate
   — and they do not: the arm's median is below the null's and it wins 9/15
   against 6/15.

Landed on the gate, with the L1d cost stated and a falsifiable prediction in
§7.

### 2.3 Arm 3 — the picker's stage transitions are literals, not `bump()`

`movepick.rs`. `next_move` is a `loop { match self.stage { … } }` and every
fall-through advanced with `self.stage = self.stage.bump()`. That is
behaviourally identical to naming the next stage and it reads better — but it
hides the destination behind arithmetic on a byte LLVM loaded from memory, so
each `continue` went back through the 16-entry jump table to reach a block the
source already names:

```
movslq 0x0(%r13,%rcx,4),%rcx
add    %r13,%rcx
jmp    *%rcx
```

Every fall-through now assigns a literal, and the shared
`CaptureInit | ProbcutInit | QcaptureInit` arm is split into three copies of one
line — same body, different successors, which is exactly the information the
merged pattern was discarding. The `MainTt | …` arm keeps `bump()`: it
*returns*, so there is no back-edge to make direct.

**−2.97 instructions/node, −1.01 branches/node.** The two numbers check each
other: a removed dispatch is exactly the three instructions above, and
1.01 × 3 = 3.03, so this deletes **one indirect jump per node** — not the
several the static count suggests (9 → 8 static sites), because LLVM had
already folded some paths and others are still reached from more than one
stage.

### 2.4 Arm 4 — the state stack is a slab plus a cursor

`board.rs`. Study 24 diagnosed this and its own fix failed; it named the
spelling to try instead, and this is that spelling.

`Vec::push` takes its argument **by value**, and `RawVec::grow` is a call that
**returns**. So `history.push(self.state)` compiled to a full stack round-trip
of the 168-byte `StateInfo` — confirmed in the base binary before writing a
line:

```
vmovups 0x18(%rdi),%zmm0   |  vmovups %zmm2,0xc8(%rsp)   |  cmp (%rdi),%r14
vmovups 0x80(%rdi),%zmm2   |  vmovups %zmm1,0xa0(%rsp)   |  je  <grow>
vmovups 0x58(%rdi),%zmm1   |  vmovups %zmm0,0x60(%rsp)   |  vmovups 0x60(%rsp),%zmm0
                                                         |  vmovups 0xa0(%rsp),%zmm1
                                                         |  vmovups 0xc8(%rsp),%zmm2
                                                         |  vmovups %zmm2,0x68(%rax,%rdx,1)
                                                         |  vmovups %zmm1,0x40(%rax,%rdx,1)
                                                         |  vmovups %zmm0,(%rax,%rdx,1)
```

Twelve 64-byte memory operations where six would do, on the one function that
runs exactly once per node. The reloads are not removable while the call is in
the way: LLVM must assume `grow` clobbers memory, so anything live across it
has to come back from somewhere.

`StateStack` keeps its `Vec` at **full length** with every slot initialised and
tracks the live depth in a separate cursor. A push is then an indexed store
straight from one field of `*self` into another — no temporary to spill, and no
call to spill it across. After: **six** memory operations, and the frame drops
from `0x108` to `0x68`.

`Deref` hands out `&slots[..depth]` so every reader is unchanged, and
`Clone`/`PartialEq` go through that slice so the dead tail below the cursor
stays unobservable. There is a new test for exactly that, which also asserts
the dead record really is still in the slab so it cannot pass vacuously.

**Three codegen details cost a build each, and none is guessable** — they are
commented at the site, and they are the whole difference between this arm
losing and winning:

| spelling | instructions/node | branches/node |
|---|---:|---:|
| `push`/`pop` on the `Vec` (base) | 5819.828 | 607.495 |
| slab, `if depth == len { grow() }` then store, `Deref` slices `..depth`, `checked_sub` in pop | 5822.229 | 609.983 |
| slab, `>=` guard, `Deref` slices `..depth.min(len)`, wrapping pop | 5820.129 | 608.512 |
| **slab, store inside the `<` arm, inherent `len()`, wrapping pop** | **5818.333** | **607.494** |

* `if depth >= len { grow() }` *followed by* the store does not discharge the
  index bounds check. The store block has two predecessors and one of them just
  called something that may have moved the slab. Putting the store in the `<`
  arm and giving `grow_and_store` its own copy fixes it: +1 branch/node and
  ~+2 instructions/node recovered.
* `Deref` must slice with `depth.min(slots.len())`, one `cmov`, not a bare
  `..depth` — otherwise every read carries a bounds test and a panic block, and
  the repetition scan reads once per node.
* `StateStack::len` must be an **inherent** method returning the cursor, or
  `history.len()` builds the `Deref` slice just to ask it something the cursor
  already knows.

Final: **−1.50 instructions/node, branches flat, −2.50 L1d load misses/node.**
Instructions understate it — the six removed 64-byte operations are 192 bytes
of store traffic per node, which `instructions:u` counts as six.

---

## 3. The one that mattered most is the one that did not fire

The commission's headline for me was branch misses. §6's panel is the verdict:

```
NULL   branch-misses = -0.280%  [-0.770, +0.892]
ARM    branch-misses = +0.747%  [-0.274, +2.114]
```

**Branch misses did not improve.** The arm's median sits above the null's, and
its interval overlaps the null's. Four arms removed roughly one branch in a
hundred from the engine's execution, and the mispredict rate is unchanged or
marginally worse.

That is not a null result about the arms — the gate counters moved ~1% each
against an exactly-zero null. It is a result about **which branches were
removed**: panic edges, a jump-table dispatch on a pattern that turns out to be
learnable, and a capture test that turns out to be predicted. The engine's
mispredicts live somewhere else, and §4.1 and §1.1 say two concrete places they
do *not* live.

The single-run probes during development repeatedly showed −1.0 branch
misses/node for arm 2 and I would have reported that as a win without the
twelve-round panel. The panel is the reason this section says what it says.

---

## 4. Rejects, with the measurement that killed each

Every reject reproduced 5,001,562 nodes and hash `5ca978e012da1bb0`. Each is a
built, run, measured arm.

### 4.1 Item 3 — the branchless capture append in `DirtyPiece::from_move`

The commission's item 3, handed over as a one-liner with the plumbing already
in place: *"`do_move` is 7.22% of engine branch misses despite being
straight-line. The culprit is `if let Some(victim) = board.piece_on(to)`.
A branchless append fixes it."*

**Two arms, both rejected, and together they refute the attribution.**

| arm | spelling | result |
|---|---|---:|
| 1a | `subs[1] = (piece_on(to).unwrap_or(mover), to); n_sub = 1 + is_some()` | **+9.50 i/n, and the branch is still in the disassembly** |
| 1b | select the SQUARE: `vic_sq = if occupied { to } else { from }`, one unconditional load off it | **+16.73 i/n, +1.00 br/n, branch misses FLAT** |

**1a is a codegen lesson.** `Option<Piece>` is niche-encoded in the *colour*
byte, so when a square is empty the piece-type byte is padding and reading it
is UB. LLVM knows that, refuses to speculate the second mailbox load, and keeps
the branch:

```
mov    $0x1,%cl                       ; n_sub = 1
movzbl 0xc0(%rsi,%r14,2),%ebx         ; mailbox[to] tag
cmp    $0x2,%bl
je     <skip>                         ; <-- still here
movzbl 0xc1(%rsi,%r14,2),%edx         ; the load it would not speculate
```

Rewriting the source to be branchless does not make the object code
branchless. Reading the disassembly caught this in one build.

**1b is the real result.** Selecting the *square* rather than the piece gives
an address that is unconditionally occupied (`from` always holds the mover), so
the load needs no branch, and the disassembly confirms it: `cmove %edx,%r11d`
then an unconditional two-byte load, with `sbb` computing `n_sub` and only the
`expect`'s never-taken panic edge left.

The branch is gone. **Branch misses did not move** (21.79 vs 21.64/node,
inside run-to-run spread), and instructions rose 16.7.

So the branch was **well predicted all along**. The plausible reason: the
picker emits every capture before every quiet, so within a node's move loop
"is this a capture" is a long run of ones followed by a long run of zeros —
about the friendliest possible shape for a TAGE-style predictor. Whatever
`do_move`'s 5.49% is, it is not this test. Study 23's estimate of ~0.6% of
engine cycles for this item should be treated as retracted.

### 4.2 Two intermediate spellings of arm 4

Recorded in the table in §2.4 rather than here, because they are the same arm:
the first two slab spellings measured **+2.40** and **+0.30** instructions/node
against the `Vec`, i.e. they *lost*, and the mechanism they were meant to
capture (six fewer 64-byte memory operations) was already fully present in
both. The whole delta between +2.40 and −1.50 is three redundant bounds tests.
**A correct mechanism can be entirely cancelled by the checks the spelling
leaves behind**, and only the disassembly says which.

### 4.3 Not built: the branchless partition of `partial_insertion_sort`

Priced before it was believed, which is why it cost no build.

The plan was study 23's remainder item 2a — hoist the `tmp.value < limit`
filter into a branchless Lomuto partition, then insertion-sort the selected
prefix. **The node-identity obstacle is soluble and this is worth recording,
because it is the constraint the commission called "the whole problem":** the
two-pass form is *provably* identical to the interleaved one. The insertion
step only ever writes `head[q..=se]`, and the next swap's target is `se+1`,
which is outside that range — so no insertion can change what a later swap
sees, and the sequence of values fed to the insertion sort is the same. The
tail permutation is therefore preserved exactly. **The tail-permutation
constraint does not rule out a two-pass split.**

What kills it is the price, on the counter that gates:

* Per node the sort examines only **3.93** elements (22.6 per call × 0.174
  calls/node).
* A branchless conditional swap is ~10 instructions per element against ~5 for
  the branchy skip — two loads, two `cmov`s and two unconditional stores where
  the current code does one load and a predicted-taken jump.
* ≈ +25 instructions/node (+0.43%) to remove ≈ 3.9 branches/node (−0.64%).

Instructions up, branches down: the two gate counters **disagree in sign**, so
the instrument returns no verdict, and the thing it buys is the filter — which
§1.1 measures at ~11% of the sort's misses against the walk's ~41%. It is the
wrong 11% at a price the gate cannot clear. Arm 2 attacked the other 41%
instead and cost nothing.

---

## 5. Method notes worth reusing

### 5.1 Measure the denominator before believing a rate

`L1-dcache-load-misses` rose on arm 5 and that is what parked the change for a
day. Misses alone cannot distinguish *more memory work*, *the same work with a
worse hit rate*, and *less work with a much worse hit rate*. Adding
`L1-dcache-loads` took two runs and flipped the reading: loads **fell** 4.1/node.
`tools/perf27/loads27.sh` exists so the next study does not repeat this.

### 5.2 Read the disassembly, then read it again after the fix

Three separate times here the source-level reasoning was right and the object
code did not follow: arm 1a's surviving `je`, arm 4's duplicated bounds test,
arm 4's `Deref` bounds test. All three were invisible to reasoning and obvious
in `objdump`. The arm-4 sequence is the sharpest case — the intended mechanism
was fully present in a version that measured **+2.40 i/n**.

### 5.3 A cycles panel can be worth running even when it cannot decide

§2.2's panel does not resolve 1%; its null spans 12.7 points. It was still
worth 15 minutes, because the question was not "how much faster" but "do cycles
point the *opposite* way from the gate". A null-carrying panel can answer that
much even when it cannot answer the first. Quoting it as anything more would be
study 22's retracted −1.16% again.

### 5.4 Skid makes per-instruction branch-miss profiles readable only in blocks

`branch-misses:u` without `:pp` attributes the sample to an instruction after
the mispredicting branch. In §1.1 the single largest row is a **never-taken
bounds-check `ja`** at 14.66% — which is not a mispredict, it is skid from the
walk exit two instructions earlier. Sum by region, never by row.

---

## 6. The combined panel

`tools/perf27/panel27.sh base final 12` — three-way round-robin, base / a
byte-identical copy of base / the combined arm, order rotated within each round
so the null is measured in the same minutes and the same slot distribution as
the arm.

```
  panel base -> final   rounds=12
    NULL  instructions= +0.000% [ -0.000, +0.000]  branches= -0.000% [ -0.000, +0.000]
          branch-misses= -0.280% [ -0.770, +0.892]  L1d-load-misses= -0.241% [ -0.683, +0.282]
    ARM   instructions= -1.087% [ -1.087, -1.087]  branches= -0.972% [ -0.972, -0.972]
          branch-misses= +0.747% [ -0.274, +2.114]  L1d-load-misses= -0.495% [ -0.911, -0.058]
```

The two gate counters have **zero spread over twelve rounds** on both the null
and the arm — that is what makes ~1% a real number on a box where cycles cannot
see 3%. `branch-misses` is inside its null and on the wrong side of it (§3).
`L1d-load-misses` clears its null narrowly, downward.

---

## 7. Two-sided prediction

On a genuinely idle box, a paired cycles panel of `base` against `final` should
show **`final` faster by 0.6–1.2%**: −1.087% instructions at the engine's
measured IPC, minus a little for the L1d and front-end costs of arm 5.

**If it comes back inside ±0.3%**, the instruction win is being absorbed by the
front end, and **arm 5 (`de381b3`) is the commit to revert first** — it is the
only one that grows code, it is 60% of the instruction win, and the other three
are pure code shape with no plausible icache cost. If it comes back *worse* than
+0.3%, revert arm 5 and re-measure before touching arms 2–4; a regression from
arms 2/3/4 would mean `instructions:u` at CGU=1 is not the gate four studies
have taken it for, which is a much bigger claim and needs its own evidence.

---

## 8. Ranked remainder

Ranked by tractability × size, with **my** measured shares, not inherited ones.

1. **`init_quiets` is the engine's largest branch-miss symbol at 14.43%, and
   ~41% of that is the insertion walk** (§1.1) — ~5.9% of all engine
   mispredicts in one loop. Arm 2 halved its *memory traffic* without touching
   its *branch* count. The remaining idea worth building is the one study 23
   listed as 2b and never tried: a **SIMD insertion-point search over the
   sorted prefix**. With arm 2's layout each `ExtMove` is one qword, so eight
   fit a `zmm`; `vpcmpd` against `tmp.value` plus a popcount gives the
   insertion point `q` directly (the prefix is sorted, so the elements
   comparing `>=` are exactly a prefix), and `valignq` shifts a block by one
   qword lane. Three blocks cover a 22-element prefix. That replaces an
   average-2.8-iteration mispredicting walk with straight-line vector code, and
   arm 2 removed the layout obstacle that made it awkward. Cost: `std::arch`
   intrinsics in a currently-safe file, behind `cfg(target_feature)` with the
   scalar path retained.
2. **Find out where `do_move`'s 5.49% actually is.** §4.1 eliminated the only
   candidate anyone had named. `do_move` inlines `push_dirty` and
   `push_sota_dirty`; a `:pp`-precise profile or an `#[inline(never)]`
   measurement build would localise it in one run. Do this before building
   anything there — this study spent two builds on an attribution that was
   simply wrong.
3. **`search::<NonPv>` at 14.36%**, unexamined at instruction level here. Its
   top rows are `kmovd %ecx,%k1` (5.47%) and one `je` at +0x1674 (4.17%); both
   need a debug-info build to map to source. Largest single symbol in scope
   with no hypothesis attached.
4. **`Board::see_ge` 2.20% + `attackers_to` 2.12% + `attackers_to_exist`
   1.78% = 6.1%**, all in owned files, all loops whose exit is genuinely
   data-dependent. Lower tractability: unlike the sort there is no
   representation trick visible.
5. **The remaining eight indirect dispatches in `next_move`.** Arm 3 removed
   one per node; the rest are blocks reachable from more than one stage.
   Turning the state machine into literal fall-through would remove them, but
   it is a rewrite of a 16-arm match for ~0.9% of engine misses total. Priced
   as not worth it, stated so the next study does not re-price it.
6. **`make_move_with` still `imul $0xa8` twice per node** (push and pop) to
   index the slab. A running pointer beside the cursor would remove both. Two
   instructions per node; listed for completeness, not recommended.
7. **The mailbox is `[Option<Piece>; 64]`, niche-encoded in the colour byte.**
   §4.1 showed this forces a branch on every "what is on this square" that
   needs the piece type. A total representation (`[Piece; 64]` with an empty
   sentinel) would delete that class of branch everywhere, not just in
   `from_move` — but it is a change to a CLAUDE.md-pinned core encoding and
   needs an owner ruling, not an arm.

---

## 9. Gates

Run on the committed tree (`de381b3`):

| gate | result |
|---|---|
| `cargo fmt --check` | clean |
| `cargo clippy --all-targets` | clean |
| `cargo test` | 611 + 9 + 11 + 10 + 6 + 7 passed, 0 failed |
| `cargo test --features sota,arch-d48x` | 704 passed, 0 failed (`RUST_TEST_THREADS=3`) |
| `cargo test --release -- --ignored perft_full_suite` (D24, mandatory — `board.rs` touched) | passed |
| `bench 16 1 6 default depth` | **587683** — the pin, unchanged |
| node identity, study frame | 5,001,562 on every arm and every reject |
| eval hash, wl50 | `5ca978e012da1bb0` on every arm and every reject |

---

## 10. Reproducing

```bash
bash tools/perf27/build27.sh <label>          # CGU=1, thin LTO, memory-gated
bash tools/perf27/probe27.sh <label> [reps]   # exact i/n + br/n + bmiss + L1d + node + hash
bash tools/perf27/loads27.sh <label>...       # L1d loads AND misses (the denominator)
bash tools/perf27/panel27.sh <base> <arm> 12  # four counters, null carried in-round
bash tools/perf27/cyclepanel27.sh <base> <arm> 15   # cycles, directional only
```

Raw output: `docs/research/perf-artifacts-27/probes.txt` (every probe, in
order), `panel.txt`, `cycles.txt`, `loads.txt`, and `patches/` — every rejected
arm's diff, so a reject can be re-measured without re-deriving it:

| patch | applies to | what it is |
|---|---|---|
| `a1b.diff` | pre-study base | §4.1's branchless capture append, the square-select spelling |
| `a5stage.diff` | commit `1552953` | arm 3 before it landed |
| `a3inline-always.diff` | commit `1552953` | arm 5 as the bare one-liner |

In a fresh worktree the scripts need `engines/canonical/net.cenn` to resolve,
which means a `data` symlink to the main checkout (`ln -s
/home/nick/projects/chessengine/data data`) — the net itself is gitignored, and
without it `probe27.sh` fails on the eval hash rather than on the bench.
