# ARCHON perf study #15 — the null-move entry, and a memset nobody had looked for

Date: 2026-07-29. Fifteenth commissioned study (13: `perf-study-13-panic-edges-2026-07-29.md`,
14: `perf-study-14-state-and-memcpy-2026-07-29.md`). Raw artifacts and instruments:
`perf-artifacts-15/`. Baseline: `main` at `8559f19` (study 14 merged and gated).
Comparison target: **`engines/bin/stockfish-eca43a97`**.

---

## 0. HEADLINE

### 0.1 The ratio moved

`bench 16 1 300000 tools/nnue/calib/wl50.fens nodes`, two arms interleaved,
`taskset -c 4`, `nice -n 19`, 5 rounds, both PGO. Raw:
`counters-headline.txt`.

| per node | **ARCHON study 15** | ARCHON study 14 | **SF `eca43a97`** |
|---|---:|---:|---:|
| **instructions** | **5950.1** | 5986.8 | **4452.4** |
| **ratio vs `eca43a97`** | **1.3364×** | 1.3446× | — |
| branches | **580.7** | 631.3 | 362.7 |
| branch-misses | 20.8 | 20.8 | 19.7 |
| nodes | 15,001,568 | 15,001,568 | 15,008,494 |
| `instructions:u` spread | 0.0001% | 0.0207% | 0.0000% |

**Stockfish reproduces 4452.4 with 0.0000% spread for the fourth study running**,
so studies 12–15 are all in one frame and the four ARCHON numbers are directly
comparable. The node count is identical to all three predecessors' to the unit.

**1.3446× → 1.3364×. −36.7 i/n, 2.4% of the 1,534.4 i/n gap, in four
node-identical changes. 1,497.7 i/n remain.**

The branch number is the one worth staring at: **−50.6 branches per node**, which
is 8.0% of our branch count for 0.6% of our instruction count. Our branch ratio
against Stockfish is now **1.601×**, down from 1.84× at study 12 and 1.74× at
study 14. Nothing in this study was aimed at branches.

### 0.2 What landed

Every arm is node-identical (5,001,562 on the probe tree, **15,001,568 on the
headline bench — the same count studies 12, 13 and 14 measured**) and
eval-hash-identical (`5ca978e012da1bb0`). All three gate pins reproduce
unchanged: `Bench: 587683`, eval hash `5ca978e0…0b38`, d13 nodes `1922947`.

| # | change | probe Δ | where |
|---|---|---:|---|
| 1 | **a null move pushes no accumulator entry at all** | **−22.5** | `eval/mod.rs`, `search/mod.rs` |
| 2 | the two null-move **alias walks**, now dead, deleted | −9.7 | `sota_threats.rs`, `sota_accumulator.rs` |
| 3 | **the head forward's `ft`/`nnz` buffers become a caller-owned `HeadScratch`** | **−25.2** | `sota_head.rs`, `sota_context.rs` |
| 4 | `#[repr(align(64))]` on that buffer (D33 reviewers B/C) | −0.6 | `sota_head.rs` |
| 5 | the D33 round's robustness + test changes | **+1.2** | (§9) |
| | **total, probe frame** | **−56.8** | |
| | **total, PGO frame (the number that counts)** | **−36.7** | |

Plus study 14 §8's process item: `tools/check.sh` now **runs** the `sota` test
suite inside the `CHECK_NNUE_GATE` block instead of only type-checking it — and
three new tests, one of which makes `SotaContext::assert_matches_refresh`
execute inside a search **for the first time ever** (§9).

The probe overstated the PGO delta by 1.55× — inside the 1.12–2.0× band the
three predecessors measured, and further evidence for study 12 §0.3's standing
rule: **no probe delta may be quoted as a result.**

### 0.3 What this study is actually about

**(a) Study 14's "the spelling is the arm" is not a curiosity; it is the
dominant term in this region.** The single mechanism of arm 1 — *stop
materialising the accumulator entry a null move would create* — was implemented
three ways, all bit-identical, and measured **+59.4, +35.5 and −22.5**. That is
an **82 i/n spread on one idea**, nearly twice study 14's 45.4. Two of the three
would have been written up as "the mechanism is refuted".

**(b) A 1280-byte libc `memset` was running at every evaluation and no
instrument this project has built could see it.** `let mut ft = [0u8; L1]` in
`SotaHead::forward_into`, immediately overwritten by `activate_ft`. LLVM cannot
dead-store-eliminate it because `activate_ft` is an indirect `kernel_dispatch!`
call, so it cannot prove the buffer is fully written. Study 14's `LD_PRELOAD`
interposer covers `memcpy`/`memmove` and **not `memset`**; the flat profile
attributes the cost to `forward_into`, a symbol three studies had already
declared "at parity with Stockfish's eval head". It took reading the
disassembly of the function's first four instructions.

**(c) Two of study 14's ranked handoff items were mis-sized, and the
measurement that says so cost one command each.** Item 2 (a surviving
2560-byte `memcpy` in `sota_accumulator.rs`, estimated −10…−25) fires
**0.0014 times per node** — worth about 0.05 i/n. Item 5 (derive the threat
snapshot from the parent by XOR, estimated 15–25 i/n of savings) measured
**+106**.

---

## 1. METHOD

Studies 8–14's rules were followed. The two that mattered:

### 1.1 NULL FIRST — and this study's null panel held

**Null A — rebuild determinism.** The same source, forced to recompile
(`touch src/lib.rs`), built and measured twice:

```
null-1           31,973,173,189   6392.6 i/n
null-2-rebuild   31,973,184,178   6392.6 i/n     10,989 instructions apart, ±0.0022 i/n
```

`null-1` reproduced study 14's final published probe line — **6392.6 against its
6392.6** — a cross-study control on the whole instrument.

**Null B — SOURCE PERTURBATION, structurally inert.** Study 14's null failed
informatively when it re-spelled `v.last().expect(…)` as `&v[n-1]` (+86.4 i/n),
and its lesson was: *a null is only inert if it is structurally inert; prefer a
reborrow.* So Null B here is a reborrow through a slice local at the exact two
sites arm 1 was about to change — `let stack = &mut self.stack[..];` in front of
each `split_at_mut` in the two `compute` functions:

```
n0-NULLB-stack-reborrow   31,973,186,554   6392.6 i/n     ±0.0027 i/n vs null-1
```

**±0.003 i/n.** The instrument survives its own null, so every arm below is a
real effect of that arm — including the three that made the engine slower.

### 1.2 Instruments

* **`probe15.sh`** — screening, inherited from `probe14lm.sh`. Thin LTO / 4
  codegen units, native, non-PGO, fixed 5,001,562-node tree; prints the wl50
  eval hash and node count so a tree or eval change cannot hide inside a speed
  claim. **Every build runs inside its own `systemd-run --user --scope -p
  MemoryMax=4G -p MemorySwapMax=0` cgroup** (§1.4).
* **`pgo15.sh` + `pstat8.sh`** — the deliverable frame, same cgroup guard.
* **`memspy15.sh`** — study 14's `LD_PRELOAD` `memcpy`/`memmove` interposer,
  wrapped so it is one command: it runs the bench, resolves every recorded
  return address against the binary's own `objdump`, and prints calls, bytes and
  **calls per node** by caller. New.
* **`flat8.sh`** — inherited instruction-attributed flat profiles.
* **`objdump -d`** — the instrument that actually found arm 3 (§3).

### 1.3 The box, honestly

A D48Y training run held the GPU at 93% and ~6.2 GB RSS for the whole session,
and a `fastchess` match (`d48y-vs-d48x-early`, concurrency 2) started on the box
partway through. Everything here rides on `instructions:u` over fixed-node trees
and on deterministic call counts, both contention-immune. **No wall-clock or NPS
number is quoted as a claim.**

### 1.4 The cgroup guard fired, and that is the point

The second gate run was **OOM-killed inside its own cgroup**
(`run-p1125822-i1127365.scope: The kernel OOM killer killed some processes in
this unit`) when the new `cargo test --features sota,arch-d48x` step ran 695
tests at one thread per core while the match was live — each `Worker` owns ~20
MiB of history tables plus two multi-megabyte accumulator stacks. The global OOM
killer was never involved and the training run never noticed. That is exactly
the failure mode that cost this project two training runs before study 14 made
the cgroup mandatory. The fix was to cap the step's `--test-threads` (§7.1), not
to raise the limit.

---

## 2. THE NULL-MOVE ENTRY — one mechanism, three spellings, 82 i/n apart

### 2.1 What was there

A null move leaves every piece where it stands. Our search recorded that as an
**empty `DirtyPiece` pushed into both accumulator stacks**:

```rust
self.eval_ctx.push_dirty(DirtyPiece::EMPTY, ply);
self.board.make_null_move();
self.eval_ctx.push_sota_dirty(DirtyPiece::EMPTY, ply, &self.board);
```

which cost, per null move: a `ThreatPosition::from_board` snapshot (six
`pieces_of_type` unions plus an ~88-byte store), two `king_crossed` scans, the
`DirtyPiece` stores — and then, at fold time, a **2560-byte parent copy per
perspective per stack** for every fold span that crossed the null ply, plus a
`build_deltas` pass over an empty record. Study 14 §3 measured the threat half of
that at **0.313 copies per node** and named it as its ranked-#1 handoff item.

**Stockfish does not do any of it.** `do_null_move` has no
`accumulatorStack.push()` and `undo_null_move` has no `pop()`
(`eca43a97:src/search.cpp:657-669`), so SF never materialises an empty-delta
child at all.

### 2.2 Spelling one — make the entry TRANSPARENT (+59.4)

Keep the push; teach both `compute` functions to *skip* an empty-delta child
inside a fold span, carrying a `src` cursor for the last materialised ply so the
next real child folds straight from it; and skip the `ThreatPosition` snapshot
for an empty record. Provably correct, and it deletes exactly the work §2.1
names.

**+59.4 i/n.**

### 2.3 Spelling two — the same thing, without the duplicated copy (+35.5)

Spelling one had duplicated study 14's chunked 2560-byte copy into a second
block, in both files, to handle the one case that must still materialise (the
fold's leaf). That is ~80 inline instructions of near-never-executed code added
to the two hottest loops in the engine — study 13 §5.5's finding, that cold code
in a hot function moves the hot path, applied to itself. Removing the
duplication recovered **23.9 i/n**, and the arm was still **+35.5**.

### 2.4 It is not the bounds check (+5.9 on top)

The obvious remaining suspect was `&before[src]`: where `&before[child - 1]`
needs no bounds check (`before.len() == child` is a fact LLVM already has), a
free variable needs a `cmp` and a panic edge. Clamping it — `&before[src.min(before.len() - 1)]`,
the idiom that makes the check provably dead — measured **+5.9 i/n worse**, not
better.

Which is study 13 §5.2's refutation reproduced at a second site, and it is worth
stating plainly: **the bounds-check lever is dead in this codebase and has now
been priced at zero three times.**

### 2.5 Spelling three — never create the entry (−22.5). LANDED

What actually costs is losing `parent == child - 1`. With a free `src` the
compiler no longer knows the fold's source is the entry immediately below its
destination, and the 40-iteration kernel loses the addressing and aliasing facts
that relation gave it. No spelling of "skip the entry" can give that back,
because the entry is still *there*.

So: don't create it. `EvalContext` gains one `usize`:

```rust
/// How many plies of the CURRENT line are null moves.
null_plies: usize,

fn acc_ply(&self, ply: usize) -> usize { ply - self.null_plies }
```

and the search brackets the null exactly as it brackets the board:

```rust
self.eval_ctx.push_null();
self.board.make_null_move();
let null_value = -self.search::<NonPv>(…, ply + 1);
self.board.unmake_null_move();
self.eval_ctx.pop_null();
```

Every accumulator index — both pushes, `evaluate`, the policy accumulators —
goes through `acc_ply(ply)`, translated **once, at the `EvalContext`
boundary**, so no inner context's signature changes and no fold loop changes by
one byte.

A scalar suffices because the search is a depth-first walk: the "current line"
*is* the recursion stack, so one counter describes it. The child of a null move
and the null-move node itself share one accumulator entry, which is correct
because their accumulators are equal by construction — the side-to-move swap the
head needs comes from the board, not from the entry.

**−22.5 i/n and −7.6 branches per node.** And it removes strictly more work than
spelling one did: the snapshot, the dirty stores, the fold-time copies, *and* the
push calls themselves.

### 2.6 The alias walks, now dead, deleted (−9.7). LANDED

Both stacks carried a null-move alias walk at the top of their accessor
(`ThreatContext::at_full`, `HalfkaContext::current_full`, from perf study #4
item #3):

```rust
while target > 0 && !computed[0] && !computed[1]
      && dp.adds().is_empty() && dp.subs().is_empty() { target -= 1; }
```

Its entire purpose was to hand back the parent instead of materialising an
empty-delta child. With §2.5 there are no empty-delta children, so the loop
could never move — it was ~4 tests per evaluated node proving a fact that is now
structural.

Deleting it is also safe for the cursor API (`ThreatContext::apply`), which a
caller outside the search can still hand an empty record: that entry is now
*materialised* by the fold — `build_deltas` yields no events, so the empty-list
branch copies it — which is the same value the alias returned.

**−9.7 i/n.**

---

## 3. THE MEMSET — 1280 bytes per evaluation, in a block three studies called "at parity"

### 3.1 How it was found, and why nothing found it earlier

Study 13 §7's block table put the eval head at **572 i/n against Stockfish's
572** and closed it. Study 14's `memcpy`/`memmove` interposer, re-run here as
step one (`memspy-baseline.txt`), reported 638 bytes per node of copy traffic
with nothing in the head at all.

Both instruments were blind to the same thing, for different reasons: a block
comparison cannot see waste that both sides' *symbol* contains, and the
interposer hooks `memcpy` and `memmove` and **not `memset`**.

What found it was disassembling the first four instructions of the hottest
symbol:

```
74dd0 <SotaHead::forward_into>:
  74de9:  lea    0x1a0(%rsp),%rdi
  74df1:  mov    $0x500,%edx          ; 1280
  74df9:  xor    %esi,%esi
  74e04:  call   *…                   ; memset@GLIBC_2.2.5
  74e0a:  …                           ; activate_ft, which overwrites all 1280
```

### 3.2 Why the compiler could not remove it

```rust
let mut ft = [0u8; L1];              // L1 = 1280
activate_ft(pst, threat, stm, &mut ft);
let mut nnz = [0u16; L1 / 4];        // 640 more bytes
let count = find_nnz(&ft, &mut nnz);
```

`activate_ft` writes every one of the 1280 lanes before anything reads them, so
the zero-init is dead. But `activate_ft` dispatches through the runtime SIMD
`kernel_dispatch!` — an **indirect call** — and LLVM cannot see through it to
prove the buffer is fully written. So the initialiser survives, as a libc call
plus 1280 bytes of stores, at every evaluation.

`nnz` is worse in one way and better in another: it is genuinely *not* fully
written (only `&nnz[..count]` is ever read), so no compiler could ever remove
its initialiser.

### 3.3 The fix is a buffer, not an `unsafe`

The reflexive fix is `MaybeUninit` plus `unsafe`, carrying a safety obligation
("every kernel writes every lane") that a future kernel could quietly break.
The fix that needs no safety argument is the one `ThreatContext::add_scratch`
already uses in this codebase: **a long-lived buffer has nothing to
initialise.**

```rust
pub struct HeadScratch { ft: [u8; L1], nnz: [u16; L1 / 4] }
```

owned by `SotaContext`, passed to `forward_into`, allocated once per thread per
net. Stale bytes are unreadable by construction: `ft` is fully written by
`activate_ft` before any read, and `nnz` is consumed as `&nnz[..count]` with the
`count` `find_nnz` just returned.

The same change collapses `SotaContext::evaluate`'s two-arm branch —
`forward_into` when a policy head is loaded, `forward` (with its own 512-byte
`[f32; L2]` local) otherwise — into one call that always writes `last_l2` and
gates only the *validity flag*. The store is the same store either way; what
goes away is a per-eval branch and a second instantiation of a ~600 i/n
function.

**−25.2 i/n and −4.6 branches per node.** The disassembly of the shipped binary
has no `memset` call in `forward_into` at all.

Two more followed from the D33 round. `HeadScratch` is `#[repr(align(64))]`
(**−0.6**, kept on reference convergence rather than on its number — §9), and
the smallnet and largenet tiers now share the context's buffer instead of
building their own: all three reviewers independently found that
`SotaHead::forward`'s convenience wrapper puts the 1920-byte zero-init straight
back, on paths that are inert today and would not be if either tier were ever
armed.

---

## 4. WHAT DID NOT LAND

### 4.1 Deriving the threat snapshot from the parent — **+106 i/n** (study 14 §8 item 5, REFUTED)

`ThreatContext::apply_at` rebuilds an ~88-byte `ThreatPosition` from the board
at every push (1.0 per node). Study 14 item 5 proposed deriving it from the
parent's snapshot by ≤4 `toggle`s and sized it at 15–25 i/n of savings. The
arithmetic agrees: derive is ~18 ops (an 88-byte copy plus 2–4 three-XOR
toggles) against `from_board`'s ~25 (six `pieces_of_type` unions, three
occupancy reads, two `king_square` scans).

**It measured +106.0 i/n**, node-identical and eval-hash-identical, and the
revert reproduced the baseline to 0.005 i/n (`a3r-revert-control`).

Why is a hypothesis with two named parts, neither measured separately: the
parent snapshot and the child snapshot are two elements of the **same
`Box<[ThreatEntry]>`**, so LLVM cannot prove the 88-byte read and the 88-byte
write do not alias; and the edit list is a **runtime-length loop** where
`from_board` is straight-line. Per CLAUDE.md rule 4 that is written down as a
hypothesis. The verdict is not: the arm is priced and negative, and the reason
now lives in a comment at the site so study 16 does not re-derive it.

### 4.2 The surviving 2560-byte `memcpy` — **~0.05 i/n, not −10…−25** (study 14 §8 item 2, MIS-SIZED)

Study 14 handed over `entry.acc[p] = parent.acc[p]` at `sota_accumulator.rs:333`
and `:303` as "same condition, same payload" as the copy it had just killed, at
high confidence, worth −10…−25 i/n.

The interposer, run before writing any code (which is what study 14's own
handoff prescribed), says the site fires **205 times over 150,059 nodes —
0.0014 per node** (`memspy-baseline.txt`), against the threat twin's 0.313. As a
libc call that is worth about 0.05 i/n. The two sites are not symmetric, and the
one-command check that shows it is the same command study 14 recommended.

Arm 1 removed the site's traffic anyway — no null move creates an empty-delta
HalfKA entry any more, and the post-arm census (`memspy-a4.txt`) has no
`HalfkaContext::current_full` row at all.

---

## 5. THE ARM LOG

`probes-fast.txt`, in order, probe frame. Node count **5,001,562** and eval hash
**`5ca978e012da1bb0`** on every line, including the three regressions and the
reverted arm.

```
null-1                       6392.6    baseline, main @ 8559f19 (= study 14's published 6392.6)
null-2-rebuild               6392.6   +-0.0022  rebuild null
n0-NULLB-stack-reborrow      6392.6   +-0.0027  structural source-perturbation null
a1-null-entry-transparent    6452.0    +59.4   rejected (S2.2)
a1b-transparent-minimal      6428.1    +35.5   rejected (S2.3)
a1c-transparent-min-index    6434.0    +41.4   rejected (S2.4) — +5.9 vs a1b
a1d-no-null-entry            6370.1    -22.5   LANDED (S2.5)
a2-drop-alias-walks          6360.4     -9.7   LANDED (S2.6)
a3-threatpos-from-parent     6466.4   +106.0   REVERTED (S4.1)
a3r-revert-control           6360.4   +-0.005  revert control
a4-head-scratch              6335.2    -25.2   LANDED (S3)
--- D33 review round ---
a5-headscratch-align64       6334.6     -0.6   LANDED (S9)
a6-postreview                6335.8     +1.2   the round's robustness + test changes
a6b-postreview-fmt           6335.8   +-0.00002 the SHIPPED tree (rustfmt only)
```

The last line is the shipped tree: **−56.8 i/n from null-1**, and it is 1.2 i/n
*worse* than the pre-review tree. That 1.2 bought `saturating_sub` on two
release-reachable index computations, a shared scratch on the two tier paths, a
tightened const assert, and three new tests — §9 argues it was cheap.

---

## 6. THREATS TO VALIDITY

1. **Probe deltas overstate PGO deltas** — here by 1.55× (−56.8 → −36.7),
   inside the 1.12–2.0× band of the three predecessors. Every per-arm figure in
   §5 is probe-frame; only the total is PGO-measured, and no per-arm PGO
   attribution is claimed.
2. **Why spelling a1 lost, and why a3 lost, are hypotheses, not measurements**
   (§2.5, §4.1). Both are labelled as such at the site and here.
3. **Nothing here was Elo-tested, and there is nothing to test**: every landed
   arm is node-identical on three independent trees (5,001,562 probe /
   15,001,568 headline / 587683 gate) and eval-bit-identical, so the engine
   plays exactly the same moves. The one behavioural surface that is *not*
   covered by a pin is the D47 policy-ordering path, which is `None` in every
   gated configuration (study 13 §9 established this) — §7.1 says what covers it
   instead.
4. **The memspy counts are from a 150,059-node bench**, a shorter tree than the
   probe's. Call *rates* per node are what is quoted; absolute totals are for
   that tree only.
5. **Box state**: a training run and a live match shared the box for the whole
   session (§1.3). No wall-clock or NPS number is quoted as a claim.
6. **§0.1's branch drop (−50.6/node) is not attributed.** The landed arms
   together account for −10.0 branches per node in the probe frame
   (683.6 → 673.6); the PGO frame shows −50.6. The difference is PGO's own
   branch layout responding to the changes, and this study did not decompose it.
   Quote the instruction number, which is the study's metric; the branch number
   is reported, not claimed.

---

## 7. GATE

`gate-s15.log` (the final tree, `CHECK_NNUE_GATE=1`).

* `bash tools/check.sh` — **`Bench: 587683`**, unchanged. The diff touches
  `eval/mod.rs` and `search/mod.rs`, which the default-feature build compiles,
  so this is a real node-identity check on arm 1's search-side half.
* **`CHECK_NNUE_GATE=1 bash tools/check.sh` — MANDATORY here** (three of the
  four changed files are under `src/eval/nnue/`) and both pins reproduce:
  * eval hash `5ca978e012da1bb086799709c3ea5350cf6d49923c254f7975b6d780b0098b38`
  * d13 nodes **1922947**
* `cargo fmt --check`, `cargo clippy --all-targets -- -D warnings`, the feature
  matrix, `cargo test` (687 lib tests), the release perft suite
  (`Nodes searched: 1839507`), and the two-run deterministic bench-perft
  fingerprint.
* The D24 perft ritual does not apply — nothing here touches `movegen`, `board`
  mutation, `zobrist` or `attacks` — but it runs inside check.sh regardless and
  passes.

### 7.1 check.sh now RUNS the sota tests (study 14 §8's process item, landed)

Study 14's handoff: `tools/check.sh` `cargo check`s the feature matrix but
`cargo test`s only at **default** features, under which `src/eval/nnue/`'s sota
half is not compiled at all. So ~20 tests — including
`threats_replay_matches_refresh` and, far more importantly, the debug
`SotaContext::assert_matches_refresh` oracle that runs **at every node of every
debug search** — were type-checked and never executed by any gate.

The `CHECK_NNUE_GATE` block now runs them. For this study that oracle is the
load-bearing correctness check, not the pins: it recomputes both accumulators
from scratch at every evaluated node and compares, so an off-by-one in
`acc_ply` or a stale `HeadScratch` lane fails a test rather than shifting a
number.

`--test-threads` is capped at 4 (`CHECK_TEST_THREADS` overrides). That is not
cosmetic: at one thread per core the step OOM-killed its own cgroup (§1.4) and,
on the run before that, tripped `stop_latency_is_bounded` — a 50 ms wall-clock
bound measured with `Threads=8` — at 67.9 ms.

### 7.2 Each landed change has a test that can fail, and each was mutation-verified

Three of these tests did not exist before the D33 round (§9).

| change | test | mutation that makes it fail |
|---|---|---|
| the `acc_ply` mapping, SOTA (a1) | **`search_sota_accumulator_stays_consistent_across_the_tree` — NEW**; per-node `SotaContext::assert_matches_refresh` over three positions, asserting the trees contain a null move | `push_sota_dirty` indexing `ply`; `at_full` reading `ply + 1`; dropping `pop_null` — **all three applied and all three FAILED** |
| the `acc_ply` mapping, v1 (a1) | `search_nnue_accumulator_stays_consistent_across_the_tree`, now with `assert!(st_nmp_try > 0)` | `acc_ply` returning `ply` ⇒ the null child indexes a never-pushed entry |
| `push_null`/`pop_null` balance (a1) | the `debug_assert!(null_plies <= ply)` in `acc_ply`, fired by both oracles | move `pop_null` inside the `if null_value >= beta` block |
| alias-walk deletion (a2) | **`an_empty_record_materialises_to_the_parent` — NEW**, plus `threats_replay_matches_refresh` and the d13 pin | an alias walk that returns an ancestor; `build_deltas` emitting an event for an empty record |
| `HeadScratch` reuse (a4) | **`a_poisoned_scratch_gives_the_same_forward_as_a_fresh_one` — NEW**, plus `dispatched_activate_matches_scalar_bitwise` with both buffers now POISONED rather than zeroed | make AVX-512 `activate` skip its last 32-lane block — **applied; both FAILED** (before this round, neither did) |
| `HeadScratch` width invariant (a4) | `const _: () = assert!(ACC.is_multiple_of(2 * LANES))` — a *compile* error | an `L1` that is a multiple of 32 but not 64 |
| `last_l2_valid` gating (a4) | *(see below)* | |

**The one hole, stated rather than papered over.** `last_l2_valid`'s new
spelling — always write `last_l2`, set the flag only when a policy head is
loaded — has **no test that can fail**, for exactly the reason study 13 §9.1
documented: `PolicyHead::sibling_of` resolves the `.cpol` sidecar against
`engines/canonical/net.cenn`, while the real sidecar sits beside the symlink's
*target*, so `policy_head` is `None` in every gated configuration and both
branches of the new `if` are unreachable in the gate. The change is verified by
inspection of `last_l2_for`'s two consumers (both require `last_l2_valid &&
last_l2_key == board.key()`), and by the fact that `eval::evaluate` already
called `invalidate_sigma()` — which clears the same flag — unconditionally
before it. Closing it needs a test-only installer for the policy head; that is
study 16's cheapest correctness debt, inherited unchanged from study 13.

---

## 8. RANKED REMAINDER — handoff to study 16

**1,497.7 i/n remain**, of which ~327 is the 1280-vs-1024 accumulator width (a
**retrain, not a code change** — study 12 §11 item 9, unchanged) and, by study
13 §7, ~0 is the accumulator fold, the threat generator, movepick or the search
body. **Do not reopen study 13 §0's closed list**, with one amendment: the "eval
head at parity" entry was true of the head's *arithmetic* and was hiding a
1280-byte memset in the same symbol (§3). Block-level parity is not per-line
parity.

Study 14's items 1 (null-move entry), 2 (the surviving `memcpy`) and 5
(`ThreatPosition` from the parent) are **closed by this study** — item 1 landed,
items 2 and 5 are priced and dead.

| # | item | size | confidence | cheapest next test |
|---|---|---:|---|---|
| 1 | **Hunt the rest of the `memset` class.** §3 found one by disassembling a hot symbol's prologue. The systematic version is one command: `objdump -d` the release binary, bucket every `call *…memset/memcpy/memmove` by enclosing symbol, and discard the constructors. On the shipped tree the survivors are `SotaContext::evaluate` (two, both on inert smallnet/largenet paths, from `HeadScratch::new()` inside the `forward` wrapper) and `HalfkaContext::{current_full, finny_refresh_into}` | ? | **high — the method just paid** | the command is in §10; **also extend `memspy.c` to interpose `memset`**, which is four lines and would have found §3 on day one |
| 2 | **`Worker::do_move`, 4.58% ≈ 300 i/n** — the largest search-side symbol after `search`/`init_quiets`, and study 13 §7 put `make/unmake` at **+359 vs SF**, most of which is this | ? | medium | inherited from study 14 item 7 and still un-split. `capture_stage`, `ContHistId::new`, `DirtyPiece::from_move` and the `Option<DirtyPiece>` carried across a 650-instruction `make_move_with` all live inside it; none has been priced. Study 13 §5.1 established that register pressure, not duplicated work, is the binding constraint here — so price the *spill*, not the derivation |
| 3 | **The eager king-cross threat refresh copies 2560 bytes by value.** `self.stack[child].acc[pi] = refresh_threat(weights, board, persp)` builds the accumulator in a stack local and then copies it into the entry — the interposer counts **0.0357 calls/node, 91 bytes/node** (`memspy-a4.txt`, the two `apply_at` rows), which is now the largest surviving `memcpy` in the engine | ~4–8 | **high — counted** | write in place, the way `finny_refresh_into` already does (`sota_accumulator.rs` §"two-destination fold"). Small, but it is measured rather than estimated |
| 4 | `attacks::tables()` — 5.62 `LazyLock` derefs/node | ~17–22 | high | unchanged from study 13 item 5 and still untried. The hot callers already hoist, so the remaining cost is the guard itself: an acquire load, a test and a branch, ×5.62. Removing it needs a statically-addressed table — `const`-evaluating the magic search (the risk is build time, and `init_magics` uses `Vec`, so it is a real refactor) or a start-up-initialised `static` |
| 5 | `Histories::correction_value` + `update_continuation_histories`, 3.42% ≈ 224 i/n | +101 vs SF | medium | `ContHistId::piece_to()` is `flat % 832` — a non-power-of-two modulus (≈5 instructions) run ~4×/node. The layout is a free permutation: encode the flat slab index as `(piece13 * 64 + to) * 4 + (in_check * 2 + capture)` instead of `((in_check*2 + capture) * 13 + pc) * 64 + to`, and `piece_to()` becomes `flat >> 2`. `SENTINEL` moves from 768 to 3072; nothing else observes the ordering |
| 6 | **The dead side-net `Option` checks**, ~13/node | ~10–15 | medium | `EvalContext` tests `nnue`/`simple_halfka`/`protected_halfka` and `SotaContext` tests `smallnet_halfka`/`largenet_halfka`/`largenet_threats` on every push, all `None` in production. One `bool` per context collapses six tests to two. Below the site sensitivity study 13 demonstrated, so it needs its own null — and study 14 §8's rule 3 applies: a win smaller than the instrument's resolution is an unresolved measurement, not a small win |
| 7 | **The FULL Stockfish transposition** — *replace* `pieces[c][pt]` with `byTypeBB & byColorBB` | ? | medium | study 14 §4.6/§8 item 4, unchanged. Removes a layer instead of adding one; costs a load + `and` on ~72 `pieces(c, pt)` sites. Needs its own null and could easily be a loss |
| 8 | `upcoming_repetition` is missing SF's `other != 0` early-out | ~10 | medium | `eca43a97:src/position.cpp:1558-1570`. Unchanged from study 14 item 9 |
| 9 | `Board::clone()` loses the `history` reserve | 0 i/n; a latency spike | low | unchanged from study 14 item 10 |
| 10 | **`HalfkaContext::compute`'s two empty-dirty branches are now PROVABLY dead** | a few | **high — D33 reviewer C** | for HalfKA a dirty record is empty only for a null move (every real move has ≥1 add and ≥1 sub, pinned by `accumulator.rs`'s own tests) and `stack[0].dirty` is never read, so the branch is now taken 0×/node. Deleting is correctness-neutral: `apply_i16_deltas_into(dst, src, w, &[], &[])` **is** `dst = src`. Unlike study 14 §3.1's case it cannot re-materialise a `memcpy`, because the path is not taken. **Do NOT do the same to `ThreatContext`'s `add_len == 0 && sub_len == 0`** — that one is live (a real move whose relational threat events cancel); count it first |
| 11 | **Move the accumulator cursor INTO the contexts** | 0 i/n; a defect class | **high — D33 reviewer B** | §9.4. SF's `AccumulatorStack::size` and Reckless's `Network::index` both make a wrong accumulator index *unrepresentable*; our `acc_ply` makes it merely wrong at four translation sites, and no assert can catch a *missing* call. Not a perf item — a correctness-shape item that this study's arm 1 created and should be paid off |
| 12 | **`tools/check.sh` clippies DEFAULT features only** | 0 i/n | **high — D33 reviewer C** | roughly a third of `src/` is never linted. Turning it on today fails on **6 pre-existing** lints in `simple_halfka.rs` and `protected_halfka.rs` (`mem::drop` of a non-`Drop`, `needless_range_loop`, `needless_return`). A cleanup commit, then a gate step |
| 10 | ~~null-move accumulator entry~~ / ~~the surviving `memcpy`~~ / ~~`ThreatPosition` from the parent~~ | | **CLOSED** | §2 landed; §4.2 and §4.1 priced and dead |

**Three method rules this study is handing over.**

1. **The spelling is the arm — and the winning spelling is usually the one that
   deletes the object, not the one that skips it.** Three spellings of one
   mechanism spanned 82 i/n (§2). The two that lost both kept the data
   structure and taught the code to step around it; the one that won removed it.
   Corollary for a study that has an arm at +59: **you are two spellings from
   the answer, not one.**
2. **Every instrument this project owns is a filter, and the thing you are
   looking for may be in the part it filters out.** A block table cannot see
   waste inside a symbol that is at parity; a `memcpy` interposer cannot see
   `memset`; a flat profile cannot see a call it attributes to its caller.
   §3 was found by reading four instructions of disassembly, which is the one
   instrument with no filter.
3. **Re-derive an inherited size before spending a build cycle on it.** Study
   14 item 2 was sized at −10…−25 and is worth 0.05 (§4.2); item 5 was sized at
   +15…+25 of savings and cost +106 (§4.1). Both checks were one command. This
   is CLAUDE.md research rule 8, and it now has two more data points.
4. **A perf arm can create a correctness defect that no pin can see, and the
   D33 round is the only thing that catches it.** Arm 1 was node-identical,
   eval-hash-identical and gate-clean, and shipped a release-only silent
   corruption path (§9.1). Arm 4 was the same, and turned an untested invariant
   from harmless into a stale-eval bug (§9.2). Neither is visible in a number.
   **Budget the review round as part of the arm, not as a formality after it.**

---

## 9. THE D33 REVIEW ROUND

Three read-only reviewer personas ran against the landed diff (A
correctness/safety, B reference-fidelity vs `eca43a97`, C Rust-idiom/perf/
test-quality). **No blockers.** The round cost **+1.2 i/n** and was worth it
three times over; here is what it changed.

### 9.1 It found a release-reachable silent-corruption path in the study's own arm

`acc_ply` was `ply - self.null_plies` behind a `debug_assert!` only (A1). In
release, an unbalanced counter does not panic — it **wraps**. `null_plies ==
ply + 1` makes `acc_ply` return `usize::MAX`, and `HalfkaContext::push`'s
`stack[ply + 1]` wraps back to **`stack[0]`**, overwriting the root entry with a
dirty record and `computed = [false; 2]`. Every later `compute` then walks to a
root it believes is uncomputed and folds from garbage: wrong evals for the rest
of the search, no crash, no assert.

Reviewer A also found the precedent three lines from where I should have looked:
`NnueContext::compute` already documents choosing a saturating ancestor walk for
exactly this reason ("`k` saturates at 0 rather than underflowing `usize` to a
wild index"). Both `acc_ply` and `pop_null` now use `saturating_sub`, the
`debug_assert!`s stay, and the release behaviour is documented rather than
assumed.

**This is the finding the round exists for.** The arm was node-identical,
eval-identical, gate-clean, and had a latent release-only failure mode that no
pin in this project could ever have caught.

### 9.2 It found that the study's biggest arm made an untested invariant load-bearing

`HeadScratch`'s whole safety argument is "`activate_ft` writes every one of `L1`
lanes before anything reads them". Reviewers A and C both noticed that **nothing
checked it**, and — worse — that arm 4 changed the consequence of it being
false. While `ft` was a fresh `[0u8; L1]` per call, a lane a kernel failed to
write was *zero*, and ~85–95% of activations are zero after the clamp, so the
defect was invisible and deterministic. Reused, that lane is the **previous
position's activation**: a silently history-dependent eval, in an engine whose
entire merge gate is bit-identity.

The existing cross-kernel test could not see it either, because it compared two
buffers that both started zeroed (C, T3). Three things landed:

* `dispatched_activate_matches_scalar_bitwise` now poisons its two buffers with
  **different** non-zero fills, so a lane neither kernel wrote is a mismatch.
* A new `a_poisoned_scratch_gives_the_same_forward_as_a_fresh_one` in
  `sota_head`, over both sides to move and three output buckets, closing the
  loop through the whole head.
* `const _: () = assert!(ACC.is_multiple_of(2 * LANES));` in the AVX-512 module
  (A4). `activate_inner` steps the half-accumulator by 32 with no remainder
  handling, so it needs `ACC % 64 == 0`; the module only asserted 32. Every
  defined `L1` satisfies it — the point is that a future width cannot silently
  leave the tail of `ft` unwritten.

**Mutation-verified**: making the AVX-512 `activate` skip its last 32-lane block
fails both tests. Before this round it failed neither.

### 9.3 It found that the SOTA accumulators had never been oracle-tested in a search

`grep` for `set_sota_net` across `src/` and `tests/` returns **zero test
callers** (A2, C M3). So `SotaContext::assert_matches_refresh` — 20 lines that
recompute both accumulators and both PSQT channels from scratch and compare, at
every evaluated node of a debug search — had **never executed inside a search**,
ever. Everything that could see the SOTA accumulators was a *snapshot*: the d13
node pin fires only if a mistake happens to perturb the tree.

That is precisely the hole study 14 §7.1 found in `has_repeated` — a claimed
path with nothing executing it — one layer up.

`search_sota_accumulator_stays_consistent_across_the_tree` closes it in ~35
lines, using the `pub(crate)` loader fixture that existed for exactly this.
**Mutation-verified in three directions**, each applied and reverted:

| mutation | result |
|---|---|
| `push_sota_dirty` indexes `ply` instead of `acc_ply(ply)` | FAILED |
| `pop_null` deleted | FAILED |
| `at_full` reads `stack[ply + 1]` | FAILED |

And its v1 twin gained one line (C, T2): `assert!(st_nmp_try > 0)`. That test's
comment had claimed for years that its positions "reach … null moves"; nothing
asserted it, so the coverage of arm 1's ply mapping was an argument. It is now a
measurement — and it passes, so the claim was true.

### 9.4 Reference fidelity: the citation holds, the shape does not quite

Reviewer B verified `eca43a97:src/search.cpp:657-669` **exactly** — `do_null_move`
is three stack-marker stores and `pos.do_null_move(st)`, with no
`accumulatorStack.push()`, against `do_move`'s push at `:644`; `undo_null_move`
at `:669` pops nothing; and `Position::do_null_move` (`:position.cpp:1367-1396`)
touches no NNUE state at all. Reckless is the same shape independently
(`Board::make_null_move` takes no `BoardObserver`, so its NNUE hook *cannot*
fire).

But the *mechanism* differs from ours and the difference is worth recording: SF
keeps a cursor **inside** the accumulator stack (`AccumulatorStack::size`,
`nnue_accumulator.h:135`), moved only by its own `push`/`pop`. That makes a
wrong index unrepresentable. Ours derives the index at four translation sites,
which makes it merely *wrong* if one is missed — and `acc_ply`'s
`debug_assert` cannot catch a missing call. The invariant is now stated on
`EvalContext::null_plies`, and moving the cursor into the contexts is §8's
follow-up rather than something hidden in a perf commit.

B also verified the exact opposite half of arm 3: Stockfish's `NNZInfo` leaves
its `u16 nnz[Dimensions / 4]` **deliberately uninitialised**
(`eca43a97:src/nnue/nnz_helper.h:30-35`), for our reason verbatim — only
`..count` is ever read — and `transformedFeatures` is an uninitialised
cache-line-aligned local (`network.cpp:144-148`). Rust cannot express "do not
initialise" without `MaybeUninit` + `unsafe`; a long-lived buffer reaches the
same place with no safety obligation. And the pinned revision **is itself an
instance of this work**: `eca43a97`'s commit message is "Remove unnecessary
memset(), memcpy() from NetworkArcitecture::propagate()".

B and C both then asked for `#[repr(align(64))]` on `HeadScratch`, which SF
(`alignas(CacheLineSize)`) and Reckless (`Aligned<T>`) both apply to the same
buffers and which this codebase already applies to `Entry`/`ThreatEntry` on a
measured −1.29% L1d. It measured **−0.6 i/n** and is kept on that convergence,
not on its number.

### 9.5 What the round cost, and the three things it changed in the code

`a5` −0.6, then `a6` +1.2 for: the two `saturating_sub`s; the `push_sota_dirty`
tripwire (C, M8 — the assert was on `push_dirty`, which feeds the **v1** stack,
not on the two stacks whose alias walks this study deleted); the tier paths
sharing `head_scratch` (A5/B M3/C M1 — all three found that
`SotaHead::forward`'s wrapper puts the 1920-byte zero-init straight back, inert
today but not if either tier is armed); and a `needless_borrow` fix.

Reviewer items deliberately **not** taken, with reasons: moving the accumulator
cursor into the contexts (B, M1 — the right shape, but it touches every
accumulator context and belongs in its own commit → §8); deleting
`HalfkaContext::compute`'s now-dead empty-dirty branches (C, follow-on 3 — a
measurable arm, not a rider on this diff → §8); adding a `--features sota`
clippy step to `check.sh` (C — correct, and it fails on **6 pre-existing** lints
in `simple_halfka.rs`/`protected_halfka.rs`, so it is a cleanup commit, not a
perf one → §8); and `no_policy_head_leaves_last_l2_invalid` (C, T5 — real, and
it belongs with the policy-head test installer that study 13 §9.1 already owes).

Three doc claims were corrected against the code: the `ThreatPosition` size
(74 B used of 80 padded, not "~88"); the arm-3 mechanism (now labelled a
hypothesis, with the note that the codebase's own `split_at_mut` aliasing idiom
was never tried); and `check.sh`'s new step, whose comment credited itself with
SOTA-in-search oracle coverage that did not exist until §9.3 — and whose
`--test-threads` cap is now documented as what it is: a workaround for a box
that runs a trainer, not a fix for a load-sensitive timing test.

---

## 10. ARTIFACTS (`perf-artifacts-15/`)

* `probes-fast.txt` — every arm with its eval hash and node count, including
  both nulls, the three rejected spellings, the reverted arm and the revert
  control.
* `counters-headline.txt` — the §0.1 two-arm PGO panel, 5 rounds, with the
  parsed ratio table appended.
* `memspy-baseline.txt`, `memspy-a4.txt` — the `memcpy`/`memmove` census by
  caller and **calls per node**, before and after.
* `flat-a2-instructions.txt`, `flat-a4-instructions.txt` —
  instruction-attributed flat profiles.
* `gate-s15.log` — the final `CHECK_NNUE_GATE=1` gate run.
* `scripts/probe15.sh`, `pgo15.sh`, `memspy15.sh` — new.
* `scripts/memspy.c`, `pstat8.sh`, `parse8.py`, `flat8.sh`, `annot8.py`,
  `opcount11.py`, `blocks13.py`, `sfshow.sh` — inherited.

The one-command `memset` census of §8 item 1:

```bash
objdump -d --demangle target-fast15/release/chessengine \
  | awk '/^[0-9a-f]+ </{s=$0} /call.*<mem(set|cpy|move)@/{print s" ||| "$0}'
```
