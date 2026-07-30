# ARCHON perf study #24 — four landed arms in the threat generator, and the cycles axis dies

Date: 2026-07-29. Twenty-fourth commissioned study (19:
`perf-study-19-at-full-split-2026-07-29.md`, 18:
`perf-study-18-lazy-king-cross-2026-07-29.md`). Baseline: `main` at
**`2f210c9`**. Scope: `src/eval/nnue/sota_threats.rs`, `src/board.rs`,
`src/movegen.rs`, `src/moves.rs`. Instruments: `tools/perf24/`.

---

## 0. HEADLINE

### 0.1 What landed — four arms, all in `sota_threats.rs`

| # | arm | Δ instructions | Δ branches | Δ L1 loads | commit |
|---|---|---:|---:|---:|---|
| r1 | `resolve_both` hand-unswitched on `has_psqt` (+ `as_chunks` hoisted out of the event loop) | **−20.93** | −7.56 | −25.19 | `c554722` |
| d1 | `#[repr(align(8))]` on `ThreatDelta` — size 7 → 8 | **−37.73** | −0.67 | −19.34 | `4766ae1` |
| k1 | `king_cross` without the intermediate `[Option<Square>; 2]` arrays | **−20.51** | −3.34 | n/r | `4766ae1` |
| h1 | `fold` split out of the 5,312-byte `ThreatEntry` into its own **sized** `Box<[[Fold;2]; N]>` | **−19.80** | **−14.24** | n/r | `5767e20` |
| | **total, bench16 frame** | **−98.98 i/n (−1.556%)** | **−25.82 b/n (−3.94%)** | **−36.13** | |

Confirmed end-to-end by a round-robin panel of the whole landed tree against
the untouched baseline and a byte-identical rebuild of it, all three cycled
together (n=9):

```
INSTR   live NULL  +0.0001 i/n     ARM  -98.9761 i/n   (6363.4255 -> 6264.4494)
BRANCH  live NULL  +0.0001 b/n     ARM  -25.8173 b/n   ( 655.2607 ->  629.4434)
L1LOAD  live NULL  +7.0532 l/n     ARM  -36.1297 l/n   (3346.2569 -> 3310.1272)
CYCLES  live NULL   -6.167%        ARM   -0.233%       (null spread -18.9% .. +25.8%)
```

The arm is ~10^6× the instruction null and ~5× the L1-load null. **The cycles
null is larger than the cycles "effect", and in the same direction** — final
confirmation that the axis is dead on this box today (§0.2).

`6363.4254 → 6264.4493 i/n`. **Node count 9,444,932 and eval hash
`5ca978e0…8b38` identical on every line, arms and rejects alike.** `Bench:`
stays 587683 — re-verified against a from-source default release build, and
structurally safe because `Cargo.toml` has no `default = [...]`, so the pinned
`bench 16 1 6 default depth` never compiles `sota_threats.rs` at all.

**Frame:** `bench 16 1 16 default depth engines/canonical/net.cenn`, 9,444,932
nodes, thin-LTO / 4 CGU / `-C target-cpu=native`, non-PGO, `--features
sota,arch-d48x`, pinned to one core with `taskset`.

### 0.2 The instrument finding, which is the more important half

**The cycles axis failed its own null and cannot resolve anything this study
did.** This was found the hard way, corroborated independently by study 22 and
by the coordinator, and it forced a retraction of arm r1's own commit headline
(§2.4).

A three-way **round-robin** panel — base / a byte-identical rebuild of base /
the arm, all three cycled *together*, order rotating each round:

```
CYCLES  live NULL  median -0.586%   spread  -8.921% .. +9.434%
CYCLES  ARM        median -0.878%   spread -15.864% .. +8.990%
INSTR   live NULL  -0.0006 i/n      ARM -20.5061 i/n
BRANCH  live NULL  -0.0005 b/n      ARM  -3.3406 b/n
```

The null and the arm are indistinguishable on cycles, and the null's own spread
is 18 points. `instructions:u` and `branches:u` are **exact**, and the arm is
**34,000×** the instruction null. Those are the axes this study gates on — which
is also the right call on the merits: study 19 measured `at_full` at only
**4.69% of cache misses**, so this block is instruction-dominated.

### 0.3 What did not land

| arm | what | verdict |
|---|---|---|
| r2 | resolve BOTH perspectives before touching EITHER PSQ1 row | **rejected**, +25.88 i/n |
| b1 | `Board::by_type` — maintain the type-major occupancy mirror | **not landed**, −2.92 i/n only |
| p1 | kill `Square::try_offset` at the four pawn-push sites | **rejected**, +41.31 i/n |
| s1 | hoist `Vec::push`'s capacity test above the 168-byte `StateInfo` read | **rejected**, +17.07 L1 loads/n |

---

## 1. METHOD

### 1.1 NULL FIRST — and then the null moved

The null panel ran before any arm was written, as required. It passed, and
**that turned out to mean nothing**, because it was a null of a *different
machine state* than the arms later ran in. The sequence is worth recording in
full, because the failure is not obvious from any single reading:

| when | what | result |
|---|---|---|
| open | `instructions:u`, three bit-identical binaries, sequential | **6363.4254 / 6363.4256 / 6363.4255** — exact, spread 0.0002 i/n |
| open | `cycles:u`, same three, **sequential** | 4421 / 4493 / 4595 / 4708 c/n — **+6.5% drift, unusable** |
| open | `cycles:u`, **paired + `taskset`**, n=6, three replicates | −0.357% / +0.024% / −0.546% — looked fine |
| open | same pairing at **n=10** | **+1.384%** — *further* from unity at higher n |
| mid | fresh pairing null, n=8, box now busier | **−2.431% / −3.802%** |
| late | **round-robin** null (interleaved with the arm) | **−0.586%, spread −8.9% .. +9.4%** |
| end | **round-robin** null, whole-tree panel, n=9 | **−6.167%, spread −18.9% .. +25.8%** |

The last row is the clearest statement available: a **44-point** spread, and a
median further from unity than any arm this study measured — on two binaries
built from byte-identical source.

Every one of those is the same two binaries. The 4th row is study 19's rule 3
("a null at a different n is not a null") reproducing exactly; the 5th and 6th
are the stronger statement that **a null at a different TIME is not a null
either**. Corroboration arrived from two directions while this was being
chased: study 22 measured four same-binary nulls at −3.40 / −0.07 / −1.51 /
+1.42 percent in one session and retracted a committed −1.16% headline over it,
and the coordinator's verification panel came back at a 35.8% spread.

**The rule this study would add to the list:** *the null must be INTERLEAVED
WITH the arm, in the same rounds, not merely run before it.* Blocked designs —
null block, then arm block — measure the drift between blocks and report it as
the effect. `tools/perf24/panel3-24.sh` is the shape.

### 1.2 What the axes are actually worth on this box

| axis | null | usable? |
|---|---:|---|
| `instructions:u` at fixed nodes | **0.0000%** (−0.0006 i/n live) | **yes — exact** |
| `branches:u` at fixed nodes | **0.0000%** (−0.0005 b/n live) | **yes — exact** |
| `L1-dcache-loads:u` | +7.1 … +12.2 l/n live (±0.19–0.37%) | only for deltas ≫ 12 l/n |
| `cycles:u` | −0.59% … −6.17% median, up to a 44-point spread | **no** |

Both `instructions:u` and `branches:u` are architectural events that are
functions of the executed instruction stream alone, so at a fixed node count
they are deterministic to the digit. That is what makes them immune to the
contention (a live 400-superbatch trainer, three sibling perf studies building)
that destroys the cycles axis. `L1-dcache-loads` is *nearly* architectural but
picked up ~12 loads/node of live wander, so it is quoted only where the effect
clears that by a wide margin, and marked **n/r** (not resolved) otherwise.

### 1.3 Instruments (`tools/perf24/`)

* `build24.sh` — the gated build. Refuses below 3500 MB available, sleeps 120 s
  and rechecks; every build inside `systemd-run --user --scope -p MemoryMax=3G
  -p MemorySwapMax=0` through `tools/oomlast.sh`, `-j2`, into a stashed
  `perf24/bin/<label>`. **Nothing was OOM-killed and the trainer was never
  touched.**
* `probe24.sh` — one run, four counters, plus the node total and the eval hash.
  The two identity checks are printed on every line, so no arm can be read
  without them.
* `panel3-24.sh` — **the round-robin panel** of §1.1. Median only, never
  min-of-N (study 22 measured min-of-N at −2.07% on two identical binaries; this
  study's own "clean-half median" estimator was abandoned for the same reason).
* `pair24.sh` — the earlier two-way paired form, kept because §2's history is
  evidence. It now counterbalances round order; that was still not enough.
* `selfprof24.sh` — the ranking self-profile.
* `stop24.sh` — kill by PID with an explicit `shell-snapshots` filter, never
  `pkill -f`.

---

## 2. RANKING ON OUR OWN PROFILE

Per the commission, no Stockfish diff was used to rank. Two self-profiles of
the baseline on the bench16 frame:

| symbol (mine) | % cycles | % instructions |
|---|---:|---:|
| `ThreatContext::at_full` | 7.10 | **8.66** |
| `Board::make_move_with` | 1.61 | 2.85 |
| `ThreatContext::apply_at` | 1.22 | 2.18 |
| `Board::see_ge` | 1.41 | 1.68 |
| `record_square_touch` | 1.34 | 1.58 |
| `Board::attackers_to` | 1.08 | 1.52 |
| `Board::unmake_move` | 0.65 | 1.11 |
| `Board::ensure_threats` | 1.06 | 1.05 |
| `movegen::generate::<0>` + `<1>` | 1.40 | 1.86 |
| `ripple` / `record_targets::<31>` / `<15>` | 1.53 | 1.88 |
| **total in this study's scope** | **~20.1%** | **~26.0%** |

### 2.1 Splitting `at_full` on the CYCLE axis — new, and it named the target

Study 19 split `at_full` in a callgrind/instruction frame. Splitting it here
with `#[inline(never)]` on `build_deltas` / `resolve_both` / `resolve_into` /
`record_mutate` / `record_outgoing` / `record_incoming_mutate`, on the bench16
frame, gives the **cycle** split for the first time:

| stage | % of ALL cycles |
|---|---:|
| **`resolve_both`** | **4.43** |
| `record_square_touch` | 1.46 |
| `build_deltas` | 1.43 |
| `apply_at` | 1.05 |
| `at_full` (compute + fold + psqt) | 0.85 |
| `ripple` | 0.71 |
| `record_incoming_mutate` | 0.61 |
| `record_targets::<31>` + `<15>` | 0.71 |
| `record_outgoing` | 0.17 |
| **sum** | **11.42** |

The unsplit total was 11.19, so the outlining tax is ≈ +0.2 points — small
enough that the split is readable.

**`resolve_both` is 4.43% of all engine cycles — 39% of the whole threat
block.** Study 19 had ruled it off-limits on the grounds that it is 1.076× SF
per unit of work. The commission's instruction to rank on our own profile
rather than on a Stockfish diff is what put it back on the table, and it is
where the first arm landed. *4.43% of our cycles is 4.43% of our cycles whether
or not somebody else pays the same.*

---

## 3. WHAT LANDED

### 3.1 r1 — `resolve_both` unswitched, −20.93 i/n

`resolve_into` has been hand-unswitched on `psqt_table.is_empty()` since perf
study 8. **`resolve_both` — which handles the common case, both perspectives
folding in one pass over the event buffer — never was.** So a loop-invariant
slice-emptiness test was re-evaluated once per resolved index per perspective,
and because this loop is register-starved the value had been *spilled*: the
disassembly carries four copies of `cmpq $0x0,0xd8(%rsp)` inside the loop body,
each a stack reload. The arm also hoists the `as_chunks::<BUCKETS>()` row view
out of the event loop.

### 3.2 d1 — `#[repr(align(8))]` on `ThreatDelta`, −37.73 i/n

The largest single win of the study, and it is one attribute. Five `u8`-sized
fields give **size 7, align 1**. At a 7-byte stride roughly one event in nine
straddles a 64-byte line, every field access is unaligned, the five stores in
`push_kept` cannot be merged, and the resolver walks the buffer with `add $0x7`
rather than a scaled index. The buffer grows 1792 → 2048 bytes and never leaves
L1.

### 3.3 k1 — `king_cross` without the `Option` arrays, −20.51 i/n

It built `from: [Option<Square>; 2]` and `to: [Option<Square>; 2]`, filled them
in two passes and paired them in a third — eight bytes of stack zeroed to
`None`, up to four conditional stores and two `Option` destructurings **on every
made move**, to answer a question that is false for the ~95% of moves that do
not move a king. It is visible in `apply_at`'s disassembly as a 2-byte-strided
stack array (`mov %cl,0x1(%rsp,%r8,2)`). A move relocates one piece, so a king
appears at most once per list and the pairing needs no storage at all.

### 3.4 h1 — `fold` out of the 5,312-byte entry, −19.80 i/n and −14.24 b/n

Study 22's handoff item, and its measured trap heeded. `fold: [Fold; 2]` was
the last two bytes of a 5,312-byte struct, and `compute` scans it *backwards*
per perspective looking for the nearest non-`FromParent` ancestor — `d`
guaranteed-distinct cache-line touches 5 KB apart to read two bytes each, twice
per evaluated node. As its own array that walk is 32 plies to a line.

**Both arrays are sized `Box<[T; N]>`, not `Box<[T]>`.** Study 22 measured this
identical split against fat-pointer boxes at **+58.7 i/n** — a regression made
entirely of length loads and dynamic bounds checks — and at −9.3 i/n once every
array was sized. That shows up here as the branch column: **−14.24 b/n, four
times any other arm**, because the ancestor walk's per-ply bounds check is now
discharged at compile time. `dp`/`position` deliberately stay in `ThreatEntry`:
they are read once per folded child and never scanned, so a third base pointer
would only add register pressure to `compute`.

---

## 4. WHAT DID NOT LAND — four rejects, with the numbers

### 4.1 r2 — pair the two PSQ1 row loads. **+25.88 i/n. The beautiful story was wrong.**

`perf annotate` of the split build put **41% of `resolve_both`'s samples** on
the four instructions immediately *after* the `vpaddd`/`vpsubd` pairs — textbook
skid off the PSQ1 row load, a random 32-byte read from a 2.7 MB table reached
through a three-deep dependent load chain (`lut1` → `offsets` → `lut2`). The
inference was that the two perspectives' loads serialise, and that resolving
both indices before touching either row would overlap them.

It lost on **every** axis against the unswitch alone: +25.88 i/n (vs −20.93),
−4.16 b/n (vs −7.56), +3.14 l/n (vs −25.19). The out-of-order window already
overlaps those loads; forcing it by hand only costs register pressure. This is
rule 6 — *when the story gets beautiful, attack it hardest* — and the mechanism
that actually paid was the boring loop-invariant test sitting next to it.

### 4.2 b1 — `Board::by_type`. **−2.92 i/n. Not worth the invariant.**

Study 19's remainder item 6, never previously attempted: maintain a type-major
occupancy mirror so `pieces_of_type` is one load instead of two loads and an OR.
Implemented in full (three placement mutators, `attackers_to` batched, the
invariant added to `assert_consistent`). Exact axes: **−2.92 i/n, +0.35 b/n,
−14.29 l/n.**

Why so small: `perf annotate` of `apply_at` shows `vorps 0x170(%rcx),%zmm0,%zmm0`
— **LLVM already vectorises `ThreatPosition::from_board`'s six colour-major
unions into a single AVX-512 lane operation**, so the gross saving is ~7 i/n,
against ~+4 i/n for the extra read-modify-write on ~1.1 placement edits per
move. *That* is the answer to study 19 item 6, and it should not be re-derived.

It was first rejected on a +4% cycles reading. **That rationale is withdrawn**
— the axis is not resolvable (§0.2). A 48-byte layout null (a dead `[Bitboard;
6]` padding field, never read or written) priced the struct growth alone at
+27.49 i/n and +0.13…+0.57% cycles, so layout did not explain the cycles
reading either. The arm is not landed on the honest ground that **0.046% of
instructions does not pay for a fourth redundant placement representation to
keep in sync.**

### 4.3 p1 — kill `Square::try_offset` at the pawn-push sites. **+41.31 i/n.**

Study 19's remainder item 4, ranked "high confidence" there. The four pawn-push
sites ask `sq.shifted(color.pawn_push())` and then `.contains()`, which is
`try_offset`: a range test, a `Square::new` bounds assert, a **Chebyshev
wrap test** and an `Option<Square>` — all to compute ±8, the one direction that
cannot wrap a file. Replacing it with `Bitboard::shift` + `.lsb()` *removes*
about ten ALU ops per site and still measured **+41.31 i/n**. The most likely
cause is `lsb()`'s own `Square::new` assert re-introducing a panic edge per
site, plus shifted inlining decisions; it was not chased further because the
exact axis is unambiguous about the outcome. **Study 19 item 4's first half is
refuted as spelled.**

### 4.4 s1 — hoist `Vec::push`'s capacity test. **+17.07 L1 loads/n.**

A real defect, correctly diagnosed, not fixed by this spelling. `perf annotate`
of `make_move_with` shows `self.history.push(self.state)` copying the 168-byte
`StateInfo` to the **stack** (three `vmovups %zmmN,0x??(%rsp)`), then the
never-taken `call` to the grow path, then three reloads and three stores into
the buffer — and the two reload instructions carry **33% + 39% of the symbol's
sampled instructions**. Cause: `Vec::push` takes its argument by value, 168
bytes is a memory-ABI argument, and the reallocation call sits between the
materialisation and the slot write, so the value must survive it.

Hoisting the identical capacity test above the push — hoping LLVM would prove
the inner test false on the fall-through path and delete the call edge — did
**not** work: −0.485 i/n, +17.07 l/n. The mechanism is still there and still
worth ~336 bytes of L1 traffic per made move; §6 item 1 says what to try next.

---

## 5. THE PREDICTION, TWO-SIDED

Pre-registered before any PGO build.

**Claim.** The four arms remove 98.98 instructions and 25.82 branches per node
from a 6363.4 i/n baseline (−1.556% / −3.94%), with no change to the executed
tree.

**Prediction.** On a PGO build measured with a properly interleaved instrument
on a quiet box, the four arms together are worth **−1.0% to −1.8% wall clock**
(NPS +1.0% to +1.8%). The floor is set by the instruction reduction being real
and exact; the ceiling by IPC not improving, which it plausibly does slightly
given −25.82 branches/node and the h1 locality change.

**Kill condition.** If a quiet-box, PGO, interleaved measurement at n ≥ 20 puts
the four-arm tree at **worse than −0.3% wall clock**, the claim that instruction
count is the right proxy for this block is wrong, and the correct conclusion is
that `at_full`'s 4.69%-of-misses figure (study 19) understates its stall
exposure — in which case §6 items 2 and 3 should be re-ranked above everything
instruction-shaped.

**What would NOT refute it:** any cycles reading taken on this box today, in
either direction, at any n. §0.2.

---

## 6. RANKED REMAINDER — handoff to study 25

**Do not reopen:** arm r2 (§4.1); study 19 item 4's pawn-push half (§4.3);
study 19 item 6 / `Board::by_type` (§4.2 — the reason is now measured, not
guessed); the PSQ1/weight-row software prefetch, which is now pre-refuted three
times (study 4 at −2.5% NPS, study 22 twice, and §4.1's failure of the same
mechanism by a different route).

| # | item | size | confidence | next test |
|---|---|---:|---|---|
| 1 | **`make_move_with`'s 168-byte double copy — diagnosed, not fixed.** §4.4: `self.history.push(self.state)` spills all 168 bytes to the stack and reloads them across the never-taken reallocation call, and the two reload instructions carry 72% of the symbol's sampled instructions. The hoisted-capacity-test spelling failed. **Next spelling: make `history` a preallocated slab (`Vec<StateInfo>` kept at full length plus a `hist_len` cursor), so a push is an indexed store with a bounds check and NO call edge.** All 12 `history` uses are inside `board.rs`; the two capacity tests at `board.rs:2541-2559` need updating with it | ~336 B of L1 traffic/node | **high on the mechanism, unproven on the fix** | D24 perft ritual required; watch `Board::clone` (a slab clones its whole length) |
| 2 | **`resolve_both` is still 4.43% of all cycles after r1** — the largest single item in this scope. The per-event work that remains: two `sf_code` re-encodes from `(Color, PieceType)` pairs (six `movzbl` plus a `shl`/`inc`/`or` triple, visible in the asm), and a `usize` index list where `u32` would do | ~40–80 i/n guessed | medium | **store `sf_code` PRE-ENCODED in `ThreatDelta`** (the attacker is a loop invariant at almost every recording site, so the encode hoists out of the emitting loop where the four loads it replaces could not). This study attempted it, corrupted the file with a regex rewrite and reverted; **do it with hand edits.** The `u32` half is blocked on `apply_i8_deltas_into`'s `&[usize]` signature in `sota_simd.rs` |
| 3 | **`build_deltas` is 1.43% of cycles for pure orchestration** and has never been split. Two slice pattern-matches and a 74-byte `ThreatPosition` copy per call at 0.62 calls/node | unknown | medium | split it the way §2.1 split `at_full` — the outlining tax there was only +0.2 points |
| 4 | **`Board::see_ge` 1.68% of instructions and `Board::attackers_to` 1.52%, neither ever studied.** `see_ge`'s least-valuable-attacker search is a 6-iteration `find_map` with an `expect` panic edge, run once per exchange step | unknown | low–medium | they are in no study's remainder because nine studies ranked by Stockfish diff and these have exact counterparts; rank them on our own share |
| 5 | **`Board::ensure_threats` 1.05%** — `compute_threats` runs four separate per-piece loops with magic lookups on every position whose Wave-G map is first read | unknown | low | measure the fill RATE first; most of the symbol may be the guard, not the fill |
| 6 | **`record_square_touch`'s two magic lookups.** The hottest single instruction in it is the magic table load at 11.3% of the symbol. `qatt` is only ever consumed inside `ripple`, where a rook slider only needs `ratt` and a bishop slider only `batt` — so the union could move into the (rarer) queen arm of section 1 | ~1 i/n | high on correctness, low on size | cheap; bundle it with something else |

### 6.1 Three method items handed over

1. **INTERLEAVE THE NULL WITH THE ARM.** §1.1. Six nulls of the same two
   binaries, spanning +1.38% to −3.80%, plus a round-robin null at −0.59% with
   an 18-point spread. A blocked design measures the drift between blocks.
2. **`branches:u` is a second exact axis and it is not being used.** It is
   architectural, deterministic at fixed nodes (live null −0.0005 b/n), and it
   separates arms that `instructions:u` cannot: arm h1 is −19.80 i/n like the
   others but **−14.24 b/n**, four times any of them, which is precisely the
   bounds-check story study 22 predicted. Quote both.
3. **`git stash` IS SHARED ACROSS WORKTREES.** `refs/stash` is a repository ref,
   not a per-worktree one, so `git stash` in an agent worktree is visible to —
   and poppable by — every concurrent sibling. This study popped a sibling's
   stash by accident, applied its `movepick.rs`/`search/mod.rs` changes into the
   wrong tree, and had to restore the entry with `git stash store <sha>`.
   **With sibling agents live, use `git diff > file` and `git checkout`, never
   `git stash`.**

---

## 7. WHAT IS OWED — gates

Green on HEAD (`5767e20`):

| check | result |
|---|---|
| `cargo fmt --check` | clean |
| `cargo clippy --all-targets -- -D warnings` | clean |
| `cargo clippy --all-targets --features sota,arch-d48x` | no diagnostic in any file this study touched (7 pre-existing `-D warnings` failures on the baseline, all in study-21-owned files) |
| `cargo test` | 608 passed, 0 failed |
| `cargo test --features sota,arch-d48x` | 699 passed, 0 failed, 8 ignored |
| **`cargo test --release -- --ignored perft_full_suite`** (D24, mandatory for this scope) | **1 passed** |
| default-build `bench 16 1 6 default depth` | **587683**, re-verified from source |
| node identity, every arm and reject | 9,444,932 |
| eval-hash identity, every arm and reject | `5ca978e0…8b38` |

`CHECK_NNUE_GATE=1 bash tools/check.sh` is the owner's to run; all three pins
(587683 / `5ca978e0…8b38` / 1922947) are expected unchanged, and the eval hash
was verified directly on every measured binary.

---

## 8. THREATS TO VALIDITY

1. **Every number here is from a non-PGO thin-LTO build.** The shipping build is
   fat-LTO PGO. Instruction counts move under PGO; the sign of a pure deletion
   should not, but arm d1 (a layout change) and arm h1 (an addressing change)
   are the two most likely to differ, and neither has been re-measured there.
2. **One frame only.** All arms are measured on bench16. Study 19 §8.0a showed
   two self-profiles of one binary disagreeing 3.5× on `at_full` purely from
   workload skew, so a per-arm size could move on a different mix — though the
   *sign* of an exact instruction deletion cannot.
3. **No wall-clock or game evidence.** §5 is a prediction, not a result. The
   cycles axis was unusable and no match was run.
4. **Arm r2's rejection is on the exact axes only.** If its MLP hypothesis is
   right and the cost is purely register pressure, a PGO build with different
   allocation could flip it. Not re-tested.
5. **`L1-dcache-loads` is quoted for r1/d1/b1 but its live null was later
   measured at +12.2 l/n.** r1's −25.19 and d1's −19.34 clear that; b1's −14.29
   only marginally. Treated as directional, never load-bearing.
6. **Arm h1 changes an allocation shape** (`Box<[T]>` → `Box<[T; N]>` plus a
   second box). The `try_into` conversions are pointer casts, but the
   `unreachable!` arms are untested by construction.
