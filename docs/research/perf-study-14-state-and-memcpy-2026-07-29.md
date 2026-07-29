# ARCHON perf study #14 — the state stack, and the memcpy nobody had traced

Date: 2026-07-29. Fourteenth commissioned study (12: `perf-study-12-fold-tile-2026-07-29.md`,
13: `perf-study-13-panic-edges-2026-07-29.md`). Raw artifacts and instruments:
`perf-artifacts-14/`. Baseline: `main` at `d6cb5d7` (study 13 merged and gated).
Comparison target: **`engines/bin/stockfish-eca43a97`**.

---

## 0. HEADLINE

### 0.1 The ratio moved

`bench 16 1 300000 tools/nnue/calib/wl50.fens nodes`, two arms interleaved,
`taskset -c 4`, `nice -n 19`, 5 rounds, both PGO. Raw:
`counters-headline.txt`.

| per node | **ARCHON study 14** | ARCHON study 13 | **SF `eca43a97`** |
|---|---:|---:|---:|
| **instructions** | **5986.8** | 6081.2 | **4452.4** |
| **ratio vs `eca43a97`** | **1.3446×** | 1.3658× | — |
| branches | **631.3** | 652.4 | 362.7 |
| branch-misses | 20.8 | 20.8 | 19.7 |
| nodes | 15,001,568 | 15,001,568 | 15,008,494 |
| `instructions:u` spread | 0.0207% | 0.0003% | 0.0000% |

**Stockfish reproduces 4452.4 with 0.0000% spread** — the third study in a row
to get that exact value from the same control, so studies 12, 13 and 14 are all
in one frame and the three ARCHON numbers are directly comparable. The node
count is identical to both predecessors' to the unit.

**1.3658× → 1.3446×. −94.4 i/n, 5.8% of the 1,628.8 i/n gap, in five
node-identical changes.** Branches fell 21.1 per node. **1,534.4 i/n remain.**

### 0.2 What landed

Every arm is node-identical (5,001,562 on the probe tree, **15,001,568 on the
headline bench — the same count studies 12 and 13 measured**) and
eval-hash-identical (`5ca978e012da1bb0`). All three gate pins reproduce
unchanged: `Bench: 587683`, eval hash `5ca978e0…0b38`, d13 nodes `1922947`.

| # | change | probe Δ | where |
|---|---|---:|---|
| 1 | **the live `StateInfo` moved out of the stack and into a `Board` field** | **−36.7** | `board.rs` |
| 2 | threat fold's empty-delta parent copy: a libc `memcpy` call → chunked inline | **−22.7** | `sota_threats.rs` |
| 3 | **`remove_piece_known` / `move_piece_known` — stop re-reading a piece the caller holds** | **−40.3** | `board.rs` |
| 4 | **`StateInfo::moved` — `unmake_move` stops re-reading the mailbox too** | (see §4.4) | `board.rs` |
| 5 | `check_squares_all` returns `&[Bitboard; 6]` | −0.8 | `board.rs`, `movepick.rs` |
| | **total, probe frame** | **−107.2** | |
| | **total, PGO frame (the number that counts)** | **−94.4** | |

A sixth arm — a redundant `type_occupancy` transpose (Stockfish's `byTypeBB`) —
measured **−3.6** and was **reverted after the D33 review**; §4.6 says why, and
it is the most useful negative result in the study.

The probe overstated the PGO delta by 1.14×, consistent with study 12's
2.0× and study 13's 1.12×, and further evidence for study 12 §0.3's standing
rule: **no probe delta may be quoted as a result.**

### 0.3 What this study is actually about

Two things, and neither is the ratio.

**(a) The unsymbolised libc block from studies 12 and 13 has a name.** Study 13
§7 left "~68 i/n spread over four unsymbolised addresses in one 16-byte range,
almost certainly a libc `memmove`" as its cheapest completely-unexplored item,
with "nobody has established what is calling it." It is
`ThreatContext::at_full` and `apply_at`, copying **2560-byte threat
accumulators**, and the instrument that found it took four minutes to write:
an `LD_PRELOAD` interposer that records every `memcpy`/`memmove` caller and
size (§3). 52,255 calls over 150,059 nodes, 133.8 MB — **891 bytes per node of
`memcpy` traffic in a search.** That is a deterministic workload fact, not a
sample.

**(b) The same mechanism, two spellings, opposite signs — 45.4 i/n apart.**
Removing that `memcpy` by *deleting* the empty-delta special case and letting
the fold kernel do the copy measured **+22.7** (a regression). Removing it by
*keeping* the special case and chunking the copy by hand measured **−22.7**.
Both are bit-identical to the original and to each other. Study 13's rule was
"price the arm, never the mechanism"; study 14's amendment is that **the arm is
not the mechanism plus a free choice of spelling — the spelling *is* the arm.**

---

## 1. METHOD

Studies 8–13's rules were followed. Two additions.

### 1.1 NULL FIRST — and this study's first null FAILED

**Null A — rebuild determinism.** The same source, forced to recompile, built
and measured twice:

```
null-1           32,509,246,605   6499.8 i/n
null-2-rebuild   32,503,673,801   6498.7 i/n     spread 0.017%, ±1.1 i/n
```

Wider than study 13's ±0.015 (which rebuilt without touching a file, so cargo
had nothing to do). ±1.1 i/n is the honest rebuild band for this instrument.

`null-1` reproduced study 13's final published probe line — **6499.8 against its
6499.8** — which is a cross-study control study 13 did not have.

**Null B — SOURCE PERTURBATION, and it is the study's first real finding.**
Study 13 mandated an inert re-spelling on top of the rebuild null. The obvious
inert re-spelling of the function this study was about to change is

```rust
fn st(&self) -> &StateInfo { let n = self.states.len(); &self.states[n - 1] }
```

instead of `self.states.last().expect(...)`. Same loads, same panic-on-empty
behaviour, same value.

**It measured +86.4 i/n** (6499.8 → 6586.2), node- and eval-identical.

So the null failed, and it failed *informatively*: before writing a line of the
real fix, the instrument had already shown that `st()`'s **spelling alone** is
worth 86 i/n in the wrong direction — i.e. that the ranked-#1 item was real and
large. A second, genuinely inert spelling (a reborrow through a `&[StateInfo]`
local, same `last().expect()` body) was then run as the actual Null B:

```
a0b-NULLB-st-reborrow   32,509,253,617   6499.8 i/n     ±0.0014 i/n vs null-1
```

**±0.0014 i/n.** The instrument survives its own null; every arm below is a real
effect of that arm.

*The lesson to carry forward: a "semantically identical" re-spelling is only a
valid null if it is also structurally identical. `Vec::last()` and
`&v[len - 1]` are the same function and not the same code.*

### 1.2 Instruments

* **`probe14fast.sh`** — screening, inherited from `probe13fast.sh`.
  `CARGO_PROFILE_RELEASE_LTO=thin` / `CODEGEN_UNITS=4`, native, non-PGO, fixed
  5,001,562-node tree; prints the wl50 eval hash and node count so a tree or
  eval change cannot hide inside a speed claim.
* **`probe14lm.sh`** — the same instrument at `-j 1` inside a
  `systemd-run --user --scope -p MemoryMax=2G` cgroup. See §1.4.
* **`pgo14.sh` + `pstat8.sh`** — the deliverable frame.
* **`memspy.c`** — the `LD_PRELOAD` `memcpy`/`memmove` interposer (§3). New.
* **`flat8.sh`** — inherited instruction-attributed flat profiles.

### 1.3 Frame control for the low-memory probe

`probe14lm.sh` differs from `probe14fast.sh` only in `CARGO_BUILD_JOBS` and the
cgroup. **`CARGO_BUILD_JOBS` is not a cargo fingerprint input**, so the two
cannot produce different binaries — and that was verified rather than assumed:
the final tree was built by `probe14lm.sh`, sha256'd, re-run through
`probe14fast.sh`, and sha256'd again.

```
sha256(bin) before   b2bf3838b6c6da1a
CONTROL-j2-frame     31,988,472,998   6395.7 i/n
sha256(bin) after    b2bf3838b6c6da1a
```

Identical binary, and 6395.7 reproduced to the digit (a rerun null of ±0.002
i/n). Arms a1 and a2b therefore sit in the same frame despite being measured by
different scripts.

### 1.4 The box, and a build discipline change

A D48Y training run held **5.3 GB RSS** for this whole session and available
memory sat at 3.5 GB for a stretch of it — below `probe13fast.sh`'s 4500 MB
floor, which stalled the study. A `cargo test` run was **OOM-killed (SIGKILL)**
during arm 3.

The fix is not a lower `free` threshold, which can race. Every build in this
study after that point ran inside
`systemd-run --user --scope -p MemoryMax=<N> -p MemorySwapMax=0`, so a runaway
build is killed **inside its own cgroup** instead of triggering the global OOM
killer that took out a training run on 2026-07-27. Tests likewise ran capped and
at `--test-threads=3`. **Study 15 should use the cgroup as the primary guard and
the `free` check only as a courtesy.**

---

## 2. THE #1 ITEM, CLOSED — `Board::st()` (−36.7)

Study 13 ranked `Board::st()` first (29.07 calls/node, counted) and could not
land it: the fix it proposed was `Box<[StateInfo; CAP]>` + a masked cursor,
blocked on `make_move_with`'s `mem::take` borrow trick, and it budgeted 1–2
hours for the refactor.

**We did not do that fix.** The masked-array fix makes `st()` cheap;
this one makes it **free**:

```rust
pub struct Board {
    …
    /// The CURRENT ply's state, held INLINE rather than as the top of a stack.
    state: StateInfo,
    /// Everything BELOW `state`, oldest first.
    history: Vec<StateInfo>,
}

fn st(&self) -> &StateInfo { &self.state }
```

`st()` was
`self.states.last().expect("state stack never empty")`: a data-pointer load, a
length load, an emptiness test, a panic edge, a decrement and an `imul $0xa8`
(168 is not a power of two) before any field is read. It is now a constant
offset that the consuming load folds into its addressing mode — **zero
instructions**, at 29 calls per node.

Three further things fell out of the change rather than being aimed at:

* **`mem::take(&mut self.states)` is gone.** It existed only so a `&mut` into
  the vector's tail could coexist with `&mut self` placement calls. With the
  live state as a field, every write is `self.state.copied.x = …` — a
  short-lived borrow the compiler is happy with, and disjoint-field borrows
  handle the rest. This is what perf study #2 option E and study 9 item 3 were
  both working around; **study 14 removed the borrow instead of the workaround.**
* **`states.push(StateInfo::EMPTY)` is gone** — 168 bytes of constant store
  (three `zmm` load/store pairs) that were then fully overwritten — and so is
  `st.copied = ancestors[child_idx - 1].copied`, because the carried block is
  already in place. What replaces both is `self.history.push(self.state)`.
* **The "never empty" invariant stopped being an assertion.** It is structural:
  there is no way to represent a `Board` without a current state. Two `expect`s
  and their panic edges disappeared with it. That is the idiomatic Rust move —
  make the illegal state unrepresentable — and here it is also the fast one.

`set_check_info(&self, st: &mut StateInfo)` could not be called as
`self.set_check_info(&mut self.state)`, so it was split: a by-value
`CheckInfo::of(&Board) -> CheckInfo` query plus a `store(&mut StateInfo)` write,
with `refresh_check_info(&mut self)` composing them for the hot paths. That is
study 13 §8's option (b), and it is the idiomatic answer to a partial-borrow
conflict — compute into a value, then write.

**What it costs, against the reference, exactly.** This is where the D33
reference review (persona B) earned its keep, and the honest framing is *not*
"we converged on Stockfish":

| bytes of state copied | make | unmake | null make | null unmake |
|---|---:|---:|---:|---:|
| **SF `eca43a97`** | **64** | **0** | 192 | **0** |
| ARCHON before | 168 const store + 64 copy | ~0 | 168 + 168 | 0 |
| **ARCHON after** | **168** | **168** | 168 | 168 |

SF's `do_move` copies exactly `offsetof(StateInfo, key)` = 64 bytes
(`eca43a97:src/position.cpp:823`) — the carried prefix, which is byte-for-byte
our `CopiedState` — then does two pointer stores. Its `undo_move` state restore
is the whole of `st = st->previous` (`eca43a97:src/position.cpp:1120`): **zero
bytes.** Both verified by reading the pinned revision, not the checkout.

So our make got ~64 B cheaper and our unmake went from free to a full 168-byte
restore — **net +104 B/node of state traffic, which is new work with no
counterpart in the reference.** We bought it with 29.07 `Vec::last().expect(…)`
sequences per node, and on *reads* we are now better than SF (a constant field
offset against SF's `st` pointer indirection). It is a different point on the
same trade, not a convergence; the old shape was the worst of both. The probe
says the trade nets **−36.7 i/n and −13.2 branches per node**, so it is
empirically right, and §8 item 1 records the shape that might get both halves.

### 2.1 The index mapping, stated once

Before the push, `history.len() == n` and the live state is at ply index `n`.
After `history.push(self.state)`, `history.len() == n + 1` and the child is at
ply index `n + 1`, so the ancestor `i` plies back is `history[history.len() - i]`
— which is exactly the old `states[child_idx - i]`, because `history[j] ==
states[j]` for every `j` below the top. The three walks (`make_move_with` step
13, `has_repeated`, `upcoming_repetition`) all use that one mapping, and the
`assert_consistent` bound became `plies_from_null <= history.len()` (it was
`< states.len()`, and `states.len() == history.len() + 1`).

---

## 3. THE MEMCPY, TRACED — a four-minute instrument for a two-study-old question

Study 13 §7 could see ~68 i/n of libc code in the flat profile and could not
attribute it: `perf` has no call graph in a PGO/thin-LTO release build with no
frame pointers, and `--call-graph lbr` does not record on this box. Both flat
profiles this study took show the same anonymous cluster:

```
0.38%  [.] 0x0000000000189e76   libc.so.6
0.21%  [.] 0x0000000000189e68   libc.so.6
0.19%  [.] 0x0000000000189e7d   libc.so.6
…
```

Disassembling libc at that address identifies it beyond doubt — it is the
4×`zmm` 256-byte-per-iteration inner loop of `__memmove_evex_unaligned_erms`
(`vmovdqu64`/`vmovdqa64` ×4, `sub`, `cmp`, `ja`). But the *caller* is the
question, and the answer came from **interposing `memcpy`/`memmove` with
`LD_PRELOAD` and recording `__builtin_return_address(0)` minus the load base**
(`scripts/memspy.c`). Return addresses resolve against the binary's own
disassembly; call counts and byte totals are exact.

Baseline, 150,059 nodes (`memspy-baseline.txt`):

| calls | bytes | size | caller |
|---:|---:|---:|---|
| 23,726 | 60,738,560 | 2560 | `ThreatContext::at_full` +0x10ed |
| 23,177 | 59,333,120 | 2560 | `ThreatContext::at_full` +0x127b |
| 2,816 | 7,208,960 | 2560 | `ThreatContext::apply_at` +0x678 |
| 2,536 | 6,492,160 | 2560 | `ThreatContext::apply_at` +0x34c |
| 9,930 | 23,988 | 2.4 | `Worker::iterative_deepening` |
| 205 | 524,800 | 2560 | `HalfkaContext::current_full` |

2560 bytes is `[i16; ACC]` at ACC = 1280 — **a threat accumulator**. The two
`at_full` sites are the two perspectives of one source line:

```rust
if add_len == 0 && sub_len == 0 {
    entry.acc[p] = parent.acc[p];      // 2560 bytes -> libc memcpy CALL
    entry.psqt[p] = parent.psqt[p];
}
```

That is **0.313 calls per node** — 31% of all folds take the empty-delta path —
and the assignment is far past any inline-store threshold, so LLVM emits a call
to `memcpy` with `vzeroupper` in front of it. The comment above it said an empty
delta list "degenerates to the plain parent copy **without touching the
kernel**", which was true and was the wrong optimisation.

### 3.1 Two spellings of the same fix, 45.4 i/n apart

**Arm a2 — delete the special case.** `apply_i8_deltas_into` with empty add/sub
lists is bit-identically a copy (`value = src[lane]; dst[lane] = value`), and
its AVX-512 body is `20 zmm loads + 20 zmm stores` per 640-lane tile, ×2 tiles
= ~84 inline instructions with no call, no dispatch, no `vzeroupper`. The psqt
branch is `parent + 0`. The mechanism is airtight.

**+22.7 i/n.**

And the memcpys really were gone — the interposer was re-run on the arm's own
binary and total copied bytes fell from **215.8 MB to 95.7 MB**
(`memspy-after-a2.txt`), with both `at_full` rows absent. So this is not "the
change did not do what I thought"; it is "the change did exactly what I thought
and the engine got slower."

**Arm a2b — keep the special case, chunk the copy by hand.**

```rust
for (d, s) in entry.acc[p].chunks_exact_mut(32).zip(parent.acc[p].chunks_exact(32)) {
    d.copy_from_slice(s);
}
```

40 chunks of 64 bytes, each below the inline threshold, so LLVM emits 40 inline
moves and no call.

**−22.7 i/n.**

Same mechanism, same bit-identical result, **45.4 i/n between the two
spellings** — 20× this study's rebuild null and 16,000× its inert-respelling
null. Study 13's rule needs the amendment in §0.3: pricing "the arm" means
pricing *this spelling of* the arm, and a losing spelling does not refute the
mechanism.

*(Why a2 lost is not established. The kernel is an indirect
`kernel_dispatch!` call and pulling the empty case through it may be
re-materialising the dispatch on a path that had none; that is a hypothesis, it
was not measured, and per CLAUDE.md rule 4 it is written down as a hypothesis.)*

---

## 4. THE REST OF WHAT LANDED

*(§4.1 is §2 — the state stack. §4.2 is §3 — the memcpy.)*

### 4.3 The placement mutators re-derived a piece the caller was holding (−40.3)

`make_move_with` computes `pc` in step 3 (`piece_on(from)`), then in step 10
calls `self.move_piece(from, to)` — which **loads the mailbox at `from` again**,
tests the `Option` discriminant, and carries a panic edge
(`.expect("move_piece from empty square")`). The same is true of every
`remove_piece` call in make and unmake: the victim is in `captured`, the
castling rook is `Piece::new(us, Rook)` by construction, the promoted piece is
`m.promotion_piece()`, the unmade king is `Piece::new(us, King)`.

Two new primitives take the value the caller already has in a register:

```rust
pub(crate) fn remove_piece_known(&mut self, piece: Piece, sq: Square);
pub(crate) fn move_piece_known(&mut self, pc: Piece, from: Square, to: Square);
```

The `Option`-returning `remove_piece` and the querying `move_piece` remain, as
thin wrappers, for the callers that genuinely do not know.

**−40.3 i/n and −10.8 branches per node — the study's largest arm.** It also
made `Board::move_piece` disappear from the flat profile: it had been an
out-of-line 0.29% symbol (study 13 §8 item 5c), and once it is a wrapper over an
`#[inline]` primitive with no panic edge on the hot path it inlines everywhere.

This is worth contrasting with study 13 §5.3, which measured **+106.3** for
replacing `piece_on(sq).expect(...)` with a total `piece_on_occupied(sq)` in the
movepick scoring loops. Both arms "remove a redundant check on a piece lookup".
The difference: §5.3 kept the *load* and softened the *check*, on a value
feeding a history index; this one deletes the load entirely, on a value feeding
address arithmetic that the caller had already computed. **Deleting work beats
softening a check** — but that generalisation has exactly two data points, so
study 15 should treat it as a lead, not a law.

### 4.4 `StateInfo::moved` — the last mailbox read on the make/unmake path (D33 reviewer C)

§4.3 threaded the piece into every placement edit that had one to thread.
`unmake_move`'s normal-move case had none: the mover is whatever sits on `to`,
and nothing recorded it, so `move_piece(to, from)` still paid a mailbox load, an
`Option` test and a panic edge — once per node, the last such site on the path.

`StateInfo` gained a `moved: Piece` field, written in make's step 11 beside
`captured` and read by unmake. **It is free in memory**: `size_of::<StateInfo>()`
was 166 bytes used of 168 with padding, and it is still **168** — which matters
here more than usual, because §2 made every make and every unmake copy that
record. The size is now pinned with `==`, not bounded with `<=`, so the next
field addition has to be argued for rather than absorbed.

It also left `Board::move_piece` — the querying spelling — with **no callers at
all**, so it is deleted. The placement mutators are now `put_piece`,
`remove_piece` (still `Option`-returning, still used by the FEN/analysis paths)
and the two `*_known` primitives.

This arm was proposed by the D33 review, not by the study, and it displaced one
that the same review recommended dropping (§4.6).

### 4.5 `check_squares_all` by reference (−0.8)

Study 13 handoff item 5f. It returned `[Bitboard; 6]` **by value** — a 48-byte
copy — to a caller that indexes it at runtime anyway. With the live state now a
field, `&self.st().check_squares` is a borrow of a `Board` field. Smallest arm
here, listed for completeness and because it is strictly better code.

### 4.6 The arm that was REVERTED, and why that is the right call

A sixth arm added `type_occupancy: [Bitboard; 6]` — Stockfish's `byTypeBB` — as
a redundant transpose of `pieces[colour][type]`, maintained by the three
placement mutators. Our layout makes a colour query one load and a **type**
query two loads plus an `or`, and the engine asks type questions ~13 times per
node (`attackers_to` 5 at 0.97 calls/node, `see_ge` 3, and
`ThreatPosition::from_board` **6 per node, every node**). Predicted: ~+24
instructions/node of query savings against ~12 of maintenance.

**It measured −3.6** — real (3× the rebuild null), node- and hash-identical, and
**six times smaller than its own arithmetic**. It was landed, and then reverted
after the D33 review. Three reasons, in order of weight:

* **In the deliverable frame it is worth ~2.4 i/n — 0.04%.** Against that,
  study 13 measured ±40 i/n of arm-caused-but-not-mechanism-caused movement from
  single-site edits, and *this study's own §1.1* moved 86 i/n by re-spelling one
  function. The number cannot survive its own instrument.
* **The reference says the prediction was wrong, and why** (D33 persona B,
  verified at `eca43a97`). SF's `pieces(ROOK, QUEEN)` is *also* 2 loads + an
  `or` (`eca43a97:src/position.cpp:642-649`) — `byTypeBB` does not make a
  multi-type query one load. Our `attackers_to` already hoists `queens` and
  shares it across both slider terms, so most of the predicted saving did not
  exist. Meanwhile SF maintains **3** bitboards per placement edit
  (`eca43a97:src/position.h:377-417`) and this arm took us to **4** — keeping
  both transposes is strictly more maintenance than the reference we cited as
  justification.
* **The price is permanent**: a fifth redundant placement layer on the engine's
  most D24-critical struct, a new invariant every future placement primitive
  must remember, and a `design/04` §3.3 plus CLAUDE.md core-encoding amendment.

Study 13's rule is "price the arm, never the mechanism" — an arm that wins is
worth landing whether or not the story is true. **The amendment this study makes
is that the rule presupposes the win is resolvable by the instrument.** At 2.4
i/n against a demonstrated ±40–86 i/n site sensitivity, a4 is not a small win;
it is an unresolved measurement, and buying permanent structural complexity with
one is how a codebase accumulates unfalsifiable folklore.

**The version with a plausible mechanism was not run** and is §8 item 4: SF's
actual layout *replaces* `pieces[c][pt]` rather than adding to it. That removes
a layer and 96 bytes instead of adding a layer and 48, and it costs a load and
an `and` on the ~72 `pieces(c, pt)` call sites that movegen is built around — so
it could easily be a loss. It is a study-15 arm with its own null, not a rider
on this diff.

---

## 5. THE ARM LOG

`probes-fast.txt`, in order, probe frame.

```
null-1                       6499.8    baseline, main @ d6cb5d7 (= study 13's published 6499.8)
null-2-rebuild               6498.7     +-1.1   rebuild null
a0-NULLB-st-respell          6586.2    +86.4   FAILED null -> a finding (S1.1)
a0b-NULLB-st-reborrow        6499.8   +-0.001  the real source-perturbation null
a1-state-inline              6463.1    -36.7   LANDED (S2)
a2-fold-kernel-copies        6485.8    +22.7   rejected (S3.1)
a2b-chunked-inline-copy      6440.4    -22.7   LANDED (S3.1)
a3-known-piece-placement     6400.1    -40.3   LANDED (S4.3)
a4-type-occupancy            6396.5     -3.6   REVERTED after the D33 review (S4.6)
a5-check-squares-by-ref      6395.7     -0.8   LANDED (S4.5)
CONTROL-j2-frame             6395.7   +-0.002  frame control (S1.3)
--- D33 review round: a4 reverted, StateInfo::moved added, S7.1's fixes ---
a6-post-review               6392.6    -3.1   vs a5, i.e. -107.2 from null-1
```

The final line is the shipped tree. It is **3.1 i/n better than the pre-review
tree even though a4 (−3.6) was removed from it** — reviewer C's replacement arm
(§4.4) paid more than the arm it displaced, which is the cleanest argument
available that the review round was worth its cost.

Node count 5,001,562 and eval hash `5ca978e012da1bb0` on **every line**,
including the two rejected ones and the failed null.

---

## 6. THREATS TO VALIDITY

1. **Probe deltas overstate PGO deltas** — here by only 1.14× (−107.2 → −94.4),
   the tightest of the three studies.
   Every per-arm figure in §5 is probe-frame; only the total is PGO-measured,
   and no per-arm PGO attribution is claimed.
2. **The rebuild null is ±1.1 i/n**, not study 13's ±0.015 — that study's null
   did not force a recompile. Arm a5 (−0.8) is *inside* this band and is
   therefore not independently established; it is kept on the grounds that the
   code is strictly better (no 48-byte copy), not on its number.
3. **Per-arm probe deltas at the `st()` site carry a much wider band than the
   nulls suggest** (D33 reviewer C). §1.1's `a0` moved 86.4 i/n by re-spelling
   one accessor — 2.4× a1's own delta, in the opposite direction. `a0` is an arm,
   not a null (`&v[n-1]` and `v.last().expect(…)` emit different panic paths), so
   the ±0.0014 inert null still stands as the instrument's resolution. But the
   *site sensitivity* is real, it is study 13's ±40 at twice the amplitude, and
   it is why §4.6 reverted a 3.6 i/n arm rather than bank it.
4. **Why a2 lost is a hypothesis, not a measurement** (§3.1).
5. **Nothing here was Elo-tested**, and there is nothing to test: every landed
   arm is node-identical on three independent trees (5,001,562 probe /
   15,001,568 headline / 587683 gate) and eval-bit-identical, so the engine
   plays exactly the same moves.
6. **Box state**: a D48Y training run shared the box at 5.3 GB RSS for the whole
   session and OOM-killed one `cargo test`. Everything here rides on
   `instructions:u` over fixed-node trees and on deterministic call counts, both
   contention-immune; no wall-clock or NPS number is quoted as a claim.
7. **The memspy counts are from a 150,059-node bench**, a shorter tree than the
   probe's. Call *rates* per node are what is quoted; absolute totals are for
   that tree only.
8. **Two release tripwires were downgraded to `debug_assert`** by §4.3/§4.4:
   `unmake_move`'s `expect("king on the castling to-square")` and its promotion
   `remove_piece(to)`. Study 13 §5.4 was burned by exactly this move. The
   difference — and reviewer A and C independently reached it — is that
   `king_square`'s invariant was **input-derived** (an illegal-but-accepted FEN),
   whereas these are **internal**: they can only break if a caller unmakes a move
   it did not make, which no path in the engine or UCI can do. Recorded rather
   than waved away.

---

## 7. GATE

`gate-s14.log` (both runs concatenated).

* `bash tools/check.sh` — **`Bench: 587683`**, unchanged. The diff is entirely
  in `board.rs` (which the default-feature build compiles) plus
  `sota_threats.rs`, so this is a real node-identity check on arms 1, 3, 4 and 5.
* **`CHECK_NNUE_GATE=1 bash tools/check.sh` — MANDATORY here** (arm 2 is under
  `src/eval/nnue/`) and both pins reproduce:
  * eval hash `5ca978e012da1bb086799709c3ea5350cf6d49923c254f7975b6d780b0098b38`
  * d13 nodes **1922947**
* **The D24 perft ritual applies and passes** — this diff rewrites `make_move`,
  `unmake_move`, `make_null_move`, `unmake_null_move` and all three placement
  mutators, which is the most D24-critical surface in the engine.
  `cargo test --release -- --ignored perft_full_suite` → **`Nodes searched:
  1839507`**, inside check.sh.
* `cargo fmt --check`, `cargo clippy --all-targets -- -D warnings`, `cargo test`
  (686 lib tests, 0 failed), the two-run deterministic bench-perft fingerprint.

**And one gate that is stronger here than the pins.** Debug builds run
`Board::assert_consistent()` — the full invariant oracle, all six Zobrist keys
recomputed from scratch, every placement layer cross-checked — **at every node
of every make and unmake**. The 686-test suite exercises that oracle across the
whole fast perft tier and the checked mini-tier. For a refactor whose risk is
"the state stack got mis-indexed", that is the load-bearing check, and it is why
the state/history split can be asserted correct rather than merely
node-identical on one tree.

### 7.1 Each landed change has a test that can fail — and the D33 review found one that did not

| change | test | mutation that makes it fail |
|---|---|---|
| state/history split (a1) | `assert_consistent` at every debug node; `repetition_encoding_and_ply_relative_draw` pins the exact value `4` on `history[0]` (the `i == hist_len` boundary); `upcoming_repetition_matches_a_brute_force_legal_move_oracle` | `history[hist_len - i]` → `[hist_len - i - 1]` in step 13 |
| **`has_repeated`'s ancestor walk (a1)** | **`has_repeated_walks_past_the_current_state_into_history` — NEW, added by this review round** | `.take(0)`, or dropping the `.rev()` |
| `CheckInfo` array order (a1) | `perft::perft_checked`'s independent `gives_check` oracle, fast tier | swap two entries of the array literal |
| `CheckInfo::store` completeness (a1) | `assert_consistent`, now seeded from `StateInfo::EMPTY` | drop any one field from `store` — and it is additionally a *compile* error, because `store` destructures |
| chunked accumulator copy (a2b) | `sota_threats::threats_replay_matches_refresh` + the `CHECK_NNUE_GATE` eval hash / d13 pin | copy 31 lanes per chunk instead of 32 |
| `*_known` placement (a3, a4) | the `debug_assert_eq!(self.piece_on(sq), Some(piece))` in each primitive, fired at every node of the debug perft tier; the release perft suite | pass the wrong piece |
| `StateInfo` size (a4) | `state_info_is_copy_and_bounded`, now `== 168` not `<= 256` | add a field that does not fit the padding |
| `check_squares_all` by ref (a5) | `check_squares_all_agrees_with_the_per_type_accessor` | return a stale/EMPTY array |

**The one real hole the review found, and it is the class study 13's review
found too.** `has_repeated`'s ancestor walk — the entire semantic content of
a1's rewrite of that function from an index loop to `.rev().take(end - 4)` — was
**dead to the whole suite**. Every assertion in the tree reached its verdict
through the `self.state.repetition != 0` short-circuit or returned early on
`end < 4`, so `.take(0)` passed all 686 tests. `has_repeated` additionally has
no production caller, which makes tests its only guarantee.

`has_repeated_walks_past_the_current_state_into_history` closes it: it plays the
knight shuffle (so an ancestor carries `repetition == 4`), then one more
reversible move so the *current* state carries `0` and only the walk can answer
`true`, and it pins `end == 5` so the boundary is load-bearing. Mutation-verified
in both directions — `.take(0)` and the `.rev()` removal each make it fail, and
neither fails anything else. It does **not** catch `.take(end - 3)`, an over-scan
of exactly one ancestor; the test's own comment argues that case is
unobservable rather than untested, and label that an argument rather than a
measurement.

---

## 8. RANKED REMAINDER — handoff to study 15

**1,534.4 i/n remain**, of which ~327 is the 1280-vs-1024 accumulator width (a
retrain, not a code change — study 12 §11 item 9, unchanged) and, by study 13
§7, ~0 is the accumulator fold, the eval head, the threat generator, movepick or
the search body. **Do not reopen study 13 §0's closed list.**

Study 13's items 1 (`Board::st()`), 5b (the unsymbolised memcpy), 5c
(`move_piece` out-of-line) and 5f (`check_squares_all`) are **closed by this
study**. What is left, re-ranked:

| # | item | size | confidence | cheapest next test |
|---|---|---:|---|---|
| 1 | **Stop pushing an accumulator entry for null moves.** SF does not: `do_null_move` has no `accumulatorStack.push()` and `undo_null_move` no `pop()` (`eca43a97:src/search.cpp:657-669`, read at the pinned revision). We push `DirtyPiece::EMPTY`, and `at_full`'s alias only rescues an EMPTY entry at the TOP of the walk — one sitting *inside* a fold span is still materialised with empty lists, which is where §3's 0.313 copies/node come from | **~25** | **high — counted + reference** | the copies are now inline (§3.1), so this deletes them outright rather than making them cheaper, and makes `at_full`'s alias loop dead code. **First re-run `memspy.c` post-a2b to confirm the surviving rate** — that is one command and it gives the number before the arm is written |
| 2 | **The same 2560-byte `memcpy` is still live in `sota_accumulator.rs`** — `entry.acc[p] = parent.acc[p]` at `:333` (and `:303`), same condition, same payload, same span walk | **−10 … −25** | **high** | apply §3.1's *winning* spelling (chunked `copy_from_slice`); the losing one is documented at the site. `memspy.c` gives the call rate first |
| 3 | **`unmake_move`'s 168-byte state restore** — the price §2 paid | 0–8 | medium | reviewer C's arithmetic says **do not chase this**: shrinking `check_squares` to 5 entries saves ~4 i/n and costs a runtime branch in `check_squares(pt)` at ~30 calls/node (net negative), and the copy-free shape is SF's pointer/cursor, which costs a load on each of 29.07 `st()` calls to save ~16. **Recorded as closed-by-arithmetic, not as an opportunity** |
| 4 | **The FULL Stockfish transposition** — *replace* `pieces[c][pt]` with `byTypeBB & byColorBB` | ? | medium | §4.6. This is NOT a bigger version of the reverted arm; it removes a layer and 96 bytes instead of adding one, and costs a load + `and` on ~72 `pieces(c, pt)` sites that movegen is built around. Needs its own null and could easily be a loss |
| 5 | **`ThreatPosition::from_board`, 1.01/node** | ~15–25 | medium | still rebuilds 6 type-unions + an ~80-byte store every node. Study 13's untested idea stands: derive the child snapshot from the PARENT's by ≤4 `toggle`s |
| 6 | `attacks::tables()` — 5.62 `LazyLock` derefs/node | ~17 | high | unchanged from study 13 item 5; still untried. A `const`-evaluated table removes the guard *and* the GOT load; the risk is build time, not correctness |
| 7 | `Worker::do_move` 4.44% ≈ 284 i/n | ? | medium | the largest search-side symbol after `search`/`init_quiets`. `capture_stage`, `ContHistId::new` and `DirtyPiece::from_move` all live inside it; none has been split out and priced |
| 8 | `Histories::correction_value` + `update_continuation_histories` | +101 vs SF | medium | 4 `corr_slot` hashes + 2 `cont_corr` reads per node. The `params::*()` getters are compile-time constants without `--features tune` (**verified** — that is not the cost) |
| 9 | `upcoming_repetition` is missing SF's `other != 0` early-out | ~10 | medium — reviewer B | `eca43a97:src/position.cpp:1558-1570` maintains an incremental key and skips the cuckoo probe in ~3 instructions instead of two probes into an 8192-entry table. Pre-existing; this study edited those exact lines |
| 10 | `Board::clone()` loses the `history` reserve | 0 i/n; a latency spike | low — reviewer C | `Board::empty()` reserves `MAX_PLY + 64`, but `#[derive(Clone)]` on a `Vec` yields `capacity == len`, and **every search thread gets a clone**. So `state_stack_is_prereserved_and_never_reallocates` tests the one configuration the search never uses. Pre-existing (the old `states` had it too); fix by reserving in `Worker::setup` or extend the test |
| 11 | ~~`Board::st()`~~ / ~~unsymbolised memcpy~~ / ~~`move_piece`~~ / ~~`check_squares_all`~~ | | **CLOSED** | §2, §3, §4.3/§4.4, §4.5 |

**One process item, not a perf item.** `tools/check.sh` runs `cargo check` across
the feature matrix but `cargo test` only at **default** features, under which
`src/eval/nnue/` is not compiled at all. So the ~20 `sota` tests — including
`threats_replay_matches_refresh`, the real correctness oracle for arm 2 — are
type-checked and **never executed** by any gate; only the opt-in
`CHECK_NNUE_GATE=1` node pin can see that code. Adding
`cargo test --features "$nnue_arch"` inside the existing `CHECK_NNUE_GATE` block
costs ~10 s and closes it (D33 reviewer C, M-1). Study 15 should land that
before its first NNUE arm.

**Three method rules this study is handing over, all paid for:**

1. **A null is only inert if it is STRUCTURALLY inert.** `&v[n-1]` for
   `v.last().expect(…)` moved the statistic 86 i/n (§1.1). Prefer a reborrow or
   a renamed local; never a different standard-library call.
2. **The spelling is the arm.** Two bit-identical implementations of the same
   removal measured +22.7 and −22.7 (§3.1). When an arm with a sound mechanism
   loses, **write the second spelling before writing the refutation** — study 13
   would have landed that 22.7 i/n a day earlier.
3. **"Price the arm, never the mechanism" presupposes the instrument can resolve
   the price.** §4.6 reverted a measured −3.6 i/n win because 3.6 is 4–9% of the
   site sensitivity the same study demonstrated, and the arm bought permanent
   structural complexity. A win smaller than your own noise is not a small win;
   it is an unresolved measurement.

And one instrument: **`LD_PRELOAD` interposition is how you attribute libc time
in a build with no frame pointers and no LBR.** It answered in four minutes a
question two studies had logged as unanswerable, and item 2 above is a second
target it found for free. `scripts/memspy.c` generalises to `malloc`, `free`, or
any interposable symbol.

---

## 9. THE D33 REVIEW ROUND

Three read-only reviewer personas ran against the landed diff (A
correctness/safety, B reference-fidelity vs `eca43a97`, C Rust-idiom/perf/
test-quality). **No blockers in the landed set** — persona A independently
re-ran the gate and reproduced `Bench: 587683`, the release perft suite and
clippy, and independently re-derived the `history[hist_len - i]` index mapping
for all four walks and the `has_repeated` trip count.

The round changed the study materially in five ways, all folded in above, and it
is the reason the final number is 24 i/n better than the pre-review tree:

* **C recommended REVERTING a4, and B's reference reading explained why its
  arithmetic failed.** §4.6. A measured −3.6 i/n win, reverted because it is
  4–9% of the site sensitivity the study itself demonstrated and it bought a
  permanent fifth placement layer. B verified at `eca43a97` that SF's
  `pieces(ROOK, QUEEN)` is *also* 2 loads + an `or` and that SF maintains 3
  bitboards per placement edit where the arm took us to 4 — so the "SF keeps
  `byTypeBB` and we don't" justification was materially incomplete.
* **C proposed the arm that replaced it.** §4.4's `StateInfo::moved`, which
  closes a3's own last site, lands in existing padding (`size_of` still 168,
  verified), and left `Board::move_piece` with no callers at all. The
  post-review tree is **3.1 i/n better than the pre-review tree with a4 removed
  from it.**
* **A and C both found `has_repeated`'s ancestor walk had NO test that could
  fail** — the entire semantic content of a1's rewrite of that function, dead to
  all 686 tests, on a function with no production caller. §7.1; the new test is
  mutation-verified in two directions.
* **C found `assert_consistent`'s check-info oracle could not see a dropped
  field**, because it seeded `expected` from the stored state. It is now seeded
  from `StateInfo::EMPTY`, and `CheckInfo::store` destructures so a missing
  field is a *compile* error. C also caught that the comment claiming
  `check_squares_all`'s test pins the array order was wrong — that test compares
  two reads of the same array and is a tautology w.r.t. ordering; the real guard
  is `perft_checked`'s independent `gives_check` oracle, and the comment now
  says so.
* **B found the §3 code comment argued for a mechanism the study's own data
  rejected** (it described arm a2, which lost, while the code shipped a2b), and
  B/C both traced the *same* 2560-byte `memcpy` still live at two unfixed sites
  in `sota_accumulator.rs`. Comment corrected at the site; sites are §8 item 2.
  B also supplied §8 item 1 — SF never pushes an accumulator entry for a null
  move — which is the largest item in the handoff and was verified here directly
  against `eca43a97:src/search.cpp:657-669` rather than taken on report.

Reviewer items deliberately **not** taken, with reasons: reinstating a
`prev_blockers` local for step 8's ep-legality read (C's own recommendation
against it — `make_move_with` is already spilling, and study 13 §5.1 priced that
class of change at +38); folding `CheckInfo` into `StateInfo` as a `check:`
sub-struct (C — the right shape, but it touches every reader and belongs in its
own commit); reserving `history` in `Worker::setup` (C — pre-existing, and it is
a latency item, not an instructions item, so it is §8 item 10); and adding
`cargo test --features sota` to `check.sh` (C's M-1 — correct and important, but
it is a gate change that would land untested inside a perf commit, so it is
§8's process item).

Documentation obligations discharged in the same commit, per CLAUDE.md's
"decisions change here first; docs follow; code follows docs": **D8 amended** in
`design/decisions.md`; `design/04-position-state.md`'s `Board` sketch, `st()`,
make steps 1/12/13/14, unmake step 1 and §8 acceptance criteria updated;
CLAUDE.md's core-encoding bullet updated; the three stale `board.states.len()`
references in `src/eval/nnue/accumulator.rs` and `design/11-evaluation.md`
fixed.

---

## 10. ARTIFACTS (`perf-artifacts-14/`)

* `probes-fast.txt` — every arm with its eval hash and node count, including
  both nulls, the failed null, the rejected arm and the frame control.
* `counters-headline.txt` — the §0.1 two-arm PGO panel, 5 rounds, with the
  parsed ratio table appended.
* `memspy-baseline.txt`, `memspy-after-a2.txt` — the §3 `memcpy` census before
  and after, by caller and size.
* `flat-a1-instructions.txt`, `flat-a3-instructions.txt` — instruction-attributed
  flat profiles before and after arm 3.
* `gate-s14.log` — both gate runs (default and `CHECK_NNUE_GATE=1`).
* `scripts/probe14fast.sh`, `probe14lm.sh`, `pgo14.sh` — new.
* `gate-s14.log` contains BOTH post-review gate runs; the pre-review runs were
  overwritten deliberately (only the shipped tree's gate is evidence).
* `scripts/memspy.c` — the `LD_PRELOAD` `memcpy`/`memmove` interposer. New.
* `scripts/pstat8.sh`, `parse8.py`, `flat8.sh`, `annot8.py`, `opcount11.py`,
  `blocks13.py`, `sfshow.sh` — inherited.
