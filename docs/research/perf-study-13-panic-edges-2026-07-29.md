# ARCHON perf study #13 — the panic edges, and two mechanisms that were wrong

Date: 2026-07-29. Thirteenth commissioned study (11: `perf-study-11-fair-target-2026-07-29.md`,
12: `perf-study-12-fold-tile-2026-07-29.md`). Raw artifacts and instruments:
`perf-artifacts-13/`. Baseline: `main` at `0ca0c3c` (study 12 merged and gated).
Comparison target: **`engines/bin/stockfish-eca43a97`**.

---

## 0. HEADLINE

### 0.1 The ratio moved

`bench 16 1 300000 tools/nnue/calib/wl50.fens nodes`, two arms interleaved,
`taskset -c 4`, `nice -n 19`, 5 rounds, both PGO. Raw:
`counters-headline.txt`.

| per node | **ARCHON study 13** | ARCHON study 12 | **SF `eca43a97`** |
|---|---:|---:|---:|
| **instructions** | **6081.2** | 6114.6 | **4452.4** |
| **ratio vs `eca43a97`** | **1.3658×** | 1.3733× | — |
| branches | **652.4** | 667.6 | 362.7 |
| branch-misses | 20.8 | 20.6 | 19.7 |
| nodes | 15,001,568 | 15,001,568 | 15,008,494 |
| `instructions:u` spread | 0.0003% | 0.0390% | 0.0000% |

**Stockfish reproduces 4452.4 with 0.0000% spread** — the same control study 12
used, at the same value, so the two ARCHON numbers are in one frame.

**1.3733× → 1.3658×. −33.4 i/n, 2.0% of the 1,662 i/n gap, in four
node-identical changes.** Branches fell 15.2 per node. **1,628.8 i/n remain.**

### 0.2 What landed

Every arm is node-identical (5,001,562 on the probe tree, 15,001,568 on the
headline bench) and eval-hash-identical (`5ca978e012da1bb0`), and the gate's
`Bench: 587683` is unchanged.

| # | change | probe Δ | where |
|---|---|---:|---|
| 1 | policy head snapshotted per search instead of an `RwLock` read per node | −4.8 | `search/mod.rs` |
| 2 | `threat_map()` — one `ensure_threats` for the quiet scorer's five reads | **−19.0** | `board.rs`, `movepick.rs` |
| 3 | `king_square` without its two panic edges | **−13.6** | `board.rs` |
| 4 | `check_squares_all()` hoisted out of the per-quiet loop | −5.6 | `board.rs`, `movepick.rs` |
| | subtotal, probe frame | −43.0 | |
| 5 | §5.4's robustness guard, the price of arm 3 | +5.7 | `uci.rs` |
| | **total, probe frame** | **−37.3** | |
| | **total, PGO frame (the number that counts)** | **−33.4** | |

The probe overstated the PGO delta by 1.29× before the guard; **after it, the
PGO panel reproduced 6081.2 i/n and 1.3658× to the digit** on the shipped tree,
with Stockfish again at 4452.4 / 0.0000% — so the guard's +5.7 probe-frame cost
is not present in the deliverable frame at all. Both panels are in
`counters-headline.txt`.

### 0.3 What this study is actually about

**Three of the seven arms were regressions, and every one of them had a
mechanism that looked airtight on paper.** They are written up at the same
length as the wins (§5), because they close doors:

* the duplicate `DirtyPiece` derivation is **cheaper left duplicated** (+38.2
  i/n to remove it);
* **our 1.84× branch ratio against Stockfish is not bounds checks** — deleting
  the densest bounds check in the engine cost +43.8 i/n and moved branches by
  −0.2 per node;
* and the winning arms' own apparent mechanism **does not generalise**: applying
  it one level down, to the `piece_on(...).expect(...)` in the per-move scoring
  loops, cost **+106.3 i/n** (§5.3).

So the operative rule this study leaves behind is not a mechanism at all:
**price the arm, never the mechanism.** An arm that wins is worth landing
whether or not the story about why is true; an arm that loses is dead even when
the story is airtight. What made that safe to say is Null B (§1.1) — without a
source-perturbation null, all three of those regressions could have been
dismissed as codegen noise.

---

## 1. METHOD, AND THE ONE THING THIS STUDY ADDS TO IT

Studies 8–12's rules were followed. One new gate, and it is the reason two of
this study's arms were thrown away instead of landed:

### 1.1 NULL FIRST — twice, and the second null is the new one

**Null A — rebuild determinism.** The same source, built and measured twice:

```
null-1   32,695,848,765   6537.1 i/n
null-2   32,695,922,205   6537.1 i/n     spread 0.00022%, ±0.015 i/n
```

**Null B — SOURCE PERTURBATION.** Null A only proves the toolchain is
deterministic. It says nothing about whether a *semantically inert* source
change moves the statistic — and at this level of optimisation that is the
question, because a single-site edit can move register allocation and inlining
far more than it moves the work. So: `AttackTables::{rook,bishop}_attacks` were
re-spelled into a three-line form with the same loads, the same arithmetic and
the same bounds check, and re-measured.

```
a0-INERT-accessor-respell   32,671,504,275   6532.3 i/n
a1 (same tree, one-line spelling difference)  32,671,559,510   6532.3 i/n
```

**±0.011 i/n.** The instrument survives its own null in both directions, so
every arm below is a real effect of that arm — including the two that made the
engine *slower*.

### 1.2 Instruments

* **`probe13fast.sh`** — screening. `CARGO_PROFILE_RELEASE_LTO=thin`
  / `CODEGEN_UNITS=4`, native, non-PGO, fixed 5,001,562-node tree, prints the
  wl50 eval hash and the node count so a tree or eval change cannot hide inside
  a speed claim. Study 12 §0.3's rule stands: **no probe delta may be quoted as
  a result**; only the PGO panel decides.
* **`pgo13.sh` + `pstat8.sh`** — the deliverable frame.
* **`counters13.patch`** — deterministic workload counters on **both** engines'
  shared quantities plus ARCHON-only call counts. Row and call counts are
  workload facts: immune to codegen, PGO, sampling and box load.
* **`blocks13.py`** — buckets two `perf report` flat profiles into comparable
  blocks from one table of regexes, so a block at parity cannot be quoted as a
  gap by accident (study 12 had to correct itself twice for exactly that).

### 1.3 The memory guard, corrected

`probe12fast.sh` refused to build whenever `free -g` reported `< 6`. `free -g`
**truncates**: 6134 MB reads as "5", and the guard refused builds with 6.0 GB
genuinely free. `probe13fast.sh` guards in megabytes (`< 4500`), which is the
same safety with honest units.

---

## 2. THE COUNTER PANEL — what actually runs per node

`counters.txt` / `counters-r2.txt`, from a build carrying relaxed-atomic
counters at each site (reverted before landing; the patch is the artifact).

| per node | count |
|---|---:|
| **`Board::st()`** | **29.0686** |
| **`Board::piece_on`** | **20.4402** |
| `AttackTables::rook_attacks` | 5.7053 |
| `AttackTables::bishop_attacks` | 5.2920 |
| `attacks::tables()` (a `LazyLock` deref) | 5.6239 |
| `set_check_info` | 1.0182 |
| `make_move` / `unmake_move` | 1.0059 |
| `ThreatPosition::from_board` | 1.0122 |
| `attackers_to` | 0.9713 |
| `attackers_to_exist` | 0.6657 |
| `between()` | 0.4365 |

Two of these decided the study.

**(a) `Board::st()` runs 29 times per node.** It is
`self.states.last().expect("state stack never empty")` — a length load, an
emptiness test, a panic edge, and only then the pointer arithmetic. Every state
accessor in the engine (`key`, `checkers`, `check_squares`, `blockers_for_king`,
`rule50`, `pawn_key`, …) goes through it.

**(b) 11.0 slider lookups per node**, against Stockfish's dual hyperbola
quintessence, which returns rook *and* bishop from one 64-byte `DualMagic` entry
(`eca43a97:src/attacks.h:90-136`). See §6 — the counter is what turns "should we
port it?" from a taste question into an arithmetic one, and the answer is that
for *this* metric it is between neutral and a loss.

---

## 3. THE ARM LOG

Every measurement this study made, in order, in the fast probe frame
(`probes-fast.txt`). The two nulls bracket the arms.

```
null-1                       6537.1    baseline, main @ 0ca0c3c
null-2                       6537.1    rebuild null, +-0.015 i/n
a1-policyhead-snapshot       6532.3     -4.8   LANDED
a2-dirty-via-hookc           6573.8    +41.5   rejected (S5.1)
a2b-hookc-byvalue            6570.5    +38.2   rejected (S5.1)
a0-INERT-accessor-respell    6532.3    +-0.0   source-perturbation null
a3-attacks-table-mask        6576.1    +43.8   rejected (S5.2)
a4-threat-map-one-fill       6513.3    -19.0   LANDED
a5-king-square-no-panic-edge 6499.7    -13.6   LANDED
a6-check-squares-hoist       6494.1     -5.6   LANDED
a7-piece-on-occupied         6600.4   +106.3   rejected (S5.3)
--- D33 review round; S5.4's robustness fix, two spellings ---
a8-post-D33-review           6584.7    +90.6   Board::is_legal_root  (rejected)
a8b-is-legal-root-cold       6585.8    +91.7   ... + #[cold]/#[inline(never)]
a8c-BISECT-guard-disabled    6534.7    +40.6   ... call site removed, method kept
a8d-guard-in-uci             6499.8     +5.7   uci::is_legal_root    (LANDED)
```

Node count 5,001,562 and eval hash `5ca978e012da1bb0` on **every line**,
including the rejected ones — the two losses are correct code that is slower,
not broken code.

## 4. WHAT LANDED, AND WHY EACH ONE WORKED

### 4.1 The policy head's `RwLock`, taken once per search instead of once per node (−4.8)

`search::<NT>` called `crate::eval::nnue::active_policy_head()` at every
interior node with no TT move, and again wherever `CPL_ORD_W` is set. That
function is

```rust
*ACTIVE_POLICY_HEAD.read().unwrap_or_else(|e| e.into_inner())
```

— a process-wide `RwLock` read lock: an atomic read-modify-write to acquire, a
poison check, another to release, for a value that **cannot change during a
search**. Nets are installed once per `go`, by `refresh_eval_net`, and D47
policy is switched off in production anyway (no `.cpol` sidecar ⇒ `None`).

`SotaContext` already snapshots the same handle at net-set time
(`sota_context.rs:213`). This is that pattern applied to the search: a
`Worker::policy_head` field written in `refresh_eval_net` beside the nets.

−4.8 i/n is the smallest arm here and it is worth noting that the instruction
count *understates* it: two `lock cmpxchg`-class operations per node cost far
more in cycles than in instructions. It is quoted at what the metric says.

### 4.2 One `ensure_threats` instead of five (−19.0)

`MovePicker::init_quiets` builds its escape/entry tiers from the D30 threat map:

```rust
let by_pawn = board.piece_threats(PieceType::Pawn);
let by_minor = by_pawn | board.piece_threats(Knight) | board.piece_threats(Bishop);
let by_rook  = by_minor | board.piece_threats(Rook);
…
let undefended = board.occupancy(them) & !board.all_threats();
```

Five accessors, **five `ensure_threats()` calls**, of which four can only ever
observe a cache the first one already filled. `ensure_threats` is not inlined,
and each call re-materialises `self.st()` — a `Vec::last()` with its own length
load, emptiness test and panic edge — then re-reads and re-compares the `Cell`
key.

`Board::threat_map()` returns `(all, by_type)` behind one fill check. Node- and
eval-identical: same sets, same source, same order.

**−19.0 i/n and −3.5 branches per node**, the study's largest arm.

### 4.3 `king_square` without its two panic edges (−13.6)

```rust
self.pieces(color, PieceType::King).lsb().expect("king must exist")
```

states the invariant twice for the optimiser to disbelieve twice: `lsb()` tests
the board for zero and wraps the result in an `Option`, then `expect` tests the
`Option` again. Two panic edges on a query that `set_check_info` alone makes
three times per node, and which also runs in movegen, the HalfKA refresh, the
threat resolver and the feature indexers.

`trailing_zeros()` of a non-zero `u64` is already in `0..64`. Masking with 63
says so to LLVM — which additionally folds away `Square::new`'s own
`assert!(index < 64)` — and the `debug_assert!(bb != 0)` is where the invariant
now lives, still checked at every node of every debug search.

**−13.6 i/n and −12.5 branches per node.** The branch figure is the one that
identifies the mechanism: ~12 branch-pairs per node disappeared, which is the
call rate this accessor has.

### 4.4 The check-square block, hoisted out of the per-quiet loop (−5.6)

The quiet scorer's check bonus reads `board.check_squares(pt)` once per scored
quiet — ~30 per node — with `pt` varying per move, so it cannot be hoisted by
hand at the call site, and LLVM only hoists the underlying `st()` when it can
prove nothing in the loop body invalidated it. `Board::check_squares_all()`
returns the whole 48-byte block, so the hoist is a property of the source.

−5.6 i/n. Small, and included because it is the same shape as §4.2 at a
different call rate, which is the evidence that the shape generalises.

---

## 5. WHAT DID NOT LAND — three mechanisms, all measured, all wrong

All three arms below are **correct** code: node-identical (5,001,562) and
eval-hash-identical (`5ca978e012da1bb0`). They are simply slower, and Null B
(§1.1) is what makes that statement safe to make. §5.4 is a fourth finding of a
different kind — a landed arm that turned out not to be free.

### 5.1 The duplicate `DirtyPiece` derivation — **+38.2 i/n**

`Worker::do_move` builds the accumulator's placement record with
`DirtyPiece::from_move(&self.board, m)` *before* the make. That call re-derives
the mover, the victim, the en-passant capture square and the castling rook
geometry — every one of which `make_move_with` computes a few instructions later
for its own Zobrist work. It measured as its own **53 i/n** symbol in study 12's
stage-split profile, and Stockfish fills its `DirtyPiece&` out-parameter inline
in `do_move` (`eca43a97:src/position.cpp`, the `dp.pc = pc; dp.from = from; …`
assignments). The seam for doing the same already existed and had never been
wired: `MakeMoveHooks::on_dirty`, Hook C, whose doc comment says *"the call site
lands with the search wiring in step 4"* — it never did.

So it was wired: an associated `const WANTS_DIRTY: bool` (so `NoHooks`, hence
perft and `Board::make_move`, monomorphizes the whole construction away), the
record built from `make_move_with`'s own locals in `from_move`'s exact emission
order, and `Worker::do_move` reading it back out of the hook.

**+41.5 i/n.** A second variant — payload passed by value instead of by
reference, and the castling rook geometry re-derived inside the guarded block
instead of carried in an `Option` local across the whole body — recovered 3.3 of
it: **+38.2 i/n**, still a loss — and that variant is, as it happens, exactly
Stockfish's own shape, because SF does not fill the castling rook half of `dp`
in `do_move`'s body either. It fills it *inside* `do_castling<true>`
(`eca43a97:src/position.cpp:1351-1354`).

The mechanism was real and the fix still made things worse: `make_move_with` is
a 650-instruction function that is already spilling, and the marginal cost of
materialising a `DirtyPiece` *there* exceeds the cost of deriving it from
scratch outside, where the register pressure is low. **The flat profile's 53 i/n
for `DirtyPiece::from_move` was not waste — it was work that has to happen
somewhere, and outside is the cheap place.** Reverted; the reason now sits in a
comment at the call site so study 14 does not re-derive it.

### 5.2 The slider table's bounds check — **+43.8 i/n**, and the thesis behind it is dead

Our headline branch count is **1.84× Stockfish's** (667.6 vs 362.7 per node,
study 12 §0.1; 1.80× after this study's arms). The obvious reading is Rust bounds checks, and the packed magic
table is the densest candidate: `self.table[magic.index(occ)]` on a
`Box<[Bitboard]>`, 11.0 times per node, each paying a length load, a `cmp` and a
`jae`. The index is in range **by construction** (the per-square blocks tile the
array and `init_magics` verifies every slot against the naive oracle), so the
check is provably dead.

Arm: allocate the table at `TABLE_SIZE.next_power_of_two()` (131,072 entries,
183 KiB of unreachable zeros) as a fixed-size `Box<[Bitboard; TABLE_CAP]>` and
range-limit the index with `& (TABLE_CAP - 1)` — the same idiom the threat
resolver's `& (DELTA_CAP - 1)` and the Finny diff's `& (CAP - 1)` already use.

**+43.8 i/n, and branches moved by −0.2 per node.**

That −0.2 is the finding. **The checks were not executing**, so there was
nothing to delete and the `and` was pure addition. A static count over the whole
disassembly confirms it independently: the square-range check `cmp $0x3f` occurs
**once or twice per hot function** and **zero times** in `see_ge`, `legal`,
`gives_check` and `init_quiets` — LLVM already knows `Move::from()`/`to()` are
masked to six bits, and CSEs the one check it cannot avoid across every
square-indexed table in the function.

**So our 1.84× branch ratio is NOT bounds checks.** Two independent
measurements say so. Study 14 should not spend a session on `get_unchecked`,
masking idioms, or a `Square` newtype with a narrower range: that lever has been
priced at zero. The branches are genuine control flow — `record_square_touch`
alone carries 127 conditional branches, `at_full` 139, `init_quiets` 50.

### 5.3 …and the same trick one level down does NOT generalise — **+106.3 i/n**

§4.3 removed two panic edges from `king_square` and won 13.6 i/n with 12.5
branches. The obvious generalisation is the *other* high-traffic `Option`
unwrap: `board.piece_on(m.from()).expect(...)`, which the quiet scorer runs once
per scored move (~30 per node) and the capture scorer once per capture. Same
shape, three times the call rate, and `Board::piece_on` is 20.44 calls per node
(§2).

Arm: a `Board::piece_on_occupied(sq) -> Piece` that keeps the precondition in a
`debug_assert` and returns a well-defined placeholder in release, so the panic
edge becomes a `cmov`.

**+106.3 i/n, and branches went UP by 10.9 per node.**

That is the largest single regression this study measured, and it kills the
tempting one-line summary of §4. **The winning shape is not "delete panic
edges".** `king_square`'s win was something narrower — `lsb()`'s zero test
*plus* the `Option` test *plus* the folding-away of `Square::new`'s own
`assert!(index < 64)`, on an accessor whose result feeds address arithmetic —
and it does not transfer to an `Option<Piece>` load whose result feeds a history
index. Written down here rather than in a code comment because the *code* is
unchanged: the reverted arm leaves no trace to hang the warning on.

Three arms measured, three mechanisms named, three wrong. **Price the arm,
never the mechanism.**

### 5.4 The one landed arm that was NOT free — and what it cost

`king_square`'s `expect("king must exist")` was described above as an invariant
the optimiser is made to disbelieve twice. **The D33 review (persona A) showed
that the invariant is not one**, and the finding is worth more than the arm:

* `from_fen` does **not** validate that kings exist — king-less analysis boards
  are a deliberately supported input class (`fen.rs`; `Board::is_check` returns
  early for them).
* A king **can** be captured. From a FEN where the side *not* to move is already
  in check — `4k3/8/8/8/8/8/4R3/4K3 w - - 0 1`, which every parser accepts —
  `generate_legal` emits `Rxe8`: its legality filter only vets pinned, king and
  en-passant moves, and it has no king-capture test.

So the `expect` was a **live release tripwire**, not dead code, and removing it
turned "panic, caught by the pool's `catch_unwind`, no `bestmove`" into "search
a board with no black king and return a plausible move" — a silent wrong answer,
which is the failure class this project's standing bar exists to prevent.

**The arm was kept and the tripwire moved to where it costs nothing.**
`Board::is_legal_root()` — both kings present, and the side not to move not in
check — is now enforced at the UCI `position` boundary, which is where an engine
that "never panics on input" (design/12 §3.3) should have been rejecting these
all along. It is one predicate on a cold path instead of two panic edges several
times per node, and it rejects the *whole* class rather than the one square
`king_square` happened to look at. Two tests cover it, both mutation-verified by
deleting each term of the predicate in turn.

**This is a net robustness improvement that the perf arm paid for**, and it is
the second time in this study that chasing instructions produced a
correctness-relevant finding the study was not looking for.

### 5.5 …and the fix itself cost +90.6 i/n until it was written differently

The obvious spelling of §5.4's guard is a `Board::is_legal_root()` method
calling `has_both_kings()`, `king_square()` and `attackers_to_exist()`, invoked
once from `cmd_position`. **It measured +90.6 i/n** — twice the study's entire
landed win, for a predicate the search never executes.

Bisected: **+51 of it is the call site in `cmd_position`, +40 is the method
merely existing.** `#[cold]` and `#[inline(never)]` on the method changed
nothing (6585.8 with them, 6584.7 without). The mechanism is the same one that
killed §5.1, §5.2 and §5.3: one new caller shifts LLVM's inline cost model for
`attackers_to_exist` and `king_square` engine-wide, and the search pays for it
on paths that never touch the new code.

Rewritten as a free `uci::is_legal_root(&Board)` built from the `attacks::`
free functions — which already have many callers — instead of Board's inlinable
helpers, the same predicate measured **+5.7 i/n**. Identical behaviour, identical
tests, both mutation-verified.

**This is the most alarming result in the study.** It says that in this codebase
the cost of a change is only weakly related to the work the change does, and
that *cold, never-executed code in an unrelated module* can move the hot path by
1.4%. Study 14 must probe every change, including the ones that "obviously
cannot matter" — and, more usefully, it says there may be **free i/n sitting in
the crate's existing structure**, in the form of caller counts and inline
decisions nobody chose. That is a lever this study did not have time to pull:
`-C llvm-args=-inline-threshold`, `#[inline(never)]` on cold-but-inlinable Board
helpers, and per-function PGO hints are all untried.

---

## 6. WHAT THE REFERENCE SAYS, AND WHY WE ARE NOT COPYING IT

Stockfish at `eca43a97` **no longer uses magic bitboards on AVX2-capable x86**
— which is every modern release binary, including our comparison target
(`stockfish-eca43a97` reports `x86-64-vnni512`). With AVX2 it compiles
`USE_DUAL_HYPERBOLA_QUINT` (`eca43a97:src/attacks.h:34-36`); the classic fancy
magic survives only as the pre-AVX2 `#else` fallback (`attacks.h:142-163`).

The AVX2 form is a `DualMagic` per square (`attacks.h:90-97`) whose
`both_attacks_bb` (`attacks.h:107-136`) computes rook *and* bishop attacks
together: **three masks plus a zero lane** in one `_mm256_load_si256`, one
hyperbola-quintessence step across all four lanes with a **byte** swap standing
in for the bit reversal (legal because file, diagonal and antidiagonal each put
their squares in different bytes), an `or`/`extract` pair to split the result
into `[bishop, rook]`, and an 8-bit rank-attack lookup for the one direction
hyperbola cannot serve. The vector load is 32 bytes but three more scalar fields
come from the same 64-byte entry, so it is **one cache line**, not one load.

The sizes are exact, not approximate: `sizeof(DualMagic)` is 64 and the array is
`alignas(64) DualMagic[64]` = **4096 B** (`attacks.h:90-97`, `attacks.cpp:34`),
plus `8 × 256 × u8` = **2048 B** of rank attacks (`attacks.cpp:73-79`). **6 KiB
total, and no attack table at all.** Ours is an **841 KiB** packed magic array
(`src/attacks.rs:469-473`) plus 4 KiB of metadata. Theirs is unconditionally
L1-resident; ours cannot be — but the per-node working set is far smaller than
841 KiB (11 lookups over ~8 distinct squares, and a single square's rook block
is 8–32 KiB), **and this study measured no cache counter at all.** "Guaranteed
L2/L3 miss" would be an unmeasured causal claim and study 14 should not inherit
it as established; `perf stat -e L1-dcache-load-misses` on both engines is what
would price it.

It is tempting, and it is **probably** the wrong trade for this metric — and the
first draft of this paragraph got the arithmetic wrong in both directions at
once, so here it is re-derived (CLAUDE.md rule 8).

`both_attacks_bb` is **15 vector instructions** (counting the intrinsics at
`attacks.h:107-136`, excluding the free `set_epi8` constant and the free cast)
**plus a ~7-instruction scalar rank-attack tail** — shift load, table-pointer
load, `shrx`, byte mask, `movzx` table load, `shlx`, `add` — so **~22 for both
piece types**. Our magic lookup is **~10**, not 5: mask load, `and`, `imul`,
shift load, `shrx`, offset `add`, the slice length load and `cmp`/`jae` of §5.2,
and the table load. So a *paired* rook+bishop query is ~22 against our ~20 —
**roughly neutral**, not a clear loss.

The loss is at the **unpaired** call sites, and it is a real asymmetry: SF's own
dispatch routes single-type `attacks_bb<ROOK>` and `attacks_bb<BISHOP>` through
the full dual body and lets DCE drop one extract (`attacks.h:279-292`), so an
unpaired query costs ~21 against our ~10. Most of our sites do pair —
`attackers_to` (`board.rs:734-735`), `set_check_info` (`:807-808`), `see_ge`
(`:1811-1818`), the threat generator (`sota_threats.rs:359-360`) — but
`movegen.rs:192,198` and the blocker loop at `board.rs:841-849` do not.

**At 11.0 slider lookups per node the port is somewhere between neutral and
+50 i/n depending on a pairing rate we did not measure.** That is enough to say
it is not a lever for *this* metric and not enough to say it is a loss. It is
clearly a **cycles** lever — 6 KiB of L1-resident tables against 841 KiB — and
it belongs to whoever is optimising NPS. Our headline is `instructions:u`, so
the two goals have diverged here for the first time, and study 14 should notice
that before treating i/n as a proxy for speed.

---

## 7. THE BLOCK TABLE, RE-DERIVED IN ONE FRAME

`blocks13.py` over study 12's two **stage-split** profiles (ARCHON at 6362 i/n,
SF at 5090 i/n — both calibrated off their own `search<NonPv>` symbol, which
study 12 priced at 606 and 617 respectively). Split frames are not comparable in
absolute terms; block *ordering* and block *differences* are.

| block | ARCHON | SF | delta |
|---|---:|---:|---:|
| search | 690 | 672 | +17 |
| movepick | 740 | 734 | +6 |
| eval head | 572 | 572 | **0** |
| accumulator fold | 1232 | 1099 | +133 |
| threat generation | 718 | 686 | +33 |
| eval glue (`SotaContext::push`/`evaluate`, `EvalContext`) | 376 | 171 | **+205** |
| **make / unmake** | 661 | 302 | **+359** |
| board queries | 493 | 380 | +113 |
| movegen | 110 | 47 | +63 |
| histories | 331 | 230 | +101 |
| tt | 56 | 81 | −25 |
| unbucketed | 238 | 23 | +215 |
| **TOTAL** | **6307** | **5059** | **+1248** |

Three of these are new information:

* **Threat generation is at parity (+33), not a defect.** Our lazy regenerate-
  at-fold design (`at_full` 363 + `record_square_touch` 140 + `record_targets`
  ~95 = ~600) costs *less* than Stockfish's eager incremental maintenance
  (`update_piece_threats<true>` 389 + `<false>` 120 + `append_changed_indices`
  177 = 686), because we only pay it on the 43% of nodes that evaluate. Study
  12's §11 item 3 sized `at_full` at "≤ +400" of headroom; the split says the
  headroom is ~0 and the 954 i/n of the unsplit profile is mostly the inlined
  i8 fold (513 when split out).
* **The accumulator fold is at its arithmetic floor.** Provable count from the
  disassembly and the row counters: `80 + 80·rows` instructions per pass, 6.8
  rows and 2.30 passes per node ⇒ ~732 i/n of unavoidable arithmetic, against
  SF's ~584 at 1024 lanes and 7.95 rows. The whole difference is width: at 1024
  we would be at 581 against their 584. **There is no code lever left here** —
  §8 item 6.
* **`make/unmake` (+359) and `eval glue` (+205) are where the gap actually is**,
  and between them they are 45% of the whole 1,248 i/n of split-frame excess.

And one row nobody has looked at: **`unbucketed` is +215**, i.e. ARCHON has 238
i/n of symbols that fit none of the twelve blocks against Stockfish's 23. Its
largest members are `l2_tail` 37 (the L2→L3 head tail, arguably "eval head"),
`Worker::stat_score` 27, **`active_policy_head` 25** (deleted by §4.1),
`Board::move_piece` 22 — *an out-of-line call inside `make_move_with`* — and
~68 i/n spread over four unsymbolised addresses in one 16-byte range, which is
almost certainly a libc `memmove`/`memcpy` variant. **Nobody has established
what is calling it.** That is a cheap and completely unexplored 68 i/n: build
with `-C force-frame-pointers=yes` (the `profiling` profile already does) and
take a call-graph profile of that address.

---

## 8. RANKED REMAINDER — handoff to study 14

**1,628.8 i/n remain**, of which ~327 is the 1280-vs-1024 accumulator width (a
retrain, not a code change — study 12 §11 item 9, unchanged here) and, by §7,
**~0 is the accumulator fold, the eval head, the threat generator, movepick or
the search body**. The code gap that is actually addressable now lives in
`make/unmake` (+359), `eval glue` (+205), `board queries` (+113), `histories`
(+101) and `movegen` (+63).

**Read §5 first.** Two of this study's four hypotheses were wrong in a way that
cost a build cycle each, and both are the kind that look obvious on paper.

| # | item | size | confidence | cheapest next test |
|---|---|---:|---|---|
| 1 | **`Board::st()` — 29.07 calls/node through `Vec::last().expect(...)`** | ~30–60 | **high — counted** | see below |
| 2 | **eval glue: `SotaContext::push` 115 + `push_sota_dirty` 139** | +205 vs SF's 171 | medium | 5 `Option` checks for absent side-nets per push, and `ThreatPosition::from_board` (1.01/node, ~25 ops + a 74-byte store) which could be derived from the parent snapshot by ≤4 `toggle`s |
| 3 | make/unmake's remaining +359 after item 1 | ? | medium | `set_check_info` is 59 i/n against SF's 33 for identical code; `move_piece` is a 22 i/n **out-of-line call** per make |
| 4 | fold-kernel row-loop control | ~20–40 | medium | 6 instructions per row per tile (`mov`/`lea`/`shl`/`add`/`cmp`/`jne`), 6.8 rows × 2 tiles = 82 i/n; resolve the byte offset at resolve time so the kernel's list is pre-scaled |
| 5 | `attacks::tables()` — 5.62 `LazyLock` derefs/node | ~17 | high | a `const`-evaluated table removes the check *and* the GOT load; ~5 M const-eval steps, so the risk is build time, not correctness |
| 5b | **the unsymbolised ~68 i/n of libc `memmove`** (§7) | ~68 | **unknown — nobody has looked** | one call-graph profile on the `profiling` profile; the caller is not in any flat table anyone has produced |
| 5c | `Board::move_piece` is a 22 i/n **out-of-line call** per make | ~15 | medium | it is the make/unmake hot path's innermost primitive and it is not being inlined; check why (`#[inline]` present? cold-call heuristics from the panic edges in `put_piece`/`remove_piece`?) |
| 5d | `gives_check` (2–3 `st()` calls) and `see_ge`'s swap loop (re-reads `pinners`/`blockers_for_king` **inside** the loop) | ~1–3 `st()`/node each | medium — D33 reviewer C | one combined arm, not two: both are inside `board.rs` where `st()` is already in scope, so neither needs new public API |
| 5e | `ord_weight()` / `ord_min_depth()` — per-node `LazyLock` derefs beside the arm that removed the `RwLock` | ~6 | high — reviewer C | reorder the guard so the now-free `self.policy_head` test short-circuits first, or fold head + weight + min-depth into one `Option<PolicyOrder>` snapshot |
| 5f | `check_squares_all` returns `[Bitboard; 6]` **by value** | small, negative-risk | reviewer C | `&self.st().check_squares` — the caller indexes it at runtime anyway, so the 48-byte copy is pure loss. (`threat_map` genuinely cannot: `Cell::get()` is a copy by definition.) |
| 6 | ~~accumulator width / fold~~ | **0 as code** | **closed** | §7 — at its arithmetic floor; the residual is 1280 vs 1024, a retrain |
| 7 | ~~threat generation~~ | **+33** | **closed** | §7 — cheaper than Stockfish's per-node equivalent |
| 8 | ~~bounds checks / the 1.84× branch ratio~~ | **0** | **REFUTED** | §5.2, two independent measurements |
| 9 | ~~`DirtyPiece::from_move` duplication~~ | **+38** | **REFUTED** | §5.1 |
| 10 | ~~dual hyperbola quintessence~~ | **negative for i/n** | **priced** | §6 — a cycles lever, not an instructions lever |

**Item 1, in full, because it is the ranked top and it is blocked on one thing.**
`st()` is 29.07 calls/node, and the shape that makes it expensive is
`Vec::last()`: a length load, a zero test and a panic edge before any useful
work. The fix is the standard one — `states: Box<[StateInfo; STATE_CAP]>` with a
`top: usize` cursor and `STATE_CAP` a power of two so the index is a mask — and
it also deletes the provably dead `states.push(StateInfo::EMPTY)` (three `zmm`
loads and three `zmm` stores of a 168-byte constant that is fully overwritten,
confirmed in the disassembly at `make_move_with+0x9e`) plus the `mem::take`
round trip.

**What blocks it**: `make_move_with` holds `st: &mut StateInfo` across calls that
need `&mut self` (`remove_piece`, `put_piece`, `move_piece`) and `&self`
(`set_check_info`, `attackers_to`). Today that is legal only because the whole
`Vec` is moved out with `mem::take` (a `Vec` has a cheap `Default`; a
`Box<[T; N]>` does not). Study 14 needs one of: (a) a `Placement<'_>` helper
struct holding disjoint `&mut` borrows of `pieces`/`color_occupancy`/
`all_occupancy`/`mailbox`, with the three placement mutators moved onto it — the
idiomatic Rust answer to a partial borrow, and it makes the borrow split
explicit instead of a `mem::take` trick; or (b) `set_check_info` returning its
ten bitboards by value so no `&mut StateInfo` needs to live across a `&self`
call. **This is a 1–2 hour refactor of the most D24-critical function in the
engine and it should be budgeted as one, not slipped in.**

**And a method warning that outranks all of the above.** Single-site changes in
this codebase move the statistic by **±40 i/n through codegen alone** — arm 5.2
added one `and` per lookup and cost 43.8. The rebuild null is ±0.015 and the
inert-respelling null is ±0.011, so this is not noise: it is real, arm-caused,
and *not attributable to the mechanism the arm names*. The practical
consequence: **price the arm, never the mechanism.** An arm that wins is worth
landing whether or not the story about why is true; an arm that loses is dead
even when the story is airtight. Study 12's five landed arms and this study's
four both fit that rule; the two refuted here are what happens when you do not.

---

## 9. GATE

`gate-s13.log`.

* `bash tools/check.sh` — **`Bench: 587683`**, unchanged. Arms 2 and 4 are in
  `movepick.rs` and `board.rs`, both of which the default-feature build
  compiles, so this is a real node-identity check on them and on `king_square`.
* `cargo fmt --check`, `cargo clippy --all-targets -- -D warnings`, `cargo test`
  (604 lib tests, 0 failed), the release perft suite (`Nodes searched: 1839507`)
  and the two-run deterministic bench-perft fingerprint.
* The D24 perft ritual applies (`king_square` is read by movegen and by the
  check bookkeeping) and passes as part of check.sh.
* **`CHECK_NNUE_GATE=1` was run even though the diff touches nothing under
  `src/eval/nnue/`** — `king_square` feeds the HalfKA king bucket, the Finny
  cache key and the threat resolver's orientation, so the bench pin alone would
  not have seen a mistake in it. Both NNUE pins reproduce:
  * eval hash `5ca978e012da1bb086799709c3ea5350cf6d49923c254f7975b6d780b0098b38`
  * d13 nodes **1922947**

**Scope of that proof, stated honestly** (D33 reviewer A): the gate triple proves
node identity for arms **4.2, 4.3 and 4.4**, which are the three that touch
`board.rs`/`movepick.rs`. It cannot observe arm **4.1** in either direction —
`active_policy_head()` is `None` in every gated configuration, because no
`.cpol` sidecar exists anywhere in the tree (every net in
`data/nnue/nets/screens/` is named `*-nocpol.cenn`) and `CPL_ORD_W` defaults to
0. Arm 4.1 is therefore verified by an exhaustive trace of the global's writers
and readers, not by a pin: the only in-session writer is `setoption EvalFile`,
and `dispatch` calls `wait_for_search_finished()` before it, so no worker can be
between its `refresh_eval_net` and a node. Every worker — main and pool helper
alike — runs `refresh_eval_net()` as the first statement of `start_searching`,
and `bench` refreshes before its position loop.

`gate-s13.log` contains **both** runs concatenated: the default run (whose
footer correctly says the NNUE gate was skipped) followed by the
`CHECK_NNUE_GATE=1` run with the two NNUE pins.

### 9.1 Each landed change has a test that can fail, and it was verified by mutation

| change | test | mutation that makes it fail |
|---|---|---|
| `threat_map` | `threat_map_on_a_cold_cache_agrees_with_the_per_type_accessors` | drop the `ensure_threats()` ⇒ reads the cold all-zero cache |
| `king_square` | `king_square_is_exact_across_the_index_range` | `& 62` instead of `& 63` (h8/a1 corners are in the fixtures) |
| `check_squares_all` | `check_squares_all_agrees_with_the_per_type_accessor` | return `[EMPTY; 6]` |
| `is_legal_root` (§5.4) | `position_illegal_root_is_rejected_and_atomic` | delete either term of the predicate — each term is caught by a different fixture |

**The policy-head snapshot (§4.1) has NO test that can fail, and no pin can see
it** (D33 reviewer C). `active_policy_head()` returns `None` in every gated
configuration: `PolicyHead::sibling_of` resolves `.cpol` against the path it is
given, and the gate loads `engines/canonical/net.cenn`, so it looks for
`engines/canonical/net.cpol` — while the real sidecar sits beside the symlink's
*target* in `data/nnue/nets/screens/`. Mutating the snapshot to a hard `None`
(i.e. silently disabling D47 ordering) passes `cargo test`, `check.sh` and
`CHECK_NNUE_GATE=1`. The arm is verified by the writer/reader trace in §9, not
by a test, and closing that hole needs a test-only installer for the global plus
a `--features sota` test — study 14's cheapest correctness debt, not a perf
item.

All three mutations were applied and **all three tests failed**, then the
mutations were reverted and the suite re-run clean. The first test's boards are
deliberately FRESH — the existing `threat_map_matches_the_attackers_to_oracle`
would have passed a `threat_map` with no fill, because it warms the cache first.

---

## 10. THE D33 REVIEW ROUND

Three read-only reviewer personas ran against the landed diff and this document
(A correctness/safety, B reference-fidelity vs `eca43a97`, C Rust-idiom/perf/
test-quality). **No blockers in the landed set.** The round changed the study
materially in four ways, all of which are folded in above:

* **A found that §4.3's stated invariant does not exist.** `from_fen` does not
  validate kings and a king *can* be captured from an illegal-but-accepted root,
  so `king_square`'s `expect` was a live release tripwire. §5.4, and the tripwire
  now lives in `Board::is_legal_root()` at the UCI `position` boundary with two
  mutation-verified tests. A also verified exhaustively that **no path** can
  change the policy head between `refresh_eval_net` and a node — including pool
  helpers, `setoption` mid-search and `ucinewgame` — which is what arm 4.1 rests
  on in the absence of a test.
* **B corrected §6's arithmetic in both directions.** `both_attacks_bb` is ~22
  instructions (15 vector + a ~7-instruction scalar rank tail), not 15; our magic
  lookup is ~10, not 5 — a fact the study's own §5.2 already implied and §6
  contradicted. The verdict survives; the derivation did not, and "priced" is now
  "between neutral and +50 depending on a pairing rate we did not measure". B also
  scoped "no longer uses magic bitboards" to AVX2 builds (the pre-AVX2 `#else`
  still has them), struck the unmeasured "guaranteed L2/L3 miss", and found that
  **`append_changed_indices` is fold-time**, so §7's eager-vs-lazy comparison was
  inflating Stockfish's eager side by 35%. B confirmed no GPL code was copied and
  that all three §6 size figures (4 KiB / 2 KiB / 841 KiB) were already exact.
* **C found that arm 4.1 has no test that can fail and that no pin can see it** —
  `PolicyHead::sibling_of` resolves `.cpol` against `engines/canonical/net.cenn`,
  while the real sidecar sits beside the symlink's *target*. §9.1 now says so
  instead of the commit message's blanket claim. C also caught two doc numbers
  that contradicted the artifact they cited ("~30 per node" for a per-`init_quiets`
  rate; "+38.2 twice" for +41.5 and +38.2), and two test holes: the check-square
  fixtures made the Pawn and Bishop sets *identical*, so a swapped index would
  have passed, and `king_square_is_exact_across_the_index_range` sampled 8 of 64
  indices when the whole domain is free. Both are fixed and the check-square test
  now also pins pairwise distinctness.
* **C independently reached the same verdict on arm 7** (§5.3) from the probe log
  before reading the write-up, and named the one remaining site with
  `king_square`'s *actual* shape — `checkers().lsb().unwrap()` at
  `movegen.rs:174` — with the correct caveat that it is not the same win, because
  `king_square` removed three branches and `a6` removed 0.1.

Reviewer items deliberately **not** taken, with reasons: collapsing
`Worker::policy_head` into `EvalContext` (C's N-3 — right on one-source-of-truth
grounds, but it is a re-measure and the two copies cannot diverge while both are
written inside `refresh_eval_net`); moving the `trailing_zeros() & 63` idiom onto
`Bitboard` as `lsb_or_a1` (C's N-4 — a strictly better home, and a codegen
cascade risk of the exact size this study spent three arms learning to respect);
and returning `&[Bitboard; 6]` from `check_squares_all` (C's N-1 — a real
improvement, listed as handoff item 5f so it is *measured* rather than assumed).

---

## 11. THREATS TO VALIDITY

1. **Probe deltas overstate PGO deltas by ~2×** (study 12 §0.3). Every per-arm
   figure in §3 is probe-frame; only the total is PGO-measured.
2. **§7's block table mixes two split builds** with different `noinline` sets and
   different compilers. Block ordering and differences are safe; absolute i/n
   are ±.
3. **Nothing here was Elo-tested**, and there is nothing to test: every landed
   arm is node-identical (5,001,562 on the probe tree) and eval-bit-identical
   (`5ca978e012da1bb0`), so the engine plays exactly the same moves.
4. **The counter build perturbs inlining** — an atomic increment inside `st()`
   prevents LLVM from hoisting it out of loops, so 29.07 is the number of
   *calls*, not of executed materialisations. The instruction cost of item 1 is
   therefore an upper bound until it is measured by an arm.
5. **Box state**: a D48Y training run shared the box for the whole session
   (GPU at 91%, ~7.1 GB RSS). Everything here rides on `instructions:u` over
   fixed-node trees and on deterministic counters, both contention-immune; no
   wall-clock or NPS number is quoted as a claim.

---

## 12. ARTIFACTS (`perf-artifacts-13/`)

* `probes-fast.txt` — every arm with its eval hash and node count, including
  both nulls and all three refuted arms.
* `blocks.txt` — the §7 block table as emitted.
* `counters.txt`, `counters-r2.txt` — the §2 deterministic panels.
* `counters-headline.txt` — the §0.1 two-arm PGO panel, 5 rounds, with the
  parsed ratio table appended.
* `gate-s13.log` — both gate runs (default and `CHECK_NNUE_GATE=1`).
* `counters13.patch` — the counter instrumentation for `counters.txt`
  (reverted). **It does not carry the `Board::st()` / `Board::piece_on` hunks**
  that produced `counters-r2.txt`'s two headline rows — those are one relaxed
  `fetch_add` at the top of each accessor, reported through the same
  `perfcount13_dump`. Study 14 should re-derive 29.07 rather than inherit it
  (D33 reviewer C, N-8), which is CLAUDE.md research rule 8 anyway.
* `scripts/probe13.sh`, `probe13fast.sh`, `pgo13.sh`, `null13.sh` — new.
* `scripts/blocks13.py` — the block bucketer; its output is `blocks.txt` (§7).
* `scripts/sfshow.sh` — pinned-revision reader for the reference clone.
* `scripts/pstat8.sh`, `parse8.py`, `flat8.sh`, `annot8.py`, `opcount11.py` —
  inherited.
