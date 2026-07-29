# ARCHON perf study #16 — the memset class closes, and four smaller things pay

Date: 2026-07-29. Sixteenth commissioned study (13: `perf-study-13-panic-edges-2026-07-29.md`,
14: `perf-study-14-state-and-memcpy-2026-07-29.md`,
15: `perf-study-15-null-entry-and-dead-memset-2026-07-29.md`). Raw artifacts and
instruments: `perf-artifacts-16/`. Baseline: `main` at `905dccd` (study 15
merged and gated). Comparison target: **`engines/bin/stockfish-eca43a97`**.

---

## 0. HEADLINE

### 0.1 THE DELIVERABLE FRAME IS NOT MEASURED — READ THIS FIRST

**There is no PGO headline panel in this study, and the merge gate did not
complete.** The box ran two studies beside a live GPU training run, went into
hard swap thrash (swap 4095/4095 MB, zero free), OOM-killed the trainer for the
fourth time and dropped it from 198k to 72k pos/s. The owner stopped the session
with a standing constraint: **no `lto="fat"` builds while training is live** —
that link step is the 4–5 GB spike — which rules out `tools/check.sh`,
`CHECK_NNUE_GATE=1` and `pgo16.sh` alike.

So this study reports the **thin-LTO probe frame only**, thin against thin, and
every predecessor's rule applies: study 12 §0.3's "no probe delta may be quoted
as a result" stands, and the probe has overstated the PGO delta by 1.12–2.0× in
each of studies 12–15. The honest statement of this study's result is therefore:

| | probe frame (thin LTO, 4 CGU, non-PGO) |
|---|---:|
| baseline (`main` @ `905dccd`, = study 15's published final line) | 6335.8 i/n |
| **shipped tree** | **6306.6 i/n** |
| **delta** | **−29.2** |
| node count, every line | 5,001,562 |
| eval hash, every line | `5ca978e012da1bb0` |

**Applying the predecessors' 1.12–2.0× band to −29.2 predicts −15 to −26 i/n in
the PGO frame, i.e. 1.3364× → about 1.3306–1.3331×.** That is a PREDICTION, not
a measurement, and it is written here so the owner's PGO run either confirms or
refutes it rather than merely replacing it.

**GATE STATUS: PENDING.** The full `CHECK_NNUE_GATE=1 tools/check.sh` **passed
on the pre-review tree** (arm `a8`) with all three pins unchanged —
`Bench: 587683`, eval hash
`5ca978e012da1bb086799709c3ea5350cf6d49923c254f7975b6d780b0098b38`, d13 nodes
`1922947`. The D33 review round then landed on top of it and cost **±0.00006
i/n** with the same node count and the same eval hash (arm `a9`), so the pins
are expected to reproduce; **they have not been re-run**, because doing so needs
the fat-LTO build this session may not do. §10.1 lists exactly what is owed.

### 0.2 What landed

| # | change | probe Δ | where |
|---|---|---:|---|
| 1 | `ContHistId`'s axes permuted so `piece_to()` is a **shift, not `% 832`** | −0.9 | `history.rs` |
| 2 | **the eager king-cross threat refresh writes in place** (a 2560-byte libc `memcpy` at 0.0357 calls/node) | **−14.6** | `sota_features.rs`, `sota_threats.rs` |
| 3 | `HalfkaContext::compute`'s two **provably-dead empty-record branches** deleted | **−8.7** | `sota_accumulator.rs` |
| 4 | five dead per-node `Option` tests collapsed behind two cached predicates | −5.0 | `sota_context.rs`, `eval/mod.rs` |
| | **total, probe frame** | **−29.2** | |

Every landed arm is node-identical on **three independent trees** — 5,001,562
(probe), 587683 (the default-feature `Bench:` pin), 1922947 (the d13 NNUE pin) —
and eval-bit-identical (`5ca978e012da1bb0`). All three gate pins reproduce
unchanged.

### 0.3 What this study is actually about

**(a) The `memset` class is EXHAUSTED, and that is the study's most useful
negative.** Study 15 §8 ranked it #1 at high confidence — "there are almost
certainly more of these" — and prescribed two instruments: extend `memspy.c` to
interpose `memset`, and bucket `objdump`'s `call *…mem*@` by enclosing symbol.
Both were built (§2) and both say the same thing: **269 `memset` calls over
150,059 nodes — 0.0018 per node, every one of them at start-up.** Not one is on
a search path. Study 15 got the only one there was. The lead cost two commands
to close and it is closed with a number.

**(b) The instrument study 15 asked for found something else instead.** With
`memset` hooked, the census's largest *hot* row is the one study 15 ranked #3
and sized at 4–8: the eager king-cross threat refresh returning 2560 bytes by
value, **0.0357 calls/node, 91 bytes/node, counted not estimated**. Writing it
in place measured **−14.6**, about **3× its own estimate** and half this study's
total. Study 15's rule 3 ("re-derive an inherited size") has now been paid in
both directions: its own items 2 and 5 were over-sized by 20× and by a sign,
and its item 3 was UNDER-sized by 3×.

**(c) The by-value mechanism does NOT generalise by size — it generalises by
whether libc gets called.** The same edit applied to the other by-value store on
the same path — `ThreatPosition::from_board`, 80 bytes, **1.0 calls/node**,
27× the call rate — measured **±0.0005 i/n** (§7.3). LLVM had already written it
in place. What the 2560-byte case was paying was not the copy; it was the
`call`. A future study should size this class by *interposer rows*, never by
bytes×rate.

**(d) Every study since #8 has divided a fixed per-process cost into its
per-node number, and nobody had measured it.** §7: **1.02 G instructions for
ARCHON and 1.06 G for Stockfish**, worth **68.0** and **70.5 i/n** of the
15,001,568-node headline bench. The reported ratio is therefore very slightly
*flattered* — the search-only ratio is **1.3423×**, not 1.3364×. It is a 0.4%
correction, it applies identically to studies 12–15, and it is now measured
rather than assumed.

---

## 1. METHOD

Studies 8–15's rules were followed.

### 1.1 NULL FIRST — the panel held, and it reproduced study 15 across a session

**Null A — rebuild determinism.** Same source, forced recompile
(`touch src/lib.rs`):

```
null-1           31,688,987,821   6335.8 i/n
null-2-rebuild   31,688,925,327   6335.8 i/n     62,494 apart, ±0.0125 i/n
```

`null-1` reproduces study 15's published final probe line — **6335.8 against its
6335.8** — a cross-study control on the whole instrument, for the second study
running.

**Null B — SOURCE PERTURBATION, structurally inert.** Study 14's null failed
informatively (+86.4) when it re-spelled `v.last().expect(…)` as `&v[n-1]`;
the lesson is *prefer a reborrow*. Null B here is a slice reborrow at the exact
two sites arm 1 was about to change (`let ss = &stack[..];` in front of the two
`cont_hist.piece_to()` pairs in `Histories::{correction_value,
update_correction_history}`):

```
n0-NULLB-stack-reborrow   31,688,949,583   6335.8 i/n     ±0.0076 i/n vs null-1
```

**±0.008 i/n.** Every arm below is a real effect of that arm, including the
three that made the engine slower.

A third control, unplanned: the revert of arm 5 reproduced the pre-arm tree to
**±0.0002 i/n** (`a5r-revert-control`).

### 1.2 Instruments

* **`probe16.sh`** — screening, inherited from `probe15.sh`. Thin LTO / 4
  codegen units, native, non-PGO, fixed 5,001,562-node tree; prints the wl50
  eval hash and node count on every line so a tree or eval change cannot hide
  inside a speed claim. **Every build runs inside its own `systemd-run --user
  --scope -p MemoryMax=4G -p MemorySwapMax=0` cgroup AND through
  `tools/oomlast.sh`** (§1.4). Build-and-measure cycle: **16 s**.
* **`memspy16.sh` + `memspy.c`** — study 14's `LD_PRELOAD` interposer with
  **`memset` added** (study 15 §8 item 1). §2.
* **`memstatic16.sh`** — the static census: `objdump -d` bucketed by enclosing
  symbol. New. §2.1.
* **`startup16.sh`** — the fixed-cost separator. New. §7.
* **`pgo16.sh` + `pstat8.sh`** — the deliverable frame.
* **`flat8.sh`**, **`sfshow.sh`** — inherited.

### 1.3 The box, honestly

A D48Y2 training run held the GPU at ~94% and ~2.2 GB RSS for the whole session,
and available memory sat between 2.4 and 5.3 GB. Everything here rides on
`instructions:u` over fixed-node trees and on deterministic call counts, both
contention-immune. **No wall-clock or NPS number is quoted as a claim.**

### 1.4 The OOM guard

Every build in this study ran as
`systemd-run --user --scope -p MemoryMax=<N> -p MemorySwapMax=0 -- tools/oomlast.sh …`.
`oomlast.sh` raises `oom_score_adj` to 1000 before `exec`, so cargo and rustc
inherit it and the kernel picks the build rather than the trainer;
`systemd-run -p OOMScoreAdjust=` does not work for user scopes. Nothing was
OOM-killed and the training run never noticed.

---

## 2. THE MEMSET CLASS — closed, with two instruments and one command each

### 2.1 The static census: nothing left

`memstatic16.sh` buckets every `call *…<mem(set|cpy|move)@…>` in the release
binary by enclosing symbol. On the study-15 tree the only `memset` inside
`src/` at all is **`SotaContext::new`** (twice — a constructor). The survivors
study 15 predicted for `SotaContext::evaluate` are not there: its arm 4 had
already removed them by threading the shared `HeadScratch` through the tier
paths.

### 2.2 The dynamic census: 0.0018 memset calls per node, all of them start-up

`memspy.c` now interposes `memset` as well as `memcpy`/`memmove`, tagging each
row with its kind (`perf-artifacts-16/memspy-baseline.txt`, 150,059 nodes):

```
memcpy : calls=24,393  bytes=94,893,562  per_node=0.1626
memmove: calls=   129  bytes=   325,120  per_node=0.0009
memset : calls=   269  bytes=18,369,032  per_node=0.0018
```

18.4 MB in 269 calls is the history tables and the TT being zeroed once. **Zero
`memset` calls come from a search path.** Study 15's item 1 is closed.

One implementation note for whoever inherits the tool: the interposer's
replacement bodies cannot call libc's `mem*` (infinite recursion), and the
obvious byte loop is unusable — the first attempt ran a 3000-node bench for over
ten minutes without finishing, because start-up zeroes ~20 MiB. Word-at-a-time
(`__builtin_memcpy` of a `u64`) brings it back to seconds.

### 2.3 What the instrument found instead

The census's hot rows, per node:

| calls/node | bytes | caller |
|---:|---:|---|
| 0.0662 | 2.4 | `Worker::iterative_deepening` (a **libc call for 2.4 bytes**) |
| 0.0312 | 2.2 | `Worker::search::<Pv>` |
| **0.0188** | **2560** | **`ThreatContext::apply_at` +0x678** |
| **0.0169** | **2560** | **`ThreatContext::apply_at` +0x34c** |
| 0.0008 | 2560 | `HalfkaContext::finny_refresh_into` (×3 rows) |

The two `apply_at` rows are exactly study 15 §8 item 3 — **0.0357 calls/node,
91 bytes/node** — and they are §3.

---

## 3. THE KING-CROSS REFRESH — 2560 bytes by value, −14.6

`ThreatContext::apply_at` rebuilds a perspective's whole threat accumulator when
that perspective's king crosses the d/e mirror:

```rust
self.stack[child].acc[pi] = refresh_threat(weights, board, persp);
```

`refresh_threat` returns `[i16; ACC]` — **2560 bytes**. So the assignment builds
the accumulator in a stack local and then hands 2560 bytes to libc. The fix is
the one `HalfkaContext::finny_refresh_into` already uses for the HalfKA twin: an
in-place writer.

```rust
pub fn refresh_threat_into(dst: &mut [i16; ACC], w: &[i8], board: &Board, persp: Color) {
    dst.fill(0);
    …
}
```

`refresh_threat` stays, as a thin wrapper for the oracle and the tests, so no
caller outside the two hot sites changes.

**−14.6 i/n**, node-identical and eval-hash-identical.

### 3.1 It did NOT remove the libc call — and the census says so

The post-arm census (`memspy-final.txt`) has no `ThreatContext::apply_at` row at
all. It has a new one:

```
5,452 calls  13,957,120 bytes  2560.0 B  0.0363/node  memset  refresh_threat_into +0x2f
```

**Same size, same rate, different libc function.** `dst.fill(0)` on a
`&mut [i16; 1280]` is above LLVM's inline threshold, so it compiles to a call —
`memset` where there used to be a `memcpy`. Total libc traffic is **757.0
bytes/node before and 757.0 after**: unchanged to the tenth of a byte.

So the arm did not win by removing a call. It won by removing a **pass**:

| | before | after |
|---|---|---|
| zero a 2560 B buffer | inline, in the stack local | libc `memset`, in the entry |
| accumulate the threat columns | into the stack local | into the entry |
| copy to the entry | **libc `memcpy`, 2560 B** | — |

One full 2560-byte pass per king cross, gone; the call stayed. That is the whole
−14.6, and it means **the remaining `memset` is still on the table** — §12
item 2. Reviewer C asked exactly this question in review ("does `dst.fill(0)`
re-materialise a `memset`?"); the honest answer needed the interposer, and the
first draft of this section asserted the opposite from a static census taken
before the review round moved the function out of line.

Two spellings are worth trying, in this order: study 14 §3.1's winning idiom
applied to a fill rather than a copy —
`for c in dst.chunks_exact_mut(32) { c.fill(0) }`, 40 chunks of 64 bytes, each
below the inline threshold — and, better, fusing the zero into the first
`add_column` so the buffer is *written* once instead of zeroed and then
accumulated.

The estimate study 15 handed over was 4–8. The measurement is 3× that. §6.3
explains why the estimate was low, and it is not the reason anyone would guess.

---

## 4. THE DEAD EMPTY-RECORD BRANCHES — −8.7

Both of `HalfkaContext::compute`'s fold loops carried:

```rust
if dirty.adds().is_empty() && dirty.subs().is_empty() {
    entry.acc[p] = parent.acc[p];
    entry.psqt[p] = parent.psqt[p];
} else { apply_dirty_into(…); }
```

For HalfKA a record is empty only for a null move — every real move has at least
one add and one sub, pinned by this module's own tests — and **since study 15
arm 1 a null move pushes no accumulator entry at all.** `stack[0]`'s record is
never read either: both loops start at `ancestor + 1 >= 1`. So the branch was
taken **0× per node**.

Study 14 §3.1 fought the analogous deletion in the THREAT fold and lost by
+22.7, and study 15 explicitly warned not to repeat that. The two cases are not
the same and the difference is the whole argument: study 14's branch was **taken
31% of the time**, so deleting it pushed live traffic through a kernel that then
re-materialised a dispatch. This one is not taken, so its `else` arm is
unchanged and only the test disappears.

**−8.7 i/n and −3.0 branches/node.** A `debug_assert!` replaces each branch, and
§9 shows the assert is *reached* — inverting it fails the suite — so the
premise is measured, not argued.

**Do NOT do the same to `ThreatContext`'s `add_len == 0 && sub_len == 0`.** That
one is live (a real move whose relational threat events cancel). Study 15 said
so and this study did not test it.

---

## 5. THE `% 832` — −0.9, AND THE ESTIMATE WAS 22× TOO BIG

`ContHistId` flattens `(in_check, capture, piece13, to)` into a `u16` slab
index. The continuation-*correction* history is keyed by the `(piece13, to)`
sub-index alone, recovered as `flat % 832` — a non-power-of-two modulus, which
LLVM lowers to a multiply-shift-multiply-subtract chain. Study 15 §8 item 5
sized re-permuting the axes so the accessor becomes a shift at **~20 i/n**.

The permutation is free — `(piece13*64 + to)*4 + (in_check*2 + capture)` is the
same bijection onto `0..3328`, nothing outside the impl observes the ordering,
and `SENTINEL` just moves 768 → 3072.

**It measured −0.9 i/n**, 110× the null so it is resolved, and **22× smaller
than its estimate.** The estimate assumed four calls per node; the truth is
twice per *evaluated* node in `correction_value` plus twice more only on nodes
whose correction history is actually updated, which is gated three ways.

Kept, because it is also the simpler code and because the round produced a
strictly better test: the old bijection test asserted `id.0 == expected` against
a nested sweep counter — i.e. it asserted the axis ORDER, a free implementation
choice, and would have blocked this arm for no reason. It now asserts the two
properties the search depends on (in range, no collisions, full cover).

**Stockfish pays nothing here, for two reasons we cannot copy** (D33 reviewer
B). It stores *two independent pointers* on the search stack —
`ss->continuationHistory` and `ss->continuationCorrectionHistory`,
`eca43a97:src/search.cpp:650-653` — and never derives one index from the other.
And its `PIECE_NB` is **16** (`eca43a97:src/types.h:210-215`), so every
`[piece][to]` flatten is `(pc<<6)|to`. **Our 13-wide packed piece axis is the
sole reason a modulus ever existed**; SF's memory order is
`[inCheck][capture][piece][to]` — exactly the "natural" order this arm
abandons, and it is free for them. The SF-shaped alternative, not priced: widen
our piece axis to 16 and keep the natural order, making `piece_to()` a mask, at
`cont` 4.88 → 6.0 MiB and `cont_corr` 1.22 → 1.5 MiB.

---

## 6. FIVE DEAD `Option` TESTS PER NODE — −5.0

`SotaContext::push` ran three `Option` tests per node for the smallnet and
largenet tiers, and `EvalContext::push_sota_dirty` two more for the experimental
side nets. All five are `None` in every shipped configuration. Two cached
predicates — `SotaContext::tiers_armed` and `EvalContext::side_nets` — collapse
them to two.

**−5.0 i/n and −3.0 branches/node**, matching study 15's 10–15 estimate at the
low end.

A cached predicate that is wrong here does not crash — it silently skips an
accumulator push and the affected net then evaluates from stale lanes. So both
caches are pinned three ways: a `checked()` postlude on every `SotaContext`
constructor, a `debug_assert_eq!` at the consumer (`push_sota_dirty`, which runs
at every node of every debug search), and — for `side_nets`, which no test in
the repo could previously reach at all — a new unit test that drives the flag
through both setters with a leaked fixture net (§9).

---

## 7. WHAT DID NOT LAND

### 7.1 `upcoming_repetition`'s `other != 0` early-out — **+17.8 AND NODE-CHANGING** (study 15 §8 item 8, REFUTED)

Stockfish keeps an incremental XOR of the odd-distance key deltas and skips the
two cuckoo probes when it is non-zero
(`eca43a97:src/position.cpp`, `upcoming_repetition`). We do not, and study 15
sized the port at ~10 i/n.

Translated verbatim into our post-study-14 state shape, it measured **+17.8 i/n
— and changed the node count, 5,001,562 → 5,001,624.** A node-count change makes
it a functional change, not a speed change, so it is out regardless of its sign;
and its sign is wrong too, because our windows are short (`end < 3` returns
before the loop on most nodes) so the extra two loads and three XORs per
iteration are paid without the probes being saved often enough.

**Why the node count moved IS established — by the reference's history, not by
the reference's code** (D33 reviewer B, and this is the review round's largest
finding). SF's `other` filter did not arrive on its own. It landed in commit
`6b782211`, which in the SAME patch **renamed `has_game_cycle` to
`upcoming_repetition` and deleted the mover-colour test** — the
`color_of(piece_on(s1 ^ s2)) == sideToMove` guard that at/above the root
distinguishes "a cycle the side to move can steer into" from "the move that
merely arrives here".

**Our function is a port of the PRE-`6b782211` shape.** We still carry that
guard (`board.rs`: `let mover = if self.piece_on(s1).is_some() { s1 } else { s2 };`
… `!= Some(self.side_to_move)`). Porting `other` *alone* into it therefore
cannot be behaviour-preserving: the filter was co-designed with the semantics
the same commit removed. The +62 nodes are the size of that difference on this
tree.

And there is a second, independent reason the port loses on speed *for us and
not for them*, which is our own shape and not SF's: SF walks a linked list, so
reading `stp->previous` already pulls the intermediate state's cache line, and
`key`/`checkersBB`/`previous` share it. We walk an index into a `Vec<StateInfo>`
whose stride is 168 bytes, and our loop touches only the ODD-distance states.
`other` forces us to touch the even ones too — **doubling the states loaded, on
distinct cache lines, to save the same two cuckoo probes.** Study 14's state-stack
rewrite inverted this trade without anyone noticing, which is a nice
demonstration that a reference-faithful port is only faithful against the
reference's *data structures* as well as its code.

The right move is therefore not to salvage the filter. It is to take
`6b782211` **whole**, as a functional change arbitrated by games — SF shipped it
with a non-regression run and it also fixes a PV-correctness issue. Our current
`ply > i` branch can report a draw the side to move cannot actually force, which
under the standing NO-ACCEPTABLE-BLUNDERS bar is a defect class, not a perf
item. §12 item 4.

### 7.2 `capture_stage`'s jump table — priced at **+6.4 and +0.7**, for −2.0 branches/node

`Board::capture_stage` is a four-way `match` on `MoveKind` run on every move
made and every move movepick scores. LLVM builds a **jump table** for it: a
`lea`, a `movslq`, an `add` and an **indirect jump**, visible in `do_move`'s
disassembly at `+0x61`.

Two spellings removed it. Peeling the dominant `Normal` arm to a straight-line
early return: **+6.4**. Hoisting the occupancy read out of the match so all four
arms are uniform: **+0.7 — and −2.0 branches per node**, i.e. the jump table
really is gone.

So the trade is real and it is priced: **2.0 branches per node are available
here for 0.7 instructions.** This study's metric is instructions, so it does not
land. Anyone attacking our branch ratio (now ~1.6× Stockfish's) should start
here, and should bring a cycles instrument.

**But neither spelling was Stockfish's**, and reviewer B found three shape
differences that together are a better third spelling than either of mine
(`eca43a97:src/position.h`, `capture_stage`):

* SF collapses two of the four arms — it tests `m.type_of() == EN_PASSANT`
  rather than enumerating `EnPassant` and `Castling` separately, so it has
  **three** cases where we have four. Four is what makes LLVM reach for a jump
  table.
* SF's `type_of()` is a **mask** on the move word; ours is an enum
  discriminant that must be materialised.
* **`m.promotion_piece()` performs a SECOND `kind()` dispatch and returns an
  `Option`.** SF's `promotion_type()` is raw bit arithmetic. That is very
  likely most of spelling one's +6.4: peeling the `Normal` arm left the
  promotion arm re-deriving the kind it had just tested.

A `promotion_piece_raw()` that skips the re-dispatch, plus the three-case
shape, is the arm study 17 should run before concluding anything about this
site.

### 7.3 `ThreatPosition` written in place — **±0.0005, and it is the study's best lesson**

`ThreatPosition::from_board` returns 80 bytes and its result is stored into
`stack[child].position` at **1.0 calls per node** — 27× §3's call rate. Same
mechanism, same edit, and it measured **±0.0005 i/n**: LLVM had already written
it in place.

Which is what §3 was actually about. 80 bytes is below the inline threshold, so
there was never a `call`; 2560 bytes is above it, so there was. **The by-value
class is sized by whether libc gets called, not by bytes moved.** The
interposer's rows ARE the class; a bytes×rate estimate is not. That is why
study 15's estimate of 4–8 was 3× low — it priced the copy.

Reverted, to keep the diff to what pays.

### 7.4 `attacks::tables()` — the static census cannot find it (study 15 §8 item 4, UNRESOLVED)

Study 13 counted 5.62 `LazyLock` derefs per node with an instrumented build and
sized the guard at 17–22 i/n; study 14 and 15 both re-ranked it without trying
it. Before writing the (large) `const`-evaluated-magics arm, this study looked
for the guard in the shipped disassembly.

**There is no reference to `chessengine::attacks::TABLES` anywhere in the
binary's disassembly, and no reachable reference to its `Once` slow path from
any hot symbol.** That is not proof the guard costs zero — the probe build
routes many calls through the GOT, which hides the symbol name (§10) — but it is
enough that the 17–22 estimate should not be spent against. It needs a counter
build to resolve, and study 13 §11.4 already warned that a counter build
perturbs exactly the inlining this question turns on.

### 7.5 `Worker::do_move`'s spill — read, and the only spelling is a wash (study 15 §8 item 2)

Study 15 ranked `do_move` #2 (~300 i/n) and said "price the *spill*, not the
derivation". The spill is now read rather than estimated. In `do_move`'s
disassembly the `DirtyPiece` costs **ten memory operations** purely to survive
the call to `make_move_with`: five to park it in a callee-preserved slot before,
five to restore it after. Plus a redundant `cmp $0x2` on the `Option<DirtyPiece>`
niche discriminant — byte 0 is a `Color`, so `2` is `None`, on a value the
control flow has already proved is `Some`.

The spelling that removes the park — hold the record in an `EvalContext` field
instead of a stack temporary — moves the same 14 bytes to the same kind of
memory. It saves the *double* copy (about five operations) and costs a field
that reads as a state machine. Not attempted; sized at ~5 and recorded so study
17 does not re-derive it.

---

## 8. THE FIXED COST INSIDE THE BENCH — 68.0 i/n of ours, 70.5 of theirs

Every study since #8 quotes `instructions / nodes` over a `bench` run, so
whatever the process does before it searches is divided into the per-node
number. `startup16.sh` separates the constant from the slope with two runs at
different node budgets (`perf-artifacts-16/startup.txt`):

| | small run | big run | per-node slope | **FIXED** | of the headline bench |
|---|---|---|---:|---:|---:|
| ARCHON | 1,853 n / 1.032 G | 1,000,820 n / 7.168 G | 6142.5 | **1,020,273,491** | **68.0 i/n** |
| SF `eca43a97` | 1,712 n / 1.065 G | 1,000,484 n / 5.575 G | 4515.8 | **1,056,952,172** | **70.5 i/n** |

Both engines pay about a gigainstruction to start (net load, magic/table
generation, TT and history allocation). Corrected for it, the search-only rates
and ratio are:

* ARCHON 5950.1 − 68.0 = **5882.1**
* SF 4452.4 − 70.5 = **4381.9**
* **search-only ratio 1.3423×**, against the reported 1.3364×

So the headline understates the true gap by 0.4%, because *Stockfish's*
start-up is the larger of the two. The correction applies identically to studies
12–15 and does not change any of their conclusions. Reported here so the number
is measured rather than assumed; the headline table keeps the established
whole-bench convention so the five studies stay comparable.

---

## 9. THE ARM LOG

`probes-fast.txt`, in order, probe frame. Node count **5,001,562** and eval hash
**`5ca978e012da1bb0`** on every line except the one that is the point of §6.1.

```
null-1                       6335.8    baseline, main @ 905dccd (= study 15's published 6335.8)
null-2-rebuild               6335.8   +-0.0125  rebuild null
n0-NULLB-stack-reborrow      6335.8   +-0.0076  structural source-perturbation null
a1-conthist-shift            6334.9     -0.9   LANDED (S5 / history)
a2-refresh-threat-in-place   6320.3    -14.6   LANDED (S3)
a3-halfka-drop-empty-branch  6311.6     -8.7   LANDED (S4)
a4-tier-and-sidenet-flags    6306.6     -5.0   LANDED (S5)
a5-uprep-other-earlyout      6324.4    +17.8   REVERTED (S6.1) -- nodes 5,001,624, FUNCTIONAL
a5r-revert-control           6306.6   +-0.0002 revert control
a6-capture-stage-peel        6313.0     +6.4   rejected (S6.2)
a6b-capture-stage-hoist-occ  6307.3     +0.7   rejected (S6.2) -- but -2.0 branches/node
a7-threatpos-in-place        6306.6   +-0.0005 neutral, reverted (S6.3)
a8-shipped-tree              6306.6    -29.2   pre-D33-review; the tree the gate PASSED on
a9-post-D33-review           6306.6   +-0.00006 the SHIPPED tree, after the review round
```

The D33 round is free to six decimal places here, which is unusual — studies 13
and 15 paid +5.7 and +1.2 for theirs. The reason is that every fix this round
produced is a comment, a `debug_assert`, a test, or a constructor postlude that
the release build folds away (§10.2); the one release-visible change,
`with_tier_flag` recomputing rather than asserting, runs once per context
construction.

---

## 10. EACH LANDED CHANGE HAS A TEST THAT CAN FAIL, AND EACH WAS MUTATION-VERIFIED

Seven mutations, each applied, built, run and reverted
(`perf-artifacts-16/mutations.txt`). **All seven failed a test.**

| change | test | mutation applied — all FAILED |
|---|---|---|
| `ContHistId` axis permutation (a1) | `cont_hist_id_is_the_flat_index_bijection` (rewritten), `cont_hist_id_sentinel_is_no_piece_a1`, three `const _: ()` asserts | **M1** `piece_to()`'s divisor 4 → 2; **M2** `ContHistId::new` reverted to the old axis order while `piece_to()` stays a shift |
| `refresh_threat_into` fully overwrites `dst` (a2) | the threat replay/refresh oracles | **M3** move `dst.fill(0)` behind the `weights.is_empty()` early return |
| the deleted empty-record branches were dead (a3) | the new `debug_assert!`s, reached at every node of every debug search via `assert_matches_refresh` | **M5** INVERT the tripwire (assert the record *is* empty) — fails, so the path is reached and the un-inverted assert is load-bearing rather than vacuous |
| `tiers_armed` cache (a4) | `checked()` on every constructor | **M4** `tiers_armed: false` in the large-tier constructor |
| `side_nets` cache (a4) | **`side_net_flag_tracks_the_optional_contexts` — NEW** | **M6** `refresh_side_net_flag` always `false`; **M7** drop the `refresh_side_net_flag()` call from one setter |

**The `side_nets` test is new because nothing in the repo could reach that code
at all.** `set_simple_halfka_net` and `set_protected_halfka_net` have exactly one
caller each, both in `Worker`, both driven by a global that no test sets — so
the *installed* branch of both setters had never executed, in any test, ever.
That is the same hole study 15 §9.3 found for the SOTA accumulators and study 14
§7.1 found for `has_repeated`, one layer out again. The fixture is a
`SimpleHalfkaNetwork` with empty weight tables, leaked to `'static`;
`SimpleHalfkaContext::new` only stores the pointer, so nothing is ever indexed.

The bijection test was also **rewritten, not just re-baselined**. It used to
assert `id.0 == expected` against a nested sweep counter — i.e. it asserted the
axis ORDER, which is a free implementation choice, and would have blocked this
arm for no reason. It now asserts the two properties the search actually depends
on: every id is in range, and no two inputs collide (a `seen` vector over
`CONT_NB`). It is strictly stronger and encoding-independent.

---

## 10.1 WHAT IS OWED BEFORE THIS MERGES

Run in this order, on an idle box, once training finishes:

1. `CHECK_NNUE_GATE=1 CHECK_TEST_THREADS=4 bash tools/check.sh` — expect
   `Bench: 587683`, eval hash `5ca978e0…0b38`, d13 nodes `1922947`, all
   unchanged. **A change in any of the three is a bug in this study, not a new
   pin**: every arm here is supposed to be node-identical, and the probe frame
   already says all three trees agree.
2. `bash docs/research/perf-artifacts-16/scripts/pgo16.sh s16` then the two-arm
   panel against `engines/bin/stockfish-eca43a97`
   (`pstat8.sh 5 300000 tools/nnue/calib/wl50.fens counters-headline.txt …`) —
   the deliverable frame, and the test of §0.1's −15…−26 prediction.
3. `bash docs/research/perf-artifacts-16/scripts/memspy16.sh final 3000` on the
   PGO binary — the thin-LTO run is already committed
   (`memspy-final.txt`): the two `ThreatContext::apply_at` rows are gone and a
   `refresh_threat_into` `memset` row took their place (§3.1). Worth repeating
   on the PGO binary, because inlining decisions are exactly what differ.

Nothing in 1–3 can change the code; they are all measurements of a tree that is
already committed.

## 10.2 WHY THE THIN FRAME IS ENOUGH TO LAND ON

Node identity, not speed, is what makes these changes safe, and node identity is
frame-independent. Three trees agree, all measured:

* the 5,001,562-node probe tree, on **every** arm line including the rejected
  ones (§9);
* the wl50 eval hash `5ca978e012da1bb0`, likewise on every line;
* the pre-review gate's `Bench: 587683` and d13 `1922947`.

and 692 tests pass under `--features sota,arch-d48x`, seven of them
mutation-verified (§10). The PGO run measures how much this is worth; it cannot
change whether it is correct.

---

## 10.3 THE D33 REVIEW ROUND

Three read-only reviewer personas ran against the landed diff (A
correctness/safety, B reference-fidelity vs `eca43a97`, C Rust-idiom/perf/
test-quality). **One blocker, one silent-defect finding, one real perf bug, and
the explanation of §7.1.** Cost: ±0.00006 i/n.

**The blocker (C).** `refresh_threat` became unused outside tests once arm 2
landed, so `cargo clippy --all-targets --features sota,arch-d48x` failed on the
import. It is now `#[cfg(test)]`. Note the gate would NOT have caught this:
`tools/check.sh` clippies **default features only** (study 15 §8 item 12, still
open) — the default-feature clippy caught a *different* unused import in the new
test module, in the harmless direction, while the `sota`-only one sat unchecked.

**The silent-defect finding (A).** `side_net_flag_tracks_the_optional_contexts`
— the test written specifically to close study 15's "no test can fail" hole —
**could not catch dropping `refresh_side_net_flag()` from
`set_protected_halfka_net`.** It only ever passes that setter a `None`, and with
`simple` armed the expected flag is `true` either way. That is the dangerous
direction: `Worker::refresh_eval_net` installs the protected net LAST, so on a
protected-only configuration a stale `false` makes `push_sota_dirty` skip the
protected push for a whole search — a wrong eval, in release, with no crash.
`side_net_flag_arms_from_the_protected_setter_alone` covers that leg.
**A test written to close a coverage hole can have a coverage hole.**

**The perf bug (C).** Arm 2's `refresh_threat` wrapper zeroes its local and then
`refresh_threat_into` zeroes it again — 2 × 2560 bytes of `memset` per call. It
was on the debug oracle's `refresh_both` path. Both are fixed: `refresh_both`
writes in place, and the wrapper is test-only.

**The shape fix (C).** `checked()` was a `debug_assert`-only postlude, which is
invisible in release AND unreached by any test, since no test arms a tier
through `SotaContext::new` — i.e. the mutation I verified it with (M4) was the
only thing that ever exercised it. It is now `with_tier_flag()`, which
**recomputes** the field; no constructor sets it, so the disagreement M4
describes is no longer expressible.

**Also taken:** `debug_assert!(flat < CONT_NB)` in `ContHistId::new` (A — the
old `% 832` fail-soft'd any corrupt id back into range and `/ 4` does not, so
the invariant is now stated where it is established); a `LazyLock` fixture in
place of `Box::leak` so the "same net" path genuinely takes the `reuse` branch
(C); the backward fold's own bound, which the forward fold's argument does not
cover (B); and `design/10`, which still carried the old `ContHistId` formula (A).

**Deliberately not taken:** deleting the tier machinery outright, which is
Stockfish's own answer (B — SF has no optional sub-net at `eca43a97`; the small
net was removed in `ab8f901d`). It is recorded at the site and in §12, because
it takes all five dead tests rather than three, but it deletes a live research
surface and is not a perf commit's business.

---

## 11. THREATS TO VALIDITY

1. **Probe deltas overstate PGO deltas** — studies 12–15 measured the factor at
   1.12–2.0×. **In this study NOTHING is PGO-measured** (§0.1), so the −29.2
   total is a probe-frame number and the PGO figure is a prediction with a
   1.7× spread. This is the study's largest threat and it is a machine
   constraint, not a methodological choice.
2. **The probe build's CALL STRUCTURE is not the shipped build's**, and this
   study is the first to check. Thin LTO with 4 codegen units emits **4,429
   indirect calls** against the PGO (fat LTO, one CGU) binary's **1,288** —
   including hot ones: in the probe, `do_move` reaches `DirtyPiece::from_move`
   and `EvalContext::push_sota_dirty` through the GOT, where the PGO binary
   calls both directly. Most of the difference is cold (panic and libc edges),
   but **an arm whose benefit is a change in call structure will mis-price in
   the probe frame.** This is a concrete mechanism for the 1.12–2.0× band the
   predecessors observed empirically, and it is why §6.4 could not be settled
   statically.
3. **§7.1's node-count change is now explained by the reference, not by us.** The explanation is reviewer B's, arrived at by reading the SF commit that introduced the filter; the study's own §7.1 draft called it unexplained. Read that as a statement about how far a measurement gets you without the reference's history.
4. **The memspy counts are from a 150,059-node bench.** Call *rates* per node are
   what is quoted; absolute totals are for that tree only.
5. **Box state**: a training run shared the box for the whole session (§1.3). No
   wall-clock or NPS number is quoted as a claim.
6. **§7's fixed-cost split is a two-point linear fit**, and `bench` is not
   perfectly linear in nodes (the small run is 1,853 nodes, so per-position
   overheads are over-represented in it). The two engines' constants agree to
   3.5%, which is the main reason to believe the fit.

---

## 12. RANKED REMAINDER — handoff to study 17

**Do not reopen study 13 §0's closed list** (eval head, search body, movepick,
threat generation, accumulator fold — the last is the 1280-vs-1024 width, a
**retrain**, ~327 i/n), and add this study's closures to it.

**CLOSED BY THIS STUDY, with numbers:** study 15 item 1 (the `memset` class —
0.0018 calls/node, all start-up), item 3 (landed, −14.6), item 6 (landed, −5.0),
item 8 (`upcoming_repetition` — +17.8, node-changing, and now EXPLAINED — §7.1), item 10 (landed, −8.7).

| # | item | size | confidence | cheapest next test |
|---|---|---:|---|---|
| 1 | **Pack `Piece` into ONE byte.** `Piece { color: Color, piece_type: PieceType }` is **2 bytes**, so `mailbox: [Option<Piece>; 64]` is **128 B = two cache lines** and a full `piece_on` is **two `movzbl`s** — visible in `do_move` at `+0x4e`/`+0x56`. Stockfish's is one byte, one line, one load. Hits BOTH of study 13's largest non-parity blocks (make/unmake +359, board queries +113) | **5–20, revised DOWN from 20–40** | medium | **Re-derived before handing over, per this study's own rule, and the first number was too big.** A static census of every scale-2 `movzbl` in the shipped binary, paired by adjacent offset off a common base+index: **40 of 75 sites (53%) read both bytes; 35 read ONE** — a colour-only, type-only or `is_some` test. So the load-halving applies to about half the sites, and at the other half a packed byte may COST an extract. What survives cleanly is `index()`: currently a `cmp`+`lea`+`cmove` off the two bytes (see `do_move` `+0xb3`), free if the byte IS the index. **Cheapest next test, before any refactor: a counter build that separates paired from single mailbox reads PER NODE** — static site counts are not rates, and study 13 §11.4 warns a counter build perturbs inlining, so counter the CALL not the load. Only then decide between SF's `color*8 + pt` (colour/type one instruction each, `index()` a 16-byte lookup) and a dense `color*6 + pt` (`index()` free, `piece_type()` a lookup). Scope: **101 `piece_on` sites, 128 `Piece::new`, 60 `.piece_type`, 29 `.color`, 13 struct literals**; touches `movegen`/`board` ⇒ full D24 perft ritual |
| 2 | **The `memset` arm 2 left behind.** `refresh_threat_into`'s `dst.fill(0)` is a **2560-byte libc `memset` at 0.0363 calls/node** (`memspy-final.txt`, §3.1) — the same rate and size as the `memcpy` arm 2 removed. Arm 2 won by deleting a pass, not the call | ~5–10 | **high — counted, and arm 2 already priced the twin at −14.6** | two spellings, in order: `for c in dst.chunks_exact_mut(32) { c.fill(0) }` (study 14 §3.1's winning idiom, applied to a fill); then fuse the zero into the first `add_column` so the buffer is written once rather than zeroed and accumulated. One `probe16.sh` cycle each — 16 s |
| 2b | **The 2.4-byte libc `memcpy`.** `Worker::iterative_deepening` calls libc `memcpy` **0.0662 times per node for 2.4 bytes**, and `search::<Pv>` 0.0312 times for 2.2 (`memspy-baseline.txt`). A call, a dispatch and a `vzeroupper` to move two bytes | 2–4 | **high — counted** | find the two sites (`objdump` offsets are in the census) and make the payload a typed store. Small, but it is the only remaining *hot* interposer row after §3 |
| 3 | **`capture_stage`'s jump table, for BRANCHES not instructions** | −2.0 branches/node for +0.7 instructions | **high — priced, §7.2** | already implemented and measured; needs a cycles instrument and an idle box to decide. Our branch ratio is ~1.6× SF's and this is the cheapest known lever on it |
| 4 | **Land Stockfish's `6b782211` WHOLE** — the `other` filter AND the semantics change it shipped with | ~10 i/n at best; **a correctness class** | **high — §7.1, D33 reviewer B** | NOT a perf arm. Our `upcoming_repetition` is the pre-`6b782211` shape and its `ply > i` branch can report a draw the side to move cannot force. Take the whole commit and arbitrate with games, not with `instructions:u`. Note the speed half is likely still negative for us: our 168-byte-stride indexed state walk makes `other` touch twice the cache lines SF's linked list already had in hand (§7.1) |
| 5 | `Worker::do_move`'s spill | ~5 | medium | §7.5. Read, not estimated. The only spelling that removes the double copy trades a stack slot for a struct field |
| 6 | **The FULL Stockfish transposition** — *replace* `pieces[c][pt]` with `byTypeBB & byColorBB` | ? | medium | study 14 §4.6/§8 item 4, unchanged. Note it interacts with item 1: do them in one study or the second will re-do the first's work |
| 7 | `attacks::tables()` | ? — **the 17–22 estimate is unsupported** | low | §7.4. Needs a counter build, and study 13 §11.4 says a counter build perturbs the inlining the question turns on. Cheapest honest test: build with the `LazyLock` replaced by a `const` for the *leaper* tables only (no magic search needed) and see whether anything moves |
| 8 | `Board::clone()` loses the `history` reserve | 0 i/n; a latency spike | low | unchanged from study 14 item 10 |
| 9 | **Move the accumulator cursor INTO the contexts** | 0 i/n; a defect class | high | unchanged from study 15 item 11 — SF's `AccumulatorStack::size` makes a wrong index unrepresentable; our `acc_ply` makes it merely wrong at four sites. This study added two more cached predicates in the same spirit (§5), each with a tripwire, which is a patch on the shape rather than a fix of it |
| 10 | **`tools/check.sh` clippies DEFAULT features only** | 0 i/n | high | unchanged from study 15 item 12. It bit this study: the gate's *default-feature* clippy rejected a `use super::*` that is only used under `sota`, which is the failure mode in the harmless direction — the dangerous direction (a lint that only fires WITH `sota`) is still unchecked |

**Four method rules this study hands over.**

1. **Close the inherited lead first, and close it with the instrument the
   handoff named.** Study 15 ranked the `memset` class #1 at high confidence.
   Two commands said it was empty (§2). That negative cost twenty minutes and
   redirected the whole session; carrying it as "probably the biggest thing"
   for another study would have cost far more.
2. **An estimate built from bytes×rate is not an estimate of the same quantity
   the interposer measures.** §3 was under-sized 3× and §6.3 was over-sized
   infinitely, by the *same* arithmetic, because what costs is the `call`, not
   the bytes. When a class has a counting instrument, its rows are the class.
3. **Re-derive your OWN handoff sizes too, not just the ones you inherit.**
   This study's ranked #1 was written at 20–40 i/n from an arithmetic that
   assumed every `piece_on` reads both bytes. A five-minute static census of
   the binary said 53%. The rule study 15 handed over is about inherited
   numbers; it applies with equal force to the number you are about to hand
   over, and a handoff item is exactly where an unchecked estimate does the
   most damage — the next session spends a day against it.

4. **Check that your screening build has the same CALL STRUCTURE as your
   deliverable build.** Four studies inferred the 1.12–2.0× probe-overstatement
   factor from its effects; it took one `objdump | grep -c 'call \*'` on both
   binaries to find a mechanism (§11.2). Do that once per instrument, not once
   per study.
4. **A node-count change ends the argument.** §6.1 had a clean reference
   citation, a plausible derivation and an unexplained +62 nodes. The +62 is the
   finding; the derivation is a hypothesis. Reverting a reference-faithful port
   because the arbiter says it is functional is the correct move, and the
   unexplained part becomes a ranked item rather than a footnote.

---

## 13. ARTIFACTS (`perf-artifacts-16/`)

* `probes-fast.txt` — every arm with its eval hash and node count, including
  both nulls, the revert control, the three rejected arms and the neutral one.
* *(no `counters-headline.txt` — the PGO panel is owed, §0.1)*
* `memspy-baseline.txt`, `memspy-final.txt` — the `memcpy`/`memmove`/**`memset`**
  census by caller and calls-per-node, before and after. Read them together:
  total bytes/node is **identical** at 757.0, which is §3.1's point.
* `startup.txt` — §7's fixed-cost split for both engines.
* `flat-s16base-instructions.txt` — instruction-attributed flat profile.
* `mutations.txt` — the seven mutation-verification runs.
* *(no `gate-s16.log` — the pre-review gate passed but its log was overwritten
  by the run the OOM event killed; the gate is owed, §0.1/§10.1)*
* `scripts/memspy16.sh`, `memstatic16.sh`, `startup16.sh`, `probe16.sh`,
  `pgo16.sh` — new or updated; `memspy.c` now hooks `memset`.
* `scripts/pstat8.sh`, `parse8.py`, `flat8.sh`, `annot8.py`, `opcount11.py`,
  `blocks13.py`, `sfshow.sh` — inherited.
