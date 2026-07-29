# ARCHON perf study #17 — the move-making side is at parity, and the block that is not was declared closed

Date: 2026-07-29. Seventeenth commissioned study (14:
`perf-study-14-state-and-memcpy-2026-07-29.md`, 15:
`perf-study-15-null-entry-and-dead-memset-2026-07-29.md`, 16:
`perf-study-16-the-memset-class-closes-2026-07-29.md`, on branch
`worktree-agent-a9f533e89bd5d8242`, unmerged). Raw artifacts and instruments:
`perf-artifacts-17/`. Baseline: `main` at `0f820a0` (studies 12–15 merged and
gated; **study 16's −29.2 is NOT in this baseline** — it is a separate unmerged
branch that touches only `src/eval/nnue/*` and `src/history.rs`, none of which
this study changes, so the two merge independently). Comparison target:
**`engines/bin/stockfish-eca43a97`**.

---

## 0. HEADLINE

### 0.1 The frame, stated first

No PGO panel and no gate: the owner's standing constraint for this session is
**no `lto="fat"` builds while the GPU trainer is live**, which rules out
`tools/check.sh`, `CHECK_NNUE_GATE=1` and `build_pgo.sh` alike. Everything below
is either the **thin-LTO probe frame** (deltas) or the **callgrind frame**
(attribution). Study 12 §0.3 still stands: no probe delta may be quoted as a
result.

| | probe frame (thin LTO, 4 CGU, non-PGO, `-C target-cpu=native`) |
|---|---:|
| baseline (`main` @ `0f820a0`) | 6335.8 i/n |
| **shipped tree** | **6238.1 i/n** |
| **delta** | **−97.7** |
| branches | 673.6 → **667.3** per node |
| node count, every line | 5,001,562 |
| eval hash, every line | `5ca978e012da1bb0` |

**Falsifiable PGO prediction.** Studies 12–15 measured the probe frame
overstating the PGO delta by **1.12–2.0×**. Applying that band to −97.7 predicts
**−49 to −87 i/n in the PGO frame**, i.e. 5950.1 → 5863–5901, and the ratio
1.3364× → **1.3170–1.3256×**. Study 16's own −29.2 (probe) predicts a further
−15 to −26. If the owner's PGO run lands outside −49…−87 for this study's diff,
the band itself is wrong and should be re-derived rather than the result
replaced.

**AND THE HONEST QUALIFIER, which the box forced (§10).** The full D24 perft
ritual and both gates are **deferred to the owner**: this box OOM-killed four of
this session's binaries at ~3.1 GB while a 3 GB trainer holds it, and the owner
revised the commission mid-session to defer the release perft tier rather than
fight for it. What that means arm by arm is in §10.1 — **u1–u6 have the full D24
release perft tier green; u7 has only the fast tier**, and for a change to
`unmake_move` node identity on the probe tree is evidence, not proof.

There is a specific reason to expect this study's arms to survive PGO **better**
than the band suggests: every one of them is a cold-path outlining, and PGO's
own profile data makes exactly that call more aggressively, not less. The
prediction is stated at the band anyway.

### 0.2 What landed

| # | change | probe Δ | where |
|---|---|---:|---|
| u1 | `unmake_move`'s castling arm → `#[cold] unmake_castling` | −9.5 | `board.rs` |
| u2 | `make_move_with` step 4's castling arm → `#[cold] make_castling` | −0.9 | `board.rs` |
| u4 | `DirtyPiece::from_move`'s castling arm → `#[cold] castling_record` | −3.0 | `board.rs` |
| u5 | `gives_check`'s promotion/ep/castling arms → `#[cold] gives_check_special` | **−34.6** | `board.rs` |
| u6 | `legal`'s castling+ep arms → `#[cold] legal_special` | **−48.3** | `movegen.rs` |
| u7 | `unmake_move`'s promotion/ep arms → `#[cold] unmake_special` | −1.4 | `board.rs` |
| | **total, probe frame** | **−97.7** | |

### 0.3 What this study is actually about

**(a) Target 2 is closed in the "it is not there" direction, with an exact
number.** Study 11 measured make/unmake at **1.93× Stockfish** and study 13's
block table put it at **+359 i/n**. Both were built on sampled profiles of
stage-split builds. Measured exactly, on both engines, in one frame, at the same
node budget on the same 50 positions:

| | ARCHON | SF | |
|---|---:|---:|---|
| **whole make side, INCLUSIVE** (`Worker::do_move`, 1.0000 calls/node both) | **899.6** | **927.3** | **0.970× — we are faster** |
| unmake (`unmake_move` / `undo_move`) | 79.3 | 53.8 | 1.474×, +25.5 |
| check info (`refresh_check_info` / `set_check_info` inclusive) | 106.2 | 137.8 | **0.771× — we are faster** |

`+25.5` is the entire surviving deficit on the move-making side, and it is
`unmake_move`'s 168-byte `StateInfo` restore against Stockfish's pointer hop.

**(b) The block the gap IS in was on the do-not-reopen list.** Threat
generation, which study 13 §7 ranked "at parity (+33), not a defect":

| | ARCHON | SF |
|---|---:|---:|
| threat generation / maintenance, per node | **1148.9** (1232.6 with the libc `memcpy` inside `at_full`) | **674.8** |

**1.70–1.83×, +474 to +558 i/n.** §4 shows exactly how a +474 gap read as +33:
study 13 hand-picked three of our nine threat symbols and compared them against
a sampled SF number that is 1.31× the exact one. Two errors, same direction,
multiplied.

**(c) An instrument, and the trap inside it.** `callgrind` counts every executed
instruction exactly, per function, on any binary — so ARCHON and Stockfish go
into one frame with neither rebuilt for the purpose, and there is no sampling
skid to be wrong by 1.44× (study 16 §11). It needed two AVX2 builds because
valgrind cannot execute AVX-512. **And that is where the trap is**: our SIMD
dispatch is *runtime*, so the AVX2 build still runs the AVX-512 kernels natively
and only falls back under valgrind. §6.2 is the near-miss headline this
produced and how it was caught.

**(d) The cheapest lever in the engine turned out to be `#[cold]`.** −96.3 i/n
for five call-site splits, no algorithm changed, no semantics touched, three
node-identity checks green. §5 also draws the boundary: the sixth split of the
same shape cost **+64.0**, and the difference between them is a rule.

---

## 1. METHOD

Studies 8–16's rules were followed.

### 1.1 NULL FIRST — and this study's null panel is three-legged

**Null A — rebuild determinism.** Same source, forced recompile:

```
null-1           31,688,912,647   6335.8 i/n
null-2-rebuild   31,688,924,538   6335.8 i/n     11,891 apart, ±0.0024 i/n
```

`null-1` reproduces study 15's and study 16's published `6335.8` baseline — the
third study running in which this instrument reproduces its predecessor's
number across a session boundary.

**Null B — SOURCE PERTURBATION, structurally inert.** Study 14's null failed
informatively (+86.4) when a "neutral" re-spelling was not; study 16's fix was
to prefer a reborrow. Null B here is an SSA-identical alias binding at exactly
the two sites arm 1 was about to change — `let bits = self.0;` in front of
`Move::promotion_piece`'s field read, and `let mv = m;` in front of
`Board::capture_stage`'s match:

```
n0-NULLB-alias   31,688,991,453   6335.8 i/n     ±0.0158 i/n vs null-1
```

**Null C — the revert control.** After the three rejected arms, reverting to the
baseline source reproduced it to **±0.0195 i/n** (`a3r-revert-control`).

Every landed arm is 500× the largest null. Every rejected arm is at least 30×
it, so the losses are real effects too, not noise.

### 1.2 Instruments

* **`probe17.sh`** — screening, inherited from `probe16.sh`. Thin LTO / 4
  codegen units, `-C target-cpu=native`, non-PGO, fixed 5,001,562-node tree;
  prints `instructions:u`, `branches:u`, the wl50 eval hash and the node count
  on every line. Every build runs inside its own
  `systemd-run --user --scope -p MemoryMax=3G -p MemorySwapMax=0` cgroup **and**
  through `tools/oomlast.sh`. Build-and-measure cycle: **6 s**.
* **`cg17.sh` + `cgsum17.py` — NEW, and the study's main contribution.** Exact
  per-function self-instructions, exact per-function CALL COUNTS, and exact
  INCLUSIVE cost per call edge, from `callgrind`. §2.
* **`cgbuild17.sh`, `sfbuild17.sh` — NEW.** The two AVX2 binaries the callgrind
  frame needs. §2.1.
* **`blocks17.py` — NEW.** Study 13's block table, re-derived over exact
  self-costs from one build of each engine instead of two stage-split ones.
* **`cgbr17.sh` — NEW.** `--branch-sim=yes`: exact per-function conditional and
  indirect branch counts. Usable for integer code only — §6.2.
* **`probeany17.sh`, `dis17.sh` — NEW.** Run the probe measurement on an
  arbitrary binary (no build); dump one symbol's disassembly.
* **`flat8.sh`, `pstat8.sh`, `pgo16.sh`** — inherited.

### 1.3 The box, honestly

A GPU training run held the box for the whole session; available memory sat
between 1.1 and 3.3 GB and **swap was 4095/4095 MB with zero free the entire
time**. Everything here rides on `instructions:u` over fixed-node trees and on
deterministic callgrind counts, both contention-immune. **No wall-clock or NPS
number is quoted as a claim.** One build at a time; `free -m`'s SWAP line
checked before each; nothing was OOM-killed and the trainer never noticed.

---

## 2. THE INSTRUMENT: EXACT PER-FUNCTION COUNTERS, ON BOTH ENGINES

Every study since #8 attributed instructions with
`perf record -e instructions:u -c <period>`. That is a sampling instrument;
study 16 §11 measured its per-opcode form **wrong by 1.44×** on this box, and
§4 below shows its per-symbol form wrong by 1.31× on Stockfish.

`callgrind` counts every executed instruction, per function, with no sampling
and no skid, on **any** binary — so neither engine has to be instrumented or
rebuilt for the purpose. It also gives two things `perf` cannot give at all:

* **exact call counts per function**, which turns an i/n into an i/call and
  makes a function comparable across trees of different depth; and
* **exact inclusive cost per call edge**, which is the only
  **inlining-immune** way to compare two engines whose compilers made different
  inlining decisions. Self-costs move between symbols when inlining changes;
  the inclusive cost of an entry point does not. §3 is built on it.

### 2.1 Two AVX2 builds, and why

valgrind cannot execute AVX-512, and **both** shipped binaries are AVX-512
(`engines/bin/stockfish-eca43a97` has 1,309 `zmm` references; ours dies with an
illegal opcode inside `std::rt::lang_start_internal` before it reaches `main`).

* **ARCHON**: `-C target-cpu=x86-64-v3` into its own target dir
  (`cgbuild17.sh`). +4.8% instructions and +8.5% branches against the native
  build, measured (`probeany17.sh`).
* **Stockfish `eca43a97`**: rebuilt at `ARCH=x86-64-avx2` (`sfbuild17.sh`) from
  a **read-only `git archive`** of the pinned revision — the clone's checkout
  state was never touched, and it sits at the older `sf_18` tag, which CLAUDE.md
  warns is the *other* direction of the same mistake. The net was exported out
  of the shipped AVX-512 binary with SF's own `export_net`; its sha256 begins
  `0ee0657fb25e`, which is exactly `EvalFileDefaultName`, so the rebuild is
  bit-identical in weights to the binary every previous study measured.

Both engines then ran the **same 50 wl50 positions at the same node budget**:
150,059 nodes against 150,053.

### 2.2 What the frame is good for, and what it is not

| | usable | not usable |
|---|---|---|
| a function at ~1.0 calls/node | **yes** — i/n is i/call, and i/call is depth-independent | |
| integer / bitboard code | **yes** — up to the measured 4.8% v3-vs-native inflation | |
| inclusive cost of a call tree | **yes** — inlining-immune | |
| the SIMD accumulator kernels | | **no** — §6.2's dispatch trap |
| absolute totals, and any ratio built from them | | **no** — same reason, plus a 3000-node-per-position tree evaluates far more often per node than the probe's 100k |

The frame's search-only slope is 15,054 i/n for ARCHON and 7,171 for SF. **That
2.10× is not a result and is not quoted as one.** The probe frame on the same
binaries and the same tree says 13,472 and 14,721 — i.e. the callgrind ARCHON
number is 1.63× what perf retires while the SF number matches to 1.6%, which is
§6.2 in one line.

One cross-check the instrument was not asked for: the fixed per-process cost
falls out of a two-point slope as **1,028,147,445 instructions for ARCHON**,
against study 16 §8's independently measured **1,020,273,491**. Two
instruments, two architectures, **0.8% apart**.

---

## 3. TARGET 2 — MAKE/UNMAKE IS AT PARITY

The commission said "make/unmake measured 1.93× SF in study 11 — on a
partially-retracted instrument. Re-verify with deterministic counters on BOTH
engines first, then attack."

Re-verified. There is nothing to attack.

### 3.1 Inclusive, which is the only fair comparison

Both engines call their `Worker::do_move` exactly 1.0000 times per node. The
inclusive cost of that call is the whole make side — the board mutation, the
state push, the check-info rebuild, the dirty-piece record, and each engine's
threat-record maintenance:

```
ARCHON  Worker::do_move    1.0000 calls/node   INCL  899.6 i/n
SF      Worker::do_move    1.0000 calls/node   INCL  927.3 i/n     0.970x
```

Below it, per node:

| role | ARCHON | SF | Δ |
|---|---:|---:|---:|
| make (`make_move_with` / `Position::do_move`), inclusive | 327.7 | 829.5 | −501.8 |
| …of which SF's eager threat maintenance | — | 440.0 | |
| check info, inclusive | 106.2 | 137.8 | **−31.6** |
| dirty-piece record (`DirtyPiece::from_move`) | 52.1 | (inside `do_move`) | +52.1 |
| worker glue (self) | 105.1 | 98.5 | +6.6 |
| **unmake** | **79.3** | **53.8** | **+25.5** |

Two of the three "make/unmake is our problem" numbers in the corpus invert.
`refresh_check_info` — study 13 §8 item 3 called it out as "59 i/n against SF's
33 for identical code" — is **31.6 i/n FASTER than Stockfish's `set_check_info`
+ `update_slider_blockers`** when both are measured inclusively in one frame.

### 3.2 The one real gap, and what it is

`unmake_move` is 79.2 i/call against `undo_move`'s 53.6. The disassembly says
where it goes:

* `mov 0x10(%rdi),%rcx; test; je` — `history.pop()`'s `expect` panic edge.
* `imul $0xa8,%rcx,%rcx` — the 168-byte `StateInfo` stride.
* `vmovups`×3 + `mov`, then four stores — **the 168-byte state restore**, 8
  instructions with AVX-512.
* `jmp *%rdx` — a jump table on `MoveKind`.

Stockfish's `undo_move` restores its state with `st = st->previous`. Study 14
deliberately traded that pointer for an inline `StateInfo` plus a `Vec` history,
and it won on the make side (that study's headline). **This is the other half of
that trade, and it is +25.5 i/n.** It is ranked, not attacked — see §11 item 3;
the alternatives all reintroduce the borrow study 14 removed.

### 3.3 TARGET 1 — `Worker::do_move`, split and priced

The commission called `do_move` "the largest single unmeasured block in the
engine … nobody has ever looked inside it", and prescribed splitting it with
`noinline` the way study 12 split Stockfish's eval. **Callgrind does the same
split without touching the source**, because every stage is already a call and
callgrind prices call edges exactly:

| stage of `Worker::do_move` | calls/node | i/n | share |
|---|---:|---:|---:|
| **`SotaContext::push`** — the accumulator + threat record | 1.0000 | **415.0** | **46%** |
| … `SotaContext::push` self (inlines `HalfkaContext::push`) | | 41.0 | |
| … `ThreatContext::apply_at` self | 1.0000 | 151.3 | |
| … **the eager king-cross threat refresh** (`refresh_threat` 181.6 + `refresh_psqt_threat` 31.5 + a libc `memcpy` 9.6) | **0.0357** | **222.7** | **25%** |
| **`make_move_with`** | 1.0000 | **327.5** | **36%** |
| … self | | 215.0 | |
| … `refresh_check_info` | 1.0006 | 105.8 | |
| … `attackers_to` (the `gives_check` confirmation) | 0.0996 | 6.9 | |
| **`Worker::do_move` self** — node counter, atomic mirror, `capture_stage`, `ContHistId::new`, the inlined `push_dirty`/`push_sota_dirty` dispatch | 1.0000 | **105.1** | 12% |
| **`DirtyPiece::from_move`** | 1.0000 | **52.1** | 6% |
| **total** | | **899.6** | |

**The expensive stage is not the board work. It is the eager king-cross threat
refresh: 222.7 i/n, a quarter of the whole make side, at 0.0357 calls per node
— 6,240 instructions per call.** That is §11 item 1, and it is the largest
single mechanism this study found.

The second-largest surprise is how cheap the rest is: 105.1 i/n of glue in a
function study 15 sized at ~300, and 52.1 for the dirty-piece record whose
"duplicate derivation" study 13 measured and rejected at +41.5.

### 3.4 Why study 11 and study 13 disagreed with this

Study 13 §7 built its block table from two **stage-split** `perf record`
profiles — ARCHON at 6362 i/n and SF at 5090, each calibrated off its own
`search<NonPv>` symbol — and said in the same paragraph that "split frames are
not comparable in absolute terms". Applying study 13's *own* bucket list to
exact counts gives ARCHON 451 vs SF 518, i.e. **we were already ahead**, against
its 661 vs 302. The `#[inline(never)]` markers a stage-split build inserts are
precisely what a self-cost profile then attributes differently.

---

## 4. WHERE THE GAP ACTUALLY IS — THREAT GENERATION, 1.70–1.83×

### 4.1 The measurement

Same frame, same 50 positions, same budget, self-costs (which sum exactly to the
engine's total, so a bucket can be checked):

| ARCHON | i/n | | SF | i/n |
|---|---:|---|---|---:|
| `ThreatContext::at_full` (inlines `compute`) | 466.19 | | `update_piece_threats<true>` (incl) | 371.0 |
| `record_square_touch` | 152.60 | | `append_changed_indices` | 216.19 |
| `ThreatContext::apply_at` | 151.31 | | `update_piece_threats<false>` (incl) | 69.0 |
| `record_targets` | 133.60 | | threat-aware `remove_piece` | 13.30 |
| `refresh_threat` | 81.28 | | threat-aware `put_piece` | 5.33 |
| `Board::ensure_threats` | 74.10 | | | |
| `ripple` | 38.34 | | | |
| `refresh_psqt_threat` | 30.30 | | | |
| `record_outgoing` | 12.79 | | | |
| out-of-line attack lookups from the above | 8.4 | | | |
| **total** | **1148.9** | | **total** | **674.8** |
| (+ the libc `memcpy` `at_full` calls, 0.3111/node) | 1232.6 | | | |

**+474 to +558 i/n. 1.70× to 1.83×.** Neither side includes the SIMD
accumulator fold: ARCHON's `at_full` has an inclusive cost of 4618.3 i/n but
3725.6 of that is `i8_deltas_into`, and SF's fold is `apply_combined`. Both
sides' attack lookups are included (SF's `both_attacks_bb` is called 3.61 times
per node **from `update_piece_threats<true>`** and is inside its 371.0).

### 4.2 How "+33, at parity" happened

Study 13 §7 wrote:

> Our lazy regenerate-at-fold design (`at_full` 363 + `record_square_touch` 140
> + `record_targets` ~95 = ~600) costs *less* than Stockfish's eager incremental
> maintenance (`update_piece_threats<true>` 389 + `<false>` 120 +
> `append_changed_indices` 177 = 686).

Two independent errors, both pointing the same way:

1. **Our side is three symbols out of nine.** `apply_at`, `ripple`,
   `refresh_threat`, `refresh_psqt_threat`, `record_outgoing` and
   `ensure_threats` are **397 i/n** that were never counted. The three that were
   counted are 752 of our 1149.
2. **Their side came from a sampler.** The same five SF symbols measure **524
   self / 674.8 inclusive** exactly, against the profile's 686. A 1.31×
   overstatement.

0.65 × 1.31 ≈ 0.85, and a +474 gap reads as +33.

**The rule this pays for: A BLOCK IS NOT A SYMBOL LIST.** A bucketing that does
not sum to the engine's total cannot be audited, and when the two engines' lists
are assembled separately by hand, both errors land in whichever direction the
author already expects. `blocks17.py` buckets by regex over *every* symbol and
prints an `unbucketed` row for exactly this reason.

### 4.3 What is NOT claimed here

* Not that our design is wrong. Lazy regenerate-at-fold pays on 41.5% of nodes
  where SF pays on 100%; that trade may still be right at a different width.
* Not a PGO number. This is the callgrind frame (§2.2), so it carries the 4.8%
  v3-vs-native inflation and nothing else from §2.2's "not usable" column —
  every symbol in the table is integer code.
* Not that it is 474 i/n of available headroom. It is 474 i/n of **difference**,
  and the first job of study 18 is to find out how much of it is the design
  choice and how much is the spelling.

---

## 5. WHAT LANDED — OUTLINE THE RARE `MoveKind` ARMS

Five hot functions run once or twice per node and answer a **Normal** move over
95% of the time, and each carried its castling / en-passant / promotion arm
inline. Those arms are what set the function's register pressure: they hold the
only code in the function that touches the rook-square table, the slider tables,
a rebuilt occupancy or an `attackers_to_exist` loop, so the prologue spills
callee-saved registers for a path that almost never runs. `unmake_move`'s
prologue is **seven pushes and seven pops** on every unmake.

```
u1-unmake-castling-cold        6326.3    -9.5   LANDED
u2-make-castling-cold          6325.4    -0.9   LANDED
u3-make-promo-keys-cold        6389.4   +64.0   REJECTED  (S5.2)
u4-dp-castling-cold            6322.4    -3.0   LANDED
u5-giveschk-special-cold       6287.8   -34.6   LANDED
u6-legal-special-cold          6239.5   -48.3   LANDED   (-7.1 branches/node)
u7-unmake-special-cold         6238.1    -1.4   LANDED
```

`gives_check` and `legal` pay the most because they are called 1.64 and 1.80
times per node — more often than `do_move` itself — and because their special
arms are the biggest of the five.

### 5.1 `unmake_move`'s promotion/ep arm (u7) — −1.4

Same shape as u5/u6, applied to the last inline special arm in the file: the
`Normal` case becomes straight-line (`move_piece_known` plus the ordinary
capture restore) and promotion/en-passant go out of line together. It is small
because u1 had already taken the expensive half of the same function.

### 5.2 The boundary — u3 cost +64.0, and the rule is narrower than "outline cold code"

`make_move_with` step 8's promotion-key block is by any static measure the best
outlining candidate in the file: two popcounts, four `ZOBRIST` table reads and a
`PIECE_VALUES` lookup, on a path taken by well under 1% of made moves. Outlined,
it measured **+64.0 i/n**.

The difference from u1/u2/u4/u5/u6 is structural: those five are whole arms of
the function's **own top-level `MoveKind` dispatch**, where the caller already
has every argument live in a register. u3's block is nested inside the
`pc.piece_type == Pawn` branch, so outlining it forced the caller to
*materialise* `pc`, `us`, `m` and `to` for a call it otherwise never makes —
and to keep them live across it.

**Outline a whole arm of the function's own top-level kind dispatch. Do not
outline a nested block.** Study 13 §5.3 hit the same wall from the other side
(+106.3 for "the same trick one level down"); this is the same rule with the
boundary drawn.

---

## 6. WHAT DID NOT LAND

### 6.1 `capture_stage`'s jump table — now dead at FIVE spellings

Study 16 §7.2 measured two spellings that remove the jump table (+6.4 and +0.7)
and commissioned a third, on reviewer B's reading of `eca43a97:src/position.h`:
SF has **three** cases where we have four, its `type_of()` is a mask, and its
`promotion_type()` is raw bit arithmetic where our `promotion_piece()` performs
a second `kind()` dispatch and returns an `Option`.

Both halves were built and measured:

* `Move::promotion_piece_raw()` — SF's `promotion_type()` exactly
  (`PieceType(((data >> 12) & 3) + KNIGHT)`, spelled as a `match` so it stays
  safe Rust), at all five hot `.expect()` sites: **+0.5**. It DID do what it was
  for — the promotion arm's `PieceType::ALL` table load became
  `not`/`test $0x3000`/`sete` in the disassembly — and it still cost.
* SF's three-case `if` chain: **+0.7**.
* A fourth spelling, testing `EnPassant | Castling` as a pair so the compiler
  sees two cases: **+21.4, and +4.7 branches per node.**

**LLVM rebuilds the switch from every source shape.** `jmp *%rdx` is still in
`Worker::do_move` at `+0x6f` in all four. The site is closed for instructions;
if anyone reopens it, it is for branches and with a cycles instrument (study 16
item 3).

The `promotion_piece_raw` API is reverted rather than kept: it costs 0.5 i/n and
it is a footgun (it returns `Knight` for a non-promotion). Recorded here so
study 18 does not re-derive it.

### 6.2 The dispatch trap — a near-miss headline, and the reason §2.2 exists

With `--branch-sim=yes`, the two accumulator delta kernels showed **1,480
conditional branches per node** — 756.8 from `i8_deltas_into` at 888 per call,
722.8 from `i16_deltas_into` at 750 per call — against Stockfish's
`apply_combined` at 112.3. Fourteen times. That is the shape of "our branch
ratio is generated by the accumulator update loops", which is exactly the
discovery the commission asked for.

It is an artifact, and three things caught it:

1. **A perf flat profile of the same binary** showed
   `sota_simd::avx512::i8_deltas_into_inner`, `avx512::i16_deltas_into_inner`,
   `avx512::activate_inner` — where callgrind showed `avx2::…`. Our SIMD
   dispatch is **runtime** (`is_x86_feature_detected!`), so a
   `-C target-cpu=x86-64-v3` binary still selects the AVX-512 kernels on this
   CPU; only under valgrind, whose CPUID hides AVX-512, does it fall back.
2. **Callgrind counted 1.63× the instructions perf retired for ARCHON** (3.287 G
   vs 2.022 G on the same tree) **while matching Stockfish to 1.6%.** An
   instrument that disagrees with another instrument on one engine and not the
   other is describing the engine, not the workload.
3. `--separate-threads=yes` ruled out the obvious alternative (a wall-clock-
   proportional spin loop): there is exactly one thread.

**Rule 6 earned its place.** The story was beautiful, it explained a standing
open question, and it was wrong. The general form: **a runtime-dispatched
kernel is invisible to any instrument that changes what the dispatch sees.**

In the shipped frame the branch decomposition (sampled, so soft) is: ARCHON
673.6 br/node vs SF 384.3, with movepick +43 (`init_quiets` 71.2 vs
`partial_insertion_sort` 16.6 + a lower `next_move`), threat generation +40, and
the SIMD kernels at parity (98.6 vs 93.9). §11 item 5.

### 6.3 One `&mut` instead of two `IndexMut` calls — +1.0

`HalfkaContext::push` writes `self.stack[ply + 1].dirty` and then
`self.stack[ply + 1].computed` — two indexed accesses, so two bounds checks and
two reloads of the boxed slice's base and length, because LLVM cannot prove the
first store misses `self`'s own header. `ThreatContext::apply_at` does the same
six times. Rewriting both to take one `&mut` to the entry measured **+1.0**.

Worth recording *why* it did not matter: the duplicated checks that ARE visible
in `Worker::do_move`'s disassembly belong to `SimpleHalfkaContext` and
`ProtectedHalfkaContext`, both of which are `None` in every shipped
configuration — the `cmpq $0x0` above them is never taken. Cold code, correctly
identified as cold by the compiler, and reading a disassembly does not tell you
which basic blocks execute. **Pair every disassembly reading with a counter.**

---

## 7. `attacks::tables()` — CLOSED, with a number

Study 13 counted 5.62 `LazyLock` derefs per node from an instrumented build and
sized the guard at 17–22 i/n; studies 14, 15 and 16 all re-ranked it without
testing it, and study 16 §7.4 could not find it in the disassembly.

Exactly, per node:

```
LazyLock<AttackTables>::force closure       0.0000 calls/node   (every call is start-up)
AttackTables::attacks (out of line)         0.5469 calls/node   13.81 i/n   25.2 i/call
attacks::init_magics                        349 M instructions  ALL of it fixed cost
```

The slow path never runs in a search. The fast-path guard is inlined, so
callgrind cannot price it directly, but the 17–22 estimate has **no per-node
mechanism behind it** and should not be spent against. What the number does say
is that `init_magics` is 349 M of ARCHON's 1,028 M fixed cost — a third of
start-up, and irrelevant to a game.

---

## 8. THE CORRECTNESS ITEM — `upcoming_repetition` VERIFIED, AND STUDY 16'S FRAMING CORRECTED

Study 16 §7.1 reported that our `upcoming_repetition` is the pre-`6b782211`
shape. **Verified against the reference, and the commit is exactly as
described.**

`6b782211` ("Limit has_game_cycle() to only upcoming repetition", 2024-06-25,
non-regression STC + LTC) does three things in one patch:

* renames `has_game_cycle` → `upcoming_repetition`;
* **adds** the incremental `other` XOR and `if (other != 0) continue;`;
* **deletes** `if (color_of(piece_on(empty(s1) ? s2 : s1)) != side_to_move())
  continue;`.

At `eca43a97` there is no `has_game_cycle` and no colour test. `board.rs:1899`
carries the colour test and no `other`. So we are the pre-`6b782211` shape,
confirmed on both halves.

**One correction to study 16's handoff, and it matters for how the item is
ranked.** Study 16 wrote that "our current `ply > i` branch can report a draw
the side to move cannot actually force", making it a NO-ACCEPTABLE-BLUNDERS
defect class. The `ply > i` branch is **byte-for-byte the same policy in both
engines** at `eca43a97`; `6b782211` did not touch it. The only semantic
difference is in the at/above-root branch, and there **our extra colour guard
makes us STRICTER than Stockfish, never looser** — it can only cause us to MISS
a cycle, never to claim one that is not there.

So the item is: a **missed-detection** difference against a reference that
removed the guard deliberately, plus a speed filter that study 16 measured at
+17.8 i/n **for us** because our 168-byte-stride indexed state walk has to touch
the even plies that SF's linked list already had in hand. Still a games
question, still not a perf arm, but it is not a false-draw defect. §11 item 6.

---

## 9. TESTS, AND EACH LANDED CHANGE HAS ONE THAT CAN FAIL

Every landed change is a call-site split with identical semantics, so the
existing suites are the oracle and the only question that matters is whether
they can FAIL. Six mutations, each applied, built, run under the FAST
`cargo test` and reverted (`mutations.txt`). **All six failed a test, and the
naming matters — not one of them was caught by a test written for the function
I split:**

| helper | mutation | caught by |
|---|---|---|
| `unmake_castling` | swap `rfrom`/`rto` | `bench::tests::spawned_bench_perft_runs_are_byte_identical` (`bench.rs:791`) |
| `make_castling` | return `0` instead of the rook delta | same |
| `castling_record` | drop `dp.sub(rook, rook_from)` | `eval::nnue::accumulator::tests::dirty_piece_covers_every_move_kind` (`accumulator.rs:379`) |
| `gives_check_special` | return `false` for `Promotion` | `board::tests::gives_check_fixture_cases` (`board.rs:1539`) |
| `legal_special` | return `true` unconditionally | `bench::tests::spawned_bench_perft_runs_are_byte_identical` (`bench.rs:816`) |
| `unmake_special` | restore the pawn on `to`, not `from` | `board::tests::gives_check_fixture_cases`, via `move_piece_known`'s occupied-square `debug_assert` (`board.rs:1225`) |

Two things worth keeping. `make_castling`'s mutation is a pure **Zobrist key**
error with no placement consequence, and perft caught it — because the bench
perft runs go through the TT and a wrong key changes the tree. And four of the
six were caught by tests in files this study did not touch, which is what "the
refactor is covered" is supposed to look like: if a split can only be caught by
a test of itself, it is not a refactor.

The panel ran under the fast tier, so it proves the mutations are *detectable*,
not that HEAD is perft-clean. Both statements are in §9.1 and §10.

### 9.1 What actually ran, and on which tree

The box OOM-killed four of this session's binaries at ~3.1 GB anon-rss while a
3 GB trainer held it, so this table is not a formality — it is the difference
between what is proven and what is asserted.

| check | u1–u6 tree | u7 tree (HEAD) |
|---|---|---|
| `cargo test` (default features — includes the FAST perft tier + checked mini-tier) | **PASS** | **PASS** — 607 passed, 0 failed |
| `cargo test --features sota,arch-d48x` | **PASS** — 690 passed, 0 failed, 8 ignored | not re-run |
| `cargo test --release -- --ignored perft_full_suite` (**D24 ritual**) | **PASS** | **not run — deferred to the owner** |
| `cargo clippy --all-targets` | **PASS** | not re-run |
| `cargo clippy --all-targets --features sota,arch-d48x` | **PASS** | not re-run |
| probe-tree node identity (5,001,562) + eval hash | PASS | PASS |

**Read that honestly: for u7, node identity on the probe tree is EVIDENCE, NOT
PROOF.** u7 rewrites `unmake_move`'s arm dispatch. A 5 M-node search over 50
positions exercises promotions and en-passant captures, so a wrong arm would
almost certainly move the node count — but "almost certainly" is what the D24
ritual exists to replace. It is item 1 of §10.

The single-threaded `sota` run is also worth recording as a box fact: the same
suite was SIGTERM'd twice at ~218/698 under
`systemd-run -p MemoryMax=4G -p MemorySwapMax=0` before it completed at
`--test-threads=1` in 135 s. The `sota_loader` tests allocate 138 MB nets; four
of them in parallel is the cap.

---

## 10. WHAT IS OWED BEFORE THIS MERGES — THE PENDING CHECKLIST

Everything here is deferred to the owner on a quiet box, per the mid-session
revision of the commission. **Nothing in this list can change the code**; they
are all measurements of a tree that is already committed. In this order:

1. **`cargo test --release -- --ignored perft_full_suite` — the D24 ritual, on
   HEAD.** This is the one that matters most and the only one whose failure
   would mean a real bug: this study touches `movegen` and `board` mutation, and
   u7 has not had it. Takes ~7 s once built.
2. **`cargo test --features sota,arch-d48x` on HEAD** (`--test-threads=1` if the
   box is busy) — expect 690 passed, 0 failed, 8 ignored.
3. **`CHECK_NNUE_GATE=1 CHECK_TEST_THREADS=4 bash tools/check.sh`** — expect
   `Bench: 587683`, eval hash `5ca978e0…0b38`, d13 nodes `1922947`, **all three
   unchanged**. Every arm here is supposed to be node-identical, so a change in
   any pin is a bug in this study, not a new pin.
4. **`bash tools/build_pgo.sh sota,arch-d48x engines/bin/archon-s17 engines/canonical/net.cenn`**
   then the two-arm panel against `engines/bin/stockfish-eca43a97`
   (`pstat8.sh 5 300000 tools/nnue/calib/wl50.fens counters-headline.txt …`) —
   the deliverable frame, and the test of §0.1's **−49…−87** prediction.
5. **`cargo clippy --all-targets --features sota,arch-d48x`** on HEAD. Note
   `check.sh` clippies default features ONLY (study 15 item 12, still open), so
   this one has to be typed by hand.

Optional, and the cheapest thing that would extend this study: rerun
`docs/research/perf-artifacts-17/scripts/cg17.sh` on an idle box with
`#[inline(never)]` on `ThreatContext::compute`, `build_deltas` and
`resolve_both` — that splits §4's 1148.9 i/n into four priced stages and is
§11 item 1's whole cheapest-next-test.

### 10.1 Node identity, arm by arm

| arm | probe tree identical | fast perft tier | **D24 full perft** |
|---|---|---|---|
| u1 `unmake_castling` | yes | yes | **yes** |
| u2 `make_castling` | yes | yes | **yes** |
| u4 `castling_record` | yes | yes | **yes** |
| u5 `gives_check_special` | yes | yes | **yes** |
| u6 `legal_special` | yes | yes | **yes** |
| u7 `unmake_special` | yes | yes | **NO — owner must run it** |

---

## 11. RANKED REMAINDER — handoff to study 18

**Closed by this study, with numbers:** make/unmake (0.970× inclusive — do not
reopen), `refresh_check_info` (0.771×), `attacks::tables()` (0.0000
force-calls/node), `capture_stage`'s jump table for instructions (five
spellings). **Re-opened by this study:** threat generation.

| # | item | size | confidence | cheapest next test |
|---|---|---:|---|---|
| 1 | **THE EAGER KING-CROSS THREAT REFRESH IS PAID ON NODES THAT NEVER EVALUATE.** §3.3. `ThreatContext::apply_at` rebuilds a perspective's whole threat accumulator from scratch when that perspective's king crosses the d/e mirror — **222.7 i/n, 25% of the entire make side, at 0.0357 calls/node (6,240 instructions per call)**. It runs at PUSH time, i.e. on every made move; only **41.5%** of nodes ever evaluate, so **~58% of it is thrown away**. Everything else in that module is already lazy; this is the one eager path | **~130** | **high — exact, and the mechanism is a one-line design choice** | it is eager only because `refresh_threat` takes a `&Board` and the live board is gone by fold time — but `apply_at` **already stores a `ThreatPosition` snapshot in the same entry**, and that struct carries exactly what `for_each_threat` reads (`by_type`, `by_color`, `occ`, `kings` vs `board.pieces(c,pt)`, `all_occupancy()`, `king_square()`). So: make `for_each_threat` take the snapshot instead of the board, mark the crossed ply "needs refresh" rather than `computed`, and teach `compute`'s ancestor walk to do the refresh when it reaches such a ply. **Needs the NNUE gate, so it is not a gate-frozen session's work**, and it collides with study 16's unmerged `refresh_threat_into` — do them in one study |
| 1b | **Threat generation overall, 1.70–1.83× SF.** §4. Our nine symbols cost 1148.9 i/n against SF's 674.8. The designs differ (lazy regenerate-at-fold on 41.5% of nodes vs eager maintenance on 100%), so part of the gap is the design and part is the spelling, and **nobody knows the split** | **+474 to +558** | **high — exact, matched, both engines** | split it before attacking it: `at_full` is 466.19 i/n at 1123 i/call and inlines `compute`; put `#[inline(never)]` on `compute`, `build_deltas`, `resolve_both` and the fold, re-run `cg17.sh`, and price the four stages. One 6 s build and one 10 s callgrind run. **Only then** decide between a cheaper regeneration and SF's eager shape |
| 2 | **The libc `memcpy` inside `at_full`** — 0.3111 calls/node, 83.7 i/n inclusive, ~269 instructions per call, counted not estimated. Not the empty-delta copy (study 14 already chunked that one) | 20–80 | **high — counted** | `memspy16.sh` from study 16, whose interposer already tags the caller and the payload size. The class it belongs to has paid twice (study 15 −22.7, study 16 −14.6) |
| 3 | **`unmake_move`'s 168-byte state restore** — the other half of study 14's trade. 79.2 i/call vs SF's 53.6 | **+25.5** | **high — exact** | not a respelling: SF restores by pointer. The honest options are (a) leave it, since study 14's make-side win was larger, or (b) make `StateInfo`'s *restored* prefix smaller by moving the fields `unmake` does not need (blockers, pinners, check_squares — 48 B) out of the copied block. (b) is a design change to doc 04 and wants its own study |
| 4 | **The `#[cold]` sweep is not finished.** §5's rule (a whole arm of a top-level kind dispatch) has more sites: `movepick`'s scoring loops, `see_ge`'s non-Normal short circuit, `pseudo_legal`'s non-Normal branch (which allocates a 520-byte `MoveList` frame) | 10–40 | medium | one `probe17.sh` cycle each, 6 s. Watch for u3's failure mode: if the caller has to materialise arguments it otherwise never computes, it loses |
| 5 | **The branch ratio is still open, and the SIMD kernels are NOT the answer.** §6.2. In the shipped frame ARCHON is 673.6 br/node vs SF 384.3; the sampled decomposition puts movepick +43 and threat generation +40, with the kernels at parity | ? | low — sampled | there is no exact per-function branch counter for an AVX-512 binary on this box. The one honest route is a *compile-time* AVX2 build of both engines (force the dispatch, do not let it read CPUID) so callgrind's `--branch-sim` sees the shipped loop shape |
| 6 | **`6b782211` whole**, as a games question | a missed-detection class, not a false-draw one (§8) | high | unchanged from study 16 item 4 except the direction. Note the speed half is measured negative for us (+17.8) |
| 7 | **Pack `Piece` into one byte** | 5–20 | medium | unchanged from study 16 item 1. Confirmed live: `Worker::do_move` at `+0x40`/`+0x48` reads both mailbox bytes with two `movzbl` |
| 8 | **`tools/check.sh` clippies DEFAULT features only** | 0 i/n | high | unchanged from study 15 item 12 and study 16 item 10. Still open, still the dangerous direction |

**Four method rules this study hands over.**

1. **A BLOCK IS NOT A SYMBOL LIST.** §4.2. If the bucketing does not sum to the
   engine's total, it cannot be audited, and two hand-built lists will err in
   whichever direction the author expects. Bucket by regex over every symbol and
   print the unbucketed row.
2. **Compare INCLUSIVE cost at a ~1.0-call/node entry point.** §3.1. It is the
   only statistic immune to the two compilers having made different inlining
   decisions, and it is what turned "make/unmake is 1.93×" into "0.970×".
3. **A runtime-dispatched kernel is invisible to any instrument that changes
   what the dispatch sees.** §6.2. Check, once per instrument, that the code you
   are measuring is the code that ships — one perf flat profile of the same
   binary is enough.
4. **Pair every disassembly reading with a counter.** §6.3. Two duplicated
   bounds checks sat in `do_move`'s hot-looking disassembly and executed zero
   times per node.

---

## 12. THREATS TO VALIDITY

1. **Nothing here is PGO-measured** (§0.1). The −96.3 is a probe-frame number
   and the PGO figure is a prediction with a 1.8× spread. Machine constraint,
   not choice.
2. **The callgrind frame is AVX2 and shallow** (§2.2). Every number quoted from
   it is either an integer-code self-cost (up to 4.8% inflated) or a
   ~1.0-call/node inclusive cost. No total and no ratio from that frame is
   quoted as a result — including the 2.10× the frame's own slope shows.
3. **§4's threat comparison depends on the bucketing being complete on both
   sides.** It was built by grepping every symbol containing a threat token in
   each engine, not by hand-picking; that is the fix for §4.2's failure mode,
   but a threat symbol named something else on either side would still be
   missed.
4. **The branch decomposition in §6.2's last paragraph is sampled**, therefore
   soft, therefore ranked at low confidence.
5. **Study 16's −29.2 is not in this baseline.** The two branches touch disjoint
   files, so they should compose, but their sum has not been measured.
6. **Box state**: a training run shared the box for the whole session, swap was
   exhausted throughout, and one debug test run was killed by memory pressure
   before it finished (§9). No wall-clock or NPS number is quoted as a claim.

---

## 13. ARTIFACTS (`perf-artifacts-17/`)

* `probes-fast.txt` — every arm with its eval hash, node count and branch count,
  including all three nulls, the revert control, and the five rejected arms.
* `cg/summary-archon-w50.txt`, `cg/summary-sf-w50.txt` — the exact per-function
  self-Ir, calls/node and i/call tables both engines' numbers come from.
* `cg/blocks-w50.txt` — the re-derived block table (read §2.2 first).
* `cg/callgrind-*.out.gz`, `cg/br-*.out.gz` — the raw callgrind files, both
  engines, two node budgets each, plus the branch-sim pair.
* `cg/annot-sf-*.txt` — `callgrind_annotate` for the SF runs.
* `flat-archon-native-branches.txt`, `flat-sf512-branches.txt` — the shipped-frame
  branch flat profiles (sampled).
* `flat-cg17bin-instructions.txt` — the profile that caught §6.2's dispatch trap.
* `mutations.txt` — the six mutation-verification runs.
* `scripts/cg17.sh`, `cgsum17.py`, `cgbr17.sh`, `cgbuild17.sh`, `sfbuild17.sh`,
  `blocks17.py`, `probe17.sh`, `probeany17.sh`, `dis17.sh` — new.
* `scripts/pstat8.sh`, `parse8.py`, `flat8.sh`, `annot8.py`, `opcount11.py`,
  `blocks13.py`, `sfshow.sh`, `memspy*.sh`, `pgo16.sh`, `startup16.sh` —
  inherited.
