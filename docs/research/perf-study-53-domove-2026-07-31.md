# Perf study 53 — `do_move`'s neighbourhood: three deletions, and two premises in my own commission that did not survive contact

**Date:** 2026-07-31 · **Box:** 7700X (Zen 4), AVX-512 VBMI2, SMT on, 32 KB L1d,
1 MB L2/core, 32 MB L3, 15 GB · **Base:** `main` at `fefa1ed5` (study 52
merged) · **Net/arch:** D48X @ `network_scale=600`, `sota,arch-d48x`, ACC 1280 ·
**Instruments:** `tools/perf53/` (rebased from studies 49/52) · **Artifacts:**
`docs/research/perf-artifacts-53/` · **Branch:** `perf-study-53-domove`

**Frame, quoted with every number below:**

```
bench 16 1 100000 tools/nnue/calib/wl50.fens nodes engines/canonical/net.cenn
```

50 FENs × 100 k nodes = **5,001,562 nodes**, one thread, 16 MB TT, pinned to
physical core 2. Non-PGO release, `-C target-cpu=native`, the same build profile
study 52 adjudicated on.

**Frame cross-check (rule 8).** This study measures **6116.58
instructions/node**. Studies 49, 50, 51 and 52 all measured **6153.06–6153.10**
on the *pre*-`rfp52` tree, and study 52 landed `rfp52` at **−0.593 %**:
6153.06 × (1 − 0.00593) = **6116.6**. The frame reproduces to the last printed
digit through a landed change. `base53`'s `.text` md5 is `e160e553`, which is
exactly the md5 study 52's gate log records for `rfp52b`, the binary it landed.

---

## ★ 0. THE ANSWER, FIRST

### 0.0 Two premises in the commission are wrong, and one of them points the wrong way

Reported first because both were load-bearing for how this study was framed.

**(a) "Study 35 looked at do_move fusion and study 39 reported that fusion is
dead" — false, and it conflates two unrelated levers.** Study 39 killed
**accumulator-channel fusion** (merging the HalfKA and FullThreats NNUE
accumulator channels — a net-architecture question), §8 row 4: *"**DO NOT
BUILD**… −23 (best) … +281 cycles/node ⇒ +1.0 … −11.5 Elo"*. It contains no
mention of `do_move`, `make_move`, `make_move_with`, `unmake_move`,
`push_sota_dirty` or `Meta` anywhere in its text. **`do_move`/`make_move_with`
fusion is the opposite of dead: study 35 §4 REVIVED it**, rebuilding study 13
§5.1's arm (rejected at +41.5 i/n) and measuring **−11.36 i/n — a win**. It went
unlanded only because the alternative (`after`) was worth 2.2× as much on
instructions and, unlike `hookc`, also moved loads. Study 35 §9 item 3 leaves
the PGO comparison explicitly open. **Anyone citing "fusion is dead" against a
`do_move` arm is citing the wrong study.**

**(b) "`do_move` … 461 % stall/own-miss-latency, the most extreme caller ratio in
the engine" is a misattribution signature, not an opportunity.** Study 49
defines `stall/lat` as (share of sampled `ex_no_retire.load_not_complete` ×
1764.05 c/node) ÷ (IBS share of miss latency × 22354.3 c/node), and states the
reading rule itself: *"Every symbol above 100 % is a scalar function that
**calls** something."* `do_move`'s **own** miss latency is **16.7 c/node**; the
76.7 c/node of stall charged to it is run-ahead absorbed from callees while
their misses are outstanding. Study 52 §0.3 then measured the same phenomenon
directly and put it at **46×** on the hottest line. **There is no 461 % of
recoverable stall inside `do_move`, and an arm priced off the 76.7 would be
priced off a counter artifact.** The genuinely unfinished business study 49
names is different and survives: `do_move` has **260.3 instructions/node and
116.3 loads/node that nobody has censused**.

This study therefore attacks op COUNT — the only currency study 52 §4 licenses
at this site — and makes no stall or locality claim.

### 0.1 ★ What LANDED — −39.34 instructions/node, −19.7 branches/node, node-identical

Two of the three commissioned items, promoted to unconditional:

* **`gives_check_by(m, pc)`** — the search's move loop passes the `moved_piece`
  it already resolved instead of letting `gives_check` read it back out of the
  mailbox (§1.1).
* **the magic slider table masked** — the shared attack table is allocated at
  `TABLE_SIZE.next_power_of_two()` and the index masked, so the
  `Box<[Bitboard]>` length load, compare, branch and panic block fold away at
  every `rook_attacks`/`bishop_attacks` — movegen, `see_ge`, `compute_threats`,
  `attackers_to_exist` (§1.3).

Measured on the landing tree, `final53` against `base53`, same frame:

| | base53 | final53 | Δ |
|---|---:|---:|---:|
| **`instructions:u` / node** | **6116.58** | **6077.24** | **−39.34** |

and on the panel (§2.3, n=10, 5 labels, realised mean position 2.00 for every
label, byte-identical null clean on all four exact counters):

| `land53` vs `base53` | median | se | neg/n | p | per node |
|---|---:|---:|---:|---:|---:|
| **`instructions:u`** | **−0.643 %** | **0.000** | **10/10** | 0.0020 | **−39.3** |
| **`branches:u`** | **−2.999 %** | **0.000** | **10/10** | 0.0020 | **−19.7** |
| `ls_dispatch.ld_dispatch:u` | −0.468 % | 0.255 | 7/10 | 0.34 | −10.6 |
| `ls_dispatch.store_dispatch:u` | −0.461 % | 0.276 | 7/10 | 0.34 | −4.2 |

**Node-identical**: `5,001,562` and `1,739,588` on every binary in this study.
**Gate green**: `All checks passed`, eval hash `2e1ede95…79c4`, d13 `1739588`,
`Bench: 587683`, and `perft_full_suite` PASSED (the D24 ritual is mandatory
here — `attacks` is touched). **No cycles claim** (§2.4).

For scale: study 52 landed −36.5 i/n, study 35 −25.0, study 36 −19.5.

### 0.2 ★ The single best result is `magic53` ALONE, and the combination is worse on loads

The three-panel picture is consistent and it is the study's most useful finding
after §0.0. `magic53` on its own is the only arm that is **10/10 with a tight
`se` on all four exact counters at once**:

| head-to-head, n=10 | instr/n | branch/n | **loads/n** | stores/n |
|---|---:|---:|---:|---:|
| `magic53` alone | −34.9 | −18.2 | **−22.3** (10/10, se 0.137) | **−14.5** (10/10) |
| `land53` = `+gc53` | **−39.3** | −19.7 | −10.6 (7/10, ns) | −4.2 (7/10, ns) |
| `full53` = `+cont53` | **−46.6** | **−26.1** | −4.9 (8/10, ns) | −3.1 (8/10, ns) |

**Instructions add almost exactly** — predicted −39.4 from the single-arm
panel, measured −39.3, an error of 0.1 i/node — **and loads do not.** Stacking
`gc53` onto `magic53` gives back ~12 loads/node that `magic53` alone had
deleted, and `cont53` gives back more. This reproduces across all three panels
and it is **study 52 §5.2's allocator-fragility rule biting a COMBINATION rather
than a single edit** — the first evidence for that rule from a source other than
the arm that produced it, which study 52 §7 item 7 explicitly asked for.

`land53` was landed rather than `magic53` alone because `instructions:u` is the
counter study 52 §4 licenses and the only one with a 0.000 null here. **That is
a judgement call between two counters that disagree, and it is recorded as one**
— a reader who weights loads above instructions should land `magic53` alone, and
§6 item 2 keeps the question open.

### 0.3 ★ `cont53` measured, priced, and NOT landed

The third item works: **−13.4 i/n, −13.6 branches/n on its own**, and the
`full53` combination reaches **−46.6 i/n**. It stays behind its feature because
of what it costs, which the commission required be stated:

**+1 179 648 B = +1.125 MiB per `Histories`, i.e. per WORKER THREAD**, taking
the catalogue from **19.979 MiB to 21.104 MiB** and **breaching design/09 §8
criterion 2's ≤21 MiB budget** — a `const` assertion the arm has to raise to
compile. On a 15 GB box shared with a live trainer, and with multi-threaded
search in the roadmap, ~7 extra instructions/node per thread is not obviously
worth 1.125 MiB per thread. Landing it is a design-doc change (design/09 §8),
not a perf change, and it should be decided there.

### 0.4 The store-forwarding axis: NULLED, and one counter FAILED its null

Study 52 §0.6 opened this and said explicitly *"do not build on this number
until it has a null and a known-answer probe."* This study ran the null.

**The rates reproduce (rule 8).** `ls_bad_status2.stli_other` = **60.03/node**
and `ls_stlf` = **113.19/node** against study 52's 59.7 and 111.7 — a 34.7 %
failure rate, independently confirmed on a different night.

**But `ls_stlf` fails its byte-identical null:** `null53` reads **+0.540 %,
se 0.189, 9/10 positive, p = 0.0215**. A binary identical to the reference moves
the counter significantly and in a consistent direction.
`ls_bad_status2.stli_other` passes its null (−0.986 %, se 0.661, 6/10, p = 0.75)
but with a ~1 % floor at n = 10, far too coarse to price a mechanism worth
tenths of a percent.

**The axis therefore stays UNPRICED — now for a measured reason rather than for
lack of time.** No known-answer probe was built, so no per-event cost is
established, and nothing in this study depends on these counters. The one arm
that moved `stli_other` significantly was `cont53` (−3.735 %, se 0.519, 10/10,
p = 0.002); **no mechanism is claimed for it** — all three arms delete memory
operations, so both forwarding counters fall with them, and there is no
per-load normalisation here to separate "fewer conflicts" from "fewer loads".

---

## 1. THE PREMISES, VERIFIED IN THE DISASSEMBLY

The commission asked for item 1 to be checked against the shipped disassembly
rather than trusted. It checks out, and the mechanism named is exactly right —
but the cost is **larger** than claimed, by one instruction class nobody
mentioned.

### 1.1 The caller reads the mailbox, spills it, stores through a `Box`, and the callee reads the mailbox again

`search::<NonPv>` in `base53`, the move loop, verbatim from
`objdump -d --section=.text`:

```
5eb8f: movzbl %r14b,%eax                    ; from-square
5eb93: movzbl 0x6430(%rbx,%rax,2),%ecx      ; mailbox[from].color        <-- CALLER load 1
5eb9b: mov    %cl,0x5f(%rsp)                ;   spilled
5eb9f: cmp    $0x2,%cl                      ;   Option discriminant
5eba2: je     5fab7                         ;   -> expect_failed
5eba8: movzbl 0x6431(%rbx,%rax,2),%eax      ; mailbox[from].piece_type   <-- CALLER load 2
5ebb0: lea    0x6368(%rbx),%rdi             ; &self.board
5ebb7: mov    %r15d,%esi                    ; m
5ebba: mov    %rax,0xe8(%rsp)               ;   spilled
5ebca: mov    0x6340(%rbx),%rax             ; self.stack        <-- the Box'd pointer
5ebd1: inc    %r12d
5ebd9: mov    %r12d,0xc(%rax,%rcx,4)        ; stack[cur].move_count = move_count
5ebe1: call   51260 <Board::gives_check>    ; re-derives BOTH loads
```

That store at `5ebd9` is the barrier the commission predicted: it goes through
`self.stack`'s heap pointer, which LLVM cannot prove disjoint from
`self.board.mailbox`, so the two loads at `5eb93`/`5eba8` are dead by the time
control reaches the callee and must be re-issued.

And the callee, at `0x51260`:

```
51266: cmpb   $0x2,0xc8(%rdi,%rcx,2)        ; mailbox[from].color        (again)
5126e: je     512fd                         ;   -> expect_failed         (again)
51274: movzbl 0xc9(%rdi,%rcx,2),%eax        ; mailbox[from].piece_type   (again)
5127c: cmp    $0x6,%rax
51280: jae    51321                         ;   -> panic_bounds_check    <-- NOT in the brief
51286: mov    0x40(%rdi,%rax,8),%r9         ; check_squares[piece_type]
```

**The third item is the one the commission missed.** `check_squares(pt)` indexes
a six-element array. When `pt` is a byte loaded out of the mailbox it carries no
range information, so LLVM must emit `cmp $0x6` / `jae panic_bounds_check`.
`PieceType` is `#[repr(u8)]` with six variants, so passing `pc` **by value**
hands LLVM the `0..=5` validity range and the check folds. Per call, the fix
therefore deletes **two loads, one `expect` panic edge, and one bounds check**.

### 1.2 The neighbouring call sites are already clean — this was the last one

Audited for the same defect class in the same move loop:

* **`Board::legal`** — its `piece_on(from)` is inside a `debug_assert_eq!` and
  compiles out in release; the live arms read `blockers_for_king`,
  `attackers_to_exist` and `aligned`. Clean.
* **`Board::capture_stage`** — pure occupancy-bitboard tests, never touches the
  mailbox. Clean.
* **`Worker::do_move`** — already takes `moved_piece: Option<Piece>` as an
  argument, and study 35's `after_move` already replaced its mailbox reads with
  the `StateInfo` stores `make_move_with` makes anyway.

So `gives_check` was the **last remaining duplicate mailbox derivation** in the
hot move loop, not one of many.

### 1.3 Item 2's premise, and the oracle that licenses the mask

`src/attacks.rs`: `table: Box<[Bitboard]>` — a genuine runtime-length slice, so
every `rook_attacks`/`bishop_attacks` pays a length load, a `cmp`, a `jae` and a
panic block. `TABLE_SIZE` = `ROOK_TABLE_SIZE` (0x19000 = 102 400) +
`BISHOP_TABLE_SIZE` (0x1480 = 5 248) = **107 648**; the next power of two is
**131 072**, so the padding is **23 424 entries = 187 392 B**.

**The commission required the init-time oracle be confirmed to cover every
reachable slot before the mask could be believed. It did not, so I built one.**
The existing `init_magics` verifies each magic is collision-free against the
naive oracle over the complete Carry-Rippler subset enumeration, and the
`offset - base_offset == slider.table_entries()` assertion catches packing bugs
— but neither statement is "every index `Magic::index` can produce is
`< TABLE_SIZE`", which is what the mask needs. Step 6 now asserts exactly that,
through `Magic::index` itself (the hot path's own arithmetic), over that same
exhaustive enumeration, as a real `assert!`:

```rust
for &occ in &occupancy[..size] {
    let idx = found.index(Bitboard::from_u64(occ));
    assert!(idx >= offset && idx < offset + size && idx < TABLE_SIZE, ...);
}
```

This matters because the failure mode changes character: today an out-of-range
index panics; behind a mask it would silently read a zeroed padding slot and
return an empty attack set. 107 648 assertions run once at startup, inside an
init already measured at ~21 ms.

### 1.4 Item 3's premise, and its price

`CONT_NB = 3328` (= 4 × 13 × 64), not a power of two, so `cont[id.0 as usize]`
cannot be proved in range and every `cont_ref`/`cont_mut` carries a bounds
check. Padding to 4096 and masking folds it.

**The cost is 768 extra slabs × 1536 B = 1 179 648 B = +1.125 MiB, per
`Histories`, i.e. per WORKER THREAD.** That takes the catalogue from
**19.979 MiB to 21.104 MiB** and **breaches design/09 §8 criterion 2's ≤21 MiB
budget**, which is enforced by a `const` assertion. The arm raises that guard
explicitly rather than deleting it, so the breach is visible in the diff.

Reindexing cannot dodge the cost: the awkward factor is the 13-piece axis
(12 pieces + the "no piece" sentinel), and widening it to 16 to make the id a
pure bit-concatenation lands on the *same* 4096 slabs.

---

---

## 2. THE PANELS

Harness: `tools/perf53/exact53.sh` + `med53.py`, rebased verbatim from studies
49/52 — **blocked Latin square** (rotate within a block of NLAB, reshuffle the
base between blocks; a plain Latin square IS cyclic rotation, study 43's
defect). Every panel carries `null53`, a byte-for-byte copy of `base53`, riding
in the rotation. Deltas are paired **within** round.

Realised mean panel position was the ideal value for **every label in every
panel** (2.50 / 1.50 / 2.00 / 2.00), so no position bias is in play.

### 2.1 Main panel — n = 18, six labels (`docs/research/perf-artifacts-53/panel-main.txt`)

Reference: `instructions:u` 6116.58, `branches:u` 657.89, `ld_dispatch` 2260.24,
`store_dispatch` 902.25 per node.

| arm | instr % | **i/n** | branch % | **br/n** | ld % | **ld/n** | store % | **st/n** |
|---|---:|---:|---:|---:|---:|---:|---:|---:|
| `null53` | 0.000 | **0.0** | −0.000 | **0.0** | −0.098 | −2.2 | −0.082 | −0.7 |
| `ctl53` | −0.112 | −6.9 | −1.155 | −7.6 | +0.147 | +3.3 | −1.057 | −9.5 |
| `gc53` | −0.185 | −11.3 | −1.393 | −9.2 | −0.189 | −4.3 | −0.731 | −6.6 |
| **`magic53`** | **−0.571** | **−34.9** | **−2.767** | **−18.2** | **−0.773** | **−17.5** | **−1.442** | **−13.0** |
| `cont53` | −0.219 | −13.4 | −2.063 | −13.6 | −0.178 | −4.0 | −1.074 | −9.7 |

Every arm is **18/18 sign-consistent on `instructions:u` and `branches:u` with
se 0.000 and p = 0.0000**.

### 2.2 ★ The control was NOT inert, so the arms are decomposed

Study 52 required an inert control that compiles byte-identical. **Mine did not.**
`ctl53` — the same source with every feature OFF, i.e. the `gives_check`/
`gives_check_by` split and the `slot()` helper and nothing else — has `.text` md5
`2dec2c5b` against `base53`'s `e160e553`, 12 fewer `panic_bounds_check` sites,
and is **−6.9 i/n and −7.6 branches/n on its own, 18/18, se 0.000**.

That is the opposite of study 52's experience (its inert binding compiled
byte-identical), and it means the single-arm column above **cannot** be read as
the feature's effect. The marginal effect of each feature, `arm − ctl53`:

| feature, marginal over `ctl53` | i/n | br/n | ld/n | st/n |
|---|---:|---:|---:|---:|
| `gc53` | −4.5 | −1.6 | −7.6 | +2.9 |
| **`magic53`** | **−28.1** | **−10.6** | **−20.8** | −3.5 |
| `cont53` | −6.5 | −6.0 | −7.3 | −0.2 |

**I did not pin the mechanism of `ctl53`'s own −6.9 i/n.** Merging panic blocks
cannot reduce *dynamic* instructions (they never execute), so the likeliest
cause is that routing `gives_check` through a wrapper changed an inlining
decision somewhere; that is a hypothesis, not a measurement, and it is recorded
as unresolved (§6 item 1).

### 2.3 Confirmation and head-to-head panels

Because §2.2 forbids assuming additivity, the combinations were **built and
measured**, not summed.

**Confirmation, n = 16, four labels** (`panel-comb.txt`), null: instr −0.000,
branches −0.000, ld −0.060, store −0.091, cycles +0.024 ± 0.304.

| arm | instr % | **i/n** | branch % | **br/n** | ld % | ld/n | store % | st/n |
|---|---:|---:|---:|---:|---:|---:|---:|---:|
| **`land53`** | −0.643 | **−39.3** | −2.999 | −19.7 | −0.271 | −6.1 | −0.250 | −2.3 |
| **`full53`** | −0.762 | **−46.6** | −3.965 | −26.1 | −0.381 | −8.6 | −0.594 | −5.4 |

**Head-to-head, n = 10, five labels** (`panel-head.txt`) — the panel that ranks
them, run inside ONE panel precisely because cross-panel ranking is how study 50
manufactured a false headline. Null here is clean on all four:
instr 0.000, branches −0.000, **ld +0.012 (5/10)**, **store +0.019 (5/10)**.

| arm | instr % | **i/n** | branch % | **br/n** | ld % (se, n) | **ld/n** | store % | **st/n** |
|---|---:|---:|---:|---:|---:|---:|---:|---:|
| `magic53` | −0.571 | −34.9 | −2.767 | −18.2 | **−0.989** (0.137, 10/10) | **−22.3** | **−1.610** (10/10) | **−14.5** |
| `land53` | −0.643 | **−39.3** | −2.999 | −19.7 | −0.468 (0.255, 7/10) | −10.6 | −0.461 (7/10) | −4.2 |
| `full53` | −0.762 | **−46.6** | −3.965 | **−26.1** | −0.217 (0.192, 8/10) | −4.9 | −0.345 (8/10) | −3.1 |

### 2.4 NO CYCLES CLAIM, and the box said so twice

The byte-identical null on `cycles:u` read **−0.237 % (se 0.248)** in the main
panel, **+0.024 % (se 0.304)** in the confirmation panel and **−1.488 %
(se 1.523)** in the head-to-head, whose reference cycles/node had drifted from
3681 to 3848 in the meantime. Every arm's cycles median sits inside that spread.
The counter cannot resolve this and **no cycles claim is made for anything** —
study 52's standard, and the same conclusion studies 35, 49 and 51 each reached
with their own nulls.

Honesty note in the other direction: `full53`'s cycles median is −1.122 %
(se 0.230, 12/16) in the confirmation panel and −3.223 % (se 1.150, 8/10) in the
head-to-head. Both point the right way and neither is significant. **They are
reported so nobody can say the favourable readings were dropped, not as
evidence.**

### 2.5 The load/store counters have a small non-zero null

In the main panel the byte-identical null read **−0.098 % on `ld_dispatch` at
14/18 (p = 0.031)** and −0.082 % on `store_dispatch` at 14/18. In the
head-to-head it was clean (+0.012 / +0.019, 5/10). So the load and store columns
carry a floor of order ±0.1 % that varies between panels, and small load deltas
(`gc53`'s −4.3/node, `cont53`'s −4.0/node) are **near that floor and should not
be quoted alone**. `magic53`'s −17.5 to −22.3 loads/node is 8–20× the floor and
is the one load result this study will stand behind.

---

## 3. THE ITEMS, PRICED

| # | item | landed? | i/n | loads/n | cost |
|---|---|---|---:|---:|---|
| 1 | `gives_check_by` | **yes** | −4.5 marginal | −7.6 marginal | none |
| 2 | **magic table mask** | **yes** | **−28.1 marginal** | **−20.8 marginal** | 187 392 B of never-read pages |
| 3 | `CONT_NB` → 4096 | **no** | −6.5 marginal | −7.3 marginal | **+1.125 MiB per thread; breaches design/09 §8** |
| — | the refactor alone (`ctl53`) | **yes** (substrate) | −6.9 | +3.3 | none |

Item 2 has the broadest reach exactly as the commission predicted — the mask
sits under `rook_attacks`/`bishop_attacks`/`queen_attacks`, which movegen,
`see_ge`, `compute_threats` and `attackers_to_exist` all call — and it deleted
**31 static `panic_bounds_check` sites** against `ctl53`'s 327.

---

## 4. GATES

| gate | result |
|---|---|
| **`CHECK_NNUE_GATE=1 CHECK_TEST_THREADS=1 bash tools/check.sh`** | **All checks passed** |
| wl50 eval hash | **`2e1ede9521624dd8…93cf79c4`** — exact |
| d13 node total | **1 739 588** — exact |
| `Bench:` trailer (depth-6 default build) | **587683** — unmoved |
| **D24 perft ritual** (`attacks` touched) | **`perft_full_suite` PASSED** |
| node identity, study frame — all 9 binaries | **5 001 562** exact |
| d13 node identity — all 9 binaries | **1 739 588** exact |
| `final53` vs measured `land53` | 6077.24 i/n both; `.text` md5 stable at `121e12b7` across the test edits |
| byte-identical null, `instructions:u` | **0.000 %**, se 0.000 (all panels) |
| byte-identical null, `branches:u` | **−0.000 %**, se 0.000 (all panels) |
| realised mean panel position | **ideal for every label in every panel** |

**No pin moves, and none should** — every arm is behaviour-preserving, so a
moved pin would have been evidence that one of the equivalences was wrong.

### 4.1 The gate caught two real defects, and both are kept

1. **`attacks::tests::packing_structural_invariants` failed**, asserting
   `table.len() == TABLE_SIZE` (107 648) against the new 131 072. Rather than
   relax it, it now pins the invariant that actually matters: the allocation is
   `TABLE_ALLOC`, the packed content still ends at `TABLE_SIZE`, and **every
   padding slot must be empty** — a non-empty one would mean a magic wrote
   outside its reserved block, which is precisely the corruption the mask turns
   from a panic into a silent wrong attack set.
2. **clippy flagged the constant-valued assertions**; they moved to module-level
   `const _: () = assert!(…)`, so they now fail the **build** rather than a test.

---

## 5. WHY EACH EQUIVALENCE IS EXACT

* **`gives_check_by`** — the body is `gives_check`'s verbatim; the only change
  is that `pc` arrives as an argument. The two are the same function by
  construction, and the wrapper still resolves `pc` the old way for every cold
  caller. Two `debug_assert`s pin the precondition (`pc.color == side_to_move`,
  and `Some(pc) == piece_on(from)`).
* **the magic mask** — sound iff every reachable index is `< TABLE_SIZE`, which
  `init_magics` step 6 now asserts **exhaustively at startup in every build**,
  through `Magic::index` itself, over the complete Carry-Rippler enumeration
  (§1.3). Off-mask occupancy bits are masked away inside `index`, so they cannot
  widen the range.
* **`CONT_NB` padding** — `ContHistId::new` is a bijection onto `0..3328`
  (a `const` assertion and a sweep test already pin this), so masking with 4095
  is a no-op on every id the engine can construct.

Node identity on both fingerprints, on all nine binaries, is the empirical
check on all three.

---

## 6. WHAT I COULD NOT ESTABLISH

Most doubtful first.

1. **`ctl53`'s own −6.9 i/n is unexplained.** A pure refactor with features off
   moved a deterministic counter by 18/18 with se 0.000, and I did not find the
   mechanism. My inlining hypothesis (§2.2) is untested. It also means **this
   study has no byte-identical source-level control**, only a byte-identical
   *binary* null — a weaker instrument than study 52 had, and the reason every
   arm is reported both raw and marginal.
2. **Whether `land53` or `magic53`-alone is the better landing is not settled by
   the data.** Instructions favour `land53` by 4.4 i/n (exact, 10/10, se 0.000);
   loads favour `magic53` by ~12 loads/node (10/10, se 0.137 vs 7/10, ns). I
   landed on the instruction counter because study 52 §4 licenses it and its
   null is exactly zero. `cycles:u` would break the tie and cannot (§2.4).
3. **Everything here is NON-PGO.** The shipping binary is PGO, study 52 §0.1
   showed the symbol's share moves 2.06× between the two pipelines, and study 35
   §9 item 3 left the same question open for its own arm. The mechanisms
   *should* survive — the aliasing barrier that blocks the `gives_check` CSE is
   the `stack[cur].move_count` store, which is present whether or not the callee
   is inlined, and the magic mask folds a check inside `attacks.rs` that is
   inlined everywhere in both pipelines — **but I did not build a PGO pair and
   this is an inference, not a measurement.**
4. **No Elo estimate, and no games.** −0.64 % of instructions is not −0.64 % of
   time, and this study has no measurement that converts one to the other. Every
   binary is node-identical, so a fixed-node match is a dead heat by
   construction (study 49 §8 item 6, study 52 §9.1) and **no match is owed**;
   the wins are real only insofar as instruction count buys time, which is
   precisely what `cycles:u` could not confirm on this box.
5. **The store-forwarding numbers are observations, not results** (§0.4). One
   counter failed its null, the other has a ~1 % floor, no known-answer probe
   was built, and `cont53`'s significant −3.735 % on `stli_other` has no
   mechanism attached.
6. **`do_move`'s 260.3 i/node and 116.3 loads/node are still uncensused.** That
   was study 49's stated open item and this study did not close it — it attacked
   the *neighbourhood* (three call sites reachable from the same move loop)
   rather than the body. §0.0(b) explains why the 461 % framing that pointed at
   the body does not justify the trip; a census still might.
7. **The allocator-fragility generalisation now has two independent sources but
   is still not a model.** Study 52 saw it in one arm plus one control; this
   study sees it in a *combination* of arms that individually behave. That is
   corroboration, and rule 6 says a tidy corroboration is when to attack — I
   could not attack it further here, and nobody can yet predict the sign or size
   of the spill-back before building.
8. **187 392 B of padding is asserted never-read, not measured never-read.** The
   step-6 oracle proves unreachability and the test proves the slots stay empty;
   neither measures whether the extra pages cost anything in TLB terms. The
   allocation is virtually contiguous and the pages are touched once at init, so
   the expectation is zero, but it is an expectation.

---

## 7. REPRODUCTION

```bash
# 0. a fresh worktree lacks every gitignored path -- provision by symlink
mkdir -p data/nnue/nets/screens
ln -sfn /path/to/d48x-1280-sb520-ema-s600.cenn \
        data/nnue/nets/screens/d48x-1280-sb520-ema-s600.cenn

# 1. the binaries. base53 must print .text md5 e160e553 (= study 52's rfp52b)
git checkout fefa1ed5 -- src/ Cargo.toml     # the pre-study base
bash tools/perf53/build53.sh base53
cp bin53/base53 bin53/null53                 # the byte-identical null
git checkout perf-study-53-domove -- src/ Cargo.toml
bash tools/perf53/build53.sh final53
bash tools/perf53/build53.sh cont53 cont53

# 2. the node gate BEFORE any timing -- both fingerprints, every binary
for b in base53 null53 final53 cont53; do
  ./bin53/$b bench 16 1 100000 tools/nnue/calib/wl50.fens nodes \
      engines/canonical/net.cenn     # 5001562
  ./bin53/$b bench 16 1 13 tools/nnue/calib/wl50.fens depth \
      engines/canonical/net.cenn     # 1739588
done

# 3. the panel. The null rides IN the rotation; rounds must be a multiple
#    of the label count or med53.py truncates to whole blocks.
P53_EV='instructions:u,branches:u,ls_dispatch.ld_dispatch:u,ls_dispatch.store_dispatch:u,cycles:u' \
P53_PANEL=docs/research/perf-artifacts-53/panel-head.txt \
  bash tools/perf53/exact53.sh 15 base53 null53 magic53 land53 full53
python3 tools/perf53/med53.py docs/research/perf-artifacts-53/panel-head.txt \
  base53 null53 magic53 land53 full53

# 4. the store-forwarding null -- ls_stlf FAILS it, expect ~+0.5% on null53
P53_EV='instructions:u,ls_stlf:u,ls_bad_status2.stli_other:u' \
P53_PANEL=docs/research/perf-artifacts-53/panel-stlf.txt \
  bash tools/perf53/exact53.sh 12 base53 null53 gc53 magic53 cont53

# 5. gates -- the D24 ritual is MANDATORY here, attacks is touched
cargo test --release -- --ignored perft_full_suite
CHECK_NNUE_GATE=1 CHECK_TEST_THREADS=1 bash tools/check.sh
```

---

## 8. WHAT THE NEXT STUDY SHOULD RUN

1. **Settle §6 item 2 on a quiet box.** Build `magic53`, `land53` and a
   byte-identical null, get a `cycles:u` null under 0.15 %, and let cycles break
   the instructions-vs-loads tie. This is a 20-minute panel the moment the box
   is idle, and it decides whether `gc53` should be reverted.
2. **Census `do_move`'s body** — study 49's genuine open item (§6 item 6):
   260.3 i/node and 116.3 loads/node, never decomposed.
   `tools/perf52/ibsline52.py` is the instrument and it already does per-inline-
   frame attribution; point it at `Worker::do_move` rather than at
   `search::<NonPv>`. **Do not price the arm off the 461 %** (§0.0b).
3. **Rebuild study 35's `hookc`** (−11.36 i/n, unlanded, never PGO-tested) now
   that "fusion is dead" is known to be a misattribution (§0.0a). Study 35 §9
   item 3 predicted the ordering would hold under PGO and did not test it.
4. **Take the PGO pair this study skipped** (§6 item 3) — one `buildpgo52.sh`
   pair of `base53`/`final53` would convert the largest inference here into a
   measurement, and study 52 §9.2 already specified the protocol (both sides
   PGO'd, and panel the same source against itself first to bound the
   retraining noise, a null nobody in this repo has ever taken).
5. **Do not spend more on the store-forwarding axis without a known-answer
   probe** (§0.4). `ls_stlf` is measurably biased; `stli_other` is too coarse.
   Study 49's `skidprobe` is the template.
