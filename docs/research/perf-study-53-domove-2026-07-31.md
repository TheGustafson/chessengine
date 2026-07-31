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

### 0.1 ★ What LANDED

*(filled in from the completed panel — see §4)*

### 0.2 The three items, and what each one actually costs

*(see §3)*

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

*(sections 2–8 completed below once the panel finished)*
