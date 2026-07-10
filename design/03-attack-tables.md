# 03 — Attack tables: leapers, rays, fancy magic bitboards

Status: Draft v1

This document specifies the `attacks` module (D13): every precomputed attack
table the engine will ever consult. Leaper attacks (knight, king, pawn), the
ray-geometry tables (`BETWEEN`, `LINE`, `RAY_PASS`, Chebyshev `DISTANCE`) are
built at **compile time** with `const fn` (D12). Slider attacks (bishop, rook,
queen) use **fancy magic bitboards**: one shared, densely packed attack table
(102,400 rook entries / 800 KB + 5,248 bishop entries / 41 KB), with the magic
multipliers found at startup by seeded trial-and-error and stored behind a
`std::sync::LazyLock` (D12). The module's contract is a handful of `#[inline]`
free functions — `attacks(pt, sq, occ)`, `pawn_attacks(color, sq)`,
`between(s1, s2)`, `line(s1, s2)`, `distance(s1, s2)` — plus a hoistable
`tables() -> &'static AttackTables` handle for hot loops. Correctness is
anchored to a naive on-the-fly ray generator that everything else is
cross-checked against exhaustively.

## 1. Purpose and scope

This document owns:

- The `src/attacks.rs` module: its data structures, initialization
  algorithms, public API, and tests.
- The compile-time tables: `KNIGHT_ATTACKS`, `KING_ATTACKS`, `PAWN_ATTACKS`,
  `PSEUDO_ATTACKS` (empty-board slider rays), `BETWEEN`, `LINE`, `RAY_PASS`,
  `DISTANCE`.
- The fancy-magic slider machinery: `Magic`, `MagicPair`, `AttackTables`, the
  packed shared table, and the full init algorithm (Carry-Rippler subset
  enumeration, sparse candidate generation, popcount prefilter, epoch-array
  verification, hardcoded per-rank PRNG seeds).
- The `Prng` (xorshift64\*) helper, defined here as `pub(crate)` because the
  Zobrist key generator will reuse it.

This document does NOT own:

- **Move generation** — which squares a piece may legally move to given pins,
  checks, castling rights. That is the movegen design doc
  (`design/05-movegen-perft.md`, filename per suite index). This module answers
  only "which squares does a piece on `sq` attack given occupancy `occ`".
- **Board state** — checkers/blockers/pinners computation lives with
  `StateInfo` in the make/unmake doc (`design/04-state-make-unmake.md`, per
  D8); it *consumes* `between`/`line`/`attacks` from here.
- **SEE** — `see_ge` is a `Board` method (D11) in its own doc; it consumes
  `attacks` with progressively x-rayed occupancy.
- **Set-wise movegen shift helpers** beyond `pawn_attacks_bb` — bulk pawn push
  generation belongs to movegen.
- `PawnPushOrAttacks` (push∪attack union table): in current Stockfish its only
  consumers are the NNUE threat features and incremental dirty-threat
  tracking. Deferred to the NNUE design doc; the builder is a two-liner on top
  of tables specified here.

## 2. References

- `stockfish_reference/01-bitboards-attacks-move-encoding.md` — distilled
  notes on SF's attack machinery, table sizes, magic init.
- `stockfish_reference/00-design-implications.md` §"Attack tables" — the
  action list this doc executes.
- Stockfish source (read-only, ideas/constants only per D27):
  - `stockfish_reference/Stockfish/src/attacks.h` — `Magic` struct, index
    formula, constexpr `PseudoAttacks`, `sliding_attack`, `safe_destination`,
    dispatch functions.
  - `stockfish_reference/Stockfish/src/attacks.cpp` — `init_magics` (seeds,
    Carry-Rippler, prefilter, epoch array), `LineBB`/`BetweenBB`/`RayPassBB`
    construction.
  - `stockfish_reference/Stockfish/src/misc.h` — `PRNG` (xorshift64\*),
    `sparse_rand`.
  - `stockfish_reference/Stockfish/src/bitboard.h` — `SquareDistance`
    (Chebyshev).
- Our code: `src/types.rs` (`Square`, `Color`, `PieceType`), `src/bitboard.rs`
  (`Bitboard` and const constructors), `src/board.rs` (consumer-side APIs).

## 3. Design

### 3.1 Module layout and public API

New module `src/attacks.rs`, re-exported from `src/lib.rs` as `pub mod
attacks;`. Everything below uses OUR existing types (`Square::index() -> u8`,
`Bitboard::from_u64`/`to_u64` are already `const fn`).

```rust
// ---- compile-time tables (all static: see §3.8 on const-vs-static) ----
pub static KNIGHT_ATTACKS: [Bitboard; 64];
pub static KING_ATTACKS:   [Bitboard; 64];
/// Indexed [Color as usize][Square]; attacks FROM sq BY a pawn of that color.
pub static PAWN_ATTACKS:   [[Bitboard; 64]; 2];
/// Empty-board slider rays, indexed by square. Queen = bishop | rook.
pub static PSEUDO_BISHOP:  [Bitboard; 64];
pub static PSEUDO_ROOK:    [Bitboard; 64];
pub static PSEUDO_QUEEN:   [Bitboard; 64];
static BETWEEN:  [[Bitboard; 64]; 64];   // access via between()
static LINE:     [[Bitboard; 64]; 64];   // access via line()
static RAY_PASS: [[Bitboard; 64]; 64];   // access via ray_pass()
static DISTANCE: [[u8; 64]; 64];         // access via distance()

// ---- leaper / geometry accessors (all #[inline], table lookups) ----
pub fn knight_attacks(sq: Square) -> Bitboard;
pub fn king_attacks(sq: Square) -> Bitboard;
pub fn pawn_attacks(color: Color, sq: Square) -> Bitboard;
/// Set-wise pawn attacks; const so movegen and eval can use it anywhere.
pub const fn pawn_attacks_bb(color: Color, pawns: Bitboard) -> Bitboard;
/// Empty-board attacks; debug-asserts pt != Pawn (pawns need a color).
pub fn pseudo_attacks(pt: PieceType, sq: Square) -> Bitboard;
/// Squares strictly between s1 and s2 OR'd with s2 (SF semantics, §3.4).
pub fn between(s1: Square, s2: Square) -> Bitboard;
/// Full line through both squares incl. endpoints; EMPTY if not aligned.
pub fn line(s1: Square, s2: Square) -> Bitboard;
/// Ray from s1 (exclusive) through s2 to the edge; EMPTY if not aligned.
pub fn ray_pass(s1: Square, s2: Square) -> Bitboard;
/// Chebyshev distance: max(|Δfile|, |Δrank|).
pub fn distance(s1: Square, s2: Square) -> u8;
/// True if s3 lies on the (extended) line through s1 and s2.
pub fn aligned(s1: Square, s2: Square, s3: Square) -> bool;

// ---- slider attacks (magic-backed, runtime-initialized) ----
/// Force/fetch the magic tables; hoist this in hot loops (§3.7).
pub fn tables() -> &'static AttackTables;
pub fn bishop_attacks(sq: Square, occupied: Bitboard) -> Bitboard;
pub fn rook_attacks(sq: Square, occupied: Bitboard) -> Bitboard;
pub fn queen_attacks(sq: Square, occupied: Bitboard) -> Bitboard;
/// General dispatch. debug-asserts pt != Pawn.
pub fn attacks(pt: PieceType, sq: Square, occupied: Bitboard) -> Bitboard;

impl AttackTables {
    pub fn bishop_attacks(&self, sq: Square, occupied: Bitboard) -> Bitboard;
    pub fn rook_attacks(&self, sq: Square, occupied: Bitboard) -> Bitboard;
    pub fn queen_attacks(&self, sq: Square, occupied: Bitboard) -> Bitboard;
    pub fn attacks(&self, pt: PieceType, sq: Square, occupied: Bitboard) -> Bitboard;
}
```

`attacks` dispatch shape (SF-identical semantics: knight/king/pseudo for
non-sliders, queen = two lookups OR'd — there is no queen magic table):

```rust
#[inline]
pub fn attacks(pt: PieceType, sq: Square, occupied: Bitboard) -> Bitboard {
    debug_assert!(!matches!(pt, PieceType::Pawn), "use pawn_attacks(color, sq)");
    match pt {
        PieceType::Knight => knight_attacks(sq),
        PieceType::King   => king_attacks(sq),
        PieceType::Bishop => bishop_attacks(sq, occupied),
        PieceType::Rook   => rook_attacks(sq, occupied),
        PieceType::Queen  => queen_attacks(sq, occupied),
        PieceType::Pawn   => Bitboard::EMPTY, // unreachable in debug
    }
}
```

### 3.2 Compile-time foundations: safe steps and the naive ray generator

Everything const-built rests on two `const fn` primitives. Const builders
operate on raw `u64` internally (operator traits are not const, §3.8) and wrap
into `Bitboard` at the boundary.

**Safe destination.** A step from square `sq` is valid iff the target index is
in `0..64` AND the file distance between source and target is at most 2. Any
genuine knight/king/slider step changes file by ≤ 2; a horizontal wraparound
changes file by ≥ 6, so this single check rejects all wraps (this is exactly
SF's `safe_destination` rule).

```rust
/// Bit of the target square, or 0 if the step leaves the board / wraps.
const fn safe_destination(sq: i32, step: i32) -> u64 {
    let to = sq + step;
    if to < 0 || to > 63 { return 0; }
    let file_delta = ((sq & 7) - (to & 7)).abs();
    if file_delta > 2 { 0 } else { 1u64 << to }
}
```

**Naive slider generator.** Walks each direction until the edge or the first
blocker (blocker square included in the attack set — capture semantics).
Direction step constants on our little-endian rank-file mapping (D3):
N=+8, S=−8, E=+1, W=−1, NE=+9, NW=+7, SE=−7, SW=−9.

```rust
const ROOK_DIRS:   [i32; 4] = [8, -8, 1, -1];
const BISHOP_DIRS: [i32; 4] = [9, 7, -7, -9];

const fn sliding_attack(dirs: [i32; 4], sq: u8, occupied: u64) -> u64 {
    let mut attacks = 0u64;
    let mut d = 0;
    while d < 4 {
        let mut s = sq as i32;
        loop {
            let dest = safe_destination(s, dirs[d]);
            if dest == 0 { break; }
            attacks |= dest;
            s += dirs[d];
            if occupied & dest != 0 { break; }
        }
        d += 1;
    }
    attacks
}

pub(crate) const fn rook_attacks_naive(sq: u8, occ: u64) -> u64 {
    sliding_attack(ROOK_DIRS, sq, occ)
}
pub(crate) const fn bishop_attacks_naive(sq: u8, occ: u64) -> u64 {
    sliding_attack(BISHOP_DIRS, sq, occ)
}
```

The naive generator is the **correctness oracle** for this entire module: the
magic init verifies against it by construction, the const geometry tables are
defined in terms of it, and the test suite cross-checks the packed tables
against it (§5).

### 3.3 Leaper tables

Step sets (facts, same on any LERF board): knight `{-17, -15, -10, -6, 6, 10,
15, 17}`, king `{-9, -8, -7, -1, 1, 7, 8, 9}`.

```rust
const fn leaper_attack(steps: [i32; 8], sq: u8) -> u64 {
    let mut b = 0u64;
    let mut i = 0;
    while i < 8 { b |= safe_destination(sq as i32, steps[i]); i += 1; }
    b
}

pub static KNIGHT_ATTACKS: [Bitboard; 64] = {
    let mut t = [Bitboard::EMPTY; 64];
    let mut s = 0;
    while s < 64 {
        t[s] = Bitboard::from_u64(leaper_attack([-17, -15, -10, -6, 6, 10, 15, 17], s as u8));
        s += 1;
    }
    t
};
// KING_ATTACKS identical with the king step set.
```

**Pawn attacks** are built from the set-wise shift function, which is public
and const because movegen/eval want it for whole pawn sets. White pawns attack
NW (+7, illegal from file A) and NE (+9, illegal from file H); black mirrors
(−9 illegal from file A, −7 from file H):

```rust
pub const fn pawn_attacks_bb(color: Color, pawns: Bitboard) -> Bitboard {
    let p = pawns.to_u64();
    let not_a = p & !Bitboard::FILE_A.to_u64();
    let not_h = p & !Bitboard::FILE_H.to_u64();
    Bitboard::from_u64(match color {
        Color::White => (not_a << 7) | (not_h << 9),
        Color::Black => (not_a >> 9) | (not_h >> 7),
    })
}

pub static PAWN_ATTACKS: [[Bitboard; 64]; 2] = { /* per-square from pawn_attacks_bb */ };
```

`PAWN_ATTACKS` is defined for all 64 squares, including ranks 1 and 8 where no
pawn can stand (SF does the same); entries there are still geometrically
correct shifts and harmless.

**Pseudo (empty-board) slider tables**: `PSEUDO_ROOK[s] =
rook_attacks_naive(s, 0)`, `PSEUDO_BISHOP[s] = bishop_attacks_naive(s, 0)`,
`PSEUDO_QUEEN[s]` their union. Used for: alignment tests during const builds,
`check_squares` seeds later, and cheap "could this piece ever reach that
square" prefilters in movegen/SEE.

Sizes: each `[Bitboard; 64]` table is 512 B; `PAWN_ATTACKS` 1 KB. All live in
`.rodata`, zero startup cost.

### 3.4 Ray geometry: BETWEEN, LINE, RAY_PASS, DISTANCE

All four are 64×64 tables built at compile time from the naive generator.
Stockfish builds these at startup *after* magics because it reuses
`attacks_bb`; we don't need to — the defining formulas only require slider
attacks with zero-or-one blockers, which `*_attacks_naive` computes in const
context. Outputs are bit-identical to SF's.

Semantics and defining formulas (for each ordered pair `(s1, s2)`; `pt` ranges
over {Bishop, Rook}; "aligned on pt" means `PSEUDO_pt[s1]` contains `s2`; two
distinct squares are aligned on at most one of the two):

- **LINE** — the full line through both squares, endpoints included; empty if
  unaligned. For the aligned `pt`:
  `LINE[s1][s2] = (naive(pt, s1, 0) & naive(pt, s2, 0)) | bit(s1) | bit(s2)`.
  Symmetric. `LINE[s][s] = EMPTY`.
- **BETWEEN** — squares strictly between the two, **OR'd with the target
  square `s2`** — SF's exact semantics, chosen because the dominant consumer
  is check evasion, where `between(king_sq, checker_sq)` is directly the
  "block the check or capture the checker" target mask, and pin handling,
  where `between(king_sq, pinner_sq)` contains the pinner (capturable). For
  the aligned `pt`:
  `BETWEEN[s1][s2] = (naive(pt, s1, bit(s2)) & naive(pt, s2, bit(s1))) | bit(s2)`;
  for unaligned pairs (and `s1 == s2`) it is just `bit(s2)`. The `| bit(s2)`
  is unconditional — every entry contains the target square. NOT symmetric.
- **RAY_PASS** — the ray from `s1` (exclusive) through `s2` and past it to the
  board edge; empty if unaligned. For the aligned `pt`:
  `RAY_PASS[s1][s2] = naive(pt, s1, 0) & (naive(pt, s2, bit(s1)) | bit(s2))`.
  Consumer in SF is the NNUE dirty-threat update (discovered-attack rays); we
  build it now because it shares the same builder loop and test harness, and
  costs 32 KB of `.rodata`.
- **DISTANCE** — Chebyshev distance `max(|file(s1)−file(s2)|,
  |rank(s1)−rank(s2)|)` as `u8`. Used by eval (king proximity), some movegen
  shortcuts, and tests.

```rust
static BETWEEN: [[Bitboard; 64]; 64] = {
    let mut t = [[Bitboard::EMPTY; 64]; 64];
    let mut s1 = 0;
    while s1 < 64 {
        let mut s2 = 0;
        while s2 < 64 {
            let bit2 = 1u64 << s2;
            let mut b = 0u64;
            if PSEUDO_BISHOP[s1].to_u64() & bit2 != 0 {
                b = bishop_attacks_naive(s1 as u8, bit2)
                  & bishop_attacks_naive(s2 as u8, 1u64 << s1);
            } else if PSEUDO_ROOK[s1].to_u64() & bit2 != 0 {
                b = rook_attacks_naive(s1 as u8, bit2)
                  & rook_attacks_naive(s2 as u8, 1u64 << s1);
            }
            t[s1][s2] = Bitboard::from_u64(b | bit2);
            s2 += 1;
        }
        s1 += 1;
    }
    t
};
// LINE, RAY_PASS: same double loop shape with their formulas; DISTANCE trivial.
```

`aligned(s1, s2, s3)` = `!(line(s1, s2) & Bitboard::from_square(s3)).is_empty()`
— the movegen legality workhorse ("does this move stay on the pin ray").

Sizes: `BETWEEN`, `LINE`, `RAY_PASS` are 32 KB each; `DISTANCE` 4 KB. Total
const data for this module ≈ 103 KB of `.rodata`.

### 3.5 Fancy magic bitboards: data structures

The magic lookup answers `slider_attacks(sq, occ)` with: mask the occupancy to
the relevant blocker squares, multiply by a per-square magic constant so the
scattered mask bits collect into the top bits, shift down, index a per-square
subtable. Formula (fact, SF-identical):

```text
index   = ((occ & mask) * magic) >> shift        // wrapping 64-bit multiply
shift   = 64 - popcount(mask)
attacks = table[base_offset + index]
```

**"Fancy"** means each square's subtable has its own size `2^popcount(mask)`
and they are packed back-to-back in one shared array, each square's block
starting where the previous ended — as opposed to "plain" magics where every
square gets a worst-case-sized 4096-entry block.

Table sizes (facts, SF-current): the mask excludes board edges (a blocker on
the edge never changes the attack set beyond it), so rook masks have 10–12
relevant bits and bishop masks 5–9. Summed over 64 squares:

| Block  | Entries              | Bytes  |
|--------|----------------------|--------|
| Rook   | 0x19000 = 102,400    | 800 KB |
| Bishop | 0x1480  = 5,248      | 41 KB  |
| Total  | 107,648              | 841 KB |

Rust structures:

```rust
#[derive(Clone, Copy)]
#[repr(C)]
struct Magic {
    mask:   u64,  // relevant-blocker mask (edges excluded)
    magic:  u64,  // multiplier found at init
    offset: u32,  // this square's block start within AttackTables::table
    shift:  u32,  // 64 - popcount(mask)
}

impl Magic {
    #[inline]
    fn index(&self, occupied: Bitboard) -> usize {
        self.offset as usize
            + ((((occupied.to_u64() & self.mask).wrapping_mul(self.magic)) >> self.shift)
                as usize)
    }
}

/// Both magics for one square packed into exactly one cache line.
#[repr(C, align(64))]
#[derive(Clone, Copy)]
struct MagicPair {
    bishop: Magic, // 24 bytes
    rook:   Magic, // 24 bytes, 16 bytes tail padding
}
const _: () = assert!(std::mem::size_of::<MagicPair>() == 64);

pub struct AttackTables {
    magics: [MagicPair; 64],       // 4 KB, cache-line aligned per square
    table:  Box<[Bitboard]>,       // len == TABLE_SIZE, rook block then bishop block
}

pub const ROOK_TABLE_SIZE:   usize = 0x19000; // 102_400
pub const BISHOP_TABLE_SIZE: usize = 0x1480;  //   5_248
const TABLE_SIZE: usize = ROOK_TABLE_SIZE + BISHOP_TABLE_SIZE;
```

Layout notes:

- SF stores an absolute `Bitboard*` per square; we store a `u32` offset into
  the shared `Box<[Bitboard]>` instead — no self-referential struct, no
  unsafe, and 4 bytes smaller. The add of `offset` folds into the load's
  addressing on x86.
- SF declares `alignas(64) Magic Magics[64][2]` — array-level alignment so
  each square's two 32-byte Magics share a cache line. Our `MagicPair` with
  `repr(C, align(64))` expresses the same guarantee per element ("repr align
  64" per D12): one line fetch serves both slider types for a square, which
  queen lookups always want.
- Rook block occupies `table[0 .. 0x19000]`, bishop block
  `table[0x19000 .. 0x1A480]`. Single allocation, per the packed-shared-table
  requirement (D12).

Lookup methods:

```rust
impl AttackTables {
    #[inline]
    pub fn rook_attacks(&self, sq: Square, occupied: Bitboard) -> Bitboard {
        self.table[self.magics[sq.index() as usize].rook.index(occupied)]
    }
    #[inline]
    pub fn bishop_attacks(&self, sq: Square, occupied: Bitboard) -> Bitboard {
        self.table[self.magics[sq.index() as usize].bishop.index(occupied)]
    }
    #[inline]
    pub fn queen_attacks(&self, sq: Square, occupied: Bitboard) -> Bitboard {
        self.rook_attacks(sq, occupied) | self.bishop_attacks(sq, occupied)
    }
}
```

### 3.6 Magic initialization algorithm (in full)

`AttackTables::new()` runs once (inside the `LazyLock`), calling
`init_magics(slider, base_offset, &mut table, &mut magics)` twice: rook with
base 0, bishop with base `ROOK_TABLE_SIZE`. Per slider type, for each square
`s` in a1..h8 order:

**Step 1 — mask.** Full empty-board ray minus edge squares that are not on
`s`'s own rank/file (an edge square on the slider's own line IS relevant when
`s` itself is on that edge):

```rust
let edges = ((Bitboard::RANK_1 | Bitboard::RANK_8) & !Bitboard::RANKS[s.rank() as usize])
          | ((Bitboard::FILE_A | Bitboard::FILE_H) & !Bitboard::FILES[s.file() as usize]);
let mask = naive_attack(slider, s, Bitboard::EMPTY) & !edges;
```

**Step 2 — shift and offset.** `shift = 64 - mask.count()`. `offset` = base
offset for the first square, otherwise previous square's `offset` + previous
square's subtable size. Subtable size is `1 << mask.count()`.

**Step 3 — Carry-Rippler subset enumeration.** Enumerate every subset of
`mask` (every possible relevant-blocker configuration) and record the true
attack set for each, using the trick that `(b - mask) & mask` steps through
all subsets of `mask` in a cycle starting and ending at 0:

```rust
let mut occupancy = [0u64; 4096];   // 2^12 = max rook subtable
let mut reference = [0u64; 4096];
let mut size = 0;
let mut b = 0u64;
loop {
    occupancy[size] = b;
    reference[size] = naive_attack(slider, s, Bitboard::from_u64(b)).to_u64();
    size += 1;
    b = b.wrapping_sub(mask.to_u64()) & mask.to_u64();
    if b == 0 { break; }
}
// size == 1 << mask.count()
```

Note `wrapping_sub`: the subtraction intentionally wraps when `b <
mask`-as-integer; a plain `-` panics in debug builds (§3.8).

**Step 4 — candidate search with prefilter and epoch verification.** Seed a
fresh PRNG from the hardcoded per-rank seed table, then try sparse random
candidates until one maps all `size` occupancies collision-free:

```rust
let mut rng = Prng::new(MAGIC_SEEDS[s.rank() as usize]);
let mut epoch = [0u32; 4096];       // shared across all squares of this slider type
let mut attempt = 0u32;             // ditto

// find a working magic for square s:
let mut i = 0;
while i < size {
    // (a) candidate generation with quality prefilter
    let mut magic = 0u64;
    while (magic.wrapping_mul(mask.to_u64()) >> 56).count_ones() < 6 {
        magic = rng.sparse_rand();
    }
    // (b) verification pass
    attempt += 1;
    i = 0;
    while i < size {
        let idx = (((occupancy[i] & mask.to_u64()).wrapping_mul(magic)) >> shift) as usize;
        if epoch[idx] < attempt {
            epoch[idx] = attempt;                       // first claim this attempt
            table[offset + idx] = Bitboard::from_u64(reference[i]);
        } else if table[offset + idx].to_u64() != reference[i] {
            break;                                      // harmful collision → next candidate
        }
        i += 1;
    }
}
// loop exits with table[offset..offset+subtable] fully and consistently filled
```

The pieces, and why each exists:

- **Sparse candidates** (`sparse_rand` = AND of three `rand64()` outputs, ~8
  bits set on average): empirically, good magics are sparse; sampling sparse
  numbers directly makes the search converge orders of magnitude faster than
  uniform sampling.
- **Popcount quality prefilter** (`popcount((magic * mask) >> 56) >= 6`,
  constants exact per SF): a cheap necessary-ish condition — a good magic must
  push many mask bits into the top byte. Rejecting weak candidates before the
  O(size) verification pass saves most of the work.
- **Epoch array instead of table clearing**: a failed candidate leaves garbage
  in the subtable. Rather than memsetting up to 4096 entries per retry, each
  slot remembers the attempt number that last wrote it (`epoch[idx]`). A slot
  with a stale epoch is "empty" for the current attempt. Benign collisions
  (two occupancies mapping to the same index with the SAME attack set) are
  allowed and are precisely why magic tables can be smaller than perfect
  hashing would suggest.
- **Hardcoded per-rank PRNG seeds** (SF-current values for 64-bit, D20):

  ```rust
  const MAGIC_SEEDS: [u64; 8] = [728, 10316, 55013, 32803, 12281, 15100, 16645, 255];
  ```

  Indexed by `s.rank()`, re-seeding per square. These were tuned (by the SF
  project) so the whole search finishes in a few milliseconds. They are tuned
  against SF's exact PRNG consumption order, so our candidate loop must
  consume the PRNG identically: prefilter rejections also advance the PRNG
  (three `rand64` calls each), as sketched above. SF also has a 32-bit seed
  row; we are 64-bit only and omit it, along with the 32-bit split-multiply
  index variant.

- **PRNG** — xorshift64\* (constants exact):

  ```rust
  pub(crate) struct Prng { s: u64 }
  impl Prng {
      pub(crate) const fn new(seed: u64) -> Self { assert!(seed != 0); Prng { s: seed } }
      pub(crate) const fn rand64(&mut self) -> u64 {
          self.s ^= self.s >> 12;
          self.s ^= self.s << 25;
          self.s ^= self.s >> 27;
          self.s.wrapping_mul(2685821657736338717) // 0x2545F4914F6CDD1D
      }
      pub(crate) const fn sparse_rand(&mut self) -> u64 {
          self.rand64() & self.rand64() & self.rand64()
      }
  }
  ```

  `pub(crate)` because the Zobrist doc reuses it (SF seeds its Zobrist PRNG
  with 1070372; recorded here so both docs agree on the generator), and
  `const fn` because that reuse is compile-time: the Zobrist tables are
  const-evaluated (doc 04 §3.4) while the magic search keeps calling the
  same generator at runtime.

**Step 5 — packing invariant.** After both passes, hard-assert the
accumulated offsets: rook block consumed exactly `0x19000` entries, bishop
exactly `0x1480`. This catches any mask/enumeration bug at startup rather
than as silent table overlap.

Scratch space: `occupancy` + `reference` (32 KB each) + `epoch` (16 KB) ≈
80 KB of stack per `init_magics` call — fine on any thread that can run
`LazyLock` init; box them if we ever care.

The whole init is deterministic (fixed seeds) and takes ~21 ms (measured,
§6).

### 3.7 Storage: LazyLock, and hoisting in hot loops

```rust
static TABLES: LazyLock<AttackTables> = LazyLock::new(AttackTables::new);

#[inline]
pub fn tables() -> &'static AttackTables { &TABLES }

#[inline]
pub fn rook_attacks(sq: Square, occupied: Bitboard) -> Bitboard {
    tables().rook_attacks(sq, occupied)
}
// bishop_attacks / queen_attacks / attacks: same pattern.
```

`std::sync::LazyLock` (stable, zero dependencies, D15) gives thread-safe
one-time init in safe Rust. Cost model: every deref performs an atomic
acquire load of the init state plus a perfectly-predicted branch. That is
cheap but not free in a loop issuing millions of lookups per second, so:

- **Hoist**: hot code (movegen loops, SEE, eval) calls `tables()` once at
  function entry and uses the `&'static AttackTables` methods thereafter; the
  one check is amortized over the whole call. Later, each search `Worker`
  (D23) stores `tables: &'static AttackTables` at construction, removing the
  check from the search path entirely.
- **Force at startup**: `main` calls `LazyLock::force(&TABLES)` (or just
  `attacks::tables()`) before entering the UCI loop, so init never happens
  mid-search and the branch is always-taken thereafter.

The convenience free functions stay available for cold paths (tests, UCI
helpers, perft driver) where the extra load is irrelevant.

### 3.8 Rust-specific concerns

- **`const fn` capabilities we rely on** (all stable): `while`/`loop`,
  mutable locals, arrays and indexing, integer arithmetic including
  `wrapping_*`, `i32::abs`, `u64::count_ones`, `match` on our field-less
  enums, and calling other `const fn`s (`Bitboard::from_u64`/`to_u64`/
  `from_square` are already const). No floating point is needed anywhere in
  this module.
- **`const fn` limits we must design around**: `for` loops are not allowed in
  const context (trait-based iteration isn't const) — all builders use
  `while` with manual counters. Trait operator impls (`BitOr` for `Bitboard`)
  are not `const` — builders compute in raw `u64` and wrap at the end.
  No heap allocation in const — fine, everything is fixed-size arrays.
- **Const-eval budget**: recent rustc has no hard const-eval step limit (only
  the `long_running_const_eval` lint for runaway evaluation). Our heaviest
  build is three 64×64 tables × a few naive slider calls each — on the order
  of 10^6–10^7 interpreter steps, adding well under a second of compile time.
- **`static`, not `const`, for big tables**: a `const` is a value inlined at
  every use site and can be duplicated in the binary; `static` guarantees one
  copy in `.rodata`. All 64×64 tables are `static` (private, behind accessor
  fns). The 512 B per-piece tables may be either; we use `static` uniformly.
- **Wrapping arithmetic is mandatory, not stylistic**: the magic multiply
  overflows by design, and the Carry-Rippler subtraction underflows by
  design. Debug builds panic on plain `*`/`-` overflow, so `wrapping_mul` /
  `wrapping_sub` are required for the code to pass `cargo test` at all.
- **Bounds checks**: `table[offset + idx]` carries a slice bounds check the
  optimizer cannot elide (the in-bounds proof — `idx < 2^popcount(mask)` and
  `offset + 2^popcount(mask) <= TABLE_SIZE` — lives in the init algorithm,
  invisible to the compiler). We ship the safe version first; if a profile
  shows it, D14(c) permits `get_unchecked` with a `// SAFETY:` comment citing
  the init-time packing assertion (§3.6 step 5) and the measurement.
- **Why not PEXT**: current Stockfish itself no longer uses BMI2 PEXT for
  slider indexing — the default x86-64 path is the plain multiply-and-shift
  magic (D12; verified in `attacks.h`, which has only magic, hyperbola-
  quintessence for aarch64/loongarch, and an AVX2 dual-HQ variant). The
  multiply is as fast as PEXT on modern cores, avoids the catastrophic
  microcoded-PEXT penalty on Zen 1/2, and needs no `unsafe` intrinsics or
  second code path in Rust. We implement exactly one slider backend.
- **`#[inline]` on accessors**: the crate is lib + bin; without `#[inline]`
  (or LTO) the tiny lookup fns wouldn't inline into the binary crate. Release
  strength builds use fat LTO anyway (D16), but `#[inline]` keeps debug/test
  perf tolerable too.

### 3.9 Alternative documented but not chosen: hardcoded magic constants

Instead of searching at startup, we could run the search once, print the 128
`(magic, shift, offset)` triples, and paste them as `const` arrays — or go
further and emit the entire 841 KB attack table into the binary via a build
script, making everything const and deleting `LazyLock`.

Rejected for now (D12): startup search costs ~21 ms (§6) and is
self-verifying against the naive oracle on every launch — a subtle table bug
cannot survive a run. Hardcoded constants would be dead weight to review, and
a full baked table bloats the binary by ~841 KB while removing the
verification. Revisit only if startup time ever matters (e.g. massive
ultra-short-TC test harnesses); the design keeps this door open since
consumers only see `tables()`.

## 4. Implementation plan

Each step is one commit, lands green (`cargo test`, `cargo clippy
--all-targets` clean), and touches nothing outside its stated scope.

1. **Const foundations + leapers.** `src/attacks.rs` with
   `safe_destination`, `sliding_attack`, `rook/bishop_attacks_naive`,
   `leaper_attack`, `KNIGHT_ATTACKS`, `KING_ATTACKS`, `pawn_attacks_bb`,
   `PAWN_ATTACKS`, `PSEUDO_*` tables and their accessor fns; `pub mod
   attacks;` in `lib.rs`. Done when: unit tests for known attack sets and
   count properties (§5.1) pass.
2. **Geometry tables.** `DISTANCE`, `BETWEEN`, `LINE`, `RAY_PASS` const
   builders, `between`/`line`/`ray_pass`/`distance`/`aligned` accessors.
   Done when: property tests (§5.2) pass and documented edge cases
   (`BETWEEN[s][s] == {s}`, `LINE[s][s]` empty) are asserted.
3. **Magic machinery.** `Prng`, `Magic`, `MagicPair`, `AttackTables`,
   `init_magics`, `LazyLock`, slider accessors, packing hard-asserts, layout
   static-asserts (`size_of::<MagicPair>() == 64`). Done when: a seeded
   random cross-check (1,000 occupancies × 64 squares × both sliders vs
   naive) passes and two independent `AttackTables::new()` calls produce
   identical magics (determinism).
4. **Exhaustive validation + startup forcing.** Full test battery of §5
   (all-subset exact test, 10,000 random occupancies per square, edge
   occupancies), `main` forces `tables()` at startup. Done when: full suite
   passes; total `cargo test` wall time still acceptable (< ~30 s debug).
5. **Bench + docs.** Criterion dev-dependency (first one; allowed per D15),
   `benches/attacks.rs` measuring rook/bishop/queen lookup on a random
   occupancy stream; record numbers in the commit message. Done when: bench
   runs and the measured numbers are recorded (§6 targets).

## 5. Testing and validation

### 5.1 Leaper unit tests

- Hand-verified sets using named square constants:
  `knight_attacks(Square::A1) == {B3, C2}`,
  `knight_attacks(Square::E4)` is the known 8-square set,
  `king_attacks(Square::H8) == {G7, G8, H7}`,
  `pawn_attacks(Color::White, Square::E4) == {D5, F5}`,
  `pawn_attacks(Color::Black, Square::A5) == {B4}`.
- Count properties over all 64 squares: king attack counts ∈ {3, 5, 8} with
  the exact corner/edge/interior distribution (4/24/36); knight counts
  ∈ {2, 3, 4, 6, 8}; pawn attack counts ∈ {1, 2}.
- Wraparound: no `pawn_attacks`/`knight_attacks`/`king_attacks` result on
  file A contains file H squares and vice versa (bulk check via
  `FILE_A`/`FILE_H` masks).
- `pawn_attacks_bb(c, full-rank set)` equals union of per-square entries.

### 5.2 Geometry property tests

For all 4096 ordered pairs `(s1, s2)`:

- `between(s1, s2)` always contains `s2`; equals exactly `{s2}` iff the
  strict-between set is empty — the pair is unaligned, equal, or
  adjacent-aligned (the E1/E2 case below; amended during step 2, the original
  "unaligned or equal" iff contradicted that bullet).
- Aligned pairs: `between(s1, s2) ^ Bitboard::from_square(s2)` equals the
  independently recomputed strict-between set
  `naive(pt, s1, bit(s2)) & naive(pt, s2, bit(s1))` — i.e. the table agrees
  with slider attacks by construction, checked from a separately-written
  test-side formula.
- `line(s1, s2)` symmetric; empty iff unaligned; contains both endpoints
  when aligned; every `s3` in `line(s1, s2)` satisfies
  `line(s1, s3) == line(s1, s2)` (when `s3 != s1`).
- `ray_pass(s1, s2)`: empty iff unaligned; for aligned pairs it excludes
  `s1`, contains `s2`, is a subset of `line(s1, s2)`, and
  `between(s1, s2) ⊆ ray_pass(s1, s2)`.
- `distance` symmetric, zero iff equal, equals
  `max(|file Δ|, |rank Δ|)` recomputed inline; `distance(A1, H8) == 7`.
- Adjacent aligned squares: `between(E1, E2) == {E2}` (evasion-mask edge
  case: checker adjacent to king → only capture, no block).

### 5.3 Slider magic validation (the core gate)

- **All-subset exact test**: for each square and each slider type, enumerate
  every subset of the magic mask with the Carry-Rippler (written independently
  in the test) and assert `magic lookup == naive`. This re-proves what init
  verified, but through the public API — it catches offset/packing bugs, not
  just magic-collision bugs.
- **Random full-board occupancies**: for each square, ≥ 10,000 occupancies
  from a seeded test PRNG — mixed density: uniform `rand64()`, sparse
  (`rand64 & rand64 & rand64`), and dense (`rand64 | rand64`) — assert
  rook/bishop/queen lookups equal the naive generator. Off-mask bits are the
  point here: full-board occupancy exercises the `& mask` step. Seeded ⇒
  reproducible failures.
- **Edge occupancies**: `EMPTY` (must equal `PSEUDO_*`), `FULL`, occupancy
  equal to the mask itself, occupancy with only the square's own bit set.
- **Structural asserts** (run at init, re-checked in a test): rook offsets
  consume exactly 102,400 entries, bishop exactly 5,248; `shift ==
  64 - popcount(mask)` for all 128 magics; subtable ranges are disjoint and
  cover `0..TABLE_SIZE`.
- **Determinism**: two `AttackTables::new()` runs yield identical magic
  arrays (guards accidental PRNG-order changes that would silently slow init).

### 5.4 Regression strategy

- This module lands before movegen, so its own exhaustive suite is the gate
  now. Once movegen + perft exist, the perft suite (D24) becomes the ongoing
  end-to-end regression net — any attack-table corruption fails perft
  immediately — but the local suite stays as the *diagnosing* layer: perft
  says "broken", these tests say "where".
- Any future change to this module (e.g. unchecked indexing per D14, layout
  changes) must leave the whole §5.3 battery untouched and passing, and must
  not change `bench` node counts (attack tables are functionally invisible —
  identical outputs — so any bench delta means a bug).

## 6. Performance notes

Targets and expectations (x86-64-v3, D16 release profile):

- Slider lookup critical path: AND + 64-bit MUL + SHR + two dependent loads
  (magic metadata, then table entry) — ≈ 10–15 cycles latency, heavily
  pipelined across a movegen loop. Criterion target: **≤ 5 ns per
  rook/bishop lookup, ≤ 10 ns per queen lookup** on a hot cache with a random
  occupancy stream (throughput-mode measurement). Measured at step 5
  (`benches/attacks.rs`, Ryzen 7 7700X / WSL2, x86-64-v3): **~1.1 ns rook /
  ~0.9 ns bishop / ~1.8 ns queen** per hoisted hot lookup — targets met with
  ~4x headroom; the free-function path costs ~4–8% more (≈ +0.04–0.10
  ns/lookup amortized), which is the measured LazyLock-deref price the §3.7
  hoisting guidance rests on. Full baseline table: `docs/perf-baseline.md`.
- Startup: full magic search **≤ 25 ms** (measured once at step 4: ~21 ms
  release on a Ryzen 7 7700X / WSL2, stable across runs and unchanged by
  `target-cpu=x86-64-v3`; the draft's "≤ 10 ms, expect ~1–5 ms" guess
  undercounted the search — with the SF-identical seeds and consumption
  order the verification loop runs 158,190 rook + 46,656 bishop attempts,
  while Carry-Rippler enumeration is only ~1 ms of the total. Amended here
  per the deviation rule; the D12 mechanism is unchanged). Assert nothing
  about it in tests (CI machines vary); it was measured once.

What to measure (criterion benches, dev-only):

- `rook_attacks`/`bishop_attacks`/`queen_attacks` on a pre-generated vec of
  (square, occupancy) pairs — measures the table path incl. bounds check.
- The same through the free-function path vs. a hoisted `tables()` handle —
  quantifies the LazyLock check so the hoisting guidance rests on a number.
- Re-run after movegen exists: perft nps is the real judge (D24).

Pitfalls (each already designed around, listed for review vigilance):

1. **LazyLock deref in inner loops** — hoist via `tables()`; force at startup
   (§3.7).
2. **Debug-build overflow panics** — `wrapping_mul`/`wrapping_sub` in the
   index function, prefilter, and Carry-Rippler (§3.8).
3. **`const` table duplication** — large tables are `static` (§3.8).
4. **Bounds checks** — accepted initially; D14(c) escape hatch documented
   with the exact invariant to cite (§3.8).
5. **Cache footprint** — the 800 KB rook table exceeds most L2s; that is the
   same trade Stockfish makes and real search locality is much better than
   uniform (popular squares/occupancies dominate). Nothing to do, but don't
   be surprised by cold-cache microbench numbers ~3–4× worse than hot.
6. **PRNG consumption order** — changing the candidate loop shape silently
   de-tunes the hardcoded seeds; the determinism test plus a startup-time
   eyeball on any refactor covers this.

## 7. Interactions

Depends on (all already implemented, no upstream design doc):

- `src/types.rs` — `Square` (LERF mapping D3), `Color`, `PieceType`.
- `src/bitboard.rs` — `Bitboard` const constructors, `FILES`/`RANKS`
  constants, set operators.

Feeds (by design-doc filename; names per suite index, adjust on
reconciliation):

- `design/04-state-make-unmake.md` — `StateInfo` derived fields (D8):
  checkers via `attacks`/`pawn_attacks`, blockers/pinners via `line` +
  slider lookups with modified occupancy, `check_squares` via
  `attacks(pt, ksq, occ)`; gives-check tests use `between`/`aligned`.
- `design/05-movegen-perft.md` — every generator loop consumes
  `attacks`/`pawn_attacks_bb`; evasions consume `between(ksq, checker)`
  directly as the block-or-capture mask (the reason BETWEEN includes the
  target square); pin legality uses `aligned`.
- The SEE doc (per D11, `design/06-see.md` or folded into the board doc) —
  iterated `attacks` lookups on x-ray-updated occupancy.
- The Zobrist section of the make/unmake doc — reuses `Prng` (xorshift64\*)
  defined here.
- The eval and NNUE docs — `distance`, `PSEUDO_*`; the NNUE threat-feature
  doc additionally consumes `ray_pass` and will add `PAWN_PUSH_OR_ATTACKS`
  (deferred from here, §1).

## 8. Acceptance criteria (D26)

The component is done when ALL of the following hold:

1. `cargo test` passes with the full §5 battery: leaper hand-cases and count
   properties; all 4096-pair geometry property tests; all-subset exact slider
   test; ≥ 10,000 seeded random occupancies × 64 squares × {rook, bishop,
   queen} against the naive generator; edge occupancies; determinism test.
2. Init hard-asserts fire on nothing: rook block exactly 102,400 entries,
   bishop exactly 5,248, all shifts equal `64 − popcount(mask)`, subtable
   ranges disjoint and exhaustive.
3. Compile-time layout asserts hold: `size_of::<MagicPair>() == 64`,
   `align_of::<MagicPair>() == 64`.
4. `cargo clippy --all-targets` is clean; zero new runtime dependencies
   (criterion is dev-only) (D15).
5. `main` forces table init before the UCI/print path runs; startup measured
   once for the magic search on a dev machine and recorded in §6 (~21 ms;
   amended at step 4 from the draft's "≤ 10 ms" guess — the mechanism (D12)
   is unchanged; not CI-enforced).
6. Criterion bench exists and recorded numbers meet §6 targets (≤ 5 ns hot
   rook lookup on x86-64-v3), with the LazyLock hoisted-vs-unhoisted delta
   documented in the bench commit.
7. Public API surface exactly as §3.1 (names may not drift without updating
   this doc), documented with `///` comments including the BETWEEN
   includes-target semantics.

## 9. Open questions

1. **Sibling doc filenames**: §7 cites `design/04-state-make-unmake.md`,
   `design/05-movegen-perft.md`, `design/06-see.md` by expected name; the
   suite index should reconcile exact filenames once all docs exist.
2. **Unchecked table indexing**: deliberately deferred, not decided — ship
   safe indexing, then measure per D14(c). If the bounds check is invisible
   in perft nps (expected), it stays forever.
3. **`Worker`-cached `&'static AttackTables` vs. per-function hoisting**:
   final call belongs to the search/threads doc (D23); this doc guarantees
   both patterns are supported and cheap.
