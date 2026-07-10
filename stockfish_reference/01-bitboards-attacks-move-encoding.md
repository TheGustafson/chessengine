# Bitboards, attack tables, and core encodings

Source: `types.h`, `bitboard.{h,cpp}`, `attacks.{h,cpp}`. In current Stockfish
the attack machinery lives in `attacks.{h,cpp}` (namespace `Attacks`);
`bitboard.cpp` only builds `PopCnt16` and `SquareDistance`.

## Move encoding (types.h:414-486)

`Move` is one u16:
- bits 0-5: **to** square, bits 6-11: **from** square (note: opposite order
  from our engine — ours is from in 0-5, to in 6-11; functionally irrelevant,
  don't churn ours)
- bits 12-13: promotion piece stored as `pt - KNIGHT` (matches ours)
- bits 14-15: MoveType — NORMAL=0, PROMOTION, EN_PASSANT, CASTLING (matches ours)

Sentinels: `Move::none() = 0`, `Move::null() = 65`. Trick: any pattern with
`from == to` can never be a real move, so those encodings are free.
`is_ok()` = "differs from both". EN_PASSANT flag is set only when a pawn can
actually be captured. There is a `to_sq_unchecked()` for branchless hot paths.

## Square/Piece/Color arithmetic (types.h)

- Color flip: `c ^ 1`. Piece color flip: `pc ^ 8`.
- `Piece` packs color in bit 3: `W_PAWN=1..W_KING=6`, `B_PAWN=9..B_KING=14`,
  `PIECE_NB=16`. `type_of = pc & 7`, `color_of = pc >> 3`. Flat 16-entry
  piece-indexed arrays everywhere (e.g. `PieceValue[16]`); the color gap slots
  are zero. (We use `[2][6]` nested arrays — fine, but flat-16 indexing by a
  packed piece byte is worth considering when history tables arrive.)
- `flip_rank(s) = s ^ 56`, `flip_file(s) = s ^ 7`,
  `relative_square(c, s) = s ^ (c * 56)` — XOR transforms, no branches.
- `Value` is int; VALUE_MATE = 32000, mate/TB thresholds derived by
  subtracting MAX_PLY (246). Piece values: P 208, N 781, B 825, R 1276, Q 2538.

## Sliding attacks: three compile-time paths (attacks.h:29-37)

1. **Fancy magic bitboards** — default on x86-64. NOTE: current SF does NOT
   use PEXT in the sliding-attack index anymore; it's a plain
   multiply-and-shift magic even on BMI2 machines.
2. **Hyperbola Quintessence** on aarch64 (hardware bit-reverse, tiny tables).
3. **Dual HQ with AVX2** — computes bishop+rook attacks simultaneously in one
   256-bit op; rank attacks from a `[8][256]` u8 table (2 KB).

### Magic path details (attacks.cpp:98-160)

`struct Magic { Bitboard mask; Bitboard* attacks; Bitboard magic; unsigned shift; }`
Index: `((occupied & mask) * magic) >> shift`, then `attacks[index]`.

- Tables: RookTable = 0x19000 entries (**800 KB**), BishopTable = 0x1480
  (**41 KB**). "Fancy" = one shared array, each square's `attacks` pointer
  offset by the previous square's size (variably-sized subtables, densely
  packed).
- Magics found at startup by trial and error:
  - Enumerate all occupancy subsets of the mask with the **Carry-Rippler**:
    `b = (b - mask) & mask` until b == 0.
  - Candidate magics from `sparse_rand` (AND of three rand64s, ~8 bits set),
    quick-filtered by `popcount((magic * mask) >> 56) >= 6`.
  - Verified with an **epoch array** (`epoch[4096]`, attempt counter) so a
    failed candidate doesn't require clearing the table.
  - Per-rank PRNG seeds hardcoded so search converges in a few ms.
- `Magics[64][2]` is `alignas(64)` (cache-line aligned).

## Precomputed tables

| Table | Size | When built | Purpose |
|---|---|---|---|
| `PseudoAttacks[8][64]` | 4 KB | **constexpr** (compile time) | empty-board attacks; rows 0/1 reused for white/black pawn attacks |
| `PawnPushOrAttacks[2][64]` | 1 KB | constexpr | push+attack union |
| `LineBB[64][64]` | 32 KB | init() | full line through 2 squares (or 0) |
| `BetweenBB[64][64]` | 32 KB | init() | strictly-between **OR'd with s2** — directly usable as "block or capture the checker" mask |
| `RayPassBB[64][64]` | 32 KB | init() | ray from s1 through/past s2 |
| `SquareDistance[64][64]` | 4 KB | init() | Chebyshev distance |
| `PopCnt16[65536]` | 64 KB | init() | popcount fallback only |
| Rook+Bishop magic tables | ~841 KB | init() (few ms) | sliding attacks |

Total static ≈ **1 MB**, dominated by the rook table.

## Bit ops

- popcount/lsb/msb: compiler intrinsics behind cfg switches
  (`__builtin_popcountll`, `ctzll`, `63 ^ clzll`). Rust: `count_ones()`,
  `trailing_zeros()`, `leading_zeros()` compile to the same instructions with
  `target-cpu` set — no work needed.
- `pop_lsb`: read `lsb` then `b &= b - 1` (we already do this).
- `more_than_one(b) = b & (b - 1)` — cheap "is this double check" test.
- `b & -b` isolates the LSB as a bitboard.

## Rust implications

- Keep our Move layout; add a `Move::NULL`-style distinct null (from==to
  trick already covered by our `Move::NULL` = a1a1).
- Build `PseudoAttacks`/`Between`/`Line` as `const` tables via `const fn`
  (Rust can do at compile time what SF does at startup).
- Magic init: runtime `std::sync::OnceLock` (or generate once into a source
  file). Matching SF means fancy-magic with Carry-Rippler + epoch
  verification. `#[repr(align(64))]` on the per-square magic struct.
- Compile with `target-cpu=native` (or `x86-64-v3`) so popcount/BMI happen;
  consider the AVX2 dual-HQ path only as a later experiment.
