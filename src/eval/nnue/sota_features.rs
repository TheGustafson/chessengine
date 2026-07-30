//! The SOTA net's **SF-faithful feature front-end** (design/11 §12.1-12.2, D40):
//! the two feature-index functions (HalfKAv2_hm + FullThreats) and the
//! from-scratch accumulator refresh that fills the `pst`/`threat` pair the
//! Reckless-shaped [`super::sota_head`] consumes.
//!
//! This is a **reimplementation of the index math** in Stockfish's
//! `half_ka_v2_hm.{h,cpp}` and `full_threats.{h,cpp}` — the arithmetic and the
//! constant tables (OrientTBL, KingBuckets, PieceSquareIndex, the threat LUTs)
//! are transcribed, but no GPLv3 code is copied (D27): the loops, types, and
//! structure are our own idiomatic Rust. Correctness is pinned by the tests at
//! the bottom, which hand-compute the SF `make_index` for known tuples.
//!
//! Behind the `sota` cargo feature; the default build never compiles it, so the
//! depth-6 bench stays 548978.
//!
//! # What this increment covers
//! - [`halfka_index`]: the 22,528-dim HalfKAv2_hm index.
//! - [`threat_index`]: the 60,720-dim FullThreats index.
//! - [`refresh_pst`] / [`refresh_threat`] / [`refresh_both`]: from-scratch
//!   accumulator builds (format-agnostic — they take weight slices, like
//!   [`super::sota_head`]).
//!
//! # TODO — later #29 increments (deliberately NOT here)
//! - Incremental (dirty) HalfKA updates on the `MakeMoveHooks` seam (§3.4.4):
//!   ≤2 sub / ≤2 add per move from `DirtyPiece`.
//! - The HalfKA "Finny table" accumulator cache keyed by `(perspective,
//!   king-bucket, mirror)` (§12.2).
//! - The FullThreats dirty-diff observer (≤80 add/remove threat events per
//!   move, the x-ray ripple) — the one genuinely heavy new machine.
//! - The CENN-v2 net loader (§12.6) that fills real weights from disk.

use crate::attacks::{attacks, king_attacks, knight_attacks, pawn_attacks, pseudo_attacks};
use crate::bitboard::Bitboard;
use crate::board::Board;
use crate::types::{Color, Piece, PieceType, Square};
use std::sync::LazyLock;

/// Accumulator width per perspective — must equal the SOTA head's `L1` (768) so
/// the refresh output feeds [`super::sota_head::SotaHead::forward`] directly.
pub const ACC: usize = super::sota_head::L1;
const _: () = assert!(
    ACC == super::sota_head::L1,
    "refresh output width must match the head"
);

// ===========================================================================
// HalfKAv2_hm — 22,528 = 64 · PS_NB / 2   (half_ka_v2_hm.{h,cpp})
// ===========================================================================

/// One 64-wide slot per piece "kind": 10 non-king (W/B × P,N,B,R,Q) + one
/// **shared** king slot. `PS_NB = 11 · 64`.
const PS_NB: usize = 704;
/// Shared king piece-slot (`10 · 64`) — both colors' kings land here.
const PS_KING: usize = 640;
/// HalfKAv2_hm feature-space dimension (`64 · PS_NB / 2`).
pub const HALFKA_DIMS: usize = 22528;

/// King-bucket number for each square (SF `KingBuckets / PS_NB`), indexed by the
/// oriented king square (`ksq ^ 56·persp`). 32 distinct buckets; the table is
/// horizontally symmetric (each row is a palindrome), which is why the index
/// does **not** need a horizontal mirror — only the vertical `^56` flip.
/// Reading order is square index 0 = a1 … 63 = h8.
#[rustfmt::skip]
const KING_BUCKETS: [u16; 64] = [
    28, 29, 30, 31, 31, 30, 29, 28,
    24, 25, 26, 27, 27, 26, 25, 24,
    20, 21, 22, 23, 23, 22, 21, 20,
    16, 17, 18, 19, 19, 18, 17, 16,
    12, 13, 14, 15, 15, 14, 13, 12,
     8,  9, 10, 11, 11, 10,  9,  8,
     4,  5,  6,  7,  7,  6,  5,  4,
     0,  1,  2,  3,  3,  2,  1,  0,
];

/// HalfKA square orientation: a **file mirror** so the king ends up on files
/// e-h. `7` (XOR flips the file) for a king on files a-d, else `0`. This is the
/// polarity that normalizes the king to the kingside; FullThreats uses the
/// *opposite* polarity (see [`orient_threats`]) — the documented fidelity trap.
#[inline]
fn orient_halfka(ksq: Square) -> u8 {
    if ksq.file() <= 3 { 7 } else { 0 }
}

/// Cache key for a HalfKAv2_hm king state: 32 bucket ids × two horizontal
/// orientations. Moving the king without changing this key leaves every
/// non-king feature's index stable, so a dirty-piece update is sufficient.
#[inline]
pub(crate) fn halfka_king_key(perspective: Color, king_sq: Square) -> usize {
    let flip = 56 * perspective as u8;
    let bucket = KING_BUCKETS[(king_sq.index() ^ flip) as usize] as usize;
    bucket * 2 + usize::from(orient_halfka(king_sq) != 0)
}

/// The piece-slot offset within a 704-block (SF `PieceSquareIndex`).
///
/// From `persp`'s point of view own pieces occupy the "white" even slots
/// (`0,128,…,512`) and enemy pieces the "black" odd slots (`64,192,…,576`);
/// kings share [`PS_KING`]. So a black-perspective refresh swaps which color's
/// pieces land in the white slots — the §3.4.2 symmetry that makes the two
/// accumulators exchange roles under a color flip.
#[inline]
fn piece_square_index(persp: Color, piece: Piece) -> usize {
    if matches!(piece.piece_type, PieceType::King) {
        return PS_KING;
    }
    // pt ∈ 0..=4 for P,N,B,R,Q → white slot = pt·128, black slot = pt·128 + 64.
    let pt = piece.piece_type as usize;
    if piece.color == persp {
        pt * 128
    } else {
        pt * 128 + 64
    }
}

/// The HalfKAv2_hm feature index of `piece` on `sq`, as seen from `perspective`
/// with the own king on `king_sq` (design/11 §12.1).
///
/// `index = (sq ^ orient(ksq) ^ 56·persp) + piece_square_index + king_bucket`,
/// exactly SF's `make_index`. The oriented square (0-63) plus the piece slot
/// (a multiple of 64, ≤640) stays inside one `PS_NB`-block; the king bucket adds
/// `bucket·704`. Range: `[0, 22528)`.
#[inline]
pub fn halfka_index(perspective: Color, piece: Piece, sq: Square, king_sq: Square) -> usize {
    halfka_row_base(perspective, piece, king_sq) + halfka_oriented(perspective, sq, king_sq)
}

/// The part of [`halfka_index`] that depends only on `(perspective, piece,
/// king_sq)` — i.e. everything a whole BITBOARD of one piece type shares.
///
/// Split out because the Finny set-diff refresh walks one piece type's added
/// and removed squares as two separate bitboard loops, and recomputed the
/// slot + bucket base inside each of them: nine instructions
/// (`shl`/`test`/`lea`/`cmove`/`cmp`/`mov`/`cmove`/`add`) per differing piece
/// type per direction, for a value that cannot change across either loop.
/// Hoisting it is a source-level change LLVM will not make for you, because
/// `piece_square_index`'s two `cmove`s make the expression look cheap enough
/// to sink.
#[inline]
pub fn halfka_row_base(perspective: Color, piece: Piece, king_sq: Square) -> usize {
    let flip = 56 * perspective as u8;
    let bucket = KING_BUCKETS[(king_sq.index() ^ flip) as usize] as usize;
    piece_square_index(perspective, piece) + bucket * PS_NB
}

/// The per-square half of [`halfka_index`]: `sq` mirrored into the king's
/// half of the board, `0..64`.
///
/// `& 63` is a no-op on the value (both operands are 6-bit) and a range fact
/// for the optimizer — the same move as [`ThreatResolver::index_decoded`].
#[inline]
pub fn halfka_oriented(perspective: Color, sq: Square, king_sq: Square) -> usize {
    let flip = 56 * perspective as u8;
    ((sq.index() ^ orient_halfka(king_sq) ^ flip) & 63) as usize
}

// ===========================================================================
// FullThreats — 60,720   (full_threats.{h,cpp})
// ===========================================================================

/// FullThreats feature-space dimension.
pub const THREAT_DIMS: usize = 60720;

/// A learned re-coding of the FullThreats index space (the Recode POC,
/// `docs/research/learned-feature-code-study-2026-07-30.md` §6.1 mode (ii)):
/// a surjection from the 60,720 geometric relation codes onto the net's actual
/// threat row space (`SotaNetwork::threat_rows` rows). Carried by the net
/// (CENN `TRM1` section), applied at the single resolver chokepoint
/// ([`ThreatResolver::index_decoded`]) that both the from-scratch and
/// incremental paths share — so the two cannot drift. `None` everywhere means
/// the identity code (every net before this change).
///
/// `u16` fits because `THREAT_DIMS < 65536` and the target row space is never
/// larger than the source. The loader validates every entry `< threat_rows`,
/// which is what lets the fold paths index the weight table without a per-event
/// bounds check.
pub type ThreatRemap = [u16; THREAT_DIMS];

/// SF piece code: white P..K = 1..6, black P..K = 9..14 (color = bit 3). The
/// gap layout is deliberate — XOR-8 flips color, which is the trick FullThreats
/// uses to swap perspective (`attacker ^ 8·persp`).
#[inline]
pub(crate) const fn sf_code(piece: Piece) -> usize {
    ((piece.color as usize) << 3) | (piece.piece_type as usize + 1)
}

/// AllPieces in SF's enumeration order — this order fixes the attacker-block
/// bases (`cumulativeOffset`), so it must not be reordered.
const ALL_PIECES: [usize; 12] = [1, 2, 3, 4, 5, 6, 9, 10, 11, 12, 13, 14];

/// `numValidTargets[code]` = 2 × (count of victim piece-types this attacker may
/// threaten). Pawn 6 (P/N/R), knight 10 (P/N/B/R/Q), bishop 8 (P/N/B/R),
/// rook 8, queen 10, king 0. Indexed by SF piece code (16-wide, gaps unused).
#[rustfmt::skip]
const NUM_VALID_TARGETS: [usize; 16] = [
    0, 6, 10, 8, 8, 10, 0, 0,
    0, 6, 10, 8, 8, 10, 0, 0,
];

/// `map[attacker_type-1][attacked_type-1]` (types PAWN..KING = 1..6): the dense
/// victim-relation ordinal for a legal attacker→victim type pair, or `-1` when
/// the pair is excluded. Kings attack nothing; pawns threaten only P/N/R;
/// bishops/rooks skip Q & K; knights/queens include Q.
#[rustfmt::skip]
const MAP: [[i8; 6]; 6] = [
    [ 0,  1, -1,  2, -1, -1], // pawn
    [ 0,  1,  2,  3,  4, -1], // knight
    [ 0,  1,  2,  3, -1, -1], // bishop
    [ 0,  1,  2,  3, -1, -1], // rook
    [ 0,  1,  2,  3,  4, -1], // queen
    [-1, -1, -1, -1, -1, -1], // king
];

/// Is the attacker→victim TYPE pair one `threat_index` keeps? Derived from
/// [`MAP`] — the single source of truth — so the observer's push-time filter
/// (perf study §6-T2: drop events at generation instead of buffering +
/// resolving them twice and dropping at fold) can never diverge from the
/// fold-time drop set. The pawn-PUSH asymmetry (push onto N/R would be kept
/// by MAP but is never a threat) is guarded at the recording site, not here.
#[inline]
pub(crate) fn threat_pair_kept(attacker: PieceType, victim: PieceType) -> bool {
    // Our PieceType is 0-based (Pawn=0..King=5); MAP is indexed the same way
    // (its doc's "1..6" refers to SF's piece codes, not the array index).
    MAP[attacker as usize][victim as usize] >= 0
}

/// [`threat_pair_kept`] as a per-attacker **bit set over victim types**, so the
/// incremental observer can turn "which victims count?" into a bitboard mask
/// instead of a per-square predicate call — the same shape
/// [`for_each_threat`]'s `pawn_targets` / `minor_slider_targets` /
/// `queen_targets` already use on the from-scratch side.
///
/// Built from [`MAP`] in a `const` block (Rust's `const` blocks allow `while`
/// but not `for`, hence the index loops), so it is materialised at compile time
/// and cannot drift from the predicate; `kept_victim_masks_match_predicate`
/// pins the two together for all 36 pairs.
pub(crate) const KEPT_VICTIMS: [u8; PieceType::COUNT] = {
    let mut out = [0u8; PieceType::COUNT];
    let mut att = 0;
    while att < PieceType::COUNT {
        let mut vic = 0;
        while vic < PieceType::COUNT {
            if MAP[att][vic] >= 0 {
                out[att] |= 1 << vic;
            }
            vic += 1;
        }
        att += 1;
    }
    out
};

/// FullThreats square orientation — the **opposite polarity** to HalfKA:
/// `0` for a king on files a-d, `7` for files e-h (design/11 §12.1, the
/// "a1/h1 polarity" fidelity trap: HalfKA's OrientTBL is a1→SQ_H1, threats' is
/// a1→SQ_A1).
#[inline]
fn orient_threats(ksq: Square) -> u8 {
    if ksq.file() <= 3 { 0 } else { 7 }
}

/// Empty-board attack geometry of an SF piece code on `from` — the same sets SF
/// builds its threat LUTs from. Pawns use *push-or-attacks* (single push |
/// both diagonal captures); everything else is the plain pseudo-attack set.
fn geom_attacks(code: usize, from: Square) -> Bitboard {
    let color = if code < 8 { Color::White } else { Color::Black };
    match code & 7 {
        1 => pawn_push_or_attacks(color, from),
        2 => knight_attacks(from),
        3 => pseudo_attacks(PieceType::Bishop, from),
        4 => pseudo_attacks(PieceType::Rook, from),
        5 => pseudo_attacks(PieceType::Queen, from),
        6 => king_attacks(from),
        _ => Bitboard::EMPTY,
    }
}

/// SF `PawnPushOrAttacks[c][from]` — the pawn's single push square unioned with
/// its two diagonal capture squares (all masked to the board). A pawn's "threat"
/// includes the pawn it is blocked by / about to push into, hence the push bit.
fn pawn_push_or_attacks(color: Color, from: Square) -> Bitboard {
    let mut bb = pawn_attacks(color, from);
    if let Some(push) = from.shifted(color.pawn_push()) {
        bb |= Bitboard::from_square(push);
    }
    bb
}

/// The three threat lookup tables, built once. Sizes: `lut2` 64 KiB, the rest a
/// few KiB — all indexed by SF piece code (16-wide, gap entries unused).
struct ThreatTables {
    /// `offsets[code][from]` = prefix-sum of empty-board attack counts over
    /// from-squares before `from` (the per-attacker from-square base).
    offsets: [[u32; 64]; 16],
    /// `piece_offset[code]` = total empty-board attack edges for the attacker
    /// (SF `cumulativePieceOffset`; the per-attacker geometry size). Folded into
    /// `lut1` for the index, but kept explicitly: the table-invariant test reads
    /// it, and the coming incremental threats observer (§12.2) needs it.
    #[allow(dead_code)]
    piece_offset: [u32; 16],
    /// `cumulative_offset[code]` = the attacker-block base (SF `cumulativeOffset`).
    /// Kept for the same reasons as `piece_offset`.
    #[allow(dead_code)]
    cumulative_offset: [u32; 16],
    /// `lut2[code][from][to]` = popcount of the attacker's attack bits strictly
    /// below `to` (the dense ordinal of `to` among `from`'s targets).
    lut2: Box<[[[u8; 64]; 64]; 16]>,
    /// `lut1[attacker][attacked][from<to]` = the block+sub-block feature base, or
    /// [`THREAT_DIMS`] (an out-of-range sentinel → dropped) for an excluded or
    /// semi-excluded pair. The `from<to` axis dedups symmetric same-type threats.
    lut1: [[[u32; 2]; 16]; 16],
}

fn build_threat_tables() -> ThreatTables {
    let mut offsets = [[0u32; 64]; 16];
    let mut piece_offset = [0u32; 16];
    let mut cumulative_offset = [0u32; 16];
    let mut lut2 = Box::new([[[0u8; 64]; 64]; 16]);

    // Per-attacker geometry: prefix-sum from-square bases + the `to`-ordinal LUT.
    let mut cumulative = 0u32;
    for &code in &ALL_PIECES {
        let is_pawn = code & 7 == 1;
        let mut cum_piece = 0u32;
        for from_i in 0..64u8 {
            offsets[code][from_i as usize] = cum_piece;
            let from = Square::new(from_i);
            let attacks = geom_attacks(code, from);
            // Fill the ordinal LUT for every from-square (SF builds it for all
            // 64, even rank 1/8 pawns that never occur in play).
            let bits = attacks.to_u64();
            for to_i in 0..64u8 {
                let below = bits & ((1u64 << to_i) - 1);
                lut2[code][from_i as usize][to_i as usize] = below.count_ones() as u8;
            }
            // Advance the prefix sum. Pawns only count from squares a2..h7
            // (a pawn never sits on rank 1 or 8), matching SF's guard.
            if !is_pawn || (8..=55).contains(&from_i) {
                cum_piece += attacks.count();
            }
        }
        piece_offset[code] = cum_piece;
        cumulative_offset[code] = cumulative;
        cumulative += NUM_VALID_TARGETS[code] as u32 * cum_piece;
    }
    debug_assert_eq!(
        cumulative, THREAT_DIMS as u32,
        "threat blocks must tile 60720"
    );

    // Per (attacker, attacked) feature base + the exclusion sentinels.
    let mut lut1 = [[[THREAT_DIMS as u32; 2]; 16]; 16];
    for &attacker in &ALL_PIECES {
        for &attacked in &ALL_PIECES {
            let same_type_diff_color = (attacker ^ attacked) == 8;
            let att_t = attacker & 7; // 1..=6
            let atd_t = attacked & 7; // 1..=6
            let map = MAP[att_t - 1][atd_t - 1];
            // Same-type threats are symmetric (A attacks B ⟺ B attacks A) for
            // every non-pawn and for opposite-color pawns, so one direction is
            // dropped to avoid double counting. Same-color pawn→pawn (a push)
            // is directional and kept.
            let semi_excluded = att_t == atd_t && (same_type_diff_color || att_t != 1);
            let excluded = map < 0;

            let feature = if excluded {
                THREAT_DIMS as u32
            } else {
                let atd_color = attacked >> 3; // 0 = white victim, 1 = black
                let num_types = NUM_VALID_TARGETS[attacker] / 2;
                let sub = atd_color * num_types + map as usize;
                cumulative_offset[attacker] + sub as u32 * piece_offset[attacker]
            };
            lut1[attacker][attacked][0] = if excluded {
                THREAT_DIMS as u32
            } else {
                feature
            };
            lut1[attacker][attacked][1] = if excluded || semi_excluded {
                THREAT_DIMS as u32
            } else {
                feature
            };
        }
    }

    ThreatTables {
        offsets,
        piece_offset,
        cumulative_offset,
        lut2,
        lut1,
    }
}

static THREAT_TABLES: LazyLock<ThreatTables> = LazyLock::new(build_threat_tables);

/// The FullThreats feature index of the directed relation "`attacker` on `from`
/// threatens `attacked` on `to`", as seen from `perspective` with own king on
/// `ksq` (design/11 §12.1). Returns `None` when the pair is excluded (kings, or
/// an illegal attacker→victim type) or semi-excluded (the dropped direction of a
/// symmetric same-type threat) — SF's `push_back_if_lt(index, Dimensions)`.
///
/// `index = lut1[att][atd][from<to] + offsets[att][from] + lut2[att][from][to]`,
/// all with the squares file/rank-oriented by `OrientTBL[ksq] ^ 56·persp` and
/// the piece codes color-swapped by `^ 8·persp`. Range when `Some`: `[0, 60720)`.
#[inline]
pub fn threat_index(
    perspective: Color,
    attacker: Piece,
    from: Square,
    to: Square,
    attacked: Piece,
    ksq: Square,
) -> Option<usize> {
    ThreatResolver::new(perspective, ksq).index(attacker, from, to, attacked)
}

/// The per-(perspective, king) state of [`threat_index`], hoisted (perf study
/// #4 item #3): the `THREAT_TABLES` `LazyLock` deref — round 2's single
/// hottest instruction — plus the orientation and color-swap derivation were
/// re-run per event; a resolve loop builds one of these per fold and pays
/// them once. [`index`](Self::index) is the same arithmetic as
/// [`threat_index`] (which now delegates here), so the two cannot drift.
pub(crate) struct ThreatResolver<'r> {
    tables: &'static ThreatTables,
    orientation: u8,
    swap: usize,
    /// The net's threat re-coding, or `None` for the identity code. `Option`
    /// rather than a static identity table so a remap-less net pays a
    /// predicted branch per event instead of a 121 KB table's worth of new
    /// data traffic (see [`ThreatRemap`]).
    remap: Option<&'r ThreatRemap>,
}

impl<'r> ThreatResolver<'r> {
    /// Identity-code resolver — the raw 60,720-space geometry, what the
    /// datagen exporters and [`threat_index`] speak.
    #[inline]
    pub(crate) fn new(perspective: Color, ksq: Square) -> Self {
        Self::with_remap(perspective, ksq, None)
    }

    /// Resolver with the net's threat re-coding applied ([`ThreatRemap`]).
    #[inline]
    pub(crate) fn with_remap(
        perspective: Color,
        ksq: Square,
        remap: Option<&'r ThreatRemap>,
    ) -> Self {
        let p = perspective as u8;
        ThreatResolver {
            tables: &THREAT_TABLES,
            orientation: orient_threats(ksq) ^ (56 * p),
            // 0 or 8: flips the piece color bit.
            swap: 8 * p as usize,
            remap,
        }
    }

    /// [`threat_index`] with the per-fold state precomputed.
    #[inline]
    pub(crate) fn index(
        &self,
        attacker: Piece,
        from: Square,
        to: Square,
        attacked: Piece,
    ) -> Option<usize> {
        self.index_decoded(sf_code(attacker), sf_code(attacked), from, to)
    }

    /// [`index`](Self::index) with the two SF piece codes already decoded.
    ///
    /// The decode is **perspective-independent** — the perspective enters only
    /// as the `^ swap` colour flip below — so a resolve that walks one event
    /// list for BOTH perspectives pays `sf_code` once instead of twice
    /// (perf study 9 item 1). Same arithmetic as [`index`](Self::index), which
    /// now delegates here, so the two cannot drift.
    #[inline]
    pub(crate) fn index_decoded(
        &self,
        attacker_code: usize,
        attacked_code: usize,
        from: Square,
        to: Square,
    ) -> Option<usize> {
        let t = self.tables;
        // `& 63` is a NO-OP on the value and a RANGE FACT for the optimizer.
        // Both operands are already 6-bit — `Square::index() < 64` by the
        // newtype's constructor assert, and `orientation` is
        // `orient_threats(ksq) ^ 56·p ∈ {0, 7, 56, 63}` — but LLVM only sees
        // `u8 ^ u8 → 0..=255`, so every `offsets[att][from_o]` and
        // `lut2[att][from_o][to_o]` index carried a `cmp $63` / `ja panic`
        // pair it could not hoist (the bound depends on the loaded event).
        // This is perf study 8 §4.1's `l1_dot` shape in the resolver: the
        // loop is ~40 instructions and runs once per event per perspective.
        let from_o = ((from.index() ^ self.orientation) & 63) as usize;
        let to_o = ((to.index() ^ self.orientation) & 63) as usize;
        let att = attacker_code ^ self.swap;
        let atd = attacked_code ^ self.swap;
        let dir = (from_o < to_o) as usize;

        let base = t.lut1[att][atd][dir];
        if base >= THREAT_DIMS as u32 {
            return None; // excluded / semi-excluded
        }
        let idx = (base + t.offsets[att][from_o] + t.lut2[att][from_o][to_o] as u32) as usize;
        // The Recode chokepoint: one table lookup re-codes the geometric index
        // into the net's row space. `.min(THREAT_DIMS - 1)` is a NO-OP on the
        // value (the tiling test pins `base + piece_offset <= THREAT_DIMS`, so
        // `idx < THREAT_DIMS` always) and a range fact for the optimizer — the
        // same move as the `& 63` above, discharging the bounds check against
        // the fixed-size [`ThreatRemap`] array without a panic edge.
        match self.remap {
            Some(remap) => Some(usize::from(remap[idx.min(THREAT_DIMS - 1)])),
            None => Some(idx),
        }
    }
}

/// Force the threat LUTs to build now (parity with `attacks::init`). Optional —
/// the first [`threat_index`] / refresh triggers the `LazyLock` anyway.
pub fn init() {
    LazyLock::force(&THREAT_TABLES);
}

// ===========================================================================
// Feature enumeration — the active features on a board, per perspective
// ===========================================================================

/// Call `f` with the HalfKA index of every piece on the board, from
/// `perspective`. Exactly one feature per piece (kings included), so the call
/// count equals the piece count.
pub fn for_each_halfka(board: &Board, perspective: Color, mut f: impl FnMut(usize)) {
    let ksq = board.king_square(perspective);
    for sq in board.all_occupancy() {
        let piece = board.piece_on(sq).expect("occupancy ⇒ a piece");
        f(halfka_index(perspective, piece, sq, ksq));
    }
}

/// The 74-byte **bitboard** snapshot of a position that the FullThreats feature
/// space reads: occupancy split by piece type and by colour, the all-occupancy
/// union, and both king squares. It carries exactly what [`for_each_threat_at`]
/// (the from-scratch enumeration) and [`super::sota_threats`]'s incremental
/// generator/resolver read — nothing else.
///
/// Two consumers, one definition:
///
/// * [`super::sota_threats::ThreatContext::apply_at`] captures it at push time,
///   so fold-time (re)generation runs against the post-move geometry even after
///   a sibling has destroyed the live [`Board`] in place; and
/// * the from-scratch refresh runs off the *same* struct, which is what lets a
///   king-cross rebuild be **deferred** to the ply that actually evaluates
///   (perf study 18 §3) instead of paid eagerly at push.
///
/// # Why bitboards and not the mailbox (perf-impl-1)
/// The original form snapshotted `board.mailbox` and asked "is the piece on
/// this square a rook or a queen?" one square at a time — up to eight identity
/// resolutions per touched square just to discover the zero-to-two sliders that
/// actually x-ray it, and the same again for knights and pawns. The
/// from-scratch enumeration never did that: it intersects with victim-type
/// bitboards. Carrying the type/colour boards instead lets the incremental
/// generator use exactly the same trick.
#[derive(Clone, Copy)]
pub(super) struct ThreatPosition {
    /// Occupancy per piece type, indexed by `PieceType as usize`.
    pub(super) by_type: [Bitboard; PieceType::COUNT],
    /// Occupancy per colour, indexed by `Color as usize`.
    pub(super) by_color: [Bitboard; Color::COUNT],
    /// The union — derived, but materialised because every geometry query in
    /// the generator takes it.
    pub(super) occ: Bitboard,
    pub(super) kings: [Square; 2],
}

impl ThreatPosition {
    /// # Precondition
    /// **Both** kings must be on the board. `Board` documents king-less
    /// positions as a supported class and `Board::king_square` is a
    /// `debug_assert` with a release fallback, so a one-king board that used to
    /// work in a debug build will now trip here: the three `&Board` entry
    /// points ([`for_each_threat`], [`refresh_threat_into`] and
    /// `sota_threats::refresh_psqt_threat`) used to read only
    /// `king_square(perspective)` and now route through this (D33 reviewer A,
    /// A6). Release is unaffected -- only `kings[perspective]` is ever read.
    /// The search path already carried this precondition through
    /// `ThreatContext::apply_at`; the `pub` datagen callers in
    /// `tools/nnue/bullet/` are the ones this note is for.
    #[inline]
    pub(super) fn from_board(board: &Board) -> Self {
        let mut by_type = [Bitboard::EMPTY; PieceType::COUNT];
        for pt in PieceType::ALL {
            by_type[pt as usize] = board.pieces_of_type(pt);
        }
        ThreatPosition {
            by_type,
            by_color: [board.occupancy(Color::White), board.occupancy(Color::Black)],
            occ: board.all_occupancy(),
            kings: [
                board.king_square(Color::White),
                board.king_square(Color::Black),
            ],
        }
    }

    #[inline]
    pub(super) fn typed(&self, pt: PieceType) -> Bitboard {
        self.by_type[pt as usize]
    }

    #[inline]
    pub(super) fn colored(&self, c: Color) -> Bitboard {
        self.by_color[c as usize]
    }

    /// The pieces of one colour and type — the snapshot's `Board::pieces`.
    #[inline]
    pub(super) fn pieces(&self, c: Color, pt: PieceType) -> Bitboard {
        self.typed(pt) & self.colored(c)
    }

    /// Rook-movers (rooks + queens) and bishop-movers (bishops + queens) — the
    /// two pre-masks the x-ray scan intersects its ray fans with.
    #[inline]
    pub(super) fn rook_movers(&self) -> Bitboard {
        self.typed(PieceType::Rook) | self.typed(PieceType::Queen)
    }

    #[inline]
    pub(super) fn bishop_movers(&self) -> Bitboard {
        self.typed(PieceType::Bishop) | self.typed(PieceType::Queen)
    }

    /// Flip `piece`'s presence at `sq`. XOR is its own inverse, so this one
    /// primitive serves both passes of the generic replay: "undo every edit to
    /// recover the pre-move position", then "redo them one at a time".
    #[inline]
    pub(super) fn toggle(&mut self, piece: Piece, sq: Square) {
        let b = Bitboard::from_square(sq);
        self.by_type[piece.piece_type as usize] ^= b;
        self.by_color[piece.color as usize] ^= b;
        self.occ ^= b;
    }

    /// The colour of the piece on `sq`; `sq` must be occupied.
    #[inline]
    pub(super) fn color_on(&self, sq: Square) -> Color {
        // Same tripwire as `piece_on`'s. This used to be reachable only THROUGH
        // `piece_on`, which asserts first; the pawn-push branch of
        // `for_each_threat_at` now calls it directly, guarded only by a pawn
        // mask (D33 reviewer A, A8). An empty square silently reads as Black.
        debug_assert!(self.occ.contains(sq), "color_on an empty square");
        if self.colored(Color::White).contains(sq) {
            Color::White
        } else {
            Color::Black
        }
    }

    /// The whole piece on `sq`; `sq` must be occupied. Five type tests, versus
    /// the mailbox form's one byte load — which is why both the generator and
    /// [`for_each_threat_at`] are structured so this is reached only where the
    /// victim's type is *not* already a loop constant.
    #[inline]
    pub(super) fn piece_on(&self, sq: Square) -> Piece {
        debug_assert!(self.occ.contains(sq), "piece_on an empty square");
        let b = Bitboard::from_square(sq);
        // King is the fallthrough: if none of the other five boards has the
        // square, an occupied square must hold a king. No panic path.
        let mut pt = PieceType::King;
        for cand in [
            PieceType::Pawn,
            PieceType::Knight,
            PieceType::Bishop,
            PieceType::Rook,
            PieceType::Queen,
        ] {
            if !(self.by_type[cand as usize] & b).is_empty() {
                pt = cand;
                break;
            }
        }
        Piece::new(self.color_on(sq), pt)
    }
}

impl Default for ThreatPosition {
    fn default() -> Self {
        ThreatPosition {
            by_type: [Bitboard::EMPTY; PieceType::COUNT],
            by_color: [Bitboard::EMPTY; Color::COUNT],
            occ: Bitboard::EMPTY,
            kings: [Square::A1; 2],
        }
    }
}

/// Call `f` with the FullThreats index of every active directed threat on the
/// board, from `perspective`. Mirrors SF `append_active_indices`: every piece of
/// both colors is an attacker; victim sets are pre-filtered to the legal types;
/// excluded / semi-excluded relations are skipped ([`threat_index`] returns
/// `None`).
pub fn for_each_threat(board: &Board, perspective: Color, f: impl FnMut(usize)) {
    for_each_threat_at(&ThreatPosition::from_board(board), perspective, None, f);
}

/// [`for_each_threat`] over a [`ThreatPosition`] snapshot rather than a live
/// [`Board`] — the form the deferred king-cross rebuild needs, since the board
/// that produced the crossing ply is long gone by the time it is evaluated.
/// `remap` is the net's threat re-coding (`None` = identity; [`ThreatRemap`]) —
/// the engine's accumulator/psqt refresh paths pass the net's table so the
/// from-scratch build speaks the same row space as the incremental fold.
pub(super) fn for_each_threat_at(
    pos: &ThreatPosition,
    perspective: Color,
    remap: Option<&ThreatRemap>,
    mut f: impl FnMut(usize),
) {
    let ksq = pos.kings[perspective as usize];
    let occ = pos.occ;
    let resolver = ThreatResolver::with_remap(perspective, ksq, remap);

    let bb = |c: Color, pt: PieceType| pos.pieces(c, pt);
    let both = |pt: PieceType| pos.typed(pt);
    let all_pawns = both(PieceType::Pawn);
    // Victim masks (both colors — a threat can land on a friendly piece too).
    let pawn_targets = both(PieceType::Pawn) | both(PieceType::Knight) | both(PieceType::Rook);
    let minor_slider_targets = pawn_targets | both(PieceType::Bishop);
    let queen_targets = minor_slider_targets | both(PieceType::Queen);

    let mut emit = |attacker: Piece, from: Square, to: Square, attacked: Piece| {
        if let Some(idx) = resolver.index(attacker, from, to, attacked) {
            f(idx);
        }
    };

    for c in Color::ALL {
        // --- Pawns: two diagonal captures + a straight push-threat ---------
        let attacker = Piece::new(c, PieceType::Pawn);
        for from in bb(c, PieceType::Pawn) {
            for to in pawn_attacks(c, from) & pawn_targets {
                emit(attacker, from, to, pos.piece_on(to));
            }
            // Push threat: a pawn directly in front (own or enemy) is threatened.
            if let Some(to) = from.shifted(c.pawn_push())
                && all_pawns.contains(to)
            {
                emit(
                    attacker,
                    from,
                    to,
                    Piece::new(pos.color_on(to), PieceType::Pawn),
                );
            }
        }

        // --- Knight, bishop, rook, queen -----------------------------------
        for pt in [
            PieceType::Knight,
            PieceType::Bishop,
            PieceType::Rook,
            PieceType::Queen,
        ] {
            let attacker = Piece::new(c, pt);
            let targets = if matches!(pt, PieceType::Knight | PieceType::Queen) {
                queen_targets
            } else {
                minor_slider_targets
            };
            for from in bb(c, pt) {
                for to in attacks(pt, from, occ) & targets {
                    emit(attacker, from, to, pos.piece_on(to));
                }
            }
        }
        // Kings are never attackers (whole row of `map` is -1), so they are
        // deliberately not looped.
    }
}

// ===========================================================================
// From-scratch accumulator refresh (format-agnostic weight slices)
// ===========================================================================

/// Add the `ACC`-wide column at feature `index` of a feature-major weight slice
/// into `acc`, using wrapping i16 math (the quantized accumulator wraps, exactly
/// like [`super::sota_head`]'s FT sum). `W` is `i16` for HalfKA, `i8` for threats.
#[inline]
fn add_column<W: Copy + Into<i16>>(acc: &mut [i16; ACC], weights: &[W], index: usize) {
    let col = &weights[index * ACC..index * ACC + ACC];
    for (a, &w) in acc.iter_mut().zip(col.iter()) {
        *a = a.wrapping_add(w.into());
    }
}

/// Build the HalfKA accumulator for `perspective` from scratch: seed with the FT
/// bias, then add the weight column of every active HalfKA feature (§12.2).
/// `halfka_weights` is feature-major (`[HALFKA_DIMS · ACC]` i16), `ft_bias` is
/// `ACC`-wide.
pub fn refresh_pst(
    halfka_weights: &[i8],
    ft_bias: &[i16],
    board: &Board,
    perspective: Color,
) -> [i16; ACC] {
    assert_eq!(
        halfka_weights.len(),
        HALFKA_DIMS * ACC,
        "HalfKA transformer has the wrong size"
    );
    assert_eq!(ft_bias.len(), ACC, "SOTA FT bias has the wrong size");
    let mut acc = [0i16; ACC];
    acc.copy_from_slice(&ft_bias[..ACC]);
    for_each_halfka(board, perspective, |idx| {
        add_column(&mut acc, halfka_weights, idx)
    });
    acc
}

/// Build the threats accumulator for `perspective` from scratch: seed with zero,
/// then add the weight column of every active threat feature (§12.2).
/// `threat_weights` is feature-major (`[THREAT_DIMS · ACC]` i8).
///
/// **Test-only.** Every production caller uses [`refresh_threat_into`]: this
/// wrapper zeroes `acc` and then `refresh_threat_into` zeroes it again, which
/// is 2 x 2560 bytes of `memset` per call (D33 reviewer C, P4). It survives
/// because the assertions in this module and in `sota_threats` read more
/// clearly against a returned value.
#[cfg(test)]
pub fn refresh_threat(threat_weights: &[i8], board: &Board, perspective: Color) -> [i16; ACC] {
    let mut acc = [0i16; ACC];
    refresh_threat_into(&mut acc, threat_weights, board, perspective, None);
    acc
}

/// [`refresh_threat`] writing through a caller-owned destination.
///
/// The by-value form returns 2560 bytes, so `entry.acc[p] = refresh_threat(..)`
/// builds the accumulator in a stack local and then copies it into the entry —
/// a libc `memcpy` the `LD_PRELOAD` census counted at **0.0357 calls/node,
/// 91 bytes/node**, the largest surviving copy in the engine after perf study
/// 14 (perf study 15 §8 item 3, `docs/research/perf-artifacts-16/memspy-baseline.txt`).
/// Writing in place is what [`super::sota_accumulator::HalfkaContext`]'s
/// `finny_refresh_into` already does for the HalfKA twin.
///
/// `dst` is fully overwritten (zeroed first, then accumulated into), so a
/// caller may pass a stale buffer.
pub fn refresh_threat_into(
    dst: &mut [i16; ACC],
    threat_weights: &[i8],
    board: &Board,
    perspective: Color,
    remap: Option<&ThreatRemap>,
) {
    refresh_threat_into_at(
        dst,
        threat_weights,
        &ThreatPosition::from_board(board),
        perspective,
        remap,
    );
}

/// [`refresh_threat_into`] over a [`ThreatPosition`] snapshot — what the
/// deferred king-cross rebuild calls (perf study 18 §3), since by the time the
/// crossing ply is evaluated the live board has moved on.
pub(super) fn refresh_threat_into_at(
    dst: &mut [i16; ACC],
    threat_weights: &[i8],
    pos: &ThreatPosition,
    perspective: Color,
    remap: Option<&ThreatRemap>,
) {
    dst.fill(0);
    if threat_weights.is_empty() {
        return;
    }
    assert_eq!(
        threat_weights.len(),
        THREAT_DIMS * ACC,
        "FullThreats transformer has the wrong size"
    );
    // Perf study 39: count the COLUMNS this from-scratch rebuild folds. Fusion
    // makes the threat channel inherit HalfKA's refresh trigger, so this row
    // count times the extra refresh rate IS fusion's cost side — quoted by
    // studies 29 and 31 from an assumption, measured here.
    #[cfg(feature = "foldcount")]
    {
        let mut rows = 0usize;
        for_each_threat_at(pos, perspective, |idx| {
            rows += 1;
            add_column(dst, threat_weights, idx);
        });
        super::foldcount::THR_REFRESH.hit(rows);
    }
    #[cfg(not(feature = "foldcount"))]
    for_each_threat_at(pos, perspective, |idx| add_column(dst, threat_weights, idx));
    match remap {
        // Identity code: the weight table spans the full geometric space.
        None => assert_eq!(
            threat_weights.len(),
            THREAT_DIMS * ACC,
            "FullThreats transformer has the wrong size"
        ),
        // Re-coded net: the table spans `threat_rows` rows; the loader has
        // validated every remap entry against that row count, so whole-rows
        // is the only shape left to check here.
        Some(_) => assert_eq!(
            threat_weights.len() % ACC,
            0,
            "FullThreats transformer is not whole rows"
        ),
    }
    for_each_threat_at(pos, perspective, remap, |idx| {
        add_column(dst, threat_weights, idx)
    });
}

/// Build both accumulators for both perspectives — the `(pst, threat)` pair
/// [`super::sota_head::SotaHead::forward`] wants (index 0 = White, 1 = Black).
pub fn refresh_both(
    halfka_weights: &[i8],
    ft_bias: &[i16],
    threat_weights: &[i8],
    board: &Board,
    threat_remap: Option<&ThreatRemap>,
) -> ([[i16; ACC]; 2], [[i16; ACC]; 2]) {
    let mut pst = [[0i16; ACC]; 2];
    let mut threat = [[0i16; ACC]; 2];
    // One snapshot for both perspectives, the same hoist `ThreatContext::refresh`
    // makes (D33 reviewer A, A9). Cold path -- the root refresh and the
    // debug-assertion oracle -- so this is tidiness, not speed.
    let pos = ThreatPosition::from_board(board);
    for (i, &persp) in Color::ALL.iter().enumerate() {
        pst[i] = refresh_pst(halfka_weights, ft_bias, board, persp);
        refresh_threat_into_at(&mut threat[i], threat_weights, &pos, persp, threat_remap);
    }
    (pst, threat)
}

#[cfg(test)]
mod tests {
    use super::*;
    use std::collections::HashSet;

    fn wp() -> Piece {
        Piece::new(Color::White, PieceType::Pawn)
    }
    fn wk() -> Piece {
        Piece::new(Color::White, PieceType::King)
    }
    fn wn() -> Piece {
        Piece::new(Color::White, PieceType::Knight)
    }

    // -------- HalfKAv2_hm --------------------------------------------------

    /// The index reproduces SF `make_index` hand-computed from the formula for
    /// several (piece, sq, ksq, perspective) tuples.
    #[test]
    fn halfka_hand_computed() {
        // White king on e1 (file e ⇒ orient 0), the king itself as a feature.
        // (4^0^0) + PS_KING(640) + KingBuckets[4](=31)·704 = 4 + 640 + 21824.
        assert_eq!(
            halfka_index(Color::White, wk(), Square::E1, Square::E1),
            4 + 640 + 31 * 704
        );
        // White pawn on a1, white king on a1 (file a ⇒ orient 7).
        // (0^7^0) + 0 + KingBuckets[0](=28)·704 = 7 + 0 + 19712 = 19719.
        assert_eq!(
            halfka_index(Color::White, wp(), Square::A1, Square::A1),
            19719
        );
        // Black perspective: WHITE pawn on a2, BLACK king on e8.
        // orient(e8)=0, flip=56 ⇒ sq 8^56=48; enemy pawn ⇒ PS_B_PAWN(64);
        // KingBuckets[60^56 = 4](=31)·704. = 48 + 64 + 21824 = 21936.
        assert_eq!(
            halfka_index(Color::Black, wp(), Square::A2, Square::E8),
            21936
        );
    }

    /// Horizontal mirror symmetry: a queenside king setup and its kingside
    /// mirror map to the *same* index (the point of the file-mirror orient +
    /// palindromic KingBuckets).
    #[test]
    fn halfka_mirror_symmetry() {
        // King b1 + knight c1  vs  mirror King g1 + knight f1.
        let qs = halfka_index(Color::White, wn(), Square::C1, Square::B1);
        let ks = halfka_index(Color::White, wn(), Square::F1, Square::G1);
        assert_eq!(qs, ks, "queenside and kingside mirror to one index");
        // Sanity: not a degenerate all-equal function.
        assert_ne!(
            halfka_index(Color::White, wn(), Square::C1, Square::B1),
            halfka_index(Color::White, wn(), Square::D1, Square::B1),
        );
    }

    /// Every (perspective, piece, sq, ksq) yields an in-range index.
    #[test]
    fn halfka_indices_in_range() {
        for persp in Color::ALL {
            for color in Color::ALL {
                for pt in PieceType::ALL {
                    let piece = Piece::new(color, pt);
                    for sq in Square::all() {
                        for ksq in Square::all() {
                            assert!(halfka_index(persp, piece, sq, ksq) < HALFKA_DIMS);
                        }
                    }
                }
            }
        }
    }

    // -------- FullThreats --------------------------------------------------

    /// The empty-board geometry sizes match SF's documented per-attacker edge
    /// counts (P 132, N 336, B 560, R 896, Q 1456, K 420) and the blocks tile
    /// exactly 60,720.
    #[test]
    fn threats_geometry_and_tiling() {
        let t = &*THREAT_TABLES;
        // code → expected piece_offset, for both colors.
        for base in [0usize, 8] {
            assert_eq!(t.piece_offset[base + 1], 132, "pawn edges");
            assert_eq!(t.piece_offset[base + 2], 336, "knight edges");
            assert_eq!(t.piece_offset[base + 3], 560, "bishop edges");
            assert_eq!(t.piece_offset[base + 4], 896, "rook edges");
            assert_eq!(t.piece_offset[base + 5], 1456, "queen edges");
            assert_eq!(t.piece_offset[base + 6], 420, "king edges");
        }
        // Telescoping: last block base + its size == Dimensions.
        let last = 14; // B_QUEEN is the last with nonzero size; B_KING adds 0.
        let end = t.cumulative_offset[last] + NUM_VALID_TARGETS[last] as u32 * t.piece_offset[last];
        assert_eq!(end, THREAT_DIMS as u32);
        // No non-sentinel lut1 base can push an index out of range.
        for &att in &ALL_PIECES {
            for &atd in &ALL_PIECES {
                for &base in &t.lut1[att][atd] {
                    if base < THREAT_DIMS as u32 {
                        assert!(base + t.piece_offset[att] <= THREAT_DIMS as u32);
                    }
                }
            }
        }
    }

    /// A hand-computed threat index for one simple geometry:
    /// White knight b1 → black pawn c3, white king a1 (orient 0), white view.
    /// lut1[Nw][Pb][from<to] = 792 + (1·5 + 0)·336 = 2472;
    /// offsets[Nw][b1] = 2 (knight a1 has 2 edges); lut2[Nw][b1][c3] = 2.
    #[test]
    fn threats_hand_computed() {
        let idx = threat_index(
            Color::White,
            Piece::new(Color::White, PieceType::Knight),
            Square::B1,
            Square::C3,
            Piece::new(Color::Black, PieceType::Pawn),
            Square::A1,
        );
        assert_eq!(idx, Some(2472 + 2 + 2));
    }

    /// The map exclusions and the semi-exclusion dedup hold.
    #[test]
    fn threats_exclusions() {
        let ksq = Square::A1; // orient 0, no flip
        // King is never an attacker.
        assert_eq!(
            threat_index(
                Color::White,
                Piece::new(Color::White, PieceType::King),
                Square::E4,
                Square::E5,
                Piece::new(Color::Black, PieceType::Pawn),
                ksq,
            ),
            None
        );
        // Pawn → bishop is excluded (pawns only threaten P/N/R).
        assert_eq!(
            threat_index(
                Color::White,
                Piece::new(Color::White, PieceType::Pawn),
                Square::B2,
                Square::C3,
                Piece::new(Color::Black, PieceType::Bishop),
                ksq,
            ),
            None
        );
        // Same-type rook→rook: the `from < to` direction is dropped, the other
        // kept (symmetric-threat dedup).
        let rw = Piece::new(Color::White, PieceType::Rook);
        let rb = Piece::new(Color::Black, PieceType::Rook);
        assert_eq!(
            threat_index(Color::White, rw, Square::A1, Square::A8, rb, ksq),
            None,
            "from<to direction of a symmetric threat is semi-excluded"
        );
        let kept = threat_index(Color::White, rw, Square::A8, Square::A1, rb, ksq);
        assert!(
            kept.is_some() && kept.unwrap() < THREAT_DIMS,
            "other direction kept"
        );
    }

    /// On real positions, every active threat index is in range and — crucially
    /// — the active indices are all distinct (the index math is collision-free).
    #[test]
    fn threats_active_in_range_and_distinct() {
        let fens = [
            crate::fen::START_FEN,
            "r1bqkbnr/pppp1ppp/2n5/4p3/2B1P3/5N2/PPPP1PPP/RNBQK2R w KQkq - 0 1",
            "8/2p5/3p4/KP5r/1R3p1k/8/4P1P1/8 w - - 0 1",
        ];
        for fen in fens {
            let board = Board::from_fen(fen).unwrap();
            for persp in Color::ALL {
                let mut seen = HashSet::new();
                for_each_threat(&board, persp, |idx| {
                    assert!(idx < THREAT_DIMS, "{fen}: threat idx {idx} out of range");
                    assert!(seen.insert(idx), "{fen}: duplicate active threat idx {idx}");
                });
            }
        }
    }

    // -------- Refresh ------------------------------------------------------

    /// A tiny synthetic HalfKA weight set: all-ones columns, so an accumulator
    /// neuron equals `bias + (#active features)` — and #active == piece count.
    fn ones_halfka() -> (Vec<i8>, Vec<i16>) {
        let weights = vec![1i8; HALFKA_DIMS * ACC];
        let bias: Vec<i16> = (0..ACC).map(|j| (j % 5) as i16).collect();
        (weights, bias)
    }

    #[test]
    fn refresh_pst_counts_and_seeds_bias() {
        let board = Board::from_fen(crate::fen::START_FEN).unwrap();
        let (w, bias) = ones_halfka();
        let n = board.all_occupancy().count() as i16;
        for persp in Color::ALL {
            let acc = refresh_pst(&w, &bias, &board, persp);
            for (j, &v) in acc.iter().enumerate() {
                assert_eq!(v, bias[j] + n, "neuron {j}: bias + active-count");
            }
        }
        assert_eq!(n, 32, "startpos has 32 pieces");
    }

    /// Stride check: with all weights zero except the white king's own feature
    /// column set to a ramp `col[j] = j`, the white-perspective accumulator must
    /// be exactly `bias[j] + j` — proving the `index·ACC + j` layout, that the
    /// king is an active feature, and bias seeding, all at once. A transposed
    /// index would read a zero column and fail.
    #[test]
    fn refresh_pst_stride_and_king_feature() {
        let board = Board::from_fen(crate::fen::START_FEN).unwrap();
        let mut w = vec![0i8; HALFKA_DIMS * ACC];
        let bias: Vec<i16> = (0..ACC).map(|j| (j % 7) as i16).collect();

        let ksq = board.king_square(Color::White);
        let king_feat = halfka_index(Color::White, wk(), ksq, ksq);
        // A per-lane ramp that still fits i8 — the point of the test is that
        // lane `j` of the king row lands in lane `j` of the accumulator (the
        // row STRIDE), and `j % 127` is as distinguishing as `j` was while
        // staying inside the i8 table the transformer now holds.
        for j in 0..ACC {
            w[king_feat * ACC + j] = (j % 127) as i8;
        }
        let acc = refresh_pst(&w, &bias, &board, Color::White);
        for (j, &v) in acc.iter().enumerate() {
            assert_eq!(v, bias[j] + (j % 127) as i16, "neuron {j}");
        }
    }

    #[test]
    fn refresh_threat_counts() {
        let board = Board::from_fen(crate::fen::START_FEN).unwrap();
        let w = vec![1i8; THREAT_DIMS * ACC];
        for persp in Color::ALL {
            // Count active threats independently.
            let mut n = 0i16;
            for_each_threat(&board, persp, |_| n += 1);
            let acc = refresh_threat(&w, &board, persp);
            for &v in acc.iter() {
                assert_eq!(v, n, "every neuron == active threat count");
            }
            assert!(n > 0, "startpos has threats");
        }
    }

    /// Refresh is deterministic and both perspectives run without panic; the
    /// `refresh_both` pair agrees with the single-perspective builders.
    #[test]
    fn refresh_deterministic_and_consistent() {
        let board =
            Board::from_fen("r1bqkbnr/pppp1ppp/2n5/4p3/2B1P3/5N2/PPPP1PPP/RNBQK2R w KQkq - 0 1")
                .unwrap();
        let (hw, bias) = ones_halfka();
        let tw = vec![2i8; THREAT_DIMS * ACC];

        let a = refresh_both(&hw, &bias, &tw, &board, None);
        let b = refresh_both(&hw, &bias, &tw, &board, None);
        assert_eq!(a, b, "deterministic");

        let (pst, threat) = &a;
        assert_eq!(pst[0], refresh_pst(&hw, &bias, &board, Color::White));
        assert_eq!(pst[1], refresh_pst(&hw, &bias, &board, Color::Black));
        assert_eq!(threat[0], refresh_threat(&tw, &board, Color::White));
        assert_eq!(threat[1], refresh_threat(&tw, &board, Color::Black));
    }

    /// End-to-end wiring: the refresh output feeds `SotaHead::forward` (types
    /// line up, runs, finite result). Trivial synthetic head — not a strength net.
    #[test]
    fn refresh_feeds_sota_head() {
        use super::super::sota_head::{BUCKETS, L1, L2, L3, SotaHead};
        let board = Board::from_fen(crate::fen::START_FEN).unwrap();
        let (hw, bias) = ones_halfka();
        let tw = vec![0i8; THREAT_DIMS * ACC];
        let (pst, threat) = refresh_both(&hw, &bias, &tw, &board, None);

        let head = SotaHead {
            l1_weights: vec![1i8; BUCKETS * L2 * L1].into(),
            l1_biases: vec![0.0f32; BUCKETS * L2].into(),
            l2_weights: vec![0.01f32; BUCKETS * L2 * L3].into(),
            l2_biases: vec![0.0f32; BUCKETS * L3].into(),
            l3_weights: vec![0.02f32; BUCKETS * L3].into(),
            l3_biases: vec![0.0f32; BUCKETS].into_boxed_slice(),
            network_scale: 380,
            threat_shift: 2,
            l3_var_weights: None,
            l3_var_biases: None,
            l3_wdl_weights: None,
            l3_wdl_biases: None,
            skip_weights: None,
            ft_scale: crate::eval::nnue::sota_head::FtScale::from_flat_iter(std::iter::repeat_n(
                1i16, ACC,
            )),
        };
        let piece_count = board.all_occupancy().count() as usize;
        let (cp, _sigma, _wdl) = head.forward(
            &pst,
            &threat,
            Color::White,
            piece_count,
            &super::super::sota_head::ZERO_PSQT,
        );
        assert!(cp.abs() < 1_000_000, "finite centipawns");
    }

    /// The incremental observer filters victims with [`KEPT_VICTIMS`] where the
    /// fold's `DeltaBuf::push` filters them with [`threat_pair_kept`]. If those
    /// two ever disagreed, the incremental accumulator would silently keep (or
    /// drop) a relation the from-scratch refresh does not — the exact class of
    /// bug the replay oracle exists to catch, caught here at compile-time cost
    /// instead. Both derive from `MAP`; this pins that they still do, for all
    /// 36 ordered type pairs.
    #[test]
    fn kept_victim_masks_match_predicate() {
        for attacker in PieceType::ALL {
            for victim in PieceType::ALL {
                let by_mask = KEPT_VICTIMS[attacker as usize] & (1 << victim as u8) != 0;
                assert_eq!(
                    by_mask,
                    threat_pair_kept(attacker, victim),
                    "KEPT_VICTIMS disagrees with threat_pair_kept for {attacker:?} -> {victim:?}"
                );
            }
        }
    }

    /// `KEPT_VICTIMS` is indexed by `PieceType as usize`, and so is `MAP`; the
    /// observer's per-type loops walk `PieceType::ALL`. All three only line up
    /// if `ALL` is in discriminant order.
    #[test]
    fn piece_type_all_is_in_discriminant_order() {
        for (i, pt) in PieceType::ALL.iter().enumerate() {
            assert_eq!(i, *pt as usize, "PieceType::ALL[{i}] is {pt:?}");
        }
    }
}
