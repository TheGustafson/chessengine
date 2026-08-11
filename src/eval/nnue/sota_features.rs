//! The SOTA net's **SF-faithful feature front-end** (design/11 §12.1-12.2, D40):
//! [`halfka_index`] (the 22,528-dim HalfKAv2_hm index), [`threat_index`] (the
//! 60,720-dim FullThreats index), and [`refresh_pst`] / [`refresh_threat`] /
//! [`refresh_both`] — from-scratch accumulator builds over weight slices.
//!
//! A **reimplementation of the index math** in Stockfish's `half_ka_v2_hm` and
//! `full_threats`: the arithmetic and the constant tables are transcribed, but
//! no GPLv3 code is copied (D27). Correctness is pinned by the tests at the
//! bottom, which hand-compute the SF `make_index` for known tuples.
//!
//! Behind the `sota` cargo feature; the default build never compiles it.

use crate::attacks::{attacks, king_attacks, knight_attacks, pawn_attacks, pseudo_attacks};
use crate::bitboard::Bitboard;
use crate::board::Board;
use crate::types::{Color, Piece, PieceType, Square};
use std::sync::LazyLock;

/// Accumulator width per perspective — must equal the SOTA head's `L1`, so the
/// refresh output feeds [`super::sota_head::SotaHead::forward`] directly.
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
/// oriented king square (`ksq ^ 56·persp`), square index 0 = a1 … 63 = h8.
/// 32 buckets; each row is a palindrome, which is why the index needs no
/// horizontal mirror — only the vertical `^56` flip.
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
/// e-h — `7` for a king on files a-d, else `0`. FullThreats uses the *opposite*
/// polarity ([`orient_threats`]); that asymmetry is a known fidelity trap.
#[inline]
fn orient_halfka(ksq: Square) -> u8 {
    if ksq.file() <= 3 { 7 } else { 0 }
}

/// Cache key for a HalfKAv2_hm king state: 32 bucket ids × two horizontal
/// orientations. Moving the king without changing this key leaves every
/// non-king index stable, so a dirty-piece update suffices.
#[inline]
pub(crate) fn halfka_king_key(perspective: Color, king_sq: Square) -> usize {
    let flip = 56 * perspective as u8;
    let bucket = KING_BUCKETS[(king_sq.index() ^ flip) as usize] as usize;
    bucket * 2 + usize::from(orient_halfka(king_sq) != 0)
}

/// The piece-slot offset within a 704-block (SF `PieceSquareIndex`). From
/// `persp`'s view own pieces occupy the "white" even slots (`0,128,…,512`) and
/// enemy pieces the "black" odd slots (`64,192,…,576`); kings share
/// [`PS_KING`]. So a black-perspective refresh swaps which colour's pieces land
/// in the white slots — the symmetry that makes the two accumulators exchange
/// roles under a colour flip.
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

/// A feature's **element offset into a feature-transformer weight table** —
/// `feature * ACC` — as consumed by every `apply_*_deltas*` fold kernel. Scaled
/// ONCE at the producer, because the kernels walk their row list once per TILE
/// (8 tiles at `ACC = 1280`, `REGS = 5`) and would otherwise rescale per tile.
///
/// **The newtype is the safety argument**: producers and kernels must agree on
/// the unit, and a silent disagreement is an arbitrarily wrong evaluation that
/// still benches. PSQ1 indexing keeps the plain feature index — its rows are
/// `BUCKETS` wide, not `ACC` — so the two units coexist and must not be mixed.
#[derive(Clone, Copy, PartialEq, Eq, Debug, Default)]
pub struct RowOffset(pub usize);

impl RowOffset {
    /// Scale a feature index into its weight-table element offset.
    #[inline(always)]
    pub const fn of(feature: usize) -> Self {
        Self(feature * ACC)
    }

    /// Recover the feature index — instrumentation and tests only.
    #[inline]
    pub const fn feature(self) -> usize {
        self.0 / ACC
    }
}

/// The HalfKAv2_hm feature index of `piece` on `sq` as seen from `perspective`
/// with the own king on `king_sq`:
/// `index = (sq ^ orient(ksq) ^ 56·persp) + piece_square_index + king_bucket`,
/// exactly SF's `make_index`. The oriented square (0-63) plus the piece slot (a
/// multiple of 64, ≤640) stays inside one `PS_NB`-block; the king bucket adds
/// `bucket·704`. Range: `[0, 22528)`.
#[inline]
pub fn halfka_index(perspective: Color, piece: Piece, sq: Square, king_sq: Square) -> usize {
    halfka_row_base(perspective, piece, king_sq) + halfka_oriented(perspective, sq, king_sq)
}

/// The part of [`halfka_index`] that depends only on `(perspective, piece,
/// king_sq)` — everything a whole BITBOARD of one piece type shares. Hoisted
/// out by hand for the Finny set-diff refresh's two per-type loops; LLVM will
/// not do it (`piece_square_index`'s two `cmove`s look cheap enough to sink).
#[inline]
pub fn halfka_row_base(perspective: Color, piece: Piece, king_sq: Square) -> usize {
    let flip = 56 * perspective as u8;
    let bucket = KING_BUCKETS[(king_sq.index() ^ flip) as usize] as usize;
    piece_square_index(perspective, piece) + bucket * PS_NB
}

/// The per-square half of [`halfka_index`]: `sq` mirrored into the king's half
/// of the board, `0..64`. `& 63` is a no-op on the value (both operands are
/// 6-bit) and a range fact for the optimizer, as in
/// [`ThreatResolver::index_decoded`].
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

/// A learned re-coding of the FullThreats index space: a surjection from the
/// 60,720 geometric relation codes onto the net's actual threat row space.
/// Carried by the net (CENN `TRM1` section), applied at the single resolver
/// chokepoint ([`ThreatResolver::index_decoded`]) both the from-scratch and
/// incremental paths share, so the two cannot drift. `None` = identity code.
/// `u16` fits because `THREAT_DIMS < 65536`; the loader validates every entry
/// `< threat_rows`, which is what lets the fold paths skip a per-event check.
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

/// `numValidTargets[code]` = 2 × (victim piece-types this attacker may
/// threaten): pawn 6 (P/N/R), knight 10 (P/N/B/R/Q), bishop 8, rook 8, queen
/// 10, king 0. Indexed by SF piece code (16-wide, gaps unused).
#[rustfmt::skip]
const NUM_VALID_TARGETS: [usize; 16] = [
    0, 6, 10, 8, 8, 10, 0, 0,
    0, 6, 10, 8, 8, 10, 0, 0,
];

/// `map[attacker_type-1][attacked_type-1]` (types PAWN..KING = 1..6): the dense
/// victim-relation ordinal for a legal attacker→victim type pair, `-1` when
/// excluded. Kings attack nothing, pawns threaten only P/N/R, bishops/rooks
/// skip Q & K, knights/queens include Q.
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
/// [`MAP`], the single source of truth, so the observer's push-time filter
/// cannot diverge from the fold-time drop set. The pawn-PUSH asymmetry (a push
/// onto N/R is kept by MAP but is never a threat) is guarded at the recording
/// site, not here.
#[inline]
pub(crate) fn threat_pair_kept(attacker: PieceType, victim: PieceType) -> bool {
    // Our PieceType is 0-based and MAP is indexed the same way (its doc's
    // "1..6" refers to SF's piece codes, not the array index).
    MAP[attacker as usize][victim as usize] >= 0
}

/// [`threat_pair_kept`] as a per-attacker **bit set over victim types**, so the
/// incremental observer turns "which victims count?" into a bitboard mask
/// instead of a per-square predicate call. Built from [`MAP`] in a `const`
/// block (which allows `while` but not `for`, hence the index loops), so it
/// cannot drift; `kept_victim_masks_match_predicate` pins all 36 pairs.
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

/// FullThreats square orientation — the **opposite polarity** to HalfKA: `0`
/// for a king on files a-d, `7` for files e-h. (The "a1/h1 polarity" trap:
/// HalfKA's OrientTBL is a1→SQ_H1, threats' is a1→SQ_A1.)
#[inline]
fn orient_threats(ksq: Square) -> u8 {
    if ksq.file() <= 3 { 0 } else { 7 }
}

/// Empty-board attack geometry of an SF piece code on `from` — the sets SF
/// builds its threat LUTs from. Pawns use *push-or-attacks*; everything else is
/// the plain pseudo-attack set.
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
/// its two diagonal captures. A pawn's "threat" includes the pawn it is about
/// to push into, hence the push bit.
fn pawn_push_or_attacks(color: Color, from: Square) -> Bitboard {
    let mut bb = pawn_attacks(color, from);
    if let Some(push) = from.shifted(color.pawn_push()) {
        bb |= Bitboard::from_square(push);
    }
    bb
}

/// The threat lookup tables, built once. Sizes: `from_to` 128 KiB, the rest a
/// few KiB — all indexed by SF piece code (16-wide, gap entries unused).
struct ThreatTables {
    /// `piece_offset[code]` = total empty-board attack edges for the attacker
    /// (SF `cumulativePieceOffset`). Folded into `lut1` for the index, but kept
    /// explicitly for the table-invariant test and the threats observer.
    #[allow(dead_code)]
    piece_offset: [u32; 16],
    /// `cumulative_offset[code]` = the attacker-block base (SF `cumulativeOffset`).
    /// Kept for the same reasons as `piece_offset`.
    #[allow(dead_code)]
    cumulative_offset: [u32; 16],
    /// `from_to[code][from][to]` = popcount of the attacker's attack bits
    /// strictly below `to` (the dense ordinal of `to` among `from`'s targets)
    /// PRE-ADDED to the from-square prefix base — the two terms of the index
    /// that share a key, fused at build time into one load and one add.
    ///
    /// The sum fits `u16` **structurally, not by luck**: both terms are bounded
    /// by the attacker's total empty-board edge count (`piece_offset`, ≤1456 for
    /// a queen), so the sum is under 2·1456. `build_threat_tables` asserts it
    /// and a test sweeps every entry. The cost is 128 KiB of `u16` for 64 KiB of
    /// `u8`, paid in address space, not touched lines: a lookup reads ONE entry.
    from_to: Box<[[[u16; 64]; 64]; 16]>,
    /// `lut1[attacker][attacked][from<to]` = the block+sub-block feature base, or
    /// [`THREAT_DIMS`] (an out-of-range sentinel → dropped) for an excluded or
    /// semi-excluded pair. The `from<to` axis dedups symmetric same-type threats.
    lut1: [[[u32; 2]; 16]; 16],
}

fn build_threat_tables() -> ThreatTables {
    let mut piece_offset = [0u32; 16];
    let mut cumulative_offset = [0u32; 16];
    let mut from_to = Box::new([[[0u16; 64]; 64]; 16]);

    // Per-attacker geometry: the `to`-ordinal LUT, pre-added to the prefix-sum
    // from-square base (see `ThreatTables::from_to`).
    let mut cumulative = 0u32;
    for &code in &ALL_PIECES {
        let is_pawn = code & 7 == 1;
        let mut cum_piece = 0u32;
        for from_i in 0..64u8 {
            let from = Square::new(from_i);
            let attacks = geom_attacks(code, from);
            // Fill the ordinal LUT for every from-square (SF builds it for all
            // 64, even rank 1/8 pawns that never occur in play).
            let bits = attacks.to_u64();
            // `assert`, not `debug_assert`: the fused table rests on this sum
            // fitting `u16`, it is built once at startup, and a silent wrap is
            // an arbitrarily wrong evaluation that still benches.
            let base = cum_piece + attacks.count();
            assert!(
                base <= u16::MAX as u32,
                "threat from-square base {base} overflows the fused u16 table"
            );
            for to_i in 0..64u8 {
                let below = bits & ((1u64 << to_i) - 1);
                from_to[code][from_i as usize][to_i as usize] =
                    (cum_piece + below.count_ones()) as u16;
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
            // Same-type threats are symmetric for every non-pawn and for
            // opposite-colour pawns, so one direction is dropped to avoid
            // double counting; same-colour pawn→pawn (a push) is directional.
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
        piece_offset,
        cumulative_offset,
        from_to,
        lut1,
    }
}

static THREAT_TABLES: LazyLock<ThreatTables> = LazyLock::new(build_threat_tables);

/// The FullThreats index space as a list of attacker BLOCKS: `(base, num_subs,
/// geom)` per attacker code in [`ALL_PIECES`] order, derived from the same
/// `THREAT_TABLES` the resolver uses so it cannot drift from the real layout.
///
/// The index is `base + sub · geom + ft`, `sub` the victim relation (colour ×
/// type ordinal) and `ft` the `(from, to)` geometry ordinal — so the victim
/// relation is the SLOWEST axis, and two rows of one fold sharing `(attacker,
/// from)` sit `geom` rows apart (1456 rows / 1.8 MB for a queen). Transposing a
/// block to `base + ft · num_subs + sub` is a pure permutation of the row space,
/// so it can be applied to the weight table at load and is bit-exact by
/// construction; `ARCHON_PERM=1` builds it.
pub(crate) fn threat_blocks() -> Vec<(usize, usize, usize)> {
    let t = &*THREAT_TABLES;
    ALL_PIECES
        .iter()
        .filter(|&&code| NUM_VALID_TARGETS[code] != 0)
        .map(|&code| {
            (
                t.cumulative_offset[code] as usize,
                NUM_VALID_TARGETS[code],
                t.piece_offset[code] as usize,
            )
        })
        .collect()
}

/// The FullThreats feature index of the directed relation "`attacker` on `from`
/// threatens `attacked` on `to`", as seen from `perspective` with own king on
/// `ksq` (design/11 §12.1). Returns `None` when the pair is excluded (kings, or
/// an illegal attacker→victim type) or semi-excluded (the dropped direction of a
/// symmetric same-type threat) — SF's `push_back_if_lt(index, Dimensions)`.
///
/// `index = lut1[att][atd][from<to] + from_to[att][from][to]`, the second term
/// being the from-square prefix base plus the `to`-ordinal, pre-added at build
/// time — all with the squares file/rank-oriented by `OrientTBL[ksq] ^ 56·persp`
/// and the piece codes color-swapped by `^ 8·persp`. `Some` range: `[0, 60720)`.
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

/// The per-(perspective, king) state of [`threat_index`], hoisted: the
/// `THREAT_TABLES` `LazyLock` deref plus the orientation and colour-swap
/// derivation, otherwise re-run per event. [`index`](Self::index) is the same
/// arithmetic as [`threat_index`], which delegates here, so the two cannot drift.
pub(crate) struct ThreatResolver<'r> {
    tables: &'static ThreatTables,
    orientation: u8,
    swap: usize,
    /// The net's threat re-coding, or `None` for the identity code. `Option`
    /// rather than a static identity table so a remap-less net pays a predicted
    /// branch per event instead of 121 KB of new data traffic.
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
        // `& 63` is a NO-OP on the value and a RANGE FACT for the optimizer:
        // both operands are already 6-bit (`Square::index() < 64` by the
        // newtype's constructor assert, `orientation ∈ {0, 7, 56, 63}`), but
        // LLVM only sees `u8 ^ u8 → 0..=255`, so without it every
        // `from_to[att][from_o][to_o]` index carries an unhoistable
        // `cmp $63` / `ja panic` pair.
        let from_o = ((from.index() ^ self.orientation) & 63) as usize;
        let to_o = ((to.index() ^ self.orientation) & 63) as usize;
        let att = attacker_code ^ self.swap;
        let atd = attacked_code ^ self.swap;
        let dir = (from_o < to_o) as usize;

        let base = t.lut1[att][atd][dir];
        if base >= THREAT_DIMS as u32 {
            return None; // excluded / semi-excluded
        }
        let idx = (base + t.from_to[att][from_o][to_o] as u32) as usize;
        // The Recode chokepoint: one table lookup re-codes the geometric index
        // into the net's row space. `.min(THREAT_DIMS - 1)` is a NO-OP on the
        // value (the tiling test pins `idx < THREAT_DIMS`) and a range fact for
        // the optimizer, discharging the [`ThreatRemap`] bounds check without a
        // panic edge — same move as the `& 63` above.
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
/// union, and both king squares — exactly what [`for_each_threat_at`] and
/// [`super::sota_threats`]'s incremental generator/resolver read, nothing else.
///
/// One definition, two consumers:
/// [`super::sota_threats::ThreatContext::apply_at`] captures it at push time so
/// fold-time regeneration runs against the post-move geometry even after a
/// sibling has destroyed the live [`Board`] in place; and the from-scratch
/// refresh runs off the *same* struct, which is what lets a king-cross rebuild
/// be **deferred** to the ply that actually evaluates.
///
/// Bitboards and not the mailbox because the incremental generator then finds
/// the zero-to-two sliders x-raying a touched square by intersecting with
/// victim-type boards — the same trick the from-scratch enumeration uses —
/// instead of up to eight per-square identity resolutions.
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
    /// `debug_assert` with a release fallback, so a one-king board trips here in
    /// debug even though only `kings[perspective]` is ever read. The search path
    /// already carries this precondition; the `pub` datagen callers in
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
    /// primitive serves both passes of the generic replay — undo every edit to
    /// recover the pre-move position, then redo them one at a time.
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
        // Same tripwire as `piece_on`'s: the pawn-push branch of
        // `for_each_threat_at` calls this directly, guarded only by a pawn
        // mask, and an empty square silently reads as Black.
        debug_assert!(self.occ.contains(sq), "color_on an empty square");
        if self.colored(Color::White).contains(sq) {
            Color::White
        } else {
            Color::Black
        }
    }

    /// The whole piece on `sq`; `sq` must be occupied. Five type tests against
    /// the mailbox form's one byte load, which is why callers are structured so
    /// this is reached only where the victim's type is not a loop constant.
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
/// board, from `perspective` — SF's `append_active_indices`: every piece of
/// both colours is an attacker, victim sets are pre-filtered to the legal
/// types, and excluded / semi-excluded relations are skipped.
pub fn for_each_threat(board: &Board, perspective: Color, f: impl FnMut(usize)) {
    for_each_threat_at(&ThreatPosition::from_board(board), perspective, None, f);
}

/// [`for_each_threat`] over a [`ThreatPosition`] snapshot rather than a live
/// [`Board`] — the form the deferred king-cross rebuild needs. `remap` is the
/// net's threat re-coding ([`ThreatRemap`]; `None` = identity), passed by the
/// refresh paths so the from-scratch build speaks the incremental fold's row
/// space.
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
        // Kings are never attackers (their whole `MAP` row is -1), so no loop.
    }
}

// ===========================================================================
// From-scratch accumulator refresh (format-agnostic weight slices)
// ===========================================================================

/// Add the `ACC`-wide column at feature `index` of a feature-major weight slice
/// into `acc` with wrapping i16 math (the quantized accumulator wraps, exactly
/// like the head's FT sum). `W` is `i16` for HalfKA, `i8` for threats.
#[inline]
fn add_column<W: Copy + Into<i16>>(acc: &mut [i16; ACC], weights: &[W], index: usize) {
    let col = &weights[index * ACC..index * ACC + ACC];
    for (a, &w) in acc.iter_mut().zip(col.iter()) {
        *a = a.wrapping_add(w.into());
    }
}

/// [`add_column`] for the FullThreats table, which may store a row in the
/// packed 6-bit form. A correctness path, not a speed one (the from-scratch
/// rebuild is ~0.02 calls/node): it decodes lane by lane with the SAME scalar
/// reference the SIMD kernel's tests are checked against.
#[inline]
fn add_threat_column(acc: &mut [i16; ACC], weights: &[i8], row_map: &[u32], index: usize) {
    let (base, narrow) = super::sota_simd::row_entry(row_map, index * ACC);
    if narrow {
        let row = &weights[base..base + super::sota_simd::NARROW_BYTES];
        for (j, a) in acc.iter_mut().enumerate() {
            *a = a.wrapping_add(super::sota_simd::unpack_narrow_lane(row, j));
        }
    } else {
        for (a, &w) in acc.iter_mut().zip(&weights[base..base + ACC]) {
            *a = a.wrapping_add(i16::from(w));
        }
    }
}

/// Build the HalfKA accumulator for `perspective` from scratch: seed with the FT
/// bias, then add the weight column of every active feature. `halfka_weights` is
/// feature-major (`[HALFKA_DIMS · ACC]`), `ft_bias` is `ACC`-wide.
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
/// then add the weight column of every active threat feature. `threat_weights`
/// is feature-major (`[THREAT_DIMS · ACC]` i8). **Test-only** — production
/// callers use [`refresh_threat_into`] and skip the doubled zeroing.
#[cfg(test)]
pub fn refresh_threat(threat_weights: &[i8], board: &Board, perspective: Color) -> [i16; ACC] {
    let mut acc = [0i16; ACC];
    refresh_threat_into(&mut acc, threat_weights, &[], board, perspective, None);
    acc
}

/// [`refresh_threat`] writing through a caller-owned destination, so the
/// accumulator is not built in a stack local and `memcpy`d — the same shape as
/// `HalfkaContext::finny_refresh_into`. `dst` is fully overwritten (zeroed
/// first, then accumulated into), so a caller may pass a stale buffer.
pub fn refresh_threat_into(
    dst: &mut [i16; ACC],
    threat_weights: &[i8],
    threat_rowmap: &[u32],
    board: &Board,
    perspective: Color,
    remap: Option<&ThreatRemap>,
) {
    refresh_threat_into_at(
        dst,
        threat_weights,
        threat_rowmap,
        &ThreatPosition::from_board(board),
        perspective,
        remap,
    );
}

/// [`refresh_threat_into`] over a [`ThreatPosition`] snapshot — what the
/// deferred king-cross rebuild calls, since by the time the crossing ply is
/// evaluated the live board has moved on.
pub(super) fn refresh_threat_into_at(
    dst: &mut [i16; ACC],
    threat_weights: &[i8],
    // The narrow-plane bitmap; empty ⇒ every row is 8-bit.
    threat_rowmap: &[u32],
    pos: &ThreatPosition,
    perspective: Color,
    remap: Option<&ThreatRemap>,
) {
    dst.fill(0);
    if threat_weights.is_empty() {
        return;
    }
    // A COMPACTED table is not whole `ACC`-byte rows any more (narrow rows are
    // 960 B and the planes carry a trailing pad), so the shape checks below
    // apply only to the flat layout: when a row map is present it IS the shape
    // statement, built by the loader from the table it also built.
    if threat_rowmap.is_empty() {
        match remap {
            // Identity code: the weight table spans the full geometric space.
            None => assert_eq!(
                threat_weights.len(),
                THREAT_DIMS * ACC,
                "FullThreats transformer has the wrong size"
            ),
            // Re-coded net: the table spans `threat_rows` rows and the loader
            // validated every remap entry against that count, so whole-rows is
            // the only shape left to check.
            Some(_) => assert_eq!(
                threat_weights.len() % ACC,
                0,
                "FullThreats transformer is not whole rows"
            ),
        }
    }
    for_each_threat_at(pos, perspective, remap, |idx| {
        add_threat_column(dst, threat_weights, threat_rowmap, idx)
    });
}

/// Build both accumulators for both perspectives — the `(pst, threat)` pair
/// [`super::sota_head::SotaHead::forward`] wants (index 0 = White, 1 = Black).
pub fn refresh_both(
    halfka_weights: &[i8],
    ft_bias: &[i16],
    threat_weights: &[i8],
    threat_rowmap: &[u32],
    board: &Board,
    threat_remap: Option<&ThreatRemap>,
) -> ([[i16; ACC]; 2], [[i16; ACC]; 2]) {
    let mut pst = [[0i16; ACC]; 2];
    let mut threat = [[0i16; ACC]; 2];
    // One snapshot for both perspectives, the same hoist
    // `ThreatContext::refresh` makes. Cold path, so this is tidiness.
    let pos = ThreatPosition::from_board(board);
    for (i, &persp) in Color::ALL.iter().enumerate() {
        pst[i] = refresh_pst(halfka_weights, ft_bias, board, persp);
        refresh_threat_into_at(
            &mut threat[i],
            threat_weights,
            threat_rowmap,
            &pos,
            persp,
            threat_remap,
        );
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

    /// The index reproduces SF `make_index`, hand-computed from the formula.
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
        // Black perspective, WHITE pawn a2, BLACK king e8: orient 0, flip 56 ⇒
        // sq 8^56=48; enemy pawn ⇒ PS_B_PAWN(64); KingBuckets[4]=31 ·704.
        assert_eq!(
            halfka_index(Color::Black, wp(), Square::A2, Square::E8),
            21936
        );
    }

    /// Horizontal mirror symmetry: a queenside king setup and its kingside
    /// mirror map to the SAME index (the point of the file-mirror orient).
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

    /// The empty-board geometry sizes match SF's per-attacker edge counts
    /// (P 132, N 336, B 560, R 896, Q 1456, K 420) and tile exactly 60,720.
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

    /// The fused `from_to` table IS `offsets[from] + lut2[from][to]` entry for
    /// entry, and every entry fits `u16`. Both summands are re-derived from
    /// their DEFINITIONS rather than read back out of the table, so this is an
    /// independent recomputation and not a tautology.
    #[test]
    fn threats_from_to_is_the_fused_sum() {
        let t = &*THREAT_TABLES;
        for &code in &ALL_PIECES {
            let is_pawn = code & 7 == 1;
            let mut cum_piece = 0u32;
            for from_i in 0..64u8 {
                let bits = geom_attacks(code, Square::new(from_i)).to_u64();
                for to_i in 0..64u8 {
                    let want = cum_piece + (bits & ((1u64 << to_i) - 1)).count_ones();
                    assert!(
                        want <= u16::MAX as u32,
                        "code {code} from {from_i} to {to_i}"
                    );
                    assert_eq!(
                        u32::from(t.from_to[code][from_i as usize][to_i as usize]),
                        want,
                        "from_to[{code}][{from_i}][{to_i}]"
                    );
                }
                if !is_pawn || (8..=55).contains(&from_i) {
                    cum_piece += bits.count_ones();
                }
            }
        }
    }

    /// Hand-computed: white knight b1 → black pawn c3, white king a1 (orient
    /// 0), white view. lut1 = 792 + (1·5 + 0)·336 = 2472; from_to[Nw][b1][c3] =
    /// 2 (knight a1 has 2 edges) + 2 (the `to`-ordinal).
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
        // Same-type rook→rook: `from < to` is dropped, the other kept.
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

    /// On real positions every active threat index is in range and DISTINCT —
    /// the index math is collision-free.
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

    /// All-ones columns, so a neuron equals `bias + #active` = bias + pieces.
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
    /// column set to a ramp, the accumulator must be exactly `bias[j] +
    /// ramp[j]` — pinning the `index·ACC + j` layout, that the king is an
    /// active feature, and bias seeding at once. A transposed index reads a
    /// zero column and fails.
    #[test]
    fn refresh_pst_stride_and_king_feature() {
        let board = Board::from_fen(crate::fen::START_FEN).unwrap();
        let mut w = vec![0i8; HALFKA_DIMS * ACC];
        let bias: Vec<i16> = (0..ACC).map(|j| (j % 7) as i16).collect();

        let ksq = board.king_square(Color::White);
        let king_feat = halfka_index(Color::White, wk(), ksq, ksq);
        // A per-lane ramp that fits i8: the point is that lane `j` of the king
        // row lands in lane `j` of the accumulator (the row STRIDE).
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

    /// Refresh is deterministic and `refresh_both` agrees with the singles.
    #[test]
    fn refresh_deterministic_and_consistent() {
        let board =
            Board::from_fen("r1bqkbnr/pppp1ppp/2n5/4p3/2B1P3/5N2/PPPP1PPP/RNBQK2R w KQkq - 0 1")
                .unwrap();
        let (hw, bias) = ones_halfka();
        let tw = vec![2i8; THREAT_DIMS * ACC];

        let a = refresh_both(&hw, &bias, &tw, &[], &board, None);
        let b = refresh_both(&hw, &bias, &tw, &[], &board, None);
        assert_eq!(a, b, "deterministic");

        let (pst, threat) = &a;
        assert_eq!(pst[0], refresh_pst(&hw, &bias, &board, Color::White));
        assert_eq!(pst[1], refresh_pst(&hw, &bias, &board, Color::Black));
        assert_eq!(threat[0], refresh_threat(&tw, &board, Color::White));
        assert_eq!(threat[1], refresh_threat(&tw, &board, Color::Black));
    }

    /// End-to-end wiring: the refresh output feeds `SotaHead::forward`.
    #[test]
    fn refresh_feeds_sota_head() {
        use super::super::sota_head::{BUCKETS, L1, L2, L3, SotaHead};
        let board = Board::from_fen(crate::fen::START_FEN).unwrap();
        let (hw, bias) = ones_halfka();
        let tw = vec![0i8; THREAT_DIMS * ACC];
        let (pst, threat) = refresh_both(&hw, &bias, &tw, &[], &board, None);

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
            dis_head: None,
            ft_scale: crate::eval::nnue::sota_head::FtScale::from_flat_iter(std::iter::repeat_n(
                1u8, ACC,
            )),
            bucket_layout: crate::eval::nnue::sota_head::LEGACY_BUCKET_LAYOUT,
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
    /// fold's `DeltaBuf::push` uses [`threat_pair_kept`]; a disagreement makes
    /// the incremental accumulator silently keep or drop a relation the
    /// from-scratch refresh does not. Both derive from `MAP`; this pins it.
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

    /// `KEPT_VICTIMS` and `MAP` are indexed by `PieceType as usize` while the
    /// observer's loops walk `PieceType::ALL`; all three line up only if `ALL`
    /// is in discriminant order.
    #[test]
    fn piece_type_all_is_in_discriminant_order() {
        for (i, pt) in PieceType::ALL.iter().enumerate() {
            assert_eq!(i, *pt as usize, "PieceType::ALL[{i}] is {pt:?}");
        }
    }
}
