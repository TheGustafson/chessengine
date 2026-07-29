use crate::types::{PieceType, Square};
use std::fmt;
use std::mem::MaybeUninit;
use std::ops::{Deref, DerefMut};

/// A move packed into 16 bits, Stockfish-style:
///
/// - bits 0-5:   from square
/// - bits 6-11:  to square
/// - bits 12-13: promotion piece (0 = knight, ..., 3 = queen)
/// - bits 14-15: kind (0 = normal, 1 = promotion, 2 = en passant, 3 = castling)
///
/// Castling is encoded as the king's move (e1g1), matching UCI notation.
#[derive(Clone, Copy, PartialEq, Eq, Hash)]
pub struct Move(u16);

#[derive(Debug, Clone, Copy, PartialEq, Eq)]
pub enum MoveKind {
    Normal,
    Promotion,
    EnPassant,
    Castling,
}

impl Move {
    /// Sentinel that is not a real move (a1 to a1). Used to fill move tables.
    pub const NULL: Move = Move(0);

    const KIND_PROMOTION: u16 = 1;
    const KIND_EN_PASSANT: u16 = 2;
    const KIND_CASTLING: u16 = 3;

    const fn encode(from: Square, to: Square, promotion: u16, kind: u16) -> Move {
        Move((from.index() as u16) | ((to.index() as u16) << 6) | (promotion << 12) | (kind << 14))
    }

    pub const fn normal(from: Square, to: Square) -> Move {
        Move::encode(from, to, 0, 0)
    }

    pub const fn promotion(from: Square, to: Square, piece_type: PieceType) -> Move {
        debug_assert!(matches!(
            piece_type,
            PieceType::Knight | PieceType::Bishop | PieceType::Rook | PieceType::Queen
        ));
        Move::encode(from, to, piece_type as u16 - 1, Move::KIND_PROMOTION)
    }

    pub const fn en_passant(from: Square, to: Square) -> Move {
        Move::encode(from, to, 0, Move::KIND_EN_PASSANT)
    }

    pub const fn castling(from: Square, to: Square) -> Move {
        Move::encode(from, to, 0, Move::KIND_CASTLING)
    }

    pub const fn from(self) -> Square {
        Square::new((self.0 & 0x3F) as u8)
    }

    pub const fn to(self) -> Square {
        Square::new(((self.0 >> 6) & 0x3F) as u8)
    }

    pub const fn kind(self) -> MoveKind {
        match self.0 >> 14 {
            0 => MoveKind::Normal,
            Move::KIND_PROMOTION => MoveKind::Promotion,
            Move::KIND_EN_PASSANT => MoveKind::EnPassant,
            _ => MoveKind::Castling,
        }
    }

    /// The piece a pawn promotes to, if this is a promotion.
    pub fn promotion_piece(self) -> Option<PieceType> {
        match self.kind() {
            MoveKind::Promotion => Some(PieceType::ALL[((self.0 >> 12) & 0b11) as usize + 1]),
            _ => None,
        }
    }

    /// The FULL 16-bit encoding as an integer: the index into the butterfly
    /// main history `[color][65536]` and the TT's `move16` field
    /// (design/02-core-types.md §3.7, D2, D19).
    ///
    /// Full — not just from/to — so promotions to different pieces and the
    /// castling flag get distinct history slots.
    pub const fn raw16(self) -> u16 {
        self.0
    }

    /// Rebuild a `Move` from raw bits (the TT probe path).
    ///
    /// Every `u16` decodes to a structurally valid `Move` (from/to are 6-bit
    /// fields, all kind values are meaningful), so this is total. SEMANTIC
    /// validity is the caller's job: TT consumers must run pseudo-legality +
    /// legality checks on any probed move before playing it (D19).
    pub const fn from_raw16(bits: u16) -> Move {
        Move(bits)
    }

    /// True for every real move, false for a none/null sentinel.
    ///
    /// The invariant is `from() != to()` (design/09 §3.4, D2): a legal move
    /// always changes the moving piece's square, whereas [`Move::NULL`]
    /// (a1→a1) and any future null-move sentinel share `from == to`. History
    /// writes and the continuation-history walk gate on this so a null move
    /// or the padding slots below the root never poison a table.
    pub const fn is_ok(self) -> bool {
        (self.0 & 0x3F) != ((self.0 >> 6) & 0x3F)
    }
}

/// UCI long algebraic notation: "e2e4", "e7e8q".
impl fmt::Display for Move {
    fn fmt(&self, f: &mut fmt::Formatter<'_>) -> fmt::Result {
        write!(f, "{}{}", self.from(), self.to())?;
        if let Some(piece_type) = self.promotion_piece() {
            write!(f, "{}", piece_type.to_char())?;
        }
        Ok(())
    }
}

impl fmt::Debug for Move {
    fn fmt(&self, f: &mut fmt::Formatter<'_>) -> fmt::Result {
        write!(f, "Move({self}, {:?})", self.kind())
    }
}

/// Upper bound on moves in any legal position (the known record is 218).
pub const MAX_MOVES: usize = 256;

/// A fixed-capacity, stack-allocated list of moves.
///
/// Dereferences to `[Move]`, so slice methods (`len`, `iter`, indexing,
/// `sort_unstable_by_key`, ...) all work on it.
///
/// **UNINITIALISED backing array** (perf study 20 arm 4) — the same exemption,
/// for the same reason, that `MovePicker::buf` has carried since perf study 2
/// option C. `moves: [Move; MAX_MOVES]` forced `MoveList::new()` to write 512
/// bytes of `Move::NULL` before every single move generation, and LLVM cannot
/// eliminate it: `generate` takes `&mut MoveList` across an outlined call, so
/// it cannot prove the callee fills the buffer. Measured in the shipped binary
/// as eight `vmovups %zmm0` plus a `vxorps`/`movq`/`vzeroupper` at the head of
/// `init_quiets`, `init_captures` and `init_evasions` — and `generate` then
/// overwrites exactly the prefix that is ever read.
///
/// SAFETY ARGUMENT. [`push`](Self::push) is the ONLY writer, and it writes
/// index `len` before incrementing `len`; `len` moves down only in
/// [`clear`](Self::clear) and [`swap_remove`](Self::swap_remove). So
/// `moves[..len]` is initialised at every observable point, which is exactly
/// the range [`Deref`] hands out. The invariant is machine-checked in debug
/// builds by the `high_water` tracker below — every write raises it, every read
/// asserts `len <= high_water` — so a future edit that advances `len` without
/// writing trips a debug assertion rather than reading a stale move in release.
/// Release compiles the tracker out.
#[derive(Clone)]
pub struct MoveList {
    moves: [MaybeUninit<Move>; MAX_MOVES],
    len: usize,
    /// Debug-only: one past the highest index ever written by `push`. See the
    /// safety argument above.
    #[cfg(debug_assertions)]
    high_water: usize,
}

impl MoveList {
    /// Deliberately NOT a `const fn`, and deliberately not a repeat-expression
    /// (perf study 20 arm 4). `[MaybeUninit::uninit(); MAX_MOVES]` in a
    /// const-evaluable constructor is materialised as a 512-byte *constant*
    /// whose uninit bytes lower to a zero store — the shipped binary still
    /// carried eight `vmovups %zmm0` at the head of `init_quiets` after the
    /// field type changed. `MaybeUninit::<[MaybeUninit<_>; N]>::uninit()
    /// .assume_init()` is the documented always-sound spelling (the outer
    /// `MaybeUninit` is discharged by an array whose elements are themselves
    /// allowed to be uninitialised) and lowers to a bare `alloca`.
    #[inline]
    #[allow(unsafe_code)]
    pub fn new() -> Self {
        MoveList {
            // SAFETY: the value produced is `[MaybeUninit<Move>; MAX_MOVES]`,
            // every element of which is *permitted* to be uninitialised — this
            // is the standard-library-documented use of `assume_init`, and no
            // element is read before `push` writes it (see the type docs).
            moves: unsafe { MaybeUninit::<[MaybeUninit<Move>; MAX_MOVES]>::uninit().assume_init() },
            len: 0,
            #[cfg(debug_assertions)]
            high_water: 0,
        }
    }

    pub fn push(&mut self, m: Move) {
        assert!(self.len < MAX_MOVES, "MoveList overflow");
        self.moves[self.len] = MaybeUninit::new(m);
        self.len += 1;
        #[cfg(debug_assertions)]
        {
            self.high_water = self.high_water.max(self.len);
        }
    }

    pub fn clear(&mut self) {
        self.len = 0;
    }

    /// The initialised prefix. The one place the uninitialised backing array is
    /// reinterpreted as `[Move]`.
    #[inline]
    #[allow(unsafe_code)]
    fn initialised(&self) -> &[Move] {
        #[cfg(debug_assertions)]
        debug_assert!(
            self.len <= self.high_water,
            "MoveList len {} outran the {} moves ever written",
            self.len,
            self.high_water
        );
        let prefix = &self.moves[..self.len];
        // SAFETY: `push` is the only writer and initialises index `len` before
        // raising `len`, so every element of `moves[..len]` has been written.
        // `MaybeUninit<Move>` is `#[repr(transparent)]` over `Move`, so the
        // slice cast is a no-op reinterpretation of the same layout.
        unsafe { &*(prefix as *const [MaybeUninit<Move>] as *const [Move]) }
    }

    /// Mutable twin of [`initialised`](Self::initialised).
    #[inline]
    #[allow(unsafe_code)]
    fn initialised_mut(&mut self) -> &mut [Move] {
        #[cfg(debug_assertions)]
        debug_assert!(
            self.len <= self.high_water,
            "MoveList len outran its writes"
        );
        let prefix = &mut self.moves[..self.len];
        // SAFETY: as `initialised`.
        unsafe { &mut *(prefix as *mut [MaybeUninit<Move>] as *mut [Move]) }
    }

    /// O(1) unordered removal: overwrite index `i` with the last move and
    /// shrink by one. Returns the removed move. Panics if `i >= len` (index
    /// invariant, matching `Vec::swap_remove`).
    ///
    /// Order is deliberately not preserved — the consumer is
    /// `movegen::generate_legal`'s lazy legality filter (design/05 §3.9),
    /// whose callers either sort (MovePicker) or count (perft).
    pub fn swap_remove(&mut self, i: usize) -> Move {
        assert!(
            i < self.len,
            "swap_remove index {i} out of range (len {})",
            self.len
        );
        let live = self.initialised_mut();
        let removed = live[i];
        let last = live[live.len() - 1];
        live[i] = last;
        self.len -= 1;
        removed
    }
}

impl Default for MoveList {
    fn default() -> Self {
        MoveList::new()
    }
}

impl Deref for MoveList {
    type Target = [Move];

    #[inline]
    fn deref(&self) -> &[Move] {
        self.initialised()
    }
}

impl DerefMut for MoveList {
    #[inline]
    fn deref_mut(&mut self) -> &mut [Move] {
        self.initialised_mut()
    }
}

impl<'a> IntoIterator for &'a MoveList {
    type Item = Move;
    type IntoIter = std::iter::Copied<std::slice::Iter<'a, Move>>;

    fn into_iter(self) -> Self::IntoIter {
        self.iter().copied()
    }
}

impl fmt::Debug for MoveList {
    fn fmt(&self, f: &mut fmt::Formatter<'_>) -> fmt::Result {
        f.debug_list().entries(self.iter()).finish()
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn normal_move_roundtrip() {
        let m = Move::normal(Square::E2, Square::E4);
        assert_eq!(m.from(), Square::E2);
        assert_eq!(m.to(), Square::E4);
        assert_eq!(m.kind(), MoveKind::Normal);
        assert_eq!(m.promotion_piece(), None);
        assert_eq!(m.to_string(), "e2e4");
    }

    #[test]
    fn moves_fit_in_two_bytes() {
        assert_eq!(std::mem::size_of::<Move>(), 2);
    }

    #[test]
    fn extreme_squares_roundtrip() {
        let m = Move::normal(Square::H8, Square::A1);
        assert_eq!(m.from(), Square::H8);
        assert_eq!(m.to(), Square::A1);
        assert_eq!(m.to_string(), "h8a1");
    }

    #[test]
    fn promotion_roundtrip_all_pieces() {
        for piece_type in [
            PieceType::Knight,
            PieceType::Bishop,
            PieceType::Rook,
            PieceType::Queen,
        ] {
            let m = Move::promotion(Square::E7, Square::E8, piece_type);
            assert_eq!(m.from(), Square::E7);
            assert_eq!(m.to(), Square::E8);
            assert_eq!(m.kind(), MoveKind::Promotion);
            assert_eq!(m.promotion_piece(), Some(piece_type));
        }
        let m = Move::promotion(Square::A7, Square::B8, PieceType::Queen);
        assert_eq!(m.to_string(), "a7b8q");
    }

    #[test]
    fn en_passant_and_castling_kinds() {
        let ep = Move::en_passant(Square::E5, Square::D6);
        assert_eq!(ep.kind(), MoveKind::EnPassant);
        assert_eq!(ep.promotion_piece(), None);
        assert_eq!(ep.to_string(), "e5d6");

        let castle = Move::castling(Square::E1, Square::G1);
        assert_eq!(castle.kind(), MoveKind::Castling);
        assert_eq!(castle.to_string(), "e1g1");
    }

    #[test]
    fn distinct_kinds_are_distinct_moves() {
        // Same from/to, different kind must not compare equal.
        let normal = Move::normal(Square::E5, Square::D6);
        let ep = Move::en_passant(Square::E5, Square::D6);
        assert_ne!(normal, ep);
    }

    // Doc 02 §5 item 11: raw16 round-trip, totality, kind-distinctness.
    #[test]
    fn raw16_roundtrip_totality_and_distinctness() {
        // Round-trip one move of each kind, including all four promotions.
        let moves = [
            Move::normal(Square::E2, Square::E4),
            Move::en_passant(Square::E5, Square::D6),
            Move::castling(Square::E1, Square::G1),
            Move::promotion(Square::E7, Square::E8, PieceType::Knight),
            Move::promotion(Square::E7, Square::E8, PieceType::Bishop),
            Move::promotion(Square::E7, Square::E8, PieceType::Rook),
            Move::promotion(Square::E7, Square::E8, PieceType::Queen),
        ];
        for m in moves {
            assert_eq!(Move::from_raw16(m.raw16()), m);
        }

        // Totality: every u16 decodes without panic to a structurally valid
        // move (semantic validity is the TT consumer's job).
        let m = Move::from_raw16(0xFFFF);
        assert_eq!(m.kind(), MoveKind::Castling);
        assert_eq!(m.from(), Square::H8);
        assert_eq!(m.to(), Square::H8);

        // Distinct kinds on the same from/to yield distinct raw16 values.
        let (from, to) = (Square::E5, Square::D6);
        let raws = [
            Move::normal(from, to).raw16(),
            Move::en_passant(from, to).raw16(),
            Move::castling(from, to).raw16(),
            Move::promotion(from, to, PieceType::Queen).raw16(),
        ];
        for (i, a) in raws.iter().enumerate() {
            for b in &raws[i + 1..] {
                assert_ne!(a, b);
            }
        }

        // Sentinel bookkeeping (doc 02 §3.7, D2): NULL is raw 0.
        assert_eq!(Move::NULL.raw16(), 0);
    }

    // design/09 §3.4 / step 1: is_ok() distinguishes real moves from the
    // none/null sentinel by the from != to invariant.
    #[test]
    fn is_ok_distinguishes_null_from_real_moves() {
        // The none-sentinel and any same-square pattern are not ok.
        assert!(!Move::NULL.is_ok());
        assert!(!Move::normal(Square::E4, Square::E4).is_ok());
        assert!(!Move::normal(Square::A1, Square::A1).is_ok());
        assert!(!Move::normal(Square::H8, Square::H8).is_ok());

        // Every real move of every kind changes the square, so is ok.
        assert!(Move::normal(Square::E2, Square::E4).is_ok());
        assert!(Move::en_passant(Square::E5, Square::D6).is_ok());
        assert!(Move::castling(Square::E1, Square::G1).is_ok());
        for pt in [
            PieceType::Knight,
            PieceType::Bishop,
            PieceType::Rook,
            PieceType::Queen,
        ] {
            assert!(Move::promotion(Square::E7, Square::E8, pt).is_ok());
        }

        // Exhaustive: from != to iff is_ok, across all from/to pairs.
        for from in Square::all() {
            for to in Square::all() {
                assert_eq!(Move::normal(from, to).is_ok(), from != to);
            }
        }
    }

    #[test]
    fn movelist_push_and_slice_access() {
        let mut list = MoveList::new();
        assert!(list.is_empty());

        list.push(Move::normal(Square::E2, Square::E4));
        list.push(Move::normal(Square::G1, Square::F3));

        assert_eq!(list.len(), 2);
        assert_eq!(list[0], Move::normal(Square::E2, Square::E4));
        assert_eq!(list[1], Move::normal(Square::G1, Square::F3));

        let uci: Vec<String> = list.iter().map(Move::to_string).collect();
        assert_eq!(uci, vec!["e2e4", "g1f3"]);

        for m in &list {
            assert_eq!(m.kind(), MoveKind::Normal);
        }

        list.clear();
        assert!(list.is_empty());
    }

    #[test]
    fn movelist_fills_to_capacity() {
        let mut list = MoveList::new();
        for i in 0..MAX_MOVES {
            list.push(Move::normal(
                Square::new((i % 64) as u8),
                Square::new((i / 4) as u8 % 64),
            ));
        }
        assert_eq!(list.len(), MAX_MOVES);
    }

    #[test]
    #[should_panic(expected = "MoveList overflow")]
    fn movelist_panics_on_overflow() {
        let mut list = MoveList::new();
        for _ in 0..=MAX_MOVES {
            list.push(Move::NULL);
        }
    }

    // design/05 §4 step 1: middle / last / single element, panic on bad index.

    #[test]
    fn swap_remove_middle_element() {
        let (a, b, c) = (
            Move::normal(Square::E2, Square::E4),
            Move::normal(Square::G1, Square::F3),
            Move::normal(Square::B1, Square::C3),
        );
        let mut list = MoveList::new();
        list.push(a);
        list.push(b);
        list.push(c);

        assert_eq!(list.swap_remove(1), b);
        assert_eq!(list.len(), 2);
        // The last element was swapped into the hole; order is not preserved.
        assert_eq!(list[0], a);
        assert_eq!(list[1], c);
    }

    #[test]
    fn swap_remove_last_element() {
        let (a, b) = (
            Move::normal(Square::E2, Square::E4),
            Move::normal(Square::G1, Square::F3),
        );
        let mut list = MoveList::new();
        list.push(a);
        list.push(b);

        assert_eq!(list.swap_remove(1), b);
        assert_eq!(list.len(), 1);
        assert_eq!(list[0], a);
    }

    #[test]
    fn swap_remove_single_element() {
        let a = Move::normal(Square::E2, Square::E4);
        let mut list = MoveList::new();
        list.push(a);

        assert_eq!(list.swap_remove(0), a);
        assert!(list.is_empty());
    }

    #[test]
    #[should_panic(expected = "swap_remove index 2 out of range (len 2)")]
    fn swap_remove_panics_on_bad_index() {
        let mut list = MoveList::new();
        list.push(Move::normal(Square::E2, Square::E4));
        list.push(Move::normal(Square::G1, Square::F3));
        list.swap_remove(2);
    }

    #[test]
    #[should_panic(expected = "swap_remove index 0 out of range (len 0)")]
    fn swap_remove_panics_on_empty_list() {
        let mut list = MoveList::new();
        list.swap_remove(0);
    }
}
