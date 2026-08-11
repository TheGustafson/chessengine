use std::fmt;
use std::str::FromStr;

#[derive(Debug, Clone, Copy, PartialEq, Eq, Hash)]
#[repr(u8)]
pub enum PieceType {
    Pawn,
    Knight,
    Bishop,
    Rook,
    Queen,
    King,
}

impl PieceType {
    pub const COUNT: usize = 6;
    pub const ALL: [PieceType; PieceType::COUNT] = [
        PieceType::Pawn,
        PieceType::Knight,
        PieceType::Bishop,
        PieceType::Rook,
        PieceType::Queen,
        PieceType::King,
    ];

    /// Lowercase FEN character for this piece type.
    pub const fn to_char(self) -> char {
        match self {
            PieceType::Pawn => 'p',
            PieceType::Knight => 'n',
            PieceType::Bishop => 'b',
            PieceType::Rook => 'r',
            PieceType::Queen => 'q',
            PieceType::King => 'k',
        }
    }

    /// Parse a lowercase FEN character.
    pub const fn from_char(c: char) -> Option<PieceType> {
        match c {
            'p' => Some(PieceType::Pawn),
            'n' => Some(PieceType::Knight),
            'b' => Some(PieceType::Bishop),
            'r' => Some(PieceType::Rook),
            'q' => Some(PieceType::Queen),
            'k' => Some(PieceType::King),
            _ => None,
        }
    }

    /// Exchange (SEE) value of this piece type; see [`PIECE_VALUES`].
    pub const fn value(self) -> Value {
        PIECE_VALUES[self as usize]
    }
}

#[derive(Debug, Clone, Copy, PartialEq, Eq, Hash)]
#[repr(u8)]
pub enum Color {
    White,
    Black,
}

impl Color {
    pub const COUNT: usize = 2;
    pub const ALL: [Color; Color::COUNT] = [Color::White, Color::Black];

    pub const fn flip(self) -> Self {
        match self {
            Color::White => Color::Black,
            Color::Black => Color::White,
        }
    }

    /// The forward pawn-push direction for this color.
    pub const fn pawn_push(self) -> Direction {
        match self {
            Color::White => Direction::North,
            Color::Black => Direction::South,
        }
    }
}

#[derive(Debug, Clone, Copy, PartialEq, Eq, Hash)]
pub struct Piece {
    pub color: Color,
    pub piece_type: PieceType,
}

impl Piece {
    /// Number of distinct (color, piece_type) pairs.
    pub const COUNT: usize = 12;

    pub const fn new(color: Color, piece_type: PieceType) -> Self {
        Piece { color, piece_type }
    }

    /// Flat table index: White P..K = 0..6, Black P..K = 6..12. Deliberately
    /// NOT Stockfish's 16-slot packed piece: `[T; 12]` tables are denser.
    pub const fn index(self) -> usize {
        self.color as usize * PieceType::COUNT + self.piece_type as usize
    }

    /// FEN character: uppercase for white, lowercase for black.
    pub fn to_char(self) -> char {
        let c = self.piece_type.to_char();
        match self.color {
            Color::White => c.to_ascii_uppercase(),
            Color::Black => c,
        }
    }

    /// Parse a FEN character: uppercase for white, lowercase for black.
    pub fn from_char(c: char) -> Option<Piece> {
        let piece_type = PieceType::from_char(c.to_ascii_lowercase())?;
        let color = if c.is_ascii_uppercase() {
            Color::White
        } else {
            Color::Black
        };
        Some(Piece::new(color, piece_type))
    }
}

impl fmt::Display for Piece {
    fn fmt(&self, f: &mut fmt::Formatter<'_>) -> fmt::Result {
        write!(f, "{}", self.to_char())
    }
}

/// A square on the board, 0..64 with a1 = 0, b1 = 1, ..., h8 = 63
/// (little-endian rank-file mapping).
#[derive(Debug, Clone, Copy, PartialEq, Eq, Hash, PartialOrd, Ord)]
pub struct Square {
    index: u8,
}

macro_rules! define_square_consts {
    ($($name:ident = $index:expr),* $(,)?) => {
        impl Square {
            $(pub const $name: Square = Square { index: $index };)*
        }
    };
}

define_square_consts!(
    A1 = 0,
    B1 = 1,
    C1 = 2,
    D1 = 3,
    E1 = 4,
    F1 = 5,
    G1 = 6,
    H1 = 7,
    A2 = 8,
    B2 = 9,
    C2 = 10,
    D2 = 11,
    E2 = 12,
    F2 = 13,
    G2 = 14,
    H2 = 15,
    A3 = 16,
    B3 = 17,
    C3 = 18,
    D3 = 19,
    E3 = 20,
    F3 = 21,
    G3 = 22,
    H3 = 23,
    A4 = 24,
    B4 = 25,
    C4 = 26,
    D4 = 27,
    E4 = 28,
    F4 = 29,
    G4 = 30,
    H4 = 31,
    A5 = 32,
    B5 = 33,
    C5 = 34,
    D5 = 35,
    E5 = 36,
    F5 = 37,
    G5 = 38,
    H5 = 39,
    A6 = 40,
    B6 = 41,
    C6 = 42,
    D6 = 43,
    E6 = 44,
    F6 = 45,
    G6 = 46,
    H6 = 47,
    A7 = 48,
    B7 = 49,
    C7 = 50,
    D7 = 51,
    E7 = 52,
    F7 = 53,
    G7 = 54,
    H7 = 55,
    A8 = 56,
    B8 = 57,
    C8 = 58,
    D8 = 59,
    E8 = 60,
    F8 = 61,
    G8 = 62,
    H8 = 63,
);

impl Square {
    pub const COUNT: usize = 64;

    pub const fn new(index: u8) -> Self {
        assert!(index < 64, "Square index must be between 0 and 63");
        Square { index }
    }

    pub const fn from_coords(file: u8, rank: u8) -> Self {
        assert!(
            file < 8 && rank < 8,
            "file and rank must be between 0 and 7"
        );
        Square {
            index: rank * 8 + file,
        }
    }

    pub const fn index(self) -> u8 {
        self.index
    }

    /// The square as a **table subscript**: `& 63` states the `< 64` range
    /// proof where the optimizer can use it, deleting the bounds check on
    /// every `[T; 64]` lookup (the mask is the identity on every `Square`).
    /// Use at subscripts; keep [`Self::index`] for arithmetic.
    #[inline]
    pub const fn idx(self) -> usize {
        self.index as usize & (Self::COUNT - 1)
    }

    /// File 0-7 (a-h).
    pub const fn file(self) -> u8 {
        self.index & 0b111
    }

    /// Rank 0-7 (1-8).
    pub const fn rank(self) -> u8 {
        self.index >> 3
    }

    /// All 64 squares in index order (a1, b1, ..., h8).
    pub fn all() -> impl Iterator<Item = Square> {
        (0..64).map(Square::new)
    }
}

impl fmt::Display for Square {
    fn fmt(&self, f: &mut fmt::Formatter<'_>) -> fmt::Result {
        write!(
            f,
            "{}{}",
            (b'a' + self.file()) as char,
            (b'1' + self.rank()) as char
        )
    }
}

#[derive(Debug, Clone, PartialEq, Eq)]
pub struct ParseSquareError;

impl fmt::Display for ParseSquareError {
    fn fmt(&self, f: &mut fmt::Formatter<'_>) -> fmt::Result {
        write!(f, "expected a square like \"e4\" (file a-h, rank 1-8)")
    }
}

impl std::error::Error for ParseSquareError {}

impl FromStr for Square {
    type Err = ParseSquareError;

    fn from_str(s: &str) -> Result<Self, Self::Err> {
        let [file_char, rank_char] = s.as_bytes() else {
            return Err(ParseSquareError);
        };
        let file = file_char.wrapping_sub(b'a');
        let rank = rank_char.wrapping_sub(b'1');
        if file < 8 && rank < 8 {
            Ok(Square::from_coords(file, rank))
        } else {
            Err(ParseSquareError)
        }
    }
}

// --- Direction and Square arithmetic ---------------------------------------

/// A one-square step, encoded as the board-index delta. Knight steps are
/// deliberately NOT `Direction`s — they are raw `i8` step values used only
/// by table builders, which is why [`Square::try_offset`] takes an `i8`.
#[derive(Debug, Clone, Copy, PartialEq, Eq)]
#[repr(i8)]
pub enum Direction {
    North = 8,
    NorthEast = 9,
    East = 1,
    SouthEast = -7,
    South = -8,
    SouthWest = -9,
    West = -1,
    NorthWest = 7,
}

impl Direction {
    /// All eight directions, clockwise from North.
    pub const ALL: [Direction; 8] = [
        Direction::North,
        Direction::NorthEast,
        Direction::East,
        Direction::SouthEast,
        Direction::South,
        Direction::SouthWest,
        Direction::West,
        Direction::NorthWest,
    ];

    /// The raw index delta.
    pub const fn offset(self) -> i8 {
        self as i8
    }
}

impl Square {
    /// File distance `|file(a) - file(b)|`.
    pub const fn file_distance(self, other: Square) -> u8 {
        self.file().abs_diff(other.file())
    }

    /// Rank distance `|rank(a) - rank(b)|`.
    pub const fn rank_distance(self, other: Square) -> u8 {
        self.rank().abs_diff(other.rank())
    }

    /// Chebyshev distance: king moves between the squares. The reference
    /// implementation behind the `attacks::distance` table.
    pub const fn distance(self, other: Square) -> u8 {
        let f = self.file_distance(other);
        let r = self.rank_distance(other);
        if f > r { f } else { r }
    }

    /// Step by a raw index delta; `None` if off-board or wrapped around an
    /// edge. The single safe primitive for table building. The wrap test is
    /// a Chebyshev bound: legitimate steps move <= 2 files/ranks, wraps >= 6.
    pub const fn try_offset(self, step: i8) -> Option<Square> {
        let to = self.index() as i8 + step;
        if to < 0 || to >= 64 {
            return None;
        }
        let to = Square::new(to as u8);
        if self.distance(to) > 2 {
            None
        } else {
            Some(to)
        }
    }

    /// Step by a [`Direction`]. `None` if off-board or wrapped. Slider-ray
    /// walkers apply this one step at a time, so the wrap bound is never a
    /// false negative.
    pub const fn shifted(self, d: Direction) -> Option<Square> {
        self.try_offset(d.offset())
    }

    /// `self + d` where the CALLER guarantees the destination exists and
    /// does not wrap. Safe Rust (D14): a range escape still panics via
    /// `Square::new`; a file wrap is a caller bug the perft suite catches.
    pub const fn offset(self, d: Direction) -> Square {
        Square::new((self.index() as i8 + d.offset()) as u8)
    }

    /// Vertical mirror: a1 <-> a8. Branchless (XOR with 56).
    pub const fn flip_rank(self) -> Square {
        Square::new(self.index() ^ 56)
    }

    /// Horizontal mirror: a1 <-> h1. Branchless (XOR with 7).
    pub const fn flip_file(self) -> Square {
        Square::new(self.index() ^ 7)
    }

    /// The square from `color`'s point of view: identity for White, vertical
    /// mirror for Black (`E2.relative_to(Black) == E7`). Branchless via the
    /// `#[repr(u8)]` discriminants.
    pub const fn relative_to(self, color: Color) -> Square {
        Square::new(self.index() ^ (color as u8 * 56))
    }

    /// Rank from `color`'s point of view (0 = own back rank).
    pub const fn relative_rank(self, color: Color) -> u8 {
        self.rank() ^ (color as u8 * 7)
    }
}

// --- Value: search/eval scores (design/02 §3.2) -----------------------------

/// A search/eval score; positive is good for the side to move. A plain
/// alias, not a newtype (D5). All in-search values lie in
/// `(-VALUE_NONE, VALUE_NONE]` and MUST fit an `i16` (the TT stores i16s).
pub type Value = i32;

/// The zero score.
pub const VALUE_ZERO: Value = 0;
/// The draw score.
pub const VALUE_DRAW: Value = 0;
/// Mate at the root: `mate_in(0)`. Every real score is `<=` this.
pub const VALUE_MATE: Value = 32000;
/// Strictly greater than any real score; used only as a search window bound.
pub const VALUE_INFINITE: Value = 32001;
/// "No value" sentinel (empty TT fields); test with [`is_valid`].
pub const VALUE_NONE: Value = 32002;

// The TT stores values in i16 fields (D19); the sentinels must fit.
const _: () = assert!(VALUE_NONE <= i16::MAX as i32);

/// Maximum search ply. `usize` because its primary job is sizing arrays.
/// Must stay `<= 252` so the TT's `u8` depth packing holds.
pub const MAX_PLY: usize = 246;

// The number line, positive side (negatives mirror):
//
// 0 ...(normal evals)... 31506 | 31507 ..(TB band).. 31753 | 31754 ..(mate band).. 32000 | 32001 INF | 32002 NONE
//                VALUE_TB_WIN_IN_MAX_PLY      VALUE_TB      VALUE_MATE_IN_MAX_PLY    VALUE_MATE
//
// Decisive scores must never be pruned, softened, averaged, or corrected;
// the bands make those gates single comparisons. The TB band exists from
// day one (even with no probing) so no threshold ever has to shift.

/// Lower edge of the mate band: `mate_in(MAX_PLY)` (31754).
pub const VALUE_MATE_IN_MAX_PLY: Value = VALUE_MATE - MAX_PLY as Value;
/// Upper edge of the mated band, mirror of [`VALUE_MATE_IN_MAX_PLY`] (-31754).
pub const VALUE_MATED_IN_MAX_PLY: Value = -VALUE_MATE_IN_MAX_PLY;

/// Tablebase win at the current node, directly below the mate band (31753).
pub const VALUE_TB: Value = VALUE_MATE_IN_MAX_PLY - 1;
/// Lower edge of the TB-win band (31507).
pub const VALUE_TB_WIN_IN_MAX_PLY: Value = VALUE_TB - MAX_PLY as Value;
/// Upper edge of the TB-loss band (-31507).
pub const VALUE_TB_LOSS_IN_MAX_PLY: Value = -VALUE_TB_WIN_IN_MAX_PLY;

/// True iff `v` is a real value and not the [`VALUE_NONE`] sentinel.
pub const fn is_valid(v: Value) -> bool {
    v != VALUE_NONE
}

/// TB win or better (includes mates) — NOT "mate only"; the narrow
/// mate-band test is [`is_mate`]. Misreading this corrupts pruning gates.
pub const fn is_win(v: Value) -> bool {
    debug_assert!(is_valid(v));
    v >= VALUE_TB_WIN_IN_MAX_PLY
}

/// TB loss or worse (includes being mated) — mirror of [`is_win`]; the
/// narrow mated-band test is [`is_mated`].
pub const fn is_loss(v: Value) -> bool {
    debug_assert!(is_valid(v));
    v <= VALUE_TB_LOSS_IN_MAX_PLY
}

/// Any decisive score — "anything search must never soften".
pub const fn is_decisive(v: Value) -> bool {
    is_win(v) || is_loss(v)
}

/// Strictly a mate-for-us score (not merely a TB win).
pub const fn is_mate(v: Value) -> bool {
    debug_assert!(is_valid(v));
    v >= VALUE_MATE_IN_MAX_PLY
}

/// Strictly a mated score (not merely a TB loss).
pub const fn is_mated(v: Value) -> bool {
    debug_assert!(is_valid(v));
    v <= VALUE_MATED_IN_MAX_PLY
}

/// Score for delivering mate at `ply` plies from the ROOT (ply-relative,
/// D5). `mate_in(0) == VALUE_MATE`; `mate_in(MAX_PLY)` is the band edge.
pub const fn mate_in(ply: i32) -> Value {
    VALUE_MATE - ply
}

/// Score for being mated at `ply`: `mated_in(p) == -mate_in(p)`.
pub const fn mated_in(ply: i32) -> Value {
    -VALUE_MATE + ply
}

/// Root-relative -> node-relative, applied before a TT store: the TT stores
/// mate/TB scores relative to the CURRENT node, otherwise the same position
/// reached at different plies would disagree.
pub const fn value_to_tt(v: Value, ply: i32) -> Value {
    debug_assert!(is_valid(v));
    if is_win(v) {
        v + ply
    } else if is_loss(v) {
        v - ply
    } else {
        v
    }
}

/// Node-relative -> root-relative, applied after a TT probe. Downgrades
/// decisive scores the 50-move clock has made unreachable: collapse to just
/// outside the TB band — still dominating normal evals, no longer tripping
/// the "never prune decisive scores" gates.
pub const fn value_from_tt(v: Value, ply: i32, rule50: i32) -> Value {
    if !is_valid(v) {
        return VALUE_NONE;
    }
    if is_win(v) {
        // 100 plies = the 50-move rule; rule50 already used up.
        if is_mate(v) && VALUE_MATE - v > 100 - rule50 {
            return VALUE_TB_WIN_IN_MAX_PLY - 1; // stale mate score
        }
        if VALUE_TB - v > 100 - rule50 {
            return VALUE_TB_WIN_IN_MAX_PLY - 1; // stale TB score
        }
        return v - ply;
    }
    if is_loss(v) {
        if is_mated(v) && VALUE_MATE + v > 100 - rule50 {
            return VALUE_TB_LOSS_IN_MAX_PLY + 1;
        }
        if VALUE_TB + v > 100 - rule50 {
            return VALUE_TB_LOSS_IN_MAX_PLY + 1;
        }
        return v + ply;
    }
    v
}

// --- Depth ------------------------------------------------------------------

/// Remaining search depth in plies. `i32`: sentinels are negative and depth
/// arithmetic freely goes through zero. A real search never recurses with
/// depth < 0 — qsearch is entered instead.
pub type Depth = i32;

/// Depth of a quiescence-search TT entry.
pub const DEPTH_QS: Depth = 0;
/// TT entries written without any search; below every real depth.
pub const DEPTH_UNSEARCHED: Depth = -2;
/// "No depth" marker for empty/invalid TT entries; below `DEPTH_UNSEARCHED`.
pub const DEPTH_NONE: Depth = -3;

// --- Key: Zobrist hash keys -------------------------------------------------

/// A Zobrist hash key: the main position hash and the four specialized keys
/// (pawn/material/minor/non-pawn). A plain alias, not a newtype —
/// `make_move` is dense XOR arithmetic. Tables live in [`crate::zobrist`].
pub type Key = u64;

// --- SEE piece values -------------------------------------------------------

// SF-current starting points (D20). Tuning constraint: summed non-pawn
// material must uniquely identify material configurations
// (`non_pawn_material` doubles as a material signature); retuning must
// preserve that.

/// Exchange (SEE) value of a pawn.
pub const PAWN_VALUE: Value = 208;
/// Exchange (SEE) value of a knight.
pub const KNIGHT_VALUE: Value = 781;
/// Exchange (SEE) value of a bishop.
pub const BISHOP_VALUE: Value = 825;
/// Exchange (SEE) value of a rook.
pub const ROOK_VALUE: Value = 1276;
/// Exchange (SEE) value of a queen.
pub const QUEEN_VALUE: Value = 2538;

/// Exchange values indexed by `PieceType as usize`. King = 0: SEE sequences
/// terminate before a king is ever "captured", and a 0 keeps loops total.
pub const PIECE_VALUES: [Value; PieceType::COUNT] = [
    PAWN_VALUE,
    KNIGHT_VALUE,
    BISHOP_VALUE,
    ROOK_VALUE,
    QUEEN_VALUE,
    VALUE_ZERO,
];

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn square_file_rank() {
        assert_eq!(Square::A1.file(), 0);
        assert_eq!(Square::A1.rank(), 0);
        assert_eq!(Square::H1.file(), 7);
        assert_eq!(Square::H1.rank(), 0);
        assert_eq!(Square::A8.file(), 0);
        assert_eq!(Square::A8.rank(), 7);
        assert_eq!(Square::H8.file(), 7);
        assert_eq!(Square::H8.rank(), 7);
        assert_eq!(Square::E4, Square::from_coords(4, 3));
    }

    #[test]
    fn square_display_parse_roundtrip() {
        for sq in Square::all() {
            let parsed: Square = sq.to_string().parse().unwrap();
            assert_eq!(parsed, sq);
        }
        assert_eq!("e4".parse::<Square>().unwrap(), Square::E4);
        assert_eq!(Square::A1.to_string(), "a1");
        assert_eq!(Square::H8.to_string(), "h8");
    }

    #[test]
    fn square_parse_rejects_garbage() {
        assert!("".parse::<Square>().is_err());
        assert!("e".parse::<Square>().is_err());
        assert!("e44".parse::<Square>().is_err());
        assert!("i1".parse::<Square>().is_err());
        assert!("a9".parse::<Square>().is_err());
        assert!("a0".parse::<Square>().is_err());
    }

    #[test]
    #[should_panic]
    fn square_new_rejects_out_of_range() {
        Square::new(64);
    }

    #[test]
    fn piece_char_roundtrip() {
        for color in Color::ALL {
            for piece_type in PieceType::ALL {
                let piece = Piece::new(color, piece_type);
                assert_eq!(Piece::from_char(piece.to_char()), Some(piece));
            }
        }
        assert_eq!(
            Piece::from_char('K'),
            Some(Piece::new(Color::White, PieceType::King))
        );
        assert_eq!(
            Piece::from_char('q'),
            Some(Piece::new(Color::Black, PieceType::Queen))
        );
        assert_eq!(Piece::from_char('x'), None);
        assert_eq!(Piece::from_char('1'), None);
    }

    #[test]
    fn color_flip() {
        assert_eq!(Color::White.flip(), Color::Black);
        assert_eq!(Color::Black.flip(), Color::White);
    }

    /// The (file delta, rank delta) a direction means in coordinate space.
    fn direction_deltas(d: Direction) -> (i8, i8) {
        match d {
            Direction::North => (0, 1),
            Direction::NorthEast => (1, 1),
            Direction::East => (1, 0),
            Direction::SouthEast => (1, -1),
            Direction::South => (0, -1),
            Direction::SouthWest => (-1, -1),
            Direction::West => (-1, 0),
            Direction::NorthWest => (-1, 1),
        }
    }

    // Exhaustive 64×8 stepping vs the coordinate model.
    #[test]
    fn square_shifted_exhaustive_vs_coordinate_model() {
        for d in Direction::ALL {
            let (df, dr) = direction_deltas(d);
            assert_eq!(d.offset(), dr * 8 + df, "encoding of {d:?}");
            for sq in Square::all() {
                let file = sq.file() as i8 + df;
                let rank = sq.rank() as i8 + dr;
                let expected = if (0..8).contains(&file) && (0..8).contains(&rank) {
                    Some(Square::from_coords(file as u8, rank as u8))
                } else {
                    None
                };
                assert_eq!(sq.shifted(d), expected, "{sq}.shifted({d:?})");
                if let Some(to) = expected {
                    assert_eq!(sq.offset(d), to, "{sq}.offset({d:?})");
                }
            }
        }
        assert_eq!(Square::H4.shifted(Direction::East), None);
        assert_eq!(Square::H4.shifted(Direction::NorthEast), None);
        assert_eq!(Square::A1.shifted(Direction::SouthWest), None);
        assert_eq!(Square::E4.shifted(Direction::North), Some(Square::E5));
        assert_eq!(Color::White.pawn_push(), Direction::North);
        assert_eq!(Color::Black.pawn_push(), Direction::South);
    }

    #[test]
    fn square_try_offset_knight_steps_and_wraps() {
        assert_eq!(Square::G1.try_offset(17), Some(Square::H3));
        assert_eq!(Square::H1.try_offset(10), None); // wraps to b3, distance 6
        assert_eq!(Square::A8.try_offset(15), None); // off-board
        assert_eq!(Square::E2.try_offset(16), Some(Square::E4)); // double push
        assert_eq!(Square::A1.distance(Square::H8), 7);
        assert_eq!(Square::H1.file_distance(Square::B3), 6);
        assert_eq!(Square::H1.rank_distance(Square::B3), 2);
    }

    #[test]
    fn square_xor_transforms() {
        assert_eq!(Square::A1.flip_rank(), Square::A8);
        assert_eq!(Square::A1.flip_file(), Square::H1);
        assert_eq!(Square::E2.relative_to(Color::Black), Square::E7);
        assert_eq!(Square::E2.relative_to(Color::White), Square::E2);
        assert_eq!(Square::E2.relative_rank(Color::White), 1);
        assert_eq!(Square::E2.relative_rank(Color::Black), 6);
        for sq in Square::all() {
            // Involutions; White's view is the identity; Black's relative
            // rank agrees with the vertical mirror.
            assert_eq!(sq.flip_rank().flip_rank(), sq);
            assert_eq!(sq.flip_file().flip_file(), sq);
            assert_eq!(sq.relative_to(Color::Black).relative_to(Color::Black), sq);
            assert_eq!(sq.relative_to(Color::White), sq);
            assert_eq!(sq.relative_rank(Color::White), sq.rank());
            assert_eq!(sq.relative_rank(Color::Black), sq.flip_rank().rank());
        }
    }

    #[test]
    fn value_constant_table() {
        assert_eq!(VALUE_ZERO, 0);
        assert_eq!(VALUE_DRAW, 0);
        assert_eq!(VALUE_MATE, 32000);
        assert_eq!(VALUE_INFINITE, 32001);
        assert_eq!(VALUE_NONE, 32002);
        assert_eq!(MAX_PLY, 246);
        assert_eq!(VALUE_MATE_IN_MAX_PLY, 31754);
        assert_eq!(VALUE_MATED_IN_MAX_PLY, -31754);
        assert_eq!(VALUE_TB, 31753);
        assert_eq!(VALUE_TB_WIN_IN_MAX_PLY, 31507);
        assert_eq!(VALUE_TB_LOSS_IN_MAX_PLY, -31507);
        assert_eq!(DEPTH_QS, 0);
        assert_eq!(DEPTH_UNSEARCHED, -2);
        assert_eq!(DEPTH_NONE, -3);
        const {
            assert!(VALUE_TB_WIN_IN_MAX_PLY < VALUE_TB);
            assert!(VALUE_TB < VALUE_MATE_IN_MAX_PLY);
            assert!(VALUE_MATE_IN_MAX_PLY < VALUE_MATE);
            assert!(VALUE_MATE < VALUE_INFINITE);
            assert!(VALUE_INFINITE < VALUE_NONE);
            assert!(DEPTH_NONE < DEPTH_UNSEARCHED);
            assert!(DEPTH_UNSEARCHED < DEPTH_QS);
        }
    }

    #[test]
    fn value_predicates_band_edges() {
        assert!(is_valid(VALUE_ZERO));
        assert!(is_valid(VALUE_MATE));
        assert!(is_valid(VALUE_INFINITE));
        assert!(is_valid(-VALUE_INFINITE));
        assert!(!is_valid(VALUE_NONE));

        assert!(is_win(VALUE_TB_WIN_IN_MAX_PLY));
        assert!(!is_win(VALUE_TB_WIN_IN_MAX_PLY - 1));
        assert!(is_loss(VALUE_TB_LOSS_IN_MAX_PLY));
        assert!(!is_loss(VALUE_TB_LOSS_IN_MAX_PLY + 1));

        // is_mate is the NARROW band: a TB win is a win but not a mate.
        assert!(is_mate(mate_in(MAX_PLY as i32)));
        assert!(!is_mate(VALUE_TB));
        assert!(is_win(VALUE_TB)); // ...but it IS a win.
        assert!(is_mated(mated_in(MAX_PLY as i32)));
        assert!(!is_mated(-VALUE_TB));
        assert!(is_loss(-VALUE_TB));

        for v in [VALUE_TB_WIN_IN_MAX_PLY, VALUE_TB, VALUE_MATE] {
            assert!(is_decisive(v));
            assert!(is_decisive(-v));
        }
        assert!(!is_decisive(VALUE_DRAW));
        assert!(!is_decisive(1000));
        assert!(!is_decisive(-1000));
        assert!(!is_decisive(VALUE_TB_WIN_IN_MAX_PLY - 1));

        assert_eq!(mate_in(0), VALUE_MATE);
        assert_eq!(mate_in(MAX_PLY as i32), VALUE_MATE_IN_MAX_PLY);
        assert_eq!(mated_in(0), -VALUE_MATE);
        assert_eq!(mated_in(MAX_PLY as i32), VALUE_MATED_IN_MAX_PLY);
        for p in [0, 1, 5, 40, MAX_PLY as i32] {
            assert_eq!(mated_in(p), -mate_in(p));
        }
    }

    // Round-trips with an unexpired clock; decisive values keep distance
    // >= 40 so every ply in the matrix is realistic.
    #[test]
    fn tt_value_roundtrip() {
        let values = [
            mate_in(50),
            mated_in(45),
            VALUE_TB - 50,
            -(VALUE_TB - 50),
            0,
            130,
            -2500,
        ];
        for v in values {
            for ply in [0, 1, 5, 40] {
                assert_eq!(
                    value_from_tt(value_to_tt(v, ply), ply, 0),
                    v,
                    "round-trip failed for v={v}, ply={ply}"
                );
            }
        }
    }

    #[test]
    fn tt_value_downgrade_rule50() {
        // mate_in(10) stored at ply 4 needs more clock than the 5 plies
        // remaining: collapse to the highest non-decisive score.
        assert_eq!(
            value_from_tt(value_to_tt(mate_in(10), 4), 4, 95),
            VALUE_TB_WIN_IN_MAX_PLY - 1
        );
        assert_eq!(
            value_from_tt(value_to_tt(mated_in(10), 4), 4, 95),
            VALUE_TB_LOSS_IN_MAX_PLY + 1
        );
        // Stale TB scores (band intact, clock expired) downgrade the same way.
        assert_eq!(
            value_from_tt(VALUE_TB - 5, 0, 98),
            VALUE_TB_WIN_IN_MAX_PLY - 1
        );
        assert_eq!(
            value_from_tt(-(VALUE_TB - 5), 0, 98),
            VALUE_TB_LOSS_IN_MAX_PLY + 1
        );
        // A TB score stored deeper than its own distance reads back stale.
        assert_eq!(
            value_from_tt(value_to_tt(VALUE_TB - 5, 40), 40, 0),
            VALUE_TB_WIN_IN_MAX_PLY - 1
        );
        assert_eq!(value_from_tt(VALUE_NONE, 12, 30), VALUE_NONE);
    }

    #[test]
    fn piece_values() {
        assert_eq!(PAWN_VALUE, 208);
        assert_eq!(KNIGHT_VALUE, 781);
        assert_eq!(BISHOP_VALUE, 825);
        assert_eq!(ROOK_VALUE, 1276);
        assert_eq!(QUEEN_VALUE, 2538);
        assert_eq!(PieceType::Pawn.value(), PAWN_VALUE);
        assert_eq!(PieceType::Knight.value(), KNIGHT_VALUE);
        assert_eq!(PieceType::Bishop.value(), BISHOP_VALUE);
        assert_eq!(PieceType::Rook.value(), ROOK_VALUE);
        assert_eq!(PieceType::Queen.value(), QUEEN_VALUE);
        assert_eq!(PieceType::King.value(), 0);
        for pt in PieceType::ALL {
            assert_eq!(pt.value(), PIECE_VALUES[pt as usize]);
        }
        const {
            assert!(PAWN_VALUE < KNIGHT_VALUE);
            assert!(KNIGHT_VALUE < BISHOP_VALUE);
            assert!(BISHOP_VALUE < ROOK_VALUE);
            assert!(ROOK_VALUE < QUEEN_VALUE);
        }
    }

    #[test]
    fn piece_index_flat() {
        let mut seen = [false; Piece::COUNT];
        for color in Color::ALL {
            for piece_type in PieceType::ALL {
                let idx = Piece::new(color, piece_type).index();
                assert!(idx < Piece::COUNT, "index {idx} out of range");
                assert!(!seen[idx], "index {idx} assigned twice");
                seen[idx] = true;
            }
        }
        assert!(seen.iter().all(|&s| s), "indices must cover exactly 0..12");
        assert_eq!(Piece::new(Color::White, PieceType::Pawn).index(), 0);
        assert_eq!(Piece::new(Color::Black, PieceType::King).index(), 11);
    }

    // Downstream code builds const tables and fixed arrays on these.
    #[test]
    fn value_layer_is_const_evaluable() {
        const MATE_AT_ROOT: Value = mate_in(0);
        const { assert!(is_decisive(mate_in(MAX_PLY as i32))) }
        let _stack = [0u8; MAX_PLY + 10];
        assert_eq!(MATE_AT_ROOT, VALUE_MATE);
    }
}
