use crate::attacks;
use crate::bitboard::Bitboard;
use crate::types::{Color, Key, MAX_PLY, PIECE_VALUES, Piece, PieceType, Square, Value};
use crate::zobrist::{self, ZOBRIST};
use std::fmt;
use std::str::FromStr;

#[derive(Debug, Clone, Copy, PartialEq, Eq, Hash, Default)]
pub struct CastlingRights(u8);

impl CastlingRights {
    pub const NONE: CastlingRights = CastlingRights(0);
    pub const WHITE_KINGSIDE: CastlingRights = CastlingRights(0b0001);
    pub const WHITE_QUEENSIDE: CastlingRights = CastlingRights(0b0010);
    pub const BLACK_KINGSIDE: CastlingRights = CastlingRights(0b0100);
    pub const BLACK_QUEENSIDE: CastlingRights = CastlingRights(0b1000);
    pub const ALL: CastlingRights = CastlingRights(0b1111);

    pub fn add(&mut self, rights: CastlingRights) {
        self.0 |= rights.0;
    }

    pub fn remove(&mut self, rights: CastlingRights) {
        self.0 &= !rights.0;
    }

    pub const fn contains(self, rights: CastlingRights) -> bool {
        (self.0 & rights.0) == rights.0
    }

    pub const fn is_empty(self) -> bool {
        self.0 == 0
    }

    /// The raw 4-bit mask (bit 0 = White kingside, 1 = White queenside,
    /// 2 = Black kingside, 3 = Black queenside): the index into the 16-entry
    /// `zobrist` castling table, which keys rights by the WHOLE mask so any
    /// rights change costs two XORs, old mask out, new mask in
    /// (design/04-position-state.md §3.1/§3.4, D9).
    pub const fn bits(self) -> u8 {
        self.0
    }

    /// Bit position 0..4 of a SINGLE right (WK = 0, WQ = 1, BK = 2, BQ = 3):
    /// the index into `Board`'s castling geometry arrays
    /// (design/04-position-state.md §3.7).
    pub const fn index(self) -> usize {
        debug_assert!(
            self.0.count_ones() == 1,
            "CastlingRights::index needs exactly one right"
        );
        self.0.trailing_zeros() as usize
    }
}

/// FEN castling field: "KQkq", any sub-combination, or "-".
impl fmt::Display for CastlingRights {
    fn fmt(&self, f: &mut fmt::Formatter<'_>) -> fmt::Result {
        if self.is_empty() {
            return write!(f, "-");
        }
        if self.contains(CastlingRights::WHITE_KINGSIDE) {
            write!(f, "K")?;
        }
        if self.contains(CastlingRights::WHITE_QUEENSIDE) {
            write!(f, "Q")?;
        }
        if self.contains(CastlingRights::BLACK_KINGSIDE) {
            write!(f, "k")?;
        }
        if self.contains(CastlingRights::BLACK_QUEENSIDE) {
            write!(f, "q")?;
        }
        Ok(())
    }
}

#[derive(Debug, Clone, PartialEq, Eq)]
pub struct ParseCastlingRightsError;

impl fmt::Display for ParseCastlingRightsError {
    fn fmt(&self, f: &mut fmt::Formatter<'_>) -> fmt::Result {
        write!(
            f,
            "expected \"-\" or a non-repeating combination of K, Q, k, q"
        )
    }
}

impl std::error::Error for ParseCastlingRightsError {}

impl FromStr for CastlingRights {
    type Err = ParseCastlingRightsError;

    fn from_str(s: &str) -> Result<Self, Self::Err> {
        if s == "-" {
            return Ok(CastlingRights::NONE);
        }
        if s.is_empty() {
            return Err(ParseCastlingRightsError);
        }
        let mut rights = CastlingRights::NONE;
        for c in s.chars() {
            let right = match c {
                'K' => CastlingRights::WHITE_KINGSIDE,
                'Q' => CastlingRights::WHITE_QUEENSIDE,
                'k' => CastlingRights::BLACK_KINGSIDE,
                'q' => CastlingRights::BLACK_QUEENSIDE,
                _ => return Err(ParseCastlingRightsError),
            };
            if rights.contains(right) {
                return Err(ParseCastlingRightsError);
            }
            rights.add(right);
        }
        Ok(rights)
    }
}

/// Fields carried over from the previous state and then incrementally
/// edited by `make_move` (design/04-position-state.md §3.2, D8). Plain
/// `Copy` — the `copied: prev.copied` assignment in `make_move` is our safe
/// spelling of Stockfish's "memcpy the struct prefix".
#[derive(Debug, Clone, Copy, PartialEq, Eq)]
pub struct CopiedState {
    /// Pawns only, both colors; baseline `ZOBRIST.no_pawns` when pawnless.
    pub pawn_key: Key,
    /// Piece counts via the count-slot trick (`zobrist::Tables::material`).
    pub material_key: Key,
    /// Knights + bishops, both colors.
    pub minor_key: Key,
    /// Per-color non-pawns INCLUDING the king.
    pub non_pawn_key: [Key; 2],
    /// Per-color sum of `PIECE_VALUES` over N, B, R, Q.
    pub non_pawn_material: [Value; 2],
    pub castling_rights: CastlingRights,
    /// Halfmove clock (50-move rule counter).
    pub rule50: i32,
    /// Plies since the last null move (bounds the repetition scan window).
    pub plies_from_null: i32,
    /// Set ONLY when an en-passant capture is actually available (D9);
    /// see the FEN validation in `from_fen` and `make_move`'s double-push
    /// legality test.
    pub ep_square: Option<Square>,
}

/// One ply of undo/search state (design/04-position-state.md §3.2, D8).
/// `Copy` so `make_null_move` can clone the whole record cheaply.
#[derive(Debug, Clone, Copy, PartialEq, Eq)]
pub struct StateInfo {
    /// The carried-over block; everything below is recomputed per move.
    pub copied: CopiedState,
    /// Full position hash, RAW — rule50 folding happens in [`Board::key`].
    pub key: Key,
    /// Enemy pieces giving check to the side to move.
    pub checkers: Bitboard,
    /// Indexed by king-owner color; contains pieces of BOTH colors.
    pub blockers_for_king: [Bitboard; 2],
    /// `pinners[c]`: color `c` sliders pinning against color `!c`'s king.
    pub pinners: [Bitboard; 2],
    /// Per `PieceType`: from-squares that would check the enemy king.
    pub check_squares: [Bitboard; 6],
    /// Piece captured by the move that produced this state.
    pub captured: Option<Piece>,
    /// Repetition encoding: 0 / +distance / -distance (doc 04 §3.9).
    pub repetition: i32,
}

impl StateInfo {
    /// The all-empty baseline record. Every real state is produced from
    /// this by `init_state` (setup) or derived from its parent (`make_move`).
    const EMPTY: StateInfo = StateInfo {
        copied: CopiedState {
            pawn_key: 0,
            material_key: 0,
            minor_key: 0,
            non_pawn_key: [0; 2],
            non_pawn_material: [0; 2],
            castling_rights: CastlingRights::NONE,
            rule50: 0,
            plies_from_null: 0,
            ep_square: None,
        },
        key: 0,
        checkers: Bitboard::EMPTY,
        blockers_for_king: [Bitboard::EMPTY; 2],
        pinners: [Bitboard::EMPTY; 2],
        check_squares: [Bitboard::EMPTY; 6],
        captured: None,
        repetition: 0,
    };
}

/// From-scratch key/material computation result — `init_state` applies it,
/// and the test/debug oracle compares it against the (eventually
/// incrementally maintained) stored state (doc 04 §3.12).
struct ComputedKeys {
    key: Key,
    pawn_key: Key,
    material_key: Key,
    minor_key: Key,
    non_pawn_key: [Key; 2],
    non_pawn_material: [Value; 2],
}

#[derive(Debug, Clone, PartialEq, Eq)]
pub struct Board {
    // Primary representation: bitboards.
    pub(crate) pieces: [[Bitboard; PieceType::COUNT]; Color::COUNT],
    pub(crate) color_occupancy: [Bitboard; Color::COUNT],
    pub(crate) all_occupancy: Bitboard,

    // Redundant mailbox for O(1) piece lookup by square.
    pub(crate) mailbox: [Option<Piece>; Square::COUNT],

    pub(crate) side_to_move: Color,
    /// Halfmoves since the game's start position (doc 04 §3.3).
    game_ply: i32,

    // Castling geometry, fixed at setup (Chess960-ready shape, D10).
    castling_rights_mask: [CastlingRights; Square::COUNT],
    castling_rook_square: [Square; 4],
    castling_path: [Bitboard; 4],

    // Ply-indexed state history; NEVER empty — every constructor pushes the
    // root state (D8). `make_move` pushes, `unmake_move` pops.
    states: Vec<StateInfo>,
}

impl Board {
    pub fn empty() -> Self {
        // Reserved past MAX_PLY so mid-search pushes never reallocate
        // (doc 04 §3.3; growth would be correct, just a latency spike).
        let mut states = Vec::with_capacity(MAX_PLY + 64);
        states.push(StateInfo::EMPTY);
        let mut board = Board {
            pieces: [[Bitboard::EMPTY; PieceType::COUNT]; Color::COUNT],
            color_occupancy: [Bitboard::EMPTY; Color::COUNT],
            all_occupancy: Bitboard::EMPTY,
            mailbox: [None; Square::COUNT],
            side_to_move: Color::White,
            game_ply: 0,
            castling_rights_mask: [CastlingRights::NONE; Square::COUNT],
            castling_rook_square: [Square::A1; 4],
            castling_path: [Bitboard::EMPTY; 4],
            states,
        };
        // Even the empty board's root state is self-consistent (its key is
        // ZOBRIST.castling[0], its pawn key ZOBRIST.no_pawns, ...).
        board.init_state();
        board
    }

    pub fn piece_on(&self, sq: Square) -> Option<Piece> {
        self.mailbox[sq.index() as usize]
    }

    /// The squares holding `color`'s pieces of type `piece_type`.
    pub fn pieces(&self, color: Color, piece_type: PieceType) -> Bitboard {
        self.pieces[color as usize][piece_type as usize]
    }

    /// All squares occupied by `color`.
    pub fn occupancy(&self, color: Color) -> Bitboard {
        self.color_occupancy[color as usize]
    }

    /// All occupied squares.
    pub fn all_occupancy(&self) -> Bitboard {
        self.all_occupancy
    }

    pub fn side_to_move(&self) -> Color {
        self.side_to_move
    }

    /// The current (top-of-stack) state. The stack is never empty (D8).
    fn st(&self) -> &StateInfo {
        self.states.last().expect("state stack never empty")
    }

    fn st_mut(&mut self) -> &mut StateInfo {
        self.states.last_mut().expect("state stack never empty")
    }

    pub fn castling_rights(&self) -> CastlingRights {
        self.st().copied.castling_rights
    }

    pub fn en_passant_square(&self) -> Option<Square> {
        self.st().copied.ep_square
    }

    pub fn halfmove_clock(&self) -> u32 {
        self.st().copied.rule50 as u32
    }

    /// Canonical rule50 accessor for search/TT/pruning callers (design/01
    /// §3.3.4 seam). [`Board::halfmove_clock`] is the D1-preserved
    /// FEN-facing alias; both read the same state field.
    pub fn rule50_count(&self) -> u32 {
        self.st().copied.rule50 as u32
    }

    pub fn fullmove_number(&self) -> u32 {
        (1 + (self.game_ply - (self.side_to_move == Color::Black) as i32) / 2) as u32
    }

    /// Halfmoves since the game's start position (doc 04 §3.3).
    pub fn game_ply(&self) -> i32 {
        self.game_ply
    }

    /// The TT-facing position hash: the raw key, rule50-folded near the
    /// 50-move rule (doc 04 §3.4, D9). Repetition detection compares
    /// [`Board::raw_key`] instead.
    pub fn key(&self) -> Key {
        self.adjust_key50(self.st().key)
    }

    /// The raw (unfolded) position hash.
    pub fn raw_key(&self) -> Key {
        self.st().key
    }

    pub fn pawn_key(&self) -> Key {
        self.st().copied.pawn_key
    }

    pub fn material_key(&self) -> Key {
        self.st().copied.material_key
    }

    pub fn minor_key(&self) -> Key {
        self.st().copied.minor_key
    }

    pub fn non_pawn_key(&self, color: Color) -> Key {
        self.st().copied.non_pawn_key[color as usize]
    }

    /// Sum of `PIECE_VALUES` over `color`'s knights, bishops, rooks, queens.
    pub fn non_pawn_material(&self, color: Color) -> Value {
        self.st().copied.non_pawn_material[color as usize]
    }

    /// Enemy pieces currently checking the side to move's king.
    pub fn checkers(&self) -> Bitboard {
        self.st().checkers
    }

    /// Pieces (of BOTH colors) that are the sole piece between an enemy
    /// sniper and `color`'s king (doc 04 §3.5). Friendly ones are pinned;
    /// enemy ones are discovered-check candidates.
    pub fn blockers_for_king(&self, color: Color) -> Bitboard {
        self.st().blockers_for_king[color as usize]
    }

    /// `color`'s sliders pinning a piece of the OTHER color against its own
    /// king (doc 04 §3.5): the snipers whose sole blocker is an enemy piece.
    pub fn pinners(&self, color: Color) -> Bitboard {
        self.st().pinners[color as usize]
    }

    /// From-squares from which a `pt` of the side to move would check the
    /// enemy king (doc 04 §3.5). EMPTY for `King` — a king never gives check.
    pub fn check_squares(&self, pt: PieceType) -> Bitboard {
        self.st().check_squares[pt as usize]
    }

    /// The piece captured by the last made move (None at the root state).
    pub fn captured_piece(&self) -> Option<Piece> {
        self.st().captured
    }

    /// `color`'s king square. Panics on king-less boards — every position
    /// the engine operates on has exactly one king per side.
    pub fn king_square(&self, color: Color) -> Square {
        self.pieces(color, PieceType::King)
            .lsb()
            .expect("king must exist")
    }

    /// Squares that must be empty for `right`'s castling move
    /// (doc 04 §3.7). `right` must be a single right.
    pub fn castling_path(&self, right: CastlingRights) -> Bitboard {
        self.castling_path[right.index()]
    }

    /// Home square of the rook that castles with the single right `right`.
    pub fn castling_rook_square(&self, right: CastlingRights) -> Square {
        self.castling_rook_square[right.index()]
    }

    /// The rights a move touching `sq` revokes: both of a color's rights on
    /// its king's home square, one right on each rook's home square, NONE
    /// elsewhere (doc 04 §3.7 revocation mask).
    pub fn castling_rights_mask(&self, sq: Square) -> CastlingRights {
        self.castling_rights_mask[sq.index() as usize]
    }

    /// All pieces of BOTH colors attacking `sq` under the given occupancy
    /// (doc 04 §3.5) — the shared attack query behind check bookkeeping and,
    /// later, movegen legality and SEE. `occupied` shapes only the slider
    /// rays; the attacker set always comes from the piece bitboards (SEE
    /// exploits exactly that to walk x-ray batteries by lifting captured
    /// pieces out of `occupied`).
    ///
    /// The pawn terms use attack symmetry: the squares a White pawn ON `sq`
    /// would attack are precisely the from-squares of Black pawns attacking
    /// `sq`, and vice versa.
    pub fn attackers_to(&self, sq: Square, occupied: Bitboard) -> Bitboard {
        let tables = attacks::tables();
        let queens = self.pieces_of_type(PieceType::Queen);
        (tables.rook_attacks(sq, occupied) & (self.pieces_of_type(PieceType::Rook) | queens))
            | (tables.bishop_attacks(sq, occupied)
                & (self.pieces_of_type(PieceType::Bishop) | queens))
            | (attacks::pawn_attacks(Color::White, sq) & self.pieces(Color::Black, PieceType::Pawn))
            | (attacks::pawn_attacks(Color::Black, sq) & self.pieces(Color::White, PieceType::Pawn))
            | (attacks::knight_attacks(sq) & self.pieces_of_type(PieceType::Knight))
            | (attacks::king_attacks(sq) & self.pieces_of_type(PieceType::King))
    }

    /// The squares holding `piece_type` pieces of both colors.
    fn pieces_of_type(&self, piece_type: PieceType) -> Bitboard {
        self.pieces[Color::White as usize][piece_type as usize]
            | self.pieces[Color::Black as usize][piece_type as usize]
    }

    /// Slider-pin geometry around `c`'s king (doc 04 §3.5, the SF
    /// `update_slider_blockers` algorithm re-expressed). Returns
    /// `(blockers, pinners)` — a tuple instead of C++'s out-parameter —
    /// where `blockers` is the value for `blockers_for_king[c]` (pieces of
    /// EITHER color standing alone between an enemy sniper and the king)
    /// and `pinners` the value for `pinners[!c]` (the snipers whose sole
    /// blocker is friendly to `c`, i.e. genuinely pinned).
    fn slider_blockers(&self, c: Color) -> (Bitboard, Bitboard) {
        let ksq = self.king_square(c);
        let them = c.flip();
        // Enemy sliders that would hit the king on an empty board.
        let snipers = (attacks::pseudo_attacks(PieceType::Rook, ksq)
            & (self.pieces(them, PieceType::Rook) | self.pieces(them, PieceType::Queen)))
            | (attacks::pseudo_attacks(PieceType::Bishop, ksq)
                & (self.pieces(them, PieceType::Bishop) | self.pieces(them, PieceType::Queen)));
        // Snipers never block each other: lifting them all out of the
        // occupancy lets a queen x-ray through a rook in front of it.
        let occupancy = self.all_occupancy ^ snipers;

        let mut blockers = Bitboard::EMPTY;
        let mut pinners = Bitboard::EMPTY;
        for sniper in snipers {
            // `between` includes its target square, but every sniper was
            // lifted out of `occupancy`, so `b` is exactly the pieces
            // strictly between king and sniper.
            let b = attacks::between(ksq, sniper) & occupancy;
            if !b.is_empty() && !b.more_than_one() {
                blockers |= b;
                if !(b & self.occupancy(c)).is_empty() {
                    pinners |= Bitboard::from_square(sniper);
                }
            }
        }
        (blockers, pinners)
    }

    /// Populate the check bookkeeping of `st` from the CURRENT board
    /// (doc 04 §3.5): blockers/pinners around both kings, plus the squares
    /// from which each piece type of the side to move would check the enemy
    /// king. Called at setup and — once make/unmake land — at the end of
    /// every make/null-move, AFTER `side_to_move` has been flipped. Does
    /// NOT touch `st.checkers`; that is the caller's job (from scratch at
    /// setup, incrementally in `make_move`). Both kings must exist.
    fn set_check_info(&self, st: &mut StateInfo) {
        for c in Color::ALL {
            let (blockers, pinners) = self.slider_blockers(c);
            st.blockers_for_king[c as usize] = blockers;
            st.pinners[c.flip() as usize] = pinners;
        }

        let tables = attacks::tables();
        let them = self.side_to_move.flip();
        let ksq = self.king_square(them);
        let occ = self.all_occupancy;
        // Pawn attack symmetry again: an enemy pawn ON `ksq` attacks exactly
        // the squares from which one of our pawns would check `ksq`.
        st.check_squares[PieceType::Pawn as usize] = attacks::pawn_attacks(them, ksq);
        st.check_squares[PieceType::Knight as usize] = attacks::knight_attacks(ksq);
        st.check_squares[PieceType::Bishop as usize] = tables.bishop_attacks(ksq, occ);
        st.check_squares[PieceType::Rook as usize] = tables.rook_attacks(ksq, occ);
        st.check_squares[PieceType::Queen as usize] = st.check_squares[PieceType::Bishop as usize]
            | st.check_squares[PieceType::Rook as usize];
        st.check_squares[PieceType::King as usize] = Bitboard::EMPTY;
    }

    /// True when each side has at least one king — the precondition for any
    /// check bookkeeping. False only for incomplete boards (`Board::empty`,
    /// mid-construction tests); every position the engine plays on has
    /// exactly one king per side.
    fn has_both_kings(&self) -> bool {
        !self.pieces(Color::White, PieceType::King).is_empty()
            && !self.pieces(Color::Black, PieceType::King).is_empty()
    }

    /// Rule50 folding of the exposed key (doc 04 §3.4).
    fn adjust_key50(&self, k: Key) -> Key {
        zobrist::fold_rule50(k, self.st().copied.rule50)
    }

    /// Register one granted castling right during setup (doc 04 §3.7):
    /// revocation-mask bits on the king and rook home squares, the rook's
    /// origin, the must-be-empty path, and the right itself in the root
    /// state. Standard chess squares today; the kingside/queenside
    /// comparison and the path formula are already Chess960-proof (D10).
    pub(crate) fn set_castling_right(&mut self, color: Color, rook_from: Square) {
        let king_from = Square::E1.relative_to(color);
        let kingside = rook_from.file() > king_from.file();
        let right = match (color, kingside) {
            (Color::White, true) => CastlingRights::WHITE_KINGSIDE,
            (Color::White, false) => CastlingRights::WHITE_QUEENSIDE,
            (Color::Black, true) => CastlingRights::BLACK_KINGSIDE,
            (Color::Black, false) => CastlingRights::BLACK_QUEENSIDE,
        };
        self.castling_rights_mask[king_from.index() as usize].add(right);
        self.castling_rights_mask[rook_from.index() as usize].add(right);
        self.castling_rook_square[right.index()] = rook_from;

        let kto = if kingside { Square::G1 } else { Square::C1 }.relative_to(color);
        let rto = if kingside { Square::F1 } else { Square::D1 }.relative_to(color);
        // Every square the king or rook crosses or lands on, minus the two
        // start squares (attacks::between includes its target square).
        self.castling_path[right.index()] = (attacks::between(rook_from, rto)
            | attacks::between(king_from, kto)
            | Bitboard::from_square(kto)
            | Bitboard::from_square(rto))
            & !(Bitboard::from_square(king_from) | Bitboard::from_square(rook_from));

        self.st_mut().copied.castling_rights.add(right);
    }

    /// Final setup step shared by the FEN path (doc 04 §3.3): stage the
    /// game-state fields into the root state, then recompute every key from
    /// scratch. Castling geometry must already be registered via
    /// [`Board::set_castling_right`].
    pub(crate) fn finish_setup(&mut self, ep_square: Option<Square>, rule50: i32, game_ply: i32) {
        debug_assert!(rule50 >= 0 && game_ply >= 0);
        self.game_ply = game_ply;
        let st = self.st_mut();
        st.copied.ep_square = ep_square;
        st.copied.rule50 = rule50;
        st.copied.plies_from_null = 0;
        self.init_state();
    }

    /// From-scratch analogue of Stockfish's `set_state` (doc 04 §3.3–3.5):
    /// recompute the six keys, non-pawn material, and the check bookkeeping
    /// (checkers/blockers/pinners/check_squares) for the current placement
    /// plus the staged game-state fields (side to move, castling rights, ep
    /// square) and overwrite the top state's derived fields. `make_move`
    /// will maintain the same quantities incrementally; this recompute is
    /// the drift oracle the debug consistency check compares against
    /// (doc 04 §3.12).
    fn init_state(&mut self) {
        let computed = self.computed_keys();
        let mut st = *self.st();
        st.key = computed.key;
        st.copied.pawn_key = computed.pawn_key;
        st.copied.material_key = computed.material_key;
        st.copied.minor_key = computed.minor_key;
        st.copied.non_pawn_key = computed.non_pawn_key;
        st.copied.non_pawn_material = computed.non_pawn_material;
        st.checkers = Bitboard::EMPTY;
        st.blockers_for_king = [Bitboard::EMPTY; 2];
        st.pinners = [Bitboard::EMPTY; 2];
        st.check_squares = [Bitboard::EMPTY; 6];
        st.captured = None;
        st.repetition = 0;
        // Check bookkeeping needs both kings; incomplete boards
        // (`Board::empty()`, mid-construction tests) keep the empty stubs.
        if self.has_both_kings() {
            let stm = self.side_to_move;
            st.checkers = self.attackers_to(self.king_square(stm), self.all_occupancy)
                & self.occupancy(stm.flip());
            self.set_check_info(&mut st);
        }
        *self.st_mut() = st;
    }

    /// Compute all six keys + material from scratch off the placement and
    /// the top state's rights/ep fields (which must be staged already).
    fn computed_keys(&self) -> ComputedKeys {
        let mut key: Key = 0;
        let mut pawn_key = ZOBRIST.no_pawns;
        let mut material_key: Key = 0;
        let mut minor_key: Key = 0;
        let mut non_pawn_key: [Key; 2] = [0; 2];
        let mut non_pawn_material: [Value; 2] = [0; 2];

        for sq in self.all_occupancy() {
            let pc = self.mailbox[sq.index() as usize].expect("occupancy and mailbox in sync");
            let psq = ZOBRIST.psq[pc.index()][sq.index() as usize];
            key ^= psq;
            match pc.piece_type {
                PieceType::Pawn => pawn_key ^= psq,
                pt => {
                    non_pawn_key[pc.color as usize] ^= psq;
                    if pt != PieceType::King {
                        non_pawn_material[pc.color as usize] += PIECE_VALUES[pt as usize];
                    }
                    if matches!(pt, PieceType::Knight | PieceType::Bishop) {
                        minor_key ^= psq;
                    }
                }
            }
        }

        // Material key: XOR of the count slots psq[pc][8 + i], i in
        // 0..count, kings included (doc 04 §3.4 trick 3). Indexed directly
        // rather than via `ZOBRIST.material`: that helper's debug bound
        // models LEGAL piece counts, but from_fen must stay panic-free on
        // well-formed yet illegal placements (e.g. nine white pawns), so
        // absurd counts are capped at the psq row instead.
        for color in Color::ALL {
            for piece_type in PieceType::ALL {
                let row = &ZOBRIST.psq[Piece::new(color, piece_type).index()];
                let count = (self.pieces(color, piece_type).count() as usize).min(row.len() - 8);
                for slot in &row[8..8 + count] {
                    material_key ^= *slot;
                }
            }
        }

        let copied = &self.st().copied;
        if let Some(ep) = copied.ep_square {
            key ^= ZOBRIST.en_passant[ep.file() as usize];
        }
        if self.side_to_move == Color::Black {
            key ^= ZOBRIST.side;
        }
        key ^= ZOBRIST.castling[copied.castling_rights.bits() as usize];

        ComputedKeys {
            key,
            pawn_key,
            material_key,
            minor_key,
            non_pawn_key,
            non_pawn_material,
        }
    }

    /// Test/debug oracle (seed of doc 04 §3.12): recompute every key and
    /// non-pawn material from scratch and assert they match the stored
    /// state. Trivially true for freshly parsed boards; it becomes the
    /// drift detector for every incremental XOR once `make_move` lands.
    #[cfg(any(test, debug_assertions))]
    pub fn assert_state_keys_consistent(&self) {
        let computed = self.computed_keys();
        let st = self.st();
        assert_eq!(st.key, computed.key, "main key drifted");
        assert_eq!(st.copied.pawn_key, computed.pawn_key, "pawn_key drifted");
        assert_eq!(
            st.copied.material_key, computed.material_key,
            "material_key drifted"
        );
        assert_eq!(st.copied.minor_key, computed.minor_key, "minor_key drifted");
        assert_eq!(
            st.copied.non_pawn_key, computed.non_pawn_key,
            "non_pawn_key drifted"
        );
        assert_eq!(
            st.copied.non_pawn_material, computed.non_pawn_material,
            "non_pawn_material drifted"
        );
    }

    /// Place `piece` on the empty square `sq`, keeping bitboards and mailbox in sync.
    pub fn put_piece(&mut self, piece: Piece, sq: Square) {
        debug_assert!(
            self.piece_on(sq).is_none(),
            "put_piece on occupied square {sq}"
        );
        let bb = Bitboard::from_square(sq);
        self.pieces[piece.color as usize][piece.piece_type as usize] |= bb;
        self.color_occupancy[piece.color as usize] |= bb;
        self.all_occupancy |= bb;
        self.mailbox[sq.index() as usize] = Some(piece);
    }

    /// Remove and return the piece on `sq`, keeping bitboards and mailbox in sync.
    pub fn remove_piece(&mut self, sq: Square) -> Option<Piece> {
        let piece = self.mailbox[sq.index() as usize].take()?;
        let bb = Bitboard::from_square(sq);
        self.pieces[piece.color as usize][piece.piece_type as usize] ^= bb;
        self.color_occupancy[piece.color as usize] ^= bb;
        self.all_occupancy ^= bb;
        Some(piece)
    }

    /// Check that the bitboards and mailbox agree. Test-only invariant check.
    #[cfg(test)]
    pub(crate) fn assert_consistent(&self) {
        let mut white = Bitboard::EMPTY;
        let mut black = Bitboard::EMPTY;
        for color in Color::ALL {
            for piece_type in PieceType::ALL {
                let bb = self.pieces(color, piece_type);
                for sq in bb {
                    assert_eq!(
                        self.piece_on(sq),
                        Some(Piece::new(color, piece_type)),
                        "mailbox disagrees with bitboard on {sq}"
                    );
                }
                match color {
                    Color::White => white |= bb,
                    Color::Black => black |= bb,
                }
            }
        }
        assert_eq!(white, self.occupancy(Color::White));
        assert_eq!(black, self.occupancy(Color::Black));
        assert_eq!(white | black, self.all_occupancy());
        assert!((white & black).is_empty());
        let mailbox_count = self.mailbox.iter().filter(|p| p.is_some()).count();
        assert_eq!(mailbox_count as u32, self.all_occupancy().count());

        // State-stack sanity (doc 04 §3.3): never empty, sane counters.
        assert!(!self.states.is_empty(), "state stack must never be empty");
        assert!(self.st().copied.rule50 >= 0);
        assert!(self.st().copied.plies_from_null >= 0);
        assert!(self.game_ply >= 0);
    }
}

impl Default for Board {
    fn default() -> Self {
        Board::empty()
    }
}

impl fmt::Display for Board {
    fn fmt(&self, f: &mut fmt::Formatter<'_>) -> fmt::Result {
        for rank in (0..8).rev() {
            write!(f, "{}", rank + 1)?;
            for file in 0..8 {
                let sq = Square::from_coords(file, rank);
                match self.piece_on(sq) {
                    Some(piece) => write!(f, " {piece}")?,
                    None => write!(f, " .")?,
                }
            }
            writeln!(f)?;
        }
        write!(f, " ")?;
        for file in 0..8u8 {
            write!(f, " {}", (b'a' + file) as char)?;
        }
        Ok(())
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    /// Readable expected sets: `bb(&[Square::E4, Square::D5])`.
    fn bb(squares: &[Square]) -> Bitboard {
        squares.iter().copied().collect()
    }

    #[test]
    fn put_piece_updates_all_layers() {
        let mut board = Board::empty();
        let piece = Piece::new(Color::White, PieceType::Pawn);
        board.put_piece(piece, Square::E2);

        assert_eq!(board.piece_on(Square::E2), Some(piece));
        assert!(
            board
                .pieces(Color::White, PieceType::Pawn)
                .contains(Square::E2)
        );
        assert!(board.occupancy(Color::White).contains(Square::E2));
        assert!(board.all_occupancy().contains(Square::E2));
        assert!(!board.occupancy(Color::Black).contains(Square::E2));
        board.assert_consistent();
    }

    #[test]
    fn remove_piece_reverses_put_piece() {
        let mut board = Board::empty();
        let piece = Piece::new(Color::Black, PieceType::Knight);
        board.put_piece(piece, Square::G8);

        assert_eq!(board.remove_piece(Square::G8), Some(piece));
        assert_eq!(board.piece_on(Square::G8), None);
        assert!(board.all_occupancy().is_empty());
        assert!(board.pieces(Color::Black, PieceType::Knight).is_empty());
        board.assert_consistent();

        // Removing from an empty square is a no-op.
        assert_eq!(board.remove_piece(Square::G8), None);
    }

    #[test]
    fn castling_rights_operations() {
        let mut rights = CastlingRights::NONE;
        assert!(rights.is_empty());

        rights.add(CastlingRights::WHITE_KINGSIDE);
        rights.add(CastlingRights::BLACK_QUEENSIDE);
        assert!(rights.contains(CastlingRights::WHITE_KINGSIDE));
        assert!(rights.contains(CastlingRights::BLACK_QUEENSIDE));
        assert!(!rights.contains(CastlingRights::WHITE_QUEENSIDE));
        assert!(!rights.contains(CastlingRights::ALL));

        rights.remove(CastlingRights::WHITE_KINGSIDE);
        assert!(!rights.contains(CastlingRights::WHITE_KINGSIDE));
        assert!(rights.contains(CastlingRights::BLACK_QUEENSIDE));

        assert!(CastlingRights::ALL.contains(CastlingRights::BLACK_KINGSIDE));
    }

    // Doc 04 §4 step 1: bits() exposes the raw 4-bit mask that indexes the
    // 16-entry zobrist castling table.
    #[test]
    fn castling_rights_bits_roundtrip() {
        assert_eq!(CastlingRights::NONE.bits(), 0b0000);
        assert_eq!(CastlingRights::WHITE_KINGSIDE.bits(), 0b0001);
        assert_eq!(CastlingRights::WHITE_QUEENSIDE.bits(), 0b0010);
        assert_eq!(CastlingRights::BLACK_KINGSIDE.bits(), 0b0100);
        assert_eq!(CastlingRights::BLACK_QUEENSIDE.bits(), 0b1000);
        assert_eq!(CastlingRights::ALL.bits(), 0b1111);

        // Round-trip: every 4-bit mask is reachable by composing single
        // rights, and bits() reads back exactly the mask that was composed.
        let single = [
            CastlingRights::WHITE_KINGSIDE,
            CastlingRights::WHITE_QUEENSIDE,
            CastlingRights::BLACK_KINGSIDE,
            CastlingRights::BLACK_QUEENSIDE,
        ];
        for mask in 0u8..16 {
            let mut rights = CastlingRights::NONE;
            for (bit, &right) in single.iter().enumerate() {
                if mask & (1 << bit) != 0 {
                    rights.add(right);
                }
            }
            assert_eq!(rights.bits(), mask);
        }
    }

    // Doc 04 §3.2: StateInfo stays Copy (make_null_move clones the whole
    // record) and within the ~three-cache-line size budget, so field creep
    // gets noticed.
    #[test]
    fn state_info_is_copy_and_bounded() {
        fn assert_copy<T: Copy>() {}
        assert_copy::<CopiedState>();
        assert_copy::<StateInfo>();
        const { assert!(size_of::<StateInfo>() <= 192) }
    }

    // Doc 04 §4 step 3: the root state exists from construction, the
    // accessors delegate to it, and even the empty board's keys are the
    // documented baselines (castling[0] is a real key; pawn_key starts at
    // no_pawns).
    #[test]
    fn empty_board_has_consistent_root_state() {
        let board = Board::empty();
        board.assert_consistent();
        board.assert_state_keys_consistent();

        assert_eq!(board.castling_rights(), CastlingRights::NONE);
        assert_eq!(board.en_passant_square(), None);
        assert_eq!(board.halfmove_clock(), 0);
        assert_eq!(board.rule50_count(), 0);
        assert_eq!(board.game_ply(), 0);
        assert_eq!(board.fullmove_number(), 1);
        assert_eq!(board.checkers(), Bitboard::EMPTY);
        assert_eq!(board.captured_piece(), None);

        assert_eq!(board.raw_key(), ZOBRIST.castling[0]);
        assert_eq!(board.key(), board.raw_key());
        assert_eq!(board.pawn_key(), ZOBRIST.no_pawns);
        assert_eq!(board.material_key(), 0);
        assert_eq!(board.minor_key(), 0);
        for color in Color::ALL {
            assert_eq!(board.non_pawn_key(color), 0);
            assert_eq!(board.non_pawn_material(color), 0);
            assert_eq!(board.blockers_for_king(color), Bitboard::EMPTY);
            assert_eq!(board.pinners(color), Bitboard::EMPTY);
        }
        for pt in PieceType::ALL {
            assert_eq!(board.check_squares(pt), Bitboard::EMPTY);
        }
    }

    #[test]
    fn king_square_finds_kings() {
        let board = Board::startpos();
        assert_eq!(board.king_square(Color::White), Square::E1);
        assert_eq!(board.king_square(Color::Black), Square::E8);
    }

    // D8: Board::clone clones the state history — per-thread copies are
    // fully independent.
    #[test]
    fn clone_is_deep_and_equal() {
        let board = Board::startpos();
        let mut clone = board.clone();
        assert_eq!(clone, board);
        assert_eq!(clone.key(), board.key());

        // Diverge the clone; the original must be untouched.
        clone.remove_piece(Square::E2);
        assert_ne!(clone, board);
        assert_eq!(
            board.piece_on(Square::E2),
            Some(Piece::new(Color::White, PieceType::Pawn))
        );
        board.assert_consistent();
        clone.assert_consistent();
    }

    // Doc 04 §3.7: castling geometry (rook squares, empty-path sets,
    // revocation masks) registered at FEN setup.
    #[test]
    fn castling_geometry_from_fen() {
        use CastlingRights as CR;
        let board = Board::startpos();

        assert_eq!(board.castling_rook_square(CR::WHITE_KINGSIDE), Square::H1);
        assert_eq!(board.castling_rook_square(CR::WHITE_QUEENSIDE), Square::A1);
        assert_eq!(board.castling_rook_square(CR::BLACK_KINGSIDE), Square::H8);
        assert_eq!(board.castling_rook_square(CR::BLACK_QUEENSIDE), Square::A8);

        let bb = |squares: &[Square]| squares.iter().copied().collect::<Bitboard>();
        assert_eq!(
            board.castling_path(CR::WHITE_KINGSIDE),
            bb(&[Square::F1, Square::G1])
        );
        assert_eq!(
            board.castling_path(CR::WHITE_QUEENSIDE),
            bb(&[Square::B1, Square::C1, Square::D1])
        );
        assert_eq!(
            board.castling_path(CR::BLACK_KINGSIDE),
            bb(&[Square::F8, Square::G8])
        );
        assert_eq!(
            board.castling_path(CR::BLACK_QUEENSIDE),
            bb(&[Square::B8, Square::C8, Square::D8])
        );

        // Revocation masks: both of a color's rights on its king square,
        // one on each rook home square, NONE anywhere else.
        let mut white_both = CR::WHITE_KINGSIDE;
        white_both.add(CR::WHITE_QUEENSIDE);
        let mut black_both = CR::BLACK_KINGSIDE;
        black_both.add(CR::BLACK_QUEENSIDE);
        assert_eq!(board.castling_rights_mask(Square::E1), white_both);
        assert_eq!(board.castling_rights_mask(Square::E8), black_both);
        assert_eq!(board.castling_rights_mask(Square::H1), CR::WHITE_KINGSIDE);
        assert_eq!(board.castling_rights_mask(Square::A1), CR::WHITE_QUEENSIDE);
        assert_eq!(board.castling_rights_mask(Square::H8), CR::BLACK_KINGSIDE);
        assert_eq!(board.castling_rights_mask(Square::A8), CR::BLACK_QUEENSIDE);
        assert_eq!(board.castling_rights_mask(Square::E4), CR::NONE);
        assert_eq!(board.castling_rights_mask(Square::B1), CR::NONE);

        // Partial rights register only their own geometry.
        let partial = Board::from_fen("r3k2r/8/8/8/8/8/8/R3K2R w Kq - 0 1").unwrap();
        assert_eq!(partial.castling_rights_mask(Square::E1), CR::WHITE_KINGSIDE);
        assert_eq!(partial.castling_rights_mask(Square::A1), CR::NONE);
        assert_eq!(
            partial.castling_rights_mask(Square::E8),
            CR::BLACK_QUEENSIDE
        );
        assert_eq!(partial.castling_rights_mask(Square::H8), CR::NONE);
    }

    // CastlingRights::index: bit positions for the geometry arrays
    // (doc 04 §3.7: WK 0, WQ 1, BK 2, BQ 3).
    #[test]
    fn castling_rights_single_bit_index() {
        assert_eq!(CastlingRights::WHITE_KINGSIDE.index(), 0);
        assert_eq!(CastlingRights::WHITE_QUEENSIDE.index(), 1);
        assert_eq!(CastlingRights::BLACK_KINGSIDE.index(), 2);
        assert_eq!(CastlingRights::BLACK_QUEENSIDE.index(), 3);
    }

    // -----------------------------------------------------------------
    // Doc 04 §4 step 4: attackers_to, slider_blockers, set_check_info
    // -----------------------------------------------------------------

    // One busy square attacked by every piece type of both colors: the
    // rook/queen and bishop rays stop at (and include) the first occupied
    // square, the leapers ignore occupancy entirely.
    #[test]
    fn attackers_to_sees_every_piece_type() {
        let board = Board::from_fen("4r3/8/2nk1p2/7Q/5P2/5N2/1B6/4K3 w - - 0 1").unwrap();
        assert_eq!(
            board.attackers_to(Square::E5, board.all_occupancy()),
            bb(&[
                Square::E8, // black rook down the open e-file
                Square::H5, // white queen along rank 5
                Square::B2, // white bishop, b2-c3-d4 all empty
                Square::F6, // black pawn
                Square::F4, // white pawn
                Square::C6, // black knight
                Square::F3, // white knight
                Square::D6, // black king
            ])
        );
        // The same rook also checks the white king: init_state derived it.
        assert_eq!(board.checkers(), bb(&[Square::E8]));
    }

    // The `occupied` parameter shapes only the slider rays; lifting a
    // blocker out exposes the x-ray piece behind it while the piece
    // bitboards still supply both attackers (the SEE battery walk).
    #[test]
    fn attackers_to_exposes_xrays_under_reduced_occupancy() {
        let board = Board::from_fen("4k3/8/8/8/4R3/8/8/4R2K b - - 0 1").unwrap();
        let all = board.all_occupancy();
        assert_eq!(board.attackers_to(Square::E8, all), bb(&[Square::E4]));
        assert_eq!(
            board.attackers_to(Square::E8, all ^ Bitboard::from_square(Square::E4)),
            bb(&[Square::E4, Square::E1])
        );
        // Black to move and in check from the front rook only.
        assert_eq!(board.checkers(), bb(&[Square::E4]));
    }

    #[test]
    fn checkers_detects_double_check() {
        // Corpus classic: knight on d6 and rook on e1 both hit e8.
        let board = Board::from_fen("4k3/8/3N4/8/8/8/8/4RK2 b - - 0 1").unwrap();
        assert_eq!(board.checkers(), bb(&[Square::D6, Square::E1]));
        assert!(Board::startpos().checkers().is_empty());
    }

    // Friendly blocker: the pinned piece lands in blockers_for_king(c) and
    // the sniper in pinners(!c).
    #[test]
    fn slider_blockers_vertical_pin() {
        let board = Board::from_fen("4r2k/8/8/8/8/8/4N3/4K3 w - - 0 1").unwrap();
        assert_eq!(board.blockers_for_king(Color::White), bb(&[Square::E2]));
        assert_eq!(board.pinners(Color::Black), bb(&[Square::E8]));
        assert_eq!(board.blockers_for_king(Color::Black), Bitboard::EMPTY);
        assert_eq!(board.pinners(Color::White), Bitboard::EMPTY);
        assert!(board.checkers().is_empty());
    }

    #[test]
    fn slider_blockers_diagonal_pin() {
        let board = Board::from_fen("7k/8/8/8/7b/8/5P2/4K3 w - - 0 1").unwrap();
        assert_eq!(board.blockers_for_king(Color::White), bb(&[Square::F2]));
        assert_eq!(board.pinners(Color::Black), bb(&[Square::H4]));
    }

    // Enemy blocker: a discovered-check candidate — it lands in
    // blockers_for_king (blockers hold pieces of BOTH colors) but its
    // sniper is NOT a pinner (the blocker is not friendly to the king).
    #[test]
    fn slider_blockers_discovered_check_candidate() {
        let board = Board::from_fen("4r2k/8/8/4n3/8/8/8/4K3 w - - 0 1").unwrap();
        assert_eq!(board.blockers_for_king(Color::White), bb(&[Square::E5]));
        assert_eq!(board.pinners(Color::Black), Bitboard::EMPTY);
    }

    // Two pieces between king and sniper: neither is a blocker.
    #[test]
    fn slider_blockers_need_a_sole_blocker() {
        let board = Board::from_fen("4r2k/8/8/4n3/8/4N3/8/4K3 w - - 0 1").unwrap();
        assert_eq!(board.blockers_for_king(Color::White), Bitboard::EMPTY);
        assert_eq!(board.pinners(Color::Black), Bitboard::EMPTY);
    }

    // Stacked snipers (queen behind rook): snipers are lifted out of the
    // occupancy together, so BOTH register as pinners of the sole blocker —
    // the SF convention this implementation pins down.
    #[test]
    fn slider_blockers_xray_through_stacked_snipers() {
        let board = Board::from_fen("4q2k/4r3/8/8/8/8/4B3/4K3 w - - 0 1").unwrap();
        assert_eq!(board.blockers_for_king(Color::White), bb(&[Square::E2]));
        assert_eq!(board.pinners(Color::Black), bb(&[Square::E7, Square::E8]));
    }

    // The mirror orientation: a white sniper pinning a black piece fills
    // blockers_for_king(Black) and pinners(White).
    #[test]
    fn slider_blockers_white_pins_black() {
        let board = Board::from_fen("4k3/8/4n3/8/8/8/4Q3/6K1 b - - 0 1").unwrap();
        assert_eq!(board.blockers_for_king(Color::Black), bb(&[Square::E6]));
        assert_eq!(board.pinners(Color::White), bb(&[Square::E2]));
        assert!(board.checkers().is_empty());
    }

    // check_squares: from-squares that would check the ENEMY king, per
    // piece type of the side to move, under the current occupancy.
    #[test]
    fn check_squares_on_startpos() {
        let board = Board::startpos(); // white to move, target king on e8
        assert_eq!(
            board.check_squares(PieceType::Pawn),
            bb(&[Square::D7, Square::F7])
        );
        assert_eq!(
            board.check_squares(PieceType::Knight),
            bb(&[Square::C7, Square::D6, Square::F6, Square::G7])
        );
        // Sliders stop at (and include) the first occupied square.
        assert_eq!(
            board.check_squares(PieceType::Bishop),
            bb(&[Square::D7, Square::F7])
        );
        assert_eq!(
            board.check_squares(PieceType::Rook),
            bb(&[Square::D8, Square::E7, Square::F8])
        );
        assert_eq!(
            board.check_squares(PieceType::Queen),
            board.check_squares(PieceType::Bishop) | board.check_squares(PieceType::Rook)
        );
        assert_eq!(board.check_squares(PieceType::King), Bitboard::EMPTY);
    }

    // Orientation: with Black to move the target is the WHITE king, and the
    // pawn entry uses white-pawn attack symmetry (a black pawn on d2 or f2
    // checks e1).
    #[test]
    fn check_squares_orientation_for_black() {
        let board = Board::from_fen("4k3/8/8/8/8/8/8/4K3 b - - 0 1").unwrap();
        assert_eq!(
            board.check_squares(PieceType::Pawn),
            bb(&[Square::D2, Square::F2])
        );
        assert_eq!(
            board.check_squares(PieceType::Knight),
            bb(&[Square::C2, Square::D3, Square::F3, Square::G2])
        );
    }

    #[test]
    fn castling_rights_display_parse_roundtrip() {
        for s in ["-", "K", "Qk", "KQkq", "Kq", "kq"] {
            let rights: CastlingRights = s.parse().unwrap();
            assert_eq!(rights.to_string(), s);
        }
        assert!("".parse::<CastlingRights>().is_err());
        assert!("KK".parse::<CastlingRights>().is_err());
        assert!("KQx".parse::<CastlingRights>().is_err());
    }

    #[test]
    fn display_empty_board() {
        let board = Board::empty();
        let expected = "\
8 . . . . . . . .
7 . . . . . . . .
6 . . . . . . . .
5 . . . . . . . .
4 . . . . . . . .
3 . . . . . . . .
2 . . . . . . . .
1 . . . . . . . .
  a b c d e f g h";
        assert_eq!(board.to_string(), expected);
    }

    #[test]
    fn display_shows_pieces_at_correct_coordinates() {
        let mut board = Board::empty();
        board.put_piece(Piece::new(Color::White, PieceType::King), Square::E1);
        board.put_piece(Piece::new(Color::Black, PieceType::King), Square::E8);
        board.put_piece(Piece::new(Color::White, PieceType::Queen), Square::A4);

        let expected = "\
8 . . . . k . . .
7 . . . . . . . .
6 . . . . . . . .
5 . . . . . . . .
4 Q . . . . . . .
3 . . . . . . . .
2 . . . . . . . .
1 . . . . K . . .
  a b c d e f g h";
        assert_eq!(board.to_string(), expected);
    }
}
