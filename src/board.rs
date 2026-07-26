use crate::attacks;
use crate::bitboard::Bitboard;
use crate::moves::{Move, MoveKind};
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

    /// The single right of `color` on the given wing — setup and make/unmake
    /// key the castling geometry arrays off this (design/04 §3.7).
    pub(crate) const fn single(color: Color, kingside: bool) -> CastlingRights {
        match (color, kingside) {
            (Color::White, true) => CastlingRights::WHITE_KINGSIDE,
            (Color::White, false) => CastlingRights::WHITE_QUEENSIDE,
            (Color::Black, true) => CastlingRights::BLACK_KINGSIDE,
            (Color::Black, false) => CastlingRights::BLACK_QUEENSIDE,
        }
    }

    /// `color`'s kingside right — movegen's castling generator keys off this
    /// (design/05-movegen.md §3.7).
    pub const fn kingside(color: Color) -> CastlingRights {
        CastlingRights::single(color, true)
    }

    /// `color`'s queenside right (design/05-movegen.md §3.7).
    pub const fn queenside(color: Color) -> CastlingRights {
        CastlingRights::single(color, false)
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
    /// Wave-G threat map (D30): every square attacked by the side-NOT-to-move,
    /// unioned (`all_threats`) and split by attacker `PieceType`
    /// (`piece_threats[pt]`). Whole-army setwise fills with the side-to-move's
    /// king removed from slider occupancy (sliders see through it). Recomputed
    /// per state; read by threat-aware move ordering (§3.4/§3.7) and the
    /// threat-coupled RFP margin (design/10 §3.4).
    pub all_threats: Bitboard,
    pub piece_threats: [Bitboard; 6],
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
        all_threats: Bitboard::EMPTY,
        piece_threats: [Bitboard::EMPTY; 6],
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

/// The piece-placement deltas of a single move, the payload the NNUE
/// accumulator consumes to update its feature columns incrementally
/// (design/11 §3.4.4). A move touches at most two "adds" (a piece appearing
/// on a square) and two "subs" (a piece leaving a square):
///
/// - normal quiet: 1 sub + 1 add (the mover);
/// - capture (incl. en passant): 2 subs (mover + victim) + 1 add;
/// - promotion: sub the pawn (+ sub a captured victim) + add the promoted
///   piece;
/// - castling: 2 subs + 2 adds (king and rook);
/// - null move: empty (0 adds, 0 subs).
///
/// **Layering note (design/11 §3.4.4, adapted).** The design sketch places
/// `DirtyPiece` under `src/eval/nnue/`, but its data is a pure
/// board-mutation fact and [`DirtyPiece::from_move`] must read `Board`'s
/// private castling geometry, so it lives HERE (the low module) and
/// `eval::nnue` re-exports it. That keeps the dependency arrow pointing
/// `eval → board` only — the foundational `board` module never reaches up
/// into `eval`. The fields are private; iterate via [`DirtyPiece::adds`] /
/// [`DirtyPiece::subs`] (behavior, not representation — CLAUDE.md).
#[derive(Debug, Clone, Copy, PartialEq, Eq)]
pub struct DirtyPiece {
    adds: [(Piece, Square); 2],
    subs: [(Piece, Square); 2],
    n_add: u8,
    n_sub: u8,
}

impl Default for DirtyPiece {
    fn default() -> Self {
        Self::EMPTY
    }
}

impl DirtyPiece {
    /// A null record (the null-move case): no pieces added or removed. The
    /// slot placeholder is never read — [`DirtyPiece::adds`]/[`subs`] bound
    /// their slices by `n_add`/`n_sub`.
    pub const EMPTY: DirtyPiece = DirtyPiece {
        adds: [(Piece::new(Color::White, PieceType::Pawn), Square::A1); 2],
        subs: [(Piece::new(Color::White, PieceType::Pawn), Square::A1); 2],
        n_add: 0,
        n_sub: 0,
    };

    /// The (piece, square) pairs a piece appears on after this move.
    #[inline]
    pub fn adds(&self) -> &[(Piece, Square)] {
        &self.adds[..self.n_add as usize]
    }

    /// The (piece, square) pairs a piece leaves after this move.
    #[inline]
    pub fn subs(&self) -> &[(Piece, Square)] {
        &self.subs[..self.n_sub as usize]
    }

    #[inline]
    fn add(&mut self, piece: Piece, sq: Square) {
        self.adds[self.n_add as usize] = (piece, sq);
        self.n_add += 1;
    }

    #[inline]
    fn sub(&mut self, piece: Piece, sq: Square) {
        self.subs[self.n_sub as usize] = (piece, sq);
        self.n_sub += 1;
    }

    /// Derive the placement deltas of `m` from the position `board` it is
    /// made in (design/11 §3.4.4). `m` must be legal for `board`'s side to
    /// move. This mirrors the mutation cases of [`Board::make_move_with`]
    /// exactly, so the NNUE accumulator stays in lockstep with the board.
    ///
    /// A null / same-square sentinel yields [`DirtyPiece::EMPTY`].
    pub fn from_move(board: &Board, m: Move) -> DirtyPiece {
        let mut dp = DirtyPiece::EMPTY;
        if !m.is_ok() {
            return dp; // null move: empty record
        }
        let us = board.side_to_move;
        let them = us.flip();
        let (from, to) = (m.from(), m.to());
        let mover = board
            .piece_on(from)
            .expect("DirtyPiece::from_move: no piece on the from-square");

        match m.kind() {
            MoveKind::Castling => {
                // King to G1/C1 (D2); the rook jumps to F1/D1. Same geometry
                // as make_move_with step 4.
                let kingside = to.file() > from.file();
                let rook_from =
                    board.castling_rook_square[CastlingRights::single(us, kingside).index()];
                let rook_to = if kingside { Square::F1 } else { Square::D1 }.relative_to(us);
                let rook = Piece::new(us, PieceType::Rook);
                dp.sub(mover, from);
                dp.sub(rook, rook_from);
                dp.add(mover, to);
                dp.add(rook, rook_to);
            }
            MoveKind::EnPassant => {
                // The captured pawn sits one rank from `to`, toward our home.
                let capsq = to.offset(them.pawn_push());
                dp.sub(mover, from);
                dp.sub(Piece::new(them, PieceType::Pawn), capsq);
                dp.add(mover, to);
            }
            MoveKind::Promotion => {
                let promoted = Piece::new(
                    us,
                    m.promotion_piece().expect("promotion kind carries a piece"),
                );
                dp.sub(mover, from);
                if let Some(victim) = board.piece_on(to) {
                    dp.sub(victim, to); // capture-promotion
                }
                dp.add(promoted, to);
            }
            MoveKind::Normal => {
                dp.sub(mover, from);
                if let Some(victim) = board.piece_on(to) {
                    dp.sub(victim, to); // ordinary capture
                }
                dp.add(mover, to);
            }
        }
        dp
    }
}

/// Hook points inside [`Board::make_move_with`] (doc 04 §3.10). D10 pins the
/// public `make_move(m, gives_check)` signature, so prefetch targets can't be
/// parameters — the TT (and later the history tables and NNUE dirty-piece
/// recording, D21) attach through this trait instead. It is monomorphized per
/// hook type and the default bodies are empty `#[inline]` no-ops, so the
/// [`NoHooks`] path compiles away entirely: the seam is bench-neutral by
/// construction.
pub trait MakeMoveHooks {
    /// Hook A: the child's full key is final (rule50-folded — the TT probe
    /// key) while board mutation is still in flight. The TT prefetches its
    /// cluster here.
    #[inline]
    fn on_key(&mut self, _key: Key) {}

    /// Hook B: the child's specialized keys are final. History/correction
    /// tables prefetch their entries here.
    #[inline]
    fn on_state(&mut self, _copied: &CopiedState) {}

    /// Hook C (design/11 §3.4.4): the move's placement deltas, for the NNUE
    /// accumulator to fold into its feature columns. Default no-op, so
    /// [`NoHooks`] (hence `make_move`, perft, and the classical search path)
    /// stays bit-identical — the accumulator layer (P13 step 4) is the only
    /// caller that will supply a real body. NOT invoked from
    /// `make_move_with` yet: computing the payload has a panic-carrying
    /// `expect`, so a speculative unconditional call could perturb NoHooks
    /// codegen; the call site lands with the search wiring in step 4.
    #[inline]
    fn on_dirty(&mut self, _dp: &DirtyPiece) {}
}

/// The default no-op hooks: [`Board::make_move`] delegates through this, and
/// the empty inline bodies monomorphize to nothing.
pub struct NoHooks;

impl MakeMoveHooks for NoHooks {}

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

    /// Plies since the last null move (design/10 §3.6 `is_shuffling`): reset to
    /// 0 by [`Board::make_null_move`], incremented by every real move.
    pub fn plies_from_null(&self) -> i32 {
        self.st().copied.plies_from_null
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

    /// Every square attacked by the side NOT to move (Wave-G threat map, D30),
    /// with the side-to-move's king x-rayed out of slider occupancy.
    pub fn all_threats(&self) -> Bitboard {
        self.st().all_threats
    }

    /// Squares attacked by the side-not-to-move's pieces of type `pt`
    /// (Wave-G threat map, D30). `piece_threats(King)` is the enemy king's
    /// attacks; kings/pawns have no cheaper attacker so the ordering scorer
    /// unions the relevant tiers itself (design/09 §3.7).
    pub fn piece_threats(&self, pt: PieceType) -> Bitboard {
        self.st().piece_threats[pt as usize]
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

        self.set_threats(st);
    }

    /// Populate the Wave-G threat map (D30): squares attacked by `them` (the
    /// side NOT to move), by piece type and unioned. Whole-army setwise fills;
    /// the side-to-move's own king is removed from slider occupancy so a
    /// slider's threat extends THROUGH it (the king cannot shield the square
    /// behind it — it must move). Called from [`set_check_info`], so it is
    /// maintained once per make/unmake/setup, AFTER `side_to_move` is flipped.
    fn set_threats(&self, st: &mut StateInfo) {
        let tables = attacks::tables();
        let us = self.side_to_move;
        let them = us.flip();
        // Sliders see through our king (king-danger x-ray).
        let slider_occ = self.all_occupancy ^ Bitboard::from_square(self.king_square(us));

        let pawns = attacks::pawn_attacks_bb(them, self.pieces(them, PieceType::Pawn));

        let mut knights = Bitboard::EMPTY;
        for sq in self.pieces(them, PieceType::Knight) {
            knights |= attacks::knight_attacks(sq);
        }
        let mut bishops = Bitboard::EMPTY;
        for sq in self.pieces(them, PieceType::Bishop) {
            bishops |= tables.bishop_attacks(sq, slider_occ);
        }
        let mut rooks = Bitboard::EMPTY;
        for sq in self.pieces(them, PieceType::Rook) {
            rooks |= tables.rook_attacks(sq, slider_occ);
        }
        let mut queens = Bitboard::EMPTY;
        for sq in self.pieces(them, PieceType::Queen) {
            queens |= tables.queen_attacks(sq, slider_occ);
        }
        let king = attacks::king_attacks(self.king_square(them));

        st.piece_threats[PieceType::Pawn as usize] = pawns;
        st.piece_threats[PieceType::Knight as usize] = knights;
        st.piece_threats[PieceType::Bishop as usize] = bishops;
        st.piece_threats[PieceType::Rook as usize] = rooks;
        st.piece_threats[PieceType::Queen as usize] = queens;
        st.piece_threats[PieceType::King as usize] = king;
        st.all_threats = pawns | knights | bishops | rooks | queens | king;
    }

    /// Does this pseudo-legal move of the side to move give check?
    /// O(1) for the common cases via `check_squares` and
    /// `blockers_for_king` (doc 04 §3.6) — search calls this once per move
    /// and feeds the answer to [`Board::make_move`].
    pub fn gives_check(&self, m: Move) -> bool {
        let us = self.side_to_move;
        let them = us.flip();
        let (from, to) = (m.from(), m.to());
        let pc = self
            .piece_on(from)
            .expect("gives_check: no piece on the from-square");
        debug_assert_eq!(pc.color, us, "gives_check: moving an enemy piece");
        let ksq = self.king_square(them);

        // Direct check: `to` is a square from which `pc`'s type checks the
        // enemy king.
        if self.check_squares(pc.piece_type).contains(to) {
            return true;
        }

        // Discovered check: the mover blocks a slider aimed at the ENEMY
        // king and steps off the king line. Castling is always a check
        // here — the king vacates a blocker square while the rook lands
        // with tempo.
        if self.blockers_for_king(them).contains(from)
            && (!attacks::aligned(from, to, ksq) || m.kind() == MoveKind::Castling)
        {
            return true;
        }

        match m.kind() {
            MoveKind::Normal => false,
            // Does the promoted piece attack the king from `to`, with the
            // pawn already gone from `from`?
            MoveKind::Promotion => {
                let promo = m.promotion_piece().expect("promotion kind has a piece");
                let occupied = self.all_occupancy ^ Bitboard::from_square(from);
                attacks::attacks(promo, to, occupied).contains(ksq)
            }
            // The one case the fast paths miss: discovered check through
            // the CAPTURED pawn's square.
            MoveKind::EnPassant => {
                let capsq = Square::from_coords(to.file(), from.rank());
                let occupied = (self.all_occupancy
                    ^ Bitboard::from_square(from)
                    ^ Bitboard::from_square(capsq))
                    | Bitboard::from_square(to);
                let tables = attacks::tables();
                let queens = self.pieces(us, PieceType::Queen);
                !((tables.rook_attacks(ksq, occupied)
                    & (self.pieces(us, PieceType::Rook) | queens))
                    | (tables.bishop_attacks(ksq, occupied)
                        & (self.pieces(us, PieceType::Bishop) | queens)))
                    .is_empty()
            }
            // Does the castling rook check from its destination?
            MoveKind::Castling => {
                let kingside = to.file() > from.file();
                let rto = if kingside { Square::F1 } else { Square::D1 }.relative_to(us);
                self.check_squares(PieceType::Rook).contains(rto)
            }
        }
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
        let right = CastlingRights::single(color, kingside);
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

    /// Move a piece between squares in one pass: from-bit ^ to-bit on three
    /// bitboards plus two mailbox writes (doc 04 §3.3). The internal fast
    /// path of make/unmake — the destination must be empty (captures and
    /// promotions go through `remove_piece`/`put_piece` instead). Placement-
    /// only like its two siblings: Zobrist deltas live in `make_move`, not
    /// here (doc 04 §9.1).
    pub(crate) fn move_piece(&mut self, from: Square, to: Square) {
        debug_assert!(
            self.piece_on(to).is_none(),
            "move_piece to occupied square {to}"
        );
        let pc = self.mailbox[from.index() as usize]
            .take()
            .expect("move_piece from empty square");
        let from_to = Bitboard::from_square(from) | Bitboard::from_square(to);
        self.pieces[pc.color as usize][pc.piece_type as usize] ^= from_to;
        self.color_occupancy[pc.color as usize] ^= from_to;
        self.all_occupancy ^= from_to;
        self.mailbox[to.index() as usize] = Some(pc);
    }

    /// Make `m` on the board (doc 04 §3.8, D10). `m` must be legal for the
    /// side to move, and `gives_check` must equal `self.gives_check(m)` —
    /// caller-supplied so quiet moves need no attack query here.
    ///
    /// Builds the child `StateInfo` and pushes it; [`Board::unmake_move`]
    /// pops it. All Zobrist deltas — main key plus the four specialized
    /// families — are applied here in Stockfish's `do_move` order, NOT in
    /// the placement mutators (doc 04 §3.3, §9.1). In particular the new ep
    /// square is set (and hashed) ONLY when an ep capture will actually be
    /// legal (D9).
    pub fn make_move(&mut self, m: Move, gives_check: bool) {
        self.make_move_with(m, gives_check, &mut NoHooks);
    }

    /// [`Board::make_move`] with prefetch/NNUE hook points (doc 04 §3.10).
    /// Same contract; `hooks` receives the child keys mid-make so consumers
    /// can overlap memory prefetches with the rest of the move. The
    /// [`NoHooks`] instantiation is exactly `make_move`.
    pub fn make_move_with<H: MakeMoveHooks>(&mut self, m: Move, gives_check: bool, hooks: &mut H) {
        // Step 1: derive the child key from the parent's, and copy the
        // carried-over prefix block — this assignment IS Stockfish's
        // "memcpy up to key" (D8). The push happens at the very end, so
        // `self.st()` refers to the PREVIOUS state throughout; its blockers
        // feed the ep-legality test in step 8, so grab them now.
        // Perf study #2 option E: construct the child IN PLACE. The states
        // Vec is taken out (a 24-byte pointer swap) so a &mut into its new
        // tail can coexist with the &self/&mut self board calls below —
        // none of which touch `states` (audited; the one states read, the
        // step-13 repetition walk, runs against this same local vec after
        // NLL ends the &mut). The old shape built a 224-byte stack temp and
        // memcpy'd it at the end-push (2.30% of ALL cycles, study §5.4).
        // Panic-safety: a panic mid-make leaves `self.states` empty — but a
        // make_move panic is already a fatal invariant violation.
        let mut states = std::mem::take(&mut self.states);
        let (mut k, prev_blockers) = {
            let prev = states.last().expect("state stack never empty");
            (prev.key ^ ZOBRIST.side, prev.blockers_for_king)
        };
        {
            let prev_copied = states.last().expect("state stack never empty").copied;
            states.push(StateInfo {
                copied: prev_copied,
                ..StateInfo::EMPTY
            });
        }
        let child_idx = states.len() - 1;
        let st = &mut states[child_idx];

        // Step 2: counters.
        self.game_ply += 1;
        st.copied.rule50 += 1;
        st.copied.plies_from_null += 1;

        // Step 3: actors. Our castling encoding is king-to-G1/C1 (D2), not
        // SF's king-captures-rook, so `piece_on(to)` is None for castling
        // and `captured` needs no special-casing.
        let us = self.side_to_move;
        let them = us.flip();
        let (from, to) = (m.from(), m.to());
        let pc = self
            .piece_on(from)
            .expect("make_move: no piece on the from-square");
        debug_assert_eq!(pc.color, us, "make_move: moving an enemy piece");
        let (captured, capsq) = if m.kind() == MoveKind::EnPassant {
            // The captured pawn sits one rank from `to`, toward our home.
            (
                Some(Piece::new(them, PieceType::Pawn)),
                to.offset(them.pawn_push()),
            )
        } else {
            (self.piece_on(to), to)
        };
        debug_assert!(
            captured.is_none_or(|cap| cap.piece_type != PieceType::King),
            "make_move: capturing a king"
        );

        // Step 4: castling — board update AND rook keys happen here; step
        // 10 skips castling entirely. Remove both, then place both: with
        // standard squares nothing overlaps, but the shape is ready for
        // Chess960's overlapping king/rook squares (D10).
        if m.kind() == MoveKind::Castling {
            let kingside = to.file() > from.file();
            let right = CastlingRights::single(us, kingside);
            let rfrom = self.castling_rook_square[right.index()];
            let rto = if kingside { Square::F1 } else { Square::D1 }.relative_to(us);
            let rook = Piece::new(us, PieceType::Rook);

            self.remove_piece(from);
            self.remove_piece(rfrom);
            self.put_piece(pc, to);
            self.put_piece(rook, rto);

            let rook_delta = ZOBRIST.psq[rook.index()][rfrom.index() as usize]
                ^ ZOBRIST.psq[rook.index()][rto.index() as usize];
            k ^= rook_delta;
            st.copied.non_pawn_key[us as usize] ^= rook_delta;
        }

        // Step 5: capture (mutually exclusive with castling).
        if let Some(cap) = captured {
            let cap_psq = ZOBRIST.psq[cap.index()][capsq.index() as usize];
            if cap.piece_type == PieceType::Pawn {
                st.copied.pawn_key ^= cap_psq;
            } else {
                st.copied.non_pawn_material[them as usize] -= PIECE_VALUES[cap.piece_type as usize];
                st.copied.non_pawn_key[them as usize] ^= cap_psq;
                if matches!(cap.piece_type, PieceType::Knight | PieceType::Bishop) {
                    st.copied.minor_key ^= cap_psq;
                }
            }
            k ^= cap_psq;
            // Count-slot removal: slot index = post-removal count. The
            // popcount is still pre-mutation — ALL board edits wait until
            // step 10 (unlike SF, which lifts the ep pawn early for
            // piece-list reasons we don't have).
            let count = self.pieces(them, cap.piece_type).count() as usize;
            st.copied.material_key ^= ZOBRIST.material(cap, count - 1);
            st.copied.rule50 = 0;
        }

        // Step 6: mover key — for castling this is the king pair; for
        // promotions the pawn's `to` key is zero by construction, so no
        // special case (D9). Then clear a stale ep square out of the key.
        k ^= ZOBRIST.psq[pc.index()][from.index() as usize]
            ^ ZOBRIST.psq[pc.index()][to.index() as usize];
        if let Some(ep) = st.copied.ep_square.take() {
            k ^= ZOBRIST.en_passant[ep.file() as usize];
        }

        // Step 7: castling-rights revocation. One mask lookup covers king
        // moves, rook moves, and rook captures (doc 04 §3.7); the whole-mask
        // castling key makes any change two XORs (a no-op change cancels).
        let mut touched = self.castling_rights_mask[from.index() as usize];
        touched.add(self.castling_rights_mask[to.index() as usize]);
        k ^= ZOBRIST.castling[st.copied.castling_rights.bits() as usize];
        st.copied.castling_rights.remove(touched);
        k ^= ZOBRIST.castling[st.copied.castling_rights.bits() as usize];

        // Step 8: pawn extras, or the non-pawn specialized-key upkeep.
        if pc.piece_type == PieceType::Pawn {
            if from.index() ^ to.index() == 16 {
                // Double push: set the ep square ONLY if an ep capture will
                // actually be legal (D9), via the cheap exact test on the
                // PREVIOUS state's blockers.
                let ep = to.offset(them.pawn_push());
                let pawns = attacks::pawn_attacks(us, ep) & self.pieces(them, PieceType::Pawn);
                if !pawns.is_empty() {
                    let ksq = self.king_square(them);
                    let not_blockers = !prev_blockers[them as usize];
                    // If the double push itself delivers discovered check,
                    // ep can never be the legal reply.
                    let no_discovery = not_blockers.contains(from) || from.file() == ksq.file();
                    // A capturer is legal iff it is not a blocker of its
                    // own king, or it captures along its pin line. The
                    // PREVIOUS blockers also catch the horizontal
                    // both-pawns-removed pin: before the push only one pawn
                    // sat between king and rook, so the capturer was
                    // already registered as a blocker.
                    if no_discovery && !(pawns & (not_blockers | attacks::line(ep, ksq))).is_empty()
                    {
                        st.copied.ep_square = Some(ep);
                        k ^= ZOBRIST.en_passant[ep.file() as usize];
                    }
                }
            } else if m.kind() == MoveKind::Promotion {
                let promo_pt = m.promotion_piece().expect("promotion kind has a piece");
                let promo = Piece::new(us, promo_pt);
                let promo_psq = ZOBRIST.psq[promo.index()][to.index() as usize];
                // No pawn-clear XOR: psq[pawn][promotion rank] is zero (D9).
                k ^= promo_psq;
                // Count slots: promoted piece pre-add count in, pawn
                // post-removal count out (popcounts still pre-mutation).
                let promo_count = self.pieces(us, promo_pt).count() as usize;
                let pawn_count = self.pieces(us, PieceType::Pawn).count() as usize;
                st.copied.material_key ^=
                    ZOBRIST.material(promo, promo_count) ^ ZOBRIST.material(pc, pawn_count - 1);
                st.copied.non_pawn_key[us as usize] ^= promo_psq;
                if matches!(promo_pt, PieceType::Knight | PieceType::Bishop) {
                    st.copied.minor_key ^= promo_psq;
                }
                st.copied.non_pawn_material[us as usize] += PIECE_VALUES[promo_pt as usize];
            }
            // Always: the `to` term is zero on the promotion rank, so this
            // degenerates to a from-only update for promotions.
            st.copied.pawn_key ^= ZOBRIST.psq[pc.index()][from.index() as usize]
                ^ ZOBRIST.psq[pc.index()][to.index() as usize];
            st.copied.rule50 = 0;
        } else {
            let mover_delta = ZOBRIST.psq[pc.index()][from.index() as usize]
                ^ ZOBRIST.psq[pc.index()][to.index() as usize];
            st.copied.non_pawn_key[us as usize] ^= mover_delta;
            if matches!(pc.piece_type, PieceType::Knight | PieceType::Bishop) {
                st.copied.minor_key ^= mover_delta;
            }
        }

        // Step 9: the child key is final. Hook A gets the TT probe key:
        // folded with the STANDARD threshold on the in-flight clock, which
        // is already exact here (incremented in step 2, zeroed by steps
        // 5/8). NOT `fold_rule50_after` — that models "parent clock + 1",
        // which is wrong after any capture or pawn move and exists solely
        // for the speculative [`Board::prefetch_key`] (doc 04 §3.8 step 9).
        st.key = k;
        hooks.on_key(zobrist::fold_rule50(k, st.copied.rule50));
        // Hook B: the specialized keys are final too.
        hooks.on_state(&st.copied);

        // Step 10: board mutation (castling already mutated in step 4).
        if m.kind() != MoveKind::Castling {
            if captured.is_some() {
                self.remove_piece(capsq);
            }
            if m.kind() == MoveKind::Promotion {
                let promo_pt = m.promotion_piece().expect("promotion kind has a piece");
                self.remove_piece(from);
                self.put_piece(Piece::new(us, promo_pt), to);
            } else {
                self.move_piece(from, to);
            }
        }

        // Step 11: captured piece and checkers. Trusting the caller's
        // `gives_check` keeps quiet moves free of any attack query.
        st.captured = captured;
        st.checkers = if gives_check {
            self.attackers_to(self.king_square(them), self.all_occupancy) & self.occupancy(us)
        } else {
            Bitboard::EMPTY
        };
        debug_assert_eq!(
            gives_check,
            !(self.attackers_to(self.king_square(them), self.all_occupancy) & self.occupancy(us))
                .is_empty(),
            "make_move: caller-supplied gives_check disagrees with the board after {m}"
        );

        // Step 12: flip the side to move, then rebuild the check
        // bookkeeping for the new side.
        self.side_to_move = them;
        self.set_check_info(st);

        // Step 13: repetition scan (doc 04 §3.9). The window is
        // min(rule50, plies_from_null): a capture/pawn move makes earlier
        // positions unreachable, and a null move must isolate the scan on
        // its side of the boundary. RAW keys — rule50 folding is TT-only.
        // `plies_from_null` never exceeds the plies recorded on this stack,
        // so `n - i` cannot underflow (doc 04 §3.3).
        let child_key = st.key;
        let end = st.copied.rule50.min(st.copied.plies_from_null);
        let mut repetition = 0;
        if end >= 4 {
            // `st`'s borrow ends above (NLL); the walk reads ancestors in
            // the same local vec. `child_idx` == the old pre-push len, so
            // the `n - i` arithmetic is unchanged.
            let mut i = 4;
            while i <= end {
                let earlier = &states[child_idx - i as usize];
                if earlier.key == child_key {
                    // +d: first repeat, d plies back. -d: that occurrence
                    // was itself a repeat — at least threefold.
                    repetition = if earlier.repetition != 0 { -i } else { i };
                    break;
                }
                i += 2;
            }
        }
        states[child_idx].repetition = repetition;

        // Step 14: publish. Debug builds run the full invariant oracle at
        // every node (doc 04 §3.12, D24); release compiles it out.
        self.states = states;
        #[cfg(debug_assertions)]
        self.assert_consistent();
    }

    /// Undo the last made move, which must be `m` (doc 04 §3.8).
    /// Deliberately dumb-cheap: restore the placement, pop the state — no
    /// key or check-info work, everything derived lives in the popped
    /// record and the parent state is intact underneath.
    pub fn unmake_move(&mut self, m: Move) {
        let st = self.states.pop().expect("state stack never empty");
        debug_assert!(
            !self.states.is_empty(),
            "unmake_move must leave the root state"
        );
        self.side_to_move = self.side_to_move.flip();
        let us = self.side_to_move;
        self.game_ply -= 1;
        let (from, to) = (m.from(), m.to());

        if m.kind() == MoveKind::Castling {
            // Remove both, place both — the exact inverse of step 4.
            let kingside = to.file() > from.file();
            let right = CastlingRights::single(us, kingside);
            let rfrom = self.castling_rook_square[right.index()];
            let rto = if kingside { Square::F1 } else { Square::D1 }.relative_to(us);
            let king = self
                .remove_piece(to)
                .expect("king on the castling to-square");
            let rook = self
                .remove_piece(rto)
                .expect("rook on the castling destination");
            self.put_piece(king, from);
            self.put_piece(rook, rfrom);
        } else {
            if m.kind() == MoveKind::Promotion {
                self.remove_piece(to);
                self.put_piece(Piece::new(us, PieceType::Pawn), from);
            } else {
                self.move_piece(to, from);
            }
            if let Some(cap) = st.captured {
                let capsq = if m.kind() == MoveKind::EnPassant {
                    to.offset(us.flip().pawn_push())
                } else {
                    to
                };
                self.put_piece(cap, capsq);
            }
        }
        #[cfg(debug_assertions)]
        self.assert_consistent();
    }

    /// Switch the side to move without moving anything (doc 04 §3.8, SF
    /// `do_null_move`) — the primitive behind null-move pruning. The side
    /// to move must NOT be in check.
    ///
    /// Copies the ENTIRE top state (both blocks — this is why `StateInfo:
    /// Copy`), then: ep cleared (with its key XOR), side key toggled,
    /// `plies_from_null` reset to 0 (which isolates repetition scans across
    /// the null boundary), check bookkeeping rebuilt for the new side.
    /// `rule50` and `game_ply` are deliberately unchanged (SF behavior).
    pub fn make_null_move(&mut self) {
        debug_assert!(self.checkers().is_empty(), "make_null_move while in check");
        let mut st = *self.st();
        if let Some(ep) = st.copied.ep_square.take() {
            st.key ^= ZOBRIST.en_passant[ep.file() as usize];
        }
        st.key ^= ZOBRIST.side;
        st.copied.plies_from_null = 0;
        st.captured = None;
        st.repetition = 0;
        self.side_to_move = self.side_to_move.flip();
        // `st.checkers` keeps the copied EMPTY set: the mover wasn't in
        // check (precondition) and the new side's king can't be either —
        // it wasn't attackable while the opponent was to move.
        self.set_check_info(&mut st);
        self.states.push(st);
        #[cfg(debug_assertions)]
        self.assert_consistent();
    }

    /// Undo the last [`Board::make_null_move`]: pop the state, flip back.
    pub fn unmake_null_move(&mut self) {
        debug_assert!(
            self.checkers().is_empty(),
            "unmake_null_move while in check"
        );
        self.states.pop();
        debug_assert!(
            !self.states.is_empty(),
            "unmake_null_move must leave the root state"
        );
        self.side_to_move = self.side_to_move.flip();
        #[cfg(debug_assertions)]
        self.assert_consistent();
    }

    /// Approximate child key for `m` BEFORE making it (doc 04 §3.10, SF
    /// `prefetch_key`): `side ^ mover from/to ^ captured@to`, rule50-folded
    /// with the after-move threshold for quiet non-pawn moves. Search uses
    /// it to start the TT prefetch even earlier than hook A.
    ///
    /// Exact — equal to the child's [`Board::key`] — for `Normal` moves
    /// that neither clear a parent ep square nor change castling rights;
    /// castling/ep/promotion details are deliberately ignored (a slightly
    /// wrong prefetch address costs nothing).
    pub fn prefetch_key(&self, m: Move) -> Key {
        let pc = self
            .piece_on(m.from())
            .expect("prefetch_key: no piece on the from-square");
        let captured = self.piece_on(m.to());
        let mut k = self.raw_key()
            ^ ZOBRIST.side
            ^ ZOBRIST.psq[pc.index()][m.from().index() as usize]
            ^ ZOBRIST.psq[pc.index()][m.to().index() as usize];
        if let Some(cap) = captured {
            k ^= ZOBRIST.psq[cap.index()][m.to().index() as usize];
        }
        // Captures and pawn moves reset the child clock, so its key is raw;
        // otherwise fold with threshold 13, modeling "parent rule50 + 1".
        if captured.is_some() || pc.piece_type == PieceType::Pawn {
            k
        } else {
            zobrist::fold_rule50_after(k, self.st().copied.rule50)
        }
    }

    /// Draw by the 50-move rule or by repetition, relative to the search
    /// ply (doc 04 §3.9). Search scores a true result as `VALUE_DRAW`.
    pub fn is_draw(&self, ply: i32) -> bool {
        let st = self.st();
        // 50-move rule, guarded: a mating move ON the 100th halfmove still
        // wins, so in check the claim stands only when an escape exists —
        // checkmate outranks the clock (doc 04 §9.3, closed by
        // design/05-movegen.md §3.9's `has_legal_moves`).
        if st.copied.rule50 > 99
            && (st.checkers.is_empty() || crate::movegen::has_legal_moves(self))
        {
            return true;
        }
        self.is_repetition(ply)
    }

    /// Repetition draw relative to the search ply: the position repeated
    /// once strictly INSIDE the search tree (`0 < repetition < ply`), or
    /// repeated twice anywhere — the negative "at least threefold" encoding
    /// is always `< ply` (doc 04 §3.9).
    pub fn is_repetition(&self, ply: i32) -> bool {
        let r = self.st().repetition;
        r != 0 && r < ply
    }

    /// Has the current position — or any ancestor since the last
    /// irreversible move — already repeated? Search bookkeeping (SF
    /// `has_repeated`); the walk shares the repetition scan's
    /// `min(rule50, plies_from_null)` window and shrinks it by one per
    /// step back.
    pub fn has_repeated(&self) -> bool {
        let copied = &self.st().copied;
        let mut end = copied.rule50.min(copied.plies_from_null);
        // `end <= plies_from_null <= top index`, and both shrink in
        // lockstep, so `idx` cannot underflow.
        let mut idx = self.states.len() - 1;
        while end >= 4 {
            if self.states[idx].repetition != 0 {
                return true;
            }
            idx -= 1;
            end -= 1;
        }
        false
    }

    /// True iff the static exchange value of `m` is at least `threshold`
    /// (doc 04 §3.11, D11): starting from `m`, both sides keep recapturing
    /// on the destination square with their least valuable attacker for as
    /// long as it pays, and the mover's final material balance must reach
    /// the bar. Expressed as the null-window swap algorithm: `swap` carries
    /// the running balance folded into the threshold, `res` flips with every
    /// recapture, and each capturer only has to "clear the bar" `res` (0 or
    /// 1) instead of computing an exact exchange value.
    ///
    /// Pinned pieces do not count as attackers while an enemy pinner is
    /// still on the board; x-ray batteries are walked by lifting each used
    /// capturer out of `occupied` and re-querying the slider rays through
    /// the vacated square. Values are the D11 `PIECE_VALUES`.
    pub fn see_ge(&self, m: Move, threshold: Value) -> bool {
        // Non-Normal kinds (promotion, en passant, castling) short-circuit
        // to a static exchange value of 0 (doc 04 §3.11 step 1).
        if m.kind() != MoveKind::Normal {
            return 0 >= threshold;
        }
        let (from, to) = (m.from(), m.to());
        let pc = self
            .piece_on(from)
            .expect("see_ge: no piece on the from-square");
        debug_assert_eq!(pc.color, self.side_to_move, "see_ge: moving an enemy piece");

        // Bar 1: even winning the target for free must reach the threshold
        // (an empty destination counts 0 — quiet-move SEE).
        let mut swap = self
            .piece_on(to)
            .map_or(0, |cap| PIECE_VALUES[cap.piece_type as usize])
            - threshold;
        if swap < 0 {
            return false;
        }

        // Bar 2: if the opponent recapturing our mover for free still
        // leaves us at the bar, no deeper exchange can sink us below it.
        // (A king mover has value 0, so it always returns true here — the
        // loop below never has to trade a king.)
        swap = PIECE_VALUES[pc.piece_type as usize] - swap;
        if swap <= 0 {
            return true;
        }

        // Clear the mover's square AND toggle `to`: a capture lifts the
        // victim out, a quiet move marks the destination occupied. The `to`
        // toggle keeps the pinner-liveness test below honest when the first
        // move captures a pinner (doc 04 §3.11 step 4).
        let mut occupied =
            self.all_occupancy ^ Bitboard::from_square(from) ^ Bitboard::from_square(to);
        let mut stm = self.side_to_move;
        let mut attackers = self.attackers_to(to, occupied);

        let tables = attacks::tables();
        let queens = self.pieces_of_type(PieceType::Queen);
        let diagonal = self.pieces_of_type(PieceType::Bishop) | queens;
        let orthogonal = self.pieces_of_type(PieceType::Rook) | queens;

        // Does the side that played `m` currently meet the threshold?
        let mut res = true;

        loop {
            stm = stm.flip();
            attackers &= occupied;

            // No recapture left: the standing verdict is final.
            let mut stm_attackers = attackers & self.occupancy(stm);
            if stm_attackers.is_empty() {
                break;
            }

            // Pinned pieces may not join while any enemy pinner is still on
            // the board; once every pinner has been traded off (or was the
            // initial victim — the `to` toggle above), formerly pinned
            // pieces participate freely (doc 04 §3.11 step 6, an
            // approximation shared with Stockfish).
            if !(self.pinners(stm.flip()) & occupied).is_empty() {
                stm_attackers &= !self.blockers_for_king(stm);
                if stm_attackers.is_empty() {
                    break;
                }
            }

            res = !res;

            // stm's least valuable attacker — `PieceType::ALL` is in
            // ascending value order P, N, B, R, Q, K.
            let (pt, sq) = PieceType::ALL
                .into_iter()
                .find_map(|pt| {
                    (stm_attackers & self.pieces(stm, pt))
                        .lsb()
                        .map(|sq| (pt, sq))
                })
                .expect("stm_attackers is non-empty");

            if pt == PieceType::King {
                // The king may only stand as the FINAL capturer: if the
                // opponent still has attackers, stepping onto `to` would be
                // illegal and the just-flipped verdict reverses back.
                return if (attackers & !self.occupancy(stm)).is_empty() {
                    res
                } else {
                    !res
                };
            }

            swap = PIECE_VALUES[pt as usize] - swap;
            if pt == PieceType::Queen {
                // A queen recapture after anything but a king always clears
                // the bar, because the previous capturer was worth at most a
                // queen — assert instead of test (doc 04 §3.11 step 7).
                debug_assert!(swap >= res as Value, "queen recapture under the bar");
            } else if swap < res as Value {
                // Null-window bar check: `res` is the margin this capture
                // must clear; failing ends the sequence with the
                // already-flipped verdict.
                break;
            }

            // Lift the capturer out of the occupancy and pick up any x-ray
            // attacker it was shading along its ray family (nothing x-rays
            // through a knight's hop).
            occupied ^= Bitboard::from_square(sq);
            match pt {
                PieceType::Pawn | PieceType::Bishop => {
                    attackers |= tables.bishop_attacks(to, occupied) & diagonal;
                }
                PieceType::Rook => {
                    attackers |= tables.rook_attacks(to, occupied) & orthogonal;
                }
                PieceType::Queen => {
                    attackers |= (tables.bishop_attacks(to, occupied) & diagonal)
                        | (tables.rook_attacks(to, occupied) & orthogonal);
                }
                PieceType::Knight | PieceType::King => {}
            }
        }
        res
    }

    /// The full position invariant oracle (doc 04 §3.12): placement layers
    /// in sync, state stack sane, every incremental key equal to a
    /// from-scratch recompute, structural chess sanity, and the check
    /// bookkeeping equal to a recompute. Make/unmake (moves and nulls) run
    /// it at every node in debug builds, so a debug perft run exercises it
    /// everywhere (D24); release builds compile it out entirely.
    ///
    /// Position-level checks assume a playable position; incomplete boards
    /// (`Board::empty()`, mid-construction `put_piece` tests) get only the
    /// placement-layer and stack checks.
    #[cfg(any(test, debug_assertions))]
    pub(crate) fn assert_consistent(&self) {
        // --- Placement layers: bitboards vs mailbox vs occupancy ---------
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

        // --- State stack (doc 04 §3.3): never empty, sane counters -------
        assert!(!self.states.is_empty(), "state stack must never be empty");
        let st = self.st();
        assert!(st.copied.rule50 >= 0);
        assert!(st.copied.plies_from_null >= 0);
        // The repetition scan indexes `plies_from_null` states back.
        assert!(
            st.copied.plies_from_null < self.states.len() as i32,
            "plies_from_null must not exceed the recorded history"
        );
        assert!(self.game_ply >= 0);

        // Everything below assumes a playable position.
        if !self.has_both_kings() {
            return;
        }

        // --- Keys: all six incremental values vs from-scratch recompute --
        self.assert_state_keys_consistent();

        // --- Structural sanity --------------------------------------------
        for color in Color::ALL {
            assert_eq!(
                self.pieces(color, PieceType::King).count(),
                1,
                "{color:?} must have exactly one king"
            );
        }
        assert!(
            (self.pieces_of_type(PieceType::Pawn) & (Bitboard::RANK_1 | Bitboard::RANK_8))
                .is_empty(),
            "no pawns on ranks 1/8"
        );
        for right in [
            CastlingRights::WHITE_KINGSIDE,
            CastlingRights::WHITE_QUEENSIDE,
            CastlingRights::BLACK_KINGSIDE,
            CastlingRights::BLACK_QUEENSIDE,
        ] {
            if st.copied.castling_rights.contains(right) {
                let color = if right.index() < 2 {
                    Color::White
                } else {
                    Color::Black
                };
                assert_eq!(
                    self.king_square(color),
                    Square::E1.relative_to(color),
                    "castling right {right} without the king at home"
                );
                let rook_sq = self.castling_rook_square[right.index()];
                assert_eq!(
                    self.piece_on(rook_sq),
                    Some(Piece::new(color, PieceType::Rook)),
                    "castling right {right} without its rook on {rook_sq}"
                );
            }
        }
        if let Some(ep) = st.copied.ep_square {
            let us = self.side_to_move;
            let them = us.flip();
            assert_eq!(ep.relative_rank(us), 5, "ep square {ep} on the wrong rank");
            assert_eq!(
                self.piece_on(ep.offset(them.pawn_push())),
                Some(Piece::new(them, PieceType::Pawn)),
                "ep square {ep} without the double-pushed pawn in front"
            );
            assert!(self.piece_on(ep).is_none(), "ep square {ep} must be empty");
            assert!(
                self.piece_on(ep.offset(us.pawn_push())).is_none(),
                "ep origin behind {ep} must be empty"
            );
        }

        // --- Check bookkeeping vs from-scratch recompute ------------------
        let stm = self.side_to_move;
        assert_eq!(
            st.checkers,
            self.attackers_to(self.king_square(stm), self.all_occupancy)
                & self.occupancy(stm.flip()),
            "checkers drifted"
        );
        let mut expected = *st;
        self.set_check_info(&mut expected);
        assert_eq!(
            st.blockers_for_king, expected.blockers_for_king,
            "blockers_for_king drifted"
        );
        assert_eq!(st.pinners, expected.pinners, "pinners drifted");
        assert_eq!(
            st.check_squares, expected.check_squares,
            "check_squares drifted"
        );
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

    /// Independent oracle for the Wave-G threat map: `them` (side not to move)
    /// threatens `sq` iff one of its pieces attacks `sq`, with the side-to-
    /// move's king x-rayed out of slider occupancy — computed via
    /// `attackers_to` (a different code path than the setwise fills).
    fn threat_oracle(b: &Board) -> Bitboard {
        let us = b.side_to_move();
        let them = us.flip();
        let slider_occ = b.all_occupancy() ^ Bitboard::from_square(b.king_square(us));
        let them_occ = b.occupancy(them);
        let mut t = Bitboard::EMPTY;
        for i in 0..64u8 {
            let sq = Square::new(i);
            if !(b.attackers_to(sq, slider_occ) & them_occ).is_empty() {
                t |= Bitboard::from_square(sq);
            }
        }
        t
    }

    #[test]
    fn threat_map_matches_the_attackers_to_oracle() {
        // G0 (D30): the setwise-filled `all_threats` equals the per-square
        // oracle across a range of positions (incl. slider x-ray through the
        // king), and the per-piece-type maps union to it.
        let fens = [
            crate::fen::START_FEN,
            "r3k2r/p1ppqpb1/bn2pnp1/3PN3/1p2P3/2N2Q1p/PPPBBPPP/R3K2R w KQkq - 0 1", // kiwipete
            "8/2p5/3p4/KP5r/1R3p1k/8/4P1P1/8 w - - 0 1",
            "6k1/8/8/8/3q4/8/8/4K3 b - - 0 1", // queen x-rays the white king
        ];
        for fen in fens {
            let b = Board::from_fen(fen).unwrap();
            assert_eq!(b.all_threats(), threat_oracle(&b), "all_threats for {fen}");
            let mut union = Bitboard::EMPTY;
            for pt in PieceType::ALL {
                union |= b.piece_threats(pt);
            }
            assert_eq!(union, b.all_threats(), "piece_threats union for {fen}");
        }
    }

    #[test]
    fn threat_map_survives_make_unmake() {
        // The threat map is maintained per state, so it is fresh after a move
        // and restored after unmake.
        let mut b = Board::from_fen(crate::fen::START_FEN).unwrap();
        let before = b.all_threats();
        let m = Move::normal(Square::E2, Square::E4);
        b.make_move(m, b.gives_check(m));
        assert_eq!(
            b.all_threats(),
            threat_oracle(&b),
            "threats fresh after make"
        );
        b.unmake_move(m);
        assert_eq!(b.all_threats(), before, "threats restored after unmake");
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

    // design/05 §4 step 1: the per-color wing helpers map to the four
    // existing constants.
    #[test]
    fn castling_rights_wing_helpers() {
        assert_eq!(
            CastlingRights::kingside(Color::White),
            CastlingRights::WHITE_KINGSIDE
        );
        assert_eq!(
            CastlingRights::queenside(Color::White),
            CastlingRights::WHITE_QUEENSIDE
        );
        assert_eq!(
            CastlingRights::kingside(Color::Black),
            CastlingRights::BLACK_KINGSIDE
        );
        assert_eq!(
            CastlingRights::queenside(Color::Black),
            CastlingRights::BLACK_QUEENSIDE
        );
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
        // The bound grew from 192 with the Wave-G threat map (D30): `all_threats`
        // + `piece_threats[6]` add 7 bitboards (56 bytes). These are recomputed
        // per state (not carried in `copied`), so make_move does not copy them;
        // only make_null_move's whole-record clone and the Vec entry pay the
        // size — an accepted cost for O(1) threat queries.
        const { assert!(size_of::<StateInfo>() <= 256) }
    }

    // Doc 04 §8 item 7: make_move performs no heap allocation — the state
    // stack is pre-reserved past MAX_PLY at construction (doc 04 §3.3), so a
    // full search-depth walk of makes and nulls never grows it. Capacity
    // equality is the observable for "no reallocation".
    #[test]
    fn state_stack_is_prereserved_and_never_reallocates() {
        let mut board = Board::startpos();
        let cap = board.states.capacity();
        assert!(
            cap >= MAX_PLY + 64,
            "state stack must be reserved past MAX_PLY ({cap} < {})",
            MAX_PLY + 64
        );

        // MAX_PLY - 6 real plies plus a few nulls: as deep as any search
        // may stack states on top of the root.
        for _ in 0..(MAX_PLY - 6) / 4 {
            for m in knight_shuffle() {
                board.make_move(m, false);
            }
        }
        board.make_null_move();
        board.make_null_move();
        assert!(board.states.len() > MAX_PLY - 10);
        assert_eq!(
            board.states.capacity(),
            cap,
            "make_move/make_null_move must never reallocate the state stack"
        );
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

        // Diverge the clone through the real mutation path (a bare
        // remove_piece would leave stale keys — exactly what the oracle
        // exists to catch); the original must be untouched.
        clone.make_move(Move::normal(Square::E2, Square::E4), false);
        assert_ne!(clone, board);
        assert_eq!(clone.piece_on(Square::E2), None);
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

    // -----------------------------------------------------------------
    // Doc 04 §4 steps 5–6: make_move / unmake_move + gives_check
    // -----------------------------------------------------------------

    const KIWIPETE: &str = "r3k2r/p1ppqpb1/bn2pnp1/3PN3/1p2P3/2N2Q1p/PPPBBPPP/R3K2R w KQkq - 0 1";

    /// Make `m` with full cross-checking (doc 04 §5): the caller-supplied
    /// `gives_check` contract, the from-scratch key oracle for all six
    /// keys, bitboard/mailbox sync, and a recomputed checkers set.
    fn make_checked(board: &mut Board, m: Move) {
        let gives_check = board.gives_check(m);
        board.make_move(m, gives_check);
        board.assert_consistent();
        board.assert_state_keys_consistent();
        let stm = board.side_to_move();
        let expected = board.attackers_to(board.king_square(stm), board.all_occupancy())
            & board.occupancy(stm.flip());
        assert_eq!(board.checkers(), expected, "checkers drifted after {m}");
        assert_eq!(
            gives_check,
            !expected.is_empty(),
            "gives_check({m}) was wrong"
        );
    }

    /// Play `moves` in order (each fully cross-checked), then unmake them
    /// in reverse, asserting the board comes back EQUAL at every ply —
    /// placement, side, counters, and the whole state stack (all six keys
    /// included) via derived `PartialEq`. Returns the final position for
    /// post-move assertions.
    fn roundtrip(fen: &str, moves: &[Move]) -> Board {
        let mut board = Board::from_fen(fen).expect("test FEN parses");
        let mut snapshots = vec![board.clone()];
        for &m in moves {
            make_checked(&mut board, m);
            snapshots.push(board.clone());
        }
        let end = board.clone();
        for &m in moves.iter().rev() {
            snapshots.pop();
            board.unmake_move(m);
            board.assert_consistent();
            assert_eq!(
                &board,
                snapshots.last().expect("root snapshot remains"),
                "unmaking {m} did not restore the previous position"
            );
        }
        end
    }

    #[test]
    fn make_unmake_roundtrip_from_startpos() {
        // Berlin-flavored opening: normal moves, double pushes, four
        // captures (N/B/P/Q), kingside castling — every ply key-oracle
        // checked, every unmake compared for full state equality.
        let end = roundtrip(
            crate::fen::START_FEN,
            &[
                Move::normal(Square::E2, Square::E4),
                Move::normal(Square::E7, Square::E5),
                Move::normal(Square::G1, Square::F3),
                Move::normal(Square::B8, Square::C6),
                Move::normal(Square::F1, Square::B5),
                Move::normal(Square::G8, Square::F6),
                Move::castling(Square::E1, Square::G1),
                Move::normal(Square::F6, Square::E4), // NxP
                Move::normal(Square::D2, Square::D4),
                Move::normal(Square::E5, Square::D4), // PxP
                Move::normal(Square::B5, Square::C6), // BxN
                Move::normal(Square::D7, Square::C6), // PxB
                Move::normal(Square::D1, Square::D4), // QxP
                Move::normal(Square::D8, Square::D4), // QxQ
            ],
        );
        assert_eq!(end.side_to_move(), Color::White);
        assert_eq!(
            end.captured_piece(),
            Some(Piece::new(Color::White, PieceType::Queen))
        );
        assert_eq!(
            end.piece_on(Square::D4),
            Some(Piece::new(Color::Black, PieceType::Queen))
        );
        assert_eq!(
            end.piece_on(Square::G1),
            Some(Piece::new(Color::White, PieceType::King))
        );
        assert_eq!(
            end.piece_on(Square::F1),
            Some(Piece::new(Color::White, PieceType::Rook))
        );
    }

    #[test]
    fn make_unmake_roundtrip_on_kiwipete() {
        // A capturable double push, the ep capture, castling, and a
        // capturing under-promotion on a rook's home square — one sequence
        // on the classic perft position.
        let mut board = Board::from_fen(KIWIPETE).unwrap();
        make_checked(&mut board, Move::normal(Square::A2, Square::A4));
        assert_eq!(board.en_passant_square(), Some(Square::A3));

        let end = roundtrip(
            KIWIPETE,
            &[
                Move::normal(Square::A2, Square::A4),
                Move::en_passant(Square::B4, Square::A3),
                Move::castling(Square::E1, Square::G1),
                Move::normal(Square::A3, Square::B2), // PxP
                Move::normal(Square::G2, Square::G3),
                Move::promotion(Square::B2, Square::A1, PieceType::Knight), // PxR=N
            ],
        );
        assert_eq!(
            end.piece_on(Square::A1),
            Some(Piece::new(Color::Black, PieceType::Knight))
        );
        assert_eq!(
            end.captured_piece(),
            Some(Piece::new(Color::White, PieceType::Rook))
        );
    }

    // Doc 04 §5 ep hashing: the new ep square is set — and its file keyed —
    // ONLY when an ep capture will actually be legal (D9).
    #[test]
    fn double_push_sets_ep_only_when_capturable() {
        // No adjacent enemy pawn: no ep square; the child hashes exactly
        // like the same position parsed from FEN with "-".
        let mut board = Board::startpos();
        make_checked(&mut board, Move::normal(Square::E2, Square::E4));
        assert_eq!(board.en_passant_square(), None);
        let reparsed =
            Board::from_fen("rnbqkbnr/pppppppp/8/8/4P3/8/PPPP1PPP/RNBQKBNR b KQkq - 0 1").unwrap();
        assert_eq!(board.raw_key(), reparsed.raw_key());

        // Adjacent enemy pawn, unpinned: ep set, keyed by its FILE.
        let mut board = Board::from_fen("4k3/8/8/8/5p2/8/4P3/4K3 w - - 0 1").unwrap();
        let before = board.raw_key();
        make_checked(&mut board, Move::normal(Square::E2, Square::E4));
        assert_eq!(board.en_passant_square(), Some(Square::E3));
        let wp = Piece::new(Color::White, PieceType::Pawn).index();
        assert_eq!(
            board.raw_key(),
            before
                ^ ZOBRIST.side
                ^ ZOBRIST.psq[wp][Square::E2.index() as usize]
                ^ ZOBRIST.psq[wp][Square::E4.index() as usize]
                ^ ZOBRIST.en_passant[Square::E3.file() as usize]
        );
    }

    #[test]
    fn double_push_withholds_ep_when_capture_is_illegal() {
        // Vertical pin of the capturer: ...dxe3 would open the d-file rook
        // onto the black king.
        let mut board = Board::from_fen("3k4/8/8/8/3p4/8/4P3/2KR4 w - - 0 1").unwrap();
        make_checked(&mut board, Move::normal(Square::E2, Square::E4));
        assert_eq!(board.en_passant_square(), None);

        // Horizontal both-pawns-removed pin (the doc 04 §5 Kh5/Pf5/Ra5
        // case): fxe6 would clear BOTH e5 and f5 off rank 5. Only the
        // PREVIOUS state's blockers see this — after the push there are
        // two pieces between rook and king.
        let mut board = Board::from_fen("4k3/4p3/8/r4P1K/8/8/8/8 b - - 0 1").unwrap();
        make_checked(&mut board, Move::normal(Square::E7, Square::E5));
        assert_eq!(board.en_passant_square(), None);

        // The double push itself delivers discovered check: ep can never
        // be the legal reply.
        let mut board = Board::from_fen("8/8/k7/8/5p2/8/4P3/4KB2 w - - 0 1").unwrap();
        make_checked(&mut board, Move::normal(Square::E2, Square::E4));
        assert_eq!(board.en_passant_square(), None);
        assert_eq!(board.checkers(), bb(&[Square::F1]));
    }

    #[test]
    fn ep_capture_roundtrip_and_pin_line_capture_keeps_ep() {
        // Plain ep capture round-trips — capsq != to, so unmake must put
        // the captured pawn back on its own square, not the to-square.
        let end = roundtrip(
            "4k3/8/8/8/5p2/8/4P3/4K3 w - - 0 1",
            &[
                Move::normal(Square::E2, Square::E4),
                Move::en_passant(Square::F4, Square::E3),
            ],
        );
        assert_eq!(
            end.piece_on(Square::E3),
            Some(Piece::new(Color::Black, PieceType::Pawn))
        );
        assert_eq!(end.piece_on(Square::E4), None);
        assert_eq!(
            end.captured_piece(),
            Some(Piece::new(Color::White, PieceType::Pawn))
        );

        // The capturer is pinned, but the ep capture travels ALONG the
        // c4-f7 pin line: ep stays set (and keyed), the capture plays.
        let fen = "4k3/4pb2/8/3P4/2K5/8/8/8 b - - 0 1";
        let mut board = Board::from_fen(fen).unwrap();
        make_checked(&mut board, Move::normal(Square::E7, Square::E5));
        assert_eq!(board.en_passant_square(), Some(Square::E6));
        make_checked(&mut board, Move::en_passant(Square::D5, Square::E6));
        assert_eq!(board.piece_on(Square::E5), None);
        board.unmake_move(Move::en_passant(Square::D5, Square::E6));
        board.unmake_move(Move::normal(Square::E7, Square::E5));
        assert_eq!(board, Board::from_fen(fen).unwrap());
    }

    #[test]
    fn promotion_roundtrip_all_pieces() {
        for pt in [
            PieceType::Knight,
            PieceType::Bishop,
            PieceType::Rook,
            PieceType::Queen,
        ] {
            let end = roundtrip(
                "4k3/1P6/8/8/8/8/8/4K3 w - - 0 1",
                &[Move::promotion(Square::B7, Square::B8, pt)],
            );
            assert_eq!(end.piece_on(Square::B8), Some(Piece::new(Color::White, pt)));
            assert!(end.pieces(Color::White, PieceType::Pawn).is_empty());
            assert_eq!(
                end.non_pawn_material(Color::White),
                PIECE_VALUES[pt as usize]
            );
            // The board went pawnless: back to the no-pawns baseline.
            assert_eq!(end.pawn_key(), ZOBRIST.no_pawns);
        }

        // Black promotes too — the zeroed promotion rank is color-specific.
        let end = roundtrip(
            "4k3/8/8/8/8/8/1p6/4K3 b - - 0 1",
            &[Move::promotion(Square::B2, Square::B1, PieceType::Queen)],
        );
        assert_eq!(
            end.piece_on(Square::B1),
            Some(Piece::new(Color::Black, PieceType::Queen))
        );
    }

    #[test]
    fn promotion_capture_revokes_castling_rights() {
        let end = roundtrip(
            "4k2r/6P1/8/8/8/8/8/4K3 w k - 0 1",
            &[Move::promotion(Square::G7, Square::H8, PieceType::Queen)],
        );
        assert_eq!(
            end.piece_on(Square::H8),
            Some(Piece::new(Color::White, PieceType::Queen))
        );
        assert_eq!(end.castling_rights(), CastlingRights::NONE);
        assert_eq!(
            end.captured_piece(),
            Some(Piece::new(Color::Black, PieceType::Rook))
        );
        assert_eq!(end.non_pawn_material(Color::Black), 0);
        assert_eq!(
            end.non_pawn_material(Color::White),
            PIECE_VALUES[PieceType::Queen as usize]
        );
    }

    #[test]
    fn castling_all_four_roundtrip() {
        let castle_fen = |stm: char| format!("r3k2r/8/8/8/8/8/8/R3K2R {stm} KQkq - 0 1");
        let cases = [
            ('w', Square::E1, Square::G1, Square::F1, "kq"),
            ('w', Square::E1, Square::C1, Square::D1, "kq"),
            ('b', Square::E8, Square::G8, Square::F8, "KQ"),
            ('b', Square::E8, Square::C8, Square::D8, "KQ"),
        ];
        for (stm, kfrom, kto, rto, rights) in cases {
            let color = if stm == 'w' {
                Color::White
            } else {
                Color::Black
            };
            let m = Move::castling(kfrom, kto);
            let end = roundtrip(&castle_fen(stm), &[m]);
            assert_eq!(
                end.piece_on(kto),
                Some(Piece::new(color, PieceType::King)),
                "{m}"
            );
            assert_eq!(
                end.piece_on(rto),
                Some(Piece::new(color, PieceType::Rook)),
                "{m}"
            );
            assert_eq!(end.piece_on(kfrom), None, "{m}");
            assert_eq!(end.castling_rights(), rights.parse().unwrap(), "{m}");
        }

        // Both sides castle in one sequence.
        let end = roundtrip(
            &castle_fen('w'),
            &[
                Move::castling(Square::E1, Square::G1),
                Move::castling(Square::E8, Square::C8),
            ],
        );
        assert_eq!(end.castling_rights(), CastlingRights::NONE);
    }

    // Doc 04 §5: rights revocation via the mask — king move, each rook
    // move, and a rook CAPTURING a never-moved rook on its home square
    // (from-mask and to-mask fire together). Unmake restores the rights
    // (checked ply-by-ply inside roundtrip).
    #[test]
    fn castling_rights_revocation_cases() {
        const FEN: &str = "r3k2r/8/8/8/8/8/8/R3K2R w KQkq - 0 1";
        let cases: [(Move, &str); 4] = [
            (Move::normal(Square::E1, Square::E2), "kq"), // king move: both rights
            (Move::normal(Square::H1, Square::G1), "Qkq"), // kingside rook move
            (Move::normal(Square::A1, Square::B1), "Kkq"), // queenside rook move
            (Move::normal(Square::A1, Square::A8), "Kk"), // RxR: one right per side
        ];
        for (m, rights) in cases {
            let end = roundtrip(FEN, &[m]);
            assert_eq!(end.castling_rights(), rights.parse().unwrap(), "{m}");
        }

        // Rights never come back by returning home — only unmake restores.
        let end = roundtrip(
            FEN,
            &[
                Move::normal(Square::E1, Square::E2),
                Move::normal(Square::E8, Square::E7),
                Move::normal(Square::E2, Square::E1),
                Move::normal(Square::E7, Square::E8),
            ],
        );
        assert_eq!(end.castling_rights(), CastlingRights::NONE);
    }

    #[test]
    fn counters_and_captured_piece_track_moves() {
        let mut board = Board::startpos();
        make_checked(&mut board, Move::normal(Square::G1, Square::F3));
        assert_eq!(board.rule50_count(), 1);
        assert_eq!(board.game_ply(), 1);
        assert_eq!(board.fullmove_number(), 1);
        assert_eq!(board.side_to_move(), Color::Black);
        make_checked(&mut board, Move::normal(Square::B8, Square::C6));
        assert_eq!(board.rule50_count(), 2);
        assert_eq!(board.game_ply(), 2);
        assert_eq!(board.fullmove_number(), 2);
        // A pawn move resets the clock...
        make_checked(&mut board, Move::normal(Square::E2, Square::E4));
        assert_eq!(board.rule50_count(), 0);
        make_checked(&mut board, Move::normal(Square::C6, Square::D4));
        assert_eq!(board.rule50_count(), 1);
        // ...and so does a capture.
        make_checked(&mut board, Move::normal(Square::F3, Square::D4));
        assert_eq!(board.rule50_count(), 0);
        assert_eq!(
            board.captured_piece(),
            Some(Piece::new(Color::Black, PieceType::Knight))
        );
        assert_eq!(board.game_ply(), 5);

        // Unmake winds every counter back to the start position.
        for m in [
            Move::normal(Square::F3, Square::D4),
            Move::normal(Square::C6, Square::D4),
            Move::normal(Square::E2, Square::E4),
            Move::normal(Square::B8, Square::C6),
            Move::normal(Square::G1, Square::F3),
        ] {
            board.unmake_move(m);
        }
        assert_eq!(board, Board::startpos());
    }

    // Doc 04 §5 gives_check fixtures. Every case is ALSO cross-checked
    // against make_move + recomputed checkers inside roundtrip.
    #[test]
    fn gives_check_fixture_cases() {
        let check = |fen: &str, m: Move| {
            let board = Board::from_fen(fen).unwrap();
            let expected = board.gives_check(m);
            roundtrip(fen, &[m]);
            expected
        };

        // Direct checks (queen slide, knight hop) and a quiet non-check.
        assert!(check(
            "4k3/8/8/8/8/8/8/4KQ2 w - - 0 1",
            Move::normal(Square::F1, Square::F8)
        ));
        assert!(check(
            "4k3/8/8/1N6/8/8/8/4K3 w - - 0 1",
            Move::normal(Square::B5, Square::D6)
        ));
        assert!(!check(
            "4k3/8/8/1N6/8/8/8/4K3 w - - 0 1",
            Move::normal(Square::B5, Square::C3)
        ));

        // Discovered check: the pawn blocker leaves the e-file...
        assert!(check(
            "4k3/8/8/3p4/4P3/8/8/4RK2 w - - 0 1",
            Move::normal(Square::E4, Square::D5)
        ));
        // ...but pushing ALONG the king line is no check at all.
        assert!(!check(
            "4k3/8/8/3p4/4P3/8/8/4RK2 w - - 0 1",
            Move::normal(Square::E4, Square::E5)
        ));

        // Promotion checking through the vacated from-square (rook), and
        // the bishop flavor that misses.
        assert!(check(
            "8/6P1/8/8/8/6k1/8/4K3 w - - 0 1",
            Move::promotion(Square::G7, Square::G8, PieceType::Rook)
        ));
        assert!(!check(
            "8/6P1/8/8/8/6k1/8/4K3 w - - 0 1",
            Move::promotion(Square::G7, Square::G8, PieceType::Bishop)
        ));

        // Ep double-discovery through the CAPTURED pawn's square: both
        // pawns leave rank 5, unmasking Rh5–Ka5.
        assert!(check(
            "8/8/8/k2pP2R/8/8/8/4K3 w - d6 0 2",
            Move::en_passant(Square::E5, Square::D6)
        ));
        // The same capture with no slider behind it: no check.
        assert!(!check(
            "4k3/8/8/3pP3/8/8/8/4K3 w - d6 0 1",
            Move::en_passant(Square::E5, Square::D6)
        ));

        // Castling: the rook lands on the king's file with check...
        assert!(check(
            "5k2/8/8/8/8/8/8/4K2R w K - 0 1",
            Move::castling(Square::E1, Square::G1)
        ));
        // ...or harmlessly.
        assert!(!check(
            "4k3/8/8/8/8/8/8/4K2R w K - 0 1",
            Move::castling(Square::E1, Square::G1)
        ));
    }

    // Doc 04 §5 transposition invariance: two move orders reaching the
    // same position produce identical keys; a knight shuffle returning
    // home restores every key (though the clock remembers the walk).
    #[test]
    fn transposition_invariance() {
        let play = |moves: &[Move]| {
            let mut board = Board::startpos();
            for &m in moves {
                make_checked(&mut board, m);
            }
            board
        };
        let a = play(&[
            Move::normal(Square::G1, Square::F3),
            Move::normal(Square::B8, Square::C6),
            Move::normal(Square::B1, Square::C3),
            Move::normal(Square::G8, Square::F6),
        ]);
        let b = play(&[
            Move::normal(Square::B1, Square::C3),
            Move::normal(Square::G8, Square::F6),
            Move::normal(Square::G1, Square::F3),
            Move::normal(Square::B8, Square::C6),
        ]);
        assert_eq!(a.raw_key(), b.raw_key());
        assert_eq!(a.key(), b.key());
        assert_eq!(a.pawn_key(), b.pawn_key());
        assert_eq!(a.material_key(), b.material_key());
        assert_eq!(a.minor_key(), b.minor_key());
        for color in Color::ALL {
            assert_eq!(a.non_pawn_key(color), b.non_pawn_key(color));
        }

        let home = play(&[
            Move::normal(Square::G1, Square::F3),
            Move::normal(Square::B8, Square::C6),
            Move::normal(Square::F3, Square::G1),
            Move::normal(Square::C6, Square::B8),
        ]);
        assert_eq!(home.raw_key(), Board::startpos().raw_key());
        assert_ne!(home.rule50_count(), 0);
    }

    // -----------------------------------------------------------------
    // Doc 04 §4 step 7: null moves, repetition, draws
    // -----------------------------------------------------------------

    /// Four plies of knight shuffling that land back on the same position.
    fn knight_shuffle() -> [Move; 4] {
        [
            Move::normal(Square::G1, Square::F3),
            Move::normal(Square::G8, Square::F6),
            Move::normal(Square::F3, Square::G1),
            Move::normal(Square::F6, Square::G8),
        ]
    }

    // Doc 04 §5 null move: key = old ^ side ^ (ep key if any); rule50 and
    // game_ply hold; plies_from_null resets; unmake restores full equality.
    #[test]
    fn null_move_key_toggle_and_roundtrip() {
        // No ep square: the raw key toggles exactly the side key.
        let mut board = Board::startpos();
        make_checked(&mut board, Move::normal(Square::G1, Square::F3));
        let before = board.clone();
        assert_eq!(before.states.last().unwrap().copied.plies_from_null, 1);
        board.make_null_move();
        board.assert_consistent();
        assert_eq!(board.raw_key(), before.raw_key() ^ ZOBRIST.side);
        assert_eq!(board.side_to_move(), Color::White);
        assert_eq!(board.rule50_count(), before.rule50_count());
        assert_eq!(board.game_ply(), before.game_ply());
        assert_eq!(board.states.last().unwrap().copied.plies_from_null, 0);
        assert!(board.checkers().is_empty());
        assert_eq!(board.captured_piece(), None);
        board.unmake_null_move();
        assert_eq!(board, before);

        // With a legal ep square: its FILE key toggles out too, and the
        // child has no ep square (a null move forfeits the capture).
        let mut board =
            Board::from_fen("rnbqkbnr/ppp1pppp/8/8/3pP3/8/PPPP1PPP/RNBQKBNR b KQkq e3 0 2")
                .unwrap();
        assert_eq!(board.en_passant_square(), Some(Square::E3));
        let before = board.clone();
        board.make_null_move();
        board.assert_consistent();
        assert_eq!(
            board.raw_key(),
            before.raw_key() ^ ZOBRIST.side ^ ZOBRIST.en_passant[Square::E3.file() as usize]
        );
        assert_eq!(board.en_passant_square(), None);
        board.unmake_null_move();
        assert_eq!(board, before);
    }

    // Doc 04 §5 repetition: the shuffle produces +d, then -d; is_draw is
    // ply-relative (in-tree twofold draws, root-side twofold does not,
    // threefold draws anywhere); unmaking rewinds cleanly.
    #[test]
    fn repetition_encoding_and_ply_relative_draw() {
        let mut board = Board::startpos();
        for m in knight_shuffle() {
            assert!(!board.has_repeated());
            make_checked(&mut board, m);
        }
        // First return: +distance.
        assert_eq!(board.states.last().unwrap().repetition, 4);
        assert_eq!(board.raw_key(), Board::startpos().raw_key());
        assert!(board.has_repeated());
        // Strictly inside the search tree (repetition < ply) it is a draw;
        // at the boundary or root a single repeat is not (yet).
        assert!(board.is_repetition(5));
        assert!(board.is_draw(5));
        assert!(!board.is_repetition(4));
        assert!(!board.is_draw(4));
        assert!(!board.is_draw(0));

        // Second return: -distance (the earlier occurrence was itself a
        // repeat, i.e. at least threefold) — a draw at ANY ply.
        for m in knight_shuffle() {
            make_checked(&mut board, m);
        }
        assert_eq!(board.states.last().unwrap().repetition, -4);
        assert!(board.is_repetition(0));
        assert!(board.is_draw(0));

        // Unmake rewinds everything back to a repetition-free start.
        for m in knight_shuffle().into_iter().rev() {
            board.unmake_move(m);
        }
        for m in knight_shuffle().into_iter().rev() {
            board.unmake_move(m);
        }
        assert_eq!(board, Board::startpos());
        assert!(!board.has_repeated());
    }

    // Doc 04 §5: the scan window is min(rule50, plies_from_null) — a null
    // move cuts it even when the clock alone would allow the match.
    #[test]
    fn repetition_window_cut_by_null_move() {
        // Mature clock at the root so rule50 does NOT clip the window.
        let fen = "rnbqkbnr/pppppppp/8/8/8/8/PPPPPPPP/RNBQKBNR w KQkq - 10 6";
        let mut board = Board::from_fen(fen).unwrap();
        let root_key = board.raw_key();
        board.make_null_move();
        make_checked(&mut board, Move::normal(Square::B8, Square::C6));
        board.make_null_move();
        make_checked(&mut board, Move::normal(Square::C6, Square::B8));
        // The position genuinely equals the root (the two null side-toggles
        // cancel) — but positions on the far side of a null are unreachable
        // by real play, so the reset plies_from_null clips the scan.
        assert_eq!(board.raw_key(), root_key);
        assert_eq!(board.states.last().unwrap().repetition, 0);
        assert!(!board.has_repeated());
        assert!(!board.is_draw(4));
    }

    // Doc 04 §5: a capture zeroes rule50, cutting the window — earlier
    // repeats stop counting, later ones are detected as usual.
    #[test]
    fn repetition_window_cut_by_capture() {
        let fen = "4k3/8/7p/8/8/8/8/4K2R w - - 0 1";
        let mut board = Board::from_fen(fen).unwrap();
        for m in [
            Move::normal(Square::H1, Square::H2),
            Move::normal(Square::E8, Square::D8),
            Move::normal(Square::H2, Square::H1),
            Move::normal(Square::D8, Square::E8),
        ] {
            make_checked(&mut board, m);
        }
        assert_eq!(board.states.last().unwrap().repetition, 4);
        assert!(board.has_repeated());

        // RxP: irreversible — the accumulated repeat is out of the window.
        make_checked(&mut board, Move::normal(Square::H1, Square::H6));
        assert_eq!(board.states.last().unwrap().repetition, 0);
        assert!(!board.has_repeated());

        // Repeats of POST-capture positions count again.
        for m in [
            Move::normal(Square::E8, Square::D8),
            Move::normal(Square::H6, Square::H1),
            Move::normal(Square::D8, Square::E8),
            Move::normal(Square::H1, Square::H6),
        ] {
            make_checked(&mut board, m);
        }
        assert_eq!(board.states.last().unwrap().repetition, 4);
        assert!(board.is_draw(5));
        assert!(!board.is_draw(0));
    }

    // Doc 04 §5: rule50 folding never hides a repetition — the scan
    // compares RAW keys, so a repeat across fold buckets still counts.
    #[test]
    fn repetition_detected_across_rule50_fold_buckets() {
        // Root at clock 12 (raw bucket); the repeat lands at 16 (folded).
        let fen = "rnbqkbnr/pppppppp/8/8/8/8/PPPPPPPP/RNBQKBNR w KQkq - 12 7";
        let mut board = Board::from_fen(fen).unwrap();
        let root = board.clone();
        for m in knight_shuffle() {
            make_checked(&mut board, m);
        }
        assert_eq!(board.raw_key(), root.raw_key());
        assert_ne!(board.key(), root.key(), "clocks 12 vs 16 fold differently");
        assert_eq!(board.states.last().unwrap().repetition, 4);
        assert!(board.is_draw(5));
    }

    // Doc 04 §3.9: rule50 > 99 draws at any ply — unless the side to move
    // is checkmated (a mating move on the 100th halfmove wins; the movegen
    // `has_legal_moves` guard landed with design/05 §3.9).
    #[test]
    fn fifty_move_rule_draw_with_check_guard() {
        let mut board = Board::from_fen("4k3/8/8/8/8/8/8/4K2R w - - 99 80").unwrap();
        assert!(!board.is_draw(0), "clock 99 is not yet a draw");
        make_checked(&mut board, Move::normal(Square::H1, Square::H2));
        assert_eq!(board.rule50_count(), 100);
        assert!(board.is_draw(0));
        assert!(board.is_draw(100));

        // In check WITH escapes on a full clock: the draw claim stands.
        let board = Board::from_fen("4k3/8/8/8/8/8/4r3/4K3 w - - 100 80").unwrap();
        assert!(!board.checkers().is_empty());
        assert!(board.is_draw(0), "check with legal escapes: still a draw");

        // Checkmated on the 100th halfmove: mate outranks the clock.
        let board =
            Board::from_fen("rnb1kbnr/pppp1ppp/8/4p3/6Pq/5P2/PPPPP2P/RNBQKBNR w KQkq - 100 3")
                .unwrap();
        assert!(!board.checkers().is_empty());
        assert!(!board.is_draw(0), "checkmate on a full clock is not a draw");
    }

    // -----------------------------------------------------------------
    // Doc 04 §4 step 8: the oracle validates the validator
    // -----------------------------------------------------------------

    #[test]
    #[should_panic(expected = "main key drifted")]
    fn assert_consistent_catches_key_corruption() {
        let mut board = Board::startpos();
        board.states.last_mut().unwrap().key ^= 1;
        board.assert_consistent();
    }

    #[test]
    #[should_panic(expected = "pawn_key drifted")]
    fn assert_consistent_catches_pawn_key_corruption() {
        let mut board = Board::startpos();
        board.states.last_mut().unwrap().copied.pawn_key ^= 1;
        board.assert_consistent();
    }

    #[test]
    #[should_panic(expected = "checkers drifted")]
    fn assert_consistent_catches_checkers_corruption() {
        let mut board = Board::startpos();
        board.states.last_mut().unwrap().checkers = Bitboard::from_square(Square::E4);
        board.assert_consistent();
    }

    #[test]
    #[should_panic(expected = "blockers_for_king drifted")]
    fn assert_consistent_catches_blockers_corruption() {
        let mut board = Board::startpos();
        board.states.last_mut().unwrap().blockers_for_king[Color::White as usize] =
            Bitboard::from_square(Square::E2);
        board.assert_consistent();
    }

    // -----------------------------------------------------------------
    // Doc 04 §4 step 10: hook seam + prefetch_key
    // -----------------------------------------------------------------

    /// Test hooks that record what the seam reports.
    #[derive(Default)]
    struct RecordingHooks {
        keys: Vec<Key>,
        copied: Vec<CopiedState>,
    }

    impl MakeMoveHooks for RecordingHooks {
        fn on_key(&mut self, key: Key) {
            self.keys.push(key);
        }
        fn on_state(&mut self, copied: &CopiedState) {
            self.copied.push(*copied);
        }
    }

    // Doc 04 §5: make_move ≡ make_move_with(NoHooks); hook A sees exactly
    // the child's TT probe key (folded on the child's EXACT clock), hook B
    // the final CopiedState.
    #[test]
    fn hooks_seam_is_a_no_op_and_reports_final_keys() {
        let moves = [
            Move::normal(Square::E2, Square::E4),
            Move::normal(Square::D7, Square::D5),
            Move::normal(Square::E4, Square::D5), // capture
            Move::normal(Square::D8, Square::D5), // recapture
            Move::normal(Square::B1, Square::C3),
        ];
        let mut plain = Board::startpos();
        let mut hooked = Board::startpos();
        let mut hooks = RecordingHooks::default();
        for m in moves {
            let gives_check = plain.gives_check(m);
            plain.make_move(m, gives_check);
            hooked.make_move_with(m, gives_check, &mut hooks);
            assert_eq!(hooked, plain, "hooked make diverged after {m}");
            assert_eq!(
                *hooks.keys.last().unwrap(),
                hooked.key(),
                "hook A must see the TT probe key after {m}"
            );
            assert_eq!(
                hooks.copied.last().unwrap(),
                &hooked.states.last().unwrap().copied,
                "hook B must see the final CopiedState after {m}"
            );
        }
        assert_eq!(hooks.keys.len(), moves.len());
        assert_eq!(hooks.copied.len(), moves.len());
        for m in moves.iter().rev() {
            plain.unmake_move(*m);
            hooked.unmake_move(*m);
            assert_eq!(hooked, plain);
        }

        // Fold check on hook A: parent clock 13, quiet non-pawn move →
        // child clock 14, first fold bucket (NOT the parent-clock fold)...
        let mut board = Board::from_fen("4k3/8/8/8/8/8/P7/RN2K3 w - - 13 40").unwrap();
        let mut hooks = RecordingHooks::default();
        let m = Move::normal(Square::B1, Square::C3);
        board.make_move_with(m, false, &mut hooks);
        assert_eq!(hooks.keys[0], board.key());
        assert_eq!(hooks.keys[0], zobrist::fold_rule50(board.raw_key(), 14));
        assert_ne!(hooks.keys[0], board.raw_key());
        board.unmake_move(m);
        // ...while a pawn move resets the child clock: raw key.
        let mut hooks = RecordingHooks::default();
        board.make_move_with(Move::normal(Square::A2, Square::A3), false, &mut hooks);
        assert_eq!(hooks.keys[0], board.raw_key());
        assert_eq!(hooks.keys[0], board.key());
    }

    // Doc 04 §4 step 10: prefetch_key equals the real child key() on normal
    // moves whose parent has no ep square and whose from/to touch no
    // castling rights — including the rule50 fold interplay on both sides
    // of the capture/pawn-move clock reset.
    #[test]
    fn prefetch_key_matches_child_key_on_normal_moves() {
        let cases = [
            // Quiet non-pawn moves: clock below, entering, and inside the
            // fold region (child clock = parent + 1).
            ("4k3/8/8/8/8/8/P7/RN2K3 w - - 0 1", Square::B1, Square::C3),
            ("4k3/8/8/8/8/8/P7/RN2K3 w - - 13 40", Square::B1, Square::C3),
            ("4k3/8/8/8/8/8/P7/RN2K3 w - - 20 40", Square::B1, Square::C3),
            // Capture and pawn push on a mature clock: child key is raw.
            (
                "4k3/8/8/3p4/8/2N5/8/4K3 w - - 20 40",
                Square::C3,
                Square::D5,
            ),
            ("4k3/8/8/8/8/8/P7/RN2K3 w - - 20 40", Square::A2, Square::A3),
            // A capture on a busy position (rights untouched: e5 -> g6).
            (KIWIPETE, Square::E5, Square::G6),
        ];
        for (fen, from, to) in cases {
            let m = Move::normal(from, to);
            let mut board = Board::from_fen(fen).unwrap();
            let predicted = board.prefetch_key(m);
            let gives_check = board.gives_check(m);
            board.make_move(m, gives_check);
            assert_eq!(predicted, board.key(), "prefetch_key({fen}, {m})");
        }
    }

    // -----------------------------------------------------------------
    // Doc 04 §4 step 9 / §5: see_ge — the SEE fixture suite (D11)
    // -----------------------------------------------------------------

    use crate::types::{KNIGHT_VALUE, PAWN_VALUE, QUEEN_VALUE, ROOK_VALUE};

    /// Assert that `m`'s static exchange value in `fen` is exactly `see`:
    /// `see_ge` holds at the value, fails one above, and — the doc 04 §5
    /// threshold-semantics requirement — is monotone non-increasing in the
    /// threshold, with its single true→false step exactly at `see`.
    fn assert_see_value(fen: &str, m: Move, see: Value) {
        let board = Board::from_fen(fen).unwrap();
        assert!(
            board.see_ge(m, see),
            "see_ge({fen}, {m}, {see}) must hold at the exact SEE value"
        );
        assert!(
            !board.see_ge(m, see + 1),
            "see_ge({fen}, {m}, {see} + 1) must fail just above the SEE value"
        );
        for t in (-2600..=2600).step_by(97) {
            assert_eq!(
                board.see_ge(m, t),
                t <= see,
                "see_ge({fen}, {m}, {t}) breaks threshold monotonicity (SEE = {see})"
            );
        }
    }

    // Doc 04 §5 fixture: RxP with no defender — SEE is exactly the pawn.
    #[test]
    fn see_simple_winning_capture() {
        assert_see_value(
            "1k1r4/1pp4p/p7/4p3/8/P5P1/1PP4P/2K1R3 w - - 0 1",
            Move::normal(Square::E1, Square::E5),
            PAWN_VALUE,
        );
    }

    // Doc 04 §5 fixture: NxP on a well-defended square — the knight is lost
    // for the pawn (the deeper R/Q/B/Q exchanges never pay for either side).
    #[test]
    fn see_losing_capture_defended_pawn() {
        assert_see_value(
            "1k1r3q/1ppn3p/p4b2/4p3/8/P2N2P1/1PP1R1BP/2K1Q3 w - - 0 1",
            Move::normal(Square::D3, Square::E5),
            PAWN_VALUE - KNIGHT_VALUE,
        );
    }

    // Doc 04 §5: a defender pinned to its king does not count while its
    // pinner is on the board. Re1 pins Ne7 against the e8 king, so RxP on
    // d5 wins the pawn outright; lift the pinning rook off the board and
    // the knight recaptures, sinking the exchange to P - R.
    #[test]
    fn see_pinned_defender_does_not_count() {
        assert_see_value(
            "4k3/4n3/8/3p4/8/8/8/3RR1K1 w - - 0 1",
            Move::normal(Square::D1, Square::D5),
            PAWN_VALUE,
        );
        // Same position without the pinner: Ne7 counts again.
        assert_see_value(
            "4k3/4n3/8/3p4/8/8/8/3R2K1 w - - 0 1",
            Move::normal(Square::D1, Square::D5),
            PAWN_VALUE - ROOK_VALUE,
        );
    }

    // The pinner-liveness flip side (the `to` toggle of doc 04 §3.11 step
    // 4): Qe7 pins Re4 against the white king, and NxQ captures the pinner
    // itself — the freed rook then guards e7, so Black's Ra7 recapture
    // (RxN, RxR) never pays and the SEE is the whole queen.
    #[test]
    fn see_capturing_the_pinner_frees_the_pinned_piece() {
        assert_see_value(
            "7k/r3q3/8/3N4/4R3/8/8/4K3 w - - 0 1",
            Move::normal(Square::D5, Square::E7),
            QUEEN_VALUE,
        );
    }

    // Doc 04 §5 x-ray case: doubled rooks on both sides of the e-file. The
    // white back rook is revealed through the vacated from-square, the
    // black back rook by lifting the used front rook out of the occupancy;
    // with even batteries the exchange still costs R for P.
    #[test]
    fn see_xray_rook_battery() {
        assert_see_value(
            "1k2r3/4r3/8/4p3/8/8/4R3/4R1K1 w - - 0 1",
            Move::normal(Square::E2, Square::E5),
            PAWN_VALUE - ROOK_VALUE,
        );
    }

    // Diagonal x-rays on both sides: Bg2 sits behind the capturing queen,
    // and Black's Bf7 is revealed when the e6 pawn recaptures. The bishop
    // behind the queen would re-win the pawn, but Bf7 punishes that, so
    // Black recaptures and the SEE is P - Q.
    #[test]
    fn see_xray_bishop_behind_queen() {
        assert_see_value(
            "1k6/5b2/4p3/3p4/8/5Q2/6B1/6K1 w - - 0 1",
            Move::normal(Square::F3, Square::D5),
            PAWN_VALUE - QUEEN_VALUE,
        );
    }

    // Doc 04 §3.11 step 7, king branch: the defending king may only
    // recapture when the attacker has no backup. With Bc3 covering e5 the
    // king recapture is illegal (SEE = P); without it the rook is lost.
    #[test]
    fn see_king_recapture_only_when_unguarded() {
        assert_see_value(
            "8/8/4k3/4p3/8/2B5/8/4R1K1 w - - 0 1",
            Move::normal(Square::E1, Square::E5),
            PAWN_VALUE,
        );
        assert_see_value(
            "8/8/4k3/4p3/8/8/8/4R1K1 w - - 0 1",
            Move::normal(Square::E1, Square::E5),
            PAWN_VALUE - ROOK_VALUE,
        );
    }

    // Doc 04 §5: quiet-move SEE — an empty destination counts 0, so moving
    // to a pawn-covered square hangs the whole knight, and moving to a safe
    // square is exactly 0.
    #[test]
    fn see_quiet_move_to_defended_square() {
        let fen = "4k3/8/2p5/8/3N4/8/8/4K3 w - - 0 1";
        assert_see_value(fen, Move::normal(Square::D4, Square::B5), -KNIGHT_VALUE);
        assert_see_value(fen, Move::normal(Square::D4, Square::F5), 0);
    }

    // Doc 04 §3.11 step 1: promotion/en-passant/castling short-circuit to
    // a static exchange value of 0 — even a queening capture reports 0.
    #[test]
    fn see_non_normal_kinds_short_circuit() {
        let board = Board::from_fen("4k3/8/8/8/4Pp2/8/8/4K3 b - e3 0 1").unwrap();
        let ep = Move::en_passant(Square::F4, Square::E3);
        assert!(board.see_ge(ep, -500));
        assert!(board.see_ge(ep, 0));
        assert!(!board.see_ge(ep, 1));

        let board = Board::from_fen("3q3k/4P3/8/8/8/8/8/4K3 w - - 0 1").unwrap();
        let promo = Move::promotion(Square::E7, Square::D8, PieceType::Queen);
        assert!(board.see_ge(promo, 0));
        assert!(!board.see_ge(promo, 1));

        let board = Board::from_fen("4k3/8/8/8/8/8/8/4K2R w K - 0 1").unwrap();
        let castle = Move::castling(Square::E1, Square::G1);
        assert!(board.see_ge(castle, 0));
        assert!(!board.see_ge(castle, 1));
    }
}
