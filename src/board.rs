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

    /// The raw 4-bit mask (bit 0 = WK, 1 = WQ, 2 = BK, 3 = BQ): the index
    /// into the 16-entry `zobrist` castling table, which keys rights by the
    /// WHOLE mask so any rights change costs two XORs.
    pub const fn bits(self) -> u8 {
        self.0
    }

    /// The single right of `color` on the given wing.
    pub(crate) const fn single(color: Color, kingside: bool) -> CastlingRights {
        match (color, kingside) {
            (Color::White, true) => CastlingRights::WHITE_KINGSIDE,
            (Color::White, false) => CastlingRights::WHITE_QUEENSIDE,
            (Color::Black, true) => CastlingRights::BLACK_KINGSIDE,
            (Color::Black, false) => CastlingRights::BLACK_QUEENSIDE,
        }
    }

    /// `color`'s kingside right.
    pub const fn kingside(color: Color) -> CastlingRights {
        CastlingRights::single(color, true)
    }

    /// `color`'s queenside right.
    pub const fn queenside(color: Color) -> CastlingRights {
        CastlingRights::single(color, false)
    }

    /// Bit position 0..4 of a SINGLE right (WK = 0, WQ = 1, BK = 2, BQ = 3):
    /// the index into `Board`'s castling geometry arrays.
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
/// edited by `make_move` (design/04 §3.2, D8). Plain `Copy` — our safe
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

/// One ply of undo/search state (design/04 §3.2, D8). `Copy` so
/// `make_null_move` can clone the whole record cheaply.
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
    /// The piece that MOVED to produce this state (the pre-promotion mover),
    /// so `unmake_move` does not have to re-read it from the mailbox. Lives
    /// in what was tail padding, so `StateInfo` did not grow. Meaningless at
    /// the root state, where `EMPTY` seeds it and nothing reads it.
    pub moved: Piece,
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
        moved: Piece::new(Color::White, PieceType::Pawn),
        repetition: 0,
    };
}

/// The check bookkeeping (doc 04 §3.5) computed from a placement, before it
/// is stored into a [`StateInfo`]. Exists so the computation can be a
/// `&self` query and the store a separate `&mut` write
/// ([`Board::refresh_check_info`]).
struct CheckInfo {
    blockers_for_king: [Bitboard; 2],
    pinners: [Bitboard; 2],
    check_squares: [Bitboard; 6],
}

impl CheckInfo {
    fn of(board: &Board) -> Self {
        let mut blockers_for_king = [Bitboard::EMPTY; 2];
        let mut pinners = [Bitboard::EMPTY; 2];
        for c in Color::ALL {
            let (b, p) = board.slider_blockers(c);
            blockers_for_king[c as usize] = b;
            pinners[c.flip() as usize] = p;
        }

        let tables = attacks::tables();
        let them = board.side_to_move.flip();
        let ksq = board.king_square(them);
        let occ = board.all_occupancy;
        let bishop = tables.bishop_attacks(ksq, occ);
        let rook = tables.rook_attacks(ksq, occ);
        // Pawn attack symmetry: an enemy pawn ON `ksq` attacks exactly the
        // squares from which our pawns would check `ksq`. The array literal
        // is in `PieceType` discriminant order — pinned by
        // `perft_checked`'s independent `gives_check` oracle.
        let check_squares = [
            attacks::pawn_attacks(them, ksq),
            attacks::knight_attacks(ksq),
            bishop,
            rook,
            bishop | rook,
            Bitboard::EMPTY,
        ];
        Self {
            blockers_for_king,
            pinners,
            check_squares,
        }
    }

    /// Destructured on purpose: adding a field to [`CheckInfo`] without
    /// storing it becomes a compile error instead of a silently
    /// unmaintained field.
    fn store(self, st: &mut StateInfo) {
        let CheckInfo {
            blockers_for_king,
            pinners,
            check_squares,
        } = self;
        st.blockers_for_king = blockers_for_king;
        st.pinners = pinners;
        st.check_squares = check_squares;
    }
}

/// From-scratch key/material computation result — `init_state` applies it;
/// the test/debug oracle compares it against the stored state.
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
/// Lives HERE rather than under `src/eval/nnue/` because its data is a pure
/// board-mutation fact and [`DirtyPiece::from_move`] must read `Board`'s
/// private castling geometry; `eval::nnue` re-exports it, keeping the
/// dependency arrow `eval → board` only. Fields are private; iterate via
/// [`DirtyPiece::adds`] / [`DirtyPiece::subs`].
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
    ///
    /// NOTE: the `.min(len)` is NOT a bounds guard — the constructors cannot
    /// push a third entry (the `debug_assert` states that) — it is a range
    /// fact for the optimizer that removes a bounds-check edge from every
    /// hot NNUE reader. Keep it.
    #[inline]
    pub fn adds(&self) -> &[(Piece, Square)] {
        debug_assert!(self.n_add as usize <= self.adds.len(), "n_add overflow");
        &self.adds[..(self.n_add as usize).min(self.adds.len())]
    }

    /// The (piece, square) pairs a piece leaves after this move.
    #[inline]
    pub fn subs(&self) -> &[(Piece, Square)] {
        debug_assert!(self.n_sub as usize <= self.subs.len(), "n_sub overflow");
        &self.subs[..(self.n_sub as usize).min(self.subs.len())]
    }

    /// `n_add` is only ever incremented from zero inside one constructor
    /// call, so LLVM already proves the bound; the `debug_assert` states
    /// the invariant [`DirtyPiece::adds`]'s `.min` relies on.
    #[inline]
    fn add(&mut self, piece: Piece, sq: Square) {
        debug_assert!((self.n_add as usize) < self.adds.len(), "add cursor");
        self.adds[self.n_add as usize] = (piece, sq);
        self.n_add += 1;
    }

    #[inline]
    fn sub(&mut self, piece: Piece, sq: Square) {
        debug_assert!((self.n_sub as usize) < self.subs.len(), "sub cursor");
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
        let (from, to) = (m.from(), m.to());
        let mover = board
            .piece_on(from)
            .expect("DirtyPiece::from_move: no piece on the from-square");

        match m.kind() {
            // Out of line: castling is under 1% of made moves and the only
            // arm needing the rook-square table.
            MoveKind::Castling => return Self::castling_record(board, us, mover, from, to),
            MoveKind::EnPassant | MoveKind::Promotion => {
                return Self::special_record(board, m, us, mover, from, to);
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

    /// The same record as [`DirtyPiece::from_move`], derived from the
    /// position the move PRODUCED — i.e. from `board` *after*
    /// [`Board::make_move_with`] has run. Reads the mover and victim from
    /// `state.moved`/`state.captured` (written by the make) instead of
    /// re-deriving them from the mailbox. Emission order is `from_move`'s,
    /// exactly; `from_move` is kept in the tree as this function's
    /// independent oracle (`after_move_reproduces_from_move_...`).
    ///
    /// # Panics (debug only)
    ///
    /// Never called for a null move: those bypass `make_move_with` entirely,
    /// so a `debug_assert` stands in for `from_move`'s `is_ok` sentinel arm.
    pub fn after_move(board: &Board, m: Move) -> DirtyPiece {
        debug_assert!(m.is_ok(), "after_move: null moves never reach make_move");
        let mover = board.state.moved;
        let us = mover.color;
        let (from, to) = (m.from(), m.to());
        let mut dp = DirtyPiece::EMPTY;

        match m.kind() {
            // `castling_record` reads only `castling_rook_square`, which no
            // move mutates, so it is valid on either side of the make.
            MoveKind::Castling => return Self::castling_record(board, us, mover, from, to),
            MoveKind::EnPassant | MoveKind::Promotion => {
                return Self::special_record_after(board, m, us, mover, from, to);
            }
            MoveKind::Normal => {
                dp.sub(mover, from);
                if let Some(victim) = board.state.captured {
                    dp.sub(victim, to); // ordinary capture
                }
                dp.add(mover, to);
            }
        }
        dp
    }

    /// The record built from the actors [`Board::make_move_with`] step 3
    /// already holds in registers (`pc`, `(captured, capsq)`, `us`) —
    /// nothing is read back out of the board or the `StateInfo`. Step 3
    /// resolves en passant into `(Some(their pawn), ep square)`, so
    /// `EnPassant` and `Normal` collapse into one straight-line arm.
    /// Emission order is `from_move`'s in every arm (oracle-checked).
    #[inline]
    fn from_actors(
        board: &Board,
        m: Move,
        us: Color,
        pc: Piece,
        captured: Option<Piece>,
        capsq: Square,
    ) -> DirtyPiece {
        let (from, to) = (m.from(), m.to());
        let mut dp = DirtyPiece::EMPTY;
        match m.kind() {
            MoveKind::Castling => return Self::castling_record(board, us, pc, from, to),
            MoveKind::Promotion => return Self::promotion_record(m, us, pc, captured, from, to),
            // En passant folded in: `capsq` is the ep pawn's square and
            // `captured` is that pawn, both resolved by step 3.
            MoveKind::Normal | MoveKind::EnPassant => {
                dp.sub(pc, from);
                if let Some(victim) = captured {
                    dp.sub(victim, capsq);
                }
                dp.add(pc, to);
            }
        }
        dp
    }

    /// The promotion arm of [`DirtyPiece::from_actors`], out of line so only
    /// the arm taken on ~99% of moves stays inline. Promotions carry
    /// `capsq == to`, so the victim square is not a parameter.
    #[cold]
    #[inline(never)]
    fn promotion_record(
        m: Move,
        us: Color,
        pc: Piece,
        captured: Option<Piece>,
        from: Square,
        to: Square,
    ) -> DirtyPiece {
        let promoted = Piece::new(
            us,
            m.promotion_piece().expect("promotion kind carries a piece"),
        );
        let mut dp = DirtyPiece::EMPTY;
        dp.sub(pc, from);
        if let Some(victim) = captured {
            dp.sub(victim, to); // capture-promotion
        }
        dp.add(promoted, to);
        dp
    }

    /// [`DirtyPiece::special_record`] for the post-move board. Separate
    /// because both arms must read the victim from `state.captured`: the
    /// `to` square now holds the promoted piece, so `piece_on(to)` would
    /// mis-report the capture (mutation-tested), and step 3 of the make
    /// already stored the en-passant pawn there. Both `expect`s are
    /// unreachable by construction.
    #[cold]
    #[inline(never)]
    fn special_record_after(
        board: &Board,
        m: Move,
        us: Color,
        mover: Piece,
        from: Square,
        to: Square,
    ) -> DirtyPiece {
        let captured = board.state.captured;
        let mut dp = DirtyPiece::EMPTY;
        if m.kind() == MoveKind::EnPassant {
            // The captured pawn sits one rank from `to`, toward our home.
            let capsq = to.offset(us.flip().pawn_push());
            dp.sub(mover, from);
            dp.sub(captured.expect("en passant always captures"), capsq);
            dp.add(mover, to);
        } else {
            let promoted = Piece::new(
                us,
                m.promotion_piece().expect("promotion kind carries a piece"),
            );
            dp.sub(mover, from);
            if let Some(victim) = captured {
                dp.sub(victim, to); // capture-promotion
            }
            dp.add(promoted, to);
        }
        dp
    }

    /// The en-passant and promotion arms of [`DirtyPiece::from_move`], out of
    /// line for the same reason as [`DirtyPiece::castling_record`].
    #[cold]
    #[inline(never)]
    fn special_record(
        board: &Board,
        m: Move,
        us: Color,
        mover: Piece,
        from: Square,
        to: Square,
    ) -> DirtyPiece {
        // `them` re-derived, not passed — stays within six argument registers.
        let them = us.flip();
        let mut dp = DirtyPiece::EMPTY;
        if m.kind() == MoveKind::EnPassant {
            // The captured pawn sits one rank from `to`, toward our home.
            let capsq = to.offset(them.pawn_push());
            dp.sub(mover, from);
            dp.sub(Piece::new(them, PieceType::Pawn), capsq);
            dp.add(mover, to);
        } else {
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
        dp
    }

    /// The castling arm of [`DirtyPiece::from_move`]. King to G1/C1 (D2); the
    /// rook jumps to F1/D1 — the same geometry as `make_move_with` step 4.
    #[cold]
    #[inline(never)]
    fn castling_record(
        board: &Board,
        us: Color,
        mover: Piece,
        from: Square,
        to: Square,
    ) -> DirtyPiece {
        let mut dp = DirtyPiece::EMPTY;
        let kingside = to.file() > from.file();
        let rook_from = board.castling_rook_square[CastlingRights::single(us, kingside).index()];
        let rook_to = if kingside { Square::F1 } else { Square::D1 }.relative_to(us);
        let rook = Piece::new(us, PieceType::Rook);
        dp.sub(mover, from);
        dp.sub(rook, rook_from);
        dp.add(mover, to);
        dp.add(rook, rook_to);
        dp
    }
}

/// Hook points inside [`Board::make_move_with`] (doc 04 §3.10). The public
/// `make_move(m, gives_check)` signature is pinned (D10), so the TT,
/// history tables and NNUE attach through this trait instead. Monomorphized
/// per hook type with empty `#[inline]` no-op defaults, so the [`NoHooks`]
/// path compiles away entirely.
pub trait MakeMoveHooks {
    /// Does this hook type want Hook C at all? An associated `const`, not a
    /// field, so `make_move_with` monomorphizes the whole record
    /// construction away for hook types that decline (NoHooks, ProbCut's
    /// verification probe) — the seam stays bench-neutral by construction.
    const WANTS_DIRTY: bool = false;

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
    /// accumulator to fold into its feature columns. Only fires when
    /// `WANTS_DIRTY` is true.
    #[inline]
    fn on_dirty(&mut self, _dp: &DirtyPiece) {}
}

/// The default no-op hooks: [`Board::make_move`] delegates through this, and
/// the empty inline bodies monomorphize to nothing.
pub struct NoHooks;

impl MakeMoveHooks for NoHooks {}

/// The Wave-G threat map (D30) — every square attacked by the
/// side-NOT-to-move, unioned (`all`) and split by attacker `PieceType` —
/// held as a LAZY, position-keyed cache rather than a `StateInfo` field:
/// only move ordering and the reverse-futility test read it, and no qsearch
/// node does, so filling on first read makes each node pay at most once.
///
/// Keyed by the RAW position hash, so make/unmake need no invalidation hook
/// and a revisited position reuses the fill. A hash collision (2^-64) can
/// hand a stale ordering hint to a heuristic; it can never affect legality.
///
/// `Clone` deliberately returns a COLD cache and `PartialEq` ignores the
/// contents: two boards are the same position whether or not either has
/// materialised its map.
#[derive(Debug, Default)]
struct ThreatCache {
    /// The `st().key` the fill belongs to; `None` until first use.
    key: std::cell::Cell<Option<Key>>,
    all: std::cell::Cell<Bitboard>,
    by_type: std::cell::Cell<[Bitboard; 6]>,
}

impl Clone for ThreatCache {
    fn clone(&self) -> Self {
        Self::default()
    }
}

impl PartialEq for ThreatCache {
    fn eq(&self, _other: &Self) -> bool {
        true
    }
}

impl Eq for ThreatCache {}

/// The stack of states BELOW the live ply (D8): a preallocated slab plus a
/// cursor, **not** a `Vec` whose length is the cursor.
///
/// NOTE: the shape matters — `Vec::push` takes its argument by value with a
/// real call (`grow`) in the way, forcing a stack round-trip of the whole
/// 168-byte record once per node; `slots[depth] = self.state` is a direct
/// memory-to-memory copy with nothing live across a call.
///
/// The idiom: `slots` is kept at its **full length** (every slot a valid
/// `StateInfo`), and `depth` says how much of it is live. [`Deref`] hands
/// out `&slots[..depth]`, so every reader sees exactly the old `Vec`, and
/// `Clone`/`PartialEq` are written against that slice so the dead tail
/// below the cursor can never leak into equality.
#[derive(Debug)]
struct StateStack {
    /// Backing storage, always fully initialised: `slots.len()` is capacity.
    slots: Vec<StateInfo>,
    /// How many slots are live. Invariant: `depth <= slots.len()`.
    depth: usize,
}

impl StateStack {
    /// Live depth. An INHERENT method, so it wins name resolution over the
    /// `Deref` slice's `len()` and skips building the slice.
    #[inline]
    fn len(&self) -> usize {
        self.depth
    }

    fn new() -> Self {
        // Reserved past MAX_PLY so mid-search pushes never grow the slab.
        StateStack {
            slots: vec![StateInfo::EMPTY; MAX_PLY + 64],
            depth: 0,
        }
    }

    /// The capacity miss: grow the slab, THEN store. Reachable by a game
    /// longer than the reserve plus a full-depth search, so it must exist.
    /// It does the store itself so the hot path keeps a single
    /// successor/predecessor and its bounds test is not re-issued.
    #[cold]
    #[inline(never)]
    fn grow_and_store(&mut self, st: StateInfo) {
        self.slots.resize(self.slots.len() * 2, StateInfo::EMPTY);
        self.slots[self.depth] = st;
    }
}

impl std::ops::Deref for StateStack {
    type Target = [StateInfo];

    #[inline]
    fn deref(&self) -> &[StateInfo] {
        // `depth <= slots.len()` is the type's invariant; `.min` states it
        // in the one form the optimiser can use (a `cmov`, no panic edge on
        // the repetition scan's once-per-node read).
        &self.slots[..self.depth.min(self.slots.len())]
    }
}

impl Clone for StateStack {
    fn clone(&self) -> Self {
        // Clone the whole slab, not just the live prefix: a clone that came
        // back with a short slab would grow on its first deep search.
        StateStack {
            slots: self.slots.clone(),
            depth: self.depth,
        }
    }
}

impl PartialEq for StateStack {
    fn eq(&self, other: &Self) -> bool {
        **self == **other
    }
}

impl Eq for StateStack {}

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

    // The CURRENT ply's state (D8), held INLINE rather than as the top of a
    // stack: a constant field offset, and the "never empty" invariant is
    // structural. Stockfish's `Position::st` pointer minus the pointer.
    state: StateInfo,
    // Everything BELOW `state`, oldest first: `make_move` pushes the parent,
    // `unmake_move` pops it back. Empty at the root.
    history: StateStack,

    /// Lazily-filled Wave-G threat map — see [`ThreatCache`]. Not part of the
    /// position: cloned cold, ignored by equality.
    threats: ThreatCache,

    /// PGO-pipeline null (`--features pgonull69`, default OFF, never ships):
    /// eight inert bytes that shift every later `Board` field offset — hence
    /// instruction encodings, function sizes and PGO inline decisions — with
    /// zero semantic content. Used to null PGO measurement panels.
    #[cfg(feature = "pgonull69")]
    _pgonull69_pad: u64,
}

impl Board {
    pub fn empty() -> Self {
        let history = StateStack::new();
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
            state: StateInfo::EMPTY,
            history,
            threats: ThreatCache::default(),
            #[cfg(feature = "pgonull69")]
            _pgonull69_pad: 0,
        };
        // Even the empty board's root state is self-consistent.
        board.init_state();
        board
    }

    pub fn piece_on(&self, sq: Square) -> Option<Piece> {
        self.mailbox[sq.idx()]
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

    /// The current ply's state (D8) — a field, so this is free.
    fn st(&self) -> &StateInfo {
        &self.state
    }

    fn st_mut(&mut self) -> &mut StateInfo {
        &mut self.state
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

    /// Canonical rule50 accessor for search/TT/pruning callers;
    /// [`Board::halfmove_clock`] is the FEN-facing alias of the same field.
    pub fn rule50_count(&self) -> u32 {
        self.st().copied.rule50 as u32
    }

    /// Plies since the last null move: reset to 0 by
    /// [`Board::make_null_move`], incremented by every real move.
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
    /// 50-move rule. Repetition detection compares [`Board::raw_key`].
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
    /// sniper and `color`'s king. Friendly ones are pinned; enemy ones are
    /// discovered-check candidates.
    pub fn blockers_for_king(&self, color: Color) -> Bitboard {
        self.st().blockers_for_king[color as usize]
    }

    /// Every square attacked by the side NOT to move (Wave-G threat map, D30),
    /// with the side-to-move's king x-rayed out of slider occupancy.
    ///
    /// Filled on first read for this position and cached — see [`ThreatCache`].
    pub fn all_threats(&self) -> Bitboard {
        self.ensure_threats();
        self.threats.all.get()
    }

    /// Fill [`Board::threats`] for the current position unless it already
    /// holds it. `&self` via `Cell` interior mutability; one
    /// compare-and-branch on the already-filled path.
    fn ensure_threats(&self) {
        let key = self.st().key;
        if self.threats.key.get() == Some(key) {
            return;
        }
        let (by_type, all) = self.compute_threats();
        self.threats.by_type.set(by_type);
        self.threats.all.set(all);
        self.threats.key.set(Some(key));
    }

    /// Squares attacked by the side-not-to-move's pieces of type `pt`
    /// (Wave-G threat map, D30). `piece_threats(King)` is the enemy king's
    /// attacks; kings/pawns have no cheaper attacker so the ordering scorer
    /// unions the relevant tiers itself (design/09 §3.7).
    pub fn piece_threats(&self, pt: PieceType) -> Bitboard {
        self.ensure_threats();
        self.threats.by_type.get()[pt as usize]
    }

    /// The WHOLE Wave-G threat map behind ONE fill check: `(all, by_type)`.
    /// Serves move ordering's five reads with a single `ensure_threats`.
    pub fn threat_map(&self) -> (Bitboard, [Bitboard; 6]) {
        self.ensure_threats();
        (self.threats.all.get(), self.threats.by_type.get())
    }

    /// `color`'s sliders pinning a piece of the OTHER color against its own
    /// king: the snipers whose sole blocker is an enemy piece.
    pub fn pinners(&self, color: Color) -> Bitboard {
        self.st().pinners[color as usize]
    }

    /// From-squares from which a `pt` of the side to move would check the
    /// enemy king. EMPTY for `King` — a king never gives check.
    pub fn check_squares(&self, pt: PieceType) -> Bitboard {
        self.st().check_squares[pt as usize]
    }

    /// All six check-square sets in one read — the quiet scorer asks per
    /// scored quiet with `pt` varying, so the hoist is done in the source.
    pub fn check_squares_all(&self) -> &[Bitboard; 6] {
        &self.st().check_squares
    }

    /// The piece captured by the last made move (None at the root state).
    pub fn captured_piece(&self) -> Option<Piece> {
        self.st().captured
    }

    /// `color`'s king square. **Total in release**: a king-less board yields
    /// `a1`; debug trips the `debug_assert` below.
    ///
    /// King-less boards are a supported class (`Board::empty()` +
    /// `put_piece` tests; `from_fen` does not reject them). Callers that can
    /// see an incomplete board must gate on that themselves; a search root
    /// is gated by `uci::is_legal_root` at the `position` boundary.
    pub fn king_square(&self, color: Color) -> Square {
        // `trailing_zeros` of a non-zero u64 is already in 0..64; the `& 63`
        // tells LLVM, so `Square::new`'s assert folds and there is no panic
        // edge on a several-times-per-node query.
        let bb = self.pieces(color, PieceType::King).to_u64();
        debug_assert!(bb != 0, "king must exist");
        Square::new((bb.trailing_zeros() as u8) & 63)
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
        self.castling_rights_mask[sq.idx()]
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
    #[inline(always)]
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
    pub(crate) fn pieces_of_type(&self, piece_type: PieceType) -> Bitboard {
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

    /// Recompute the check bookkeeping (doc 04 §3.5) into the CURRENT state:
    /// blockers/pinners around both kings, plus the squares from which each
    /// piece type of the side to move would check the enemy king. Called at
    /// setup and at the end of every make/null-move, AFTER `side_to_move`
    /// has been flipped. Does NOT touch `checkers` — that is the caller's
    /// job (from scratch at setup, incrementally in `make_move`). Both kings
    /// must exist.
    ///
    /// Computing the block by value ([`CheckInfo::of`]) and storing it
    /// second is the idiomatic answer to the partial-borrow conflict a
    /// `&self` + `&mut self.state` spelling would hit.
    fn refresh_check_info(&mut self) {
        CheckInfo::of(self).store(&mut self.state);
    }

    /// Compute the Wave-G threat map (D30): squares attacked by `them` (the
    /// side NOT to move), by piece type and unioned. The side-to-move's own
    /// king is removed from slider occupancy so a slider's threat extends
    /// THROUGH it (the king cannot shield the square behind it). Reads
    /// `side_to_move`, so only valid AFTER a make/null-move has flipped it.
    /// Called on demand, not per move — see [`ThreatCache`].
    fn compute_threats(&self) -> ([Bitboard; 6], Bitboard) {
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

        let mut by_type = [Bitboard::EMPTY; 6];
        by_type[PieceType::Pawn as usize] = pawns;
        by_type[PieceType::Knight as usize] = knights;
        by_type[PieceType::Bishop as usize] = bishops;
        by_type[PieceType::Rook as usize] = rooks;
        by_type[PieceType::Queen as usize] = queens;
        by_type[PieceType::King as usize] = king;
        (by_type, pawns | knights | bishops | rooks | queens | king)
    }

    /// Does this pseudo-legal move of the side to move give check?
    /// O(1) for the common cases via `check_squares` and
    /// `blockers_for_king` — search calls this once per move and feeds the
    /// answer to [`Board::make_move`].
    // NOTE: `inline(always)` is deliberate — search::<NT>'s two
    // monomorphizations deny LLVM's single-call-site inline bonus, and
    // forcing small helpers inline measured a win. It INVERTS on large
    // bodies (spills); do not extend without measuring.
    #[inline(always)]
    pub fn gives_check(&self, m: Move) -> bool {
        let pc = self
            .piece_on(m.from())
            .expect("gives_check: no piece on the from-square");
        self.gives_check_by(m, pc)
    }

    /// [`Board::gives_check`] with the mover already in hand: the search's
    /// move loop resolves the mover once per iteration, so the mailbox read
    /// inside `gives_check` was duplication, and taking `pc` by value gives
    /// LLVM the enum's `0..=5` range so the `check_squares` index check
    /// folds. Cold call sites (perft, UCI, ProbCut, the singular probe) stay
    /// on the wrapper above.
    pub fn gives_check_by(&self, m: Move, pc: Piece) -> bool {
        let us = self.side_to_move;
        let them = us.flip();
        let (from, to) = (m.from(), m.to());
        debug_assert_eq!(pc.color, us, "gives_check: moving an enemy piece");
        debug_assert_eq!(
            Some(pc),
            self.piece_on(from),
            "gives_check_by: `pc` is not the piece on the from-square"
        );
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

        // Everything above answers a NORMAL move (>95% of made moves); the
        // three special kinds go out of line together.
        if m.kind() == MoveKind::Normal {
            return false;
        }
        self.gives_check_special(m, ksq, from, to)
    }

    /// The promotion / en-passant / castling arms of [`Board::gives_check`].
    /// Same `#[cold]` caveat as [`Board::unmake_special`]: honest for castling
    /// and en passant everywhere, honest for promotions outside a pawn endgame.
    #[cold]
    #[inline(never)]
    fn gives_check_special(&self, m: Move, ksq: Square, from: Square, to: Square) -> bool {
        // `us` re-read rather than passed — stays within six argument regs.
        let us = self.side_to_move;
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
    /// state. The formulas are already Chess960-proof (D10).
    pub(crate) fn set_castling_right(&mut self, color: Color, rook_from: Square) {
        let king_from = Square::E1.relative_to(color);
        let kingside = rook_from.file() > king_from.file();
        let right = CastlingRights::single(color, kingside);
        self.castling_rights_mask[king_from.idx()].add(right);
        self.castling_rights_mask[rook_from.idx()].add(right);
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

    /// From-scratch recompute of the six keys, non-pawn material, and the
    /// check bookkeeping for the current placement plus the staged
    /// game-state fields (mirrors Stockfish's `set_state`). `make_move`
    /// maintains the same quantities incrementally; this recompute is the
    /// drift oracle the debug consistency check compares against.
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
        let both_kings = self.has_both_kings();
        if both_kings {
            let stm = self.side_to_move;
            st.checkers = self.attackers_to(self.king_square(stm), self.all_occupancy)
                & self.occupancy(stm.flip());
        }
        *self.st_mut() = st;
        if both_kings {
            self.refresh_check_info();
        }
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
            let pc = self.mailbox[sq.idx()].expect("occupancy and mailbox in sync");
            let psq = ZOBRIST.psq[pc.index()][sq.idx()];
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

        // Material key: XOR of the count slots psq[pc][8 + i], kings
        // included. Indexed directly rather than via `ZOBRIST.material`:
        // from_fen must stay panic-free on well-formed yet illegal
        // placements (e.g. nine white pawns), so absurd counts are capped.
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

    /// Test/debug oracle: recompute every key and non-pawn material from
    /// scratch and assert they match the stored state — the drift detector
    /// for every incremental XOR in `make_move`.
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
        self.mailbox[sq.idx()] = Some(piece);
    }

    /// Remove and return the piece on `sq`, keeping bitboards and mailbox in sync.
    pub fn remove_piece(&mut self, sq: Square) -> Option<Piece> {
        let piece = self.mailbox[sq.idx()]?;
        self.remove_piece_known(piece, sq);
        Some(piece)
    }

    /// [`Board::remove_piece`] when the caller ALREADY holds the piece —
    /// make/unmake know every piece they touch, so they skip the mailbox
    /// re-read; the `Option`-returning form stays for genuine queries.
    #[inline]
    pub(crate) fn remove_piece_known(&mut self, piece: Piece, sq: Square) {
        debug_assert_eq!(
            self.piece_on(sq),
            Some(piece),
            "remove_piece_known: {piece:?} is not on {sq}"
        );
        let bb = Bitboard::from_square(sq);
        self.pieces[piece.color as usize][piece.piece_type as usize] ^= bb;
        self.color_occupancy[piece.color as usize] ^= bb;
        self.all_occupancy ^= bb;
        self.mailbox[sq.idx()] = None;
    }

    /// Move a piece between squares in one pass: from-bit ^ to-bit on three
    /// bitboards plus two mailbox writes. The internal fast path of
    /// make/unmake — the destination must be empty (captures and promotions
    /// go through `remove_piece`/`put_piece`). Placement-only like its
    /// siblings: Zobrist deltas live in `make_move`, not here (doc 04 §9.1).
    #[inline]
    pub(crate) fn move_piece_known(&mut self, pc: Piece, from: Square, to: Square) {
        debug_assert_eq!(
            self.piece_on(from),
            Some(pc),
            "move_piece_known: {pc:?} is not on {from}"
        );
        debug_assert!(
            self.piece_on(to).is_none(),
            "move_piece to occupied square {to}"
        );
        let from_to = Bitboard::from_square(from) | Bitboard::from_square(to);
        self.pieces[pc.color as usize][pc.piece_type as usize] ^= from_to;
        self.color_occupancy[pc.color as usize] ^= from_to;
        self.all_occupancy ^= from_to;
        self.mailbox[from.idx()] = None;
        self.mailbox[to.idx()] = Some(pc);
    }

    /// Archive the live state onto the stack — [`StateStack`]'s whole reason
    /// for existing. An indexed store rather than a `push` so the 168-byte
    /// record goes straight from one field of `*self` to another with no
    /// stack temporary and no call in the way.
    #[inline]
    fn push_state(&mut self) {
        // The store lives in the `<` arm: the guard IS the bounds proof, so
        // the hot arm is a bare indexed store (see grow_and_store).
        let depth = self.history.depth;
        if depth < self.history.slots.len() {
            self.history.slots[depth] = self.state;
        } else {
            self.history.grow_and_store(self.state);
        }
        self.history.depth = depth + 1;
    }

    /// Restore the parent state. The mirror of [`Board::push_state`].
    #[inline]
    fn pop_state(&mut self) {
        debug_assert!(
            self.history.depth > 0,
            "unmake must have a parent state to restore"
        );
        // ONE edge, not two: the wrapping form lets the single bounds check
        // catch underflow too (a cursor of 0 wraps far out of bounds). What
        // makes either unreachable is the make/unmake pairing.
        let idx = self.history.depth.wrapping_sub(1);
        self.state = self.history.slots[idx];
        self.history.depth = idx;
    }

    pub fn make_move(&mut self, m: Move, gives_check: bool) {
        self.make_move_with(m, gives_check, &mut NoHooks);
    }

    /// Make `m` on the board (doc 04 §3.8, D10). `m` must be legal for the
    /// side to move, and `gives_check` must equal `self.gives_check(m)` —
    /// caller-supplied so quiet moves need no attack query here.
    ///
    /// Builds the child `StateInfo` and pushes it; [`Board::unmake_move`]
    /// pops it. All Zobrist deltas — main key plus the four specialized
    /// families — are applied here in Stockfish's `do_move` order, NOT in
    /// the placement mutators (doc 04 §3.3, §9.1). The new ep square is set
    /// (and hashed) ONLY when an ep capture will actually be legal (D9).
    /// `hooks` receives the child keys mid-make so consumers can overlap
    /// memory prefetches; the [`NoHooks`] instantiation is exactly
    /// `make_move`.
    pub fn make_move_with<H: MakeMoveHooks>(&mut self, m: Move, gives_check: bool, hooks: &mut H) {
        // Step 1: archive the parent and edit the current state IN PLACE
        // into the child (Stockfish's "memcpy up to key", D8). The
        // non-`copied` fields are dead from here until rewritten below —
        // EXCEPT `blockers_for_king`, which step 8's ep-legality test reads
        // as the PARENT's until step 12 overwrites it.
        //
        // Panic-safety: a panic mid-make leaves a structurally-valid but
        // half-built state; survivable only because a worker panic is caught
        // and every `go` re-seeds from a fresh board clone.
        self.push_state();
        let mut k = self.state.key ^ ZOBRIST.side;

        // Step 2: counters.
        self.game_ply += 1;
        self.state.copied.rule50 += 1;
        self.state.copied.plies_from_null += 1;

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
            // Out of line (see `unmake_castling`); the rook key delta comes
            // back so the two XORs stay with the other key updates here.
            let rook_delta = self.make_castling(pc, us, from, to);
            k ^= rook_delta;
            self.state.copied.non_pawn_key[us as usize] ^= rook_delta;
        }

        // Step 5: capture (mutually exclusive with castling).
        if let Some(cap) = captured {
            let cap_psq = ZOBRIST.psq[cap.index()][capsq.idx()];
            if cap.piece_type == PieceType::Pawn {
                self.state.copied.pawn_key ^= cap_psq;
            } else {
                self.state.copied.non_pawn_material[them as usize] -=
                    PIECE_VALUES[cap.piece_type as usize];
                self.state.copied.non_pawn_key[them as usize] ^= cap_psq;
                if matches!(cap.piece_type, PieceType::Knight | PieceType::Bishop) {
                    self.state.copied.minor_key ^= cap_psq;
                }
            }
            k ^= cap_psq;
            // Count-slot removal: slot index = post-removal count. The
            // popcount is still pre-mutation — ALL board edits wait until
            // step 10.
            let count = self.pieces(them, cap.piece_type).count() as usize;
            self.state.copied.material_key ^= ZOBRIST.material(cap, count - 1);
            self.state.copied.rule50 = 0;
        }

        // Step 6: mover key — for castling this is the king pair; for
        // promotions the pawn's `to` key is zero by construction, so no
        // special case (D9). Then clear a stale ep square out of the key.
        k ^= ZOBRIST.psq[pc.index()][from.idx()] ^ ZOBRIST.psq[pc.index()][to.idx()];
        if let Some(ep) = self.state.copied.ep_square.take() {
            k ^= ZOBRIST.en_passant[ep.file() as usize];
        }

        // Step 7: castling-rights revocation. One mask lookup covers king
        // moves, rook moves, and rook captures (doc 04 §3.7); the whole-mask
        // castling key makes any change two XORs (a no-op change cancels).
        let mut touched = self.castling_rights_mask[from.idx()];
        touched.add(self.castling_rights_mask[to.idx()]);
        k ^= ZOBRIST.castling[self.state.copied.castling_rights.bits() as usize];
        self.state.copied.castling_rights.remove(touched);
        k ^= ZOBRIST.castling[self.state.copied.castling_rights.bits() as usize];

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
                    // Step 12 has not run yet, so these are still the
                    // PARENT's blockers — which is exactly what this test
                    // wants (see the comment below).
                    let not_blockers = !self.state.blockers_for_king[them as usize];
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
                        self.state.copied.ep_square = Some(ep);
                        k ^= ZOBRIST.en_passant[ep.file() as usize];
                    }
                }
            } else if m.kind() == MoveKind::Promotion {
                let promo_pt = m.promotion_piece().expect("promotion kind has a piece");
                let promo = Piece::new(us, promo_pt);
                let promo_psq = ZOBRIST.psq[promo.index()][to.idx()];
                // No pawn-clear XOR: psq[pawn][promotion rank] is zero (D9).
                k ^= promo_psq;
                // Count slots: promoted piece pre-add count in, pawn
                // post-removal count out (popcounts still pre-mutation).
                let promo_count = self.pieces(us, promo_pt).count() as usize;
                let pawn_count = self.pieces(us, PieceType::Pawn).count() as usize;
                self.state.copied.material_key ^=
                    ZOBRIST.material(promo, promo_count) ^ ZOBRIST.material(pc, pawn_count - 1);
                self.state.copied.non_pawn_key[us as usize] ^= promo_psq;
                if matches!(promo_pt, PieceType::Knight | PieceType::Bishop) {
                    self.state.copied.minor_key ^= promo_psq;
                }
                self.state.copied.non_pawn_material[us as usize] += PIECE_VALUES[promo_pt as usize];
            }
            // Always: the `to` term is zero on the promotion rank, so this
            // degenerates to a from-only update for promotions.
            self.state.copied.pawn_key ^=
                ZOBRIST.psq[pc.index()][from.idx()] ^ ZOBRIST.psq[pc.index()][to.idx()];
            self.state.copied.rule50 = 0;
        } else {
            let mover_delta =
                ZOBRIST.psq[pc.index()][from.idx()] ^ ZOBRIST.psq[pc.index()][to.idx()];
            self.state.copied.non_pawn_key[us as usize] ^= mover_delta;
            if matches!(pc.piece_type, PieceType::Knight | PieceType::Bishop) {
                self.state.copied.minor_key ^= mover_delta;
            }
        }

        // Step 9: the child key is final. Hook A gets the TT probe key,
        // folded on the child clock, which is already exact here. NOT
        // `fold_rule50_after` — that models "parent clock + 1", wrong after
        // any capture or pawn move.
        self.state.key = k;
        hooks.on_key(zobrist::fold_rule50(k, self.state.copied.rule50));
        // Hook B: the specialized keys are final too.
        hooks.on_state(&self.state.copied);

        // Step 10: board mutation (castling already mutated in step 4).
        if m.kind() != MoveKind::Castling {
            if let Some(cap) = captured {
                self.remove_piece_known(cap, capsq);
            }
            if m.kind() == MoveKind::Promotion {
                let promo_pt = m.promotion_piece().expect("promotion kind has a piece");
                self.remove_piece_known(pc, from);
                self.put_piece(Piece::new(us, promo_pt), to);
            } else {
                self.move_piece_known(pc, from, to);
            }
        }

        // Step 11: captured piece and checkers. Trusting the caller's
        // `gives_check` keeps quiet moves free of any attack query.
        let checkers = if gives_check {
            self.attackers_to(self.king_square(them), self.all_occupancy) & self.occupancy(us)
        } else {
            Bitboard::EMPTY
        };
        self.state.captured = captured;
        self.state.moved = pc;
        self.state.checkers = checkers;
        // Hook C: the placement record, built from step 3's own actors while
        // they are still in registers; placed AFTER step 10 (mutation).
        if H::WANTS_DIRTY {
            hooks.on_dirty(&DirtyPiece::from_actors(self, m, us, pc, captured, capsq));
        }
        debug_assert_eq!(
            gives_check,
            !(self.attackers_to(self.king_square(them), self.all_occupancy) & self.occupancy(us))
                .is_empty(),
            "make_move: caller-supplied gives_check disagrees with the board after {m}"
        );

        // Step 12: flip the side to move, then rebuild the check
        // bookkeeping for the new side.
        self.side_to_move = them;
        self.refresh_check_info();

        // Step 13: repetition scan (doc 04 §3.9). The window is
        // min(rule50, plies_from_null): a capture/pawn move makes earlier
        // positions unreachable, and a null move must isolate the scan on
        // its side of the boundary. RAW keys — rule50 folding is TT-only.
        // `plies_from_null` never exceeds the plies recorded on this stack,
        // so `n - i` cannot underflow (doc 04 §3.3).
        let child_key = self.state.key;
        let end = self
            .state
            .copied
            .rule50
            .min(self.state.copied.plies_from_null);
        let mut repetition = 0;
        if end >= 4 {
            // The child is `self.state`; the ancestor `i` plies back is
            // `history[len - i]`, because the push in step 1 made
            // `history.len()` equal the child's ply index.
            // NOTE: the indexed loop beats a reversed step_by(2) slice
            // iterator here — `end` is small, iterator setup dominates.
            let hist_len = self.history.len();
            let mut i = 4;
            while i <= end {
                let earlier = &self.history[hist_len - i as usize];
                if earlier.key == child_key {
                    // +d: first repeat, d plies back. -d: that occurrence
                    // was itself a repeat — at least threefold.
                    repetition = if earlier.repetition != 0 { -i } else { i };
                    break;
                }
                i += 2;
            }
        }
        self.state.repetition = repetition;

        // Step 14: debug builds run the full invariant oracle at every node;
        // release compiles it out.
        #[cfg(debug_assertions)]
        self.assert_consistent();
    }

    /// Undo the last made move, which must be `m` (doc 04 §3.8).
    /// Deliberately dumb-cheap: restore the placement, pop the state — no
    /// key or check-info work, everything derived lives in the popped
    /// record and the parent state is intact underneath.
    // NOTE: `inline(always)` deliberate — same rationale as `gives_check`.
    #[inline(always)]
    pub fn unmake_move(&mut self, m: Move) {
        // Read the child's fields before the pop overwrites them.
        let captured = self.state.captured;
        let moved = self.state.moved;
        self.pop_state();
        self.side_to_move = self.side_to_move.flip();
        let us = self.side_to_move;
        self.game_ply -= 1;
        let (from, to) = (m.from(), m.to());

        // The NORMAL arm is straight-line; the other three are cold calls
        // (see `unmake_castling`).
        match m.kind() {
            MoveKind::Normal => {
                self.move_piece_known(moved, to, from);
                if let Some(cap) = captured {
                    self.put_piece(cap, to);
                }
            }
            MoveKind::Castling => self.unmake_castling(us, from, to),
            _ => self.unmake_special(m, us, moved, captured),
        }
        #[cfg(debug_assertions)]
        self.assert_consistent();
    }

    /// The castling arm of `make_move_with` step 4, out of line (see
    /// [`Board::unmake_castling`]). Remove both, then place both — ready
    /// for Chess960's overlapping king/rook squares (D10) — and return the
    /// rook's Zobrist `psq` delta for the caller's key updates.
    #[cold]
    #[inline(never)]
    fn make_castling(&mut self, pc: Piece, us: Color, from: Square, to: Square) -> Key {
        let kingside = to.file() > from.file();
        let right = CastlingRights::single(us, kingside);
        let rfrom = self.castling_rook_square[right.index()];
        let rto = if kingside { Square::F1 } else { Square::D1 }.relative_to(us);
        let rook = Piece::new(us, PieceType::Rook);

        self.remove_piece_known(pc, from);
        self.remove_piece_known(rook, rfrom);
        self.put_piece(pc, to);
        self.put_piece(rook, rto);

        ZOBRIST.psq[rook.index()][rfrom.idx()] ^ ZOBRIST.psq[rook.index()][rto.idx()]
    }

    /// The promotion and en-passant arms of [`Board::unmake_move`] — the
    /// exact inverse of `make_move_with` step 10 for those two kinds.
    /// NOTE: `#[cold]` is honest except in promotion-heavy pawn endgames;
    /// if such a régime shows up in measurement, split `Promotion` out.
    #[cold]
    #[inline(never)]
    fn unmake_special(&mut self, m: Move, us: Color, moved: Piece, captured: Option<Piece>) {
        // `from`/`to` re-derived rather than passed: seven arguments would
        // overflow the six SysV registers onto the stack.
        let (from, to) = (m.from(), m.to());
        if m.kind() == MoveKind::Promotion {
            let promo = Piece::new(us, m.promotion_piece().expect("promotion kind has a piece"));
            self.remove_piece_known(promo, to);
            self.put_piece(Piece::new(us, PieceType::Pawn), from);
        } else {
            self.move_piece_known(moved, to, from);
        }
        if let Some(cap) = captured {
            let capsq = if m.kind() == MoveKind::EnPassant {
                to.offset(us.flip().pawn_push())
            } else {
                to
            };
            self.put_piece(cap, capsq);
        }
    }

    /// The castling arm of [`Board::unmake_move`], out of line — the exact
    /// inverse of `make_move_with` step 4: remove both, place both.
    /// Split out because inline it sets `unmake_move`'s register pressure
    /// (seven callee-saved pushes on every unmake) while castling is under
    /// 1% of moves; `#[cold]` also stops LLVM hoisting it back.
    #[cold]
    #[inline(never)]
    fn unmake_castling(&mut self, us: Color, from: Square, to: Square) {
        let kingside = to.file() > from.file();
        let right = CastlingRights::single(us, kingside);
        let rfrom = self.castling_rook_square[right.index()];
        let rto = if kingside { Square::F1 } else { Square::D1 }.relative_to(us);
        // Both pieces are known by construction — this is the inverse of a
        // castling we ourselves made.
        let king = Piece::new(us, PieceType::King);
        let rook = Piece::new(us, PieceType::Rook);
        self.remove_piece_known(king, to);
        self.remove_piece_known(rook, rto);
        self.put_piece(king, from);
        self.put_piece(rook, rfrom);
    }

    /// Switch the side to move without moving anything (mirrors SF's
    /// `do_null_move`) — the primitive behind null-move pruning. The side
    /// to move must NOT be in check.
    ///
    /// Copies the ENTIRE top state (why `StateInfo: Copy`), then: ep
    /// cleared (with its key XOR), side key toggled, `plies_from_null`
    /// reset to 0 (isolates repetition scans across the null boundary),
    /// check bookkeeping rebuilt. `rule50` and `game_ply` are deliberately
    /// unchanged (SF behavior).
    #[cfg_attr(feature = "icold67", inline(never))]
    pub fn make_null_move(&mut self) {
        debug_assert!(self.checkers().is_empty(), "make_null_move while in check");
        // Same shape as `make_move_with` step 1: archive the parent, then
        // edit the live state in place.
        self.push_state();
        let st = &mut self.state;
        if let Some(ep) = st.copied.ep_square.take() {
            st.key ^= ZOBRIST.en_passant[ep.file() as usize];
        }
        st.key ^= ZOBRIST.side;
        st.copied.plies_from_null = 0;
        st.captured = None;
        st.repetition = 0;
        self.side_to_move = self.side_to_move.flip();
        // `state.checkers` keeps the carried EMPTY set: neither side can be
        // in check across a null move (precondition + attack symmetry).
        self.refresh_check_info();
        #[cfg(debug_assertions)]
        self.assert_consistent();
    }

    /// Undo the last [`Board::make_null_move`]: pop the state, flip back.
    #[cfg_attr(feature = "icold67", inline(never))]
    pub fn unmake_null_move(&mut self) {
        debug_assert!(
            self.checkers().is_empty(),
            "unmake_null_move while in check"
        );
        self.pop_state();
        self.side_to_move = self.side_to_move.flip();
        #[cfg(debug_assertions)]
        self.assert_consistent();
    }

    /// Approximate child key for `m` BEFORE making it (mirrors SF's
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
            ^ ZOBRIST.psq[pc.index()][m.from().idx()]
            ^ ZOBRIST.psq[pc.index()][m.to().idx()];
        if let Some(cap) = captured {
            k ^= ZOBRIST.psq[cap.index()][m.to().idx()];
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
        // checkmate outranks the clock.
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
        let end = copied.rule50.min(copied.plies_from_null);
        if end < 4 {
            return false;
        }
        // The current state, then `end - 4` ancestors, newest first — the
        // window shrinks by one per step back. `end <= plies_from_null <=
        // history.len()`, so `take` never runs off the front.
        self.state.repetition != 0
            || self
                .history
                .iter()
                .rev()
                .take((end - 4) as usize)
                .any(|s| s.repetition != 0)
    }

    /// Can the side to move reach, in ONE move, a position already on the
    /// path — i.e. is a draw by repetition available right now?
    ///
    /// Marcel van Kervinck's cuckoo trick (mirrors SF's `has_game_cycle`).
    /// Every reversible non-pawn move `s1 -> s2` shifts the key by exactly
    /// `psq[pc][s1] ^ psq[pc][s2] ^ side`, independent of the rest of the
    /// board, so "is the difference between two path keys a single
    /// reversible move?" is a two-probe table lookup ([`crate::cuckoo`]) —
    /// no move generation at all. Search calls this before its other early
    /// exits: a side that can force a draw need never search a losing
    /// subtree.
    pub fn upcoming_repetition(&self, ply: i32) -> bool {
        let copied = &self.st().copied;
        // Same window as the repetition scan: an irreversible move or a null
        // makes anything beyond it unreachable.
        let end = copied.rule50.min(copied.plies_from_null);
        if end < 3 {
            return false;
        }

        let original_key = self.st().key;
        let occupancy = self.all_occupancy;
        let tables = crate::cuckoo::tables();
        // `end <= plies_from_null <= history.len()`, so this cannot underflow.
        let hist_len = self.history.len();

        // Only ODD distances can close a cycle — the same side must be to
        // move at both ends, so we step two plies back per iteration.
        for i in (3..=end).step_by(2) {
            let earlier = &self.history[hist_len - i as usize];
            let Some(mv) = tables.lookup(original_key ^ earlier.key) else {
                continue;
            };
            let (s1, s2) = (mv.from(), mv.to());
            // The key delta proves "some reversible move" but says nothing
            // about today's occupancy: a slider's path must still be clear.
            // `between` includes `s2`, so XOR it back out.
            let blockers = (attacks::between(s1, s2) ^ Bitboard::from_square(s2)) & occupancy;
            if !blockers.is_empty() {
                continue;
            }

            // Strictly inside the tree, merely reaching a path position is
            // enough: the opponent had a chance to deviate earlier, so this
            // is a real cycle the side to move can steer into.
            if ply > i {
                return true;
            }

            // At or above the root that is too weak — the "cycle" could be
            // the move that simply arrives at the current position. Require
            // a genuine repetition, and require the mover to be US. The
            // table stores each unordered pair once (s1 < s2), so the piece
            // may sit at either end; pick whichever endpoint is occupied.
            let mover = if self.piece_on(s1).is_some() { s1 } else { s2 };
            // An empty pair simply fails this test instead of reading garbage.
            if self.piece_on(mover).map(|pc| pc.color) != Some(self.side_to_move) {
                continue;
            }
            if earlier.repetition != 0 {
                return true;
            }
        }
        false
    }

    /// True iff the static exchange value of `m` is at least `threshold`
    /// (doc 04 §3.11, D11): both sides keep recapturing on the destination
    /// with their least valuable attacker for as long as it pays, and the
    /// mover's final balance must reach the bar. Null-window swap algorithm:
    /// `swap` carries the running balance folded into the threshold, `res`
    /// flips with every recapture, and each capturer only has to "clear the
    /// bar" `res` (0 or 1) instead of computing an exact exchange value.
    ///
    /// Pinned pieces do not count as attackers while an enemy pinner is
    /// still on the board; x-ray batteries are walked by lifting each used
    /// capturer out of `occupied` and re-querying the slider rays.
    pub fn see_ge(&self, m: Move, threshold: Value) -> bool {
        // Non-Normal kinds short-circuit to a static exchange value of 0.
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
        // move captures a pinner.
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
            // initial victim), formerly pinned pieces participate freely —
            // an approximation shared with Stockfish.
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
                // the bar (the previous capturer was worth at most a queen)
                // — assert instead of test.
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
    /// it at every node in debug builds; release compiles it out.
    ///
    /// Position-level checks assume a playable position; incomplete boards
    /// get only the placement-layer and stack checks.
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

        // --- State stack: sane counters -----------------------------------
        let st = self.st();
        assert!(st.copied.rule50 >= 0);
        assert!(st.copied.plies_from_null >= 0);
        // The repetition scan indexes `plies_from_null` states back.
        assert!(
            st.copied.plies_from_null <= self.history.len() as i32,
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
        // Seeded from EMPTY, NOT from `*st`: a `*st` seed would let the
        // oracle pass if `CheckInfo::store` stopped writing a field.
        let mut expected = StateInfo::EMPTY;
        CheckInfo::of(self).store(&mut expected);
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

    /// Independent oracle for the threat map, computed per square via
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
        // The setwise-filled `all_threats` equals the per-square oracle
        // (incl. slider x-ray through the king); per-type maps union to it.
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

    /// Every board here is FRESH (cold `ThreatCache`) and `threat_map` is
    /// the FIRST threat query, so a dropped `ensure_threats()` fill fails;
    /// also pins agreement with the per-type accessors.
    #[test]
    fn threat_map_on_a_cold_cache_agrees_with_the_per_type_accessors() {
        let fens = [
            crate::fen::START_FEN,
            "r3k2r/p1ppqpb1/bn2pnp1/3PN3/1p2P3/2N2Q1p/PPPBBPPP/R3K2R w KQkq - 0 1",
            "6k1/8/8/8/3q4/8/8/4K3 b - - 0 1",
        ];
        for fen in fens {
            let cold = Board::from_fen(fen).unwrap();
            let (all, by_type) = cold.threat_map();
            assert_eq!(all, threat_oracle(&cold), "threat_map().0 for {fen}");
            assert_ne!(all, Bitboard::EMPTY, "no position here is threat-free");

            let warm = Board::from_fen(fen).unwrap();
            for pt in PieceType::ALL {
                assert_eq!(
                    by_type[pt as usize],
                    warm.piece_threats(pt),
                    "threat_map().1[{pt:?}] for {fen}"
                );
            }
            assert_eq!(all, warm.all_threats(), "all_threats agreement for {fen}");
        }
    }

    /// A stale, zeroed or mis-ordered `check_squares_all` fails against the
    /// per-type accessor, on positions where five of six sets are non-empty.
    #[test]
    fn check_squares_all_agrees_with_the_per_type_accessor() {
        for fen in [
            crate::fen::START_FEN,
            "r3k2r/p1ppqpb1/bn2pnp1/3PN3/1p2P3/2N2Q1p/PPPBBPPP/R3K2R w KQkq - 0 1",
            // Open king diagonal AND file, so no two of the six sets
            // coincide and a swapped index cannot pass.
            "4k3/8/8/8/8/8/8/4K2R w K - 0 1",
        ] {
            let b = Board::from_fen(fen).unwrap();
            let all = b.check_squares_all();
            let mut nonempty = 0;
            for pt in PieceType::ALL {
                assert_eq!(
                    all[pt as usize],
                    b.check_squares(pt),
                    "check_squares_all[{pt:?}] for {fen}"
                );
                nonempty += usize::from(!all[pt as usize].is_empty());
            }
            // Exactly five: King is always EMPTY.
            assert_eq!(
                nonempty, 5,
                "{fen} must exercise five piece types, got {nonempty}"
            );
            // No two of the six may coincide on the last fixture.
            if fen.starts_with("4k3") {
                for a in 0..6 {
                    for b2 in (a + 1)..5 {
                        assert_ne!(all[a], all[b2], "{fen}: sets {a} and {b2} coincide");
                    }
                }
            }
        }
    }

    /// `king_square`'s `& 63` mask is the part that can be wrong (`& 62`
    /// still compiles and returns plausible squares), so pin the exact
    /// square over the whole index range, both colors.
    #[test]
    fn king_square_is_exact_across_the_index_range() {
        for i in 0..64u8 {
            let sq = Square::new(i);
            let mut b = Board::empty();
            b.put_piece(Piece::new(Color::White, PieceType::King), sq);
            assert_eq!(b.king_square(Color::White), sq, "white king on {sq}");
            let mut b = Board::empty();
            b.put_piece(Piece::new(Color::Black, PieceType::King), sq);
            assert_eq!(b.king_square(Color::Black), sq, "black king on {sq}");
        }
        // The king-less class (debug would trip the assert; shape only).
        let empty = Board::empty();
        assert!(empty.pieces(Color::White, PieceType::King).is_empty());
    }

    #[test]
    fn threat_map_survives_make_unmake() {
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

    #[test]
    fn castling_rights_bits_roundtrip() {
        assert_eq!(CastlingRights::NONE.bits(), 0b0000);
        assert_eq!(CastlingRights::WHITE_KINGSIDE.bits(), 0b0001);
        assert_eq!(CastlingRights::WHITE_QUEENSIDE.bits(), 0b0010);
        assert_eq!(CastlingRights::BLACK_KINGSIDE.bits(), 0b0100);
        assert_eq!(CastlingRights::BLACK_QUEENSIDE.bits(), 0b1000);
        assert_eq!(CastlingRights::ALL.bits(), 0b1111);

        // Every 4-bit mask is reachable by composing single rights.
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

    // StateInfo stays Copy (make_null_move clones the whole record) and its
    // size stays pinned, so field creep gets noticed.
    #[test]
    fn state_info_is_copy_and_bounded() {
        fn assert_copy<T: Copy>() {}
        assert_copy::<CopiedState>();
        assert_copy::<StateInfo>();
        // Pinned EXACTLY, not bounded: every make/unmake copies this record,
        // so a new field is real per-node cost a loose bound would absorb.
        const { assert!(size_of::<StateInfo>() == 168) }
    }

    // make_move performs no heap allocation: the state stack is pre-reserved
    // past MAX_PLY, and the slab's LENGTH is the no-reallocation observable.
    #[test]
    fn state_stack_is_prereserved_and_never_reallocates() {
        let mut board = Board::startpos();
        let cap = board.history.slots.len();
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
        assert!(board.history.len() > MAX_PLY - 11);
        assert_eq!(
            board.history.slots.len(),
            cap,
            "make_move/make_null_move must never grow the state slab"
        );
        assert_eq!(
            board.history.depth,
            board.history.len(),
            "the cursor and the exposed slice must agree"
        );
    }

    // Equality and cloning must read through the cursor: a made-and-unmade
    // move's dead record in `slots[depth]` must not be observable.
    #[test]
    fn state_stack_ignores_the_dead_tail_below_the_cursor() {
        let mut board = Board::startpos();
        let pristine = board.clone();
        let m = Move::normal(Square::E2, Square::E4);
        board.make_move(m, false);
        board.unmake_move(m);

        assert_eq!(board.history.depth, 0, "the cursor is back at the root");
        assert_ne!(
            board.history.slots[0], pristine.history.slots[0],
            "the dead record really is still in the slab (else this test is vacuous)"
        );
        assert_eq!(board, pristine, "the dead record must not affect equality");
        assert_eq!(
            board.clone(),
            pristine,
            "nor survive a clone into an observable difference"
        );
    }

    // The root state exists from construction; even the empty board's keys
    // are the documented baselines.
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

    // Board::clone clones the state history — per-thread copies are
    // fully independent.
    #[test]
    fn clone_is_deep_and_equal() {
        let board = Board::startpos();
        let mut clone = board.clone();
        assert_eq!(clone, board);
        assert_eq!(clone.key(), board.key());

        // Diverge the clone through the real mutation path; the original
        // must be untouched.
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

    // Castling geometry (rook squares, paths, revocation masks) at setup.
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

    #[test]
    fn castling_rights_single_bit_index() {
        assert_eq!(CastlingRights::WHITE_KINGSIDE.index(), 0);
        assert_eq!(CastlingRights::WHITE_QUEENSIDE.index(), 1);
        assert_eq!(CastlingRights::BLACK_KINGSIDE.index(), 2);
        assert_eq!(CastlingRights::BLACK_QUEENSIDE.index(), 3);
    }

    // --- attackers_to, slider_blockers, check info --------------------

    // One busy square attacked by every piece type of both colors.
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

    // Lifting a blocker out of `occupied` exposes the x-ray piece behind
    // it (the SEE battery walk).
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
        let board = Board::from_fen("4k3/8/3N4/8/8/8/8/4RK2 b - - 0 1").unwrap();
        assert_eq!(board.checkers(), bb(&[Square::D6, Square::E1]));
        assert!(Board::startpos().checkers().is_empty());
    }

    // Friendly blocker: pinned piece in blockers_for_king(c), sniper in
    // pinners(!c).
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

    // Enemy blocker: a discovered-check candidate — in blockers_for_king
    // but its sniper is NOT a pinner.
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

    // Stacked snipers (queen behind rook): BOTH register as pinners of the
    // sole blocker (the SF convention).
    #[test]
    fn slider_blockers_xray_through_stacked_snipers() {
        let board = Board::from_fen("4q2k/4r3/8/8/8/8/4B3/4K3 w - - 0 1").unwrap();
        assert_eq!(board.blockers_for_king(Color::White), bb(&[Square::E2]));
        assert_eq!(board.pinners(Color::Black), bb(&[Square::E7, Square::E8]));
    }

    // Mirror orientation: white sniper pinning a black piece.
    #[test]
    fn slider_blockers_white_pins_black() {
        let board = Board::from_fen("4k3/8/4n3/8/8/8/4Q3/6K1 b - - 0 1").unwrap();
        assert_eq!(board.blockers_for_king(Color::Black), bb(&[Square::E6]));
        assert_eq!(board.pinners(Color::White), bb(&[Square::E2]));
        assert!(board.checkers().is_empty());
    }

    // check_squares: from-squares that would check the ENEMY king.
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

    // With Black to move the target is the WHITE king (pawn symmetry).
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

    // --- make_move / unmake_move + gives_check ------------------------

    const KIWIPETE: &str = "r3k2r/p1ppqpb1/bn2pnp1/3PN3/1p2P3/2N2Q1p/PPPBBPPP/R3K2R w KQkq - 0 1";

    /// Make `m` with full cross-checking: the gives_check contract, the
    /// key oracle, layer sync, and a recomputed checkers set.
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

    /// Play `moves` in order (each cross-checked), then unmake in reverse,
    /// asserting whole-board equality at every ply. Returns the final
    /// position for post-move assertions.
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
        // Normal moves, double pushes, four captures, kingside castling.
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
        // Double push, ep capture, castling, capturing under-promotion.
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

    // The new ep square is set (and keyed) ONLY when capturable.
    #[test]
    fn double_push_sets_ep_only_when_capturable() {
        // No adjacent enemy pawn: no ep square; hashes like the "-" FEN.
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
                ^ ZOBRIST.psq[wp][Square::E2.idx()]
                ^ ZOBRIST.psq[wp][Square::E4.idx()]
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

        // Horizontal both-pawns-removed pin: fxe6 would clear BOTH e5 and
        // f5 off rank 5. Only the PREVIOUS state's blockers see this.
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
        // capsq != to: unmake must restore the pawn to its own square.
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

    // Rights revocation via the mask — king move, each rook move, and RxR
    // on a home square (from-mask and to-mask fire together).
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

    // Every case is ALSO cross-checked against make_move + recomputed
    // checkers inside roundtrip.
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
        // The BISHOP half of the same arm: exd6 e.p. clears d5 and the
        // a2-g8 diagonal opens onto the black king.
        assert!(check(
            "6k1/8/8/3pP3/8/8/B7/4K3 w - d6 0 1",
            Move::en_passant(Square::E5, Square::D6)
        ));
        // Same board, king one file over and off the diagonal: no check.
        assert!(!check(
            "7k/8/8/3pP3/8/8/B7/4K3 w - d6 0 1",
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

    // Two move orders reaching the same position produce identical keys; a
    // knight shuffle returning home restores every key.
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

    // --- null moves, repetition, draws --------------------------------

    /// Four plies of knight shuffling that land back on the same position.
    fn knight_shuffle() -> [Move; 4] {
        [
            Move::normal(Square::G1, Square::F3),
            Move::normal(Square::G8, Square::F6),
            Move::normal(Square::F3, Square::G1),
            Move::normal(Square::F6, Square::G8),
        ]
    }

    // Null move: key = old ^ side ^ (ep key if any); rule50 and game_ply
    // hold; plies_from_null resets; unmake restores full equality.
    #[test]
    fn null_move_key_toggle_and_roundtrip() {
        // No ep square: the raw key toggles exactly the side key.
        let mut board = Board::startpos();
        make_checked(&mut board, Move::normal(Square::G1, Square::F3));
        let before = board.clone();
        assert_eq!(before.state.copied.plies_from_null, 1);
        board.make_null_move();
        board.assert_consistent();
        assert_eq!(board.raw_key(), before.raw_key() ^ ZOBRIST.side);
        assert_eq!(board.side_to_move(), Color::White);
        assert_eq!(board.rule50_count(), before.rule50_count());
        assert_eq!(board.game_ply(), before.game_ply());
        assert_eq!(board.state.copied.plies_from_null, 0);
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

    // `has_repeated` must find a repetition that is NOT the current state —
    // i.e. actually walk `history`. Every other test short-circuits on
    // `repetition != 0` or `end < 4`, so without this one the ancestor walk
    // is dead to the suite (mutation-verified: `.take(0)` and dropping
    // `.rev()` each fail only here).
    #[test]
    fn has_repeated_walks_past_the_current_state_into_history() {
        let mut board = Board::startpos();
        for m in knight_shuffle() {
            make_checked(&mut board, m);
        }
        // The shuffle's last ply IS the repetition (short-circuit case).
        assert_eq!(board.state.repetition, 4);

        // One more reversible move: the current state is fresh, so the ONLY
        // way to answer `true` is the ancestor one ply back.
        make_checked(&mut board, Move::normal(Square::B1, Square::C3));
        assert_eq!(
            board.state.repetition, 0,
            "the new position is not itself a repeat"
        );
        let copied = &board.state.copied;
        assert_eq!(
            copied.rule50.min(copied.plies_from_null),
            5,
            "the window must reach exactly one ancestor past the current state"
        );
        assert!(
            board.has_repeated(),
            "has_repeated must find the repetition one ply back in history"
        );

        // Negative control: an irreversible move slams the window shut.
        let mut fresh = Board::startpos();
        for m in knight_shuffle() {
            make_checked(&mut fresh, m);
        }
        make_checked(&mut fresh, Move::normal(Square::E2, Square::E4));
        assert!(
            !fresh.has_repeated(),
            "a pawn move resets rule50, so the window cannot reach the repeat"
        );
    }

    // The shuffle produces +d, then -d; is_draw is ply-relative (in-tree
    // twofold draws, root-side twofold does not, threefold anywhere).
    #[test]
    fn repetition_encoding_and_ply_relative_draw() {
        let mut board = Board::startpos();
        for m in knight_shuffle() {
            assert!(!board.has_repeated());
            make_checked(&mut board, m);
        }
        // First return: +distance.
        assert_eq!(board.state.repetition, 4);
        assert_eq!(board.raw_key(), Board::startpos().raw_key());
        assert!(board.has_repeated());
        // In-tree (repetition < ply) draws; at the boundary or root not yet.
        assert!(board.is_repetition(5));
        assert!(board.is_draw(5));
        assert!(!board.is_repetition(4));
        assert!(!board.is_draw(4));
        assert!(!board.is_draw(0));

        // Second return: -distance (at least threefold) — a draw at ANY ply.
        for m in knight_shuffle() {
            make_checked(&mut board, m);
        }
        assert_eq!(board.state.repetition, -4);
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

    #[test]
    fn upcoming_repetition_sees_the_move_back_into_the_cycle() {
        let mut board = Board::startpos();
        // Nothing on the path yet: the window is shorter than a cycle.
        assert!(!board.upcoming_repetition(64));
        for m in knight_shuffle() {
            make_checked(&mut board, m);
        }
        // Back at the start after 4 reversible plies. Playing Ng1f3 AGAIN
        // would reach the position 3 plies back, so a draw is available.
        assert!(
            board.upcoming_repetition(64),
            "Ng1f3 re-enters the cycle at distance 3"
        );
        // Root-adjacent plies are stricter: the move must close a real
        // repetition, not merely arrive here.
        assert!(!board.upcoming_repetition(0));
    }

    // A slider whose path is now blocked must NOT count.
    #[test]
    fn upcoming_repetition_requires_a_clear_slider_path() {
        let mut board = Board::from_fen("4k3/8/8/8/8/8/8/R3K2R w KQ - 0 1").unwrap();
        for m in [
            Move::normal(Square::A1, Square::A4),
            Move::normal(Square::E8, Square::D8),
            Move::normal(Square::A4, Square::A1),
            Move::normal(Square::D8, Square::E8),
        ] {
            make_checked(&mut board, m);
        }
        // Ra1a4 re-enters the cycle and the file is empty.
        assert!(board.upcoming_repetition(64));

        // A black rook on a3 blocks the a-file: the a1-a4 delta is in the
        // table yet unreachable.
        let mut blocked = Board::from_fen("4k3/8/8/8/8/r7/8/R3K2R w KQ - 0 1").unwrap();
        for m in [
            Move::normal(Square::H1, Square::H4),
            Move::normal(Square::E8, Square::D8),
            Move::normal(Square::H4, Square::H1),
            Move::normal(Square::D8, Square::E8),
        ] {
            make_checked(&mut blocked, m);
        }
        // Rh1h4 is still clear, so this position DOES have a cycle; the
        // randomized oracle below pins the occupancy tracking properly.
        assert!(blocked.upcoming_repetition(64));
    }

    /// INDEPENDENT oracle: make every legal move and ask whether the key
    /// already sits on the path — never touches the cuckoo table. If the
    /// oracle finds a legal cycling move the query MUST find it (a miss is
    /// a lost draw); the converse may differ, since the query skips full
    /// legality (pins), so those are counted, not asserted.
    #[test]
    fn upcoming_repetition_matches_a_brute_force_legal_move_oracle() {
        use crate::moves::MoveList;

        // Same tiny SplitMix64 the other modules' tests use.
        struct SplitMix64(u64);
        impl SplitMix64 {
            fn next_u64(&mut self) -> u64 {
                self.0 = self.0.wrapping_add(0x9E37_79B9_7F4A_7C15);
                let mut z = self.0;
                z = (z ^ (z >> 30)).wrapping_mul(0xBF58_476D_1CE4_E5B9);
                z = (z ^ (z >> 27)).wrapping_mul(0x94D0_49BB_1331_11EB);
                z ^ (z >> 31)
            }
        }

        fn oracle(board: &mut Board) -> bool {
            let copied = &board.st().copied;
            let end = copied.rule50.min(copied.plies_from_null);
            if end < 3 {
                return false;
            }
            let hist_len = board.history.len();
            let ancestors: Vec<Key> = (3..=end)
                .step_by(2)
                .map(|i| board.history[hist_len - i as usize].key)
                .collect();
            let mut list = MoveList::new();
            crate::movegen::generate_legal(board, &mut list);
            let moves: Vec<Move> = list.iter().copied().collect();
            for m in moves {
                let gives_check = board.gives_check(m);
                board.make_move(m, gives_check);
                let reached = board.st().key;
                board.unmake_move(m);
                if ancestors.contains(&reached) {
                    return true;
                }
            }
            false
        }

        let mut rng = SplitMix64(0x0DDB_A11C_0FFE_E123);
        let mut boards_seen = 0usize;
        let mut oracle_hits = 0usize;
        let mut query_only = 0usize;

        // Short random games from reversible-heavy starts, so cycles occur.
        for start in [
            "4k3/8/8/8/8/8/8/R3K2R w KQ - 0 1",
            "r3k2r/8/8/8/8/8/8/R3K2R w KQkq - 0 1",
            "4k3/6n1/8/8/8/8/1N6/4K3 w - - 0 1",
            "2bqkb2/8/8/8/8/8/8/2BQKB2 w - - 0 1",
        ] {
            for _ in 0..60 {
                let mut board = Board::from_fen(start).unwrap();
                for _ in 0..24 {
                    let mut list = MoveList::new();
                    crate::movegen::generate_legal(&board, &mut list);
                    if list.is_empty() {
                        break;
                    }
                    // Compare BEFORE moving on: ply is well inside the tree.
                    let query = board.upcoming_repetition(64);
                    let truth = oracle(&mut board);
                    boards_seen += 1;
                    if truth {
                        oracle_hits += 1;
                        assert!(query, "missed a legal cycling move in {}", board.to_fen());
                    } else if query {
                        query_only += 1;
                    }
                    let m = list[(rng.next_u64() % list.len() as u64) as usize];
                    let gives_check = board.gives_check(m);
                    board.make_move(m, gives_check);
                }
            }
        }
        // A statistical test should be able to show its own power.
        eprintln!(
            "upcoming_repetition census: {boards_seen} positions, \
             {oracle_hits} legal cycles (all found), {query_only} legality-only"
        );
        // The suite is only meaningful if cycles actually showed up.
        assert!(
            oracle_hits > 50,
            "oracle found only {oracle_hits} cycles in {boards_seen} positions — \
             the corpus is not exercising the query"
        );
        // Legality-only disagreements should be a small tail, not the norm.
        assert!(
            query_only * 4 < oracle_hits,
            "query fired without a legal cycling move {query_only} times against \
             {oracle_hits} real cycles — that is too many to be pin/legality noise"
        );
    }

    // The scan window is min(rule50, plies_from_null) — a null move cuts
    // it even when the clock alone would allow the match.
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
        // The position genuinely equals the root, but positions on the far
        // side of a null are unreachable by real play: the scan is clipped.
        assert_eq!(board.raw_key(), root_key);
        assert_eq!(board.state.repetition, 0);
        assert!(!board.has_repeated());
        assert!(!board.is_draw(4));
    }

    // A capture zeroes rule50, cutting the window — earlier repeats stop
    // counting, later ones are detected as usual.
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
        assert_eq!(board.state.repetition, 4);
        assert!(board.has_repeated());

        // RxP: irreversible — the accumulated repeat is out of the window.
        make_checked(&mut board, Move::normal(Square::H1, Square::H6));
        assert_eq!(board.state.repetition, 0);
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
        assert_eq!(board.state.repetition, 4);
        assert!(board.is_draw(5));
        assert!(!board.is_draw(0));
    }

    // Rule50 folding never hides a repetition — the scan compares RAW keys.
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
        assert_eq!(board.state.repetition, 4);
        assert!(board.is_draw(5));
    }

    // rule50 > 99 draws at any ply — unless the side to move is checkmated
    // (a mating move on the 100th halfmove wins).
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

    // --- the oracle validates the validator ---------------------------

    #[test]
    #[should_panic(expected = "main key drifted")]
    fn assert_consistent_catches_key_corruption() {
        let mut board = Board::startpos();
        board.state.key ^= 1;
        board.assert_consistent();
    }

    #[test]
    #[should_panic(expected = "pawn_key drifted")]
    fn assert_consistent_catches_pawn_key_corruption() {
        let mut board = Board::startpos();
        board.state.copied.pawn_key ^= 1;
        board.assert_consistent();
    }

    #[test]
    #[should_panic(expected = "checkers drifted")]
    fn assert_consistent_catches_checkers_corruption() {
        let mut board = Board::startpos();
        board.state.checkers = Bitboard::from_square(Square::E4);
        board.assert_consistent();
    }

    #[test]
    #[should_panic(expected = "blockers_for_king drifted")]
    fn assert_consistent_catches_blockers_corruption() {
        let mut board = Board::startpos();
        board.state.blockers_for_king[Color::White as usize] = Bitboard::from_square(Square::E2);
        board.assert_consistent();
    }

    // --- hook seam + prefetch_key -------------------------------------

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

    // make_move ≡ make_move_with(NoHooks); hook A sees exactly the child's
    // TT probe key (folded on the child's EXACT clock), hook B the final
    // CopiedState.
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
                &hooked.state.copied,
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

    // prefetch_key equals the real child key() on normal moves whose parent
    // has no ep square and whose from/to touch no castling rights —
    // including the fold interplay across the clock reset.
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

    // --- see_ge: the SEE fixture suite (D11) --------------------------

    use crate::types::{KNIGHT_VALUE, PAWN_VALUE, QUEEN_VALUE, ROOK_VALUE};

    /// Assert that `m`'s static exchange value in `fen` is exactly `see`:
    /// `see_ge` holds at the value, fails one above, and is monotone in the
    /// threshold with its single true→false step exactly at `see`.
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

    // RxP with no defender — SEE is exactly the pawn.
    #[test]
    fn see_simple_winning_capture() {
        assert_see_value(
            "1k1r4/1pp4p/p7/4p3/8/P5P1/1PP4P/2K1R3 w - - 0 1",
            Move::normal(Square::E1, Square::E5),
            PAWN_VALUE,
        );
    }

    // NxP on a well-defended square — the knight is lost for the pawn.
    #[test]
    fn see_losing_capture_defended_pawn() {
        assert_see_value(
            "1k1r3q/1ppn3p/p4b2/4p3/8/P2N2P1/1PP1R1BP/2K1Q3 w - - 0 1",
            Move::normal(Square::D3, Square::E5),
            PAWN_VALUE - KNIGHT_VALUE,
        );
    }

    // A defender pinned to its king does not count while its pinner is on
    // the board; without the pinner the knight recaptures (P - R).
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

    // The pinner-liveness flip side (the `to` toggle): NxQ captures the
    // pinner itself, freeing the pinned rook to guard the recapture square,
    // so the SEE is the whole queen.
    #[test]
    fn see_capturing_the_pinner_frees_the_pinned_piece() {
        assert_see_value(
            "7k/r3q3/8/3N4/4R3/8/8/4K3 w - - 0 1",
            Move::normal(Square::D5, Square::E7),
            QUEEN_VALUE,
        );
    }

    // Doubled rooks on both sides of the e-file: back rooks are revealed by
    // the x-ray walk; with even batteries the exchange still costs R for P.
    #[test]
    fn see_xray_rook_battery() {
        assert_see_value(
            "1k2r3/4r3/8/4p3/8/8/4R3/4R1K1 w - - 0 1",
            Move::normal(Square::E2, Square::E5),
            PAWN_VALUE - ROOK_VALUE,
        );
    }

    // Diagonal x-rays on both sides: Bg2 behind the capturing queen, Bf7
    // revealed by the pawn recapture — SEE is P - Q.
    #[test]
    fn see_xray_bishop_behind_queen() {
        assert_see_value(
            "1k6/5b2/4p3/3p4/8/5Q2/6B1/6K1 w - - 0 1",
            Move::normal(Square::F3, Square::D5),
            PAWN_VALUE - QUEEN_VALUE,
        );
    }

    // King branch: the defending king may only recapture when the attacker
    // has no backup.
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

    // Quiet-move SEE: an empty destination counts 0, so a pawn-covered
    // square hangs the knight and a safe square is exactly 0.
    #[test]
    fn see_quiet_move_to_defended_square() {
        let fen = "4k3/8/2p5/8/3N4/8/8/4K3 w - - 0 1";
        assert_see_value(fen, Move::normal(Square::D4, Square::B5), -KNIGHT_VALUE);
        assert_see_value(fen, Move::normal(Square::D4, Square::F5), 0);
    }

    // Promotion/en-passant/castling short-circuit to SEE 0.
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

    /// [`DirtyPiece::after_move`] must reproduce [`DirtyPiece::from_move`]
    /// for every move of every kind — the two derive the same fact from
    /// opposite sides of the make, so `from_move` is the independent oracle
    /// and stays in the tree for that reason. The whole-record `assert_eq!`
    /// also pins EMISSION ORDER: the accumulator folds these in sequence
    /// and the Finny backward walk indexes `subs()[0]` positionally.
    #[test]
    fn after_move_reproduces_from_move_for_every_move_kind() {
        // Roots chosen so the walk actually reaches all four move kinds.
        let roots = [
            crate::fen::START_FEN,
            "r3k2r/p1ppqpb1/bn2pnp1/3PN3/1p2P3/2N2Q1p/PPPBBPPP/R3K2R w KQkq - 0 1",
            "8/2p5/3p4/KP5r/1R3p1k/8/4P1P1/8 w - - 0 1",
            "n1n5/PPPk4/8/8/8/8/4Kppp/5N1N b - - 0 1",
            "r3k2r/Pppp1ppp/1b3nbN/nP6/BBP1P3/q4N2/Pp1P2PP/R2Q1RK1 w kq - 0 1",
            "4k3/8/8/8/4Pp2/8/8/4K3 b - e3 0 1",
        ];

        // Per-kind coverage, so a refactor that stops reaching (say) en
        // passant fails instead of silently testing less.
        let mut seen = [0usize; 4];

        fn walk(board: &mut Board, depth: usize, seen: &mut [usize; 4]) {
            if depth == 0 {
                return;
            }
            let mut list = crate::MoveList::new();
            crate::movegen::generate_legal(board, &mut list);
            let moves: Vec<Move> = list.iter().copied().collect();
            for m in moves {
                let before = DirtyPiece::from_move(board, m);
                let gives_check = board.gives_check(m);
                board.make_move(m, gives_check);
                let after = DirtyPiece::after_move(board, m);
                assert_eq!(
                    before,
                    after,
                    "after_move disagreed with from_move for {m} ({:?})",
                    m.kind()
                );
                seen[m.kind() as usize] += 1;
                walk(board, depth - 1, seen);
                board.unmake_move(m);
            }
        }

        for fen in roots {
            let mut board = Board::from_fen(fen).expect("test FEN parses");
            walk(&mut board, 3, &mut seen);
        }

        for kind in [
            MoveKind::Normal,
            MoveKind::Promotion,
            MoveKind::EnPassant,
            MoveKind::Castling,
        ] {
            assert!(
                seen[kind as usize] > 0,
                "the walk never reached a {kind:?} move — coverage claim is false"
            );
        }
    }

    /// The twin of the test above for [`DirtyPiece::from_actors`], the
    /// THIRD derivation of the same record — exercised through
    /// `make_move_with` itself, so what is checked is the record the SEARCH
    /// receives through Hook C. Same roots, walk, coverage assertion, and
    /// whole-record (order-pinning) `assert_eq!`.
    #[test]
    fn from_actors_reproduces_from_move_for_every_move_kind() {
        /// Hook C, recording; `WANTS_DIRTY` makes the record get built.
        struct Recorder(Option<DirtyPiece>);
        impl MakeMoveHooks for Recorder {
            const WANTS_DIRTY: bool = true;
            fn on_dirty(&mut self, dp: &DirtyPiece) {
                self.0 = Some(*dp);
            }
        }

        let roots = [
            crate::fen::START_FEN,
            "r3k2r/p1ppqpb1/bn2pnp1/3PN3/1p2P3/2N2Q1p/PPPBBPPP/R3K2R w KQkq - 0 1",
            "8/2p5/3p4/KP5r/1R3p1k/8/4P1P1/8 w - - 0 1",
            "n1n5/PPPk4/8/8/8/8/4Kppp/5N1N b - - 0 1",
            "r3k2r/Pppp1ppp/1b3nbN/nP6/BBP1P3/q4N2/Pp1P2PP/R2Q1RK1 w kq - 0 1",
            "4k3/8/8/8/4Pp2/8/8/4K3 b - e3 0 1",
        ];

        let mut seen = [0usize; 4];

        fn walk(board: &mut Board, depth: usize, seen: &mut [usize; 4]) {
            if depth == 0 {
                return;
            }
            let mut list = crate::MoveList::new();
            crate::movegen::generate_legal(board, &mut list);
            let moves: Vec<Move> = list.iter().copied().collect();
            for m in moves {
                let before = DirtyPiece::from_move(board, m);
                let gives_check = board.gives_check(m);
                let mut rec = Recorder(None);
                board.make_move_with(m, gives_check, &mut rec);
                let via_hook = rec.0.expect("Hook C must fire when WANTS_DIRTY");
                assert_eq!(
                    before,
                    via_hook,
                    "from_actors disagreed with from_move for {m} ({:?})",
                    m.kind()
                );
                seen[m.kind() as usize] += 1;
                walk(board, depth - 1, seen);
                board.unmake_move(m);
            }
        }

        for fen in roots {
            let mut board = Board::from_fen(fen).expect("test FEN parses");
            walk(&mut board, 3, &mut seen);
        }

        for kind in [
            MoveKind::Normal,
            MoveKind::Promotion,
            MoveKind::EnPassant,
            MoveKind::Castling,
        ] {
            assert!(
                seen[kind as usize] > 0,
                "the walk never reached a {kind:?} move — coverage claim is false"
            );
        }
    }
}
