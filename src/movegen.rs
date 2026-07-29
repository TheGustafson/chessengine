//! Type-parameterized pseudo-legal move generation (design/05-movegen.md).
//!
//! Stockfish monomorphizes one generator per `GenType`; the stable-Rust
//! equivalent is a `u8` const-generic parameter with named constants in
//! [`gen_type`] (doc 05 §3.1, §9 Q3 — `adt_const_params` is not stable, so
//! no enum in const position yet). `generate` dispatches once on the side
//! to move into a color-monomorphized worker, so all 4×2 instantiations
//! compile to straight-line routines with the dead branches removed — the
//! effect of Stockfish's `if constexpr` (never its code, D27).
//!
//! Semantics per type (doc 05 §3.2 — the split the MovePicker's staged
//! generation depends on): `CAPTURES` includes non-capturing *queen*
//! promotions; `QUIETS` includes non-capturing *under*promotions; their
//! disjoint union is `NON_EVASIONS`. One target-mask `Bitboard`
//! parameterizes every non-king generator (doc 05 §3.3).
//!
//! Hot-path rules: no heap allocation ([`MoveList`] is stack-only), no
//! `unsafe` (D14 — crate-wide `deny`).
//!
//! P3 step 7 status (doc 05 §4): generation for all four GenTypes
//! (steps 2–5), the legality layer (`Board::legal`, [`generate_legal`],
//! [`has_legal_moves`], `Board::capture_stage`/`is_capture`, step 6) and
//! the TT-move validator (`Board::pseudo_legal` + the 64k fuzz, step 7)
//! are all in. Still to come: the full-depth perft gate (step 8, once
//! `src/perft.rs` lands per design/06).

use crate::attacks;
use crate::bitboard::Bitboard;
use crate::board::{Board, CastlingRights};
use crate::moves::{Move, MoveKind, MoveList};
use crate::types::{Color, Direction, PieceType, Square};

/// The `GenType` constants: what `generate::<GT>` emits (doc 05 §3.1).
pub mod gen_type {
    /// Captures (incl. en passant), all capturing promotions (Q/R/B/N),
    /// and non-capturing QUEEN promotions (doc 05 §3.2).
    pub const CAPTURES: u8 = 0;
    /// Non-captures except castling, plus non-capturing UNDERpromotions
    /// (R/B/N). Never emits a capturing promotion.
    pub const QUIETS: u8 = 1;
    /// Check evasions; only valid while the side to move is in check.
    pub const EVASIONS: u8 = 2;
    /// Everything pseudo-legal: `CAPTURES` ⊎ `QUIETS` (disjoint union).
    pub const NON_EVASIONS: u8 = 3;
}

/// Append all pseudo-legal moves of the given type to `list`.
///
/// Does NOT clear `list` first — callers compose sub-ranges by remembering
/// `list.len()` before the call (doc 05 §3.3).
///
/// Debug-asserts that `GT == EVASIONS` if and only if the side to move is
/// in check (doc 05 §3.1).
pub fn generate<const GT: u8>(board: &Board, list: &mut MoveList) {
    debug_assert_eq!(
        GT == gen_type::EVASIONS,
        !board.checkers().is_empty(),
        "generate::<EVASIONS> iff the side to move is in check"
    );
    // Color as a second const parameter: pawn directions and rank masks
    // become compile-time constants, exactly SF's `template<Color Us>`.
    match board.side_to_move() {
        Color::White => generate_all::<GT, true>(board, list),
        Color::Black => generate_all::<GT, false>(board, list),
    }
}

/// Append all pseudo-legal captures (doc 05 §3.2 row `CAPTURES`).
#[inline]
pub fn generate_captures(board: &Board, list: &mut MoveList) {
    generate::<{ gen_type::CAPTURES }>(board, list);
}

/// Append all pseudo-legal quiet moves (doc 05 §3.2 row `QUIETS`).
#[inline]
pub fn generate_quiets(board: &Board, list: &mut MoveList) {
    generate::<{ gen_type::QUIETS }>(board, list);
}

/// Append all pseudo-legal check evasions (doc 05 §3.6).
#[inline]
pub fn generate_evasions(board: &Board, list: &mut MoveList) {
    generate::<{ gen_type::EVASIONS }>(board, list);
}

/// Append every pseudo-legal move (doc 05 §3.2 row `NON_EVASIONS`).
#[inline]
pub fn generate_non_evasions(board: &Board, list: &mut MoveList) {
    generate::<{ gen_type::NON_EVASIONS }>(board, list);
}

/// Append all fully legal moves for the side to move (doc 05 §3.9):
/// EVASIONS or NON_EVASIONS by check status, then the lazy legality
/// filter. Only three kinds of pseudo-legal moves can be illegal — moves
/// of a pinned piece, king moves, and en passant — so only those suspects
/// are tested with [`Board::legal`]; everything else is legal by
/// construction. Removal is swap-with-last, so the pinned emission order
/// does NOT survive filtering (consumers sort or count).
///
/// Appends without clearing, like [`generate`]. The search hot path never
/// calls this — the MovePicker emits pseudo-legal moves and the search
/// runs `legal` only on the move it is about to play (doc 09); this is
/// for perft, root-move setup, UCI `position ... moves` parsing, and
/// mate/stalemate detection.
pub fn generate_legal(board: &Board, list: &mut MoveList) {
    let start = list.len();
    if board.checkers().is_empty() {
        generate::<{ gen_type::NON_EVASIONS }>(board, list);
    } else {
        generate::<{ gen_type::EVASIONS }>(board, list);
    }

    // `blockers_for_king(us)` holds blockers of BOTH colors; intersecting
    // with our occupancy yields the pinned set — an enemy blocker is the
    // enemy's problem (doc 05 §3.9).
    let us = board.side_to_move();
    let pinned = board.blockers_for_king(us) & board.occupancy(us);
    let ksq = board.king_square(us);

    let mut i = start;
    while i < list.len() {
        let m = list[i];
        let suspect =
            pinned.contains(m.from()) || m.from() == ksq || m.kind() == MoveKind::EnPassant;
        if suspect && !board.legal(m) {
            // Do not advance `i`: the swapped-in last move is unvetted.
            list.swap_remove(i);
        } else {
            i += 1;
        }
    }
}

/// True if the side to move has at least one legal move — the
/// mate/stalemate probe consumed by `Board::is_draw`'s 50-move guard and,
/// later, the UCI layer (doc 05 §3.9). Same generation + lazy-filter
/// logic as [`generate_legal`], but returns on the first survivor.
pub fn has_legal_moves(board: &Board) -> bool {
    let mut list = MoveList::new();
    if board.checkers().is_empty() {
        generate::<{ gen_type::NON_EVASIONS }>(board, &mut list);
    } else {
        generate::<{ gen_type::EVASIONS }>(board, &mut list);
    }

    let us = board.side_to_move();
    let pinned = board.blockers_for_king(us) & board.occupancy(us);
    let ksq = board.king_square(us);

    list.iter().any(|&m| {
        let suspect =
            pinned.contains(m.from()) || m.from() == ksq || m.kind() == MoveKind::EnPassant;
        !suspect || board.legal(m)
    })
}

/// The color-monomorphized worker: computes the single target mask
/// (doc 05 §3.3) and runs the per-piece generators in the pinned emission
/// order (doc 05 §5.4): pawns, N, B, R, Q, K, castles.
fn generate_all<const GT: u8, const WHITE: bool>(board: &Board, list: &mut MoveList) {
    use gen_type::{CAPTURES, EVASIONS, NON_EVASIONS, QUIETS};
    let us = if WHITE { Color::White } else { Color::Black };
    let ksq = board.king_square(us);
    // One atomic-acquire deref for the whole generation pass; the slider
    // closures below then use the `&'static` tables directly (attacks.rs
    // hot-loop pattern).
    let tables = attacks::tables();

    // Double check: only king moves can be legal — skip all non-king work.
    if GT != EVASIONS || !board.checkers().more_than_one() {
        let target = match GT {
            // `between` includes the checker square by convention
            // (doc 03 §3.4), so this mask is "block OR capture" as-is.
            EVASIONS => attacks::between(ksq, board.checkers().lsb().unwrap()),
            NON_EVASIONS => !board.occupancy(us),
            CAPTURES => board.occupancy(us.flip()),
            _ => !board.all_occupancy(), // QUIETS
        };

        pawn_moves::<GT, WHITE>(board, target, list);

        let occ = board.all_occupancy();
        piece_moves(
            board.pieces(us, PieceType::Knight),
            target,
            attacks::knight_attacks,
            list,
        );
        piece_moves(
            board.pieces(us, PieceType::Bishop),
            target,
            |from| tables.bishop_attacks(from, occ),
            list,
        );
        piece_moves(
            board.pieces(us, PieceType::Rook),
            target,
            |from| tables.rook_attacks(from, occ),
            list,
        );
        piece_moves(
            board.pieces(us, PieceType::Queen),
            target,
            |from| tables.queen_attacks(from, occ),
            list,
        );
    }

    // King moves: under EVASIONS the king may go anywhere not our own —
    // fleeing is not restricted to the block/capture ray. Safety testing
    // (including the x-ray through the vacated square) is `Board::legal`'s
    // job (doc 05 §3.8), not the generator's.
    let king_target = match GT {
        EVASIONS | NON_EVASIONS => !board.occupancy(us),
        CAPTURES => board.occupancy(us.flip()),
        _ => !board.all_occupancy(), // QUIETS
    };
    for to in attacks::king_attacks(ksq) & king_target {
        list.push(Move::normal(ksq, to));
    }

    // A castle never captures and never evades (the mover is not in check
    // in both remaining contexts) — QUIETS/NON_EVASIONS only (doc 05 §3.7).
    if GT == QUIETS || GT == NON_EVASIONS {
        castling_moves::<WHITE>(board, list);
    }
}

/// Pawn pushes, promotions, captures, and en passant (doc 05 §3.4).
///
/// Never loops per pawn: the whole pawn set is shifted at once and the
/// from-square recovered as `to − offset` — every pawn in one batch moved
/// by the same offset, and a shifted-and-masked destination has a valid,
/// non-wrapping origin by construction, so `Square::offset` (panics only
/// on a genuine bug) is the right primitive. `WHITE` makes every
/// direction and rank mask below a compile-time constant.
fn pawn_moves<const GT: u8, const WHITE: bool>(
    board: &Board,
    target: Bitboard,
    list: &mut MoveList,
) {
    use gen_type::{CAPTURES, EVASIONS, QUIETS};

    let us = if WHITE { Color::White } else { Color::Black };
    let them = us.flip();
    let up = if WHITE {
        Direction::North
    } else {
        Direction::South
    };
    let up_right = if WHITE {
        Direction::NorthEast
    } else {
        Direction::SouthWest
    };
    let up_left = if WHITE {
        Direction::NorthWest
    } else {
        Direction::SouthEast
    };
    let down = opposite(up);
    // Relative rank 7 (pawns about to promote) and the rank-3 relay rank.
    let rank7 = if WHITE {
        Bitboard::RANKS[6]
    } else {
        Bitboard::RANKS[1]
    };
    let rank3 = if WHITE {
        Bitboard::RANKS[2]
    } else {
        Bitboard::RANKS[5]
    };

    let empty = !board.all_occupancy();
    // EVASIONS: a pawn capture can only resolve the check by taking the
    // checker itself; interpositions are the push loops' target mask.
    let enemies = if GT == EVASIONS {
        board.checkers()
    } else {
        board.occupancy(them)
    };
    let pawns = board.pieces(us, PieceType::Pawn);
    let on7 = pawns & rank7;
    let not7 = pawns & !rank7;

    // --- 1. Single and double pushes (everything except CAPTURES) ---
    if GT != CAPTURES {
        let mut b1 = not7.shift(up) & empty;
        // Rank-3 relay trick: a double push exists iff the single push
        // landed on relative rank 3 AND one more shift is also empty —
        // both-empty tested with two ANDs, zero per-pawn work.
        let mut b2 = (b1 & rank3).shift(up) & empty;
        if GT == EVASIONS {
            // Only interpositions on the block/capture ray count.
            b1 &= target;
            b2 &= target;
        }
        for to in b1 {
            list.push(Move::normal(to.offset(down), to));
        }
        for to in b2 {
            list.push(Move::normal(to.offset(down).offset(down), to));
        }
    }

    // --- 2. Promotions (the §3.2 split, encoded by `push_promotions`) ---
    if !on7.is_empty() {
        if GT != QUIETS {
            // Capturing promotions, both diagonals.
            for to in on7.shift(up_right) & enemies {
                push_promotions::<GT, true>(list, to.offset(opposite(up_right)), to);
            }
            for to in on7.shift(up_left) & enemies {
                push_promotions::<GT, true>(list, to.offset(opposite(up_left)), to);
            }
        }
        let mut b3 = on7.shift(up) & empty;
        if GT == EVASIONS {
            b3 &= target; // a push promotion must land on the block ray
        }
        for to in b3 {
            push_promotions::<GT, false>(list, to.offset(down), to);
        }
    }

    // --- 3. Ordinary and en-passant captures (not for QUIETS) ---
    if GT != QUIETS {
        for to in not7.shift(up_right) & enemies {
            list.push(Move::normal(to.offset(opposite(up_right)), to));
        }
        for to in not7.shift(up_left) & enemies {
            list.push(Move::normal(to.offset(opposite(up_left)), to));
        }
        if let Some(ep) = board.en_passant_square() {
            // D9 (design/04 §3.7): an ep square is only ever recorded when
            // a legal — hence pseudo-legal — ep capture exists, both on FEN
            // load and in make_move, so a capturer must be present.
            debug_assert!(
                !(not7 & attacks::pawn_attacks(them, ep)).is_empty(),
                "D9 violated: recorded ep square {ep} has no pseudo-legal capturer"
            );
            // EVASIONS: if the double-push origin (one step `up` of the ep
            // square from our view) lies on the checking ray, the check is
            // a discovered slider check through that square — an ep capture
            // can never resolve it. Skip. (Unreachable while D9 holds; kept
            // as defense-in-depth per doc 05 §3.4.)
            if GT == EVASIONS && target.contains(ep.offset(up)) {
                return;
            }
            // Our pawns attacking the ep square = pawn attacks FROM the ep
            // square as if an enemy pawn stood there (reversal trick).
            for from in not7 & attacks::pawn_attacks(them, ep) {
                list.push(Move::en_passant(from, ep));
            }
        }
    }
}

/// Emit the promotions for one from/to pair per the doc 05 §3.2 table:
/// queen first, then R, B, N (pinned emission order, §5.4). `CAPTURE`
/// distinguishes capturing promotions (all four pieces belong to CAPTURES)
/// from push promotions (queen to CAPTURES, underpromotions to QUIETS).
#[inline(always)]
fn push_promotions<const GT: u8, const CAPTURE: bool>(
    list: &mut MoveList,
    from: Square,
    to: Square,
) {
    use gen_type::{CAPTURES, EVASIONS, NON_EVASIONS, QUIETS};
    let all = GT == EVASIONS || GT == NON_EVASIONS;
    if GT == CAPTURES || all {
        list.push(Move::promotion(from, to, PieceType::Queen));
    }
    if (GT == CAPTURES && CAPTURE) || (GT == QUIETS && !CAPTURE) || all {
        list.push(Move::promotion(from, to, PieceType::Rook));
        list.push(Move::promotion(from, to, PieceType::Bishop));
        list.push(Move::promotion(from, to, PieceType::Knight));
    }
}

/// Direction negation. Doc 05 §3.4 keeps this private to movegen until
/// doc 02 grows a `Direction::opposite`.
const fn opposite(d: Direction) -> Direction {
    match d {
        Direction::North => Direction::South,
        Direction::NorthEast => Direction::SouthWest,
        Direction::East => Direction::West,
        Direction::SouthEast => Direction::NorthWest,
        Direction::South => Direction::North,
        Direction::SouthWest => Direction::NorthEast,
        Direction::West => Direction::East,
        Direction::NorthWest => Direction::SouthEast,
    }
}

/// One shared inner loop, monomorphized per call site via `impl Fn` — each
/// call gets its own copy with the attack lookup inlined, the Rust idiom
/// for SF's `template<PieceType Pt>` (doc 05 §3.5). Slider closures capture
/// occupancy by value (`Bitboard` is `Copy`); leapers ignore it.
#[inline(always)]
fn piece_moves(
    pieces: Bitboard,
    target: Bitboard,
    attack: impl Fn(Square) -> Bitboard,
    list: &mut MoveList,
) {
    for from in pieces {
        for to in attack(from) & target {
            list.push(Move::normal(from, to));
        }
    }
}

/// Castling generation (doc 05 §3.7): the right must be present and the
/// between-king-and-rook path empty at generation time — the rights
/// themselves guarantee king/rook placement (make/unmake revokes them on
/// any king/rook move or capture). Attacked-square testing is deliberately
/// deferred to `Board::legal` (step 6), matching SF.
///
/// Only reached under QUIETS / NON_EVASIONS: a castle never captures and
/// never evades, since the mover is not in check in both contexts. The
/// encoding is the king's two-square move (D2): e1g1 / e1c1 relative.
/// Emission order: kingside before queenside (doc 05 §5.4 order pin).
fn castling_moves<const WHITE: bool>(board: &Board, list: &mut MoveList) {
    let us = if WHITE { Color::White } else { Color::Black };
    let from = Square::E1.relative_to(us);
    for (right, to) in [
        (CastlingRights::kingside(us), Square::G1.relative_to(us)),
        (CastlingRights::queenside(us), Square::C1.relative_to(us)),
    ] {
        if board.castling_rights().contains(right)
            && (board.all_occupancy() & board.castling_path(right)).is_empty()
        {
            list.push(Move::castling(from, to));
        }
    }
}

/// The generator-coupled predicates live next to the generator (doc 05
/// §3.11): `legal` and `capture_stage` must mirror the generation split
/// exactly, and `pseudo_legal` literally calls [`generate`].
impl Board {
    /// True iff any piece of `c` attacks `sq` under `occupied` — the
    /// early-exit boolean twin of [`Board::attackers_to`] that legality
    /// testing hammers (doc 05 §3.8). The `||` chain short-circuits
    /// between the five terms; sliders first, they answer most often on
    /// the king-safety probes this serves.
    pub fn attackers_to_exist(&self, sq: Square, occupied: Bitboard, c: Color) -> bool {
        let tables = attacks::tables();
        let queens = self.pieces(c, PieceType::Queen);
        !(tables.rook_attacks(sq, occupied) & (self.pieces(c, PieceType::Rook) | queens))
            .is_empty()
            || !(tables.bishop_attacks(sq, occupied)
                & (self.pieces(c, PieceType::Bishop) | queens))
                .is_empty()
            // Attack symmetry: the squares a `c` pawn attacking `sq` may
            // stand on are the squares a `!c` pawn ON `sq` would attack.
            || !(attacks::pawn_attacks(c.flip(), sq) & self.pieces(c, PieceType::Pawn)).is_empty()
            || !(attacks::knight_attacks(sq) & self.pieces(c, PieceType::Knight)).is_empty()
            || !(attacks::king_attacks(sq) & self.pieces(c, PieceType::King)).is_empty()
    }

    /// Does playing the pseudo-legal move `m` leave our king unattacked
    /// (doc 05 §3.8)? Uses only current state — no make/unmake.
    ///
    /// Precondition: `m` is pseudo-legal for the side to move (generated
    /// by [`generate`] or validated by [`Board::pseudo_legal`]).
    ///
    /// Deviation note (doc 05 §9 Q1): current SF has no en-passant branch
    /// here at all — it relies on the D9 recording invariant (an ep square
    /// exists only when a legal ep capture does). We keep the classic
    /// occupancy-swap test as defense-in-depth: `legal` stays correct even
    /// on hand-forced boards that violate D9, at the cost of two slider
    /// probes on a rare move type.
    pub fn legal(&self, m: Move) -> bool {
        let us = self.side_to_move();
        let them = us.flip();
        let (from, to) = (m.from(), m.to());
        let ksq = self.king_square(us);
        debug_assert_eq!(
            self.piece_on(from).map(|pc| pc.color),
            Some(us),
            "legal() precondition: {m:?} must move a piece of the side to move"
        );

        match m.kind() {
            // 1+2. Castling and en passant, out of line together (perf study
            //      17 §5). Between them they are the only arms that need the
            //      slider tables, a rebuilt occupancy and an
            //      `attackers_to_exist` loop, and they are a fraction of a
            //      percent of the moves `legal` is asked about; inline, they
            //      set this function's prologue and spill.
            MoveKind::Castling | MoveKind::EnPassant => self.legal_special(m, us, ksq, from, to),

            // 3. King steps: the destination must be safe with the king
            //    REMOVED from occupancy, so a checking slider's ray extends
            //    through the vacated square (fleeing along the ray is
            //    illegal).
            _ if from == ksq => !self.attackers_to_exist(
                to,
                self.all_occupancy() ^ Bitboard::from_square(from),
                them,
            ),

            // 4. Everything else: legal iff not pinned, or moving along
            //    the pin ray (toward or away from the king).
            _ => !self.blockers_for_king(us).contains(from) || attacks::aligned(from, to, ksq),
        }
    }

    /// The castling and en-passant arms of [`Board::legal`].
    #[cold]
    #[inline(never)]
    fn legal_special(&self, m: Move, us: Color, ksq: Square, from: Square, to: Square) -> bool {
        // `them` is re-derived, not passed: with it this took SEVEN integer
        // arguments, one past the six SysV registers, so every call site wrote
        // a stack slot for a path it almost never takes (D33 reviewer C).
        // `Color::flip` is one `xor`. Splitting the two arms into separate
        // helpers -- the reviewer's preferred shape -- was also tried and cost
        // **+19.4 i/n**, because it turns one cold call site into two and hands
        // the caller's `match` back its four-way dispatch. Measured, not argued.
        let them = us.flip();
        match m.kind() {
            // 1. Castling: the two squares the king crosses/lands on must
            //    be unattacked. `from` (e1/e8) is not tested — castling is
            //    never generated nor pseudo_legal-validated while in check.
            //    Current occupancy is correct: the only ray on which the
            //    unmoved king could shield a tested square is the back rank
            //    itself, and a back-rank attacker behind the king would
            //    already be check (Chess960's rook-shield case is deferred
            //    with D10).
            MoveKind::Castling => {
                let tested = if to.file() > from.file() {
                    [Square::F1, Square::G1] // kingside
                } else {
                    [Square::D1, Square::C1] // queenside
                };
                !tested.iter().any(|&s| {
                    self.attackers_to_exist(s.relative_to(us), self.all_occupancy(), them)
                })
            }

            // 2. En passant: the only move that clears TWO squares at
            //    once. Rebuild the after-occupancy and probe sliders from
            //    the king square (this is what catches the horizontal
            //    both-pawns-off-the-rank pin).
            MoveKind::EnPassant => {
                let capsq = to.offset(opposite(us.pawn_push()));
                let occ = (self.all_occupancy()
                    ^ Bitboard::from_square(from)
                    ^ Bitboard::from_square(capsq))
                    | Bitboard::from_square(to);
                let tables = attacks::tables();
                let queens = self.pieces(them, PieceType::Queen);
                (tables.rook_attacks(ksq, occ) & (self.pieces(them, PieceType::Rook) | queens))
                    .is_empty()
                    && (tables.bishop_attacks(ksq, occ)
                        & (self.pieces(them, PieceType::Bishop) | queens))
                        .is_empty()
            }

            // Unreachable: the caller only routes the two kinds above here.
            // `false` is the fail-soft direction — a spurious "illegal" drops a
            // move, where `true` would let an illegal one be played (D33
            // reviewer A). No `let _ = …` suppression here: both parameters are
            // live in the arms above, so a future unused-parameter warning
            // would be REAL and must not be pre-silenced.
            _ => {
                debug_assert!(false, "legal_special called with {m:?}");
                false
            }
        }
    }

    /// The from-scratch TT-move validator (doc 05 §3.10): would
    /// `generate::<EVASIONS|NON_EVASIONS>` emit exactly this move in the
    /// CURRENT position? TT moves are raw `u16`s that may be corrupted by
    /// data races or key aliasing (D19); they must pass this — and then
    /// [`Board::legal`] separately — before being played. Total over all
    /// 65,536 encodings; never panics.
    pub fn pseudo_legal(&self, m: Move) -> bool {
        let us = self.side_to_move();
        let them = us.flip();
        let (from, to) = (m.from(), m.to());

        // 1. Non-normal kinds (promotion / ep / castling) are rare:
        //    validate by containment in a fresh generation. This inherits
        //    every rule (promotion geometry, ep square match, castling
        //    rights + empty path) with zero duplicated logic — including
        //    the full-16-bit equality that rejects garbage promotion bits
        //    on ep/castling encodings. Legality is NOT included.
        if m.kind() != MoveKind::Normal {
            let mut list = MoveList::new();
            if self.checkers().is_empty() {
                generate::<{ gen_type::NON_EVASIONS }>(self, &mut list);
            } else {
                generate::<{ gen_type::EVASIONS }>(self, &mut list);
            }
            return list.contains(&m);
        }

        // 2. Bit hygiene: a Normal move must have zero promotion bits, or
        //    it could never equal a generated move (and would corrupt
        //    stats keyed by the raw encoding). Rejects garbage u16s. Also
        //    rejects Move::NULL: from == to fails checks 3-5 in every
        //    case (our piece on `from` means our piece on `to`).
        if m.raw16() & 0x3000 != 0 {
            return false;
        }

        // 3. A piece of ours must sit on `from`.
        let Some(pc) = self.piece_on(from) else {
            return false;
        };
        if pc.color != us {
            return false;
        }

        // 4. `to` must not hold one of our pieces.
        if self.occupancy(us).contains(to) {
            return false;
        }

        if pc.piece_type == PieceType::Pawn {
            // 5a. Pawns: the destination cannot be a promotion rank (that
            //     is kind Promotion), and the move must be a capture, a
            //     single push, or a double push.
            if (Bitboard::RANK_1 | Bitboard::RANK_8).contains(to) {
                return false;
            }
            let up = us.pawn_push();
            let start_rank = if us == Color::White { 1 } else { 6 };
            let is_capture =
                attacks::pawn_attacks(us, from).contains(to) && self.occupancy(them).contains(to);
            let is_single = from.shifted(up) == Some(to) && !self.all_occupancy().contains(to);
            // Short-circuit keeps `to.offset(down)` (the relay square) on
            // the board: it only evaluates once `to` is the true
            // double-push destination.
            let is_double = from.rank() == start_rank
                && from.shifted(up).and_then(|s| s.shifted(up)) == Some(to)
                && !self.all_occupancy().contains(to)
                && !self.all_occupancy().contains(to.offset(opposite(up)));
            if !(is_capture || is_single || is_double) {
                return false;
            }
        }
        // 5b. Other pieces: `to` must be attacked from `from` on the
        //     current occupancy (`attacks` dispatches N/B/R/Q/K, so this
        //     handles king steps too).
        else if !attacks::attacks(pc.piece_type, from, self.all_occupancy()).contains(to) {
            return false;
        }

        // 6. In check, a Normal move is pseudo-legal only if the evasion
        //    generator would emit it (blocks, checker captures, king
        //    flights — with the evasion target masks applied).
        if !self.checkers().is_empty() {
            let mut list = MoveList::new();
            generate::<{ gen_type::EVASIONS }>(self, &mut list);
            return list.contains(&m);
        }

        true
    }

    /// True iff `m` is generated by the CAPTURES stage (doc 05 §3.2): any
    /// capture, or any queen promotion (even non-capturing). The
    /// MovePicker relies on this mirroring the generator split exactly to
    /// avoid emitting a move twice across the capture/quiet stages; the
    /// search uses it for `ttCapture` and stats classification.
    ///
    /// Precondition: `m` is pseudo-legal in `self`. On arbitrary moves the
    /// answer is meaningless but never panics.
    pub fn capture_stage(&self, m: Move) -> bool {
        match m.kind() {
            MoveKind::Normal => self.all_occupancy().contains(m.to()),
            MoveKind::Promotion => {
                self.all_occupancy().contains(m.to())
                    || m.promotion_piece() == Some(PieceType::Queen)
            }
            MoveKind::EnPassant => true,
            MoveKind::Castling => false,
        }
    }

    /// True iff `m` actually captures a piece — material changes
    /// (doc 05 §3.2). Same precondition as [`Board::capture_stage`].
    pub fn is_capture(&self, m: Move) -> bool {
        match m.kind() {
            MoveKind::Normal | MoveKind::Promotion => self.all_occupancy().contains(m.to()),
            MoveKind::EnPassant => true,
            MoveKind::Castling => false,
        }
    }
}

#[cfg(test)]
mod tests {
    use super::*;
    use crate::moves::MoveKind;
    use std::collections::HashSet;

    /// Parse a FEN, panicking with context on failure.
    fn board(fen: &str) -> Board {
        Board::from_fen(fen).unwrap_or_else(|e| panic!("bad test FEN {fen:?}: {e}"))
    }

    fn generated<const GT: u8>(board: &Board) -> MoveList {
        let mut list = MoveList::new();
        generate::<GT>(board, &mut list);
        list
    }

    /// Moves in `list` made by a `pt` of the side to move.
    fn count_for_piece(board: &Board, list: &MoveList, pt: PieceType) -> usize {
        list.iter()
            .filter(|m| {
                board
                    .piece_on(m.from())
                    .expect("generated move must start on a piece")
                    .piece_type
                    == pt
            })
            .count()
    }

    fn move_set(list: &MoveList) -> HashSet<Move> {
        let set: HashSet<Move> = list.iter().copied().collect();
        assert_eq!(set.len(), list.len(), "duplicate moves within one list");
        set
    }

    /// The pawn-made subset of one generation pass (promotions and ep
    /// included by construction; piece/king/castle moves filtered out).
    fn pawn_move_set<const GT: u8>(board: &Board) -> HashSet<Move> {
        let list = generated::<GT>(board);
        move_set(&list)
            .into_iter()
            .filter(|m| {
                board
                    .piece_on(m.from())
                    .expect("generated move must start on a piece")
                    .piece_type
                    == PieceType::Pawn
            })
            .collect()
    }

    const ALL_PROMOS: [PieceType; 4] = [
        PieceType::Queen,
        PieceType::Rook,
        PieceType::Bishop,
        PieceType::Knight,
    ];
    const UNDER_PROMOS: [PieceType; 3] = [PieceType::Rook, PieceType::Bishop, PieceType::Knight];

    /// The promotion fan `from -> to` for the given pieces, as a set.
    fn promo_fan(from: Square, to: Square, pieces: &[PieceType]) -> HashSet<Move> {
        pieces
            .iter()
            .map(|&pt| Move::promotion(from, to, pt))
            .collect()
    }

    /// Hand fixtures with per-piece counts hand-verified against the board
    /// diagram (doc 05 §4 step 2, §5 for this step). No mover-side pawns
    /// (pawn generation contributes nothing) and no castling rights, so
    /// every count below stays valid once `castling_moves` (step 5) lands.
    /// Fields: FEN, then expected (knight, bishop, rook, queen, king)
    /// NON_EVASIONS counts, then expected CAPTURES total.
    const FIXTURES: &[(&str, [usize; 5], usize)] = &[
        // Nc3 vs pawn d5: 8 knight moves (1 capture), 5 king moves.
        ("4k3/8/8/3p4/8/2N5/8/4K3 w - - 0 1", [8, 0, 0, 0, 5], 1),
        // Be2 vs pawn g4: rays stop at the capture; e2 is own-blocked for
        // the king (4 king moves, not 5).
        ("4k3/8/8/8/6p1/8/4B3/4K3 w - - 0 1", [0, 8, 0, 0, 4], 1),
        // Rd4 between pawns b4/g4: both rank rays end in captures.
        ("k7/8/8/8/1p1R2p1/8/8/4K3 w - - 0 1", [0, 0, 12, 0, 5], 2),
        // Black to move (color-dispatch check): open-board queen on d5.
        ("4k3/8/8/3q4/8/8/8/4K3 b - - 0 1", [0, 0, 0, 27, 5], 0),
        // King capture: e2 is the only enemy piece adjacent to e1.
        ("4k3/8/8/8/8/8/4p3/4K3 w - - 0 1", [0, 0, 0, 0, 5], 1),
        // Boxed rook: own knight (a2) and own king (b1) block both rays —
        // zero rook moves proves the target mask excludes own pieces and
        // occupancy stops sliders.
        ("4k3/8/8/8/8/8/N7/RK6 w - - 0 1", [3, 0, 0, 0, 3], 0),
    ];

    #[test]
    fn per_piece_counts_on_hand_fixtures() {
        for &(fen, per_piece, _) in FIXTURES {
            let b = board(fen);
            let all = generated::<{ gen_type::NON_EVASIONS }>(&b);
            let expected: [(PieceType, usize); 5] = [
                (PieceType::Knight, per_piece[0]),
                (PieceType::Bishop, per_piece[1]),
                (PieceType::Rook, per_piece[2]),
                (PieceType::Queen, per_piece[3]),
                (PieceType::King, per_piece[4]),
            ];
            for (pt, count) in expected {
                assert_eq!(
                    count_for_piece(&b, &all, pt),
                    count,
                    "{pt:?} count in {fen}"
                );
            }
            assert_eq!(
                all.len(),
                per_piece.iter().sum::<usize>(),
                "total NON_EVASIONS count in {fen}"
            );
        }
    }

    #[test]
    fn capture_counts_on_hand_fixtures() {
        for &(fen, _, captures) in FIXTURES {
            let b = board(fen);
            assert_eq!(
                generated::<{ gen_type::CAPTURES }>(&b).len(),
                captures,
                "CAPTURES count in {fen}"
            );
        }
    }

    /// doc 05 §3.3 target masks: CAPTURES lands on enemy pieces only,
    /// QUIETS on empty squares only, NON_EVASIONS never on own pieces —
    /// and every from-square holds a piece of the side to move.
    #[test]
    fn target_mask_filtering() {
        for &(fen, _, _) in FIXTURES {
            let b = board(fen);
            let us = b.side_to_move();

            for m in &generated::<{ gen_type::CAPTURES }>(&b) {
                assert!(
                    b.occupancy(us.flip()).contains(m.to()),
                    "CAPTURES move {m} in {fen} must land on an enemy piece"
                );
            }
            for m in &generated::<{ gen_type::QUIETS }>(&b) {
                assert!(
                    !b.all_occupancy().contains(m.to()),
                    "QUIETS move {m} in {fen} must land on an empty square"
                );
            }
            for m in &generated::<{ gen_type::NON_EVASIONS }>(&b) {
                assert!(
                    !b.occupancy(us).contains(m.to()),
                    "move {m} in {fen} may never land on an own piece"
                );
                assert_eq!(
                    b.piece_on(m.from()).map(|p| p.color),
                    Some(us),
                    "move {m} in {fen} must start on a mover piece"
                );
                // These movers have no pawns and no castling rights.
                assert_eq!(m.kind(), MoveKind::Normal);
            }
        }
    }

    /// doc 05 §5.1 property 1 on this step's fixtures: no duplicates within
    /// a type, CAPTURES ∩ QUIETS = ∅, and CAPTURES ∪ QUIETS = NON_EVASIONS
    /// (lengths add up).
    #[test]
    fn captures_and_quiets_partition_non_evasions() {
        for &(fen, _, _) in FIXTURES {
            let b = board(fen);
            let captures = move_set(&generated::<{ gen_type::CAPTURES }>(&b));
            let quiets = move_set(&generated::<{ gen_type::QUIETS }>(&b));
            let all = move_set(&generated::<{ gen_type::NON_EVASIONS }>(&b));

            assert!(
                captures.is_disjoint(&quiets),
                "CAPTURES ∩ QUIETS must be empty in {fen}"
            );
            let union: HashSet<Move> = captures.union(&quiets).copied().collect();
            assert_eq!(union, all, "CAPTURES ⊎ QUIETS != NON_EVASIONS in {fen}");
            assert_eq!(captures.len() + quiets.len(), all.len());
        }
    }

    /// Exact move-set equality on the knight fixture: every generated move
    /// lands precisely on the hand-listed target squares (doc 05 §4 step 2).
    #[test]
    fn knight_fixture_exact_move_set() {
        let b = board("4k3/8/8/3p4/8/2N5/8/4K3 w - - 0 1");
        let expected: HashSet<Move> = [
            // Nc3 (d5 is the lone capture):
            (Square::C3, Square::A2),
            (Square::C3, Square::A4),
            (Square::C3, Square::B1),
            (Square::C3, Square::B5),
            (Square::C3, Square::D1),
            (Square::C3, Square::D5),
            (Square::C3, Square::E2),
            (Square::C3, Square::E4),
            // Ke1:
            (Square::E1, Square::D1),
            (Square::E1, Square::D2),
            (Square::E1, Square::E2),
            (Square::E1, Square::F1),
            (Square::E1, Square::F2),
        ]
        .into_iter()
        .map(|(from, to)| Move::normal(from, to))
        .collect();

        assert_eq!(
            move_set(&generated::<{ gen_type::NON_EVASIONS }>(&b)),
            expected
        );
        assert_eq!(
            move_set(&generated::<{ gen_type::CAPTURES }>(&b)),
            HashSet::from([Move::normal(Square::C3, Square::D5)])
        );
    }

    /// Exact move set for the boxed-rook fixture — the rook contributes
    /// nothing, so this pins that blocked sliders emit no moves at all.
    #[test]
    fn boxed_rook_exact_move_set() {
        let b = board("4k3/8/8/8/8/8/N7/RK6 w - - 0 1");
        let expected: HashSet<Move> = [
            (Square::A2, Square::B4),
            (Square::A2, Square::C1),
            (Square::A2, Square::C3),
            (Square::B1, Square::B2),
            (Square::B1, Square::C1),
            (Square::B1, Square::C2),
        ]
        .into_iter()
        .map(|(from, to)| Move::normal(from, to))
        .collect();

        assert_eq!(
            move_set(&generated::<{ gen_type::NON_EVASIONS }>(&b)),
            expected
        );
    }

    /// doc 05 §5.4: emission order is part of bench reproducibility. Within
    /// one generate call the mover piece types appear in the pinned class
    /// order N → B → R → Q → K (pawns first and castles last once they land).
    #[test]
    fn emission_order_class_is_pinned() {
        fn stage(pt: PieceType) -> usize {
            match pt {
                PieceType::Pawn => 0,
                PieceType::Knight => 1,
                PieceType::Bishop => 2,
                PieceType::Rook => 3,
                PieceType::Queen => 4,
                PieceType::King => 5,
            }
        }
        for &(fen, _, _) in FIXTURES {
            let b = board(fen);
            let all = generated::<{ gen_type::NON_EVASIONS }>(&b);
            let stages: Vec<usize> = all
                .iter()
                .map(|m| stage(b.piece_on(m.from()).unwrap().piece_type))
                .collect();
            assert!(
                stages.windows(2).all(|w| w[0] <= w[1]),
                "emission order not monotone in {fen}: {stages:?}"
            );
        }
    }

    /// `generate` appends without clearing (doc 05 §3.3): callers compose
    /// sub-ranges, so CAPTURES followed by QUIETS into the same list must
    /// concatenate.
    #[test]
    fn generate_appends_without_clearing() {
        let b = board("k7/8/8/8/1p1R2p1/8/8/4K3 w - - 0 1");
        let captures = generated::<{ gen_type::CAPTURES }>(&b);
        let quiets = generated::<{ gen_type::QUIETS }>(&b);

        let mut composed = MoveList::new();
        generate_captures(&b, &mut composed);
        assert_eq!(composed.len(), captures.len());
        generate_quiets(&b, &mut composed);
        assert_eq!(composed.len(), captures.len() + quiets.len());
        assert_eq!(&composed[..captures.len()], &captures[..]);
        assert_eq!(&composed[captures.len()..], &quiets[..]);
    }

    /// The EVASIONS↔in-check contract is a debug_assert (doc 05 §3.1);
    /// debug builds only — release strips it by design.
    #[cfg(debug_assertions)]
    #[test]
    #[should_panic(expected = "generate::<EVASIONS> iff the side to move is in check")]
    fn evasions_without_check_panics_in_debug() {
        let b = board("4k3/8/8/8/8/8/8/4K3 w - - 0 1");
        let mut list = MoveList::new();
        generate_evasions(&b, &mut list);
    }

    // -----------------------------------------------------------------------
    // P3 step 3: pawn generation (doc 05 §3.4, §4 step 3, §5.2)
    // -----------------------------------------------------------------------

    #[test]
    fn startpos_pawn_pushes() {
        let b = board(crate::fen::START_FEN);
        // 8 single + 8 double pushes, all quiet; nothing to capture.
        let mut expected = HashSet::new();
        for file in 0..8 {
            let from = Square::from_coords(file, 1);
            expected.insert(Move::normal(from, Square::from_coords(file, 2)));
            expected.insert(Move::normal(from, Square::from_coords(file, 3)));
        }
        assert_eq!(pawn_move_set::<{ gen_type::QUIETS }>(&b), expected);
        assert!(pawn_move_set::<{ gen_type::CAPTURES }>(&b).is_empty());
        // Doubles as the perft(1) = 20 pin: the castle paths are occupied,
        // so this count stays exact once step 5 lands.
        assert_eq!(generated::<{ gen_type::NON_EVASIONS }>(&b).len(), 20);
    }

    /// Doc 05 §4 step 3 "double-push blocking": a blocker on the relay
    /// square kills both pushes; a blocker one further kills only the
    /// double. Both colors (the rank-3 relay mask is per-color).
    #[test]
    fn blocked_and_relayed_pushes() {
        // Directly blocked: no single push, hence no double either.
        let b = board("4k3/8/8/8/8/4p3/4P3/4K3 w - - 0 1");
        assert!(pawn_move_set::<{ gen_type::NON_EVASIONS }>(&b).is_empty());

        // Relay square e3 free, e4 occupied: single push only.
        let b = board("4k3/8/8/8/4p3/8/4P3/4K3 w - - 0 1");
        assert_eq!(
            pawn_move_set::<{ gen_type::NON_EVASIONS }>(&b),
            HashSet::from([Move::normal(Square::E2, Square::E3)])
        );

        // Black mirrors of both shapes.
        let b = board("4k3/4p3/4P3/8/8/8/8/4K3 b - - 0 1");
        assert!(pawn_move_set::<{ gen_type::NON_EVASIONS }>(&b).is_empty());
        let b = board("4k3/4p3/8/4P3/8/8/8/4K3 b - - 0 1");
        assert_eq!(
            pawn_move_set::<{ gen_type::NON_EVASIONS }>(&b),
            HashSet::from([Move::normal(Square::E7, Square::E6)])
        );
    }

    /// Diagonal capture shifts in both directions for both colors, with the
    /// from-square recovered as `to − offset`.
    #[test]
    fn captures_both_directions_both_colors() {
        let b = board("4k3/8/8/3p1p2/4P3/8/8/4K3 w - - 0 1");
        assert_eq!(
            pawn_move_set::<{ gen_type::CAPTURES }>(&b),
            HashSet::from([
                Move::normal(Square::E4, Square::D5),
                Move::normal(Square::E4, Square::F5),
            ])
        );
        assert_eq!(
            pawn_move_set::<{ gen_type::QUIETS }>(&b),
            HashSet::from([Move::normal(Square::E4, Square::E5)])
        );

        let b = board("4k3/8/8/4p3/3P1P2/8/8/4K3 b - - 0 1");
        assert_eq!(
            pawn_move_set::<{ gen_type::CAPTURES }>(&b),
            HashSet::from([
                Move::normal(Square::E5, Square::D4),
                Move::normal(Square::E5, Square::F4),
            ])
        );
        assert_eq!(
            pawn_move_set::<{ gen_type::QUIETS }>(&b),
            HashSet::from([Move::normal(Square::E5, Square::E4)])
        );
    }

    /// A/H-file capture shifts must not wrap: each fixture places the only
    /// enemy piece exactly where the unmasked ±9/±7 shift would land.
    #[test]
    fn capture_shifts_do_not_wrap_files() {
        // White h5; unmasked NE (<<9) from h5 would land on a7.
        let b = board("4k3/p7/8/7P/8/8/8/4K3 w - - 0 1");
        assert!(pawn_move_set::<{ gen_type::CAPTURES }>(&b).is_empty());
        assert_eq!(
            pawn_move_set::<{ gen_type::QUIETS }>(&b),
            HashSet::from([Move::normal(Square::H5, Square::H6)])
        );

        // White a5; unmasked NW (<<7) from a5 would land on h5.
        let b = board("4k3/8/8/P6p/8/8/8/4K3 w - - 0 1");
        assert!(pawn_move_set::<{ gen_type::CAPTURES }>(&b).is_empty());

        // Black a4; unmasked SW (>>9) from a4 would land on h2.
        let b = board("4k3/8/8/8/p7/8/7P/4K3 b - - 0 1");
        assert!(pawn_move_set::<{ gen_type::CAPTURES }>(&b).is_empty());

        // Black h4; unmasked SE (>>7) from h4 would land on a3.
        let b = board("4k3/8/8/8/7p/P7/8/4K3 b - - 0 1");
        assert!(pawn_move_set::<{ gen_type::CAPTURES }>(&b).is_empty());
    }

    /// Non-capturing promotions (§3.2 table, right column of row 2): the
    /// queen goes to CAPTURES, the underpromotions to QUIETS, all four to
    /// NON_EVASIONS. Both colors.
    #[test]
    fn push_promotion_split() {
        for (fen, from, to) in [
            ("4k3/6P1/8/8/8/8/8/4K3 w - - 0 1", Square::G7, Square::G8),
            ("4k3/8/8/8/8/8/1p6/4K3 b - - 0 1", Square::B2, Square::B1),
        ] {
            let b = board(fen);
            assert_eq!(
                pawn_move_set::<{ gen_type::CAPTURES }>(&b),
                promo_fan(from, to, &[PieceType::Queen]),
                "CAPTURES in {fen}"
            );
            assert_eq!(
                pawn_move_set::<{ gen_type::QUIETS }>(&b),
                promo_fan(from, to, &UNDER_PROMOS),
                "QUIETS in {fen}"
            );
            assert_eq!(
                pawn_move_set::<{ gen_type::NON_EVASIONS }>(&b),
                promo_fan(from, to, &ALL_PROMOS),
                "NON_EVASIONS in {fen}"
            );
        }
    }

    /// Capturing promotions (§3.2 table row 1): all four pieces belong to
    /// CAPTURES in both directions; QUIETS gets only the *push*
    /// underpromotions. Both colors, both diagonals at once.
    #[test]
    fn promotion_captures_full_split() {
        for (fen, from, (left, up, right)) in [
            (
                "1n1nk3/2P5/8/8/8/8/8/4K3 w - - 0 1",
                Square::C7,
                (Square::B8, Square::C8, Square::D8),
            ),
            (
                "4k3/8/8/8/8/8/6p1/4KN1N b - - 0 1",
                Square::G2,
                (Square::F1, Square::G1, Square::H1),
            ),
        ] {
            let b = board(fen);
            let mut captures = promo_fan(from, left, &ALL_PROMOS);
            captures.extend(promo_fan(from, right, &ALL_PROMOS));
            captures.extend(promo_fan(from, up, &[PieceType::Queen]));
            let quiets = promo_fan(from, up, &UNDER_PROMOS);

            assert_eq!(
                pawn_move_set::<{ gen_type::CAPTURES }>(&b),
                captures,
                "CAPTURES in {fen}"
            );
            assert_eq!(
                pawn_move_set::<{ gen_type::QUIETS }>(&b),
                quiets,
                "QUIETS in {fen}"
            );
            let mut all = captures;
            all.extend(quiets);
            assert_eq!(
                pawn_move_set::<{ gen_type::NON_EVASIONS }>(&b),
                all,
                "NON_EVASIONS in {fen}"
            );
        }
    }

    /// Doc 05 §5.2's classic promotion position, both colors (the position
    /// is its own color mirror): blocked push promotions contribute nothing
    /// (a7/c7 resp. f2/h2), the free pawn gets the full 12-move fan.
    #[test]
    fn classic_promotion_position_partition() {
        for (fen, from, (left, up, right)) in [
            (
                "n1n5/PPPk4/8/8/8/8/4Kppp/5N1N w - - 0 1",
                Square::B7,
                (Square::A8, Square::B8, Square::C8),
            ),
            (
                "n1n5/PPPk4/8/8/8/8/4Kppp/5N1N b - - 0 1",
                Square::G2,
                (Square::F1, Square::G1, Square::H1),
            ),
        ] {
            let b = board(fen);
            let mut captures = promo_fan(from, left, &ALL_PROMOS);
            captures.extend(promo_fan(from, right, &ALL_PROMOS));
            captures.extend(promo_fan(from, up, &[PieceType::Queen]));
            let quiets = promo_fan(from, up, &UNDER_PROMOS);

            assert_eq!(
                pawn_move_set::<{ gen_type::CAPTURES }>(&b),
                captures,
                "CAPTURES in {fen}"
            );
            assert_eq!(
                pawn_move_set::<{ gen_type::QUIETS }>(&b),
                quiets,
                "QUIETS in {fen}"
            );
        }
    }

    /// En passant: emitted (with `MoveKind::EnPassant` and the recovered
    /// from-square) by CAPTURES and NON_EVASIONS, never by QUIETS; both
    /// capturers found when two pawns flank the target; nothing emitted
    /// without a recorded ep square.
    #[test]
    fn en_passant_generation() {
        fn ep_moves<const GT: u8>(b: &Board) -> HashSet<Move> {
            generated::<GT>(b)
                .iter()
                .copied()
                .filter(|m| m.kind() == MoveKind::EnPassant)
                .collect()
        }

        // Single white capturer (corpus opening after 1.e4 e6 2.e5 d5).
        let b = board("rnbqkbnr/ppp2ppp/4p3/3pP3/8/8/PPPP1PPP/RNBQKBNR w KQkq d6 0 3");
        let ep = Move::en_passant(Square::E5, Square::D6);
        assert_eq!(ep_moves::<{ gen_type::CAPTURES }>(&b), HashSet::from([ep]));
        assert_eq!(
            ep_moves::<{ gen_type::NON_EVASIONS }>(&b),
            HashSet::from([ep])
        );
        assert!(ep_moves::<{ gen_type::QUIETS }>(&b).is_empty());

        // Two white capturers flanking d5.
        let b = board("4k3/8/8/2PpP3/8/8/8/4K3 w - d6 0 1");
        assert_eq!(
            ep_moves::<{ gen_type::CAPTURES }>(&b),
            HashSet::from([
                Move::en_passant(Square::C5, Square::D6),
                Move::en_passant(Square::E5, Square::D6),
            ])
        );

        // Two black capturers flanking e4 (black's shifts and rank masks).
        let b = board("4k3/8/8/8/3pPp2/8/8/4K3 b - e3 0 1");
        assert_eq!(
            ep_moves::<{ gen_type::CAPTURES }>(&b),
            HashSet::from([
                Move::en_passant(Square::D4, Square::E3),
                Move::en_passant(Square::F4, Square::E3),
            ])
        );

        // No ep square recorded: same placement, zero ep moves.
        let b = board("4k3/8/8/2PpP3/8/8/8/4K3 w - - 0 1");
        assert!(ep_moves::<{ gen_type::NON_EVASIONS }>(&b).is_empty());
    }

    /// Doc 05 §5.4 order pins within pawn generation: pushes → promotions
    /// (Q, R, B, N) → captures → ep, then pieces and king — asserted as one
    /// exact emission sequence on a fixture exercising all four phases.
    #[test]
    fn pawn_emission_order_is_pinned() {
        let b = board("4k3/6P1/8/3pP3/2P5/8/P7/4K3 w - d6 0 2");
        let all = generated::<{ gen_type::NON_EVASIONS }>(&b);
        let expected = [
            // Single pushes (destination bitboard order), then doubles.
            Move::normal(Square::A2, Square::A3),
            Move::normal(Square::C4, Square::C5),
            Move::normal(Square::E5, Square::E6),
            Move::normal(Square::A2, Square::A4),
            // Push promotions, Q → R → B → N.
            Move::promotion(Square::G7, Square::G8, PieceType::Queen),
            Move::promotion(Square::G7, Square::G8, PieceType::Rook),
            Move::promotion(Square::G7, Square::G8, PieceType::Bishop),
            Move::promotion(Square::G7, Square::G8, PieceType::Knight),
            // Captures (up-right batch first), then en passant.
            Move::normal(Square::C4, Square::D5),
            Move::en_passant(Square::E5, Square::D6),
            // King last (no other pieces on the board).
            Move::normal(Square::E1, Square::D1),
            Move::normal(Square::E1, Square::F1),
            Move::normal(Square::E1, Square::D2),
            Move::normal(Square::E1, Square::E2),
            Move::normal(Square::E1, Square::F2),
        ];
        assert_eq!(&all[..], &expected[..]);
    }

    /// Pawn-heavy positions for the step-3 "Done" gate (doc 05 §4): the
    /// disjoint-union property and the §3.2 per-type semantics must hold
    /// with promotions and en passant in play. Castling rights present in
    /// some fixtures are inert until step 5 — the properties below are
    /// unaffected either way.
    const PAWN_FIXTURES: &[&str] = &[
        crate::fen::START_FEN,
        // Kiwipete.
        "r3k2r/p1ppqpb1/bn2pnp1/3PN3/1p2P3/2N2Q1p/PPPBBPPP/R3K2R w KQkq - 0 1",
        // Perft pos 3 (ep-pin geometry), pos 5 (promotion storm), pos 6
        // (quiet middlegame breadth). Pos 4 is excluded here on purpose:
        // its mover is in check, so it belongs to step 4's EVASIONS tests.
        "8/2p5/3p4/KP5r/1R3p1k/8/4P1P1/8 w - - 0 1",
        "rnbq1k1r/pp1Pbppp/2p5/8/2B5/8/PPP1NnPP/RNBQK2R w KQ - 1 8",
        "r4rk1/1pp1qppp/p1np1n2/2b1p1B1/2B1P1b1/P1NP1N2/1PP1QPPP/R4RK1 w - - 0 10",
        // The classic promotion position, both colors.
        "n1n5/PPPk4/8/8/8/8/4Kppp/5N1N w - - 0 1",
        "n1n5/PPPk4/8/8/8/8/4Kppp/5N1N b - - 0 1",
        // Live ep squares, one per color.
        "rnbqkbnr/ppp2ppp/4p3/3pP3/8/8/PPPP1PPP/RNBQKBNR w KQkq d6 0 3",
        "rnbqkbnr/pppp1ppp/8/8/4PpP1/8/PPPP3P/RNBQKBNR b KQkq g3 0 3",
    ];

    #[test]
    fn pawn_fixture_partition_and_semantics() {
        for &fen in PAWN_FIXTURES {
            let b = board(fen);
            let captures = move_set(&generated::<{ gen_type::CAPTURES }>(&b));
            let quiets = move_set(&generated::<{ gen_type::QUIETS }>(&b));
            let all = move_set(&generated::<{ gen_type::NON_EVASIONS }>(&b));

            // Doc 05 §5.1 property 1.
            assert!(
                captures.is_disjoint(&quiets),
                "CAPTURES ∩ QUIETS must be empty in {fen}"
            );
            let union: HashSet<Move> = captures.union(&quiets).copied().collect();
            assert_eq!(union, all, "CAPTURES ⊎ QUIETS != NON_EVASIONS in {fen}");

            // §3.2 semantics: everything CAPTURES emits is a real capture
            // or a queen promotion (the rule `capture_stage` will mirror in
            // step 6); QUIETS never captures, never queens, never ep.
            let enemies = b.occupancy(b.side_to_move().flip());
            for &m in &captures {
                let stage_capture = match m.kind() {
                    MoveKind::Normal => enemies.contains(m.to()),
                    MoveKind::Promotion => {
                        enemies.contains(m.to()) || m.promotion_piece() == Some(PieceType::Queen)
                    }
                    MoveKind::EnPassant => Some(m.to()) == b.en_passant_square(),
                    MoveKind::Castling => false,
                };
                assert!(stage_capture, "CAPTURES emitted {m} in {fen}");
            }
            for &m in &quiets {
                assert!(
                    !b.all_occupancy().contains(m.to()),
                    "QUIETS move {m} lands on a piece in {fen}"
                );
                assert_ne!(m.kind(), MoveKind::EnPassant, "QUIETS emitted ep in {fen}");
                if m.kind() == MoveKind::Promotion {
                    assert_ne!(
                        m.promotion_piece(),
                        Some(PieceType::Queen),
                        "QUIETS emitted a queen promotion in {fen}"
                    );
                }
            }
        }
    }

    // -----------------------------------------------------------------------
    // P3 step 4: evasion generation (doc 05 §3.6, §4 step 4, §5.2)
    // -----------------------------------------------------------------------

    use crate::types::Piece;

    /// The unrestricted pseudo-legal move set on an IN-CHECK position — the
    /// reference the evasion oracle filters. `generate::<NON_EVASIONS>`
    /// debug-asserts "not in check" (that assert is itself part of the
    /// contract under test), so the reference bypasses the dispatcher and
    /// calls the color worker directly.
    fn all_pseudo_ignoring_check(board: &Board) -> MoveList {
        let mut list = MoveList::new();
        match board.side_to_move() {
            Color::White => generate_all::<{ gen_type::NON_EVASIONS }, true>(board, &mut list),
            Color::Black => generate_all::<{ gen_type::NON_EVASIONS }, false>(board, &mut list),
        }
        list
    }

    /// Make/unmake legality oracle (doc 05 §5.1.5): play the move, ask
    /// whether our king is attacked, undo. Castling needs two extras the
    /// after-state alone cannot see: it is illegal by rule while in check
    /// (the king may still land on a safe square), and the king must not
    /// PASS THROUGH an attacked square — probed independently of
    /// `Board::legal` by playing an ordinary king step onto the transit
    /// square and reading the after-state of that.
    fn oracle_legal(board: &mut Board, m: Move) -> bool {
        let us = board.side_to_move();
        // A king step into contact with the enemy king is illegal by the
        // touch rule, resolved WITHOUT make/unmake: the resulting position
        // is outside make_move's contract (its debug gives_check
        // cross-check would see the moved king "checking" the other).
        if m.from() == board.king_square(us)
            && attacks::king_attacks(board.king_square(us.flip())).contains(m.to())
        {
            return false;
        }
        if m.kind() == MoveKind::Castling {
            if !board.checkers().is_empty() {
                return false;
            }
            let transit = if m.to().file() > m.from().file() {
                Square::F1
            } else {
                Square::D1
            }
            .relative_to(us);
            if !oracle_legal(board, Move::normal(m.from(), transit)) {
                return false;
            }
            // Fall through: the castle itself tests the landing square.
        }
        let gives_check = board.gives_check(m);
        board.make_move(m, gives_check);
        let safe = (board.attackers_to(board.king_square(us), board.all_occupancy())
            & board.occupancy(us.flip()))
        .is_empty();
        board.unmake_move(m);
        safe
    }

    /// Doc 05 §4 step 4 completeness gate: EVASIONS, filtered through the
    /// make/unmake oracle, must equal the full pseudo-legal superset under
    /// the same filter — no legal block/capture/flee missing, nothing
    /// extra. Pins the §3.6 shape rules along the way.
    fn assert_evasion_completeness(b: &Board, fen: &str) {
        let us = b.side_to_move();
        let ksq = b.king_square(us);
        assert!(!b.checkers().is_empty(), "fixture must be in check: {fen}");

        let evasions = move_set(&generated::<{ gen_type::EVASIONS }>(b));
        let superset = move_set(&all_pseudo_ignoring_check(b));
        assert!(
            evasions.is_subset(&superset),
            "EVASIONS must be a subset of the pseudo-legal superset in {fen}"
        );

        // §3.6 rule 5: castling is never an evasion. Rule 2: double check
        // is king-only. Rule 3: single-check non-king moves land on the
        // block-or-capture mask (en passant resolves on the captured
        // pawn's square, not on `to`, so it is exempt from the to-test).
        for &m in &evasions {
            assert_ne!(m.kind(), MoveKind::Castling, "castle evasion {m} in {fen}");
        }
        if b.checkers().more_than_one() {
            for &m in &evasions {
                assert_eq!(
                    m.from(),
                    ksq,
                    "non-king evasion {m} under double check in {fen}"
                );
            }
        } else {
            let target = attacks::between(ksq, b.checkers().lsb().unwrap());
            for &m in &evasions {
                if m.from() != ksq && m.kind() != MoveKind::EnPassant {
                    assert!(
                        target.contains(m.to()),
                        "evasion {m} off the block/capture ray in {fen}"
                    );
                }
            }
        }

        let mut scratch = b.clone();
        let legal_evasions: HashSet<Move> = evasions
            .into_iter()
            .filter(|&m| oracle_legal(&mut scratch, m))
            .collect();
        let legal_superset: HashSet<Move> = superset
            .into_iter()
            .filter(|&m| oracle_legal(&mut scratch, m))
            .collect();
        assert_eq!(
            legal_evasions, legal_superset,
            "evasion completeness in {fen}"
        );
    }

    /// In-check corpus: single checks with blocks/captures/flights, pawn
    /// and knight checks, ep-as-evasion, promotion evasions, double
    /// checks, castling rights while in check, x-ray flights, and one
    /// checkmate (pseudo evasions exist, none legal).
    const IN_CHECK_FIXTURES: &[&str] = &[
        // Perft pos 4 (white in check by Bb6) and its color mirror.
        "r3k2r/Pppp1ppp/1b3nbN/nP6/BBP1P3/q4N2/Pp1P2PP/R2Q1RK1 w kq - 0 1",
        "r2q1rk1/pP1p2pp/Q4n2/bbp1p3/Np6/1B3NBn/pPPP1PPP/R3K2R b KQ - 0 1",
        // Rook check: capture (Nxe4), interposition (Be3), flights.
        "4k3/8/8/8/4r3/8/3N4/4K1B1 w - - 0 1",
        // The checker is pawn-capturable; a second enemy piece is not.
        "k7/8/8/8/3r1b2/4P3/7K/8 w - - 0 1",
        // Interposition by a double pawn push through the relay square.
        "8/2p5/8/k6R/8/8/8/4K3 b - - 0 1",
        // Ep capture of the checking double-pushed pawn.
        "8/8/8/2k5/3Pp3/8/8/4K3 b - d3 0 1",
        // Capturing promotions resolve the check; push promotions cannot.
        "4r3/3P1P2/8/8/8/8/8/4K2k w - - 0 1",
        // A push promotion lands on the checking ray and blocks.
        "K6r/2P5/8/8/8/8/8/7k w - - 0 1",
        // Knight check: no interposition square exists.
        "4k3/8/8/8/8/5n2/8/4K3 w - - 0 1",
        // Pawn check, capturable by knight or king.
        "4k3/8/8/8/8/8/3p4/1N2K3 w - - 0 1",
        // Double checks: king moves only.
        "r3k3/8/8/1B6/8/8/8/4RK2 b - - 0 1",
        "k3r3/8/8/8/8/5n2/8/4K3 w - - 0 1",
        // Castling rights present while in check (no castle evasions).
        "r3k2r/8/8/8/B7/8/8/4K3 b kq - 0 1",
        // King on the checker's ray: the x-ray flight e5 and the on-ray
        // step e3 are pseudo-generated, both oracle-illegal.
        "4k3/8/8/8/4K3/8/8/4r3 w - - 0 1",
        // Fool's mate: the lone pseudo evasion Kf2 is illegal.
        "rnb1kbnr/pppp1ppp/8/4p3/6Pq/5P2/PPPPP2P/RNBQKBNR w KQkq - 1 3",
    ];

    #[test]
    fn evasions_complete_on_in_check_fixtures() {
        for &fen in IN_CHECK_FIXTURES {
            assert_evasion_completeness(&board(fen), fen);
        }
    }

    /// Exact block/capture/flee set for a single rook check (doc 05 §5.2):
    /// Nd2 captures the checker, Bg1 interposes on e3, the king has four
    /// flights (d2 is own-occupied). Ke2 stays on the checker's file:
    /// pseudo-generated — flight safety is legal()'s job (§3.6 rule 4) —
    /// and rejected by the oracle.
    #[test]
    fn single_check_block_capture_flee() {
        let b = board("4k3/8/8/8/4r3/8/3N4/4K1B1 w - - 0 1");
        let expected: HashSet<Move> = [
            Move::normal(Square::D2, Square::E4),
            Move::normal(Square::G1, Square::E3),
            Move::normal(Square::E1, Square::D1),
            Move::normal(Square::E1, Square::E2),
            Move::normal(Square::E1, Square::F1),
            Move::normal(Square::E1, Square::F2),
        ]
        .into_iter()
        .collect();
        assert_eq!(move_set(&generated::<{ gen_type::EVASIONS }>(&b)), expected);

        let mut scratch = b.clone();
        assert!(!oracle_legal(
            &mut scratch,
            Move::normal(Square::E1, Square::E2)
        ));
        assert!(oracle_legal(
            &mut scratch,
            Move::normal(Square::D2, Square::E4)
        ));
    }

    /// Double check emits king moves only (doc 05 §3.6 rule 2, via
    /// `more_than_one`) — including flights that legal() will reject.
    #[test]
    fn double_check_generates_king_moves_only() {
        // Re1 + Bb5 both check e8; the a8 rook could otherwise block.
        let b = board("r3k3/8/8/1B6/8/8/8/4RK2 b - - 0 1");
        assert!(b.checkers().more_than_one());
        let expected: HashSet<Move> = [Square::D7, Square::D8, Square::E7, Square::F7, Square::F8]
            .into_iter()
            .map(|to| Move::normal(Square::E8, to))
            .collect();
        assert_eq!(move_set(&generated::<{ gen_type::EVASIONS }>(&b)), expected);

        // Nf3 + Re8 both check e1 (white side of the color dispatch).
        let b = board("k3r3/8/8/8/8/5n2/8/4K3 w - - 0 1");
        assert!(b.checkers().more_than_one());
        let expected: HashSet<Move> = [Square::D1, Square::D2, Square::E2, Square::F1, Square::F2]
            .into_iter()
            .map(|to| Move::normal(Square::E1, to))
            .collect();
        assert_eq!(move_set(&generated::<{ gen_type::EVASIONS }>(&b)), expected);
    }

    /// §3.4 `enemies = checkers()`: the e3 pawn attacks both d4 (a rook
    /// that is NOT the checker) and f4 (the checking bishop); only the
    /// checker capture is emitted.
    #[test]
    fn evasion_pawn_captures_only_the_checker() {
        let b = board("k7/8/8/8/3r1b2/4P3/7K/8 w - - 0 1");
        let expected: HashSet<Move> = [
            Move::normal(Square::E3, Square::F4),
            Move::normal(Square::H2, Square::G1),
            Move::normal(Square::H2, Square::G2),
            Move::normal(Square::H2, Square::G3),
            Move::normal(Square::H2, Square::H1),
            Move::normal(Square::H2, Square::H3),
        ]
        .into_iter()
        .collect();
        let evasions = move_set(&generated::<{ gen_type::EVASIONS }>(&b));
        assert!(!evasions.contains(&Move::normal(Square::E3, Square::D4)));
        assert_eq!(evasions, expected);
    }

    /// Rh5 checks a5 along rank 5; c7-c5 — a double push through the empty
    /// relay square — is the only non-king evasion (the single push c6
    /// misses the ray and must NOT appear).
    #[test]
    fn evasion_pawn_interposition_by_double_push() {
        let b = board("8/2p5/8/k6R/8/8/8/4K3 b - - 0 1");
        let expected: HashSet<Move> = [
            Move::normal(Square::C7, Square::C5),
            Move::normal(Square::A5, Square::A4),
            Move::normal(Square::A5, Square::A6),
            Move::normal(Square::A5, Square::B4),
            Move::normal(Square::A5, Square::B5),
            Move::normal(Square::A5, Square::B6),
        ]
        .into_iter()
        .collect();
        assert_eq!(move_set(&generated::<{ gen_type::EVASIONS }>(&b)), expected);
    }

    /// Promotions under EVASIONS (§3.2 table): the full four-piece fan is
    /// emitted, but only where it resolves the check — capturing the
    /// checker on e8, or landing a push promotion on the checking ray.
    #[test]
    fn evasion_promotions() {
        // Re8 checks e1: dxe8 and fxe8 capture the checker (full fans);
        // the non-resolving pushes d8/f8 are masked out.
        let b = board("4r3/3P1P2/8/8/8/8/8/4K2k w - - 0 1");
        let mut expected = promo_fan(Square::D7, Square::E8, &ALL_PROMOS);
        expected.extend(promo_fan(Square::F7, Square::E8, &ALL_PROMOS));
        for to in [Square::D1, Square::D2, Square::E2, Square::F1, Square::F2] {
            expected.insert(Move::normal(Square::E1, to));
        }
        assert_eq!(move_set(&generated::<{ gen_type::EVASIONS }>(&b)), expected);

        // Rh8 checks a8: the push promotion c8 lands ON the ray and
        // blocks (`b3 &= target` keeps it), again with the full fan.
        let b = board("K6r/2P5/8/8/8/8/8/7k w - - 0 1");
        let mut expected = promo_fan(Square::C7, Square::C8, &ALL_PROMOS);
        for to in [Square::A7, Square::B7, Square::B8] {
            expected.insert(Move::normal(Square::A8, to));
        }
        assert_eq!(move_set(&generated::<{ gen_type::EVASIONS }>(&b)), expected);
    }

    /// Doc 05 §5.2 "ep as evasion": white just played d2-d4 giving check
    /// with the pawn itself; exd3 en passant captures the checker. D9
    /// keeps the ep square (the capture is legal), and the ep move is
    /// exempt from the to-on-ray mask — its resolution square is d4.
    #[test]
    fn ep_capture_of_the_checking_pawn_is_an_evasion() {
        let b = board("8/8/8/2k5/3Pp3/8/8/4K3 b - d3 0 1");
        assert_eq!(b.en_passant_square(), Some(Square::D3));
        let ep = Move::en_passant(Square::E4, Square::D3);
        let mut expected = HashSet::from([ep]);
        for to in [
            Square::B4,
            Square::B5,
            Square::B6,
            Square::C4,
            Square::C6,
            Square::D4,
            Square::D5,
            Square::D6,
        ] {
            expected.insert(Move::normal(Square::C5, to));
        }
        assert_eq!(move_set(&generated::<{ gen_type::EVASIONS }>(&b)), expected);
        assert!(oracle_legal(&mut b.clone(), ep));
    }

    /// Doc 05 §5.2 "ep discovered-check skip": d2-d4 opened the c1-g5
    /// diagonal, so Bc1 checks THROUGH the vacated push origin and exd3
    /// can never resolve the check (§3.4). Through FEN, D9 already drops
    /// the unusable ep square at load; the generator's own skip branch is
    /// defense-in-depth, exercised by forcing the D9-violating state onto
    /// the board directly.
    #[test]
    fn ep_skipped_when_check_is_discovered_through_the_push_origin() {
        let b = board("8/8/8/6k1/3Pp3/8/8/2B1K3 b - d3 0 1");
        assert_eq!(b.en_passant_square(), None, "D9 must drop the ep square");
        let evasions = move_set(&generated::<{ gen_type::EVASIONS }>(&b));
        assert!(evasions.iter().all(|m| m.kind() != MoveKind::EnPassant));
        // The e4 pawn still blocks on the diagonal with a plain push.
        assert!(evasions.contains(&Move::normal(Square::E4, Square::E3)));

        // Same placement with ep pinned to d3: now the §3.4 skip (not D9)
        // must reject the ep capture, while everything else is unchanged.
        let mut forced = Board::empty();
        for (piece, sq) in [
            (Piece::new(Color::White, PieceType::Bishop), Square::C1),
            (Piece::new(Color::White, PieceType::King), Square::E1),
            (Piece::new(Color::White, PieceType::Pawn), Square::D4),
            (Piece::new(Color::Black, PieceType::Pawn), Square::E4),
            (Piece::new(Color::Black, PieceType::King), Square::G5),
        ] {
            forced.put_piece(piece, sq);
        }
        forced.side_to_move = Color::Black;
        forced.finish_setup(Some(Square::D3), 0, 1);
        assert_eq!(forced.en_passant_square(), Some(Square::D3));
        let evasions = move_set(&generated::<{ gen_type::EVASIONS }>(&forced));
        assert!(evasions.iter().all(|m| m.kind() != MoveKind::EnPassant));
        assert!(evasions.contains(&Move::normal(Square::E4, Square::E3)));
        assert_evasion_completeness(&forced, "forced ep-discovered-check fixture");
    }

    /// §3.6 rule 5: castling is never generated under EVASIONS, even with
    /// both rights present and both paths empty — while the unrestricted
    /// superset on the same position emits both castles, proving the
    /// exclusion is the EVASIONS rule and not a rights/path failure.
    #[test]
    fn no_castling_under_evasions() {
        let b = board("r3k2r/8/8/8/B7/8/8/4K3 b kq - 0 1");
        assert!(!b.checkers().is_empty());
        let evasions = generated::<{ gen_type::EVASIONS }>(&b);
        assert!(evasions.iter().all(|m| m.kind() != MoveKind::Castling));
        let superset = move_set(&all_pseudo_ignoring_check(&b));
        assert!(superset.contains(&Move::castling(Square::E8, Square::G8)));
        assert!(superset.contains(&Move::castling(Square::E8, Square::C8)));
    }

    /// The other direction of the §3.1 debug_assert: requesting
    /// NON_EVASIONS while in check panics in debug builds.
    #[cfg(debug_assertions)]
    #[test]
    #[should_panic(expected = "generate::<EVASIONS> iff the side to move is in check")]
    fn non_evasions_while_in_check_panics_in_debug() {
        let b = board("4k3/8/8/8/8/5n2/8/4K3 w - - 0 1");
        let mut list = MoveList::new();
        generate_non_evasions(&b, &mut list);
    }

    // -----------------------------------------------------------------------
    // P3 step 5: castling generation (doc 05 §3.7, §4 step 5, §5.2)
    // -----------------------------------------------------------------------

    const KIWIPETE: &str = "r3k2r/p1ppqpb1/bn2pnp1/3PN3/1p2P3/2N2Q1p/PPPBBPPP/R3K2R w KQkq - 0 1";

    /// The step-5 "Done" gate (doc 05 §4): Kiwipete emits both white
    /// castles in QUIETS — as the last two moves, kingside first (§5.4
    /// order pin) — and in NON_EVASIONS, never in CAPTURES.
    #[test]
    fn kiwipete_emits_both_white_castles_in_quiets() {
        let b = board(KIWIPETE);
        let oo = Move::castling(Square::E1, Square::G1);
        let ooo = Move::castling(Square::E1, Square::C1);

        let quiets = generated::<{ gen_type::QUIETS }>(&b);
        assert_eq!(&quiets[quiets.len() - 2..], &[oo, ooo]);

        let all = move_set(&generated::<{ gen_type::NON_EVASIONS }>(&b));
        assert!(all.contains(&oo) && all.contains(&ooo));
        assert!(
            generated::<{ gen_type::CAPTURES }>(&b)
                .iter()
                .all(|m| m.kind() != MoveKind::Castling)
        );
    }

    /// Black's rank-8 constants via Kiwipete with the side to move flipped
    /// (black is not in check there).
    #[test]
    fn kiwipete_reversed_emits_both_black_castles() {
        let b = board("r3k2r/p1ppqpb1/bn2pnp1/3PN3/1p2P3/2N2Q1p/PPPBBPPP/R3K2R b KQkq - 0 1");
        let quiets = generated::<{ gen_type::QUIETS }>(&b);
        assert_eq!(
            &quiets[quiets.len() - 2..],
            &[
                Move::castling(Square::E8, Square::G8),
                Move::castling(Square::E8, Square::C8),
            ]
        );
    }

    /// §3.7 condition 1: each right gates exactly its own castle.
    #[test]
    fn castling_requires_the_right() {
        for (fen, kingside, queenside) in [
            ("4k3/8/8/8/8/8/8/R3K2R w KQ - 0 1", true, true),
            ("4k3/8/8/8/8/8/8/R3K2R w K - 0 1", true, false),
            ("4k3/8/8/8/8/8/8/R3K2R w Q - 0 1", false, true),
            ("4k3/8/8/8/8/8/8/R3K2R w - - 0 1", false, false),
            ("r3k2r/8/8/8/8/8/8/4K3 b kq - 0 1", true, true),
            ("r3k2r/8/8/8/8/8/8/4K3 b k - 0 1", true, false),
            ("r3k2r/8/8/8/8/8/8/4K3 b q - 0 1", false, true),
        ] {
            let b = board(fen);
            let us = b.side_to_move();
            let from = Square::E1.relative_to(us);
            let quiets = move_set(&generated::<{ gen_type::QUIETS }>(&b));
            assert_eq!(
                quiets.contains(&Move::castling(from, Square::G1.relative_to(us))),
                kingside,
                "kingside in {fen}"
            );
            assert_eq!(
                quiets.contains(&Move::castling(from, Square::C1.relative_to(us))),
                queenside,
                "queenside in {fen}"
            );
        }
    }

    /// §3.7 condition 2, one blocker per path square (doc 05 §4 step 5):
    /// any piece — own or enemy — on f1/g1 kills the kingside castle and
    /// on b1/c1/d1 the queenside one, without disturbing the other wing.
    /// b1 is rook-path-only (the king never crosses it) yet still blocks.
    #[test]
    fn castling_blocked_on_each_path_square() {
        for (fen, kingside, queenside) in [
            ("4k3/8/8/8/8/8/8/R3KN1R w KQ - 0 1", false, true), // f1 own
            ("4k3/8/8/8/8/8/8/R3K1NR w KQ - 0 1", false, true), // g1 own
            ("4k3/8/8/8/8/8/8/RN2K2R w KQ - 0 1", true, false), // b1 own
            ("4k3/8/8/8/8/8/8/R1N1K2R w KQ - 0 1", true, false), // c1 own
            ("4k3/8/8/8/8/8/8/R2NK2R w KQ - 0 1", true, false), // d1 own
            ("4k3/8/8/8/8/8/8/Rn2K2R w KQ - 0 1", true, false), // b1 enemy
        ] {
            let b = board(fen);
            let quiets = move_set(&generated::<{ gen_type::QUIETS }>(&b));
            assert_eq!(
                quiets.contains(&Move::castling(Square::E1, Square::G1)),
                kingside,
                "kingside in {fen}"
            );
            assert_eq!(
                quiets.contains(&Move::castling(Square::E1, Square::C1)),
                queenside,
                "queenside in {fen}"
            );
        }
    }

    /// Attacked-square testing is deferred to `Board::legal` (§3.7): the
    /// f3 queen attacks f1 (kingside crossing) and d1 (queenside crossing)
    /// without giving check, yet generation still emits both castles —
    /// step 6's legal() is what will drop them.
    #[test]
    fn castling_generated_through_attacked_squares() {
        let b = board("4k3/8/8/8/8/5q2/8/R3K2R w KQ - 0 1");
        assert!(b.checkers().is_empty());
        let quiets = move_set(&generated::<{ gen_type::QUIETS }>(&b));
        assert!(quiets.contains(&Move::castling(Square::E1, Square::G1)));
        assert!(quiets.contains(&Move::castling(Square::E1, Square::C1)));
    }

    // -----------------------------------------------------------------------
    // P3 step 6: Board::legal, generate_legal, capture_stage / is_capture
    // (doc 05 §3.2, §3.8-3.9, §4 step 6, §5.1-5.2)
    // -----------------------------------------------------------------------

    fn legal_set(b: &Board) -> HashSet<Move> {
        let mut list = MoveList::new();
        generate_legal(b, &mut list);
        move_set(&list)
    }

    /// The pseudo-legal set that legality filters: EVASIONS in check, else
    /// NON_EVASIONS (doc 05 §3.9).
    fn pseudo_list(b: &Board) -> MoveList {
        let mut list = MoveList::new();
        if b.checkers().is_empty() {
            generate::<{ gen_type::NON_EVASIONS }>(b, &mut list);
        } else {
            generate::<{ gen_type::EVASIONS }>(b, &mut list);
        }
        list
    }

    /// Doc 05 §5.1.5 (the legality oracle) plus the §3.9 filter contract
    /// in one sweep: `legal` must agree with make/unmake on EVERY
    /// pseudo-legal move — including the non-suspect ones generate_legal
    /// trusts without testing — and `generate_legal` must equal the
    /// oracle-filtered set.
    fn assert_legal_matches_oracle(b: &Board, ctx: &str) {
        let pseudo = pseudo_list(b);
        let mut scratch = b.clone();
        let mut expected = HashSet::new();
        for &m in pseudo.iter() {
            let want = oracle_legal(&mut scratch, m);
            assert_eq!(
                b.legal(m),
                want,
                "legal({m:?}) disagrees with the make/unmake oracle in {ctx}"
            );
            if want {
                expected.insert(m);
            }
        }
        assert_eq!(legal_set(b), expected, "generate_legal mismatch in {ctx}");
    }

    /// Depth-1 legal-move counts: the six standard perft positions — the
    /// step-6 "Done" gate (doc 05 §4, §5.3) — plus the doc 06 §3.4 tricky
    /// table (ep pins, the castling ladder, promotion checks, stalemate
    /// corners). All counts are the published / python-chess-verified d1
    /// values from those tables.
    const PERFT1_COUNTS: &[(&str, usize)] = &[
        (crate::fen::START_FEN, 20),
        (KIWIPETE, 48),
        ("8/2p5/3p4/KP5r/1R3p1k/8/4P1P1/8 w - - 0 1", 14),
        (
            "r3k2r/Pppp1ppp/1b3nbN/nP6/BBP1P3/q4N2/Pp1P2PP/R2Q1RK1 w kq - 0 1",
            6,
        ),
        (
            "rnbq1k1r/pp1Pbppp/2p5/8/2B5/8/PPP1NnPP/RNBQK2R w KQ - 1 8",
            44,
        ),
        (
            "r4rk1/1pp1qppp/p1np1n2/2b1p1B1/2B1P1b1/P1NP1N2/1PP1QPPP/R4RK1 w - - 0 10",
            46,
        ),
        // doc 06 §3.4 tricky positions, by their table names:
        ("3k4/3p4/8/K1P4r/8/8/8/8 b - - 0 1", 18), // ep-pin-h
        ("8/8/8/8/k2Pp2Q/8/8/3K4 b - d3 0 1", 6),  // ep-pin-rank (D9 drops ep)
        ("8/8/4k3/8/2p5/8/B2P2K1/8 w - - 0 1", 13), // ep-pin-diag
        ("8/8/1k6/2b5/2pP4/8/5K2/8 b - d3 0 1", 15), // ep-gives-check
        ("r3k2r/1b4bq/8/8/8/8/7B/R3K2R w KQkq - 0 1", 26), // castle-rights
        ("r3k2r/8/3Q4/8/8/8/8/R3K2R b KQkq - 0 1", 20), // castle-prevented
        ("5k2/8/8/8/8/8/8/4K2R w K - 0 1", 15),    // castle-check-k
        ("3k4/8/8/8/8/8/8/R3K3 w Q - 0 1", 16),    // castle-check-q
        ("n1n5/PPPk4/8/8/8/8/4Kppp/5N1N b - - 0 1", 24), // promo-storm
        ("2K2r2/4P3/8/8/8/8/8/3k4 w - - 0 1", 11), // promo-out-of-check
        ("4k3/1P6/8/8/8/8/K7/8 w - - 0 1", 9),     // promo-give-check
        ("8/P1k5/K7/8/8/8/8/8 w - - 0 1", 6),      // underpromo-check
        ("4k3/8/3N4/8/8/8/8/4RK2 b - - 0 1", 3),   // double-check
        ("8/8/1P2K3/8/2n5/1q6/8/5k2 b - - 0 1", 29), // discovered-check
        ("K1k5/8/P7/8/8/8/8/8 w - - 0 1", 2),      // self-stalemate
        ("8/k1P5/8/1K6/8/8/8/8 w - - 0 1", 10),    // stale-checkmate
        ("8/8/2k5/5q2/5n2/8/5K2/8 b - - 0 1", 37), // double-checkmate2
    ];

    /// The step-6 "Done" gate (doc 05 §4): perft(1) via `generate_legal`
    /// alone, no make/unmake needed.
    #[test]
    fn perft1_gate_on_known_positions() {
        for &(fen, expected) in PERFT1_COUNTS {
            assert_eq!(
                legal_set(&board(fen)).len(),
                expected,
                "perft(1) via generate_legal in {fen}"
            );
        }
    }

    /// Doc 05 §5.1.5 over the whole corpus: hand fixtures, pawn-heavy
    /// positions, every in-check fixture, and the perft d1 table.
    #[test]
    fn legal_agrees_with_the_make_unmake_oracle() {
        for &(fen, _, _) in FIXTURES {
            assert_legal_matches_oracle(&board(fen), fen);
        }
        for &fen in PAWN_FIXTURES {
            assert_legal_matches_oracle(&board(fen), fen);
        }
        for &fen in IN_CHECK_FIXTURES {
            assert_legal_matches_oracle(&board(fen), fen);
        }
        for &(fen, _) in PERFT1_COUNTS {
            assert_legal_matches_oracle(&board(fen), fen);
        }
    }

    /// Doc 05 §5.2 "horizontal ep double-removal": capturing en passant
    /// removes BOTH pawns from rank 4 and exposes Ka4 to Qh4. D9 already
    /// drops the ep square on FEN load (defense layer 1); forcing the
    /// state onto the board exercises legal()'s occupancy-swap branch
    /// (defense layer 2 — the §9 Q1 deviation from current SF).
    #[test]
    fn horizontal_ep_double_removal_is_illegal() {
        let fen = "8/8/8/8/k2Pp2Q/8/8/3K4 b - d3 0 1";
        assert_eq!(
            board(fen).en_passant_square(),
            None,
            "D9 must drop the unusable ep square at FEN load"
        );

        let mut forced = Board::empty();
        for (piece, sq) in [
            (Piece::new(Color::White, PieceType::King), Square::D1),
            (Piece::new(Color::White, PieceType::Pawn), Square::D4),
            (Piece::new(Color::White, PieceType::Queen), Square::H4),
            (Piece::new(Color::Black, PieceType::King), Square::A4),
            (Piece::new(Color::Black, PieceType::Pawn), Square::E4),
        ] {
            forced.put_piece(piece, sq);
        }
        forced.side_to_move = Color::Black;
        forced.finish_setup(Some(Square::D3), 0, 1);
        assert_eq!(forced.en_passant_square(), Some(Square::D3));
        assert!(
            forced.checkers().is_empty(),
            "the queen's rank-4 ray is blocked by the two pawns"
        );

        let ep = Move::en_passant(Square::E4, Square::D3);
        assert!(
            move_set(&pseudo_list(&forced)).contains(&ep),
            "exd3 must still be pseudo-generated"
        );
        assert!(!forced.legal(ep), "exd3 exposes the king along rank 4");
        assert!(!legal_set(&forced).contains(&ep));
        assert_legal_matches_oracle(&forced, "forced horizontal ep pin");
    }

    /// The DIAGONAL half of [`Board::legal_en_passant`], which nothing in the
    /// repo reached (D33 reviewer C, perf study 17 §9.2). Every other
    /// en-passant fixture — here, in `board.rs`, and in `tests/data/perft.epd`
    /// — is a rook/rank case, so deleting that arm's `bishop_attacks` term was
    /// killed only by an aggregate perft total, which cannot localise a fault.
    ///
    /// Two capturers, and only one of them is pinned: the black king sits on
    /// e6, a white bishop on a2, and the a2–e6 diagonal runs a2-b3-c4-d5-e6.
    /// `c4xd3` e.p. steps the c4 pawn OFF that diagonal and exposes the king;
    /// `e4xd3` e.p. does not. **The second capturer is what makes the test
    /// work**: with only the pinned one, D9's ep-validity gate
    /// (`make_move_with` step 8) proves no ep capture can be legal and never
    /// sets the ep square at all, so this arm is never reached. That is why
    /// `perft.epd`'s `ep-pin-diag` does NOT cover it — verified by mutation,
    /// deleting the bishop term leaves that FEN's perft totals unchanged.
    #[test]
    fn diagonal_ep_pin_is_illegal_for_the_pinned_capturer_only() {
        let mut forced = Board::empty();
        for (piece, sq) in [
            (Piece::new(Color::White, PieceType::King), Square::G2),
            (Piece::new(Color::White, PieceType::Bishop), Square::A2),
            (Piece::new(Color::White, PieceType::Pawn), Square::D4),
            (Piece::new(Color::Black, PieceType::King), Square::E6),
            (Piece::new(Color::Black, PieceType::Pawn), Square::C4),
            (Piece::new(Color::Black, PieceType::Pawn), Square::E4),
        ] {
            forced.put_piece(piece, sq);
        }
        forced.side_to_move = Color::Black;
        forced.finish_setup(Some(Square::D3), 0, 1);
        assert_eq!(forced.en_passant_square(), Some(Square::D3));
        assert!(
            forced.checkers().is_empty(),
            "the bishop's a2-e6 diagonal is blocked by the c4 pawn"
        );

        let pinned = Move::en_passant(Square::C4, Square::D3);
        let free = Move::en_passant(Square::E4, Square::D3);
        let pseudo = move_set(&pseudo_list(&forced));
        assert!(pseudo.contains(&pinned) && pseudo.contains(&free));

        assert!(
            !forced.legal(pinned),
            "cxd3 e.p. leaves the a2-e6 diagonal open onto the black king"
        );
        assert!(
            forced.legal(free),
            "exd3 e.p. is off the pin line and legal"
        );
        assert!(!legal_set(&forced).contains(&pinned));
        assert!(legal_set(&forced).contains(&free));
        assert_legal_matches_oracle(&forced, "forced diagonal ep pin");
    }

    /// Doc 05 §5.2 "ep as evasion": the checker IS the double-pushed pawn,
    /// so exd3 must survive the legality filter.
    #[test]
    fn ep_evasion_survives_the_legality_filter() {
        let b = board("8/8/8/2k5/3Pp3/8/8/4K3 b - d3 0 1");
        assert!(legal_set(&b).contains(&Move::en_passant(Square::E4, Square::D3)));
    }

    /// Doc 05 §5.2 "castling through check": generation emits the castles
    /// (path empty), legal() drops exactly the ones whose king-transit
    /// squares are attacked — and an attack on the rook-only path square
    /// (b1, never crossed by the king) drops neither.
    #[test]
    fn castling_transit_attacks_are_dropped_by_legal() {
        let oo = Move::castling(Square::E1, Square::G1);
        let ooo = Move::castling(Square::E1, Square::C1);

        // Qf3 attacks f1 (kingside transit) and d1 (queenside transit).
        let b = board("4k3/8/8/8/8/5q2/8/R3K2R w KQ - 0 1");
        let pseudo = move_set(&pseudo_list(&b));
        assert!(pseudo.contains(&oo) && pseudo.contains(&ooo));
        assert!(!b.legal(oo) && !b.legal(ooo));
        let legal = legal_set(&b);
        assert!(!legal.contains(&oo) && !legal.contains(&ooo));

        // Rg8 attacks only g1: kingside dropped, queenside survives.
        let b = board("4k1r1/8/8/8/8/8/8/R3K2R w KQ - 0 1");
        assert!(!b.legal(oo));
        assert!(b.legal(ooo));

        // Rb8 attacks only b1: both castles stay legal.
        let b = board("1r2k3/8/8/8/8/8/8/R3K2R w KQ - 0 1");
        assert!(b.legal(oo) && b.legal(ooo));

        // Kiwipete step-6 sanity: both white castles fully legal.
        let legal = legal_set(&board(KIWIPETE));
        assert!(legal.contains(&oo) && legal.contains(&ooo));
    }

    /// Doc 05 §5.2 "pinned-piece filter": the pinned piece may slide along
    /// the pin ray in BOTH directions — toward or away from the king,
    /// including capturing the pinner — but never off it.
    #[test]
    fn pinned_piece_moves_along_the_ray_only() {
        // Re2 pinned by Re7 on the e-file: all file moves stay legal.
        let b = board("4k3/4r3/8/8/8/8/4R3/4K3 w - - 0 1");
        let mut expected: HashSet<Move> =
            [Square::E3, Square::E4, Square::E5, Square::E6, Square::E7]
                .into_iter()
                .map(|to| Move::normal(Square::E2, to))
                .collect();
        for to in [Square::D1, Square::D2, Square::F1, Square::F2] {
            expected.insert(Move::normal(Square::E1, to));
        }
        assert_eq!(legal_set(&b), expected);
        assert!(!b.legal(Move::normal(Square::E2, Square::D2)));
        assert!(!b.legal(Move::normal(Square::E2, Square::A2)));

        // d2 pawn pinned by Bc3 diagonally: capturing the pinner is the
        // pawn's ONLY legal move (both pushes leave the ray).
        let b = board("4k3/8/8/8/8/2b5/3P4/4K3 w - - 0 1");
        let legal = legal_set(&b);
        assert!(legal.contains(&Move::normal(Square::D2, Square::C3)));
        assert!(!legal.contains(&Move::normal(Square::D2, Square::D3)));
        assert!(!legal.contains(&Move::normal(Square::D2, Square::D4)));
    }

    /// Doc 05 §5.2 "double check": every legal move is a king move, and
    /// the on-ray flights are gone (king-removed occupancy test).
    #[test]
    fn double_check_legal_moves_are_king_only() {
        // Re1 + Bb5: d7 (bishop ray) and e7 (rook x-ray through e8) are
        // pseudo-generated flights that legal() rejects.
        let b = board("r3k3/8/8/1B6/8/8/8/4RK2 b - - 0 1");
        let expected: HashSet<Move> = [Square::D8, Square::F7, Square::F8]
            .into_iter()
            .map(|to| Move::normal(Square::E8, to))
            .collect();
        assert_eq!(legal_set(&b), expected);
    }

    /// The §3.2 predicate table, one move per class.
    #[test]
    fn capture_stage_and_is_capture_by_kind() {
        // Normal capture / quiet.
        let b = board("4k3/8/8/3p4/8/2N5/8/4K3 w - - 0 1");
        let cap = Move::normal(Square::C3, Square::D5);
        let quiet = Move::normal(Square::C3, Square::E4);
        assert!(b.capture_stage(cap) && b.is_capture(cap));
        assert!(!b.capture_stage(quiet) && !b.is_capture(quiet));

        // En passant: a capture even though `to` is empty.
        let b = board("rnbqkbnr/ppp2ppp/4p3/3pP3/8/8/PPPP1PPP/RNBQKBNR w KQkq d6 0 3");
        let ep = Move::en_passant(Square::E5, Square::D6);
        assert!(b.capture_stage(ep) && b.is_capture(ep));

        // Castling: never a capture, never capture-stage.
        let b = board(KIWIPETE);
        let oo = Move::castling(Square::E1, Square::G1);
        assert!(!b.capture_stage(oo) && !b.is_capture(oo));

        // Push promotions: the queen belongs to the capture stage WITHOUT
        // capturing; push underpromotions to neither predicate.
        let b = board("4k3/6P1/8/8/8/8/8/4K3 w - - 0 1");
        let queen = Move::promotion(Square::G7, Square::G8, PieceType::Queen);
        let rook = Move::promotion(Square::G7, Square::G8, PieceType::Rook);
        assert!(b.capture_stage(queen) && !b.is_capture(queen));
        assert!(!b.capture_stage(rook) && !b.is_capture(rook));

        // Capturing underpromotion: both.
        let b = board("1n1nk3/2P5/8/8/8/8/8/4K3 w - - 0 1");
        let under = Move::promotion(Square::C7, Square::B8, PieceType::Knight);
        assert!(b.capture_stage(under) && b.is_capture(under));
    }

    /// Doc 05 §5.1 property 2: `capture_stage` answers `true` for exactly
    /// the CAPTURES emission and `false` for exactly the QUIETS one — the
    /// invariant the MovePicker's two-stage dedup silently depends on.
    #[test]
    fn capture_stage_is_consistent_with_generation() {
        for &fen in PAWN_FIXTURES {
            let b = board(fen);
            for &m in generated::<{ gen_type::CAPTURES }>(&b).iter() {
                assert!(b.capture_stage(m), "CAPTURES move {m:?} in {fen}");
            }
            for &m in generated::<{ gen_type::QUIETS }>(&b).iter() {
                assert!(!b.capture_stage(m), "QUIETS move {m:?} in {fen}");
            }
        }
    }

    /// Doc 05 §5.1 property 3: in check, generate_legal ⊆ EVASIONS, and
    /// the legal()-surviving EVASIONS are exactly generate_legal.
    #[test]
    fn legal_evasions_match_filtered_evasions() {
        for &fen in IN_CHECK_FIXTURES {
            let b = board(fen);
            let evasions = move_set(&generated::<{ gen_type::EVASIONS }>(&b));
            let legal = legal_set(&b);
            assert!(
                legal.is_subset(&evasions),
                "generate_legal ⊄ EVASIONS in {fen}"
            );
            let filtered: HashSet<Move> = evasions.into_iter().filter(|&m| b.legal(m)).collect();
            assert_eq!(
                filtered, legal,
                "filtered EVASIONS != generate_legal in {fen}"
            );
        }
    }

    /// `has_legal_moves` agrees with generate_legal emptiness everywhere
    /// and answers the two probe cases (mate, stalemate) it exists for.
    #[test]
    fn has_legal_moves_probe() {
        for &(fen, count) in PERFT1_COUNTS {
            assert_eq!(has_legal_moves(&board(fen)), count > 0, "{fen}");
        }
        for &fen in IN_CHECK_FIXTURES {
            let b = board(fen);
            assert_eq!(has_legal_moves(&b), !legal_set(&b).is_empty(), "{fen}");
        }
        // Fool's mate: in check, no legal moves.
        assert!(!has_legal_moves(&board(
            "rnb1kbnr/pppp1ppp/8/4p3/6Pq/5P2/PPPPP2P/RNBQKBNR w KQkq - 1 3"
        )));
        // Stalemate: not in check, still no legal moves.
        let b = board("7k/5Q2/6K1/8/8/8/8/8 b - - 0 1");
        assert!(b.checkers().is_empty());
        assert!(!has_legal_moves(&b));
    }

    /// generate_legal appends without clearing, like generate (§3.9's
    /// `start = list.len()` composition guard).
    #[test]
    fn generate_legal_appends_without_clearing() {
        let b = board(KIWIPETE);
        let mut list = MoveList::new();
        generate_legal(&b, &mut list);
        let first = list.len();
        generate_legal(&b, &mut list);
        assert_eq!(list.len(), first * 2);
        let (head, tail) = list.split_at(first);
        assert_eq!(
            head.iter().collect::<HashSet<_>>(),
            tail.iter().collect::<HashSet<_>>()
        );
    }

    // -----------------------------------------------------------------------
    // P3 step 7: Board::pseudo_legal + the 64k fuzz
    // (doc 05 §3.10, §4 step 7, §5.1.4)
    // -----------------------------------------------------------------------

    /// Doc 05 §5.1 property 4, the direct unit test of the TT-corruption
    /// defense: over ALL 65,536 raw u16 encodings, `pseudo_legal` must
    /// equal membership in the generated pseudo-legal set, and `legal`
    /// must not panic on any accepted move.
    fn assert_pseudo_legal_matches_generation(b: &Board, ctx: &str) {
        let expected: HashSet<u16> = pseudo_list(b).iter().map(|m| m.raw16()).collect();
        for raw in 0..=u16::MAX {
            let m = Move::from_raw16(raw);
            assert_eq!(
                b.pseudo_legal(m),
                expected.contains(&raw),
                "pseudo_legal disagrees with the generator on {m:?} (raw {raw:#06x}) in {ctx}"
            );
            if expected.contains(&raw) {
                let _ = b.legal(m); // §5.1.4: must never panic here
            }
        }
    }

    /// The fuzz corpus (doc 05 §8.4): the six perft positions plus
    /// in-check, promotion-, ep- and castling-rich positions covering
    /// every branch of the validator.
    const FUZZ_FIXTURES: &[&str] = &[
        crate::fen::START_FEN,
        KIWIPETE,
        "r3k2r/p1ppqpb1/bn2pnp1/3PN3/1p2P3/2N2Q1p/PPPBBPPP/R3K2R b KQkq - 0 1",
        "8/2p5/3p4/KP5r/1R3p1k/8/4P1P1/8 w - - 0 1",
        // Perft pos 4: the in-check containment path on a busy board.
        "r3k2r/Pppp1ppp/1b3nbN/nP6/BBP1P3/q4N2/Pp1P2PP/R2Q1RK1 w kq - 0 1",
        "rnbq1k1r/pp1Pbppp/2p5/8/2B5/8/PPP1NnPP/RNBQK2R w KQ - 1 8",
        "r4rk1/1pp1qppp/p1np1n2/2b1p1B1/2B1P1b1/P1NP1N2/1PP1QPPP/R4RK1 w - - 0 10",
        // The classic promotion position, both colors.
        "n1n5/PPPk4/8/8/8/8/4Kppp/5N1N w - - 0 1",
        "n1n5/PPPk4/8/8/8/8/4Kppp/5N1N b - - 0 1",
        // Live ep squares, one per color.
        "rnbqkbnr/ppp2ppp/4p3/3pP3/8/8/PPPP1PPP/RNBQKBNR w KQkq d6 0 3",
        "rnbqkbnr/pppp1ppp/8/8/4PpP1/8/PPPP3P/RNBQKBNR b KQkq g3 0 3",
        // Ep capture that is also an evasion (checker = double-pushed pawn).
        "8/8/8/2k5/3Pp3/8/8/4K3 b - d3 0 1",
        // Full castling rights with attacked transit squares.
        "r3k2r/8/3Q4/8/8/8/8/R3K2R b KQkq - 0 1",
        // Double check and a checkmate: king-only / empty evasion sets.
        "4k3/8/3N4/8/8/8/8/4RK2 b - - 0 1",
        "rnb1kbnr/pppp1ppp/8/4p3/6Pq/5P2/PPPPP2P/RNBQKBNR w KQkq - 1 3",
    ];

    #[test]
    fn pseudo_legal_matches_generation_64k_fuzz() {
        for &fen in FUZZ_FIXTURES {
            assert_pseudo_legal_matches_generation(&board(fen), fen);
        }
    }

    /// design/06 §3.5 differential oracle: a legal-move generator built ONLY
    /// from the per-move predicates (`pseudo_legal` + `legal`), scanned over
    /// the whole 64k encodable-move space at every node — wholly independent
    /// of the bulk `generate_legal` path. Recursed as perft; any divergence
    /// from [`crate::perft::perft`] means the generator and the predicate pair
    /// disagree *inside* the tree, not merely at a fuzz-corpus root (which the
    /// 64k fuzz above already covers).
    fn perft_via_predicate(b: &mut Board, depth: u32) -> u64 {
        if depth == 0 {
            return 1;
        }
        let legal: Vec<Move> = (0..=u16::MAX)
            .map(Move::from_raw16)
            .filter(|&m| b.pseudo_legal(m) && b.legal(m))
            .collect();
        if depth == 1 {
            return legal.len() as u64;
        }
        let mut nodes = 0;
        for m in legal {
            let gives_check = b.gives_check(m);
            b.make_move(m, gives_check);
            nodes += perft_via_predicate(b, depth - 1);
            b.unmake_move(m);
        }
        nodes
    }

    /// Fast tier: the predicate-only generator must reproduce `perft` at
    /// shallow depth across the fuzz corpus (depth 1 everywhere — the count
    /// now also exercises the `legal` filter, not just `pseudo_legal` — plus
    /// depth 2 from the start position for one ply of recursion).
    #[test]
    fn perft_predicate_oracle_matches_perft() {
        for &fen in FUZZ_FIXTURES {
            assert_eq!(
                perft_via_predicate(&mut board(fen), 1),
                crate::perft::perft(&mut board(fen), 1),
                "predicate-perft(1) != perft(1) for {fen}"
            );
        }
        assert_eq!(
            perft_via_predicate(&mut board(crate::fen::START_FEN), 2),
            crate::perft::perft(&mut board(crate::fen::START_FEN), 2),
        );
    }

    /// Release/nightly tier (heavier — a 64k scan at every interior node):
    /// recurse the differential oracle to depth 3 from the most branching
    /// standard perft positions.
    #[test]
    #[ignore = "release/nightly perft tier: 64k-per-node predicate scan"]
    fn perft_predicate_oracle_deep() {
        for (fen, depth) in [
            (crate::fen::START_FEN, 3),
            (KIWIPETE, 3),
            ("8/2p5/3p4/KP5r/1R3p1k/8/4P1P1/8 w - - 0 1", 3),
        ] {
            assert_eq!(
                perft_via_predicate(&mut board(fen), depth),
                crate::perft::perft(&mut board(fen), depth),
                "predicate-perft({depth}) != perft({depth}) for {fen}"
            );
        }
    }

    /// A few explicit TT-corruption shapes, named for documentation — the
    /// 64k fuzz already covers every one of them.
    #[test]
    fn pseudo_legal_rejects_classic_tt_corruption() {
        let b = board(crate::fen::START_FEN);
        let good = Move::normal(Square::E2, Square::E4);
        assert!(b.pseudo_legal(good));
        // Garbage promotion bits on an otherwise-valid normal move.
        assert!(!b.pseudo_legal(Move::from_raw16(good.raw16() | 0x1000)));
        // NULL, enemy mover, empty from-square, own-piece destination.
        assert!(!b.pseudo_legal(Move::NULL));
        assert!(!b.pseudo_legal(Move::normal(Square::E7, Square::E5)));
        assert!(!b.pseudo_legal(Move::normal(Square::E4, Square::E5)));
        assert!(!b.pseudo_legal(Move::normal(Square::D1, Square::E2)));
        // Stale moves from other positions: blocked castle, blocked slide.
        assert!(!b.pseudo_legal(Move::castling(Square::E1, Square::G1)));
        assert!(!b.pseudo_legal(Move::normal(Square::A1, Square::A3)));

        // In check (perft pos 4, Bb6xg1 diagonal): a well-formed normal
        // move that does not evade is rejected by step 6; a block on the
        // checking ray is accepted.
        let b = board("r3k2r/Pppp1ppp/1b3nbN/nP6/BBP1P3/q4N2/Pp1P2PP/R2Q1RK1 w kq - 0 1");
        assert!(!b.checkers().is_empty());
        assert!(!b.pseudo_legal(Move::normal(Square::G2, Square::G3)));
        assert!(b.pseudo_legal(Move::normal(Square::F3, Square::D4)));
    }
}
