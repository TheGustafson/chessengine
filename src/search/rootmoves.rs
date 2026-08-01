//! Root-move bookkeeping: per-root-move scores, EMAs, effort and selDepth,
//! and the stable descending sort the iterative-deepening loop relies on
//! (design/07 §3.4).
//!
//! The root is cold — searched once per iteration per MultiPV line — so
//! `Vec` PVs are fine here; the no-allocation rule applies to the tree,
//! not the root (design/07 §6).

use crate::board::Board;
use crate::movegen::generate_legal;
use crate::moves::{Move, MoveList};
use crate::types::{VALUE_INFINITE, Value, is_loss};

/// "Unsearched this iteration" sentinel for [`RootMove::score`],
/// `average_score`, and friends.
const SCORE_UNSET: Value = -VALUE_INFINITE;

/// Init sentinel for [`RootMove::mean_squared_score`]: `-VALUE_INFINITE²`.
///
/// `i64`, deviating from Stockfish's `int` deliberately (design/07 §3.4,
/// Open Question 3): the EMA's intermediate `value·|value| + prev` reaches
/// ~2.05e9 — inside `i32` but within 5% of overflow; we take the free
/// headroom at this cold, root-only site.
const MSS_UNSET: i64 = -(VALUE_INFINITE as i64 * VALUE_INFINITE as i64);

/// One root move's search state, persisted across iterations
/// (design/07 §3.4). Field updates inside the root node's move loop are
/// specified there; the ID loop snapshots `previous_*` per iteration.
pub struct RootMove {
    /// `pv[0]` is the root move itself; extended from `pv_table[1]` when
    /// the move's score is set.
    pub pv: Vec<Move>,
    /// `-VALUE_INFINITE` when unsearched this iteration.
    pub score: Value,
    /// Last completed iteration's score.
    pub previous_score: Value,
    /// What we print (clamped to the aspiration window on bounds).
    pub uci_score: Value,
    /// EMA of iteration scores; feeds aspiration centering (§3.6).
    pub average_score: Value,
    /// EMA of `value·|value|`; feeds the aspiration delta (§3.6).
    pub mean_squared_score: i64,
    pub score_lowerbound: bool,
    pub score_upperbound: bool,
    pub sel_depth: i32,
    /// Nodes spent under this root move (a time-management input).
    pub effort: u64,
    // tb_rank / tb_score: added with Syzygy support later.
}

pub type RootMoves = Vec<RootMove>;

impl RootMove {
    pub fn new(m: Move) -> Self {
        RootMove {
            pv: vec![m],
            score: SCORE_UNSET,
            previous_score: SCORE_UNSET,
            uci_score: SCORE_UNSET,
            average_score: SCORE_UNSET,
            mean_squared_score: MSS_UNSET,
            score_lowerbound: false,
            score_upperbound: false,
            sel_depth: 0,
            effort: 0,
        }
    }

    pub fn score_is_bound(&self) -> bool {
        self.score_lowerbound || self.score_upperbound
    }

    /// An *exact* mated/TB-loss score for this move — the aborted-loss
    /// hygiene trigger (design/07 §3.5): such a score from a stopped
    /// iteration is untrusted.
    pub fn score_is_exact_loss(&self) -> bool {
        self.score != SCORE_UNSET && is_loss(self.score) && !self.score_is_bound()
    }

    /// Descending sort key: `(score, previous_score)`. Moves unsearched
    /// this iteration tie on `score` and fall back to last iteration's
    /// score, then (via stable sort) to last iteration's order.
    pub fn sort_key(&self) -> (Value, Value) {
        (self.score, self.previous_score)
    }

    /// EMA update after a root child search returns `value`
    /// (design/07 §3.4): first real value seeds the average, thereafter
    /// `avg = (value + avg) / 2`.
    pub fn update_average_score(&mut self, value: Value) {
        self.average_score = if self.average_score == SCORE_UNSET {
            value
        } else {
            (value + self.average_score) / 2
        };
    }

    /// EMA of the sign-preserving square `value·|value|` (as `i64`),
    /// same first-value seeding as [`Self::update_average_score`]. Feeds
    /// the aspiration half-width `delta = 5 + |mss| / 10588` (§3.6).
    pub fn update_mean_squared_score(&mut self, value: Value) {
        let sq = value as i64 * (value as i64).abs();
        self.mean_squared_score = if self.mean_squared_score == MSS_UNSET {
            sq
        } else {
            (sq + self.mean_squared_score) / 2
        };
    }
}

/// Build the root move list for a `go` (design/07 §3.4, design/12 §3.6):
/// every legal move in `board`, filtered by the `searchmoves` whitelist when
/// it is non-empty. Initial order is generation order; iterative deepening's
/// stable sort takes over from iteration 2 on.
///
/// A filter that would leave *zero* root moves in a position that has legal
/// moves is ignored entirely — bad GUI input must never turn a playable
/// position into `bestmove (none)` (design/12 §3.6). The UCI layer detects
/// that case (non-empty `searchmoves`, result spanning all legal moves) to
/// print its `info string` warning; an empty return therefore always means
/// "no legal moves at all" (mate or stalemate).
pub fn build_root_moves(board: &Board, searchmoves: &[Move]) -> RootMoves {
    let mut legal = MoveList::new();
    generate_legal(board, &mut legal);
    let filtered: RootMoves = legal
        .iter()
        .filter(|m| searchmoves.is_empty() || searchmoves.contains(m))
        .map(|&m| RootMove::new(m))
        .collect();
    if filtered.is_empty() && !legal.is_empty() {
        return legal.iter().map(|&m| RootMove::new(m)).collect();
    }
    filtered
}

/// Stable descending sort by [`RootMove::sort_key`].
///
/// Stability is load-bearing (design/07 §3.4): after an aspiration attempt
/// only the searched PV move carries a fresh score — everything else sits
/// at `-VALUE_INFINITE` — so a stable sort floats the new best to the front
/// while preserving last iteration's order for the rest, and that order IS
/// the next iteration's move ordering. An unstable sort here would be a
/// silent Elo-losing bug (§6), hence this named seam instead of ad-hoc
/// sort calls. `slice::sort_by_key` is guaranteed stable; descending comes
/// from the [`std::cmp::Reverse`] key.
pub fn sort_descending(root_moves: &mut [RootMove]) {
    root_moves.sort_by_key(|rm| std::cmp::Reverse(rm.sort_key()));
}

#[cfg(test)]
mod tests {
    use super::*;
    use crate::types::{Square, mated_in};

    fn rm(from: u8, to: u8) -> RootMove {
        RootMove::new(Move::normal(Square::new(from), Square::new(to)))
    }

    fn front_move(r: &RootMove) -> Move {
        r.pv[0]
    }

    #[test]
    fn new_starts_at_sentinels() {
        let m = Move::normal(Square::E2, Square::E4);
        let r = RootMove::new(m);
        assert!(r.pv.len() == 1 && r.pv[0] == m);
        assert_eq!(r.score, -VALUE_INFINITE);
        assert_eq!(r.previous_score, -VALUE_INFINITE);
        assert_eq!(r.uci_score, -VALUE_INFINITE);
        assert_eq!(r.average_score, -VALUE_INFINITE);
        assert_eq!(
            r.mean_squared_score,
            -(VALUE_INFINITE as i64 * VALUE_INFINITE as i64)
        );
        assert!(!r.score_is_bound());
        assert_eq!(r.sel_depth, 0);
        assert_eq!(r.effort, 0);
    }

    // --- EMA formulas (design/07 §5: init/update match §3.4) ---

    #[test]
    fn average_score_seeds_then_halves() {
        let mut r = rm(0, 1);
        r.update_average_score(100);
        assert_eq!(r.average_score, 100); // first value seeds, no blend with -INF
        r.update_average_score(50);
        assert_eq!(r.average_score, 75); // (50 + 100) / 2
        r.update_average_score(-75);
        assert_eq!(r.average_score, 0); // (-75 + 75) / 2
    }

    #[test]
    fn mean_squared_score_seeds_then_halves_sign_preserving() {
        let mut r = rm(0, 1);
        r.update_mean_squared_score(100);
        assert_eq!(r.mean_squared_score, 10_000); // 100 * |100|
        r.update_mean_squared_score(-50);
        assert_eq!(r.mean_squared_score, 3_750); // (-50*50 + 10000) / 2
        // The i64 headroom case that motivated the widening: a mate-band
        // value squared overflows i32's positive range comfortably.
        let mut r = rm(0, 1);
        r.update_mean_squared_score(VALUE_INFINITE - 1);
        assert_eq!(
            r.mean_squared_score,
            (VALUE_INFINITE as i64 - 1) * (VALUE_INFINITE as i64 - 1)
        );
    }

    // --- Bound / loss predicates ---

    #[test]
    fn score_is_exact_loss_requires_searched_loss_without_bounds() {
        let mut r = rm(0, 1);
        assert!(!r.score_is_exact_loss()); // unsearched

        r.score = mated_in(5);
        assert!(r.score_is_exact_loss()); // exact mated score

        r.score_lowerbound = true;
        assert!(!r.score_is_exact_loss()); // bound scores are not exact

        r.score_lowerbound = false;
        r.score = -100;
        assert!(!r.score_is_exact_loss()); // an ordinary bad score is no loss
    }

    // --- Stable descending sort (design/07 §5: semantics matter) ---

    #[test]
    fn sort_orders_by_score_then_previous_score() {
        let mut moves = vec![rm(0, 1), rm(2, 3), rm(4, 5)];
        moves[0].score = 10;
        moves[1].score = 30;
        moves[2].score = 30;
        moves[2].previous_score = 99; // breaks the 30-30 tie
        sort_descending(&mut moves);
        assert!(front_move(&moves[0]) == Move::normal(Square::E1, Square::F1));
        assert!(front_move(&moves[1]) == Move::normal(Square::C1, Square::D1));
        assert!(front_move(&moves[2]) == Move::normal(Square::A1, Square::B1));
    }

    // --- build_root_moves (design/07 §3.4, design/12 §3.6) ---

    #[test]
    fn build_root_moves_lists_every_legal_move_without_filter() {
        let board = Board::from_fen(crate::fen::START_FEN).unwrap();
        let rms = build_root_moves(&board, &[]);
        assert_eq!(rms.len(), 20);
        // Every entry is fresh: pv = [move], scores at sentinels.
        assert!(rms.iter().all(|rm| rm.pv.len() == 1));
        assert!(rms.iter().all(|rm| rm.score == -VALUE_INFINITE));
    }

    #[test]
    fn build_root_moves_applies_the_searchmoves_whitelist() {
        let board = Board::from_fen(crate::fen::START_FEN).unwrap();
        let filter = [
            Move::normal(Square::E2, Square::E4),
            Move::normal(Square::G1, Square::F3),
        ];
        let rms = build_root_moves(&board, &filter);
        assert_eq!(rms.len(), 2);
        assert!(rms.iter().all(|rm| filter.contains(&rm.pv[0])));
    }

    #[test]
    fn build_root_moves_ignores_a_filter_with_no_legal_moves() {
        // e2e5 and a move for the wrong side are both illegal here: the
        // whole filter is ignored, never yielding an empty list in a
        // playable position (design/12 §3.6).
        let board = Board::from_fen(crate::fen::START_FEN).unwrap();
        let filter = [
            Move::normal(Square::E2, Square::E5),
            Move::normal(Square::E7, Square::E5),
        ];
        assert_eq!(build_root_moves(&board, &filter).len(), 20);
    }

    #[test]
    fn build_root_moves_is_empty_only_without_legal_moves() {
        // Fool's mate: white is checkmated; even a non-empty filter yields
        // an empty list (the caller emits `bestmove (none)`).
        let fen = "rnb1kbnr/pppp1ppp/8/4p3/6Pq/5P2/PPPPP2P/RNBQKBNR w KQkq - 1 3";
        let board = Board::from_fen(fen).unwrap();
        assert!(build_root_moves(&board, &[]).is_empty());
        let filter = [Move::normal(Square::E2, Square::E4)];
        assert!(build_root_moves(&board, &filter).is_empty());
    }

    #[test]
    fn stable_sort_preserves_prior_order_of_unsearched_moves() {
        // Aspiration-iteration shape: four moves in last iteration's order,
        // all reset to (-INF score, equal previous_score) except one that
        // was just searched. The searched move must float to the front and
        // the tied rest must keep their exact prior order — that order is
        // the next iteration's move ordering (design/07 §3.4).
        let mut moves = vec![rm(0, 1), rm(2, 3), rm(4, 5), rm(6, 7)];
        for m in &mut moves {
            m.previous_score = 42; // fully tied sort keys
        }
        moves[2].score = 55; // the freshly searched PV move

        sort_descending(&mut moves);

        assert!(front_move(&moves[0]) == Move::normal(Square::E1, Square::F1));
        // The other three keep prior relative order despite equal keys.
        assert!(front_move(&moves[1]) == Move::normal(Square::A1, Square::B1));
        assert!(front_move(&moves[2]) == Move::normal(Square::C1, Square::D1));
        assert!(front_move(&moves[3]) == Move::normal(Square::G1, Square::H1));
    }
}
