//! Eval color-flip symmetry suite and Phase A anchors
//! (design/11 §3.2.6, §4 step 4, §5).
//!
//! The workhorse invariant: `evaluate(pos) == evaluate(flipped(pos))` —
//! EQUAL, not negated, because both are side-to-move relative and flipping
//! swaps the side to move along with everything else. It is the single best
//! tripwire for table-flip, perspective, and feature-index bugs, and it
//! survives all three eval phases.
//!
//! Two symmetry layers:
//! 1. every position of the FEN corpus, flipped through a FEN-text flip
//!    helper (which also exercises `to_fen`/`from_fen`);
//! 2. a >= 10k-position harvest from depth-2 perft walks of the corpus,
//!    where the original and flipped boards advance in lockstep through
//!    mirrored moves (no FEN round-trip per node, so the walk stays cheap).

mod common;

use chessengine::eval::{EvalContext, evaluate, is_material_draw};
use chessengine::movegen::generate_legal;
use chessengine::types::{VALUE_DRAW, VALUE_TB_LOSS_IN_MAX_PLY, VALUE_TB_WIN_IN_MAX_PLY, Value};
use chessengine::{Board, Move, MoveKind, MoveList, START_FEN};

fn eval_opt(board: &Board, optimism: Value) -> Value {
    // Phase A ignores the context; a fresh one per call keeps this helper
    // honest about `evaluate` not depending on hidden state.
    evaluate(board, &mut EvalContext::new(), optimism)
}

fn eval0(board: &Board) -> Value {
    eval_opt(board, 0)
}

// ---------------------------------------------------------------------------
// Board flip (test-side helper, design/11 §3.2.6)
// ---------------------------------------------------------------------------

/// Color-flip a canonical 6-field FEN: mirror all pieces across the
/// horizontal axis, swap colors, swap the side to move, mirror castling
/// rights (K<->k, Q<->q), mirror the ep square, keep both clocks.
fn flip_fen(fen: &str) -> String {
    let fields: Vec<&str> = fen.split_whitespace().collect();
    assert_eq!(fields.len(), 6, "expected a canonical 6-field FEN: {fen:?}");

    let swap = |c: char| {
        if c.is_ascii_uppercase() {
            c.to_ascii_lowercase()
        } else {
            c.to_ascii_uppercase()
        }
    };

    // Board: reverse the rank rows (rank 8 first in FEN), swap piece case.
    let placement: Vec<String> = fields[0]
        .split('/')
        .rev()
        .map(|rank| rank.chars().map(swap).collect())
        .collect();
    let placement = placement.join("/");

    let stm = match fields[1] {
        "w" => "b",
        "b" => "w",
        other => panic!("bad side-to-move field {other:?} in {fen:?}"),
    };

    // Castling: swap case, then re-normalize to canonical KQkq order.
    let castling = if fields[2] == "-" {
        "-".to_string()
    } else {
        let swapped: String = fields[2].chars().map(swap).collect();
        ['K', 'Q', 'k', 'q']
            .into_iter()
            .filter(|c| swapped.contains(*c))
            .collect()
    };

    // En passant: mirror the rank digit ('3' <-> '6').
    let ep = if fields[3] == "-" {
        "-".to_string()
    } else {
        let bytes = fields[3].as_bytes();
        assert_eq!(bytes.len(), 2, "bad ep field in {fen:?}");
        format!("{}{}", bytes[0] as char, (b'1' + b'8' - bytes[1]) as char)
    };

    format!(
        "{placement} {stm} {castling} {ep} {} {}",
        fields[4], fields[5]
    )
}

/// The flipped counterpart of `board`, built through the FEN text flip.
fn flipped(board: &Board) -> Board {
    common::board(&flip_fen(&board.to_fen()))
}

/// Mirror a move for the flipped board: ranks flip, kind and promotion
/// piece carry over.
fn flip_move(m: Move) -> Move {
    let from = m.from().flip_rank();
    let to = m.to().flip_rank();
    match m.kind() {
        MoveKind::Normal => Move::normal(from, to),
        MoveKind::Promotion => Move::promotion(
            from,
            to,
            m.promotion_piece().expect("promotion carries a piece"),
        ),
        MoveKind::EnPassant => Move::en_passant(from, to),
        MoveKind::Castling => Move::castling(from, to),
    }
}

// ---------------------------------------------------------------------------
// Fixture FENs (dead draws and near misses are part of the symmetry suite,
// design/11 §5)
// ---------------------------------------------------------------------------

const DEAD_DRAW_FENS: &[&str] = &[
    "k7/8/8/8/8/8/8/K7 w - - 0 1",   // KvK
    "k7/8/8/8/8/8/8/KN6 w - - 0 1",  // KNvK, white strong
    "kn6/8/8/8/8/8/8/K7 w - - 0 1",  // KNvK, black strong
    "k7/8/8/8/8/8/8/KB6 w - - 0 1",  // KBvK, white strong
    "kb6/8/8/8/8/8/8/K7 w - - 0 1",  // KBvK, black strong
    "k7/8/8/8/8/8/8/KNN5 w - - 0 1", // KNNvK, white strong
    "knn5/8/8/8/8/8/8/K7 w - - 0 1", // KNNvK, black strong
];

const NEAR_MISS_FENS: &[&str] = &[
    "k7/8/8/8/8/8/8/KR6 w - - 0 1",  // KRvK
    "k7/8/8/8/8/8/P7/K7 w - - 0 1",  // KPvK
    "kn6/8/8/8/8/8/8/KB6 w - - 0 1", // KBvKN
];

fn full_corpus() -> Vec<String> {
    let mut fens = common::load_fens();
    fens.extend(DEAD_DRAW_FENS.iter().map(|f| f.to_string()));
    fens.extend(NEAR_MISS_FENS.iter().map(|f| f.to_string()));
    fens
}

// ---------------------------------------------------------------------------
// The flip helper is itself under test
// ---------------------------------------------------------------------------

#[test]
fn flip_fen_startpos_and_involution() {
    // The start position's placement is its own mirror image; only the
    // side to move changes.
    assert_eq!(
        flip_fen(START_FEN),
        "rnbqkbnr/pppppppp/8/8/8/8/PPPPPPPP/RNBQKBNR b KQkq - 0 1"
    );

    // A hand-checked asymmetric example: pieces mirror ranks and swap
    // color, castling swaps sides, the ep rank mirrors, clocks stay.
    assert_eq!(
        flip_fen("rnbqkbnr/ppp2ppp/4p3/3pP3/8/8/PPPP1PPP/RNBQKBNR w KQkq d6 0 3"),
        "rnbqkbnr/pppp1ppp/8/8/3Pp3/4P3/PPP2PPP/RNBQKBNR b KQkq d3 0 3"
    );

    // Flipping is an involution on every canonical corpus FEN (the corpus
    // round-trips byte-identically through from_fen -> to_fen, so text
    // equality is well-defined).
    for fen in full_corpus() {
        assert_eq!(flip_fen(&flip_fen(&fen)), fen, "involution broke on {fen}");
        // The flipped FEN must also be a legal, parseable position.
        common::board(&flip_fen(&fen));
    }
}

// ---------------------------------------------------------------------------
// Symmetry layer 1: the full corpus, through the FEN flip
// ---------------------------------------------------------------------------

#[test]
fn corpus_symmetry() {
    for fen in full_corpus() {
        let a = common::board(&fen);
        let b = flipped(&a);
        // Flipping preserves check status; evaluate is only defined
        // outside check (design/11 §3.1).
        assert_eq!(
            a.checkers().is_empty(),
            b.checkers().is_empty(),
            "flip changed check status: {fen}"
        );
        if !a.checkers().is_empty() {
            continue;
        }
        // Equal (not negated) for any shared optimism: both values are
        // side-to-move relative.
        for optimism in [0, 33, -77] {
            assert_eq!(
                eval_opt(&a, optimism),
                eval_opt(&b, optimism),
                "asymmetric eval (optimism {optimism}) for {fen}"
            );
        }
    }
}

// ---------------------------------------------------------------------------
// Symmetry layer 2: >= 10k positions harvested from perft walks
// ---------------------------------------------------------------------------

/// Advance `a` and its flipped twin `b` in lockstep through every legal
/// move of `a` to `depth`, checking eval symmetry (and the non-TB range
/// invariant, design/11 §5) at every non-check node.
fn walk_symmetry(a: &mut Board, b: &mut Board, depth: u32, checked: &mut u64) {
    if a.checkers().is_empty() {
        assert!(
            b.checkers().is_empty(),
            "flip changed check status at {}",
            a.to_fen()
        );
        let (va, vb) = (eval0(a), eval0(b));
        assert_eq!(va, vb, "asymmetric eval at {}", a.to_fen());
        assert!(
            VALUE_TB_LOSS_IN_MAX_PLY < va && va < VALUE_TB_WIN_IN_MAX_PLY,
            "eval escaped the non-TB band at {}",
            a.to_fen()
        );
        *checked += 1;
    } else {
        assert!(
            !b.checkers().is_empty(),
            "flip lost a check at {}",
            a.to_fen()
        );
    }
    if depth == 0 {
        return;
    }

    let mut moves = MoveList::new();
    generate_legal(a, &mut moves);
    let mut mirrored = MoveList::new();
    generate_legal(b, &mut mirrored);
    // The flipped position has exactly the mirrored move set.
    assert_eq!(
        moves.len(),
        mirrored.len(),
        "legal-move count differs from flipped twin at {}",
        a.to_fen()
    );

    for &m in moves.iter() {
        let fm = flip_move(m);
        assert!(
            b.pseudo_legal(fm) && b.legal(fm),
            "mirrored move {fm} illegal on flipped board at {}",
            a.to_fen()
        );
        a.make_move(m, a.gives_check(m));
        b.make_move(fm, b.gives_check(fm));
        walk_symmetry(a, b, depth - 1, checked);
        b.unmake_move(fm);
        a.unmake_move(m);
    }
}

#[test]
fn perft_walk_symmetry_over_10k_positions() {
    let mut checked = 0;
    for fen in full_corpus() {
        let mut a = common::board(&fen);
        let mut b = flipped(&a);
        walk_symmetry(&mut a, &mut b, 2, &mut checked);
    }
    assert!(
        checked >= 10_000,
        "symmetry corpus too small: only {checked} positions harvested (need >= 10k, design/11 §3.2.6)"
    );
}

// ---------------------------------------------------------------------------
// Phase A anchors (design/11 §5)
// ---------------------------------------------------------------------------

#[test]
fn startpos_evaluates_to_zero() {
    // Color-symmetric position, no tempo term (design/11 §3.2.6).
    let startpos = common::board(START_FEN);
    assert_eq!(eval0(&startpos), 0);
    assert_eq!(eval0(&flipped(&startpos)), 0);
}

#[test]
fn material_anchors() {
    // A queen up is decisively positive for the side to move ...
    let kqvk_w = common::board("4k3/8/8/8/8/8/8/Q3K3 w - - 0 1");
    assert!(eval0(&kqvk_w) > 500, "KQvK, strong side to move");
    // ... and decisively negative when the bare king is to move.
    let kqvk_b = common::board("4k3/8/8/8/8/8/8/Q3K3 b - - 0 1");
    assert!(eval0(&kqvk_b) < -500, "KQvK, weak side to move");
    // Same placement, opposite side to move: exact negation (the whole
    // pipeline is odd under stm flip at fixed placement).
    assert_eq!(eval0(&kqvk_w), -eval0(&kqvk_b));

    // A rook up clears the guard and scores like a rook.
    let krvk = common::board("k7/8/8/8/8/8/8/KR6 w - - 0 1");
    assert!(eval0(&krvk) > 300, "KRvK");

    // One extra pawn in a king endgame: psqt_eval pins 87 exactly
    // (hand-computed, psqt.rs); the outer shape's material blend keeps it
    // in pawn-ish range.
    let kpvk = common::board("4k3/8/8/8/4P3/8/8/4K3 w - - 0 1");
    assert!(
        (60..=120).contains(&eval0(&kpvk)),
        "single-pawn eval out of range: {}",
        eval0(&kpvk)
    );
}

#[test]
fn dead_draw_guard_over_fixtures() {
    for fen in DEAD_DRAW_FENS {
        let b = common::board(fen);
        assert!(is_material_draw(&b), "guard must fire: {fen}");
        assert_eq!(eval0(&b), VALUE_DRAW, "dead draw must eval 0: {fen}");
        // The guard is symmetric: the flipped twin is also a dead draw.
        assert!(is_material_draw(&flipped(&b)), "guard asymmetric: {fen}");
    }
    for fen in NEAR_MISS_FENS {
        let b = common::board(fen);
        assert!(!is_material_draw(&b), "guard overfired: {fen}");
        assert!(!is_material_draw(&flipped(&b)), "guard asymmetric: {fen}");
    }
}
