//! The §3.4.4 correctness linchpin: the incrementally-maintained NNUE
//! accumulator must equal a from-scratch refresh at **every** computed node.
//!
//! This is the NNUE analogue of the `make_unmake_walk` key-invariance stress.
//! Two complementary drivers over the `tests/data/perft.epd` roots, both
//! asserting `incremental == refresh` (via
//! [`NnueContext::assert_matches_refresh`]) at each node with a deterministic
//! pseudo-random test net:
//!
//! 1. **Sibling-reuse DFS** — a bounded depth-first walk that, at each node,
//!    tries several *distinct* moves and recurses. This is what exercises the
//!    invalidation rule: after unmaking move A at ply `k` and making a
//!    different move B, `stack[k+1]` must have been re-marked uncomputed, or B
//!    would silently reuse A's stale accumulator and the assert would fire.
//! 2. **Deep linear chains** — long random walks so the lazy walk-back folds
//!    many dirty records forward from a distant computed ancestor (promotions,
//!    captures, castling, and late-game material along the way).
//!
//! Deterministic seed: any failure reproduces exactly; no RNG dependency (D15).

mod common;

use chessengine::eval::nnue::network::Activation;
use chessengine::eval::nnue::{DirtyPiece, Network, NnueContext, evaluate_forward};
use chessengine::movegen::generate_legal;
use chessengine::{Board, MoveList};

/// Deterministic SplitMix64 (same construction as `tests/make_unmake_walk.rs`).
struct SplitMix64(u64);

impl SplitMix64 {
    fn next_u64(&mut self) -> u64 {
        self.0 = self.0.wrapping_add(0x9E37_79B9_7F4A_7C15);
        let mut z = self.0;
        z = (z ^ (z >> 30)).wrapping_mul(0xBF58_476D_1CE4_E5B9);
        z = (z ^ (z >> 27)).wrapping_mul(0x94D0_49BB_1331_11EB);
        z ^ (z >> 31)
    }
    fn below(&mut self, bound: usize) -> usize {
        (self.next_u64() % bound as u64) as usize
    }
}

/// Sibling-reuse DFS (see module docs): at each node try `branch` distinct
/// moves and recurse to `depth`. Asserts `incremental == refresh` at every
/// made node.
#[allow(clippy::too_many_arguments)]
fn explore(
    board: &mut Board,
    ctx: &mut NnueContext,
    ply: usize,
    depth: usize,
    branch: usize,
    rng: &mut SplitMix64,
    nodes: &mut u64,
) {
    if depth == 0 {
        return;
    }
    let mut list = MoveList::new();
    generate_legal(board, &mut list);
    if list.is_empty() {
        return;
    }
    // Distinct siblings (offset-rotated so the tried set varies per node).
    let count = list.len().min(branch);
    let start = rng.below(list.len());
    for j in 0..count {
        let m = list[(start + j) % list.len()];
        let dp = DirtyPiece::from_move(board, m);
        let gives_check = board.gives_check(m);
        board.make_move(m, gives_check);
        ctx.push(ply, dp);
        // The oracle is compiled out of release builds (like the board's
        // `assert_state_keys_consistent`); the real gate is the debug
        // `cargo test` run. Force computation regardless so release still
        // exercises the lazy walk.
        #[cfg(debug_assertions)]
        ctx.assert_matches_refresh(board, ply + 1);
        let _ = ctx.evaluate(board, ply + 1);
        *nodes += 1;
        explore(board, ctx, ply + 1, depth - 1, branch, rng, nodes);
        board.unmake_move(m);
    }
}

#[test]
fn incremental_accumulator_equals_refresh_over_random_games() {
    let records = common::load_perft_epd().unwrap_or_else(|e| panic!("{e}"));
    assert!(!records.is_empty(), "perft.epd contains no cases");

    // Alternate the two activations across roots so BOTH the SCReLU and CReLU
    // forward paths run inside the incremental walk. The accumulator invariant
    // itself is activation-independent (refresh/add_sub never read Activation),
    // but alternating also drives evaluate()'s CReLU branch under lazy
    // fold-forward — at no extra runtime, since each root uses one net not both.
    let screlu = Network::deterministic_test_net(0xACC0_1ADE, Activation::SCReLU);
    let crelu = Network::deterministic_test_net(0xACC0_1ADF, Activation::CReLU);
    let mut rng = SplitMix64(0x1234_5678_9ABC_DEF0);
    let mut nodes = 0u64;

    for (idx, record) in records.iter().enumerate() {
        let net = if idx % 2 == 0 { &screlu } else { &crelu };
        let mut board = common::board(&record.fen);
        let mut ctx = NnueContext::new(net);
        ctx.refresh_root(&board);

        // Phase 1: sibling-reuse DFS (invalidation stress).
        explore(&mut board, &mut ctx, 0, 4, 4, &mut rng, &mut nodes);

        // Phase 2: a deep linear chain from the same root (long walk-backs).
        let mut ctx = NnueContext::new(net);
        ctx.refresh_root(&board);
        let mut trail = Vec::new();
        for ply in 0..80usize {
            let mut list = MoveList::new();
            generate_legal(&board, &mut list);
            if list.is_empty() {
                break;
            }
            let m = list[rng.below(list.len())];
            let dp = DirtyPiece::from_move(&board, m);
            let gives_check = board.gives_check(m);
            board.make_move(m, gives_check);
            ctx.push(ply, dp);
            // Evaluate at every node so the lazy walk actually folds forward
            // (release); assert it equals a from-scratch refresh (debug).
            #[cfg(debug_assertions)]
            ctx.assert_matches_refresh(&board, ply + 1);
            let _ = ctx.evaluate(&board, ply + 1);
            trail.push(m);
            nodes += 1;
        }
        while let Some(m) = trail.pop() {
            board.unmake_move(m);
        }
    }

    assert!(
        nodes > 5_000,
        "expected a substantial walk, got {nodes} nodes"
    );
}

/// Color-flip a canonical 6-field FEN (mirror across the horizontal axis, swap
/// piece colors, side to move, castling, and the ep rank). Same construction
/// as `tests/eval_symmetry.rs::flip_fen`.
fn flip_fen(fen: &str) -> String {
    let fields: Vec<&str> = fen.split_whitespace().collect();
    assert_eq!(fields.len(), 6, "expected a 6-field FEN: {fen:?}");
    let swap = |c: char| {
        if c.is_ascii_uppercase() {
            c.to_ascii_lowercase()
        } else {
            c.to_ascii_uppercase()
        }
    };
    let placement: String = fields[0]
        .split('/')
        .rev()
        .map(|rank| rank.chars().map(swap).collect::<String>())
        .collect::<Vec<_>>()
        .join("/");
    let stm = if fields[1] == "w" { "b" } else { "w" };
    let castling = if fields[2] == "-" {
        "-".to_string()
    } else {
        let swapped: String = fields[2].chars().map(swap).collect();
        ['K', 'Q', 'k', 'q']
            .into_iter()
            .filter(|c| swapped.contains(*c))
            .collect()
    };
    let ep = if fields[3] == "-" {
        "-".to_string()
    } else {
        let b = fields[3].as_bytes();
        format!("{}{}", b[0] as char, (b'1' + b'8' - b[1]) as char)
    };
    format!(
        "{placement} {stm} {castling} {ep} {} {}",
        fields[4], fields[5]
    )
}

/// The §3.4.2 symmetry guarantee over the whole FEN corpus: the color-flip of
/// a position evaluates EQUAL (both are stm-relative). Guards the feature
/// indexer + perspective construction (G4).
#[test]
fn forward_pass_is_color_flip_symmetric_over_corpus() {
    let net = Network::deterministic_test_net(0x0005_EED5, Activation::SCReLU);
    let crelu = Network::deterministic_test_net(0x0005_EED6, Activation::CReLU);
    let mut checked = 0;
    for fen in common::load_fens() {
        // Corpus is 6-field FENs; skip anything else defensively.
        if fen.split_whitespace().count() != 6 {
            continue;
        }
        let flipped = flip_fen(&fen);
        let a = common::board(&fen);
        let b = common::board(&flipped);
        for n in [&net, &crelu] {
            assert_eq!(
                evaluate_forward(n, &a),
                evaluate_forward(n, &b),
                "color-flip symmetry broken ({:?}):\n  {fen}\n  {flipped}",
                n.activation
            );
        }
        checked += 1;
    }
    assert!(checked >= 50, "expected a broad corpus, checked {checked}");
}
