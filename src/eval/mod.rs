//! Evaluation: the single `evaluate` seam (D22) and its Phase A body,
//! material + tapered PSQT (design/11 §3.1, §3.2, §3.6; roadmap D21).
//!
//! The signature is pinned across all three eval phases so PSQT → HCE →
//! NNUE swap without touching search. Contract highlights (§3.1):
//!
//! - **Perspective**: the returned [`Value`] is from the side to move
//!   (negamax convention).
//! - **Precondition**: never called in check — search skips the static
//!   eval at in-check nodes.
//! - **Range**: always strictly inside the non-TB band; never a mate or
//!   TB value.
//! - **Purity boundary**: reads `board`, mutates only `ctx`. Correction
//!   history is applied in *search*, never here; the raw return value is
//!   what the TT's `eval16` stores (D22).

pub mod nnue;
pub mod params;
pub mod psqt;
pub mod score;

use crate::board::{Board, DirtyPiece};
use crate::types::{
    BISHOP_VALUE, Color, KNIGHT_VALUE, PieceType, VALUE_DRAW, VALUE_TB_LOSS_IN_MAX_PLY,
    VALUE_TB_WIN_IN_MAX_PLY, Value,
};
use params::{EVAL_BLEND_BASE, MATERIAL_PAWN_WEIGHT, OPTIMISM_BLEND_BASE, RULE50_DAMPING_DEN};

/// Per-thread mutable evaluation state, owned by the search `Worker` (D23).
///
/// Phase A holds nothing — but the parameter is part of the pinned
/// three-argument `evaluate` signature (design/11 §3.1): Phase B's
/// pawn-structure hash and Phase C's NNUE accumulator stack land in here
/// without the seam ever changing. Per-thread mutable eval state cannot
/// hide behind `&Board` without interior mutability or `thread_local!`,
/// both banned on the search path (design/13 §3.12).
#[derive(Debug, Default)]
pub struct EvalContext {
    /// The per-thread NNUE accumulator stack over the active net (design/11
    /// §3.4.4), or `None` when NNUE is off (⇒ PSQT). Installed from the
    /// process-wide active net at each search's eval boundary
    /// ([`EvalContext::set_net`]), refreshed at the root of every search
    /// ([`EvalContext::refresh_root`]), and folded forward lazily per node.
    /// `None` leaves the PSQT hot path — and the depth-6 bench — untouched.
    nnue: Option<nnue::NnueContext<'static>>,
}

impl EvalContext {
    pub fn new() -> Self {
        Self::default()
    }

    /// Install (or tear down) the NNUE accumulator for `net` (design/11
    /// §3.4.4). `None` turns NNUE off. A net whose pointer already matches the
    /// installed one keeps its heap allocation (its columns are re-refreshed by
    /// [`EvalContext::refresh_root`]); a different net allocates a fresh stack.
    /// The *process-wide* active net is read at the search boundary and handed
    /// in here, so this method never touches global state — search unit tests
    /// (and the parallel test harness) stay hermetic.
    pub fn set_net(&mut self, net: Option<&'static nnue::Network>) {
        match net {
            None => self.nnue = None,
            Some(net) => {
                let reuse = matches!(&self.nnue, Some(ctx) if std::ptr::eq(ctx.net(), net));
                if !reuse {
                    self.nnue = Some(nnue::NnueContext::new(net));
                }
            }
        }
    }

    /// True iff NNUE is the active eval for this thread.
    #[inline]
    pub fn nnue_active(&self) -> bool {
        self.nnue.is_some()
    }

    /// Refresh the NNUE root accumulator (`stack[0]`) from `board`, both
    /// perspectives (design/11 §3.4.4). Called once at the start of every
    /// search; a no-op when NNUE is off.
    pub fn refresh_root(&mut self, board: &Board) {
        if let Some(ctx) = self.nnue.as_mut() {
            ctx.refresh_root(board);
        }
    }

    /// Record a move's placement deltas into the child accumulator entry at
    /// `ply + 1`, marking it uncomputed (the §3.4.4 invalidation rule). A no-op
    /// when NNUE is off. Pass [`DirtyPiece::EMPTY`] for a null move.
    #[inline]
    pub fn push_dirty(&mut self, dp: DirtyPiece, ply: usize) {
        if let Some(ctx) = self.nnue.as_mut() {
            ctx.push(ply, dp);
        }
    }

    /// Test-only: the RAW NNUE forward value at the root (`ply 0`) from the
    /// incrementally-maintained accumulator. Lets the search-consistency test
    /// compare the incremental root eval against a from-scratch forward in
    /// RELEASE too (where the per-node `assert_matches_refresh` oracle is
    /// compiled out). Panics if NNUE is off.
    #[cfg(test)]
    pub fn nnue_root_forward_for_test(&mut self, board: &Board) -> Value {
        self.nnue
            .as_mut()
            .expect("NNUE must be active for the root-forward check")
            .evaluate(board, 0)
    }
}

/// Static evaluation of `board` from the side to move's perspective.
///
/// `optimism` is computed by search at aspiration-window setup (owned by
/// the search doc) and blended in here; pass `0` for a neutral eval
/// (tools, tests, tuner). `ply` is the SEARCH ply of `board` (root = 0),
/// needed only to index the per-thread NNUE accumulator stack when a net is
/// loaded (design/11 §3.4.4); PSQT ignores it. Standalone callers outside a
/// search (`eval` command, tools, tuner) pass `0` after a `refresh_root`.
///
/// A four-argument reading of the §3.1 seam: the accumulator stack is
/// ply-indexed, so NNUE dispatch needs the ply the PSQT path never did
/// (design/11 §3.1/§3.4.4).
pub fn evaluate(board: &Board, ctx: &mut EvalContext, optimism: Value, ply: usize) -> Value {
    debug_assert!(
        board.checkers().is_empty(),
        "evaluate must never be called in check (design/11 §3.1)"
    );

    // Dead-draw guard (ALL phases inherit it, NNUE included): insufficient
    // mating material is an exact draw — bypass the whole pipeline. Search's
    // rule50/repetition draw handling is untouched (design/11 §3.6).
    if is_material_draw(board) {
        return VALUE_DRAW;
    }

    // Inner eval, stm-relative. NNUE when a net is loaded (design/11 §3.4),
    // PSQT otherwise — both route through the SAME §3.6 outer shape below. The
    // first net has no psqt/positional split (§3.6: `(nnue_out, 0)`), so
    // `complexity = 0` and the complexity blend still vanishes exactly as it
    // does for PSQT; NNUE therefore shares PSQT's optimism/material/rule-50
    // consumption verbatim (the raw-vs-corrected boundary is unchanged, D22).
    let inner = match ctx.nnue.as_mut() {
        Some(nnue) => {
            // Debug oracle (design/11 §3.4.4): the incrementally-maintained
            // accumulator at this ply must equal a from-scratch refresh — trips
            // any stale-accumulator bug the instant search evaluates a node.
            // Compiled out of release (the AVX2/scalar forward is the shipping
            // path).
            #[cfg(debug_assertions)]
            nnue.assert_matches_refresh(board, ply);
            nnue.evaluate(board, ply)
        }
        None => psqt::psqt_eval(board),
    };

    apply_outer_shape(
        inner,
        optimism,
        blend_material(board),
        board.rule50_count() as i64,
    )
}

/// The dead-draw material guard: true iff no side can possibly mate —
/// no pawns, one side a bare king, the other holding at most a single
/// minor or exactly two knights (KvK, KNvK, KBvK, KNNvK; KNNvK has only
/// helpmates). Cheap: reads the cached `non_pawn_material` (D8), whose
/// SEE values (D11) make the comparisons exact. Symmetric by construction,
/// so the color-flip symmetry suite stays green (design/11 §3.6).
pub fn is_material_draw(board: &Board) -> bool {
    let no_pawns = board.pieces(Color::White, PieceType::Pawn).is_empty()
        && board.pieces(Color::Black, PieceType::Pawn).is_empty();
    if !no_pawns {
        return false;
    }
    let npm_w = board.non_pawn_material(Color::White);
    let npm_b = board.non_pawn_material(Color::Black);
    let (strong, weak) = if npm_w >= npm_b {
        (npm_w, npm_b)
    } else {
        (npm_b, npm_w)
    };
    weak == 0 && (strong <= BISHOP_VALUE || strong == 2 * KNIGHT_VALUE)
}

/// The material term of the optimism blend: pawns weighted by
/// [`MATERIAL_PAWN_WEIGHT`], everything else by its SEE value via the
/// cached per-color `non_pawn_material` (design/11 §3.6).
fn blend_material(board: &Board) -> i64 {
    let pawns = (board.pieces(Color::White, PieceType::Pawn)
        | board.pieces(Color::Black, PieceType::Pawn))
    .count() as i64;
    let npm = i64::from(board.non_pawn_material(Color::White))
        + i64::from(board.non_pawn_material(Color::Black));
    MATERIAL_PAWN_WEIGHT * pawns + npm
}

/// The outer shape every eval phase wears (design/11 §3.6, complexity
/// blend elided while `complexity = 0`): blend the inner eval with
/// optimism weighted by material, damp toward zero as the rule-50 clock
/// grows, and clamp strictly inside the non-TB band.
///
/// i64 intermediates are mandatory: `v * (EVAL_BLEND_BASE + material)`
/// reaches ~3.3e9 and would overflow i32.
fn apply_outer_shape(inner: Value, optimism: Value, material: i64, rule50: i64) -> Value {
    let mut v = i64::from(inner);
    let opt = i64::from(optimism);

    // Material scaling: heavier boards trust eval more, lighter boards
    // trust optimism relatively more.
    v = (v * (EVAL_BLEND_BASE + material) + opt * (OPTIMISM_BLEND_BASE + material))
        / EVAL_BLEND_BASE;

    // Rule-50 damping: shuffle toward zero as the counter grows.
    v -= v * rule50 / RULE50_DAMPING_DEN;

    // Never collide with the TB/mate bands.
    v.clamp(
        i64::from(VALUE_TB_LOSS_IN_MAX_PLY) + 1,
        i64::from(VALUE_TB_WIN_IN_MAX_PLY) - 1,
    ) as Value
}

#[cfg(test)]
mod tests {
    use super::*;
    use crate::fen::START_FEN;

    fn board(fen: &str) -> Board {
        Board::from_fen(fen).unwrap()
    }

    fn eval0(b: &Board) -> Value {
        evaluate(b, &mut EvalContext::new(), 0, 0)
    }

    // Both sides of the startpos material blend: 16 pawns and full
    // non-pawn material, the maximum a legal game starts from.
    const STARTPOS_MATERIAL: i64 = 534 * 16 + 2 * (2538 + 2 * 1276 + 2 * 825 + 2 * 781);

    #[test]
    fn startpos_evaluates_to_zero() {
        assert_eq!(eval0(&board(START_FEN)), 0);
        assert_eq!(blend_material(&board(START_FEN)), STARTPOS_MATERIAL);
    }

    // design/11 §4 step 3: rule-50 damping at the clock boundaries.
    // With material = 0 and optimism = 0 the blend is the identity, so the
    // damping acts on the raw inner value.
    #[test]
    fn outer_shape_rule50_damping() {
        assert_eq!(apply_outer_shape(199, 0, 0, 0), 199);
        assert_eq!(apply_outer_shape(199, 0, 0, 100), 99);
        assert_eq!(apply_outer_shape(199, 0, 0, 199), 0);
        // Damping pulls toward zero from both signs.
        assert_eq!(apply_outer_shape(-199, 0, 0, 100), -99);
    }

    // design/11 §4 step 3: the clamp engages on synthetic ±40000 inner
    // values, and the i64 path survives max material (the product reaches
    // ~4.1e9, past i32::MAX).
    #[test]
    fn outer_shape_clamps_inside_tb_band() {
        assert_eq!(
            apply_outer_shape(40000, 0, 0, 0),
            VALUE_TB_WIN_IN_MAX_PLY - 1
        );
        assert_eq!(
            apply_outer_shape(-40000, 0, 0, 0),
            VALUE_TB_LOSS_IN_MAX_PLY + 1
        );
        assert_eq!(
            apply_outer_shape(40000, 0, STARTPOS_MATERIAL, 0),
            VALUE_TB_WIN_IN_MAX_PLY - 1
        );
    }

    // design/11 §5: positive optimism raises the stm eval; the optimism
    // weight grows with material on the board.
    #[test]
    fn outer_shape_optimism_sign_and_material_scaling() {
        // (0 + 100 * 7191) / 77871 = 9.
        assert_eq!(apply_outer_shape(0, 100, 0, 0), 9);
        assert_eq!(apply_outer_shape(0, -100, 0, 0), -9);
        assert!(apply_outer_shape(50, 100, 0, 0) > apply_outer_shape(50, 0, 0, 0));
        // (0 + 100 * (7191 + 25148)) / 77871 = 41 > 9.
        assert!(apply_outer_shape(0, 100, STARTPOS_MATERIAL, 0) > apply_outer_shape(0, 100, 0, 0));
    }

    // design/11 §4 step 3 / §3.6: the guard fires for exactly
    // KvK, KNvK, KBvK, KNNvK (either color as the strong side) ...
    #[test]
    fn dead_draws_evaluate_to_exact_zero() {
        let draws = [
            "k7/8/8/8/8/8/8/K7 w - - 0 1",   // KvK
            "k7/8/8/8/8/8/8/KN6 w - - 0 1",  // KNvK, white strong
            "k7/8/8/8/8/8/8/KN6 b - - 0 1",  //   ... black to move
            "kn6/8/8/8/8/8/8/K7 w - - 0 1",  // KNvK, black strong
            "k7/8/8/8/8/8/8/KB6 w - - 0 1",  // KBvK, white strong
            "kb6/8/8/8/8/8/8/K7 w - - 0 1",  // KBvK, black strong
            "k7/8/8/8/8/8/8/KNN5 w - - 0 1", // KNNvK, white strong
            "knn5/8/8/8/8/8/8/K7 w - - 0 1", // KNNvK, black strong
        ];
        for fen in draws {
            let b = board(fen);
            assert!(is_material_draw(&b), "guard must fire for {fen}");
            assert_eq!(eval0(&b), VALUE_DRAW, "dead draw must eval 0: {fen}");
        }
    }

    // ... and does NOT fire for the near misses.
    #[test]
    fn near_misses_do_not_hit_the_guard() {
        let krvk = board("k7/8/8/8/8/8/8/KR6 w - - 0 1");
        assert!(!is_material_draw(&krvk));
        assert!(eval0(&krvk) > 0, "KRvK is winning for the side to move");

        let kpvk = board("k7/8/8/8/8/8/P7/K7 w - - 0 1");
        assert!(!is_material_draw(&kpvk));
        assert!(eval0(&kpvk) != 0);

        let kbvkn = board("kn6/8/8/8/8/8/8/KB6 w - - 0 1");
        assert!(!is_material_draw(&kbvkn));
    }

    #[cfg(debug_assertions)]
    #[test]
    #[should_panic(expected = "never be called in check")]
    fn evaluate_asserts_not_in_check() {
        // Black to move, black king on e8 checked by the rook on e1.
        let b = board("4k3/8/8/8/8/8/8/K3R3 b - - 0 1");
        eval0(&b);
    }
}
