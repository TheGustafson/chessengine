//! Evaluation: the single `evaluate` seam (D22), pinned across all eval
//! phases so PSQT → NNUE swap without touching search (design/11 §3.1).
//!
//! Contract: the returned [`Value`] is from the side to move; never called
//! in check (search skips the static eval at in-check nodes); strictly
//! inside the non-TB band; reads `board`, mutates only `ctx`. Correction
//! history is applied in *search*, never here — the raw return value is
//! what the TT's `eval16` stores (D22).

pub mod nnue;
pub mod params;
pub mod psqt;
pub mod score;

use crate::board::{Board, DirtyPiece};
use crate::types::{
    BISHOP_VALUE, Color, KNIGHT_VALUE, PieceType, VALUE_DRAW, VALUE_TB_LOSS_IN_MAX_PLY,
    VALUE_TB_WIN_IN_MAX_PLY, Value,
};
use params::{
    EVAL_BLEND_BASE, EVAL_SCALE, MATERIAL_PAWN_WEIGHT, OPTIMISM_BLEND_BASE, RULE50_DAMPING_DEN,
};

/// Per-thread mutable evaluation state (accumulator stacks), owned by the
/// search `Worker` (D23); part of the pinned `evaluate` signature.
#[derive(Debug, Default)]
pub struct EvalContext {
    /// The per-thread NNUE accumulator stack over the active net (design/11
    /// §3.4.4), or `None` when NNUE is off (⇒ PSQT).
    nnue: Option<nnue::NnueContext<'static>>,
    /// The per-thread SOTA (D40) eval context, or `None`. Checked BEFORE
    /// `nnue` in `evaluate()`, so a loaded CENN-v2 net takes priority.
    #[cfg(feature = "sota")]
    sota: Option<nnue::sota_context::SotaContext>,
    #[cfg(feature = "sota")]
    simple_halfka: Option<nnue::simple_halfka::SimpleHalfkaContext>,
    #[cfg(feature = "sota")]
    protected_halfka: Option<nnue::protected_halfka::ProtectedHalfkaContext>,
    /// How many plies of the CURRENT line (root to this node) are null moves.
    /// A null move pushes no accumulator entry (mirrors Stockfish), so the
    /// stacks are indexed by `ply - null_plies`, not the search ply.
    ///
    /// **INVARIANT: every search ply that reaches an accumulator stack goes
    /// through [`acc_ply`](Self::acc_ply).** The `debug_assert` there catches
    /// an unbalanced counter but NOT a missing `acc_ply` call — a new
    /// accumulator-indexed accessor must add one.
    null_plies: usize,
    /// Cached `simple_halfka.is_some() || protected_halfka.is_some()`, so
    /// [`push_sota_dirty`](Self::push_sota_dirty) skips two dead `Option`
    /// tests per node. Maintained by the two setters (the only writers) and
    /// pinned by the `debug_assert_eq!` in `push_sota_dirty`.
    #[cfg(feature = "sota")]
    side_nets: bool,
}

impl EvalContext {
    pub fn new() -> Self {
        Self::default()
    }

    /// Install (or tear down) the NNUE accumulator for `net`; `None` turns
    /// NNUE off. A net whose pointer matches the installed one keeps its
    /// heap allocation. Never touches global state — tests stay hermetic.
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

    /// Install (or tear down) the D40 SOTA eval context; `None` turns SOTA
    /// off. When active, `evaluate()` routes through `SotaContext::evaluate`.
    #[cfg(feature = "sota")]
    pub fn set_sota_net(&mut self, net: Option<&'static nnue::sota_loader::SotaNetwork>) {
        match net {
            None => self.sota = None,
            Some(net) => {
                let reuse = matches!(&self.sota, Some(ctx) if std::ptr::eq(ctx.net(), net));
                if !reuse {
                    self.sota = Some(nnue::sota_context::SotaContext::new(net));
                }
                // Refresh the policy-head snapshot even on context reuse: a
                // sidecar can change between `go`s; the hot path must never
                // touch the RwLock.
                if let Some(ctx) = &mut self.sota {
                    ctx.refresh_policy_head();
                }
            }
        }
    }

    /// Recompute the [`side_nets`](Self::side_nets) cache; called by both
    /// setters so the flag cannot drift.
    #[cfg(feature = "sota")]
    fn refresh_side_net_flag(&mut self) {
        self.side_nets = self.simple_halfka.is_some() || self.protected_halfka.is_some();
    }

    #[cfg(feature = "sota")]
    pub fn set_simple_halfka_net(
        &mut self,
        net: Option<&'static nnue::simple_halfka::SimpleHalfkaNetwork>,
    ) {
        match net {
            None => self.simple_halfka = None,
            Some(net) => {
                let reuse =
                    matches!(&self.simple_halfka, Some(ctx) if std::ptr::eq(ctx.net(), net));
                if !reuse {
                    self.simple_halfka = Some(nnue::simple_halfka::SimpleHalfkaContext::new(net));
                }
            }
        }
        self.refresh_side_net_flag();
    }

    #[cfg(feature = "sota")]
    pub fn set_protected_halfka_net(
        &mut self,
        net: Option<&'static nnue::protected_halfka::ProtectedHalfkaNetwork>,
    ) {
        match net {
            None => self.protected_halfka = None,
            Some(net) => {
                let reuse =
                    matches!(&self.protected_halfka, Some(ctx) if std::ptr::eq(ctx.net(), net));
                if !reuse {
                    self.protected_halfka =
                        Some(nnue::protected_halfka::ProtectedHalfkaContext::new(net));
                }
            }
        }
        self.refresh_side_net_flag();
    }

    /// The value-uncertainty σ̂ (per-position value std in sigmoid/WDL space)
    /// from the most recent SOTA `evaluate()`, or `None` when SOTA is not
    /// active or the net has no variance head; consumers fall back on `None`.
    #[inline]
    pub fn last_sigma(&self) -> Option<f32> {
        #[cfg(feature = "sota")]
        if let Some(ctx) = self.sota.as_ref() {
            return ctx.last_sigma();
        }
        None
    }

    /// Census-dump accessor: the last head forward's u8 FT vector, `None`
    /// off the sota path. See `SotaContext::last_ft_slice`.
    #[inline]
    pub fn last_ft_slice(&self) -> Option<&[u8]> {
        #[cfg(feature = "sota")]
        if let Some(ctx) = self.sota.as_ref() {
            return Some(ctx.last_ft_slice());
        }
        None
    }

    /// The disagreement head on the last SOTA eval's FT vector — `None` on a
    /// net without a DIS1 section or when the cheap path ran. VALIDITY: the
    /// cached vector belongs to the last position that ran a fresh eval, so
    /// a TT-served node must not poll it.
    #[inline]
    pub fn disagreement_cached(&mut self) -> Option<[f32; 2]> {
        #[cfg(feature = "sota")]
        if let Some(ctx) = self.sota.as_mut() {
            return ctx.disagreement_cached();
        }
        None
    }

    /// The disagreement head for `board` at the ROOT, re-running the FT
    /// activation rather than reusing a cached vector. Once-per-search
    /// callers only; see `disagreement_cached` for the per-node form.
    #[inline]
    #[cfg_attr(not(feature = "sota"), allow(unused_variables))]
    pub fn disagreement_root(&mut self, board: &Board) -> Option<[f32; 2]> {
        #[cfg(feature = "sota")]
        if let Some(ctx) = self.sota.as_mut() {
            return ctx.disagreement(board, 0);
        }
        None
    }

    /// The WDL probabilities [L,D,W] from the most recent SOTA eval, or
    /// `None` when the net has no WDL head / the cheap path ran; consumers
    /// fall back to no-op on `None`.
    #[inline]
    pub fn last_wdl(&self) -> Option<[f32; 3]> {
        #[cfg(feature = "sota")]
        if let Some(ctx) = self.sota.as_ref() {
            return ctx.last_wdl();
        }
        None
    }

    /// The shared-bottleneck activation for exactly `board`'s position
    /// (Zobrist-stamped freshness), for the policy-ordering consumer; `None`
    /// when no policy head is active, the last eval took a cheap path, or
    /// the stash belongs to a different node.
    #[cfg(feature = "sota")]
    #[inline]
    pub fn last_l2_for(
        &self,
        board: &crate::Board,
    ) -> Option<&[f32; crate::eval::nnue::sota_head::L2]> {
        if let Some(ctx) = self.sota.as_ref() {
            return ctx.last_l2_for(board);
        }
        None
    }

    /// True iff NNUE (v1 or SOTA) is the active eval for this thread.
    #[inline]
    pub fn nnue_active(&self) -> bool {
        #[cfg(feature = "sota")]
        if self.sota.is_some() {
            return true;
        }
        #[cfg(feature = "sota")]
        if self.simple_halfka.is_some() {
            return true;
        }
        #[cfg(feature = "sota")]
        if self.protected_halfka.is_some() {
            return true;
        }
        self.nnue.is_some()
    }

    /// Refresh the root accumulators (`stack[0]`) from `board`, both
    /// perspectives, once at the start of every search; no-op when NNUE off.
    pub fn refresh_root(&mut self, board: &Board) {
        // Every search starts with an empty null-move stack, whatever the
        // last one left behind (a caught panic can skip a `pop_null`).
        self.null_plies = 0;
        if let Some(ctx) = self.nnue.as_mut() {
            ctx.refresh_root(board);
        }
        #[cfg(feature = "sota")]
        if let Some(ctx) = self.sota.as_mut() {
            ctx.refresh_root(board);
        }
        #[cfg(feature = "sota")]
        if let Some(ctx) = self.simple_halfka.as_mut() {
            ctx.refresh_root(board);
        }
        #[cfg(feature = "sota")]
        if let Some(ctx) = self.protected_halfka.as_mut() {
            ctx.refresh_root(board);
        }
    }

    /// The ACCUMULATOR ply for search ply `ply` — see [`Self::null_plies`].
    /// `saturating_sub` because `null_plies <= ply` is a SEARCH invariant:
    /// release clamps to accumulator ply 0 — stale but IN-RANGE — instead of
    /// wrapping `usize` and silently poisoning every later fold.
    #[inline]
    fn acc_ply(&self, ply: usize) -> usize {
        debug_assert!(
            self.null_plies <= ply,
            "null_plies {} exceeds ply {ply} — push_null/pop_null are unbalanced",
            self.null_plies
        );
        ply.saturating_sub(self.null_plies)
    }

    /// Enter a null move: the accumulator stacks do NOT advance, so this ply
    /// and its child share one entry. Must be paired with [`Self::pop_null`].
    #[inline]
    pub fn push_null(&mut self) {
        self.null_plies += 1;
    }

    /// Leave a null move — the inverse of [`Self::push_null`].
    #[inline]
    pub fn pop_null(&mut self) {
        debug_assert!(self.null_plies > 0, "pop_null without a matching push_null");
        // `saturating_sub` as in `acc_ply`: an unmatched pop must not wrap.
        self.null_plies = self.null_plies.saturating_sub(1);
    }

    /// Record a move's placement deltas into the child accumulator entry,
    /// marking it uncomputed (the §3.4.4 invalidation rule). A no-op when
    /// NNUE is off. Null moves use [`push_null`](Self::push_null) instead.
    #[inline]
    pub fn push_dirty(&mut self, dp: DirtyPiece, ply: usize) {
        debug_assert!(
            !(dp.adds().is_empty() && dp.subs().is_empty()),
            "an empty dirty record is a null move; use push_null"
        );
        let acc = self.acc_ply(ply);
        if let Some(ctx) = self.nnue.as_mut() {
            ctx.push(acc, dp);
        }
    }

    /// Classic v1 accumulator pair for a frozen policy head; `None` when the
    /// active evaluator is PSQT or D40.
    #[cfg(feature = "search-control")]
    pub fn classic_policy_accumulators(
        &mut self,
        board: &Board,
        ply: usize,
    ) -> Option<(&[i16; nnue::L1], &[i16; nnue::L1], nnue::Activation)> {
        #[cfg(feature = "sota")]
        if self.sota.is_some() {
            return None;
        }
        let acc = self.acc_ply(ply);
        self.nnue
            .as_mut()
            .map(|ctx| ctx.policy_accumulators(board, acc))
    }

    /// Fold a just-made move into the SOTA HalfKA and FullThreats child
    /// stacks; must run after board mutation (threat x-rays read the child).
    #[cfg(feature = "sota")]
    #[inline]
    pub fn push_sota_dirty(&mut self, dp: DirtyPiece, ply: usize, board: &Board) {
        // Same tripwire as `push_dirty`'s: an empty record arriving here
        // would be materialised rather than aliased.
        debug_assert!(
            !(dp.adds().is_empty() && dp.subs().is_empty()),
            "an empty dirty record is a null move; use push_null"
        );
        let acc = self.acc_ply(ply);
        if let Some(ctx) = self.sota.as_mut() {
            ctx.push(dp, acc, board);
        }
        // A stale `false` here would silently skip a side-net push (wrong
        // eval, no crash), so the cache is pinned at its consumer.
        debug_assert_eq!(
            self.side_nets,
            self.simple_halfka.is_some() || self.protected_halfka.is_some(),
            "side_nets drifted from the side-net contexts"
        );
        if self.side_nets {
            if let Some(ctx) = self.simple_halfka.as_mut() {
                ctx.push(dp, acc);
            }
            if let Some(ctx) = self.protected_halfka.as_mut() {
                ctx.push(dp, acc);
            }
        }
    }

    /// Test-only: the RAW NNUE forward value at the root from the incremental
    /// accumulator, so tests can check incremental == from-scratch in RELEASE
    /// too. Panics if NNUE is off.
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
/// `optimism` is computed by search and blended in here; pass `0` for a
/// neutral eval. `ply` is the SEARCH ply of `board` (root = 0), used only to
/// index the accumulator stacks; standalone callers pass `0` after a
/// `refresh_root`.
pub fn evaluate(board: &Board, ctx: &mut EvalContext, optimism: Value, ply: usize) -> Value {
    debug_assert!(
        board.checkers().is_empty(),
        "evaluate must never be called in check (design/11 §3.1)"
    );

    // Invalidate σ̂ up front: any path that skips the SOTA forward this node
    // must leave `last_sigma() == None`, never a stale prior-position value.
    // Only a fresh `SotaContext::evaluate` re-sets it.
    #[cfg(feature = "sota")]
    if let Some(sota) = ctx.sota.as_mut() {
        sota.invalidate_sigma();
    }

    // Dead-draw guard (all phases, NNUE included): insufficient mating
    // material is an exact draw — bypass the whole pipeline (design/11 §3.6).
    if is_material_draw(board) {
        return VALUE_DRAW;
    }

    // Inner eval, stm-relative: SOTA (D40) > v1 NNUE > PSQT, all through the
    // SAME §3.6 outer shape. The accumulator ply is bound to a DIFFERENT
    // name, not shadowing `ply`: a future ply-dependent term in the outer
    // shape must not silently pick up the accumulator index.
    let acc = ctx.acc_ply(ply);
    #[cfg(feature = "sota")]
    if let Some(residual) = ctx.protected_halfka.as_mut() {
        let base = ctx
            .nnue
            .as_mut()
            .expect("CENN-v4 always installs its frozen classic base")
            .evaluate(board, acc);
        let inner = base + residual.evaluate_residual(board, acc);
        return apply_outer_shape(
            inner,
            optimism,
            blend_material(board),
            board.rule50_count() as i64,
        );
    }
    #[cfg(feature = "sota")]
    if let Some(sota) = ctx.sota.as_mut() {
        let inner = sota.evaluate(board, acc);
        return apply_outer_shape(
            inner,
            optimism,
            blend_material(board),
            board.rule50_count() as i64,
        );
    }
    #[cfg(feature = "sota")]
    if let Some(simple) = ctx.simple_halfka.as_mut() {
        let inner = simple.evaluate(board, acc);
        return apply_outer_shape(
            inner,
            optimism,
            blend_material(board),
            board.rule50_count() as i64,
        );
    }

    let inner = match ctx.nnue.as_mut() {
        Some(nnue) => {
            // Debug oracle (design/11 §3.4.4): the incremental accumulator
            // must equal a from-scratch refresh. Compiled out of release.
            #[cfg(debug_assertions)]
            nnue.assert_matches_refresh(board, acc);
            nnue.evaluate(board, acc)
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

/// The dead-draw material guard: true iff no side can possibly mate — no
/// pawns, one side a bare king, the other at most a single minor or exactly
/// two knights (KvK, KNvK, KBvK, KNNvK; KNNvK has only helpmates).
/// Symmetric by construction (design/11 §3.6).
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
/// [`MATERIAL_PAWN_WEIGHT`], everything else by its SEE value (design/11 §3.6).
fn blend_material(board: &Board) -> i64 {
    let pawns = (board.pieces(Color::White, PieceType::Pawn)
        | board.pieces(Color::Black, PieceType::Pawn))
    .count() as i64;
    let npm = i64::from(board.non_pawn_material(Color::White))
        + i64::from(board.non_pawn_material(Color::Black));
    MATERIAL_PAWN_WEIGHT() * pawns + npm
}

/// The outer shape every eval phase wears (design/11 §3.6): blend the inner
/// eval with optimism weighted by material, damp toward zero as the rule-50
/// clock grows, and clamp strictly inside the non-TB band.
///
/// i64 intermediates are mandatory: `v * (EVAL_BLEND_BASE + material)`
/// reaches ~3.3e9 and would overflow i32.
fn apply_outer_shape(inner: Value, optimism: Value, material: i64, rule50: i64) -> Value {
    let mut v = i64::from(inner);
    let opt = i64::from(optimism);

    // Heavier boards trust eval more; lighter boards trust optimism more.
    v = (v * (EVAL_BLEND_BASE() + material) + opt * (OPTIMISM_BLEND_BASE() + material))
        / EVAL_BLEND_BASE();

    // Rule-50 damping: shuffle toward zero as the counter grows.
    v -= v * rule50 / RULE50_DAMPING_DEN();

    // Eval-frame scale: identity at 1024 (the shipped engine). Applied LAST
    // so it scales the finished eval rather than re-weighting the blend
    // against optimism. Rounded half away from zero: truncation would bias
    // toward zero, sign-asymmetrically.
    if EVAL_SCALE() != 1024 {
        let n = v * EVAL_SCALE();
        v = if n >= 0 {
            (n + 512) / 1024
        } else {
            -((-n + 512) / 1024)
        };
    }

    // Never collide with the TB/mate bands (also bounds EVAL_SCALE above).
    v.clamp(
        i64::from(VALUE_TB_LOSS_IN_MAX_PLY) + 1,
        i64::from(VALUE_TB_WIN_IN_MAX_PLY) - 1,
    ) as Value
}

// Gated on `sota`: at default features the module would be empty.
#[cfg(all(test, feature = "sota"))]
mod side_net_flag_tests {
    use super::*;
    use std::sync::LazyLock;

    /// POISON fixture: a structurally valid network with EMPTY weight tables,
    /// exercising the INSTALLED branch of the setters (the constructors only
    /// store the pointer). **Never `refresh_root` or `evaluate` through one**
    /// — the refresh PANICS on the table-length mismatch; test the flag, not
    /// the forward pass. `LazyLock` keeps the pointer STABLE so a second
    /// install of the "same" net takes the setter's `reuse` path.
    static POISON_SIMPLE: LazyLock<nnue::simple_halfka::SimpleHalfkaNetwork> =
        LazyLock::new(|| nnue::simple_halfka::SimpleHalfkaNetwork {
            halfka_weights: Box::new([]),
            ft_bias: Box::new([]),
            out_weights: Box::new([]),
            out_bias: 0,
        });

    /// The `protected_halfka` twin: that net installs LAST in
    /// `Worker::refresh_eval_net`, so protected-only is decided by its setter.
    static POISON_PROTECTED: LazyLock<nnue::protected_halfka::ProtectedHalfkaNetwork> =
        LazyLock::new(|| nnue::protected_halfka::ProtectedHalfkaNetwork {
            base: nnue::network::Network {
                ft_weights: Box::new([]),
                ft_bias: Box::new([]),
                out_weights: Box::new([]),
                out_bias: 0,
                activation: nnue::network::Activation::SCReLU,
                qa: 1,
                qb: 1,
                scale: 1,
            },
            halfka_weights: Box::new([]),
            ft_bias: Box::new([]),
            out_weights: Box::new([]),
            out_bias: 0,
        });

    // A stale `false` in the cache is a silent wrong-eval bug, not a crash.
    // Drive it through the only two writers.
    #[test]
    fn side_net_flag_tracks_the_optional_contexts() {
        let mut ctx = EvalContext::new();
        assert!(!ctx.side_nets, "a fresh context has no side nets");

        ctx.set_simple_halfka_net(Some(&POISON_SIMPLE));
        assert!(ctx.side_nets, "installing a side net must arm the flag");

        // A reuse-path install does not touch the field — the flag survives.
        ctx.set_simple_halfka_net(Some(&POISON_SIMPLE));
        assert!(ctx.side_nets, "reuse must leave the flag armed");

        ctx.set_simple_halfka_net(None);
        assert!(!ctx.side_nets, "removing the last side net must disarm it");

        // And the flag is the DISJUNCTION: the other setter must not clear it.
        ctx.set_simple_halfka_net(Some(&POISON_SIMPLE));
        ctx.set_protected_halfka_net(None);
        assert!(
            ctx.side_nets,
            "clearing the OTHER side net must not disarm the flag"
        );
    }

    // The PROTECTED leg, which the test above cannot reach: only this test
    // catches a missing refresh in the protected setter (that net installs
    // LAST, so protected-only is the dangerous configuration).
    #[test]
    fn side_net_flag_arms_from_the_protected_setter_alone() {
        let mut ctx = EvalContext::new();
        assert!(!ctx.side_nets);

        ctx.set_protected_halfka_net(Some(&POISON_PROTECTED));
        assert!(
            ctx.side_nets,
            "the protected setter must arm the flag on its own"
        );

        // The other setter clearing its own (absent) slot must not disarm it.
        ctx.set_simple_halfka_net(None);
        assert!(ctx.side_nets, "protected is still installed");

        ctx.set_protected_halfka_net(None);
        assert!(!ctx.side_nets, "removing the last side net disarms it");
    }
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

    // Startpos blend material: 16 pawns + full non-pawn material.
    const STARTPOS_MATERIAL: i64 = 534 * 16 + 2 * (2538 + 2 * 1276 + 2 * 825 + 2 * 781);

    #[test]
    fn startpos_evaluates_to_zero() {
        assert_eq!(eval0(&board(START_FEN)), 0);
        assert_eq!(blend_material(&board(START_FEN)), STARTPOS_MATERIAL);
    }

    // Rule-50 damping at the clock boundaries; with material = optimism = 0
    // the blend is the identity, so damping acts on the raw inner value.
    #[test]
    fn outer_shape_rule50_damping() {
        assert_eq!(apply_outer_shape(199, 0, 0, 0), 199);
        assert_eq!(apply_outer_shape(199, 0, 0, 100), 99);
        assert_eq!(apply_outer_shape(199, 0, 0, 199), 0);
        // Damping pulls toward zero from both signs.
        assert_eq!(apply_outer_shape(-199, 0, 0, 100), -99);
    }

    // The clamp engages on synthetic ±40000 inner values, and the i64 path
    // survives max material (the product reaches ~4.1e9, past i32::MAX).
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

    // Positive optimism raises the stm eval; its weight grows with material.
    #[test]
    fn outer_shape_optimism_sign_and_material_scaling() {
        // (0 + 100 * 7191) / 77871 = 9.
        assert_eq!(apply_outer_shape(0, 100, 0, 0), 9);
        assert_eq!(apply_outer_shape(0, -100, 0, 0), -9);
        assert!(apply_outer_shape(50, 100, 0, 0) > apply_outer_shape(50, 0, 0, 0));
        // (0 + 100 * (7191 + 25148)) / 77871 = 41 > 9.
        assert!(apply_outer_shape(0, 100, STARTPOS_MATERIAL, 0) > apply_outer_shape(0, 100, 0, 0));
    }

    // The guard fires for exactly KvK, KNvK, KBvK, KNNvK (either color) ...
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
