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
use params::{
    EVAL_BLEND_BASE, EVAL_SCALE, MATERIAL_PAWN_WEIGHT, OPTIMISM_BLEND_BASE, RULE50_DAMPING_DEN,
};

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
    /// The per-thread SOTA (D40) eval context, or `None`. Checked BEFORE `nnue`
    /// in `evaluate()`, so a loaded CENN-v2 net takes priority. Behind the `sota`
    /// feature; the default build never compiles this field.
    #[cfg(feature = "sota")]
    sota: Option<nnue::sota_context::SotaContext>,
    #[cfg(feature = "sota")]
    simple_halfka: Option<nnue::simple_halfka::SimpleHalfkaContext>,
    #[cfg(feature = "sota")]
    protected_halfka: Option<nnue::protected_halfka::ProtectedHalfkaContext>,
    /// How many plies of the CURRENT line are null moves (perf study 15 arm 1).
    ///
    /// Counted over the path from the root to the current node — NOT the
    /// subtree below it.
    ///
    /// A null move leaves every piece where it stands, so its accumulator entry
    /// would be a bit-identical copy of its parent's. Stockfish therefore pushes
    /// **no** accumulator entry for one — `do_null_move` has no
    /// `accumulatorStack.push()` and `undo_null_move` no `pop()`
    /// (`eca43a97:src/search.cpp:657-669`) — and neither do we. The accumulator
    /// stacks are consequently indexed by `ply - null_plies`, not by the search
    /// ply, and the two diverge by one for every null move between the root and
    /// the current node.
    ///
    /// A scalar is enough because the search is a depth-first walk: the "current
    /// line" is exactly the recursion stack, so [`push_null`](Self::push_null)
    /// and [`pop_null`](Self::pop_null) bracket it the same way `make_null_move`
    /// / `unmake_null_move` bracket the board.
    ///
    /// **INVARIANT: every search ply that reaches an accumulator stack goes
    /// through [`acc_ply`](Self::acc_ply).** Both references keep the cursor
    /// inside the accumulator stack itself instead — Stockfish's
    /// `AccumulatorStack::size` (`eca43a97:src/nnue/nnue_accumulator.h:135`),
    /// moved only by its own `push`/`pop` — which makes a wrong index
    /// unrepresentable, where this shape makes it merely wrong at four
    /// translation sites (D33 reviewer B, M1). The `debug_assert` in `acc_ply`
    /// catches an unbalanced counter; it CANNOT catch a missing `acc_ply` call,
    /// so a new accumulator-indexed accessor must add one. Moving the cursor
    /// into the contexts is the follow-up that would close it structurally.
    null_plies: usize,
    /// `true` iff either experimental side net is installed — the cached
    /// disjunction of `simple_halfka.is_some()` and
    /// `protected_halfka.is_some()`.
    ///
    /// Neither is ever installed in a shipped configuration, so
    /// [`push_sota_dirty`](Self::push_sota_dirty) ran two dead `Option` tests
    /// on every node (perf study 16 §6). Maintained by the two setters, which
    /// are the only writers of either field, and by the `debug_assert_eq!` in
    /// [`push_sota_dirty`](Self::push_sota_dirty).
    #[cfg(feature = "sota")]
    side_nets: bool,
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

    /// Install (or tear down) the D40 SOTA eval context. `None` turns SOTA off.
    /// When SOTA is active, `evaluate()` routes through `SotaContext::evaluate`
    /// instead of the v1 NNUE or PSQT paths.
    #[cfg(feature = "sota")]
    pub fn set_sota_net(&mut self, net: Option<&'static nnue::sota_loader::SotaNetwork>) {
        match net {
            None => self.sota = None,
            Some(net) => {
                let reuse = matches!(&self.sota, Some(ctx) if std::ptr::eq(ctx.net(), net));
                if !reuse {
                    self.sota = Some(nnue::sota_context::SotaContext::new(net));
                }
                // The policy-head snapshot refreshes even on context reuse: a
                // sidecar can be (un)installed between `go`s with the same
                // net, and this is the once-per-eval-boundary read the slot's
                // contract prescribes (perf study 3 §5.2 — the hot path must
                // never touch the RwLock).
                if let Some(ctx) = &mut self.sota {
                    ctx.refresh_policy_head();
                }
            }
        }
    }

    /// Recompute the [`side_nets`](Self::side_nets) cache from the two fields
    /// it summarises. Called by both setters — which are the only writers of
    /// either field — so the flag cannot drift.
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

    /// EXP-UNC-1 (search study bet 1): the value-uncertainty σ̂ produced by the
    /// most recent SOTA `evaluate()`, or `None` when SOTA is not the active eval
    /// or the loaded net has no variance head. σ̂ is the per-position value std in
    /// sigmoid (WDL) space; the search's uncertainty consumers (RFP/futility
    /// margins, TM soft-stop) read it and fall back to their constants on `None`,
    /// so a variance-less net is bit-identical to today. See
    /// [`nnue::sota_context::SotaContext::last_sigma`].
    #[inline]
    pub fn last_sigma(&self) -> Option<f32> {
        #[cfg(feature = "sota")]
        if let Some(ctx) = self.sota.as_ref() {
            return ctx.last_sigma();
        }
        None
    }

    /// D41 rich-net: the WDL probabilities [L,D,W] from the most recent SOTA eval,
    /// or `None` when the net has no WDL head / the cheap path ran. Search reads it
    /// for the decisiveness / optimism consumers; `None` → those fall back to no-op.
    #[inline]
    pub fn last_wdl(&self) -> Option<[f32; 3]> {
        #[cfg(feature = "sota")]
        if let Some(ctx) = self.sota.as_ref() {
            return ctx.last_wdl();
        }
        None
    }

    /// D47: the shared-bottleneck activation for exactly `board`'s position
    /// (Zobrist-stamped freshness), for the policy-ordering consumer. `None`
    /// when no policy head is active, the last eval took a cheap path, or the
    /// stash belongs to a different node.
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

    /// Refresh the NNUE root accumulator (`stack[0]`) from `board`, both
    /// perspectives (design/11 §3.4.4). Called once at the start of every
    /// search; a no-op when NNUE is off. Both v1 and SOTA seed their ply-zero
    /// stacks here.
    pub fn refresh_root(&mut self, board: &Board) {
        // The root is ply 0 AND accumulator ply 0 — every search starts with an
        // empty null-move stack, whatever the last one left behind (a panic
        // caught by the pool's `catch_unwind` can skip a `pop_null`).
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
    ///
    /// `null_plies` counts the null moves on the path from the root to this
    /// node, so a balanced `push_null`/`pop_null` bracketing keeps it `<= ply`
    /// and the subtraction exact. That is a SEARCH invariant, not a
    /// construction-level one (D33 reviewer A, A8), so release does not rely on
    /// it: `saturating_sub` clamps to accumulator ply 0 — a stale but IN-RANGE
    /// entry — instead of underflowing `usize` and wrapping
    /// `HalfkaContext::push`'s `stack[ply + 1]` back onto the ROOT entry, which
    /// would silently poison every later fold. Same reasoning, and the same
    /// choice, as `NnueContext::compute`'s saturating ancestor walk.
    #[inline]
    fn acc_ply(&self, ply: usize) -> usize {
        debug_assert!(
            self.null_plies <= ply,
            "null_plies {} exceeds ply {ply} — push_null/pop_null are unbalanced",
            self.null_plies
        );
        ply.saturating_sub(self.null_plies)
    }

    /// Enter a null move: the accumulator stacks do NOT advance, so this ply and
    /// its child share one entry. Must be paired with [`Self::pop_null`], which
    /// the search brackets exactly like `make_null_move`/`unmake_null_move`.
    #[inline]
    pub fn push_null(&mut self) {
        self.null_plies += 1;
    }

    /// Leave a null move — the inverse of [`Self::push_null`].
    #[inline]
    pub fn pop_null(&mut self) {
        debug_assert!(self.null_plies > 0, "pop_null without a matching push_null");
        // `saturating_sub` for the same reason as `acc_ply`'s: an unmatched pop
        // would wrap to `usize::MAX` and turn every later `acc_ply` into
        // `ply + 1` — a silent off-by-one into a stale sibling entry.
        self.null_plies = self.null_plies.saturating_sub(1);
    }

    /// Record a move's placement deltas into the child accumulator entry,
    /// marking it uncomputed (the §3.4.4 invalidation rule). A no-op when NNUE
    /// is off. Null moves do NOT come through here — they use
    /// [`push_null`](Self::push_null), which pushes no entry at all.
    /// SOTA consumes the same record after board mutation via
    /// [`EvalContext::push_sota_dirty`].
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

    /// Classic v1 accumulator pair for a frozen policy head. `None` when the
    /// active evaluator is PSQT or D40 rather than the bit-identical CENN-v1
    /// sister expected by this experiment.
    #[cfg(any(feature = "policy", feature = "search-control"))]
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

    /// Fold a just-made move into the SOTA HalfKA and FullThreats child stacks.
    /// This must run after board mutation because threat x-rays read the child.
    #[cfg(feature = "sota")]
    #[inline]
    pub fn push_sota_dirty(&mut self, dp: DirtyPiece, ply: usize, board: &Board) {
        // The same tripwire as `push_dirty`'s, and it matters MORE here: these
        // are the two stacks whose null-move alias walks perf study 15 deleted,
        // so an empty record arriving from a search would now be materialised
        // rather than aliased (D33 reviewer C, M8).
        debug_assert!(
            !(dp.adds().is_empty() && dp.subs().is_empty()),
            "an empty dirty record is a null move; use push_null"
        );
        let acc = self.acc_ply(ply);
        if let Some(ctx) = self.sota.as_mut() {
            ctx.push(dp, acc, board);
        }
        // ONE test for both experimental side nets (perf study 16 §6). They are
        // installed only by `set_simple_halfka_net` / `set_protected_halfka_net`,
        // which fire only for a CENN-v3/v4 `EvalFile` (`uci.rs`) — never for the
        // canonical net — and both maintain `side_nets`.
        //
        // A cached predicate that is WRONG here does not crash: it silently
        // skips a side net's accumulator push and the net then evaluates from
        // stale lanes. So the cache is pinned at its consumer, which runs at
        // every node of every debug search — the same place
        // `assert_matches_refresh` catches a stale accumulator.
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

    // Perf study 65: same freshness discipline as σ̂ below, for the PSQT-only
    // elision candidate — a material-draw shortcut or a non-SOTA net must leave
    // `last_psqt_only() == None` rather than the PREVIOUS position's value.
    #[cfg(feature = "foldcount")]
    crate::eval::nnue::foldcount::clear_psqt_only();

    // EXP-UNC-1 (D33 review, reviewers A/C): invalidate the SOTA value-uncertainty
    // σ̂ up front, so any path that does NOT run the SOTA forward this node
    // (the material-draw shortcut below, or a non-SOTA active net) leaves
    // `last_sigma() == None` rather than a STALE prior-position value that the
    // search would fold into its σ̄ EMA / root capture. Only a fresh
    // `SotaContext::evaluate` below re-sets it.
    #[cfg(feature = "sota")]
    if let Some(sota) = ctx.sota.as_mut() {
        sota.invalidate_sigma();
    }

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
    // SOTA (D40) takes priority over v1 NNUE, which takes priority over PSQT.
    // The cfg(feature) gate compiles this branch out of default builds entirely.
    //
    // Every accumulator stack below is indexed by the ACCUMULATOR ply, which is
    // the search ply minus the null moves on the current line — a null move
    // pushes no entry (`EvalContext::null_plies`). Bound to a DIFFERENT name,
    // not shadowing `ply`: `ply`'s doc comment above says "the SEARCH ply", and
    // a future ply-dependent term in the outer shape must not silently pick up
    // the accumulator index instead (D33 reviewer B, M2).
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
            // Debug oracle (design/11 §3.4.4): the incrementally-maintained
            // accumulator at this ply must equal a from-scratch refresh — trips
            // any stale-accumulator bug the instant search evaluates a node.
            // Compiled out of release (the AVX2/scalar forward is the shipping
            // path).
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

/// Perf study 65 (instrument only) — put a CANDIDATE inner value through the
/// exact outer shape [`evaluate`] applies, so a cheap predictor's number is
/// directly comparable with the eval it is trying to replace.
///
/// The alternative was to re-derive the optimism/material/rule-50 blend in the
/// analysis script, which is the "a fact in a doc gets forgotten, a constructor
/// cannot" failure mode: this way the census and the engine share one
/// implementation by construction.
#[cfg(feature = "foldcount")]
pub fn shape_inner(board: &Board, optimism: Value, inner: Value) -> Value {
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
    MATERIAL_PAWN_WEIGHT() * pawns + npm
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
    v = (v * (EVAL_BLEND_BASE() + material) + opt * (OPTIMISM_BLEND_BASE() + material))
        / EVAL_BLEND_BASE();

    // Rule-50 damping: shuffle toward zero as the counter grows.
    v -= v * rule50 / RULE50_DAMPING_DEN();

    // Eval-frame scale (design note: eval-frame-scale-model §10). Identity at
    // 1024, so this is a no-op in the shipped engine. Applied LAST, after the
    // blend and the damping, so it scales the finished eval rather than
    // re-weighting the blend against optimism — the blend's own constants are
    // ratios and must not move with the frame.
    //
    // Rounded half away from zero rather than truncated: truncation would bias
    // every eval toward zero and, being sign-asymmetric, would bias the two
    // sides of the score differently.
    if EVAL_SCALE() != 1024 {
        let n = v * EVAL_SCALE();
        v = if n >= 0 {
            (n + 512) / 1024
        } else {
            -((-n + 512) / 1024)
        };
    }

    // Never collide with the TB/mate bands. This also bounds the scale above:
    // a large EVAL_SCALE cannot push a normal eval into the decisive bands.
    v.clamp(
        i64::from(VALUE_TB_LOSS_IN_MAX_PLY) + 1,
        i64::from(VALUE_TB_WIN_IN_MAX_PLY) - 1,
    ) as Value
}

// Gated on `sota` as a whole: both items inside are, so at default features
// the module would be empty and `use super::*` an unused import.
#[cfg(all(test, feature = "sota"))]
mod side_net_flag_tests {
    use super::*;
    use std::sync::LazyLock;

    /// POISON fixtures: structurally valid networks with EMPTY weight tables.
    ///
    /// `SimpleHalfkaContext::new` / `ProtectedHalfkaContext::new` only store
    /// the pointer and allocate a stack, so nothing here is ever indexed —
    /// which is the whole point: these exist to exercise the INSTALLED branch
    /// of the two setters, which no other test in the repo reaches.
    ///
    /// **Never `refresh_root` or `evaluate` through one** (D33 reviewer C,
    /// F2): `SimpleHalfkaContext::refresh` does
    /// `acc.copy_from_slice(&self.net.ft_bias)` into a `[i16; SIMPLE_ACC]`,
    /// which PANICS on a length mismatch — and a dimensionally-valid zero net
    /// would be 25 MB, so shrinking is not the fix. Test the flag, not the
    /// forward pass.
    ///
    /// `LazyLock` rather than `Box::leak` so the pointer is STABLE across
    /// calls (D33 reviewer C, F1) — that is what makes the second install of
    /// the "same" net actually take `set_simple_halfka_net`'s `reuse` path.
    static POISON_SIMPLE: LazyLock<nnue::simple_halfka::SimpleHalfkaNetwork> =
        LazyLock::new(|| nnue::simple_halfka::SimpleHalfkaNetwork {
            halfka_weights: Box::new([]),
            ft_bias: Box::new([]),
            out_weights: Box::new([]),
            out_bias: 0,
        });

    /// The `protected_halfka` twin. Needed because the two setters maintain
    /// `side_nets` INDEPENDENTLY, and `Worker::refresh_eval_net` calls the
    /// protected one LAST — so for a protected-only configuration it is the
    /// protected setter's refresh that decides the flag (D33 reviewer A).
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

    // `side_nets` is a CACHE of `simple_halfka.is_some() ||
    // protected_halfka.is_some()`, and `push_sota_dirty` skips both pushes when
    // it is false — so a stale `false` is a silent wrong-eval bug, not a crash
    // (perf study 16 §6). Drive it through the only two writers.
    #[test]
    fn side_net_flag_tracks_the_optional_contexts() {
        let mut ctx = EvalContext::new();
        assert!(!ctx.side_nets, "a fresh context has no side nets");

        ctx.set_simple_halfka_net(Some(&POISON_SIMPLE));
        assert!(ctx.side_nets, "installing a side net must arm the flag");

        // A second install of the same net takes the `reuse` path, which does
        // not touch the field — the flag must survive it.
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

    // The PROTECTED leg, which the test above cannot reach: it only ever hands
    // `set_protected_halfka_net` a `None`, and with `simple` armed the expected
    // flag is `true` either way — so deleting that setter's refresh passes it.
    // That is the dangerous direction: `Worker::refresh_eval_net` installs the
    // protected net LAST, so on a protected-only configuration a stale `false`
    // makes `push_sota_dirty` skip the protected push for a whole search — a
    // wrong eval with no crash (D33 reviewer A).
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
