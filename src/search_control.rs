//! Versioned, shadow-only contract for D41 search-consequence heads.
//!
//! This module contains no planner and has no search authority.  It accepts
//! only information available before an action is selected and turns any
//! malformed input, representation mismatch, or invalid prediction into an
//! explicit request to use the classical controller.

use crate::types::MAX_PLY;

/// Version of the engine-side inference contract, independent of model files.
pub const SEARCH_CONTROL_CONTRACT_VERSION: u16 = 1;

/// Conservative bound for logarithmic cost predictions.
///
/// `ln(nodes)` and `ln(cycles)` cannot approach this in a real search.  The
/// generous finite ceiling catches corrupt models without clipping valid data.
pub const MAX_LOG_COST: f32 = 64.0;

/// Identity of the latent representation consumed by a consequence head.
///
/// `latent_fingerprint` is assigned by the representation exporter and must
/// cover feature semantics, ordering, quantization/activation, and checkpoint.
/// It is not inferred from `latent_dim`: equal dimensions are not compatibility.
#[derive(Debug, Clone, Copy, PartialEq, Eq)]
pub struct RepresentationSpecV1 {
    pub contract_version: u16,
    pub latent_fingerprint: u64,
    pub latent_dim: u32,
}

impl RepresentationSpecV1 {
    pub const fn new(latent_fingerprint: u64, latent_dim: u32) -> Self {
        Self {
            contract_version: SEARCH_CONTROL_CONTRACT_VERSION,
            latent_fingerprint,
            latent_dim,
        }
    }

    const fn is_valid(self) -> bool {
        self.contract_version == SEARCH_CONTROL_CONTRACT_VERSION
            && self.latent_fingerprint != 0
            && self.latent_dim != 0
    }
}

/// Node classification known before the proposed move is searched.
#[repr(u8)]
#[derive(Debug, Clone, Copy, PartialEq, Eq)]
pub enum NodeKindV1 {
    Root = 0,
    Pv = 1,
    Cut = 2,
    All = 3,
    Qsearch = 4,
}

/// TT bound classification captured before action selection.
#[repr(u8)]
#[derive(Debug, Clone, Copy, PartialEq, Eq)]
pub enum TtBoundV1 {
    None = 0,
    Upper = 1,
    Lower = 2,
    Exact = 3,
}

/// Search information available before choosing a move's computation.
///
/// Outcome fields such as returned score, nodes, alpha raise, cutoff, and
/// re-search are intentionally absent.  Those belong to training observations,
/// not inference features.
#[derive(Debug, Clone, Copy, PartialEq, Eq)]
pub struct SearchContextV1 {
    pub remaining_depth: i16,
    pub ply: u16,
    pub alpha: i32,
    pub beta: i32,
    pub static_eval: i32,
    pub node_kind: NodeKindV1,
    pub in_check: bool,
    pub improving: bool,
    pub cut_node: bool,
    pub tt_hit: bool,
    pub tt_depth: i16,
    pub tt_bound: TtBoundV1,
    pub legal_move_count: u16,
    pub quiet_move_count: u16,
    pub move_index: u16,
    pub previous_pv_stability: u8,
}

impl SearchContextV1 {
    fn is_valid(&self) -> bool {
        self.ply as usize <= MAX_PLY
            && self.alpha < self.beta
            && self.quiet_move_count <= self.legal_move_count
            && self.move_index < self.legal_move_count
            && self.previous_pv_stability <= 100
            && (!self.tt_hit || self.tt_bound != TtBoundV1::None)
    }
}

/// Stable move flags; unknown bits make the input invalid rather than silently
/// changing semantics across contract versions.
pub mod move_flags {
    pub const CAPTURE: u16 = 1 << 0;
    pub const PROMOTION: u16 = 1 << 1;
    pub const GIVES_CHECK: u16 = 1 << 2;
    pub const QUIET: u16 = 1 << 3;
    pub const TT_MOVE: u16 = 1 << 4;
    pub const PREVIOUS_PV: u16 = 1 << 5;
    pub const KNOWN_MASK: u16 = CAPTURE | PROMOTION | GIVES_CHECK | QUIET | TT_MOVE | PREVIOUS_PV;
}

/// Legal-move features available before the move is searched.
#[derive(Debug, Clone, Copy, PartialEq, Eq)]
pub struct MoveFeaturesV1 {
    pub raw_move: u16,
    pub from: u8,
    pub to: u8,
    /// Engine piece index, `0..12`; `12` means absent for captured/promotion.
    pub moving_piece: u8,
    pub captured_piece: u8,
    pub promotion_piece: u8,
    pub flags: u16,
    pub classical_order_score: i32,
}

impl MoveFeaturesV1 {
    fn is_valid(&self) -> bool {
        self.raw_move != 0
            && self.from < 64
            && self.to < 64
            && self.from != self.to
            && self.moving_piece < 12
            && self.captured_piece <= 12
            && self.promotion_piece <= 12
            && self.flags & !move_flags::KNOWN_MASK == 0
            && (self.flags & move_flags::CAPTURE != 0) == (self.captured_piece < 12)
            && (self.flags & move_flags::PROMOTION != 0) == (self.promotion_piece < 12)
    }
}

/// Neural inference tier proposed for this action.
#[repr(u8)]
#[derive(Debug, Clone, Copy, PartialEq, Eq)]
pub enum PolicyTierV1 {
    None = 0,
    Cheap = 1,
    Full = 2,
}

/// One candidate computation.  This is an input, never a learned target class.
#[derive(Debug, Clone, Copy, PartialEq, Eq)]
pub struct ProposedActionV1 {
    pub reduction_plies: u8,
    pub extension_plies: u8,
    pub policy_tier: PolicyTierV1,
}

impl ProposedActionV1 {
    fn is_valid(&self) -> bool {
        // Initial D41 experiments enumerate small, bounded actions.  Widening
        // these ranges changes the contract and requires an explicit revision.
        self.reduction_plies <= 8 && self.extension_plies <= 2
    }
}

/// Consequences predicted for one `(position, context, move, action)` tuple.
#[derive(Debug, Clone, Copy, PartialEq)]
pub struct ActionPredictions {
    pub best_move_probability: f32,
    pub alpha_raise_probability: f32,
    pub beta_cutoff_probability: f32,
    /// The proposed scout/reduced result disagrees with the conservative
    /// reference before any verification or full-depth re-search repairs it.
    pub scout_bound_error_probability: f32,
    /// The result still disagrees after the configured verification and
    /// re-search behavior.  This is the planner's safety constraint; scout
    /// error additionally drives expected verification cost.
    pub residual_bound_error_probability: f32,
    pub research_probability: f32,
    pub expected_log_nodes: f32,
    pub expected_log_cycles: f32,
}

impl ActionPredictions {
    pub fn is_valid(&self) -> bool {
        let probabilities = [
            self.best_move_probability,
            self.alpha_raise_probability,
            self.beta_cutoff_probability,
            self.scout_bound_error_probability,
            self.residual_bound_error_probability,
            self.research_probability,
        ];
        probabilities
            .iter()
            .all(|&value| value.is_finite() && (0.0..=1.0).contains(&value))
            && [self.expected_log_nodes, self.expected_log_cycles]
                .iter()
                .all(|&value| value.is_finite() && (0.0..=MAX_LOG_COST).contains(&value))
    }
}

/// Model-side inference seam. Implementations must not allocate in this call.
///
/// Validation is centralized in [`predict_checked`]; an implementation may be
/// generated code and is never trusted to enforce the engine's safety rules.
pub trait SearchConsequenceHeadV1 {
    fn representation(&self) -> RepresentationSpecV1;

    fn predict_unchecked(
        &self,
        latent: &[f32],
        context: &SearchContextV1,
        mv: &MoveFeaturesV1,
        action: &ProposedActionV1,
    ) -> ActionPredictions;
}

/// Why inference was rejected and the classical controller must be used.
#[derive(Debug, Clone, Copy, PartialEq, Eq)]
pub enum FallbackReason {
    InvalidExpectedRepresentation,
    RepresentationMismatch,
    LatentDimensionMismatch,
    NonFiniteLatent,
    InvalidContext,
    InvalidMove,
    InvalidAction,
    InvalidPrediction,
}

/// Checked inference result. `ClassicalFallback` grants no neural authority.
#[derive(Debug, Clone, Copy, PartialEq)]
pub enum PredictionOutcome {
    Trusted(ActionPredictions),
    ClassicalFallback(FallbackReason),
}

/// Validate the complete inference boundary without allocating.
pub fn predict_checked<H: SearchConsequenceHeadV1>(
    head: &H,
    expected: RepresentationSpecV1,
    latent: &[f32],
    context: &SearchContextV1,
    mv: &MoveFeaturesV1,
    action: &ProposedActionV1,
) -> PredictionOutcome {
    if !expected.is_valid() {
        return PredictionOutcome::ClassicalFallback(FallbackReason::InvalidExpectedRepresentation);
    }
    if head.representation() != expected {
        return PredictionOutcome::ClassicalFallback(FallbackReason::RepresentationMismatch);
    }
    if latent.len() != expected.latent_dim as usize {
        return PredictionOutcome::ClassicalFallback(FallbackReason::LatentDimensionMismatch);
    }
    if latent.iter().any(|value| !value.is_finite()) {
        return PredictionOutcome::ClassicalFallback(FallbackReason::NonFiniteLatent);
    }
    if !context.is_valid() {
        return PredictionOutcome::ClassicalFallback(FallbackReason::InvalidContext);
    }
    if !mv.is_valid() {
        return PredictionOutcome::ClassicalFallback(FallbackReason::InvalidMove);
    }
    if !action.is_valid() {
        return PredictionOutcome::ClassicalFallback(FallbackReason::InvalidAction);
    }

    let prediction = head.predict_unchecked(latent, context, mv, action);
    if prediction.is_valid() {
        PredictionOutcome::Trusted(prediction)
    } else {
        PredictionOutcome::ClassicalFallback(FallbackReason::InvalidPrediction)
    }
}

/// Deterministic smoke-test implementation. It is intentionally conservative
/// and is not a trainable or deployable model.
#[derive(Debug, Clone, Copy)]
pub struct ConservativeDummyHeadV1 {
    representation: RepresentationSpecV1,
}

impl ConservativeDummyHeadV1 {
    pub const fn new(representation: RepresentationSpecV1) -> Self {
        Self { representation }
    }
}

impl SearchConsequenceHeadV1 for ConservativeDummyHeadV1 {
    fn representation(&self) -> RepresentationSpecV1 {
        self.representation
    }

    fn predict_unchecked(
        &self,
        _latent: &[f32],
        _context: &SearchContextV1,
        _mv: &MoveFeaturesV1,
        _action: &ProposedActionV1,
    ) -> ActionPredictions {
        ActionPredictions {
            best_move_probability: 0.0,
            alpha_raise_probability: 0.0,
            beta_cutoff_probability: 0.0,
            scout_bound_error_probability: 1.0,
            residual_bound_error_probability: 1.0,
            research_probability: 1.0,
            expected_log_nodes: MAX_LOG_COST,
            expected_log_cycles: MAX_LOG_COST,
        }
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    const REP: RepresentationSpecV1 = RepresentationSpecV1::new(0x4152_4348_4f4e_0001, 4);

    fn context() -> SearchContextV1 {
        SearchContextV1 {
            remaining_depth: 8,
            ply: 3,
            alpha: -20,
            beta: -19,
            static_eval: -10,
            node_kind: NodeKindV1::Cut,
            in_check: false,
            improving: true,
            cut_node: true,
            tt_hit: true,
            tt_depth: 5,
            tt_bound: TtBoundV1::Lower,
            legal_move_count: 20,
            quiet_move_count: 14,
            move_index: 4,
            previous_pv_stability: 75,
        }
    }

    fn move_features() -> MoveFeaturesV1 {
        MoveFeaturesV1 {
            raw_move: 0x1234,
            from: 12,
            to: 28,
            moving_piece: 1,
            captured_piece: 12,
            promotion_piece: 12,
            flags: move_flags::QUIET,
            classical_order_score: 42,
        }
    }

    fn action() -> ProposedActionV1 {
        ProposedActionV1 {
            reduction_plies: 2,
            extension_plies: 0,
            policy_tier: PolicyTierV1::None,
        }
    }

    #[test]
    fn deterministic_head_returns_a_valid_conservative_prediction() {
        let outcome = predict_checked(
            &ConservativeDummyHeadV1::new(REP),
            REP,
            &[0.0, 0.25, -0.25, 1.0],
            &context(),
            &move_features(),
            &action(),
        );
        assert!(matches!(outcome, PredictionOutcome::Trusted(_)));
    }

    #[test]
    fn representation_mismatch_falls_back_without_calling_the_head() {
        struct PanicIfCalledHead;

        impl SearchConsequenceHeadV1 for PanicIfCalledHead {
            fn representation(&self) -> RepresentationSpecV1 {
                RepresentationSpecV1::new(REP.latent_fingerprint ^ 1, REP.latent_dim)
            }

            fn predict_unchecked(
                &self,
                _latent: &[f32],
                _context: &SearchContextV1,
                _mv: &MoveFeaturesV1,
                _action: &ProposedActionV1,
            ) -> ActionPredictions {
                panic!("mismatched head must not run")
            }
        }

        let outcome = predict_checked(
            &PanicIfCalledHead,
            REP,
            &[0.0; 4],
            &context(),
            &move_features(),
            &action(),
        );
        assert_eq!(
            outcome,
            PredictionOutcome::ClassicalFallback(FallbackReason::RepresentationMismatch)
        );
    }

    #[test]
    fn out_of_range_probability_is_invalid() {
        let prediction = ActionPredictions {
            best_move_probability: 1.01,
            alpha_raise_probability: 0.0,
            beta_cutoff_probability: 0.0,
            scout_bound_error_probability: 0.0,
            residual_bound_error_probability: 0.0,
            research_probability: 0.0,
            expected_log_nodes: 1.0,
            expected_log_cycles: 1.0,
        };
        assert!(!prediction.is_valid());
    }

    struct NonFiniteHead;

    impl SearchConsequenceHeadV1 for NonFiniteHead {
        fn representation(&self) -> RepresentationSpecV1 {
            REP
        }

        fn predict_unchecked(
            &self,
            _latent: &[f32],
            _context: &SearchContextV1,
            _mv: &MoveFeaturesV1,
            _action: &ProposedActionV1,
        ) -> ActionPredictions {
            ActionPredictions {
                best_move_probability: f32::NAN,
                alpha_raise_probability: 0.0,
                beta_cutoff_probability: 0.0,
                scout_bound_error_probability: 0.0,
                residual_bound_error_probability: 0.0,
                research_probability: 0.0,
                expected_log_nodes: 1.0,
                expected_log_cycles: 1.0,
            }
        }
    }

    #[test]
    fn nonfinite_output_falls_back() {
        let outcome = predict_checked(
            &NonFiniteHead,
            REP,
            &[0.0; 4],
            &context(),
            &move_features(),
            &action(),
        );
        assert_eq!(
            outcome,
            PredictionOutcome::ClassicalFallback(FallbackReason::InvalidPrediction)
        );
    }

    #[test]
    fn nonfinite_latent_falls_back_before_model_inference() {
        let outcome = predict_checked(
            &ConservativeDummyHeadV1::new(REP),
            REP,
            &[0.0, f32::INFINITY, 0.0, 0.0],
            &context(),
            &move_features(),
            &action(),
        );
        assert_eq!(
            outcome,
            PredictionOutcome::ClassicalFallback(FallbackReason::NonFiniteLatent)
        );
    }
}
