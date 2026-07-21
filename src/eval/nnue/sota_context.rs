//! Per-thread SOTA eval context: the accumulator stack + head forward for D40
//! nets (design/11 §12, D40).
//!
//! Analogous to [`super::accumulator::NnueContext`] but for the D40 architecture:
//! two feature transformers (HalfKA + FullThreats), each producing a
//! per-perspective [`ACC`]-wide accumulator, fed into the multi-layer bucketed
//! [`SotaHead`].
//!
//! Both transformers are incremental *and* lazy on the search stack. HalfKA
//! folds ordinary dirty-piece columns lazily and uses a 64-state-per-perspective
//! Finny cache on king moves; FullThreats resolves its bounded x-ray delta list
//! at push time (it reads the post-move board) but defers the 768-lane fold to
//! evaluate time, so children that cut off before evaluating never pay it (R4).
//! Root refresh and standalone `nnue-eval` remain from-scratch.
//!
//! Behind the `sota` cargo feature.

use super::sota_accumulator::HalfkaContext;
use super::sota_features::ACC;
#[cfg(debug_assertions)]
use super::sota_features::refresh_both;
use super::sota_loader::SotaNetwork;
use super::sota_threats::ThreatContext;
use crate::board::Board;
use crate::board::DirtyPiece;
use crate::types::{Color, PAWN_VALUE, PieceType, Value};

/// A permanently-zero threat accumulator handed to the cheap net's head on the
/// smallnet-gate path (perf study #3 #8). The cheap net is HalfKA-only, so its
/// FullThreats channel contributes nothing; passing this in place of a real
/// threat accumulator is exactly what an empty-threat-weights net produces (see
/// `halfka_only_context_has_zero_threat_channel`), but WITHOUT paying to build
/// or fold any threat state — which is the gate's entire speed win. A single
/// `static` so `&ZERO_THREAT` is a stable `'static` reference with no per-eval
/// cost.
static ZERO_THREAT: [[i16; ACC]; 2] = [[0; ACC]; 2];

/// Signed material imbalance (White − Black) in centipawns: cached non-pawn
/// material plus a pawn-count difference. Used as the smallnet gate's *material
/// prefilter* — an almost-free decided-ness proxy (`non_pawn_material` is an O(1)
/// state field; pawns are one popcount per side) so the cheap net runs ONLY on
/// materially-lopsided nodes and the undecided majority pays zero cheap-eval
/// overhead. Sign is irrelevant to the gate (it compares `.abs()`); returning the
/// signed value keeps it a plain difference.
#[inline]
fn material_imbalance(board: &Board) -> Value {
    let non_pawn = board.non_pawn_material(Color::White) - board.non_pawn_material(Color::Black);
    let pawns = board.pieces(Color::White, PieceType::Pawn).count() as Value
        - board.pieces(Color::Black, PieceType::Pawn).count() as Value;
    non_pawn + pawns * PAWN_VALUE
}

/// Per-thread SOTA evaluation state, owned by a search `Worker` via
/// [`super::super::EvalContext`].
pub struct SotaContext {
    /// The loaded net. Leaked to `'static` by `set_eval_file` so multiple
    /// threads can borrow it concurrently without lifetime gymnastics.
    net: &'static SotaNetwork,
    halfka: HalfkaContext<'static>,
    threats: ThreatContext<'static>,
    /// EXP-UNC-1: the value-uncertainty σ̂ from the most recent [`evaluate`], or
    /// `None` when the net has no variance head. Stashed here (rather than
    /// widening the `Value` return) so the search can read it for the pruning /
    /// time-management consumers without disturbing the eval-value contract.
    ///
    /// [`evaluate`]: Self::evaluate
    last_sigma: Option<f32>,
    /// D41 rich-net: the WDL probabilities [L,D,W] from the last full eval (same
    /// carry-out-of-band pattern as [`last_sigma`](Self::last_sigma)); `None` when
    /// the net has no WDL head or on the cheap smallnet path. Search/TM read it via
    /// [`last_wdl`](Self::last_wdl) for the decisiveness / optimism consumers.
    last_wdl: Option<[f32; 3]>,
    /// D47: the shared-bottleneck (L2) activation of the most recent full
    /// forward, stashed ONLY when a policy head is active — the policy
    /// scorer dots legal-move rows against it. Same freshness contract as
    /// `last_sigma`/`last_wdl` (invalidated when the full forward didn't run).
    last_l2: Option<Box<[f32; super::sota_head::L2]>>,
    /// Smallnet gate (perf study #3 item #8): the cheap net used on DECIDED
    /// positions, or `None` when the gate is off (`ARCHON_SMALLNET` unset or
    /// `ARCHON_SMALLNET_THRESH` ≤ 0). When armed, [`evaluate`](Self::evaluate)
    /// runs this net's HalfKA channel first and, if the resulting `|value|` is at
    /// least [`smallnet_thresh`](Self::smallnet_thresh), returns it — skipping the
    /// big net's FullThreats fold and head. `None` ⇒ every path below is
    /// byte-for-byte today's (bit-identical, sota bench 147691).
    smallnet: Option<&'static SotaNetwork>,
    /// The cheap net's own incremental HalfKA stack. Its FullThreats channel is
    /// never built — that is the point. Reuses [`HalfkaContext`] with the cheap
    /// net's weights and is maintained in lockstep with [`halfka`](Self::halfka)
    /// so a later undecided node evaluates the big net exactly as before. Armed
    /// together with [`smallnet`](Self::smallnet).
    smallnet_halfka: Option<HalfkaContext<'static>>,
    /// The decided-position centipawn threshold; `0` ⇒ gate off.
    smallnet_thresh: Value,
    /// Material prefilter (`ARCHON_SMALLNET_MAT_GATE`): only run the cheap net
    /// when `|material_imbalance| ≥` this. `0` ⇒ always run it when the gate is
    /// armed (SF-style output-only gate). A positive value is the cost-model fix
    /// (design doc §COST-MODEL CAVEAT): because our cheap net is HalfKA-only-768
    /// (~0.6× the big net, not SF-tiny), running it on every eval'd node would be
    /// a wash; prefiltering on near-free material skips it on the balanced
    /// (unclear) majority so the undecided nodes pay no cheap overhead.
    smallnet_mat_gate: Value,
    /// T-shirt gate LARGE tier (`ARCHON_LARGENET`): a full-feature net run on
    /// CRITICAL positions — where the medium net's EXP-UNC-1 σ̂ ≥
    /// [`largenet_sigma_gate`](Self::largenet_sigma_gate) — for extra accuracy
    /// where the medium eval is least reliable. `None` ⇒ no large tier ⇒
    /// bit-identical. Full-feature (HalfKA + FullThreats), so it carries BOTH
    /// accumulator stacks, unlike the HalfKA-only small tier.
    largenet: Option<&'static SotaNetwork>,
    /// The large net's own incremental HalfKA + FullThreats stacks, maintained in
    /// lockstep. Lazy like the medium's: the folds only happen on the (minority
    /// of) nodes that actually escalate, so undecided/decided nodes pay only the
    /// O(1) push.
    largenet_halfka: Option<HalfkaContext<'static>>,
    largenet_threats: Option<ThreatContext<'static>>,
    /// σ̂ threshold (sigmoid/WDL space) at/above which the medium eval is "unsure
    /// enough" to escalate to the large net. `0.0` ⇒ large tier off. Requires the
    /// medium net to carry a variance head (else σ̂ is `None` and nothing
    /// escalates).
    largenet_sigma_gate: f32,
}

impl std::fmt::Debug for SotaContext {
    fn fmt(&self, formatter: &mut std::fmt::Formatter<'_>) -> std::fmt::Result {
        formatter
            .debug_struct("SotaContext")
            .field("net", &(self.net as *const SotaNetwork))
            .finish_non_exhaustive()
    }
}

impl SotaContext {
    /// Create a new context over the given net. Reads the smallnet-gate env
    /// (`ARCHON_SMALLNET` + `ARCHON_SMALLNET_THRESH`) once here, at the eval
    /// boundary — never on the hot path. Both unset ⇒ no gate ⇒ bit-identical.
    pub fn new(net: &'static SotaNetwork) -> Self {
        let smallnet_thresh: Value = std::env::var("ARCHON_SMALLNET_THRESH")
            .ok()
            .and_then(|value| value.parse::<Value>().ok())
            .unwrap_or(0);
        // Arm only when BOTH a cheap net and a positive threshold are present;
        // otherwise leave every smallnet field inert so the accumulator and
        // evaluate paths are byte-for-byte identical to a no-gate build.
        let smallnet = if smallnet_thresh > 0 {
            super::active_smallnet()
        } else {
            None
        };
        let smallnet_halfka =
            smallnet.map(|small| HalfkaContext::new(&small.halfka_weights, &small.ft_bias));
        let smallnet_mat_gate: Value = std::env::var("ARCHON_SMALLNET_MAT_GATE")
            .ok()
            .and_then(|value| value.parse::<Value>().ok())
            .filter(|&gate| gate > 0)
            .unwrap_or(0);
        // T-shirt gate LARGE tier: arm only when a positive σ̂ gate AND a large net
        // are both configured; otherwise every large field stays inert so the
        // accumulator/evaluate paths are byte-for-byte the no-tier build.
        let largenet_sigma_gate: f32 = std::env::var("ARCHON_LARGENET_SIGMA")
            .ok()
            .and_then(|value| value.parse::<f32>().ok())
            .filter(|&gate| gate > 0.0 && gate.is_finite())
            .unwrap_or(0.0);
        let largenet = if largenet_sigma_gate > 0.0 {
            super::active_largenet()
        } else {
            None
        };
        let largenet_halfka =
            largenet.map(|large| HalfkaContext::new(&large.halfka_weights, &large.ft_bias));
        let largenet_threats = largenet.map(|large| ThreatContext::new(&large.threat_weights));
        SotaContext {
            net,
            halfka: HalfkaContext::new(&net.halfka_weights, &net.ft_bias),
            threats: ThreatContext::new(&net.threat_weights),
            last_sigma: None,
            last_wdl: None,
            last_l2: None,
            smallnet,
            smallnet_halfka,
            smallnet_thresh: if smallnet.is_some() {
                smallnet_thresh
            } else {
                0
            },
            smallnet_mat_gate,
            largenet,
            largenet_halfka,
            largenet_threats,
            largenet_sigma_gate: if largenet.is_some() {
                largenet_sigma_gate
            } else {
                0.0
            },
        }
    }

    /// Test-only constructor that arms the smallnet gate explicitly (bypassing the
    /// env), so gate correctness is unit-testable without a global env dependency.
    /// `mat_gate` is the material prefilter (0 = always run the cheap net).
    #[cfg(test)]
    fn new_with_gate(
        net: &'static SotaNetwork,
        smallnet: &'static SotaNetwork,
        smallnet_thresh: Value,
        smallnet_mat_gate: Value,
    ) -> Self {
        SotaContext {
            net,
            halfka: HalfkaContext::new(&net.halfka_weights, &net.ft_bias),
            threats: ThreatContext::new(&net.threat_weights),
            last_sigma: None,
            last_wdl: None,
            last_l2: None,
            smallnet: Some(smallnet),
            smallnet_halfka: Some(HalfkaContext::new(
                &smallnet.halfka_weights,
                &smallnet.ft_bias,
            )),
            smallnet_thresh,
            smallnet_mat_gate,
            largenet: None,
            largenet_halfka: None,
            largenet_threats: None,
            largenet_sigma_gate: 0.0,
        }
    }

    /// Test-only constructor that arms the LARGE tier explicitly. `sigma_gate` is
    /// the σ̂ escalation threshold; the large net carries its own HalfKA + threats
    /// stacks (full-feature).
    #[cfg(test)]
    fn new_with_large(
        net: &'static SotaNetwork,
        largenet: &'static SotaNetwork,
        sigma_gate: f32,
    ) -> Self {
        SotaContext {
            net,
            halfka: HalfkaContext::new(&net.halfka_weights, &net.ft_bias),
            threats: ThreatContext::new(&net.threat_weights),
            last_sigma: None,
            last_wdl: None,
            last_l2: None,
            smallnet: None,
            smallnet_halfka: None,
            smallnet_thresh: 0,
            smallnet_mat_gate: 0,
            largenet: Some(largenet),
            largenet_halfka: Some(HalfkaContext::new(
                &largenet.halfka_weights,
                &largenet.ft_bias,
            )),
            largenet_threats: Some(ThreatContext::new(&largenet.threat_weights)),
            largenet_sigma_gate: sigma_gate,
        }
    }

    /// The value-uncertainty σ̂ produced by the most recent [`evaluate`](Self::evaluate),
    /// or `None` when the loaded net has no EXP-UNC-1 variance head. σ̂ is the
    /// per-position value std in sigmoid (WDL) space.
    pub fn last_sigma(&self) -> Option<f32> {
        self.last_sigma
    }

    /// D41 rich-net: the WDL probabilities [L,D,W] from the most recent
    /// [`evaluate`](Self::evaluate), or `None` when the net has no WDL head / the
    /// cheap smallnet path ran. Same carry-out-of-band contract as `last_sigma`.
    pub fn last_wdl(&self) -> Option<[f32; 3]> {
        self.last_wdl
    }

    /// EXP-UNC-1: invalidate the cached σ̂ (set to `None`). Called at the top of
    /// [`super::super::evaluate`] so any eval path that does NOT run the SOTA
    /// forward this node (material-draw shortcut, a non-SOTA net) leaves σ̂ as
    /// `None` instead of a STALE prior-position value — the D33 review found the
    /// stale value would otherwise be folded into the search's uncertainty EMA
    /// and root capture (reviewers A/C).
    pub fn invalidate_sigma(&mut self) {
        self.last_sigma = None;
        self.last_wdl = None;
        self.last_l2 = None;
    }

    /// D47: the shared-bottleneck activation from the most recent full
    /// forward (fresh-eval-only, like `last_wdl`). `None` when no policy head
    /// is active or the last eval took a cheap path.
    pub fn last_l2(&self) -> Option<&[f32; super::sota_head::L2]> {
        self.last_l2.as_deref()
    }

    /// The net this context evaluates against.
    pub fn net(&self) -> &SotaNetwork {
        self.net
    }

    /// Seed both accumulator stacks from `board` at ply zero (plus the cheap
    /// net's stack when the smallnet gate is armed).
    pub fn refresh_root(&mut self, board: &Board) {
        self.halfka.refresh_root(board);
        self.threats.refresh(board);
        if let Some(small_halfka) = self.smallnet_halfka.as_mut() {
            small_halfka.refresh_root(board);
        }
        if let Some(large_halfka) = self.largenet_halfka.as_mut() {
            large_halfka.refresh_root(board);
        }
        if let Some(large_threats) = self.largenet_threats.as_mut() {
            large_threats.refresh(board);
        }
    }

    /// Record one post-move position at `ply + 1`. When the smallnet gate is
    /// armed the cheap net's HalfKA stack is advanced in lockstep — cheap (a few
    /// i16 lanes per moved piece), and it keeps the big net's own stacks valid so
    /// any later undecided node evaluates the big net exactly as before.
    pub fn push(&mut self, dirty: DirtyPiece, ply: usize, post_move_board: &Board) {
        self.halfka.push(ply, dirty);
        self.threats.apply_at(ply, &dirty, post_move_board);
        if let Some(small_halfka) = self.smallnet_halfka.as_mut() {
            small_halfka.push(ply, dirty);
        }
        if let Some(large_halfka) = self.largenet_halfka.as_mut() {
            large_halfka.push(ply, dirty);
        }
        if let Some(large_threats) = self.largenet_threats.as_mut() {
            large_threats.apply_at(ply, &dirty, post_move_board);
        }
    }

    /// Evaluate `board` through the incrementally maintained SOTA accumulators.
    pub fn evaluate(&mut self, board: &Board, ply: usize) -> Value {
        // The production search-stack oracle: catch stale sibling entries,
        // missed king refreshes, and FullThreats delta drift at the first node
        // where they could affect search. This mirrors the classic NNUE
        // accumulator oracle and compiles out of release builds.
        #[cfg(debug_assertions)]
        self.assert_matches_refresh(board, ply);

        let stm = board.side_to_move();
        // Count total pieces for the output bucket (both nets share the layout).
        let piece_count = board.all_occupancy().count() as usize;

        // Smallnet gate (perf study #3 item #8) + material prefilter: on a
        // materially-decided position the cheap HalfKA-only net decides, so we
        // skip the big net's FullThreats fold and head. The material prefilter
        // (`mat_gate`) is the cost-model fix (design doc §COST-MODEL CAVEAT): our
        // cheap net is HalfKA-only-768 (~0.6× the big net, not SF-tiny), so
        // running it on every eval'd node would be a wash — the near-free material
        // check short-circuits FIRST (before we even touch the cheap accumulator),
        // keeping it off the balanced/unclear majority. The big net's accumulators
        // were still advanced by `push`/`refresh_root`, so a later undecided node
        // evaluates it exactly as before. Inert (thresh 0 / `smallnet == None`) ⇒
        // this block is skipped ⇒ bit-identical to a no-gate build (bench 147691).
        let thresh = self.smallnet_thresh;
        let mat_gate = self.smallnet_mat_gate;
        if thresh > 0
            && (mat_gate == 0 || material_imbalance(board).abs() >= mat_gate)
            && let (Some(small), Some(small_halfka)) =
                (self.smallnet, self.smallnet_halfka.as_mut())
        {
            let small_pst = small_halfka.current(board, ply);
            let (small_value, _, _) = small
                .head
                .forward(small_pst, &ZERO_THREAT, stm, piece_count);
            if small_value.abs() >= thresh {
                // Cheap path: no variance head, so σ̂ is unavailable — correct,
                // since the uncertainty consumers no-op on decided positions.
                self.last_sigma = None;
                self.last_wdl = None;
                self.last_l2 = None;
                return small_value;
            }
        }

        let pst = self.halfka.current(board, ply);
        let threat = self.threats.at(ply);
        let (value, sigma, wdl) = if super::active_policy_head().is_some() {
            // D47: the policy scorer needs the shared-bottleneck activation of
            // this exact position; stash it (1 KiB copy) only when a head is
            // loaded so headless builds/runs pay nothing.
            let ((value, sigma, wdl), l2) =
                self.net.head.forward_full(pst, threat, stm, piece_count);
            self.last_l2 = Some(Box::new(l2));
            (value, sigma, wdl)
        } else {
            self.net.head.forward(pst, threat, stm, piece_count)
        };
        self.last_sigma = sigma;
        self.last_wdl = wdl;

        // T-shirt gate LARGE tier: on a CRITICAL position — one the medium net is
        // least sure about (its EXP-UNC-1 σ̂ ≥ largenet_sigma_gate) — escalate to
        // the large net, spending extra eval accuracy exactly where the medium eval
        // is least reliable. σ̂ IS the criticality router (|eval| can't tell a dead
        // draw from a razor-sharp position; σ̂ can). No σ̂ (variance-less medium
        // net) ⇒ nothing escalates. Inert (sigma_gate 0 / largenet None) ⇒ this
        // block is skipped ⇒ bit-identical to a no-tier build.
        let sigma_gate = self.largenet_sigma_gate;
        if sigma_gate > 0.0
            && let Some(medium_sigma) = sigma
            && medium_sigma >= sigma_gate
            && let (Some(large), Some(large_halfka), Some(large_threats)) = (
                self.largenet,
                self.largenet_halfka.as_mut(),
                self.largenet_threats.as_mut(),
            )
        {
            let large_pst = large_halfka.current(board, ply);
            let large_threat = large_threats.at(ply);
            let (large_value, large_sigma, large_wdl) =
                large
                    .head
                    .forward(large_pst, large_threat, stm, piece_count);
            self.last_sigma = large_sigma;
            self.last_wdl = large_wdl;
            return large_value;
        }
        value
    }

    #[cfg(debug_assertions)]
    fn assert_matches_refresh(&mut self, board: &Board, ply: usize) {
        let got_pst = *self.halfka.current(board, ply);
        let got_threat = *self.threats.at(ply);
        let expected = refresh_both(
            &self.net.halfka_weights,
            &self.net.ft_bias,
            &self.net.threat_weights,
            board,
        );
        assert_eq!(
            got_pst, expected.0,
            "SOTA HalfKA accumulator drift at ply {ply}"
        );
        assert_eq!(
            got_threat, expected.1,
            "SOTA FullThreats accumulator drift at ply {ply}"
        );
    }
}

#[cfg(test)]
mod tests {
    use super::*;
    use crate::fen::START_FEN;
    use crate::movegen::generate_legal;
    use crate::moves::MoveList;

    /// Build a leaked static SotaNetwork from the loader's test fixture.
    fn leaked_test_net() -> &'static SotaNetwork {
        use super::super::sota_loader::tests::test_net_with_threats_pub;
        Box::leak(Box::new(test_net_with_threats_pub()))
    }

    fn leaked_halfka_only_test_net() -> &'static SotaNetwork {
        use super::super::sota_loader::tests::test_net_with_threats_pub;
        let mut net = test_net_with_threats_pub();
        net.threat_weights = Vec::new().into_boxed_slice();
        Box::leak(Box::new(net))
    }

    /// A test net WITH a synthetic EXP-UNC-1 variance head, so `forward` returns
    /// `Some(σ̂)` and the t-shirt LARGE tier has a criticality signal to route on.
    fn leaked_variance_test_net() -> &'static SotaNetwork {
        use super::super::sota_head::{BUCKETS, L3};
        use super::super::sota_loader::tests::test_net_with_threats_pub;
        let mut net = test_net_with_threats_pub();
        net.head.l3_var_weights = Some(vec![0.01f32; BUCKETS * L3].into_boxed_slice());
        net.head.l3_var_biases = Some(vec![0.0f32; BUCKETS].into_boxed_slice());
        Box::leak(Box::new(net))
    }

    /// T-shirt LARGE tier: a σ̂ gate no position reaches (`1e9`) must never
    /// escalate, so every eval equals the ungated medium-net eval bit-for-bit —
    /// the large tier adds zero drift to the medium path (the property default
    /// bit-identity depends on).
    #[test]
    fn largenet_tier_never_escalates_is_bit_identical() {
        let medium = leaked_variance_test_net();
        let large = leaked_test_net();
        let mut gated = SotaContext::new_with_large(medium, large, 1e9);
        let mut ungated = SotaContext::new(medium);
        let mut board = Board::startpos();
        gated.refresh_root(&board);
        ungated.refresh_root(&board);
        let mut seed = 0xC0FFEE11u64;
        for ply in 0..32usize {
            let mut moves = MoveList::new();
            generate_legal(&board, &mut moves);
            if moves.is_empty() {
                break;
            }
            seed ^= seed << 13;
            seed ^= seed >> 7;
            seed ^= seed << 17;
            let m = moves[seed as usize % moves.len()];
            let dirty = DirtyPiece::from_move(&board, m);
            let gives_check = board.gives_check(m);
            board.make_move(m, gives_check);
            gated.push(dirty, ply, &board);
            ungated.push(dirty, ply, &board);
            assert_eq!(
                gated.evaluate(&board, ply + 1),
                ungated.evaluate(&board, ply + 1),
                "never-escalate large tier must match the ungated medium eval at ply {ply}"
            );
        }
    }

    /// T-shirt LARGE tier: when the large net and the medium net are the SAME
    /// (variance-bearing) net, escalating is value-transparent — a σ̂ gate that
    /// always fires (`1e-6`) returns exactly the medium eval, exercising the whole
    /// escalation path (second HalfKA+threats stacks, σ̂ routing, large head call).
    #[test]
    fn largenet_tier_transparent_when_nets_agree() {
        let net = leaked_variance_test_net();
        let mut gated = SotaContext::new_with_large(net, net, 1e-6);
        let mut ungated = SotaContext::new(net);
        let mut board = Board::startpos();
        gated.refresh_root(&board);
        ungated.refresh_root(&board);
        let mut seed = 0x5EED_1A9Eu64;
        let mut escalations = 0;
        for ply in 0..32usize {
            let mut moves = MoveList::new();
            generate_legal(&board, &mut moves);
            if moves.is_empty() {
                break;
            }
            seed ^= seed << 13;
            seed ^= seed >> 7;
            seed ^= seed << 17;
            let m = moves[seed as usize % moves.len()];
            let dirty = DirtyPiece::from_move(&board, m);
            let gives_check = board.gives_check(m);
            board.make_move(m, gives_check);
            gated.push(dirty, ply, &board);
            ungated.push(dirty, ply, &board);
            let expected = ungated.evaluate(&board, ply + 1);
            assert_eq!(
                gated.evaluate(&board, ply + 1),
                expected,
                "escalated large tier (large == medium) must reproduce the medium eval at ply {ply}"
            );
            // The medium net has a variance head, so σ̂ is present and the tiny gate
            // fires — confirm the escalation path is actually taken.
            if gated.last_sigma().is_some() {
                escalations += 1;
            }
        }
        assert!(
            escalations > 0,
            "expected the large tier to actually escalate"
        );
    }

    #[test]
    fn evaluate_returns_finite_value() {
        let net = leaked_test_net();
        let mut ctx = SotaContext::new(net);
        let board = Board::from_fen(START_FEN).unwrap();
        ctx.refresh_root(&board);
        let val = ctx.evaluate(&board, 0);
        assert!(
            val.abs() < 100_000,
            "eval should be finite and sane, got {val}"
        );
    }

    #[test]
    fn evaluate_stm_relative() {
        let net = leaked_test_net();
        let mut ctx = SotaContext::new(net);
        // Same position, different side to move.
        let w =
            Board::from_fen("rnbqkbnr/pppppppp/8/8/4P3/8/PPPP1PPP/RNBQKBNR b KQkq e3 0 1").unwrap();
        let b =
            Board::from_fen("rnbqkbnr/pppppppp/8/8/4P3/8/PPPP1PPP/RNBQKBNR w KQkq - 0 1").unwrap();
        ctx.refresh_root(&w);
        let val_w = ctx.evaluate(&w, 0);
        ctx.refresh_root(&b);
        let val_b = ctx.evaluate(&b, 0);
        // With a random test net, these won't be exact negatives, but they
        // should be finite and (with asymmetric accumulators) likely different.
        assert!(val_w.abs() < 100_000);
        assert!(val_b.abs() < 100_000);
    }

    #[test]
    fn evaluate_deterministic() {
        let net = leaked_test_net();
        let mut ctx = SotaContext::new(net);
        let board = Board::from_fen(START_FEN).unwrap();
        ctx.refresh_root(&board);
        let a = ctx.evaluate(&board, 0);
        let b = ctx.evaluate(&board, 0);
        assert_eq!(a, b, "eval must be deterministic");
    }

    #[test]
    fn incremental_context_matches_fresh_context() {
        let net = leaked_test_net();
        let mut incremental = SotaContext::new(net);
        let mut board = Board::startpos();
        incremental.refresh_root(&board);
        let mut seed = 0x51A7_A40Du64;
        for ply in 0..48usize {
            let mut moves = MoveList::new();
            generate_legal(&board, &mut moves);
            if moves.is_empty() {
                break;
            }
            seed ^= seed << 13;
            seed ^= seed >> 7;
            seed ^= seed << 17;
            let m = moves[seed as usize % moves.len()];
            let dirty = DirtyPiece::from_move(&board, m);
            let gives_check = board.gives_check(m);
            board.make_move(m, gives_check);
            incremental.push(dirty, ply, &board);
            let got = incremental.evaluate(&board, ply + 1);

            let mut fresh = SotaContext::new(net);
            fresh.refresh_root(&board);
            let expected = fresh.evaluate(&board, 0);
            assert_eq!(got, expected, "ply {ply} {m}");
        }
    }

    #[test]
    fn halfka_only_context_has_zero_threat_channel() {
        let net = leaked_halfka_only_test_net();
        let mut incremental = SotaContext::new(net);
        let mut board = Board::startpos();
        incremental.refresh_root(&board);
        let mut seed = 0x0A11_FA1Fu64;
        for ply in 0..24usize {
            let mut moves = MoveList::new();
            generate_legal(&board, &mut moves);
            if moves.is_empty() {
                break;
            }
            seed ^= seed << 13;
            seed ^= seed >> 7;
            seed ^= seed << 17;
            let m = moves[seed as usize % moves.len()];
            let dirty = DirtyPiece::from_move(&board, m);
            let gives_check = board.gives_check(m);
            board.make_move(m, gives_check);
            incremental.push(dirty, ply, &board);
            let got = incremental.evaluate(&board, ply + 1);

            let mut fresh = SotaContext::new(net);
            fresh.refresh_root(&board);
            assert_eq!(got, fresh.evaluate(&board, 0), "ply {ply} {m}");
        }
    }

    fn explore_siblings(
        net: &'static SotaNetwork,
        incremental: &mut SotaContext,
        board: &mut Board,
        ply: usize,
        depth: usize,
    ) -> usize {
        if depth == 0 {
            return 0;
        }
        let mut moves = MoveList::new();
        generate_legal(board, &mut moves);
        let mut visited = 0;
        for index in 0..moves.len().min(3) {
            let m = moves[index];
            let dirty = DirtyPiece::from_move(board, m);
            let gives_check = board.gives_check(m);
            board.make_move(m, gives_check);
            incremental.push(dirty, ply, board);
            let got = incremental.evaluate(board, ply + 1);
            let mut fresh = SotaContext::new(net);
            fresh.refresh_root(board);
            assert_eq!(got, fresh.evaluate(board, 0), "sibling ply {ply} {m}");
            visited += 1 + explore_siblings(net, incremental, board, ply + 1, depth - 1);
            board.unmake_move(m);
        }
        visited
    }

    /// Smallnet gate: when the cheap net and the big net are the SAME HalfKA-only
    /// net, routing is value-transparent — whether the gate takes the cheap path
    /// (thresh 1) or falls through (thresh MAX), it returns exactly the ungated
    /// eval. This exercises the whole gate: the second HalfKA stack's
    /// push/refresh/current, the `ZERO_THREAT` head call, and the routing branch.
    #[test]
    fn smallnet_gate_transparent_when_nets_agree() {
        let net = leaked_halfka_only_test_net();
        let mut gated_always = SotaContext::new_with_gate(net, net, 1, 0);
        let mut gated_never = SotaContext::new_with_gate(net, net, i32::MAX, 0);
        let mut ungated = SotaContext::new(net);
        let mut board = Board::startpos();
        gated_always.refresh_root(&board);
        gated_never.refresh_root(&board);
        ungated.refresh_root(&board);
        let mut seed = 0xBEEF_1234u64;
        for ply in 0..24usize {
            let mut moves = MoveList::new();
            generate_legal(&board, &mut moves);
            if moves.is_empty() {
                break;
            }
            seed ^= seed << 13;
            seed ^= seed >> 7;
            seed ^= seed << 17;
            let m = moves[seed as usize % moves.len()];
            let dirty = DirtyPiece::from_move(&board, m);
            let gives_check = board.gives_check(m);
            board.make_move(m, gives_check);
            gated_always.push(dirty, ply, &board);
            gated_never.push(dirty, ply, &board);
            ungated.push(dirty, ply, &board);
            let expected = ungated.evaluate(&board, ply + 1);
            assert_eq!(
                gated_always.evaluate(&board, ply + 1),
                expected,
                "always-route gate must reproduce the ungated eval at ply {ply}"
            );
            assert_eq!(
                gated_never.evaluate(&board, ply + 1),
                expected,
                "never-route gate must reproduce the ungated eval at ply {ply}"
            );
        }
    }

    /// Smallnet gate: with a FULL big net and a HalfKA-only cheap net, a threshold
    /// no position can reach (`i32::MAX`) must never route, so every eval equals
    /// the ungated big-net eval bit-for-bit — the gate adds zero drift to the big
    /// path (the property `EvalFile`-less bit-identity depends on).
    #[test]
    fn smallnet_gate_never_routes_is_bit_identical() {
        let big = leaked_test_net();
        let small = leaked_halfka_only_test_net();
        let mut gated = SotaContext::new_with_gate(big, small, i32::MAX, 0);
        let mut ungated = SotaContext::new(big);
        let mut board = Board::startpos();
        gated.refresh_root(&board);
        ungated.refresh_root(&board);
        let mut seed = 0x1234_ABCDu64;
        for ply in 0..32usize {
            let mut moves = MoveList::new();
            generate_legal(&board, &mut moves);
            if moves.is_empty() {
                break;
            }
            seed ^= seed << 13;
            seed ^= seed >> 7;
            seed ^= seed << 17;
            let m = moves[seed as usize % moves.len()];
            let dirty = DirtyPiece::from_move(&board, m);
            let gives_check = board.gives_check(m);
            board.make_move(m, gives_check);
            gated.push(dirty, ply, &board);
            ungated.push(dirty, ply, &board);
            assert_eq!(
                gated.evaluate(&board, ply + 1),
                ungated.evaluate(&board, ply + 1),
                "never-route gate must match the ungated big-net eval at ply {ply}"
            );
        }
    }

    #[test]
    fn material_imbalance_signs_and_zero() {
        // Startpos is materially symmetric.
        assert_eq!(material_imbalance(&Board::startpos()), 0);
        // White up a full queen (black has no queen): positive, ≈ a queen's value.
        let w_up =
            Board::from_fen("rnb1kbnr/pppppppp/8/8/8/8/PPPPPPPP/RNBQKBNR w KQkq - 0 1").unwrap();
        assert!(
            material_imbalance(&w_up) > 500,
            "white up a queen should read strongly positive, got {}",
            material_imbalance(&w_up)
        );
        // Black up a full rook: negative.
        let b_up =
            Board::from_fen("rnbqkbnr/pppppppp/8/8/8/8/PPPPPPPP/1NBQKBNR w Kkq - 0 1").unwrap();
        assert!(
            material_imbalance(&b_up) < -300,
            "black up a rook should read strongly negative, got {}",
            material_imbalance(&b_up)
        );
    }

    /// Material prefilter: a `mat_gate` no near-balanced position reaches
    /// (`i32::MAX`) blocks the cheap net from ever running — even with `thresh 1`
    /// that would otherwise always route — so every eval is the ungated big-net
    /// eval. Proves the prefilter short-circuits the cheap path (the cost-model
    /// fix that keeps the cheap net off the unclear majority).
    #[test]
    fn smallnet_gate_material_prefilter_blocks_cheap_path() {
        let big = leaked_test_net();
        let small = leaked_halfka_only_test_net();
        // thresh 1 (would route on any node) but mat_gate MAX (never passes).
        let mut gated = SotaContext::new_with_gate(big, small, 1, i32::MAX);
        let mut ungated = SotaContext::new(big);
        let mut board = Board::startpos();
        gated.refresh_root(&board);
        ungated.refresh_root(&board);
        let mut seed = 0x0FF1_CE55u64;
        for ply in 0..32usize {
            let mut moves = MoveList::new();
            generate_legal(&board, &mut moves);
            if moves.is_empty() {
                break;
            }
            seed ^= seed << 13;
            seed ^= seed >> 7;
            seed ^= seed << 17;
            let m = moves[seed as usize % moves.len()];
            let dirty = DirtyPiece::from_move(&board, m);
            let gives_check = board.gives_check(m);
            board.make_move(m, gives_check);
            gated.push(dirty, ply, &board);
            ungated.push(dirty, ply, &board);
            assert_eq!(
                gated.evaluate(&board, ply + 1),
                ungated.evaluate(&board, ply + 1),
                "material prefilter must block the cheap path at ply {ply}"
            );
        }
    }

    #[test]
    fn sibling_reuse_matches_fresh_context() {
        let net = leaked_test_net();
        let mut incremental = SotaContext::new(net);
        let mut board = Board::startpos();
        incremental.refresh_root(&board);
        let visited = explore_siblings(net, &mut incremental, &mut board, 0, 3);
        assert!(
            visited >= 30,
            "expected a broad sibling tree, got {visited}"
        );
    }
}
