# SFNNv16: PP_3Wide pawn-pair features (commit f4bcd404, 2026-07)

*Distilled from the commit diff/message (not yet a full source read). Ideas
only — GPLv3, never copy. Relevant because our D40/D47 feature set is the
SF HalfKAv2_hm + FullThreats(60720) design this commit just evolved.*

## What SF changed

- New input feature family **PP_3Wide** (`nnue/features/pp_3wide.*`):
  indexes **pairs of pawns on the same or adjacent files** (a 3-file-wide
  window). Pawn-structure motifs — phalanxes, chains, doubled/backward
  shapes — become *explicit input features* instead of something the net
  must reconstruct from independent piece placement.
- **FullThreats shrinks 60720 → 59808**: the pawn-pusher threat features
  were removed as superseded — pawn-pair geometry carries that signal
  better than threat-typed pawn interactions did.
- New `PawnPairBB` lookup + `pawn_pair_bb()` in bitboard.h (valid pairing
  squares precomputed).
- Net: nn-89cb98a217f7. Arch tag SFNNv16.

## The test-result shape is the interesting part

- **STC FAILED (LLR −2.94); LTC/VLTC/VVLTC all PASSED.** ~3.5% slowdown.
- SF accepted an eval-quality-for-speed trade that only pays at depth —
  the same trade shape as our D47 go-big bet. Even the ratchet takes
  richer-eval-slower-net deals when the LTC evidence is strong; STC-only
  reads would have killed this patch (and would kill ours).

## Implications for ARCHON (queue post-D47c, ideas-only)

1. **Feature-set upgrade candidate**: add a pawn-pair family to our
   feature stack and drop pawn-pusher threat rows (our FullThreats is the
   pre-v16 60720 layout — `sota_features.rs`, `d40_features.rs`). SF has
   effectively run the ablation for us at scale: pawn-pair > pawn-threat
   rows. Cost model matches "full threats is cheap": tiny active-count
   delta, table growth modest.
2. **Sizing note**: pawn pairs are ~O(few) active per position; the
   accumulator add is negligible — the cost is table width + one more
   feature-extraction pass (their ~3.5% total slowdown includes it).
3. **Sequencing**: NOT during the d47c straight shot. Slot into the
   arch-ablation suite after the sb240 verdict; a d47-arch retrain with
   pp-pairs replacing pawn-pusher threats is a clean single-variable
   experiment on infrastructure we already have (feature parity harness,
   gate_d47).
4. **Trainer side**: needs the mirrored feature in the bullet mapper
   (same anti-trap rule as policy_vocab: one shared implementation or a
   parity fixture pinning engine vs trainer indices).
5. **Meta-signal**: upstream is actively mining *structure-typed* input
   features (threats, now pawn pairs). Our multihead trunk consumes the
   same inputs — upstream feature R&D transfers to us nearly for free as
   long as we keep the parity harness discipline.
