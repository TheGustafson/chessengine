# Architecture ablation suite (runnable test matrix)

Concrete, runnable expansion of Axis B of `post-d40-research-direction.md`. Designed
with the user 2026-07-20. This is the *checklist* — execute a row at a time once the
data foundation is solid (arch is the POLISH layer; data comes first). Priors are
guesses; the point is to replace them with gated numbers.

## Methodology (every row)
- **One variable** changed from a fixed baseline (the current D40 arch: HalfKA+FullThreats
  → 2×768 accumulator → CReLU+pairwise-mul trunk → per-bucket `768→16→32→1` head, 8
  piece-count buckets).
- **Fixed-node screen → equal-time gate.** Fixed-node isolates the eval change; equal-time
  is the ship bar (head/accumulator changes cost NPS and must earn it back).
- Train on the **good streamed data** (UHO/farseer + fold-in), same seed as the control,
  gate vs a same-recipe control (like the `foldval` ctrl/mix pattern).
- **Expectation: arch is not where the big Elo is right now** (width already plateaued at
  2048 on old data — re-verify that; see T3). Batch the cheap rows; don't over-invest.

## Baseline dims (as built, verified)
`HL=768` (accumulator width, ×2 perspectives) · `L2=16, L3=32` (per-bucket head
`768→16→32→1`) · `NUM_BUCKETS=8` (piece-count `OUTPUT_BUCKETS_LAYOUT`) · HalfKA
`22528` dims / Threats `60720` dims · trunk `crelu().pairwise_mul()`, head `screlu()`.

---

## Tier 1 — cheap, runnable now (highest value/cost)

| id | test | configs | prior | gotcha |
|---|---|---|---|---|
| **T0** | **King-bucket factorization (#27)** | training-time weight-sharing across king-square-symmetric positions | **small win at ZERO deploy cost** — the safest lever in the suite; run FIRST | training-only; verify parity at deploy |
| **T1** | **Head bottleneck/shape** | `L2→L3` ∈ {16→32(ctrl), 32→32, 64→32, 32→16, 32→1(drop mid)} | wider bottleneck (16→32/64) wins a little — the 16 is too tight | gate **equal-time** (bigger head = small NPS cost) |
| **T2** | **Extra head layer** | `768→16→32→16→1` vs `768→16→32→1` | marginal-to-slightly-negative (NPS cost vs diminishing returns) | batch with T1 |
| **T3** | **Accumulator width re-sweep on NEW data** | HL ∈ {768(ctrl), 1024, 1536, 2048} | **the anchor for the "is capacity the bottleneck yet" question** — plateau may have moved on richer data | equal-time; wider = slower hot path |
| **T4** | **Trunk activation** | `crelu→pairwise-mul`(ctrl) vs `screlu` trunk vs `crelu`-only vs alt pairing | pairwise-mul likely stays best (field-converged) | quant range shifts with activation |

## Tier 2 — structural, still cheap-ish

| id | test | configs | prior | gotcha |
|---|---|---|---|---|
| **T5** | **Second routing axis (more experts)** | piece-count(8) × 2nd-heuristic. START at 8×queens=16, then stack | a 2nd axis *should* help — eval regime flips are real | sparse grid cells (e.g. 32-pieces ∧ no-queens) starve; merge or balance. Combine with T1 if it wins. |
| **T6** | **Feature-set contribution** | (a) HalfKA-only (drop Threats) — *the actionable one*; (b) Threats-only (drop HalfKA) | HalfKA is the workhorse; Threats-only likely hurts a lot; HalfKA-only measures Threats' marginal value | **keep accumulator FIXED** — don't confound "drop features" with "bigger accumulator" |
| **T6b** | **Pawn-pair features (SFNNv16 PP_3Wide adaptation)** — add a pawn-pair input family (pairs of pawns on same/adjacent files as explicit features) and DROP the pawn-pusher threat rows they supersede (SF: FullThreats 60720→59808) | one retrain of the current-best arch with pp-pairs swapped in, vs same-seed control | **small-positive at depth** — SF ran this ablation at fishtest scale (commit `f4bcd404`): STC FAIL, LTC/VLTC/VVLTC PASS, ~3.5% slowdown accepted. Depth-scaling eval gain = the D47 trade shape; our deep-TC gates should see it | ideas-only reimplementation (GPLv3 — never copy); needs the pp-pair mapper mirrored in the bullet trainer + a parity fixture pinning engine-vs-trainer indices (same anti-trap rule as policy_vocab); active-count delta is tiny so NPS cost ≈ table width only. Ref: `stockfish_reference/08-sfnnv16-pp3wide-pawn-pairs.md` |

**Routing methodology (user, 2026-07-22): queens-on/off is the canonical
FIRST ablation for ANY routing decision** — value buckets, policy
specialists (policy roadmap §2c), learned routers (T7). Binary, balanced,
cheap, orthogonal to piece count, and the hardest single-bit regime flip we
have. If a routing mechanism shows nothing on the queens axis, don't bother
with subtler axes — it's the canary.

### Routing heuristics for T5 (ranked — orthogonal to piece count, flips the eval regime, cheap, balanced)
1. **Queens on vs off** — TOP pick. Eval is a different function with queens (king-safety/tactics) vs without (positional/endgame). Binary, balanced, cheap, orthogonal to material count. → 8×2 = 16 buckets.
2. **King-safety regime** — same-side vs opposite-side castling (or castled/not). Attacking races vs maneuvering. ~4 levels → up to 32.
3. **Pawn-structure openness** — open vs closed (locked/tension pawn count). Piece-activity regime.
4. (specific) opposite-colored bishops present; passed pawns present.
- **Reaching 64:** hand-designed axes naturally give 16–32; **64 wants the learned router** (T7), not an 8-level hand axis.

## Tier 3 — bigger / structurally novel (do after data is rich + foundation solid)

| id | test | configs | prior | gotcha |
|---|---|---|---|---|
| **T7** | **Learned-router MoE** | replace hard piece-count route with a learned gate over the accumulator; 16→64 experts | the "let the net find better splits than piece count" upgrade; **the efficient way to add capacity for the rich-data net** (scale params, sparse activation) | router collapse / load-balance loss / non-diff top-k — training complexity |
| **T8** | **Multiple / split accumulators** | **(a) KING-SPLIT (lead): a BIG king-INDEPENDENT accumulator (piece-square/threats — never refreshes, so it can be wide) + a SMALL king-relative accumulator (cheap refresh ∵ small).** (b) cheap dual (one full + one factorized). (c) 4-way white/black-square split. | **the "bigger-net-on-CPU" lever** — king moves force a full HalfKA refresh (~10× a normal move, ~⅓–½ of avg update cost even *with* the Finny cache we already have); isolating the king-relative part into a SMALL accumulator frees NPS to make the big accumulator wide. Keeps king-relativity's *value*, drops its *cost*. | equal-time gate (the whole point is NPS→width); demoting HalfKA to a small side-accumulator may cost some eval — measure the trade. We ALREADY have the Finny refresh-cache (`sota_accumulator.rs`, 64 king keys), so (a) is the step *beyond* it. |
| **T9** | **Trunk-mixing for multi-target** *(GATED on multi-head machinery existing)* | split point ∈ {share-accumulator-only, +1st head layer, all-but-last, separate}; per output (eval/WDL/policy) | share accumulator always; split LATE for correlated (eval+WDL), separate head for policy | can only run once WDL/policy heads are wired (D41); this is the D41 tuning step |

## Dependency / run order
1. **T0 → T1+T2 → T3 → T4** (Tier-1 batch — cheap, now-ish; T0 first).
2. **T5 (start 16) → T6 → T6b** (Tier-2; T6b is high-priority within the tier —
   SF pre-validated it at scale, we only pay the adaptation cost).
3. **T7 → T8 → T9** (Tier-3; T9 gated on the multi-head rich-data net existing — the D41 arc).

**Freshness note (2026-07-22):** the baseline above predates D47. Once d47c
gates, the control for new rows becomes the **d47c trunk** (1280 acc / 128
bottleneck / threats / multihead), T9 is partially LIVE (policy head shipped;
split-point question remains for WDL/σ), and T3's width question folds into
the D47 shrink/optimize phase.

Governing caveat: **data first.** A bigger/fancier net with nothing rich to learn from just
overfits. The endgame is a differently-shaped net (T7/T8 + rich-data heads) that turns our
search data into something no other engine has — but only once the data can fill it.
