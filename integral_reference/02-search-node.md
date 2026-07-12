# Integral study — the negamax node: pruning, reductions, extensions

Source studied: `integral_reference/integral` (C++, no explicit license →
all-rights-reserved; **nothing below is copied code** — every formula is
hand-transcribed prose/arithmetic in our own words, file:line citations point
back at the vendored source for anyone who wants to re-verify). Primary
files: `src/engine/search/search.cc` (1472 lines, read in full),
`search.h` (224 lines), `stack.h` (137 lines), `constants.h` (105 lines, the
full SPSA-tunable constant list), `move_picker.h`/`.cc`. Git history mined
with `git log --oneline -- src/engine/search/search.cc` (205 commits touch
this one file, out of 669 total repo commits spanning 2023-10-25 to
2026-04-02) plus `git show` on ~25 individual tweak commits.

**Framing.** Read relative to three things we already have: (1)
`stockfish_reference/stockfish_search_notes.md` (the SF node our
`design/10-pruning-extensions.md` is explicitly modeled on), (2)
`reckless_reference/07-search-node-pruning.md` (the prior AGPLv3-study of
Reckless's node — already a rigorous 3-way SF/Reckless/us comparison,
so this note leans on it for the Reckless column rather than re-deriving
it), and (3) our own `design/10` + current `src/search/mod.rs`, which as of
this writing has **landed** Waves A–E in full (reduction table, RFP, NMP+
verification, LMR core, razoring, LMP, capture/parent futility, quiet SEE,
IIR, singular extensions + multicut + negative extensions, ProbCut + the
in-check small ProbCut) and most of Wave F/G with some sub-items explicitly
deferred (see `design/10` §4 for the current per-item status — this note
does not re-litigate what's already tracked there, only what Integral does
differently).

**Integral's `search.cc` is 1472 lines — noticeably larger than Reckless's
1448-line `search.rs` or our node — for one structural reason worth stating
up front**: Integral inlines its qsearch as a full sibling function
(`QuiescentSearch`, 248-481) with its own TT probe/store, its own
razoring-flavored futility, and its own history updates, rather than sharing
much machinery with `PVSearch` (484-1290). SF, Reckless, and our engine all
do the same (qsearch as a separate function) — this is not a genuine
size difference in complexity, just where the line count landed.

---

## 1. Data structures

`StackEntry` (`stack.h:55-108`) is architecturally the same idea as SF's
`Stack`/our `Stack` (D18): per-ply `pv`, `move`, `moved_piece`,
`excluded_tt_move`, continuation-history entry pointers, `threats`, and a
`reduction` field read by the parent for hindsight depth adjustment — all of
which line up with our design's N6/N8 hindsight machinery. Two differences:

- **Killer moves are alive** — `std::array<Move, 2> killer_moves`
  (`stack.h:73`, `AddKillerMove`/`ClearKillerMoves`, `stack.h:81-91`) — see
  §6 below; this is the single biggest structural divergence from D17.
- `Stack::kPadding = 6` (`stack.h:112`) vs our design's convention of 7
  (`ss-7 ... ss+2` always in bounds, D18) — Integral reads back only to
  `ss-4` (the `past_stack` search at `search.cc:701-706`) and forward to
  `ss+1`, so 6 is enough for its own use; not a meaningful difference, just
  a reminder that the padding depth is a function of how far back/forward a
  given node actually reads, not a universal constant.

`Thread` (`search.h:103-154`) mirrors our `Worker`: per-thread `Board`,
`history::History`, `Stack`, atomic node counter, `RootMoveList`,
`nmp_min_ply` (verification-search recursion guard, matching our design's
N11 field). `RootMoveList::SortNextMove` (`search.h:53-67`) is a plain
selection sort per MultiPV slot — same idea as our design's stable sort per
`§3.4 RootMove and RootMoves`, just insertion-sort-shaped instead of a
library stable-sort call.

**The SPSA tunable macro is the standout data-structure-adjacent finding.**
`src/tuner/spsa.h` defines `TUNABLE(name, value, min, max, disabled)` /
`TUNABLE_STEP(name, value, min, max, disabled, step)`, which under
`#define SPSA_TUNE` registers the constant as a live UCI option AND prints
one line per constant in the exact format an OpenBench-style SPSA harness
consumes: `name, type, value, min, max, step, learning_rate`
(`spsa.h:38-39,47-48`). Every single tunable in the entire pruning/reduction
suite — all ~90 of them, `constants.h:1-105` — is declared this way. This is
the same convention Stockfish's own `tune.h`/fishtest SPSA workflow uses,
and it is the mechanism that turned "Aron eyeballs a constant" into
"12,000+ iterations over 100,000+ games decide the constant" (commit
subjects literally say `LTC SPSA Tune (25k iterations, 100k games)`,
e.g. `5348ae81`, `8babfda9`, `84a3e833`, `276b58be`). We do not yet have an
SPSA-tuning pass at all (`design/10` §4 calls tuned values "starting points,"
D20) — this is a concrete, low-effort-high-value gap. See "Adapt to us,"
item 1, and §9.

---

## 2. Node walkthrough, in execution order

Numbering below reuses our design's N-labels (`design/10` §3.4-§3.10) where
the step lines up, purely so the two documents can be read side by side.

**N0-N4 — entry, qsearch handoff, upcoming-repetition, MDP.**
`search.cc:502-553`. Byte-for-byte the same shape as SF/our design: depth-0
→ qsearch (532-535); an upcoming-repetition early draw
(`board.HasUpcomingRepetition`, 524-529, cuckoo-table-backed — see the
`cuckoo.cc`/`.h` files, which we don't have yet, our design's Wave E item 19
note); repetition/max-ply draw; mate distance pruning (546-552) identical to
SF's `alpha.max(-mate+ply)`/`beta.min(mate-ply-1)`.

**N5/N6b — TT probe + non-PV cutoff.** `search.cc:555-585`. The cutoff test:

```text
!excluded && !PV && can_use_tt_eval
  && (cut_node || tt.score <= alpha)
  && tt.depth > depth - (tt.score <= beta)
```

The **depth-slack term** (`tt.depth > depth - (tt.score <= beta)`) is
byte-for-byte the same shape as our design's N6b first bullet (sourced from
current SF) — a genuine match, not a divergence. The **direction-agreement
term** is where Integral diverges: SF/our design require
`cut_node == (tt.value >= beta) || depth > 4` (a stricter, symmetric
"the stored bound's direction must agree with what this node type expects,
unless we're deep enough to trust it regardless" test); Integral instead
accepts EITHER `cut_node` outright (no value-direction check at all for cut
nodes) OR `tt.score <= alpha` (a plain fail-low test, with no depth-4
escape hatch for all-nodes). A looser, cheaper test overall. Integral also
has **no TT-cutoff quiet-history bonus/parent-malus pairing** the way our
design's N6b closing paragraphs specify (SF: reward a quiet ttMove that
still causes a cutoff; penalize the parent's early quiet), no rule50 guard,
and no child-TT verification/wrong-direction-penalize machinery. Integral's
TT cutoff is a strictly cheaper, less-hardened version of the SF/our-design
idea — the same overall verdict
`reckless_reference/07-search-node-pruning.md` R3 reaches for Reckless's own
TT cutoff (also missing the bonus/malus pairing and the child verification),
independently, on a different formula.

**Syzygy probe** (587-638) — plumbing, not a pruning heuristic per se; sits
exactly where our design's N7 TB probe sits (after TT, before static eval),
with the same "PV nodes narrow alpha/beta from bound-only TB results, non-PV
nodes return outright" structure.

**N6 — static eval, correction, material scale, 50-move damp.**
`AdjustStaticEval` (220-245) is a free function, not inlined at the call
site — three multiplicative/additive layers applied to the raw eval, in
order: (1) **material-phase scaling** — `eval *= (kMaterialScaleBase(27600) +
material_phase) / 32768`, where `material_phase` is the SEE-piece-value-
weighted count of knights/bishops/rooks/queens (both sides, unconditional on
whose turn) — structurally the *same idea* as Reckless's
`(raw_eval·(21032+material) + optimism·(1548+material)) / 27015`
(`reckless_reference/07-search-node-pruning.md` R4) and SF's own
material-scaled eval: **three independent engines converged on "scale the
eval down when there's less material on the board" as a load-bearing
correction**, worth treating as a settled idea rather than a per-engine
quirk once we do our own eval-scale pass (D21 HCE deferred, but the *shape*
transfers to PSQT-plus-NNUE evals too). (2) **correction history**
(`thread.history.correction_history->CorrectStaticEval`, 236-237) — same
purpose as our N6 correction-history substitution. (3) **50-move-rule damp**
— `eval *= (220 - fifty_moves_clock) / 220` (241) — same idea as our design's
rule50 scaling, different divisor (220 vs SF's 200-ish per our notes; a
gentler taper, since Integral's damp only reaches zero at clock=220, past
the actual 100-ply draw threshold — i.e. it never fully zeroes out inside a
legal game).

Notably: **material scaling used to live in the evaluation module and was
moved into search** (`d6ce508b`, "Move material scaling to search (#233)",
2026-03-17, bench 1970867) — a real refactor two-plus years into the
project, not a day-one design choice. Worth remembering when we do our own
eval/search boundary decisions: the "right" home for a term isn't always
obvious up front even to an experienced solo dev, and moving it later is
cheap as long as the bench signature is checked before/after (see §9's
`Bench:`-trailer discipline).

**Improving / opponent_worsening + hindsight** (677-720). `past_stack` looks
back to `ss-2`, falling back to `ss-4` if `ss-2` has no static eval (in-check
plies don't get one) — same two-ply/four-ply fallback our design's N6
specifies. `opponent_worsening = stack->static_eval + (stack-1)->static_eval
> 1` — near-identical to our design's `improving |= eval >= beta` (N11b)
sibling condition, just computed earlier and reused at RFP. The **hindsight
depth bump** (716-719): `if prev_stack->reduction >= 4096 &&
!opponent_worsening { ++depth; }` — reads the *raw pre-division* `r` value
(4096 = 4 plies in the suite's 1024ths-of-a-ply unit), the same "prior
reduction was large but our position didn't improve as a result → the
reduction likely under-searched, deepen this node" idea as our design's N6
hindsight, but **one-directional**: Integral has no counterpart to our
design's *negative* hindsight adjustment (`prior_reduction >= 2 &&
eval_delta > 0 → depth -= 1`). Reckless has both directions
(`reckless_reference/07-search-node-pruning.md` R6); we do too. Integral
only deepens, never additionally shallows here — a genuine three-way
divergence point, not just constant differences.

Also at this point (685-688): a **quiet move-ordering eval-diff nudge** —
when the *parent's* move was quiet, not in check, and didn't capture,
compute `their_loss = static_eval + parent_static_eval - bias` and feed a
clamped bonus into quiet-history AND pawn-history for the parent's move.
Same shape as our design's N8.

**N9 — Razoring.** `search.cc:741-748`. `!excluded && depth <=
kRazoringDepth(4) && alpha < 2000 && static_eval + kRazoringMult(393) *
(depth - !improving) < alpha` → qsearch verify at `(alpha, alpha+1)`, return
if it still fails low. Same overall purpose as SF's/our design's/Reckless's
N9 (a shallow-node "this looks hopeless, drop straight to qsearch" test),
but a genuinely different, and explicitly **depth-capped**, shape: SF/our
design use `eval < alpha - 465 - 300·depth²` with **no depth cap at all**
("the quadratic margin makes it self-limiting," `design/10` N9), while
Integral's margin is **linear in depth** (`393·(depth - !improving)`) AND
hard-capped at `depth <= 4`. The magnitudes diverge sharply, not just the
curve: at depth 4, SF's/our design's margin is `465 + 300·16 = 5265`, while
Integral's is only `393·4 = 1572` (improving) or `393·3 = 1179` (not
improving) — roughly a third the size, meaning Integral's razoring fires far
more readily than SF's shape would at the same depth, compensated for by
Integral simply turning it off past depth 4 rather than letting the margin
grow large enough to self-limit the way SF's quadratic does. Two
legitimately different engineering choices reaching for the same effect.
The extra `alpha < 2000` guard (don't razor when already near a
forced-mate-scale alpha) has no direct SF/our-design counterpart but
Reckless has the identical guard (`alpha < 2048`) — independent
convergence, or one project read the other's source (both are public;
Integral's README explicitly credits borrowing ideas from sibling engines,
e.g. "Using Alexandria's soft limit calculations," commit subject from the
pre-PR era — see §9).

**N10 — Reverse futility pruning.** `search.cc:723-737`. `depth <=
kRevFutDepth(11)`, margin:

```text
margin = depth·kRevFutMargin(50)
       - improving·kRevFutImprovingMargin(94)·!opponent_easy_capture
       - opponent_worsening·kRevFutOppWorseningMargin(25)
       + eval_complexity·kRevFutComplexityMargin(12)/32
       + prev_stack.history_score/kRevFutHistoryDiv(586)
margin = max(margin, kRevFutMinMargin(13))
prune if eval - margin >= beta → return lerp(eval, beta, kRevFutLerpFactor(0.516))
```

`kRevFutDepth = 11` is, perhaps counter-intuitively, **lower** than our own
design's RFP depth cap — `RFP_MAX_DEPTH = 17` (exclusive, `design/10` §3.1,
SF-current) — even though Integral's flat-linear margin (`depth·50`, ignoring
the improving/complexity/history adjustment terms) is the smaller of the two
functions almost everywhere: SF's/our design's per-ply multiplier itself
grows with depth up to a cap (`min(40+4d,80)`), so the two margins are close
at `d=1` (44 vs 50 — Integral's is actually slightly larger there) but SF's
pulls steadily ahead as depth grows (`d=11`: `880` vs Integral's `550`,
a ~1.6× gap that would only widen were Integral's cap not stopping it at
`d=11`). Two independently-tuned engines land on visibly different depth
caps for the "give up on RFP beyond here" cutoff (11 vs 17) while using
margins of genuinely different shape and magnitude — a reminder that the
depth cap and the per-ply margin are coupled through the SPRT/SPSA process
that produced them, not independently meaningful numbers to compare in
isolation. There's a whole
commit trail retuning this margin, e.g. `db0a0cb2` "Use previous move
history in RFP margin," `c3e552ce` "Tweak RFP margin depending on opponent's
evaluation trend," `3f22846f` "Use static evaluation correction difference in
RFP". `eval_complexity = |static_eval - raw_static_eval|`
(`search.cc:674`) — how much the correction-history term moved the eval —
feeding a "the correction was large, trust the eval less, prune less
readily" margin term. This is the exact same *purpose* as our design's
"folding correction into RFP margin" (Wave F item 21, `RFP_CORR_DIV`), a
different-shaped implementation (Integral scales by the raw absolute
correction magnitude; our design/SF scale by the signed `correction_value`
divided by a large constant). The `opponent_easy_capture` term
(`board.GetOpponentWinningCaptures() != 0`, 721) — RFP applies its full
improving-bonus margin ONLY if the opponent doesn't have an obvious winning
capture sitting on the board — is functionally the same idea as our design's
Wave G "threat-coupled RFP margin" (D30, Reckless-derived), independently
present in Integral without any Reckless influence (Integral predates our
Wave-G study) — a third confirmation that "RFP should trust the eval less
when the position has live tactics" is a real, transferable idea, not
something we invented by reading Reckless alone.

**N11 — Null move pruning + verification.** `search.cc:750-799`. Gate:
`!(ss-1).move.IsNull() && eval >= beta && static_eval >= beta +
kNmpBetaBase(146) - kNmpBetaMult(7)·depth && !excluded && ply >=
nmp_min_ply`, plus the standard "some non-pawn-king material exists"
zugzwang guard. Reduction:

```text
eval_reduction = min(2, (eval - beta) / kNmpEvalDiv(169))
reduction = depth/kNmpRedDiv(3) + kNmpRedBase(4) + eval_reduction + improving
reduction = clamp(reduction, 0, depth)
```

Same three ingredients as SF's classic `R = 3 + depth/3 (+min(2,(eval-beta)/
200))`-family formula our design cites as "current SF dropped the eval-based
term" (`design/10` node step 11 commentary) — **Integral kept the
eval-based term**, and its own commit history shows *why*: it was added
(`ba78f44c`, "Eval based NMP reduction (7 elo)") when the base reduction was
still the simpler `depth/4 + 4`, then the base was independently retuned
(`ab39eb8d` "Tweak NMP base divisor") after the eval term was already
proven. The **`improving` additive term** on the reduction — reduce *more*
when we're improving — was added later still (`0f4ee79b`, "Utilize improving
heuristic in NMP (#101)"); Reckless has a *subtractive* improving term
(`-94·improving/1024` shrinking the margin, not the reduction) at the gate,
not the reduction — a genuinely different place to apply the same signal.
Verification (785-798): standard high-depth re-verify at `nmp_min_ply = ply +
3·(depth-reduction)/4`, matching SF's/our design's N11 verification
*structure* exactly, including "only verify if `nmp_min_ply == 0`" (don't
nest verifications) — with one small numeric offset: Integral verifies from
`depth > 14` (i.e. `depth >= 15`) while our design's `NMP_VERIFY_DEPTH = 16`
gate only verifies from `depth >= 16`, a one-ply difference in where the
(rare, high-depth-only) verification search kicks in.

**ProbCut.** `search.cc:801-865`. `depth >= kProbcutDepth(5) &&
|beta| < TBWinInMaxPly && (!tt_hit || tt.depth+3 < depth || tt.score >=
pc_beta)`, `pc_beta = beta + kProbcutBetaDelta(213)`, SEE threshold
`pc_beta - eval`, probe depth **flat** `depth - 3`. This is SF's classic
**flat two-step** shape (qsearch pre-check, then one real search at a fixed
shallower depth) — notably **simpler** than Reckless's dynamic
`probcut_depth`/`adjusted_beta`-shifting scheme
(`reckless_reference/07-search-node-pruning.md` R9b, which our design's
Wave E item 16 fidelity note already flags as "worth building against
directly" once we land ProbCut — we already have; see `design/10` §4 Wave E).
Two numeric details differ from our own N13, though the shape matches:
our design triggers at `depth >= 3` with `probcut_beta = beta + 214 -
59·improving` and an improving-adjusted probe depth
(`depth - 4 - improving`); Integral triggers later (`depth >= 5`), uses a
flat, improving-independent beta delta (213, no adjustment term) and a flat
probe-depth offset (`depth - 3`, no improving term either). Since we've
landed the SF-flat *shape* already, Integral's ProbCut doesn't teach us a
new structure here, but it's useful negative evidence on the improving-
adjustment specifically: a top-6 engine ships ProbCut with NO improving
coupling anywhere in it, where our own design (SF-sourced) has it in two
places — worth knowing if we ever want an isolated "does the improving
term in ProbCut actually earn its keep" experiment. More generally: the
extra dynamic-depth/beta-shift complexity Reckless's ProbCut carries isn't
obviously necessary for top-10 strength, since Integral ships without it.

**IIR.** `search.cc:869-874`. `(PV || cut_node) && depth >=
kIirDepth(4) && !excluded && (!tt_move || tt.depth + 4 < depth)` → `depth -=
1`. Fires two plies earlier than our own N12 (`IIR_MIN_DEPTH = 6`,
`design/10` N12) — a real, tuned threshold gap, not just the shallow-ttMove
disjunct discussed next. The `tt.depth + 4 < depth` disjunct (IIR fires even WITH a shallow
ttMove, not only with none) is Integral's own addition
(`644beb5a`, "Perform IIR if `tt_entry->depth + 4 < depth` (#131)") — a
refinement past the "any ttMove at all suppresses IIR" baseline our design's
Wave B item 9 describes (which we've reverted anyway per `design/10` §4 —
Wave B item 9 regressed ~−15 Elo for us and sits parked on
`wip/p9-item9-iir-followpv`). Worth re-testing this exact shallow-ttMove
variant when IIR is revisited post-NNUE.

---

## 3. The move loop — per-move pruning

`search.cc:886-979`. Move-count/history-driven guards, all requiring
`!in_root && best_score > -TBWinInMaxPly`:

- **LMR-depth prediction first** — Integral computes a *speculative*
  reduction (908-932, same table lookup + a subset of the real LMR
  adjustment list: non-PV, quiet-history, not-improving) purely to derive
  `lmr_depth` for the pruning margins below, then throws it away and
  recomputes the real reduction later in the move loop (1052-1112) with the
  full adjustment list (cut-node, ttPv, gives-check, killer, complexity).
  This two-pass "predict, then actually reduce" structure matches our
  design's N-move-loop section exactly (`lmr_depth = new_depth - r/1024`
  computed once for pruning, the *real* reduction computed again at make-move
  time) — not a divergence, just confirmation the idiom is standard across
  all three codebases.
- **LMP** (936-941): `moves_seen >= (kLmpBase(5) + depth²) /
  (3 - (improving || eval >= beta))` → `SkipQuiets()`. The `eval >= beta`
  disjunct (LMP kicks in more readily when the position already looks like a
  fail-high, not just when improving) has **no SF counterpart in our notes**
  and is a genuinely later Integral addition (`ea32ffc3`, "LMP more often
  when static eval suggests a fail high (#203)," 2025-06-10) — worth its own
  isolated proxy-SPRT for us (cheap: one extra disjunct on an existing
  formula).
- **Futility (parent-futility-equivalent)** (943-954): `lmr_depth <=
  kFutPruneDepth(8) && !in_check && is_quiet && static_eval + margin <= alpha
  && !gives_direct_check`, margin = `128 + 81·lmr_frac_depth/1024 +
  history/124`. Same shape as our design's parent futility, with the history
  term folded straight into the margin (ours: history buys effective depth
  separately, `design/10` §3.5's DEFERRED half of item 11) rather than
  Integral's simpler "history nudges the margin directly." The
  `!gives_direct_check` guard (only checked via a cheap direct-check proxy;
  matches Reckless's `is_direct_check`, not a full `gives_check`) is a very
  recent addition (`484db145`, "Update move ordering and futility pruning if
  a move directly delivers a check (#236)," 2026-03-21) — before that,
  Integral (like current SF/our design) pruned checking quiets too as long as
  they were "quiet." Exempting checks from futility is intuitive (a check
  might matter tactically regardless of material) but notably NOT something
  current SF/our design do at this site — a candidate experiment for us.
- **History pruning** (956-964): `depth <= kHistPruneDepth(5) &&
  history_score <= margin`, margin = `-250 - 2198·depth` (quiet) or
  `-447 - 1773·depth` (noisy). **Ordered BEFORE SEE pruning** as of a very
  recent commit (`0f65df61`, "Perform History Pruning before SEE Pruning
  (#235)," 2026-03-20) — confirmed by `git show 0f65df61`: the diff is a
  pure 20-line reorder of the two existing blocks, no formula or constant
  changes. Our design's Wave C item 11 note independently
  documents the identical ordering call ("history-before-SEE," per this
  study's assignment brief) — we already ship it this way; Integral converged
  on the same order two-plus years later via its own SPRT, not by reading us
  (we didn't exist as prior art for them). Cheap, independent confirmation
  the ordering choice is real and not a coin-flip.
- **SEE pruning** (966-978): quiet threshold `kSeeQuietThresh(-17)·
  lmr_depth²`; noisy threshold `kSeeNoisyThresh(-98)·depth -
  history/kSeePruneHistDiv(131)`. Quadratic-in-lmr_depth for quiets (matches
  our design's `QSEE_MULT·lmr_depth²` shape exactly, different constant:
  Integral's -17 vs our SF-sourced -25); linear-in-depth with a
  history-discount for noisy moves — same *shape* as our design's capture
  SEE margin (`175·depth + 34·capt_hist/1024`, sign convention flipped since
  ours adds a positive capture-history bonus to a positive margin while
  Integral subtracts a positive history score from a negative threshold —
  same net effect).

---

## 4. Singular extensions, multicut, negative/double/triple extensions

`search.cc:981-1030`. Trigger: `!in_root && depth >= kSeDepth(5) && move ==
tt_move && tt.depth+3 >= depth && tt.flag != Upper && |tt.score| <
TBWinInMaxPly && ply < 2·root_depth`. The **`ply < 2·root_depth`** global cap
(989) — instead of a per-node extension counter — is worth calling out on
its own: Integral's *first* singular-extension commit
(`2d852baa`, "Double Extensions (#12)") used a `StackEntry::double_extensions`
int field, inherited ply-to-ply and capped at 8, to bound how many times a
single search line could stack double-extensions (classic SF-era idiom).
Five weeks later (`0995a2a4`, "Lower singular extension depth limit (#49)")
that field was **deleted outright** and replaced by the `ply <
2·root_depth` global check we see in the final code — a strictly simpler
mechanism (one integer compare, no per-node state to thread through the
stack) that serves the same "don't let a single line explode the tree via
runaway extensions" purpose. This mirrors current SF's approach almost
exactly (SF's actual per-search-line extension budget is also root-depth-
relative, not a per-node counter) — Integral converged onto the SF-shaped
solution by simplifying its own more complicated first attempt, not by
copying SF outright (the commit predates any stated SF cross-reference).

```text
singular_beta  = tt.score - kSeBetaMargin(17)·depth/16
singular_depth = kSeDepthReduction(7)·(depth-1)/16      // ≈ 0.4375·(depth-1)
value = search::<NonPV>(singular_beta-1, singular_beta, singular_depth,
                        cut_node)              // same ply, excluded_tt_move set
```

`singular_depth`'s `7/16 ≈ 0.4375` fraction of `(depth-1)` is smaller than
SF's/our design's/Reckless's uniform `1/2·(depth-1)` — a real, tuned
difference (a slightly cheaper exclusion search), reachable for us as an
isolated SPSA-style experiment once we have a tuner (§1 above explains why
we don't yet).

- **Singular → extend**: `value < singular_beta`. Extension size:
  `2 + (!PV && is_quiet && value < singular_beta - kSeTripleMargin(58))`
  when `value < singular_beta - kSeDoubleMargin(8) -
  kSePvDoubleMargin(150)·PV`, else `1`; **and** `depth +=
  !PV && depth < kSeDepthExtensionDepth(13)` — the WHOLE node deepens by one
  extra ply, on top of the per-move extension, specifically in non-PV nodes
  below depth 13. This second, node-level deepening is distinct from SF's
  scheme (SF only deepens the node via `new_depth += extension`, no separate
  flat `depth++`). **The double-extend eligibility test was itself widened
  recently**: `8103098a`, "Double extend in PV nodes (#228)," Nov 2025 —
  before this commit the double-margin check was gated `!in_pv_node`
  outright (PV nodes could only ever get a single-ply extension, never a
  double); the commit drops that gate and instead makes PV nodes pay a
  bigger margin to qualify (`kSeDoubleMargin + kSePvDoubleMargin·PV`), while
  the node-level `depth++` bump and the triple-extension bonus both keep
  their `!in_pv_node` restriction unconditionally. Net effect: PV nodes
  gained the ability to double-extend a singular ttMove for the first time,
  gated by a stiffer margin, while triple-extensions and the whole-node
  depth bump remain non-PV-only. A concrete, later-stage refinement of an
  already-shipped heuristic, not a day-one design choice — worth remembering
  when we eventually revisit our own singular-extension margins: our
  `design/10` §3.6 double/triple margin formulas already carry PV-conditioned
  terms (`SE_DOUBLE_PV`/`SE_TRIPLE_PV` widen the margin under PV rather than
  excluding PV outright), so we may already be ahead of Integral's pre-#228
  state here — this commit is independent evidence that a PV-conditioned
  margin, not a flat PV exclusion, is the right shape for double-extending
  under a PV node.
- **Multicut**: `value >= beta && |value| < TBWinInMaxPly` → `return value`
  directly — **no fail-soft blend toward beta** the way Reckless's multicut
  does (`lerp(value, beta, 0.4027)`) and unlike Reckless there is **no
  `tt_move_history` scalar dock** on the multicut path at all (Integral has
  no `tt_move_history`-equivalent statistic anywhere in this file — a
  history axis both SF and Reckless have that Integral simply doesn't track).
- **Negative extensions**: `tt.score >= beta → -3`, `else if cut_node → -2`
  — the **two-tier** split (SF's/our design's shape) that Reckless collapses
  to a single `-3`. Integral's own history shows this two-tier split emerging
  incrementally: the *first* negative-extension commit
  (`64dc7d94`, "Negative Extensions on potential TT move cutoff (5 elo)")
  had only the `tt.score >= beta → -1` case; `fb46cfe9` ("Negative extensions
  in cutnodes (#47)") folded `cut_node` into the SAME `-1` tier
  (`tt.score >= beta || cut_node`); only later tuning
  (`75d3c608`, "Increase assumed fail high negative extension amount (#184)")
  pushed the magnitudes up and, per the final code, re-split them into
  distinct `-3`/`-2` tiers. Three revisions to reach the SF-shaped two-tier
  split from a single-tier start — useful evidence that the "obviously
  correct" final shape of a heuristic is rarely obvious in advance even to
  the person tuning it.

**Check extensions were removed, deliberately, once the engine got
stronger.** Early code (visible from `0995a2a4`, Aug 2024) carried a flat
`if (in_check) { ++extensions; }` for every move, with the comment *"Check
Extensions: Integral's not yet strong enough to simplify this out :)"* — an
explicit, self-aware acknowledgment that plain check extensions are known
folklore being kept as a crutch. Seven weeks later, `d3189f19`
("Simplify out check extensions (#82)," bench 1858301) deletes the three
lines outright with no other change. **This is the single cleanest piece of
evidence in the entire history that "no plain check extensions" (our
design/10 §3.6 closing line: *"There are no plain check extensions anywhere
(D17 spirit: we follow current SF, not folklore)"*) is not a
current-SF-only idiosyncrasy** — an independent solo dev on a different
codebase rediscovered the same fact empirically, at the point where their
singular-extension/pruning machinery had matured enough to make the plain
check extension a net negative rather than a net positive. We should read
our own "no check extensions" stance as *already validated by a second data
point*, not merely inherited from SF by citation.

---

## 5. Late move reductions

`search.cc:1052-1142`, base table `search.cc:19-48`. The base:

```text
LateMoveReduction[is_quiet][depth][move_count] =
    base + ln(depth)·ln(move_count) / divisor
```

with **separate `(base, divisor)` pairs for quiet vs tactical (noisy) moves**
(`kLmrQuietBase=0.794, kLmrQuietDiv=2.07` vs `kLmrTactBase=-0.34,
kLmrTactDiv=2.86`) baked into a **precomputed 2-D table indexed by
`[is_quiet][depth][move_count]`**, rebuilt only on `NewGame`/hash-clear
(`search.cc:1441`). This is architecturally the **classic, pre-2023-era SF
shape** (a `Reductions[improving][depth][moveNumber]`-style lookup table) —
genuinely different from BOTH current SF (per our `design/10` §3.3, a
*continuous* closed-form using two independent 1-D tables multiplied
together, `reductions[depth]·reductions[move_number]`, no quiet/tactical
split, plus an aspiration-window-width term `- delta·617/root_delta` neither
Integral nor Reckless have) AND Reckless (`ilog2(depth)`-only, no
move-count dimension at all,
`reckless_reference/07-search-node-pruning.md` §5.1 — the single biggest
Reckless divergence from SF, flagged there as our design's "post-Wave-C
standalone experiment" candidate). **All three engines disagree on the base
reduction's functional shape.** Given a top-6 engine ships the pre-table
shape and still reaches its strength, our own commitment to current SF's
continuous two-table shape isn't obviously required for strength — but
neither is it refuted; this is exactly the kind of question an SPSA/fixed-
node-count proxy comparison (once we have a tuner) could actually answer
for our own eval/move-ordering mix, rather than assuming any one engine's
choice transfers.

The **adjustment list** (`search.cc:1059-1101`, all in 1024ths): non-PV
(+`kLmrNonPvNode`=743), was-tt-pv (-`kLmrWasPvNode`=1168), cut-node
(+`kLmrCutNode`=2205), gives-check (-`kLmrGivesCheck`=942), quiet/capture
history score (-history·`kLmrHist{Quiet,Capture}`/`kLmrHistDiv`), not-
improving (+`kLmrNotImproving`=949), high `eval_complexity`
(-`kLmrComplexity`=778 if above `kLmrComplexityDiff`=73), **killer move**
(-`kLmrKillerMoves`=886) — every one of these has a direct analogue in
SF's/our design's reduction-finalization list (`design/10` §3.7) EXCEPT the
killer-move term, which only exists because Integral tracks killers at all
(§6). No `stat_score`-style combined capture/quiet-history reduction term
the way our design's `RED_STATSCORE_NUM` (Wave C item 10) has — Integral
applies history directly rather than through a combined stat_score.

**Deeper/shallower re-search** (`1114-1121`): `do_deeper = score >
best_score + kDoDeeperBase(30) + kDoDeeperMult(33)·new_depth/16`,
`do_shallower = score < best_score + kDoShallowerBase(5)`,
`new_depth += do_deeper - do_shallower`, gated `!in_root`
(`453c1dc4`, "Don't perform deeper/shallower adjustment at root (#230)" —
another recent, narrow fix: the adjustment was firing at the root and
presumably mis-sizing the root move's re-search depth). Same idea as our
design's deeper/shallower rule (`design/10` §3.8: `do_deeper` margin `52`,
`do_shallower` margin `9`, both flat, no depth-scaling term) — Integral's
`do_deeper` margin scales UP with depth (`30 + 33·new_depth/16`, so it grows
past 52 above roughly depth 11) where our SF-sourced margin is a flat 52 at
every depth; Integral's `do_shallower` margin (flat 5) is close to our flat
9.

---

## 6. Killer moves — the D17 tension, worth naming directly

Integral tracks two killer moves per ply (`stack.h:73`, `MovePicker::Stage::
{kFirstKiller,kSecondKiller}`, `move_picker.h:29-30`) and uses them in two
places: move ordering (killer stage sits between good-noisy and quiet
generation) and LMR (killers get reduced `kLmrKillerMoves`=886 less,
`search.cc:1097-1099`). **Both Reckless (`reckless_reference/05-movepick-
history.md:222,431`) and current Stockfish
(`stockfish_reference/03-movegen-movepick-history.md:126`,
`stockfish_reference/NOTES.md:26`) have removed killers entirely** —
history-based ordering supersedes them, per our own D17 ("No killer/
countermove tables, ever — modern SF removed them; we skip that detour
entirely"). **Integral is the one data point among our three studied
engines that keeps killers, and it is a top-6 engine.** This does not by
itself argue for reintroducing killers (D17's SF-removal rationale is
sound, and Integral's own history shows no commit ever tried removing
them to measure the delta — so there's no evidence killers are *positively*
helping Integral either, only that they haven't been ripped out). Read as:
killers-vs-none is probably a small, second-order effect either way at this
maturity level, not a load-bearing strength differentiator — D17 stands,
but the "modern engines removed them" framing should be read as "some
strong modern engines," not "all of them," if this decision is ever
revisited.

---

## 7. Aspiration windows

`search.cc:92-140`. Per MultiPV line: `window = kAspWindowDelta(10) +
average_score²/kAspWindowScoreDiv(15954)`, symmetric around the move's own
running `average_score` (an EMA of the score across iterations,
`search.cc:1170-1171`, not just the last iteration's score). Fail-low:
`beta = alpha; alpha = max(-inf, score - window)`. Fail-high: `alpha =
max(beta - window, alpha); beta = min(inf, score + window)`; a
`fail_high_count` (capped effectively by the `alpha < 2000` guard) shrinks
the **next iteration's depth** via `depth - fail_high_count` at the call
site (`search.cc:104`) — the exact "each consecutive fail-high re-search
costs one shallower ply" idiom our own `design/07-search-core.md` §3.6
already specifies as SF-current (`adjusted_depth = root_depth -
failed_high_cnt - ...`). **This exact scheme (symmetric alpha/beta widening,
not asymmetric lerp-based widening) was only adopted by Integral in
`1568901f`, "Use SF-style aspiration window adjustments (#234)," dated
2026-03-17** — the commit message says so explicitly, and the diff replaces
an older `std::lerp(beta, alpha, kAspBetaLerpFactor)`-based asymmetric
widening with the symmetric form. **We have shipped the scheme Integral only
adopted a few months ago from day one of our aspiration-window work**
(`design/07` §3.6, sourced directly from the SF tree) — concrete evidence
our "copy current SF's exact current formula, don't reinvent" strategy
(D20/D27) is not merely "safe," it is presently *ahead* of a top-6 engine on
this one specific mechanism. `kAspWindowDepth` (`constants.h:13`) is declared
but **entirely unused** anywhere in `search.cc` — leftover cruft from a
prior depth-gated aspiration scheme, never deleted after the `#234`
rewrite. Small, harmless, and a useful reminder that a live, strong,
actively-tuned engine still accumulates a bit of dead code; this is not
something to over-index on cleaning up preemptively in our own project.

---

## 8. Comparison table (Integral / current SF / Reckless / us)

| Heuristic | Integral | Current SF (per our notes) | Reckless | Us (`design/10`, landed) |
|---|---|---|---|---|
| RFP depth cap | 11 | 17 (exclusive, `RFP_MAX_DEPTH`) | none stated (quadratic, uncapped) | 17, same as SF (landed) |
| RFP margin shape | linear-ish w/ complexity+history terms | `min(40+4d,80)·d` (capped-linear) | quadratic `1140·d²/128` | capped-linear (SF-sourced) |
| RFP threat coupling | yes (`opponent_easy_capture`, pre-dates our Wave G) | no | yes (`-54` if no own piece threatened) | yes (Wave G, D30, Reckless-derived) |
| Razoring margin shape | linear `393·(d-!improving)` | quadratic | quadratic w/ extra guards | quadratic (SF-sourced) |
| NMP reduction | `d/3+4+min(2,Δ/169)+improving` | flat `7+d/3` (no eval term, per our notes) | fixed-point eval-scaled, `~4.3+0.9·improving+0.26d+margin-term` | SF-current flat form (no eval term) landed |
| NMP extra gates | zugzwang-material only | zugzwang-material only | + ttPv margin, cutoffCnt discount, TT-capture veto, `!potential_singularity` | zugzwang-material only |
| ProbCut shape | flat two-step (SF-classic) | flat two-step | dynamic depth/beta-shift + fallback re-verify | flat two-step (landed) |
| IIR trigger/depth | PV\|\|cut, `d>=4`, `!ttMove \|\| tt.depth+4<depth` | PV\|\|cut, `d>=6`, `!ttMove` | (not covered in Reckless note scope) | PV\|\|cut, `d>=6`, `!ttMove` (shallow-ttMove variant NOT yet tried) |
| LMR base shape | 2-D table, quiet/tactical split, `ln(d)·ln(m)` | continuous, 2×1-D tables + window-width term | `ilog2(depth)` only, no move-count axis | continuous (SF-sourced), landed |
| Killer moves | yes, 2 slots, affect ordering + LMR | no | no | no (D17) |
| Singular exclusion-depth fraction | `7/16·(depth-1) ≈ 0.4375` | `1/2·(depth-1)` | `1/2·(depth-1)` | `1/2·(depth-1)` (SF-sourced) |
| Singular trigger depth | `depth >= 5` (declared tunable range oddly 6-12) | `depth >= 6+ttPv` | `depth >= 5+ttPv` | `depth >= 6+ttPv` (landed) |
| Singular bound-type-conditioned margin | no | no | yes (`d/4` if Exact bound) | no (matches SF) |
| `is_shuffling` rule50-oscillation guard | no | yes | no | yes (landed — ahead of both Integral and Reckless here) |
| Multicut return | plain `return value` | plain return (per our notes) | fail-soft blend `lerp(value,beta,0.40)` | plain return (landed) |
| `tt_move_history` scalar | absent | present | present | not yet landed (Wave D item 15, deferred) |
| Negative extension tiers | two (`-3`/`-2`), reached after 3 revisions | two (`-3`/`-2`) | one (`-3`, both conditions) | two (`-3`/`-2`, landed) |
| Double/triple extension gate | global `ply < 2·root_depth` | root-depth-relative | (not detailed in Reckless note) | not yet landed at this granularity (Wave D deferred sub-items) |
| Plain check extensions | removed (`d3189f19`, 2024-09) | absent | absent (per our notes) | absent (D17) |
| LDSE (low-depth singular ext.) | absent | absent | present | absent |
| Aspiration window shape | symmetric SF-style (since 2026-03) | symmetric | (not detailed here) | symmetric (landed since inception) |
| Material-scaled eval | yes (in search, since `#233`) | yes (in eval) | yes (in search, `correct_eval`) | not yet (D21 HCE deferred; shape transfers to future eval work) |
| SPSA-tunable constant framework | yes, ~90 constants, OpenBench-format print | yes (fishtest) | present but "currently-orphaned" per Reckless study | **absent** — biggest single gap, see "Adapt to us" item 1 |

---

## 9. What the git-log mining shows about process (supporting evidence, not this note's main topic)

Per-file history on `search.cc` (205 of 669 total commits) reads as a
textbook **incremental-SPRT** discipline once it starts, but it did NOT
start that way:

- **Commits 1-~300 (Oct 2023 - ~Apr 2024)**: informal messages ("stuff",
  "logic fixes", "lots of fixes", "search changes, pesto eval testing") —
  ordinary early-hobby-project style, no measurement discipline visible in
  the message itself.
- **~Apr 2024 onward**: commit subjects start carrying explicit Elo deltas
  in parentheses — `c03f4c45` ("...futility pruning (~30 elo)...") is the
  first, then a long run of `(N elo)` subjects through mid-2024 (`00a4286a`
  "Using quiet history in LMR (4 elo)", `9c25c954` "Split LMR into tactical/
  quiets (15 elo)", `0e06ea7c` "Singular Extensions (21 elo)", `049ebcee`
  "Multicut pruning (7 elo)", etc.) — this is a solo dev running some form
  of proxy/SPRT testing (STC-scale, per some subjects: "TM Move Stability
  Scaling (7 elo STC)") well BEFORE any PR-numbered workflow existed.
- **`20e5da17` ("Defended pawns and history pruning (#7)", 2024-07-09)** is
  the FIRST commit with both a GitHub PR number in its subject AND a
  `Bench: <N>` trailer in its body. From this point on, **313 of the
  remaining commits (out of 669 total) carry a `Bench:` trailer** —
  virtually every commit that can affect search behavior, exactly our own
  D24/`Bench:` convention, independently arrived at. The PR-numbered,
  Bench-trailered discipline and the transition away from raw "(N elo)"
  subjects happen at the SAME commit — i.e., once GitHub PRs entered the
  workflow, the elo-in-subject convention was retired in favor of the
  Bench-trailer-plus-external-SPRT-result convention (Elo numbers move out
  of the commit message and presumably into an OpenBench-style PR
  description/CI check not visible in the git history itself).
- Individual pruning/reduction/extension heuristics are near-universally
  **one-commit-per-idea**, frequently followed by a SEPARATE later commit
  that only reorders or retunes the same block (`0f65df61` — a pure
  20-line reorder swap of history-pruning vs SEE-pruning with nothing else
  touched; `27ceb3ba` "Simplify LMR extensions"; `f6cbeea5` "Simplify LMR
  condition") — matching our own D24/D25 "one heuristic (or one tightly-
  coupled cluster) per commit, gated, bench-recorded" ground rule
  (`design/10` §4) almost exactly, arrived at independently by a solo dev
  over ~2.5 years rather than designed up front.
- Ideas are openly cross-credited between engines in commit subjects
  despite the no-license/all-rights-reserved posture on Integral's OWN code
  — e.g. "Using Alexandria's soft limit calculations (43 elo LTC, 68 elo
  STC)" — confirming the wider top-engine community treats *techniques* as
  freely shared even when *source* is not, which is exactly the posture
  this whole study series (D27/D31 and this note) already takes toward
  Integral itself.

None of this is the primary deliverable for THIS note (docs 09-11 in this
study series own the "how did a solo dev reach top-10" question in full);
it's recorded here only because mining `search.cc`'s own commit history was
an explicit part of this note's assignment, and the pattern it reveals is
specific to how search-heuristic work in particular gets tuned over years,
not just a generic observation.

---

## Adapt to us

1. **Build an SPSA-tuning pipeline modeled on Integral's `TUNABLE`/
   `TUNABLE_STEP` macro convention** — our `params.rs` constants (D20)
   are currently "starting points"; Integral's identical-shaped `constants.h`
   with a compile-time-gated UCI-option/print mechanism is exactly the shape
   an OpenBench-style SPSA harness needs, and it's a small, mechanical,
   one-time investment (one macro/attribute + a build-time cfg flag) rather
   than a design change. Target: a `#[cfg(feature = "spsa")]`-gated variant
   of `search/params.rs`'s constant declarations, wired to UCI options,
   printing `name, type, value, min, max, step` on startup. **Impact: large**
   — every constant in `design/10` currently has zero empirical tuning
   behind it; this is the single biggest lever this study surfaces, and it's
   process/tooling, not a search-tree change, so it's near-zero Elo risk to
   build.
2. **Try the shallow-ttMove IIR variant**: `(!tt_move || tt.depth + 4 <
   depth)` instead of our current `!tt_move` (`search.cc:872`,
   `644beb5a`). One-line, isolated, cheap SPRT `[0,3]`. Revisit alongside
   Wave B item 9's parked IIR revert (`wip/p9-item9-iir-followpv`) — it's
   plausible our reverted IIR regressed partly because it never covered the
   "shallow but present" ttMove case Integral's variant adds.
3. **LMP's `eval >= beta` disjunct** (`ea32ffc3`, "LMP more often when
   static eval suggests a fail high"): add `|| eval >= beta` to our own LMP
   improving-divisor condition (`design/10` §3.5's `(3 - improving)` term
   currently has no such disjunct). One-line, isolated, SPRT `[0,3]`.
4. **Exempt direct-check-giving quiets from futility pruning**
   (`484db145`): our design's parent futility currently has no
   `!board.gives_direct_check(m)` exemption. Cheap (we already compute an
   O(1) direct-check proxy per the Wave-G nps-micro-ops note) and worth an
   isolated SPRT `[0,3]` — Integral measured this positive enough to land it
   in the most recent handful of commits to the file.
5. **Re-test a smaller singular-exclusion-depth fraction** — Integral's
   `7/16 ≈ 0.4375·(depth-1)` versus our current SF-sourced `1/2·(depth-1)`.
   Only actionable once the SPSA pipeline (item 1) exists; flag as a
   candidate coefficient for the FIRST SPSA run rather than a one-off manual
   SPRT, since it's exactly the kind of continuous constant SPSA is suited
   to and a single manual guess would be low-information.
6. **Do NOT add killer moves.** Two of our three studied engines (current
   SF, Reckless) independently removed/never-had them; the third
   (Integral) never measured the removal, so it's not counter-evidence, only
   an absence of evidence. D17 stands as correctly-reasoned; this is a
   confirmation entry, not a reversal.
7. **Do NOT adopt Integral's classic 2-D quiet/tactical LMR table wholesale**
   — we already ship current SF's continuous, more-refined shape (`design/10`
   §3.3), and Reckless's still-different `ilog2`-only shape is already
   flagged in our own doc as a standalone post-Wave-C experiment. Integral's
   table-based shape is a THIRD distinct option; worth knowing it exists (a
   top-6 engine ships it) but not worth building against without evidence
   our continuous shape underperforms it for us specifically — again, an
   SPSA-pipeline question, not a manual-guess one.
8. **The material-phase eval-scaling idiom** (`eval *= (base + material)/
   32768`, `search.cc:220-233`) — independently present in Integral AND
   Reckless with near-identical shape, both applied at the search layer, not
   the eval layer. When our own eval work (post-P13-NNUE or a PSQT-phase
   revisit) reaches this territory, treat "scale the raw eval by a
   material-based factor, applied at the point of consumption in search
   rather than baked into the eval function" as a validated, transferable
   pattern rather than a from-scratch design question.
9. **When retuning an existing shipped constant's shape (not just its
   value), it's fine to leave the tunable's originally-declared `[min,max]`
   stale** — Integral's own `constants.h` has several tunables (`kSeDepth`,
   `kRevFutDepth`) whose current base *value* now sits outside the declared
   `[min, max]` from a prior tuning epoch, apparently harmlessly (the bounds
   are the SPSA search window at declaration time, not a runtime-enforced
   invariant). Low-priority, but worth knowing before treating a similar
   drift as a bug if/when our own future SPSA tooling accumulates the same
   history.

---

## Citation index

- `integral_reference/integral/src/engine/search/search.cc` — full file
  read; key line ranges: aspiration loop 92-140; `AdjustStaticEval` 220-245;
  qsearch 248-481; node entry/MDP/repetition 502-553; TT probe/cutoff
  555-585; Syzygy probe 587-638; static eval + material/correction/50-move
  241/236-237; improving/hindsight 677-720; razoring 741-748; RFP 723-737;
  NMP+verification 750-799; ProbCut 801-865; IIR 869-874; move-loop pruning
  (LMP/futility/history/SEE) 886-979; singular/multicut/negative-ext
  981-1030; LMR search+adjustments 1052-1142; node result/TT store/
  correction update 1229-1289. HEAD at time of study: `3f15ba6b` (2026-04-02,
  "Play quiet moves in QS if the TT move was quiet and not from a fail-low
  (#238)").
- `.../search/search.h:1-224` — `NodeType`, `SearchType`, `RootMove`/
  `RootMoveList` (39-101), `Thread` (103-154), `Searcher` (156-220).
- `.../search/stack.h:1-138` — `PVLine` (10-53), `StackEntry` (55-108,
  killer moves 73/81-91), `Stack` (110-134, `kPadding=6`).
- `.../search/constants.h:1-105` — every SPSA-tunable constant cited above,
  by name, with its declared `(value, min, max, step)`.
- `.../search/move_picker.h:23-72` — `MovePicker::Stage` enum (25-34,
  killer stages 29-30).
- `.../tuner/spsa.h:1-60` — `TUNABLE`/`TUNABLE_STEP` macros, the OpenBench-
  format print at lines 38-39/47-48.
- Git revisions cited (all in `integral_reference/integral`, `git show
  <rev>` or `git show <rev> -- src/engine/search/search.cc`):
  `8d6ed33b` (initial commit, 2023-10-25), `c03f4c45` (first "(N elo)"
  subject), `20e5da17` (#7, first PR-numbered + `Bench:`-trailered commit,
  2024-07-09), `0e06ea7c` (Singular Extensions, 21 elo), `049ebcee`
  (Multicut pruning, 7 elo), `2d852baa` (#12, Double Extensions,
  `double_extensions` counter introduced), `64dc7d94` (first negative
  extension, single `-1` tier), `fb46cfe9` (#47, folds `cut_node` into the
  same tier), `0995a2a4` (#49, `double_extensions` counter removed, replaced
  by `ply < 2·root_depth`), `9c25c954` (Split LMR into tactical/quiets, 15
  elo), `d3189f19` (#82, "Simplify out check extensions"), `ba78f44c`
  (Eval-based NMP reduction, 7 elo), `0f4ee79b` (#101, improving term added
  to NMP reduction), `ab39eb8d` (#103, NMP base divisor retuned), `aa618df6`
  (#40, "Only NMP when static eval is above beta"), `28182d36` (#36,
  ProbCut), `644beb5a` (#131, IIR shallow-ttMove disjunct), `b00fb202` (#89,
  Triple extensions), `8103098a` (#228, "Double extend in PV nodes"),
  `f6cbeea5` (#229, "Simplify LMR condition"), `453c1dc4` (#230, "Don't
  perform deeper/shallower adjustment at root"), `0f65df61` (#235, History
  Pruning reordered before SEE Pruning), `484db145` (#236, direct-check
  exemption in futility/ordering), `ea32ffc3` (#203, "LMP more often when
  static eval suggests a fail high"), `1568901f` (#234, "Use SF-style
  aspiration window adjustments", 2026-03-17), `d6ce508b` (#233, "Move
  material scaling to search"), `75d3c608` (#184, "Increase assumed fail
  high negative extension amount"), `3f15ba6b` (#238, HEAD at time of
  study).
- Cross-referenced own docs: `design/10-pruning-extensions.md` §3.3-§3.11,
  §4 (Wave A-G status); `design/07-search-core.md` §3.6 (aspiration
  windows); `design/decisions.md` D17/D20/D24/D27/D30/D31;
  `reckless_reference/07-search-node-pruning.md` (R3/R6/R7/R8/R9/R9b/R10,
  §5.1); `stockfish_reference/stockfish_search_notes.md`;
  `stockfish_reference/03-movegen-movepick-history.md:126`.
