# Reckless study — the negamax node: pruning, reductions, extensions

Source studied: `reckless_reference/Reckless` (crate `reckless`, `version =
"0.10.0-dev"`, edition 2024), commit as vendored. Primary files:
`src/search.rs` (1448 lines, read in full) and `src/parameters.rs` (34
lines, read in full). Followed into: `src/thread.rs` (`ThreadData` fields),
`src/stack.rs` (`Stack`/`StackEntry`), `src/types/score.rs` (`Score`
constants and predicates), `src/board.rs` (`material`, `all_threats`,
`is_direct_check`, `key_after`, `fiftymove_clock_bucket`), `src/board/see.rs`
(`see`), `src/movepick.rs` (`Stage`, `MovePicker::new`/`next`),
`src/evaluation.rs` (`correct_eval`), `src/types/piece.rs` (piece values).

Reckless is AGPLv3. Nothing below is copied code — every formula is
reproduced as hand-transcribed prose/arithmetic (own words), with at most
short expressions quoted verbatim where exactness matters for tuning
purposes. File:line citations point back at the vendored source for anyone
who wants to re-verify.

**Framing.** This note reads Reckless's node relative to two things we
already have: (1) `stockfish_reference/stockfish_search_notes.md` §4-§5 (the
SF node walkthrough and reduction function this project explicitly modeled
`design/10-pruning-extensions.md` on), and (2) our own
`design/10-pruning-extensions.md` + current `src/search/mod.rs` /
`src/search/params.rs`, which already implement a SF-faithful Wave A
(reduction table, RFP, NMP+verification, LMR core) and most of Wave B
(razoring, LMP, capture/parent futility, quiet SEE, IIR) per
`design/00-master-plan.md`'s P9 status. Wave C (statScore-coupled
reduction/pruning), Wave D (singular extensions/multicut), and Wave E
(ProbCut) are **not yet landed** in our engine — this is exactly the territory
where Reckless has the most to teach us, and where it also diverges hardest
from the SF shape our design doc committed to.

**Scale note (read this before comparing any raw constant):** Reckless's
internal piece values (`src/types/piece.rs:140-145`: P 109, N 403, B 435,
R 679, Q 1242) are roughly **half** Stockfish's (P 208, N 781, B 825,
R 1276, Q 2538, per our SF notes §8.2). Reckless's `eval`/`Score` numbers
therefore live on a compressed centipawn-like scale versus SF's. Any
side-by-side constant comparison below implicitly carries this ~2× factor —
called out inline where it matters, but keep it in mind generally.

---

## 1. Big-picture shape

Same skeleton as SF and as our design: `NodeType` marker trait
(`Root`/`PV`/`NonPV`, `search.rs:32-53`) monomorphizes one `search::<NODE>`
function (`search.rs:300`); PVS with a runtime `cut_node` bool; qsearch at
`depth <= 0` (`search.rs:320-322`). The iterative-deepening driver
(`start`, `search.rs:55-298`) is aspiration-window based and structurally
close to SF's (per-thread-less here — no `threadIdx % 8` jitter visible in
this excerpt of `start`; the diversity instead comes from the **LMR node/thread
jitter term**, §4 below — a very Reckless-specific substitution for SF's
aspiration-window jitter).

Two overarching **themes** that recur at almost every heuristic and are
worth stating up front because they're the most exportable ideas in the
whole file:

1. **Continuous signals over boolean gates.** Where SF multiplies a margin
   by a bool (`improving`, `opponentWorsening`) Reckless routinely uses the
   raw signed magnitude (`improvement`, `estimated_score - beta`, `history`)
   scaled by a `/1024`-ish divisor. RFP, NMP, and LMR all do this. It buys
   smoother tuning surfaces at the cost of one more multiply per node — cheap
   given these are all already-computed values.
2. **One "history" score computed once per move, reused everywhere.** Unlike
   SF/our design (which composes a different history sum at each pruning
   site — conthist-only for continuation-pruning, `+2·mainHistory` for the
   lmrDepth-buying step, `captHist` alone for captures), Reckless computes a
   single `history` value per move up front (`search.rs:767-772`) and feeds
   that same number into LMP, futility, "bad noisy" futility, history
   pruning, the SEE threshold, and LMR — see §5 for the exact composition.

---

## 2. Data structures (just enough to read the node)

- **`ThreadData`** (`thread.rs:213+`) is the per-search-thread god-object
  (board, stack, histories, TT handle, time manager, root moves). Relevant
  scalar fields used by this suite: `nmp_min_ply: i32` (NMP-disabled floor,
  same idea as our `nmp_min_ply`), `root_delta: i32` (`beta - alpha` at the
  aspiration window, `thread.rs:231`), `cutoff_count: PlyArray<i32, {MAX_PLY
  + 16}>` and `excluded: PlyArray<Move, ...>` (`thread.rs:240-241`) — both
  ply-indexed side arrays parallel to the search stack rather than fields
  *on* the stack entry (our design keeps `cutoff_cnt`/`excluded_move` as
  `Stack` fields instead — a structural choice, see §6).
- **`Stack`** (`stack.rs:5-33`) is a boxed fixed array of `StackEntry`
  (`MAX_PLY + 16` slots) plus a **shared sentinel row** (`sentinel: [[i16;
  64]; 13]`, `stack.rs:7`) that every entry's `conthist`/`contcorrhist` raw
  pointers point at by default (`Stack::new`, `stack.rs:15-23`) — so reads at
  negative ply (before the root, or right after a null move) land on a
  zeroed table instead of needing bounds checks or `Option`. `StackEntry`
  (`stack.rs:36-46`) carries `mv`, `piece`, `eval`, `tt_move`, `tt_pv`,
  `move_count: u16`, `reduction: i32`, and the two raw conthist pointers
  (`*mut [[i16; 64]; 13]`, wrapped in an `unsafe impl Send`). Indexing is
  `data[(index + 8) as usize]` (`stack.rs:66-79`) — an 8-slot negative-index
  offset, smaller than SF's 7/10 and our design's `SS_OFFSET = 7`; harmless,
  just a different constant.
- **`Score`** (`types/score.rs:4-70`): `NONE = 32002`, `INFINITE = 32001`,
  `MATE = 32000`, `TB_WIN = MATE_IN_MAX - 1`. `draw(td)` is
  **`(td.nodes() % 5) as i32 - 2`** (`score.rs:20-22`) — a ±2 node-count
  dither around zero, wider than SF's ±1 (`VALUE_DRAW - 1 + (nodes & 2)`).
  Same anti-3-fold-blindness idea, different amplitude.
- **`Bound`** (used via `transposition::Bound`) is matched as a plain 3-way
  `Upper | Lower | Exact` (`search.rs:389-393`, `Bound::Exact` compared
  directly at `search.rs:694`) rather than SF's bitwise `LOWER|UPPER=EXACT`
  encoding — singular-extension margin selection (§7) explicitly branches on
  `== Exact` vs not, which SF's design doesn't do.
- **`MovePicker`** (`movepick.rs`) stages `HashMove → GenerateNoisy →
  GoodNoisy → Quiet → BadNoisy` (`Stage` enum, `movepick.rs:10-15`).
  `MovePicker::new(tt_move, threshold)` (`movepick.rs:29-34`) takes an
  **optional SEE threshold** used only by ProbCut's picker; the main-search
  good/bad-noisy split falls back to `threshold = -entry.score/47 + 116`
  (`movepick.rs:64`) when `None` — a linear function of the move's own
  ordering score, not a flat divisor of victim value like SF's `-value/18`.

---

## 3. Node walkthrough, in execution order

Numbering below is descriptive, not Reckless's own (the source has no
`N0..N14` labels the way our design doc does) — it's chosen to line up with
`design/10-pruning-extensions.md §3.4-§3.10`'s N-numbers so the two can be
read side by side.

**R0/R1 — entry, stop flag, qsearch handoff, upcoming-repetition.**
`search.rs:307-330`. Standard: `depth <= 0 → qsearch`; the repetition draw
snap raises `alpha` to `draw_score` under `board.upcoming_repetition(ply)`.
No separate "init" comment block — `in_check`/`excluded` are computed as
locals (`search.rs:308-309`), not written into a `Stack` field the way our
`ss.in_check` is.

**R2 — mate distance pruning.** `search.rs:350-356`, byte-for-byte the same
shape as SF/our design (`alpha.max(mated_in(ply))`, `beta.min(mate_in(ply +
1))`).

**R3 — TT probe + early cutoff.** `search.rs:366-407`. The cutoff condition
splits the bound match explicitly by direction instead of SF's single
`cutNode == (ttValue >= beta) || depth > 4`:

```text
cutoff iff ALL of:
  1. stored depth clears the target: tt_depth > depth - [tt_score < beta]
     (the same one-ply rebate SF applies when the score would fail low)
  2. the stored score is a real value (not NONE)
  3. the BOUND admits it, split by direction:
       UPPER bound: needs tt_score <= alpha, and (non-cut node OR depth > 5)
       LOWER bound: needs tt_score >= beta,  and (cut node     OR depth > 5)
       EXACT: always admits
```

Same *shape* as our design's N6b, threshold `5` instead of SF's `4`; note
the depth-slack test uses **strict** `tt_score < beta` where SF/our design
use `tt_score <= beta` (an off-by-one-at-equality difference, `search.rs:387`
vs `design/10 §3.4` N6b). On a cutoff, if the ttMove is quiet and
`tt_score >= beta` and the *parent's* move count was `< 4`
(`search.rs:395`), it gets a bonus (`min(190·depth - 81, 1691)` quiet history
+ `min(96·depth - 73, 1206)` conthist) — **but there is no parent-malus half**
of SF's pairing (SF also penalizes the parent's own early quiet by −2187
conthist; Reckless only rewards the ttMove, `search.rs:395-401`). The rule50
guard is a flat `< 90` (`search.rs:403`) vs SF's `< 96`/our design's
`TT_CUTOFF_RULE50_LIMIT = 96`. **Absent entirely**: SF's depth ≥ 7
child-TT-verification probe and the "wrong-direction bound → `penalize(1)`"
branch (our design's N6b closing paragraphs) — Reckless's TT cutoff is a
strictly cheaper, less-hardened version of the same idea.

**R4 — static eval + correction + TT-substitution.** `search.rs:446-483`.
Three-way branch (in-check / excluded / TT-hit-or-miss) matches our design's
N6 shape exactly, including the "TT doubles as an eval cache" write on a TT
miss (`search.rs:461-466`, `TtDepth::SOME` not a real search depth). The
TT-value-as-tighter-bound substitution (`search.rs:470-481`) is identical in
spirit to SF's. `correct_eval` itself (`evaluation.rs:3-11`) is simpler than
SF's `evaluate()`: no separate psqt/positional complexity split, just
`(raw_eval·(21032 + material) + optimism[stm]·(1548 + material)) / 27015`,
then a flat `·(200 - fiftymove_clock)/200` rule50 damp, then `+=
correction_value`. `material` here is a **combined-both-sides** running
total (`board.rs:31`, decremented on every capture, incremented on
promotion delta, `board/makemove.rs:69-93`) — not a per-side non-pawn-material
count the way SF's `non_pawn_material(us)` is.

**R5 — quiet move-ordering eval-diff nudge.** `search.rs:490-502`. Same
shape as our design's N8, but gated `depth < 6 || entry.is_none()`
(`search.rs:496`) instead of SF's plain "fresh node" gate — an extra
depth-6 reapplication window not in our design doc; worth double-checking
against current SF before porting (may be SF-current and our notes stale, or
a Reckless addition — flag for the next SF re-sync).

**R6 — hindsight reductions.** `search.rs:504-516`:

```
eval_delta = eval + stack[ply-1].eval
reduction  = stack[ply-1].reduction        // raw milli-reduction, see §4!
if reduction >= 2249 && eval_delta < 0:            depth += 1
if !tt_pv && depth >= 2 && reduction > 0 && eval_delta > 57:  depth -= 1
```

Structurally matches our design's N6 hindsight adjustment
(`prior_reduction >= 3 && !opponent_worsening → depth+1`;
`prior_reduction >= 2 && ... → depth-1`) but reads **the raw
pre-division `r` value** stored at `search.rs:899`
(`td.stack[ply].reduction = reduction;`, before the `/1024`), not the
*actual applied ply delta* (`new_depth - reduced_depth`) the way SF's
`ss->reduction = newDepth - d` and our own `self.stack[cur].reduction =
new_depth - d` (`src/search/mod.rs:1501`) do. See §4/§6 — this is a real
semantic divergence, not just a units relabeling, because the milli-value
and the clamped ply-delta don't move together once the `clamp(1, new_depth +
2)` starts biting.

**R7 — razoring.** `search.rs:536-545`. Same shape as SF/our N9
(`estimated_score < alpha - 237 - 254·depth²`, no depth cap), with two
extra guards SF's razoring doesn't have: `alpha < 2048` (only razor when
alpha itself is a modest, non-mating value) and `!tt_move.is_quiet() &&
tt_bound != Lower` (don't razor away a position whose TT move is a quiet
move, or whose TT bound already says we fail high here). Uses
`estimated_score`, i.e. the TT-substituted value, where our design N9 uses
plain `eval` — see the scale note below (§7): Reckless is consistent about
substituted-value usage across razoring/RFP/NMP; SF (and our design,
citing it) is *not* — SF's NMP explicitly reads `ss->staticEval`, not the
substituted `eval`. This is a real, repeated choice-point, not a one-off.

**R8 — reverse futility pruning.** `search.rs:547-563`:

```
margin = 1140·depth²/128 - 120·improvement/1024 + 22·depth
        + 669·|correction_value|/1024
        - 54·(no piece of ours is currently threatened)
        - 19
margin = margin.max(2)
if !tt_pv && !in_check && !excluded
   && estimated_score >= beta + margin
   && !is_loss(beta) && !is_win(estimated_score):
       return lerp(estimated_score, beta, 0.6945)
```

Three things worth flagging hard:

1. **Quadratic-in-depth margin** (`1140·depth²/128 ≈ 8.9·depth²`), not SF's
   linear-with-cap `min(40 + 4·depth, 80)·depth` (our
   `RFP_MULT_BASE/SLOPE/CAP`). At depth 1 Reckless's raw quadratic term is
   tiny (~9) vs SF's ~40; by depth 10 it's ~890 vs SF's ~800 (comparable);
   past depth ~12 it keeps growing quadratically while SF's is capped at
   80·depth. Genuinely different margin *shape*, not just different
   constants — this changes how RFP behaves at the depth extremes.
2. **Threat-coupled term**: `- 54 * (all_threats() & colors(stm)).is_empty()`
   (`search.rs:556`) — RFP's margin shrinks by 54 (prunes *more* readily)
   specifically when **none of our own pieces are currently attacked**.
   `all_threats()`/`colors()` are incrementally-maintained board state
   (`board.rs:132-171`, `update_threats`, `board.rs:431-461`), not something
   computed on the fly. SF's RFP has no threat term at all. This is the
   single most interesting "history/threat-coupled pruning" find in RFP.
3. The return is `lerp(estimated_score, beta, 0.6945)` — see §8 for the
   `lerp` helper; numerically ≈ SF's `(716·beta + 308·eval)/1024`
   (0.6992/0.3008), i.e. the same blend, just parameterized as one float
   fraction instead of two integer weights.

**R9 — null move pruning.** `search.rs:565-629`. Gate:

```
cut_node && !in_check && !excluded && !potential_singularity
&& estimated_score >= beta + max(2,
     -9·depth + 110·tt_pv - 94·improvement/1024
     - 21·(cutoff_count[ply+1] < 2) + 337)
&& ply >= nmp_min_ply
&& board.material() > 491                      // combined-material floor
&& !is_loss(beta) && !is_win(estimated_score)
&& !(tt_bound == Lower && tt_move is a capture of a piece >= a knight)
```

Versus our design's N11 (`ss.static_eval >= beta - 14·depth - 45·improving +
374`), every one of these is a genuine addition:

- **`!potential_singularity`** — Reckless refuses to null-move-prune a node
  that's *also* a candidate for singular-extension analysis (same
  `potential_singularity` flag the SE block uses, `search.rs:518-522`,
  computed once before either block runs). SF has no such interaction; our
  design doesn't either.
- **`+ 110·tt_pv`** — extra margin (harder to trigger) specifically on
  ttPv nodes. SF's NMP margin has no ttPv term.
- **`- 21·(cutoff_count[ply+1] < 2)`** — a **cutoffCnt-coupled** margin
  discount: when the child hasn't already been failing high a lot, NMP
  fires slightly more easily. This is the "history-coupled pruning" pattern
  applied to NMP specifically, and SF doesn't do it (SF only uses cutoffCnt
  in the LMR reduction, never in the NMP eval gate).
- **the TT-capture veto** (`search.rs:581-583`) — refuse NMP outright if the
  TT already tells us there's a beta-beating capture of knight-or-better
  material sitting here. A "don't null-move into a known tactic" guard,
  absent from SF/our design entirely.
- **combined-material floor** (`board.material() > 491`, ≈ one knight on
  Reckless's scale) instead of SF's/our design's *per-side*
  `non_pawn_material(us) != 0` — cheaper (no side-conditional bitboard
  popcount) but coarser: a position where the side to move has zero
  non-pawn material but the opponent has plenty would still pass Reckless's
  gate.

Reduction: **`R = (4407 + 917·improving + 265·depth + 477·clamp(estimated_score
- beta, 0, 1187)/128) / 1024`** (`search.rs:587-588`) — i.e. `R ≈ 4.3 + 0.90
+ 0.26·depth` plies, plus up to `~4.4` more from an **eval-margin-scaled
term**. Our SF notes (§4, node step 11) explicitly say current mainline SF
dropped the eval-based R term ("no eval-based term currently" — flat `R = 7
+ depth/3`). Reckless brings that idea back in fixed-point form. Worth a
note for whoever next re-syncs the SF study: this looks like older-SF
folklore Reckless kept/revived, not something current SF does.

The null-search **window is TT-narrowed**: `bound = tt_score` if
`is_valid(tt_score) && beta > tt_score && tt_bound == Lower && depth - 2 <=
tt_depth`, else `beta` (`search.rs:598-602`); the null search runs at
`(-bound, -bound+1)`, not always `(-beta, -beta+1)`. This is new versus SF
and our design — when the TT already promises a lower bound below beta, the
null-move probe searches a tighter (cheaper, and arguably more informative)
window instead of the full beta window. Verification-search structure
(`nmp_min_ply` floor, `3·(depth-r)/4` recursion guard, `search.rs:604-628`)
otherwise matches SF's/our N11 exactly.

**R9b — ProbCut.** `search.rs:631-685`. Trigger:
`cut_node && !is_win(beta) && (tt_score valid ? tt_score >= probcut_beta &&
!decisive : eval >= beta) && !tt_move.is_quiet()`, `probcut_beta = beta +
254 - 85·improving`. The **candidate-verification loop is more elaborate**
than SF's fixed two-step (qsearch verify, then one real search at
`depth-4-improving`):

```text
step 1  cheap check: a zero-window qsearch at probcut_beta.
step 2  budget: the base verification depth is depth - 4 - improving
        (floored at 0), and the ACTUAL verification depth is docked by
        (qsearch surplus over probcut_beta) / 319 — the further above the
        window the qsearch landed, the less real depth is spent — clamped
        back into [0, base].
step 3  if the qsearch cleared the window and any real depth remains:
        RAISE the window by 197 per depth-point docked in step 2
        (capped below infinity) and run the zero-window real search there.
        - shifted-window search fails AND the window was actually raised:
          fall back to one more real search at the ORIGINAL window and the
          FULL base depth (the un-shifted re-verify).
        - otherwise: the raised window becomes the effective probcut_beta.
step 4  on success: store a LOWER bound at verification-depth + 1 and
        return the score — decisive scores raw, everything else pulled
        27% of the way toward beta (their 0.2695 lerp; SF returns raw).
```

The dynamic `probcut_depth`/`adjusted_beta` shift (how far above
`probcut_beta` the qsearch score landed determines both how much real-search
depth to spend and how much the window itself gets raised, with a fallback
re-verify at the *un*-shifted window if the shifted one fails) has no SF
equivalent in our notes. It's a genuinely more elaborate (and presumably
more accurate per node spent) ProbCut than SF's flat two-step. Since we
haven't implemented ProbCut at all yet (Wave E item 16), this is worth
building against directly rather than the plainer SF shape once we get
there.

**R9c — small in-check ProbCut** — **absent from Reckless.** SF's
"`probCutBeta = beta + 428`, in-check TT shortcut before the move loop"
(our design's N14) has no counterpart here; Reckless's ProbCut block
(R9b) requires `!in_check` implicitly (it sits in the same
"not in check" region as razoring/RFP/NMP, all skipped when `in_check` —
confirm at the top-level structure: there's no separate in-check jump target
the way SF's/our node has one; Reckless just runs the move loop directly
after ProbCut for in-check nodes since none of R7-R9b's individual
conditions can fire with `in_check` true, except NMP/ProbCut which
explicitly test `!in_check`... actually note R8/RFP tests `!in_check`
directly too, R7/razoring likewise — so the "skip to move loop" happens by
each guard testing `!in_check` rather than one shared early jump).

**R10 — singular extensions, multicut, negative extension, LDSE.**
`search.rs:687-736`. Computed **once, before the move loop**, not inside a
`move == tt_move` check the way SF/our design structure it — this works
because MovePicker always emits the TT move first when one exists
(`Stage::HashMove`, `movepick.rs`), so `extension` (applied only when
`move_count == 1`, `search.rs:839`) is equivalent to SF's per-move gate
without needing the comparison inside the loop. A clean control-flow
simplification worth copying regardless of the SE internals.

Trigger — `potential_singularity` (computed at `search.rs:518-522`):
`depth >= 5 + tt_pv, tt_depth >= depth - 3, tt_bound != Upper, tt_score
valid and non-decisive`. (One ply earlier than SF's/our `depth >= 6 +
tt_pv`; **no `is_shuffling` rule50-oscillation guard at all** — our design
already specifies `is_shuffling`, so we're already more thorough here than
Reckless, not behind it.)

```
singular_margin = (tt_bound == Exact ? ceil(depth/4) : depth)
                  + depth·(tt_pv && !PV)
singular_beta   = tt_score - singular_margin
singular_depth  = (depth - 1) / 2
value = search::<NonPV>(singular_beta - 1, singular_beta, singular_depth,
                        cut_node, ply)     // same ply, excluded_move = tt_move
```

The **bound-type-conditioned margin** (`depth/4` when the TT entry is
`Exact`, full `depth` — doubled to `2·depth` under `tt_pv && !PV` — when it's
only `Lower`) has no SF equivalent (our SF notes' formula is
`(60+70·(ttPv&&!PV))·depth/59` regardless of bound type). The intuition —
an `Exact` TT entry is fully trustworthy (previously resolved as a PV score)
so a *tighter* singularity test is warranted; a `Lower` entry is just a
fail-high, so demand a much bigger gap before calling the move singular —
reads as a genuine improvement on SF's shape.

Outcomes, same three-way structure as SF/our design:

- `value < singular_beta` → singular: `extension = 1 + (value < singular_beta
  - double_margin) + (value < singular_beta - triple_margin)`,
  `depth += 1` for the whole node (`search.rs:709-721`). Margins:
  `double_margin = 195·PV + 48·(PV && !tt_was_pv) - 16·tt_move.is_quiet() -
  16·|correction_value|/128`; `triple_margin` similar with different
  constants (`search.rs:710-716`).
- `value >= beta && !is_decisive(value)` → **multicut**: `return
  lerp(value, beta, 0.4027)` (`search.rs:723-724`) — no TT-move-history dock
  the way SF's multicut has (SF: `ttMoveHistory << (-442 - 108·depth)`,
  Reckless has no `tt_move_history`-equivalent scalar visible in this file at
  all — may live elsewhere or may simply not exist in this version).
- `else if singular_score > tt_score && stack[ply].mv != NULL: tt_move =
  Move::NULL` (`search.rs:725-727`) — a branch **not present in SF's design
  at all**: if the exclusion search's score beat the *original* ttScore
  (without beating beta outright), Reckless **discards the TT move for
  ordering/reduction purposes for the rest of this node** — presumably
  because the fact that excluding it produced a *better* score than the
  ttMove's own stored value casts doubt on the ttMove being genuinely best.
  Small but distinctive; worth understanding better before porting (may
  interact with move-picker internals not covered in this note's scope).
- `else if tt_score >= beta || cut_node: extension = -3` — SF/our design has
  two negative-extension tiers (`-3` on `ttValue >= beta`, `-2` on
  `cut_node` alone); Reckless collapses both to `-3` (`search.rs:729-731`).

**LDSE — "Low Depth Singular Extensions"**, `search.rs:733-736`, fires in
the **`else`** of the SE trigger (i.e., when the node *isn't* a
`potential_singularity` candidate at all):

```
else if depth <= 7 && !in_check && cut_node && estimated_score <= alpha - 25:
    extension = 1
```

**This has no Stockfish counterpart** in our notes. Reading it: at shallow
depth, in a cut-node whose overall position estimate already sits well below
alpha (by 25+), extend the first move (again, the TT move by ordering
convention) anyway by one ply — a cheap fallback singular-like extension for
exactly the shallow-depth regime where the real SE machinery's `depth >=
5+ttPv` gate can't fire (no TT entry deep enough yet). Genuinely novel,
cheap (one comparison), and directly portable once our own SE lands (Wave
D) — it needs nothing SE doesn't already need (`estimated_score`, `cut_node`,
`in_check`).

---

## 4. The move loop: per-move pruning

`search.rs:750-833`. Per move: `move_count += 1`; `is_quiet =
mv.is_quiet()`; `is_direct_check = board.is_direct_check(mv)` — note this is
the **cheap direct-check-only proxy** (`checking_squares[pt].contains(to)`,
`board.rs:427-428`, precomputed per-side as part of check-info bookkeeping),
**not** a full discovered-check-aware `gives_check` the way our design's
N-loop uses (`design/10 §3.5`: `gives_check = board.gives_check(m)`).
Reckless accepts missing the "don't prune a discovered check" exemption in
exchange for an O(1) bitboard test everywhere it needs "is this a check" in
the hot per-move loop (also reused in qsearch's LMP, `search.rs:1297`).

The unified **`history`** score (`search.rs:767-772`), computed once and fed
into every gate below:

```
history = is_quiet
    ? quiet_history[threats][us][mv] + conthist(ply,1)[mv] + conthist(ply,2)[mv]
    : noisy_history[threats][moved_piece][to][captured_type]
```

(`quiet_history`/`noisy_history` are threat-bucketed — `all_threats()` is an
extra index dimension into the butterfly/capture tables, a structural
extension of SF's plain `[color][move]` mainHistory that's really a
move-ordering/history-table topic, noted here only because this same
`history` value is what feeds every pruning formula below.)

Pruning block, gated `!ROOT && !is_loss(best_score)` (Reckless doesn't
explicitly re-test `non_pawn_material(us) != 0` here the way SF/our design
do — the material floor lives only in the NMP gate above):

- **LMP** (`search.rs:774-785`):
  `move_count >= (2818 + 78·improvement/16 + 1351·depth² + 74·history/1024) /
  1024`, gated `!in_check && !is_direct_check && is_quiet && !is_win(beta)`.
  Versus SF's/our `(3 + depth²) / (2 - improving)`: Reckless (a) uses the
  raw signed `improvement` instead of a boolean, and (b) **adds a
  history-coupled term directly to the move-count budget** — a move with
  good ordering history buys more tries before quiets get cut off entirely.
  No SF/our-design counterpart couples LMP's threshold to history at all.
- **Futility (parent, quiet)** (`search.rs:787-801`):
  `futility_value = eval + 79·depth + 55·history/1024 + 77·(eval>=beta) +
  555·|correction_value|/1024 - 127`; skip if `<= alpha` (gated
  `!in_check && !is_direct_check && is_quiet && depth < 14`). Same idea as
  SF's/our parent futility but built directly from raw `depth`
  (no `lmr_depth` surrogate at all — see the architecture note below) with a
  direct history term instead of history-buys-lmrDepth.
- **"Bad Noisy" futility (BNFP)** (`search.rs:803-816`) — **no SF
  counterpart as a separate named step**: `noisy_futility_value = eval +
  84·depth + 82·history/1024 + 24`; if `move_picker.stage() ==
  Stage::BadNoisy && noisy_futility_value <= alpha`, **`break`** (not
  `continue`!) the whole move loop. Exploits that `BadNoisy` moves are
  emitted in descending SEE/score order, so once the *best* remaining bad
  capture fails this test every worse one would too — cheap, correct, and
  something SF's/our design's plain capture-futility (gated only on
  `lmr_depth < 7`, `continue`d per move) doesn't exploit, because SF doesn't
  expose "which movepicker stage is this move from" as a pruning-relevant
  fact the way Reckless's `Stage` enum does.
- **History pruning (HP)** (`search.rs:818-821`) — a small **dedicated**
  quiet-history gate distinct from the continuation-pruning step SF/our
  design specify: `!in_check && is_quiet && depth < 5 && history < -948·depth
  → continue`. Cheap, shallow-only, uses the same unified `history` (main +
  cont(1) + cont(2)), not the conthist-only sum SF's continuation-pruning
  step (`history < -4313·depth`, our `QHIST_PRUNE_MULT`) uses.
- **SEE pruning** (`search.rs:823-832`), one formula for quiets and one for
  noisy, **both built from raw `depth`, no `lmr_depth`**:
  ```
  quiet_threshold = min(0, -12·depth² + 56·depth - 27·history/1024 + 27)
  noisy_threshold = min(0, - 7·depth² - 36·depth - 39·history/1024 + 14)
  if !in_check && !board.see(mv, threshold): continue
  ```
  Versus SF's/our design: quiet SEE margin is `-25·lmr_depth²` (built from
  the *predicted post-LMR* depth, which itself already absorbed a
  history-buys-depth adjustment, `design/10 §3.5`'s `LMR_DIVISOR` table);
  capture SEE margin is `-(175·depth + 34·capt_hist/1024)` (linear in raw
  depth, direct history term — closer to Reckless's noisy formula, but
  Reckless additionally makes it quadratic in depth). **Reckless has no
  `lmr_depth` concept anywhere in its move-loop pruning block at all** —
  every one of LMP, futility, BNFP, HP, and SEE reads straight off `depth`
  plus a direct, per-formula `history` term. This is the single biggest
  *structural* difference in the whole per-move pruning section: SF/our
  design funnel "how good is this move's ordering" through one intermediate
  quantity (`lmr_depth`) reused by several margins; Reckless instead lets
  each formula have its own independently-tuned linear/quadratic
  `depth`-and-`history` polynomial. Fewer moving parts to reason about per
  site, more independent knobs to tune overall — a real design tradeoff, not
  a strict improvement either way.
- One thing SF's design explicitly carries that this file's SEE step
  **omits**: the "don't prune a capture that sacrifices our last non-pawn
  piece when `alpha < 0`" stalemate-trap guard (our `CAP_SEE` exception,
  `design/10 §3.5`). Not present in Reckless's SEE-pruning line.

---

## 5. Make move, LMR — the biggest divergence from SF's shape

`search.rs:835-963`. This is where Reckless departs from SF/our design most
structurally: **there is no single, unified `r` computed once and then
applied two different ways** (SF's/our shape: compute `r` via `reduction()`
+ a long adjustment list in §3.7, *then* branch on whether LMR proper
applies). Reckless instead computes **two entirely separate `reduction`
locals**, one per branch, each starting from its own base term and
accumulating its own (similar-but-not-identical) adjustment list.

### 5.1 Reduction base: `ilog2(depth)`, not `reductions[depth]·reductions[move_number]`

LMR branch base: **`269 · depth.ilog2()`** (`search.rs:844`, integer
`u32::ilog2`, i.e. floor(log2(depth)) — a couple of CPU cycles, no table, no
float `ln`). Full-depth-search (non-LMR) branch base: **`207 ·
depth.ilog2()`** (`search.rs:918`). Compare to SF's/our
`reductions[depth]·reductions[move_number]` product of two `2834/128·ln(i)`
table lookups (`design/10 §3.3`): **Reckless's base term does not depend on
`move_count`/move-number at all.** Confirmed by grepping every use of
`move_count` in the file (`search.rs:395,487,761-762,780,839,843,917,...`)
— none of them feed a reduction-scaling multiplier; `move_count`'s only
reduction-adjacent role is the `>= 2` gate for LMR eligibility itself
(`search.rs:843`).

What replaces "later moves get reduced more"? **The per-move `history`
value**, added directly into each branch's reduction (`- 179·history/1024`
for quiets, `- 130·history/1024` for noisy in the LMR branch,
`search.rs:857,861`; `- 118·history/1024` / `- 63·history/1024` in the FDS
branch, `search.rs:925,928`). Since MovePicker already sorts moves by
(roughly) this same score, `history` is a **continuous, richer proxy** for
"how deep into the ordered move list are we" than a discrete move-number —
a move picked 8th with unusually good history gets less reduction than a
move picked 4th with bad history, which SF's `reductions[moveNumber]` table
can't express (it only knows position-in-list, not quality-at-that-position).
This is the single most exportable idea in the whole reduction section, and
directly usable once our Wave C (statScore/history-coupled reduction,
`design/10` §4 wave C item 10) lands — it would mean rethinking whether our
`reduction()` (`src/search/mod.rs:1742`, currently a straight port of SF's
2-D table, matching `design/10 §3.3`) keeps the move-number dimension at
all, or whether a future retune drops it in favor of a bigger history
coefficient. **Not a drop-in swap** — it's a different point in reduction-formula
design space from what our design doc already committed to and Wave A
already shipped: treat as an experiment for after Wave C, not a Wave A/B
patch.

### 5.2 The rest of the LMR-branch adjustment list (`search.rs:843-899`)

In our own words, all terms additive to the `269·ilog2(depth)` base unless
noted (units: 1024ths of a ply, same convention as SF/our design):

```
reduction -= clamp(425·improvement/128, -241, 1155)
reduction -= 3417·|correction_value|/1024
reduction += 1412·(bound == Exact)
reduction += 464·(tt_score valid && tt_score <= alpha)
reduction += 326·(tt_score valid && tt_depth < depth)
reduction += 1024·is_win(beta)
if is_quiet:
    reduction += 2171 - 179·history/1024
              + 418·clamp(alpha - estimated_score, -65, 91)/128
else:
    reduction += 1426 - 130·history/1024
if PV:      reduction -= 519 + 437·(beta-alpha)/root_delta
if tt_pv:   reduction -= 333 + 611·(tt_score valid && tt_score > alpha)
                              + 685·(tt_score valid && tt_depth >= depth)
elif cut_node:
            reduction += 1852 + 2204·tt_move.is_null()
if board.in_check():                reduction -= 955   // gives_check(mv), post-move
if cutoff_count[ply+1] > 2:
            reduction += 1151 + 400·(!PV && !cut_node)
if tt_move_score valid && singular_score valid:
    margin = tt_move_score - singular_score
    reduction += clamp(496·(margin - 185)/128, 0, 2021)
if !PV && stack[ply-1].reduction > reduction + 414:
    reduction += 136
reduction += ((nodes() + thread_id·27) % 128) - 59      // ± jitter, see 5.3
```

Three items worth calling out beyond the base-term point above:

- **`+ 418·clamp(alpha - estimated_score, -65, 91)/128`** (`search.rs:858`,
  quiet-only): reduces quiets *more* when the whole node's TT-substituted
  estimate already sits below alpha (position looks like it's failing low
  here already, so don't waste effort on quiets), and *less* when the
  position looks great. No SF/our-design counterpart — it's a node-level
  quantity (fixed for the whole node) folded into a per-move reduction,
  which is unusual (everything else in the list is either global-to-the-node
  constants or genuinely per-move quantities).
- **The `tt_move_score - singular_score` margin term** (`search.rs:886-889`,
  appearing in *both* the LMR and FDS branches with different constants,
  `search.rs:944-947`): after the TT move (move 1) has actually been
  searched, its realized score gets compared against the singular-verification
  score computed earlier (§3, R10) — the bigger the gap (TT move
  dramatically outperformed what the singular search predicted), the more
  *every subsequent move* gets reduced. This couples the singular-extension
  machinery's byproduct (a score at a shallower depth) into ordinary LMR for
  the rest of the node, which SF's design doesn't do at all (SF's SE margins
  only ever affect `extension`, never later moves' `r`). Needs our own SE
  (Wave D) to land before this becomes portable, but it's a genuinely novel,
  low-marginal-cost idea worth flagging now.
- **Node/thread jitter**: `((nodes() + id·27) % 128) - 59` (LMR branch,
  `search.rs:895`) and `((nodes() + id·26) % 128) - 56` (FDS branch,
  `search.rs:957`) — a small pseudo-random perturbation (range roughly
  ±60-68) added straight into the reduction, derived from the running node
  counter and thread id. SF gets its lazy-SMP search diversity from
  aspiration-window jitter (`threadIdx % 8` in the window delta) and doesn't
  perturb LMR at all; Reckless does both kinds of jitter absent from the
  aspiration loop excerpt we read (see §1) *and* this per-node LMR jitter.
  Deterministic given `(nodes, thread_id)` (so still bench-reproducible
  single-threaded, per D24's requirement) but a real behavioral difference
  from what we've built (our LMR has zero jitter today).

### 5.3 Applying `r`, deeper/shallower, PV re-search (`search.rs:895-973`)

```
reduced_depth = clamp(new_depth - reduction/1024, 1, new_depth + 2) + 2·PV
stack[ply].reduction = reduction          // the RAW milli-value, not new_depth - reduced_depth!
value = -search::<NonPV>(-alpha-1, -alpha, reduced_depth, cut_node=true, ply+1)
stack[ply].reduction = 0
if value > alpha:
    if !ROOT:
        new_depth += (value > best_value + 57)
        new_depth -= (value < best_value + 9)
    if new_depth > reduced_depth:
        value = -search::<NonPV>(-alpha-1, -alpha, new_depth, !cut_node, ply+1)
```

Matches our design's §3.8 shape closely (`+2·PV` instead of our/SF's `+1·PV`
— Reckless gives PV nodes a full 2-ply LMR floor bump, not 1; deeper/shallower
margins `57`/`9` vs our `LMR_DEEPER_MARGIN=52`/`LMR_SHALLOWER_MARGIN=9` — the
shallower margin matches exactly, the deeper margin doesn't). **Confirms
the §3 finding**: `stack[ply].reduction` is assigned the *raw pre-division*
`reduction`, not `new_depth - reduced_depth` — genuinely different from what
our own code does at the equivalent site
(`src/search/mod.rs:1501`: `self.stack[cur].reduction = new_depth - d;`,
matching SF's/our design's convention). No flat `+1415` conthist bonus on
the LMR move the way SF's design has (our Wave C item 10 is what would add
that on our side; Reckless's file has no equivalent bonus visible at either
LMR call site).

PV re-search (`search.rs:965-973`): `if PV && (move_count == 1 || value >
alpha)`, with the ttMove min-depth clamp simplified to `mv == tt_move &&
tt_depth > 1 → new_depth = new_depth.max(1)` — drops SF's extra
`(tt_score valid && decisive && tt_depth > 0)` alternative condition
entirely.

---

## 6. Node result, bookkeeping

`search.rs:975-1176`. Broadly the same shape as SF's/our design's §3.9-§3.10
(root effort bookkeeping, best-move/alpha/beta updates, `cutoff_count`
incrementing, 32-capacity quiet/noisy lists via `ArrayVec<Move, 32>`
matching our design's `SearchedMoves`). A few notable deltas kept brief
since move-ordering/history internals are a different note's territory:

- **No equal-value dither** the way SF's design has (`(nodes & 14) == 0`
  near-leaf coin-flip) — not found anywhere in this file.
- **Fail-high softening is absent as a separate step**; the softening
  instead happens *inline* at each individual return site via the shared
  `lerp` helper (RFP, multicut, ProbCut, qsearch stand-pat/fail-high) rather
  than once at node-result time the way SF's/our design's single
  `(bestValue·depth + beta)/(depth+1)` step does. Structurally: Reckless
  never returns a raw `beta`-exceeding value from *any* pruning return site;
  our design centralizes the same idea into one place (§3.10 step 1) plus
  qsearch's own two blends (§8 below) — six independent call sites of the
  same `lerp(value, bound, t)` idiom vs SF's one shared post-hoc squash. Both
  achieve the same "never return a raw unsoftened score" property.
- **`update_correction_histories`** (`search.rs:1385-1413`) divides the
  read-side sum by a flat **`/64`** (`eval_correction`, `search.rs:1364-1383`)
  versus SF's/our `/131072` — again, don't compare the two divisors
  numerically without also comparing the respective correction-table
  update-side bonus scales; the two systems are internally consistent but
  not on the same absolute footing.
- **The `lerp` helper** (`search.rs:1446-1448`):
  ```text
  lerp(a, b, t) over integer scores: widen a and b to f32, take one fused
  multiply-add of t onto the (b - a) span, truncate back to i32
  ```
  One tiny, reusable, `f32`-based helper (uses `f32::mul_add` for a fused
  multiply-add) standing in for what SF expresses as five-plus separate
  integer weighted-average expressions at five-plus call sites (each with
  its own two `/1024`-denominator integer weights). This is the clearest
  "more Rust-idiomatic, more maintainable" finding in the file — trivially
  portable to us regardless of anything else in this note (see §9).

---

## 7. Qsearch — brief, since SEE pruning is an explicit focus

`search.rs:1178-1362`. Same overall shape as SF's/our qsearch (TT probe,
stand-pat with TT-value substitution, one capture/promotion-only
MovePicker stage, LMP-of-2, futility, SEE, fail-high/stand-pat blends via
`lerp`). Two deltas worth recording because they land squarely on "SEE
pruning":

- **SEE threshold formula** (`search.rs:1302`):
  `!board.see(mv, (alpha - eval)/8 - min(|correction_value|, 68) - 74)`.
  Versus SF's/our qsearch SEE (`alpha - futility_base` where `futility_base
  = static_eval + 335`, i.e. effectively a flat `alpha - eval - 335`, no
  division): Reckless divides the `alpha - eval` gap by **8** before using
  it as (part of) the SEE threshold — a hugely gentler scaling that keeps
  the qsearch SEE bar close to 0 even when the position looks very lost,
  rather than SF's threshold growing linearly (and unboundedly negative,
  i.e., ever more permissive of losing captures) with the gap. A materially
  different qsearch-SEE aggressiveness curve, worth a note for whoever
  eventually retunes our qsearch SEE margin (`QS_SEE_FLOOR` neighborhood,
  `design/07`).
- **Stand-pat / fail-high blends** use the same `lerp` idiom:
  `lerp(best_score, beta, 0.8256)` on stand-pat (`search.rs:1271`) — a much
  stronger pull toward `beta` (82.6%) than SF's/our `(467/1024, 557/1024)`
  ≈ 54.4%; `lerp(best_score, beta, 0.5072)` on the search fail-high
  (`search.rs:1351`) ≈ SF's ≈53% (`QS_FAILHIGH_BLEND`, close match). The
  stand-pat blend in particular looks like a much bigger instability-damping
  choice than SF's/our current constant.

---

## 8. `parameters.rs` — a currently-orphaned SPSA scaffold

The entire file (`parameters.rs:1-34`) is two versions of a `define!` macro
— one (`cfg(not(feature = "spsa"))`) expands `type name: value;` pairs into
plain `const fn` accessors; the other (`cfg(feature = "spsa")`) expands the
same list into `static mut` globals plus `set_parameter`/`print_options` for
UCI-tunable SPSA runs (wired from `uci.rs:165-166,322-324`). **Grepping the
entire crate for `define!` invocations finds none** — the macro is declared
but never called anywhere, in either build configuration. Every one of the
~150+ numeric constants cited throughout this note is a **bare literal
inlined directly in `search.rs`'s expressions**, not routed through this
scaffold at all. Read together with `version = "0.10.0-dev"` and `spsa = []`
sitting unused in `Cargo.toml:14`, this reads as a mid-development,
between-tuning-cycles state — plausibly constants get inlined for a
final/production build after an SPSA session elsewhere, or the macro is
reserved for a different subsystem (eval?) not covered by this note's
file list. Either way: **our own D20 discipline (`design/10 §3.1`: every
tuned number lives in a named `pub const` in `src/search/params.rs`, `grep
-nE '[0-9]{3,}' src/search/search.rs` returns nothing) is already stricter
and more consistently applied than what Reckless's current tree actually
does** — nothing to import here, just a confirmation we're ahead on process.

---

## 9. Adapt to us

Ordered roughly by effort-to-impact; each tagged with the design doc/section
and roadmap phase it touches.

1. **`lerp(a, b, t)` helper for every "blend toward the bound" return.**
   `design/10-pruning-extensions.md §3.4` (RFP return blend) and
   `design/07-search-core.md §3.7.2` (qsearch stand-pat/fail-high blends)
   already specify these returns as two-integer-weight blends
   (`RFP_RET_BETA_W`/`RFP_RET_EVAL_W`, `QS_STANDPAT_BLEND`,
   `QS_FAILHIGH_BLEND`). Adding one `fn lerp(a: Value, b: Value, t: f32) ->
   Value` in `src/search/mod.rs` (or `params.rs`) and expressing each site as
   `lerp(x, bound, T)` for a single named `f32` constant per site is a pure
   refactor — same bench count, clearer call sites, one idiom instead of N
   ad hoc integer-weight pairs. Also directly useful for Wave D's multicut
   return and Wave E's ProbCut return once those land (both are `lerp`
   sites in Reckless). **Effort: low (an hour). Impact: readability/hygiene
   only, no Elo claim** — good small PR independent of any wave.
2. **Threat-coupled RFP margin term.** `design/10-pruning-extensions.md §3.4`
   N10 (`RFP_*` params) currently has no threat term. Reckless's `- 54·(no
   piece of ours threatened)` (`search.rs:556`) needs an incrementally
   maintained "am I currently threatened" bitboard/bool — our board layer
   doesn't have Reckless's `all_threats()`/`piece_threats()` state yet
   (design/09's `threats_by_lesser` is computed on the fly for move
   ordering, not persisted as board state). **Effort: medium** (needs the
   threat-bitboard machinery first, which design/09 already gestures at as a
   possible future `Board` promotion, §"threats_by_lesser home" open
   question) **— Impact: unclear until SPRT'd, but cheap to test once the
   threat state exists.** Tag: Wave F (correction/threat couplings) or a
   new Wave item, `design/10 §4`.
3. **cutoffCnt-coupled NMP margin.** `design/10 §3.4` N11 has no
   `cutoff_cnt` term at all (our `cutoff_cnt` array already exists per the
   Stack sketch in `design/10 §3.2`, just unused by NMP). Adding `- K·(
   (ss+1).cutoff_cnt < 2)` to the NMP eval-margin gate is a one-line,
   already-available-data change once NMP itself is done (it already is,
   Wave A item 3). **Effort: low. Impact: small (SF doesn't do this, so
   it's genuinely untested folklore) — SPRT at `[0, 3]`** like our other
   Wave-E-scale refinements.
4. **TT-capture veto for NMP** (`search.rs:581-583`: don't null-move-prune
   when the TT already promises a beta-beating capture of knight-or-better
   material). One-line addition to `design/10 §3.4` N11's gate, needs only
   `tt_bound`/`tt_move`/`piece_on(tt_move.to())`, all already available at
   that point in our node. **Effort: low. Impact: small, untested outside
   Reckless — SPRT `[0, 3]`.**
5. **TT-narrowed NMP window** (`bound = tt_score` instead of always `beta`
   when a fresher `Lower` TT entry sits below beta, `search.rs:598-602`).
   Slightly more involved (needs the `depth - 2 <= tt_depth` freshness check
   plumbed alongside the existing NMP block) but still local to
   `design/10 §3.4` N11. **Effort: low-medium. Impact: unclear, likely
   small — SPRT `[0, 3]`.**
6. **LDSE ("Low Depth Singular Extensions")** — a one-line extension
   (`depth <= 7 && !in_check && cut_node && estimated_score <= alpha - 25 →
   extension = 1`) that fires in the *else* branch of the singular trigger,
   needing nothing beyond what Wave D (singular extensions,
   `design/10 §3.6`) already brings in. **Effort: low once Wave D lands.
   Impact: small but plausible (fills the shallow-depth gap where full SE
   can't trigger) — SPRT `[0, 3]`.** Tag onto Wave D as an extra item.
7. **Bound-type-conditioned singular margin** (`depth/4` on `Exact`, `depth`
   — doubled under `tt_pv && !PV` — on `Lower`, vs our design's single
   `(60+70·(ttPv&&!PV))·depth/59` formula regardless of bound). Requires our
   TT's `Bound` to be checked for `== Exact` specifically at the SE call
   site — already possible, our `Bound` enum (per `design/08`) presumably
   supports the equality check. **Effort: low (once Wave D's SE skeleton
   exists). Impact: plausible improvement on singularity-test precision —
   worth an isolated SPRT once Wave D lands**, `design/10 §3.6`.
8. **Elaborate dynamic-depth/dynamic-beta ProbCut re-verify** (§3 R9b above)
   — build our Wave E ProbCut (`design/10 §3.4` N13) against this shape
   instead of SF's flatter two-step, since we haven't shipped anything yet
   and this is directly at the point of first implementation. **Effort:
   medium (more branches than SF's version, but same primitives — qsearch
   probe, real search, TT store). Impact: unclear vs the simpler SF shape;
   worth SPRT-ing both if time allows, otherwise ship the more elaborate one
   since a top-2 engine settled on it.** Tag: Wave E item 16.
9. **History-coupled LMP threshold** (`move_count >= (2818 + 78·improvement/16
   + 1351·depth² + 74·history/1024)/1024` vs our design's plain
   `(3+depth²)/(2-improving)`, `design/10 §3.5`). Needs the per-move
   `history` value computed before the LMP check runs (currently our LMP
   check precedes history computation in the move loop per our own code
   order — would need reordering). **Effort: medium (reorder + new
   composite formula + retune). Impact: plausible (Reckless's own SPSA
   presumably validated this shape) — tag Wave C item 11 (history-coupled
   pruning) since it's exactly that wave's theme**, `design/10 §4` Wave C.
10. **Reconsider the reduction base term's move-count dependence** (§5.1) —
    the deepest, riskiest, and potentially highest-value idea: after Wave C
    lands (statScore/history plumbed into the reduction formula), run an
    isolated experiment replacing (or supplementing) our `reductions[depth]
    · reductions[move_number]` product (`design/10 §3.3`, already shipped in
    Wave A) with a `depth`-only `ilog2` base plus a bigger direct-history
    coefficient, mirroring Reckless's shape. **This is a genuine
    architecture change to a component we already shipped and SPRT-gated —
    do not fold into Wave C's normal item list; treat as a standalone,
    clearly-labeled experiment** (revert-friendly, single commit, its own
    SPRT) after Wave C's simpler history-coupling items land, so we have the
    `history` plumbing in place to feed it fairly. **Effort: medium-high.
    Impact: potentially the largest single idea in this note, also the
    least certain — it's a different point in design space, not a strict
    superset of what SF/our design already does.**
11. **Direct-check-only proxy for the pruning-exemption checks**
    (`is_direct_check` instead of full `gives_check`, §4). Our design
    already has a full `Board::gives_check` (per project status: "P2 ...
    `gives_check(m)` for every move kind"). Swapping the *pruning-exemption*
    call sites specifically (LMP/futility "don't prune a checking move") to
    a cheap direct-check bitboard test instead of the full function is a
    pure micro-optimization (fewer moves get the "it's a check" exemption —
    only discovered checks lose it — in exchange for O(1) vs a real
    function call in the hottest loop in the engine). **Effort: low (if we
    already have or cheaply can add a `checking_squares`-style precomputed
    table — check `design/03-attack-tables.md`/`design/04` for whether
    check-info bookkeeping already exposes this). Impact: nps only, not
    strength — verify with the bench fingerprint (D24), not an SPRT.**
12. **Node/thread LMR jitter** (§5.2, `((nodes()+id·27) % 128) - 59`) — a
    cute idea for lazy-SMP diversity once we reach P11 (per
    `design/00-master-plan.md`'s ordering, threading is P11, well after P9's
    pruning waves) but pointless to add single-threaded now: it would only
    ever perturb bench-fingerprint reproducibility (D24) for zero benefit
    before we have multiple threads to diversify. **Defer explicitly to
    P11 (`design/13-threading-smp.md`)**, flag now so it isn't reinvented
    from scratch later.

**Explicitly not recommended to port as-is:** the raw-`depth`-instead-of-
`lmr_depth` restructuring of the whole move-loop pruning block (§4) — our
design doc already committed to, and Wave A/B already shipped, the SF-faithful
`lmr_depth`-mediated shape; swapping architectures mid-wave would be design
churn without a clear win, and the two are genuinely different points in
design space rather than one strictly dominating the other. Worth
revisiting only as a deliberate, isolated experiment (paired with item 10
above) after Waves C-E are done and stable.

---

## 10. Citation index (file:line)

| Topic | Reckless location |
|---|---|
| `NodeType` trait, `Root`/`PV`/`NonPV` | `search.rs:32-53` |
| Iterative deepening / aspiration loop | `search.rs:55-298` |
| `search::<NODE>` entry, MDP | `search.rs:300-356` |
| TT probe + early cutoff + quiet bonus | `search.rs:366-407` |
| Static eval, TT-substitution, `improving` | `search.rs:446-534` |
| Eval-diff quiet-ordering nudge | `search.rs:490-502` |
| Hindsight reductions | `search.rs:504-516` |
| `potential_singularity` flag | `search.rs:518-522` |
| Razoring | `search.rs:536-545` |
| Reverse futility pruning | `search.rs:547-563` |
| Null move pruning + verification | `search.rs:565-629` |
| ProbCut | `search.rs:631-685` |
| Singular extensions / multicut / LDSE | `search.rs:687-736` |
| Move-loop pruning (LMP/futility/BNFP/HP/SEE) | `search.rs:741-833` |
| `make_move` + LMR reduction (both branches) | `search.rs:835-963` |
| PVS re-search | `search.rs:965-973` |
| Root bookkeeping | `search.rs:975-1014` |
| History/stat updates at node result | `search.rs:1054-1151` |
| TT store, correction-history gate | `search.rs:1153-1176` |
| `qsearch` | `search.rs:1178-1362` |
| `eval_correction` / correction-history update | `search.rs:1364-1413` |
| `update_continuation_histories` | `search.rs:1415-1422` |
| `make_move`/`undo_move`/`lerp` helpers | `search.rs:1424-1448` |
| `parameters.rs` macro (unused) | `parameters.rs:1-34` |
| `Score` constants/predicates, draw dither | `types/score.rs:4-70` |
| `Stack`/`StackEntry` | `stack.rs:1-81` |
| `ThreadData` scalar fields | `thread.rs:213-294` |
| `Board::material/all_threats/colors/is_direct_check` | `board.rs:132-171,427-461` |
| `Board::see` | `board/see.rs:12` |
| `correct_eval` | `evaluation.rs:3-11` |
| Piece values | `types/piece.rs:138-146` |
| `MovePicker`/`Stage`, good/bad-noisy split | `movepick.rs:10-99` |
| SPSA feature flags | `Cargo.toml:12-14` |
