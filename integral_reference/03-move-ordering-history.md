# Move ordering & history/correction (Integral)

Source: `src/engine/search/move_picker.{h,cc}` (~320 lines total),
`src/engine/search/history/{history.h,bonus.h,quiet_history.h,capture_history.h,
continuation_history.h,continuation_entries.h,pawn_history.h,
correction_history.h}` (~470 lines total), plus the call sites in
`src/engine/search/search.cc` (`PVSearch`/`QuiescentSearch`) and the threat/
check-zone machinery in `src/chess/board.{h,cc}`. Git history walked via
`git -C integral_reference/integral log --oneline -- src/engine/search/history
src/engine/search/move_picker.cc` (669 commits total in the repo; ~55 touch
this subsystem, spanning 2024-06 to 2026-04). Ideas, constants, and measured
deltas only — **no Integral source reproduced** (no explicit license =
all-rights-reserved by default, stricter than our GPL/AGPL handling of SF/
Reckless, D27).

Compare against: `stockfish_reference/03-movegen-movepick-history.md` +
`stockfish_reference/stockfish_search_notes.md` §8-10, `reckless_reference/
05-movepick-history.md`, our `design/09-move-ordering-history.md`,
`src/movepick.rs`, `src/history.rs`.

## 1. The big picture: a lean 8-stage single pipeline, killers included

Integral runs **one** `MovePicker` with a single 8-variant `Stage` enum
(`move_picker.h:25-34`): `kTTMove → kGenerateNoisys → kGoodNoisys →
kFirstKiller → kSecondKiller → kGenerateQuiets → kQuiets → kBadNoisys`. There
is no separate Evasion/ProbCut/Qsearch pipeline the way Stockfish/our design
run four (16 stages total, design/09 §3.5): mode is two constructor
parameters instead — `see_threshold` (0 for main search, a real value for
ProbCut) and `force_evasions` (a bool set by the *caller* to keep emitting
even after the noisy stage would otherwise return early in qsearch,
`move_picker.h:41-42`, wired at `search.cc:360-364`). This is structurally
the same simplification Reckless makes (`reckless_reference/05` §1: one
5-variant enum + `threshold: Option<i32>` + `skip_quiets`), just with two
extra explicit killer stages Reckless doesn't have at all. Net stage count:
Integral 8, Reckless 5, SF/us 16 — **both top engines studied so far collapse
SF's four-pipeline design into one**, only disagreeing on how many stages
that one pipeline needs.

**Killer moves are alive and still tuned.** `StackEntry::killer_moves:
std::array<Move, 2>` (`stack.h:73`) is written on every beta cutoff
(`stack->AddKillerMove(move)`, `search.cc:1199`, only for quiet best moves),
cleared one ply ahead of recursion (`(stack + 1)->ClearKillerMoves()`,
`search.cc:713` — the standard "clear the child's slot before descending, not
the current one" pattern so a stale killer from a sibling subtree at the same
ply can't leak in from two iterations ago), and consulted in the picker's
`kFirstKiller`/`kSecondKiller` stages ahead of quiet generation (`move_picker.
cc:94-118`) with a dedup filter against the TT move and against noisy killers.
Commit `8b017b02` ("Prevent noisy moves from being played as killer moves",
Mar 2025, `Bench: 2400341`) hardened the noisy-killer exclusion — evidence
this exact mechanism was still being *measurably* tuned less than a year
before this study, not a vestigial holdover. Killers also feed **LMR**
directly: `search.cc:1096-1099`, `reduction -= kLmrKillerMoves` (tunable,
current value 886/2048, `constants.h:94`) when the move being searched
matches either killer slot — a use SF dropped when it removed killers
entirely and Reckless never had. **This directly contradicts our D17**
("no killer/countermove tables, ever ... we skip that detour entirely") —
see §7/Adapt-to-us item 1.

### 1.1 Scoring formulas

**Noisy** (`move_picker.cc:189-214`): promotions get a fixed top-of-list
score (`1e9 - 1` queen, `1e9 - 2` knight, `-1e9` everything else — under-
promotions are effectively last); captures score `100 * PIECE_VALUE[victim] +
capture_history[turn][attacker][to][victim]` — pure MVV+history, no LVA term,
matching SF's/our design's shape (design/09 §3.7) but Integral's piece values
(`kPawnScore..kQueenScore = 102, 301, 281, 523, 889`, `move_picker.cc:7-11`,
themselves SPSA-tunable) are roughly half SF's board-tuned set, same as
Reckless's independently-tuned values.

**The good/bad noisy split is a *history-adjusted* SEE gate**
(`move_picker.cc:78-84`):
```
loses_material = !SEE(move, see_threshold - capture_history_score / 92, state)
```
i.e. a capture with strong capture-history is allowed to lose up to roughly
capture-history/92 (raw eval centipawns) of material and still count as
"good"; a capture with bad history needs to *win* material by that much more.
Same idea as SF's `-value/18` and Reckless's `-score/47 + 116` (design/09
§3.6 step 3, `reckless_reference/05` §1.5) — three independently-built
engines converge on "let the SEE gate flex by the move's own history score,"
just with three different divisors (18 / 47 / 92) reflecting each engine's
own capture-history magnitude scale. Under-promotions are excluded from the
"good" set even when SEE-legal (`!move.IsUnderPromotion()`, `move_picker.
cc:80`).

**Quiet** (`move_picker.cc:216-246`): three additive terms beyond the
blended history score (`history_.GetQuietMoveScore`, §2 below):
- A **threat escape/entry term**, structurally identical to SF's/our
  `threats_by_lesser` and Reckless's `threatened[pt]` (design/09 §3.7,
  `reckless_reference/05` §2): queen vs rook-or-lower threats (±19451/17612),
  rook vs minor-or-pawn threats (±13177/13758), minor vs pawn threats
  (±7852/8475) — all four SPSA-tunable and **asymmetric** (the "flee" bonus
  and the "step into" penalty are independently tuned, not a single signed
  constant like SF's/our `20 * PieceValue[pt]`).
- A flat **direct-check bonus**, `2048 * MoveGivesDirectCheck(move)`
  (`move_picker.cc:240`), unconditional — no SEE gate, unlike SF's/our
  `16384` bonus gated on `see_ge(m, -75)` (design/09 §3.7). `MoveGivesDirectCheck`
  (`board.cc:468-481`, landed by commit `484db145`, Mar 2026 — one of the
  *most recent* commits in the whole repo) tests the moving piece's target
  square against a per-piece-type "check zone" superpiece bitboard from the
  enemy king (`state.check_zones[pt]`, computed once per node alongside pins
  in `CalculateKingThreats`, `board.cc:582-591`) — the same asset is reused
  to **exempt** direct-checking quiet moves from futility pruning in the same
  commit (`search.cc:949-951`, `!board.MoveGivesDirectCheck(move)` added to
  the FP guard). One bitboard test, two call sites (ordering + pruning) — a
  clean compute-once/use-twice pattern worth copying independent of its
  specific tuning.

## 2. `stack->history_score`: one blended number, read by five different heuristics

`History::GetMoveScore`/`GetQuietMoveScore` (`history.h:40-62`) combines
**five weighted terms** into a single `I32`, scaled back down by
`kHistoryWeightScale = 1024`:
```
score = ( 1055 * quiet_history[stm][from][to][threat(from)][threat(to)]
        + 1275 * cont_history(ply-1)
        +  974 * cont_history(ply-2)
        +  910 * cont_history(ply-4)
        + 1036 * pawn_history[pawn_key][piece][to]
        ) / 1024
```
(quiet weight `kQuietHistoryWeight`, continuation weights
`kFirstContinuationHistoryWeight`/`kSecondContinuationHistoryWeight`/
`kFourthContinuationHistoryWeight`, pawn weight `kPawnHistoryWeight`, all
SPSA-tunable, `history.h:13-19`). Note **only offsets 1, 2, 4 are read for
scoring** — offset 6 exists in the table (§3) and is written, but never read
here; see §3 for why.

The distinctive part: this exact blended value is computed **once per move,
before making it** (`search.cc:903`, main search; `search.cc:830`, ProbCut)
and stashed in `stack->history_score`, then reused — unmodified — as the
single "how good has this move type historically been" signal by **five
separate pruning/reduction decisions** at the same node:
1. LMR reduction (`search.cc:1080-1084`, `1081`/`1083`: different divisors
   for quiet vs. capture, `kLmrHistDiv=11680` / `kLmrCaptHistDiv=11008`).
2. Root/pre-move LMR-depth estimate for LMP/FP thresholds
   (`search.cc:918`).
3. Futility-pruning margin widening (`search.cc:948`,
   `/ kFutMarginHistDiv = 124`).
4. History pruning's own cutoff test (`search.cc:961`, compared directly
   against a depth-scaled `history_margin`).
5. The noisy SEE-pruning threshold (`search.cc:973`,
   `/ kSeePruneHistDiv = 131`).
6. (Bonus, at the *parent* node) Reverse futility pruning's margin, reading
   the **previous ply's** stashed value: `(stack - 1)->history_score /
   kRevFutHistoryDiv` (`search.cc:732`, div `586`) — "if the move that got us
   here had strong history support, trust this position's eval a bit less
   defensively" (a smaller subtraction from the RFP margin).

This is architecturally tidy: compute the blended ordering signal once,
reuse it as a scalar everywhere a pruning heuristic wants "is this kind of
move usually good," rather than re-deriving or re-reading history tables at
each guard (which is what SF's `statScore`/our design's per-site reads
already do individually — the difference here is that *all* of Integral's
guards share literally the same precomputed field, computed from the exact
same function the picker itself uses to order moves, whereas our design/09
§3.9 computes ordering scores and pruning-guard statScore from separately-
timed reads). Cheap, correct only because pruning-guard reads happen
*before* the move is played and the field isn't clobbered until the next
move's `GetMoveScore` call in the same loop iteration — a fragile-looking but
correct single-owner-per-node data-flow that would need a code comment in a
review context.

## 3. History tables — shapes, bounds, and a shared global gravity constant

| Table | Shape (paraphrased) | Init | Notes |
|---|---|---|---|
| `QuietHistory` | `[turn][from][to][from∈threats][to∈threats]` → `I16` | 0 | `quiet_history.h:12-53`; the from/to-threatened axes are baked into storage (see §6), landed early (commit `296901c3`, Jul 2024) |
| `CaptureHistory` | `[turn][attacker_piece][to][captured_piece]` → `I16` | 0 | `capture_history.h:10-49`; no threat axis (unlike Reckless's `NoisyHistory`) |
| `PawnHistory` | `[pawn_key & 16383][turn][piece][to]` → `I16` | **−919** (`kPawnHistFill`, tunable) | `pawn_history.h:12-57`; landed commit `7578fe26` (Dec 2024) as an *addition* alongside `QuietHistory`, not a replacement — both exist simultaneously and are separately weighted in `GetMoveScore` |
| `ContinuationHistory` | outer `[turn][piece][to]` → inner `[turn][piece][to]` → `I16` | 0 | `continuation_history.h:11-86`; read offsets {1,2,4} (§2), **write** offsets {1,2,4,6} (§4) |
| `CorrectionHistory` (unified) | `pawn`/`major` tables `[key & 16383][turn]`; `non_pawn` table `[key & 16383][turn][color]`; all `I16` | 0 (bonus-gated writes keep it near 0 until real signal accrues) | `correction_history.h:15-148`; **no minor-piece key** — see §5 |
| `ContinuationCorrectionHistory` | `[turn][piece][to]` → `[turn][piece][to]` → `I16`, addressed by *stack ply-2 and ply-3* | 0 | `correction_history.h:50-58`, `102-106`, `146-147`; note the **{2,3} offset pair**, not SF's/our {2,4} — see §5 |
| low-ply history | absent | — | matches Reckless; SF/us have it |
| tt-move reliability scalar | absent | — | matches Reckless; SF/us have it |
| countermove table | absent | — | matches SF-current, Reckless, and us (D17) |
| killer moves (2/ply) | **present** | — | matches SF-*historical*/many engines; absent from SF-current, Reckless, and us (D17) — see §1 |

All tables are plain (non-atomic) per-search-thread state owned by a
`history::History` struct of five `unique_ptr`s (`history.h:69-75`),
reinitialized wholesale on `ucinewgame` via `Initialize()`
(`history.h:23-38`) — **no per-`go` decay step anywhere** in this subsystem
(same finding as Reckless, `reckless_reference/05` §3.3; opposite of SF's/
our design's 789/1024 per-`go` butterfly decay + low-ply refill,
design/09 §3.4).

### 3.1 The gravity operator: one *shared* constant across four tables, not a per-table bound

The write primitive (`bonus.h:29-31`) is the same self-saturating formula as
SF's/ours/Reckless's:
```
new_score = bonus - old_score * |bonus| / gravity
```
— algebraically, repeated same-sign bonuses converge the entry toward
`±gravity` regardless of the bonus's own magnitude (the fixed point of
`v = v*(1 - bonus/gravity) + bonus` is `v = gravity`). **The structural
difference from SF/our design**: SF/us give *each table its own bound* as a
const-generic type parameter (`StatEntry<const D: i32>`, design/09 §3.3 —
`MAIN_D=7183`, `CAPT_D=10692`, `CONT_D=30000`, `PAWN_D=8192`). Integral
instead defaults `ScaleBonus`'s `gravity` parameter to **one single tunable
constant, `kHistBonusGravity = 11422`** (`bonus.h:9,29`), shared by
`QuietHistory`, `CaptureHistory`, `PawnHistory`, and `ContinuationHistory`
alike (each calls `ScaleBonus(score, bonus)` with no override,
`quiet_history.h:19`/`capture_history.h:21,35`/`pawn_history.h:22`/
`continuation_history.h:80`) — only `CorrectionHistory::UpdateTableScore`
overrides it to `1024` (`correction_history.h:116`, matching SF's/our
correction bound exactly). The *per-update bonus* is separately capped by
`HistoryBonus`/`HistoryPenalty` (`bonus.h:15-26`,
`clamp(scale*depth [- bias], ±kHistBonusMaxBonus)`, current
`kHistBonusMaxBonus = 1294`) — this is the same two-tier structure SF/we use
(a small per-node bonus cap, a much larger asymptotic table bound), just with
**one shared asymptotic bound across four structurally different tables**
instead of four separately-tuned ones. Given continuation history in SF/our
design gets 4× the headroom of the main table (30000 vs 7183) specifically
because it's read far more selectively (per predecessor-move context, not
globally), Integral's unification is a real design bet, not an oversight —
and one only visible by reading `bonus.h` and every table's call site
together, since no single file states it. `bonus.h:15-18` also carries a
**flat bias subtracted from the bonus itself**, `kHistBonusBias = 126`
(landed commit `007ca05d`, "Add a static bias to history bonuses", Nov
2024) — at low depth this can make `HistoryBonus(depth)` computed for
depth 0 come out *negative*; in practice it's never called at depth 0
outside the fixed `UpdateScore(state, move, 1)` qsearch capture-cutoff bonus
(`search.cc:427`, `QuiescentSearch`), which nets `168*1 - 126 = 42`, a small
but positive bonus — the bias exists to blunt the *smallest* legitimate
bonuses, not to invert them.

### 3.2 Continuation history write: a "total score" gravity shared across offsets

`ContinuationHistory::UpdateMoveScore` (`continuation_history.h:30-42`,
landed by commit `b9905bb3`, "Update continuation history with total score
as the base influence", Nov 2025, `Bench: 1708751` — recent, ~8 months
before this study) does something neither SF, Reckless, nor our design does:
before writing anything, it first **reads and sums** the current scores at
all four write offsets {1,2,4,6}:
```
total = cont(ply-1) + cont(ply-2) + cont(ply-4) + cont(ply-6)
```
then feeds that combined `total` — not each offset's own individual score —
as the *gravity-scaling reference* into `ScaleBonus(total, bonus)` for
**every** one of the four writes (`continuation_history.h:38-41`,
`UpdateIndividualScore`, passed the same `total_score` argument each time).
Since `ScaleBonus(score, bonus) = bonus - score * |bonus|/gravity`, this
means each individual entry's own damping is driven by how *saturated the
whole predecessor chain already is*, not how saturated that one entry alone
is — a chain that already strongly agrees (all four high) will damp new
bonuses harder even on an offset whose own value happens to be low, and
vice versa. SF's/our design's analogous "consistency multiplier"
(design/09 §3.9, `CMHC` array) approximates the same intuition
(amplify agreeing chains) from the opposite direction — bonus *scaled up* by
a *count* of already-positive predecessors — rather than this shared-gravity
*damping* scheme. Two different mechanisms aimed at the same "correlated
predecessors" idea; worth flagging as an alternative worth an isolated test
rather than assuming either is better untested.

Read offsets remain {1,2,4} only (§2) — offset 6 is written but never read
for *ordering*; it exists purely to feed the "total_score" gravity computation
above and the LMR-research continuation-only update (§4). This mirrors SF's
"offset 5 written but never read for ordering" oddity (design/09 §3.9) —
apparently a recurring pattern across engines that every history-consuming
codebase eventually accumulates a write-only offset.

## 4. Update call sites

**Node-result burst on beta cutoff** (`search.cc:1195-1214`): `history_depth
= depth + (alpha > beta + kHistoryBonusMargin)` — a **generous-cutoff depth
bump**: if the fail-high beat beta by more than `kHistoryBonusMargin = 44`
centipawns, every history write for this node is computed as if searched one
ply deeper (`search.cc:1196-1197`). Quiet best move: `quiet_history`,
`pawn_history`, and `continuation_history` (which internally applies the
malus to every non-best *tried* quiet, §3.2) all get `UpdateScore` calls with
this bumped depth; capture best move: `capture_history->UpdateScore`. In
**both** cases, `capture_history->Penalize` runs over *every* tried capture
(`search.cc:1212`) — "good captures are expected to be the best move, so even
a quiet-cutoff node still punishes captures that didn't get there first,"
called out explicitly in-source. Qsearch has its own smaller-scale mirror
(`search.cc:420-450`): a flat `UpdateScore(.., 1)` bonus (depth pinned to 1)
on a capture beta-cutoff, always followed by `Penalize(.., 1, captures)`.

**Killer-move recording**: `stack->AddKillerMove(move)` fires in the same
quiet-cutoff branch (`search.cc:1199`, see §1).

**Post-LMR re-search, continuation-only, sign-and-magnitude from the actual
result** (`search.cc:1135-1141`): when a reduced null-window re-search
follows an LMR reduction, the continuation history (only — not quiet/pawn)
gets a bonus/malus whose *sign and size depend on where the re-search score
landed*, not a flat constant:
```
bonus = score <= alpha ? HistoryPenalty(new_depth)
      : score >= beta  ? HistoryBonus(depth)
      : 0                                            // ambiguous, no write
```
This is richer than SF's/our design's flat `+1415` post-LMR conthist bonus
(design/09 §3.9 "external write sites") — Integral only rewards/punishes
when the re-search actually resolved one way or the other, and skips the
write entirely for a re-search that landed strictly between alpha and beta
(informative failure, not yet actionable).

**Fail-low parent nudge** (`search.cc:1245-1253`): when *every* move at a
node failed to raise alpha (`!best_move`) and the move that led to this node
was quiet and non-promoting, the **parent's** move gets a `quiet_history`
bonus (`HistoryBonus(depth)`) *and* a `pawn_history` bonus at half that size
(`history_bonus / 2`). The in-source comment explains the intent precisely:
the parent will already get a bonus from its own beta-cutoff branch one
level up, but "we also give a history bonus in the event of a fail low to
allow history tweaks to occur in PVS re-searches" — i.e. don't wait for the
PVS re-search machinery to resolve before crediting a move that just proved
itself by holding the opponent to nothing better than what got us here. SF's
equivalent (design/09 §3.9 "fail-high/fail-low parent nudges") is present
but structured as symmetric main-table nudges scaled by `bonus_scale`; this
is a smaller, more targeted, quiet+pawn-only version.

**Eval-diff nudge feeds `quiet_history`/`pawn_history` *directly*, not a
correction table** (`search.cc:678-689`): across two static-eval-bearing
plies with a quiet, non-check parent move, `their_loss = static_eval +
prev_static_eval - kEvalHistUpdateBias` (near zero if the position didn't
change), and `bonus = clamp(-kEvalHistUpdateMult * their_loss / 10,
-kEvalHistUpdateMin, kEvalHistUpdateMax)` (asymmetric clamp: min 68, max
108) is applied to the parent's move in **both** the quiet and pawn tables.
This is the same underlying idea as SF's eval-diff *correction-history*
nudge (design/09 §3.10's `eval-diff nudges: main << 10*eval_diff, pawn <<
13*eval_diff` — note SF's own version *also* targets `main`/`pawn`, not
correction history, despite living in doc 09's correction-history section by
citation proximity) and structurally identical to Reckless's version
(`reckless_reference/05` §5, "same idea ... but here it feeds the quiet
butterfly table directly"). **All three engines converge** on routing this
signal into the ordinary quiet/pawn tables rather than a dedicated table.

**Correction history write gating** (`search.cc:1283-1286`):
`if (!stack->in_check && (!best_move || !best_move.IsNoisy(state)))` — update
fires when not in check and the best move (if any) is quiet, matching SF's/
our design's D22 rule (raw eval in TT, corrected eval used for pruning,
correction learned only from quiet/no-improvement outcomes) exactly.

## 5. Correction history internals: major (not minor) key, {2,3} continuation offsets, and a derived "complexity" signal

**Keys tracked**: `pawn`, `major` (rooks+queens — `major_key`, `board.h:95`,
maintained incrementally like `pawn_key`), `non_pawn[white]`,
`non_pawn[black]`, and a continuation-correction term (`correction_history.
h:61-100`). **There is no minor-piece (knight+bishop) key** — SF and our
design/09 §3.10 both track `minor_key` instead. Git history shows this
wasn't an oversight: commit `d520069e` ("Minor & major piece indexed
Correction History", Sep 2024) originally added **both** `minor_table_` and
`major_table_` side by side (averaging their contributions,
`(minor_correction + major_correction) / 2` per the pre-simplification
code); commit `de739232` ("Simplify away minor-piece correction history",
Nov 2024, `Bench: 1860024`) measured minor-key correction as not pulling its
weight and deleted it outright, keeping only major — the **opposite** of
SF's/our choice to keep minor and not major. Concrete, measured (bench-
trailered) evidence that at least one top engine's SPRT process picked the
other half of this pair. Weights (current): `kPawnCorrectionWeight=42`,
`kNonPawnCorrectionWeight=40`, `kMajorCorrectionWeight=37`,
`kContinuationCorrectionWeight=52`, combined then divided by `512`
(`correction_history.h:93-99`) — the update mechanism itself was rewritten
onto the shared gravity primitive later still (commit `02052229`, "Switch to
history gravity formula for correction history", Dec 2024) — i.e. correction
history shipped with its own bespoke clamped-linear update first and was only
unified onto the one shared `ScaleBonus` primitive roughly 5 months into the
subsystem's life, a "ship first, generalize later" sequencing worth noting
for §8.

**Continuation-correction offsets are {2, 3}**, not SF's/our {2, 4}
(`correction_history.h:50,81`, `for (int ply_ago : {2, 3})`) — the outer
index is keyed by *that* ply's `continuation_correction_entry`, inner by the
piece now on the previous move's to-square. Landed incrementally: `8db9a161`
("Continuation Correction History", Oct 2024, offset presumably just {2}
initially) then `2f95801e` ("3-ply Continuation Correction History", Feb
2025, `Bench: 2739677`) added the {3} branch — "3-ply" here names *how far
back the deepest read reaches*, not an offset count.

**`eval_complexity`: correction magnitude reused as a search-wide reliability
signal.** `stack->eval_complexity = |corrected_static_eval - raw_static_eval|`
(`search.cc:674`, computed once per node right after `AdjustStaticEval`/
`CorrectStaticEval` run) — literally "how far did correction history move
the static eval this node." This derived quantity, not present in SF, our
design, or the Reckless study, is then consumed by **two** unrelated pruning
decisions:
- **Reverse futility pruning**: `futility_margin += eval_complexity *
  kRevFutComplexityMargin / 32` (`search.cc:731`, `kRevFutComplexityMargin =
  12`) — a heavily-corrected static eval is treated as *less* trustworthy, so
  RFP's cutoff margin widens (RFP prunes less eagerly) in proportion to how
  much correction history disagreed with the raw material/PSQT eval.
- **LMR**: `if (eval_complexity > kLmrComplexityDiff) reduction -=
  kLmrComplexity` (`search.cc:1092-1093`, threshold `73`, discount `778`) —
  the same "don't trust this node's shallow signal as much" intuition
  applied to reduction depth instead of a pruning margin.
This is a clean, cheap ("one subtraction, already-computed operands"),
genuinely portable idea: correction history already measures raw-vs-learned
eval disagreement as a side effect of its normal job; Integral is the only
one of the three studied engines that *also* treats that disagreement's
*magnitude* as a first-class search signal in its own right, independent of
the corrected value itself.

## 6. Threat-awareness: contextual storage *and* additive scoring, used together — not a substitute pair

Our Reckless study speculated (`reckless_reference/05` §3, "One plausible
reading: this substitutes for SF/our... escape/entry term") that a
threat-context axis baked into `QuietHistory`'s storage shape might be an
*alternative* to a hand-tuned scoring-time threat bonus, not something an
engine would carry alongside such a bonus. **Integral's git history disproves
that reading.** Commit `296901c3` ("Add threatened index to quiet history",
Jul 2024, `Bench: 2092587`) added the `[from∈threats][to∈threats]` storage
axis to `QuietHistory` first (`quiet_history.h:14-53`; the current shape).
Then, **69 pull requests later**, commit `c49e2192` ("Use enemy threats for
quiet move ordering", Sep 2024, `Bench: 2219229`) *separately* added the
additive, hand-tuned queen/rook/minor escape-or-entry scoring bonus
(`move_picker.cc:216-238`, §1.1 above) — on top of the already-existing
storage axis, not instead of it. Both mechanisms are present and tuned
independently in the current source. **Takeaway for our own design/09 §9
open question 6 and the Reckless-derived Adapt item 3/8**: contextual
storage and an additive scoring bonus are not competing designs to pick one
of — a second data point (Integral, independently of Reckless) suggests they
capture different signal (a *learned*, move-specific "has escaping-a-threat-
looking-like-this been good" statistic vs. a *hand-tuned*, piece-value-scaled
"escaping this specific threat class is worth roughly X" prior) and may
compose. Still untested by us; still worth an isolated SPRT rather than
assuming the composition helps — but the "substitutes for" framing in the
Reckless note should be read as refuted, not confirmed, by this second data
point.

## 7. Comparison summary — Stockfish / Reckless / Integral / us

| Aspect | SF / our design/09 | Reckless | Integral |
|---|---|---|---|
| Pipelines / stages | 4 pipelines, 16 stages | 1 pipeline, 5 stages + mode flags | 1 pipeline, 8 stages (adds 2 killer stages) + mode flags |
| Selection | partial insertion sort | linear max-scan + swap-remove, O(n) per pick | selection sort per stage (`SelectionSort`, O(n) per pick, `move_picker.cc:153-169`) |
| Killers/countermoves | absent (design D17: "never") | absent | **2 killers/ply, present and tuned**; no countermove table |
| Table count | 8 (main, low_ply, capture, cont, pawn, corr, cont_corr, tt_move) | 6 (no low_ply, no tt_move) | 6 (no low_ply, no tt_move) — same count as Reckless, different split (Integral keeps *both* a separate `PawnHistory` table *and* a threat-contextual `QuietHistory`, where Reckless folds threat-context into `QuietHistory` alone and has no separate pawn-indexed quiet table) |
| Threat-awareness | additive scoring bonus only | contextual storage axis only | **both**, added independently 69 PRs apart (§6) |
| Table init | negative-biased per table | all zero | **mixed**: quiet/capture/continuation/correction start at 0; `PawnHistory` alone starts at −919 |
| Gravity bound | per-table `D` (7183/8192/10692/30000/1024) | per-table (own values) | **one shared constant, 11422**, across quiet/capture/pawn/continuation; correction overridden to 1024 (matches SF/us) |
| Per-`go` decay | 789/1024 + low-ply refill | none | **none** (matches Reckless) |
| Correction keys | pawn, **minor**, non-pawn(w/b), cont(2,4) | pawn, non-pawn(w/b), cont(2,4) — no minor | pawn, **major**, non-pawn(w/b), cont(**2,3**) — no minor; tried minor+major together, measured, dropped minor |
| Correction-derived signal reused elsewhere | not modeled | not modeled | **`eval_complexity`** (|corrected−raw|) feeds RFP margin and LMR (§5) |
| Continuation-history write | per-offset weight table + growing consistency multiplier | flat, unweighted, same bonus on all offsets | **shared "total score" gravity** across offsets {1,2,4,6} (§3.2) — a third distinct mechanism |
| Eval-diff nudge target | correction-history-adjacent (`main`/`pawn` per design/09 §3.10, despite the name) | quiet butterfly table directly | quiet + pawn tables directly — **matches Reckless, not SF's table choice** |
| Single "move quality" scalar reused by pruning | `stack->statScore`-equivalent, computed at scoring time | not centralized the same way | `stack->history_score`, computed once, read by 5+ distinct guards (§2) |
| Check bonus | `+16384`, gated on `see_ge(m,-75)` | `+10723`, ungated | `+2048`, ungated, **also exempts the move from futility pruning** via the same `check_zones` bitboard (§1.1) |
| ContHist stack plumbing | our design: safe `ContHistId` (D14); SF: raw pointer | raw pointer, `unsafe` | raw pointer via `StackEntry::continuation_entry`/`continuation_correction_entry` (`stack.h:70-71`), `unsafe`-adjacent (not reviewed in depth here; out of scope for a no-`unsafe`-budget engine like ours) |

## 8. Process notes relevant to the success-factor mandate

Not this note's primary focus (see the dedicated success-factor studies),
but directly visible while reading this subsystem's git history and worth
recording here since it was gathered as a side effect:

- **Every commit in this subsystem carries a `Bench: <nodes>` trailer**
  (e.g. `Bench: 1708751`, `Bench: 2400341`) — the same convention as our own
  D24/`tools/check.sh` and SF's own commit convention. Confirms this is an
  ecosystem-standard practice (traceable to Stockfish's Fishtest culture,
  propagated through the OpenBench-adjacent tooling most modern top engines
  including this one appear to share), not something we invented in
  isolation.
- **Individual Elo deltas in commit messages, but only early on.** The
  earliest history-related commits (`dc00fe58` "Capture History (17 elo)",
  `17707e0e` "Reduced SEE splitting threshold (8 elo)", `f46de732`
  "Generate evasions in QS ... (11 elo)", June–July 2024, within the first
  ~15% of the repo's commit history by count) cite per-change SPRT Elo gains
  directly in the message. Later commits stop doing this and instead cite
  bench node counts plus periodic **"LTC SPSA Tune"** commits (`#13`, `#97`,
  `#144`, `#164`, `#180`, `#204`, `#220`) each naming exact iteration/game
  counts (25k iterations/100k games; 9375 iterations/75000 games; 12.5k
  iterations/100k games) recurring roughly every 1–3 months from mid-2024
  through mid-2025. Reads as a maturity progression: cheap per-PR Elo
  citations while the engine is young and gains are large/individually
  legible, replaced by bulk periodic re-tuning once enough tunable constants
  exist that only a joint SPSA sweep can find the current optimum.
- **The tunable-parameter framework (`SPSA_TUNE`/`TUNABLE`/`TUNABLE_STEP`
  macros, `src/tuner/spsa.h`) was introduced deliberately as its own commit**
  ("Make search parameters tunable", Jul 2024) and an SPSA tuning pass landed
  **five days later** (`#13`, Jul 16 2024) — a very short gap between
  "build the tuning infrastructure" and "use it at scale," suggesting the
  infrastructure was built specifically because a tuning run was already
  queued up, not spec-first.
- **Design evolves by measured simplification, not just addition.** Two
  clear examples in this one subsystem: minor-key correction history added
  then removed 2 months later once major-key correction was shown to
  dominate it (§5); the correction-history update mechanism shipped bespoke
  and was only unified onto the shared gravity primitive ~5 months later
  once that primitive existed elsewhere (§5). Both changes carry a `Bench:`
  trailer, i.e. both were run through the same SPRT gate as additions would
  be — removal is not treated as free just because it deletes code.

## 9. File:line index

- `move_picker.h`: `Stage` 25-34; `MovePicker` fields 60-72; constructor
  signature 36-42.
- `move_picker.cc`: tunables 5-32; constructor 34-49; `Next()` 51-144
  (TT move 54-63, good-noisy split + history-adjusted SEE 65-92, killers
  94-118, quiets 120-133, bad noisys 135-141); `SkipQuiets` 146-151;
  `SelectionSort` 153-169; `GenerateAndScoreMoves` 171-187 (killer-dedup
  filter 176-186); `ScoreMove` 189-246 (promotions 194-203, captures
  209-214, threat term 216-238, check bonus 240, final blend 245).
- `history/history.h`: weights + scale 13-19; `GetMoveScore`/
  `GetQuietMoveScore`/`GetCaptureMoveScore` 40-67.
- `history/bonus.h`: tunables 9-13; `HistoryBonus`/`HistoryPenalty` 15-26;
  `ScaleBonus` 29-31.
- `history/quiet_history.h`: table shape 52; `UpdateMoveScore`/`UpdateScore`
  14-37; `GetScore` 39-44.
- `history/capture_history.h`: `UpdateScore` 14-22; `Penalize` 24-37;
  `GetScore` 39-44; table shape 47-48.
- `history/continuation_history.h`: `UpdateScore` 15-28; `UpdateMoveScore`
  (total-score gravity) 30-42; `GetEntry`/`GetScore` 44-62;
  `UpdateIndividualScore` 65-81; table shape 84-85.
- `history/pawn_history.h`: fill tunable 10; table 12-53.
- `history/correction_history.h`: weight tunables 10-13; `UpdateScore`
  23-59 (continuation-correction offsets {2,3} at 50-58); `CorrectStaticEval`
  61-100 (continuation term 78-92); `CalculateBonus` 109-113; index helpers
  129-140; table shapes 143-147.
- `stack.h`: `StackEntry` fields 55-80 (`continuation_entry`/
  `continuation_correction_entry` 70-71, `killer_moves` 73, `history_score`
  60, `eval_complexity` 59); `AddKillerMove`/`ClearKillerMoves` 81-91; `Stack`
  padding (`kPadding = 6`) 112.
- `search.cc`: `AdjustStaticEval` 220-245; qsearch history reads/writes
  366-450 (movepicker construction + `force_evasions` 360-366, cutoff bonus
  427, penalty 450); eval-diff nudge 678-689; RFP with complexity/history
  terms 723-737; `eval_complexity` computation 645/674; killer-clear-ahead
  713; main move loop 886-1229 (history_score set 903, LMR pruning guards
  905-978, singular extensions 981-1030, LMR + killer/complexity terms
  1052-1121, post-LMR-research conthist update 1135-1141, node-result burst
  1182-1228 with `kHistoryBonusMargin` at 1196-1197); fail-low parent nudge
  1241-1253; correction-history write gate 1283-1286.
- `board.cc`: `CalculateThreats`/`threatened_by` population 511-544;
  `GetOpponentWinningCaptures` 595-608; `MoveGivesDirectCheck` 468-481;
  `check_zones` population inside `CalculateKingThreats` 583-591.
- `constants.h`: full tunable catalogue for this subsystem 28-35 (RFP incl.
  `kRevFutComplexityMargin`/`kRevFutHistoryDiv`), 55-76 (FP/history-prune/
  LMR-history/complexity), 90-100 (LMR killer/complexity/history-bonus-margin).
- Commits cited: `296901c3` (threat-context quiet-history axis, Jul 2024),
  `c49e2192` (additive threat scoring bonus, Sep 2024), `dc00fe58`/
  `17707e0e`/`f46de732` (early per-PR Elo citations, Jun–Jul 2024), `d520069e`
  (minor+major correction added, Sep 2024), `de739232` (minor dropped, Nov
  2024), `02052229` (correction unified onto shared gravity, Dec 2024),
  `007ca05d` (history-bonus static bias, Nov 2024), `8db9a161`/`2f95801e`
  (continuation-correction history, offsets grown to {2,3}, Oct 2024/Feb
  2025), `7578fe26` (separate `PawnHistory` table added, Dec 2024),
  `b9905bb3` (continuation-history "total score" gravity, Nov 2025),
  `8b017b02` (noisy-killer exclusion hardened, Mar 2025), `484db145`
  (direct-check bonus + FP exemption, Mar 2026), `3f15ba6b` (qsearch
  quiet-TT-move handling, Apr 2026, most recent commit in the repo).

## Adapt to us

Ordered roughly by impact/effort ratio; each is independently testable via
our fixed-node-proxy SPRT (design/09 §5, D25) without disturbing the rest of
the stack. All touch `design/09-move-ordering-history.md` and/or
`design/10-pruning-extensions.md`; phase target is **P9** (current pruning-
wave work) unless noted.

1. **Reopen D17 (no killers) for a real test, on the strength of this
   evidence.** *Why:* our D17 reasoning ("modern SF removed them ... we skip
   that detour entirely") is sound as a description of *SF's* trajectory, but
   Integral — a top-6 engine with a fully modern history/continuation stack
   including 6-ply continuation history — still carries two killer moves per
   ply, still tunes their exclusion rules (commit `8b017b02`), and still
   feeds them into LMR (`kLmrKillerMoves`, §1). This is direct counter-
   evidence to "killers are strictly subsumed by continuation history,"
   from a second independently-built top engine (not just an assumption
   carried over from SF's own history). *Effort:* small to prototype (two
   `Move` slots per `Stack` entry, two extra `MovePicker` stages, one LMR
   discount term) but requires a `design/decisions.md` D17 amendment *before*
   landing, since D17 currently says "ever." *Impact:* unknown sign —
   exactly the kind of "verify, don't assume" case D17 itself should have
   been (and should become, if reopened).
2. **`eval_complexity` (|corrected static eval − raw static eval|) as a
   reusable RFP-margin/LMR-reduction input** (§5). *Touches:* design/09 §3.10
   (`correction_value`/`to_corrected_static_eval` already compute both raw
   and corrected values — this needs only a subtraction and a stash of the
   result on `Stack`, no new table) and design/10 (RFP margin formula, LMR
   reduction formula). *Effort:* tiny — the raw and corrected static evals
   already both exist at the point `to_corrected_static_eval` is called; one
   new `i32` field on our `Stack` entry, two new additive terms at two
   existing call sites. *Impact:* plausible free win — it's "reuse a value
   you already computed as a second signal," the cheapest possible kind of
   idea to test, and neither SF nor Reckless model it at all (a genuine
   Integral-original contribution as far as this three-engine comparison
   shows).
3. **A "combined move-quality scalar, computed once per move and reused by
   every pruning guard at that node"** (§2) as an explicit refactor target,
   *whether or not* the specific formula changes. *Touches:* design/09 §3.9
   (documents the update functions) and design/10 (documents every pruning
   guard's own statScore read) — currently doc 10 owns each call site
   individually; this would formalize "read once at loop top, reuse for
   LMR/LMP/FP/HistPrune/SEE-prune" as the doc'd contract rather than an
   emergent property of call-site ordering. *Effort:* small, mostly a
   documentation-and-signature change if our `stat_score`-equivalent isn't
   already structured this way; verify against current `src/search/mod.rs`
   before assuming a gap exists. *Impact:* code-quality/perf (fewer
   redundant table reads), not expected to change search behavior (bench-
   neutral if done as a pure refactor) — low risk, worth doing regardless of
   the SPRT-testable ideas above.
4. **A direct-check bitboard computed once per node, reused for both the
   quiet check-bonus *and* a futility-pruning exemption** (§1.1). *Touches:*
   design/09 §3.7 (our check-squares check bonus is already `see_ge`-gated,
   design-doc-specified) and design/10 (FP guard). *Effort:* small — if our
   `check_squares(pt)` helper (design/09 §3.7, sourced from doc 04) isn't
   already exempting direct-checking quiets from FP, this is a one-line
   guard addition reusing an existing bitboard test; if it's SEE-gated
   already (ours is, Integral's isn't), keep the gate — Integral dropping it
   is unverified, not something to copy blind. *Impact:* small, cheap to
   isolate.
5. **Reconsider whether a `major_key` correction table (rooks+queens)
   outperforms or complements our `minor_key` one** (§5). *Touches:*
   design/09 §3.10 (correction-history key catalogue), design/04 (D9 key
   definitions — a new incremental Zobrist-style key alongside
   `minor_key`/`non_pawn_keys`). *Effort:* medium — needs a new incrementally-
   maintained key (design/04 territory) plus a new correction-table field
   and weight. *Impact:* unverified; Integral's own history shows they tried
   minor+major *together* first and measured minor as the one *not* worth
   keeping (§5) — the direct experiment for us would be "add major, keep
   minor, see if major dominates it the way it did for them," not a blind
   swap.
6. **The shared "total-score" gravity for continuation-history writes**
   (§3.2) as an alternative to our design's per-offset-weight +
   consistency-multiplier scheme (design/09 §3.9). *Touches:* design/09 §3.9
   `update_continuation_histories`. *Effort:* small — replace the weighted-
   offset loop with "sum the four raw scores once, pass that sum as the
   `ScaleBonus` reference for all four writes." *Impact:* unknown; a third
   distinct mechanism (ours: consistency multiplier that *amplifies* on
   agreement; Reckless: flat unweighted, no cross-offset interaction at all;
   Integral: shared-gravity *damping* on agreement) aimed at the same
   intuition — genuinely three-way undetermined, good candidate for a
   three-arm isolated SPRT rather than a two-way "adopt or don't."
7. **A single shared gravity/bound across quiet+capture+pawn+continuation
   history, in place of four independently-sized `StatEntry<D>` bounds**
   (§3.1). *Touches:* design/09 §3.4 (table catalogue — would collapse
   `MAIN_D`/`CAPT_D`/`PAWN_D`/`CONT_D` into one `const`) and every weight
   constant in `GetMoveScore`'s blend (they'd need re-tuning together, since
   our weights were chosen assuming continuation history's 30000 headroom
   vastly exceeds the main table's 7183 — collapsing the bound changes the
   *relative* dynamic range between tables, not just an absolute scale).
   *Effort:* large *if* done properly (full re-tune), tiny if done as a
   throwaway experiment. *Impact:* high uncertainty, high blast radius —
   flagged as a **complexity-reduction candidate to test in isolation, not
   to adopt on the strength of "a top engine does it this way"**; unlike
   items 2/4 above this changes four tables' saturation behavior at once and
   needs its own dedicated SPRT wave, not a slot-in.
8. **NOT recommended to port as a standalone change: the "both contextual
   storage AND additive scoring bonus" threat-awareness combination (§6),
   without first landing whichever one we don't have yet.** We currently
   have (design/09 §3.7) only the additive scoring-time term, no threat-
   context storage axis on `main`/quiet history (that's the Reckless-derived
   Adapt item 8 in `reckless_reference/05`, still unimplemented per that
   note). This Integral finding *strengthens the case* for eventually trying
   that Reckless item (§6 shows two independent engines value it, and shows
   it composes with a hand-tuned term rather than replacing it) but doesn't
   change its effort/risk profile — still a table-reshaping, bench-signature-
   affecting, post-P9 change per the existing Reckless note's own item 8.
