# Loss forensics: how to find out exactly why ARCHON lost a game

This is a **method document**, written for an agent picking up cold. It is the
successor to `ENGINE-INVESTIGATION-PLAYBOOK.md`, which covered screening; this
one covers the thing that actually produced results: taking a single lost game
and driving it to a mechanism.

Read this whole file before running anything. The traps below each cost hours.

---

## 0. The one rule

**Ask WHY until you reach a mechanism you can change, and do not stop early.**

That sounds like a slogan. It is not — it is the difference between every
finding in this document and every dead end. The dead ends all share a shape:
a plausible answer arrived, it felt like an explanation, and the asking
stopped. WHY did the engine lose? "It blundered." WHY? "The eval was wrong."
WHY? — and that is where a session ends with nothing.

The chain that worked looked like this. WHY did we lose game 18? Because we
played Be2 instead of Rc5. WHY? Because we did not see Black's refutation Bb5.
WHY? Not because we cannot evaluate it — handed the move, our search scores it
**+442 at depth 12**, better than Stockfish's +298. WHY, then, did we not see
it? Because the normal search never entered that subtree. WHY? Because pruning
discarded it.

> **RIGHT ANSWER, BROKEN EVIDENCE (audited 2026-07-28 — see §13).** The chain
> continues "WHY, which pruning? It is **RFP**, and only RFP: disabling it
> flips the game position to the correct move." The conclusion is **correct**
> and now rests on a class-scale measurement (RFP alone is **81%** of the whole
> gate effect over 2624 positions, +16.1 +- 1.8 cp against a null of 0.0 +-
> 0.6). But **every word of the evidence quoted here was wrong**:
>
> - The arm called "disabling RFP" (`RFP_MULT_BASE = RFP_MULT_CAP = 30000`)
>   does not disable RFP; it *inverts* it into an unconditional shallow prune
>   (§13.1). Two verified clean disables do NOT flip the game position.
> - "Disabling it flips the move" is not a measurable claim at all: a **+-1
>   change to `RAZOR_BASE`** flips this position too (§13.3). The argmax of one
>   search is chaotic.
> - The closing "WHY does Stockfish not do this" is answered by an eval
>   comparison that never needed to be made: SF18 simply **finds Rc5 at 25k
>   nodes where we need 600k** (§13.5).
>
> WHY this stays in §0 instead of being quietly edited: it is the document's
> showcase of the method, and it shows that six WHYs of a broken instrument can
> land on a true conclusion and teach you nothing you can build on. The next
> agent (me) followed the same chain with fixed probes, reached "it is PFUT",
> and had to retract *that* too. Depth of questioning does not substitute for
> verifying the probe. §1 is first for a reason, and both of us skipped §1.5.

Corollaries, all learned the hard way:

- **Do not back up.** When a probe fails, the instinct is to retreat to a
  broader measurement. Resist it. WHY did the probe fail is itself a question
  with an answer, and the answer is usually the next step forward.
- **Do not declare victory.** Every single-position finding in this session
  that felt conclusive either shrank or reversed under a proper test. WHY it
  felt conclusive is worth asking: because a coherent story is more persuasive
  than a wide error bar looks alarming.
- **Do not generalise from one position.** WHY not — because three of the four
  generalisations attempted here were refuted at scale within the hour.

---

## 1. Verify the instrument before you believe it — every single time

WHY does this come first? Because on 2026-07-27 five separate measurement bugs
each produced confident, plausible, self-consistent numbers, and every one
survived until something was measured a *second* way. WHY does that keep
happening? Because a broken instrument produces smooth, believable output — it
does not announce itself.

The verification checklist, in the order things have actually broken:

**1.1 Is the net live?** WHY: `target/release/chessengine` silently ignores
`setoption name EvalFile` — the CEN2 reader is behind `sota,arch-d47v`. A
net-less binary is a PSQT engine producing entirely plausible numbers. This
invalidated a full day, twice.

```bash
cargo build --release --features tune,sota,arch-d47v --target-dir target/qtrace
printf 'uci\nsetoption name EvalFile value <net>\nisready\nquit\n' | target/qtrace/release/chessengine | grep 'info string eval'
#  -> "info string eval: NNUE(sota-v2)"   good
#  -> "info string ERROR EvalFile: ..."   stop
```

`screen.py` and `rematch.py` call `assert_net_live()` and refuse to run
otherwise. **Do not remove that guard.** WHY: absence of an error is weaker
evidence than presence of a success — the loader error was always printed, and
a `head -6` hid it behind option lines.

**1.2 Does the ARCH match the net?** WHY: `arch-d47v` pins accumulator width
768; a 1280-wide net is rejected with `acc_width 1280 != 768`. Build
`--features sota,arch-d48x` for the 1280 nets. The features are mutually
exclusive by design.

**1.3 Is the box quiet enough for what you are measuring?** WHY this matters
so much: instruments split into two classes and mixing them up ruins data.

| class | examples | load-sensitive? |
|---|---|---|
| deterministic | fixed-node cp, fixed-depth bench, fhf counters, node counts | **no** — safe at `nice 19` beside anything |
| wall-clock | NPS, timed games, time-management anything | **yes** — needs an idle box |

Verified empirically: a `nice 19` screen running beside a timed match
reproduced its control row *exactly*. WHY that matters: it means you can always
make progress on deterministic work, and you must never trust an NPS number
taken at load average 21 (I did; it read 0.755× and the true value was 0.845×).

**1.4 Is your frame conversion valid here?** WHY this is subtle: our eval
measures ~0.64× Stockfish's, so cross-engine comparisons need a conversion —
and the conversion has a validity domain. `sf_a(material)` **clamps** material
to [17,78]; in a 12-point endgame you are outside the model and the number is
meaningless. WHY I know: I "proved" a 1.77-pawn net defect that way and had to
retract it.

**The frame-free alternative, and it is much better.** Both engines share
*identical* pruning constants (RAZOR_BASE 465 in both, verified against
`eca43a97`). So **eval measured in units of a shared margin** is comparable
with no conversion at all:

```
ARCHON  -665 / 465 = 1.43 margins below      <- crosses the razoring threshold
SF18    -353 / 465 = 0.76 margins below      <- does not
```

WHY is this the right instrument? Because it answers the question that
actually matters — *does the gate fire* — in the only units the gate sees.
Prefer it to any pawn conversion.

---

## 2. The corpus blind spots — WHY your screen cannot see the failure

**2.1 `oracle-obs.json` is 100% positions with 21+ pieces.** Zero sparse
endgames. WHY that is devastating: every screen either agent ran all day was
blind to the entire endgame regime. Check the mix before trusting a screen:

```python
Counter("<=7" if n<=7 else "8-12" if n<=12 else "13-20" if n<=20 else "21+" ...)
```

**2.2 A loss-derived or sparse corpus is dominated by mate scores.** WHY: the
oracle clips mates at ±20000, and endgame corpora are full of them, so a
handful of positions own every mean. Symptom: paired errors of ±50 to ±85 cp.
WHY it matters: those tables look like data and are noise. Use **top-1 rates**
(a rate cannot be moved by one position) and paired deltas, never absolute
means, on such corpora.

**2.3 Screen power.** Paired sem is ~0.23 cp at 4000 positions and ~0.15 at
the full 9992, while the effects worth acting on are 0.2–0.3 cp. WHY that is
fatal: a 4000-position screen runs at roughly **one sigma**. Concrete casualty:
`hind111` read −0.32±0.23 with the sign holding on all three tiers, was called
the find of the day, and went to **+0.03±0.15** at full corpus and −17 Elo in
games. WHY the screen was not lying: it reported its error bar honestly and I
read a 1.4σ result as a finding. Use `--limit` to rank directions; use
`--limit 0` to decide anything.

**2.4 Oracle tier strength.** `screen.py --oracle-audit` scores each tier's own
best move under the deepest tier, putting tiers and arms in the same unit:

```
 4k tier   6.42 cp   61.8% top-1        <- WEAKER than a 50k-node arm
16k tier   3.48 cp   70.0%
64k tier   0.00 cp  100.0%              <- reference, not merit; own error ~1.9 cp
```

WHY this changed a standing rule: the old advice was "require the sign to hold
across all three tiers", but a tier weaker than the arm scores agreement with a
*worse* search, so requiring it biased us toward weaker arms. New rule: require
the sign only on tiers **deeper** than the arm's budget.

---

## 3. The forensic ladder — what to run, in order

Each rung answers a different WHY. Do not skip rungs; each one eliminates a
class of explanation and the eliminations are what make the answer sharp.

**Rung 1 — get real losses.** `tools/rematch.py` plays ARCHON vs SF18 (or
self-play A/B) and chains match_report + loss_positions automatically. It runs
both legs; the taxonomy diff between equal-time and equal-nodes separates speed
from judgement. WHY both: a loss class that shrinks at equal nodes was bought
by speed; one that survives is judgement.

**Rung 2 — find the REAL errors, not the eval noise.**
`tools/blunder_forensics.py` classifies own-eval collapses into `search_miss` /
`eval_instability` / `already_lost` using a deep `searchmoves` oracle.

> **CRITICAL: the threshold must be in WIN PROBABILITY, not pawns.** WHY: a
> 2-pawn drop from +16.2 to +12.8 is two totally won positions and nothing
> happened. Sparse endgames carry far more extreme evals, so a pawn threshold
> *manufactures* blunders preferentially where the board is empty — which is
> exactly the spurious "sparse endgame concentration" the first run reported.
> Switching to a 0.15 win-probability drop took 73 "blunders" down to 20 real
> ones. **77% of the original set was `already_lost` — the consequence of
> losing, not a cause.**

**Rung 3 — is it depth, or is it the search?** `tools/depth_to_find.py` runs
both engines on the identical FEN, walks their iteration ladders, and records
the first depth from which each *settles* on the target and never leaves, then
pulls depth reached / seconds spent / clock remaining from the PGN. Two
differences carry the whole answer:

```
ours_needed - theirs_needed  > 0  =>  our SEARCH is the problem
ours_needed - ours_reached   > 0  =>  DEPTH IN TIME is the problem
```

WHY this rung exists: game 22 needed depth 21 and reached 13, having spent
**0.078 s** with 1.12 s left while SF18 had 2.59 s. No aggregate could have
told you that.

**Rung 4 — is the move hidden by pruning?** Compare the normal search against
the same depth with **all five eval gates disabled** (RAZOR, RFP, PFUT, CFUT,
QS_FUTILITY):

```bash
setoption name RAZOR_BASE value 30000        # etc for all five
```

> **Do NOT use MultiPV for this.** WHY: MultiPV only unhides moves at the
> **root's immediate children**. The game-18 mechanism was a refutation hidden
> one ply down, *inside* the losing move's subtree, and MultiPV scored it as
> "not pruning" — misclassifying the exact failure mode it was built to find.

**Rung 5 — which gate, exactly?** Disable one family at a time, then bisect
pairs and triples. WHY bisect rather than accept the first hit: in game 18 no
*single* gate off at the after-Be2 position recovered the line, but all five
did — and yet at the **actual game position** exactly one gate (RFP) flipped
the move. WHY the difference: the two positions are one ply apart and the
answer was not transferable. Always run the final test on the position the
engine actually faced.

**Rung 6 — WHY does Stockfish not have this problem?** Do not skip this. It is
the rung that converts "our engine does X" into "our engine does X *because*".
Use the shared-margin comparison from §1.4. Read `eca43a97` — never the
checkout, never a release tag:

```bash
git -C stockfish_reference/Stockfish show eca43a97:src/search.cpp
```

WHY the exact revision matters: our constants were transcribed from
`eca43a97`, which is 223 commits *newer* than the `sf_18` tag. Reading the tag
already caused one bad commit.

---

## 4. The failure-mode taxonomy found so far

Measured over every real error from a 120-game match. WHY publish the
distribution rather than a single cause: because there is no single cause, and
assuming one is how you end up fixing 1/3 of the problem.

| mode | share | signature | status |
|---|---|---|---|
| **Aspiration root instability** | — | root eval oscillates across iterations; ours 1.44 pawns vs SF18 0.45 on the same position with the identical PV | **FIXED, shipped**: `ASPIRATION_MSS_DIVISOR 10588→6776` |
| **Depth not reached in time** | — | `ours_needed > ours_reached`; the critical move gets a fraction of a second | open; TM arm under test |
| **Eval gates hide a refutation** | ~1/3 | gates-off changes the move | **open — global blunting refuted** |
| **Residual eval error** | ~2/3 | gates-off does not change the move | open, least understood |

### The clock finding (mode 2 evidence)

```
move#    ARCHON    SF18
  20+      4.61    5.29
  30+      2.16    3.54
  40+      0.49    2.33
moves under 1.0 s:  ARCHON 47.3%   SF18 25.2%
```

We spend **9.98 s of a 10 s bank before move 33**; SF18 spends 8.78 and keeps
a reserve. WHY that is the right diagnosis and not just "we are slower": the
distribution *shape* is identical (top-10% share 37.3% vs 35.9%), so the
overspend is uniform, not tail-driven. WHY it matters: grind losses fall from
22.5 to 16.7 per 100 games at equal nodes while tactical losses are unchanged —
two instruments, one phenomenon.

WHY the first fix attempt was under-dosed: `OPT_SCALE_BASE` is only the
CONSTANT term and the ply term outgrows it (base is 42% of `opt_scale` at ply
10, 30% at ply 40), so a "−15%" arm is a −5% experiment. Scale the whole OPT
family together.

---

## 5. What has been REFUTED — do not re-run without a new mechanism

WHY this section is as valuable as the findings: every entry here is a
campaign someone would otherwise repeat.

- **Eval-frame rescaling as a source of Elo**, in five formulations. WHY it
  keeps failing: our eval's *median* disagreement with SF18's is −0.02 pawns
  over 30,000 move pairs — there is no bias to correct. Precision, not
  calibration.
- **`hind111` / HINDSIGHT_EVAL_SUM.** Neutral at full corpus (+0.03±0.15),
  −17.4 Elo in games. WHY the frame-corrected value was worse: dimensional
  correctness fixes UNITS, it does not locate our optimum.
- **The eval→history converters** (`EVALDIFF_*_MULT` ×1.5625): 0.0 Elo over
  140 games.
- **Removing `RFP_THREAT_MARGIN`** (the D30 threat relaxation): fixes game 18
  but costs +0.23 cp and **+7.5% nodes** at full corpus. It earns its place.
  WHY that was surprising: it is an untested ARCHON addition, so the prior was
  that it was junk. It is not.
- **`SEE_THRESH_MULT` 1024→1600**: cp neutral-to-worse at +4.7% nodes.
- **"Our net under-rates compensation."** Refuted at n=250: on positions where
  the side to move is down ≥2 with an advanced passer, we read −1.01 margins
  and SF18 reads −1.59 — we are *more* optimistic. Game 18 was an outlier
  against the trend.
- **Tablebases as the explanation for endgame losses.** SF18 played every one
  of these games with `tbhits 0` and no Syzygy files on the box.

---

## 6. The dimensional rule (this is the one that shipped Elo)

A transcribed constant must scale by **r^k** where r = 0.64 and k is set by the
constant's **dimensional role** — not +1 everywhere.

```
k = +1   divisor of a SQUARED eval quantity   (the term then scales r^2)
k = -1   coefficient converting eval into another unit (history, time)
k =  0   a ratio, or a converter whose input is already in our frame
```

WHY every previous frame campaign missed the one that mattered: they scaled
constants **linearly**, and a squared quantity's divisor cannot be fixed that
way. WHY the type-derived audit also missed it: `ASPIRATION_MSS_DIVISOR` is an
`i64`, `FALLING_EVAL_*` are `f64`, `EVALDIFF_*_MULT` are `i32` — **none is
typed `Value`**, because each is typed by its own unit rather than by the eval
it meets. `tools/frame_audit.py` now carries a `CROSS_UNIT` table and a guard
that greps for any `params::` constant appearing beside an eval-derived
identifier and is unclassified. WHY the guard earned its keep: it found
`SEE_THRESH_MULT` on its first run.

---

## 7. The traps, each of which cost real time

- **`nodes` in UCI info is CUMULATIVE.** Consecutive-iteration ratios are the
  branching factor, not explosions.
- **Our engine treats stdin EOF as a stop.** Piping `go depth N` and closing
  the pipe yields a bestmove and no info lines. Hold stdin open.
- **python-chess strips braces from PGN comments.** A `\{...\}` regex silently
  matches nothing. WHY it bit twice: the module-level `TOK` in match_report is
  brace-anchored because it scans raw movetext; a second regex is needed for
  `node.comment`.
- **SF prints no NNUE line for positions in check** — index-based pairing
  corrupts everything after the first one. Drop in-check positions.
- **`bench` writes to stderr** and prints per-position node counts before the
  total.
- **`build_pgo.sh` takes a binary NAME**, not a path; it prepends
  `engines/bin/`. The built artifact survives at `target-pgo-<features>/` if
  the copy fails.
- **The trainer's `.bin` is not CENN.** Pack with
  `D40_ACC=<width> D40_L2=<l2> D40_L3=<l3> D40_POLICY=1 netpack.py <quant.bin> <out.cenn> d40`.
  WHY the wrong invocation is dangerous-looking but safe: it reads garbage and
  the packer refuses with `out_weight > 127`.
- **Self-play A/B: the opponent block needs its own `EvalFile`** or the
  baseline plays PSQT and you measure the net instead of the change.
  `rematch.py` now refuses.
- **`pkill -f` matches your own shell.** Kill by PID.
- **Check `free -g` before any build.** A build beside the trainer has OOM-ed
  it.
- **A failing test may be a real signal.** `lmr_shrinks_the_effective_branching_factor`
  caught that `ASPIRATION_MSS_DIVISOR=3000` — the value a 4000-position screen
  *preferred* — blows the tree up 8× at depth 8 in a quiet middlegame. WHY the
  screen could not see it: a mean over a corpus does not see a tail that fires
  on a minority of positions. The derived value (6776) shipped instead.

---

## 8. Composition — WHY bundling destroys findings

`RFP_THREAT_MARGIN` was found **twice** and lost **twice**, both times by being
judged inside a bundle. Once as one of 12 eval constants no rescale had
touched — packaged as `FRAME12`, screened as a group, read +0.02 cp, discarded.
Once as an untested D30 Wave-G addition. WHY the group test hid it: a neutral
composite can contain a member that matters, and `FRAME12` also contained
`ASPIRATION_BASE_DELTA=3` which costs +11.6% nodes on its own.

WHY this generalises: parameters interact strongly. `razorcorr` screened −0.29
standalone and **+0.61** on a `hind111` base. Use `--base` chaining, test
members individually before bundling, and re-screen neighbours around anything
accepted.

WHY also beware the opposite error: `q_all` (three changes together) read 0.0
while `q_mss` alone read +30, and I nearly wrote an interaction story. The
arithmetic said +30 − 17 + 0 = +13 against a measured −2, well inside noise —
**no interaction was needed to explain it.** One of the parts was simply
negative.

---

## 9. Statistics that keep saving me

- **Compute the Elo error from the real W/D/L split**, not the 0.25 shortcut —
  the shortcut halves it. `var = (w(1-s)² + d(0.5-s)² + l(0-s)²)/n²`.
- **Two independent runs of the same config read +27.9 and −14.7.** WHY that
  matters more than either: pooled over 389 games it is +7.1 ± 13, and a
  single 200-game sample is not evidence. Always ask WHY a result would
  replicate before believing it.
- **Regression to the mean is severe.** Exploratory arms routinely shrink 2–5×
  and several flip sign.
- **A mechanism measured at 3.6σ beats an Elo measured at 1.7σ.** The mss
  change shipped because −13.1% nodes (two instruments) and +0.218±0.061 plies
  deeper (3.6σ) predicted the observed Elo magnitude — not because the Elo
  itself was significant.

---

## 10. Where to go next

The open modes, in order of how much is known:

1. **Eval gates hide a refutation (~1/3 of errors).** Global blunting is
   refuted. The shape of the right fix: RFP's premise is "nothing of ours is
   attacked ⇒ calm ⇒ trust the static eval", and `all_threats()` is an
   **attack map** that cannot represent a passed pawn three moves from
   queening, a zugzwang, or a fortress break. WHY that is a real defect: the
   relaxation is most aggressive exactly where static eval is least
   trustworthy. Gate it on the absence of an advanced enemy passer and it
   keeps its +0.23 cp and −7.5% nodes everywhere it is right. **Not yet
   built.** WHY it must be validated on ≥3 independent losses of the same mode
   before shipping: one position is a patch, three is a fix.
2. **Residual eval error (~2/3).** Least understood. The split between
   depth-fixable and eval-fixable has not been completed. WHY it matters most:
   it is the biggest bucket.
3. **Time management.** Best-evidenced aggregate, mechanism confirmed (the arm
   holds ~1 s more clock at move 40 and cuts time-trouble moves 32.8%→25.4%),
   Elo still unmeasured at 10+0.1.

WHY none of this is finished: because every time a mechanism looked settled,
one more WHY moved it. WHY keep going anyway: because the one change that
shipped came from six consecutive WHYs, and nothing shallower produced
anything at all. WHY should the next agent trust that? Because the refutation
list in §5 is longer than the findings list, and that ratio is what an honest
investigation looks like. WHY does that ratio not mean the method is failing?
Because each refutation permanently removes a campaign someone would otherwise
run, and the survivors are then worth shipping. WHY is this the last paragraph?
Because the next thing to do is not read — it is to pick a loss and ask WHY.

---

## 11. The WHY ladder — the literal questions, in order

Print this. Walk it on every loss. Each rung is a question that has an
answerable, measurable form; the point is that you are never allowed to stop
at a rung that still has a WHY under it.

1. WHY did we lose this game? → find the move, not the game.
2. WHY that move? → what did the oracle want instead (`searchmoves`, deep)?
3. WHY did we prefer ours? → per-move values by depth, both engines.
4. WHY is our value wrong — is it the move we chose, or the one we rejected?
   (Overvaluing ours and undervaluing theirs are different bugs.)
5. WHY did we not find the right move — can our search evaluate it *at all*?
   Hand it the move with `searchmoves`. WHY this rung is pivotal: if the
   forced value is good, the eval is exonerated and it is a search problem.
6. WHY did the search not enter that line — is it depth? Compare
   `ours_needed` against `ours_reached`.
7. WHY was depth short — was it the clock? Reconstruct the remaining-time
   trace; WHY did TM spend what it spent on *that* move?
8. WHY, if depth was sufficient, was the line pruned? Gates-off at the same
   depth.
9. WHY that gate — bisect families, then pairs, on the *actual game position*.
10. WHY does that gate fire here — what is its premise, and WHY is the premise
    false in this position?
11. WHY does Stockfish's identical gate not fire — shared-margin comparison.
12. WHY is our eval different at those nodes — static, frame-free, in margins.
13. WHY is the eval different *there* specifically — is it a class property or
    a one-off? Test the class at n≥250 before believing it.
14. WHY would the fix generalise — find a second loss of the same mode. WHY a
    third — because two can share an accident.
15. WHY would the fix not hurt — full-corpus screen, and read the **nodes**
    column, not only cp.
16. WHY do I believe this measurement — which of §1's instrument checks did I
    actually run this time?

And the meta-questions, which are the ones that caught the most errors here:

- WHY does this number look so clean? WHY is the error bar that small? WHY did
  it not replicate? WHY am I comparing these two quantities — are they in the
  same units? WHY is this corpus the right one? WHY would this position be
  representative? WHY did I stop asking?

WHY sixteen rungs and not three: because the answer in this codebase has never
been fewer than five deep. WHY write them down: because under context pressure
the instinct is to compress the ladder into "the eval is bad", and WHY that is
the single most expensive sentence available. WHY does an agent do that:
because a conclusion feels like progress and a question feels like failure —
the reverse is true here. WHY does the reverse hold: because every conclusion
reached in under four WHYs in this session was later retracted, and the two
that survived took six. WHY should you believe that claim rather than test it:
you should not — WHY not go and check the retraction notices in §5 yourself.

---

## 12. The quality bar — and the detector-recall trap

**Work backwards from a top-tier engine, not forward from this one.** Top-tier
engines almost never blunder big moves. A catastrophic loss is a DEFECT with a
findable cause; it is never a line item to weigh against other channels by its
current Elo share. WHY this needs saying: on 2026-07-28 I ranked blunders below
cumulative drift on exactly that basis, and the ranking was wrong twice over.

**WHY 1 — the detector had ~9% recall.** `blunder_forensics.py` fires on an
own-eval collapse within 2 plies. A move whose refutation lands ten plies later
never collapses our eval and is invisible. It reported 5 errors in 120 games. A
**detector-independent audit** — ask SF18 @120k with `searchmoves` about EVERY
move we played in LOST games — found:

```
260 moves audited      median cp loss 5.0    mean 10.7
moves losing >=100 cp:  1.5%   ->  ~1.0 per LOST game
win probability lost:   0.40 game-equivalents per lost game
```

WHY that is the number to trust: it does not depend on our engine noticing its
own error. Always audit against an oracle before believing a blunder rate.

**WHY 2 — centipawns do not add into outcomes.** Summing per-move cp loss into
"cp per game" assumes losses accumulate linearly. They do not: most small
inaccuracies are absorbed and never converted, while one 126 cp error at a
decision point converts a result outright. Do the channel accounting in **win
probability**, and check which games the errors actually occurred in (four of
five were in games we lost — they are not spread evenly).

**WHY 3 — and this is the strongest reason: catastrophic losses are the
CHEAPEST defects to fix.** Each one hands you a complete trace — a position,
the move played, the move that was right, a measurable gap, and a reproducible
search you can bisect gate by gate. That is why game 18 reached a mechanism in
six WHYs. Cumulative drift offers none of it: there is no position to load, no
gate to disable, no ladder to walk. WHY that inverts the priority I originally
assigned: work should be ranked by **tractability x Elo**, not Elo share, and
on tractability the two channels are not close. A blunder you do not dive into
is free, high-resolution diagnostic information thrown away — and each one
fixed removes a CLASS of failure, not a single position.

`$SCRATCH/harvest.py` builds the full corpus: every >=100 cp error in every
lost game, oracle-scored, sorted by win probability lost. Work it top-down.

---

## 13. The 2026-07-28 instrument audit — five probes that were lying

WHY this section exists: every measurement in §0-§5 that attributed a failure
to a specific pruning gate was made with at least one broken probe. The
findings that survive are re-derived below with the probes fixed. The pattern
is the one §1 already warned about and it caught me anyway: **a broken probe
produces smooth, plausible, self-consistent output.**

### 13.1 `RFP_MULT_BASE/CAP = 30000` is not a disable — it inverts RFP

The margin is

```
futility_mult*depth - (2934*improving + 343*opp_worsening)*futility_mult/1024
```

The trend discount is ~2.87 x `futility_mult`, so the whole expression is
negative whenever `depth < 2.87`. At the transcribed `futility_mult` ~52 that
is a harmless -100ish at depth 1-2. Blow `futility_mult` up to 30000 and the
discount is ~86,000 against a `futility_mult*depth` of 30,000-60,000: the
margin is about **-56,000**, so `eval - margin >= beta` is satisfied by every
node that reaches the gate. The arm does not turn RFP off. It turns RFP into an
**unconditional prune at every shallow improving non-ttPv node.**

WHY it was not caught: it "worked" — it changed the move on game 18, in the
direction that looked like a finding.

The clean disable is `RFP_MAX_DEPTH = 0`. Verified two ways (both required —
one disable that agrees with itself proves nothing):

| arm | d12 | d15 | d16 |
|---|---|---|---|
| control | h5e2 / -1 / 17k | h5e2 / 32 / 54k | h5e2 / 8 / 115k |
| `RFP_MULT_*=30000` (old) | h5e2 / 28 / 54k | **a5c5** / -11 / 152k | **a5c5** / -18 / 162k |
| `RFP_MAX_DEPTH=0` | h5e2 / 34 / 33k | h5e2 / 1 / 119k | h5e2 / 1 / 120k |
| mult=30000 **and** both trend weights = 0 | h5e2 / 34 / 33k | h5e2 / 1 / 119k | h5e2 / 1 / 120k |

The two clean disables are bit-identical — same move, same score, same **node
count** — and both leave game 18 unfixed. WHY node count is the load-bearing
column: two arms can agree on a move by luck; agreeing on the node count to the
digit means they are the same search.

**Everything attributed to RFP with the old recipe has to be re-measured.**
`tools/forensics/uciprobe.py:GATES_OFF` now carries the corrected recipe and
this derivation.

### 13.2 `printf | sleep | engine` truncates the search

Our engine treats stdin EOF as a STOP. `classify2.py` (sleep 7) and
`depthsplit.py` (sleep 9) reported **different moves for the same position at
the same depth** — and the binaries were verified bit-identical first (17
positions at depth 12: same move, same score, same node count, `archon-d47v-pgo`
vs `archon-d47v-tune-pgo`). The difference was entirely the sleep. Hold stdin
open and read to `bestmove` — `uciprobe.Engine` does.

**The same trap applies to Stockfish.** `printf ... | stockfish18` on the
game-18 position answers `g3g4`; the identical query through a persistent
process answers `a5c5`. An oracle reading taken through a pipe is not an oracle
reading.

### 13.3 The argmax of a single search is CHAOTIC — always run a null panel

Twelve semantically-null arms on game 18 (a 1-unit change to a constant that
cannot matter here, plus five hash sizes):

- **Hash 16/24/32/64/128: bit-identical output.** Good — the harness is
  deterministic, so what follows is not measurement noise.
- `RAZOR_BASE 465 -> 464` **changes the move played at depth 17.**
  `PFUT_BASE 40 -> 41` changes it at depth 15.
- Across all 2624 scanned positions the null argmax flip rate is **7.9%**
  (`RAZOR_BASE-1`) and **14.9%** (`CFUT_BASE+1`), against **30.9%** for
  all-gates-off. So roughly half of gates-off's move changes are chaos.

WHY this is the most expensive lesson here: "arm X flips this position" was the
unit of evidence for the whole gate-attribution effort, and on its own it is
worth about one bit against a 8-15% background. **Never report a single-position
argmax flip without the null band beside it.** `uciprobe.NULL_ARMS` exists so
there is no excuse.

The **forced value** of a named move (`searchmoves <move>`) is the instrument
that survives: on game 18 all twelve nulls land in [-17, +126] while a clean
gates-off gives **-619**. Class-wide, paired over 2624 positions:

```
mean( V_control - V_arm )   for the move we actually played, depth 14
  all five eval gates off   +19.8 +- 1.6      <- 12 sigma
  NULL RAZOR_BASE-1          +0.6 +- 0.6
  NULL CFUT_BASE+1           -0.3 +- 0.7
```

**With the gates on, our search over-values the move it chose by ~20 cp.** That
is the mode, measured as a class instead of anecdotally.

### 13.4 A 120k-node SF18 is blind to exactly the errors we hunt

On the game-18 position:

| SF18 budget | verdict |
|---|---|
| 120k (MultiPV 4) | **h5e2 -14**, a5c5 -27 — prefers OUR losing move |
| 1M | a5c5 -41, h5e2 -241 |
| 8M | a5c5 -6, h5e2 outside the top 4 |
| 50M | a5c5 -21, h5e2 **-345** |

A deep-refutation blunder is by construction one a shallow search does not see,
so a shallow oracle **systematically under-counts this failure mode** — the
audit rate in §12 is a lower bound, biased against the very class it is meant
to find. Two further oracle traps, both live in the §12 audit script: it reuses
one engine across positions **without `ucinewgame`**, and `MultiPV > 1`
materially weakens SF (MultiPV 4 at 120k prefers h5e2; MultiPV 1 at 25k plays
a5c5). Adjudicate with: fresh `ucinewgame`, MultiPV 1, persistent process, and
a **stability requirement** — the same best move at 400k, 1.6M and 8M. On
near-equal endgames SF18's own argmax flips between budgets; those are close
calls, not blunders, and 15 of 19 candidates died on that test.

### 13.5 What game 18 actually is

Node-to-find, both engines, `ucinewgame` per probe, truth = SF18 @8M (`a5c5`,
stable at every budget from 25k):

```
                   25k  50k 100k 200k 400k 800k 1.6M 3.2M
SF18                C    C    C    C    C    C    C    C
ARCHON control      .    .    .    .    .    C    C    C
ARCHON gates-off    .    C    C    C    C    C    C    C
ARCHON -pfut        C    C    C    C    C    C    C    C
ARCHON -rfp         .    .    .    .    .    .    C    C     <- WORSE than control
```

- **SF18 has no difficulty here at all.** It needs 25k nodes; we need 600-800k.
  A 24x search-efficiency gap on one position, and the honest form of rung 11
  ("WHY does Stockfish not have this problem") is not "its eval is lower" — it
  is that its search finds the move at a budget we do not.
- In the game we spent **0.049 s** (~50-70k nodes) on this move. At that budget
  SF18 is already right and we are wrong, so this loss is **not** the
  time-management mode. More clock would also have fixed it (600k nodes is
  ~0.5 s) — but the search-efficiency gap is the bigger and more transferable
  defect.
- On the ARGMAX ladder `-rfp` looks WORSE than control here (1.6M vs 800k). It
  is not. See 13.6 — the argmax was the wrong instrument and this row is the
  clearest single demonstration of that in the whole document.

### 13.6 Two instruments, opposite answers — and which one to believe

The argmax ladder above says **PFUT**. It is wrong, and I published that
conclusion internally before checking it against a second instrument. Do not
repeat this.

The right instrument is the **forced value** `V(P, our_move)` — search the move
we actually played with `searchmoves` at fixed depth and read the score. It
answers "does our search know this move is bad", which is the question, rather
than "did the argmax happen to land elsewhere this time". The null panel proves
it is the sound one: `RFP_OPP_WORSENING +1` is **bit-inert on all four
positions**, and on the two null-quiet positions the other nulls move V by 0.

Forced value at depth 14 / 16, four adjudicated blunders (control first):

| arm | g35p98 | g2p81 | g41p35 | g24p14 |
|---|---|---|---|---|
| control | 11 / 18 | -345 / -784 | -273 / -362 | -266 / -404 |
| all five gates off | **-594 / -612** | **-740 / -857** | **-504 / -524** | **-512 / -550** |
| **rfp-off (clean)** | **-550 / -626** | -402 / -740 | **-470 / -559** | -292 / -251 |
| -pfut | 1 / 11 | -340 / -734 | -240 / -240 | -366 / -406 |
| pfut_lmrd<9 / <6 | 7 / 1 · 85 / 1 | ... | ... | ... |
| pfutbase 120 / 250 | 1 / 14 · 31 / 0 | ... | ... | ... |
| followpv=1 | 72 / 43 | -383 / -452 | -251 / -248 | -448 / -409 |
| NULL razor-1 | 11 / 18 | -345 / -706 | -273 / -291 | -485 / -507 |
| NULL rfpopp+1 | 11 / 18 | -345 / -784 | -273 / -362 | -266 / -404 |

**Every PFUT arm sits inside the null band.** `-pfut` changed the move on game
18 while leaving our valuation of the losing move exactly where control had it
(1 vs 11) — it did not learn anything, it rolled the dice differently. Clean
RFP-off, by contrast, produces the truth on both null-quiet positions at both
depths.

And at class scale, paired over 2624 positions (`modescan.py`), which is what
turns this from two anecdotes into a finding:

```
mean( V_control - V_arm ), our own played move, depth 14
  all five eval gates off   +19.77 +- 1.62   (12.2 sigma)
  RFP off ALONE             +16.11 +- 1.77   ( 9.1 sigma)   <- 81% of the total
  the other four, residual   +3.66 +- 1.69   ( 2.2 sigma)
  NULL RAZOR_BASE-1          +0.65 +- 0.61
  NULL CFUT_BASE+1           -0.27 +- 0.70

on the 77 tail positions (dG >= the null p99.5 of 184 cp):
  all five off  +298 +- 13     RFP alone  +197 +- 21     nulls  +17 +- 7
```

**So the original §0 verdict — RFP — is right after all.** It just could not
have been established by anything that session or this one did until the
forced-value instrument and the null panel existed together.

Two corollaries worth keeping:

- **The gates are partially redundant**, which is why single-gate ablation on
  one position is hopeless: correlation between the gates-off and rfp-off
  deltas is only 0.51, so on any given position a different gate may be the one
  that happens to catch the refutation.
- **`RFP_THREAT_MARGIN` is exonerated.** `RFP_THREAT_MARGIN = 0` moves the
  forced value by +1 to -20 (null band) and changes the argmax at no node
  budget from 25k to 400k. The §10 hypothesis — gate the D30 threat relaxation
  on an advanced enemy passer — is aimed at a term that does not cause this.
  Do not build it.

### 13.7 Status of the fix — NOT established

RFP is the mechanism; the dose is not known. RFP off entirely is not a patch,
it is a diagnosis — RFP is a core pruning gate and the class number says it
buys us a lot of tree. What is owed:

1. A dose ladder on the RFP structure (`RFP_MAX_DEPTH`, `RFP_MULT_BASE`,
   `RFP_IMPROVING` — the improving discount is ~2.87x `futility_mult`, so it
   drives the margin NEGATIVE at depth 1-2 even at stock values), scored with
   the **forced-value class instrument**, not the argmax.
2. A **full-corpus** (`--limit 0`) screen of the survivor, reading the nodes
   column as hard as the cp column.
3. SPRT only after both.

### 13.8 The rules this session adds

- **Verify that a disable disables.** Switch the gate off two structurally
  different ways and require the two to agree on the NODE COUNT, not just the
  move. A margin with a subtractive companion term can be driven into an
  inverted regime by the "obviously safe" huge value.
- **Never attribute anything to an argmax flip.** Run `uciprobe.NULL_ARMS`
  first and print the band. The flip rate for a semantically null change is
  8-15%.
- **Measure the forced value of the move you are indicting**, not the move the
  engine happens to pick. It has a null band near zero and it answers the
  actual question.
- **An oracle needs its own instrument check**: fresh `ucinewgame`, MultiPV 1,
  a persistent process (no pipe-EOF), enough nodes, and a stability
  requirement across budgets.

---

## 14. REFUTED: "eval gates hide a refutation" is a node-count artifact

WHY this section supersedes §13 and the §4 taxonomy row of the same name: the
probe that defines the mode — search with all five eval gates off **at the same
depth** and see whether the picture changes — silently hands the gates-off arm
**3.47x the nodes** (median, measured). Any search given 3.5x the nodes finds
more refutations. The mode was never a mechanism; it was a node budget wearing
a mechanism's clothes.

### 14.1 The measurement

220 positions sampled from the scan corpus. For each, the forced value of the
move we actually played, gates-off at depth 14, against control deepened until
control has spent at least as many nodes:

| control nodes / gates-off nodes | mean( V_control − V_gates_off ) |
|---:|---:|
| 0.29x — depth-matched, both d14 | **+24.1 ± 6.3** |
| 0.58x — one ply pair below the node match | **+19.0 ± 4.8** |
| 1.56x — first depth at or above the node match (median d18) | **−25.7 ± 5.1** |

A clean monotone dose-response **in the node ratio**, crossing zero at about
1.0x. The "+19.8 ± 1.6 over 2624 positions" headline in §13.6 is this and
nothing else.

**And it holds in the tail**, which is where the mode was supposed to live:

```
12 tail positions (depth-matched delta >= the 184 cp null p99.5)
    depth-matched   +290.0 +- 25.5     <- looks like a huge hidden refutation
    NODE-matched     -88.8 +- 35.6     <- control is BETTER by 89 cp
```

So even on the positions selected *for* looking like hidden refutations, a
node-matched control beats gates-off. WHY that is the expected result and
should have been the prior: the gates are in the engine because they buy Elo.
An ablation that "improves" the search is nearly always an ablation that was
handed more nodes.

### 14.2 What this kills

- **The §4 taxonomy row "eval gates hide a refutation, ~1/3 of errors."** That
  share came from this probe. There is no measured support for the mode.
- **Rung 4 of §3 as written.** "Compare the normal search against the same
  depth with all five eval gates disabled" is not a valid probe. If you run it,
  node-match it — deepen the control until its node count is at or above the
  ablated arm's — and report the node ratio next to every number.
- **The RFP attribution, in both directions.** §13.6 restored it on a
  class-scale forced-value measurement; that measurement is this artifact, and
  RFP-off blows the tree up for the same reason gates-off does. RFP is neither
  convicted nor exonerated by anything in this document. It is unmeasured.
- **The §10 plan** to gate the D30 threat relaxation on an advanced enemy
  passer. Already dead in §13.6 on its own merits (`RFP_THREAT_MARGIN = 0` is
  inside the null band at every probe); now the mode it was meant to fix is
  dead too.
- **`FOLLOW_PV_MODE = 1`**, screened full-corpus while this was being measured:
  `-0.11 / -0.24 / -0.04 cp` by tier at **+7.4% nodes**. At ~120 Elo per node
  doubling that costs ~12 Elo to buy ~2. Not the fix either.

### 14.3 What survives, and where to go instead

One fact from this investigation is node-controlled end to end and is not
touched by any of the above:

> **On the game-18 position, SF18 plays the correct move at 25k nodes. ARCHON
> needs 600–800k. Both measured with `go nodes N`, `ucinewgame` per probe,
> MultiPV 1, persistent processes. A 24x search-efficiency gap.**

We spent ~50–70k nodes on that move in the game, a budget at which SF18 is
already right — so it is not the time-management mode either. And it is not the
pruning gates, because at matched nodes turning them off makes us worse.

That leaves the things this investigation never instrumented: **move ordering**
(our first-move cutoff rate is 89.4% against SF's 93–95%, and that gap compounds
with depth), **extensions**, and the **eval's ordering of near-equal moves**.
WHY those are the live suspects: they are exactly the mechanisms that change
which nodes a fixed node budget is spent on, and node efficiency at a fixed
budget is precisely what the surviving fact measures.

The instrument to carry forward is **nodes-to-find against SF18 on an
oracle-stable blunder set** — cross-engine, node-controlled, and immune to
every trap in §13. Building that set is the prerequisite: of 19 candidates
adjudicated here, only **2** had an oracle stable across 400k/1.6M/8M
(`g35p98`, 276 cp; `g24p14`, 82 cp, where ARCHON never finds the move at any
budget to 3.2M and SF18 has it at 25k). One 120-game match does not contain
enough of them. Get more games first.
