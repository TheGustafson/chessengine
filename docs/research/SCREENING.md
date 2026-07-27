# Screening tunables — the protocol

**Read this before proposing or promoting any search-parameter change.** It is
written to be handed to an agent with no other context.

Everything here was learned by getting it wrong on 2026-07-27: a configuration
measured **+32/1277 tactical solutions at 3.5σ** and then **−30 Elo in games**
(SPRT, 211 games, 45.75%). The protocol exists to make that failure
unrepeatable.

---

## 1. The one command

```bash
python3 tools/screen.py 'RED2600:RED_TABLE_NUM=2600' 'RFPup:RFP*1.25'
```

Arms are `name:OVERRIDES`, where overrides are comma-separated
`PARAM=VALUE` or `FAMILY*FACTOR`. Families are defined in
`tools/arm_screen.py:FAMILIES` and expand to every eval-denominated member,
rounded **half away from zero** (truncation once turned `SE_DOUBLE_BASE` −3
into −1, a 0.33 factor where 0.58 was meant).

Chaining — this is the important flag:

```bash
python3 tools/screen.py --base RED_TABLE_NUM=2600 'RFPup:RFP*1.25'
```

`--base` is applied to the control **and** every arm, so each round measures
the marginal effect of a change *on top of what has already been accepted*.
Accept a winner, fold it into `--base`, run the next round. That is what makes
a reasoned sequence of combined ablations possible instead of a pile of
independent one-offs that do not compose.

Requires a `--features tune` build:

```bash
RUSTFLAGS="-C target-cpu=x86-64-v3" cargo build --release --features tune
```

---

## 2. The three instruments, ranked by how much they deserve belief

### cp@TIER — mean centipawn loss (PRIMARY)

Mean loss of the arm's chosen move against SF18 root-analysis labels:
`loss = best_move_score − chosen_move_score`, in SF18 centipawns.

Why it is the primary metric:

- **Continuous.** Every position contributes a real number, not one bit.
- **Uncensored.** The corpus scores *every legal move* (`arm_mode:
  searchmoves`), so nothing is floored at a top-K boundary. A MultiPV oracle
  understates loss exactly where a config behaves worst.
- **Independent.** The oracle is Stockfish, so this is not agreement with our
  own eval's biases — which is what using our own engine as its own oracle
  would measure.
- **In real units.** SF18 centipawns are advantage given up, not our
  compressed frame (our eval is ~0.64× SF18's).
- **~10k positions**, so the standard error is small enough to read
  sub-centipawn deltas.

**The tiers are diagnostic, not decoration.** Labels exist at 4k/16k/64k
oracle nodes. An arm that improves against 4k while worsening against 64k is
trading deep mistakes for shallow ones — a single-tier screen calls that a
wash. Require the sign to hold across all three before believing an arm.

### nodes — tree size at fixed depth (A COST)

Not a strength measure. It is one input to strength (time-to-depth) with the
other input — decision quality — unmeasured. On its own it rewards pruning
everything, which is exactly the mistake this protocol exists to prevent.
Read it *alongside* cp loss to see what an improvement costs.

### tactics — mate_in_3 at fixed nodes (WEAKEST; never promote on it)

Kept only because it is cheap and occasionally catches a catastrophe.

It is what produced the +32σ/−30 Elo failure. The mechanism is a trap, not
bad luck: **a mate suite is made of positions containing a forced win**, so
reaching depth faster is nearly free there and any pruning increase reads as
a pure gain. Games are decided in quiet positions where the same extra
pruning discards the right move. Both numbers were true measurements — of
different things, and only one is strength.

---

## 3. Rules

1. **Promote on cp loss.** Sanity-check with nodes. Never promote on tactics.
2. **Require the cp sign to hold across all three tiers.**
3. **Fixed-depth tactics is meaningless for a pruning change.** It measures
   decisions per depth, so a search that prunes more can only look worse, and
   the benefit — reaching further on the same budget — is normalised away.
   Always screen at fixed *nodes*.
4. **A screen is not a verdict.** Anything promising gets an SPRT before it
   lands. cp loss is a much better predictor than tactics, but it is still a
   proxy: it scores single-position move choice, not clock handling, search
   instability, or anything time-dependent.
5. **Never test a time-management change this way at all.** Fixed-node
   screening cannot see TM.

---

## 4. What the protocol has established so far

- **We are over-pruning, not under-pruning.** Every node-saving change tested
  costs move quality. The one improvement found is `RED_TABLE_NUM 2600` —
  *less* LMR reduction, ~16% *more* nodes — which improves against all three
  tiers (−0.33 / −0.86 / −0.52 cp) with top-1 up 43.0% → 45.2%. Consistent
  with our tree already being ~35% smaller than SF16's at equal depth.
- **Razoring is good.** Turning it off measured +2.8 cp *worse*. An earlier
  tactics-based screen concluded the opposite and was wrong.
- **The eval-frame correction does not pay**, in any formulation: global,
  its dual, per family, per mechanism, or class 4 alone. See
  `eval-frame-scale-model-2026-07-27.md` §11.
- **Contribution ≠ leverage.** A mechanism can shape half the tree and be
  untunable. LMP removes 23–47% of the tree and had one low-leverage
  parameter because the term that sets its threshold, `depth²`, had no
  coefficient. 16 such implicit-×1 terms are now named and tunable at
  identity — see `prune-budget-2026-07-27.md`.

---

## 5. Rebuilding the oracle

Cached at `tools/nnue/calib/oracle-obs.json` (gitignored, ~15 MB):

```bash
gcloud storage cp gs://archon-main-bucket/corpora/obs/big-v1-sf18.ceobs data/obs/
cargo build --release --features datagen --bin obs-tool
./target/release/obs-tool export data/obs/big-v1-sf18.ceobs \
  | python3 tools/obs_to_oracle.py --out tools/nnue/calib/oracle-obs.json
```

Verify against the ledger before use — `ledger/data.jsonl`, id
`obs-big-v1-sf18`, sha256 `dd90700423d7e003…`. 9,992 positions, 336,122
scored moves per tier.

---

## 6. Improving the tool

This protocol is expected to change as ablations teach us more. When it does,
update **this file and `tools/screen.py` together**, and record what
measurement forced the change — every rule above exists because something
was measured wrong first.

---

## 7. Power, and why most screening is exploratory

Paired standard error on this corpus, measured:

| positions (`--limit`) | paired sem |
|---|---|
| 250 | ~2.0 cp |
| 800 | ~1.1 cp |
| 1500 | ~0.8 cp |
| **9992 (`--limit 0`)** | **~0.3 cp** |

The effects worth chasing are **~1 cp**. So anything below ~3000 positions is
**exploratory only** — good for ranking directions, useless for a verdict —
and `screen.py` labels its own output that way. Use `--limit 0` before
believing any single arm.

This is not a theoretical caution. On 2026-07-27 a set of 1–3 cp deltas at
n=300 was read as signal; the same arms measured the **opposite sign** at
n=1500 on the uncensored oracle.

---

## 8. Per-gate directions measured so far

The single most useful result: **there is no global direction.** Each gate
has its own, and a uniform correction averages them into nothing — which is
precisely why the global eval-frame rescale failed.

| gate | direction that helps | measured |
|---|---|---|
| LMR (`RED_TABLE_NUM`) | **less** reduction, mildly — 2700, not 2500 | −1.79 / −1.76 / −1.60 |
| RFP margin | **larger** (×1.25) — prune less | −0.70 / −0.71 / −0.78, +1.6% nodes |
| parent futility | **smaller** (×0.64) — prune more | −0.78 / −0.87 / −1.20 |
| NMP bar | **lower** (374→168) — fire more | −1.39 / −1.08 / −0.86 |
| razoring | **off** | −0.63 / −0.52 / −0.36 |
| LMP depth coeff | slightly larger (1200) | −0.22 / −0.71 / −0.30 |

Note RFP and parent futility move in **opposite** directions, and RFP
contradicts what the eval-frame model predicts for it (the model says scale
margins *down*; RFP wants *up*). Treat the frame ratio as a hint about where
to look, never as the answer.

Also note the cost axis differs enormously: RFP buys −0.7 cp for +1.6% nodes,
LMR buys −1.7 cp for +12.4%. Prefer cheap improvements when composing, since
node cost compounds and cp gains do not obviously do so.

**Caveat on all of the above:** these are n=800–1500 exploratory readings,
~1.1–1.7σ each. Consistent signs across three tiers is what makes them worth
pursuing, not their individual significance.

---

## 9. Screen at a budget matched to the target time control

Measured on this box: at **8+0.08** with ~40 moves, a move gets ~280 ms, and
at 4.7 Mnps that is **~1.3M nodes**. The default screening budget of 50k is
therefore **26× too small**.

That is not a rounding error, because the mechanisms being tuned are
**depth-gated**:

- RFP applies only at `depth < RFP_MAX_DEPTH` (17). At 50k nodes we rarely
  exceed depth ~13, so RFP is active nearly everywhere in the tree; at 1.3M
  we reach depth 18–20, where it is inactive near the root.
- razoring's margin grows as `d²`, LMP's threshold as `depth²`, and the NMP
  bar moves with `−14·depth`.

So a parameter screened at 50k is being asked a different question from the
one it will face in a game, and an arm can win the shallow question and lose
the deep one. This is a live candidate explanation for the 2026-07-27 SPRT
failure, where a config screened at 50k lost at ~1.3M.

**Use `--budgets` for anything you intend to game-test:**

```bash
python3 tools/screen.py --budgets 50000,200000,800000 --limit 1000 \
    'RED2700:RED_TABLE_NUM=2700'
```

An arm whose gain shrinks or flips as the budget grows was tuned for a
shallower search than the one it will play in. Require the sign to hold
across budgets as well as across oracle tiers.

Cost note: budget × positions is the whole expense, so trade them against
each other — full corpus at 50k, or ~1000 positions at 800k, not both.

---

## 10. Hypothesis: node-level vs move-level gates want opposite treatment

Round 1 produced an oddity worth explaining rather than just recording. RFP
wants a **larger** margin (prune less) while parent futility wants a
**smaller** one (prune more). Both are static-eval futility tests; a global
"we over-prune" story cannot produce opposite signs.

The distinction that does: **what a wrong prune costs.**

| kind | gates | cost of a wrong prune |
|---|---|---|
| **node-level** | RFP, razoring, ProbCut, IIR, (NMP) | an entire subtree's worth of information — the node returns a value nothing else will correct |
| **move-level** | parent futility, LMP, quiet SEE, conthist, capture futility | one move — the other moves at the node still get searched, and a good alternative usually survives |

That predicts node-level gates should be tuned **conservatively** and
move-level gates **aggressively**, and it matches what round 1 measured:
RFP looser, razoring off entirely, parent futility tighter.

**NMP is the apparent exception and probably a real one**: it wants to fire
*more*, not less. Two reasons it is not a counterexample — its prune is
*verified* by a re-search above `NMP_VERIFY_DEPTH`, so the cost of being
wrong is bounded rather than unbounded, and it was measurably **starved** at
0.2% activation, i.e. far outside its operating range rather than slightly
mis-set (`prune-budget-2026-07-27.md` §2).

**Status after the prediction test: 6 of 7 correct.** The hypothesis was
formed after seeing round 1, so round 2 screened seven gates it had never
seen, and it called the direction on six (sign test p ≈ 0.06):

| gate | class | predicted | measured (cp@4k / 16k / 64k) | nodes |
|---|---|---|---|---|
| **IIR 6→8** | node | loosen | **−0.86 / −1.50±0.64 / −0.99** | **+3.4%** |
| ProbCut ×1.25 | node | loosen | −0.17 / −0.79 / −0.48 | +0.8% |
| TT cutoff slack 0 | node | loosen | −0.05 / −0.42 / −0.26 | +8.8% |
| capture futility ×0.64 | move | tighten | −0.36 / −0.60 / −0.28 | +0.6% |
| conthist 4313→3000 | move | tighten | −0.20 / −0.79 / −0.38 | +3.1% |
| quiet SEE 25→15 | move | tighten | −0.10 / −0.25 / −0.29 | −0.5% |
| LMP depth coeff 850 | move | tighten | +0.11 / −0.19 / +0.29 ✗ | −3.1% |

`IIR_MIN_DEPTH` 6→8 is the strongest single arm found so far — **−1.50±0.64
(2.3σ) at the middle tier for +3.4% nodes** — and it was a genuine prediction,
not a post-hoc fit. For comparison, `RED_TABLE_NUM 2700` buys −0.24 to −0.33
cp for +12.4% nodes: roughly a quarter the gain at four times the cost.

The one miss (LMP) is the same gate whose *contribution* is large and whose
*leverage* is near nil (§8 of `prune-budget-2026-07-27.md`), so it may simply
be too inert to move either way rather than a counterexample.

**Cost-effectiveness is now the composing rule.** Rank by cp gained per node
spent, not by cp alone: IIR and ProbCut and capture futility are nearly free,
LMR reduction is expensive, TT cutoff slack is expensive for what it gives.
