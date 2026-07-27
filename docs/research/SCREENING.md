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
