# Playbook: investigating scale and tuning problems in ARCHON

Written for an agent picking this up cold. It is a **method**, not a result
log: the specific numbers from the session that produced it were largely
invalidated (see §9) and are deliberately omitted. What survived is the
toolchain, the sequencing, and a list of traps that each cost hours.

Your job: find Elo in the search's parameters and calibration. Iterate until
you run out of ideas or you have a game-verified gain.

---

## 1. The single most important rule

**Verify the instrument before believing the measurement.**

Over one session, *five* separate measurement bugs each produced confident,
plausible, self-consistent numbers, and each survived until something was
measured a second way:

| bug | how it presented |
|---|---|
| engine silently ran **PSQT instead of NNUE** | every number plausible; invalidated a whole day, twice |
| oracle **censored** moves outside its top-K | losses floored exactly where a config behaved worst |
| **unpaired** standard errors | 1–3 cp noise read as signal |
| **prefix** subsampling instead of random | biased toward hard positions; inflated every delta |
| **frame conversion** applied to the wrong units | a ~1.1x difference reported as 4.27x |

None was caught by thinking harder. Every one was caught by re-measuring
differently. Budget for this: assume roughly one in three of your surprising
results is an artifact, and design so artifacts are visible.

**Concretely, before any campaign:**

```bash
# the binary MUST be able to load the net; a net-less build reports PSQT
cargo build --release --features tune,sota,arch-d47v --target-dir target/qtrace
printf 'uci\nsetoption name EvalFile value <net>\nisready\nquit\n' | target/qtrace/release/chessengine | grep 'info string eval'
#   -> "info string eval: NNUE(sota-v2)"        good
#   -> "info string ERROR EvalFile: ..."        stop, fix the build
```

`tools/screen.py` and `tools/engine_compare.py` call `assert_net_live()` and
refuse to run otherwise. **Do not remove that guard.** Also include a **no-op
arm** (a parameter set to its default) in every batch — it must read exactly
zero. If it doesn't, the harness is broken, not the engine.

---

## 2. Read the code before instrumenting

The highest-yield step, and the one easiest to skip.

- Read **our** source and **Stockfish's** side by side. The reference is
  commit **`eca43a97`**, *not* the checkout and *not* a release tag — our
  constants were transcribed from it, and `sf_18` is 223 commits older.
  `git -C stockfish_reference/Stockfish show eca43a97:src/search.cpp`.
- Compare **structure first** (what is generated, what is skipped, in what
  order), then constants. Structural differences never show up in a sweep.
- Watch for **mixed units in one expression**: eval + material, eval vs SEE,
  eval² / divisor. Both engines often have the *identical integer* there, so
  a constants diff shows nothing while the meaning differs.
- Prefer reading to adding engine logging. Counters go in the hot path, cost
  NPS, and answer a narrower question than the source does.

---

## 3. The instruments, ranked by how much they deserve belief

### cp loss vs an SF18 oracle — PRIMARY
`tools/screen.py`. Mean centipawn loss of the arm's chosen move against
Stockfish root-analysis labels, per oracle tier (4k/16k/64k nodes).
Continuous, ~10k positions, **every legal move scored** so nothing is
censored, denominated in real advantage rather than our compressed frame.

Rebuild the oracle:
```bash
gcloud storage cp gs://archon-main-bucket/corpora/obs/big-v1-sf18.ceobs data/obs/
cargo build --release --features datagen --bin obs-tool
./target/release/obs-tool export data/obs/big-v1-sf18.ceobs \
  | python3 tools/obs_to_oracle.py --out tools/nnue/calib/oracle-obs.json
```
Verify the sha256 against `ledger/data.jsonl` id `obs-big-v1-sf18` first.

**Require the sign to hold across all three tiers.** An arm that improves
against 4k labels while worsening against 64k is trading deep mistakes for
shallow ones.

### nodes — a COST, not a strength
One input to strength with the other (decision quality) unmeasured. Alone it
rewards pruning everything.

### fhf (first-move cutoff rate) — cheap ordering proxy
`--fast` gives fhf + nodes in seconds. Good first filter over a large family;
never a verdict.

### tactics suites — WEAKEST, never promote on them
A mate suite is made of positions containing a forced win, so pruning harder
looks free there while it discards the right move in the quiet positions that
decide games. This produced a "+3.5σ" configuration that lost ~30 Elo.

### games — the only verdict
`fastchess`, same binary both sides with different `option.X=Y`, so the pair
is automatically fair. Nothing lands without an SPRT.

---

## 4. Statistics that actually matter

- **Paired deltas.** Control and arm see identical positions; the paired
  difference cancels position difficulty. Reporting unpaired means is how
  noise gets read as signal.
- **Power.** The sem figures quoted in older docs (~2.0 cp at 250 positions,
  ~0.3 at the full set) were measured on a **PSQT** engine whose mean loss was
  ~17.7 cp. With the net live the control's mean loss is ~4.6 cp, top-1
  agreement ~66%, and the blunder rate ~0.1% — a different regime, with
  smaller absolute effects **and** smaller variance. **Re-measure the power
  curve before trusting any sample size**: run one arm at several `--limit`
  values and watch the paired sem. The rule that survives is structural, not
  numeric — exploratory runs rank directions, full-corpus runs settle them.
- **Regression to the mean is severe.** Exploratory arms routinely shrink 2–5×
  at full corpus, and several flip sign. Never promote on an exploratory run.
- **Multiple comparisons.** After ~30 arms the best-looking one is selected on
  noise by construction. A 2.3σ result in that context needs full-corpus
  confirmation, not celebration.
- **Random-sample, never prefix.** `--limit` must draw randomly under a fixed
  seed; the corpus is ordered and prefixes are biased.
- **Report the tail.** The control's median loss is ~1–2 cp against a mean of
  ~17, so the mean is essentially a blunder statistic. Track blunder rate
  (≥100 cp) and p90 separately: two arms with the same mean are not equal if
  one has a fatter tail.

---

## 5. Chained ablation

Use `--base` so each round measures the marginal effect **on top of what has
already been accepted**:

```bash
python3 tools/screen.py 'A:PARAM=X'                    # round 1
python3 tools/screen.py --base PARAM=X 'B:OTHER=Y'     # round 2, on top of A
```

Without this you get a pile of one-offs that do not compose. Parameters
interact strongly: arms that each help alone have cancelled when combined,
and arms that do nothing alone have helped in combination.

**Corollary, and a mistake worth avoiding:** a change that looks bad
*ceteris paribus* may be good once the rest of the search is re-tuned around
it. The engine sits at a joint optimum; if you move one gate, re-screen its
neighbours on the new base before concluding.

Every arm is appended to `docs/research/screen-results.jsonl` and the tool
warns when it sees a config already measured. Check there before re-deriving.

---

## 6. Forensics: from a loss to a parameter

1. **Play games vs SF18** (`fastchess`, `-pgnout`) — equal-time and
   equal-nodes legs; the taxonomy diff between them separates speed from
   quality.
2. **`tools/loss_positions.py`** localises the move that lost each game,
   classifies it (misjudgment / tactical / grind / timepressure), and emits
   the losing positions as a **targeted corpus** — far more informative to
   screen on than random positions.
   `--compare BASELINE.pgn CANDIDATE.pgn` diffs the taxonomies, so a change is
   judged by whether it changes **how** we lose.
3. **`tools/engine_compare.py`** runs both engines on the *same* positions:
   tree width at equal depth, iteration stability, eval bias, and `--static`
   for raw net-vs-net with no search. `--our-options` re-runs the same
   forensics with a candidate applied, closing the loop.
4. **`tools/search_forensics.py`** — `ladder` (per-iteration trace both
   engines), `research` (aspiration re-search rate), `ebf`, `qshare`,
   `shallow` (locate where a ratio is born), `aspiration` sweep.
5. **`tools/frame_audit.py`** prints every eval-denominated threshold in
   *real pawns* for both engines.

---

## 7. The debugger

When aggregates stall, watch a single search decide.

```bash
cargo build --release --features tune,search-control,sota,arch-d47v --target-dir target/qtrace
ARCHON_QTRACE=1 target/qtrace/release/chessengine    # then uci/position/go
```

`qtrace()` is feature-gated and compiled out of shipped builds, so it is a
debugger rather than logging. It prints, per qsearch decision, the window,
the stand-pat value, the move, and which filter rejected it. Extend it to
other gates the same way.

**Hold stdin open past the search** — our engine treats EOF as a stop, so
piping `go depth N` and closing the pipe yields a bestmove and *no info lines
at all*.

---

## 8. Traps, each of which cost real time

- **`nodes` in UCI info is CUMULATIVE.** Consecutive-iteration ratios are the
  branching factor (~1.6/ply is normal), not "node explosions".
- **Depth is not comparable across engines**; nodes-at-equal-depth is.
- **Depth-1 probes are degenerate**: correction history is empty, the TT is
  cold, ordering cannot act. Conclusions there do not transfer. Measure where
  the effect actually peaks.
- **Stockfish only publishes an aspiration-fail info line above
  `NODES_LIMIT_OUTPUT` (10M nodes).** Counting its info lines below that
  measures its *output policy*, not its behaviour.
- **`bench` writes its report to stderr** (`2>&1 >/dev/null`).
- **`bench`'s optional net argument is not the same path as `setoption`.**
  Prefer `setoption name EvalFile`.
- **python-chess strips braces from PGN comments** — a regex anchored on
  `{...}` silently matches nothing.
- **Truncating a FEN for display and then reusing it** creates a position
  that never existed. (Yes, really.)
- **`pkill -f` matches your own shell.** Kill by PID or `systemctl --user`.
- **Check `free -g` before any build**; a PGO build beside the trainer has
  OOM-killed it.
- **A single-position trace generalises badly.** A gate that provably cannot
  fire in one hand-built position fires hundreds of times across real ones.

---

## 9. What the previous session established (and what it did not)

**Robust:**
- Our constants are faithful transcriptions of SF master `eca43a97` — all 260
  screened, no drift. `tools/spsa/param_provenance.py` re-checks this.
- Our eval frame is ~0.64× SF18's, uniformly across the eval range
  (quantile-matched), and our eval's *error* scales with it, so relative
  calibration is comparable.
- **17 implicit-×1 coefficients** were found and made tunable — terms with an
  invisible coefficient of 1 that no "bare literal" rule can catch
  (`+ Depth::from(pv_node)` is not a literal). The whole TT replacement policy
  had no tunables at all. **Look for more of these.**
- ~46% of tunables are in no SPSA campaign.

**Refuted, repeatedly and in several formulations:** the eval-frame rescale
as a source of Elo — globally, per family, as its algebraic dual, per
mechanism, and class-4 alone. Several corrections were *correctly derived*
and measured ~zero because the gate they touched was rarely reached or
dominated by frame-clean filters. **Do not re-run these without a new
mechanism.**

**Invalidated and needing redoing on a net-live binary:** essentially every
quantitative search-shape claim from that session — tree width, qsearch
amplification, stand-pat cut rate, iteration stability, fhf, and the cp
screens. Early corrected numbers suggest the engine is far closer to
Stockfish's tree shape than the PSQT-contaminated figures implied.

**Treat every number in the older docs as suspect unless it was taken with a
net-live binary.**

---

## 10. Suggested order of attack

1. Verify the instrument (§1). Run a no-op arm. Confirm `eval: NNUE(...)`.
2. Re-establish the baseline shape on a net-live binary: width vs SF18 by
   depth, fhf, stability, qsearch amplification.
3. Hunt implicit-×1 coefficients and untunable structural constants (§9) —
   these are pure additions to the tuning surface and cost nothing at
   identity.
4. Screen families with `--fast`, promote survivors to full-corpus cp.
5. Chain with `--base`; re-tune neighbours around anything accepted.
6. Loss-mine against SF18; screen on the loss-derived corpus.
7. SPRT anything that survives. Only games count.

Record what you refute as carefully as what you confirm. On this problem the
refutations have been more valuable than the confirmations, and the biggest
single risk is believing your own instrument.
