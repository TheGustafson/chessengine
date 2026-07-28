# START HERE — fresh session

Read this, then `README.md` (the method), then `LEDGER.md` (every claim + what killed the
dead ones). Do not go reading `docs/research/` for context; that is a pile of
half-refuted narrative and going shopping in it is how 2026-07-28 lost a day.

---

## The problem, in one sentence

**The search discards the right move too early.** The move is generated and is valued
correctly once it is actually searched — it just never gets the search it needs, so a
worse move keeps the root.

**Eval is NOT the problem.** This is a standing ruling and it is measured (our net
correlates +0.803 with SF's deep scores; SF's own static eval manages +0.722). Treat it as
an axiom. Every time the previous session reached for eval it was wrong, and it was always
at the moment the search analysis got hard.

## OUT OF SCOPE — do not spend a minute on this

**The net is not your subject.** NNUE training, architecture, accumulator width, retraining
D48 — **none of it is searchopt work, and none of it is your decision.** Those are human
calls, made outside this folder.

That includes the D48X width question below. It is recorded here only so you know the net
under you is fixed and why; it is **not** a task, not a thing to reason about, and not a
thing to have an opinion on. If a search experiment makes you want to change the net, the
answer is no — write down what you found and move on.

Combined with the eval axiom, the rule is simple: **you change `src/` search code and
nothing else.**

## Where things stand

| | |
|---|---|
| canonical net | **D48X** (gen-2, 1280 acc) — `engines/canonical/`, promoted 2026-07-28 |
| D48X vs D47V | −9.1 [−34.4, +16.1] over 382 games. Promoted deliberately: D48X is the better eval and loses only because the search cannot exploit it. **This is settled and is not yours to revisit** — see OUT OF SCOPE above. |
| loss corpus | **DONE 2026-07-28.** 404 games vs SF18 (−132 Elo, 174 lost games, 10,112 moves probed) → **5,379 still-playable moves scored**, **25 blunders**, split **OURS 3 · BOTH 1 · NEITHER 21**. A blunder = **wp_loss ≥ 0.06 (~88 cp) from a position better than −2.5 pawns**. The old "20-50 positions" estimate was the *blunder* count (25, close), not the actionable subset. |
| how bad are the blunders | wp ≥ 0.10: **6 of 5,379**. wp ≥ 0.20: **zero, corpus-wide.** We are not losing to catastrophic single moves — see LEDGER. |
| validated search gains so far | **none.** Every arm tried is null or refuted (`LEDGER.md`) |

## The corpus is in — this section is now just how to verify it

Committed 2026-07-28. To confirm what you have:

```bash
python3 -c "import json;d=json.load(open('searchopt/positions/losses.json'));\
print(d['_provenance'], len(d['rows']))"
```

Expect provenance `d48x-1280-sb520-ema.cenn` and 5,379 rows. **A bare list instead of a
dict means a mine is mid-flight** — `mine_losses.py` checkpoints every 50 jobs as a plain
list and only wraps it in `{_provenance, rows}` at the end, so that is how you tell
"still running" from "done" (this tripped the check command that used to live here).

Re-mine only if the canonical net changes — the 404-game PGN is on disk, so it needs no
new match:

```bash
uv run --no-project --with chess python searchopt/tools/mine_losses.py \
    --pgn engines/runs/corpus-d48x/games.pgn
python3 searchopt/tools/classify_losses.py --losses searchopt/positions/losses.json
```

## Do this first

```bash
python3 searchopt/tools/walk.py --index 0        # a position from positions/ours.json
```

That answers method step 2 by itself: whose defect the position is, the forced ladder for
both candidate moves in both engines, and a descent to the node that manufactures the
error — then it prints the exact `searchdbg.py` command for that node. Work `ours.json`
only; `both.json` is position hardness (SF blunders there too) and `neither.json` is
in-game context or instability. Neither is a search defect you can fix.

## The three rules that were paid for in a lost day

1. **Eval is an axiom, not a hypothesis.** See above.
2. **Null the METRIC, not just the arm — and do it FIRST.** `arm.py` runs the
   opposite-direction arm automatically and refuses a verdict without it. This is not
   ceremony: `RED_GIVES_CHECK=+512` cleared a *seven*-sample inert null band and was
   reported as the day's one solid result; then `−512` — reducing checks *more* — cleared
   the same band. The metric was rewarding perturbation. Inert ±1 nulls cannot catch that,
   because they do not move along the arm's axis.
3. **Stay on ONE position.** No aggregate statistics during diagnosis. Aggregates always
   return a number, which is exactly why they are the escape hatch when a trace gets hard.

Only **proof-grade** outcomes count as evidence: a forced mate found, or a value SF
independently confirms. "Plays a different move", "score collapsed", "AUC improved" are
all satisfiable by noise.

## What is known about the losses

**Re-derived on the D48X corpus, 2026-07-28** (25 blunders; the D47V shares below it are
kept only for comparison):

| n | class | actionable |
|---|---|---|
| 21 | **NEITHER** — our own cold 110k search does not play the move | in-game context (warm history/aspiration) or argmax instability |
| 3 | **OURS** — SF avoids at 110k, our cold search does not | **yes — start here** |
| 1 | BOTH — SF blunders there too | no — position hardness |

**Order of work (owner, 2026-07-28): fix OURS first, then move on to the other losing
positions and see if those can be fixed too.** The corpus is a queue, not a ceiling.

Two cautions, both paid for:

* **The classifier has not been nulled.** OURS-vs-NEITHER is one free-search argmax per
  engine at one budget, and an inert ±1 flips ~25% of these positions (`walk.py`
  docstring). `walk.py`'s 5-budget ladder re-adjudicates each position as you walk it —
  trust that over the class label.
* **"We are confidently wrong" is weaker than it was written.** The 55-vs-10.5 headline
  compared OUR raw units against SF's printed cp; corrected it is 26.7 vs 10.5 = **2.54×,
  not 5.27×**. On the D48X OURS set it is 70.0 vs 58.0 = 1.21× at n=3 — no conclusion
  available. `classify_losses.py` now converts at the tool boundary.

⇒ The lever remains making refuting lines cheaper to resolve, so danger reaches the root.
But note the corpus says blunders are RARE (0.46%, none above wp 0.20): most of the −132
Elo is elsewhere.

## Open, in priority order

1. **Walk the OURS positions one at a time.** Only two were ever walked, and one produced
   a mechanism later refuted. This is the work.
2. Why RGC-style arms are non-monotone (512 misses what 256 and 1024 find) — a hint the
   search is chaotic at ply-boundary granularity, unexplained.
3. `DRAW_CONTEMPT` exists (default 0) with a real, measured mechanism — a perpetual worth
   −1 sets alpha and starves the winning sibling, 52k nodes vs 260k with the draw
   excluded — but **no admissible evidence**; its root effect died 0/6 under perturbation.
*(There is deliberately no net/training item on this list. See OUT OF SCOPE.)*

## Housekeeping

- Build the traced binary with `searchopt/tools/build.sh` (needs BOTH `search-control`
  and `tune`; without `tune` every arm returns byte-identical numbers and you will
  conclude "no effect").
- `Bench: 587683` and `CHECK_NNUE_GATE=1 bash tools/check.sh` (eval `d82caf8a…`,
  d13 `1972607`) must both hold after any change.
- Disk sits at ~73%; `old/` holds 423 archived files and nothing there is trusted.
