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
| loss corpus | **404 games vs SF18** (−132 Elo, 174 lost games, 10,112 moves oracle-scored) → `positions/{ours,both,neither}.json`. A blunder = **wp_loss ≥ 0.06 (~88 cp) from a position better than −2.5 pawns** — a move played from an already-lost game is not a blunder, it is a loss. Expect **~47 positions**; raise `--wp-min` to 0.10 for the ~9 severe ones. |
| validated search gains so far | **none.** Every arm tried is null or refuted (`LEDGER.md`) |

## FIRST: is the corpus actually there?

The mining run was left detached at the end of the previous session and takes ~100
minutes. Check before you trust `positions/`:

```bash
ls -la searchopt/positions/{losses,ours,both,neither}.json
python3 -c "import json;d=json.load(open('searchopt/positions/losses.json'));\
print(d['_provenance'], len(d['rows']))"
```

* **All four exist and provenance says `d48x-…`** — good, work `ours.json`.
* **Only `losses.json`** — the mine finished, the classify did not. Run:
  `python3 searchopt/tools/classify_losses.py --losses searchopt/positions/losses.json`
* **Neither, or provenance names a different net** — re-mine:
  `uv run --no-project --with chess python searchopt/tools/mine_losses.py --pgn engines/runs/corpus-d48x/games.pgn`
  (the 404-game PGN is on disk, so this does not need another match).
* `ours.json` may still be the **D47V-era 14-position set** if the chain never landed —
  it has no provenance stamp, which is how you tell.

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

Split at the in-game budget (110k nodes), from the D47V corpus — **re-derive these shares
on the new corpus, the numbers may move, the categories will not**:

| share | class | actionable |
|---|---|---|
| ~14% | SF blunders there too | no — position hardness |
| ~41% | near-tie for **both** engines (23 vs 28 cp separation) | no — anyone errs |
| ~14% | warm history/aspiration flips it (the warm **TT helps**) | narrow |
| **~32%** | **SF avoids at 110k, we don't** | **yes — work here** |

**The character of that 32% is the load-bearing finding: we are confidently wrong, not
uncertain.** Top1−top2 separation ~55 cp for us vs ~10 for SF, SF near-tied in 64% of
them. We commit conviction to a losing move; SF declines to commit.

⇒ **Sharpening the eval would make this worse.** The lever is making refuting lines
cheaper to resolve, so the danger reaches the root and flattens a confidence that should
never have formed.

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
