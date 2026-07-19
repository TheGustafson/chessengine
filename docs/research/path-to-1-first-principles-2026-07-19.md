# The path to #1, from first principles — are we on the right path?

Date: 2026-07-19. Type: strategic study (commissioned as a ground-up sanity
check, not a profiling pass). Method: work BACKWARDS from "ARCHON is the #1
engine in the world" to a realistic path from today's measured position, using
the experiment ledger (103 experiments, 42 nets) as ground truth for what
actually moves Elo for us. Citations: `exp-NNNN` = `ledger/experiments.jsonl`.

**Executive verdict: mostly YES on direction, NO on one thing — the plan keeps
deferring its own most-proven conclusion. Three independent studies (Integral,
what-holds-us-back, this one) now converge on distributed testing compute as
the binding constraint, D36 already resolved "rent it when needed," and the
evidence says "needed" was two weeks ago. Meanwhile the lever the plan ranks
first (eval/data) is the one whose one-box headroom the ledger shows is nearly
exhausted, and the lever it just closed (speed) is the only one that delivered
external Elo this week.**

---

## 1. Backwards from #1: the gap, decomposed

### 1.1 Where we are (measured, current builds)

- ARCHON (D40 720sb + opt1+opt2 speed package) vs **SF16 at equal time:
  −42.8 ± 14.6** (310g, 10+0.1, exp-0102). The classic incumbent measures the
  same: −41.9 ± 33 / −43.1 ± 20.2 (exp-0098). Both eval lines have converged
  at ≈ SF16 − 43.
- Measured local ladder above us: SF16 → SF18 = +113 ± 29 (triangle,
  exp-0060/0062/0064). SF18 → the 2026 frontier (SF-dev, Lc0 on its hardware)
  ≈ +20–50 more.
- **Total gap to #1: ≈ 175 ± 40 Elo.** The interim north star (SF16+30) is
  ≈ +73 from here. In CCRL-blitz-equivalent terms ARCHON is roughly
  3450–3480 — already plausibly top-15–25 class *if it were on the list*
  (it is not; see §5).

### 1.2 What the frontier has that we don't — and what is solo-closable

| Asset | Frontier scale | ARCHON today | Solo+one-box closable? |
|---|---|---|---|
| Eval data flywheel | 10^10–10^11 positions, regenerated on-policy as the engine improves | Rides SF's published binpacks (the only thing that ever worked for us — §2A) | **Partially.** Consuming their data: done. Producing better-than-their data: unproven (pilot pending), cloud-sized at scale. |
| Validation compute | Fishtest: ~10^7–10^8 games/yr, resolves +1–2 Elo patches at 1.5% acceptance | One box: resolves ±13 internal, ±15–30 vs SF16; D45 SPSA season shipped 8/8 nothing (exp-0075..0088) | **No — and it's rentable.** 5–10 spot workers ≈ 10–20× throughput for ~$100–300/mo (inside the stated ceiling). This is D36, decided but not stood up. |
| A decade of tuned constants | Millions of games of SPSA baked into every margin | SF defaults, never tuned for our tree; +50% LMR detune costs only −8 ± 13 (exp-0078) — a high plateau | **Not by hand.** Only minable at fleet scale. But the residual is smaller than believed: −25.5 ± 28.4/node vs Reckless at real depth (exp-0097). |
| Inference engineering | SF18: 7.4k instr/node | Was 16.9k; opt1+2 banked +17% NPS = **+28.8 ± 10.3 ET Elo for opt1 alone** (exp-0101) | **Yes — mostly done.** Remaining ~+8–15% NPS identified and low-risk. |

### 1.3 The first-principles observation the plan should internalize

ARCHON's velocity to date (0 → SF16−43 in weeks) was a **replication
dividend**: a decade of published knowledge — SF patch history, official
binpacks, Reckless/Integral architecture studies, bullet, the UHO book — was
consumed at AI-loop speed. That dividend runs out almost exactly at SF16-level,
*which is where we are standing now*. Past SF16+ε there is nothing left to
copy: Elo must be either **mined statistically** (fishtest-style, needs
throughput we don't have) or **discovered** (novel research, where our only
differentiated asset is the D41 program, currently play-negative). This is why
the binding constraint flipped from "which algorithm next" to "testing
throughput" — and why it flipped *now*.

---

## 2. Lever-by-lever ROI critique, against the ledger's win/loss record

### A) EVAL / DATA — the plan's #1 lever; the ledger says its one-box form is nearly tapped

**The win/loss record is lopsided and unambiguous.** Losses: every self-sourced
data bet failed, most catastrophically — Lc0 data −405 (exp-0006), −131.7 even
after the label-scale fix at equal volume (exp-0018), all mixes negative
(exp-0019/0021/0023/0024), own 5k-node datagen labels are 99.6% aspiration
bounds and quarantined, fishpack-position cells −328 to −478 (exp-0037/39/42),
king-bucket sparse-init −409 (exp-0041). Wins: **exclusively the official SF
binpack pipeline** — UHO transplant +47.8 (exp-0038), the width ladder to
+166 ET over net-0 (exp-0048), +44 (exp-0054), farseer stage-2 +55 (exp-0056),
and D40-on-UHO winning per node (+30.2, exp-0067). We are proven
**data-consumers, not yet data-producers**.

**And the consumer road just ended.** In the last 48 hours, every remaining
rung was measured and closed: stage-1 saturated (1080sb = −9.3 ± 15.4,
exp-0100), stage-2 dead for D40 (−12.7 / −72.9, exp-0094/0095), classic width
saturated at 2048 (4096 = +15 n.s., exp-0058), king-bucket factorization null
(+6.9 ± 18.0, exp-0099). One-box eval headroom from here is realistically
+10–30 external, not the +50+ the lever ranking implies.

**Is SF18-relabel/obs a detour?** In principle, no — it is the *only* data
route that can eventually exceed the teacher: same positions, deeper labels
than the binpacks' own (the pre-gate found 15% sign-flips, 136cp RMS in
decisive positions — real label error to harvest). But the base rate of our
"data quality" interventions is ~0-for-15, so the prior must be skeptical and
the pilot gate (deep-label net ≥ +15 fixed-node vs cheap-label twin; 3/10
chunks done) must be respected as a **kill-switch, not a formality**. The
mis-investment risk is real and already materializing: the obs-schema routing,
cloud-factory rebuild, and GCP relabel factory are substantial engineering
built *ahead* of the pilot verdict. If the pilot is null, all of it is
stranded. **Stop building data infrastructure until the pilot pays.**

**Untested rung worth one screen later:** D40 at 1024 wide (SF18's own
equilibrium is 1024+threats; ours is 768). NPS-taxed, so it only makes sense
after the speed tail is banked and only via a hard ET gate — rank it below the
pilot and the fleet.

**Verdict: B-.** Right long-term theory (relabel-to-exceed-teacher), correctly
gated pilot, but over-weighted in the current ranking and over-built ahead of
evidence. At full scale it is a D36/cloud program anyway — which is the point.

### B) SEARCH — machinery complete; the residual is real but un-minable locally

The ledger killed both the pessimistic and the optimistic story this week. The
−87/node deficit vs Reckless was stale: re-measured with eval controlled it is
**−49 ± 25 at 100k and −25.5 ± 28.4 at 200k nodes** (exp-0096/0097) — at real
depth our search is statistically near a top-2 engine. Meanwhile the mining
attempt failed exactly as fishtest math predicts: the 8-decision SPSA season
shipped nothing (exp-0075..0088), and the landscape is a plateau (exp-0078).
The one big-ticket item left in the queue paid huge (S1 correction history:
+72.9 ± 16.2 internal, exp-0074) — **but deflated to ≤ +15 n.s. at the SF16
anchor** (exp-0091). There are no more S1-sized items on the list; what remains
(SF16-late mechanics bundle, IIR retest, optimism, nodesEffort TM) sums to
maybe +10–30 internal, deflating 2–3×.

**Can a solo effort close the search gap by hand-porting patches?** The ledger
answer: the porting phase already succeeded (that's the −87 → −25 move) and is
over. What remains is constants, and constants are a statistics problem, not a
cleverness problem — ~48k games per +2 Elo verdict (design/14 §3.8). Without a
fleet, further search grinding is negative-ROI. With a fleet, the global SPSA
retune vs the shipped net (matrix item #1, +15–40 est.) is the single largest
pool.

**Verdict: C+ as a solo program, B+ as the first fleet workload.** Keep the
Reckless-net instrument (deflation-free fixed-node bed) as the acceptance test
for any search change.

### C) SPEED — just proved it's the best Elo per unit effort on the board; finish it

The most important measurement of the month is exp-0101: **bit-identical
binaries differing only by +13–15% NPS = +28.8 ± 10.3 Elo at equal time** —
~2× the D39 rule, because our eval is expensive per node, so extra nodes pay
disproportionately. This week's +17% NPS moved the SF16 anchor from ~−63/−78
to −42.8 for zero retraining risk. The remaining identified tail (prefetch
+2–5%, static dispatch +1–3%, observer constants +2–4%, quant unification
+2–3% at next net rev, huge pages, bulk parse) sums to ~+8–15% NPS ≈ +10–25
internal ET Elo at the measured slope. It is days of work, bench-node-verified,
and de-risked by the exact harness that just worked twice.

**Verdict: A. Not diminishing returns yet — the slope was measured this week
and it is steep.** Finish the package, rebuild PGO with it, then declare speed
closed unless a re-measured slope says otherwise. (Also: this lever is the
reason D40 promotion is now in reach — D40 was never quality-limited, only
NPS-limited.)

### D) HELM / D41 — the only differentiated asset, and correctly parked; park it more explicitly

Record: v1.1 play test **−29.6 ± 12.9 at fixed nodes** (exp-0092) — negative
even before the NPS tax. The one positive artifact ever (gen-0 bilinear head,
+68.6 ± 23 at 2k fixed nodes, exp-0029) never survived equal time. First
principles are against it *at this stage*: a learned ordering prior competes
with a history stack that is nearly free and already excellent, while paying
inference tax at an engine whose measured Elo-per-NPS slope is the steepest
we've seen. Every CPU frontier engine converged on cheap heuristics + massive
tuning; the learned-allocation counterexample (Lc0) required a different
search paradigm and GPU inference.

**But do not kill it.** It is the only thing in the portfolio that could, in
principle, beat SF at something SF cannot copy overnight — everything else we
do is convergent replication. The honest framing: **D41 is the post-SF16+30
research program.** Its gate should be explicit: (a) eval foundation frozen,
(b) fleet live (so its game tests are cheap), (c) a pre-registered fixed-node
+X *with NPS-tax accounting* before any new training spend.

**Verdict: correct to deprioritize; make the re-entry condition explicit so it
stops leaking box-time (gold-v2 trains, capture harvests) in the interim.**

---

## 3. The real binding constraint — verdict

**Testing/measurement throughput, unambiguously — and more precisely: the
inability to SEE sub-30-Elo external progress.** Three independent lines of
evidence now agree:

1. **The Integral study's arithmetic** (the only observed solo path to top-6:
   ~half the explanation is a free distributed fleet; several million games).
2. **The what-holds-us-back study** (both sweeps land on D36 as the meta-
   constraint).
3. **This week's ledger**: the SPSA season burned days to ship nothing
   (resolution floor); the SF16 anchor wandered −78 → −63 → −43 across
   100–310-game matches whose CIs all overlap (exp-0060/0091/0098/0102) — we
   spent three days *unable to know if we had improved*; and the two largest
   queued programs (search retune, data at scale) are both explicitly
   fleet-gated by their own docs.

The path from −43 to +30 vs SF16 is ~5–8 increments of +10–20 external Elo
each. Each increment needs 1000–3000 games against external anchors to verify.
That is exactly what a rented fleet does and exactly what one box cannot.
Every week without it, we pay the cost in a currency worse than Elo:
**unfalsifiable progress**.

The secondary constraint, stated honestly: **idea supply past SF16**. A fleet
without ideas mines ~+30–50/yr (fishtest's own rate at maturity). Our current
queues (A+B+C) plausibly sum to +60–100 external — enough for SF16+30, not for
#1. Past that, the flywheel (data + tuning + D41) has to generate the ideas.

---

## 4. What's missing from the plan

1. **A standing external anchor protocol.** The lesson "stop banking internal
   Elo" is written but not operationalized. Institute: one 2000+ game SF16
   anchor match per week (fleet-run, pinned book/seeds), ledgered as the ONLY
   progress metric. Everything else is screening.
2. **Any public strength signal.** No CCRL submission, no lichess-bot, no
   released binary. Cheap, converts internal claims into external truth,
   and top-20-class today is a genuinely marketable result. (Planned post-M9;
   never scheduled.)
3. **Syzygy tablebases.** Declined under D31c for FFI hygiene — right call
   then; wrong call for the rating-list era. Worth +5–15 at blitz, more at
   list TCs, and every engine above us probes them. Scope a safe wrapper or
   accept the licensing/FFI cost as a non-default feature. Not urgent; missing
   entirely from the roadmap.
4. **SMP + list-conditions validation.** The +279 Elo 4-thread scaling number
   is PSQT-era (exp-0002, pre-NNUE). No NNUE-era SMP re-measurement, no
   Hash=256MB+ validation, no 2+1/40/15 TM robustness check. If we ever submit
   to a list, these are the conditions that count.
5. **A STOP list.** The ledger shows extraordinary throughput diffused across
   many parallel programs (obs platform, cloud factory, helm trains, attention
   trainer, transformer explorations, media). The wins concentrated in exactly
   two veins (official data, speed). Breadth is the visible process risk;
   nothing in the plan prunes it. §5 supplies one.
6. **An explicit feasibility ladder** (what #1 actually requires, below) so
   that milestone claims and resource asks stay honest.

---

## 5. The recommended path

### The next 5 moves, ROI-ranked

1. **Finish the speed tail (days, ~free).** Items #3/#4/#5/#7 (+ huge pages),
   rebuild `probe-sota-pgo` with the full package, re-run the NPS→Elo slope
   check. Expected +10–25 internal ET; measured-slope-backed. Then close the
   program.
2. **Stand up D36 THIS WEEK (the decision is already made).** 5–10 GCE spot
   workers; OpenBench proper or even sharded-fastchess on the existing
   tools/cloud governance rails. First three workloads, in order: (a) the
   weekly 2000-game SF16 anchor (ends the ±30 fog permanently); (b) the global
   SPSA gainer season vs the shipped net at scale (matrix #1, +15–40, the
   largest single pool left); (c) SPRT queue for the SF16-late mechanics
   bundle / IIR / optimism / nodesEffort. Budget $100–300/mo, inside the
   stated ceiling. **This unblocks both stalled programs simultaneously and is
   the only move that changes the slope of everything else.**
3. **Run the label-value pilot to verdict; obey it.** ≥+15 → the targeted
   SF18-relabel program is validated: scale it via the (already-built) Tier-F
   pipeline as a cloud campaign. Null → **hard-stop the entire relabel/obs
   factory line** and take the remaining consumer rungs instead (full official
   multi-binpack interleave recipe; later one D40-1024 screen). Either way:
   one eval train in flight at a time, ET-gated.
4. **Promote D40 and retire the classic line.** After move 1, D40 should clear
   the ET gate vs classic (it is at parity pre-tail, wins +30/node). Single
   incumbent, single inference path to optimize, half the maintenance. Keep
   classic as a frozen control.
5. **Institutionalize the anchor cadence + go public.** Weekly fleet anchor
   match as the progress metric; lichess-bot account; prepare a CCRL
   submission once SF16-parity is verified. External truth, recruitment of
   attention, and a free regression suite.

### STOP doing (until their gates flip)

- **Helm/D41 training runs** (gate: eval frozen + fleet live + pre-registered
  NPS-accounted target). The queued d41-gold-v2 train: cancel, not defer.
- **All classic-line training** (saturated: width, stage-1, stage-2 all
  measured flat/negative).
- **Building more obs/cloud data infrastructure ahead of the pilot verdict.**
- **One-box SPSA and sub-15-Elo search hand-tuning** (resolution floor is
  measured; it's fleet work now).
- **Media/visualization side-quests** (already user-postponed; keep it that
  way).

### Honest feasibility and the target ladder

- **Beat SF16 at equal time (+43):** plausible in 1–2 months (speed tail +
  fleet SPSA + D40 promotion), likely by 3. High confidence.
- **SF16+30 (the north star, ≈ top-10-CCRL-class):** 3–9 months WITH the
  fleet; likely out of reach without it (the prior study's conclusion stands).
- **Integral-class / top-5-solo territory (≈ SF16+50–100):** 6–18 months of
  sustained fleet + a validated data flywheel. This is "strongest
  solo/AI-built engine" achievement level and a defensible public goal.
- **World #1 (beat SF-dev at equal hardware):** **not a plannable milestone
  from here, and it should not pretend to be.** It requires out-improving an
  opponent that gains ~+30–55/yr backed by ~10^8 games/yr of validation and
  the Leela data ecosystem — no solo or small-team effort has done it in the
  NNUE era; Lc0 with a large team and a GPU fleet has not done it
  consistently. The only honest routes are (a) SF stagnates (not in our
  control) or (b) a genuine algorithmic discontinuity that fishtest cannot
  quickly copy — in our portfolio that is the D41 learned-allocation bet plus
  the AI-loop's iteration speed, both unproven at positive Elo. Keep #1 as the
  north star that *justifies keeping the D41 program alive after SF16+30*;
  spend nothing on it before then. Probability of actual #1 within 3 years:
  low single-digit percent. Probability of top-10 CCRL and the strongest
  solo-built engine trajectory ever recorded: real, and worth exactly this
  plan.

### The one-line answer to "are we on the right path?"

The engine-building has been close to flawless — the mis-step is strategic
sequencing: the plan is still allocating its scarcest resource (one box, one
loop) to the lever with the least remaining one-box headroom (eval/data),
while the decided-but-unbuilt fleet (D36) — which every line of evidence says
is the actual bottleneck — waits. Invert that this week: finish the speed
tail, stand up the fleet, let the pilot's verdict route the data program, and
hold every claim to the weekly external anchor.
