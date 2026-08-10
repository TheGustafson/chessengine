# ARCHON DEVIATION PROGRAM — ranked, executable (2026-08-10)

**Goal (owner):** deviate MORE from Stockfish and Reckless — a distinct engine
identity — via clever original mechanisms AND/OR simplification. Deliverable:
a ranked program, not an essay. Every item names its first experiment and an
instrument actually powered for its expected effect.

**Method constraint, up front (the d62 lesson, ledger exp-0169/0170):** a
600-game screen (±17 Elo) refuted its own +10..+22 "gate ladder" when re-run
at power. No proposal below is allowed a verdict from an instrument whose null
band exceeds its expected effect. The instrument menu, with measured powers:

| instrument | null / power | decides |
|---|---|---|
| gate census `fhf` (`census70`) | ±0.15 pp (re-search rate ±0.03 pp) | node-level ordering/gate effects, screening only |
| 1000-position paired sign test (`tools/perf70/screen70.py`) | 505 ± 6 across inert seeds | directional node effects |
| `ipn70.sh` instructions/node | spread ≤0.01% (`instructions:u` is the only counter that survives its own null) | executed-work claims |
| fn25k non-inferiority GSPRT, elo0=−3 elo1=+1, segmented `chain70.sh` + `pool_sprt.py` | concludes ~3–8k games | "removal is not materially worse" |
| fixed-node match, n ≥ 3000 | ±7–8 Elo | ≤10 Elo mechanism claims (the clamped-SEE standard: +11.00 ± 7.20 @ 3350) |
| promotion SPRT elo0=0 elo1=5 | sequential | anything becoming default |
| real-TC quiet box | — | ALL time-management claims (`nodestime` gave +52.6 where the real clock gave −31.4; never again) |
| twin-position move-change panel | discrete, frame-free | eval-input changes (castling/rule50) |
| equal-time gauntlet vs canonical | ±7–8 @ 3000+ | net promotions (never a 600-game screen) |
| `prune_budget.py` / `gate_census.py` | node-contribution only | classifying gates live/dead — NEVER Elo |

**Box constraint:** the d63 700sb warm-restart owns GPU+RAM. CPU fixed-node
matches at modest concurrency are the demonstrated compatible workload; no
cargo/rustc while the trainer is active (negotiated build windows); real-TC
and GPU-gated arms queue behind d63 or run in explicit pauses.

---

## Part I — THE SIMPLIFICATION LADDER

### 1.0 Where the ladder already stands (do not re-derive)

The 2026-08-01 program (`search-simplification-plan-2026-08-01.md` +
execution log) landed: tier 0/0b (313 → 225 tunables, ~3,300 lines deleted,
bit-identical by construction) and **basket A** (the ARCHON-original null
trio; shipped on owner ruling, SPRT −3.66 ± 11.88 @ 380 games — consistent
with neutral, not evidence of it). Today: **223 tunables, 60 ever in any SPSA
campaign, 163 running SF's values on pure faith, and only 3 values ≥20 that
are not SF's** (`param_provenance.py`, exit 0, 2026-08-10):
`ASPIRATION_MSS_DIVISOR` 6776 (derived), `QS_CAPTURE_DEFENDED_NUM` 56
(study 70), `QUIET_SEE_PENALTY` 49152 (study 71). Three stale campaign
entries need cleanup (`FALLING_EVAL_*` in s4-searchgap, `QUIET_OFFENSE_FACTOR`
in s5-ordering — their params died in tier 0/basket A).

### 1.1 The census of what remains, by category

**(a) Multi-term formulas whose individual terms have never been measured on
ARCHON** — the deepest untested surface:

- LMR finalization: ~18 accretion terms (`RED_TTPV_*` ×6, `RED_CUTOFFCNT_*`
  ×5, `RED_TTCAPTURE`, `RED_ALLNODE_*` ×3, movecount/cutnode/statScore docks).
- Singular double/triple margins: 10 knobs passing **93% / 65%** of singular
  verdicts (census) — margins that almost always pass are not discriminating,
  and the three extension knobs measurably do not compose (+0.60/+0.30/+0.27
  alone, −0.21 together).
- Fail-low parent bonus (`FL_*`): **16 tunables** — five additive "bump"
  terms with magic margins (103, 78), a stat-score divisor, a three-way
  weight split. Zero campaign coverage, fires at every fail-low.
- History-update weight family: ~20 knobs (bonus/malus depth ramps, the
  geometric quiet-malus decay, per-table weights 824/1136/956/1366/1518/683…).
- Fitted const tables transcribed verbatim: `LMR_DIVISOR[16]` (wobbles
  2782..3599 around ~3050), `CONT_CONSISTENCY_MULT[7]` (96..127 around ~110),
  `CONT_OFFSET_WEIGHTS[6]`.
- Correction-history blend: 5 read weights + 4 update numerators + 4 bonus
  shapes (~13 knobs).

**(b) SF mechanisms that exist for SF's regime, not ours** (its own *Scaler
comments, or our measurements):

- IIR aggressiveness: SF's source says "(*Scaler) Making IIR more aggressive
  scales poorly"; measured HERE at −24.17 ± 15.11 when extended to upper
  entries. The shipped IIR earns its keep (+24.9% contribution) — the point
  is that SF's *coupling choices* around it are regime-specific.
- NMP verification: **29 fires in 1.2M nodes** at d12 — zugzwang insurance
  priced for fishtest depths (kept in basket B's bisect-first slot because
  zugzwang losses are catastrophic-shaped).
- TT-cutoff hardening trio, eval-diff nudge, conthist pruning (the ONE gate
  whose ablation *shrinks* the tree, −8.8%): late-generation SF accretions,
  each ~1–3 Elo on 100k-game fishtest bars, unfalsifiable passengers here.

**(c) Never-tuned:** 163 of 223. Even of the 60 covered, SPSA moved almost
nothing (`RED_MOVECOUNT` drifted ~0 over 3,280 games while neighbours moved
100+). The honest count of constants that have ever demonstrably mattered on
ARCHON is ~30–40 — the ones the prune budget and fire census show carrying
the tree.

### 1.2 The ladder — named batches, in execution order

Standing method for every games-gated batch (all rules paid for): fn25k
one-match A/B vs current tip, same-source both-PGO pair, `EvalFile` verified
on the live argv, segmented chain + pooled GSPRT[−3, +1], census null BEFORE
the match, bisect by halves on H0, ledger row per match. Bench/NNUE-gate pins
move only with the landing commit.

**S1 — SINGULAR COLLAPSE (spec already written, execution log §2 stage 3).**
Double/triple margin family + tmh coupling → one margin (`SE_EXT_MARGIN ≈
−160`, the bar the census says passes 93% anyway), extension ∈ {1,2}; the
`tt_move_history` scalar dies with its only consumers. **Net −13 tunables**,
two pure functions, the SE block becomes readable. Gates: census first;
non-inferiority SPRT; endgame-weighted book alongside; `mate_in_3` >2σ
tripwire (extensions are mate-line machinery). This is the deepest single
identity statement available: SF's most filigreed modern mechanism replaced
by a one-parameter form.

**S2 — SF-CARGO BASKET (spec already written, execution log §2 basket B).**
Eval-diff nudge (5 knobs, 2 history writes at 639/1k nodes — one of the
busiest blocks in the engine), conthist pruning (tree-negative), TT-cutoff
hardening trio, NMP verification. **~−10 tunables, ~150 lines.** On H0 bisect
{1.5,1.6} vs {1.7,1.8} and restore NMP-verify FIRST — NO ACCEPTABLE BLUNDERS
outranks knob count.

**S3 — FAIL-LOW LEDGER COLLAPSE (new).** `FL_*` 16 → ~4: keep
`scaled_bonus = min(A·depth − B, CAP) · scale` with ONE flat scale; delete
the five conditional bumps and the three-way weight split (fold to the
conthist weight). **−12 tunables.** Mandatory control: history MAGNITUDE
feeds LMR through `RED_STATSCORE_NUM` (study 70's standing finding — a
history arm's node win is a pruning win until a magnitude control says
otherwise), so the census screen must include a magnitude-matched arm before
the SPRT. Expected effect ~0; the win is surface.

**S4 — FITTED-TABLE FLATTENING (new).** `LMR_DIVISOR[16]` → one constant
(~3050); `CONT_CONSISTENCY_MULT[7]` → one (~110). **−21 constants** (const
arrays, not tunables — this is line/identity surface, not SPSA surface).
Screen: paired sign test + census; then ride inside the next games-gated
batch rather than buying its own match (expected effect ≈ 0 by construction:
the tables are near-flat).

**S5 — HISTORY-WEIGHT FOLD (new, higher risk).** The per-table update weight
family → shared bonus/malus weights (~20 → ~8). Same magnitude-control
discipline as S3, run as its own basket AFTER S3 survives (they share the
mechanism; landing both at once un-attributes a failure).

**S6 — CORRECTION-BLEND FOLD (conditional).** 5 read weights → equal-weight
(or 2 degrees), update numerators → 1. FIRST check campaign coverage — the
D45 season opened with a correction-history campaign; if SPSA actually moved
these, they are load-bearing and stay. Correction history itself is a KEEP
(the engine's only online learned-error machinery, the template any learned
head must beat).

**S7 — STAGE-B SPSA: END THE TRANSCRIPTION LINEAGE.** After S1–S5, the
surviving load-bearing set (~30–40 knobs: LMR family, RFP margins, NMP bar,
futility bases, aspiration delta, history scales, correction weights) gets
retuned on ARCHON's OWN games. Cost, quantified from the live kit
(`tools/spsa/`, fn20k, 8 games/iter): 2400 iters ≈ 19,200 games ≈ 8–10 h of
box time per campaign at the measured ~1.6 s/game and concurrency 4–6, CPU
only, tune build (non-PGO is correct: fixed-node, same binary both sides).
Three campaigns (reductions+LMR / margins+futility / history+correction) ≈
2–3 days interleaved with training. Acceptance per campaign: equal-time SPRT
ship gate on a quiet box (D45 protocol). **After S7, `param_provenance.py`'s
transcription screen stops being a faithfulness check and becomes a
divergence certificate — the numbers are ours.** This is the cheapest
wholesale identity move in the program.

### 1.3 The double win, and its honest odds

Simpler+stronger has happened here three times: the clamped SEE penalty
(+11.00 ± 7.20 — the version that wins every cheap instrument is the version
that loses; SF's exact form is not our optimum), the s600 rescale
(+14.78 ± 5.45 — a 4-byte simplification of the frame program), and
`ASPIRATION_MSS_DIVISOR` (derived, not tuned). Expect most batches to be
neutral — that is the design (non-inferiority) — and treat any batch that
WINS as a study-71-class finding to be mechanism-attributed, not banked
silently.

### 1.4 What the ladder must NOT touch (receipts)

Per the re-measured prune budget (post-s600, all gates live): LMR core
(+162.6% contribution), RFP (+114.6%), parent futility (+45.3%), IIR
(+24.9%), NMP core (+20.9%), quiet SEE (+16.7%), razoring (+15.7% — the July
"harmful" verdict is stale), capture futility, ProbCut, LMP (pays in executed
ops), the staged movepick + good/bad SEE split, TT probe/store discipline,
qsearch discipline, correction history, `stat_score` + `RED_STATSCORE` dock
(the learned-reduction seam), `set_quiet_prior` (the policy seam), the
soft-stop factor product (the TM-head seam).

---

## Part II — ORIGINAL-MECHANISM PORTFOLIO

Verified in-tree state as of 2026-08-10 (not the audit's snapshot). Ranked by
tractability × expected Elo × uniqueness. "Uniqueness" = does any reference
engine ship it.

### Ranked table

| # | mechanism | in-tree today | tractability | expected Elo (honest) | unique? |
|---|---|---|---|---|---|
| M1 | DIS-TM confirmation (learned root-level time allocation) | arm + random-redistribution control BOTH implemented (`ARCHON_DIS_TM_C`, `dis_tm_random`, census-calibrated pivot 127) | HIGH — one match + one control | +5–10 if the +9.38 ± 10.64 (LOS 95.8%) holds; 0 if the control matches it | YES — no engine ships a learned "keep thinking" factor |
| M2 | Pin/x-ray ordering signal class | nothing yet; `StateInfo` already carries blockers/pinners; census instrument ready | HIGH — days | 0–10; the CLASS is proven (SEE penalty +11) but each member is a lottery ticket | term would be ours; class opened by study 71 |
| M3 | Castling + rule50 NNUE inputs | NOT plumbed; blindness measured (800/800 bit-identical twins vs SF move-change 39%/37%); ~40-line trainer change; 5 dense inputs at L1 bottleneck, zero accumulator cost, zero-init = warm-compatible | MEDIUM — GPU-gated, next net generation | unknown; 0–15. The 39% move-change is an upper bound on *relevance*, not an Elo estimate | YES — SF and Reckless are provably equally blind |
| M4 | Rule50-bucketed correction history | not in tree; Reckless-idea (notes 05 §3.2), neither we nor SF ship it; our corrhist is table-per-key, bucket axis is a small change | HIGH — days | 0–5 | semi (attributed idea, no shipping engine has ours+minor-key+buckets) |
| M5 | Draw contempt revival (`DRAW_CONTEMPT`, default 0) | knob + mechanism in tree; P2 loss class documented (perpetual sets alpha, starves the win 5×) | MEDIUM — needs a mined perpetual-class validation set; its root-level statistic already died an inert-null death (0/6), so GAMES ONLY | small aggregate; removes a catastrophic class | YES vs SF (none since SF12); not in Reckless notes |
| M6 | ThreatRemap learned FT compression (60,720 → 8–12k) | engine side LANDED + measured (eval-bit-identical arms: −1.49% cycles quiet, −3..−16% loaded, −69 MB net, −316M-cycle launch); trainer refit panel built, armed, GPU-gated | MEDIUM — panel launch is owner's call, queues behind d63 | ~+2.5 Elo speed quiet, more loaded; retrain may add quality | YES — SF structurally cannot retrofit (fused release pipeline) |
| M7 | Root re-scoring mid-iteration | not in tree; Reckless notes 05 §1.3; SF never does it | MEDIUM | 0–5 | semi (attributed) |
| M8 | Adaptive eval cost (smallnet gate) | gate machinery LANDED (`sota_context.rs`: `ARCHON_SMALLNET_THRESH`, material-imbalance prefilter, `ZERO_THREAT` channel, CEN3 arch for the small net) — inert without a trained small net | MEDIUM-HIGH for the ceiling; GPU-gated for deployment | potentially 10–30 speed-axis (75% of node time is eval; PSQT path is 2.13 vs 0.54 Mnps) — OR ~0 if the node fraction is low | deviation from our own reference: eca43a97 DELETED smallnet; SF18 shipped it. For a 1280-wide net the economics differ from both |
| M9 | Learned counterfactual LMR (design/19, the helm) | ALL infra in tree (`search_trace.rs` obs v3, `search_intervention.rs` replay, `search_control.rs` contract); Phase-1 held-out 0.706/0.258 but in-game NULL at gen-0 wiring (exp-0086) | LOW-MEDIUM — a research program, not an experiment | unknown; every learned consumer so far is null, AND the incumbent is strong: a node's own inputs predict its re-search at AUC 0.8772 | YES — flagship; no counterpart anywhere |
| M10 | WDL/variance heads → root consumers (TM decisiveness, aspiration width) | trainer has the heads; d63 line threads the DIS trailer; deleted TM hooks re-enter as ONE line in the soft-stop product (by design, plan §6) | gated on a canonical net that actually carries heads | 0–8 | YES for the consumer shape |
| M11 | Age-aware TT root-cutoff trust | pre-designed arm, OWNER-GATED: only if the NEITHER blunder class survives better labels (stale-anchor verdict: carryover is +39.4 net; fix labels first) | blocked until post-d63 re-mine | class-removal value | YES |
| M12 | Frontier self-labeling flywheel as a *feature* (data moat) | LIVE — d63 mixes sfdev5k-a (19.7M own-frontier positions, deep-relabeled) at 5%; d53mix20 was the first net our own data improved | running | this IS the current main line | YES — none of the solo references has the obs store / ledger / mining loop in-tree |

### First experiments, in detail

**M1 — DIS-TM.** The only positive learned-consumer signal in ~10,500 games
of trying. First experiment (already fully specified in LEDGER 2026-08-07):
(1) re-run the arm at n ≥ 3000, 5+0.05, vs the **random-redistribution
control** (same ±spread assigned by position hash — `c=0` is NOT the control,
because at fixed TC the arm only *moves* time); (2) if dis beats random,
promotion SPRT at real TC on a quiet box (10+0.1). Real TC ONLY. Consumes no
GPU. If the control ties the arm, the head contributed nothing and the
redistribution did — report it, keep whichever is simpler (which would itself
be a finding: a *contentless* redistribution factor worth Elo).

**M2 — pin/x-ray ordering.** Study 71 named the blind spot: SEE mis-prices
pins, x-rays, sacrifices, and the whole seven-plane history stack is blind to
static-exchange facts. Concrete first candidates, in census order: (i) cancel
or dampen `QUIET_SEE_PENALTY` when the destination's defender is pinned to
its king/queen (the SEE verdict is a lie exactly there); (ii) a capture-
ordering bump when the victim's sole defender is pinned. Instrument ladder:
census fhf (±0.15 pp null) with the study-71 predicate-vs-demotion RANDOM
control, paired sign test, then a 3000+ game fixed-node match, then SPRT.
Expected honesty: most members of this class will die at the census; the SEE
penalty precedent says roughly one in several such predicates pays.

**M3 — castling/rule50 inputs.** First experiment: twin D-arch nets ± the 5
inputs on the SAME corpus (a gated arm of the post-d63 generation — the d63
run itself stays data-only, one change at a time). Judge on (1) the
twin-position move-change class (does the net now separate the 800 twin
pairs, and in the oracle's direction — a discrete, frame-free instrument),
(2) heldout on rule50>40 / castling-live slices, (3) equal-time gauntlet at
n ≥ 3000. NOT on aggregate heldout loss (loss is not Elo). Trainer change is
~40 lines in `model.py` + spec/loader plumbing mirroring the OBL1 pattern
(net-carried section, legacy default = bit-identical old nets — the
acceptance test is the eval-hash pin, same as OBL1).

**M4 — rule50-bucketed corrhist.** `(rule50.saturating_sub(8)/8).min(15)`
bucket axis on pawn-corrhist (16× its 12 KiB table region, not the 12 MiB
pawn *history*). First experiment: implement behind a knob, census+cp screen
(the corrhist family's effect shows in eval-error reduction — use the
staticeval-vs-deep-oracle MAE panel plus nodes sign test), then
non-inferiority basket or 3000-game match. Attribution note: Reckless idea,
documented as such (D27 discipline).

**M5 — draw contempt.** Do NOT re-litigate the root statistic (it died 0/6
under inert perturbation). First experiment: mine the perpetual/repetition
loss class from the current corpus (`mine_losses.py` + classify on
repetition-draw outcomes), require ≥3 independent losses of the mode, then
small `DRAW_CONTEMPT` values (our internal frame: ~10–25) at fixed nodes
n=3000 AND verify the ≥3 specimens flip. One position is a patch, three is a
fix. Uniqueness: a *principled, loss-derived* contempt — SF deleted contempt
entirely; engines that keep it (Komodo-era) tune it for rating-list draws,
not for a diagnosed alpha-starvation mechanism.

**M6 — ThreatRemap.** First experiment is already built: the four-arm refine
panel (`tools/nnue/refine_panel/`, oracle / headbucket / constant-row /
null, all sb0 bit-parity-verified) — launch when the GPU frees, owner's
call. Deploy gate: equal-time games (the −1.5% cycles converts at the
measured ~0.45× instructions→wall ratio; claim +2–3 Elo, not more), plus the
69 MB net and cheaper launch as engineering wins. The identity value (a
net-carried learned feature code) exceeds the Elo value; say so in any
writeup.

**M7 — root re-scoring.** After each completed root subtree, re-sort the
remaining root moves from just-updated histories (SF sorts once per
iteration). First experiment: implement root-only (zero interior-node risk),
measure nodes-to-depth on the root census + sign test; then 3000-game match.
Cheap, contained, attributed.

**M8 — smallnet ceiling.** The LEDGER's uncosted one-hour stub-and-diff:
(a) node-level fraction tripping `|simple_eval| > T` at various T (the gate
machinery already exists — this is a census run, no training); (b)
instructions/node PSQT vs full net (known: ~4×); (c) multiply. If the node
fraction at a sane T is <10%, the route dies for free. If it is 30%+, a CEN3
small net becomes a post-d63 training item and the fidelity cost gets priced
with `evaldiff.py` exactly as study 8 priced quantisation. Honesty: eca43a97
deleted this for complexity — our defense is that our width/cost point is
2.4× theirs, which is a measured difference in the economics, not a hope.

**M9 — learned counterfactual LMR.** Keep as the flagship RESEARCH line, not
the next experiment. The validity lesson that killed the DIS interior
consumers governs the design: train ON the decision distribution (the
`search_trace` rows ARE interior nodes — this is exactly what design/19
already specifies, and what the DIS trunk never had). The bar to clear is
named and high: the node's own inputs reach AUC 0.8772 on re-search
prediction; a consequence head must beat THAT stacked, not merely beat zero
(pre-registered gate: stacked − own-inputs ≥ +0.02 AUC, the same gate that
closed DIS-RFP honestly). First concrete step when GPU frees: Phase-2
reduction-risk head on LmrObservation v3 rows with the intervention replay as
the offline validator; authority starts at clip(ΔR, −2, +2) veto on the
top-decile worst predictions, shadow first. Games decide, at 3000+.

**M10 — WDL/variance root consumers.** One-line re-insertions in `soft_stop`
/ aspiration setup by design. Gated: wait for a canonical net that carries
the heads (the d63 line threads DIS; WDL/σ sidecars are a training decision,
not an engine one). When they exist, the TM factor follows the M1 protocol
(real TC + random control); the aspiration-width factor is root-level and
fixed-node testable. Do not rebuild the deleted speculative hook apparatus —
that deletion was correct (plan §6).

**M11 — age-aware TT root trust.** Stays gated on the owner-ratified
condition: re-run the mine → classify → warm/cold → bisect pipeline on the
post-d63 net; only if the NEITHER class survives better labels does this arm
get designed, with pre-registered gates. Listed so it is not re-invented.

---

## Part III — IDENTITY AS STRATEGY

**The thesis, two sentences:** ARCHON's identity is *the engine that measures
its own search and learns from its own games* — a simplified, own-constant
alpha-beta substrate (fewer mechanisms than SF, each one earning its slot on
ARCHON's measurements) driving a data flywheel that mines its own frontier,
and re-investing what it learns at the ROOT, where learned signals are valid.
Everything in this program feeds one of those three lines; scattered one-offs
that feed none of them are ranked low regardless of expected Elo.

**Why these compound:** the simplification ladder (Part I) is not cosmetic —
it is what makes the learned layer tractable (a future singularity head
replaces ONE margin, not an 11-knob family; a smaller, better-conditioned
knob space is where a learned controller can act). Stage-B SPSA ends the
transcription lineage, converting "SF-parity baseline" into "ARCHON's own
tuned search" — the honest publication framing changes from *port* to
*platform*. The flywheel (M12, live) plus the root-level learned consumers
(M1, M10) are the unique research story: no reference engine has the obs
store, the counterfactual replay harness, the provenance ledger, or a
games-proven learned TM factor. The audit's remediation items (own bench
suite staged, own magic seeds harnessed, OBL1 landed) ride the same identity
line at zero strength cost — activate the bench suite at the next Bench-pin
move.

**Sequencing on one box, nothing preempting d63 (GPU+RAM):**

1. **Now, CPU, beside the trainer:** S1 (singular collapse — spec exists,
   branch, census, chain match), then S2 (SF cargo). Each ~1 session of work
   + ~3 h of match time. Meanwhile M2's census screening (minutes per
   predicate) and M8's ceiling census (one hour) — both need only the
   existing binaries.
2. **First quiet window / trainer pause:** M1's control match (the single
   highest information-per-hour item in the portfolio: it either mints the
   first learned-consumer win or closes the line cleanly).
3. **CPU, after S1/S2 land:** S3/S4, then the pooled cumulative A/B of the
   whole ladder vs pre-simplification main (10 × 1500 games pooled → ±3.9).
4. **Stage-B SPSA (S7)** once the surviving knob set is stable — 2–3 days
   interleaved.
5. **When d63 completes (GPU frees):** ThreatRemap refine panel (M6, armed);
   castling/rule50 twin arm (M3) as a gated arm of the next generation;
   heads (M10) as a training decision; M9 Phase-2 when the trace corpus is
   refreshed on the new net.
6. **Post-d63 re-mine:** the blunder pipeline on the new net decides M11 and
   refreshes M5's specimen set.

---

## Part IV — CONSIDERED AND RANKED LOW / DEAD (do not re-propose without a new mechanism)

Closed by measurement, with the number that closed it:

- **Interior-node learned consumers of the DIS head** — DIS-RFP (both
  decision directions, AUC gate −0.118 at 3k roots), DIS-EVAL (−97 to −22 in
  four framings), DIS-LMR continuous (curve wanders inside ±5), LMR-veto
  (tail AUC 0.5055 vs own-inputs 0.8772). The interior consumer space for
  this trunk is exhausted; only root-level consumers are valid.
- **Self-residual disagreement target** — structurally empty (r² 0.0017): a
  converged model's residual against its own features is unpredictable by
  construction.
- **Best-move-instability target** — ceiling 0.083 partial r², 7× weaker
  than a target that already produced zero Elo; pre-registered gate failed,
  no games spent.
- **TM budget multipliers below 1.41×** — absorbed by the one-ply node
  quantum; **TM back-loading** — +52.6 on nodestime, **−36.6 ± 22.5 at real
  TC** (and nodestime is permanently disqualified as a TM verdict
  instrument).
- **Eval-frame rescales, any direction** — settled interior optimum (6000
  games: SF parity −18, over-shrink −12); `EVAL_SCALE` −27.8. The frame is
  not a lever.
- **IIR on upper-bound entries / TT upper-blind** — −24.17 ± 15.11 @ 792.
- **TT fail-low move store** — −0.06 pp fhf, +2.3% nodes vs its own pruning
  control; SF and Reckless store nothing there, and now we know why.
- **Global pruning/reduction level retunes** — every knob at a local optimum
  vs inert ±1 at equal nodes; the 154-value refresh from newer SF:
  −1.39 ± 10.43 @ 1500 (no transcription-drift Elo exists).
- **Single-knob A/B at n≈400** — ±40–50 observed; structurally unable to see
  +1..+5 Elo effects. SPSA or n ≥ 1500, always.
- **Unclamped SEE ordering demotion** — wins every proxy, −5.35 ± 9.26 in
  games; any ordering change that can cross `GOOD_QUIET_THRESHOLD` is a
  pruning change.
- **Quiet-malus shape inversion / cross-kind malus / evasion full-history /
  LVA tiebreak / equal-attacker tier without SEE / full quiet sort** — all
  census-dead or magnitude artifacts (study 70).
- **Aspiration widening** — asp100 −49.6 (real turnover); asp30 UNRESOLVED
  (+10.3 ± 34), not dead — but not cheap either; only re-open with a powered
  SPRT and a reason.
- **Perf routes** — sub-byte FT formats, locality/row reorder, prefetch
  (fold is MSHR-limited); code layout at every granularity (machine function
  splitter: 30.7% span compaction, 0.000% instructions); branchless
  insertion-sort rewrites (four losses, break-even 22.4 instr/mispredict);
  LazyLock guard hoisting (the biggest one LOST as a Board field); the dead
  hot symbols (`i16_deltas_*`) do not exist — re-derive any profile before
  targeting it.
- **Killers / countermoves / Syzygy** — declined by design (D17/D15) and now
  part of the simplification identity: their ABSENCE is a feature. Do not
  add them "for completeness".
- **Node-count verdicts on position sets** — the wl50 total moves +18.3%
  under an inert ±1; it is a fingerprint, not a measurement. Census, sign
  test, games — in that order.
- **Carryover "fix" as a search patch** — TT carryover is +39.4 ± 12.3 net;
  the blunder tail it causes is an eval/label property (owner-ratified).
  M11 is the only sanctioned successor, and it is gated.

---

*Cross-references: `docs/research/search-simplification-plan-2026-08-01.md`
(§4 methodology, §5 keep list), `search-simplification-execution-2026-08-01.md`
(stage-3/basket-B patch specs), `searchopt/LEDGER.md` (every number quoted
here), `docs/audit/2026-08-originality/00-synthesis.md` (the banked five +
license posture), `design/19` (M9), ledger exp-0169/0170 (the instrument
bar).*
