---
name: implementation-runbook
description: "How to seamlessly resume the chessengine phase-implementation work after session turnover — mandate, live state, workflow mechanics, environment facts"
metadata: 
  node_type: memory
  type: project
  originSessionId: c0a1b520-8665-4b2f-85c5-35473d97415c
---

# Chessengine implementation runbook (update at every phase transition)

## Standing mandate (user, 2026-07-10)
Proceed through design/00-master-plan.md phases P0→P14 autonomously "until
the full design is complete, using best judgement". One background Workflow
per phase: sequential implementation agents (explicit awaits — see Lessons)
+ an exit auditor checking the owning docs' §8 acceptance criteria. Local
commits only — NEVER push (user pushes).

## ✅ P13 CHECKPOINT CLEARED — GO GRANTED (user, 2026-07-12)
The pause-before-P13 checkpoint is DONE: P9/P10/P11 complete + reported, Integral
study landed + baked in (D36/D37), P13 plan finalized and user gave the go. The
full finalized plan is in **design/11 §11** and **D35** (decisions.md) — those +
the Live status below are authoritative. START P13 step 1 (inference skeleton)
per that plan. (Historical: this was originally a hard "do not start P13 without
the user" gate; that gate is now satisfied.)

## Model (changed 2026-07-11)
Session model is now **Opus 4.8 (1M context)** — the user switched off Fable
after hitting its monthly spend limit mid-P7-audit. Subagents inherit the
session model, so they now run on Opus by default; the earlier
[[subagent-model-preference]] note (explicit `model:'opus'` for P8+) is now
satisfied automatically but still fine to set. Use Co-Authored-By trailer
"Claude Opus 4.8 (1M context)" on commits from here.


## ★ CURRENT STATE (2026-07-15 — ARCHON/D41 era; onboarded after codex/user handoff). Read THIS first; everything below is historical.

**THE AUTHORITATIVE LIVE STATE IS IN THE REPO, NOT HERE:**
`docs/research/policy-automl-state.md` (2,786 lines) — read it COMPLETELY,
especially the final "Takeover checkpoint" section, before any research loop;
update it after every completed/rejected/redirected trial. Also read (per its
continuation invariant): `design/19-learned-search-allocation.md` + decisions
D40–D44, `docs/research/archon-execution-plan.md`,
`docs/research/archon-self-improvement-roadmap.md`,
`docs/research/sf18-label-quality-plan.md`. This memory section only carries
orientation + non-repo facts.

**Orientation (2026-07-15):**
- Engine RENAMED **ARCHON**. `GEMINI.md` is the codex/gemini copy of CLAUDE.md
  (updated for the rename; CLAUDE.md itself is stale on this).
  `.agents/rules/` holds a mirror of these memory files for other agents.
- **D41 pivot governs:** policy imitation is no longer the goal; learn search-
  action *consequences* (doc 19). Protected/frozen evaluator invariant; shadow-
  only until gates pass; equal-time Elo is the ship gate. D42 SF18 = value
  oracle; D43 observational vs interventional data; D44 Reckless→ARCHON
  curriculum. Master plan phase P15 / milestone M11.
- **net-0 (SF16-teacher classic 768→2x256) is STILL the strongest evaluator.**
  Every bulk/scalar-label attempt to beat it failed (raw/calibrated SF18,
  Reckless, WDL targets, consensus, bulk matrix all cells, D40 variants).
  Root-cause hypothesis (takeover checkpoint): target/search-contract mismatch
  — bulk isolated-cp nets get good static fit but poor LOCAL COHERENCE
  (`tools/nnue/audit_local_coherence.py` is now a mandatory pre-game gate);
  naive 1.65x output scaling made it WORSE.
- **HIGH-SEVERITY LABEL CONTRACT:** hard-node-limit (5k) search labels are
  frequently aspiration-window BOUNDS, not scalars (~40-68% bounded). Never
  feed bounded values to a scalar loss. The ~66M-row paired Reckless/SF18 5k
  cloud+local corpora are QUARANTINED for pointwise value losses (fine for
  policy/difficulty/bound-aware). Production value oracle = tiered SF18
  completed-deep-iteration protocol (sf18-label-quality-plan.md; 256-pos
  frontier measured, false-early-stability audit pending).
- Real positives so far: gen-0 r1 rank-8 bilinear policy head + net-0 unchanged
  = **+68.6 ±23 Elo @2k fixed nodes** (200 games; fixed-node only, NOT
  equal-time — huge NPS tax); D41 causal result = context/action predicts
  reduction risk (AUC .809), frozen net-0 latent adds NOTHING (closed);
  protected zero-init expansions (512-wide, HalfKA residual) preserve net-0
  bit-exactly but no confirmed gain yet.
- **Next work order (takeover checkpoint + codex):** (1) independently audit
  Lichess/Fishpack label semantics (perspective, score kind, projection rule)
  vs fresh pinned SF18 — do NOT trust exporter self-parity; (2) improve the
  coherence metric (same-edge teacher comparison); (3) minimal causal target
  experiment on classic256/1024 (outcomes/WDL + local grouping vs isolated cp)
  BEFORE more D40 budget; (4) never naively combine Lichess+Fishpack; (5) D40
  resumes only staged/protected with coherence-aware checkpoint selection.
  Best bulk candidate: Lichess classic1024 (−100/−72/−39 @2k/5k/20k).

**Non-repo environment facts (verified 2026-07-15):**
- Worktree HEAVILY DIRTY on top of HEAD `da75ce8` (~139 paths, all D41-era
  work UNCOMMITTED). Takeover order: treat as authoritative, NEVER reset;
  do not retrain completed models (artifacts under `data/nnue/nets/`,
  campaigns, audits are the record).
- No training/match/relabel processes running (clean pause verified). Disk
  33GB free. GCP project `archon-502500`: zone-D label job SUCCEEDED 134/134;
  zone B/C jobs stopped+gone (bound quarantine), ~266 shard objects durable in
  `gs://archon-main-bucket/corpora/lc0-selected/v1/dual-labels-r5k-sf18-5k/`;
  fishpack32.binpack (5.58GB, SHA verified) durable in GCS AND local at
  `data/corpora/official-stockfish-master-binpacks/main/raw/`; all 17 Lichess
  parquets mirrored to GCS, local parquets deleted, canonical 8.5GB export
  stream kept local. Raw LC0 tars deleted locally (GCS-verified first).
- **D38 LEDGER GAP:** last ledger rows are 2026-07-13 (net-10/exp-0024) —
  NONE of the D41-era nets/corpora/experiments were ledgered. Backfill when
  convenient; the state doc is the de-facto log meanwhile.
- Old 13-net `d40-policy-4x3` campaign = superseded historical control on an
  invalid LC0 corpus; its gauntlet resume depends on a VANISHED temp Bullet
  checkout — fix only if the historical record is worth it. Pinned toolchains
  now live durably at `.archon-state/toolchains` (bullet `cebc78a`,
  bulletformat `cb6a360`).
- Session model is Fable 5 again (user was on Opus 4.8 for a while; codex/
  gemini also worked in the repo since).

---
## ★ (SUPERSEDED 2026-07-15) CURRENT STATE (2026-07-13 LATE — P13-SOTA sprint IN BUILD, ~60% done).

**SPRINT PROGRESS (2026-07-13, multi-agent build of the D40 12-net experiment):**
- **#29 inference (engine, behind `sota` feature, bench-neutral 548978):**
  ✅ HEAD ported to main (`src/eval/nnue/sota_head.rs`, commit 605d495 — pairwise-mul
     → int8-sparse-FT/f32-tail → 8 buckets; format-agnostic, takes pst+threat accs).
  ✅ FEATURE FRONT-END (`src/eval/nnue/sota_features.rs`, commit 8b81244 — HalfKAv2_hm
     + FullThreats index math, both OrientTBL polarities, from-scratch refresh; 12
     tests incl. hand-computed indices; VERIFIED by me). Includes `pawn_push_or_attacks`
     (SF pushes — the key cross-check with the incremental path).
  ⏳ INCREMENTAL THREATS OBSERVER — spec'd (design/11 §12.9.1) but NOT built (agent
     died on rate-limit mid-work, no partial files). Self-checking replay-test gate
     (incremental==refresh at every node). NEXT engine task. Simpler §3.1 first-cut OK.
  ⏳ CENN-v2 LOADER — byte layout spec'd (design/11 §12.6). Bullet export order KNOWN:
     `l0w/l0b i16(255), l1w i8(64) TRANSPOSED, l1b/l2w/l2b/l3w/l3b f32` (weights
     transposed on save — transcode must handle). d40_eval net = C1 HalfKA-only shape
     (one FT tensor, threat_dims=0).
  ⏳ PARITY GATE (Rust forward == bullet forward) — after loader.
- **#28 trainer (bullet, scratchpad — PROVEN, agent DONE before rate-limit):** the D40
  eval trunk is ALL-NATIVE bullet + TRAINED a real net (loss 0.0244→0.0112, 30 SB on
  49M mix corpus; ckpts at `<scratch>/d40_eval_real/d40-eval-0-{10,20,30}/quantised.bin`,
  11.9MB). Both custom pieces scaffolded + RAN: FullThreats two-summed-tensor
  SparseInputType (placeholder index math — port from sota_features.rs), and the
  low-level policy trainer (from×to 4096 head + combined eval+λ·policy-CE loss, ran on
  synthetic target). 3 examples in `<scratch>/bullet/examples/d40_{eval,features,policy}.rs`
  — EPHEMERAL scratchpad, PRESERVE to tools/nnue/ (decision pending). Two bullet-API
  fixes baked in: `l0.init_with_effective_input_size(32)` (king-bucket sparse init
  scale — else flat loss), and drop the threats-FT bias (CSE compile fail + matches §12.2).
- **#27 data (DONE):** `data/lc0_nnue/policy_eval_ds_100m/` = 100M dual-target (50M
  white+black, 22GB) eval+policy, meta.json complete, bijection move-fail 1.03%. Feeds
  C3/C4. Being ledgered. Builder `data/lc0_nnue/policy_eval_build_mm.py`. 86 tars kept.

**RESOURCES (2026-07-13):** a weekly limit was briefly hit mid-sprint (killed #28
post-report + #29-observer mid-work) but the user FIXED billing/budget — resources are
NOT constrained. FAN OUT with agents freely (default posture, D33). Subagents are
token-EFFICIENT (empty cache on start) GIVEN good self-contained instructions — prefer
2+ parallel agents with thorough prompts over serial main-loop work.

**TRAINING IS AUTOMATED — do NOT use a subagent to train a net (user, 2026-07-13).**
`tools/nnue/train_run.py` runs the whole RUNBOOK pipeline (preflight→corpus→train→
verify→pack→sanity→SPRT→ledger) from a JSON config: `uv run python
tools/nnue/train_run.py --config tools/nnue/configs/<net>.json [--dry-run|--commit|
--only|--skip|--set]`. Workflow = write/edit a config in `tools/nnue/configs/` (tracked
templates; example-classic.json + example-d40-c1.json), ensure engine code is in a good
state (preflight checks sha/dirty/net_id-free), then run the one command. Per-run
manifest → gitignored `tools/nnue/runs/<id>/run.json`; ledger row is the tracked
provenance. Agents are still fine for the CUSTOM parts (new bullet examples, index
math, engine code) — but the standard train-a-net loop is now one script, no agent.

**NEXT (critical path):** (1) build the incremental threats observer (§12.9.1, replay-gated);
(2) port the real FullThreats index (sota_features.rs) into bullet's SparseInputType +
train the first REAL 83k net; (3) CENN-v2 loader + transcode bullet quantised.bin; (4)
parity gate; (5) then the 12-net grid train + fixed-node SPRT vs net-8 (#30). Tasks #27✅ #28~ #29~ #30 pending.

---
## ★ (SUPERSEDED) CURRENT STATE (2026-07-13 — P13-SOTA sprint: D40 SOTA-net design FINALIZED + planned).

**NOW (2026-07-13).** Policy-in-ordering experiment CONCLUDED — **NEGATIVE** for additive
quiet-ordering (net-neutral tree on 400 realistic positions; helps endgames, hurts
open/middlegame where search time goes; code kept feature-gated, bench-neutral 548978;
see [[policy-integration-status]]). Then a full DESIGN REVIEW finalized the next big
thing: **D40 — a SOTA NNUE net** (design/11 §12, committed HEAD 95b0576):
- **Features:** SF's exact HalfKAv2_hm (22,528) + FullThreats (60,720) = 83,248, extracted
  from `stockfish_reference/` (index math, 32 king buckets, two OrientTBL polarities). This
  IS attack-features-for-eval (old task #26 SUBSUMED).
- **Head:** Reckless-shaped lean hybrid (pairwise-mul FT → int8-sparse-FT/f32-tail → 8
  buckets, no PSQT, no skip) — **already reimplemented bit-exactly on `experiment/reckless-net`**
  (our own D31-clean code → moves to main; only Reckless WEIGHTS stay firewalled).
- **Triple-net grid:** every config trains {eval-only, eval+small (rank-8 bilinear),
  eval+full (4096)} on one trunk arch, Lc0 soft-policy target; eval-only is the control
  that policy isn't degrading eval. Policy heads NOVEL (no SF/Reckless precedent, D34).
- **Trainer:** bullet-extended (FullThreats feature + configurable policy head + combined loss).

**THE SPRINT (user, 2026-07-13): a 12-net experiment = 4 configs × 3 policy variants.**
Configs: C1 HalfKA-only/SF, C2 +FullThreats/SF (threats Elo), C3 +threats/Lc0 (teacher +
rich policy), C4 +threats/SF+Lc0 mix. Head+768acc+8buckets constant. Eval/policy data
tension: SF=best eval/one-hot policy, Lc0=rich policy/weaker eval → each config's triple
shares one corpus. Tasks: **#27 data (format extension + curation)** → **#28 train** →
**#29 inference (port Reckless head + SF front-end + CENN-v2, ∥)** → **#30 SPRT ×12 vs
net-8 + real-TC gate + ledger**. Docs updated: design/11 §12, design/00 P13-SOTA, decisions
D40. **NEXT: start #27 (data) and #29 (port the Reckless head to main — data-independent,
can begin now).**

**⚡ OVERNIGHT AUTONOMOUS RUN (2026-07-13 ~02:15, user ASLEEP — do these unattended):**
Lc0 MAPPING FIX **VALIDATED +449 Elo** (net-4s 4M-atanh beats net-1s 4M-display, equal
data — see [[nnue-label-scale-lesson]], [[elo-decomposition-testing-protocol]]).
SEQUENCE: (1) **25M net-4 convert finishing** — driver at data/lc0_nnue/driver_atanh.py,
PREFETCH=6, KEEP_TARS=1 (tars kept in data/lc0_nnue/tars/), OUTDIR=raw-net4; **Monitor
b53hfpe7b** pings on `FINISHED`. **ON FINISH → build+train+SPRT net-4:** `BULLETFORMAT_CONVERT=$SCR/nnue_convert/target/release/nnue_convert\ convert BULLET_UTILS_SHUFFLE=$SCR/shuffle_wrap.sh BULLET_UTILS_ILEAVE=$SCR/ileave_wrap.sh bash tools/nnue/build_corpus.sh --raw data/lc0_nnue/raw-net4 --out data/nnue/net4 --seed 1 --require-convert` (multi-shard → interleave works, gives train.bin); train **FULL** (`NET0_SUPERBATCHES=30 NET0_BPS=1520`, NOT the smoke's 244) via `$SCR/bullet/target/release/examples/net0` with NET0_DATA=…/net4/train.bin NET0_OUT=…/ckpt-net4 NET0_ID=net-4 CUDA_PATH=/usr; pack `uv run --with numpy python tools/nnue/netpack.py <ckpt>/quantised.bin data/nnue/nets/net-4.cenn screlu`; nnue-eval sanity; **fixed-node SPRT** `bash $SCR/nnue_sprt.sh <net-4> <net-0> net-4 net-0 25000 500 12 net4_vs_net0` AND vs net-1; ledger add-net(+nps)/add-exp; commit LOCAL. (2) then **#19 bigger arch** + **#21 SF+Lc0 mix** (scale-compatible confirmed: queen≈900 both) on the 25M corpus. (3) BACKGROUND **`$SCR/hoard_tars.sh`** curls more test91 tars into tars/ 6-wide, STOPS at disk-free<25GB. **DISK REALITY: 500M kept pos = ~400 tars = 336GB, does NOT fit 96GB free → hoard caps ~60-80 tars (~100M pos); report shortfall; 500M needs decode-to-compact or more disk (do NOT decode without user OK).** Guardrails: FIXED-NODE screens only, ledger every net, commit local NEVER push.

**⚡ OVERNIGHT RESULTS (2026-07-13 ~03:00 — Lc0 campaign COMPLETE, honest verdict):**
net-4 (Lc0-atanh 25M): **+358.8 vs net-1(display), −131.7 vs net-0(SF)** → mapping FIXED,
Lc0 USABLE but a LOWER-QUALITY eval teacher than SF (best_q win-prob < SF search cp).
net-5 (SF+Lc0 50/50, 50M): −52.9 vs net-0 → mix HURTS even at 2× data. net-6 (SF+9% Lc0):
−78.1 vs net-0 → even a small dose hurts (data-aug hypothesis refuted). **VERDICT: SF >
mix > Lc0; do NOT dilute SF with Lc0.** All ledgered (nets net-4/5/6, exps 0017-0021, mix
corpora) + committed LOCAL (HEAD 313508b). HOARD DONE: **86 tars (~107M positions), 66GB** in data/lc0_nnue/tars/, stopped at
the 30GB disk floor (500M unreachable: ~400 tars/336GB > 96GB disk). Raw data ready
for future decode (env-param driver_atanh.py: TARGET/N_TARS/OUTDIR/KEEP_TARS/PREFETCH). DEFERRED (need
user/engine, NOT done overnight): **#19 bigger arch** (blocked: engine L1=256 compile
const — needs runtime-L1 feature); **#20 more Lc0 volume** (could close the −132 but won't
beat SF; needs a decode the user deferred); **deblunder/quiet-filter** (V6 invariance_info
offset undocumented — don't reverse-engineer a label filter unattended). NEXT SESSION:
await user direction on those levers; net-0 (SF) remains the best net.

**BIGGER-NET / MIX EXPERIMENTS DONE (2026-07-13, user asks):** nnue-l1-512 cargo feature
(committed b331602, bench-neutral 548978; mod.rs L1 cfg-gated; netpack NET_L1 env; bullet
net0.rs NET0_HIDDEN env). Binaries scratchpad/{eng-256,eng-512}; per-binary SPRT helper
scratchpad/nnue_sprt2.sh. Results (fixed-node 25k, ledger nets 8/9/10, exps 0022-0024):
2x width ~+16 net-quality on SF (net-8 512 SF, -8.3% nps -> ~+9 real-TC); **SF>mix>Lc0
holds at 512 too (net-8 512 SF is our BEST net)**; bigger net does NOT rescue Lc0 (net-9
512 Lc0 is -225 vs net-8, worse vs SF than 256 Lc0 -> overfits inferior teacher). LESSON:
**cross-binary (eng-512 vs eng-256) fixed-node SPRT is UNRELIABLE — always confirm with
same-binary** (the "net-10 mix +44 vs net-0" was a glitch; same-binary net-10<net-8 -57).

**★ POLICY-INFORMED NET — IN PROGRESS (user pivot 2026-07-13, "begin without me"):**
design/17 Thrust II (§3.4): policy head on shared trunk -> move ordering; metric = cutoffs
on 1st move / node-reduction @ equal depth. FOUNDATION DONE + VALIDATED: (1) Lc0 V6
policy[1858] @ offset 8 (sum=1 confirmed). (2) 1858->move map = `data/lc0_nnue/policy_map_1858.json`
(~99.8%/move, empirical: legal-move intersection + bijection-elimination; scripts
scratchpad/policy_map_finalize.py). (3) dataset `data/lc0_nnue/policy_ds_min.npz` (600k
white-to-move, STRICT-bijection self-filtered = clean targets; feats[N,32]+pfrom[64]+pto[64]
+bestq+result). NOW: PyTorch prototypes (bullet is value-only) — first arch training bg
**bg9j5mo6h** (trunk-512 from+to+value heads, scratchpad/policy_train.py, metric from/to
top-1). Plan (task #23): 4-5 candidate archs, offline top-1/top-k move accuracy, THEN wire
best into MovePicker.
**FIRST RESULTS (2026-07-13):** policy head WORKS — full-move (from×to 4096, legal-masked)
head gets **move top-1 ~26% (3.2× random 8%), top-3 ~50%**. Arch sweep (trunk 256/512/1024,
deep, policy-only) ALL ~26% → NOT capacity-limited. Train-vs-val (scratchpad/policy_trainval.py):
**TRAIN 67% / VAL 25% at ep40 = severe OVERFITTING → DATA-limited** (600k too few). LEVER =
MORE DATA (340M available). Building 3M dataset (scratchpad/policy_ds_big.py -> policy_ds_3m.npz) to confirm val lifts.
**DATA LEVER CONFIRMED (3M, scratchpad/policy_train3m.py):** VAL top1 25%(600k)->29%(3M);
overfit gap 42pt->15pt. More data helps + halves overfitting. Diminishing returns => next =
BOTH scale to ~25M (streaming, see RAM note) AND richer inputs (threat/attack maps, design/17
§3.5 + Reckless threat features) to push val toward ~40%, then wire prior into MovePicker.
Best policy net so far: trunk-1024 from×to head, VAL move-top1 ~29% / top3 ~55%.
**ATTACK/THREAT FEATURES AS *POLICY* INPUTS — NO MOVE-PREDICTION LIFT (2026-07-13, user
ask, 4 exps + deep confirm). SCOPE IS NARROW — this is NOT a verdict on attack features for
the NNUE *EVAL/value* net, where they are a STAPLE of the best NNUE engines right now and stay
firmly on our roadmap (user, 2026-07-13: "don't want to abandon them, come back later"). The
value net is a different question than move-prediction top1 and was not tested here.** built
bitboard-derivable ablatable feature space (scratchpad/policy_ds_feat.py: [0,768) piece-sq,
[768,1024) attack maps, [1024,2560) hanging/capturable threat features; SF/Reckless-informed,
engine-ready via attacks(pt,sq,occ)). Ablation (scratchpad/policy_ablation.py + policy_deep.py):
E1 base 29.2, E2 +attackmaps 29.2, E3 +threats 29.4, E4 all 29.6, trunk1024 30.1, DEEP+bilinear
base 29.1 / all 29.7 — **ALL ~29-30%, features add ~0 even with the pairwise-mul architecture
designed to exploit them.** REVISED LEVERS for policy: NOT features (net learns attacks from
placement); it's (a) DATA (25->29% from 600k->3M, plateauing ~30%) and (b) ARCHITECTURE — a
flat MLP caps ~30% top1 vs Lc0's own conv policy ~55%; a spatially-aware (conv/attention) net
is the real lever BUT too slow for in-tree per-node use. TENSION: cheap flat policy ~30% top1 /
53% top3 (3.2x random) is what the engine can afford. NEXT DECISION: test whether that cheap
prior actually helps MovePicker ordering (design/17 §3.4.4 metric = cutoffs-on-1st-move) —
a 30%/53% prior blended with history may still help even if weak standalone; that's the real
go/no-go, not more feature/arch experiments.
**ORDERING-BENEFIT TEST — POSITIVE (2026-07-13, scratchpad/policy_ordering_test.py, easy
offline proxy = rank of Lc0-best-move under each ordering, K=3000 held-out):** random first
9.3%/rank12.0; MVV-LVA+PST(static baseline) first 19.9%/top3 38%/rank9.1; **MVV-LVA+POLICY
quiets first 25.4%/top3 55.7%/rank4.6** (+28% rel first-move, avg-rank HALVED); pure-policy
first 39.3%/rank4.1. => the cheap flat policy DOES improve quiet-move ordering meaningfully.
CAVEATS: "best"=Lc0 argmax not our engine's best; baseline is static (no HISTORY, which the
real engine has — policy's marginal value over history+MVV-LVA needs the engine SPRT; but
design/17 §3.4.1 says policy helps most where history is COLD = early in lines). VERDICT:
positive enough to justify the ENGINE integration (policy inference in Rust + MovePicker
blend + fixed-depth node-count / SPRT) = the definitive test, task #24. Saved net
scratchpad/policy_net.npz (768->512->4096, numpy weights). Parallel: agent ab1f19b0 scaling
policy data to 15M + black-flip.
**ENGINE-FEASIBILITY RESOLVED (2026-07-13) — the KEY design decision:** the full 4096 from×to
head that gave the strong offline signal is TOO EXPENSIVE per-node (~2M MAC/node = ~50x CPU
budget). Tested cheaper heads' ordering benefit (avg-rank of Lc0-best under MVV-LVA+policy-quiets,
baseline avg-rank 9.09): additive from+to = 7.12 (only 44% of benefit — factorization kills the
from-to CORRELATION); **BILINEAR low-rank from_emb·to_emb: k8=5.38, k16=5.46, k32=5.65 (~83% of
the full head's benefit, plateaus at k=8), cost ~256k MAC/node (8x cheaper than 4096, ~2.5x eval
head)**. So **USE A RANK-8 BILINEAR from×to policy head on the SHARED eval trunk** — engine-
affordable AND keeps most of the ordering signal. Scripts scratchpad/policy_{factorized,bilinear}.py.
NEXT (task #24, well-justified now): Rust policy inference = read the NNUE accumulator (shared
trunk, free) -> from_emb[64,8]+to_emb[64,8] heads -> per-quiet-move score = dot; blend into
MovePicker quiet ordering (design/17 §3.4.1); test via fixed-depth node count + SPRT. This is a
real src change (new head on the eval net + MovePicker) — the definitive Elo test.
**⚠ BOX HAS ONLY 7GB RAM** — accumulating 5M positions as Python lists OOM-killed the build
(dmesg confirmed). Use PRE-ALLOCATED numpy arrays (fill by index) or stream to disk; keep the
in-RAM dataset + the torch load under ~4GB. For BIG policy corpora (25M+) stream/chunk to a
memmap. (GPU=8GB RTX 4060 Ti separate; the 7GB limit is system RAM.) Harness scratchpad/policy_sweep.py. Scripts + npz reusable. If 5M lifts
val: scale to 25M+, add richer inputs (threat maps, design/17 §3.5) + full-1858/black-flip,
then wire prior into MovePicker (blend w/ history) per §3.4.1, metric cutoffs-on-1st-move. Upgrade path: full per-move policy (not marginals), black-to-move flip,
perfect the map, and SF18+/Reckless "Root Move Analysis" datagen (more faithful than Lc0 MCTS
per §3.4.1). torch added via uv (--with torch). 100M Lc0 value corpus (raw-lc0-100m) ready but
net-7 DEFERRED (low value post-pivot, disk 26GB tight).


**main HEAD ~066f427 + later ledger commits (bb53bc1 embed-net, c1388df/066f427 net-2/3).
Default bench 548978.** Reckless work is on branch **`experiment/reckless-net`** (NEVER
merge to main, never push — firewalled AGPL).

**P13 DONE — NNUE works.** net-0 (`768x2x256x1` SCReLU, SF-teacher) beats PSQT
**+696 Elo** (exp-0004); draws full SF16 4/12; NNUE+SMP scale (exp-0010/11/12).
Wired opt-in via `EvalFile` (default PSQT → bench-neutral). **A single self-contained
Windows binary (net-0 embedded via `--features embed-net`, commit bb53bc1) is
delivered to the user at `C:\Users\Nick\Downloads\chessengine-net0\chessengine.exe`.**
**Our SEARCH is validated STRONG** — our-search+Reckless-net beats our-search+net-0
**+759** at equal nodes → *the NET is the bottleneck, not the search*; the path to
strength is bigger/better nets.

**Experiment ledger (D38, `ledger/` on main):** nets net-0..net-3; exps exp-0000..0016.
Corpora sf-100 (24.95M SF, THE baseline), lc0-test91 (25M Lc0), lc0-test91-atanh400
(confounded), lc0-test91-g065 (net-2). SF-teacher data is the winner.

**⚠ RECKLESS-NET PORT — DONE + FIREWALLED (branch `experiment/reckless-net`, 4
commits, never merged/pushed).** Feature `reckless-net` OFF by default → feature-off
bench 548978 bit-identical; AGPL net gitignored (`RECKLESS_NET` path). Faithful
full-refresh reimpl of Reckless's threat-map+king-buckets+i8-sparse-L1+f32-tail:
**golden-file gate = 0 cp over 42 positions (bit-exact)**. Branch-local exp-0000 =
our-search+Reckless-net +759 vs net-0. Spec: `reckless_reference/15-*.md`.

**⚠ CORRECTED 2026-07-13 (user challenged the battery's speed read — was RIGHT):**
OUR ENGINE IS NOT SLOWER THAN RECKLESS — it is **~2.5× FASTER in raw nps**. Fixed-node
(2M) self-reported nps, contended box: **net0 (our engine) 2.36 Mnps**, psqt (NNUE off)
2.67 M, **full Reckless 0.95 M**, our-port-of-Reckless-net 0.34 M. Perft = 301 Mnps
(clean release, debug-asserts off) → movegen is NOT a bottleneck. At EQUAL nodes we
reach ~equal depth to Reckless (depth-probe: 15/17 vs 16/19 @400k) → search efficiency
competitive. The battery's T3 "us=271k nps, 4× slower" was a MEASUREMENT ARTIFACT: it
timed the **firewalled PORT** (0.34 M — our search dragging Reckless's 63 MB net through
our un-SIMD-tuned port inference), mislabeled "us." Reckless's 0.95 M is LOWER than our
2.36 M because their big net is expensive per eval — they spend nps to buy eval quality.
STRATEGIC READ: we have huge nps HEADROOM; the ONLY gap is NET QUALITY (→ Lc0 campaign).
Lesson: never quote a fixed-TIME nps across engines without checking (a) which binary,
(b) node-def, (c) build flags; confirm with fixed-NODE self-reported nps + perft.

**Reckless signal battery — DONE + CUT SHORT (user, 2026-07-13: "enough good info").**
Agent a0469…, clean box, STOPPED via TaskStop + ops-safe pkill -x. Final axis reads:
**search** (our-search vs Reckless-search, SAME net, equal nodes) −86.68@100k →
−68.43@500k (NARROWS with depth ⇒ pruning SOUND, just under-tuned — D39 rule g);
**net quality** (Reckless-net vs net-0, our search) +338@25k → +285@100k (flatter than
exp-0000's +759 because that was vs PSQT-floor, net-0 is a strong trained baseline);
**speed** we're ~2.5× faster (D39). The two axes scale OPPOSITELY with depth: net edge
GROWS, search deficit SHRINKS → net-first is right. All enshrined in D39/§3.11.

**IN PROGRESS (USER PRIORITY): HAMMER ON LC0 — autonomous campaign until user returns.**
ROOT BUG (user-corrected): converter used Lc0 cp-DISPLAY `90·tan(1.5637·q)` → bullet
`sigmoid(cp/400)` = double-sigmoid → compressed labels (median|cp|=16). **Correct =
`cp = 400·atanh(best_q)` from RAW best_q, FULL range, KEEP winning positions.** See
[[nnue-label-scale-lesson]], [[elo-decomposition-testing-protocol]].
**THE FIX IS ALREADY WRITTEN + CORRECT: `data/lc0_nnue/driver_atanh.py`** (raw best_q,
full-range 400·atanh(clamp ±0.9999), NO high-eval drop — fixes net-3's confound which
inverted already-clamped cp @cap 935). Now env-parametrized (TARGET/N_TARS/OUTDIR/
METAFILE/LOGFILE). Needs re-download of test91 tars (raw/ text corpus can't be re-mapped
— it already dropped 6.6M winning positions). **Currently: 3-tar SMOKE conversion running
(bg bb53y2tju → OUTDIR=raw-net4-smoke).** Next: verify label dist (want median|cp|~45+,
mean_cp_won~250+, max~1981 — NOT 16), train smoke net, fixed-node SPRT vs net-0 + net-1.
If good → full 25M (OUTDIR=raw-net4, corpus id `lc0-test91-atanhraw`). PIPELINE = the
RUNBOOK (`tools/nnue/RUNBOOK.md`): build_corpus.sh → train_net0.rs in bullet (GPU RTX
4060 Ti, ~1 min/30sb, ±1.98 out-clip MANDATORY) → netpack.py → nnue-eval sanity →
FIXED-NODE fastchess SPRT w/ per-engine option.EvalFile → ledger add-net(+nps)/add-exp.
TASKS: #16 net-4 (in progress), #18 net-4 full, #19 bigger arch, #20 more data, #21
Lc0+SF safe mix. GUARDRAILS: D39 fixed-node only (no long tests), smoke-first, ledger
every net, commit LOCAL never push. If net-4 weak: layer SF pipeline (deblunder V6
invariance bits 3/5/6, Syzygy rescore, lambda 1→0.5, early-ply deweight, more data).

**⚠ OPS DISCIPLINE (I tripped this HARD this session — do NOT repeat):** (a) bare
agents AUTO-RELAUNCH their detached download/train jobs — to stop one, **TaskStop the
AGENT first**, THEN kill the leftover job by **exact PID/PGID** (`ps -o pgid= -p <pid>`).
(b) NEVER kill by a `grep`/`pkill -f` pattern that appears in your OWN command line —
I killed my own shell doing that. (c) Prefer TaskStop over process kills for agent work.
See [[smoke-test-before-long-jobs]].

**Other pending:** task #8 = P13 step-4 accumulator/SMP review (SMP correctness IS
verified — Threads=4 debug-oracle bench, no panic; but the all-make-move-sites +
code review still open). Datagen long-stopped; frozen SF corpus 24.95M @ data/nnue/raw/.
uv is the tooling-Python manager. Dashboard: `uv run tensorboard --logdir tools/nnue/runs`.

---
## Live status (HISTORICAL step-by-step — superseded by CURRENT STATE above)

**HEAD 4efbbaa (P13 step 1 complete). Bench (depth-6, canonical) = 548978.**
Engine ~**2720 Elo single-core PSQT** (anchored vs SF16 UCI_Elo rungs). **Datagen
STOPPED (user, 2026-07-12) — enough to experiment.** Frozen SF-teacher corpus:
**24.95M positions**, 6 shards, 1.6G at data/nnue/raw/*.txt = net-0 training input.
More data later via Lc0 or, post-bootstrap, our own self-play (D28b). Heartbeat
monitor stopped (was datagen-watch).

**Lc0 data — EXPLORED, verdict GO (2026-07-12).** Full write-up + decoder in
gitignored `data/lc0_explore/` (FINDINGS.txt, parse_lc0.py). Newest run with data
= `test91/` (~24 tars × 870MB; test90 is empty placeholders). Format: **V6, 8356-B
struct**, `input_format=1`; has policy[1858] + planes[104] + STM-relative floats
(best_q/result_q). Reconstruction: **file-mirror sq^7 always + vertical ^56 when
black-to-move**; value NNUE uses only frame-0 planes 0-11. **Q→cp: `cp = 90·tan(
1.5637541897·q)`**. Convert via OUR text format (`FEN | cp_white | result_white`),
emit castling `-` universally (also handles the 960 games test91 interleaves),
reuse bf_convert. ~4.17M positions/tar; ~6 tars (~5GB) ≈ our 23M SF corpus.
Converter effort ≈ 0.5-1 day when we run the A/B (do it AFTER net-0 works on SF
data). Policy targets retainable for a future D34 policy head (needs Lc0's 1858
move-index map; out of scope for the value experiment). Provenance: external
teacher like SF (not self-play-clean) → experimental net only; a LICENSE.txt in
the storage root to check before any redistribution (we only consume).

**DONE + GATED:** P0-P8; **P9** all pruning waves A-G (C-G batch D32: D33 review
5 rounds + tripwire proxy SPRT +208 Elo H1, commit 80dc33d; closeout c1b87ba);
**P10** TM completion + full UCI + MultiPV + nodestime + Reckless robustness
bundle (D33-reviewed; exit gate 0 forfeits/1000 @ 10+0.1 + TM +21 Elo; commits
..41cc588, strength-log 4cdc81c); **P11** lazy SMP (5 impl commits ..843d43c +
D33 review 3 rounds + confirm ..07f184a; Threads=1 bench bit-identical 548978,
TSan clean, stop-latency 0.53ms, 6.94x nps @ 8t; scaling SPRT Threads-4-vs-1
+279 Elo 0 losses, commit bbbb423). Anchored Elo gauntlet logged (b6b5901).
Reference studies: stockfish_reference/, reckless_reference/ (D30/D31),
integral_reference/ (D36/D37 — headline: distributed test compute is the top
solo-dev lever; validated our plan).

**P13 (NNUE) IN PROGRESS — GO GRANTED (user, 2026-07-12).** Approach D35 + full
step plan in **design/11 §11** (READ IT FIRST). Locked choices: (1) first net
SMALL `Chess768 -> 2x256 -> 1` (not final); (2) train net-0 on WHATEVER the corpus
has now; (3) execution = consolidated impl agent -> D33 review loop -> MY
bench-neutrality + oracle gates -> local SPRT vs PSQT baseline (04ffd19);
(4) Reckless-net A/B DEFERRED until after our own net works + shows big Elo, then
firewalled loader (AGPL, gitignored, never shipped) as stress-test/baseline.
uv is the tooling-Python manager now (see Environment facts).

**Module lives at `src/eval/nnue/`** (the owning doc §3.4.4 path, NOT `src/nnue/`).

- **STEP 1 DONE + GATED (4efbbaa, was 66d6b7e + fixes 4efbbaa).** Inference
  skeleton: CENN loader (§3.4.5) + `Network` (feature-major) + perspective
  accumulator with the §3.4.4 invalidation rule + complete scalar `Kernels` seam
  (add_sub / add_sub_sub / add2_sub2 / refresh / forward) + `DirtyPiece` +
  `on_dirty` no-op hook (NOT yet called from search). Zero unsafe, zero deps,
  PURELY ADDITIVE (evaluate() unchanged). Gates ALL green: bench 548978
  bit-identical; incremental==refresh fuzz (both activations); color-flip
  symmetry; loader rejects malformed CENN; **Python forward oracle 113/113 both
  activations, 0 mismatches** (tools/nnue/{nnue_ref.py,verify_forward.sh}, run
  under uv). D33 review: 3 personas, ZERO blocker/major; reviewer B verified the
  feature-major l0w layout + stm-half-first out_weights + Chess768 index against
  bullet's OWN SOURCE (the two silent-Elo traps are correct). Dev subcommand
  `chessengine nnue-eval <net.cenn> <fen>`.
- **STEP 2 DONE + GATED (8efc1b2 + fixes b9cce91).** AVX2 backend behind the
  `KERNELS` LazyLock dispatch; sanctioned SIMD unsafe (D14b) contained in
  `simd::avx2` (safe fn-pointers out, `pub(super)` surface). 4.33x faster forward.
  Bit-exact to scalar (constructed test forces total 4.23e9 > i32::MAX to exercise
  the i64 reduction). **Loader now ENFORCES `|out_weight| <= 127` clip** (new
  `NnueError::OutWeightOutOfClip`) — the AVX2 mullo trick's precondition, so
  **step-3 bullet training MUST apply the L1 weight clip (|w1_float| <= 1.98)** or
  the net is rejected at load (design/11 §3.4.3/§3.4.7). Gates green: bench 548978,
  oracle 113/113, clippy clean, portable+aarch64 compile. 2 focused reviewers
  (soundness + bit-exactness): no live UB, proof holds.
- **STEP 3 DONE (fedd159).** net-0 TRAINED: `768x2x256x1` SCReLU, QA255/QB64/
  scale400, wdl0.3, 30 superbatches on 24.95M SF corpus (1m08s, RTX 4060 Ti,
  final train loss 0.0087). **max|out_weight|=127 (clip PASS)** — bullet's ±1.98
  AdamW clip did the job, loader accepts net-0 first try. Net at
  **data/nnue/nets/net-0.cenn** (394,804 B, gitignored; raw/quantised at
  data/nnue/nets/ckpt/net-0-30/). **vs-bullet cross-check CLOSED (deferred from
  step 1): 113/113 match, max diff 0** — CENN packing (incl. l1b i16->out_bias i32
  sign-extend) is bit-faithful to bullet. New tooling committed:
  tools/nnue/{netpack.py, train_net0.rs, dashboard.py, vs_bullet.py,
  net_registry.csv}. **Dashboard LIVE: `uv run tensorboard --logdir tools/nnue/runs`
  -> localhost:6006** (train loss/LR/throughput; bullet emits no val loss). corpus
  bin at data/nnue/{train,val}.bin. ⚠ **NET-0 EVAL RUNS ~2-3x HOT** vs PSQT cp scale
  (startpos +125, knight-up +981 where ~300 normal) — inference is bit-faithful so
  it's the trained net's calibration, NOT a bug; matters for step-4 because search
  pruning margins are cp-tuned for PSQT. If step-4 SPRT underwhelms, recalibrate
  SCALE (cheap re-pack, ~400->~160) and re-SPRT.
- **STEP 4 DONE — MILESTONE CLEARED (f91fce4 wiring + fc6a40a strength-log).**
  NNUE wired as active eval, OPT-IN via UCI `EvalFile` (PSQT default). Net loaded
  process-wide (RwLock<Option<&'static Network>>, Box::leak); NnueContext per-thread
  in EvalContext; do_move-wrapper derives DirtyPiece::from_move + pushes (covers
  main/ProbCut/qsearch); refresh_root at each search root; evaluate() gained `ply`
  arg, routes NNUE through the SAME §3.6 outer-shape. Zero overhead when off ->
  **default bench 548978 bit-identical** (verified); NNUE-on bench 336890. Debug
  oracle assert_matches_refresh fires at every NNUE eval node; search-consistency
  test green (incremental==refresh across LMR/PVS/null/qsearch).
  **RESULT: net-0 (NNUE) BEATS PSQT +696.15 ± 139.32 Elo, 98.25% (277-8-1/286),
  LOS 100%** (exp-0004). Hot scale did NOT blunt it. vs SF16 rungs 100/97.5/93.8%
  (PSQT was 85.5/76/38.5). Approx anchored ~3400 (matches user's prediction).
  Net binary + match logs gitignored.
- **P13 CORE COMPLETE.** The classical->NNUE transition works end-to-end
  (datagen->train->pack->infer(AVX2)->wire->beats-PSQT). NNUE machinery in
  src/eval/nnue/ (loader/accumulator/simd scalar+AVX2/DirtyPiece), net-0 in ledger.
- **NEXT:**
  1. **Step-4 review (task #8):** accumulator covers ALL make-move sites (incl.
     singular/excluded-move/IIR)? SMP thread-safety under Threads>1 (match was 1t;
     per-thread NnueContext + shared immutable net — verify no shared mutable NNUE
     state, RwLock read once not per-node; run a Threads=4 NNUE game + debug-oracle
     SMP run). Do THIS before building further.
  2. **Step 5 (iterate):** bigger arch (L1 512/1024, king/output buckets) + more/
     fresh data + the SF-vs-Lc0 A/B (data ids sf-100 vs lc0-test91, ledger) + D36
     SPSA scaffold; each net SPRT-gated vs net-0 (parent), recorded in ledger.
     Then self-play retrain for clean provenance (D28b) -> D34 research heads.
  3. **Reckless-net A/B** (D35) now UNLOCKED (first NNUE works + big Elo shown):
     firewalled loader, AGPL net gitignored, never shipped, stress-test only.
- (was) STEP 2 = AVX2 SIMD forward pass. Fill the AVX2 backend behind
  the existing `KERNELS` LazyLock dispatch (src/eval/nnue/simd.rs). Sanctioned
  SIMD unsafe (D14b): `#[target_feature(enable="avx2")] unsafe fn` + one
  `// SAFETY: detected at startup` at dispatch; localized `#[allow(unsafe_code)]`
  (crate is `#![deny(unsafe_code)]`; see tt.rs prefetch D14a for the pattern).
  §3.4.6 caveats NOW IN THE DOC: unaligned loads (`_mm256_loadu`); i16 `mullo`
  bit-exact vs i64 scalar only while `|clamp(a)·w1|` fits i16 (bound test-net
  weights); sum-then-single-`/QA` is the step-3 vs-bullet checkpoint. GATE:
  AVX2 == scalar bit-identical over the oracle set + fuzz; bench still 548978
  (still not wired); clippy clean; portable + aarch64 builds still compile
  (cfg-gated fallback). Training half ALREADY validated (commit 241b937,
  tools/nnue/TRAINING.md, bullet cebc78a, RTX 4060 Ti); converter bf_convert.rs.
- **STEPS 3-5:** train net-0 (bullet) + CENN pack + TensorBoard dashboard
  (design/11 §11.1) -> wire as eval + `on_dirty` call site in search do_move
  (src/search/mod.rs:3554) + NnueContext in EvalContext + local SPRT vs PSQT
  (="it works", expect +400-700 Elo) -> iterate (bigger arch + D36 SPSA scaffold
  + self-play retrain D28b -> D34 research heads). Reckless A/B after step 4.

**D36 prep (near-zero cost, do alongside P13):** SPSA `TUNABLE` scaffold over
search/params.rs (cargo feature="spsa"); keep bench stdout OpenBench-parseable.
Distributed testing = user rents 5-10 GCE spot OpenBench workers at the TUNING
GRIND (post-first-NNUE), not before.

**Recovery reflex:** ground truth = `git log --oneline` (commits named "P<n>"),
docs/strength-log.md, design/11 §11 (P13 plan), decisions.md D35/D36/D37.

## Recovering a workflow after session death
1. Find state: journal at `<session-dir>/subagents/workflows/<runId>/journal.jsonl`
   — count `"type":"result"` vs `"type":"started"`; agent-*.jsonl mtimes show
   liveness (stale > ~15 min with no cargo/rustc/fastchess processes = dead).
2. If the harness still lists the task as running but transcripts are stale:
   `TaskStop({task_id})`, then `Workflow({scriptPath, resumeFromRunId})` —
   completed agents return cached; prompt changes invalidate cache.
3. Script files persist across sessions at their absolute paths — old
   session-dir paths remain readable.

## Workflow-authoring lessons (learned the hard way)
- NEVER `pipeline([null], ...)` for a sequential chain — null items are
  silently skipped (this bug silently skipped ALL of P6's TT steps once).
  Write explicit `const a = await agent(...); const b = await agent(...)`.
- Verify early in each run that the started agents match the intended
  phase structure (check journal + newest transcript's first message).
- Every implementation agent prompt carries: contract-doc reading list,
  the design/14 §3.9 finishing protocol (fmt/clippy/test/check.sh/commit
  with Bench: + Co-Authored-By trailers, no push), STRICT FILE OWNERSHIP
  when parallel, and "never copy Stockfish code (D27)".
- Auditors get: exit-criteria list, instruction to FIX small gaps + commit
  "P<n> audit fixes", report structural ones; from P7 on they also run a
  strength match vs the previous phase binary (build baseline from git
  worktree) — regression = fail.

## Environment facts (not in the repo)
- fastchess + Stockfish 16 installed and working (used by tools/gauntlet.sh,
  sprt.sh; see tools/ scripts + docs/strength-log.md for invocations).
- python-chess 1.11.2 installed system-wide (perft oracle, selfplay driver).
- **Experimental ledger (D38, design/18): `ledger/` + `tools/ledger/ledger.py`**
  (stdlib CLI). Content-addressed provenance DAG + results ledger. After ANY net
  train / corpus build / match / SPRT / SPSA run, add a row
  (`python3 tools/ledger/ledger.py add-net|add-data|add-exp …`); identity =
  sha256; `verify` checks hash drift + integrity; `show dag` / `reproduce <id>`.
  Backfilled: net-0, sf-100 (24.95M SF), lc0-test91 (25M Lc0 A/B corpus),
  P9/P10/P11 + anchored-2720 experiments. Bytes gitignored; GCS mirror + OpenBench
  = later (D36). PENDING: add net-0-vs-PSQT exp row after step-4 SPRT.
- **uv (2026-07-12): repo is now a `uv` project** (pyproject.toml/uv.lock/
  .python-version=3.13; tensorboard+tensorflow pinned). Run ALL tooling Python
  via `uv run …`; add deps with `uv add` (never ad-hoc pip). Engine crate stays
  zero-dep (D15) — uv is tooling-layer only. Already-running datagen was launched
  with system python3; leave it, use uv for anything new.
- miri installed on nightly toolchain (`cargo +nightly miri test --lib tt::`).
- aarch64 check target installed (cross-check for cfg'd unsafe).
- WSL2 host (Ryzen 7 7700X): timing noise few %, huge pages deferred,
  vm.overcommit caveat on the absurd-resize test (kept #[ignore]d).
- Session effort: ultracode (re-enable via /effort if a new session lost it).
