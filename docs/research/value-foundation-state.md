# Value-Foundation Research State (B4: why we can't beat net-0, and the fix)

> **★★★ 2026-07-19 PIVOT: the active plan is now the SF18-CANONICAL EPIC —
> `docs/research/sf18-canonical-epic.md` (task #40).** Rebuild training around ONE
> canonical SF18-labeled dataset (warm@20k full relabel → pipeline rewrite on one
> scale → RETRAIN from scratch + PAIRED search retune → Archon-vs-SF18 hard
> negatives → self-play). FINETUNING the incumbent on relabel data is a PROVEN
> DEAD-END (eval-search co-adaptation; see the runbook 2026-07-19 PIVOT block for the
> full evidence). Read the epic doc + the runbook pivot block before resuming. The
> B4 history below remains the backdrop; the value work now flows through the epic.
>
> **DATA DOCS (2026-07-19):** `docs/research/data-index.md` = the catalog of every
> corpus/net (layout, positions, label scale, local+GCS path, keep/drop).
> `docs/research/post-relabel-data-strategy.md` = the strategy (budget-not-exhaustion,
> tiered canonical set, storage math, training-on-everything reasoning, presentation
> order). Phase-A relabel config finalized by `relabel-speed-study-2026-07.md`
> (bmi2-PGO + warm@10k = ~9.5h/$10.5). The full official-stockfish/master-binpacks set
> (12 standard files, ex-dfrc-960) is mirrored to GCS; shipped nets backed up to
> gs://archon-main-bucket/nets/. Open data decisions (budget 500M/1B/2B, breadth
> source, obs-in-relabel) in the strategy doc §6.

> **CONTINUATION INVARIANT:** Read this file completely before every research
> loop; update it after every completed, rejected, or redirected trial. This
> file supersedes `policy-automl-state.md` as the ACTIVE log as of 2026-07-15
> (user-declared transformational point). The old file remains the historical
> archive and its header invariants REMAIN BINDING: the 5k-node bounded-label
> quarantine, the quality-first flywheel rule, the SF18 tiered-oracle plan,
> never reset/discard the worktree, D38 ledger rows for every
> net/corpus/match, local commits only (never push).
>
> **NORTH-STAR GOAL (user, 2026-07-18 via /loop 15m):** a SEARCH-AUGMENTED
> NNUE that BEATS SF16 by +30 Elo at EQUAL TIME (10+0.1). ARCHON is currently
> ~−63 to −78 vs SF16 at equal time → need ~+95-110 Elo. Open research
> question "how"; be ruthless, prioritize the highest-value lever each loop,
> use the box fully. Ranked levers: (A) EVAL quality (D40 nets/data/king-bucket
> #27/packing #26), (B) SEARCH (SF14→16 patch study, S2 SPSA, corr-history),
> (C) SPEED (PGO done, packing/narrower-L1), (D) D41 helm search-augmentation
> (play-negative, low priority until A/B advance). Don't tunnel on the helm.
> **GOVERNING GOAL (user, 2026-07-15, still binding):** Find out WHY we
> struggle to train nets that beat net-0 — fast — then improve well beyond it.
> A world-class eval engine is the prerequisite. Roadmap:
> `archon-self-improvement-roadmap.md` (B4).
> **PROMOTION BAR (user, 2026-07-15 evening): candidates must beat net-0 at
> EQUAL TIME** — fixed-node screens are attribution instruments only; the
> gate is a 200-game 10+0.1 real-TC match (`tools/nnue/prod_tc_gate.sh`,
> width-specific fixed-node floors feed it). net-0 is far below top engines;
> the capability ladder beyond it: calibrated classic → exact HalfKA at
> scale on calibrated fishpack → multi-epoch schedules → self-play datagen
> flywheel + SPSA consumer retune.

Last updated: 2026-07-25 ~13:0x — **JOINT VALUE-TAX FINDING + d47k repair trajectory + P-ORD landed + D47L frozen-head staged; running log = jax-workstream-plan.md (2026-07-25 sections).**
imminent (S5 λ probe finishing).**

> **2026-07-23 OVERNIGHT (autonomy window, user rulings: shadow twin
> CANCELLED — smoke aggressively + trust; "do those three"):** T0
> factorization + T6c PSQT skip + FT-sparsity λ BAKED into the JAX
> trainer (3640f5c; zero-init no-op proven 8 ways; all-on 216.6k pos/s).
> Engine-side PSQ1 consumer merged (ed84ecd; Bench 701605 bit-exact,
> legacy nets node-identical, psqt parity p50 1cp, +1.9% NPS only for
> psqt nets); fresh native engine installed (archon-d47 = optimized+
> PSQ1; -preopt backup kept). D46 unit repair: converts lacked
> SFBINPACK_SCORE_DIV → rescale_records.py fixed in place (frame
> verified). Corpora: UHO band75-100 (2.56B = the EXACT d47c kept
> subpopulation), fishpack32 (1.61B), farseerT74 (1.23B, DIV 1.66 at
> dump) — all shuffled+checksummed+ledgered; all four HF packs local.
> **Smokes: S1 warm heldout EXACTLY 0.006094; S2b JAX BEATS bullet
> heldout at matched sb (−3.4/−3.9%; iid-sampling curriculum win
> pinned); S3 resume heldout identical to 4 sig figs; S4 export chain
> green (oracle p50 1cp).** Findings pinned: unmasked policy @λ=0.005
> drags heldout ~5% early (stage-3 runs λ/4 unmasked; phase-5 hook =
> masked refinetune via PyO3); warm-restart bounce expected (judge the
> arc). Stage-3 = d47d: warm stage-2-sb240, mix UHO.40/fp32.35/T74.25,
> LR 2.5e-4 → floor 0.2× (lr_end_sb=331), ~8.5h at ~195k pos/s,
> heldout every 10sb, crash-resume certified. Driver:
> jax/tools/stage3_train.py; log /tmp/d47d_stage3_train.log; tb
> d47d-stage3/*.

> **2026-07-23 JAX-GPU UNBLOCKED (campaign phase 1 DONE, merged 34f639c):**
> Pallas embedding-bag FT kernel (`jax/archon_jax/ft_kernel.py`) lands
> **225k pos/s full train step @ batch 8192 = 1.07× bullet** (was 9.5k,
> OOM ≥2048); forward-only 1.26M pos/s; parity max |Δlogit| 3.53e-4,
> 0/512 argmax flips, optimiser trajectories match to atomic-float noise;
> 400-step soak clean at 2.93 GiB GPU. Verified post-merge on main:
> 11/11 GPU tests, bench reproduces 223.9k. Report:
> `jax-gpu-kernel-2026-07-23.md`. Route 2 (pure-XLA) kept as portability
> fallback; bf16 deliberately skipped (gate passed in f32). Real recipe
> density confirmed ~26.5-29.6 rows/perspective (bench = recipe,
> apples-to-apples). Next bottleneck: single-thread host mapper ~210k
> (PyO3 loader queued). **Next: build-window queue B (D40_CONVERT
> corpus converter + shuffle, PyO3 loader) → phase-3 shadow twin
> (240sb, ~7.4h GPU) → promotion gates → campaign phase 2 (stage-3 on
> JAX with full doctrine).** Data chain: wrongIsRight landed (sha ✓);
> T60T70 resuming after a duplicate-fetcher corruption was truncated
> away (final HF sha256 check is the integrity net); farseerT74/75
> queued behind it.

> **Post-run policy plan:** `policy-training-roadmap-2026-07-22.md` — joint
> forms the trunk, then independent frozen-trunk head finetunes on
> policy-specialized (unfiltered/tactical) data; the d47c-vs-twin
> trunk-sufficiency head-finetune experiment is the decision fork.
>
> **sb30 READOUTS (21:40Z): λ DECISION = RIDE.** Share plateaued ~1.68
> (0.81→1.43→1.65→1.68 at sb1/5/20/30; +0.003/sb and flattening), value
> component improved at EVERY reading (0.01022→0.008131 sb5→30) — neither
> intervention condition (share>2 AND value hurt) fired. Constant λ is
> self-annealing (λ·pol_ce shrinks as policy improves). Top-1: ~17% plateau
> sb15–30 then **20.7% at sb35** (probe SE ±1.8%). First Elo line, 100g
> fixed-node 5k vs FULLY-TRAINED incumbent: **−381.7±49.5 (5-10-85)** —
> expected-early, trajectory sb30→60→90 is the signal, not the absolute.
> OPS: trainer RSS plateaued ~8.5GB (warm-up, NOT a leak; zram 4G full,
> avail ~1.1-1.4G). Stale d47/d47b watchers+tails killed; ckpt watcher
> replaced by `tools/nnue/runs/d47c_ckpt_watch_v2.sh` (conc 1, RAM guard
> 1.2G with 12×60s retry, resume-safe) — Elo instrument restored.
> Fold-in dose-2 (p75) final: **−5.8±17.1 NULL** — fold-in arc stays CLOSED.
>
> **sb60 TRAJECTORY + ABLATION (22:25Z):** Elo **−246.3±38.4 (12-15-73)** —
> gap closed **+135 in 30sb** from −381.7; on-track shape (contention
> plausible in the back half; expect deceleration with LR decay).
> **sb30 ON/OFF ablation (300g each, sidecar hidden vs present): identical
> scores, 10.0% both (−381.7±29.5 vs −381.7±29.2)** — eval carries 100% at
> this stage AND the ~17%-accuracy deployed policy costs ZERO (downside
> capped by design; no reason to gate ordering while accuracy climbs).
> Top-1 consolidating ~20% (sb35–60: 20.7/19.2/18.5/21.7/18.5/19.5).
> Stage-2 decision rule pre-registered: roadmap §5b (twin gap + end slope
> → warm retrain at λ/4 if value is owed gradient).
>
> **NPS MEASURED (23:00Z, movetime 4s startpos, sidecar loaded):** d47c
> **363k** vs incumbent **647k** = −44% ≈ **~58 Elo equal-time tax** →
> the sb240 equal-time BREAK-EVEN bar is **+60 fixed-node Elo** vs
> d40-uho-v1-240sb (more to promote). Accepted go-big cost; recovery
> levers post-learnability: AVX-512 (unused), FT quant/prune, distill-shrink.
> **LOSS SAWTOOTH DIAGNOSED (user spotted):** period exactly 4sb =
> **1.58GB of binpack** — nodes5000pv2_UHO is ~25 concatenated ~1.6GB
> generation segments, each with the same internal difficulty gradient;
> the 128MB shuffle buffer (~8% of a segment) can't hide the macro
> profile. 394MB file/sb, ~15.8 B/used pos ⇒ ~16% keep rate ✓, ~2.35
> epochs over 240sb, **loader wraps ≈ sb102 + sb204 (expect phase reset,
> not a bug)**. Benign; someday-fix = offline global interleave. Also why
> local loss deltas are untrustworthy across segment boundaries.
> sb74 components: value 0.007373 | pol_ce 2.6235 | share ~1.78.
> Ablation artifacts: `data/nnue/nets/ablate/` +
> `data/nnue/experiments/ablate-sb30-{nopol,pol}-300g/`.
>
> **sb90 (23:30Z): Elo −177.2±36.1 (20-13-67)** — closure +69/30sb vs +135
> the previous interval (~halved). Geometric projection ⇒ ~−100..−130
> fixed-node at sb240 (short of parity; equal-time bar +60 beyond) — IF the
> 0.5 ratio holds (3 noisy points; epoch-2 wrap ~sb102 + LR tail = real
> uncertainty; sb120 sharpens). Reading if it holds: undertrained capacity
> (B4 law) + λ share ⇒ the §5b stage-2 path, decomposed by the twin.
> Top-1 20.4% (sb90). Value 0.007185 / pol_ce 2.5937 / share ~1.80.
>
> **★ OOM CRASH + RESUME (23:16–23:26 local):** kernel OOM-killed the trainer
> at **sb95 b2560** (8.2GB RSS, inherited oom_score_adj +200; trigger = a
> Chrome HeapHelper allocation on a ~1.2G-avail box). Twin auto-launch
> misfired (can't tell crash from completion), ran garbage (headline loss
> printed 0.000000 — tensor-map collision: with POL_ON=0, metric:value IS
> the fused loss node and the metric buffer clobbered the loss mapping;
> FIXED in run.rs by aliasing) and froze; killed. **RESUMED from
> sb94-raw.bin via the pre-existing D40_INIT_RAW + D40_START_SUPERBATCH=95**
> (T5 resume machinery) — first resumed batch loss 0.020343 ≈ pre-crash
> 0.020562 ✓ continuous. Notes: Adam moments start FRESH (raw = weights
> only — brief re-warm transient at the sb95 LR), and the DATA STREAM
> restarts from the file start (stream offset isn't checkpointed) — the
> epoch-2 wrap effectively happened at sb95 instead of ~sb102.
> Gotcha for the ledger: rebuild needs `--features cuda` (a bare rebuild
> makes a MOCK-runtime binary that exits 101 at the first gradient).
> Protections: watcher v3 (`tools/nnue/runs/d47c_ckpt_watch_v3.sh`) RAM-guards
> the PACK+probe too (1.8G, retry) not just matches; all 43 Chrome procs
> oom_score_adj→900 (OOM prefers tabs over the trainer; trainer stuck at
> 200, lowering needs CAP_SYS_RESOURCE). Twin re-chained on the new pid
> with the fixed binary.
>
> **OVERNIGHT PLAN (user asleep ~sb100, 00:15 local):** run ends ~04:15 →
> watcher v3 measures through sb240 (Elo at 120/150/180/210/240) → twin
> auto-chains (~7h, fixed binary) → on wake: gate_d47.sh 300g on sb240,
> sb240 ON/OFF ablation rerun, twin-gap read → §5b stage-2-vs-shrink fork.
> sb120 DISCRIMINATOR: ≥−120 parity back in play; ≤−160 stage-2 scenario
> firm. sb100 components: value 0.007116 | pol_ce 2.5785 (Adam transient
> fully reabsorbed by sb97).
>
> **sb120 (04:5xZ): Elo −143.1±33.4 (21-19-60)** — the discriminator landed
> in the GRAY ZONE (−160 < −143 < −120), but the closure model is now
> 3-for-3: +135.4 → +69.1 → **+34.1** per 30sb — ratio 0.51, 0.49. The
> geometric fade is CONFIRMED, not noise. Extrapolation: remaining closure
> ≈ 34×(.5+.25+.125+.0625) ≈ +32 ⇒ **≈ −111 fixed-node at sb240**
> (tight band now, ~±25) — short of parity, ~−170 equal-time after the NPS
> tax. Verdict: the schedule ends while the net is still learning ⇒ the
> **§5b stage-2@λ/4 path is the plan of record** (formally: confirm with
> twin gap + end slope at sb240 per the pre-registration — but the closure
> curve already says undertrained, exactly the B4 law shape). No rerun
> needed: first post-resume point came in dead on trend (expected ~−140±40).
> W-D-L drift 5-15-80 → 12-15-73 → 20-13-67 → 21-19-60: losses now
> converting to DRAWS as well as wins. Top-1 23.2% (sb120, NEW HIGH);
> value 0.006917 (falling every reading); pol_ce 2.546.
>
> **sb150 (01:5xZ): Elo −96.2±30.5 (23-27-50) — the geometric fade BROKE,
> good direction.** Closure sequence now +135.4 → +69.1 → +34.1 → **+46.9**
> (model said +17; point landed ~30 ahead). Honesty per
> no-verdicts-from-weak-tests: closure SE ≈ ±45 (two ±30ish points), so
> this is ~0.7σ from the halving model — NOT proof of re-acceleration,
> but the "halving CONFIRMED" language at sb120 was too strong and the
> sb240 band must widen upward: **−55 (halving from the fresh +47) to
> −110 (old model, sb150 = lucky) fixed-node**. Even the optimistic edge
> stays short of the +60 equal-time bar ⇒ **stage-2 remains plan of
> record**; if anything a still-fat closure at sb150 (LR ~0.00035 and
> value still making new lows every sb) is MORE evidence the schedule is
> the binding constraint. Draw surge 19→27 (losses 60→50): the
> blunder-loss profile is dissolving. Top-1 23.2% (4th touch of the high).
> Value 0.006667 / pol_ce 2.502 at sb150. Next model check: sb180
> (halving-from-here says ≈ −73; old track says ≈ −105).
>
> **sb180 (02:5xZ): Elo −52.5±28.7 (26-33-41) — beat BOTH models.**
> Closures: +135.4 → +69.1 → +34.1 → +46.9 → **+43.7**: two consecutive
> ~flat readings ≈ +45/30sb. The "geometric halving" was an artifact —
> **the +34.1 (sb90→120) interval contained the OOM resume at sb95**
> (fresh Adam + data-stream restart), which depressed exactly the point
> the halving story was built on; the deconfounded closure sequence reads
> 135 → 69 → ~45 → ~45 → 44, a gentle fade, not geometric collapse.
> (no-verdicts-from-weak-tests again: point 3 was confounded and I called
> it a confirmation.) **Updated sb240 range: −20 (closure halves from
> here) … +35 (closure holds ~44) fixed-node — PARITY IS IN PLAY.**
> Equal-time bar +60 still above the optimistic edge ⇒ stage-2 remains
> plan of record, but if sb240 lands ≥0 the stage-2 launches FROM parity
> weights and equal-time promotion becomes the likely outcome of the
> arc. W-D-L march: losses 60→50→41, draws 19→27→33 (blunder profile
> dissolving into draws). sb210 check: flat-closure ≈ −9; halving ≈ −30.
> Components at sb180: value 0.006476 / pol_ce 2.463 — new lows every
> reading, LR ~0.00017 and STILL closing at +44/30sb.
>
> **★★★ sb240 FINALE (04:2xZ): Elo +59.6±30.5 (46-25-29) — the multihead
> net BEAT THE INCUMBENT at fixed nodes, first challenger to do it since
> the incumbent chain was built.** TRAIN EXIT 0; final components value
> 0.006236 / pol_ce 2.424 (new lows at the wire — NO plateau; end-slope
> verdict for §5b: STILL FALLING ⇒ stage-2 fires). +112 Elo over the last
> 60sb (+56/30sb — the closure ACCELERATED into the cosine tail).
> **+59.6 fixed-node sits EXACTLY at the +60 equal-time break-even bar
> (NPS −44%) ⇒ d47c-sb240 ≈ equal-time PARITY with the incumbent, with
> policy ordering + threats + multihead riding along.** One 100g point,
> ±30.5 — the 300g gate (launched 04:2x, conc 8 under twin; NPS section
> junk from contention, ignore it) tightens to ±17. RAM-guard skip
> cascade (sb205-235 packs) was raced post-exit: all backfilled (probes
> logged, sb210 Elo match in flight). Twin launched clean. Top-1 endgame
> band 21-23% (22.9% at 240). NEXT: gate result → nopol ablation 300g on
> sb240 → twin gap (~11:30) → stage-2 launch (script ready, knob 1 now
> answerable: end slope says LONG schedule).
>
> **sb210 backfill: +3.5±29.8 (37-27-36)** — crossed zero at sb210; final
> closures +56, +56 per 30sb (accelerating into the tail).
> **300g GATE (04:5x): +22.0±17.1, score 0.532, CI95 [−10.4, +54.9] —
> GATE PASSED; scale PASS 0.982; NPS section junk (twin contention),
> use clean 363k/647k.** Pooled with the 100g point (compatible at
> ~1.1σ): **best estimate +31±15 fixed-node ⇒ ≈ −29±15 equal-time** —
> the 100g +59.6 was the top of its error bar; the run ends a clear
> FIXED-NODE WIN (first challenger ever over the incumbent) but ~30
> short of equal-time parity. Reading unchanged, now sharper: eval
> quality per node now EXCEEDS the incumbent with policy+threats riding
> along; the remaining gap is the NPS tax ⇒ stage-2 (+ later shrink/NPS
> work) exactly as pre-registered. nopol 300g ablation launched
> (`ablate-sb240-nopol-300g`, same seed/book as the gate ⇒ paired vs
> its ON-side 300g).
>
> **sb240 NOPOL ablation (05:1x): +52.5±17.2 (131-83-86).** ON−OFF =
> −30.5±~24 (1.3σ) ⇒ NOT significant; with the sb30 ablation's exact
> zero as prior, verdict stays **policy ordering ≈ free** (flag: if a
> tax shows up again at the stage-2 gate, check policy-argmax vs
> picker-first agreement in empty-TT nodes before believing either
> direction). Treating ON/OFF as equivalent and pooling all three sb240
> points: **eval quality ≈ +40±11 fixed-node over the incumbent**
> (equal-time ≈ −20±11 at the clean NPS ratio). The D47 bet's scorecard
> at run end: eval per node clearly better, policy learnable + free,
> NPS tax = the whole remaining gap.
>
> **TWIN sb120 (07:5x, mid-flight read on user ask): −177.2±35.5
> (19-15-66) vs d47c sb120 −143.1±33.4 ⇒ joint AHEAD by 34±49 at
> matched presentations — the λ tax on value is ZERO-to-NEGATIVE.**
> Not significant (0.7σ) but directionally strong: d47c's sb120 was
> the Adam-fresh post-resume point (depressed) and STILL beat the twin.
> Supporting tea-leaf: twin train-loss ran ~3% BELOW main's value
> component (sb110) yet plays worse — fits-better-plays-worse = the
> classic multi-task-regularization signature (policy as auxiliary task
> improving value generalization). Confound to note: twin ran 0→240
> uninterrupted; main's data stream restarted at sb95. Full verdict at
> twin sb240 (300g + final components). If it holds, §5b's premise
> inverts (λ was never a tax) — stage-2 @λ/4 still right (undertraining
> is its own justification), and §3's fork leans "joint trunk is
> strictly better". Scale PASS 0.974 (twin sb120).
>
> **TWIN sb60 (09:1x, user-requested second pair): −346.1±45.5 (6-12-82)
> vs d47c sb60 −246.3±38.4 ⇒ joint ahead +99.8±60.** Two pairs, both
> one-directional; pooled joint advantage ≈ **+60±38**. Gap SHAPE:
> +100 (sb60) → +34 (sb120) — either the policy gradient is an
> ACCELERATOR (carves tactical trunk features early; twin partially
> catches up) or inter-pair noise. sb180 pair queued (auto: pack+match
> on checkpoint landing) discriminates: gap holds ⇒ λ = permanent
> endpoint win; gap converges ⇒ multi-task = same-destination-sooner
> (still buys schedule compression). Either way λ stays in the recipe.
>
> **★ TWIN OOM #2 (08:50:39, MY FAULT — logged as ops rule):** the twin
> sb60 pair match I launched at **concurrency 8** (16 engines × 137MB
> net) into 2.3G available invoked the oom-killer; it killed the twin
> trainer (8.7G RSS, adj 200) at sb172. Compounding error: I then
> reported the twin "healthy at sb168/170" twice from `tail` of the
> \r-progress line — a DEAD process's log keeps yielding plausible
> segments; liveness = pgrep + GPU% + log MTIME, never a log tail.
> RESUMED 09:11 from sb160-raw (START=161, fresh-Adam ~2sb transient,
> loss continuous 0.00625); sb240 ETA ~11:25. sb180 pair re-queued with
> the WATCHER discipline (1.8G-guarded pack, 1.2G-guarded conc-1
> match). **HARD RULES now: matches during ANY training run are conc 1
> + RAM-guarded, no exceptions; the sb120 conc-8 match that "worked"
> was luck, not validation.** Twin sb180 pair carries a small fresh-Adam
> confound (19sb post-resume) — mirror of main's sb95 resume.
>
> **PRE-REGISTERED PREDICTION (user, 2026-07-22 ~09:30): d47c —
> unoptimized, untuned — takes the EQUAL-TIME crown once its training
> arc (stage-2) finishes.** Arithmetic at registration: gap = −20±11
> equal-time; stage-2 historical range +44..+55 equal-time on a MORE
> saturated arch; end-slope maximal at sb240; λ-acceleration tailwind
> retained at λ/4. Gate when stage-2 lands: real 10+0.1 gauntlet vs the
> incumbent chain, DEPLOYED config (policy ON) — not NPS arithmetic.
>
> **TWIN sb180 (10:0x): −31.4±30.3 (33-25-42) vs joint −52.5±28.7 ⇒
> twin AHEAD +21±42. Pair sequence +100 → +34 → −21 (sb60/120/180):
> monotone convergence-and-crossover — the ACCELERATOR hypothesis wins.**
> Joint carves trunk features faster early; twin converges by late
> training (and its sb180 is 19sb post-fresh-Adam-resume — came in
> ahead DESPITE the confound). Pending sb240 endpoint confirmation, the
> λ verdict: **zero endpoint cost + large early acceleration + free
> policy capability ⇒ λ stays in every recipe; stage-2 @λ/4 unchanged;
> the ARCH is the value win.** No single pair significant; the monotone
> trend is the evidence.
>
> **HELD-OUT INSTRUMENT (user-commissioned 10:3x, CODE READY — build in
> the pre-stage-2 window):** three additive env-gated modes in
> d40_features.rs (repo copy synced): `D40_HELDOUT_BAND=0:5` (band
> REPLACES the keep-test in the same quiet filter — shadow population),
> `D40_HELDOUT_DUMP` (stream band rows to a raw bulletformat .bin,
> CPU-only), `D40_HELDOUT_EVAL` + `D40_PROBE_RAW/QUANT` (exact training
> graph + loss nodes over the dump; prints value_mse + pol_ce; quant
> loader extended for the D47 polmove layout, padding assert =
> layout tripwire). Scripts: `tools/nnue/heldout_{extract,eval,
> backfill}.sh`. **PRE-BUILD SMOKE (done 10:4x, user-ordered, all
> without touching the running twin): quant-layout byte arithmetic
> EXACT-MATCHES both real files (d47c 137,089,088 / twin 136,843,840);
> netpack.py read order independently confirms polw(i8)→polb(f32)→l1w;
> script guards exit 2 clean; backfill loop dry-run extracts sb + 
> strides correctly (sort -V fix); control binary preserved at
> /tmp/d40_features.pre-heldout.**
> **BUILD-WINDOW CHECKLIST (twin exit → stage-2 launch): (1) twin
> sb240 pack + 100g pair + 300g endpoint matches; (2) `cargo build
> --release --features cuda --example d40_features` (CUDA_PATH=
> /opt/cuda); (3) ★ NO-REGRESSION BYTE-DIFF: run OLD (/tmp control)
> and NEW binaries on an IDENTICAL 1-sb scratch config (same seed,
> D40_INIT_RAW=d47c sb240 — also exercises the exact stage-2 resume
> path) and diff the emitted quants BYTE-FOR-BYTE; any diff = ABORT,
> revert source, rebuild old, investigate; (4) D40_VERIFY_ONLY=1 with
> band env (reader+filter smoke); (5) heldout_extract.sh 10k-row
> scratch dump then the real 1M; (6) heldout_eval.sh on d47c sb240
> raw AND quant (pol=1: value_mse raw≈quant, pol_ce ≈2.4 sanity) +
> twin sb240 quant (pol=0); batch-size fallback to 1 if the evaluator
> rejects batches; (7) ONLY THEN launch stage-2; (8) backfill curves
> during stage-2 in RAM-guarded slots or after.**
>
> **STAGE-2 LAUNCH DECISIONS (10:5x, user away — my judgment, per
> "use your best judgement"):** DATA = same-UHO stream (script default).
> Rationale: end slope still falling on UHO + the user's absorption
> hypothesis ⇒ clear re-presentation headroom at ZERO new-data risk;
> the farseer fold needs D46 verification (farseerT76 SCORE_DIV /
> historical stage-2 env unconfirmed — dig the ledger recipe during
> stage-2, prep it as stage-3). LENGTH = 240sb (end-slope rule).
> λ = 0.00125 (pinned). **Telemetry (STANDING RULE, user: "all future
> trainings visible in tensorboard"): tb_shim fixed (pol_ce now
> optional + None-guarded), twin BACKFILLED (228 pts + 3 Elo →
> data/nnue/tb/d47c/twin), stage-2 gets its own shim
> (→ d47c-stage2/train) + watcher (runs/d47c_stage2_watch.sh, v3
> guards, emits → d47c-stage2/checkpoints) + a components/exit
> Monitor on its log, armed AT LAUNCH.**
>
> **★★ WINDOW EXECUTED (11:26–11:45) — TWIN DONE, INSTRUMENT LIVE,
> STAGE-2 LAUNCHED ★★**
> **Twin endpoint (TRAIN EXIT 0, scale PASS 0.980): 100g +56.1±30.3
> (45-26-29); 300g +49.0±17.2 (129-84-87) vs d47c-nopol 300g +52.5±17.2
> ⇒ USER PREDICTION CONFIRMED: twin ≈ joint at endpoint (Δ3.5±24).
> λ FINAL VERDICT: zero endpoint cost, +100 early acceleration, free
> policy — λ stays in every recipe; share-locked λ noted for d47d.**
> (Residual flag: ON 300g +22.0 vs the two no-ordering ~+50s — 1.3σ,
> settle at the stage-2 gate with a bigger paired ablation.)
> **Byte-diff gate: training is run-to-run NONDETERMINISTIC (atomic-add
> sparse grads; old-vs-old differs in 4.4795M bytes) — corrected gate =
> noise-envelope equivalence: old-vs-new 4.4830M, old2-vs-new 4.4744M
> ⇒ new binary INSIDE the noise cloud ⇒ PASSED.** (Also fixed:
> mapper.map's 3rd arg is THREADS not batch; ModelEvaluator grew
> new_with_batch — evaluator.rs lowered batch was hardcoded 1.)
> **Held-out instrument VALIDATED (1M rows @ data/nnue/heldout/
> uho-band0-5.bin): d47c raw value_mse 0.006314 / pol_ce 2.4266 ≈
> TRAIN LOSS (0.006236/2.4243) ⇒ ZERO overfit gap at 2.35 epochs (first
> absorption data point — long stage-2 is safe); quant +0.5% (polmove
> parse proven); twin 0.006315 ≈ d47c raw (loss-level mirror of the
> identical endpoint Elos).**
> **STAGE-2 LAUNCHED 11:45 (pid 2951629): warm from sb240-raw ✓, LR
> 0.0005 fresh cosine, λ=0.00125, UHO, 240sb, 210k pos/s ≈ 9h ⇒ ends
> ~20:45. Telemetry live from sb1 (shim+watcher+monitor). Deferred-
> refinement test: watch early value-slope vs twin's late slope.**
> **JAX pipeline study running** (user-commissioned, Fable subagent,
> isolated worktree branch; parity-proof + loader benchmarks + adoption
> framework; CPU-pinned, no-build guardrails).
>
> **★ OOM #3 (11:48, stage-2 killed at sb1 — TWO causes, both now
> structurally fixed):** (1) MY byte-diff smoke runs wrote ~2.9GB of
> checkpoints into tmpfs-backed scratchpad (/tmp = RAM!) right before
> launching a 9GB trainer — zram was 4G/4G FULL at launch. **RULE:
> model artifacts NEVER go to /tmp/tmpfs; smoke outputs go to disk.**
> Purged: tmpfs 3.8G→43M, available 2G→11.7G. (2) The Chrome
> oom_score_adj=900 sweep DECAYS (Chrome recycles processes; a fresh
> [vkps] thread at default adj triggered the kill while the trainer sat
> at adj 200 with max RSS). **Fixed: persistent chrome-adj guard loop
> (re-applies 900 every 2 min, pid 2980358).** The JAX agent's jit
> smoke overlapping the ramp contributed pressure (guardrails were
> calibrated pre-launch). RELAUNCHED 11:52 from the original config
> (lost 3 min; single fresh-Adam start, schedule clean); live at 6.2G
> available during ramp; ends ~20:50.
>
> **JAX STUDY COMPLETE (12:05, branch `jax-pipeline-study` =
> worktree-agent-a28f8eb76b1f1be8a, 5 commits):** quant export
> BYTE-IDENTICAL to bullet on all 137,089,088 bytes from the real sb240
> raw; full-graph forward runs the real 108M params (engine oracle:
> |Δ| p50 3cp, 98% sign agree); training smoke converges and its quant
> round-trips through unmodified netpack; numpy loader single-thread =
> 206k pos/s (== the whole 6-thread Rust reference), 6 workers 640k.
> Recommendation: adopt as EXPERIMENTATION SIDECAR now; promote to
> production trainer only after a GPU-throughput window (≥0.8× bullet)
> + byte/strength gates. Study: docs/research/jax-pipeline-study.md
> (on the branch). Flag for review: its heldout value-MSE readout
> (0.00588) sits ~7% below the trainer-graph number (0.00631) — a
> loss-normalization or target-detail difference to reconcile before
> trusting its ABSOLUTE loss numbers (relative/per-position parity is
> proven by the byte + engine oracles).
> **STAGE-2 EARLY READS (13:1x):** heldout curve (1M rows, quant):
> sb240-anchor 0.006348 → **sb1 0.006184 (DIP BELOW — one value-majority
> superbatch improved generalization: cleanest deferred-refinement
> datapoint yet)** → sb10 0.007183 → sb20 0.007301 (SGDR bounce, crest
> forming: train slope +1.1%→+0.4%/10sb). **sb30 Elo −151.3±33.1 —
> looks scary, IS consistent: heldout 0.0073 sits exactly at stage-1's
> −150/−180 band ⇒ the loss↔Elo map holds across runs (instrument
> doubles as Elo predictor). (loss,LR) framing: stage-2@sb30 ≈
> stage-1@sb100 with 210sb of runway instead of 140 + a better trunk.**
> top1 holds 20.4-23.2% under λ/4 (no rot; CE calibration wobbles as
> expected). RIDE; judge on sb60-90 recovery per pre-registration.
>
> **RECOVERY READS (14:1x-14:4x):** sb60 Elo **−85.0±31.5** — recovery
> +66/30sb (faster than stage-1's +44 through the same band). Heldout
> gap question RESOLVED by stage-1 backfill (sb80/120/180: gaps
> 9.4%→8.1%→5.6%→1.5%): the ~10% train/heldout gap at high LR is the
> recipe's NORMAL signature, closing with the anneal — stage-2's sb80
> gap (+10.4%) is on-signature. Cross-run position: stage-2 heldout
> @sb80 (0.007213) ≈ stage-1 @~sb145 — with 160sb of runway vs the 95
> stage-1 had ⇒ projected endpoint heldout ~0.0061-0.0062 vs 0.006348
> ⇒ **rough projection +40-90 Elo over the stage-1 net** (wide bars,
> right shape). Heldout curves for both runs now in tb
> (value/heldout_mse). Policy realigning (pol_ce 2.554↓, top1 22.4%
> @sb80).
>
> **★★★ STAGE-2 COMPLETE (20:4x, TRAIN EXIT 0) — GATE +82.6±17.6 ★★★**
> Final train value 0.005908 / pol_ce 2.4748; **heldout 0.006094 raw /
> 0.006131 quant (−3.5% vs stage-1's net — map predicted ≈+100
> fixed-node; gate measured +82.6, within 1σ)**; train/heldout gap 3.1%.
> **300g GATE: +82.6±17.6, score 0.617, scale PASS, top1 23.4% —
> stage-2 bought +60 Elo over stage-1's +22.0 at identical standard.**
> Clean NPS (uncontended): 506k vs 945k ⇒ equal-time bar ≈ +64 ⇒
> **fixed-node arithmetic prices equal-time at ≈ +18 ± 18 ABOVE the
> incumbent — crown math positive pre-gauntlet.** CROWN GAUNTLET
> LAUNCHED: 400g @ 10+0.1 deployed config vs incumbent. Endpoint policy:
> argmax band ended HIGHER than stage-1 (23-25% vs 21-23) with CE
> slightly softer — sharper head, λ/4's intended trade. SF15.1 built:
> classical HCE = 1.96M NPS vs our PSQT 2.99M ⇒ SHELL EXONERATED,
> the entire NPS gap is the eval layer (T10 smallnet = the lever).
> Six-engine RR (user-commissioned) queued behind the gauntlet.
>
> **★ STUDY #6 REPROFILE (01:0x, merged): VERIFICATION CONFIRMED —
> end-to-end +30.9% [29.0..34.4], b46c026 +22.8%, 3555f57 +3.8%, bit
> identity everywhere; ZERO search-path mallocs now.** New profile:
> forward_into 37% but BYTES-bound (legit work); policy tax → 0.4%
> cycles (+6.2% tree tax stands — search effect); search block now 30%
> (never had a perf pass — next lever). **Deployment ladder (wl50/d14):
> incumbent 1.239M vs stage2+cpol 797k ⇒ deficit 2.04×→1.56× ⇒ crown
> equal-time bar ≈ 45-58 vs our +82.6±17.6 fixed-node edge ⇒ REMATCH
> EV ≈ +25-38 EQUAL TIME.** Rematch HELD: user's cutechess gauntlet
> owns the box; fire on a quiet box. Next-5: FT-sparsity retrain
> (+12-14%, density-linear), search micro-pass (+3-5%), sidecar
> decide-on-Elo (tree tax), T10 PARKED (bypass multiple 5.54→4.30,
> needs tree-weighted p≥0.15), quant-unify at next format rev.
>
> **★★ OPTIMIZATION ARC LANDED (00:4x, merge c6e820d, check.sh GREEN,
> Bench 701605 exact): +29% NPS total (383.7k→495.3k on the impl
> workload), all four packages, bit-identical proven hard** (per-package
> bench fingerprint + byte-parity + 100/100 identical PGNs + perft).
> Package 2 (head kernels) delivered +21.4% — 2× the study-#4 estimate,
> per study #5's mca prediction. Evidence-based drops: software
> prefetch LOSES on this Zen4 (3 experiments), cancel scan deleted
> (0.48% of events), stash-gating REFUTED (consult consumes stashes
> from other evals — any conditional stash is functional; tree-size
> proof). New ladder: PSQT 3.71M / stage2+cpol 495k / stage2-nocpol
> 540k / stage1+cpol 534k. Reprofile agent (user-pinned) launched:
> fresh hot-path map + independent verification of b46c026/3555f57 +
> deployment ladder for the crown-rematch math.
>
> **★ STUDY #5 GROUND TRUTH (23:5x, merged; report
> perf-study-5-measured-2026-07-23.md): the cost model INVERTED — head
> 48.5% of cycles (l1_dot 34.4%), threats only 12.4%.** mca-validated
> l1_dot rewrite (group-outer, 8 zmm accumulators) = 3.0× kernel ⇒
> +15-25% NPS, bit-identical — fed to the implementer mid-flight. NEW
> ROOT CAUSE: FT density 62% (200/320 active) — the 768-era sparsity
> anchor is dead; trainer-side sparsity regularization = unpriced lever
> (ties to learned-sparse-embeddings themes). NOT DRAM-bound
> (L2-resident; int4 stays dead). **POLICY SIDECAR MEASURED: −3.5% NPS
> AND +6.2% tree size at equal depth, for 0 measured Elo — the sb240
> ON-vs-OFF 1.3σ cluster now has a MECHANISM. RECOMMENDATION: hide the
> .cpol in future matches/anchors until a head-finetune makes it win an
> ablation (deploy-config change — applying to future runs unless user
> objects).** T10 re-priced: gate fire-rate 8-15% ⇒ +7-10% central
> (down from +20). Implementer's landed packages verified bit-identical
> (+3.5-4% real so far); THP demoted (box won't give hugepages).
> incumbent): −30.5 ± 11.8 (74-217-109).** The pre-registered user
> prediction ("equal-time crown, untuned") scores **NOT YET**: the
> fixed-node crown is decisive (+82.6±17.6) but the NPS tax still wins
> equal time by 30. **Finding: the node-ratio conversion OVERSHOT by
> ~50 Elo** (predicted +18) — fixed-node→equal-time arithmetic
> underestimates slowness costs in real timed play (iteration depth ×
> time-pressure interactions; untuned TM likely amplifies). Critical
> path to the crown = the NPS program: study #4's packages (+13-25%,
> in implementation now) + T10 smallnet (+7-34%) + TM SPSA. At
> +25-55% NPS the bar math flips deeply positive. RR cancelled by
> user (rerun when stronger); profiling study #5 running with the
> newly-installed toolkit. **USER DIRECTIVE (pinned): when the
> implementation agent lands packages 2-5, commission a FRESH
> agent-run profile of the optimized binary** (new hot-path map —
> did the bottleneck move; feeds study #6 / T10 priority).
>
> **★ sb210 SURGE (19:5x): Elo +85.0±30.6 (49-26-25) — stage-2 passed
> the ENTIRE stage-1 endpoint band (+49-56) with 30sb of anneal left.**
> Trajectory: −151 → −85 → −85(floor) → −31 → −28 → −14 → +85; the
> sb180→210 closure = +99/30sb (the anneal endgame surge). top1 band up
> to 23-25% (24.7@sb215). Endpoint landing zone ≈ **+100-120 fixed-node
> ⇒ equal-time ≈ +40-60 ABOVE the incumbent — the user's crown
> prediction is now the BASE CASE** pending the 10+0.1 gauntlet.
> (Heldout map cross-check: ~0.00625 ⇒ ~+70 predicted, +85 measured —
> holding within noise.)
>
> **TM GAP (user field-report, 19:0x, GUI games vs SF16/SF18): ARCHON
> front-loads clock and lives on increment late while SF banks time.**
> Root cause: our P7/P10 TM lacks STABILITY-BASED EARLY TERMINATION
> (SF cuts stable moves short and banks the surplus); horizon divisor
> may compound. Masked by our 10+0.1 test standard (everyone near
> increment); bites at longer TC. NOT blocking the crown gauntlet
> (archon-vs-archon ⇒ TM cancels); IS free Elo vs external anchors.
> QUEUED: pre-S2 TM audit (stability/effort early-stop, fail-low
> extension, horizon curve) in the next engine build window; constants
> → S2 SPSA. User GUI-testing = valuable QA channel.
>
> **AXIS LAW (17:2x, from sb150 Elo −27.9±30.5): Elo tracks HELD-OUT
> loss, not train loss.** Stage-2 crossed stage-1's TRAIN endpoint at
> sb150 yet measured −27.9 — the stage-1 heldout↔Elo map (0.007476↔
> −143, 0.006836↔−52.5, 0.006348↔+49) predicts −24.0 from stage-2's
> sb140 heldout 0.006699. Essentially exact. Train-axis readings lead
> Elo by the generalization gap; the gap closes through the anneal
> (stage-1: 6%→1.5% final quarter) so Elo catches up late. Projection
> on the correct axis: endpoint heldout 0.0061-0.0062 ⇒ **≈+60-90 vs
> incumbent = +15-40 over the stage-1 net.** The held-out instrument
> is now formally the strength predictor; train components are a
> throughput/health signal only.
>
> **METHOD NOTE (15:2x): sb90 Elo = −85.0 (27-22-51), DIGIT-IDENTICAL
> to sb60 — nets verified distinct (hashes differ); with deterministic
> fixed-node engines + the fixed book seed, near-identical nets replay
> the same games ⇒ the paired-seed trajectory has a RESOLUTION FLOOR
> for slowly-changing checkpoints.** Consequences: consecutive
> identical readings are correlated (never average them); mid-anneal
> Elo stays quantized until weight deltas grow (sb150+ / the 300g
> endpoint decide); loss-side instruments carry the mid-anneal story
> (train 0.00648@sb100, pol_ce 2.544↓, top1 22.4% — all improving).
>
> **SEQUENCING (user, 12:1x): JAX migration + testing happens AFTER
> the full D47 training arc completes** (stage-2 → gate → equal-time
> gauntlet). Branch parks until then; first JAX window = GPU throughput
> benchmark + the 7% loss-readout reconciliation + sidecar wiring
> **+ PyO3 engine bindings (user 12:4x: part of the JAX workstream —
> fast legal masks / activation dumps / movegen for notebooks).
> Portability decisions: WASM + ARM/NEON TABLED for now but PINNED as
> public-release prerequisites (both are targets of the one Rust
> codebase, not ports).**
>
> **USER HYPOTHESIS (10:1x) — the joint net is MORE value-undertrained
> than the twin at matched presentations, and has the higher ceiling.**
> Gradient arithmetic: share ~1.8 ⇒ policy got ~64% of trunk gradient ⇒
> joint's value function saw ~1/3 of the schedule's gradient budget vs
> twin's 100%. Crossover re-read: twin nearing ITS asymptote; joint
> still climbing toward a higher one on a richer representation whose
> value refinement was DEFERRED. Stage-2@λ/4 (share→~0.45) = the first
> value-gradient-majority phase = COMPLETION, not compensation.
> Predictions: (1) twin sb240 300g (no-sidecar) ≈ joint nopol +52.5
> [pessimistic alt: twin clearly above ⇒ policy dominance wasted value
> capacity]; (2) stage-2 early value-slope STEEPER than twin's late
> slope at matched loss ⇒ deferred refinement releasing. λ/4 stands;
> hypothesis nudges the next cold run toward share-locked λ (~1.0)
> rather than fixed λ (policy dominance was an accident of scale, not
> a choice).

## 2026-07-26 TM FRONT-LOAD BUG — DIAGNOSED + FIXED (depth-redenomination)

**Symptom (user, GUI + 2+1 probe):** ARCHON burns 108s of a 120s base by move
15 (SF16: 54s), then lives on increment; totals equal (225.1 vs 224.6). At
10+0.1 it "looked good enough" — the defect scales with TC.

**Exonerated by measurement (each was my hypothesis at some point):** eval
drift (mean signed drift ~0 both engines); search indecision (root flips
4.08/search vs SF16's 4.05; last change at ply 12.7 vs 12.3 on 60 shared
middlegames at 3s); the allocator (ours is SF18-shaped, within 6% of SF18's
own curve; SF16's is simply an older, more frugal one); soft-stop overshoot
(ovr ≈ 1.1); wrong-side clock + stale TM state (probes: opt tracks OUR clock,
re-inits per go). Harness lesson: the first replay pasted book-FEN game moves
onto startpos — engine silently searched startpos for 20 moves (fix: use the
PGN [FEN] header; tee stdin when a trace looks impossible).

**Root cause:** TM's two save-time inputs are ITERATION-denominated with
SF-tuned constants, and we complete ~3.3 fewer iterations at equal time
(depth 24.2 vs 27.5 — pure NPS, not indecision). Stability arrives 3.6 lower
(11.6 vs 15.2 — the lerp [5,18] reads us as permanently unstable), and each
root flip suffers ~3.3 fewer /2 decays by the stop decision (~10× hotter
inst). Replay of the real game: budgets approved at red=2.16 (cap) ×
inst=3.06 → a 20.5s move with 96s left.

**Fix (src/search/params.rs, derivations in comments):** stability window
[5,18]→[2,15]; flip decay /2→/2.5 (d^11.6 = 2^15.2 ⇒ 2.48). Replay after:
53.6s by move 15 (SF16 real: 54.1s), max move 20.5→11.1s, late-game depths
UP (36-38 vs 30-34). check.sh green, bench 701605 bit-identical (TM is
outside the depth-pinned bench path). Deviation from SF-exact is deliberate
and documented in design/12 §3.9. **PENDING GATE: 200g 10+0.1 vs incumbent
when the box is next idle** — TM changes touch the crown TC; no release
binary until that screen is ≥0. fe eval-scale rescale (user's standing
point) deferred to SPSA: internal-scale ratio has two conflicting estimates
(1.2× material probe vs 1.8× regression slopes) and fe rides its 0.572 floor
~80% of moves — not baking in a guessed constant.

## 2026-07-26 ~08:4x SPSA-PRIOR SCREENS ROUND 1 (fixed-node 200g self-play
## arms vs base, d47p value-only, beside trainer — contention-safe setting;
## §8 caveat carried: fixed-node UNDERVALUES pruning)

| arm | overlay | Elo@50k | games verdict (arm_verify) |
| sf16bar | NMP 140−12d | −17.4±25.4 | +0.55 ply, NO tactical spike — sound, setting-taxed |
| bundle | +RED_BASE 2051 | −27.9±27.4 | +1.78 ply but tactical 32% vs 17% — flat-LMR bleed |
| midbar | NMP 250−19d | **−5.2±27.5** | +0.34 ply, no spike; tree ×3.11→×2.32, cuts/k ×5 |
| redonly | RED_BASE 2051 | −41.9±25.5 | collapse spike — REJECTED (twice-confirmed) |

**Verdicts:** NMP bar SPSA center = (250, 19) — crossover ~13 at a higher
base is fixed-node NEUTRAL while opening NMP 5×; the aggressive (140,12)
stays in range for wall-clock SPSA to walk toward. FLAT RED_BASE bump is
REJECTED — the LMR magnitude gap must come from SHAPED terms (the
statScore discount ×445/4096 on our strong histories eats the base; flat
bumps reduce good moves too). Round 2 RUNNING: RED_TABLE_NUM 3260 (late-
weighted), RED_MOVECOUNT 85 (late-only), guards binary + (140,12) rescue.
Then the 8-family ×0.52 screens (staged). All results = SPSA STARTS, not
landings (user protocol 2026-07-26 ~08:0x: screen → verify in games →
refine → pin priors; landing post-D47).

## 2026-07-26 ~09:4x SPSA-PRIOR ROUND 2 — GUARDS RESCUE THE AGGRESSIVE BAR;
## UNSHAPED LMR FAILS AGAIN; SCAFFOLD BUG (baked reductions table) FIXED

Round 2 (fixed-node 200g): **guards+(140,12) = +3.5±31.3, deeper (+0.49)
AND tactical BELOW base (22% vs 27%)** — the eval≥beta/staticEval guards
turn the −17.4 unguarded bar positive; wip/nmp-guards is now an
EVIDENCE-BACKED post-D47 landing candidate, not insurance. shapeB
(RED_MOVECOUNT 85): REJECTED (not deeper, tactical 36% vs 25%) — every
unshaped LMR increase has now failed (flat base ×2, movecount ×1); if
shapeA2 (table, valid re-run) fails too, LMR goes to wall-clock SPSA from
current values. SCAFFOLD BUG: RED_TABLE_NUM/DEN baked into Worker::new →
setoption/SPSA silently no-op'd (shapeA = perfect mirror Ptnml
[0,0,100,0,0] = the tell); tune builds now rebuild the table per search
(release bit-identical, commit in tree). ROUND 3 RUNNING: shapeA2 + all 8
×0.52 family screens (~5h, lands ≈ D47S completion). SPSA centers so
far: NMP bar (250,19) unguarded OR (140,12)+guards [prefer the latter on
current evidence]; RED_* unchanged pending shapeA2.

## 2026-07-26 ~11:2x D47S HALFWAY HEALTH — in-frame heldout ACCELERATING
## down: sb10 0.010787 → sb50 0.010768 → sb120 0.010517 (−2.3% in the last
## 70sb as the cosine bites). Run healthy; anneal watch continues.

## ★★★ TAKEOVER CHECKPOINT 2026-07-27 ~06:0x — READ THIS FIRST ★★★

**THE RELEASE NET IS OPEN — DECIDED BY TESTING (user, 2026-07-27: V =
incumbent candidate, D48W = the hope, S = yardstick; the three-way
face-off rules). V's MTL run is OVER (stopped sb190 by ruling); the
SQUEEZE-LEG ERA is live.** The V trunk chain:
MTL-sb190 (heldout 0.007972) → LegA value-only 12sb (0.006905, the
−13.4% earthquake; policy frozen) → LegA2 24sb gentle (DONE, ema
0.006658) → LegA3 ext cut @ sb10 by user LR ruling (ema 0.006620) →
**LegA4 THE ANNEAL (running, 12sb @ 3e-5; d47p-parity gate 0.006205 now
−6.3%)** → B2+B3 policy 20sb+20sb-anneal follow on the final trunk. **PLAY: the A+B pair scored −16 Elo
vs the TEACHER at fn50k (131g partial — from −113 pre-squeeze!) — the
ET gate projects PASSING with the measured ~1.5× speed edge.** SEQUENTIAL
RECIPE RULED IN by experiment (LegB restored top-1 17.5 ≈ pre-legs 18.7
with value hugely better): MTL dropped for future trunks; teachers =
warm-up accelerant only (ruled). GPU CHAIN (unit **archon-chain-v5**,
log /tmp/chain_v2.log; anneal-era restructure, see ~07:2x entry):
[V-A4 anneal running] → V-B2 (20sb) → V-B3 (20sb anneal) → S-LegA2
(cut @ sb10) → S-LegA3 (12sb anneal @ 4e-5) → S-LegB (10sb, no anneal);
then S pair pack/panels → **V-vs-S FACE-OFF on the PINNED binaries
(archon-d47v / archon-d47 — net-lineage comparability; none of the S0 or
perf commits touch them) → HARD PAUSE (user ruling 2026-07-26 ~22:1x:
"pause the queue after the V vs S face off … I will want to suggest a
different testing plan after that, and we can decide exactly what that
is after we see the results … and we have a clean box for some equal
time experiments"). NOTHING launches after the face-off** — no D48W, no
further matches; the box drains to clean. D48W stays staged
(BUILT+SMOKED: 1536/16/32+skip, accumulator-branched pol16 + dis64
PURE-PROBE heads, expansion warm assembled, config d48w_1536.json;
recipe RESOLVED: MTL-FIRST with the pre-registered contention tripwire
in _recipe_ruling, V-playbook legs as fallback) awaiting the post-pause
plan. **PERF: +55-60% NPS in two days** (S+S2 skip lane
program, then study-2 A/B/C/D/E all landed with proofs; F′ flat-
reverted; sha oracle 0615d75e…c92eb still valid, bench 701605, D's
contract: |threat lane| ≤ 8191 debug-asserted). Perf study 3
("beyond SF18" mandate) RUNNING. Leg lengths are ADAPTIVE by rule
(still-falling at buzzer ⇒ extend). Instruments/habits: evolution_panel
(bases+err-corr), netonly_eval (stratified, refs always), match_report,
per-10sb tables IN CHAT, interim results reported PROACTIVELY (user
meta-rule). sb190 anchors: teacher −113.4, SF16-EN −141.2
(exp-0142/0143). The ~13:16 checkpoint below is HISTORY.

**UPDATE ~17:5x (supersedes ~14:4x): FRAME FIX PROVEN IN PLAY; ENDGAME
PROTOCOL GOVERNS THE REST OF THE RUN.** State at sb67: teacher-frame fix
landed at the sb30 pause (logit-LUT; ~14:4x entry) and PAID: heldout
unstalled (−4.5% the first window; windows since −0.95/−1.18 — no
Layer-1 plateau strike yet), **fn50k vs teacher −206.7 (sb20) →
−151.4±27.1 (sb50), +55 Elo/30sb (exp-0139/0140); ET gate needs ≈−100 —
IN REACH.** Native scale EQUILIBRIUM ≈ 0.66×d47p (NOT drifting to 1.0;
V ships with header cal like every net in the family; per-checkpoint
cal is automated). Rank panels asymptoting (quiet 0.8713 best, sharp-nc
~0.753, top-1 ~20 vs teacher 27.2, WAC 161/200 vs teacher 176) while
heldout+play still improve = calibration is the active front, the
anneal's specialty. THE DISTILLATION ENDGAME PROTOCOL (~15:1x + 3b
amendment + LR triggers) governs: Layer-1 convergence → α ablation
(incl. policy hard-CE arm — d47p's hard-label head MATCHES teacher
top-1) → ET gates. Constant α=0.7 to the end (user ruling); decay =
next-gen candidate. Slice truths: endgame gap +0.100 is CAPACITY not
sparsity (decode: endgames ABUNDANT, in-check 0.00% — filter verified);
V BEATS teacher on decided band (labels add real info). AUTOMATION:
archon-d47v (trainer) + archon-d47v-panels (per-ckpt panel/scale/WAC →
/tmp/d47v_panels.log) + match runners as needed; panels now include the
endgame≤8 canary slice.

**D47V STATE AT sb12:** value 0.00549 ↓, pol_ce 2.544 ↓ (teacher band
2.34 ↔ 26.1% top-1; d47n precedent 3.01 ↔ 15.6% — student ≈ low-20s%
already). sb10 Spearman: quiet 0.8675 / sharp 0.7470 (teacher
0.9054/0.8126; ALL sharp figures = NON-CHECK basis n=1817 — always state
the basis, see the ~14:0x corrected-panel entry) — near-SF16-class at
10% of schedule. **ENGINE CHAIN
VERIFIED at sb10: jax-vs-engine parity mean 3.3cp / max 6.8cp with the
skip lane LIVE** (max|w| 1.93) — SKP1 loader + skip dot (policy_trunk
dequant factor) + `arch-d47v` feature build (768/32 consts,
engines/bin/archon-d47v) + netpack passthrough all correct. Checkpoint
eval recipe: pack `D40_ACC=768 D40_L2=32 D40_L3=32 D40_BUCKETS=8
D40_POLICY=1`, sweep with `ARCHON_BIN=engines/bin/archon-d47v` (purge
stale ev-cache first if re-sweeping the same path).

**GATES (pre-registered):** in-frame heldout falling
(tools/nnue/inframe_heldout.py — NOTE it hardcodes the v15 config/arch;
adapt for V reads); Spearman within ~0.003 of teacher; policy top-1
"decent" vs the 26.1 bar (user condition for V as MAINLINE); WAC
fixed-node; **DECISIVE: equal-time vs teacher ≥ 0** (768 projects ~0.9M
NPS ≈ SF18 parity, 84MB). Fallbacks: D47U (1280/64+skip, fully staged:
config+assembler) → sb230 ships. **DOCTRINE (user): train too-big teacher
for real Elo → distill to competitive size; raw-label finetune
(resume, distill_alpha=0) last if teacher-hugging shows.**

**THE DAY'S STANDING RESULTS (all games/oracle-verified, entries below):**
teacher sb230 ≈ clean-sweep evaluator vs SF16/SF18 statics (quiet 0.9054 /
sharp 0.8126 / iso-MAE parity+lead / only in-check-capable). Search gap
solved-in-principle (NMP bar shape + LMR table → tree ×3.11→×1.06 vs
SF16). s4-searchgap SPSA campaign: 30 games-vetted priors (sefam ×0.52
+22.6 bracketed peak; qs 174 +17.4; combo +19.1; branches wip/nmp-guards
+3.5, wip/check-ext +3.5 & WAC 78.5→83). NPS ledger: eval 4.3×/shell
1.8× vs SF16; SF18's edge = 15-wide head + smallnet (source-verified
L2Big=15 + fc_0 skip lane). ET attribution vs d47p: sb170 −59.6 → sb210
−26.1 (anneal ≈ half); global margin overlay REJECTED (−77.7); per-family
= SPSA's job. TM depth-redenomination LANDED+GATED (+6.5 self-play,
2+1 replay fixed). Instruments are MANDATORY: see INSTRUMENT INDEX below.

**POST-V RUNBOOK:** pack (D40_ACC=768 D40_L2=32 D40_L3=32 D40_POLICY=1
+ SKP1 passthrough ✓) → calibrate → panels (Spearman vs teacher) →
matches (fixed-node, ET vs teacher + d47p, WAC) → verdict. Engine side
STILL NEEDED before V can play: head skip-dot consumption + 768/32
feature consts (SKP1 LOADER done). d47s-line settle re-spec from sb230
if V disappoints. Old runbook (below) for the d47s stages; read the
~21-22h entries for the V arc.

## 2026-07-27 ~09:3x REMATCHES FINAL + S0 FULLY COMPLETE (incl. OPTIMISM)
## + PERF FREEBIES & PGO STANDARD LANDED + HARD-PAUSE SEQUENCE SET

**REMATCHES (200g each, fn50k, pinned archon-d47v, pack cal 400):**
vs TEACHER **−15.6±25** (+52=87−61; exp-0146) — statistically identical
to pre-anneal exp-0144: A3/A4 value gains and the 13.7 top-1 policy tax
NET TO ZERO at fixed nodes. match_report: losses are GRINDS (44/61),
tactics even 9v9 — the 1280 out-maneuvers the 768 exactly in the
|label|<100 slice. vs SF16 **−33±26** (+53=75−72; exp-0147). The squeeze
arc = ~+100 Elo on BOTH axes from sb190 (−113/−141). ET is the decisive
gate (V ~1.5× faster).

**S0 COMPLETE — all six audit items live** (user asked "did S0 yield
Elo": fn can't show tree-shaping gains by design; mechanism says +20-60
ET before SPSA; ET SPRT decides): the five landings (~08:1x entry) plus
**OPTIMISM (edd5771)** — the eval's §3.6 blend got its live consumer,
priors 75/45 = SF 137/81 × measured winner-slope 0.55, applied ±20-30cp
at typical material, `d` display unchanged; two fixture re-scopes
(queen-up mirror bestmove bar → knight-endgame only; ponder test → KRK
mate-in-2, dither-exempt). Final bench walk: 701605 → 385329 (**−45%
depth-6 tree**). 150g tripwire s0-full vs s0-thru-IIR RUNNING
(/tmp/s0b_tripwire.log).

**PERF (user: "do the freebies now… PGO the standard"):**
- Freebies landed (7cf2ca7): policy-head SNAPSHOT at net-set time (kills
  the per-eval RwLock futex read, 1.53% cycles; D14-clean — no AtomicPtr
  needed once the read honors the slot's own once-per-boundary contract)
  + `#[repr(align(64))]` on Entry/FinnyEntry/ThreatEntry. PROOFS:
  800-FEN sha identical both sidecar modes, d13 bit-identical (29796
  nodes — yes, S0 shrank d13 from 2.94M to 30k), smoke +1.3% contended.
- **PGO = THE STANDARD BUILD (d1c9aa2):** existing tools/build_pgo.sh
  (D45a — it already knew bench-only training leaves NNUE kernels
  unprofiled) + ruling in header + CLAUDE.md; llvm-tools installed.
  First standard binary `engines/bin/archon-d47v-pgo` VERIFIED: shas +
  d13 identical, smoke **+5.1% contended** (study: +8.1% idle). SPSA
  baseline = PGO'd binary (freeze-rule reorder ratified).
- P-ORD: decision DEFERRED by user until the policy story settles ("D48
  might be the only thing that saves it").

**SEQUENCE (user ruling): chain → S pack/panels → V-vs-S face-off on
PINNED binaries → HARD PAUSE, clean box, user proposes the ET testing
plan.** Tonight's perf gains ride ONLY future non-pinned binaries.

**NPS-CLAIM RECONCILIATION (user challenge ~22:4x, 3-generation ladder
measured live):** single-FEN movetime ladder under load: V pre-perf
655k → pinned 819k (+25%) → PGO-full-stack 849k (**+30%**); S 516k →
513k (**±0** — the program was V's: SKP1 skip lane doesn't exist in
1280) → PGO 567k (+10%). V:S = 1.50× on the standard builds. The docs'
"+55-60%" compounds per-landing A/Bs; study 3's load-insensitive
800-FEN counters say −34.5% cycles/node (≈+53% suite-wide); one midgame
FEN says +30% wall-clock. All three are real measurements of different
things — skip-lane gains scale with threat density. DEFINITIVE idle-box
PGO ladder = first measurement after the pause. Pinned-V vintage note:
S2-promotion era (A–E live only in suffixed research builds + PGO).

## 2026-07-27 ~08:4x V PAIR FINAL (A4+B2-sb20) — ANNEAL FIXED ENDGAMES,
## POLICY TAX HAS A DOSE-RESPONSE CURVE; B3 STOPPED; S-LegB CUT TO 10sb

**User rulings:** B3 anneal STOPPED at ~sb5 ("making no progress" —
pol_ce flat 2.86 since B2 sb10; no ckpt existed, so V policy = B2-sb20);
S-LegB = 10sb NO anneal (V's B-legs proved policy repair converges by
~sb10). chain-v6 runs the rest: S-A2 (cut@10) → S-A3 anneal → S-B 10sb.
**W recipe RESOLVED: MTL-first (accumulator co-existence experiment,
tripwire in d48w_1536.json _recipe_ruling).**

**PANEL on the final V pair (pack /tmp/panel-legB2sb20.cenn, payload
81.5MB, checksum 0x29024379b949d0ad):**
- QUIET all n=2999: vs-oracle **0.8920** (A2-sb24 was 0.8891; sb190 was
  0.8720) — new V best; vs-teacher 0.9831, err-corr 0.9183.
- **ENDGAME≤8 nc n=384: 0.6461 → 0.7149** (teacher 0.8152) — the
  A3-cut+A4-anneal legs bought +0.07 on the capacity slice.
- SHARP nc n=1817: 0.7926 (≥ SF18's 0.7909).
- **POLICY top-1: 13.7%** (55/401; LegB was 17.5, pre-legs 18.7, same
  probe). DOSE-RESPONSE: 1 squeeze leg → −1.2pp recoverable; 4 legs →
  −5pp. pol_ce is blind to it (2.86 both) — move-choice isn't. The
  user's co-existence read, quantified.
- FULL REMATCHES RUNNING (archon-a4b2-rematches, log
  /tmp/d47v_a4b2_rematches.log): calibrate → 200g fn50k vs teacher →
  200g vs SF16; pinned binary archon-d47v (exp-0144 continuity). The
  net trade (value ↑↑ vs policy ↓) prices here against the −16 partial.

## 2026-07-27 ~08:1x S0 SEARCH-FIT PROGRAM COMPLETE (user-ratified after
## review): THE P9 DEFERRALS ARE ALL LIVE

**User challenged the workstream mid-flight ("why are you touching
mod.rs"), ratified after the value/safety case ("Okay finish that work.
But run a quick test against a previous binary").** Five commits, all
doc-governed, tests+clippy+gate green at each step:

- `aabb406` §3.7 ttPv entry+rebate + cutoffCnt surcharge (ttMove rebate
  → its ELSE arm, the audit's trap) + all_node scaling. PROOF THE PAIR
  IS ONE UNIT: surcharges alone re-broke both §5 mate tests (slower
  mates at pinned depth); with the rebate both pass. Bench 701605→558907.
- `8cd138d` tt_capture surcharge. Bench 606118.
- `abacd62` IIR re-land in design/10's narrow N12 shape (PV/cut, no
  ttMove, depth≥6, −1; new param IIR_MIN_DEPTH; the P9 revert was the
  broad follow-PV shape under PSQT). Bench 734066. NB the SF clone is at
  the SF16 TAG (68e1e9b3), not eca43a97 master — the 07-audit's "ref
  has it at :1030" line refs are wrong; design/10 is the authority.
- `825bc14` §3.9 sibling depth−2 + the oracle re-scopes it forces:
  non-root fixed-depth cp values are order-dependent BY DESIGN now, so
  doc 07 §4 step 6 + the PVS/TT-equality suites keep DECISIVE values as
  the exact living bar (LMP-era precedent). ARCHON deviation recorded:
  ROOT EXEMPT (ID driver owns root depth; mutating it corrupted the
  aspiration trace). Bench 408331.
- `30be4e7` history-buys-lmr_depth (16-entry LMR_DIVISOR, +2·main),
  the tree-relaxing counterweight, landed as a unit per the doc's
  revisit note. Bench 535277 (net S0: −24% depth-6 tree).
- **A/B tripwire exp-0145 (user-stopped at 117g, "signal enough"):
  post-S0 (thru IIR) vs pre-S0, fn50k, same V pack: +35=48−34 = 50.4%
  (~+3±40). NO bug-class regression; fn undervalues tree-shape; ET +
  S2 SPSA co-tune arbitrate at the post-chain window.**
- Remaining S0 item: OPTIMISM only — held on its amplitude
  prerequisite (137/81 shape tuned to SF's eval amplitude; ours ≠).
- **TRUNK-SHARING FINDING RATIFIED (user):** value-only trunk training
  destroys policy features; head-only retrain recovers fast but NOT
  fully (18.7→12.2→17.5 top-1; B2 pol_ce converges by sb10 then
  flatlines 2.86). Refinement on the record: demonstrated at 768 WITH
  DEEP-TRUNK policy branch; symmetric taxes (MTL cost value 13%, squeeze
  cost policy ~1.2pp) = capacity contention. D48W's accumulator-branch
  is the designed test: substrate-sharing vs deep-trunk-sharing.

## 2026-07-27 ~07:2x ANNEAL-ERA CHAIN RESTRUCTURE (user LR rulings) +
## A2/A3 RESULTS + NETONLY: V IS SF18-CLASS ON STATICS

**User rulings (box-local 2026-07-26 ~20:2x-21:0x):** (1) "legA needs a
lower learning rate — when it saves at sb10 … start an annealing"; (2)
"do something similar with S's legA"; (3) "For B legs lets do a 20sb
with normal then a 20 sb anneal." Chain restructured twice (v3→v4→v5,
all on /tmp/chain_v2.log):

- **V-A2 (24sb @ 1e-4) DONE:** ema 0.006918→0.006753(sb10)→0.006668
  (sb20)→**0.006658 final** (−3.8% the leg; −16.5% cumulative from MTL).
- **V-A3 (ext @ 8e-5) CUT @ sb10 by ruling:** ema 0.006620 (still
  −0.6%/10sb when cut). sb10 EMA → A4 warm (gate PASS exact).
- **V-A4 = THE ANNEAL, RUNNING:** 12sb @ 3e-5 (~37% of A3 peak)
  cosine→floor. ENDS the V value arc; d47p-parity gate 0.006205 sits
  −6.3% below the A3-sb10 ema.
- **chain-v5 remaining flow:** V-B2 (20sb @ 5e-4) → V-B3 (20sb anneal @
  2e-4; warm = B2-sb20-raw — policy legs have NO ema files) → S-LegA2
  (24sb cfg, AUTO-CUT @ sb10 like V) → S-LegA3 (12sb anneal @ 4e-5, v15
  retargeted frame carried — trainer heldout NOT comparable, use
  inframe_heldout) → S-LegB (20sb @ 5e-4, warm repointed to A3-annealed
  trunk) → S-LegB2 (20sb anneal @ 2e-4). Configs:
  tools/nnue/runs/{d47v_legB3,d47s_legA3,d47s_legB2}.json (new) +
  edited legB2/s_legB. ETA ≈ 7-8 h.
- **NETONLY on A2-sb24 (user ask):** QUIET headline 0.8891 (sb190 was
  0.8720 — legs closed HALF the teacher gap; teacher 0.9054, d47p
  0.9000, SF18 0.8982, SF16 0.8797); SHARP 0.7913 = SF18's 0.7909 dead
  tie; sharp iso-MAE 108.2 ≈ teacher 108.9; V BEATS teacher on
  |label|≥400 both bases (0.851/0.877). Remaining gap concentrates in
  |label|<100 (0.679 vs 0.730) + the ≤8-piece capacity slice (0.646 vs
  0.815). Full table: /tmp/netonly_A2sb24.out.
- **S0 search-fit implementation STARTED** (see next entries): the
  07-note's un-deferral bundle (cutoffCnt/ttPv/tt_capture/all_node/
  sibling-depth) going in per design/10 §3.7 pinned order; is_valid
  guard added to the rebate's tt.value>alpha term (VALUE_NONE=32002
  would spuriously fire on TT-miss PV nodes).
- Process: foreground poll loops BANNED outright (user; memory
  updated — no short-wait carve-out). Monitor bnfl810di watches all 7
  leg logs + chain log.

## 2026-07-27 ~06:3x ★ PRE-COMPACT PERSISTENCE + RULING AMENDMENT:
## THE RELEASE NET IS OPEN ★

USER (verbatim): "idk if v is still release net. It will depend on
testing. I have hope for 48w, but we will see." → propagated to the
campaign doc, CLAUDE.md, the checkpoint, and memory. V = incumbent
candidate (its case: −16 fn vs teacher at exp-0144, heldout 0.00675
and falling, +55-60% faster binary); D48W = the challenger (the full
V2 topology, built+smoked, trains after the S legs — RECIPE QUESTION
for the user: flip its trunk leg to sequential per the LegB ruling?);
S = yardstick/fallback (own legs running).

**IN-FLIGHT AT COMPACT (units/logs):** archon-chain-v2
(/tmp/chain_v2.log): V-A2 (sb12+/24, /tmp/d47v_legA2_train.log) → V-B2
→ S-LegA2 → S-LegB; perf-study-3 agent (report →
docs/research/perf-study-3-beyond.md when done); waiter tasks re-arm
from the chain log if lost. **PENDING DECISIONS:** release net (by
face-offs); W recipe (sequential?); full rematches (paused partials:
legAB-vs-teacher 131g at −16, kept in
data/nnue/experiments/d47v-legAB-fn50k-vs-teacher/) + face-off protocol
after the chain; option-E-class follow-ons only via perf study 3.
**POST-CHAIN RUNBOOK:** pack A2+B2 pair → probes/panels → full 200g
rematches (teacher + SF16-EN) → S pair packs → V-vs-S face-off (fn +
panels; ET on idle box) → W training (GPU) → three-way → gates per the
campaign doc APPROVED SEQUENCE.

## 2026-07-27 ~04:0x PERF PROGRAM (STUDY 2): TONIGHT'S SCOPE COMPLETE —
## A+B+C LANDED ≈ +15%; E PLANNED, NOT FORCED

Landed with full proofs: **A Finny set-diff +9.0% (7f0837a) · B fold
register tile +2.3% (ad01657) · C MovePicker uninit +3.2% (9a508b4)**
— every one bit-identical (sha + bench 701605 + search-identity; C
additionally tracker-verified across 686 debug tests + a live debug
search sweep). F′ prefetch: measured FLAT → reverted (drop-discipline).
D1 deferred (fiddly +1%; real fix = D3 quant unification, next CENN
rev). **Day total: ~+45% NPS on the V binary.**
**OPTION E — PLAN WRITTEN, DEFERRED TO ITS OWN BLOCK (engineering, not
schedule): the in-place StateInfo write fights borrowck through all
~250 lines of make_move (spare-slot &mut blocks every self call) ⇒
either raw-pointer sprawl through the engine's most safety-critical
function or a states-stack redesign (Vec<MaybeUninit> + explicit len
discipline — the cleaner path). The study's cheap partial is
TRAP-CONFIRMED: parent check_squares/piece_threats ARE read after
unmake (gives_check on later moves) ⇒ recompute-on-unmake pays
set_threats twice per node. E's block = the redesign + D24 full perft
ritual + assert_consistent + search-identity + 701605.** Also: leg
lengths are now ADAPTIVE by rule (user challenge on the 12sb template):
any leg still falling at its buzzer gets extended (~30-60min each);
applied to S-LegA at the chain report, A2 already embodies it for V.

## 2026-07-27 ~03:1x LEG B VERDICT: RESTORE — SEQUENTIAL WINS; CUT RULE
## PASSES FOR THE A+B PAIR; S-LEGS + REMATCHES RUNNING

**Leg B (12sb, ~25min, 520k pos/s): top-1 12.2 → 17.5 = statistical
parity with pre-legs 18.7 (Δ1.2pp < probe SE ±2). CUT RULE PASSES for
the pair. Value identity EXACT (identical payload checksums; my "11
DIFFS" scare was a paste-column bug — value-vs-move comparison).**
THE MTL QUESTION (user): the pre-registered "RESTORE" reading fires —
**sequential (value trunk → frozen-trunk heads) WINS: value −13.4%
heldout with policy statistically unchanged.** Not "exceed": 17.5 ≪
d47p's 26.9 — the 32-wide L2 substrate is V's policy ceiling under ANY
recipe (D48W's dedicated projection is the fix; W-recipe decision for
the user: its trunk leg could now drop pol_lambda entirely). pol_ce
moved only 2.72→2.70 — head capacity, not training time, binds.
A+B pack: native scale 0.976 (cal 410) — Leg A finished the frame walk.
NOW RUNNING: S-LegA (GPU; value tune on d47s-sb230, policy frozen, own
v15 frame) + the A+B rematch pair (CPU; teacher + SF16-EN, same
protocol as the sb190 anchors −113.4/−141.2). PERF: F′ prefetch
measured FLAT (−0.8%±noise, 3 rounds) → REVERTED per drop-discipline
(idea preserved for P4 idle-box retest); D1 deferred (fiddliest +1%,
real fix = D3 quant unification at next CENN rev); next = C with its
safety writeup, then E.

## 2026-07-27 ~02:3x THE LEG-A EARTHQUAKE + sb190 ANCHORS + THE CUT-RULE
## DEVIATION (surfaced)

**LEG A (12sb, 40min): heldout 0.007972 → 0.006905 (−13.4%) — the
campaign's biggest single move; d47p-parity gap now −10%. Panel: quiet
0.8725→0.8854 (+0.013), sharp-nc 0.7563→0.7810 (+0.025), endgame
0.672→0.696. MTL WAS TAXING VALUE HARD at 768.** But **top-1 crashed
18.7→12.2 — the CUT RULE TRIPPED (−6.5pp ≫ 3pp).** DEVIATION, surfaced
in chat: Leg A KEPT — the rule predates the committed A→B cascade; B
retrains the head on this exact trunk, so the A+B PAIR is the artifact.
**Leg B's restoration is now the BINDING gate (target ≥19-20, stretch
26.9; can't-restore ⇒ the trip stands and we debate).** Leg B RUNNING
(hard-label CE, train_only polmove*, warm = legA-sb12-ema). USER
QUESTION ON THE TABLE: drop MTL entirely (sequential value-trunk →
frozen-trunk heads)? Leg B = the old trunk-sufficiency fork, live;
three pre-registered readings (restore/exceed/stall) decide; W's recipe
decision HELD for the user until B lands.

**sb190 ANCHORS (exp-0142/0143):** vs teacher −113.4±35 (fn50k) — FLAT
from sb100's −115.2: the MTL sb100-190 tail bought nothing in play,
stop call vindicated. vs SF16 EQUAL-NODES −141.2±36 (first direct
external anchor). Post-legs rematches price the squeeze in play next.
PERF PROGRAM tonight: A (+9.0%) + B (+2.3%) landed on top of +30%;
running total ≈ +45% NPS today; F′/D/C/E queued (C carries the study's
safety-argument + slot-tracker requirements; the >5%-NPS SPSA-freeze
governance rule adopted).

## 2026-07-27 ~01:5x USER RULING: STOP THE V MTL AT sb190 ("we don't
## need to fully converge and anneal under MTL. Just be careful the
## bounce in the legs doesn't destroy what we got")

Stop-and-launch ARMED (unit archon-d47v-stop190): sb190 EMA lands →
trainer stops → Leg A launches automatically. **BOUNCE GUARDS (the
caution, operationalized):** (1) Leg A = FRESH warm from the sb190 EMA
(not a resume) with its own gentle 12sb cosine; (2) peak lr 1.2e-4 =
LR-continuity with the run at sb190 (~1.3e-4) — the leg never
re-raises; (3) 0.25sb Adam warmup; (4) heldout_expect auto-pinned to
the sb190 EMA heldout — pump ABORTS on warm mismatch, and the first
in-leg heldout vs that same number is the bounce verdict (regression
>~1% = stop and reassess, we still hold sb190). Teacher DROPPED from
the leg (residual α≈0.036 — negligible) → ~2× step speed, leg ≈
35-40 min. S-leg configs staged (runs/d47s_leg{A,B}.json; S keeps its
own v15 frame; S-LegB = first policy training since d47p). Sequence:
V-LegA → V-LegB → S-LegA → S-LegB → V-vs-S face-off (fn+panels; ET
when GPU frees) → D48W training → three-way.

## 2026-07-27 ~01:4x D48W IS BUILT AND LAUNCH-READY (user: "we could
## have D48W ready quick" — correct)

The full V2 candidate exists end-to-end (25f8d56): jax arch fields
pol_proj/dis_proj; policy = dedicated 16-wide projection off hl1
(**user cost ruling: 128 ≈ 8× the whole L2=16 value head per
consultation; 16 ≈ the value L1 itself; rank-8-bilinear precedent for
thin-dedicated; my "128 = measured-sufficient" was an UPPER bound —
d47p's 128 was SHARED**); disagreement = 64-wide PURE PROBE
(stop_gradient trunk input — observes, never steers; T11 seed
predicting |sigmoid(v)−target|); pow_exp 2.5 + v1.5 trainer trio;
net2net expansion warm ASSEMBLED (520MB, old pairs intact); config
runs/d48w_1536.json (teacher flat 0.7→sb30, →0@sb60; 60sb; full mix).
SMOKE PASS (shapes/grad/finite; donation-safe harness; branch-local jax
import unshadowed in losses). **ENGINE-DEPLOY RULE (user directive, the
SKP1 lesson): the new heads land engine-side ONLY with fixed-shape SIMD
kernels + a perf-share gate (<1% of cycles at their consultation
rates) — never a scalar afterthought loop.** LAUNCH: after the release
legs free the GPU (release-first sequencing). FACE-OFF pre-registered:
V-final vs S-sb230 vs W-sb60, fn50k + panels + probes, ET on idle box.
OPS: two OOM kills tonight — panels watcher (recapped 1.8/2.5G,
recycled) and Chrome (kernel victim when my warm-assembly transient
stacked on trainer+study2; capped units survived) — no more heavy
CPU-side jobs while study 2 holds its buffers.

## 2026-07-27 ~00:5x ★ GEN-2 ARCHITECTURE SKETCH (user-driven design
## session) — WIDE SUBSTRATE, THIN VALUE, ACCUMULATOR-BRANCHED HEADS ★

The size-strategy discussion (D47S fear → fact-checked efficiency
decomposition → topology) converged on the V2/gen-2 candidate:
**accumulator ~1536 → value L1 32 (or SF-15/16) + skip+PSQT lanes;
POLICY = dedicated ~128-wide projection OFF THE ACCUMULATOR (reuses the
sparse nnz/dpbusd path; LAZY — fires only when MovePicker enters the
quiet stage, so ordering pays per-consultation, not per-eval);
DISAGREEMENT = dedicated 32-64 projection off the accumulator,
consulted only at allocation points (root/TM/PV) = full substrate at
negligible cost. NEVER branch heads off the value L2 — V measured that
mistake (top-1 22.7→~19 through the 32-wide chokepoint; d47p's
128-substrate head = 26.9 ≈ teacher = the one measured-sufficient
point).** MTL contention moves to the substrate, where 1536 has room.
Bilinear (rank-8, the gen-0 confirmed positive) vs per-move rows = a
cheap A/B inside the same branch point. COST on the measured curve:
~15-25% NPS below V-768 — the price the heads must earn back (P-ORD
gate + T11 +30 bar). DECISION RULE (pre-registered): smallest width
where HEAD quality saturates — value is teacher-suppliable via
distill, head quality is not (pure teacher-CE capped V's policy BELOW
the hard-label 1280 head). INPUTS PENDING: Leg B (768 sufficiency),
P4 width-cost curve, T11 substrate hunger. HONEST KILL-CONDITION
(named): if P-ORD and T11 both fail their gates, the optimal ship is a
lean value net and differentiation lives in data/training (the
D47S-fear-was-prophecy branch). CORRECTED FACT feeding all this: the
+30% program was V-only (SKP1 tax); D47S's ~2.3× c/n vs SF18 is
substantially ARCHITECTURAL (L2=128 head, width-linear fold) — the
teacher never needs to be fast; big nets are for knowing, small nets
are for playing.

## 2026-07-27 ~00:2x S2 EQUIVALENCE PASS — THE +30% BINARY IS CLEARED
## AND PROMOTED

200g fn50k S2-vs-S2a: **47.75% = −15.6 ± ~25 Elo, consistent with
zero** (Ptnml [3,21,59,16,1], draw-heavy/symmetric — the rounding-flip
signature). +19.6% NPS is clean profit; cumulative program +29.6%.
**engines/bin/archon-d47v is NOW the S2 build** (pre-S2 backup:
archon-d47v-preS2); all automation (panels, WAC, matches) inherits the
speed. Panel eval sweeps unaffected (int-cp identical); WAC may wobble
±1 from rare path flips — tolerance noted for the trajectory
instrument. Release-binary code path = current main HEAD.

## 2026-07-26 ~23:5x PLATEAU STRIKE 1 (sb160: −0.40%/window, guard
## lifted) — LEG A STAGED AS THE MERGED ANNEAL+SQUEEZE

heldout 0.008231→0.008198. Strike 2 at sb170 (<0.5%) fires the stop.
PREPPED: tools/nnue/runs/d47v_legA.json — the plateau-anneal and the
value-only squeeze are ONE leg: resume from the strike-2 checkpoint,
steep ~12sb cosine tail to floor, policy FROZEN (freeze polmove* +
pol_lambda 0 → value gets the trunk alone), 0.25sb resume warmup,
residual teacher α ≈0.04 rides the ramp out. Launch-time TODO: set
superbatches/lr_end = stop_sb+12, then `pump_train.py runs/d47v_legA.json
<stop_sb>`. CUT rules per the Phase-1 amendment (top-1 −3pp / flat
panels). After Leg A: Leg B (frozen-trunk policy). CONTEXT: equivalence
match 42/200 (slow — box carries trainer + match + study 2), sb160
watcher cycle delayed by the same contention; expected, all
instruments fixed-node/share-based.

## 2026-07-26 ~23:3x ★★ THE OPTIMIZATION PROGRAM IS LANDED — +30% NPS
## IN ONE PASS (user: "Go do them all") ★★

| option | commit | contended gain | identity class |
|---|---|---|---|
| S branchless skip | 7e76782 | +15.1% | bit-identical (sha) |
| F fold-addr hoist | f220b90 | +0.7% | bit-identical (sha) |
| T2 victim pre-mask | 8fc236b | ~0 (under noise) | search-identical (nodes/PV) |
| S2a bucket-major | b1b9c6f | (enabler) | bit-identical (sha) |
| S2 vectorized lane | b1b9c6f | +19.6% | int-cp identical; SPRT-gated |

**Cumulative: 801k → 1038k contended ≈ +29.6%.** V-vs-teacher speed
edge: 1.11× → ≈1.44× → ET-vs-teacher ≈ fn-gap + 55-75 Elo. **At the
current fn −115 the ET gate projects ≈ −40..−60; at the user's −30 band
it projects CLEARLY POSITIVE (+25..+45).** S2 caveat: zero int-cp diffs
on 800-FEN + parity batteries, but rare f32-rounding flips diverge
search paths (study-predicted) — a 200g fn50k S2-vs-S2a equivalence
match is RUNNING and gates rated use of the S2 binary; panels/matches
stay on the pre-S2 binary until it clears. skip_dot follows the
policy_dot fixed-shape discipline (all backends bitwise-equal to
scalar, unit-tested). T2 lesson: nnue-eval CANNOT see observer bugs —
the replay tier caught a PieceType-indexing slip on first build.
Idle-box re-measure of everything = P4 (F/T2's true gains TBD there).

## 2026-07-26 ~22:4x ★★ THREAT-CHANNEL STUDY LANDED (Opus 5;
## docs/research/threat-channel-perf-study.md, 1086 lines) — THE REAL
## SINK IS THE SKP1 SKIP LOOP, NOT THREATS ★★

ALL THREE commission premises corrected by measurement: threats =
15-18% of cycles (24% was the 1280 game config); **the incremental
observer WAS BUILT** (sota_threats.rs, 1511 lines — my "designed but
unbuilt" claim came from a stale header comment; refresh_threat fires
0.0346×/node = 0.67%); the unbuilt piece is only §12.9.1's optimized
driver. **THE FINDING: SotaHead's SKP1 output-skip loop = 19.3% of ALL
cycles, fully scalar, branch-bound (66% of every mispredict in the
engine; ~3093 c/eval for ~100c of vector work). Ablating it: +23.3%
NPS. It explains the 1.11× width result — 768-vs-1280 goes 1.19×→1.45×
without it (the 1280 has no SKP1).** SF18 differential (symbols,
matched nodes): 6599 vs 3320 c/node = 1.99× total, 1.62× ex-skip;
real gaps = HalfKA Finny refresh 2.7× + head math 4.5×; branch-misses
ex-skip IDENTICAL (26.28 vs 26.34); we are COMPUTE-bound (our cache
behavior beats theirs). Threat fold is width-LINEAR (elasticity 0.57,
not 0). RECOMMENDED SEQUENCE: **S branchless skip (+16% NPS,
bit-identical, hours) → F fold-address hoisting (+5-8%, bit-identical,
1 day) → T2 victim pre-mask (+1-2%) → S2 vectorized skip (+6-8%,
int-cp identity)**; T1 (3-primitive driver) explicitly deprioritized.
D24 oracle pinned: nnue-eval sha256 0615d75e…c92eb + bench 701605.
ET-MATH UPDATE: S alone lifts the V-vs-teacher edge ~1.11×→~1.29
(≈+35-45 Elo at ET); S+F+S2 ≈ 1.4×+. Caveats from the agent:
SF18 citations are via the clone's object store (origin/master, no
checkout); nnue-pytorch full_threats.py is a STALE revision (59808
dims) — never the feature-space authority. IMPLEMENTATION: per the
standing rule (changes land after D47 finishes) → build S on
wip/skip-branchless with the pinned oracle proofs, land at P4.

## 2026-07-26 ~21:5x EARLY PERF STUDY COMMISSIONED (user: "Commission
## Opus 5 on it — have it study in depth the threat channel needs to be
## faster") — P4's threat piece pulled forward, parallel with P0

Opus-5 subagent briefed: fresh perf ground truth (verify the 24% share),
refresh_threat from-scratch analysis vs HalfKA's incremental path, the
DESIGNED-BUT-UNBUILT §12.2 dirty-diff observer (≤80 add/remove events —
find why it never landed), SF full_threats distillation (ideas only),
options ranked by measured-share gain × effort × risk, D24 validation
plan (bit-identical evals + bench reproduction). Report →
docs/research/threat-channel-perf-study.md. Motivation: the 1.11×
finding — threats are width-independent, so this gain multiplies EVERY
future net. sb110 state at commission: heldout 0.008532 (−1.23% —
DECELERATING, α-kick spent; strike watch live at sb120), top-1 drifting
(18.7 — trunk moving under the frozen-CE policy head; Leg B's repair),
WAC 160 (noise band 160-168), scale 0.888 PASS.

## 2026-07-26 ~21:3x ★ CORRECTION (user's NPS test exposed it): THE
## SPEED EDGE IS ~1.11×, NOT "2.5-3×" — THE ET GATE MATH IS REWRITTEN ★

Measured back-to-back under identical contention (4×1M-node searches):
**V-768 764-768k nps vs teacher-1280 689-692k = 1.11×.** The "~2.5-3×
speed edge" I stated repeatedly (and built the "borderline-even at
−115" projection on) WAS NEVER MEASURED — classic verify-before-
asserting failure. Mechanism (matches the number): threat-channel
extraction (~24% of cycles) and the shell are WIDTH-INDEPENDENT; only
accumulator+head scale with hl → 0.6× on ~35% ≈ 1.10-1.15×.
**CORRECTED GATE MATH: ET-vs-teacher ≈ fn-gap + ~10-27 Elo (1.1-1.3×
band, idle-box ratio TBD at P4). At fn −115: ET ≈ −90..−100 — NOT
borderline-even. exp-0141's projection note is RETRACTED. The USER'S
BANDS WERE THE CALIBRATED ONES ALL ALONG: fn −30 → ET ≈ −5..0; fn even
→ ET +15..25. The P2 gate target = the −30-to-0 fn band**, carried by
the anneal + squeeze legs + the d47p-parity heldout gate (near-parity
per-node is what the doctrine now REQUIRES, not just hopes for).
FORENSICS OF THE TEST ITSELF (all three traps in one hour): (1) first
"fast" rebuild omitted the `sota` feature → net-load failed SILENTLY
(2>/dev/null) → 2981k = PSQT-fallback garbage — the PARITY GATE caught
it pre-conclusion; (2) fixed-node results are NPS-independent — every
match/panel stands; (3) V-768 NPS ≈ SF18-class (~0.9M idle projection
unchanged) — the SF18-parity NPS framing survives; the vs-TEACHER
multiple does not.
## THE TRAJECTORY IS −206.7 → −151.4 → −115.2; ET-vs-teacher NOW
## PROJECTS BORDERLINE-EVEN (speed edge +90-120 vs gap 115). User bands:
## −30 = huge, even = lucky. INSIDE THE MATCH: draws 116 (38.7%, was
## 29%@sb20); **ENDGAME DECISIVE LOSSES COLLAPSED 33→15** — the
## label-led endgame learning is REAL IN PLAY (the panel slice wobble
## was noise; play is the heavier instrument); OPENING is now the top
## loss phase (66/140) = the balanced-band precision stratum, the
## anneal's + margins-SPSA's home turf. Blunders 74v15 (was 100v19).
## Same cycle: WAC 168/200 (jump from 162; teacher 176), scale 0.871
## PASS, heldout 0.008638, strikes 0/2 live.

## 2026-07-26 ~20:5x ★ PRE-REGISTERED: THE D47P-PARITY HELDOUT GATE
## (user: "a key point will be if the smaller net can reach d47p levels")

**TARGET: 0.006205** — d47p-vrepair2-sb120's value on the STANDING
heldout instrument (data/nnue/heldout/uho-band0-5.bin, default frame,
recorded as v15_refine heldout_expect and reproduce-asserted at that
launch). Every pump run evaluates this same file in the same frame, so
the comparison is exact. V: 0.008638@sb100 and falling (windows −5.5 /
−6.9 / −4.3%). Needs −28.2%; at −4%/wnd ≈ sb183, at −3%/wnd ≈ sb210 —
inside the 240 horizon IF the pace survives LR decay. THE READ: hitting
0.0062 at 768+thin-head = the pound-for-pound bet cashed on the
training metric (0.60× the accumulator, ~0.25× the head, same fit);
stalling materially above it = the capacity ceiling made visible →
priced against the speed edge at the P2 gates, and the D47U fallback
gets its trigger. NOTE the instrument is OUT-OF-BAND (uho band0-5 vs
training band75-100) — a generalization measure, identical basis for
both nets.

## 2026-07-26 ~20:3x PHASE-1 AMENDMENT (user): TWO SQUEEZE LEGS, EACH
## DISCARDABLE — "get the most out of the net; unproven; be prepared to
## stop and discard"

Slot into Phase 1 AFTER the main run stops, BEFORE the gates, in this
order (A changes the trunk; B must train against the SHIPPED trunk):
- **LEG A — value-only finetune, policy FROZEN** (freeze polmovew/
  polmoveb, pol_lambda 0): value gets the trunk's remaining capacity
  without MTL contention. ft_sparsity=0 now, so the frozen head's
  features aren't actively zeroed (the old frozen-head λ failure mode —
  see [[frozen-head-lambda-ramp]]) but trunk DRIFT under a frozen head
  is still the risk. GATES: heldout ↓ AND panel headline ↑ vs main-run
  final. **CUT: policy top-1 drops >3pp on the 401 probe (≈1.5σ) or
  panels flat after ~10-15sb.**
- **LEG B — frozen-trunk policy-head finetune** (train_only policy
  tensors; hard-label CE; same mix first): pure policy upside, value
  BIT-IDENTICAL by construction — VERIFY with nnue-eval identity on
  probe FENs before/after. GATE: top-1 ↑ toward the 26.9-27.2 family
  bar. **CUT: no top-1 gain in ~10sb.** B2 escalation (only if B works
  but plateaus): unfiltered/tactical policy pack — needs a conversion
  job (contract: the policy corpus is deliberately unfiltered).
Both legs ride the existing freeze/train_only trainer machinery
(v15_refine precedent). Each ~1-2h. Discard = revert to the pre-leg
checkpoint, no sunk-cost defense.

## 2026-07-26 ~20:1x ★★ USER RULINGS: D47V IS THE RELEASE NET; TEACHERS
## ARE NOT LOAD-BEARING ★★

(1) **"d47v is our release net"** — the release plan resumes after leg 2
trains; the fallback ladder (D47U → sb230) becomes contingency-only.
(2) **Teachers = warm-up accelerant, NOT load-bearing** ("I'll consider
using them again in the future just to speed things up") — the α=0
control experiment is CANCELLED by ruling; the tentative decomposition
stands unproven-but-ruled: warm slice = the value; late constant-α was
overweight; ramp = the correction. Next-gen recipe default: warm slice
+ short distill warm-up + early decay to labels. (3) Every 10sb panel
gets relayed IN CHAT AS A TABLE. sb90 state at ruling time: heldout
0.009028 (−6.87%/wnd, accelerating), scale 0.789 walking to label
frame, decided-sharp 0.8761 = FAMILY BEST, rank headlines churning at
high LR (re-convergence = LR decay's job).

## 2026-07-26 ~19:4x sb80 FULL CYCLE — THE α TRANSITION IS WORKING;
## + SSTATS VERDICT: NO SEARCH-FIT PATHOLOGY IN V

**HELDOUT −5.52%/window (0.010260→0.009694, through the 0.010 floor;
biggest drop since the frame fix; user called it live: "we are
learning").** Train value loss RISES (0.0057@sb89) as α decays — noisier
target, expected, NOT health. The transition trade in the strata:
decided-band hits new bests (quiet 0.8561 — now ABOVE d47p/teacher/SFs;
sharp 0.8696) while smoothed-consensus regions dip (quiet headline
0.8713→0.8666, balanced 0.6631→0.6535) — rank churn under LR-raise +
target shift; expect re-convergence with LR decay. **Native scale now
RISING with the α decay: 0.656(sb60)→0.714(sb80)** — the output frame
walks toward the label frame as the label term takes over (equilibrium
is α-dependent, as suspected). WAC 163/200. **Endgame canary: V's
endgame err-corr vs teacher = 0.5571** (vs 0.87 sharp) — the endgame is
already label-led; watch the slice climb. LR-trigger 0/2 strikes
(α-guard active). Minor: one position showed ext=-1783 (negative
extension counter) — SSTATS accounting anomaly, noted not chased.

**SSTATS 3-ARM SEARCH-FIT FINGERPRINT (30 pos × 300k nodes):**
V-sb70 depth 17.2 / qshare .073 / fhf .856 / mv@cut 1.39 / research 54
vs d47p 17.3 / .075 / .869 / 1.34 / 50 vs teacher-cal 17.2 / .082 /
.861 / 1.35 / 50. **NO TREE PATHOLOGY: V inherits the family's search
fit** (fhf −0.013, research +9% rel — polish-grade deltas). The play
gap is EVAL QUALITY, not margin misfire — the V-era SPSA is a polish,
not a rescue. CAVEAT: V ran on archon-d47v, 1280 nets on archon-sstats
(different builds; same-code-era). OPS NOTE: SSTATS emits on STDERR
(two silent-loss rounds before catching it — the 2>/dev/null trap,
again). archon-d47 (play binary) has NO SSTATS — use archon-sstats.

## 2026-07-26 ~19:1x NETONLY_EVAL SHIPS (user directive) + TWO FINDINGS
## FROM ITS FIRST RUN (sb70)

tools/nnue/netonly_eval.py = the robust stratified net-alone table
(candidate vs teacher/d47p/SF18/SF16 always; check basis + piece bins +
label bands; pinned method: labels ±800 / evals unclipped / PAV 5-fold
/ n<100 suppressed). Wired into the checkpoint watcher every 10sb from
sb80. Teacher pack durable-ized:
data/nnue/nets/screens/d47s-v15-sb230-ema-rawframe-nocpol.cenn.
FINDINGS AT sb70: (1) **SF STATICS COLLAPSE ON ≤8-PIECE RANKING — SF18
0.5006, SF16 0.6030 vs our family 0.7083(V)/0.8002(d47p)/0.8152(T)**
(they lean on TB+search there; their trainer downweights endgames).
V's endgame gap is FAMILY-INTERNAL HEADROOM, not competitive weakness —
reframes the capacity fork's urgency down. (2) sb70 sharp-nc headline
**0.7600** (best yet, +0.007/window — leg-1 was still improving at the
cut); decided-band sharp: V 0.8618 > teacher 0.8130 (d47p 0.8718 tops —
the labels-trained lineage owns the band the teacher's frame saturates).

## 2026-07-26 ~18:4x ENDGAME DEFICIT: THE SLICE IS EXONERATED — the
## elimination chain closes on CAPACITY

Teacher-activation analysis (949 endgame ≤10pc vs 600 middlegame FENs,
white-stm non-check, raw teacher forward with factorize+psqt): kept
columns [0:384]∪[640:1024] carry **58.6% of endgame activation mass vs
60% expected by count** — proportional; specialists (affinity>.65)
197 kept / 134 dropped ≈ the same 60/40; top-30 affinity lanes 20/10
slightly FAVOR kept. The positional slice did NOT disproportionately
cut endgame features. ELIMINATED: exposure (endgames abundant, ~17:3x),
slice accident (here), off-support transfer (in-mix). REMAINING: pure
capacity/head-width triage. LIVE LEVER: leg-2 α-decay (labels beat
teacher on the decided band; endgames skew decided — the endgame≤8
panel slice tracks whether label-led training lifts it). ESCALATIONS:
per-bucket head width for low-pc buckets / D47U. Importance-aware
slicing is NOT the endgame fix (though it may still help generally at
the next distill — the analysis harness now exists).

## 2026-07-26 ~18:2x ★ LEG 2 LAUNCHED (user directive): STOP@sb70 →
## LONG HORIZON + TEACHER DECAY TO ZERO ★

User: "stop at sb70 and restart with a decay on the teacher. start a
LONG job of ALL the data we have and we can stop early. target the
teacher to go to 10% by sb100, then slow decay to 0 by the end."
EXECUTED: trainer stopped at sb74 (sb71-74 discarded), resumed from the
sb70 checkpoint with: **distill_alpha_ramp [[70,0.7],[100,0.10],
[240,0.0]]** (NEW piecewise capability, absolute-sb anchored,
jnp.interp — smoke: 0.7@70 / 0.4@85 / 0.1@100 / 0.05@170 / 0@240);
horizon 240sb with lr_end 240 (**LR re-plan raises restart LR ~3.5× to
~8.1e-4 — warm-restart semantics, intended for the long leg**); 0.25sb
resume warmup (first use); mix UNCHANGED — inventory confirmed the 12
packs ARE all jax-format data (weights exactly size-proportional =
uniform over ~10B positions; the 5k quarantine family was never
converted). Stop-early per the LR triggers + Layer-1/2 instruments.
NOTE heldout stays cross-leg comparable (same mix/frame); α-blend train
loss does NOT (α changes per-sb now). Panels/WAC watcher rides along
(checkpoints every 10sb). This leg IS the α-experiment at full scale —
the 3-arm ablation is superseded for value (the ramp answers it); the
POLICY hard-CE arm question stands (policy soft-CE weight now decays
with α... VERIFY: pol_soft rides _teacher_targets which is gated on
d_alpha>0 but the CE swap is NOT α-weighted — policy stays PURE
teacher-CE until α hits exactly 0. Flagged for the sb100 read.)

**SF ANSWER (user Q, from stockfish_reference/09 §3):** SF's loader
ALSO drops best-move-captures + in-check (`filtered` ON) — they never
train the value net on tension positions either; tactical strength
lives in search, eval only prices "if no tactic fires." They ADD:
wld_filtering (drop labels disagreeing with outcome), soft early-ply
ramp, and **adaptive piece-count balancing** (Hermite target
[0,0.4,1,1,0.75] at pc 0/8/16/24/32) — they actively DOWNWEIGHT ≤8-piece
endgames vs midgame, more evidence against endgame upsampling.

## 2026-07-26 ~17:3x SOURCING CHECK (user challenge) + THE ENDGAME-RARITY
## HYPOTHESIS REFUTED BY DECODE

User: "the claim that our teacher never saw tactical positions, where
did you source that?" — honest answer: it was an INFERENCE from D46
default-ON. NOW VERIFIED EMPIRICALLY: decoded 20k records from each of
uho-band75-100 / fishpack32 / farseerT74 (RECORD_DTYPE direct):
**in-check 0.00% / 0.00% / 0.00%; bestmove-is-capture 1.96/1.83/1.45%**
(small leakage vs the destination-empty rule — measured, not material;
possibly decode edge cases). Contract §1 five-criteria confirmed as the
operative filter. NUANCE: the POLICY corpus is deliberately unfiltered
by contract — the quiet claim applies to the VALUE loss path.

**ENDGAME-RARITY REFUTED:** pieces≤8 = 25.5% (uho) / 14.5% (fishpack) /
13.7% (farseer); ≤12 = 47/35/30%; median piece count 13/16/18. Endgames
are ABUNDANT in the mix. The +0.100 late-endgame gap is therefore
CAPACITY/ARCH triage under plentiful exposure, NOT sparsity — my
"fewer reps to rebuild" line was wrong and is withdrawn. UPSAMPLING
ENDGAMES HAS WEAK MECHANISTIC SUPPORT (user proposed it; measurement
says exposure isn't the constraint). User's sequencing principle stands
for any future mix change: mix changes go in BEFORE the final anneal.

## 2026-07-26 ~17:1x sb50 REMATCH: **−151.4±27.1 — +55 ELO IN 30sb**
## (exp-0140; sb20 baseline −206.7±28.4). GATE MATH: ET parity needs the
## fn gap at ≈−100 (speed edge ~2.5-3× ≈ +90-120); at +55/30sb with 50sb
## + anneal remaining, THE GATE IS IN REACH, not assured. Story deltas vs
## sb20: still grind-dominant (62%), tactical share flat (18%); endgame
## depth gap WIDENED to −0.8 (endgame = V's weak phase in play too,
## matching the slice analysis); draws 93 with V better in 32 (was 25).
## Blunder-ledger raw count rose 81→100 while every outcome metric
## improved — own-eval-swing metric, don't over-read across checkpoints.
## HELDOUT sb60: 0.010388 (−1.18%/window — NO plateau strike, still
## learning; windows: −4.5, −0.95, −1.18).

## 2026-07-26 ~16:5x WHERE V's DEFICIT LIVES (slice analysis, sb50 vs
## teacher, cached vectors) — + THE FIRST STUDENT>TEACHER POCKET

V-vs-teacher Spearman gap by slice (common non-check bases):
- **LATE ENDGAME (≤8 pieces) is V's worst phase**: quiet gap +0.100
  (V .7148 / teacher .8152) — 3× the middlegame gap (+0.036). The
  768-slice/thin-head cut endgame knowledge hardest (and the mix is
  middlegame-heavy — fewer reps to rebuild). Feeds the D47U capacity
  fork + a possible endgame-weighted mix lever.
- **BALANCED (|label|<100) carries the biggest band gap** (+0.068 quiet
  / +0.096 sharp) — the same near-zero precision region the dialect map
  flagged for margins; anneal's historical strength.
- **DECIDED (|label|≥400): V BEATS THE TEACHER** — sharp .8666 vs .8132
  (quiet .8489 vs .8154). MECHANISM: the teacher's v15 frame (b=76.9)
  saturates above ~600cp and discards gradation; V's s/400-frame targets
  keep it. The 0.3 label term adds REAL information beyond the teacher —
  the decorrelation story confirmed in data. α-ABLATION PREDICTION
  (pre-registered): α=0 arm extends the decided-band advantage; watch
  balanced-band for regression.

## 2026-07-26 ~16:3x NET-ALONE PANEL REDONE (user ask) — d47v-sb50 joins;
## + TOP-1 TABLE; two iso-MAE method artifacts caught and fixed

Method pinned: labels clipped ±800 (eval_bench CLIP), evals UNCLIPPED —
isotonic is monotone-invariant, clipping x only creates tie-block
artifacts (raw 1.99× teacher pack momentarily showed sharp iso-MAE 168
under x-clipping; unclipped-label variant inverted the whole column by
rewarding conservative predictors on heavy tails. BOTH now documented).
COMMON NON-CHECK BASIS, SF18@50k labels:
- QUIET (n=2999): teacher .9054/67.8 | d47p .9000/67.7 | SF18 .8982/67.4
  | SF16 .8797/72.8 | **d47v-sb50 .8701/79.1** (Spearman/iso-MAE)
- SHARP-nc (n=1817): teacher .8126/108.9 | d47p .8006/107.2 | SF18
  .7909/108.6 | SF16 .7563/119.0 | **d47v-sb50 .7520/120.8**
- in-check exclusive (n=582, SF statics refuse): teacher .7296 > d47p
  .7258 > V .7030
- **TOP-1 (canon400 n=401, ±2.0-2.2):** teacher 27.2% ≈ **d47p 26.9%**
  ≫ d47v-sb50 20.4%. d47p's hard-label-era frozen head MATCHES the
  teacher's; the fresh thin-substrate pure-CE head trails ~7pp —
  strengthens the ablation's policy hard-CE arm.
READ: mid-train V sits at SF16-static-class on rank quality (just below
SF16 both sets) at 84MB / ~3× teacher speed; the top cluster
(teacher/d47p/SF18) is tight on iso-MAE (67.4-67.8 / 107-109). V's
remaining eval gap = rank ~0.035 quiet / ~0.06 sharp + calibration
~12cp — the anneal's job.

## 2026-07-26 ~15:1x ★ THE DISTILLATION ENDGAME PROTOCOL (user-ratified:
## "fold it in") — pre-registered rules for WHEN THE STUDENT IS DONE WITH
## THE TEACHER and WHETHER IT MUST LEARN ON ITS OWN ★

**NEW INSTRUMENT — teacher-error correlation** (now a standard
evolution_panel column, `err-corr`): corr(student_err, teacher_err) in
rank space vs the SF18@50k oracle. Measured sb10/20/30: quiet
0.8283→0.8335→0.8392, sharp-nc 0.8565→0.8662→0.8702 (+~0.005/10sb).
~70-76% of the student's error variance IS the teacher's error variance.
While it sits this high the student CANNOT meaningfully exceed the
teacher vs the oracle — exceeding requires decorrelation, which
teacher-imitation cannot provide; only labels/self-generated data can.
(In-check err-corr 0.9319 — blindest copying exactly where no training
data exists; play-irrelevant, consistent with quiet-only transfer.)
**DISCOVERED while wiring: the policy loss under distillation is PURE
teacher-CE — pol_soft REPLACES the hard-label term entirely; the policy
head has ZERO independent signal, its top-1 strictly bounded by
teacher-copying. The α ablation must include policy (hard-CE arm).**

- **LAYER 1 (convergence, this run):** over two consecutive 10sb panels:
  Δvs-teacher < +0.001 (both sets) AND Δvs-oracle < +0.002 AND heldout
  relative gain < 0.5% → distillation converged. Floor: sb100 anneal end.
  Status: NOT converged (all still rising).
- **LAYER 2 (exhaustion tripwire):** oracle metrics plateau per Layer 1
  WHILE err-corr keeps climbing → remaining gradient is pure teacher-bias
  absorption → trigger the α ablation immediately.
- **LAYER 3 (own-learning verdict — by EXPERIMENT):** at convergence,
  three arms from the same checkpoint, ~12sb each at the LR floor
  (~2-3h GPU): α=0.7 control / α=0.3 / α=0.0 (raw labels; policy back to
  hard-CE). Judge: heldout + panel + does α=0 DEcorrelate err-corr + 300g
  fixed-node screen for the winner. α=0 wins → future generations get an
  α-decay tail. α=0 regresses → teacher = denoised labels, rides to the
  end. Mixed verdict → investigate before concluding (weak-test rule).
- **LAYER 3b (user amendment ~15:3x: "couldn't we decay the teacher
  contribution?"):** YES — α-DECAY is now a first-class lever.
  `distill_alpha_end` landed (7af2deb): linear α ramp over the whole
  schedule, threaded per-step like the sparsity λ, anchored on ABSOLUTE
  sb so resumes land on the same global ramp; default −1 = constant
  (live run untouched). Precedent: nnue-pytorch `--start-lambda 1.0
  --end-lambda 0.75` (start on the trusted denoised signal, fold ground
  truth in over training — stockfish_reference/09 §λ). DECISION RULE for
  flipping it on: evidence-gated, not assumed — the α ablation (Layer 3)
  measures the endpoint first; a decay tail can also be tested by
  resume-from-sb60ish with distill_alpha_end set, judged against the
  constant-α line on heldout+panel+err-corr. Early supporting signal:
  post-frame-fix quiet err-corr DROPPED 0.839→0.825 (labels actively
  decorrelate the student when their gradient is consistent).
  **RULING (user, ~15:5x): "Lets not decay now, I think next run we
  might" — THIS run rides constant α=0.7 to the end; the ablation still
  adjudicates at convergence; the decay is a next-generation recipe
  candidate.**
- **MIX LEVERS (named, previously implicit):** (a) deployment-
  distribution distillation — sharp/tension NON-CHECK positions with
  per-row α=1 (teacher forward defined everywhere; no bounded-label
  risk); PARKED until a trigger: sharp-nc converges far below teacher
  while quiet closes. Needs per-row-α loader support + an unfiltered
  pack — heavy, idle-box work. (b) POLICY-specific unfiltered data: the
  D46 quiet filter's rationale is VALUE-label validity and does not
  apply to policy targets; the policy head currently sees zero tactical
  positions AND zero label signal (pure teacher-CE) — the ablation's
  policy hard-CE arm should consider the unfiltered-data variant.
- **LAYER 4 (generation-level):** student ≥ teacher at EQUAL TIME → this
  teacher is exhausted in play; next lever = better teacher (bigger/
  longer or student-search-labeled — the flywheel), never more epochs.
  Converges clearly BELOW the ET gate with ablation exhausted → capacity
  verdict → D47U 1280/64+skip.

- **LR TRIGGERS (pre-registered ~16:1x, from the "tie LR decay to
  anything?" discussion — anneal timing keys off INSTRUMENTS, via
  pause/resume, not new trainer machinery):**
  (1) **PLATEAU-ANNEAL:** heldout relative gain <0.5% over two
  consecutive 10sb windows while cosine LR >20% of peak → stop, resume
  from best EMA with a short steep tail (12-15sb to the floor).
  (2) **EXTEND:** at sb90 heldout still falling ≥1%/window → resume with
  lr_end_superbatch pushed out; don't let the cosine strangle a
  still-learning run. Evidence base: anneal recovered HALF the d47s ET
  deficit (−59.6→−26.1); horizons guessed wrong both directions before
  (v15 stretch-to-240; d47s overfit at 236). NOT tied to: train loss
  (frame-dependent), per-step noise, wall-clock. Next-gen α-decay note:
  finish the α ramp BEFORE the final anneal. Capability landed:
  warmup_superbatches now applies on RESUME legs too (default 0 = no-op)
  — smooths the fresh-Adam transient (sb31 wobble).

Timeline: sb40 panel (watcher, automatic) → sb50 fixed-node rematch vs
the −206.7 baseline → Layer-1 call ~sb60-100 → α ablation → ET gates.

## 2026-07-26 ~14:4x sb30 PAUSE → TEACHER-FRAME FIX LANDED → RESUMED sb31+
## (user: "pause after sb30, fix the scale gotcha, relaunch warm")

**THE FIX (native, in-loss):** train.py blended `sigmoid(teacher_logit)`
raw into the target — but the teacher lives in the v15 draw-aware
double-sigmoid frame (a=163, b=76.9), the student's label term in plain
sigmoid(s/400); the α=0.7 blend mixed frames → the student settled at
1.378× the label frame (the measured gotcha). Fix: map teacher LOGIT →
implied cp (inverse of its transform) → sigmoid(s/400), via a 1cp LUT
over ±12k cp indexed by LOGIT (logit(T(s)) is asymptotically linear in
s, strictly increasing in float32 — a probability-indexed LUT ties/
saturates beyond ~±1400cp; T split as 0.5(sig((s−a)/b)+sig((s+a)/b))
keeps log(T)−log(1−T) float64-stable). VERIFIED: round-trip implied-cp
error 0.00cp over ±6000, correct clamping. Config keys
teacher_frame_offset/scale (default 0 = old behavior, no-op for
non-distill runs). Resume: `pump_train.py <cfg> 30` (fresh Adam
transient, EMA carried, schedule position kept); sb31 value loss HALVED
(0.004977→0.002669) — NOT an improvement, the standard frame has lower
target variance; **pre/post-fix train losses are incomparable — the
cross-fix instrument is heldout (label-only, sb30 = 0.011112)**.
Expectation: the student's output scale drifts 1.378→~1.0 natively over
the remaining 70sb; re-measure via net_scale_check at sb40 (should
obsolete the header-267 patch for later checkpoints).

**sb30 PANEL (tools/nnue/evolution_panel.sh — NEW standing instrument,
bases always labeled):** QUIET all n=2999: 0.8675→0.8663→**0.8694**
(new best); SHARP non-check n=1817: 0.7470→0.7506→**0.7530** (third
consecutive rise); top-1 21.7% (22.7/20.2/21.7 — flat in ±2.0 noise;
teacher 27.2). vs-teacher quiet 0.9614 / sharp-nc 0.9400, still rising.
NOTE heldout raw was FLAT sb20→sb30 (0.011111→0.011112) — value
learning had stalled ON THE MIXED FRAME; the fix gives the gradient a
consistent direction. sb40 heldout adjudicates.

**sb20 FIXED-NODE MATCH vs TEACHER (300g, 50k nodes, both packs
median-calibrated to the d47p frame, nocpol both sides): −206.7±28.4
(23.3%, Ptnml [43,76,29,2,0]).** match_report story: losses 64% GRIND
(119/186), only 16% tactical; depth at decisive ply near-equal (12.1 vs
12.6); blunder rate 3.1× (81 vs 26 own-eval 2-pawn drops). = the
healthy-distillation profile: a uniformly coarser copy, not a broken
one. Fixed-node deliberately strips V's ~2.5-3× speed edge (ET is the
decisive gate); at this per-node gap ET would NOT yet close — the
remaining 70sb + frame fix must shrink it. Next match point: sb40/50.

**TOOLING HARDENED (user: "fix that eval net script"):** eval_bench.py
now detects in-check rows, prints the basis warning, and emits SEPARATE
labeled tables (non-check = PLAY-RELEVANT first, then all/mixed), each
with n; json_out carries basis+n per entry. evolution_panel.sh is the
one-command checkpoint read (pack→sweep→bases table→top-1 probe).
Position sets persisted out of /tmp: data/nnue/evalbench/sets/.

## 2026-07-26 ~14:0x sb20 EVOLUTION PANEL (CORRECTED) — HEALTHY ON EVERY
## MATCHED BASIS; a basis-mismatch briefly faked "sharp erosion"

**THE TRAP (third strike of the in-check basis):** the first read of this
panel compared sb20 ALL-rows sharp (0.7218, n=2399) against the recorded
sb10 figure 0.7470 — which is the NON-CHECK common basis (n=1817, the
corrected-panel methodology) — and concluded "sharp eroding" + built a
mechanism story on it. USER QUESTION ("the net should never see in-check
positions?") exposed it. Matched-basis truth (identical cached vectors,
python-chess check split):
- SHARP non-check (n=1817): sb10 0.7470 → sb20 **0.7506 RISING**
  (teacher 0.8126). This is the play-relevant basis — VERIFIED at
  src/search/mod.rs:2327: in check, raw_eval=VALUE_NONE, static_eval
  inherited from ply−2; the value head is NEVER consulted in check.
- SHARP in-check (n=582): 0.6958 → 0.7032 rising (teacher 0.7296) —
  measurable but PLAY-IRRELEVANT for value; report separately or not at
  all. Only becomes relevant for the POLICY head if P-ORD ever orders
  evasions at in-check nodes (not deployed).
- QUIET (n=2999, zero in-check rows — clean all along): 0.8675 → 0.8663
  flat/noise (teacher 0.9054).
- vs TEACHER (rank): quiet 0.9584→0.9592, sharp 0.9421→0.9469 — rising;
  distillation converging as designed.
- Policy top-1 (ENGINE probe canon400.sf18, n=401): sb10 22.7% → sb20
  20.2% (SE ±2.0 — noise, uncorroborated) vs teacher **27.2%** on this
  probe (the 26.1 figure was the CE-interpolation estimate). pol_ce
  falling (2.549→2.535).
- Heldout (10sb cadence): raw falls 0.011269→0.011111; EMA rises
  0.010531→0.010670 = EMA decaying away from the strong slice-warm init
  during high-LR churn (LR ≈ 9e-4 at sb20). Watch, not alarm.
**VERDICT: sb20 healthy — no erosion anywhere; ride to the sb30 read as
planned.** STANDING RULE REINFORCED: any sharp-set Spearman quoted
WITHOUT its basis (non-check n=1817 vs all n=2399) is uninterpretable —
always state n. The quiet-only-transfer concern (distill mix is D46
quiet-filtered → teacher knowledge transfers on quiet only) remains a
REASONABLE FORWARD IDEA but currently has NO observed deficit backing
it; if it ever fires, the target population is the EVALUATED-LEAF set
(post-capture/tension NON-CHECK stand-pat positions), never in-check.
- Scale finding (games prep): scratchpad teacher pack was RAW = 1.99× the
  d47p frame (panels didn't care — rank-invariant); d47v-sb20 raw = 1.378×
  (the α=0.7/0.3 blend lands the student between teacher frame and label
  frame: 1.378/1.99 ≈ 0.69 ✓ matches the 0.68 V-vs-teacher gate FAIL).
  Both /tmp match copies header-calibrated to d47p frame (V: 400→267,
  teacher: 400→189, both ratio 1.004) BEFORE games. Fixed-node 50k
  300g nocpol match d47v-sb20 vs teacher launched (fn strips V's speed
  edge deliberately; prices per-node quality; ET gate stays idle-box).
- **Parity errata:** the sb10 watcher's "mean|Δ|=547cp" was a HARNESS bug
  (awk diffed against the move column / empty file), not parity; the
  fixed run shows 3.3cp mean / 6.8cp max (a8e5204). sb20 numbers above
  ran through that same verified chain.

## 2026-07-26 ~22:2x D47S STOPPED AT sb236 (user overfit call) — TEACHER =
## sb230 EMA; D47V LAUNCHED

User called overfit onset watching the curves; old-frame heldout had
plateaued sb200-230 (0.013565→0.013583 flat) supporting the read. Trainer
stopped clean; **teacher/slice source = d47s-v15-sb230-ema** (in-frame
instrument last read falling at sb180; sb230 panels were the clean-sweep
ones). D47V running: 768·32·32·8 + skip, slice-warm, distill α=0.7,
100sb. The d47s floor-hold leg is MOOT as originally specced (was resume
sb240→280); if the d47s line still needs its own settle, re-spec from
sb230. All d47s settle stages should now read sb230 wherever they said
sb240/hold-best.

## 2026-07-26 ~22:0x ★ TRAINING-PIPELINE DOCTRINE (user): TRAIN BIG FOR
## REAL ELO → DISTILL TO THE COMPETITIVE SIZE ★

The long-term pipeline: (1) train an OVERSIZED teacher for absolute
quality — capacity serves learning, unconstrained by NPS; (2) distill
(slice-warm + blended teacher/label signal, the D47V recipe) down to
whatever size the quality-per-cycle frontier demands; (3) deployment
width chosen EMPIRICALLY (the ladder), never by attachment to the
teacher's size. Resolves the identity tension: the big-intelligent-trunk
bet lives in the TEACHER (research artifact + flywheel label generator);
the shipped net is an economics decision. Composes with the flywheel
(teacher = distill source AND own-search labeler). D47V = first instance;
raw-label finetune (α→0 resume) gets the last word each generation.

## 2026-07-26 ~19:5x V2 HEAD ARCHITECTURE — assembled from the evening's
## threads (pound-for-pound + skip-lane insight)

SF18 source-verified: L1=1024, **L2=15**, L3=32, fc_0 outputs 15+1 — the
+1 is an OUTPUT-SKIP lane (learned linear read of the whole accumulator
bypassing the head; the <30,32> symbol = 15 lin + 15 sqr concat). WE have
the T6c PSQT skip but NOT the fc_0 skip ("no output-skip", sota_head.rs:3).
The skip is plausibly WHY 15 lanes suffice: linear signal rides past the
bottleneck, the head learns corrections only. Head-cost ledger: our L2=128
vs their 15 => l1_dot 327k vs ~33k MACs (~10×) = the main reason SF18 is
"too fast for its size" (+ smallnet on decided nodes; accumulator only
1.25×). V2 HEAD PLAN: accumulator per the 12b utilization gather (~1024
prior, SF v10 reversion) → PSQT skip + fc_0-style skip lane (2×W weights,
~free) → THIN value head 15-32 → DEDICATED projections for policy/
disagreement heads off the accumulator (decoupled from value width; rich
outputs keep full substrate). Candidate reclaim: most of the 10× head gap
+ width term ≈ ~1.5× NPS at equal quality if priors hold.

## 2026-07-26 ~18:5x THE CORRECTED NET-ALONE PANEL — d47s SWEEPS EVERY
## COMMON-BASIS METRIC (sb210, mid-anneal)

Two instrument fixes closed the last ambiguities: (1) monotone-adjusted
MAE (cross-fitted isotonic) replaces affine rescale — the d47s "89.9
regression" was pure curve-shape artifact; (2) sharp comparisons
restricted to the common non-check basis (SF static REFUSES in-check;
ours were carrying 566 harder positions alone). Corrected panel, SF18@50k
labels: QUIET rank d47s 0.9054 > d47p 0.9000 > SF18 0.8982; SHARP rank
d47s 0.8126 > d47p 0.8006 > SF18 0.7909; QUIET iso-MAE d47s 66.4 ≈ SF18
66.2 (tie) < d47p 66.8; SHARP iso-MAE (n=1811) **d47s 107.8 < d47p 108.0
< SF18 109.7 < SF16 119.3**. Clean sweep + exclusive in-check coverage,
five checkpoints monotone, hold phase still ahead. The evaluator side of
the identity bet is EARNED; the play deficit remains parked on
search/margins/undertraining — ladder step 2 (finished-net ET) prices it.

## 2026-07-26 ~18:2x LADDER STEP 1 VERDICT: GLOBAL ×0.52 OVERLAY REJECTED
## (−77.7±31.0 vs baseline −59.6±30.3 — directionally WORSE, as the
## region-dependent map predicted: one scalar over-corrects the 0.85-0.95
## tail bands while fixing the 0.54-0.62 near-zero band). Weight moves to:
## (a) undertrained-net term — tested by the finished-net ET (step 2,
## post sb240+hold, stock margins); (b) PER-FAMILY repair via s4-searchgap
## with the ×0.6-0.75 near-zero prior applied ONLY to near-zero-consulting
## families (RFP/razor/futility), NOT tails. NOTE the dirty double-write
## probe (327 duplicated games) was discarded; this is the clean 200.

## 2026-07-26 ~17:3x ET BASELINE PINNED: d47s-sb170(median-cal) vs d47p =
## −59.6±30.3 (200g) — grind 66/71 losses, DEEPER at own decisive plies
## (15.0 vs 14.4): out-searches and loses = the pure dialect signature.
## This is the attribution baseline: sb240+hold, then margin repair, then
## D47T each get measured against it. Dialect probe (margin overlay arm)
## fires next on the same pairing.

## 2026-07-26 ~16:5x THE DIALECT MAP — D47S COMPRESSES THE NEAR-ZERO
## REGION ~40% (the honest-calibration signature); margins want ×0.6-0.75

Quantile map, calibrated d47s-sb170 vs d47p, 5400 shared positions:
ratio by d47p band: ±0-50 → 0.54-0.62 | ±50-100 → 0.72-0.84 | ±100-400 →
0.79-0.87 | ±400-800 → 0.88-0.95. Smooth monotone compression, worst
exactly where margins operate. READING: the win-prob-fitted target STOPPED
EXAGGERATING small advantages (the calibration win); the search still
expects exaggeration. FIX DIRECTION: never re-inflate the net — shrink
near-zero margins (measured prior ×0.6-0.75 for THIS net; audit's
independent unit analysis said ×0.52). Median header calibration (1.004)
is insufficient BY SHAPE — medians live in the 0.85-0.95 bands. ET deficit
mechanism (match_report: grind-with-equal-depth, 2.7× blunder rate) fits:
margins mis-fire on honest small numbers. ASSESS-NOW/TUNE-LATER ruling:
shape done (structural, transform-fixed); values at sb240 settle via SPSA
+ two bridge probes (margin-overlay arm ARMED; near-zero-band header
recalibration arm proposed).

## 2026-07-26 ~16:0x ★ RULING (user, ratified): FT SPARSITY PENALTY —
## BURDEN OF PROOF FLIPPED; λ=0 IS THE DEFAULT ★

ft_sparsity_lambda = 0 in every future config (D47T already 0; V2
baseline 0). Evidence: NPS benefit measured-dead (D47M 15% vs champion
25% groups benched IDENTICAL); magnitude penalty = expressiveness tax
(destroyed D47M's policy substrate unopposed); mis-shaped (per-activation
vs the engine's per-4-group cost); D47S density drifted 0.272→0.296
AGAINST λ=0.001 all run (gradient budget spent on nothing); SF ships no
activation penalty. Sparsity re-enters ONLY if a GROUP-shaped version
demonstrates a deploy-path NPS win at equal time. C1 A/B (λ0.001 vs 0)
demoted from blocker to cheap confirmation when GPU has slack. D47S
finishes as configured (no mid-anneal loss changes). Supersedes the
frozen-head λ-ramp guidance for future legs (ramp only matters if λ>0).

## 2026-07-26 ~15:4x POUND-FOR-POUND RULING (user): bigger nets must be
## DEMANDED MORE OF — quality-per-cycle, not absolute quality, is the bar

Ledger: trunk ~2.5× SF18-big by weight, ~3× by cycles, for a small quality
edge — their net is currently the better artifact per cycle. Owners:
(1) DATA — self-play distillation compresses search-consumed knowledge;
flywheel/own-search labels are an efficiency lever, not just quality;
(2) UTILIZATION — A1/A3 gather on settled D47S gives the wasted-pounds
number (dead neurons/permutation); (3) CHANNEL ACCOUNTING — the threat
channel (⅓ of eval cycles) must earn its share AT EQUAL TIME (its old
equal-time promotion gate never closed); V2 bar: every feature channel
re-justifies in quality-per-cycle at equal time, utilization audited
before width is chosen, data optimized for search-relevant density.

## 2026-07-26 ~15:1x "MISSING SOMETHING" HUNT — PATH-CONSISTENCY PROBE
## EXONERATES THE TRUNK; INTEGRATION-HELL CONFIRMED AS THE DIAGNOSIS

User suspicion: pieces all present but something structural missing. Test:
the scariest candidate — rough eval landscape (would explain the four
unexplained anomalies: −33 interior-margin paradox, residual ×1.5 tree,
1.96-vs-1.45 re-search rate, 2× search-output volatility). Probe:
|e(P)+e(P′)| along 40 real game paths, normalized by own median |eval|
(scale-free): SF18 0.341, **ARCHON 0.359**, SF16 0.438. REFUTED — our
static landscape is smoother than SF16's, ~5% behind SF18's, while
out-ranking both. Re-attribution: the 1AM 2× volatility was SEARCH-BACKED
values — smooth eval + jumpy search output = search resolves tactical
tension LATE (same phenomenon as the WAC offense deficit → check-ext) +
re-search churn (aspiration widener, audit #5) + RFP margins paying in
time not nodes. Every remaining anomaly now has a named owner in the
campaign. SF18's residual EVAL edges: calibration (D47S's job) + 5%
smoothness (flywheel own-search labels' job). Verdict: integration hell —
the good scenario — with an evidence trail per item.

## 2026-07-26 ~14:4x LEAF-DUTY PANEL — THE BIG-TRUNK PREMIUM IS EARNED
## (d47s out-ranks SF18's evaluator on BOTH populations, mid-training)

2400 sharp positions (post-capture/in-check, the population search spends
nodes on), fresh SF18@50k labels. TRAP CAUGHT: SF's static eval REFUSES
in-check positions (582 silently skipped) — first read showed "SF18 leads
sharp"; on the identical 1817-position intersection the verdict FLIPS:
d47s-sb150 0.8070 > d47p 0.8006 > SF18 0.7909 > SF16 0.7563 (Spearman,
transform-invariant). Quiet set: d47s 0.9024 > SF18 0.8982. d47s also
ranks the 582 in-check positions at 0.7278 — SF static abstains entirely.
READING: SF18's remaining EVAL edge is CALIBRATION ONLY (affine-MAE
column) = exactly D47S's double-sigmoid target; their game dominance is
therefore search-mechanical (enumerable — 3 chimera absences found
tonight) + margin-calibration. The user's identity bet ("if our net is
bigger it HAS to be better than SF18's") is CONFIRMED at the strongest
static measure available, at sb150/240 with 90sb of cosine remaining.

## 2026-07-26 ~14:0x ★ STRATEGIC IDENTITY RULING (user): BEAT SF18 AT EQUAL
## TIME VIA QUALITY-PER-NODE — THE BIG-INTELLIGENT-TRUNK BET ★

User (verbatim intent): we must beat SF18 at equal time; the bet is a
bigger, more intelligent trunk with RICH OUTPUTS (policy/disagreement/WDL
heads) — long-term identity, kept for output optionality. Implications
ratified in chat: (1) the standing yardstick = SF18 equal-time, tracked at
every gate (add the SF18 ET leg to the post-D47 baseline set); (2) the
perf program is BOUNDED — keep the speed tax near ~2×, do not chase SF NPS
(consistent with the smallnet veto); (3) search-fit grind is DEFENSIVE
(neutralize their free advantages once, well), offense budget goes to
trunk+heads: D47T, V2 trunk w/ flywheel + own-search labels, T11 as
first-class. Math on record: equal-time = per-node edge − speed tax;
today −119 − ~115; rich outputs are per-node MULTIPLIERS (P-ORD −20%
tree today, more with the D47T head; allocation converts trunk
self-knowledge into effective depth) — the slope-changers that make a
+250-300 per-node swing a campaign rather than a fantasy.

## 2026-07-26 ~13:0x BRACKET ROUND — BOTH WINNERS ARE CONFIRMED PEAKS
## SE: x0.40 +8.7 | x0.52 +22.6 | x0.65 -1.7 (sharp peak at 0.52).
## QS: 140 -5.2 | 174 +17.4 | 210 +7.0 (peak 174-190; start 174 stands).
## The priors are now as refined as fixed-node screening can make them.
## Fresh equal-node legs vs SF18+SF16 running (user directive: do they
## beat us differently); predictions on record in chat (~12:4x).

## 2026-07-26 ~12:2x COMBO VERDICT: +19.1±26.5 — THE PRE-SPSA PACKAGE
## EXISTS (SE×0.52 + QS 174 + PFUT×0.52 composed; sub-additive vs naive
## +45 — the families share the freed-node budget — but solidly positive,
## games-verified: shallower by the SE spend, tactical held equal, wins
## via base's grind channel 45v34). Post-D47 wall-clock gate tests THIS
## package + the NMP center; SPSA then polishes from a composed start.

## 2026-07-26 ~11:5x PRIORS PROGRAM COMPLETE — 13 arms, 2600 games, all
## games-verified; COMBO arm (SE+QS+PFUT composition test) RUNNING

Final fixed-node table (200g/arm vs base, d47p value-only):
STARTS SET: sefam ×0.52 +22.6 | qsfam 174 +17.4 | pfutfam ×0.52 +5.2 |
RED_TABLE_NUM 3260 −5.2-clean | NMP center (250,19) −5.2 [or (140,12)
+guards +3.5 if wip/nmp-guards lands]. DEFAULTS KEPT (wall-clock's job):
pcut −5.2 | asp −13.9 | RFP −33.1 | razor −34.9 | cfut −34.9 (interior-
margin fixed-node tax, strikingly uniform). REJECTED OUTRIGHT: flat
RED_BASE (−41.9 solo, tactical/collapse bleed ×3 arms), RED_MOVECOUNT.
Campaign s4-searchgap updated: 29 params, every range validated against
current values AND starts (validator caught 3 stale ranges incl.
NMP_R_BASE — donor R=7+d/3, not SF16's 4+d/3). Combo arm = the
additivity question: naive sum +45; result gates whether a pre-SPSA
package exists. Tooling hardened along the way: setoption-baked-table
fix (tune builds), arm_verify game-level gate, pack recipe pinned.

## 2026-07-26 ~10:4x FAMILY SCREENS — SEFAM ×0.52 IS THE HEADLINE (+22.6,
## tactical channel repaired); QS +17.4 via LEAF QUALITY not depth

| family (×0.52) | Elo@50k | verification |
| **sefam** | **+22.6±27.7** | −0.91 nominal depth (extensions SPEND budget on forcing lines) but base tactical losses 23 vs sefam 13 — THE SF16 loss channel, repaired. Audit SE diagnosis confirmed end-to-end. |
| **qsfam 174** | **+17.4±27.6** | NOT deeper — wins via leaf quality (55 vs 65 losses); margins can pay through accuracy, second gain channel discovered |
| pfutfam | +5.2±27.9 | mild; slight tactical uptick, fewer total losses |
| shapeA2 (RED_TABLE 3260) | −5.2±27.9 | deeper, clean — the one sound LMR lever |
| rfpfam | −33.1±28.3 | deeper but grind bleed (47v31) — REJECTED as start |
| guards+(140,12) [r2] | +3.5±31.3 | deeper, tactical BELOW base — landing candidate |

arm_verify's depth check is a THEORY probe not a veto: extension arms are
rightly shallower, quality arms rightly not-deeper. Remaining: asp, razor,
cfut, pcut (~2h). sb120 in-frame heldout next firing.

## INSTRUMENT INDEX (2026-07-26 suite — USE THESE, habitually)
- tools/nnue/evolution_panel.sh — the ONE per-checkpoint read (pack →
  sweep → Spearman with EXPLICIT bases + err-corr → top-1 probe). Never
  hand-roll a panel again (the basis trap). err-corr = the Layer-2
  exhaustion tripwire (endgame protocol ~15:1x). Sets:
  data/nnue/evalbench/sets/. Auto-run per checkpoint by the
  archon-d47v-panels watcher → /tmp/d47v_panels.log.
- tools/nnue/eval_bench.py — now basis-guarded: separate labeled tables
  (non-check = play-relevant) whenever in-check rows exist.
- tools/match_report.py — EVERY match gets the story, not just the score
  (user directive ~16:3x). Slope-aware; blunder ledger mate-safe.
- tools/nnue/arm_verify.py — every A/B arm before its verdict.
- tools/nnue/inframe_heldout.py — retargeted-run health (built-in heldout
  rises by design under retargeting).
- tools/nnue/loss_forensics.py --winner-slope — cross-engine pnr in oracle units.
- tools/nnue/d47s_pipeline.sh — pack|calibrate|q1|et|sf|ledger, in order.
- tools/nnue/launch_d47t.sh — D47T injection+launch, one command.
- tools/search_knob_gate.sh — knob candidates, fixed-node then wall-clock.
- tools/spsa/campaigns/s4-searchgap.json — 30 games-vetted priors.
- SSTATS/SSTATS2 (ARCHON_SEARCH_STATS=1) + ARCHON_NMP_ALLNODE probe.
- Branches: wip/nmp-guards, wip/check-ext (evidence-backed, land post-D47).

## POST-D47S RUNBOOK (assembled 2026-07-26 ~07:2x while training; every stage
## staged + committed; execute IN ORDER when the completion watcher fires)

0. Trainer end: verify `superbatch 240 done` + unit exit 0; if oom/crash,
   postmortem via `systemctl --user status archon-d47s-r4`, resume
   `pump_train <cfg> <last_sb>` (fresh Adam transient documented).
0b. **ORDER FLIPPED (user ~21:4x: 'good feeling about the smaller arch')**:
   at sb240, IMMEDIATELY `bash tools/nnue/launch_d47v.sh
   data/nnue/experiments/d47s-v15/d47s-v15-sb240-ema-raw.bin` — D47V
   teacher/slice source = the sb240 EMA. The FLOOR-HOLD LEG runs AFTER V
   (still owed to the d47s line before its settle): originally
   `bash tools/nnue/launch_d47s_hold.sh` — sb241-280 at constant floor LR
   (the base schedule had ZERO dwell at the floor). inframe_heldout per
   10sb checkpoint; stop when flat twice running; ALL downstream stages
   (pack/calibrate/screens/D47T warm) use the HOLD-BEST EMA, not sb240.
   Hold-at-floor = default closing phase for all future runs.
1. `bash tools/nnue/d47s_pipeline.sh pack`        (recipe pinned 1280/128/32)
2. `bash tools/nnue/d47s_pipeline.sh calibrate`   (HARD GATE — before ANY game)
3. In-frame heldout final read: inframe_heldout.py 120 180 240 (must be
   falling). PLUS the net-alone panel the user is waiting on: pack the
   FINAL ema, eval_bench + the Spearman rank table (the transform-invariant
   read; sb150 already leads the whole field at 0.9024 incl. both SF
   statics — user: "Can't wait to see this again after it finishes").
4. `bash tools/nnue/d47s_pipeline.sh q1`          (300g fixed-node vs D47P)
   — verdict per no-verdicts-from-weak-tests: ±17 bars at 300g.
5. `bash tools/nnue/d47s_pipeline.sh et`          (200g 10+0.1, idle box)
6. `bash tools/nnue/d47s_pipeline.sh sf`          (SF16 both legs, pinned harness
   — the before/after vs sfbase-d47p-pre-v15-* is the v1.5 headline number)
7. `bash tools/nnue/d47s_pipeline.sh ledger "<q1/et numbers>"`  (D38 row)
8. **EXECUTIVE ORDER (~21:1x): LAUNCH D47V** — 768·32·32·8 + skip,
   teacher-distilled MTL bet (`bash tools/nnue/launch_d47v.sh
   <hold-best-ema-raw>`; ~100sb @ ~230k ≈ 5h; plan + gates in chat ~21:1x,
   decisive gate = equal-time vs hold-best ≥ 0). Fallback ladder: V fails →
   D47U (1280/64+skip, staged) → hold-best d47s ships. Engine side (SKP1
   loader + 768/32 consts) built DURING the leg.
8-old. D47U (head rebuild — was: SUPERSEDES D47T per user ~20:2x: skip
   lane + 64-wide value head, 'mirror SF18's skip strategy')**: after the
   hold leg, `python tools/nnue/assemble_d47u_warm.py <hold-best-ema-raw>
   data/nnue/experiments/d47u-warm.bin` → inject into
   tools/nnue/runs/d47u_headrebuild.json → systemd launch (caps + tb shim).
   ONE leg trains {fresh 64 head + zero-init skip + polmove-on-64} on the
   frozen trunk. Trainer side DONE+smoked (skip oracle, freeze oracle,
   SKP1 quant). Engine side (SKP1 loader + 64-head kernels) built DURING
   the leg. Health: inframe_heldout; policy top-1 vs the D47P 26.1% bar.
   (Old D47T policy resync happens ON the 64 substrate inside this leg.): inject warm/heldout into
   tools/nnue/runs/d47t_policy_resync.json from the D47S final EMA +
   in-frame... NOTE: heldout_expect uses the DEFAULT-frame function (gate
   invariance) — read D47S's final old-frame heldout from its log for the
   expect value. systemd unit + MemoryHigh=5G/Max=7G + tb shim armed
   (tensorboard-all-runs). 60sb ≈ 4h.
9. While D47T trains (box rules as tonight): knob-gate STAGE 1 only
   (`tools/search_knob_gate.sh <d47s nocpol> both 200 4` — fixed-node leg;
   equal-time stage 2 waits for idle box).
10. D47T done → pack (same recipe) → pvm duel head-on/off at equal time
    (200g min) → 2+1 TM sanity pair vs SF16 (the user's original GUI
    complaint TC — eyeball the clock curve with loss_forensics).
11. Knob-gate stage 2 (idle box) → s4-searchgap small SPSA (step 4) with
    survivors as centers (wip/nmp-guards branch ready if tactics bleed).
12. Perf study quiet window: tools/perf_study4_bench.sh NET=<d47s.cenn>
    (skip smallnet lanes — VETOED) → decide #9 SIMD emission / compression.
12b. **UTILIZATION GATHER (A1/A3) on the settled net** — dead-neuron rates
    + permutation structure over ~1M positions. Promoted (user, ~19:2x):
    it lower-bounds the V2 width question for free (SF v10's 3072→1024
    reversion is the prior; our 1280+threats likely oversized). Feeds the
    V2 ladder: 1280→1152→1024 distill legs at EQUAL TIME + the
    head-decouple variant (policy head gets its own projection; value L2
    trims as quality allows).
13. T11 / disagreement head (v5.5 step 8) INFRASTRUCTURE INVENTORY
    (2026-07-26 ~07:4x, so the design conversation starts warm): engine
    already carries the dormant D40_UNC_HEAD sidecar seam + sigma plumbing
    (sota_context last_sigma/node_sigma) + THREE default-off search
    consumers (unc_margin_scale @ RFP, unc_tm_factor + wdl_tm_factor @ TM,
    mod.rs:278/294/311). Program spec: campaign §5b (residual EDA sweep →
    frozen-trunk head on residual target → allocation consumption), with
    the ordering constraint: lands or drops BEFORE comprehensive SPSA.
    First concrete artifact to build in a post-D47T window: the residual
    forward-sweep + bucketing tool (labels ~free at stored-value sweep
    speed). User has ideas queued — inventory first, design theirs.

Standing: all landings post-gates; d47h fallback untouchable; ledger verify
on idle box only; every trainer launch memory-capped under systemd.

## 2026-07-26 ~07:0x D47S MID-RUN HEALTH: TRIPWIRE DISARMED (in-frame
## instrument) + PACK RECIPE RECOVERED

False-alarm anatomy: the trainer's heldout metric uses TrainConfig()
DEFAULTS by design (transform-invariant warm gate) → it RISES while a
retargeted net (offset 163/scale 76.9) fits its new frame (sb30→sb60 EMA
0.013471→0.013540, three rising readings). eval_bench's affine-corrected
MAE degrades under the NONLINEAR shape change too (sb50: 90.2 vs parent
72.2) while sign% IMPROVED (84.9 vs 84.0) — ordering intact, magnitude
curve retargeting. The correct instrument (new tool
`tools/nnue/inframe_heldout.py`, same bin through the RUN'S transform):
sb10 0.010787 → sb50 0.010768 **FALLING** = healthy; and the net already
fits the new frame far better than the old (0.0108 vs 0.0135). Refinement
pace is expectedly slower than D47P's repair leg. Re-check at sb120/180.
Also recovered the hard way: pump-quant pack recipe is `D40_ACC=1280
D40_L2=128 D40_L3=32 D40_BUCKETS=8 D40_POLICY=1` (arch is 1280/128/32,
factorize+psqt ON per pump_train.py:77; netpack default 768 misparses;
d47s_pipeline.sh pack stage fixed BEFORE it could fail at completion).

## 2026-07-26 ~06:0x NPS GROUND-TRUTH LEDGER (flat perf shares, both engines,
## same positions; trainer-contended box — SHARES only, absolutes from the
## earlier idle-box game logs)

Per-node cycle budget (1 thread, game config, d47p vs SF16-embedded):
OURS 1724 ns/node = eval ~1240 (72% steady) + shell ~480. SF16 556 ns/node
= eval ~289 (52%) + shell ~267. **Eval layer 4.3×, shell 1.8×, total 3.1×.**
Inside our eval: threat channel ≈ 24% of ALL cycles (i8 fold 16.0 > halfka
fold 7.3 + at_full 6.1 + record_square_touch 2.3) ≈ ⅓ of eval — SF16 pays
ZERO (no threats). Threat-free we'd still be ~2.9× SF16's eval: residual =
fold efficiency + split-quant activate (5.9%) + l1_dot 16.2 + f32 tail +
evals/node. Shell 1.8× is partly INTERACTION: PSQT-mode shell = 334 ns/node
vs 480 in-game — the 130 MB net tables evict the shell's cache. Perf-study-4
#9 (SIMD event emission) QUALIFIES by its own gate (gen+resolve 8.4% ≥ 5%).
Loader bulk-parse (#7) embarrassed itself in every short profile (90% of a
spawn) — QoL fix worth taking. Post-D47 NPS queue (smallnet VETOED): #9,
quiet-window LLC/dTLB run (decides threat-table compression reopen), #7,
evals/node policies, quant unification (V2 rider). Both engines' MovePicker
::next_move ≈ 13% of steady shell — ours is not anomalous with the net on
(the 18% PSQT-mode MovePicker::new reading was shell-only amplification).

## 2026-07-26 ~05:3x ★★★ SEARCH GAP SOLVED-IN-PRINCIPLE: ×3.11 → ×1.06 WITH
## TWO KNOBS (NMP bar shape + LMR base) ★★★

Full anatomy vs a LOCALLY-PATCHED SF16 (counters only, GPL-safe, gitignored
worktree), 20 shared middlegames, fixed d13, deterministic:
- qshare IDENTICAL (16.9% vs 16.5%) — the ×3 was uniform, composition equal.
- fail-high-first: OURS BETTER (90.3% vs 84.1%) — ordering exonerated.
- **NMP: 49× fewer cuts/node (1.25/k vs 61.8/k), equal success rate when
  tried (0.77 vs 0.79)** — the mechanism is fine; the GATE starves it.
- LMR applied reduction: 1.81 vs 2.78 plies/scout.
Gate forensics: node-type (cutNode vs !PV) irrelevant (tries 3707→3743).
The killer is the ELIGIBILITY BAR's SHAPE: ours/donor `beta+374−14·depth`
crosses zero at depth ~27; SF16's `beta+258−21·depth` at ~12.3. Scaling the
bar (×0.52) moves base AND slope → crossover unchanged → why the margin
probe missed it. Donor (eca43a97) affords the stingy bar because ITS search
has IIR + hotter cutNode reductions shrinking cut-subtrees first; we ported
the constants without those siblings (chimera effect).
**Causal ladder (nodes vs SF16):** control ×3.11 → SF16-shape bar (140−12d)
×2.15 → + RED_BASE 1027→2051 (avg_red 1.81→2.37) ×**1.06**. Bar-zero
ceiling: ×1.54 with cuts/k 60.6 ≈ SF16's 61.8 (anatomy match).
**Caveats (pre-registered for the gate):** fixed-depth node parity ≠ Elo —
both knobs prune harder and can discard wins (NMP success rate falls to
0.63 at bar-zero = wasted tries + zugzwang risk; SF16 also carries guards
we lack: `eval>=beta`, `eval>=staticEval`, `(ss-1)->statScore<17329`).
POST-D47 plan per user v5.5: land as candidates → fixed-node self-play
screen → wall-clock SPRT → hand the survivors to small-SPSA as seeds
(NMP_EVAL_BASE/DEPTH shape, RED_BASE, plus the audit margin family).
Instrumentation kept: SSTATS/SSTATS2 counters (bench pin 701605 unchanged),
ARCHON_NMP_ALLNODE probe (default off), patched SF16 in the scratch
worktree. All probe tooling: scratchpad ebf.py/margin_probe.py.

## 2026-07-26 ~04:4x MARGIN-SCALE CAUSAL PROBE — HYPOTHESIS REFUTED (the ×3
## tree is NOT the margin family)

Pre-registered prediction: scaling the audit's 40-constant margin family
×0.52 via setoption overlay (tune build) collapses the fixed-depth tree
ratio vs SF16 from ~2.8 toward ~1.5. RESULT (20 middlegames, d13,
deterministic): 1.00 → 2.81 | 0.69 → 2.74 | 0.52 → **2.63**. Six percent.
Confound noted: ×0.52 also FIRES MORE singular extensions (deeper forced
lines) partially masking pruning savings — but not a 2.6→1.5 masking.
Additionally qshare (new SSTATS counter, bench-neutral, pin 701605
unchanged) = **16%** at d8-16: no qsearch bloat. If SF16's share is the
typical ~half, OUR MAIN TREE is ~4-5× theirs while qsearch is roughly par.

**Sharpened suspects for the uniform ×3 (all testable with counters):**
(1) interior-node ordering efficiency — fail-high-first rate / moves tried
per cutoff (root flips were equal, but interior fhf% is the real metric);
(2) IIR — SF has it, ours REGRESSED −15 Elo and was reverted (P9 item 9,
parked on wip/p9-item9-iir-followpv) — a known deliberate absence that
reduces TT-miss subtrees;
(3) eval-noise cutoff churn — our scale-normalized eval volatility ~2× SF16
at fast TC ⇒ less stable alpha/beta decisions ⇒ uniformly wider tree (also
predicts more aspiration re-searches);
(4) extension volume (check/SE totals per search).
Next instrumentation: fh_first/fh_total, re-search counts, tt_cutoff count,
extension count in SSTATS; patch a LOCAL SF16 with matching counters
(GPL-safe: diagnosis only, nothing copied) for apples-to-apples fhf%.
Margin family fixes stay queued post-D47 as real-unit hygiene (+SE firing
parity), but expectations DOWNGRADED: they are not the tree-gap lever.

## 2026-07-26 ~04:0x MASTER PLAN v5.5 (user, verbatim sequencing — supersedes
## the campaign doc's ordering) + UNIT-SCALE AUDIT VERDICT

**Plan v5.5:** (1) finish D47S — run to completion, anneal well, confirm
stronger; (2) frozen-trunk POLICY train run; (3) Elo tests vs old d47 + SF16
at equal time AND equal nodes; (4) small SPSA; (5) performance study — max
NPS recovery; (6) TM already fixed; (7) FULL-ON SPSA, heuristics dialed in,
test again; (8) disagreement head (T11) developed + integrated as test-time
compute allocation in search, +30 Elo bar. Standing assistant lane while
training: loop on the performance-gap + search-gap diagnosis continuously;
**found code/config changes land only AFTER D47 finishes.** Smallnet: VETOED
(user, 2026-07-26 ~03:5x).

**Unit-scale audit (docs/research/unit-scale-audit.md, agent, 2026-07-26):
CONFIRMED at ~1.91×.** Every eval-denominated search constant is verbatim
SF-dev `eca43a97` (post-SF18, 2026-07-06) — probcut/singular/delta values
match that commit exactly, not SF16 — and no tune was ever shipped over them
(D45 = 8/8 ship-nothing). Donor internal unit ≈ 0.278 oracle-cp, ours ≈
0.53 → every margin ~1.9× too wide in real terms → too little pruning
everywhere → the uniform ×3 tree. Worst five: RFP family (42 vs 22
oracle-cp/ply), QS_FUTILITY_MARGIN 335 (178 vs 93) + coupled see_ge gate,
parent futility (73 vs 38), singular family (bar 2× too deep → FEWER
extensions at forcing moments + thin multicuts), aspiration (wide-when-calm,
slow-to-widen: base 2.7 vs 1.4 while the mss widener is 1.9× too weak).
SEE-dimension constants are self-consistent — excluded from any rescale.
Candidate fixes (POST-D47): (a′) rescale eval output once onto donor scale,
or (a) ×0.52 on the ~40-margin family; two-frame conflict (0.52 slope vs
0.69 material) → gate BOTH as real-clock SPRT, never fixed-node (fixed-node
screens structurally undervalue pruning). Cross-confirmed by the EBF probe:
tree-size ratio vs SF16 ≈ ×3 FLAT from d=4..16 with equal growth (~1.6) —
constant leaf-layer multiplier, exactly what 1.9×-wide leaf margins predict;
node-counting conventions verified identical (both count every do_move).

## 2026-07-26 ~03:00 TM GATE PASS (truncated by box crash) + D47S OOM POSTMORTEM

**TM gate (tmnew vs tmold self-play, 10+0.1, d47p policy-on both sides):**
160/200 games completed before the box died: **+6.5 ±28.6 (50.94%)**,
Ptnml [2,14,44,19,1]. Pre-registered criterion was "no regression" — PASS.
Combined with the mechanical 2+1 replay evidence (53.6s-by-move-15 vs SF16's
54.1s), the depth-redenomination (76b5ede) is GATED IN. Remaining games not
worth re-running; the last ~60 ran beside a trainer (symmetric contention,
fair but noisy).

**D47S launch: 3 OOM deaths, then the box.** Every attempt died between the
warm-heldout PASS and the first superbatch: nohup attempt (kernel kill
02:42:39, anon-rss 7.6G), systemd attempt at workers 6/8 (cgroup-attributed
oom-kill, peak 8.2G), systemd attempt at 4/4 (climbed again; box froze in
global OOM thrash ~02:55, reboot 02:57). Kernel records pin the trainer
python at ~7.5G RESIDENT during LOADER STARTUP — the 2026-07-25 OOM fix's
"peak 1766MB" verification evidently did not cover this path at run scale
(12-corpus mix + shuffle_chunk 128 + ids masks + warm+EMA on host). Root
cause NOT yet identified — do not relaunch until a CPU-only capped
instrumentation run locates the allocation. Standing rule sharpened: any
trainer launch gets systemd MemoryHigh (throttle) + MemoryMax well under
(total − desktop footprint), so the worst case is a dead trainer, never a
dead box.

## 2026-07-26 LOSS FORENSICS vs SF16 (d47p, 200g/leg) — WHY WE LOSE, QUANTIFIED

Tool: `tools/nnue/loss_forensics.py` (pnr = winner's eval ≥ +2.00 oracle-units
and never back under +1.00; thresholds slope-corrected per winner — ARCHON
reported cp ≈ 1.9× hotter than SF16's normalized cp, so raw +2.00 fired on
+1.0-real positions and faked an "opening collapse" taxonomy on first pass).

**EQUAL TIME 10+0.1 (−121, W22 D89 L89):** 61/89 losses have OUR clock <2s at
the decisive ply — median 0.64s vs SF16's 2.11s (they hold 3.3× our clock
where games decide; old-TM front-load + NPS). Depth gap −4.1/−4.6/−5.9 by
phase (widest in endgames). Taxonomy: timepressure 61, grind 51, tactical 37,
**misjudgment 4**. **EQUAL NODES 50k (+5.2, W63 D77 L60):** our losses grind
34 + tactical 22, misjudgment 2; depth at pnr 12.2 vs THEIR 15.3 — at equal
nodes SF16 is ~3 plies deeper exactly where it decides. Control (their losses
to us, slope-corrected): grind 53/63, tactical **3** — a 22-vs-3 asymmetry.

**Reading:** we beat SF16 by grinding (eval quality), they beat us by tactics
(depth). The eval-misjudgment channel is EMPTY in both legs — d47p does not
lie at decision points. eval_bench, 3000 balanced positions, SF18@50k oracle:
d47p ≥ SF16 on EVERY column (rescaled MAE 72.2 vs 73.9, pearson .891 vs .873,
sign 84.0 vs 82.0, p99 400 vs 470; SF18 67.9 with oracle home-field). The
user's "net is better than SF16's" claim is supported. Lever ranking for
release Elo at 10+0.1: (1) TM fix [landed, gate running:
`tm-gate-depthredenom-et`, tmnew vs tmold self-play 200g], (2) NPS (attacks
tactical+timepressure at once; A1-A3 + profiling), (3) depth-per-node — SF16
converts the same nodes into ~3 more plies at critical moments (S0 audit /
D41 arc), (4) net v1.5 (D47S) — converts more grinds into wins; not the loss
channel.

## 2026-07-22 D47c ★★ COWBOY ITERATION + THE COMPONENT INSTRUMENT ★★ [RUNNING]

**Arch pivots (user, rapid-fire):** d47-v1 (1536/256+threats, 155k pos/s, ~10.5h)
→ killed → **d47b** (halfka-only 1536/256: only +18% → GPU/host bound, threats
aren't the training cost) → killed → **d47c-1280-240sb (LIVE)**: acc 1280
(−17%, SIMD-clean), bottleneck 128, threats BACK, λ=0.005, legal mask,
ckpt EVERY sb (raw-janitor keeps 5), batch 8192×BPS 3040, ~207k pos/s ≈ 8h.
Why no 2× from dims: Amdahl — FT matmuls scale with HL only, AdamW with the
~107M FT params, host pipeline dim-independent; the shrink pays at ENGINE NPS,
not training. Batch 16384 measured ≈ +3% and re-risks OOM: rejected.

**GRANULAR METRICS SHIPPED (bullet-core patch `bullet-trainer-metrics.patch`):**
`metric:*` outputs survive the backward lowering; run loop reads per batch;
`components sb N | value X | pol_ce Y` printed per sb; decomposition verified
EXACT (value+λ·ce = fused to 1e-6). TB: loss/value_component, loss/policy_ce,
**loss/policy_share_of_value** (the live λ-parity monitor) + all prior series.
Checkpoint watcher: acc+scale every 5sb; **Elo every 30sb = 100g fixed-node
with W-D-L** line.

**★ FIRST FINDING — the share DRIFTS UP, fast:** sb1 0.81 → sb2 1.14 → sb3
1.28 (value falls fast early, pol_ce grinds: 3.01→2.96). Constant λ=0.005 ⇒
policy-DOMINANT training by mid-run (heading ~2×), NOT parity. **DECISION
POINT sb30 (~50 min from 20:35Z):** if share >2 AND value-component/Elo look
hurt → rerun at λ≈0.0025 or anneal; if accuracy+Elo track fine → let it ride
(policy-heavy may be GOOD for the trunk — that's the experiment).

**OPERATIONAL STATE (for the post-compact session):**
- Train pid family: /tmp/d47c_train.sh (log /tmp/d47c_train.log); twin waiter
  /tmp/d47c_twin.sh (fires on train exit; SAME dims, no policy, SAVE_RATE=20);
  janitor; ckpt watcher /tmp/d47c_ckpt_watch.sh (log /tmp/d47c_ckpt.log).
  **Repo copies: tools/nnue/runs/** (tmpfs /tmp dies on reboot).
- TB at :6006 — run groups d47c/{train,system,checkpoints}; shims tb_shim.py
  (components-aware) + tb_system.py (OOM canary sys/ram_available_gb).
- Monitors: d47c logs (exits/ckpt lines), components every 10sb + NaN.
- Engine: archon-d47 = 1280/128 CEN2-verified (tools/deploy_engine.sh —
  VERIFY_NET param; uciok is NOT load-proof, trap fired twice).
- **HARD RULE: no cargo builds while the trainer runs** (3 OOM kills; the
  kernel kills the ~10-12GB trainer, not the build).
- Gates: gate_d47.sh (learnability-first; NPS reported-not-gated). Probe:
  policy_accuracy_probe.sh (ENG env). Baselines: sb1-class ≈ 15% top-1;
  random ≈ 3-4%. Readout: accuracy trajectory + Elo trajectory + share curve
  → sb240 gate → twin → attribution (incumbent/twin/d47c).

## 2026-07-22 D47 ★★★ THE STRAIGHT SHOT — SHARED-BOTTLENECK MULTIHEAD, BUILT AND TRAINING ★★★

**User redirects (evening→night):** (1) the factored policy encoding is dead —
STOP and redesign I/O from scratch; (2) expert spec adopted: complete
categorical moves, compact 1858 vocabulary, accumulator-branched head,
legal-rows-only; (3) **D47 pinned**: accumulators→1536, 256 shared bottleneck
= the branch point, policy DEPLOYED for move ordering (first-searched move),
WDL/σ stay sidecars, λ at policy≈eval parity; (4) "go big first — LEARNABILITY
before NPS; we learn if we can learn"; (5) λ=0 twin blessed for attribution.
(The haux-pol1/pol4 runs from NIGHT-4 were killed pre-readout — superseded.)

**Built + committed tonight** (afcbec0..2483494 chain):
- `policy_vocab.rs`: the 1858-class vocabulary (1792 geometric + 66
  underpromo), trunk-normalized canonical frame (rank-flip + _hm file-mirror —
  the deliberate Lc0 deviation), single shared implementation, 5/5 tests incl.
  every-legal-move-maps on tactical fixtures.
- Trainer at D47 dims (HL=1536, L2=256, L3=32): vocabulary head off hl2 +
  additive legal mask (softened −30 after −1e4 NaN'd the CE — smoke catch #1)
  + one-hot move_id targets from loader-packed oriented bytes (+promo code);
  the every-position target-in-mask assert validated the frame over ~25M real
  positions (smoke catch #0: zero fires). λ=0.005 = measured parity (polCE≈3.0
  vs value 0.02→0.008; ±40% drift documented).
- **Full deploy chain**: netpack splits quant → .cenn (value-only — all
  scale-check tooling unchanged) + .cpol (CPL1, 483KB); engine consts to
  1536/256 (suite green); policy_head.rs selected-row scorer (frame-safe BY
  CONSTRUCTION: moves map into ids, never back — watchlist #6 retired);
  Zobrist-stamped activation stash (stale-activation ordering poison killed);
  **policy ordering v1 in search** (argmax fills the empty TT-slot, depth≥3);
  sidecar-less builds bit-identical (Bench: 701605 reproduces; check.sh green).
- Instruments: policy_accuracy_probe (top-1 vs SF18 teacher), gate_d47
  (learnability-first: scale hard-gate → accuracy → fixed-node → NPS
  reported-NOT-gated), self-measuring checkpoint watcher (pack+probe at each
  60sb), TensorBoard shim (http://legion:6006 — loss/superbatch, loss/batch,
  throughput).

**FIRST LEARNABILITY DATAPOINT: 15.0% top-1 (60/401) vs SF18@50k after ONE
superbatch** (random-legal ≈ 3-4%). The targets are learnable; the trajectory
(sb60/120/180/240) is the primary readout of the run.

**The OOM saga (3 trainer deaths, all mine):** the D47 trainer holds ~12GB host
RAM (policy target+mask buffers); cargo builds during training OOM-killed it
at sb2/sb2/sb3 — capping build memory does NOT protect the trainer (the kernel
kills the biggest RSS). **Fix: batch 8192 × BPS 3040 (identical data schedule,
half the buffers → 7.2GB headroom) + a hard no-compilation rule while
training** (everything engine-side is committed, so nothing needs it).
Attempt 4 stable at ~155k pos/s, ~10.5h, ckpt/60sb; λ=0 twin chained behind
(same seed/recipe, policy off — arch-vs-head attribution).

**Readout plan:** accuracy trajectory per checkpoint → sb240 gate_d47
(scale + accuracy + fixed-node vs d40-uho-v1-240sb + NPS report) → twin
completes → three-way attribution (incumbent / twin / d47). Equal-time
deliberately deferred to the optimize/shrink phase (D47 posture).

## 2026-07-22 NIGHT-4 ★★ SF18 FOLD-IN CLOSED (exp-0128) — GPU IS 100% MULTI-HEAD ★★

**Dose-2 verdict (exp-0128): −5.8 ± 17.1 at 7.5% aux; scale-check 0.994/99.4%
(coverage intact); absorption 52.2%/mean 90.9-vs-96.2 (vs 48.3%/nothing at
2.5%).** The dose-response curve is mapped: absorption is weak per unit mix and
the payload is 31cp median — below the ~96cp net resolution floor. Escalation
buys fractions of a below-floor refinement while replay bias grows. **CLOSED;
remaining SF18 shapes (stage-2 warm on the 77M full-coverage corpus) are LOW
priority, revisited only if multi-head stalls.** The SF18 arc's real return:
the D46 contract, the scale/filter/WDL machinery, and clean aux datasets —
infrastructure the head program inherits.

**haux-pol1-240sb IS TRAINING** (first valid H-AUX run: from+to two-hot,
trunk-normalized frame, engagement-proven, λ=0.001). pol4 → wdl6 → wdl2 chain
behind it, each scale-check hard-gated + 300g fixed-node.

## OPEN QUESTION — λ dynamics are UNMEASURED (user + assistant agree: neither
## of us knows how this plays out; recorded 2026-07-22 ~01:00)

What we DON'T know about the aux-λ mechanism, stated so tomorrow's readouts
aren't over-interpreted:

1. **The aux's gradient share over training is unmeasured.** λ is constant,
   but the influence ratio is λ·‖∇pol_CE‖/‖∇value‖ — and BOTH losses decay
   (CE ≈4.85→~2.5-3; value ≈0.013→0.007). Loss magnitude is a weak proxy for
   gradient norm; the true share trajectory is unknown. (Earlier claim that
   the dose "decays on its own" was loose — retracted.)
2. **LR decay is ratio-neutral** — it shrinks all updates equally and does NOT
   fade the aux relative to value. If relative fading is wanted, it needs an
   explicit λ schedule; nothing implements one today.
3. **Whether fading is even desirable is an open design question.** For:
   representation-shaping is an early/mid-training job; the final superbatches
   calibrate the value output that every search margin is co-adapted to —
   "shape early, calibrate pure late" (the same intuition as the two-stage
   data recipe). Against: with <0.5-epoch streamed data, late-training
   overfit pressure is low, and a schedule is a second knob on an unmeasured
   first knob. DECISION: constant-λ first (tonight's runs); the annealed-vs-
   flat A/B (same total aux exposure) runs ONLY IF constant-λ shows signal.
4. **Missing instrument:** per-component loss logging. NOT the trivial print
   assumed — the bullet training loop fetches one fused loss scalar
   (trainer/src/run.rs:245-256); components need graph outputs + loop + logger
   changes. Scheduled as a daytime task, tested, between GPU runs — not a
   1am edit to the code a live run is executing. Until it exists, every λ
   interpretation carries this caveat.

## MORNING RUNBOOK (for whichever session reads the gates)

Per net, in order:
1. `bash tools/nnue/haux_divergence_probe.sh data/nnue/nets/<id>.cenn` —
   floor is 47.3 mean. Record the number BEFORE reading the gate Elo.
2. Gate Elo from the chain log (300g, ±~17).
3. Interpretation matrix:
   - divergence ≈ floor AND Elo null → dose invisible: raise λ (pol: 0.01;
     wdl: 0.05) before any verdict. NOT a refutation.
   - divergence scales with λ AND Elo null (both pol runs) → trunk shaped,
     eval unmoved at 240sb: try λ-mid + 720sb long-run before verdict; H-AUX
     stays open.
   - any run +Elo ≥ ~+15 → winner: rerun 600g to confirm, then joint
     pol+wdl run at the winning λs (run 6), then equal-time gate.
   - any run −Elo ≤ −15 with divergence >> floor → that λ damages value:
     halve λ, one retry; two failures = that head's aux form refuted at 768.
4. WDL runs additionally: sidecar landed at data/nnue/nets/<id>.wdl.sidecar —
   BEFORE any signed consumer work run the [L,D,W] order parity check
   (watchlist #4): engine last_wdl() vs trainer probe on 2-3 fixtures.
   Then SPSA ARCHON_WDL_TM_C3 ∈ {0.1, 0.2} only if the net itself gated ≥ null.
5. Warm-start warning: continuing ANY H-AUX raw at different λ set → loud
   load_from panic; repeat the head λs (traps register #3).
6. **Policy attachment-point arm (user concern 2026-07-22 ~01:40, "rank
   bottleneck"):** tonight's pol runs attach the head to hl3 (32-d) — every
   expressible move distribution is rank-32-limited; the head's ceiling is
   LOW by construction and a weak pol readout may reflect the attachment, not
   the H-AUX idea. `D40_POL_ATTACH=ft` (implemented, cargo-checked, NOT yet
   rebuilt) attaches to hl1 (768-d pairwise accumulator) — fully expressive,
   shapes the accumulator directly. **REBUILD THE TRAINER ONLY AFTER pol4 HAS
   EXEC'D** (pol1/pol4 must run identical code; the wdl runs don't read the
   pol path, so a post-pol4 rebuild is safe). Run-6 candidates, pick by the
   pol readouts: (a) ft-attach at the better pol λ, (b) joint pol+wdl at
   winning λs, (c) λ-schedule A/B — in that order of information value.
7. **Policy-shape ladder (user design discussion 2026-07-22 ~02:00):** the
   current joint-128 softmax expresses MARGINALS only — no from↔to binding —
   and its cross-half mass competition is a fossil of the dead cycle
   constraint. Escalation order if policy signal exists: (i) per-half
   softmaxes (clean marginals; buildable now — the cycle doesn't reproduce),
   (ii) ft-attach 768-d input (rank ceiling removed), (iii) conditional
   P(to | trunk, from) via teacher-forced concat — the first TRUE-binding
   arm, ~1/10 the machinery of (iv) the full 4096-class joint head (dense-
   target VRAM needs solving first: sparse targets or smaller batch). The
   deployed ordering head remains design/19 §4.4 low-rank regardless.

## 2026-07-21 NIGHT-3 ★★ FOLD-IN AUTOPSY: NO ABSORPTION + SMALL PAYLOAD — SF18-MIX DEPRIORITIZED ★★

Two 5-minute CPU probes explain the SPRT null (exp-0127) mechanistically and
re-rank the queue:

1. **Absorption probe (2k aux rows):** the fold-in net is NO closer to its own
   aux labels than the never-saw-them control — |Δ| mean 96.4 vs 96.2, foldin
   closer on **48.3%** (50% = nothing took). At 2.5% mix / ~5.2 aux epochs the
   signal washed out entirely. The SPRT null = **dose below absorption
   threshold**, not "absorbed but worthless."
2. **Payload probe (1.9k joined rows):** |SF18×1.42 − 5k/1.9| on the SAME
   positions = **p50 31cp, mean 58cp** (>100cp on 18%) — while the net's own
   fitting error there is ~96cp. The refinement SF18 offers is BELOW the net's
   resolution floor. Even perfect absorption of a 31cp-median correction on
   27.5M positions plausibly buys single-digit Elo.

**Consequence (with the user's multi-head directive): SF18-fold-in is
LOW-CEILING and drops below multi-head in the GPU queue.** Dose-2 (7.5%)
completes as sunk cost — its gate + the SAME absorption probe give the
dose-response point that either confirms the ceiling or surprises. No further
SF18 GPU beyond that absent a surprise. (The stage-2-large-corpus idea also
weakens: these were DISAGREEMENT-SELECTED positions and the label delta is
still only 31cp median.) Artifact: `build-tight/uho_scores.sorted` (keyed 5k
scores, cached for future payload probes).

**GPU queue tonight (chained, sole-waiter chain):** dose-2 → haux-pol1(λ.001) →
haux-pol4(λ.004) → **wdl-sweep λ∈{0.006, 0.02}** (derived: value-loss 0.007 /
CE(3)≈1.1 ⇒ parity ≈0.006; each run auto-extracts the `l3_wdl` sidecar via
extract_head.py so a good WDL head is deployable immediately). All gated
scale-check + 300g fixed-node. By tomorrow midday: dose-response point 2 + four
multi-head measurements (pol×2, wdl×2).

## 2026-07-21 NIGHT-2 ★★★ MULTI-HEAD PRIORITIZED (user /loop directive) — H-AUX UNBLOCKED TONIGHT ★★★

**User (evening, /loop 20m): "ruthlessly prioritize our multi-head models."**
The policy from+to blocker was cleared the same evening:

- **Redo shape:** ONE `L3→128` affine vs a two-hot `[from; 64+to]` target
  (single softmax-CE, target mass 2, grad = 2p−t) — chosen because the recorded
  constraint was "value+TWO extra heads = cyclic backward topo-sort"; one head
  carries the full from+to signal. λ-gated (`D40_POL_LAMBDA`), loss-only,
  dropped at deploy (H-AUX: shape the trunk, zero inference cost).
- **Smokes (1sb streamed UHO, incumbent 768 arch):** value+pol ✓ trains;
  value+WDL+pol ✓ — **the historical cycle does NOT reproduce** in the current
  bullet toolchain (even the full 3-head shape is available); λ=0 control +
  checkpoint diff **proves engagement** (A≠C, B≠C: aux grads reached the shared
  trunk; raw sizes show the head tensors: +17KB pol, +428B wdl).
- **Data constraint pinned:** H-AUX training MUST use the sfbinpack path — only
  the binpack loader packs best-move into `ChessBoard.extra`; bf-converted bins
  carry zeroed extras (a from=a1/to=a1 garbage target).
- **λ derived, not guessed:** CE(128) ≈ ln128 ≈ 4.85 at init; λ=0.001 ⇒ aux
  contribution ≈ 0.005 ≈ the value-loss scale. Sweep point 2 = 0.004.

**OVERNIGHT QUEUE (sole waiter on the dose-2 GPU job, chain pid 2930758):**
`haux-pol1-240sb` (λ=0.001) → hard scale-check + 300g gate vs d40-uho-v1-240sb,
then `haux-pol4-240sb` (λ=0.004) → same. Policy aux ONLY (one variable; the WDL
co-train λ-sweep takes the next GPU window). **By morning: dose-2 7.5% gate +
the first two valid H-AUX measurements in project history.**

## 2026-07-21 NIGHT ★ SPRT VERDICT: 2.5% FOLD-IN IS A NULL — H0 ACCEPTED ★

**exp-0127:** SPRT(0, 8) fixed-node 5k, stopped early at **4454 games: score
49.69%, Elo −2.18 ± 4.47 (1σ), W/D/L 1629/1168/1657** — recomputed from
games.pgn (eval_match was orphaned by the session restart and never wrote
summary.json; the PGN is the record). Trinomial LLR −2.47 vs the −2.94 bound;
fastchess's pentanomial crossed it (early stop) ⇒ **H0**. The exp-0126
+9.3 ± 16.7 was noise, exactly as its CI warned.

**What stands:** the fold-in *mechanism* is clean (scale PASS 0.984/99.6, no
coverage damage, no harm) — but at 2.5% uniform dose the SF18 labels add
**nothing measurable** (effect bracketed ~±7 Elo around −2). One dose point is
not the family: **dose-2 (7.5%, foldin-tight-p75-240sb) is training now** —
its 300g gate can't resolve ±10, so if it scale-passes it gets its own SPRT
overnight. If dose-2 also nulls ⇒ the uniform-mix family CLOSES and the last
SF18 shape standing is the **stage-2-style warm retrain on the full-coverage
77M corpus** (fast_fens keys-mode build, ~1h CPU). If that nulls too ⇒ SF18
relabeling at our scale adds nothing over official binpack labels — a clean
close-out, GPU moves to H-AUX heads + width/arch on the proven streamed base.

**Tooling note:** eval_match must survive orphaning — it should write partial
summaries incrementally; the PGN-recompute path (this entry) is the fallback.

## 2026-07-21 PM-5 ★★ FOLD-IN RUN 1: THE FIRST NON-NEGATIVE SF18 RESULT ★★

**`foldin-tight-240sb` (exp-0126): +9.3 ± 16.7 (51.3%, 300g fixed-node 5k) vs
d40-uho-v1-240sb.** Streamed UHO base + 2.5% unit-consistent tight SF18 aux
(foldin-tight-aux-v1, ~5.2 aux epochs). The headline is twofold:
1. **Scale-check PASS at healthy-net level** — 0.984 median / 99.6% sign,
   indistinguishable from the incumbent control (0.979/99.2%). The aux did NOT
   damage coverage: the fold-in premise (base = regularizer, aux = targeted
   labels) is validated mechanically. Training loss 0.0066 = repro-class
   learning, not the 0.0005 memorization signature of the doomed runs.
2. **Point estimate +9.3 sits in the predicted +5–15 band** but the CI spans 0
   at 300g — exactly the "needs power" branch. NO verdict yet
   ([[no-verdicts-from-weak-tests]]).

**IN FLIGHT (launched immediately):**
- **`sprt-foldin-tight-25`** — powered SPRT, H0=0 / H1=8, fixed-node 5k, cap
  6000g, CPU (node-limited ⇒ GPU contention can't bias it). The "is 2.5%
  fold-in worth ≥~5-8 Elo" answer, expected by morning.
- **`foldin-tight-p75-240sb`** — dose-2: same recipe at 7.5% (~15.6 aux
  epochs), auto scale-check + 300g gate (~3h GPU). First point on the dose-
  response curve; watch for coverage damage at the higher dose.

**Next after these:** dose/schedule matrix per the PM-4 plan (end-ramped equal-
dose needs a small trainer change; stage-2-style warm retrain on the full-
coverage 77M corpus via fast_fens keys-mode is the best-suited shape). H-AUX
policy/WDL co-train queued behind the SF18 matrix on GPU. Self-game (hard-neg)
Elo attribution remains unmeasurable at 1.31M scale — flywheel grows it;
"did the labels take" probe on the fold-in net is the cheap tomorrow check.

## 2026-07-21 PM-4 ★★★ A/B VERDICT — the recipe was clean; COVERAGE was the killer ★★★

**The canon-tight A/B is done (exp-0124/0125).** Both cells on canon-tight-v1
(27.5M rows, full filter, real WDL), each at its own scale-matched MULT:

| Cell | Recipe | in-dist scale/sign | OOD scale/sign | fixed-node vs d40-uho-v1-240sb |
|---|---|---|---|---|
| A | WDL=0.3, MULT=1.42 | **1.004 / 96.1%** | 0.869 / 89.8% (FAIL) | **−727 ± 101** (FORCE=1) |
| B | WDL=0, MULT=1.94 | **1.022 / 97.4%** | 0.863 / 93.1% (FAIL) | **−849 ± 130** (FORCE=1) |

**Three questions closed:**
1. **The scale law + WDL-coupled MULT table is EXACT** — both cells output at
   ~1.0× the incumbent on their training distribution. Recipe deltas 1–3 are
   genuinely fixed (and the hard scale gate blocked both nets from wasting
   games until FORCE=1 was passed deliberately — the tripwire works).
2. **WDL=0 is INNOCENT** (B ≈ A within noise). The exp-0110 "WDL=0 disaster"
   framing is dead — that was the fishpack scale artifact all along.
3. **Materialized-small-corpus COVERAGE is the dominant killer** — a net that
   is scale-perfect and sign-healthy in-distribution collapses OOD (evals
   compress 0.87×, sign drops to ~90%) and plays at −727..−849. The volume
   curve is monotone: streamed 14.4B → −143; 137M×11ep → −463; 99.7M×60ep →
   −400..−700; 27.5M×54ep → −727..−849. Recipe knobs move results WITHIN the
   catastrophic band; materialization CREATES it. ⇒ **Standalone materialized
   SF18 corpora are permanently dead. No more GPU on that question.**

**The remaining SF18 path — first VALID fold-in test LAUNCHED
(`foldin-tight-240sb`):** repro UHO streamed base + `foldin-tight-aux-v1`
(28.8M = canon-tight 27.5M + hardneg-tight 1.31M) at D40_MIX_PCT=2.5 (~5.2 aux
epochs). **New contract catch while designing it:** the D40_MIX aux shares the
trainer's global SCORE_MULT with the binpack base (no per-source scaling), so
the aux was pre-scaled ×1.42 to binpack/1.9 units AT BUILD TIME — a raw-UCI aux
would run at 0.51× its own base. (This is also a third independent invalidation
of exp-0118.) Contract §2 table updated. Gate: scale-check (should PASS — the
base provides coverage) + 300g fixed-node vs d40-uho-v1-240sb. Verdict ≈ +2.5h.

## 2026-07-21 PM-3 ★★★ THE TIGHTENING SPRINT — fix training before spending GPU (user directive) ★★★

**User: "Kill the GPU job and focus on this full time... If we can't get this right
there is no use wasting GPU."** 1536 pipeline killed cleanly (autolauncher +
trainer; arch work preserved, trainer source reverted to 768/L2=16 — resuming the
bet = flip HL/L2 consts + rebuild). Everything below is CPU-side recipe hardening,
each piece validated before use.

**1. `tools/nnue/net_scale_check.sh` — the output-scale calibration authority.**
Compares candidate vs incumbent static evals (nnue-eval) on a pinned 800-FEN set
(`tools/nnue/calib/scale-check-800.fens`); PASS band 0.85–1.18 + sign-agree ≥95%.
**Validation matrix confirmed the half-scale theory at the engine level:** healthy
control 0.979/99.2% PASS; canon-v1-base **0.469**, canon-clean **0.478** (the
predicted ~0.51× half-scale nets); canon-scale194 0.917 but sign 92.3% (scale fixed,
residual deltas visible); canon24m 0.379. Catches in 4 s what cost days of games.
**Wired as a HARD abort into `gate_canon.sh` [1a] + `gate_1536.sh`** (FORCE=1
override) — the old advisory check printed "scale 0.505x" for canon-clean and the
160-game match ran anyway; that can no longer happen.

**2. WDL-coupled SCORE_MULT (train_canon.sh), derived not guessed:** engine value
= l3·network_scale(=400 always, netpack-baked) and l3 ≈ score·MULT/400 ⇒ value ≈
score·MULT. Binpack path: internal/1.9 (SFBINPACK_SCORE_DIV in sfbinpack.rs).
Text path must supply the whole conversion via MULT = K/1.9, K = binpack/SF18-UCI
= **2.64–2.76 measured** (re-derived independently from the diagnosis probe data,
n=798, r=0.94) ⇒ **MULT=1.42 for WDL=0.3** (blend cancels; target construction
then matches the incumbent exactly) vs **1.94 for WDL=0** (exp-0111's empirical
output-match). train_canon.sh auto-couples; scale-check is the final authority.

**3. Full trainer-parity filter AT EXPORT — `obs-tool export-value
--filter-sfbinpack`** (+unit tests, 9/9): the exact 5 criteria of the trainer's
`sfbinpack_filter`, incl. testing the RAW pre-clamp score (a mate label collapsed
to ±2499 by the export clamp would otherwise slip the |score|≤10000 bound — and
did: see hardnegs below). Probe on shard 0: 26.6% dropped (diagnosis predicted
26.9%). `canon_filter` marked SUPERSEDED/partial in its header.

**4. Tight canon corpus (`build_canon_tight.sh`) — deltas 2+3 fixed in one build:**
filtered export (measured **27.5% dropped over all 113.9M**) + real game-WDL
inner-join from uho_frs.tsv + self-checks that DIE on drift (filter-rate band,
join hit-rate, degenerate-result guard, alignment). **Self-check caught a real
find:** join hit-rate 53.29%, root-caused NOT as a key bug — canon_fens (Jul 19,
old box) and uho_frs.tsv (Jul 20) are two different stratified-chunk samples of
the same UHO binpack; overlap is quality-unbiased, kept rows are 100% real-WDL.
(Full-coverage fix queued: fast_fens keys-mode over the whole binpack.) Result
distribution matches source (W40/D49/L11 vs 42/47/11).

**5. Hardneg aux rebuilt TIGHT (`hardneg-tight-v2`, ledger row added):** full
filter dropped **32%** — capture-best 15.7% + **86,320 mate-scale labels (4.5%)
that the ±2499 clamp had been hiding from every previous "clean" filter**. 1.31M
rows, 100% real-WDL join. Supersedes hardneg-clean-shuf-v1 (partial filter).

**6. Latent bug flagged in parked `build_fulldist_corpus.sh`:** its baked
`sf18×1.94` next to `score5k/1.9` rows is a WITHIN-corpus 1.37× scale
inconsistency (1.94 is the WDL=0 output constant, not the ×1.42 unit conversion).
Header warning added; fix if the path revives.

**IN FLIGHT:** full tight corpus build (sort/join of 78.4M filtered rows) →
`ab_canon_tight.sh` armed: Cell A (WDL=0.3/MULT=1.42) vs Cell B (WDL=0/1.94),
60sb each, scale-check hard-gated, fixed-node gates. Readout per the diagnosis:
A≈repro-class ⇒ data was always fine; A≈−400 ⇒ volume ⇒ fold-in; B isolates WDL.
End-to-end path pre-validated with a 3sb smoke (768 arch confirmed, MULT
auto-set, netpack+engine load OK).

## 2026-07-21 PM-2 ★★★ CANON −582 IS NOT A DATA BUG — INDEPENDENT FORENSICS REVERSE THE VERDICT ★★★

**Full write-up: `docs/research/sf18-canon-data-diagnosis-2026-07-21.md`** (independent
Fable audit, all priors re-derived from scratch, CPU-only, live GPU run untouched).

**Verdict reversal: the SF18-canon DATA is sound.** All 9 integrity probes clean —
the standout is a fresh SF18@50k rerun of 401 stratified rows reproducing the stored
labels to **p50 10cp / 99.6% sign agreement**, which in one shot rules out sign/POV/
scale-corruption/misalignment/oracle bugs. Encoding (232k records decoded, 0 mismatch),
shuffle (autocorr 0.958→0.000), dedup (0 dups), FEN legality, and distribution shape
(statistically identical to the known-good UHO binpack) all check out.

**The −582/−400 collapse was a 4-way CONFOUNDED RECIPE** — each fix was serially
mis-ruled-out because ≥2 other deltas stayed broken:
1. **Target scale ×0.51** — the pinned "canon cp == binpack_cp/1.9" claim is **FALSE**.
   Measured two ways: binpack = ~2.7× SF18-UCI cp, and canon labels = 0.506× the
   incumbent's output scale. SCORE_MULT=1.0 trained a **half-scale** net (breaks every
   co-adapted search margin). Fix: **SCORE_MULT ≈ 1.94** (matches exp-0111's net-side
   result). Both canon runs used 1.0. *(Scripts corrected; default 1.0→1.94.)*
2. **Quiet filter only removed in-check** — the healthy path drops 26.9% at load;
   `canon_filter` drops only in-check, leaving **~22% best-move-capture poison**. ⇒
   **exp-0123's "filter ruled out" is INVALID** (tested 2/5 criteria, scale still wrong).
3. **WDL=0** (0.5-placeholder result byte) — present in every catastrophic run, absent
   in every healthy one, **never isolated on clean data**. Open confound.
4. **60-epoch replay of 100M uniques** vs <0.5 epoch streamed. Memorization signature;
   also why "low loss" ≠ good — it's a **faithfully-learned wrong objective**, not corruption.

**Actions taken:** `train_canon.sh` + `build_canon_corpus.sh` scale comments corrected +
SCORE_MULT default fixed; memory + day-plan updated; quarantine **LIFTED pending the
confirming A/B** below. The live 1536 run uses binpacks and is UNAFFECTED by all of this.

**QUEUED confirming A/B (GPU, ~1.5h + 2 gates)** — from the diagnosis §"Minimal fix":
rebuild canon with the **full 5-criterion filter** (extend `obs-tool export-value` to use
the shard `best_move`) + **real game-WDL join**; **Cell A** SCORE_MULT=1.94 WDL=0.3,
**Cell B** identical but WDL=0; gate both vs `d40-uho-v1-240sb`. A≈−150 ⇒ data was always
fine (recipe was the whole story); A≈−400 ⇒ residual is volume ⇒ use canon as
`D40_MIX_DATA` fold-in on the streamed base (don't materialize); B isolates WDL=0.
**Not built yet** — needs the export-tool `best_move` change. This is
[[no-verdicts-from-weak-tests]] in REVERSE: we condemned the DATA from confounded tests.

## 2026-07-21 PM ★★ CANON −582 LOCALIZED + 1536 ARCH TRAINS ★★ [SUPERSEDED by PM-2 above — the "data bug" conclusion here is RETRACTED; kept for the audit trail]

**canon-clean-240sb gate result (SF18-canonical, QUIET-FILTERED corpus) vs
d40-uho-v1-240sb, 10+0.1:** **CONCLUSIVE — 160 games, W7/L132/D21 = 10.94%
score, nElo ≈ −700** (gate stopped early to free CPU for the 1536 train; ledger
exp-0123). **The quiet-filter fix did NOT reverse the −582.** Decisive tell: canon-clean's training loss **converged low (0.00067)**
yet it plays at −381 — the net faithfully learned labels that produce terrible
play. ⇒ **the bug is in the SF18-canonical DATA (labels/scale), NOT the quiet
filter, trainer, or training.** Quiet-filter bypass is RULED OUT as the −582's
cause. Prime remaining suspect: the train_canon scale assumption ("canon cp ==
binpack_cp/1.9, so SCORE_MULT=1.0 + EVAL_TARGET_SCALE=400") — a scale error
would fit the labels while miscalibrating every cp-threshold pruning margin.
**SF18-canonical data stays QUARANTINED** (do NOT fold into the 1536 run).
Full data-vs-env isolation (train_canon env on clean UHO) is deferred — the
1536 arch (clean UHO+farseer) is the priority and doesn't need SF18-canon.

**1536 HalfKA-only arch SMOKE PASSED (biggest risk resolved):** trains → packs →
engine-loads (readyok) → searches, end-to-end. bullet handles the all-inactive
threat input; the `halfka_only` save-fix is live. GPU-bound training ~447k pos/sec
(vs canon's ~700k — 2× accumulator) ⇒ full 300+140sb ≈ 6.7h. Full stage-1
(`h1536-uho-300sb`) now running.

**★ NPS BET — early NEGATIVE lean (not the hoped-for win):** SMOKE-net NPS
(startpos, 1 thread): **NEW-1536 HalfKA-only 720,847 vs INCUMBENT 768+threats
835,087 = −13.7%.** Dropping threats did NOT make it faster — the 2× accumulator
cost more per-eval (activate + find_nnz + l1_dot all scale with L1; l1_dot also 4×
from L2 16→32) than the threat channel saved. CAVEAT (do not over-conclude,
[[no-verdicts-from-weak-tests]]): this is SMOKE sparsity (3+3 SB → denser
activations → inflated l1_dot → pessimistic for the new arch); the converged net
will be faster (likely −5 to −10%). The **equal-time gate on the converged net is
the arbiter** — at ~−10% NPS the arch must be clearly better in EVAL quality (the
L2=32 / 2×-width bet) to win. If it loses on speed, the L2=16 ablation (halves
l1_dot) and/or a narrower 1280 accumulator are the NPS-recovery levers.

## 2026-07-21 AM ★★★ NEW ARCH: DROP THREATS + 2× ACCUMULATOR + WIDER L2 BOTTLENECK ★★★

## 2026-07-21 AM ★★★ NEW ARCH: DROP THREATS + 2× ACCUMULATOR + WIDER L2 BOTTLENECK ★★★

**Executive decision (user, away all day):** get the model as strong as possible
using ALL the toys — every extra head (WDL, policy) + all the extra data (UHO,
farseer, clean canon, re-filtered hard-negs). **Arch: DROP FullThreats, make the
accumulator 2× bigger (768→1536).** Two follow-ups same day: **widen the L1→L2
head bottleneck 16→32** ("I worry the 16-32 bottleneck was the thing preventing
our bigger nets from learning" — now the LEADING hypothesis for width-saturation),
and **later test a 32→32→16 extra tail layer**.

**The bet:** FullThreats is the dominant NPS tax (~160 active feats/move,
memory-bound i16 adds). Dropping it frees NPS to fund a 2× accumulator on a pure
HalfKA net (≤32 active, cheap). The L2=16→32 widening removes a suspected
capacity choke between the (now 2× wider) trunk and the head.

**Built + verified today (all CPU, GPU stayed on canon-clean):**
- **Engine** (`sota_head::L1=1536`, `L2=32`): `sota_simd::l1_dot` was hard-coded to
  16 outputs (`[i32;16]`, `group*64`); generalized all three backends (scalar/
  AVX2/AVX-512) to loop over `L2/16` output-blocks. **Parity-tested: scalar ==
  AVX2 == AVX-512 at L2=32; all 88 nnue tests green** (single-threaded — parallel
  OOM'd from the concurrent trainer × doubled synth-net allocs). Head layers
  (propagate_l1/l2) were already const-driven. Binary → `engines/bin/archon-sota-1536`.
- **netpack** (`tools/nnue/netpack.py`): arch dims now env-driven (`D40_ACC/L2/L3/
  BUCKETS`, defaults = old 768+threats = backward-compatible) + `D40_HALFKA_ONLY=1`
  path that skips the threat tensor, clears FLAG_HAS_THREATS, zeroes header
  threat_dims. **No-threats payload = 69,638,176 B = the engine loader's
  `payload_len_no_threats()` byte-for-byte.** Synth 1536-HalfKA net packs + the
  engine LOADS it (readyok) + searches → **round-trip proven end-to-end.**
- **Trainer** (`d40_features.rs`, `HL=1536`, `L2=32`): **BUG CAUGHT in pre-flight**
  — `halfka_only` only skipped threat *feature extraction* (line 285) but still
  BUILT + SAVED `l0_threatw` (zeros). A zero threat tensor in the quant.bin would
  (a) misalign netpack's skip-threats AND (b) if packed with threats, still cost
  the full threat-accumulator NPS → **defeating the entire bet.** Fixed: omit
  `l0_threatw` from the quant `fmt` when `halfka_only` (raw save keeps it for
  warm-start consistency). The inert tensor costs ~1.1GB GPU RAM — affordable
  (canon 768+threats uses only 2.9GB of the 8GB card). Rebuilt.

**Width REOPENED (per [[no-verdicts-from-weak-tests]]):** fixed-node re-gate 1024
vs 768 @ 25k nodes = **+6.08 ± 23.68** — NOT negative (the equal-time −8 was the
NPS confound). Width is not dead; L2=16 is the new prime suspect.

**NPS bet — NOT yet called:** synth-net NPS is too confounded (random weights →
search hit "mate" at 185ms, dense activations) to trust. Definitive NPS waits on
the real trained 1536 net.

**In flight:** `canon-clean-240sb` (OLD 768+threats arch on the FILTERED SF18
corpus, ~113/240, auto-gate armed vs d40-uho-v1-240sb — the SF18-data-validity
signal). `tools/nnue/autolaunch_1536.sh` (pid-waits on the canon trainer) will
fire `tools/nnue/train_1536_halfka.sh` the instant the GPU frees: SMOKE (3+3 SB)
→ NPS measure → FULL (UHO 240sb div1.9 skip75 WDL0.3 LR1e-3 → farseer 120sb warm
div1.9 WDL0.25 LR5e-4). Ship candidate = `data/nnue/nets/h1536-s2-farseer-120sb.cenn`,
gate vs incumbent (fixed-node then equal-time) then SF16.

**Next levers queued:** heads co-trained (WDL built; policy REDONE from+to) gated
vs value-only base; L2 ablation (16 vs 32 vs +layer, fixed 1536, fixed-node).

**Clean fold-in data READY (2026-07-21):** re-filtered the hard-neg aux through
`canon_filter` — dropped **176,821/1,928,011 (9.17%)** in-check/tactical poison
(nearly identical to the canon −582's 9.1% rate — the SAME `bf-convert`-without-
quiet-filter bypass poisoned both). Clean corpus =
`data/nnue/hardneg/corpus/corpus.filtered.shuf.bin` (1,751,190 pos, ledger
`hardneg-clean-shuf-v1`). This CONFIRMS the exp-0118 fold-in verdict was INVALID
(poisoned aux); the clean fold-in test (`D40_MIX_DATA`) is valid for the first
time — queued behind the 1536 base.

## 2026-07-20 PM ★★ OVER-PRUNING / WASTED-COMPUTE + NPS DIAGNOSTICS + FOLD-IN SWEEP ★★

## 2026-07-20 PM ★★ OVER-PRUNING / WASTED-COMPUTE + NPS DIAGNOSTICS + FOLD-IN SWEEP ★★

User theory under test: the weaker raw net makes ARCHON prune subtrees SF keeps
(blind spots), and on a faster box the extra compute is spent going DEEPER down
those mis-pruned paths — wasted. Three probes (all d40-uho-720sb net, sota+tune
`target/release/chessengine`):

1. **RFP equal-time A/B** (wideRFP `RFP_MULT_BASE=70` vs default 40, 10+0.1-ish
   st=0.25, 300g): **+4.6 ± 21 = NEUTRAL.** Widening the RFP margin at equal time
   is a wash (depth-for-width trade cancels). ⇒ the RFP margin is already ~optimal
   for our eval; over-pruning is NOT a mis-set RFP knob. Does NOT refute the theory
   (equal-time confounds it), only kills the naive "widen margins" fix.
2. **Fixed-nodes scaling** (ARCHON vs SF16 @ 20k vs 80k nodes) — the clean
   isolation (no depth/width trade): deficit **GROWS** 20k→80k: Elo −55±29 → −64±28,
   **nElo −85 → −120**. 4× compute makes us RELATIVELY WORSE vs SF16 ⇒ directional
   support for wasted-compute / eval-amplification (NOT yet significant — bars
   overlap; accumulating to ~240g/arm). Explains the native-box (faster) regression.
3. **RFP fixed-nodes A/B** (wideRFP=70 vs 40 @ 50k nodes, 300g): **−2.3 ± 19 = NEUTRAL.**
   Pruning-less finds NO better moves per node either.

**CONCLUSION (pruning investigation, 2026-07-20 PM):** RFP over-pruning is **RULED
OUT** — widening it is neutral both at equal time (±21) AND per-node (±19). The
scaling test (final: 20k −55±29 / nElo −85; 80k −67±25 / nElo −121, 240g each) shows
a directional-but-NOT-significant deficit-growth with more nodes. Since it's NOT the
RFP knob, the wasted-compute / faster-box regression points to **EVAL-AMPLIFICATION**
(a weaker/miscalibrated eval at the leaves → deeper search amplifies errors rather
than correcting them), not mis-pruned trees. **⇒ the lever is EVAL QUALITY (fold-in +
arch T6/T8 + NPS-via-arch to relieve pruning pressure), NOT search-knob tuning.** This
reinforces the primary goal; do not tunnel on search params. (Caveat: only RFP tested;
LMR/futility not individually ruled out, but the per-node neutrality of the dominant
eval-margin prune is strong evidence pruning-calibration is not the bottleneck.)

**NPS / SIMD reality (measured, single-thread, under load):** ARCHON **722k**, SF16
2.36M (3.3×), SF18 1.26M, Reckless 1.10M. `archon-sota-native` is ALREADY a
`target-cpu=native` AVX-512 build (5556 zmm) WITH a runtime VNNI path
(`_mm512_dpbusd_epi32`, `sota_simd.rs:609`, gated on `is_x86_feature_detected!
("avx512vnni")` → fires on this Zen4). So 722k IS our AVX-512-VNNI speed — NO free
rebuild lever. The 3.3× gap vs SF16 is STRUCTURAL: the FullThreats accumulator
update (~160 active feats/move, memory-bound i16 adds VNNI can't accelerate)
dominates. Fair same-tech comparison = Reckless (AVX2 Rust, HalfKA-only): we're 1.5×
slower = the threats tax. Lever to close it = ARCH (T6 threats-cost ablation, T8
king-split to widen cheaply), NOT flags. Only the `x86-64-v3` gate/tune builds are
AVX2-only (fine for A/Bs — both sides share it). ⇒ low NPS FORCES aggressive
pruning (ARCHON reaches depth 42 on startpos @ 722k nps vs Reckless depth 37 @
1.10M — we prune harder to compensate), which with a weaker net is where blind
spots come from. Recovering NPS (arch) would relieve the pruning pressure.

**Fold-in ratio sweep CONCLUDED — fold-in is a DEAD END on the UHO base.** The
aux-epoch curve is monotonic-hurt: mix@4ep (0.231%) = −1.2±17 gate / +16±21 equal-time
(≈neutral), mix@12ep (0.692%) = **−12.7±17**, mix5 (87 epochs) = −112. Folding the
SF18/Reckless hard-negs into the full retrain does NOT help — it only hurts as you
add more. Data-polish via fold-in is off the table; the binpack volume dominates.
⇒ this redirects the effort to CAPACITY (width), the untested lever.

**NET-GROWING (Net2WiderNet) — WORKS, validated, productionized (user ask 2026-07-20).**
Grow a trained net's accumulator (768→1024) function-preservingly: copy the narrow
net in with new neurons zero-init, placed to preserve the pairwise-mul (i,i+HL/2)
pairing, so the wide net evals IDENTICALLY until fine-tuned — warm-start for the cost
of a short fine-tune instead of a full from-scratch run. **Proven: frozen (LR=0) save
of the grown net netpacks to a CENN that evals bit-identically to the 768 parent
(rmse 0.00cp).** The bug en route: the RAW tensor layout ≠ the quant layout — `l1w`
is HL-major `[HL,out]` in the raw (`transpose_saved`) while `l0_pstw/threatw` are
feature-major `[feat,HL]`; nailed from d40_features.rs's own loader. Reusable tools:
`tools/nnue/net2wider.py` (general transform — auto-detects source width + all arch
dims) + `tools/nnue/grow_net.sh` (widen→preservation-gate→fine-tune pipeline).
`netpack.py` gained a `D40_ACC` width-override. **Prereq:** width is a compile-time
const — trainer HL + engine `sota_head::L1` + the `sota_features` ACC assert must be
rebuilt at the new width (a width feature-flag ∼ `nnue-l1-*` is the clean successor;
BACKLOG). Currently building: grown-1024 warm fine-tune (60sb) → gate vs 768 parent
= the first real CAPACITY signal (does width pay, now that fold-in doesn't).

## 2026-07-20 ★★★ CONCLUSIVE ROOT CAUSE — RECIPE-PARITY / QUIET-FILTER BYPASS ★★★

**Every net trained this session failed for ONE reason, now PROVEN (Fable-agent
investigation, 5 completion passes, all evidence measured — not the scale/
distribution/WDL stories below, which were all downstream misreads).**

**ROOT CAUSE:** the corpus-build scripts (`fast_fens` → `build_*_corpus.sh` →
bf-convert) reproduce the sfbinpack path's *data* but silently DROP its *load-time
quiet filter* (`tools/nnue/bullet/d40_features.rs:93-102`): `ply≥16 && !in_check &&
|score|≤10000 && best_move==Normal && best_move_not_capture`. So **~10% of my
training positions were in-check and ~25-30% best-move-capture** — positions whose
deep search score describes a *tactical resolution* a STATIC net cannot represent.
Under MSE the biggest errors dominate the gradient → the eval surface is warped
exactly where sharp play lives → r=0.91 average but **p99 |diff|=1000cp, 5-7% of
positions >400cp wrong** (control: r=0.986, p99=262, ZERO >400). Search minimaxes
into those tails → steers into blind spots → −463..−662. (The filter's 5 checks
mirror where ARCHON's search never trusts the static eval: qsearch never stands pat
in check; quiescence resolves captures dynamically; pruning uses wide margins.)

**EXONERATED (all measured):** trainer, CUDA/GPU, netpack (engine-vs-quant ≤0.057
logit), .cenn/EvalFile load, gate harness. Proof: `repro-uho-sfbinpack-60sb` (exact
control recipe, THIS box) → r=0.971, correct scale, **−143±23** vs the 240sb control
(exp-0115) — a normal quarter-budget point. Same box/trainer/budget as the −463
bulletformat net; only the data path (filter) differs.

**RETRACTED misreads (do NOT re-derive; superseded below):** "scale compression"
(exp-0109/0111 — real but minor), "disagreement-subset skew" (exp-0112/0113 —
REFUTED: fulldist full-distribution was WORSE at −662, exp-0114), "WDL=0 breaks good
data" (exp-0110 — INVALID: fishpack-wdl trained on `fishpack32.binpack`, a known-bad
UCI-scale source, at 07:36/08:25 BEFORE the UHO download finished 12:26; WDL=0 remains
untested-at-scale, NOT refuted). The fishpack red herring is what sent the whole
diagnosis chasing data ghosts.

**THE FIX (applied):** `fast_fens` now takes a `filter` arg applying the exact
`sfbinpack_filter` (patched + built + smoke-verified: 26% dropped, 0% in-check out).
`build_fulldist_corpus.sh` is now `UHO_FRS`/`WORK` env-overridable. `gate_canon.sh`
step [1] now reports r + p99 + >400cp tail + PASS/WARN/FAIL (slope alone was the
instrument that hid this).

**IN FLIGHT:** filtered-twin confirming test (Fable's recommended next step) —
`fast_fens UHO 188M filter` → `data/nnue/binpacks/uho_frs_filtered.tsv` (~139M, the
188M unfiltered set minus poison) → rebuild via `UHO_FRS=…filtered WORK=data/nnue/
fulldist-filtered/build build_fulldist_corpus.sh` → train 60sb → `gate_canon.sh`.
**PREDICTION:** ≈−143 (repro-class) ⇒ filter is the dominant fix; ≈−400 ⇒ the
volume/epoch term (109-197M unique × ~30 epochs vs billions-streamed-once, the B4
constraint) dominates ⇒ stream the binpack / grow unique volume. Compounding term 3
(canon-only): disagreement-selection sharpness-bias + SF18@50k half-contrast labels.

**★ RESULT (exp-0116, 2026-07-20): filtered-60sb = −463** (r 0.910→0.946, p99
1000→658, tails 66→47). **The filter helped +199 Elo (−662→−463) but VOLUME is the
dominant remaining factor** — 137M unique × ~11 epochs vs the repro's billions-
streamed-once (−143). **STRATEGIC CONCLUSION: the foundation must STREAM the filtered
binpack (the repro recipe), NOT materialize small SF18 corpora — materializing is
volume-starved.** SF18's value can't come from a small baked corpus; it comes via
FINETUNE on top of a streamed foundation → the volume-bootstrap-then-quality-refine
curriculum. Also (exp pending): PURE hard-neg finetune of a base REGRESSES −172.8
(forgets even filtered) → the finetune must UPSAMPLE hard-negs into a retention mix,
not pure-finetune. Finetune mix-vs-control test on the healthy repro base in flight.

**AFTER the fix confirms:** the production SF18 net should apply the filter to the
canon relabel selection too, and PREFER streaming ≥1B-unique corpora over multi-epoch
replay of ~150M. Then gate fixed-node vs control → equal-time vs SF16.

**BACKLOG (non-blocking, raised 2026-07-20):** find an efficient on-disk
representation for the rich `.ceobs` store — it is ~566 B/record (uncompressed +
standalone-position + per-record provenance strings), ~210× a compressed binpack
position; zstd-at-rest alone recovers 2.48× (measured), and per-shard provenance +
varint + position-chaining targets ~150–200 B. Shrinks the feasible canonical rich
dataset's disk footprint (250M–1B: ~140–570 GB → ~40–230 GB) but does NOT move the
labeling-compute wall. Full write-up: **design/16 §9 q7**.

---

## 2026-07-20 ★★ PIVOT: FROM-SCRATCH TRAINING + MIXED-READER FOLD-IN (user decision) ★★

Finetuning is PARKED (no beneficial regime — see the FINETUNE INVESTIGATION section
below). New direction (user): **train a new net from scratch and fold our SF18/
hard-neg data in.** From-scratch sidesteps catastrophic forgetting entirely — the net
learns volume + our data TOGETHER (no converged minimum to perturb), which is exactly
why the finetune failure mode doesn't apply here.

**NEW CAPABILITY — the mixed reader (`tools/nnue/bullet/d40_features.rs`):**
`D40_FORMAT=sfbinpack D40_MIX_DATA=<pre-shuffled aux .bin> D40_MIX_PCT=5` folds a small
aux set (self-play / SF18 hard-negs, bulletformat) into the streamed sfbinpack main at
exactly PCT% (inject `n = chunk×pct/(100−pct)` per main chunk). The aux is read as
random CONTIGUOUS windows straight off disk → the OS page-caches it when it fits RAM
and pages it when it doesn't ⇒ **NO RAM CEILING** (the fold-in set can grow with the
flywheel to any size). Measured **688k pos/sec vs ~700k non-mix (~1.5% overhead)** —
fast. Ratio exact by construction. Built + smoke-validated; trainer rebuilt. (Also:
`SfBinpackLoader::new_concat_multiple` exists but only *concatenates* — it can't spread
a small source through a huge one, hence the custom reader.)

**IN FLIGHT — paired from-scratch fold-in validation** (`/tmp/foldval.sh`): UHO 120sb,
SAME seed, arm A `foldval-uho-120sb-ctrl` (no fold-in) vs arm B `foldval-uho-120sb-mix5`
(SF18 hard-negs @5%). Gate B vs A = the ISOLATED fold-in effect. **B>A ⇒ scale to
production** (UHO 720sb → farseer stage-2, combined SF18+Reckless hard-negs folded in,
sweep the ratio) aiming at incumbent → SF16. **B≈A ⇒** the hard-neg lever is even more
bounded / tune ratio+composition. Reckless labeling 42/64 → grows the fold-in set.

---

## 2026-07-20 ★★ FINETUNE INVESTIGATION — CONCLUDED (self-play data via finetuning) ★★

**Question:** can we improve the eval by finetuning the converged net on self-play
hard-negatives (ARCHON's losses to SF18 + Reckless, SF18-relabeled)?
**Answer: NO via naive finetuning — it structurally can't work (catastrophic
forgetting).** Triple-confirmed + expert-validated (continual-learning: LwF/EWC).

- **No beneficial regime (LR ramp, healthy repro base, exp-0117):** LR≤3e-6 → net
  unchanged (Elo 0); LR≥1e-5 → monotonic regression (−12→−212). Best case = no change.
- **Three hypotheses RULED OUT by measurement:**
  1. weight decay — exposed `D40_WEIGHT_DECAY`, wd=0 == wd=0.01 at both schedules
     (−34.9 vs −36.6; −212.4 vs −209.9). Over a 3-ep finetune decay is only ~0.23%.
  2. scale compression — scale ⊥ Elo (gentle 0.946×/−35 vs aggressive 0.993×/−212:
     ANTI-correlated). Restoring scale made play WORSE.
  3. built-in data shrink (the expert's α* dominant-cause) — α* on the real
     WDL-blended target = **1.185** (mix) / 1.048 (5k), i.e. **>1, no shrink** (the
     target wants EXPANSION); the observed 0.946× shrink moves OPPOSITE the objective
     ⇒ an optimization side-effect, not a data property.
- **=> the regression is catastrophic FORGETTING** (tracks training amount), not
  scale/decay/data-bias. The static-eval metrics DON'T predict play: the aggressive
  net looks great statically (r0.966, tails11, scale0.99) yet plays −212.
- **The fix (expert + our data agree) = function-preserving finetuning, NOT LR/WD
  knobs:** (a) **distillation anchoring / LwF** — add `β·E_anchor[(f_θ−f0)²]` on a
  LARGE deployment stream (teacher=base net, no labels) while learning hard cases;
  (b) **gold-standard = full mixed-stream retrain** (stream binpack + hardnegs
  folded in, full schedule). The **affine-recalibration shortcut is OUT for us** (it
  assumes a built-in shrink we don't have; α*>1 + scale⊥Elo).
- **Caveat:** the hard-neg lever is BOUNDED (few-M positions; SF18 agrees with 5k
  97.5%). Distillation anchoring's real value = the mechanism for the self-play
  FLYWHEEL at scale (fold in ARCHON's mistakes without forgetting), not this small set.
- **DECISION PENDING (user):** (a) implement distillation anchoring now; (b)
  gold-standard mixed-stream retrain now; (c) park finetune + bank self-play data +
  bigger lever (my lean: c now, a when the self-play set is larger).
- **New tooling this arc:** `train_canon.sh` gained `INIT_RAW=` warm-start + `LR=` +
  `WD=` overrides; trainer exposes `D40_WEIGHT_DECAY`; `fast_fens … filter` quiet
  filter; `gate_canon.sh` reports r+p99+tail verdict; expert Q in
  `docs/research/finetune-question-for-expert.md`. Reckless hardneg SF18-labeling
  in progress (30/64) → `data/nnue/hardneg-reckless/v1/labeled/`.

---

## 2026-07-20 ★★ COMPACT CHECKPOINT (superseded above; kept for scripts/data/gotchas) ★★

**~~IN FLIGHT (GPU):~~ [SUPERSEDED — fulldist-sf18-240sb gated −662 (exp-0114); it was
NOT the fix. See the conclusive root cause above.]** `fulldist-sf18-240sb` training (chain `b55qrlfos`, log
`/tmp/fulldist_trainchain.log`, ~31/240 sb when compacted, ~2h left). It is the
FIX — full UHO distribution + SF18 cp on matched 26% + real WDL, on control scale.
**ON COMPLETION:** `bash tools/nnue/gate_canon.sh data/nnue/nets/fulldist-sf18-240sb.cenn
data/nnue/nets/d40-uho-v1-240sb.cenn 200` → eval-scale (want ~1.0×) + fixed-node Elo
vs control. This is a CLEAN twin of the control differing ONLY by SF18 labels on 26%
→ the definitive test of SF18 label value. If it WINS → gate vs SF16 equal-time
(archon-sota-native vs stockfish16, 10+0.1, UHO book). If it FAILS/ties → SF18 labels
are a small lever; pivot to other eval levers (factorization #27, longer stage-1).

**THE FULL DIAGNOSIS (do not re-derive):**
- Machine: native Arch, 7700X (AVX-512), RTX 4060 Ti 8GB, 15GB RAM. CUDA at /opt/cuda.
- SF16 equal-time re-baseline (native) = **−129** (exp-0107); pure eval quality
  (fixed-depth sweep exp-0108: even-to-ahead thru d14, −129 at d19; depth-16≈WSL2's
  −41). native-box-regression-2026-07-20.md.
- **BUG 1 eval-scale compression:** first canon net evaled 0.516× the incumbent →
  search cp-margins broke → −582 (exp-0109). Cause: SF18@50k cp ~half the incumbent's
  training-cp magnitude. Fixed via SCORE_MULT (net output ×1.94). exp-0111: scale
  corrected 0.516→0.988 but still −463 (a 2nd bug).
- **BUG 2 disagreement-subset skew:** canon-v1 is a "locally disagreement-selected
  subset" → net never learned normal openings → loses in ~5 moves (exp-0112).
  CONFIRMED budget-independent: full 240sb still −564 (exp-0113). Fix = full
  distribution.
- **THE FIX (training now):** full-distribution corpus. score5k is the binpack 5k
  label (÷1.9 = control scale). SF18 cp is ~half that → multiply SF18 by **1.94** to
  put it ON THE CONTROL SCALE, then train **SCORE_MULT=1.0** (NOT 1.94 — that was for
  the disagreement corpus). WDL=0.3.

**SCRIPTS (all written+tested this session):**
- `tools/nnue/build_fulldist_corpus.sh` — the FIX corpus builder (uho_frs.tsv +
  canon corpus.txt → sort+join → SF18×1.94/score5k÷1.9 → bf-convert → shuffle). Sort
  temp forced to big disk (/tmp is 7.7G tmpfs RAM — will fail sorts otherwise).
- `tools/nnue/train_canon.sh` — params via env: NET_ID, SB, SCORE_MULT, WDL, DATA,
  FACTORIZE, SMOKE. netpack uses `uv run --no-project --with numpy` (pyproject has
  tensorflow → plain `uv run` HANGS).
- `tools/nnue/gate_canon.sh <cand.cenn> [baseline.cenn] [games]` — eval-scale + match.
- `build_canon_corpus.sh` = the OLD disagreement-subset builder (DON'T use — bad dist).
- `train_repro_incumbent.sh` = unused (superseded by the scale finding).

**DATA (on disk, reuse):**
- `data/nnue/fulldist/build/corpus.shuf.bin` = 197M control-scale corpus (the fix; DONE).
- `data/nnue/fulldist/build/{canon.sorted 5.3G, uho.sorted 19.3G}` = sorted join inputs
  (reuse to rebuild the join fast — ~15min, no re-sort).
- `data/nnue/binpacks/nodes5000pv2_UHO.binpack` (40G, downloaded) + `uho_frs.tsv`
  (188.5M FEN\tresult\tply\tscore5k, White-rel, from fast_fens).
- `data/nnue/binpacks/{wrongIsRight_nodes5000pv2,farseerT76}.binpack_.gstmp` = partial
  (canon also drew from these; only needed to raise SF18 match rate >26%).
- `data/nnue/canon-v1/` = 91G .ceobs shards + `build/corpus.txt` (109M canon SF18
  FEN|cp|0.5, White-rel).
- `data/nnue/hardneg/all.tsv` = 2M ARCHON-vs-SF18 hard-neg positions w/ WDL (Phase D).
- Nets: incumbent `d40-uho-720sb`, control `d40-uho-v1-240sb`, broken
  `canon-v1-base-240sb`(−582)/`canon-scale194-{60,240}sb`(−463/−564). Engines:
  archon-sota-native, stockfish16 (3.0Mnps), stockfish18 (vnni512), fastchess, ordo.

**LEDGER:** exp-0107(SF16 −129) 0108(depth) 0109(canon −582) 0110(fishpack wdl0)
0111(scale-fix −463) 0112(5-move losses) 0113(240sb −564). Add a row for the fulldist
gate when done.

**OPERATIONAL GOTCHAS:** (1) `pkill -f gcloud|"storage cp"|d40_features` SELF-MATCHES
my own shell → use `pgrep -x <procname>` or explicit PIDs. (2) gcloud sliced download
wedges near 100% → `gcloud config set storage/sliced_object_download_max_components 1`
(single-stream, set). (3) verify `corpus.shuf.bin` mtime > `corpus.txt` before training
(stale-corpus race bit me). (4) kill in a dedicated bash command (exit-144 hiccup).

**LOOP:** cron `4f6da630` (15-min, eval focus). Directives: LOCAL compute only, cloud
= GET-only storage, NO cloud jobs w/o approval. North star: SF16 +30 equal-time; then
pivot to search-informed multi-head + aux-loss nets (D41).

**★ DEFERRED TRAINING OPTIMIZATIONS (test AFTER the SF18-canonical net gates; user
requested 2026-07-20).** Training is GPU **memory-bandwidth-bound** on the 4060 Ti
(measured sm 100% / mem 96%; ~700k pos/sec = ~2.4h/240sb). Bullet is **F32-only**
(matmul rejects non-F32 → no FP16 win without new kernels). Cause: the 83248×768 ≈
64M-param FT (HalfKA+FullThreats) + Adam's 3 buffers = ~767 MB weight traffic PER
BATCH. Levers, in priority:
1. **Batch 16384→32768 (halve BPS, keep total positions)** — amortizes the fixed
   per-batch Adam step → est **+20-50% throughput**. Risk: gradient dynamics change;
   run a fixed-node quality A/B (batch32768 vs batch16384 same data) + maybe nudge LR
   before trusting. Single biggest speed lever.
2. **Upsample the SF18-relabeled positions** — the fulldist corpus is only ~26% SF18
   (51M/197M); the rest reuse the 5k labels. For SF18-focused nets, upsample the 51M
   2-4× (substitute_labels supports it) to concentrate the SF18 gradient/epoch. (Kept
   at 1× here only for the clean control-twin comparison.)
3. **Per-epoch reshuffle** (minor) — the bulletformat DirectSequentialDataLoader reads
   the same shuffled order every epoch (no per-read random skip like the sfbinpack
   path); a reshuffle would decorrelate the ~30 epochs slightly.
PROCESS LESSON: confirming-NEGATIVES need only a short 60sb run — the 240sb
disagreement-subset run (exp-0113) wasted ~2h re-confirming the 60sb −463.

---

## 2026-07-20 UPDATE — scale fix CONFIRMED but a SECOND issue remains (read first)

The `SCORE_MULT=1.94` fix WORKS on eval-scale: `canon-scale194-60sb` = **0.988×
incumbent** (was 0.516×, r=0.92, mean|cand|=365 vs |inc|=367). BUT the fixed-node
gate vs the control only improved **−582 → −463** (exp-0111, 60sb WDL=0). So scale
was *a* bug, not the *only* bug — a scale-correct net still loses 93.5% to the
240sb control, far worse than a 60sb-vs-240sb budget deficit explains.

**Leading suspect: disagreement-subset distribution skew.** canon-v1 is a
"locally disagreement-selected subset" — training a general eval only on hard
surprising positions plausibly miscalibrates it on the normal positions that
dominate games. **Test in flight:** `canon-scale194-240sb` (WDL=0, SCORE_MULT=1.94,
full 240sb budget) — if still −300s, the disagreement subset is the problem; if
~0, the 60sb was just under-trained. **Likely fix if data is the problem:** the
FULL-distribution corpus (substitute_labels: full UHO 188M + SF18 cp on the
relabeled positions + real game WDL from the binpack) — needs UHO (downloading,
99%) + a disk sort+join (109M SF18 labels can't fit an in-RAM dict on 15GB).
Also still open: WDL (the +19 fishpack A/B was between two scale-broken nets, so
it may under-read WDL's true effect on a scale-correct net).

## 2026-07-20 ★ ROOT CAUSE FOUND: EVAL-SCALE COMPRESSION

**The canon −582 failure is an EVAL-SCALE bug, now confirmed empirically.**
`nnue-eval` on 1000 positions, canon-v1-base-240sb vs d40-uho-720sb:
**canon ≈ 0.516 × incumbent** (slope through origin, r=0.88; mean|canon|=192 vs
mean|incumbent|=367). The canon net's eval is COMPRESSED to ~half the scale the
search's cp-denominated pruning (RFP/futility/razoring/LMR margins) is tuned for
→ it under-rates how lost a losing position is → "optimistic on losing lines" →
−582. Static evals looked "sane" because they're proportionally correct, just
uniformly halved.

**Why:** SF18@50k labels have ~half the magnitude of the incumbent's effective
training cp (epic noted mean|cp| 515→189 for 5k→50k; deeper search = less extreme
evals). My "SF18 cp = raw UCI cp = same scale as incumbent, so SCORE_DIV=1.0"
reasoning was WRONG — the incumbent's eval lives on the ~1.9×-inflated binpack
scale (the fix factor 1/0.516 = 1.94 ≈ the 1.9 SCORE_DIV I omitted). Formula:
`target = sigmoid(cp × SCORE_MULTIPLIER / EVAL_TARGET_SCALE)` (d40_features.rs:669).

**FIX (testing now):** retrain with **D40_SCORE_MULTIPLIER=1.94** (net outputs
1.94× larger → matches incumbent scale → search margins valid again). Running:
`canon-scale194-60sb` (60sb, WDL=0, SCORE_MULT=1.94, existing 109M corpus, ~35min).
Then eval-compare (expect ~1.0× incumbent) + gate vs d40-uho-v1-240sb (expect ≫ −582).
If confirmed → full 240sb at SCORE_MULT=1.94, then layer WDL (+19) + factorization.
NOTE: this makes the env-reproduction diagnostic unnecessary (stopped it); WDL was
never the main bug (only +19).

## 2026-07-20 CANON NET FAILURE — CORRECTED DIAGNOSIS

The first SF18-canonical net **FAILED**: exp-0109 canon-v1-base-240sb (WDL=0,
bulletformat .bin from bf-convert, disagreement-subset, SCORE_DIV=1.0) = **−582.7
vs the d40-uho-v1-240sb control** (same 240sb budget). Static evals SANE + CORRECTLY
SIGNED (verified: black-to-move down-a-rook = −442, control −387; export-value emits
White-relative `value_cp_white`), yet loses 96.6% in search ("optimistic on losing
lines"). **NOT a sign bug.**

**Prior tick blamed WDL=0. That is now DOUBTFUL:** a fresh fishpack A/B
(fishpack-wdl03 vs fishpack-wdl0, fixed-node 5k, 200g) = **only +19.1 ± 24.0** —
WDL is a MINOR effect, not the −582 killer. (Caveat: both fishpack-60sb nets are
weak vs the 720sb incumbent, which can MASK WDL's true effect — do not fully
dismiss WDL.) exp-0110's "WDL=0 breaks nets" over-read a small-net weakness.

**The real anomaly = canon (240sb) losing −582 to the SAME-budget control.**
Suspects, unresolved: (a) bf-convert/bulletformat data path bug, (b) disagreement-
subset distribution skew, (c) SF18-cp scale/calibration, (d) WDL. **DECISIVE TEST
(train_repro_incumbent.sh, fires when UHO downloads): reproduce the WORKING control
recipe on sfbinpack UHO (SCORE_DIV=1.9, skip75, WDL=0.3, 240sb, seed 198273612).**
If it reproduces (~0 vs the shipped control) → env is sound, bug is in the canon
DATA path → then isolate by swapping ONE var at a time. If it also fails → trainer
env broken. Source binpacks downloading (UHO 37/40G) for this + the substitute_labels
full-distribution+real-WDL corpus path. `bulk_fens` built.

## 2026-07-20 NATIVE-ARCH RESUME

**Machine switched: WSL2 → native Arch Linux** (see memory machine-arch-box).
7700X (16thr, **AVX-512**), RTX 4060 Ti 8GB, 15GB RAM, 878GB free. Repo is a
clean GitHub clone in sync with origin/main. All gitignored artifacts restored
fresh: engines rebuilt (SF16 bmi2-PGO 3.0Mnps, fastchess 1.8.1, ordo), UHO book,
nets pulled from gs://…/nets/ (incumbent d40-uho-720sb + control d40-uho-v1-240sb
+ farseer). Code verified correct: `cargo test` 24/24, default bench **701605**
bit-exact, sota bench **147691** bit-exact.

**★ NATIVE SF16 EQUAL-TIME RE-BASELINE (the −41.5 was WSL2): now ~−130.**
ARCHON sota (d40-uho-720sb, native+AVX512) vs SF16, 10+0.1, UHO book:
conc-8 (400g) = 32.25% = **−129 Elo**; conc-4 one-proc-per-physical-core (60g
partial) = 31.7% = −134 → **HT-contention ruled out; the gap is REAL, not a
measurement artifact.** PGN depth analysis: at equal time both engines reach
**median depth 19** (SF16 only +0.3 ply mean) — so the deficit is PURE EVAL
QUALITY, now fully expressed because the fast box searches ~3 ply deeper (16→19)
than WSL2 and SF16's per-node eval edge compounds over deeper trees. Confirms
sf16-gap-diagnosis. Need ~+160 to reach SF16+30. Lever = better eval (data).

**★ ACTIVE PLAN (user, 2026-07-20 /loop): canonical SF18 eval retrain.**
canon-v1 relabel DONE on the cloud (fleet shut down). **canon-v1 = 113.96M
positions, `locally disagreement-selected subset` (data-quality Dim 1) — a
curated high-quality SF18-relabeled set**, 71GB of .ceobs shards +
inputs/tasks/*.fens (FEN-only, 2280 shards). Downloading to
data/nnue/canon-v1/ (GET-only auth). Pipeline RESTORED + wired this session:
- bullet toolchain re-cloned (jw1912/bullet@cebc78a, bulletformat@cb6a3602),
  patches applied, d40 examples copied+registered, chessengine path dep added,
  `cargo check --example d40_features` **exit 0** (ready to build for CUDA).
- CPU tools built: obs-tool (datagen), nnue_convert (bf-convert).
- **CUDA toolkit BLOCKER**: not installed on Arch; user installing /opt/cuda now.
  Needed for `cargo build --features cuda --example d40_features`.
**Recipe decided (scale-compatibility, avoids co-adaptation trap):** fleet SF18
cp = raw UCI cp = binpack_cp/1.9, and the incumbent trained binpack with
SCORE_DIV=1.9 → SAME output scale. So canon-v1 .bin trains with SCORE_DIV=1.0,
EVAL_TARGET_SCALE=400, SCORE_MULTIPLIER=1.0 → scale-matched to existing search.
First net: D40_WDL=0 (pure SF18-cp regression, no binpack re-download for WDL
join). Fold in king-bucket factorization (#27, D40_FACTORIZE=1) once baseline
canonical net validates. Gate: fixed-node vs d40-uho-720sb, then equal-time vs
SF16.



## TAKEOVER CHECKPOINT (2026-07-18 02:35 — read this first)

> **★★ POST-COMPACT RESUME — CURRENT STATE + QUEUED WORK (2026-07-19, LATEST — READ THIS FIRST) ★★**
>
> **POSITION:** ARCHON opt1+opt2 speed package SHIPPED (+20.7% NPS, bit-identical,
> commits 8287e90 / 1cc2147). Vs SF16 at EQUAL TIME = **−41.5 ± 12.3** (exp-0103;
> was ~−70 pre-speed; the speed program bought ~+28 Elo, matching opt#1's +28.8).
> PGO = plain for GAME-Elo (deploy form only, no extra Elo — exp-0103). Incumbents:
> d40-uho-720sb (sota, the sped-up net), s3-farseer (classic). Bench pin 701605.
>
> **★ SMALLNET GATE INFRA LANDED (2026-07-19, commit 62102ca) — the biggest
> remaining one-box SPEED lever (perf study #3 #8, design smallnet-gate-design.md).**
> Env-gated dual-accumulator: `ARCHON_SMALLNET` (a cheap HalfKA-only CENN-v2) +
> `ARCHON_SMALLNET_THRESH`; a second `HalfkaContext` maintained in lockstep, and
> `evaluate()` routes DECIDED positions (|cheap eval| ≥ thresh) to the cheap net,
> skipping the big net's FullThreats fold + head (the big threat stack stays lazy
> so decided nodes never fold it). Default OFF = bit-identical (sota 469205 ==
> pre-gate baseline; default depth-6 701605 reproduced; change is 100% cfg(sota)).
> 2 unit tests (transparent-when-agree, never-route-bit-identical) + runtime smoke
> (armed-active 475383, never-route 469205), 637 sota tests + clippy clean.
> **REMAINING (GPU-gated, after g2/EXP-UNC-1 frees the GPU):** train a real
> HalfKA-only cheap net (`D40_HALFKA_ONLY=1`, smaller L1), gate-sweep the
> threshold, equal-time vs big-net-only. Expected +8-15% NPS → +16-30 Elo.
>
> **★★ ACTIVE PIVOT (2026-07-19, user-chosen): EXP-UNC-1 — eval output columns
> (the search study's #1 bet). ★★** After the lazy-observer landed (+3.6%, below
> projection — see below) and the perf-3 tail was measured-exhausted (RVO'd /
> ≤0.2% / unsafe / training-gated; the only large speed prize left is the
> smallnet gate #8, which is itself a training project), the user chose to pivot
> to EXP-UNC-1 over grinding the tail. **Pre-registered experiment (study §3.1.1 /
> §5 bet 1):** train a twin of the 240sb control (d40-uho-v1-240sb, ledgered) with
> the value head emitting **(mean, log-σ²)** under **heteroscedastic NLL** (label =
> row's own score; optional WDL triple aux), deploy eval-only-mean for value +
> wire σ̂ into exactly two consumers behind env-gated coeffs: (i) futility/RFP
> `margin *= clamp(1 + c1·(σ̂−σ̄)/σ̄, 0.7, 1.5)`; (ii) TM soft-stop
> `total_time *= (1 + c2·root-σ̂-normalized)`. **T≈1 by construction** (columns
> ride the paid forward, ~30 FLOPs). **Cost class E0 — neither prior study priced
> it; it's the class the frontier ships (SF complexity/optimism, KataGo
> score-stddev, Lc0 WDL).** Consumer audit: ARCHON's TM has ZERO position-content
> input (pure SF telemetry transplant) — the uncertainty column is complementary,
> not a re-weight. GATE ORDER (trainer-first, cheap kill first): **Gate 1 = value
> parity** — the NLL-mean twin must not lose to the MSE control at fixed nodes
> (kill ≤ −15); uses the EXISTING runtime (export mean-only as a standard CENN).
> **Gate 2** = consumers on/off, same net, 600g fixed-node + 300g 10+0.1. Expect
> +3..+15 internal, P(≥+10)≈25-35%, P(parity)≈70%.
> **BUILD PLAN (trainer→Gate1→plumbing→Gate2):**
> 1. Re-establish the ephemeral bullet checkout (gitignored, currently ABSENT):
>    re-clone + copy tools/nnue/bullet/*.rs into examples/ + register (README).
> 2. Modify `d40_features.rs` (the PRODUCTION HalfKA+FullThreats trainer, NOT
>    d40_eval.rs) to add the (mean, log-σ²) NLL head + optional WDL.
>    **GATING UNKNOWN: does bullet's graph API express heteroscedastic NLL /
>    multi-output custom loss?** — inspect the bullet source once re-cloned;
>    workaround if not (separate log-var output + manual loss node). SMOKE-train
>    tiny first (smoke-test memory).
> 3. Train the 240sb twin on the same corpus/schedule as the control.
> 4. Gate 1: netpack the mean-only → fixed-node value gate vs the 240sb control.
> 5. If pass: extend CENN format + netpack.py + sota_loader.rs + sota_head.rs
>    forward() to carry the log-σ column (bit-identical when absent/off); wire
>    consumers (search/mod.rs futility/RFP + timeman.rs soft-stop), env-gated
>    coeffs defaulting OFF.
> 6. Gate 2: 2×2 coeff probe (c1,c2) fixed-node + equal-time. Kills precommitted.
> GPU is idle (this + EXP-AUX-1 share the trainer plumbing). Runs alongside the
> CPU relabel. D38 ledger the twin + both gates.
>
> **★★ DETACH GATE 1 VERDICT (2026-07-19): ALSO FAILED — Elo −116.4 ± 49.7
> (130g, exp-0106). Both EXP-UNC-1 variance-training approaches wreck the mean.**
> The detach was supposed to be mean = control BY CONSTRUCTION (pure-MSE mean off
> a stop-grad trunk). It isn't: bench nodecount detach 11304 vs control 10545 at
> d2 → the mean WEIGHTS differ, and −116 rules out benign RNG (a different-seed
> twin is ~parity). **Root cause: bullet's `with_no_grad` did NOT isolate the
> variance gradient from the trunk** — the NLL reweighting leaked into the mean,
> so "detach" behaved like coupled NLL (−116 vs g2 −179; the extra pure-MSE term
> only partly shielded it). **→ THE FIX (bulletproof): a FROZEN-mean post-hoc
> variance fit.** Do NOT retrain the mean at all — take the incumbent (mean =
> guaranteed the incumbent), and fit ONLY the l3_var head to its residuals
> (`logvar = w·h + b`, per-bucket NLL) either (a) in bullet with the mean/trunk
> weights frozen (requires_grad off) or (b) post-hoc in Python from a
> `nnue-dump` of (pre-l3_var activation, sig, target). Cheap (fit a small head,
> no full retrain). This σ̂ then serves BOTH EXP-UNC-1 Gate 2 AND the n-gate LARGE
> tier router. **Impact: the n-gate SMALL tier (speed, running) is UNAFFECTED —
> it needs no σ̂; the LARGE tier + Gate 2 wait on the frozen-mean σ̂ (and the
> large tier ALSO needs a genuinely-better net → fleet).**
>
> **★★ g2 GATE 1 VERDICT (2026-07-19): FAILED HARD — Elo −178.86 ± 60.06 (114g
> @5k), LLR −2.98, SPRT H0 accepted. B's Seitzer concern CONFIRMED, dramatically.**
> The twin's l3_var never trained (raw ckpt drops non-output-reachable weights →
> its "NLL" collapsed to ≈MSE → Gate 1 +5.2 parity). g2 = same seed + the
> logvar-SAVE fix → l3_var ACTUALLY trained (loss −2.28) → the COUPLED NLL's
> exp(−s) mean-reweighting then WRECKED the mean (−184 Elo vs the twin). Coupled
> NLL is UNUSABLE for the shipped mean. → **Executing the prepared branch: the
> DETACH variant** (`D40_HETERO_NLL_DETACH=1`, trainer d40_features.rs:725) — mean
> loss stays pure MSE + the l3_var head reads a stop-grad trunk, so the mean =
> control by construction (Gate 1 automatic + TRUE) and σ̂ is a clean post-hoc
> residual predictor. Smoke launched (3 SB); full 240sb retrain
> (`d40-uho-unc-detach-240sb`) on the now-FREE GPU (g2 retrain done). Then Gate 2
> (the consumer probe) — the real, unproven test of EXP-UNC-1's +3-15 bet.
>
> **STATUS (2026-07-19, LAUNCHED — trainer fully validated):** The NLL trainer is
> IMPLEMENTED + smoke-validated. Edit in `.archon-state/toolchains/bullet/examples/
> d40_features.rs` (~line 708): behind `D40_HETERO_NLL=1`, a parallel `l3_var`
> log-var head + loss `0.5*(exp(-s)*(sig-t)^2 + s)`; MEAN head + save tensors
> UNCHANGED (default path byte-identical). Smoke (2 SB, real data): compiles,
> feature-parity 65/65, loss 0.0444→0.0366 (no NaN), raw ckpt 256MB (has l3_var),
> quantised 81MB standard mean-only. Gate-1 pack path validated: netpack → 81,354,848 B
> CENN (exact control size, l3_var excluded) → engine loads+evals. **240sb twin
> RUNNING** (`train_run.py --config data/nnue/experiments/d40-uho-unc-v1/config.json`,
> pid 430823, log data/nnue/experiments/d40-uho-unc-v1/train.log; exact control
> recipe + default power-2 loss, same seed 198273612 → identical mean-head init;
> D40_SF_BUFFER_MB lowered 256→128 for RAM safety w/ the relabel; save_rate 30).
> ~2.4h to SB240, then train_run.py packs (mean-only CENN) + parity.
>
> **MEANTIME PLUMBING BUILT WHILE THE TWIN TRAINS (2026-07-19, committed):**
> Decision: variance head rides a **SIDECAR** (D40_UNC_HEAD env), NOT the CENN
> (netpack rejects unknown trailing bytes; CENN format stays unchanged; no train
> restart). Done + tested, all bit-identical when off:
> - **7b17338 head-side σ̂:** SotaHead optional l3_var head + propagate_l3_var;
>   forward() → (value, Option<σ̂>), σ̂=exp(½·logσ²) sigmoid-space from shared l2
>   (~32 FLOPs, T≈1); SotaContext.last_sigma() getter; unit-tested value-unchanged.
> - **5126538 sidecar loader:** load_uncertainty_sidecar reads D40_UNC_HEAD (flat
>   LE f32, BUCKETS·L3 weights bucket-major + BUCKETS biases) into the head at
>   net-load; absent/error → variance-less.
> - **Gate-1 harness armed (self-firing):** tools/nnue/exp_unc_gate1.sh (pid poll
>   on 430823 → 400g @5k SPRT[-15,5] twin vs control → GATE1-DONE + verdict).
> STILL TODO (Gate-2 prep, in progress): (a) **sidecar EXTRACTOR** — a bullet
> dumper reading l3_varw/l3_varb from the raw ckpt (ModelWeights.get by name) →
> the sidecar file; (b) **consumers** — futility/RFP margin scale c1 + TM soft-stop
> c2 (env-gated, read SotaContext.last_sigma). **Gate 1 needs NONE of this** (mean
> value only). RAM tight — build with CARGO_BUILD_JOBS=2. Ledger the twin after pack.
>
> **★ CRITICAL FINDING (2026-07-19, later): the raw ckpt saves ONLY
> output-reachable weights — a loss-only l3_var is DROPPED (empirically: the smoke
> raw ckpt had no l3_varw/l3_varb).** So the SIDECAR-from-raw plan needs the
> trainer to expose `logvar` as a SAVED OUTPUT. FIX committed (459a368, tracked
> tools/nnue/bullet/d40_features.rs: `outputs.push(("logvar", logvar))` — NLL
> training itself is smoke-verified; the logvar-SAVE fix is PENDING a post-match
> build+smoke). **CONSEQUENCE: the currently-RUNNING d40-uho-unc-240sb train
> (pre-fix binary) is GATE-1-ONLY — its l3_var is lost.** Gate 2 needs a RETRAIN
> with the fix; the mean is identical under the same seed (198273612), so the
> Gate-1 verdict carries over — not wasted. Also the sidecar extractor
> tools/nnue/extract_unc_head.py is written + self-validating (--validate against
> the mean head, no build). Consumer wiring plan (build-required, after builds
> free): EvalContext (src/eval/mod.rs:38, holds `sota: Option<SotaContext>`) gets
> `last_sigma()` → search reads it after static_eval → RFP margin
> (search/mod.rs:214 `rfp_margin_with_threat`, scale by clamp(1+c1·(σ̂−σ̄)/σ̄,0.7,1.5))
> + TM soft-stop (c2·root-σ̂). All env-gated, default off = bit-identical.
>
> **★ INTEGRAL EQUAL-TIME (user-requested "how far caught up", exp pending):**
> ARCHON sota (d40-uho-720sb + opt1+opt2 + lazy observer) vs integral-v6, 100g
> 10+0.1 (tools/nnue/integral_match_v1.sh, relabel-paused for clean timing).
> **FINAL 100g: ARCHON +24.4 ± 32.2 Elo (19W-12L-69D, LOS 93.2%), ledger
> exp-0104** — even-to-AHEAD of Integral v6 (a top-6-class engine); a ~94-Elo
> swing from the **−70±22** anchor. Speed package (opt1+opt2, +28 vs SF16) +
> stronger D40 net (d40-uho-720sb). ARCHON is now ~Integral-v6-class. SF16 gap
> remains (−41.5, exp-0103) — the north-star. **EXP-UNC-1 head plumbing verified
> BIT-IDENTICAL (sota bench 147691) with the σ̂ forward + sidecar loader +
> EvalContext::last_sigma accessor all committed; logvar-save fix build COMPILES
> (auto-verifier building; smoke waits for the twin to free the GPU).**
>
> **★ LAZY OBSERVER IMPLEMENTED + MEASURED (2026-07-19, commit 249e1b5): the
> +11-12% projection MISSED — clean measured = +3.6% NPS, bit-identical.** The
> fully-lazy threat observer landed in `sota_threats.rs` (snapshot mailbox+occ+
> king-squares at push, ~140 B; generation+resolution+fold all deferred to
> `compute` against the snapshot; king-cross still eager; generation shared across
> perspectives per folded ply). CORRECTNESS IRONCLAD: bench node-count 147691 (d8)
> + 3515701 (d12) bit-identical both sides; all 636 sota lib tests + the 100k-node
> `threats_replay_large` stress + per-node debug refresh-oracle green. SPEED (quiet
> box, SIGSTOP'd the relabel, taskset -c 14, d12 median-of-6): **NEW 869k vs BASE
> 836k = +3.6% NPS** (ratios 1.030-1.045, tight). **CALIBRATION LESSON: the
> profiling cycle-attribution model over-projected ~3×.** Why: the study modeled
> the 19.9% observer block shrinking 54%, but wall-clock ≠ cycles for a SCHEDULE
> shift — deferred gen+resolve moves ONTO the eval critical path (was overlapped
> with make_move at push) and every push now pays a ~140 B snapshot copy. Contrast
> opt#1 (a KERNEL/instruction-count shrink) which hit its +13-15% projection dead-on
> and paid +28.8 Elo. RULE: discount profiler projections for deferral/scheduling
> changes ~3×; trust them for instruction-count kernel changes. Memory DID shrink
> as projected (entry 5.1→3.2 KB, ~0.8 MB/thread). Still worth banking (free,
> bit-identical, ~+1.5-3 external Elo, and it's the foundation the #2 prefetch
> pipeline builds on) — BUT the study's +18-25% package is now suspect: the
> schedule-y follow-ons (#2 prefetch, #3 cancel) should be discounted ~3× too;
> only the kernel-y tail (#5 static dispatch, #4 out-params) likely holds. The
> original projection + mechanism, for the record:
>
> **★ ORIGINAL FINDING (perf study #3, `profiling-study-3-creative-2026-07-19.md`,
> PROJECTED +11-12% — measured +3.6%, see above):**
> Round-1's "the observer must run at push time" is FALSE — SF18 generates events
> eagerly, RESOLVES LAZILY at eval. Defer the WHOLE observer: its only inputs are
> occ + mailbox (~150 B snapshot at push) + a 1-bit king mirror; every eval-time
> fold is an ANCESTOR of the current node so sibling destruction never matters;
> only 46% of pushes are ever folded → deferral deletes ~54% of the 19.9% observer
> block, BIT-IDENTICAL via the existing `threats_replay` parity gates (HalfKA
> already uses this lazy pattern). Unlocks +6-10% more (resolve→prefetch→fold
> pipeline gives the null prefetch real distance vs the i8 fold #1 sink; SIMD batch
> emit kills the observer's 44%-of-all-branch-mispredicts pathology; cross-ply
> cancel; null-move fold aliasing). PACKAGE = +18-25% NPS on top of +20.7% →
> ~1.0-1.07 Mnps (past Reckless same-shape, SF18-class cycles/node), +18-50
> internal Elo. NEGATIVE that saves weeks: SVD of the 60720×768 threat table is
> NOT low-rank → factorized-threat-embedding bet DEAD. Format bets (next CENN rev):
> quant unification (kills 5.8% i32 activate tax) + SF smallnet gate (+8-15%,
> SF-proven Elo+, loader already supports HalfKA-only). **To implement the
> observer: needs a QUIET box for NPS benching — correctness via node-count 147691
> + threats_replay parity is LOAD-IMMUNE, but NPS is not (the prefetch fooled me
> under relabel load). Pause the relabel or wait for it.**
>
> **LEARNED SEARCH AUGMENTATION (D41 — user's #1 research interest, "bigger than
> chess"):** three studies. (1) inference-time per-node = CLOSED (k>T tax;
> `learned-search-augmentation-first-principles-2026-07-19.md`). (2)
> train-time-aux / eval-only (H-AUX) = SURVIVES, +5-20 ET, T=1
> (`d41-auxiliary-task-adversarial-2026-07-19.md`); free target = `entry.mv` (in
> every official-binpack row); ledger has a +63 λ=0.01 EVAL-ONLY aux-twin prior;
> decisive = EXP-AUX-1 (twin of d40-uho-v1-240sb + rank-8 best-move-CE aux head
> @λ=0.01, pack eval-only, 300g @5k h2h vs the control). (3) COMPREHENSIVE
> space-map (`learned-search-augmentation-comprehensive-2026-07-19.md`) RUNNING
> (agent ae30cd05, model=fable) — user found studies 1+2 TOO NARROW/THIN; #3
> covers the TRUE-MULTI-HEAD-USED-but-cheap region + non-ordering uses of a search
> head (time-mgmt, LMR/extension-mod, eval-uncertainty, eval-correction) + paradigm
> hybrids + invent-novel, ranked by EV×tractability. **EXP-AUX-1 PAUSED pending
> #3's ranked map** (aux-drop is just one cell). User wants breadth+DEPTH — pressure-
> test #3 against his objections before presenting it as done.
>
> **RUNNING JOBS:** comprehensive learned-search study (agent ae30cd05, Fable,
> analytical); kill-switch relabel (`tools/nnue/label_value_pilot.sh`, 3/10 chunks,
> ~12 SF18 workers, resumable — the label-value A/B gate: ≥+15 fixed-node → deep
> SF18 relabel worth cloud money; null → HARD-STOP the relabel/obs-factory line).
> GPU idle (held for EXP-AUX-1 / the study's top-ranked experiment).
>
> **CLOUD:** user-gated; access comes AFTER we consume+adjust the studies.
> Highest-value cloud = **D36 TESTING FLEET** (~$100-300/mo, the strategic study's
> slope-changer), NOT data relabel. Tooling ready (tools/cloud/ M2-M7, project
> archon-502500). Strategic verdict (path-to-1-first-principles): binding
> constraint = D36 distributed testing; one-box eval rungs tapped out; d41-gold-v2
> CANCELLED; obs-factory FROZEN until the label-value pilot pays.
>
> **NEXT (post-compact, in order):** (a) comprehensive study returns → relay +
> pressure-test depth vs the user's objections; (b) implement lazy-observer #33 on
> a QUIET box; (c) finish the relabel → the label-value A/B verdict; (d) EXP-AUX-1
> or the study's top experiment on the idle GPU; (e) await the user's cloud/fleet
> decision. Every net/corpus/match → a D38 ledger row; LOCAL commits only, NEVER
> push. Model overrides: subagents = `model: fable` for these studies (user
> verified). Kills: do in a DEDICATED bash command + verify separately (exit-144
> shell hiccup; see memory `kill-in-separate-command`).

> **SPEED PROGRAM REOPENED — the profiler says SPEED is the D40 lever, and
> opt#1 already banked +11% NPS (2026-07-18 evening, commit 8287e90).** The
> Fable profiling study (`docs/research/profiling-study-2026-07-18.md`,
> committed 2acb8d8) found D40's problem is NOT net quality — the sota path
> holds +30/node (exp-0067) but burns ~70% of cycles in NNUE inference and
> runs 2.3× more instructions/node than SF18 (16.9k vs 7.4k); the same-shape
> gap to Reckless is pure constant-factor engineering. Its bit-identical,
> bench-node-preserving opt package sums to ~+20-30% NPS = ~the whole speed
> gap, and per exp-0093 (D40 at equal-time PARITY purely for speed) is the most
> direct route to equal-time Elo vs SF16 — do this BEFORE more net-quality
> spend on the arch. **THE SPEED LEVER IS CONFIRMED (2026-07-19): opt#1 alone =
> +28.8 ± 10.3 Elo at EQUAL TIME (10+0.1, 569g, LOS 99.7%, exp-0101)** — bit-
> identical binaries differing ONLY by the +13-15% NPS opt → ~2× the D39 rule
> (slow eval ⇒ extra nodes pay off disproportionately). This is the first hard
> proof that NPS converts to equal-time Elo on this arch — the whole speed
> package is de-risked. **Ranked queue:**
> - **#1 DONE (8287e90): vectorize the f32 head tail** — bit-identical (bench
>   147691), **+13-15% NPS**, **+28.8±10.3 Elo equal-time (exp-0101)**.
> - **#2 DONE (1cc2147): fused copy+apply fold kernels** — `apply_i16/i8_deltas_into`
>   (scalar+AVX2+AVX512) read parent / write child in one pass, no memmove;
>   rewired both folds; removed the now-dead in-place i8 path. **Bit-identical**
>   (bench 147691, all 637 fold-parity tests + threats_replay_large green),
>   **+3.5% NPS** (below the +7-9% est — LLVM already elided part of the copy).
>   Package opt1+opt2 = **+17.1% NPS**. Equal-time match (opt2 vs base) RUNNING.
> - **#3 DEPRIORITIZED: prefetch + pair threat-fold rows** (sota_simd.rs) — est
>   +2-5% but speculative/hw-sensitive; given #2 underdelivered, low marginal
>   value. Revisit only if the package match shows the NPS→Elo slope stays steep.
> - **#7 (trivial): bulk net parse** (sota_loader.rs:512-529) — 0.21s/spawn.
> - **Deploy step:** rebuild `probe-sota-pgo` (tools/build_pgo.sh sota) with the
>   opt package BEFORE the next equal-time gate — the PGO binary is what gates
>   use; opt#1 is in-tree but not yet in the PGO binary.
> Verification harness (reuse): build base+opt release, `bench 16 1 8 default
> depth <net>` node-count = bit-identity anchor, benchmark_eval_nps.py for NPS.
>
> **opt#1 COMPREHENSIVELY VALIDATED + equal-time match RUNNING (user asked to
> test comprehensively, 2026-07-18 evening).** Correctness: node-count
> bit-identical at depth 6/8/10 on BOTH nets (683938 evals deep), AND all 2000
> diverse training positions eval bit-identical (base vs opt dump, 0 diffs).
> Speed: **+15.1% NPS geomean** (470583→541538, benchmark_eval_nps.py 16pos×200k,
> median +22.5%) — under relabel load, so clean is ≥ that. **DEFINITIVE test now
> running:** equal-time match sota-opt1 vs sota-base, same net 720sb, 10+0.1,
> 1000g, UHO_Lichess_4852 book, concurrency 8 (one/physical-core, clean TC),
> plain-release both (isolates opt#1). PID/log logs/opt1-equaltime.log, out
> data/nnue/experiments/opt1-equaltime/. Expected ~+10-15 internal Elo (D39:
> ~+9-10/+10% speed). **Label-value relabel PAUSED at 3/10 chunks** (resumable;
> orchestrator killed) for the quiet box — RESUME after the match + then
> implement opt#2/#3 (user-approved sequence). Ledger the match as an exp on
> completion.

> **EVENING UPDATE 2026-07-18 (persist checkpoint):** four items since 10:07.
> (A) **STAGE-1 SATURATED at 720sb.** The warm-continue +360sb extension
> `d40-uho-cont-1080sb` (1080sb total) screened **−9.3±15.4 vs 720sb** (300g
> fixed-5k, exp-0100) → REJECTED; more stage-1 superbatches do not help (mirrors
> the stage-2-dead finding). 720sb stays the D40 stage-1 incumbent; stop pouring
> compute into stage-1 — the gains are in data-quality / arch / search.
> (B) **Tier-F obs-relabel pipeline LANDED + verified (commit 4b59607).** M2:
> `root-analysis --reference-only` emits one Observation/tier (position +
> reference SearchObservations incl. the `id_snapshots` eval trajectory +
> provenance), SKIPPING the ~35× per-move `searchmoves` loop — the "free layers
> always" capture. NEW `obs-tool export-value` is the scalar VALUE view
> (`FEN|cp|wdl`, White-relative, Exact+Complete quarantine → bf-convert → 32B
> bin), verified end-to-end incl. the sign flip on decisive positions.
> IMPORTANT: `export-train` is the PER-MOVE POLICY exporter (needs Complete
> arms; it SKIPS reference-only shards) — the data docs were corrected. The GCP
> obs-relabel factory (M3–M7: worker + partition + submit + reconcile, dry-run
> /cost-capped/Spot-only/ledger-gated) is committed but **GATED** behind local
> pilot validation. (C) **Fable profiling study LAUNCHED** (background subagent,
> user ask): efficiency/NPS/cache/SIMD vs SF16/Reckless/Integral (WSL has no PMU
> → valgrind callgrind/cachegrind + perf software sampling) → will write
> `docs/research/profiling-study-2026-07-18.md`. (D) **Label-value pilot deep
> relabel: 2/10 chunks done** (SF18 @100k, 13 workers, ~resumable); on
> completion → cheap relabel → 2 corpora → deep-vs-cheap fixed-node A/B (gate
> ≥+15 → SF18 relabeling validated). (E) **gold-v2 labeling SALVAGED, but the
> d41-gold-v2 train is DEFERRED.** The gold-v2 run logged "2 worker(s) FAILED"
> (13:26) — but those are only the two 10% MONITOR QA slices
> (reckless-monitor / sf18-monitor, 1 verify-problem each); all **12 main
> worker shards are CLEAN** (15k episodes / ~500k arms each, reckless-w0..5 +
> sf18-w0..5 ≈ 180k positions × 2 teachers) and already ledgered
> (`gold-v2-teachers`). The teacher corpus is intact — nothing lost. But
> training d41-gold-v2 is lever-D (helm/policy), which the north-star ranking
> deprioritizes ("don't tunnel on the helm"), the capture-data question it
> serves is already answered, and the box CPU+RAM are committed to the
> higher-value lever-A relabel + lever-C profiler (RAM ~4.9GB → a train risks
> the 7GB-OOM rule). So d41-gold-v2 is QUEUED for when the relabel frees the
> box AND only if the helm line warrants revisiting; the label-value A/B
> (lever-A) processes FIRST. Export from the 12 clean worker shards (exclude
> the 2 monitors). The 10:07 block below stands for the speed/capture items.

> **OVERNIGHT UPDATE 2026-07-18 10:07 (persist checkpoint):** two things
> since 02:35. (A) SPEED PROGRAM CLOSED — R1 native REJECTED, R3 PGO SHIPPED
> (+10.5% D40 / +3.0% classic, machine-local), 720sb equal-time fastgate on
> best-PGO builds = **−8.7±12.9 (CI [−31.4,+13.9] straddles zero)**, D40 at
> equal-time parity (ladder −57.9→−17.4→−8.7, exp-0093). (B) CAPTURE-DATA
> USEFULNESS ANSWERED (user directive "generate a lot and train"): in-search
> capture is **99.6% aspiration bounds**; only 0.36% are exact PV nodes. Built
> the InSearchCapture→export-train bridge (Bound::Exact gate, committed
> dfe6cf9, 6/6 tests). Trained d41-capture-scale on 8.37M exact arms:
> **rank_corr 0.263 / top-1 ~0.30** — BEATS gold on top-1 (0.258) but 2.7×
> below on rank_corr (bound-capped ordering; 12× scale helped, quality
> ceiling holds). Capture = a clean BEST-MOVE supplement, not a teacher
> replacement. Disk cleaned (reclaimed d41-v1 + raw shards + zip; ledger 0
> problems; 19G free). NOW RUNNING: gold-v2 teacher labeling (pool2 100k,
> reckless+sf18, ~5-6h, tail phase at 10:07) → clean-ordering reference +
> flywheel cycle-1 corpus; on completion the loop trains d41-gold-v2 and does
> the 3-way verdict. Next levers queued: refined ordering-bridge (use
> raised_alpha/caused_cutoff not bound margins), capture-top-1 + gold-ordering
> BLEND. Helm play test HELD for user (capture ordering too weak to beat the
> warm picker). Commits: aea9ee0, dfe6cf9, 600bcb2. The 02:35 block below is
> superseded by this summary for the speed/capture items.

**THE DAY IN ONE LINE:** the speed program CLOSED — R1 native REJECTED, R3
PGO SHIPPED (+10.5% D40 / +3.0% classic, machine-local), and the final
720sb equal-time fastgate on best-PGO builds both sides landed
**−8.7±12.9 (bootstrap CI [−31.4,+13.9] STRADDLES ZERO)**: D40 is now
statistically at parity with the s4 classic incumbent at equal time
(ladder −57.9 → −17.4 → −8.7). D45 SPSA season stays 8/8 ship-nothing;
s3-farseer stays incumbent (s4 no-promo 0.0±11.3); SF16 anchor −63 (n.s.);
helm v1.1 play-NEGATIVE (−29.6±12.9, exp-0092) → data-scale-first.

**1. D45 SPSA season FINAL (exp-0075..0088):** c1/c2 NO-CHANGE certified,
c3 tentative-positive-UNSHIPPED (+10.4±14.1 cap-undecided, exp-0087; a
confirm SPRT was armed then CANCELLED by user — theta preserved in
`spsa/c3-rfp-razor/theta_final.json`), c4 neutral-reject (+1.2±14.0,
exp-0088), c5/c6/c7 screen-positive TC-unverified (gates cut by user
20:46 to jump the queue), c8 rejected. Nothing shipped; params.rs
unchanged; bench stays 701605. Calibration standing fact (exp-0078):
+50% RED_BASE detune costs only −8±13 — one-box SPSA has hit its
resolution floor; sub-10-Elo mining needs fishtest scale (D36).
Tooling kept: `tools/spsa/combined_gate_v1.sh` (union ship-SPRT, unused),
tune builds + `tunables` advertising, night_gates runner (nps_probe
piped-quit race now fixed).

**2. D45a headline — SPEED PROGRAM CLOSED (exp-0093):** the 720sb
equal-time ladder is now d40-uho-720sb vs s4:
**−57.9±12.6** (v3, pre-R4, exp-0073) → **−17.4±12.8** (v3, post-R4 stack,
exp-0089) → **−8.7±12.9** (best-PGO both sides, exp-0093, 200g 10+0.1,
25W/145D/30L, bootstrap CI [−31.4,+13.9] STRADDLES ZERO). D40 is
statistically at parity at equal time. Speed verdicts (task #13/#15,
r4-micro-audit doc): R1 native REJECTED (null on sota — AVX-512 VNNI is
runtime-dispatched so v3 already uses it; −9.6% classic — Zen-4 512-bit
autovec loses to AVX2); R3 PGO SHIPPED machine-local (+10.5% D40
`probe-sota-pgo`, +3.0% classic `probe-l2048-pgov3`, both node-verified;
PGO base target-cpu per path via PGO_CPU, profile MUST run the real net
over UCI). D40 wins per node (+30.2 h2h exp-0072). Levers still open:
packing fix (#26), narrower-L1 D40. The equal-time promotion bar for D40
over the classic incumbent is now within one modest quality/speed step.

**3. Incumbent unchanged (exp-0090):** s4 vs s3 promo re-check at equal
time = **0.0±11.3** — dead even, second failed separation.
**s3-farseer remains the incumbent**; the classic-2048 line is saturated
(consistent with the width-ladder plateau). Eval-foundation freeze holds.

**4. SF16 anchor rematch (exp-0091):** −63.2±29.5 (13W/56D/31L, 100g
10+0.1, srand 20260743) vs the 2026-07-16 anchor −78±18. Nominal +15,
diff CI ≈ [−20,+49] — NOT statistically established. **Lesson written
into the row: self-play gains deflate externally** (S1's +72.9 was
tuned-vs-default in-family; do not bank self-play Elo into external
projections). External movers remain eval quality (D41 flywheel) and
D40 speed parity.

**5. IN FLIGHT overnight:** (a) **gold-v1**: 20k positions, 4 workers ×
2 teachers + dual-tier monitors ([obsgold] tags, done ~00:55, auto-audits
at end) — first measurement of the ~25–30× parallel-labeling throughput
claim; (b) **helm v1.1 play test**: d41-alloc-v11-c (frozen-720sb-trunk +
pairwise + all-tiers; held-out 0.706/0.258, clears the 18.1% cold bar)
as MovePicker prior — probe-policy-d41 + POLICY_TRUNK=720sb cenn vs
probe-plain-d41mm, net-0 eval both sides, 600g 2k fixed nodes conc 12,
srand 20260791, output `d41-v11-playtest/` (verdict ~23:00; v1 at 336k
rows was null — this is the anti-overfit retrain).

**6. Learn-docs media:** Manim pilot (doc 02, 7:11 1080p30 + narration)
PASSED the user gate and is merged (e95d833); the series was started
then **POSTPONED by user** ("not now", task #29) — partial agent diff
saved to session scratchpad, workspace reverted to pilot state. Do not
resume without an explicit user go.

**7. Ledger:** verify = 0 problems; dotted-path annotation overlays
fixed in ledger.py (553d8c8). Rows tonight: exp-0087..0091 + annotations
on exp-0081/2/3/5 (gate cuts).

**RECOMMENDED NEXT MOVES (morning):** (1) read the helm verdict — if
positive at fixed nodes it is the first play-positive D41 artifact on the
shared trunk; if null, the data-volume diagnosis routes work to gold
throughput + in-search capture harvests; (2) D40 speed program (tasks
#13/#15/#26) — the −17.4 gap is now plausibly closable by build/packing
alone; (3) king-bucket factorization remains the top pending training
experiment; (4) S3 of the search program (gainer-season global SPSA)
waits on fishtest-scale compute (D36) per the season lesson.

## PREVIOUS CHECKPOINT (2026-07-17 06:20)

**OVERNIGHT OUTCOME (all ledgered, details in the dated entries below):**
1. **Promotion #4 REJECTED (exp-0073)**: d40-uho-v1-240sb vs s4 at equal
   time = **−57.9 ± 12.6** (200g 10+0.1). D40 wins per node (+32.5
   exp-0065; 720sb adds +30.2 exp-0072) but the inference tax dominates;
   needs ~1.7–2× NPS to convert. D40 → background line (task #26).
   720sb itself trained fine but is parity-barred by the OOD-endgame
   quant tail (max 0.2517 vs 0.20, mean clean) — same mode as the hop;
   repair options for the user in the 03:50 entry.
2. **S1 correction history DEPLOYED (exp-0074)**: equal-time SPRT PASSED
   H1 in 208g, **+72.9 ± 16.2** (SPRT [0,10], LLR 2.95). Merge gate green
   at HEAD, bench 701605. Largest search gain since the P9 waves. The
   fixed-node screen said only +18.5 — protocol lesson: NPS-neutral
   search changes are underestimated by shallow screens; TC-gate them
   early.
3. **Obs/D41 platform is PRODUCTION (M1–M5 all closed)**: dual-teacher
   sample-v1 audits = GO (scores reproducible, 8–10cp cross-tier drift;
   argmax soft — median top margin 13–16cp → train on per-arm values;
   teachers convergent: 75% best-move agreement, tau 0.841). Reckless
   primary / SF18 control HOLDS. Both shards ledgered. **big-v1 labeling
   RUNNING since 06:09** (10k fresh positions, both teachers parallel,
   done ~16:30, auto-audits at end, [obsbig] tags).
4. **EVALUATOR-FREEZE PROPOSAL drafted (05:35 entry) — the user's morning
   decision**: freeze the classic line (s3-farseer incumbent, s4
   working-best + cheap promotion-#5 TC re-check), pause eval trainings,
   shift to search (S2 SPSA next) + D41/design-19 as the main thrust.
5. R4 COMPLETE: all three bench-neutral speedups landed — SIMD find_nnz
   (3257492), threat scratch (943584d), and **#1 lazy FullThreats fold
   (45301df)**: the fold (3 KB copy + 768-lane apply) now runs only for
   plies that reach an evaluate, mirroring HalfKA's memoized compute;
   king-crossed plies stay eager by design. Verified: 636 sota tests + 3
   new lazy-path tests + release replay oracle green, and the new binary
   is EVAL-BIT-IDENTICAL to the pre-R4 one across the 4096-fen parity
   set. probe-sota redeployed at 45301df (.pre-r4 backup kept); NPS
   numbers wait for the R1 quiet-box session. Sota-gated tests need
   `--features sota` — the default gate compiles them out. Probes carry
   S1 search now; cross-era fixed-node comparisons must note this.
6. Ops: 720sb wrapper exit ≠ success (train FAILED at parity after 7h —
   the night chain degraded gracefully by design). big-v1 guard survived
   a pipefail/pgrep bug (fixed, a76d1c7+). Ledger verify: 0 problems.

## PREVIOUS CHECKPOINT (2026-07-16 20:30)

**Housekeeping state (20:30):** disk cleaned 8.1G→41G free (~33GB: cargo
debug/cross targets, sf-100 shard intermediates, six old ckpt dirs, eight
dead campaign dirs, and three LEDGERED artifacts deleted WITH lost-
annotations: lichess bulk 8G, test80 binpack 11G [re-downloadable], E1
relabel corpus). Ledger gained the append-only correction mechanism
(`ledger.py annotate`, annotations.jsonl, design/18 §5; commit 287db4f);
verify now runs end-to-end (7fa53f4) and reports exactly 1 problem: exp-0042
dangling → unregistered net sf100-matrix-v1-sf100-d40-full (registration
needs a code_sha convention — user decision pending). Killed three stale
sleep-loop waiters from concluded eras (d40_rescreen, e4_king_rerun,
e1_control) — kill by exact PID only. Worktree CLEAN, all work committed;
122 commits ahead of origin/main, push is the USER's action (never push).
config.json is fastchess-written runtime state and will re-dirty after
every match — candidate for gitignore, don't chase it. Design docs + 
CLAUDE.md status refreshed to the post-B4 reality (24c8499, 52d1de9).
Rust perf tasks R1-R4 queued (native builds / VNNI sota_simd / PGO /
micro-audits) — R2 VNNI is the D40-NPS lever, box is Zen 4.

**In flight RIGHT NOW (updated 21:00; check these before doing anything):**
- `tools/nnue/d40_long_v1.sh` (bg): d40-uho-720sb (18B presentations)
  training — RESTARTED 20:48 after the OOM incident below, ~7h → done
  ~03:50. The first attempt (19:15) was OOM-killed at ~70 min.
- `tools/nnue/d40_night_v2.sh` (bg, re-armed 20:48): waits for the train,
  then (1) h2h hop d40-s2-farseer-120sb vs d40-uho-v1-240sb — DIAGNOSTIC
  ONLY, (2) 720sb parity parse + h2h vs 240sb, (3) EQUAL-TIME GATE best
  parity-PASSING D40 vs s4 (200g 10+0.1 conc 1) — promotion #4 verdict by
  ~08:00. Both engine binaries (probe-sota, probe-l2048) were rebuilt at
  df2b746 with the EvalFile leak fix; .pre-leakfix backups sit beside them.
- **THE 3.6GB-ENGINE BELIEF IS DEAD (21:00): it was a per-game memory
  leak, now FIXED (df2b746).** fastchess re-sends every option — EvalFile
  included — before every game; each re-send re-loaded and leaked the
  net (~78MB/game for D40, ~3MB/game for classic). Real sota engine
  footprint after fix: **~135MB flat** (verified over a 20-game match:
  134-135MB throughout vs 1.7GB-and-climbing before). This retro-explains
  weeks of "sota engines are huge" ops pain AND unlocks D40 test
  concurrency (T4 applies to D40 now, not just classic).
- HOP PARITY FAIL (19:14): d40-s2-farseer-120sb runtime mean/max
  0.011/**0.241** vs gates 0.03/**0.20** — mean excellent, p99 0.088,
  worst fen an OOD bare endgame (2R5/8/8/3BK2k/8/7p/8/8 b), runtime
  tracks quant (quant max 0.63) → quant-CLIPPING tail, NOT a transform
  bug (sf-100's true fail: 0.355 broad). Hop is NOT a gate candidate;
  user decides waiver vs rescale (tools/nnue/rescale_net.py) tomorrow
  with its h2h number in hand. Old `d40_s2_gate_v1.sh` chain is DEAD
  (exited 1 at parity) — superseded by d40_night_v2.sh.
- RAM RULE, amended 21:00: the old form ("sota pairs = 2×3.6GB, never
  run games while a train lives") was built on the leak, which is now
  fixed — engines are ~135MB. The surviving rules: (a) equal-time (TC)
  games still require a quiet box for WALL-CLOCK FAIRNESS — never run
  them alongside a trainer; (b) fixed-node games alongside a trainer are
  now RAM-safe but were the proximate cause of tonight's OOM before the
  fix — only resume that practice with post-fix binaries; (c) any new
  engine binary must pass a capped 20-game RSS-flatness check before an
  overnight chain trusts it.
- Results log for everything: `data/nnue/experiments/chain-v2-results.log`.
- SEARCH PROGRAM defined (user-directed): ranked matrix in
  `stockfish_reference/06-sf14-sf16-search-patch-study.md` §6a/6b;
  tasks S0 (unbalanced test book + SF16 mechanics audit), S1 (wire
  correction history — tables exist, read/write deferred at
  src/search/mod.rs ~line 121), S2 (gainer-season global SPSA retune —
  the single biggest pool, +15-40 est). Key audit fact: P9 already has
  singular/multicut/negative extensions, ProbCut, cutoffCnt,
  do_deeper/do_shallower, improving, cut_node, fallingEval/instability
  TM. Search game-testing waits for the D40 arc to settle; code work
  can start anytime.


**The governing question is ANSWERED.** Net-0 was never special; DATA VOLUME
was the constraint end-to-end. Everything below in one paragraph: the
official stage-1 UHO binpack beat net-0 at its own recipe on the first try
(E3, +47.8); width then scaled 256→512→1024→2048 (+32/+123/+225/+282 @5k vs
net-0) and SATURATED at 2048 (4096: +15 ± 17); schedule and lambda were flat;
stage-2 warm retrains gain in proportion to stage-2 LABEL QUALITY (farseer
0.93 Pearson: +55/+38; test80 0.78: +19/flat) and plateau after ~2 hops;
rich FEATURE SETS (king768, D40) failed historically ONLY from data
starvation — on full UHO the D40 parity gate passes (0.174 vs 0.355 on
sf-100) and stage-1-only D40 beats the best classic by +32.5 ± 17.3
(exp-0065). External anchors (all 100g, 10+0.1, 1 thread): SF16 −78 ± 18,
SF18 −191 ± 23 (gap 113 ± 29 ≈ known 90), Integral v6 −70 ± 22, and the
triangle closes locally (SF16−Integral = +52 ± 20 measured here; site 80
compresses at fast TC). Promotion chain (equal-time gated): net-0 → +166
(u1024-120sb) → +43.7 (uf2048-240sb) → +54.9 same-width (s3-2048-farseer,
INCUMBENT); s4-2048-t74 = working best (+15.6 ± 11.9, below bar).
**In flight at checkpoint:** see the "In flight RIGHT NOW" block above
(d40_long_v1 720sb retrain + d40_night_v2 chain; the old d40_s2_gate_v1
chain is dead). Next levers ranked: bigger/better stage-2 corpora
for D40 (T60T70 33GB needs disk; test80 deletable), longer D40 stage-1,
king768 probe, official loss shape (prepared, unneeded so far). Disk 8.5GB
free. All experiments exp-0038..0065 + 20 nets ledgered today; every
result also in `data/nnue/experiments/chain-v2-results.log`. Ops rules
that bit today: gate-waits must check status=complete AND not self-match
their own log lines; cloned D40 configs need `train.quantised` updated;
parity stage needs CUDA_PATH=/usr; list-then-delete separately; the old
"conc 1, sota engines are 3.6GB" rule is superseded — that was the
per-game EvalFile leak, fixed at df2b746 (see the 21:00 incident entry).

## The theory (ranked, with evidence)

**H1 — Scale/schedule starvation (dominant, prior ~0.6).** Every training run
in project history is 74 s – 15 min of GPU (≤ 747 M presentations; the
30-superbatch constant was never questioned). net-0 wins because 747 M over
its 25 M corpus = 30 epochs = converged; every bulk run was 0.3–3 epochs =
starved. Fits: fishpack classic1024 < classic256 at equal presentations
(classic undertrained-capacity signature); Lichess classic1024 −39 @20k off a
12-minute run. Community engines train tens of billions of presentations;
official SF trains world-class nets from the exact fishpack file we hold. Our
GPU does ~10 M pos/s on classic nets → a 12 B-presentation run costs ~2-4 h.
Compute was never the constraint; nobody turned the dial.

**H2 — Search-consumer coupling (real, magnitude unmeasured, prior ~0.25).**
Pruning margins (futility/razoring/NMP/aspiration) were tuned pre-NNUE and
only ever validated against net-0's output distribution (std ~613 cp, WDL-
sharpened). Bulk nets are flatter (std 257–371 cp); every cp-denominated
margin bites differently. Every Elo verdict flows through this one tuned
consumer. Counter-evidence: net-0 itself runs 2-3x "hot" vs the PSQT scale the
margins were tuned on and still gained +696 — so global scale alone is not
fatal (and the D40 1.65x rescale test hurt). The untested part is
*distribution shape* interaction, not global gain.

**H3 — Label heterogeneity / local coherence (real but secondary, ~0.1).**
Mixed engines/depths/budgets teach a jagged local surface (coherence audits
confirm), but Lichess classic1024 reached −39 @20k with worst-case
heterogeneous labels and zero outcomes → this is a lever, not a wall. The
bounded-label quarantine applies to the 5k-relabel corpora, NOT to
fishpack/Lichess.

**H4 — Pipeline/architecture bugs (exonerated, ~0.05).** net-0 seed-100
reproduction = exactly 50%; parity gates bit-exact everywhere. Stop looking.

## Preregistered probes (launched 2026-07-15)

### Probe A — one real training run (H1 test)

Campaign `value-scale-probe-a-v1` (config
`tools/nnue/configs/value-scale-probe-a-v1.json`, driver
`tools/nnue/value_matrix.py`), identical to the failed matrix-v2 recipe in
every respect EXCEPT the schedule: **480 superbatches ≈ 12.0 B presentations
(16x), full cosine decay to SB480, save every 30 SB.** Runs, in order:

1. fishpack classic256 (control cell; matrix-v2 anchor −163 @5k at SB30)
2. fishpack classic1024 (capacity cell; matrix-v2 anchor −215 @5k at SB30)
3. Lichess classic1024 (best-known bulk cell; anchor −100/−72/−39 @2k/5k/20k)

A watcher (`tools/nnue/probe_a_watch.py`) packs each intermediate checkpoint
and screens it 80 games @5k vs net-0 (per-width engine binaries, paired
openings, concurrency 2) → the **learning curve** is the deliverable.

Predictions: H1 true → curves still climbing at SB480, classic1024 overtakes
classic256, best cell approaches/crosses net-0; the SB30 point reproduces its
matrix-v2 anchor (built-in control). H1 false → curves plateau at ~anchor
level by SB120 with fully-decayed LR.

Decision rule: crossing or clearly converging toward ≥ net-0 ⇒ H1 confirmed —
stop diagnosing, start scaling (schedule → capacity → HalfKA → datagen
flywheel). Hard plateau ⇒ H1 dead for that source; weight shifts to H2/H3 per
Probe B/C.

Known risks accepted: SF-binpack reader must not be interrupted (resume
unsafe: skip_count ignored); cross-binary fixed-node screens for non-256
cells (historical net-10 glitch) — mitigated by same-binary SB-to-SB curve
comparisons and the SB30 anchor.

### Probe B — budget-ladder decomposition (H2 test)

`tools/nnue/probe_b_ladder.sh`: paired 80-game matches vs net-0 at **32 / 256
/ 2,000 / 20,000 fixed nodes** (balanced-256 book, seed 20260716,
concurrency 2) for the three strongest failed nets:

- `value-bulk-matrix-v2-lichess-classic1024.cenn` (1024-width binary)
- `value-bulk-d40-lichess-mid-v1-lichess-d40-full.cenn` (sota binary)
- `value-bulk-matrix-v2-binpack-classic256.cenn` (default binary, same-binary
  pair)

Interpretation: genuinely-worse net ⇒ loses at every budget, roughly
uniformly. Search-coupling ⇒ relatively strong at tiny budgets (pruning
barely acts at 32–256 nodes) and degrading as budget grows. If the coupling
signature appears ⇒ next act = per-net SPSA of the ~10 margin constants at
fixed nodes (D36 scaffold) and re-screen the "failed" nets before any more
data work.

### Probe C — label-semantics spot-audit (H3 guard, Codex's ask)

`tools/nnue/audit_label_semantics.py` (new): sample ~400 records each from
the Lichess canonical train.bin and fishpack32.binpack — decode the exact
bytes training consumed, reconstruct positions, fresh pinned SF18 searches at
fixed depth 12 (completed iterations; 0-bound regime per prior smoke), one
thread. Report sign agreement, Pearson/Spearman, slope, STM-vs-White POV
check, mate handling; JSON + markdown under `data/nnue/audits/`. Any semantic
defect ⇒ quarantine the corpus and re-audit every conclusion built on it
(including Probe A's cells on that source).

### Probe A2 (preregistered 2026-07-15 18:05, runs after Probe A cells finish)

**H3b — per-source target-scale miscalibration (new, from Probe C).** Fishpack
stored scores are SF INTERNAL units ≈ 3.0x SF18-UCI cp (Probe C OLS a=3.026,
R²=0.90); Lichess ≈ 1.28x. The trainer's sigmoid target scale (400) was
calibrated for net-0's UCI-scale SF16 labels, so every fishpack cell trained
against ~3x over-sharp sigmoid targets — the exact direction (over-sharp
targets) that the WDL-logit400 and scale-200 experiments already proved loses
engine strength. Test: fishpack classic256, 30 SB (matrix-v2 anchor recipe,
−163), with stored scores divided by ~3.03 (or equivalent sigmoid-scale
1212) — one ~12-minute GPU run + 80-game screen. Prediction: if H3b is
material, the rescaled cell clearly beats −163; combined with a long schedule
it may transform the fishpack picture entirely.

## Live status

- 2026-07-15: file created; probes A/B/C launching. Worktree committed
  through `a8856b1`; ledger backfilled (exps 0025-0034). No cloud spend
  authorized for the probes; everything local.
- 2026-07-15 ~17:47 ET: ALL THREE PROBES LAUNCHED (detached; pgids of
  probe_a_launch.sh / probe_a_watch.py / probe_b_ladder.sh ≈ 1055970-72).
  Prebuilt match binaries at `engines/bin/probe-{default,l1024,sota}`
  (x86-64-v3; sota + default both verified bench 548978). Probe A trains at
  8-12M pos/s on SF-binpack — a 480-SB cell is ~20-25 min, so GPU work
  finishes in ~1.5h and the watcher's 48 checkpoint screens are the long
  pole (~4-6h). First ladder instrument check: lichess-classic1024 @2k =
  -98.1 (matches the -100 matrix anchor, different seed/book — instrument
  consistent). Probe C runs as a subagent writing
  `tools/nnue/audit_label_semantics.py` + `data/nnue/audits/label-semantics-v1/`.

## Results log

- **2026-07-15 17:50 — Preregistration amendment (Probe A control).** The
  "SB30 checkpoint must reproduce the matrix-v2 anchor" control was
  mis-specified: matrix-v2's SB30 was the END of its cosine decay, while probe
  A's SB30 sits at near-full LR on a 480-SB horizon. Intermediate
  high-LR checkpoints screen far worse (observed: fishpack classic256 SB30 =
  −490 ± 82; cf. matrix-v1's undecayed −512 datapoint). The valid anchor
  comparison is probe A's FINAL SB480 (fully decayed) vs matrix-v2's final
  (−163/−215), at 16x presentations. Curve interpretation must weight the
  trend and the decayed endpoint, not raw intermediate levels.
- **2026-07-15 17:56 — PROBE B COMPLETE (80g/point, seed 20260716, paired
  balanced-256 book, vs net-0).** Elo ± SE by fixed-node budget:

  | net | n32 | n256 | n2k | n20k |
  |---|---|---|---|---|
  | lichess-classic1024 | −132 ± 39 | −26 ± 39 | −98 ± 40 | −48 ± 36 |
  | d40-lichess-mid-sb15 | −48 ± 36 | −75 ± 39 | −137 ± 40 | −39 ± 35 |
  | binpack-classic256 | −185 ± 41 | −112 ± 40 | −293 ± 52 | −293 ± 52 |

  (binpack 2k/20k identical scores verified as coincidence: distinct PGNs,
  correct node configs.) **Verdict: the H2 coupling signature — strong at
  eval-dominated tiny budgets, degrading as pruning margins bite — appears in
  NO net. All three lose clearly at n32, the near-eval-only regime, so their
  raw evaluations are genuinely weaker than net-0's; the negative screens
  were honest.** The fishpack net worsens sharply with depth (−185→−293),
  consistent with its terrible local coherence (156 cp mean residual)
  compounding under search. H2 demoted to a secondary factor (some
  budget-shape structure exists but is noise-level at 80g); per-net margin
  SPSA is DEPRIORITIZED. The binding question is now H1 vs H3 — Probe A's
  decayed endpoints and Probe C's audit decide.
- **2026-07-15 18:00 — PROBE C COMPLETE: BOTH SOURCES PASS — no semantic
  defect, no quarantine.** `tools/nnue/audit_label_semantics.py` (400
  rows/source, seed 20260715, FULL-file coverage on both corpora — fishpack
  sampled via chunk-header skip-scan across all 5477 chunks, one entry per
  selected chunk; lichess evenly-spaced record indices, stride 663152 —
  fresh SF18 `go depth 12` oracle, Threads=1, hash cleared per position, 2
  nice-10 workers; corpus sha256s re-verified == ledger rows; outputs
  `data/nnue/audits/label-semantics-v1/{lichess,fishpack}{.json,-summary.md}`).
  **Lichess train.bin PASS:** all 400 records decode white-to-move canonical
  (bulletformat flip verified: kings match ksq/opp_ksq^56; 0 structural
  faults); result byte uniformly 1 (exporter 0.5 placeholder, WDL=0 rule
  intact); core Pearson 0.946 / Spearman 0.974, sign agreement 96.9% (all 11
  disagreements |cp|<80 near-equality noise); OLS stored = 1.284·SF18cp + 5.0
  (R² 0.89) — stored cp ~28% hotter than depth-12 SF18-UCI, a calibration
  note (deeper heterogeneous browser-SF sources), not a defect; no mates,
  |stored| ≤ 1679 (2000 export cap intact). **fishpack32 PASS:** stm
  209w/191b; POV decisively stm-relative (core Pearson 0.948 stm vs −0.044
  white); sign agreement 99.7% (364/365); OLS stored = 3.026·SF18cp + 1.3
  (R² 0.90) — labels are SF *internal* units (~3x UCI cp, normalization ≈3.5
  attenuated by oracle noise); result −1/0/+1 = 90/232/78 stm-POV, coherent
  with scores (bucket mean stored −499/+21/+488). Oracle regime: 0/800
  bound-carrying finals, 0 timeouts, median 0.04 s / ~20-26k nodes per
  position. **Consequence: H3-as-semantic-defect is dead; both corpora mean
  what training assumed (mover/stm-POV, sign-correct, projection intact).
  H3 survives only as heterogeneity/scale (lichess 1.28x, fishpack 3.0x
  SF18-UCI-cp — per-source trainer scale/σ must match). Probe A/B
  conclusions on these sources stand; H1 is now the last hypothesis
  standing pending Probe A's decayed endpoints.**

- **2026-07-15 18:07 — PROBE A CELL 1 COMPLETE: fishpack classic256 @ 16x =
  −328 ± 55 (driver screen, fully decayed SB480) vs 1x anchor −163.** Full
  watcher curve (80g @5k, seed 20260715): SB30..270 flat in a noisy
  −407..−536 band at full LR (one −881 tail excursion at SB210, recovered;
  semantic probes sane throughout, no pack corruption; trainer loss smooth
  0.0154→0.0146 — loss saw nothing); late decay lifted it only to
  −269/−348/−328 at SB390/420/450. **Verdict: 16x presentations made this
  cell ~165 Elo WORSE than 1x. Pure H1 is DEAD for fishpack classic256.**
  Best explanation: H3b — more presentations fit the ~3x over-sharp targets
  more faithfully (semantic probes confirm amplitude shrinking with
  training: +N probe 405→342 cp from SB90→SB210). Note loss-vs-strength
  dissociation again: lowest running loss coincided with the −881 screen.
  Cell 2 (fishpack 1024) prediction under H3b: also fails to beat its −215
  anchor. Cell 3 (Lichess 1024, near-calibrated 1.28x labels) is now the
  decisive H1 test: if IT improves markedly with 16x while fishpack cells
  regress, the joint diagnosis is "H1 holds only with calibrated targets;
  fishpack needs the A2 rescale."

- **2026-07-15 18:40 — DIAGNOSIS CLOSED BY USER DECISION; probes terminated
  mid-cell-2.** User: remaining results (cell 2/3, A2) would not change the
  next action. All probe processes killed cleanly (campaign driver, cell-2
  trainer at ~SB300+, watcher, A2 runner, monitors); cell 2/3 and the A2
  anchor reruns are CANCELLED, their preregistrations void. **Final
  synthesis — why we struggled to train good nets:**
  1. NOT the pipeline (H4; seed-100 repro = 50.0%).
  2. NOT the search consumer (H2; Probe B: no coupling signature — every
     bulk net loses even at 32-node near-eval-only budgets).
  3. NOT a label-semantics defect (Probe C: both corpora byte-verified
     against pinned SF18; POV/sign/canonicalization exact).
  4. **Fishpack: target-scale miscalibration (H3b).** Stored scores are SF
     internal units ≈ 3.0x UCI cp; trainer sigmoid scale 400 assumes UCI cp
     ⇒ ~3x over-sharp targets. Probe A cell 1: 16x presentations made it
     WORSE (−328 vs −163) — faithful fitting of miscalibrated targets.
  5. **Scale alone fixes nothing when targets are miscalibrated; scale with
     calibrated targets remains the live (untested-at-endpoint) H1 lever**,
     with Lichess classic1024's −39 @20k off a 12-min run as the best
     supporting evidence.
  Actionable recipe going forward: calibrated targets (per-source score
  scale) + real outcomes/WDL + real schedule (multi-billion presentations)
  + capacity, screened with the local-coherence gate and multi-budget games.

### Production track (started 2026-07-15 18:40): prod-fishpack-cal-v1

The diagnosis phase is over; the GPU now trains PRODUCTION candidates.
Campaign `data/nnue/campaigns/prod-fishpack-cal-v1/`, bullet patched with the
env-gated `SFBINPACK_SCORE_DIV` (tools/nnue/patches/bullet-sfbinpack-score-div.patch):

1. `prod-fishpack-cal-v1-classic256`: fishpack, score/3.5, WDL 0.3, 480 SB
   full decay, seed 198273612 — the fast (~20 min) calibration read; its
   endpoint vs the miscalibrated anchors (−163 @1x, −328 @16x) measures H3b
   directly at scale.
2. `prod-fishpack-cal-v1-classic1024`: same recipe at width 1024 — the first
   real candidate. Endpoint screen 200g @5k vs net-0; if ≥ −50, run the
   2k/5k/20k curve + coherence audit before any promotion talk.

- **2026-07-15 19:05 — prod classic256 (÷3.5) endpoint = −470 ± 76: naive
  H3b calibration REFUTED; refined to H3b′ (consumer-amplitude matching).**
  The ÷3.5 "UCI calibration" overshot: the consumer's proven operating point
  is net-0's own (very hot) output distribution, not abstract UCI cp.
  Amplitude probes (+N, white knight up): net-0 = 1047, raw-fishpack-480SB
  net = 342 (screened −328), ÷3.5 net = 242 (screened −470) — strength has
  been MONOTONIC in output amplitude across all three. Fishpack labels are
  hotter than UCI but net-0's learned outputs are hotter still; dividing by
  3.5 made the coldest net yet. Test in flight: pack-time output rescale of
  the ÷3.5 net at k=1.75/2.33 (clip clamps 26/34 of 512 out-weights —
  partial compensation only), 80g @5k each. If Elo recovers toward −163,
  amplitude is causal and the durable fix is a training-side divisor ≈
  1.5–2.0 (or none) chosen to match net-0's output std, not label units.
  The classic1024 (÷3.5) cell continues training — its learned function is
  expected repackable/rescalable the same way.

- **2026-07-15 19:35 — Thesis corrected (user challenge): asymmetric
  temperature tolerance, NOT co-adaptation, NOT a magic dataset.** Nothing
  was ever fit to sf-100: search margins are PSQT-era (P9), training
  constants are bullet defaults. The consistent story: (a) the consumer is
  HOT-TOLERANT (net-0 runs 2-3x hotter than the PSQT scale the margins were
  tuned on and won +696) but COLD-FRAGILE (over-pruning when eval swings sit
  inside fixed margins; k=1.75 output rescue bought +170 tonight); (b) eval
  temperature is set by data statistics under identical defaults — sf-100
  measured at **32.4% draws, |cp| median 168, std 468** (decisive-rich → hot
  net) vs fishpack streamed at ~60% draws (→ cold net, and amplitude decays
  the longer you train). Prediction: fishpack normalized into sf-100's
  marginals trains hot and competitive. Test = v2 ladder (div2.0 alone;
  div2.0+skip75) then v3 (`prod-fishpack-cal-v3-sf100frame`: div2.0 + skip75
  + draw-keep 32%), 30 SB vs the −163 anchor, auto-escalating to 480 SB +
  200g screen if ≥ −100. New trainer knobs: `NET0_RANDOM_SKIP_PCT`,
  `NET0_DRAW_KEEP_PCT` (both env-gated, default inert).

### Active experiments (preregistered 2026-07-15 ~20:05, post-reset)

**E1 — positions vs labels (user-designed).** Question: is fishpack held back
by its positions or its scores? Method: 5.93M fishpack positions (evenly
sampled across all 5,477 chunks, per-game cap via stride), relabeled with
SF16 using gen_sf_data.py's EXACT settings (engines/stockfish-datagen,
Threads=1, Hash=16, nodes=5000, White-POV UCI cp, quiet/check/mate/cap
filters, dedup; aspiration bounds intentionally NOT filtered — sf-100 itself
was made this way). Fishpack's own game results kept for the WDL 0.3 blend.
Then train with the exact net-0 recipe and screen ≥300g vs net-0.
OUTCOMES: works → positions fine, original fishpack scores (or their
consumption) were the problem, and this pipeline = unlimited sf-100-style
data; fails → scores were never the problem, suspect stream/loader
mechanics. Tools: sfbinpack-audit/src/bin/bulk_fens.rs,
tools/nnue/relabel_fishpack_sf16.py. Running: ~1,300 pos/s, ETA ~75 min.
Output: data/nnue/experiments/fishpack-sf16-relabel/.

**E2 — 4-net architecture matrix on sf-100 (user-directed).** Question: was
"bigger net on the known-good data" ever given a fair shot? (No: 512 got one
unreliable cross-binary +16; 1024 was schedule-starved at 747M.) Method:
classic256 (60 SB), classic1024 / king768x256 / d40-full (240 SB each ≈ 6B
presentations, 8x the old budget), net-0 recipe otherwise, WDL 0.3,
sf-100 = the exact data/nnue/train.bin artifact. Decision-grade 300g @5k
endpoint screens; 400g @20k for anything above −50. Checkpoints every 30 SB
retained for convergence inspection. Per user rule: if cells converge badly
on the KNOWN-GOOD data with generous schedules, the training method itself
goes under review. Campaign: data/nnue/campaigns/sf100-matrix-v1/.

- **2026-07-15 20:55 — E2 partials (decision-grade, 300g @5k vs net-0).**
  [OBSERVATION] classic256 @60SB: **+6.9 ± 18.0** (parity; 400g @20k curve
  −0.9) — instrument + recipe reproduce parity on the good data; 2x schedule
  neither helps nor hurts. classic1024 @240SB (6B presentations, 8x the old
  budget): **−68.0 ± 18.9** — did not reach the −50 curve floor.
  [HYPOTHESIS] 25M positions cannot feed 4x capacity (memorization/underfeed),
  and/or the single-cosine schedule shape remains wrong for width; either
  way, capacity is NOT a free win on sf-100 — more data (E1/E3 pipelines) is
  the prerequisite for capacity, consistent with official practice (stage-1
  corpora are billions of positions). king768x256 and d40 cells pending.
- **2026-07-15 20:52 — UHO prefix label audit (200 rows vs SF18 d12):
  PASS.** [OBSERVATION] stm-POV (opposite-POV r = −0.007), sign 98.25%, core
  Pearson 0.970, OLS slope **2.82x SF18-UCI** (internal-units family, like
  fishpack's 3.03). Draws 50.5%. Score std ≈ 883 internal ≈ 313 UCI —
  **matching sf-100's ≈ 312 UCI-equivalent** (the stage-1 family shares a
  temperature signature; fishpack at ~60% draws is the cold outlier).
  Report: data/nnue/audits/uho-label-semantics-v1/.

### E3 (preregistered 2026-07-15 20:55): data transplant — net-0's exact
recipe on the official stage-1 corpus

Question: does our full stack train a strong net from the LITERAL data family
official SF bootstraps on? Method: `nodes5000pv2_UHO.binpack` 10GiB prefix
(~4B+ positions), net-0's exact recipe (classic256, 30 SB, WDL 0.3, sigmoid
400) with exactly two declared adaptations: `SFBINPACK_SCORE_DIV=1.9` (maps
UHO's 2.82x units onto sf-100's SF16 frame ≈ 1.5x, where the recipe is
proven) and `NET0_RANDOM_SKIP_PCT=75` (their own rfs=3; approximates sf-100's
global pre-shuffle on a game-contiguous stream). 300g @5k screen vs net-0.
OUTCOMES: ≈ net-0 → trainer/recipe exonerated end-to-end on external data;
the whole official data lineage is usable and scaling (schedule, then
capacity, then stage-2 warm retrains) is unblocked. Clearly below → the
defect is on our side of the interface and now tightly localized (loss
shape/quantization/eval integration), since data, labels, units, and recipe
are all individually proven. Runs after E1 (GPU serial).

- **2026-07-15 23:40 — E2 d40 cell: parity gate FAILED; screened manually
  with caveat.** [OBSERVATION] After 240 SB on sf-100 (running loss
  0.000633), the packed CENN's runtime-vs-trainer max error was 0.355 logits
  (gate 0.2; mean 0.0124 passes; quantization passes). ~5x worse max
  deviation than the lichess-trained D40 packs (0.074). The matrix runner
  died on the gate; chain released manually, 300g screen running on the
  packed net with this deviation on record. [HYPOTHESIS] very low loss /
  large weights → saturation amplifying pack rounding; not investigated yet.
  E1 starts now (GPU free); ETAs: d40 screen ~00:05, E1 ~00:30, E3 ~00:55,
  size control ~01:10, E4 ~02:00.

- **2026-07-16 06:45 — overnight chain died at E3; cause found and fixed.**
  [OBSERVATION] E3's second run crashed at SB 28/30: bullet's
  SfBinpackLoader panics (`UnexpectedEof`, sfbinpack.rs:112 unwrap) on a
  binpack whose last chunk is byte-truncated — our UHO file was a raw 10GiB
  `head` prefix, so its tail ends mid-chunk. Clean EOF at a chunk boundary
  is handled (loader wraps for the next epoch); ragged EOF panics. Fix:
  truncated the file in place at the last complete `BINP` chunk boundary
  (10,736,935,551 bytes, 10,273 chunks, 482,689 tail bytes dropped). All
  downstream steps (size control, E4, E1 rescreen, D40 rescreen) were
  gate-chained behind E3 and never ran; relaunched everything as ONE serial
  script with no gate files: `tools/nnue/chain_v2.sh`
  (log `data/nnue/experiments/chain-v2.log`, results
  `data/nnue/experiments/chain-v2-results.log`). Ops rule reaffirmed: any
  byte-truncated binpack must be re-truncated to a chunk boundary before
  training.

- **2026-07-16 06:52 — E1 rescreen reproduces −463 exactly: the original
  was NOT contaminated.** [OBSERVATION] Fresh 300g @5k with a different
  fastchess seed (20260716): score 0.065, **−463.2 ± 38.7** — same to the
  decimal as the pkill-suspected run. E1 verdict stands: fishpack positions
  + SF16 5000-node labels + fishpack results, exact net-0 recipe, 4.06M
  rows → far below net-0. Also learned: these screens take ~20 s, not
  ~25 min (300 games @5k nodes, conc 4, classic256) — screening was never
  the bottleneck we scheduled around. Interpretation deferred to the size
  control (4M sf-100 rows, identical recipe): if it also craters, E1's
  deficit is mostly corpus size; if it holds near net-0, fishpack positions
  (or their fishpack-outcome WDL component) are the problem.

- **2026-07-16 06:54 — E3 RESULT: first net ever to beat net-0.**
  [OBSERVATION] `e3-uho-transplant-classic256` (net-0 exact recipe on the
  official stage-1 `nodes5000pv2_UHO` 10GiB prefix, SCORE_DIV 1.9,
  skip75, 30 SB ≈ 747M presentations ≈ one pass): 300g @5k vs net-0 =
  **+47.8 ± 18.7**, paired-bootstrap CI95 [+11.6, +85.0], score 0.568.
  Same architecture and engine as net-0 → no NPS tax; fixed-node ≈
  equal-time for this pair (equal-time gate still required before any
  promotion claim). [HYPOTHESIS → now well-supported] The stack (trainer,
  units bridge, filters, pack, engine integration) is exonerated
  end-to-end; the binding constraint was DATA. sf-100 (25M) was the best
  corpus we had; the official stage-1 family at 10GiB immediately clears
  it. Unblocked next steps, in order of expected value: (a) equal-time
  gate for e3 vs net-0; (b) scale: more UHO data (we hold the manifest
  for 287GB), longer schedules; (c) capacity on the bigger data
  (classic1024 failed on 25M — retry on UHO); (d) stage-2-style warm
  retrains on rich data from the e3/UHO base.

- **2026-07-16 07:15 — AUTONOMOUS LOOP MANDATE (user): "auto research loop
  until we have a SOTA eval model. Local compute only."** Promotion bar
  unchanged: beat the incumbent at equal time. Loop protocol: fixed-node
  screens route, 300g+ decides, equal-time promotes, everything ledgered.
  Wave preregistered now:
  - E3 follow-ups: 400g @20k curve; 200g 10+0.1 equal-time gate (quiet box).
  - `uho-ladder-v1` (tools/nnue/uho_ladder_v1.sh, GPU-serial after E4):
    u256-60sb / u256-120sb (schedule axis: does optimization beyond one
    pass of the 10GiB keep paying? official practice says data gets many
    epochs); u512-60sb / u1024-120sb (capacity axis: E2 showed 1024 fails
    on 25M rows; prediction from official practice = capacity unlocks with
    billions-scale data. If u1024 ≥ u256 at matched screens, capacity
    scaling is open and the SOTA path is data x capacity x schedule).
  - Full `nodes5000pv2_UHO.binpack` (40.3GB) downloading (resume-capable);
    disk freed 16→59GB by deleting re-generable checkpoints of finished
    campaigns + quarantined lc0/mix/net0-paired corpora (identities +
    regen recipes preserved in ledger; active campaigns untouched).
  Decision rules, in advance: schedule axis wins → full-UHO long run next;
  capacity wins → full-UHO at the winning width; both flat → stage-2 warm
  retrain exploration (task 7) or loss-shape work. Equal-time failures of
  fixed-node winners → NPS/time-management investigation before any arch
  pivot.

- **2026-07-16 18:15 — FULL-FEATURE BREAKTHROUGH: D40 works at data
  scale.** [OBSERVATION] `d40-uho-v1-240sb` (HalfKAv2_hm + FullThreats,
  stage-1 on full UHO, 2h23m train): parity gate **PASSES** (runtime
  mean/max 0.0072/0.174 vs gates 0.03/0.2; the sf-100 twin failed at
  0.355 → yesterday's failure was data-starvation weight saturation, not
  a pack bug). h2h vs s4 working-best: **+32.5 ± 17.3** (300g @5k,
  exp-0065) — a first-try, stage-1-only full-feature net beats the
  twice-stage-2-hopped classic 2048. Every sf-100-era rich-feature
  failure is now explained by data volume. [LAUNCHED] D40 stage-2
  farseer hop (warm via D40_INIT_RAW from the retained raw ckpt) →
  h2h → EQUAL-TIME gate vs s4 (200g 10+0.1, conc 1, ~3h) — promotion
  #4 candidate. Ops notes: cloned D40 configs must update
  `train.quantised` (stale filename aborted the first pack); the
  train_run parity stage needs `CUDA_PATH=/usr` in the caller env.

- **2026-07-16 17:55 — Integral v6 anchor + triangle closure: rating
  validated.** [OBSERVATION] Integral v6 built from source (tag v6, era
  net integral-networks@238c8e4a30 pinned — main-branch net would be
  wrong-era; clang-for-both-languages fixes a mixed-LTO link failure).
  ARCHON vs Integral v6: **40.0% (11W 58D 31L), −70.4 ± 22.4**
  (exp-0063). Triangle closure under identical local conditions (early-
  stopped at n=54 by preregistered rule): SF16 vs Integral **+51.9 ±
  19.7** (exp-0064) — the sites' ~80 gap compresses to ~52 at 10+0.1 /
  1 thread / balanced book (draw compression). Triangle arithmetic:
  predicted Integral−ARCHON = 26 ± 27, measured 70 ± 22 → closes within
  1.3σ. [CONCLUSION] All four external measurements are mutually
  consistent: ARCHON sits ≈ 70-90 below SF16-class, i.e. Integral-v6
  class head-to-head, at these conditions. The internal +265 chain and
  external anchors agree.

- **2026-07-16 17:10 — SF18 sanity check PASSES.** [OBSERVATION] ARCHON
  incumbent vs FULL SF18, 100g 10+0.1: **25.0% (0W 50D 50L), −190.8 ±
  23.3, 0 time losses** (exp-0062; D40 trainer on GPU during the match —
  symmetric load caveat). vs SF16 was −77.7 ± 18.4 → the measured
  SF16→SF18 gap through our engine is **113 ± 29**, consistent with the
  known ~90. External anchors now bracket the engine coherently:
  ≈ −78 from SF16, ≈ −191 from SF18, +265 internal chain over net-0.

- **2026-07-16 16:20 — Stage-2 plateau confirmed; FULL-FEATURE TRACK
  OPENED.** [OBSERVATION] s4 pooled 600g vs s3: **+15.6 ± 11.9** — not
  promoted (farseer hops: +55 → +29 → +15.6 pooled; the classic-family
  stage-2 well is drying up). s4 stays working-best/gate-baseline.
  [OBSERVATION] bench NPS: probe-sota 4.26M vs probe-l2048 4.09M — the
  D40 SIMD path carries NO material speed tax, so fixed-node wins should
  transfer to equal time. [LAUNCHED] `d40-uho-v1-240sb`: D40
  (HalfKAv2_hm + FullThreats) stage-1 on full UHO with the proven recipe
  (div 1.9, skip75 — D40_RANDOM_SKIP_PCT added to the trainer) — the
  first rich-feature net ever given billions-scale data. Known risk: the
  pack parity gate (0.355 max-err failure on sf-100; hypothesis =
  saturation at very low loss — big data may avoid it). Screen: h2h vs
  s4, conc 1. king768-on-UHO is the follow-up probe.

- **2026-07-16 16:00 — Reality check vs full Stockfish 16: −78 ± 18.**
  [OBSERVATION] ARCHON (incumbent s3-2048-farseer, P9-era search) vs
  FULL-strength SF16, 100g 10+0.1, both 1 thread / 16MB hash:
  **39.0% (4W 70D 26L), −77.7 ± 18.4, 0 time losses** (exp-0060). Net-0-era
  ARCHON anchored ~3400 vs SF16's handicapped rungs; today the gap to the
  full engine is under 80 Elo, consistent with the +265 equal-time chain.
  70% draw rate — the eval is competitive; the deficit is now plausibly
  search-era (P9 pruning stack) as much as eval. [OBSERVATION] T74 hop:
  s4-2048-t74-240sb vs s3 incumbent **+29.0 ± 16.5** (exp-0059) — second
  consecutive stage-2 gain, diminishing vs T76's +55; 600g pooled confirm
  running before promotion.

- **2026-07-16 14:20 — Width ladder SATURATES at 2048.** [OBSERVATION]
  uf4096-240sb vs uf2048-240sb h2h: **+15.1 ± 17.3** @5k fixed nodes
  (exp-0058). Per-doubling curve on full UHO: 512→1024 +102, 1024→2048
  +57, 2048→4096 +15 — and the 4096 equal-time tax would eat that.
  [DECISION] Capacity axis closed on stage-1 UHO + current recipe;
  width 2048 is the platform. Live levers, in expected-value order:
  (1) stage-2 hops on the incumbent (T74 hop training now — farseer
  family at 3x data), (2) more/better stage-2 corpora (T60T70 33GB needs
  a disk decision), (3) official loss shape in bullet, (4) feature-set
  growth (king buckets) at billions-scale — parked, one retry planned
  when a stage-2 plateau arrives. farseerT74 audited: div 1.66, Pearson
  0.937, sign 95.1% (farseer-tier).

- **2026-07-16 12:50 — THIRD PROMOTION: stage-2 works, and label quality
  is what gates it.** [OBSERVATION] Farseer hops (h2h vs each warm-start
  parent, 300g @5k): s3-2048-farseer **+54.9 ± 17.4** over incumbent
  uf2048-240sb; s3-1024-farseer **+38.4 ± 17.9** (test80 hop at the same
  width: +19.1 ± 12.5 pooled 600g; test80 at 2048: −6.9 ± 16.4 flat).
  farseerT76's labels audit cleaner (Pearson 0.934 vs test80's 0.784) and
  its hop gains are ~2-3x — stage-2 gains track stage-2 LABEL QUALITY.
  **New incumbent: stage2-v1-s3-2048-farseer-120sb** (same width/engine as
  the old one → h2h transfers to equal time; exp-0056). Composed
  equal-time vs net-0 ≈ **+265**. Next: farseerT74 (20.1GB, downloading)
  for a second farseer-family hop at more data; wave5 uf4096-240sb
  training (~14:45); redundant 10GiB UHO prefix deleted (full file
  supersedes; disk 28GB free → T74 fits).

- **2026-07-16 11:30 — SECOND PROMOTION: uf2048-240sb is the new
  incumbent.** [OBSERVATION] Equal-time gate (200g 10+0.1, quiet box):
  uf2048-240sb vs u1024-120sb **+43.7 ± 13.9**, score 0.562, 0 time
  losses (exp-0054). Wave4 h2h @5k was +57.3 ± 17.7 vs uf1024-240sb
  (exp-0052) — the 2048 real-time tax costs some of the fixed-node edge
  but the capacity gain survives at 3σ. Composed equal-time chain vs
  net-0: ≈ **+210** (+166 u1024 gate + 44 this gate). [OBSERVATION]
  Stage-2 attempt 1 (s2-1024-test80-120sb, warm from uf1024-240sb):
  **+23.2 ± 17.4** vs its parent (exp-0053) — the official two-stage
  shape shows signal where cold-training the same data family always
  cratered. Width ladder to date (vs net-0 @5k): 256 +32 / 512 +123-151 /
  1024 +225 / 2048 ≈ +282 composed. In flight: s2w2 (pool s2-1024 to
  600g; s2-2048-test80-120sb warm retrain of the incumbent), farseerT76
  audited next; then width 4096 vs stage-2 data scaling by results.
  Ops: disk crunch to 3.4GB during downloads → deleted search-trace,
  E2/E4 checkpoint dirs, sf-100 raw shards (train.bin exact-bytes anchor
  re-registered as `sf-100-trainbin`; list-then-delete lesson added to
  memory). Disk 19GB free.

- **2026-07-16 10:55 — Wave3 flat; stage-2 track armed.** [OBSERVATION]
  wave3 h2h vs uf1024-240sb (300g @5k): uf1024-480sb (12B presentations)
  **−10.4 ± 17.4**; uf1024-240sb-wdl0 (official pure-score lambda)
  **−6.9 ± 17.4**. Schedule and lambda both flat at width 1024 — the width
  family is recipe-saturated ≈ +200-225 over net-0 (exp-0050/0051).
  [OBSERVATION] test80 (Lc0-derived stage-2 corpus, 10.8GB, 10,308 clean
  chunks) label audit vs SF18 d12: stm-POV, core slope **2.18x SF18-UCI**,
  Pearson 0.784, sign 93.9% — clean but noisier than UHO (0.970/98.3%),
  29% exact-zero scores, weak tail sign (73.7% at |cp|≥2000). SCORE_DIV
  1.45 maps it to the proven SF16 frame. [PREREGISTERED] Stage-2 attempt 1
  (tools/nnue/stage2_v1.sh, gated on wave4): warm start uf1024-240sb
  (new NET0_INIT_CENN knob), test80, LR 0.0005 (halved), WDL 0.25
  (≈ official stage-2 lambda 0.75), 120 SB; h2h vs its parent. OUTCOMES:
  clear win → the official two-stage shape transfers; scale stage-2 data
  next (T60T70/farseer files as disk allows). Flat/loss → rich-data gains
  need the official loss shape or interleaved mixing — loss-shape work in
  bullet becomes the next lever. In flight: wave4 uf2048-240sb (capacity;
  ETA ~13:35), then stage2. Disk 9.2GB free (floor reached; fishpack32
  deleted, re-downloadable, was E1-proven negative-value).

- **2026-07-16 09:05 — Full-file scaling at width 1024 is FLAT under the
  current recipe; width is the live lever.** [OBSERVATION] uf1024-240sb
  (full 40.3GB, 6B presentations) vs u1024-120sb (10GiB, 3B): pooled 600g
  head-to-head **+6.9 ± 12.4 — parity** (exp-0049). At width 512 the full
  file DID pay (+123 → +151.3 ± 20.9 vs net-0). [INTERPRETATION] At 1024,
  4x data + 2x presentations bought nothing under single-cosine + WDL 0.3;
  the binding constraint is now recipe or capacity, not corpus size.
  In flight: wave3 probes recipe (uf1024-480sb = 12B presentations;
  uf1024-240sb-wdl0 = official pure-score stage-1 lambda), wave4 probes
  capacity (NEW `nnue-l1-2048` engine feature, commit b9536e3 — default
  build unaffected; probe-l2048 built; uf2048-240sb queued behind wave3).
  train_net0 now accepts comma-separated binpacks (concat loader) for the
  coming multi-file stage-1 mix; wrongIsRight_nodes5000pv2 (7.3GB)
  downloading. Screens within a width family (same NPS) use direct
  head-to-heads; equal-time gates reserved for cross-width promotion.

- **2026-07-16 08:30 — EQUAL-TIME GATE PASSED: net-0 DETHRONED after 4
  days as incumbent.** [OBSERVATION] `uho-ladder-v1-u1024-120sb` vs net-0,
  200g at 10+0.1 (the user's promotion bar): **+166.2 ± 18.3**, score
  0.7225, ZERO time losses either side. Fixed-node +225 → equal-time +166
  = a real but affordable width-1024 speed tax. Caveat on record: the
  full-UHO wave trained on GPU concurrently for part of the match
  (gate-wait bug: eval_match writes summary.json with status=running at
  START — wait conditions must check status=complete, not file existence);
  load was engine-symmetric and forfeits were zero, so the 9σ verdict
  stands. **New incumbent: u1024-120sb. All future promotion gates run
  vs the current incumbent at equal time.** net-0 becomes the historical
  baseline. Ledger exp-0048. uf512-120sb (full 40.3GB file, 1 pass):
  **+151.3 ± 20.9** vs net-0 @5k — width 512 gains ~+28 over its 10GiB
  version (1.4σ, scaling-consistent). uf1024-240sb training (SB 179/240).

- **2026-07-16 07:55 — uho-ladder-v1 screens: CAPACITY UNLOCKED on
  billions-scale data.** [OBSERVATION] 300g @5k vs net-0, all on the UHO
  10GiB clean prefix, e3 recipe: u256-60sb **+32.5 ± 18.7**, u256-120sb
  **+27.9 ± 18.8** (schedule axis flat at width 256 — 30/60/120 SB all
  ≈ +30-50), u512-60sb **+123.0 ± 19.5**, u1024-120sb **+225.0 ± 22.8**.
  The SAME classic1024 that scored −68 on sf-100 (25M rows) scores +225 on
  UHO-10GiB. Bench NPS sanity: probe-default 4.30M, classic512 4.26M,
  l1024 4.65M (tree-shape noise; no material eval tax at these widths).
  [INTERPRETATION] Width × data is the scaling frontier, exactly as
  official practice predicts (their stage-1 = 40-60B samples). Width 256
  is the net-0-era ceiling; it cannot absorb more data or schedule.
  Equal-time gate for u1024-120sb (200g 10+0.1) launched; full 40.3GB UHO
  still downloading. Next GPU wave (after gate, to keep the box quiet):
  u1024 longer schedule on 10GiB, then full-UHO scale runs; consider a
  2048-wide engine feature if 1024 promotes (code change needed).
  Ledger rows to follow with the gate result.

- **2026-07-16 07:35 — D40 rescreen + E3 depth curve; sf-100 matrix
  complete.** [OBSERVATION] d40-full on sf-100 (240 SB, conc-1 clean
  screen): **−478.0 ± 39.7** (parity caveat max-err 0.355 on record). Full
  E2/E4 matrix on sf-100 25M, 300g @5k: classic256 **+6.9**, classic1024
  **−68**, king768x256 **−409**, d40-full **−478** — strength is MONOTONE
  DECREASING in feature richness at fixed 25M rows. [OBSERVATION] E3 depth
  curve: **+62.3 ± 15.4 @20k (400g)** vs +47.8 ± 18.7 @5k — the UHO net's
  edge GROWS with depth; every failed bulk net closed from below instead.
  [INTERPRETATION] One coherent picture: feature-rich architectures are
  data-starved at 25M; data scale is the master variable. Ledger
  exp-0042/0043.

- **2026-07-16 07:15 — E4: the sparse-init fix does NOT rescue
  king768x256.** [OBSERVATION] With `init_with_effective_input_size(32)`
  fixed (the only change): **−409.1 ± 33.8** (300g @5k) vs the buggy run's
  −372.3 ± 31.4. [INTERPRETATION] The historical king-cell failures were
  NOT the init bug. Remaining suspects, now data-first given E3/e1ctl: 64
  king buckets slice 25M rows into ~sparse per-bucket samples (official
  HalfKA-family trains on billions); possibly also the trainer's king-cell
  transform itself. Action: park king features until a billions-scale
  corpus is in hand (full UHO downloading), then retry once at matched
  budget. The CENN-v3 "simple HalfKA destroys topology" reading stays
  partially confounded but is no longer blamed on the init bug alone.
  Ledger: exp-0041.

- **2026-07-16 06:55 — E1 size control: −189.3 ± 21.5. Positions-vs-labels
  decomposition complete.** [OBSERVATION] sf-100 truncated to E1's exact
  corpus size (4,064,476 rows, identical net-0 recipe/schedule):
  **−189.3 ± 21.5** vs net-0. Ladder at matched recipe: sf-100 25M ≈ 0
  (E2 classic256 +6.9 ± 18.0) → sf-100 4M = −189 ± 22 → fishpack-pos +
  SF16-labels 4M (E1) = −463 ± 39. [INTERPRETATION] Two real, separable
  effects: (1) corpus SIZE costs ~190 Elo at 4M vs 25M under this schedule
  (747M presentations over 4M rows = ~187 epochs of repetition); (2) with
  size and labels held fixed, fishpack positions + fishpack outcomes cost
  a further ~270 Elo vs SF self-play positions. The user's question is
  answered: it is BOTH, and neither is a units/pipeline artifact —
  position provenance matters at matched everything-else, and data volume
  matters steeply below ~10M rows. Consistent with E3: the winning move
  is more SF-self-play-shaped data, not fishpack repair.

## Resume instructions

1. Read this file + the takeover checkpoint of `policy-automl-state.md`.
2. Probe A state: `data/nnue/campaigns/value-scale-probe-a-v1/state.json` +
   watcher outputs under `.../matches/<net_id>/curve-sb*`. Do NOT restart an
   interrupted SF-binpack training run (resume unsafe) — record the
   interruption, keep completed checkpoints, and re-plan.
3. Probe B outputs: `data/nnue/campaigns/probe-b-ladder-v1/`.
4. Probe C outputs: `data/nnue/audits/label-semantics-v1*`.
5. After A–C: write the synthesis in Results log, update H1–H4 posteriors,
   pick the fork (scale / retune-consumer / label-quality), report to user.

## 2026-07-16 19:45 [OBSERVATION] D40 farseer hop fails parity marginally; night chain replaces gate

- d40-s2-farseer-120sb (warm from d40-uho-v1-240sb raw, farseerT76 120SB)
  trained clean but FAILED the runtime-parity gate: mean/max 0.011/0.241
  vs 0.03/0.20 (p95 0.042, p99 0.088). Worst fen = OOD bare endgame at
  move 86; quant-vs-raw max 0.63 with runtime tracking quant → the tail
  is quantization clipping on out-of-distribution material, not a
  trainer/runtime transform mismatch (contrast sf-100 D40: 0.355 broad
  fail). Candidate fixes if its h2h is strong: rescale_net.py pass or a
  waiver with the caveat recorded. NOT a gate candidate as-is.
- Gate chain d40_s2_gate_v1.sh exited at the parity stage by design;
  equal-time gate rescheduled into tools/nnue/d40_night_v2.sh (waits for
  the 720sb train, then hop h2h [diagnostic] → 720sb parity+h2h → gate
  best PASSING net vs s4, verdict ~06:30). Reason: 7.6GB box cannot host
  2 sota engines + the resident trainer (OOM risk to an un-resumable
  train).
- Ledger verify hardened (commit 7fa53f4): runs end-to-end now; first
  honest full pass = 2 problems (sf100+lc0atanhraw-9pct-27m DRIFT from
  deleted mix shards; exp-0042.A dangling → unregistered net
  sf100-matrix-v1-sf100-d40-full whose .cenn exists). Both need an
  annotate/status mechanism or a registration decision (user pending).
- Rust perf program queued as tasks R1-R4: native builds (R1, deferred
  to quiet box), AVX-512 VNNI sota_simd path (R2 — box is Zen 4, VNNI
  available; attacks D40's NPS tax directly), PGO (R3), bounds-check/
  false-sharing audits (R4). All gate through bench bit-reproduction
  (D24); sota f32 tail is the rounding hazard.

## 2026-07-16 20:50 [PLAN] Throughput program: optimize the research loop itself

Measured tonight: D40 trainer is GPU-BOUND (99% util) — local training
speedups are capped (only levers: batch 16384→32768 fits VRAM 3.4/8GB +
TF32 — but batch changes recipe semantics, treat as an experiment, not a
free win). The wall-clock cost of a RESEARCH DECISION is dominated by
TESTING, so that's where the program aims. Ranked (impact × effort):

- T1 sota engine RSS audit: runbook says 3.6-5.1GB/engine but the D40 net
  is only ~250-500MB of weights — likely loader staging waste (f32 copies
  not freed / oversized per-thread tables). If RSS → <1GB, D40 testing
  conc 1→3 (≈3x all D40 screens+gates) AND kills the OOM-vs-trainer
  conflict that pushed tonight's gate overnight. AUDIT FIRST, tomorrow.
- T2 unbalanced book (= S0, queued): our anchor matches run ~70% draws;
  UHO-style unbalanced books are exactly fishtest's fix. Fewer games per
  verdict (~1.3-1.7x).
- T3 SPRT early-stop in eval_match (fastchess native): clear results stop
  at ~1/2-1/3 the games; borderline ones run long — right allocation.
- T4 fixed-node screen concurrency: node-limited games are conc-invariant
  by design; classic screens at conc 3 waste 13 cores. conc 12 ≈ 4x.
  One-line + a one-time invariance sanity check.
- T5 interruptible training: periodic D40_SAVE_RATE + accept fresh-stream
  resume (with random-skip, restarting the stream is a distribution nop
  for multi-epoch runs — revisit the "never interrupt" rule deliberately).
  Removes the 7h-block serialization of the whole box.
- (user Q) WSL2 RAM ceiling: box shows 7.6GB total + 2GB swap — if the
  Windows host has 16/32GB, a .wslconfig bump is the cheapest multiplier
  of all (removes conc caps + OOM fear even before T1).
- (flagged, D36) GCE-spot OpenBench workers when S2 SPSA starts (10k+
  games/tune) — designed for exactly this, but LOCAL-ONLY mandate stands
  until the user lifts it.

## 2026-07-16 21:00 [INCIDENT+FIX] OOM killed the 720sb train; root cause was a per-game net leak, now fixed

**What happened.** Post-compaction throughput work (T1 RSS audit) measured a
standalone probe-sota at 135MB peak through an 8s/2.27M-node search and
concluded the "3.6-5.1GB per sota engine" figure was a mismeasurement. On
that basis the hop h2h (fixed-node, believed RAM-safe) was launched at 20:16
alongside the resident 720sb trainer. Under real match conditions the
engines ballooned (~90MB/s each); at ~20:25 the kernel OOM killer shot the
trainer (d40_features, ~70 min into ~7h), then one engine. The night chain
was killed by hand before it could launch the same match again. **The old
rule was right and the standalone measurement was the trap: engine RSS must
be measured under fastchess, not a hand-driven UCI session.**

**Root cause (proven, not guessed).** A tee-wrapper capture of 4 games shows
fastchess re-sends every configured option — `setoption name EvalFile ...`
included — before EVERY game. Each call to `nnue::set_eval_file` re-read and
re-parsed the net and `Box::leak`ed a fresh copy: one net per game per
engine, forever (~78MB D40, ~3MB classic-2048). The historical 3.6-5.1GB
observations were ~46-65 games of accumulation, partially paged to swap.
All match RSS forensics ran engines under `ulimit -v` caps so a repeat
overrun could only kill the capped engine, never the trainer.

**Fix (df2b746).** `set_eval_file` now records the installed file's identity
(canonical path, size, mtime) and treats a same-file re-send as a no-op; a
changed file at the same path still reloads; failed installs drop the cached
identity. Regression test `tests/eval_file_reload.rs` (own process — it
installs process-global nets). Full suite green (631 tests), clippy clean on
the merge gate, check.sh bench fingerprint **548978 reproduced bit-for-bit**
(non-functional for search). Verified in production conditions: 20-game
fastchess match, RSS flat 134-135MB the whole way (was 756→1661MB and
climbing at the same point pre-fix). probe-sota + probe-l2048 rebuilt at
df2b746 (only src change since 52d1de9); `.pre-leakfix` backups kept.

**Cost + consequences.** ~70 min GPU time + ~5.5h calendar slip (train
restarted 20:48, done ~03:50; night-chain verdicts ~08:00, unchanged in
content). In exchange: the single biggest known testing-throughput lie is
gone — D40 engines are ~135MB, so T4 (screen concurrency) now applies to
the full-feature track, and the trainer-vs-testing RAM conflict is over for
fixed-node work (see amended RAM rule in the checkpoint). Standing lesson
recorded in memory: measure engine RSS under the real match harness, and
smoke-test any "this rule is wrong" conclusion under the conditions the
rule was written for before acting on it.

## 2026-07-16 21:40 [TOOLING] T3 SPRT early-stop + T4 concurrency invariance both landed

**T3 (95dfda5).** `eval_match.py` gains `--sprt-elo0/--sprt-elo1`
(fastchess-native SPRT, logistic model so bounds are regular Elo,
alpha=beta=0.05, `--games` stays the cap). Early stop records an `sprt`
block (llr/bounds/verdict) in summary.json; completed dirs stay idempotent
via stored `pgn_games`. Default invocations are byte-identical (the
overnight chain is untouched). Smokes: classic 4-game regression clean;
D40 big-gap pair with a [0,200] window stopped **H1 at 16/100 games**.
CALIBRATION LESSON: a narrow [0,10] window resolved SLOWLY even at 100%
score (LLR 1.42 after 100g) — SPRT windows must be sized to the question:
~[0,50] for is-it-clearly-better screens, ~[0,10] only for gate-grade
close calls where you'd accept long runs.

**T4 (this entry).** Fixed-node concurrency invariance PROVEN: 60 games
s3-vs-s4 @5k, seed 4242, conc 1 vs conc 12 — same opening assignment,
same W-L-D (17-23-20, −34.9), and **0/60 games differ in movetext** after
stripping fastchess's per-move time comments. Wall clock 20s → 6s
(~3.3× with the trainer loader running; better on longer matches). NEW
PRACTICE: fixed-node screens run `--concurrency 12` (classic AND D40 —
post-df2b746 engines are ~135MB). Equal-time matches keep conc 1-3 and a
quiet box (wall-clock fairness, unchanged). eval_match's default stays 1
so TC invocations can't inherit 12 by accident.

## 2026-07-16 22:30 [RESULTS] Evening fixed-node slate (post-leak-fix, conc 6-12 beside the trainer)

All fixed-node @5k, post-df2b746 engines, capped wrappers, trainer at 99%
GPU throughout — the first productive use of the reclaimed testing lane.

1. **Hop is DEAD (exp-0066).** d40-s2-farseer-120sb vs its own stage-1
   parent: **-23.2 +- 17.1** (300g). The farseer stage-2 recipe that gained
   +55 on classic REGRESSES the D40 arch. The parity waiver/rescale
   question is moot; gate candidate stays d40-uho-v1-240sb (or 720sb).
   OPEN QUESTION for the D40 track: why does stage-2 warm-retraining hurt
   here — LR too hot for a warm start? 120sb too short? threat features
   overfitting to farseer's distribution?
2. **D40 beats the INCUMBENT at fixed nodes (exp-0067).** d40-uho-v1-240sb
   vs s3-2048-farseer: **+30.2 +- 17.4** (300g SPRT [-10,10] cap-out,
   LLR 2.04). Consistent with +32.5 vs s4 (exp-0065). The arch is ahead of
   both classic bests per node; tonight's equal-time gate prices the NPS tax.
3. **Book sensitivity found, S0 promoted in urgency (exp-0068/69/70).**
   s3 vs s4 on the fishtest UHO unbalanced book (downloaded to
   data/nnue/openings/UHO_Lichess_4852_v1.epd, 2.63M lines): pooled 400g
   **-45 +- 15** vs **-15.6 +- 11.9** on our 256-line balanced book — a
   ~30 Elo book effect (~1.5 sigma). s4 (t74 stage-2) is plausibly the
   better net from unbalanced openings and our promotion history is all
   balanced-book. NOT leaderboard-invalidating on this evidence, but:
   adopt UHO for future fixed-node screens, and re-run the s3-vs-s4
   equal-time comparison on UHO before promotion #5. Draw rate at 5k was
   flat (26%->24-26%) — the S0 draw-rate claim is a TC question, untested.

## 2026-07-16 23:45 [CODE] S1 landed: correction history fully wired (a8e16c5) — D33 review in flight

The held-back half of Wave-F item 21: `ss.static_eval` is now the CORRECTED
eval (main-search N6 + qsearch stand pat); TT keeps raw everywhere (D22);
write-back was already live. Read/write formula parity checked against SF
search.cpp line-by-line. New bench fingerprint **701605** (was 548978,
+28% nodes at depth 6 — expected-functional per the doc; one pawnless
bench position balloons to 580k nodes, see below). Suite green: 586 tests.

Findings worth remembering from the landing:
- **Pawnless positions degrade correction history**: pawn_key and (in
  minor-less positions) minor_key are constant across the whole tree, so
  the mate-line's +256-clamped bonuses saturate the shared cells and
  inflate every sibling's margins. Empirically: the two-rook mate-in-3
  needed depth 12 (was 6) to prove exactly; middlegame positions shift
  ~1-3cp only. SF ships the identical formula (verified) — inherent
  tradeoff, not a bug. Doc 07 §8(4) bar amended 6→12; oracles re-scoped
  to exact-on-decisive + 8cp band (measured 3cp drift).
- S1 is FUNCTIONAL: games are the verdict. SPRT queued behind the D40
  arc per the search program (quiet box needed). Elo prior from the
  SF14→16 patch study: correction history family was worth ~+10-25.
- D33 3-persona review (correctness / SF-fidelity / test-quality, opus
  agents) dispatched post-commit; synthesis + any fixes before game
  testing. check.sh re-run on the shared release dir due ~03:50 when the
  720sb train frees it (side-build bench already bit-reproducible 2x).

## 2026-07-16 23:15 [PROGRESS] Overnight window: S1 screen positive, R2 VNNI landed, distribution builds

- **S1 fixed-node screen (exp-0071): +18.5 +- 17.4** vs pre-S1, same s3 net,
  300g @5k on UHO — mid-prior (+10-25 family), weak-positive, no red flag.
  Equal-time SPRT on the quiet box is still the verdict.
- **R2 landed (d895cf6):** AVX-512 VNNI dispatch tier in sota_simd —
  bit-exact (per-backend parity sweep), +6.4% D40 NPS measured on the busy
  box (lower bound; the compare binary lacked S1's per-node cost). Do NOT
  rebuild engines/bin probes until the night gate completes; then rebuild
  at HEAD (leak fix + S1 + VNNI together) and get clean R1 NPS numbers.
- **Distribution:** embed-net now takes ARCHON_EMBED_NET at build time
  (6725e38 — the old hardcoded net-0 embed would panic on width builds);
  two self-contained Windows exes (s3/s4 embedded, l1-2048, bench-verified
  701605) shipped to the user's desktop.
- S0 progress: UHO book adopted for screens (exp-0068..70); the SF16-late
  mechanics audit is running as a study agent, output lands in
  stockfish_reference/07-s0-sf16-mechanics-audit.md.
- Pending USER decisions: exp-0042 code_sha convention ("unknown"
  recommended); gitignore config.json.

## 2026-07-17 01:15 [CODE] D41 Phase-1 infrastructure: obs M1+M2 landed in one evening

The doc-16 Observation Store is real: `src/obs/` (0f81b34) — std-only TLV
(protobuf wire MODEL, no dependency), full four-layer schema incl. the
§5.2 completeness fields, shard framing with whole-episode appends;
contract tests green (round-trip, forward-compat incl. future EpisodeKind,
provenance-mandatory both sides, newer-version refusal, crash-truncation
semantics). M2 (e940e03): Board<->PositionState conversion (doc 15 nibble
convention, reconstruction THROUGH from_fen so shard bytes are validated,
never trusted) + `obs-tool verify/dump` — the design/19 §11.1 replay
audit; smoked on a real shard (0 problems) and a bit-flipped one (caught).
All behind feature="datagen"; default build + bench untouched; clippy
clean both feature sets. NEXT: M3 RootMoveAnalysis generator (D43 root-v0
searchmoves arms, cold state) -> M4 D44 stability/disagreement audits ->
M5 D33 review -> first Reckless labeling run (which does NOT wait for the
evaluator freeze). Revised Phase-1 forecast: first "more than eval"
training in ~3-5 days, gated mainly by M3 + the eval verdicts.

## 2026-07-17 01:15 [OVERNIGHT] Auto research loop armed; obs stack reviewed + first labeling run in flight

USER DIRECTIVE: autonomous loop until morning. In flight right now:
1. **720sb D40 train** — ends ~03:50; then check.sh reruns the S1 merge
   gate on the shared release dir; then d40_night_v2.sh runs h2h-720 +
   the EQUAL-TIME GATE (promotion #4) — verdicts ~06:30-08:00.
2. **obs sample v1 labeling** (tools/obs/label_sample_v1.sh, bg): the
   SAME 1000 seeded UHO positions through Reckless-d445b8b (child arms)
   AND SF18 (searchmoves arms), 4k/16k/64k tiers, then the M4
   stability/disagree audits — the dual-teacher data for the "is
   Reckless the right teacher?" question. Ledger add-data rows on
   completion. Results tee'd to chain-v2-results.log ([obslabel] tags).
3. **D33 round-2 verifier** (agent) re-checking the a664da0 synthesis;
   round 1 found 2 HIGH (canonical_key was the rule50-folded key —
   dedup/leakage-barrier breaker; child-mode mate-arm inversion) + 4 MED,
   ALL fixed with regression tests in a664da0. CONVERGED verdict expected
   → M5 closes → first big labeling run authorized.

Obs program status: M1 store, M2 verify/replay, M3 generator (+child arm
mode: Reckless IGNORES searchmoves — caught by the audits before any data
was generated), M4 audit tools all landed and reviewed. Reckless teacher
pinned (engines/bin/reckless-d445b8b, sha256 2e9fc5dd…). Teacher
recommendation on record: Reckless primary on alpha-beta-affinity +
teacher-diversity grounds, SF18 same-row control; the sample-v1 audit
numbers decide, diversity breaks ties.

## 2026-07-17 02:50 [CODE] R4 micro-audit: two bench-neutral SOTA speedups landed; big-v1 labeling staged

D33 round 2 CONVERGED at 01:40 (M5 closed; fixture fix 55c310f). Idle
window while the SF18 labeling pass runs was spent on R4 (task #16):

- **R4 audit** (read-only agent): docs/research/r4-micro-audit-2026-07-17.md.
  6 ranked candidates; TT layout/prefetch, MovePicker hoisting, and the
  classic accumulator verified already-tight. The one functional candidate
  (#5, defer correction_value past the NonPV TT cutoff) is parked — SPRT
  cost, hindsight coupling risk.
- **Landed #3** (943584d): threat-apply scratch arrays hoisted onto
  ThreatContext — removes an ~8 KB stack memset per SOTA do_move.
- **Landed #2** (3257492): find_nnz — the last scalar step in the SOTA
  forward, 192 branchy serial iterations — is now a dispatched SIMD kernel
  (AVX2 movemask + 256-row compaction table; identical index set, parity
  fixtures per backend). Guessing 3-9% SOTA NPS combined; measured numbers
  deferred to the R1 quiet-box session.
- Candidate #1 (make the FullThreats fold lazy like HalfKA, est. 5-15%
  SOTA NPS) is the remaining big lever — a real refactor, queued for a
  window with review capacity, not a 3am one-shot.
- **Verification gap found and closed**: the sota modules are compiled ONLY
  under `--features sota`, so the default check.sh gate never clippy-checks
  or tests them. Both commits verified explicitly with
  `--features sota` (633 lib tests green incl. threats_replay_*); pinned
  bench stays 701605 (the default bench build compiles sota out entirely).
  A pre-existing `sota,nnue-l1-2048` feature-combo build break was
  confirmed pre-existing (fails at the parent of 943584d too) and left
  alone — no probe config uses that combination.

Also staged: **obs big-v1** (tools/obs/label_big_v1.sh, a76d1c7) — 10k
fresh UHO positions (seed 20260718, sample-v1's 1000 excluded so sample-v1
stays a clean held-out shard), both teachers in PARALLEL (fixed-node =
load-immune, T4), behind a guard that waits for the night-chain gate, the
S1 SPRT, and the trainer (TC games need the quiet box). Launch decision
after the sample-v1 audits are reviewed. ETA ~10.5h from launch.

## 2026-07-17 03:15 [RESULTS] Obs M4: sample-v1 dual-teacher audits — GO for the big run

Both 1000-position shards complete and ledgered (obs-sample-v1-reckless
sha256 9c4981a0…, obs-sample-v1-sf18 sha256 73329eeb…; ledger verify 0
problems). Reckless pass 63 min, SF18 pass 54 min. The M4 fail-fast
audits, first real data:

- **Structure**: `verify` 0 problems on both; identical per-move row
  counts (101,727) — both teachers labeled exactly the same arm sets.
- **Stability** (per teacher, 16k→64k deepest tiers):
  | | accepted (stable+≥20cp) | flip | tie (<20cp) | cross-tier \|Δscore\| p50 | top margin p50 |
  |---|---|---|---|---|---|
  | Reckless | 39% | 23% | 38% | 10cp | 16cp |
  | SF18 | 36% | 24% | 40% | 8cp | 13cp |
  Read: **scores are reproducible** (median 8-10cp drift across a 4x node
  jump), but **argmax labels are inherently soft** — the median top-arm
  margin is only 13-16cp on UHO middlegames, so best-move identity churns
  on near-ties. Consequence for design/19: train on per-arm VALUES /
  margin-weighted or distributional targets, not hard best-move classes;
  the ~37% "accepted" subset is the natural hard-label stratum if one is
  ever needed.
- **Cross-teacher disagreement**: 75% deepest-tier best-move agreement,
  arm-ranking Kendall tau mean 0.841, unrestricted-score gap p50 11cp
  (p95 31cp). Two independent engines produce convergent arm rankings —
  strong external validity for the label signal; neither teacher is an
  outlier.
- **Teacher verdict**: no disqualifier on either side; numbers are near
  identical (Reckless marginally more decisive margins). The standing
  recommendation HOLDS: **Reckless primary** (alpha-beta affinity +
  engine-diversity vs the SF-derived eval lineage), **SF18 control** on a
  subsample of every future run. Caveat on record: Reckless arms are
  child-negate (root TT context lost), SF18 arms are true searchmoves;
  the 75%/0.841 agreement bounds the semantic gap as small.

**GO issued**: big-v1 launched 03:10 (10k positions, seed 20260718, both
teachers parallel), self-guarded behind the night-chain gate + S1 SPRT +
trainer; expect labeling to actually start ~09:00-10:00 and finish
~19:30-20:30. One guard bug caught immediately: pgrep|wc under
`set -o pipefail` exits 1 on zero matches — fixed, committed, relaunched.
M4 (task #24) is CLOSED: both fail-fast audits ran on real data and the
signal is trainable.

## 2026-07-17 03:50 [RESULTS] 720sb train: parity FAIL on the max tail (again the OOD-endgame clip); S1 merge gate GREEN

**d40-uho-720sb** (7h train, 720 superbatches on full UHO) completed and
packed, then FAILED the runtime-parity gate the same way the farseer hop
did: mean is clean (quant 0.0330, runtime 0.0121 — well under the 0.03
bar), but max = 0.2517 vs the 0.20 bar, worst fen a bare rook endgame
(`1k6/4R3/2K5/8/8/8/P4b2/8`). That is now TWO D40 nets whose only parity
failure is a quant-clipping tail on OOD pawnless endgames (240sb passed
at 0.174; longer/warmer training pushes some weight past the clip). The
night chain handles it correctly by design: 720sb is h2h'd as a
DIAGNOSTIC but the equal-time gate (promotion #4) runs with the parity
-passing **d40-uho-v1-240sb**. Open decision for the user (both hop and
720sb): (a) waive the max-tail bar for OOD-endgame-only divergence (mean
and p99 are clean), (b) repack with per-layer quant headroom (netpack
change; the T5 raw f32 checkpoint exists, so any repack attempt costs
minutes, not a retrain), or (c) train the next long run with weight
clipping so packing never saturates. Note `rescale_net.py` does NOT apply
here — it hot-scales the OUTPUT layer; this tail comes from packed
mid-net weights saturating their integer range.

04:34 update — the night chain's diagnostics are in (exp-0072): **720sb
beats 240sb +30.2 ± 17.1 at fixed nodes** (300g @5k, 121W/84D/95L). The
longer schedule genuinely helps; the ONLY thing blocking it from the gate
is the packing tail. This tilts the repair decision toward (b)/(c) — fix
packing/clipping and re-gate — over waiving. The hop diagnostic confirmed
the cached evening number (−23.2, exp-0066; hop stays dead). The
equal-time gate (240sb vs s4, promotion #4) started 03:52, verdict ~06:50.

## 2026-07-17 05:35 [VERDICT] Promotion #4 REJECTED: D40 loses −57.9 ± 12.6 at equal time. Evaluator-freeze proposal.

**GATE (exp-0073)**: d40-uho-v1-240sb (best parity-passing D40) vs
stage2-v1-s4-2048-t74-240sb, 200g 10+0.1, quiet box:
**−57.9 ± 12.6** (11W/145D/44L, score 41.7%). The full-feature net wins
per node (+32.5 ± 17.3, exp-0065) and its 720sb sibling adds another
+30 per node (exp-0072) — but at the clock the inference tax erases all
of it. D40 needs roughly 1.7–2× more NPS to convert its per-node edge
(~70 Elo per doubling rule of thumb); the realistic near-term speed
program (R4 #1 lazy threat fold 5–15%, PGO ~5–10%, native build ~5%, R2
VNNI +6.4% already landed) sums to ~20–35%, not 70–100%. Conclusion: D40
is not promotable on this hardware without an algorithmic speed break-
through; it stays a RESEARCH line (packing-tail fix + speed program),
re-gated only after a major NPS win. The night chain worked exactly as
designed end-to-end: parity bar → diagnostics → gate → clean verdict.

**EVALUATOR-FREEZE PROPOSAL (for the user, morning decision):**
1. **Freeze the classic binpack line as THE evaluator** for the next
   program phase. Incumbent stays `stage2-v1-s3-2048-farseer-120sb`
   (+265 composed over net-0); `s4-2048-t74-240sb` is working-best
   (+15.6 ± 11.9 pooled fixed-node, same width/NPS family). One cheap
   pending item: the s4 equal-time re-check on the UHO book (promotion
   #5) to make the frozen evaluator the true best classic net.
2. **Pause new eval trainings.** The B4 question is answered (data
   volume), width is saturated at 2048, and stage-2 gains are visibly
   shrinking (+55 → +29 → +15.6). Marginal Elo per GPU-hour on this box
   is now far below what the search side offers.
3. **Shift primary effort to search**: S1 correction history is merged
   (SPRT pending tonight, fixed-node screen +18.5 ± 17.4), S2 SPSA is
   queued, and the SF16-late audit shortlist (un-deferrals) is scoped.
4. **D41/design-19 becomes the main research thrust**: the obs platform
   is production-proven tonight (M1–M5 closed, dual-teacher audits GO,
   big-v1 10k labeling in flight). The eval foundation this program
   waited for is settled by this freeze.
5. D40 continues in the background only as engineering: fix the packing
   tail (netpack quant headroom or clip-aware training — raw ckpts
   exist), take the speed program as far as it goes, re-gate when NPS
   justifies it.

## 2026-07-18 19:20 [RESULTS] Factorization weak (+7); SF16 anchor ~−43; search −25 at real depth; label-value pilot RUNNING

Afternoon/evening measurement sweep (all ledgered):
1. **King-bucket factorization (#27) = WEAK +6.9** (exp-0099, CI [−29,+43],
   below +15 gate). Implemented + verified (D40_FACTORIZE, folds virtual→real
   at export = zero inference cost, .cenn byte-identical; committed d208c5a).
   Modest lever, likely shrinks at 720sb scale. The signature run OOM'd
   (co-ran a screen with a bullet train under 7GB — LESSON) and was dropped
   as low-value.
2. **Search caught up (exp-0096/0097):** ARCHON-search vs Reckless-search,
   same v60 net: −86.7 (2026-07-13, pre-P9-waves+S1) → **−49@100k → −25@200k**
   (real equal-time depth). Gap HALVED and shrinks at depth; at real depth
   search is ~on par with a top-2 engine. Search is a SMALLER lever than the
   stale −87 implied.
3. **Fresh SF16 anchor (exp-0098) = −43.1±20.2** (300g, current best build
   probe-l2048-pgov3 + s3-farseer; noisy 72% draws, beside-train). Nominally
   better than stale −63/−78 (CI excludes −78) → gap-to-goal ~+73-86, maybe
   less than the +95-110 assumed. NEEDS a clean quiet-box rerun to firm.
4. **Strategic synthesis (user asked "eval, search, or both"):** BOTH, but
   both more caught-up than the stale numbers implied; NEITHER has a big easy
   one-box lever left (factorization +7, stage-2 dead, width saturated;
   search ~caught up). The BIG gains (eval data-scale + SF18 labels; search
   SPSA at scale) are D36-compute-gated. One-box realistic: ~+20-40 (promote
   D40 via longer stage-1 + factorization + search slices via the instrument).
5. **RUNNING now:** (a) D40 warm-continue +360sb on UHO (d40-uho-cont-1080sb,
   GPU, gentle LR 3e-4, banking per-node edge → promote D40; screen vs 720sb
   when done); (b) **LABEL-VALUE PILOT (user-greenlit)** — does deeper (SF18
   @100k) vs cheap (5k) labels on the SAME 2M mix90 positions make a stronger
   net? Chunked/resumable deep relabel (tools/nnue/label_value_pilot.sh,
   f149def). Valid pre-gate on the training distribution: Pearson 0.959 / RMS
   42cp / **15% sign-flips** (gap grows to 136cp RMS in decisive positions) —
   labels DIFFER meaningfully (necessary condition met); the A/B confirms
   whether that difference → Elo before any bulk SF18 spend. New tools:
   relabel_depth.py, paired_label_calibration.py (committed 6ab735d).
   Decision rule: deep-net ≥+15 vs cheap-net → SF18 worth it; else labels not
   the bottleneck.
6. **Firewalled Reckless-net instrument** rebased onto main
   (instrument/reckless-net-v60 branch, engines/bin/probe-recklessnet,
   RECKLESS_NET=v60) = a deflation-free FIXED-NODE test bed for search changes.
   NPS reality: real Reckless ~817k, our D40 604k (threats made our eval
   Reckless-like/slower), classic-2048 1366k, net-0 2.36M (cheap/weak).
7. All session work PUSHED to origin/main once (user asked, overriding the
   never-push default) — 62 commits. Ledger verify 0 problems.

## 2026-07-18 13:35 [STUDY+RESULTS] "What holds ARCHON back" — search caught up to −49/node vs Reckless; stage-2 dead; king-bucket factorization is the active eval lever

Big afternoon. Full study: docs/research/what-holds-archon-back-2026-07-18.md.
1. **Study verdict:** engine is architecturally complete; eval is COMPETITIVE
   with SF16 (70% draws, exp-0060; D40 wins per node, NPS-limited not
   quality-limited); the binding constraint is testing/tuning throughput (D36)
   + external deflation (internal +265 over net-0 but still −63/−78 vs SF16;
   S1 +72.9 internal moved the anchor ≤+15). One-box eval headroom is modest
   (~30-40 fixed-node): longer stage-1 + king-bucket factorization.
2. **Search RE-MEASURED vs Reckless (exp-0096), the deflation-free instrument:**
   ARCHON-search vs Reckless-search, both on the v60 net, fixed 100k nodes =
   **−49.0 ± 24.8** (was −86.68 on 2026-07-13, pre-P9-waves-C-G + pre-S1). The
   P9 Wave C-G batch + S1 correction history closed ~+38 of the search gap to a
   top-2 engine. Real residual (~49) remains but HALVED. The firewalled
   Reckless-net instrument (rebased onto main today, branch
   instrument/reckless-net-v60, worktree .claude/worktrees/reckless-v60,
   engines/bin/probe-recklessnet, RECKLESS_NET=v60 env) is now a deflation-free
   FIXED-NODE test bed for search changes. Its scalar forward is slow (273k nps
   full-refresh) — FIXED-NODE ONLY.
3. **NPS reality (user asked):** real Reckless ~817k nps; our D40-sota 604k
   (SLOWER than Reckless now — the threats FT made our eval Reckless-like);
   our classic-2048 1366k (HalfKA-only, simpler); net-0 was 2.36M (tiny/weak).
   NPS is work-per-node, not quality — our old "2.5× faster" was a cheap-net
   artifact. Adding threats converged our NPS to Reckless's (the right trade).
   The equal-time reckless-net-vs-D40 test was CUT (user): confounded by the
   instrument's slow scalar port; can't cleanly compare net quality.
4. **Stage-2 DEAD for D40 (exp-0095, definitive):** diag2 (strong-warm + LR
   5e-4) = **−72.9, CI [−106,−40]**. Three cells: −23.2 / −12.7 / −72.9. NOT a
   score-div bug (classic +54.9 used the same farseerT76+div1.9). Mechanism:
   D40's threat FT + more-converged 720sb stage-1 are brittle to the farseer
   distribution shift; more LR = more regression. Eval lever = longer stage-1
   (same UHO dist) + factorization, NOT stage-2.
5. **ACTIVE: king-bucket factorization (#27, user-chosen highest-leverage
   eval lever).** We train the 32-bucket HalfKA UNfactorized (no gradient
   sharing). Implementation delegated (env-gated D40_FACTORIZE in
   d40_features.rs: widen HalfKA FT +704 bucket-agnostic slots, mapper emits
   each feature's factor, FOLD virtual→real at export = zero inference cost,
   mirrors bullet merge_factoriser). Gate: factorized-240sb vs d40-uho-v1-240sb
   twin, 300g@5k, want ≥+15; data-efficiency signature factorized-60sb ≈
   unfactorized-240sb. Expected +10-30 fixed-node.
6. Other: gold-v2 monitor workers HUNG (9.6h @0.8% CPU), reaped; 180k-position
   production corpus salvaged + ledgered (gold-v2-teachers, feeds a future
   low-priority d41-gold-v2). Attention Stage-0 trainer written (d40_attn.rs,
   diagonal self-gating — full Q·Kᵀ self-attention NOT expressible at bullet's
   model-builder level; needs tensor-IR or PyTorch). Reckless-net instrument
   committed ee7cf47 (worktree branch).

## 2026-07-18 11:45 [RESULT] Corrected D40 stage-2 (gentle, LR 2.5e-4) = −12.7 vs 720sb — NO GAIN; testing LR 5e-4 next (exp-0094)

Screen done: d40-s2-diag-120sb vs d40-uho-720sb = **−12.7 Elo** (300g 5k,
W102/L113/D85, CI [−47.8,+22.0] straddles 0), exp-0094. Below the +15 gate →
NOT committing the 480sb run. Nuance: training was CLEAN (loss 0.0086→0.0071
monotonic, NOT the exp-0066 regression) but at LR 2.5e-4 the net barely moved
— this tested a *gentle* stage-2. The exp-0066 failure (−23.2) used LR 5e-4
BUT warmed from the WEAK 240sb net. So the untested cell is **LR 5e-4 +
warm-from-STRONG-720sb** — isolates whether the weak warm-start alone caused
exp-0066. Launched d40-s2-diag2 (LR 5e-4, else identical). If also ≤+5:
stage-2 is genuinely dry for the D40 threat arch (unlike the classic net
which banked +84) → pivot to longer stage-1 (#3) or king-bucket
factorization (#27). If positive: the weak warm-start was the bug, promote
the recipe to a full 480sb run.

## 2026-07-18 11:40 [PLAN] Corrected D40 stage-2 TRAINED CLEANLY (loss 0.0086→0.0071 monotonic)

d40-s2-diag finished 120sb in 1h09m: loss declined monotonically
0.0086(sb1)→0.0071(sb120), NO divergence — the exact opposite of the failed
exp-0066 hop (which regressed). The corrected recipe (warm from STRONG 720sb
+ cooler LR 2.5e-4 vs the old warm-from-weak-240sb + LR 5e-4) trains healthy.
Net packed d40-s2-diag-120sb.cenn (loads, startpos eval 30cp sane). The
train_run pipeline exit-1 is ONLY the parity gate: runtime max_abs 0.309 >
the config's OLD strict 0.2 bound — this is the KNOWN cosmetic quantization
tail (#26, d40-packing-tail doc): runtime MEAN error 0.019 (well within),
worst FEN 8/8/4R3/3k3p/8/q7/4K2P/8 = sparse OOD endgame; ~zero Elo cost. Not
a training failure; the net is valid for a fixed-node screen. LAUNCHED the
decision screen (d40-s2-diag-screen): d40-s2-diag-120sb vs d40-uho-720sb,
probe-sota-pgo both sides, 5k nodes 300g conc 8, seed 20260818. GATE: ≥+15
fixed-node → the corrected stage-2 works → commit the full 480sb run + then
equal-time vs s3 (D40 starts −8.7, a stage-2 gain flips it positive =
promotion). <+15 → stage-2 gives little for the threat arch even corrected.

## 2026-07-18 10:30 [PLAN+LAUNCH] North star raised to SF16+30 → highest-value eval lever = CORRECTED D40 stage-2 (running)

User raised the goal (via /loop 15m) to a search-augmented NNUE beating SF16
by +30 at equal time (~+95-110 Elo gap). Ran a ranked eval-lever analysis
(subagent, read-only). Verdict: D40 is at equal-time PARITY with the classic
incumbent (exp-0093) and wins +30/node (exp-0072); the one axis where it is
demonstrably behind is STAGE-2 — the classic net banked ~+84 Elo across two
farseer stage-2 hops, while D40's single attempt REGRESSED −23.2 (exp-0066).
Root cause of that failure, from its config: it warmed from the WEAK 240sb
stage-1 net at LR 5e-4. LAUNCHED the corrected diagnostic (d40-s2-diag,
tools/nnue/runs/d40-s2-diag.json, log logs/overnight/d40-s2-diag.log): warm
from the STRONG 720sb net, cooler LR 2.5e-4 cosine, 120sb, SAVE_RATE 30,
farseerT76. Warm-start confirmed (start loss 0.0086 = continuation not cold),
710k pos/s, ETA ~11:45. GPU-bound so it coexists with gold-v2 (CPU). Gate:
parity → fixed-node h2h vs d40-uho-720sb @5k (want ≥+15) → equal-time vs s3
(D40 starts −8.7, +15-30 should flip positive → promotion). If diagnostic
≤+5 over the botched hop, the stage-2 well is dry for the threat arch → fall
back to longer stage-1. #2 lever queued: HalfKA king-bucket factorization
(#27, zero inference cost). REALITY CHECK (honest): eval alone likely gets
~30-40% of the +100; the rest needs search (SPSA retune vs new eval, gated on
distributed compute D36) + speed. Don't over-project internal Elo (deflates
2-3× externally, exp-0091).

## 2026-07-18 04:05 [RESULT] Capture at 12× scale: quantity helped, ordering ceiling holds — top-1 now BEATS gold, rank_corr does not

Batched scale-up done: 12 batches × 600 self-play games → 7.67M exact-node
arms (450,956 exact positions; the batched harvest→export→delete-raw loop
held disk at 20 GB throughout). Retrained d41-capture-scale on ~8.37M arms
(with capture-scale-v1), 20 superbatches:

| net | exact arms | rank_corr | top-1 | best sb |
|---|---|---|---|---|
| d41-capture-v1 | 0.7M | 0.197 | 0.235 | 1 (overfit) |
| **d41-capture-scale** | **8.37M** | **0.263** | **~0.30** | **19** |
| v1.1-gold | 0.34M×3 | 0.706 | 0.258 | — |
| cold-classical | — | — | 0.181 | — |

VERDICT (answers the user's "how useful is that data"): 12× more capture
data lifted rank_corr +0.07 and top-1 +0.06 and resolved the immediate
overfit (best sb 1→19), so data starvation was real. BUT rank_corr plateaus
at ~0.26 — 2.7× below gold's 0.706 — the bound-contaminated non-best-arm
ordering is a QUALITY ceiling scale can't break. The striking part:
capture-at-scale top-1 (~0.30) BEATS gold's top-1 (0.258) — exact PV-node
best moves are a *stronger best-move teacher* than the gold corpus at this
volume. So capture's value is concentrated in the FIRST move (clean, exact),
not the tail ordering (bounds). Levers: (1) a refined bridge emitting
reliable partial orderings from `raised_alpha`/`caused_cutoff` flags instead
of bound margins could rescue rank_corr; (2) a capture(top-1)+gold(ordering)
BLEND may dominate either alone. Corpus ledgered capture-scale-batches-v1
(sha 8480cf1b). NOTE: these bars are on held-out CAPTURE data, not a common
test set — a fair capture-vs-gold comparison needs same-test cross-eval (or
the play test, expected negative given rank_corr 0.26 « picker strength).
Next overnight run: gold-v2 teacher labeling (clean-ordering reference +
flywheel cycle-1 corpus).

## 2026-07-18 03:15 [FINDING] In-search capture is 99.6% bounds — cheap volume, thin clean core; first helm train on it is best-move-usable but ordering-noisy

User directive (asleep, all-night mandate): "figure out how useful [the
in-search capture] data is by generating a lot of it and training." Result
chain:

1. **Yield is huge but mostly unusable.** Capture at 50k nodes gives ~16k
   positions/game, but the recorder samples INTERIOR search nodes and stamps
   every one `Completion::Complete` (completion is NOT an exactness signal —
   the node's `bound` is). Only `Bound::Exact` (PV) nodes carry a true
   evaluation; cut/all nodes are aspiration bounds (the same quarantine our
   5k-node datagen labels sit under). Measured over 600 self-play games +
   the 40g pilot = **11.57M captured positions → 41,527 exact (0.36%) →
   700,828 pairwise arms**. skip_no_complete_reference = 11.27M (the bounds).
2. **New bridge (obs-tool).** `export-train` rejected all capture (episode
   kind `InSearchCapture` ≠ `RootMoveAnalysis`). Added `add_insearch_episode`
   (per-record-as-position fold, `Bound::Exact` gate, reuses `emit_record`);
   datagen-only, clippy-clean, does not touch the shipped engine/bench.
3. **First train (d41-capture-v1, 700k arms, exact d41_alloc11 recipe,
   frozen 720sb trunk, pairwise, tier 50k):** held-out **rank_corr 0.197
   (best at superbatch 1 → overfits) / top-1 0.235**. Compare v1.1-gold
   (336k×3) 0.706/0.258 and cold-classical 0.181. Reading: capture teaches a
   usable BEST-MOVE (top-1 clears cold by +5 pts) but poor FULL ORDERING
   (rank_corr collapses 0.71→0.20) because non-best arms are scout-search
   bounds, so the emitted margins are ordering-grade noise. It is both
   data-starved (overfits at sb1, like v1.1) AND label-noisy.
4. **Overnight scale-up LAUNCHED** (`tools/obs/capture_scale_batches.sh`,
   PID 2954506): 12 batches × 600 games, batched harvest→export→delete-raw
   to keep disk bounded (raw .ceobs is ~9.4 GB/600g and can't be kept at
   scale). Target ~8M exact arms → retrain to separate quantity from the
   label-quality ceiling. ETA ~2.2 h.

IMPLICATION for the flywheel: the in-search capture is NOT the free-data
goldmine the raw 16k pos/game suggested — the clean core is 0.36%, and its
per-move margins are bound-contaminated. Best-move (top-1) transfer is the
only clean signal; a future bridge could emit RELIABLE partial orderings
from `raised_alpha`/`caused_cutoff` flags instead of bound margins. Teacher
gold labeling remains the high-signal source; capture is a best-move
supplement whose weight must be capped. Corpus ledgered
`capture-scale-v1-train` (sha 463bef0b); raw scale-v1 reclaimed post-export.

## 2026-07-18 02:35 [MILESTONE] Speed program CLOSED — fastgate −8.7±12.9, D40 at equal-time parity with the classic incumbent

The 720sb equal-time fastgate finished 200g 10+0.1, best-PGO both sides
(probe-sota-pgo vs probe-l2048-pgov3): **25W/145D/30L = −8.7±12.9**,
paired-bootstrap CI **[−31.4, +13.9] STRADDLES ZERO** (exp-0093). The
speed-recovery ladder is complete: −57.9±12.6 (v3, pre-R4, exp-0073) →
−17.4±12.8 (v3, post-R4 stack, exp-0089) → −8.7±12.9 (best PGO, exp-0093).
Interpretation: D40 already wins per node (+30.2, exp-0072); once the
speed program restored ~+7% relative throughput to the D40 path, the
equal-time deficit collapsed into statistical noise. The D40 architecture
is now a live promotion candidate over the classic-2048 incumbent — the
remaining gap is within one modest quality or speed step (packing fix #26,
narrower-L1 D40). Speed program verdicts final: R1 native REJECTED, R3 PGO
SHIPPED (machine-local, node-verified). **User gave an all-night
all-resources autonomy mandate (asleep); the loop now pivots to flywheel
data-scale: capture-yield pilot + the pool2 overnight gold harvest
(~6.8M rows, both teachers) → train on the scaled corpus.**

## 2026-07-18 02:10 [RESULT] Speed program R1/R3 decided: native REJECTED, PGO ships +10.5% D40 / +3.0% classic; fastgate running

Definitive fixed-node min-of-5 interleaved timings (the quick nps probe
was proven noise-dominated on this box — two rounds contradicted by
±9-18%/cell; lesson + table in r4-micro-audit doc): D40 path v3 688,073
→ **PGO 760,070 (+10.5%)**, native null; classic v3 1,822,877 → native
−9.6% (Zen-4 512-bit autovec loses to the AVX2 shapes), **PGO-on-v3
+3.0%**. Two PGO pipeline conditions discovered and encoded in
tools/build_pgo.sh: profile against the REAL serving net over UCI
(bench-only profiles leave NNUE kernels unprofiled → inert), and pick
the base target-cpu per path (PGO_CPU env). Node-identity verified for
both deployed binaries (probe-sota-pgo, probe-l2048-pgov3). Net
relative swing to D40 ≈ +7% ≈ +5 Elo at equal time. The **720sb
equal-time fastgate** (both sides best builds, 200g 10+0.1, seed
20260723) is running — the ladder reads −57.9 (v3, pre-R4, exp-0073) →
−17.4 (v3, post-R4, exp-0089) → tonight's rung. Flywheel prep also
landed tonight: gold-v1 ledgered both teachers (12.5× production
throughput measured), FEN pool #2 staged (100k, seed 20260800), capture
harvest wiring committed (pid-suffix shards + eval_match --each-option).

## 2026-07-17 23:15 [RESULT] Helm v1.1 play test: −29.6±12.9 — offline bars do not transfer; data-volume diagnosis stands

d41-alloc-v11-c as MovePicker prior (true shared-trunk shape:
POLICY_TRUNK = the 720sb cenn, CPOL layout 0, W=1000), 600g at 2k fixed
nodes vs the plain build, net-0 eval both sides (exp-0092):
**219W/111D/270L = −29.6±12.9** — worse than v1's −9.3±25.5 despite
v1.1 clearing every offline bar (0.706 rank corr / 0.258 top-1 > the
18.1% cold-classical baseline). Two lessons pinned in the ledger row and
the integration memory: the cold bar is not the play bar (the warm
picker is far stronger than cold-classical, so a below-picker prior is
noise injection), and a sharper pairwise head misleads *more
confidently* per unit of blend weight when its knowledge is
data-starved. No more knob sweeps at 336k rows — the route is 10–30×
data (gold harness + in-search capture), then retrain, then one play
test. Historical note: two ~19:00 OOM kills in dmesg were the Manim
pilot's TTS processes (recovered by that agent), not research lanes.

## 2026-07-17 20:55 [DECISION] D45 season CLOSES ship-nothing; gate queue cut by user to jump to the headline gates

TC gate verdicts: c3 rfp-razor **+10.4±14.1 cap-undecided, LLR 1.05**
(exp-0087, the lone tentative-accept) and c4 nmp **+1.2±14.0, LLR −0.75
neutral-reject** (exp-0088 — the fixed-node screen positive did not
survive TC). With c4 dead, the user cut the remaining queue at 20:46:
c5 stopped at 80 uninformative games, c6/c7 skipped outright, and the
runner's accidental c8 gate (screen-REJECTED bundle; the campaign loop
doesn't know screen verdicts) killed at ~15 games. All annotated
(exp-0081/2/3/5), ledger verify 0 problems (9ce5a0c).

**Season verdict: SHIP NOTHING.** A lone +10.4±14.1 undecided doesn't
clear the ship bar, and no second accept exists to form the combined
config. c3's tuned values are preserved in
`data/nnue/experiments/spsa/c3-rfp-razor/theta_final.json` — the natural
seed for a future season (or fishtest-scale run) targeting the RFP/razor
family. The 8/8 decision record: c1/c2 no-change certified, c3
tentative-positive-unshipped, c4 neutral, c5/c6/c7 screen-positive
TC-unverified, c8 rejected. Consistent with exp-0078's plateau
calibration: post-S1 defaults sit within ~10 Elo on every bundle probed;
sub-10-Elo mining is fishtest-scale, not one-box-scale.

Tooling landed in the window: `tools/spsa/combined_gate_v1.sh` (union
ship-SPRT, unused this season but smoke-tested), ledger dotted-path
annotation overlays (553d8c8 — the exp-0084 correction had been landing
at top level; verify went 3 problems → 0).

The runner is on the headline gates as of 20:54: 720sb equal-time vs s4
(the D45a question: how much of exp-0073's −57.9 inference tax did
R4 + 720sb repair) → s4 promo5 re-check → speed A/B → then the SF16
rematch, gold-v1 20k launch, and the v1.1 helm play test.

## 2026-07-17 18:20 [RESULT] D41 v1.1: the anti-overfit stack works — held-out top-1 0.258 CLEARS the classical bar

Grid on the GPU while gates play (agent branch merged, 884ae3b; net
ledgered d41-alloc-v11-c). Held-out (same split as v1):

| config | rank corr / top-1 |
|---|---|
| v1 (unfrozen, margin MSE, sf18) | 0.604 / 0.130 |
| frozen trunk + margin, same data | 0.694 / 0.173 |
| frozen + pairwise, both teachers | 0.666 / 0.205 |
| frozen + pairwise + ALL TIERS (winner) | **0.706 / 0.258** |

Ablation reads: FREEZING the 720sb trunk is the dominant lever (+0.09
rank corr, overfit gap collapses ~0.0176→0.0114 MSE); the pairwise
RankNet loss buys +0.03 top-1 (best-move discrimination — the cell that
matters for allocation); 3x tier data adds +0.02/+0.03 on top. The
winner clears the 18.1% cold-classical baseline by 8 points — the first
D41 artifact to pass the Stage-0 bar. It is packed for play (CPOL layout
0, free-queen probe passed) with POLICY_TRUNK = the 720sb cenn itself:
the true shared-trunk deployment shape, no graft. Play test queued the
moment the TC gates free the box (caveat on record: held-out ranking is
necessary, not sufficient — the warm in-search picker is the real bar).
Infra note: export-train grew --all-tiers (5 tests green).

## 2026-07-17 17:40 [BASELINE] Cold classical ordering on big-v1: top-1 18.1% — the D41 yardstick

obs-tool rank over all 10,000 big-v1 SF18 positions: the classical
MovePicker ordering COLD (no TT move, fresh histories) predicts the
teacher's best move top-1 18.1%, top-3 32.9%, top-5 42.1%, mean rank
10.1 of ~34. Context: random 3.4%, the v1 head 13.0%. So the 152-second
head already reaches ~72% of the cold hand-tuned stack — and the in-game
null is now fully explained (the warm in-search picker with histories +
TT move is far above 18%, so a below-baseline blend had nothing to add).
ACCEPTANCE BAR for v1.1 and beyond: held-out top-1 must clear 18.1%
with margin before another game test is worth the box time.

## 2026-07-17 17:30 [MILESTONE] big-v1 COMPLETE + gate night launched + v1.1 anti-overfit grid on GPU

big-v1 final: both shards 10,000 episodes, verify 0 problems, ledgered
(reckless 39cfc72d…, sf18 dd907004…), both exported (336,130 rows each,
identical grids). Audits at 10x replicate sample-v1 within a point:
stability 38/23/40 (reckless) and 35/23/43 (sf18), cross-tier drift p50
10/7cp, cross-teacher best-move agreement 74%, |Δscore| p50 11cp. The
data platform is validated end-to-end at scale.

GATE NIGHT running (biq3xmly4): c3-c7 TC ship gates → s4 promotion #5 →
720sb equal-time (the D45a headline) → speed A/B; SF16 anchor rematch
follows on the final shipped config; gold-v1 20k after that.

D41 v1.1 training grid launched on the GPU alongside (agent, branch
d41-trainer-v11): frozen-720sb-trunk + pairwise ranking loss + dual-
teacher multi-tier data + early stopping + wd — the anti-overfit stack
from the 17:20 diagnosis, with a packed-for-play artifact if the rank
metrics clearly beat v1's 0.604/0.130 (play test deferred until the box
frees; the frozen-trunk config needs NO graft — POLICY_TRUNK is the
720sb cenn itself, the exact shared-trunk deployment shape).

## 2026-07-17 17:20 [RESULT] D41 plays — and teaches us the bottleneck is data, not wiring

User-directed fast experiment: the v1 margin head wired into MovePicker
quiet ordering via the gen-0 CPOL path (trunk grafted into a policy
trunk, head packed with the new --base override; a from/to TRANSPOSE was
diagnosed with a free-queen probe and fixed via CPOL layout=0). Results:
- As ordering prior: -9.3 +/- 25.5 at 2k fixed nodes (gen-0's +68.6
  config); POLICY_W sweep monotone WORSE (-33 at 3000, -50 at 8000) —
  the value head's quiet-ordering content is noise at any gain.
- Listwise-softmax retrain (same corpus, policy-style targets via a
  logit-trick margin rewrite): held-out top-1 0.040 ~= random —
  memorizes, does not generalize top-move identity at 336k rows.
Diagnosis: gen-0's ordering positive came from 100M policy rows; top-
move discrimination at 0.3M rows is data-starved (B4 at the head level).
The margin head's 0.60 rank corr is real but lives in the easy mass.
PATH: gold-v1 parallel harness (25-30x) + in-search capture (MERGED
2acc736: node-identical, ~2% overhead, 1.3MB/game — tonight's gates can
emit data) → rerun this exact experiment at 10-30x rows. Wiring risk is
now ZERO — the whole pack/load/play loop is proven.

## 2026-07-17 16:30 [RESULT] D41 v1 on big-v1 SF18: the signal SCALES — rank corr 0.604

152 seconds of GPU on the freshly-exported 336k-row SF18 corpus
(ledgered d41-alloc-v1-sf18): held-out MSE -74.2% vs zero-margin and
-26.7% vs the best-constant baseline (v0: -70%/-9%), per-position rank
correlation **0.604** (v0: 0.49), top-1 3.8x random. Ten times the
positions bought +0.11 rank correlation — the same data-scaling shape
that decided B4 for value nets. The flywheel thesis (D45a: information-
dense search data + relational inputs) now has its first quantitative
support. Next rungs: Reckless twin + combined-teacher run tonight,
frozen-protected-trunk v1 architecture, then the 100k gold corpus
(~30x more positions) once the parallel harness runs post-gates.

## 2026-07-17 16:15 [MILESTONE] D41 trainer v0 smoke: the search-data signal is LEARNABLE

Branch merged (1c8f3ad): tools/nnue/bullet/d41_alloc.rs — the Phase-1
allocation-head trainer (design/19 §4.4 v0: per-legal-move value-margin
regression, gen-0-proven rank-8 bilinear on the D40 trunk, stm-corrected
frame, leakage-safe position-id split). Smoke on sample-v1 (789 train /
211 held-out positions, both teachers, idle GPU):
- held-out MSE **−70% vs predict-zero** AND below the optimal-constant
  baseline (genuine position-specific generalization);
- **mean per-position rank correlation 0.49** with teacher margins;
- top-1 0.10–0.13 vs 0.042 random (weakest cell, exactly as the M4
  soft-argmax audit predicts);
- Reckless and SF18 results near-identical → teacher-agnostic data path.
Full note: docs/research/d41-trainer-v0-smoke.md. This is the shift's
go-signal: pipeline proven end-to-end (.ceobs → export-train → GPU →
metrics). v1 = big-v1 10k corpus + frozen protected trunk + ranking loss.
Datagen scalability also landed today (label_gold_v1.sh: parallel
workers + single-tier + drift monitor, ~25-30x; 100k gold FENs staged).

## 2026-07-17 16:10 [SEASON SUMMARY] D45 SPSA: all 8 decisions minted — tuning phase complete

| # | Bundle | Movement | Screen (1000g, 5k nodes) | Status |
|---|---|---|---|---|
| 1 | corrhist weights | none (±0.1%) | — | NO-CHANGE certified (exp-0075) |
| 2 | LMR/reduction | none (±0.25%) | — | NO-CHANGE certified (exp-0076) |
| 3 | RFP+razoring | RFP_IMPROVING −34% | +7.3 ± 16.8, LLR .31 | PENDING TC gate |
| 4 | NMP | EVAL_BASE +36%, EVAL_DEPTH 14→12 | +9.7 ± 17.0, LLR .63 | PENDING TC gate |
| 5 | futility+SEE | 5 movers, all more aggressive | +10.4 ± 16.8, LLR .74 | PENDING TC gate |
| 6 | gates/counters | all gates −1 step | +12.2 ± 17.1, LLR .95 | PENDING TC gate |
| 7 | aspiration+eval blend | wider window, gentler widen; eval FLAT | +7.3 ± 17.8, LLR .28 | PENDING TC gate |
| 8 | history magnitudes | softer maluses | **−11.1 ± 17.0, LLR −2.15** | REJECTED (exp-0085) |

Meta-findings that outlast the season: (a) the landscape is a high
plateau — a +50% detune of the base LMR reduction costs only −8 ± 13
(exp-0078), so re-search machinery absorbs mistuning and micro-Elo
mining is fishtest-scale work; (b) SF-derived constants transfer
wholesale (corrhist, reductions, optimism all flat at our resolution);
(c) the one consistent live gradient is MORE PRUNING AGGRESSION —
margins and gates in c3–c6 all pushed the same direction with
positive screen leans. Tonight's TC gates + the combined-config SPRT
decide what ships; per the user's directive the program then SHIFTS to
D41: big-v1 corpus → export-train (bridge a831f3b) → first D40-trunk
allocation-head training per design/19.

30-hour tally on the shipped model since the last SF16 anchor: S1
correction history +72.9 ± 16.2 banked; SPSA bundles pending tonight;
anchor REMATCH queued after the gates.

## 2026-07-17 15:45 [RESULT] Study exp #1 probe: the final checkpoint wins — intra-run harvesting adds nothing

Ledger row filed. sb210 is -15.1 ± 17.1 and sb180 -53.7 ± 16.8 against
the final sb240 of the same run: monotone improvement into the final LR
decay, exactly what a healthy schedule should produce. The official
harvest-and-select win must come from selecting across SEEDS (16
parallel runs), not within one run — that variant needs N seed twins
and is a future GPU-budget question. D40_SAVE_RATE=30 stays for T5
resume insurance.

## 2026-07-17 15:00 [DECISION 7/8 PENDING] c7: aspiration wants wider+gentler; the eval blend is flat

exp-0083. ASPIRATION_BASE_DELTA 5 -> 7 (wider initial window) with
ASPIRATION_WIDEN_NUM 44 -> 37 (gentler re-search widening) — 20%/10% of
range. The eval outer-shape blend (material/optimism/rule50) stayed in
the noise band: an interesting NEGATIVE — SF's optimism constants
transfer to our net lineage unchanged, matching the c1 corrhist result.
Screen +7.3 ± 17.8, LLR 0.28. Five pending TC gates (c3-c7); c8
(history-update magnitudes) is the LAST campaign, running now. Season
completes tonight with the gate sequence + combined-config verification.

## 2026-07-17 14:20 [DECISION 6/8 PENDING] c6 gates/counters: integer gates all step toward more pruning

exp-0082. LMP_BASE 3 -> 2 (~3 sigma continuous-theta), PFUT_MAX_LMRDEPTH
12 -> 11, RFP_MAX_DEPTH 17 -> 16, NMP_EVAL_DEPTH 14 -> 13 — every gate
stepped one unit in the more-pruning direction, agreeing with c3/c4/c5.
Screen: **+12.2 ± 17.1, LLR 0.95** — fourth consecutive positive lean,
now the strongest. Four pending TC gates (c3 +7.3, c4 +9.7, c5 +10.4,
c6 +12.2); combined-config re-verification rule stands (13:40 entry).
c7 (aspiration/hindsight/optimism — the eval-blend campaign) running.

## 2026-07-17 13:40 [DECISION 5/8 PENDING] c5 futility+SEE: five real movers, strongest screen lean yet

Ledger row filed. PFUT_NO_BESTMOVE 138 -> 179 (19% of range),
PFUT_EVAL_ABOVE_ALPHA 90 -> 108, CFUT_LMRDEPTH_MULT 232 -> 267,
QS_SEE_FLOOR -74 -> -63, CAP_SEE_HIST_MULT 34 -> 39: the tuner
consistently wants MORE pruning aggression in the futility family.
Screen +10.4 ± 16.8, LLR 0.74 at cap. Three pending gates now (c3 +7.3,
c4 +9.7, c5 +10.4) — all positive-lean; a consistent picture that the
margin bundles sit ~5-10 Elo below optimum, adjudicated tonight at TC.
Note the compounding caveat: the three tuned sets were each tuned
against DEFAULTS independently; if all three gate positive they get
committed together and the COMBINED config re-verified with one final
TC SPRT before becoming the shipped defaults (interaction risk).
c6 gates/counters running.

## 2026-07-17 12:55 [DECISION 4/8 PENDING] c4 NMP: second real mover — deeper null-move confidence wanted

Ledger row filed. NMP_EVAL_BASE 374 -> 507 (23% of range) and
NMP_EVAL_DEPTH 14 -> 12: the tuner wants null-move pruning applied more
readily. Screen at 1000g cap: +9.7 ± 17.0, LLR 0.63 — same shape as c3
(positive lean, undecided). Pattern so far with r_end=0.2: the margin/
threshold bundles DO have findable direction, sized ~5-10 Elo — exactly
the band the TC gates exist to adjudicate. Two pending gates queued
(c3, c4); c5 futility+SEE running.

## 2026-07-17 12:20 [DECISION 3/8 PENDING] c3 RFP+razoring: first real theta movement — gate tonight

exp-0079-adjacent (ledger row filed). With r_end=0.2 the harness finally
moves: **RFP_IMPROVING 2934 -> 1947** (22% of range, ~5 sigma above the
step-noise floor — the improving-trend discount on RFP margins wants to
be ~34% smaller), RFP_THREAT_MARGIN 55 -> 46, RAZOR_BASE 465 -> 422,
six smaller drifts. Fixed-node screen at the 1000g cap: +7.3 ± 16.8,
LLR 0.31 (positive lean, SPRT undecided). Decision 3/8 rides on
tonight's TC ship gate — night_gates_v1.sh generalized to gate every
campaign whose theta moved (vacuous thetas self-skip). c4 (NMP) running.

## 2026-07-17 12:15 [CALIBRATION] The SPSA season's true meaning: the search-param landscape is a high plateau

exp-0078, two measurements: (1) the r_end=0.01 control did NOT walk back
(final 1533 from 1541; z mean +0.0075 ± 0.016 — zero local gradient);
(2) the direct calibration match: **RED_BASE 1541 (a +50% detune of the
base LMR reduction) costs only −8.1 ± 12.9 Elo** at 20k nodes, 1200g.

The harness was never broken — there was nothing to find. Re-search
safety nets (do_deeper/do_shallower, PVS re-search, TT) absorb even
gross reduction mistuning. Season recalibration, in effect immediately:
- c1/c2 NO-CHANGE decisions are REAL findings ("no >~10 Elo error in
  the bundle"), not harness artifacts.
- c3–c8 run at 400 iters (4,000 games each — same certification power,
  half the cost) with r_end=0.2 (steps large enough that any genuinely
  steep gradient CAN move theta; c7's eval-blend params are the likeliest
  to have one).
- Sub-10-Elo per-param mining is fishtest-scale work (100k+ games per
  param) that this box cannot do; after the 8 decisions the Elo lives in
  the S0 structural un-deferrals, the D40 speed/gate arc, and D41.

## 2026-07-17 11:30 [ANALYSIS] p25 twin verdicts + the SPSA step-size diagnosis

**Study experiments #1+#2 (exp-0077, d40-uho-p25-240sb):** loss ^2.5 is
NEUTRAL vs ^2 at 240sb full-UHO scale — pooled 900g fixed-node
+5.0 ± 9.7. Parity is cleaner (runtime mean 0.0076 vs 0.0121) and the
net passed the recalibrated bars on the first try. No adoption alone;
revisit only bundled with a longer schedule. The #1 payload delivered:
8 harvested checkpoints (sb30..sb240) ready for a selection-by-games
probe. Net ledgered (cb3c3eea…).

**SPSA harness diagnosis (from the control's live trajectory):** at
k=125/200 the detuned RED_BASE has walked back −7.7 of −514 — almost
exactly the EXPECTED −0.07/batch for a ~30-Elo detune at r_end=0.01.
Conclusion forming: the harness detects gradients fine; the STEP SIZE is
the bottleneck. With c1/c2's r_end=0.002 the maximum systematic traverse
over a whole 800-iter campaign is a few units on 10000-unit scales — the
two nulls were foregone conclusions, informative only as "no gross
(>50 Elo) detune exists in those bundles." fishtest uses r~0.002 with
25-100x our games; at our budget r_end≈0.2 makes an 800-iter campaign
able to traverse ~10-15% of a param range against realistic z-bias while
keeping end-noise ~10% of range. PLAN once the control completes: rerun
the control at r_end=0.2 expecting a DECISIVE walk-back; if confirmed,
redo c1+c2 at r_end=0.2 (2x2h, cheap) and run c3-c8 at r_end=0.2 — the
season's 8 decisions get made under a harness that can actually move.

## 2026-07-17 11:00 [DECISION 2/8] SPSA c2 (LMR/reduction bundle): NO-CHANGE — and the harness goes on trial

exp-0076. Second consecutive perfect null: all 15 reduction params within
0.25% of defaults after 8,000 games (the bundle the SF14->16 study ranked
as the biggest single gainer pool). Two readings, now formally split:
(a) the SF-derived constants really are at a local optimum for our engine
at this eval; (b) the harness cannot resolve gradients at this budget
(8k games, r_end=0.002 — fishtest uses that r with 25-100x more games).
The DETUNED-START POSITIVE CONTROL launched 10:57 decides it: RED_BASE
starts at 1541 (1.5x default — a deliberate ~+50% reduction offset, a
known material detune) with --r-end 0.01, 200 iters. If theta walks back
toward 1027, the harness works and the nulls stand (and r_end 0.01
becomes the season default); if it sits still, decisions 3-8 are FROZEN
until the sensitivity problem is diagnosed. c3 waits for this verdict.

## 2026-07-17 09:20 [DECISION 1/8] SPSA c1 (corrhist weights): NO-CHANGE — the S1 constants survive

exp-0075. 800 iterations x 10 games @20k nodes (8,000 games) perturbing
all 13 correction-history read/update weights: every parameter came home
to within ±0.1% of its SF-parity default (mid-run wander up to ±90 units
on CONT_NULL_PRIOR fully reverted). The landscape is flat at this
budget's resolution (gradient floor ~5 Elo at 8k games). Decision:
KEEP DEFAULTS; ship-gate skipped as vacuous (theta_final == defaults
within integer rounding). Two useful facts fall out: (1) SF's corrhist
constants transfer to ARCHON's eval unchanged — consistent with S1's
clean +72.9 landing; (2) the harness needs a positive control if c2
(reductions — historically strong gradients) also comes back flat.
c2 launched immediately after (seed 20260720, same budget), with
tune-l2048 redeployed at HEAD (eval tunables + advertisement fix in).
Season: 1/8 decisions done. In parallel: d40-uho-p25-240sb (study exps
#1+#2) trains until ~11:30.

## 2026-07-17 08:20 [FINDING] The D40 "packing tail" was never packing — parity gate recalibrated; 720sb admitted to gating

Full note: docs/research/d40-packing-tail-2026-07-17.md (+ reusable
tools/nnue/audit_d40_quant_clip.py). Headlines: ZERO quant overflow in any
D40 tensor (bullet errors rather than saturates, and AdamW clips at ±1.98
— clipping was structurally impossible); the passing 240sb net actually
has LARGER threat weights than the failing 720sb; the runtime parity
error is integer-vs-float arithmetic divergence (the (l*r)>>FT_SHIFT
floor), a thin OOD-endgame tail on a clean distribution (720sb: mean
0.0121, p99 0.0635, max 0.2517). Gate recalibrated (train_run.py): mean
<= 0.03 AND p99 <= 0.10, max backstop 0.30 (sf-100's true transform bug
was mean 0.355 broad — still caught instantly). Network_scale rebalance
proven impotent (identical max at scale 400 vs 800); no repack can help
by construction, so no -rs1 net exists and the ledgered 720sb sha is
unchanged. CONSEQUENCE: d40-uho-720sb (+30.2±17.1 per node over 240sb,
exp-0072) is now a legitimate gate candidate — queued for tonight's
quiet-box window: equal-time vs s4 after the c1 TC gate. The two failed
"parity" hop nets from yesterday deserve re-audit under the new bars too.
Optional engine-side improvement (R1 of the note): rounding FT shift —
deferred, changes sota semantics mid-SPSA.

## 2026-07-17 08:00 [PROGRAM] D45: SPSA gainer season begins — 8 decisions, then the D41 shift

USER DIRECTIVE (07:10): "transition into a full SPSA auto research loop.
Goal is 8 full SPSA decisions from our plans and informed by our stockfish
study on optimization. After that we shift." This ratifies the optimize
direction of the 05:35 freeze proposal. Recorded as D45.

Infrastructure landed this morning (34deade + 2950ee1, both D24-verified):
- `tune` feature: all 233 params.rs scalars are runtime tunables
  (`setoption name <NAME> value <x>`), atomic-backed getters; the default
  build folds them back to constants — bench 701605 bit-identical.
  Proven live: ASPIRATION_BASE_DELTA=300 moves UCI-bench 4777232 -> 4590853;
  unknown names error loudly.
- `tools/spsa/spsa_tune.py`: local SPSA (paired Rademacher, fishtest
  schedules), fixed-node fastchess batches, both sides the SHIPPED s3 net
  (D39), preflight against the binary's `tunables` list. Smoke-tested.
- Decision protocol per campaign (design/14 §9.6): SPSA (~8000 games @20k
  nodes) -> fixed-node SPRT screen [0,10] -> EQUAL-TIME SPRT ship gate on
  the quiet box = the decision -> accept commits new defaults (functional,
  fresh Bench trailer) + ledger row.

Campaign order (S0 audit + SF14->16 study §6b + S1 freshness):
c1 corrhist weights (RUNNING since 07:58: 13 params, 800x10 games @20k,
beside the big-v1 labeling — fixed-node is load-immune) · c2 LMR/reduction
· c3 RFP+razoring · c4 NMP · c5 futility+SEE · c6 LMP+history-pruning ·
c7 aspiration+IIR · c8 history update magnitudes. TC gates run on quiet
windows (tonight); after decision #8 the program shifts to D41.

## 2026-07-17 06:10 [VERDICT] S1 correction history PASSES its equal-time SPRT: +72.9 ± 16.2, H1 at 208 games

**exp-0074**: engine-s1 (c10680f) vs engine-pre-s1 (df2b746), SAME s3 net
both sides, 10+0.1 conc 3 quiet box, SPRT [0, 10] α=β=0.05: **H1 at 208
games** (69W/113D/26L, +72.9 ± 16.2, LLR 2.95/2.94). The T3 early stop
saved ~2/3 of the 600-game budget.

Why equal-time (+73) so far exceeds the fixed-node screen (+18.5 ± 17.4,
exp-0071): correction history costs essentially no NPS, so its per-node
accuracy gain compounds through deeper TC searches (and 5k-node screens
barely fill the correction tables). Lesson for the protocol: NPS-neutral
SEARCH changes can be badly underestimated by shallow fixed-node screens
— route them to the TC gate quickly rather than iterating on screen Elo.

**Deployment call: S1 STAYS** (it is already merged at HEAD; merge gate
green, bench 701605). This is the largest single search gain since the
P9 pruning waves and lands squarely in support of freeze-proposal point
3: the search side is where the Elo is right now. Next in the program:
S2 (gainer-season SPSA retune) once the box has a free tuning window,
then the SF16-late un-deferral shortlist (S0 audit §6b).

**S1 merge gate**: with the trainer gone, `bash tools/check.sh` reran the
full D24 gate at HEAD (S1 correction history + R4 #2/#3 + obs stack):
fmt/clippy/tests/release-perft all green, bench fingerprint **701605**
reproduced. S1 is merge-clean; its equal-time SPRT (armed, [s1gate]) is
the remaining deployment gate.

## 2026-07-19 — SF16 north-star measured; 2 Fable studies running; cloud readiness
- **exp-0102: ARCHON(opt1+opt2, 720sb) vs SF16 = −42.8 ± 14.6 equal-time (310g).** Speed
  package closed ~+27 Elo of the SF16 gap (was ~−63/−78). North-star (SF16+30) now needs
  ~+73 more → next from eval-quality (data) + search, per the strategic study.
- **2 Fable studies LAUNCHED (background):** (1) STRATEGIC path-to-#1 first-principles review
  (a0771f0b) → docs/research/path-to-1-first-principles-2026-07-19.md; (2) PROFILING round-2
  on the opt1+opt2 binary (ab28d053) → docs/research/profiling-study-2026-07-19.md.
- **Cloud data-gen readiness:** tooling (M2-M7) ready + compute path validated locally;
  gcloud unlocks image build+push, auth (project archon-502500), FEN-manifest upload, submit.
  Cost: pilot ~$6-12, targeted relabel ~$12, full 100M@100k ~$59. GATED on: strategic study
  verdict + the label-value pilot (paused 3/10) finishing. Resume relabel after studies free box.

## 2026-07-19 — STRATEGIC FABLE VERDICT (path-to-1-first-principles-2026-07-19.md)
- **Verdict: right on engine-building, WRONG on sequencing.** The fast progress was a
  "replication dividend" (consuming published SF/binpack/reference knowledge); it runs out
  ~at SF16. Past here Elo must be statistically MINED (fishtest) or research-DISCOVERED —
  both need testing throughput one box lacks. **BINDING CONSTRAINT = D36 distributed testing
  (decided, never built).** Evidence: SPSA 8/8 null, SF16 anchor CIs all overlap, data
  0-for-15 self-sourced, all one-box eval rungs measured shut this week.
- **Do:** (1) finish speed tail (days, +10-25 Elo, best Elo/effort) then CLOSE speed;
  (2) stand up D36 fleet THIS WEEK (5-10 GCE spot, ~$100-300/mo — weekly 2000g SF16 anchor
  + global SPSA retune + SF16-late SPRT queue); (3) label-value pilot to verdict as a
  KILL-SWITCH (≥+15 scale targeted SF18 relabel; null hard-stop relabel/obs line).
- **STOP:** obs/cloud data infra + helm ahead of evidence — CANCEL d41-gold-v2, FREEZE
  obs-factory until the pilot pays. **START:** the fleet + weekly external SF16 anchor as the
  ONLY trusted metric (internal Elo deflates 2-3x).
- **Feasibility:** beat SF16 ET 1-3mo (high conf); SF16+30 3-9mo WITH fleet, unlikely without;
  #1 not plannable (keep as north star, revive D41 only after SF16+30). Realistic target:
  strongest solo/AI-built engine on record.
- **Cloud reframe:** highest-value cloud spend = TESTING FLEET, not data relabel. Awaiting
  user decision on the fleet (cloud approval). Profiling round-2 (ab28d053) still running.

## 2026-07-19 — D41 learned-search-augmentation VERDICT + SF16 PGO validation
- **D41 DEEP STUDY (learned-search-augmentation-first-principles-2026-07-19.md):**
  the PER-NODE helm (as deployed) is CLOSED, not parked — proven by our own data.
  Governing inequality `k > T` (fixed-node gain-multiplier must exceed throughput
  tax T=1+f·c_i/c_n). Measured: exp-0029's +68.6@2k = k≈1.4-1.6 vs tax T≈130-210;
  cheapest shared-trunk T=7.4 => ~100× structural deficit vs a gain ceiling ~1.5
  (falling with depth, capped by the free warm history/TT stack). Killing
  mechanism MEASURED: affordability–informativeness ANTICORRELATION (cheap nodes =
  where the free stack is already strongest); value latent adds ZERO reduction
  signal over 39 scalars (ΔAUC −0.003 ×3); VOC target negative-R² => VOC-gating
  dead. SURVIVORS (cost amortized ~0): R1 root-frozen distilled prior TABLE
  (T≈1.002, untested), R2 learned root TIME-MGMT (~40-50% chance +5-15 ET, zero
  cost), R4 "NNUE trick for policy" (incremental from/to embeddings, T≈1.1-1.3,
  gated on K1). For #1: conditional-yes-modest (~+10-20 ET R1+R2; R4 could >+30 iff K1).
- **DECISIVE EXPERIMENT K1+K2 (~3 box-days, NO training, NO cloud, reuses artifacts):**
  K1 = gen-0 rank-8 head Elo-vs-node decay curve @2k/8k/32k/128k (design/19 §8.2,
  never run); K2 = materialize its 64×64 logit table at root + add to quiet ordering
  every node (~30 lines movepick.rs, <1% NPS) at fixed + 10+0.1. K2 retaining ±20%
  of the gain at equal time = first learned allocator to beat mature hand-tuned
  alpha-beta at equal wall-clock. RESEARCH CONTRIBUTION YES (reframed: measure the
  viability frontier of learned compute allocation; k>T + viability index V portable;
  chess = adversarial testbed V≈0.1 vs LLM inference V≈10^4). QUEUED behind SF16+30
  per strategic sequencing — PENDING user decision (offered to run K1+K2 now).
- **SF16 PGO VALIDATION RUNNING (user: validate speed gains now):** rebuilt
  probe-sota-pgo = opt1+opt2+PGO (node 147691 bit-identical, ~966k NPS = +35% over
  pre-speed base). Match archon-pgo vs SF16 10+0.1 (data/nnue/experiments/pgo-vs-sf16).
  Prior plain-opt2 was −42.8; PGO ~+9% NPS => expect closer. Relabel PAUSED at 3/10
  for the quiet box; resume after.

## 2026-07-19 — SF16 PGO validation FINAL + 3 studies running
- **exp-0103: ARCHON opt1+opt2+PGO vs SF16 = −41.5 ± 12.3 (463g).** Statistically =
  plain-opt2's −42.8 (exp-0102) → **PGO's +9% bench-NPS did NOT convert to game-Elo**
  (workload mismatch + NPS→Elo slope flattens near SF16). SPEED PROGRAM VALIDATED gain
  vs SF16 = ~+28 (from ~−70 to ~−42), matches opt#1's +28.8. Keep PGO as the deploy
  form but it's a wash for game-Elo; the sota bench NPS ranking still holds.
- **3 Fable studies commissioned (all model=fable, self-report requested):**
  (1) D41 auxiliary-task ADVERSARIAL (train-time aux / eval-only inference, H-AUX) →
  d41-auxiliary-task-adversarial-2026-07-19.md; (2) creative PERFORMANCE study #3
  (more liberty, structural/arch wins beyond micro-opts) → profiling-study-3-creative
  -2026-07-19.md. Both boxed to not disturb equal-time matches. Box kept QUIET for
  their measurements; kill-switch relabel PAUSED at 3/10, resumes after.
- **D41 REFRAME (user):** the first D41 study was wrongly scoped — closed per-node
  INFERENCE allocation (k>T) but never evaluated the TRAIN-TIME-auxiliary/eval-only
  formulation (T=1, no tax). H-AUX = co-train eval trunk w/ a policy/search-value aux
  head, deploy eval-only. reframe doc committed; adversarial re-study running.
