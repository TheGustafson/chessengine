# Day plan — strongest model (user away all day, 2026-07-21)

**User executive decisions (verbatim intent):** get the model as strong as possible
using ALL the toys — every extra head + all the extra data. Architecture: **DROP
FullThreats** and make the **accumulator 2× bigger (768 → 1536)**. Start there, iterate
as I see fit. Full autonomy for the day.

**The bet (why it's coherent):** FullThreats is the dominant NPS tax (~160 active
feats/move, memory-bound i16 adds; measured ~1.5× slower vs a lean HalfKA-only engine
like Reckless). Dropping it FREES that NPS to fund a 2× accumulator on a PURE HalfKA net
(≤32 active feats, cheap updates). Net effect: more capacity, NPS-affordable — the exact
"recover NPS to buy width" move. This is also the honest way to reopen the width
question (see [[no-verdicts-from-weak-tests]]: prior width test was a 120sb equal-time
screen, inconclusive).

## Standing discipline (do not regress)
- NO verdicts from weak/underpowered/confounded/dirty-data screens (memory lesson).
  Fixed-node gate isolates eval; equal-time is the ship bar. Tight CI before any "dead."
- Ledger every net/corpus/gate (D38). Keep ONE of any waiter (comm==bash count, kill by
  explicit PID — pgrep -f self-matches this shell).
- Measure λ/scale/recipe before launching a train; don't guess a knob into a run.

## Corrected prior conclusions (all REOPENED, none dead)
- fold-in exp-0118: INVALID (aux poisoned by unfiltered build_hardneg_corpus.sh). Clean fold-in untested.
- width exp-0120: INCONCLUSIVE (120sb equal-time screen). Re-testing fixed-node + the new 1536 arch is the real test.
- policy exp-0121: INCONCLUSIVE (one bad config). Redo: from+to, correct _hm orient, λ sweep, strong base.

## LIVE STATE — 2026-07-21 PM-3 (TIGHTENING SPRINT — supersedes the PM block below)

**USER DIRECTIVE (midday): killed the 1536 GPU run — "focus on fixing training
full time; if we can't get this right there is no use wasting GPU."** The PM
block below is retained as history; the 1536 bet is PARKED (trainer consts
reverted to 768/L2=16; resuming = flip HL/L2 + rebuild; engine/scripts intact).

**Everything now flows through the tight recipe** (full detail:
`value-foundation-state.md` PM-3): net_scale_check.sh authority (validated on 6
nets, hard-wired into both gates), WDL-coupled SCORE_MULT (0.3→1.42, 0→1.94,
derived from measured K≈2.7), `obs-tool export-value --filter-sfbinpack` (full
trainer-parity filter, unit-tested), build_canon_tight.sh (self-checking corpus
build), hardneg-tight-v2 (ledger'd; 86k clamp-hidden mate labels removed),
fast_fens keys-mode (full-coverage WDL join, smoke-verified, queued).

**IN FLIGHT:** tight corpus (78.4M filtered → ~27.5M WDL-joined) → ab_canon_tight.sh
auto-chain: Cell A (WDL=0.3/1.42) vs Cell B (WDL=0/1.94), 60sb, scale-check
hard-gated, 200g fixed-node gates vs d40-uho-v1-240sb. ~2.5h to verdict.
Readout: A≈repro-class ⇒ recipe was everything; A≈−400 ⇒ volume ⇒ fold-in path;
B isolates WDL. Then: ledger rows + full-coverage extraction if A validates.

## LIVE STATE — 2026-07-21 PM (historical; 1536 run KILLED mid-stage-1)

**Arch BUILT + SMOKE-VALIDATED end-to-end** (engine `archon-sota-1536`: L1=1536,
L2=32, SIMD `l1_dot` generalized to L2/16 blocks, parity-tested; trainer HL=1536
L2=32 + `halfka_only` quant-save fix; netpack env-driven + threats-less). git HEAD
`8412fe9` (arch changes uncommitted; bullet trainer is gitignored).

**RUNNING NOW:** full train `h1536-uho-300sb` STAGE 1 (halfka-only 1536, UHO,
div1.9 skip75 WDL0.3 300sb LR1e-3), ~sb15/300, ~477k pos/sec, ~4.5h for stage1.
- Driver: `tools/nnue/autolaunch_1536.sh` (pid 1043362) → after stage1 runs STAGE 2
  `h1536-s2-farseer-140sb` (warm from stage1 raw, div1.9 WDL0.25 140sb LR5e-4) →
  then AUTO-GATES `data/nnue/nets/h1536-s2-farseer-140sb.cenn` via
  `tools/nnue/gate_1536.sh` vs incumbent `d40-uho-720sb` (fixed-node + equal-time)
  + SF16. Full pipeline ~7h to the arch verdict.
- Logs: `/tmp/train_1536.log` (train+gate), `/tmp/gate_1536.log` (gate detail).
- Monitor `b2yfex58d` (tail pid 962406) watches train_1536.log for DONE/FAILED/GATE.
- Scripts: `train_1536_halfka.sh` (SMOKE=1 for 3+3sb; S1_SB/S2_SB env-overridable),
  `gate_1536.sh` (ROUNDS env), `autolaunch_1536.sh`.

**RESULTS SO FAR:**
- **NPS (SMOKE, pessimistic):** NEW-1536 720,847 vs incumbent 768+threats 835,087 =
  **−13.7%**. The drop-threats bet did NOT buy net speed (2× accumulator + L2 4×
  cost > threats saved). Converged net will be faster (~−10%). ARCH MUST WIN ON
  EVAL QUALITY at the equal-time gate. Fallback NPS levers if it loses on speed:
  **L2=16 ablation** (halves l1_dot) and/or **narrower 1280 accumulator**.
- **canon −582 RE-DIAGNOSED (2026-07-21, independent Fable forensics ⇒
  `docs/research/sf18-canon-data-diagnosis-2026-07-21.md`): NOT a data bug.**
  All 9 data-integrity probes clean (labels are faithful SF18@50k white-relative
  UCI cp — rerun reproduces them to p50 10cp/99.6% sign; encoding/POV/shuffle/
  dedup/distribution sound). The collapse was a **4-way confounded recipe**, each
  fix serially mis-ruled-out: (1) **target scale ×0.51** — "canon cp==binpack/1.9"
  is FALSE (measured ~2.7×); SCORE_MULT must be ~1.94 not 1.0 [both canon runs used
  1.0]; (2) **quiet filter only removed in-check** — ~22% best-move-capture poison
  remained, so **exp-0123's "filter ruled out" is INVALID**; (3) **WDL=0** (0.5
  placeholder byte) — open confound, never isolated on clean data; (4) 60-epoch
  replay of 100M uniques. **Quarantine LIFTED, pending a confirming A/B** (below).
  Scripts corrected (`train_canon.sh`/`build_canon_corpus.sh`: false scale comment
  + SCORE_MULT default 1.0→1.94). The live 1536 run uses binpacks and is UNAFFECTED.
  - **QUEUED A/B (GPU, ~1.5h + 2 gates), from the diagnosis:** rebuild canon with
    the FULL 5-criterion filter (extend `obs-tool export-value` to use the shard
    `best_move`) + real game-WDL join; **Cell A** SCORE_MULT=1.94 WDL=0.3, **Cell B**
    same but WDL=0; gate both vs `d40-uho-v1-240sb`. A≈−150 ⇒ data was always fine;
    A≈−400 ⇒ residual is volume ⇒ use canon as `D40_MIX_DATA` fold-in on the streamed
    base; B isolates the WDL=0 killer. Not built yet (needs the export-tool change).
- **width reopened (exp-0122):** fixed-node 1024 vs 768 = +6±24 (not negative).

**QUEUED (blocked on the base net / GPU):** heads co-trained (WDL built; policy
NEEDS from+to redo — cycle-bug risk, smoke-test before committing); L2 ablation
(16 vs 32 vs 32→32→16 extra layer, fixed 1536, fixed-node); clean fold-in test
(data READY = `data/nnue/hardneg/corpus/corpus.filtered.shuf.bin`, ledger
`hardneg-clean-shuf-v1`, 1.75M pos, 9.17% poison removed).

**Ledger rows added:** exp-0122 (width), exp-0123 (canon), data hardneg-clean-shuf-v1.

## Execution order (the day)
1. **Build the new arch:** trainer HL=1536 + D40_HALFKA_ONLY=1; engine sota L1=1536 +
   threats-less eval path; netpack D40_ACC=1536 + no-threats. VERIFY: trains, evals,
   loader round-trips, and MEASURE NPS (must be ≥ the 768+threats net — the whole point).
2. **Strong base train:** 1536 HalfKA-only on the best data. Proven recipe = UHO stage-1
   → farseer stage-2 (both local, filtered loader). Add clean canon if canon-clean wins.
   Train to real depth (NOT 120sb — width needs data to fill; several hundred sb).
3. **Heads (all the toys), co-trained, deploy-eval-only:** WDL (built) + policy REDONE
   properly (from+to, correct orientation, λ measured/swept). Gate each vs the value-only
   base at FIXED NODE then equal time.
4. **Data (all the toys):** UHO + farseer + clean-canon (if good) + RE-FILTERED hard-negs
   (rebuild the aux THROUGH canon_filter first — the old aux was poisoned).
5. **Gate the ship candidate** vs the incumbent (d40-uho-720sb) then SF16 at 10+0.1.
   Iterate width/heads/data/λ from the numbers.

## Key facts / paths
- Trainer: `.archon-state/toolchains/bullet/examples/d40_features.rs` (const HL; D40_HALFKA_ONLY, D40_WDL_LAMBDA, D40_POL_LAMBDA env). Rebuild: `cd .archon-state/toolchains/bullet && CUDA_PATH=/opt/cuda cargo build --release --features cuda --example d40_features`.
- Engine sota width: `src/eval/nnue/sota_head.rs` const L1; ACC assert in `sota_features.rs`. Build: `RUSTFLAGS="-C target-cpu=native" cargo build --release --features sota` → cp to `engines/bin/archon-sota-<HL>`.
- netpack: `tools/nnue/netpack.py` (D40_ACC env override for width).
- canon filter tool: `src/bin/canon_filter.rs` (drops in-check poison + |cp|>10000).
- Data: UHO `data/nnue/binpacks/nodes5000pv2_UHO.binpack`, farseerT76 `data/nnue/binpacks/farseerT76.binpack`, clean canon `data/nnue/canon-v1/build/corpus.quiet.shuf.bin`, hard-negs `data/nnue/hardneg/` (NEEDS re-filter).
- Incumbent `data/nnue/nets/d40-uho-720sb.cenn`; 768 engine `engines/bin/archon-sota-native`; SF16 `engines/bin/stockfish16`.
- Log to `value-foundation-state.md` + ledger after each result. Update THIS doc's "in flight".
