# ARCHON originality & license audit — SEARCH DOMAIN

Date: 2026-08-09. Auditor: search-domain agent (one of three parallel domain
audits). Scope: `src/search/*`, `src/movepick.rs`, `src/history.rs`,
`src/tt.rs`, `src/timeman.rs`, `src/threads.rs`, `src/search_control.rs`,
`src/search_intervention.rs`, `src/search_trace.rs`, `src/tune.rs`,
`src/cuckoo.rs`, the search design docs, and the tracked distilled notes in
`stockfish_reference/` and `reckless_reference/`.

References compared: **Stockfish at `eca43a97`** (GPLv3; the transcription
source — all value/code comparisons below were made against
`git show eca43a97:src/<file>`, not the sf_18 checkout), **Reckless
v0.10.0-dev** (AGPLv3, vendored checkout), **Integral** (no license = all
rights reserved; clone ABSENT from the repo, influence audited via design/
decisions and src grep — `integral_reference/` does not exist, nothing in
`old/` either).

Rubric: L0 independent · L1 idea-level adoption, independently expressed ·
L2 structural echo (same decomposition/order/names/call shapes) · L3
near-translation (line-by-line correspondence, transliterated expression) ·
L4 verbatim.

**Headline.** No L4 anywhere. No L3 *as textual transliteration* — every
audited function is expressed in idiomatic Rust with named parameters,
original doc comments, and its own decomposition. But the project
deliberately built an SF-parity search (D20/D27, the S0 program), so the
dominant rating is **L2 with line-by-line *logical* correspondence** across
the whole pruning/reduction/history apparatus: same step order, same
formulas term-for-term including integer-truncation order, same ~259 tuned
constants, and in several small functions the same variable names modulo
case convention. Two blocks sit at the L2/L3 boundary and are named below
with side-by-side excerpts. Reckless expression-level similarity — the
highest a-priori risk, Rust-to-Rust — was checked structure-by-structure
and is **absent**; Reckless contributed ideas only (D30/D31), each with a
clearly different ARCHON expression. Integral contributed **nothing** to the
search domain (D37 records "few novel search techniques"; zero `integral`
citations in `src/`).

---

## 1. Verdict table

### 1.1 Modules

| Module | Rubric | Originality tag | Basis |
|---|---|---|---|
| `src/search/mod.rs` (6376 ln) | **L2** (logic corresponds step-for-step to SF `search()`/`qsearch()`; expression independent; heavy ARCHON additions) | SF-SPECIFIC core + ARCHON-ORIGINAL instrumentation/divergences | §2.1, §3 |
| `src/search/params.rs` (1529 ln, 259 tunables) | **L2** — the deliberate constant transcription | SF-SPECIFIC (facts) + documented ARCHON deviations | §2.2 |
| `src/search/stack.rs` | L2 | SF-SPECIFIC layout (`ss-7..ss+2`, MAX_PLY+10) with documented deviations (no `pv` ptr, no killers, index-not-pointer) | stack.rs:1–100 vs SF search.h:104–120 |
| `src/search/rootmoves.rs` | L2 | SF-SPECIFIC (`RootMove` fields incl. `effort`, `averageScore`, `meanSquaredScore` EMAs; ARCHON i64 deviation documented rootmoves.rs:18–24) | vs SF search.h:130–156 |
| `src/movepick.rs` | **L2, two L2/L3-boundary blocks** (§3.1, §3.2) | SF-SPECIFIC staging + ARCHON-ORIGINAL ordering terms (§4.1) | §3 |
| `src/history.rs` | L2 | SF-SPECIFIC (gravity update, exact D bounds 7183/10692/30000/8192/1024, exact init fills, `update_all_stats` mirror) ; const-generic `StatEntry<D>` is an idiomatic re-expression of SF's `StatsEntry<T,D>` template | history.rs:59–118, 679–786 vs SF history.h + search.cpp:1898–1956 |
| `src/tt.rs` | **L2, `save()` at the L2/L3 boundary** (§3.3) | SF-SPECIFIC (10-byte entry, 3×32B cluster, 5-bit gen, `mul_hi64`, penalize, secondary aging, hashfull-1000-sample) | §3.3 |
| `src/timeman.rs` | L2 | SF-SPECIFIC (every fitted constant and formula from `timeman.cpp:46–135`, transcribed and documented as D20; independent structure + original test suite + ARCHON dose-warning analysis) | timeman.rs:26–113, 285–390 |
| `src/threads.rs` | L1–L2 | SF-STANDARD lazy SMP; SF-SPECIFIC main/helper + voting protocol (L2); safe-Rust `Mutex<ThreadState>+Condvar` shell and `Job` enum are original expression; `Job::Custom` closure dispatch credited to Reckless (threads.rs:110) | vs SF thread.cpp:289–457 |
| `src/search_control.rs` | **L0** | **ARCHON-ORIGINAL** (D41 versioned shadow-inference contract; no analogue in SF/Reckless) | whole file |
| `src/search_intervention.rs` | **L0** | **ARCHON-ORIGINAL** (single-action LMR counterfactual replay oracle) | whole file |
| `src/search_trace.rs` | **L0** | **ARCHON-ORIGINAL** (LMR observation schema v3, latent capture) | whole file |
| `src/tune.rs` | L1 | RECKLESS-influenced concept (`define!` const/SPSA macro, `parameters.rs:1–33`); ARCHON expression is materially different (atomics + `Result` errors + f64-bits + D50 `constants!` twin vs Reckless's `static mut` + panic) | §3.4 |
| `src/cuckoo.rs` | L1–L2 | Published algorithm (van Kervinck); SF-SPECIFIC parameters (8192 slots, `h1 = k & 0x1fff` / `h2 = (k>>16) & 0x1fff`, the 3668-entry count used as an explicit oracle, cuckoo.rs:42) ; build loop and two-parallel-array layout own expression | vs SF position.cpp:106–160 |

### 1.2 Mechanisms

Every distinct search mechanism, rated. "SF-STANDARD" = present in every top
engine; "SF-SPECIFIC" = SF's particular formula/constants/quirk, knowingly
transcribed under D20/D27.

| Mechanism | Where (ours) | Rubric | Originality |
|---|---|---|---|
| Upcoming-repetition early draw | mod.rs:2602–2619 | L2 | SF-SPECIFIC gating (`alpha < DRAW`, raise-alpha shape, pre-qsearch placement); mechanism published |
| Mate-distance pruning | mod.rs:2714–2720 | L1 | SF-STANDARD |
| TT probe → cutoff, +child-TT verify, +wrong-bound `penalize(1)` | mod.rs:3012–3142 | L2 | SF-SPECIFIC (child-verify and penalize are current-SF quirks); ARCHON divergence: cutoff block deliberately also runs in check (documented mod.rs:3004–3011 — SF's does too; ARCHON runs *hindsight* in check where SF does not, mod.rs:2980–2989) |
| TT-as-eval-cache (`DEPTH_UNSEARCHED` write) | mod.rs:2959–2970 | L2 | SF-SPECIFIC |
| Correction history (pawn/minor/non-pawn/continuation; raw-eval-in-TT discipline) | history.rs:849–1009, mod.rs:4994–5013 | L2 | SF-SPECIFIC |
| Hindsight depth adjustment | mod.rs:2991–3002 | L2 | SF-SPECIFIC (constants 3/2/173 as params) |
| Eval-diff parent history nudge | mod.rs:3178–3198 | L2 | SF-SPECIFIC |
| Razoring | mod.rs:3217–3228 | L2 | SF-SPECIFIC (`465 + 300·d²` at eca43a97) |
| Reverse futility pruning | mod.rs:320–337, 3324–3497 | L2 | SF-SPECIFIC formula; ARCHON additions: removed threat-coupling (basket A), experimental DIS-RFP margin scale (default-off) |
| Null-move pruning + verification | mod.rs:3499–3592 | L2 | SF-SPECIFIC (cutNode-only gate, `R = 7 + d/3`, `nmp_min_ply` window); ARCHON A/B knob `ARCHON_NMP_ALLNODE` |
| IIR | mod.rs:3600–3611 | L2 | SF-SPECIFIC; ARCHON deviation: no followPV exemption (deliberate, measured — mod.rs:2664–2670) |
| ProbCut + in-check small ProbCut | mod.rs:3613–3711 | L2 | SF-SPECIFIC |
| LMP (movecount pruning) | mod.rs:409–412, 3954–3965 | L2 | SF-SPECIFIC (`(3+d²)/(2−improving)`) |
| Capture futility / capture-SEE pruning | mod.rs:445–489, 3989–4033 | L2 | SF-SPECIFIC incl. stalemate-trap guard |
| Continuation-history pruning + history-buys-lmr_depth | mod.rs:565–580, 4034–4084 | L2 | SF-SPECIFIC (LMR_DIVISOR 16-entry table verbatim values) |
| Parent futility + quiet-SEE pruning | mod.rs:414–443, 4086–4124 | L2 | SF-SPECIFIC |
| Singular extensions / multicut / negative extensions / shuffle guard | mod.rs:491–563, 4128–4220, 5199–5210 | L2 | SF-SPECIFIC (incl. double/triple margins, `is_shuffling`, ttMoveHistory dock) |
| LMR: log-table `reduction()`, finalization list, deeper/shallower, full-depth docks | mod.rs:5085–5119, 4293–4562 | L2 | SF-SPECIFIC term-for-term; ARCHON divergence: ttValue-rebate `is_valid` guard (measured, mod.rs:4294–4300); DIS-LMR hook (default-off) |
| Sibling depth-2 reduction on alpha raise | mod.rs:4769–4788 | L2 | SF-SPECIFIC; ARCHON deviation: root exempt (documented §3.9 note) |
| Fail-high softening blends (node exit + qsearch + RFP return) | mod.rs:4810–4821 etc. | L2 | SF-SPECIFIC (ARCHON folded the free weights to one D50 parameter each) |
| `update_all_stats` history burst (geometric quiet malus, refutation malus, breadth bonus) | history.rs:679–786 | L2 | SF-SPECIFIC |
| Continuation-history update (offsets/weights/consistency multipliers) | history.rs:564–622, params.rs:37–43 | L2 | SF-SPECIFIC (`{1040,780,300,537,129,423}`, `{96,113,...}` verbatim values) |
| TT store discipline (bound select, no-moves depth bonus, raw eval) | mod.rs:4944–4988 | L2 | SF-SPECIFIC; ARCHON extras: fail-low move store + upper-blind masks (default-off knobs, study 71) |
| Qsearch (stand pat, futility_base, 2-move late cutoff, SEE-vs-window, SEE floor, stalemate scan) | mod.rs:5293–5720 | L2 | SF-SPECIFIC; + Reckless-idea stop re-poll at entry (D31, L1); + ARCHON recapture discount (§4.1) |
| Aspiration windows | mod.rs:2404–2592 | L2 | SF-SPECIFIC; ARCHON deviation: `ASPIRATION_DELTA_MSS_DIV = 6776` frame correction (params.rs:83–100, derived not tuned) |
| ID loop, root hygiene (forgotten mate, aborted loss), search-again | mod.rs:1989–2402 | L2 | SF-SPECIFIC |
| Soft stop (fallingEval, stability reduction, instability, effort) | mod.rs:6130–6339 | L2 | SF-SPECIFIC; ARCHON deviation: `INSTABILITY_CHANGES_DECAY = 2.5` vs SF's 2 (depth-redenomination, mod.rs:2159–2164) |
| Hard stop / poll cadence | mod.rs:5973–6010, 6341–6358 | L1–L2 | SF-STANDARD check; node-bitmask cadence + `completed_depth==0` guard + qsearch re-poll = RECKLESS-SPECIFIC ideas (D31), own expression |
| TM allocation (opt/max scale, original_time_adjust, nodestime) | timeman.rs:285–427 | L2 | SF-SPECIFIC (all fitted constants) |
| Best-thread voting | mod.rs:749–818 | L2 | SF-SPECIFIC (`score − min + 14` vote, decisive protection, PV-length tiebreak) |
| Lazy SMP (Worker-owns-all, shared TT + 3 flags, main-only manager) | threads.rs | L1–L2 | SF-STANDARD architecture, SF-SPECIFIC protocol details, original safe-Rust expression |
| Cuckoo tables | cuckoo.rs | L1–L2 | published algorithm; SF-SPECIFIC parameters |
| Draw contempt (`value_draw_contempt`, default 0) | mod.rs:97–118 | **L0** | **ARCHON-ORIGINAL** (derived from loss-P2 forensics: perpetual-sets-alpha starvation) |
| Quiet SEE ordering penalty, clamped above GOOD_QUIET_THRESHOLD | movepick.rs:1040–1057 | **L0** | **ARCHON-ORIGINAL** (+11.00 ± 7.20 Elo over 3350 games, study 71; the clamp — the load-bearing half — is ARCHON's finding) |
| Quiescence recapture discount (`QS_CAPTURE_DEFENDED_NUM`, `recapture_map`) | movepick.rs:227–294 | **L0** | **ARCHON-ORIGINAL** (study 70; SF has no defended-victim term in capture ordering) |
| Evasion capture-history term | movepick.rs (`init_evasions`) | L0/L1 | ARCHON fix (study 70 census; SF scores evasion captures without capture history) |
| DIS heads (DIS-EVAL/RFP/LMR/TM + census + dump) | mod.rs:152–318 and sites | **L0** | **ARCHON-ORIGINAL** (disagreement-head search consumers; all default-off) |
| Observation store capture (design/16 §3.5.5) | mod.rs `obs_*` sites | **L0** | **ARCHON-ORIGINAL** |
| SearchStats / Gate census / census70 / foldcount instrumentation | mod.rs:894–1180 etc. | **L0** | **ARCHON-ORIGINAL** |

### 1.3 Removed mechanisms (evidence of original capacity; revival material)

Confirmed via `git show 52996883` ("basket A: remove the ARCHON-original null
trio", 2026-08-01) and the tier-0 commits (5c161b59…7eb1f478):

- **D30 offense term** (reverse-attack fans onto undefended enemies in quiet
  ordering) — labeled "ARCHON-original" in the commit, but the *idea* is
  Reckless's (`reckless_reference/05-movepick-history.md:162–171` documents
  Reckless's `offense[pt]` term explicitly; D30 credits the study). The
  *construction* (vulnerable-set cumulative unions + setwise reverse fans)
  was ARCHON's. Removed as null (+0.07 pp fhf inside census noise).
- **RFP threat coupling** (`no_own_threat` margin relaxation + the two-margin
  lazy-`all_threats` machinery) — genuinely ARCHON-original; removed at
  −3.5 ± 14.1 (neutral).
- **Step-12 equal-value dither + `child_pv_fresh` truncation** — ARCHON's
  own variant of near-leaf tie diversity (SF has a different equal-value
  dither; ARCHON's fired 0.94/1k nodes, never validated); removed.
- **followPV tracking** (`FOLLOW_PV_MODE`) — ARCHON's re-shape of SF's
  followPV/IIR-exemption; never won a measurement; deleted (mod.rs:2664–2670
  records the divergence that remains: ARCHON ships *without* SF's followPV).
- **Speculative TM/uncertainty head hooks** (`score_tm_factor`, EXP-UNC-1,
  WDL-TM) — ARCHON-original, refuted at real TC (−36.6 ± 22.5), deleted.
- **D41/D47 policy-ordering plumbing** (`.cpol` sidecar, `set_quiet_prior`
  fill) — ARCHON-original; the seam (`MovePicker::set_quiet_prior`) remains.

---

## 2. License-risk findings, most severe first

### 2.1 F1 — The search as a whole: wholesale replication of SF's selection & arrangement (judgment call, MEDIUM)

This is the audit's most serious finding and it is a legal judgment call,
not a code-copying finding. The facts:

- The node functions follow SF's step structure exactly (our `search()`
  steps map 1:1 onto SF search.cpp:744–1586's Steps 1–21; our `qsearch()`
  onto search.cpp:1596–1828's Steps 1–9), by design (design/07/10 are
  transcription contracts; doc comments cite "SF parity" throughout).
- All 259 tunables in params.rs are SF's exact values at eca43a97
  (params.rs:1–6 declares it; `tools/spsa/param_provenance.py` screens it;
  CLAUDE.md records a full verification 2026-07-27).
- Formula expressions preserve SF's term order and integer truncation
  ("the product is formed before the divide" is a recurring, deliberate
  doc-comment pattern, e.g. mod.rs:146–149, 455–460, 482–484).

Individually, each constant and each formula is an unprotectable fact/idea
(D27's stated theory, and the mainstream view). The residual risk is that a
court could view the *curated set* — which heuristics, in which order, with
which couplings, with exactly these 259 values — as a protectable selection
and arrangement, making the search a derivative work of a GPLv3 program.
Points in each direction, honestly weighed:

- *For risk:* the selection was not independently derived; it was read out
  of SF's source and reproduced wholesale, including quirks with no
  independent engineering rationale (e.g. the child-TT verification, the
  `penalize(1)` wrong-bound aging, the exact LMR finalization term list).
  The scenes-à-faire defense ("every top engine looks like this") is weaker
  for the *exact current-SF configuration* than for the generic PVS toolkit.
- *Against risk:* constants are functional, dictated-by-efficiency values
  (merger/scènes-à-faire doctrines); the arrangement is substantially the
  industry-standard alpha-beta architecture; the expression is demonstrably
  independent (different language, different decomposition into ~30 pure
  helper functions, original commentary exceeding the code in volume,
  documented deviations with measurements); and the project has begun
  diverging (deviations list in §1.2, ARCHON originals in §4.1).

**Recommendation.** (a) If ARCHON is ever distributed, license it GPLv3 —
the norm for SF-family engines (every SF derivative and most independent
top engines are GPL); this converts F1 from a compliance question into a
non-issue and costs nothing given the stated goals (rating lists accept GPL
engines). (b) Keep D27's provenance discipline: the doc-comment citations
are an asset, not a liability — they demonstrate idea-level process. (c) Do
not present ARCHON's search as independently derived in publications;
present it as "SF-parity baseline + named original mechanisms", which is
also the honest research framing.

### 2.2 F2 — The 260-constant transcription specifically (judgment call, MEDIUM-LOW, subsumed by F1)

The prompt asks for this separately. `src/search/params.rs` (259 getters) +
`timeman.rs` constants + `history.rs` D-bounds/init-fills are a curated,
SF-tuned value set adopted verbatim (D20). Analysis: individual tuned
values are facts (uncopyrightable under Feist; SPSA output is not
authorship in any conventional sense), and a value *set* has thin
protection at best — but it is the same "selection" question as F1 and
should be treated as one finding, not two. Mitigating facts specific to the
constants: several have already been *re-derived away* from SF for ARCHON's
own frame — `ASPIRATION_DELTA_MSS_DIV` 10588 → 6776 (a derived frame
correction with a worked derivation in params.rs:85–100),
`INSTABILITY_CHANGES_DECAY` 2 → 2.5, plus ARCHON-only constants
(`QUIET_SEE_PENALTY`, `QS_CAPTURE_DEFENDED_NUM`, `DRAW_CONTEMPT`,
`VOTE_TEMPO` kept but re-documented). An SPSA season (D45) that moves the
set materially would further attenuate this. **Not something to wave
through, not something to catastrophize: fold into the F1 decision.**

### 2.3 F3 — `partial_insertion_sort`: the closest thing to L3 in the domain (LOW severity, HIGH confidence)

Same algorithm, same names (`sorted_end`/`p`/`q`/`tmp`/`limit`), same
hole-punch trick, ~15 lines. Side-by-side:

SF `movepick.cpp:111,134–143` (eca43a97, scalar path):

```cpp
for (; p < end; ++p)
    if (p->value >= limit)
    {
        ExtMove tmp = *p, *q;
        *p          = *++sortedEnd;
        for (q = sortedEnd; q != begin && *(q - 1) < tmp; --q)
            *q = *(q - 1);
        *q = tmp;
    }
```

ARCHON `src/movepick.rs:114–167` (excerpted; bounds-proof scaffolding and
comments elided):

```rust
for p in 1..moves.len() {
    let tmp = moves[p];
    if tmp.value < limit { continue; }
    sorted_end += 1;
    let se = sorted_end.min(p);
    let head = &mut moves[..=p];
    head[p] = head[se];
    let mut q = se;
    while q > 0 && head[q - 1].value < tmp.value {
        head[q] = head[q - 1];
        q -= 1;
    }
    head[q] = tmp;
}
```

Assessment: line-by-line logical correspondence with matching identifiers —
this is a *derivation*, not a coincidence. Mitigations: it is a ~15-line
textbook micro-algorithm (partial insertion sort with a limit); the ARCHON
version carries substantial original expression around it (invariant
asserts, bounds-check-discharge engineering from perf studies 20/23/55/56,
a documented tail-order contract, its own test battery movepick.rs:1349+);
and function-scale, standard-algorithm similarity is at the bottom of what
GPL enforcement has ever pursued. **Action if desired:** rename the locals
and restructure the loop (e.g. drain-iterator style) — cheap insurance, but
node-identity (`BadQuiet` emits the tail in buffer order) constrains any
real algorithm change (movepick.rs:158–160).

### 2.4 F4 — `TtEntry::save` and the movepick stage machine: L2/L3-boundary structural echoes (LOW)

- `src/tt.rs:250–319` vs SF `tt.cpp:92–125`: the same three rules in the
  same order with the same condition composition (`b == EXACT || key differs
  || new_d + 2·pv > old_d − 4 || relative_age != 0`, then the same
  secondary-aging else-arm). Ours adds named constants, a race-guard on
  `v16`, and rule-by-rule commentary; the logic is a mirror by design
  (doc 08 §3.7 "mirrors SF's save").
- `src/movepick.rs:303–324` vs SF `movepick.cpp:33–57`: identical 16-stage
  enum, same names modulo case (`MainTt/CaptureInit/GoodCapture/QuietInit/
  GoodQuiet/BadCapture/BadQuiet/EvasionTt/.../Qcapture`); cursor fields
  `cur/end_cur/end_bad_captures/end_captures/end_generated` are SF's pointer
  names; `select(pred)` reproduces SF's predicate-closure API
  (movepick.rs:1300 vs movepick.cpp:265–272). The state *machine* expression
  differs (`bump()` + `Stage::from_u8` vs `++stage` + `goto top`
  fallthrough; index cursors vs pointers; a `sorted_covers`/`bad_quiet_start`
  optimization SF does not have).

Same analysis as F3: deliberate, documented, design-level replication with
independent expression. Bundled into the F1 decision rather than actionable
alone.

### 2.5 F5 — Quoted AGPL fragments inside tracked Reckless notes (LOW)

The tracked notes are distilled prose, but `reckless_reference/
05-movepick-history.md` (4 rust fences) and `07-search-node-pruning.md`
(1 rust fence, the 3-line `lerp` helper) contain short quoted code
fragments of an AGPLv3 work, ≤10 lines each, embedded in commentary. Checks
against the vendored v0.10.0-dev checkout: at least two of the quotes do
**not** match the current source verbatim (the quoted 5-variant `Stage` enum
vs the checkout's 6 variants at `Reckless/src/movepick.rs:9–16`; `fn lerp`
absent from the checkout's `search.rs`) — i.e., they are transcriptions
against an earlier revision or paraphrases, and `NOTES.md:14–28` carries an
explicit provenance/licensing warning stating the notes' method. The SF
notes (`stockfish_reference/*.md`) contain **zero** code fences of C++ —
formulas appear as prose/arithmetic. **Assessment:** quotation-scale use for
study commentary; distribution risk of the repo is minimal but non-zero
since the fragments are AGPL text in a tracked file. **Action if desired:**
convert the five rust-fenced fragments in notes 05/07 to pseudocode (10
minutes' work) — then the notes contain no licensed text at all.

### 2.6 F6 — Reckless expression-level similarity: checked and ABSENT (null finding, reported as required)

Rust-to-Rust was the stated highest risk. Structure-by-structure comparison:

| Structure | Reckless | ARCHON | Verdict |
|---|---|---|---|
| MovePicker | 6-variant Stage, `ArrayVec` + `list.remove(best_idx)` linear scan, `bad_noisy` side-vec (`movepick.rs:9–135`) | 16-variant SF staging, single buffer + index cursors + partial insertion sort | different design entirely (ours is SF's, not Reckless's) |
| TT | 8-byte entries, cluster SWAR tag word, `unsafe impl Sync` (`transposition.rs`) | 10-byte SF entries, per-field relaxed atomics, no unsafe | different |
| Search stack | self-referential boxed Stack | flat array + `SS_OFFSET` indexing | different |
| Histories | `HugeBox` mmap, zero-init, no per-go decay | boxed arrays, SF init fills + per-go decay, const-generic gravity | different |
| Tunables macro | `define!` — `static mut` + panic on unknown (`parameters.rs:1–33`) | `tunables!` — atomics, `Result`, f64-to-bits, D50 `constants!` twin (`tune.rs`) | same concept (L1), clearly different expression |
| Node loop | TT-narrowed windows, lerp blends, dynamic ProbCut re-verify, ilog2 LMR | SF's shapes throughout | ARCHON did not adopt Reckless's search shapes |

What ARCHON *did* take from Reckless is idea-level and attributed: D30
threat map (eval/board-side; consumed here as ordering features), D31
robustness bundle (qsearch entry stop re-poll mod.rs:5286–5292; node-mask
poll cadence mod.rs:6341–6358; `completed_depth == 0` hard-stop guard),
`Job::Custom` closure dispatch (threads.rs:110), the tunables-macro concept.
All L1. No AGPL-relevant similarity found in `src/`.

### 2.7 F7 — Integral: NIL (null finding)

`integral_reference/` does not exist in the tree (nor under `old/`); zero
`integral` matches in `src/`; D37 (decisions.md:398–420) records that the
study surfaced few novel search techniques and its adopted value was
process-level (bench trailer discipline, tooling, self-bootstrap
validation). No search mechanism traces to Integral. The all-rights-reserved
concern is moot for this domain.

---

## 3. Method notes on the closest comparisons (receipts)

Representative side-by-side beyond §2.3 — RFP, the highest-traffic formula
block. SF `search.cpp:974–988`:

```cpp
if (!ss->ttPv && depth < 17 && eval >= beta && (!ttData.move || ttCapture)
    && !is_loss(beta) && !is_win(eval))
{
    Value futilityMult = std::min(40 + depth * 4, 80);
    futilityMult -= 20 * !ss->ttHit;
    Value futilityMargin = futilityMult * depth
                         - (2934 * improving + 343 * opponentWorsening) * futilityMult / 1024
                         + std::abs(correctionValue) / 182069;
    if (eval - futilityMargin >= beta)
        return (716 * beta + 308 * eval) / 1024;
}
```

ARCHON `mod.rs:320–337` (margin) + `mod.rs:3324–3496` (gate; experiment
scaffolding elided):

```rust
let futility_mult = (params::RFP_MULT_BASE() + params::RFP_MULT_SLOPE() * depth)
    .min(params::RFP_MULT_CAP())
    - params::RFP_MULT_NO_TTHIT() * Value::from(!tt_hit);
futility_mult * depth
    - (params::RFP_IMPROVING() * Value::from(improving)
        + params::RFP_OPP_WORSENING() * Value::from(opponent_worsening))
        * futility_mult / params::RFP_TREND_DEN()
    + correction_value.abs() / params::RFP_CORR_DIV()
// gate: !tt_pv && depth < RFP_MAX_DEPTH && eval >= beta
//       && (ttMove==NULL || tt_capture) && !is_loss(beta) && !is_win(eval)
// return (RFP_RET_BETA_W·beta + (1024−RFP_RET_BETA_W)·eval) / 1024
```

This is the domain's typical texture: term-for-term logical identity, every
constant lifted to a named parameter, the return blend *generalized* (D50
one-degree-of-freedom fold), and 100+ lines of original commentary and
experiment scaffolding around 15 lines of SF-shaped logic. That pattern —
not transliteration, not independence — is why the dominant rating is L2.

Comparisons performed the same way (ours vs `git show eca43a97:src/…`),
with no worse verdict found: NMP (mod.rs:3519–3592 vs search.cpp:990–1026),
ProbCut (3613–3711 vs 1033–1099), the §3.5 pruning block (3950–4125 vs
1147–1216), singular (4136–4220 vs 1218–1283), reduction finalization
(4293–4351 vs 1284–1330), LMR/full-depth (4420–4562 vs 1330–1390),
`update_all_stats` (history.rs:679–786 vs search.cpp:1898–1956),
`update_continuation_histories` (history.rs:564–622 vs search.cpp:1958–1984),
qsearch pruning (5614–5682 vs 1731–1768), aspiration (2444+ vs 357–439),
voting (766–806 vs thread.cpp:350–401), TM init (timeman.rs:285–390 vs
timeman.cpp:46–135), cuckoo (cuckoo.rs vs position.cpp:106–160).

---

## 4. Originality census

Rough proportions over the search domain's *shipped decision logic* (what
the engine actually does in a match, weighting mechanisms not lines):

- **SF-SPECIFIC (SF's exact formula/constants/quirk, deliberately
  transcribed): ~70%.** The entire pruning/reduction/extension wave set,
  history family, TT policy, TM math, voting, aspiration.
- **SF-STANDARD (generic top-engine architecture): ~15%.** PVS negamax, ID,
  TT-as-cache concept, lazy SMP shape, mate-distance pruning, qsearch
  concept. (The boundary with the row above is blurry; together they are
  ~85% and that is the honest headline.)
- **RECKLESS-SPECIFIC (idea-level, attributed): ~5%.** D31 robustness
  bundle (poll cadence, qsearch re-poll, completed-depth guard), D30 threat
  map as an ordering feature, job dispatch, tunables-macro concept.
- **ARCHON-ORIGINAL: ~10% of shipped decision logic**, but a much larger
  share of the *codebase* (instrumentation + research infrastructure ≈
  20–25% of the domain's lines).

**ARCHON-original mechanisms, LIVING (shipped decisions):**
1. Quiet SEE ordering penalty with the good-pile clamp (movepick.rs:1057) —
   the domain's one games-proven original win (+11.00 ± 7.20, LOS 99.9%).
2. Quiescence recapture discount via `recapture_map`/`all_threats`
   (movepick.rs:227–294).
3. Evasion capture-history ordering term (study 70 census fix).
4. Aspiration `mss` divisor frame re-derivation (6776, params.rs:85–100).
5. Soft-stop instability decay 2.5 re-denomination (mod.rs:2159–2164).
6. LMR ttValue-rebate `is_valid` guard (mod.rs:4294–4300, measured).
7. Root exemption from sibling depth reduction (mod.rs:4773–4781).
8. Hindsight adjustments active in check (mod.rs:2980–2989).
9. No-followPV (a deletion-divergence, measured neutral-or-better).

**ARCHON-original infrastructure, LIVING (default-off / non-decision):**
draw contempt (tune-only), DIS-head consumers (eval/RFP/LMR/TM + census +
binary dump), TT fail-low-move store + upper-blind bitmask knobs,
`ARCHON_NMP_ALLNODE` A/B, SearchStats + 38-gate census, census70, foldcount,
the observation store (design/16 §3.5.5), `search_control.rs` (D41
contract), `search_trace.rs` (LmrObservation v3), `search_intervention.rs`
(counterfactual replay), the D50 `constants!`/`tunables!` split.

**ARCHON-original mechanisms, REMOVED (revival material, §1.3):** RFP threat
coupling; step-12 equal-value dither + child_pv_fresh; followPV mode;
TM/uncertainty head hooks; (offense term — Reckless idea, ARCHON build);
D41/D47 policy ordering plumbing (seam retained).

---

## 5. Differentiation opportunities

ARCHON's competitive claim cannot be its SF-parity core — that is a
baseline by construction. What it uniquely owns is (a) a search wired for
*measurement and intervention* at a granularity no reference engine has,
and (b) a licensed-clean pipeline for learned allocation. Concrete plays,
ordered by leverage:

1. **Learned counterfactual search allocation (design/19, D41) — the
   flagship.** The infrastructure is already built and original:
   `search_trace.rs` records every LMR decision with full context + latent;
   `search_intervention.rs` replays a single counterfactual scout depth
   deterministically; `search_control.rs` defines the shadow-inference
   contract with authority gating. Neither SF nor Reckless has anything in
   this class — SF's allocation is 30 hand-tuned coupled terms.
   *First experiment:* train a consequence head on LmrObservation v3 rows
   from the fn-game corpus, validate offline via `LmrIntervention` replay on
   three independent losses of one mode (the searchopt bar), then a
   fixed-node shadow SPRT with the head vetoing only the top-decile
   worst-predicted reductions.
2. **The disagreement head as a search signal (DIS-LMR/DIS-TM).** The hooks
   are live (default-off) and the negative results are already mapped
   (DIS-RFP null, DIS-EVAL negative — the eval+thresholds are jointly
   calibrated, mod.rs:4352–4363). DIS-LMR is the surviving channel that
   touches neither the eval nor a margin; no top engine modulates reduction
   by a learned "how much would search move this eval" estimate.
   *First experiment:* the coefficient panel already specified at the hook
   (fixed-node screen on the sign-test instrument, then real-TC for DIS-TM
   with the `ARCHON_DIS_TM_RANDOM` magnitude control).
3. **The ordering-signal class the SEE-penalty win opened.** Study 71
   proved a whole class exists: signals the seven-plane history stack is
   blind to (static exchange was the first). The census (38 gates +
   census70) is the screening instrument. Candidates: pin/x-ray-aware
   ordering terms (SEE's known blind spots, named in the study), and the
   Reckless ideas neither we nor SF ship — **fifty-move-bucketed correction
   history** (reckless notes 05 §3.2 flags it as "genuinely novel, cheap")
   and **root re-scoring mid-iteration** (notes 05 §1.3; refreshes sibling
   order from just-updated histories — SF never does this).
   *First experiment each:* census fhf + 1000-position paired sign test
   (study 70's instrument), magnitude control mandatory, then SPRT.
4. **Anti-starvation draw handling (draw contempt, revived).** The
   mechanism came from a real diagnosed blunder class (perpetual worth −1
   sets alpha and starves a mating sibling by 10× nodes — mod.rs:100–112),
   not from imitation; SF has no contempt at all since SF 12.
   *First experiment:* small `DRAW_CONTEMPT` values at real TC, validated
   on ≥3 independent perpetual-class losses (the no-acceptable-blunders
   bar), then SPRT.
5. **The observation store as a public research artifact.** Per-node,
   per-move Layer-4 search records (design/16 §3.5.5) are something no
   open engine publishes. As a publication/differentiation vehicle
   ("search-decision dataset + counterfactual replay harness") it converts
   infrastructure that is already written into the research-engineering
   visibility the project's owner wants.

---

## 6. Bottom line

- **Compliance:** no verbatim or transliterated GPL/AGPL text in `src/`;
  two small L2/L3-boundary blocks (partial_insertion_sort, TtEntry::save)
  are the worst concrete items and both are micro-scale with independent
  expression around a mirrored logic core. The real exposure is the
  architectural one (F1): a deliberately SF-parity search with SF's full
  constant set. Licensing ARCHON GPLv3 on any distribution retires the
  question completely and is the recommended posture.
- **Originality:** ~85% of shipped search decisions are SF-shaped (that was
  the S0 program's explicit goal and it is documented everywhere); ~5%
  Reckless-idea; ~10% ARCHON — but the ARCHON share contains the only
  games-proven original strength win in the domain (the clamped SEE
  ordering penalty) and a measurement/intervention/learned-allocation stack
  that has no counterpart in any reference engine. The differentiation path
  runs through that stack, not through further SF parity.
