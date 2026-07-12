# What made Integral succeed as a small solo project

**THE HEADLINE.** This is the payoff question of the whole Integral study:
*how did one college student, working ~part-time, take a ~10,500-line C++
engine from "initial commit" to a top-6, TCEC-competitive, Stockfish-beating
program in ~30 months?* This note ranks the factors by how load-bearing they
were, grounds each in the repo's own git history / README / build files, and
turns each into a concrete, actionable lesson for us — with explicit emphasis
on **what we are NOT currently leveraging and should.**

Distilled from the 11 topic notes in this directory, principally the two
success-factor deep-dives: `09-dev-history-progression.md` (the 669-commit
development story) and `10-ecosystem-force-multipliers.md` (the ecosystem
arithmetic). Companion code-adaptation payoff:
**[00-design-implications.md](00-design-implications.md)**.

> **Provenance & licensing.** `aronpetko/integral`, C++, **no LICENSE file at
> the repo root ⇒ all-rights-reserved by default** — strictly stronger than
> Stockfish's GPLv3 or Reckless's AGPLv3. Per D27 (applied here even harder),
> everything below is **facts** — commit hashes, dates, counts, file:line
> pointers, parameter names — never copied source. Ideas only.

---

## Executive summary — the top takeaways

1. **The single most decisive factor is free access to a distributed
   SPRT/SPSA test fleet (OpenBench), not individual talent.** The repo's own
   history claims ~51 commits with single-digit-Elo gains and runs seven
   named "75k–100k-game LTC SPSA" sessions; our own `design/14` §3.8 error-bar
   math (±44 Elo @100 games, error ∝ 1/√N) says a "2 Elo" verdict needs
   ~48,000 games. That volume is physically impossible on one desktop at the
   observed 1.5–3-month cadence. **Roughly half of "how one person reached
   top-6" is "they had a volunteer test fleet"** — the one resource no amount
   of extra developer-hours can substitute for. **This is the biggest thing we
   are not leveraging** (`tools/sprt.sh` is a single local machine; `design/14`
   §3.8 itself calls this "a concession to local hardware").

2. **Infrastructure was wired in *early*, before the engine was strong — it
   was a precondition for the climb, not a reward for it.** OpenBench support
   (`5f45ebc6`, "ob support stuff") landed 2024-04-23, ~6 months in and right
   at the start of the Feb–Sep 2024 sprint that took the engine from "plays
   legal chess" to top-6. The lesson: stand up the testing/tuning
   infrastructure *ahead of* needing it, not after.

3. **A disciplined, low-ceremony, one-idea-per-PR rhythm did the compounding.**
   Branch-per-hypothesis (~944 branch refs), squash-merge, a `Bench: <nodes>`
   trailer on 313/669 commits (47% — exactly the search-affecting ones, an
   independent external validation of our D24 line), commits shrinking to
   1–3 files / <20 lines by the mature era. None of this is clever; all of it
   is *sustained*. We already converged on the identical `Bench:` convention
   (D24) independently — confirmation we drew that line right.

4. **The NNUE data strategy is a two-phase bootstrap plus a deliberately
   cheap retrain loop.** HCE was a *thin bridge* (only 3 texel-tuner commits
   on `main`, 2024-06-01 → 2024-06-13; ~3 months from there to the first
   NNUE prototype) built to make `v4` "good enough to bootstrap." NNUE
   superseded it soon after — but not "within the hour": a first NNUE
   prototype (`cd3f7acb` "first nnue", 2024-08-27 — a `main`-unreachable
   commit, i.e. never merged as such; presumably reworked before landing)
   predates the `v4` HCE tag (`d4359007`, 2024-09-01) by 5 days, and the
   NNUE commit that actually merged and landed the switch (`99b7b22e`
   "NNUE (#52)", confirmed on `main`) followed `v4` by
   5 days and ~18.5 hours (2024-09-06 18:43, vs. `d4359007`'s 2024-09-01
   00:12) — corrected here from an earlier "same day, within the hour"
   claim in this note, which did not hold up against the commit timestamps.
   The expensive part was the ~6-month *architecture* search; once the
   architecture froze and the net was embedded in the binary, retrain cadence
   dropped to ~13 days (Zekrom v2→v7 in 67 days). This is almost exactly our
   D28/D34 plan — validating it — with one sharpened lesson: **get the full
   datagen→train→embed→bench loop working end-to-end early and cheaply, even
   at a weak first architecture, before polishing the architecture.**

5. **The whole thing runs on a shared ecosystem nobody re-invented.** CPW PSQT
   values, formulas openly borrowed from other engines (Alexandria's TM,
   "SF-style aspiration windows"), `bullet` (the same trainer we pinned in
   D28), Fathom (Syzygy), the UHO_Lichess_4852_v1 book (the same one in our
   `tools/sprt.sh`), `lichess-bot` for a free rating ladder. SF, Reckless, and
   Integral all triangulate on the *same* Fishtest-lineage testing and the
   same commodity plumbing. We already made the right calls on the book and
   the trainer — those are confirmations, not gaps.

6. **Calibrate expectations to the S-curve.** Integral's cheap, large wins
   (bundled TT/NMP/LMR/PVS, a single "+450 elo" commit) all clustered in the
   first ~6 serious months; the following ~18 months are rarer, smaller,
   more-expensively-gated gains. We are finishing P9–P11 — Integral's rough
   month-6–10 equivalent — so its history argues for **moving toward P13
   (NNUE) promptly once P9–P11 close, rather than over-investing in more
   P9-style micro-gains** whose Integral analogues were already single-digit
   Elo before NNUE ever landed.

7. **Our documentation/review overhead is not fat to trim — it is the
   substitute for the one thing a solo human has for free.** Integral has
   *zero* design docs, no `CONTRIBUTING.md`, no CI config: its entire "why"
   lives in one continuously-in-context human's head plus GitHub PR pages.
   Our "developer" is a rotating cast of context-reset agent sessions, so
   `design/`, `decisions.md`, `CLAUDE.md` (persistent memory) and the D33
   multi-agent review loop (a stand-in for self-review a solo dev gets free)
   are *structurally necessary*, not ceremony. Integral's leanness is not a
   model we can copy.

---

## The ranked factors

Ranked by how irreplaceable-by-effort-alone each was. Grouped into the three
axes the task names: **ecosystem / force-multipliers**, **development
discipline**, and the **iterative-NNUE data strategy** — but ordered globally
by leverage.

---

### #1 — Distributed test compute (OpenBench-shaped) — the dominant lever
*Axis: ecosystem/force-multipliers.*

**What it is.** A shared/self-hostable volunteer-CPU cluster that runs SPRT
(pass/fail an idea) and SPSA (tune constants) game-matches at a scale a single
machine cannot. The engine plugs in; it does not build the cluster.

**Evidence (in-repo, hard):**
- `src/engine/uci/uci.cc:248` carries the literal comment "OpenBench requires
  the bench command to be parsed from the command line."
- `src/tests/bench_test.cc` prints `"<nodes> nodes <nps> nps"` from a fixed
  49-FEN suite at pinned depth 12 — the exact string OpenBench's harness
  regexes to fingerprint a build.
- `Makefile` defines exactly six micro-arch build targets (`native, vnni512,
  avx512, avx2_bmi2, avx2, sse41_popcnt`) — OpenBench's contract for compiling
  on whatever CPU a volunteer worker reports. A hobbyist building only for
  themselves has no reason to maintain six.
- Wired in *early*: `5f45ebc6` "ob support stuff" (2024-04-23) changed
  `main()` to accept `argv` so `integral bench 12` runs from a shell — a
  test-cluster calling convention, not a human-at-a-UCI-prompt one. Flanked by
  visibly frustrated commits ("bench WTFFF", "ok last bench?", "kms") fighting
  to make the node count reproduce identically across machines.
- **The arithmetic (`10` §6, using our own `design/14` §3.8 constants):** a
  "~2 elo" claim needs ~48,000 games at 95% CI; a "5 elo" claim ~7,700. The
  log has 51 explicit "N elo" commits (mostly single-digit) plus 214
  PR-numbered commits, most SPRT-gated. The five SPSA sessions with disclosed
  counts alone sum to ≈448,000 games. **Total plausible lower bound: several
  million games** — multiple years of a dedicated 32-thread server running
  nothing else, versus the observed 1.5–3-month cadence. Only distributed
  testing closes the gap.

**Lesson for us — the single highest-leverage item in the entire study.**
We already emit the artifact OpenBench needs (`bench` subcommand + `Bench:`
trailer, D24) and already run local SPRT (`tools/sprt.sh`, D25). The gap is
purely the *distribution layer*. Concretely, what we are NOT leveraging:
- **Stand up or join an OpenBench instance** (the open-source Django app is
  self-hostable). Tag it to P14; it is a self-hosting/ops exercise, not an
  engine code change, so it is independent of the phase work.
- **Verify our `bench` stdout is OpenBench-parseable** (`"<n> nodes <n> nps"`)
  — ours currently reports via `check.sh`/`docs/perf-baseline.md`; confirm the
  exact regex-able string.
- **Add named micro-arch build targets** mirroring Integral's Makefile, beyond
  our single `RUSTFLAGS=target-cpu=x86-64-v3` recipe (D16), so a volunteer's
  worker can build the binary its detected CPU wants.
- **Budget the statistics prospectively**: before starting any change we expect
  to be worth single-digit Elo, write down the game-count it needs to clear the
  noise floor. If that number is unaffordable on one machine, that is itself
  the argument for OpenBench — not a reason to skip the test.

Without this, we stay capped at "however many games one local machine runs,"
which §6's math says means we can only ever resolve changes above ~10–15 Elo at
affordable N — and our own `design/10` backlog already exceeds what we have
had test-budget to land.

---

### #2 — SPSA auto-tuning at scale, on a recurring cadence
*Axis: ecosystem/force-multipliers × development discipline.*

**What it is.** A tunable-parameter scaffold (`src/tuner/spsa.h`: a
`TUNABLE`/`TUNABLE_STEP` macro pair) that is zero-cost `constexpr` in normal
builds but, under a build flag, registers each constant as a UCI option and
prints it in OpenBench's exact SPSA paste format (`name, type, value, min, max,
step, learning_rate`). Fed by the #1 cluster.

**Evidence:**
- **126 tunables** across 9 files (`constants.h` 76, `move_picker.cc` 14,
  `time_mgmt.cc` 10, `evaluation.h` 7, `history.h` 5, `bonus.h` 5,
  `correction_history.h` 4, `spsa.h` 4, `pawn_history.h` 1). ~27 are explicitly
  *frozen* (`disabled=true`) — active curation of the tuning surface, not
  "tune everything forever."
- **Seven named LTC SPSA campaigns over ~13 months**, spaced every 1.5–3
  months, escalating to 100k games / 12.5k iterations (PR #13, #97, #144, #164,
  #180, #204, #220). Each clusters *right after* a batch of new parameters
  landed (e.g. #164 follows "Fractional LMR" in the same PR) — SPSA re-settles
  the space each time enough knobs accumulate, never a one-shot finish.
- The tuning *infrastructure itself* needed iteration (`007f50d7` "SPSA
  automatic floating type conversions", `fdea7f88` "Disabled SPSA tuning") —
  tooling is not free even for an established convention.

**Lesson for us — build the scaffold now, ahead of the compute.**
- **Add a `cargo feature = "spsa"` tunable scaffold** (consts → UCI options +
  OpenBench-format print), populating `search/params.rs`. This is already on our
  roadmap (`design/00` P14, `design/10` §9.1) and the Reckless study flagged the
  same pattern ("Adapt #10"). Integral is a *second* independent confirmation
  the shape is worth building **before** the distributed compute exists, so the
  params file is ready the day capacity arrives. It is a mechanical, low-risk,
  one-time build — not a search-tree change.
- **Plan SPSA as a recurring cadence tied to milestones**, sized ~75–100k
  games / 10–12.5k iterations at LTC as a first budget estimate — run a pass
  after each pruning-wave batch and after each future NNUE net swap, not once at
  the end. This directly informs P14 sizing.
- **Run SPSA in large, infrequent, all-parameters-at-once batches** (Integral
  tuned 60–120 constants per session): the per-iteration cost is dominated by
  fixed game-setup overhead that does not scale with parameter count, so
  batching amortizes it — validating our own D32 batched-wave decision.
- **Curate the tunable set explicitly** — track "in rotation" vs
  "frozen/converged" as a first-class property (27/126 of Integral's are frozen,
  mostly depth-gate integers and all 10 TM constants), with small prep-commits
  ahead of each run to adjust ranges.

---

### #3 — Disciplined, low-ceremony, one-idea-per-PR development rhythm
*Axis: development discipline.*

**What it is.** Branch-per-hypothesis → self-reviewed, OB-tested PR →
squash-merge, each carrying a `Bench:` node-count trailer; commits shrinking to
single-parameter tweaks as the engine matured.

**Evidence:**
- **~944 branch refs**, descriptively named one-idea-each (`corrhist`,
  `singular_extensions`, `7buckets`, `nmp_eval_red`, `50mr_scaling`,
  `qs_lmp_nonpv`, `dynamic_qsfp`) — the branch name *is* the hypothesis.
- **Only 3 explicit reverts in 669 commits** — not because every idea worked,
  but because an SPRT-gated workflow means failed branches never merge into
  `main`. The visible log is survivorship-biased: **669 merged commits ≈ the
  tip of a much larger iceberg of tried-and-rejected ideas** (plausibly several
  times 669).
- **`Bench:` on 313/669 commits (47%)**, and the ones without it are
  overwhelmingly the non-functional commits (docs, Makefile, README) our own
  D24 explicitly exempts — an external data point that our D24 line ("commits
  that cannot affect search may omit the trailer") is drawn in the right place.
- **A sharp three-era discipline curve**, arrived at over ~2.5 years, not
  designed up front: Era 1 (2023-10→2024-05, 258 commits, 0% PR-numbered,
  bundled multi-feature commits, direct-to-`main`), Era 2 (the OpenBench pivot),
  Era 3 (2024-10→2026-04, 67.5% PR-numbered, near-total `Bench:` discipline,
  1–3-file commits). The commit *shape itself* (diff size shrinking) is a
  leading indicator of process maturity.

**Lesson for us:**
- **Our D24 `Bench:` trailer and D25/D32 SPRT rhythm are validated as
  industry-standard, not idiosyncratic** — a completely independent solo C++
  project converged on the identical conventions. Keep them.
- **Formalize a rejected-idea parking lot** (we already do this informally —
  "item 9 IIR reverted... parked on `wip/p9-item9-iir-followpv`"). Treat it as
  core process, not incidental cleanup, so a future context-reset session does
  not silently re-propose and re-test something already known to regress.
- **Expect our accepted-idea rate to look small vs the true attempt rate** and
  do not read that as a problem — it is the normal shape of SPRT-gated work.
- **Adopt the disposable-branch habit** (push, let the gate test it,
  merge-or-abandon, no cleanup ceremony) once a distributed gate exists —
  ~1 experiment branch/day is what a solo top-10 trajectory actually looks
  like.

---

### #4 — The iterative NNUE data strategy (HCE bridge + cheap retrain loop)
*Axis: iterative-NNUE data strategy.*

**What it is.** A pure self-bootstrap: the first NNUE was distilled from
Integral's *own* texel-tuned HCE (`v4`) via self-play; every net since was
retrained on fresh self-play data from the *prior* net (the literal
AlphaGo→AlphaZero loop) — no foreign teacher engine, ever. The HCE was a thin,
short-lived bridge, and once the NNUE architecture froze the retrain loop was
made deliberately cheap.

**Evidence:**
- **HCE was a bridge, not a destination**: only 3 texel-tuner commits on
  `main` (`0e2475f3`, `495ee23b`, `5fb587e4`), all clustered 2024-06-01 →
  2024-06-13 (a 4th, `aa826771` "Fixed tuner turned on", 2024-07-10, exists
  only on a never-merged branch and is not part of the `main` history);
  `v4` (`d4359007`, 2024-09-01) is the last HCE-era tag. A first NNUE
  prototype (`cd3f7acb` "first nnue") actually predates `v4` by 5 days
  (2024-08-27), and the NNUE commit that landed the switch (`99b7b22e`,
  #52) followed `v4` by 5 days and ~18.5 hours, not the "same day, ~1 hour
  later" this note previously claimed. After NNUE, texel-tuning commits
  stop forever.
- **Architecture search was the expensive part** (~6 months, ~4 arch-changing
  commits: `62618129` HL→1536 + 9 buckets, `581cebc1` L2/L3 tail, two "New
  network" tunes); **retraining afterward was cheap** — once the net was
  preprocessed and embedded directly in the binary (`576bef0b`, 2025-04-24),
  Zekrom v2→v7 landed in a 67-day window (~13 days average, one pair 1 day
  apart).
- **A branch-level A/B declined a foreign bootstrap source**: `own_hce_data`
  vs `lichess_hce_data` (both 2024-08-08) — self-play won, real-game positions
  were not shipped.
- **A DATAGEN build flag** (`#ifndef DATAGEN`, PR #174) gates search-only score
  adjustments (material-phase scaling, fifty-move damping) OUT of the recorded
  training label — a "decision-policy vs eval-accuracy" distinction discovered
  ~5.5 months into the NNUE era, not designed in.
- **Adjudication thresholds** independently near-identical to our own
  `design/15` §3.4 (win ≥2500cp/5 plies, draw ≤2cp/8 plies after ply 100), with
  two commits walking back an initially-too-aggressive draw adjudicator.
- **The loop has no finish line**: a `hardswishswiglu` activation branch was
  still live 2026-04-02, 19 months and 7 net generations after the first net.

**Lesson for us:**
- **Get the full datagen→train→embed→bench loop working end-to-end early and
  cheaply**, even at a weak/small first architecture, *before* polishing the
  architecture. This is the sharpened version of our D35 inference-first plan.
- **Pure self-bootstrap reaching top-6 validates D28/D34** and proves the
  SF-teacher shortcut (D28a) is *optional insurance*, not a requirement — keep
  the self-bootstrap path open as a real fallback if the SF pipeline hits
  friction.
- **Pre-empt the DATAGEN-flag footgun**: audit every score-adjusting heuristic
  (our `design/11` §3.6 rule-50 / optimism terms) so search-time fudge factors
  are not baked into training labels. Write it into `design/16` acceptance
  criteria before Tier-B's first corpus.
- **Budget P13 as a perpetual research thread**, not a milestone with an end
  state — expect a long low-yield stretch of net experiments after the first
  strong net, with search/SMP work carrying most of the Elo in between; do not
  mistake a run of losing net-retrain SPRTs for the NNUE bet failing.
- **Our planned CENN net format (magic + version + checksum) is stricter than
  both Integral's and Reckless's raw unvalidated struct read** — confirmed as
  the right call, not a gap.

---

### #5 — Standing on a shared idea/technique + commodity-plumbing pool
*Axis: ecosystem/force-multipliers.*

**What it is.** Techniques, tables, formulas, infrastructure code, books, and
rating ladders that already existed in the community, reused rather than
rebuilt.

**Evidence:**
- **Ideas credited in commit prose**: `1345776a` "CPW Piece Square Tables";
  `3418de6b` "Using Alexandria's soft limit calculations (43 elo LTC, 68 elo
  STC)" (a TM formula from a *different* named open engine, reported with its
  own re-measured Elo); `1568901f` "Use SF-style aspiration window adjustments."
  Same publish-a-technique-plus-its-Elo, re-derive-and-adopt norm we follow
  under D27.
- **Commodity plumbing vendored, not written**: `fathom` (Syzygy probing, MIT,
  multi-author incl. the TB format's own creator), `fmt` (MIT), `incbin`
  (public-domain-ish, net embedding). None chess-specific differentiation.
- **`bullet`** (jw1912's NNUE trainer, credited by name) — the *same trainer
  we pinned in D28*.
- **UHO_Lichess_4852_v1** opening book (README) — the *same book already in our
  `tools/sprt.sh`*.
- **A free rating ladder** via `lichess-bot` (config-file-only bridge to
  Lichess's rated bot pool) + a TCEC berth — external stages a solo dev walks
  onto for free.
- **Cross-engine human overlap**: 10 of 669 commits from 3 recognized community
  contributors (xu-shawn, Yoshie2000 — Weiss author, contributed the very
  correction-history commit that introduced the first `Bench:` trailer;
  Jonathan Hallström) who circulate between engines.

**Lesson for us:**
- **We already made the right calls on the book (UHO) and the trainer (bullet)**
  — two unrelated top engines converged on both. Confirmation, not a gap; worth
  a one-line "independently corroborated by the Integral study" note in
  `design/14`/`design/15` so a future reader does not second-guess it.
- **When Syzygy is scheduled, wrap Fathom (or a safe-Rust equivalent), don't
  design a prober** — consistent with our D27 reuse stance. (Caveat: Fathom is
  C + FFI + runtime libc, which conflicts D14/D15 — scope it as a non-default
  opt-in feature with its own decision, per D31c.)
- **The idea-pool leg is real but only a multiplier on judgment, not a
  substitute** — plenty of engines read the same CPW pages and don't reach
  top-6. And it is the leg we *already have* (three deep engine studies in);
  the leg we have *nothing* built for is distributed validation compute (#1).
  If forced to choose where the next unit of infra effort goes: **testing
  throughput before technique breadth.**

---

### #6 — Net distribution as a one-line diff (separate release repo)
*Axis: development discipline / iterative-NNUE strategy.*

**What it is.** The net is auto-downloaded at build-configure time from a
*separate* GitHub Releases repo (`aronpetko/integral-networks`) keyed by a
single `NETWORK_NAME` string (`CMakeLists.txt:23-45`); shipping a new net is a
one-line diff + a release upload, and the main repo never bloats with binary
history (PR #212 "Switch to Zekrom v7 network" is literally that diff).

**Lesson for us.** This directly answers our open question in `design/00` P13
row ("Net binary distribution: release assets vs git LFS"): **recommend release
assets over git-LFS**, mirroring this exact mechanism. Adopt a themed-codename +
linear-version naming scheme (Integral's Zekrom v2–v7). Rust's `include_bytes!`
gives us the binary-embed step Integral needed a third-party `incbin` header for,
for free. Keep the offline weight-layout preprocessing (transpose/permute for
SIMD) as a separate step from trainer export and runtime loader, so the
one-line-swap story survives our stricter CENN validation.

---

### #7 — Sequencing: ride the S-curve, don't fight it
*Axis: development discipline (process wisdom).*

**Evidence.** Commit volume is a visible S-curve: peak 96/month (May 2024)
during the bootstrap→NNUE sprint, tapering to a handful every few months by
2025–2026. The cheap large wins (bundled TT/NMP/LMR/PVS, the "+450 elo" commit)
all landed in the first ~6 serious months, *before* NNUE. Diff size and Elo-per-
commit both shrank steadily; the *rate of ideas* dried up faster than the
*rigor* did. Only 262 of ~890 calendar days had any commit (~29%) — a side
project with real gaps, including a 3.5-month early hiatus.

**Lesson for us.** We are finishing P9–P11 (pruning waves + TM + lazy SMP) —
Integral's rough month-6–10 equivalent. Per `design/00`'s own framing, **NNUE
(P13) is the next big-lever phase, not another round of hand-tuned
search-parameter polish.** Integral's history argues for moving toward P13
promptly once P9–P11 close, rather than over-investing in P9-style micro-gains
whose Integral analogues were already single-digit-Elo commits before NNUE
landed. (This dovetails with D32's framing: incremental Elo optimization is not
the P9 bottleneck — NNUE, SIMD, and concurrency are the priorities.)

---

### #8 — The individual's irreplaceable contribution (taste + 30 months of grind)
*Axis: the honest counterweight.*

**What the ecosystem could NOT supply**, and the developer had to bring:
- **Taste** — which of hundreds of possible experiments is worth a scarce test
  slot. The cluster validates ideas; it does not generate them. Someone reads
  SF/CPW/other engines, forms a hypothesis specific to *this* engine's current
  weaknesses, and implements it correctly enough to test at all.
- **~30 months of unglamorous discipline** — the `spsa.h` scaffold had to be
  *built* before external compute could fill it; the `Bench:`/PR hygiene
  *maintained* commit after commit; `.clang-tidy`/`.clang-format` kept clean.
- **The judgment to know *when* a borrowed technique transfers** — Alexandria's
  TM formula was re-measured for its Elo in Integral's own context, not
  copied-and-trusted.

**Lesson for us.** This is where our situation *differs structurally* and our
process must be **heavier** than Integral's, not lighter:
- Integral gets **self-review for free** (one continuous mind wrote and reviews
  the code). Our sessions have boundaries, so **D33's multi-agent A/B/C review
  loop is doing necessary work Integral never needed** — keep investing in it.
- Integral needs **no design docs** because one human holds all the "why." Our
  "developer" is rotating context-reset agents, so `design/`, `decisions.md`,
  `CLAUDE.md`, and auto-memory are the *substitute* for that continuous memory —
  **not overhead we could shed by copying Integral's leanness.**
- **A public strength signal** (Lichess bot + rating list) is a cheap,
  unilateral validation loop worth planning as a post-M9 (NNUE) milestone in
  `design/00` — not urgent now, but currently unplanned.

---

## Where we already match or exceed Integral (record, so we don't "fix" it)

- **`Bench:` trailer + SPRT gating (D24/D25)** — converged on independently;
  Integral confirms the convention, doesn't extend it.
- **`bullet` trainer (D28)** and **UHO book** — same choices, already decided.
- **Stricter net-file format** (planned CENN: magic + version + checksum) vs
  Integral's raw unvalidated struct read.
- **Persistent design/decision layer + multi-agent review (D33)** — necessary
  for our rotating-context situation; Integral's zero-doc leanness is *not*
  transferable to us.
- **Conservative draw adjudication + provenance metadata** — our `design/15`
  §3.6 `meta.json` discipline is a genuine improvement over Integral's zero
  committed per-run provenance.

## The one-line verdict

**Individual taste and 30 months of disciplined engineering were necessary but
not sufficient; the sufficient ingredient a solo dev cannot self-supply is
statistically-confident validation at scale (a volunteer test fleet). We
already have the ideas, the discipline conventions (D24/D25/D32/D33), the book,
and the trainer. The one leg we have nothing built for — and the highest-
leverage thing to build next on the infrastructure side — is distributed
testing compute.**
