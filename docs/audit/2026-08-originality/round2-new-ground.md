# ARCHON originality & license audit — ROUND 2: new ground (2026-08-10)

Round 1 (`00-synthesis.md` + three domain reports) covered `src/`, the
distilled notes, the repo-wide comment shingle sweep, and constants; its
remediation branch is **merged on main** (verified: `e3a8b78f`, `89c3dcf9`,
`1424f8c9`, `4540388c`, `34af97b7`, `823c1287` are all ancestors of `main`).
Round 2 covers the ground round 1 did not: **(1)** the SF-net teacher
distillation, **(2)** git history as distribution, **(3)** training-data
provenance, **(4)** the unswept trees (`tools/` in full, `searchopt/`,
`tests/`, `benches/`, `design/`, `jax/` round-2), **(5)** the release-artifact
surface.

Two sub-audits (tools/tests/benches; searchopt/design/jax) ran as parallel
agents; the supervising session **independently re-verified** their
load-bearing claims before this synthesis: the `design/11` `Eval::evaluate`
transcription side-by-side vs `eca43a97:src/evaluate.cpp:55-68`, the
`design/10` LMR accretion block, the tracked-UHO-slice subset test
(`grep -c -x -F -f` → 300/300 and 1000/1000), the `verify_forward.sh`
heredoc (35/35 lines in SF's `Defaults`), and the bullet MIT license text.
The CC0-teacher facts were verified against the live
`official-stockfish/networks` repository, not a search summary; R2-1 (the
tracked SF source copy) is the supervising session's own finding, made by a
GPL-header + SF-symbol sweep of tracked `docs/`+`old/` files that round 1's
comment-shingle method did not reach.

## 0. Findings ranked by severity

| # | finding | level | disposition |
|---|---------|-------|-------------|
| R2-1 | **A complete verbatim copy of Stockfish source is git-tracked in the CURRENT tree**: `docs/research/perf-artifacts-11/sfcount11-nnue_accumulator.cpp.txt` — all 819 lines of `nnue_accumulator.cpp` incl. the GPL header. Companion class: perf-annotate **disassembly listings of the SF binary** (`annot-sf*`, 9 files, 3 with full `Disassembly of section .text` bodies). Round 1's shingle sweep did not reach `docs/research` artifact text files | **L4, current tree** | delete the source copy + the disassembly listings (research value is in the numbers, which the study docs already quote); history copies then fall under R2-2 (§2f) |
| R2-2 | **Git history carries every pre-remediation L4 blob** — 52 commits of `sota_head.rs` with the verbatim AGPL bucket table, the GPL/AGPL `*-evalraw-v2.patch` files, the 16 MB Reckless profraw — and `origin` (private GitHub) already holds a pre-remediation snapshot | L4-in-history | scrub or fresh-history publish before ANY public push; reset `origin` (§2) |
| R2-3 | **`design/` docs transcribe SF search/eval/timeman statement-for-statement** as Rust pseudocode with all constants — `Eval::evaluate`, aspiration loop, `RootMove`, LMR accretion, history updates, TM init (91% constant-overlap in doc 09) | L3 (docs) | GPLv3 at distribution moots it (same bucket as round 1 finding 4); rewrite only if a non-GPL posture is ever wanted (§4a) |
| R2-4 | **SF-net teacher distillation** (alpha 0.3–0.4, entire current canonical lineage) | **resolved: LOW** | the teacher net is **CC0** — verified upstream. Disclose provenance; keep the converted teacher checkpoint private (§1) |
| R2-5 | **ODbL attribution notice owed on shipped nets** — our corpora include Lc0 data (ODbL+DBCL) and official-SF binpacks (declared ODbL) | obligation at ship | one notice in the release/NET-CARD; produced-work analysis in §3 |
| R2-6 | `tools/nnue/verify_forward.sh:36-72`: 35 SF bench FENs pasted as a heredoc **in SF's order, without naming SF** ("drawn from the bench suite") — the un-attributed twin of round 1 finding 5; verified 35/35 vs `benchmark.cpp` | L4 (data compilation, unattributed) | attribute now; switch to the staged ARCHON suite when it activates (§4d) |
| R2-7 | `tools/perf70/uho{300,1000}.fens`: 1,300 verbatim UHO book lines **git-tracked** (verified 300/300 + 1000/1000 subsets), violating the repo's own never-vendor policy (TESTING.md:231, D15); no in-file provenance | policy/hygiene | untrack or add provenance header (§4c) |
| R2-8 | No MIT notice anywhere for bullet-derived code: `jax/` parity kernels (`optim.py` AdamW port, checkpoint padding) AND `tools/nnue/train_smoke.rs`/`train_net0.rs`, close adaptations of bullet's MIT `examples/simple.rs` (12-13 whitespace-identical lines each) | notice hygiene | add THIRD-PARTY-NOTICES at ship (§4b, §4d) |
| R2-9 | ARCHON's own `pgo-prof-s28-*` profraw/profdata still git-tracked despite round 1's class-wide gitignore | hygiene | untrack (§2h) |

**Nulls that matter (each with method, §§below):** searchopt position sets are
credibly ARCHON's own (16,432 board-fields, zero overlap with SF/Reckless/
WAC/UHO); the tools/tests/benches verbatim-line sweep (691 files vs 1,950
Reckless + 6,110 SF source lines) found 5 hits, **all CPW perft FEN
literals**; `tests/data/perft.epd` is CPW positions 1-6 + the TalkChess/
Sedlak set + one ARCHON original, attributed in-file, transcribed from CPW
not SF (counter-field evidence, §4d); no STS/Arasan/ECM/Bratko suites
anywhere; no nnue-pytorch structural lift in `jax/`; no Integral source
anywhere in `tools/`; zero Reckless symbols in any tracked doc artifact
(§4e sweep); no foreign weights ever committed (`git ls-files` + 8,241-path
ever-committed sweep: no `.nnue`, teacher checkpoints never tracked);
tactical suites correctly fetched-not-vendored; round 1's
zero-verbatim-comments null re-confirmed by a method-different spot check
(seven distinctive SF comment phrases grepped; sole hit is an attributed
one-line quote, `src/search/params.rs:598`).

---

## 1. The teacher-distillation question — RESOLVED, mostly favorably

### 1a. The facts (receipts)

- **Mechanism.** Stockfish master's own net is converted weight-for-weight
  into an ARCHON parameter dict (`tools/nnue/sf_to_archon_warm.py`, header:
  "SF's weights can be poured into ARCHON's own parameter dict"; exactness
  gate 9550/9550 vs the `sf-eval` oracle,
  `docs/research/sf-teacher-blend-2026-08-01.md` §3) and run as a frozen
  live teacher. The student's value target becomes
  `da * teacher + (1-da) * label` (`jax/archon_jax/train.py:215`); the
  teacher is loaded from the run spec's `teacher_warm`
  (`jax/tools/pump_train.py:282-298`).
- **The teacher file.** `/home/nick/sf-eca43a97/src/nn-0ee0657fb25e.nnue`
  (named in `docs/research/sf-teacher-blend-2026-08-01.md:434-438`, sha256
  `0ee0657fb25e…`), a copy at `engines/bin/nn-0ee0657fb25e.nnue`; converted
  checkpoint `data/nnue/teachers/sf-eca43a97-teacher-raw.bin`. **Neither is
  git-tracked** (checked against the full 8,241-path ever-committed list).
- **Which nets carry the signal.** The ENTIRE current canonical lineage:
  `d49sfb-sb1758-s600` (fresh student, alpha 0.4, canonical 2026-08-02,
  `ledger/nets.jsonl`) → `d50w768-sb1030-s600` (alpha 0.4) →
  **`d53mix20-sb100-s600` (alpha 0.30) = the current canonical**
  (`engines/canonical/net.cenn`), and the in-flight 700 sb run
  (`tools/nnue/runs/d63_warm700.json`: alpha 0.30, warm from d53mix20).
  Nets promoted before 2026-08-02 (D48X and earlier) predate the SF teacher.
- **Annealing machinery exists.** `distill_alpha_ramp` schedules alpha to any
  value including 0 (`spec.py:542-552`), and the performance-triggered
  cutover (`train.py:313-336`, BAM-style, commit `2d493602`) decays a `trig`
  multiplier when the student catches the teacher, with the schedule as a
  hard backstop.

### 1b. The decisive fact: the teacher is CC0, not GPL

The premise of the question — "the teacher is Stockfish's own
GPL-distributed NNUE net" — is **wrong in the direction that matters**.
Verified 2026-08-10 against the live repository:

- `https://github.com/official-stockfish/networks` (the SF project's archive
  of every master-branch network) states in its README, verbatim: **"All
  networks in this repository were uploaded by the authors under the CC0
  license."** The repository itself is labeled CC0-1.0.
- **`nn-0ee0657fb25e.nnue` is present in that repository** (86.4 MB, Git
  LFS, master branch) — the exact teacher net.

CC0 is a public-domain dedication: no copyleft, no attribution condition, no
derivative-work restriction. Every GPL-inheritance argument about the
teacher's weights collapses at this step. Even the format-shifted teacher
checkpoint (`sf-eca43a97-teacher-raw.bin` — a literal derivative of the
weights) is unencumbered, and the distillation question does not even need
the derivative-work analysis below to come out clean.

### 1c. The analysis that would apply if the teacher were GPL (kept on the record)

Because the CC0 assertion is upstream's and could in principle be wrong for
some future teacher, the general question deserves its answer.

**Would distributing a student distilled from a GPL teacher create a
derivative work of the teacher?** The strong majority position is **no**,
on three independent grounds:

1. **No expression is copied.** A derivative work must incorporate protected
   *expression* from the original. The student (768-wide, different tail,
   own initialization lineage) contains none of the teacher's weight values
   — at different shapes, literal copying is not even possible. What the
   student absorbs is the teacher's input→output *behavior*, and behavior /
   functionality is exactly what copyright does not protect (the
   idea/expression dichotomy, 17 U.S.C. §102(b); the interoperability line
   of Sega v. Accolade through Google v. Oracle).
2. **GPLv3 disclaims output coverage by its own text** (§2, Basic
   Permissions): "The output from running a covered work is covered by this
   License **only if the output, given its content, constitutes a covered
   work**." A scalar evaluation of a chess position cannot constitute a
   covered work. Distillation consumes outputs at scale, but scale does not
   change the kind: the FSF's own traditional derivative-work theory is
   about linked/combined *code*, and even copyleft-maximalist commentary
   concedes the weights-from-outputs extension is beyond it and legally
   untested (see the 2025 survey "The Current State of the Theory that GPL
   Propagates to AI Models Trained on GPL Code", shujisado.org).
3. **Weights may not be protectable at all.** Machine-generated parameters
   with no human expressive authorship sit on thin copyright ice (US
   Copyright Office AI reports, 2023-2025: human authorship is required).
   For the teacher to encumber the student, *two* unsettled propositions
   must both resolve adversely — weights are protected expression AND
   function-approximation copies that expression. Each is individually
   doubtful; the conjunction is remote.

**The chess-engine community's revealed norms point the same way.** The
Fat Fritz 2 affair (2021) and the Stockfish v. ChessBase suit (settled
2022) were about shipping SF's *code and binary* with a swapped net — not
about training on SF's outputs. The line the community actually enforces
(TCEC/rating-list clone rules, the early-NNUE "borrowed net" bans) is:
**do not ship someone else's weights or code**. Training on another
engine's output is, by contrast, the ecosystem's standard practice —
Stockfish's own official nets are trained on Lc0-derived data plus
positions labeled by earlier SF searches; Lc0 exists in AlphaZero's
methodological shadow; Leela-data-trained NNUE engines (with attribution)
are unremarkable. ARCHON's blend — majority own-label signal, teacher at
0.3–0.4 for part of the schedule — is *more* conservative than a pure
relabel, which the ecosystem already accepts.

**Honest counter-argument, stated at full strength:** a distillation run
with alpha→1 over the teacher's full input distribution is functionally a
lossy copy of the teacher; someone who believes weights are expression can
argue the student "reimplements" that expression the way paraphrased code
still infringes. If a court ever accepted both premises, systematic
distillation would be the fact pattern most likely to trigger it. That risk
is real but small, and it attaches to a hypothetical GPL teacher — not to
this CC0 one.

### 1d. Labels from SF search (the easier half)

All `sf-datagen` corpora carry scores produced by *running* SF binaries
(`tools/nnue/gen_sf_data.py` and the stage-5 chain). Engine output is facts
about positions; GPLv3 §2 (quoted above) puts scores outside the license's
reach, and no copyright theory reaches a centipawn number. The evalraw
patches applied to SF/Reckless checkouts to extract those labels were
private modification (explicitly permitted; copyleft binds on conveyance,
and the patches were never conveyed — though their in-history copies are
§2's problem). The same logic covers the `obs-*-reckless` corpora labeled by
running Reckless. **No encumbrance flows through labels.**

### 1e. What still needs doing, and the options

Residual obligations/optics, in order:

1. **ODbL, one hop removed.** The teacher was itself trained on Lc0 ODbL
   data. SF's own CC0 dedication of that net *is* the community's
   produced-work interpretation in action (and our nets also touch Lc0 data
   directly — §3). An attribution notice at ship covers both hops.
2. **Honest provenance.** ARCHON's originality story ("first net our own
   data improved", ledger d53mix20) must not elide the teacher. A NET-CARD
   that states the distillation recipe (teacher net hash, alpha schedule,
   label sources) costs nothing and pre-empts the only genuine attack
   surface here, which is reputational, not legal. Note the run specs that
   record teacher provenance live in gitignored `tools/nnue/runs/` — the
   ledger row is currently the only tracked record; keep it complete.
3. **Never distribute the teacher-side artifacts**: `engines/bin/
   nn-0ee0657fb25e.nnue` (CC0, so even this would be legal — but it is not
   ours to ship), `data/nnue/teachers/*.bin`, `archon-sfswapbin`. All are
   untracked today; keep them out of any release recipe.

**Options for release nets:**

| option | cost | what it buys |
|---|---|---|
| **A. Ship the SF-teacher-taught lineage as-is, with a NET-CARD** | none | fully defensible: CC0 teacher + produced-work data analysis + disclosure. **Recommended.** |
| B. A + terminal teacher-free anneal (ramp alpha→0 for the last N sb of release checkpoints; machinery already live) | ~zero (schedule edit) | a cleaner headline ("final training phase teacher-free"); does NOT change lineage and should not be advertised as if it did |
| C. Teacher-free lineage for release (branch from D48X-era or scratch, own labels only) | weeks of GPU + measured Elo (d49sfb was +30.65 over D48X) | only worth it if the owner wants the marketing claim "no SF-net signal anywhere in the lineage"; not required by any license |

**Recommendation: A** (optionally + B as a free polish). The CC0 verification
removes the legal driver; what remains is a disclosure exercise, and
disclosure is cheaper and more credible than laundering the lineage.

---

## 2. Git history as distribution

**Premise check first:** the repo has a GitHub remote
(`https://github.com/TheGustafson/chessengine.git`). Unauthenticated API
returns 404 → **private**; no public distribution has occurred, and pushing
to one's own private repo is not conveyance to the public. But
`origin/main` sits at `0d9fab6d` (2026-08-03) — a **pre-remediation**
snapshot whose *working tree* still contains the verbatim AGPL bucket
table, the evalraw patches, and the profraw. Flipping that repo public (or
adding a collaborator) as-is would distribute round 1's L4 findings twice
over: once in the tree at `origin/main`, once in history.

Method for the sweep: `git log --diff-filter=D --name-only` (5,238 deleted
paths), the full 8,241-path ever-committed list
(`git log --all --name-only`), an all-blobs >2 MB size scan
(`git rev-list --objects --all | git cat-file --batch-check`), targeted
`git show` on every suspicious path, and per-commit content tests for the
AGPL table.

### The concrete scrub list (what a public push would distribute)

| bucket | contents | receipts | severity |
|---|---|---|---|
| a | **`src/eval/nnue/sota_head.rs` pre-remediation blobs**: the verbatim Reckless `OUTPUT_BUCKETS_LAYOUT` (round 1 finding 1, L4) and the pre-restatement `activate`/`find_nnz` kernels | **52 of 54** commits touching the file contain the 33-entry table (tested per-commit by content grep) | HIGH |
| b | **`old/tools/nnue/patches/`**: `stockfish18-evalraw-v2.patch` + `reckless-evalraw-v2.patch` carry verbatim GPL/AGPL context lines (round 1 finding 3); the `bullet-*.patch` files are MIT-context, fine | 2 commits (`be4b6c94` added, `1424f8c9` deleted) | HIGH |
| c | **`reckless_reference/pgo-prof-28/`**: 16 MB Reckless-derived profraw/profdata (round 1 finding 8) | 9 blobs, deleted `1424f8c9` | MED |
| d | **Pre-`4540388c` Reckless notes** (05/07/13) with the quoted AGPL fragments (round 1 finding 7) | note blobs in every commit before the conversion | MED |
| e | **15 deleted `integral_reference/*.md` notes** (upstream has NO license). Round 1 judged the content pure pseudocode (L0/L1 prose about Integral) | deletion commit in history | LOW |
| f | **R2-1's current-tree files, once deleted, join this list**: the 819-line SF source copy + the SF-binary disassembly listings under `docs/research/perf-artifacts-{10,11,17}/` have been tracked since their studies landed | `git ls-files` + SF-symbol sweep (§4e method) | HIGH while tracked; history copies handled by the same scrub decision |
| f2 | **Branch-only firewalled modules**: `src/eval/sf_swap.rs` (`9b9d5aa1`) and `src/eval/reckless_swap.rs` (`9321fe27`) — **NOT on main**, live only on side branches (e.g. `experiment/reckless-net-2`). Headers self-attest own implementation, "never copied"; the foreign *weights* were never in git. 121 local branches exist; only `main` was ever pushed | headers quoted in this audit's session log | LOW (publish main only) |
| g | **Bloat, not license**: a full committed-then-deleted `target32/` debug tree (largest blob 44 MB; includes MIT/Apache rlibs of criterion/proptest/clap — redistribution-safe classes), `perf42data/` IBS dumps (~110 MB), one perf `.data` file. `.git` totals **566 MB** | size scan | hygiene |
| h | **Currently tracked profraw**: `pgo-prof-s28-sota-arch-d47v/`, `pgo-prof-s28-sota-arch-d48x/` — ARCHON's own profiles (no foreign content), tracked from before round 1's gitignore, which does not untrack | `git ls-files` | hygiene |

Secrets: the ever-committed path list has zero hits for
credential/key/token/pem filename patterns (method: pattern grep over all
8,241 paths). That is a *filename* null only — run a content scanner
(gitleaks) before ever publishing history.

### Remediation (pick one, decision at first distribution)

1. **Fresh-history publication (RECOMMENDED).** Publish the audited tree as
   a new root (orphan branch or new repo); keep the research repo private.
   One command, zero risk of a missed blob, kills buckets a-h and the
   secrets question wholesale, and sidesteps the 121-branch surface. Cost:
   the public repo starts without the development narrative (the ledger
   carries the provenance story anyway).
2. **`git filter-repo` scrub.** Path-deletes for buckets b/c/e/g/h plus a
   `--replace-text` rule for the table lines in bucket a (52 blobs across
   the file's whole life — a content rewrite, not a path delete), then
   force-push. Preserves history shape; materially more error-prone;
   every clone/branch must be re-rooted anyway.
3. In both cases: **delete or force-reset `origin`** — it holds the
   pre-remediation tree today — and untrack bucket h in the live repo now
   (`git rm --cached`), which is worth doing regardless of distribution.

---

## 3. Training-data provenance

The repo's own records settle most of this (a genuinely good state —
round 1's "no foreign weights, disclosed data" posture extends to data
licensing):

| corpus class (ledger `data.jsonl`, 61 rows) | upstream license | receipt |
|---|---|---|
| `official-stockfish` binpacks (`fishpack32-*`, `nodes5000pv2-uho-binpack`) | **ODbL, declared by the dataset** | `docs/research/official-stockfish-binpacks.md:10` ("License declared by the dataset: ODbL"), source huggingface.co/datasets/official-stockfish/master-binpacks |
| Lc0 archives (`lc0-test91*`, `test80-*`, `farseert74/75/76`, T60/T70 bands) | **ODbL 1.0 (database) + DBCL 1.0 (contents)** | `docs/research/lc0-data-survey-2026-08-01.md:9-11`, upstream `LICENSE.txt` fetched verbatim 2026-08-01 |
| Lichess evals (`lichess-evals-v14-canonical-265m`) | **CC0** | `docs/research/lichess-evaluations-corpus.md:15` |
| `sf-datagen` house corpora (labels from running SF binaries) | outputs = facts; no encumbrance (§1d) | `tools/nnue/gen_sf_data.py` |
| `self-play` (`oracle-uho1-94m`, capture-scale) | ARCHON's own | ledger rows |

**Nets trained on this mix are ODbL "Produced Works", not derivative
databases.** ODbL §4.3 attaches exactly one obligation to a produced work: a
notice "reasonably calculated" to tell recipients that it was produced from
the licensed database and where the license lives. No share-alike, no
copyleft on the net. This is not a stretch reading — it is the reading the
repo's own survey records (`lc0-data-survey-2026-08-01.md:12-13`: "A net
trained on it is a 'produced work', not a derivative database"), the one the
ecosystem operates on (engines using Leela data ship attributed, per
`nnue-training-gap-study-2026-07-28.md:514` — "Alexandria: Leela data
(ODbL-attributed)"), and the one Stockfish itself instantiates by dedicating
its Lc0-data-trained nets to CC0 with Leela credited.

**Obligations, concretely:**
- **Shipping a net** → include the ODbL attribution notice (one paragraph in
  the release notes / NET-CARD: Lc0 training data, ODbL 1.0; official-SF
  binpacks, ODbL-declared; Lichess evals, CC0). That is the whole burden.
- **Shipping a corpus** (or any "substantial extract") → ODbL share-alike
  for the database applies. Do not publish corpora without deliberately
  doing ODbL compliance (attribution + ODbL-licensed redistribution +
  §4.7's parallel-open-copy rule if DRM'd). Today no corpus bytes are
  tracked, so this is dormant.
- **The UHO opening book** (`engines/books/UHO_Lichess_4852_v1.epd`, on
  disk, never tracked): Stefan Pohl's book, built from 2023 human Lichess
  games (Lichess database = CC0), redistributed openly by the
  official-stockfish/books repo. His explicit terms were **not found** by web
  search (method: three searches over sp-cc.de and talkchess). Default:
  don't bundle it in a release; using it locally for testing carries no
  obligation. If bundling is ever wanted, verify terms at sp-cc.de first.
  The tracked 1,300-line slices are R2-7 (§4c).

---

## 4. The unswept trees

### 4a. `design/` — SF transcription in the docs (R2-3, the worst new finding outside history and R2-1)

The numbered design docs carry Stockfish's search, eval and time-management
**statement-for-statement** as ```rust/```text pseudocode with every tuned
constant intact — under an explicit written policy
(`design/decisions.md:129-133`, D20: SF values "recorded verbatim in the
design docs"). Sub-audit method: every distinct 3-7-digit integer per doc
intersected against the same set from `eca43a97:src` (271 distinct — low
base rate), then side-by-side reads of the dense blocks. Supervisor
re-verified (a) and (e) directly.

Highest-density receipts (full table in the sub-audit, preserved in this
section's source notes):

- **(a)** `design/11-evaluation.md:927-943` **is `Eval::evaluate`** —
  `eca43a97:src/evaluate.cpp:55-68` in Rust: same statement order, same
  476 / 18236 / 534 / 77871 / 7191 / 199 constants, same TB clamp.
  Re-verified side-by-side this session.
- **(b)** `design/07-search-core.md:493-530` = the aspiration loop
  (`search.cpp:372-436`), constants and control flow (`5 + |mss|/10588`,
  `137*avg/(|avg|+81)`, `delta += 44*delta/128`).
- **(c)** `design/07:347-407` = `RootMove` field-for-field and
  method-for-method incl. byte-identical `score_is_exact_loss()` body.
- **(e)** `design/10-pruning-extensions.md:599-619` = the LMR reduction
  accretion (`search.cpp:1290-1328`) line-for-line — re-verified: 2766 /
  1017 / 838 / 923 / 955 / 3995 / 1059 / 1039 / 236 / 1079 / 1143 / 2016 /
  809 / 445/4096 / 272/(256d+285), same order, same conditionals.
- **(f)** `design/09-move-ordering-history.md:549-602` = the history-update
  family down to the `bonus > -7 ? 1038 : 525` asymmetry; **91%** of doc
  09's distinct integers appear in SF source.
- **(g)** `design/12-uci-time-management.md:573-586,735-760` = all
  seventeen `timeman.cpp:106-130` floats + the `search.cpp:568-587` set.

Attribution is present (docs name the SF files read; D20 is candid), so this
is disclosed transcription, not concealment — but "ideas and constants only"
(`design/09:52-68`) understates docs that also carry SF's statement order,
conditional structure and function decomposition. **Disposition:** the same
as round 1 finding 4, and the same cure: **GPLv3 at first distribution moots
it** (the docs then live in a GPL work citing GPL source). What this finding
*changes* is the option space: round 1 framed AGPL-vs-GPL as the only forced
choice; R2-3 means a permissive or proprietary ARCHON would additionally
require rewriting ~6 design docs, not just the code remediation. Record that
in the distribution decision.

Nulls in `design/`: docs 02/03/05 show no transcription (constant-overlap
10-24%, magic constants searched-at-init not hardcoded; the `32000/32001/
32002/246` value band is cited data). `design/06`'s bench-list adoption is
round 1's finding 5, already on the remediation runbook (staged replacement
suite `34af97b7`).

### 4b. `jax/` — bullet parity, MIT notice owed (R2-8)

Round 2's answer to "anything lifted beyond idea level" is: **from bullet
(MIT), yes and self-declared; from nnue-pytorch (GPL), no.**

- `jax/README.md:3-5` declares the trainer "a parity reimplementation" of
  bullet; `jax/archon_jax/optim.py:1-24` ports bullet's AdamW CUDA kernel
  semantics (verified against `adam.rs:35-52`: identical update order and
  default constants); `checkpoint.py:391,417` reproduces bullet's raw
  checkpoint padding (cyclic `b"bullet"`); `jax/rust-loader-design/src/
  lib.rs:26` depends on `bullet_lib` directly.
- **Disambiguation that clears most of it:** the "faithful transcription of
  the ModelDefinition in d40_features.rs" transcribes **ARCHON's own file**
  — `d40_features.rs` is ARCHON code dropped into bullet's `examples/`
  (untracked in the bullet checkout), so the model definition is L0. The
  bullet checkout itself (`.archon-state/toolchains/bullet`, MIT © 2023
  Jamie Whiting) is gitignored; only ARCHON's own `tools/nnue/bullet/*.rs`
  wrappers are tracked.
- Residual: MIT requires the copyright/permission notice in "copies or
  substantial portions". The ported kernel is small; whether it is a
  "substantial portion" is arguable — but a THIRD-PARTY-NOTICES file naming
  bullet is one paragraph and removes the argument. Required unambiguously
  if `rust-loader-design` (which links `bullet_lib`) is ever built into a
  distributed artifact.
- **nnue-pytorch null (method):** grep across all of `jax/` for its
  structural vocabulary (`LayerStacks`, `coalesce_ft_weights`,
  `virtual_feature`, `nnue_dataset`, `ranger`, `LightningModule`, `torch.`,
  `nnue2score`, `weight_clipping`, the 600/361 idiom) — zero hits beyond
  one attributed prose mention (`train.py:127`) and one docstring
  (`losses.py:36-51`) that quotes the loss shape with citation and
  **explicitly declines the constants** ("a shape reference, not values to
  copy", `losses.py:51`; `spec.py` ships `qp_asymmetry = 0.0`). L1 with
  attribution. No `torch` import exists; deps are numpy/jax/optax.

### 4c. `searchopt/` — clean, with the receipts that make it provable (and R2-7)

- **Position sets are ARCHON's own.** Method: all 28 files in
  `searchopt/positions/` sampled; **16,432 distinct board-fields** from the
  whole tree exact-intersected against SF `benchmark.cpp` (0), all FENs in
  SF source (0), all FENs in Reckless (0), WAC-revised (0 — WAC is
  Reinfeld's copyrighted book, so this null matters), the mate-in-N suites
  (0), and 200k lines of UHO (0). Provenance headers in 8/27 JSONs name an
  ARCHON net + ARCHON PGN + SF-as-oracle, and both cited PGNs exist on disk
  with consistent headers; the other 19 files are FEN-identical derived
  subsets. L0.
- **`searchopt/tools/` original** (12 file headers read; zero hardcoded FEN
  literals).
- **R2-7:** `tools/perf70/uho300.fens` / `uho1000.fens` are verbatim slices
  of the UHO book and **git-tracked** (re-verified: 300/300 exact-line
  subset), contradicting `TESTING.md:231-232` / D15 ("never vendored") —
  the only vendored third-party position data in the repo. Low absolute
  risk (opening FENs from CC0-derived games are facts) but a policy breach
  with no provenance header. Fix: untrack + regenerate from own games, or
  keep with a provenance/source header and a deliberate D15 exception.

### 4d. `tools/` (full), `tests/`, `benches/` — one unattributed L4 data lift, otherwise clean with strong nulls

Sub-audit scope: 878 non-build files across all 58 `tools/` subtrees, all 12
`tests/*.rs` + 4 `tests/data/*`, all 3 `benches/*.rs`. Supervisor re-verified
the top finding (35/35 heredoc-line subset test vs `benchmark.cpp`).

**Findings:**

- **R2-6 — `tools/nnue/verify_forward.sh:36-72`**: 35 FENs byte-identical to
  SF's `Defaults` vector, in SF's own order (monotonic subsequence of
  `benchmark.cpp:32-89`), introduced only as "drawn from the bench suite"
  (`:34`) — **never naming Stockfish**. Individual FENs are facts; the
  selection+ordering is SF's editorial compilation from a GPLv3 file. The
  un-attributed twin of round 1 finding 5 (which the remediation runbook's
  staged ARCHON bench suite already covers for `src/bench.rs`). Fix: add the
  attribution line now; retarget the script at the ARCHON suite when it
  activates.
- `tests/data/fens.txt:24-40`: 40 SF bench FENs, **attributed in-file**
  ("# --- Middlegames (Stockfish bench list; kiwipete) ---", declared at
  `:5`). Same class, disclosed; rides the same remediation.
- **SF WDL model in 5 tools files** (`sf_teacher_calibration.py:102`,
  `eval_scale_study.py:64-90`, `net_scale_check.sh:105-112`,
  `forensics/uciprobe.py:63-70`, `eval_error_study.py:62,88-90`): the
  `win_rate_params` coefficient vector + 17/78 clamp + /58.0 anchor from
  `eca43a97:src/uci.cpp:506-522`. Four of five attribute with file:line;
  `eval_error_study.py` carries the constants bare. These are fitted
  parameters (published in `official-stockfish/WDL_model`) mirroring what
  `src/uci.rs:1576-1600` already ships — round 1's constants-as-facts bucket.
  Fix: one comment line in `eval_error_study.py`.
- `tools/nnue/sf_to_archon_warm.py` (the teacher converter, §1): L2
  from-spec reimplementation of SF's `.nnue` container
  (`b"COMPRESSED_LEB128"` ↔ `nnue_common.h:67`, arch constants, serialization
  order), SF named with file:line throughout. No verbatim text. Fine.
- **R2-8 (tools half) — `tools/nnue/train_smoke.rs:31-103` and
  `train_net0.rs`**: close adaptations of bullet's MIT
  `examples/simple.rs:8-110` (12-13 whitespace-identical lines each, incl.
  the `sfbinpack_filter` body; headers say "a `bullet` example" but carry no
  MIT copyright/permission notice). Covered by the THIRD-PARTY-NOTICES item.
- `tools/nnue/bullet/d40_features.rs:305-326` + `d40_policy.rs:49-67`: the
  Reckless bucket boundaries as **attributed computed ranges** — the
  round 1 remediation form, correctly propagated. Confirmed identical
  partition to `Reckless/src/nnue.rs:82-91`; acceptable pending the OBL1
  retune retiring the legacy layout.
- Dependency notes (no action while private): `tools/nnue/sfbinpack-audit/`
  links the GPLv3 `sfbinpack 0.6.4` crate and its own `src/bin/*.rs` sources
  were deleted (only gitignored `target/` artifacts remain) — restore or
  document if the tool is ever needed again; `tools/ops/cloud/
  build_payload.sh:117-125` copies python-chess (GPL-3.0-or-later) into the
  GCP Batch payload tarball — internal compute, not conveyance to the
  public, but never repurpose that payload as a release artifact.
  `tools/nnue/bf_convert.rs` is clean (thin CLI over the MIT `bulletformat`
  crate; its own header's "GPLv? -> actually MIT" guess should just say MIT).

**`tests/data/perft.epd` provenance — answered, and it is fine:** the file is
three attributed parts. (1) `:21-41` the Chess Programming Wiki "Perft
Results" positions 1-6 (named in-file; our kiwipete/cpw3 carry CPW's `0 1`
counters where SF's bench uses `0 10`/`0 11` — evidence we transcribed CPW,
not SF). (2) `:43-88` the community TalkChess/Martin Sedlak tricky-position
set (named in-file; none of the 20 positions exists in the SF or Reckless
clones — grepped both). (3) `:88` one ARCHON-original from the M2 mutation
audit (`:83-87`). Independent-derivation marks: a recorded python-chess
re-verification of 130 counts, and `:57-60` *disagreeing* with a circulated
TalkChess value in favor of the recomputed one. Community-standard perft
positions are exactly the acceptable case — and the file documents it.

**Nulls (method named, sub-audit; spot-verified):** a normalized verbatim-line
sweep of all 691 `.rs/.py/.sh/.c/.md/.txt` files in the ground against
1,950 Reckless lines + 6,110 SF C++ lines (≥45 chars) found **5 hits, all
CPW perft FEN literals** (`benches/movegen.rs:26-31`,
`train_run.py:106` startpos); the SF scripts pool (583 lines) hit one
coincidental idiom; the bullet tracked-file pool (4,285 lines) hit only
API-call shapes outside F7's two templates; license-header grep found zero
pasted GPL/AGPL text in the ground (3 hits = the Reckless *attributions*);
the embedded-position-set sweep (all 15 files with ≥5 FEN lines vs SF bench
/ UHO / WAC / mate-in-N) accounted for every set — `calib/wl50.fens` and
`scale-check-800.fens` are ARCHON-generated (0 overlap anywhere);
`tools/spsa/` is Spall's published SPSA parameterization + fishtest-style
conventions, attributed, L1; no Integral source anywhere in `tools/` (8
case-insensitive hits, all one match script running a downloaded binary);
the binpack salvage tool reads only outer chunk framing, no codec port; a
URL census found 7 external URLs, every one a fetch, never a source copy.

### 4e. R2-1 in detail — the tracked SF source copy and disassembly artifacts

Method: every git-tracked file under `docs/` and `old/` (excluding
png/svg/data) tested for SF C++ symbol prefixes (`Stockfish::`); 47 hits,
classified; the same sweep for Reckless symbols found **zero**. GPL-header
grep over the same ground found exactly one file.

- **`docs/research/perf-artifacts-11/sfcount11-nnue_accumulator.cpp.txt`**:
  the complete `nnue_accumulator.cpp` (819 lines, 13 includes, full GPL
  preamble "Stockfish, a UCI chess playing engine derived from Glaurung
  2.1…") — a study-11 working copy that was committed with the perf
  artifacts. Verbatim GPL source in the current tree, exactly round 1
  finding 3's class after that finding was remediated. Delete.
- **`annot-sf*` perf-annotate listings** (perf-artifacts-10/-11/-17, 9
  files, 3 containing full `Disassembly of section .text` bodies of SF
  functions): disassembled GPL object code. GPL conveyance of object code
  "in any form" carries source obligations; a research listing is a gray
  zone not worth occupying. Delete alongside.
- The remaining ~37 SF-symbol files are flat profiles / block summaries /
  scripts — symbol names and percentages, i.e. measurement facts about a
  locally-run binary. Keep.
- `docs/research/perf-artifacts-11/counters11.patch` and `nulls11.patch`
  were checked and are diffs against ARCHON's own tree. Clean.

---

## 5. Release-artifact surface

**No packaging script exists** (null: no `tools/release*`/`dist*`, no
tar/zip recipe in `tools/*.sh`) — the release tarball is currently a
decision, not an artifact, which is the right time to pin its manifest:

| artifact | status | requires at ship |
|---|---|---|
| engine binary / source | ours; round 1 remediation merged; runbook steps (kernel property test, eval-hash gate, seeds, bench suite activation) still owed on the no-cargo box | LICENSE (GPLv3 per round 1 — moots R2-3 and finding 4) + `Cargo.toml` license field |
| net (`d53mix20` or successor) | SF-teacher-taught (CC0 teacher, §1); trained on ODbL data (§3) | NET-CARD: distillation provenance + ODbL attribution notice; pick the net's own license (CC0 à la SF, or GPLv3 with the engine) |
| opening book | NOT bundled today; UHO terms unverified (§3) | if ever bundled: verify sp-cc.de terms + credit; else ship book-less |
| trainer (`jax/`) if published | bullet-parity kernels (§4b) | THIRD-PARTY-NOTICES (bullet MIT); nnue-pytorch needs attribution-in-docs only |
| must NOT ride along | `engines/bin/*` (SF nets, sfswap/reckless binaries, external engines), `data/` (corpora, teacher checkpoints), `stockfish_reference/` / `reckless_reference/` clones, `engines/books/` | all currently gitignored — keep the tarball built from `git archive` of the public repo so gitignored trees cannot leak |

---

## 6. Updated pre-distribution checklist

Round 1 items (statuses from `00-synthesis.md`, unchanged today):
1-5 **DONE** (bucket retune in-worktree→merged; kernels restated; patches +
profraw deleted; note quotes converted). 6 **RUNBOOK/STAGED** (own magic
seeds; bench-suite activation — both need cargo). 7 **OPEN** (LICENSE at
distribution).

Round 2 additions (items 8-10 cost minutes and need no cargo — do now):

8. **Delete the tracked SF source copy and disassembly listings** (R2-1,
   §4e): `docs/research/perf-artifacts-11/sfcount11-nnue_accumulator.cpp.txt`
   + the `annot-sf*` files in perf-artifacts-10/-11/-17. The flat profiles
   stay.
9. **Attribute the SF bench heredoc** in `tools/nnue/verify_forward.sh:34`
   (R2-6) and add the one-line SF citation in `tools/eval_error_study.py`
   (§4d); retarget both bench-list users at the staged ARCHON suite when
   runbook step 5 activates it.
10. **Untrack ARCHON's own `pgo-prof-s28-*` profraw dirs** (R2-9, §2h) and
    the **UHO slices** (R2-7 — or header-and-except them).
11. **History: fresh-history publish (recommended) or filter-repo scrub**
    (§2 buckets a-g), and **delete/force-reset `origin`** — it holds a
    pre-remediation snapshot. Publish `main` only (branch-only swap modules,
    §2f2). Run a content-level secrets scan (gitleaks) if history is ever
    published instead of squashed.
12. **NET-CARD for every shipped net**: teacher provenance (net hash, alpha
    schedule, cutover) + ODbL attribution notice (Lc0 + official-SF
    binpacks) + label-source list (§1e, §3). Ledger rows already carry most
    of it; the run specs are gitignored, so the card is the tracked record.
13. **Decide the release-net teacher policy** (§1e): Option A (ship +
    disclose) recommended; Option B (terminal teacher-free anneal) free
    polish; Option C (teacher-free lineage) only for the marketing claim.
14. **THIRD-PARTY-NOTICES** naming bullet (MIT) — covers the `jax/` parity
    kernels and `train_smoke.rs`/`train_net0.rs` (R2-8); include at license
    time regardless — one paragraph.
15. **Never distribute** the corpora without deliberate ODbL compliance;
    never distribute `data/nnue/teachers/*`, `engines/bin/*` foreign
    binaries/nets, and never repurpose the GCP payload tarball (bundles
    GPL python-chess + the UHO book, §4d) as a release artifact (§3, §5).
16. **If a non-GPL posture is ever reconsidered**: R2-3 adds ~6 design docs
    to the rewrite bill on top of round 1's code findings — price it then,
    not now. GPLv3 remains the coherent, norm-aligned choice.
