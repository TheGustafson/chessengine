# Integral — the 669-commit development story (SUCCESS-FACTOR study)

Source studied: the full git history of `integral_reference/integral`
(`aronpetko/integral`, C++, no explicit LICENSE file at the repo root —
**all-rights-reserved by default**; per the task's stricter-than-D27 rule
this note contains **zero copied source**, only prose analysis of commit
metadata, timestamps, file lists, and the README, plus my own re-derived
pseudocode where a mechanism is described). Primary sources used: `git log
--oneline` (669 commits total), `git log --format` with dates/authors/stats,
`git log --merges`, `git tag -l` (`v1`…`v7`), `git show --stat` on ~15
pivotal commits, `README.md`, `Makefile`, `.gitignore`, and file listings of
`src/`, `shared/`, `third-party/`, `preprocess/`. No `.cc`/`.h` file content
is quoted below beyond identifiers already public in the README (net
architecture string) and the standard 50-FEN OpenBench bench suite (which is
a shared community artifact, not Integral-original — `src/tests/bench_test.cc`).

This note is the "how did one person do this" story. A companion review of
concrete techniques (search/eval/NNUE mechanisms) belongs in the other
numbered notes in this directory; here technique names appear only as
timeline markers for the process story.

---

## Executive summary

1. **Integral is ~98.5% one person.** Of 669 commits, 659 are Aron
   Petkovski under 4 git identities; only 10 commits (1.5%) come from three
   named outside contributors, each a known figure in the wider
   engine-programming community (§8). This is as close to "solo" as a
   top-10 engine gets.
2. **The project has a sharp, dateable three-era discipline curve**, not a
   flat process from day one: (1) Oct 2023–May 2024, no PRs, no bench
   trailer, bundled commits, direct-to-`main`; (2) Jun–Sep 2024, the
   OpenBench pivot — bench trailers and PR numbers appear and gradually take
   over, NNUE lands; (3) Oct 2024 onward, near-total PR+`Bench:` discipline,
   commits shrink to single-parameter tweaks (§2–4).
3. **The real inflection point is infrastructure, not talent**: on
   2024-04-23/24, right after the biggest single Elo jump in the whole log
   (`54b86b59`, "lmr and 3fold pvs (+450 elo), bench"), the dev spent one
   messy night wiring up OpenBench support (`5f45ebc6` "ob support stuff",
   flanked by commits literally titled "bench WTFFF" and "kms"). Commit
   density and rigor both step-change immediately after this, not before.
4. **The git log understates the actual experimentation rate.** Only 3
   commits in 669 are explicit reverts. This is not because every idea
   worked — it's because a SPRT-gated workflow means failed experiments
   never get merged into the history you're reading at all (§4.4). The true
   attempt count is invisible and almost certainly several times 669.
5. **NNUE happened once the HCE was "good enough to bootstrap," not once
   HCE was "finished."** `v4` (tagged at the datagen-infrastructure commit,
   2024-09-01) is explicitly the last hand-crafted-eval version; NNUE landed
   5 days later and texel-tuning commits basically stop forever (§6). The
   HCE tuner itself only has 4 commits total in the whole history — it was a
   short-lived bridge, not a destination.
6. **SPSA re-tuning is a recurring campaign, not a one-off**: 7 named LTC
   SPSA PRs from 2024-07 to 2025-08, spaced roughly every 2–4 months,
   escalating from unspecified size to 100k games / 12.5k iterations by the
   last one (§7).
7. **No design docs, no CONTRIBUTING.md, no CI config were ever committed.**
   The entire process lives in GitHub PR titles/descriptions (invisible to
   this clone) plus two mechanical conventions borrowed wholesale from the
   Stockfish/OpenBench ecosystem: the `Bench: <n>` trailer and PR-number
   squash commits. The process was **inherited, not invented** (§8, §9).
8. **The pace is a visible S-curve.** Commit volume peaks at 96/month
   (May 2024) during the bootstrap-to-NNUE run, and by 2025-08 through
   2026-04 the project produces only a handful of commits every few months —
   large, cheap wins cluster early; later commits are single-line parameter
   nudges gated by full LTC SPRT (§4, §10).

---

## 1. The arc at a glance — three eras

| Era | Span | Commits | PR-numbered | `Bench:` trailer | Character |
|---|---|---|---|---|---|
| 1 — Bootstrap | 2023-10-25 → 2024-05-31 | 258 | **0 (0%)** | ~0 | Solo hacking, direct-to-`main`, bundled multi-feature commits, no external testing tool |
| 2 — Pivot | 2024-06-01 → 2024-09-30 | 211 | 79 (37%) | growing | OpenBench wired in, `Bench:` trailer appears, NNUE + Lazy SMP land, Elo-in-subject convention dies out |
| 3 — Mature | 2024-10-01 → 2026-04-02 | 200 | 135 (**67.5%**) | high | Near-total PR+SPRT gating, single-parameter commits, periodic SPSA/net-swap campaigns, tapering cadence |

(Counts via `git log --since/--until --oneline` and `--format='%s' | grep -cE
'\(#[0-9]+\)'` over each window; see §10 for the corroborating diff-size and
revert-rate evidence.)

Total real-world span: 2023-10-25 to 2026-04-02 (the newest commit in this
clone) is ~890 calendar days, but commits land on only **262 distinct
days** (~29% of the span) — this is a side project fit around school/other
commitments, not a continuous full-time effort, confirmed directly by a
3.5-month total silence from 2023-11-05 (`8de9711f`) to 2024-02-21
(`e30f5dbd`) right after the very first movegen/checks commits — a
classic "started strong, disappeared over a break, came back" arc.

---

## 2. Era 1 — bootstrapping alone (Oct 2023 – May 2024)

The first 8 commits (2023-10-25/26) build board representation and basic
move make/undo; commit 2 is literally `99129a01` "downgraded to c++17,
implement basic move make & undo" — even the language standard was still
being decided. After the Nov–Feb hiatus, commits resume 2024-02-21 and are
large, bundled, and un-tested-in-any-formal-sense: `048c4e26` "negamax move
searching, alpha-beta pruning, piece square table evaluations,
optimizations, etc" (2024-02-27) lands three separate architectural
components in one commit. Zobrist hashing, a transposition table, null-move
pruning, iterative deepening, magic bitboards, a PV line, LMR, an
"improving" heuristic, and SEE-based QS pruning all land between late
February and late March 2024 — all pre-formal-testing, judged (as far as the
history shows) by eyeballing/gut feel and ad hoc self-play, not SPRT.

The tag `v1` (`9d2ce3db`, "README update", 2024-05-18) is revealing: the 20
commits made that single day are almost entirely Makefile/build-system work
("Makefile update" ×3, "makefile debug" ×3, "Build support to Makefile",
"Makefile finalizations") — `v1` was cut the moment the engine became
**installable by someone else**, not at a feature milestone. This is a
useful released-vs-done distinction: the first tagged "version" of a
personal chess engine is a packaging event, not a strength event.

---

## 3. Era 2 — the OpenBench pivot (Jun – Sep 2024): the real inflection

The first commit subject to record an explicit Elo delta is `c03f4c45`
"search features shuffled around, futility pruning (~30 elo), 3fold PVS"
(2024-04-01) — informal SPRT-style testing (presumably local
cutechess-cli/fastchess runs) predates any tooling formalization by about
three weeks. The pattern intensifies: `54b86b59` "lmr and 3fold pvs (+450
elo), bench" (2024-04-23 17:20) is the single largest annotated jump in the
entire log, and the word "bench" in that subject is the tell — a `bench`
subcommand was just added.

Within hours of that commit, the same evening (2024-04-23 18:34 → 2024-04-24
02:02), a burst of ~14 commits wires up OpenBench compatibility:
`a9fa8b0c` "bench fixes, perft suite, see suite data, test command",
`5f45ebc6` "ob support stuff" (a 3-file, 9-line UCI-option-plumbing change —
`src/chess/uci.cc`, `src/chess/uci.h`, `src/main.cc`), then a visibly
frustrated sequence of `d3561e0f` "makefile fix", `317cbc37` "bench again",
`5d9f3361`/`0e6ca441` "bench WTFFF" (×2), `5a20071c`/`7a6b46a7` "ok last
bench?" (×2), `98df5b87`/`d3ce41a6` "kms". This is a solo developer fighting
their own build system to get the `Bench:` node count to reproduce
identically across machines — the exact mechanical requirement OpenBench
imposes (its web UI validates NPS scaling by dividing worker-reported bench
nodes by worker-reported bench time, so every worker's binary must print the
*same* node count for the *same* fixed position suite regardless of CPU).
The first commit to carry the now-familiar `Bench:` trailer is `d0fc3734`
"Correction history indexed by pawn hash. Bench: 1159053" (2024-06-17) — 8
weeks after the OpenBench-wiring night, once the convention had settled.

Two other landmark features close out this era: **Lazy SMP** ships as
`080d346f` "Lazy SMP (#30)" (2024-07-29), preceded by `05a0f454` "Thread pool
foundation (#25)" (2024-07-25); and **NNUE** lands as `99b7b22e` "NNUE (#52)"
(2024-09-06 18:43), one hour after `12d7d0f8` "Update version number for v4"
(2024-09-06 17:55) — i.e. the dev tagged the last hand-crafted-eval version
and merged NNUE within the same hour (`git log -1 --format='%ai %h %s' v4` →
`d4359007`, "Data Generation (#51)", 2024-09-01, is the actual tagged
commit; the version-string bump five days later is bookkeeping around that
tag). This is the exact HCE→NNUE bootstrap our own D28/D34 plan is modeled
on: the datagen pipeline is built and proven on the mature HCE first
(`d4359007`), *then* NNUE replaces the eval function it was built to feed.

Also in this window: the last "elo" annotation anywhere in the log is
`8af5a61c` "Make dynamic improving more nuanced (~2 elo)" (2024-07-06); after
NNUE lands (2024-09-06) **zero** commit subjects mention "elo" again for the
rest of the project's life (`git log --since=2024-09-06 -i --grep=elo` →
empty). This is not a rigor regression — it is the "why" moving off the
commit subject line and onto the GitHub PR page (title + OpenBench test
link), which this clone cannot see. Once PR-number discipline took over, the
commit message became purely mechanical (feature name + `Bench:` count); the
evidence trail moved one layer up.

---

## 4. Era 3 — mature discipline (Oct 2024 – Apr 2026)

### 4.1 PR + `Bench:` become the norm, not the exception

From 2024-10-01 onward, 135 of 200 commits (67.5%) carry a `(#N)` PR
reference (vs. literally 0% before June 2024). Across the whole repo,
`git log --format='%B' | grep -c '^Bench:'` → 313 of 669 commits (47%)
carry the trailer — and the ones that don't are overwhelmingly the
non-functional commits our own D24 explicitly exempts (docs, Makefile,
`.editorconfig`, README) plus the pre-2024-06 era where the convention
didn't exist yet. That 47% figure is an independent, external data point
that our own D24 line — "commits that cannot affect search may omit the
trailer" — is drawn in the right place: a mature OpenBench-gated project
converges to *almost every commit that touches search/eval carrying the
trailer*, and *no others*.

### 4.2 Commits shrink to single-parameter tweaks

`git log --shortstat` over the last 30 commits (Nov 2025 – Apr 2026) shows a
typical diff of 1–3 files and single-digit-to-low-double-digit line counts:
"1 file changed, 1 insertion(+), 1 deletion(-)" appears five separate times
in that window; the largest is "11 files changed, 143 insertions(+), 126
deletions(-)". Compare this to the Era-1 commits at the same relative
position in the history (`git log --reverse` positions 40–60, Mar 2024),
which are all multi-word bundles like "mailboxed piece types, uci
improvements" or "search/eval changes" touching many files with no
size discipline at all. The commit *shape itself* is a leading indicator of
process maturity, independent of any commit-message metadata.

### 4.3 Feature branches, not a firehose into `main`

`git log --merges` surfaces ~944 distinct branch refs across the project's
life (`git log --all --format='%D' | grep -oE 'origin/[a-zA-Z0-9_-]+' | sort
-u | wc -l`), with descriptive names matching one idea each: `corrhist`,
`singular_extensions`, `lazy_smp2`, `7buckets`, `more_sext`,
`prefetch_tweak`, `nmp_eval_red`, `50mr_scaling`, `subnet`,
`qs_lmp_nonpv`, `dynamic_qsfp`. Even fully-solo work goes through a
branch → (self-reviewed, OB-tested) PR → squash-merge cycle in the mature
era; the branch name is itself the one-line hypothesis being tested.

### 4.4 Survivorship bias: the log only shows what won

Only 1 commit in the entire 669-commit `main` history is an explicit
revert: `03efc83f` "lmr revert" (2024-03-20, the *pre*-OpenBench era,
Era 1) — `git log --oneline main | grep -i revert` returns exactly this
one line. (`git log --all --grep=revert` turns up two more candidates,
`55998fd5` "revert" and `3eebba4a` "Reverted evaluation", but
`git merge-base --is-ancestor <hash> main` shows neither is reachable
from `main` — they sit on the never-merged `smarter_permuting` and
`data_generation` branches respectively, so they are not part of the
669-commit corpus this note otherwise scopes itself to; an earlier draft
of this note conflated an all-branches search with the main-only count.
If anything this strengthens the point below: those two are exactly the
"branch abandoned, no revert commit needed" case argued for two sentences
down, caught in the act.) In an OpenBench-gated workflow a near-zero
on-`main` revert rate is expected, not remarkable: a branch that fails its
SPRT test is simply never merged, so it never enters `git log main` at
all — there is no "revert commit" because there was nothing to revert
*from* `main`. The practical implication: **669 merged commits represents
a much larger number of *tried* ideas** — plausibly several times that,
going by typical SPRT pass rates reported in engine-dev communities, and
consistent with the ~947-branch count (§8) — and the git history is a
survivorship-biased view of "what worked," not a complete lab notebook.

### 4.5 The pace itself is an S-curve

Commits/month: 8 (2023-10) → 1 → [hiatus] → 19 (2024-02) → 54 → **79** →
**96** (2024-05, the peak) → 74 → 71 → 18 → 49 → 32 → 40 → 18 → 20 → 9 → 13 →
23 → 10 → 13 → 8 → 4 → 0 → 0 → 2 → 2 → 0 → 0 → 5 (2026-03) → 1 (2026-04).
The five-month run from February through June 1 2024 (258 commits) is where
essentially the entire classical alpha-beta search stack (TT, NMP, LMR,
PVS, aspiration windows, SEE-QS, staged move generation, killers,
history heuristics) got built and the OpenBench pivot happened; every era
after that is refinement with shrinking marginal commit volume — the
project reaches "large parameter space already explored" territory and the
easy wins run out. By the last 8 months in this clone (Aug 2025 – Apr 2026)
the cadence is roughly one PR every 3–6 weeks, each still individually
gated exactly as rigorously as the busiest month two years earlier (e.g.
`3f15ba6b` "Play quiet moves in QS if the TT move was quiet and not from a
fail-low (#238)", `484db145` "Update move ordering and futility pruning if a
move directly delivers a check (#236)") — the *rate* of ideas dried up
faster than the *rigor* did.

---

## 5. Condensed technique-progression timeline

(Dates from `git log -1 --format='%ai %h %s' <rev>`; PR numbers as they
appear in the subject; this table exists to anchor the *process* narrative
in real milestones, not to teach the mechanisms — see the other numbered
notes for that.)

| Date | Rev | Milestone |
|---|---|---|
| 2023-10-25 | `8d6ed33b` | Initial commit |
| 2024-02-27 | `048c4e26` | Negamax + alpha-beta + PSQT (bundled) |
| 2024-03-06 | `1da15166` | TT, null-move heuristic, iterative deepening |
| 2024-03-16 | `6da6e3da` | Magic bitboards |
| 2024-03-20 | `409c28bf` | LMR reduction table |
| 2024-03-29/30 | `d793c263`/`4c6f553f` | SEE-based QS pruning, LMP |
| 2024-04-01 | `c03f4c45` | First Elo-annotated commit (futility pruning, ~30 elo) |
| 2024-04-23 | `54b86b59` | Largest single jump on record: LMR+3-fold PVS, **+450 elo** |
| 2024-04-23/24 | `5f45ebc6` + burst | **OpenBench integration night** — the inflection point |
| 2024-04-24 | `a03ed018` | Killer moves |
| 2024-05-18 | `9d2ce3db` (tag `v1`) | First tagged/installable release (build-system milestone) |
| 2024-05-27 | `0fe92f51` | Tapered evaluation (145 elo) |
| 2024-06-07 | `495ee23b` (tag `v2`) | Texel tuner + new eval |
| 2024-06-17 | `d0fc3734` | **First `Bench:` trailer** commit (external PR from Yoshie2000) |
| 2024-07-09 | `20e5da17` | First `(#N)`-numbered commit in a subject (#7) |
| 2024-07-25/29 | `05a0f454`/`080d346f` | Thread pool, Lazy SMP (#25/#30) |
| 2024-07-30 (tag `v3`) | `fa9881c4` | |
| 2024-08-27 | `cd3f7acb` | "first nnue" (working branch) |
| 2024-09-01 (tag `v4`) | `d4359007` | Data-generation infra — last HCE-era tag |
| 2024-09-06 | `99b7b22e` | **NNUE (#52)** lands, 5 days + ~18.5h after the v4 version bump (corrected from an earlier "~1 hour" in this table — `d4359007` is 2024-09-01 00:12, `99b7b22e` is 2024-09-06 18:43) |
| 2024-09-18/19 | `796c4dca`/`5fa0c7e1` | NNUE horizontal mirroring; phase scaling (#69/#71) |
| 2024-10-22 | `62618129` | Net arch grows to 1536 HL, 9 factorized king buckets (#99) |
| 2024-10-23 | `c822a25f` | "Allow OB network caching" (#102) — explicit OpenBench net-caching support |
| 2024-11-01 (tag `v5`) | `61c92ad5` | |
| 2024-12-16 (tag `v6`) | `26d38b86` | |
| 2025-01-01 | `581cebc1` | Multi-layer network — L2/L3 added (#153) |
| 2025-01-30 / 03-17 | `f51c7200`/`8215e459` | "New network" ×2 (#167/#178) |
| 2025-02-14 (tag `v7`) | `9b7c23d6` | |
| 2025-04-24 | `576bef0b` | Net preprocessed + embedded directly in the binary (#190) |
| 2025-04-26 | `ecb262bd` | **Zekrom v2** (first *named* net) |
| 2025-05-15/25, 06-01/02 | `733036df`/`8ff9916f`/`ddf4558c`/`7a6999aa` | Zekrom v3, v4, v5, v6 (#196/#199/#201/#202) |
| 2025-07-02 | `322b6fe9` | Zekrom v7 (#212) |
| 2025-08-01 | `aa10d228` | NUMA first-touch policy for `Thread` objects (#218) |
| 2025-08-05 | `8babfda9` | LTC SPSA, 100k games / 12.5k iterations (#220) — largest SPSA campaign recorded |
| 2026-03 – 04 | `1568901f` … `3f15ba6b` | Continued single-parameter search tweaks (#234–#238) |

---

## 6. The NNUE story in detail: HCE bridge → first net → Zekrom v2–v7

The README states the final architecture as a horizontally-mirrored
perspective network with 12 factorized king input buckets, `768×12
(Factorized) → 1536` per perspective, then `16 → 32 → 1` across 8 output
buckets — i.e. it grew from a first cut (Sep 2024, unspecified small hidden
layer) through three distinct architecture-growth commits before reaching
that shape:

1. `62618129` (2024-10-22) grows the hidden layer to 1536 and adds 9
   factorized king buckets — the single-hidden-layer size and bucket count
   most of the "shape" work.
2. `581cebc1` (2025-01-01) adds the L2/L3 stages (the `16 → 32 → 1` tail) —
   "Multi-layer network."
3. Two more `f51c7200`/`8215e459` "New network" commits (Jan/Mar 2025)
   presumably tune bucket count to the final 12/8 and other hyperparameters
   before the architecture is considered stable enough to name.

Only *after* the architecture stabilized does `576bef0b` (2025-04-24)
preprocess the net and embed it directly in the binary (via `incbin`, per
`third-party/incbin/`) — and the very next commits are the first **named**
net swaps: `ecb262bd` "Switch to Zekrom v2 network" (2025-04-26). This
ordering matters: naming/versioning nets only started once (a) the arch was
frozen and (b) the retrain pipeline was cheap enough to produce nets on a
predictable cadence worth naming. From that point, Zekrom v2 → v7 (5
transitions) land in a 67-day window (2025-04-26 → 2025-07-02), averaging
~13 days apart but bursty — v5→v6 is one day apart (`ddf4558c`→`7a6999aa`,
2025-06-01/02, likely a quick fix to the v5 net or its integration), while
v6→v7 is 30 days. The lesson embedded in the *shape* of this cadence: once
architecture experimentation stops and the loop is "generate data with
current best net → retrain → swap," the loop becomes cheap enough to run
almost weekly — the earlier architecture-search phase (Oct 2024 – Apr 2025,
~6 months, 4 architecture-changing commits) is the expensive, slow part;
pure iterative retraining afterward is fast.

The HCE bridge itself is thin by design: only 4 commits in the whole
history mention "tuner"/"texel" (`0e2475f3` "Added tuner + tuned eval
terms", `495ee23b` tag `v2` "New evaluation + tuner", `5fb587e4` "Tuner
compilation error fix", `aa826771` "Fixed tuner turned on"), all clustered
around 2024-06-07. HCE tuning was never developed into a long-running
system — it existed for ~3 months (Jun–Sep 2024) purely to get `v4` strong
enough to be a credible datagen engine, then NNUE fully superseded it.

---

## 7. SPSA campaigns — a recurring, escalating tuning cadence

| Date | PR | Scale |
|---|---|---|
| 2024-07-16 | `407d2150` "LTC SPSA Tune (#13)" | unspecified |
| 2024-10-20 | `5348ae81` "LTC SPSA Tune (25k iterations, 100k games) (#97)" | 25k iter / 100k games |
| 2024-12-12 | `2ce10ccd` "LTC SPSA Tune (#144)" | unspecified |
| 2025-01-15 | `170f8c3f` "Fractional LMR & LTC SPSA Tune (9375 iterations, 75000 games) (#164)" | 9,375 iter / 75k games |
| 2025-03-25 | `84a3e833` "LTC SPSA Tune (9375 iterations, 75000 games) (#180)" | 9,375 iter / 75k games |
| 2025-06-16 | `276b58be` "LTC SPSA (98k games, 12k iterations) (#204)" | 12k iter / 98k games |
| 2025-08-05 | `8babfda9` "LTC SPSA (100k games, 12.5k iterations) (#220)" | 12.5k iter / 100k games |

Seven campaigns over ~13 months, spaced every 2–4 months — never a single
"tune once at the end" event. Campaigns cluster right after batches of new
search parameters land (e.g. the #164 tune follows "Fractional LMR" landing
in the *same* PR) — SPSA is used to re-settle the parameter space each time
enough new knobs accumulate, not as a one-shot finishing move. Game/iteration
counts trend upward over the project's life (from unspecified early runs to
a stable ~75–100k games / 10–12.5k iterations by 2025), consistent with the
dev's compute budget or confidence threshold growing alongside the engine's
maturity. Note also two disabled/incomplete SPSA references in the log
(`007f50d7` "SPSA automatic floating type conversions", `fdea7f88`
"Disabled SPSA tuning") showing the tuning *infrastructure* itself needed
iteration before campaigns could run smoothly — tooling is not free even for
an established convention.

---

## 8. The ecosystem effect: a solo dev standing on shared infrastructure

**No CI config, ever.** `git log --all --diff-filter=A --name-only | grep
-iE '\.ya?ml$|\.github'` returns nothing but build-directory noise
(`CMakeFiles/CMakeConfigureLog.yaml`) — no `.github/workflows/*.yml` was
ever committed. This is not an oversight: OpenBench is a standalone web
service (typically self-hosted or a shared community instance) that polls/
is pointed at a GitHub repo directly; it needs no in-repo workflow file. The
entire SPRT-gating apparatus that let a solo student compete with
corporate/team-backed engines was **adopted wholesale from the existing
Stockfish/OpenBench ecosystem**, not built. Evidence this is a genuine
external convention, not Integral-specific: `src/tests/bench_test.cc`'s
50-FEN bench suite is the same shared community bench-position list used
across many independently-developed OpenBench-tested engines (it is not
chess positions unique to Integral's own games or tuning).

**External contributors are a small, real signal, not noise.** Of 669
commits, exactly 10 (1.5%) are from three people who are not Aron
Petkovski:

- **Yoshie2000** (`a22294c2`, "Merge pull request #3 from Yoshie2000/corrhist",
  2024-06-17) — the author of another well-known engine (Weiss) contributed
  the correction-history feature directly, and this is the very commit that
  introduces the first `Bench:` trailer. A borrowed idea and a borrowed
  process convention landed in the same PR.
- **xu-shawn** (8 commits, 2024-05 → 2025-05: `99fcfe0c` UCI score-output
  fix, `0c598482` "4-ply Continuation History (#10)", `1246f30a`
  continuation-history tweak, `72ded2e4` `.editorconfig`, `22db2a9d`
  counter-move history bonus, `14f91f33` `MultiArray` refactor, `ceeeae44`
  aspiration-window tweak, `82e8bad7` "Template fused updates (#179)") — a
  cross-engine contributor known in the community for small, targeted
  drive-by PRs across *many* independently-developed engines, not just this
  one.
- **Jonathan Hallström** (`45e0e036`, "Finny table fused updates (#192)",
  2025-04-29) — one targeted performance PR.

**Techniques are explicitly credited from other engines in commit
prose**, e.g. `3418de6b` "Using Alexandria's soft limit calculations (43 elo
LTC, 68 elo STC)" (2024-06-05) names another open engine (Alexandria) as the
direct source of a time-management formula, with both LTC and STC Elo gains
reported for adopting it verbatim-in-spirit (reimplemented, not copied — the
same D27-style norm our own project already follows relative to
Stockfish/Reckless). This community habit — publish a technique + its
measured Elo, let anyone re-derive and adopt it — is precisely the "standing
on SF/open ideas then tuning" pattern named in the task; Integral is a
*consumer* of that norm as much as our own project is, and occasionally
(via `Bench:`/PR conventions, and via features other engines then borrow
back) a *contributor* to it.

The build system also encodes ecosystem awareness directly: the Makefile
exposes `native | vnni512 | avx512 | avx2_bmi2 | avx2 | sse41_popcnt`
targets (`Makefile:80-101`) — this granularity exists *because* OpenBench
workers are volunteer/heterogeneous machines with varying SIMD support, and
every worker's bench count must match regardless of which target it built.
This is the direct analog of our own D16 fast-build (`target-cpu=x86-64-v3`)
discipline, generalized further because Integral's testing pool is wider
than a single controlled machine.

---

## 9. Comparison: Integral vs. Stockfish vs. Reckless vs. us (process axis)

| Dimension | Stockfish | Reckless | Integral | Us |
|---|---|---|---|---|
| Team size | large, decades-old open contributor base | small (author unknown scale, ~AGPLv3, #2 SPCC at study time) | **~1 person**, 1.5% outside patches | 1 human + rotating LLM agents |
| Upfront design docs | none in-repo (fishtest wiki + mailing-list history instead) | none observed in our study | **none** — no `design/`, no `CONTRIBUTING.md` | `design/` suite, D-numbered decisions, phase gates (M1–M10) |
| Strength-testing tool | fishtest (bespoke, SF-specific) | (not established in our study) | **OpenBench** (shared, external, adopted not built) | `tools/sprt.sh`/`gauntlet.sh` (fastchess + ordo, self-hosted, not OB) |
| Commit-level gate | fishtest LTC/STC SPRT + bench sig | unclear from clone | `Bench:` trailer + PR number, near-100% by Era 3 | `Bench:` trailer (D24) — same convention, arrived at independently |
| Typical mature-era diff size | varies, often small | unclear | **1–3 files, <20 lines**, one idea per PR | similar target (D24/D32 batching is the deliberate exception) |
| Revert rate visible in log | low (fishtest gates before merge) | unclear | **1 in 669 (0.15%)** on `main` — survivorship bias | expect similarly low once D24/SPRT gating is fully load-bearing |
| Review process | many human reviewers, mailing-list discussion | unclear | **self**-review only (solo dev merges own PRs) | multi-agent A/B/C review loop (D33) — a *substitute* for having no second human |
| Net-iteration naming | `nn-<hash>.nnue` (content-hash, no marketing name) | unclear | **Zekrom v2–v7**, named + tagged | not yet established (P13) |
| Tuning cadence | continuous, distributed SPSA via fishtest | unclear | 7 named campaigns/13 months, escalating scale | planned (P14 `tune` cfg, doc 10 §9.1) — not yet run |
| License stance | GPLv3 | AGPLv3 | **none — all rights reserved** | (our own code; MIT/Apache not yet decided, out of scope here) |
| Public presence | CCRL/CEGT/TCEC, huge community | (smaller) | Lichess bot (`IntegralBot`), rating lists, TCEC | none yet |

The single largest structural difference from us: **Integral has no written
design layer at all.** Every "why" for 238 numbered PRs lives only in GitHub
PR titles/descriptions this clone cannot see, plus whatever OpenBench test
pages still exist. This works because the project has exactly one
continuously-in-context human who remembers all of it. It is not evidence
that design docs are unnecessary — see §11 item 6.

---

## What this means for us as a small project

Ordered roughly by how directly actionable each item is; several tie
straight into named phases already in `design/00-master-plan.md`.

1. **Seriously consider adopting OpenBench itself, not just its `Bench:`
   trailer convention.** We already independently converged on the same
   commit-trailer idea (D24) — Integral is strong external validation that
   the convention alone is right. What we're missing is the *infrastructure*
   OpenBench provides on top of the convention: a shared/self-hostable test
   queue, standardized SPRT bound presets, automatic bench-count validation
   across machines, and (if we ever open the repo) a zero-onboarding-cost
   path for outside contributors who already know the convention — exactly
   how xu-shawn and Yoshie2000 were able to land useful PRs into Integral
   with no bespoke documentation at all. This doesn't replace
   `tools/sprt.sh`; it's a candidate distribution/scaling layer for the same
   fastchess-based testing we already do. Tag against P14 (ongoing
   tuning/testing infra); effort is a self-hosting exercise, not a code
   change, so it's independent of engine phases.

2. **Expect our own accepted-idea rate to look small compared to the true
   attempt rate, and don't read that as a problem.** Integral's 669
   commits and near-zero revert count is the visible tip of an iceberg of
   OB-tested-and-rejected branches. For our own D32 batched-wave process:
   log rejected/parked ideas somewhere lightweight (we already do this
   informally — e.g. "item 9 IIR reverted... parked on
   `wip/p9-item9-iir-followpv`" per CLAUDE.md) so a future session doesn't
   silently re-propose and re-test something already known to regress.
   Treat that parking-lot practice as core process, not incidental cleanup.

3. **Re-tune on a cadence tied to structural milestones, not once at the
   end.** Integral's 7 SPSA campaigns (§7) landed roughly every 2–4 months,
   each triggered by a batch of new parameters accumulating (e.g. right
   after "Fractional LMR" landed). This directly informs the sizing/cadence
   for our own planned P14 `tune` work (doc 10 §9.1): don't treat SPSA as a
   single terminal step after P9–P11 land; schedule a pass after each major
   pruning-wave batch and again after each future NNUE net swap, sized in
   the same ballpark Integral converged on late in its life (~75–100k games
   / 10–12.5k iterations at LTC) as a first budget estimate for our own
   compute planning.

4. **Design the NNUE/datagen pipeline (P13, doc 15/16) explicitly for cheap
   re-runs, because that's what made Integral's net-iteration cadence work.**
   The expensive part of Integral's NNUE story was the ~6-month
   *architecture* search (Oct 2024–Apr 2025, only 4 arch-changing commits);
   once the architecture froze and the net got embedded directly in the
   binary (`576bef0b`, "Pre-process the network and use it directly from the
   binary"), retraining cadence dropped to ~13 days average, sometimes 1 day.
   For us: prioritize getting the full datagen→train→embed→bench loop
   working end-to-end *early and cheaply*, even at a weak/small first
   architecture, over polishing the architecture before ever running the
   loop once. (Note: Rust's `include_bytes!` gives us the binary-embedding
   step Integral needed a third-party `incbin` header for, for free — worth
   flagging as a small implementation-ease win when P13 gets there, though
   that belongs properly in a code-topic note, not this one.)

5. **Do not read "no design docs" as license to cut ours.** Integral's
   entire process discipline is carried by one person's continuous memory
   plus two borrowed mechanical conventions (`Bench:`, PR numbers) — it has
   no equivalent of `design/decisions.md` because it has never needed one:
   there's only ever been one mind holding the "why." Our situation is the
   opposite: our "developer" is a rotating cast of context-reset LLM agent
   sessions. `design/`, `decisions.md`, and `CLAUDE.md` are not overhead we
   could shed by copying Integral's leanness — they are the *substitute* for
   the continuous human memory that lets a solo human dev skip documentation
   entirely. If anything, Integral's success is evidence that *some* team
   needs a persistent "why" layer, and for a human solo dev that layer can
   live entirely in their own head; for us it has to live in files.

6. **Expect (and don't fight) an S-curve, and calibrate current
   expectations to where we are on it.** Integral's cheap, large wins
   (bundled TT/NMP/LMR/PVS/SEE, the +450 elo jump) all landed in its first
   ~6 months of *serious* work (Feb–Jul 2024) before NNUE; the following ~18
   months of this clone are visibly smaller, rarer gains gated by
   increasingly expensive LTC SPSA/SPRT. We are finishing P9–P11 (pruning
   waves + TM + lazy SMP) — the rough equivalent of Integral's month
   6–10 — so per design/00-master-plan's own framing, NNUE (P13) is the next
   big-lever phase, not another round of hand-tuned search-parameter
   polish. Integral's own history argues for moving toward P13 promptly once
   P9–P11 close out, rather than over-investing further in P9-style
   micro-gains whose Integral analogues were already down to single-digit
   Elo commits before NNUE ever landed.

7. **The multi-agent review loop (D33) is our functional replacement for
   "a solo dev who reviews their own PR before merging."** Integral has no
   analog to D33 because self-review by the same continuous mind that wrote
   the code is implicitly always happening — there is no session boundary to
   lose context across. Since our sessions *do* have boundaries, D33's
   parallel-persona review is doing necessary work Integral gets for free;
   this is a case where our process should be *heavier* than the reference
   project's, not lighter, precisely because our situation is structurally
   different (many short-lived agent contexts vs. one long-lived human one).

8. **A public strength signal (bot/rating-list presence) is worth planning
   for once P13 lands, not before.** Integral's Lichess bot and rating-list
   presence are cheap, high-visibility validation loops that a small project
   can set up unilaterally (no permission needed, unlike TCEC-style
   invitations). Not urgent at our current phase, but worth flagging in
   `design/00-master-plan.md`'s milestone ladder as a post-M9 (NNUE)
   consideration rather than leaving it unplanned.
