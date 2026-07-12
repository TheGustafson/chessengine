# Further study — coverage/accuracy audit findings and open threads

A follow-up pass over the Integral study (coverage + accuracy critic role,
2026-07-12): confirms subsystem/success-factor coverage, spot-checks three
notes against `integral_reference/integral`'s real source and git history,
corrects what didn't hold up, and records the residual small gaps. No new
subsystem note is warranted — the corpus already covers everything asked
for; this file is the audit trail plus a couple of standing cautions for
whoever extends the study next.

## 1. Coverage — confirmed complete

Cross-checking `NOTES.md`'s index against the ask: search (`02`, `03`),
NNUE architecture (`04`) + inference (`05`), SMP/NUMA (`06`), datagen/
training (`07`), movegen (`01`) are each a dedicated note, and the three
success-factor axes — git-history progression (`09`), ecosystem/force-
multipliers (`10`), NNUE data story (`11`) — are each a dedicated deep-dive
feeding the `00-success-factors.md` synthesis. `08` (timeman/UCI/tuning/
build) is bonus coverage beyond the ask, not a gap. Every path in NOTES.md's
"Quick map of Integral source" table was verified to exist in the vendored
clone (`src/chess/*`, `src/magics/*`, `src/engine/search/*`,
`src/engine/evaluation/nnue/*`, `shared/*`, `preprocess/*`, `src/data_gen/*`,
`src/tuner/spsa.h`, `src/engine/uci/uci.cc`, `src/main.cc`,
`src/engine/search/syzygy/*`, `third-party/{fathom,fmt,incbin}`,
`src/utils/*`, `src/tests/*` — 38 paths checked, 0 missing). All 13 note
files are linked from `NOTES.md`; all `design/NN-*.md` cross-refs in
`00-design-implications.md` (design/00, 03, 04, 05, 09, 10, 11, 12, 13, 14,
15, 16, 17) resolve to real files in this repo's `design/` directory. No
dangling links found.

## 2. Accuracy — corrections applied this pass

Spot-checking `06-smp-numa-scaling.md`, `07-datagen-training.md`, and
`09-dev-history-progression.md` against `git -C integral_reference/integral
log`/`show` turned up one significant and three minor misstatements, all now
corrected in place (not just recorded here):

1. **The NUMA saga's central claim was wrong** (`06` §6.2, most
   significant): the note previously said the buggy `e61b7c89` ("#198")
   commit "shipped and stayed live on `main` for exactly ten weeks." It
   didn't. `git merge-base --is-ancestor e61b7c89 main` fails, the commit
   never appears in `git log main`, and — the clincher — `main`'s merged
   PR-number sequence around that date runs `#197, #199, #200, ...` with
   `#198` cleanly missing. The commit is only reachable via two unrelated,
   never-merged branches (`origin/capture_text`, `origin/cfix`). Corrected
   reading: PR #198 was a false start, apparently caught before merge (by
   review or local testing — not evidenced which), not a bug that ran live
   in production. This also fixes a second, independent error: the note's
   own §6/§6.5 headers claimed "fifteen months" for the saga's span while
   §1 correctly said "three months" for the *same* span — an internal
   self-contradiction. The real span (first false start to final fix) is
   ~2.5 months; §1's figure was the one that was right.
2. **The "3 reverts in 669 commits" survivorship-bias stat was wrong** (`09`
   §4.4, propagated into its own comparison table and into
   `NOTES.md`'s summary line): 2 of the 3 cited hashes (`55998fd5`,
   `3eebba4a`) are not reachable from `main` (they live on the abandoned
   `smarter_permuting` and `data_generation` branches) — the original count
   conflated a `git log --all --grep=revert` search with a `main`-only
   claim. Corrected: **1** revert on `main` (`03efc83f`, 2024-03-20,
   Era 1) — 0.15%, not 0.4%. Ironically the 2 miscounted commits are
   textbook examples of the note's *own* argument two sentences later
   ("a branch that fails is simply abandoned, no revert commit needed") —
   the correction strengthens the point rather than undermining it.
3. **The "NNUE landed the same day / ~1 hour after the v4 HCE tag" claim was
   wrong** (`00-success-factors.md` items #2/#4, `09` §3 table): the v4 tag
   (`d4359007`) is 2024-09-01 00:12; the NNUE commit that actually merged
   (`99b7b22e`, "#52") is 2024-09-06 18:43 — 5 days and ~18.5 hours apart,
   not "the same day." Note `11-nnue-success-story.md` §1 already had this
   exactly right ("Five days later...") and had already done the harder
   work of reconciling it with the earlier off-`main` `cd3f7acb` prototype
   (2024-08-27, pre-dates the v4 tag by 5 days, confirmed not an ancestor
   of `main`) — the error was isolated to the two files that hadn't
   cross-checked against `11`'s own findings. Both are now fixed to match.
4. **A garbled commit hash** (`07-datagen-training.md` §8.3): `1840658` is
   not a valid object in the repo at all. Corrected to `3f15ba6b`, the
   actual tip of `main` at the time of this study (2026-04-02) — which
   independently checks out: its subject line ("Play quiet moves in QS...")
   is the exact commit `07` quotes two sentences later as one of the
   "search side kept shipping gains" examples.

**What did check out, for the record** (so a future pass doesn't re-verify
these from scratch): the net architecture string `(768×12 (Factorized) →
1536)×2 → (16 → 32 → 1)×8` matches `README.md` and `shared/nnue/
definitions.h` exactly; the output-bucket formula `(popcount−2)/4` matches
`accumulator.h:395-397` verbatim; the 7 named LTC SPSA campaigns
(`407d2150`, `5348ae81`, `2ce10ccd`, `170f8c3f`, `84a3e833`, `276b58be`,
`8babfda9`) are all real, on `main`, in that order; the `own_hce_data`/
`lichess_hce_data` branch pair cited for the datagen-strategy A/B both
exist; the 659/669-solo-commit split and the ~947-branch count both check
out against `git shortlog -sn` / `git branch -a | wc -l`; the
`BUILD_HAS_AVX512VNNI`-defined-but-never-consumed claim (`05` §8) is exactly
right down to the cited line numbers; the Syzygy `PopCount() <= 7` gate and
every `data_gen.cc` line citation checked (`GameLoop` 211-319, `RandomSeed`
215, `ResizeHash(16)` 230) landed on the exact cited lines.

## 3. A standing caution for future git-archaeology on this repo

Every corrected item above traces to the same root cause: a claim built from
`git log --all` (or a cached/earlier clone state) presented as if it
described `main`'s history. Concrete discipline for next time: when citing a
commit as evidence for "what shipped" or "what `main`'s history contains,"
run `git merge-base --is-ancestor <hash> main` (or `git log --oneline main
| grep <hash>`) before writing the citation down, not just `git show
<hash>` (which resolves any object in the database, reachable or not, and
will happily "confirm" a commit that never merged). Cross-checking a
merged-PR-number sequence for gaps (as done for `#198` above) is a good
second signal when a commit *looks* like a normal squash-merged PR but
ancestry says otherwise.

## 4. Remaining small gaps (not worth a new note, just flagging)

- No dedicated note covers Integral's Syzygy/Fathom integration end-to-end;
  it's touched tangentially in `02` (search-side probe gate) and `07`
  (datagen tablebase use) but there's no single place that walks
  `third-party/fathom`'s vendoring or the `SyzygyPath`/`SyzygyProbeDepth`
  UCI surface as its own topic. Low priority — Syzygy isn't on our critical
  path until post-P13, and the two mentions that exist are both accurate
  (verified above).
- `04-nnue-architecture.md`'s own architecture-evolution table (§8) lists
  `cd3f7acb` without repeating the "not an ancestor of `main`" caveat that
  `11-nnue-success-story.md` §1 spells out — not wrong (it never claims
  `cd3f7acb` merged; the very next row explicitly says "`99b7b22e` | NNUE
  merged to main"), just less explicit than `11`'s treatment. Not corrected
  here to keep this pass surgical; a reader who wants the full reconciliation
  should follow the pointer `11` §1 already gives to `04` §8.
