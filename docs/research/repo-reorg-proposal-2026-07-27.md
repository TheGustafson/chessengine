# Repository reorganization proposal (2026-07-27)

Status: **PROPOSAL — nothing in this document has been executed.** The only
file this work created is this one.

Commissioned because "it is HARD for a human to jump in here and figure out
what is what and where to look." The test applied throughout is *human
comprehension*, not tidiness: can a competent engineer landing cold find what
they need and build an accurate mental model quickly?

Method: read the design suite, `CLAUDE.md`, `TESTING.md`, the real source
modules, the ledger, and a broad sample of `docs/`, `tools/`, and `jax/`;
three parallel read-only inventory passes over `tools/` (330 tracked files),
`docs/` (113), and the Python/JAX stack. No file was moved, renamed, edited,
or deleted. **No cargo build was run** — the box was memory-constrained by a
live GPU training run, so every claim below comes from reading, `git grep`,
`git log`, and `systemctl`. Where a claim genuinely needs a build to confirm,
it is written into §6 as a verification step rather than asserted.

---

## 0. Headline

**The problem is not documentation quality — it is topology and truth
maintenance.**

The per-module doc comments in `src/` are, module for module, better than most
production Rust. `design/` is a genuinely reviewed 19-document contract suite.
`ledger/README.md`, `docs/research/net-naming-guide.md`, `tools/check.sh` and
`tools/nnue/RUNBOOK.md` are models of how to write a navigational document.
The repository already knows how to do this well.

What has failed is everything *between* the documents:

1. **There is no entry point.** No `README.md` exists at the repo root. The
   de-facto entry point is `CLAUDE.md`, an agent-instruction file whose status
   section is 40 lines of undefined jargon.
2. **The navigation chains are broken and point at superseded truth.**
   37 dangling `design/*.md` references, 19 dangling `tools/*` references,
   3 dangling `docs/*.md` references, 2 dangling `.agents/rules/` links. The
   suite index sends a newcomer to a doc that CLAUDE.md declares superseded.
3. **The invariants are asserted in the wrong places and are stale.** The
   pinned depth-6 bench is **385329**. That number appears exactly **once**
   in the whole tracked tree — in CLAUDE.md prose. The superseded pin
   **548978** appears **12 times**, including in three source files' doc
   comments and three design docs, each stating it as current.
4. **The live work is invisible to git.** The currently-running D48X training
   campaign — its launcher, its watcher, and its architecture config — lives
   in `tools/nnue/runs/`, which `.gitignore` excludes. 85 files on disk, 9
   tracked (force-added once, all belonging to a campaign ten letters stale).
   Eight `ledger/*.jsonl` regeneration recipes already point at files that are
   not in git.
5. **30% of `src/` is not covered by the merge gate.** `tools/check.sh` and CI
   run `cargo test` / `cargo clippy --all-targets` with default features only.
   Roughly 15,600 lines behind `sota`, `datagen`, `policy`, `search-control`
   and `tune` — including every line the match binaries actually execute — are
   never compiled, linted, or tested by the gate.

The reorganization that follows is therefore weighted heavily toward
**adding navigation and repairing truth**, and only lightly toward moving
files. That is not timidity. It is what the reference graph demands: 261 of
680 tracked files (38%) contain a `tools/` path string, 174 contain
`tools/nnue/`, 210 contain `data/nnue`, and the append-only ledger cites tool
paths in 82 rows that must not be rewritten. In this repository, moving a
directory is expensive and adding an index is free.

---

## 1. What a newcomer actually experiences

This section is primary data: these are the places **I** got lost, in the
order I hit them.

**1.1 There is no front door.** `ls` at the root shows 32 entries and no
`README.md`. GitHub would render nothing. The reader's next guess is
`CLAUDE.md`, which opens with project instructions for an AI agent. It is a
good document *for that job*; it is not an orientation document.

**1.2 The status paragraph is unreadable without the 7,000-line log.**
`CLAUDE.md`'s CURRENT block reads, verbatim: *"the 4-phase candidate campaign
collapsed by user ruling to: fn pentagon (DONE — V−Spost −35, Spost = SF16
parity at fn, all exp-0149..0153) + ET V-only vs S/SF16/SF18…"* Nowhere in
the repo is there a glossary defining `fn` (fixed-node), `ET` (equal-time),
`V`, `S`, `S-post`, `sb` (superbatch), or the D47/D48 letter series. I had to
reconstruct these from context across several thousand lines.

**1.3 The suite index routes to the wrong live-state doc.**
`design/README.md` (lines 43–46) says the current research checkpoint lives in
`docs/research/archon-execution-plan.md` and continuation state in
`docs/research/policy-automl-state.md` "(legacy filename retained)".
`archon-execution-plan.md` is dated 2026-07-14 and itself says to read
`policy-automl-state.md`. But `CLAUDE.md` says `policy-automl-state.md` is the
*predecessor* and `value-foundation-state.md` is authoritative. Three hops to
the wrong file.

**1.4 The decision record has run out of numbers.** `design/decisions.md`
carries D1–D47. Its own header, `design/README.md`, and `CLAUDE.md` all say
"D1–D44". `src/lib.rs`'s crate doc says "D1–D27". Meanwhile **D48W** and
**D48X** appear in `Cargo.toml` feature comments, `CLAUDE.md`, `src/eval/nnue/
sota_head.rs`, and 37 times in the research log — but there is **no D48 entry
in `decisions.md`**. The process rule "decisions change there first; docs
follow; code follows docs" is currently inverted for the architecture the
engine is training right now.

**1.5 I could not find how to build a playable engine.** `cargo run` produces
a PSQT engine (~2720 Elo). The real engine needs `--features sota,arch-d47v`,
a PGO build via `tools/build_pgo.sh`, and an `EvalFile` pointing at a `.cenn`
under gitignored `data/`. No document states which net is current. I found the
answer — `data/nnue/nets/screens/d47v-a4b2-cal400.cenn` — by reading the
`ExecStart` line of a running systemd unit.

**1.6 Five live net formats, no index.** `src/eval/nnue/mod.rs` routes on magic
bytes across CENN v1 (`Network`), CEN2 (`SotaNetwork`), CENN-v3
(`simple_halfka`), CENN-v4 (`protected_halfka`), plus a `.cpol` policy sidecar
and two env-var-loaded sidecars (`D40_UNC_HEAD`, `D40_WDL_HEAD`). No single
document lists them or says which are live.

**1.7 Nineteen undocumented runtime switches.** `src/` reads 19 distinct
environment variables that change engine behaviour
(`ARCHON_SMALLNET`, `ARCHON_LARGENET`, `ARCHON_LARGENET_SIGMA`,
`ARCHON_UNC_RFP_C1`, `ARCHON_UNC_TM_C2`, `ARCHON_WDL_TM_C3`,
`ARCHON_UNC_EMA_SHIFT`, `ARCHON_SMALLNET_THRESH`, `ARCHON_SMALLNET_MAT_GATE`,
`POLICY_NET`, `POLICY_TRUNK`, `POLICY_ROOT`, `POLICY_W`, `POLICY_CAPTURE_W`,
`POLICY_CAPTURES`, `POLICY_MIN_DEPTH`, `POLICY_MAX_PLY`, `CPL_ORD_W`,
`CPL_ORD_MIN_DEPTH`), plus `D40_UNC_HEAD`/`D40_WDL_HEAD`/`ARCHON_EMBED_NET`.
**Zero** are mentioned in `CLAUDE.md`, `TESTING.md`, or any `design/` doc.

**1.8 Module docs describing a different engine.**
`src/eval/nnue/mod.rs` — 493 lines, the routing hub for every net format —
opens with: *"**Scope (P13 step 1).** This is the inference skeleton… It is
purely additive — the active evaluation still returns the PSQT value."* That
was true three architectural generations ago. The same file cites bench
548978 as current.

**1.9 Two directories named `tools`.** `tools/` and `jax/tools/`. The
documentation already conflates them: 19 markdown references to `tools/<x>.py`
resolve only as `jax/tools/<x>.py`, and one **ledger regeneration recipe**
cites `tools/pump_train.py`, which does not exist (it is
`jax/tools/pump_train.py`).

**1.10 `tools/ledger/` contains no ledger; `ledger/` contains no tools.**
`tools/ledger/ledger.py` is a 621-line CLI operating on the top-level
`ledger/*.jsonl`. Reasonable once known, a trap on first contact.

**1.11 Root-level items that look like project structure but are not.**
`config.json` (fastchess runtime state, gitignored, maximally generic name),
`vgcore.534151` (116 MB core dump, untracked and **not** gitignored),
`d40_feat_ckpt/` (empty), `patches/` (one 33 KB patch that no longer applies —
verified with `git apply --check`), `perf-artifacts/` and `perf-artifacts-6/`
(38 tracked raw asm/callgrind dumps) while the in-flight study writes
`docs/research/perf-artifacts-7/`, and five `target-pgo-*` build trees
including one whose directory name contains a comma.

**1.12 `GEMINI.md` is a stale fork of `CLAUDE.md`.** 310 vs 370 lines, last
touched 2026-07-15, diverging by ~122 lines. It still names
`policy-automl-state.md` as the live research doc. A newcomer who opens it
first gets a twelve-day-old picture with no warning.

**1.13 The research directory has no map.** 70 markdown files, five
simultaneous naming conventions, and two *different* numbered series covering
the same lineage: `profiling-study-{2026-07-18, 2026-07-19, 3-creative}` then
`perf-study-{4,5,6}`, then a *new* chain starting at
`threat-channel-perf-study.md` (unnumbered, but called "study 1" by its
successor) → `perf-study-2-broad` → `perf-study-3-beyond`. Alphabetically,
`perf-study-2` and `perf-study-6` sit adjacent and belong to different chains.

**1.14 The live-state document contradicts itself.**
`docs/research/value-foundation-state.md` is 7,001 lines with **four** blocks
titled "TAKEOVER CHECKPOINT" (lines 686, 2015, 2090, 4658). Two of them
simultaneously say "READ THIS FIRST"; the one at line 686 defers to one 1,300
lines *below* it. At line 948 a banner reads *"⛔⛔ THE STOP RECOMMENDATION IS
WITHDRAWN … READ THIS BEFORE ANY 'recommend stop' TEXT BELOW — all of it is
SUPERSEDED"*. A reader going top-to-bottom will act on retracted advice unless
they already know the file's internal geometry. Ordering is neither
chronological nor reverse-chronological.

**1.15 The live campaign is not in git.** I discovered the running D48X job by
listing systemd units. Its launcher (`tools/nnue/runs/d48x_launch.sh`), watcher
(`d48x_watch.sh`), pruner (`d48x_prune.sh`) and architecture config
(`d48x_1280.json`) are all in a **gitignored** directory. So are the
D47D/F/H/J/K/M/P/V configs and every `joint_*`/`union_*`/`stage*` recipe.
The nine files that *are* tracked there were force-added in commit `5e49d66`
("tmpfs originals die on reboot") and all belong to the d47c campaign.

**1.16 Thirty-one tracked files hardcode a path from a machine that no longer
exists.** `/root/projects/learnrust/chessengine` — the pre-migration container
root — is baked into `checkpoint_dir` fields in 18 `tools/nnue/*.sh` campaign
scripts, 8 `tools/nnue/configs/*.json`, `tools/nnue/TRAINING.md`, and even one
`ledger/nets.jsonl` row. The box migrated to `/home/nick/projects/chessengine`
on 2026-07-19. These scripts are already broken; no reorg is needed to break
them further.

---

## 2. Structural findings, with numbers

### 2.1 Code

| File | Total | Inline tests | Real code |
|---|---:|---:|---:|
| `src/search/mod.rs` | 9,137 | **4,191** | 4,946 |
| `src/board.rs` | 3,328 | 1,536 | 1,792 |
| `src/uci.rs` | 2,734 | 1,040 | 1,694 |
| `src/bin/obs_tool.rs` | 2,354 | 520 | 1,834 |
| `src/movegen.rs` | 2,312 | 1,649 | 663 |
| `src/eval/nnue/sota_simd.rs` | 1,889 | 352 | 1,537 |

`src/` is 51,186 lines total, of which roughly 16,000 are inline `mod tests`.
Inside `search/mod.rs`'s 4,946 lines of code, `impl Worker` spans lines
1021–4936 and a single function — `fn search<NT: NodeType>` — spans lines
2223–4095 (**1,872 lines**). That last one mirrors Stockfish's own `search()`
and is defensible; the 4,191-line test module underneath it is not.

Five of 54 source files lack a `//!` module header:
`src/board.rs` (the largest non-search module and design/01's designated
"growth center"), `src/types.rs`, `src/bitboard.rs`, `src/moves.rs`,
`src/fen.rs`. These are the five oldest modules, pre-dating the convention.

Module placement that has drifted from the concept map:
- `search_control.rs`, `search_intervention.rs`, `search_trace.rs` — three
  crate-root siblings, all gated on one feature, all one subsystem.
- `policy.rs` at the crate root, while `eval/nnue/policy_head.rs` and
  `eval/nnue/policy_vocab.rs` sit two levels down. Three "policy" modules in
  two places.
- `eval/nnue/` holds 17 files, 9 prefixed `sota_`. "SOTA" is a value judgment
  that has already aged: the `sota` feature now carries four *different*
  architectures selected by `arch-*` features, the newest of which
  (`arch-d48x`) is two generations past the net the prefix was coined for.

Feature-flag surface: 15 Cargo features, of which 5 (`nnue-l1-512/1024/2048/
4096` + default) are one mutually-exclusive width group and 4 (`arch-d47v`,
`arch-d48w`, `arch-d48x`, `arch-hka2048`) are another. The width group has a
`compile_error!` guard in `eval/nnue/mod.rs:128`; **the `arch-*` group has
none**, despite Cargo.toml declaring the same exclusivity in prose. The
dimension-selection `#[cfg]` ladders are split across `eval/nnue/mod.rs`
(55 lines) and `eval/nnue/sota_head.rs` (40 lines).

Five separate `Cargo.toml` files, **no `[workspace]`**: the root crate,
`pymask/` (a live PyO3 extension the running trainer calls every batch),
`tools/nnue/sfbinpack-audit/`, `tools/nnue/bf-convert/` (whose `[[bin]] path`
points *up* at `../bf_convert.rs`), and `jax/rust-loader-design/` (a design
sketch containing `todo!()` stubs that would not compile — but which looks
buildable because it has a manifest).

### 2.2 The merge gate's blind spot

`tools/check.sh` runs `cargo fmt --check`, `cargo clippy --all-targets -D
warnings`, `cargo test`, `cargo build --release`, the perft tier, and the
bench fingerprint. `.github/workflows/ci.yml` runs the same three cargo
commands. **None passes `--features`.**

Consequences, all verified by reading `#[cfg]` attributes and `tests/*.rs`
headers:
- `tests/cenn_v2_smoke.rs`, `tests/eval_file_reload.rs`, `tests/obs_capture.rs`
  are `#![cfg(feature = ...)]`-gated and compile to nothing under the gate.
- Roughly **15,600 lines** — all `sota_*` (8,478), `obs/` (2,433),
  `policy.rs` (554), the search-control trio (896), and the two feature-gated
  bins (3,248) — are never compiled, clippy-linted, or tested by the gate.
- This is the code every match and SPRT binary runs.

Separately, `check.sh` **prints** the bench number for the trailer but does not
**assert** it against the pin. Nothing in the repo mechanically enforces 385329.

### 2.3 Docs

113 tracked files. `docs/research/` is 70 markdown files ≈ 28,600 lines, of
which two files (`value-foundation-state.md` 7,001 + `policy-automl-state.md`
2,785) are 34%. By character: ~36 finished studies, ~20 plans, ~9 specs,
~4 live-state, ~4 runbooks, ~3 explicitly dead.

Reference cost of moving the heavily-cited docs (these are **not** docs-only
edits):

| Doc | Occurrences | Files | Referenced from code? |
|---|---:|---:|---|
| `value-foundation-state.md` | 42 | 32 | **yes** — `src/bin/obs_tool.rs`, `src/search/params.rs`, `ledger/experiments.jsonl`, 14 `tools/nnue/*` |
| `data-recipe-contract.md` | 9 | 8 | **yes** — `jax/archon_jax/data.py`, `tools/nnue/bullet/d40_features.rs`, 2 more |
| `design/decisions.md` | 66 | 34 | yes (doc comments throughout `src/`) |
| `tools/check.sh` | 95 | 43 | — |
| `CLAUDE.md` | 81 | 34 | — |

`docs/learn/` (16 chapters, 4,872 lines) has **zero** inbound references from
outside `docs/` — the only genuinely free-to-move doc tree. It is a good
course, written 2026-07-17 and untouched since; chapters 01–14 are conceptual
and still accurate, chapter 15 ("the research frontier") describes a
pre-D47 world. `docs/learn/video/` is not documentation at all: it is a
Manim + TTS rendering pipeline (7 Python modules, 2 shell scripts, a
`PLAN.md`) with one pilot scene built out of fifteen.

`docs/research/artifacts/` — 11 GCP job/plan JSONs from the July LC0 campaign.
Exactly **one** inbound reference exists anywhere, and it is from
`policy-automl-state.md`, which is itself frozen. Orphaned audit records.

`docs/perf-baseline.md` — last touched 2026-07-11, still accurate for what it
covers (criterion attack/movegen numbers) but never extended into the NNUE era;
a parallel benchmark-recording tradition now lives in the `perf-study-*` docs.

`docs/strength-log.md` — the D25 audit trail, *contractually* appended by
`sprt.sh`/`gauntlet.sh`/`tactics.sh`. It has a **15-day gap** (2026-07-12 →
2026-07-27) during which 164 experiments were ledgered. The stated process
stopped matching reality and nobody noticed.

### 2.4 `tools/` — path entanglement is the binding constraint

330 tracked files. `tools/nnue/` alone is 252, of which ~186 sit flat at one
level. Functional breakdown of those 186:

- **~65–70 one-shot campaign shell scripts** (`e1_*`, `e3_*`, `e4_*`, `d40_*`,
  `d47*`, `uho_*`, `stage2_*`, `probe_*`, `gate_*`, `exp_unc_*`, `prod_*`).
  All share the same shape: `cd "$(dirname "$0")/../.."`, poll a sibling's
  `DONE` file, run one step, `touch` its own `DONE`. Each is named after one
  concluded experiment.
- **~10 durable Python tools** (`netpack.py`, `nnue_ref.py`, `train_run.py`,
  `eval_match.py`, `value_matrix.py`, `dashboard.py`, `tb_shim.py`,
  `tb_system.py`, `policy_sweep.py`, `d40_campaign.py`).
- **~45–50 data-pipeline / relabel tools**.
- **~30–35 audit/diagnostic tools**, including a self-contained 22-file
  counterfactual-attribution cluster all created 2026-07-15 and never touched
  again.
- 30 config JSONs, of which `RUNBOOK.md` designates 2 as templates; the other
  28 are untouched since 2026-07-19 and at least 10 name nets absent from the
  58 rows of `ledger/nets.jsonl`.

Entanglement, measured:

| String | Tracked files containing it |
|---|---:|
| `tools/` (any) | **261 of 680 (38%)** |
| `tools/nnue/` | 174 |
| `data/nnue` | 210 |
| `engines/bin` | 97 |
| `target/release/chessengine` | 42 |
| `/root/projects/learnrust` (stale) | 31 |

Plus: 82 `ledger/*.jsonl` rows cite a `tools/` path in a regeneration recipe,
and eight of those already resolve to gitignored files. Plus eight
same-directory bare Python imports inside `tools/nnue/` (`eval_match.py →
policy_sweep.py`, `collect_search_trace.py → root_move_oracle.py`, and six
more) that break the moment the flat directory is subdivided.

**This is the single most important constraint in the proposal: `tools/nnue/`
cannot be bulk-moved or subdivided cheaply.** Any plan that renames it is
wrong. The affordable intervention is an index, not a move.

### 2.5 The Python/JAX side

`jax/archon_jax/` (11 modules, ~2,679 lines) is genuinely cohesive: `spec.py`
owns the constants and config dataclasses, `features.py` the numpy
reimplementation of the engine's feature math, `model.py` the forward graph,
`ft_kernel.py` the Pallas GPU kernel, `checkpoint.py` bullet-compatible IO.
It is the healthiest tree in the repo. `train.py`'s `make_update_step` carries
five independently-toggled features in one closure but is not tangled.

The problems are around it, not in it:
- Of 14 `jax/tools/` scripts, **2 are durable** (`pump_train.py` — the live
  trainer driver; `export_warm_quant.py`) and 12 are one-shot parity proofs
  and probes from the July 22–23 migration.
- **`jax/pyproject.toml` is documentation, not mechanism.** Nothing runs
  `uv sync --project jax`. Every launcher instead uses
  `uv run --no-project --with numpy==2.5.1 --with "jax[cuda12]==0.11.0"
  --with optax==0.2.8`, with the pins hand-copied as literal strings into a
  dozen shell scripts. `archon_jax` is never installed; 13 of 14 tools do
  `sys.path.insert(0, …parents[1])`.
- The root `pyproject.toml` + `uv.lock` + `.venv` (Python 3.13, numpy /
  tensorflow / python-chess) is a *different, disjoint* environment used by
  `netpack.py` and `dashboard.py`. Two pyprojects, one lockfile, one venv, and
  no document explaining which is which.
- `jax/rust-loader-design/` is a `todo!()`-stubbed sketch with a Cargo.toml.
- `pymask/` is real, live, load-bearing (the running D48X job calls
  `archon_mask.legal_move_ids` every batch) — and is built by hand with a
  manual `.so` copy into `jax/`, outside any workspace.
- **JAX is the current trainer; `tools/nnue/bullet/` is historical.** The
  handoff happened 2026-07-23. Nothing labels it.

---

## 3. Recommended target structure

Deliberately conservative on moves. Paths marked **NEW** are additions;
**MOVE** entries are the only relocations proposed, and each is justified in §4.

```
chessengine/
├── README.md                      NEW  ← the human front door (see §4, R1)
├── CLAUDE.md                           slimmed: conventions + pointers only
├── GEMINI.md                           → 3-line pointer at CLAUDE.md (R4)
├── TESTING.md                          (keep; small corrections)
├── Cargo.toml  Cargo.lock
├── .gitignore                          + vgcore.*, − tools/nnue/runs/ (R2)
│
├── design/                             UNCHANGED paths (66 refs to decisions.md)
│   ├── README.md                       fix the pointer chain (R10)
│   ├── decisions.md                    add D48; fix the "D1–D44" header
│   ├── 00-master-plan.md … 19-*.md
│   └── GLOSSARY.md                NEW  fn / ET / sb / V / S / D-letter series
│
├── docs/
│   ├── MAP.md                     NEW  "where does X live" — one table
│   ├── ENVIRONMENT.md             NEW  the 19 env switches + 5 net formats
│   ├── perf-baseline.md
│   ├── strength-log.md
│   ├── learn/
│   │   └── 00-README.md … 15-*.md      (video/ moves out — R19)
│   └── research/
│       ├── README.md              NEW  the classification index (R3)
│       ├── STATE.md               NEW  ≤500 lines, overwritten not appended
│       ├── value-foundation-state.md   STAYS (32 inbound, 16 from code)
│       │                               → becomes a pointer + current arc only
│       ├── data-recipe-contract.md     STAYS (code-referenced)
│       ├── log/                   NEW  MOVE the historical arcs out of the 7k log
│       │   ├── value-foundation-2026-07-13-to-07-19.md
│       │   └── value-foundation-2026-07-20-to-07-27.md
│       ├── perf/                  NEW  MOVE the perf/profiling chain + artifacts
│       │   ├── README.md               the true chronology of the two series
│       │   ├── artifacts-2/            MOVE ← /perf-artifacts/
│       │   ├── artifacts-6/            MOVE ← /perf-artifacts-6/
│       │   └── artifacts-7/            (the in-flight study writes here)
│       ├── archive/               NEW  MOVE the 3 DEAD + 11 orphaned artifacts/
│       └── (everything else stays put)
│
├── src/
│   ├── lib.rs                          fix the stale "D1–D27" crate doc
│   ├── search/
│   │   ├── mod.rs                      4,946 lines of code (was 9,137)
│   │   ├── tests.rs               NEW  MOVE ← the 4,191-line inline mod tests
│   │   ├── margins.rs             NEW  MOVE ← the pure pruning-margin fns
│   │   ├── timectl.rs             NEW  MOVE ← the 7 TM helper fns
│   │   ├── report.rs              NEW  MOVE ← SearchReport / UciSink / election
│   │   ├── params.rs  stack.rs  rootmoves.rs      UNCHANGED
│   ├── search_control/            NEW dir, same public root path
│   │   ├── mod.rs                      MOVE ← search_control.rs
│   │   ├── intervention.rs             MOVE ← search_intervention.rs
│   │   └── trace.rs                    MOVE ← search_trace.rs
│   ├── eval/nnue/
│   │   ├── arch.rs                NEW  ALL dimension #[cfg] ladders, one place
│   │   ├── mod.rs                      routing + registry only; doc rewritten
│   │   └── (sota_* keep their names for now — see R20)
│   └── (everything else UNCHANGED — see the do-not-touch list, §7)
│
├── tools/                              PATHS UNCHANGED (261 files reference them)
│   ├── README.md                  NEW  the 22 durable tools vs the rest (R9)
│   └── nnue/
│       ├── README.md              NEW  bucket map of the 252 files
│       └── runs/                       UN-IGNORED and TRACKED (R2)
│
├── jax/                                UNCHANGED
│   ├── README.md                  NEW  the real invocation chain + env story
│   └── rust-loader-design/             → docs/research/ or clearly marked
│
├── ledger/                             UNCHANGED — append-only, do not rewrite
│   └── moved-paths.md             NEW  old→new map for any tool that moves
│
└── stockfish_reference/  reckless_reference/  integral_reference/   UNCHANGED
```

Not moved, deliberately: `tools/nnue/` contents, `data/`, `engines/`,
`.archon-state/`, `src/movegen.rs`, `src/board.rs`, `src/zobrist.rs`,
`src/attacks.rs`, `src/search/params.rs`, `tests/data/`. Reasons in §7.

---

## 4. Ranked recommendations

Ranked by **human-comprehension benefit per unit of risk**. Tier A is
cheap-and-safe and can be taken whole or piecemeal in any order. Tier B is
mechanical but needs the bench pin verified. Tier C is ambitious and should
only be attempted after Tier A lands and the training/perf work quiesces.

Risk key: **none** = cannot affect the binary; **low** = mechanical, fully
covered by an existing gate; **medium** = needs a gate that does not exist yet
or touches string references; **high** = touches D24-guarded code or live jobs.

---

### Tier A — cheap and safe (no engine risk whatsoever)

**R1. Write `README.md` at the repo root.** ~80 lines.
*Problem:* no entry point exists; the front door is an agent-instruction file.
*Content:* what ARCHON is and its current strength; the four-line quick start
(`cargo build --release`, `cargo test`, `bash tools/check.sh`, `cargo run`);
**how to build the real engine** (`bash tools/build_pgo.sh sota,arch-d47v
archon-play <net>.cenn`) and where the current net is; a 12-row "where things
live" table; the four documents that matter (`design/decisions.md`,
`design/00-master-plan.md`, `docs/research/STATE.md`, `TESTING.md`) with one
line each on when to read them; and an explicit "this repo has 773 GB of
untracked `data/` and 1.3 GB of untracked `engines/` — see §X" note.
*Work:* 1–2 hours. *Risk:* **none**. *Verify:* a reader who has never seen the
repo can name the current net and build a playing binary.

**R2. Un-ignore `tools/nnue/runs/` and track the campaign scripts + configs.**
*Problem:* the live D48X campaign's launcher, watcher and architecture config
are invisible to git; 8 ledger regen recipes already point into this directory;
the 9 files that *are* tracked belong to a ten-campaigns-stale experiment;
`.gitignore`'s comment for the rule ("NNUE training dashboard tfevents") no
longer describes what is in there.
*Work:* replace `tools/nnue/runs/` in `.gitignore` with a narrower rule
(`tools/nnue/runs/**/events.out.tfevents.*`, `tools/nnue/runs/*.log`), then
`git add tools/nnue/runs/*.sh tools/nnue/runs/*.json tools/nnue/runs/*.py`
(~76 files, 364 KB total). Optionally split into `runs/active/` and
`runs/archive/` in the same commit — but note the systemd units currently
running reference `tools/nnue/runs/d48x_watch.sh` and `d48x_prune.sh` by path,
so **those specific files must not move while the units are alive**.
*Risk:* **none to the engine**; the only hazard is moving a file a live unit
reads, which the constraint above avoids. *Verify:* `git status` clean;
`ledger.py verify` no longer reports missing recipe targets for those eight
rows; the running units keep logging.

**R3. Add `docs/research/README.md` — the classification index.**
*Problem:* 70 files, five naming conventions, two colliding numbered series,
no way to tell a binding spec from a concluded study.
*Content:* one table, one row per file: filename, class
(LIVE / SPEC / STUDY / PLAN / RUNBOOK / ARCHIVED), one-line description, date.
Plus a short section naming the four clusters explicitly — the data-strategy
lineage (`data-quality-strategy` → `data-schema-and-storage` →
`post-relabel-data-strategy` → `data-quality-curriculum` → **`data-recipe-
contract` = the one that binds**), the four-part learned-search study chain,
the `sf18-*` pair, and above all **the true chronology of the perf/profiling
series**, which alphabetical order actively misrepresents.
The classification for all 70 files already exists in the inventory that
produced this proposal; it can be pasted in.
*Work:* 2 hours. *Risk:* **none**.

**R4. Retire `GEMINI.md`.** Replace its 310 lines with a three-line pointer to
`CLAUDE.md` (or make it a symlink, if the tooling that reads it follows links —
**unknown, check before symlinking**). *Risk:* **none** to the engine; verify
with whichever agent tool reads it. Consider `AGENTS.md` as the shared name.

**R5. Root hygiene.** Delete `vgcore.534151` (116 MB, untracked, not ignored)
and add `vgcore.*` to `.gitignore`; remove the empty `d40_feat_ckpt/`; move
`patches/1536-arch-bet-full.patch` to `docs/research/archive/` or delete it
(**verified dead**: `git apply --check` fails on three files, and the 1536
arch it encodes was superseded by D48X at 1280 on 2026-07-27; git history
retains it either way). *Risk:* **none**.

**R6. Sweep the dangling references.** 37 `design/*.md`
(`05-movegen-perft.md` ×12, `07-search.md` ×7, `09-movepick-history.md` ×5,
`17-tablebases.md` ×4, `04-state-make-unmake.md` ×3, `06-see.md` ×2,
`04-make-unmake.md` ×2, `04-board-state.md` ×2), 3 `docs/*.md`
(`1-basics.md`, `3-data.md`, `nnue.md`), 2 `.agents/rules/` links
(`elo-decomposition-testing-protocol.md`, `policy-integration-status.md`), and
the `tools/*.py` → `jax/tools/*.py` class (19 refs). `design/00-master-plan.md`
§6.2 flagged this sweep as a "decide now" item and it was never done.
*Work:* mostly `sed`, but each hit should be eyeballed — some name docs that
were *split*, not renamed. *Risk:* **none**. *Verify:* a link-check script that
extracts every `design/…\.md`, `docs/…\.md`, `tools/…\.(sh|py|rs)` string from
tracked files and asserts each resolves. Add it to `check.sh`.

**R7. Fix the bench pin so it is stated once and checked by a machine.**
*Problem:* 385329 appears once (CLAUDE.md prose); 548978 appears 12 times as
if current, in `src/eval/nnue/mod.rs`, `sota_features.rs`, `sota_head.rs`,
`design/10`, `design/11`, `design/13`, and `tools/nnue/s1_gate_v1.sh`.
*Work:* (a) add `BENCH_PIN=385329` to `tools/check.sh` and make it **fail**
when the measured total differs, with a message explaining that a functional
change should update the pin in the same commit; (b) delete every other
assertion of a bench number from source and design doc comments, replacing
them with "the pinned value lives in `tools/check.sh`"; (c) keep the historical
numbers in `docs/strength-log.md` and commit trailers, where they are dated
evidence rather than claims about the present.
*Risk:* **none** (the script change cannot alter the binary). *Payoff:* this is
the safety net for every item in Tiers B and C.

**R8. Extend the merge gate to the feature matrix.** *Problem:* 30% of `src/`
is never compiled, linted, or tested by `check.sh` or CI, including everything
the match binaries run.
*Work:* add to `check.sh` (and CI) a loop over
`sota`, `sota,arch-d47v`, `sota,arch-d48x`, `sota,policy`, `datagen`,
`search-control`, `tune`, running `cargo clippy --all-targets -- -D warnings`
and `cargo test` for each. Also add `--features tune` bench-identity check
(D45 already claims the tune build is bit-identical; nothing verifies it).
*Risk:* **none to behaviour**, but expect it to surface existing clippy
warnings and possibly failing tests in the ungated code — budget time for that,
and land the fixes separately. *Cost:* this is the expensive one in Tier A in
wall-clock, and **it must not be run while the GPU trainer holds memory** —
`lto="fat"` + `codegen-units=1` makes each release build multi-gigabyte.
Schedule it for a quiet box.

**R9. `tools/README.md` and `tools/nnue/README.md`.**
*Content for `tools/README.md`:* the ~22 tools a newcomer must know
(`check.sh`, `sprt.sh`, `gauntlet.sh`, `tactics.sh`, `build_pgo.sh`,
`deploy_engine.sh`, `perft_diff.py`, `verify_perft.py`, `match_report.py`,
`selfplay_smoke.py`, `ledger/ledger.py`, and the durable `nnue/` set), plus a
warning that three top-level scripts (`run_rr_tournament.sh`,
`search_knob_gate.sh`, `perf_study4_bench.sh`) are dated one-offs, plus the
`tools/ledger/` ≠ `ledger/` note, plus **"there is a second tools directory at
`jax/tools/`; the training driver is `jax/tools/pump_train.py`"**.
*Content for `tools/nnue/README.md`:* the six functional buckets with counts,
naming the ~10 durable Python tools and the 8 durable shell scripts
explicitly, and stating that files matching `e[0-9]_*`, `d4[0-9]_*`, `uho_*`,
`stage2_*`, `probe_*`, `gate_*`, `exp_unc_*`, `prod_*` are one-shot campaign
drivers preserved for provenance, not tools. Add the loud warning that **31
tracked files hardcode `/root/projects/learnrust/chessengine`** and will not
run as checked in.
*Work:* 3 hours. *Risk:* **none**. *Payoff:* very high — this is the single
worst discoverability failure in the repo and the index costs nothing.

**R10. Repair the design suite's pointer chain and its own metadata.**
- `design/README.md`: change the "current research checkpoint" pointer from
  `archon-execution-plan.md`/`policy-automl-state.md` to
  `docs/research/STATE.md` (or `value-foundation-state.md` until R12 lands);
  update "D1–D44" to the real range; drop the "Some doc headers still read
  Draft v1" caveat by actually fixing the two (`10`, `15`) that still do.
- `design/decisions.md`: add the missing **D48** entry (the D48W→D48X gen-2
  architecture ruling, currently recorded only in `Cargo.toml` comments,
  `CLAUDE.md`, and the research log) and fix the header's decision range. The
  process rule says decisions land here *first*; right now the architecture
  being trained has no decision row.
- `design/01-architecture.md` §3.1: the repo tree is materially wrong — it
  shows `see.rs` and `eval.rs` (neither exists; `see_ge` is in `board.rs`,
  `eval` is a directory) and omits `obs/`, `policy.rs`, `cuckoo.rs`, `tune.rs`,
  `src/bin/`, `jax/`, `ledger/`, `docs/`, and `tools/`'s subdirectories.
- `src/lib.rs`: the crate doc says decisions "D1–D27".
*Risk:* **none**.

**R11. Add `docs/ENVIRONMENT.md`.** One page listing the 19 runtime env
switches (name, what reads it, what it does, whether it is live or a parked
experiment) and the five net container formats with their magic bytes and
which loader handles each. For each env var, state whether it is part of a
current experiment or residue — and where you cannot tell, say so. This is the
single densest piece of undocumented behaviour in the engine.
*Risk:* **none**.

**R12. Split `value-foundation-state.md`.**
*Problem:* 7,001 lines, four "TAKEOVER CHECKPOINT" blocks, two claiming
primacy, a superseded-advice banner mid-file, non-monotonic ordering.
*Proposal:*
- `docs/research/STATE.md` — **new, ≤500 lines, overwritten in place, never
  appended.** Exactly one checkpoint. Structure: current incumbent chain and
  external anchors; in-flight jobs with their unit names and config paths;
  binding invariants; open questions; a link to the log.
- `docs/research/log/value-foundation-2026-07-13-to-07-19.md` — the B4 arc
  (the bottom ~2,350 lines).
- `docs/research/log/value-foundation-2026-07-20-to-07-27.md` — the current arc.
- `docs/research/value-foundation-state.md` **stays at its path** as a short
  redirect stub, because 32 files reference it and 16 of those references are
  in `src/`, `tools/`, `jax/`, and `ledger/*.jsonl` rows that must not be
  rewritten.
*Work:* half a day of careful splitting; the checkpoint hierarchy is already
marked in the file. *Risk:* **low** — losing content. Mitigation: `git mv` the
bulk and verify `wc -l` conservation across the split before deleting anything.
*Payoff:* very high. This is the document every session must read first.

---

### Tier B — mechanical code moves (bench-verifiable, no semantic change)

A general fact that makes this tier much safer than it looks: **the depth-6
bench figure is a node count, and node counts are a pure function of search
semantics, not of codegen.** A behaviour-preserving module move cannot change
385329 — only NPS could move, and with `codegen-units = 1` + `lto = "fat"` the
whole crate is one codegen unit, so intra-crate module boundaries have no
codegen effect either. The bench is therefore a *complete* oracle for this
tier, provided R7 has made it assert rather than print. (Splitting into
separate *crates* is a different matter — cross-crate inlining does change —
which is why the workspace item is in Tier C.)

**R13. Move `src/search/mod.rs`'s test module to `src/search/tests.rs`.**
4,191 lines out of a 9,137-line file. *Verified safe by reading:* the module
is `#[cfg(test)] mod tests { use super::*; use crate::fen::START_FEN; use
crate::movegen::{…}; use crate::types::Square; … }`. As a child file module,
`super::*` resolves to `crate::search` exactly as it does inline, and child
modules retain access to the parent's private items. The change is
`#[cfg(test)] mod tests;` in `mod.rs` plus a `git mv`-style extraction.
*Work:* 20 minutes. *Risk:* **low** — compile error or nothing.
*Verify:* `cargo test` count unchanged; `check.sh` bench = 385329.
*Payoff:* the largest file in the repo drops 46%, and the reader stops
scrolling past 4,000 lines of tests to reach `qsearch`.

**R14. Extract the pure free functions from `search/mod.rs`.** Three new files,
all mechanical:
- `search/margins.rs` ← `rfp_margin`, `rfp_prune_margin`, `lmp_move_count`,
  `parent_futility_value`, `capture_futility_value`, `capture_see_margin`,
  `singular_verification_beta`, `singular_double_margin`,
  `singular_triple_margin`, `continuation_history_prunes` (lines ~214–509).
- `search/timectl.rs` ← `Worker::falling_eval`, `stability_reduction`,
  `best_move_instability`, `high_best_move_effort` (lines 4708–4789 — all
  associated functions taking no `self`, so they lift out cleanly), plus
  `unc_tm_factor` and `wdl_tm_factor`.
- `search/report.rs` ← `SearchInfo`, `UciSink`, `SearchReport`, `best_thread`,
  `elect_winner`, `ponder_from_tt` (lines ~565–740).
Call sites change from `Self::falling_eval(…)` to `timectl::falling_eval(…)`
and from bare names to `margins::…`. *Work:* 2 hours. *Risk:* **low**.
*Verify:* bench 385329, `cargo test`, and the feature matrix from R8.
*Payoff:* `search/mod.rs` drops to ~4,000 lines of which the iterative-
deepening / PVS / qsearch core is finally the majority, and the pruning
formulas — the things most often tuned — get their own greppable home next to
`params.rs`.

**R15. Group the search-control trio into `src/search_control/`.**
`search_control.rs` → `mod.rs`, `search_intervention.rs` → `intervention.rs`,
`search_trace.rs` → `trace.rs`. Consumers are only `src/lib.rs` and
`src/search/mod.rs` (verified: no test, bench, or bin imports them). The public
root path `chessengine::search_control` is preserved; the two siblings move
under it. *Work:* 30 minutes. *Risk:* **low** (feature-gated code — needs R8's
matrix gate to be meaningful, so land R8 first).

**R16. Consolidate NNUE architecture selection into `src/eval/nnue/arch.rs`,
and add the missing exclusivity guard.** Today the dimension `#[cfg]` ladders
are split between `eval/nnue/mod.rs` (55 lines for `L1` across five
`nnue-l1-*` features) and `eval/nnue/sota_head.rs` (40 lines for `L1`/`L2`
across four `arch-*` features), and the `arch-*` group has **no
`compile_error!`** despite Cargo.toml declaring the same exclusivity the
`nnue-l1-*` group enforces. Move both ladders into one `arch.rs` that
re-exports the constants, and add the guard.
*Work:* 1–2 hours. *Risk:* **low** — const values are unchanged, so the
default build is bit-identical; the feature builds need R8's matrix to verify.
*Payoff:* "what shape is this build?" becomes one file instead of a
cross-module scavenger hunt.

**R17. Add `//!` headers to the five modules that lack one** (`board.rs`,
`types.rs`, `bitboard.rs`, `moves.rs`, `fen.rs`) and **rewrite
`src/eval/nnue/mod.rs`'s header**, which currently describes a P13-step-1
skeleton that returns PSQT. *Risk:* **none** (comments only) — but note that
`board.rs` is D24-guarded, so even a comment-only patch there should carry a
`Bench:` trailer proving the number did not move.

---

### Tier C — ambitious; do after Tier A and when the box is quiet

**R18. Restructure `docs/research/` into subdirectories.**
`perf/` (the 9-file perf/profiling chain + the three artifact dirs),
`log/` (the split state logs), `archive/` (the 3 explicitly-dead docs +
`docs/research/artifacts/`'s 11 orphaned JSONs), and leave everything else
flat. **Do not move** `value-foundation-state.md`, `data-recipe-contract.md`,
or `sf18-canon-data-diagnosis-2026-07-21.md` — all three are referenced from
`src/`, `tools/`, `jax/`, and `ledger/*.jsonl`, so moving them is a code
change, not a docs change.
*Risk:* **medium** — string references. *Verify:* the link-checker from R6.

**R19. Consolidate the perf artifacts.** `/perf-artifacts/` → `docs/research/
perf/artifacts-2/`, `/perf-artifacts-6/` → `docs/research/perf/artifacts-6/`,
matching the convention the in-flight study is establishing with
`docs/research/perf-artifacts-7/`. Only 10 inbound references exist, in two
doc files. **Coordinate with the running perf study before touching anything
under `docs/research/perf-*`.** *Risk:* **low**, but timing-sensitive.

**R20. Move `docs/learn/video/` out of `docs/`.** It is a rendering codebase
(7 Python modules, 2 shell scripts, Manim scenes, a build plan), not
documentation. `tools/video/` or `media/`. Zero inbound references from
outside `docs/`. *Risk:* **none** to the engine. Also worth deciding whether
the course itself gets a "written 2026-07-17, chapter 15 is pre-D47" banner —
cheap honesty that costs nothing and prevents a reader trusting a stale
frontier chapter.

**R21. Archive the concluded `tools/nnue/` campaign scripts.**
~70 one-shot drivers + the 22-file counterfactual cluster + the 28 non-example
configs into `tools/nnue/archive/<campaign>/`.
*Risk:* **medium-high**, and this is the item most likely to cause regret:
- 82 ledger rows cite `tools/` paths as regeneration recipes. Those rows are
  append-only and must not be edited. Mitigation: add `ledger/moved-paths.md`
  as an old→new lookup table, and reference it from `ledger/README.md`.
- Eight bare same-directory Python imports break the moment the flat directory
  is subdivided (`eval_match.py → policy_sweep.py`, `build_root_consequence_
  npz.py → nnue_ref.py`, `collect_search_trace.py → root_move_oracle.py`,
  `relabel_trace_counterfactual.py → root_move_oracle.py`,
  `stratify_quality_fens.py → relabel_fen_static_paired.py`,
  `test_lc0_policy_map.py → lc0_policy_map.py`,
  `bootstrap_independent_baseline.py → bootstrap_exact_head_compare.py`,
  `build_{counterfactual,exact_counterfactual}_npz.py → build_search_trace_
  npz.py`). Any subdivision must keep each importer with its import partner,
  or add a `sys.path` shim.
- `tools/nnue/bf-convert/Cargo.toml` has `[[bin]] path = "../bf_convert.rs"`.
*Recommendation:* **do R9 (the index) instead, and defer R21.** An index buys
most of the comprehension benefit at none of the risk. Revisit only if the
directory keeps growing.

**R22. Cargo workspace for `pymask/` + `tools/nnue/sfbinpack-audit/` +
`tools/nnue/bf-convert/`.** *Attraction:* one `cargo build` builds the PyO3
extension the live trainer depends on, instead of a manual `.so` copy; one
lockfile. *Risk:* **medium-high** — a workspace unifies `Cargo.lock`, which can
move dev-dependency versions (criterion, proptest) and changes target-dir
layout, which `tools/build_pgo.sh` and 42 scripts referencing
`target/release/chessengine` depend on. Cross-crate inlining also becomes
relevant if any engine code is ever pulled out. *Verify:* bench 385329 plus a
full re-run of `build_pgo.sh` and `deploy_engine.sh`. **Do not attempt while
the SPSA campaign is consuming `engines/bin/archon-d47v-tune-pgo`.**

**R23. Rename the `sota_*` prefix.** `src/eval/nnue/sota_*.rs` →
`src/eval/nnue/cenn2/{head,loader,features,threats,simd,context,accumulator}.rs`,
matching the `CEN2` magic the loader actually checks and dropping a value
judgment that has already aged out. *Risk:* **medium** on its own (≈200
intra-crate use sites, all mechanical) but **high right now**: a performance
study is actively editing these exact files. Defer until that lands, then do it
as a single no-other-changes commit.

**R24. Consolidate the three "policy" modules.** `src/policy.rs` +
`eval/nnue/policy_head.rs` + `eval/nnue/policy_vocab.rs` →
`src/eval/nnue/policy/{mod,net,head,vocab}.rs`. *Risk:* **medium**. Low urgency
— the live ruling ditched the policy head from the D48X run (`pol_lambda 0`),
so this subsystem is dormant. Worth doing *because* it is dormant: no merge
conflicts. But its dormancy also means low payoff.

---

## 5. Things I could not determine, and how to find out

Stated as open questions rather than guesses, per the mandate.

1. **Are `nnue-l1-512`, `nnue-l1-1024`, `nnue-l1-4096`, and `embed-net` dead?**
   Each has ≤3 mentions in tracked scripts/docs and was last touched
   2026-07-12/16. *How to find out:* grep `ledger/nets.jsonl` for nets whose
   `arch` field names those widths and check whether any is still a live
   candidate; then check `engines/bin/` for a binary built with them. If both
   come back empty, they are removable — which would delete ~50 lines of
   `#[cfg]` ladder.
2. **Are `tools/spsa/campaigns/c1-*.json … c8-*.json` superseded by
   `s2-stageA/B`?** The design shape suggests yes; no document says so.
   *How to find out:* `grep -c 'c1-corrhist\|c2-reduction\|…'
   ledger/experiments.jsonl` — an SPSA campaign that ran has a `kind:"spsa"`
   row.
3. **Which of the 28 non-example `tools/nnue/configs/*.json` ever produced a
   registered net?** At least 10 name `net_id`s absent from
   `ledger/nets.jsonl`'s 58 rows. *How to find out:* for each, test
   `data/nnue/nets/<net_id>.cenn` — present-but-unregistered is a
   `RUNBOOK.md` rule violation to fix; absent means the config never ran.
4. **Are the five undocumented `tools/nnue/bullet/*.rs` files dead?**
   (`classic_frozen_policy`, `d40_attn`, `d41_alloc`, `d41_alloc11`,
   `halfka_residual` — none named as current by `bullet/README.md`.)
   *How to find out:* they are only reachable via `cargo build --example` from
   the gitignored bullet checkout in `.archon-state/`; check whether any
   tracked or `runs/` script names them. My grep found none, but `runs/` was
   only partly searchable while untracked — R2 fixes that.
5. **Does the tooling that reads `GEMINI.md` follow symlinks?** Determines
   whether R4 can symlink or must duplicate a stub.
6. **Do the feature builds currently pass clippy and tests?** Unknown — the
   gate has never run them and I could not build. This is R8's first finding
   and should be budgeted for, not assumed clean.
7. **Is `tools/analysis/`'s hardcoded default PGN
   (`data/nnue/experiments/pgo-vs-sf16/games.pgn`) still present?** Determines
   whether those three scripts are runnable as checked in.

---

## 6. Migration plan for the top items

Sequencing principle: **install the safety nets before moving anything.**
R7 (bench pin asserted) and R8 (feature matrix) exist so that every later step
has an oracle. Nothing in Tier B should be attempted before them.

### Phase 0 — safety nets (no moves)
1. **R7** — pin `BENCH_PIN=385329` in `check.sh`; make it fail on mismatch;
   delete the 11 stale in-source/in-doc bench assertions.
   *Verify:* `bash tools/check.sh` exits 0 and prints the trailer. Deliberately
   break the pin once (set it to a wrong value) to confirm it fails, then
   restore.
   *Reversible:* fully.
2. **R8** — add the feature-matrix loop to `check.sh` and CI.
   ⚠ **Run only when the GPU trainer is not holding memory.** `lto = "fat"`
   with `codegen-units = 1` makes each release build multi-gigabyte, and the
   trainer runs with `MemorySwapMax=0`.
   *Verify:* each matrix arm compiles and its tests pass; expect to fix
   pre-existing clippy findings in the ungated 15,600 lines — land those as
   separate commits before proceeding.
   *Reversible:* fully.
3. **R6 link-checker** — add a script that asserts every
   `design/…\.md` / `docs/…\.md` / `tools/…\.(sh|py|rs)` string in a tracked
   file resolves; run it, fix the 61 current failures, wire it into
   `check.sh`. *Reversible:* fully.

### Phase 1 — the free comprehension wins (still no engine risk)
4. **R2** — un-ignore and track `tools/nnue/runs/`. Do **not** move
   `d48x_watch.sh`, `d48x_prune.sh`, or `d48x_1280.json` — live systemd units
   read them by path. *Verify:* `systemctl --user status archon-d48x-watch`
   still active; `python3 tools/ledger/ledger.py verify` clean.
   *Reversible:* yes (re-ignore, `git rm --cached`).
5. **R1, R3, R9, R10, R11** — README, research index, tools indexes, design
   pointer/metadata repair, environment doc. Pure additions and comment edits.
   *Verify:* the link-checker from step 3 stays green.
   *Reversible:* fully.
6. **R4, R5** — GEMINI.md stub, root hygiene (`vgcore.*`, `d40_feat_ckpt/`,
   the dead patch). *Reversible:* the deletions are recoverable from git except
   `vgcore.534151`, which was never tracked and is a core dump — losing it is
   the point.
7. **R12** — split `value-foundation-state.md`; leave a stub at the old path.
   *Verify:* `wc -l` across the three new files + stub ≥ 7,001; every heading
   present in the original is present somewhere; the 32 inbound references
   still resolve (they point at the stub).
   *Reversible:* yes, but re-merging is manual — take the split slowly and in
   one commit.

### Phase 2 — the code moves (each its own commit, each bench-verified)
8. **R13** — extract `search/tests.rs`.
   *Verify after:* `cargo test` (same test count), `cargo clippy
   --all-targets`, `bash tools/check.sh` → **Bench: 385329 unchanged**.
   Commit trailer states the unchanged number, per D24's non-functional rule.
   *Reversible:* trivially.
9. **R14** — extract `margins.rs`, `timectl.rs`, `report.rs`.
   *Verify:* identical to step 8, plus the R8 matrix (these files are touched
   by `sota` and `search-control` builds).
   *Reversible:* trivially.
10. **R17** — module headers, including the `eval/nnue/mod.rs` rewrite.
    *Verify:* `board.rs` is D24-guarded, so run `cargo test --release --
    --ignored perft_full_suite` as well as the bench, even for a comment.
11. **R16** — `eval/nnue/arch.rs` + the missing `compile_error!`.
    *Verify:* bench unchanged **and** each `arch-*` matrix arm still compiles
    **and** a deliberate double-feature build now fails with a readable
    message.
12. **R15** — `src/search_control/`.
    *Verify:* `--features search-control` arm of the matrix.

### Phase 3 — doc restructuring (after the perf study lands)
13. **R19** then **R18** then **R20**. Each followed by the link-checker.
    Coordinate R19 with whoever owns `docs/research/perf-artifacts-7/`.

### Deferred indefinitely
R21 (tools archive), R22 (workspace), R23 (`sota_` rename), R24 (policy
consolidation) — all four should wait for a quiet tree, and R21 should
probably never happen given R9 buys most of its value.

### What is reversible and what is not

| Step | Reversible? |
|---|---|
| Everything in Phase 0 and Phase 1 except the `vgcore` deletion | Yes, by `git revert` |
| `vgcore.534151` deletion | No — never tracked. Intended. |
| R12 log split | Yes, but re-merging is manual work |
| All Phase 2 code moves | Yes, trivially — `git revert`; bench proves equivalence both ways |
| R2 un-ignore | Yes, but re-ignoring after a reboot could lose newly-created files again — that is the failure mode it exists to prevent |
| R21 tools archive | Technically yes, but ledger recipe lookups degrade in the interim |
| R22 workspace | Yes, but `Cargo.lock` unification may not restore byte-identically |

---

## 7. Do not touch — load-bearing things that look like clutter

**Engine correctness surface (D24 ritual applies to every byte):**
- `src/movegen.rs`, `src/board.rs`, `src/zobrist.rs`, `src/attacks.rs`. Even
  a module *move* here demands `cargo test --release -- --ignored
  perft_full_suite` plus the bench. Do not reorganize these to reduce file
  size; `movegen.rs` is 2,312 lines of which 1,649 are tests, and extracting
  those (a safe R13-style move) is the *only* change worth making here.
- `tests/data/perft.epd` — the correctness oracle. Counts are externally
  verified facts. Never edit; the suite only grows.
- `tests/data/openings/openings-8ply.epd`, `tests/data/tactics-smoke.epd` —
  every position machine-verified; regenerating is not free.

**Tuning surface with a live consumer:**
- `src/search/params.rs` (1,100 lines, no tests). It looks like an obvious
  split candidate — do not split it and do not rename a single constant. The
  D20 contract is that every SPSA-movable scalar is greppable by its
  SCREAMING_SNAKE name in one file, and `tools/spsa/campaigns/*.json` names
  those constants **literally**. An SPSA campaign
  (`archon-spsa-stageA.service`) is running right now against
  `tools/spsa/campaigns/s2-stageA.json`.
- `src/tune.rs` — 113 lines, looks vestigial; it is the `tunables!` macro that
  makes the whole SPSA harness possible.

**Provenance surface:**
- `ledger/*.jsonl` — append-only. Never rewrite a historical row, including a
  row whose `recipe` path is now wrong. Identity is the sha256; rewriting the
  recipe destroys the audit trail. If a tool moves, record it in a new
  `ledger/moved-paths.md`.
- `tools/ledger/ledger.py` — the only writer for the above.
- `tools/nnue/calib/scale-check-800.fens` and `wl50.fens` — two data files that
  look like scratch; they are the input to `net_scale_check.sh`, which D46
  makes a **hard pre-game gate**.
- `tools/nnue/bullet/*.rs` + `patches/*.patch` — the bullet checkout itself is
  gitignored and ephemeral; these tracked copies are the only record of what
  actually trained the historical nets. Label them historical; do not delete.

**Untracked but essential:**
- `data/` (773 GB) and `engines/` (1.3 GB) — referenced by 210 and 97 tracked
  files respectively. Never move, never rename, never add to a reorg script's
  glob.
- `.archon-state/toolchains/{bullet,bulletformat}` — the pinned trainer
  checkouts the live jobs depend on.
- `config.json` at the root — fastchess runtime state. It is gitignored and
  correctly so; the only defensible change is documenting in the new README
  that it is *not* project configuration.
- The five `target-pgo-*` directories, including the one with a comma in its
  name — `tools/build_pgo.sh` derives `TD=target-pgo-$FEATURES` verbatim.
  Ugly, load-bearing.

**Reference trees:**
- `stockfish_reference/`, `reckless_reference/`, `integral_reference/` — these
  look like vendored third-party code. They are not: the clones are gitignored;
  the 44 tracked files are original distilled study notes, cited by
  `decisions.md` (D27/D31/D37) as the provenance chain for every constant
  adopted from Stockfish. `stockfish_reference/07-s0-sf16-mechanics-audit.md`
  is the source for the current search program.

**Live processes:**
- Sixteen transient `systemd-run` units are active. Because they are transient,
  **there are no unit files on disk** — the invocation exists only in systemd's
  runtime state and in the launch scripts under the (currently gitignored)
  `tools/nnue/runs/`. Do not stop, restart, or reconfigure any of them, and do
  not move any file they name: `tools/nnue/runs/{d48x_watch,d48x_prune,
  d47v_ckpt_watch}.sh`, `tools/nnue/runs/d48x_1280.json`,
  `jax/tools/pump_train.py`, `tools/nnue/tb_shim.py`,
  `tools/spsa/{stageA_chain.sh,run_spsa_dist.sh}`,
  `tools/spsa/campaigns/s2-stageA.json`, `engines/bin/archon-d47v-tune-pgo`,
  `data/nnue/nets/screens/d47v-a4b2-cal400.cenn`.
- `docs/research/perf-study-7-*` and `docs/research/perf-artifacts-7/` are
  being written by a concurrent study.

**Finally: use `git ls-files`, never a filesystem glob, in any reorg script.**
Four locations hold ~130 untracked-but-present files that sit interleaved with
tracked ones and are indistinguishable by eye: `tools/nnue/runs/` (76),
`tools/nnue/__pycache__/`, `tools/nnue/{bf-convert,sfbinpack-audit}/target/`,
and `tools/spsa/out/`. A `cp -r` or `mv` that ignores `.gitignore` sweeps them
along.

---

## 8. What this buys

If Tier A alone lands, a newcomer's first hour changes from "read a 7,000-line
log to find out what is true" to:

- `README.md` → what this is, how to build it, what to read next (5 min)
- `docs/research/STATE.md` → current incumbent, in-flight jobs, invariants
  (10 min)
- `docs/MAP.md` + `tools/README.md` → where anything lives (5 min)
- `design/decisions.md` → why it is shaped this way (as long as they like)

and every cross-reference they follow resolves, every bench number they read is
the real one, and the code they are looking at is described by its own header.

Tier B halves the largest file in the repository and gives the pruning
formulas, the time-control math, and the thread-election logic their own
homes — without changing a single node of search behaviour, which the bench
pin proves.

The cost of Tier A is roughly two days of writing and one careful pass with a
build machine that is not fighting a GPU trainer for memory. It moves almost
nothing, which is precisely why it is worth doing first.

---

## ADDENDUM — independent verification of the path findings (2026-07-27 ~10:2x)

Added by the main session after spot-checking two of this proposal's
automated claims. **One held, one did not.** Both matter for how much to
trust the remaining path counts.

### REFUTED: "a ledger regen recipe points at `tools/pump_train.py`, which doesn't exist"

It exists and the recipe is correct. The recipe begins **`cd jax && …`**, so
`tools/pump_train.py` resolves to `jax/tools/pump_train.py` from that working
directory. Exactly one row cites it and it is correctly prefixed; zero rows
are broken. The checker did not account for the `cd`. I was one step from
appending a correction annotation to a correct row.

The related claim that "eight ledger regen recipes point at untracked files"
is a different and probably fair statement — *untracked* is not *missing*
(e.g. `tools/nnue/train_canon.sh` exists but is gitignored). Worth keeping,
but it is a versioning concern, not a broken-recipe concern.

### CONFIRMED: the dangling documentation references are real

An independent sweep of our own markdown (excluding the three reference
clones — my first attempt scanned them by mistake and reported 1,992
"missing" refs that were Reckless and Stockfish notes citing their own
sources) finds **44 distinct dangling targets across 94 occurrences** in
1,098 checked references. Same order as this proposal's ~61; the finding
stands.

**Most of the volume is four doc renames plus a file-extension change**, all
with verified existing targets:

| dangling reference | occurrences | actual file |
|---|---:|---|
| `design/07-search.md` | 7 | `design/07-search-core.md` |
| `design/09-movepick-history.md` | 5 | `design/09-move-ordering-history.md` |
| `design/04-state-make-unmake.md` | 3 | `design/04-position-state.md` |
| `ledger/{data,nets,experiments}.json` | 10 | the same files with `.jsonl` |

**One is NOT a rename and must not be sed-ed:** `design/05-movegen-perft.md`
(12 occurrences) was **split into two files** — `05-movegen.md` and
`06-perft-validation.md`. Each citation has to be routed by what it actually
cites. That is the single largest cluster and it needs a human pass.

### Implication for this proposal's other automated counts

Two samples, one wrong. The structural observations in this document have
been good; the automated path arithmetic is over-eager and should be
re-verified per item before anyone spends time on it — particularly the
"19 dangling `tools/*` refs" and the "31 files hardcoding
`/root/projects/learnrust/`" figures, which nobody has checked yet.

### CONFIRMED, and worse than stated: the stale absolute path breaks 13 ledger regen recipes

The proposal notes 31 tracked files hardcoding `/root/projects/learnrust/chessengine`
(a pre-migration root) and calls those scripts "already broken". Verified: **32
files** — 27 under `tools/`, 4 docs, 1 ledger row — of which **18 are shell
scripts, and in all 18 the stale path is LIVE CODE, not a comment.**

The part the proposal did not connect: **13 of those 18 scripts are cited by
ledger regen recipes**, including `uho_ladder_v1.sh` (4 rows),
`uho_full_wave_v1.sh` (2), `uho_wave3_v1.sh` (2), `uho_wave4_v1.sh`,
`stage2_v1.sh`, `chain_v2.sh`, `e1_finish.sh`. So D38's stated guarantee —
"any version is reproducible, any net's lineage is provable" — is **not
currently true for those artifacts**: following the recipe fails on a path
that has not existed since the 2026-07-19 machine migration.

The fix is mechanical and low-risk. Every occurrence has the identical shape
`/root/projects/learnrust/chessengine/data/nnue/experiments/…` — a repo-root
prefix followed by a repo-relative path — so it is a prefix substitution, not
18 individual judgements. The only care needed is that the occurrences sit
inside embedded Python heredocs, so the replacement must be either the
current absolute root or a relative path with a guaranteed `cd` to the repo
root (preferred — it survives the next migration).

**Deliberately NOT done by the main session.** The correctness point is real,
but the practical value is low: these are one-shot campaign scripts from
finished experiments, and nobody is planning to re-run `uho_wave3_v1.sh`. The
honest framing is that the ledger rows are provenance records whose recipes
are aspirationally executable. Worth fixing as part of a Tier-A sweep, not
worth an hour on its own — and it belongs to the user's tier decision, not to
an unasked cleanup.
