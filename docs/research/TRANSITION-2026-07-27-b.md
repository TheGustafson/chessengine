# TRANSITION — 2026-07-27 (second handoff, late session)

Supersedes `TRANSITION-2026-07-27.md` for anything after ~14:00. Read this
first, then `ENGINE-INVESTIGATION-PLAYBOOK.md`.

---

## 1. LIVE STATE

| thing | state |
|---|---|
| **D48X training** | **~sb445/520**, `archon-d48x.service` active, ~175.5k pos/s |
| anneal | started at sb400; heldout **0.007132 → 0.006893**, EMA 0.006783 → 0.006671 — descending cleanly |
| ETA | ~75 superbatches left, ~4 h |
| restart | `bash tools/nnue/d48x_launch.sh` (auto-detects resume; plain `uv run` FAILS on `jax/`) |
| SPSA | stopped, deliberately |
| git | all work committed on branch `wip/class4-material-eval` |

---

## 2. THE HEADLINE: a whole day of search measurements were PSQT

`target/release/chessengine` **silently ignores `setoption name EvalFile`**.
The CEN2 net reader lives behind `sota,arch-d47v`; a build lacking them cannot
load the net and stays on PSQT. `screen.py`, `engine_compare.py`,
`search_forensics.py`, all three SPRTs, and the SF18 loss-forensics games all
pointed at that binary.

**Everything quantitative about search shape from this session is invalid.**

Corrected spot-checks with a net-live binary:

| measurement | PSQT (reported all day) | NNUE (real) |
|---|---|---|
| qsearch amplification | 2.65 | **1.59** (SF18 ≈ 1.5) |
| stand-pat cut rate | 30.7% | **52.5%** |
| width ratio @d10 | 2.97 | **1.40** |
| width ratio @d14 | 1.87 | **0.93** (we are *smaller*) |
| fhf | 89.49% | **85.55%** |
| mean cp loss @64k | 17.74 | **4.64** |
| top-1 agreement | 45.2% | **66.5%** |
| blunder rate | 2.1% | **0.1%** |
| iteration stability vs SF | "4.27×" | **1.74×** (also a units bug) |

**There is no 2–3× tree-width gap and no qsearch node explosion.** The engine
is close to Stockfish's tree shape. The entire "width gap" investigation —
including its terminus in `SCREENING.md` §17 — chased a PSQT artifact.

### Fixed so this cannot recur
- `nnue::active_eval_kind()` + UCI reports `info string eval: NNUE(sota-v2)`
  on every load, or `info string ERROR EvalFile: …` naming the missing
  feature.
- `assert_net_live()` in `screen.py` / `engine_compare.py` probes depth-8
  nodes with and without the net and **refuses to run** if identical.
- Tools default to `target/qtrace/release/chessengine`, honour `ARCHON_BIN`.
- Build: `cargo build --release --features tune,sota,arch-d47v --target-dir target/qtrace`

A detail worth remembering: the loader error was **always printed**. A
diagnostic grep with `head -6` let six `option name …EVAL…` lines fill the
window first. The engine was not silent; the filter was. Hence the *positive*
confirmation line — absence of an error is weaker evidence than presence of a
success.

---

## 3. What still stands (net-independent)

- **Constants are faithful** to SF master `eca43a97` — all 260 screened, no
  drift. `tools/spsa/param_provenance.py` re-checks. Note `eca43a97` is 223
  commits *newer* than the `sf_18` tag; reading the tag is a real mistake that
  already cost one bad commit (reverted).
- **Eval frame ≈0.64× SF18's**, uniform across the range (quantile-matched,
  in-check positions excluded), and the eval's *error* scales with it.
- **17 implicit-×1 coefficients** found and made tunable — terms with an
  invisible coefficient of 1, invisible to any "bare literal" rule. Includes
  the whole TT replacement policy, which had none. **More surely exist.**
- **~46% of tunables are in no SPSA campaign.**
- **Eval-frame rescale is refuted** as an Elo source in five formulations
  (global, dual, per-family, per-mechanism, class-4 alone). Several
  corrections were *correctly derived* and measured ~zero because the gate was
  rarely reached or dominated by frame-clean filters.

---

## 4. Tools built this session (all committed)

| tool | purpose |
|---|---|
| `tools/screen.py` | one command, 3 instruments, `--base` chaining, results ledger, net guard |
| `tools/engine_compare.py` | ARCHON vs SF on identical positions: width, stability, eval bias, `--static` net-vs-net, `--our-options` |
| `tools/search_forensics.py` | `ladder`, `research`, `ebf`, `qshare`, `shallow`, `aspiration` |
| `tools/loss_positions.py` | localise the losing move; loss-derived corpus; `--compare` taxonomy diff |
| `tools/obs_to_oracle.py` | ceobs → tiered SF18 oracle (9,992 positions, every legal move scored) |
| `tools/frame_audit.py` | every eval-denominated threshold in real pawns, both frames |
| `tools/param_provenance.py` | constants vs `eca43a97`, campaign coverage |
| `qtrace()` in `src/search/mod.rs` | feature-gated qsearch decision debugger |
| `docs/research/screen-results.jsonl` | append-only results ledger; warns on repeats |

---

## 5. Immediate next steps

1. **Re-run everything on a net-live binary.** In priority order: baseline
   shape (width/fhf/stability/amplification), the top candidates
   (`QUIET_LOW_PLY_WEIGHT 12`, `NMP_EVAL_BASE 168`, `RED_TABLE_NUM 2700`,
   `QS_FUTILITY_MARGIN 60`), then the SF18 games + loss forensics.
2. **Re-measure the power curve** — the old sem figures were PSQT-scale
   (mean loss 17.7 vs 4.6 now).
3. Hunt more implicit-×1 coefficients.
4. SPRT anything that survives, **on net-live binaries both sides**.

---

## 6. Engine state

Clean. `QS_PIECE_MULT`, `CFUT_PIECE_MULT`, `SEE_THRESH_MULT`,
`RAZOR_CORR_DIV`, `NMP_CORR_DIV`, `IIR_REDUCTION`, `LMP_DEPTH_MULT`,
`LMP_IMPROVING_DEN`, the 12 LMR/SE/TT implicit-coefficient tunables and the
6 TT policy tunables are all wired **at identity** and verified inert
(default bench 14,442,305 on the 800-fen depth-10 set). No behaviour change
has been landed today.

The temporary qsearch counters added mid-session were removed; only the
feature-gated `qtrace` remains.
