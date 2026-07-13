# NNUE training-run RUNBOOK — the solved procedure

**Purpose:** run a standard net training experiment end-to-end with minimal
figuring-out. Training is a *solved procedure* here — follow these steps in
order; only deviate when the experiment is deliberately about the training
itself. Every step's exact tool interface is in the file it names (read that
file's header/`--help` for flags); this runbook owns the FLOW, the non-obvious
CONSTRAINTS, and the ledger/SPRT commands.

The validated reference is `tools/nnue/TRAINING.md` (§2 convert, §3 build bullet,
§4 train, §6 CENN pack) + `net-0` (the first net; `ledger.py show nets`). This
runbook is the checklist over it.

## Automation — the one-command path (`train_run.py`)

**Prefer this over running the steps by hand.** The whole flow below (corpus →
train → verify → pack → sanity → SPRT → ledger) is codified in
`tools/nnue/train_run.py`. A run is now: **write a config, run one command** — no
subagent, no hand-orchestration.

```bash
# 1. copy an example config and edit the params for your experiment:
cp tools/nnue/configs/example-classic.json tools/nnue/configs/net-9.json   # or example-d40-c1.json
#    (edit net_id, corpus, train.env knobs, sprt.baseline_net, notes …)

# 2. dry-run FIRST — prints the resolved plan + every command, runs nothing:
uv run python tools/nnue/train_run.py --config tools/nnue/configs/net-9.json --dry-run

# 3. run it end-to-end (add --commit to also make a LOCAL ledger commit):
uv run python tools/nnue/train_run.py --config tools/nnue/configs/net-9.json --commit
```

Flags: `--only a,b` / `--skip a,b` (resume a partial run without redoing the slow
corpus/train/sprt stages), `--set k.v=x` (one-off override, e.g.
`--set sprt.nodes=25000`). Every run writes a reproducibility manifest to the
gitignored `tools/nnue/runs/<net_id>/run.json` (all resolved params + per-stage
status/timing/results); the git-tracked provenance is the ledger row.

**Config schema** (defaults in `train_run.py:DEFAULTS`; two examples in
`tools/nnue/configs/`): `net_id` (required, must be free in `nets.jsonl`), `arch`,
`provenance`, `parent`, `notes`; `corpus.{id, raw_dir|train_bin}`;
`train.{bullet_dir, example, env{…}}`; `verify_out_clip` (true for the classic
i16-out net; **false** for the D40 hybrid save, which hard-checks overflow
itself); `pack.activation`; `sprt.{enable, baseline_net, baseline_id, nodes,
rounds, concurrency}`. The `preflight` stage builds the fast binary, records the
git sha, warns on a dirty `src/`, and refuses a net_id already in the ledger — so
"is the engine in a good state for the experiment" is checked automatically.

The manual steps below are the **reference** for what each stage does and its
tool's flags — read them to understand or debug a stage, or to run a
deliberately-experimental flow the driver doesn't cover.

## Parameters (set these per experiment)
| var | example | meaning |
|---|---|---|
| `CORPUS_DIR` | `data/nnue/raw` (SF) or `data/lc0_nnue/raw` (Lc0) | text shards `FEN\|cp\|result` |
| `CORPUS_ID` | `sf-100`, `lc0-test91` | ledger `data.jsonl` id (already registered) |
| `ARCH` | `768x2x256x1` (net-0) | grow only when the experiment is about arch |
| `BASELINE_NET` | `net-0` (`data/nnue/nets/net-0.cenn`) | the net to SPRT against |
| `NET_ID` | `net-1`, `net-2`, … | new net id (must be unused in `nets.jsonl`) |

Preconditions (verified 2026-07-12; re-confirm cheaply): RTX 4060 Ti + CUDA 12.0
(`CUDA_PATH=/usr`); bullet + bulletformat clones at
`/tmp/claude-0/-root-projects-learnrust-chessengine/c0a1b520-8665-4b2f-85c5-35473d97415c/scratchpad/{bullet,bulletformat}`;
`uv` for all Python. **RUN EVERYTHING FROM THE REPO ROOT** — `uv run` resolves the
repo's `.venv`; `cd`-ing into the bullet dir picks the wrong Python env and
SIGPIPEs the trainer (the one gotcha that bit step 3).

## Steps

1. **Corpus → training bin (+ shuffle/interleave).**
   `bash tools/nnue/build_corpus.sh` over `CORPUS_DIR` → `data/nnue/<tag>/train.bin`
   + `val.bin` + `meta.json`. Use the real `bullet-utils` shuffle+interleave (NOT
   the cp/cat fallback) so the loader doesn't see shard-ordered data. (See
   `build_corpus.sh` header + TRAINING.md §2 for the env vars.) Use a per-corpus
   output tag so SF and Lc0 bins don't collide.

2. **Train (bullet, GPU).** Copy `tools/nnue/train_net0.rs` into the bullet
   checkout as an example and build+run it (TRAINING.md §3–4 has the exact
   `cargo build --features cuda --example …` + env-var invocation;
   `train_net0.rs`'s header lists the knobs: HIDDEN/SUPERBATCHES/LR/WDL/DATA/OUT).
   **⚠ MANDATORY: the output-layer weight clip.** Training MUST clamp the L1/out
   weights so quantized `|out_weight| ≤ 127` (float `|w1| ≤ 1.98`) — the AVX2
   `mullo` SCReLU trick requires it and **our loader hard-rejects any net that
   violates it** (`NnueError::OutWeightOutOfClip`). bullet's default ±1.98 AdamW
   clip already does this (net-0 landed at max=127); if you change the optimizer/
   arch, re-confirm the clip holds. Defaults that worked for net-0: 30 superbatches,
   LR 0.001 cosine, batch 16384, wdl 0.3 (~1 min on this GPU).

3. **Verify the clip BEFORE packing.** Check `max|out_weight| ≤ 127` in bullet's
   `quantised.bin` (the last 512 i16 of the l1w block + the l1b). If it exceeds,
   tighten the clip and retrain — do NOT hand the packer a net the loader rejects.

4. **Pack CENN.** `uv run python tools/nnue/netpack.py …` wraps
   `quantised.bin` → `data/nnue/nets/<NET_ID>.cenn` (magic/version/dims/checksum;
   **sign-extends the i16 l1b → i32 out_bias**). (`netpack.py --help`.)

5. **Load-sanity.** `./target/release/chessengine nnue-eval data/nnue/nets/<NET_ID>.cenn "<fen>"`
   on: startpos (small, and equal for w/b = symmetry), a clearly White-up-material
   position (>0), a Black-up one (<0). Loader must ACCEPT it (no `NnueError`) and
   all cp finite. If it's rejected → the clip failed (step 3).

6. **SPRT vs the baseline (FIXED-NODE proxy — contention-immune).** Both sides the
   same fast binary (`RUSTFLAGS="-C target-cpu=x86-64-v3" cargo build --release`),
   toggled by `option.EvalFile`. Use **fixed nodes** (e.g. `nodes=25000` via
   `tc=inf` + `nodestime`/`go nodes`, or fastchess `-each tc=... nodes=...`) so
   concurrent jobs don't skew results; openings `tests/data/openings/openings-8ply.epd`
   (the committed book), ~1000–4000 games or until the SPRT bound is hit. **Note:**
   `tools/sprt.sh`'s `-engine` blocks are hardcoded (cmd/name only) and can't inject
   a per-engine `EvalFile`, which a same-binary net-vs-net A/B needs — use a direct
   `fastchess` call with `-engine … option.EvalFile=<net>` on each side (mirror
   sprt.sh's other settings). (TODO: add an EvalFile-toggle path to sprt.sh.) A new net should be
   ≥ ~0 vs net-0 to be worth keeping (small regressions kept for later SPSA per
   D32; large regression = a training/packing bug → investigate).

7. **Register in the ledger (ALWAYS — D38).**
   ```
   python3 tools/ledger/ledger.py add-net --id <NET_ID> --file data/nnue/nets/<NET_ID>.cenn \
     --arch <ARCH> --train-data <CORPUS_ID> --parent <BASELINE_NET> --provenance <sf-distill|lc0-distill|self-play> \
     --bullet-commit cebc78a --hparams '{"superbatches":N,"lr":"…","wdl":0.3}' --final-loss <loss> \
     --recipe "tools/nnue/RUNBOOK.md; train_net0.rs + netpack.py" --notes "…"
   python3 tools/ledger/ledger.py add-exp --kind sprt --purpose "<NET_ID> vs <BASELINE_NET>" \
     --a '{"net":"<NET_ID>"}' --b '{"net":"<BASELINE_NET>"}' --tc "<fixed-node>" --games <n> \
     --wdl '[w,d,l]' --elo <e> --err <err> --notes "…"
   python3 tools/ledger/ledger.py verify   # must be 0 problems
   ```
   Also tail the training stdout through the dashboard shim
   (`tools/nnue/dashboard.py` → `tools/nnue/runs/<NET_ID>/`) so TensorBoard shows
   the curve, and add the net to `tools/nnue/net_registry.csv`.

## Do NOT
- Commit net / corpus / match binaries (all gitignored under `data/`, `engines/`).
- `cd` into the bullet dir before `uv run` (env gotcha, step 3).
- Touch `src/` (a training run is data + GPU + tooling only).
- Skip the ledger registration — an unregistered net has no provenance.

## Token discipline (why this file exists)
The flow above is fixed. Read exactly three files for tool-specific flags —
`build_corpus.sh`, `train_net0.rs` (header), `netpack.py --help` — plus
`TRAINING.md` §3–4 for the bullet build/train command. Don't re-derive the
pipeline; don't explore. Deviate only for a deliberately-experimental run, and
say so in the net's `--notes`.
