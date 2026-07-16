# 18 — Experimental ledger

Status: Living (D38, 2026-07-12)

The system that tracks the **evolution** of the engine — nets, corpora, code
versions, and their measured results — as a content-addressed provenance DAG
plus an append-only results ledger. It answers, reproducibly: *which net came
from what data trained on which code, and how strong was it?* Git tracks the
code; this tracks the artifacts and outcomes git can't.

---

## 1. Purpose and scope

**This doc owns:**
- The `ledger/` registries (`nets.jsonl`, `data.jsonl`, `experiments.jsonl`,
  `milestones.md`) — their schemas and the append-only discipline.
- The `tools/ledger/` CLI contract (stdlib-only Python; `add-net`/`add-data`/
  `add-exp`/`verify`/`show`/`reproduce`).
- The two keystone invariants: **content-addressing** (identity = sha256) and
  the **regeneration recipe** (git SHA + input hashes + command).
- The storage split: metadata (registries) in git; artifact bytes in a
  content-addressed store (local now; GCS mirror later, D36).

**This doc does NOT own** (it references the contract with):
- Datagen provenance internals — `design/15` / `design/16` (D29). The data
  registry row is the *index* into a corpus; the corpus's own `meta.json`
  remains the detailed provenance record.
- The net file format (CENN) — `design/11` §3.4.5.
- Strength-measurement methodology (SPRT/gauntlet/anchoring) — `design/14`.
  Experiment rows are the structured *record* of runs; `design/14` defines how
  the runs are conducted. `docs/strength-log.md` stays the human narrative.
- Distributed test execution + web UI — OpenBench (D36), the eventual execution
  layer this ledger feeds into (a net's `id`/`hash` is what OpenBench references).

---

## 2. Model — a provenance DAG + a results ledger

Artifacts depend on each other in a loop (the D28 bootstrap):

```
code(git SHA) ─┬─► net(hash) ─► self-play data(hash) ─► net'(hash) ─► …
data(hash) ────┘        └─► experiment(SPRT) ─► Elo ± err
```

The DAG makes lineage provable — essential for the D28 goal of a **shipped net
with clean self-play provenance** ("N generations removed from any teacher").
The ledger makes strength history queryable and every version reproducible.

## 3. Entities and schema (JSONL rows)

- **net** (`nets.jsonl`): `id`, `hash` (sha256 of `.cenn`), `size`, `arch`,
  `act`, `quant`, `train_data` (corpus id), `parent_net`, `provenance`
  (`sf-distill`|`lc0-distill`|`self-play`|`mixed`), `bullet_commit`, `hparams`,
  `final_loss`, `elo_vs_parent`, `nps` (single-thread nps of this net's eval
  path — the **D39** inference-speed axis, kept beside net quality so any
  real-TC result decomposes into net-quality + speed Elo), `code_sha`, `recipe`,
  `file`, `created`, `notes`.
- **data** (`data.jsonl`): `id`, `hash` (manifest sha256 over sorted
  `name:filehash`), `size`, `source`, `positions`, `format`, `gen_recipe`,
  `generator_net` (self-play), `path`, `glob`, `created`, `notes`.
- **experiment** (`experiments.jsonl`): `id` (`exp-NNNN`), `date`, `kind`
  (`sprt`|`gauntlet`|`tactics`|`spsa`), `purpose`, `A`/`B` (`{code_sha, net,
  opts}`), `tc`, `book`, `games`, `wdl`, `elo`, `err`, `llr`, `hw`, `cmd`, `notes`.
- **milestone** (`milestones.md`): curated table — the strong points worth
  running back to (git SHA + net + anchored strength + reproduce pointer).

SPSA tuning runs are experiments too (`kind:"spsa"`): a tuned param-set is a
version like a net is (D36).

## 4. Keystone invariants

1. **Identity is the hash.** Friendly `id`s are labels; the sha256 is truth —
   detects silent corruption, makes artifact identity unfalsifiable, and is the
   key a future GCE-spot worker fetches by (D36).
2. **Every artifact carries its regeneration recipe** (git SHA + input hashes +
   command). Extends the existing recipe discipline: the `Bench:` trailer (D24,
   code), corpus `meta.json` (D29, data). Reproducibility = recipe + inputs.
3. **Append-only.** History is added, never rewritten — negative results and
   *why* a net was rejected stay (as valuable as the kept ones).
4. **Metadata in git, bytes in a content store.** The small diffable JSONL is
   the source of truth; nets/corpora live under gitignored `data/…` (local now;
   a hash-keyed GCS mirror later, D36). `verify` recomputes hashes to catch
   drift/corruption and checks referential integrity.

## 5. CLI (`tools/ledger/ledger.py`, stdlib-only)

`add-net` · `add-data` · `add-exp` (append a row, hashing the artifact) ·
`verify` (hashes vs registry + dangling-reference check) · `show
{nets|data|exps|dag}` · `reproduce <id>` (print the git-checkout + build + regen
recipe). Zero deps (runs under `python3` or `uv run`), consistent with the
zero-runtime-dep ethos (tooling layer, not the engine crate — D15).

`verify` semantics (2026-07-16 hardening): a row whose artifact bytes are
absent locally — file gone, or a shard dir whose glob matches no files (e.g.
sources of a mix corpus deleted) — reports `MISSING` and does **not** count as
a problem (bytes are allowed to live only on the future mirror, D36) and must
never abort the run; every row is always checked and the
referential-integrity pass always runs. `DRIFT` (content hash mismatch) and
`BADSIZE` (hash matches but the row's recorded `size` is wrong) are problems
and fail the run. `add-*` still refuses to register an empty shard dir.

## 6. Phasing (deliberately minimal; grow toward OpenBench)

The value is *discipline*, not tooling — the compute lever (D36) is where the
hours go, not ledger infra (Integral lesson, D37).
- **Now (built):** hash+recipe discipline, the three JSONL registries +
  `milestones.md`, the CLI, backfilled history (net-0, sf-100, lc0-test91,
  P9/P10/P11 + anchored-Elo experiments).
- **Later:** a hash-keyed **GCS mirror** of the byte store (durable + fetchable
  by the D36 GCE-spot workers).
- **At the tuning grind:** stand up **OpenBench** as the execution + web-UI
  layer, referencing net hashes from here.
- **Only if the JSONL gets unwieldy:** a small local browse view (or SQLite
  behind it). Not before the pain is felt.

## 7. Provenance / licensing note

External-teacher corpora (SF-distill, Lc0) are `provenance` other than
`self-play` and are D28 *scaffold* — a net trained on them is not
self-play-clean. The ledger records provenance precisely so the eventual
clean-provenance claim is auditable. Lc0/SF/Reckless artifacts stay gitignored
and are never redistributed (D27/D35); check upstream `LICENSE` before any
redistribution — we only consume.

## 8. Acceptance criteria

- [x] `ledger/{nets,data,experiments}.jsonl` + `milestones.md` exist, git-tracked.
- [x] Every registered artifact has a sha256 identity + a regeneration recipe.
- [x] `verify` recomputes hashes and checks referential integrity (0 problems on
  the backfilled set).
- [x] net-0, sf-100, lc0-test91, and the P9/P10/P11 + anchored-Elo experiments
  are backfilled; `show dag` renders the net→data lineage.
- [x] net-0-vs-PSQT (P13 step-4) recorded (`exp-0004`, +696 ± 139 Elo) +
  `milestones.md` strength cell filled.
- [ ] GCS content-addressed mirror (deferred, D36).
- [ ] OpenBench integration (deferred, D36).

## 9. Open questions

- **Backfill fidelity** of pre-ledger experiments: dates/SHAs are best-effort
  from `docs/strength-log.md`; treat pre-2026-07-12 rows as approximate.
- **`net_registry.csv`** (the dashboard's simple view, `tools/nnue/`) overlaps
  `nets.jsonl`; `nets.jsonl` is canonical — fold or auto-derive the CSV later.
- **Corpus hashing cost**: manifest-hashing multi-GB corpora is a few seconds
  today; if it grows, cache per-shard hashes.
