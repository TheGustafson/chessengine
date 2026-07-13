# Experimental ledger (D38, design/18)

A content-addressed **provenance DAG + results ledger** for the engine's
evolution: which net came from what data trained on which code, and how strong
it was. Git tracks the code; this tracks the *artifacts and their measured
results*, so any past version is reproducible and any net's lineage is provable
(the D28 clean-provenance goal).

## Files (all append-only, git-tracked)

| File | One row per | Identity |
|---|---|---|
| `nets.jsonl` | trained NNUE net | `sha256` of the `.cenn` |
| `data.jsonl` | training corpus | manifest `sha256` (sorted `name:filehash`) |
| `experiments.jsonl` | match / SPRT / gauntlet / SPSA run | `exp-NNNN` |
| `milestones.md` | curated reproducible strong point | git SHA + net hash |

**Identity is the hash, not the friendly `id`.** Names are labels; the sha256
is truth (detects silent corruption, makes "is this the net I think it is?"
unfalsifiable). Every artifact also records a **regeneration recipe** (git SHA +
input hashes + command) — reproducibility = recipe + inputs.

**Metadata here is tracked; the bytes are not.** Nets/corpora live under
gitignored `data/…` (local now; a GCS content-addressed mirror keyed by hash
comes later, D36). These small JSONL files are the source of truth for
metadata + hashes; a future GCE-spot worker fetches the exact net by hash.

## CLI — `tools/ledger/ledger.py` (stdlib only)

```bash
python3 tools/ledger/ledger.py add-net  --id net-1 --file data/nnue/nets/net-1.cenn \
        --arch 768x2x256x1 --train-data sf-100 --parent net-0 --provenance self-play …
python3 tools/ledger/ledger.py add-data --id sf-100 --path data/nnue/raw --source sf-datagen …
python3 tools/ledger/ledger.py add-exp  --kind sprt --purpose "net-1 vs net-0" --elo 34 --err 8 …
python3 tools/ledger/ledger.py verify        # recompute hashes vs registry + referential integrity
python3 tools/ledger/ledger.py show dag      # print the provenance tree
python3 tools/ledger/ledger.py reproduce net-0   # print the recipe to regenerate it
```

## Workflow

- **After training a net:** `add-net` (hashes the `.cenn`, links `--train-data`
  and `--parent`). Record the SPRT-Elo vs its parent once measured.
- **After building a corpus:** `add-data` (manifest-hashes the shards, records
  the gen recipe + `--generator-net` for self-play).
- **After a match/tune:** `add-exp` (both sides' code SHA + net + opts, TC, book,
  W/D/L, Elo±err, the exact harness command).
- **Before trusting a stored artifact:** `verify` (catches hash drift / missing
  files / dangling references).
- **Promote a strong point** to `milestones.md` by hand — the curated list you'd
  actually run back to.

The narrative history stays in `docs/strength-log.md`; this ledger is its
structured backing. Binds to OpenBench (D36) later as the execution + web-UI
layer — a net's `id`/`hash` is what OpenBench will reference.
