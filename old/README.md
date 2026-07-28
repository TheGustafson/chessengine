# `old/` — archived, not deleted

Everything here was moved out of the live tree on 2026-07-28 because nothing the project
currently depends on referenced it. It is kept (rather than deleted) because it is the
reproduction record for past campaigns, and `git mv` preserved the history.

**Nothing in here is loaded, run, or trusted by the current engine.** If you find yourself
reading a document in this folder for guidance, stop — it is almost certainly superseded,
and several of these files contain conclusions that were later refuted.

## What was moved and why

| what | count | why |
|---|---|---|
| `old/tools/` | 206 | one-shot campaign scripts (`e1_*`, `d40_*`, per-experiment gates, watchers). Not referenced by `check.sh`, `CLAUDE.md`, `design/`, `src/`, or `searchopt/` |
| `old/docs/` | 129 | per-session research write-ups from earlier campaigns |
| `old/perf-artifacts*` | 38 | raw profiling output from perf studies 1–7 |
| `old/resume-builder-ideas.md` | 1 | not engine work |

## How the split was made

A file is **live** if it is reachable, transitively, from any of: `tools/check.sh`,
`CLAUDE.md`, `TESTING.md`, `design/`, `src/`, `tests/`, `benches/`, `ledger/`, `jax/`,
`engines/canonical/`, or `searchopt/`. Everything else was moved.

Verified afterwards:
- **no live file references any moved path** (scripted check over every tracked
  `.sh/.py/.rs/.md/.toml/.json`)
- `CHECK_NNUE_GATE=1 bash tools/check.sh` → **exit 0**, `Bench: 587683` unchanged

## Where the surviving knowledge went

- **`searchopt/LEDGER.md`** — every claim from the 2026-07-28 loss forensics that survived
  its own controls, with the number that decides it, plus the refuted list.
- **`searchopt/README.md`** — the method.
- **`engines/canonical/MANIFEST.md`** — the release net/engine and their evidence.

## Restoring something

```bash
git mv old/<path> <path>
```
If you do, add it to whatever live anchor made you want it, or the next sweep will move it
back out.
