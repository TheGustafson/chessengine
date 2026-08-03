# `tools/ops/` — ARCHON operations automation

Every long-running thing we drive — training, matches, guards, builds — lives
here, is git-tracked, and resolves its own paths. Before this tree existed the
same 50 scripts were scattered across `~/.local/share/archon-d49/`,
`~/.cache/archon-simplify/` and `tools/`, none of them version-controlled, each
one assuming a working directory. **That assumption cost seven separate
incidents on 2026-08-01 alone** (see [Failure modes](#failure-modes-this-tree-exists-to-prevent)).

## Quick start

```bash
# every script is self-locating; run from anywhere
bash tools/ops/train/d49_launch.sh              # start/resume the D49 trainer
bash tools/ops/match/screen_vs_canonical.sh     # 1500-game screen vs canonical
systemctl --user is-active d49train             # NB: see the is-active warning
```

Scripts source `lib/common.sh`, which sets `ARCHON_ROOT`, `cd`s there, and
provides the memory / process / waiting helpers. **A script in this tree never
depends on the caller's cwd.**

## Layout

| dir | holds |
|---|---|
| `lib/` | `common.sh` — shared helpers, sourced by everything |
| `train/` | launching, resuming and guarding training runs |
| `match/` | screens, SPRTs, and verdict harnesses |
| `guard/` | memory / OOM / watchdog daemons |
| `build/` | PGO builds and merge gates |
| `data/` | the self-play → labelled-corpus pipeline, and the studies that set its parameters |
| `archive/` | superseded one-shots, kept for provenance, **not maintained** |

## Index

### `lib/`

| script | purpose |
|---|---|
| `common.sh` | `ARCHON_ROOT` resolution, `need_mem`/`hold_for_mem`, `safe_pids`/`safe_kill`, `unit_running`, `wait_file`, `log`. **Source this first in every script.** |

### `train/`

| script | purpose |
|---|---|
| `d49_launch.sh` | Launch/resume a training run as a systemd **service**. Resumes from the newest checkpoint, holds when memory is low, retries after GPU-settle, reads its finish line from the run spec. |
| `ckpt_reaper.sh` | Bounds the checkpoint directory. Prefix-agnostic, never reaps symlinks. `DRY_RUN=1` to preview. |

### `match/`

| script | purpose |
|---|---|
| `screen_vs_canonical.sh` | Pack the newest checkpoint and screen it against `engines/canonical/net.cenn`. 1500 games @ 25k fixed nodes. Reads eval stdev **before** the games. |
| `warm_point_verdict.sh` | Screen a checkpoint against **the exact checkpoint it warmed from** — isolates a change from ordinary training progress. Requires `REF_SB`. |
| `cross_arch_screen.sh` | Screen two nets needing DIFFERENT builds (e.g. 768 vs 1280). A `.cenn` records `acc_width` and the loader refuses a mismatch, so one binary cannot drive both. Verifies each binary loads its net **before** spending games — fastchess reports 0 games and exits 0 otherwise. |
| `sprt.sh` | SPRT a built engine vs a reference at a **real time control** — the only instrument that can price a search change's time cost. |
| `vs_stockfish.sh` | External anchor vs a Stockfish binary. Equal-node (`NODES=`, load-immune, quality-per-node) or real clock (`TC=`, load-sensitive, who-wins). Never a merge gate. |

### `guard/`

| script | purpose |
|---|---|
| `oom_priority.sh` | Raises `oom_score_adj` on everything disposable so the trainer is the last victim. Structural matching (repo + compilers + node/npm). **Kills nothing.** Merges three earlier scripts. |
| `watchdog.sh` | Restarts a training unit when its **payload** dies — progress-gated, halts after `MAX_STALL` restarts with no advance. |

### `data/`

The self-play corpus pipeline. Four stages, each resumable, each leaving its
own stats file; `run_pipeline.sh` chains them. Design and the measured
operating point: `docs/research/selfplay-data-design-2026-08-02.md`.

**For a production run use `gen_campaign.sh`, not `run_pipeline.sh` directly.**
A single long run that dies at hour 23 loses 23 hours, and the four-stage
layout writes 0.45 GB per 1 M kept positions of which 0.03 GB is the
deliverable — at 1.03 B positions that is 464 GB against 266 GB free. The
campaign driver shards the work, reaps the intermediates, and keeps three files
per shard (0.05 GB per 1 M).

| script | purpose |
|---|---|
| `gen_campaign.sh` | **Start here for anything at scale.** Oracle-vs-oracle generation in resumable shards with disjoint book slices, per-shard memory gating, intermediate reaping, and a running `campaign.json`. Restart skips every shard that already has its artifact. |
| `merge_shards.py` | Campaign shards → one globally shuffled corpus. Two-pass external shuffle (scatter to K random buckets, shuffle each in memory), bucket count derived from byte count so it scales from a 3 MB smoke to a 30 GB campaign without an OOM knob. A `cat` would leave ~1.5 h of consecutive games from one book slice adjacent in the file. |
| `label_inline.py` | Stage 3 **without running the oracle** — uses the score it already produced in play. Valid only when the oracle played both sides; reads the manifest and refuses any other mode. Removes the labelling stage: 5.85 → 2.90 core-h per 1 M. Justified by measurement, not assertion: in-game vs clean re-label over 3,437 paired positions is 0.027 σ biased, median \|Δ\| 19 cp — inside the bar §4.3 already accepted for shallow labels. |
| `run_pipeline.sh` | Stages 1–4 for one mode → a trainer-ready corpus plus the provenance for every row. Stage boundaries are resumable. `INLINE=1` swaps stage 3 for `label_inline.py`. |
| `gen_games.sh` | Stage 1. Plays games in four modes (`self` / `sf18` / `sf16` / `oracle`) at a fixed node budget, keeps the PGN as the provenance artifact, writes a manifest with sha256 of engine, net, opponent and book. `-repeat` only in the vs-Stockfish modes (self-play with it produces bit-identical duplicate games — measured); resign adjudication is `twosided` because our printed cp is ~2.07× Stockfish's and a one-sided threshold truncates asymmetrically. `oracle` mode is the pure-SF pair this file long said it never generated, so its thresholds are halved into the SF frame — left alone they would have been a 2× looser bar than every other mode. |
| `pgn_positions.py` | Stage 2. PGN → candidate positions, sampled evenly across each game's plies so phases appear in the proportion we play them. Emits absolute game ply, the White-relative result, `plies_to_end` and `termination`. |
| `label_positions.py` | Stage 3. **The D46 filter site** — the only layer holding both the raw score and the oracle best move. Filters, then clamps, never the reverse. `--null` relabels and diffs to prove labels are reproducible; `--no-filter` is audit-only and its output is refused by stage 4. |
| `export_corpus.sh` | Stage 4. Labelled TSV → `FEN \| cp \| wdl` → 32-byte bulletformat, with the contract's §5 checks as hard aborts. WDL join hit-rate is 100% by construction here (the result rides with the position), so a miss is a bug, not an acceptable loss. |
| `census.py` | What a corpus is a sample OF: phase, ply, label bands, decisiveness, and the share of rows where the WDL column is actually doing work (neither dead-drawn nor already-decided). |
| `study_label_quality.py` | Picks the oracle node count. Eval-outcome disagreement vs nodes, on a paired position set, with a sampling null printed beside every tier. |
| `study_twins.py` | Irreducible label noise: pairs identical in the network's input space that differ only in something it cannot see. Measures the part of a label no capacity can ever fit. |
| `study_oracles.py` | Whether the choice of Stockfish actually changes the labels, with a same-binary null arm. |
| `lc0_fetch.sh` | (vetoed 2026-08-01, kept for reference) Fetch Lc0 training tars newest-first with a manifest, a per-file disk floor, and resume. |

### `build/`

| script | purpose |
|---|---|
| `pgo.sh` | Memory-gated PGO build. Takes a **bare name**, always writes to `$ARCHON_ROOT/engines/bin/`, echoes the absolute path. |
| `merge_gate.sh` | `CHECK_NNUE_GATE=1 tools/check.sh` under a cap, echoing which pins it checked. |
| `promote_net.sh` | Promote a packed net to canonical: place, repoint, **re-measure BOTH pins** (a net change moves the eval hash too), rewrite check.sh + the CLAUDE.md **current** slot, prove with the full gate, add the ledger row. Refuses rather than half-finishing. |

## Operating rules these scripts encode

1. **Long-running work is a systemd transient SERVICE**, never a `--scope`.
   A `--scope` dies with the session — that is what killed a 5880-game screen
   at game 1043.
2. **Never `pkill -f`.** It matches the full command line of the shell running
   the very command that typed the pattern, so it kills its own caller. Use
   `safe_kill`, a PID file, or `systemctl --user stop`.
3. **`systemctl is-active` does NOT mean "the work is running."** For a
   `while :; do ... done` loop service it is true forever, including after the
   payload exits. Use `unit_running UNIT PAYLOAD_PATTERN`, or wait on an
   artifact with `wait_file`.
4. **`free -m` before any build; refuse under 2500 MB.** Builds run capped:
   `-p MemoryMax=<cap> -p MemorySwapMax=0`, `CARGO_BUILD_JOBS=2`.
5. **Wait on artifacts, not on unit state.** A file on disk cannot lie about
   whether it exists.
6. **Match binaries are PGO** (`tools/build_pgo.sh <features> <out> <eval-file>`
   — the eval-file argument is REQUIRED, or the NNUE kernels go unprofiled and
   PGO is inert). SPRT pairs are both-PGO or both-not.
7. **Never hardcode a net filename.** Reference `engines/canonical/net.cenn`.

## Failure modes this tree exists to prevent

Each of these happened on 2026-08-01 and is now structurally prevented:

| failure | cost | prevented by |
|---|---|---|
| Script assumed cwd; binary written to one worktree, read from another | 7 incidents: a merge gate, a net path, `fastchess`, `sprt.sh`, the trainer launcher, a PGO binary | `ARCHON_ROOT` in `common.sh` |
| `is-active` on a loop service read as "still working" | starved a frame arm, an A/B twice, and the whole build→match chain | `unit_running`, `wait_file` |
| `pkill -f` killed its own caller | turn died with bare exit 143, twice | `safe_kill` |
| Build started next to the trainer | OOM killed the trainer mid-run | `need_mem`, capped units |
| Orchestration lived in `/tmp` (tmpfs) or an agent worktree | one `git worktree prune` from unreproducible | this tree is git-tracked |

## Adding a script

1. Put it in the right subdir; name it for what it does, not for the run that
   prompted it (`screen_vs_canonical.sh`, not `x500_watch.sh`).
2. Header comment: **what it is for**, **how to invoke it**, **what it
   produces**, and **why any non-obvious guard is there**.
3. `source "$(dirname "${BASH_SOURCE[0]}")/../lib/common.sh"` on the first
   working line.
4. Add a row to the Index above. A script not in the index does not exist.
5. One-shots that have served their purpose move to `archive/` rather than
   being deleted — several carry results that are still cited.
