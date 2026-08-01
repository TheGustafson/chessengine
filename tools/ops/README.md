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
| `archive/` | superseded one-shots, kept for provenance, **not maintained** |

## Index

### `lib/`

| script | purpose |
|---|---|
| `common.sh` | `ARCHON_ROOT` resolution, `need_mem`/`hold_for_mem`, `safe_pids`/`safe_kill`, `unit_running`, `wait_file`, `log`. **Source this first in every script.** |

### `train/`

| script | purpose |
|---|---|
| `d49_launch.sh` | Launch/resume the D49 SF-teacher run as a systemd **service**. Resumes from the newest checkpoint, holds when memory is low, retries after GPU-settle. |
| `train_guard.sh` | Serialisation guard: refuses builds while a trainer is live. |
| `ckpt_reaper.sh` | Prunes old checkpoints so the disk does not fill. |

### `match/`

| script | purpose |
|---|---|
| `screen_vs_canonical.sh` | Pack the newest checkpoint and screen it against `engines/canonical/net.cenn`. 1500 games @ 25k nodes. |
| `teacher_verdict.sh` | Screen a distilled checkpoint against **the exact checkpoint it warmed from** — isolates the teacher from ordinary training progress. |
| `main_sprt.sh` | SPRT of a built engine vs the canonical engine at a real time control. |

### `guard/`

| script | purpose |
|---|---|
| `oomguard.sh` | Watches `MemAvailable`, logs pressure, records the last OOM victim. |
| `oomshield.sh` | Pre-emptively pauses non-critical work when memory falls below a floor. |
| `d49_watchdog.sh` | Restarts the trainer if its payload dies (checks the payload, not `is-active`). |

### `build/`

| script | purpose |
|---|---|
| `pgo.sh` | Wrapper over `tools/build_pgo.sh`: memory-gated, capped, writes into the primary checkout regardless of the invoking tree. |
| `merge_gate.sh` | `CHECK_NNUE_GATE=1 tools/check.sh` under a memory cap, with the pins echoed. |

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
