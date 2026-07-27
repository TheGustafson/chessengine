# tools/spsa — the SPSA tuning toolchain (D45 local + D36-lite distributed)

## Status (2026-07-26)

Built and smoke-tested; **PARKED by user ruling** — revisit after D48W
training and the local SPSA rounds. gcloud is GET-only until the user
lifts it, so only `file://` exchanges run today.

## Pieces

| File | Role |
|---|---|
| `spsa_tune.py` | The original local tuner (D45): classic SPSA, paired fixed-node fastchess batches, `option.NAME` per side, `tunables` preflight. Single process, one box. |
| `spsa_dist.py` | Distributed coordinator: **identical math** (imports spsa_tune's schedules/estimator), but each iteration's batch fans out as work units through an exchange. Stale units reissue; duplicate results dedupe (first wins); resumes from `state/theta-latest.json`; `--stage-only` pre-stages a bucket. Runs on the local box — the brain never leaves home. |
| `worker.py` | Stateless worker: poll → claim → play → upload. Identical on the local box and spot VMs; preemption-safe; exits on `control/STOP` or 30 min idle. |
| `exchange.py` | The shared prefix: `file:///dir` (dry runs, local workers) or `gs://bucket/pfx` (spot fleets) behind put/get/list. No daemon, no DB, no locks (claims are advisory). |
| `run_spsa_dist.sh` | **The one-script launcher**: coordinator + N local workers. Same command with a `gs://` URI + `workers=0` = the cloud run. |
| `campaigns/*.json` | Param manifests (name/min/max, optional c_end/r_end/float). `s2-stageA.json` = the 12 highest-leverage params (S0 transplants + optimism + PSQT-era margins); stage B re-tunes the FULL set from stage A's vector — never freeze stage-A winners (interactions co-tune). |
| `gcp/RUNBOOK.md` | Copy-paste bucket/template/MIG commands for the spot fleet, cost math (~$10-20 per 40k-game campaign), teardown. |
| `gcp/worker_startup.sh` | Spot-VM metadata startup: bootstraps worker.py from the bucket, one worker per vCPU. |

## The engine binary

Must be a `tune`-feature build (exposes `tunables` + `option.NAME`), PGO
per the standing ruling; **x86-64-v3 for cloud fleets** (heterogeneous):

    PGO_CPU=x86-64-v3 bash tools/build_pgo.sh sota,arch-d47v,tune \
        archon-d47v-tune <net.cenn>

## Local run (works today)

    bash tools/spsa/run_spsa_dist.sh tools/spsa/campaigns/s2-stageA.json \
        file:///tmp/spsa-ex engines/bin/archon-d47v-tune /tmp/<net>.cenn 8

Watch `tools/spsa/out/<name>-<stamp>/coordinator.log`; stop early by
creating `control/STOP` on the exchange. Fixed-node games are
load-immune, so this coexists with fn match campaigns (NOT with
equal-time matches — those need the box quiet).

## Why fixed-node tuning distributes cleanly

The gradient sign comes from paired θ+/θ− games at equal node budgets —
machine speed cancels entirely, so preempted, throttled, or mismatched
spot VMs contribute noise-free samples. This is spsa_tune.py's original
design decision carried through unchanged.

## Verification discipline (unchanged)

SPSA output NEVER lands in `params.rs` without the 10+0.1 SPRT verify
gate vs the untuned baseline (screen → verify → bracket → pin), and the
tuned candidate still needs the equal-time ship gate (D39).
