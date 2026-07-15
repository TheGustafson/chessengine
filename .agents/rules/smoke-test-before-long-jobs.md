---
name: smoke-test-before-long-jobs
description: Validate any pipeline with a tiny fast run before launching long/expensive jobs; verify end-to-end first
metadata: 
  node_type: memory
  type: feedback
  originSessionId: c0a1b520-8665-4b2f-85c5-35473d97415c
---

Before kicking off any long-running or expensive job (bulk datagen, multi-hour
compute, big SPRT, migration, fan-out), FIRST run a tiny end-to-end smoke of
the EXACT invocation on a small input, verify the outputs/format/round-trip,
and only then scale up. Catch wiring/format/config errors in seconds, not hours.

**Why:** user instruction 2026-07-11 — "Start small and test before kicking off
any long running jobs to catch things early" and "add that discipline to your
memory." I had launched a ~3-hour datagen run before smoke-testing that specific
invocation; the user (rightly) stopped it.

**How to apply:** small run (seconds–2 min) → inspect output → then the full
job. Even when a component was validated earlier, re-smoke the ACTUAL
command/params you're about to run at scale. Default to this for datagen,
training, SPRT batches, and anything with a long tail.

**Related ops-safety lessons (same session, both real footguns hit):**
- Never `pkill -f <pattern>` / `pgrep -f <pattern>` to KILL when `<pattern>`
  also appears in your own shell command — `-f` matches the invoking shell's
  cmdline and can kill the script mid-run (observed: exit 144). Kill background
  jobs by numeric PID/PGID (exclude `$$`), or match a string guaranteed absent
  from the kill command.
- Killing a Python parent that spawned engine subprocesses (python-chess
  `SimpleEngine` / Popen) with SIGKILL ORPHANS the children — the Stockfish
  engines kept running at ~80% CPU each. Kill the whole process TREE (the
  `stockfish-datagen`/engine PIDs too), or let the job exit cleanly so it
  closes its own engines. Verify with `ps --sort=-pcpu` afterward.
