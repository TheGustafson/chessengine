# Memory index

- [Implementation runbook](implementation-runbook.md) — HOW TO RESUME the phase work after session turnover: mandate, live phase status, workflow recovery mechanics, environment facts. Read FIRST on any new session; update at every phase transition.
- [Smoke-test before long jobs](smoke-test-before-long-jobs.md) — always validate a pipeline with a tiny fast run before launching long/expensive jobs (+ pkill / orphaned-subprocess ops-safety lessons).
- [Subagent model preference](subagent-model-preference.md) — P8+ chessengine workflow agents use model 'opus' (Opus 4.8), max effort.
- [NNUE label scale lesson](nnue-label-scale-lesson.md) — NNUE training labels must be trainer-matched (`cp = scale·atanh(q)`), NOT an engine's cp-DISPLAY formula; the Lc0 converter bug behind net-1/2/3 failing + how to actually make Lc0 data work.
