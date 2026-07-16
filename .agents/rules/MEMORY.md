# Memory index

- [Implementation runbook](implementation-runbook.md) — HOW TO RESUME the phase work after session turnover: mandate, live phase status, workflow recovery mechanics, environment facts. Read FIRST on any new session; update at every phase transition.
- [Smoke-test before long jobs](smoke-test-before-long-jobs.md) — always validate a pipeline with a tiny fast run before launching long/expensive jobs (+ pkill / orphaned-subprocess ops-safety lessons).
- [Subagent model preference](subagent-model-preference.md) — P8+ chessengine workflow agents use model 'opus' (Opus 4.8), max effort.
- [NNUE label scale lesson](nnue-label-scale-lesson.md) — NNUE training labels must be trainer-matched (`cp = scale·atanh(q)`), NOT an engine's cp-DISPLAY formula; the Lc0 converter bug behind net-1/2/3 failing + how to actually make Lc0 data work.
- [Elo decomposition testing protocol](elo-decomposition-testing-protocol.md) — D39: Elo ≈ f(net quality, search quality, inference speed); fixed-node isolates net/search, real-TC is a ship gate, log nps, tune on the shipped net, verify external suggestions before adopting.
- [Policy integration status](policy-integration-status.md) — learned bilinear move-prior wired into MovePicker behind `policy` feature (commits 182bd08/8ed9208); first node-count signal weak-positive with 2M net; pending deployable 15M net → sweep → speed-opt → SPRT.
- [Plain language rule](plain-language-rule.md) — state fact + why it matters; no flowery phrasing; user feedback 2026-07-15.
