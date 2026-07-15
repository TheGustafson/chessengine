---
name: subagent-model-preference
description: "From phase P8 onward, chess engine workflow subagents must use Opus 4.8 (model 'opus')"
metadata: 
  node_type: memory
  type: feedback
  originSessionId: c0a1b520-8665-4b2f-85c5-35473d97415c
---

For the chessengine project's phase workflows (P8 and later), the user wants
all subagents launched with `model: 'opus'` (Opus 4.8) at maximum effort.

**Why:** user instruction given 2026-07-10 while P7 was running ("switch to
using Opus 4.8 Max effort if you can for subagents in the future after 7").

**How to apply:** in Workflow scripts, pass `model: 'opus'` in every agent()
opts object (implementation steps AND auditors). Note: there is no per-agent
effort parameter — effort inherits the session setting; keep the session at
ultracode/xhigh. P7 and earlier phases ran on the inherited default; do not
retrofit.
